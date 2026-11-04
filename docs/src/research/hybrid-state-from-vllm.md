# Hybrid state ideas from vLLM

Created: 2026-11-04. Status: parts 1 and 3 are built and measured (see [Prefix reuse](../architecture/prefix-reuse.md#snapshots-and-the-mtp-head)); part 2 is a design, not yet built or measured.

vLLM's Qwen3.5 work between September and November 2026 was mostly tuning for NVIDIA and ROCm: FP8 and NVFP4 fusions, CUDA-graph sizing, tile configurations. Three of its changes address limits that Tur also has, because they come from how a hybrid model's recurrent state behaves rather than from a GPU. This page designs Tur's version of each, in the order they should be built:

1. [MTP drafts with prefix reuse](#1-mtp-drafts-with-prefix-reuse): drop the rule that `--mtp` turns prefix reuse off.
2. [Committing accepted drafts by replay](#2-committing-accepted-drafts-by-replay): stop writing a full recurrent state after every draft token.
3. [Snapshots inside a prefill pass](#3-snapshots-inside-a-prefill-pass): stop ending prefill chunks at snapshot boundaries.

Parts 2 and 3 both change `RecurrentSpan`, so whichever is built second reuses the first one's addressing.

## 1. MTP drafts with prefix reuse

### Problem (MTP)

The MTP head at position `p` combines the embedding of token `p + 1` with the final hidden state at `p` and writes its keys and values at `p` into the head's attention layer, which lives in the request's pages like any other layer. Its entry at `p` therefore depends on the token after `p`. A snapshot at boundary `B` shares its pages with the request that produced it, and the head's entry at `B - 1` in those pages was computed from that producer's token `B`. A second request restoring the snapshot may continue with a different token, and the head would attend over a key and value it never produced. `PipelineConfig` resolves this by turning prefix reuse off whenever `--mtp` is on.

That rule costs the most where both features matter: agents and multi-turn chat, which reuse long prefixes and decode many tokens per turn.

### What vLLM did (MTP)

[vLLM #61589](https://github.com/vllm-project/vllm/pull/61589) puts the token after each block into the block's hash. A cached block matches only when its lookahead token matches too, so a hit reads exactly the draft keys and values a cold run would write. Hits then reach the same depth as without speculation, apart from prefixes that end exactly on a block boundary.

### What Tur built (MTP)

The lookahead key alone was built first and measured on Qwen3.5-0.8B over four six-turn conversations. It kept every hit inside prompts but lost every snapshot at a template-rewrite boundary or a reply's end, because a rewrite boundary is by definition where the next prompt's token differs. Cached prompt tokens fell from 8,667 without MTP to 5,376. So Tur stores the final hidden state at `B - 1` with each snapshot as well, and recomputes the head's entry when the token differs:

- The pipeline inserts a pass's snapshots in `settle`, after the draft head has run over that pass, including for requests that finished in it. The head's side pass needs pages that no snapshot shares, and inserting first would have made it fail.
- A `PrefixCache` entry carries a `HeadEntry`: the token the head read after `B`, and the hidden row at `B - 1`. A request that needs the head accepts an entry whose token matches, or whose snapshot ends inside a page and has a hidden row. The restoring request's first pass copies that tail page, so its head can rewrite position `B - 1` in a page nobody else sees. Its first head run starts at `B - 1` with the stored row prepended.
- An entry the head never reached (a constrained producer) serves only requests without the head, until a snapshot of the same tokens with head entries replaces its state.
- A Gemma 4 assistant keeps no state, so its requests restore any snapshot.

With both mechanisms, the six-turn conversations cached 8,915 of 13,765 prompt tokens with MTP on and finished in 9.9 s, against 11.2 s for reuse alone and 16.0 s for MTP alone. The head's drafts were 92% accepted. Eight users behind a shared 1.5K-token system prompt cached the same 12,288 tokens with and without MTP. Throughput there fell from 158 to 137 tokens/s, as it does without reuse, because the head stops drafting above four decoding requests but still writes its keys and values.

Output exactness never depended on this. Verification matches the target model, so a wrong head entry could only lower acceptance. An integration test checks the stronger property: after a restore that recomputes the entry, the head drafts exactly as it does after computing the whole prompt.

## 2. Committing accepted drafts by replay

### Problem (replay)

Verification reserves `Boundaries::EveryToken`, and `delta_scan` and `delta_conv` write a complete recurrent slot after every verified token. `Reservation::commit` then keeps the slot at the accepted boundary. A slot holds every DeltaNet layer's matrix and convolution history:

| Model | DeltaNet layers | Matrix state per slot | Convolution history per slot |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 18 | 18 MiB | 1.3 MiB |
| Qwen3.6-35B-A3B | 30 | 60 MiB | 2.8 MiB |
| Qwen3.8-27B | 48 | 144 MiB | 5.6 MiB |

(F32 state, 128 by 128 per value head; from the pinned configs.)

Verifying `k` drafts writes `k + 1` slots per sequence. On Qwen3.6-35B-A3B, whose decode reads only about 2 GB of active weights per token at 4 bits, three drafts write about 250 MB per sequence per pass. The memory plan also reserves `k + 1` slots per decoding request up front. Both costs help explain why the planner turns speculation off past a few requests, and why it does so earlier on larger models.

### What vLLM did (replay)

ReplaySSM and RecoverSSM ([#63045](https://github.com/vllm-project/vllm/pull/63045), [#65033](https://github.com/vllm-project/vllm/pull/65033)) verify from one state slot and store small per-token records instead of states. After sampling, a commit kernel folds the accepted tokens into the slot from those records. vLLM's first Triton commit was slower per step than storing states, and it only won after hand-written CuTe kernels that fold every layer in one launch. That history is the main risk for this part.

### Design (replay)

- Verification reserves one final slot instead of `k + 1`. The scan writes the final state as an ordinary pass does, and also writes per-token records into a records buffer: the post-convolution `q`, `k` and `v` rows the scan reads (`mixed`), the `prep` row (normalizing scales, decay, beta), and the pre-convolution input rows that the convolution history is rebuilt from. That comes to about 49 KB per DeltaNet layer per token on the 0.8B model, under 1 MB per token across the model, against 19 MB for a slot.
- If every draft is accepted, the final slot is the committed state, and nothing more runs.
- If fewer are accepted, a new `delta_commit` dispatch per DeltaNet layer reads the base slot, reruns the scan's state update over the accepted rows from the records, and writes a fresh slot. It also rebuilds the convolution history from the last `taps - 1` input rows. The final slot is released. The commit has to finish before the request's next pass reads the new slot, and since both go through the same queue, ordering them is enough.
- `Boundaries` gains a `Replay` variant. `Reservation::commit` for a partial accept returns a state whose slot the commit dispatch fills, and the engine encodes all of a pass's commits together.
- The records buffer is sized by the memory plan from `max_running`, `max_draft` and the model's shapes, and replaces the per-draft slot reservation.

Expected traffic per sequence per pass, in slot sizes `S`, with `k` drafts and acceptance rate `a` for the whole chain:

| | Today | Replay |
| --- | --- | --- |
| All accepted | `(k + 1)S` written | `S` written |
| Partial accept | `(k + 1)S` written | `S` written, then `S` read and `S` written |

With one draft, the expected cost is `2S` today against `(3 - 2a)S` with replay, which only wins when acceptance is above one half. The gain grows with chain depth and with prompt-lookup drafts, which often run four or more tokens deep. The memory saving holds at every depth.

### Exactness (replay)

The commit must store exactly the state that `EveryToken` would have stored at the accepted boundary. The dot products in the scan reduce across lanes, so the result depends on the lane layout: `delta_commit` has to share the scan's update code and its `ROWS` and column layout, not reimplement it. Factor the per-token update out of `delta_scan` into an inline function that both kernels call.

The same restructuring offers a cheaper side gain. Speculation currently requires F32 state, because verification keeps F32 intermediates between drafts while ordinary decoding rounds after every token. If the scan rounds its register state through `store_state` after each token when speculating, verification rounds at the same points as ordinary decoding, and speculation could run with F16 state at half the slot memory. That alone isn't enough, though: ordinary decoding runs `delta_step` while verification runs `delta_scan`, and the two reduce their dot products in different lane layouts. In F32 their results differ by about 1e-7, which hasn't changed a token in the speculation tests. With F16 storage, the same difference can tip a stochastic rounding and move an element by a whole F16 ulp. So this side gain first needs `delta_step` and `delta_scan` to produce bit-identical states for one token, with a kernel test that checks it.

### Tests and measurement (replay)

- Kernel: for every accepted count, state format and `ROWS` setting, the replayed slot is bit-identical to the matching `EveryToken` slot. Use the existing differential scan tests as the base.
- Integration: the existing same-seed speculation comparisons pass with replay, and with F16 state once per-token rounding is in.
- `just bench kernels`: `delta_commit` time per layer at 1 to 16 sequences and 1 to 8 accepted tokens.
- Serving on Qwen3.5-0.8B with lookup and MTP drafts at 1, 4 and 8 requests, then on Qwen3.6-35B-A3B on the M3 Pro. Report decode tokens per second, the planner's drafting rate, and state-pool memory. If replay doesn't beat per-token slots beyond noise at any request count, it gets a row in [Rejected experiments](../design/rejected.md). The memory saving alone doesn't justify a slower decode.

## 3. Snapshots inside a prefill pass

### Problem (in-pass snapshots)

A recurrent state exists only where a pass ended, so the scheduler never lets a prefill chunk cross a snapshot position (`PrefixCache::next_boundary`), and drafting caps a verification at the next boundary. Template-rewrite boundaries and prompt ends also stop chunks. Each stop can leave a short pass behind it. The template-rewrite snapshot on first turns was [rejected](../design/rejected.md) because its extra prefill pass made an uncached request 15% slower to its first token.

### What vLLM did (in-pass snapshots)

[vLLM #60050](https://github.com/vllm-project/vllm/pull/60050) has the DeltaNet prefill kernel write the state at a requested position inside the pass, so a prompt that crosses its cacheable boundary runs in one step. Time to first token fell 10 to 20% on 8K-token prompts on a GB200, and [#61601](https://github.com/vllm-project/vllm/pull/61601) did the same for Qwen3.8-Flash-Next's convolution layer.

### Expected value in Tur

Less than in vLLM. On the M1, 256- and 512-token chunks prefill an 8K-token prompt as fast as one pass ([Open work](../design/open.md#performance)), so cutting passes saves little. The concrete gains are:

- first-turn template-rewrite snapshots without a split pass, which would make the rejected boundary worth re-measuring;
- verification no longer capped at a snapshot boundary;
- chunk size and snapshot spacing chosen independently, which matters more on Macs where prefill is faster relative to fixed per-pass costs, such as the M5.

### Design (in-pass snapshots)

- `RecurrentSpan` replaces its `every` flag with a list of write offsets: the last token for an ordinary pass, every token for speculation, and checkpoint offsets plus the last token for a prefill that crosses snapshot positions. The scan and convolution kernels write a slot after token `t` only when `t` is in the span's offsets; the host passes the offsets sorted, so each kernel advances one cursor.
- With F16 or BF16 state, the scan rounds its register state through `store_state` at each checkpoint. A request restoring the snapshot then starts from the same values that the producer continued from, as it does today when the chunk ends there.
- `Boundaries` gains a checkpoint variant listing the offsets. `Reservation::commit(n)` accepts any listed offset, and the pipeline inserts a snapshot for each.
- The scheduler stops treating snapshot positions as chunk limits. `next_boundary` becomes a list of positions to checkpoint inside the chunk, and the drafting cap at boundaries goes away.
- Attention needs no change. A snapshot inside a pass sees only its own positions, and copy-on-write already covers a snapshot that ends partway through a page.

### What Tur built (in-pass snapshots)

- `RecurrentSpan` lists write ends instead of an every-token flag. The scan writes a slot when it passes an end and continues from the stored value, and the convolution writes the history there. A kernel test checks that states, outputs and the final state equal those of passes ending at the same tokens, bit for bit, in F32, F16 and BF16.
- `Boundaries::Checkpoints` lists a pass's in-pass ends, and `SequenceDescriptor::checkpoints` carries them to the engine, which returns the committed state at each with the request's commit. The pipeline inserts them as snapshots in `settle`, after the draft head has run. The head's side pass accepts pages held by those not-yet-inserted states.
- Chunks no longer stop at aligned or template-rewrite positions, and the scheduler's memory check counts each chunk's extra slots.
- The template-rewrite boundary is also taken on first turns.

On Qwen3.5-0.8B over four six-turn conversations at 256-token spacing, cached prompt tokens rose from 8,667 to 9,681 of 13,765, last-turn time to first token fell from 1.19 s to 0.62 s, and the run took 10.4 s instead of 11.2 s. Decode speed and the shared-system-prompt workload stayed within run-to-run noise. Verification passes still stop at the next aligned position, because speculative sequences keep a state per token and take no checkpoints.

## Not ported

These vLLM changes either have equivalents in Tur already or don't apply on Apple GPUs:

- Running `in_proj_ba` on a second stream beside `in_proj_qkvz` ([#64425](https://github.com/vllm-project/vllm/pull/64425)). Tur already projects `qkv | z | b | a` with one matrix.
- A fused GDN verification kernel ([#61855](https://github.com/vllm-project/vllm/pull/61855)). `delta_scan` already verifies every draft of every sequence in one dispatch per layer.
- Admitting requests that share a committed prefix in the same step ([#62841](https://github.com/vllm-project/vllm/pull/62841)). Tur's waiting requests already restore the oldest request's snapshot once it exists.
- 64-bit offsets in prefill kernels ([#63761](https://github.com/vllm-project/vllm/pull/63761)). Tur's scan and convolution kernels already index with `ulong`.
- Fusing RMSNorm with FP8 or NVFP4 activation quantization ([#64509](https://github.com/vllm-project/vllm/pull/64509)). Tur keeps activations in F32 and quantizes only weights.
- Choosing verification depth on the GPU ([#60229](https://github.com/vllm-project/vllm/pull/60229)). Tur's planner already chooses depth per request from measured costs, and vLLM gained only at depths of four or more.
