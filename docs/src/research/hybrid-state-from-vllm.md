# Hybrid state ideas from vLLM

Created: 2026-11-04. Status: design; nothing here is built or measured in Tur.

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

### Design (MTP)

Key MTP snapshots by their lookahead token, as vLLM does.

- A `PrefixCache` entry at `B` gains `lookahead: Option<u32>`, the token at position `B` that the producer's head used for its entry at `B - 1`. It is set only when the producer's head has written that entry (`mtp_done >= B`), and is `None` for entries taken with MTP off.
- With MTP on, an entry at `B` is inserted after the producer's head has run over `B - 1`. For aligned and template boundaries inside a prompt, the lookahead token is the prompt's token at `B`. For prompt-end and finished-request snapshots, it is the token sampled after `B`, known once the pass that ended at `B` has sampled. The pipeline already runs `draft_head` after each pass and before the next schedule, and all work goes through one command queue, so the head's write at `B - 1` is ordered before any pass that restores the entry.
- `find` accepts an MTP entry at `B` when the request's token at `B` equals `lookahead`. When the request has no token at `B` yet, as with an exact repeat that samples from the entry's stored logits, the entry is still restored. If the sampled token then differs from `lookahead`, the request's head is marked behind, and it drafts only from prompt lookup, as it does today after a head failure.
- A restoring request starts with `mtp_done = B`, so its first `draft_head` call continues from the snapshot.
- Lookups probe one extra key per candidate length. Probes stay proportional to the number of entries.

Prompt-interior boundaries lose a hit only when two prompts share exactly `B` tokens and differ at `B`. Finished-request snapshots keep their hits in chat: the last sampled token is the end-of-turn marker, which the next turn's prompt repeats at the same position. Check this against the pinned Qwen3.5 template, including the variant that drops thinking blocks.

Output exactness doesn't depend on any of this. Verification matches the target model, so a wrong head entry can lower acceptance but can't change a token. The lookahead key keeps acceptance, and therefore speed, the same whether a prefix came from the cache or from a cold run.

### Alternative (MTP)

Store the final hidden row at `B - 1` with each snapshot (4 KB on the 0.8B model, 20 KB on Qwen3.8-27B), and have the restoring request compute its own head entry at `B - 1` once it knows its token `B`. This keeps every hit, but the entry has to go into a page that the snapshot and its producer share. Tur's copy-on-write covers only appends past a shared tail, so this would add a second kind of copy-on-write for a position behind the committed one. Build it only if the lookahead key measurably loses hits.

### Tests and measurement (MTP)

- Unit: an entry with a mismatching lookahead is skipped, and lookup falls back to a shallower entry. An exact repeat whose sample differs from `lookahead` marks the head behind.
- Integration: with `--mtp`, a restored request's tokens match a cold run's for greedy and seeded sampling, and its head acceptance matches the cold run's.
- Real run on Qwen3.5-0.8B: the six-turn conversation and tool-loop workloads used in [Prefix reuse](../architecture/prefix-reuse.md#decisions), with MTP on and reuse on against each one alone. Report the share of prompt tokens cached, time to first token, and decode tokens per second.

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

The same restructuring offers a cheaper side gain. Speculation currently requires F32 state, because verification keeps F32 intermediates between drafts while ordinary decoding rounds after every token. If the scan rounds its register state through `store_state` after each token when speculating, verification reproduces ordinary decoding's rounding exactly, and speculation can run with F16 state at half the slot memory. That change touches only the scan and can land before replay.

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

### Tests and measurement (in-pass snapshots)

- Kernel: a checkpoint written inside a pass is bit-identical to the slot written by a pass that ends at the same position, for every state format.
- Integration: a request that restores a mid-pass snapshot generates exactly the tokens of an uncached run.
- Real run: re-measure first-turn template-rewrite snapshots, and run the shared-prompt serving benchmark at the default 256-token spacing and at 64. Keep the change only if it removes a measured cost. If it only matches today's numbers, record it in [Rejected experiments](../design/rejected.md) with the patch.

## Not ported

These vLLM changes either have equivalents in Tur already or don't apply on Apple GPUs:

- Running `in_proj_ba` on a second stream beside `in_proj_qkvz` ([#64425](https://github.com/vllm-project/vllm/pull/64425)). Tur already projects `qkv | z | b | a` with one matrix.
- A fused GDN verification kernel ([#61855](https://github.com/vllm-project/vllm/pull/61855)). `delta_scan` already verifies every draft of every sequence in one dispatch per layer.
- Admitting requests that share a committed prefix in the same step ([#62841](https://github.com/vllm-project/vllm/pull/62841)). Tur's waiting requests already restore the oldest request's snapshot once it exists.
- 64-bit offsets in prefill kernels ([#63761](https://github.com/vllm-project/vllm/pull/63761)). Tur's scan and convolution kernels already index with `ulong`.
- Fusing RMSNorm with FP8 or NVFP4 activation quantization ([#64509](https://github.com/vllm-project/vllm/pull/64509)). Tur keeps activations in F32 and quantizes only weights.
- Choosing verification depth on the GPU ([#60229](https://github.com/vllm-project/vllm/pull/60229)). Tur's planner already chooses depth per request from measured costs, and vLLM gained only at depths of four or more.
