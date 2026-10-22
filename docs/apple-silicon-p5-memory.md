# P5 paged and quantized state memory

Status (2026-10-22): attention K/V and recurrent state live in persistent, device-resident
pools. Every attention layer reads its keys and values in place through page tables, so
decode no longer rebuilds any cache. New K/V are quantized as they are written. Pages and
slots are generation-tagged, reference-counted leases:

- partially filled shared tails are copied before appending (copy-on-write);
- released memory is reclaimed only when no unfenced work can reference it;
- one memory plan sizes the pools against the device working set.

K/V can be stored as F32, F16 or Q8 (two's-complement codes with one F16 scale per 32,
64, 128 or 256 dimensions). Page size and group size are validated independently. Both,
together with a layout version, are part of the state identity that checkpoints carry.

On the real Qwen3.5-0.8B checkpoint, over 8,192 held-out tokens:

- **Q8 K/V (group 32):** mean KL from F32 K/V is 9.7e-5 nats, 99.2% top-1 agreement,
  unchanged NLL.
- **F16 K/V:** indistinguishable from F32.

End to end on that model, F16 K/V is the fastest format, and it is now the default
(16-token pages). At four 8K-token contexts it decodes 1.2× faster than F32 K/V. Q8
halves its state again for capacity-bound deployments: 295 MB against 483 MB for those
four contexts. It is 8–12% slower in long prefill.

The P0 M1 is the only measured device; Qwen3.8-27B has still not been executed.

## Design overview

1. **Pools instead of per-request tensors.**
   - Each attention layer owns a key and a value tensor of `[pages × page_tokens,
     kv_heads, head_dim]` codes, plus F16 scales for Q8.
   - Each recurrent layer owns `[slots, conv_width, channels]` histories and
     `[slots, heads, key_dim, value_dim]` matrices.
   - A page id spans every attention layer and a slot id every recurrent layer, so one
     page table and one slot address the whole model.
   - Pools are allocated once, at engine construction, from a byte budget.
2. **Snapshots are page tables.**
   - A `ModelState` is a position, a page table and one recurrent slot.
   - Cloning shares every lease, so checkpoints and forks cost nothing until one of them
     appends.
   - Slot 0 is a permanent zero state shared by every fresh sequence. Allocating a request
     reserves nothing.
3. **The model writes state in place.**
   - Before a pass, the engine reserves the pages the pass opens and fresh recurrent slots
     for every sequence.
   - The model stores new keys, values and recurrent states directly into them. The model
     returns only logits.
   - At `finish`, the accepted prefix becomes the new snapshot without copying anything.
   - The P2 staging copies (`LayerUpdate`, `StagedUpdate`) and the P4 per-pass K/V
     concatenation are gone.
4. **Visible tokens are immutable.**
   - A sequence appends into its partially filled tail page in place only when no other
     snapshot holds that page. Otherwise the tail's valid prefix is first copied to a fresh
     page.
   - Recurrent state is never updated in place: each pass writes new slots, so rejecting
     drafts or a failed pass only discards them.
5. **Reclamation follows completion.**
   - Dropping the last lease retires an id. Retired ids return to the free list only
     through `StatePool::reclaim`.
   - The engine calls `reclaim` only while no batch is in flight and its quarantine is
     empty.
   - Reclaiming bumps the generation, so an address captured before reuse no longer
     resolves.
6. **Explicit memory errors.**
   - The engine checks a whole pass's page and slot demand against free capacity before
     reserving or encoding anything. A pass that does not fit fails with
     `TurError::MemoryAllocation` and changes nothing.
   - The pipeline schedules only sequences whose next chunk fits. When nothing fits, it
     stops the newest request.

## Pools, leases and state

`tur_metal::pool` holds the storage:

| Item | Contract |
| --- | --- |
| `PoolConfig { pages, recurrent_slots }` | Capacities; `for_budget` reserves two slots per concurrent sequence (its committed state and one in-flight state) and fills the rest of a byte budget with pages |
| `StatePool` | Validated layout, zeroed device tensors, page and slot allocators, occupancy (`usage`), `reclaim` |
| `Id { index, generation }` | A page or slot address; resolves only while its lease is live and its generation current |
| `AttentionPool::gather/scatter`, `RecurrentPool::read/write` | Reference reads and writes for the CPU oracle and tests; kernels address pages directly |
| `quantize` / `dequantize` | Host reference of the kernels' quantization, used by the reference path and tests |

`tur_metal::state` holds the snapshots:

| Item | Contract |
| --- | --- |
| `ModelState::new(&pool)` | Empty sequence at position 0 on the zero slot; leases nothing |
| `demand(tokens, boundaries)` | Pages (including a copy-on-write tail) and slots a pass would lease |
| `reserve(tokens, boundaries)` | Leases them; writes nothing until `Reservation::prepare` enqueues the tail copy |
| `Reservation::binding()` | Generation-tagged `SequenceState` the model resolves against the pool |
| `Reservation::commit(accepted)` | Snapshot after `accepted` tokens. Keeps the pages that cover it and the exact boundary slot. A partial accept requires `EveryToken` |
| `read_layers()` | Detached, dequantized F32 copies for inspection. Waits for the copies, so a released snapshot leaves no unfenced reader |

A reservation's pages and slots stay leased until the batch is fenced. A failed,
cancelled or rejected pass drops its reservation, and the base snapshot never changed.
Holding a checkpoint keeps its pages leased, and the pool reports that.

## Model and engine contract

```rust
pub struct PackedForward<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [usize],
    pub sequences: &'a [SequenceForward<'a>],
    pub pool: &'a StatePool,
}
pub struct SequenceForward<'a> {
    pub tokens: Range<usize>,
    pub state: &'a SequenceState,   // committed pages and slot, plus reserved ones
    pub logits: &'a [usize],
}
pub struct SequenceStep { pub logits: Tensor }

HybridEngine::new(Arc<M>, Device, PoolConfig) -> Result<HybridEngine<M>>;
memory(&mut self) -> Result<PoolUsage>;            // reclaims when idle
demand(&self, StateHandle, tokens, speculative) -> Result<Demand>;
```

A pass must write K/V for every new token and the requested recurrent boundaries into
its reserved locations, and nothing else. `SequenceState::resolve` rejects a foreign
pool and stale ids before any work.

`ModelDescription` now carries the K/V layout:

- `kv: KvLayout { page_tokens, format, version }` applies to every attention layer.
- `LayerStateKind::Attention` keeps only `kv_heads` and `head_dim`.
- Restore and fork accept only checkpoints from the same pool, which pins model identity,
  layout and device.

`ExecutionRequirements::state_update_bytes` is the exact memory the pass newly leases.
That is page-rounded, plus a copy-on-write tail and one slot per retained boundary.

## K/V formats and kernels

| Format | Bytes per K or V element | Stored as |
| --- | --- | --- |
| `f32` | 4 | F32 |
| `f16` | 2 | F16 (round to nearest even) |
| `q8_<g>` | 1 + 2/g | two's-complement codes in U8 storage, one F16 scale per `g` dimensions |

The doc's formula gives per-token K/V payload: `2 × kv_heads × head_dim × element bytes`
over attention layers only, plus scales (`ModelDescription::kv_token_bytes`). For
Qwen3.5-0.8B (6 attention layers of 2 × 256):

- F32: 24,576 bytes per token;
- F16: 12,288 bytes per token;
- Q8 group 32: 6,528 bytes per token.

A recurrent slot (18 layers) is 20.2 MB. For Qwen3.8-27B (16 attention layers of 4 × 256,
48 recurrent layers), F16 K/V is 65,536 bytes per token and Q8 is 34,816.

Q8 quantization is symmetric per group. Kernel and host reference compute it the same
way:

```text
scale = f16(max|x| · (1/127))
code  = clamp(round(x / max(scale, FLT_MIN)), −127, 127)
```

A group whose scale underflows to zero stores zero codes.

Kernels (layout version 2):

- **`kv_write_r{2,4,8}_{f32,f16,q8}`.** Quantize-on-write for keys and values of a pass
  in one dispatch. Each new token goes to position `visible − 1` of its page table.
  Values are read in place from the `k | v` projection.
- **`attention_r{2,4,8}_{f32,f16,q8}`.** Ragged causal attention reading pages in place,
  with dequantization fused into the read.
  - Each lane owns 4 consecutive dimensions (2 for 64-dim heads). Rows are therefore read
    with `float4`/`half4`/`char4` loads, and a Q8 lane needs one scale per vector.
  - Split-K and the reduction are unchanged.
- **`delta_conv` / `delta_scan`.** Read the committed slot and write the final state, or
  one state per token, into slots given by an index list. Speculative boundaries need no
  contiguous run.

Every input is validated before encoding: dtypes, extents, page entries inside the pool,
scale shapes and slot indices. A pass may not write its own committed slot.

## Correctness

Kernel tests compare against independent F64 loops on the exact stored values. Unwritten
pool slots hold NaN, either in the data or in the Q8 scales, so a stray read fails.

| Check | Bound |
| --- | --- |
| Attention over F32/F16/Q8_32/Q8_64 pools, head dims 64/128/256, with and without split-K, reversed page tables, 1000-key decode | < 1e-5 vs reference on the stored values |
| Quantize-on-write vs host quantizer, strided value source, 3 sequences | F32/F16 exact; Q8 within one step on < 0.1% of codes (division rounding at ties), poison elsewhere untouched |
| Q8 roundtrip | Every element within half a step; zero groups store zeros |
| Q8 attention vs F32 attention over 1000 keys | < 2e-2 (group 32), < 3e-2 (group 256); F16 < 2e-3 |
| Delta conv/scan into slot pools | < 1e-5; committed slot unchanged; unwritten slots keep their poison |
| Slot misuse (write the committed slot, out of range, wrong boundary count) | Rejected before encoding |

Pool and state unit tests (CPU) cover:

- in-place tail appends;
- copy-on-write of shared tails with exact snapshots for both forks;
- exact partial-accept boundaries and release of unaccepted pages;
- explicit exhaustion errors;
- reclaim restoring capacity;
- stale bindings failing after release and after reuse;
- reservation byte accounting.

Engine contract tests (CPU fixture model writing through the pool API) keep every P2
case, plus:

- a pass that exceeds the pool fails before execution and leaves every request
  unchanged; a retained checkpoint keeps its pages leased;
- 20 rounds of allocate, run, checkpoint, fork, in-flight cancel and release return
  every page and slot, with peak occupancy within capacity;
- a request whose output fails after it wrote its tail page and slot publishes nothing
  and later continues exactly like a control.

Fused model tests on Metal (small fixture, identical weights):

| Check | Bound |
| --- | --- |
| Every layer and all logits over 150 tokens, fused vs reference | < 1e-4 |
| Chunked prefill (150/65/37/16/1) vs reference whole prefill | < 1e-4 |
| Batch of 3 vs each alone: logits and committed K/V and recurrent state | < 1e-5 |
| Every speculative prefix (1–5 tokens) publishes the reference boundary | < 1e-4 |
| Page sizes 1/7/32/64/128 vs 16 | < 1e-5 |
| F16/Q8 K/V: fused vs reference on the same stored values | < 1e-3 |
| F16 / Q8_32 / Q8_64 logits vs F32 K/V | 1.5e-4 / 4.6e-3 / 4.4e-3 |
| Decode at 1,500 tokens allocates beyond decode at 40 tokens | 32 KB (split partials), vs 1.5 MB a cache rebuild would copy |
| 37 decode steps of two sequences with committed pages and slot turnover | no device allocation growth |
| Prefixes ending mid-page and on a page boundary: greedy continuation, forked suffix, restore and a partially accepted speculative pass all match fresh computation; the checkpoint never changes | exact tokens |
| Memory pressure: a request that cannot fit is rejected every step while a live request decodes 20 tokens identical to an unpressured run; releasing it admits the other; the pool never grows | exact tokens |

The pipeline tests cover three cases:

- A request that could never fit the pool is rejected at admission.
- Requests that fit only one at a time wait and complete in turn.
- When two requests both need a second page, the newer is stopped with a memory error
  and the older finishes exactly.

On the real checkpoint (`Qwen/Qwen3.5-0.8B@2fc06364`) with F32 K/V:

- the Transformers comparison is unchanged: worst layer 2.3e-6 on Metal and 2.5e-6 on
  CPU, logits ≤ 3.6e-6, 162/162 argmax;
- greedy generation matches Transformers;
- fused vs reference: worst layer 4.71e-6 and logits 3.54e-6, with 97/97 argmax and 32
  identical greedy tokens in a batch of 3. Vector dot products changed the summation
  order from P4's 3.78e-6.

## Quality of quantized K/V

`cargo bench -p tur-engine --bench kv_quality` runs teacher-forced next-token prediction
over 8,192 tokens of held-out text: this repository's design documents at commit
`7ca1ecb`, written after the model's release (SHA-256 `d4b954c2…`). It uses Q8_0 weights,
256-token chunks and 16-token pages. Every format is compared with F32 K/V on identical
weights ([artifact](baselines/2026-10-22-m1-p5-quality/kv_quality.json)).

| K/V | Mean KL (nats) | Max KL | Top-1 agreement | Mean NLL (F32 K/V: 3.6997) |
| --- | --- | --- | --- | --- |
| f16 | 7.5e-8 | 2.0e-6 | 100% | 3.6997 |
| q8_32 | 9.7e-5 | 1.9e-3 | 99.23% | 3.6997 |
| q8_64 | 1.3e-4 | 2.6e-3 | 99.28% | 3.6999 |
| q8_256 | 2.7e-4 | 7.1e-3 | 98.88% | 3.7005 |

Error does not grow with context. For Q8_32, mean KL is 9.2e-5 over positions 0–1K and
1.05e-4 over 4–8K, and top-1 agreement is 99.4% and 99.1%. On the 97-token chat prompt,
Q8_32 keeps 99.0% top-1 and the first 47 greedy tokens, where a near-tie diverges.

This is one text domain and one model. It is not the held-out language/code/tools/retrieval
suite the plan requires before release, and the 27B has not been evaluated.

## Measurements

The device is the P0 M1 (8 GPU cores, 16 GB). Every timed region ends at device
completion. Artifacts:

- [kernels](baselines/2026-10-22-m1-p5-kernels/kernels.json):
  `cargo bench -p tur-metal --bench kernels`;
- [model](baselines/2026-10-22-m1-p5-model/qwen3_5.json):
  `TUR_QWEN35_0_8B=… cargo bench -p tur-engine --bench qwen3_5`;
- [quality](baselines/2026-10-22-m1-p5-quality/kv_quality.json).

Attention per layer at Qwen3.5-0.8B shapes (8 query heads, 2 K/V heads, head dim 256),
16-token pages:

| Shape | F32 | F16 | Q8_32 | Q8_256 |
| --- | --- | --- | --- | --- |
| Decode, 2K context | 0.163 ms | 0.103 ms | 0.152 ms | 0.154 ms |
| Decode, 8K context | 0.735 ms | 0.429 ms | 0.583 ms | 0.574 ms |
| Decode, 32K context | 2.65 ms | 1.45 ms | 1.96 ms | 2.00 ms |
| Decode, 8 × 2K | 1.28 ms | 0.70 ms | 0.97 ms | 0.98 ms |
| Prefill, 2,048 tokens | 79.9 ms | 72.7 ms | 115.7 ms | 115.5 ms |

- **Vector reads.** The first Q8 kernel read one byte per lane per register and was
  slower than F32 everywhere: 0.789 ms at 8K, 149 ms for prefill. The vector-read layout
  made Q8 decode 1.35–1.43× faster and F16 decode up to 1.2× faster. F32 stayed within
  ±3%.
- **Page size.** 16, 32, 64 and 128 tokens are within 3% of each other for every format
  and shape. The page table read is not a bottleneck, so the smallest page, which wastes
  the least tail memory and copies the least on copy-on-write, stays the default.
- **Quantize-on-write.** 4–10 µs per layer for one decode token, 0.11–0.31 ms for 2,048
  prefill tokens. That is under 0.4% of the attention it feeds.
- **DeltaNet.** The slot-addressed kernels cost the same as P4: one-token scan 67 µs, and
  19.7 ms for 2,048 tokens on 0.8B shapes.

Model benchmark, real 0.8B checkpoint, fused path, 16-token pages, synthetic token ids.
Decode re-runs one token over committed context; "state" is the pool memory leased by
the decoding sequences, including one recurrent slot each.

| Weights | K/V | Prefill 2K | Decode ×1 @128 | ×8 @128 | ×1 @8K | ×4 @8K | State, 4 × 8K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q8_0 | f32 | 2.16 s | 53.3 tok/s | 180.4 | 45.0 | 95.6 | 886 MB |
| Q8_0 | f16 | 2.15 s | 53.0 | 182.9 | 47.4 | 114.7 | 483 MB |
| Q8_0 | q8_32 | 2.32 s | 53.4 | 178.2 | 45.9 | 102.9 | 295 MB |
| Q4K | f32 | 2.22 s | 84.7 | 164.9 | 63.0 | 80.6 | 886 MB |
| Q4K | f16 | 2.17 s | 85.2 | 169.6 | 71.7 | 94.2 | 483 MB |
| Q4K | q8_32 | 2.43 s | 84.8 | 170.2 | 66.4 | 87.9 | 295 MB |

- **Short contexts.** At 128 tokens the K/V format barely matters: weights dominate.
- **Comparison with P4 (Q8_0 weights, F32 K/V).** One-sequence decode went from
  50.9 tok/s to 53.3 and eight-sequence decode from 157.8 to 180.4, because nothing is
  concatenated or staged any more. 2K prefill is within 4% (2.08 s in P4).
- **Through `HybridEngine`.** One sequence decodes at 48.5 tok/s (Q8_0) and 72 tok/s
  (Q4K) for every K/V format.
- **Remaining host cost.** Fence, sampling and publication add about 2 ms per token,
  which is P7's GPU-sampling work.

## Memory planning

`MemoryPlan` (in `tur-engine`) implements the plan's formula:

```text
resident = packed weights + attention KV + recurrent state + retained checkpoints
         + activation/scratch peak + speculative state + in-flight buffers + runtime overhead
```

K/V pages, recurrent slots, retained checkpoints and speculative boundaries share one pool,
so the pool's capacity bounds their sum.

| Term | How it is measured |
| --- | --- |
| Weights | `Qwen35::weight_bytes` (packed projections, embedding, norms, gates) |
| Scratch | The fused kernels' planned peak (`planned_workspace_bytes`) for `max_batch_size` sequences each prefilling one chunk at the sizing context |
| In-flight | Logits plus packed inputs of that pass |
| Overhead | A tenth of the device's recommended working set |

Without `--state-memory-mb`, the pool is sized for `max_batch_size` contexts of 8,192
tokens, capped by what the working set leaves. An explicit budget that would exceed the
working set is rejected rather than left to swap. On CPU there is no device limit.

Admission and pressure:

- `submit_request` rejects a request whose prompt plus generation limit needs more pages
  than the whole pool.
- Each step packs, in admission order, only the sequences whose chunk fits the free pages
  and slots. The others wait.
- If nothing fits, the newest request fails with `MemoryAllocation` and its memory is
  released. Recompute-based preemption and fairness are P6 work.

## Decisions

- **Default K/V: F16, 16-token pages.** F16 matched F32 quality (KL 7.5e-8) and was the
  fastest format in every model measurement. Q8 is the capacity option (`--kv-format
  q8`): within 1e-4 nats of F32 at half of F16's memory, but slower to prefill.
  For Qwen3.8-27B on a 16 GB Mac, state rather than speed is the binding constraint, so
  Q8 is likely right there. That needs a measurement on a machine that can run it.
- **Q8 group 32.** Groups of 64 and 256 save 1.5% and 2.9% of Q8 memory and are no
  faster, while KL rises 1.3× and 2.8×.
- **Page size 16.** No measurable speed difference from 16 to 128 tokens. The smallest
  page wastes the least tail memory and copies the least on copy-on-write.
- **No recent high-precision region.** Q8 error does not grow with context in the
  evaluation, so there is nothing for such a window to correct.
- **Recurrent state stays F32 and double-buffered.** The plan keeps recurrent
  accumulation in F32. Writing new slots gives exact rollback without truncation or
  copies.

## Exit criteria

| Criterion | Evidence |
| --- | --- |
| No whole-cache reconstruction on decode | Attention reads pages in place. Decode at 1,500 tokens allocates 32 KB more than at 40, against 1.5 MB for a rebuild. No concatenation exists in the fused or engine path |
| Bounded allocation growth | Pools are allocated once. 37 committed decode steps and 20 churn rounds show no device or pool growth |
| Quantization roundtrip and attention quality | Host and kernel roundtrip tests; attention over quantized pools exact on stored values; long-text quality evaluation |
| Exact full- and partial-prefix restore | Mid-page and page-boundary prefixes: forks, restores and partial speculative acceptance reproduce fresh computation |
| Memory pressure without corruption or swap reliance | Engine and pipeline pressure tests. Live requests are unchanged by rejected work. The default budget fits the working set, and explicit budgets beyond it are refused |
| Generation-tagged handles | Stale and foreign `SequenceState`s fail to resolve; reclaim bumps generations |
| Copy-on-write tails, reclamation after completion, consistent accounting | State, contract and Metal tests; one `PoolUsage` for both pools, reported bytes equal to the plan |

## Open items

- **Recent high-precision region.** Not implemented. The quality evaluation shows no
  context-dependent Q8 error to correct. Revisit for 4-bit K/V, or if the release
  quality suite finds a regression.
- **4-bit K/V.** Not implemented. The plan evaluates it separately after 8-bit.
- **Q8 prefill attention** costs 1.45× F32's: each query re-dequantizes every key. A tiled
  (matrix-unit) prefill kernel, already open from P4, would dequantize each tile once.
- **Per-sequence DeltaNet dispatch.** It remains: the slot pools make one batched
  dispatch possible, but that change belongs to P6's packed execution work.
- **Recurrent memory.** Every in-flight request holds two recurrent slots: its committed
  state and the state being written. That is 20.2 MB each on 0.8B and 157 MB on 27B.
  This is the price of atomic rollback without copies.
- **Persistence.** Prefix caches and persisted state do not exist yet (P7).
  `KvLayout.version` and the state format are already part of the identity they must
  check.
- **Preemption.** Fairness, recompute-based preemption and multi-model budgets are P6.
- **Scope.** Only the M1 is measured, and the 27B is not executed.

## Validation

```sh
cargo test --offline --release --workspace
cargo test --offline -p tur-core -p tur-metal -p tur-engine --no-default-features
TUR_QWEN35_0_8B=<snapshot> cargo test --offline --release -p tur-engine --test qwen3_5_fused -- --ignored --test-threads 1
TUR_QWEN35_0_8B=<snapshot> TUR_QWEN35_GOLDEN=<golden> cargo test --offline --release -p tur-engine -p tur-models --test qwen3_5 -- --ignored --test-threads 1
cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench qwen3_5
TUR_QWEN35_0_8B=<snapshot> TUR_EVAL_TEXT=<text> cargo bench --offline -p tur-engine --bench kv_quality
```

`TUR_DEVICE=metal` runs the Transformers comparisons on Metal. Device tests need host GPU
access and fail, rather than skip, without it.
