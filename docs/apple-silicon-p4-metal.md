# P4 Metal runtime and first fused kernels

P5 update: the per-pass K/V pool described below is replaced by persistent page pools
that attention reads in place, K/V formats beyond F32, and slot-addressed DeltaNet
state; the attention kernel now reads vectors per lane. See the
[P5 report](apple-silicon-p5-memory.md).

Status (2026-10-22): `tur-metal` has a Metal kernel runtime and fused MSL kernels for
the Qwen3.5 hybrid decoder. On a Metal device the fused path is the default for the
engine, CLI and server. On the real Qwen3.5-0.8B checkpoint it matches the F32
reference within 3.8e-6, and through `HybridEngine` it reproduces the reference's
greedy tokens exactly.

It is faster than the P3 reference in every measured regime:

- **Prefill:** 3.5–4×.
- **Decode, one sequence:** 1.6–3.0×.
- **Decode, eight sequences:** 3.1–4.7×.

Attention prefill on the matrix units is now implemented; see the
[tiled attention report](apple-silicon-tiled-attention.md). Chunked parallel DeltaNet
prefill remains open. The P0 M1 is the only measured device.

## Design overview

P4 replaces the P3 reference path on Metal (Candle tensor ops plus host-side loops)
with a small custom kernel runtime and fused kernels. The reference stays as the CPU
path and as the numerical oracle the fused path is tested against. Later sections give
the details and measurements.

1. **A thin runtime on Candle, not a parallel one.**
   - Custom kernels encode through Candle's command encoder, so Candle's hazard
     barriers, queue ordering, buffer pool and the existing completion fence cover
     them unchanged. There is no second queue and no new unsafe code.
   - All pipelines compile at model load.
   - Inputs are validated on the host before encoding, so a bad input is an error,
     never an out-of-bounds GPU read or a silent CPU fallback.
   - Projection slices are read in place through row strides.
   - Launch shapes start from device capabilities, are clamped per pipeline and can be
     overridden from benchmarks.
2. **Fusion where the reference made many small passes.**
   - Residual addition fuses with the next layer's norm.
   - Q/K norm fuses with partial rotary, using host-computed frequencies that match the
     reference exactly.
   - SwiGLU and the DeltaNet output gate are single passes.
   - DeltaNet is a convolution kernel plus one scan kernel that keeps the recurrent
     state in registers and fuses normalization, decay, the delta-rule update and the
     readout. The next token's inputs load while the current one computes.
3. **Paged, ragged attention with online softmax.**
   - One dispatch serves sequences of different lengths through a page table, with
     direct grouped-query addressing.
   - Online softmax means no score matrix or mask is built.
   - Keys are split across threadgroups (split-K) only when tokens × heads cannot fill
     the GPU. The sigmoid output gate is fused into the final write.
4. **Packed weights stay packed.**
   - GGML Q8_0/Q4K blocks are held once, shared by Candle's kernels and the custom
     small-batch GEMV.
   - The GEMV reads each weight once for several decoding sequences. Its row range
     follows measured crossovers.
   - Prefill keeps Candle's tiled GEMM, which already runs near the GPU's F32 peak.
5. **Model integration.**
   - Stateless work runs once per packed batch. The stateful DeltaNet kernels run once
     per sequence and write straight into batch-wide outputs.
   - All attention layers of a pass share one page geometry, so P5 can swap in
     persistent page-native pools without changing the kernel.
   - The implementation (`Execution::Fused` or `Reference`) can be switched on loaded
     weights.
   - `workspace_bytes` gives engine admission the real kernel scratch.
6. **Evidence-driven choices.**
   - Kernels are tested against independent scalar F64 references. Pages are poisoned
     with NaN to catch out-of-range reads.
   - The model is tested fused against reference on identical weights, including on the
     real checkpoint.
   - Every kernel, threadgroup size and dispatch range was benchmarked against the path
     it replaces, and anything slower was not adopted.

## Runtime

`tur_metal::kernels` (behind the `metal` feature) holds the runtime:

- **Capability query.** Records the Apple GPU family, Metal 3/4, unified memory,
  threadgroup thread and memory limits, buffer limit and recommended working set.
- **Pipelines.** One MSL library ([`tur.metal`](../crates/tur-metal/src/kernels/tur.metal))
  is compiled per device with `MTLMathMode::Safe`, and every pipeline is built at model
  load, so no request pays compilation. Each variant's name encodes its dtype and shape
  family (`attention_r8_f32`, `qmv_q4k_b3_f32`). The runtime belongs to one device, and
  `LAYOUT_VERSION` is recorded with benchmark artifacts.
- **Launch tuning.** `Tuning` starts from capabilities and is clamped again per
  pipeline: its thread limit, and the 32-wide execution width the kernels require.
  Threadgroup memory limits shrink the DeltaNet tile. Values can be replaced from
  benchmarks (`Kernels::with_tuning`). No launch shape is fixed for every device.
- **Validation before encoding.** Every input passes:
  - dtype and device checks;
  - unit column stride with an explicit row stride, so projection slices are read in
    place;
  - checked extents against the buffer length;
  - row/column agreement;
  - page-table entries inside the pool;
  - visible keys covered by the table;
  - supported head and key sizes.

  Errors return before anything is encoded. CPU tensors are rejected rather than
  executed on the host.
- **Dispatch.** Kernels encode through Candle's command encoder for the same device.
  Candle's hazard barriers (driven by input/output marking), queue ordering, pooled
  buffer reuse and the existing [`Completion`](../crates/tur-metal/src/completion.rs)
  fence therefore cover custom kernels unchanged. No second queue or synchronization
  path exists. A device test checks that a kernel output is retained until the fence.
- **Outputs and pools.** `Output` lets several dispatches fill one pooled buffer, for
  example one row range per sequence, before it becomes a tensor. Allocation uses
  Candle's pooled allocator.
- **Scratch planning.** `ScratchPlan` tracks live and peak bytes, rounded to the pool's
  power-of-two buckets. `Qwen35::workspace_bytes` replays the fused pass's live set, so
  `HybridEngine` admission now charges real kernel scratch: projections, convolved
  channels, the per-pass K/V pool and split-K partials.
- **Packed weights.** `PackedLinear` stores dense F32 or GGML Q8_0/Q4K blocks. It
  quantizes on the host, uploads the exact blocks once, and keeps a second handle to the
  same `MTLBuffer` for the custom GEMV, while Candle's `QMatMul` still serves other row
  counts. Nothing is expanded, and a tied F32 head shares the embedding's storage.

No unsafe Rust was added. `candle-metal-kernels` and `objc2-metal` are now direct
optional dependencies of `tur-metal`. Both are already in the root
`[workspace.dependencies]` at the locked versions approved in P0, so no package or
version entered the lockfile.

## Kernels

All activations are F32, and reductions, softmax and recurrent state accumulate in F32.

| Kernel | What it fuses |
| --- | --- |
| `rms_norm`, `add_rms_norm` | Norm; residual addition with the next layer's input norm (or the final norm) |
| `qk_norm_rope` | Per-head RMSNorm, then split-half partial rotary at `position · inv_freq` |
| `swiglu` | `silu(gate) · up` over the fused gate\|up projection |
| `gated_rms` | DeltaNet output norm × weight × `silu(z)`, with `z` read in place |
| `delta_conv` | Causal depthwise convolution over history + new rows, SiLU, final or per-token histories |
| `delta_scan` | L2-normalised q/k, decay, gates, delta-rule update and query readout |
| `attention`, `attention_reduce` | Ragged queries over a page table, direct GQA addressing, online softmax, split-K with a reduction, sigmoid output gate |
| `qmv_q8_0`, `qmv_q4k` | Small-batch GEMV on GGML blocks: each weight read once for 1–8 rows |

Kernel notes:

- **`qk_norm_rope`.** It reads heads in place from the interleaved `q|gate` projection
  at any row and head stride. Inverse frequencies are computed on the host exactly as
  the reference does: a one-ulp difference from a GPU `pow` grows linearly with
  position, to about 2e-3 radians at 32K.
- **`delta_scan`.** It reads `q|k|v` and the `b`/`a` gates in place. Each simdgroup
  holds four value columns of one head in registers, with lane `l` owning key rows
  `l + 32r`. The next token's inputs load while the current one updates.
  - The `[dk, dv]` state moves through a padded threadgroup tile, so device reads and
    writes are whole contiguous rows.
  - The final state is written once. Per-token states for speculation are written in
    the same pass.
- **`attention`.** It writes the gated output directly when one split suffices.

The model dispatches stateless work once per packed batch. The DeltaNet conv and scan
dispatch once per sequence (each sequence's state is its own buffer) and write straight
into batch-wide outputs; P5/P6 state pools can remove the per-sequence dispatch.
Attention builds one per-pass K/V pool from committed pages plus new keys, padded to
whole pages, and addresses it through the block table the kernel already takes.

## Correctness

Kernel tests ([`tests/kernels.rs`](../crates/tur-metal/tests/kernels.rs)) compare
against independent scalar F64 loops on the same F32 inputs. The suite uses
non-aligned shapes throughout, and every case is under 1e-5 relative error:

- strided rows of width 1000 and 37;
- `dv = 22`, which gives partial scan column groups;
- `n = 37` GEMV rows;
- head sizes 64/128/256 and key sizes 64/128/256;
- attention with 0–999 committed keys over a reversed page table whose unused slots
  hold NaN, so any out-of-range read fails;
- with and without split-K.

Model tests ([`tests/qwen3_5_fused.rs`](../crates/tur-engine/tests/qwen3_5_fused.rs))
run a small random checkpoint with kernel-sized heads, fused against the reference on
identical weights:

| Check | Bound |
| --- | --- |
| Every layer and all logits over 150 tokens (16-token pages, 64-token reference chunks) | < 1e-4 |
| Fused prefill in chunks of 150/65/37/16/1 vs reference whole prefill | < 1e-4 |
| Batch of 3 sequences (committed 37, 16 and 0 tokens) vs each alone: logits, K/V and recurrent updates | < 1e-5 |
| Per-token speculative boundaries: logits and states | < 1e-4 |
| Q8_0/Q4K batched decode (fused GEMV) vs reference one row at a time (Candle F32 one-row GEMV) | < 1e-4 |
| Q8_0/Q4K prefill (both use Candle's tiled GEMM) | < 1e-4 |
| Packed projections stay packed | < 1/3 of F32 bytes |
| Declared workspace bounds measured device growth of a 512-token pass beyond its outputs | holds |
| 20 decode passes after warm-up allocate nothing new | exact |

On the real checkpoint (`Qwen/Qwen3.5-0.8B@2fc06364`, a fixed 97-token chat prompt,
Metal), fused vs reference:

| Format | Worst layer | Logits | Argmax |
| --- | --- | --- | --- |
| F32 | 3.78e-6 | 3.72e-6 | 97/97 |
| Q8_0 | — | 3.24e-6 | 97/97 |
| Q4K | — | 3.48e-6 | 97/97 |

Three identical requests decoded as one fused batch through `HybridEngine` each
reproduce the reference's 32 greedy tokens. P3 validated the reference against
Transformers at ≤ 3.5e-6. A Transformers environment was not available for P4, so
P4's link to Transformers runs through that reference.

Finding: Candle's tiled quantized GEMM (`kernel_mul_mm`, from llama.cpp) stages weight
tiles in `threadgroup half`. Its output differs from F32 dequantization by about 2e-4
relative per projection, and 7e-3 in logits on the small fixture. The reference's
batched decode used that path. The fused GEMV dequantizes in F32.

## Measurements

The device is the P0 M1 (8 GPU cores, 16 GB) at F32 activations. Every timed region
ends at device completion. Artifacts:

- [kernels](baselines/2026-10-22-m1-p4-kernels/kernels.json): `cargo bench -p tur-metal --bench kernels`;
- [model](baselines/2026-10-22-m1-p4-model/qwen3_5.json): `TUR_QWEN35_0_8B=… cargo bench -p tur-engine --bench qwen3_5`.

Kernel benchmarks run each custom kernel against the Candle path it replaces, at
Qwen3.5-0.8B and Qwen3.8-27B shapes.

| Kernel and shape | Tur | Existing path | Speedup |
| --- | --- | --- | --- |
| add+RMSNorm, 1 × 1024 | 7.6 µs | 19.6 µs (Candle ops) | 2.6× |
| add+RMSNorm, 2048 × 1024 | 0.62 ms | 5.45 ms | 8.8× |
| SwiGLU, 2048 × 1024 output | 0.51 ms | 8.00 ms | 15.7× |
| Attention prefill, 128 tokens (8 heads, 2 KV, dim 256) | 0.42 ms | 4.14 ms (P3 reference) | 9.8× |
| Attention prefill, 2048 tokens | 79.5 ms | 310.7 ms | 3.9× |
| Attention decode, 2K context | 0.19 ms | 45.7 ms | 243× |
| Attention decode, 8K context | 0.77 ms | 208.4 ms | 272× |
| Attention decode, 8 × 2K context | 1.35 ms | 345.8 ms | 257× |
| Q8_0 GEMV, 0.8B lm_head, 2 / 4 / 6 rows | — | Candle tiled GEMM | 1.85× / 1.72× / 1.10× |
| Q4K GEMV, 0.8B lm_head, 2 / 4 / 6 rows | — | Candle tiled GEMM | 1.95× / 1.12× / 0.79× |
| Q8_0 / Q4K GEMV, 1 row | — | Candle `mul_mv` | 0.79–1.01× (Candle kept) |

DeltaNet has no like-for-like kernel baseline, because the P3 reference loops over
Candle ops. Absolute costs per layer:

| Shape | Conv | Scan | Scan per token |
| --- | --- | --- | --- |
| 0.8B, 1 token | 7.8 µs | 72.7 µs | — |
| 0.8B, 2048 tokens | 3.46 ms | 20.7 ms | 10.1 µs |
| 27B shapes, 1 token | 10.3 µs | 201.5 µs | — |
| 27B shapes, 2048 tokens | 6.30 ms | 70.5 ms | 34.4 µs |

The reference's cost shows up in the model table below.

Model benchmark, real 0.8B checkpoint, fused vs reference on identical weights.
Decode steps run one token per sequence over a 128-token context.

| Format | Prefill 128 | Prefill 2048 | Decode ×1 | Decode ×2 | Decode ×4 | Decode ×8 |
| --- | --- | --- | --- | --- | --- | --- |
| F32 fused | 137 ms | 2.19 s | 16.4 tok/s | 26.7 | 47.8 | 88.1 |
| F32 reference | 529 ms | 8.54 s | 10.5 tok/s | 15.6 | 23.1 | 28.8 |
| Q8_0 fused | 109 ms | 2.08 s | 50.9 tok/s | 86.8 | 120.9 | 157.8 |
| Q8_0 reference | 495 ms | 7.85 s | 22.6 tok/s | 23.9 | 29.5 | 33.9 |
| Q4K fused | 119 ms | 2.29 s | 78.1 tok/s | 87.1 | 103.1 | 148.6 |
| Q4K reference | 506 ms | 8.42 s | 25.8 tok/s | 23.0 | 28.9 | 34.0 |

These are model forward passes on synthetic token ids, not serving throughput. P0's
HTTP numbers were for a different model (Qwen3-0.6B), so they are not comparable.

Profile notes:

- **Decode is bandwidth-bound.** One Q8_0 decode step takes 19.6 ms. Streaming the
  799 MB of packed projections at the GEMV's ~55 GB/s accounts for ~14.5 ms of it.
- **Vocabulary projection (kernel order 5).** The 248K × 1024 head takes 23% of a
  one-sequence Q8_0 step (4.6 ms) and 19% for Q4K (2.5 ms). It already runs at
  bandwidth, so a faster kernel has little to gain. Cutting this cost needs fewer
  bytes (precision or vocabulary choices) or more rows per pass (batching,
  speculation), not a new kernel. No vocabulary-specific kernel was written.
- **Engine overhead.** Fence, sampling and state publish through `HybridEngine` add
  about 2.3 ms per token for one sequence (Q8_0: 21.9 ms engine vs 19.6 ms forward).
  GPU sampling is P7 work.
- **Prefill at 2K tokens (Q8_0, 2.08 s).**
  - attention: 6 layers × 79.5 ms ≈ 23%;
  - DeltaNet scan: 18 × 20.7 ms ≈ 18%;
  - convolution: ≈ 3%;
  - the remainder is mostly Candle's tiled projection GEMMs, about 2.25 TFLOP.

## Dispatch decisions

- **Packed projections.**
  - Q8_0 uses the custom GEMV for 2–6 rows and Q4K for 2–4.
  - At one row Candle's `mul_mv` is equal or faster (Q8_0 0.77–1.02×, Q4K 0.72–0.87×
    for the custom kernel). From 7 (Q8_0) or 5–6 (Q4K) rows, Candle's tiled GEMM wins.
  - Prefill keeps Candle's tiled GEMM. On 2048 tokens it runs at about 2 TFLOPS, close
    to this GPU's F32 peak, so a custom tiled GEMM has little to gain here. It was not
    written.
- **Attention.** One split while tokens × heads reaches 1024 simdgroups. Otherwise keys
  split into ≥ 64-key ranges and a reduction kernel combines them.
- **DeltaNet scan threadgroups.** The sweep over 1/2/4/8 simdgroups is flat within
  ~5%, and 2 is the default.

## Relation to cited research

P4 uses established kernel techniques:

- paged, ragged attention addressing (FlashInfer is the document's design reference;
  its plan/execute split is not adopted);
- online softmax with split-K for decode (FlashAttention, flash-decoding);
- GGML block formats;
- a fused recurrent DeltaNet scan.

It does not implement:

- the DeepInfer techniques (asynchronous scheduling, confidence-scheduled
  verification, batch-invariant execution);
- KIVI or TurboQuant KV quantization;
- speculation.

The plan schedules those in P5–P8, after page-native memory, packed scheduling and
exact rollback exist.

## Open items

- **Chunked parallel DeltaNet prefill (kernel order 3).** Not implemented. Prefill uses
  the fused recurrent scan, which is exact and yields every boundary. It costs about
  10 µs per token per layer on 0.8B and about 30 µs on 27B shapes, around a sixth of
  0.8B prefill at 2K tokens. A chunked kernel (UT transform on the matrix units) is the
  next DeltaNet step and must match the scan's boundary states.
- **Attention prefill on the matrix units: completed after P5.** The
  [tiled attention path](apple-silicon-tiled-attention.md) replaces per-query SIMD
  dot products for sufficiently populated query tiles, retaining split-K for decode.
- **Per-pass K/V pool.** Attention still concatenates committed pages each pass, as P3
  did. The kernel already addresses a block table, so P5 can switch to persistent
  page-native pools without changing the kernel.
- **DeltaNet decode state traffic.** About 80 µs per layer for 0.8B, or roughly 25 GB/s
  effective on 2 MB of state read and write. It is insensitive to threadgroup shape.
- **Scope of support.** F16/BF16 activations, other head or key sizes and more devices
  are not supported or measured. Kernels reject unsupported shapes. Qwen3.8-27B has
  still not been executed: its 4-bit weights exceed this Mac's working set.

## Validation

```sh
cargo test --offline --release -p tur-metal
cargo test --offline --release -p tur-engine --test qwen3_5_fused
TUR_QWEN35_0_8B=<snapshot> cargo test --offline --release -p tur-engine --test qwen3_5_fused -- --ignored --test-threads 1
cargo bench --offline -p tur-metal --bench kernels
TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench qwen3_5
```

Device tests need host GPU access and fail, rather than skip, without it. Tests that
compare device-wide allocation run serially.
