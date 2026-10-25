# Metal kernels

Tur's GPU code lives in `tur-metal`: MSL kernels compiled at runtime, a dispatch layer
and the packed weight types. Candle remains the tensor library. Custom kernels replace
Candle operations only where they measurably win.

## Runtime

- **Same queue as Candle.** Every dispatch goes through Candle's command encoder for the
  device. Custom kernels share Candle's command ordering, hazard barriers, buffer-reuse
  rules and the completion fence, and there is no second queue or synchronization.
- **No unsafe code.** Kernels are compiled and dispatched through safe bindings. Before
  encoding, every input is validated: dtype, device, strides, extents and index ranges.
  Nothing falls back to the CPU silently.
- **Capabilities and caching.** Pipelines are compiled once and cached. Launch shapes
  (simdgroups per threadgroup, crossover points) come from a per-device tuning table.
  Compiling the whole library from cold takes about 0.85 s, and warm loads are fast.
- **Scratch planning.** Each pass's workspace is planned from its shape before
  submission and charged to the [memory plan](memory.md).

## What runs where

| Operation | Condition | Kernel |
| --- | --- | --- |
| Q8_0 projection | 1 / 2–3 / 4–16 / >16 rows | Candle `mul_mv` / `qmv_q8_0` / `qmm_q8_0` / Candle F16 tiles |
| Q4K projection | 1 / 2–16 / >16 rows | Candle `mul_mv` / `qmm_q4k` / Candle F16 tiles |
| Recurrent decode | single-token sequences | `delta_step`, 16 key rows per simdgroup |
| Recurrent prefill and verification | multi-token sequences | `delta_prepare` (scales only), then the transposed `delta_scan` |
| Convolution | aligned float4 rows / otherwise | `delta_conv4` / `delta_conv` |
| Prefill attention | ≥ 16 threadgroups and ≥ 16 queries per 32-query tile | `attention_tiled_r*`: matrix units, pages read in place |
| Decode attention, Q8 pools | below the tiled crossover, even query heads per K/V head | `attention_h2_r*_q8`: two query heads share each K/V load |
| Decode attention, F32/F16 pools | below the tiled crossover | `attention_r*`, split-K over ≥ 64-key ranges |
| MoE experts | ≤ 3 / 4–16 / > 16 rows per expert | GEMV / matrix units / wide F16-staged tiles |
| Sampling | device sampling enabled | `penalize_f32`, `sample_f32` (see [Sampling](sampling.md)) |

Norms, residual additions, rotary embeddings, activations and gates are fused into the
neighbouring kernels where profiling justified it. Attention never materializes the
attention matrix. Online softmax and every accumulator stay in F32, and Q8 keys and
values are dequantized once per staged tile.

The tiled prefill kernel handles 32 queries of one head per 128-thread threadgroup. It
sweeps keys in 32-row blocks, staging K and V in 64-dimension chunks. Over F32 and F16
pools, chunks alternate between two staging buffers and each thread reads the next
chunk into registers before multiplying the current one, with one barrier per chunk.
Q8 pools stage through one buffer, reading and dequantizing each chunk in place. Head dimensions
64, 128 and 256 are supported, with GQA and the sigmoid output gate.

Gemma uses the same packed matrix and RMS kernels. The per-query attention path also
supports ungated output, explicit score scaling, sliding-window lower bounds and
512-dimensional heads. These semantics apply to F32, F16 and Q8 pools, including
split-K reduction. Qwen retains its existing gated scale and tiled-prefill selection;
Gemma currently uses the per-query path for both prefill and decode. The MoE router can
apply learned per-expert scales after top-k normalization without changing selection.

## Decisions

Measured on an M1:

- **Retain Candle where it wins.** Candle's one-row GEMV is equal or faster at one row.
  Prefill projections keep Candle's tiled GEMM, which runs at about 2 TFLOPS on 2K
  tokens, near the M1's F32 peak, so a custom tiled GEMM was not written.
- **Small-batch matrix-unit kernels.** Q8_0 at 4–16 rows and Q4K at 2–16 rows run on the
  matrix units in F32, so small batches and verification windows compute what
  single-token decoding does. Q4K at 2–8 rows is 2.4–2.6× faster than Candle's F16 tiles.
- **Coalesced recurrent decode.** The batch-8 model pass fell from 43.4 to 28.6 ms, and
  greedy serving output at 8 and 16 requests rose 1.72× and 1.66×. F32 recurrent-state
  bandwidth is now the main per-sequence decode cost.
- **Transposed DeltaNet scan.** 2.0–2.7× faster than the original token-serial scan; it
  is about 12% of 2K-token prefill.
- **The prepare pass writes scales instead of normalized copies.** Scratch fell 98%, with
  no model-level speed change.
- **Two heads per simdgroup for Q8 decode attention.** 1.06–1.25× faster attention and up
  to 1.09× model decode at 8K context. It was not adopted for F32/F16 pools, where it
  ties or loses, and four heads lose everywhere.
- **Prefetched staging for F32 and F16 pools.** One 8K-token prefill pass's attention
  fell from 408 to 379 ms per layer (F16), chunks after a long context are 3–20% faster,
  and whole 8K prefill is 2% faster; 2K prefill and short prompts are unchanged. Removing
  the K/V reads entirely would have cut it to 256 ms, so read latency is only partly
  hidden. Q8 pools keep single-buffered staging (see
  [Rejected experiments](../design/rejected.md)).
- **F32 activations.** F16/BF16 activations would be faster at 9–16 rows, but are
  deferred until held-out quality budgets exist.
- **Launch shapes are M1 tuning,** not constants for every device. The DeltaNet scan's
  simdgroup sweep was flat within 5%.
- **Rejected:** chunked (WY/UT) DeltaNet prefill, TensorOps GEMM, and seven kernel
  variants (see [Rejected experiments](../design/rejected.md)).
- **Open:** the MoE wide tier (more than 16 rows per expert) trails Candle by 1.2–1.5×;
  tiled prefill attention runs at about 28% of the matrix units' F32 peak.

- **Extend shared attention for Gemma semantics.** The window, ungated output and
  512-dimensional path are checked against independent F64 scalar references across
  K/V formats and splits. Gemma-specific launch tuning and tiled prefill remain open.
