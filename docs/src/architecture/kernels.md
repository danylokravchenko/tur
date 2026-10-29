# Metal kernels

Tur's GPU code lives in `tur-metal`: MSL kernels compiled at runtime, a dispatch layer and the packed weight types. Candle stays the tensor library, and Tur replaces a Candle operation with a custom kernel only where the custom one measurably wins.

Custom kernels pay off in three places. Fusing small steps such as norms, gates and residual additions into their neighbours saves a trip through memory for each step. Reading attention and recurrent state in the layout Tur's pools use avoids copying it into the layout a generic kernel wants. And small batches, which generic matrix kernels handle poorly, can still use the GPU's matrix units.

## Runtime

- Custom kernels go through Candle's command encoder for the device, so they share Candle's command ordering, hazard barriers, buffer reuse and completion fence. There is no second queue to synchronize.
- Kernels are compiled and dispatched through safe bindings, with no `unsafe`. Every input is checked before encoding: dtype, device, strides, extents and index ranges. Nothing falls back to the CPU silently.
- Pipelines are compiled once and cached. Launch shapes, such as simdgroups per threadgroup and the points where one kernel overtakes another, come from a per-device tuning table. Compiling the whole library from cold takes under a second.
- Each pass's scratch space is planned from its shape before submission and charged to the [memory plan](memory.md).

## What runs where

| Operation | Condition | Kernel |
| --- | --- | --- |
| Q8_0 projection | 1 / 2–3 / 4–16 / >16 rows | Candle `mul_mv` / `qmv_q8_0` / `qmm_q8_0` / Candle F16 tiles |
| Q4K projection | 1 / 2–16 / >16 rows | Candle `mul_mv` / `qmm_q4k` / Candle F16 tiles |
| Recurrent decode | one-token sequences | `delta_step`, 16 key rows per simdgroup |
| Recurrent prefill and verification | multi-token sequences | `delta_prepare` (scales only), then the transposed `delta_scan` |
| Convolution | rows aligned to float4 / otherwise | `delta_conv4` / `delta_conv` |
| Prefill attention | at least 16 threadgroups (64 for ungated heads) and 16 queries per 32-query tile, head dimension up to 256 | `attention_tiled_r*`: matrix units, pages read in place |
| Decode attention, Q8 pools | below the tiled crossover, even query heads per K/V head | `attention_h2_r*_q8`: two query heads share each K/V load |
| Decode attention, F32/F16 pools | below the tiled crossover | `attention_r*`, split over ranges of at least 64 keys |
| MoE experts | ≤ 3 / 4–16 / > 16 rows per expert | GEMV / matrix units / wide F16-staged tiles |
| Sampling | device sampling on | `penalize_f32`, `sample_f32` (see [Sampling](sampling.md)) |

Norms, residual additions, rotary embeddings, activations and gates are fused into neighbouring kernels where profiling showed a gain. Attention never builds the full attention matrix. Softmax runs online and every accumulator stays F32, and Q8 keys and values are dequantized once per staged tile.

The tiled prefill kernel handles 32 queries of one head per 128-thread threadgroup. It walks the keys in blocks of 32 rows, staging K and V in 64-dimension chunks. Over F32 and F16 pools it alternates between two staging buffers, loading the next chunk into registers while it multiplies the current one, with one barrier per chunk. Q8 pools stage through one buffer and dequantize each chunk in place. It supports head dimensions 64, 128 and 256, grouped-query attention, and the sigmoid output gate.

Gemma reuses the same matrix, norm and rotary kernels. Each K/V-producing layer projects `q | k | v` with one matrix and reads every head in place from the result, and its value norm runs on the same kernel with a zero rotation. `gate | up` is one matrix whose halves feed `gelu_mul` in place, and `residual_rms_norm` fuses each residual update with the layer scale and the next norm. The per-query attention kernel also handles ungated output, explicit score scaling, sliding windows and 512-dimensional heads, over F32, F16 and Q8 pools. It splits each query's window rather than the whole context, so a sliding layer keeps every split busy at long contexts. The tiled kernel handles windowed heads too, starting each tile's sweep at its first token's window. Gemma's 512-dimensional global heads use the per-query kernel. The MoE router can apply learned per-expert scales after top-k normalization without changing which experts are picked.

## Decisions

Measured on an M1:

- Candle stays where it wins. Its one-row GEMV is as fast or faster at one row. For prefill projections, Candle's tiled GEMM runs near the M1's F32 peak on 2K tokens, so Tur has no custom tiled GEMM.
- Small batches use the matrix units. Q8_0 at 4 to 16 rows and Q4K at 2 to 16 rows run on the matrix units in F32, so a small batch or a verification window computes exactly what single-token decoding does. Q4K at 2 to 8 rows is 2.5× faster than Candle's F16 tiles.
- Recurrent decode reads memory in coalesced order. This cut an eight-sequence model pass by a third, and serving throughput at 8 and 16 requests rose about 1.7×. Reading F32 recurrent state is now the main per-sequence cost of decoding.
- The DeltaNet prefill scan is transposed. It runs 2 to 2.7× faster than the original token-by-token scan and is about 12% of 2K-token prefill.
- The prepare pass writes scales instead of normalized copies. That cut its scratch memory by 98% with no change in model speed.
- Q8 decode attention processes two heads per simdgroup. It is up to 1.25× faster in attention and 1.09× in model decode at 8K tokens. F32 and F16 pools don't use it because it ties or loses there, and four heads lose everywhere.
- Tiled attention over F32 and F16 pools prefetches the next chunk. It makes chunks after a long context 3 to 20% faster and 8K prefill 2% faster, with short prompts unchanged. Removing the K/V reads entirely would be much faster still, so read latency is only partly hidden. Q8 pools keep single-buffered staging (see [Rejected experiments](../design/rejected.md)).
- Activations stay F32. F16 or BF16 activations would be faster at 9 to 16 rows, but wait until there are held-out quality budgets to check them against.
- Launch shapes are M1 tuning, not constants for every Mac.
- Rejected: chunked (WY/UT) DeltaNet prefill, TensorOps GEMM and seven kernel variants (see [Rejected experiments](../design/rejected.md)).
- Still open: the wide MoE tier (more than 16 rows per expert) trails Candle by 1.2 to 1.5×, and tiled prefill attention reaches about 28% of the matrix units' F32 peak.
- Shared attention gained Gemma's semantics. Windows, ungated output and 512-dimensional heads are checked against independent F64 scalar references across K/V formats and splits, including windows that span several splits or key blocks.
- Gemma's sliding layers use the tiled kernel from 64 threadgroups. At E2B's shape it loses below 256 queries and wins by up to 1.8× above. End-to-end E2B prefill didn't change beyond noise, because attention is a small part of it.
- Windowed attention splits over the window. Splitting the whole context left a sliding layer's 512 keys in one or two splits. Splitting the window raised E2B decode at 8K tokens about 10% with one sequence and 5% with four.
