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
| Q2_K projection | 1–8 / >8 rows | `qmv_blocks` to 3 rows, then `qmm_blocks` / Candle |
| Q3_K, Q5_K, Q6_K and legacy projections | 1 / 2–8 / >8 rows | Candle `mul_mv` / `qmv_blocks` to 3 rows, then `qmm_blocks` / Candle F16 tiles |
| Codebook, MXFP4, F16 and BF16 runs | 1–3 / 4–16 / >16 rows | `qmv_blocks` / `qmm_blocks` / `qmm_blocks_wide` |
| Every part of a matrix mixing formats | >8 rows | `qmm_blocks_wide`, into one output |
| Recurrent decode | one-token sequences | `delta_step`, 16 key rows per simdgroup |
| Recurrent prefill and verification | multi-token sequences | `delta_prepare` (scales only), then the transposed `delta_scan` |
| Convolution | rows aligned to float4 / otherwise | `delta_conv4` / `delta_conv` |
| Prefill attention | at least 16 threadgroups (64 for ungated heads) and 16 queries per 32-query tile, head dimension up to 256 | `attention_split_f16` for dense F16 tiles of 256-dimension heads, `attention_tiled_r*` otherwise: matrix units, pages read in place |
| Decode attention, Q8 pools | below the tiled crossover, even query heads per K/V head | `attention_h2_r*_q8`: two query heads share each K/V load |
| Decode attention, F32/F16 pools | below the tiled crossover | `attention_r*`, split over ranges of at least 64 keys |
| MoE experts | ≤ 3 / 4–16 / > 16 rows per expert | GEMV / matrix units / wide F16-staged tiles |
| Sampling | device sampling on | `penalize_f32`, `sample_f32` (see [Sampling](sampling.md)) |

One decoder, `block_weights8`, reads every GGUF block format, and a function constant specializes each pipeline to one format, so the compiler removes the rest. Those pipelines build when a model that needs them loads, so other models don't pay for them. MoE experts use the same decoder.

Norms, residual additions, rotary embeddings, activations and gates are fused into neighbouring kernels where profiling showed a gain. Attention never builds the full attention matrix. Softmax runs online and every accumulator stays F32, and Q8 keys and values are dequantized once per staged tile.

The tiled prefill kernel handles 32 queries of one head per 128-thread threadgroup. It walks the keys in blocks of 32 rows, staging K and V in 64-dimension chunks. Over F32 and F16 pools it alternates between two staging buffers, loading the next chunk into registers while it multiplies the current one, with one barrier per chunk. F16 chunks stay F16 in staging and multiply against F32 queries and weights, so nothing is rounded and the accumulators stay F32. Q8 pools stage through one buffer and dequantize each chunk in place to F32. F16 and Q8 pools apply the score scale once per key block rather than to every query fragment. It supports head dimensions 64, 128 and 256, grouped-query attention, and the sigmoid output gate.

Dense F16 tiles of 256-dimension heads, the shape of Qwen's full layers and Gemma's sliding layers, run `attention_split_f16` instead. GPU counters on the per-row kernel showed 8% compute occupancy, one threadgroup of four simdgroups per core because its threadgroup memory filled the core, with the ALUs about half idle. The split kernel walks keys in blocks of 16, which keeps each threadgroup at 14.3 KiB so two fit on a core, and one threadgroup's barriers hide behind the other's work. Each simdgroup computes the scores and online softmax of its 8 rows in registers, then stores the weights to threadgroup memory and owns a 64-dimension slice of the output for all 32 rows, so every value fragment it loads multiplies four row blocks of weights. A row rescales its output only when its running maximum grows by more than 8 (natural log), which keeps weights below e^8 and skips almost every rescale on long contexts; the final division uses the same reference, so the result is exact.

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
- Block kernels replace Candle only at 2 to 8 rows for formats Candle knows, where they run 1.3 to 1.9× faster. At one row and above eight, Candle stays 1.1 to 2.4× ahead, except that Q2_K's one-row GEMV leads on the 0.8B's shapes and raised `UD-Q2_K_XL` decode 10%. Candle can't read codebook formats, which stay on the block kernels at every shape.
- The wide block kernel follows ggml's `mul_mm`: decode into registers before the barrier, stage F16 in matrix order so simdgroups load without transposing, and store full tiles straight to the output. Against the first version, which staged results through threadgroup memory, this raised it from 0.72× of Candle's Q3_K tiles to level across codebook formats, even though the decoders changed little.
- One-row codebook and Q2_K decode gives one lane a whole 32-value chunk and folds each output row straight into its sum, as ggml's IQ kernels do, so a chunk's scale and sign fields load once and only one row's weights occupy registers. With sign flips on the float's sign bit, one-row IQ2 and IQ3 rose from 0.7× of Candle's Q3_K to level, and `UD-Q2_K_XL` single-sequence decode rose 12%. Q3_K, Q5_K and Q6_K stay on Candle at one row, where it leads the same path by up to 1.7×.
- The wide block kernel decodes K-quants 16 values at a time, reading their shared scale once as ggml's dequantizers do. Q5_K rose from 0.85× to level with Candle's tiles and Q3_K from 0.77× to about 0.9×.
- Launch shapes are M1 tuning, not constants for every Mac.
- Rejected: chunked (WY/UT) DeltaNet prefill, TensorOps GEMM and seven kernel variants (see [Rejected experiments](../design/rejected.md)).
- Tiled attention over F16 pools stages F16 and scales scores once per block. Attention at 2K tokens is 7 to 10% faster, and 8K prefill up to 8% depending on chunk size, while 2K and 4K prefill don't change beyond noise. Q8 pools gain up to 4% from the scale alone. F32 pools keep the scale on each query fragment, because moving it made them 9% slower.
- Dense F16 prefill tiles of 256-dimension heads run `attention_split_f16`. The attention kernel alone is 1.46 to 2.89× faster than the per-row kernel at all 19 tiled shapes measured on the 0.8B, and 1.53× for a 2048-token chunk after 32K. Of its three changes, keeping the 32-key layout but sharing value fragments made that chunk 1.2× faster, skipping rescales another few percent, and 16-key blocks, which double the threadgroups per core, the rest. 8-key blocks were slower (more barriers per key), and so were 64-row tiles of 8 simdgroups, with 32- or 16-key blocks: one threadgroup's barriers then stall twice the simdgroups, and its accumulators spilled 272 bytes per thread.
- Still open: the wide MoE tier (more than 16 rows per expert) trails Candle by 1.2 to 1.5×. Prefill attention over Q8 and F32 pools, key lists and 512-dimension heads keeps the per-row kernel, which the same changes would likely speed up.
- Shared attention gained Gemma's semantics. Windows, ungated output and 512-dimensional heads are checked against independent F64 scalar references across K/V formats and splits, including windows that span several splits or key blocks.
- Gemma's sliding layers use the tiled kernel from 64 threadgroups. At E2B's shape it loses below 256 queries and wins by up to 1.8× above. End-to-end E2B prefill didn't change beyond noise, because attention is a small part of it.
- Windowed attention splits over the window. Splitting the whole context left a sliding layer's 512 keys in one or two splits. Splitting the window raised E2B decode at 8K tokens about 10% with one sequence and 5% with four.
