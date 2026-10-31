# Open work

## Models and hardware

- Run a real Qwen MoE checkpoint through the expert cache.
- Make pruning generalize. Search on KL now exists and overfits its calibration text, so the next step is far more varied calibration, or recovery training, judged by task benchmarks rather than KL.
- Run the larger models. Qwen3.8-27B and the Qwen3.6-35B MoE need Macs that fit them, and then measurements of quality, the K/V format choice (Q8 is the likely default under 16 GB), and the wide MoE tier, which trails Candle by 1.2 to 1.5× and needs real routing traces.
- Run a real expert checkpoint's MTP head and measure its draft acceptance. Only a tiny Transformers fixture has run it so far.
- Finish Gemma 4. Run the larger checkpoints on suitable Macs and set held-out quality budgets. Assistant chains pay only with one request decoding (see [Speculative decoding](../architecture/speculation.md#decisions)); several requests need a cheaper chained step, for example gathering candidate heads in a batch without a CPU round trip per step. Calibrated Gemma artifacts (`imatrix`, GPTQ) need layer-by-layer calibration that handles PLE and shared K/V.
- More Macs. Every number so far comes from one M1. Launch shapes and crossover points need per-device sweeps, and the minimum supported macOS and chip still has to be decided.

## Audio

- An E4B checkpoint run.
- A faster Whisper encoder. Attention now runs on Tur's tiled kernel, and the projections take most of the remaining 2 s per window on the M1. They run on Candle's GEMM at about 1.5 TFLOP/s, which a matrix-unit kernel did not beat (see [Rejected experiments](rejected.md)), so the next step is hardware with faster matrix units.
- Finer silence handling in Whisper. The energy gate works on whole windows, so a segment after a long silence starts at its window's opening, and text can still be invented in a quiet stretch inside a window with speech. Voice activity detection would address both.

## Performance

- Long-prompt prefill. llama.cpp prefills 8K-token prompts 6 to 11% faster than Tur, while 2K-token prompts are level ([Compared with llama.cpp](../guide/performance.md#compared-with-llamacpp)). In an 8K-token prefill, tiled attention takes the largest share, about 2.3 s across six layers. Matching llama.cpp's attention speed needs faster matrix work, for example F16 staging.
- Tiled prefill attention reaches about 28% of the matrix units' F32 peak, with eight barriers per 32-key block at head dimension 256 over F32 and F16 pools, and sixteen over Q8.
- Broader comparisons. llama.cpp is measured for Q8_0 and Q4_K_M at up to 16 sequences. Still to do: MLX-LM on this Mac, 32K-token prompts, cached and uncached prefixes, and sustained load.
- Quality budgets. Held-out budgets, declared in advance, for language, code, tool calls and JSON, and long-context retrieval, with confidence intervals.
- Chained MTP drafts, estimated at about 10% with one request.
- Layer-by-layer sensitivity measurement for `--budget-bits`, which still needs the whole F32 model in memory.
- F16 or BF16 activations, once quality budgets exist.

## Serving

- Soak tests: multi-hour cancellation and churn runs to check thermal behaviour and leaks.
- Lints: tighten `indexing_slicing` crate by crate, starting with `tur-core` and `tur-cli`.
