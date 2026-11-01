# Open work

## Models and hardware

- Check the real Qwen3.6-35B-A3B artifact against Transformers. It generates on a 36 GB M3 Pro, resident and through a 4 GiB expert cache ([Qwen3.6-35B-A3B](../guide/performance.md#qwen36-35b-a3b)), but its logits haven't been compared with a reference or with the resident run, and only an RTN Q4K artifact exists. Q8_0 and GPTQ artifacts would show how much quality the 4-bit rounding costs.
- Fit sixteen sequences of Qwen3.6-35B-A3B resident on a 36 GB Mac. llama.cpp completes that burst with all weights in memory, while Tur's resident plan fails with a Metal out-of-memory error even though the planned 27.0 GB is below the 30.2 GB working set ([Qwen3.6-35B-A3B](../guide/performance.md#qwen36-35b-a3b)). The state pool's share of that gap is fixed (Candle rounded its buffers up to powers of two); recheck this burst.
- Report the expert cache's hit rate and read time in the serving benchmark. The sixteen-request burst varies from 20.7 to 25.7 tokens/s between runs, and the JSON doesn't say whether misses explain it.
- Make pruning generalize. Search on KL now exists and overfits its calibration text, so the next step is far more varied calibration, or recovery training, judged by task benchmarks rather than KL.
- Qwen3.8-27B on a 36 GB Mac: GPTQ, a Transformers check of the RTN artifact, and a comparison with MLX are unmeasured ([Qwen3.8-27B](../guide/performance.md#qwen38-27b)).
- Run the larger models. Qwen3.8-27B runs on a 36 GB Mac but not yet on a 16 GB one. For both it and the Qwen3.6-35B MoE, quality, the K/V format choice (Q8 is the likely default under 16 GB) and the wide MoE tier, which trails Candle by 1.2 to 1.5× and needs real routing traces, are still unmeasured.
- Run the Qwen3.6-35B-A3B MTP head through the expert cache and with eight or more requests. It is measured with resident experts at one to four requests ([Qwen3.6-35B-A3B](../guide/performance.md#qwen36-35b-a3b)); whether its own expert reads pay off behind a cache, and why a second pipeline in one process ran out of GPU memory, are open.
- Finish Gemma 4. Run the larger checkpoints on suitable Macs and set held-out quality budgets. Assistant chains pay only with one request decoding (see [Speculative decoding](../architecture/speculation.md#decisions)); several requests need a cheaper chained step, for example gathering candidate heads in a batch without a CPU round trip per step. Calibrated Gemma artifacts (`imatrix`, GPTQ) need layer-by-layer calibration that handles PLE and shared K/V.
- More Macs. Most numbers come from one M1; the Qwen3.6-35B-A3B and Qwen3.8-27B sections and a short Qwen3.5-0.8B run are the only ones from an M3 Pro. Launch shapes and crossover points need per-device sweeps, and the minimum supported macOS and chip still has to be decided.

## Audio

- An E4B checkpoint run.
- A faster Whisper encoder. Attention now runs on Tur's tiled kernel, and the projections take most of the remaining 2 s per window on the M1. They run on Candle's GEMM at about 1.5 TFLOP/s, which a matrix-unit kernel did not beat (see [Rejected experiments](rejected.md)), so the next step is hardware with faster matrix units.
- Finer silence handling in Whisper. The energy gate works on whole windows, so a segment after a long silence starts at its window's opening, and text can still be invented in a quiet stretch inside a window with speech. Voice activity detection would address both.

## Performance

- Long-prompt prefill. llama.cpp prefills 8K-token prompts 6 to 11% faster than Tur, while 2K-token prompts are level ([Compared with llama.cpp](../guide/performance.md#compared-with-llamacpp)). Tiled attention is about 2.4 s of an 8K-token prefill across six layers, and the other 6 s grows linearly with tokens, so that remainder is the larger target. Chunk size isn't the lever: 256- and 512-token chunks match a single pass at 8K.
- Tiled prefill attention reaches about 30% of the matrix units' F32 peak, with eight barriers per 32-key block at head dimension 256 over F32 and F16 pools, and sixteen over Q8. Barriers aren't the limit, since cutting them to two made it slower. Removing K/V reads would save about 17%, the softmax 10% and query loads about 13%, and the multiplies themselves take most of the rest.
- Broader comparisons. llama.cpp is measured for Q8_0 and Q4_K_M at up to 16 sequences. Qwen3.6-35B-A3B is measured against llama.cpp on an M3 Pro, and against a reported oMLX figure. Still to do: MLX-LM on this Mac, 32K-token prompts, cached and uncached prefixes, and sustained load.
- Quality budgets. Held-out budgets, declared in advance, for language, code, tool calls and JSON, and long-context retrieval, with confidence intervals.
- Chained MTP drafts, estimated at about 10% with one request.
- Layer-by-layer sensitivity measurement for `--budget-bits`, which still needs the whole F32 model in memory.
- F16 or BF16 activations, once quality budgets exist.

## Serving

- Soak tests: multi-hour cancellation and churn runs to check thermal behaviour and leaks.
- Lints: tighten `indexing_slicing` crate by crate, starting with `tur-core` and `tur-cli`.
