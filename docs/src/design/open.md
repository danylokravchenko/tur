# Open work

## Models and hardware

- Run the larger models. Qwen3.8-27B and the Qwen3.6-35B MoE need Macs that fit them, and then measurements of quality, the K/V format choice (Q8 is the likely default under 16 GB), and the wide MoE tier, which trails Candle by 1.2 to 1.5× and needs real routing traces.
- An MTP head for the expert checkpoints.
- Finish Gemma 4. Run the larger checkpoints on suitable Macs and set held-out quality budgets. Assistant chains pay only with one request decoding (see [Speculative decoding](../architecture/speculation.md#decisions)); several requests need a cheaper chained step, for example gathering candidate heads in a batch without a CPU round trip per step. Calibrated Gemma artifacts (`imatrix`, GPTQ) need layer-by-layer calibration that handles PLE and shared K/V.
- More Macs. Every number so far comes from one M1. Launch shapes and crossover points need per-device sweeps, and the minimum supported macOS and chip still has to be decided.

## Audio

- Faster Whisper encoding. The large-v3-turbo encoder takes about 2.9 s per 30-second window on the M1, at about 0.8 TFLOP/s through Candle matmuls; Tur's packed kernels or F16 activations should help. Decoding a sentence takes about 0.4 s.
- Whisper's timestamp-guided seeking for clips beyond 30 s, so words at window boundaries are not cut, and the `verbose_json`, `srt` and `vtt` formats that need timestamps.
- Gemma 4 audio from prepacked artifacts, which store the text tower only, and an E4B checkpoint run.
- Encoding clips off the worker thread: today a clip is encoded when its request is submitted, between passes.

## Performance

- Long-prompt prefill. llama.cpp prefills 8K-token prompts 6 to 11% faster than Tur, while 2K-token prompts are level ([Compared with llama.cpp](../guide/performance.md#compared-with-llamacpp)). In an 8K-token prefill, tiled attention takes the largest share, about 2.3 s across six layers. Matching llama.cpp's attention speed needs faster matrix work, for example F16 staging.
- Tiled prefill attention reaches about 28% of the matrix units' F32 peak, with eight barriers per 32-key block at head dimension 256 over F32 and F16 pools, and sixteen over Q8.
- Broader comparisons. llama.cpp is measured for Q8_0 and Q4_K_M at up to 16 sequences. Still to do: MLX-LM on this Mac, 32K-token prompts, cached and uncached prefixes, and sustained load.
- Quality budgets. Held-out budgets, declared in advance, for language, code, tool calls and JSON, and long-context retrieval, with confidence intervals.
- Chained MTP drafts, estimated at about 10% with one request.
- Layer-by-layer sensitivity measurement for `--budget-bits`, which still needs the whole F32 model in memory.
- F16 or BF16 activations, once quality budgets exist.

## Serving

- Production hardening: health and readiness endpoints, request metrics, graceful shutdown, stop sequences, model revision reporting, and memory and kernel diagnostics.
- Soak tests: multi-hour cancellation and churn runs to check thermal behaviour and leaks.
- Lints: tighten `indexing_slicing` crate by crate, starting with `tur-core` and `tur-cli`.
