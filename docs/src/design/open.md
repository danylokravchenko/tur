# Open work

## Models and hardware

- **Run the flagship.** Execute Qwen3.8-27B and the Qwen3.6-35B MoE on Macs that fit
  them. Measure quality, the K/V format choice (Q8 is the likely default under 16 GB) and
  the MoE wide tier, which trails Candle by 1.2–1.5× and needs trained routing traces.
- **Flagship MTP head** for expert checkpoints.
- **Gemma 4 follow-through.** Execute larger checkpoints on suitable Macs; establish
  held-out quality budgets, tune prefill, and add native MTP and offline prepacking.
- **More devices.** Every number so far is from one M1. Launch shapes and crossovers
  need per-device sweeps, and the minimum supported macOS and chip still has to be
  decided.

## Performance

- **Long-prompt prefill.** llama.cpp prefills 8K-token prompts 6–11% faster than Tur on
  the M1, while 2K-token prompts are level
  ([Compared with llama.cpp](../guide/performance.md#compared-with-llamacpp)). In one
  8K-token Q8_0 prefill pass, tiled attention takes 2.28 s (6 layers × 379 ms, F16 K/V),
  the DeltaNet scan and convolution 0.95 s, and projections and the rest 0.58 ms per
  token at both 2K and 8K. Prefetched staging already hides part of the K/V read latency
  (see [Kernels](../architecture/kernels.md)); matching llama.cpp's attention, about
  217 ms per layer, needs faster matrix work, for example F16 staging (see tiled prefill
  attention below).
- **Comparative baseline.** llama.cpp is measured for Q8_0 and Q4_K_M at up to 16
  sequences. Still to do: MLX-LM, 32K-token prompts, cold and warm prefixes, and sustained
  load.
- **Quality budgets.** Predeclared held-out budgets for language, code, tool/JSON and
  long-context retrieval, with confidence intervals.
- **Chained MTP drafts.** A second draft through the head; an estimated 10% at one
  sequence.
- **Layer-at-a-time calibration,** so `tur quantize` scales to the flagship.
- **Tiled prefill attention** runs at about 28% of the matrix units' F32 peak (0.73
  TFLOPS at 8K tokens), with eight barriers per 32-key block at head dimension 256 over
  F32 and F16 pools and sixteen over Q8.
- **F16/BF16 activations**, once quality budgets exist.

## Serving

- **Production hardening.** Health and readiness endpoints, request metrics, graceful
  shutdown, stop sequences, model revision reporting, and memory and kernel diagnostics.
- **Soak tests.** Multi-hour cancellation and churn runs for thermal and leak behaviour.
- **Lints.** Ratchet `indexing_slicing` (678 sites today) crate by crate, starting with
  `tur-core` and `tur-cli`.
