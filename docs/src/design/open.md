# Open work

## Models and hardware

- **Run the flagship.** Execute Qwen3.8-27B and the Qwen3.6-35B MoE on Macs that fit
  them. Measure quality, the K/V format choice (Q8 is the likely default under 16 GB) and
  the MoE wide tier, which trails Candle by 1.2–1.5× and needs trained routing traces.
- **Flagship MTP head** for expert checkpoints.
- **Gemma 4** as a separate architecture adapter, not a renamed Qwen model.
- **More devices.** Every number so far is from one M1. Launch shapes and crossovers
  need per-device sweeps, and the minimum supported macOS and chip still has to be
  decided.

## Performance

- **Comparative baseline.** Compare against pinned MLX-LM and llama.cpp on the same Mac at
  matched quantization, with prompts of 128/2K/8K/32K tokens, concurrency from 1 to 16,
  cold and warm prefixes, and sustained load.
- **Quality budgets.** Predeclared held-out budgets for language, code, tool/JSON and
  long-context retrieval, with confidence intervals.
- **Chained MTP drafts.** A second draft through the head; an estimated 10% at one
  sequence.
- **Layer-at-a-time calibration,** so `tur quantize` scales to the flagship.
- **Tiled prefill attention** runs at about 25% of the matrix units' F32 peak. Staging is
  single-buffered, with sixteen barriers per 32-key block at head dimension 256.
- **F16/BF16 activations**, once quality budgets exist.

## Serving

- **Production hardening.** Health and readiness endpoints, request metrics, graceful
  shutdown, stop sequences, model revision reporting, and memory and kernel diagnostics.
- **Soak tests.** Multi-hour cancellation and churn runs for thermal and leak behaviour.
- **Multi-turn reuse.** When a template re-renders history differently from the
  generated text, reuse loses up to 255 tokens per turn. Fixing it needs a pass boundary
  where the next turn's rendering diverges.
- **Lints.** Ratchet `indexing_slicing` (678 sites today) crate by crate, starting with
  `tur-core` and `tur-cli`.
