# Sampling

Sampling turns each sequence's logit row into a token. By default it runs on the GPU,
immediately after the model pass and inside the same fenced batch, so only the chosen
token and a status word return to the host.

## The algorithm

One algorithm is implemented twice: once as the `penalize_f32`/`sample_f32` kernels and
once as a host reference in `tur-metal`'s `sampling` module. A request therefore draws
from the same distribution on either path.

1. **Penalties.** The repetition penalty divides positive logits (and multiplies
   negative ones) of every prompt or generated token. The presence penalty is subtracted
   once per generated token, and the frequency penalty once per occurrence.
2. **Weights.** `w = exp((logit − max) / temperature)`. A temperature of 0 is greedy
   argmax.
3. **Truncation.** Token `j` survives when fewer than `top_k` tokens have a strictly
   larger weight, and their mass is below `top_p` of the total. That is top-k, then top-p
   measured on the whole distribution, as Candle's sampler does.
4. **Draw without sorting.** Each round inverts the CDF of the weights above a pivot,
   with a uniform `u(seed, draw, round)`. An accepted token is returned. A rejected token
   lies outside the truncated set, and so does every token no more probable than it, so
   the pivot rises to its weight and the next round resamples from the strictly heavier
   rest. This is exact rejection sampling from the renormalized truncated distribution.
   The heaviest token is always accepted, so rounds terminate. After 64 rounds the
   device reports the row unconverged and the host samples it explicitly.

Uniforms come from a counter-based generator keyed by the request's seed and its draw
counter. A request's draws never depend on which other requests share its batch, or on
their order.

## Grammar-constrained requests

A request with a `grammar` (an llguidance `TopLevelGrammar`) gets a token mask before
every sample, and the grammar state commits after it. The mask is applied on the device,
so constrained requests sample in the same pass as unconstrained ones. A request
completes when its grammar terminates. Constrained requests never draft (see
[Speculative decoding](speculation.md)).

## When the host samples

Rows go to the host sampler when device sampling is disabled
(`PipelineConfig::device_sampling = false`), when the model runs on the CPU reference,
for speculative or multi-output grammar sequences, and for rows the device could not
settle. All host rows are concatenated before the fence and copied once.

## Decisions

- **Sample on the GPU.** With temperature 0.7, top-k 20 and top-p 0.95, inter-token
  latency p50 at 8 and 16 concurrent requests fell 1.4× compared with copying rows to
  the host (39.9 → 28.5 ms and 72.6 → 50.5 ms). Sampled serving runs within about 1 ms of
  greedy serving.
- **No sort.** Rejection sampling against a rising pivot keeps the kernel a few coalesced
  passes over the row.
- **Per-request RNG, independent of batch order.** Device sampling passes chi-square tests
  against exact truncated distributions, and run-to-run determinism is tested on Metal.
- **Device masks for grammars.** Constrained passes cost the same as unconstrained ones.
- **Known differences.** The host and device paths sample the same distribution, but the
  same uniform can pick different tokens, because the device inverts in a fixed
  permutation of token order. Uniforms have 24 bits, so tokens below about 2⁻²⁴ of the
  remaining mass are effectively never drawn.
