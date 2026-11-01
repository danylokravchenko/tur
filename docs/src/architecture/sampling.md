# Sampling

Sampling turns each request's logits into its next token. A model's vocabulary has hundreds of thousands of entries, and copying every request's row back to the CPU each step costs time that grows with the batch. Tur samples on the GPU instead, right after the model pass and inside the same batch, so only the chosen token and a status word come back.

## The algorithm

The same algorithm exists twice: as the `penalize_f32` and `sample_f32` kernels, and as a CPU reference in `tur-metal`'s `sampling` module. A request draws from the same distribution on either path.

1. Penalties. The repetition penalty divides the positive logits (and multiplies the negative ones) of every token already in the prompt or output. The presence penalty is subtracted once per generated token, and the frequency penalty once per occurrence.
2. Weights. `w = exp((logit - max) / temperature)`. A temperature of 0 means greedy argmax.
3. Truncation. Token `j` survives if fewer than `top_k` tokens have a strictly larger weight and those tokens hold less than `top_p` of the total. That is top-k followed by top-p measured on the whole distribution, as Candle's sampler does it.
4. Drawing without sorting. Sorting a large vocabulary on the GPU is expensive, so Tur uses rejection sampling instead. Each round inverts the cumulative distribution of the weights above a pivot with a uniform random number. If the token it lands on survives truncation, that's the answer. If not, every token no more likely than it is also outside the truncated set, so the pivot rises to its weight and the next round samples only from the heavier tokens. The result is exactly a sample from the renormalized truncated distribution. The heaviest token always survives, so the rounds end; after 64 rounds the GPU marks the row unfinished and the CPU samples it.

The random numbers come from a counter-based generator keyed by the request's seed and its draw count. A request's draws never depend on which other requests share its batch or in what order.

## Grammar-constrained requests

A request with a `grammar` (an llguidance `TopLevelGrammar`) gets a mask of allowed tokens before every sample, and its grammar state advances after it. The mask is applied on the GPU, so constrained requests sample in the same pass as unconstrained ones. A request ends when its grammar is complete. Constrained requests never draft (see [Speculative decoding](speculation.md)).

The server compiles `response_format` and forced `tool_choice` into such grammars, one form per model family, because each writes tool calls differently. Special tokens appear in the grammar by id, since a grammar can match them only as tokens. A finished grammar allows the checkpoint's own end-of-generation tokens, so a mask can't hide the token a model ends its turn with.

When a mask allows exactly one token, sampling would return it with probability 1, so Tur publishes it without a pass. Forced tokens then run together as the request's next inputs, the way a prompt runs, and the scheduler treats the run as a decode so it doesn't wait for prefill budget. Each forced token consumes one random draw, as sampling it would, so same-seed outputs stay identical.

Under `tool_choice=auto` a grammar is lazy. It sleeps, costing nothing and leaving drafting on, until the model commits the tool-call marker, holds the call to its schema, and sleeps again. If the marker lands inside a drafted pass, the pass's later outputs are dropped as if their draft had been rejected, because their rows weren't masked.

## Log probabilities

The GPU returns only the chosen token, so a request that asks for log probabilities also copies its model logits to the host, one row per output. Sampling still happens where it would have, on the same row, and the copy is read only after that. The request's tokens therefore match a request that didn't ask.

## When the CPU samples

The CPU sampler handles rows when device sampling is off (`PipelineConfig::device_sampling = false`), when the model runs on the CPU reference, for speculative requests and grammar requests with several output rows, and for rows the GPU couldn't settle. All such rows are gathered before the pass finishes and copied back once.

## Decisions

- Sampling runs on the GPU. With temperature 0.7, top-k 20 and top-p 0.95, the median gap between tokens at 8 and 16 concurrent requests fell 1.4× compared with copying rows to the CPU. Sampled serving now runs within about a millisecond per token of greedy serving.
- There is no sort. Rejection sampling against a rising pivot keeps the kernel to a few passes over the row.
- Each request has its own random stream, independent of batch order. Device sampling passes chi-square tests against the exact truncated distributions, and repeated runs on Metal produce the same tokens.
- Grammar masks are applied on the GPU, so constrained passes cost the same as unconstrained ones.
- JSON grammars allow no free whitespace, which a model can otherwise emit until it runs out of tokens.
- Forced tokens run in one pass. On JSON with long keys, Qwen3.5-0.8B on an M1 reached 155, 340 and 424 tok/s with 1, 4 and 8 requests, against 52, 150 and 237 sampling each one. With few forced tokens throughput is level, but passes holding runs are longer: at 8 requests the p95 token gap rose from 27 to 47 ms. Whole-document clients care about completion, so forcing stays on.
- Forced runs schedule as decodes. As prefill, one chunk per pass, they cost 5% of throughput at 8 requests.
- Reasoning is optional before forced output and off by default on Qwen and Gemma, whose reasoning Tur splits from the answer only for structured requests. A response to any other request is unchanged.
- Lazy grammars are best effort. A schema they can't enforce falls back to unconstrained calls, because the request may never make a call.
- Log probabilities cost about 2 ms per token on a 248K vocabulary: at 1, 4 and 8 requests throughput fell 6%, 22% and 25%, with little difference for `top_logprobs`.
- Log probabilities describe the model's distribution before penalties, temperature, truncation and masks, because that is the same whichever sampler runs.
- The two paths sample the same distribution, but the same random number can pick different tokens, because the GPU walks tokens in a fixed permuted order. Random numbers have 24 bits, so a token with less than about 2⁻²⁴ of the remaining probability is effectively never drawn.
