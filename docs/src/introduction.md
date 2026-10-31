# Introduction

Tur is a Rust inference engine for language models on Apple Silicon. It runs Qwen3.5-family hybrid models, Gemma 4 and gpt-oss models on the Mac's GPU, with audio input for Gemma 4 E2B and E4B and Whisper transcription, and you can use it from the command line, through an OpenAI-compatible HTTP server, or as a Rust library.

## What makes it fast

A Mac's GPU shares memory with the CPU, and generating text spends most of its time moving bytes from that memory: the model's weights, plus each request's attention keys and values and recurrent state. Tur is designed to read each byte as few times as possible and to skip work it has already done.

Continuous batching is the biggest lever. Every model pass packs the next token of each running request, along with chunks of newly arrived prompts, into one execution without padding. The weights are read once for the whole batch, so on the M1 eight sequences decode about five times as many tokens per second as one. Requests join and leave between passes, so nobody waits for a batch to fill or drain.

State lives in fixed pools. Attention keys and values sit in pages and recurrent state sits in slots, all allocated once at startup. Kernels read them where they are, which keeps per-token cost flat as a conversation grows and makes memory use predictable: Tur knows before loading a single weight whether a configuration fits.

Prefix reuse avoids computing the same prompt twice. Tur snapshots the state at exact token boundaries, and a later request that starts with the same tokens resumes from the snapshot. Taking a snapshot copies nothing, because the snapshot and the request share the same pages and slots.

Sampling runs on the GPU right after the model pass. Penalties, temperature, top-k, top-p and grammar masks all apply there, and each request draws from its own random stream, so its output doesn't depend on which other requests share the batch.

Speculative decoding drafts likely next tokens, from the request's own text or from a small prediction head, and checks them all in one pass. Every published token is exactly what normal decoding would produce, so it's a pure speed-up when drafts are often right.

Prepacked weights load as stored. `tur quantize` writes 4-bit or 8-bit weights, optionally calibrated on your own text so the 4-bit version loses less quality, and the server starts in a fraction of a second.

## Status

Qwen3.5-0.8B, Gemma 4 E2B and gpt-oss-20b are validated against pinned Transformers references and run on real checkpoints. Larger Qwen and Gemma sizes share the same code and pass configuration and small-fixture checks, but their full checkpoints haven't been run yet. [Scope and principles](design/scope.md) lists the evidence for each model. Tur accepts text and WAV audio; image and video input are rejected (see [Audio](architecture/audio.md)).

Every measurement in this book comes from one Apple M1 with 16 GB unless it says otherwise. [Measured performance](guide/performance.md) collects them.

## How this book is organized

The [Guide](guide/quickstart.md) starts with a quickstart and then covers configuration, the command line, the HTTP API, quantization, tuning and the Rust library.

The [Architecture](architecture/overview.md) part follows a request from HTTP to GPU and back. Each chapter explains one mechanism and ends with a "Decisions" section that records what was chosen and the measurement behind it.

[Development](development/workflow.md) covers the project rules, the test suite and the benchmarks. [Design](design/scope.md) states the scope, lists experiments that lost, and tracks open work.

Run `just docs` to read the book locally at `http://localhost:3000`.
