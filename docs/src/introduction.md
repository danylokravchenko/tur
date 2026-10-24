# Introduction

Tur is a Rust inference engine for hybrid language models on Apple Silicon. It loads a
Qwen3.5-family checkpoint, which interleaves Gated DeltaNet recurrent layers with gated
full-attention layers, and runs it on fused Metal kernels. It serves the model through a
command-line tool, an OpenAI-style HTTP server, or a Rust library.

What it does:

- **Packed continuous batching.** Every model pass packs decodes and prefill chunks from
  many requests into one execution, without padding. Requests enter and leave between
  passes.
- **Paged, persistent state.** Attention keys and values live in page pools, and
  recurrent states live in slot pools. Kernels read both in place. K/V can be stored as
  F32, F16 or Q8.
- **Prefix reuse.** Snapshots of a prompt's state, at exact token boundaries, are reused
  by later requests that share the prefix. Taking a snapshot copies nothing.
- **Device sampling.** Temperature, top-k, top-p, penalties and grammar masks run on the
  GPU, with a per-request random stream that does not depend on batch order.
- **Exact speculative decoding.** Drafts from prompt lookup and from the checkpoint's own
  multi-token-prediction head are verified in one pass. The output is identical to
  ordinary decoding.
- **Prepacked weights.** `tur quantize` writes calibrated Q4K/Q8_0 artifacts (GPTQ,
  importance-weighted, or mixed precision under a byte budget) that load as stored.

## Status

| Model | State |
| --- | --- |
| Qwen3.5-0.8B (dense hybrid) | Validated against Transformers per layer and in logits on CPU and Metal |
| Qwen3.8-27B (dense hybrid) | Same code path; config, tensor headers and template validated; not executed (needs more than 16 GB) |
| Qwen3.6-35B-A3B (MoE hybrid) | Experimental: passes a small independent fixture and pinned headers only |

Only text input is supported. All measurements so far come from one M1 with 16 GB, and
every performance number in this book is from that machine unless stated otherwise.
[Measured performance](guide/performance.md) collects the absolute numbers: model
speed, serving throughput and latency by concurrency, prefix reuse, speculation,
loading and quality.

## How this book is organized

- **[Guide](guide/quickstart.md)** starts with a quickstart (install, download, run,
  serve, quantize). It then covers the command line, the HTTP API, choosing a
  quantization, tuning recipes and the Rust library.
- **[Architecture](architecture/overview.md)** explains how a request becomes packed GPU
  work: the model contract, the engine, state memory, scheduling, prefix reuse,
  sampling, speculation, kernels and the server.
- **[Development](development/workflow.md)** covers the workflow, the rules every change
  follows, the test suite and the benchmarks.
- **[Design](design/scope.md)** states the scope and principles, lists rejected
  experiments with pointers to their evidence, and tracks open work. Every architecture
  chapter ends with the decisions behind it and the measurements that justified them.

Build the book locally with `just docs`, which serves it at `http://localhost:3000`.
