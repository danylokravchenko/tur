# Scope and principles

## What Tur is for

Tur is an inference service for Apple Silicon Macs. It aims for three things at once: fast prompt processing and first tokens, fast sustained generation, and steady latency when many requests run together. One Mac is the unit of deployment.

## In scope

- Apple Silicon Macs with Metal. CPU execution exists only as a numerical reference for tests.
- Qwen3.5-family hybrid text models (dense Qwen3.5 and Qwen3.8, and experimentally the Qwen3.6 MoE) and Gemma 4 text models. The architecture is identified from checkpoint fields, never from the name.
- Speech: audio input to Gemma 4 E2B and E4B through their own encoder, and Whisper transcription (see [Audio](../architecture/audio.md)).
- The `tur` CLI, an OpenAI-compatible HTTP API and the Rust library.

## Out of scope

- Intel Macs, CUDA and Linux.
- Image and video input, which are rejected until encoders and their position handling exist.
- Speech output and audio containers other than WAV.
- Other model families, such as plain Qwen3 transformers.
- Splitting a model across GPUs or machines, spilling state to SSD, and the Apple Neural Engine. These wait until a measured workload needs them.

## Validation status

| Model | Evidence |
| --- | --- |
| Qwen3.5-0.8B | Matches Transformers per layer and in logits on CPU and Metal (within 3.5e-6 relative error, against a 1e-4 gate); Q8_0 meets its 95% top-1 floor |
| Qwen3.8-27B | Pinned config, tensor headers and template validated, and it shares the 0.8B code path; not yet run, because its Q4K weights exceed a 16 GB Mac |
| Qwen3.6-35B-A3B | Passes an independent small Transformers fixture and pinned headers; full execution, quality and expert-kernel performance are open |
| Gemma 4 E2B | Pinned Transformers F32 architecture fixtures, and real Q8_0 checkpoint generation; its Q8_0 audio encoder matches the F32 Transformers encoder at cosine similarity 0.9999 |
| Whisper large-v3-turbo | Small Transformers fixture for the encoder, decoder and cached greedy decode; real F32 and Q8_0 transcripts equal Transformers' on a pinned clip |
| Gemma 4 E4B, 12B, 26B-A4B, 31B | Architecture fixtures and configuration checks; full checkpoints not yet run |

## Principles

- Evidence over intent. Every performance claim comes from a measurement on named hardware. All numbers in this book come from one M1 with 16 GB, running `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, unless stated otherwise. A kernel or format wins only beyond run-to-run noise, and the losers are deleted and listed in [Rejected experiments](rejected.md).
- Exact by default. Speculation, prefix reuse and preemption never change a request's output distribution. Approximate modes such as token eviction or sparse attention would have to be explicit and opt-in.
- Breaking changes over shims. Interfaces change when a better design needs it, and replaced code is deleted in the same change.
- Safety. No unsafe Rust and no panics; every failure is a typed error (see [Workflow and rules](../development/workflow.md)).

## Where the evidence is

This book keeps the decisions and the current measurements ([Measured performance](../guide/performance.md)). The development notes, raw benchmark output and rejected patches behind them live in git history at commit `99b073e`:

```sh
git ls-tree -r --name-only 99b073e docs
```
