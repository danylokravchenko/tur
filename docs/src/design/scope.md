# Scope and principles

## What Tur is for

A Rust inference service for Apple Silicon Macs that optimizes, together:

- prompt throughput and time to first token;
- sustained token generation;
- tail latency under concurrency.

A single Mac is the deployment unit.

## In scope

- **Platform:** Apple Silicon macOS with Metal. CPU execution exists only as a numerical
  reference for tests.
- **Models:** Qwen3.5-family hybrid text models: dense Qwen3.5 and Qwen3.8, and
  experimentally the Qwen3.6 MoE, plus Gemma 4 text decoders. The architecture is selected from checkpoint fields,
  never from the name.
- **Interfaces:** the `tur` CLI, an OpenAI-style HTTP API, and the Rust library.

## Out of scope

- Intel Macs, CUDA and Linux.
- Image, video and audio input. These are rejected until encoders and their position
  handling exist.
- Ordinary Qwen3 and Granite, whose previous implementations were deleted. Gemma 4
  has a separate attention-only architecture adapter over the common engine.
- Expert, attention or context parallelism, multi-machine serving, SSD spill and the
  Apple Neural Engine. All are deferred until a measured workload justifies them.

## Validation status

| Model | Evidence |
| --- | --- |
| Qwen3.5-0.8B | Per-layer and logit agreement with Transformers (≤ 3.5e-6 relative error against a 1e-4 gate) on CPU and Metal; Q8_0 meets its 0.95 top-1 floor |
| Qwen3.8-27B | Pinned config, tensor headers and template validated; shares the 0.8B code path; **not executed** (its Q4K weights exceed a 16 GB Mac) |
| Qwen3.6-35B-A3B | Independent small Transformers fixture and pinned headers pass; full execution, quality and expert-kernel performance are open |
| Gemma 4 E2B | Text inference; pinned Transformers F32 architecture fixtures and real Q8_0 checkpoint generation validated |
| Gemma 4 E4B / 12B / 26B-A4B / 31B | Architecture fixtures and configuration coverage; full checkpoints not executed |

## Principles

- **Evidence over intent.** Every performance claim comes from a measurement on named
  hardware. All numbers in this book are from one M1 with 16 GB, running
  `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, unless stated otherwise.
  A kernel or format wins only beyond run-to-run noise, and losers are deleted and listed
  in [Rejected experiments](rejected.md).
- **Exactness by default.** Speculation, prefix reuse and preemption never change a
  request's output distribution. Approximate modes (token eviction, sparse attention)
  would have to be explicit and opt-in.
- **Breaking changes over shims.** Interfaces change when a better design needs it, and
  replaced code is deleted in the same change (a project decision of 2026-10-21).
- **Safety.** No unsafe Rust and no panics; every failure is a typed error (see
  [Workflow and rules](../development/workflow.md)).

## Where the evidence is

This book keeps the decisions and the current measurements
([Measured performance](../guide/performance.md)). The development notes, raw benchmark
output and rejected patches behind the decisions were removed from the tree and remain in
git history at commit `99b073e`:

```sh
git ls-tree -r --name-only 99b073e docs
```
