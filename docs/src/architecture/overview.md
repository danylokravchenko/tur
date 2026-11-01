# Overview

This part of the book follows a request from the HTTP server to the GPU and back. This page gives the map: how the crates fit together, what happens to a request, and the few principles that every chapter relies on. To decide whether Tur fits your setup, start with [Tur and other engines](comparison.md).

## Crates

```text
tur-cli      the `tur` binary: run, serve, quantize
   |
tur          facade re-exports and HTTP server library; all integration tests and benchmarks
   |
tur-engine   HybridEngine, scheduler, prefix cache, speculation, InferencePipeline, memory plan
   |
tur-models   checkpoint validation and loading, Qwen and Gemma text models, chat templates, tool parsing
   |
tur-metal    state pools, completion fences, Metal runtime and kernels, packed weights, sampler
   |
tur-core     shared errors (TurError, Result) and checked packed-execution descriptors
```

Each crate depends only on the ones below it. There is one error type, `TurError`, and no global configuration struct: model-specific settings stay with their model, and each component owns its own config (`SchedulerConfig`, `PrefixConfig`, `SpeculationConfig` and so on).

## Life of a request

```text
HTTP handler --WorkerMsg--> PipelineWorker (one thread, owns everything below)
                              |
                              v
                       InferencePipeline
   submit:     render chat template, tokenize, check limits, restore a prefix snapshot
   begin_step: expire deadlines, pick decodes and prefill chunks,
               reserve pages and slots, pack one PackedExecution, HybridEngine::submit
                              |
                              v
                 HybridModel::forward (fused Metal kernels)
       reads committed pages and slots in place, writes into reserved ones,
       samples on the GPU; a notifier thread waits for the pass to finish
                              |
                              v
   finish_step: HybridEngine::finish, sample or verify per request, commit,
                detokenize, take prefix snapshots, publish text and results
                              |
                              v
            bounded stream channel or one-shot result --> HTTP response
```

The checkpoint's own chat template turns the request into tokens. Each pass then packs every scheduled request into a single call of the model: some contribute one new token, others a chunk of their prompt. The model reads each request's state where it already lives, writes the new keys, values and recurrent state into freshly reserved space, and returns logits only for the rows that need a token sampled.

Once the GPU finishes, the engine handles each request on its own: it samples or verifies the next tokens and commits the new state. A failure in one request never touches the others. The pipeline then decodes text, takes prefix snapshots and sends results back.

## Principles

- Every layer declares its state explicitly: attention keys and values in pages, or recurrent state in slots. A request owns its state, and weights are shared and read-only.
- Nothing is read before the GPU finishes. Results are read only after the work that produced them completes, and memory is reused only when no unfinished work can still touch it.
- The hot path makes no copies. Kernels read state in place, snapshots share state instead of copying it, and logits come back to the CPU at most once per pass.
- Defaults come from measurements. Every default and every custom kernel won a measurement on the M1, and the variants that lost were deleted. Each chapter's "Decisions" section records the evidence, and [Rejected experiments](../design/rejected.md) lists what lost.
