# Overview

## Crates

```text
tur-core     shared errors (TurError, Result) and checked packed-execution descriptors
   ↑
tur-metal    state pools, completion fences, Metal runtime and kernels, packed weights, sampler
   ↑
tur-models   checkpoint validation and loading, Qwen and Gemma text models, chat templates, tool parsing
   ↑
tur-engine   HybridEngine, scheduler, prefix cache, speculation, InferencePipeline, memory plan
   ↑
tur          facade re-exports, HTTP server library; all integration tests and benchmarks
   ↑
tur-cli      the `tur` binary: run, serve, quantize
```

Dependencies only point down. There is one error type, `TurError` in `tur-core`, and no
universal configuration struct: model-specific configuration stays with its model, and
each component owns its own config (`SchedulerConfig`, `PrefixConfig`,
`SpeculationConfig` and so on).

Loading and speculation share one model-owned `Option<DraftHead>`. `None` means the
native drafting head is not loaded; a present head owns its optional vocabulary bound.
There is no separate enable flag that could disagree with it.

## Life of a request

```text
HTTP handler ──WorkerMsg──▶ PipelineWorker (one thread, owns everything below)
                              │
                              ▼
                       InferencePipeline
   submit: render chat template → tokenize → validate limits → restore prefix snapshot
   begin_step: expire deadlines → Scheduler picks decodes + prefill chunks
               → reserve pages/slots → pack one PackedExecution → HybridEngine::submit
                              │
                              ▼
                 HybridModel::forward (fused Metal kernels)
       reads committed pages/slots in place, writes into reserved ones,
       samples on the device; a notifier thread fences the pass
                              │
                              ▼
   finish_step: HybridEngine::finish → per-request sample/verify → commit boundary
               → detokenize → snapshot prefixes → publish text and outcomes
                              │
                              ▼
            bounded stream channel / oneshot result ──▶ HTTP response
```

A request's text is rendered by the checkpoint's own chat template, and the template's
token IDs feed the pipeline. Each pass packs every scheduled sequence into one call
of the model. Some sequences are decoding one token; others contribute a prefill chunk
of up to `--prefill-chunk-size` tokens. The model reads the committed state of every
sequence where it lives, writes new keys, values and recurrent states into freshly
reserved pages and slots, and returns logits only for the rows that need sampling.

After the device completes, the engine samples or verifies each sequence independently
and commits its new boundary, so one request's failure never affects the others. The
pipeline decodes text, records prefix snapshots and publishes results.

## Principles

- **Explicit state.** Every layer declares its state: paged attention K/V, or recurrent
  convolution and matrix state. A request owns its state; weights are shared and
  immutable.
- **Nothing published before the fence.** Results are read only after the GPU work that
  produced them completes, and state is reclaimed only when no unfinished work can
  reference it.
- **No copies on the hot path.** Kernels read committed state in place, snapshots share
  state rather than copying it, and logits are copied to the host once per pass, if at
  all.
- **Measured defaults.** Every default and every custom kernel was chosen by measurement,
  and losing variants were deleted. The "Decisions" section of each chapter records
  the evidence, and [Rejected experiments](../design/rejected.md) lists what was tried
  and dropped.
