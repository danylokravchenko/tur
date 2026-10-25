# Serving

The HTTP server is an axum application (`tur::server`) in front of a `PipelineWorker`.

## The worker

`PipelineWorker` owns one `InferencePipeline` on a dedicated thread, because the engine
is a single owner and model work must never block async handlers.

- **Startup.** `PipelineWorker::spawn` loads and validates the checkpoint, plans memory
  and leases it before returning. The CLI binds the listener only afterwards, so a bound
  port means a usable model.
- **Intake.** Handlers send a `WorkerMsg` (request, optional stream sender, result
  sender) over a bounded channel. The worker drains intake between passes, and also while
  the GPU runs one, so tokenization and admission overlap device work.
- **Loop.** Each iteration runs `begin_step`, handles intake and disconnects while the
  pass is in flight, then runs `finish_step`, publishes text to streams and completes
  finished requests.
- **Isolation.** A request's error goes only to that request. If a pass fails as a
  whole, every pending request receives the error, and the worker stops and reports it.

## Streaming

Each streaming request has a bounded channel (64 messages) and a coalescing buffer.

- **Coalescing.** Each pass sends at most one message per request. Text the channel
  cannot take waits in the buffer.
- **Slow clients.** If the buffer exceeds 16 KiB, the request is cancelled with
  `slow_client` (HTTP 408, or an SSE error event) and its state is released. A stalled
  client therefore holds at most 65 messages of bounded size.
- **Completion never waits.** The final text is sent with the result, and the SSE adapter
  emits whatever the channel could not take.
- **Disconnects** are detected every pass, while the device runs, and release the
  request's state.

The listener, model ID, admission and deadline settings, prefix retention and streaming
bounds can be set in [`tur.toml`](../guide/configuration.md). CLI flags override file
values. Streaming limits are file-only controls, checked before loading the model.

## Decisions

- **Persist deployment settings in TOML.** Stream channel and pending-text bounds join
  the existing server limits without adding more flags. HTTP generation settings remain
  per request.
- **One resident model per server.** Multi-model arbitration exists in the library (one
  `DeviceBudget` shared by several pipelines), but the server loads one model.
- **Overlap intake with execution.** Under streaming Poisson arrivals, TTFT p50 fell
  13–14%, and device idle fell from 9.5% to 5% of wall time. The device idles at most
  0.06–0.31 ms between passes (a lower bound).
- **Bounded everything.** Intake, waiting requests, waiting tokens, stream messages and
  stream buffers are all bounded, and every bound has an explicit error.
- **Errors are typed.** Queue overflow, deadlines, memory and slow clients map to distinct
  HTTP statuses (see [HTTP API](../guide/http-api.md#errors)).
- **Not yet production-hardened:** there are no health or readiness endpoints, metrics,
  graceful shutdown or stop sequences (see [Open work](../design/open.md)).
