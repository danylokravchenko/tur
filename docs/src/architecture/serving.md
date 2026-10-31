# Serving

The HTTP server is an axum application (`tur::server`) in front of a `PipelineWorker`, and optionally a `TranscriptionWorker` for speech. Its job is to keep the GPU busy while staying responsive to many clients, and to make sure no single client, slow or broken, can hold up the rest.

## The worker

`PipelineWorker` owns one `InferencePipeline` on a dedicated thread. The engine has a single owner, and model work must never block the async handlers.

- `PipelineWorker::spawn` loads and validates the checkpoint and reserves its memory before returning. The CLI opens the port only afterwards, so a listening server can answer.
- Handlers send each request to the worker as a `WorkerMsg` (the request, an optional stream sender and a result sender) over a bounded channel. The worker accepts new requests between passes and also while the GPU is running one, so tokenizing and admitting requests overlaps GPU work.
- Each loop iteration runs `begin_step`, handles new requests and disconnects while the pass is in flight, then runs `finish_step`, sends text to streams and completes finished requests.
- A request's error goes only to that request. If a whole pass fails, every pending request gets the error, and the worker stops and reports it.

## Transcription

`TranscriptionWorker` owns a Whisper model on its own thread and Metal queue, and takes clips from a bounded queue (16 by default, `--max-waiting-transcriptions`). A full queue returns HTTP 429. Handlers decode uploads and base64 audio on Tokio's blocking pool, so a large WAV file never stalls the async runtime. Chat clips go to a separate encoder thread that holds up to 8 waiting requests, and a full one returns HTTP 429; each request rejoins the worker's intake once its clips are encoded. [Audio](audio.md) covers the models.

## Reporting

Each pass the worker publishes its counters and pool occupancy to a small shared structure, so a scrape never interrupts the worker or waits for a pass. `/metrics` and `/diagnostics` read that copy, and a background task adds system memory every 10 seconds. HTTP counters are keyed by route pattern, so client-chosen paths can't grow them. See [HTTP API](../guide/http-api.md#operating-the-server).

## Shutdown

On SIGINT or SIGTERM the listener closes, new requests on open connections get 503 `server_shutting_down`, and accepted requests run to completion, including queued ones. Then the workers' senders drop, each worker exits and releases its model, and the process exits with status 0.

If requests are still running after `--shutdown-timeout-secs`, or at a second signal, the worker fails what it holds with `worker_unavailable` between passes, so no buffer is released while the GPU may use it.

## Streaming

Each streaming request has a bounded channel of 64 messages and a buffer behind it.

- Each pass sends at most one message per request, and text the channel can't take waits in the buffer.
- If a client falls more than 16 KiB behind, its request is cancelled with `slow_client` (HTTP 408, or an error event in the stream) and its memory is released. A stalled client can therefore hold only a small, fixed amount of memory.
- Finishing never waits on a client. The final text travels with the result, and the stream adapter sends whatever the channel couldn't take.
- Disconnects are checked every pass, while the GPU runs, and free the request's state.

The listener, model ID, admission and deadline settings, prefix retention and streaming limits can all go in [`tur.toml`](../guide/configuration.md), and flags override the file. Streaming limits are file-only and are checked before the model loads.

## Decisions

- One chat model per server, plus at most one Whisper model. The library can share one GPU budget among several pipelines; the Whisper worker leases its memory first, and the chat pool is sized from what remains.
- Request bodies may reach 64 MiB, enough for a few minutes of 48 kHz stereo WAV in base64.
- New requests are accepted while the GPU runs. With requests arriving at random, this cut the median time to first token by 13 to 14% and halved the time the GPU sat idle, to about 5%.
- Everything is bounded: intake, waiting requests, waiting tokens, stream messages and stream buffers, each with its own error.
- Errors are typed. A full queue, a missed deadline, lack of memory and a slow client each map to a distinct HTTP status (see [HTTP API](../guide/http-api.md#errors)).
- Deployment settings can live in TOML. Stream limits join the other server limits without more flags, and generation settings stay per request.
- Stop sequences hold text back. Text that could still become a sequence stays in the worker until it completes or can't, so a stream never shows part of a stop sequence. Only the answer is searched; reasoning isn't.
- Metrics are written by hand in Prometheus text, with no new dependency. The format is small, and the counters already had one writer.
- `/health` also fails when a worker has stopped. Nothing restarts a worker, so a supervisor should replace the process.
- Local checkpoints report as `local`. Clients get the revision, not the server's file paths, and no endpoint reports prompts.
