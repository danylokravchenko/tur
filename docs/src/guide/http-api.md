# HTTP API

`tur serve` exposes an OpenAI-compatible API for one chat model and, with `--whisper`, one speech-to-text model. Chat requests carry text, plus audio for models with an audio encoder (Gemma 4 E2B and E4B). Requests and responses follow OpenAI's shapes, with a few extra fields for the chat template and prefix reuse. Request bodies may be up to 64 MiB.

## `GET /v1/models`

Lists the served chat model, and the transcription model when one is loaded:

```json
{"object":"list","data":[{"id":"qwen-hybrid","object":"model","created":1792867193,"owned_by":"tur","revision":"2fc06364715b967f1860aea9cf38778875588b17"}]}
```

`revision` is the checkpoint's immutable revision, so a client can tell exactly what is answering it.

## `POST /v1/chat/completions`

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-hybrid",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 16,
    "temperature": 0,
    "thinking": false
  }'
```

### Request

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `messages` | array | required, non-empty | Conversation turns, see below |
| `model` | string | served model | If present, must equal `--served-model-name` (otherwise 404) |
| `max_tokens` | integer > 0 | 1024, or what the context leaves | Most tokens to generate, counting EOS. An explicit value must fit the context |
| `stop` | string or array | none | End the answer before the first of these texts (up to 16, each up to 1024 bytes). The text is left out, and `finish_reason` is `stop` |
| `stream` | bool | false | Server-sent events instead of one JSON response |
| `temperature` | number ≥ 0 | checkpoint | 0 is greedy |
| `top_k` | integer > 0 | checkpoint | |
| `top_p` | number in (0, 1] | checkpoint | |
| `repetition_penalty` | number > 0 | none | Above 1 discourages any token already in the prompt or output |
| `presence_penalty`, `frequency_penalty` | number | none | OpenAI semantics over generated tokens |
| `seed` | integer | 299792458 | Seed of the request's random stream; other requests in the batch don't affect it |
| `tools` | array | none | `{type:"function",function:{name,description,parameters}}` |
| `tool_choice` | `"auto"` | `"auto"` | Any other value is rejected |
| `thinking` (alias `enable_thinking`) | bool | template default | Passed to the chat template |
| `reasoning_effort` | `low`, `medium`, `xhigh` | template default | Passed to the chat template |
| `preserve_thinking` | bool | template default | Passed to the chat template |
| `cache_salt` | string | none | Prefix-reuse namespace: only requests with the same salt share cached prompts |

Each message has a `role` of `system`, `user`, `assistant` or `tool`. `content` is a string or an array of `{"type":"text","text":...}` parts. Assistant turns may carry `reasoning_content` and `tool_calls` (function calls with JSON-encoded `arguments`), and tool results are `tool` turns. The checkpoint's own chat template renders all of them.

User content may also hold audio parts, in OpenAI's `input_audio` form or as an `audio_url` holding a `data:` URL:

```json
{"role": "user", "content": [
  {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}},
  {"type": "text", "text": "Transcribe this audio."}
]}
```

Audio must be WAV (integer PCM or float, any rate and channel count) and 40 ms to 30 s long. Only models with an audio encoder accept it; others answer 400. Remote URLs are rejected, since the server never fetches media. Audio positions count toward `prompt_tokens`, 25 per second. See [Audio](../architecture/audio.md).

Image and video parts are rejected, and so are prompts containing the model's image placeholder tokens or an audio placeholder without a clip.

### Response

```json
{
  "id": "chatcmpl-5973f2662efe4acba8df912a7343688c",
  "object": "chat.completion",
  "created": 1792867193,
  "model": "qwen-hybrid",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 14, "completion_tokens": 12, "total_tokens": 26,
    "prompt_tokens_details": {"cached_tokens": 0}
  }
}
```

`finish_reason` is `stop` when the model ends its turn, `length` when it hits `max_tokens`, or `tool_calls`. `completion_tokens` counts sampled tokens, including the end-of-turn token. `cached_tokens` counts prompt tokens restored from a prefix snapshot rather than computed, which is a quick way to check that prefix reuse is working.

For gpt-oss, which writes its reasoning on a separate Harmony channel, `message.reasoning_content` holds the reasoning and `content` only the answer. Send it back in `reasoning_content` on later turns with tool calls, and the template replays it.

When the model emits complete function calls, Tur parses them against each tool's parameter schema and returns them as `tool_calls` with no `content`. Tur never runs the tools itself.

### Streaming

With `"stream": true`, the response is `text/event-stream`:

1. A chunk with `delta: {"role": "assistant"}`.
2. Text chunks with `delta.content`, one per model pass at most. Reasoning a model separates arrives the same way in `delta.reasoning_content`.
3. Tool-call chunks, if any.
4. A final chunk with `finish_reason` and `usage`.
5. `data: [DONE]`.

The server sends a keep-alive comment every 15 seconds. If generation fails mid-stream, it sends an error object as a data event, then `[DONE]`.

A slow client never holds up other requests. Each stream has a small bounded queue, and text that doesn't fit waits in a buffer. If a client falls more than 16 KiB behind, the server cancels its request with a `slow_client` error and frees its memory. A disconnected client is noticed within one pass.

## `POST /v1/audio/transcriptions`

Transcribes speech with the `--whisper` model. The body is `multipart/form-data`, as OpenAI's API takes it:

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=whisper-1 -F file=@speech.wav -F language=en
```

```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `file` | required | A WAV file |
| `model` | the transcription model | Must equal `--transcription-model-name` (otherwise 404) |
| `language` | detected | ISO 639-1 code such as `en` |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt` or `vtt` |
| `timestamp_granularities[]` | `segment` | `segment` or `word`; `word` needs `verbose_json` and adds a `words` list |
| `prompt` | none | Text the decoder reads before every window, for names and style |
| `condition_on_previous_text` | `false` | `true` also reads the text decoded so far (a Tur extension) |
| `temperature` | 0 | 0 decodes greedily and retries doubtful windows at rising temperatures; up to 1 samples at that temperature, without retries |

`verbose_json` adds the task, language, duration in seconds and timed segments:

```json
{"task": "transcribe", "language": "en", "duration": 2.66, "text": "The quick brown fox jumps over the lazy dog.",
 "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": 2.66, "text": " The quick brown fox jumps over the lazy dog.", "tokens": [50365, 440, 1702]}]}
```

Silent windows produce no text. Clips longer than 30 s are transcribed window by window, each resuming where the last complete segment of the one before it ended, so no word is cut between them. The 64 MiB request limit bounds a clip's length. A full transcription queue answers 429.

## `POST /v1/audio/translations`

Takes the same form and returns English text translated from the speech. Whisper large-v3-turbo was not trained for translation, so prefer a multilingual checkpoint such as large-v3 for this endpoint.

## Operating the server

These endpoints have no authentication, like the rest of the API, so keep the port away from untrusted clients.

### `GET /health`

Liveness. Returns 200 with `{"status":"ok"}` while the process answers and every worker runs, and 503 with `{"status":"worker_stopped","workers":[...]}` once one has stopped. Nothing restarts a stopped worker, so restart the server when this fails.

### `GET /ready`

Readiness. Returns 200 with `{"status":"ready"}` when new requests would be served, and 503 for a stopped worker or a server that is draining. The listener opens only after every model has loaded, so a server that accepts connections has finished loading.

### `GET /metrics`

Counters and gauges in Prometheus text format. The ones you'll usually want:

| Metric | Meaning |
| --- | --- |
| `tur_requests_total{model,outcome}` | Finished requests: `ok`, `cancelled` or an error code. It counts failures mid-stream, which HTTP status can't show |
| `tur_http_requests_total{path,status}` | Finished HTTP exchanges by route |
| `tur_request_duration_seconds`, `tur_time_to_first_token_seconds` | Latency histograms of successful requests |
| `tur_requests_running`, `tur_requests_waiting` | Current load |
| `tur_prompt_tokens_total`, `tur_cached_prompt_tokens_total`, `tur_generated_tokens_total` | Token counts of finished requests |
| `tur_preemptions_total` | Recomputations forced by memory pressure; a steady rise means the pool is too small for the load |
| `tur_kv_pages_leased`, `tur_kv_pages_peak`, `tur_kv_pages_capacity` | State pool occupancy |
| `tur_context_tokens`, `tur_memory_plan_*_bytes` | What startup planned |
| `tur_device_allocated_bytes`, `tur_process_resident_bytes` | What is actually allocated |
| `tur_system_spare_memory_bytes`, `tur_system_swapouts_total` | Memory pressure on the machine; swap is counted from server start |
| `tur_pass_phase_seconds_total{phase}`, `tur_device_seconds_total` | Where pass time goes |
| `tur_model_info{source,revision}` | The served checkpoint, as labels |

### `GET /diagnostics`

One JSON document for a person debugging the server: version, uptime and system memory, then for each model its revision, weight and K/V formats, context limit, the whole memory plan, scheduler limits, device capabilities and kernel tuning, pool occupancy, and sampling, prefix and speculation counters. `context` reads:

```json
{"limit_tokens": 8192, "full_contexts": 4, "sized_for_tokens": 8192, "max_running": 4}
```

`limit_tokens` is the most one request may hold, and `full_contexts` is how many requests of `sized_for_tokens` the pool holds at once. When `full_contexts` is below `max_running`, some requests wait or recompute at full length.

## Errors

Errors use OpenAI's shape:

```json
{"error": {"message": "Request queue is full: …", "type": "rate_limit_error", "code": "queue_full"}}
```

| Status | `type` / `code` | Cause |
| --- | --- | --- |
| 400 | `invalid_request_error` | Empty messages, unknown role, invalid sampling values, unsupported `tool_choice`, undecodable or unsupported audio, or a request the engine rejects (`invalid_request`) |
| 400 | `invalid_request_error` / `context_length_exceeded` | The prompt and `max_tokens` don't fit the context the server allows |
| 404 | `invalid_request_error` | Unknown `model` |
| 408 | `timeout_error` / `slow_client` | A streaming client stopped reading |
| 413 | (plain text) | The body exceeds 64 MiB |
| 422 | (plain text) | The body doesn't deserialize, for example an image content part |
| 429 | `rate_limit_error` / `queue_full` | The queue already holds `--max-waiting` requests |
| 503 | `server_error` / `insufficient_memory` | The request can't fit in memory even running alone |
| 503 | `server_error` / `worker_unavailable` | The inference worker stopped |
| 503 | `server_error` / `server_shutting_down` | The server is draining |
| 504 | `timeout_error` / `deadline_exceeded` | `--request-timeout-secs` elapsed |
| 500 | `server_error` / `internal_error` | Anything else |

CORS is permissive, and every request is logged under the `tower_http` target.
