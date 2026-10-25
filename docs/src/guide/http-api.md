# HTTP API

`tur serve` exposes an OpenAI-style API for one resident model. Only text input is
supported. Responses follow OpenAI's chat-completion shapes; extra request fields
control Qwen's chat template and prefix reuse.

## `GET /v1/models`

Lists the served model:

```json
{"object":"list","data":[{"id":"qwen-hybrid","object":"model","created":1792867193,"owned_by":"tur"}]}
```

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
| `model` | string | served model | Must equal `--served-model-name` when present (else 404) |
| `max_tokens` | integer > 0 | 1024 | Generated-token limit, including EOS; must fit the model context |
| `stream` | bool | false | Server-sent events instead of one JSON response |
| `temperature` | number ≥ 0 | checkpoint | 0 is greedy |
| `top_k` | integer > 0 | checkpoint | |
| `top_p` | number in (0, 1] | checkpoint | |
| `repetition_penalty` | number > 0 | none | Above 1 discourages every prompt or generated token |
| `presence_penalty`, `frequency_penalty` | number | none | OpenAI semantics over generated tokens |
| `seed` | integer | 299792458 | Seed of the request's random stream; independent of batching |
| `tools` | array | none | `{type:"function",function:{name,description,parameters}}` |
| `tool_choice` | `"auto"` | `"auto"` | Any other value is rejected |
| `thinking` (alias `enable_thinking`) | bool | template default | Passed to the chat template |
| `reasoning_effort` | `low`, `medium`, `xhigh` | template default | Passed to the chat template |
| `preserve_thinking` | bool | template default | Passed to the chat template |
| `cache_salt` | string | none | Prefix-reuse namespace: only requests with the same salt share cached prompts |

Messages have a `role` of `system`, `user`, `assistant` or `tool`. `content` is a string
or an array of `{"type":"text","text":...}` parts. Assistant turns may carry
`reasoning_content` and `tool_calls` (function calls with JSON-encoded `arguments`). Tool
results are `tool` turns. The checkpoint's own template renders all of them.

Image, video and audio content parts are rejected, as are prompts that contain the
model's modality placeholder tokens.

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

- `finish_reason` is `stop` (EOS), `length` (`max_tokens`) or `tool_calls`.
- `completion_tokens` counts sampled tokens, including EOS.
- `cached_tokens` counts prompt tokens restored from a prefix snapshot instead of
  recomputed.
- When the model emits complete Qwen XML function calls, they are parsed against each
  tool's parameter schema and returned as `tool_calls`, with `content` omitted. Tur never
  executes tools.

### Streaming

With `"stream": true` the response is `text/event-stream`:

1. A chunk with `delta: {"role": "assistant"}`.
2. Text chunks with `delta.content`, coalesced per model pass.
3. Tool-call chunks, if any.
4. A final chunk with `finish_reason` and `usage`.
5. `data: [DONE]`.

A keep-alive comment is sent every 15 seconds. If generation fails after streaming
began, an error object is sent as a data event, followed by `[DONE]`.

The server never waits for a slow client. Each stream has a bounded channel of 64
messages. Text that does not fit is buffered, and a client whose undelivered text
exceeds 16 KiB is cancelled with a `slow_client` error. A disconnected client is detected
within one pass, and its state is released.

## Errors

Errors use OpenAI's shape:

```json
{"error": {"message": "Request queue is full: …", "type": "rate_limit_error", "code": "queue_full"}}
```

| Status | `type` / `code` | Cause |
| --- | --- | --- |
| 400 | `invalid_request_error` | Empty messages, unknown role, invalid sampling values, unsupported `tool_choice`, or a request the engine rejects (`invalid_request`) |
| 404 | `invalid_request_error` | Unknown `model` |
| 408 | `timeout_error` / `slow_client` | A streaming client stopped reading |
| 422 | (plain text) | Body does not deserialize, for example a non-text content part |
| 429 | `rate_limit_error` / `queue_full` | `--max-waiting` requests are already queued |
| 503 | `server_error` / `insufficient_memory` | The request cannot fit the state pool even alone |
| 503 | `server_error` / `worker_unavailable` | The inference worker stopped |
| 504 | `timeout_error` / `deadline_exceeded` | `--request-timeout-secs` elapsed |
| 500 | `server_error` / `internal_error` | Anything else |

CORS is permissive, and every request is traced at the `tower_http` log target.
