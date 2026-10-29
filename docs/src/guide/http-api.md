# HTTP API

`tur serve` exposes an OpenAI-compatible API for one model. It accepts text only. Requests and responses follow OpenAI's chat-completion shapes, with a few extra fields for the chat template and prefix reuse.

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
| `model` | string | served model | If present, must equal `--served-model-name` (otherwise 404) |
| `max_tokens` | integer > 0 | 1024 | Most tokens to generate, counting EOS; must fit the model's context |
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

Image, video and audio parts are rejected, and so are prompts containing the model's image or audio placeholder tokens.

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

When the model emits complete function calls, Tur parses them against each tool's parameter schema and returns them as `tool_calls` with no `content`. Tur never runs the tools itself.

### Streaming

With `"stream": true`, the response is `text/event-stream`:

1. A chunk with `delta: {"role": "assistant"}`.
2. Text chunks with `delta.content`, one per model pass at most.
3. Tool-call chunks, if any.
4. A final chunk with `finish_reason` and `usage`.
5. `data: [DONE]`.

The server sends a keep-alive comment every 15 seconds. If generation fails mid-stream, it sends an error object as a data event, then `[DONE]`.

A slow client never holds up other requests. Each stream has a small bounded queue, and text that doesn't fit waits in a buffer. If a client falls more than 16 KiB behind, the server cancels its request with a `slow_client` error and frees its memory. A disconnected client is noticed within one pass.

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
| 422 | (plain text) | The body doesn't deserialize, for example a non-text content part |
| 429 | `rate_limit_error` / `queue_full` | The queue already holds `--max-waiting` requests |
| 503 | `server_error` / `insufficient_memory` | The request can't fit in memory even running alone |
| 503 | `server_error` / `worker_unavailable` | The inference worker stopped |
| 504 | `timeout_error` / `deadline_exceeded` | `--request-timeout-secs` elapsed |
| 500 | `server_error` / `internal_error` | Anything else |

CORS is permissive, and every request is logged under the `tower_http` target.
