/// Integration tests for the OpenAI-compatible HTTP server.
///
/// All tests use a mock `PipelineWorker` (via `PipelineWorker::from_sender`)
/// so no model weights are needed.  The mock task simulates the contract that
/// the real continuous-batching worker upholds:
///   1. Drain `WorkerMsg`s from the channel concurrently.
///   2. Stream tokens via `token_tx` for each request independently.
///   3. Close `token_tx` and send `GenerationStats` via `result_tx` when done.
///
/// The axum router is exercised via `tower::ServiceExt::oneshot` (no TCP
/// socket needed) for single-request tests, and via a shared `axum::Router`
/// clone for concurrent-request tests.
use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

use tur::{
    backend::{pipeline::GenerationStats, tools::ToolCall},
    server::{
        AppState, build_router,
        worker::{PipelineWorker, WorkerMsg},
    },
};

/// Spawn a Tokio task that acts as the pipeline backend, replying to every
/// generation request with `response_text` split into words.
///
/// The channel capacity matches the real worker (256) so concurrent tests
/// don't block on enqueue.
fn mock_worker(response_text: &'static str) -> PipelineWorker {
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(256);
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            // Stream words via token_tx (for SSE/streaming consumers).
            for word in response_text.split_whitespace() {
                let _ = msg.token_tx.send(word.to_string());
                let _ = msg.token_tx.send(" ".to_string());
            }
            drop(msg.token_tx);
            // generated_text is used by the non-streaming path in blocking_response.
            let _ = msg.result_tx.send(Ok(GenerationStats {
                generated_tokens: response_text.split_whitespace().count(),
                prompt_tokens: 0,
                elapsed: Duration::from_millis(1),
                tool_calls: vec![],
                generated_text: response_text.to_string(),
            }));
        }
    });
    PipelineWorker::from_sender(tx)
}

/// Spawn a mock that returns a single tool call instead of text.
fn mock_tool_call_worker(tool_name: &'static str, args_json: &'static str) -> PipelineWorker {
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(256);
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            drop(msg.token_tx);
            let tc = ToolCall {
                name: tool_name.to_string(),
                arguments: serde_json::from_str(args_json).unwrap_or_default(),
            };
            let _ = msg.result_tx.send(Ok(GenerationStats {
                generated_tokens: 1,
                prompt_tokens: 0,
                elapsed: Duration::from_millis(1),
                tool_calls: vec![tc],
                generated_text: String::new(),
            }));
        }
    });
    PipelineWorker::from_sender(tx)
}

fn test_state(response_text: &'static str) -> AppState {
    let mut workers = std::collections::HashMap::new();
    workers.insert("test-model".to_string(), mock_worker(response_text));
    AppState {
        workers,
        default_model: "test-model".to_string(),
    }
}

fn json_request(uri: &str, method: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builder")
}

async fn body_json(body: axum::body::Body) -> Value {
    let bytes = body.collect().await.expect("collect body").to_bytes();
    serde_json::from_slice(&bytes).expect("parse JSON body")
}

// ── /v1/models ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn models_returns_configured_model_id() {
    let app = build_router(test_state("irrelevant"));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await;
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "test-model");
    assert_eq!(json["data"][0]["object"], "model");
}

// ── /v1/chat/completions — non-streaming ─────────────────────────────────────

#[tokio::test]
async fn chat_non_streaming_returns_assistant_message() {
    let app = build_router(test_state("Hello from the assistant"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}]
            }),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await;
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "test-model");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert!(
        content.contains("Hello"),
        "content should contain mock tokens, got: {content}"
    );
}

#[tokio::test]
async fn chat_non_streaming_usage_reflects_token_count() {
    let app = build_router(test_state("one two three"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "count"}]
            }),
        ))
        .await
        .expect("response");

    let json = body_json(response.into_body()).await;
    // The mock reports generated_tokens = number of whitespace-separated words.
    assert_eq!(json["usage"]["completion_tokens"], 3);
}

#[tokio::test]
async fn chat_empty_messages_returns_400() {
    let app = build_router(test_state("irrelevant"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({ "model": "test-model", "messages": [] }),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let json = body_json(response.into_body()).await;
    assert!(json["error"]["message"].as_str().is_some());
}

#[tokio::test]
async fn chat_multi_turn_conversation_accepted() {
    let app = build_router(test_state("I remember our chat"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ]
            }),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
}

#[tokio::test]
async fn chat_max_tokens_field_accepted() {
    let app = build_router(test_state("short reply"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 64
            }),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
}

// ── Tool calls ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tool_call_response_has_correct_structure() {
    let mut workers = std::collections::HashMap::new();
    workers.insert(
        "test-model".to_string(),
        mock_tool_call_worker("get_weather", r#"{"location": "Paris", "unit": "celsius"}"#),
    );
    let app = build_router(AppState {
        workers,
        default_model: "test-model".to_string(),
    });

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                }]
            }),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await;
    assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");

    let tool_calls = &json["choices"][0]["message"]["tool_calls"];
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");

    let args: Value = serde_json::from_str(
        tool_calls[0]["function"]["arguments"]
            .as_str()
            .expect("arguments string"),
    )
    .expect("args JSON");
    assert_eq!(args["location"], "Paris");
}

// ── /v1/chat/completions — streaming ─────────────────────────────────────────

#[tokio::test]
async fn chat_streaming_returns_sse_content_type() {
    let app = build_router(test_state("streamed response"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": true
            }),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);

    let ct = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(
        ct.contains("text/event-stream"),
        "expected SSE content-type, got: {ct}"
    );
}

#[tokio::test]
async fn chat_streaming_body_contains_done_sentinel() {
    let app = build_router(test_state("hello world"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": true
            }),
        ))
        .await
        .expect("response");

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let body = std::str::from_utf8(&bytes).expect("utf8");

    assert!(body.contains("[DONE]"), "SSE stream must end with [DONE]");
    assert!(
        body.contains("chat.completion.chunk"),
        "SSE stream must contain chunk objects"
    );
}

#[tokio::test]
async fn chat_streaming_first_chunk_has_role() {
    let app = build_router(test_state("response text"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": true
            }),
        ))
        .await
        .expect("response");

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let body = std::str::from_utf8(&bytes).expect("utf8");

    // Extract the first `data:` line and parse it.
    let first_data = body
        .lines()
        .find(|l| l.starts_with("data:") && !l.contains("[DONE]"))
        .expect("at least one data line");

    let chunk: Value =
        serde_json::from_str(first_data.trim_start_matches("data:").trim()).expect("chunk JSON");

    assert_eq!(
        chunk["choices"][0]["delta"]["role"], "assistant",
        "first chunk must establish role"
    );
}

// ── Concurrent / batching behaviour ──────────────────────────────────────────

/// Send N non-streaming requests simultaneously and assert every one
/// returns 200 with an assistant message.  This validates that the HTTP layer
/// and mock worker correctly handle multiple in-flight requests — the same
/// contract the continuous-batch worker upholds for real model requests.
#[tokio::test]
async fn concurrent_non_streaming_requests_all_complete() {
    const N: usize = 8;

    // axum::Router is cheaply clone-able — each `oneshot` call consumes a clone.
    let router = Arc::new(build_router(test_state("parallel answer")));

    let handles: Vec<_> = (0..N)
        .map(|i| {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let app = (*router).clone();
                let resp = app
                    .oneshot(json_request(
                        "/v1/chat/completions",
                        "POST",
                        json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": format!("msg {i}")}]
                        }),
                    ))
                    .await
                    .expect("response");

                let status = resp.status();
                let json = body_json(resp.into_body()).await;
                (status, json)
            })
        })
        .collect();

    for handle in handles {
        let (status, json) = handle.await.expect("task");
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        assert!(
            content.contains("parallel"),
            "unexpected content: {content}"
        );
    }
}

/// Same as above but with streaming.  Every SSE stream must terminate with
/// `[DONE]` and carry at least one `chat.completion.chunk` event, even when
/// multiple streams are open at the same time.
#[tokio::test]
async fn concurrent_streaming_requests_all_complete() {
    const N: usize = 8;

    let router = Arc::new(build_router(test_state("streamed token")));

    let handles: Vec<_> = (0..N)
        .map(|i| {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let app = (*router).clone();
                let resp = app
                    .oneshot(json_request(
                        "/v1/chat/completions",
                        "POST",
                        json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": format!("stream {i}")}],
                            "stream": true
                        }),
                    ))
                    .await
                    .expect("response");

                assert_eq!(resp.status(), StatusCode::OK);

                let bytes = resp
                    .into_body()
                    .collect()
                    .await
                    .expect("collect")
                    .to_bytes();

                std::str::from_utf8(&bytes).expect("utf8").to_string()
            })
        })
        .collect();

    for handle in handles {
        let body = handle.await.expect("task");
        assert!(body.contains("[DONE]"), "stream must end with [DONE]");
        assert!(
            body.contains("chat.completion.chunk"),
            "stream must contain chunk objects"
        );
    }
}

/// Verify that a mix of streaming and non-streaming requests in-flight at the
/// same time all receive correct responses — this mirrors real usage where
/// different clients use different modes simultaneously.
#[tokio::test]
async fn mixed_streaming_and_non_streaming_concurrent() {
    let router = Arc::new(build_router(test_state("mixed result")));

    let non_stream_handle = {
        let router = Arc::clone(&router);
        tokio::spawn(async move {
            (*router)
                .clone()
                .oneshot(json_request(
                    "/v1/chat/completions",
                    "POST",
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "blocking"}]
                    }),
                ))
                .await
                .expect("response")
        })
    };

    let stream_handle = {
        let router = Arc::clone(&router);
        tokio::spawn(async move {
            (*router)
                .clone()
                .oneshot(json_request(
                    "/v1/chat/completions",
                    "POST",
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "streaming"}],
                        "stream": true
                    }),
                ))
                .await
                .expect("response")
        })
    };

    let (non_stream_resp, stream_resp) = tokio::join!(non_stream_handle, stream_handle);

    let ns = non_stream_resp.expect("non-stream task");
    assert_eq!(ns.status(), StatusCode::OK);
    let json = body_json(ns.into_body()).await;
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");

    let ss = stream_resp.expect("stream task");
    assert_eq!(ss.status(), StatusCode::OK);
    let bytes = ss.into_body().collect().await.expect("collect").to_bytes();
    let body = std::str::from_utf8(&bytes).expect("utf8");
    assert!(body.contains("[DONE]"));
}

/// The worker channel has capacity 256.  Enqueuing more requests than the old
/// capacity (4) must never return 503 — all requests must eventually complete.
#[tokio::test]
async fn large_burst_of_requests_does_not_return_503() {
    const N: usize = 32;

    let router = Arc::new(build_router(test_state("burst ok")));

    let handles: Vec<_> = (0..N)
        .map(|_| {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                (*router)
                    .clone()
                    .oneshot(json_request(
                        "/v1/chat/completions",
                        "POST",
                        json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "burst"}]
                        }),
                    ))
                    .await
                    .expect("response")
                    .status()
            })
        })
        .collect();

    for handle in handles {
        let status = handle.await.expect("task");
        assert_ne!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "no request in a burst of {N} should return 503"
        );
        assert_eq!(status, StatusCode::OK);
    }
}
