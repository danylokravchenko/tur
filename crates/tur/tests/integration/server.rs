//! Integration tests for the OpenAI-compatible HTTP server.
//!
//! All tests use a mock `PipelineWorker` (via `PipelineWorker::from_sender`)
//! so no model weights are needed.  The mock task simulates the contract that
//! the real continuous-batching worker upholds:
//!   1. Drain `WorkerMsg`s from the channel concurrently.
//!   2. Stream tokens via `token_tx` for each request independently.
//!   3. Close `token_tx` and send `GenerationStats` via `result_tx` when done.
//!
//! The axum router is exercised via `tower::ServiceExt::oneshot` (no TCP
//! socket needed) for single-request tests, and via a shared `axum::Router`
//! clone for concurrent-request tests.
use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<T = ()> = Result<T, TestError>;

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
            if let Some(token_tx) = &msg.token_tx {
                for fragment in response_text.split_inclusive(' ') {
                    // The adapter requires streamed text to be a prefix of the final text.
                    drop(
                        token_tx
                            .send(tur::server::worker::TextDelta::Content(
                                fragment.to_string(),
                            ))
                            .await,
                    );
                }
            }
            drop(msg.token_tx);
            // generated_text is used by the non-streaming path in blocking_response.
            drop(msg.result_tx.send(Ok(GenerationStats {
                finish_reason: "stop",
                generated_tokens: response_text.split_whitespace().count(),
                prompt_tokens: 0,
                cached_tokens: 0,
                elapsed: Duration::from_millis(1),
                first_token: Duration::from_millis(1),
                preemptions: 0,
                draft_tokens: 0,
                accepted_draft_tokens: 0,
                tool_calls: vec![],
                generated_text: response_text.to_string(),
                reasoning_text: String::new(),
            })));
        }
    });
    PipelineWorker::from_sender(tx)
}

/// Spawn a mock that returns a single tool call instead of text.
fn mock_tool_call_worker(
    tool_name: &'static str,
    args_json: &'static str,
) -> TestResult<PipelineWorker> {
    let arguments: Value = serde_json::from_str(args_json)?;
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(256);
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            drop(msg.token_tx);
            let tc = ToolCall {
                name: tool_name.to_string(),
                arguments: arguments.clone(),
            };
            drop(msg.result_tx.send(Ok(GenerationStats {
                finish_reason: "stop",
                generated_tokens: 1,
                prompt_tokens: 0,
                cached_tokens: 0,
                elapsed: Duration::from_millis(1),
                first_token: Duration::from_millis(1),
                preemptions: 0,
                draft_tokens: 0,
                accepted_draft_tokens: 0,
                tool_calls: vec![tc],
                generated_text: String::new(),
                reasoning_text: String::new(),
            })));
        }
    });
    Ok(PipelineWorker::from_sender(tx))
}

fn test_state(response_text: &'static str) -> AppState {
    let mut workers = HashMap::new();
    workers.insert("test-model".to_string(), mock_worker(response_text));
    AppState {
        workers,
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "test-model".to_string(),
    }
}

fn json_request(uri: &str, method: &str, body: &Value) -> TestResult<Request<Body>> {
    Ok(Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))?)
}

async fn body_json(body: Body) -> TestResult<Value> {
    let bytes = body.collect().await?.to_bytes();
    Ok(serde_json::from_slice(&bytes)?)
}

// ── /v1/models ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_models_returns_configured_model_id() -> TestResult {
    let app = build_router(test_state("irrelevant"));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await?;
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "test-model");
    assert_eq!(json["data"][0]["object"], "model");
    Ok(())
}

// ── /v1/chat/completions — non-streaming ─────────────────────────────────────

#[tokio::test]
async fn test_chat_non_streaming_returns_assistant_message() -> TestResult {
    let app = build_router(test_state("Hello from the assistant"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            &json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}]
            }),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await?;
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "test-model");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");

    assert_eq!(
        json["choices"][0]["message"]["content"],
        "Hello from the assistant"
    );
    assert_eq!(json["usage"]["completion_tokens"], 4);
    Ok(())
}

#[tokio::test]
async fn test_chat_empty_messages_returns_400() -> TestResult {
    let app = build_router(test_state("irrelevant"));

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            &json!({ "model": "test-model", "messages": [] }),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let json = body_json(response.into_body()).await?;
    assert!(json["error"]["message"].as_str().is_some());
    Ok(())
}

// ── Tool calls ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tool_call_response_has_correct_structure() -> TestResult {
    let mut workers = HashMap::new();
    workers.insert(
        "test-model".to_string(),
        mock_tool_call_worker("get_weather", r#"{"location": "Paris", "unit": "celsius"}"#)?,
    );
    let app = build_router(AppState {
        workers,
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "test-model".to_string(),
    });

    let response = app
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            &json!({
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
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await?;
    assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");

    let tool_calls = &json["choices"][0]["message"]["tool_calls"];
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");

    let args: Value = serde_json::from_str(
        tool_calls[0]["function"]["arguments"]
            .as_str()
            .ok_or("arguments string")?,
    )?;
    assert_eq!(args["location"], "Paris");
    Ok(())
}

// ── /v1/chat/completions — streaming ─────────────────────────────────────────

async fn assert_text_stream(response: axum::response::Response, expected: &str) -> TestResult {
    assert_eq!(response.status(), StatusCode::OK);

    let ct = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert_eq!(ct.split(';').next(), Some("text/event-stream"));
    let bytes = response.into_body().collect().await?.to_bytes();
    let body = std::str::from_utf8(&bytes)?;

    let mut events: Vec<&str> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect();
    assert_eq!(events.pop(), Some("[DONE]"), "terminal SSE event");
    let chunks: Vec<Value> = events
        .into_iter()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<_>>()?;
    let first = chunks.first().ok_or("missing first chunk")?;
    assert_eq!(
        first["choices"][0]["delta"]["role"], "assistant",
        "first chunk must establish role"
    );
    for chunk in &chunks {
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["model"], "test-model");
        assert_eq!(chunk["id"], first["id"]);
    }
    let text: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, expected);
    let last = chunks.last().ok_or("missing final chunk")?;
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        last["usage"]["completion_tokens"],
        expected.split_whitespace().count()
    );
    Ok(())
}

#[tokio::test]
async fn test_chat_streaming_preserves_sse_event_order_and_text() -> TestResult {
    let response = build_router(test_state("hello world"))
        .oneshot(json_request(
            "/v1/chat/completions",
            "POST",
            &json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": true
            }),
        )?)
        .await?;
    assert_text_stream(response, "hello world").await?;
    Ok(())
}

// ── Concurrent / batching behaviour ──────────────────────────────────────────

/// Send N non-streaming requests simultaneously and assert every one
/// returns 200 with an assistant message. The burst exceeds the former intake
/// capacity of four; real-worker admission is covered separately in `serving`.
#[tokio::test]
async fn test_concurrent_non_streaming_requests_all_complete() -> TestResult {
    const N: usize = 32;

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
                        &json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": format!("msg {i}")}]
                        }),
                    )?)
                    .await?;

                let status = resp.status();
                let json = body_json(resp.into_body()).await?;
                Ok::<_, TestError>((status, json))
            })
        })
        .collect();

    for handle in handles {
        let (status, json) = handle.await??;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "parallel answer");
    }
    Ok(())
}

/// Same as above but with streaming.  Every SSE stream must terminate with
/// `[DONE]` and carry at least one `chat.completion.chunk` event, even when
/// multiple streams are open at the same time.
#[tokio::test]
async fn test_concurrent_streaming_requests_all_complete() -> TestResult {
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
                        &json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": format!("stream {i}")}],
                            "stream": true
                        }),
                    )?)
                    .await?;

                assert_text_stream(resp, "streamed token").await
            })
        })
        .collect();

    for handle in handles {
        handle.await??;
    }
    Ok(())
}

/// Verify that a mix of streaming and non-streaming requests in-flight at the
/// same time all receive correct responses — this mirrors real usage where
/// different clients use different modes simultaneously.
#[tokio::test]
async fn test_mixed_streaming_and_non_streaming_concurrent() -> TestResult {
    let router = Arc::new(build_router(test_state("mixed result")));

    let non_stream_handle = {
        let router = Arc::clone(&router);
        tokio::spawn(async move {
            (*router)
                .clone()
                .oneshot(json_request(
                    "/v1/chat/completions",
                    "POST",
                    &json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "blocking"}]
                    }),
                )?)
                .await
                .map_err(TestError::from)
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
                    &json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "streaming"}],
                        "stream": true
                    }),
                )?)
                .await
                .map_err(TestError::from)
        })
    };

    let (non_stream_resp, stream_resp) = tokio::join!(non_stream_handle, stream_handle);

    let ns = non_stream_resp??;
    assert_eq!(ns.status(), StatusCode::OK);
    let json = body_json(ns.into_body()).await?;
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "mixed result");

    let ss = stream_resp??;
    assert_text_stream(ss, "mixed result").await?;
    Ok(())
}
