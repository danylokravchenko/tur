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
    backend::{
        pipeline::{GenerationStats, TokenLogprob},
        structured::StructuredOutput,
        tools::ToolCall,
    },
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
                logprobs: Vec::new(),
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
                logprobs: Vec::new(),
            })));
        }
    });
    Ok(PipelineWorker::from_sender(tx))
}

fn test_state(response_text: &'static str) -> AppState {
    let mut workers = HashMap::new();
    workers.insert("test-model".to_string(), mock_worker(response_text));
    AppState {
        status: tur::server::ServerStatus::shared(),
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
        status: tur::server::ServerStatus::shared(),
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

// ── /v1/chat/completions — structured output ─────────────────────────────────

/// What the worker saw of a request: forced output, log-probability count and tool count.
type Seen = (Option<StructuredOutput>, Option<usize>, usize);

/// A router whose worker answers with `stats` and forwards each request's forced output and
/// log-probability choice.
fn structured_router(
    stats: fn() -> GenerationStats,
) -> (axum::Router, mpsc::UnboundedReceiver<Seen>) {
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(8);
    let (seen_tx, seen_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            drop(seen_tx.send((
                msg.request.structured.clone(),
                msg.request.sampling.logprobs,
                msg.request.tools.len(),
            )));
            let answer = stats();
            if let Some(token_tx) = &msg.token_tx {
                let text = tur::server::worker::TextDelta::Content(answer.generated_text.clone());
                drop(token_tx.send(text).await);
            }
            drop(msg.token_tx);
            drop(msg.result_tx.send(Ok(answer)));
        }
    });
    let router = build_router(AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("m".to_string(), PipelineWorker::from_sender(tx))]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "m".to_string(),
    });
    (router, seen_rx)
}

fn plain_stats() -> GenerationStats {
    GenerationStats {
        finish_reason: "stop",
        generated_tokens: 1,
        prompt_tokens: 0,
        cached_tokens: 0,
        elapsed: Duration::from_millis(1),
        first_token: Duration::from_millis(1),
        preemptions: 0,
        draft_tokens: 0,
        accepted_draft_tokens: 0,
        tool_calls: vec![],
        generated_text: "{}".to_string(),
        reasoning_text: String::new(),
        logprobs: Vec::new(),
    }
}

fn forced_call_stats() -> GenerationStats {
    GenerationStats {
        tool_calls: vec![ToolCall {
            name: "f".to_string(),
            arguments: json!({}),
        }],
        generated_text: "<function=f></function>".to_string(),
        ..plain_stats()
    }
}

fn logprob_stats() -> GenerationStats {
    let token = |token: &str, logprob, top| TokenLogprob {
        token: token.to_string(),
        bytes: token.as_bytes().to_vec(),
        logprob,
        top,
    };
    GenerationStats {
        logprobs: vec![token("{", -0.5, vec![token("{", -0.5, vec![])])],
        ..plain_stats()
    }
}

fn chat_body(extra: &Value) -> Value {
    let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    body
}

fn tool(name: &str) -> Value {
    json!({"type": "function", "function": {"name": name}})
}

#[tokio::test]
async fn test_response_format_and_tool_choice_reach_the_worker_as_forced_output() -> TestResult {
    let schema = json!({"type": "object"});
    for (extra, forced, tools) in [
        (json!({"response_format": {"type": "text"}}), None, 0),
        (
            json!({"response_format": {"type": "json_object"}}),
            Some(StructuredOutput::JsonObject),
            0,
        ),
        (
            json!({"response_format": {"type": "json_schema",
                "json_schema": {"name": "n", "schema": schema}}}),
            Some(StructuredOutput::JsonSchema(schema.clone())),
            0,
        ),
        (
            json!({"tools": [tool("f")], "tool_choice": "auto"}),
            None,
            1,
        ),
        // `none` hides the tools from the template, so a response format may join it.
        (
            json!({"tools": [tool("f")], "tool_choice": "none",
                "response_format": {"type": "json_object"}}),
            Some(StructuredOutput::JsonObject),
            0,
        ),
        (
            json!({"tools": [tool("f")], "tool_choice": "required"}),
            Some(StructuredOutput::Tool(None)),
            1,
        ),
        (
            json!({"tools": [tool("f"), tool("g")],
                "tool_choice": {"type": "function", "function": {"name": "g"}}}),
            Some(StructuredOutput::Tool(Some("g".to_string()))),
            2,
        ),
    ] {
        let (router, mut seen) = structured_router(plain_stats);
        let request = json_request("/v1/chat/completions", "POST", &chat_body(&extra))?;
        let response = router.oneshot(request).await?;
        assert_eq!(response.status(), StatusCode::OK, "{extra}");
        let (structured, logprobs, offered) = seen.recv().await.ok_or("no request")?;
        assert_eq!(structured, forced, "{extra}");
        assert_eq!(logprobs, None, "{extra}");
        assert_eq!(offered, tools, "{extra}");
    }
    Ok(())
}

#[tokio::test]
async fn test_unsatisfiable_structured_requests_return_400() -> TestResult {
    for extra in [
        json!({"tool_choice": "required"}),
        json!({"tool_choice": "sometimes", "tools": [tool("f")]}),
        json!({"tool_choice": {"type": "function", "function": {"name": "g"}},
            "tools": [tool("f")]}),
        json!({"tool_choice": {"type": "function"}, "tools": [tool("f")]}),
        json!({"response_format": {"type": "json_object"}, "tools": [tool("f")]}),
        json!({"top_logprobs": 2}),
        json!({"logprobs": true, "top_logprobs": 21}),
    ] {
        let (router, mut seen) = structured_router(plain_stats);
        let request = json_request("/v1/chat/completions", "POST", &chat_body(&extra))?;
        let response = router.oneshot(request).await?;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{extra}");
        assert!(seen.try_recv().is_err(), "{extra}");
    }
    Ok(())
}

async fn stream_events(router: axum::Router, extra: &Value) -> TestResult<Vec<Value>> {
    let mut body = chat_body(extra);
    body["stream"] = json!(true);
    let response = router
        .oneshot(json_request("/v1/chat/completions", "POST", &body)?)
        .await?;
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(String::from_utf8(bytes.to_vec())?
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?)
}

#[tokio::test]
async fn test_forced_calls_stream_only_the_parsed_call() -> TestResult {
    let extra = json!({"tools": [tool("f")], "tool_choice": "required"});
    let (router, _seen) = structured_router(forced_call_stats);
    let events = stream_events(router, &extra).await?;
    assert!(
        events
            .iter()
            .all(|event| event["choices"][0]["delta"]["content"].is_null()),
        "{events:?}"
    );
    let calls: Vec<&Value> = events
        .iter()
        .filter(|event| !event["choices"][0]["delta"]["tool_calls"].is_null())
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "f"
    );
    Ok(())
}

#[tokio::test]
async fn test_logprobs_follow_the_openai_shape_when_blocking_and_streaming() -> TestResult {
    let extra = json!({"logprobs": true, "top_logprobs": 1});
    let expected = json!({"content": [{
        "token": "{", "logprob": -0.5, "bytes": [123],
        "top_logprobs": [{"token": "{", "logprob": -0.5, "bytes": [123]}],
    }]});
    let (router, mut seen) = structured_router(logprob_stats);
    let request = json_request("/v1/chat/completions", "POST", &chat_body(&extra))?;
    let response = router.clone().oneshot(request).await?;
    let json = body_json(response.into_body()).await?;
    assert_eq!(json["choices"][0]["logprobs"], expected);
    assert_eq!(seen.recv().await.ok_or("no request")?.1, Some(1));

    let events = stream_events(router, &extra).await?;
    let carrying: Vec<&Value> = events
        .iter()
        .map(|event| &event["choices"][0]["logprobs"])
        .filter(|logprobs| !logprobs.is_null())
        .collect();
    assert_eq!(carrying, [&expected]);
    Ok(())
}
