use crate::common;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::{collections::HashMap, time::Duration};
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;
use tur::{
    Result, TurError,
    backend::{
        chat_template::Role,
        memory::DeviceBudget,
        pipeline::{GenerationRequest, PipelineConfig},
    },
    server::{
        AppState, build_router,
        worker::{PipelineWorker, StreamConfig, WorkerConfig, WorkerMsg},
    },
};

fn worker(fixture: &common::Checkpoint, batch: usize) -> Result<PipelineWorker> {
    PipelineWorker::spawn(WorkerConfig {
        factory: fixture.factory(),
        pipeline: PipelineConfig::new(batch, 2),
        budget: DeviceBudget::new(None),
        request_timeout: None,
        stream: StreamConfig::default(),
    })
}

fn request(body: &Value) -> Result<Request<Body>> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .map_err(|error| TurError::Other(error.to_string()))
}

async fn json_body(response: axum::response::Response) -> Result<Value> {
    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .map_err(|error| TurError::Other(error.to_string()))?
        .to_bytes();
    Ok(serde_json::from_slice(&body)?)
}

#[tokio::test]
async fn test_real_worker_serves_concurrent_requests_with_queued_admission() -> Result<()> {
    let fixture = common::Checkpoint::new()?;
    let state = AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("tiny".into(), worker(&fixture, 1)?)]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "tiny".into(),
    };
    let router = build_router(state);
    let a = router.clone().oneshot(request(
        &json!({"messages":[{"role":"user","content":"t10 t11 t12 t13 t14"}],"max_tokens":3}),
    )?);
    let b = router.oneshot(request(
        &json!({"model":"tiny","messages":[{"role":"user","content":"t20"}],"max_tokens":2}),
    )?);
    let (a, b) = tokio::join!(a, b);
    let a = json_body(a.map_err(|error| TurError::Other(error.to_string()))?).await?;
    let b = json_body(b.map_err(|error| TurError::Other(error.to_string()))?).await?;
    assert_eq!(a["choices"][0]["message"]["content"], "t5 t5 t5");
    assert_eq!(a["choices"][0]["finish_reason"], "length");
    assert_eq!(a["usage"]["prompt_tokens"], 5);
    assert_eq!(a["usage"]["completion_tokens"], 3);
    assert_eq!(b["choices"][0]["message"]["content"], "t5 t5");
    assert_eq!(b["usage"]["prompt_tokens"], 1);
    Ok(())
}

#[tokio::test]
async fn test_real_stream_stops_at_tokenizer_eos_and_reports_usage() -> Result<()> {
    let fixture = common::Checkpoint::new()?;
    let mut config: Value = serde_json::from_str(&std::fs::read_to_string(
        fixture.0.join("tokenizer_config.json"),
    )?)?;
    config["eos_token"] = json!("t5");
    std::fs::write(fixture.0.join("tokenizer_config.json"), config.to_string())?;
    let state = AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("tiny".into(), worker(&fixture, 2)?)]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "tiny".into(),
    };
    let response = build_router(state).oneshot(request(&json!({"messages":[{"role":"user","content":"t10 t11 t12"}],"stream":true,"max_tokens":20}))?).await
        .map_err(|error| TurError::Other(error.to_string()))?;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|error| TurError::Other(error.to_string()))?
        .to_bytes();
    let text = std::str::from_utf8(&bytes).map_err(|error| TurError::Other(error.to_string()))?;
    assert!(text.contains("[DONE]"));
    let last = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .rfind(|data| *data != "[DONE]")
        .ok_or_else(|| TurError::Other("missing final SSE chunk".into()))?;
    let last: Value = serde_json::from_str(last)?;
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["usage"]["completion_tokens"], 1);
    assert_eq!(last["usage"]["prompt_tokens"], 3);
    assert!(!text.contains("t5"));
    Ok(())
}

#[tokio::test]
async fn test_worker_reclaims_cancelled_request_before_admitting_next() -> Result<()> {
    let fixture = common::Checkpoint::new()?;
    let worker = worker(&fixture, 1)?;
    let (tx, mut rx) = worker.stream_channel();
    let (done_tx, done_rx) = oneshot::channel();
    worker
        .send(WorkerMsg {
            request: GenerationRequest::new("t10".into(), 400),
            token_tx: Some(tx),
            result_tx: done_tx,
        })
        .await?;
    let token = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .map_err(|error| TurError::Other(error.to_string()))?;
    assert!(token.is_some());
    drop(rx);
    drop(done_rx);
    let (done_tx, done_rx) = oneshot::channel();
    worker
        .send(WorkerMsg {
            request: GenerationRequest::new("t20".into(), 2),
            token_tx: None,
            result_tx: done_tx,
        })
        .await?;
    let stats = tokio::time::timeout(Duration::from_secs(10), done_rx)
        .await
        .map_err(|error| TurError::Other(error.to_string()))?
        .map_err(|error| TurError::Other(error.to_string()))??;
    assert_eq!(stats.generated_text, "t5 t5");
    Ok(())
}

#[tokio::test]
async fn test_stalled_stream_is_cancelled_with_bounded_buffering() -> Result<()> {
    let fixture = common::Checkpoint::new()?;
    let worker = PipelineWorker::spawn(WorkerConfig {
        factory: fixture.factory(),
        pipeline: PipelineConfig::new(1, 2),
        budget: DeviceBudget::new(None),
        request_timeout: None,
        stream: StreamConfig {
            channel_capacity: 2,
            max_buffered_bytes: 32,
        },
    })?;
    // The client never reads: two messages fill its channel, then "t5 " fragments
    // accumulate until the bound cancels the request.
    let (tx, rx) = worker.stream_channel();
    let (done_tx, done_rx) = oneshot::channel();
    worker
        .send(WorkerMsg {
            request: GenerationRequest::new("t10".into(), 400),
            token_tx: Some(tx),
            result_tx: done_tx,
        })
        .await?;
    let outcome = tokio::time::timeout(Duration::from_secs(10), done_rx)
        .await
        .map_err(|error| TurError::Other(error.to_string()))?
        .map_err(|error| TurError::Other(error.to_string()))?;
    assert!(
        matches!(outcome, Err(TurError::SlowClient(_))),
        "{outcome:?}"
    );
    assert_eq!(rx.len(), 2);
    // The cancelled request's state was released: the single running slot serves the next.
    let (done_tx, done_rx) = oneshot::channel();
    worker
        .send(WorkerMsg {
            request: GenerationRequest::new("t20".into(), 2),
            token_tx: None,
            result_tx: done_tx,
        })
        .await?;
    let stats = tokio::time::timeout(Duration::from_secs(10), done_rx)
        .await
        .map_err(|error| TurError::Other(error.to_string()))?
        .map_err(|error| TurError::Other(error.to_string()))??;
    assert_eq!(stats.generated_text, "t5 t5");
    Ok(())
}

#[tokio::test]
async fn test_http_preserves_limits_sampling_thinking_and_tool_history() -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(1);
    let state = AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("tiny".into(), PipelineWorker::from_sender(tx))]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "tiny".into(),
    };
    let handle = tokio::spawn(async move {
        let msg = rx
            .recv()
            .await
            .ok_or_else(|| TurError::Other("missing request".into()))?;
        assert_eq!(msg.request.max_tokens, 64);
        assert_eq!(
            msg.request
                .messages
                .iter()
                .map(|message| (message.role, message.content.as_text().unwrap_or_default()))
                .collect::<Vec<_>>(),
            [
                (Role::System, "You are helpful."),
                (Role::User, "query"),
                (Role::Assistant, ""),
                (Role::Tool, "value"),
                (Role::User, "continue"),
            ]
        );
        assert_eq!(msg.request.sampling.temperature, Some(0.0));
        assert_eq!(msg.request.sampling.top_k, Some(1));
        assert_eq!(msg.request.sampling.top_p, Some(1.0));
        assert_eq!(msg.request.render.enable_thinking, Some(false));
        assert_eq!(msg.request.render.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(msg.request.render.preserve_thinking, Some(true));
        assert_eq!(msg.request.seed, 7);
        assert_eq!(
            msg.request.messages[2].reasoning_content.as_deref(),
            Some("reason")
        );
        assert_eq!(msg.request.messages[2].tool_calls[0].arguments["id"], "123");
        drop(msg.token_tx);
        drop(
            msg.result_tx
                .send(Ok(tur::backend::pipeline::GenerationStats {
                    generated_tokens: 0,
                    cached_tokens: 0,
                    prompt_tokens: 0,
                    elapsed: Duration::ZERO,
                    first_token: Duration::ZERO,
                    preemptions: 0,
                    draft_tokens: 0,
                    accepted_draft_tokens: 0,
                    generated_text: String::new(),
                    reasoning_text: String::new(),
                    tool_calls: Vec::new(),
                    finish_reason: "stop",
                })),
        );
        Ok::<(), TurError>(())
    });
    let response = build_router(state).oneshot(request(&json!({
        "messages":[{"role":"system","content":"You are helpful."}, {"role":"user","content":"query"}, {"role":"assistant","reasoning_content":"reason","tool_calls":[{
            "id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"id\":\"123\"}"}
        }]}, {"role":"tool","tool_call_id":"call_1","content":"value"}, {"role":"user","content":"continue"}],
        "max_tokens":64,"temperature":0.0,"top_k":1,"top_p":1.0,"enable_thinking":false,"reasoning_effort":"low","preserve_thinking":true,"seed":7
    }))?).await.map_err(|error| TurError::Other(error.to_string()))?;
    assert_eq!(response.status(), StatusCode::OK);
    handle
        .await
        .map_err(|error| TurError::Other(error.to_string()))??;
    Ok(())
}

#[tokio::test]
async fn test_http_rejects_unsupported_media_and_invalid_controls() -> Result<()> {
    let (tx, _rx) = mpsc::channel(1);
    let router = build_router(AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("tiny".into(), PipelineWorker::from_sender(tx))]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "tiny".into(),
    });
    for body in [
        json!({"messages":[{"role":"unknown","content":"x"}]}),
        json!({"messages":[{"role":"user","content":"x"}],"top_k":0}),
        json!({"messages":[{"role":"user","content":"x"}],"temperature":-1}),
        json!({"messages":[{"role":"user","content":"x"}],"tool_choice":"required"}),
    ] {
        let response = router
            .clone()
            .oneshot(request(&body)?)
            .await
            .map_err(|error| TurError::Other(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    let response = router.oneshot(request(&json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}))?).await
        .map_err(|error| TurError::Other(error.to_string()))?;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    Ok(())
}

#[tokio::test]
async fn test_http_maps_admission_and_deadline_failures_and_streams_error_codes() -> Result<()> {
    for (error, status, code) in [
        (
            TurError::QueueFull("waiting tokens".into()),
            StatusCode::TOO_MANY_REQUESTS,
            "queue_full",
        ),
        (
            TurError::DeadlineExceeded,
            StatusCode::GATEWAY_TIMEOUT,
            "deadline_exceeded",
        ),
        (
            TurError::MemoryAllocation("state pool".into()),
            StatusCode::SERVICE_UNAVAILABLE,
            "insufficient_memory",
        ),
        (
            TurError::WorkerUnavailable,
            StatusCode::SERVICE_UNAVAILABLE,
            "worker_unavailable",
        ),
    ] {
        let (tx, mut rx) = mpsc::channel::<WorkerMsg>(1);
        let router = build_router(AppState {
            status: tur::server::ServerStatus::shared(),
            workers: HashMap::from([("tiny".into(), PipelineWorker::from_sender(tx))]),
            transcribers: HashMap::new(),
            default_transcriber: None,
            default_model: "tiny".into(),
        });
        let backend = tokio::spawn(async move {
            if let Some(msg) = rx.recv().await {
                drop(msg.token_tx);
                drop(msg.result_tx.send(Err(error)));
            }
        });
        let response = router
            .oneshot(request(
                &json!({"messages":[{"role":"user","content":"t10"}]}),
            )?)
            .await
            .map_err(|error| TurError::Other(error.to_string()))?;
        assert_eq!(response.status(), status);
        let bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|error| TurError::Other(error.to_string()))?
            .to_bytes();
        let body: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(body["error"]["code"], code);
        backend
            .await
            .map_err(|error| TurError::Other(error.to_string()))?;
    }
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(1);
    let router = build_router(AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("tiny".into(), PipelineWorker::from_sender(tx))]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "tiny".into(),
    });
    let backend = tokio::spawn(async move {
        if let Some(msg) = rx.recv().await {
            drop(msg.token_tx);
            drop(msg.result_tx.send(Err(TurError::DeadlineExceeded)));
        }
    });
    let response = router
        .oneshot(request(
            &json!({"stream":true,"messages":[{"role":"user","content":"t10"}]}),
        )?)
        .await
        .map_err(|error| TurError::Other(error.to_string()))?;
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|error| TurError::Other(error.to_string()))?
        .to_bytes();
    let text = std::str::from_utf8(&bytes).map_err(|error| TurError::Other(error.to_string()))?;
    assert!(text.contains("deadline_exceeded"));
    assert!(text.contains("[DONE]"));
    backend
        .await
        .map_err(|error| TurError::Other(error.to_string()))?;
    Ok(())
}
