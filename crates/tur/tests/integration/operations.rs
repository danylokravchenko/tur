//! What an operator relies on: stop sequences, context limits, health, metrics, diagnostics
//! and graceful shutdown, over the HTTP surface and a real worker where it matters.

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::mpsc,
};
use tower::ServiceExt;
use tur::{
    Result, TurError,
    backend::{memory::DeviceBudget, pipeline::PipelineConfig},
    server::{
        AppState, ServerStatus, build_router, serve_until,
        worker::{PipelineWorker, StreamConfig, WorkerConfig, WorkerMsg},
    },
};

use crate::common::Checkpoint;

const PROMPT: &str = "t10 t11 t12";

fn real_worker(fixture: &Checkpoint, context_tokens: Option<usize>) -> Result<PipelineWorker> {
    PipelineWorker::spawn(WorkerConfig {
        factory: fixture.factory(),
        pipeline: PipelineConfig {
            context_tokens,
            ..PipelineConfig::new(2, 2)
        },
        budget: DeviceBudget::new(None),
        request_timeout: None,
        stream: StreamConfig::default(),
    })
}

fn state(worker: PipelineWorker) -> AppState {
    AppState {
        status: ServerStatus::shared(),
        workers: HashMap::from([("tiny".into(), worker)]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "tiny".into(),
    }
}

fn error(error: impl std::fmt::Display) -> TurError {
    TurError::Other(error.to_string())
}

fn chat(extra: &Value) -> Result<Request<Body>> {
    let mut body = json!({"messages": [{"role": "user", "content": PROMPT}], "temperature": 0});
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .map_err(error)
}

async fn send(router: &Router, request: Request<Body>) -> Result<(StatusCode, String)> {
    let response = router.clone().oneshot(request).await.map_err(error)?;
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(error)?
        .to_bytes();
    Ok((status, String::from_utf8_lossy(&bytes).into_owned()))
}

async fn get(router: &Router, path: &str) -> Result<(StatusCode, String)> {
    send(
        router,
        Request::get(path).body(Body::empty()).map_err(error)?,
    )
    .await
}

async fn post(router: &Router, extra: &Value) -> Result<(StatusCode, Value)> {
    let (status, body) = send(router, chat(extra)?).await?;
    Ok((status, serde_json::from_str(&body)?))
}

#[tokio::test]
async fn test_stop_sequence_ends_generation_and_streams_match_the_final_text() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let router = build_router(state(real_worker(&fixture, None)?));
    // Greedy output is "t5 t5 t5 ...": this sequence is split across decoded tokens.
    let extra = json!({"max_tokens": 20, "stop": ["nothing", "5 t5"]});
    let (status, body) = post(&router, &extra).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "t");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body["usage"]["completion_tokens"].as_u64() < Some(20));

    // A single string is accepted too, and a stream shows nothing of the sequence.
    let extra = json!({"max_tokens": 20, "stream": true, "stop": "5 t5"});
    let (status, text) = send(&router, chat(&extra)?).await?;
    assert_eq!(status, StatusCode::OK);
    let events: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let streamed: String = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(streamed, "t");
    let last = events.last().ok_or_else(|| error("no events"))?;
    assert_eq!(last["choices"][0]["finish_reason"], "stop");

    // A sequence the output never holds changes nothing.
    let (_, body) = post(&router, &json!({"max_tokens": 3, "stop": "absent"})).await?;
    assert_eq!(body["choices"][0]["message"]["content"], "t5 t5 t5");
    assert_eq!(body["choices"][0]["finish_reason"], "length");

    let (status, body) = post(&router, &json!({"stop": [""]})).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_request");
    Ok(())
}

#[tokio::test]
async fn test_configured_context_limits_requests_and_is_reported() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let router = build_router(state(real_worker(&fixture, Some(16))?));

    // The prompt's 3 tokens and 14 outputs use 16 positions, since the last is never consumed.
    let (status, body) = post(&router, &json!({"max_tokens": 14})).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["usage"]["completion_tokens"], 14);
    let (status, body) = post(&router, &json!({"max_tokens": 15})).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "context_length_exceeded");

    // Without an explicit limit the output fills what the context leaves.
    let (status, body) = post(&router, &json!({})).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["usage"]["completion_tokens"], 14);
    assert_eq!(body["choices"][0]["finish_reason"], "length");

    // A prompt too long by itself fails however the limit was chosen.
    let long = vec!["t10"; 17].join(" ");
    let extra = json!({"messages": [{"role": "user", "content": long}]});
    let (status, body) = post(&router, &extra).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "context_length_exceeded");

    let (_, text) = get(&router, "/diagnostics").await?;
    let diagnostics: Value = serde_json::from_str(&text)?;
    let context = &diagnostics["models"]["tiny"]["checkpoint"]["context"];
    assert_eq!(context["limit_tokens"], 16);
    assert!(context["full_contexts"].as_u64() >= Some(2));
    let (_, text) = get(&router, "/metrics").await?;
    assert!(
        text.contains("tur_context_tokens{model=\"tiny\"} 16\n"),
        "{text}"
    );
    Ok(())
}

#[test]
fn test_context_beyond_the_model_or_the_device_budget_fails_before_serving() -> Result<()> {
    let fixture = Checkpoint::new()?;
    assert!(matches!(
        real_worker(&fixture, Some(1 << 40)),
        Err(TurError::ContextExceeded(_))
    ));
    assert!(real_worker(&fixture, Some(0)).is_err());
    // A budget with room for the model but not for this context names what would fit.
    let plan = |context, limit| {
        PipelineWorker::spawn(WorkerConfig {
            factory: fixture.factory(),
            pipeline: PipelineConfig {
                context_tokens: Some(context),
                ..PipelineConfig::new(2, 2)
            },
            budget: DeviceBudget::new(limit),
            request_timeout: None,
            stream: StreamConfig::default(),
        })
    };
    let small = plan(16, None)?;
    let needed = small
        .report()
        .ok_or_else(|| error("no report"))?
        .plan
        .resident();
    drop(small);
    let outcome = plan(512, Some(needed + needed / 5));
    let Err(TurError::MemoryAllocation(message)) = outcome else {
        return Err(error(format!("{:?}", outcome.err())));
    };
    assert!(message.contains("largest context that fits"), "{message}");
    Ok(())
}

#[test]
fn test_configured_context_shrinks_the_reserved_memory() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let resident = |context| -> Result<(usize, usize)> {
        let worker = real_worker(&fixture, context)?;
        let plan = worker.report().ok_or_else(|| error("no report"))?.plan;
        Ok((plan.scratch_peak, plan.state_pool))
    };
    let (unbounded_scratch, unbounded_pool) = resident(None)?;
    let (scratch, pool) = resident(Some(64))?;
    assert!(scratch <= unbounded_scratch);
    assert!(pool < unbounded_pool);
    Ok(())
}

#[tokio::test]
async fn test_health_readiness_metrics_and_revision_report_a_serving_worker() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let app = state(real_worker(&fixture, None)?);
    let status = Arc::clone(&app.status);
    let router = build_router(app);

    assert_eq!(get(&router, "/health").await?.0, StatusCode::OK);
    assert_eq!(get(&router, "/ready").await?.0, StatusCode::OK);
    let (_, body) = post(&router, &json!({"max_tokens": 3})).await?;
    assert_eq!(body["usage"]["completion_tokens"], 3);

    let (_, text) = get(&router, "/v1/models").await?;
    let models: Value = serde_json::from_str(&text)?;
    assert_eq!(models["data"][0]["revision"], "tiny-adapter-fixture-v1");

    // Each pass publishes before the next message is taken; the reply precedes that.
    let mut text = String::new();
    for _ in 0..200 {
        text = get(&router, "/metrics").await?.1;
        if text.contains("tur_passes_total{model=\"tiny\"} 0\n") {
            tokio::time::sleep(Duration::from_millis(5)).await;
        } else {
            break;
        }
    }
    for line in [
        "tur_worker_up{model=\"tiny\"} 1\n",
        "tur_model_info{model=\"tiny\",source=\"local\",revision=\"tiny-adapter-fixture-v1\"} 1\n",
        "tur_requests_total{model=\"tiny\",outcome=\"ok\"} 1\n",
        "tur_generated_tokens_total{model=\"tiny\"} 3\n",
        "tur_request_duration_seconds_count{model=\"tiny\"} 1\n",
        "tur_http_requests_total{path=\"/v1/chat/completions\",status=\"200\"} 1\n",
    ] {
        assert!(text.contains(line), "missing {line} in {text}");
    }

    let (_, text) = get(&router, "/diagnostics").await?;
    assert!(!text.contains(PROMPT), "diagnostics exposed a prompt");
    let diagnostics: Value = serde_json::from_str(&text)?;
    assert_eq!(
        diagnostics["pid"].as_u64(),
        Some(u64::from(std::process::id()))
    );
    let model = &diagnostics["models"]["tiny"];
    assert_eq!(model["checkpoint"]["revision"], "tiny-adapter-fixture-v1");
    assert_eq!(model["checkpoint"]["source"], "local");
    assert!(model["checkpoint"]["memory_plan"]["resident_bytes"].as_u64() > Some(0));
    assert!(model["activity"]["pool"]["kv_pages"]["capacity"].as_u64() > Some(0));
    let latency = &model["activity"]["latency"]["request_seconds"];
    assert!(latency["count"].as_u64() > Some(0));
    assert_eq!(
        latency["bounds"].as_array().map(Vec::len),
        latency["buckets"].as_array().map(Vec::len)
    );

    // Draining fails readiness and new work, but not liveness or read-only endpoints.
    status.begin_drain();
    assert_eq!(get(&router, "/health").await?.0, StatusCode::OK);
    assert_eq!(
        get(&router, "/ready").await?.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(get(&router, "/metrics").await?.0, StatusCode::OK);
    let (code, body) = post(&router, &json!({})).await?;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "server_shutting_down");
    Ok(())
}

#[tokio::test]
async fn test_stopped_worker_fails_health_and_readiness() -> Result<()> {
    let (tx, rx) = mpsc::channel::<WorkerMsg>(1);
    let router = build_router(state(PipelineWorker::from_sender(tx)));
    assert_eq!(get(&router, "/health").await?.0, StatusCode::OK);
    drop(rx);
    let (status, text) = get(&router, "/health").await?;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        text.contains("worker_stopped") && text.contains("tiny"),
        "{text}"
    );
    assert_eq!(
        get(&router, "/ready").await?.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    let (_, text) = get(&router, "/metrics").await?;
    assert!(text.contains("tur_worker_up{model=\"tiny\"} 0\n"), "{text}");
    Ok(())
}

/// Send a chat request over a real connection; the response is read to its end later.
async fn open_request(addr: SocketAddr, extra: &Value) -> Result<TcpStream> {
    let mut stream = TcpStream::connect(addr).await?;
    let mut body = json!({"messages": [{"role": "user", "content": PROMPT}]});
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    let body = body.to_string();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: tur\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    Ok(stream)
}

async fn read_response(mut stream: TcpStream) -> Result<String> {
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await?;
    Ok(String::from_utf8_lossy(&response).into_owned())
}

#[tokio::test]
async fn test_shutdown_closes_the_listener_and_finishes_accepted_requests() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let addr = listener.local_addr()?;
    // Hold the worker's reply until after the signal, by holding the worker's intake.
    let (tx, mut held) = mpsc::channel::<WorkerMsg>(4);
    let (release, mut released) = mpsc::channel::<()>(1);
    let real = real_worker(&fixture, None)?;
    let forward = tokio::spawn(async move {
        let msg = held.recv().await;
        // Either the release or the test's end lets the request through.
        released.recv().await;
        if let Some(msg) = msg {
            drop(real.send(msg).await);
        }
    });
    let (signal, signals) = mpsc::channel(2);
    let app = state(PipelineWorker::from_sender(tx));
    let server = tokio::spawn(serve_until(app, listener, signals, Duration::from_secs(60)));
    let accepted = open_request(addr, &json!({"max_tokens": 3})).await?;
    // Wait until the handler has passed the request on, so it counts as accepted.
    tokio::time::sleep(Duration::from_millis(100)).await;
    signal.send(()).await.map_err(error)?;
    let mut refused = false;
    for _ in 0..200 {
        if TcpStream::connect(addr).await.is_err() {
            refused = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(refused, "the listener stayed open during the drain");
    assert!(
        !server.is_finished(),
        "the drain ignored an accepted request"
    );
    release.send(()).await.map_err(error)?;
    let response = read_response(accepted).await?;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("t5 t5 t5"), "{response}");
    server.await.map_err(error)??;
    forward.await.map_err(error)?;
    Ok(())
}

#[tokio::test]
async fn test_drain_timeout_fails_held_requests_and_stops_the_worker() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let addr = listener.local_addr()?;
    let worker = real_worker(&fixture, None)?;
    let observer = worker.clone();
    let (signal, signals) = mpsc::channel(2);
    let server = tokio::spawn(serve_until(
        state(worker),
        listener,
        signals,
        Duration::ZERO,
    ));
    let accepted = open_request(addr, &json!({"max_tokens": 500})).await?;
    // Signal as soon as the request runs: no drain time, and most of its tokens still to come.
    while observer.telemetry().counters().running == 0 {
        tokio::task::yield_now().await;
    }
    drop(observer);
    signal.send(()).await.map_err(error)?;
    tokio::time::timeout(Duration::from_secs(30), server)
        .await
        .map_err(error)?
        .map_err(error)??;
    let response = tokio::time::timeout(Duration::from_secs(30), read_response(accepted))
        .await
        .map_err(error)??;
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(response.contains("worker_unavailable"), "{response}");
    Ok(())
}
