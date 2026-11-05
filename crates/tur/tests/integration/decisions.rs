//! Decisions: candidate logits read inside a prompt, through the scheduled pipeline on random
//! weights and through `/v1/decisions`. A read before an inner marker equals a read at the end
//! of a prompt cut there, survives prefix restores, and the route renders exactly the prompt
//! Intern-Decision's own `inference.py` renders. The real checkpoint is compared with that
//! reference's probabilities (`#[ignore]`; it downloads `internlm/Intern-Decision-0.8B`).
use crate::common::Checkpoint;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::{collections::HashMap, time::Duration};
use tokio::sync::mpsc;
use tower::ServiceExt;
use tur::{
    InferencePipeline, Result, TurError,
    backend::{
        chat_template::ChatTemplate,
        memory::DeviceBudget,
        pipeline::{GenerationRequest, GenerationStats, PipelineConfig, Scoring},
        prefix::PrefixConfig,
    },
    server::{
        AppState, build_router,
        worker::{PipelineWorker, StreamConfig, WorkerConfig, WorkerMsg},
    },
};

const GOLDEN: &str = include_str!("../fixtures/intern_decision_golden.json");
const TEMPLATE: &str = include_str!("../fixtures/checkpoints/small-chat_template.jinja");

/// Forty tokens of the random fixture's vocabulary, none of them the marker `t90`.
fn shared() -> String {
    (0..40)
        .map(|t| format!("t{}", 2 + (t * 17 + 5) % 80))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A request reading every vocabulary token before each of `markers` occurrences of `t90`.
fn scoring(content: String, markers: usize) -> GenerationRequest {
    let mut request = GenerationRequest::new(content, 1);
    request.sampling.temperature = Some(0.0);
    request.scoring = Some(Scoring {
        marker: "t90".into(),
        candidates: vec![(0..96).map(|id| format!("t{id}")).collect(); markers],
    });
    request
}

/// Snapshots every 16 tokens, chunks of up to 64 tokens.
fn cached() -> PipelineConfig {
    PipelineConfig {
        prefix: PrefixConfig {
            spacing: 16,
            entries: 8,
            state_share: None,
            restorable_first: false,
            tool_hold: Duration::ZERO,
        },
        ..PipelineConfig::new(4, 64)
    }
}

fn assert_close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tolerance, "{x} differs from {y}");
    }
}

#[test]
fn test_inner_marker_reads_match_a_prompt_cut_there_and_survive_restores() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached())?;
    let two = scoring(format!("{} t90 t3 t4 t5 t90 t6", shared()), 2);
    let cold = pipeline.run(&two, |_| {})?;
    assert_eq!(cold.scores.len(), 2);
    assert_eq!(cold.generated_tokens, 1);

    // The first read equals the end of a prompt cut at the first marker.
    let one = pipeline.run(&scoring(format!("{} t90", shared()), 1), |_| {})?;
    assert_close(&cold.scores[0], &one.scores[0], 1e-4);
    // The last read's row is the one the request's greedy output came from.
    let best = (0..96)
        .max_by(|&a, &b| cold.scores[1][a].total_cmp(&cold.scores[1][b]))
        .ok_or_else(|| TurError::Other("no scores".into()))?;
    assert_eq!(cold.generated_text.trim(), format!("t{best}"));

    // Without snapshot boundaries the prompt fits one chunk, and both reads come from its pass.
    let mut single = InferencePipeline::new(
        &fixture.factory(),
        PipelineConfig {
            prefix: PrefixConfig::DISABLED,
            ..PipelineConfig::new(1, 64)
        },
    )?;
    let before = single.scheduler().iteration();
    let one_pass = single.run(&two, |_| {})?;
    assert!(one_pass.prompt_tokens <= 64);
    assert_eq!(single.scheduler().iteration() - before, 1);
    for (one_pass, cold) in one_pass.scores.iter().zip(&cold.scores) {
        assert_close(one_pass, cold, 1e-4);
    }

    // A repeat restores the shared prefix but still computes every read.
    let warm = pipeline.run(&two, |_| {})?;
    assert!(warm.cached_tokens > 0 && warm.cached_tokens < warm.prompt_tokens);
    for (warm, cold) in warm.scores.iter().zip(&cold.scores) {
        assert_close(warm, cold, 1e-4);
    }
    Ok(())
}

#[test]
fn test_scoring_rejects_mismatched_markers_and_multi_token_candidates() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached())?;
    let mut request = scoring(format!("{} t90", shared()), 2);
    assert!(pipeline.run(&request, |_| {}).is_err());
    request = scoring(format!("{} t90", shared()), 1);
    request.max_tokens = 2;
    assert!(pipeline.run(&request, |_| {}).is_err());
    request = scoring(format!("{} t90", shared()), 1);
    request.scoring = Some(Scoring {
        marker: "t90".into(),
        candidates: vec![vec!["t3 t4".into()]],
    });
    assert!(pipeline.run(&request, |_| {}).is_err());
    Ok(())
}

fn plain_stats(scores: Vec<Vec<f32>>) -> GenerationStats {
    GenerationStats {
        finish_reason: "length",
        generated_tokens: 1,
        prompt_tokens: 10,
        cached_tokens: 4,
        elapsed: Duration::from_millis(1),
        first_token: Duration::from_millis(1),
        preemptions: 0,
        draft_tokens: 0,
        accepted_draft_tokens: 0,
        tool_calls: vec![],
        generated_text: String::new(),
        reasoning_text: String::new(),
        logprobs: Vec::new(),
        scores,
    }
}

/// A router whose decision worker scores each marker's first candidate `ln 3` above the others,
/// at candidate temperature 1, and forwards the requests it receives.
fn decision_router() -> (axum::Router, mpsc::UnboundedReceiver<GenerationRequest>) {
    let (tx, mut rx) = mpsc::channel::<WorkerMsg>(8);
    let (seen_tx, seen_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let scores = msg
                .request
                .scoring
                .iter()
                .flat_map(|scoring| &scoring.candidates)
                .map(|set| {
                    (0..set.len())
                        .map(|index| if index == 0 { 3f32.ln() } else { 0.0 })
                        .collect()
                })
                .collect();
            drop(msg.result_tx.send(Ok(plain_stats(scores))));
            drop(seen_tx.send(msg.request));
        }
    });
    let router = build_router(AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([
            (
                "decider".to_string(),
                PipelineWorker::from_sender(tx).with_decisions(1.0),
            ),
            (
                "chat".to_string(),
                PipelineWorker::from_sender(mpsc::channel(1).0),
            ),
        ]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "decider".to_string(),
    });
    (router, seen_rx)
}

async fn post(router: &axum::Router, body: &Value) -> Result<(StatusCode, Value)> {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .map_err(|error| TurError::Other(error.to_string()))?;
    let response = router
        .clone()
        .oneshot(request)
        .await
        .map_err(|error| TurError::Other(error.to_string()))?;
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|error| TurError::Other(error.to_string()))?
        .to_bytes();
    Ok((
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    ))
}

fn golden() -> Result<Value> {
    Ok(serde_json::from_str(GOLDEN)?)
}

fn number(value: &Value) -> f32 {
    #[allow(clippy::cast_possible_truncation, reason = "probabilities are f32")]
    let number = value.as_f64().unwrap_or(f64::NAN) as f32;
    number
}

#[tokio::test]
async fn test_route_renders_the_reference_prompt_of_every_golden_case() -> Result<()> {
    let (router, mut seen) = decision_router();
    let template = ChatTemplate::from_template(TEMPLATE)?;
    for case in golden()?["cases"].as_array().into_iter().flatten() {
        let (status, _) = post(&router, &case["request"]).await?;
        assert_eq!(status, StatusCode::OK);
        let request = seen
            .recv()
            .await
            .ok_or_else(|| TurError::Other("no request".into()))?;
        assert_eq!(request.max_tokens, 1);
        assert!(request.sampling.greedy());
        assert!(!request.render.add_generation_prompt);
        assert_eq!(request.render.enable_thinking, Some(false));
        let rendered = template.render(&request.messages, &[], &request.render)?;
        assert_eq!(Some(rendered.as_str()), case["text"].as_str());
        let fields = case["fields"].as_array().map_or(0, Vec::len);
        assert_eq!(request.scoring.map(|s| s.candidates.len()), Some(fields));
    }
    Ok(())
}

#[tokio::test]
async fn test_route_answers_from_each_markers_symbols() -> Result<()> {
    let (router, _seen) = decision_router();
    let body = json!({
        "input": "The package arrived with a broken screen.",
        "questions": [
            {"type": "predicate", "name": "damaged", "instructions": "Is an item damaged?"},
            {"type": "choice", "name": "queue", "instructions": "Which team?", "choices": [
                {"value": "billing"}, {"value": "shipping"}, {"value": true}
            ]},
            {"type": "score", "instructions": "How urgent?",
             "levels": [{"label": "low"}, {"label": "high"}]}
        ]
    });
    let (status, json) = post(&router, &body).await?;
    assert_eq!(status, StatusCode::OK);
    let answers = &json["answers"];
    // The first symbol is `no`, so `yes` holds the remaining quarter.
    assert!((number(&answers[0]["probability"]) - 0.25).abs() < 1e-6);
    assert_eq!(answers[1]["choice"], "billing");
    assert!((number(&answers[1]["confidence"]) - 0.6).abs() < 1e-6);
    assert_eq!(answers[1]["probabilities"][2]["value"], true);
    assert_eq!(answers[2]["name"], Value::Null);
    assert!((number(&answers[2]["score"]) - 0.25).abs() < 1e-6);
    assert_eq!(json["usage"]["input_tokens"], 10);
    assert_eq!(json["usage"]["output_tokens"], 0);
    assert_eq!(json["usage"]["input_tokens_details"]["cached_tokens"], 4);
    Ok(())
}

#[tokio::test]
async fn test_route_rejects_requests_a_decision_checkpoint_cannot_answer() -> Result<()> {
    let (router, _seen) = decision_router();
    let question = json!({"type": "predicate", "instructions": "?"});
    let choices: Vec<Value> = (0..63).map(|i| json!({"value": i.to_string()})).collect();
    for body in [
        json!({"input": "x", "questions": []}),
        json!({"input": "x", "questions": vec![question.clone(); 17]}),
        json!({"input": "x", "questions": [{"type": "choice", "instructions": "?", "choices": choices}]}),
        json!({"input": "x", "questions": [{"type": "score", "instructions": "?", "levels": [{"label": "only"}]}]}),
        json!({"input": "say <decision>", "questions": [question]}),
        json!({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}],
               "questions": [question]}),
        json!({"model": "chat", "input": "x", "questions": [question]}),
    ] {
        let (status, _) = post(&router, &body).await?;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    Ok(())
}

/// The pinned checkpoint served through the route, with `format` weights on Metal.
fn intern_router(format: tur::models::WeightFormat) -> Result<axum::Router> {
    use tur::backend::factory::{ModelFactory, ModelSource};
    let worker = PipelineWorker::spawn(WorkerConfig {
        factory: ModelFactory {
            source: ModelSource::HuggingFace("internlm/Intern-Decision-0.8B".into()),
            revision: "42457fafd13d10d0b0d9be46cb30a23caa409994".into(),
            format: Some(format),
            device: candle_core::Device::new_metal(0)?,
            kv: tur::models::DEFAULT_KV_LAYOUT,
            recurrent: tur::models::DEFAULT_RECURRENT_STATE,
            audio: false,
            vision: true,
            video: false,
            expert_cache: None,
            gguf: None,
            sparse: None,
        },
        pipeline: PipelineConfig::new(1, 512),
        budget: DeviceBudget::new(None),
        request_timeout: None,
        stream: StreamConfig::default(),
    })?;
    Ok(build_router(AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("intern".to_string(), worker)]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "intern".to_string(),
    }))
}

/// Calibrated probabilities of every golden case's fields, against the reference's CPU F32
/// run: `F32` weights within `f32_tolerance`, and the same likeliest option.
async fn conformance(format: tur::models::WeightFormat, tolerance: f32) -> Result<()> {
    let router = intern_router(format)?;
    for case in golden()?["cases"].as_array().into_iter().flatten() {
        let (status, json) = post(&router, &case["request"]).await?;
        assert_eq!(status, StatusCode::OK, "{json}");
        // The prompt ends before its last marker, the reference's `<decision>` token.
        let cut = case["input_ids"]
            .as_array()
            .and_then(|ids| ids.iter().rposition(|id| id.as_u64() == Some(248_077)));
        assert_eq!(
            json["usage"]["input_tokens"].as_u64(),
            cut.map(|cut| cut as u64)
        );
        let answers = json["answers"].as_array().into_iter().flatten();
        for (answer, field) in answers.zip(case["fields"].as_array().into_iter().flatten()) {
            let expected: Vec<f32> = field["calibrated"]
                .as_array()
                .into_iter()
                .flatten()
                .map(number)
                .collect();
            let actual: Vec<f32> = match answer["type"].as_str() {
                Some("predicate") => {
                    let yes = number(&answer["probability"]);
                    vec![1.0 - yes, yes]
                }
                _ => answer["probabilities"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|p| number(&p["probability"]))
                    .collect(),
            };
            assert_close(&actual, &expected, tolerance);
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads internlm/Intern-Decision-0.8B"]
async fn test_intern_decision_matches_its_reference_probabilities() -> Result<()> {
    conformance(tur::models::WeightFormat::F32, 2e-3).await?;
    conformance(tur::models::WeightFormat::Q8_0, 3e-2).await
}
