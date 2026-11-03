//! What the chat completions and responses endpoints share: generation controls, tool choice,
//! the worker lookup and submission.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tokio::sync::{mpsc, oneshot};

use crate::{
    backend::{
        chat_template::{ChatMessage, RenderOptions},
        pipeline::{GenerationRequest, GenerationStats},
        structured::StructuredOutput,
        tools::ToolDefinition,
    },
    hybrid::SamplingOptions,
    server::{
        AppState,
        routes::chat::execution_error,
        types::ApiErrorResponse,
        worker::{PipelineWorker, TextDelta, WorkerMsg},
    },
};

pub(crate) fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiErrorResponse::bad_request(message)),
    )
        .into_response()
}

pub(crate) fn failure(error: &crate::TurError) -> Response {
    let (status, body) = execution_error(error);
    (status, Json(body)).into_response()
}

/// Sampling, length and template controls of one request, as each API names them.
#[derive(Default)]
pub(crate) struct Controls {
    pub(crate) max_tokens: Option<usize>,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<usize>,
    pub(crate) presence_penalty: Option<f64>,
    pub(crate) frequency_penalty: Option<f64>,
    pub(crate) repetition_penalty: Option<f64>,
    pub(crate) seed: Option<u64>,
    pub(crate) thinking: Option<bool>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) preserve_thinking: Option<bool>,
    pub(crate) cache_salt: Option<String>,
    pub(crate) stop: Vec<String>,
    pub(crate) logprobs: Option<usize>,
}

impl Controls {
    fn valid(&self) -> bool {
        self.max_tokens != Some(0)
            && !self.temperature.is_some_and(|v| !v.is_finite() || v < 0.0)
            && !self
                .top_p
                .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
            && self.top_k != Some(0)
            && [self.presence_penalty, self.frequency_penalty]
                .iter()
                .flatten()
                .all(|v| v.is_finite())
            && !self
                .repetition_penalty
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
            && self
                .reasoning_effort
                .as_ref()
                .is_none_or(|v| matches!(v.as_str(), "low" | "medium" | "xhigh"))
    }
}

/// A request's `tool_choice`, whatever shape its API gives it.
pub(crate) enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

/// What `choice` and a forced `answer` format make of `tools`, or why they can't be honored.
/// `none` also clears the tools, so the model isn't told about them.
pub(crate) fn forced_output(
    choice: ToolChoice,
    answer: Option<StructuredOutput>,
    tools: &mut Vec<ToolDefinition>,
) -> Result<Option<StructuredOutput>, String> {
    let tool = match choice {
        ToolChoice::Auto => None,
        ToolChoice::None => {
            tools.clear();
            None
        }
        ToolChoice::Required if tools.is_empty() => {
            return Err("tool_choice=required needs at least one tool".into());
        }
        ToolChoice::Required => Some(StructuredOutput::Tool(None)),
        ToolChoice::Named(name) => {
            if !tools.iter().any(|tool| tool.name == name) {
                return Err(format!("tool_choice names unknown tool '{name}'"));
            }
            Some(StructuredOutput::Tool(Some(name)))
        }
    };
    // A forced answer leaves a model no way to reach a tool it was told about.
    if answer.is_some() && !tools.is_empty() {
        return Err(
            "a forced answer format can't be combined with tools; set tool_choice to none".into(),
        );
    }
    Ok(tool.or(answer))
}

/// A response that ends a request before generation, boxed since a `Response` is large.
pub(crate) type Refusal = Box<Response>;

/// The worker serving `model`, or the default one when it is empty.
pub(crate) fn worker(state: &AppState, model: &str) -> Result<(String, PipelineWorker), Refusal> {
    let model = if model.is_empty() {
        state.default_model.clone()
    } else {
        model.to_owned()
    };
    match state.workers.get(&model) {
        Some(worker) => Ok((model, worker.clone())),
        None => Err(Box::new(
            (
                StatusCode::NOT_FOUND,
                Json(ApiErrorResponse::bad_request(format!(
                    "model '{model}' not found; available: {}",
                    state.workers.keys().cloned().collect::<Vec<_>>().join(", ")
                ))),
            )
                .into_response(),
        )),
    }
}

/// Run a request's message conversion off the async executor, since it decodes media.
pub(crate) async fn convert<T: Send + 'static>(
    convert: impl FnOnce() -> crate::Result<T> + Send + 'static,
) -> Result<T, Refusal> {
    match tokio::task::spawn_blocking(convert).await {
        Ok(Ok(converted)) => Ok(converted),
        Ok(Err(error)) => Err(Box::new(bad_request(error.to_string()))),
        Err(_) => Err(Box::new(
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiErrorResponse::internal("message decoding failed")),
            )
                .into_response(),
        )),
    }
}

/// A submitted request's text stream, when it asked for one, and its outcome.
pub(crate) struct Submitted {
    pub(crate) tokens: Option<mpsc::Receiver<TextDelta>>,
    pub(crate) result: oneshot::Receiver<crate::Result<GenerationStats>>,
}

/// Validate `controls` and queue one generation on `worker`.
pub(crate) async fn submit(
    worker: &PipelineWorker,
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    structured: Option<StructuredOutput>,
    controls: Controls,
    stream: bool,
) -> Result<Submitted, Refusal> {
    if !controls.valid() {
        return Err(Box::new(bad_request("invalid generation controls")));
    }
    let (token_tx, tokens) = if stream {
        let (tx, rx) = worker.stream_channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let (result_tx, result) = oneshot::channel();
    let request = GenerationRequest {
        messages,
        tools,
        max_tokens: controls.max_tokens.unwrap_or(1024),
        fit_context: controls.max_tokens.is_none(),
        stop: controls.stop,
        sampling: SamplingOptions {
            temperature: controls.temperature,
            top_p: controls.top_p,
            top_k: controls.top_k,
            repetition_penalty: controls.repetition_penalty,
            presence_penalty: controls.presence_penalty,
            frequency_penalty: controls.frequency_penalty,
            logprobs: controls.logprobs,
        },
        seed: controls.seed.unwrap_or(299_792_458),
        timeout: None,
        submitted_at: None,
        cache_salt: controls.cache_salt,
        grammar: None,
        structured,
        encoded_media: Vec::new(),
        scoring: None,
        render: RenderOptions {
            add_generation_prompt: true,
            enable_thinking: controls.thinking,
            reasoning_effort: controls.reasoning_effort,
            preserve_thinking: controls.preserve_thinking,
            add_vision_id: None,
        },
    };
    let msg = WorkerMsg {
        request,
        token_tx,
        result_tx,
    };
    worker
        .send(msg)
        .await
        .map_err(|error| Box::new(failure(&error)))?;
    Ok(Submitted { tokens, result })
}
