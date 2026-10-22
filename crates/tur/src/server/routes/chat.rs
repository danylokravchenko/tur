use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use tokio::sync::{mpsc, oneshot};
use tracing::warn;
use uuid::Uuid;

use crate::{
    backend::{
        chat_template::{ChatMessage, RenderOptions, Role, ToolCall},
        pipeline::{GenerationRequest, GenerationStats},
        tools::ToolDefinition,
    },
    hybrid::SamplingOptions,
    server::{
        AppState,
        types::{
            ApiErrorResponse, AssistantFunction, AssistantToolCall, ChatChoice,
            ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
            ChatResponseMessage, ChunkChoice, DeltaFunction, DeltaMessage, DeltaToolCall,
            MessageContent, ResponseFunction, ResponseToolCall, Usage,
        },
        worker::WorkerMsg,
    },
};

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn new_completion_id() -> String {
    format!("chatcmpl-{}", Uuid::new_v4().simple())
}

fn to_internal_messages(
    msgs: &[crate::server::types::ChatMessage],
) -> crate::Result<Vec<ChatMessage>> {
    msgs.iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => {
                    return Err(crate::TurError::Other(format!(
                        "unsupported message role '{}'",
                        m.role
                    )));
                }
            };
            let tool_calls = m
                .tool_calls
                .iter()
                .map(|call| {
                    if call.call_type != "function" {
                        return Err(crate::TurError::Other(
                            "only function tools are supported".into(),
                        ));
                    }
                    Ok(ToolCall {
                        name: call.function.name.clone(),
                        arguments: serde_json::from_str(&call.function.arguments)?,
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            Ok(ChatMessage {
                role,
                content: m
                    .content
                    .as_ref()
                    .map(MessageContent::as_text)
                    .unwrap_or_default(),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls,
            })
        })
        .collect()
}

fn to_internal_tools(tools: &[crate::server::types::OpenAITool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|t| ToolDefinition {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: t.function.parameters.clone(),
        })
        .collect()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiErrorResponse::bad_request("messages must not be empty")),
        )
            .into_response();
    }

    let model_id = if req.model.is_empty() {
        state.default_model.clone()
    } else {
        req.model.clone()
    };

    let worker = match state.workers.get(&model_id) {
        Some(w) => w.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiErrorResponse::bad_request(format!(
                    "model '{}' not found; available: {}",
                    model_id,
                    state.workers.keys().cloned().collect::<Vec<_>>().join(", ")
                ))),
            )
                .into_response();
        }
    };

    let messages = match to_internal_messages(&req.messages) {
        Ok(messages) => messages,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiErrorResponse::bad_request(error.to_string())),
            )
                .into_response();
        }
    };
    if req
        .tool_choice
        .as_ref()
        .is_some_and(|choice| choice != "auto")
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiErrorResponse::bad_request(
                "only tool_choice=auto is supported",
            )),
        )
            .into_response();
    }
    if req.max_tokens == Some(0)
        || req.temperature.is_some_and(|v| !v.is_finite() || v < 0.0)
        || req
            .top_p
            .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
        || req.top_k == Some(0)
        || req
            .reasoning_effort
            .as_ref()
            .is_some_and(|v| !matches!(v.as_str(), "low" | "medium" | "xhigh"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiErrorResponse::bad_request("invalid generation controls")),
        )
            .into_response();
    }
    let tools = to_internal_tools(&req.tools);
    let max_tokens = req.max_tokens.unwrap_or(1024);

    let (token_tx, token_rx) = mpsc::unbounded_channel::<String>();
    let (result_tx, result_rx) = oneshot::channel();

    let worker_msg = WorkerMsg {
        request: GenerationRequest {
            messages,
            tools,
            max_tokens,
            sampling: SamplingOptions {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
            },
            seed: req.seed.unwrap_or(299_792_458),
            render: RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.thinking,
                reasoning_effort: req.reasoning_effort,
                preserve_thinking: req.preserve_thinking,
            },
        },
        token_tx,
        result_tx,
    };

    if let Err(e) = worker.send(worker_msg).await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiErrorResponse::internal(e.to_string())),
        )
            .into_response();
    }

    if req.stream {
        stream_response(model_id, token_rx, result_rx).into_response()
    } else {
        blocking_response(model_id, token_rx, result_rx)
            .await
            .into_response()
    }
}

// ── Non-streaming ─────────────────────────────────────────────────────────────

async fn blocking_response(
    model_id: String,
    mut token_rx: mpsc::UnboundedReceiver<String>,
    result_rx: oneshot::Receiver<crate::Result<GenerationStats>>,
) -> Response {
    let stats = match result_rx.await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiErrorResponse::internal(e.to_string())),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiErrorResponse::internal(
                    "Pipeline worker dropped result channel",
                )),
            )
                .into_response();
        }
    };

    // Drain the token channel (it will be closed by this point) but rely on
    // stats.generated_text for the actual response content — individual tokens
    // decoded per-step may be incomplete for BPE tokenizers.
    while token_rx.try_recv().is_ok() {}
    let full_text = stats.generated_text.clone();

    let (content, response_tool_calls) = if stats.tool_calls.is_empty() {
        (Some(full_text), vec![])
    } else {
        let calls: Vec<AssistantToolCall> = stats
            .tool_calls
            .iter()
            .map(|tc| AssistantToolCall {
                id: format!("call_{:08x}", Uuid::new_v4().as_fields().0),
                call_type: "function".to_string(),
                function: AssistantFunction {
                    name: tc.name.clone(),
                    arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                },
            })
            .collect();
        (None, calls)
    };

    let finish_reason = if response_tool_calls.is_empty() {
        stats.finish_reason
    } else {
        "tool_calls"
    };
    let prompt_tokens = stats.prompt_tokens as u32;
    let completion_tokens = stats.generated_tokens as u32;

    Json(ChatCompletionResponse {
        id: new_completion_id(),
        object: "chat.completion",
        created: unix_now(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content,
                tool_calls: response_tool_calls
                    .iter()
                    .map(|tc| ResponseToolCall {
                        id: tc.id.clone(),
                        call_type: "function",
                        function: ResponseFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect(),
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

// ── Streaming ─────────────────────────────────────────────────────────────────

fn stream_response(
    model_id: String,
    mut token_rx: mpsc::UnboundedReceiver<String>,
    result_rx: oneshot::Receiver<crate::Result<GenerationStats>>,
) -> Response {
    let id = new_completion_id();
    let created = unix_now();

    let stream = async_stream::stream! {
        // First delta establishes the assistant role.
        let first = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage { role: Some("assistant"), content: None, tool_calls: vec![] },
                finish_reason: None,
            }],
            usage: None,
        };
        match serde_json::to_string(&first) {
            Ok(data) => yield Ok::<Event, String>(Event::default().data(data)),
            Err(e) => { yield Err(e.to_string()); return; }
        }

        while let Some(token) = token_rx.recv().await {
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage { role: None, content: Some(token), tool_calls: vec![] },
                    finish_reason: None,
                }],
                usage: None,
            };
            match serde_json::to_string(&chunk) {
                Ok(data) => yield Ok(Event::default().data(data)),
                Err(e) => warn!("chunk serialisation error: {e}"),
            }
        }

        let stats = match result_rx.await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => { warn!("pipeline error: {e}"); return; }
            Err(_) => { warn!("result channel dropped"); return; }
        };

        // Emit parsed tool-call deltas when the model produced any.
        for (idx, tc) in stats.tool_calls.iter().enumerate() {
            let call_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content: None,
                        tool_calls: vec![DeltaToolCall {
                            index: idx as u32,
                            id: Some(format!("call_{:08x}", Uuid::new_v4().as_fields().0)),
                            call_type: Some("function"),
                            function: Some(DeltaFunction {
                                name: Some(tc.name.clone()),
                                arguments: Some(
                                    serde_json::to_string(&tc.arguments).unwrap_or_default(),
                                ),
                            }),
                        }],
                    },
                    finish_reason: None,
                }],
                usage: None,
            };
            match serde_json::to_string(&call_chunk) {
                Ok(data) => yield Ok(Event::default().data(data)),
                Err(e) => warn!("tool-call chunk serialisation error: {e}"),
            }
        }

        let prompt_tokens = stats.prompt_tokens as u32;
        let completion_tokens = stats.generated_tokens as u32;
        let finish_reason = if stats.tool_calls.is_empty() { stats.finish_reason } else { "tool_calls" };
        let final_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage { role: None, content: None, tool_calls: vec![] },
                finish_reason: Some(finish_reason),
            }],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
        };
        match serde_json::to_string(&final_chunk) {
            Ok(data) => yield Ok(Event::default().data(data)),
            Err(e) => warn!("final chunk serialisation error: {e}"),
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    let sse = Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    );
    ([(header::CACHE_CONTROL, "no-cache")], sse).into_response()
}
