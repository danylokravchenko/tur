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
    backend::{chat_template::Message, pipeline::GenerationStats, tools::ToolDefinition},
    server::{
        AppState,
        types::{
            ApiErrorResponse, AssistantFunction, AssistantToolCall, ChatChoice,
            ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
            ChatResponseMessage, ChunkChoice, DeltaFunction, DeltaMessage, DeltaToolCall,
            ResponseFunction, ResponseToolCall, Usage,
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

fn to_internal_messages(msgs: &[crate::server::types::ChatMessage]) -> Vec<Message> {
    msgs.iter()
        .map(|m| Message {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
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

    let messages = to_internal_messages(&req.messages);
    let tools = to_internal_tools(&req.tools);
    let max_tokens = req.max_tokens.unwrap_or(1024);
    let model_id = req.model.clone();

    let (token_tx, token_rx) = mpsc::unbounded_channel::<String>();
    let (result_tx, result_rx) = oneshot::channel();

    let worker_msg = WorkerMsg {
        messages,
        tools,
        max_tokens,
        thinking: req.thinking,
        token_tx,
        result_tx,
    };

    if let Err(e) = state.worker.send(worker_msg).await {
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

    let mut full_text = String::new();
    while let Ok(t) = token_rx.try_recv() {
        full_text.push_str(&t);
    }

    let (content, response_tool_calls) = if stats.tool_calls.is_empty() {
        (Some(full_text), vec![])
    } else {
        let calls: Vec<AssistantToolCall> = stats
            .tool_calls
            .iter()
            .map(|tc| AssistantToolCall {
                id: format!("call_{}", &Uuid::new_v4().simple().to_string()[..8]),
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
        "stop"
    } else {
        "tool_calls"
    };
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
            prompt_tokens: 0,
            completion_tokens,
            total_tokens: completion_tokens,
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
                            id: Some(format!("call_{}", &Uuid::new_v4().simple().to_string()[..8])),
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
            };
            match serde_json::to_string(&call_chunk) {
                Ok(data) => yield Ok(Event::default().data(data)),
                Err(e) => warn!("tool-call chunk serialisation error: {e}"),
            }
        }

        let finish_reason = if stats.tool_calls.is_empty() { "stop" } else { "tool_calls" };
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
