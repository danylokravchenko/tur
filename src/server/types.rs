use serde::{Deserialize, Serialize};

/// Represents either a plain string or a typed content-part array,
/// matching the OpenAI spec's two allowed forms for message content.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
}

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    #[serde(default)]
    pub tools: Vec<OpenAITool>,
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, alias = "enable_thinking")]
    pub thinking: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Content is a string or typed part array for user/system/assistant text
    /// turns, null for assistant messages that only emit tool_calls.
    pub content: Option<MessageContent>,
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<AssistantToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAITool {
    pub function: OpenAIFunctionDef,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_empty_object")]
    pub parameters: serde_json::Value,
}

fn default_empty_object() -> serde_json::Value {
    serde_json::json!({"type": "object", "properties": {}})
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AssistantToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: AssistantFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AssistantFunction {
    pub name: String,
    pub arguments: String,
}

// ── Non-streaming response ────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ResponseToolCall>,
}

#[derive(Debug, Serialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: ResponseFunction,
}

#[derive(Debug, Serialize)]
pub struct ResponseFunction {
    pub name: String,
    /// JSON-encoded arguments string (as OpenAI specifies).
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming chunks ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<DeltaToolCall>,
}

#[derive(Debug, Serialize)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<DeltaFunction>,
}

#[derive(Debug, Serialize)]
pub struct DeltaFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ── Models endpoint ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

// ── Error response ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub code: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorResponse {
    pub error: ApiErrorBody,
}

impl ApiErrorResponse {
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: message.into(),
                error_type: "server_error",
                code: None,
            },
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: message.into(),
                error_type: "invalid_request_error",
                code: None,
            },
        }
    }
}
