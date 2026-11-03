use serde::{Deserialize, Serialize};

/// Represents either a plain string or a typed content-part array,
/// matching the `OpenAI` spec's two allowed forms for message content.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A typed content part: text; audio as `OpenAI`'s `input_audio` (base64 data) or a
/// `data:` URL in the `audio_url` form other servers accept; or an image as `OpenAI`'s
/// `image_url` holding a `data:` URL, or a video in the `video_url` form other servers accept.
/// Other part types fail to deserialize, so the request
/// is rejected rather than silently stripped.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    InputAudio { input_audio: InputAudio },
    AudioUrl { audio_url: MediaUrl },
    ImageUrl { image_url: MediaUrl },
    VideoUrl { video_url: MediaUrl },
}

#[derive(Debug, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded file bytes.
    pub data: String,
    /// Container format; only `wav` is decoded.
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MediaUrl {
    pub url: String,
}

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    /// `OpenAI`'s newer name for `max_tokens`; it wins when both are set.
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Vec<OpenAITool>,
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, alias = "enable_thinking")]
    pub thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub preserve_thinking: Option<bool>,
    /// Prefix-reuse namespace: only requests with the same salt share cached prompts.
    pub cache_salt: Option<String>,
    pub stop: Option<StopSequences>,
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub logprobs: bool,
    pub top_logprobs: Option<usize>,
}

/// `OpenAI`'s `response_format`. Unknown types fail to deserialize.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        json_schema: JsonSchemaFormat,
    },
    /// A Tur extension, shaped like `OpenAI`'s grammar format for custom tools.
    Grammar {
        syntax: GrammarSyntax,
        definition: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarSyntax {
    Regex,
    Lark,
}

#[derive(Debug, Deserialize)]
pub struct JsonSchemaFormat {
    pub schema: serde_json::Value,
}

/// `OpenAI`'s two forms of the `stop` field.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(stop) => vec![stop],
            Self::Many(stops) => stops,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Content is a string or typed part array for user/system/assistant text
    /// turns, null for assistant messages that only emit `tool_calls`.
    pub content: Option<MessageContent>,
    pub tool_call_id: Option<String>,
    pub reasoning_content: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
    pub finish_reason: &'static str,
}

/// `OpenAI`'s `logprobs` object of a choice.
#[derive(Debug, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<LogprobEntry>,
}

#[derive(Debug, Serialize)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Reasoning the model separates from its answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    /// JSON-encoded arguments string (as `OpenAI` specifies).
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    /// Prompt tokens restored from a prefix snapshot instead of recomputed.
    pub cached_tokens: u32,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    /// The checkpoint's immutable revision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// `json` response of the speech-to-text endpoints.
#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// `response_format=verbose_json` of the transcription endpoints.
#[derive(Debug, Serialize)]
pub struct VerboseTranscription {
    pub task: &'static str,
    pub language: Option<String>,
    /// Seconds of audio.
    pub duration: f64,
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    /// Empty unless `timestamp_granularities[]=word` was requested.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<TranscriptionWord>,
}

/// One timed word of a [`VerboseTranscription`].
#[derive(Debug, Serialize)]
pub struct TranscriptionWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// One timed stretch of a [`VerboseTranscription`].
#[derive(Debug, Serialize)]
pub struct TranscriptionSegment {
    pub id: usize,
    /// Feature frame at which this segment's window started.
    pub seek: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

// ── Decisions endpoint ────────────────────────────────────────────────────────

/// `OpenAI`'s `POST /v1/decisions` body: typed questions answered with probabilities
/// about one shared input. Unknown fields such as `safety_identifier` are ignored.
#[derive(Debug, Deserialize)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: String,
    pub input: DecisionInput,
    pub questions: Vec<DecisionQuestion>,
    /// Partitions prefix reuse like a chat request's `cache_salt`.
    pub prompt_cache_key: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DecisionInput {
    Text(String),
    Messages(Vec<DecisionMessage>),
}

/// One input message. Tool-call items have no `role` and fail to deserialize.
#[derive(Debug, Deserialize)]
pub struct DecisionMessage {
    pub role: String,
    pub content: DecisionContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DecisionContent {
    Text(String),
    Parts(Vec<DecisionPart>),
}

/// A Responses-style content part. Images must be `data:` URLs.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionPart {
    InputText { text: String },
    OutputText { text: String },
    InputImage { image_url: String },
    InputJson { data: serde_json::Value },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionQuestion {
    Predicate {
        instructions: String,
        name: Option<String>,
    },
    Choice {
        instructions: String,
        name: Option<String>,
        choices: Vec<DecisionChoice>,
    },
    Score {
        instructions: String,
        name: Option<String>,
        /// Ordered from lowest to highest.
        levels: Vec<DecisionLevel>,
    },
}

#[derive(Debug, Deserialize)]
pub struct DecisionChoice {
    pub value: ChoiceValue,
    pub description: Option<String>,
}

/// A choice's value; `"true"` and `true` are different values.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChoiceValue {
    Bool(bool),
    Text(String),
}

#[derive(Debug, Deserialize)]
pub struct DecisionLevel {
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    /// In question order.
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    Predicate {
        name: Option<String>,
        probability: f32,
    },
    Choice {
        name: Option<String>,
        choice: ChoiceValue,
        confidence: f32,
        probabilities: Vec<ChoiceProbability>,
    },
    Score {
        name: Option<String>,
        /// Probability-weighted level index, from 0.
        score: f32,
        confidence: f32,
        probabilities: Vec<LevelProbability>,
    },
}

#[derive(Debug, Serialize)]
pub struct ChoiceProbability {
    pub value: ChoiceValue,
    pub probability: f32,
}

#[derive(Debug, Serialize)]
pub struct LevelProbability {
    /// The level's index, from 0.
    pub value: usize,
    pub label: String,
    pub probability: f32,
}

#[derive(Debug, Serialize)]
pub struct DecisionUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub input_tokens_details: DecisionInputDetails,
    pub output_tokens_details: DecisionOutputDetails,
}

#[derive(Debug, Serialize)]
pub struct DecisionInputDetails {
    pub cached_tokens: u32,
    pub cache_write_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionOutputDetails {
    pub reasoning_tokens: u32,
}

// ── Responses endpoint ────────────────────────────────────────────────────────

/// `OpenAI`'s `POST /v1/responses` body. Nothing is stored, so a conversation arrives whole in
/// `input`. Unknown fields such as `store`, `include` and `parallel_tool_calls` are ignored.
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: String,
    pub input: ResponsesInput,
    /// A system turn ahead of `input`.
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Vec<ResponsesTool>,
    pub tool_choice: Option<serde_json::Value>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    pub reasoning: Option<ReasoningConfig>,
    /// A Tur extension, as in chat completions: the template's thinking switch.
    #[serde(default, alias = "enable_thinking")]
    pub thinking: Option<bool>,
    pub text: Option<TextConfig>,
    /// Refused: there is no stored response to continue from.
    pub previous_response_id: Option<String>,
    /// Partitions prefix reuse like a chat request's `cache_salt`.
    pub prompt_cache_key: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

/// One input item. Messages may leave out their `type`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    Typed(TypedItem),
    Message(InputMessage),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedItem {
    Message(InputMessage),
    FunctionCall {
        call_id: String,
        name: String,
        /// JSON-encoded, as the model wrote them.
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: FunctionOutput,
    },
    /// Reasoning of the assistant turn that follows it.
    Reasoning {
        #[serde(default)]
        summary: Vec<ReasoningPart>,
        #[serde(default)]
        content: Vec<ReasoningPart>,
    },
}

#[derive(Debug, Deserialize)]
pub struct InputMessage {
    pub role: String,
    pub content: InputContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    Text(String),
    Parts(Vec<InputPart>),
}

/// A content part. Images must be `data:` URLs; other part types fail to deserialize.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputPart {
    InputText { text: String },
    OutputText { text: String },
    Refusal { refusal: String },
    InputImage { image_url: String },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum FunctionOutput {
    Text(String),
    Parts(Vec<InputPart>),
}

#[derive(Debug, Deserialize)]
pub struct ReasoningPart {
    pub text: String,
}

/// A tool the model may call. Hosted tools such as `mcp` or `web_search` would need the server
/// to run them, which Tur never does.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    Function {
        name: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        parameters: Option<serde_json::Value>,
    },
    #[serde(other)]
    Hosted,
}

#[derive(Debug, Deserialize)]
pub struct ReasoningConfig {
    pub effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TextConfig {
    pub format: Option<TextFormat>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    Text,
    JsonObject,
    JsonSchema { schema: serde_json::Value },
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    /// `in_progress`, `completed`, `incomplete` or `failed`.
    pub status: &'static str,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub usage: Option<ResponsesUsage>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub error: Option<ResponseError>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Reasoning {
        id: String,
        summary: Vec<OutputContent>,
        content: Vec<OutputContent>,
        status: &'static str,
    },
    Message {
        id: String,
        status: &'static str,
        role: &'static str,
        content: Vec<OutputContent>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        /// JSON-encoded.
        arguments: String,
        status: &'static str,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContent {
    OutputText {
        text: String,
        annotations: Vec<serde_json::Value>,
    },
    ReasoningText {
        text: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub input_tokens_details: PromptTokensDetails,
    pub output_tokens: u32,
    pub output_tokens_details: DecisionOutputDetails,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct IncompleteDetails {
    pub reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseError {
    pub code: &'static str,
    pub message: String,
}

/// A streamed event; `type` names it, and the SSE event carries the same name.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponseEvent {
    #[serde(rename = "response.created")]
    Created { response: ResponseObject },
    #[serde(rename = "response.in_progress")]
    InProgress { response: ResponseObject },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OutputContent,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OutputContent,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: usize,
        name: String,
        arguments: String,
    },
    #[serde(rename = "response.completed")]
    Completed { response: ResponseObject },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponseObject },
    #[serde(rename = "response.failed")]
    Failed { response: ResponseObject },
}

/// A [`ResponseEvent`] numbered in stream order, from 0.
#[derive(Debug, Serialize)]
pub struct SequencedEvent {
    #[serde(flatten)]
    pub event: ResponseEvent,
    pub sequence_number: u64,
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

    /// The answer to new work while the server drains.
    pub fn shutting_down() -> Self {
        Self {
            error: ApiErrorBody {
                message: "the server is shutting down".into(),
                error_type: "server_error",
                code: Some("server_shutting_down"),
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
