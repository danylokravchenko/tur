//! `OpenAI`'s Responses API, over the same generation as chat completions. Tur stores nothing,
//! so a request carries its whole conversation in `input`, and the client runs the tools: a
//! call comes back as a `function_call` item, and its result returns as `function_call_output`.

use std::collections::HashSet;

use axum::{
    Json,
    extract::State,
    http::header,
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
        chat_template::{ChatMessage, Content, ContentPart, Role, ToolCall},
        pipeline::GenerationStats,
        structured::StructuredOutput,
        tools::{self, ToolDefinition},
    },
    core::execution::invalid,
    server::{
        AppState,
        routes::{
            chat::{execution_error, image},
            generation::{self, Controls, ToolChoice, bad_request, forced_output},
        },
        types::{
            DecisionOutputDetails, FunctionOutput, IncompleteDetails, InputContent, InputItem,
            InputMessage, InputPart, OutputContent, OutputItem, PromptTokensDetails, ResponseError,
            ResponseEvent, ResponseObject, ResponsesInput, ResponsesRequest, ResponsesTool,
            ResponsesUsage, SequencedEvent, TextFormat, TypedItem,
        },
        worker::TextDelta,
    },
};

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn function_tools(tools: &[ResponsesTool]) -> Result<Vec<ToolDefinition>, String> {
    tools
        .iter()
        .map(|tool| match tool {
            ResponsesTool::Function {
                name,
                description,
                parameters,
            } => Ok(ToolDefinition {
                name: name.clone(),
                description: description.clone().unwrap_or_default(),
                parameters: parameters
                    .clone()
                    .filter(|parameters| !parameters.is_null())
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            }),
            ResponsesTool::Hosted => Err("only function tools are supported: Tur doesn't run \
                 hosted tools such as mcp or web_search, so run them on the client and offer \
                 them as functions"
                .to_owned()),
        })
        .collect()
}

fn structured_output(
    req: &mut ResponsesRequest,
    tools: &mut Vec<ToolDefinition>,
) -> Result<Option<StructuredOutput>, String> {
    let choice = match req.tool_choice.take() {
        None => ToolChoice::Auto,
        Some(serde_json::Value::String(mode)) => match mode.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => return Err(format!("unsupported tool_choice '{other}'")),
        },
        Some(choice) => ToolChoice::Named(
            choice["name"]
                .as_str()
                .filter(|_| choice["type"] == "function")
                .ok_or("tool_choice must be a mode or name a function")?
                .to_owned(),
        ),
    };
    let answer = match req.text.take().and_then(|text| text.format) {
        None | Some(TextFormat::Text) => None,
        Some(TextFormat::JsonObject) => Some(StructuredOutput::JsonObject),
        Some(TextFormat::JsonSchema { schema }) => Some(StructuredOutput::JsonSchema(schema)),
    };
    forced_output(choice, answer, tools)
}

/// Parts as one content: text parts alone render exactly like string content.
fn content(parts: Vec<InputPart>) -> crate::Result<Content> {
    let mut converted = Vec::with_capacity(parts.len());
    for part in parts {
        converted.push(match part {
            InputPart::InputText { text } | InputPart::OutputText { text } => {
                ContentPart::Text(text)
            }
            InputPart::Refusal { refusal } => ContentPart::Text(refusal),
            InputPart::InputImage { image_url } => image(&image_url)?,
        });
    }
    if converted.iter().all(|p| matches!(p, ContentPart::Text(_))) {
        return Ok(Content::Text(
            converted
                .into_iter()
                .filter_map(|p| match p {
                    ContentPart::Text(text) => Some(text),
                    _ => None,
                })
                .collect(),
        ));
    }
    Ok(Content::Parts(converted))
}

/// Input items as chat turns. A call joins the assistant turn before it, and reasoning the
/// assistant turn after it.
#[derive(Default)]
struct Turns {
    messages: Vec<ChatMessage>,
    reasoning: Option<String>,
    calls: HashSet<String>,
}

impl Turns {
    fn item(&mut self, item: InputItem) -> crate::Result<()> {
        match item {
            InputItem::Message(message) | InputItem::Typed(TypedItem::Message(message)) => {
                self.message(message)
            }
            InputItem::Typed(TypedItem::FunctionCall {
                call_id,
                name,
                arguments,
            }) => {
                let arguments = serde_json::from_str(&arguments).map_err(|error| {
                    invalid(format!(
                        "function_call {call_id} has invalid arguments: {error}"
                    ))
                })?;
                self.calls.insert(call_id);
                self.assistant()
                    .tool_calls
                    .push(ToolCall { name, arguments });
                Ok(())
            }
            InputItem::Typed(TypedItem::FunctionCallOutput { call_id, output }) => {
                if !self.calls.contains(&call_id) {
                    return Err(invalid(format!(
                        "function_call_output {call_id} answers no earlier function_call"
                    )));
                }
                let content = match output {
                    FunctionOutput::Text(text) => Content::Text(text),
                    FunctionOutput::Parts(parts) => content(parts)?,
                };
                self.push(Role::Tool, content);
                Ok(())
            }
            InputItem::Typed(TypedItem::Reasoning { summary, content }) => {
                // The full reasoning when the client kept it, else its summary.
                let parts = if content.is_empty() { summary } else { content };
                let text: Vec<String> = parts.into_iter().map(|part| part.text).collect();
                self.reasoning = Some(text.join("\n\n"));
                Ok(())
            }
        }
    }

    fn message(&mut self, message: InputMessage) -> crate::Result<()> {
        let role = match message.role.as_str() {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(invalid(format!("unsupported message role '{other}'"))),
        };
        let content = match message.content {
            InputContent::Text(text) => Content::Text(text),
            InputContent::Parts(parts) => content(parts)?,
        };
        self.push(role, content);
        Ok(())
    }

    fn push(&mut self, role: Role, content: Content) {
        let reasoning_content = if role == Role::Assistant {
            self.reasoning.take()
        } else {
            None
        };
        self.messages.push(ChatMessage {
            role,
            content,
            reasoning_content,
            tool_calls: Vec::new(),
        });
    }

    /// The assistant turn a call belongs to: the last one, unless another turn followed it.
    fn assistant(&mut self) -> &mut ChatMessage {
        if self
            .messages
            .last()
            .is_none_or(|m| m.role != Role::Assistant)
        {
            self.push(Role::Assistant, Content::Text(String::new()));
        }
        let last = self.messages.len() - 1;
        &mut self.messages[last]
    }
}

fn messages(
    instructions: Option<String>,
    input: ResponsesInput,
) -> crate::Result<Vec<ChatMessage>> {
    let mut turns = Turns::default();
    if let Some(instructions) = instructions {
        turns.push(Role::System, Content::Text(instructions));
    }
    match input {
        ResponsesInput::Text(text) => turns.push(Role::User, Content::Text(text)),
        ResponsesInput::Items(items) => {
            for item in items {
                turns.item(item)?;
            }
        }
    }
    if turns.messages.is_empty() {
        return Err(invalid("input must not be empty"));
    }
    Ok(turns.messages)
}

pub async fn responses(
    State(state): State<AppState>,
    Json(mut req): Json<ResponsesRequest>,
) -> Response {
    if req.previous_response_id.is_some() {
        return bad_request(
            "previous_response_id is unsupported: Tur stores no responses, so send the whole \
             conversation in input",
        );
    }
    let (model, worker) = match generation::worker(&state, &req.model) {
        Ok(found) => found,
        Err(response) => return *response,
    };
    let mut tools = match function_tools(&req.tools) {
        Ok(tools) => tools,
        Err(message) => return bad_request(message),
    };
    let structured = match structured_output(&mut req, &mut tools) {
        Ok(structured) => structured,
        Err(message) => return bad_request(message),
    };
    let forced_call = matches!(structured, Some(StructuredOutput::Tool(_)));
    let (instructions, input) = (req.instructions, req.input);
    let messages = match generation::convert(move || messages(instructions, input)).await {
        Ok(messages) => messages,
        Err(response) => return *response,
    };
    let controls = Controls {
        max_tokens: req.max_output_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        thinking: req.thinking,
        reasoning_effort: req.reasoning.and_then(|reasoning| reasoning.effort),
        cache_salt: req.prompt_cache_key,
        ..Controls::default()
    };
    let submitted = match generation::submit(
        &worker, messages, tools, structured, controls, req.stream,
    )
    .await
    {
        Ok(submitted) => submitted,
        Err(response) => return *response,
    };
    let response = Started {
        id: new_id("resp"),
        created_at: unix_now(),
        model,
    };
    match submitted.tokens {
        Some(tokens) => stream(response, tokens, submitted.result, forced_call),
        None => match submitted.result.await {
            Ok(Ok(stats)) => {
                let output = output(&stats);
                Json(response.finished(output, &stats)).into_response()
            }
            Ok(Err(error)) => generation::failure(&error),
            Err(_) => generation::failure(&crate::TurError::WorkerUnavailable),
        },
    }
}

/// A response's identity, fixed when it starts.
struct Started {
    id: String,
    created_at: u64,
    model: String,
}

impl Started {
    fn object(&self, status: &'static str, output: Vec<OutputItem>) -> ResponseObject {
        ResponseObject {
            id: self.id.clone(),
            object: "response",
            created_at: self.created_at,
            status,
            model: self.model.clone(),
            output,
            usage: None,
            incomplete_details: None,
            error: None,
        }
    }

    /// The finished response; running out of tokens leaves it `incomplete`.
    fn finished(&self, output: Vec<OutputItem>, stats: &GenerationStats) -> ResponseObject {
        let incomplete = stats.finish_reason == "length";
        let mut response = self.object(
            if incomplete {
                "incomplete"
            } else {
                "completed"
            },
            output,
        );
        response.incomplete_details = incomplete.then_some(IncompleteDetails {
            reason: "max_output_tokens",
        });
        let (input, output) = (stats.prompt_tokens as u32, stats.generated_tokens as u32);
        response.usage = Some(ResponsesUsage {
            input_tokens: input,
            input_tokens_details: PromptTokensDetails {
                cached_tokens: stats.cached_tokens as u32,
            },
            output_tokens: output,
            output_tokens_details: DecisionOutputDetails {
                reasoning_tokens: 0,
            },
            total_tokens: input + output,
        });
        response
    }
}

fn reasoning_item(id: String, text: String, status: &'static str) -> OutputItem {
    OutputItem::Reasoning {
        id,
        summary: Vec::new(),
        content: vec![OutputContent::ReasoningText { text }],
        status,
    }
}

fn message_item(id: String, text: String, status: &'static str) -> OutputItem {
    OutputItem::Message {
        id,
        status,
        role: "assistant",
        content: vec![OutputContent::OutputText {
            text,
            annotations: Vec::new(),
        }],
    }
}

fn call_items(calls: &[tools::ToolCall]) -> Vec<OutputItem> {
    calls
        .iter()
        .map(|call| OutputItem::FunctionCall {
            id: new_id("fc"),
            call_id: new_id("call"),
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
            status: "completed",
        })
        .collect()
}

/// Reasoning, then the answer, then each call. A reply of calls alone has no message.
fn output(stats: &GenerationStats) -> Vec<OutputItem> {
    let mut items = Vec::new();
    if !stats.reasoning_text.is_empty() {
        items.push(reasoning_item(
            new_id("rs"),
            stats.reasoning_text.clone(),
            "completed",
        ));
    }
    if !stats.generated_text.is_empty() || stats.tool_calls.is_empty() {
        items.push(message_item(
            new_id("msg"),
            stats.generated_text.clone(),
            "completed",
        ));
    }
    items.extend(call_items(&stats.tool_calls));
    items
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Reasoning,
    Message,
}

/// The output item text is streaming into.
struct Open {
    kind: Kind,
    id: String,
    text: String,
}

/// A response's events as its text arrives: each run of reasoning or answer becomes an item.
struct Events {
    sequence: u64,
    output: Vec<OutputItem>,
    open: Option<Open>,
    /// All text streamed of each kind, which the final text continues.
    reasoning: String,
    content: String,
}

impl Events {
    fn event(&mut self, event: ResponseEvent) -> Event {
        let event = SequencedEvent {
            event,
            sequence_number: self.sequence,
        };
        self.sequence += 1;
        match serde_json::to_value(&event) {
            Ok(value) => {
                let name = value["type"].as_str().unwrap_or_default().to_owned();
                Event::default().event(name).data(value.to_string())
            }
            Err(error) => {
                warn!("response event serialisation error: {error}");
                Event::default().comment("unserialisable event")
            }
        }
    }

    fn push(&mut self, kind: Kind, text: String) -> Vec<Event> {
        let mut events = Vec::new();
        if self.open.as_ref().is_some_and(|open| open.kind != kind) {
            events.extend(self.close());
        }
        if self.open.is_none() {
            events.extend(self.start(kind));
        }
        let Some(open) = self.open.as_mut() else {
            return events;
        };
        open.text.push_str(&text);
        let (item_id, output_index) = (open.id.clone(), self.output.len());
        let event = match kind {
            Kind::Reasoning => {
                self.reasoning.push_str(&text);
                ResponseEvent::ReasoningTextDelta {
                    item_id,
                    output_index,
                    content_index: 0,
                    delta: text,
                }
            }
            Kind::Message => {
                self.content.push_str(&text);
                ResponseEvent::OutputTextDelta {
                    item_id,
                    output_index,
                    content_index: 0,
                    delta: text,
                }
            }
        };
        events.push(self.event(event));
        events
    }

    fn start(&mut self, kind: Kind) -> Vec<Event> {
        let id = new_id(if kind == Kind::Reasoning { "rs" } else { "msg" });
        let output_index = self.output.len();
        let (item, part) = match kind {
            Kind::Reasoning => (
                reasoning_item(id.clone(), String::new(), "in_progress"),
                OutputContent::ReasoningText {
                    text: String::new(),
                },
            ),
            Kind::Message => (
                OutputItem::Message {
                    id: id.clone(),
                    status: "in_progress",
                    role: "assistant",
                    content: Vec::new(),
                },
                OutputContent::OutputText {
                    text: String::new(),
                    annotations: Vec::new(),
                },
            ),
        };
        self.open = Some(Open {
            kind,
            id: id.clone(),
            text: String::new(),
        });
        vec![
            self.event(ResponseEvent::OutputItemAdded { output_index, item }),
            self.event(ResponseEvent::ContentPartAdded {
                item_id: id,
                output_index,
                content_index: 0,
                part,
            }),
        ]
    }

    fn close(&mut self) -> Vec<Event> {
        let Some(Open { kind, id, text }) = self.open.take() else {
            return Vec::new();
        };
        let output_index = self.output.len();
        let (done, part, item) = match kind {
            Kind::Reasoning => (
                ResponseEvent::ReasoningTextDone {
                    item_id: id.clone(),
                    output_index,
                    content_index: 0,
                    text: text.clone(),
                },
                OutputContent::ReasoningText { text: text.clone() },
                reasoning_item(id.clone(), text, "completed"),
            ),
            Kind::Message => (
                ResponseEvent::OutputTextDone {
                    item_id: id.clone(),
                    output_index,
                    content_index: 0,
                    text: text.clone(),
                },
                OutputContent::OutputText {
                    text: text.clone(),
                    annotations: Vec::new(),
                },
                message_item(id.clone(), text, "completed"),
            ),
        };
        self.output.push(item.clone());
        vec![
            self.event(done),
            self.event(ResponseEvent::ContentPartDone {
                item_id: id,
                output_index,
                content_index: 0,
                part,
            }),
            self.event(ResponseEvent::OutputItemDone { output_index, item }),
        ]
    }

    /// Each call as one item, its arguments in a single delta.
    fn calls(&mut self, calls: &[tools::ToolCall]) -> Vec<Event> {
        let mut events = Vec::new();
        for item in call_items(calls) {
            let OutputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
                ..
            } = item.clone()
            else {
                continue;
            };
            let output_index = self.output.len();
            let added = OutputItem::FunctionCall {
                id: id.clone(),
                call_id,
                name: name.clone(),
                arguments: String::new(),
                status: "in_progress",
            };
            events.push(self.event(ResponseEvent::OutputItemAdded {
                output_index,
                item: added,
            }));
            events.push(self.event(ResponseEvent::FunctionCallArgumentsDelta {
                item_id: id.clone(),
                output_index,
                delta: arguments.clone(),
            }));
            events.push(self.event(ResponseEvent::FunctionCallArgumentsDone {
                item_id: id,
                output_index,
                name,
                arguments,
            }));
            self.output.push(item.clone());
            events.push(self.event(ResponseEvent::OutputItemDone { output_index, item }));
        }
        events
    }

    /// Text the worker held back, the items still open, the calls and the final response.
    fn finish(
        &mut self,
        started: &Started,
        stats: &GenerationStats,
        forced_call: bool,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        if let Some(rest) = stats.reasoning_text.strip_prefix(self.reasoning.as_str())
            && !rest.is_empty()
        {
            events.extend(self.push(Kind::Reasoning, rest.to_owned()));
        }
        if !forced_call
            && let Some(rest) = stats.generated_text.strip_prefix(self.content.as_str())
            && !rest.is_empty()
        {
            events.extend(self.push(Kind::Message, rest.to_owned()));
        }
        let answered = self
            .open
            .as_ref()
            .is_some_and(|open| open.kind == Kind::Message)
            || self
                .output
                .iter()
                .any(|item| matches!(item, OutputItem::Message { .. }));
        if !answered && stats.tool_calls.is_empty() {
            events.extend(self.close());
            events.extend(self.start(Kind::Message));
        }
        events.extend(self.close());
        events.extend(self.calls(&stats.tool_calls));
        let response = started.finished(self.output.clone(), stats);
        let event = if response.status == "incomplete" {
            ResponseEvent::Incomplete { response }
        } else {
            ResponseEvent::Completed { response }
        };
        events.push(self.event(event));
        events
    }

    fn failed(&mut self, started: &Started, error: &crate::TurError) -> Event {
        let (_, body) = execution_error(error);
        let mut response = started.object("failed", std::mem::take(&mut self.output));
        response.error = Some(ResponseError {
            code: body.error.code.unwrap_or("internal_error"),
            message: body.error.message,
        });
        self.event(ResponseEvent::Failed { response })
    }
}

fn stream(
    started: Started,
    mut tokens: mpsc::Receiver<TextDelta>,
    result: oneshot::Receiver<crate::Result<GenerationStats>>,
    forced_call: bool,
) -> Response {
    let stream = async_stream::stream! {
        let mut events = Events {
            sequence: 0,
            output: Vec::new(),
            open: None,
            reasoning: String::new(),
            content: String::new(),
        };
        let created = started.object("in_progress", Vec::new());
        yield Ok::<Event, String>(events.event(ResponseEvent::Created { response: created.clone() }));
        yield Ok(events.event(ResponseEvent::InProgress { response: created }));
        while let Some(delta) = tokens.recv().await {
            let pushed = match delta {
                TextDelta::Content(_) if forced_call => continue,
                TextDelta::Content(text) => events.push(Kind::Message, text),
                TextDelta::Reasoning(text) => events.push(Kind::Reasoning, text),
            };
            for event in pushed {
                yield Ok(event);
            }
        }
        match result.await {
            Ok(Ok(stats)) => {
                for event in events.finish(&started, &stats, forced_call) {
                    yield Ok(event);
                }
            }
            Ok(Err(error)) => yield Ok(events.failed(&started, &error)),
            Err(_) => warn!("result channel dropped"),
        }
    };
    let sse = Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    );
    ([(header::CACHE_CONTROL, "no-cache")], sse).into_response()
}
