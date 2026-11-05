//! Agent traffic end to end: chat completions and Responses requests shaped as agents and
//! MCP clients send them, through the real worker, pipeline, call grammar and parser.
//!
//! A scripted checkpoint makes the model's side deterministic. Its embeddings are one-hot and
//! every decoder update is zero except layer 0's MLP, which maps each token to its scripted
//! successor, so the next token depends only on the last one: a prompt's last token picks the
//! reply. The fixture template ends a prompt with the last message's text, or with
//! `cue_answer` after a tool result, then opens `<think>` when thinking is on, as Qwen's do.
//! It fails to render a tool result without its call.
use crate::common;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use tower::ServiceExt;
use tur::{
    Result, TurError,
    backend::{memory::DeviceBudget, pipeline::PipelineConfig},
    server::{
        AppState, build_router,
        worker::{PipelineWorker, StreamConfig, WorkerConfig},
    },
};

const EOS: &str = "<|im_end|>";
/// Hidden width, and so the largest vocabulary one-hot embeddings can hold.
const WIDTH: usize = 128;

/// A preamble, then a call; whitespace separates them, as Qwen writes it.
const WEATHER: &[&str] = &[
    "cue_weather",
    "Checking",
    ".",
    "\n\n",
    "<tool_call>",
    "\n<function=get_weather>\n<parameter=city>\n",
    "Paris",
    "\n</parameter>\n</function>\n",
    "</tool_call>",
];
const ANSWER: &[&str] = &["cue_answer", "It", " is", " sunny."];
/// A call with every value kind an MCP schema gives its parameters. Where the grammar forces
/// text, a grammar allows only the tokenizer's own first token of it, so the script breaks
/// there: after the opening marker, and inside `meta`.
const ISSUE: &[&str] = &[
    "cue_issue",
    "<tool_call>",
    "\n<function=",
    "github__create_issue>\n<parameter=repo>\n",
    "acme/tur",
    "\n</parameter>\n<parameter=title>\n",
    "Crash on empty input",
    "\n</parameter>\n<parameter=labels>\n",
    r#"["bug"]"#,
    "\n</parameter>\n<parameter=milestone>\n",
    "7",
    "\n</parameter>\n<parameter=meta>\n",
    r#"{"priority":"#,
    "1}",
    "\n</parameter>\n</function>\n",
    "</tool_call>",
];
/// Reasoning in the thinking block the prompt opened, then a call.
const THINK: &[&str] = &[
    "<think>",
    "A city's weather needs a lookup.",
    "</think>",
    "\n\n",
    "<tool_call>",
    "\n<function=get_weather>\n<parameter=city>\n",
    "Paris",
    "\n</parameter>\n</function>\n",
    "</tool_call>",
];
/// Tokens outside every script, for forced text that a script token only continues.
const FALLBACK: &[&str] = &["["];

const TEMPLATE: &str = "\
{%- for tool in tools or [] %}{{ tool.function.name }}: {{ tool.function.parameters | tojson }}
{% endfor %}
{%- set ns = namespace(calls=0, results=0) %}
{%- for message in messages %}
{%- if message.role == 'tool' %}{% set ns.results = ns.results + 1 %}
{%- if ns.results > ns.calls %}{{ raise_exception('a tool result without its call') }}{% endif %}
{%- endif %}
{{ message.role }}: {{ message.content }}
{%- for call in message.tool_calls or [] %}{% set ns.calls = ns.calls + 1 %}
<tool_call>{{ call.function.name }} {{ call.function.arguments | tojson }}</tool_call>
{%- endfor %}
{%- endfor %}
{%- if add_generation_prompt and messages[-1].role == 'tool' %} cue_answer{% endif %}
{%- if add_generation_prompt and enable_thinking is defined and enable_thinking %}<think>{% endif %}";

/// Each token's scripted successor; a script's last token ends the turn.
fn successors(scripts: &[&[&str]]) -> Result<(Vec<String>, HashMap<String, String>)> {
    let mut words: Vec<String> = ["<|endoftext|>", EOS, "<tool_call>", "</tool_call>"]
        .map(str::to_owned)
        .to_vec();
    words.extend(FALLBACK.iter().map(|word| (*word).to_owned()));
    let mut next = HashMap::new();
    for script in scripts {
        for (at, word) in script.iter().enumerate() {
            let after = script.get(at + 1).copied().unwrap_or(EOS);
            if next
                .insert((*word).to_owned(), after.to_owned())
                .is_some_and(|old| old != after)
            {
                return Err(TurError::Other(format!("{word:?} has two successors")));
            }
            if !words.iter().any(|w| w == word) {
                words.push((*word).to_owned());
            }
        }
    }
    if words.len() > WIDTH {
        return Err(TurError::Other("scripts exceed the vocabulary".into()));
    }
    Ok((words, next))
}

/// Every scripted token is an added one, so tokenizing a script's text yields its tokens: a
/// grammar forces text through the tokenizer's own encoding of it. The markers stay
/// non-special, as in Qwen's tokenizer.
fn tokenizer(words: &[String]) -> Value {
    let added_tokens: Vec<Value> = words
        .iter()
        .enumerate()
        .map(|(id, word)| {
            json!({"id": id, "content": word, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": id < 2})
        })
        .collect();
    let vocab: serde_json::Map<String, Value> = words
        .iter()
        .enumerate()
        .map(|(id, word)| (word.clone(), id.into()))
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
        "pre_tokenizer": null, "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
            "use_regex": false},
        "added_tokens": added_tokens,
        "model": {"type": "BPE", "merges": [], "vocab": vocab, "unk_token": "<|endoftext|>"}
    })
}

/// Set `values[row * WIDTH + column]` of a row-major matrix.
fn set(values: &mut [f32], row: usize, column: usize) {
    if let Some(value) = values.get_mut(row * WIDTH + column) {
        *value = 1.0;
    }
}

/// A checkpoint that answers each script: its first token, ending a prompt, makes the model
/// write the rest and end its turn.
fn scripted(scripts: &[&[&str]]) -> Result<common::Checkpoint> {
    let (words, next) = successors(scripts)?;
    let id = |word: &str| words.iter().position(|w| w == word).unwrap_or(1);
    let successor: Vec<usize> = words
        .iter()
        .map(|word| next.get(word).map_or(1, |after| id(after)))
        .collect();
    let mut config: Value = serde_json::from_str(common::TINY)?;
    config["hidden_size"] = json!(WIDTH);
    config["intermediate_size"] = json!(WIDTH);
    config["vocab_size"] = json!(WIDTH);
    // Text outside the scripts tokenizes a character at a time, so tool schemas run long.
    config["max_position_embeddings"] = json!(4096);
    if let Some(config) = config.as_object_mut() {
        config.remove("image_token_id");
    }
    let checkpoint = common::Checkpoint::write(&config.to_string(), |name, count, _| {
        let mut values = vec![0f32; count];
        if name.ends_with("embed_tokens.weight") {
            (0..WIDTH).for_each(|token| set(&mut values, token, token));
        } else if name.ends_with("layers.0.mlp.gate_proj.weight")
            || name.ends_with("layers.0.mlp.up_proj.weight")
        {
            (0..WIDTH).for_each(|unit| set(&mut values, unit, unit));
        } else if name.ends_with("layers.0.mlp.down_proj.weight") {
            // Unit `token` writes its successor's direction: silu(x)·x ≈ 128 outweighs the
            // token's own embedding by two orders of magnitude.
            for (token, &after) in successor.iter().enumerate() {
                set(&mut values, after, token);
            }
        }
        values
    })?;
    std::fs::write(
        checkpoint.0.join("tokenizer.json"),
        tokenizer(&words).to_string(),
    )?;
    std::fs::write(
        checkpoint.0.join("tokenizer_config.json"),
        json!({"chat_template": TEMPLATE, "eos_token": EOS}).to_string(),
    )?;
    Ok(checkpoint)
}

fn router(checkpoint: &common::Checkpoint) -> Result<axum::Router> {
    let worker = PipelineWorker::spawn(WorkerConfig {
        factory: checkpoint.factory(),
        pipeline: PipelineConfig::new(2, 16),
        budget: DeviceBudget::new(None),
        request_timeout: None,
        stream: StreamConfig::default(),
    })?;
    Ok(build_router(AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("agent".into(), worker)]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "agent".into(),
    }))
}

fn error(error: impl std::fmt::Display) -> TurError {
    TurError::Other(error.to_string())
}

async fn post(router: &axum::Router, uri: &str, body: &Value) -> Result<(StatusCode, String)> {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .map_err(error)?;
    let response = router.clone().oneshot(request).await.map_err(error)?;
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(error)?
        .to_bytes();
    Ok((status, String::from_utf8(bytes.to_vec()).map_err(error)?))
}

async fn post_json(router: &axum::Router, uri: &str, body: &Value) -> Result<Value> {
    let (status, text) = post(router, uri, body).await?;
    assert_eq!(status, StatusCode::OK, "{text}");
    Ok(serde_json::from_str(&text)?)
}

/// Server-sent events as `(event name, data)`; chat completions name none and end with
/// `[DONE]`, which is left out.
fn events(text: &str) -> Result<Vec<(String, Value)>> {
    let mut events = Vec::new();
    for block in text.split("\n\n").filter(|block| !block.trim().is_empty()) {
        let mut name = String::new();
        for line in block.lines() {
            if let Some(event) = line.strip_prefix("event: ") {
                event.clone_into(&mut name);
            } else if let Some(data) = line.strip_prefix("data: ")
                && data != "[DONE]"
            {
                events.push((name.clone(), serde_json::from_str(data)?));
            }
        }
    }
    Ok(events)
}

/// A weather tool as the `OpenAI` SDK sends it, `strict` included.
fn weather_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.", "strict": true,
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }, "required": ["city"], "additionalProperties": false}
    }})
}

fn arguments(call: &Value) -> Result<Value> {
    let text = call["function"]["arguments"]
        .as_str()
        .or_else(|| call["arguments"].as_str())
        .ok_or_else(|| error("arguments are not a string"))?;
    Ok(serde_json::from_str(text)?)
}

#[tokio::test]
async fn test_chat_agent_loop_returns_a_call_and_answers_its_result() -> Result<()> {
    let checkpoint = scripted(&[WEATHER, ANSWER])?;
    let router = router(&checkpoint)?;
    let mut messages = vec![
        json!({"role": "developer", "content": "Use tools when they help."}),
        json!({"role": "user", "content": "What's the weather in Paris? cue_weather"}),
    ];
    let request = |messages: &[Value]| {
        json!({"model": "agent", "messages": messages, "tools": [weather_tool()],
            "tool_choice": "auto", "parallel_tool_calls": false, "max_completion_tokens": 64})
    };
    let first = post_json(&router, "/v1/chat/completions", &request(&messages)).await?;
    let choice = &first["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    // The preamble stays, without the whitespace before the call or the call's markup.
    assert_eq!(choice["message"]["content"], "Checking.");
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .ok_or_else(|| error("no tool_calls"))?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(arguments(&calls[0])?, json!({"city": "Paris"}));
    let id = calls[0]["id"].as_str().ok_or_else(|| error("no call id"))?;
    assert!(id.starts_with("call_"), "{id}");

    // The agent sends the reply back as history, then the tool's result as content parts.
    messages.push(choice["message"].clone());
    messages.push(json!({"role": "tool", "tool_call_id": id, "content": [
        {"type": "text", "text": "{\"temperature\": 21, \"conditions\": \"sunny\"}"}
    ]}));
    let second = post_json(&router, "/v1/chat/completions", &request(&messages)).await?;
    let choice = &second["choices"][0];
    assert_eq!(choice["finish_reason"], "stop");
    assert_eq!(choice["message"]["content"], "It is sunny.");
    assert!(choice["message"]["tool_calls"].is_null());
    Ok(())
}

#[tokio::test]
async fn test_chat_stream_sends_calls_only_as_tool_call_deltas() -> Result<()> {
    let checkpoint = scripted(&[WEATHER])?;
    let router = router(&checkpoint)?;
    let body = json!({"messages": [{"role": "user", "content": "Weather? cue_weather"}],
        "tools": [weather_tool()], "stream": true, "stream_options": {"include_usage": true}});
    let (status, text) = post(&router, "/v1/chat/completions", &body).await?;
    assert_eq!(status, StatusCode::OK);
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks = events(&text)?;
    let content: String = chunks
        .iter()
        .filter_map(|(_, chunk)| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Checking.");
    let calls: Vec<&Value> = chunks
        .iter()
        .filter_map(|(_, chunk)| chunk["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .collect();
    assert_eq!(calls.len(), 1, "{text}");
    assert_eq!(calls[0]["index"], 0);
    assert!(
        calls[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("call_"))
    );
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(arguments(calls[0])?, json!({"city": "Paris"}));
    let (_, last) = chunks.last().ok_or_else(|| error("no chunks"))?;
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(last["usage"]["completion_tokens"], 9);
    Ok(())
}

/// Tools as an MCP client lists them for the model: server-prefixed names, `$schema`, enums,
/// nested and closed objects, and in strict mode's style every field required, with optional
/// ones nullable.
fn mcp_tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "github__create_issue", "description": "Open an issue.",
            "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/name"},
                    "title": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string", "enum": ["bug", "feature"]}},
                    "milestone": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
                    "meta": {"type": "object", "properties": {"priority": {"type": "integer"}},
                        "required": ["priority"], "additionalProperties": false}
                },
                "required": ["repo", "title", "labels", "milestone", "meta"],
                "additionalProperties": false
            }
        }},
        {"type": "function", "function": {
            "name": "filesystem.read-file", "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"]}
        }},
        {"type": "function", "function": {"name": "time__now"}}
    ])
}

#[tokio::test]
async fn test_mcp_tool_schemas_keep_every_argument_type_through_grammar_and_parser() -> Result<()> {
    let checkpoint = scripted(&[ISSUE])?;
    let router = router(&checkpoint)?;
    let body = json!({"messages": [{"role": "user", "content": "File it. cue_issue"}],
        "tools": mcp_tools()});
    let reply = post_json(&router, "/v1/chat/completions", &body).await?;
    let message = &reply["choices"][0]["message"];
    assert!(message["content"].is_null(), "{message}");
    assert_eq!(
        message["tool_calls"][0]["function"]["name"],
        "github__create_issue"
    );
    assert_eq!(
        arguments(&message["tool_calls"][0])?,
        json!({"repo": "acme/tur", "title": "Crash on empty input", "labels": ["bug"],
            "milestone": 7, "meta": {"priority": 1}})
    );
    Ok(())
}

#[tokio::test]
async fn test_a_call_cut_off_by_the_token_limit_comes_back_as_text() -> Result<()> {
    let checkpoint = scripted(&[WEATHER])?;
    let router = router(&checkpoint)?;
    let body = |limit: &str| {
        let mut body = json!({"messages": [{"role": "user", "content": "Weather? cue_weather"}],
            "tools": [weather_tool()]});
        body[limit] = json!(5);
        body
    };
    let reply = post_json(
        &router,
        "/v1/chat/completions",
        &body("max_completion_tokens"),
    )
    .await?;
    let choice = &reply["choices"][0];
    assert_eq!(choice["finish_reason"], "length");
    assert_eq!(
        choice["message"]["content"],
        "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n"
    );
    assert!(choice["message"]["tool_calls"].is_null());
    assert_eq!(reply["usage"]["completion_tokens"], 5);

    let mut body = body("max_output_tokens");
    body["input"] = body["messages"].take();
    body["tools"] = json!([{"type": "function", "name": "get_weather",
        "parameters": weather_tool()["function"]["parameters"]}]);
    let response = post_json(&router, "/v1/responses", &body).await?;
    assert_eq!(response["status"], "incomplete");
    assert_eq!(
        response["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(response["output"].as_array().map(Vec::len), Some(1));
    assert_eq!(response["output"][0]["type"], "message");
    Ok(())
}

/// The weather tool in the Responses API's flat form.
fn response_tools() -> Value {
    let chat = weather_tool();
    json!([{"type": "function", "name": "get_weather", "strict": true,
        "description": chat["function"]["description"],
        "parameters": chat["function"]["parameters"]}])
}

#[tokio::test]
async fn test_responses_agent_loop_round_trips_output_items() -> Result<()> {
    let checkpoint = scripted(&[WEATHER, ANSWER])?;
    let router = router(&checkpoint)?;
    let mut input = vec![json!({"role": "user", "content": [
        {"type": "input_text", "text": "What's the weather in Paris? cue_weather"}
    ]})];
    let request = |input: &[Value]| {
        json!({"model": "agent", "instructions": "Use tools when they help.", "input": input,
            "tools": response_tools(), "store": false, "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 64})
    };
    let first = post_json(&router, "/v1/responses", &request(&input)).await?;
    assert_eq!(first["object"], "response");
    assert_eq!(first["status"], "completed");
    assert!(
        first["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("resp_"))
    );
    let output = first["output"]
        .as_array()
        .ok_or_else(|| error("no output"))?;
    let kinds: Vec<&Value> = output.iter().map(|item| &item["type"]).collect();
    assert_eq!(kinds, ["message", "function_call"]);
    assert_eq!(output[0]["content"][0]["type"], "output_text");
    assert_eq!(output[0]["content"][0]["text"], "Checking.");
    assert_eq!(output[1]["name"], "get_weather");
    assert_eq!(arguments(&output[1])?, json!({"city": "Paris"}));
    let call_id = output[1]["call_id"].clone();
    assert_ne!(call_id, output[1]["id"]);
    assert_eq!(first["usage"]["output_tokens"], 9);

    // An agent replays the output items verbatim, then adds each call's result.
    input.extend(output.iter().cloned());
    input.push(json!({"type": "function_call_output", "call_id": call_id,
        "output": "{\"temperature\": 21, \"conditions\": \"sunny\"}"}));
    let second = post_json(&router, "/v1/responses", &request(&input)).await?;
    assert_eq!(second["status"], "completed");
    assert_eq!(second["output"].as_array().map(Vec::len), Some(1));
    assert_eq!(second["output"][0]["role"], "assistant");
    assert_eq!(second["output"][0]["content"][0]["text"], "It is sunny.");
    Ok(())
}

#[tokio::test]
async fn test_responses_stream_numbers_events_and_completes_with_the_streamed_items() -> Result<()>
{
    let checkpoint = scripted(&[WEATHER])?;
    let router = router(&checkpoint)?;
    let body = json!({"input": "Weather? cue_weather", "tools": response_tools(), "stream": true});
    let (status, text) = post(&router, "/v1/responses", &body).await?;
    assert_eq!(status, StatusCode::OK);
    let events = events(&text)?;
    for (at, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], name.as_str());
        assert_eq!(data["sequence_number"], at);
    }
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    let deltas = names
        .iter()
        .filter(|name| **name == "response.output_text.delta")
        .count();
    let mut expected = vec![
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ];
    expected.extend(std::iter::repeat_n("response.output_text.delta", deltas));
    expected.extend([
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]);
    assert_eq!(names, expected);
    let text_deltas: String = events
        .iter()
        .filter(|(name, _)| name == "response.output_text.delta")
        .filter_map(|(_, data)| data["delta"].as_str())
        .collect();
    assert_eq!(text_deltas, "Checking.");
    // The completed response holds the items as streamed, ids included.
    let done: Vec<&Value> = events
        .iter()
        .filter(|(name, _)| name == "response.output_item.done")
        .map(|(_, data)| &data["item"])
        .collect();
    let (_, completed) = events.last().ok_or_else(|| error("no events"))?;
    let response = &completed["response"];
    assert_eq!(response["status"], "completed");
    assert_eq!(response["id"], events[0].1["response"]["id"]);
    assert_eq!(
        response["output"]
            .as_array()
            .map(|items| items.iter().collect::<Vec<_>>()),
        Some(done)
    );
    assert_eq!(response["output"][1]["type"], "function_call");
    assert_eq!(arguments(&response["output"][1])?, json!({"city": "Paris"}));
    Ok(())
}

#[tokio::test]
async fn test_responses_refuses_what_a_stateless_server_cannot_honor() -> Result<()> {
    let checkpoint = scripted(&[ANSWER])?;
    let router = router(&checkpoint)?;
    let call = json!({"type": "function_call", "call_id": "call_1", "name": "get_weather",
        "arguments": "{\"city\":\"Paris\"}"});
    for (body, says) in [
        (
            json!({"input": "hi", "previous_response_id": "resp_1"}),
            "previous_response_id",
        ),
        (
            json!({"input": "hi", "tools": [{"type": "mcp", "server_label": "github",
                "server_url": "https://example.com/mcp"}]}),
            "hosted tools",
        ),
        (
            json!({"input": [{"type": "function_call_output", "call_id": "call_9",
                "output": "x"}]}),
            "answers no earlier function_call",
        ),
        (
            json!({"input": [call, {"type": "function_call_output", "call_id": "call_2",
                "output": "x"}]}),
            "answers no earlier function_call",
        ),
        (
            json!({"input": "hi", "tools": response_tools(),
                "tool_choice": {"type": "function", "name": "get_time"}}),
            "unknown tool",
        ),
        (
            json!({"input": "hi", "tools": response_tools(),
                "text": {"format": {"type": "json_object"}}}),
            "can't be combined with tools",
        ),
        (json!({"input": []}), "must not be empty"),
        (
            json!({"input": "hi", "reasoning": {"effort": "extreme"}}),
            "controls",
        ),
    ] {
        let (status, text) = post(&router, "/v1/responses", &body).await?;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {text}");
        let reply: Value = serde_json::from_str(&text)?;
        let message = reply["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains(says), "{body}: {message}");
    }
    Ok(())
}

#[tokio::test]
async fn test_reasoning_reaches_its_own_field_in_both_apis_and_never_the_answer() -> Result<()> {
    let checkpoint = scripted(&[THINK])?;
    let router = router(&checkpoint)?;
    let reasoning = "A city's weather needs a lookup.";
    let messages = json!([{"role": "user", "content": "Weather in Paris?"}]);
    let chat = json!({"messages": messages, "tools": [weather_tool()], "thinking": true});
    let reply = post_json(&router, "/v1/chat/completions", &chat).await?;
    let message = &reply["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], reasoning);
    assert!(message["content"].is_null(), "{message}");
    assert_eq!(
        arguments(&message["tool_calls"][0])?,
        json!({"city": "Paris"})
    );

    let mut stream = chat.clone();
    stream["stream"] = json!(true);
    let (_, text) = post(&router, "/v1/chat/completions", &stream).await?;
    let chunks = events(&text)?;
    let field = |name: &str| -> String {
        chunks
            .iter()
            .filter_map(|(_, chunk)| chunk["choices"][0]["delta"][name].as_str())
            .collect()
    };
    assert_eq!(field("reasoning_content"), reasoning);
    assert_eq!(field("content"), "");
    assert!(!text.contains("think>"), "{text}");

    let responses = json!({"input": messages, "tools": response_tools(), "stream": true,
        "enable_thinking": true});
    let (_, text) = post(&router, "/v1/responses", &responses).await?;
    let events = events(&text)?;
    let (_, completed) = events.last().ok_or_else(|| error("no events"))?;
    let output = &completed["response"]["output"];
    let kinds: Vec<&Value> = output
        .as_array()
        .into_iter()
        .flatten()
        .map(|item| &item["type"])
        .collect();
    assert_eq!(kinds, ["reasoning", "function_call"], "{output}");
    assert_eq!(output[0]["content"][0]["text"], reasoning);
    let deltas: String = events
        .iter()
        .filter(|(name, _)| name == "response.reasoning_text.delta")
        .filter_map(|(_, data)| data["delta"].as_str())
        .collect();
    assert_eq!(deltas, reasoning);
    Ok(())
}

fn real_router() -> Result<axum::Router> {
    use tur::{
        ModelFactory, ModelSource,
        models::{DEFAULT_KV_LAYOUT, WeightFormat},
    };
    let path = std::env::var_os("TUR_QWEN35_0_8B").ok_or_else(|| error("set TUR_QWEN35_0_8B"))?;
    let worker = PipelineWorker::spawn(WorkerConfig {
        factory: ModelFactory {
            source: ModelSource::LocalPath(path.into()),
            revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
            format: Some(WeightFormat::Q8_0),
            device: candle_core::Device::new_metal(0)?,
            kv: DEFAULT_KV_LAYOUT,
            recurrent: candle_core::DType::F32,
            audio: false,
            vision: false,
            video: false,
            expert_cache: None,
            gguf: None,
            sparse: None,
        },
        pipeline: PipelineConfig::new(2, 256),
        budget: DeviceBudget::new(None),
        request_timeout: None,
        stream: StreamConfig::default(),
    })?;
    Ok(build_router(AppState {
        status: tur::server::ServerStatus::shared(),
        workers: HashMap::from([("qwen".into(), worker)]),
        transcribers: HashMap::new(),
        default_transcriber: None,
        default_model: "qwen".into(),
    }))
}

const ASK: &str = "What's the weather in Paris right now?";
const RESULT: &str = "{\"temperature\": 21, \"conditions\": \"sunny\"}";

#[tokio::test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint and host GPU access"]
async fn test_real_checkpoint_completes_a_chat_tool_loop_without_markup() -> Result<()> {
    let router = real_router()?;
    let mut messages = vec![json!({"role": "user", "content": ASK})];
    let body = |messages: &[Value], stream: bool| {
        json!({"messages": messages, "tools": [weather_tool()], "temperature": 0,
            "max_completion_tokens": 256, "stream": stream})
    };
    let reply = post_json(&router, "/v1/chat/completions", &body(&messages, false)).await?;
    let message = &reply["choices"][0]["message"];
    assert_eq!(
        reply["choices"][0]["finish_reason"], "tool_calls",
        "{reply}"
    );
    assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
    let city = arguments(&message["tool_calls"][0])?["city"].clone();
    assert!(city.as_str().is_some_and(|c| c.contains("Paris")), "{city}");

    let (_, text) = post(&router, "/v1/chat/completions", &body(&messages, true)).await?;
    let streamed: String = events(&text)?
        .iter()
        .filter_map(|(_, chunk)| chunk["choices"][0]["delta"]["content"].as_str())
        .map(str::to_owned)
        .collect();
    assert!(
        !streamed.contains("<tool_call>") && !streamed.contains("<function="),
        "{streamed}"
    );
    assert!(text.contains("\"tool_calls\""), "{text}");

    messages.push(message.clone());
    let id = message["tool_calls"][0]["id"].clone();
    messages.push(json!({"role": "tool", "tool_call_id": id, "content": RESULT}));
    let reply = post_json(&router, "/v1/chat/completions", &body(&messages, false)).await?;
    let answer = reply["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert_eq!(reply["choices"][0]["finish_reason"], "stop", "{reply}");
    assert!(
        answer.contains("21") || answer.to_lowercase().contains("sunny"),
        "{answer}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint and host GPU access"]
async fn test_real_checkpoint_completes_a_responses_tool_loop() -> Result<()> {
    let router = real_router()?;
    let mut input = vec![json!({"role": "user", "content": ASK})];
    let body = |input: &[Value]| {
        json!({"input": input, "tools": response_tools(), "temperature": 0,
            "max_output_tokens": 256})
    };
    let first = post_json(&router, "/v1/responses", &body(&input)).await?;
    let output = first["output"]
        .as_array()
        .ok_or_else(|| error("no output"))?;
    let call = output
        .iter()
        .find(|item| item["type"] == "function_call")
        .ok_or_else(|| error(format!("no call in {first}")))?;
    assert_eq!(call["name"], "get_weather");
    assert!(
        arguments(call)?["city"]
            .as_str()
            .is_some_and(|c| c.contains("Paris"))
    );
    let call_id = call["call_id"].clone();
    input.extend(output.iter().cloned());
    input.push(json!({"type": "function_call_output", "call_id": call_id, "output": RESULT}));
    let second = post_json(&router, "/v1/responses", &body(&input)).await?;
    let text = second["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item["type"] == "message")
        .filter_map(|item| item["content"][0]["text"].as_str())
        .collect::<String>();
    assert_eq!(second["status"], "completed", "{second}");
    assert!(
        text.contains("21") || text.to_lowercase().contains("sunny"),
        "{text}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint and host GPU access"]
async fn test_real_checkpoint_keeps_thinking_out_of_the_answer() -> Result<()> {
    let router = real_router()?;
    let body = json!({"messages": [{"role": "user", "content": "What is 17 times 3?"}],
        "thinking": true, "temperature": 0, "max_completion_tokens": 512});
    let reply = post_json(&router, "/v1/chat/completions", &body).await?;
    let message = &reply["choices"][0]["message"];
    assert_eq!(reply["choices"][0]["finish_reason"], "stop", "{reply}");
    let content = message["content"].as_str().unwrap_or_default();
    assert!(
        !content.contains("think>") && content.contains("51"),
        "{message}"
    );
    assert!(
        message["reasoning_content"]
            .as_str()
            .is_some_and(|r| !r.is_empty() && !r.contains("think>")),
        "{message}"
    );
    Ok(())
}
