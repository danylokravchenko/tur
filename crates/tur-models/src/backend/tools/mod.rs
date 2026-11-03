mod gemma;
use serde::{Deserialize, Serialize};

/// How a model writes tool calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolSyntax {
    /// `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    Xml,
    /// `<|tool_call>call:name{key:value}<tool_call|>`
    Gemma,
    /// `<|channel|>commentary to=functions.name <|constrain|>json<|message|>{...}`
    Harmony,
}

/// Definition of a callable tool (function) that the model can invoke.
///
/// Passed to the checkpoint chat template when rendering a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name the model uses when calling this tool.
    pub name: String,
    /// Human-readable description that guides the model on when/how to call it.
    pub description: String,
    /// JSON-Schema `object` describing the tool's parameters.
    /// Build it with [`serde_json::json!`].
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A tool call parsed from the model's generated output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Name of the tool being called.
    pub name: String,
    /// Arguments as a JSON object.
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Parse complete Gemma calls and Qwen hybrid XML calls, using the tool schema to preserve string
    /// parameters and decode structured values. JSON call envelopes are also accepted.
    /// Malformed or incomplete blocks are ignored.
    pub fn parse_from_output(text: &str, tools: &[ToolDefinition]) -> Vec<Self> {
        let mut calls = Vec::new();
        let mut remaining = text;
        loop {
            let markers = [
                ("<tool_call>", "</tool_call>"),
                ("<|tool_call>", "<tool_call|>"),
            ];
            let Some((at, open, close)) = markers
                .iter()
                .filter_map(|&(open, close)| remaining.find(open).map(|at| (at, open, close)))
                .min_by_key(|(at, _, _)| *at)
            else {
                break;
            };
            let Some(after) = remaining.get(at + open.len()..) else {
                break;
            };
            let Some((body, rest)) = after.split_once(close) else {
                break;
            };
            let body = body.trim();
            let call = if open == "<|tool_call>" {
                gemma::parse(body, tools)
            } else {
                serde_json::from_str::<Self>(body)
                    .ok()
                    .or_else(|| parse_xml_call(body, tools))
            };
            if let Some(call) = call {
                calls.push(call);
            }
            remaining = rest;
        }

        calls
    }
}

fn parse_xml_call(text: &str, tools: &[ToolDefinition]) -> Option<ToolCall> {
    let (name, body) = text.strip_prefix("<function=")?.split_once('>')?;
    let tool = tools.iter().find(|tool| tool.name == name)?;
    let mut body = body.trim().strip_suffix("</function>")?.trim();
    let mut arguments = serde_json::Map::new();
    while !body.is_empty() {
        let (parameter, rest) = body.strip_prefix("<parameter=")?.split_once('>')?;
        if parameter.is_empty() {
            return None;
        }
        let (value, rest) = rest.split_once("</parameter>")?;
        let value = xml_value(value.trim(), &tool.parameters["properties"][parameter])?;
        if arguments.insert(parameter.to_string(), value).is_some() {
            return None;
        }
        body = rest.trim();
    }
    Some(ToolCall {
        name: name.to_string(),
        arguments: arguments.into(),
    })
}

/// The JSON types `schema` admits, from `type` (a name or a list) or the alternatives of
/// `anyOf` and `oneOf`, as MCP servers often describe optional parameters. Empty when it
/// names none.
fn schema_types(schema: &serde_json::Value) -> Vec<&str> {
    let mut types: Vec<&str> = match &schema["type"] {
        serde_json::Value::String(name) => vec![name.as_str()],
        serde_json::Value::Array(names) => names.iter().filter_map(|n| n.as_str()).collect(),
        _ => Vec::new(),
    };
    for key in ["anyOf", "oneOf"] {
        for alternative in schema[key].as_array().into_iter().flatten() {
            types.extend(schema_types(alternative));
        }
    }
    types
}

/// One parameter's text as its value. A parameter that can't be a string must be JSON; one
/// that can be either is JSON when it reads as one of its other types, and text otherwise.
fn xml_value(text: &str, schema: &serde_json::Value) -> Option<serde_json::Value> {
    let types = schema_types(schema);
    let text_value = || serde_json::Value::String(text.to_string());
    if types.iter().all(|name| *name == "string") {
        return Some(text_value());
    }
    let parsed = serde_json::from_str::<serde_json::Value>(text).ok();
    if !types.contains(&"string") {
        return parsed;
    }
    Some(
        parsed
            .filter(|value| !value.is_string())
            .unwrap_or_else(text_value),
    )
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert tool-call parsing contracts"
)]
mod tests {
    use super::*;
    use crate::Result;

    #[test]
    fn test_json_calls_preserve_names_arguments_order_and_skip_malformed_blocks() -> Result<()> {
        for (text, expected) in [
            (
                r#"Sure, let me check the weather.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call>"#,
                serde_json::json!([{
                    "name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}
                }]),
            ),
            (
                r#"<tool_call>
{"name": "search", "arguments": {"query": "rust"}}
</tool_call>
<tool_call>
{"name": "fetch", "arguments": {"url": "https://example.com"}}
</tool_call>"#,
                serde_json::json!([
                    {"name": "search", "arguments": {"query": "rust"}},
                    {"name": "fetch", "arguments": {"url": "https://example.com"}}
                ]),
            ),
            (
                "Just a normal response with no tool calls.",
                serde_json::json!([]),
            ),
            (
                r#"<tool_call>not json</tool_call>
<tool_call>{"name": "ok", "arguments": {}}</tool_call>"#,
                serde_json::json!([{"name": "ok", "arguments": {}}]),
            ),
        ] {
            assert_eq!(
                serde_json::to_value(ToolCall::parse_from_output(text, &[]))?,
                expected,
                "{text}"
            );
        }
        Ok(())
    }
    #[test]
    fn test_xml_calls_preserve_schema_types_and_reject_incomplete_calls() {
        let tools = [ToolDefinition::new(
            "lookup",
            "",
            serde_json::json!({"properties": {
                "id": {"type":"string"}, "count": {"type":"integer"}, "data": {"type":"object"}
            }}),
        )];
        let text = "<tool_call><function=lookup>\n<parameter=id>\n123\n</parameter>\n<parameter=count>2</parameter><parameter=data>{\"ok\":true}</parameter></function></tool_call>";
        let calls = ToolCall::parse_from_output(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["id"], "123");
        assert_eq!(calls[0].arguments["count"], 2);
        assert_eq!(calls[0].arguments["data"]["ok"], true);
        assert!(
            ToolCall::parse_from_output(
                "<tool_call><function=lookup><parameter=id>unfinished",
                &tools
            )
            .is_empty()
        );
    }

    #[test]
    fn test_xml_union_types_decode_their_non_string_alternatives() {
        let tools = [ToolDefinition::new(
            "f",
            "",
            serde_json::json!({"properties": {
                "milestone": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "note": {"type": ["string", "null"]},
                "tag": {"oneOf": [{"type": "string"}, {"type": "array"}]},
                "plain": {"description": "no type at all"}
            }}),
        )];
        let call = |parameter: &str, value: &str| {
            let text = format!(
                "<tool_call><function=f><parameter={parameter}>{value}</parameter></function></tool_call>"
            );
            ToolCall::parse_from_output(&text, &tools)
                .first()
                .map(|call| call.arguments[parameter].clone())
        };
        assert_eq!(call("milestone", "7"), Some(serde_json::json!(7)));
        assert_eq!(call("milestone", "null"), Some(serde_json::Value::Null));
        // Neither an integer nor null: not a valid call.
        assert_eq!(call("milestone", "soon"), None);
        assert_eq!(call("note", "null"), Some(serde_json::Value::Null));
        assert_eq!(call("note", "hello"), Some(serde_json::json!("hello")));
        // A string alternative keeps text that happens to be quoted JSON as written.
        assert_eq!(call("tag", "\"v1\""), Some(serde_json::json!("\"v1\"")));
        assert_eq!(call("tag", "[\"a\"]"), Some(serde_json::json!(["a"])));
        assert_eq!(call("plain", "42"), Some(serde_json::json!("42")));
    }
}
