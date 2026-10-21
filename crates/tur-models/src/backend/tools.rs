use serde::{Deserialize, Serialize};

/// Definition of a callable tool (function) that the model can invoke.
///
/// Pass a slice of these to [`GenerationRequest::with_tools`]; the pipeline
/// injects them into the prompt via the model's `format_prompt_with_tools`
/// implementation and then parses any [`ToolCall`]s from the generated text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name the model uses when calling this tool.
    pub name: String,
    /// Human-readable description that guides the model on when/how to call it.
    pub description: String,
    /// JSON-Schema `object` describing the tool's parameters.
    /// Build it with [`serde_json::json!`] or [`ToolParameters`].
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
    /// Scan `text` for `<tool_call>…</tool_call>` blocks (Qwen3 format) and
    /// return all successfully parsed calls in order.
    pub fn parse_from_output(text: &str) -> Vec<Self> {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";

        let mut calls = Vec::new();
        let mut remaining = text;

        while let Some(start) = remaining.find(OPEN) {
            let after_open = &remaining[start + OPEN.len()..];
            match after_open.find(CLOSE) {
                Some(end) => {
                    let json_str = after_open[..end].trim();
                    if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                        calls.push(call);
                    }
                    remaining = &after_open[end + CLOSE.len()..];
                }
                None => break,
            }
        }

        calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tool_call() {
        let text = r#"Sure, let me check the weather.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call>"#;

        let calls = ToolCall::parse_from_output(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "Paris");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let text = r#"<tool_call>
{"name": "search", "arguments": {"query": "rust"}}
</tool_call>
<tool_call>
{"name": "fetch", "arguments": {"url": "https://example.com"}}
</tool_call>"#;

        let calls = ToolCall::parse_from_output(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "fetch");
    }

    #[test]
    fn parse_no_tool_calls() {
        let calls = ToolCall::parse_from_output("Just a normal response with no tool calls.");
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_malformed_json_skipped() {
        let text = r#"<tool_call>not json</tool_call>
<tool_call>{"name": "ok", "arguments": {}}</tool_call>"#;

        let calls = ToolCall::parse_from_output(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ok");
    }
}
