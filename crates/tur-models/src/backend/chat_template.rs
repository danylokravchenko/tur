use serde::Serialize;
use std::path::Path;

use crate::backend::tools::ToolDefinition;
use crate::{Result, TurError};

/// Renders a checkpoint's own Hugging Face-compatible Jinja2 chat template.
pub struct ChatTemplate {
    source: String,
}

/// Message author for [`ChatTemplate::render`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A function call emitted by the assistant; arguments keep their insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

impl Serialize for ToolCall {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serde_json::json!({
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        })
        .serialize(serializer)
    }
}

/// A conversation turn with the optional fields Hugging Face chat templates read.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    /// A plain text turn.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
        }
    }
}

/// Template variables. `None` leaves a variable undefined, which templates distinguish
/// from `false` (Qwen3.8 thinks by default when `enable_thinking` is undefined).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub preserve_thinking: Option<bool>,
}

impl ChatTemplate {
    /// Load a chat template from a model's `tokenizer_config.json`.
    ///
    /// Expects the file to contain a top-level `"chat_template"` string field.
    pub fn from_tokenizer_config(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let config: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| TurError::Other(format!("tokenizer_config.json parse error: {e}")))?;
        let template_str = config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TurError::Other(
                    "tokenizer_config.json is missing the 'chat_template' field".to_string(),
                )
            })?
            .to_string();
        Self::from_template(template_str)
    }

    /// Build a `ChatTemplate` from a raw Jinja2 template string.
    ///
    /// The template is validated at construction time; returns an error if
    /// the syntax is invalid.
    pub fn from_template(template: impl Into<String>) -> Result<Self> {
        let source = template.into();
        faithful_environment()
            .add_template("chat", &source)
            .map_err(|e| TurError::Other(format!("invalid chat template syntax: {e}")))?;
        Ok(Self { source })
    }

    /// Load a standalone `chat_template.jinja`, falling back to `tokenizer_config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let jinja = dir.join("chat_template.jinja");
        if jinja.exists() {
            return Self::from_template(std::fs::read_to_string(jinja)?);
        }
        Self::from_tokenizer_config(&dir.join("tokenizer_config.json"))
    }

    /// Render exactly as Hugging Face `apply_chat_template` does: Jinja2 `trim_blocks` and
    /// `lstrip_blocks`, Python string methods, `json.dumps`-compatible `tojson`, and
    /// undefined (not `false`) variables for unset options.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        options: &RenderOptions,
    ) -> Result<String> {
        let mut env = faithful_environment();
        env.add_template("chat", &self.source)
            .map_err(|e| TurError::Other(format!("chat template compile error: {e}")))?;
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), serde_json::to_value(messages)?);
        ctx.insert(
            "add_generation_prompt".into(),
            options.add_generation_prompt.into(),
        );
        if !tools.is_empty() {
            let wrapped: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            ctx.insert("tools".into(), wrapped.into());
        }
        if let Some(value) = options.enable_thinking {
            ctx.insert("enable_thinking".into(), value.into());
        }
        if let Some(value) = &options.reasoning_effort {
            ctx.insert("reasoning_effort".into(), value.clone().into());
        }
        if let Some(value) = options.preserve_thinking {
            ctx.insert("preserve_thinking".into(), value.into());
        }
        env.get_template("chat")
            .and_then(|t| {
                t.render(minijinja::Value::from_serialize(serde_json::Value::Object(
                    ctx,
                )))
            })
            .map_err(|e| TurError::Other(format!("chat template render error: {e}")))
    }
}

/// Registered as a Jinja2 global so templates that call `raise_exception(msg)`
/// produce a descriptive render error rather than a silent undefined call.
fn raise_exception(msg: String) -> std::result::Result<minijinja::Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// Environment matching Hugging Face's sandboxed Jinja2 setup for chat templates.
fn faithful_environment() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_function("raise_exception", raise_exception);
    env.add_filter("tojson", python_tojson);
    env.set_unknown_method_callback(python_method);
    env
}

fn template_error(message: impl Into<String>) -> minijinja::Error {
    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message.into())
}

/// Python `str` methods used by chat templates, with Python semantics.
fn python_method(
    _: &minijinja::State<'_, '_>,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let Some(text) = value.as_str() else {
        return Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod));
    };
    let arg = |i: usize| args.get(i).and_then(|a| a.as_str());
    let strip = |chars: Option<&str>, start: bool, end: bool| -> String {
        let keep = |c: char| chars.map_or_else(|| c.is_whitespace(), |set| set.contains(c));
        let mut s = text;
        if start {
            s = s.trim_start_matches(keep);
        }
        if end {
            s = s.trim_end_matches(keep);
        }
        s.to_string()
    };
    Ok(match method {
        "startswith" => arg(0)
            .map(|p| text.starts_with(p))
            .ok_or_else(|| template_error("startswith needs a string"))?
            .into(),
        "endswith" => arg(0)
            .map(|p| text.ends_with(p))
            .ok_or_else(|| template_error("endswith needs a string"))?
            .into(),
        "split" => {
            let parts: Vec<String> = match arg(0) {
                Some("") => return Err(template_error("empty separator")),
                Some(sep) => text.split(sep).map(String::from).collect(),
                None => text.split_whitespace().map(String::from).collect(),
            };
            parts.into()
        }
        "strip" => strip(arg(0), true, true).into(),
        "lstrip" => strip(arg(0), true, false).into(),
        "rstrip" => strip(arg(0), false, true).into(),
        "lower" => text.to_lowercase().into(),
        "upper" => text.to_uppercase().into(),
        "replace" => match (arg(0), arg(1)) {
            (Some(from), Some(to)) => text.replace(from, to).into(),
            _ => return Err(template_error("replace needs two strings")),
        },
        _ => return Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)),
    })
}

/// `json.dumps(value, ensure_ascii=False)` separators: `", "` and `": "`.
struct PythonJson;

impl serde_json::ser::Formatter for PythonJson {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

fn python_tojson(
    value: &minijinja::Value,
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonJson);
    value
        .serialize(&mut serializer)
        .map_err(|e| template_error(format!("tojson: {e}")))?;
    let text = String::from_utf8(out).map_err(|e| template_error(format!("tojson: {e}")))?;
    Ok(minijinja::Value::from_safe_string(text))
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_unset_thinking_differs_from_explicit_false() -> Result<()> {
        let template = ChatTemplate::from_template(
            "{% if enable_thinking is not defined %}default{% elif enable_thinking %}yes{% else %}no{% endif %}",
        )?;
        assert_eq!(
            template.render(&[], &[], &RenderOptions::default())?,
            "default"
        );
        assert_eq!(
            template.render(
                &[],
                &[],
                &RenderOptions {
                    enable_thinking: Some(false),
                    ..Default::default()
                }
            )?,
            "no"
        );
        Ok(())
    }

    #[test]
    fn test_invalid_template_is_rejected() {
        assert!(ChatTemplate::from_template("{% unclosed block").is_err());
    }
}
