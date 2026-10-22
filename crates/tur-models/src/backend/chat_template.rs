use serde::Serialize;
use std::path::Path;

use crate::backend::tools::ToolDefinition;
use crate::{Result, TurError};

/// A single turn in a conversation.
///
/// Constructed with [`Message::user`], [`Message::system`], or
/// [`Message::assistant`] and serialised directly into the Jinja2 template
/// context as `{"role": "...", "content": "..."}`.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Renders chat prompts from a HuggingFace-compatible Jinja2 template.
///
/// Load from a model's `tokenizer_config.json` (which contains the
/// `chat_template` field) or from a raw template string.  Once constructed,
/// call [`ChatTemplate::format`] to render a conversation into a tokenisable
/// string.
///
/// The template receives the following variables:
///
/// | Variable | Type | Description |
/// |---|---|---|
/// | `messages` | `[{role, content}]` | Conversation history |
/// | `tools` | `[{type, function: {name, description, parameters}}]` | Available tools (absent when none) |
/// | `add_generation_prompt` | `bool` | Append model generation opener |
/// | `enable_thinking` | `bool` | Qwen3 chain-of-thought `/think` mode |
///
/// Standard Jinja2 built-ins (`tojson`, `namespace`, `raise_exception`, …)
/// are registered automatically.
pub struct ChatTemplate {
    template_str: String,
    /// Unmodified template source for the Jinja2-faithful [`ChatTemplate::render`] path.
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
        let template_str = preprocess_template(&source);
        // Validate syntax eagerly so callers get a clear error at load time
        // rather than at the first render call.
        let mut env = minijinja::Environment::new();
        env.add_template("chat", &template_str)
            .map_err(|e| TurError::Other(format!("invalid chat template syntax: {e}")))?;
        faithful_environment()
            .add_template("chat", &source)
            .map_err(|e| TurError::Other(format!("invalid chat template syntax: {e}")))?;
        Ok(Self {
            template_str,
            source,
        })
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

    /// Render the template into a prompt string ready for tokenisation.
    ///
    /// # Arguments
    /// * `messages` — Conversation history; must contain at least the current user turn.
    /// * `tools` — Optional tool definitions; absent from the template context when `None`.
    /// * `add_generation_prompt` — Append the model's generation opener (e.g.
    ///   `<|im_start|>assistant\n`) when `true`.
    /// * `enable_thinking` — Pass `enable_thinking=true` to the template
    ///   (controls Qwen3's `/think` / `/no_think` tags).
    pub fn format(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String> {
        // Build a fresh environment per call.  Template compilation is cheap
        // compared to the model forward pass (< 1 ms vs hundreds of ms).
        let mut env = minijinja::Environment::new();

        // `raise_exception` is called by some HuggingFace templates to reject
        // unsupported configurations (e.g. system messages in tool-call mode).
        env.add_function("raise_exception", raise_exception);

        // Python string methods that HuggingFace templates rely on but minijinja
        // does not provide natively.  The template preprocessor rewrites method
        // call syntax (`.startswith(…)`) to filter syntax (`| startswith(…)`)
        // so these filters are always registered.
        env.add_filter("startswith", |s: &str, prefix: &str| s.starts_with(prefix));
        env.add_filter("endswith", |s: &str, suffix: &str| s.ends_with(suffix));
        env.add_filter("split_first", |s: &str, sep: &str| {
            s.split(sep).next().unwrap_or(s).to_string()
        });
        env.add_filter("split_last", |s: &str, sep: &str| {
            s.rsplit(sep).next().unwrap_or(s).to_string()
        });
        env.add_filter("py_rstrip", |s: &str, chars: &str| -> String {
            let set: Vec<char> = chars.chars().collect();
            s.trim_end_matches(|c| set.contains(&c)).to_string()
        });
        env.add_filter("py_lstrip", |s: &str, chars: &str| -> String {
            let set: Vec<char> = chars.chars().collect();
            s.trim_start_matches(|c| set.contains(&c)).to_string()
        });
        env.add_filter("py_strip", |s: &str, chars: &str| -> String {
            let set: Vec<char> = chars.chars().collect();
            s.trim_matches(|c| set.contains(&c)).to_string()
        });

        env.add_template("chat", &self.template_str)
            .map_err(|e| TurError::Other(format!("chat template compile error: {e}")))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| TurError::Other(format!("chat template lookup error: {e}")))?;

        // Serialise the context as a serde_json map so every value (arrays,
        // objects, booleans) round-trips through the same serde path and
        // minijinja can introspect types correctly.
        let mut ctx = serde_json::Map::new();

        ctx.insert(
            "messages".to_string(),
            serde_json::to_value(messages)
                .map_err(|e| TurError::Other(format!("failed to serialise messages: {e}")))?,
        );
        ctx.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        ctx.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(enable_thinking),
        );

        if let Some(ts) = tools {
            // Wrap each tool in the {"type":"function","function":{…}} envelope
            // that HuggingFace templates expect.
            let wrapped: Vec<serde_json::Value> = ts
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
            ctx.insert("tools".to_string(), serde_json::Value::Array(wrapped));
        }

        let ctx_val = minijinja::Value::from_serialize(serde_json::Value::Object(ctx));

        tmpl.render(ctx_val)
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
    _: &minijinja::State,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let Some(text) = value.as_str() else {
        return Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod));
    };
    let arg = |i: usize| args.get(i).and_then(|a| a.as_str());
    let strip = |chars: Option<&str>, start: bool, end: bool| -> String {
        let keep = |c: char| match chars {
            Some(set) => set.contains(c),
            None => c.is_whitespace(),
        };
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

/// Convert Python string method calls to equivalent minijinja filter chains.
///
/// HuggingFace Jinja2 templates use Python-style methods (`.startswith()`,
/// `.split()[0]`, `.rstrip()`, etc.) that minijinja does not support on plain
/// string values.  This function rewrites those method calls to the registered
/// custom filters, which minijinja evaluates correctly.
///
/// Ordering matters: longer/overlapping patterns are replaced before shorter
/// ones to avoid partial matches (e.g. `.rstrip` before `.strip`).
fn preprocess_template(template: &str) -> String {
    // Replace .split('X')[0] and .split('X')[-1] before the simpler strip
    // variants so that the chain A.split(X)[0].rstrip(Y) becomes
    // A | split_first(X) | py_rstrip(Y) after all passes.
    //
    // Pass order:
    //   1. rstrip / lstrip (before plain strip to avoid substring collision)
    //   2. strip
    //   3. split(…)[-1]  (before split(…)[0] so [-1] is handled first)
    //   4. split(…)[0]
    //   5. startswith / endswith
    template
        .replace(".rstrip(", " | py_rstrip(")
        .replace(".lstrip(", " | py_lstrip(")
        .replace(".strip(", " | py_strip(")
        .replace(".split('</think>')[0]", " | split_first('</think>')")
        .replace(".split('</think>')[-1]", " | split_last('</think>')")
        .replace(".split('<think>')[-1]", " | split_last('<think>')")
        .replace(".split(\"</think>\")[0]", " | split_first('</think>')")
        .replace(".split(\"</think>\")[-1]", " | split_last('</think>')")
        .replace(".split(\"<think>\")[-1]", " | split_last('<think>')")
        .replace(".startswith(", " | startswith(")
        .replace(".endswith(", " | endswith(")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_TEMPLATE: &str = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:\n{% endif %}";

    #[test]
    fn format_user_message() {
        let ct = ChatTemplate::from_template(SIMPLE_TEMPLATE).expect("valid template");
        let messages = vec![Message::user("Hello")];
        let out = ct.format(&messages, None, true, false).expect("render ok");
        assert!(out.contains("user: Hello"));
        assert!(out.contains("assistant:"));
    }

    #[test]
    fn format_without_generation_prompt() {
        let ct = ChatTemplate::from_template(SIMPLE_TEMPLATE).expect("valid template");
        let messages = vec![Message::user("Hi")];
        let out = ct.format(&messages, None, false, false).expect("render ok");
        assert!(!out.contains("assistant:"));
    }

    #[test]
    fn invalid_template_syntax_is_rejected() {
        let result = ChatTemplate::from_template("{% unclosed block");
        assert!(result.is_err(), "invalid syntax must return an error");
    }

    #[test]
    fn tools_appear_in_context() {
        let template = "{% if tools %}has_tools{% else %}no_tools{% endif %}";
        let ct = ChatTemplate::from_template(template).expect("valid template");
        let tool = ToolDefinition::new(
            "ping",
            "Ping",
            serde_json::json!({"type":"object","properties":{},"required":[]}),
        );
        let with_tools = ct
            .format(&[Message::user("hi")], Some(&[tool]), false, false)
            .expect("render ok");
        let without_tools = ct
            .format(&[Message::user("hi")], None, false, false)
            .expect("render ok");
        assert_eq!(with_tools.trim(), "has_tools");
        assert_eq!(without_tools.trim(), "no_tools");
    }

    #[test]
    fn enable_thinking_propagated() {
        let template = "{% if enable_thinking %}think{% else %}nothink{% endif %}";
        let ct = ChatTemplate::from_template(template).expect("valid template");
        let msgs = vec![Message::user("x")];
        assert_eq!(
            ct.format(&msgs, None, false, true).expect("ok").trim(),
            "think"
        );
        assert_eq!(
            ct.format(&msgs, None, false, false).expect("ok").trim(),
            "nothink"
        );
    }

    #[test]
    fn python_string_methods_work_in_template() {
        // The Qwen3 template uses .startswith() and .endswith() on message content.
        let template =
            "{% if messages[0].content | startswith('Hello') %}yes{% else %}no{% endif %}";
        let ct = ChatTemplate::from_template(
            "{% if messages[0].content.startswith('Hello') %}yes{% else %}no{% endif %}",
        )
        .expect("valid template");
        let msgs = vec![Message::user("Hello world")];
        assert_eq!(ct.format(&msgs, None, false, false).expect("ok"), "yes");
        let msgs2 = vec![Message::user("Goodbye")];
        assert_eq!(ct.format(&msgs2, None, false, false).expect("ok"), "no");

        // Verify the preprocessed form also works directly.
        let ct2 = ChatTemplate::from_template(template).expect("valid template");
        assert_eq!(ct2.format(&msgs, None, false, false).expect("ok"), "yes");
    }

    #[test]
    fn split_first_and_last_filters_work() {
        let template = concat!(
            "{%- set before = text | split_first('</think>') %}\n",
            "{%- set after = text | split_last('</think>') %}\n",
            "before={{ before }},after={{ after }}"
        );
        let _ct = ChatTemplate::from_template(template).expect("valid template");
        let mut ctx = serde_json::Map::new();
        ctx.insert(
            "text".to_string(),
            serde_json::Value::String("think\n<think>content</think>\nresult".to_string()),
        );
        ctx.insert("messages".to_string(), serde_json::Value::Array(vec![]));
        ctx.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(false),
        );
        ctx.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        // Use format directly with a custom context to test the filters.
        let mut env = minijinja::Environment::new();
        env.add_filter("split_first", |s: &str, sep: &str| {
            s.split(sep).next().unwrap_or(s).to_string()
        });
        env.add_filter("split_last", |s: &str, sep: &str| {
            s.rsplit(sep).next().unwrap_or(s).to_string()
        });
        env.add_template("t", template).expect("ok");
        let out = env
            .get_template("t")
            .expect("ok")
            .render(minijinja::Value::from_serialize(serde_json::Value::Object(
                ctx,
            )))
            .expect("render ok");
        assert!(out.contains("before=think\n<think>content"), "got: {out}");
        assert!(out.contains("after=\nresult"), "got: {out}");
    }
}
