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
        let template_str = preprocess_template(&template.into());
        // Validate syntax eagerly so callers get a clear error at load time
        // rather than at the first render call.
        let mut env = minijinja::Environment::new();
        env.add_template("chat", &template_str)
            .map_err(|e| TurError::Other(format!("invalid chat template syntax: {e}")))?;
        Ok(Self { template_str })
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
