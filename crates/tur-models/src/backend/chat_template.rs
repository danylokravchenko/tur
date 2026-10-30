use serde::Serialize;
use std::{fmt::Write as _, path::Path};

use crate::backend::tools::ToolDefinition;
use crate::{Result, TurError, audio::Audio};

/// Renders a checkpoint's own Hugging Face-compatible Jinja2 chat template.
pub struct ChatTemplate {
    source: String,
    tokens: serde_json::Map<String, serde_json::Value>,
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

/// One typed piece of a turn, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    Text(String),
    /// A clip, rendered as Hugging Face's `{"type": "audio"}` part; the template emits
    /// the model's audio placeholder for it.
    Audio(Audio),
}

/// A turn's content: plain text, or typed parts as Hugging Face templates receive them.
/// Text serializes as a string, so text-only templates see exactly what they expect.
#[derive(Debug, Clone, PartialEq)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Content {
    /// The text when the content holds nothing else.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(_) => None,
        }
    }

    /// Clips in the order the template renders their placeholders.
    pub fn audio(&self) -> impl Iterator<Item = &Audio> {
        let parts = match self {
            Self::Text(_) => &[][..],
            Self::Parts(parts) => parts,
        };
        parts.iter().filter_map(|part| match part {
            ContentPart::Audio(clip) => Some(clip),
            ContentPart::Text(_) => None,
        })
    }
}

impl Serialize for Content {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Text(text) => serializer.serialize_str(text),
            Self::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text(text) => serde_json::json!({"type": "text", "text": text}),
                    ContentPart::Audio(_) => serde_json::json!({"type": "audio"}),
                })
                .collect::<Vec<_>>()
                .serialize(serializer),
        }
    }
}

/// A conversation turn with the optional fields Hugging Face chat templates read.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Content,
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
            content: Content::Text(content.into()),
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
        Self::from_template(template_str)?.with_tokens(&config)
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
        Ok(Self {
            source,
            tokens: serde_json::Map::new(),
        })
    }

    /// Load a standalone `chat_template.jinja`, falling back to `tokenizer_config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let jinja = dir.join("chat_template.jinja");
        if jinja.exists() {
            let config: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
            return Self::from_template(std::fs::read_to_string(jinja)?)?.with_tokens(&config);
        }
        Self::from_tokenizer_config(&dir.join("tokenizer_config.json"))
    }

    fn with_tokens(mut self, config: &serde_json::Value) -> Result<Self> {
        for (name, value) in config.as_object().into_iter().flatten() {
            // Tokenizer insertion policies are not special-token template variables.
            if !name.ends_with("_token") || name.starts_with("add_") || value.is_null() {
                continue;
            }
            let token = value
                .as_str()
                .or_else(|| value.get("content").and_then(serde_json::Value::as_str))
                .ok_or_else(|| TurError::Tokenizer(format!("invalid {name}")))?;
            self.tokens.insert(name.clone(), token.into());
        }
        Ok(self)
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
        let mut ctx = self.tokens.clone();
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

/// `strftime_now(format)`, as Transformers offers templates, in UTC. Templates only stamp
/// a date, so `%Y`, `%m`, `%d` and `%%` are supported and anything else is an error.
fn strftime_now(format: &str) -> std::result::Result<String, minijinja::Error> {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| template_error(format!("strftime_now: {e}")))?
        .as_secs();
    format_date(format, seconds / 86_400)
}

fn date_error(error: std::fmt::Error) -> minijinja::Error {
    template_error(format!("strftime_now: {error}"))
}

/// `format` for the civil date `days` after 1970-01-01 (Hinnant's `civil_from_days`).
fn format_date(format: &str, days: u64) -> std::result::Result<String, minijinja::Error> {
    let shifted = days + 719_468;
    let era = shifted / 146_097;
    let of_era = shifted - era * 146_097;
    let year_of_era = (of_era - of_era / 1460 + of_era / 36_524 - of_era / 146_096) / 365;
    let day_of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    let mut out = String::with_capacity(format.len() + 8);
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => write!(out, "{year:04}").map_err(date_error)?,
            Some('m') => write!(out, "{month:02}").map_err(date_error)?,
            Some('d') => write!(out, "{day:02}").map_err(date_error)?,
            Some('%') => out.push('%'),
            other => {
                return Err(template_error(format!(
                    "strftime_now does not support %{}",
                    other.unwrap_or(' ')
                )));
            }
        }
    }
    Ok(out)
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
    env.add_function("strftime_now", strftime_now);
    env.add_filter("tojson", python_tojson);
    env.set_unknown_method_callback(python_method);
    env
}

fn template_error(message: impl Into<String>) -> minijinja::Error {
    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message.into())
}

/// Python string and mapping methods used by checkpoint templates.
fn python_method(
    _: &minijinja::State<'_, '_>,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    if value.kind() == minijinja::value::ValueKind::Map && method == "get" {
        let (key, default) = match args {
            [key] => (key, minijinja::Value::from(())),
            [key, default] => (key, default.clone()),
            _ => return Err(template_error("get needs a key and an optional default")),
        };
        let item = value.get_item(key)?;
        return Ok(if item.is_undefined() { default } else { item });
    }
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
    fn test_special_token_context_ignores_tokenizer_flags() -> Result<()> {
        let template = ChatTemplate::from_template(
            "{{ bos_token }}{{ eos_token }}{% if add_bos_token is defined %}flag{% endif %}",
        )?
        .with_tokens(&serde_json::json!({
            "bos_token": "<bos>",
            "eos_token": {"content": "<eos>"},
            "add_bos_token": false,
            "add_eos_token": true
        }))?;
        assert_eq!(
            template.render(&[], &[], &RenderOptions::default())?,
            "<bos><eos>"
        );
        Ok(())
    }

    #[test]
    fn test_strftime_formats_civil_dates_and_rejects_other_directives() {
        assert_eq!(
            format_date("%Y-%m-%d", 0).ok().as_deref(),
            Some("1970-01-01")
        );
        // 2024-02-29 and the day after, across a leap day.
        assert_eq!(
            format_date("%Y-%m-%d", 19_782).ok().as_deref(),
            Some("2024-02-29")
        );
        assert_eq!(
            format_date("%Y/%m/%d %%", 19_783).ok().as_deref(),
            Some("2024/03/01 %")
        );
        assert!(format_date("%H", 0).is_err());
    }

    #[test]
    fn test_invalid_template_is_rejected() {
        assert!(ChatTemplate::from_template("{% unclosed block").is_err());
    }
}
