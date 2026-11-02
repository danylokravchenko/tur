//! Grammars for forced output: JSON answers and tool calls in each model's own format.
//!
//! Special tokens are written by id, because a grammar can only match them as tokens: their
//! text is not bytes the lexer sees. That includes `<tool_call>`, which the tokenizer doesn't
//! mark special but llguidance treats as special for its `<...>` shape.

use super::guidance::TopLevelGrammar;
use crate::Result;
use serde_json::{Value, json};
use std::fmt::Write as _;
use tokenizers::Tokenizer;
use tur_core::execution::invalid;
use tur_models::backend::tools::{ToolDefinition, ToolSyntax};

/// Output a request forces, compiled against the model's format at submission.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuredOutput {
    /// Any JSON object.
    JsonObject,
    /// JSON matching this schema.
    JsonSchema(Value),
    /// Text matching this regular expression.
    Regex(String),
    /// Text matching this Lark grammar, which needs its own `start` rule.
    Lark(String),
    /// A call to the named tool, or one or more calls to any of the request's tools.
    Tool(Option<String>),
}

/// What the grammar lets a model write before the forced output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preamble {
    /// Nothing.
    None,
    /// Reasoning, in whichever form this model writes it. Its text is unconstrained.
    Reasoning,
}

/// Compile `output` for a model writing `syntax`. `eos_tokens` can't appear in reasoning.
pub(crate) fn grammar(
    output: &StructuredOutput,
    syntax: ToolSyntax,
    tools: &[ToolDefinition],
    tokenizer: &Tokenizer,
    preamble: Preamble,
    eos_tokens: &[u32],
) -> Result<TopLevelGrammar> {
    let mut lark = Lark::new(tokenizer);
    let body = match output {
        StructuredOutput::JsonObject => lark.answer(syntax, &json!({"type": "object"}))?,
        StructuredOutput::JsonSchema(schema) => lark.answer(syntax, schema)?,
        StructuredOutput::Regex(regex) => {
            let rule = lark.rule("", &format!("/{}/", regex_literal(regex)));
            lark.framed(syntax, rule)?
        }
        StructuredOutput::Lark(definition) => {
            let rule = lark.rule("", &format!("%lark {{\n{definition}\n}}"));
            lark.framed(syntax, rule)?
        }
        StructuredOutput::Tool(name) => {
            let offered: Vec<&ToolDefinition> = tools
                .iter()
                .filter(|tool| name.as_ref().is_none_or(|name| &tool.name == name))
                .collect();
            if offered.is_empty() {
                return Err(invalid(name.as_ref().map_or_else(
                    || "tool_choice=required needs at least one tool".to_owned(),
                    |name| format!("tool_choice names unknown tool '{name}'"),
                )));
            }
            lark.calls(syntax, &offered, name.is_none())?
        }
    };
    let start = if preamble == Preamble::Reasoning {
        format!("{} {body}", lark.reasoning(syntax, eos_tokens)?)
    } else {
        body
    };
    Ok(TopLevelGrammar::from_lark(lark.finish(&start)))
}

/// A grammar for what follows a tool-call marker, and the marker's token, for the text between
/// markers to run free under `tool_choice=auto`. Harmony has no such marker: its calls open
/// like any other message, and only the header names a recipient.
pub(crate) fn lazy_call(
    syntax: ToolSyntax,
    tools: &[ToolDefinition],
    tokenizer: &Tokenizer,
) -> Result<Option<(TopLevelGrammar, u32)>> {
    if syntax == ToolSyntax::Harmony || tools.is_empty() {
        return Ok(None);
    }
    let mut lark = Lark::new(tokenizer);
    let offered: Vec<&ToolDefinition> = tools.iter().collect();
    let (body, trigger) = lark.call_after_marker(syntax, &offered)?;
    Ok(Some((
        TopLevelGrammar::from_lark(lark.finish(&body)),
        trigger,
    )))
}

/// A Lark grammar under construction. Rules are numbered, so names from a request's schema
/// never reach the grammar's own namespace.
struct Lark<'a> {
    tokenizer: &'a Tokenizer,
    rules: Vec<String>,
}

impl<'a> Lark<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            rules: Vec::new(),
        }
    }

    fn finish(mut self, start: &str) -> String {
        self.rules.insert(0, format!("start: {start}"));
        self.rules.join("\n")
    }

    fn id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .ok_or_else(|| invalid(format!("this model's tokenizer has no {token} token")))
    }

    fn token(&self, token: &str) -> Result<String> {
        Ok(format!("<[{}]>", self.id(token)?))
    }

    /// Add a rule and return its name.
    fn rule(&mut self, attributes: &str, body: &str) -> String {
        let name = format!("r{}", self.rules.len());
        self.rules.push(format!("{name}{attributes}: {body}"));
        name
    }

    /// A rule matching compact JSON under `schema`, without the optional whitespace a model
    /// can otherwise emit until it runs out of tokens.
    fn json(&mut self, schema: &Value) -> Result<String> {
        let mut schema = schema.clone();
        if let Some(object) = schema.as_object_mut() {
            object
                .entry("x-guidance")
                .or_insert_with(|| json!({"whitespace_flexible": false, "item_separator": ",", "key_separator": ":"}));
        }
        let text = serde_json::to_string(&schema)?;
        Ok(self.rule("", &format!("%json {text}")))
    }

    /// A JSON answer, on Harmony's final channel for models that use channels.
    fn answer(&mut self, syntax: ToolSyntax, schema: &Value) -> Result<String> {
        let json = self.json(schema)?;
        self.framed(syntax, json)
    }

    /// `answer` where this model's answers go: on Harmony's final channel, for models with
    /// channels.
    fn framed(&self, syntax: ToolSyntax, answer: String) -> Result<String> {
        if syntax != ToolSyntax::Harmony {
            return Ok(answer);
        }
        let (channel, message) = (self.token("<|channel|>")?, self.token("<|message|>")?);
        Ok(format!("{channel} \"final\" {message} {answer}"))
    }

    /// The optional reasoning before forced output. Qwen's is not optional: the prompt has
    /// already opened it, so it must close.
    fn reasoning(&mut self, syntax: ToolSyntax, eos_tokens: &[u32]) -> Result<String> {
        let excluding = |lark: &Self, tokens: &[&str]| -> Result<String> {
            let mut ids = tokens
                .iter()
                .map(|token| lark.id(token))
                .collect::<Result<Vec<_>>>()?;
            ids.extend_from_slice(eos_tokens);
            let ids: Vec<String> = ids.iter().map(u32::to_string).collect();
            Ok(format!("<[^{}]>*", ids.join(",")))
        };
        Ok(match syntax {
            ToolSyntax::Xml => {
                let text = excluding(self, &["</think>"])?;
                format!("{text} {} \"\\n\\n\"", self.token("</think>")?)
            }
            ToolSyntax::Gemma => {
                let text = excluding(self, &["<channel|>"])?;
                let (open, close) = (self.token("<|channel>")?, self.token("<channel|>")?);
                let rule = self.rule("", &format!("{open} \"thought\\n\" {text} {close}"));
                format!("{rule}?")
            }
            ToolSyntax::Harmony => {
                let (channel, message) = (self.token("<|channel|>")?, self.token("<|message|>")?);
                let (end, start) = (self.token("<|end|>")?, self.token("<|start|>")?);
                let text = excluding(
                    self,
                    &["<|channel|>", "<|message|>", "<|end|>", "<|start|>"],
                )?;
                let rule = self.rule(
                    "",
                    &format!("{channel} \"analysis\" {message} {text} {end} {start} \"assistant\""),
                );
                format!("{rule}?")
            }
        })
    }

    /// One call as it continues after its opening marker, and that marker's token.
    fn call_after_marker(
        &mut self,
        syntax: ToolSyntax,
        tools: &[&ToolDefinition],
    ) -> Result<(String, u32)> {
        let calls = tools
            .iter()
            .map(|tool| match syntax {
                ToolSyntax::Gemma => self.gemma_call(tool),
                _ => self.xml_call(tool),
            })
            .collect::<Result<Vec<_>>>()?
            .join(" | ");
        Ok(if syntax == ToolSyntax::Gemma {
            let close = self.token("<tool_call|>")?;
            (format!("({calls}) {close}"), self.id("<|tool_call>")?)
        } else {
            let close = self.token("</tool_call>")?;
            (
                format!("\"\\n\" ({calls}) \"\\n\" {close}"),
                self.id("<tool_call>")?,
            )
        })
    }

    fn calls(
        &mut self,
        syntax: ToolSyntax,
        tools: &[&ToolDefinition],
        repeat: bool,
    ) -> Result<String> {
        let calls = tools
            .iter()
            .map(|tool| match syntax {
                ToolSyntax::Xml => self.xml_call(tool),
                ToolSyntax::Gemma => self.gemma_call(tool),
                ToolSyntax::Harmony => self.harmony_call(tool),
            })
            .collect::<Result<Vec<_>>>()?
            .join(" | ");
        Ok(match syntax {
            ToolSyntax::Xml => {
                let (open, close) = (self.token("<tool_call>")?, self.token("</tool_call>")?);
                let call = self.rule("", &format!("{open} \"\\n\" ({calls}) \"\\n\" {close}"));
                if repeat {
                    format!("{call} (\"\\n\" {call})*")
                } else {
                    call
                }
            }
            ToolSyntax::Gemma => {
                let (open, close) = (self.token("<|tool_call>")?, self.token("<tool_call|>")?);
                let call = self.rule("", &format!("{open} ({calls}) {close}"));
                if repeat { format!("{call}+") } else { call }
            }
            // A Harmony call ends the turn with `<|call|>`, so a turn holds exactly one.
            ToolSyntax::Harmony => calls,
        })
    }

    fn harmony_call(&mut self, tool: &ToolDefinition) -> Result<String> {
        let (channel, message) = (self.token("<|channel|>")?, self.token("<|message|>")?);
        let constrain = self.token("<|constrain|>")?;
        let header = quoted(&format!("commentary to=functions.{} ", tool.name))?;
        let arguments = self.json(&tool.parameters)?;
        Ok(format!(
            "{channel} {header} {constrain} \"json\" {message} {arguments}"
        ))
    }

    /// `<function=name>` with each parameter in declared order. Values follow the rule
    /// `ToolCall::parse_from_output` reads them by: JSON for the types it decodes, and
    /// otherwise text up to the closing tag.
    fn xml_call(&mut self, tool: &ToolDefinition) -> Result<String> {
        const CLOSE: &str = "\n</parameter>\n";
        let mut body = quoted(&format!("<function={}>\n", tool.name))?;
        let (properties, required) = fields(&tool.parameters)?;
        for (name, schema) in properties {
            let open = quoted(&format!("<parameter={name}>\n"))?;
            let close = quoted(CLOSE)?;
            let value = match schema["type"].as_str() {
                Some("number" | "integer" | "boolean" | "object" | "array" | "null") => {
                    format!("{} {close}", self.json(schema)?)
                }
                _ => match schema["enum"].as_array() {
                    Some(values) => format!("({}) {close}", string_choices(values)?),
                    // A lazy lexeme: the text ends at the first closing tag.
                    None => self.rule(&format!("[suffix={close}]"), r"/(.|\n)*/"),
                },
            };
            let optional = if required.contains(&name.as_str()) {
                ""
            } else {
                "?"
            };
            write!(body, " ({open} {value}){optional}").map_err(|e| invalid(e.to_string()))?;
        }
        body.push(' ');
        body.push_str(&quoted("</function>")?);
        Ok(self.rule("", &body))
    }

    fn gemma_call(&mut self, tool: &ToolDefinition) -> Result<String> {
        let arguments = self.gemma_object(&tool.parameters, 0)?;
        Ok(format!(
            "{} {arguments}",
            quoted(&format!("call:{}", tool.name))?
        ))
    }

    fn gemma_string(&self) -> Result<(String, String)> {
        let quote = self.token("<|\"|>")?;
        Ok((format!("{quote} /(.|\\n)*/ {quote}"), quote))
    }

    /// `{key:value,...}` in declared order. Commas depend on whether anything came before,
    /// so `head` is built backwards from the last parameter: it holds what may follow when
    /// nothing has been written yet, and `tail` what may follow once something has.
    fn gemma_object(&mut self, schema: &Value, depth: usize) -> Result<String> {
        let (properties, required) = fields(schema)?;
        let (mut head, mut tail) = (String::new(), String::new());
        for (name, value) in properties.iter().rev() {
            let key = self.gemma_key(name)?;
            let member = format!("{key} \":\" {}", self.gemma_value(value, depth + 1)?);
            let first = format!("{member} {tail}");
            let is_required = required.contains(&name.as_str());
            head = match (is_required, head.is_empty()) {
                (true, _) => first,
                (false, true) => format!("({first})?"),
                (false, false) => format!("({first} | {head})"),
            };
            tail = if is_required {
                format!("\",\" {member} {tail}")
            } else {
                format!("(\",\" {member})? {tail}")
            };
        }
        Ok(self.rule("", &format!("\"{{\" {head} \"}}\"")))
    }

    fn gemma_key(&self, name: &str) -> Result<String> {
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            quoted(name)
        } else {
            let quote = self.token("<|\"|>")?;
            Ok(format!("{quote} {} {quote}", quoted(name)?))
        }
    }

    /// Gemma's arguments aren't JSON, so `%json` can't check them. Keywords this can't
    /// enforce are refused rather than ignored: the call must match its schema.
    fn gemma_value(&mut self, schema: &Value, depth: usize) -> Result<String> {
        const ANNOTATIONS: [&str; 5] = ["description", "title", "default", "examples", "$schema"];
        const CHECKED: [&str; 6] = [
            "type",
            "enum",
            "properties",
            "required",
            "items",
            "additionalProperties",
        ];
        if depth > 16 {
            return Err(invalid("tool schema nests too deeply"));
        }
        let object = schema
            .as_object()
            .ok_or_else(|| invalid("tool schema must be an object"))?;
        if let Some(keyword) = object
            .keys()
            .find(|key| !ANNOTATIONS.contains(&key.as_str()) && !CHECKED.contains(&key.as_str()))
        {
            return Err(invalid(format!(
                "forcing a tool call on this model doesn't support schema keyword '{keyword}'"
            )));
        }
        if object
            .get("additionalProperties")
            .is_some_and(|v| v != false)
        {
            return Err(invalid(
                "forcing a tool call on this model needs additionalProperties to be false",
            ));
        }
        if let Some(values) = schema["enum"].as_array() {
            let (_, quote) = self.gemma_string()?;
            let choices = values
                .iter()
                .map(|value| match value {
                    Value::String(text) => Ok(format!("{quote} {} {quote}", quoted(text)?)),
                    Value::Bool(_) | Value::Number(_) | Value::Null => quoted(&value.to_string()),
                    _ => Err(invalid("enum values must be scalars on this model")),
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(format!("({})", choices.join(" | ")));
        }
        let types: Vec<&str> = match &schema["type"] {
            Value::String(name) => vec![name],
            Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if types.is_empty() {
            return Err(invalid(
                "forcing a tool call on this model needs a type or enum for every parameter",
            ));
        }
        let choices = types
            .iter()
            .map(|name| self.gemma_type(name, schema, depth))
            .collect::<Result<Vec<_>>>()?;
        Ok(format!("({})", choices.join(" | ")))
    }

    fn gemma_type(&mut self, name: &str, schema: &Value, depth: usize) -> Result<String> {
        Ok(match name {
            "string" => self.gemma_string()?.0,
            "integer" => "/-?(0|[1-9][0-9]*)/".to_owned(),
            "number" => r"/-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/".to_owned(),
            "boolean" => "(\"true\" | \"false\")".to_owned(),
            "null" => "\"null\"".to_owned(),
            "object" => self.gemma_object(schema, depth)?,
            "array" => {
                let items = schema.get("items").ok_or_else(|| {
                    invalid("forcing a tool call on this model needs items for every array")
                })?;
                let item = self.gemma_value(items, depth + 1)?;
                format!("\"[\" ({item} (\",\" {item})*)? \"]\"")
            }
            other => return Err(invalid(format!("unknown schema type '{other}'"))),
        })
    }
}

/// `regex` as the body of a Lark `/.../` literal, which ends at an unescaped slash or line end.
fn regex_literal(regex: &str) -> String {
    let mut literal = String::with_capacity(regex.len());
    let mut chars = regex.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                literal.push(c);
                literal.extend(chars.next());
            }
            '/' => literal.push_str("\\/"),
            '\n' => literal.push_str("\\n"),
            '\r' => literal.push_str("\\r"),
            other => literal.push(other),
        }
    }
    literal
}

/// A Lark string literal; JSON's escapes are a subset of what it reads.
fn quoted(text: &str) -> Result<String> {
    Ok(serde_json::to_string(text)?)
}

fn string_choices(values: &[Value]) -> Result<String> {
    Ok(values
        .iter()
        .map(|value| match value {
            Value::String(text) => quoted(text),
            other => Err(invalid(format!("non-string enum value {other}"))),
        })
        .collect::<Result<Vec<_>>>()?
        .join(" | "))
}

type Properties<'a> = Vec<(&'a String, &'a Value)>;

/// An object schema's properties in declared order, and which of them are required.
fn fields(schema: &Value) -> Result<(Properties<'_>, Vec<&str>)> {
    let properties: Properties<'_> = schema["properties"]
        .as_object()
        .map(|properties| properties.iter().collect())
        .unwrap_or_default();
    let required: Vec<&str> = schema["required"]
        .as_array()
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if let Some(missing) = required
        .iter()
        .find(|name| !properties.iter().any(|(key, _)| key.as_str() == **name))
    {
        return Err(invalid(format!(
            "tool schema requires undeclared parameter '{missing}'"
        )));
    }
    Ok((properties, required))
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert which outputs each grammar admits"
)]
mod tests {
    use super::*;
    use crate::{
        TurError,
        backend::guidance::{GrammarMask, GuidanceControl, build_llg_factory},
    };
    use serde_json::Map;
    use std::sync::Arc;
    use tur_models::backend::tools::ToolCall;

    const SPECIALS: [&str; 18] = [
        "<eos>",
        "<tool_call>",
        "</tool_call>",
        "<|tool_call>",
        "<tool_call|>",
        "<|\"|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|start|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>",
        "<|im_end|>",
        "<unused>",
        "</think>",
        "<|channel>",
        "<channel|>",
    ];

    /// One token per printable ASCII byte, after the formats' special tokens.
    fn tokenizer() -> Result<Tokenizer> {
        let mut vocab = Map::new();
        let mut added = Vec::new();
        for (id, token) in SPECIALS.iter().enumerate() {
            vocab.insert((*token).into(), id.into());
            added.push(
                json!({"id": id, "content": token, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}),
            );
        }
        // GPT-2's byte-level alphabet maps these bytes to themselves; `Ċ` is its newline.
        for (offset, byte) in (b'!'..=b'~').enumerate() {
            vocab.insert((byte as char).to_string(), (SPECIALS.len() + offset).into());
        }
        vocab.insert("Ġ".into(), (SPECIALS.len() + 94).into());
        vocab.insert("Ċ".into(), (SPECIALS.len() + 95).into());
        let tokenizer = json!({
            "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": false, "use_regex": false},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
                "use_regex": false},
            "added_tokens": added,
            "model": {"type": "BPE", "merges": [], "vocab": vocab, "unk_token": "<eos>"}
        });
        Tokenizer::from_bytes(serde_json::to_vec(&tokenizer)?)
            .map_err(|error| TurError::Tokenizer(error.to_string()))
    }

    /// Output text as tokens: `⟦name⟧` is a special token, and anything else one byte each.
    fn tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(after) = rest.strip_prefix('⟦') {
                let (name, tail) = after
                    .split_once('⟧')
                    .ok_or_else(|| TurError::Other("unclosed marker".into()))?;
                out.push(
                    tokenizer
                        .token_to_id(name)
                        .ok_or_else(|| TurError::Other(format!("no token {name}")))?,
                );
                rest = tail;
                continue;
            }
            let (byte, tail) = rest.split_at(1);
            let word = match byte {
                " " => "Ġ",
                "\n" => "Ċ",
                other => other,
            };
            out.push(
                tokenizer
                    .token_to_id(word)
                    .ok_or_else(|| TurError::Other(format!("no token {word:?}")))?,
            );
            rest = tail;
        }
        Ok(out)
    }

    /// Whether the grammar admits `text` to its end, then finishes.
    fn admits(grammar: TopLevelGrammar, text: &str) -> Result<bool> {
        let tokenizer = tokenizer()?;
        let factory = build_llg_factory(tokenizer.clone(), Some(SPECIALS.len() + 96), &[0])?;
        let mut control = GuidanceControl::new(Arc::clone(&factory));
        control.activate(grammar)?;
        for token in tokens(&tokenizer, text)? {
            match control.mask()? {
                Some(GrammarMask::Allowed(bits))
                    if bits
                        .get((token / 32) as usize)
                        .is_some_and(|word| word >> (token % 32) & 1 == 1) =>
                {
                    control.commit(token)?;
                }
                _ => return Ok(false),
            }
        }
        // Complete means stopped, or that the end of generation is allowed next.
        Ok(match control.mask()? {
            Some(GrammarMask::Stop) => true,
            Some(GrammarMask::Allowed(bits)) => bits.first().is_some_and(|word| word & 1 == 1),
            None => false,
        })
    }

    fn weather() -> ToolDefinition {
        ToolDefinition::new(
            "weather",
            "",
            json!({"type": "object", "required": ["city"], "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            }}),
        )
    }

    fn compile(
        output: &StructuredOutput,
        syntax: ToolSyntax,
        preamble: Preamble,
    ) -> Result<TopLevelGrammar> {
        grammar(output, syntax, &[weather()], &tokenizer()?, preamble, &[0])
    }

    #[test]
    fn test_json_answers_match_the_schema_without_free_whitespace() -> Result<()> {
        let schema = StructuredOutput::JsonSchema(json!({
            "type": "object", "required": ["a"], "additionalProperties": false,
            "properties": {"a": {"type": "integer"}},
        }));
        for (text, expected) in [
            (r#"{"a":1}"#, true),
            (r#"{"a":"x"}"#, false),
            (r#"{ "a":1}"#, false),
            (r#"{"b":1}"#, false),
            ("prose", false),
        ] {
            let grammar = compile(&schema, ToolSyntax::Xml, Preamble::None)?;
            assert_eq!(admits(grammar, text)?, expected, "{text}");
        }
        let object = compile(
            &StructuredOutput::JsonObject,
            ToolSyntax::Gemma,
            Preamble::None,
        )?;
        assert!(admits(object, r#"{"any":[1,2]}"#)?);
        Ok(())
    }

    #[test]
    fn test_harmony_answers_may_reason_first_unless_thinking_is_off() -> Result<()> {
        let schema = StructuredOutput::JsonObject;
        let direct = r#"⟦<|channel|>⟧final⟦<|message|>⟧{"a":1}"#;
        let reasoned = format!(
            "⟦<|channel|>⟧analysis⟦<|message|>⟧free text!⟦<|end|>⟧⟦<|start|>⟧assistant{direct}"
        );
        assert!(admits(
            compile(&schema, ToolSyntax::Harmony, Preamble::Reasoning)?,
            direct
        )?);
        assert!(admits(
            compile(&schema, ToolSyntax::Harmony, Preamble::Reasoning)?,
            &reasoned
        )?);
        assert!(!admits(
            compile(&schema, ToolSyntax::Harmony, Preamble::None)?,
            &reasoned
        )?);
        assert!(!admits(
            compile(&schema, ToolSyntax::Harmony, Preamble::Reasoning)?,
            r#"{"a":1}"#
        )?);
        Ok(())
    }

    /// Calls a grammar admits must also read back as the same call, or forcing them is useless.
    fn assert_call(syntax: ToolSyntax, text: &str, arguments: &Value) -> Result<()> {
        let forced = StructuredOutput::Tool(Some("weather".into()));
        assert!(
            admits(compile(&forced, syntax, Preamble::None)?, text)?,
            "{text}"
        );
        let raw = text.replace(['⟦', '⟧'], "");
        if syntax != ToolSyntax::Harmony {
            let calls = ToolCall::parse_from_output(&raw, &[weather()]);
            assert_eq!(calls.len(), 1, "{raw}");
            assert_eq!(calls[0].name, "weather");
            assert_eq!(&calls[0].arguments, arguments);
        }
        Ok(())
    }

    #[test]
    fn test_xml_calls_are_admitted_only_in_the_parsers_form() -> Result<()> {
        let call = |parameters: &str| {
            format!("⟦<tool_call>⟧\n<function=weather>\n{parameters}</function>\n⟦</tool_call>⟧")
        };
        let city = "<parameter=city>\nKyiv </b> x\n</parameter>\n";
        assert_call(
            ToolSyntax::Xml,
            &call(&format!(
                "{city}<parameter=days>\n2\n</parameter>\n<parameter=unit>\nc\n</parameter>\n"
            )),
            &json!({"city": "Kyiv </b> x", "days": 2, "unit": "c"}),
        )?;
        let forced = StructuredOutput::Tool(Some("weather".into()));
        for bad in [
            call(""),
            call(&format!("{city}<parameter=days>\ntwo\n</parameter>\n")),
            call(&format!("{city}<parameter=unit>\nk\n</parameter>\n")),
        ] {
            let grammar = compile(&forced, ToolSyntax::Xml, Preamble::None)?;
            assert!(!admits(grammar, &bad)?, "{bad}");
        }
        Ok(())
    }

    #[test]
    fn test_gemma_calls_are_admitted_only_when_they_match_the_schema() -> Result<()> {
        let call = |body: &str| format!("⟦<|tool_call>⟧call:weather{{{body}}}⟦<tool_call|>⟧");
        let quote = "⟦<|\"|>⟧";
        let city = format!("city:{quote}Kyiv, {{x}}{quote}");
        assert_call(
            ToolSyntax::Gemma,
            &call(&format!("{city},days:2,unit:{quote}f{quote}")),
            &json!({"city": "Kyiv, {x}", "days": 2, "unit": "f"}),
        )?;
        assert_call(
            ToolSyntax::Gemma,
            &call(&city),
            &json!({"city": "Kyiv, {x}"}),
        )?;
        let forced = StructuredOutput::Tool(Some("weather".into()));
        for bad in [
            call("days:2"),
            call(&format!("{city},days:x")),
            call(&format!("{city},unit:{quote}k{quote}")),
        ] {
            let grammar = compile(&forced, ToolSyntax::Gemma, Preamble::None)?;
            assert!(!admits(grammar, &bad)?, "{bad}");
        }
        Ok(())
    }

    #[test]
    fn test_gemma_refuses_schema_keywords_it_cannot_enforce() -> Result<()> {
        let tool = ToolDefinition::new(
            "t",
            "",
            json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}}}),
        );
        let forced = StructuredOutput::Tool(None);
        let refused = grammar(
            &forced,
            ToolSyntax::Gemma,
            &[tool],
            &tokenizer()?,
            Preamble::None,
            &[0],
        );
        assert!(refused.is_err_and(|error| error.to_string().contains("minimum")));
        Ok(())
    }

    #[test]
    fn test_harmony_calls_carry_json_arguments() -> Result<()> {
        let call = r#"⟦<|channel|>⟧commentary to=functions.weather ⟦<|constrain|>⟧json⟦<|message|>⟧{"city":"Kyiv"}"#;
        assert_call(ToolSyntax::Harmony, call, &json!({}))?;
        let forced = StructuredOutput::Tool(None);
        let missing = r#"⟦<|channel|>⟧commentary to=functions.weather ⟦<|constrain|>⟧json⟦<|message|>⟧{"days":1}"#;
        assert!(!admits(
            compile(&forced, ToolSyntax::Harmony, Preamble::Reasoning)?,
            missing
        )?);
        Ok(())
    }

    #[test]
    fn test_required_allows_several_calls_and_a_named_choice_only_one() -> Result<()> {
        let one = "⟦<|tool_call>⟧call:weather{city:⟦<|\"|>⟧a⟦<|\"|>⟧}⟦<tool_call|>⟧";
        let two = format!("{one}{one}");
        let required = StructuredOutput::Tool(None);
        let named = StructuredOutput::Tool(Some("weather".into()));
        assert!(admits(
            compile(&required, ToolSyntax::Gemma, Preamble::None)?,
            &two
        )?);
        assert!(!admits(
            compile(&named, ToolSyntax::Gemma, Preamble::None)?,
            &two
        )?);
        let unknown = StructuredOutput::Tool(Some("other".into()));
        assert!(compile(&unknown, ToolSyntax::Gemma, Preamble::None).is_err());
        Ok(())
    }

    #[test]
    fn test_qwen_and_gemma_may_reason_before_forced_output() -> Result<()> {
        let json = StructuredOutput::JsonObject;
        let qwen = |text: &str| -> Result<bool> {
            admits(compile(&json, ToolSyntax::Xml, Preamble::Reasoning)?, text)
        };
        assert!(qwen("free <text>!⟦</think>⟧\n\n{\"a\":1}")?);
        assert!(!qwen("{\"a\":1}")?);
        assert!(!qwen("free⟦</think>⟧\n\nprose")?);
        let gemma = |text: &str| -> Result<bool> {
            admits(
                compile(&json, ToolSyntax::Gemma, Preamble::Reasoning)?,
                text,
            )
        };
        assert!(gemma("⟦<|channel>⟧thought\nfree!⟦<channel|>⟧{\"a\":1}")?);
        assert!(gemma("{\"a\":1}")?);
        assert!(!gemma("⟦<|channel>⟧thought\nfree⟦<channel|>⟧prose")?);
        Ok(())
    }

    /// Drive a lazy grammar through `text`, and report whether each marked position allowed
    /// the next token.
    fn lazy_admits(syntax: ToolSyntax, text: &str) -> Result<bool> {
        let tokenizer = tokenizer()?;
        let factory = build_llg_factory(tokenizer.clone(), Some(SPECIALS.len() + 96), &[0])?;
        let mut control = GuidanceControl::new(factory);
        let (grammar, trigger) = lazy_call(syntax, &[weather()], &tokenizer)?
            .ok_or_else(|| TurError::Other("no lazy grammar".into()))?;
        control.arm(grammar, trigger)?;
        for token in tokens(&tokenizer, text)? {
            if let Some(GrammarMask::Allowed(bits)) = control.mask()?
                && bits
                    .get((token / 32) as usize)
                    .is_none_or(|word| word >> (token % 32) & 1 == 0)
            {
                return Ok(false);
            }
            if control.commit(token)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[test]
    fn test_lazy_grammars_constrain_only_calls_and_rearm_after_each() -> Result<()> {
        let quote = "⟦<|\"|>⟧";
        let call = format!("⟦<|tool_call>⟧call:weather{{city:{quote}a{quote}}}⟦<tool_call|>⟧");
        assert!(lazy_admits(ToolSyntax::Gemma, "free text")?);
        assert!(lazy_admits(
            ToolSyntax::Gemma,
            &format!("free {call} more free {call} end")
        )?);
        // A second call is held to the schema as firmly as the first.
        let bad = "⟦<|tool_call>⟧call:weather{days:1}";
        assert!(!lazy_admits(ToolSyntax::Gemma, bad)?);
        assert!(!lazy_admits(
            ToolSyntax::Gemma,
            &format!("{call} then {bad}")
        )?);
        let xml = "⟦<tool_call>⟧\n<function=weather>\n<parameter=city>\nKyiv\n</parameter>\n</function>\n⟦</tool_call>⟧";
        assert!(lazy_admits(ToolSyntax::Xml, &format!("free {xml} free"))?);
        assert!(!lazy_admits(ToolSyntax::Xml, "⟦<tool_call>⟧\nprose")?);
        assert!(lazy_call(ToolSyntax::Harmony, &[weather()], &tokenizer()?)?.is_none());
        Ok(())
    }

    #[test]
    fn test_forced_reports_only_a_single_allowed_token() -> Result<()> {
        let tokenizer = tokenizer()?;
        let factory = build_llg_factory(tokenizer.clone(), Some(SPECIALS.len() + 96), &[0])?;
        let mut control = GuidanceControl::new(factory);
        control.activate(TopLevelGrammar::from_regex("ab(c|d)"))?;
        let id = |text: &str| -> Result<u32> {
            tokens(&tokenizer, text)?
                .first()
                .copied()
                .ok_or_else(|| TurError::Other("empty".into()))
        };
        assert_eq!(control.forced()?, Some(id("a")?));
        control.commit(id("a")?)?;
        assert_eq!(control.forced()?, Some(id("b")?));
        control.commit(id("b")?)?;
        assert_eq!(control.forced()?, None);
        Ok(())
    }

    #[test]
    fn test_regex_and_lark_answers_keep_the_models_framing() -> Result<()> {
        let date = StructuredOutput::Regex(r"[0-9]{2}/[0-9]{2}".into());
        let sql = StructuredOutput::Lark("start: \"SELECT \" NAME \";\"\nNAME: /[a-z]+/\n".into());
        for (output, valid, invalid) in [
            (&date, "12/31", "12-31"),
            (&sql, "SELECT name;", "DROP name;"),
        ] {
            for (syntax, frame) in [
                (ToolSyntax::Xml, ""),
                (ToolSyntax::Gemma, ""),
                (ToolSyntax::Harmony, "⟦<|channel|>⟧final⟦<|message|>⟧"),
            ] {
                let check = |text: &str| -> Result<bool> {
                    admits(
                        compile(output, syntax, Preamble::None)?,
                        &format!("{frame}{text}"),
                    )
                };
                assert!(check(valid)?, "{valid}");
                assert!(!check(invalid)?, "{invalid}");
            }
        }
        let reasoned = "free⟦</think>⟧\n\n12/31";
        assert!(admits(
            compile(&date, ToolSyntax::Xml, Preamble::Reasoning)?,
            reasoned
        )?);
        Ok(())
    }

    #[test]
    fn test_regex_literals_escape_what_would_end_them() -> Result<()> {
        assert_eq!(regex_literal(r"a/b\/c\\/d"), r"a\/b\/c\\\/d");
        assert_eq!(regex_literal("a\nb"), r"a\nb");
        let newline = StructuredOutput::Regex("a\nb".into());
        assert!(admits(
            compile(&newline, ToolSyntax::Xml, Preamble::None)?,
            "a\nb"
        )?);
        Ok(())
    }
}
