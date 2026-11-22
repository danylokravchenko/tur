//! Output formats that write each part of a reply as a message of its own, delimited by
//! special tokens: gpt-oss's Harmony and Inkling 2's.
//!
//! ```text
//! <|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>final<|message|>answer
//! <|channel|>commentary to=functions.name <|constrain|>json<|message|>{"arguments":1}
//!
//! <|message_model|><|content_thinking|>reasoning<|end_message|><|message_model|><|content_text|>answer<|end_message|>
//! <|message_model|>name<|content_invoke_tool_json|>{"name":"name","args":{"arguments":1}}<|end_message|>
//! ```
//!
//! Parsing token IDs rather than text means a literal marker written in a message is just
//! text, which it cannot be for the tokenizer's own special tokens.

use crate::{Result, TurError, backend::tokenizer::TokenOutputStream};
use serde_json::Value;
use tur_models::backend::tools::{ToolCall, ToolDefinition, ToolSyntax};

/// How a format tells what a message's body holds.
#[derive(Clone, Copy)]
enum Format {
    /// The header before `<|message|>` names the channel and any recipient.
    Harmony { message: u32 },
    /// The token that opens the body names its kind, and a call's header names its tool.
    Inkling { text: u32, thinking: u32, call: u32 },
}

/// Where a message's text goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Body {
    Content,
    Reasoning,
    Call,
}

/// Text ready to publish after a token, each part for its own audience.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Fragment {
    pub(crate) content: Option<String>,
    pub(crate) reasoning: Option<String>,
}

/// Splits one request's generated tokens by message.
pub(crate) struct MessageSplit {
    format: Format,
    /// Starts a message, ending any still open.
    start: u32,
    end: u32,
    tokenizer: tokenizers::Tokenizer,
    /// Tokens of the current message's header, before its body opens.
    header: Vec<u32>,
    body: Option<Body>,
    content: TokenOutputStream,
    reasoning: TokenOutputStream,
    /// Header name and body tokens of each message that calls a tool.
    calls: Vec<(String, Vec<u32>)>,
    /// Whether any message has begun. Output without one is plain text.
    opened: bool,
}

impl MessageSplit {
    /// A parser, when `syntax` writes messages and `tokenizer` has their special tokens.
    pub(crate) fn new(syntax: ToolSyntax, tokenizer: &tokenizers::Tokenizer) -> Option<Self> {
        let id = |token: &str| tokenizer.token_to_id(token);
        let (start, end, format) = match syntax {
            ToolSyntax::Harmony => (
                id("<|start|>")?,
                id("<|end|>")?,
                Format::Harmony {
                    message: id("<|message|>")?,
                },
            ),
            ToolSyntax::Inkling => (
                id("<|message_model|>")?,
                id("<|end_message|>")?,
                Format::Inkling {
                    text: id("<|content_text|>")?,
                    thinking: id("<|content_thinking|>")?,
                    call: id("<|content_invoke_tool_json|>")?,
                },
            ),
            ToolSyntax::Xml | ToolSyntax::Gemma => return None,
        };
        Some(Self {
            format,
            start,
            end,
            tokenizer: tokenizer.clone(),
            header: Vec::new(),
            body: None,
            content: TokenOutputStream::new(tokenizer.clone()),
            reasoning: TokenOutputStream::new(tokenizer.clone()),
            calls: Vec::new(),
            opened: false,
        })
    }

    fn decode_raw(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(|error| TurError::Tokenizer(format!("cannot decode messages: {error}")))
    }

    /// Consume one generated token that is not an end of generation.
    pub(crate) fn push(&mut self, token: u32) -> Result<Fragment> {
        if token == self.start || token == self.end {
            return self.close();
        }
        let mut fragment = Fragment::default();
        match self.body {
            None => match self.opens(token)? {
                Some((body, name)) => self.open(body, name),
                None => self.header.push(token),
            },
            Some(Body::Content) => fragment.content = self.content.next_token(token)?,
            Some(Body::Reasoning) => fragment.reasoning = self.reasoning.next_token(token)?,
            Some(Body::Call) => {
                if let Some((_, tokens)) = self.calls.last_mut() {
                    tokens.push(token);
                }
            }
        }
        Ok(fragment)
    }

    /// The body `token` opens after the current header, if it opens one, with the name of
    /// the tool a call addresses.
    fn opens(&self, token: u32) -> Result<Option<(Body, String)>> {
        let header = || self.decode_raw(&self.header);
        Ok(match self.format {
            Format::Harmony { message } if token == message => Some(harmony_body(&header()?)),
            Format::Inkling { text, .. } if token == text => Some((Body::Content, String::new())),
            Format::Inkling { thinking, .. } if token == thinking => {
                Some((Body::Reasoning, String::new()))
            }
            Format::Inkling { call, .. } if token == call => {
                Some((Body::Call, header()?.trim().to_owned()))
            }
            Format::Harmony { .. } | Format::Inkling { .. } => None,
        })
    }

    fn open(&mut self, body: Body, name: String) {
        self.header.clear();
        self.opened = true;
        if body == Body::Call {
            self.calls.push((name, Vec::new()));
        }
        self.body = Some(body);
    }

    /// End the current message, publishing what its stream was still holding back.
    fn close(&mut self) -> Result<Fragment> {
        let body = self.body.take();
        self.header.clear();
        Ok(Fragment {
            content: if body == Some(Body::Content) {
                self.content.take_rest()?
            } else {
                None
            },
            reasoning: if body == Some(Body::Reasoning) {
                self.reasoning.take_rest()?
            } else {
                None
            },
        })
    }

    /// End generation. Output that never opened a message is the model ignoring the format,
    /// and reaches the user as it is.
    pub(crate) fn finish(&mut self) -> Result<Fragment> {
        let mut plain = String::new();
        if !self.opened {
            for token in std::mem::take(&mut self.header) {
                plain.push_str(&self.content.next_token(token)?.unwrap_or_default());
            }
            self.body = Some(Body::Content);
        }
        let mut last = self.close()?;
        if !plain.is_empty() {
            plain.push_str(&last.content.take().unwrap_or_default());
            last.content = Some(plain);
        }
        Ok(last)
    }

    pub(crate) fn content(&self) -> Result<String> {
        self.content.decode_all()
    }

    pub(crate) fn reasoning(&self) -> Result<String> {
        self.reasoning.decode_all()
    }

    /// Complete calls to tools the request offered; anything else is ignored, as malformed
    /// calls are for every other format.
    pub(crate) fn tool_calls(&self, tools: &[ToolDefinition]) -> Result<Vec<ToolCall>> {
        let mut calls = Vec::new();
        for (name, tokens) in &self.calls {
            let Ok(body) = serde_json::from_str::<Value>(&self.decode_raw(tokens)?) else {
                continue;
            };
            let (name, arguments) = match self.format {
                Format::Harmony { .. } => (name.clone(), body),
                Format::Inkling { .. } => inkling_call(name, body),
            };
            if arguments.is_object() && tools.iter().any(|tool| tool.name == name) {
                calls.push(ToolCall { name, arguments });
            }
        }
        Ok(calls)
    }
}

/// What a finished Harmony header opens: a recipient makes a tool call, and `analysis` is
/// reasoning. Every other channel is for the user, including `commentary` preambles.
fn harmony_body(header: &str) -> (Body, String) {
    let word = |after: &str| {
        header.split_once(after).map(|(_, rest)| {
            rest.trim_start()
                .split(|c: char| c.is_whitespace() || c == '<')
                .next()
                .unwrap_or_default()
                .to_owned()
        })
    };
    match word("to=") {
        Some(recipient) => {
            let name = recipient.strip_prefix("functions.").unwrap_or(&recipient);
            (Body::Call, name.to_owned())
        }
        None if word("<|channel|>").as_deref() == Some("analysis") => {
            (Body::Reasoning, String::new())
        }
        None => (Body::Content, String::new()),
    }
}

/// The tool and arguments of an Inkling call body, `{"name":…,"args":{…}}`. The header names
/// the tool too, and the body's name stands in when the header has none.
fn inkling_call(header: &str, mut body: Value) -> (String, Value) {
    let named = body.get("name").and_then(Value::as_str).unwrap_or_default();
    let name = if header.is_empty() { named } else { header }.to_owned();
    (name, body.get_mut("args").map_or(Value::Null, Value::take))
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use tokenizers::{AddedToken, models::wordlevel::WordLevel};

    const WORDS: [&str; 14] = [
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "analysis",
        "final",
        "commentary",
        "to=functions.lookup",
        "to=functions.other",
        "json",
        "thought",
        "answer",
        "{}",
    ];

    fn tokenizer() -> Result<tokenizers::Tokenizer> {
        words(&WORDS, 5)
    }

    /// A word-level tokenizer of `words`, whose first `special` are special tokens.
    fn words(words: &[&str], special: usize) -> Result<tokenizers::Tokenizer> {
        let model = WordLevel::builder()
            .vocab(
                words
                    .iter()
                    .enumerate()
                    .map(|(id, word)| ((*word).to_owned(), id as u32))
                    .collect(),
            )
            .unk_token("thought".into())
            .build()
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.add_special_tokens(
            &words[..special]
                .iter()
                .map(|word| AddedToken::from(*word, true))
                .collect::<Vec<_>>(),
        );
        Ok(tokenizer)
    }

    fn id(word: &str) -> u32 {
        WORDS.iter().position(|w| *w == word).unwrap_or(0) as u32
    }

    const INKLING: [&str; 10] = [
        "<|message_model|>",
        "<|end_message|>",
        "<|content_text|>",
        "<|content_thinking|>",
        "<|content_invoke_tool_json|>",
        "lookup",
        "other",
        "thought",
        "answer",
        r#"{"name":"lookup","args":{}}"#,
    ];

    /// Run Inkling `tokens` and return everything published, as each audience would see it.
    fn run_inkling(tokens: &[&str]) -> Result<(MessageSplit, String, String)> {
        let tokenizer = words(&INKLING, 5)?;
        let mut split = MessageSplit::new(ToolSyntax::Inkling, &tokenizer)
            .ok_or_else(|| TurError::Other("no Inkling markers".into()))?;
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut add = |fragment: Fragment| {
            content.push_str(&fragment.content.unwrap_or_default());
            reasoning.push_str(&fragment.reasoning.unwrap_or_default());
        };
        for token in tokens {
            let id = tokenizer
                .token_to_id(token)
                .ok_or_else(|| TurError::Other(format!("no token {token}")))?;
            add(split.push(id)?);
        }
        add(split.finish()?);
        Ok((split, content, reasoning))
    }

    /// Run `words` and return everything published, as each audience would see it.
    fn run(split: &mut MessageSplit, words: &[&str]) -> Result<(String, String)> {
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut add = |fragment: Fragment| {
            content.push_str(&fragment.content.unwrap_or_default());
            reasoning.push_str(&fragment.reasoning.unwrap_or_default());
        };
        for word in words {
            add(split.push(id(word))?);
        }
        add(split.finish()?);
        Ok((content, reasoning))
    }

    #[test]
    fn test_analysis_is_reasoning_and_final_is_the_answer() -> Result<()> {
        let mut harmony = MessageSplit::new(ToolSyntax::Harmony, &tokenizer()?)
            .ok_or_else(|| TurError::Other("no".into()))?;
        let (content, reasoning) = run(
            &mut harmony,
            &[
                "<|channel|>",
                "analysis",
                "<|message|>",
                "thought",
                "thought",
                "<|end|>",
                "<|start|>",
                "<|channel|>",
                "final",
                "<|message|>",
                "answer",
            ],
        )?;
        assert_eq!(content.trim(), "answer");
        assert_eq!(reasoning.split_whitespace().count(), 2);
        assert_eq!(harmony.content()?, "answer");
        assert_eq!(harmony.reasoning()?, "thought thought");
        Ok(())
    }

    #[test]
    fn test_calls_are_parsed_and_unoffered_tools_and_bad_json_ignored() -> Result<()> {
        let tools = [ToolDefinition::new("lookup", "", serde_json::json!({}))];
        let mut harmony = MessageSplit::new(ToolSyntax::Harmony, &tokenizer()?)
            .ok_or_else(|| TurError::Other("no".into()))?;
        let (content, reasoning) = run(
            &mut harmony,
            &[
                "<|channel|>",
                "commentary",
                "to=functions.other",
                "<|message|>",
                "{}",
                "<|end|>",
                "<|start|>",
                "<|channel|>",
                "commentary",
                "to=functions.lookup",
                "<|constrain|>",
                "json",
                "<|message|>",
                "thought",
                "<|end|>",
                "<|start|>",
                "<|channel|>",
                "commentary",
                "to=functions.lookup",
                "<|constrain|>",
                "json",
                "<|message|>",
                "{}",
            ],
        )?;
        assert_eq!((content.as_str(), reasoning.as_str()), ("", ""));
        let calls = harmony.tool_calls(&tools)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
        Ok(())
    }

    #[test]
    fn test_truncated_and_unstructured_output_loses_nothing() -> Result<()> {
        let tokenizer = tokenizer()?;
        let mut cut = MessageSplit::new(ToolSyntax::Harmony, &tokenizer)
            .ok_or_else(|| TurError::Other("no".into()))?;
        let (_, reasoning) = run(
            &mut cut,
            &["<|channel|>", "analysis", "<|message|>", "thought"],
        )?;
        assert_eq!(reasoning.trim(), "thought");
        let mut plain = MessageSplit::new(ToolSyntax::Harmony, &tokenizer)
            .ok_or_else(|| TurError::Other("no".into()))?;
        let (content, _) = run(&mut plain, &["answer", "answer"])?;
        assert_eq!(content.split_whitespace().count(), 2);
        // No Harmony tokens at all: not this format.
        let other = WordLevel::builder()
            .vocab([("a".to_owned(), 0)].into_iter().collect())
            .unk_token("a".into())
            .build()
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        assert!(
            MessageSplit::new(ToolSyntax::Harmony, &tokenizers::Tokenizer::new(other)).is_none()
        );
        Ok(())
    }

    #[test]
    fn test_inkling_thinking_is_reasoning_and_text_is_the_answer() -> Result<()> {
        let (split, content, reasoning) = run_inkling(&[
            "<|message_model|>",
            "<|content_thinking|>",
            "thought",
            "thought",
            "<|end_message|>",
            "<|message_model|>",
            "<|content_text|>",
            "answer",
            "<|end_message|>",
        ])?;
        assert_eq!(content.trim(), "answer");
        assert_eq!(reasoning.split_whitespace().count(), 2);
        assert_eq!(split.content()?, "answer");
        assert_eq!(split.reasoning()?, "thought thought");
        Ok(())
    }

    #[test]
    fn test_inkling_calls_are_parsed_and_unoffered_tools_ignored() -> Result<()> {
        let tools = [ToolDefinition::new("lookup", "", serde_json::json!({}))];
        let call = r#"{"name":"lookup","args":{}}"#;
        let (split, content, reasoning) = run_inkling(&[
            "<|message_model|>",
            "<|content_thinking|>",
            "thought",
            "<|end_message|>",
            "<|message_model|>",
            "other",
            "<|content_invoke_tool_json|>",
            call,
            "<|end_message|>",
            "<|message_model|>",
            "lookup",
            "<|content_invoke_tool_json|>",
            call,
            "<|end_message|>",
        ])?;
        assert_eq!((content.as_str(), reasoning.trim()), ("", "thought"));
        let calls = split.tool_calls(&tools)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
        Ok(())
    }

    #[test]
    fn test_inkling_truncated_thinking_and_unstructured_output_lose_nothing() -> Result<()> {
        let (_, content, reasoning) =
            run_inkling(&["<|message_model|>", "<|content_thinking|>", "thought"])?;
        assert_eq!((content.as_str(), reasoning.trim()), ("", "thought"));
        let (_, content, reasoning) = run_inkling(&["answer", "answer"])?;
        assert_eq!(content.split_whitespace().count(), 2);
        assert_eq!(reasoning, "");
        Ok(())
    }
}
