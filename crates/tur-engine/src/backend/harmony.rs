//! gpt-oss's Harmony output format: messages on named channels, separated by special tokens.
//!
//! ```text
//! <|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>final<|message|>answer
//! <|channel|>commentary to=functions.name <|constrain|>json<|message|>{"arguments":1}
//! ```
//!
//! Parsing token IDs rather than text means a literal `<|channel|>` written in a message
//! is just text, which it cannot be for the tokenizer's own special tokens.

use crate::{Result, TurError, backend::tokenizer::TokenOutputStream};
use tur_models::backend::tools::{ToolCall, ToolDefinition};

/// The special tokens that structure a message.
#[derive(Clone, Copy)]
struct Markers {
    start: u32,
    end: u32,
    message: u32,
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

/// Splits one request's generated tokens by channel.
pub(crate) struct Harmony {
    markers: Markers,
    tokenizer: tokenizers::Tokenizer,
    /// Tokens of the current message's header, before its `<|message|>`.
    header: Vec<u32>,
    body: Option<Body>,
    content: TokenOutputStream,
    reasoning: TokenOutputStream,
    /// Recipient and argument tokens of each message addressed to a tool.
    calls: Vec<(String, Vec<u32>)>,
    /// Whether any message has begun. Output without one is plain text.
    opened: bool,
}

impl Harmony {
    /// A parser, when `tokenizer` has Harmony's special tokens.
    pub(crate) fn new(tokenizer: &tokenizers::Tokenizer) -> Option<Self> {
        let id = |token: &str| tokenizer.token_to_id(token);
        Some(Self {
            markers: Markers {
                start: id("<|start|>")?,
                end: id("<|end|>")?,
                message: id("<|message|>")?,
            },
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
            .map_err(|error| TurError::Tokenizer(format!("cannot decode Harmony: {error}")))
    }

    /// Consume one generated token that is not an end of generation.
    pub(crate) fn push(&mut self, token: u32) -> Result<Fragment> {
        let mut fragment = Fragment::default();
        if token == self.markers.start || token == self.markers.end {
            fragment = self.close()?;
        } else if token == self.markers.message && self.body.is_none() {
            self.open()?;
        } else {
            match self.body {
                None => self.header.push(token),
                Some(Body::Content) => fragment.content = self.content.next_token(token)?,
                Some(Body::Reasoning) => fragment.reasoning = self.reasoning.next_token(token)?,
                Some(Body::Call) => {
                    if let Some((_, tokens)) = self.calls.last_mut() {
                        tokens.push(token);
                    }
                }
            }
        }
        Ok(fragment)
    }

    /// Read the finished header: a recipient makes a tool call, and `analysis` is reasoning.
    /// Every other channel is for the user, including `commentary` preambles.
    fn open(&mut self) -> Result<()> {
        let header = self.decode_raw(&self.header)?;
        self.header.clear();
        self.opened = true;
        let word = |after: &str| {
            header.split_once(after).map(|(_, rest)| {
                rest.trim_start()
                    .split(|c: char| c.is_whitespace() || c == '<')
                    .next()
                    .unwrap_or_default()
                    .to_owned()
            })
        };
        self.body = Some(if let Some(recipient) = word("to=") {
            let name = recipient.strip_prefix("functions.").unwrap_or(&recipient);
            self.calls.push((name.to_owned(), Vec::new()));
            Body::Call
        } else if word("<|channel|>").as_deref() == Some("analysis") {
            Body::Reasoning
        } else {
            Body::Content
        });
        Ok(())
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
            if !tools.iter().any(|tool| &tool.name == name) {
                continue;
            }
            if let Ok(arguments) =
                serde_json::from_str::<serde_json::Value>(&self.decode_raw(tokens)?)
                && arguments.is_object()
            {
                calls.push(ToolCall {
                    name: name.clone(),
                    arguments,
                });
            }
        }
        Ok(calls)
    }
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
        let model = WordLevel::builder()
            .vocab(
                WORDS
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
            &WORDS[..5]
                .iter()
                .map(|word| AddedToken::from(*word, true))
                .collect::<Vec<_>>(),
        );
        Ok(tokenizer)
    }

    fn id(word: &str) -> u32 {
        WORDS.iter().position(|w| *w == word).unwrap_or(0) as u32
    }

    /// Run `words` and return everything published, as each audience would see it.
    fn run(harmony: &mut Harmony, words: &[&str]) -> Result<(String, String)> {
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut add = |fragment: Fragment| {
            content.push_str(&fragment.content.unwrap_or_default());
            reasoning.push_str(&fragment.reasoning.unwrap_or_default());
        };
        for word in words {
            add(harmony.push(id(word))?);
        }
        add(harmony.finish()?);
        Ok((content, reasoning))
    }

    #[test]
    fn test_analysis_is_reasoning_and_final_is_the_answer() -> Result<()> {
        let mut harmony =
            Harmony::new(&tokenizer()?).ok_or_else(|| TurError::Other("no".into()))?;
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
        let mut harmony =
            Harmony::new(&tokenizer()?).ok_or_else(|| TurError::Other("no".into()))?;
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
        let mut cut = Harmony::new(&tokenizer).ok_or_else(|| TurError::Other("no".into()))?;
        let (_, reasoning) = run(
            &mut cut,
            &["<|channel|>", "analysis", "<|message|>", "thought"],
        )?;
        assert_eq!(reasoning.trim(), "thought");
        let mut plain = Harmony::new(&tokenizer).ok_or_else(|| TurError::Other("no".into()))?;
        let (content, _) = run(&mut plain, &["answer", "answer"])?;
        assert_eq!(content.split_whitespace().count(), 2);
        // No Harmony tokens at all: not this format.
        let other = WordLevel::builder()
            .vocab([("a".to_owned(), 0)].into_iter().collect())
            .unk_token("a".into())
            .build()
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        assert!(Harmony::new(&tokenizers::Tokenizer::new(other)).is_none());
        Ok(())
    }
}
