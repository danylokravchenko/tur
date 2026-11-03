//! Holds a model's tool calls out of its answer. Qwen's call markers aren't special tokens, so
//! user-facing decoding keeps them, and Gemma's are, so it drops only the markers: either way
//! a streaming client would read the call's raw text as the answer.

use crate::{Result, TurError};
use tokenizers::Tokenizer;
use tur_models::backend::tools::{ToolCall, ToolDefinition, ToolSyntax};

/// Takes every token from a call's opening marker to its closing one.
pub(crate) struct CallSplit {
    open: u32,
    close: u32,
    /// Each call's tokens, markers included; the last is still open while `inside`.
    calls: Vec<Vec<u32>>,
    inside: bool,
}

/// The parsed calls, and the decoded text of any span that isn't a complete call.
pub(crate) struct Calls {
    pub(crate) calls: Vec<ToolCall>,
    pub(crate) unparsed: String,
}

impl CallSplit {
    /// A split for `syntax`'s markers, when `tokenizer` has them as single tokens. Harmony
    /// calls are channels, which its own parser routes.
    pub(crate) fn new(syntax: ToolSyntax, tokenizer: &Tokenizer) -> Option<Self> {
        let (open, close) = match syntax {
            ToolSyntax::Xml => ("<tool_call>", "</tool_call>"),
            ToolSyntax::Gemma => ("<|tool_call>", "<tool_call|>"),
            ToolSyntax::Harmony => return None,
        };
        Some(Self {
            open: tokenizer.token_to_id(open)?,
            close: tokenizer.token_to_id(close)?,
            calls: Vec::new(),
            inside: false,
        })
    }

    /// Whether `token` belongs to a call, keeping it if it does.
    pub(crate) fn take(&mut self, token: u32) -> bool {
        if self.inside {
            if let Some(call) = self.calls.last_mut() {
                call.push(token);
            }
            self.inside = token != self.close;
            true
        } else if token == self.open {
            self.calls.push(vec![token]);
            self.inside = true;
            true
        } else {
            false
        }
    }

    /// Parse each span against `tools`. A span that holds no valid call, such as one cut off
    /// by the token limit, comes back as text, so nothing the model wrote is lost.
    pub(crate) fn finish(&self, tokenizer: &Tokenizer, tools: &[ToolDefinition]) -> Result<Calls> {
        let decode = |tokens: &[u32], skip_special| {
            tokenizer
                .decode(tokens, skip_special)
                .map_err(|error| TurError::Tokenizer(format!("cannot decode tool call: {error}")))
        };
        let mut parsed = Calls {
            calls: Vec::new(),
            unparsed: String::new(),
        };
        for span in &self.calls {
            match ToolCall::parse_from_output(&decode(span, false)?, tools).as_slice() {
                [call] => parsed.calls.push(call.clone()),
                _ => parsed.unparsed.push_str(&decode(span, true)?),
            }
        }
        Ok(parsed)
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert how calls are split from the answer"
)]
mod tests {
    use super::*;
    use tokenizers::{AddedToken, models::wordlevel::WordLevel};

    fn tokenizer() -> Result<Tokenizer> {
        let words = [
            "<tool_call>",
            "</tool_call>",
            "<function=f>",
            "</function>",
            "hi",
        ];
        let model = WordLevel::builder()
            .vocab((0u32..).zip(words).map(|(id, w)| (w.into(), id)).collect())
            .unk_token("hi".into())
            .build()
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        // Qwen's markers are added but not special, so user-facing decoding keeps them.
        tokenizer.add_tokens(&[
            AddedToken::from("<tool_call>", false),
            AddedToken::from("</tool_call>", false),
        ]);
        Ok(tokenizer)
    }

    #[test]
    fn test_calls_leave_the_answer_and_an_unfinished_one_returns_as_text() -> Result<()> {
        let tokenizer = tokenizer()?;
        let mut split = CallSplit::new(ToolSyntax::Xml, &tokenizer)
            .ok_or_else(|| TurError::Other("markers missing".into()))?;
        let answer: Vec<u32> = [4, 0, 2, 3, 1, 4, 0, 2]
            .into_iter()
            .filter(|&token| !split.take(token))
            .collect();
        // The answer keeps only the text around calls; the second call never closed.
        assert_eq!(answer, [4, 4]);
        let tools = [ToolDefinition::new("f", "", serde_json::json!({}))];
        let calls = split.finish(&tokenizer, &tools)?;
        assert_eq!(calls.calls.len(), 1);
        assert_eq!(calls.calls[0].name, "f");
        assert_eq!(calls.unparsed, "<tool_call> <function=f>");
        // A call to a tool that wasn't offered isn't one either.
        let calls = split.finish(&tokenizer, &[])?;
        assert!(calls.calls.is_empty());
        assert_eq!(
            calls.unparsed,
            "<tool_call> <function=f> </function> </tool_call><tool_call> <function=f>"
        );
        Ok(())
    }

    #[test]
    fn test_harmony_and_tokenizers_without_markers_have_no_split() -> Result<()> {
        let tokenizer = tokenizer()?;
        assert!(CallSplit::new(ToolSyntax::Harmony, &tokenizer).is_none());
        assert!(CallSplit::new(ToolSyntax::Gemma, &tokenizer).is_none());
        Ok(())
    }
}
