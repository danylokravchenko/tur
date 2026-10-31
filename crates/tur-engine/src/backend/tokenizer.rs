use crate::{Result, TurError};

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    /// Preserve protocol delimiters for tool parsing; user-facing decoding still removes
    /// special tokens, as the tokenizer requests.
    pub(crate) fn decode_raw(&self) -> Result<String> {
        self.tokenizer
            .decode(&self.tokens, false)
            .map_err(|err| TurError::Tokenizer(format!("cannot decode tool output: {err}")))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => Err(TurError::Tokenizer(format!("cannot decode: {err}"))),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().is_some_and(char::is_alphanumeric) {
            let text = text
                .get(prev_text.len()..)
                .ok_or_else(|| TurError::Tokenizer("unstable UTF-8 decode boundary".into()))?;
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text
                .get(prev_text.len()..)
                .ok_or_else(|| TurError::Tokenizer("unstable UTF-8 decode boundary".into()))?;
            Ok(Some(text.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Text still held back, ending this run of tokens so that none of it is published twice.
    pub(crate) fn take_rest(&mut self) -> Result<Option<String>> {
        let rest = self.decode_rest()?;
        self.prev_index = self.tokens.len();
        self.current_index = self.tokens.len();
        Ok(rest)
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert user and protocol decoding separately"
)]
mod tests {
    use super::*;
    #[test]
    fn test_tool_decoding_preserves_special_delimiters_that_user_decoding_removes() -> Result<()> {
        use tokenizers::{AddedToken, models::wordlevel::WordLevel};
        let vocabulary = [
            ("<|tool_call>".into(), 0),
            ("call:lookup{}".into(), 1),
            ("<tool_call|>".into(), 2),
        ];
        let model = WordLevel::builder()
            .vocab(vocabulary.into_iter().collect())
            .unk_token("call:lookup{}".into())
            .build()
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|tool_call>", true),
            AddedToken::from("<tool_call|>", true),
        ]);
        let mut stream = TokenOutputStream::new(tokenizer);
        for token in [0, 1, 2] {
            drop(stream.next_token(token)?);
        }
        assert_eq!(stream.decode_all()?, "call:lookup{}");
        assert_eq!(
            stream.decode_raw()?,
            "<|tool_call> call:lookup{} <tool_call|>"
        );
        Ok(())
    }
}
