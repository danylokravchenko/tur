//! Separates a forced answer from the reasoning before it, for models that write both in one
//! stream. Without this, a client parsing `content` as JSON would read the reasoning too.

use super::{harmony::Fragment, tokenizer::TokenOutputStream};
use crate::Result;
use tokenizers::Tokenizer;

/// Splits one request's generated tokens at the special token that ends its reasoning.
pub(crate) struct ReasoningSplit {
    /// Opens the reasoning; `None` when the prompt already did.
    open: Option<u32>,
    close: u32,
    state: State,
    tokenizer: Tokenizer,
    reasoning: TokenOutputStream,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nothing written yet, and the model may skip reasoning.
    Before,
    /// Inside the channel's name line, which isn't reasoning.
    Label,
    Reasoning,
    /// Reasoning is over; the whitespace after it belongs to neither part.
    Closed,
    Answer,
}

/// Where one token goes, as [`ReasoningSplit::push`] decides.
pub(crate) enum Route {
    /// Part of the answer.
    Answer,
    /// Consumed here, with any reasoning text it completed.
    Taken(Fragment),
}

impl ReasoningSplit {
    /// Reasoning that the prompt opened and `close` ends.
    pub(crate) fn opened(tokenizer: &Tokenizer, close: u32) -> Self {
        Self::new(tokenizer, None, close, State::Reasoning)
    }

    /// Reasoning the model may open with `open`, whose first line names its channel.
    pub(crate) fn channel(tokenizer: &Tokenizer, open: u32, close: u32) -> Self {
        Self::new(tokenizer, Some(open), close, State::Before)
    }

    fn new(tokenizer: &Tokenizer, open: Option<u32>, close: u32, state: State) -> Self {
        Self {
            open,
            close,
            state,
            tokenizer: tokenizer.clone(),
            reasoning: TokenOutputStream::new(tokenizer.clone()),
        }
    }

    fn text(&self, token: u32) -> Result<String> {
        self.tokenizer
            .decode(&[token], false)
            .map_err(|error| crate::TurError::Tokenizer(error.to_string()))
    }

    pub(crate) fn push(&mut self, token: u32) -> Result<Route> {
        let taken = |reasoning| {
            Ok(Route::Taken(Fragment {
                content: None,
                reasoning,
            }))
        };
        match self.state {
            State::Before if Some(token) == self.open => {
                self.state = State::Label;
                taken(None)
            }
            State::Label => {
                if self.text(token)?.contains('\n') {
                    self.state = State::Reasoning;
                }
                taken(None)
            }
            State::Reasoning if token == self.close => {
                self.state = State::Closed;
                taken(self.reasoning.take_rest()?)
            }
            State::Reasoning => taken(self.reasoning.next_token(token)?),
            State::Closed if self.text(token)?.trim().is_empty() => taken(None),
            State::Before | State::Closed | State::Answer => {
                self.state = State::Answer;
                Ok(Route::Answer)
            }
        }
    }

    /// Reasoning still held back when generation ended inside it.
    pub(crate) fn take_rest(&mut self) -> Result<Option<String>> {
        self.reasoning.take_rest()
    }

    pub(crate) fn reasoning(&self) -> Result<String> {
        self.reasoning.decode_all()
    }
}
