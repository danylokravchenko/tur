#![forbid(unsafe_code)]
//! Ending generation at a text sequence, however tokens split it.

use tur_core::{Result, execution::invalid};

/// Sequences one request may stop at, and the longest of them. They are searched in every
/// pass's text and held back from streams, so both stay small.
const MAX_STOP_SEQUENCES: usize = 16;
const MAX_STOP_BYTES: usize = 1024;

/// Finds a request's stop sequences in its generated text. Text that could still become one
/// is held back, so a stream never shows a client any part of a sequence that completes.
#[derive(Debug, Default)]
pub(super) struct StopMatcher {
    sequences: Vec<String>,
    held: String,
    matched: bool,
}

impl StopMatcher {
    pub(super) fn new(sequences: &[String]) -> Result<Self> {
        if sequences.len() > MAX_STOP_SEQUENCES {
            return Err(invalid(format!(
                "at most {MAX_STOP_SEQUENCES} stop sequences are allowed"
            )));
        }
        if sequences
            .iter()
            .any(|stop| stop.is_empty() || stop.len() > MAX_STOP_BYTES)
        {
            return Err(invalid(format!(
                "stop sequences must hold 1 to {MAX_STOP_BYTES} bytes"
            )));
        }
        Ok(Self {
            sequences: sequences.to_vec(),
            ..Self::default()
        })
    }

    pub(super) fn matched(&self) -> bool {
        self.matched
    }

    /// Where the earliest stop sequence begins in `text`.
    fn find(&self, text: &str) -> Option<usize> {
        self.sequences
            .iter()
            .filter_map(|stop| text.find(stop.as_str()))
            .min()
    }

    /// Take in newly decoded text and return what is safe to publish.
    pub(super) fn push(&mut self, text: String) -> Option<String> {
        if self.matched {
            return None;
        }
        if self.sequences.is_empty() {
            return Some(text);
        }
        self.held.push_str(&text);
        if let Some(at) = self.find(&self.held) {
            self.matched = true;
            self.held.truncate(at);
            return self.flush();
        }
        let keep = self
            .held
            .char_indices()
            .map(|(at, _)| at)
            .find(|&at| {
                self.held
                    .get(at..)
                    .is_some_and(|tail| self.sequences.iter().any(|stop| stop.starts_with(tail)))
            })
            .unwrap_or(self.held.len());
        let rest = self.held.split_off(keep);
        Some(std::mem::replace(&mut self.held, rest)).filter(|text| !text.is_empty())
    }

    /// Text still held back. Generation ended for another reason, so none of it completes
    /// a sequence.
    pub(super) fn flush(&mut self) -> Option<String> {
        Some(std::mem::take(&mut self.held)).filter(|text| !text.is_empty())
    }

    /// Cut a request's final text at its earliest stop sequence.
    pub(super) fn truncate(&self, text: &mut String) {
        if let Some(at) = self.find(text) {
            text.truncate(at);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn matcher(sequences: &[&str]) -> Result<StopMatcher> {
        StopMatcher::new(
            &sequences
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        )
    }

    /// Everything published for `pieces`, with whether a sequence matched.
    fn published(sequences: &[&str], pieces: &[&str]) -> Result<(String, bool)> {
        let mut stop = matcher(sequences)?;
        let mut text = String::new();
        for piece in pieces {
            text.extend(stop.push((*piece).to_owned()));
        }
        text.extend(stop.flush());
        Ok((text, stop.matched()))
    }

    #[test]
    fn test_stop_sequence_split_across_pieces_is_never_published() -> Result<()> {
        assert_eq!(
            published(&["END"], &["one E", "N", "D two"])?,
            ("one ".to_owned(), true)
        );
        // The earliest of several wins, whatever their order.
        assert_eq!(
            published(&["later", "cd"], &["abcd later"])?,
            ("ab".to_owned(), true)
        );
        Ok(())
    }

    #[test]
    fn test_partial_match_is_released_when_it_fails_or_generation_ends() -> Result<()> {
        let mut stop = matcher(&["END"])?;
        assert_eq!(stop.push("oneEN".into()).as_deref(), Some("one"));
        assert_eq!(stop.push("Ds".into()), None);
        assert!(stop.matched());
        let mut stop = matcher(&["END"])?;
        assert_eq!(stop.push("EN".into()), None);
        assert_eq!(stop.push("X".into()).as_deref(), Some("ENX"));
        assert!(!stop.matched());
        assert_eq!(
            published(&["END"], &["tail E"])?,
            ("tail E".to_owned(), false)
        );
        // Without sequences text passes through.
        assert_eq!(published(&[], &["a", "b"])?, ("ab".to_owned(), false));
        Ok(())
    }

    #[test]
    fn test_held_text_keeps_multibyte_characters_whole() -> Result<()> {
        assert_eq!(
            published(&["é!"], &["caf", "é", "!"])?,
            ("caf".to_owned(), true)
        );
        assert_eq!(
            published(&["é!"], &["日本é", "x"])?,
            ("日本éx".to_owned(), false)
        );
        Ok(())
    }

    #[test]
    fn test_final_text_is_cut_at_the_same_place_as_the_stream() -> Result<()> {
        let stop = matcher(&["b", "cd"])?;
        let mut text = "a cd b".to_owned();
        stop.truncate(&mut text);
        assert_eq!(text, "a ");
        Ok(())
    }

    #[test]
    fn test_empty_oversized_and_excess_sequences_are_rejected() {
        assert!(matcher(&[""]).is_err());
        assert!(StopMatcher::new(&["x".repeat(MAX_STOP_BYTES + 1)]).is_err());
        assert!(StopMatcher::new(&vec!["x".to_owned(); MAX_STOP_SEQUENCES + 1]).is_err());
    }
}
