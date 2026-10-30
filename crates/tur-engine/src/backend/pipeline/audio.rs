//! Audio clips in prompts: placeholder expansion, encoding and packed embedding rows.
//!
//! A clip occupies its framing tokens and one position per soft token. Those positions
//! hold identity tokens derived from the clip's samples, above every vocabulary, so prefix
//! keys and restores (which compare tokens) tell clips apart, and a snapshot taken after
//! a clip is reused only for the same clip. Packing feeds the model its slot token there
//! and the encoder's rows as [`EmbeddedInput`]s.
use super::InferencePipeline;
use crate::{Result, TurError};
use candle_core::Tensor;
use std::ops::Range;
use tur_core::execution::{EmbeddedInput, invalid};
use tur_models::backend::chat_template::ChatMessage;

/// Tokens at or above this mark a clip's soft-token position, not a vocabulary entry.
const IDENTITY: u32 = 1 << 31;

/// One clip's encoded soft tokens at prompt positions `start..start + rows`.
pub(super) struct AudioSpan {
    pub(super) start: usize,
    /// F32 `[soft tokens, hidden]` on the model device.
    pub(super) rows: Tensor,
}

impl AudioSpan {
    fn positions(&self) -> Result<Range<usize>> {
        Ok(self.start..self.start + self.rows.dim(0)?)
    }
}

/// Identity of soft token `index` of the clip with `fingerprint`: 31 hashed bits, so a
/// clip of `n` soft tokens collides with another of the same length with probability
/// 2^-31n, and only where their surrounding tokens also match.
fn identity(fingerprint: u64, index: usize) -> u32 {
    let mut hash = fingerprint ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // SplitMix64 finalizer.
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    hash ^= hash >> 31;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "keeps the low 31 bits by design"
    )]
    let low = (hash as u32) & (IDENTITY - 1);
    IDENTITY | low
}

/// Whether a prompt token is a clip identity rather than a vocabulary entry.
pub(super) fn is_identity(token: u32) -> bool {
    token >= IDENTITY
}

impl InferencePipeline {
    /// Replace each audio placeholder the template rendered with the clip's framing and
    /// identity tokens, and encode the clips. The template must render exactly one
    /// placeholder per clip, in message and part order.
    pub(super) fn expand_audio(
        &self,
        messages: &[ChatMessage],
        tokens: Vec<u32>,
    ) -> Result<(Vec<u32>, Vec<AudioSpan>)> {
        let clips: Vec<_> = messages.iter().flat_map(|m| m.content.audio()).collect();
        let Some(prompt) = self.model.audio_prompt() else {
            if clips.is_empty() {
                return Ok((tokens, Vec::new()));
            }
            return Err(invalid("this model accepts no audio input"));
        };
        let placeholders = tokens.iter().filter(|&&t| t == prompt.placeholder).count();
        if placeholders != clips.len() {
            return Err(invalid(format!(
                "the chat template rendered {placeholders} audio placeholders for {} clips",
                clips.len()
            )));
        }
        if clips.is_empty() {
            return Ok((tokens, Vec::new()));
        }
        let mut clips = clips.into_iter();
        let mut expanded = Vec::with_capacity(tokens.len());
        let mut spans = Vec::new();
        for token in tokens {
            if token != prompt.placeholder {
                expanded.push(token);
                continue;
            }
            let clip = clips
                .next()
                .ok_or_else(|| invalid("audio placeholder without a clip"))?;
            let count = self.model.audio_tokens(clip)?;
            let rows = self.model.encode_audio(clip)?;
            if rows.dim(0)? != count {
                return Err(TurError::Other(
                    "audio encoder returned the wrong number of rows".into(),
                ));
            }
            let fingerprint = clip.fingerprint();
            expanded.push(prompt.begin);
            spans.push(AudioSpan {
                start: expanded.len(),
                rows,
            });
            expanded.extend((0..count).map(|index| identity(fingerprint, index)));
            expanded.push(prompt.end);
        }
        Ok((expanded, spans))
    }

    /// The model's tokens for `tokens` at prompt positions from `start`, and the embedded
    /// rows of the clips they overlap, at packed offsets from `packed`.
    pub(super) fn embedded(
        &self,
        spans: &[AudioSpan],
        tokens: &[u32],
        (start, packed): (usize, usize),
        out: (&mut Vec<u32>, &mut Vec<EmbeddedInput>),
    ) -> Result<()> {
        let (model_tokens, embeddings) = out;
        let slot = self.model.audio_prompt().map(|p| p.slot);
        for &token in tokens {
            model_tokens.push(if is_identity(token) {
                slot.ok_or_else(|| invalid("audio positions in a text-only model"))?
            } else {
                token
            });
        }
        let window = start..start + tokens.len();
        for span in spans {
            let positions = span.positions()?;
            let (from, to) = (
                positions.start.max(window.start),
                positions.end.min(window.end),
            );
            if from >= to {
                continue;
            }
            embeddings.push(EmbeddedInput {
                tokens: packed + (from - start)..packed + (to - start),
                rows: span.rows.narrow(0, from - positions.start, to - from)?,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert identity properties"
)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_identities_lie_above_vocabularies_and_differ_by_clip_and_position() {
        let a: Vec<u32> = (0..64).map(|i| identity(7, i)).collect();
        let b: Vec<u32> = (0..64).map(|i| identity(8, i)).collect();
        assert!(a.iter().chain(&b).all(|&t| is_identity(t)));
        assert!(a.iter().zip(&b).all(|(x, y)| x != y));
        let unique: std::collections::HashSet<_> = a.iter().collect();
        assert_eq!(unique.len(), a.len());
        assert!(!is_identity(262_143));
    }
}
