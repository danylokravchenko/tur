//! Clips and images in prompts: placeholder expansion, encoding and packed embedding rows.
//!
//! An input occupies its framing tokens and one position per soft token. Those positions
//! hold identity tokens derived from the input's content, above every vocabulary, so prefix
//! keys and restores (which compare tokens) tell inputs apart, and a snapshot taken after
//! one is reused only for the same one. Packing feeds the model its slot token there
//! and the encoder's rows as [`EmbeddedInput`]s.
use super::{GenerationRequest, InferencePipeline};
use crate::{Result, TurError};
use candle_core::Tensor;
use std::{ops::Range, sync::Arc};
use tur_core::execution::{EmbeddedInput, invalid};
use tur_models::{
    media::{Digest, Media},
    models::TextModel,
};

/// Tokens at or above this mark a soft-token position, not a vocabulary entry.
const IDENTITY: u32 = 1 << 31;

/// One input's encoded soft tokens at prompt positions `start..start + rows`.
pub(super) struct MediaSpan {
    pub(super) start: usize,
    /// F32 `[soft tokens, hidden]` on the model device.
    pub(super) rows: Tensor,
    /// `(rows, columns)` of soft tokens when text rotary reads them as a grid.
    pub(super) grid: Option<(usize, usize)>,
}

impl MediaSpan {
    fn positions(&self) -> Result<Range<usize>> {
        Ok(self.start..self.start + self.rows.dim(0)?)
    }
}

/// Identity of soft token `index` of the input with `digest`: 31 hashed bits, so an input of
/// `n` soft tokens collides with another of the same length with probability 2^-31n, and only
/// where their surrounding tokens also match.
fn identity(digest: &Digest, index: usize) -> u32 {
    let mut leading = [0u8; 8];
    leading.copy_from_slice(&digest[..8]);
    let fingerprint = u64::from_le_bytes(leading);
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

/// Whether a prompt token is an input's identity rather than a vocabulary entry.
pub(super) fn is_identity(token: u32) -> bool {
    token >= IDENTITY
}

/// Rotary (t, h, w) positions of prompt positions `range`, as `get_rope_index` lays them out:
/// text counts up, a grid's soft tokens read their row and column from where it starts, and
/// the text after it continues past the grid's longer side instead of past its tokens.
pub(super) fn axial_positions(
    spans: &[MediaSpan],
    range: Range<usize>,
    out: &mut Vec<[u32; 3]>,
) -> Result<()> {
    let to_u32 = |value: usize| u32::try_from(value).map_err(|_| invalid("position overflow"));
    let mut spans = spans
        .iter()
        .filter_map(|span| Some((span, span.grid?)))
        .peekable();
    // Text before the next grid at position `at` rotates at `at + shift` (in `base`'s terms).
    let (mut slot, mut rotary) = (0usize, 0usize);
    for position in range {
        while let Some(&(span, (rows, columns))) = spans.peek() {
            let extent = span.positions()?;
            if position < extent.end {
                break;
            }
            rotary += extent.start - slot + rows.max(columns);
            slot = extent.end;
            spans.next();
        }
        let row = match spans.peek() {
            Some(&(span, (_, columns))) if position >= span.start => {
                let base = rotary + (span.start - slot);
                let index = position - span.start;
                [
                    to_u32(base)?,
                    to_u32(base + index / columns)?,
                    to_u32(base + index % columns)?,
                ]
            }
            _ => [to_u32(rotary + (position - slot))?; 3],
        };
        out.push(row);
    }
    Ok(())
}

impl InferencePipeline {
    /// The model, for encoding inputs off the execution thread, when it reads any.
    pub fn media_model(&self) -> Option<Arc<TextModel>> {
        self.model.has_encoders().then(|| Arc::clone(&self.model))
    }

    /// Replace each placeholder the template rendered with its input's framing and identity
    /// tokens, and encode the inputs. The template must render exactly one placeholder per
    /// input, in message and part order. Inputs the request brought encoded are not encoded
    /// again.
    pub(super) fn expand_media(
        &self,
        request: &GenerationRequest,
        tokens: Vec<u32>,
    ) -> Result<(Vec<u32>, Vec<MediaSpan>)> {
        let inputs: Vec<Media<'_>> = request
            .messages
            .iter()
            .flat_map(|m| m.content.media())
            .collect();
        if !request.encoded_media.is_empty() && request.encoded_media.len() != inputs.len() {
            return Err(invalid("encoded media does not match the request's inputs"));
        }
        for input in &inputs {
            if self.model.media_prompt(*input).is_none() {
                return Err(invalid(match input {
                    Media::Audio(_) => "this model accepts no audio input",
                    Media::Image(_) => "this model accepts no image input",
                    Media::Video(_) => "this model accepts no video input",
                }));
            }
        }
        let prompts = [
            self.model.audio_prompt(),
            self.model.image_prompt(),
            self.model.video_prompt(),
        ];
        let placeholder = |token: u32| {
            prompts
                .iter()
                .flatten()
                .any(|prompt| prompt.placeholder == token)
        };
        let placeholders = tokens.iter().filter(|&&t| placeholder(t)).count();
        if placeholders != inputs.len() {
            return Err(invalid(format!(
                "the chat template rendered {placeholders} media placeholders for {} inputs",
                inputs.len()
            )));
        }
        if inputs.is_empty() {
            return Ok((tokens, Vec::new()));
        }
        let mut encoded = request.encoded_media.iter();
        let mut inputs = inputs.into_iter();
        let mut expanded = Vec::with_capacity(tokens.len());
        let mut spans = Vec::new();
        for token in tokens {
            if !placeholder(token) {
                expanded.push(token);
                continue;
            }
            let input = inputs
                .next()
                .ok_or_else(|| invalid("media placeholder without an input"))?;
            let rows = match encoded.next() {
                Some(rows) => rows.clone(),
                None => self.model.encode_media(input)?,
            };
            self.expand_input(input, token, &rows, &mut expanded, &mut spans)?;
        }
        Ok((expanded, spans))
    }

    /// Write one input's segments in place of its placeholder `token`.
    fn expand_input(
        &self,
        input: Media<'_>,
        token: u32,
        rows: &Tensor,
        expanded: &mut Vec<u32>,
        spans: &mut Vec<MediaSpan>,
    ) -> Result<()> {
        let prompt = self
            .model
            .media_prompt(input)
            .filter(|prompt| prompt.placeholder == token)
            .ok_or_else(|| invalid("the chat template reordered the request's media"))?;
        let layout = self.model.media_layout(input)?;
        if rows.dim(0)? != layout.tokens() {
            return Err(TurError::Other(
                "media encoder returned the wrong number of rows".into(),
            ));
        }
        let digest = input.digest();
        let mut at = 0;
        for segment in &layout.segments {
            if !segment.label.is_empty() {
                let label = self
                    .tokenizer
                    .encode(segment.label.as_str(), false)
                    .map_err(|error| TurError::Tokenizer(error.to_string()))?;
                expanded.extend_from_slice(label.get_ids());
            }
            expanded.extend(prompt.begin);
            spans.push(MediaSpan {
                start: expanded.len(),
                rows: rows.narrow(0, at, segment.tokens)?,
                grid: segment.grid,
            });
            expanded.extend((at..at + segment.tokens).map(|index| identity(&digest, index)));
            expanded.extend(prompt.end);
            at += segment.tokens;
        }
        Ok(())
    }

    /// The model's tokens for `tokens` at prompt positions from `start`, and the embedded
    /// rows of the inputs they overlap, at packed offsets from `packed`.
    pub(super) fn embedded(
        &self,
        spans: &[MediaSpan],
        tokens: &[u32],
        (start, packed): (usize, usize),
        out: (&mut Vec<u32>, &mut Vec<EmbeddedInput>),
    ) -> Result<()> {
        let (model_tokens, embeddings) = out;
        let slot = self.model.media_slot();
        for &token in tokens {
            model_tokens.push(if is_identity(token) {
                slot.ok_or_else(|| invalid("media positions in a text-only model"))?
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
    use candle_core::{DType, Device};

    fn span(start: usize, tokens: usize, grid: (usize, usize)) -> Result<MediaSpan> {
        Ok(MediaSpan {
            start,
            rows: Tensor::zeros((tokens, 1), DType::F32, &Device::Cpu)?,
            grid: Some(grid),
        })
    }

    /// Expected values from `Qwen3_5Model.get_rope_index` on two images, the second wider.
    #[test]
    fn test_axial_positions_follow_each_grid_and_continue_past_its_longer_side() -> Result<()> {
        let spans = [span(3, 6, (3, 2))?, span(10, 2, (1, 2))?];
        let mut all = Vec::new();
        axial_positions(&spans, 0..14, &mut all)?;
        let expected = [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [3, 3, 4],
            [3, 4, 3],
            [3, 4, 4],
            [3, 5, 3],
            [3, 5, 4],
            [6, 6, 6],
            [7, 7, 7],
            [7, 7, 8],
            [9, 9, 9],
            [10, 10, 10],
        ];
        assert_eq!(all, expected);
        for start in 0..14 {
            let mut part = Vec::new();
            axial_positions(&spans, start..14, &mut part)?;
            assert_eq!(part, &expected[start..]);
        }
        Ok(())
    }

    #[test]
    fn test_clip_identities_lie_above_vocabularies_and_differ_by_clip_and_position() {
        let a: Vec<u32> = (0..64).map(|i| identity(&[7; 32], i)).collect();
        let b: Vec<u32> = (0..64).map(|i| identity(&[8; 32], i)).collect();
        assert!(a.iter().chain(&b).all(|&t| is_identity(t)));
        assert!(a.iter().zip(&b).all(|(x, y)| x != y));
        let unique: std::collections::HashSet<_> = a.iter().collect();
        assert_eq!(unique.len(), a.len());
        assert!(!is_identity(262_143));
    }
}
