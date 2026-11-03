//! Candidate logits read inside a prompt, as decision checkpoints answer: each field's answer
//! is the next-token distribution just before its marker. The prompt is cut before its last
//! marker, prefill chunks end at every read position with an output there, and the token the
//! engine samples at an inner position is dropped, since the prompt continues instead.

use super::{InferencePipeline, media::MediaSpan};
use crate::{Result, TurError};
use candle_core::{DType, Tensor};
use tur_core::execution::invalid;

/// Read candidate logits inside the prompt: before each occurrence of `marker`, those of that
/// occurrence's candidates, which must each be one token. Replaces generation: the request's
/// one output is sampled where its last marker was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scoring {
    /// A single token, such as an added `<decision>` token.
    pub marker: String,
    pub candidates: Vec<Vec<String>>,
}

/// Where a scoring request reads and what it has read. Reads survive preemption: recomputing
/// the prompt produces the same rows.
pub(super) struct Scores {
    /// Consumed prompt lengths after which a row is read: the markers' indices, increasing.
    ends: Vec<usize>,
    candidates: Vec<Vec<u32>>,
    read: Vec<Option<Vec<f32>>>,
}

impl Scores {
    /// The first read position after `consumed`, where the next prefill chunk must stop.
    pub(super) fn next_end(&self, consumed: usize) -> Option<usize> {
        self.ends.iter().copied().find(|&end| end > consumed)
    }

    /// Whether a chunk ending at `end` needs an output there.
    pub(super) fn reads_at(&self, end: usize) -> bool {
        self.ends.binary_search(&end).is_ok()
    }

    /// The longest prefix a snapshot may restore: one short of the first unread position, so
    /// the pass computing that row still runs.
    pub(super) fn restore_limit(&self) -> usize {
        self.ends
            .iter()
            .zip(&self.read)
            .find(|(_, read)| read.is_none())
            .map_or(usize::MAX, |(&end, _)| end - 1)
    }

    /// Keep the candidates' logits from `logits`, the row of a commit's only output, when
    /// `consumed` is a read position not yet read.
    pub(super) fn record(
        &mut self,
        consumed: usize,
        outputs: usize,
        logits: Option<&Tensor>,
    ) -> Result<()> {
        let Ok(index) = self.ends.binary_search(&consumed) else {
            return Ok(());
        };
        if self.read.get(index).is_some_and(Option::is_some) {
            return Ok(());
        }
        let logits = logits
            .filter(|_| outputs == 1)
            .ok_or_else(|| invalid("a scored position's logits are unavailable"))?;
        let ids = Tensor::new(self.candidates[index].as_slice(), logits.device())?;
        let row = logits
            .index_select(&ids, 1)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        self.read[index] = Some(row);
        Ok(())
    }

    /// Every position's logits, in marker order.
    pub(super) fn finish(&mut self) -> Result<Vec<Vec<f32>>> {
        self.read
            .iter_mut()
            .map(|read| {
                read.take()
                    .ok_or_else(|| invalid("a scored position was never read"))
            })
            .collect()
    }
}

impl InferencePipeline {
    fn single_token(&self, text: &str) -> Result<u32> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        match encoding.get_ids() {
            &[token] => Ok(token),
            _ => Err(invalid(format!("scored text {text:?} is not one token"))),
        }
    }

    /// Locate `scoring`'s markers in `tokens` and cut the prompt before the last one, whose
    /// row the request's final output reads. Media must lie before the cut.
    pub(super) fn scores(
        &self,
        scoring: &Scoring,
        tokens: &mut Vec<u32>,
        media: &[MediaSpan],
    ) -> Result<Scores> {
        let marker = self.single_token(&scoring.marker)?;
        let ends: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter_map(|(index, &token)| (token == marker).then_some(index))
            .collect();
        if ends.len() != scoring.candidates.len() || ends.first().is_none_or(|&end| end == 0) {
            return Err(invalid(format!(
                "the prompt holds {} scoring markers for {} candidate sets",
                ends.len(),
                scoring.candidates.len()
            )));
        }
        let cut = ends[ends.len() - 1];
        for span in media {
            if span.start + span.rows.dim(0)? > cut {
                return Err(invalid("media must precede every scoring marker"));
            }
        }
        tokens.truncate(cut);
        let candidates = scoring
            .candidates
            .iter()
            .map(|texts| {
                if texts.is_empty() {
                    return Err(invalid("a scoring marker needs candidates"));
                }
                texts.iter().map(|text| self.single_token(text)).collect()
            })
            .collect::<Result<Vec<Vec<u32>>>>()?;
        Ok(Scores {
            read: vec![None; ends.len()],
            ends,
            candidates,
        })
    }
}
