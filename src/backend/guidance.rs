use std::sync::Arc;

use crate::{Result, TurError};
use llguidance::{Constraint, ParserFactory as LlgParserFactory};
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::ByteTokenizer;
use tracing::warn;

pub type ParserFactory = LlgParserFactory;
pub use llguidance::api::TopLevelGrammar;

pub fn build_llg_factory(
    tokenizer: Tokenizer,
    vocab_size: Option<usize>,
) -> Result<Arc<ParserFactory>> {
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    let target_vocab = vocab_size.map(|v| {
        if v < tokenizer_vocab {
            warn!(
                "Requested vocab size {} is smaller than tokenizer vocab size {}. Using tokenizer size.",
                v,
                tokenizer_vocab
            );
            tokenizer_vocab
        } else {
            v
        }
    });
    let env = ByteTokenizer::from_tokenizer(tokenizer)
        .map_err(|e| TurError::Guidance(e.to_string()))?
        .into_tok_env(target_vocab)
        .map_err(|e| TurError::Guidance(e.to_string()))?;
    let factory = ParserFactory::new_simple(&env).map_err(|e| TurError::Guidance(e.to_string()))?;
    Ok(Arc::new(factory))
}

/// Per-request grammar constraint. Activated before prefill, dropped on completion.
///
/// The loop is: `apply_mask()` before each sample → `commit()` after each sample.
pub struct GuidanceControl {
    factory: Arc<ParserFactory>,
    constraint: Option<Constraint>,
    /// Set when the grammar reaches an accepting terminal state so we stop
    /// calling `compute_mask()` (which panics if called after stop).
    stopped: bool,
}

impl GuidanceControl {
    pub fn new(factory: Arc<ParserFactory>) -> Self {
        Self {
            factory,
            constraint: None,
            stopped: false,
        }
    }

    /// Create and start a new constraint for this grammar. Call once before prefill.
    pub fn activate(&mut self, grammar: TopLevelGrammar) -> Result<()> {
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(|e| TurError::Guidance(e.to_string()))?;
        let mut constraint = Constraint::new(parser);
        constraint.start_without_prompt();
        self.constraint = Some(constraint);
        self.stopped = false;
        Ok(())
    }

    /// Drop the active constraint. Call after generation completes or on error.
    pub fn deactivate(&mut self) {
        self.constraint = None;
        self.stopped = false;
    }

    /// Returns `true` while the constraint is active and has not yet reached a
    /// terminal state.
    pub fn is_active(&self) -> bool {
        self.constraint.is_some() && !self.stopped
    }

    /// Compute the grammar token mask and apply it to `logits` in place.
    /// Returns `true` if the grammar signals that generation should stop.
    /// Once the grammar stops, subsequent calls are no-ops (returns `false`).
    pub fn apply_mask(&mut self, logits: &mut [f32]) -> Result<bool> {
        if self.stopped {
            return Ok(true);
        }
        let Some(constraint) = self.constraint.as_mut() else {
            return Ok(false);
        };

        let step = constraint
            .compute_mask()
            .map_err(|e| TurError::Guidance(e.to_string()))?;

        if step.is_stop() {
            self.stopped = true;
            return Ok(true);
        }

        if let Some(mask) = &step.sample_mask {
            // `apply_to` sets each **allowed** token's logit to 0.0, leaving
            // disallowed positions unchanged.  We use a bias array initialised
            // to -∞ so that after the call only allowed positions have finite
            // values, then add the bias to the raw logits.
            let mut bias = vec![f32::NEG_INFINITY; logits.len()];
            mask.apply_to(&mut bias);
            for (l, b) in logits.iter_mut().zip(bias.iter()) {
                if b.is_finite() {
                    // token is allowed — keep logit as-is (bias is 0.0)
                } else {
                    *l = f32::NEG_INFINITY;
                }
            }
        }

        Ok(false)
    }

    /// Advance the grammar state after a token has been sampled.
    /// Returns `true` if the grammar signals stop after this token.
    pub fn commit(&mut self, token: u32) -> Result<bool> {
        if self.stopped {
            return Ok(true);
        }
        let Some(constraint) = self.constraint.as_mut() else {
            return Ok(false);
        };
        let result = constraint
            .commit_token(Some(token))
            .map_err(|e| TurError::Guidance(e.to_string()))?;
        if result.stop {
            self.stopped = true;
        }
        Ok(result.stop)
    }
}
