use std::sync::Arc;

use crate::{Result, TurError};
use llguidance::{Constraint, ParserFactory as LlgParserFactory, toktrie::TokTrie};
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::ByteTokenizer;
use tracing::warn;

pub type ParserFactory = LlgParserFactory;

/// The grammar's constraint on the next token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrammarMask {
    /// The grammar is done; nothing more is sampled.
    Stop,
    /// Bit `t % 32` of word `t / 32` is set for every allowed token `t`.
    Allowed(Vec<u32>),
}
pub use llguidance::api::TopLevelGrammar;

/// Build the parser factory over `tokenizer`. `eos_tokens` are the checkpoint's ends of
/// generation, which a finished grammar allows next; llguidance would otherwise guess one
/// from token names, and a mask would hide the token the model actually ends its turn with.
pub fn build_llg_factory(
    tokenizer: Tokenizer,
    vocab_size: Option<usize>,
    eos_tokens: &[u32],
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
    let mut bytes =
        ByteTokenizer::from_tokenizer(tokenizer).map_err(|e| TurError::Guidance(e.to_string()))?;
    if !eos_tokens.is_empty() {
        bytes.set_eos_tokens(eos_tokens);
    }
    let env = bytes
        .into_tok_env(target_vocab)
        .map_err(|e| TurError::Guidance(e.to_string()))?;
    let factory = ParserFactory::new_simple(&env).map_err(|e| TurError::Guidance(e.to_string()))?;
    Ok(Arc::new(factory))
}

/// The exact bytes of `token`, without the marker llguidance puts before special tokens.
/// Decoding one token as text would lose bytes of a character split across tokens.
pub(crate) fn token_bytes(factory: &ParserFactory, token: u32) -> Vec<u8> {
    let bytes = factory.tok_env().tok_trie().token(token);
    bytes
        .strip_prefix(&[TokTrie::SPECIAL_TOKEN_MARKER])
        .unwrap_or(bytes)
        .to_vec()
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
    /// Check vocabulary bounds and single-token semantics before hybrid request admission.
    /// Splices/backtracking require model-state transitions that this interface cannot publish.
    pub fn validate_token_contract(&self, vocab_size: usize) -> Result<()> {
        if self.factory.tok_env().tok_trie().vocab_size() != vocab_size {
            return Err(TurError::Guidance(
                "grammar/model vocabulary mismatch".into(),
            ));
        }
        if let Some(constraint) = &self.constraint {
            let caps = &constraint.parser.inference_caps;
            if caps.ff_tokens || caps.conditional_ff_tokens || caps.backtrack || caps.fork {
                return Err(TurError::Guidance(
                    "hybrid execution requires single-token grammar commits".into(),
                ));
            }
        }
        Ok(())
    }

    /// Deep-copy parser state for an independent request checkpoint or speculative branch.
    #[must_use]
    pub fn fork(&self) -> Self {
        Self {
            factory: Arc::clone(&self.factory),
            constraint: self.constraint.as_ref().map(Constraint::deep_clone),
            stopped: self.stopped,
        }
    }

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

    /// The allowed-token mask for the next sample without publishing anything, so it can be
    /// computed while the device runs and applied there. Recomputing before the next
    /// [`Self::commit`] gives the same mask. `None` without an active constraint.
    pub fn mask(&mut self) -> Result<Option<GrammarMask>> {
        if self.stopped {
            return Ok(Some(GrammarMask::Stop));
        }
        let Some(constraint) = self.constraint.as_mut() else {
            return Ok(None);
        };
        // A stop seen by an earlier mask must not be computed again.
        if constraint.step_result().is_stop() {
            return Ok(Some(GrammarMask::Stop));
        }
        let step = constraint
            .compute_mask()
            .map_err(|e| TurError::Guidance(e.to_string()))?;
        if step.is_stop() {
            return Ok(Some(GrammarMask::Stop));
        }
        Ok(step
            .sample_mask
            .as_ref()
            .map(|mask| GrammarMask::Allowed(mask.as_slice().to_vec())))
    }

    /// Record that the grammar stopped before sampling, as [`Self::apply_mask`] does.
    pub fn stop(&mut self) {
        self.stopped = true;
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
