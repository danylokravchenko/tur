#![forbid(unsafe_code)]
//! Explicit-state packed execution contract over persistent state pools.

use candle_core::Tensor;
use std::ops::Range;
use tur_core::{Result, execution::ModelDescription};
pub use tur_metal::{
    pool::StatePool,
    state::{Addresses, Boundaries, SequenceState},
};

/// One request's slice of a packed forward pass.
pub struct SequenceForward<'a> {
    /// Range into [`PackedForward::tokens`] and [`PackedForward::positions`].
    pub tokens: Range<usize>,
    /// Pool addressing: committed pages and recurrent slot, plus the pages and slots
    /// reserved for this pass's new tokens and boundaries.
    pub state: &'a SequenceState,
    /// Strictly increasing packed positions whose logits are required.
    pub logits: &'a [usize],
}

/// Variable-length requests packed without padding.
pub struct PackedForward<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [usize],
    pub sequences: &'a [SequenceForward<'a>],
    /// Persistent state of every sequence; built from the model's own description.
    pub pool: &'a StatePool,
}

/// One request's output: F32 `[logits.len(), vocab]` logits. State is written in place.
pub struct SequenceStep {
    pub logits: Tensor,
}

/// Shared immutable model weights. All mutable sequence state lives in the pool.
///
/// A pass writes keys and values for every new token at their reserved page positions,
/// and the recurrent state after the last token (or after every token, as the sequence's
/// [`Boundaries`] request) into its reserved slots. It must not write anywhere else:
/// committed pages and slots are shared with checkpoints and other sequences.
pub trait HybridModel: Send + Sync + 'static {
    /// Declare the exact model identity, state layout, modalities, vocabulary and limits.
    fn description(&self) -> ModelDescription;

    /// Kernel scratch this plan needs beyond its inputs, outputs and pools, for admission.
    fn workspace_bytes(&self, _batch: &PackedForward<'_>) -> Result<usize> {
        Ok(0)
    }

    /// Encode one packed pass. The outer error fails the whole batch (a shared dispatch);
    /// each inner result is one sequence, in order, so one request's failure is isolated.
    /// Implementations should only enqueue device work; the engine fences before reading.
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>>;
}
