#![forbid(unsafe_code)]
//! Explicit-state packed execution contract. Legacy decoders remain in `models`.

use candle_core::Tensor;
use std::ops::Range;
use tur_core::{Result, execution::ModelDescription};
pub use tur_metal::state::{Boundaries, LayerRead, LayerUpdate};

/// One request's slice of a packed forward pass.
pub struct SequenceForward<'a> {
    /// Range into [`PackedForward::tokens`] and [`PackedForward::positions`].
    pub tokens: Range<usize>,
    /// Committed state before the first token; shared storage, read-only.
    pub state: &'a [LayerRead],
    /// Strictly increasing packed positions whose logits are required.
    pub logits: &'a [usize],
    /// Recurrent states to return; attention always returns K/V for every token.
    pub boundaries: Boundaries,
}

/// Variable-length requests packed without padding.
pub struct PackedForward<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [usize],
    pub sequences: &'a [SequenceForward<'a>],
}

/// One request's output: complete per-layer updates and F32 `[logits.len(), vocab]` logits.
pub struct SequenceStep {
    pub layers: Vec<LayerUpdate>,
    pub logits: Tensor,
}

/// Shared immutable model weights. All mutable sequence state is passed explicitly.
/// Implementations must not retain or mutate inputs after returning, or hide request state.
pub trait HybridModel: Send + Sync + 'static {
    /// Declare the exact model identity, state layout, modalities, vocabulary and limits.
    fn description(&self) -> ModelDescription;

    /// Kernel scratch this plan needs beyond its inputs and outputs, for admission.
    fn workspace_bytes(&self, _batch: &PackedForward<'_>) -> Result<usize> {
        Ok(0)
    }

    /// Encode one packed pass. The outer error fails the whole batch (a shared dispatch);
    /// each inner result is one sequence, in order, so one request's failure is isolated.
    /// Implementations should only enqueue device work; the engine fences before reading.
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>>;
}
