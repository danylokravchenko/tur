#![forbid(unsafe_code)]
//! Explicit-state packed execution contract over persistent state pools.

use candle_core::Tensor;
use std::ops::Range;
use tur_core::{
    Result,
    execution::{EmbeddedInput, ModelDescription, invalid},
};
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
    /// Also return the final normalized hidden state of every new token, which a draft
    /// head consumes ([`HybridModel::draft`]).
    pub hidden: bool,
}

/// Variable-length requests packed without padding.
pub struct PackedForward<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [usize],
    pub sequences: &'a [SequenceForward<'a>],
    /// Rows replacing the input embeddings of packed token ranges, validated by
    /// [`tur_core::execution::PackedExecution::validate`].
    pub embeddings: &'a [EmbeddedInput],
    /// Persistent state of every sequence; built from the model's own description.
    pub pool: &'a StatePool,
}

/// One request's output: F32 `[logits.len(), vocab]` logits. State is written in place.
pub struct SequenceStep {
    pub logits: Tensor,
    /// F32 `[tokens, hidden]` final normalized hidden states, when requested.
    pub hidden: Option<Tensor>,
}

/// One request's slice of a draft-head pass over committed positions `p`: the head pairs
/// each position's final hidden state with the token that followed it, and predicts the
/// token after that.
pub struct DraftSequence<'a> {
    /// Range into [`DraftForward::tokens`] (the token at `p + 1`) and
    /// [`DraftForward::positions`] (`p`).
    pub tokens: Range<usize>,
    /// Side addressing of the committed positions ([`tur_metal::state::ModelState::side`]).
    pub state: &'a SequenceState,
    /// F32 `[tokens, hidden]` final normalized hidden states at the positions.
    pub hidden: &'a Tensor,
    /// Chained drafts to return logits for after the last position: row `i` drafts the
    /// token `i + 2` positions after it, each row conditioned on the greedy drafts before
    /// it. Zero returns none. A head that cannot chain returns at most one row.
    pub drafts: usize,
}

/// A packed draft-head pass. It writes only the head's own state layers.
pub struct DraftForward<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [usize],
    pub sequences: &'a [DraftSequence<'a>],
    pub pool: &'a StatePool,
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

    /// Compiled Metal kernels the engine may reuse for logit reductions, when the model
    /// runs on them; `None` keeps the engine's host path.
    #[cfg(feature = "metal")]
    fn kernels(&self) -> Option<tur_metal::kernels::Kernels> {
        None
    }

    /// Encode one packed pass. The outer error fails the whole batch (a shared dispatch);
    /// each inner result is one sequence, in order, so one request's failure is isolated.
    /// Implementations should only enqueue device work; the engine fences before reading.
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>>;

    /// Encode a draft-head pass (native multi-token prediction): per sequence, the F32
    /// `[n, vocab]` logits of up to [`DraftSequence::drafts`] chained drafts after its last
    /// position, or `None` when none were requested. Models without a head reject it.
    fn draft(&self, _batch: &DraftForward<'_>) -> Result<Vec<Result<Option<Tensor>>>> {
        Err(invalid("model has no draft head"))
    }
}
