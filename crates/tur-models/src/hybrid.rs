#![forbid(unsafe_code)]
//! Explicit-state reference execution contract. Legacy decoders remain in `models`.

use candle_core::Tensor;
use tur_core::{Result, execution::ModelDescription};
use tur_metal::state::{LayerRead, LayerUpdate};

/// All layer updates and vocabulary logits at one exact token boundary.
pub struct TokenStep {
    pub layers: Vec<LayerUpdate>,
    pub logits: Tensor,
}

/// Shared immutable model weights. All mutable sequence state is passed explicitly.
/// Implementations must not retain or mutate inputs after returning, or hide request state.
/// P2 executes one token at a time; packed GPU mathematics belongs to later packages.
pub trait HybridModel: Send + Sync + 'static {
    /// Declare the exact model identity, state layout, vocabulary and positional limits.
    fn description(&self) -> ModelDescription;

    /// Compute a complete token transition without changing request state or weights.
    /// Inputs are detached from live/checkpoint state; returned tensors are validated and copied.
    fn step(&self, token: u32, position: usize, state: &[LayerRead]) -> Result<TokenStep>;
}
