//! Hybrid text models.
mod checkpoint;
mod generation;
mod math;
mod options;
pub use generation::GenerationDefaults;
pub(crate) use weights::STAGING_BYTES;
pub mod weights;
pub use options::{
    DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, DraftHead, Execution, LoadOptions, WeightFormat,
    Weights, parse_recurrent_state,
};
pub use weights::{Embedding, Method, Packer, Precision, Projection, WeightPlan};
pub mod gemma4;
pub mod qwen3_5;

mod text;
pub use text::{ModelConfig, TextModel};
