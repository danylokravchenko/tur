//! Hybrid text models.
mod artifact;
mod checkpoint;
mod encoder;
mod generation;
mod gguf;
mod math;
mod media_cache;
mod options;
#[cfg(feature = "metal")]
mod paging;
pub use artifact::{ARTIFACT_FILE, CalibrationInfo, Conversion, Manifest, Quantizer, is_artifact};
pub use checkpoint::Adjust;
pub use generation::GenerationDefaults;
pub use gguf::{GgufFile, GgufTensor};
pub(crate) use gguf::{shard_name, split_name};
pub(crate) use weights::STAGING_BYTES;
pub mod weights;
pub use options::{
    DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, DraftHead, Execution, LoadOptions, SparseAttention,
    WeightFormat, Weights, parse_recurrent_state,
};
pub use weights::{Embedding, Method, Packer, Precision, Projection, WeightPlan};
pub mod gemma4;
pub mod gpt_oss;
pub mod qwen3_5;
pub mod whisper;

mod text;
pub use text::{ModelConfig, TextModel};
