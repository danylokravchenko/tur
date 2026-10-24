#![forbid(unsafe_code)]
//! Dense Qwen3.5 text model, shared by verified Qwen3.8 configs.
mod artifact;
mod calibration;
mod config;
mod math;
mod model;
mod plan;
mod quantize;
mod weights;

pub use artifact::{ARTIFACT_FILE, CalibrationInfo, Manifest, Quantizer, is_artifact};
pub use calibration::{Calibration, Site};
pub use config::{Config, GenerationDefaults, Projection};
pub use model::{
    DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, Execution, LoadOptions, Qwen35, WeightFormat,
    Weights, parse_recurrent_state,
};
pub use plan::{Embedding, Precision, WeightPlan};
pub use quantize::{
    Conversion, Method, Packer, Reference, Sensitivity, allocate, artifact_config, sensitivities,
    write_artifact,
};
