#![forbid(unsafe_code)]
//! Dense Qwen3.5 text model, shared by verified Qwen3.8 configs.
mod artifact;
mod config;
mod math;
mod model;
mod quantize;

pub use artifact::{ARTIFACT_FILE, CalibrationInfo, Manifest, Quantizer, is_artifact};
pub use config::Config;
pub use model::Qwen35;
pub use quantize::{
    Conversion, Reference, Sensitivity, allocate, artifact_config, sensitivities, write_artifact,
    write_calibrated_artifact,
};
