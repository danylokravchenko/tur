#![forbid(unsafe_code)]
//! Dense Qwen3.5 text model, shared by verified Qwen3.8 configs.
mod config;
mod math;
mod model;
mod quantize;

pub use config::Config;
pub use model::Qwen35;
pub use quantize::{
    Reference, Sensitivity, allocate, artifact_config, sensitivities, write_artifact,
    write_calibrated_artifact,
};
