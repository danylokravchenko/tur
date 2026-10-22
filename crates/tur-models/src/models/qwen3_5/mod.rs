#![forbid(unsafe_code)]
//! Dense Qwen3.5 text model, shared by verified Qwen3.8 configs.
mod config;
mod math;
mod model;
mod weights;

pub use config::{Config, GenerationDefaults};
pub use model::{LoadOptions, Qwen35, WeightFormat};
