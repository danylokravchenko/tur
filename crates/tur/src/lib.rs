//! Public facade over the Tur crates, plus the HTTP server library and adapter setup.
pub mod server;
pub mod shared;

pub use tur_core as core;
pub use tur_engine::backend;
/// Hybrid request execution contracts.
pub use tur_engine::hybrid;
pub use tur_metal as runtime;
pub use tur_models::{models, quantization};

pub use backend::{
    factory::{ModelFactory, ModelSource},
    pipeline::InferencePipeline,
};
pub use shared::{Result, TurError};
