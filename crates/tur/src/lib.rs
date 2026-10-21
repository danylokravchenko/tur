pub use tur_engine::backend;
pub use tur_models::models;
pub mod server;
pub mod shared;
pub use tur_models::weights;

pub use backend::{
    audio_encoder::AudioEncoder,
    factory::{AnyModel, AutoModelFactory, ModelFactory, ModelKind, ModelSource},
    pipeline::{InferencePipeline, ModalInput},
    progress::ProgressReporter,
};
pub use shared::{Result, TurError};
pub use weights::{Downloader, VarBuilderX};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub use tur_core as core;
/// Hybrid request execution contracts.
pub use tur_engine::hybrid;
pub use tur_metal as runtime;
