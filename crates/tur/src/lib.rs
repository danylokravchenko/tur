pub use tur_engine::backend;
pub use tur_models::models;
pub mod server;
pub mod shared;
pub use backend::{
    factory::{ModelFactory, ModelSource},
    pipeline::InferencePipeline,
};
pub use shared::{Result, TurError};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub use tur_core as core;
/// Hybrid request execution contracts.
pub use tur_engine::hybrid;
pub use tur_metal as runtime;
