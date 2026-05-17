pub mod backend;
pub mod models;
pub mod shared;
pub mod weights;

pub use backend::{
    audio_encoder::AudioEncoder,
    factory::{AnyModel, AutoModelFactory, ModelFactory, ModelKind, ModelSource},
    pipeline::{ModalInput, InferencePipeline},
    progress::ProgressReporter,
};
pub use shared::{Result, TurError};
pub use weights::{Downloader, VarBuilderX};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
