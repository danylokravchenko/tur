//! Gemma 4 text decoder with per-layer embeddings and shared paged attention.
mod config;
mod model;
pub use config::Config;
pub use model::{Gemma4, write_artifact};
