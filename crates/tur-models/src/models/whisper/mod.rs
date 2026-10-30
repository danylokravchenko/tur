//! Whisper speech recognition: an encoder-decoder run outside the paged text engine.
mod config;
mod model;
mod transcribe;
pub use config::{Config, Generation};
pub use model::{Decoder, Whisper};
pub use transcribe::{Task, TranscribeOptions, Transcriber, Transcription};
