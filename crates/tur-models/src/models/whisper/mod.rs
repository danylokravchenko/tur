//! Whisper speech recognition: an encoder-decoder run outside the paged text engine.
mod alignment;
mod config;
mod fallback;
mod model;
mod timestamps;
mod transcribe;
pub use alignment::Word;
pub use config::{Config, Generation};
pub use fallback::Fallback;
pub use model::{Decoder, Whisper};
pub use transcribe::{Segment, Task, TranscribeOptions, Transcriber, Transcription};
