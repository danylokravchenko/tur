use candle_core::Tensor;

use crate::Result;

/// Encoder that converts raw PCM audio into a sequence of embedding vectors
/// suitable for injection into a multimodal LLM's forward pass.
///
/// Implement this trait for a model's audio tower (e.g. a Whisper-style encoder).
/// The pipeline holds an `Option<Box<dyn AudioEncoder>>` and calls `encode` for
/// every [`ModalInput::Audio`](super::pipeline::ModalInput) in a request before
/// dispatching the embeddings to [`InferenceEngine::prefill_with_audio`].
pub trait AudioEncoder: Send + Sync {
    /// Encode `pcm` samples (f32, mono, at `sample_rate` Hz) into a tensor of
    /// shape `[T, D]` where `T` is the number of audio frames and `D` is the
    /// model's hidden dimension.
    fn encode(&self, pcm: &[f32], sample_rate: u32) -> Result<Tensor>;
}
