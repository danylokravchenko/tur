//! Flags shared by `run` and `serve`.
use std::path::Path;

use candle_core::DType;
use clap::builder::{PossibleValuesParser, TypedValueParser as _};
use tur::{
    ModelFactory, ModelSource, Result, TurError,
    backend::{pipeline::PipelineConfig, speculation::SpeculationConfig},
    core::execution::{KvFormat, KvLayout},
    models::qwen3_5::{DEFAULT_KV_LAYOUT, WeightFormat, parse_recurrent_state},
};

/// Checkpoint and state storage.
#[derive(Debug, clap::Args)]
pub(crate) struct ModelArgs {
    /// Local checkpoint directory, or a fully qualified Hugging Face repository.
    #[arg(long, env = "TUR_MODEL")]
    model: String,
    /// Remote commit, or an explicit immutable identity for a local checkpoint.
    #[arg(long, env = "TUR_REVISION")]
    revision: String,
    /// Convert `SafeTensors` projections into this resident format: f32, `q8_0`, q4k.
    /// Omitted, a prepacked artifact loads as stored and any other checkpoint converts
    /// to `q8_0`.
    #[arg(long)]
    weight_format: Option<WeightFormat>,
    /// K/V storage: f32, f16, q8 (one scale per 32 dimensions, half of F16's memory) or
    /// `q8_<group>`.
    #[arg(long, default_value_t = DEFAULT_KV_LAYOUT.format)]
    kv_format: KvFormat,
    /// Tokens per K/V page.
    #[arg(long, default_value_t = DEFAULT_KV_LAYOUT.page_tokens)]
    kv_page_tokens: usize,
    /// Recurrent matrix state: f32, or f16, which halves recurrent memory and traffic with
    /// stochastic rounding but adds error that grows with decode length.
    #[arg(
        long,
        default_value = "f32",
        value_parser = PossibleValuesParser::new(["f32", "f16"]).try_map(|s| parse_recurrent_state(&s)),
    )]
    recurrent_state: DType,
    /// Use the CPU numerical reference for development checks.
    #[arg(long)]
    cpu_reference: bool,
}

impl ModelArgs {
    /// An existing directory loads locally; anything else names a Hugging Face repository.
    pub(crate) fn factory(&self) -> Result<ModelFactory> {
        let source = if Path::new(&self.model).is_dir() {
            ModelSource::LocalPath(self.model.clone().into())
        } else {
            ModelSource::HuggingFace(self.model.clone())
        };
        Ok(ModelFactory {
            source,
            revision: self.revision.clone(),
            format: self.weight_format,
            device: tur::shared::inference_device(self.cpu_reference)?,
            kv: KvLayout::new(self.kv_page_tokens, self.kv_format),
            recurrent: self.recurrent_state,
        })
    }
}

/// Pass shape, state memory and speculation.
#[derive(Debug, clap::Args)]
pub(crate) struct PipelineArgs {
    /// Largest prefill chunk of one request in one pass.
    #[arg(long, default_value_t = 256)]
    prefill_chunk_size: usize,
    /// Memory for K/V pages and recurrent state, in MiB. Default: room for each concurrent
    /// request's 8K-token context within the device's working set.
    #[arg(long)]
    state_memory_mb: Option<usize>,
    /// Maximum prompt-lookup drafts per decode. Opt-in throughput mode: tokens arrive in
    /// bursts and pauses can grow. Zero disables it; draft length adapts to measured cost.
    /// Requires F32 recurrent state; narrower state automatically uses ordinary decoding.
    #[arg(long, default_value_t = 0)]
    speculative_draft: usize,
    /// Also draft from the checkpoint's native multi-token prediction head. Implies at
    /// least one draft and disables prefix reuse.
    #[arg(long)]
    mtp: bool,
}

impl PipelineArgs {
    pub(crate) fn config(&self, max_batch_size: usize) -> Result<PipelineConfig> {
        let state_bytes = self
            .state_memory_mb
            .map(|mb| {
                mb.checked_mul(1 << 20).ok_or_else(|| {
                    TurError::Other("state memory in MiB exceeds usize bytes".into())
                })
            })
            .transpose()?;
        Ok(PipelineConfig {
            speculation: if self.mtp {
                SpeculationConfig::with_mtp(self.speculative_draft.max(1))
            } else {
                SpeculationConfig::new(self.speculative_draft)
            },
            state_bytes,
            ..PipelineConfig::new(max_batch_size, self.prefill_chunk_size)
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible configuration"
)]
mod tests {
    use super::*;

    #[test]
    fn test_state_memory_overflow_is_rejected() -> Result<()> {
        let max_mb = usize::MAX / (1 << 20);
        let mut args = PipelineArgs {
            prefill_chunk_size: 256,
            state_memory_mb: Some(max_mb),
            speculative_draft: 0,
            mtp: false,
        };
        assert_eq!(args.config(1)?.state_bytes, Some(max_mb * (1 << 20)));
        args.state_memory_mb = Some(max_mb + 1);
        assert!(args.config(1).is_err());
        Ok(())
    }
}
