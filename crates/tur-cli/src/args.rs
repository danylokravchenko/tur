//! Flags shared by `run` and `serve`.
use std::{path::Path, time::Duration};

use candle_core::DType;
use clap::builder::{PossibleValuesParser, TypedValueParser as _};
use serde::Deserialize;
use tur::{
    ModelFactory, ModelSource, Result, TurError,
    backend::{pipeline::PipelineConfig, speculation::SpeculationConfig},
    core::execution::{KvFormat, KvLayout},
    models::qwen3_5::{DEFAULT_KV_LAYOUT, WeightFormat, parse_recurrent_state},
};

use crate::config::{inherit, optional_from_str, optional_recurrent};

/// Checkpoint and state storage.
#[derive(Debug, Default, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ModelArgs {
    /// Local checkpoint directory, or a fully qualified Hugging Face repository.
    #[arg(long, env = "TUR_MODEL")]
    #[serde(rename = "source")]
    model: Option<String>,
    /// Remote commit, or an explicit immutable identity for a local checkpoint.
    #[arg(long, env = "TUR_REVISION")]
    revision: Option<String>,
    /// Convert `SafeTensors` projections into this resident format: f32, `q8_0`, q4k.
    /// Omitted, a prepacked artifact loads as stored and any other checkpoint converts
    /// to `q8_0`.
    #[arg(long)]
    #[serde(deserialize_with = "optional_from_str")]
    weight_format: Option<WeightFormat>,
    /// K/V storage: f32, f16, q8 (one scale per 32 dimensions, half of F16's memory) or
    /// `q8_<group>` (default: f16).
    #[arg(long)]
    #[serde(deserialize_with = "optional_from_str")]
    kv_format: Option<KvFormat>,
    /// Tokens per K/V page (default: 16).
    #[arg(long)]
    kv_page_tokens: Option<usize>,
    /// Recurrent matrix state: f32, or f16, which halves recurrent memory and traffic with
    /// stochastic rounding but adds error that grows with decode length (default: f32).
    #[arg(
        long,
        value_parser = PossibleValuesParser::new(["f32", "f16"]).try_map(|s| parse_recurrent_state(&s)),
    )]
    #[serde(deserialize_with = "optional_recurrent")]
    recurrent_state: Option<DType>,
    /// Use the CPU numerical reference for development checks.
    #[arg(long)]
    #[serde(skip)]
    cpu_reference: bool,
}

impl ModelArgs {
    inherit!(
        model,
        revision,
        weight_format,
        kv_format,
        kv_page_tokens,
        recurrent_state
    );

    /// Explicit paths and existing directories load locally; other values name a Hugging Face repository.
    pub(crate) fn factory(&self) -> Result<ModelFactory> {
        let model = self
            .model
            .as_ref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                TurError::Other(
                    "model is required: set --model, TUR_MODEL or [model].source in tur.toml"
                        .into(),
                )
            })?;
        let revision = self.revision.as_ref().filter(|value| !value.trim().is_empty())
            .ok_or_else(|| TurError::Other("revision is required: set --revision, TUR_REVISION or [model].revision in tur.toml".into()))?;
        let path = Path::new(model);
        let source = if path.is_dir()
            || path.is_absolute()
            || path.starts_with(".")
            || path.starts_with("..")
        {
            ModelSource::LocalPath(model.into())
        } else {
            ModelSource::HuggingFace(model.clone())
        };
        if self.kv_layout().page_tokens == 0 {
            return Err(TurError::Other("K/V page tokens must be positive".into()));
        }
        Ok(ModelFactory {
            source,
            revision: revision.clone(),
            format: self.weight_format,
            device: tur::shared::inference_device(self.cpu_reference)?,
            kv: self.kv_layout(),
            recurrent: self.recurrent_state.unwrap_or(DType::F32),
        })
    }

    pub(crate) fn relative_to(&mut self, directory: &Path) {
        if let Some(model) = &mut self.model {
            let path = Path::new(model);
            if path.is_relative()
                && (path.starts_with(".")
                    || path.starts_with("..")
                    || directory.join(path).is_dir())
            {
                *model = directory.join(path).to_string_lossy().into_owned();
            }
        }
    }

    pub(crate) fn kv_layout(&self) -> KvLayout {
        KvLayout::new(
            self.kv_page_tokens.unwrap_or(DEFAULT_KV_LAYOUT.page_tokens),
            self.kv_format.unwrap_or(DEFAULT_KV_LAYOUT.format),
        )
    }
}

/// Pass shape, state memory and speculation.
#[derive(Debug, Default, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PipelineArgs {
    /// Largest prefill chunk of one request in one pass (default: 256).
    #[arg(long)]
    prefill_chunk_size: Option<usize>,
    /// Memory for K/V pages and recurrent state, in MiB. Default: room for each concurrent
    /// request's 8K-token context within the device's working set.
    #[arg(long)]
    state_memory_mb: Option<usize>,
    /// Maximum prompt-lookup drafts per decode. Opt-in throughput mode: tokens arrive in
    /// bursts and pauses can grow. Zero disables it; draft length adapts to measured cost.
    /// Requires F32 recurrent state; narrower state automatically uses ordinary decoding. Default: 0.
    #[arg(long)]
    speculative_draft: Option<usize>,
    /// Also draft from the checkpoint's native multi-token prediction head. Implies at
    /// least one draft and disables prefix reuse.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    mtp: Option<bool>,
    #[arg(skip)]
    device_sampling: Option<bool>,
    #[arg(skip)]
    max_waiting_tokens: Option<usize>,
    #[arg(skip)]
    min_prefill_tokens: Option<usize>,
    #[arg(skip)]
    target_pass_ms: Option<u64>,
    #[arg(skip)]
    max_queue_wait_ms: Option<u64>,
    #[arg(skip)]
    prefix_spacing: Option<usize>,
}

impl PipelineArgs {
    inherit!(
        prefill_chunk_size,
        state_memory_mb,
        speculative_draft,
        mtp,
        device_sampling,
        max_waiting_tokens,
        min_prefill_tokens,
        target_pass_ms,
        max_queue_wait_ms,
        prefix_spacing
    );

    pub(crate) fn config(&self, max_batch_size: usize) -> Result<PipelineConfig> {
        let state_bytes = self
            .state_memory_mb
            .map(|mb| {
                mb.checked_mul(1 << 20).ok_or_else(|| {
                    TurError::Other("state memory in MiB exceeds usize bytes".into())
                })
            })
            .transpose()?;
        let mut config =
            PipelineConfig::new(max_batch_size, self.prefill_chunk_size.unwrap_or(256));
        let drafts = self.speculative_draft.unwrap_or(0);
        config.speculation = if self.mtp.unwrap_or(false) {
            SpeculationConfig::with_mtp(drafts.max(1))
        } else {
            SpeculationConfig::new(drafts)
        };
        config.state_bytes = state_bytes;
        config.device_sampling = self.device_sampling.unwrap_or(config.device_sampling);
        config.prefix.spacing = self.prefix_spacing.unwrap_or(config.prefix.spacing);
        let scheduler = &mut config.scheduler;
        scheduler.max_waiting_tokens = self
            .max_waiting_tokens
            .unwrap_or(scheduler.max_waiting_tokens);
        scheduler.min_prefill_tokens = self
            .min_prefill_tokens
            .unwrap_or(scheduler.min_prefill_tokens);
        scheduler.target_pass = self
            .target_pass_ms
            .map_or(scheduler.target_pass, Duration::from_millis);
        scheduler.max_queue_wait = self
            .max_queue_wait_ms
            .map_or(scheduler.max_queue_wait, Duration::from_millis);
        Ok(config)
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
            prefill_chunk_size: Some(256),
            state_memory_mb: Some(max_mb),
            ..PipelineArgs::default()
        };
        assert_eq!(args.config(1)?.state_bytes, Some(max_mb * (1 << 20)));
        args.state_memory_mb = Some(max_mb + 1);
        assert!(args.config(1).is_err());
        Ok(())
    }

    #[test]
    fn test_model_file_formats_paths_and_cli_precedence() -> Result<()> {
        use clap::{Args as _, FromArgMatches as _};

        let defaults: ModelArgs = toml::from_str(
            r#"
            source = "./models/checkpoint"
            revision = "file-revision"
            weight_format = "q4k"
            kv_format = "q8_64"
            kv_page_tokens = 32
            recurrent_state = "f16"
        "#,
        )
        .map_err(|error| TurError::Other(error.to_string()))?;
        let matches = ModelArgs::augment_args(clap::Command::new("model"))
            .mut_arg("model", |arg| arg.env(None::<&str>))
            .mut_arg("revision", |arg| arg.env(None::<&str>))
            .try_get_matches_from(["model", "--revision", "cli-revision", "--cpu-reference"])
            .map_err(|error| TurError::Other(error.to_string()))?;
        let mut args = ModelArgs::from_arg_matches(&matches)
            .map_err(|error| TurError::Other(error.to_string()))?;
        args.inherit(defaults);
        args.relative_to(Path::new("/tmp/tur-settings"));
        let factory = args.factory()?;
        assert!(
            matches!(factory.source, ModelSource::LocalPath(ref path) if path == Path::new("/tmp/tur-settings/models/checkpoint"))
        );
        assert_eq!(factory.revision, "cli-revision");
        assert_eq!(factory.format, Some(WeightFormat::Q4K));
        assert_eq!(factory.kv, KvLayout::new(32, KvFormat::Q8 { group: 64 }));
        assert_eq!(factory.recurrent, DType::F16);
        Ok(())
    }

    #[test]
    fn test_environment_overrides_file_and_cli_overrides_environment() -> Result<()> {
        use clap::{Args as _, FromArgMatches as _};

        // Bind to an existing variable rather than mutating this multithreaded test
        // process's environment (which would require unsafe in edition 2024).
        let expected = std::env::var("PATH").map_err(|error| TurError::Other(error.to_string()))?;
        for (values, expected_model) in [
            (vec!["model"], expected.as_str()),
            (vec!["model", "--model", "cli-model"], "cli-model"),
        ] {
            let matches = ModelArgs::augment_args(clap::Command::new("model"))
                .mut_arg("model", |arg| arg.env("PATH"))
                .try_get_matches_from(values)
                .map_err(|error| TurError::Other(error.to_string()))?;
            let mut args = ModelArgs::from_arg_matches(&matches)
                .map_err(|error| TurError::Other(error.to_string()))?;
            args.inherit(ModelArgs {
                model: Some("file-model".into()),
                ..ModelArgs::default()
            });
            assert_eq!(args.model.as_deref(), Some(expected_model));
        }
        Ok(())
    }

    #[test]
    fn test_missing_model_identity_and_zero_page_size_are_rejected() -> Result<()> {
        let mut args: ModelArgs =
            toml::from_str("").map_err(|error| TurError::Other(error.to_string()))?;
        args.cpu_reference = true;
        assert!(args.factory().is_err());
        args.model = Some("org/checkpoint".into());
        assert!(args.factory().is_err());
        args.revision = Some("revision".into());
        args.kv_page_tokens = Some(0);
        assert!(args.factory().is_err());
        Ok(())
    }
}
