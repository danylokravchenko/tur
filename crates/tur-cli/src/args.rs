//! Flags shared by `run` and `serve`.
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use tur::models::{DEFAULT_KV_LAYOUT, DraftHead, WeightFormat, parse_recurrent_state};

use candle_core::DType;
use clap::builder::{PossibleValuesParser, TypedValueParser as _};
use serde::Deserialize;
use tur::{
    ModelFactory, ModelSource, Result, TurError,
    backend::{
        factory::{
            GgufSource, WhisperFactory, latest_revision, resolve_checkpoint, resolve_drafter,
        },
        pipeline::PipelineConfig,
        speculation::SpeculationConfig,
    },
    core::execution::{KvFormat, KvLayout},
};

use crate::{
    catalog,
    config::{inherit, optional_from_str, optional_recurrent},
};

/// Explicit paths and existing directories load locally; other values name a Hugging Face
/// repository.
pub(crate) fn model_source(model: &str) -> ModelSource {
    let path = Path::new(model);
    if path.is_dir() || path.is_absolute() || path.starts_with(".") || path.starts_with("..") {
        ModelSource::LocalPath(model.into())
    } else {
        ModelSource::HuggingFace(model.to_owned())
    }
}

/// The local directory of a job's `--model`, downloading a repository's snapshot if needed.
pub(crate) fn checkpoint_dir(model: &Path, revision: &str) -> Result<PathBuf> {
    let source = model_source(&model.to_string_lossy());
    if matches!(source, ModelSource::LocalPath(_)) && !model.is_dir() {
        return Err(TurError::Other(format!(
            "no checkpoint directory at {}",
            model.display()
        )));
    }
    resolve_checkpoint(&source, revision)
}

/// Where `model` loads from and the revision naming it. An omitted revision is a local
/// directory's own name, or a repository's latest commit, reported so a run can be repeated.
pub(crate) fn identity(model: &str, revision: Option<&str>) -> Result<(ModelSource, String)> {
    let source = model_source(model);
    if matches!(&source, ModelSource::HuggingFace(id) if !id.contains('/')) {
        return Err(TurError::Other(format!(
            "unknown model '{model}'; `tur list` shows the names, and a checkpoint is a \
             directory or an organization/name repository"
        )));
    }
    let revision = match (&source, revision.filter(|value| !value.trim().is_empty())) {
        (_, Some(revision)) => revision.to_owned(),
        (ModelSource::LocalPath(path), None) => directory_name(path),
        (ModelSource::HuggingFace(id), None) => latest_commit(id, "--revision")?,
    };
    Ok((source, revision))
}

fn directory_name(path: &Path) -> String {
    let absolute = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    absolute.file_name().map_or_else(
        || absolute.to_string_lossy().into_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

fn latest_commit(repository: &str, flag: &str) -> Result<String> {
    let commit = latest_revision(repository).map_err(|error| {
        TurError::Other(format!(
            "cannot look up the latest commit of {repository}: {}; give one with {flag}",
            crate::ui::describe(&error)
        ))
    })?;
    tracing::warn!("Using {repository} at its latest commit; {flag} {commit} keeps these weights");
    Ok(commit)
}

/// Checkpoint and state storage.
#[derive(Debug, Default, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
#[command(next_help_heading = "Model")]
pub(crate) struct ModelArgs {
    /// A name from `tur list`, a checkpoint directory, or an organization/name Hugging Face
    /// repository. Default: `[model].source` in tur.toml.
    #[arg(value_name = "MODEL")]
    #[serde(rename = "source")]
    model: Option<String>,
    /// What MODEL was called, for clients of a server.
    #[arg(skip)]
    #[serde(skip)]
    named: Option<String>,
    /// Commit of a repository, or a name for a directory's weights. Default: the commit a
    /// name from `tur list` is pinned to, a repository's latest, or the directory's name.
    #[arg(long)]
    revision: Option<String>,
    /// Convert `SafeTensors` projections into this resident format: f32, `q8_0`, q4k.
    /// Omitted, a prepacked artifact loads as stored and any other checkpoint converts
    /// to `q8_0`.
    #[arg(long)]
    #[serde(deserialize_with = "optional_from_str")]
    weight_format: Option<WeightFormat>,
    /// Load weights from this GGUF file as stored: a local path, or
    /// `organization/repository/file.gguf` at `--gguf-revision`. MODEL then supplies only the
    /// config, tokenizer and template, and its weights are not downloaded.
    #[arg(long, env = "TUR_GGUF")]
    gguf: Option<String>,
    /// Commit of a remote `--gguf` file. Default: the repository's latest.
    #[arg(long, env = "TUR_GGUF_REVISION")]
    gguf_revision: Option<String>,
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
    /// Load the checkpoint's audio encoder when it has one, so chat requests may carry
    /// audio (Gemma 4 E2B and E4B). `--audio-encoder=false` saves its memory (default: true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    audio_encoder: Option<bool>,
    /// Load the checkpoint's vision encoder when it has one, so chat requests may carry
    /// images (Qwen3.5 and Gemma 4). `--vision-encoder=false` saves its memory (default: true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    vision_encoder: Option<bool>,
    /// Keep routed experts on disk and read them on demand into this many MiB of device
    /// memory, for gpt-oss checkpoints or Qwen `MoE` artifacts larger than the GPU's working
    /// set. Reads only: the cache never writes to disk (default: every expert resident).
    #[arg(long)]
    expert_cache_mb: Option<usize>,
    /// Use the CPU numerical reference for development checks.
    #[arg(long)]
    #[serde(skip)]
    cpu_reference: bool,
}

impl ModelArgs {
    /// Whether experts stay on disk behind a bounded cache.
    pub(crate) fn caches_experts(&self) -> bool {
        self.expert_cache_mb.is_some()
    }

    inherit!(
        model,
        revision,
        weight_format,
        gguf,
        gguf_revision,
        kv_format,
        kv_page_tokens,
        recurrent_state,
        audio_encoder,
        vision_encoder,
        expert_cache_mb
    );

    /// Settle which model runs. A model on the command line replaces the file's along with its
    /// revision, which names a commit of that other model. A catalog name then brings its
    /// commit and limits beneath anything set explicitly.
    pub(crate) fn select(&mut self, mut file: Self) -> Option<&'static catalog::Model> {
        if self.model.is_some() {
            file.model = None;
            file.revision = None;
        }
        self.inherit(file);
        self.named.clone_from(&self.model);
        let entry = catalog::find(self.model.as_deref()?)?;
        self.model = Some(entry.repository.into());
        self.revision
            .get_or_insert_with(|| entry.revision.to_owned());
        self.expert_cache_mb = self.expert_cache_mb.or(entry.expert_cache_mb);
        Some(entry)
    }

    /// The model ID a server offers unless told otherwise: the last part of what the user wrote.
    pub(crate) fn served_name(&self) -> Option<String> {
        let name = Path::new(self.named.as_deref()?).file_name()?;
        Some(name.to_string_lossy().into_owned())
    }

    /// A `--gguf` value is local when the path exists, and a Hub file otherwise.
    fn gguf_source(&self) -> Result<Option<GgufSource>> {
        let Some(gguf) = self.gguf.as_ref().filter(|value| !value.trim().is_empty()) else {
            return Ok(None);
        };
        if Path::new(gguf).is_file() {
            return Ok(Some(GgufSource::LocalPath(gguf.into())));
        }
        let parts: Vec<&str> = gguf.splitn(3, '/').collect();
        let &[organization, repository, file] = parts.as_slice() else {
            return Err(TurError::Other(format!(
                "no GGUF file at {gguf}; a Hub file is organization/repository/file.gguf"
            )));
        };
        let repository = format!("{organization}/{repository}");
        let revision = match self
            .gguf_revision
            .clone()
            .filter(|value| !value.trim().is_empty())
        {
            Some(revision) => revision,
            None => latest_commit(&repository, "--gguf-revision")?,
        };
        Ok(Some(GgufSource::HuggingFace {
            repository,
            file: file.to_owned(),
            revision,
        }))
    }

    pub(crate) fn factory(&self) -> Result<ModelFactory> {
        let model = self
            .model
            .as_ref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                TurError::Other(
                    "which model? For example `tur run qwen`; `tur list` shows the names".into(),
                )
            })?;
        let (source, revision) = identity(model, self.revision.as_deref())?;
        if self.kv_layout().page_tokens == 0 {
            return Err(TurError::Other("K/V page tokens must be positive".into()));
        }
        Ok(ModelFactory {
            source,
            revision,
            format: self.weight_format,
            gguf: self.gguf_source()?,
            device: tur::shared::inference_device(self.cpu_reference)?,
            kv: self.kv_layout(),
            recurrent: self.recurrent_state.unwrap_or(DType::F32),
            audio: self.audio_encoder.unwrap_or(true),
            vision: self.vision_encoder.unwrap_or(true),
            expert_cache: self
                .expert_cache_mb
                .map(|mb| {
                    mb.checked_mul(1 << 20)
                        .ok_or_else(|| TurError::Other("expert cache size overflow".into()))
                })
                .transpose()?,
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

/// Speech-to-text checkpoint, served apart from the chat model.
#[derive(Debug, Default, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
#[command(next_help_heading = "Speech")]
pub(crate) struct WhisperArgs {
    /// Whisper checkpoint for /v1/audio/transcriptions and `tur transcribe`: a local
    /// directory or a Hugging Face repository at `--whisper-revision`.
    #[arg(long)]
    #[serde(rename = "source")]
    whisper: Option<String>,
    /// Immutable commit of a remote `--whisper` checkpoint, or an identity for a local one.
    #[arg(long)]
    #[serde(rename = "revision")]
    whisper_revision: Option<String>,
    /// Whisper projection storage: f32, `q8_0` or q4k (default: `q8_0`).
    #[arg(long)]
    #[serde(rename = "weight_format", deserialize_with = "optional_from_str")]
    whisper_weight_format: Option<WeightFormat>,
}

impl WhisperArgs {
    inherit!(whisper, whisper_revision, whisper_weight_format);

    pub(crate) fn relative_to(&mut self, directory: &Path) {
        if let Some(model) = &mut self.whisper {
            let path = Path::new(model);
            if path.is_relative() && (path.starts_with(".") || directory.join(path).is_dir()) {
                *model = directory.join(path).to_string_lossy().into_owned();
            }
        }
    }

    /// The configured checkpoint on `device`, if any; local directories load as is.
    pub(crate) fn factory(&self, device: &candle_core::Device) -> Result<Option<WhisperFactory>> {
        let Some(model) = self.whisper.as_ref().filter(|m| !m.trim().is_empty()) else {
            return Ok(None);
        };
        let source = model_source(model);
        let revision = self
            .whisper_revision
            .clone()
            .filter(|r| !r.trim().is_empty())
            .ok_or_else(|| {
                TurError::Other("--whisper needs --whisper-revision (or [whisper].revision)".into())
            })?;
        Ok(Some(WhisperFactory {
            source,
            revision,
            format: self.whisper_weight_format.unwrap_or(WeightFormat::Q8_0),
            device: device.clone(),
        }))
    }
}

/// Pass shape, state memory and speculation.
#[derive(Debug, Default, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
#[command(next_help_heading = "Engine")]
pub(crate) struct PipelineArgs {
    /// Largest prefill chunk of one request in one pass (default: 256).
    #[arg(long)]
    prefill_chunk_size: Option<usize>,
    /// Memory for K/V pages and recurrent state, in MiB. Default: room for each concurrent
    /// request's 8K-token context within the device's working set.
    #[arg(long)]
    state_memory_mb: Option<usize>,
    /// Most tokens one request may hold: its prompt and output. Sizes the state pool for that
    /// many tokens per concurrent request and bounds kernel scratch; a start that cannot fit
    /// it fails and names what would. Default: whatever the pool and the model allow.
    #[arg(long)]
    context_tokens: Option<usize>,
    /// Maximum prompt-lookup drafts per decode. Opt-in throughput mode: tokens arrive in
    /// bursts and pauses can grow. Zero disables it; draft length adapts to measured cost.
    /// Requires F32 recurrent state; narrower state automatically uses ordinary decoding. Default: 0.
    #[arg(long)]
    speculative_draft: Option<usize>,
    /// Also draft from the checkpoint's native multi-token prediction head. Implies at
    /// least one draft and disables prefix reuse.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    mtp: Option<bool>,
    /// Draft from a separate multi-token prediction checkpoint (a Gemma 4 assistant): a
    /// local directory or a Hugging Face repository at `--draft-revision`. Implies at
    /// least one draft and disables prefix reuse.
    #[arg(long)]
    draft_model: Option<String>,
    /// Immutable commit of a remote `--draft-model`, or an identity for a local one.
    #[arg(long)]
    draft_revision: Option<String>,
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
    shortest_prompt_first: Option<bool>,
    #[arg(skip)]
    layered_prefill: Option<bool>,
    #[arg(skip)]
    prefix_spacing: Option<usize>,
    #[arg(skip)]
    prefix_restorable_first: Option<bool>,
}

impl PipelineArgs {
    /// Prefill defaults when experts stay on disk: every pass can read the whole expert set
    /// whatever its size, so fewer, larger passes read far less. A 2,037-token prompt on
    /// gpt-oss-20b took 32 s in 256-token chunks and 18 s in 1024-token chunks. Snapshots
    /// sit on chunk boundaries, so their spacing has to grow with the chunk.
    pub(crate) fn prefer_large_passes(&mut self) {
        self.prefill_chunk_size.get_or_insert(1024);
        self.prefix_spacing.get_or_insert(1024);
    }

    inherit!(
        prefill_chunk_size,
        state_memory_mb,
        context_tokens,
        speculative_draft,
        mtp,
        draft_model,
        draft_revision,
        device_sampling,
        max_waiting_tokens,
        min_prefill_tokens,
        target_pass_ms,
        max_queue_wait_ms,
        shortest_prompt_first,
        layered_prefill,
        prefix_spacing,
        prefix_restorable_first
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
        config.context_tokens = self.context_tokens;
        let drafts = self.speculative_draft.unwrap_or(0);
        config.speculation = if let Some(model) = &self.draft_model {
            let revision = self
                .draft_revision
                .as_deref()
                .ok_or_else(|| TurError::Other("--draft-model needs --draft-revision".into()))?;
            let source = model_source(model);
            SpeculationConfig {
                mtp: Some(DraftHead {
                    vocab: None,
                    model: Some(resolve_drafter(&source, revision)?),
                }),
                ..SpeculationConfig::new(drafts.max(1))
            }
        } else if self.mtp.unwrap_or(false) {
            SpeculationConfig::with_mtp(drafts.max(1))
        } else {
            SpeculationConfig::new(drafts)
        };
        config.state_bytes = state_bytes;
        config.device_sampling = self.device_sampling.unwrap_or(config.device_sampling);
        config.prefix.spacing = self.prefix_spacing.unwrap_or(config.prefix.spacing);
        config.prefix.restorable_first = self
            .prefix_restorable_first
            .unwrap_or(config.prefix.restorable_first);
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
        scheduler.shortest_prompt_first = self
            .shortest_prompt_first
            .unwrap_or(scheduler.shortest_prompt_first);
        scheduler.layered_prefill = self.layered_prefill.unwrap_or(scheduler.layered_prefill);
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
    fn test_paths_load_locally_and_names_come_from_the_hub() {
        for local in ["/absent/model", "./model", "../model", "."] {
            assert!(
                matches!(model_source(local), ModelSource::LocalPath(_)),
                "{local}"
            );
        }
        assert!(matches!(
            model_source("org/name"),
            ModelSource::HuggingFace(_)
        ));
    }

    #[test]
    fn test_a_missing_job_checkpoint_is_named() -> Result<()> {
        let error = checkpoint_dir(Path::new("/absent/model"), "x")
            .err()
            .ok_or_else(|| TurError::Other("missing directory was accepted".into()))?;
        assert!(error.to_string().contains("/absent/model"));
        Ok(())
    }

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

    /// Flags as a command line would give them.
    fn model_flags(values: &[&str]) -> Result<ModelArgs> {
        use clap::{Args as _, FromArgMatches as _};

        let matches = ModelArgs::augment_args(clap::Command::new("model"))
            .try_get_matches_from(values)
            .map_err(|error| TurError::Other(error.to_string()))?;
        ModelArgs::from_arg_matches(&matches).map_err(|error| TurError::Other(error.to_string()))
    }

    #[test]
    fn test_model_file_formats_paths_and_cli_precedence() -> Result<()> {
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
        let mut args = model_flags(&["model", "--revision", "cli-revision", "--cpu-reference"])?;
        assert!(args.select(defaults).is_none());
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
    fn test_named_model_replaces_the_files_model_and_revision() -> Result<()> {
        let file = || ModelArgs {
            model: Some("org/other".into()),
            revision: Some("commit-of-other".into()),
            kv_page_tokens: Some(32),
            ..ModelArgs::default()
        };
        let mut args = model_flags(&["model", "gpt-oss", "--cpu-reference"])?;
        assert_eq!(
            args.select(file()).map(|entry| entry.name),
            Some("gpt-oss:20b")
        );
        let factory = args.factory()?;
        assert!(
            matches!(&factory.source, ModelSource::HuggingFace(id) if id == "openai/gpt-oss-20b")
        );
        assert_eq!(factory.revision, "6cee5e81ee83917806bbde320786a8fb61efebee");
        assert_eq!(factory.expert_cache, Some(2048 << 20));
        assert_eq!(
            factory.kv.page_tokens, 32,
            "other file settings still apply"
        );
        assert_eq!(args.served_name().as_deref(), Some("gpt-oss"));

        let mut args = model_flags(&[
            "model",
            "gpt-oss:20b",
            "--revision",
            "cli-revision",
            "--expert-cache-mb",
            "4096",
            "--cpu-reference",
        ])?;
        args.select(ModelArgs::default());
        let factory = args.factory()?;
        assert_eq!(factory.revision, "cli-revision");
        assert_eq!(factory.expert_cache, Some(4096 << 20));
        assert_eq!(args.served_name().as_deref(), Some("gpt-oss:20b"));

        let mut args = model_flags(&["model", "--revision", "cli-revision"])?;
        args.select(file());
        assert_eq!(args.revision.as_deref(), Some("cli-revision"));
        assert_eq!(args.model.as_deref(), Some("org/other"));
        Ok(())
    }

    #[test]
    fn test_served_name_is_the_last_part_of_what_the_user_wrote() -> Result<()> {
        for (written, served) in [
            ("org/name", "name"),
            ("./models/mine/", "mine"),
            ("qwen", "qwen"),
        ] {
            let mut args = model_flags(&["model", written])?;
            args.select(ModelArgs::default());
            assert_eq!(args.served_name().as_deref(), Some(served), "{written}");
        }
        Ok(())
    }

    #[test]
    fn test_directories_name_themselves_and_unknown_names_are_refused() -> Result<()> {
        let (source, revision) = identity("/absent/mine", None)?;
        assert!(matches!(source, ModelSource::LocalPath(_)));
        assert_eq!(revision, "mine");
        assert_eq!(identity("/absent/mine", Some("v1"))?.1, "v1");
        let error = identity("llama", Some("v1"))
            .err()
            .ok_or_else(|| TurError::Other("an unknown name was accepted".into()))?;
        assert!(
            error.to_string().contains("unknown model 'llama'"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn test_missing_model_and_zero_page_size_are_rejected() -> Result<()> {
        let mut args: ModelArgs =
            toml::from_str("").map_err(|error| TurError::Other(error.to_string()))?;
        args.cpu_reference = true;
        assert!(args.factory().is_err());
        args.model = Some("/absent/mine".into());
        args.factory()?;
        args.kv_page_tokens = Some(0);
        assert!(args.factory().is_err());
        Ok(())
    }
}
