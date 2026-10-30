#![forbid(unsafe_code)]
//! Revision-pinned checkpoint loading for the hybrid text model.

use crate::models::GenerationDefaults;
use crate::models::{DraftHead, Execution, LoadOptions, WeightFormat, Weights};
use std::{
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use candle_core::Device;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use tokenizers::Tokenizer;
use tur_core::execution::KvLayout;

use crate::{
    Result, TurError,
    backend::chat_template::ChatTemplate,
    models::{Manifest, ModelConfig, TextModel, is_artifact},
};

/// A local `SafeTensors` directory or a fully qualified Hugging Face repository.
#[derive(Debug, Clone)]
pub enum ModelSource {
    LocalPath(PathBuf),
    HuggingFace(String),
}

/// Loads a validated hybrid checkpoint, its tokenizer and its own chat template.
#[derive(Clone)]
pub struct ModelFactory {
    pub source: ModelSource,
    /// Immutable commit for remote sources; a caller-provided identity for local weights.
    pub revision: String,
    /// Convert `SafeTensors` projections into this format. `None` loads the checkpoint's
    /// prepacked artifact when it has one and converts to `Q8_0` otherwise.
    pub format: Option<WeightFormat>,
    pub device: Device,
    /// Paged K/V page size and storage format.
    pub kv: KvLayout,
    /// Recurrent matrix storage: F32, F16 or BF16.
    pub recurrent: candle_core::DType,
    /// Load the checkpoint's audio encoder, when it has one ([`LoadOptions::audio`]).
    pub audio: bool,
    /// Device bytes of a bounded expert cache ([`LoadOptions::expert_cache`]).
    pub expert_cache: Option<usize>,
}

fn hub_error(error: impl std::fmt::Display) -> TurError {
    TurError::HfHub(error.to_string())
}

fn checkpoint_file(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(hub_error(format!("invalid checkpoint filename '{name}'")));
    }
    Ok(())
}

impl ModelFactory {
    /// Resolve the source, rejecting unsupported architectures before downloading weights.
    pub fn resolve(&self) -> Result<PathBuf> {
        resolve(
            &self.source,
            &self.revision,
            |json| ModelConfig::from_json(json).map(drop),
            &["tokenizer.json", "tokenizer_config.json"],
        )
    }

    /// Load the text model and serving metadata. Missing or invalid templates are errors.
    /// `Q8_0/Q4K` explicitly convert `SafeTensors` matrices into packed resident weights;
    /// a prepacked artifact is uploaded as stored. Legacy ordinary-attention GGUF
    /// checkpoints are unsupported.
    pub fn create_model(&self) -> Result<(TextModel, Tokenizer, ChatTemplate, GenerationDefaults)> {
        let (model, tokenizer, template, defaults, ()) =
            self.create_model_with(None, |_, _| Ok(()))?;
        Ok((model, tokenizer, template, defaults))
    }

    /// Validate serving metadata, then reserve resources before loading any weight
    /// payload or uploading tensors. The reservation is returned with the model and is
    /// dropped automatically on a loading failure. `draft` also loads the MTP head.
    pub fn create_model_with<R>(
        &self,
        draft: Option<DraftHead>,
        reserve: impl FnOnce(&ModelConfig, &LoadOptions) -> Result<R>,
    ) -> Result<(TextModel, Tokenizer, ChatTemplate, GenerationDefaults, R)> {
        let dir = self.resolve()?;
        let config = ModelConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let generation = match std::fs::read_to_string(dir.join("generation_config.json")) {
            Ok(json) => Some(json),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let mut defaults = config.generation_defaults(generation.as_deref())?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&id| id as usize >= config.vocab_size())
        {
            return Err(TurError::Tokenizer(
                "tokenizer IDs exceed the model vocabulary".into(),
            ));
        }
        let tokenizer_config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
        if let Some(eos) = tokenizer_config
            .get("eos_token")
            .filter(|value| !value.is_null())
        {
            let text = eos
                .as_str()
                .or_else(|| eos.get("content").and_then(serde_json::Value::as_str))
                .ok_or_else(|| TurError::Tokenizer("invalid tokenizer eos_token".into()))?;
            let id = tokenizer.token_to_id(text).ok_or_else(|| {
                TurError::Tokenizer("tokenizer eos_token has no vocabulary ID".into())
            })?;
            if !defaults.eos_tokens.contains(&id) {
                defaults.eos_tokens.push(id);
            }
        }
        let template = ChatTemplate::from_model_dir(&dir)?;
        let weights = match (self.format, is_artifact(&dir)) {
            (None, true) => Weights::Prepacked(Arc::new(Manifest::read(&dir)?)),
            (None, false) => WeightFormat::Q8_0.into(),
            (Some(_), true) => {
                return Err(TurError::Other(
                    "checkpoint is a prepacked artifact; omit the weight format".into(),
                ));
            }
            (Some(format), false) => format.into(),
        };
        let options = LoadOptions {
            revision: match &self.source {
                ModelSource::HuggingFace(id) => format!("{id}@{}", self.revision),
                ModelSource::LocalPath(path) => {
                    format!("{}@{}", path.canonicalize()?.display(), self.revision)
                }
            },
            weights,
            device: self.device.clone(),
            execution: Execution::Fused,
            kv: self.kv,
            recurrent: self.recurrent,
            mtp: draft,
            audio: self.audio,
            expert_cache: self.expert_cache,
        };
        let config = if options.mtp.is_some() {
            config.with_mtp()?
        } else {
            config
        };
        let reservation = reserve(&config, &options)?;
        let model = config.load(&dir, &options)?;
        Ok((model, tokenizer, template, defaults, reservation))
    }
}

/// Resolve a drafter checkpoint for `--draft-model`: a local directory, or a repository
/// at an immutable commit whose config names a Gemma 4 assistant. The target validates
/// the drafter against itself when it loads it.
pub fn resolve_drafter(source: &ModelSource, revision: &str) -> Result<PathBuf> {
    resolve(
        source,
        revision,
        |json| {
            let raw: serde_json::Value = serde_json::from_str(json)?;
            if raw["model_type"].as_str() == Some("gemma4_assistant") {
                Ok(())
            } else {
                Err(hub_error("draft model is not a Gemma 4 assistant"))
            }
        },
        &[],
    )
}

/// Loads a revision-pinned Whisper checkpoint for transcription.
#[derive(Clone)]
pub struct WhisperFactory {
    pub source: ModelSource,
    /// Immutable commit for remote sources; a caller-provided identity for local weights.
    pub revision: String,
    /// Storage of projection matrices (`Q8_0` matched Transformers' F32 transcripts).
    pub format: WeightFormat,
    pub device: Device,
}

impl WhisperFactory {
    /// Resolve the source, rejecting other architectures before downloading weights.
    pub fn resolve(&self) -> Result<PathBuf> {
        resolve(
            &self.source,
            &self.revision,
            |json| crate::models::whisper::Config::from_json(json).map(drop),
            &["tokenizer.json", "generation_config.json"],
        )
    }

    /// Device bytes the loaded model will hold, and its peak encoder activations, for
    /// admission against a device budget before any weight is read.
    pub fn memory_requirements(&self) -> Result<(usize, usize)> {
        let dir = self.resolve()?;
        let config = crate::models::whisper::Config::from_json(&std::fs::read_to_string(
            dir.join("config.json"),
        )?)?;
        Ok((
            crate::models::whisper::Whisper::planned_bytes(&config, self.format)?,
            crate::models::whisper::Whisper::scratch_bytes(&config)?,
        ))
    }

    pub fn create(&self) -> Result<crate::models::whisper::Transcriber> {
        crate::models::whisper::Transcriber::load(&self.resolve()?, self.format, &self.device)
    }
}

/// Snapshot directory of `source`: validate `config.json` with `validate` before any
/// weight file downloads, and require the repository to report exactly `revision`.
fn resolve(
    source: &ModelSource,
    revision: &str,
    validate: impl Fn(&str) -> Result<()>,
    required: &[&str],
) -> Result<PathBuf> {
    if revision.trim().is_empty() {
        return Err(hub_error("checkpoint revision must not be empty"));
    }
    let id = match source {
        ModelSource::LocalPath(path) => return Ok(path.clone()),
        ModelSource::HuggingFace(id) => id,
    };
    if revision.len() != 40 || !revision.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(hub_error(
            "remote --revision must be an immutable 40-character commit",
        ));
    }
    if id.split('/').count() != 2 || id.split('/').any(str::is_empty) {
        return Err(hub_error(
            "use a fully qualified Hugging Face repository (organization/model)",
        ));
    }
    let api = ApiBuilder::from_env().build().map_err(hub_error)?;
    let repo = api.repo(Repo::with_revision(
        id.clone(),
        RepoType::Model,
        revision.to_owned(),
    ));
    let config_path = repo.get("config.json").map_err(hub_error)?;
    validate(&std::fs::read_to_string(&config_path)?)?;
    let info = repo.info().map_err(hub_error)?;
    if info.sha != revision {
        return Err(hub_error(
            "repository returned a different checkpoint revision",
        ));
    }
    for name in required {
        repo.get(name).map_err(hub_error)?;
    }
    for file in &info.siblings {
        let name = file.rfilename.as_str();
        if serving_file(name) {
            checkpoint_file(name)?;
            repo.get(name).map_err(hub_error)?;
        }
    }
    config_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| hub_error("checkpoint has no directory"))
}

/// Files a load needs from a repository. Repositories may keep other copies of the weights
/// in subdirectories (gpt-oss has `original/` and `metal/`), which are never downloaded.
fn serving_file(name: &str) -> bool {
    !name.contains('/')
        && (name.ends_with(".safetensors")
            || matches!(
                name,
                "model.safetensors.index.json"
                    | "generation_config.json"
                    | "chat_template.jinja"
                    | "processor_config.json"
                    | "preprocessor_config.json"
            ))
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert checkpoint validation outcomes"
)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_paths_cannot_escape_snapshot() -> Result<()> {
        for name in [
            "../weights.safetensors",
            "/tmp/weights.safetensors",
            "sub/weights.safetensors",
            "..",
        ] {
            assert!(checkpoint_file(name).is_err());
        }
        checkpoint_file("model-00001-of-00002.safetensors")?;
        Ok(())
    }

    #[test]
    fn test_only_root_serving_files_are_downloaded() {
        assert!(serving_file("model-00000-of-00002.safetensors"));
        assert!(serving_file("chat_template.jinja"));
        assert!(!serving_file("original/model.safetensors"));
        assert!(!serving_file("metal/model.bin"));
        assert!(!serving_file("README.md"));
    }

    #[test]
    fn test_mutable_remote_revision_rejected_without_network() -> Result<()> {
        let factory = ModelFactory {
            source: ModelSource::HuggingFace("Qwen/Qwen3.5-0.8B".into()),
            revision: "main".into(),
            format: Some(WeightFormat::F32),
            device: Device::Cpu,
            kv: crate::models::DEFAULT_KV_LAYOUT,
            recurrent: crate::models::DEFAULT_RECURRENT_STATE,
            audio: false,
            expert_cache: None,
        };
        let error = factory
            .resolve()
            .err()
            .ok_or_else(|| hub_error("mutable revision was accepted"))?;
        assert!(matches!(
            error,
            TurError::HfHub(message) if message.contains("immutable 40-character commit")
        ));
        Ok(())
    }
}
