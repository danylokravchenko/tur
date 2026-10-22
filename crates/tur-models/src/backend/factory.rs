#![forbid(unsafe_code)]
//! Revision-pinned checkpoint loading for the hybrid text model.

use std::path::{Component, Path, PathBuf};

use candle_core::Device;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use tokenizers::Tokenizer;
use tur_core::execution::KvLayout;

use crate::{
    Result, TurError,
    backend::chat_template::ChatTemplate,
    models::qwen3_5::{Config, Execution, GenerationDefaults, LoadOptions, Qwen35, WeightFormat},
};

/// A local `SafeTensors` directory or a fully qualified Hugging Face repository.
#[derive(Debug, Clone)]
pub enum ModelSource {
    LocalPath(PathBuf),
    HuggingFace(String),
}

/// Loads a validated hybrid checkpoint, its tokenizer and its own chat template.
pub struct ModelFactory {
    pub source: ModelSource,
    /// Immutable commit for remote sources; a caller-provided identity for local weights.
    pub revision: String,
    pub format: WeightFormat,
    pub device: Device,
    /// Paged K/V page size and storage format.
    pub kv: KvLayout,
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
        if self.revision.trim().is_empty() {
            return Err(hub_error("checkpoint revision must not be empty"));
        }
        let id = match &self.source {
            ModelSource::LocalPath(path) => return Ok(path.clone()),
            ModelSource::HuggingFace(id) => id,
        };
        if self.revision.len() != 40 || !self.revision.bytes().all(|c| c.is_ascii_hexdigit()) {
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
            self.revision.clone(),
        ));
        let config_path = repo.get("config.json").map_err(hub_error)?;
        Config::from_json(&std::fs::read_to_string(&config_path)?)?;
        let info = repo.info().map_err(hub_error)?;
        if info.sha != self.revision {
            return Err(hub_error(
                "repository returned a different checkpoint revision",
            ));
        }
        for required in ["tokenizer.json", "tokenizer_config.json"] {
            repo.get(required).map_err(hub_error)?;
        }
        for file in &info.siblings {
            let name = file.rfilename.as_str();
            if name.ends_with(".safetensors")
                || matches!(
                    name,
                    "model.safetensors.index.json"
                        | "generation_config.json"
                        | "chat_template.jinja"
                )
            {
                checkpoint_file(name)?;
                repo.get(name).map_err(hub_error)?;
            }
        }
        config_path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| hub_error("checkpoint has no directory"))
    }

    /// Load the text model and serving metadata. Missing or invalid templates are errors.
    /// `Q8_0/Q4K` explicitly convert `SafeTensors` matrices into packed resident weights;
    /// legacy ordinary-attention GGUF checkpoints are unsupported.
    pub fn create_model(&self) -> Result<(Qwen35, Tokenizer, ChatTemplate, GenerationDefaults)> {
        let dir = self.resolve()?;
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
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
        let model = Qwen35::load(
            &dir,
            &LoadOptions {
                revision: match &self.source {
                    ModelSource::HuggingFace(id) => format!("{id}@{}", self.revision),
                    ModelSource::LocalPath(path) => {
                        format!("{}@{}", path.canonicalize()?.display(), self.revision)
                    }
                },
                format: self.format,
                device: self.device.clone(),
                execution: Execution::Fused,
                kv: self.kv,
            },
        )?;
        Ok((model, tokenizer, template, defaults))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_paths_cannot_escape_snapshot() {
        for name in [
            "../weights.safetensors",
            "/tmp/weights.safetensors",
            "sub/weights.safetensors",
            "..",
        ] {
            assert!(checkpoint_file(name).is_err());
        }
        assert!(checkpoint_file("model-00001-of-00002.safetensors").is_ok());
    }

    #[test]
    fn test_mutable_remote_revision_rejected_without_network() {
        let factory = ModelFactory {
            source: ModelSource::HuggingFace("Qwen/Qwen3.5-0.8B".into()),
            revision: "main".into(),
            format: WeightFormat::F32,
            device: Device::Cpu,
            kv: crate::models::qwen3_5::DEFAULT_KV_LAYOUT,
        };
        assert!(factory.resolve().is_err());
    }
}
