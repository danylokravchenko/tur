use crate::{Result, TurError};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::trace;

// File name constants
mod file_names {
    pub const TOKENIZER: &str = "tokenizer.json";
    pub const TOKENIZER_CONFIG: &str = "tokenizer_config.json";
    pub const CONFIG: &str = "config.json";
    pub const GENERATION_CONFIG: &str = "generation_config.json";
    pub const CHAT_TEMPLATE: &str = "chat_template.json";
    pub const MODEL_SAFETENSORS: &str = "model.safetensors";
    pub const MODEL_SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
}

// File extension constants
mod extensions {
    pub const SAFETENSORS: &str = ".safetensors";
    pub const GGUF: &str = ".gguf";
    pub const INDEX_JSON: &str = ".index.json";
}

// Repository constants
mod repos {
    pub const DEFAULT_ORG: &str = "Qwen";
    pub const GGUF_ORG: &str = "unsloth";
    pub const GGUF_SUFFIX: &str = "-GGUF";
    pub const DEFAULT_REVISION: &str = "main";
}

// Retry configuration constants
mod retry {
    pub const MAX_RETRIES: u32 = 5;
    pub const BASE_DELAY_SECS: u64 = 5;
}

#[derive(Debug, Clone)]
pub struct Downloader {
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub quantization: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelPaths {
    pub tokenizer_filename: PathBuf,
    pub tokenizer_config_filename: PathBuf,
    pub config_filename: PathBuf,
    pub generation_config_filename: PathBuf,
    pub filenames: Vec<PathBuf>,
    pub auxiliary_filenames: Vec<PathBuf>,
    pub chat_template_filename: Option<PathBuf>,
}

impl ModelPaths {
    pub fn get_config_filename(&self) -> PathBuf {
        self.config_filename.clone()
    }

    pub fn get_tokenizer_filename(&self) -> PathBuf {
        self.tokenizer_filename.clone()
    }

    pub fn get_tokenizer_config_filename(&self) -> PathBuf {
        self.tokenizer_config_filename.clone()
    }

    pub fn get_weight_filenames(&self) -> Vec<PathBuf> {
        self.filenames.clone()
    }

    pub fn get_auxiliary_filenames(&self) -> Vec<PathBuf> {
        self.auxiliary_filenames.clone()
    }

    pub fn get_generation_config_filename(&self) -> PathBuf {
        self.generation_config_filename.clone()
    }

    pub fn get_chat_template_filename(&self) -> Option<PathBuf> {
        self.chat_template_filename.clone()
    }
}

impl Downloader {
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        quantization: Option<String>,
    ) -> Self {
        Self {
            model_id,
            weight_path,
            quantization,
        }
    }

    pub fn prepare_model_weights(
        &self,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<(ModelPaths, bool)> {
        let (paths, gguf): (ModelPaths, bool) = match (
            &self.model_id,
            &self.weight_path,
            &self.quantization,
        ) {
            (None, Some(path), None) => {
                if !Path::new(path).is_dir() {
                    return Err(TurError::HfHub(
                        "Safetensor weight path must be a directory! \n\t***Tips: use `--f` to specify gguf model file!***".to_string()
                    ));
                }

                let mut filenames = vec![];
                let index_path = Path::new(path).join(file_names::MODEL_SAFETENSORS_INDEX);
                if index_path.exists() {
                    filenames = load_local_safetensors(path, file_names::MODEL_SAFETENSORS_INDEX)?;
                } else {
                    filenames.push(Path::new(path).join(file_names::MODEL_SAFETENSORS));
                }

                (
                    ModelPaths {
                        tokenizer_filename: Path::new(path).join(file_names::TOKENIZER),
                        tokenizer_config_filename: Path::new(path)
                            .join(file_names::TOKENIZER_CONFIG),
                        config_filename: Path::new(path).join(file_names::CONFIG),
                        generation_config_filename: if Path::new(path)
                            .join(file_names::GENERATION_CONFIG)
                            .exists()
                        {
                            Path::new(path).join(file_names::GENERATION_CONFIG)
                        } else {
                            PathBuf::new()
                        },
                        filenames,
                        auxiliary_filenames: Vec::new(),
                        chat_template_filename: if Path::new(path)
                            .join(file_names::CHAT_TEMPLATE)
                            .exists()
                        {
                            Some(Path::new(path).join(file_names::CHAT_TEMPLATE))
                        } else {
                            None
                        },
                    },
                    false,
                )
            }
            (Some(_), None, Some(_)) => (self.download_gguf_model(hf_token, hf_token_path)?, true),
            (Some(_), None, None) => (
                self.download_safetensors_model(hf_token, hf_token_path)?,
                false,
            ),
            _ => {
                return Err(TurError::HfHub(
                    "Invalid configuration!\n***Tips***: \n \t For local model weights: --weight-path <path/to/folder>\n \t For remote SafeTensors: --model-id Qwen3-0.6B\n \t For remote GGUF: --model-id Qwen3-0.6B --quantization Q4_K_M".to_string(),
                ));
            }
        };

        Ok((paths, gguf))
    }

    fn hf_get_with_retry(
        &self,
        api: &hf_hub::api::sync::ApiRepo,
        rfilename: &str,
        retries: u32,
        base_delay: std::time::Duration,
    ) -> Result<PathBuf> {
        let mut last_err: Option<crate::TurError> = None;

        for attempt in 1..=retries {
            match api.get(rfilename).map_err(candle_core::Error::wrap) {
                Ok(path) => return Ok(path),
                Err(e) => {
                    last_err = Some(e.into());
                    if attempt == retries {
                        break;
                    }
                    std::thread::sleep(base_delay * (1u32 << (attempt - 1)));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            TurError::HfHub(
                format!(
                    "Failed downloading {} after {} attempts",
                    rfilename, retries
                )
                .to_string(),
            )
        }))
    }

    fn download_safetensors_model(
        &self,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<ModelPaths> {
        let model_id = self.model_id.as_ref().unwrap();
        let full_repo = resolve_model_repo(model_id)?;

        trace!("Downloading SafeTensors model from {}", full_repo);

        let token = get_token(hf_token, hf_token_path)?;
        let mut builder = ApiBuilder::new().with_progress(true);
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }

        let api = builder.build().map_err(candle_core::Error::wrap)?;
        let repo = api.repo(Repo::with_revision(
            full_repo.clone(),
            RepoType::Model,
            repos::DEFAULT_REVISION.to_string(),
        ));

        let tokenizer_filename = repo
            .get(file_names::TOKENIZER)
            .map_err(candle_core::Error::wrap)?;
        let config_filename = repo
            .get(file_names::CONFIG)
            .map_err(candle_core::Error::wrap)?;

        let tokenizer_config_filename = match repo.get(file_names::TOKENIZER_CONFIG) {
            Ok(f) => f,
            _ => PathBuf::new(),
        };

        let generation_config_filename = match repo.get(file_names::GENERATION_CONFIG) {
            Ok(f) => f,
            _ => PathBuf::new(),
        };

        let mut filenames = Vec::new();
        for rfilename in repo
            .info()
            .map_err(candle_core::Error::wrap)?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| {
                // Include .safetensors files but exclude the index file
                x.ends_with(extensions::SAFETENSORS) && !x.contains(extensions::INDEX_JSON)
            })
        {
            let filename = self.hf_get_with_retry(
                &repo,
                &rfilename,
                retry::MAX_RETRIES,
                std::time::Duration::from_secs(retry::BASE_DELAY_SECS),
            )?;
            filenames.push(filename);
        }

        trace!("Downloaded SafeTensors files: {:?}", filenames);

        Ok(ModelPaths {
            tokenizer_filename,
            tokenizer_config_filename,
            config_filename,
            generation_config_filename,
            filenames,
            auxiliary_filenames: Vec::new(),
            chat_template_filename: None,
        })
    }

    fn download_gguf_model(
        &self,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<ModelPaths> {
        let model_id = self.model_id.as_ref().unwrap();
        let quantization = self.quantization.as_ref().unwrap();

        // Resolve repos: config from main repo, weights from unsloth GGUF repo
        let (config_repo, gguf_repo) = resolve_gguf_repos(model_id)?;
        let gguf_filename = format!(
            "{}-{}{}",
            model_id.split('/').last().unwrap_or(model_id),
            quantization,
            extensions::GGUF
        );

        trace!(
            "Downloading GGUF model:\n  Config/tokenizer from: {}\n  GGUF weights from: {} (file: {})",
            config_repo, gguf_repo, gguf_filename
        );

        let token = get_token(hf_token, hf_token_path)?;
        let mut builder = ApiBuilder::new().with_progress(true);
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }

        let api = builder.build().map_err(candle_core::Error::wrap)?;

        // Download config and tokenizer from main repo
        let config_repo_api = api.repo(Repo::with_revision(
            config_repo.clone(),
            RepoType::Model,
            repos::DEFAULT_REVISION.to_string(),
        ));

        let tokenizer_filename = config_repo_api
            .get(file_names::TOKENIZER)
            .map_err(candle_core::Error::wrap)?;

        let config_filename = config_repo_api
            .get(file_names::CONFIG)
            .map_err(candle_core::Error::wrap)?;

        let tokenizer_config_filename = match config_repo_api.get(file_names::TOKENIZER_CONFIG) {
            Ok(f) => f,
            _ => PathBuf::new(),
        };

        let generation_config_filename = match config_repo_api.get(file_names::GENERATION_CONFIG) {
            Ok(f) => f,
            _ => PathBuf::new(),
        };

        let chat_template_filename = match config_repo_api.get(file_names::CHAT_TEMPLATE) {
            Ok(f) => Some(f),
            _ => None,
        };

        // Download GGUF weight file from unsloth repo
        let gguf_repo_api = api.repo(Repo::with_revision(
            gguf_repo.clone(),
            RepoType::Model,
            repos::DEFAULT_REVISION.to_string(),
        ));

        let downloaded_file = gguf_repo_api
            .get(&gguf_filename)
            .map_err(candle_core::Error::wrap)?;

        trace!("Downloaded GGUF file: {:?}", downloaded_file);
        trace!("Downloaded tokenizer: {:?}", tokenizer_filename);
        trace!("Downloaded config: {:?}", config_filename);

        Ok(ModelPaths {
            tokenizer_filename,
            tokenizer_config_filename,
            config_filename,
            generation_config_filename,
            filenames: vec![downloaded_file],
            auxiliary_filenames: Vec::new(),
            chat_template_filename,
        })
    }
}

fn load_local_safetensors(path: &str, index_name: &str) -> Result<Vec<PathBuf>> {
    let index_path = Path::new(path).join(index_name);
    let data = fs::read_to_string(&index_path).map_err(candle_core::Error::wrap)?;
    let value: Value = serde_json::from_str(&data).map_err(candle_core::Error::wrap)?;
    let weight_map = value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| candle_core::Error::msg("safetensors index missing weight_map"))?;

    let mut filenames: Vec<PathBuf> = weight_map
        .keys()
        .map(|filename| Path::new(path).join(filename))
        .collect();
    filenames.sort();
    Ok(filenames)
}

/// Resolve a simplified model ID to full HuggingFace repo path
/// Examples:
///   - "Qwen3-0.6B" -> "Qwen/Qwen3-0.6B"
///   - "Qwen/Qwen3-0.6B" -> "Qwen/Qwen3-0.6B" (already full)
fn resolve_model_repo(model_id: &str) -> Result<String> {
    if model_id.contains('/') {
        // Already a full repo path
        Ok(model_id.to_string())
    } else {
        // Simplified name - assume Qwen org
        Ok(format!("{}/{}", repos::DEFAULT_ORG, model_id))
    }
}

/// Resolve repos for GGUF downloads
/// Returns (config_repo, gguf_repo)
/// Examples:
///   - "Qwen3-0.6B" -> ("Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF")
///   - "Qwen/Qwen3-0.6B" -> ("Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF")
fn resolve_gguf_repos(model_id: &str) -> Result<(String, String)> {
    let config_repo = resolve_model_repo(model_id)?;
    let model_name = config_repo.split('/').last().unwrap_or(model_id);
    let gguf_repo = format!("{}/{}{}", repos::GGUF_ORG, model_name, repos::GGUF_SUFFIX);

    Ok((config_repo, gguf_repo))
}

fn get_token(hf_token: Option<String>, hf_token_path: Option<String>) -> Result<String> {
    Ok(match (hf_token, hf_token_path) {
        (Some(token), None) => token.trim().to_string(),
        (None, Some(path)) => fs::read_to_string(path)
            .map_err(candle_core::Error::wrap)?
            .trim()
            .to_string(),
        (None, None) => String::new(),
        (Some(token), Some(_)) => token.trim().to_string(),
    })
}
