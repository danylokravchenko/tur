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

    pub fn prepare_model_weights(&self) -> Result<(ModelPaths, bool)> {
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
            (Some(_), None, Some(_)) => (self.download_gguf_model()?, true),
            (Some(_), None, None) => (self.download_safetensors_model()?, false),
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

    fn download_safetensors_model(&self) -> Result<ModelPaths> {
        let model_id = self.model_id.as_ref().unwrap();
        let full_repo = resolve_model_repo(model_id)?;

        trace!("Downloading SafeTensors model from {}", full_repo);

        let builder = ApiBuilder::new().with_progress(true);

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

    fn download_gguf_model(&self) -> Result<ModelPaths> {
        let model_id = self.model_id.as_ref().unwrap();
        let quantization = self.quantization.as_ref().unwrap();

        // Resolve repos: config from main repo, weights from unsloth GGUF repo
        let (config_repo, gguf_repo) = resolve_gguf_repos(model_id)?;
        let gguf_filename = format!(
            "{}-{}{}",
            model_id.split('/').next_back().unwrap_or(model_id),
            quantization,
            extensions::GGUF
        );

        trace!(
            "Downloading GGUF model:\n  Config/tokenizer from: {}\n  GGUF weights from: {} (file: {})",
            config_repo, gguf_repo, gguf_filename
        );

        let builder = ApiBuilder::new().with_progress(true);

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

        let chat_template_filename = config_repo_api.get(file_names::CHAT_TEMPLATE).ok();

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
    let model_name = config_repo.split('/').next_back().unwrap_or(model_id);
    let gguf_repo = format!("{}/{}{}", repos::GGUF_ORG, model_name, repos::GGUF_SUFFIX);

    Ok((config_repo, gguf_repo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_downloader_new() {
        let downloader = Downloader::new(
            Some("Qwen3-0.6B".to_string()),
            None,
            Some("Q4_K_M".to_string()),
        );
        assert_eq!(downloader.model_id, Some("Qwen3-0.6B".to_string()));
        assert_eq!(downloader.weight_path, None);
        assert_eq!(downloader.quantization, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn test_downloader_new_all_none() {
        let downloader = Downloader::new(None, None, None);
        assert_eq!(downloader.model_id, None);
        assert_eq!(downloader.weight_path, None);
        assert_eq!(downloader.quantization, None);
    }

    #[test]
    fn test_downloader_new_with_weight_path() {
        let downloader = Downloader::new(None, Some("/path/to/weights".to_string()), None);
        assert_eq!(downloader.model_id, None);
        assert_eq!(downloader.weight_path, Some("/path/to/weights".to_string()));
        assert_eq!(downloader.quantization, None);
    }

    #[test]
    fn test_model_paths_get_config_filename() {
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        assert_eq!(paths.get_config_filename(), PathBuf::from("config.json"));
    }

    #[test]
    fn test_model_paths_get_tokenizer_filename() {
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        assert_eq!(
            paths.get_tokenizer_filename(),
            PathBuf::from("tokenizer.json")
        );
    }

    #[test]
    fn test_model_paths_get_tokenizer_config_filename() {
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        assert_eq!(
            paths.get_tokenizer_config_filename(),
            PathBuf::from("tokenizer_config.json")
        );
    }

    #[test]
    fn test_model_paths_get_weight_filenames() {
        let weight_files = vec![
            PathBuf::from("model-00001.safetensors"),
            PathBuf::from("model-00002.safetensors"),
        ];
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: weight_files.clone(),
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        assert_eq!(paths.get_weight_filenames(), weight_files);
    }

    #[test]
    fn test_model_paths_get_auxiliary_filenames() {
        let aux_files = vec![PathBuf::from("auxiliary.bin")];
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![],
            auxiliary_filenames: aux_files.clone(),
            chat_template_filename: None,
        };
        assert_eq!(paths.get_auxiliary_filenames(), aux_files);
    }

    #[test]
    fn test_model_paths_get_generation_config_filename() {
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        assert_eq!(
            paths.get_generation_config_filename(),
            PathBuf::from("generation_config.json")
        );
    }

    #[test]
    fn test_model_paths_get_chat_template_filename_some() {
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![],
            auxiliary_filenames: vec![],
            chat_template_filename: Some(PathBuf::from("chat_template.json")),
        };
        assert_eq!(
            paths.get_chat_template_filename(),
            Some(PathBuf::from("chat_template.json"))
        );
    }

    #[test]
    fn test_model_paths_get_chat_template_filename_none() {
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        assert_eq!(paths.get_chat_template_filename(), None);
    }

    #[test]
    fn test_resolve_model_repo_simple_name() {
        let result = resolve_model_repo("Qwen3-0.6B").unwrap();
        assert_eq!(result, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn test_resolve_model_repo_full_path() {
        let result = resolve_model_repo("Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(result, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn test_resolve_model_repo_custom_org() {
        let result = resolve_model_repo("custom-org/model-name").unwrap();
        assert_eq!(result, "custom-org/model-name");
    }

    #[test]
    fn test_resolve_gguf_repos_simple_name() {
        let (config_repo, gguf_repo) = resolve_gguf_repos("Qwen3-0.6B").unwrap();
        assert_eq!(config_repo, "Qwen/Qwen3-0.6B");
        assert_eq!(gguf_repo, "unsloth/Qwen3-0.6B-GGUF");
    }

    #[test]
    fn test_resolve_gguf_repos_full_path() {
        let (config_repo, gguf_repo) = resolve_gguf_repos("Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(config_repo, "Qwen/Qwen3-0.6B");
        assert_eq!(gguf_repo, "unsloth/Qwen3-0.6B-GGUF");
    }

    #[test]
    fn test_resolve_gguf_repos_custom_org() {
        let (config_repo, gguf_repo) = resolve_gguf_repos("custom-org/model-name").unwrap();
        assert_eq!(config_repo, "custom-org/model-name");
        assert_eq!(gguf_repo, "unsloth/model-name-GGUF");
    }

    #[test]
    fn test_downloader_clone() {
        let downloader = Downloader::new(
            Some("Qwen3-0.6B".to_string()),
            None,
            Some("Q4_K_M".to_string()),
        );
        let cloned = downloader.clone();
        assert_eq!(cloned.model_id, downloader.model_id);
        assert_eq!(cloned.weight_path, downloader.weight_path);
        assert_eq!(cloned.quantization, downloader.quantization);
    }

    #[test]
    fn test_model_paths_clone() {
        let paths = ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: PathBuf::from("tokenizer_config.json"),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: PathBuf::from("generation_config.json"),
            filenames: vec![PathBuf::from("model.safetensors")],
            auxiliary_filenames: vec![],
            chat_template_filename: Some(PathBuf::from("chat_template.json")),
        };
        let cloned = paths.clone();
        assert_eq!(cloned.tokenizer_filename, paths.tokenizer_filename);
        assert_eq!(cloned.config_filename, paths.config_filename);
        assert_eq!(cloned.filenames, paths.filenames);
        assert_eq!(cloned.chat_template_filename, paths.chat_template_filename);
    }

    #[test]
    fn test_prepare_model_weights_invalid_config_all_none() {
        let downloader = Downloader::new(None, None, None);
        let result = downloader.prepare_model_weights();
        assert!(result.is_err());
        if let Err(TurError::HfHub(msg)) = result {
            assert!(msg.contains("Invalid configuration"));
        }
    }

    #[test]
    fn test_prepare_model_weights_invalid_config_model_and_path() {
        let downloader = Downloader::new(
            Some("Qwen3-0.6B".to_string()),
            Some("/path/to/weights".to_string()),
            None,
        );
        let result = downloader.prepare_model_weights();
        assert!(result.is_err());
        if let Err(TurError::HfHub(msg)) = result {
            assert!(msg.contains("Invalid configuration"));
        }
    }

    #[test]
    fn test_prepare_model_weights_invalid_config_all_some() {
        let downloader = Downloader::new(
            Some("Qwen3-0.6B".to_string()),
            Some("/path/to/weights".to_string()),
            Some("Q4_K_M".to_string()),
        );
        let result = downloader.prepare_model_weights();
        assert!(result.is_err());
        if let Err(TurError::HfHub(msg)) = result {
            assert!(msg.contains("Invalid configuration"));
        }
    }
}
