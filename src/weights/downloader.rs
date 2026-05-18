use crate::{Result, TurError};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::trace;

mod file_names {
    pub const TOKENIZER: &str = "tokenizer.json";
    pub const TOKENIZER_CONFIG: &str = "tokenizer_config.json";
    pub const CONFIG: &str = "config.json";
    pub const GENERATION_CONFIG: &str = "generation_config.json";
    pub const CHAT_TEMPLATE_JSON: &str = "chat_template.json";
    pub const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";
    pub const MODEL_SAFETENSORS: &str = "model.safetensors";
    pub const MODEL_SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
}

mod extensions {
    pub const SAFETENSORS: &str = ".safetensors";
    pub const GGUF: &str = ".gguf";
    pub const INDEX_JSON: &str = ".index.json";
}

mod repos {
    pub const DEFAULT_ORG: &str = "Qwen";
    pub const GGUF_ORG: &str = "unsloth";
    pub const GGUF_SUFFIX: &str = "-GGUF";
    pub const DEFAULT_REVISION: &str = "main";
}

mod retry {
    pub const MAX_RETRIES: u32 = 5;
    pub const BASE_DELAY_SECS: u64 = 5;
}

#[derive(Debug, Clone)]
pub struct Downloader {
    model_id: Option<String>,
    weight_path: Option<String>,
    quantization: Option<String>,
}

/// Resolved paths to all files needed to load a model.
///
/// Optional files (`tokenizer_config_filename`, `generation_config_filename`,
/// `chat_template_filename`) are `None` when the model doesn't provide them.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelPaths {
    pub tokenizer_filename: PathBuf,
    pub tokenizer_config_filename: Option<PathBuf>,
    pub config_filename: PathBuf,
    pub generation_config_filename: Option<PathBuf>,
    pub filenames: Vec<PathBuf>,
    pub auxiliary_filenames: Vec<PathBuf>,
    pub chat_template_filename: Option<PathBuf>,
}

impl ModelPaths {
    pub fn config_filename(&self) -> &Path {
        &self.config_filename
    }

    pub fn tokenizer_filename(&self) -> &Path {
        &self.tokenizer_filename
    }

    pub fn tokenizer_config_filename(&self) -> Option<&Path> {
        self.tokenizer_config_filename.as_deref()
    }

    pub fn weight_filenames(&self) -> &[PathBuf] {
        &self.filenames
    }

    pub fn auxiliary_filenames(&self) -> &[PathBuf] {
        &self.auxiliary_filenames
    }

    pub fn generation_config_filename(&self) -> Option<&Path> {
        self.generation_config_filename.as_deref()
    }

    pub fn chat_template_filename(&self) -> Option<&Path> {
        self.chat_template_filename.as_deref()
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
        match (&self.model_id, &self.weight_path, &self.quantization) {
            (None, Some(path), None) => {
                if !Path::new(path).is_dir() {
                    return Err(TurError::HfHub(
                        "Safetensor weight path must be a directory! \n\t***Tips: use `--f` to specify gguf model file!***".to_string()
                    ));
                }

                let base = Path::new(path);
                let filenames = if base.join(file_names::MODEL_SAFETENSORS_INDEX).exists() {
                    load_local_safetensors(path)?
                } else {
                    vec![base.join(file_names::MODEL_SAFETENSORS)]
                };

                let paths = ModelPaths {
                    tokenizer_filename: base.join(file_names::TOKENIZER),
                    tokenizer_config_filename: Some(base.join(file_names::TOKENIZER_CONFIG)),
                    config_filename: base.join(file_names::CONFIG),
                    generation_config_filename: {
                        let p = base.join(file_names::GENERATION_CONFIG);
                        p.exists().then_some(p)
                    },
                    filenames,
                    auxiliary_filenames: Vec::new(),
                    chat_template_filename: {
                        let jinja = base.join(file_names::CHAT_TEMPLATE_JINJA);
                        let json = base.join(file_names::CHAT_TEMPLATE_JSON);
                        if jinja.exists() { Some(jinja) } else if json.exists() { Some(json) } else { None }
                    },
                };
                Ok((paths, false))
            }
            (Some(_), None, Some(_)) => Ok((self.download_gguf_model()?, true)),
            (Some(_), None, None) => Ok((self.download_safetensors_model()?, false)),
            _ => Err(TurError::HfHub(
                "Invalid configuration!\n***Tips***: \n \t For local model weights: --weight-path <path/to/folder>\n \t For remote SafeTensors: --model-id Qwen3-0.6B\n \t For remote GGUF: --model-id Qwen3-0.6B --quantization Q4_K_M".to_string(),
            )),
        }
    }

    fn build_api(&self) -> Result<hf_hub::api::sync::Api> {
        Ok(ApiBuilder::new()
            .with_progress(true)
            .build()
            .map_err(candle_core::Error::wrap)?)
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

        Err(last_err.expect("loop ran at least once since retries > 0"))
    }

    fn download_safetensors_model(&self) -> Result<ModelPaths> {
        let model_id = self.model_id.as_ref().unwrap();
        let full_repo = resolve_model_repo(model_id);

        trace!("Downloading SafeTensors model from {}", full_repo);

        let api = self.build_api()?;
        let repo = api.repo(Repo::with_revision(
            full_repo,
            RepoType::Model,
            repos::DEFAULT_REVISION.to_string(),
        ));

        let tokenizer_filename = repo
            .get(file_names::TOKENIZER)
            .map_err(candle_core::Error::wrap)?;
        let config_filename = repo
            .get(file_names::CONFIG)
            .map_err(candle_core::Error::wrap)?;

        let tokenizer_config_filename = repo.get(file_names::TOKENIZER_CONFIG).ok();
        let generation_config_filename = repo.get(file_names::GENERATION_CONFIG).ok();
        let chat_template_filename = repo
            .get(file_names::CHAT_TEMPLATE_JINJA)
            .ok()
            .or_else(|| repo.get(file_names::CHAT_TEMPLATE_JSON).ok());

        let mut filenames = Vec::new();
        for rfilename in repo
            .info()
            .map_err(candle_core::Error::wrap)?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.ends_with(extensions::SAFETENSORS) && !x.contains(extensions::INDEX_JSON))
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
            chat_template_filename,
        })
    }

    fn download_gguf_model(&self) -> Result<ModelPaths> {
        let model_id = self.model_id.as_ref().unwrap();
        let quantization = self.quantization.as_ref().unwrap();

        let (config_repo, gguf_repo) = resolve_gguf_repos(model_id);
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

        let api = self.build_api()?;

        let config_repo_api = api.repo(Repo::with_revision(
            config_repo,
            RepoType::Model,
            repos::DEFAULT_REVISION.to_string(),
        ));

        let tokenizer_filename = config_repo_api
            .get(file_names::TOKENIZER)
            .map_err(candle_core::Error::wrap)?;
        let config_filename = config_repo_api
            .get(file_names::CONFIG)
            .map_err(candle_core::Error::wrap)?;

        let tokenizer_config_filename = config_repo_api.get(file_names::TOKENIZER_CONFIG).ok();
        let generation_config_filename = config_repo_api.get(file_names::GENERATION_CONFIG).ok();
        let chat_template_filename = config_repo_api
            .get(file_names::CHAT_TEMPLATE_JINJA)
            .ok()
            .or_else(|| config_repo_api.get(file_names::CHAT_TEMPLATE_JSON).ok());

        let gguf_repo_api = api.repo(Repo::with_revision(
            gguf_repo,
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

/// Load shard filenames from a safetensors index file.
///
/// The index maps weight parameter names to shard filenames — we deduplicate
/// and sort the shard filenames (many parameters share one shard).
fn load_local_safetensors(path: &str) -> Result<Vec<PathBuf>> {
    let index_path = Path::new(path).join(file_names::MODEL_SAFETENSORS_INDEX);
    let data = fs::read_to_string(&index_path).map_err(candle_core::Error::wrap)?;
    let value: Value = serde_json::from_str(&data).map_err(candle_core::Error::wrap)?;
    let weight_map = value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| candle_core::Error::msg("safetensors index missing weight_map"))?;

    // Keys are parameter names; values are shard filenames. Deduplicate via HashSet
    // since many parameters share one shard file.
    let mut dedup: HashSet<&str> = HashSet::new();
    let mut shards: Vec<PathBuf> = weight_map
        .values()
        .filter_map(|v| v.as_str())
        .filter(|s| dedup.insert(s))
        .map(|s| Path::new(path).join(s))
        .collect();
    shards.sort();
    Ok(shards)
}

/// Resolve a simplified model ID to its full HuggingFace repo path.
///
/// - `"Qwen3-0.6B"` → `"Qwen/Qwen3-0.6B"`
/// - `"Qwen/Qwen3-0.6B"` → `"Qwen/Qwen3-0.6B"` (already qualified)
fn resolve_model_repo(model_id: &str) -> String {
    if model_id.contains('/') {
        model_id.to_string()
    } else {
        format!("{}/{}", repos::DEFAULT_ORG, model_id)
    }
}

/// Resolve the two repos needed for a GGUF download.
///
/// Returns `(config_repo, gguf_repo)`.
///
/// - `"Qwen3-0.6B"` → `("Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF")`
fn resolve_gguf_repos(model_id: &str) -> (String, String) {
    let config_repo = resolve_model_repo(model_id);
    let model_name = config_repo.split('/').next_back().unwrap_or(model_id);
    let gguf_repo = format!("{}/{}{}", repos::GGUF_ORG, model_name, repos::GGUF_SUFFIX);
    (config_repo, gguf_repo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_paths() -> ModelPaths {
        ModelPaths {
            tokenizer_filename: PathBuf::from("tokenizer.json"),
            tokenizer_config_filename: Some(PathBuf::from("tokenizer_config.json")),
            config_filename: PathBuf::from("config.json"),
            generation_config_filename: Some(PathBuf::from("generation_config.json")),
            filenames: vec![],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        }
    }

    #[test]
    fn test_model_paths_config_filename() {
        let paths = make_paths();
        assert_eq!(paths.config_filename(), Path::new("config.json"));
    }

    #[test]
    fn test_model_paths_tokenizer_filename() {
        let paths = make_paths();
        assert_eq!(paths.tokenizer_filename(), Path::new("tokenizer.json"));
    }

    #[test]
    fn test_model_paths_tokenizer_config_filename() {
        let paths = make_paths();
        assert_eq!(
            paths.tokenizer_config_filename(),
            Some(Path::new("tokenizer_config.json"))
        );
    }

    #[test]
    fn test_model_paths_tokenizer_config_filename_none() {
        let mut paths = make_paths();
        paths.tokenizer_config_filename = None;
        assert_eq!(paths.tokenizer_config_filename(), None);
    }

    #[test]
    fn test_model_paths_weight_filenames() {
        let weight_files = vec![
            PathBuf::from("model-00001.safetensors"),
            PathBuf::from("model-00002.safetensors"),
        ];
        let mut paths = make_paths();
        paths.filenames = weight_files.clone();
        assert_eq!(paths.weight_filenames(), weight_files.as_slice());
    }

    #[test]
    fn test_model_paths_auxiliary_filenames() {
        let aux_files = vec![PathBuf::from("auxiliary.bin")];
        let mut paths = make_paths();
        paths.auxiliary_filenames = aux_files.clone();
        assert_eq!(paths.auxiliary_filenames(), aux_files.as_slice());
    }

    #[test]
    fn test_model_paths_generation_config_filename() {
        let paths = make_paths();
        assert_eq!(
            paths.generation_config_filename(),
            Some(Path::new("generation_config.json"))
        );
    }

    #[test]
    fn test_model_paths_generation_config_filename_none() {
        let mut paths = make_paths();
        paths.generation_config_filename = None;
        assert_eq!(paths.generation_config_filename(), None);
    }

    #[test]
    fn test_model_paths_chat_template_filename_json() {
        let mut paths = make_paths();
        paths.chat_template_filename = Some(PathBuf::from("chat_template.json"));
        assert_eq!(
            paths.chat_template_filename(),
            Some(Path::new("chat_template.json"))
        );
    }

    #[test]
    fn test_model_paths_chat_template_filename_jinja() {
        let mut paths = make_paths();
        paths.chat_template_filename = Some(PathBuf::from("chat_template.jinja"));
        assert_eq!(
            paths.chat_template_filename(),
            Some(Path::new("chat_template.jinja"))
        );
    }

    #[test]
    fn test_model_paths_chat_template_filename_none() {
        let paths = make_paths();
        assert_eq!(paths.chat_template_filename(), None);
    }

    #[test]
    fn test_model_paths_clone() {
        let mut paths = make_paths();
        paths.filenames = vec![PathBuf::from("model.safetensors")];
        paths.chat_template_filename = Some(PathBuf::from("chat_template.jinja"));
        let cloned = paths.clone();
        assert_eq!(cloned.tokenizer_filename(), paths.tokenizer_filename());
        assert_eq!(cloned.config_filename(), paths.config_filename());
        assert_eq!(cloned.weight_filenames(), paths.weight_filenames());
        assert_eq!(
            cloned.chat_template_filename(),
            paths.chat_template_filename()
        );
    }

    // ── file_names constants ─────────────────────────────────────────────────

    #[test]
    fn file_names_chat_template_jinja_constant() {
        assert_eq!(file_names::CHAT_TEMPLATE_JINJA, "chat_template.jinja");
    }

    #[test]
    fn file_names_chat_template_json_constant() {
        assert_eq!(file_names::CHAT_TEMPLATE_JSON, "chat_template.json");
    }

    // ── local-path chat template probe ───────────────────────────────────────

    fn chat_template_probe(base: &Path) -> Option<PathBuf> {
        let jinja = base.join(file_names::CHAT_TEMPLATE_JINJA);
        let json = base.join(file_names::CHAT_TEMPLATE_JSON);
        if jinja.exists() {
            Some(jinja)
        } else if json.exists() {
            Some(json)
        } else {
            None
        }
    }

    #[test]
    fn local_path_prefers_jinja_over_json() {
        let base = std::env::temp_dir().join("tur_test_jinja_over_json");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("chat_template.jinja"), "jinja").unwrap();
        std::fs::write(base.join("chat_template.json"), "json").unwrap();

        let result = chat_template_probe(&base);
        assert_eq!(result, Some(base.join("chat_template.jinja")));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn local_path_falls_back_to_json_when_no_jinja() {
        let base = std::env::temp_dir().join("tur_test_json_fallback");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("chat_template.json"), "json").unwrap();

        let result = chat_template_probe(&base);
        assert_eq!(result, Some(base.join("chat_template.json")));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn local_path_returns_none_when_no_chat_template_files() {
        let base = std::env::temp_dir().join("tur_test_no_chat_template");
        std::fs::create_dir_all(&base).unwrap();
        // Ensure neither file exists in this fresh dir
        let _ = std::fs::remove_file(base.join("chat_template.jinja"));
        let _ = std::fs::remove_file(base.join("chat_template.json"));

        let result = chat_template_probe(&base);
        assert_eq!(result, None);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_resolve_model_repo_simple_name() {
        assert_eq!(resolve_model_repo("Qwen3-0.6B"), "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn test_resolve_model_repo_full_path() {
        assert_eq!(resolve_model_repo("Qwen/Qwen3-0.6B"), "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn test_resolve_model_repo_custom_org() {
        assert_eq!(
            resolve_model_repo("custom-org/model-name"),
            "custom-org/model-name"
        );
    }

    #[test]
    fn test_resolve_gguf_repos_simple_name() {
        let (config_repo, gguf_repo) = resolve_gguf_repos("Qwen3-0.6B");
        assert_eq!(config_repo, "Qwen/Qwen3-0.6B");
        assert_eq!(gguf_repo, "unsloth/Qwen3-0.6B-GGUF");
    }

    #[test]
    fn test_resolve_gguf_repos_full_path() {
        let (config_repo, gguf_repo) = resolve_gguf_repos("Qwen/Qwen3-0.6B");
        assert_eq!(config_repo, "Qwen/Qwen3-0.6B");
        assert_eq!(gguf_repo, "unsloth/Qwen3-0.6B-GGUF");
    }

    #[test]
    fn test_resolve_gguf_repos_custom_org() {
        let (config_repo, gguf_repo) = resolve_gguf_repos("custom-org/model-name");
        assert_eq!(config_repo, "custom-org/model-name");
        assert_eq!(gguf_repo, "unsloth/model-name-GGUF");
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
