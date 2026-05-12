use candle_core::Result;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::trace;

#[derive(Debug, Clone)]
pub struct Downloader {
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub weight_file: Option<String>,
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
        weight_file: Option<String>,
    ) -> Self {
        Self {
            model_id,
            weight_path,
            weight_file,
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
            &self.weight_file,
        ) {
            (None, Some(path), None) => {
                if !Path::new(path).is_dir() {
                    candle_core::bail!(
                        "Safetensor weight path must be a directory! \n\t***Tips: use `--f` to specify gguf model file!***"
                    );
                }

                let mut filenames = vec![];
                let index_path = Path::new(path).join("model.safetensors.index.json");
                if index_path.exists() {
                    filenames = load_local_safetensors(path, "model.safetensors.index.json")?;
                } else {
                    filenames.push(Path::new(path).join("model.safetensors"));
                }

                (
                    ModelPaths {
                        tokenizer_filename: Path::new(path).join("tokenizer.json"),
                        tokenizer_config_filename: Path::new(path).join("tokenizer_config.json"),
                        config_filename: Path::new(path).join("config.json"),
                        generation_config_filename: if Path::new(path)
                            .join("generation_config.json")
                            .exists()
                        {
                            Path::new(path).join("generation_config.json")
                        } else {
                            PathBuf::new()
                        },
                        filenames,
                        auxiliary_filenames: Vec::new(),
                        chat_template_filename: if Path::new(path)
                            .join("chat_template.json")
                            .exists()
                        {
                            Some(Path::new(path).join("chat_template.json"))
                        } else {
                            None
                        },
                    },
                    false,
                )
            }
            (Some(_), None, Some(_)) => (self.download_gguf_model(hf_token, hf_token_path)?, true),
            (Some(_), None, None) => (self.download_model(hf_token, hf_token_path)?, false),
            _ => {
                candle_core::bail!(
                    "No model id or weight_path/weight_file provided!\n***Tips***: \n \t For local model weights, `--w <path/to/folder>` for safetensors models or gguf models.\n \t For remote safetensor models, `--m <model_id>` to download from HuggingFace hub. \n \t For remote gguf models, `--m <model_id> --f <weight_file>` to download from HuggingFace hub."
                );
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
        let mut last_err: Option<candle_core::Error> = None;

        for attempt in 1..=retries {
            match api.get(rfilename).map_err(candle_core::Error::wrap) {
                Ok(path) => return Ok(path),
                Err(e) => {
                    last_err = Some(e);
                    if attempt == retries {
                        break;
                    }
                    std::thread::sleep(base_delay * (1u32 << (attempt - 1)));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            candle_core::Error::msg(format!(
                "Failed downloading {} after {} attempts",
                rfilename, retries
            ))
        }))
    }

    fn download_model(
        &self,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<ModelPaths> {
        let model_id = self.model_id.as_ref().unwrap();
        trace!("Downloading model {model_id} from HF Hub");

        let token = get_token(hf_token, hf_token_path)?;
        let mut builder = ApiBuilder::new().with_progress(true);
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }

        let api = builder.build().map_err(candle_core::Error::wrap)?;
        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo
            .get("tokenizer.json")
            .map_err(candle_core::Error::wrap)?;
        let config_filename = repo.get("config.json").map_err(candle_core::Error::wrap)?;

        let tokenizer_config_filename = match repo.get("tokenizer_config.json") {
            Ok(f) => f,
            _ => PathBuf::new(),
        };

        let generation_config_filename = match repo.get("generation_config.json") {
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
                x.ends_with(".safetensors") && !x.contains(".index.json")
            })
        {
            let filename =
                self.hf_get_with_retry(&repo, &rfilename, 5, std::time::Duration::from_secs(5))?;
            filenames.push(filename);
        }

        trace!("Downloaded files for the model: {:?}", filenames);

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
        let filename = self.weight_file.as_ref().unwrap();

        let token = get_token(hf_token, hf_token_path)?;
        let mut builder = ApiBuilder::new().with_progress(true);
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }

        let api = builder.build().map_err(candle_core::Error::wrap)?;
        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            "main".to_string(),
        ));

        let downloaded_file = repo
            .get(filename.as_str())
            .map_err(candle_core::Error::wrap)?;
        Ok(ModelPaths {
            tokenizer_filename: PathBuf::new(),
            tokenizer_config_filename: PathBuf::new(),
            config_filename: PathBuf::new(),
            generation_config_filename: PathBuf::new(),
            filenames: vec![downloaded_file],
            auxiliary_filenames: Vec::new(),
            chat_template_filename: None,
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
