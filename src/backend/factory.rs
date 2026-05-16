use candle_core::{DType, Device};
use tokenizers::Tokenizer;
use tracing::{debug, trace};

use crate::models::ModelImpl;
use crate::weights::{Downloader, VarBuilderX};
use crate::{ProgressReporter, Result};

/// Source for model weights — exactly one of HuggingFace or a local path.
///
/// Using an enum instead of two `Option<String>` fields makes invalid states
/// (both absent, or both present) unrepresentable at the type level.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// HuggingFace model ID (e.g. `"Qwen/Qwen3-0.6B"` or shorthand `"Qwen3-0.6B"`)
    HuggingFace(String),
    /// Local filesystem path to a directory containing model weights
    LocalPath(String),
}

/// Factory for creating LLM models with unified weight loading, tokenization, and model instantiation.
///
/// # Type Parameters
/// * `T` - The model type implementing [`ModelImpl`] trait
pub struct ModelFactory<T: ModelImpl> {
    source: ModelSource,
    quantization: Option<String>,
    device: Device,
    dtype: DType,
    _phantom: std::marker::PhantomData<T>,
}

/// Trait for model constructors that can be instantiated from configuration and weights
pub trait ModelConstructor: ModelImpl + Sized {
    /// The configuration type for this model
    type Config;

    /// Create a new model instance with progress reporting
    fn new_with_progress(
        config: &Self::Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<Self>;

    /// Load the model configuration from file paths
    fn load_config(paths: &crate::weights::ModelPaths) -> Result<Self::Config>;
}

impl<T: ModelConstructor> ModelFactory<T> {
    /// Creates a new `ModelFactory`.
    ///
    /// # Arguments
    /// * `source` - Where to load weights from (HuggingFace or local path)
    /// * `quantization` - Quantization level for GGUF models (e.g., `"Q4_K_M"`)
    /// * `device` - Compute device (CPU, CUDA, Metal, etc.)
    /// * `dtype` - Data type for non-quantized operations (BF16, F32, etc.)
    pub fn new(
        source: ModelSource,
        quantization: Option<String>,
        device: Device,
        dtype: DType,
    ) -> Self {
        trace!(
            source = ?source,
            quantization = ?quantization,
            device = ?device,
            dtype = ?dtype,
            "Initializing ModelFactory",
        );

        Self {
            source,
            quantization,
            device,
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a reference to the device used by this factory.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Creates and loads a model with its tokenizer.
    pub fn create_model(&self, progress: Option<&ProgressReporter>) -> Result<(T, Tokenizer)> {
        debug!(
            device = ?self.device,
            dtype = ?self.dtype,
            "Creating model",
        );

        let (paths, gguf) = self.prepare_weights()?;
        let config = self.load_config(&paths)?;
        let tokenizer = self.load_tokenizer(&paths)?;
        let vb = self.create_var_builder(&paths, gguf)?;
        let model = self.instantiate_model(&config, vb, progress)?;

        if gguf {
            debug!("Loaded quantized model (GGUF format)");
        } else {
            debug!("Loaded full-precision model");
        }
        debug!("Model creation completed successfully");
        Ok((model, tokenizer))
    }

    /// Prepares and downloads model weights.
    fn prepare_weights(&self) -> Result<(crate::weights::ModelPaths, bool)> {
        trace!("Step 1: Preparing model weights");

        // Log source details and build Downloader — inline tracing fields avoid
        // eager String allocation when trace is disabled.
        let downloader = match &self.source {
            ModelSource::HuggingFace(model_id) => {
                let model_name = model_id.split('/').next_back().unwrap_or(model_id);
                if let Some(ref quant) = self.quantization {
                    debug!(model_name, quant, "Downloading quantized GGUF model");
                } else {
                    debug!(model_name, "Downloading full-precision SafeTensors model");
                }
                trace!(model_id, "Model source: HuggingFace");
                Downloader::new(Some(model_id.clone()), None, self.quantization.clone())
            }
            ModelSource::LocalPath(path) => {
                debug!(path, "Using local model weights");
                trace!(path, "Model source: local path");
                Downloader::new(None, Some(path.clone()), self.quantization.clone())
            }
        };

        let (paths, gguf) = downloader.prepare_model_weights()?;
        debug!(gguf_format = gguf, "Prepared model weights");
        Ok((paths, gguf))
    }

    /// Loads and parses the model configuration.
    fn load_config(&self, paths: &crate::weights::ModelPaths) -> Result<T::Config> {
        trace!("Step 2: Loading model configuration");
        T::load_config(paths)
    }

    /// Loads the tokenizer from the prepared weights.
    fn load_tokenizer(&self, paths: &crate::weights::ModelPaths) -> Result<Tokenizer> {
        trace!("Step 3: Loading tokenizer");

        let tokenizer_path = paths.tokenizer_filename();
        trace!(tokenizer_path = %tokenizer_path.display(), "Tokenizer path");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| crate::TurError::Other(format!("Failed to load tokenizer: {}", e)))?;

        debug!(tokenizer_path = %tokenizer_path.display(), "Loaded tokenizer");
        Ok(tokenizer)
    }

    /// Creates a VarBuilder for loading model weights.
    fn create_var_builder(
        &self,
        paths: &crate::weights::ModelPaths,
        gguf: bool,
    ) -> Result<VarBuilderX<'static>> {
        trace!("Step 4: Creating VarBuilder for model weights");

        let num_files = paths.weight_filenames().len();

        if gguf {
            debug!(num_files, "Loading GGUF quantized model");
            if self.device.is_cpu() {
                debug!("CPU mode: linear layers use quantized weights (QMatMul)");
            } else {
                debug!(dtype = ?self.dtype, "GPU/Metal mode: dequantizing linear layers");
            }
        } else {
            debug!(num_files, dtype = ?self.dtype, "Loading full-precision SafeTensors model");
        }

        let vb = VarBuilderX::new(paths, gguf, self.dtype, &self.device)?;
        trace!("VarBuilder created successfully");
        Ok(vb)
    }

    /// Instantiates the model from config and weights.
    fn instantiate_model(
        &self,
        config: &T::Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<T> {
        trace!("Step 5: Instantiating model");
        T::new_with_progress(config, vb, progress)
    }
}

impl ModelConstructor for crate::models::Qwen35ModelForCausalLM {
    type Config = crate::models::qwen3::Config;

    fn new_with_progress(
        config: &Self::Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<Self> {
        crate::models::Qwen35ModelForCausalLM::new_with_progress(config, vb, progress)
            .map_err(|e| e.into())
    }

    fn load_config(paths: &crate::weights::ModelPaths) -> Result<Self::Config> {
        trace!("Loading Qwen3 model configuration");

        let config_path = paths.config_filename();
        trace!(config_path = %config_path.display(), "Config file path");

        let config_content = std::fs::read_to_string(config_path)?;
        // Single-pass parse: no intermediate serde_json::Value allocation
        let config: Self::Config = serde_json::from_str(&config_content)?;

        debug!(
            num_layers = config.num_hidden_layers,
            hidden_size = config.hidden_size,
            "Model config loaded"
        );
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_factory_creation() {
        let factory = ModelFactory::<crate::models::Qwen35ModelForCausalLM>::new(
            ModelSource::HuggingFace("Qwen3-0.6B".to_string()),
            Some("Q4_K_M".to_string()),
            Device::Cpu,
            DType::F32,
        );

        // Test through the public interface, not internal fields
        assert!(factory.device().is_cpu());
    }

    #[test]
    fn test_model_factory_local_path() {
        let factory = ModelFactory::<crate::models::Qwen35ModelForCausalLM>::new(
            ModelSource::LocalPath("/tmp/model".to_string()),
            None,
            Device::Cpu,
            DType::BF16,
        );

        assert!(factory.device().is_cpu());
    }
}
