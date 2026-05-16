use candle_core::{DType, Device};
use tokenizers::Tokenizer;
use tracing::{debug, trace};

use crate::models::Qwen35ModelForCausalLM;
use crate::models::qwen3::Config;
use crate::weights::{Downloader, VarBuilderX};
use crate::{ProgressReporter, Result};

/// Factory for creating LLM models with unified weight loading, tokenization, and model instantiation.
///
/// This factory encapsulates the complexity of:
/// - Downloading model weights (both SafeTensors and GGUF formats)
/// - Loading model configuration
/// - Loading tokenizers
/// - Creating VarBuilders for different quantization formats
/// - Instantiating the model
///
/// # Example
/// ```ignore
/// let factory = ModelFactory::new(
///     Some("Qwen3-0.6B".to_string()),
///     None,
///     Some("Q4_K_M".to_string()),
///     device,
///     DType::BF16,
/// );
///
/// let (model, tokenizer) = factory.create_model(None)?;
/// ```
pub struct ModelFactory {
    model_id: Option<String>,
    weight_path: Option<String>,
    quantization: Option<String>,
    device: Device,
    dtype: DType,
}

impl ModelFactory {
    /// Creates a new ModelFactory with the specified configuration.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "Qwen3-0.6B")
    /// * `weight_path` - Local path to model weights (alternative to model_id)
    /// * `quantization` - Quantization level for GGUF models (e.g., "Q4_K_M")
    /// * `device` - Compute device (CPU, CUDA, Metal, etc.)
    /// * `dtype` - Data type for non-quantized operations (BF16, F32, etc.)
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        quantization: Option<String>,
        device: Device,
        dtype: DType,
    ) -> Self {
        trace!(
            model_id,
            weight_path = ?weight_path,
            quantization = ?quantization,
            device = ?device,
            dtype = ?dtype,
            "Initializing ModelFactory",
        );

        Self {
            model_id,
            weight_path,
            quantization,
            device,
            dtype,
        }
    }

    /// Creates and loads a model with its tokenizer.
    ///
    /// This method:
    /// 1. Downloads/prepares model weights
    /// 2. Loads the model configuration
    /// 3. Loads the tokenizer
    /// 4. Creates a VarBuilder for weight loading
    /// 5. Instantiates the model
    ///
    /// # Arguments
    /// * `progress` - Optional progress reporter for tracking model loading
    ///
    /// # Returns
    /// A tuple containing the loaded model and tokenizer
    pub fn create_model(
        &self,
        progress: Option<&ProgressReporter>,
    ) -> Result<(Qwen35ModelForCausalLM, Tokenizer)> {
        debug!(
            device = ?self.device,
            dtype = ?self.dtype,
            "Creating model factory",
        );

        // Step 1: Download and prepare weights
        self.prepare_weights().and_then(|(paths, gguf)| {
            // Step 2: Load configuration
            let config = self.load_config(&paths)?;

            // Step 3: Load tokenizer
            let tokenizer = self.load_tokenizer(&paths)?;

            // Step 4: Create VarBuilder
            let vb = self.create_var_builder(&paths, gguf)?;

            // Step 5: Create model
            let model = self.instantiate_model(&config, vb, progress, gguf)?;

            debug!("Model creation completed successfully");
            Ok((model, tokenizer))
        })
    }

    /// Prepares and downloads model weights.
    fn prepare_weights(&self) -> Result<(crate::weights::ModelPaths, bool)> {
        trace!("Step 1: Preparing model weights");

        let model_source = if let Some(ref model_id) = self.model_id {
            let model_name = model_id.split('/').next_back().unwrap_or(model_id);
            if let Some(ref quant) = self.quantization {
                debug!(model_name, quant, "Downloading quantized GGUF model",);
            } else {
                debug!(model_name, "Downloading full-precision SafeTensors model",);
            }
            format!("{:?} (model_id)", model_id)
        } else if let Some(ref path) = self.weight_path {
            debug!(path, "Using local model weights");
            format!("{:?} (local path)", path)
        } else {
            unreachable!("ModelFactory requires either model_id or weight_path")
        };

        trace!(model_source, "Model source");
        let downloader = Downloader::new(
            self.model_id.clone(),
            self.weight_path.clone(),
            self.quantization.clone(),
        );
        let (paths, gguf) = downloader.prepare_model_weights()?;

        debug!(gguf_format = gguf, "Prepared model weights");
        trace!("Weights prepared successfully");

        Ok((paths, gguf))
    }

    /// Loads and parses the model configuration.
    fn load_config(&self, paths: &crate::weights::ModelPaths) -> Result<Config> {
        trace!("Step 2: Loading model configuration");

        let config_path = paths.get_config_filename();
        trace!(config_path = %config_path.display(), "Config file path");

        let config_content = std::fs::read_to_string(&config_path)?;
        trace!("Config file read successfully");

        let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
        let config: Config = serde_json::from_value(config_json)?;

        debug!(
            num_layers = config.num_hidden_layers,
            hidden_size = config.hidden_size,
            "Model Config loaded"
        );
        trace!("Configuration loaded successfully");

        Ok(config)
    }

    /// Loads the tokenizer from the prepared weights.
    fn load_tokenizer(&self, paths: &crate::weights::ModelPaths) -> Result<Tokenizer> {
        trace!("Step 3: Loading tokenizer");

        let tokenizer_path = paths.get_tokenizer_filename();
        trace!(tokenizer_path = %tokenizer_path.display(), "Tokenizer path");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| crate::TurError::Other(format!("Failed to load tokenizer: {}", e)))?;

        debug!(tokenizer_path = %tokenizer_path.display(), "Loaded Tokenizer");
        trace!("Tokenizer loaded successfully");

        Ok(tokenizer)
    }

    /// Creates a VarBuilder for loading model weights.
    fn create_var_builder(
        &self,
        paths: &crate::weights::ModelPaths,
        gguf: bool,
    ) -> Result<VarBuilderX<'static>> {
        trace!("Step 4: Creating VarBuilder for model weights");

        let weight_files = paths.get_weight_filenames();

        if gguf {
            debug!(
                num_files = weight_files.len(),
                "Loading GGUF quantized model",
            );
            if self.device.is_cpu() {
                debug!(
                    "CPU mode: Linear layers will use quantized weights (QMatMul) for memory efficiency"
                );
            } else {
                debug!(
                    dtype = ?self.dtype,
                    "GPU/Metal mode: Dequantizing linear layers for better performance",
                );
            }
            debug!(dtype = ?self.dtype, "Embeddings and norms will use");
        } else {
            debug!(
                "Loading full-precision SafeTensors model from {} files",
                weight_files.len()
            );
            debug!(dtype = ?self.dtype, "All operations will use");
        }

        let vb = VarBuilderX::new(paths, gguf, self.dtype, &self.device)?;
        trace!("VarBuilder created successfully");

        Ok(vb)
    }

    /// Instantiates the model from the VarBuilder.
    fn instantiate_model(
        &self,
        config: &Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
        gguf: bool,
    ) -> Result<Qwen35ModelForCausalLM> {
        trace!("Step 5: Instantiating model");

        let model = Qwen35ModelForCausalLM::new_with_progress(config, vb, progress)?;

        if gguf {
            debug!("✓ Loaded quantized model (GGUF format)");
        } else {
            debug!("✓ Loaded full-precision model");
        }

        trace!("Model instantiated successfully");
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_factory_creation() {
        let device = Device::Cpu;
        let factory = ModelFactory::new(
            Some("Qwen3-0.6B".to_string()),
            None,
            Some("Q4_K_M".to_string()),
            device,
            DType::F32,
        );

        assert_eq!(factory.model_id, Some("Qwen3-0.6B".to_string()));
        assert_eq!(factory.quantization, Some("Q4_K_M".to_string()));
        assert!(factory.weight_path.is_none());
    }
}
