use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;
use tracing::{debug, trace};

use crate::backend::chat_template::ChatTemplate;
use crate::models::{ModelImpl, ModelInput, kv_cache::PagedKvCache};
use crate::weights::{Downloader, VarBuilderX};
use crate::{ProgressReporter, Result, TurError};

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
    fn load_chat_template_from_paths(paths: &crate::weights::ModelPaths) -> Option<ChatTemplate> {
        // Primary: inline `chat_template` field inside tokenizer_config.json.
        if let Some(config_path) = paths.tokenizer_config_filename() {
            match ChatTemplate::from_tokenizer_config(config_path) {
                Ok(ct) => return Some(ct),
                Err(e) => tracing::debug!(
                    "No inline chat_template in tokenizer_config.json ({e}); \
                     trying standalone file"
                ),
            }
        }

        // Fallback: standalone chat_template.jinja / chat_template.json file.
        if let Some(jinja_path) = paths.chat_template_filename() {
            match std::fs::read_to_string(jinja_path) {
                Ok(raw) => match ChatTemplate::from_template(raw) {
                    Ok(ct) => {
                        tracing::debug!(
                            path = %jinja_path.display(),
                            "Loaded chat template from standalone file"
                        );
                        return Some(ct);
                    }
                    Err(e) => tracing::warn!(
                        path = %jinja_path.display(),
                        "Failed to parse standalone chat template: {e}"
                    ),
                },
                Err(e) => tracing::warn!(
                    path = %jinja_path.display(),
                    "Failed to read standalone chat template: {e}"
                ),
            }
        }

        None
    }

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

    /// Loads only the tokenizer without instantiating the model weights.
    ///
    /// Useful when only the tokenizer is needed (e.g. building a guidance
    /// factory) and loading the full model would be wasteful.
    pub fn load_tokenizer_only(&self) -> Result<Tokenizer> {
        let (paths, _gguf) = self.prepare_weights()?;
        self.load_tokenizer(&paths)
    }

    /// Creates and loads a model with its tokenizer and chat template.
    ///
    /// The chat template is loaded from `tokenizer_config.json` in the same
    /// weight-preparation pass, so `prepare_weights` is called only once.
    /// Returns `None` for the template when the file is absent or has no
    /// `chat_template` field.
    pub fn create_model(
        &self,
        progress: Option<&ProgressReporter>,
    ) -> Result<(T, Tokenizer, Option<ChatTemplate>)> {
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
        let chat_template = Self::load_chat_template_from_paths(&paths);

        if gguf {
            debug!("Loaded quantized model (GGUF format)");
        } else {
            debug!("Loaded full-precision model");
        }
        debug!("Model creation completed successfully");
        Ok((model, tokenizer, chat_template))
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

impl ModelConstructor for crate::models::Granite41ModelForCausalLM {
    type Config = crate::models::granite_41::Config;

    fn new_with_progress(
        config: &Self::Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<Self> {
        crate::models::Granite41ModelForCausalLM::new_with_progress(config, vb, progress)
            .map_err(|e| e.into())
    }

    fn load_config(paths: &crate::weights::ModelPaths) -> Result<Self::Config> {
        trace!("Loading Granite 4.1 model configuration");

        let config_path = paths.config_filename();
        let config_content = std::fs::read_to_string(config_path)?;
        let config: Self::Config = serde_json::from_str(&config_content)?;

        debug!(
            num_layers = config.num_hidden_layers,
            hidden_size = config.hidden_size,
            "Model config loaded"
        );
        Ok(config)
    }
}

// ─── Model detection ──────────────────────────────────────────────────────────

/// Identifies which model architecture is present in a weight directory.
///
/// Add a new variant here (plus the matching arms in [`AnyModel`] and
/// [`AnyModelConfig`]) to introduce support for a new architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Qwen3 / Qwen2 family — shares the same architecture in this codebase.
    Qwen3,
    /// IBM Granite 4.1 Dense.
    Granite41,
}

impl ModelKind {
    /// Recognise the model kind from the `model_type` field in `config.json`.
    fn from_config_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3" | "qwen2" => Some(Self::Qwen3),
            "granite" => Some(Self::Granite41),
            _ => None,
        }
    }

    /// Infer the model kind from a HuggingFace repo ID or local directory name
    /// when `config.json` is absent or its `model_type` field is unknown.
    fn from_name_hint(name: &str) -> Option<Self> {
        let lower = name.to_lowercase();
        if lower.contains("qwen3") || lower.contains("qwen2") {
            return Some(Self::Qwen3);
        }
        if lower.contains("granite") {
            return Some(Self::Granite41);
        }
        None
    }

    /// Detect the model kind for the weights at `paths`.
    ///
    /// Strategy (in priority order):
    /// 1. Read `model_type` from `config.json` — most reliable.
    /// 2. Pattern-match the model ID / local path string — fallback when the
    ///    config is absent or has an unrecognised `model_type`.
    pub fn detect(paths: &crate::weights::ModelPaths, source: &ModelSource) -> Result<Self> {
        let config_path = paths.config_filename();
        if let Ok(content) = std::fs::read_to_string(config_path)
            && let Ok(raw) = serde_json::from_str::<serde_json::Value>(&content)
            && let Some(mt) = raw["model_type"].as_str()
        {
            if let Some(kind) = Self::from_config_type(mt) {
                debug!(model_type = mt, kind = ?kind, "Detected model kind from config.json");
                return Ok(kind);
            }
            trace!(
                model_type = mt,
                "config.json model_type not recognised; falling back to name hint"
            );
        }

        let hint = match source {
            ModelSource::HuggingFace(id) => id.as_str(),
            ModelSource::LocalPath(p) => p.as_str(),
        };
        Self::from_name_hint(hint).ok_or_else(|| {
            TurError::HfHub(format!(
                "Cannot determine model architecture for '{hint}'. \
                 Ensure config.json contains a recognised 'model_type', \
                 or use ModelFactory<T> with an explicit model type."
            ))
        })
    }
}

// ─── AnyModelConfig ───────────────────────────────────────────────────────────

/// Typed configuration for every architecture supported by [`AutoModelFactory`].
/// One variant per [`ModelKind`].
pub enum AnyModelConfig {
    Qwen3(crate::models::qwen3::Config),
    Granite41(crate::models::granite_41::Config),
}

// ─── AnyModel ─────────────────────────────────────────────────────────────────

/// A sum type over all model architectures supported by [`AutoModelFactory`].
///
/// Implements [`ModelImpl`] by delegating every method to the inner variant,
/// so the rest of the inference stack (`InferenceEngine`, `TurPipeline`, …) is
/// architecture-agnostic at runtime while remaining zero-cost after
/// monomorphisation.
///
/// To add a new model:
/// 1. Add a variant here and in [`AnyModelConfig`] / [`ModelKind`].
/// 2. Add matching arms to every `match self` block below.
/// 3. Register the new `model_type` string(s) in `ModelKind::from_config_type`.
pub enum AnyModel {
    Qwen3(crate::models::Qwen35ModelForCausalLM),
    Granite41(crate::models::Granite41ModelForCausalLM),
}

impl ModelImpl for AnyModel {
    fn name(&self) -> &'static str {
        match self {
            Self::Qwen3(m) => m.name(),
            Self::Granite41(m) => m.name(),
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.num_layers(),
            Self::Granite41(m) => m.num_layers(),
        }
    }

    fn num_kv_heads(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.num_kv_heads(),
            Self::Granite41(m) => m.num_kv_heads(),
        }
    }

    fn head_dim(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.head_dim(),
            Self::Granite41(m) => m.head_dim(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Qwen3(m) => m.dtype(),
            Self::Granite41(m) => m.dtype(),
        }
    }

    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward(input, offset),
            Self::Granite41(m) => m.forward(input, offset),
        }
    }

    fn forward_modal(&mut self, input: ModelInput) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward_modal(input),
            Self::Granite41(m) => m.forward_modal(input),
        }
    }

    fn forward_batch(
        &mut self,
        input: &Tensor,
        positions: &[usize],
        paged_caches: Option<&mut [Vec<PagedKvCache>]>,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward_batch(input, positions, paged_caches),
            Self::Granite41(m) => m.forward_batch(input, positions, paged_caches),
        }
    }

    fn format_prompt(&self, prompt: &str, thinking: bool) -> String {
        match self {
            Self::Qwen3(m) => m.format_prompt(prompt, thinking),
            Self::Granite41(m) => m.format_prompt(prompt, thinking),
        }
    }

    fn format_prompt_with_tools(
        &self,
        prompt: &str,
        tools: &[crate::backend::tools::ToolDefinition],
        thinking: bool,
    ) -> String {
        match self {
            Self::Qwen3(m) => m.format_prompt_with_tools(prompt, tools, thinking),
            Self::Granite41(m) => m.format_prompt_with_tools(prompt, tools, thinking),
        }
    }

    fn get_kv_cache_state(&self) -> candle_core::Result<Vec<(Tensor, Tensor)>> {
        match self {
            Self::Qwen3(m) => m.get_kv_cache_state(),
            Self::Granite41(m) => m.get_kv_cache_state(),
        }
    }

    fn set_kv_cache_state(&mut self, state: Vec<(Tensor, Tensor)>) -> candle_core::Result<()> {
        match self {
            Self::Qwen3(m) => m.set_kv_cache_state(state),
            Self::Granite41(m) => m.set_kv_cache_state(state),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Self::Qwen3(m) => m.clear_kv_cache(),
            Self::Granite41(m) => m.clear_kv_cache(),
        }
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        match self {
            Self::Qwen3(m) => m.eos_token_ids(),
            Self::Granite41(m) => m.eos_token_ids(),
        }
    }
}

// ─── ModelConstructor for AnyModel ────────────────────────────────────────────

impl ModelConstructor for AnyModel {
    type Config = AnyModelConfig;

    fn load_config(paths: &crate::weights::ModelPaths) -> Result<Self::Config> {
        let config_path = paths.config_filename();
        let content = std::fs::read_to_string(config_path)?;

        // Parse once into a Value so we can both detect `model_type` and
        // deserialise into the typed struct without a second file read.
        let raw: serde_json::Value = serde_json::from_str(&content)?;

        let model_type = raw["model_type"].as_str().ok_or_else(|| {
            TurError::HfHub("config.json is missing the 'model_type' field".to_string())
        })?;

        let kind = ModelKind::from_config_type(model_type).ok_or_else(|| {
            TurError::HfHub(format!(
                "Unsupported model_type '{model_type}' in config.json. \
                 Add a ModelKind variant and the matching AnyModel arm to add support."
            ))
        })?;

        debug!(model_type, kind = ?kind, "AnyModel config loaded");

        match kind {
            ModelKind::Qwen3 => {
                let cfg: crate::models::qwen3::Config = serde_json::from_value(raw)?;
                Ok(AnyModelConfig::Qwen3(cfg))
            }
            ModelKind::Granite41 => {
                let cfg: crate::models::granite_41::Config = serde_json::from_value(raw)?;
                Ok(AnyModelConfig::Granite41(cfg))
            }
        }
    }

    fn new_with_progress(
        config: &Self::Config,
        vb: VarBuilderX,
        progress: Option<&ProgressReporter>,
    ) -> Result<Self> {
        match config {
            AnyModelConfig::Qwen3(cfg) => {
                let model =
                    crate::models::Qwen35ModelForCausalLM::new_with_progress(cfg, vb, progress)
                        .map_err(TurError::from)?;
                Ok(AnyModel::Qwen3(model))
            }
            AnyModelConfig::Granite41(cfg) => {
                let model =
                    crate::models::Granite41ModelForCausalLM::new_with_progress(cfg, vb, progress)
                        .map_err(TurError::from)?;
                Ok(AnyModel::Granite41(model))
            }
        }
    }
}

// ─── AutoModelFactory ─────────────────────────────────────────────────────────

/// A [`ModelFactory`] that automatically detects the model architecture from
/// `config.json` and constructs the appropriate [`AnyModel`] variant.
///
/// Use this when the model type is not known at compile time, e.g. when loading
/// arbitrary HuggingFace checkpoints.  For a known architecture, prefer the
/// typed `ModelFactory<T>` to keep the concrete type visible throughout.
///
/// # Example
/// ```no_run
/// # use candle_core::{DType, Device};
/// # use tur::backend::factory::{AutoModelFactory, ModelSource};
/// # use tur::backend::pipeline::TurPipeline;
/// let factory = AutoModelFactory::new(
///     ModelSource::HuggingFace("Qwen/Qwen3-0.6B".to_string()),
///     Some("Q4_K_M".to_string()),
///     Device::Cpu,
///     DType::BF16,
/// );
/// let pipeline = TurPipeline::builder(&factory, Device::Cpu).build();
/// ```
pub type AutoModelFactory = ModelFactory<AnyModel>;

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelFactory construction ────────────────────────────────────────────

    #[test]
    fn typed_factory_hf_source() {
        let factory = ModelFactory::<crate::models::Qwen35ModelForCausalLM>::new(
            ModelSource::HuggingFace("Qwen3-0.6B".to_string()),
            Some("Q4_K_M".to_string()),
            Device::Cpu,
            DType::F32,
        );
        assert!(factory.device().is_cpu());
    }

    #[test]
    fn typed_factory_local_source() {
        let factory = ModelFactory::<crate::models::Qwen35ModelForCausalLM>::new(
            ModelSource::LocalPath("/tmp/model".to_string()),
            None,
            Device::Cpu,
            DType::BF16,
        );
        assert!(factory.device().is_cpu());
    }

    // ── ModelKind::from_config_type ──────────────────────────────────────────

    #[test]
    fn config_type_qwen3_maps_to_qwen3() {
        assert_eq!(ModelKind::from_config_type("qwen3"), Some(ModelKind::Qwen3));
    }

    #[test]
    fn config_type_qwen2_maps_to_qwen3() {
        // Qwen2 and Qwen3 share the same architecture in this codebase.
        assert_eq!(ModelKind::from_config_type("qwen2"), Some(ModelKind::Qwen3));
    }

    #[test]
    fn config_type_matching_is_case_sensitive() {
        // HuggingFace config.json always uses lowercase; upper-case must not match.
        assert_eq!(ModelKind::from_config_type("Qwen3"), None);
        assert_eq!(ModelKind::from_config_type("QWEN3"), None);
        assert_eq!(ModelKind::from_config_type("Qwen2"), None);
    }

    #[test]
    fn config_type_unknown_returns_none() {
        for unknown in &["llama", "mistral", "phi3", "gemma", "falcon", ""] {
            assert_eq!(
                ModelKind::from_config_type(unknown),
                None,
                "expected None for model_type = '{unknown}'"
            );
        }
    }

    // ── ModelKind::from_name_hint ────────────────────────────────────────────

    #[test]
    fn name_hint_qwen3_full_hf_id() {
        assert_eq!(
            ModelKind::from_name_hint("Qwen/Qwen3-0.6B"),
            Some(ModelKind::Qwen3)
        );
    }

    #[test]
    fn name_hint_qwen3_shorthand() {
        assert_eq!(
            ModelKind::from_name_hint("Qwen3-0.6B"),
            Some(ModelKind::Qwen3)
        );
    }

    #[test]
    fn name_hint_qwen3_is_case_insensitive() {
        for id in &["QWEN3-72B", "qwen3-instruct", "QWen3-4B-Instruct"] {
            assert_eq!(
                ModelKind::from_name_hint(id),
                Some(ModelKind::Qwen3),
                "expected Qwen3 for id = '{id}'"
            );
        }
    }

    #[test]
    fn name_hint_qwen2_family_maps_to_qwen3() {
        for id in &[
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen2-72B",
            "qwen2-0.5b",
            "QWEN2.5-CODER",
        ] {
            assert_eq!(
                ModelKind::from_name_hint(id),
                Some(ModelKind::Qwen3),
                "expected Qwen3 for id = '{id}'"
            );
        }
    }

    #[test]
    fn name_hint_unknown_models_return_none() {
        for id in &[
            "meta-llama/Llama-3.1-8B",
            "mistralai/Mistral-7B-v0.1",
            "microsoft/phi-3-mini",
            "google/gemma-7b",
            "tiiuae/falcon-7b",
            "",
        ] {
            assert_eq!(
                ModelKind::from_name_hint(id),
                None,
                "expected None for id = '{id}'"
            );
        }
    }

    // ── AutoModelFactory ─────────────────────────────────────────────────────

    #[test]
    fn auto_factory_is_model_factory_any_model() {
        // Compile-time assertion: AutoModelFactory is ModelFactory<AnyModel>.
        // If the type alias changes, this assignment will fail to compile.
        let factory: AutoModelFactory = AutoModelFactory::new(
            ModelSource::HuggingFace("Qwen/Qwen3-0.6B".to_string()),
            Some("Q4_K_M".to_string()),
            Device::Cpu,
            DType::BF16,
        );
        assert!(factory.device().is_cpu());
    }

    #[test]
    fn auto_factory_local_path() {
        let factory = AutoModelFactory::new(
            ModelSource::LocalPath("/tmp/model".to_string()),
            None,
            Device::Cpu,
            DType::F32,
        );
        assert!(factory.device().is_cpu());
    }
}
