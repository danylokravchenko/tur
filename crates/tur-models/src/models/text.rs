//! Architecture dispatch at loading and execution boundaries.
use super::{
    GenerationDefaults, LoadOptions,
    gemma4::{self, Gemma4},
    qwen3_5::{self, Qwen35},
};
use crate::hybrid::{DraftForward, HybridModel, PackedForward, SequenceStep};
use std::path::Path;
use tur_core::{
    Result,
    execution::{ModelDescription, invalid},
};

/// Validated model-specific settings selected from explicit checkpoint architecture fields.
#[derive(Clone, Debug)]
pub enum ModelConfig {
    Qwen35(qwen3_5::Config),
    Gemma4(gemma4::Config),
}
impl ModelConfig {
    /// Parse the architecture before any weight download or allocation.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        match raw["model_type"].as_str() {
            Some("gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text") => {
                Ok(Self::Gemma4(gemma4::Config::from_json(json)?))
            }
            _ => Ok(Self::Qwen35(qwen3_5::Config::from_json(json)?)),
        }
    }
    /// Exclusive text position limit.
    pub fn max_position(&self) -> usize {
        match self {
            Self::Qwen35(c) => c.max_position(),
            Self::Gemma4(c) => c.max_position(),
        }
    }
    /// Output vocabulary width.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen35(c) => c.vocab_size(),
            Self::Gemma4(c) => c.vocab_size(),
        }
    }
    /// Merge generation metadata with architecture EOS tokens.
    pub fn generation_defaults(&self, json: Option<&str>) -> Result<GenerationDefaults> {
        match self {
            Self::Qwen35(c) => c.generation_defaults(json),
            Self::Gemma4(c) => c.generation_defaults(json),
        }
    }
    pub(crate) fn with_mtp(self) -> Result<Self> {
        match self {
            Self::Qwen35(c) => Ok(Self::Qwen35(c.with_mtp()?)),
            Self::Gemma4(_) => Err(invalid("Gemma 4 native draft heads are not implemented")),
        }
    }
    /// Reserve the selected model's exact weights and bounded scratch before loading.
    pub fn memory_requirements(
        &self,
        options: &LoadOptions,
        tokens: usize,
        contexts: &[usize],
    ) -> Result<(ModelDescription, usize, usize)> {
        match self {
            Self::Qwen35(c) => Qwen35::memory_requirements(c, options, tokens, contexts),
            Self::Gemma4(c) => Gemma4::memory_requirements(c, options, tokens, contexts),
        }
    }
    pub(crate) fn load(&self, dir: &Path, options: &LoadOptions) -> Result<TextModel> {
        match self {
            Self::Qwen35(_) => Ok(TextModel::Qwen35(Box::new(Qwen35::load(dir, options)?))),
            Self::Gemma4(_) => Ok(TextModel::Gemma4(Box::new(Gemma4::load(dir, options)?))),
        }
    }
}

/// One immutable text model, driven by the same engine, state pools and scheduler.
pub enum TextModel {
    Qwen35(Box<Qwen35>),
    Gemma4(Box<Gemma4>),
}
impl TextModel {
    /// Device bytes held by loaded weights.
    pub fn weight_bytes(&self) -> usize {
        match self {
            Self::Qwen35(m) => m.weight_bytes(),
            Self::Gemma4(m) => m.weight_bytes(),
        }
    }
    /// Exclusive text position limit.
    pub fn max_position(&self) -> usize {
        match self {
            Self::Qwen35(m) => m.config().max_position(),
            Self::Gemma4(m) => m.config().max_position(),
        }
    }
    fn model(&self) -> &dyn HybridModel {
        match self {
            Self::Qwen35(m) => m.as_ref(),
            Self::Gemma4(m) => m.as_ref(),
        }
    }
}
impl HybridModel for TextModel {
    fn description(&self) -> ModelDescription {
        self.model().description()
    }
    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> {
        self.model().workspace_bytes(batch)
    }
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>> {
        self.model().forward(batch)
    }
    fn draft(&self, batch: &DraftForward<'_>) -> Result<Vec<Result<Option<candle_core::Tensor>>>> {
        self.model().draft(batch)
    }
    #[cfg(feature = "metal")]
    fn kernels(&self) -> Option<tur_metal::kernels::Kernels> {
        self.model().kernels()
    }
}
