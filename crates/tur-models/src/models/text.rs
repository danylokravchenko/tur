//! Architecture dispatch at loading and execution boundaries.
use super::{
    GenerationDefaults, LoadOptions,
    gemma4::{self, Gemma4},
    gpt_oss::{self, GptOss},
    qwen3_5::{self, Qwen35},
};
use crate::{
    audio::{Audio, AudioPrompt},
    backend::tools::ToolSyntax,
    hybrid::{DraftForward, HybridModel, PackedForward, SequenceStep},
};
use std::path::Path;
use tur_core::{
    Result,
    execution::{ModelDescription, invalid},
};

fn no_audio() -> tur_core::TurError {
    invalid("this model has no audio encoder")
}

/// Validated model-specific settings selected from explicit checkpoint architecture fields.
#[derive(Clone, Debug)]
pub enum ModelConfig {
    Qwen35(qwen3_5::Config),
    Gemma4(gemma4::Config),
    GptOss(gpt_oss::Config),
}
impl ModelConfig {
    /// Parse the architecture before any weight download or allocation.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        match raw["model_type"].as_str() {
            Some("gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text") => {
                Ok(Self::Gemma4(gemma4::Config::from_json(json)?))
            }
            Some("gpt_oss") => Ok(Self::GptOss(gpt_oss::Config::from_json(json)?)),
            _ => Ok(Self::Qwen35(qwen3_5::Config::from_json(json)?)),
        }
    }
    /// Exclusive text position limit.
    pub fn max_position(&self) -> usize {
        match self {
            Self::Qwen35(c) => c.max_position(),
            Self::Gemma4(c) => c.max_position(),
            Self::GptOss(c) => c.max_position(),
        }
    }
    /// Output vocabulary width.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen35(c) => c.vocab_size(),
            Self::Gemma4(c) => c.vocab_size(),
            Self::GptOss(c) => c.vocab_size(),
        }
    }
    /// Merge generation metadata with architecture EOS tokens.
    pub fn generation_defaults(&self, json: Option<&str>) -> Result<GenerationDefaults> {
        match self {
            Self::Qwen35(c) => c.generation_defaults(json),
            Self::Gemma4(c) => c.generation_defaults(json),
            Self::GptOss(c) => c.generation_defaults(json),
        }
    }
    pub(crate) fn with_mtp(self) -> Result<Self> {
        match self {
            Self::Qwen35(c) => Ok(Self::Qwen35(c.with_mtp()?)),
            // The assistant is a separate checkpoint, loaded from the draft head's directory.
            Self::Gemma4(c) => Ok(Self::Gemma4(c)),
            Self::GptOss(_) => Err(invalid("gpt-oss has no draft head")),
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
            Self::GptOss(c) => GptOss::memory_requirements(c, options, tokens, contexts),
        }
    }
    pub(crate) fn load(&self, dir: &Path, options: &LoadOptions) -> Result<TextModel> {
        match self {
            Self::Qwen35(_) => Ok(TextModel::Qwen35(Box::new(Qwen35::load(dir, options)?))),
            Self::Gemma4(_) => Ok(TextModel::Gemma4(Box::new(Gemma4::load(dir, options)?))),
            Self::GptOss(_) => Ok(TextModel::GptOss(Box::new(GptOss::load(dir, options)?))),
        }
    }
}

/// One immutable text model, driven by the same engine, state pools and scheduler.
pub enum TextModel {
    Qwen35(Box<Qwen35>),
    Gemma4(Box<Gemma4>),
    GptOss(Box<GptOss>),
}
impl TextModel {
    /// Load a validated architecture from a local checkpoint using shared placement settings.
    pub fn load(dir: &Path, options: &LoadOptions) -> Result<Self> {
        ModelConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?
            .load(dir, options)
    }
    /// Device holding immutable weights and executing model passes.
    pub fn device(&self) -> &candle_core::Device {
        match self {
            Self::Qwen35(m) => m.device(),
            Self::Gemma4(m) => m.device(),
            Self::GptOss(m) => m.device(),
        }
    }
    /// Resident bytes of projection matrices, excluding embeddings and auxiliary tensors.
    pub fn projection_bytes(&self) -> usize {
        match self {
            Self::Qwen35(m) => m.projection_bytes(),
            Self::Gemma4(m) => m.projection_bytes(),
            Self::GptOss(m) => m.projection_bytes(),
        }
    }
    /// Switch between the common Metal path and the numerical reference on the same weights.
    pub fn with_execution(self, execution: super::Execution) -> Result<Self> {
        match self {
            Self::Qwen35(m) => Ok(Self::Qwen35(Box::new((*m).with_execution(execution)?))),
            Self::Gemma4(m) => Ok(Self::Gemma4(Box::new((*m).with_execution(execution)?))),
            Self::GptOss(m) => Ok(Self::GptOss(Box::new((*m).with_execution(execution)?))),
        }
    }
    /// Device bytes held by loaded weights.
    pub fn weight_bytes(&self) -> usize {
        match self {
            Self::Qwen35(m) => m.weight_bytes(),
            Self::Gemma4(m) => m.weight_bytes(),
            Self::GptOss(m) => m.weight_bytes(),
        }
    }
    /// Exclusive text position limit.
    pub fn max_position(&self) -> usize {
        match self {
            Self::Qwen35(m) => m.config().max_position(),
            Self::Gemma4(m) => m.config().max_position(),
            Self::GptOss(m) => m.config().max_position(),
        }
    }
    /// How this model writes tool calls. [`ToolSyntax::Harmony`] models write all of their
    /// output in Harmony channels.
    pub fn tool_syntax(&self) -> ToolSyntax {
        match self {
            Self::Qwen35(_) => ToolSyntax::Xml,
            Self::Gemma4(_) => ToolSyntax::Gemma,
            Self::GptOss(_) => ToolSyntax::Harmony,
        }
    }
    /// How a clip appears in prompts, when the model has an audio encoder loaded.
    pub fn audio_prompt(&self) -> Option<AudioPrompt> {
        match self {
            Self::Qwen35(_) | Self::GptOss(_) => None,
            Self::Gemma4(m) => m.audio_prompt(),
        }
    }
    /// Prompt positions a clip's soft tokens occupy, between its framing tokens.
    pub fn audio_tokens(&self, clip: &Audio) -> Result<usize> {
        match self {
            Self::Qwen35(_) | Self::GptOss(_) => Err(no_audio()),
            Self::Gemma4(m) => m.audio_tokens(clip),
        }
    }
    /// Encode a clip into F32 `[audio_tokens, hidden]` embedding rows on the model device.
    pub fn encode_audio(&self, clip: &Audio) -> Result<candle_core::Tensor> {
        match self {
            Self::Qwen35(_) | Self::GptOss(_) => Err(no_audio()),
            Self::Gemma4(m) => m.encode_audio(clip),
        }
    }
    fn model(&self) -> &dyn HybridModel {
        match self {
            Self::Qwen35(m) => m.as_ref(),
            Self::Gemma4(m) => m.as_ref(),
            Self::GptOss(m) => m.as_ref(),
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
