//! Architecture dispatch at loading and execution boundaries.
use super::{
    GenerationDefaults, LoadOptions,
    gemma4::{self, Gemma4},
    gpt_oss::{self, GptOss},
    qwen3_5::{self, Qwen35},
};
use crate::{
    backend::tools::ToolSyntax,
    hybrid::{DraftForward, HybridModel, LayerCarry, LayerStep, PackedForward, SequenceStep},
    media::{Media, MediaLayout, MediaPrompt},
};
use std::path::Path;
use tur_core::{
    Result,
    execution::{ModelDescription, invalid},
};

fn no_audio() -> tur_core::TurError {
    invalid("this model has no audio encoder")
}

fn no_vision() -> tur_core::TurError {
    invalid("this model has no vision encoder")
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
    pub fn audio_prompt(&self) -> Option<MediaPrompt> {
        match self {
            Self::Qwen35(_) | Self::GptOss(_) => None,
            Self::Gemma4(m) => m.audio_prompt(),
        }
    }
    /// How an image appears in prompts, when the model has a vision encoder loaded.
    pub fn image_prompt(&self) -> Option<MediaPrompt> {
        match self {
            Self::Qwen35(m) => m.image_prompt(),
            Self::GptOss(_) => None,
            Self::Gemma4(m) => m.image_prompt(),
        }
    }
    /// How a video appears in prompts, when the model has a vision encoder loaded.
    pub fn video_prompt(&self) -> Option<MediaPrompt> {
        match self {
            Self::Qwen35(m) => m.video_prompt(),
            Self::GptOss(_) => None,
            Self::Gemma4(m) => m.video_prompt(),
        }
    }
    /// How `media` appears in prompts, when this model's encoder for it is loaded.
    pub fn media_prompt(&self, media: Media<'_>) -> Option<MediaPrompt> {
        match media {
            Media::Audio(_) => self.audio_prompt(),
            Media::Image(_) => self.image_prompt(),
            Media::Video(_) => self.video_prompt(),
        }
    }
    /// The token a soft-token position feeds the model, under its embedded row.
    pub fn media_slot(&self) -> Option<u32> {
        match self {
            Self::Qwen35(m) => m.image_prompt().map(|prompt| prompt.placeholder),
            Self::GptOss(_) => None,
            Self::Gemma4(m) => m.media_slot(),
        }
    }
    /// The runs of soft tokens an input occupies, and what the prompt holds around each.
    pub fn media_layout(&self, media: Media<'_>) -> Result<MediaLayout> {
        match (self, media) {
            (Self::Gemma4(m), Media::Audio(clip)) => {
                Ok(MediaLayout::single(m.audio_tokens(clip)?, None))
            }
            (Self::Gemma4(m), Media::Image(image)) => {
                Ok(MediaLayout::single(m.image_tokens(image)?, None))
            }
            (Self::Gemma4(m), Media::Video(video)) => m.video_layout(video),
            (Self::Qwen35(m), Media::Image(image)) => m.image_layout(image),
            (Self::Qwen35(m), Media::Video(video)) => m.video_layout(video),
            (_, Media::Audio(_)) => Err(no_audio()),
            (_, Media::Image(_) | Media::Video(_)) => Err(no_vision()),
        }
    }
    /// F32 `[soft tokens, hidden]` embedding rows of an input on the model device, its
    /// segments in order. Rows encoded recently are returned without encoding again.
    pub fn encode_media(&self, media: Media<'_>) -> Result<candle_core::Tensor> {
        let cache = match self {
            Self::Qwen35(m) => Some(m.media_cache()),
            Self::Gemma4(m) => Some(m.media_cache()),
            Self::GptOss(_) => None,
        };
        let key = media.digest();
        if let Some(rows) = cache.and_then(|cache| cache.get(&key)) {
            return Ok(rows);
        }
        let rows = match (self, media) {
            (Self::Gemma4(m), Media::Audio(clip)) => m.encode_audio(clip)?,
            (Self::Gemma4(m), Media::Image(image)) => m.encode_image(image)?,
            (Self::Gemma4(m), Media::Video(video)) => m.encode_video(video)?,
            (Self::Qwen35(m), Media::Image(image)) => m.encode_image(image)?,
            (Self::Qwen35(m), Media::Video(video)) => m.encode_video(video)?,
            (_, Media::Audio(_)) => return Err(no_audio()),
            (_, Media::Image(_) | Media::Video(_)) => return Err(no_vision()),
        };
        if let Some(cache) = cache {
            cache.insert(key, &rows);
        }
        Ok(rows)
    }
    /// Whether the model reads any non-text input, so requests may need encoding.
    pub fn has_encoders(&self) -> bool {
        self.audio_prompt().is_some() || self.image_prompt().is_some()
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
    fn pausable_layers(&self) -> Option<usize> {
        self.model().pausable_layers()
    }
    fn forward_layers(
        &self,
        batch: &PackedForward<'_>,
        carry: Option<LayerCarry>,
        end: usize,
    ) -> Result<LayerStep> {
        self.model().forward_layers(batch, carry, end)
    }
    fn draft(&self, batch: &DraftForward<'_>) -> Result<Vec<Result<Option<candle_core::Tensor>>>> {
        self.model().draft(batch)
    }
    #[cfg(feature = "metal")]
    fn kernels(&self) -> Option<tur_metal::kernels::Kernels> {
        self.model().kernels()
    }
}
