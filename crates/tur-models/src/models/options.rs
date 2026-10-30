//! Placement and execution settings shared by text models.
use super::{
    artifact::Manifest,
    weights::{WeightLayout, WeightPlan},
};
use candle_core::{DType, Device};
use std::sync::Arc;
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout},
};

/// Storage for projection matrices. Norms, gates, convolution and state stay F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    /// Full-precision reference.
    F32,
    /// Groupwise 8-bit (32-element blocks, one scale).
    Q8_0,
    /// Groupwise 4-bit K-quant (256-element super-blocks).
    Q4K,
}

impl std::str::FromStr for WeightFormat {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "q8_0" => Ok(Self::Q8_0),
            "q4k" => Ok(Self::Q4K),
            _ => Err("weight format must be f32, q8_0 or q4k (SafeTensors conversion)".into()),
        }
    }
}

impl WeightFormat {
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4k",
        }
    }
}

/// Where projection weights come from and how they are stored.
#[derive(Clone, Debug)]
pub enum Weights {
    /// Convert `SafeTensors` at load time, every projection in one format.
    Uniform(WeightFormat),
    /// Convert `SafeTensors` at load time by an explicit plan, rounding to nearest.
    Plan(Arc<WeightPlan>),
    /// Upload the directory's prepacked artifact as stored. The manifest, read when the
    /// load was planned, must still describe the artifact.
    Prepacked(Arc<Manifest>),
}

impl From<WeightFormat> for Weights {
    fn from(format: WeightFormat) -> Self {
        Self::Uniform(format)
    }
}

impl Weights {
    /// Storage precision of every projection and the embedding.
    pub fn plan(&self, config: &impl WeightLayout) -> Result<WeightPlan> {
        match self {
            Self::Uniform(format) => WeightPlan::uniform(config, *format),
            Self::Plan(plan) => Ok(plan.as_ref().clone()),
            Self::Prepacked(manifest) => Ok(manifest.plan.clone()),
        }
    }

    /// Weight identity appended to the checkpoint revision.
    pub(crate) fn tag(&self) -> String {
        match self {
            Self::Uniform(format) => format.tag().to_owned(),
            Self::Plan(plan) => plan.tag(),
            Self::Prepacked(manifest) => manifest.identity(),
        }
    }
}

/// Which implementation executes the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Execution {
    /// Fused Metal kernels on a Metal device; the reference on the CPU, which has no
    /// kernels. A Metal model never falls back to the CPU.
    #[default]
    Fused,
    /// F32 reference operations over Candle tensors: the numerical oracle the fused
    /// kernels are tested against, on any device.
    Reference,
}

/// K/V storage chosen from M1 measurements (`docs/src/architecture/memory.md`): F16 was the
/// fastest format end to end and indistinguishable from F32 in quality; page sizes of
/// 16–128 tokens performed alike, and 16 wastes the least tail memory. Q8 halves K/V
/// memory again for capacity-bound deployments.
pub const DEFAULT_KV_LAYOUT: KvLayout = KvLayout {
    page_tokens: 16,
    format: KvFormat::F16,
    version: tur_core::execution::KV_LAYOUT_VERSION,
};

/// Recurrent matrix storage. F32 until a quality evaluation justifies a narrower format.
pub const DEFAULT_RECURRENT_STATE: DType = DType::F32;

/// Parse a recurrent state storage name: `f32`, `f16` or `bf16`.
pub fn parse_recurrent_state(name: &str) -> std::result::Result<DType, String> {
    match name {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        _ => Err(format!(
            "recurrent state must be f32, f16 or bf16, not '{name}'"
        )),
    }
}

/// How to place a checkpoint.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Exact checkpoint identity (repository and commit). The weight identity is appended.
    pub revision: String,
    pub weights: Weights,
    pub device: Device,
    pub execution: Execution,
    /// Paged K/V page size and storage format; part of the state identity.
    pub kv: KvLayout,
    /// Storage of recurrent matrix state (F32, F16 or BF16); part of the state identity.
    pub recurrent: DType,
    /// Load the checkpoint's native multi-token prediction head ([`super::qwen3_5::Config::with_mtp`]).
    pub mtp: Option<DraftHead>,
    /// Load the checkpoint's audio encoder, when it declares one. Prepacked artifacts
    /// hold the text tower only and load without it.
    pub audio: bool,
}

/// A native multi-token prediction drafter, loaded for drafting: Qwen3.5's head inside
/// the checkpoint, or a Gemma 4 assistant checkpoint beside it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftHead {
    /// Draft only token IDs below this bound, using a copy of the LM head's leading
    /// rows. Byte-level BPE assigns IDs in merge order, so low IDs are frequent tokens.
    /// `None` uses the whole LM head. Qwen3.5 only.
    pub vocab: Option<usize>,
    /// Local directory of a separate drafter checkpoint (a Gemma 4 assistant); `None`
    /// loads the head inside the target checkpoint.
    pub model: Option<std::path::PathBuf>,
}

impl DraftHead {
    /// Whether the head chains several drafts per pass. A Gemma 4 assistant keeps no state
    /// and redrafts from its own output; Qwen3.5's head writes its own keys and values and
    /// drafts one token.
    pub fn chains(&self) -> bool {
        self.model.is_some()
    }
}
