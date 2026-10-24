#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Packed Qwen3.5/3.8 text execution over explicit hybrid state.

use super::{
    artifact::{self, Manifest},
    calibration::{Calibration, Recorder},
    config::{Config, LayerKind, Projection},
    math::{self, DeltaInputs},
    plan::{self, Embedding, Precision, WeightPlan},
    quantize::{Packer, STAGING_BYTES},
    weights::Checkpoint,
};
use crate::hybrid::{
    Addresses, Boundaries, DraftForward, DraftSequence, HybridModel, PackedForward,
    SequenceForward, SequenceStep, StatePool,
};
use candle_core::{DType, Device, Tensor};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, ModelDescription, checked_elements, invalid},
};
use tur_metal::{packed::PackedLinear, pool::PoolConfig, state::ModelState};

#[cfg(feature = "metal")]
mod fused;
mod moe;

/// Every projection of a model in [`Config::projections`] order, borrowed shared or (with
/// `mut`) mutably: one traversal serves both.
macro_rules! projections {
    ($model:expr $(, $m:tt)?) => {{
        let model = $model;
        let mut out = Vec::new();
        let head = & $($m)? model.lm_head;
        for layer in & $($m)? model.layers {
            projections!(@layer out, layer $(, $m)?);
        }
        if let Some(mtp) = & $($m)? model.mtp {
            out.push(& $($m)? mtp.fc);
            projections!(@layer out, & $($m)? mtp.layer $(, $m)?);
        }
        out.push(head);
        out
    }};
    (@layer $out:ident, $layer:expr $(, $m:tt)?) => {{
        let layer = $layer;
        match & $($m)? layer.mixer {
            Mixer::Attention(a) => {
                $out.push(& $($m)? a.q);
                $out.push(& $($m)? a.kv);
                $out.push(& $($m)? a.o);
            }
            Mixer::Delta(net) => {
                $out.push(& $($m)? net.proj);
                $out.push(& $($m)? net.out);
            }
        }
        match & $($m)? layer.mlp {
            Mlp::Dense(mlp) => {
                $out.push(& $($m)? mlp.gate_up);
                $out.push(& $($m)? mlp.down);
            }
            Mlp::Moe(moe) => {
                $out.push(& $($m)? moe.gate_up);
                $out.push(& $($m)? moe.down);
                $out.push(& $($m)? moe.shared.gate_up);
                $out.push(& $($m)? moe.shared.down);
            }
        }
    }};
}

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
    fn tag(self) -> &'static str {
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
    pub fn plan(&self, config: &Config) -> Result<WeightPlan> {
        match self {
            Self::Uniform(format) => WeightPlan::uniform(config, *format),
            Self::Plan(plan) => Ok(plan.as_ref().clone()),
            Self::Prepacked(manifest) => Ok(manifest.plan.clone()),
        }
    }

    /// Weight identity appended to the checkpoint revision.
    fn tag(&self) -> String {
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

/// K/V storage chosen from M1 measurements (`docs/decisions.md`): F16 was the
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
    /// Load the checkpoint's native multi-token prediction head ([`Config::with_mtp`]).
    pub mtp: bool,
    /// Draft only token IDs below this bound: the head projects onto a copy of the LM
    /// head's leading rows instead of the whole vocabulary. Byte-level BPE assigns IDs in
    /// merge order, so low IDs are the frequent tokens. `None` uses the whole LM head.
    pub draft_vocab: Option<usize>,
}

struct Attention {
    q: PackedLinear,
    kv: PackedLinear,
    o: PackedLinear,
    q_norm: Tensor,
    k_norm: Tensor,
}

struct DeltaNet {
    /// Fused `in_proj_qkv | in_proj_z | in_proj_b | in_proj_a`.
    proj: PackedLinear,
    conv: Tensor,
    neg_a: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out: PackedLinear,
}

enum Mixer {
    Attention(Attention),
    Delta(DeltaNet),
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    mlp: Mlp,
}

struct DenseMlp {
    /// Fused `gate_proj | up_proj`.
    gate_up: PackedLinear,
    down: PackedLinear,
}

enum Mlp {
    Dense(DenseMlp),
    Moe(Box<moe::Moe>),
}

impl DenseMlp {
    fn load(tensors: &Tensors, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_up: tensors.projection(&format!("{prefix}.gate_up_proj"))?,
            down: tensors.projection(&format!("{prefix}.down_proj"))?,
        })
    }

    fn bytes(&self) -> usize {
        self.gate_up.resident_bytes() + self.down.resident_bytes()
    }

    fn forward(&self, h: &Tensor, recorder: Option<&Recorder>) -> Result<Tensor> {
        let i = self.down.in_features();
        if let Some(recorder) = recorder {
            recorder.record(&self.gate_up, h)?;
        }
        let gate_up = self.gate_up.forward(h)?;
        let act = (candle_nn::ops::silu(&gate_up.narrow(1, 0, i)?)? * gate_up.narrow(1, i, i)?)?;
        if let Some(recorder) = recorder {
            recorder.record(&self.down, &act)?;
        }
        self.down.forward(&act)
    }
}

#[derive(Clone, Copy)]
struct Dims {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    key_width: usize,
    value_width: usize,
    channels: usize,
    #[cfg(feature = "metal")]
    intermediate: usize,
    eps: f64,
    rotary_dim: usize,
    theta: f64,
}

impl Dims {
    fn new(config: &Config) -> Result<Self> {
        let t = &config.text;
        Ok(Self {
            heads: t.num_attention_heads,
            kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            key_heads: t.linear_num_key_heads,
            value_heads: t.linear_num_value_heads,
            key_dim: t.linear_key_head_dim,
            value_dim: t.linear_value_head_dim,
            key_width: config.key_width()?,
            value_width: config.value_width()?,
            channels: config.conv_channels()?,
            #[cfg(feature = "metal")]
            intermediate: t.intermediate_size,
            eps: t.rms_norm_eps,
            rotary_dim: config.rotary_dim(),
            theta: t.rope_parameters.rope_theta,
        })
    }
}

/// Qwen3.5 dense or expert hybrid decoder (text tower only). Weights stay immutable.
pub struct Qwen35 {
    config: Config,
    description: ModelDescription,
    device: Device,
    dims: Dims,
    embed: Embeddings,
    lm_head: PackedLinear,
    norm: Tensor,
    layers: Vec<Layer>,
    mtp: Option<MtpHead>,
    #[cfg(feature = "metal")]
    fused: Option<fused::Fused>,
}

/// Native multi-token prediction head: `fc` joins the normalized embedding of the next
/// token with the normalized final hidden state, then one full-attention decoder layer
/// and a final norm feed the shared LM head. Its keys and values are the state layer
/// after every text layer.
struct MtpHead {
    fc: PackedLinear,
    embedding_norm: Tensor,
    hidden_norm: Tensor,
    layer: Layer,
    norm: Tensor,
    /// Leading rows of the LM head ([`LoadOptions::draft_vocab`]); `None` shares it whole.
    head: Option<PackedLinear>,
}

impl MtpHead {
    fn lm_head<'a>(&'a self, shared: &'a PackedLinear) -> &'a PackedLinear {
        self.head.as_ref().unwrap_or(shared)
    }
}

/// Input embedding storage.
enum Embeddings {
    /// A dense F32 or BF16 table.
    Table(Tensor),
    /// A packed table, dequantized one gathered row at a time.
    Packed(PackedLinear),
    /// Rows of the tied LM head: no second table.
    Head,
}

/// One live sequence's rows within the packed hidden states and its resolved state.
struct Span<'a> {
    start: usize,
    len: usize,
    /// Committed tokens before this pass.
    past: usize,
    every: bool,
    addresses: Addresses,
    sequence: &'a SequenceForward<'a>,
}

fn load_attention(tensors: &Tensors, a: &str, device: &Device) -> Result<Attention> {
    Ok(Attention {
        q: tensors.projection(&format!("{a}.q_proj"))?,
        kv: tensors.projection(&format!("{a}.kv_proj"))?,
        o: tensors.projection(&format!("{a}.o_proj"))?,
        q_norm: centred(&tensors.take(&format!("{a}.q_norm.weight"))?, device)?,
        k_norm: centred(&tensors.take(&format!("{a}.k_norm.weight"))?, device)?,
    })
}

/// Rows of a separate draft LM head: [`LoadOptions::draft_vocab`] when the head is loaded
/// and the bound excludes part of the vocabulary.
fn draft_vocab(config: &Config, options: &LoadOptions) -> Result<Option<usize>> {
    match options.draft_vocab {
        Some(0) => Err(invalid("draft vocabulary must not be empty")),
        Some(rows) if options.mtp && rows < config.vocab_size() => Ok(Some(rows)),
        _ => Ok(None),
    }
}

fn f32_on(tensor: &Tensor, device: &Device) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F32)?.to_device(device)?)
}

/// Zero-centred `RMSNorm` stores `w`; fold the `1 + w` scale once at load.
fn centred(tensor: &Tensor, device: &Device) -> Result<Tensor> {
    Ok((tensor.to_dtype(DType::F32)? + 1.0)?.to_device(device)?)
}

/// Validated weight source of one load: a `SafeTensors` checkpoint converted by a plan,
/// or a prepacked artifact uploaded as stored.
struct Tensors {
    checkpoint: Checkpoint,
    prepacked: bool,
    plan: WeightPlan,
    projections: BTreeMap<String, Projection>,
    device: Device,
}

impl Tensors {
    fn open(dir: &Path, config: &Config, weights: &Weights, device: &Device) -> Result<Self> {
        let (checkpoint, plan, prepacked) = match weights {
            Weights::Prepacked(manifest) => {
                let (checkpoint, found) = artifact::open_for(dir, config)?;
                if found != **manifest {
                    return Err(invalid(
                        "prepacked artifact changed after its load was planned",
                    ));
                }
                (checkpoint, found.plan, true)
            }
            weights => {
                if artifact::is_artifact(dir) {
                    return Err(invalid(
                        "directory holds a prepacked artifact; load it as Weights::Prepacked",
                    ));
                }
                let plan = weights.plan(config)?;
                plan.validate(config)?;
                (Checkpoint::open(dir, config)?, plan, false)
            }
        };
        Ok(Self {
            checkpoint,
            prepacked,
            plan,
            projections: config
                .projections()?
                .into_iter()
                .map(|p| (p.name.clone(), p))
                .collect(),
            device: device.clone(),
        })
    }

    /// A small tensor (norm, gate, convolution) on the host in its stored dtype.
    fn take(&self, name: &str) -> Result<Tensor> {
        self.checkpoint.take(name)
    }

    /// Projection `name` on the device in its planned precision.
    fn projection(&self, name: &str) -> Result<PackedLinear> {
        let projection = self
            .projections
            .get(name)
            .ok_or_else(|| invalid(format!("model has no projection {name}")))?;
        let precision = self.plan.precision(name)?;
        let shape = (projection.rows, projection.cols);
        if self.prepacked {
            return self.stored(name, precision, shape);
        }
        Packer::RTN
            .pack(&self.checkpoint, projection, precision)?
            .upload(precision, shape, &self.device)
    }

    /// An artifact matrix uploaded exactly as stored; its layout was validated on open.
    fn stored(
        &self,
        name: &str,
        precision: Precision,
        shape: (usize, usize),
    ) -> Result<PackedLinear> {
        let (entry, bytes) = self.checkpoint.take_raw(name)?;
        match precision.ggml() {
            Some(format) => PackedLinear::from_blocks(&bytes, format, shape, &self.device),
            None => PackedLinear::dense(
                &Tensor::from_raw_buffer(
                    &bytes,
                    entry.candle_dtype(name)?,
                    &entry.shape,
                    &Device::Cpu,
                )?,
                &self.device,
            ),
        }
    }

    /// The input embedding in its planned storage. Dense tables are filled on the device
    /// a staging chunk at a time, so the host never holds the whole table.
    fn embedding(&self, config: &Config) -> Result<Embeddings> {
        let name = config.embedding_name();
        let (rows, cols) = (config.vocab_size(), config.hidden_size());
        let precision = match self.plan.embedding {
            Embedding::Head => return Ok(Embeddings::Head),
            Embedding::Table(precision) => precision,
        };
        let dtype = match precision {
            Precision::F32 => DType::F32,
            Precision::Bf16 => DType::BF16,
            _ if self.prepacked => {
                return Ok(Embeddings::Packed(self.stored(
                    &name,
                    precision,
                    (rows, cols),
                )?));
            }
            _ => {
                let table = Projection {
                    name: name.clone(),
                    parts: vec![(name, rows)],
                    rows,
                    cols,
                    experts: 1,
                };
                return Ok(Embeddings::Packed(
                    Packer::RTN
                        .pack(&self.checkpoint, &table, precision)?
                        .upload(precision, (rows, cols), &self.device)?,
                ));
            }
        };
        let table = Tensor::zeros((rows, cols), dtype, &self.device)?;
        let mut row = 0;
        for chunk in self
            .checkpoint
            .take_rows(&name, (STAGING_BYTES / (cols * 4)).max(1))?
        {
            let chunk = chunk?.to_dtype(dtype)?.to_device(&self.device)?;
            let count = chunk.dim(0)?;
            table.slice_set(&chunk, 0, row)?;
            row += count;
        }
        Ok(Embeddings::Table(table))
    }
}

impl Qwen35 {
    /// Description, resident weight bytes and peak workspace computed from validated
    /// metadata, before reading weight payloads or uploading weights. `contexts` bounds
    /// each sequence's resulting position, and `tokens` bounds the packed pass.
    pub fn memory_requirements(
        config: &Config,
        options: &LoadOptions,
        tokens: usize,
        contexts: &[usize],
    ) -> Result<(ModelDescription, usize, usize)> {
        let with_mtp;
        let config = if options.mtp && !config.mtp() {
            with_mtp = config.clone().with_mtp()?;
            &with_mtp
        } else {
            config
        };
        let description = config.description(
            format!("{}#{}", options.revision, options.weights.tag()),
            options.kv,
            options.recurrent,
        )?;
        description.validate()?;
        let plan = options.weights.plan(config)?;
        plan.validate(config)?;
        let mut weights = plan.bytes(config)?;
        if let Some(rows) = draft_vocab(config, options)? {
            weights = plan
                .precision("lm_head")?
                .bytes(rows, config.hidden_size())
                .ok()
                .and_then(|bytes| weights.checked_add(bytes))
                .ok_or_else(|| invalid("weight total overflow"))?;
        }
        let mut stacked: std::collections::BTreeSet<String> = config
            .projections()?
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(|(name, _)| name))
            .collect();
        stacked.insert(config.embedding_name());
        // Every other tensor (norms, convolution, decay, expert routers) stays F32.
        for (name, shape) in config.weight_shapes()? {
            if !stacked.contains(&name) {
                weights = checked_elements(&shape)?
                    .checked_mul(4)
                    .and_then(|bytes| weights.checked_add(bytes))
                    .ok_or_else(|| invalid("weight total overflow"))?;
            }
        }
        let scratch = {
            #[cfg(feature = "metal")]
            if options.execution == Execution::Fused && matches!(options.device, Device::Metal(_)) {
                let kernels = tur_metal::kernels::Kernels::new(&options.device)?;
                fused::workspace(config, &kernels, &options.kv, tokens, contexts)?
            } else {
                0
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = (tokens, contexts);
                0
            }
        };
        Ok((description, weights, scratch))
    }

    /// Validate config and checkpoint headers, then load the text tower onto `options.device`.
    pub fn load(dir: &Path, options: &LoadOptions) -> Result<Self> {
        let mut config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if options.mtp {
            config = config.with_mtp()?;
        }
        let (tokenizer_config, tokenizer) = (
            dir.join("tokenizer_config.json"),
            dir.join("tokenizer.json"),
        );
        if tokenizer_config.exists() && tokenizer.exists() {
            config.forbid_tokenizer_modalities(
                &std::fs::read_to_string(tokenizer_config)?,
                &std::fs::read_to_string(tokenizer)?,
            )?;
        }
        let device = &options.device;
        let tensors = Tensors::open(dir, &config, &options.weights, device)?;
        let dims = Dims::new(&config)?;
        let prefix = config.text_prefix().to_string();
        let mut layers = Vec::with_capacity(config.layer_kinds().len());
        for (i, kind) in config.layer_kinds().iter().enumerate() {
            let p = format!("{prefix}.layers.{i}");
            let mixer = match kind {
                LayerKind::FullAttention => {
                    Mixer::Attention(load_attention(&tensors, &format!("{p}.self_attn"), device)?)
                }
                LayerKind::LinearAttention => {
                    let l = format!("{p}.linear_attn");
                    let conv = tensors.take(&format!("{l}.conv1d.weight"))?;
                    let a_log = tensors.take(&format!("{l}.A_log"))?.to_dtype(DType::F32)?;
                    Mixer::Delta(DeltaNet {
                        proj: tensors.projection(&format!("{l}.in_proj"))?,
                        conv: f32_on(&conv.squeeze(1)?, device)?,
                        neg_a: a_log.exp()?.neg()?.to_device(device)?,
                        dt_bias: f32_on(&tensors.take(&format!("{l}.dt_bias"))?, device)?,
                        norm: f32_on(&tensors.take(&format!("{l}.norm.weight"))?, device)?,
                        out: tensors.projection(&format!("{l}.out_proj"))?,
                    })
                }
            };
            let mlp = match config.moe {
                Some(settings) => Mlp::Moe(Box::new(moe::Moe::load(
                    &tensors,
                    &format!("{p}.mlp"),
                    settings,
                    device,
                )?)),
                None => Mlp::Dense(DenseMlp::load(&tensors, &format!("{p}.mlp"))?),
            };
            layers.push(Layer {
                input_norm: centred(
                    &tensors.take(&format!("{p}.input_layernorm.weight"))?,
                    device,
                )?,
                post_norm: centred(
                    &tensors.take(&format!("{p}.post_attention_layernorm.weight"))?,
                    device,
                )?,
                mixer,
                mlp,
            });
        }
        let mtp = if config.mtp() {
            let p = "mtp.layers.0";
            Some(MtpHead {
                fc: tensors.projection("mtp.fc")?,
                embedding_norm: centred(
                    &tensors.take("mtp.pre_fc_norm_embedding.weight")?,
                    device,
                )?,
                hidden_norm: centred(&tensors.take("mtp.pre_fc_norm_hidden.weight")?, device)?,
                layer: Layer {
                    input_norm: centred(
                        &tensors.take(&format!("{p}.input_layernorm.weight"))?,
                        device,
                    )?,
                    post_norm: centred(
                        &tensors.take(&format!("{p}.post_attention_layernorm.weight"))?,
                        device,
                    )?,
                    mixer: Mixer::Attention(load_attention(
                        &tensors,
                        &format!("{p}.self_attn"),
                        device,
                    )?),
                    mlp: Mlp::Dense(DenseMlp::load(&tensors, &format!("{p}.mlp"))?),
                },
                norm: centred(&tensors.take("mtp.norm.weight")?, device)?,
                head: None,
            })
        } else {
            None
        };
        let lm_head = tensors.projection("lm_head")?;
        let mtp = match (mtp, draft_vocab(&config, options)?) {
            (Some(mtp), Some(rows)) => Some(MtpHead {
                head: Some(lm_head.leading_rows(rows)?),
                ..mtp
            }),
            (mtp, _) => mtp,
        };
        let embed = tensors.embedding(&config)?;
        let norm = centred(&tensors.take(&format!("{prefix}.norm.weight"))?, device)?;
        let revision = format!("{}#{}", options.revision, options.weights.tag());
        Self {
            description: config.description(revision, options.kv, options.recurrent)?,
            config,
            device: device.clone(),
            dims,
            embed,
            lm_head,
            norm,
            layers,
            mtp,
            #[cfg(feature = "metal")]
            fused: None,
        }
        .with_execution(options.execution)
    }

    /// Switch implementation without reloading weights, e.g. to compare the fused kernels
    /// against the reference on identical weights.
    pub fn with_execution(self, execution: Execution) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let fused = match (execution, &self.device) {
                (Execution::Fused, Device::Metal(_)) => Some(fused::Fused::new(&self)?),
                _ => None,
            };
            Ok(Self { fused, ..self })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = execution;
            Ok(self)
        }
    }

    /// The implementation that executes forward passes.
    pub fn execution(&self) -> Execution {
        #[cfg(feature = "metal")]
        if self.fused.is_some() {
            return Execution::Fused;
        }
        Execution::Reference
    }

    /// Device bytes of every weight: packed projections plus embeddings, norms,
    /// convolution and gate parameters. An embedding gathered from the head adds none.
    pub fn weight_bytes(&self) -> usize {
        let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let mut total = self.projection_bytes()
            + bytes(&self.norm)
            + match &self.embed {
                Embeddings::Table(table) => bytes(table),
                Embeddings::Packed(table) => table.resident_bytes(),
                Embeddings::Head => 0,
            };
        if let Some(mtp) = &self.mtp {
            total += bytes(&mtp.embedding_norm) + bytes(&mtp.hidden_norm) + bytes(&mtp.norm);
        }
        for layer in self
            .layers
            .iter()
            .chain(self.mtp.as_ref().map(|m| &m.layer))
        {
            total += bytes(&layer.input_norm) + bytes(&layer.post_norm);
            if let Mlp::Moe(moe) = &layer.mlp {
                total += moe.router_bytes();
            }
            total += match &layer.mixer {
                Mixer::Attention(a) => bytes(&a.q_norm) + bytes(&a.k_norm),
                Mixer::Delta(net) => {
                    bytes(&net.conv) + bytes(&net.neg_a) + bytes(&net.dt_bias) + bytes(&net.norm)
                }
            };
        }
        total
    }

    /// Kernel scratch of a pass of `tokens` packed tokens whose sequences end at
    /// `contexts` tokens, for memory planning before any request exists. The reference
    /// path declares none.
    pub fn planned_workspace_bytes(&self, tokens: usize, contexts: &[usize]) -> Result<usize> {
        #[cfg(feature = "metal")]
        if let Some(fused) = &self.fused {
            return self.planned_workspace(fused, tokens, contexts);
        }
        let _ = (tokens, contexts);
        Ok(0)
    }

    /// Device bytes held by projection weights, all in their packed storage format.
    pub fn projection_bytes(&self) -> usize {
        let mut total = self.lm_head.resident_bytes()
            + self.mtp.as_ref().map_or(0, |mtp| {
                mtp.fc.resident_bytes() + mtp.head.as_ref().map_or(0, PackedLinear::resident_bytes)
            });
        for layer in self
            .layers
            .iter()
            .chain(self.mtp.as_ref().map(|m| &m.layer))
        {
            let mixer = match &layer.mixer {
                Mixer::Attention(a) => [&a.q, &a.kv, &a.o]
                    .iter()
                    .map(|l| l.resident_bytes())
                    .sum::<usize>(),
                Mixer::Delta(net) => net.proj.resident_bytes() + net.out.resident_bytes(),
            };
            total += mixer
                + match &layer.mlp {
                    Mlp::Dense(mlp) => mlp.bytes(),
                    Mlp::Moe(moe) => moe.projection_bytes(),
                };
        }
        total
    }

    /// Run `tokens` from an empty state and return the residual stream after every decoder
    /// layer (`[tokens, hidden]` each, before the final norm) plus logits for `rows`.
    /// For validation against reference implementations.
    pub fn trace(&self, tokens: &[u32], rows: &[usize]) -> Result<(Vec<Tensor>, Tensor)> {
        let mut layers = Vec::with_capacity(self.layers.len());
        let logits = self.single(tokens, rows, Some(&mut layers), None)?;
        Ok((layers, logits))
    }

    /// One pass over `tokens` from an empty state in a private pool.
    fn single(
        &self,
        tokens: &[u32],
        rows: &[usize],
        trace: Option<&mut Vec<Tensor>>,
        recorder: Option<&Recorder>,
    ) -> Result<Tensor> {
        let pool = StatePool::new(
            Arc::new(self.description.clone()),
            &self.device,
            PoolConfig {
                pages: self.description.pages_for(tokens.len()).max(1),
                recurrent_slots: 1,
            },
        )?;
        let mut reservation = ModelState::new(&pool).reserve(tokens.len(), Boundaries::Final)?;
        reservation.prepare()?;
        let state = reservation.binding();
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let sequences = [SequenceForward {
            tokens: 0..tokens.len(),
            state: &state,
            logits: rows,
            hidden: false,
        }];
        let batch = PackedForward {
            tokens,
            positions: &positions,
            sequences: &sequences,
            pool: &pool,
        };
        let span = self.check(&batch, &sequences[0], 0)?;
        let mut steps = self.run(&batch, &[span], trace, recorder)?;
        steps
            .pop()
            .map(|s| s.logits)
            .ok_or_else(|| invalid("pass produced no output"))
    }

    /// Record every projection's input statistics over `samples`, each run from an empty
    /// state through the F32 reference operations, with second moments when `hessians`.
    /// Use F32 weights: statistics describe the unquantized model. The MTP head is not
    /// run and gets none.
    pub fn calibrate(&self, samples: &[Vec<u32>], hessians: bool) -> Result<Calibration> {
        let recorder = Recorder::new(hessians);
        let mut ids = Vec::new();
        for sample in samples {
            self.single(sample, &[], None, Some(&recorder))?;
            self.device.synchronize()?;
            ids.extend(sample.iter().flat_map(|t| t.to_le_bytes()));
        }
        let names: Vec<String> = self
            .config
            .projections()?
            .into_iter()
            .map(|p| p.name)
            .collect();
        let named: Vec<(String, &PackedLinear)> =
            names.into_iter().zip(self.projection_refs()).collect();
        recorder.finish(
            &named,
            samples.len(),
            samples.iter().map(Vec::len).sum(),
            plan::fnv1a(&ids),
        )
    }

    /// Projection `name`.
    pub fn projection(&self, name: &str) -> Result<&PackedLinear> {
        let index = self.projection_index(name)?;
        self.projection_refs()
            .into_iter()
            .nth(index)
            .ok_or_else(|| invalid("projection list is shorter than the config's"))
    }

    /// Replace projection `name` with same-shaped `weights`, returning the previous ones;
    /// e.g. to measure one projection's quantization sensitivity without reloading.
    pub fn replace_projection(
        &mut self,
        name: &str,
        weights: PackedLinear,
    ) -> Result<PackedLinear> {
        let index = self.projection_index(name)?;
        let slot = self
            .projection_refs_mut()
            .into_iter()
            .nth(index)
            .ok_or_else(|| invalid("projection list is shorter than the config's"))?;
        if (slot.out_features(), slot.in_features())
            != (weights.out_features(), weights.in_features())
        {
            return Err(invalid(format!("replacement for {name} has another shape")));
        }
        Ok(std::mem::replace(slot, weights))
    }

    fn projection_index(&self, name: &str) -> Result<usize> {
        self.config
            .projections()?
            .iter()
            .position(|p| p.name == name)
            .ok_or_else(|| invalid(format!("model has no projection {name}")))
    }

    fn projection_refs(&self) -> Vec<&PackedLinear> {
        projections!(self)
    }

    fn projection_refs_mut(&mut self) -> Vec<&mut PackedLinear> {
        projections!(self, mut)
    }

    /// F32 input embeddings of `ids`.
    fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(match &self.embed {
            Embeddings::Table(table) => table.index_select(ids, 0)?.to_dtype(DType::F32)?,
            Embeddings::Packed(table) => table.rows(ids)?,
            Embeddings::Head => self.lm_head.rows(ids)?,
        })
    }

    /// Device holding the weights; an engine executing this model must use the same handle.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Validated architecture settings, EOS IDs and forbidden modality tokens.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Reject a sequence before any work (bounds, modality placeholders, positions and
    /// stale state ids); returns its span at packed row `start`.
    fn check<'a>(
        &self,
        batch: &PackedForward<'_>,
        sequence: &'a SequenceForward<'a>,
        start: usize,
    ) -> Result<Span<'a>> {
        let tokens = batch
            .tokens
            .get(sequence.tokens.clone())
            .filter(|t| !t.is_empty())
            .ok_or_else(|| invalid("sequence range outside packed tokens"))?;
        let past = *batch
            .positions
            .get(sequence.tokens.start)
            .ok_or_else(|| invalid("sequence range outside packed positions"))?;
        if let Some(token) = tokens
            .iter()
            .find(|t| self.config.forbidden_tokens().contains(t))
        {
            return Err(invalid(format!(
                "token {token} is an image/video/audio placeholder; text-only execution rejects it"
            )));
        }
        if sequence.logits.iter().any(|p| !sequence.tokens.contains(p)) {
            return Err(invalid("logit position outside sequence"));
        }
        let state = sequence.state;
        if state.past() != past || state.tokens() != tokens.len() {
            return Err(invalid("sequence state does not match its positions"));
        }
        let addresses = state.resolve(batch.pool)?;
        let every = state.boundaries() == Boundaries::EveryToken;
        let boundaries = if every { tokens.len() } else { 1 };
        if addresses.pages.len() != self.description.pages_for(past + tokens.len())
            || addresses.writes.len() != boundaries
        {
            return Err(invalid("sequence state does not cover its new tokens"));
        }
        Ok(Span {
            start,
            len: tokens.len(),
            past,
            every,
            addresses,
            sequence,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        a: &Attention,
        h: &Tensor,
        rotary: &(Tensor, Tensor),
        pool: &StatePool,
        layer: usize,
        spans: &[Span<'_>],
        recorder: Option<&Recorder>,
    ) -> Result<Tensor> {
        let d = self.dims;
        let n = h.dim(0)?;
        if let Some(recorder) = recorder {
            recorder.record(&a.q, h)?;
            recorder.record(&a.kv, h)?;
        }
        let q_gate = a.q.forward(h)?.reshape((n, d.heads, 2 * d.head_dim))?;
        let q = q_gate.narrow(2, 0, d.head_dim)?.contiguous()?;
        let gate = q_gate
            .narrow(2, d.head_dim, d.head_dim)?
            .contiguous()?
            .reshape((n, d.heads * d.head_dim))?;
        let kv = a.kv.forward(h)?;
        let width = d.kv_heads * d.head_dim;
        let k = kv
            .narrow(1, 0, width)?
            .contiguous()?
            .reshape((n, d.kv_heads, d.head_dim))?;
        let v = kv
            .narrow(1, width, width)?
            .contiguous()?
            .reshape((n, d.kv_heads, d.head_dim))?;
        let q = math::rotary(&math::rms(&q, &a.q_norm, d.eps)?, &rotary.0, &rotary.1)?;
        let k = math::rotary(&math::rms(&k, &a.k_norm, d.eps)?, &rotary.0, &rotary.1)?;
        let store = pool.attention(layer)?;
        let mut outputs = Vec::with_capacity(spans.len());
        for span in spans {
            let pages = &span.addresses.pages;
            store.scatter(
                pages,
                span.past,
                &k.narrow(0, span.start, span.len)?,
                &v.narrow(0, span.start, span.len)?,
            )?;
            // Read new keys back from the pool, so the reference attends to stored values
            // exactly as the kernels do.
            let (keys, values) = store.gather(pages, span.past + span.len)?;
            outputs.push(math::attention(
                &q.narrow(0, span.start, span.len)?,
                &keys,
                &values,
                span.past,
            )?);
        }
        let attended = Tensor::cat(&outputs, 0)?.reshape((n, d.heads * d.head_dim))?;
        let gated = (attended * candle_nn::ops::sigmoid(&gate)?)?;
        if let Some(recorder) = recorder {
            recorder.record(&a.o, &gated)?;
        }
        a.o.forward(&gated)
    }

    fn delta(
        &self,
        net: &DeltaNet,
        h: &Tensor,
        pool: &StatePool,
        layer: usize,
        spans: &[Span<'_>],
        recorder: Option<&Recorder>,
    ) -> Result<Tensor> {
        let d = self.dims;
        let n = h.dim(0)?;
        if let Some(recorder) = recorder {
            recorder.record(&net.proj, h)?;
        }
        let proj = net.proj.forward(h)?;
        let qkv = proj.narrow(1, 0, d.channels)?;
        let z = proj.narrow(1, d.channels, d.value_width)?.contiguous()?;
        let b = proj.narrow(1, d.channels + d.value_width, d.value_heads)?;
        let a = proj.narrow(1, d.channels + d.value_width + d.value_heads, d.value_heads)?;
        let beta = candle_nn::ops::sigmoid(&b.contiguous()?)?;
        let g = math::decay(&a.contiguous()?, &net.dt_bias, &net.neg_a)?;
        let groups = d.value_heads / d.key_heads;
        let store = pool.recurrent(layer)?;
        let mut outputs = Vec::with_capacity(spans.len());
        for span in spans {
            let (convolution, recurrent) = store.read(span.addresses.read)?;
            let every = span.every;
            let len = span.len;
            let conv = math::convolution(
                &qkv.narrow(0, span.start, len)?.contiguous()?,
                &convolution,
                &net.conv,
                every,
            )?;
            let mixed = &conv.output;
            let heads = |offset: usize, width: usize, count: usize, dim: usize| -> Result<Tensor> {
                Ok(mixed
                    .narrow(1, offset, width)?
                    .contiguous()?
                    .reshape((len, count, dim))?)
            };
            let q = math::repeat_heads(&heads(0, d.key_width, d.key_heads, d.key_dim)?, groups)?;
            let k = math::repeat_heads(
                &heads(d.key_width, d.key_width, d.key_heads, d.key_dim)?,
                groups,
            )?;
            let v = heads(2 * d.key_width, d.value_width, d.value_heads, d.value_dim)?;
            let g = g.narrow(0, span.start, len)?.contiguous()?;
            let beta = beta.narrow(0, span.start, len)?.contiguous()?;
            let inputs = DeltaInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            };
            // The reference uses the recurrence for single-token decode and chunks otherwise.
            let core = if every || len == 1 {
                math::delta_recurrent(&inputs, &recurrent, every)?
            } else {
                math::delta_chunked(&inputs, &recurrent)?
            };
            outputs.push(core.output);
            for (b, &slot) in span.addresses.writes.iter().enumerate() {
                store.write(slot, &conv.histories, &core.states, b)?;
            }
        }
        let rows = n * d.value_heads;
        let core = Tensor::cat(&outputs, 0)?.reshape((rows, d.value_dim))?;
        let gated = math::gated_rms(&core, &z.reshape((rows, d.value_dim))?, &net.norm, d.eps)?
            .reshape((n, d.value_width))?;
        if let Some(recorder) = recorder {
            recorder.record(&net.out, &gated)?;
        }
        net.out.forward(&gated)
    }

    fn mlp(layer: &Layer, h: &Tensor, recorder: Option<&Recorder>) -> Result<Tensor> {
        match &layer.mlp {
            Mlp::Dense(mlp) => mlp.forward(h, recorder),
            Mlp::Moe(moe) => moe.reference(h, recorder),
        }
    }

    /// Execute live spans. Recording calibration statistics always uses the reference.
    fn run(
        &self,
        batch: &PackedForward<'_>,
        spans: &[Span<'_>],
        trace: Option<&mut Vec<Tensor>>,
        recorder: Option<&Recorder>,
    ) -> Result<Vec<SequenceStep>> {
        let mut ids = Vec::new();
        let mut positions = Vec::new();
        for span in spans {
            let range = span.sequence.tokens.clone();
            let (Some(tokens), Some(at)) =
                (batch.tokens.get(range.clone()), batch.positions.get(range))
            else {
                return Err(invalid("sequence range outside packed input"));
            };
            ids.extend_from_slice(tokens);
            positions.extend_from_slice(at);
        }
        let mut rows = Vec::new();
        for span in spans {
            let first = span.sequence.tokens.start;
            rows.extend(
                span.sequence
                    .logits
                    .iter()
                    .map(|p| u32::try_from(span.start + (p - first))),
            );
        }
        let rows = rows
            .into_iter()
            .collect::<std::result::Result<Vec<u32>, _>>()
            .map_err(|_| invalid("logit row overflow"))?;
        let pass = Pass {
            ids,
            positions,
            rows,
            hidden: spans.iter().any(|span| span.sequence.hidden),
            spans,
            pool: batch.pool,
        };
        #[cfg(feature = "metal")]
        let (logits, hidden) = match &self.fused {
            Some(fused) if recorder.is_none() => self.run_fused(fused, &pass, trace)?,
            _ => self.run_reference(&pass, trace, recorder)?,
        };
        #[cfg(not(feature = "metal"))]
        let (logits, hidden) = self.run_reference(&pass, trace, recorder)?;
        let mut offset = 0;
        let mut steps = Vec::with_capacity(spans.len());
        for span in spans {
            let rows = span.sequence.logits.len();
            steps.push(SequenceStep {
                logits: logits.narrow(0, offset, rows)?.contiguous()?,
                hidden: match (&hidden, span.sequence.hidden) {
                    (Some(hidden), true) => Some(hidden.narrow(0, span.start, span.len)?),
                    _ => None,
                },
            });
            offset += rows;
        }
        Ok(steps)
    }

    /// F32 reference body: returns logits for `pass.rows`, and every row's final hidden
    /// state when `pass.hidden`, and writes state into the pool.
    fn run_reference(
        &self,
        pass: &Pass<'_, '_>,
        mut trace: Option<&mut Vec<Tensor>>,
        recorder: Option<&Recorder>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let d = self.dims;
        let spans = pass.spans;
        let ids = Tensor::new(pass.ids.as_slice(), &self.device)?;
        let mut x = self.embedding(&ids)?;
        let rotary = math::rotary_tables(&pass.positions, d.rotary_dim, d.theta, &self.device)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let h = math::rms(&x, &layer.input_norm, d.eps)?;
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => {
                    self.attention(a, &h, &rotary, pass.pool, index, spans, recorder)?
                }
                Mixer::Delta(net) => self.delta(net, &h, pass.pool, index, spans, recorder)?,
            };
            x = (x + mixed)?;
            let h = math::rms(&x, &layer.post_norm, d.eps)?;
            x = (&x + Self::mlp(layer, &h, recorder)?)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        if let Some(recorder) = recorder {
            recorder.record(&self.lm_head, &math::rms(&x, &self.norm, d.eps)?)?;
        }
        let hidden = pass
            .hidden
            .then(|| math::rms(&x, &self.norm, d.eps))
            .transpose()?;
        if pass.rows.is_empty() {
            let empty = Tensor::zeros((0, self.description.vocab_size), DType::F32, &self.device)?;
            return Ok((empty, hidden));
        }
        let index = Tensor::new(pass.rows.as_slice(), &self.device)?;
        let h = math::rms(&x.index_select(&index, 0)?, &self.norm, d.eps)?;
        Ok((self.lm_head.forward(&h)?, hidden))
    }

    /// Validate a draft-head sequence; returns its span at packed row `start`.
    fn check_draft<'a>(
        &self,
        batch: &DraftForward<'_>,
        sequence: &DraftSequence<'_>,
        forward: &'a SequenceForward<'a>,
        start: usize,
    ) -> Result<Span<'a>> {
        let len = sequence.tokens.len();
        let positions = batch
            .positions
            .get(sequence.tokens.clone())
            .filter(|_| len > 0 && batch.tokens.get(sequence.tokens.clone()).is_some())
            .ok_or_else(|| invalid("draft range outside packed input"))?;
        let past = positions[0];
        let state = sequence.state;
        if positions.iter().enumerate().any(|(i, &p)| p != past + i)
            || state.past() != past
            || state.tokens() != len
            || sequence.hidden.dims() != [len, self.config.text.hidden_size]
        {
            return Err(invalid("draft state, positions or hidden rows disagree"));
        }
        let addresses = state.resolve(batch.pool)?;
        if addresses.pages.len() < self.description.pages_for(past + len)
            || !addresses.writes.is_empty()
        {
            return Err(invalid(
                "draft state does not cover its committed positions",
            ));
        }
        Ok(Span {
            start,
            len,
            past,
            every: false,
            addresses,
            sequence: forward,
        })
    }

    /// Reference body of a draft-head pass: logits for `pass.rows`.
    fn draft_reference(
        &self,
        mtp: &MtpHead,
        pass: &Pass<'_, '_>,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let d = self.dims;
        let ids = Tensor::new(pass.ids.as_slice(), &self.device)?;
        let embedded = self.embedding(&ids)?;
        let joined = Tensor::cat(
            &[
                math::rms(&embedded, &mtp.embedding_norm, d.eps)?,
                math::rms(hidden, &mtp.hidden_norm, d.eps)?,
            ],
            1,
        )?;
        let mut x = mtp.fc.forward(&joined)?;
        let Mixer::Attention(a) = &mtp.layer.mixer else {
            return Err(invalid("draft head layer is not attention"));
        };
        let rotary = math::rotary_tables(&pass.positions, d.rotary_dim, d.theta, &self.device)?;
        let h = math::rms(&x, &mtp.layer.input_norm, d.eps)?;
        x = (x + self.attention(
            a,
            &h,
            &rotary,
            pass.pool,
            self.layers.len(),
            pass.spans,
            None,
        )?)?;
        let h = math::rms(&x, &mtp.layer.post_norm, d.eps)?;
        x = (&x + Self::mlp(&mtp.layer, &h, None)?)?;
        let index = Tensor::new(pass.rows.as_slice(), &self.device)?;
        mtp.lm_head(&self.lm_head).forward(&math::rms(
            &x.index_select(&index, 0)?,
            &mtp.norm,
            d.eps,
        )?)
    }
}

/// Packed inputs of one pass, gathered from the live sequences.
struct Pass<'s, 'a> {
    ids: Vec<u32>,
    positions: Vec<usize>,
    /// Packed rows whose logits are returned, in sequence order.
    rows: Vec<u32>,
    /// Also return every row's final normalized hidden state.
    hidden: bool,
    spans: &'s [Span<'a>],
    pool: &'s StatePool,
}

impl HybridModel for Qwen35 {
    fn description(&self) -> ModelDescription {
        self.description.clone()
    }

    #[cfg(feature = "metal")]
    fn kernels(&self) -> Option<tur_metal::kernels::Kernels> {
        self.fused.as_ref().map(|fused| fused.kernels().clone())
    }

    /// Peak fused-kernel scratch of this plan (projections, convolved channels, page
    /// addressing and split partials). The reference path is a development oracle
    /// and declares none.
    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> {
        #[cfg(feature = "metal")]
        if let Some(fused) = &self.fused {
            return self.fused_workspace(fused, batch);
        }
        let _ = batch;
        Ok(0)
    }

    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>> {
        if batch.pool.description() != &self.description
            || !batch.pool.device().same_device(&self.device)
        {
            return Err(invalid(
                "state pool was built for another model layout or device",
            ));
        }
        let mut results: Vec<Option<Result<SequenceStep>>> =
            batch.sequences.iter().map(|_| None).collect();
        let mut spans = Vec::with_capacity(batch.sequences.len());
        let mut live = Vec::with_capacity(batch.sequences.len());
        let mut start = 0;
        for (index, sequence) in batch.sequences.iter().enumerate() {
            match self.check(batch, sequence, start) {
                Ok(span) => {
                    start += span.len;
                    spans.push(span);
                    live.push(index);
                }
                Err(error) => results[index] = Some(Err(error)),
            }
        }
        if !spans.is_empty() {
            for (index, step) in live.into_iter().zip(self.run(batch, &spans, None, None)?) {
                results[index] = Some(Ok(step));
            }
        }
        Ok(results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("sequence was not executed"))))
            .collect())
    }

    /// Native multi-token prediction: the head's keys and values for every given position
    /// are written to its own state layer, and logits of each requesting sequence's last
    /// position predict the token after the next.
    fn draft(&self, batch: &DraftForward<'_>) -> Result<Vec<Result<Option<Tensor>>>> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| invalid("model was loaded without its MTP head"))?;
        if batch.pool.description() != &self.description
            || !batch.pool.device().same_device(&self.device)
        {
            return Err(invalid(
                "state pool was built for another model layout or device",
            ));
        }
        let last: Vec<Vec<usize>> = batch
            .sequences
            .iter()
            .map(|s| {
                s.tokens
                    .end
                    .checked_sub(1)
                    .filter(|_| s.logits)
                    .into_iter()
                    .collect()
            })
            .collect();
        let forwards: Vec<SequenceForward<'_>> = batch
            .sequences
            .iter()
            .zip(&last)
            .map(|(s, logits)| SequenceForward {
                tokens: s.tokens.clone(),
                state: s.state,
                logits,
                hidden: false,
            })
            .collect();
        let mut results: Vec<Option<Result<Option<Tensor>>>> =
            batch.sequences.iter().map(|_| None).collect();
        let (mut spans, mut live, mut hidden) = (Vec::new(), Vec::new(), Vec::new());
        let mut start = 0;
        for (index, (sequence, forward)) in batch.sequences.iter().zip(&forwards).enumerate() {
            match self.check_draft(batch, sequence, forward, start) {
                Ok(span) => {
                    start += span.len;
                    spans.push(span);
                    live.push(index);
                    hidden.push(sequence.hidden.clone());
                }
                Err(error) => results[index] = Some(Err(error)),
            }
        }
        if !spans.is_empty() {
            let (mut ids, mut positions, mut rows) = (Vec::new(), Vec::new(), Vec::new());
            for span in &spans {
                let range = span.sequence.tokens.clone();
                ids.extend_from_slice(&batch.tokens[range.clone()]);
                positions.extend_from_slice(&batch.positions[range]);
                if !span.sequence.logits.is_empty() {
                    rows.push(
                        u32::try_from(span.start + span.len - 1)
                            .map_err(|_| invalid("draft row overflow"))?,
                    );
                }
            }
            let pass = Pass {
                ids,
                positions,
                rows,
                hidden: false,
                spans: &spans,
                pool: batch.pool,
            };
            let hidden = Tensor::cat(&hidden, 0)?.to_dtype(DType::F32)?;
            #[cfg(feature = "metal")]
            let logits = match &self.fused {
                Some(fused) => self.draft_fused(fused, mtp, &pass, &hidden)?,
                None => self.draft_reference(mtp, &pass, &hidden)?,
            };
            #[cfg(not(feature = "metal"))]
            let logits = self.draft_reference(mtp, &pass, &hidden)?;
            let mut row = 0;
            for (span, index) in spans.iter().zip(live) {
                results[index] = Some(if span.sequence.logits.is_empty() {
                    Ok(None)
                } else {
                    row += 1;
                    logits.narrow(0, row - 1, 1).map(Some).map_err(Into::into)
                });
            }
        }
        Ok(results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("draft sequence was not executed"))))
            .collect())
    }
}
