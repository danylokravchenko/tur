#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Packed Qwen3.5/3.8 text execution over explicit hybrid state.

use super::{
    config::{Config, LayerKind},
    math::{self, DeltaInputs},
    weights::Checkpoint,
};
use crate::hybrid::{
    Addresses, Boundaries, HybridModel, PackedForward, SequenceForward, SequenceStep, StatePool,
};
use candle_core::{DType, Device, Tensor, quantized::GgmlDType};
use std::{path::Path, sync::Arc};
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, ModelDescription, checked_elements, invalid},
};
use tur_metal::{packed::PackedLinear, pool::PoolConfig, state::ModelState};

#[cfg(feature = "metal")]
mod fused;
mod moe;

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

/// K/V storage chosen from the P5 measurements on the M1 (see the P5 report): F16 was the
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
    /// Exact checkpoint identity (repository and commit). The weight format is appended.
    pub revision: String,
    pub format: WeightFormat,
    pub device: Device,
    pub execution: Execution,
    /// Paged K/V page size and storage format; part of the state identity.
    pub kv: KvLayout,
    /// Storage of recurrent matrix state (F32, F16 or BF16); part of the state identity.
    pub recurrent: DType,
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
    fn load(
        tensors: &mut Tensors,
        prefix: &str,
        format: WeightFormat,
        device: &Device,
    ) -> Result<Self> {
        let gate = tensors.take(&format!("{prefix}.gate_proj.weight"))?;
        let up = tensors.take(&format!("{prefix}.up_proj.weight"))?;
        Ok(Self {
            gate_up: linear(&Tensor::cat(&[gate, up], 0)?, format, device)?,
            down: linear(
                &tensors.take(&format!("{prefix}.down_proj.weight"))?,
                format,
                device,
            )?,
        })
    }

    fn bytes(&self) -> usize {
        self.gate_up.resident_bytes() + self.down.resident_bytes()
    }

    fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let i = self.down.in_features();
        let gate_up = self.gate_up.forward(h)?;
        let act = (candle_nn::ops::silu(&gate_up.narrow(1, 0, i)?)? * gate_up.narrow(1, i, i)?)?;
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
    embed: Tensor,
    lm_head: PackedLinear,
    /// The head is the embedding itself (tied F32 weights), not a second copy.
    shares_embedding: bool,
    norm: Tensor,
    layers: Vec<Layer>,
    #[cfg(feature = "metal")]
    fused: Option<fused::Fused>,
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

fn linear(weight: &Tensor, format: WeightFormat, device: &Device) -> Result<PackedLinear> {
    match format {
        WeightFormat::F32 => PackedLinear::dense(weight, device),
        WeightFormat::Q8_0 => PackedLinear::quantized(weight, GgmlDType::Q8_0, device),
        WeightFormat::Q4K => PackedLinear::quantized(weight, GgmlDType::Q4K, device),
    }
}

fn f32_on(tensor: &Tensor, device: &Device) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F32)?.to_device(device)?)
}

/// Zero-centred `RMSNorm` stores `w`; fold the `1 + w` scale once at load.
fn centred(tensor: &Tensor, device: &Device) -> Result<Tensor> {
    Ok((tensor.to_dtype(DType::F32)? + 1.0)?.to_device(device)?)
}

struct Tensors(Checkpoint);

impl Tensors {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        self.0.take(name)
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
        let description = config.description(
            format!("{}#{}", options.revision, options.format.tag()),
            options.kv,
            options.recurrent,
        )?;
        description.validate()?;
        let quant = match options.format {
            WeightFormat::F32 => None,
            WeightFormat::Q8_0 => Some(GgmlDType::Q8_0),
            WeightFormat::Q4K => Some(GgmlDType::Q4K),
        };
        let projection_bytes = |shape: &[usize]| -> Result<usize> {
            let elements = checked_elements(shape)?;
            if let Some(format) = quant {
                if shape
                    .last()
                    .is_none_or(|width| !width.is_multiple_of(format.block_size()))
                {
                    return Err(invalid(
                        "projection width does not fit its quantization block",
                    ));
                }
                (elements / format.block_size()).checked_mul(format.type_size())
            } else {
                elements.checked_mul(4)
            }
            .ok_or_else(|| invalid("weight size overflow"))
        };
        let mut weights = 0usize;
        for (name, shape) in config.weight_shapes()? {
            let embedding = name.ends_with("embed_tokens.weight");
            // Every other rank-two checkpoint tensor is a projection, including the
            // in_proj_qkv/z/a/b matrices that are concatenated by the loader.
            let router = name.ends_with("mlp.gate.weight")
                || name.ends_with("mlp.shared_expert_gate.weight");
            let expert = name.ends_with("mlp.experts.gate_up_proj")
                || name.ends_with("mlp.experts.down_proj");
            let bytes = if (shape.len() == 2 || expert) && !embedding && !router {
                projection_bytes(&shape)?
            } else {
                let element_bytes = if embedding && quant.is_some() { 2 } else { 4 };
                checked_elements(&shape)?
                    .checked_mul(element_bytes)
                    .ok_or_else(|| invalid("weight size overflow"))?
            };
            weights = weights
                .checked_add(bytes)
                .ok_or_else(|| invalid("weight total overflow"))?;
        }
        if config.tied_embeddings() && quant.is_some() {
            weights = weights
                .checked_add(projection_bytes(&[
                    config.text.vocab_size,
                    config.text.hidden_size,
                ])?)
                .ok_or_else(|| invalid("tied head size overflow"))?;
        }
        let scratch = {
            #[cfg(feature = "metal")]
            if options.execution == Execution::Fused && matches!(options.device, Device::Metal(_)) {
                let kernels = tur_metal::kernels::Kernels::new(&options.device)?;
                fused::workspace(config, &kernels, options.kv.page_tokens, tokens, contexts)?
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
        let checkpoint = Checkpoint::open(dir, &config)?;
        let mut tensors = Tensors(checkpoint);
        let device = &options.device;
        let format = options.format;
        let dims = Dims::new(&config)?;
        let prefix = config.text_prefix().to_string();
        let mut layers = Vec::with_capacity(config.layer_kinds().len());
        for (i, kind) in config.layer_kinds().iter().enumerate() {
            let p = format!("{prefix}.layers.{i}");
            let mixer = match kind {
                LayerKind::FullAttention => {
                    let a = format!("{p}.self_attn");
                    let k = tensors.take(&format!("{a}.k_proj.weight"))?;
                    let v = tensors.take(&format!("{a}.v_proj.weight"))?;
                    Mixer::Attention(Attention {
                        q: linear(
                            &tensors.take(&format!("{a}.q_proj.weight"))?,
                            format,
                            device,
                        )?,
                        kv: linear(&Tensor::cat(&[k, v], 0)?, format, device)?,
                        o: linear(
                            &tensors.take(&format!("{a}.o_proj.weight"))?,
                            format,
                            device,
                        )?,
                        q_norm: centred(&tensors.take(&format!("{a}.q_norm.weight"))?, device)?,
                        k_norm: centred(&tensors.take(&format!("{a}.k_norm.weight"))?, device)?,
                    })
                }
                LayerKind::LinearAttention => {
                    let l = format!("{p}.linear_attn");
                    let parts = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                        .iter()
                        .map(|n| tensors.take(&format!("{l}.{n}.weight")))
                        .collect::<Result<Vec<_>>>()?;
                    let conv = tensors.take(&format!("{l}.conv1d.weight"))?;
                    let a_log = tensors.take(&format!("{l}.A_log"))?.to_dtype(DType::F32)?;
                    Mixer::Delta(DeltaNet {
                        proj: linear(&Tensor::cat(&parts, 0)?, format, device)?,
                        conv: f32_on(&conv.squeeze(1)?, device)?,
                        neg_a: a_log.exp()?.neg()?.to_device(device)?,
                        dt_bias: f32_on(&tensors.take(&format!("{l}.dt_bias"))?, device)?,
                        norm: f32_on(&tensors.take(&format!("{l}.norm.weight"))?, device)?,
                        out: linear(
                            &tensors.take(&format!("{l}.out_proj.weight"))?,
                            format,
                            device,
                        )?,
                    })
                }
            };
            let mlp = match config.moe {
                Some(settings) => Mlp::Moe(Box::new(moe::Moe::load(
                    &mut tensors,
                    &format!("{p}.mlp"),
                    settings,
                    format,
                    device,
                )?)),
                None => Mlp::Dense(DenseMlp::load(
                    &mut tensors,
                    &format!("{p}.mlp"),
                    format,
                    device,
                )?),
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
        let embed_source = tensors.take(&format!("{prefix}.embed_tokens.weight"))?;
        // Quantized formats keep the lookup table in BF16; only selected rows become F32.
        let embed = match format {
            WeightFormat::F32 => f32_on(&embed_source, device)?,
            _ => embed_source.to_dtype(DType::BF16)?.to_device(device)?,
        };
        let shares_embedding = config.tied_embeddings() && format == WeightFormat::F32;
        let lm_head = match (config.tied_embeddings(), format) {
            (true, WeightFormat::F32) => PackedLinear::dense(&embed, device)?,
            (true, _) => linear(&embed_source, format, device)?,
            (false, _) => linear(&tensors.take("lm_head.weight")?, format, device)?,
        };
        let norm = centred(&tensors.take(&format!("{prefix}.norm.weight"))?, device)?;
        let revision = format!("{}#{}", options.revision, format.tag());
        Self {
            description: config.description(revision, options.kv, options.recurrent)?,
            config,
            device: device.clone(),
            dims,
            embed,
            lm_head,
            shares_embedding,
            norm,
            layers,
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
    /// convolution and gate parameters. A tied F32 head shares the embedding's storage.
    pub fn weight_bytes(&self) -> usize {
        let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let mut total = self.projection_bytes() + bytes(&self.norm);
        if !self.shares_embedding {
            total += bytes(&self.embed);
        }
        for layer in &self.layers {
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
        let mut total = self.lm_head.resident_bytes();
        for layer in &self.layers {
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
        }];
        let batch = PackedForward {
            tokens,
            positions: &positions,
            sequences: &sequences,
            pool: &pool,
        };
        let span = self.check(&batch, &sequences[0], 0)?;
        let mut layers = Vec::with_capacity(self.layers.len());
        let mut steps = self.run(&batch, &[span], Some(&mut layers))?;
        let logits = steps
            .pop()
            .map(|s| s.logits)
            .ok_or_else(|| invalid("trace produced no output"))?;
        Ok((layers, logits))
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

    fn attention(
        &self,
        a: &Attention,
        h: &Tensor,
        rotary: &(Tensor, Tensor),
        pool: &StatePool,
        layer: usize,
        spans: &[Span<'_>],
    ) -> Result<Tensor> {
        let d = self.dims;
        let n = h.dim(0)?;
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
        a.o.forward(&(attended * candle_nn::ops::sigmoid(&gate)?)?)
    }

    fn delta(
        &self,
        net: &DeltaNet,
        h: &Tensor,
        pool: &StatePool,
        layer: usize,
        spans: &[Span<'_>],
    ) -> Result<Tensor> {
        let d = self.dims;
        let n = h.dim(0)?;
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
        let gated = math::gated_rms(&core, &z.reshape((rows, d.value_dim))?, &net.norm, d.eps)?;
        net.out.forward(&gated.reshape((n, d.value_width))?)
    }

    fn mlp(layer: &Layer, h: &Tensor) -> Result<Tensor> {
        match &layer.mlp {
            Mlp::Dense(mlp) => mlp.forward(h),
            Mlp::Moe(moe) => moe.reference(h),
        }
    }

    fn run(
        &self,
        batch: &PackedForward<'_>,
        spans: &[Span<'_>],
        trace: Option<&mut Vec<Tensor>>,
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
            spans,
            pool: batch.pool,
        };
        #[cfg(feature = "metal")]
        let logits = match &self.fused {
            Some(fused) => self.run_fused(fused, &pass, trace)?,
            None => self.run_reference(&pass, trace)?,
        };
        #[cfg(not(feature = "metal"))]
        let logits = self.run_reference(&pass, trace)?;
        let mut offset = 0;
        let mut steps = Vec::with_capacity(spans.len());
        for span in spans {
            let rows = span.sequence.logits.len();
            steps.push(SequenceStep {
                logits: logits.narrow(0, offset, rows)?.contiguous()?,
            });
            offset += rows;
        }
        Ok(steps)
    }

    /// F32 reference body: returns logits for `pass.rows` and writes state into the pool.
    fn run_reference(
        &self,
        pass: &Pass<'_, '_>,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let d = self.dims;
        let spans = pass.spans;
        let ids = Tensor::new(pass.ids.as_slice(), &self.device)?;
        let mut x = self.embed.index_select(&ids, 0)?.to_dtype(DType::F32)?;
        let rotary = math::rotary_tables(&pass.positions, d.rotary_dim, d.theta, &self.device)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let h = math::rms(&x, &layer.input_norm, d.eps)?;
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => self.attention(a, &h, &rotary, pass.pool, index, spans)?,
                Mixer::Delta(net) => self.delta(net, &h, pass.pool, index, spans)?,
            };
            x = (x + mixed)?;
            let h = math::rms(&x, &layer.post_norm, d.eps)?;
            x = (&x + Self::mlp(layer, &h)?)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        if pass.rows.is_empty() {
            return Ok(Tensor::zeros(
                (0, self.description.vocab_size),
                DType::F32,
                &self.device,
            )?);
        }
        let index = Tensor::new(pass.rows.as_slice(), &self.device)?;
        let h = math::rms(&x.index_select(&index, 0)?, &self.norm, d.eps)?;
        self.lm_head.forward(&h)
    }
}

/// Packed inputs of one pass, gathered from the live sequences.
struct Pass<'s, 'a> {
    ids: Vec<u32>,
    positions: Vec<usize>,
    /// Packed rows whose logits are returned, in sequence order.
    rows: Vec<u32>,
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
            for (index, step) in live.into_iter().zip(self.run(batch, &spans, None)?) {
                results[index] = Some(Ok(step));
            }
        }
        Ok(results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("sequence was not executed"))))
            .collect())
    }
}
