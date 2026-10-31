#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Packed Qwen3.5/3.8 text execution over explicit hybrid state.

use super::{
    config::{Config, LayerKind},
    math::{self, DeltaInputs},
};
use crate::hybrid::{
    Addresses, Boundaries, DraftForward, DraftSequence, HybridModel, PackedForward,
    SequenceForward, SequenceStep, StatePool,
};
use crate::models::weights::{Embeddings, Tensors};
use crate::models::{Execution, LoadOptions, Weights, artifact};
use crate::quantization::{Calibration, Recorder};
use candle_core::{DType, Device, Tensor};
use std::{path::Path, sync::Arc};
use tur_core::progress::Stage;
use tur_core::{
    Result,
    execution::{ModelDescription, checked_elements, invalid},
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

pub(super) mod streaming;

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

impl Layer {
    fn load(tensors: &Tensors, config: &Config, i: usize) -> Result<Self> {
        let device = &tensors.device;
        let prefix = config.text_prefix();
        let kind = &config.layer_kinds()[i];
        let p = format!("{prefix}.layers.{i}");
        let mixer = match kind {
            LayerKind::FullAttention => {
                Mixer::Attention(load_attention(tensors, &format!("{p}.self_attn"), device)?)
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
                tensors,
                i,
                &format!("{p}.mlp"),
                settings,
                device,
            )?)),
            None => Mlp::Dense(DenseMlp::load(tensors, &format!("{p}.mlp"))?),
        };
        Ok(Self {
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
        })
    }
    fn reference(
        &self,
        dims: Dims,
        x: Tensor,
        pass: &Pass<'_, '_>,
        index: usize,
        recorder: Option<&Recorder>,
    ) -> Result<Tensor> {
        let h = math::rms(&x, &self.input_norm, dims.eps)?;
        let mixed = match &self.mixer {
            Mixer::Attention(a) => {
                let rotary =
                    math::rotary_tables(&pass.positions, dims.rotary_dim, dims.theta, x.device())?;
                Qwen35::attention(dims, a, &h, &rotary, pass.pool, index, pass.spans, recorder)?
            }
            Mixer::Delta(net) => {
                Qwen35::delta(dims, net, &h, pass.pool, index, pass.spans, recorder)?
            }
        };
        let x = (x + mixed)?;
        let h = math::rms(&x, &self.post_norm, dims.eps)?;
        Ok((&x + Qwen35::mlp(self, &h, recorder)?)?)
    }
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
    /// Bounded expert storage the layers' banks read, when this load has one.
    #[cfg(feature = "metal")]
    cache: Option<Arc<tur_metal::experts::ExpertCache>>,
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
    /// Leading rows of the LM head ([`DraftHead::vocab`]); `None` shares it whole.
    head: Option<PackedLinear>,
}

impl MtpHead {
    fn lm_head<'a>(&'a self, shared: &'a PackedLinear) -> &'a PackedLinear {
        self.head.as_ref().unwrap_or(shared)
    }
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

/// Rows of a separate draft LM head: [`DraftHead::vocab`] when the head is loaded
/// and the bound excludes part of the vocabulary.
fn draft_vocab(config: &Config, options: &LoadOptions) -> Result<Option<usize>> {
    if options
        .mtp
        .as_ref()
        .is_some_and(|head| head.model.is_some())
    {
        return Err(invalid(
            "Qwen3.5 drafts from the head inside its checkpoint, not a separate drafter",
        ));
    }
    match options.mtp.as_ref().and_then(|head| head.vocab) {
        Some(0) => Err(invalid("draft vocabulary must not be empty")),
        Some(rows) if rows < config.vocab_size() => Ok(Some(rows)),
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

/// Experts per layer and the bank shapes of a checkpoint whose experts a cache can serve: a
/// prepacked artifact on the Metal kernels, whose banks are on disk as the kernels read them.
#[cfg(feature = "metal")]
fn cache_banks(
    config: &Config,
    plan: &crate::models::WeightPlan,
    options: &LoadOptions,
) -> Result<(
    usize,
    tur_metal::experts::BankShape,
    tur_metal::experts::BankShape,
)> {
    use tur_metal::experts::{BankFormat, BankShape};
    let moe = config
        .moe
        .ok_or_else(|| invalid("the expert cache needs a checkpoint with experts"))?;
    if !matches!(options.weights, Weights::Prepacked(_))
        || options.execution != Execution::Fused
        || !options.device.is_metal()
    {
        return Err(invalid(
            "the Qwen expert cache reads a prepacked artifact on the Metal kernels",
        ));
    }
    let banks: Vec<_> = config
        .projections()?
        .into_iter()
        .filter(|p| p.experts > 1)
        .collect();
    let format = |suffix: &str| -> Result<(BankFormat, usize, usize)> {
        let mut shape = None;
        for bank in banks.iter().filter(|p| p.name.ends_with(suffix)) {
            let ggml = plan
                .precision(&bank.name)?
                .ggml()
                .ok_or_else(|| invalid("the expert cache needs packed expert banks"))?;
            let found = (
                BankFormat::Packed(ggml),
                bank.rows / bank.experts,
                bank.cols,
            );
            if *shape.get_or_insert(found) != found {
                return Err(invalid(
                    "the expert cache needs one bank format per projection",
                ));
            }
        }
        shape.ok_or_else(|| invalid("checkpoint has no expert banks"))
    };
    let bank = |(format, rows, cols)| BankShape { rows, cols, format };
    Ok((
        moe.num_experts,
        bank(format("gate_up_proj")?),
        bank(format("down_proj")?),
    ))
}

/// `weights` with the expert banks replaced by a cache of `budget` bytes.
fn cached_weights(
    config: &Config,
    plan: &crate::models::WeightPlan,
    weights: usize,
    budget: usize,
    options: &LoadOptions,
) -> Result<usize> {
    #[cfg(feature = "metal")]
    {
        use tur_metal::experts::ExpertCache;
        let (experts, gate_up, down) = cache_banks(config, plan, options)?;
        let slots = ExpertCache::planned_slots(budget, experts, gate_up, down)?;
        let arena = checked_elements(&[slots, ExpertCache::expert_bytes(gate_up, down)?])?;
        let mut total = weights;
        for bank in config.projections()?.iter().filter(|p| p.experts > 1) {
            total = total
                .checked_sub(plan.precision(&bank.name)?.bytes(bank.rows, bank.cols)?)
                .ok_or_else(|| invalid("expert banks exceed the planned weights"))?;
        }
        total
            .checked_add(arena)
            .ok_or_else(|| invalid("weight total overflow"))
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (config, plan, weights, budget, options);
        Err(invalid("the expert cache needs Metal"))
    }
}

/// A cache over the expert banks of the open artifact behind `tensors`.
#[cfg(feature = "metal")]
fn open_cache(
    tensors: &Tensors,
    config: &Config,
    budget: usize,
    options: &LoadOptions,
) -> Result<Arc<tur_metal::experts::ExpertCache>> {
    use tur_metal::experts::{BankSource, ExpertCache, ExpertSources};
    let text_layers = config.layer_kinds().len();
    let (experts, gate_up, down) = cache_banks(config, &tensors.plan, options)?;
    let moe = config
        .moe
        .ok_or_else(|| invalid("the expert cache needs a checkpoint with experts"))?;
    let bank = |layer: usize, name: &str| -> Result<BankSource> {
        let (file, blocks) = tensors.checkpoint.location(&if layer < text_layers {
            format!("{}.layers.{layer}.mlp.experts.{name}", config.text_prefix())
        } else {
            format!("mtp.layers.0.mlp.experts.{name}")
        })?;
        Ok(BankSource {
            file: file.to_path_buf(),
            blocks,
            scales: 0,
        })
    };
    let sources = ExpertSources {
        experts,
        top_k: moe.num_experts_per_tok,
        gate_up,
        down,
        layers: (0..text_layers + usize::from(config.mtp()))
            .map(|i| Ok([bank(i, "gate_up_proj")?, bank(i, "down_proj")?]))
            .collect::<Result<_>>()?,
    };
    // Weights besides the arena, which the budget bounds from above.
    let unloaded = cached_weights(
        config,
        &tensors.plan,
        tensors.plan.bytes(config)?,
        budget,
        options,
    )?
    .saturating_sub(budget);
    Ok(Arc::new(ExpertCache::new(
        &sources,
        budget,
        unloaded,
        &options.device,
    )?))
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
                (config.open_checkpoint(dir)?, plan, false)
            }
        };
        Ok(Self {
            #[cfg(feature = "metal")]
            expert_cache: None,
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
        let config = if options.mtp.is_some() && !config.mtp() {
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
        if let Some(budget) = options.expert_cache {
            weights = cached_weights(config, &plan, weights, budget, options)?;
        }
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
        if options.mtp.is_some() {
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
        #[cfg_attr(
            not(feature = "metal"),
            allow(unused_mut, reason = "only Metal caches")
        )]
        let mut tensors = Tensors::open(dir, &config, &options.weights, device)?;
        #[cfg(feature = "metal")]
        let cache = options
            .expert_cache
            .map(|budget| open_cache(&tensors, &config, budget, options))
            .transpose()?;
        #[cfg(feature = "metal")]
        {
            tensors.expert_cache.clone_from(&cache);
        }
        #[cfg(not(feature = "metal"))]
        if options.expert_cache.is_some() {
            return Err(invalid("the expert cache needs Metal"));
        }
        let dims = Dims::new(&config)?;
        let prefix = config.text_prefix().to_string();
        let mut layers = Vec::with_capacity(config.layer_kinds().len());
        let mut loading = Stage::begin(
            "Loading weights",
            config.layer_kinds().len() as u64,
            "layers",
        );
        for i in 0..config.layer_kinds().len() {
            layers.push(Layer::load(&tensors, &config, i)?);
            loading.advance(1);
        }
        loading.finish();
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
                    mlp: match config.moe {
                        // The head's bank is one more layer of a cache.
                        Some(settings) => Mlp::Moe(Box::new(moe::Moe::load(
                            &tensors,
                            config.layer_kinds().len(),
                            &format!("{p}.mlp"),
                            settings,
                            device,
                        )?)),
                        None => Mlp::Dense(DenseMlp::load(&tensors, &format!("{p}.mlp"))?),
                    },
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
        let embed = tensors.embedding(
            &config.embedding_name(),
            config.vocab_size(),
            config.hidden_size(),
        )?;
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
            #[cfg(feature = "metal")]
            cache,
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
        let mut total = self.projection_bytes() + bytes(&self.norm) + self.embed.bytes();
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

    /// What the expert cache has done, when this load has one.
    #[cfg(feature = "metal")]
    pub fn expert_stats(&self) -> Option<tur_metal::experts::ExpertStats> {
        self.cache.as_ref().map(|cache| cache.stats())
    }

    /// Device bytes held by projection weights, all in their packed storage format.
    pub fn projection_bytes(&self) -> usize {
        #[cfg(feature = "metal")]
        let cache = self
            .cache
            .as_ref()
            .map_or(0, |cache| cache.resident_bytes());
        #[cfg(not(feature = "metal"))]
        let cache = 0;
        let mut total = cache
            + self.lm_head.resident_bytes()
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
                window_pages: self.description.pages_for(tokens.len()).max(1),
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
            embeddings: &[],
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
        let mut stage = Stage::begin("Calibrating", samples.len() as u64, "samples");
        for sample in samples {
            self.single(sample, &[], None, Some(&recorder))?;
            self.device.synchronize()?;
            ids.extend(sample.iter().flat_map(|t| t.to_le_bytes()));
            stage.advance(1);
        }
        stage.finish();
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
            crate::models::weights::fnv1a(&ids),
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
        self.embed.rows(ids, &self.lm_head)
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

    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the fused attention's inputs for differential tests"
    )]
    fn attention(
        d: Dims,
        a: &Attention,
        h: &Tensor,
        rotary: &(Tensor, Tensor),
        pool: &StatePool,
        layer: usize,
        spans: &[Span<'_>],
        recorder: Option<&Recorder>,
    ) -> Result<Tensor> {
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
                (d.head_dim as f64).sqrt().recip(),
                None,
                None,
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
        d: Dims,
        net: &DeltaNet,
        h: &Tensor,
        pool: &StatePool,
        layer: usize,
        spans: &[Span<'_>],
        recorder: Option<&Recorder>,
    ) -> Result<Tensor> {
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
        let ids = Tensor::new(pass.ids.as_slice(), &self.device)?;
        let mut x = self.embedding(&ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.reference(self.dims, x, pass, index, recorder)?;
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
        x = (x + Self::attention(
            self.dims,
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
        if !batch.embeddings.is_empty() {
            return Err(invalid("Qwen3.5 accepts text input only"));
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
                    .filter(|_| s.drafts > 0)
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
