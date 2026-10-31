//! gpt-oss equations over the common packed weights and persistent state pools.
use super::Config;
use super::config::SWIGLU_ALPHA;
use super::prune::ExpertProbe;
#[cfg(feature = "metal")]
use super::prune::LayerProbe;
#[cfg(feature = "metal")]
use crate::models::paging::Paging;
use crate::{
    hybrid::{
        Addresses, Boundaries, HybridModel, PackedForward, SequenceForward, SequenceStep, StatePool,
    },
    models::{
        Execution, LoadOptions, Precision, WeightPlan, Weights, artifact,
        checkpoint::{Checkpoint, weight_files},
        math,
        weights::{Embeddings, Tensors, WeightLayout},
    },
};
use candle_core::{DType, Device, Tensor};
use std::{path::Path, sync::Arc};
use tur_core::progress::Stage;
use tur_core::{
    Result,
    execution::{ModelDescription, checked_elements, invalid},
};
use tur_metal::{packed::PackedLinear, pool::PoolConfig, state::ModelState};

struct Attention {
    q: PackedLinear,
    q_bias: Tensor,
    /// `k | v`, and their biases as one row.
    kv: PackedLinear,
    kv_bias: Tensor,
    o: PackedLinear,
    o_bias: Tensor,
    sinks: Tensor,
}

struct Experts {
    router: Tensor,
    router_bias: Tensor,
    /// Resident `gate_up` and `down` banks; `None` when the model's cache serves them.
    banks: Option<(PackedLinear, PackedLinear)>,
    gate_up_bias: Tensor,
    down_bias: Tensor,
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    attention: Attention,
    experts: Experts,
}

/// Input embeddings: a token only ever needs its own row of a table over a gigabyte in
/// size, so a load that is short of memory reads rows from the checkpoint on demand.
enum Embedding {
    Resident(Embeddings),
    OnDisk(Checkpoint),
}

/// gpt-oss text tower; immutable weights, with all request state in the engine's pools.
pub struct GptOss {
    config: Config,
    description: ModelDescription,
    device: Device,
    embedding: Embedding,
    head: PackedLinear,
    norm: Tensor,
    layers: Vec<Layer>,
    #[cfg(feature = "metal")]
    inv_freq: Tensor,
    #[cfg(feature = "metal")]
    amplitude: Tensor,
    #[cfg(feature = "metal")]
    kernels: Option<tur_metal::kernels::Kernels>,
    #[cfg(feature = "metal")]
    cache: Option<tur_metal::experts::ExpertCache>,
    /// Filled while [`Self::expert_probe`] runs.
    #[cfg(feature = "metal")]
    probe: parking_lot::Mutex<Option<ExpertProbe>>,
    /// Per layer, a router penalty that hides experts a candidate pruning removes.
    #[cfg(feature = "metal")]
    hidden_experts: parking_lot::Mutex<Option<Vec<Tensor>>>,
}

fn checkpoint(dir: &Path, config: &Config) -> Result<Checkpoint> {
    let (checkpoint, _) = Checkpoint::scan(weight_files(dir)?)?;
    let shapes = config.weight_shapes();
    for (name, shape) in &shapes {
        let entry = checkpoint
            .entries()
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        if &entry.shape != shape {
            return Err(invalid(format!(
                "{name} has shape {:?}, expected {shape:?}",
                entry.shape
            )));
        }
        let packed = entry.dtype == safetensors::Dtype::U8;
        if packed != Config::is_packed(name) || (!packed && entry.candle_dtype(name)? == DType::U8)
        {
            return Err(invalid(format!("{name} has dtype {:?}", entry.dtype)));
        }
    }
    if let Some(extra) = checkpoint
        .entries()
        .keys()
        .find(|n| !shapes.contains_key(*n))
    {
        return Err(invalid(format!("unexpected tensor {extra}")));
    }
    Ok(checkpoint)
}

/// Experts stay in MXFP4 except where the host multiplies them: reference execution, or
/// an all-F32 plan. A plan's precision describes only the other projections.
fn packed_experts(plan: &WeightPlan, options: &LoadOptions) -> bool {
    options.execution == Execution::Fused
        && options.device.is_metal()
        && plan.projections.values().any(|p| *p != Precision::F32)
}

fn f32_tensor(tensors: &Tensors, name: &str) -> Result<Tensor> {
    Ok(tensors
        .take(name)?
        .to_dtype(DType::F32)?
        .to_device(&tensors.device)?)
}

/// How one load stores experts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// Decoded to F32 for the host or an all-F32 plan.
    Dense,
    /// Whole banks in MXFP4 on the device.
    Resident,
    /// In a bounded cache, read from the checkpoint on demand.
    Cached,
}

impl Placement {
    /// A cache reads MXFP4 as stored whatever the other projections use, so an all-F32
    /// plan can check real weights against a reference without expanding the experts.
    fn of(plan: &WeightPlan, options: &LoadOptions) -> Result<Self> {
        let fused = options.execution == Execution::Fused && options.device.is_metal();
        match (options.expert_cache, packed_experts(plan, options)) {
            (None, true) => Ok(Self::Resident),
            (None, false) => Ok(Self::Dense),
            (Some(_), _) if fused && cfg!(feature = "metal") => Ok(Self::Cached),
            (Some(_), _) => Err(invalid("the expert cache needs fused Metal execution")),
        }
    }
}

fn expert_bank(tensors: &Tensors, name: &str, packed: bool) -> Result<PackedLinear> {
    let (blocks_entry, blocks) = tensors.checkpoint.take_raw(&format!("{name}_blocks"))?;
    let (_, scales) = tensors.checkpoint.take_raw(&format!("{name}_scales"))?;
    let [experts, rows, groups, _] = blocks_entry.shape.as_slice() else {
        return Err(invalid("MXFP4 blocks are not a stack of matrices"));
    };
    let shape = (
        checked_elements(&[*experts, *rows])?,
        checked_elements(&[*groups, tur_metal::quantize::MXFP4_BLOCK_VALUES])?,
    );
    if packed {
        return PackedLinear::mxfp4(&blocks, &scales, shape, &tensors.device);
    }
    let values = tur_metal::quantize::mxfp4_decode(&blocks, &scales)?;
    PackedLinear::dense(
        &Tensor::from_vec(values, shape, &Device::Cpu)?,
        &tensors.device,
    )
}

impl Layer {
    fn load(tensors: &Tensors, index: usize, placement: Placement) -> Result<Self> {
        let p = format!("model.layers.{index}");
        let a = format!("{p}.self_attn");
        let m = format!("{p}.mlp");
        let bias = |name: &str| f32_tensor(tensors, name);
        Ok(Self {
            input_norm: bias(&format!("{p}.input_layernorm.weight"))?,
            post_norm: bias(&format!("{p}.post_attention_layernorm.weight"))?,
            attention: Attention {
                q: tensors.projection(&format!("{a}.q_proj"))?,
                q_bias: bias(&format!("{a}.q_proj.bias"))?,
                kv: tensors.projection(&format!("{a}.kv_proj"))?,
                kv_bias: Tensor::cat(
                    &[
                        bias(&format!("{a}.k_proj.bias"))?,
                        bias(&format!("{a}.v_proj.bias"))?,
                    ],
                    0,
                )?,
                o: tensors.projection(&format!("{a}.o_proj"))?,
                o_bias: bias(&format!("{a}.o_proj.bias"))?,
                sinks: bias(&format!("{a}.sinks"))?,
            },
            experts: Experts {
                router: bias(&format!("{m}.router.weight"))?,
                router_bias: bias(&format!("{m}.router.bias"))?,
                banks: match placement {
                    Placement::Cached => None,
                    placement => {
                        let packed = placement == Placement::Resident;
                        Some((
                            expert_bank(tensors, &format!("{m}.experts.gate_up_proj"), packed)?,
                            expert_bank(tensors, &format!("{m}.experts.down_proj"), packed)?,
                        ))
                    }
                },
                gate_up_bias: bias(&format!("{m}.experts.gate_up_proj_bias"))?,
                down_bias: bias(&format!("{m}.experts.down_proj_bias"))?,
            },
        })
    }

    fn bytes(&self) -> usize {
        let a = &self.attention;
        let e = &self.experts;
        [
            &self.input_norm,
            &self.post_norm,
            &a.q_bias,
            &a.kv_bias,
            &a.o_bias,
            &a.sinks,
            &e.router,
            &e.router_bias,
            &e.gate_up_bias,
            &e.down_bias,
        ]
        .iter()
        .map(|t| t.elem_count() * 4)
        .sum()
    }

    fn projection_bytes(&self) -> usize {
        let a = &self.attention;
        let banks = self
            .experts
            .banks
            .as_ref()
            .map_or(0, |(g, d)| g.resident_bytes() + d.resident_bytes());
        banks
            + [&a.q, &a.kv, &a.o]
                .iter()
                .map(|p| p.resident_bytes())
                .sum::<usize>()
    }
}

struct Span<'a> {
    start: usize,
    len: usize,
    past: usize,
    addresses: Addresses,
    sequence: &'a SequenceForward<'a>,
}

/// Addressing shared by every layer of one packed pass.
struct Pass<'a, 'b> {
    pool: &'a StatePool,
    spans: &'a [Span<'b>],
    positions: &'a [usize],
}

/// Bound live hidden and projection buffers plus routed rows and split-attention partials.
fn workspace(config: &Config, tokens: usize, split: usize) -> Result<usize> {
    let t = &config.text;
    let query = checked_elements(&[t.num_attention_heads, t.head_dim])?;
    let kv = checked_elements(&[t.num_key_value_heads, t.head_dim])?;
    let activations = checked_elements(&[tokens, t.hidden_size * 6 + query * 4 + kv * 6, 4])?;
    let routed = checked_elements(&[
        tokens,
        t.num_experts_per_tok,
        t.intermediate_size * 5 + t.hidden_size * 2,
        4,
    ])?;
    let routing = checked_elements(&[tokens, t.num_experts_per_tok, 5, 4])?
        .checked_add(checked_elements(&[t.num_local_experts, tokens.max(1), 8])?)
        .ok_or_else(|| invalid("gpt-oss routing workspace overflow"))?;
    activations
        .checked_add(routed)
        .and_then(|n| n.checked_add(routing))
        .and_then(|n| n.checked_add(split))
        .ok_or_else(|| invalid("gpt-oss workspace overflow"))
}

impl GptOss {
    /// Plan packed weights and peak activations before loading anything.
    pub fn memory_requirements(
        config: &Config,
        options: &LoadOptions,
        tokens: usize,
        contexts: &[usize],
    ) -> Result<(ModelDescription, usize, usize)> {
        let description = config.description(
            format!("{}#{}", options.revision, options.weights.tag()),
            options.kv,
        )?;
        let plan = options.weights.plan(config)?;
        plan.validate(config)?;
        let accounted: std::collections::BTreeSet<String> = config
            .projection_layout()?
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(|(name, _)| name))
            .chain(["model.embed_tokens.weight".to_owned()])
            .collect();
        let placement = Placement::of(&plan, options)?;
        let mut weights = plan.bytes(config)?;
        if placement == Placement::Cached {
            let table = match plan.embedding {
                crate::models::Embedding::Table(precision) => {
                    precision.bytes(config.vocab_size(), config.text.hidden_size)?
                }
                crate::models::Embedding::Head => 0,
            };
            weights = weights
                .checked_sub(table)
                .ok_or_else(|| invalid("embedding table exceeds the planned weights"))?;
        }
        for (name, shape) in config.weight_shapes() {
            if accounted.contains(&name) {
                continue;
            }
            let bytes = if name.ends_with("_blocks") {
                match placement {
                    Placement::Dense => checked_elements(&shape)?
                        .checked_mul(8)
                        .ok_or_else(|| invalid("gpt-oss weight bytes overflow"))?,
                    Placement::Resident => checked_elements(&shape)?,
                    Placement::Cached => 0,
                }
            } else if name.ends_with("_scales") {
                if placement == Placement::Resident {
                    checked_elements(&shape)?
                } else {
                    0
                }
            } else {
                checked_elements(&shape)?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("gpt-oss weight bytes overflow"))?
            };
            weights = weights
                .checked_add(bytes)
                .ok_or_else(|| invalid("gpt-oss weight bytes overflow"))?;
        }
        #[cfg(feature = "metal")]
        let mut staging = 0;
        #[cfg(feature = "metal")]
        if let (Placement::Cached, Some(cache)) = (placement, options.expert_cache) {
            use tur_metal::experts::ExpertCache;
            let (experts, gate_up, down) = config.bank_shapes();
            let slots = ExpertCache::planned_slots(cache, experts, gate_up, down)?;
            let arena = checked_elements(&[slots, ExpertCache::expert_bytes(gate_up, down)?])?;
            weights = weights
                .checked_add(arena)
                .ok_or_else(|| invalid("gpt-oss weight bytes overflow"))?;
            staging = ExpertCache::staging_bytes(gate_up, down)?;
        }
        #[cfg(feature = "metal")]
        let split = if options.device.is_metal() {
            let kernels = tur_metal::kernels::Kernels::new(&options.device)?;
            Self::split_workspace(config, &kernels, options.kv, tokens, contexts)
        } else {
            0
        };
        #[cfg(not(feature = "metal"))]
        let split = {
            let _ = contexts;
            0
        };
        #[cfg(not(feature = "metal"))]
        let staging = 0;
        let scratch = workspace(config, tokens, split)?
            .checked_add(staging)
            .ok_or_else(|| invalid("gpt-oss workspace overflow"))?;
        Ok((description, weights, scratch))
    }

    #[cfg(feature = "metal")]
    fn split_workspace(
        config: &Config,
        kernels: &tur_metal::kernels::Kernels,
        kv: tur_core::execution::KvLayout,
        tokens: usize,
        contexts: &[usize],
    ) -> usize {
        let t = &config.text;
        kernels.attention_scratch_bytes(
            tokens,
            t.num_attention_heads,
            t.num_key_value_heads,
            t.head_dim,
            contexts.iter().copied().max().unwrap_or(0),
            kv.format,
        )
    }

    /// Validate headers, then load the text tower onto `options.device`.
    pub fn load(dir: &Path, options: &LoadOptions) -> Result<Self> {
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if matches!(options.weights, Weights::Prepacked(_)) || artifact::is_artifact(dir) {
            return Err(invalid(
                "gpt-oss loads its SafeTensors directly; it has no prepacked artifacts",
            ));
        }
        if options.mtp.is_some() {
            return Err(invalid("gpt-oss has no draft head"));
        }
        let plan = options.weights.plan(&config)?;
        plan.validate(&config)?;
        let placement = Placement::of(&plan, options)?;
        let device = &options.device;
        let tensors = Tensors {
            #[cfg(feature = "metal")]
            expert_cache: None,
            checkpoint: checkpoint(dir, &config)?,
            prepacked: false,
            plan,
            projections: config
                .projection_layout()?
                .into_iter()
                .map(|p| (p.name.clone(), p))
                .collect(),
            device: device.clone(),
        };
        let mut loading = Stage::begin("Loading weights", config.layers() as u64, "layers");
        let layers = (0..config.layers())
            .map(|i| {
                let layer = Layer::load(&tensors, i, placement);
                loading.advance(1);
                layer
            })
            .collect::<Result<Vec<_>>>()?;
        loading.finish();
        #[cfg(feature = "metal")]
        let cache = options
            .expert_cache
            .map(|cache| {
                tur_metal::experts::ExpertCache::new(
                    &config.expert_sources(&tensors.checkpoint)?,
                    cache,
                    0,
                    device,
                )
            })
            .transpose()?;
        #[cfg(feature = "metal")]
        let (frequencies, amplitude) = config.yarn();
        let width = config.text.hidden_size;
        Self {
            description: config.description(
                format!("{}#{}", options.revision, options.weights.tag()),
                options.kv,
            )?,
            device: device.clone(),
            embedding: if placement == Placement::Cached {
                Embedding::OnDisk(checkpoint(dir, &config)?)
            } else {
                Embedding::Resident(tensors.embedding(
                    "model.embed_tokens.weight",
                    config.vocab_size(),
                    width,
                )?)
            },
            head: tensors.projection("lm_head")?,
            norm: f32_tensor(&tensors, "model.norm.weight")?,
            #[cfg(feature = "metal")]
            inv_freq: Tensor::new(frequencies.as_slice(), device)?,
            #[cfg(feature = "metal")]
            amplitude: Tensor::new(&[amplitude], device)?,
            layers,
            config,
            #[cfg(feature = "metal")]
            kernels: None,
            #[cfg(feature = "metal")]
            cache,
            #[cfg(feature = "metal")]
            probe: parking_lot::Mutex::new(None),
            #[cfg(feature = "metal")]
            hidden_experts: parking_lot::Mutex::new(None),
        }
        .with_execution(options.execution)
    }

    /// Select Metal kernels or the F32 Candle reference. MXFP4 experts run only on the
    /// kernels, so a reference pass over them fails rather than expanding them.
    pub fn with_execution(self, execution: Execution) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let kernels = if execution == Execution::Fused && self.device.is_metal() {
                Some(tur_metal::kernels::Kernels::new(&self.device)?)
            } else {
                None
            };
            Ok(Self { kernels, ..self })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = execution;
            Ok(self)
        }
    }

    /// Device holding the immutable weights.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Validated settings.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Whether Metal kernels or Candle reference operations execute passes.
    pub fn execution(&self) -> Execution {
        #[cfg(feature = "metal")]
        if self.kernels.is_some() {
            return Execution::Fused;
        }
        Execution::Reference
    }

    /// Resident bytes held by projections and expert banks or the expert cache.
    pub fn projection_bytes(&self) -> usize {
        #[cfg(feature = "metal")]
        let cache = self
            .cache
            .as_ref()
            .map_or(0, tur_metal::experts::ExpertCache::resident_bytes);
        #[cfg(not(feature = "metal"))]
        let cache = 0;
        cache
            + self.head.resident_bytes()
            + self
                .layers
                .iter()
                .map(Layer::projection_bytes)
                .sum::<usize>()
    }

    /// What the expert cache has done, when this load has one.
    #[cfg(feature = "metal")]
    pub fn expert_stats(&self) -> Option<tur_metal::experts::ExpertStats> {
        self.cache
            .as_ref()
            .map(tur_metal::experts::ExpertCache::stats)
    }

    /// Resident bytes loaded onto the device.
    pub fn weight_bytes(&self) -> usize {
        self.projection_bytes()
            + match &self.embedding {
                Embedding::Resident(table) => table.bytes(),
                Embedding::OnDisk(_) => 0,
            }
            + self.norm.elem_count() * 4
            + self.layers.iter().map(Layer::bytes).sum::<usize>()
    }

    fn check<'a>(
        &self,
        batch: &PackedForward<'_>,
        sequence: &'a SequenceForward<'a>,
        start: usize,
    ) -> Result<Span<'a>> {
        let tokens = batch
            .tokens
            .get(sequence.tokens.clone())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("empty or invalid sequence range"))?;
        let positions = batch
            .positions
            .get(sequence.tokens.clone())
            .ok_or_else(|| invalid("missing packed positions"))?;
        let past = sequence.state.past();
        let end = past
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("position overflow"))?;
        if end > self.config.max_position()
            || sequence.state.tokens() != tokens.len()
            || positions
                .iter()
                .enumerate()
                .any(|(i, &p)| past.checked_add(i) != Some(p))
            || tokens
                .iter()
                .any(|id| *id as usize >= self.config.vocab_size())
            || sequence.logits.iter().any(|r| !sequence.tokens.contains(r))
            || sequence.logits.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(invalid(
                "invalid gpt-oss positions, tokens, state or logit rows",
            ));
        }
        let addresses = sequence.state.resolve(batch.pool)?;
        let pages = self.description.pages_for(end);
        if (self.description.has_full_attention() && addresses.pages.len() != pages)
            || (self.description.window().is_some() && addresses.windows.len() != pages)
        {
            return Err(invalid("sequence pages do not cover new tokens"));
        }
        Ok(Span {
            start,
            len: tokens.len(),
            past,
            addresses,
            sequence,
        })
    }

    fn run(
        &self,
        batch: &PackedForward<'_>,
        spans: &[Span<'_>],
        trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Vec<SequenceStep>> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for s in spans {
            tokens.extend_from_slice(&batch.tokens[s.sequence.tokens.clone()]);
            positions.extend_from_slice(&batch.positions[s.sequence.tokens.clone()]);
        }
        let x = match &self.embedding {
            Embedding::Resident(table) => {
                table.rows(&Tensor::new(tokens.as_slice(), &self.device)?, &self.head)?
            }
            Embedding::OnDisk(checkpoint) => checkpoint
                .gather_rows("model.embed_tokens.weight", &tokens)?
                .to_dtype(DType::F32)?
                .to_device(&self.device)?,
        };
        let pass = Pass {
            pool: batch.pool,
            spans,
            positions: &positions,
        };
        #[cfg(feature = "metal")]
        let hidden = match &self.kernels {
            Some(k) => self.run_fused(k, &pass, x, trace)?,
            None => self.run_reference(&pass, x, trace)?,
        };
        #[cfg(not(feature = "metal"))]
        let hidden = self.run_reference(&pass, x, trace)?;
        spans
            .iter()
            .map(|s| {
                let rows = s
                    .sequence
                    .logits
                    .iter()
                    .map(|r| {
                        u32::try_from(s.start + r - s.sequence.tokens.start)
                            .map_err(|_| invalid("logit row overflow"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let logits = if rows.is_empty() {
                    Tensor::zeros((0, self.config.vocab_size()), DType::F32, &self.device)?
                } else {
                    let selected =
                        hidden.index_select(&Tensor::new(rows.as_slice(), &self.device)?, 0)?;
                    self.linear(&self.head, &selected)?
                };
                Ok(SequenceStep {
                    logits,
                    hidden: s
                        .sequence
                        .hidden
                        .then(|| hidden.narrow(0, s.start, s.len))
                        .transpose()?,
                })
            })
            .collect()
    }

    fn linear(&self, w: &PackedLinear, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(k) = &self.kernels {
            return k.linear(w, x);
        }
        w.forward(x)
    }

    /// Reference decoder over Candle operations; returns the final-normed hidden rows.
    fn run_reference(
        &self,
        pass: &Pass<'_, '_>,
        mut x: Tensor,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let eps = self.config.text.rms_norm_eps;
        for (i, layer) in self.layers.iter().enumerate() {
            let h = math::rms(&x, &layer.input_norm, eps)?;
            x = (x + self.reference_attention(layer, i, &h, pass)?)?;
            let h = math::rms(&x, &layer.post_norm, eps)?;
            x = (x + self.reference_experts(&layer.experts, &h)?)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        math::rms(&x, &self.norm, eps)
    }

    fn reference_attention(
        &self,
        layer: &Layer,
        index: usize,
        h: &Tensor,
        pass: &Pass<'_, '_>,
    ) -> Result<Tensor> {
        let t = &self.config.text;
        let a = &layer.attention;
        let tokens = h.dim(0)?;
        let (heads, kv_heads, dim) = (t.num_attention_heads, t.num_key_value_heads, t.head_dim);
        let (cos, sin) = self.rotary_tables(pass.positions)?;
        let query =
            a.q.forward(h)?
                .broadcast_add(&a.q_bias)?
                .reshape((tokens, heads, dim))?;
        let query = math::rotary(&query, &cos, &sin)?;
        let kv = a.kv.forward(h)?.broadcast_add(&a.kv_bias)?;
        let width = kv_heads * dim;
        let shape = (tokens, kv_heads, dim);
        let key = math::rotary(&kv.narrow(1, 0, width)?.reshape(shape)?, &cos, &sin)?;
        let value = kv.narrow(1, width, width)?.reshape(shape)?;
        let windowed = self.config.windowed(index);
        let store = pass.pool.attention(index)?;
        for s in pass.spans {
            store.scatter(
                table(&s.addresses, windowed),
                s.past,
                &key.narrow(0, s.start, s.len)?,
                &value.narrow(0, s.start, s.len)?,
            )?;
        }
        let window = windowed.then_some(t.sliding_window);
        let outputs = pass
            .spans
            .iter()
            .map(|s| {
                let (keys, values) = store.gather(table(&s.addresses, windowed), s.past + s.len)?;
                math::attention(
                    &query.narrow(0, s.start, s.len)?,
                    &keys,
                    &values,
                    s.past,
                    (dim as f64).sqrt().recip(),
                    window,
                    Some(&a.sinks),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let merged = Tensor::cat(&outputs, 0)?.reshape((tokens, heads * dim))?;
        Ok(a.o.forward(&merged)?.broadcast_add(&a.o_bias)?)
    }

    fn rotary_tables(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let (frequencies, amplitude) = self.config.yarn();
        let mut cos = Vec::with_capacity(positions.len() * frequencies.len());
        let mut sin = Vec::with_capacity(cos.capacity());
        for &position in positions {
            for &frequency in &frequencies {
                let angle = position as f32 * frequency;
                cos.push(angle.cos() * amplitude);
                sin.push(angle.sin() * amplitude);
            }
        }
        let shape = (positions.len(), frequencies.len());
        Ok((
            Tensor::from_vec(cos, shape, &self.device)?,
            Tensor::from_vec(sin, shape, &self.device)?,
        ))
    }

    /// Only routing decisions cross to the host. Each selected expert projects all of its
    /// tokens at once, and outputs accumulate in expert ID order.
    fn reference_experts(&self, e: &Experts, h: &Tensor) -> Result<Tensor> {
        let t = &self.config.text;
        let (experts, inner, hidden) = (t.num_local_experts, t.intermediate_size, t.hidden_size);
        let logits = h.matmul(&e.router.t()?)?.broadcast_add(&e.router_bias)?;
        let mut groups = vec![(Vec::new(), Vec::new()); experts];
        for (token, scores) in logits.to_vec2::<f32>()?.iter().enumerate() {
            let (selected, weights) = math::route(scores, t.num_experts_per_tok)?;
            let token = u32::try_from(token).map_err(|_| invalid("too many expert tokens"))?;
            for (expert, weight) in selected.into_iter().zip(weights) {
                groups[expert].0.push(token);
                groups[expert].1.push(weight);
            }
        }
        let mut output = Tensor::zeros(h.dims(), DType::F32, h.device())?;
        for (expert, (tokens, weights)) in groups.into_iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let count = tokens.len();
            let ids = Tensor::from_vec(tokens, count, h.device())?;
            let rows = h.index_select(&ids, 0)?;
            let (bank_gate_up, bank_down) = e
                .banks
                .as_ref()
                .ok_or_else(|| invalid("the reference needs resident experts"))?;
            let gate_up = bank_gate_up
                .forward_rows(&rows, expert * 2 * inner, 2 * inner)?
                .broadcast_add(&e.gate_up_bias.narrow(0, expert, 1)?)?;
            let act = reference_glu(&gate_up, t.swiglu_limit)?;
            let routed = bank_down
                .forward_rows(&act, expert * hidden, hidden)?
                .broadcast_add(&e.down_bias.narrow(0, expert, 1)?)?
                .broadcast_mul(&Tensor::from_vec(weights, (count, 1), h.device())?)?;
            output = output.index_add(&ids, &routed, 0)?;
        }
        Ok(output)
    }

    /// Run a fresh sequence and return per-layer residual streams and selected logits.
    pub fn trace(&self, tokens: &[u32], rows: &[usize]) -> Result<(Vec<Tensor>, Tensor)> {
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
        let binding = reservation.binding();
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let sequences = [SequenceForward {
            tokens: 0..tokens.len(),
            state: &binding,
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
        let spans = [self.check(&batch, &sequences[0], 0)?];
        let mut trace = Vec::new();
        let step = self
            .run(&batch, &spans, Some(&mut trace))?
            .pop()
            .ok_or_else(|| invalid("missing gpt-oss output"))?;
        Ok((trace, step.logits))
    }
}

impl GptOss {
    /// What every expert would output for every token of `samples`, reduced to what a
    /// pruning search needs: see [`ExpertProbe`]. Offline only: it runs every expert of every
    /// layer, so each sample reads the whole expert set.
    #[cfg(not(feature = "metal"))]
    pub fn expert_probe(&self, _samples: &[Vec<u32>]) -> Result<ExpertProbe> {
        Err(invalid("expert outputs are measured on the Metal kernels"))
    }

    /// Hide every expert outside `kept` from the routers. Pruning searches need the kernels.
    #[cfg(not(feature = "metal"))]
    pub fn hide_experts(&self, _kept: Option<&[Vec<usize>]>) -> Result<()> {
        Err(invalid("experts are hidden on the Metal kernels"))
    }

    /// What every expert would output for every token of `samples`, reduced to what a
    /// pruning search needs: see [`ExpertProbe`]. Offline only: it runs every expert of every
    /// layer, so each sample reads the whole expert set.
    #[cfg(feature = "metal")]
    pub fn expert_probe(&self, samples: &[Vec<u32>]) -> Result<ExpertProbe> {
        if self.kernels.is_none() {
            return Err(invalid("expert outputs are measured on the Metal kernels"));
        }
        let t = &self.config.text;
        *self.probe.lock() = Some(ExpertProbe {
            experts: t.num_local_experts,
            top_k: t.num_experts_per_tok,
            layers: vec![LayerProbe::default(); self.layers.len()],
        });
        let mut stage = Stage::begin("Probing experts", samples.len() as u64, "samples");
        let result = samples.iter().try_for_each(|sample| {
            let last = sample
                .len()
                .checked_sub(1)
                .ok_or_else(|| invalid("empty calibration sample"))?;
            self.trace(sample, &[last]).map(drop)?;
            stage.advance(1);
            Ok::<(), tur_core::TurError>(())
        });
        if result.is_ok() {
            stage.finish();
        }
        let probe = self
            .probe
            .lock()
            .take()
            .ok_or_else(|| invalid("expert outputs were not recorded"))?;
        result?;
        Ok(probe)
    }

    /// Hide every expert outside `kept` from the routers, which routes exactly as removing
    /// those experts from the checkpoint does, without writing one. `None` restores them all.
    #[cfg(feature = "metal")]
    pub fn hide_experts(&self, kept: Option<&[Vec<usize>]>) -> Result<()> {
        // Far below any router score, yet finite as the routing kernels require.
        const HIDDEN: f32 = -1e9;
        let experts = self.config.text.num_local_experts;
        let penalties = kept
            .map(|layers| {
                layers
                    .iter()
                    .map(|kept| {
                        let mut row = vec![HIDDEN; experts];
                        for &expert in kept {
                            *row.get_mut(expert)
                                .ok_or_else(|| invalid("kept expert outside the bank"))? = 0.0;
                        }
                        Ok(Tensor::new(row.as_slice(), &self.device)?)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        *self.hidden_experts.lock() = penalties;
        Ok(())
    }

    /// Log-probabilities at every fourth position of `sample`, teacher-forced.
    pub fn log_probs(&self, sample: &[u32]) -> Result<Vec<Vec<f32>>> {
        let rows: Vec<usize> = (0..sample.len()).step_by(4).collect();
        let (_, logits) = self.trace(sample, &rows)?;
        Ok(candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?.to_vec2::<f32>()?)
    }

    /// Run all of `index`'s experts on `h` and add their overlaps to the probe.
    #[cfg(feature = "metal")]
    fn record_probe(
        &self,
        k: &tur_metal::kernels::Kernels,
        index: usize,
        e: &Experts,
        h: &Tensor,
        logits: &Tensor,
    ) -> Result<()> {
        let t = &self.config.text;
        let experts = t.num_local_experts;
        let routes = k.route(logits, experts)?;
        let (routes, gate_up_bank, down_bank) = match (&self.cache, &e.banks) {
            (Some(cache), _) => (
                cache.route(k, index, routes, None)?,
                cache.gate_up(),
                cache.down(),
            ),
            (None, Some((gate_up, down))) => (routes, gate_up, down),
            (None, None) => return Err(invalid("experts have no weights")),
        };
        let gate_up = k.expert_linear_routed(gate_up_bank, h, &routes)?;
        let act = k.expert_glu(
            &gate_up,
            &e.gate_up_bias,
            &routes,
            SWIGLU_ALPHA,
            t.swiglu_limit,
        )?;
        let out = k.expert_bias(
            &k.expert_linear(down_bank, &act, &routes)?,
            &e.down_bias,
            &routes,
        )?;
        let rows = routes.assignment_rows()?;
        let values = out.flatten_all()?.to_vec1::<f32>()?;
        let scores = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut probe = self.probe.lock();
        let layer = probe
            .as_mut()
            .and_then(|p| p.layers.get_mut(index))
            .ok_or_else(|| invalid("expert probe has no such layer"))?;
        layer.add(experts, t.hidden_size, &rows, &values, &scores)
    }
}

/// `(clamp(up) + 1) * gate * sigmoid(gate * alpha)` over interleaved gate|up columns.
fn reference_glu(gate_up: &Tensor, limit: f32) -> Result<Tensor> {
    let (rows, cols) = gate_up.dims2()?;
    let limit = f64::from(limit);
    let every = |offset: usize| -> Result<Tensor> {
        let ids: Vec<u32> = (offset..cols)
            .step_by(2)
            .map(|c| u32::try_from(c).map_err(|_| invalid("gate width overflow")))
            .collect::<Result<_>>()?;
        let count = ids.len();
        Ok(gate_up.index_select(&Tensor::from_vec(ids, count, gate_up.device())?, 1)?)
    };
    debug_assert_eq!(rows, gate_up.dim(0)?);
    let gate = every(0)?.clamp(f64::NEG_INFINITY, limit)?;
    let up = every(1)?.clamp(-limit, limit)?;
    let sigmoid = candle_nn::ops::sigmoid(&(&gate * f64::from(SWIGLU_ALPHA))?)?;
    Ok(((up + 1.0)? * (gate * sigmoid)?)?)
}

/// The page table a layer's keys live in: window pages for windowed layers.
fn table(addresses: &Addresses, windowed: bool) -> &[u32] {
    if windowed {
        &addresses.windows
    } else {
        &addresses.pages
    }
}

#[cfg(feature = "metal")]
impl GptOss {
    /// Fused decoder: every layer runs once over the packed batch, with residual updates
    /// fused into the following norm. Returns the final-normed rows.
    fn run_fused(
        &self,
        k: &tur_metal::kernels::Kernels,
        pass: &Pass<'_, '_>,
        mut x: Tensor,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let eps = self.config.text.rms_norm_eps;
        let tokens = x.dim(0)?;
        let paging = Paging::new(
            k,
            &self.description,
            pass.pool,
            pass.spans.iter().map(|s| (&s.addresses, s.past, s.len)),
            tokens,
        )?;
        let positions = pass
            .positions
            .iter()
            .map(|&p| u32::try_from(p).map_err(|_| invalid("position exceeds u32")))
            .collect::<Result<Vec<_>>>()?;
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        let first = self
            .layers
            .first()
            .ok_or_else(|| invalid("model has no layers"))?;
        let mut h = k.rms_norm(&x, &first.input_norm, eps)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let attended = self.fused_attention(k, layer, i, &h, &positions, pass.pool, &paging)?;
            let (sum, normed) = k.add_rms_norm(&x, &attended, &layer.post_norm, eps)?;
            let routed = self.fused_experts(k, i, &layer.experts, &normed)?;
            let next = self.layers.get(i + 1).map_or(&self.norm, |n| &n.input_norm);
            (x, h) = k.add_rms_norm(&sum, &routed, next, eps)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        Ok(h)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per input the attention dispatch binds"
    )]
    fn fused_attention(
        &self,
        k: &tur_metal::kernels::Kernels,
        layer: &Layer,
        index: usize,
        h: &Tensor,
        positions: &Tensor,
        pool: &StatePool,
        paging: &Paging,
    ) -> Result<Tensor> {
        let t = &self.config.text;
        let a = &layer.attention;
        let (heads, kv_heads, dim) = (t.num_attention_heads, t.num_key_value_heads, t.head_dim);
        let at = |count: usize, col_offset: usize| tur_metal::kernels::Heads {
            count,
            dim,
            head_stride: dim,
            col_offset,
        };
        let windowed = self.config.windowed(index);
        let paged = paging.layer(windowed)?;
        let query = k.rope_scaled(
            &k.linear(&a.q, h)?.broadcast_add(&a.q_bias)?,
            at(heads, 0),
            &self.amplitude,
            positions,
            &self.inv_freq,
        )?;
        let kv = k.linear(&a.kv, h)?.broadcast_add(&a.kv_bias)?;
        let width = kv_heads * dim;
        let key = k.rope_scaled(
            &kv,
            at(kv_heads, 0),
            &self.amplitude,
            positions,
            &self.inv_freq,
        )?;
        let store = pool.attention(index)?;
        k.kv_write(
            &key.reshape((key.dim(0)?, width))?,
            at(kv_heads, 0),
            &kv,
            at(kv_heads, width),
            store,
            paged,
        )?;
        let window = windowed.then_some(t.sliding_window);
        let attended = k.attention_sinks(
            &query,
            store,
            paged,
            window,
            (dim as f64).sqrt().recip(),
            &a.sinks,
        )?;
        Ok(k.linear(&a.o, &attended)?.broadcast_add(&a.o_bias)?)
    }

    fn fused_experts(
        &self,
        k: &tur_metal::kernels::Kernels,
        index: usize,
        e: &Experts,
        h: &Tensor,
    ) -> Result<Tensor> {
        let t = &self.config.text;
        let logits = h.matmul(&e.router.t()?)?.broadcast_add(&e.router_bias)?;
        let hidden = self.hidden_experts.lock();
        let hide = |layer: usize, logits: Tensor| -> Result<Tensor> {
            Ok(match hidden.as_ref().and_then(|h| h.get(layer)) {
                Some(penalty) => logits.broadcast_add(penalty)?,
                None => logits,
            })
        };
        let logits = hide(index, logits)?;
        if self.probe.lock().is_some() {
            self.record_probe(k, index, e, h, &logits)?;
        }
        let routes = k.route(&logits, t.num_experts_per_tok)?;
        // Encoded before the routes are read, so reading them waits for this too.
        let next_logits = match (&self.cache, self.layers.get(index + 1)) {
            (Some(_), Some(next)) => Some(hide(
                index + 1,
                h.matmul(&next.experts.router.t()?)?
                    .broadcast_add(&next.experts.router_bias)?,
            )?),
            _ => None,
        };
        let (routes, gate_up_bank, down_bank) = match (&self.cache, &e.banks) {
            (Some(cache), _) => (
                cache.route(k, index, routes, next_logits.as_ref())?,
                cache.gate_up(),
                cache.down(),
            ),
            (None, Some((gate_up, down))) => (routes, gate_up, down),
            (None, None) => return Err(invalid("experts have no weights")),
        };
        let gate_up = k.expert_linear_routed(gate_up_bank, h, &routes)?;
        let act = k.expert_glu(
            &gate_up,
            &e.gate_up_bias,
            &routes,
            SWIGLU_ALPHA,
            t.swiglu_limit,
        )?;
        let out = k.expert_linear(down_bank, &act, &routes)?;
        k.expert_scatter(&k.expert_bias(&out, &e.down_bias, &routes)?, &routes)
    }
}

impl HybridModel for GptOss {
    fn description(&self) -> ModelDescription {
        self.description.clone()
    }
    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> {
        #[cfg(feature = "metal")]
        let split = match &self.kernels {
            Some(kernels) => {
                let contexts = batch
                    .sequences
                    .iter()
                    .map(|s| {
                        s.state
                            .past()
                            .checked_add(s.tokens.len())
                            .ok_or_else(|| invalid("context overflow"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Self::split_workspace(
                    &self.config,
                    kernels,
                    self.description.kv,
                    batch.tokens.len(),
                    &contexts,
                )
            }
            None => 0,
        };
        #[cfg(not(feature = "metal"))]
        let split = 0;
        workspace(&self.config, batch.tokens.len(), split)
    }
    #[cfg(feature = "metal")]
    fn kernels(&self) -> Option<tur_metal::kernels::Kernels> {
        self.kernels.clone()
    }
    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>> {
        if batch.pool.description() != &self.description
            || !batch.pool.device().same_device(&self.device)
        {
            return Err(invalid("state pool belongs to another model or device"));
        }
        let mut out: Vec<Option<Result<SequenceStep>>> =
            batch.sequences.iter().map(|_| None).collect();
        let mut spans = Vec::new();
        let mut live = Vec::new();
        let mut start = 0;
        for (i, sequence) in batch.sequences.iter().enumerate() {
            match self.check(batch, sequence, start) {
                Ok(s) => {
                    start += s.len;
                    spans.push(s);
                    live.push(i);
                }
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        if !spans.is_empty() {
            for (i, step) in live.into_iter().zip(self.run(batch, &spans, None)?) {
                out[i] = Some(Ok(step));
            }
        }
        Ok(out
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("sequence was not executed"))))
            .collect())
    }
}
