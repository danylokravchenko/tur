//! Gemma 4 equations over the common packed weights and persistent state pools.
use super::{Config, config::LayerKind};
use crate::{
    hybrid::{
        Addresses, Boundaries, HybridModel, PackedForward, SequenceForward, SequenceStep, StatePool,
    },
    models::{
        Execution, LoadOptions, WeightPlan, Weights,
        checkpoint::{Checkpoint, weight_files},
        math,
        weights::{Embeddings, Tensors, WeightLayout},
    },
};
use candle_core::{DType, Device, Tensor};
use std::{collections::BTreeSet, path::Path, sync::Arc};
use tur_core::{
    Result,
    execution::{ModelDescription, checked_elements, invalid},
};
use tur_metal::{packed::PackedLinear, pool::PoolConfig, state::ModelState};
mod moe;

struct Layer {
    q: PackedLinear,
    k: Option<PackedLinear>,
    v: Option<PackedLinear>,
    o: PackedLinear,
    q_norm: Tensor,
    k_norm: Option<Tensor>,
    input_norm: Tensor,
    post_attention: Tensor,
    pre_ff: Tensor,
    post_ff: Tensor,
    scalar: Tensor,
    gate: PackedLinear,
    up: PackedLinear,
    down: PackedLinear,
    ple: Option<LayerInput>,
    moe: Option<moe::Moe>,
}
struct LayerInput {
    gate: PackedLinear,
    projection: PackedLinear,
    norm: Tensor,
}
struct PerLayer {
    table: Vec<Tensor>,
    projection: PackedLinear,
    norm: Tensor,
}

/// Gemma 4 text tower; immutable weights, with all request state in the engine's pools.
pub struct Gemma4 {
    config: Config,
    description: ModelDescription,
    device: Device,
    embedding: Embeddings,
    head: PackedLinear,
    norm: Tensor,
    layers: Vec<Layer>,
    ple: Option<PerLayer>,
    #[cfg(feature = "metal")]
    kernels: Option<tur_metal::kernels::Kernels>,
}

fn format_plan(config: &Config, options: &LoadOptions) -> Result<WeightPlan> {
    if options.mtp.is_some() {
        return Err(invalid("Gemma 4 native draft heads are not implemented"));
    }
    if matches!(options.weights, Weights::Prepacked(_)) {
        return Err(invalid("Gemma 4 prepacked artifacts are not implemented"));
    }
    let plan = options.weights.plan(config)?;
    plan.validate(config)?;
    Ok(plan)
}

fn checkpoint(dir: &Path, config: &Config) -> Result<Checkpoint> {
    let (checkpoint, _) = Checkpoint::scan(weight_files(dir)?)?;
    let shapes = config.weight_shapes()?;
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
        if !matches!(
            entry.candle_dtype(name)?,
            DType::F32 | DType::BF16 | DType::F16
        ) {
            return Err(invalid("unsupported Gemma 4 tensor dtype"));
        }
    }
    for (name, entry) in checkpoint.entries() {
        if !name.starts_with(&format!("{}.", config.prefix)) || shapes.contains_key(name) {
            continue;
        }
        // Published checkpoints retain unused K/V matrices in the sharing suffix.
        let mut known = false;
        for i in 0..config.layers() {
            if config.owners[i] == i {
                continue;
            }
            let prefix = format!("{}.layers.{i}.self_attn", config.prefix);
            let expected = if name == &format!("{prefix}.k_norm.weight") {
                Some(vec![config.dim(i)])
            } else if name == &format!("{prefix}.k_proj.weight")
                || name == &format!("{prefix}.v_proj.weight")
            {
                Some(vec![
                    config.kv_heads(i) * config.dim(i),
                    config.hidden_size(),
                ])
            } else {
                None
            };
            if let Some(expected) = expected {
                if entry.shape != expected {
                    return Err(invalid(format!("invalid unused shared K/V tensor {name}")));
                }
                entry.candle_dtype(name)?;
                known = true;
                break;
            }
        }
        if !known {
            return Err(invalid(format!("unexpected text tensor {name}")));
        }
    }
    Ok(checkpoint)
}

impl Config {
    fn layers(&self) -> usize {
        self.text.num_hidden_layers
    }
}
impl Layer {
    fn load(t: &Tensors, c: &Config, i: usize) -> Result<Self> {
        let p = format!("{}.layers.{i}", c.prefix);
        let projection = |s: &str| t.projection(&format!("{p}.{s}"));
        let norm = |s: &str| -> Result<Tensor> {
            Ok(t.take(&format!("{p}.{s}.weight"))?
                .to_dtype(DType::F32)?
                .to_device(&t.device)?)
        };
        let own = c.owners[i] == i;
        Ok(Self {
            moe: if c.text.enable_moe_block {
                Some(moe::Moe::load(
                    t,
                    &p,
                    c.text
                        .num_experts
                        .ok_or_else(|| invalid("missing expert count"))?,
                    c.text
                        .top_k_experts
                        .ok_or_else(|| invalid("missing top-k"))?,
                )?)
            } else {
                None
            },
            q: projection("self_attn.q_proj")?,
            k: own.then(|| projection("self_attn.k_proj")).transpose()?,
            v: (own && !c.unified_kv(i))
                .then(|| projection("self_attn.v_proj"))
                .transpose()?,
            o: projection("self_attn.o_proj")?,
            q_norm: norm("self_attn.q_norm")?,
            k_norm: own.then(|| norm("self_attn.k_norm")).transpose()?,
            input_norm: norm("input_layernorm")?,
            post_attention: norm("post_attention_layernorm")?,
            pre_ff: norm("pre_feedforward_layernorm")?,
            post_ff: norm("post_feedforward_layernorm")?,
            scalar: t
                .take(&format!("{p}.layer_scalar"))?
                .to_dtype(DType::F32)?
                .to_device(&t.device)?,
            gate: projection("mlp.gate_proj")?,
            up: projection("mlp.up_proj")?,
            down: projection("mlp.down_proj")?,
            ple: (c.text.hidden_size_per_layer_input > 0)
                .then(|| -> Result<LayerInput> {
                    Ok(LayerInput {
                        gate: projection("per_layer_input_gate")?,
                        projection: projection("per_layer_projection")?,
                        norm: norm("post_per_layer_input_norm")?,
                    })
                })
                .transpose()?,
        })
    }
}

impl Gemma4 {
    /// Load only the validated text tower; multimodal inputs and native drafting are rejected.
    pub fn load(dir: &Path, options: &LoadOptions) -> Result<Self> {
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let plan = format_plan(&config, options)?;
        let checkpoint = checkpoint(dir, &config)?;
        let t = Tensors {
            checkpoint,
            prepacked: false,
            plan,
            projections: config
                .projection_layout()?
                .into_iter()
                .map(|p| (p.name.clone(), p))
                .collect(),
            device: options.device.clone(),
        };
        let head = t.projection("lm_head")?;
        let embedding = t.embedding(
            &format!("{}.embed_tokens.weight", config.prefix),
            config.vocab_size(),
            config.hidden_size(),
        )?;
        let norm = t
            .take(&format!("{}.norm.weight", config.prefix))?
            .to_dtype(DType::F32)?
            .to_device(&options.device)?;
        let layers = (0..config.layers())
            .map(|i| Layer::load(&t, &config, i))
            .collect::<Result<Vec<_>>>()?;
        let ple = if config.text.hidden_size_per_layer_input > 0 {
            let name = format!("{}.embed_tokens_per_layer.weight", config.prefix);
            Some(PerLayer {
                table: t.column_tables(
                    &name,
                    config.text.vocab_size_per_layer_input,
                    config.text.hidden_size_per_layer_input,
                    config.layers(),
                    ple_dtype(&options.weights),
                )?,
                projection: t
                    .projection(&format!("{}.per_layer_model_projection", config.prefix))?,
                norm: t
                    .take(&format!(
                        "{}.per_layer_projection_norm.weight",
                        config.prefix
                    ))?
                    .to_dtype(DType::F32)?
                    .to_device(&options.device)?,
            })
        } else {
            None
        };
        let (description, _, _) = Self::memory_requirements(&config, options, 1, &[1])?;
        Self {
            config,
            description,
            device: options.device.clone(),
            embedding,
            head,
            norm,
            layers,
            ple,
            #[cfg(feature = "metal")]
            kernels: None,
        }
        .with_execution(options.execution)
    }

    /// Select common Metal kernels or the F32 Candle reference, retaining the same weights.
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
    /// Validated Gemma 4 settings.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Resident bytes loaded into the device.
    pub fn weight_bytes(&self) -> usize {
        let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let mut total = self.embedding.bytes() + self.head.resident_bytes() + bytes(&self.norm);
        if let Some(ple) = &self.ple {
            total += ple.table.iter().map(bytes).sum::<usize>()
                + bytes(&ple.norm)
                + ple.projection.resident_bytes();
        }
        for l in &self.layers {
            total += [&l.q, &l.o, &l.gate, &l.up, &l.down]
                .iter()
                .map(|p| p.resident_bytes())
                .sum::<usize>();
            total += l.k.as_ref().map_or(0, PackedLinear::resident_bytes)
                + l.v.as_ref().map_or(0, PackedLinear::resident_bytes);
            total += [
                &l.q_norm,
                &l.input_norm,
                &l.post_attention,
                &l.pre_ff,
                &l.post_ff,
                &l.scalar,
            ]
            .iter()
            .map(|t| bytes(t))
            .sum::<usize>()
                + l.k_norm.as_ref().map_or(0, bytes);
            if let Some(ple) = &l.ple {
                total +=
                    ple.gate.resident_bytes() + ple.projection.resident_bytes() + bytes(&ple.norm);
            }
            if let Some(moe) = &l.moe {
                total += moe.bytes();
            }
        }
        total
    }

    /// Plan packed weights, per-layer embedding storage and peak activations before loading.
    pub fn memory_requirements(
        config: &Config,
        options: &LoadOptions,
        tokens: usize,
        contexts: &[usize],
    ) -> Result<(ModelDescription, usize, usize)> {
        let plan = format_plan(config, options)?;
        let description = config.description(
            format!("{}#{}", options.revision, options.weights.tag()),
            options.kv,
        )?;
        let projections = config.projection_layout()?;
        let mut accounted: BTreeSet<String> = projections
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(|(n, _)| n))
            .collect();
        accounted.insert(format!("{}.embed_tokens.weight", config.prefix));
        let mut weights = plan.bytes(config)?;
        for (name, shape) in config.weight_shapes()? {
            if accounted.contains(&name) {
                continue;
            }
            let dtype = if name.ends_with("embed_tokens_per_layer.weight") {
                ple_dtype(&options.weights)
            } else {
                DType::F32
            };
            weights = checked_elements(&shape)?
                .checked_mul(dtype.size_in_bytes())
                .and_then(|b| weights.checked_add(b))
                .ok_or_else(|| invalid("Gemma 4 weight bytes overflow"))?;
        }
        #[cfg(feature = "metal")]
        let attention = if options.execution == Execution::Fused && options.device.is_metal() {
            let kernels = tur_metal::kernels::Kernels::new(&options.device)?;
            Some(attention_workspace(
                config, &kernels, options.kv, tokens, contexts,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "metal"))]
        let attention = None;
        Ok((
            description,
            weights,
            workspace(config, tokens, contexts, attention)?,
        ))
    }
    fn linear(&self, w: &PackedLinear, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(k) = &self.kernels {
            return k.linear(w, x);
        }
        w.forward(x)
    }
    fn rms(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(k) = &self.kernels {
            let shape = x.shape().clone();
            let width = x.dim(candle_core::D::Minus1)?;
            return Ok(k
                .rms_norm(
                    &x.reshape((x.elem_count() / width, width))?,
                    w,
                    self.config.text.rms_norm_eps,
                )?
                .reshape(shape)?);
        }
        math::rms(x, w, self.config.text.rms_norm_eps)
    }
    fn inputs(&self, ids: &Tensor, embedded: &Tensor) -> Result<Option<Tensor>> {
        let Some(ple) = &self.ple else {
            return Ok(None);
        };
        let c = &self.config;
        let shape = (
            ids.elem_count(),
            c.layers(),
            c.text.hidden_size_per_layer_input,
        );
        let context = (self.linear(&ple.projection, embedded)? / (c.hidden_size() as f64).sqrt())?
            .reshape(shape)?;
        let context = self.rms(&context, &ple.norm)?;
        let gathered = ple
            .table
            .iter()
            .map(|table| table.index_select(ids, 0))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let identity = (Tensor::stack(&gathered, 1)?.to_dtype(DType::F32)?
            * (c.text.hidden_size_per_layer_input as f64).sqrt())?
        .reshape(shape)?;
        Ok(Some(
            ((context + identity)? * std::f64::consts::FRAC_1_SQRT_2)?,
        ))
    }
    fn run(
        &self,
        batch: &PackedForward<'_>,
        spans: &[Span<'_>],
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Vec<SequenceStep>> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for s in spans {
            tokens.extend_from_slice(&batch.tokens[s.sequence.tokens.clone()]);
            positions.extend_from_slice(&batch.positions[s.sequence.tokens.clone()]);
        }
        let ids = Tensor::new(tokens.as_slice(), &self.device)?;
        let mut x =
            (self.embedding.rows(&ids, &self.head)? * (self.config.hidden_size() as f64).sqrt())?;
        let inputs = self.inputs(&ids, &x)?;
        #[cfg(feature = "metal")]
        let paging = self.paging(batch.pool, spans, tokens.len())?;
        for (i, l) in self.layers.iter().enumerate() {
            let h = self.rms(&x, &l.input_norm)?;
            let attention = self.attention(
                l,
                i,
                &h,
                &positions,
                batch.pool,
                spans,
                #[cfg(feature = "metal")]
                paging.as_ref(),
            )?;
            x = (x + self.rms(&attention, &l.post_attention)?)?;
            let h = self.rms(&x, &l.pre_ff)?;
            let gated = (self.linear(&l.gate, &h)?.gelu()? * self.linear(&l.up, &h)?)?;
            let dense = self.linear(&l.down, &gated)?;
            let mixed = match &l.moe {
                Some(moe) => moe.forward(self, &x, &dense)?,
                None => dense,
            };
            x = (x + self.rms(&mixed, &l.post_ff)?)?;
            if let (Some(ple), Some(inputs)) = (&l.ple, &inputs) {
                let gate = self.linear(&ple.gate, &x)?.gelu()?;
                let signal = (gate * inputs.narrow(1, i, 1)?.squeeze(1)?)?;
                x = (&x + self.rms(&self.linear(&ple.projection, &signal)?, &ple.norm)?)?;
            }
            x = x.broadcast_mul(&l.scalar)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        let hidden = self.rms(&x, &self.norm)?;
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
                    let raw = self.linear(
                        &self.head,
                        &hidden.index_select(&Tensor::new(rows.as_slice(), &self.device)?, 0)?,
                    )?;
                    match self.config.text.final_logit_softcapping {
                        Some(cap) => ((raw / cap)?.tanh()? * cap)?,
                        None => raw,
                    }
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
    #[allow(
        clippy::too_many_arguments,
        reason = "attention combines immutable layer weights, packed rows and pool addressing"
    )]
    fn attention(
        &self,
        layer: &Layer,
        index: usize,
        hidden: &Tensor,
        positions: &[usize],
        pool: &StatePool,
        spans: &[Span<'_>],
        #[cfg(feature = "metal")] paging: Option<&tur_metal::kernels::PagedKv>,
    ) -> Result<Tensor> {
        let config = &self.config;
        let tokens = hidden.dim(0)?;
        let dim = config.dim(index);
        let heads = config.text.num_attention_heads;
        let (cos, sin) = rotary_tables(positions, dim, config.rope(index), &self.device)?;
        let query = self
            .linear(&layer.q, hidden)?
            .reshape((tokens, heads, dim))?;
        let query = math::rotary(&self.rms(&query, &layer.q_norm)?, &cos, &sin)?;
        let store = pool.attention(config.state_layer(index))?;
        if let (Some(kernels), Some(norm)) = (&layer.k, &layer.k_norm) {
            let raw =
                self.linear(kernels, hidden)?
                    .reshape((tokens, config.kv_heads(index), dim))?;
            let value = match &layer.v {
                Some(weights) => self.linear(weights, hidden)?.reshape(raw.shape())?,
                None => raw.clone(),
            };
            let value = math::rms(
                &value,
                &Tensor::ones(dim, DType::F32, &self.device)?,
                config.text.rms_norm_eps,
            )?;
            let key = math::rotary(&self.rms(&raw, norm)?, &cos, &sin)?;
            #[cfg(feature = "metal")]
            if let (Some(kernels), Some(paging)) = (&self.kernels, paging) {
                let layout = tur_metal::kernels::Heads {
                    count: config.kv_heads(index),
                    dim,
                    head_stride: dim,
                    col_offset: 0,
                };
                kernels.kv_write(
                    &key.reshape((tokens, config.kv_heads(index) * dim))?,
                    layout,
                    &value.reshape((tokens, config.kv_heads(index) * dim))?,
                    layout,
                    store,
                    paging,
                )?;
            } else {
                scatter(store, spans, &key, &value)?;
            }
            #[cfg(not(feature = "metal"))]
            scatter(store, spans, &key, &value)?;
        }
        let window = (config.text.layer_types[index] == LayerKind::SlidingAttention)
            .then_some(config.text.sliding_window);
        #[cfg(feature = "metal")]
        if let (Some(kernels), Some(paging)) = (&self.kernels, paging) {
            return self.linear(
                &layer.o,
                &kernels.attention_ungated(&query, store, paging, window, 1.0)?,
            );
        }
        let outputs = spans
            .iter()
            .map(|s| {
                let (keys, values) = store.gather(&s.addresses.pages, s.past + s.len)?;
                math::attention(
                    &query.narrow(0, s.start, s.len)?,
                    &keys,
                    &values,
                    s.past,
                    1.0,
                    window,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.linear(
            &layer.o,
            &Tensor::cat(&outputs, 0)?.reshape((tokens, heads * dim))?,
        )
    }
    #[cfg(feature = "metal")]
    fn paging(
        &self,
        pool: &StatePool,
        spans: &[Span<'_>],
        tokens: usize,
    ) -> Result<Option<tur_metal::kernels::PagedKv>> {
        let Some(k) = &self.kernels else {
            return Ok(None);
        };
        let mut table = Vec::new();
        let mut first = Vec::with_capacity(tokens);
        let mut visible = Vec::with_capacity(tokens);
        for s in spans {
            let base = u32::try_from(table.len()).map_err(|_| invalid("page table overflow"))?;
            table.extend_from_slice(&s.addresses.pages);
            for p in s.past..s.past + s.len {
                first.push(base);
                visible.push(u32::try_from(p + 1).map_err(|_| invalid("context overflow"))?);
            }
        }
        Ok(Some(k.paged_kv(
            self.description.kv.page_tokens,
            pool.attention(0)?.key.data.dim(0)?,
            &table,
            &first,
            &visible,
        )?))
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
            || tokens.iter().any(|id| {
                *id as usize >= self.config.vocab_size() || self.config.forbidden.contains(id)
            })
            || sequence.logits.iter().any(|r| !sequence.tokens.contains(r))
            || sequence.logits.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(invalid(
                "invalid Gemma 4 positions, tokens, state or logit rows",
            ));
        }
        let addresses = sequence.state.resolve(batch.pool)?;
        if addresses.pages.len() != self.description.pages_for(end) {
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
    /// Run a fresh sequence and return per-layer residual streams and selected logits.
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
        let binding = reservation.binding();
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let sequence = SequenceForward {
            tokens: 0..tokens.len(),
            state: &binding,
            logits: rows,
            hidden: false,
        };
        let sequences = [sequence];
        let batch = PackedForward {
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
            .ok_or_else(|| invalid("missing Gemma 4 output"))?;
        Ok((trace, step.logits))
    }
}
struct Span<'a> {
    start: usize,
    len: usize,
    past: usize,
    addresses: Addresses,
    sequence: &'a SequenceForward<'a>,
}
fn scatter(
    store: &tur_metal::pool::AttentionPool,
    spans: &[Span<'_>],
    key: &Tensor,
    value: &Tensor,
) -> Result<()> {
    for s in spans {
        store.scatter(
            &s.addresses.pages,
            s.past,
            &key.narrow(0, s.start, s.len)?,
            &value.narrow(0, s.start, s.len)?,
        )?;
    }
    Ok(())
}
fn ple_dtype(weights: &Weights) -> DType {
    if matches!(weights, Weights::Uniform(crate::models::WeightFormat::F32)) {
        DType::F32
    } else {
        DType::BF16
    }
}
fn rotary_tables(
    positions: &[usize],
    dim: usize,
    rope: &super::config::Rope,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let rotated = (dim as f64 * rope.partial_rotary_factor) as usize / 2;
    let frequencies: Vec<f32> = (0..dim / 2)
        .map(|i| {
            if i < rotated {
                1.0 / (rope.rope_theta as f32).powf((2 * i) as f32 / dim as f32)
            } else {
                0.0
            }
        })
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * dim / 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for &position in positions {
        for &frequency in &frequencies {
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (positions.len(), dim / 2), device)?,
        Tensor::from_vec(sin, (positions.len(), dim / 2), device)?,
    ))
}
#[cfg(feature = "metal")]
fn attention_workspace(
    config: &Config,
    kernels: &tur_metal::kernels::Kernels,
    kv: tur_core::execution::KvLayout,
    tokens: usize,
    contexts: &[usize],
) -> usize {
    let visible = contexts.iter().copied().max().unwrap_or(0);
    (0..config.layers())
        .map(|i| {
            kernels.attention_scratch_bytes(
                tokens,
                config.text.num_attention_heads,
                config.kv_heads(i),
                config.dim(i),
                visible,
                kv.format,
            )
        })
        .max()
        .unwrap_or(0)
}
fn workspace(
    config: &Config,
    tokens: usize,
    contexts: &[usize],
    split: Option<usize>,
) -> Result<usize> {
    if tokens == 0 {
        return Ok(0);
    }

    // Bound live hidden/projection/PLE buffers, plus the reference score block or split-K
    // partials. Every temporary scales with admitted tokens and contexts.
    let width = config.text.intermediate_size
        * if config.text.use_double_wide_mlp {
            2
        } else {
            1
        };
    let ple = config
        .layers()
        .checked_mul(config.text.hidden_size_per_layer_input)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| invalid("per-layer input workspace overflow"))?;
    let activations = checked_elements(&[
        tokens,
        config.hidden_size() * 8
            + width * 4
            + ple
            + config.text.num_attention_heads * config.text.global_head_dim * 6,
        4,
    ])?;
    let attention = match split {
        Some(bytes) => bytes,
        None => checked_elements(&[
            tokens.min(256),
            contexts.iter().copied().max().unwrap_or(0),
            config.text.num_attention_heads,
            8,
        ])?,
    };
    let routed = if config.text.enable_moe_block {
        let width = checked_elements(&[config.hidden_size(), 2])?
            .checked_add(checked_elements(&[
                config.text.moe_intermediate_size.unwrap_or(0),
                4,
            ])?)
            .ok_or_else(|| invalid("expert workspace width overflow"))?;
        checked_elements(&[tokens, config.text.top_k_experts.unwrap_or(0), width, 4])?
    } else {
        0
    };
    activations
        .checked_add(attention)
        .and_then(|n| n.checked_add(routed))
        .ok_or_else(|| invalid("Gemma 4 workspace overflow"))
}
impl HybridModel for Gemma4 {
    fn description(&self) -> ModelDescription {
        self.description.clone()
    }
    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> {
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
        #[cfg(feature = "metal")]
        let attention = self.kernels.as_ref().map(|kernels| {
            attention_workspace(
                &self.config,
                kernels,
                self.description.kv,
                batch.tokens.len(),
                &contexts,
            )
        });
        #[cfg(not(feature = "metal"))]
        let attention = None;
        workspace(&self.config, batch.tokens.len(), &contexts, attention)
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
