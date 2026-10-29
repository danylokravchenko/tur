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
    /// `q | k | v` on K/V owners (`q | k` when values reuse keys), queries alone otherwise.
    qkv: PackedLinear,
    o: PackedLinear,
    q_norm: Tensor,
    k_norm: Option<Tensor>,
    input_norm: Tensor,
    post_attention: Tensor,
    pre_ff: Tensor,
    post_ff: Tensor,
    scalar: Tensor,
    gate_up: PackedLinear,
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
    fused: Option<Fused>,
}

/// Compiled kernels and per-model constants of fused execution.
#[cfg(feature = "metal")]
struct Fused {
    kernels: tur_metal::kernels::Kernels,
    /// Each layer's rotary inverse frequencies, zero for unrotated pairs.
    rope: Vec<Tensor>,
    /// Weight of the unweighted value norm, as wide as the widest head.
    ones: Tensor,
    /// One zero frequency: the value norm runs the query/key norm kernel without rotation.
    unrotated: Tensor,
    /// Unit scale of residual updates that the layer scalar does not follow.
    one: Tensor,
}

fn format_plan(config: &Config, options: &LoadOptions) -> Result<WeightPlan> {
    if options.mtp.is_some() {
        return Err(invalid("Gemma 4 native draft heads are not implemented"));
    }
    if matches!(options.weights, Weights::Prepacked(_)) {
        return Err(invalid("Gemma 4 prepacked artifacts are not implemented"));
    }
    #[cfg(feature = "metal")]
    if options.execution == Execution::Fused
        && options.device.is_metal()
        && [config.text.head_dim, config.text.global_head_dim]
            .iter()
            .any(|dim| !matches!(dim, 64 | 128 | 256 | 512))
    {
        return Err(invalid(
            "Gemma 4 Metal attention supports head dimensions 64, 128, 256 and 512",
        ));
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
            qkv: projection(if own {
                "self_attn.qkv_proj"
            } else {
                "self_attn.q_proj"
            })?,
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
            gate_up: projection("mlp.gate_up_proj")?,
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
        let mut config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if dir.join("tokenizer_config.json").exists() && dir.join("tokenizer.json").exists() {
            crate::models::generation::forbid_tokenizer_modalities(
                &mut config.forbidden,
                &std::fs::read_to_string(dir.join("tokenizer_config.json"))?,
                &std::fs::read_to_string(dir.join("tokenizer.json"))?,
            )?;
        }
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
            fused: None,
        }
        .with_execution(options.execution)
    }

    /// Select common Metal kernels or the F32 Candle reference, retaining the same weights.
    pub fn with_execution(self, execution: Execution) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let fused = if execution == Execution::Fused && self.device.is_metal() {
                Some(self.fused()?)
            } else {
                None
            };
            Ok(Self { fused, ..self })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = execution;
            Ok(self)
        }
    }
    #[cfg(feature = "metal")]
    fn fused(&self) -> Result<Fused> {
        let c = &self.config;
        let rope = (0..c.layers())
            .map(|i| {
                let frequencies = inverse_frequencies(c.dim(i), c.rope(i));
                Ok(Tensor::new(frequencies.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let widest = c.text.head_dim.max(c.text.global_head_dim);
        Ok(Fused {
            kernels: tur_metal::kernels::Kernels::new(&self.device)?,
            rope,
            ones: Tensor::ones(widest, DType::F32, &self.device)?,
            unrotated: Tensor::zeros(1, DType::F32, &self.device)?,
            one: Tensor::ones(1, DType::F32, &self.device)?,
        })
    }
    #[cfg(feature = "metal")]
    fn fused_kernels(&self) -> Option<&tur_metal::kernels::Kernels> {
        self.fused.as_ref().map(|f| &f.kernels)
    }
    /// Device holding the immutable weights.
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Validated Gemma 4 settings.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Whether common Metal kernels or Candle reference operations execute passes.
    pub fn execution(&self) -> Execution {
        #[cfg(feature = "metal")]
        if self.fused.is_some() {
            return Execution::Fused;
        }
        Execution::Reference
    }
    /// Resident bytes held by packed projection matrices.
    pub fn projection_bytes(&self) -> usize {
        let mut bytes = self.head.resident_bytes()
            + self
                .ple
                .as_ref()
                .map_or(0, |p| p.projection.resident_bytes());
        for layer in &self.layers {
            bytes += [&layer.qkv, &layer.o, &layer.gate_up, &layer.down]
                .iter()
                .map(|p| p.resident_bytes())
                .sum::<usize>();
            bytes += layer.ple.as_ref().map_or(0, |p| {
                p.gate.resident_bytes() + p.projection.resident_bytes()
            });
            bytes += layer.moe.as_ref().map_or(0, moe::Moe::projection_bytes);
        }
        bytes
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
            total += [&l.qkv, &l.o, &l.gate_up, &l.down]
                .iter()
                .map(|p| p.resident_bytes())
                .sum::<usize>();
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
        if let Some(k) = self.fused_kernels() {
            return k.linear(w, x);
        }
        w.forward(x)
    }
    fn rms(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(k) = self.fused_kernels() {
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
        trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Vec<SequenceStep>> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for s in spans {
            tokens.extend_from_slice(&batch.tokens[s.sequence.tokens.clone()]);
            positions.extend_from_slice(&batch.positions[s.sequence.tokens.clone()]);
        }
        let ids = Tensor::new(tokens.as_slice(), &self.device)?;
        let x =
            (self.embedding.rows(&ids, &self.head)? * (self.config.hidden_size() as f64).sqrt())?;
        let inputs = self.inputs(&ids, &x)?;
        let pass = Pass {
            pool: batch.pool,
            spans,
            positions: &positions,
        };
        #[cfg(feature = "metal")]
        let hidden = match &self.fused {
            Some(fused) => self.run_fused(fused, &pass, x, inputs, trace)?,
            None => self.run_reference(&pass, x, inputs.as_ref(), trace)?,
        };
        #[cfg(not(feature = "metal"))]
        let hidden = self.run_reference(&pass, x, inputs.as_ref(), trace)?;
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
    /// Gated feed-forward update of one layer from residual `x` and its pre-norm `h`.
    fn mlp(&self, l: &Layer, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let gate_up = self.linear(&l.gate_up, h)?;
        let width = gate_up.dim(1)? / 2;
        let (gate, up) = (
            gate_up.narrow(1, 0, width)?,
            gate_up.narrow(1, width, width)?,
        );
        #[cfg(feature = "metal")]
        let activated = match self.fused_kernels() {
            Some(k) => k.gelu_mul(&gate, &up)?,
            None => (gate.gelu()? * up)?,
        };
        #[cfg(not(feature = "metal"))]
        let activated = (gate.gelu()? * up)?;
        let dense = self.linear(&l.down, &activated)?;
        match &l.moe {
            Some(moe) => moe.forward(self, x, &dense),
            None => Ok(dense),
        }
    }
    /// Reference decoder over Candle operations; returns the final-normed hidden rows.
    fn run_reference(
        &self,
        pass: &Pass<'_, '_>,
        mut x: Tensor,
        inputs: Option<&Tensor>,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        for (i, l) in self.layers.iter().enumerate() {
            let h = self.rms(&x, &l.input_norm)?;
            let attention = self.reference_attention(l, i, &h, pass)?;
            x = (x + self.rms(&attention, &l.post_attention)?)?;
            let h = self.rms(&x, &l.pre_ff)?;
            let mixed = self.mlp(l, &x, &h)?;
            x = (x + self.rms(&mixed, &l.post_ff)?)?;
            if let (Some(ple), Some(inputs)) = (&l.ple, inputs) {
                let gate = self.linear(&ple.gate, &x)?.gelu()?;
                let signal = (gate * inputs.narrow(1, i, 1)?.squeeze(1)?)?;
                x = (&x + self.rms(&self.linear(&ple.projection, &signal)?, &ple.norm)?)?;
            }
            x = x.broadcast_mul(&l.scalar)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        self.rms(&x, &self.norm)
    }
    fn reference_attention(
        &self,
        layer: &Layer,
        index: usize,
        hidden: &Tensor,
        pass: &Pass<'_, '_>,
    ) -> Result<Tensor> {
        let (pool, spans) = (pass.pool, pass.spans);
        let config = &self.config;
        let tokens = hidden.dim(0)?;
        let dim = config.dim(index);
        let heads = config.text.num_attention_heads;
        let (cos, sin) = rotary_tables(pass.positions, dim, config.rope(index), &self.device)?;
        let projected = self.linear(&layer.qkv, hidden)?;
        let query = projected
            .narrow(1, 0, heads * dim)?
            .reshape((tokens, heads, dim))?;
        let query = math::rotary(&self.rms(&query, &layer.q_norm)?, &cos, &sin)?;
        let store = pool.attention(config.state_layer(index))?;
        if let Some(norm) = &layer.k_norm {
            let width = config.kv_heads(index) * dim;
            let shape = (tokens, config.kv_heads(index), dim);
            let raw = projected.narrow(1, heads * dim, width)?.reshape(shape)?;
            let value = if config.unified_kv(index) {
                raw.clone()
            } else {
                projected
                    .narrow(1, heads * dim + width, width)?
                    .reshape(shape)?
            };
            let value = math::rms(
                &value,
                &Tensor::ones(dim, DType::F32, &self.device)?,
                config.text.rms_norm_eps,
            )?;
            let key = math::rotary(&self.rms(&raw, norm)?, &cos, &sin)?;
            scatter(store, spans, &key, &value)?;
        }
        let window = (config.text.layer_types[index] == LayerKind::SlidingAttention)
            .then_some(config.text.sliding_window);
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
}
#[cfg(feature = "metal")]
impl Gemma4 {
    /// Fused decoder: every layer runs once over the packed batch on Tur's kernels, with
    /// residual updates fused into the following norm. Returns the final-normed rows.
    fn run_fused(
        &self,
        f: &Fused,
        pass: &Pass<'_, '_>,
        mut x: Tensor,
        inputs: Option<Tensor>,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let k = &f.kernels;
        let eps = self.config.text.rms_norm_eps;
        let tokens = x.dim(0)?;
        let paging = self.paging(k, pass.pool, pass.spans, tokens)?;
        let positions = pass
            .positions
            .iter()
            .map(|&p| u32::try_from(p).map_err(|_| invalid("position exceeds u32")))
            .collect::<Result<Vec<_>>>()?;
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        // Per-layer inputs as `[tokens, layers × width]`; each layer reads its columns.
        let inputs = inputs
            .map(|t| {
                t.reshape((
                    tokens,
                    self.config.layers() * self.config.text.hidden_size_per_layer_input,
                ))
            })
            .transpose()?;
        let first = self
            .layers
            .first()
            .ok_or_else(|| invalid("model has no layers"))?;
        let mut h = k.rms_norm(&x, &first.input_norm, eps)?;
        for (i, l) in self.layers.iter().enumerate() {
            let attention = self.fused_attention(f, l, i, &h, &positions, pass.pool, &paging)?;
            let (sum, normed) = k.residual_rms_norm(
                &x,
                &attention,
                &l.post_attention,
                &f.one,
                Some(&l.pre_ff),
                eps,
            )?;
            let normed = normed.ok_or_else(|| invalid("missing feed-forward input"))?;
            let mixed = self.mlp(l, &sum, &normed)?;
            // The last update fuses with the final norm.
            let next = self.layers.get(i + 1).map_or(&self.norm, |n| &n.input_norm);
            (x, h) = self.fused_update(f, l, i, &sum, &mixed, inputs.as_ref(), next)?;
            if let Some(trace) = trace.as_mut() {
                trace.push(x.clone());
            }
        }
        Ok(h)
    }
    /// Feed-forward residual, per-layer input injection and layer scalar, fused with the
    /// next norm: returns the next residual stream and its normed rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per operand of the fused residual update"
    )]
    fn fused_update(
        &self,
        fused: &Fused,
        layer: &Layer,
        index: usize,
        x: &Tensor,
        mixed: &Tensor,
        inputs: Option<&Tensor>,
        next: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let k = &fused.kernels;
        let eps = self.config.text.rms_norm_eps;
        let (x, h) = match (&layer.ple, inputs) {
            (Some(ple), Some(inputs)) => {
                let (x, _) =
                    k.residual_rms_norm(x, mixed, &layer.post_ff, &fused.one, None, eps)?;
                let width = self.config.text.hidden_size_per_layer_input;
                let column = index
                    .checked_mul(width)
                    .ok_or_else(|| invalid("per-layer input column overflow"))?;
                let signal =
                    k.gelu_mul(&k.linear(&ple.gate, &x)?, &inputs.narrow(1, column, width)?)?;
                let injected = k.linear(&ple.projection, &signal)?;
                k.residual_rms_norm(&x, &injected, &ple.norm, &layer.scalar, Some(next), eps)?
            }
            _ => k.residual_rms_norm(x, mixed, &layer.post_ff, &layer.scalar, Some(next), eps)?,
        };
        Ok((x, h.ok_or_else(|| invalid("missing next layer input"))?))
    }
    /// Projects `q | k | v` once and norms and rotates queries and keys in place from it;
    /// K/V owners store their normed values before attending.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per input the attention dispatch binds"
    )]
    fn fused_attention(
        &self,
        fused: &Fused,
        layer: &Layer,
        index: usize,
        hidden: &Tensor,
        positions: &Tensor,
        pool: &StatePool,
        paging: &tur_metal::kernels::PagedKv,
    ) -> Result<Tensor> {
        let config = &self.config;
        let k = &fused.kernels;
        let eps = config.text.rms_norm_eps;
        let dim = config.dim(index);
        let heads = config.text.num_attention_heads;
        let rope = fused
            .rope
            .get(index)
            .ok_or_else(|| invalid("missing rotary frequencies"))?;
        let at = |count: usize, col_offset: usize| tur_metal::kernels::Heads {
            count,
            dim,
            head_stride: dim,
            col_offset,
        };
        let projected = k.linear(&layer.qkv, hidden)?;
        let query = k.qk_norm_rope(
            &projected,
            at(heads, 0),
            &layer.q_norm,
            positions,
            rope,
            eps,
        )?;
        let store = pool.attention(config.state_layer(index))?;
        if let Some(norm) = &layer.k_norm {
            let kv_heads = config.kv_heads(index);
            let key_col = heads * dim;
            // Unified K=V values are the raw key projection, read before key norm and RoPE.
            let value_col = if config.unified_kv(index) {
                key_col
            } else {
                key_col + kv_heads * dim
            };
            let key = k.qk_norm_rope(
                &projected,
                at(kv_heads, key_col),
                norm,
                positions,
                rope,
                eps,
            )?;
            let value = k.qk_norm_rope(
                &projected,
                at(kv_heads, value_col),
                &fused.ones.narrow(0, 0, dim)?,
                positions,
                &fused.unrotated,
                eps,
            )?;
            let width = kv_heads * dim;
            k.kv_write(
                &key.reshape((key.dim(0)?, width))?,
                at(kv_heads, 0),
                &value.reshape((value.dim(0)?, width))?,
                at(kv_heads, 0),
                store,
                paging,
            )?;
        }
        let window = (config.text.layer_types[index] == LayerKind::SlidingAttention)
            .then_some(config.text.sliding_window);
        k.linear(
            &layer.o,
            &k.attention_ungated(&query, store, paging, window, 1.0)?,
        )
    }
    fn paging(
        &self,
        k: &tur_metal::kernels::Kernels,
        pool: &StatePool,
        spans: &[Span<'_>],
        tokens: usize,
    ) -> Result<tur_metal::kernels::PagedKv> {
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
        k.paged_kv(
            self.description.kv.page_tokens,
            pool.attention(0)?.key.data.dim(0)?,
            &table,
            &first,
            &visible,
        )
    }
}

impl Gemma4 {
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
/// Addressing shared by every layer of one packed pass.
struct Pass<'a, 'b> {
    pool: &'a StatePool,
    spans: &'a [Span<'b>],
    positions: &'a [usize],
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
/// Inverse rotary frequencies over `dim / 2` pairs; proportional rotation leaves the pairs past
/// its rotated fraction at zero frequency.
fn inverse_frequencies(dim: usize, rope: &super::config::Rope) -> Vec<f32> {
    let rotated = (dim as f64 * rope.partial_rotary_factor) as usize / 2;
    (0..dim / 2)
        .map(|i| {
            if i < rotated {
                1.0 / (rope.rope_theta as f32).powf((2 * i) as f32 / dim as f32)
            } else {
                0.0
            }
        })
        .collect()
}
fn rotary_tables(
    positions: &[usize],
    dim: usize,
    rope: &super::config::Rope,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let frequencies = inverse_frequencies(dim, rope);
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
        let attention = self.fused_kernels().map(|kernels| {
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
        self.fused_kernels().cloned()
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
