//! Inkling 2's block drafter (`mtp.safetensors`), as vLLM and `SGLang` run it (their
//! `DSpark` drafter for Inkling 2).
//!
//! Each committed position mixes four target features (the residual stream entering three
//! layer applications, and the final-normed output) with learned softmax weights, once per
//! draft layer, and writes the result's keys and values into that layer's state. A draft
//! runs one block: the target's normalized embedding of the anchor (the token after the
//! last committed position) and `block_size - 1` copies of a learned mask row, through a
//! small dense decoder whose attention is bidirectional over the context and the block.
//! Mask `k` predicts the token at its own position. A recurrent head then walks the masks
//! in order, adding a bias from the previously drafted token to each mask's logits before
//! the greedy pick, so drafts chain without another decoder pass.
use super::{Config, Inkling2};
use crate::{
    hybrid::{DraftForward, DraftSequence},
    models::{
        LoadOptions, Projection, WeightPlan, Weights,
        checkpoint::Checkpoint,
        inkling2::config::Mtp,
        paging::Paging,
        weights::{Embedding, Precision, Tensors, WeightLayout},
    },
};
use candle_core::{DType, Device, Tensor};
use std::{collections::BTreeMap, path::Path};
use tur_core::{
    Result,
    execution::{checked_elements, invalid},
};
use tur_metal::{
    kernels::{Heads, Kernels},
    packed::PackedLinear,
};

/// The drafter's checkpoint, beside the text tower's.
pub(super) const FILE: &str = "mtp.safetensors";

/// Context keys a block reads per online-softmax step, so its scratch stays bounded
/// whatever the context.
const CHUNK: usize = 4096;

struct DraftLayer {
    input_norm: Tensor,
    post_norm: Tensor,
    /// Norm of this layer's mix of target features.
    mix_norm: Tensor,
    /// `q | k | v` of the block rows.
    qkv: PackedLinear,
    /// `k | v` of the context, from mixed target features.
    context: PackedLinear,
    output: PackedLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    context_norm: Tensor,
    gate_up: PackedLinear,
    down: PackedLinear,
    sliding: bool,
}

/// Loaded drafter weights.
pub(super) struct Drafter {
    layers: Vec<DraftLayer>,
    norm: Tensor,
    /// `[1, hidden]` input of every mask row.
    mask: Tensor,
    /// Softmax of `fuse_weight`: per draft layer, then the recurrent head, a weight per
    /// target feature.
    mix: Vec<Vec<f32>>,
    markov_norm: Tensor,
    /// `[padded vocab, rank]` BF16 embedding of the previous draft.
    previous: Tensor,
    /// `rnn_proj` split by input: `[2 * rank, 3 * rank]` for the state and the previous
    /// draft, `[2 * hidden, 3 * rank]` for the mask's hidden state and the context feature,
    /// both transposed for `x @ w`, and its bias.
    recurrent: Tensor,
    context: Tensor,
    bias: Tensor,
    /// The unembedding times `hidden_out`, `[padded vocab, rank]`, and the unembedding times
    /// its bias: the recurrent head's logit bias without a full head projection per step.
    fold: PackedLinear,
    fold_bias: Tensor,
    inv_freq: Tensor,
    settings: Mtp,
}

/// The drafter's projections in `mtp.safetensors`.
struct Layout<'a> {
    config: &'a Config,
    d: &'a Mtp,
}

fn layer(i: usize) -> String {
    format!("layers.{i}")
}

impl WeightLayout for Layout<'_> {
    fn projection_layout(&self) -> Result<Vec<Projection>> {
        let d = self.d;
        let hidden = self.config.text.hidden_size;
        let (q, kv) = (
            checked_elements(&[d.num_attention_heads, d.head_dim])?,
            checked_elements(&[d.num_key_value_heads, d.head_dim])?,
        );
        let mut projections = Vec::with_capacity(5 * d.num_hidden_layers);
        for i in 0..d.num_hidden_layers {
            let a = format!("{}.self_attn", layer(i));
            let m = format!("{}.mlp", layer(i));
            let part = |name: String, rows: usize| (format!("{name}.weight"), rows);
            projections.extend([
                Projection::new(
                    format!("{a}.qkv"),
                    vec![
                        part(format!("{a}.q_proj"), q),
                        part(format!("{a}.k_proj"), kv),
                        part(format!("{a}.v_proj"), kv),
                    ],
                    hidden,
                ),
                Projection::new(
                    format!("{a}.context"),
                    vec![
                        part(format!("{a}.k_proj_ctx"), kv),
                        part(format!("{a}.v_proj_ctx"), kv),
                    ],
                    hidden,
                ),
                Projection::new(
                    format!("{a}.output"),
                    vec![part(format!("{a}.o_proj"), hidden)],
                    q,
                ),
                Projection::new(
                    format!("{m}.gate_up"),
                    vec![
                        part(format!("{m}.gate_proj"), d.intermediate_size),
                        part(format!("{m}.up_proj"), d.intermediate_size),
                    ],
                    hidden,
                ),
                Projection::new(
                    format!("{m}.down"),
                    vec![part(format!("{m}.down_proj"), hidden)],
                    d.intermediate_size,
                ),
            ]);
        }
        Ok(projections)
    }
    /// The drafter embeds with the target's table and keeps none of its own, which a tied
    /// layout with [`Embedding::Head`] states: the plan then counts no table.
    fn embeddings_tied(&self) -> bool {
        true
    }
    fn vocabulary(&self) -> usize {
        self.config.padded_vocab()
    }
    fn residual_width(&self) -> usize {
        self.config.text.hidden_size
    }
    fn embedding_tensor_name(&self) -> String {
        "mask_embed".into()
    }
}

/// Every tensor of `mtp.safetensors` with its shape.
fn shapes(config: &Config, d: &Mtp) -> Result<BTreeMap<String, Vec<usize>>> {
    let hidden = config.text.hidden_size;
    let (rank, features) = (d.markov_rank, d.features());
    let mut shapes = BTreeMap::new();
    for p in (Layout { config, d }).projection_layout()? {
        for (name, rows) in p.parts {
            shapes.insert(name, vec![rows, p.cols]);
        }
    }
    for i in 0..d.num_hidden_layers {
        let l = layer(i);
        for norm in ["input_layernorm", "post_attention_layernorm", "dflare_norm"] {
            shapes.insert(format!("{l}.{norm}.weight"), vec![hidden]);
        }
        for norm in ["q_norm", "k_norm", "k_norm_ctx"] {
            shapes.insert(format!("{l}.self_attn.{norm}.weight"), vec![d.head_dim]);
        }
    }
    let m = "markov_head";
    for (name, shape) in [
        (
            "fuse_weight".to_owned(),
            vec![d.num_hidden_layers + 1, features],
        ),
        ("mask_embed".into(), vec![hidden]),
        ("norm.weight".into(), vec![hidden]),
        (format!("{m}.dflare_norm.weight"), vec![hidden]),
        (format!("{m}.hidden_out.weight"), vec![hidden, rank]),
        (format!("{m}.hidden_out.bias"), vec![hidden]),
        (
            format!("{m}.markov_w1.weight"),
            vec![config.padded_vocab(), rank],
        ),
        (
            format!("{m}.rnn_proj.weight"),
            vec![3 * rank, 2 * rank + 2 * hidden],
        ),
        (format!("{m}.rnn_proj.bias"), vec![3 * rank]),
    ] {
        shapes.insert(name, shape);
    }
    Ok(shapes)
}

/// The drafter's projections packed as the plan packs the target's; a separate checkpoint
/// cannot follow a per-tensor plan.
fn plan(config: &Config, d: &Mtp, options: &LoadOptions) -> Result<WeightPlan> {
    let layout = Layout { config, d };
    let mut plan = match &options.weights {
        Weights::Uniform(_) => options.weights.plan(&layout)?,
        _ => {
            return Err(invalid(
                "Inkling 2's drafter packs its projections in one uniform format",
            ));
        }
    };
    // Its embedding is the target's; the mask row is a parameter, not a table.
    plan.embedding = Embedding::Head;
    plan.validate(&layout)?;
    Ok(plan)
}

/// The fold's format: the unembedding's, where its blocks divide the rank.
fn fold_precision(head: Precision, rank: usize) -> Precision {
    match head {
        Precision::Q8_0 if rank.is_multiple_of(32) => Precision::Q8_0,
        Precision::Q4k if rank.is_multiple_of(256) => Precision::Q4k,
        _ => Precision::F32,
    }
}

impl Drafter {
    /// Device bytes [`Self::load`] will hold, before reading any weights.
    pub(super) fn planned_bytes(config: &Config, options: &LoadOptions) -> Result<usize> {
        let d = config
            .mtp()
            .ok_or_else(|| invalid("Inkling 2 was configured without its drafter"))?;
        let hidden = config.text.hidden_size;
        let (rank, vocab) = (d.markov_rank, config.padded_vocab());
        let head = options.weights.plan(config)?.precision("lm_head")?;
        let f32s = checked_elements(&[d.num_hidden_layers, 3 * hidden + 3 * d.head_dim])?
            + 3 * hidden
            + d.head_dim / 2
            + vocab
            + checked_elements(&[3 * rank, 2 * rank + 2 * hidden + 1])?;
        [
            plan(config, d, options)?.bytes(&Layout { config, d })?,
            checked_elements(&[f32s, 4])?,
            checked_elements(&[vocab, rank, 2])?,
            fold_precision(head, rank).bytes(vocab, rank)?,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| invalid("Inkling 2 drafter bytes overflow"))
        })
    }

    /// Load `mtp.safetensors` from `dir`, folding the recurrent head's output projection
    /// into the target's unembedding (read again from `target`) and packing the result as
    /// `head` is packed.
    pub(super) fn load(
        dir: &Path,
        config: &Config,
        options: &LoadOptions,
        target: &Tensors,
        head: &PackedLinear,
    ) -> Result<Self> {
        let d = config
            .mtp()
            .ok_or_else(|| invalid("Inkling 2 was configured without its drafter"))?
            .clone();
        let (checkpoint, _) = Checkpoint::scan(vec![dir.join(FILE)])?;
        let expected = shapes(config, &d)?;
        for (name, entry) in checkpoint.entries() {
            if expected.get(name) != Some(&entry.shape) {
                return Err(invalid(format!("unexpected drafter tensor {name}")));
            }
        }
        if let Some(missing) = expected
            .keys()
            .find(|n| !checkpoint.entries().contains_key(*n))
        {
            return Err(invalid(format!("missing drafter tensor {missing}")));
        }
        let device = options.device.clone();
        let t = Tensors {
            #[cfg(feature = "metal")]
            expert_cache: None,
            checkpoint,
            prepacked: false,
            plan: plan(config, &d, options)?,
            projections: Layout { config, d: &d }
                .projection_layout()?
                .into_iter()
                .map(|p| (p.name.clone(), p))
                .collect(),
            device: device.clone(),
        };
        let f32 = |name: &str| -> Result<Tensor> {
            Ok(t.take(name)?.to_dtype(DType::F32)?.to_device(&device)?)
        };
        let layers = (0..d.num_hidden_layers)
            .map(|i| {
                let (l, a) = (layer(i), format!("{}.self_attn", layer(i)));
                Ok(DraftLayer {
                    input_norm: f32(&format!("{l}.input_layernorm.weight"))?,
                    post_norm: f32(&format!("{l}.post_attention_layernorm.weight"))?,
                    mix_norm: f32(&format!("{l}.dflare_norm.weight"))?,
                    qkv: t.projection(&format!("{a}.qkv"))?,
                    context: t.projection(&format!("{a}.context"))?,
                    output: t.projection(&format!("{a}.output"))?,
                    q_norm: f32(&format!("{a}.q_norm.weight"))?,
                    k_norm: f32(&format!("{a}.k_norm.weight"))?,
                    context_norm: f32(&format!("{a}.k_norm_ctx.weight"))?,
                    gate_up: t.projection(&format!("{l}.mlp.gate_up"))?,
                    down: t.projection(&format!("{l}.mlp.down"))?,
                    sliding: d.sliding(i),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mix = mix_weights(&t.take("fuse_weight")?)?;
        let m = "markov_head";
        let rank = d.markov_rank;
        let rnn = t
            .take(&format!("{m}.rnn_proj.weight"))?
            .to_dtype(DType::F32)?;
        let (fold, fold_bias) = fold(
            target,
            &t.take(&format!("{m}.hidden_out.weight"))?,
            &t.take(&format!("{m}.hidden_out.bias"))?,
            fold_precision(precision(head), rank),
            &device,
        )?;
        let rotary = d.head_dim / 2;
        let inv_freq: Vec<f32> = (0..rotary)
            .map(|i| (d.rope_theta.powf(-(2.0 * i as f64) / d.head_dim as f64)) as f32)
            .collect();
        Ok(Self {
            layers,
            norm: f32("norm.weight")?,
            mask: f32("mask_embed")?.unsqueeze(0)?,
            mix,
            markov_norm: f32(&format!("{m}.dflare_norm.weight"))?,
            previous: t
                .take(&format!("{m}.markov_w1.weight"))?
                .to_dtype(DType::BF16)?
                .to_device(&device)?,
            recurrent: rnn
                .narrow(1, 0, 2 * rank)?
                .t()?
                .contiguous()?
                .to_device(&device)?,
            context: rnn
                .narrow(1, 2 * rank, rnn.dim(1)? - 2 * rank)?
                .t()?
                .contiguous()?
                .to_device(&device)?,
            bias: f32(&format!("{m}.rnn_proj.bias"))?.unsqueeze(0)?,
            fold,
            fold_bias: fold_bias.unsqueeze(0)?,
            inv_freq: Tensor::from_vec(inv_freq, rotary, &device)?,
            settings: d,
        })
    }

    /// Resident device bytes; equals [`Self::planned_bytes`].
    pub(super) fn bytes(&self) -> usize {
        let mut tensors = vec![
            &self.norm,
            &self.mask,
            &self.markov_norm,
            &self.recurrent,
            &self.context,
            &self.bias,
            &self.fold_bias,
            &self.inv_freq,
        ];
        for l in &self.layers {
            tensors.extend([
                &l.input_norm,
                &l.post_norm,
                &l.mix_norm,
                &l.q_norm,
                &l.k_norm,
                &l.context_norm,
            ]);
        }
        let projections: usize = self
            .layers
            .iter()
            .flat_map(|l| [&l.qkv, &l.context, &l.output, &l.gate_up, &l.down])
            .map(PackedLinear::resident_bytes)
            .sum();
        tensors.iter().map(|t| t.elem_count() * 4).sum::<usize>()
            + projections
            + self.previous.elem_count() * 2
            + self.fold.resident_bytes()
    }

    /// Scratch of one draft pass: a chunk of gathered keys and values with the block's
    /// appended, and its scores.
    pub(super) fn workspace(config: &Config) -> Result<usize> {
        let Some(d) = config.mtp() else {
            return Ok(0);
        };
        let keys = CHUNK + d.block_size;
        let kv = checked_elements(&[keys, d.num_key_value_heads, d.head_dim, 4, 4])?;
        let scores = checked_elements(&[d.block_size, d.num_attention_heads, keys, 4, 2])?;
        kv.checked_add(scores)
            .ok_or_else(|| invalid("draft workspace overflow"))
    }

    /// Drafts a block yields after its anchor.
    pub(super) fn depth(&self) -> usize {
        self.settings.block_size - 1
    }
}

fn precision(head: &PackedLinear) -> Precision {
    match head.format() {
        Some(candle_core::quantized::GgmlDType::Q8_0) => Precision::Q8_0,
        Some(candle_core::quantized::GgmlDType::Q4K) => Precision::Q4k,
        _ => Precision::F32,
    }
}

/// Row-wise softmax of `fuse_weight`, in F32 as the references take it.
fn mix_weights(fuse: &Tensor) -> Result<Vec<Vec<f32>>> {
    fuse.to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|row| {
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|v| (v - top).exp()).collect();
            let total: f32 = exp.iter().sum();
            if !(total.is_finite() && total > 0.0) {
                return Err(invalid("drafter feature weights are not finite"));
            }
            Ok(exp.into_iter().map(|e| e / total).collect())
        })
        .collect()
}

/// `unembed @ hidden_out` and `unembed @ bias`, computed on `device` in row chunks so the
/// F32 unembedding never exists whole, then packed in `precision`.
fn fold(
    target: &Tensors,
    projection: &Tensor,
    bias: &Tensor,
    precision: Precision,
    device: &Device,
) -> Result<(PackedLinear, Tensor)> {
    const ROWS: usize = 16_384;
    let unembed = target.take(&format!("{}.unembed.weight", super::PREFIX))?;
    let rows = unembed.dim(0)?;
    let projection = projection.to_dtype(DType::F32)?.to_device(device)?;
    let bias = bias.to_dtype(DType::F32)?.to_device(device)?.unsqueeze(1)?;
    let (mut weights, mut biases) = (Vec::new(), Vec::new());
    for start in (0..rows).step_by(ROWS) {
        let chunk = unembed
            .narrow(0, start, ROWS.min(rows - start))?
            .to_dtype(DType::F32)?
            .to_device(device)?;
        weights.push(chunk.matmul(&projection)?.to_device(&Device::Cpu)?);
        biases.push(chunk.matmul(&bias)?.squeeze(1)?);
    }
    let weight = Tensor::cat(&weights, 0)?;
    let packed = match precision.ggml() {
        Some(format) if precision != Precision::F32 => {
            PackedLinear::quantized(&weight, format, device)?
        }
        _ => PackedLinear::dense(&weight, device)?,
    };
    Ok((packed, Tensor::cat(&biases, 0)?))
}

/// One sequence of a draft pass, checked.
struct Span<'a> {
    sequence: &'a DraftSequence<'a>,
    addresses: crate::hybrid::Addresses,
    /// Its first committed position and how many it writes.
    past: usize,
    len: usize,
}

impl Inkling2 {
    /// The drafter's pass: context keys and values of every given position, then for each
    /// sequence asking for drafts, its block's chained logits.
    pub(super) fn draft_pass(
        &self,
        k: &Kernels,
        drafter: &Drafter,
        batch: &DraftForward<'_>,
    ) -> Result<Vec<Result<Option<Tensor>>>> {
        let mut out: Vec<Option<Result<Option<Tensor>>>> =
            batch.sequences.iter().map(|_| None).collect();
        let mut spans = Vec::new();
        for (i, sequence) in batch.sequences.iter().enumerate() {
            match self.check_draft(batch, drafter, sequence) {
                Ok(span) => spans.push((i, span)),
                Err(error) => out[i] = Some(Err(error)),
            }
        }
        if !spans.is_empty() {
            self.write_context(k, drafter, batch, &spans)?;
            for (i, span) in &spans {
                out[*i] = Some(
                    self.block(k, drafter, batch, span)
                        .map(|logits| (span.sequence.drafts > 0).then_some(logits).flatten()),
                );
            }
        }
        Ok(out
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("sequence was not drafted"))))
            .collect())
    }

    fn check_draft<'a>(
        &self,
        batch: &DraftForward<'_>,
        drafter: &Drafter,
        sequence: &'a DraftSequence<'a>,
    ) -> Result<Span<'a>> {
        let positions = batch
            .positions
            .get(sequence.tokens.clone())
            .filter(|p| !p.is_empty())
            .ok_or_else(|| invalid("empty or invalid draft range"))?;
        let tokens = &batch.tokens[sequence.tokens.clone()];
        let past = positions[0];
        let width = checked_elements(&[drafter.settings.features(), self.config.text.hidden_size])?;
        if positions
            .iter()
            .enumerate()
            .any(|(i, &p)| past.checked_add(i) != Some(p))
            || sequence.state.past() != past
            || sequence.state.tokens() != positions.len()
            || sequence.hidden.dims() != [positions.len(), width]
            || sequence.drafts > drafter.depth()
            || tokens
                .iter()
                .any(|&t| t as usize >= self.config.vocab_size())
            || past
                .checked_add(positions.len() + drafter.settings.block_size)
                .is_none_or(|end| end > self.config.max_position())
        {
            return Err(invalid(
                "invalid Inkling 2 draft positions, tokens, features or depth",
            ));
        }
        Ok(Span {
            sequence,
            addresses: sequence.state.resolve(batch.pool)?,
            past,
            len: positions.len(),
        })
    }

    /// Draft layer `i`'s state layer: after both layers of every application.
    fn draft_store<'p>(
        &self,
        pool: &'p crate::hybrid::StatePool,
        i: usize,
    ) -> Result<&'p tur_metal::pool::AttentionPool> {
        pool.attention(2 * self.config.applications().len() + i)
    }

    /// `rms(Σ_f mix[f] · feature_f) · norm` of `[rows, features * hidden]` features.
    fn mix(
        &self,
        k: &Kernels,
        features: &Tensor,
        mix: &[f32],
        norm: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        let hidden = self.config.text.hidden_size;
        let weights = Tensor::from_slice(mix, (1, mix.len(), 1), &self.device)?;
        let mixed = features
            .reshape((features.dim(0)?, mix.len(), hidden))?
            .broadcast_mul(&weights)?
            .sum(1)?;
        k.rms_norm(&mixed, norm, eps)
    }

    fn write_context(
        &self,
        k: &Kernels,
        drafter: &Drafter,
        batch: &DraftForward<'_>,
        spans: &[(usize, Span<'_>)],
    ) -> Result<()> {
        let d = &drafter.settings;
        let features = Tensor::cat(
            &spans
                .iter()
                .map(|(_, s)| s.sequence.hidden)
                .collect::<Vec<_>>(),
            0,
        )?;
        let rows = features.dim(0)?;
        let positions: Vec<u32> = spans
            .iter()
            .flat_map(|(_, s)| s.past..s.past + s.len)
            .map(|p| u32::try_from(p).map_err(|_| invalid("position exceeds u32")))
            .collect::<Result<_>>()?;
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        let paging = Paging::new(
            k,
            &self.description,
            batch.pool,
            spans.iter().map(|(_, s)| (&s.addresses, s.past, s.len)),
            rows,
        )?;
        let kv = d.num_key_value_heads * d.head_dim;
        let heads = |col_offset| Heads {
            count: d.num_key_value_heads,
            dim: d.head_dim,
            head_stride: d.head_dim,
            col_offset,
        };
        for (i, l) in drafter.layers.iter().enumerate() {
            let context = self.mix(k, &features, &drafter.mix[i], &l.mix_norm, d.rms_norm_eps)?;
            let projected = k.linear(&l.context, &context)?;
            let key = k
                .qk_norm_rope(
                    &projected,
                    heads(0),
                    &l.context_norm,
                    &positions,
                    &drafter.inv_freq,
                    d.rms_norm_eps,
                )?
                .reshape((rows, kv))?;
            k.kv_write(
                &key,
                heads(0),
                &projected,
                heads(kv),
                self.draft_store(batch.pool, i)?,
                paging.layer(l.sliding)?,
            )?;
        }
        Ok(())
    }

    /// One sequence's block: `[drafts, vocab]` chained logits, or `None` without drafts.
    fn block(
        &self,
        k: &Kernels,
        drafter: &Drafter,
        batch: &DraftForward<'_>,
        span: &Span<'_>,
    ) -> Result<Option<Tensor>> {
        let drafts = span.sequence.drafts;
        if drafts == 0 {
            return Ok(None);
        }
        let d = &drafter.settings;
        let eps = d.rms_norm_eps;
        let size = d.block_size;
        let anchor = *batch
            .tokens
            .get(span.sequence.tokens.end - 1)
            .ok_or_else(|| invalid("draft has no anchor token"))?;
        let end = span.past + span.len;
        let positions: Vec<u32> = (end..end + size)
            .map(|p| u32::try_from(p).map_err(|_| invalid("position exceeds u32")))
            .collect::<Result<_>>()?;
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        let ids = Tensor::new(&[anchor], &self.device)?;
        let embedded = k.rms_norm(
            &self.embedding.rows(&ids, &self.head)?,
            &self.embed_norm,
            self.config.text.rms_norm_eps,
        )?;
        let x = Tensor::cat(&[embedded, drafter.mask.repeat((size - 1, 1))?], 0)?;
        let mut residual = x.clone();
        let mut h = k.rms_norm(&x, &drafter.layers[0].input_norm, eps)?;
        let mut out = None;
        for (i, l) in drafter.layers.iter().enumerate() {
            if let Some(out) = out.take() {
                (residual, h) = k.add_rms_norm(&residual, &out, &l.input_norm, eps)?;
            }
            let attended = self.block_attention(k, drafter, batch, span, l, i, (&h, &positions))?;
            let (sum, normed) = k.add_rms_norm(
                &residual,
                &k.linear(&l.output, &attended)?,
                &l.post_norm,
                eps,
            )?;
            residual = sum;
            out = Some(k.linear(&l.down, &k.swiglu(&k.linear(&l.gate_up, &normed)?)?)?);
        }
        let out = out.ok_or_else(|| invalid("drafter has no layers"))?;
        let (_, hidden) = k.add_rms_norm(&residual, &out, &drafter.norm, eps)?;
        let last = span.sequence.hidden.narrow(0, span.len - 1, 1)?;
        let feature = self.mix(
            k,
            &last,
            &drafter.mix[d.num_hidden_layers],
            &drafter.markov_norm,
            eps,
        )?;
        Ok(Some(self.chain(
            k,
            drafter,
            &hidden.narrow(0, 1, drafts)?,
            &feature,
            anchor,
        )?))
    }

    /// Bidirectional attention of the block's rows over the layer's context keys (its
    /// window before the block in sliding layers) and the block's own keys.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per input the draft layer's attention reads"
    )]
    fn block_attention(
        &self,
        kernels: &Kernels,
        drafter: &Drafter,
        batch: &DraftForward<'_>,
        span: &Span<'_>,
        layer: &DraftLayer,
        index: usize,
        (input, positions): (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let settings = &drafter.settings;
        let (heads, kv_heads, dim) = (
            settings.num_attention_heads,
            settings.num_key_value_heads,
            settings.head_dim,
        );
        let size = settings.block_size;
        let qkv = kernels.linear(&layer.qkv, input)?;
        let at = |count, col_offset| Heads {
            count,
            dim,
            head_stride: dim,
            col_offset,
        };
        let rotate = |heads: Heads, norm: &Tensor| {
            kernels.qk_norm_rope(
                &qkv,
                heads,
                norm,
                positions,
                &drafter.inv_freq,
                settings.rms_norm_eps,
            )
        };
        let queries = rotate(at(heads, 0), &layer.q_norm)?;
        let keys = rotate(at(kv_heads, heads * dim), &layer.k_norm)?;
        let values = qkv
            .narrow(1, (heads + kv_heads) * dim, kv_heads * dim)?
            .reshape((size, kv_heads, dim))?;
        let group = heads / kv_heads;
        // Query heads grouped by the K/V head they read: `[kv_heads, size * group, dim]`.
        let grouped = queries
            .reshape((size, kv_heads, group, dim))?
            .permute((1, 0, 2, 3))?
            .contiguous()?
            .reshape((kv_heads, size * group, dim))?;
        let block = (&keys, &values);
        Ok(self
            .chunked_softmax(drafter, batch, span, index, &grouped, block)?
            .reshape((kv_heads, size, group, dim))?
            .permute((1, 0, 2, 3))?
            .contiguous()?
            .reshape((size, heads * dim))?)
    }

    /// Softmax attention of `grouped` queries over draft layer `index`'s context and the
    /// block's own keys and values, by online softmax over chunks of the context so its
    /// scratch stays bounded; the block's keys join the last chunk.
    fn chunked_softmax(
        &self,
        drafter: &Drafter,
        batch: &DraftForward<'_>,
        span: &Span<'_>,
        index: usize,
        grouped: &Tensor,
        (block_keys, block_values): (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let scale = (drafter.settings.head_dim as f64).sqrt().recip();
        let end = span.past + span.len;
        let first = if drafter.layers[index].sliding {
            end.saturating_sub(self.config.text.sliding_window_size)
        } else {
            0
        };
        let mut chunks: Vec<_> = (first..end).step_by(CHUNK).collect();
        if chunks.is_empty() {
            chunks.push(end);
        }
        let last = chunks.len() - 1;
        let mut merged: Option<(Tensor, Tensor, Tensor)> = None;
        for (chunk, &start) in chunks.iter().enumerate() {
            let range = start..(start + CHUNK).min(end);
            let (mut keys, mut values) = self.context_rows(drafter, batch, span, index, range)?;
            if chunk == last {
                keys = Tensor::cat(&[&keys, block_keys], 0)?;
                values = Tensor::cat(&[&values, block_values], 0)?;
            }
            let scores = (grouped.matmul(&keys.permute((1, 2, 0))?.contiguous()?)? * scale)?;
            let top = scores.max_keepdim(2)?;
            let weights = scores.broadcast_sub(&top)?.exp()?;
            let total = weights.sum_keepdim(2)?;
            let out = weights.matmul(&values.permute((1, 0, 2))?.contiguous()?)?;
            merged = Some(match merged {
                None => (top, total, out),
                Some((max, sum, acc)) => {
                    let joined = max.maximum(&top)?;
                    let (old, new) = ((max - &joined)?.exp()?, (top - &joined)?.exp()?);
                    (
                        joined,
                        ((sum * &old)? + (total * &new)?)?,
                        (acc.broadcast_mul(&old)? + out.broadcast_mul(&new)?)?,
                    )
                }
            });
        }
        let (_, total, out) = merged.ok_or_else(|| invalid("draft attention read no keys"))?;
        Ok(out.broadcast_div(&total)?)
    }

    /// F32 `[keys, kv_heads, dim]` keys and values of draft layer `i` at context
    /// `positions`, gathered from its pages in one selection each.
    fn context_rows(
        &self,
        drafter: &Drafter,
        batch: &DraftForward<'_>,
        span: &Span<'_>,
        i: usize,
        positions: std::ops::Range<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let store = self.draft_store(batch.pool, i)?;
        let table = if drafter.layers[i].sliding {
            &span.addresses.windows
        } else {
            &span.addresses.pages
        };
        let page = store.page_tokens;
        let slots: Vec<u32> = positions
            .map(|p| {
                let physical = *table
                    .get(p / page)
                    .ok_or_else(|| invalid("draft context outside its page table"))?;
                physical
                    .checked_mul(u32::try_from(page).map_err(|_| invalid("page exceeds u32"))?)
                    .and_then(|s| s.checked_add(u32::try_from(p % page).ok()?))
                    .ok_or_else(|| invalid("draft context slot overflow"))
            })
            .collect::<Result<_>>()?;
        let d = &drafter.settings;
        if slots.is_empty() {
            let empty = Tensor::zeros(
                (0, d.num_key_value_heads, d.head_dim),
                DType::F32,
                &self.device,
            )?;
            return Ok((empty.clone(), empty));
        }
        let slots = Tensor::new(slots.as_slice(), &self.device)?;
        let read = |pool: &tur_metal::pool::KvPool| -> Result<Tensor> {
            let data = pool.data.index_select(&slots, 0)?;
            let scales = pool
                .scales
                .as_ref()
                .map(|s| s.index_select(&slots, 0))
                .transpose()?;
            tur_metal::pool::dequantize(store.format, &data, scales.as_ref())
        };
        Ok((read(&store.key)?, read(&store.value)?))
    }

    /// The recurrent head over the masks' `[drafts, hidden]` final states: each step adds
    /// its previous draft's bias to the mask's logits, softcaps them and picks greedily on
    /// the device, so no step waits for the host.
    fn chain(
        &self,
        k: &Kernels,
        drafter: &Drafter,
        hidden: &Tensor,
        feature: &Tensor,
        anchor: u32,
    ) -> Result<Tensor> {
        let drafts = hidden.dim(0)?;
        let rank = drafter.settings.markov_rank;
        let base = k.linear(&self.head, hidden)?;
        let joined = Tensor::cat(&[hidden, &feature.repeat((drafts, 1))?], 1)?;
        let context = joined
            .matmul(&drafter.context)?
            .broadcast_add(&drafter.bias)?;
        let cap = self.config.text.final_logit_softcapping;
        let vocab = self.config.vocab_size();
        let mut state = Tensor::zeros((1, rank), DType::F32, &self.device)?;
        let mut previous = Tensor::new(&[anchor], &self.device)?;
        let mut logits = Vec::with_capacity(drafts);
        for step in 0..drafts {
            let embedded = drafter
                .previous
                .index_select(&previous, 0)?
                .to_dtype(DType::F32)?;
            let raw = (Tensor::cat(&[&state, &embedded], 1)?.matmul(&drafter.recurrent)?
                + context.narrow(0, step, 1)?)?;
            let gate = candle_nn::ops::sigmoid(&raw.narrow(1, 0, rank)?)?;
            let candidate = raw.narrow(1, rank, rank)?.tanh()?;
            let latent = raw.narrow(1, 2 * rank, rank)?.tanh()?;
            state = ((&gate * &state)? + ((1.0 - &gate)? * candidate)?)?;
            let bias = k
                .linear(&drafter.fold, &latent)?
                .broadcast_add(&drafter.fold_bias)?;
            let row = ((((base.narrow(0, step, 1)? + bias)? / cap)?.tanh()? * cap)?)
                .narrow(1, 0, vocab)?;
            previous = row.argmax(1)?;
            logits.push(row);
        }
        Ok(Tensor::cat(&logits, 0)?)
    }
}
