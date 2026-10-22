#![forbid(unsafe_code)]
//! Fused Metal execution of the Qwen3.5 hybrid decoder.
//!
//! Stateless work (projections, norms, rotary, activations, attention) runs once over the
//! packed batch. The stateful DeltaNet convolution and scan dispatch once per sequence,
//! because each sequence's state is its own buffer, and write straight into batch-wide
//! outputs. Attention reads one paged K/V pool per layer built from committed pages and
//! new keys; P5 replaces that per-pass pool with persistent page-native storage.

use super::{Attention, DeltaNet, Mixer, Pass, Qwen35, Span};
use crate::hybrid::{Boundaries, LayerRead, LayerUpdate, PackedForward};
use candle_core::{DType, Tensor};
use tur_core::{
    Result,
    execution::{LayerStateKind, invalid},
};
use tur_metal::{
    kernels::{DeltaShape, Heads, Kernels, PagedKv},
    scratch::ScratchPlan,
};

/// Compiled kernels and per-model constants for fused execution.
pub(super) struct Fused {
    kernels: Kernels,
    inv_freq: Tensor,
    page: usize,
    delta: DeltaShape,
}

impl Fused {
    pub(super) fn new(model: &Qwen35) -> Result<Self> {
        let kernels = Kernels::new(&model.device)?;
        let d = model.dims;
        let page = model
            .description
            .layers
            .iter()
            .find_map(|layer| match layer {
                LayerStateKind::Attention { page_tokens, .. } => Some(*page_tokens),
                LayerStateKind::Recurrent { .. } => None,
            })
            .unwrap_or(1);
        Ok(Self {
            inv_freq: kernels.rotary_frequencies(d.rotary_dim, d.theta)?,
            page,
            delta: DeltaShape {
                key_heads: d.key_heads,
                value_heads: d.value_heads,
                key_dim: d.key_dim,
                value_dim: d.value_dim,
                b_col: d.channels + d.value_width,
                a_col: d.channels + d.value_width + d.value_heads,
            },
            kernels,
        })
    }

    /// Per-sequence page geometry of this pass: committed keys, then new keys, padded to
    /// whole pages. Shared by every attention layer, whose states have equal lengths.
    fn pool(&self, model: &Qwen35, pass: &Pass<'_, '_>) -> Result<Pool> {
        let page = self.page;
        let mut table = Vec::new();
        let mut first = Vec::with_capacity(pass.ids.len());
        let mut visible = Vec::with_capacity(pass.ids.len());
        let mut padding = Vec::with_capacity(pass.spans.len());
        for span in pass.spans {
            let past = *pass
                .positions
                .get(span.start)
                .ok_or_else(|| invalid("span outside packed positions"))?;
            let length = past + span.len;
            let blocks = length.div_ceil(page);
            let base = u32::try_from(table.len()).map_err(|_| invalid("page table overflow"))?;
            // The pass pool stores each sequence contiguously, so its pages are in order.
            let ids = (table.len()..table.len() + blocks)
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<u32>, _>>()
                .map_err(|_| invalid("page table overflow"))?;
            table.extend(ids);
            for t in 0..span.len {
                first.push(base);
                visible.push(u32::try_from(past + t + 1).map_err(|_| invalid("context overflow"))?);
            }
            padding.push(blocks * page - length);
        }
        let slots = table.len() * page;
        let d = model.dims;
        Ok(Pool {
            kv: self
                .kernels
                .paged_kv(page, slots, &table, &first, &visible)?,
            zeros: Tensor::zeros((page, d.kv_heads, d.head_dim), DType::F32, &model.device)?,
            padding,
        })
    }
}

/// A sequence's convolution history, recurrent matrix and boundary mode at `layer`.
fn recurrent<'a>(span: &Span<'a>, layer: usize) -> Result<(&'a Tensor, &'a Tensor, bool)> {
    let every = span.sequence.boundaries == Boundaries::EveryToken;
    match span.sequence.state.get(layer) {
        Some(LayerRead::Recurrent {
            convolution,
            recurrent,
        }) => Ok((convolution, recurrent, every)),
        _ => Err(invalid("state kind does not match layer")),
    }
}

/// Paged addressing of one pass plus the zero rows that complete each sequence's last page.
struct Pool {
    kv: PagedKv,
    zeros: Tensor,
    padding: Vec<usize>,
}

impl Qwen35 {
    /// Fused body of [`Qwen35::run`]: returns logits for `pass.rows` and pushes updates.
    pub(super) fn run_fused(
        &self,
        fused: &Fused,
        pass: &Pass<'_, '_>,
        updates: &mut [Vec<LayerUpdate>],
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let k = &fused.kernels;
        let d = self.dims;
        let positions = pass
            .positions
            .iter()
            .map(|&p| u32::try_from(p))
            .collect::<std::result::Result<Vec<u32>, _>>()
            .map_err(|_| invalid("position exceeds u32"))?;
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        let has_attention = self
            .layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::Attention(_)));
        let pool = if has_attention {
            Some(fused.pool(self, pass)?)
        } else {
            None
        };
        let ids = Tensor::new(pass.ids.as_slice(), &self.device)?;
        let mut x = self.embed.index_select(&ids, 0)?.to_dtype(DType::F32)?;
        let first = self
            .layers
            .first()
            .ok_or_else(|| invalid("model has no layers"))?;
        let mut h = k.rms_norm(&x, &first.input_norm, d.eps)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => {
                    let pool = pool
                        .as_ref()
                        .ok_or_else(|| invalid("attention layer without a K/V layout"))?;
                    self.fused_attention(
                        fused, a, &h, &positions, pool, index, pass.spans, updates,
                    )?
                }
                Mixer::Delta(net) => {
                    self.fused_delta(fused, net, &h, index, pass.spans, updates)?
                }
            };
            let (sum, normed) = k.add_rms_norm(&x, &mixed, &layer.post_norm, d.eps)?;
            let act = k.swiglu(&k.linear(&layer.gate_up, &normed)?)?;
            let mlp = k.linear(&layer.down, &act)?;
            // The residual add fuses with the next layer's input norm, or the final norm.
            let next = self
                .layers
                .get(index + 1)
                .map_or(&self.norm, |l| &l.input_norm);
            (x, h) = k.add_rms_norm(&sum, &mlp, next, d.eps)?;
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
        let rows = Tensor::new(pass.rows.as_slice(), &self.device)?;
        k.linear(&self.lm_head, &h.index_select(&rows, 0)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_attention(
        &self,
        fused: &Fused,
        a: &Attention,
        h: &Tensor,
        positions: &Tensor,
        pool: &Pool,
        layer: usize,
        spans: &[Span<'_>],
        updates: &mut [Vec<LayerUpdate>],
    ) -> Result<Tensor> {
        let k = &fused.kernels;
        let d = self.dims;
        let n = h.dim(0)?;
        let heads = |count: usize, head_stride: usize, col_offset: usize| Heads {
            count,
            dim: d.head_dim,
            head_stride,
            col_offset,
        };
        // q_proj rows interleave `q | gate` per head.
        let q_gate = k.linear(&a.q, h)?;
        let kv = k.linear(&a.kv, h)?;
        let q = k.qk_norm_rope(
            &q_gate,
            heads(d.heads, 2 * d.head_dim, 0),
            &a.q_norm,
            positions,
            &fused.inv_freq,
            d.eps,
        )?;
        let key = k.qk_norm_rope(
            &kv,
            heads(d.kv_heads, d.head_dim, 0),
            &a.k_norm,
            positions,
            &fused.inv_freq,
            d.eps,
        )?;
        let width = d.kv_heads * d.head_dim;
        let value = kv
            .narrow(1, width, width)?
            .contiguous()?
            .reshape((n, d.kv_heads, d.head_dim))?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for ((span, updates), &pad) in spans.iter().zip(updates.iter_mut()).zip(&pool.padding) {
            let LayerRead::Attention { pages } = &span.sequence.state[layer] else {
                return Err(invalid("state kind does not match layer"));
            };
            for (page_key, page_value) in pages {
                keys.push(page_key.clone());
                values.push(page_value.clone());
            }
            let new_key = key.narrow(0, span.start, span.len)?;
            let new_value = value.narrow(0, span.start, span.len)?;
            keys.push(new_key.clone());
            values.push(new_value.clone());
            if pad > 0 {
                keys.push(pool.zeros.narrow(0, 0, pad)?);
                values.push(pool.zeros.narrow(0, 0, pad)?);
            }
            updates.push(LayerUpdate::Attention {
                key: new_key,
                value: new_value,
            });
        }
        let attended = k.attention(
            &q,
            &Tensor::cat(&keys, 0)?,
            &Tensor::cat(&values, 0)?,
            &pool.kv,
            &q_gate,
            heads(d.heads, 2 * d.head_dim, d.head_dim),
        )?;
        k.linear(&a.o, &attended)
    }

    fn fused_delta(
        &self,
        fused: &Fused,
        net: &DeltaNet,
        h: &Tensor,
        layer: usize,
        spans: &[Span<'_>],
        updates: &mut [Vec<LayerUpdate>],
    ) -> Result<Tensor> {
        let k = &fused.kernels;
        let d = self.dims;
        let n = h.dim(0)?;
        let proj = k.linear(&net.proj, h)?;
        let mixed = k.output(n, d.channels)?;
        let mut histories = Vec::with_capacity(spans.len());
        for span in spans {
            let (history, _, every) = recurrent(span, layer)?;
            histories.push(k.delta_conv(
                &proj, span.start, span.len, history, &net.conv, every, &mixed, span.start,
            )?);
        }
        let mixed = k.finish(mixed, (n, d.channels))?;
        let core = k.output(n, d.value_width)?;
        for ((span, updates), history) in spans.iter().zip(updates.iter_mut()).zip(histories) {
            let (_, state, every) = recurrent(span, layer)?;
            let states = k.delta_scan(
                &mixed,
                &proj,
                span.start,
                span.len,
                &fused.delta,
                &net.dt_bias,
                &net.neg_a,
                state,
                every,
                &core,
                span.start,
            )?;
            updates.push(LayerUpdate::Recurrent {
                convolution: history,
                recurrent: states,
            });
        }
        let core = k.finish(core, (n * d.value_heads, d.value_dim))?;
        let gated = k.gated_rms(&core, &proj, d.channels, d.value_heads, &net.norm, d.eps)?;
        k.linear(&net.out, &gated)
    }

    /// Peak scratch of the fused pass, mirroring its live set layer by layer. State
    /// updates and logits are excluded: the engine charges those separately.
    pub(super) fn fused_workspace(
        &self,
        fused: &Fused,
        batch: &PackedForward<'_>,
    ) -> Result<usize> {
        let d = self.dims;
        let f = DType::F32;
        let n = batch.tokens.len();
        let hidden = self.embed.dim(1)?;
        let mut pool_slots = 0usize;
        let mut max_visible = 0usize;
        for sequence in batch.sequences {
            let past = *batch
                .positions
                .get(sequence.tokens.start)
                .ok_or_else(|| invalid("sequence outside packed positions"))?;
            let length = past + sequence.tokens.len();
            pool_slots += length.div_ceil(fused.page) * fused.page;
            max_visible = max_visible.max(length);
        }
        let mut plan = ScratchPlan::new();
        // Uploaded ids, positions, logit rows and page addressing.
        plan.transient(4 * n + batch.sequences.len(), DType::U32)?;
        let x = plan.reserve(n * hidden, f)?;
        let h = plan.reserve(n * hidden, f)?;
        let mut kinds = (false, false);
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Attention(a) if !kinds.0 => {
                    kinds.0 = true;
                    let q_gate = plan.reserve(n * a.q.out_features(), f)?;
                    let q = plan.reserve(n * d.heads * d.head_dim, f)?;
                    let keys = plan.reserve(pool_slots * d.kv_heads * d.head_dim, f)?;
                    let values = plan.reserve(pool_slots * d.kv_heads * d.head_dim, f)?;
                    let split =
                        fused
                            .kernels
                            .attention_scratch_bytes(n, d.heads, d.head_dim, max_visible);
                    plan.transient(split, DType::U8)?;
                    let attended = plan.reserve(n * d.heads * d.head_dim, f)?;
                    for region in [q, keys, values, q_gate] {
                        plan.release(region);
                    }
                    plan.transient(n * hidden, f)?;
                    plan.release(attended);
                }
                Mixer::Delta(net) if !kinds.1 => {
                    kinds.1 = true;
                    let proj = plan.reserve(n * net.proj.out_features(), f)?;
                    let mixed = plan.reserve(n * d.channels, f)?;
                    let core = plan.reserve(n * d.value_width, f)?;
                    let gated = plan.reserve(n * d.value_width, f)?;
                    for region in [mixed, core, proj] {
                        plan.release(region);
                    }
                    plan.transient(n * hidden, f)?;
                    plan.release(gated);
                }
                _ => {}
            }
        }
        // Residual sum and post-norm, then the MLP: gate|up, activation and down.
        let sum = plan.reserve(n * hidden, f)?;
        let normed = plan.reserve(n * hidden, f)?;
        let gate_up = plan.reserve(n * 2 * d.intermediate, f)?;
        let act = plan.reserve(n * d.intermediate, f)?;
        plan.release(gate_up);
        plan.transient(n * hidden, f)?;
        for region in [act, normed, sum, h, x] {
            plan.release(region);
        }
        Ok(plan.peak_bytes())
    }
}
