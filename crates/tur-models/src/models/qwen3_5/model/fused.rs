#![forbid(unsafe_code)]
//! Fused Metal execution of the Qwen3.5 hybrid decoder.
//!
//! Stateless work (projections, norms, rotary, activations, attention) runs once over the
//! packed batch. The stateful DeltaNet convolution and scan dispatch once per sequence,
//! reading the committed slot and writing reserved slots of the layer's recurrent pool.
//! Attention stores new keys and values into the persistent page pool (quantizing to its
//! format) and reads every key in place through one page table shared by all layers; no
//! committed key is copied.

use super::{Attention, DeltaNet, Mixer, Pass, Qwen35};
use crate::hybrid::PackedForward;
use candle_core::{DType, Tensor};
use tur_core::{Result, execution::invalid};
use tur_metal::{
    kernels::{DeltaShape, Heads, Kernels, PagedKv, Slots},
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
        let page = model.description.kv.page_tokens;
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

    /// Page addressing of this pass, shared by every attention layer: each sequence's page
    /// table, and for each new token its first table entry and visible key count.
    fn paging(&self, model: &Qwen35, pass: &Pass<'_, '_>) -> Result<PagedKv> {
        let mut table = Vec::new();
        let mut first = Vec::with_capacity(pass.ids.len());
        let mut visible = Vec::with_capacity(pass.ids.len());
        for span in pass.spans {
            let base = u32::try_from(table.len()).map_err(|_| invalid("page table overflow"))?;
            table.extend_from_slice(&span.addresses.pages);
            for t in 0..span.len {
                first.push(base);
                visible.push(
                    u32::try_from(span.past + t + 1).map_err(|_| invalid("context overflow"))?,
                );
            }
        }
        let slots = pass
            .pool
            .attention(model.first_attention()?)?
            .key
            .data
            .dim(0)?;
        self.kernels
            .paged_kv(self.page, slots, &table, &first, &visible)
    }
}

impl Qwen35 {
    fn first_attention(&self) -> Result<usize> {
        self.layers
            .iter()
            .position(|l| matches!(l.mixer, Mixer::Attention(_)))
            .ok_or_else(|| invalid("model has no attention layer"))
    }
}

impl Qwen35 {
    /// Fused body of [`Qwen35::run`]: returns logits for `pass.rows` and writes state.
    pub(super) fn run_fused(
        &self,
        fused: &Fused,
        pass: &Pass<'_, '_>,
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
        let paging = if has_attention {
            Some(fused.paging(self, pass)?)
        } else {
            None
        };
        let has_delta = self
            .layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::Delta(_)));
        // One upload per sequence serves every recurrent layer: slot ids span all layers.
        let slots = if has_delta {
            pass.spans
                .iter()
                .map(|span| k.slots(&span.addresses.writes))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
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
                    let paging = paging
                        .as_ref()
                        .ok_or_else(|| invalid("attention layer without page addressing"))?;
                    self.fused_attention(fused, a, &h, &positions, pass, paging, index)?
                }
                Mixer::Delta(net) => self.fused_delta(fused, net, &h, pass, &slots, index)?,
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
        pass: &Pass<'_, '_>,
        paging: &PagedKv,
        layer: usize,
    ) -> Result<Tensor> {
        let k = &fused.kernels;
        let d = self.dims;
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
        let pool = pass.pool.attention(layer)?;
        // Values are read in place from the `k | v` projection.
        k.kv_write(
            &key,
            heads(d.kv_heads, d.head_dim, 0),
            &kv,
            heads(d.kv_heads, d.head_dim, d.kv_heads * d.head_dim),
            pool,
            paging,
        )?;
        let attended = k.attention(
            &q,
            pool,
            paging,
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
        pass: &Pass<'_, '_>,
        slots: &[Slots],
        layer: usize,
    ) -> Result<Tensor> {
        let k = &fused.kernels;
        let d = self.dims;
        let n = h.dim(0)?;
        let pool = pass.pool.recurrent(layer)?;
        let proj = k.linear(&net.proj, h)?;
        let mixed = k.output(n, d.channels)?;
        for (span, writes) in pass.spans.iter().zip(slots) {
            k.delta_conv(
                &proj,
                span.start,
                span.len,
                &pool.convolution,
                span.addresses.read,
                writes,
                &net.conv,
                &mixed,
                span.start,
            )?;
        }
        let mixed = k.finish(mixed, (n, d.channels))?;
        let core = k.output(n, d.value_width)?;
        for (span, writes) in pass.spans.iter().zip(slots) {
            k.delta_scan(
                &mixed,
                &proj,
                span.start,
                span.len,
                &fused.delta,
                &net.dt_bias,
                &net.neg_a,
                &pool.state,
                span.addresses.read,
                writes,
                &core,
                span.start,
            )?;
        }
        let core = k.finish(core, (n * d.value_heads, d.value_dim))?;
        let gated = k.gated_rms(&core, &proj, d.channels, d.value_heads, &net.norm, d.eps)?;
        k.linear(&net.out, &gated)
    }

    /// Peak scratch of the fused pass, mirroring its live set layer by layer. Pools and
    /// logits are excluded: the engine charges those separately.
    pub(super) fn fused_workspace(
        &self,
        fused: &Fused,
        batch: &PackedForward<'_>,
    ) -> Result<usize> {
        let contexts = batch
            .sequences
            .iter()
            .map(|sequence| {
                batch
                    .positions
                    .get(sequence.tokens.start)
                    .map(|past| past + sequence.tokens.len())
                    .ok_or_else(|| invalid("sequence outside packed positions"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.planned_workspace(fused, batch.tokens.len(), &contexts)
    }

    /// Peak scratch of a pass of `n` packed tokens whose sequences end at `contexts`.
    pub(super) fn planned_workspace(
        &self,
        fused: &Fused,
        n: usize,
        contexts: &[usize],
    ) -> Result<usize> {
        let d = self.dims;
        let f = DType::F32;
        let hidden = self.embed.dim(1)?;
        let pages: usize = contexts.iter().map(|c| c.div_ceil(fused.page)).sum();
        let max_visible = contexts.iter().copied().max().unwrap_or(0);
        let mut plan = ScratchPlan::new();
        // Uploaded ids, positions, logit rows, page addressing and recurrent slots.
        plan.transient(4 * n + pages + 2 * contexts.len(), DType::U32)?;
        let x = plan.reserve(n * hidden, f)?;
        let h = plan.reserve(n * hidden, f)?;
        let mut kinds = (false, false);
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Attention(a) if !kinds.0 => {
                    kinds.0 = true;
                    let q_gate = plan.reserve(n * a.q.out_features(), f)?;
                    let kv = plan.reserve(n * a.kv.out_features(), f)?;
                    let q = plan.reserve(n * d.heads * d.head_dim, f)?;
                    let key = plan.reserve(n * d.kv_heads * d.head_dim, f)?;
                    let split =
                        fused
                            .kernels
                            .attention_scratch_bytes(n, d.heads, d.head_dim, max_visible);
                    plan.transient(split, DType::U8)?;
                    let attended = plan.reserve(n * d.heads * d.head_dim, f)?;
                    for region in [q, key, kv, q_gate] {
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
