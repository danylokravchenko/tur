#![forbid(unsafe_code)]
//! Fused Metal execution of the Qwen3.5 hybrid decoder.
//!
//! Every layer runs once over the packed batch: projections, norms, rotary, activations,
//! attention and the stateful `DeltaNet` convolution and scan, which read each sequence's
//! committed slot and write its reserved slots of the layer's recurrent pool.
//! Attention stores new keys and values into the persistent page pool (quantizing to its
//! format) and reads every key in place through one page table shared by all layers; no
//! committed key is copied.

use super::{Attention, Config, DeltaNet, Dims, LayerKind, Mixer, Mlp, MtpHead, Pass, Qwen35};
use crate::hybrid::PackedForward;
use candle_core::{DType, Tensor};
use tur_core::{
    Result,
    execution::{KvLayout, checked_elements, invalid},
};
use tur_metal::{
    kernels::{DeltaShape, Heads, Kernels, PagedKv, RecurrentSpan, RecurrentSpans},
    packed::PackedLinear,
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
    pub(super) fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub(super) fn new(model: &Qwen35) -> Result<Self> {
        let kernels = Kernels::new(&model.device)?;
        let d = model.dims;
        let page = model.description.kv.page_tokens;
        Ok(Self {
            inv_freq: kernels.rotary_frequencies(d.rotary_dim, d.theta)?,
            page,
            delta: delta_shape(&d),
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
    /// Fused body of [`Qwen35::run`]: returns logits for `pass.rows`, and every row's final
    /// hidden state when `pass.hidden`, and writes state.
    pub(super) fn run_fused(
        &self,
        fused: &Fused,
        pass: &Pass<'_, '_>,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let k = &fused.kernels;
        let d = self.dims;
        let positions = self.position_tensor(pass)?;
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
        // One upload serves every recurrent layer: slot ids span all layers.
        let recurrent = if has_delta {
            let spans: Vec<_> = pass
                .spans
                .iter()
                .map(|span| RecurrentSpan {
                    tokens: span.len,
                    position: span.past,
                    read: span.addresses.read,
                    writes: &span.addresses.writes,
                })
                .collect();
            Some(k.recurrent_spans(&spans)?)
        } else {
            None
        };
        let ids = Tensor::new(pass.ids.as_slice(), &self.device)?;
        let mut x = self.embedding(&ids)?;
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
                Mixer::Delta(net) => {
                    let spans = recurrent
                        .as_ref()
                        .ok_or_else(|| invalid("recurrent layer without slot addressing"))?;
                    self.fused_delta(fused, net, &h, pass, spans, index)?
                }
            };
            let (sum, normed) = k.add_rms_norm(&x, &mixed, &layer.post_norm, d.eps)?;
            let mlp = match &layer.mlp {
                Mlp::Dense(mlp) => {
                    k.linear(&mlp.down, &k.swiglu(&k.linear(&mlp.gate_up, &normed)?)?)?
                }
                Mlp::Moe(moe) => moe.fused(k, &normed)?,
            };
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
        let hidden = pass.hidden.then(|| h.clone());
        Ok((self.row_logits(k, &self.lm_head, &h, pass)?, hidden))
    }

    /// Fused draft-head pass: logits for `pass.rows`; writes the head's keys and values.
    pub(super) fn draft_fused(
        &self,
        fused: &Fused,
        mtp: &MtpHead,
        pass: &Pass<'_, '_>,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let k = &fused.kernels;
        let d = self.dims;
        let Mixer::Attention(a) = &mtp.layer.mixer else {
            return Err(invalid("draft head layer is not attention"));
        };
        let positions = self.position_tensor(pass)?;
        let ids = Tensor::new(pass.ids.as_slice(), &self.device)?;
        let embedded = self.embedding(&ids)?;
        let joined = Tensor::cat(
            &[
                k.rms_norm(&embedded, &mtp.embedding_norm, d.eps)?,
                k.rms_norm(hidden, &mtp.hidden_norm, d.eps)?,
            ],
            1,
        )?;
        let x = k.linear(&mtp.fc, &joined)?;
        let h = k.rms_norm(&x, &mtp.layer.input_norm, d.eps)?;
        let paging = fused.paging(self, pass)?;
        let mixed =
            self.fused_attention(fused, a, &h, &positions, pass, &paging, self.layers.len())?;
        let (sum, normed) = k.add_rms_norm(&x, &mixed, &mtp.layer.post_norm, d.eps)?;
        let Mlp::Dense(mlp) = &mtp.layer.mlp else {
            return Err(invalid("draft head MLP is not dense"));
        };
        let out = k.linear(&mlp.down, &k.swiglu(&k.linear(&mlp.gate_up, &normed)?)?)?;
        let (_, h) = k.add_rms_norm(&sum, &out, &mtp.norm, d.eps)?;
        self.row_logits(k, mtp.lm_head(&self.lm_head), &h, pass)
    }

    fn position_tensor(&self, pass: &Pass<'_, '_>) -> Result<Tensor> {
        let positions = pass
            .positions
            .iter()
            .map(|&p| u32::try_from(p))
            .collect::<std::result::Result<Vec<u32>, _>>()
            .map_err(|_| invalid("position exceeds u32"))?;
        Ok(Tensor::new(positions.as_slice(), &self.device)?)
    }

    /// `head`'s logits for the rows of `h` the pass requested.
    fn row_logits(
        &self,
        k: &Kernels,
        head: &PackedLinear,
        h: &Tensor,
        pass: &Pass<'_, '_>,
    ) -> Result<Tensor> {
        if pass.rows.is_empty() {
            return Ok(Tensor::zeros(
                (0, head.out_features()),
                DType::F32,
                &self.device,
            )?);
        }
        let rows = Tensor::new(pass.rows.as_slice(), &self.device)?;
        k.linear(head, &h.index_select(&rows, 0)?)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per input the attention dispatch binds"
    )]
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
        spans: &RecurrentSpans,
        layer: usize,
    ) -> Result<Tensor> {
        let k = &fused.kernels;
        let d = self.dims;
        let pool = pass.pool.recurrent(layer)?;
        let proj = k.linear(&net.proj, h)?;
        let mixed = k.delta_conv(&proj, spans, &pool.convolution, &net.conv)?;
        let core = k.delta_scan(
            &mixed,
            &proj,
            spans,
            &fused.delta,
            &net.dt_bias,
            &net.neg_a,
            &pool.state,
        )?;
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
        workspace(
            &self.config,
            &fused.kernels,
            &self.description.kv,
            n,
            contexts,
        )
    }
}

/// Recurrent geometry of the fused kernels: the projection row is `q | k | v | z | b | a`.
fn delta_shape(d: &Dims) -> DeltaShape {
    DeltaShape {
        key_heads: d.key_heads,
        value_heads: d.value_heads,
        key_dim: d.key_dim,
        value_dim: d.value_dim,
        b_col: d.channels + d.value_width,
        a_col: d.channels + d.value_width + d.value_heads,
    }
}

/// The same live-set calculation before and after weight loading.
pub(super) fn workspace(
    config: &Config,
    kernels: &Kernels,
    kv: &KvLayout,
    n: usize,
    contexts: &[usize],
) -> Result<usize> {
    let (page, kv_format) = (kv.page_tokens, kv.format);
    let d = Dims::new(config)?;
    let f = DType::F32;
    let hidden = config.text.hidden_size;
    let pages = contexts.iter().try_fold(0usize, |total, context| {
        total
            .checked_add(context.div_ceil(page))
            .ok_or_else(|| invalid("page table size overflow"))
    })?;
    let elements = |width| checked_elements(&[n, width]);
    let q_width = checked_elements(&[d.heads, d.head_dim])?;
    let kv_width = checked_elements(&[d.kv_heads, d.head_dim])?;
    let projection_width = d
        .value_heads
        .checked_mul(2)
        .and_then(|gates| gates.checked_add(d.value_width))
        .and_then(|width| width.checked_add(d.channels))
        .ok_or_else(|| invalid("recurrent projection width overflow"))?;
    let max_visible = contexts.iter().copied().max().unwrap_or(0);
    let mut plan = ScratchPlan::new();
    // Addressing stays live for the whole pass. Bound each upload's allocator rounding,
    // including recurrent span/order tables and attention query tiles.
    let rounded = |value: usize| {
        if value == 0 {
            Some(0)
        } else {
            value.checked_next_power_of_two()
        }
    };
    let addressing = rounded(n)
        .and_then(|tokens| tokens.checked_mul(7))
        .and_then(|tokens| tokens.checked_add(rounded(pages)?))
        .and_then(|total| total.checked_add(rounded(contexts.len())?.checked_mul(16)?))
        .ok_or_else(|| invalid("addressing size overflow"))?;
    let addressing = plan.reserve(addressing, DType::U32)?;
    let x = plan.reserve(elements(hidden)?, f)?;
    let h = plan.reserve(elements(hidden)?, f)?;
    let mut kinds = (false, false);
    for layer in config.layer_kinds() {
        match layer {
            LayerKind::FullAttention if !kinds.0 => {
                kinds.0 = true;
                let q_gate = plan.reserve(
                    elements(
                        q_width
                            .checked_mul(2)
                            .ok_or_else(|| invalid("query width overflow"))?,
                    )?,
                    f,
                )?;
                let kv = plan.reserve(
                    elements(
                        kv_width
                            .checked_mul(2)
                            .ok_or_else(|| invalid("K/V width overflow"))?,
                    )?,
                    f,
                )?;
                let q = plan.reserve(elements(q_width)?, f)?;
                let key = plan.reserve(elements(kv_width)?, f)?;
                let split = kernels.attention_scratch_bytes(
                    n,
                    d.heads,
                    d.kv_heads,
                    d.head_dim,
                    max_visible,
                    kv_format,
                );
                plan.transient(split, DType::U8)?;
                let attended = plan.reserve(elements(q_width)?, f)?;
                for region in [q, key, kv, q_gate] {
                    plan.release(region);
                }
                plan.transient(elements(hidden)?, f)?;
                plan.release(attended);
            }
            LayerKind::LinearAttention if !kinds.1 => {
                kinds.1 = true;
                let proj = plan.reserve(elements(projection_width)?, f)?;
                let mixed = plan.reserve(elements(d.channels)?, f)?;
                // The scan's prepared rows (q and k scales, q·k and gates) live beside
                // its output.
                let prepared = plan.reserve(elements(delta_shape(&d).prepared_width())?, f)?;
                let core = plan.reserve(elements(d.value_width)?, f)?;
                plan.release(prepared);
                let gated = plan.reserve(elements(d.value_width)?, f)?;
                for region in [mixed, core, proj] {
                    plan.release(region);
                }
                plan.transient(elements(hidden)?, f)?;
                plan.release(gated);
            }
            _ => {}
        }
    }
    // Residual sum and post-norm, then the MLP: gate|up, activation and down.
    let sum = plan.reserve(elements(hidden)?, f)?;
    let normed = plan.reserve(elements(hidden)?, f)?;
    let gate_up = plan.reserve(
        elements(
            d.intermediate
                .checked_mul(2)
                .ok_or_else(|| invalid("MLP width overflow"))?,
        )?,
        f,
    )?;
    let act = plan.reserve(elements(d.intermediate)?, f)?;
    if let Some(moe) = config.moe {
        // The shared expert uses the dense regions above. Coexisting with them: router
        // logits, the shared output and its gate per token; per assignment the routing
        // pair, permutation and tile slots, expert gate|up, activation and down rows; and
        // block histograms plus group offsets. Doubled for allocator rounding.
        let k = moe.num_experts_per_tok;
        let assigned = checked_elements(&[n, k])?;
        let per_assignment = moe
            .moe_intermediate_size
            .checked_mul(3)
            .and_then(|v| v.checked_add(hidden))
            .and_then(|v| v.checked_add(8))
            .ok_or_else(|| invalid("MoE workspace overflow"))?;
        let per_token = moe
            .num_experts
            .checked_add(hidden)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| invalid("MoE workspace overflow"))?;
        let routing = assigned
            .div_ceil(256)
            .checked_add(1)
            .and_then(|blocks| blocks.checked_mul(moe.num_experts))
            .ok_or_else(|| invalid("MoE workspace overflow"))?;
        let extra = checked_elements(&[assigned, per_assignment])?
            .checked_add(elements(per_token)?)
            .and_then(|v| v.checked_add(routing))
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("MoE workspace overflow"))?;
        plan.transient(extra, f)?;
    }
    plan.release(gate_up);
    plan.transient(elements(hidden)?, f)?;
    for region in [act, normed, sum, h, x, addressing] {
        plan.release(region);
    }
    Ok(plan.peak_bytes())
}
