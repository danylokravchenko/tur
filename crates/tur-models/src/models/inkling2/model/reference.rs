//! Reference decoder over Candle operations: the numerical oracle the kernels are tested
//! against, on any device.
use super::{Attention, Executor, Inkling2, InputNorms, Layer, MessageGates, Pass};
use crate::models::math;
use candle_core::{D, DType, IndexOp, Tensor};
use tur_core::{Result, execution::invalid};

/// Epsilon of the gainless norms inside attention residuals, fixed by the architecture.
pub(super) const RESIDUAL_EPS: f64 = 1e-6;

/// `x / sqrt(mean(x^2) + eps)` over the last dimension.
fn unit_rms(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(x.broadcast_div(&(mean + eps)?.sqrt()?)?)
}

/// `x * min(threshold / (|x| + 1e-6), 1)` over the last dimension.
fn clip(x: &Tensor, threshold: f64) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let factor = (norm + 1e-6)?
        .recip()?
        .affine(threshold, 0.0)?
        .clamp(f64::NEG_INFINITY, 1.0)?;
    Ok(x.broadcast_mul(&factor)?)
}

/// `softcap(x, cap) = cap * tanh(x / cap)`.
fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    Ok(((x / cap)?.tanh()? * cap)?)
}

/// What one span's short convolution returns: outputs and the histories its slots take.
struct Convolved {
    output: Tensor,
    /// `[boundaries, taps - 1, channels]`.
    histories: Tensor,
}

/// `(conv(x) + x) / divisor` of one sequence: `x` follows `history` (`[taps - 1, C]`),
/// `taps` is `[C, taps]` by lag, and `dynamic` adds per-token taps per head of `head_dim`.
fn short_conv(
    x: &Tensor,
    history: &Tensor,
    taps: &Tensor,
    dynamic: Option<(&Tensor, usize)>,
    divisor: f64,
    ends: &[usize],
) -> Result<Convolved> {
    let (n, channels) = x.dims2()?;
    let (kept, width) = history.dims2()?;
    let count = taps.dim(1)?;
    if width != channels || count != kept + 1 || taps.dim(0)? != channels {
        return Err(invalid("invalid short convolution dimensions"));
    }
    let full = Tensor::cat(&[history, x], 0)?;
    let mut acc = Tensor::zeros((n, channels), DType::F32, x.device())?;
    for lag in 0..count {
        let input = full.narrow(0, kept - lag, n)?;
        let mut weight = taps.narrow(1, lag, 1)?.t()?.broadcast_as((n, channels))?;
        if let Some((dyn_taps, head_dim)) = dynamic {
            let heads = channels / head_dim;
            let per_head = Tensor::cat(
                &(0..heads)
                    .map(|h| dyn_taps.narrow(1, h * count + lag, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?,
                1,
            )?;
            let per_channel = per_head
                .unsqueeze(2)?
                .broadcast_as((n, heads, head_dim))?
                .reshape((n, channels))?;
            weight = (weight + per_channel)?;
        }
        acc = (acc + (input * weight)?)?;
    }
    let output = ((acc + x)? / divisor)?;
    let histories = ends
        .iter()
        .map(|&end| full.narrow(0, end, kept))
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Convolved {
        output,
        histories: Tensor::stack(&histories, 0)?,
    })
}

/// `1 + alpha * ln(max((position + 1) / floor, 1))` per position, as a `[n, 1, 1]` factor.
#[allow(
    clippy::suboptimal_flops,
    reason = "a fused multiply-add rounds differently from the reference"
)]
fn log_scale(
    model: &Inkling2,
    positions: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let t = &model.config.text;
    let floor = t.log_scaling_n_floor as f32;
    let values: Vec<f32> = positions
        .iter()
        .map(|&p| 1.0 + t.log_scaling_alpha as f32 * ((p as f32 + 1.0) / floor).max(1.0).ln())
        .collect();
    Ok(Tensor::from_vec(values, (positions.len(), 1, 1), device)?)
}

/// Attention of one sequence's `[n, heads, dim]` queries over all its keys, with the
/// relative bias, a sink logit per query and head, and its window.
struct Scores<'a> {
    rel: &'a Tensor,
    proj: &'a Tensor,
    sinks: &'a Tensor,
    window: Option<usize>,
    scale: f64,
}

fn attend(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    past: usize,
    scores: &Scores<'_>,
) -> Result<Tensor> {
    let (rows, heads, dim) = queries.dims3()?;
    let (context, kv_heads, _) = keys.dims3()?;
    let groups = heads / kv_heads;
    let key_heads = math::repeat_heads(keys, groups)?
        .transpose(0, 1)?
        .contiguous()?;
    let value_heads = math::repeat_heads(values, groups)?
        .transpose(0, 1)?
        .contiguous()?;
    let queries = queries.transpose(0, 1)?.contiguous()?;
    let dots = (queries.matmul(&key_heads.t()?.contiguous()?)? * scores.scale)?;
    // Bias by backward distance: [heads, rows, extent] gathered into [heads, rows, context].
    let extent = scores.proj.dim(1)?;
    let table = scores
        .rel
        .transpose(0, 1)?
        .contiguous()?
        .broadcast_matmul(scores.proj)?;
    let mut index = Vec::with_capacity(rows * context);
    let mut near_mask = Vec::with_capacity(rows * context);
    let mut visible = Vec::with_capacity(rows * context);
    for i in 0..rows {
        let position = past + i;
        for j in 0..context {
            let distance = position.checked_sub(j);
            let inside = distance.is_some_and(|d| scores.window.is_none_or(|w| d < w));
            let near = distance.filter(|&d| d < extent);
            index.push(u32::try_from(near.unwrap_or(0)).map_err(|_| invalid("distance"))?);
            near_mask.push(if near.is_some() { 1f32 } else { 0.0 });
            visible.push(if inside { 0f32 } else { f32::NEG_INFINITY });
        }
    }
    let device = queries.device();
    let shape = (1, rows, context);
    let index = Tensor::from_vec(index, shape, device)?
        .broadcast_as((heads, rows, context))?
        .contiguous()?;
    let bias = table
        .contiguous()?
        .gather(&index, 2)?
        .broadcast_mul(&Tensor::from_vec(near_mask, shape, device)?)?;
    let logits = (dots + bias)?.broadcast_add(&Tensor::from_vec(visible, shape, device)?)?;
    let sinks = scores.sinks.t()?.unsqueeze(2)?;
    let weights =
        candle_nn::ops::softmax_last_dim(&Tensor::cat(&[logits, sinks], 2)?.contiguous()?)?
            .narrow(2, 0, context)?
            .contiguous()?;
    Ok(weights
        .matmul(&value_heads)?
        .transpose(0, 1)?
        .reshape((rows, heads * dim))?)
}

/// The reference's view of one pass.
pub(super) struct Reference<'a, 'b> {
    pub(super) model: &'a Inkling2,
    pub(super) pass: &'a Pass<'a, 'b>,
}

impl Reference<'_, '_> {
    fn config(&self) -> &super::Config {
        &self.model.config
    }

    /// Mix `message` into `x` through gate `i` of `gates`.
    fn gate(&self, gates: &MessageGates, i: usize, x: &Tensor, message: &Tensor) -> Result<Tensor> {
        let eps = self.config().text.rms_norm_eps;
        let (down, up) = gates.get(i)?;
        let pair = Tensor::cat(&[unit_rms(x, eps)?, unit_rms(message, eps)?], 1)?;
        let gates = pair.matmul(&down)?.matmul(&up)?.tanh()?;
        let hidden = x.dim(1)?;
        let forget = gates.narrow(1, 0, hidden)?;
        let gain = gates.narrow(1, hidden, hidden)?;
        Ok(((x - (&forget * x)?)? + (gain * message)?)?)
    }

    /// Per sequence: history slot read, convolution, and the histories to write.
    fn conv(
        &self,
        state: usize,
        x: &Tensor,
        taps: &Tensor,
        (column, dynamic, divisor): (usize, Option<(&Tensor, usize)>, f64),
        pending: &mut [Vec<Tensor>],
    ) -> Result<Tensor> {
        let store = self.pass.pool.recurrent(state)?;
        let channels = x.dim(1)?;
        let mut outputs = Vec::with_capacity(self.pass.spans.len());
        for (span, pending) in self.pass.spans.iter().zip(pending.iter_mut()) {
            let (history, _) = store.read(span.addresses.read)?;
            let history = history.narrow(1, column, channels)?.contiguous()?;
            let rows = |t: &Tensor| t.narrow(0, span.start, span.len);
            let dynamic = dynamic.map(|(d, h)| rows(d).map(|d| (d, h))).transpose()?;
            let conv = short_conv(
                &rows(x)?.contiguous()?,
                &history,
                taps,
                dynamic.as_ref().map(|(d, h)| (d, *h)),
                divisor,
                &span.ends,
            )?;
            outputs.push(conv.output);
            pending.push(conv.histories);
        }
        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Store each sequence's histories, concatenated in column order, after each of its
    /// boundaries.
    fn commit(&self, state: usize, pending: Vec<Vec<Tensor>>) -> Result<()> {
        let store = self.pass.pool.recurrent(state)?;
        for (span, histories) in self.pass.spans.iter().zip(pending) {
            let histories = Tensor::cat(&histories, 2)?;
            let placeholder =
                Tensor::zeros((span.ends.len(), 1, 1, 1), DType::F32, histories.device())?;
            for (b, &slot) in span.addresses.writes.iter().enumerate() {
                store.write(slot, &histories, &placeholder, b)?;
            }
        }
        Ok(())
    }

    fn attention(
        &self,
        application: usize,
        index: usize,
        inputs: [&Tensor; 3],
        pending: &mut [Vec<Tensor>],
    ) -> Result<Tensor> {
        let config = self.config();
        let t = &config.text;
        let a: &Attention = &self.model.layers[index].attention;
        let (heads, dim, rel) = (t.num_attention_heads, t.head_dim, t.d_rel);
        let (kv_heads, taps) = (t.num_key_value_heads, t.sconv_kernel_size);
        let n = inputs[0].dim(0)?;
        let sliding = config.sliding(index);
        let columns = config.conv_columns();
        let state = 2 * application + 1;
        let projected = a.query.forward(inputs[0])?;
        let tau = (!sliding)
            .then(|| log_scale(self.model, self.pass.positions, projected.device()))
            .transpose()?;
        let scaled = |x: Tensor| -> Result<Tensor> {
            Ok(match &tau {
                Some(tau) => x.broadcast_mul(tau)?,
                None => x,
            })
        };
        let q = projected
            .narrow(1, 0, heads * dim)?
            .reshape((n, heads, dim))?;
        let q = scaled(clip(
            &q.broadcast_mul(&a.q_scale.reshape((heads, dim))?)?,
            t.qk_l2_clip_threshold,
        )?)?;
        let gate = projected.narrow(1, heads * dim, heads * dim)?;
        let states = projected
            .narrow(1, 2 * heads * dim, heads * rel)?
            .reshape((n, heads, rel))?;
        let states = scaled(clip(
            &states.broadcast_mul(&a.rel_scale.reshape((heads, rel))?)?,
            t.rel_l2_clip_threshold,
        )?)?;
        let sinks = projected.narrow(1, 2 * heads * dim + heads * rel, heads)?;
        let width = kv_heads * dim;
        let divisor = if sliding {
            ((taps + 1) as f64).sqrt()
        } else {
            ((2 * taps + 1) as f64).sqrt()
        };
        let mut side =
            |input: &Tensor, weights: &tur_metal::packed::PackedLinear, conv: &Tensor, column| {
                let projected = weights.forward(input)?;
                let dynamic = (!sliding)
                    .then(|| projected.narrow(1, width, kv_heads * taps))
                    .transpose()?;
                let rows = projected.narrow(1, 0, width)?;
                let spec = (column, dynamic.as_ref().map(|d| (d, dim)), divisor);
                Ok::<_, tur_core::TurError>(
                    self.conv(state, &rows, conv, spec, pending)?
                        .reshape((n, kv_heads, dim))?,
                )
            };
        let key = clip(
            &side(inputs[1], &a.key, &a.key_taps, columns.key)?,
            t.qk_l2_clip_threshold,
        )?;
        let value = unit_rms(
            &side(inputs[2], &a.value, &a.value_taps, columns.value)?,
            t.rms_norm_eps,
        )?;
        let store = super::attention_store(self.pass.pool, application)?;
        let window = sliding.then_some(t.sliding_window_size);
        let mut outputs = Vec::with_capacity(self.pass.spans.len());
        for s in self.pass.spans {
            let table = super::table(&s.addresses, sliding);
            store.scatter(
                table,
                s.past,
                &key.narrow(0, s.start, s.len)?,
                &value.narrow(0, s.start, s.len)?,
            )?;
            let (keys, values) = store.gather(table, s.past + s.len)?;
            let rows = |x: &Tensor| x.narrow(0, s.start, s.len);
            outputs.push(attend(
                &rows(&q)?,
                &keys,
                &values,
                s.past,
                &Scores {
                    rel: &rows(&states)?,
                    proj: &a.rel_proj,
                    sinks: &rows(&sinks)?,
                    window,
                    scale: (dim as f64).sqrt().recip(),
                },
            )?);
        }
        let attended = Tensor::cat(&outputs, 0)?;
        let gated = (attended * candle_nn::ops::sigmoid(&gate.contiguous()?)?)?;
        a.output.forward(&gated)
    }

    /// Routing decisions cross to the host; each selected expert projects all of its tokens
    /// at once, and outputs accumulate in expert ID order.
    fn experts(&self, layer: &Layer, h: &Tensor) -> Result<Tensor> {
        let t = &self.config().text;
        let e = &layer.experts;
        let (inner, hidden) = (t.intermediate_size, t.hidden_size);
        let logits = h.matmul(&e.router.t()?.contiguous()?)?;
        let scores = candle_nn::ops::sigmoid(&logits)?.to_vec2::<f32>()?;
        let bias = e.bias.to_vec1::<f32>()?;
        let factors = e.factors.to_vec2::<f32>()?;
        let mut groups = vec![(Vec::new(), Vec::new()); t.n_routed_experts];
        for (token, scores) in scores.iter().enumerate() {
            let mut order: Vec<usize> = (0..scores.len()).collect();
            // Highest score plus bias first; ties to the lower expert ID.
            order.sort_by(|&a, &b| {
                (scores[b] + bias[b])
                    .total_cmp(&(scores[a] + bias[a]))
                    .then(a.cmp(&b))
            });
            let selected = &order[..t.num_experts_per_tok];
            let total: f32 = selected.iter().map(|&i| scores[i]).sum();
            let token = u32::try_from(token).map_err(|_| invalid("too many expert tokens"))?;
            for &expert in selected {
                groups[expert].0.push(token);
                groups[expert]
                    .1
                    .push(scores[expert] / (total + 1e-20) * t.route_scale as f32);
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
            let [up_factor, down_factor] = factors[expert][..] else {
                return Err(invalid("expert factors are not pairs"));
            };
            let gate_up = (e
                .gate_up
                .forward_rows(&rows, expert * 2 * inner, 2 * inner)?
                * f64::from(up_factor))?
            .reshape((count, inner, 2))?;
            let gate = gate_up.i((.., .., 0))?;
            let up = gate_up.i((.., .., 1))?;
            let act = ((softcap(&gate, t.mlp_gate_softcap)?
                * candle_nn::ops::sigmoid(&gate.contiguous()?)?)?
                * softcap(&up, t.mlp_up_softcap)?)?;
            let routed =
                (e.down
                    .forward_rows(&(act * f64::from(down_factor))?, expert * hidden, hidden)?)
                .broadcast_mul(&Tensor::from_vec(weights, (count, 1), h.device())?)?;
            output = output.index_add(&ids, &routed, 0)?;
        }
        Ok(output)
    }

    fn inputs(
        &self,
        layer: &Layer,
        index: usize,
        x: &Tensor,
        blocks: &[Tensor],
        messages: Option<&Tensor>,
    ) -> Result<[Tensor; 3]> {
        let config = self.config();
        let eps = config.text.rms_norm_eps;
        let mut inputs = match &layer.attention_queries {
            None => [x.clone(), x.clone(), x.clone()],
            Some(queries) => {
                let read = self.residual(blocks, x, queries)?;
                if config.sliding(index) {
                    let h = read.get(0)?;
                    [h.clone(), h.clone(), h]
                } else {
                    [read.get(0)?, read.get(1)?, read.get(2)?]
                }
            }
        };
        if let (Some(messages), Some(gates)) = (messages, &layer.gates) {
            for (i, input) in inputs.iter_mut().enumerate() {
                *input = self.gate(gates, i, input, &messages.get(i)?)?;
            }
        }
        match &layer.norms {
            InputNorms::Shared(norm) => inputs.map(|input| math::rms(&input, norm, eps)),
            InputNorms::Separate(norms) => {
                let [query, key, value] = inputs;
                [
                    math::rms(&query, &norms.get(0)?, eps),
                    math::rms(&key, &norms.get(1)?, eps),
                    math::rms(&value, &norms.get(2)?, eps),
                ]
            }
        }
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| invalid("attention inputs are not three"))
    }
}

impl Executor for Reference<'_, '_> {
    fn norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        math::rms(x, weight, self.config().text.rms_norm_eps)
    }

    fn residual(&self, blocks: &[Tensor], partial: &Tensor, queries: &Tensor) -> Result<Tensor> {
        let mut states: Vec<Tensor> = blocks.to_vec();
        states.push(partial.clone());
        let normed = unit_rms(&Tensor::stack(&states, 1)?, RESIDUAL_EPS)?;
        let scale = (partial.dim(1)? as f64).sqrt().recip();
        // [tokens, states, queries] scores, softmax over the states.
        let scores = (normed.broadcast_matmul(&queries.t()?.contiguous()?)? * scale)?;
        let weights = candle_nn::ops::softmax(&scores, 1)?;
        Ok(weights
            .transpose(1, 2)?
            .contiguous()?
            .matmul(&normed)?
            .transpose(0, 1)?
            .contiguous()?)
    }

    fn layer(
        &self,
        application: usize,
        index: usize,
        x: &Tensor,
        blocks: &mut Vec<Tensor>,
        messages: Option<&Tensor>,
    ) -> Result<Tensor> {
        let config = self.config();
        let text = &config.text;
        let layer = &self.model.layers[index];
        let columns = config.conv_columns();
        let state = 2 * application + 1;
        let divisor = ((text.sconv_kernel_size + 1) as f64).sqrt();
        let mut pending: Vec<Vec<Tensor>> = vec![Vec::new(); self.pass.spans.len()];
        let [query, key, value] = self.inputs(layer, index, x, blocks, messages)?;
        let attended = self.attention(application, index, [&query, &key, &value], &mut pending)?;
        let attended = self.conv(
            state,
            &attended,
            &layer.attention_taps,
            (columns.attention, None, divisor),
            &mut pending,
        )?;
        let residual = if config.boundary(index) {
            blocks.push(x.clone());
            attended
        } else {
            (x + attended)?
        };
        let mut read = self.residual(blocks, &residual, &layer.mlp_query)?.get(0)?;
        if let (Some(messages), Some(gates)) = (messages, &layer.gates) {
            read = self.gate(gates, 3, &read, &messages.get(3)?)?;
        }
        let routed = self.experts(
            layer,
            &math::rms(&read, &layer.post_norm, text.rms_norm_eps)?,
        )?;
        let routed = self.conv(
            state,
            &routed,
            &layer.mlp_taps,
            (columns.mlp, None, divisor),
            &mut pending,
        )?;
        self.commit(state, pending)?;
        Ok((residual + routed)?)
    }
}

impl Inkling2 {
    pub(super) fn run_reference(
        &self,
        pass: &Pass<'_, '_>,
        x: &Tensor,
        capture: super::Capture<'_>,
    ) -> Result<Tensor> {
        self.decode(&Reference { model: self, pass }, x, capture)
    }
}
