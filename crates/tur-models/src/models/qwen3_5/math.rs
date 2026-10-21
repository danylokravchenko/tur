//! F32 reference operations. State tensors are values; no input is mutated.

use candle_core::{Tensor, D};
use tur_core::{Result, execution::invalid};

pub(super) fn rms(x: &Tensor, weight: &Tensor, eps: f64, centered: bool) -> Result<Tensor> {
    let scale = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    let normalized = x.broadcast_mul(&scale)?;
    let weight = if centered { (weight + 1.0)? } else { weight.clone() };
    Ok(normalized.broadcast_mul(&weight)?)
}

fn l2(x: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?)
}

/// History stores the last K *unactivated* projected inputs, oldest first.
/// The extra oldest slot matches P2's K-wide state and is discarded before use.
pub(super) fn convolution(x: &Tensor, history: &Tensor, weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let (width, channels) = history.dims2()?;
    if width == 0 || x.dims() != [channels] || weight.dims() != [channels, 1, width] {
        return Err(invalid("invalid causal convolution dimensions"));
    }
    let tail = history.narrow(0, 1, width - 1)?;
    let next = Tensor::cat(&[tail, x.unsqueeze(0)?], 0)?;
    let output = (&next * weight.squeeze(1)?.t()?)?.sum(0)?;
    Ok((candle_nn::ops::silu(&output)?, next))
}

/// Normalized Gated DeltaNet: decay first, then correct the decayed prediction.
/// q/k are repeated per value-head group by the caller.
pub(super) fn delta(q: &Tensor, k: &Tensor, v: &Tensor, a: &Tensor, b: &Tensor,
    a_log: &Tensor, dt_bias: &Tensor, state: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_, key_dim) = q.dims2()?;
    let q = (l2(q)? / (key_dim as f64).sqrt())?;
    let k = l2(k)?;
    // log(1+exp(x)) = max(x,0) + log(1+exp(-abs(x))) avoids overflow.
    let dt = (a + dt_bias)?;
    let softplus = (dt.maximum(0.0)? + (dt.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
    let decay = (a_log.exp()? * softplus)?.neg()?.exp()?.unsqueeze(1)?.unsqueeze(2)?;
    let decayed = state.broadcast_mul(&decay)?;
    let prediction = decayed.broadcast_mul(&k.unsqueeze(2)?)?.sum(1)?;
    let correction = (v - prediction)?.broadcast_mul(&candle_nn::ops::sigmoid(b)?.unsqueeze(1)?)?;
    let next = (decayed + k.unsqueeze(2)?.broadcast_mul(&correction.unsqueeze(1)?)?)?;
    let output = next.broadcast_mul(&q.unsqueeze(2)?)?.sum(1)?;
    Ok((output, next))
}

/// Text positions share all three M-RoPE axes. Rotate only the leading dimensions
/// using split halves (not adjacent pairs); the remaining coordinates pass through.
pub(super) fn rotary(x: &Tensor, position: usize, dim: usize, theta: f64) -> Result<Tensor> {
    let (_, head_dim) = x.dims2()?;
    if dim < 2 || dim > head_dim || dim % 2 != 0 { return Err(invalid("invalid rotary dimension")); }
    let half = dim / 2;
    let angles = (0..half).map(|i| position as f32 / theta.powf((2 * i) as f64 / dim as f64) as f32).collect::<Vec<_>>();
    let angles = Tensor::from_vec(angles, (1, half), x.device())?;
    let (cos, sin) = (angles.cos()?, angles.sin()?);
    let left = x.narrow(1, 0, half)?;
    let right = x.narrow(1, half, half)?;
    let first = (left.broadcast_mul(&cos)? - right.broadcast_mul(&sin)?)?;
    let second = (right.broadcast_mul(&cos)? + left.broadcast_mul(&sin)?)?;
    Ok(Tensor::cat(&[first, second, x.narrow(1, dim, head_dim - dim)?], 1)?)
}

pub(super) fn repeat_heads(x: &Tensor, groups: usize) -> Result<Tensor> {
    let (heads, dim) = x.dims2()?;
    let total = heads.checked_mul(groups).ok_or_else(|| invalid("head count overflow"))?;
    Ok(x.unsqueeze(1)?.broadcast_as((heads, groups, dim))?.reshape((total, dim))?)
}

/// Reference GQA over concatenated detached pages and the current token.
pub(super) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, gate: &Tensor,
    pages: &[(Tensor, Tensor)], position: usize) -> Result<Tensor> {
    let (kv_heads, dim) = k.dims2()?;
    let (heads, qdim) = q.dims2()?;
    if kv_heads == 0 || heads % kv_heads != 0 || qdim != dim || v.dims() != k.dims() || gate.dims() != q.dims() {
        return Err(invalid("invalid gated attention dimensions"));
    }
    let mut length = 0usize;
    for (key, value) in pages {
        let (tokens, h, d) = key.dims3()?;
        if tokens == 0 || h != kv_heads || d != dim || value.dims() != key.dims() { return Err(invalid("invalid attention page")); }
        length = length.checked_add(tokens).ok_or_else(|| invalid("attention position overflow"))?;
    }
    if length != position { return Err(invalid("attention pages do not match position")); }
    let groups = heads / kv_heads;
    let mut keys: Vec<Tensor> = pages.iter().map(|(k, _)| k.clone()).collect();
    let mut values: Vec<Tensor> = pages.iter().map(|(_, v)| v.clone()).collect();
    keys.push(k.unsqueeze(0)?);
    values.push(v.unsqueeze(0)?);
    let length = length.checked_add(1).ok_or_else(|| invalid("attention length overflow"))?;
    let expand = |parts: &[Tensor]| -> Result<Tensor> {
        Ok(Tensor::cat(parts, 0)?.transpose(0, 1)?.unsqueeze(1)?
            .broadcast_as((kv_heads, groups, length, dim))?.reshape((heads, length, dim))?)
    };
    let keys = expand(&keys)?;
    let values = expand(&values)?;
    let scores = (q.unsqueeze(1)?.matmul(&keys.transpose(1, 2)?.contiguous()?)? / (dim as f64).sqrt())?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let output = probs.matmul(&values.contiguous()?)?.squeeze(1)?;
    Ok((output * candle_nn::ops::sigmoid(gate)?)?)
}
