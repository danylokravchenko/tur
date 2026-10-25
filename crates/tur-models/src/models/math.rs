//! Shared F32 normalization, rotary and grouped attention reference operations.
use candle_core::{D, Device, Tensor};
use tur_core::{Result, execution::invalid};
const ATTENTION_BLOCK: usize = 256;

/// `x * rsqrt(mean(x²) + eps) * scale` over the last dimension. Zero-centred norms pass
/// `1 + weight` as `scale`; the loader folds that offset in once.
pub(crate) fn rms(x: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
    let inv = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?.broadcast_mul(scale)?)
}

/// Repeat each head `groups` times consecutively (`repeat_interleave` on the head axis).
pub(crate) fn repeat_heads(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (n, heads, dim) = x.dims3()?;
    let total = heads
        .checked_mul(groups)
        .ok_or_else(|| invalid("head count overflow"))?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((n, heads, groups, dim))?
        .reshape((n, total, dim))?)
}

/// Rotary tables `[n, dim/2]` for text positions, computed in F32 like the reference.
pub(crate) fn rotary_tables(
    positions: &[usize],
    dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &p in positions {
        for &f in &inv {
            let angle = p as f32 * f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let shape = (positions.len(), half);
    Ok((
        Tensor::from_vec(cos, shape, device)?,
        Tensor::from_vec(sin, shape, device)?,
    ))
}

/// Rotate the leading `2 * cos.dim(1)` coordinates of `[n, heads, head_dim]` using split
/// halves; the remaining coordinates pass through.
pub(crate) fn rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (n, _, head_dim) = x.dims3()?;
    let half = cos.dim(1)?;
    if cos.dims() != [n, half] || 2 * half > head_dim {
        return Err(invalid("invalid rotary dimension"));
    }
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let left = x.narrow(2, 0, half)?;
    let right = x.narrow(2, half, half)?;
    let first = (left.broadcast_mul(&cos)? - right.broadcast_mul(&sin)?)?;
    let second = (right.broadcast_mul(&cos)? + left.broadcast_mul(&sin)?)?;
    let pass = x.narrow(2, 2 * half, head_dim - 2 * half)?;
    Ok(Tensor::cat(&[first, second, pass], 2)?)
}

/// Causal grouped-query attention for `n` new queries at positions `past..past+n`.
/// `keys`/`values` hold all `past + n` positions as `[length, kv_heads, head_dim]`.
pub(crate) fn attention(
    q: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    past: usize,
    scale: f64,
    window: Option<usize>,
) -> Result<Tensor> {
    let (n, heads, dim) = q.dims3()?;
    let (length, kv_heads, kdim) = keys.dims3()?;
    if kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || kdim != dim
        || values.dims() != keys.dims()
        || past.checked_add(n) != Some(length)
    {
        return Err(invalid("invalid gated attention dimensions"));
    }
    let groups = heads / kv_heads;
    let keys = repeat_heads(keys, groups)?.transpose(0, 1)?.contiguous()?;
    let values = repeat_heads(values, groups)?
        .transpose(0, 1)?
        .contiguous()?;
    let keys_t = keys.transpose(1, 2)?.contiguous()?;
    let mut blocks = Vec::with_capacity(n.div_ceil(ATTENTION_BLOCK));
    for start in (0..n).step_by(ATTENTION_BLOCK) {
        let rows = ATTENTION_BLOCK.min(n - start);
        let visible = past + start + rows;
        let q_block = q.narrow(0, start, rows)?.transpose(0, 1)?.contiguous()?;
        let scores = (q_block.matmul(&keys_t.narrow(2, 0, visible)?.contiguous()?)? * scale)?;
        let mask: Vec<f32> = (0..rows * visible)
            .map(|i| {
                if i % visible <= past + start + i / visible
                    && window.is_none_or(|w| (i % visible) + w > past + start + i / visible)
                {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, rows, visible), q.device())?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let block = probs.matmul(&values.narrow(1, 0, visible)?.contiguous()?)?;
        blocks.push(block.transpose(0, 1)?);
    }
    Ok(Tensor::cat(&blocks, 0)?.contiguous()?)
}

/// F32 top-k probabilities, normalized over the selected experts. Softmax over all
/// experts cancels in the renormalization; subtract the selected maximum for stability.
#[allow(
    clippy::float_cmp,
    reason = "routing ties include both signs of zero, as on the device"
)]
pub(crate) fn route(logits: &[f32], top_k: usize) -> Result<(Vec<usize>, Vec<f32>)> {
    if top_k == 0 || top_k > logits.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid MoE router scores or top-k"));
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        if logits[a] == logits[b] {
            a.cmp(&b)
        } else {
            logits[b].total_cmp(&logits[a])
        }
    });
    indices.truncate(top_k);
    let max = logits[indices[0]];
    let mut weights: Vec<f32> = indices.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok((indices, weights))
}
