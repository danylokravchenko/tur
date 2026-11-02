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

/// `x` with each input's rows replacing its packed range; `start_of` maps that range to `x`'s
/// rows, and rows of sequences that failed validation map to nothing and are skipped with them.
pub(crate) fn splice_embedded(
    x: &Tensor,
    inputs: &[tur_core::execution::EmbeddedInput],
    start_of: impl Fn(&std::ops::Range<usize>) -> Option<usize>,
) -> Result<Tensor> {
    if inputs.is_empty() {
        return Ok(x.clone());
    }
    let (rows, width) = x.dims2()?;
    let mut pieces = Vec::new();
    let mut cursor = 0;
    for input in inputs {
        let Some(start) = start_of(&input.tokens) else {
            continue;
        };
        if start < cursor || input.rows.dims2()? != (input.tokens.len(), width) {
            return Err(invalid("embedded input rows do not match the model width"));
        }
        if start > cursor {
            pieces.push(x.narrow(0, cursor, start - cursor)?);
        }
        pieces.push(input.rows.to_device(x.device())?);
        cursor = start + input.tokens.len();
    }
    if pieces.is_empty() {
        return Ok(x.clone());
    }
    if cursor < rows {
        pieces.push(x.narrow(0, cursor, rows - cursor)?);
    }
    Ok(Tensor::cat(&pieces, 0)?)
}

/// Rotary whose frequencies each read one coordinate of a position: multimodal text
/// positions and image patches. The leading `runs` runs of `run` columns rotate apart.
#[derive(Clone, Debug)]
pub(crate) struct AxialTable {
    pub(crate) inv_freq: Vec<f32>,
    pub(crate) axis: Vec<u32>,
    pub(crate) run: usize,
    pub(crate) runs: usize,
}

fn inverse_frequencies(count: usize, dim: usize, theta: f64) -> Vec<f32> {
    (0..count)
        .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / dim as f32))
        .collect()
}

impl AxialTable {
    /// Text rotary over `dim` columns, with interleaved (t, h, w) frequencies: every third
    /// frequency reads the same coordinate, up to each axis's `sections` share.
    pub(crate) fn interleaved(dim: usize, theta: f64, sections: [usize; 3]) -> Self {
        let half = dim / 2;
        let mut axis = vec![0u32; half];
        for (coordinate, section) in [(1u32, sections[1]), (2, sections[2])] {
            for at in (coordinate as usize..3 * section).step_by(3) {
                if let Some(slot) = axis.get_mut(at) {
                    *slot = coordinate;
                }
            }
        }
        Self {
            inv_freq: inverse_frequencies(half, dim, theta),
            axis,
            run: dim,
            runs: 1,
        }
    }

    /// Image patches: `dim / 2` columns of a head per (first, second) coordinate's frequencies,
    /// in the reference's run layout.
    pub(crate) fn patches(dim: usize, theta: f64, run: usize, runs: usize) -> Self {
        let quarter = dim / 4;
        let frequencies = inverse_frequencies(quarter, dim / 2, theta);
        Self {
            inv_freq: frequencies.repeat(2),
            axis: [vec![0u32; quarter], vec![1u32; quarter]].concat(),
            run,
            runs,
        }
    }

    /// `[rows, frequencies]` cosine and sine tables at `positions`, `axes` coordinates per row.
    pub(crate) fn tables(
        &self,
        positions: &[u32],
        axes: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let rows = positions.len() / axes;
        let (mut cos, mut sin) = (
            Vec::with_capacity(rows * self.inv_freq.len()),
            Vec::with_capacity(rows * self.inv_freq.len()),
        );
        for at in positions.chunks_exact(axes) {
            for (&f, &axis) in self.inv_freq.iter().zip(&self.axis) {
                let angle = at[axis as usize] as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let shape = (rows, self.inv_freq.len());
        Ok((
            Tensor::from_vec(cos, shape, device)?,
            Tensor::from_vec(sin, shape, device)?,
        ))
    }

    /// Rotate `[n, heads, head_dim]` at `positions`, `axes` coordinates per row.
    pub(crate) fn rotate(&self, x: &Tensor, positions: &[u32], axes: usize) -> Result<Tensor> {
        let (rows, _, dim) = x.dims3()?;
        let half = self.run / 2;
        if positions.len() != rows * axes
            || self.inv_freq.len() != half * self.runs
            || self.run * self.runs > dim
        {
            return Err(invalid("invalid axial rotary shape"));
        }
        let (cos, sin) = self.tables(positions, axes, x.device())?;
        let mut parts = Vec::with_capacity(self.runs + 1);
        for run in 0..self.runs {
            parts.push(rotary(
                &x.narrow(2, run * self.run, self.run)?,
                &cos.narrow(1, run * half, half)?,
                &sin.narrow(1, run * half, half)?,
            )?);
        }
        let rotated = self.run * self.runs;
        if rotated < dim {
            parts.push(x.narrow(2, rotated, dim - rotated)?);
        }
        Ok(Tensor::cat(&parts, 2)?)
    }
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
/// `sinks`, one logit per query head, join each softmax and are dropped from its output.
pub(crate) fn attention(
    q: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    past: usize,
    scale: f64,
    window: Option<usize>,
    sinks: Option<&Tensor>,
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
        let scores = scores.broadcast_add(&mask)?;
        let probs = match sinks {
            Some(sinks) => {
                let sinks = sinks
                    .reshape((heads, 1, 1))?
                    .broadcast_as((heads, rows, 1))?
                    .contiguous()?;
                let joined = Tensor::cat(&[&scores.contiguous()?, &sinks], 2)?;
                candle_nn::ops::softmax_last_dim(&joined)?
                    .narrow(2, 0, visible)?
                    .contiguous()?
            }
            None => candle_nn::ops::softmax_last_dim(&scores)?,
        };
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
