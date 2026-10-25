#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! F32 reference operations over one sequence. State tensors are values; no input is mutated.

pub(super) use crate::models::math::{attention, repeat_heads, rms, rotary, rotary_tables};
use candle_core::{D, DType, Device, Tensor};
use tur_core::{Result, execution::invalid};

/// Chunk length of the parallel delta rule, as in the reference implementation.
pub(super) const DELTA_CHUNK: usize = 64;

/// Gated `RMSNorm`: normalise, scale by the (not zero-centred) weight, then gate with SiLU(z).
pub(super) fn gated_rms(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    Ok((rms(x, weight, eps)? * candle_nn::ops::silu(z)?)?)
}

/// `x * rsqrt(sum(x²) + 1e-6)`, matching the reference's FLA-compatible L2 norm.
fn l2(x: &Tensor) -> Result<Tensor> {
    let inv = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?)
}

/// Numerically stable `log(1 + exp(x))`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    Ok((x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()?)?)
}

/// Convolution output and the histories retained for the engine.
pub(super) struct Convolved {
    /// SiLU(conv) for every new token: `[n, channels]`.
    pub output: Tensor,
    /// `[boundaries, history, channels]`: one history per requested boundary.
    pub histories: Tensor,
}

/// Depthwise causal convolution over `history` (`[K-1, C]`, oldest first) followed by `x`
/// (`[n, C]`). `weight` is `[C, K]`. Returns the last history or one history per token.
pub(super) fn convolution(
    x: &Tensor,
    history: &Tensor,
    weight: &Tensor,
    every_token: bool,
) -> Result<Convolved> {
    let (n, channels) = x.dims2()?;
    let (kept, history_channels) = history.dims2()?;
    let (weight_channels, kernel) = weight.dims2()?;
    if n == 0 || kernel != kept + 1 || channels != history_channels || channels != weight_channels {
        return Err(invalid("invalid causal convolution dimensions"));
    }
    let full = Tensor::cat(&[history, x], 0)?;
    let mut output = full
        .narrow(0, 0, n)?
        .broadcast_mul(&weight.narrow(1, 0, 1)?.t()?)?;
    for tap in 1..kernel {
        let column = weight.narrow(1, tap, 1)?.t()?;
        output = (output + full.narrow(0, tap, n)?.broadcast_mul(&column)?)?;
    }
    let histories = if every_token {
        let slices = (1..=n)
            .map(|end| full.narrow(0, end, kept))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Tensor::stack(&slices, 0)?
    } else {
        full.narrow(0, n, kept)?.unsqueeze(0)?
    };
    Ok(Convolved {
        output: candle_nn::ops::silu(&output)?,
        histories,
    })
}

/// Inputs of the gated delta rule for one sequence, already split into heads.
pub(super) struct DeltaInputs<'a> {
    /// `[n, heads, key_dim]`, repeated to value heads, before L2 normalisation.
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    /// `[n, heads, value_dim]`.
    pub v: &'a Tensor,
    /// Log-space decay `[n, heads]`, always `<= 0`.
    pub g: &'a Tensor,
    /// Write strength `[n, heads]` in `(0, 1)`.
    pub beta: &'a Tensor,
}

/// Delta-rule output and the states retained for the engine.
pub(super) struct DeltaOutput {
    /// `[n, heads, value_dim]`.
    pub output: Tensor,
    /// `[boundaries, heads, key_dim, value_dim]`.
    pub states: Tensor,
}

/// `log(decay) = -exp(A_log) * softplus(a + dt_bias)`; `neg_a` is `-exp(A_log)`.
pub(super) fn decay(a: &Tensor, dt_bias: &Tensor, neg_a: &Tensor) -> Result<Tensor> {
    Ok(softplus(&a.broadcast_add(dt_bias)?)?.broadcast_mul(neg_a)?)
}

fn normalise(inputs: &DeltaInputs<'_>) -> Result<(Tensor, Tensor)> {
    let key_dim = inputs.q.dim(D::Minus1)?;
    let q = (l2(inputs.q)? / (key_dim as f64).sqrt())?;
    Ok((q, l2(inputs.k)?))
}

/// Token-by-token gated delta rule: decay, read the prediction, write the beta-scaled
/// correction, then read with the query. Exact per-token states for speculation.
pub(super) fn delta_recurrent(
    inputs: &DeltaInputs<'_>,
    state: &Tensor,
    every_token: bool,
) -> Result<DeltaOutput> {
    let (q, k) = normalise(inputs)?;
    let n = q.dim(0)?;
    let mut state = state.clone();
    let mut outputs = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(if every_token { n } else { 1 });
    for t in 0..n {
        let q_t = q.get(t)?.unsqueeze(1)?;
        let k_t = k.get(t)?;
        let v_t = inputs.v.get(t)?;
        let decay_t = inputs.g.get(t)?.exp()?.unsqueeze(1)?.unsqueeze(2)?;
        state = state.broadcast_mul(&decay_t)?;
        let prediction = k_t.unsqueeze(1)?.matmul(&state)?.squeeze(1)?;
        let beta_t = inputs.beta.get(t)?.unsqueeze(1)?;
        let correction = (v_t - prediction)?.broadcast_mul(&beta_t)?;
        state = (state + k_t.unsqueeze(2)?.matmul(&correction.unsqueeze(1)?)?)?;
        outputs.push(q_t.matmul(&state)?.squeeze(1)?);
        if every_token {
            states.push(state.clone());
        }
    }
    if !every_token {
        states.push(state);
    }
    Ok(DeltaOutput {
        output: Tensor::stack(&outputs, 0)?,
        states: Tensor::stack(&states, 0)?,
    })
}

/// Invert the unit lower-triangular `I + strict_lower(a)` for every `[..., c, c]` block by
/// forward substitution, matching the reference's export-safe solver.
fn unit_lower_inverse(a: &Tensor) -> Result<Tensor> {
    let c = a.dim(D::Minus1)?;
    let lower = a.broadcast_mul(&strict_lower_mask(c, a.device())?)?.neg()?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(c);
    rows.push(lower.narrow(D::Minus2, 0, 1)?);
    for i in 1..c {
        let row = lower.narrow(D::Minus2, i, 1)?;
        let solved = Tensor::cat(&rows, D::Minus2)?;
        // Row i depends only on its first i entries and the already solved rows above it.
        let head = row.narrow(D::Minus1, 0, i)?.contiguous()?;
        rows.push((&row + head.matmul(&solved)?)?);
    }
    let eye = Tensor::eye(c, DType::F32, a.device())?;
    Ok(Tensor::cat(&rows, D::Minus2)?.broadcast_add(&eye)?)
}

/// `[c, c]` mask with ones strictly below the diagonal.
fn strict_lower_mask(c: usize, device: &Device) -> Result<Tensor> {
    let values: Vec<f32> = (0..c * c)
        .map(|i| if i % c < i / c { 1.0 } else { 0.0 })
        .collect();
    Ok(Tensor::from_vec(values, (c, c), device)?)
}

/// Chunk-parallel gated delta rule (UT transform), mathematically equal to the recurrence.
/// Returns only the final state; intermediate boundaries need [`delta_recurrent`].
pub(super) fn delta_chunked(inputs: &DeltaInputs<'_>, state: &Tensor) -> Result<DeltaOutput> {
    let (q, k) = normalise(inputs)?;
    let (n, heads, key_dim) = q.dims3()?;
    let value_dim = inputs.v.dim(2)?;
    let c = DELTA_CHUNK;
    let chunks = n.div_ceil(c);
    let padded = chunks * c;
    let pad = |x: &Tensor| -> Result<Tensor> {
        // [n, h, ...] -> [h, chunks, c, ...] with zero padding at the end.
        let x = x.pad_with_zeros(0, 0, padded - n)?;
        let mut shape = vec![chunks, c];
        shape.extend_from_slice(&x.dims()[1..]);
        let x = x.reshape(shape)?;
        let rank = x.rank();
        let mut order: Vec<usize> = (0..rank).collect();
        order.swap(0, 2);
        order.swap(1, 2);
        Ok(x.permute(order)?.contiguous()?)
    };
    let q = pad(&q)?;
    let k = pad(&k)?;
    let v = pad(inputs.v)?;
    let beta = pad(inputs.beta)?;
    let g = pad(inputs.g)?;
    let v_beta = v.broadcast_mul(&beta.unsqueeze(3)?)?;
    let k_beta = k.broadcast_mul(&beta.unsqueeze(3)?)?;
    let cum = g.cumsum(2)?;
    let mask = strict_lower_mask(c, q.device())?;
    let lower_or_diag = (mask + Tensor::eye(c, DType::F32, q.device())?)?;
    // exp(cum_i - cum_j) for j <= i; masked entries are zeroed after a safe exp.
    let pairwise = cum
        .unsqueeze(3)?
        .broadcast_sub(&cum.unsqueeze(2)?)?
        .broadcast_mul(&lower_or_diag)?
        .exp()?
        .broadcast_mul(&lower_or_diag)?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let ut = (k_beta.matmul(&k_t)? * &pairwise)?;
    let intra = (q.matmul(&k_t)? * &pairwise)?;
    let solve = unit_lower_inverse(&ut)?;
    let new_values = solve.matmul(&v_beta)?;
    let k_cumdecay = solve.matmul(&k_beta.broadcast_mul(&cum.exp()?.unsqueeze(3)?)?)?;
    let q = q.broadcast_mul(&cum.exp()?.unsqueeze(3)?)?;
    let last = cum.narrow(2, c - 1, 1)?;
    let k = k.broadcast_mul(&last.broadcast_sub(&cum)?.exp()?.unsqueeze(3)?)?;
    let chunk_decay = last.exp()?.unsqueeze(3)?;
    let at = |x: &Tensor, i: usize| -> Result<Tensor> {
        Ok(x.narrow(1, i, 1)?.squeeze(1)?.contiguous()?)
    };
    let mut state = state.contiguous()?;
    let mut outputs = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let v_new = (at(&new_values, i)? - at(&k_cumdecay, i)?.matmul(&state)?)?;
        let inter = at(&q, i)?.matmul(&state)?;
        outputs.push((inter + at(&intra, i)?.matmul(&v_new)?)?);
        let k_t = at(&k, i)?.transpose(1, 2)?.contiguous()?;
        state = (state.broadcast_mul(&at(&chunk_decay, i)?)? + k_t.matmul(&v_new)?)?;
    }
    // [h, chunks*c, v] -> [n, h, v]
    let output = Tensor::cat(&outputs, 1)?
        .narrow(1, 0, n)?
        .transpose(0, 1)?
        .contiguous()?;
    if output.dims() != [n, heads, value_dim] || state.dims() != [heads, key_dim, value_dim] {
        return Err(invalid("chunked delta rule produced an inconsistent shape"));
    }
    Ok(DeltaOutput {
        output,
        states: state.unsqueeze(0)?,
    })
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn random(shape: &[usize], seed: u64) -> Result<Tensor> {
        // Deterministic LCG values in (-1, 1); independent of RNG crates.
        let count: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let values: Vec<f32> = (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0, -1.0)
            })
            .collect();
        Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
    }

    fn delta_inputs(n: usize, heads: usize, dk: usize, dv: usize) -> Result<[Tensor; 5]> {
        let a = random(&[n, heads], 5)?;
        let neg_a = random(&[heads], 6)?.abs()?.neg()?;
        Ok([
            random(&[n, heads, dk], 1)?,
            random(&[n, heads, dk], 2)?,
            random(&[n, heads, dv], 3)?,
            decay(&a, &random(&[heads], 7)?, &neg_a)?,
            candle_nn::ops::sigmoid(&random(&[n, heads], 4)?)?,
        ])
    }

    #[test]
    fn test_chunked_delta_matches_recurrence_across_partial_chunks() -> Result<()> {
        for n in [1, 63, 64, 65, 150] {
            let [q, k, v, g, beta] = delta_inputs(n, 2, 8, 4)?;
            let inputs = DeltaInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            };
            let state = random(&[2, 8, 4], 9)?;
            let recurrent = delta_recurrent(&inputs, &state, false)?;
            let chunked = delta_chunked(&inputs, &state)?;
            assert!(
                max_diff(&recurrent.output, &chunked.output)? < 1e-4,
                "n={n}"
            );
            assert!(
                max_diff(&recurrent.states, &chunked.states)? < 1e-4,
                "n={n}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_delta_split_sequence_equals_whole_sequence() -> Result<()> {
        let [q, k, v, g, beta] = delta_inputs(10, 2, 4, 3)?;
        let all = DeltaInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        };
        let zero = Tensor::zeros((2, 4, 3), DType::F32, &Device::Cpu)?;
        let whole = delta_recurrent(&all, &zero, true)?;
        let part = |s: usize, l: usize| -> Result<[Tensor; 5]> {
            Ok([
                q.narrow(0, s, l)?,
                k.narrow(0, s, l)?,
                v.narrow(0, s, l)?,
                g.narrow(0, s, l)?,
                beta.narrow(0, s, l)?,
            ])
        };
        let [q1, k1, v1, g1, b1] = part(0, 4)?;
        let first = delta_recurrent(
            &DeltaInputs {
                q: &q1,
                k: &k1,
                v: &v1,
                g: &g1,
                beta: &b1,
            },
            &zero,
            false,
        )?;
        let [q2, k2, v2, g2, b2] = part(4, 6)?;
        let second = delta_recurrent(
            &DeltaInputs {
                q: &q2,
                k: &k2,
                v: &v2,
                g: &g2,
                beta: &b2,
            },
            &first.states.get(0)?,
            false,
        )?;
        assert!(max_diff(&whole.states.get(3)?, &first.states.get(0)?)? < 1e-6);
        assert!(max_diff(&whole.states.get(9)?, &second.states.get(0)?)? < 1e-5);
        assert!(max_diff(&whole.output.narrow(0, 4, 6)?, &second.output)? < 1e-5);
        Ok(())
    }

    #[test]
    fn test_convolution_matches_stepwise_window_and_histories() -> Result<()> {
        let x = random(&[5, 3], 1)?;
        let history = random(&[3, 3], 2)?;
        let weight = random(&[3, 4], 3)?;
        let whole = convolution(&x, &history, &weight, true)?;
        let mut state = history.clone();
        for t in 0..5 {
            let step = convolution(&x.narrow(0, t, 1)?, &state, &weight, false)?;
            assert!(max_diff(&step.output.get(0)?, &whole.output.get(t)?)? < 1e-6);
            state = step.histories.get(0)?;
            assert!(max_diff(&state, &whole.histories.get(t)?)? < 1e-6);
        }
        // Explicit window: out[t] = sum_j w[:, j] * full[t + j].
        let full = Tensor::cat(&[&history, &x], 0)?;
        let window = full.narrow(0, 2, 4)?;
        let expected = (window * weight.t()?)?.sum(0)?;
        let expected = candle_nn::ops::silu(&expected)?;
        assert!(max_diff(&expected, &whole.output.get(2)?)? < 1e-6);
        Ok(())
    }

    #[test]
    fn test_attention_prefill_matches_incremental_queries() -> Result<()> {
        let q = random(&[6, 4, 8], 1)?;
        let keys = random(&[6, 2, 8], 2)?;
        let values = random(&[6, 2, 8], 3)?;
        let whole = attention(&q, &keys, &values, 0, 8.0f64.sqrt().recip(), None)?;
        for t in 0..6 {
            let step = attention(
                &q.narrow(0, t, 1)?,
                &keys.narrow(0, 0, t + 1)?,
                &values.narrow(0, 0, t + 1)?,
                t,
                8.0f64.sqrt().recip(),
                None,
            )?;
            assert!(max_diff(&step.get(0)?, &whole.get(t)?)? < 1e-6);
        }
        // First query sees only the first key: output equals its value for each group.
        let first = whole.get(0)?;
        assert!(max_diff(&first.get(0)?, &values.get(0)?.get(0)?)? < 1e-6);
        assert!(max_diff(&first.get(3)?, &values.get(0)?.get(1)?)? < 1e-6);
        Ok(())
    }

    #[test]
    fn test_rotary_rotates_leading_halves_and_passes_tail() -> Result<()> {
        let x = random(&[2, 1, 6], 1)?;
        let (cos, sin) = rotary_tables(&[0, 3], 4, 10000.0, &Device::Cpu)?;
        let rotated = rotary(&x, &cos, &sin)?;
        assert!(max_diff(&rotated.get(0)?, &x.get(0)?)? < 1e-7);
        let row = x.get(1)?.get(0)?.to_vec1::<f32>()?;
        let out = rotated.get(1)?.get(0)?.to_vec1::<f32>()?;
        for i in 0..2 {
            let angle = 3.0f32 / 10000f32.powf((2 * i) as f32 / 4.0);
            let (l, r) = (row[i], row[i + 2]);
            assert!((out[i] - f32::mul_add(r, -angle.sin(), l * angle.cos())).abs() < 1e-6);
            assert!((out[i + 2] - f32::mul_add(l, angle.sin(), r * angle.cos())).abs() < 1e-6);
        }
        assert_eq!(&out[4..], &row[4..]);
        Ok(())
    }

    #[test]
    fn test_norms_follow_reference_formulas() -> Result<()> {
        let x = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu)?;
        let scale = Tensor::new(&[1.0f32, 2.0], &Device::Cpu)?;
        let out = rms(&x, &scale, 0.0)?.to_vec2::<f32>()?;
        let r = (12.5f32).sqrt();
        assert!((out[0][0] - 3.0 / r).abs() < 1e-6 && (out[0][1] - 8.0 / r).abs() < 1e-6);
        let z = Tensor::new(&[[0.0f32, 1.0]], &Device::Cpu)?;
        let gated = gated_rms(&x, &z, &scale, 0.0)?.to_vec2::<f32>()?;
        assert!(gated[0][0].abs() < 1e-7);
        let silu = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((8.0 / r).mul_add(-silu, gated[0][1]).abs() < 1e-6);
        Ok(())
    }
}
