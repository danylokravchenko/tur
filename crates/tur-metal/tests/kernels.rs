#![cfg(feature = "metal")]
#![forbid(unsafe_code)]
//! Device-versus-reference differential tests for the fused Metal kernels. References are
//! independent scalar F64 loops over the same F32 inputs. These tests need host GPU access
//! and fail, rather than skip, when no Metal device is available.

use candle_core::{
    DType, Device, Tensor,
    quantized::{GgmlDType, QTensor},
};
use tur_core::Result;
use tur_metal::{
    completion::{Completion, Quarantine},
    kernels::{DeltaShape, Heads, Kernels, Tuning},
    packed::PackedLinear,
};

/// Device-wide allocation counters are shared by every test in the process, so tests in
/// this file run one at a time.
static GPU: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn setup() -> Result<(Device, Kernels, parking_lot::MutexGuard<'static, ()>)> {
    let guard = GPU.lock();
    let device = Device::new_metal(0)?;
    let kernels = Kernels::new(&device)?;
    Ok((device, kernels, guard))
}

/// Deterministic values in (-scale, scale).
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn upload(device: &Device, data: &[f32], shape: &[usize]) -> Result<Tensor> {
    Ok(Tensor::from_slice(data, shape, device)?)
}

fn read(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

/// Max absolute error relative to the reference's largest magnitude.
fn relative_error(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "length");
    assert!(actual.iter().all(|v| v.is_finite()), "non-finite output");
    let scale = expected
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-30);
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (f64::from(*a) - e).abs())
        .fold(0.0, f64::max)
        / scale
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn rms_row(x: &[f32], w: &[f32], eps: f64) -> Vec<f64> {
    let mean = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, w)| f64::from(*v) * inv * f64::from(*w))
        .collect()
}

#[test]
fn test_capabilities_and_pipelines_compile() -> Result<()> {
    let (_, kernels, _gpu) = setup()?;
    let caps = kernels.capabilities();
    assert!(caps.apple_family >= 7, "{caps:?}");
    assert!(caps.max_threads_per_threadgroup >= 32);
    assert!(caps.recommended_working_set_bytes > 0);
    let tuning = kernels.tuning();
    assert!(tuning.row_threads <= caps.max_threads_per_threadgroup);
    assert!(
        kernels
            .clone()
            .with_tuning(Tuning {
                simdgroups: 0,
                ..tuning
            })
            .is_err()
    );
    Ok(())
}

#[test]
fn test_rms_norm_matches_reference_on_strided_rows() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let (rows, stride, width) = (5, 1030, 1000);
    let base = values(rows * stride, 1, 3.0);
    let w = values(width, 2, 1.0);
    let x = upload(&device, &base, &[rows, stride])?.narrow(1, 3, width)?;
    let out = kernels.rms_norm(&x, &upload(&device, &w, &[width])?, 1e-6)?;
    let expected: Vec<f64> = (0..rows)
        .flat_map(|r| rms_row(&base[r * stride + 3..r * stride + 3 + width], &w, 1e-6))
        .collect();
    assert!(relative_error(&read(&out)?, &expected) < 1e-6);
    Ok(())
}

#[test]
fn test_add_rms_norm_returns_sum_and_normalized_sum() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let (rows, width) = (3, 37);
    let x = values(rows * width, 3, 2.0);
    let r = values(rows * width, 4, 2.0);
    let w = values(width, 5, 1.0);
    let (sum, normed) = kernels.add_rms_norm(
        &upload(&device, &x, &[rows, width])?,
        &upload(&device, &r, &[rows, width])?,
        &upload(&device, &w, &[width])?,
        1e-6,
    )?;
    let s: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();
    assert_eq!(read(&sum)?, s);
    let expected: Vec<f64> = s
        .chunks(width)
        .flat_map(|row| rms_row(row, &w, 1e-6))
        .collect();
    assert!(relative_error(&read(&normed)?, &expected) < 1e-6);
    Ok(())
}

#[test]
fn test_qk_norm_rope_reads_interleaved_heads_and_rotates_partially() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    for dim in [64usize, 256] {
        let (tokens, heads, rotary) = (4, 3, dim / 4);
        let width = heads * 2 * dim;
        let x = values(tokens * width, 6, 2.0);
        let w = values(dim, 7, 1.0);
        let positions = [0u32, 7, 513, 32767];
        let inv_freq = kernels.rotary_frequencies(rotary, 1e7)?;
        let freqs = read(&inv_freq)?;
        let out = kernels.qk_norm_rope(
            &upload(&device, &x, &[tokens, width])?,
            Heads {
                count: heads,
                dim,
                head_stride: 2 * dim,
                col_offset: 0,
            },
            &upload(&device, &w, &[dim])?,
            &Tensor::from_slice(&positions, tokens, &device)?,
            &inv_freq,
            1e-6,
        )?;
        let half = rotary / 2;
        let mut expected = Vec::new();
        for (t, &p) in positions.iter().enumerate() {
            for h in 0..heads {
                let start = t * width + h * 2 * dim;
                let y = rms_row(&x[start..start + dim], &w, 1e-6);
                for i in 0..dim {
                    expected.push(if i < rotary {
                        let f = i % half;
                        // The angle is formed in F32, as on the device and in the reference.
                        let angle = f64::from(p as f32 * freqs[f]);
                        let (sin, cos) = angle.sin_cos();
                        if i < half {
                            y[i] * cos - y[i + half] * sin
                        } else {
                            y[i] * cos + y[i - half] * sin
                        }
                    } else {
                        y[i]
                    });
                }
            }
        }
        let error = relative_error(&read(&out)?, &expected);
        assert!(error < 2e-6, "dim {dim}: {error}");
        assert_eq!(out.dims(), [tokens, heads, dim]);
    }
    Ok(())
}

#[test]
fn test_swiglu_and_gated_rms_match_reference() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let (rows, width) = (3, 45);
    let gu = values(rows * 2 * width, 8, 4.0);
    let out = kernels.swiglu(&upload(&device, &gu, &[rows, 2 * width])?)?;
    let expected: Vec<f64> = gu
        .chunks(2 * width)
        .flat_map(|row| {
            (0..width).map(move |i| silu(f64::from(row[i])) * f64::from(row[width + i]))
        })
        .collect();
    assert!(relative_error(&read(&out)?, &expected) < 1e-6);

    let (tokens, heads, dim, z_col, proj_cols) = (3, 4, 22, 9, 120);
    let core = values(tokens * heads * dim, 9, 2.0);
    let proj = values(tokens * proj_cols, 10, 3.0);
    let w = values(dim, 11, 1.0);
    let out = kernels.gated_rms(
        &upload(&device, &core, &[tokens * heads, dim])?,
        &upload(&device, &proj, &[tokens, proj_cols])?,
        z_col,
        heads,
        &upload(&device, &w, &[dim])?,
        1e-6,
    )?;
    let mut expected = Vec::new();
    for t in 0..tokens {
        for h in 0..heads {
            let row = (t * heads + h) * dim;
            let y = rms_row(&core[row..row + dim], &w, 1e-6);
            for (i, y) in y.iter().enumerate() {
                expected.push(y * silu(f64::from(proj[t * proj_cols + z_col + h * dim + i])));
            }
        }
    }
    assert!(relative_error(&read(&out)?, &expected) < 1e-6);
    assert_eq!(out.dims(), [tokens, heads * dim]);
    Ok(())
}

#[test]
fn test_delta_conv_writes_offset_rows_and_histories() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let (rows, cols, channels, taps) = (7, 42, 37, 4);
    let (row, tokens, out_row) = (2, 4, 3);
    let x = values(rows * cols, 12, 2.0);
    let history = values((taps - 1) * channels, 13, 2.0);
    let weight = values(channels * taps, 14, 1.0);
    let full = |r: usize, c: usize| -> f32 {
        if r < taps - 1 {
            history[r * channels + c]
        } else {
            x[(row + r - (taps - 1)) * cols + c]
        }
    };
    for every in [false, true] {
        let out = kernels.output(10, channels)?;
        let histories = kernels.delta_conv(
            &upload(&device, &x, &[rows, cols])?,
            row,
            tokens,
            &upload(&device, &history, &[taps - 1, channels])?,
            &upload(&device, &weight, &[channels, taps])?,
            every,
            &out,
            out_row,
        )?;
        let out = kernels.finish(out, (10, channels))?;
        let actual = read(&out.narrow(0, out_row, tokens)?)?;
        let mut expected = Vec::new();
        for t in 0..tokens {
            for c in 0..channels {
                let acc: f64 = (0..taps)
                    .map(|j| f64::from(full(t + j, c)) * f64::from(weight[c * taps + j]))
                    .sum();
                expected.push(silu(acc));
            }
        }
        assert!(relative_error(&actual, &expected) < 1e-6);
        let kept: Vec<usize> = if every {
            (0..tokens).collect()
        } else {
            vec![tokens - 1]
        };
        let expected: Vec<f64> = kept
            .iter()
            .flat_map(|&t| {
                (0..taps - 1).flat_map(move |r| (0..channels).map(move |c| (t + 1 + r, c)))
            })
            .map(|(r, c)| f64::from(full(r, c)))
            .collect();
        assert_eq!(histories.dims(), [kept.len(), taps - 1, channels]);
        assert!(relative_error(&read(&histories)?, &expected) == 0.0);
    }
    Ok(())
}

struct ScanCase {
    shape: DeltaShape,
    mixed_cols: usize,
    proj_cols: usize,
}

/// Scalar gated delta rule in F64; returns outputs `[tokens, heads, dv]` and states.
fn scan_reference(
    case: &ScanCase,
    mixed: &[f32],
    proj: &[f32],
    rows: std::ops::Range<usize>,
    dt_bias: &[f32],
    neg_a: &[f32],
    initial: &[f32],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let s = case.shape;
    let (dk, dv, heads) = (s.key_dim, s.value_dim, s.value_heads);
    let groups = heads / s.key_heads;
    let kw = s.key_width();
    let mut state: Vec<f64> = initial.iter().map(|v| f64::from(*v)).collect();
    let mut outputs = Vec::new();
    let mut states = Vec::new();
    for t in rows {
        let row = &mixed[t * case.mixed_cols..];
        let pr = &proj[t * case.proj_cols..];
        for h in 0..heads {
            let kh = h / groups;
            let norm = |v: &[f32]| -> Vec<f64> {
                let inv =
                    1.0 / (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() + 1e-6).sqrt();
                v.iter().map(|x| f64::from(*x) * inv).collect()
            };
            let q: Vec<f64> = norm(&row[kh * dk..(kh + 1) * dk])
                .into_iter()
                .map(|v| v / (dk as f64).sqrt())
                .collect();
            let k = norm(&row[kw + kh * dk..kw + (kh + 1) * dk]);
            let v = &row[2 * kw + h * dv..2 * kw + (h + 1) * dv];
            let decay = (softplus(f64::from(pr[s.a_col + h]) + f64::from(dt_bias[h]))
                * f64::from(neg_a[h]))
            .exp();
            let beta = sigmoid(f64::from(pr[s.b_col + h]));
            let st = &mut state[h * dk * dv..(h + 1) * dk * dv];
            st.iter_mut().for_each(|x| *x *= decay);
            for j in 0..dv {
                let prediction: f64 = (0..dk).map(|i| k[i] * st[i * dv + j]).sum();
                let correction = (f64::from(v[j]) - prediction) * beta;
                (0..dk).for_each(|i| st[i * dv + j] += k[i] * correction);
            }
            outputs.extend((0..dv).map(|j| (0..dk).map(|i| q[i] * st[i * dv + j]).sum::<f64>()));
        }
        states.push(state.clone());
    }
    (outputs, states)
}

#[test]
fn test_delta_scan_matches_scalar_recurrence() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    for (dk, tokens) in [(64usize, 1usize), (128, 5), (128, 70), (256, 3)] {
        let shape = DeltaShape {
            key_heads: 2,
            value_heads: 4,
            key_dim: dk,
            value_dim: 22,
            b_col: 3,
            a_col: 8,
        };
        let case = ScanCase {
            shape,
            mixed_cols: shape.channels() + 3,
            proj_cols: 12,
        };
        let (row, rows) = (1, tokens + 2);
        let mixed = values(rows * case.mixed_cols, 15, 1.0);
        let proj = values(rows * case.proj_cols, 16, 3.0);
        let dt_bias = values(4, 17, 1.0);
        let neg_a: Vec<f32> = values(4, 18, 1.0).iter().map(|v| -v.abs() - 0.1).collect();
        let initial = values(4 * dk * 22, 19, 0.1);
        let (expected_out, expected_states) = scan_reference(
            &case,
            &mixed,
            &proj,
            row..row + tokens,
            &dt_bias,
            &neg_a,
            &initial,
        );
        for every in [false, true] {
            let out = kernels.output(tokens + 1, 4 * 22)?;
            let states = kernels.delta_scan(
                &upload(&device, &mixed, &[rows, case.mixed_cols])?,
                &upload(&device, &proj, &[rows, case.proj_cols])?,
                row,
                tokens,
                &shape,
                &upload(&device, &dt_bias, &[4])?,
                &upload(&device, &neg_a, &[4])?,
                &upload(&device, &initial, &[4, dk, 22])?,
                every,
                &out,
                1,
            )?;
            let out = kernels.finish(out, (tokens + 1, 4 * 22))?;
            let error = relative_error(&read(&out.narrow(0, 1, tokens)?)?, &expected_out);
            assert!(error < 1e-5, "dk {dk} tokens {tokens}: output {error}");
            let expected: Vec<f64> = if every {
                expected_states.concat()
            } else {
                expected_states.last().cloned().unwrap_or_default()
            };
            let error = relative_error(&read(&states)?, &expected);
            assert!(error < 1e-5, "dk {dk} tokens {tokens}: state {error}");
        }
    }
    Ok(())
}

struct AttentionCase {
    heads: usize,
    kv_heads: usize,
    dim: usize,
    page: usize,
    /// (past tokens, new tokens) per sequence.
    sequences: Vec<(usize, usize)>,
}

/// Pool keys, pool values, slots, page table, first blocks and visible key counts.
type Pool = (Vec<f32>, Vec<f32>, usize, Vec<u32>, Vec<u32>, Vec<u32>);

/// Build a shuffled page pool whose unused slots hold NaN, so any read outside the
/// visible keys poisons the result.
fn paged_pool(case: &AttentionCase, seed: u64) -> Pool {
    let pages: Vec<usize> = case
        .sequences
        .iter()
        .map(|(past, new)| (past + new).div_ceil(case.page))
        .collect();
    let total: usize = pages.iter().sum::<usize>() + 2;
    // Reverse-interleaved physical order exercises the indirection.
    let physical: Vec<u32> = (0..total as u32).rev().collect();
    let row = case.kv_heads * case.dim;
    let slots = total * case.page;
    let mut k = vec![f32::NAN; slots * row];
    let mut v = vec![f32::NAN; slots * row];
    let (mut table, mut first, mut visible) = (Vec::new(), Vec::new(), Vec::new());
    let mut next = 0;
    for (s, &(past, new)) in case.sequences.iter().enumerate() {
        let base = table.len() as u32;
        table.extend_from_slice(&physical[next..next + pages[s]]);
        next += pages[s];
        let kv = values(2 * (past + new) * row, seed + s as u64, 1.0);
        for j in 0..past + new {
            let slot = table[base as usize + j / case.page] as usize * case.page + j % case.page;
            k[slot * row..(slot + 1) * row].copy_from_slice(&kv[j * row..(j + 1) * row]);
            let vj = (past + new + j) * row;
            v[slot * row..(slot + 1) * row].copy_from_slice(&kv[vj..vj + row]);
        }
        for t in 0..new {
            first.push(base);
            visible.push((past + t + 1) as u32);
        }
    }
    (k, v, slots, table, first, visible)
}

#[allow(clippy::too_many_arguments)]
fn attention_reference(
    case: &AttentionCase,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    table: &[u32],
    first: &[u32],
    visible: &[u32],
) -> Vec<f64> {
    let (heads, dim) = (case.heads, case.dim);
    let groups = heads / case.kv_heads;
    let row = case.kv_heads * dim;
    let mut out = Vec::new();
    for t in 0..visible.len() {
        for h in 0..heads {
            let g = h / groups;
            let qh = &q[(t * heads + h) * dim..][..dim];
            let slots: Vec<usize> = (0..visible[t] as usize)
                .map(|j| {
                    table[first[t] as usize + j / case.page] as usize * case.page + j % case.page
                })
                .collect();
            let scores: Vec<f64> = slots
                .iter()
                .map(|s| {
                    let kr = &k[s * row + g * dim..][..dim];
                    qh.iter()
                        .zip(kr)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for i in 0..dim {
                let value: f64 = slots
                    .iter()
                    .zip(&weights)
                    .map(|(s, w)| w * f64::from(v[s * row + g * dim + i]))
                    .sum();
                let gi = f64::from(gate[t * heads * 2 * dim + h * 2 * dim + dim + i]);
                out.push(value / total * sigmoid(gi));
            }
        }
    }
    out
}

fn check_attention(kernels: &Kernels, device: &Device, case: &AttentionCase) -> Result<()> {
    let (k, v, slots, table, first, visible) = paged_pool(case, 30);
    let tokens = visible.len();
    let q = values(tokens * case.heads * case.dim, 20, 1.0);
    let gate = values(tokens * case.heads * 2 * case.dim, 21, 2.0);
    let expected = attention_reference(case, &q, &k, &v, &gate, &table, &first, &visible);
    let paged = kernels.paged_kv(case.page, slots, &table, &first, &visible)?;
    let out = kernels.attention(
        &upload(device, &q, &[tokens, case.heads, case.dim])?,
        &upload(device, &k, &[slots, case.kv_heads, case.dim])?,
        &upload(device, &v, &[slots, case.kv_heads, case.dim])?,
        &paged,
        &upload(device, &gate, &[tokens, case.heads * 2 * case.dim])?,
        Heads {
            count: case.heads,
            dim: case.dim,
            head_stride: 2 * case.dim,
            col_offset: case.dim,
        },
    )?;
    let error = relative_error(&read(&out)?, &expected);
    assert!(
        error < 1e-5,
        "dim {} split {:?}: {error}",
        case.dim,
        kernels.attention_split(tokens, case.heads, paged.max_visible())
    );
    Ok(())
}

#[test]
fn test_attention_paged_gqa_matches_reference_with_and_without_splits() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let split = kernels.clone().with_tuning(Tuning {
        min_split_keys: 8,
        ..kernels.tuning()
    })?;
    for dim in [64usize, 128, 256] {
        let case = AttentionCase {
            heads: 8,
            kv_heads: 2,
            dim,
            page: 16,
            sequences: vec![(37, 3), (0, 20), (15, 1)],
        };
        assert_eq!(kernels.attention_split(24, 8, 40).splits, 1);
        assert!(split.attention_split(24, 8, 40).splits > 1);
        check_attention(&kernels, &device, &case)?;
        check_attention(&split, &device, &case)?;
    }
    // Long-context decode: default tuning splits 1000 keys across threadgroups.
    let case = AttentionCase {
        heads: 4,
        kv_heads: 1,
        dim: 128,
        page: 32,
        sequences: vec![(999, 1)],
    };
    assert!(kernels.attention_split(1, 4, 1000).splits > 1);
    check_attention(&kernels, &device, &case)
}

#[test]
fn test_gemv_matches_dequantized_reference_for_every_batch_size() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    // Exercise the GEMV at every row count it supports.
    let kernels = kernels.clone().with_tuning(Tuning {
        gemv_q8_0_rows: (1, 8),
        gemv_q4k_rows: (1, 8),
        ..kernels.tuning()
    })?;
    let (n, k) = (37, 512);
    let weight = upload(&Device::Cpu, &values(n * k, 22, 1.0), &[n, k])?;
    for format in [GgmlDType::Q8_0, GgmlDType::Q4K] {
        let layer = PackedLinear::quantized(&weight, format, &device)?;
        assert_eq!(
            layer.resident_bytes(),
            n * k / format.block_size() * format.type_size()
        );
        let dequantized = read(&QTensor::quantize(&weight, format)?.dequantize(&Device::Cpu)?)?;
        for rows in 1..=9 {
            let x = values(rows * k, 23 + rows as u64, 1.0);
            let expected: Vec<f64> = (0..rows)
                .flat_map(|r| {
                    let x = &x[r * k..(r + 1) * k];
                    let dequantized = &dequantized;
                    (0..n).map(move |o| {
                        x.iter()
                            .zip(&dequantized[o * k..(o + 1) * k])
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>()
                    })
                })
                .collect();
            let out = kernels.linear(&layer, &upload(&device, &x, &[rows, k])?)?;
            let error = relative_error(&read(&out)?, &expected);
            // Above the GEMV range Candle's tiled GEMM stages weight tiles as F16.
            let bound = if rows <= 8 { 1e-5 } else { 2e-3 };
            assert!(error < bound, "{format:?} rows {rows}: {error}");
        }
    }
    Ok(())
}

#[test]
fn test_invalid_inputs_are_rejected_before_encoding() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let w = upload(&device, &[1.0; 4], &[4])?;
    let cpu = Tensor::ones((2, 4), DType::F32, &Device::Cpu)?;
    assert!(kernels.rms_norm(&cpu, &w, 1e-6).is_err());
    let half = Tensor::ones((2, 4), DType::F16, &device)?;
    assert!(kernels.rms_norm(&half, &w, 1e-6).is_err());
    let wide = Tensor::ones((2, 5), DType::F32, &device)?;
    assert!(kernels.rms_norm(&wide, &w, 1e-6).is_err());
    let transposed = Tensor::ones((4, 2), DType::F32, &device)?.t()?;
    assert!(kernels.rms_norm(&transposed, &w, 1e-6).is_err());
    // Page entry 2 needs slots 32..48 but the pool has 40.
    assert!(kernels.paged_kv(16, 40, &[0, 2], &[0], &[20]).is_err());
    // Visible keys need two pages; the table provides one from this offset.
    assert!(kernels.paged_kv(16, 64, &[0, 1], &[1], &[20]).is_err());
    assert!(kernels.paged_kv(16, 64, &[0], &[0], &[0]).is_err());
    let shape = DeltaShape {
        key_heads: 1,
        value_heads: 1,
        key_dim: 96,
        value_dim: 4,
        b_col: 0,
        a_col: 1,
    };
    let mixed = Tensor::ones((1, 200), DType::F32, &device)?;
    let ones = Tensor::ones(1, DType::F32, &device)?;
    let state = Tensor::zeros((1, 96, 4), DType::F32, &device)?;
    let out = kernels.output(1, 4)?;
    assert!(
        kernels
            .delta_scan(
                &mixed, &mixed, 0, 1, &shape, &ones, &ones, &state, false, &out, 0
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn test_repeated_passes_reach_a_stable_allocation_bound() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let Device::Metal(metal) = &device else {
        return Err(tur_core::TurError::Other("expected Metal".into()));
    };
    let case = AttentionCase {
        heads: 8,
        kv_heads: 2,
        dim: 128,
        page: 16,
        sequences: vec![(100, 1), (3, 1)],
    };
    let (k, v, slots, table, first, visible) = paged_pool(&case, 40);
    let k = upload(&device, &k, &[slots, 2, 128])?;
    let v = upload(&device, &v, &[slots, 2, 128])?;
    let q = upload(&device, &values(2 * 8 * 128, 41, 1.0), &[2, 8, 128])?;
    let gate = upload(&device, &values(2 * 8 * 256, 42, 1.0), &[2, 8 * 256])?;
    let w = upload(&device, &values(1024, 43, 1.0), &[1024])?;
    let x = upload(&device, &values(2 * 1024, 44, 1.0), &[2, 1024])?;
    let step = || -> Result<()> {
        let paged = kernels.paged_kv(16, slots, &table, &first, &visible)?;
        let heads = Heads {
            count: 8,
            dim: 128,
            head_stride: 256,
            col_offset: 128,
        };
        let attended = kernels.attention(&q, &k, &v, &paged, &gate, heads)?;
        let (sum, _) = kernels.add_rms_norm(&x, &attended, &w, 1e-6)?;
        let _normed = kernels.rms_norm(&sum, &w, 1e-6)?;
        Ok(device.synchronize()?)
    };
    for _ in 0..3 {
        step()?;
    }
    let settled = metal.current_allocated_size();
    for _ in 0..20 {
        step()?;
    }
    assert_eq!(metal.current_allocated_size(), settled);
    Ok(())
}

#[test]
fn test_completion_retains_kernel_outputs_until_the_fence() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let x = upload(&device, &values(64, 45, 1.0), &[1, 64])?;
    let w = upload(&device, &[1.0; 64], &[64])?;
    let out = kernels.rms_norm(&x, &w, 1e-6)?;
    let mut completion = Completion::new(device.clone(), &Quarantine::new(), out);
    assert!(completion.take().is_none());
    completion.wait()?;
    let out = completion
        .take()
        .ok_or_else(|| tur_core::TurError::Other("not retained".into()))?;
    let expected = rms_row(&read(&x)?, &[1.0; 64], 1e-6);
    assert!(relative_error(&read(&out)?, &expected) < 1e-6);
    Ok(())
}
