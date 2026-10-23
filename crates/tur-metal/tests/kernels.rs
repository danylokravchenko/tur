#![allow(
    clippy::float_cmp,
    reason = "quantization tests count exact stored-code differences"
)]
#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![cfg(feature = "metal")]
#![forbid(unsafe_code)]
//! Device-versus-reference differential tests for the fused Metal kernels. References are
//! independent scalar F64 loops over the same F32 inputs. These tests need host GPU access
//! and fail, rather than skip, when no Metal device is available.

use candle_core::{
    DType, Device, Tensor,
    quantized::{GgmlDType, QTensor},
};
use tur_core::{Result, execution::KvFormat};
use tur_metal::{
    completion::{Completion, Quarantine},
    kernels::{DeltaShape, Heads, Kernels, RecurrentSpan, Tuning},
    packed::PackedLinear,
    pool::{AttentionPool, KvPool, dequantize, quantize},
    sampling::{self, PenaltyEntry, PenaltyParams, SampleParams, SampleStatus},
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
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0, -1.0) * scale
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
                            y[i + half].mul_add(-sin, y[i] * cos)
                        } else {
                            y[i - half].mul_add(sin, y[i] * cos)
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

/// Sequences of one packed recurrent test pass: (tokens, committed slot, written slots).
type Ragged = [(usize, u32, Vec<u32>)];

fn spans(ragged: &Ragged) -> Vec<RecurrentSpan<'_>> {
    ragged
        .iter()
        .map(|(tokens, read, writes)| RecurrentSpan {
            tokens: *tokens,
            position: 7,
            read: *read,
            writes,
        })
        .collect()
}

/// A `[slots, rest..]` pool poisoned with NaN except each committed slot, which holds the
/// state generated from its slot id.
fn committed_pool(
    device: &Device,
    slots: usize,
    rest: &[usize],
    ragged: &Ragged,
    scale: f32,
) -> Result<(Tensor, Vec<Vec<f32>>)> {
    let size: usize = rest.iter().product();
    let mut data = vec![f32::NAN; slots * size];
    let initial: Vec<Vec<f32>> = (0..slots)
        .map(|slot| values(size, 40 + slot as u64, scale))
        .collect();
    for (_, read, _) in ragged {
        let read = *read as usize;
        data[read * size..(read + 1) * size].copy_from_slice(&initial[read]);
    }
    let mut shape = vec![slots];
    shape.extend_from_slice(rest);
    Ok((upload(device, &data, &shape)?, initial))
}

#[test]
fn test_delta_conv_runs_ragged_sequences_in_one_dispatch() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let (cols, channels, taps) = (42, 37, 4);
    // Mixed decode, prefill and speculative sequences; two forks share committed slot 2.
    let ragged: Vec<(usize, u32, Vec<u32>)> = vec![
        (1, 2, vec![5]),
        (4, 0, vec![9, 6, 8, 1]),
        (3, 2, vec![7]),
        (1, 3, vec![4]),
    ];
    let rows: usize = ragged.iter().map(|(tokens, _, _)| tokens).sum();
    let x = values(rows * cols, 12, 2.0);
    let weight = values(channels * taps, 14, 1.0);
    let (pool, initial) = committed_pool(&device, 10, &[taps - 1, channels], &ragged, 2.0)?;
    let spans = kernels.recurrent_spans(&spans(&ragged))?;
    assert_eq!((spans.sequences(), spans.tokens()), (4, rows));
    let out = kernels.delta_conv(
        &upload(&device, &x, &[rows, cols])?,
        &spans,
        &pool,
        &upload(&device, &weight, &[channels, taps])?,
    )?;
    let actual = read(&out)?;
    let mut expected = Vec::new();
    let mut row = 0;
    for (tokens, slot, writes) in &ragged {
        let history = &initial[*slot as usize];
        let full = |r: usize, c: usize| -> f32 {
            if r < taps - 1 {
                history[r * channels + c]
            } else {
                x[(row + r - (taps - 1)) * cols + c]
            }
        };
        for t in 0..*tokens {
            for c in 0..channels {
                let acc: f64 = (0..taps)
                    .map(|j| f64::from(full(t + j, c)) * f64::from(weight[c * taps + j]))
                    .sum();
                expected.push(silu(acc));
            }
        }
        let kept: Vec<usize> = if writes.len() > 1 {
            (0..*tokens).collect()
        } else {
            vec![tokens - 1]
        };
        for (&slot, &t) in writes.iter().zip(&kept) {
            let expected: Vec<f64> = (0..taps - 1)
                .flat_map(|r| (0..channels).map(move |c| (t + 1 + r, c)))
                .map(|(r, c)| f64::from(full(r, c)))
                .collect();
            assert_eq!(
                relative_error(&read(&pool.get(slot as usize)?)?, &expected),
                0.0
            );
        }
        row += tokens;
    }
    assert_eq!(out.dims(), [rows, channels]);
    assert!(relative_error(&actual, &expected) < 1e-6);
    // Committed slots are untouched.
    for slot in [0, 2, 3] {
        assert_eq!(read(&pool.get(slot)?)?, initial[slot]);
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
            for x in st.iter_mut() {
                *x *= decay;
            }
            for j in 0..dv {
                let prediction: f64 = (0..dk).map(|i| k[i] * st[i * dv + j]).sum();
                let correction = (f64::from(v[j]) - prediction) * beta;
                (0..dk)
                    .for_each(|i| st[i * dv + j] = f64::mul_add(k[i], correction, st[i * dv + j]));
            }
            outputs.extend((0..dv).map(|j| (0..dk).map(|i| q[i] * st[i * dv + j]).sum::<f64>()));
        }
        states.push(state.clone());
    }
    (outputs, states)
}

#[test]
fn test_delta_scan_matches_scalar_recurrence_over_ragged_sequences() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    // Single-token sequences take the step kernel, or the scan when tuning disables it.
    // Narrow storage rounds each stored state once; a pass accumulates in F32, so outputs
    // match a reference started from the rounded committed state.
    for (dk, delta_step, dtype) in [64usize, 128, 256].into_iter().flat_map(|dk| {
        [DType::F32, DType::F16, DType::BF16]
            .into_iter()
            .flat_map(move |dtype| [(dk, true, dtype), (dk, false, dtype)])
    }) {
        let stored = match dtype {
            DType::F32 => 1e-5,
            DType::F16 => 1e-3,
            _ => 8e-3,
        };
        let kernels = kernels.clone().with_tuning(Tuning {
            delta_step,
            ..kernels.tuning()
        })?;
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
        // A long prefill, decodes forked from one slot, a speculative sequence and a
        // single-token speculative boundary.
        let ragged: Vec<(usize, u32, Vec<u32>)> = vec![
            (70, 1, vec![0]),
            (1, 3, vec![2]),
            (5, 4, (5..10).rev().collect()),
            (1, 3, vec![10]),
            (1, 11, vec![12]),
        ];
        let rows: usize = ragged.iter().map(|(tokens, _, _)| tokens).sum();
        let mixed = values(rows * case.mixed_cols, 15, 1.0);
        let proj = values(rows * case.proj_cols, 16, 3.0);
        let dt_bias = values(4, 17, 1.0);
        let neg_a: Vec<f32> = values(4, 18, 1.0).iter().map(|v| -v.abs() - 0.1).collect();
        let (pool, _) = committed_pool(&device, 14, &[4, dk, 22], &ragged, 0.1)?;
        let pool = pool.to_dtype(dtype)?;
        let widen =
            |slot: usize| -> Result<Vec<f32>> { read(&pool.get(slot)?.to_dtype(DType::F32)?) };
        let initial: Vec<Vec<f32>> = (0..14).map(widen).collect::<Result<_>>()?;
        let out = kernels.delta_scan(
            &upload(&device, &mixed, &[rows, case.mixed_cols])?,
            &upload(&device, &proj, &[rows, case.proj_cols])?,
            &kernels.recurrent_spans(&spans(&ragged))?,
            &shape,
            &upload(&device, &dt_bias, &[4])?,
            &upload(&device, &neg_a, &[4])?,
            &pool,
        )?;
        assert_eq!(out.dims(), [rows * 4, 22]);
        let mut row = 0;
        for (tokens, slot, writes) in &ragged {
            let (expected_out, expected_states) = scan_reference(
                &case,
                &mixed,
                &proj,
                row..row + tokens,
                &dt_bias,
                &neg_a,
                &initial[*slot as usize],
            );
            let actual = read(&out.narrow(0, row * 4, tokens * 4)?)?;
            let error = relative_error(&actual, &expected_out);
            assert!(
                error < 1e-5,
                "{dtype:?} dk {dk} step {delta_step} tokens {tokens}: output {error}"
            );
            let boundaries: Vec<&Vec<f64>> = if writes.len() > 1 {
                expected_states.iter().collect()
            } else {
                expected_states.last().into_iter().collect()
            };
            for (&slot, expected) in writes.iter().zip(boundaries) {
                let error = relative_error(&widen(slot as usize)?, expected);
                assert!(
                    error < stored,
                    "{dtype:?} dk {dk} step {delta_step} tokens {tokens}: state {error}"
                );
            }
            assert_eq!(widen(*slot as usize)?, initial[*slot as usize]);
            row += tokens;
        }
        assert!(widen(13)?.iter().all(|v| v.is_nan()));
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

/// Store F32 pool rows in `format`. Quantized pools keep NaN poison in the scales of
/// unused slots (codes cannot hold NaN); returns the device pool and its exact dequantized
/// host values, which are what a correct kernel reads.
fn stored(
    device: &Device,
    rows: &[f32],
    slots: usize,
    kv_heads: usize,
    dim: usize,
    format: KvFormat,
) -> Result<(KvPool, Vec<f32>)> {
    let host = Tensor::from_slice(rows, (slots, kv_heads, dim), &Device::Cpu)?;
    let (data, scales) = quantize(format, &host)?;
    let scales = match scales {
        Some(scales) => {
            let mut values = scales
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let per = values.len() / slots;
            for slot in 0..slots {
                if rows[slot * kv_heads * dim].is_nan() {
                    values[slot * per..(slot + 1) * per].fill(f32::NAN);
                }
            }
            Some(Tensor::from_vec(values, scales.dims(), &Device::Cpu)?.to_dtype(DType::F16)?)
        }
        None => None,
    };
    let exact = read(&dequantize(format, &data, scales.as_ref())?)?;
    Ok((
        KvPool {
            data: data.to_device(device)?,
            scales: scales.map(|s| s.to_device(device)).transpose()?,
        },
        exact,
    ))
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

fn gate_heads(case: &AttentionCase) -> Heads {
    Heads {
        count: case.heads,
        dim: case.dim,
        head_stride: 2 * case.dim,
        col_offset: case.dim,
    }
}

/// Run attention over a pool stored in `format`; returns the F32 output and the reference
/// over the values that format actually stores.
fn run_attention(
    kernels: &Kernels,
    device: &Device,
    case: &AttentionCase,
    format: KvFormat,
) -> Result<(Vec<f32>, Vec<f64>)> {
    let (k, v, slots, table, first, visible) = paged_pool(case, 30);
    let tokens = visible.len();
    let q = values(tokens * case.heads * case.dim, 20, 1.0);
    let gate = values(tokens * case.heads * 2 * case.dim, 21, 2.0);
    let (key, k_exact) = stored(device, &k, slots, case.kv_heads, case.dim, format)?;
    let (value, v_exact) = stored(device, &v, slots, case.kv_heads, case.dim, format)?;
    let pool = AttentionPool {
        key,
        value,
        format,
        page_tokens: case.page,
        kv_heads: case.kv_heads,
        head_dim: case.dim,
    };
    let expected = attention_reference(
        case, &q, &k_exact, &v_exact, &gate, &table, &first, &visible,
    );
    let paged = kernels.paged_kv(case.page, slots, &table, &first, &visible)?;
    let out = kernels.attention(
        &upload(device, &q, &[tokens, case.heads, case.dim])?,
        &pool,
        &paged,
        &upload(device, &gate, &[tokens, case.heads * 2 * case.dim])?,
        gate_heads(case),
    )?;
    Ok((read(&out)?, expected))
}

fn check_attention(kernels: &Kernels, device: &Device, case: &AttentionCase) -> Result<()> {
    for format in [
        KvFormat::F32,
        KvFormat::F16,
        KvFormat::Q8,
        KvFormat::Q8 { group: 64 },
        KvFormat::Q8 { group: 128 },
        KvFormat::Q8 { group: 256 },
    ] {
        if format.group().is_some_and(|group| group > case.dim) {
            continue;
        }
        let (actual, expected) = run_attention(kernels, device, case, format)?;
        let error = relative_error(&actual, &expected);
        assert!(error < 1e-5, "{format} dim {}: {error}", case.dim);
    }
    Ok(())
}

#[test]
fn test_attention_paged_gqa_matches_reference_for_every_format_and_split() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let kernels = kernels.clone().with_tuning(Tuning {
        attention_tiled_groups: usize::MAX,
        ..kernels.tuning()
    })?;
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
fn test_tiled_attention_matches_reference_for_every_format_and_tile_shape() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let tiled = kernels.clone().with_tuning(Tuning {
        attention_tiled_groups: 0,
        ..kernels.tuning()
    })?;
    for dim in [64usize, 128, 256] {
        // Partial and whole query tiles, key counts off the 32-key blocks, a single-token
        // sequence and a long prefill continuing 37 committed keys.
        let case = AttentionCase {
            heads: 8,
            kv_heads: 2,
            dim,
            page: 16,
            sequences: vec![
                (37, 3),
                (0, 20),
                (15, 1),
                (37, 70),
                (0, 32),
                (65, 7),
                (31, 8),
                (0, 9),
            ],
        };
        let (_, _, slots, table, first, visible) = paged_pool(&case, 30);
        let paged = tiled.paged_kv(case.page, slots, &table, &first, &visible)?;
        assert_eq!(paged.tiles(), Some(10));
        assert!(tiled.attention_tiled(&paged, case.heads, dim));
        check_attention(&tiled, &device, &case)?;
    }
    for (heads, kv_heads, page) in [(1, 1, 1), (3, 3, 7), (6, 2, 64)] {
        let case = AttentionCase {
            heads,
            kv_heads,
            dim: 64,
            page,
            sequences: vec![(0, 1), (63, 33)],
        };
        check_attention(&tiled, &device, &case)?;
    }
    check_attention(
        &tiled,
        &device,
        &AttentionCase {
            heads: 1,
            kv_heads: 1,
            dim: 64,
            page: 32,
            sequences: vec![(2047, 33)],
        },
    )?;
    // Non-consecutive positions inside one sequence cannot tile.
    let paged = tiled.paged_kv(16, 64, &[0, 1], &[0, 0], &[3, 5])?;
    assert_eq!(paged.tiles(), None);
    assert!(!tiled.attention_tiled(&paged, 8, 64));
    Ok(())
}

#[test]
fn test_tiled_attention_dispatch_preserves_decode_parallelism() -> Result<()> {
    let (_, kernels, _gpu) = setup()?;
    let decode = kernels.paged_kv(16, 32, &[0, 1], &[0, 1, 0, 1], &[16; 4])?;
    assert!(!kernels.attention_tiled(&decode, 32, 128));

    let visible: Vec<u32> = (1..=256).collect();
    let table: Vec<u32> = (0..16).collect();
    let short = kernels.paged_kv(16, 256, &table, &[0; 32], &visible[..32])?;
    let crossover = kernels.paged_kv(16, 256, &table, &[0; 64], &visible[..64])?;
    assert!(!kernels.attention_tiled(&short, 8, 256));
    assert!(kernels.attention_tiled(&crossover, 8, 256));
    let prefill = kernels.paged_kv(16, 256, &table, &[0; 256], &visible)?;
    for dim in [64, 128, 256] {
        assert!(kernels.attention_tiled(&prefill, 32, dim));
    }
    for dim in [0, 32, 192, 512] {
        assert!(!kernels.attention_tiled(&prefill, 32, dim));
    }
    let per_query = kernels.clone().with_tuning(Tuning {
        attention_tiled_groups: usize::MAX,
        ..kernels.tuning()
    })?;
    assert!(!per_query.attention_tiled(&prefill, 32, 128));
    let forced = kernels.clone().with_tuning(Tuning {
        attention_tiled_groups: 0,
        ..kernels.tuning()
    })?;
    assert!(forced.attention_tiled(&decode, 32, 128));
    Ok(())
}

#[test]
fn test_quantized_kv_attention_stays_close_to_f32_attention() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    // Decode and prefill over 1000 keys: the error the K/V format adds to the output.
    let case = AttentionCase {
        heads: 8,
        kv_heads: 2,
        dim: 256,
        page: 64,
        sequences: vec![(999, 1), (200, 30)],
    };
    let (exact, _) = run_attention(&kernels, &device, &case, KvFormat::F32)?;
    let exact: Vec<f64> = exact.iter().map(|v| f64::from(*v)).collect();
    for (format, bound) in [
        (KvFormat::F16, 2e-3),
        (KvFormat::Q8, 2e-2),
        (KvFormat::Q8 { group: 256 }, 3e-2),
    ] {
        let (actual, _) = run_attention(&kernels, &device, &case, format)?;
        let error = relative_error(&actual, &exact);
        assert!(error < bound, "{format}: {error}");
    }
    Ok(())
}

#[test]
fn test_kv_write_quantizes_into_paged_slots_like_the_host_reference() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let case = AttentionCase {
        heads: 4,
        kv_heads: 2,
        dim: 128,
        page: 16,
        sequences: vec![(37, 3), (0, 20), (15, 1)],
    };
    let (_, _, slots, table, first, visible) = paged_pool(&case, 31);
    let tokens = visible.len();
    let width = case.kv_heads * case.dim;
    // Keys contiguous; values read in place from a wider interleaved projection.
    let key = values(tokens * width, 32, 4.0);
    let projection = values(tokens * (2 * width + 5), 33, 4.0);
    let value: Vec<f32> = (0..tokens)
        .flat_map(|t| projection[t * (2 * width + 5) + width..][..width].to_vec())
        .collect();
    let paged = kernels.paged_kv(case.page, slots, &table, &first, &visible)?;
    for format in [
        KvFormat::F32,
        KvFormat::F16,
        KvFormat::Q8,
        KvFormat::Q8 { group: 128 },
    ] {
        let poison = vec![f32::NAN; slots * width];
        let (key_pool, _) = stored(&device, &poison, slots, case.kv_heads, case.dim, format)?;
        let (value_pool, _) = stored(&device, &poison, slots, case.kv_heads, case.dim, format)?;
        let pool = AttentionPool {
            key: key_pool,
            value: value_pool,
            format,
            page_tokens: case.page,
            kv_heads: case.kv_heads,
            head_dim: case.dim,
        };
        kernels.kv_write(
            &upload(&device, &key, &[tokens, case.kv_heads, case.dim])?,
            Heads {
                count: case.kv_heads,
                dim: case.dim,
                head_stride: case.dim,
                col_offset: 0,
            },
            &upload(&device, &projection, &[tokens, 2 * width + 5])?,
            Heads {
                count: case.kv_heads,
                dim: case.dim,
                head_stride: case.dim,
                col_offset: width,
            },
            &pool,
            &paged,
        )?;
        for (rows, kv) in [(&key, &pool.key), (&value, &pool.value)] {
            let host = Tensor::from_slice(
                rows.as_slice(),
                (tokens, case.kv_heads, case.dim),
                &Device::Cpu,
            )?;
            let (codes, scales) = quantize(format, &host)?;
            let expected = read(&dequantize(format, &codes, scales.as_ref())?)?;
            let step = scales.map(|s| read(&s.to_dtype(DType::F32)?)).transpose()?;
            let stored_rows = read(&dequantize(
                format,
                &kv.data.to_device(&Device::Cpu)?,
                kv.scales
                    .as_ref()
                    .map(|s| s.to_device(&Device::Cpu))
                    .transpose()?
                    .as_ref(),
            )?)?;
            let mut written = vec![false; slots];
            let mut flips = 0;
            for t in 0..tokens {
                let j = visible[t] as usize - 1;
                let slot =
                    table[first[t] as usize + j / case.page] as usize * case.page + j % case.page;
                written[slot] = true;
                for i in 0..width {
                    let (a, b) = (stored_rows[slot * width + i], expected[t * width + i]);
                    if a != b {
                        // Division rounding may move a Q8 code across a .5 tie by one step.
                        let group = format.group().unwrap_or(case.dim);
                        let s = step.as_ref().map_or(0.0, |s| s[(t * width + i) / group]);
                        assert!(
                            (a - b).abs() <= s * 1.0001,
                            "{format} token {t} element {i}"
                        );
                        flips += 1;
                    }
                }
            }
            assert!(
                flips * 1000 <= tokens * width,
                "{format}: {flips} code flips"
            );
            // Unwritten slots keep their poison: nothing outside the target slots changed.
            for (slot, done) in written.iter().enumerate() {
                if !done {
                    assert!(
                        stored_rows[slot * width..(slot + 1) * width]
                            .iter()
                            .all(|v| v.is_nan())
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_gemv_matches_dequantized_reference_for_every_batch_size() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    // Exercise the GEMV at every row count it supports, and separately the small-batch
    // matrix kernel at every row count it supports (Q8_0 only).
    let gemv = kernels.clone().with_tuning(Tuning {
        gemv_q8_0_rows: (1, 8),
        gemv_q4k_rows: (1, 8),
        qmm_q8_0_rows: (16, 16),
        ..kernels.tuning()
    })?;
    let qmm = kernels.clone().with_tuning(Tuning {
        qmm_q8_0_rows: (1, 16),
        ..kernels.tuning()
    })?;
    // 37 outputs leave partial column tiles; 512 inputs give 16 blocks to split.
    let (n, k) = (37, 512);
    let weight = upload(&Device::Cpu, &values(n * k, 22, 1.0), &[n, k])?;
    for format in [GgmlDType::Q8_0, GgmlDType::Q4K] {
        let layer = PackedLinear::quantized(&weight, format, &device)?;
        assert_eq!(
            layer.resident_bytes(),
            n * k / format.block_size() * format.type_size()
        );
        let dequantized = read(&QTensor::quantize(&weight, format)?.dequantize(&Device::Cpu)?)?;
        for rows in 1..=16 {
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
            let input = upload(&device, &x, &[rows, k])?;
            if rows <= 9 {
                let error = relative_error(&read(&gemv.linear(&layer, &input)?)?, &expected);
                // Above the GEMV range Candle's tiled GEMM stages weight tiles as F16.
                let bound = if rows <= 8 { 1e-5 } else { 2e-3 };
                assert!(error < bound, "{format:?} rows {rows}: {error}");
            }
            if format == GgmlDType::Q8_0 {
                let error = relative_error(&read(&qmm.linear(&layer, &input)?)?, &expected);
                assert!(error < 1e-5, "matrix kernel rows {rows}: {error}");
                // Strided rows read in place: every other row of a wider tensor.
                let doubled: Vec<f32> = x
                    .chunks(k)
                    .flat_map(|r| r.iter().chain(r))
                    .copied()
                    .collect();
                let wide = upload(&device, &doubled, &[rows, 2 * k])?;
                let strided = wide.narrow(1, k, k)?;
                let error = relative_error(&read(&qmm.linear(&layer, &strided)?)?, &expected);
                assert!(error < 1e-5, "strided matrix kernel rows {rows}: {error}");
            }
        }
    }
    Ok(())
}

#[test]
fn test_argmax_keeps_the_first_maximum_and_flags_nonfinite_rows() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let cols = 248_320;
    let mut rows = [
        values(cols, 31, 4.0),
        values(cols, 32, 4.0),
        values(cols, 33, 4.0),
    ];
    // Row 0: a maximum repeated later; row 1: a maximum in the last column; row 2: NaN.
    rows[0][1000] = 9.0;
    rows[0][200_000] = 9.0;
    rows[1][cols - 1] = 9.0;
    rows[2][5] = f32::NAN;
    let flat: Vec<f32> = rows.concat();
    let out = read(&kernels.argmax(&upload(&device, &flat, &[3, cols])?)?)?;
    assert_eq!(out[..4], [1000.0, 1.0, (cols - 1) as f32, 1.0]);
    assert_eq!(out[5], 0.0);
    // Strided rows: the right half of a wider tensor.
    let wide: Vec<f32> = rows[..2]
        .iter()
        .flat_map(|r| r.iter().chain(r))
        .copied()
        .collect();
    let strided = upload(&device, &wide, &[2, 2 * cols])?.narrow(1, cols, cols)?;
    assert_eq!(
        read(&kernels.argmax(&strided)?)?,
        [1000.0, 1.0, (cols - 1) as f32, 1.0]
    );
    Ok(())
}

fn sample_params(temperature: f32, top_k: u32, top_p: f32, seed: u64, draw: u64) -> SampleParams {
    SampleParams {
        temperature,
        top_k,
        top_p,
        seed,
        draw,
    }
}

/// Device tokens and status words for each parameter set over rows `logits`.
fn device_samples(
    device: &Device,
    kernels: &Kernels,
    logits: &[f32],
    cols: usize,
    params: &[SampleParams],
) -> Result<Vec<(u32, SampleStatus)>> {
    let rows = params.len();
    let out = read(&kernels.sample(&upload(device, logits, &[rows, cols])?, params)?)?;
    Ok(out
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[token, word]| (token as u32, SampleStatus::from_word(word)))
        .collect())
}

/// Exact truncated probabilities in F64: kept tokens and their renormalized weights.
fn truncated(logits: &[f32], temperature: f32, top_k: u32, top_p: f32) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|&l| f64::from((l - max) / temperature).exp())
        .collect();
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
    let sum: f64 = weights.iter().sum();
    let mut kept = vec![0.0; weights.len()];
    let mut above = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        if (top_k != 0 && rank >= top_k as usize) || above >= f64::from(top_p) * sum {
            break;
        }
        kept[i] = weights[i];
        above += weights[i];
    }
    let mass: f64 = kept.iter().sum();
    kept.iter().map(|w| w / mass).collect()
}

#[test]
fn test_device_and_host_sampling_match_truncated_vocabulary_distributions() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let cols = 248_320;
    // Peaked like a language model's row: a few dominant logits over a long random tail.
    let mut row = values(cols, 41, 3.0);
    let heads = [17, 90_001, 123_456, cols - 1];
    for (&i, boost) in heads.iter().zip([9.0, 8.5, 8.0, 7.5]) {
        row[i] += boost;
    }
    let batch = 16;
    let logits = upload(
        &device,
        &(0..batch)
            .flat_map(|_| row.iter().copied())
            .collect::<Vec<_>>(),
        &[batch, cols],
    )?;
    for (temperature, top_k, top_p) in [
        (1.0, 0, 1.0),
        (0.6, 20, 0.95),
        (1.0, 20, 0.8),
        (1.5, 50, 1.0),
    ] {
        let truth = truncated(&row, temperature, top_k, top_p);
        // Buckets: each dominant token, then every other kept token.
        let bucket = |token: usize| {
            heads
                .iter()
                .position(|&h| h == token)
                .unwrap_or(heads.len())
        };
        let mut expected = vec![0.0; heads.len() + 1];
        for (token, &q) in truth.iter().enumerate() {
            expected[bucket(token)] += q;
        }
        let draws = 2048u64;
        let mut device_counts = vec![0u64; expected.len()];
        let mut host_counts = vec![0u64; expected.len()];
        for launch in 0..draws / batch as u64 {
            let params: Vec<SampleParams> = (0..batch as u64)
                .map(|r| sample_params(temperature, top_k, top_p, 99, launch * batch as u64 + r))
                .collect();
            let out = read(&kernels.sample(&logits, &params)?)?;
            for (pair, p) in out.as_chunks::<2>().0.iter().zip(&params) {
                assert_eq!(SampleStatus::from_word(pair[1]), SampleStatus::Sampled);
                let token = pair[0] as usize;
                assert!(truth[token] > 0.0, "device sampled truncated token {token}");
                device_counts[bucket(token)] += 1;
                let host = sampling::sample(&row, p)? as usize;
                assert!(truth[host] > 0.0, "host sampled truncated token {host}");
                host_counts[bucket(host)] += 1;
            }
        }
        for (path, counts) in [("device", &device_counts), ("host", &host_counts)] {
            let chi: f64 = expected
                .iter()
                .zip(counts)
                .filter(|(q, _)| **q > 1e-9)
                .map(|(q, &n)| {
                    let e = q * draws as f64;
                    (n as f64 - e).powi(2) / e
                })
                .sum();
            // 99.9th percentile of chi-square with at most 4 degrees of freedom.
            assert!(
                chi < 18.5,
                "{path} T={temperature} k={top_k} p={top_p}: χ²={chi:.1}, {counts:?} vs {expected:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_device_draws_match_exact_truncated_distributions() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let cols = 10;
    let logits: Vec<f32> = (0..cols)
        .map(|i| (((i * 3) % 10) as f32).mul_add(0.35, -1.5))
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for &(temperature, top_k, top_p) in
        &[(1.0, 0, 1.0), (0.8, 4, 1.0), (1.2, 0, 0.7), (1.0, 6, 0.85)]
    {
        let weights: Vec<f64> = logits
            .iter()
            .map(|&l| f64::from((l - max) / temperature).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        let kept: Vec<bool> = (0..cols)
            .map(|j| {
                let above: Vec<f64> = weights
                    .iter()
                    .copied()
                    .filter(|&w| w > weights[j])
                    .collect();
                (top_k == 0 || above.len() < top_k as usize)
                    && above.iter().sum::<f64>() < f64::from(top_p) * sum
            })
            .collect();
        let mass: f64 = (0..cols).filter(|&i| kept[i]).map(|i| weights[i]).sum();
        let draws = 20_000;
        let params: Vec<SampleParams> = (0..draws)
            .map(|draw| sample_params(temperature, top_k, top_p, 5, draw))
            .collect();
        let flat: Vec<f32> = (0..draws).flat_map(|_| logits.iter().copied()).collect();
        let mut counts = vec![0usize; cols];
        for (token, status) in device_samples(&device, &kernels, &flat, cols, &params)? {
            assert_eq!(status, SampleStatus::Sampled);
            assert!(kept[token as usize], "truncated token {token}");
            counts[token as usize] += 1;
        }
        let chi: f64 = (0..cols)
            .filter(|&i| kept[i])
            .map(|i| {
                let e = weights[i] / mass * draws as f64;
                (counts[i] as f64 - e).powi(2) / e
            })
            .sum();
        // 99.9th percentile of chi-square with at most 9 degrees of freedom.
        assert!(
            chi < 27.9,
            "T={temperature} k={top_k} p={top_p}: χ²={chi:.1}"
        );
    }
    Ok(())
}

#[test]
fn test_device_penalties_match_the_reference_and_invalid_rows_are_flagged() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let cols = 1000;
    let rows = [values(cols, 51, 2.0), values(cols, 52, 2.0)];
    let params = [
        PenaltyParams {
            repetition: 1.3,
            presence: 0.4,
            frequency: 0.2,
        },
        PenaltyParams::NONE,
    ];
    let entries = [
        (0, PenaltyEntry { token: 3, count: 0 }),
        (
            0,
            PenaltyEntry {
                token: 999,
                count: 4,
            },
        ),
        (1, PenaltyEntry { token: 3, count: 1 }),
    ];
    // Strided rows: the right half of a wider tensor, penalized in place.
    let wide: Vec<f32> = rows
        .iter()
        .flat_map(|r| r.iter().chain(r))
        .copied()
        .collect();
    let logits = upload(&device, &wide, &[2, 2 * cols])?.narrow(1, cols, cols)?;
    kernels.penalize(&logits, &params, &entries)?;
    let mut expected = rows;
    for (row, entry) in &entries {
        let r = *row as usize;
        sampling::penalize(&mut expected[r], &params[r], &[*entry])?;
    }
    assert_eq!(read(&logits.contiguous()?)?, expected.concat());
    let duplicate = [entries[0], entries[0]];
    assert!(kernels.penalize(&logits, &params, &duplicate).is_err());
    assert!(
        kernels
            .penalize(
                &logits,
                &params,
                &[(2, PenaltyEntry { token: 0, count: 0 })]
            )
            .is_err()
    );
    let mut bad = values(cols, 53, 1.0);
    bad[7] = f32::NAN;
    let p = [sample_params(1.0, 0, 1.0, 1, 0)];
    assert_eq!(
        device_samples(&device, &kernels, &bad, cols, &p)?[0].1,
        SampleStatus::Invalid
    );
    assert_eq!(
        device_samples(&device, &kernels, &vec![f32::NEG_INFINITY; cols], cols, &p)?[0].1,
        SampleStatus::Invalid
    );
    assert!(
        kernels
            .sample(
                &upload(&device, &bad, &[1, cols])?,
                &[sample_params(0.0, 0, 1.0, 1, 0)]
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn test_grammar_masks_restrict_argmax_and_sampling_to_allowed_tokens() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let cols: usize = 248_320;
    let words = cols.div_ceil(32);
    let rows = [
        values(cols, 71, 4.0),
        values(cols, 72, 4.0),
        values(cols, 73, 4.0),
    ];
    let logits = upload(&device, &rows.concat(), &[3, cols])?;
    let allowed = [3usize, 1000, 248_000];
    let mut bits = vec![0u32; 2 * words];
    for &token in &allowed {
        bits[token / 32] |= 1 << (token % 32);
    }
    // Row 1 keeps three tokens, row 2 none, row 0 is untouched.
    kernels.mask(&logits, &[1, 2], &bits)?;
    let masked = read(&logits)?;
    assert_eq!(masked[..cols], rows[0][..]);
    for (token, &value) in masked[cols..2 * cols].iter().enumerate() {
        if allowed.contains(&token) {
            assert_eq!(value, rows[1][token]);
        } else {
            assert_eq!(value, f32::NEG_INFINITY);
        }
    }
    let best = *allowed
        .iter()
        .max_by(|&&a, &&b| rows[1][a].total_cmp(&rows[1][b]))
        .ok_or_else(|| tur_core::TurError::Other("empty".into()))?;
    let out = read(&kernels.argmax(&logits)?)?;
    assert_eq!(out[1], 1.0);
    assert_eq!(out[2..4], [best as f32, 1.0]);
    // A row without any allowed token is invalid rather than an arbitrary index.
    assert_eq!(out[5], 0.0);
    let row = logits.narrow(0, 1, 1)?;
    for draw in 0..64 {
        let pair = read(&kernels.sample(&row, &[sample_params(1.5, 0, 1.0, 9, draw)])?)?;
        assert_eq!(SampleStatus::from_word(pair[1]), SampleStatus::Sampled);
        assert!(allowed.contains(&(pair[0] as usize)), "{}", pair[0]);
    }
    assert!(kernels.mask(&logits, &[1, 1], &bits).is_err());
    assert!(kernels.mask(&logits, &[3], &bits[..words]).is_err());
    assert!(kernels.mask(&logits, &[0], &bits).is_err());
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
    let mixed = Tensor::ones((2, 200), DType::F32, &device)?;
    let ones = Tensor::ones(1, DType::F32, &device)?;
    let span = |tokens: usize, read: u32, writes: &'static [u32]| RecurrentSpan {
        tokens,
        position: 0,
        read,
        writes,
    };
    let state = Tensor::zeros((3, 1, 96, 4), DType::F32, &device)?;
    let decodes = kernels.recurrent_spans(&[span(1, 0, &[1]), span(1, 0, &[2])])?;
    // Unsupported key dim, then addressing misuse at a supported one.
    assert!(
        kernels
            .delta_scan(&mixed, &mixed, &decodes, &shape, &ones, &ones, &state)
            .is_err()
    );
    let supported = DeltaShape {
        key_dim: 64,
        ..shape
    };
    let state = Tensor::zeros((3, 1, 64, 4), DType::F32, &device)?;
    let scan = |spans: &[RecurrentSpan<'_>]| {
        kernels.delta_scan(
            &mixed,
            &mixed,
            &kernels.recurrent_spans(spans)?,
            &supported,
            &ones,
            &ones,
            &state,
        )
    };
    assert!(scan(&[span(1, 0, &[1]), span(1, 0, &[2])]).is_ok());
    assert!(scan(&[span(2, 0, &[1])]).is_ok());
    assert!(
        scan(&[span(1, 1, &[1]), span(1, 0, &[2])]).is_err(),
        "writing its committed slot"
    );
    assert!(
        scan(&[span(1, 2, &[1]), span(1, 0, &[2])]).is_err(),
        "writing another's committed slot"
    );
    assert!(
        scan(&[span(1, 0, &[1]), span(1, 0, &[1])]).is_err(),
        "one slot written twice"
    );
    assert!(
        scan(&[span(1, 0, &[1]), span(1, 0, &[3])]).is_err(),
        "slot outside the pool"
    );
    assert!(
        scan(&[span(1, 3, &[1]), span(1, 0, &[2])]).is_err(),
        "read outside the pool"
    );
    assert!(
        scan(&[span(1, 0, &[1, 2]), span(1, 0, &[])]).is_err(),
        "wrong boundary counts"
    );
    assert!(
        scan(&[span(1, 0, &[1])]).is_err(),
        "spans cover one of two rows"
    );
    assert!(
        scan(&[span(0, 0, &[1]), span(2, 0, &[2])]).is_err(),
        "empty sequence"
    );
    assert!(kernels.recurrent_spans(&[]).is_err());
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
    let (keys, vals, slots, table, first, visible) = paged_pool(&case, 40);
    let (key, _) = stored(&device, &keys, slots, 2, 128, KvFormat::Q8)?;
    let (value, _) = stored(&device, &vals, slots, 2, 128, KvFormat::Q8)?;
    let pool = AttentionPool {
        key,
        value,
        format: KvFormat::Q8,
        page_tokens: 16,
        kv_heads: 2,
        head_dim: 128,
    };
    let queries = upload(&device, &values(2 * 8 * 128, 41, 1.0), &[2, 8, 128])?;
    let gate = upload(&device, &values(2 * 8 * 256, 42, 1.0), &[2, 8 * 256])?;
    let weight = upload(&device, &values(1024, 43, 1.0), &[1024])?;
    let input = upload(&device, &values(2 * 1024, 44, 1.0), &[2, 1024])?;
    let step = || -> Result<()> {
        let paged = kernels.paged_kv(16, slots, &table, &first, &visible)?;
        let heads = Heads {
            count: 8,
            dim: 128,
            head_stride: 256,
            col_offset: 128,
        };
        let attended = kernels.attention(&queries, &pool, &paged, &gate, heads)?;
        let (sum, _) = kernels.add_rms_norm(&input, &attended, &weight, 1e-6)?;
        let _normed = kernels.rms_norm(&sum, &weight, 1e-6)?;
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
    let mut completion = Completion::new(device, &Quarantine::new(), out);
    assert!(completion.take().is_none());
    completion.wait()?;
    let out = completion
        .take()
        .ok_or_else(|| tur_core::TurError::Other("not retained".into()))?;
    let expected = rms_row(&read(&x)?, &[1.0; 64], 1e-6);
    assert!(relative_error(&read(&out)?, &expected) < 1e-6);
    Ok(())
}

#[test]
fn test_sampling_repeats_exactly_for_identical_inputs() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    for cols in [160usize, 248_320] {
        let logits = upload(&device, &values(cols, 81, 2.0), &[1, cols])?;
        for draw in 0..32 {
            let p = [sample_params(1.2, 40, 1.0, 6, draw)];
            let first = read(&kernels.sample(&logits, &p)?)?;
            for _ in 0..8 {
                assert_eq!(
                    read(&kernels.sample(&logits, &p)?)?,
                    first,
                    "cols {cols} draw {draw}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_penalties_on_allocated_copies_repeat_exactly_in_any_entry_order() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let cols = 160usize;
    let logits = upload(&device, &values(cols, 81, 2.0), &[1, cols])?;
    let penalties = [PenaltyParams {
        repetition: 1.2,
        presence: 0.0,
        frequency: 0.0,
    }];
    let mut entries: Vec<(u32, PenaltyEntry)> = (0..21u32)
        .map(|t| {
            (
                0,
                PenaltyEntry {
                    token: (t * 37 + 11) % 148 + 2,
                    count: 0,
                },
            )
        })
        .collect();
    entries.sort_by_key(|e| e.1.token);
    entries.dedup_by_key(|e| e.1.token);
    let mut outcomes = Vec::new();
    for rotation in 0..entries.len() {
        entries.rotate_left(1);
        // `Tensor::copy` shares storage on Metal: in-place kernels need an allocated copy.
        let copy = logits.force_contiguous()?;
        kernels.penalize(&copy, &penalties, &entries)?;
        let penalized = read(&copy)?;
        let pair = read(&kernels.sample(&copy, &[sample_params(1.2, 40, 1.0, 6, 0)])?)?;
        outcomes.push((penalized, pair, rotation));
    }
    for (penalized, pair, rotation) in &outcomes {
        assert_eq!(penalized, &outcomes[0].0, "rotation {rotation}");
        assert_eq!(pair, &outcomes[0].1, "rotation {rotation}");
    }
    Ok(())
}
