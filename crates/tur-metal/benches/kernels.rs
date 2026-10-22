#![forbid(unsafe_code)]
//! Kernel benchmarks: each custom kernel against the existing Candle path at Qwen3.5
//! shapes. Prints one JSON document. Timing covers completed GPU work: every sample
//! enqueues `reps` operations and then synchronizes.
//!
//! `cargo bench --offline -p tur-metal --bench kernels > out.json`

use candle_core::{D, DType, Device, Tensor, quantized::GgmlDType};
use serde_json::{Value, json};
use std::time::Instant;
use tur_core::Result;
use tur_metal::{
    kernels::{DeltaShape, Heads, Kernels, Tuning},
    packed::PackedLinear,
};

const WARMUP: usize = 3;
const SAMPLES: usize = 15;

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

fn random(device: &Device, shape: &[usize], seed: u64, scale: f32) -> Result<Tensor> {
    let count = shape.iter().product();
    Ok(Tensor::from_vec(values(count, seed, scale), shape, device)?)
}

/// Target wall time per sample; reps grow until synchronization latency is amortized.
const SAMPLE_SECONDS: f64 = 0.05;

/// Median and p90 seconds per operation over `SAMPLES` batches of operations.
fn time(device: &Device, mut op: impl FnMut() -> Result<Tensor>) -> Result<Value> {
    for _ in 0..WARMUP {
        op()?;
        device.synchronize()?;
    }
    let start = Instant::now();
    let probe = op()?;
    device.synchronize()?;
    drop(probe);
    let estimate = start.elapsed().as_secs_f64().max(1e-6);
    let reps = ((SAMPLE_SECONDS / estimate).ceil() as usize).clamp(1, 1000);
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        device.synchronize()?;
        let start = Instant::now();
        // Outputs drop immediately so the pool recycles them, as a serving loop would;
        // queue ordering keeps that safe and the timed region ends at device completion.
        for _ in 0..reps {
            op()?;
        }
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() / reps as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok(json!({
        "median_seconds": samples[SAMPLES / 2],
        "p90_seconds": samples[SAMPLES * 9 / 10],
        "reps_per_sample": reps,
    }))
}

fn speedup(custom: &Value, existing: &Value) -> f64 {
    existing["median_seconds"].as_f64().unwrap_or(0.0)
        / custom["median_seconds"].as_f64().unwrap_or(f64::INFINITY)
}

fn candle_rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let inv = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?.broadcast_mul(w)?)
}

/// The P3 reference attention for one sequence: repeated K/V heads, a masked score
/// block and softmax through Candle matmuls.
fn candle_attention(q: &Tensor, k: &Tensor, v: &Tensor, past: usize) -> Result<Tensor> {
    let (n, heads, dim) = q.dims3()?;
    let (length, kv_heads, _) = k.dims3()?;
    let groups = heads / kv_heads;
    let repeat = |x: &Tensor| -> Result<Tensor> {
        Ok(x.unsqueeze(2)?
            .broadcast_as((length, kv_heads, groups, dim))?
            .reshape((length, heads, dim))?
            .transpose(0, 1)?
            .contiguous()?)
    };
    let keys_t = repeat(k)?.transpose(1, 2)?.contiguous()?;
    let values = repeat(v)?;
    let scores = (q.transpose(0, 1)?.contiguous()?.matmul(&keys_t)? / (dim as f64).sqrt())?;
    let mask: Vec<f32> = (0..n * length)
        .map(|i| {
            if i % length <= past + i / length {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let mask = Tensor::from_vec(mask, (1, n, length), q.device())?;
    let probs = candle_nn_free_softmax(&scores.broadcast_add(&mask)?)?;
    Ok(probs.matmul(&values)?.transpose(0, 1)?.contiguous()?)
}

/// Softmax over the last dimension with Candle tensor ops (tur-metal has no candle-nn).
fn candle_nn_free_softmax(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    Ok(e.broadcast_div(&e.sum_keepdim(D::Minus1)?)?)
}

fn gemv_cases(device: &Device, kernels: &Kernels) -> Result<Vec<Value>> {
    // (label, out, in): Qwen3.5-0.8B projections and Qwen3.8-27B MLP shapes.
    let shapes = [
        ("0.8b delta in_proj", 8224usize, 1024usize),
        ("0.8b gate_up", 7168, 1024),
        ("0.8b down", 1024, 3584),
        ("0.8b lm_head", 248_320, 1024),
        ("27b gate_up", 34816, 5120),
        ("27b down", 5120, 17408),
    ];
    let gemv = kernels.clone().with_tuning(Tuning {
        gemv_q8_0_rows: (1, 8),
        gemv_q4k_rows: (1, 8),
        ..kernels.tuning()
    })?;
    let mut cases = Vec::new();
    for (label, n, k) in shapes {
        let weight = random(&Device::Cpu, &[n, k], 1, 0.05)?;
        for format in [GgmlDType::Q8_0, GgmlDType::Q4K] {
            let layer = PackedLinear::quantized(&weight, format, device)?;
            for rows in [1usize, 2, 3, 4, 5, 6, 7, 8, 16, 128] {
                let x = random(device, &[rows, k], 2, 1.0)?;
                let existing = time(device, || layer.forward(&x))?;
                let mut case = json!({
                    "operation": "packed_linear", "shape": label, "out": n, "in": k,
                    "format": format!("{format:?}"), "rows": rows,
                    "weight_bytes": layer.resident_bytes(),
                    "candle": existing,
                    "candle_kernel": if rows == 1 { "mul_mv" } else { "mul_mm (F16 tiles)" },
                });
                if rows <= 8 {
                    // Force the GEMV for every measured row count, whatever the dispatch range.
                    let custom = time(device, || gemv.linear(&layer, &x))?;
                    let seconds = custom["median_seconds"].as_f64().unwrap_or(f64::INFINITY);
                    case["tur_gemv"] = custom.clone();
                    case["speedup"] = json!(speedup(&custom, &existing));
                    case["tur_weight_gb_per_s"] =
                        json!(layer.resident_bytes() as f64 / seconds / 1e9);
                }
                cases.push(case);
            }
        }
    }
    Ok(cases)
}

fn attention_cases(device: &Device, kernels: &Kernels) -> Result<Vec<Value>> {
    let (heads, kv_heads, dim, page) = (8usize, 2usize, 256usize, 16usize);
    let mut cases = Vec::new();
    // (label, past, new tokens per sequence, sequences)
    let shapes = [
        ("prefill 128", 0usize, 128usize, 1usize),
        ("prefill 2048", 0, 2048, 1),
        ("decode ctx 2048", 2047, 1, 1),
        ("decode ctx 8192", 8191, 1, 1),
        ("decode ctx 2048 x8", 2047, 1, 8),
    ];
    for (label, past, new, sequences) in shapes {
        let length = past + new;
        let pages = length.div_ceil(page);
        let slots = pages * page * sequences;
        let k_pool = random(device, &[slots, kv_heads, dim], 3, 1.0)?;
        let v_pool = random(device, &[slots, kv_heads, dim], 4, 1.0)?;
        let tokens = new * sequences;
        let q = random(device, &[tokens, heads, dim], 5, 1.0)?;
        let gate = random(device, &[tokens, heads * 2 * dim], 6, 1.0)?;
        let table: Vec<u32> = (0..(pages * sequences) as u32).collect();
        let mut first = Vec::new();
        let mut visible = Vec::new();
        for s in 0..sequences {
            for t in 0..new {
                first.push((s * pages) as u32);
                visible.push((past + t + 1) as u32);
            }
        }
        let gate_heads = Heads {
            count: heads,
            dim,
            head_stride: 2 * dim,
            col_offset: dim,
        };
        let custom = time(device, || {
            let paged = kernels.paged_kv(page, slots, &table, &first, &visible)?;
            kernels.attention(&q, &k_pool, &v_pool, &paged, &gate, gate_heads)
        })?;
        // The existing path also needs per-sequence K/V, gating and concatenation.
        let k_seq = k_pool.narrow(0, 0, length)?;
        let v_seq = v_pool.narrow(0, 0, length)?;
        let existing = time(device, || {
            let mut outputs = Vec::with_capacity(sequences);
            for s in 0..sequences {
                let qs = q.narrow(0, s * new, new)?;
                outputs.push(candle_attention(&qs, &k_seq, &v_seq, past)?);
            }
            let attended = Tensor::cat(&outputs, 0)?.reshape((tokens, heads * dim))?;
            let g = gate
                .reshape((tokens, heads, 2 * dim))?
                .narrow(2, dim, dim)?
                .contiguous()?
                .reshape((tokens, heads * dim))?;
            let sig = (g.neg()?.exp()? + 1.0)?.recip()?;
            Ok((attended * sig)?)
        })?;
        cases.push(json!({
            "operation": "attention", "shape": label, "heads": heads, "kv_heads": kv_heads,
            "head_dim": dim, "context": length, "sequences": sequences, "page": page,
            "split": format!("{:?}", kernels.attention_split(tokens, heads, length)),
            "tur": custom, "candle_reference": existing,
            "speedup": speedup(&custom, &existing),
        }));
    }
    Ok(cases)
}

fn delta_cases(device: &Device, kernels: &Kernels) -> Result<Vec<Value>> {
    // Qwen3.5-0.8B: 16 key/value heads of 128; Qwen3.8-27B: 16 key and 48 value heads.
    let mut cases = Vec::new();
    for (label, key_heads, value_heads) in [("0.8b", 16usize, 16usize), ("27b", 16, 48)] {
        let shape = DeltaShape {
            key_heads,
            value_heads,
            key_dim: 128,
            value_dim: 128,
            b_col: 0,
            a_col: value_heads,
        };
        let channels = shape.channels();
        for tokens in [1usize, 128, 2048] {
            let mixed = random(device, &[tokens, channels], 7, 1.0)?;
            let proj = random(device, &[tokens, 2 * value_heads], 8, 1.0)?;
            let history = random(device, &[3, channels], 9, 1.0)?;
            let conv_w = random(device, &[channels, 4], 10, 0.5)?;
            let dt_bias = random(device, &[value_heads], 11, 1.0)?;
            let neg_a = (random(device, &[value_heads], 12, 1.0)?.abs()? * -1.0)?;
            let state = random(device, &[value_heads, 128, 128], 13, 0.1)?;
            let conv = time(device, || {
                let out = kernels.output(tokens, channels)?;
                kernels.delta_conv(&mixed, 0, tokens, &history, &conv_w, false, &out, 0)?;
                kernels.finish(out, (tokens, channels))
            })?;
            let scan = time(device, || {
                let out = kernels.output(tokens, value_heads * 128)?;
                kernels.delta_scan(
                    &mixed, &proj, 0, tokens, &shape, &dt_bias, &neg_a, &state, false, &out, 0,
                )?;
                kernels.finish(out, (tokens, value_heads * 128))
            })?;
            let per_token = scan["median_seconds"].as_f64().unwrap_or(0.0) / tokens as f64;
            // Threadgroup shape sweep: the staged state tile trades coalescing for occupancy.
            let mut sweep = Vec::new();
            for scan_simdgroups in [1usize, 2, 4, 8] {
                let tuned = kernels.clone().with_tuning(Tuning {
                    scan_simdgroups,
                    ..kernels.tuning()
                })?;
                let timing = time(device, || {
                    let out = tuned.output(tokens, value_heads * 128)?;
                    tuned.delta_scan(
                        &mixed, &proj, 0, tokens, &shape, &dt_bias, &neg_a, &state, false, &out, 0,
                    )?;
                    tuned.finish(out, (tokens, value_heads * 128))
                })?;
                sweep.push(json!({"scan_simdgroups": scan_simdgroups, "scan": timing}));
            }
            cases.push(json!({
                "operation": "delta_net", "shape": label, "tokens": tokens,
                "key_heads": key_heads, "value_heads": value_heads, "key_dim": 128, "value_dim": 128,
                "conv": conv, "scan": scan, "scan_seconds_per_token": per_token,
                "scan_simdgroup_sweep": sweep,
            }));
        }
    }
    Ok(cases)
}

fn norm_cases(device: &Device, kernels: &Kernels) -> Result<Vec<Value>> {
    let mut cases = Vec::new();
    for (hidden, rows) in [(1024usize, 1usize), (1024, 2048), (5120, 8)] {
        let x = random(device, &[rows, hidden], 14, 1.0)?;
        let r = random(device, &[rows, hidden], 15, 1.0)?;
        let w = random(device, &[hidden], 16, 1.0)?;
        let custom = time(device, || Ok(kernels.add_rms_norm(&x, &r, &w, 1e-6)?.1))?;
        let existing = time(device, || candle_rms(&(&x + &r)?, &w, 1e-6))?;
        cases.push(json!({
            "operation": "add_rms_norm", "hidden": hidden, "rows": rows,
            "tur": custom, "candle_ops": existing, "speedup": speedup(&custom, &existing),
        }));
        let gu = random(device, &[rows, 2 * hidden], 17, 1.0)?;
        let custom = time(device, || kernels.swiglu(&gu))?;
        let existing = time(device, || {
            let g = gu.narrow(1, 0, hidden)?;
            let silu = (&g * (g.neg()?.exp()? + 1.0)?.recip()?)?;
            Ok((silu * gu.narrow(1, hidden, hidden)?)?)
        })?;
        cases.push(json!({
            "operation": "swiglu", "hidden": hidden, "rows": rows,
            "tur": custom, "candle_ops": existing, "speedup": speedup(&custom, &existing),
        }));
    }
    Ok(cases)
}

/// `TUR_BENCH_ONLY=delta|linear` restricts the run to one group (normalization runs with
/// delta).
fn only(group: &str) -> bool {
    std::env::var("TUR_BENCH_ONLY").is_ok_and(|v| v == group)
}

fn skip(group: &str) -> bool {
    std::env::var("TUR_BENCH_ONLY").is_ok_and(|v| v != group)
}

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let kernels = Kernels::new(&device)?;
    let document = json!({
        "schema_version": 1,
        "workload": "synthetic kernel shapes from Qwen3.5-0.8B and Qwen3.8-27B; not model or serving throughput",
        "dtype": DType::F32.as_str(),
        "capabilities": format!("{:?}", kernels.capabilities()),
        "tuning": format!("{:?}", kernels.tuning()),
        "layout_version": tur_metal::kernels::LAYOUT_VERSION,
        "warmup": WARMUP, "samples": SAMPLES,
        "timing": "host wall time per operation over an adaptive batch of reps ending at Device::synchronize; outputs are recycled through the pool between reps",
        "cases": {
            "norm": if skip("delta") { json!([]) } else { json!(norm_cases(&device, &kernels)?) },
            "delta_net": if skip("delta") { json!([]) } else { json!(delta_cases(&device, &kernels)?) },
            "attention": if only("delta") || only("linear") { json!([]) } else { json!(attention_cases(&device, &kernels)?) },
            "packed_linear": if skip("linear") { json!([]) } else { json!(gemv_cases(&device, &kernels)?) },
        },
    });
    let text = serde_json::to_string_pretty(&document)?;
    std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes())?;
    Ok(())
}
