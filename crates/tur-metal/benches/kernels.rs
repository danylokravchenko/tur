#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmarks emit JSON artifacts and timing summaries"
)]
#![forbid(unsafe_code)]
//! Kernel benchmarks: each custom kernel against the existing Candle path at Qwen3.5
//! shapes. Prints one JSON document. Timing covers completed GPU work: every sample
//! enqueues `reps` operations and then synchronizes.
//!
//! `cargo bench --offline -p tur-metal --bench kernels > out.json`
//! `TUR_BENCH_ONLY=delta|linear|paged|tiled` restricts the run to one group.
//! `TUR_BENCH_SHAPE=<substring>` selects tiled attention shapes for focused tuning.

use candle_core::{D, DType, Device, Tensor, quantized::GgmlDType};
use serde_json::{Value, json};
use std::time::Instant;
use tur_core::{Result, execution::KvFormat};
use tur_metal::{
    kernels::{DeltaShape, Heads, Kernels, PagedKv, Tuning},
    packed::PackedLinear,
    pool::{AttentionPool, KvPool},
};

const WARMUP: usize = 3;
const SAMPLES: usize = 15;

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

/// Contiguous page tables for `sequences` sequences of `past` committed and `new` tokens,
/// and a pool of `format` holding random keys and values written by the kernel itself.
struct Paged {
    pool: AttentionPool,
    kv: PagedKv,
    /// Addressing that writes one key per committed and new position, to fill the pool.
    fill: PagedKv,
    slots: usize,
}

#[allow(clippy::too_many_arguments)]
fn paged(
    device: &Device,
    kernels: &Kernels,
    format: KvFormat,
    page: usize,
    kv_heads: usize,
    dim: usize,
    past: usize,
    new: usize,
    sequences: usize,
) -> Result<Paged> {
    let length = past + new;
    let pages = length.div_ceil(page);
    let slots = pages * page * sequences;
    let store = || -> Result<KvPool> {
        Ok(KvPool {
            data: Tensor::zeros((slots, kv_heads, dim), format.data_dtype(), device)?,
            scales: format
                .group()
                .map(|g| Tensor::zeros((slots, kv_heads, dim / g), DType::F16, device))
                .transpose()?,
        })
    };
    let pool = AttentionPool {
        key: store()?,
        value: store()?,
        format,
        page_tokens: page,
        kv_heads,
        head_dim: dim,
    };
    let table: Vec<u32> = (0..(pages * sequences) as u32).collect();
    let (mut first, mut visible) = (Vec::new(), Vec::new());
    let (mut fill_first, mut fill_visible) = (Vec::new(), Vec::new());
    for s in 0..sequences {
        let base = (s * pages) as u32;
        for t in 0..new {
            first.push(base);
            visible.push((past + t + 1) as u32);
        }
        for t in 0..length {
            fill_first.push(base);
            fill_visible.push((t + 1) as u32);
        }
    }
    let kv = kernels.paged_kv(page, slots, &table, &first, &visible)?;
    let fill = kernels.paged_kv(page, slots, &table, &fill_first, &fill_visible)?;
    let rows = random(device, &[length * sequences, kv_heads, dim], 3, 1.0)?;
    let heads = Heads {
        count: kv_heads,
        dim,
        head_stride: dim,
        col_offset: 0,
    };
    kernels.kv_write(&rows, heads, &rows, heads, &pool, &fill)?;
    device.synchronize()?;
    Ok(Paged {
        pool,
        kv,
        fill,
        slots,
    })
}

const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;

fn gate_heads() -> Heads {
    Heads {
        count: HEADS,
        dim: HEAD_DIM,
        head_stride: 2 * HEAD_DIM,
        col_offset: HEAD_DIM,
    }
}

fn attention_cases(device: &Device, kernels: &Kernels) -> Result<Vec<Value>> {
    let (heads, kv_heads, dim, page) = (HEADS, KV_HEADS, HEAD_DIM, 16usize);
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
        let paged = paged(
            device,
            kernels,
            KvFormat::F32,
            page,
            kv_heads,
            dim,
            past,
            new,
            sequences,
        )?;
        let tokens = new * sequences;
        let q = random(device, &[tokens, heads, dim], 5, 1.0)?;
        let gate = random(device, &[tokens, heads * 2 * dim], 6, 1.0)?;
        let custom = time(device, || {
            kernels.attention(&q, &paged.pool, &paged.kv, &gate, gate_heads())
        })?;
        // The existing path also needs per-sequence K/V, gating and concatenation.
        let k_seq = paged.pool.key.data.narrow(0, 0, length)?;
        let v_seq = paged.pool.value.data.narrow(0, 0, length)?;
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
            "kv_format": "f32",
            "split": format!("{:?}", kernels.attention_split(tokens, heads, length)),
            "tur": custom, "candle_reference": existing,
            "speedup": speedup(&custom, &existing),
        }));
    }
    Ok(cases)
}

/// K/V format × page size sweep at Qwen3.5-0.8B attention shapes: attention over pages
/// in place, and the quantize-on-write cost of the pass's new rows.
fn paged_cases(device: &Device, kernels: &Kernels) -> Result<Vec<Value>> {
    let formats = [
        KvFormat::F32,
        KvFormat::F16,
        KvFormat::Q8,
        KvFormat::Q8 { group: 64 },
        KvFormat::Q8 { group: 256 },
    ];
    // (label, past, new tokens per sequence, sequences)
    let shapes = [
        ("decode ctx 2048", 2047usize, 1usize, 1usize),
        ("decode ctx 8192", 8191, 1, 1),
        ("decode ctx 32768", 32767, 1, 1),
        ("decode ctx 2048 x8", 2047, 1, 8),
        ("prefill 2048", 0, 2048, 1),
    ];
    let mut cases = Vec::new();
    for (label, past, new, sequences) in shapes {
        let tokens = new * sequences;
        let q = random(device, &[tokens, HEADS, HEAD_DIM], 5, 1.0)?;
        let gate = random(device, &[tokens, HEADS * 2 * HEAD_DIM], 6, 1.0)?;
        let rows = random(device, &[tokens, KV_HEADS, HEAD_DIM], 7, 1.0)?;
        let heads = Heads {
            count: KV_HEADS,
            dim: HEAD_DIM,
            head_stride: HEAD_DIM,
            col_offset: 0,
        };
        for format in formats {
            for page in [16usize, 32, 64, 128] {
                let paged = paged(
                    device, kernels, format, page, KV_HEADS, HEAD_DIM, past, new, sequences,
                )?;
                let attention = time(device, || {
                    kernels.attention(&q, &paged.pool, &paged.kv, &gate, gate_heads())
                })?;
                let write = time(device, || {
                    kernels.kv_write(&rows, heads, &rows, heads, &paged.pool, &paged.kv)?;
                    Ok(rows.clone())
                })?;
                let bytes = format.token_bytes(KV_HEADS, HEAD_DIM)? * 2 * paged.slots;
                cases.push(json!({
                    "operation": "paged_attention", "shape": label, "kv_format": format.to_string(),
                    "page": page, "context": past + new, "sequences": sequences,
                    "pool_bytes": bytes, "fill_tokens": paged.fill.tokens(),
                    "split": format!("{:?}", kernels.attention_split(tokens, HEADS, past + new)),
                    "attention": attention, "kv_write": write,
                }));
            }
        }
    }
    Ok(cases)
}

/// Tiled matrix-unit kernel against the per-query kernel (with its split-K choice), to
/// place the dispatch crossover: (label, past keys, new tokens per sequence, sequences).
fn tiled_cases(device: &Device, kernels: &Kernels) -> Result<Vec<Value>> {
    let tiled = kernels.clone().with_tuning(Tuning {
        attention_tiled_groups: 0,
        ..kernels.tuning()
    })?;
    let simd = kernels.clone().with_tuning(Tuning {
        attention_tiled_groups: usize::MAX,
        ..kernels.tuning()
    })?;
    let shapes = [
        ("prefill 16", 0usize, 16usize, 1usize),
        ("prefill 32", 0, 32, 1),
        ("prefill 64", 0, 64, 1),
        ("prefill 128", 0, 128, 1),
        ("prefill 512", 0, 512, 1),
        ("prefill 2048", 0, 2048, 1),
        ("prefill 8192", 0, 8192, 1),
        ("chunk 16 after 2048", 2048, 16, 1),
        ("chunk 64 after 2048", 2048, 64, 1),
        ("chunk 256 after 8192", 8192, 256, 1),
        ("chunk 32 x4 after 1024", 1024, 32, 4),
        ("decode ctx 2048 x8", 2047, 1, 8),
    ];
    let mut cases = Vec::new();
    let shape_filter = std::env::var("TUR_BENCH_SHAPE").ok();
    for (label, past, new, sequences) in shapes {
        if shape_filter
            .as_ref()
            .is_some_and(|filter| !label.contains(filter))
        {
            continue;
        }
        let tokens = new * sequences;
        let q = random(device, &[tokens, HEADS, HEAD_DIM], 5, 1.0)?;
        let gate = random(device, &[tokens, HEADS * 2 * HEAD_DIM], 6, 1.0)?;
        for format in [KvFormat::F32, KvFormat::F16, KvFormat::Q8] {
            let paged = paged(
                device, kernels, format, 16, KV_HEADS, HEAD_DIM, past, new, sequences,
            )?;
            let run = |k: &Kernels| {
                time(device, || {
                    k.attention(&q, &paged.pool, &paged.kv, &gate, gate_heads())
                })
            };
            let (t, s) = (run(&tiled)?, run(&simd)?);
            eprintln!(
                "{label:23} {format:5}: tiled {:8.3} ms  per-query {:8.3} ms  x{:.2}",
                t["median_seconds"].as_f64().unwrap_or(0.0) * 1e3,
                s["median_seconds"].as_f64().unwrap_or(0.0) * 1e3,
                speedup(&t, &s),
            );
            cases.push(json!({
                "operation": "attention_tiled", "shape": label, "kv_format": format.to_string(),
                "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                "context": past + new, "sequences": sequences, "tiles": paged.kv.tiles(),
                "default_kernel": if kernels.attention_tiled(&paged.kv, HEADS, HEAD_DIM) {
                    "tiled"
                } else {
                    "per_query"
                },
                "tiled": t, "per_query": s, "speedup": speedup(&t, &s),
            }));
        }
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
            // Pools of two slots: the committed state in slot 0, the new one in slot 1.
            let history = random(device, &[2, 3, channels], 9, 1.0)?;
            let conv_w = random(device, &[channels, 4], 10, 0.5)?;
            let dt_bias = random(device, &[value_heads], 11, 1.0)?;
            let neg_a = (random(device, &[value_heads], 12, 1.0)?.abs()? * -1.0)?;
            let state = random(device, &[2, value_heads, 128, 128], 13, 0.1)?;
            let writes = kernels.slots(&[1])?;
            let conv = time(device, || {
                let out = kernels.output(tokens, channels)?;
                kernels.delta_conv(&mixed, 0, tokens, &history, 0, &writes, &conv_w, &out, 0)?;
                kernels.finish(out, (tokens, channels))
            })?;
            let scan = time(device, || {
                let out = kernels.output(tokens, value_heads * 128)?;
                kernels.delta_scan(
                    &mixed, &proj, 0, tokens, &shape, &dt_bias, &neg_a, &state, 0, &writes, &out, 0,
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
                        &mixed, &proj, 0, tokens, &shape, &dt_bias, &neg_a, &state, 0, &writes,
                        &out, 0,
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

/// `TUR_BENCH_ONLY=delta|linear|paged|tiled` restricts the run to one group (normalization runs
/// with delta, the F32 attention baseline with paged).
fn skip(group: &str) -> bool {
    std::env::var("TUR_BENCH_ONLY").is_ok_and(|v| v != group)
}

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let kernels = Kernels::new(&device)?;
    let document = json!({
        "schema_version": 3,
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
            "attention": if skip("paged") { json!([]) } else { json!(attention_cases(&device, &kernels)?) },
            "paged_attention": if skip("paged") { json!([]) } else { json!(paged_cases(&device, &kernels)?) },
            "tiled_attention": if skip("tiled") { json!([]) } else { json!(tiled_cases(&device, &kernels)?) },
            "packed_linear": if skip("linear") { json!([]) } else { json!(gemv_cases(&device, &kernels)?) },
        },
    });
    let text = serde_json::to_string_pretty(&document)?;
    std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes())?;
    Ok(())
}
