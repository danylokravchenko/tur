//! Expert cache measurements on one gpt-oss checkpoint in a fresh process: what teacher-forced
//! passes read and how long they wait, whether anything was written to the SSD or swapped,
//! and optionally the quality of a pruned or smaller-cache checkpoint against a reference.
//!
//! `cargo bench --offline -p tur --bench perf -- experts <dir> <cache MiB> <text file>
//!  [reference dir]`. Tokens and samples: `TUR_EXPERTS_TOKENS` (2048), `TUR_EXPERTS_SAMPLE`
//! (128). SSD counters are system-wide, so run it on an otherwise idle machine.

use candle_core::{D, DType, Device};
use std::{path::Path, process::Command, time::Instant};
use tur_core::{Result, TurError, execution::invalid};
use tur_models::models::{
    DEFAULT_KV_LAYOUT, Execution, LoadOptions, WeightFormat, gpt_oss::GptOss,
};

fn numbers_after(text: &str, key: &str) -> u64 {
    text.lines()
        .filter(|line| line.contains(key))
        .filter_map(|line| {
            let digits: String = line
                .rsplit(['=', ':'])
                .next()?
                .chars()
                .filter(char::is_ascii_digit)
                .collect();
            digits.parse::<u64>().ok()
        })
        .sum()
}

/// Bytes the internal SSD has written since boot, and pages swapped out.
fn counters() -> Result<(u64, u64)> {
    let run = |program: &str, args: &[&str]| -> Result<String> {
        let output = Command::new(program).args(args).output()?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    };
    let storage = run("ioreg", &["-c", "IOBlockStorageDriver", "-r", "-w", "0"])?;
    let written = numbers_after(&storage, "\"Bytes (Write)\"");
    let swapped = numbers_after(&run("vm_stat", &[])?, "Swapouts");
    Ok((written, swapped))
}

fn load(dir: &Path, cache_mib: usize, device: &Device) -> Result<GptOss> {
    GptOss::load(
        dir,
        &LoadOptions {
            revision: "expert-measurement".into(),
            weights: WeightFormat::Q8_0.into(),
            device: device.clone(),
            execution: Execution::Fused,
            kv: DEFAULT_KV_LAYOUT,
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            expert_cache: Some(cache_mib << 20),
        },
    )
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn samples(dir: &Path, text: &Path) -> Result<Vec<Vec<u32>>> {
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let encoded = tokenizer
        .encode(std::fs::read_to_string(text)?.as_str(), false)
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let ids = encoded.get_ids();
    let ids = &ids[..ids.len().min(env("TUR_EXPERTS_TOKENS", 2048))];
    Ok(ids
        .chunks(env("TUR_EXPERTS_SAMPLE", 128))
        .map(<[u32]>::to_vec)
        .collect())
}

/// Log-probabilities at every fourth position of each sample, so a whole run's references
/// stay small enough to hold while the other checkpoint loads.
fn log_probs(model: &GptOss, sample: &[u32]) -> Result<Vec<Vec<f32>>> {
    let rows: Vec<usize> = (0..sample.len()).step_by(4).collect();
    let (_, logits) = model.trace(sample, &rows)?;
    Ok(candle_nn::ops::log_softmax(&logits, D::Minus1)?.to_vec2::<f32>()?)
}

/// Mean `KL(reference ‖ candidate)` in nats and the rate of equal top tokens.
fn compare(reference: &[Vec<Vec<f32>>], candidate: &[Vec<Vec<f32>>]) -> (f64, f64) {
    let top = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    let (mut kl, mut same, mut rows) = (0.0, 0usize, 0usize);
    for (r, c) in reference.iter().flatten().zip(candidate.iter().flatten()) {
        kl += r
            .iter()
            .zip(c)
            .map(|(p, q)| f64::from(p.exp()) * f64::from(p - q))
            .sum::<f64>();
        same += usize::from(top(r) == top(c));
        rows += 1;
    }
    (kl / rows.max(1) as f64, same as f64 / rows.max(1) as f64)
}

/// Expert projection at gpt-oss-20b's `gate_up` shape over random blocks: `rows` tokens
/// routed to four experts, in MXFP4 and in `Q8_0`, and the cost of one GPU round trip.
fn kernel_case() -> Result<()> {
    use candle_core::{Tensor, quantized::GgmlDType};
    use tur_metal::{kernels::Kernels, packed::PackedLinear};
    let device = Device::new_metal(0)?;
    let k = Kernels::new(&device)?;
    let (experts, rows, cols) = (32, 5760, 2880);
    let blocks_count = experts * rows * (cols / 32);
    let mut state = 88_172_645_463_325_252u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let blocks: Vec<u8> = (0..blocks_count * 16).map(|_| next() as u8).collect();
    let scales: Vec<u8> = (0..blocks_count)
        .map(|_| 118 + (next() % 6) as u8)
        .collect();
    let shape = (experts * rows, cols);
    let mxfp4 = PackedLinear::mxfp4(&blocks, &scales, shape, &device)?;
    let values = tur_metal::quantize::mxfp4_decode(&blocks, &scales)?;
    let q8 = PackedLinear::quantized(
        &Tensor::from_vec(values, shape, &Device::Cpu)?,
        GgmlDType::Q8_0,
        &device,
    )?;
    let mut results = Vec::new();
    for tokens in [1usize, 4, 16, 64, 256] {
        let input: Vec<f32> = (0..tokens * cols)
            .map(|_| (next() % 100) as f32 * 0.01)
            .collect();
        let x = Tensor::from_vec(input, (tokens, cols), &device)?;
        let scores: Vec<f32> = (0..tokens * experts)
            .map(|_| (next() % 1000) as f32)
            .collect();
        let logits = Tensor::from_vec(scores, (tokens, experts), &device)?;
        let time = |layer: &PackedLinear| -> Result<f64> {
            let mut runs = Vec::new();
            for _ in 0..15 {
                let routes = k.route(&logits, 4)?;
                device.synchronize()?;
                let started = Instant::now();
                drop(k.expert_linear_routed(layer, &x, &routes)?);
                device.synchronize()?;
                runs.push(started.elapsed().as_secs_f64() * 1e3);
            }
            runs.sort_by(f64::total_cmp);
            Ok(runs[runs.len() / 2])
        };
        results.push(serde_json::json!({
            "tokens": tokens, "mxfp4_ms": time(&mxfp4)?, "q8_0_ms": time(&q8)?,
        }));
    }
    let probe = Tensor::zeros((1, 32), DType::F32, &device)?;
    let mut trips = Vec::new();
    for _ in 0..200 {
        let started = Instant::now();
        drop((&probe + 1.0)?.to_vec2::<f32>()?);
        trips.push(started.elapsed().as_secs_f64() * 1e6);
    }
    trips.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({ "projection": results, "round_trip_us": trips[100] })
    );
    Ok(())
}

pub(crate) fn run(args: &[String]) -> Result<()> {
    if args.first().is_some_and(|case| case == "kernels") {
        return kernel_case();
    }
    let [dir, cache, text, rest @ ..] = args else {
        return Err(TurError::Other(
            "usage: perf -- experts <dir> <cache MiB> <text file> [reference dir]".into(),
        ));
    };
    let (dir, text) = (Path::new(dir), Path::new(text));
    let cache: usize = cache.parse().map_err(|_| invalid("cache must be MiB"))?;
    let device = Device::new_metal(0)?;
    let samples = samples(dir, text)?;
    let reference = rest
        .first()
        .map(|path| -> Result<Vec<_>> {
            // Its own device, so its buffers are gone before the candidate loads.
            let device = Device::new_metal(0)?;
            let model = load(Path::new(path), cache, &device)?;
            samples.iter().map(|s| log_probs(&model, s)).collect()
        })
        .transpose()?;
    let loaded = Instant::now();
    let model = load(dir, cache, &device)?;
    device.synchronize()?;
    let load_seconds = loaded.elapsed().as_secs_f64();
    let before = counters()?;
    let started = Instant::now();
    let mut outputs = Vec::with_capacity(samples.len());
    for sample in &samples {
        outputs.push(log_probs(&model, sample)?);
        device.synchronize()?;
    }
    let seconds = started.elapsed().as_secs_f64();
    let after = counters()?;
    let stats = model
        .expert_stats()
        .ok_or_else(|| invalid("missing expert stats"))?;
    // One-token passes over cached experts: what a decode costs with no reads at all.
    let warm_token = [samples
        .first()
        .and_then(|sample| sample.first())
        .copied()
        .ok_or_else(|| invalid("no token to measure"))?];
    for _ in 0..3 {
        drop(model.trace(&warm_token, &[0])?);
        device.synchronize()?;
    }
    let mut warm: Vec<f64> = (0..20)
        .map(|_| -> Result<f64> {
            let started = Instant::now();
            drop(model.trace(&warm_token, &[0])?);
            device.synchronize()?;
            Ok(started.elapsed().as_secs_f64() * 1e3)
        })
        .collect::<Result<_>>()?;
    warm.sort_by(f64::total_cmp);
    let tokens: usize = samples.iter().map(Vec::len).sum();
    let quality = reference.as_ref().map(|r| compare(r, &outputs));
    let touched = stats.hits + stats.misses;
    let document = serde_json::json!({
        "dir": dir,
        "cache_mib": cache,
        "weight_bytes": model.weight_bytes(),
        "load_seconds": load_seconds,
        "tokens": tokens,
        "samples": samples.len(),
        "seconds": seconds,
        "tokens_per_second": tokens as f64 / seconds,
        "cache_hits": stats.hits,
        "cache_misses": stats.misses,
        "hit_rate": stats.hits as f64 / touched.max(1) as f64,
        "evictions": stats.evictions,
        "gib_read": stats.bytes_read as f64 / (1u64 << 30) as f64,
        "ssd_bytes_read_per_token": stats.bytes_read as f64 / tokens.max(1) as f64,
        "wait_seconds": stats.wait.as_secs_f64(),
        "ssd_bytes_written_system": after.0.saturating_sub(before.0),
        "swapouts_system": after.1.saturating_sub(before.1),
        "warm_pass_ms_median": warm[warm.len() / 2],
        "warm_pass_ms_min": warm[0],
        "kl_nats": quality.map(|q| q.0),
        "top1_agreement": quality.map(|q| q.1),
    });
    println!("{}", serde_json::to_string_pretty(&document)?);
    Ok(())
}
