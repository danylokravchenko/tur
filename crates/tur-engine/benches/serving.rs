#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmarks emit JSON artifacts and timing summaries"
)]
#![forbid(unsafe_code)]
//! Serving benchmark on a real checkpoint: bursts of concurrent chat requests through
//! `InferencePipeline` on Metal, reporting aggregate throughput, time to first token,
//! inter-token latency and pass counts. Every pass ends at device completion, so a step's
//! wall time includes its GPU work. Prints one JSON document.
//!
//! `TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench serving > out.json`
//! narrows with `TUR_BENCH_CONCURRENCY=1,8` and `TUR_BENCH_OUTPUT=64`.

use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, time::Instant};
use tokenizers::Tokenizer;
use tur_core::{Result, TurError, execution::invalid};
use tur_engine::backend::{
    factory::{ModelFactory, ModelSource},
    pipeline::{GenerationRequest, InferencePipeline, PipelineConfig},
};
use tur_models::models::qwen3_5::{DEFAULT_KV_LAYOUT, WeightFormat};

const CHUNK: usize = 256;

fn env_sizes(name: &str, default: &str) -> Result<Vec<usize>> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|v| v.parse().map_err(|_| invalid(name.to_owned())))
        .collect()
}

fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// A prompt of at least `tokens` tokens, distinct per `salt`.
fn prompt(tokenizer: &Tokenizer, tokens: usize, salt: usize) -> Result<String> {
    let words = [
        "river", "lantern", "copper", "meadow", "signal", "harbor", "violet", "engine", "quartz",
        "summit", "orchard", "beacon", "glacier", "thunder", "saffron", "compass",
    ];
    let mut text = format!("Request {salt}. Continue the list:");
    let mut i = salt;
    loop {
        let count = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .len();
        if count >= tokens {
            return Ok(text);
        }
        for _ in 0..(tokens - count).div_ceil(2) {
            text.push(' ');
            text.push_str(words[i % words.len()]);
            i = i.wrapping_mul(31).wrapping_add(7);
        }
    }
}

struct Burst {
    prompts: Vec<String>,
    output: usize,
}

/// Submit every prompt at once and step until all finish.
fn burst(pipeline: &mut InferencePipeline, case: &Burst) -> Result<Value> {
    let start = Instant::now();
    let mut ids = HashMap::new();
    for (index, prompt) in case.prompts.iter().enumerate() {
        let mut request = GenerationRequest::new(prompt.clone(), case.output);
        request.seed = index as u64;
        // Greedy, so both sides of a comparison generate the same lengths.
        request.sampling.temperature = Some(0.0);
        ids.insert(pipeline.submit_request(&request)?, index);
    }
    let mut emitted: Vec<Vec<f64>> = vec![Vec::new(); ids.len()];
    let (mut generated, mut prompt_tokens, mut passes) = (0, 0, 0);
    let mut pass_seconds = Vec::new();
    while pipeline.active_request_count() > 0 {
        let step = Instant::now();
        let output = pipeline.step_streaming()?;
        pass_seconds.push(step.elapsed().as_secs_f64());
        passes += 1;
        let now = start.elapsed().as_secs_f64();
        for (id, _) in &output.new_tokens {
            if let Some(&index) = ids.get(id) {
                emitted[index].push(now);
            }
        }
        for (_, result) in output.completed {
            let stats = result?;
            generated += stats.generated_tokens;
            prompt_tokens += stats.prompt_tokens;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let ttft: Vec<f64> = emitted.iter().filter_map(|t| t.first().copied()).collect();
    let itl: Vec<f64> = emitted
        .iter()
        .flat_map(|t| t.windows(2).map(|w| w[1] - w[0]))
        .collect();
    Ok(json!({
        "requests": ids.len(), "prompt_tokens": prompt_tokens, "generated_tokens": generated,
        "wall_seconds": wall, "output_tokens_per_second": generated as f64 / wall,
        "total_tokens_per_second": (generated + prompt_tokens) as f64 / wall,
        "ttft_seconds": {"p50": percentile(&ttft, 0.5), "p95": percentile(&ttft, 0.95), "max": percentile(&ttft, 1.0)},
        "itl_seconds": {"p50": percentile(&itl, 0.5), "p95": percentile(&itl, 0.95), "p99": percentile(&itl, 0.99)},
        "passes": passes, "pass_seconds": {"p50": percentile(&pass_seconds, 0.5), "p95": percentile(&pass_seconds, 0.95)},
    }))
}

fn main() -> Result<()> {
    let dir = std::env::var("TUR_QWEN35_0_8B")
        .map(PathBuf::from)
        .map_err(|_| invalid("set TUR_QWEN35_0_8B to a Qwen3.5-0.8B snapshot"))?;
    let concurrency = env_sizes("TUR_BENCH_CONCURRENCY", "1,2,4,8,16")?;
    let output = env_sizes("TUR_BENCH_OUTPUT", "64")?
        .first()
        .copied()
        .unwrap_or(64);
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let factory = ModelFactory {
        source: ModelSource::LocalPath(dir.clone()),
        revision: "bench".into(),
        format: WeightFormat::Q8_0,
        device: candle_core::Device::new_metal(0)?,
        kv: DEFAULT_KV_LAYOUT,
    };
    let max = concurrency.iter().copied().max().unwrap_or(1);
    let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::new(max, CHUNK))?;
    // Warm pipelines and kernels.
    let warm = Burst {
        prompts: vec![prompt(&tokenizer, 64, 99)?],
        output: 8,
    };
    burst(&mut pipeline, &warm)?;
    let mut runs = Vec::new();
    for &requests in &concurrency {
        let short = Burst {
            prompts: (0..requests)
                .map(|i| prompt(&tokenizer, 128, i))
                .collect::<Result<_>>()?,
            output,
        };
        let result = burst(&mut pipeline, &short)?;
        eprintln!(
            "decode-heavy c={requests}: {:.1} output tok/s, ITL p50 {:.1} ms",
            result["output_tokens_per_second"].as_f64().unwrap_or(0.0),
            result["itl_seconds"]["p50"].as_f64().unwrap_or(0.0) * 1e3
        );
        runs.push(json!({"workload": "decode_heavy", "prompt": 128, "concurrency": requests, "result": result}));
    }
    // Mixed: half the requests carry 2K-token prompts, arriving with short ones.
    let mixed = Burst {
        prompts: (0..max)
            .map(|i| prompt(&tokenizer, if i % 2 == 0 { 128 } else { 2048 }, i))
            .collect::<Result<_>>()?,
        output,
    };
    let result = burst(&mut pipeline, &mixed)?;
    eprintln!(
        "mixed c={max}: {:.1} output tok/s, TTFT p95 {:.0} ms, ITL p95 {:.1} ms",
        result["output_tokens_per_second"].as_f64().unwrap_or(0.0),
        result["ttft_seconds"]["p95"].as_f64().unwrap_or(0.0) * 1e3,
        result["itl_seconds"]["p95"].as_f64().unwrap_or(0.0) * 1e3
    );
    runs.push(
        json!({"workload": "mixed", "prompt": [128, 2048], "concurrency": max, "result": result}),
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "benchmark": "serving", "checkpoint": dir, "weights": "q8_0",
            "kv": format!("{:?}", DEFAULT_KV_LAYOUT), "prefill_chunk": CHUNK,
            "output_tokens": output, "workload": "closed-loop bursts submitted at once; greedy sampling",
            "runs": runs,
        }))?
    );
    Ok(())
}
