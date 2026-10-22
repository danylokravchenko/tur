#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmarks emit JSON artifacts and timing summaries"
)]
#![forbid(unsafe_code)]
//! Qwen3.5 model-level benchmark on a real checkpoint: fused Metal passes over persistent
//! state pools for prefill and batched decode at several contexts, per weight and K/V
//! format, plus engine overhead. Prints one JSON document. Every timed region ends at
//! device completion.
//!
//! `TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench qwen3_5 > out.json`
//! narrows with `TUR_BENCH_FORMATS=q8_0,q4k`, `TUR_BENCH_KV=f32,q8`, `TUR_BENCH_PAGES=16,64`,
//! `TUR_BENCH_PREFILL=128,2048` and `TUR_BENCH_CONTEXTS=2048,8192`. `TUR_BENCH_REFERENCE=1`
//! adds the F32 reference execution with F32 K/V.

use candle_core::Device;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tur_core::{
    Result, TurError,
    execution::{KvFormat, KvLayout, PackedExecution, SequenceDescriptor, invalid},
};
use tur_engine::hybrid::{
    Boundaries, HybridEngine, HybridModel, ModelState, PackedForward, PoolConfig, Reservation,
    SamplingOptions, SequenceForward, StatePool,
};
use tur_models::models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat};

const SHORT_CONTEXT: usize = 128;
const DECODE_STEPS: usize = 12;
const WARMUP: usize = 2;

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(str::to_owned)
        .collect()
}

fn env_sizes(name: &str, default: &str) -> Result<Vec<usize>> {
    env_list(name, default)
        .iter()
        .map(|v| v.parse().map_err(|_| invalid(name.to_owned())))
        .collect()
}

fn tokens(len: usize, salt: usize) -> Vec<u32> {
    // Ordinary vocabulary ids well below the special-token range.
    (0..len)
        .map(|i| ((i * 7919 + salt * 104_729) % 150_000 + 1000) as u32)
        .collect()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// A pool holding `contexts` sequences plus one pass of new tokens each.
fn pool(model: &Qwen35, contexts: &[usize], extra: usize) -> Result<StatePool> {
    let description = model.description();
    let pages = contexts
        .iter()
        .map(|c| description.pages_for(c + extra) + 1)
        .sum();
    StatePool::new(
        Arc::new(description),
        model.device(),
        PoolConfig {
            pages,
            recurrent_slots: 2 * contexts.len() + 2,
        },
    )
}

/// One forward pass of `sequences` with the last row of each as logits: reserve pages and
/// slots, encode, wait. Nothing is committed, so the pass can repeat on the same states.
fn pass(model: &Qwen35, pool: &StatePool, sequences: &[(&ModelState, &[u32])]) -> Result<f64> {
    let mut packed = Vec::new();
    let mut positions = Vec::new();
    let mut ranges = Vec::new();
    for (state, new) in sequences {
        let start = packed.len();
        packed.extend_from_slice(new);
        positions.extend(state.position()..state.position() + new.len());
        ranges.push(start..packed.len());
    }
    let logits: Vec<[usize; 1]> = ranges.iter().map(|r| [r.end - 1]).collect();
    model.device().synchronize()?;
    let start = Instant::now();
    let mut reservations = sequences
        .iter()
        .map(|(state, new)| state.reserve(new.len(), Boundaries::Final))
        .collect::<Result<Vec<_>>>()?;
    for reservation in &mut reservations {
        reservation.prepare()?;
    }
    let bindings: Vec<_> = reservations.iter().map(Reservation::binding).collect();
    let forwards: Vec<SequenceForward<'_>> = bindings
        .iter()
        .zip(&ranges)
        .zip(&logits)
        .map(|((state, range), logits)| SequenceForward {
            tokens: range.clone(),
            state,
            logits,
        })
        .collect();
    for result in model.forward(&PackedForward {
        tokens: &packed,
        positions: &positions,
        sequences: &forwards,
        pool,
    })? {
        result?;
    }
    model.device().synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    drop(reservations);
    pool.reclaim();
    Ok(seconds)
}

/// Committed state after prefilling `prompt` in chunks of 2048 tokens.
fn prefilled(model: &Qwen35, pool: &StatePool, prompt: &[u32]) -> Result<ModelState> {
    let mut state = ModelState::new(pool);
    for piece in prompt.chunks(2048) {
        let mut reservation = state.reserve(piece.len(), Boundaries::Final)?;
        reservation.prepare()?;
        let binding = reservation.binding();
        let positions: Vec<usize> = (state.position()..state.position() + piece.len()).collect();
        model
            .forward(&PackedForward {
                tokens: piece,
                positions: &positions,
                sequences: &[SequenceForward {
                    tokens: 0..piece.len(),
                    state: &binding,
                    logits: &[],
                }],
                pool,
            })?
            .pop()
            .ok_or_else(|| invalid("missing step"))??;
        state = reservation.commit(piece.len())?;
        drop(reservation);
        model.device().synchronize()?;
        pool.reclaim();
    }
    Ok(state)
}

fn prefill_case(model: &Qwen35, length: usize) -> Result<Value> {
    let pool = pool(model, &[0], length)?;
    let empty = ModelState::new(&pool);
    let prompt = tokens(length, 1);
    let iterations = if length > 1024 { 2 } else { 4 };
    for _ in 0..WARMUP.min(iterations) {
        pass(model, &pool, &[(&empty, &prompt)])?;
    }
    let samples = (0..iterations)
        .map(|_| pass(model, &pool, &[(&empty, &prompt)]))
        .collect::<Result<Vec<_>>>()?;
    let seconds = median(samples.clone());
    Ok(json!({
        "prompt_tokens": length, "median_seconds": seconds,
        "tokens_per_second": length as f64 / seconds, "samples_seconds": samples,
    }))
}

fn decode_case(model: &Qwen35, batch: usize, context: usize) -> Result<Value> {
    let pool = pool(model, &vec![context; batch], 1)?;
    let states = (0..batch)
        .map(|i| prefilled(model, &pool, &tokens(context, i + 2)))
        .collect::<Result<Vec<_>>>()?;
    let next: Vec<Vec<u32>> = (0..batch).map(|i| tokens(1, i + 50)).collect();
    let sequences: Vec<(&ModelState, &[u32])> = states
        .iter()
        .zip(&next)
        .map(|(s, t)| (s, t.as_slice()))
        .collect();
    for _ in 0..WARMUP {
        pass(model, &pool, &sequences)?;
    }
    let samples = (0..DECODE_STEPS)
        .map(|_| pass(model, &pool, &sequences))
        .collect::<Result<Vec<_>>>()?;
    let seconds = median(samples.clone());
    let usage = pool.usage()?;
    Ok(json!({
        "sequences": batch, "context": context, "median_step_seconds": seconds,
        "tokens_per_second": batch as f64 / seconds, "samples_seconds": samples,
        "state_leased_bytes": usage.leased_bytes(),
    }))
}

/// Per-token latency through `HybridEngine` (reservation, forward, fence, sampling,
/// publication).
fn engine_case(model: Arc<Qwen35>, generated: usize) -> Result<Value> {
    let device = model.device().clone();
    let config = PoolConfig {
        pages: model.description().pages_for(SHORT_CONTEXT + generated) + 1,
        recurrent_slots: 2,
    };
    let mut engine = HybridEngine::new(model, device, config)?;
    let handle = engine.allocate(0, SamplingOptions::default(), None)?;
    let mut next = tokens(SHORT_CONTEXT, 99);
    let mut samples = Vec::new();
    for step in 0..=generated {
        let meta = engine.metadata(handle.request)?;
        let n = next.len();
        let start = Instant::now();
        engine.submit(PackedExecution {
            positions: (meta.position..meta.position + n).collect(),
            tokens: std::mem::take(&mut next),
            sequences: vec![SequenceDescriptor {
                state: meta.handle,
                tokens: 0..n,
                outputs: vec![n - 1],
                speculative: false,
            }],
        })?;
        let token = engine
            .finish()?
            .pop()
            .ok_or_else(|| invalid("missing result"))?
            .outcome?
            .outputs
            .first()
            .map(|o| o.token)
            .ok_or_else(|| invalid("no output"))?;
        // The first step is the prompt prefill.
        if step > 0 {
            samples.push(start.elapsed().as_secs_f64());
        }
        next = vec![token];
    }
    let seconds = median(samples.clone());
    Ok(json!({
        "generated_tokens": generated, "median_token_seconds": seconds,
        "tokens_per_second": 1.0 / seconds, "samples_seconds": samples,
    }))
}

fn run(model: &Qwen35, prefill: &[usize], contexts: &[usize]) -> Result<Value> {
    let prefill = prefill
        .iter()
        .map(|&length| prefill_case(model, length))
        .collect::<Result<Vec<_>>>()?;
    let mut decode = [1, 2, 4, 8, 16]
        .into_iter()
        .map(|batch| decode_case(model, batch, SHORT_CONTEXT))
        .collect::<Result<Vec<_>>>()?;
    for &context in contexts {
        for batch in [1, 4] {
            decode.push(decode_case(model, batch, context)?);
        }
    }
    let description = model.description();
    Ok(json!({
        "kv_format": description.kv.format.to_string(), "page_tokens": description.kv.page_tokens,
        "kv_token_bytes": description.kv_token_bytes()?,
        "recurrent_slot_bytes": description.recurrent_slot_bytes()?,
        "projection_bytes": model.projection_bytes(), "weight_bytes": model.weight_bytes(),
        "prefill": prefill, "decode": decode,
    }))
}

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var_os("TUR_QWEN35_0_8B")
            .ok_or_else(|| TurError::Other("set TUR_QWEN35_0_8B".into()))?,
    );
    let device = Device::new_metal(0)?;
    let prefill = env_sizes("TUR_BENCH_PREFILL", "128,2048")?;
    let contexts = env_sizes("TUR_BENCH_CONTEXTS", "2048,8192")?;
    let pages = env_sizes("TUR_BENCH_PAGES", "16")?;
    let reference = std::env::var("TUR_BENCH_REFERENCE").is_ok_and(|v| v == "1");
    let mut runs = Vec::new();
    for name in env_list("TUR_BENCH_FORMATS", "q8_0,q4k") {
        let format: WeightFormat = name.parse().map_err(TurError::Other)?;
        for kv in env_list("TUR_BENCH_KV", "f32,f16,q8") {
            let kv: KvFormat = kv.parse().map_err(TurError::Other)?;
            for &page in &pages {
                let load_start = Instant::now();
                let model = Qwen35::load(
                    &dir,
                    &LoadOptions {
                        revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
                        format,
                        device: device.clone(),
                        execution: Execution::Fused,
                        kv: KvLayout::new(page, kv),
                    },
                )?;
                let load_seconds = load_start.elapsed().as_secs_f64();
                let mut result = run(&model, &prefill, &contexts)?;
                result["format"] = json!(name);
                result["execution"] = json!("Fused");
                result["load_seconds"] = json!(load_seconds);
                runs.push(result);
                if reference && kv == KvFormat::F32 {
                    let model = model.with_execution(Execution::Reference)?;
                    let mut result = run(&model, &prefill, &[])?;
                    result["format"] = json!(name);
                    result["execution"] = json!("Reference");
                    runs.push(result);
                    let model = Arc::new(model.with_execution(Execution::Fused)?);
                    runs.push(json!({
                        "format": name, "execution": "Fused", "kv_format": kv.to_string(),
                        "page_tokens": page, "engine": engine_case(model, 32)?,
                    }));
                } else {
                    runs.push(json!({
                        "format": name, "execution": "Fused", "kv_format": kv.to_string(),
                        "page_tokens": page, "engine": engine_case(Arc::new(model), 32)?,
                    }));
                }
            }
        }
    }
    let document = json!({
        "schema_version": 2,
        "checkpoint": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
        "workload": "synthetic token ids; one prefill pass per prompt length; decode steps re-run one token over committed context without publishing",
        "short_context": SHORT_CONTEXT, "warmup": WARMUP, "decode_steps": DECODE_STEPS,
        "activations": "F32",
        "timing": "host wall time from a synchronized device through reservation, encoding and Device::synchronize",
        "runs": runs,
    });
    let text = serde_json::to_string_pretty(&document)?;
    std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes())?;
    Ok(())
}
