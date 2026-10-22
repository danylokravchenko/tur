#![forbid(unsafe_code)]
//! Qwen3.5 model-level benchmark on a real checkpoint: fused Metal kernels against the F32
//! reference on identical weights, for prefill and batched decode, plus engine overhead.
//! Prints one JSON document. Every timed region ends at device completion.
//!
//! `TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench qwen3_5 > out.json`
//! `TUR_BENCH_FORMATS=q8_0,q4k` and `TUR_BENCH_PREFILL=128,2048` narrow the matrix.

use candle_core::Device;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tur_core::{
    Result, TurError,
    execution::{PackedExecution, SequenceDescriptor, invalid},
};
use tur_engine::hybrid::{
    Boundaries, HybridEngine, HybridModel, PackedForward, SamplingOptions, SequenceForward,
};
use tur_metal::state::ModelState;
use tur_models::models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat};

const DECODE_CONTEXT: usize = 128;
const DECODE_STEPS: usize = 12;
const WARMUP: usize = 2;

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(str::to_owned)
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

/// One forward pass of `sequences` with the last row of each as logits, synchronized.
fn pass(model: &Qwen35, sequences: &[(&ModelState, &[u32])]) -> Result<f64> {
    let reads: Vec<_> = sequences.iter().map(|(s, _)| s.bind_layers()).collect();
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
    let forwards: Vec<SequenceForward<'_>> = reads
        .iter()
        .zip(&ranges)
        .zip(&logits)
        .map(|((state, range), logits)| SequenceForward {
            tokens: range.clone(),
            state,
            logits,
            boundaries: Boundaries::Final,
        })
        .collect();
    model.device().synchronize()?;
    let start = Instant::now();
    for result in model.forward(&PackedForward {
        tokens: &packed,
        positions: &positions,
        sequences: &forwards,
    })? {
        result?;
    }
    model.device().synchronize()?;
    Ok(start.elapsed().as_secs_f64())
}

/// Committed state after prefilling `prompt` in one pass.
fn prefilled(model: &Qwen35, prompt: &[u32]) -> Result<ModelState> {
    let state = ModelState::allocate(Arc::new(model.description()), model.device())?;
    let reads = state.bind_layers();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let step = model
        .forward(&PackedForward {
            tokens: prompt,
            positions: &positions,
            sequences: &[SequenceForward {
                tokens: 0..prompt.len(),
                state: &reads,
                logits: &[],
                boundaries: Boundaries::Final,
            }],
        })?
        .pop()
        .ok_or_else(|| invalid("missing step"))??;
    let staged = state.stage(step.layers, prompt.len(), Boundaries::Final)?;
    state.apply(&staged, prompt.len())
}

fn prefill_case(model: &Qwen35, length: usize) -> Result<Value> {
    let empty = ModelState::allocate(Arc::new(model.description()), model.device())?;
    let prompt = tokens(length, 1);
    let iterations = if length > 1024 { 2 } else { 4 };
    for _ in 0..WARMUP.min(iterations) {
        pass(model, &[(&empty, &prompt)])?;
    }
    let samples = (0..iterations)
        .map(|_| pass(model, &[(&empty, &prompt)]))
        .collect::<Result<Vec<_>>>()?;
    let seconds = median(samples.clone());
    Ok(json!({
        "prompt_tokens": length, "median_seconds": seconds,
        "tokens_per_second": length as f64 / seconds, "samples_seconds": samples,
    }))
}

fn decode_case(model: &Qwen35, batch: usize) -> Result<Value> {
    let states = (0..batch)
        .map(|i| prefilled(model, &tokens(DECODE_CONTEXT, i + 2)))
        .collect::<Result<Vec<_>>>()?;
    let next: Vec<Vec<u32>> = (0..batch).map(|i| tokens(1, i + 50)).collect();
    let sequences: Vec<(&ModelState, &[u32])> = states
        .iter()
        .zip(&next)
        .map(|(s, t)| (s, t.as_slice()))
        .collect();
    for _ in 0..WARMUP {
        pass(model, &sequences)?;
    }
    let samples = (0..DECODE_STEPS)
        .map(|_| pass(model, &sequences))
        .collect::<Result<Vec<_>>>()?;
    let seconds = median(samples.clone());
    Ok(json!({
        "sequences": batch, "context": DECODE_CONTEXT, "median_step_seconds": seconds,
        "tokens_per_second": batch as f64 / seconds, "samples_seconds": samples,
    }))
}

/// Per-token latency through `HybridEngine` (forward, fence, sampling, state publish).
fn engine_case(model: Arc<Qwen35>, generated: usize) -> Result<Value> {
    let device = model.device().clone();
    let mut engine = HybridEngine::new(model, device)?;
    let handle = engine.allocate(0, SamplingOptions::default(), None)?;
    let mut next = tokens(DECODE_CONTEXT, 99);
    let mut samples = Vec::new();
    for step in 0..generated + 1 {
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

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var_os("TUR_QWEN35_0_8B")
            .ok_or_else(|| TurError::Other("set TUR_QWEN35_0_8B".into()))?,
    );
    let device = Device::new_metal(0)?;
    let prefill: Vec<usize> = env_list("TUR_BENCH_PREFILL", "128,2048")
        .iter()
        .map(|v| v.parse().map_err(|_| invalid("TUR_BENCH_PREFILL")))
        .collect::<Result<_>>()?;
    let mut runs = Vec::new();
    for name in env_list("TUR_BENCH_FORMATS", "f32,q8_0,q4k") {
        let format: WeightFormat = name.parse().map_err(TurError::Other)?;
        let load_start = Instant::now();
        let mut model = Qwen35::load(
            &dir,
            &LoadOptions {
                revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
                format,
                device: device.clone(),
                execution: Execution::Fused,
            },
        )?;
        let load_seconds = load_start.elapsed().as_secs_f64();
        for execution in [Execution::Fused, Execution::Reference] {
            model = model.with_execution(execution)?;
            let prefill = prefill
                .iter()
                .map(|&length| prefill_case(&model, length))
                .collect::<Result<Vec<_>>>()?;
            let decode = [1, 2, 4, 8]
                .into_iter()
                .map(|batch| decode_case(&model, batch))
                .collect::<Result<Vec<_>>>()?;
            runs.push(json!({
                "format": name, "execution": format!("{execution:?}"),
                "load_seconds": load_seconds, "projection_bytes": model.projection_bytes(),
                "prefill": prefill, "decode": decode,
            }));
        }
        model = model.with_execution(Execution::Fused)?;
        let model = Arc::new(model);
        runs.push(json!({
            "format": name, "execution": "Fused", "engine": engine_case(model, 32)?,
        }));
    }
    let document = json!({
        "schema_version": 1,
        "checkpoint": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
        "workload": "synthetic token ids; one prefill pass per prompt length; decode steps re-run one token over committed context without publishing",
        "decode_context": DECODE_CONTEXT, "warmup": WARMUP, "decode_steps": DECODE_STEPS,
        "activations": "F32",
        "timing": "host wall time from a synchronized device to Device::synchronize after the pass",
        "runs": runs,
    });
    let text = serde_json::to_string_pretty(&document)?;
    std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes())?;
    Ok(())
}
