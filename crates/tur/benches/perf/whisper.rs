//! Whisper's encoder operations at their real shapes; with checkpoints, whole windows and
//! transcriptions, and text passes while a Gemma clip encodes alongside them. Prints one
//! JSON line per measurement; timing covers completed GPU work.
//!
//! `cargo bench --offline -p tur --bench perf -- whisper`
//! `TUR_AUDIO_SPEECH=<wav>` with `TUR_WHISPER=<dir>` and `TUR_GEMMA4_E2B=<dir>` add the
//! checkpoint measurements.

use candle_core::{DType, Device, Tensor, quantized::GgmlDType};
use serde_json::json;
use std::{path::Path, sync::Arc, time::Instant};
use tur::{
    InferencePipeline, ModelFactory, ModelSource,
    backend::pipeline::{GenerationRequest, PipelineConfig},
};
use tur_core::{
    Result, TurError,
    execution::{KvFormat, KvLayout},
};
use tur_metal::{
    kernels::Kernels,
    packed::PackedLinear,
    pool::{AttentionPool, KvPool},
};
use tur_models::{
    audio::Audio,
    models::{
        Execution, LoadOptions, WeightFormat,
        gemma4::Gemma4,
        whisper::{TranscribeOptions, Transcriber, Whisper},
    },
};

const REPS: usize = 7;
const TOKENS: usize = 1500;
const HEADS: usize = 20;
const DIM: usize = 64;

fn random(device: &Device, shape: (usize, usize), scale: f64) -> Result<Tensor> {
    Ok((Tensor::randn(0f32, 1.0, shape, device)? * scale)?)
}

/// Median seconds of `op` to device completion, after a warmup that fills the pool.
fn seconds(device: &Device, mut op: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    drop(op()?);
    device.synchronize()?;
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        drop(op()?);
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[REPS / 2])
}

fn projections(device: &Device) -> Result<()> {
    for (out, input) in [(1280, 1280), (5120, 1280), (1280, 5120)] {
        let weight = random(&Device::Cpu, (out, input), 0.02)?;
        let x = random(device, (TOKENS, input), 1.0)?;
        let flops = 2.0 * (TOKENS * out * input) as f64;
        let dense = PackedLinear::dense(&weight, device)?;
        let packed = PackedLinear::quantized(&weight, GgmlDType::Q8_0, device)?;
        for (name, layer) in [("f32", &dense), ("q8_0", &packed)] {
            let time = seconds(device, || layer.forward(&x))?;
            println!(
                "{}",
                json!({"op": "linear", "out": out, "in": input, "format": name, "path": "candle",
                       "seconds": time, "tflops": flops / time / 1e12})
            );
        }
    }
    Ok(())
}

fn attention(device: &Device, kernels: &Kernels) -> Result<()> {
    let q = random(device, (TOKENS, HEADS * DIM), 1.0)?;
    let k = random(device, (TOKENS, HEADS * DIM), 1.0)?;
    let v = random(device, (TOKENS, HEADS * DIM), 1.0)?;
    let flops = 4.0 * (HEADS * TOKENS * TOKENS * DIM) as f64;
    let scale = (DIM as f64).sqrt().recip();
    let split = |x: &Tensor| -> Result<Tensor> {
        Ok(x.reshape((TOKENS, HEADS, DIM))?
            .transpose(0, 1)?
            .contiguous()?)
    };
    let candle = seconds(device, || {
        let (qh, kh, vh) = (split(&q)?, split(&k)?, split(&v)?);
        let mut groups = Vec::new();
        for start in (0..HEADS).step_by(4) {
            let n = 4.min(HEADS - start);
            let scores = (qh
                .narrow(0, start, n)?
                .matmul(&kh.narrow(0, start, n)?.transpose(1, 2)?.contiguous()?)?
                * scale)?;
            let exp = scores.exp()?;
            let probs = exp.broadcast_div(&exp.sum_keepdim(candle_core::D::Minus1)?)?;
            groups.push(probs.matmul(&vh.narrow(0, start, n)?)?);
        }
        Ok(Tensor::cat(&groups, 0)?)
    })?;
    println!(
        "{}",
        json!({"op": "attention", "path": "candle", "seconds": candle, "tflops": flops / candle / 1e12})
    );
    let pool = AttentionPool {
        key: KvPool {
            data: k.reshape((TOKENS, HEADS, DIM))?,
            scales: None,
        },
        value: KvPool {
            data: v.reshape((TOKENS, HEADS, DIM))?,
            scales: None,
        },
        format: KvFormat::F32,
        page_tokens: TOKENS,
        kv_heads: HEADS,
        head_dim: DIM,
    };
    let visible = vec![u32::try_from(TOKENS).unwrap_or(0); TOKENS];
    let paged = kernels.paged_kv(TOKENS, TOKENS, &[0], &vec![0; TOKENS], &visible)?;
    let query = q.reshape((TOKENS, HEADS, DIM))?;
    let tiled = seconds(device, || {
        kernels.attention_ungated(&query, &pool, &paged, None, scale)
    })?;
    println!(
        "{}",
        json!({"op": "attention", "path": "tiled", "is_tiled": kernels.attention_tiled_ungated(&paged, HEADS, DIM),
               "seconds": tiled, "tflops": flops / tiled / 1e12, "speedup": candle / tiled})
    );
    Ok(())
}

pub(crate) fn run() -> Result<()> {
    let device = Device::new_metal(0)?;
    let kernels = Kernels::new(&device)?;
    projections(&device)?;
    attention(&device, &kernels)?;
    let Some(wav) = std::env::var_os("TUR_AUDIO_SPEECH") else {
        return Ok(());
    };
    let wav = Path::new(&wav);
    if let Some(dir) = std::env::var_os("TUR_WHISPER") {
        checkpoint(&device, Path::new(&dir), wav)?;
    }
    if let Some(dir) = std::env::var_os("TUR_GEMMA4_E2B") {
        gemma(&device, Path::new(&dir), wav)?;
        gemma_contention(Path::new(&dir), wav)?;
    }
    Ok(())
}

/// How long encoding a chat clip holds the calling thread, against its completed GPU work.
fn gemma(device: &Device, dir: &Path, wav: &Path) -> Result<()> {
    let model = Gemma4::load(
        dir,
        &LoadOptions {
            revision: "google/gemma-4-E2B-it".into(),
            weights: WeightFormat::Q8_0.into(),
            device: device.clone(),
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F16),
            recurrent: DType::F32,
            mtp: None,
            audio: true,
            expert_cache: None,
        },
    )?;
    let clip = Audio::from_wav(&std::fs::read(wav)?)?;
    let limit = clip.samples().len().min(480_000);
    for samples in [80_000.min(limit), limit] {
        let part = Audio::from_samples(&clip.samples()[..samples], 16_000)?;
        let encode = seconds(device, || model.encode_audio(&part))?;
        println!(
            "{}",
            json!({"op": "gemma_clip", "audio_seconds": part.duration().as_secs_f64(),
                   "encode_seconds": encode})
        );
    }
    Ok(())
}

/// Text passes while a 30 s clip encodes on another thread, against none encoding: the
/// slowest pass bounds how long a stream can pause for someone else's clip.
fn gemma_contention(dir: &Path, wav: &Path) -> Result<()> {
    let factory = ModelFactory {
        source: ModelSource::LocalPath(dir.to_owned()),
        revision: "google/gemma-4-E2B-it".into(),
        format: Some(WeightFormat::Q8_0),
        device: Device::new_metal(0)?,
        kv: KvLayout::new(16, KvFormat::F16),
        recurrent: DType::F32,
        audio: true,
        expert_cache: None,
    };
    let clip = Audio::from_wav(&std::fs::read(wav)?)?;
    let clip = Audio::from_samples(&clip.samples()[..clip.samples().len().min(480_000)], 16_000)?;
    let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::new(1, 256))?;
    let model = pipeline
        .audio_model()
        .ok_or_else(|| TurError::Other("no audio encoder".into()))?;
    for concurrent in [false, true] {
        let mut request = GenerationRequest::new("Write a long story about a river.".into(), 400);
        request.sampling.temperature = Some(0.0);
        let id = pipeline.submit_request(&request)?;
        let (model, clip) = (Arc::clone(&model), clip.clone());
        let encoder = concurrent.then(|| {
            std::thread::spawn(move || -> Result<f64> {
                let start = Instant::now();
                model.encode_audio(&clip)?;
                Ok(start.elapsed().as_secs_f64())
            })
        });
        let (mut passes, mut last) = (Vec::new(), Instant::now());
        while pipeline.active_request_count() > 0 && passes.len() < 120 {
            pipeline.begin_step()?;
            pipeline.finish_step()?;
            passes.push(last.elapsed().as_secs_f64());
            last = Instant::now();
        }
        if pipeline.active_request_count() > 0 {
            pipeline.cancel(id)?;
        }
        let encode = encoder
            .map(|thread| {
                thread
                    .join()
                    .map_err(|_| TurError::Other("encoder".into()))?
            })
            .transpose()?;
        passes.sort_by(f64::total_cmp);
        println!(
            "{}",
            json!({"op": "gemma_contention", "concurrent_clip": concurrent, "encode_seconds": encode,
                   "median_pass": passes[passes.len() / 2], "slowest_pass": passes[passes.len() - 1]})
        );
    }
    Ok(())
}

/// One decoder over a fixed token stream: building it, the host time to encode a step and
/// the whole step with its logits read back, at short and long decoded lengths.
fn decoding(device: &Device, model: &Whisper, features: &Tensor) -> Result<()> {
    let encoded = model.encode(features)?;
    device.synchronize()?;
    let started = Instant::now();
    let mut decoder = model.decoder(&encoded)?;
    device.synchronize()?;
    let setup = started.elapsed().as_secs_f64();
    let last = |logits: &Tensor| -> Result<Vec<f32>> {
        Ok(logits.get(logits.dim(0)? - 1)?.to_vec1::<f32>()?)
    };
    drop(last(&decoder.step(&[50258, 50259, 50360, 50364])?)?);
    let (mut encoding, mut whole) = (Vec::new(), Vec::new());
    for i in 0..300u32 {
        let started = Instant::now();
        let logits = decoder.step(&[440 + i])?;
        encoding.push(started.elapsed().as_secs_f64());
        drop(last(&logits)?);
        whole.push(started.elapsed().as_secs_f64());
    }
    let median = |samples: &[f64]| {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    println!(
        "{}",
        json!({"op": "decoder", "setup_seconds": setup,
               "step_at_30": median(&whole[..60]), "step_at_270": median(&whole[240..]),
               "encode_at_30": median(&encoding[..60]), "encode_at_270": median(&encoding[240..])})
    );
    Ok(())
}

/// One window's features, encoder and decoder on the real checkpoint, then the whole clip.
fn checkpoint(device: &Device, dir: &Path, wav: &Path) -> Result<()> {
    let clip = Audio::from_wav(&std::fs::read(wav)?)?;
    for format in [WeightFormat::Q8_0, WeightFormat::F32] {
        let transcriber = Transcriber::load(dir, format, device)?;
        let model = transcriber.model();
        let window = &clip.samples()[..clip.samples().len().min(480_000)];
        let started = Instant::now();
        let features = model.features(window)?;
        let features_seconds = started.elapsed().as_secs_f64();
        let encode = seconds(device, || model.encode(&features))?;
        decoding(device, model, &features)?;
        let started = Instant::now();
        let result = transcriber.transcribe(&clip, &TranscribeOptions::default())?;
        println!(
            "{}",
            json!({"format": format!("{format:?}"), "features_seconds": features_seconds,
                   "encode_seconds": encode, "transcribe_seconds": started.elapsed().as_secs_f64(),
                   "audio_seconds": clip.duration().as_secs_f64(), "tokens": result.tokens,
                   "windows": result.windows, "text": result.text})
        );
    }
    Ok(())
}
