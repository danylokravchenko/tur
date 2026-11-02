//! Image and video input: the axial rotary kernel against the plain one, and with a checkpoint,
//! what an input costs to encode at its sizes, what reusing its rows saves, and what its
//! encoding costs a stream beside it.
//!
//! `TUR_QWEN35_0_8B=<dir>` adds the checkpoint measurements.
use candle_core::{DType, Device, Tensor};
use serde_json::json;
use std::{path::PathBuf, sync::Arc, time::Instant};
use tur::{
    InferencePipeline, ModelFactory, ModelSource, Result, TurError,
    backend::pipeline::{GenerationRequest, PipelineConfig},
    core::execution::{KvFormat, KvLayout},
    image::Image,
    media::Media,
    models::{TextModel, WeightFormat},
    runtime::kernels::{Heads, Kernels},
    video::Video,
};

const REPEATS: usize = 20;

/// A deterministic image with detail at every scale, so the resampler has work to do. Each
/// `seed` differs in content, so encoding it is real work rather than a reuse of its rows.
fn picture(side: u32, seed: u32) -> Result<Image> {
    let mut pixels = Vec::with_capacity((side * side * 3) as usize);
    for y in 0..side {
        for x in 0..side {
            let ramp = (x * 255 / side) as u8;
            let hiss =
                (x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503) ^ (seed << 24)) >> 24;
            pixels.extend_from_slice(&[ramp, (y * 255 / side) as u8, hiss as u8]);
        }
    }
    Image::from_rgb(side, side, pixels)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Text keys and queries through the axial kernel cost what they do through the plain one.
fn rotary() -> Result<()> {
    let device = Device::new_metal(0)?;
    let kernels = Kernels::new(&device)?;
    let dim = 256;
    let heads = Heads {
        count: 8,
        dim,
        head_stride: dim,
        col_offset: 0,
    };
    let half = 32;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / 1e7f32.powf(i as f32 / half as f32))
        .collect();
    let axis: Vec<u32> = (0..half as u32).map(|f| f % 3).collect();
    let rotation = kernels.axial_rotation(&inv, &axis, 2 * half, 1)?;
    let frequencies = Tensor::new(inv.as_slice(), &device)?;
    let weight = Tensor::ones(dim, DType::F32, &device)?;
    for tokens in [1usize, 64, 1024] {
        let x = Tensor::randn(0f32, 1.0, (tokens, 8 * dim), &device)?;
        let plain = Tensor::from_vec((0..tokens as u32).collect::<Vec<_>>(), tokens, &device)?;
        let axial = Tensor::from_vec(
            (0..tokens as u32).flat_map(|p| [p; 3]).collect::<Vec<_>>(),
            (tokens, 3),
            &device,
        )?;
        let time = |run: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
            let mut samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let start = Instant::now();
                run()?;
                device.synchronize()?;
                samples.push(start.elapsed().as_secs_f64());
            }
            Ok(median(samples))
        };
        let plain = time(&|| kernels.qk_norm_rope(&x, heads, &weight, &plain, &frequencies, 1e-6))?;
        let axial =
            time(&|| kernels.axial_rope(&x, heads, Some((&weight, 1e-6)), &axial, &rotation))?;
        println!(
            "{}",
            json!({"op": "rotary", "tokens": tokens, "plain_seconds": plain, "axial_seconds": axial})
        );
    }
    Ok(())
}

fn encoding(model: &TextModel) -> Result<()> {
    for side in [256u32, 512, 1024, 3000] {
        let image = picture(side, 0)?;
        let mut samples = Vec::with_capacity(3);
        for seed in 0..3 {
            let image = picture(side, seed)?;
            let start = Instant::now();
            model.encode_media(Media::Image(&image))?;
            samples.push(start.elapsed().as_secs_f64());
        }
        println!(
            "{}",
            json!({"op": "encode", "side": side,
                   "soft_tokens": model.media_layout(Media::Image(&image))?.tokens(),
                   "seconds": median(samples)})
        );
    }
    Ok(())
}

/// A video of `seconds` at 10 fps, sampled at 2 per second by Qwen.
fn clip(side: u32, seconds: usize) -> Result<Video> {
    let base = picture(side, 3)?;
    let frames = (0..seconds * 10)
        .map(|at| {
            let mut frame = base.pixels().to_vec();
            let shift = at * 3 * side as usize % frame.len();
            frame.rotate_left(shift);
            frame
        })
        .collect();
    Video::from_frames(side as usize, side as usize, 10.0, frames)
}

fn video_encoding(model: &TextModel) -> Result<()> {
    for (side, seconds) in [(320, 5), (640, 5), (640, 30)] {
        let video = clip(side, seconds)?;
        let layout = model.media_layout(Media::Video(&video))?;
        let start = Instant::now();
        model.encode_media(Media::Video(&video))?;
        println!(
            "{}",
            json!({"op": "encode_video", "side": side, "seconds_of_video": seconds,
                   "steps": layout.segments.len(), "soft_tokens": layout.tokens(),
                   "seconds": start.elapsed().as_secs_f64()})
        );
    }
    Ok(())
}

/// A repeated request's encoding: the first encodes, and the second finds its rows.
fn reuse(model: &TextModel) -> Result<()> {
    let image = picture(600, 4)?;
    for attempt in ["first", "repeat"] {
        let start = Instant::now();
        model.encode_media(Media::Image(&image))?;
        println!(
            "{}",
            json!({"op": "reuse", "attempt": attempt, "seconds": start.elapsed().as_secs_f64()})
        );
    }
    Ok(())
}

/// Text passes while the largest image encodes on another thread, against none encoding: the
/// slowest pass bounds how long a stream can pause for someone else's image.
fn contention(pipeline: &mut InferencePipeline, model: &Arc<TextModel>) -> Result<()> {
    let image = picture(1024, 5)?;
    for concurrent in [false, true] {
        let mut request = GenerationRequest::new("Write a long story about a river.".into(), 400);
        request.sampling.temperature = Some(0.0);
        let id = pipeline.submit_request(&request)?;
        let (model, image) = (Arc::clone(model), image.clone());
        let encoder = concurrent.then(|| {
            std::thread::spawn(move || -> Result<f64> {
                let start = Instant::now();
                model.encode_media(Media::Image(&image))?;
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
            json!({"op": "contention", "concurrent_image": concurrent, "encode_seconds": encode,
                   "median_pass": passes[passes.len() / 2], "slowest_pass": passes[passes.len() - 1]})
        );
    }
    Ok(())
}

pub(crate) fn run() -> Result<()> {
    rotary()?;
    let Some(dir) = std::env::var_os("TUR_QWEN35_0_8B").map(PathBuf::from) else {
        return Ok(());
    };
    let factory = ModelFactory {
        source: ModelSource::LocalPath(dir),
        revision: "Qwen/Qwen3.5-0.8B".into(),
        format: Some(WeightFormat::Q8_0),
        device: Device::new_metal(0)?,
        kv: KvLayout::new(16, KvFormat::F16),
        recurrent: DType::F32,
        audio: false,
        vision: true,
        expert_cache: None,
        gguf: None,
    };
    let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::new(1, 256))?;
    let model = pipeline
        .media_model()
        .ok_or_else(|| TurError::Other("no vision encoder".into()))?;
    encoding(&model)?;
    video_encoding(&model)?;
    reuse(&model)?;
    contention(&mut pipeline, &model)
}
