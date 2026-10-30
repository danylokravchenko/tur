#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Gemma 4's conformer audio encoder and its projection into the text embedding space.
//!
//! Transformers computes the encoder's chunked local attention in blocks of
//! `attention_chunk_size` queries against `chunk + context_left - 1` keys, adds a
//! relative-position term through a "rel shift", and masks everything but the causal
//! window of `context_left - 1` keys ending at the query. With `context_right == 0` and
//! `chunk == context_left - 1` (both checked when the config is parsed) the shift
//! assigns the key at distance `d = t - s` the sinusoid of position `d`, so the blocked
//! form equals dense attention over that band, which is what this file computes.
use super::super::config::{Audio, AudioConfig};
use crate::{
    audio::{self, GemmaFeatures, gemma_frames},
    models::{
        Precision,
        artifact::Companion,
        checkpoint::Checkpoint,
        weights::{tiled, upload_projection},
    },
};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops::{sigmoid, softmax_last_dim};
use std::{collections::BTreeMap, path::Path};
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

/// Epsilon of the conformer's layer and feed-forward norms, which Transformers builds
/// with `Gemma4RMSNorm`'s default rather than the configured value.
const DEFAULT_EPS: f64 = 1e-6;

/// A projection whose input and output are clamped to calibrated bounds.
struct Clipped {
    linear: PackedLinear,
    bounds: Option<[f32; 4]>,
}

impl Clipped {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let Some([in_min, in_max, out_min, out_max]) = self.bounds else {
            return self.linear.forward(x);
        };
        Ok(self
            .linear
            .forward(&x.clamp(in_min, in_max)?)?
            .clamp(out_min, out_max)?)
    }
}

struct FeedForward {
    pre: Tensor,
    up: Clipped,
    down: Clipped,
    post: Tensor,
}

struct LightConv {
    pre: Tensor,
    start: Clipped,
    /// `[kernel, hidden]`: tap `k` multiplies the input `kernel - 1 - k` frames back.
    taps: Tensor,
    norm: Tensor,
    end: Clipped,
}

struct Layer {
    ff1: FeedForward,
    ff2: FeedForward,
    q: Clipped,
    k: Clipped,
    v: Clipped,
    post: Clipped,
    /// `[context_left, hidden]` relative keys of distances `0..context_left`, projected
    /// once at load: the sinusoids are constants.
    relative: Tensor,
    /// `[head_dim]` query scale: `head_dim^-0.5 / ln 2 · softplus(per_dim_scale)`.
    query_scale: Tensor,
    norm_pre_attn: Tensor,
    norm_post_attn: Tensor,
    norm_out: Tensor,
    lconv: LightConv,
}

/// The loaded encoder, the features it reads and the projector into the text model.
pub(super) struct AudioTower {
    config: AudioConfig,
    features: GemmaFeatures,
    conv: [(Tensor, Tensor); 2],
    input: PackedLinear,
    layers: Vec<Layer>,
    output: PackedLinear,
    output_bias: Tensor,
    projection: PackedLinear,
    device: Device,
    bytes: usize,
}

/// Soft tokens of a clip of `samples` at 16 kHz: two stride-2 convolutions over its
/// valid feature frames.
pub(super) fn soft_tokens(samples: usize) -> usize {
    gemma_frames(samples).div_ceil(2).div_ceil(2)
}

/// Storage of the encoder's projections: F32 beside an F32 text model, `Q8_0` otherwise.
pub(super) fn precision(text_f32: bool) -> Precision {
    if text_f32 {
        Precision::F32
    } else {
        Precision::Q8_0
    }
}

/// The encoder's tensors as an artifact stores them: wide projections packed in `precision`
/// and the rest as the checkpoint has them. Relative keys are folded at load, so stay exact.
pub(super) fn stored_tensors(
    audio: &Audio,
    text_hidden: usize,
    precision: Precision,
) -> BTreeMap<String, Companion> {
    audio
        .weight_shapes(text_hidden)
        .into_iter()
        .map(|(name, shape)| {
            let packed = match shape.as_slice() {
                &[_, cols]
                    if name.ends_with(".weight") && !name.ends_with("relative_k_proj.weight") =>
                {
                    Some(tiled(precision, cols)).filter(|p| p.ggml().is_some())
                }
                _ => None,
            };
            (name, Companion { shape, packed })
        })
        .collect()
}

/// Resident bytes of the encoder in `precision`: projections packed, the rest F32.
/// Relative-key projections are folded into `[context_left, hidden]` tables at load.
pub(super) fn planned_bytes(
    audio: &Audio,
    text_hidden: usize,
    precision: Precision,
) -> Result<usize> {
    let c = &audio.config;
    let mut total = 0usize;
    let mut add = |bytes: usize| -> Result<()> {
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("audio weight bytes overflow"))?;
        Ok(())
    };
    for (name, shape) in audio.weight_shapes(text_hidden) {
        if name.ends_with("relative_k_proj.weight") {
            add(c.attention_context_left * c.hidden_size * 4)?;
        } else if name.ends_with(".weight") && shape.len() == 2 {
            add(tiled(precision, shape[1]).bytes(shape[0], shape[1])?)?;
        } else if !shape.is_empty() {
            add(shape.iter().product::<usize>() * 4)?;
        }
    }
    Ok(total)
}

/// Reject feature-extractor settings other than the ones [`GemmaFeatures`] implements.
fn validate_processor(dir: &Path) -> Result<()> {
    let path = dir.join("processor_config.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let processor: serde_json::Value = serde_json::from_str(&raw)?;
    let features = &processor["feature_extractor"];
    let number = |value: &serde_json::Value, expected: f64| {
        value.is_null() || value.as_f64().is_some_and(|v| (v - expected).abs() < 1e-9)
    };
    let expected = [
        ("feature_size", 128.0),
        ("sampling_rate", f64::from(audio::SAMPLE_RATE)),
        ("frame_length", 320.0),
        ("hop_length", 160.0),
        ("fft_length", 512.0),
        ("min_frequency", 0.0),
        ("max_frequency", 8000.0),
        ("preemphasis", 0.0),
        ("dither", 0.0),
        ("input_scale_factor", 1.0),
        ("mel_floor", 0.001),
    ];
    let supported = expected.iter().all(|(key, v)| number(&features[*key], *v))
        && features["per_bin_mean"].is_null()
        && features["per_bin_stddev"].is_null()
        && features["fft_overdrive"].as_bool() != Some(true)
        && number(&processor["audio_ms_per_token"], 40.0)
        && number(&processor["audio_seq_length"], 750.0);
    if !supported {
        return Err(invalid(
            "unsupported Gemma 4 audio feature extractor settings",
        ));
    }
    Ok(())
}

/// The checkpoint's tensors for one encoder, uploaded to `device`.
struct Loader<'a> {
    checkpoint: &'a Checkpoint,
    device: &'a Device,
    precision: Precision,
    bytes: usize,
    /// Input width of each projection, which packed bytes alone do not say.
    packed_cols: BTreeMap<String, usize>,
}

impl Loader<'_> {
    fn dense(&mut self, name: &str) -> Result<Tensor> {
        let tensor = self
            .checkpoint
            .take(name)?
            .to_dtype(DType::F32)?
            .to_device(self.device)?;
        self.bytes += tensor.elem_count() * 4;
        Ok(tensor)
    }
    fn linear(&mut self, name: &str) -> Result<PackedLinear> {
        let entry = self
            .checkpoint
            .entries()
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        let linear = if entry.dtype == safetensors::Dtype::U8 {
            // A prepacked artifact's blocks, uploaded as stored.
            let (entry, bytes) = self.checkpoint.take_raw(name)?;
            let (&[rows, _], Some(format), Some(cols)) = (
                entry.shape.as_slice(),
                self.precision.ggml(),
                self.packed_cols.get(name),
            ) else {
                return Err(invalid(format!("{name} is not a packed matrix")));
            };
            PackedLinear::from_blocks(&bytes, format, (rows, *cols), self.device)?
        } else {
            upload_projection(&self.checkpoint.take(name)?, self.precision, self.device)?
        };
        self.bytes += linear.resident_bytes();
        Ok(linear)
    }
    fn scalar(&self, name: &str) -> Result<f32> {
        Ok(self
            .checkpoint
            .take(name)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?)
    }
    fn clipped(&mut self, name: &str, clip: bool) -> Result<Clipped> {
        let bounds = if clip {
            Some([
                self.scalar(&format!("{name}.input_min"))?,
                self.scalar(&format!("{name}.input_max"))?,
                self.scalar(&format!("{name}.output_min"))?,
                self.scalar(&format!("{name}.output_max"))?,
            ])
        } else {
            None
        };
        Ok(Clipped {
            linear: self.linear(&format!("{name}.linear.weight"))?,
            bounds,
        })
    }
    fn feed_forward(&mut self, name: &str, clip: bool) -> Result<FeedForward> {
        Ok(FeedForward {
            pre: self.dense(&format!("{name}.pre_layer_norm.weight"))?,
            up: self.clipped(&format!("{name}.ffw_layer_1"), clip)?,
            down: self.clipped(&format!("{name}.ffw_layer_2"), clip)?,
            post: self.dense(&format!("{name}.post_layer_norm.weight"))?,
        })
    }
}

/// `[context_left, hidden]` sinusoids of distances `0..context_left`: `sin` over the
/// first half of the width, `cos` over the second, with geometric timescales up to 10⁴.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "positions and widths are small; the table is stored in f32"
)]
fn sinusoids(distances: usize, hidden: usize, device: &Device) -> Result<Tensor> {
    let half = hidden / 2;
    let increment = 10_000f64.ln() / (half.saturating_sub(1).max(1)) as f64;
    let mut table = Vec::with_capacity(distances * hidden);
    for d in 0..distances {
        let angles: Vec<f64> = (0..half)
            .map(|i| d as f64 * (-(i as f64) * increment).exp())
            .collect();
        table.extend(angles.iter().map(|a| a.sin() as f32));
        table.extend(angles.iter().map(|a| a.cos() as f32));
    }
    Ok(Tensor::from_vec(table, (distances, hidden), device)?)
}

impl AudioTower {
    /// Load the encoder and projector of a checkpoint whose config declares them.
    pub(super) fn load(
        dir: &Path,
        checkpoint: &Checkpoint,
        audio: &Audio,
        text_hidden: usize,
        precision: Precision,
        device: &Device,
    ) -> Result<Self> {
        validate_processor(dir)?;
        let mut packed_cols = BTreeMap::new();
        for (name, shape) in audio.weight_shapes(text_hidden) {
            let entry = checkpoint
                .entries()
                .get(&name)
                .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
            if let (safetensors::Dtype::U8, &[_, cols]) = (entry.dtype, shape.as_slice()) {
                // Open-time validation checked the packed shape.
                packed_cols.insert(name, cols);
            } else if entry.shape != shape {
                return Err(invalid(format!(
                    "{name} has shape {:?}, expected {shape:?}",
                    entry.shape
                )));
            }
        }
        let c = audio.config.clone();
        let mut t = Loader {
            checkpoint,
            device,
            precision,
            bytes: 0,
            packed_cols,
        };
        let tower = "model.audio_tower";
        let sub = format!("{tower}.subsample_conv_projection");
        let conv = [
            (
                t.dense(&format!("{sub}.layer0.conv.weight"))?,
                t.dense(&format!("{sub}.layer0.norm.weight"))?,
            ),
            (
                t.dense(&format!("{sub}.layer1.conv.weight"))?,
                t.dense(&format!("{sub}.layer1.norm.weight"))?,
            ),
        ];
        let input = t.linear(&format!("{sub}.input_proj_linear.weight"))?;
        let positions = sinusoids(c.attention_context_left, c.hidden_size, device)?;
        let dim = audio.head_dim();
        #[allow(clippy::cast_precision_loss, reason = "head widths are small")]
        let query = (dim as f64).powf(-0.5) / std::f64::consts::LN_2;
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let p = format!("{tower}.layers.{i}");
                let clip = c.use_clipped_linears;
                let relative = PackedLinear::dense(
                    &checkpoint.take(&format!("{p}.self_attn.relative_k_proj.weight"))?,
                    device,
                )?
                .forward(&positions)?;
                t.bytes += relative.elem_count() * 4;
                let per_dim = t.dense(&format!("{p}.self_attn.per_dim_scale"))?;
                // softplus(x) = ln(1 + eˣ)
                let query_scale = (per_dim.exp()?.affine(1.0, 1.0)?.log()? * query)?;
                let kernel = t.dense(&format!("{p}.lconv1d.depthwise_conv1d.weight"))?;
                Ok(Layer {
                    ff1: t.feed_forward(&format!("{p}.feed_forward1"), clip)?,
                    ff2: t.feed_forward(&format!("{p}.feed_forward2"), clip)?,
                    q: t.clipped(&format!("{p}.self_attn.q_proj"), clip)?,
                    k: t.clipped(&format!("{p}.self_attn.k_proj"), clip)?,
                    v: t.clipped(&format!("{p}.self_attn.v_proj"), clip)?,
                    post: t.clipped(&format!("{p}.self_attn.post"), clip)?,
                    relative,
                    query_scale,
                    norm_pre_attn: t.dense(&format!("{p}.norm_pre_attn.weight"))?,
                    norm_post_attn: t.dense(&format!("{p}.norm_post_attn.weight"))?,
                    norm_out: t.dense(&format!("{p}.norm_out.weight"))?,
                    lconv: LightConv {
                        pre: t.dense(&format!("{p}.lconv1d.pre_layer_norm.weight"))?,
                        start: t.clipped(&format!("{p}.lconv1d.linear_start"), clip)?,
                        // `[hidden, 1, kernel]` → `[kernel, hidden]`, oldest tap first.
                        taps: kernel.squeeze(1)?.t()?.contiguous()?,
                        norm: t.dense(&format!("{p}.lconv1d.conv_norm.weight"))?,
                        end: t.clipped(&format!("{p}.lconv1d.linear_end"), clip)?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let output = t.linear(&format!("{tower}.output_proj.weight"))?;
        let output_bias = t.dense(&format!("{tower}.output_proj.bias"))?;
        let projection = t.linear("model.embed_audio.embedding_projection.weight")?;
        Ok(Self {
            features: GemmaFeatures::new(Audio::MELS, device)?,
            config: c,
            conv,
            input,
            layers,
            output,
            output_bias,
            projection,
            device: device.clone(),
            bytes: t.bytes,
        })
    }

    /// Device bytes of the encoder's weights and tables.
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }

    /// RMS norm with the configured epsilon, as the light convolution and projector use.
    fn rms(&self, x: &Tensor, scale: Option<&Tensor>) -> Result<Tensor> {
        rms(x, scale, self.config.rms_norm_eps)
    }
}

/// `x · (mean(x²) + eps)^-½`, times `scale` when given.
fn rms(x: &Tensor, scale: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let inv = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    let normed = x.broadcast_mul(&inv)?;
    Ok(match scale {
        Some(scale) => normed.broadcast_mul(scale)?,
        None => normed,
    })
}

impl AudioTower {
    fn clip(&self, x: &Tensor) -> Result<Tensor> {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the bound saturates to f32's range, as Transformers clamps it"
        )]
        let bound = self.config.gradient_clipping.min(f64::from(f32::MAX)) as f32;
        Ok(x.clamp(-bound, bound)?)
    }

    fn feed_forward(&self, ff: &FeedForward, x: &Tensor) -> Result<Tensor> {
        let h = rms(&self.clip(x)?, Some(&ff.pre), DEFAULT_EPS)?;
        let h = ff.down.forward(&ff.up.forward(&h)?.silu()?)?;
        let h = rms(&self.clip(&h)?, Some(&ff.post), DEFAULT_EPS)?;
        Ok(((h * self.config.residual_weight)? + x)?)
    }

    /// Two stride-2 3×3 convolutions over time and frequency, each followed by a
    /// channel `LayerNorm` without bias and a `ReLU`, then a projection to the model width.
    fn subsample(&self, features: &Tensor) -> Result<Tensor> {
        let mut x = features.unsqueeze(0)?.unsqueeze(0)?;
        for (kernel, norm) in &self.conv {
            let y = x.conv2d(kernel, 1, 2, 1, 1)?.permute((0, 2, 3, 1))?;
            let mean = y.mean_keepdim(D::Minus1)?;
            let centred = y.broadcast_sub(&mean)?;
            let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
            let normed = centred
                .broadcast_div(&(var + self.config.rms_norm_eps)?.sqrt()?)?
                .broadcast_mul(norm)?;
            x = normed.relu()?.permute((0, 3, 1, 2))?.contiguous()?;
        }
        // `[1, channels, time, frequency]` → `[time, frequency · channels]`.
        let (_, channels, time, frequency) = x.dims4()?;
        let flat = x
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .reshape((time, frequency * channels))?;
        self.input.forward(&flat)
    }

    /// Dense attention over each query's causal window of `context_left - 1` keys.
    fn attention(&self, layer: &Layer, x: &Tensor) -> Result<Tensor> {
        let c = &self.config;
        let (frames, hidden) = x.dims2()?;
        let heads = c.num_attention_heads;
        let dim = hidden / heads;
        let split = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((frames, heads, dim))?
                .transpose(0, 1)?
                .contiguous()?)
        };
        let q = split(
            layer
                .q
                .forward(x)?
                .broadcast_mul(&layer.query_scale.repeat(heads)?)?,
        )?;
        #[allow(clippy::cast_possible_truncation, reason = "a fixed constant")]
        let k_scale = (std::f64::consts::E.ln_1p() / std::f64::consts::LN_2) as f32;
        let k = split((layer.k.forward(x)? * f64::from(k_scale))?)?;
        let v = split(layer.v.forward(x)?)?;
        let window = c.attention_context_left - 1;
        // `[heads, frames, distances]`: each query against the relative key of each distance.
        let distances = layer.relative.dim(0)?;
        let relative = layer
            .relative
            .reshape((distances, heads, dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let by_distance = q.matmul(&relative.transpose(1, 2)?)?;
        let (index, mask) = band(frames, window, &self.device)?;
        let positional = by_distance.gather(
            &index
                .unsqueeze(0)?
                .broadcast_as((heads, frames, frames))?
                .contiguous()?,
            2,
        )?;
        let cap = c.attention_logit_cap;
        let logits = (((q.matmul(&k.transpose(1, 2)?)? + positional)? / cap)?.tanh()? * cap)?;
        // Masked keys take the configured logit, as Transformers' `masked_fill`.
        let fill = c.attention_invalid_logits_value;
        let logits = logits
            .broadcast_mul(&mask)?
            .broadcast_add(&mask.affine(-fill, fill)?)?;
        let out = softmax_last_dim(&logits)?.matmul(&v)?;
        layer.post.forward(
            &out.transpose(0, 1)?
                .contiguous()?
                .reshape((frames, hidden))?,
        )
    }

    /// Gated linear unit, causal depthwise convolution, norm, `SiLU` and projection.
    fn light_conv(&self, lconv: &LightConv, x: &Tensor) -> Result<Tensor> {
        let (frames, hidden) = x.dims2()?;
        let gated = lconv.start.forward(&self.rms(x, Some(&lconv.pre))?)?;
        let h = gated
            .narrow(1, 0, hidden)?
            .mul(&sigmoid(&gated.narrow(1, hidden, hidden)?)?)?;
        let kernel = lconv.taps.dim(0)?;
        let padded = Tensor::cat(
            &[
                Tensor::zeros((kernel - 1, hidden), DType::F32, &self.device)?,
                h,
            ],
            0,
        )?;
        let mut conv = Tensor::zeros((frames, hidden), DType::F32, &self.device)?;
        for tap in 0..kernel {
            conv = (conv
                + padded
                    .narrow(0, tap, frames)?
                    .broadcast_mul(&lconv.taps.narrow(0, tap, 1)?)?)?;
        }
        let h = self.rms(&self.clip(&conv)?, Some(&lconv.norm))?.silu()?;
        Ok((lconv.end.forward(&h)? + x)?)
    }

    fn layer(&self, layer: &Layer, x: &Tensor) -> Result<Tensor> {
        let x = self.feed_forward(&layer.ff1, x)?;
        let h = rms(&self.clip(&x)?, Some(&layer.norm_pre_attn), DEFAULT_EPS)?;
        let h = self.attention(layer, &h)?;
        let h = rms(&self.clip(&h)?, Some(&layer.norm_post_attn), DEFAULT_EPS)?;
        let x = self.light_conv(&layer.lconv, &(h + x)?)?;
        let x = self.feed_forward(&layer.ff2, &x)?;
        rms(&self.clip(&x)?, Some(&layer.norm_out), DEFAULT_EPS)
    }

    /// The encoder's `[tokens, output_proj_dims]` output for `features`.
    pub(super) fn encode_features(&self, features: &Tensor) -> Result<Tensor> {
        let mut x = self.subsample(features)?;
        for layer in &self.layers {
            x = self.layer(layer, &x)?;
        }
        Ok(self.output.forward(&x)?.broadcast_add(&self.output_bias)?)
    }

    /// F32 `[tokens, text_hidden]` soft tokens of a clip, at most [`Audio::MAX_TOKENS`].
    pub(super) fn encode(&self, clip: &audio::Audio) -> Result<Tensor> {
        let tokens = soft_tokens(clip.samples().len());
        if tokens == 0 || tokens > Audio::MAX_TOKENS {
            return Err(invalid(
                "audio clip outside the encoder's 40 ms to 30 s range",
            ));
        }
        let encoded = self.encode_features(&self.features.extract(clip.samples())?)?;
        if encoded.dim(0)? != tokens {
            return Err(invalid("audio encoder produced an unexpected token count"));
        }
        self.projection.forward(&self.rms(&encoded, None)?)
    }
}

/// `[frames, frames]` distance indices `t - s` (clamped into the window) and the mask
/// of keys `s` with `0 ≤ t - s ≤ window - 1`, as F32 ones and zeros.
fn band(frames: usize, window: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut index = Vec::with_capacity(frames * frames);
    let mut mask = Vec::with_capacity(frames * frames);
    for t in 0..frames {
        for s in 0..frames {
            let inside = s <= t && t - s < window;
            index.push(if inside {
                u32::try_from(t - s).map_err(|_| invalid("audio distance overflow"))?
            } else {
                0
            });
            mask.push(if inside { 1.0f32 } else { 0.0 });
        }
    }
    Ok((
        Tensor::from_vec(index, (frames, frames), device)?,
        Tensor::from_vec(mask, (frames, frames), device)?,
    ))
}
