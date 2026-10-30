#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Whisper's encoder and cached decoder over Candle tensor operations.
use super::config::{Config, validate_preprocessor};
use crate::{
    audio::{WhisperClip, WhisperFeatures},
    models::{
        Precision, WeightFormat,
        checkpoint::{Checkpoint, weight_files},
        weights::{tiled, upload_projection},
    },
};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::{layer_norm, softmax_last_dim};
use std::path::Path;
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

/// Whisper's `LayerNorm` epsilon (`PyTorch`'s default).
const EPS: f32 = 1e-5;
/// Encoder heads attended at once: bounds the `[heads, 1500, 1500]` score buffer.
const HEAD_GROUP: usize = 4;

/// Metal attention over rows that every query sees whole: the paged kernels read the
/// projections in place, as one page of a pool that needs no copy. 10× faster than
/// Candle's matmuls and softmax on the encoder's shape.
#[cfg(feature = "metal")]
struct FusedAttention {
    kernels: tur_metal::kernels::Kernels,
    head_dim: usize,
}

#[cfg(feature = "metal")]
impl FusedAttention {
    /// Kernels for `device` when it is Metal and has kernels for the head width.
    fn new(config: &Config, device: &Device) -> Result<Option<Self>> {
        let head_dim = config.d_model / config.encoder_attention_heads;
        if !device.is_metal()
            || !matches!(head_dim, 64 | 128 | 256)
            || config.d_model / config.decoder_attention_heads != head_dim
        {
            return Ok(None);
        }
        Ok(Some(Self {
            kernels: tur_metal::kernels::Kernels::new(device)?,
            head_dim,
        }))
    }

    /// `[queries, heads · dim]` attention of query rows over `[keys, heads · dim]` rows.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        use tur_metal::pool::{AttentionPool, KvPool};
        let ((queries, width), keys) = (q.dims2()?, k.dim(0)?);
        let (heads, dim) = (width / self.head_dim, self.head_dim);
        let visible = u32::try_from(keys).map_err(|_| invalid("too many Whisper keys"))?;
        let pool = AttentionPool {
            key: KvPool {
                data: k.reshape((keys, heads, dim))?,
                scales: None,
            },
            value: KvPool {
                data: v.reshape((keys, heads, dim))?,
                scales: None,
            },
            format: tur_core::execution::KvFormat::F32,
            page_tokens: keys,
            kv_heads: heads,
            head_dim: dim,
        };
        let kv =
            self.kernels
                .paged_kv(keys, keys, &[0], &vec![0; queries], &vec![visible; queries])?;
        #[allow(clippy::cast_precision_loss, reason = "head widths are small")]
        let scale = (dim as f64).powf(-0.5);
        self.kernels
            .attention_ungated(&q.reshape((queries, heads, dim))?, &pool, &kv, None, scale)
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(layer_norm(x, &self.weight, &self.bias, EPS)?)
    }
}

/// A projection and its bias, which Whisper's keys lack.
struct Linear {
    weight: PackedLinear,
    bias: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.weight.forward(x)?;
        Ok(match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        })
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm: Norm,
}

struct Layer {
    attention: Attention,
    /// Decoder layers attend to the encoder output.
    cross: Option<Attention>,
    fc1: Linear,
    fc2: Linear,
    norm: Norm,
}

/// One loaded Whisper checkpoint: immutable weights shared by every transcription.
pub struct Whisper {
    config: Config,
    features: WhisperFeatures,
    conv: [(Tensor, Tensor); 2],
    encoder_positions: Tensor,
    encoder: Vec<Layer>,
    encoder_norm: Norm,
    /// Token embedding and, tied, the output projection.
    embedding: PackedLinear,
    decoder_positions: Tensor,
    decoder: Vec<Layer>,
    decoder_norm: Norm,
    device: Device,
    bytes: usize,
    #[cfg(feature = "metal")]
    fused: Option<FusedAttention>,
}

struct Loader<'a> {
    checkpoint: &'a Checkpoint,
    precision: Precision,
    device: &'a Device,
    bytes: usize,
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
    fn projection(&mut self, name: &str) -> Result<PackedLinear> {
        let linear = upload_projection(&self.checkpoint.take(name)?, self.precision, self.device)?;
        self.bytes += linear.resident_bytes();
        Ok(linear)
    }
    fn linear(&mut self, name: &str, bias: bool) -> Result<Linear> {
        Ok(Linear {
            weight: self.projection(&format!("{name}.weight"))?,
            bias: bias
                .then(|| self.dense(&format!("{name}.bias")))
                .transpose()?,
        })
    }
    fn norm(&mut self, name: &str) -> Result<Norm> {
        Ok(Norm {
            weight: self.dense(&format!("{name}.weight"))?,
            bias: self.dense(&format!("{name}.bias"))?,
        })
    }
    fn attention(&mut self, name: &str, norm: &str) -> Result<Attention> {
        Ok(Attention {
            q: self.linear(&format!("{name}.q_proj"), true)?,
            k: self.linear(&format!("{name}.k_proj"), false)?,
            v: self.linear(&format!("{name}.v_proj"), true)?,
            out: self.linear(&format!("{name}.out_proj"), true)?,
            norm: self.norm(norm)?,
        })
    }
    fn layer(&mut self, prefix: &str, cross: bool) -> Result<Layer> {
        Ok(Layer {
            attention: self.attention(
                &format!("{prefix}.self_attn"),
                &format!("{prefix}.self_attn_layer_norm"),
            )?,
            cross: cross
                .then(|| {
                    self.attention(
                        &format!("{prefix}.encoder_attn"),
                        &format!("{prefix}.encoder_attn_layer_norm"),
                    )
                })
                .transpose()?,
            fc1: self.linear(&format!("{prefix}.fc1"), true)?,
            fc2: self.linear(&format!("{prefix}.fc2"), true)?,
            norm: self.norm(&format!("{prefix}.final_layer_norm"))?,
        })
    }
}

/// Projection storage for `format`.
fn precision(format: WeightFormat) -> Precision {
    Precision::from(format)
}

/// `[heads, tokens, head_dim]` view of `[tokens, heads · head_dim]` rows.
fn heads(x: &Tensor, heads: usize) -> Result<Tensor> {
    let (tokens, width) = x.dims2()?;
    Ok(x.reshape((tokens, heads, width / heads))?
        .transpose(0, 1)?
        .contiguous()?)
}

/// Scaled dot-product attention of `[heads, q, dim]` queries over `[heads, k, dim]`
/// keys and values, optionally causal with `past` cached keys before the queries.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, causal: Option<usize>) -> Result<Tensor> {
    let (_, rows, dim) = q.dims3()?;
    #[allow(clippy::cast_precision_loss, reason = "head widths are small")]
    let scale = (dim as f64).powf(-0.5);
    let mut scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
    if let Some(past) = causal {
        let keys = k.dim(1)?;
        let mask: Vec<f32> = (0..rows)
            .flat_map(|r| {
                (0..keys).map(move |c| {
                    if c <= past + r {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        scores = scores.broadcast_add(&Tensor::from_vec(mask, (rows, keys), q.device())?)?;
    }
    Ok(softmax_last_dim(&scores)?.matmul(v)?)
}

/// `[tokens, heads · dim]` rows of `[heads, tokens, dim]` outputs.
fn merge(x: &Tensor) -> Result<Tensor> {
    let (heads, tokens, dim) = x.dims3()?;
    Ok(x.transpose(0, 1)?
        .contiguous()?
        .reshape((tokens, heads * dim))?)
}

impl Whisper {
    /// Load a validated checkpoint with projections in `format`; norms, biases,
    /// convolutions and positions stay F32. The token embedding doubles as the output
    /// projection and is stored in `format` too.
    pub fn load(dir: &Path, format: WeightFormat, device: &Device) -> Result<Self> {
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        validate_preprocessor(dir, &config)?;
        let (checkpoint, _) = Checkpoint::scan(weight_files(dir)?)?;
        for (name, shape) in config.weight_shapes() {
            let entry = checkpoint
                .entries()
                .get(&name)
                .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
            if entry.shape != shape {
                return Err(invalid(format!(
                    "{name} has shape {:?}, expected {shape:?}",
                    entry.shape
                )));
            }
        }
        let mut t = Loader {
            checkpoint: &checkpoint,
            precision: precision(format),
            device,
            bytes: 0,
        };
        let conv = [
            (
                t.dense("model.encoder.conv1.weight")?,
                t.dense("model.encoder.conv1.bias")?,
            ),
            (
                t.dense("model.encoder.conv2.weight")?,
                t.dense("model.encoder.conv2.bias")?,
            ),
        ];
        let encoder_positions = t.dense("model.encoder.embed_positions.weight")?;
        let encoder = (0..config.encoder_layers)
            .map(|i| t.layer(&format!("model.encoder.layers.{i}"), false))
            .collect::<Result<Vec<_>>>()?;
        let encoder_norm = t.norm("model.encoder.layer_norm")?;
        let embedding = t.projection("model.decoder.embed_tokens.weight")?;
        let decoder_positions = t.dense("model.decoder.embed_positions.weight")?;
        let decoder = (0..config.decoder_layers)
            .map(|i| t.layer(&format!("model.decoder.layers.{i}"), true))
            .collect::<Result<Vec<_>>>()?;
        let decoder_norm = t.norm("model.decoder.layer_norm")?;
        let bytes = t.bytes;
        Ok(Self {
            features: WhisperFeatures::new(config.num_mel_bins, device)?,
            #[cfg(feature = "metal")]
            fused: FusedAttention::new(&config, device)?,
            config,
            conv,
            encoder_positions,
            encoder,
            encoder_norm,
            embedding,
            decoder_positions,
            decoder,
            decoder_norm,
            device: device.clone(),
            bytes,
        })
    }

    /// Resident bytes [`Self::load`] will hold for `config` in `format`.
    pub fn planned_bytes(config: &Config, format: WeightFormat) -> Result<usize> {
        let precision = precision(format);
        config
            .weight_shapes()
            .into_iter()
            .try_fold(0usize, |total, (name, shape)| {
                let bytes = match shape.as_slice() {
                    &[rows, cols]
                        if name.ends_with(".weight") && !name.contains("embed_positions") =>
                    {
                        tiled(precision, cols).bytes(rows, cols)?
                    }
                    _ => shape.iter().product::<usize>() * 4,
                };
                total
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("Whisper weight bytes overflow"))
            })
    }

    /// Peak transient device bytes of one encoder pass: the widest activation (the
    /// feed-forward rows, or the second convolution's unfolded input), a head group's
    /// scores and its softmax, and the residual stream with its normed copy.
    pub fn scratch_bytes(config: &Config) -> Result<usize> {
        let frames = config.max_source_positions;
        let wide = config
            .encoder_ffn_dim
            .max(config.d_model * 3 * 2)
            .checked_mul(frames * 2);
        let scores = HEAD_GROUP.min(config.encoder_attention_heads) * frames * frames * 2;
        wide.and_then(|w| w.checked_add(scores))
            .and_then(|b| b.checked_add(4 * frames * config.d_model))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| invalid("Whisper scratch size overflow"))
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Device bytes of the loaded weights.
    pub fn weight_bytes(&self) -> usize {
        self.bytes
    }

    /// `[mels, 3000]` log-mel features of at most 30 s of 16 kHz samples.
    pub fn features(&self, samples: &[f32]) -> Result<Tensor> {
        self.features.extract(samples)
    }

    /// Host features of a whole clip, to window for long audio.
    pub(crate) fn clip_features(&self, samples: &[f32]) -> Result<WhisperClip> {
        self.features.clip(samples)
    }

    /// The encoder's `[1500, d_model]` states for one window's features.
    pub fn encode(&self, features: &Tensor) -> Result<Tensor> {
        let mut x = features.unsqueeze(0)?;
        for (stride, (weight, bias)) in [1, 2].into_iter().zip(&self.conv) {
            x = x
                .conv1d(weight, 1, stride, 1, 1)?
                .broadcast_add(&bias.reshape((1, (), 1))?)?
                .gelu_erf()?;
        }
        let mut x = x
            .squeeze(0)?
            .t()?
            .contiguous()?
            .broadcast_add(&self.encoder_positions)?;
        let heads = self.config.encoder_attention_heads;
        for layer in &self.encoder {
            let a = &layer.attention;
            let h = a.norm.forward(&x)?;
            let attended = self.attend_whole(
                &a.q.forward(&h)?,
                &a.k.forward(&h)?,
                &a.v.forward(&h)?,
                heads,
            )?;
            x = (x + a.out.forward(&attended)?)?;
            x = (&x + Self::feed_forward(layer, &x)?)?;
        }
        self.encoder_norm.forward(&x)
    }

    /// `[queries, heads · dim]` attention over `[keys, heads · dim]` rows every query sees.
    fn attend_whole(&self, q: &Tensor, k: &Tensor, v: &Tensor, count: usize) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(fused) = &self.fused {
            return fused.attend(q, k, v);
        }
        let (q, k, v) = (heads(q, count)?, heads(k, count)?, heads(v, count)?);
        let groups = (0..count)
            .step_by(HEAD_GROUP)
            .map(|start| {
                let n = HEAD_GROUP.min(count - start);
                attend(
                    &q.narrow(0, start, n)?,
                    &k.narrow(0, start, n)?,
                    &v.narrow(0, start, n)?,
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        merge(&Tensor::cat(&groups, 0)?)
    }

    fn feed_forward(layer: &Layer, x: &Tensor) -> Result<Tensor> {
        layer
            .fc2
            .forward(&layer.fc1.forward(&layer.norm.forward(x)?)?.gelu_erf()?)
    }

    /// A decoding over one window's encoder states: cross-attention keys and values are
    /// projected once here, and self-attention keys and values accumulate per step.
    pub fn decoder(&self, encoded: &Tensor) -> Result<Decoder<'_>> {
        let cross = self
            .decoder
            .iter()
            .map(|layer| {
                let cross = layer
                    .cross
                    .as_ref()
                    .ok_or_else(|| invalid("decoder layer without cross-attention"))?;
                Ok((cross.k.forward(encoded)?, cross.v.forward(encoded)?))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Decoder {
            model: self,
            cross,
            cache: vec![None; self.decoder.len()],
            position: 0,
        })
    }
}

fn heads_of(linear: &Linear, x: &Tensor, count: usize) -> Result<Tensor> {
    heads(&linear.forward(x)?, count)
}

/// Incremental decoder state of one window.
pub struct Decoder<'a> {
    model: &'a Whisper,
    /// Each layer's cross-attention `[1500, heads · dim]` keys and values.
    cross: Vec<(Tensor, Tensor)>,
    cache: Vec<Option<(Tensor, Tensor)>>,
    position: usize,
}

impl Decoder<'_> {
    /// Tokens decoded so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Append `tokens` and return F32 `[tokens.len(), vocab]` logits after each.
    pub fn step(&mut self, tokens: &[u32]) -> Result<Tensor> {
        let model = self.model;
        let config = &model.config;
        let end = self.position + tokens.len();
        if tokens.is_empty() || end > config.max_target_positions {
            return Err(invalid("Whisper decoder positions exhausted"));
        }
        if tokens.iter().any(|&t| t as usize >= config.vocab_size) {
            return Err(invalid("token outside the Whisper vocabulary"));
        }
        let ids = Tensor::new(tokens, &model.device)?;
        let mut x = model.embedding.rows(&ids)?;
        if config.scale_embedding {
            #[allow(clippy::cast_precision_loss, reason = "model widths are small")]
            let scale = (config.d_model as f64).sqrt();
            x = (x * scale)?;
        }
        x = (x + model
            .decoder_positions
            .narrow(0, self.position, tokens.len())?)?;
        let heads = config.decoder_attention_heads;
        for (i, layer) in model.decoder.iter().enumerate() {
            let a = &layer.attention;
            let h = a.norm.forward(&x)?;
            let q = self::heads(&a.q.forward(&h)?, heads)?;
            let (mut k, mut v) = (heads_of(&a.k, &h, heads)?, heads_of(&a.v, &h, heads)?);
            if let Some((past_k, past_v)) = &self.cache[i] {
                k = Tensor::cat(&[past_k, &k], 1)?;
                v = Tensor::cat(&[past_v, &v], 1)?;
            }
            let attended = attend(&q, &k, &v, Some(self.position))?;
            self.cache[i] = Some((k, v));
            x = (x + a.out.forward(&merge(&attended)?)?)?;
            let cross = layer
                .cross
                .as_ref()
                .ok_or_else(|| invalid("decoder layer without cross-attention"))?;
            let q = cross.q.forward(&cross.norm.forward(&x)?)?;
            let (k, v) = &self.cross[i];
            x = (&x + cross.out.forward(&model.attend_whole(&q, k, v, heads)?)?)?;
            x = (&x + Whisper::feed_forward(layer, &x)?)?;
        }
        self.position = end;
        model.embedding.forward(&model.decoder_norm.forward(&x)?)
    }
}

#[cfg(all(test, feature = "metal"))]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert numerics after fallible setup"
)]
mod tests {
    use super::*;

    fn rows(device: &Device, tokens: usize, width: usize, seed: f32) -> Result<Tensor> {
        let values: Vec<f32> = (0..tokens * width)
            .map(|i| ((i as f32).mul_add(0.37, seed)).sin())
            .collect();
        Ok(Tensor::from_vec(values, (tokens, width), device)?)
    }

    /// Candle's matmuls and softmax are the numerical reference for the fused kernels: the
    /// tiled shape the encoder runs and the per-query shape of decoder cross-attention.
    #[test]
    fn test_fused_attention_matches_the_candle_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (heads, dim) = (4, 64);
        let fused = FusedAttention {
            kernels: tur_metal::kernels::Kernels::new(&device)?,
            head_dim: dim,
        };
        for (queries, keys) in [(1500, 1500), (3, 1500), (1, 40)] {
            let q = rows(&device, queries, heads * dim, 0.5)?;
            let k = rows(&device, keys, heads * dim, 1.5)?;
            let v = rows(&device, keys, heads * dim, 2.5)?;
            let reference = merge(&attend(
                &self::heads(&q, heads)?,
                &self::heads(&k, heads)?,
                &self::heads(&v, heads)?,
                None,
            )?)?;
            let actual = fused.attend(&q, &k, &v)?;
            assert_eq!(actual.dims(), reference.dims());
            let error = (actual - &reference)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error < 1e-3, "{queries}×{keys}: {error}");
        }
        Ok(())
    }
}
