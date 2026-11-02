#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Whisper's encoder and cached decoder over Candle tensor operations.
use super::config::{Config, validate_preprocessor};
#[cfg(feature = "metal")]
use crate::models::encoder::WholeAttention;
use crate::{
    audio::{WhisperClip, WhisperFeatures},
    models::{
        Precision, WeightFormat,
        checkpoint::{Checkpoint, weight_files},
        encoder::{Linear, Loader, Norm, attend, attend_whole, head_scale, heads, merge},
        weights::tiled,
    },
};
#[cfg(feature = "metal")]
use candle_core::DType;
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use std::path::Path;
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

/// Whisper's `LayerNorm` epsilon (`PyTorch`'s default).
const EPS: f32 = 1e-5;
/// Encoder heads attended at once: bounds the `[heads, 1500, 1500]` score buffer.
const HEAD_GROUP: usize = 4;

/// Whole-row attention plus Whisper's choice of cross-attention storage.
#[cfg(feature = "metal")]
struct FusedAttention {
    whole: WholeAttention,
    /// Cross-attention reads 15 MB per token per layer in F32, at the memory's limit, so
    /// beside packed projections its keys and values are stored F16.
    cross: DType,
}

#[cfg(feature = "metal")]
impl FusedAttention {
    /// Kernels for `device` when it is Metal and has kernels for the head width.
    fn new(config: &Config, format: WeightFormat, device: &Device) -> Result<Option<Self>> {
        let head_dim = config.d_model / config.encoder_attention_heads;
        if config.d_model / config.decoder_attention_heads != head_dim {
            return Ok(None);
        }
        Ok(WholeAttention::new(head_dim, device)?.map(|whole| Self {
            whole,
            cross: if format == WeightFormat::F32 {
                DType::F32
            } else {
                DType::F16
            },
        }))
    }
}

/// A decoder's keys and values on the attention kernels.
#[cfg(feature = "metal")]
struct DecoderPools {
    /// Each layer's own tokens, written as they are decoded.
    own: Vec<tur_metal::pool::AttentionPool>,
    /// Each layer's view of the encoder states.
    cross: Vec<tur_metal::pool::AttentionPool>,
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

impl Loader<'_> {
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

/// Storage of the token table and output head. It streams 70 MB per token in `Q8_0`, the
/// largest read of a decoder step, so a packed model keeps it in Q4K: about 4% off an 80 s
/// clip, with the same words on it.
fn head_precision(format: WeightFormat) -> Precision {
    match format {
        WeightFormat::Q8_0 => Precision::Q4k,
        format => Precision::from(format),
    }
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
            eps: EPS,
            bytes: 0,
            packed_cols: std::collections::BTreeMap::new(),
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
        t.precision = head_precision(format);
        let embedding = t.projection("model.decoder.embed_tokens.weight")?;
        t.precision = precision(format);
        let decoder_positions = t.dense("model.decoder.embed_positions.weight")?;
        let decoder = (0..config.decoder_layers)
            .map(|i| t.layer(&format!("model.decoder.layers.{i}"), true))
            .collect::<Result<Vec<_>>>()?;
        let decoder_norm = t.norm("model.decoder.layer_norm")?;
        let bytes = t.bytes;
        Ok(Self {
            features: WhisperFeatures::new(config.num_mel_bins, device)?,
            #[cfg(feature = "metal")]
            fused: FusedAttention::new(&config, format, device)?,
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
                        let table = name.ends_with("embed_tokens.weight");
                        let stored = if table {
                            head_precision(format)
                        } else {
                            precision
                        };
                        tiled(stored, cols).bytes(rows, cols)?
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
        // A decoder's own keys and values, and the encoder states projected for each layer.
        let decoder =
            2 * config.decoder_layers * (config.max_target_positions + frames) * config.d_model;
        wide.and_then(|w| w.checked_add(scores))
            .and_then(|b| b.checked_add(4 * frames * config.d_model))
            .and_then(|b| b.checked_add(decoder))
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
            let kv = fused.whole.whole(q.dim(0)?, k.dim(0)?)?;
            let scale = head_scale(fused.whole.head_dim);
            return fused.whole.attend(q, &fused.whole.pool(k, v)?, &kv, scale);
        }
        let scale = head_scale(q.dim(1)? / count);
        attend_whole((q, k, v), count, HEAD_GROUP, scale)
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
        #[cfg(feature = "metal")]
        let pools = self.decoder_pools(&cross)?;
        Ok(Decoder {
            model: self,
            #[cfg(feature = "metal")]
            pools,
            cross,
            cache: vec![None; self.decoder.len()],
            position: 0,
            alignment: Vec::new(),
        })
    }
}

#[cfg(feature = "metal")]
impl Whisper {
    /// Empty pools for the decoder's tokens, and the encoder's keys and values as a pool.
    fn decoder_pools(&self, cross: &[(Tensor, Tensor)]) -> Result<Option<DecoderPools>> {
        let Some(fused) = &self.fused else {
            return Ok(None);
        };
        let (positions, width) = (self.config.max_target_positions, self.config.d_model);
        let empty = || Tensor::zeros((positions, width), DType::F32, &self.device);
        Ok(Some(DecoderPools {
            own: (0..self.decoder.len())
                .map(|_| fused.whole.pool(&empty()?, &empty()?))
                .collect::<Result<_>>()?,
            cross: cross
                .iter()
                .map(|(k, v)| {
                    fused
                        .whole
                        .pool(&k.to_dtype(fused.cross)?, &v.to_dtype(fused.cross)?)
                })
                .collect::<Result<_>>()?,
        }))
    }
}

fn heads_of(linear: &Linear, x: &Tensor, count: usize) -> Result<Tensor> {
    heads(&linear.forward(x)?, count)
}

/// Incremental decoder state of one window.
pub struct Decoder<'a> {
    model: &'a Whisper,
    #[cfg(feature = "metal")]
    pools: Option<DecoderPools>,
    /// Each layer's cross-attention `[1500, heads · dim]` keys and values.
    cross: Vec<(Tensor, Tensor)>,
    cache: Vec<Option<(Tensor, Tensor)>>,
    position: usize,
    /// Selected `[layer, head]` pairs and their `[tokens, states]` cross-attention so far.
    alignment: Vec<([usize; 2], Vec<Vec<f32>>)>,
}

impl Decoder<'_> {
    /// Record the cross-attention of `heads` for every token fed from now on.
    pub(super) fn record_alignment(&mut self, heads: &[[usize; 2]]) {
        self.alignment = heads.iter().map(|&pair| (pair, Vec::new())).collect();
    }

    /// `[selected heads][tokens][states]` cross-attention recorded so far.
    pub(super) fn alignment(&self) -> Vec<Vec<Vec<f32>>> {
        self.alignment
            .iter()
            .map(|(_, rows)| rows.clone())
            .collect()
    }

    /// Append layer `layer`'s attention weights of `[tokens, heads · dim]` queries to the
    /// recording. These are computed apart from the fused attention, which keeps none.
    fn record(&mut self, layer: usize, q: &Tensor, k: &Tensor, count: usize) -> Result<()> {
        let dim = q.dim(1)? / count;
        #[allow(clippy::cast_precision_loss, reason = "head widths are small")]
        let scale = (dim as f64).powf(-0.5);
        for (pair, rows) in &mut self.alignment {
            let [selected, head] = *pair;
            if selected != layer {
                continue;
            }
            let keys = k.narrow(1, head * dim, dim)?.t()?.contiguous()?;
            let scores = (q.narrow(1, head * dim, dim)?.contiguous()?.matmul(&keys)? * scale)?;
            rows.extend(softmax_last_dim(&scores)?.to_vec2::<f32>()?);
        }
        Ok(())
    }

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
        #[cfg(feature = "metal")]
        if self.pools.is_some() {
            return self.step_fused(x, tokens.len());
        }
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
            let attended = attend(&q, &k, &v, Some(self.position), head_scale(q.dim(2)?))?;
            self.cache[i] = Some((k, v));
            x = (x + a.out.forward(&merge(&attended)?)?)?;
            let cross = layer
                .cross
                .as_ref()
                .ok_or_else(|| invalid("decoder layer without cross-attention"))?;
            let q = cross.q.forward(&cross.norm.forward(&x)?)?;
            let (k, v) = &self.cross[i];
            let attended = model.attend_whole(&q, k, v, heads)?;
            if !self.alignment.is_empty() {
                let k = k.clone();
                self.record(i, &q, &k, heads)?;
            }
            x = (&x + cross.out.forward(&attended)?)?;
            x = (&x + Whisper::feed_forward(layer, &x)?)?;
        }
        self.position = end;
        model.embedding.forward(&model.decoder_norm.forward(&x)?)
    }
}

#[cfg(feature = "metal")]
impl Decoder<'_> {
    /// The layers over `[tokens, d_model]` rows on the attention kernels: each layer's keys
    /// and values go into its pool, so a step costs the same whatever was decoded before.
    fn step_fused(&mut self, mut x: Tensor, tokens: usize) -> Result<Tensor> {
        use tur_metal::kernels::Heads;
        let model = self.model;
        let (Some(fused), Some(pools)) = (&model.fused, &self.pools) else {
            return Err(invalid("fused decoder step without kernels"));
        };
        let count = model.config.decoder_attention_heads;
        let dim = fused.whole.head_dim;
        let scale = head_scale(dim);
        let at = Heads {
            count,
            dim,
            head_stride: dim,
            col_offset: 0,
        };
        let positions = model.config.max_target_positions;
        let visible = (self.position + 1..=self.position + tokens)
            .map(|v| u32::try_from(v).map_err(|_| invalid("decoder position overflow")))
            .collect::<Result<Vec<_>>>()?;
        let own_kv =
            fused
                .whole
                .kernels
                .paged_kv(positions, positions, &[0], &vec![0; tokens], &visible)?;
        let cross_kv = fused
            .whole
            .whole(tokens, model.config.max_source_positions)?;
        let mut alignment = Vec::new();
        for (i, layer) in model.decoder.iter().enumerate() {
            let a = &layer.attention;
            let h = a.norm.forward(&x)?;
            let (q, k, v) = (a.q.forward(&h)?, a.k.forward(&h)?, a.v.forward(&h)?);
            fused
                .whole
                .kernels
                .kv_write(&k, at, &v, at, &pools.own[i], &own_kv)?;
            x = (x + a
                .out
                .forward(&fused.whole.attend(&q, &pools.own[i], &own_kv, scale)?)?)?;
            let cross = layer
                .cross
                .as_ref()
                .ok_or_else(|| invalid("decoder layer without cross-attention"))?;
            let q = cross.q.forward(&cross.norm.forward(&x)?)?;
            let attended = fused.whole.attend(&q, &pools.cross[i], &cross_kv, scale)?;
            if !self.alignment.is_empty() {
                alignment.push((i, q));
            }
            x = (&x + cross.out.forward(&attended)?)?;
            x = (&x + Whisper::feed_forward(layer, &x)?)?;
        }
        for (i, q) in alignment {
            let k = self.cross[i].0.clone();
            self.record(i, &q, &k, count)?;
        }
        self.position += tokens;
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
            whole: WholeAttention::new(dim, &device)?
                .ok_or_else(|| invalid("no kernels for the head width"))?,
            cross: DType::F32,
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
                head_scale(dim),
            )?)?;
            let kv = fused.whole.whole(queries, keys)?;
            let actual =
                fused
                    .whole
                    .attend(&q, &fused.whole.pool(&k, &v)?, &kv, head_scale(dim))?;
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
