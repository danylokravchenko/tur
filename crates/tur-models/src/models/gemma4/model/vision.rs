#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Gemma 4's vision tower: patches with learned 2D positions, attention over the whole image,
//! and average pooling of each 3×3 block into one soft token.
use super::{
    super::config::{Vision, VisionConfig},
    audio::{Clipped, clipped_tensor_names},
};
#[cfg(feature = "metal")]
use crate::models::encoder::WholeAttention;
use crate::{
    image::Image,
    models::{
        Precision,
        artifact::{Companion, Part},
        checkpoint::Checkpoint,
        encoder::{Loader, attend_whole},
        math::{self, AxialTable},
        weights::tiled,
    },
};
use candle_core::{DType, Device, Tensor};
use std::{collections::BTreeMap, path::Path};
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

/// Heads attended at once on the Candle path: bounds the score buffer.
const HEAD_GROUP: usize = 4;
/// `do_rescale`'s factor, in F32 as the reference multiplies by it.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the reference stores this factor in F64 and scales F32 pixels"
)]
const RESCALE: f32 = (1.0 / 255.0) as f32;

struct Layer {
    input_norm: Tensor,
    q: Clipped,
    k: Clipped,
    v: Clipped,
    o: Clipped,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_norm: Tensor,
    pre_feedforward_norm: Tensor,
    gate: Clipped,
    up: Clipped,
    down: Clipped,
    post_feedforward_norm: Tensor,
}

/// The loaded tower, its projector into the text model and how they read an image.
pub(super) struct VisionTower {
    config: VisionConfig,
    max_patches: usize,
    input: PackedLinear,
    /// `[size, hidden]` learned positions of the x and y axes.
    positions: [Tensor; 2],
    layers: Vec<Layer>,
    /// `standardize`'s bias and scale, in the checkpoints that have it.
    standard: Option<(Tensor, Tensor)>,
    projection: PackedLinear,
    /// A unit scale for norms the reference builds without a weight.
    unit: Tensor,
    table: AxialTable,
    #[cfg(feature = "metal")]
    fused: Option<(WholeAttention, tur_metal::kernels::AxialRotation)>,
    device: Device,
    bytes: usize,
}

/// A preprocessed image: patches in raster order, each interleaved RGB.
pub(super) struct Patches {
    pub(super) pixels: Tensor,
    /// `(x, y)` of each patch.
    positions: Vec<u32>,
    columns: usize,
}

/// Storage of the tower's projections: F32 beside an F32 text model, `Q8_0` otherwise.
pub(super) fn precision(text_f32: bool) -> Precision {
    if text_f32 {
        Precision::F32
    } else {
        Precision::Q8_0
    }
}

/// The tower's tensors as an artifact stores them: projections packed in `precision`, the rest
/// as the checkpoint has them.
pub(super) fn stored_tensors(
    vision: &Vision,
    text_hidden: usize,
    precision: Precision,
) -> BTreeMap<String, Companion> {
    vision
        .weight_shapes(text_hidden)
        .into_iter()
        .map(|(name, shape)| {
            let packed = stored_precision(&name, &shape, precision);
            (
                name,
                Companion {
                    shape,
                    packed,
                    part: Part::Vision,
                },
            )
        })
        .collect()
}

fn stored_precision(name: &str, shape: &[usize], precision: Precision) -> Option<Precision> {
    match shape {
        &[_, cols] if name.ends_with(".weight") => {
            Some(tiled(precision, cols)).filter(|p| p.ggml().is_some())
        }
        _ => None,
    }
}

/// Resident bytes of the tower in `precision`.
pub(super) fn planned_bytes(
    vision: &Vision,
    text_hidden: usize,
    precision: Precision,
) -> Result<usize> {
    let mut total = 0usize;
    for (name, shape) in vision.weight_shapes(text_hidden) {
        let bytes = match stored_precision(&name, &shape, precision) {
            Some(packed) => packed.bytes(shape[0], shape[1])?,
            // Clip bounds are read into the layers rather than kept as tensors.
            None if shape.is_empty() => 0,
            None => shape.iter().product::<usize>() * 4,
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("vision weight bytes overflow"))?;
    }
    Ok(total)
}

/// Soft tokens per image, from `processor_config.json`'s image processor, or the shipped
/// value without one. Other settings the tower doesn't implement are rejected.
fn max_soft_tokens(dir: &Path, config: &VisionConfig) -> Result<usize> {
    let raw = match std::fs::read_to_string(dir.join("processor_config.json")) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(280),
        Err(error) => return Err(error.into()),
    };
    let raw: serde_json::Value = serde_json::from_str(&raw)?;
    let processor = &raw["image_processor"];
    let matches =
        |key: &str, expected: usize| processor[key].as_u64().is_none_or(|v| v == expected as u64);
    let tokens = processor["max_soft_tokens"].as_u64().unwrap_or(280);
    if !matches("patch_size", config.patch_size)
        || !matches("pooling_kernel_size", config.pooling_kernel_size)
        || ![70, 140, 280, 560, 1120].contains(&tokens)
    {
        return Err(invalid("unsupported Gemma 4 image processor settings"));
    }
    usize::try_from(tokens).map_err(|_| invalid("soft token count overflow"))
}

/// Transformers' `get_aspect_ratio_preserving_size`: the largest size with the image's aspect
/// ratio, at most `max_patches` patches and sides that are multiples of a pooled block.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "image sides are far below 2^52 and every value is non-negative"
)]
fn target_size(
    height: usize,
    width: usize,
    patch: usize,
    pool: usize,
    max_patches: usize,
) -> (usize, usize) {
    let (h, w) = (height as f64, width as f64);
    let factor = ((max_patches * patch * patch) as f64 / (h * w)).sqrt();
    let block = pool * patch;
    let mut target_height = (factor * h / block as f64).floor() as usize * block;
    let mut target_width = (factor * w / block as f64).floor() as usize * block;
    let longest = (max_patches / (pool * pool)) * block;
    if target_height == 0 {
        target_height = block;
        target_width = (width / height.max(1) * block).clamp(block, longest);
    } else if target_width == 0 {
        target_width = block;
        target_height = (height / width.max(1) * block).clamp(block, longest);
    }
    (target_height, target_width)
}

impl VisionTower {
    pub(super) fn load(
        dir: &Path,
        checkpoint: &Checkpoint,
        vision: &Vision,
        text_hidden: usize,
        precision: Precision,
        device: &Device,
    ) -> Result<Self> {
        let config = vision.config.clone();
        let tokens = max_soft_tokens(dir, &config)?;
        let mut packed_cols = BTreeMap::new();
        for (name, shape) in vision.weight_shapes(text_hidden) {
            let entry = checkpoint.entry(&name)?;
            if let (safetensors::Dtype::U8, &[_, cols]) = (entry.dtype, shape.as_slice()) {
                packed_cols.insert(name, cols);
            } else if entry.shape != shape {
                return Err(invalid(format!(
                    "{name} has shape {:?}, expected {shape:?}",
                    entry.shape
                )));
            }
        }
        let mut t = Loader {
            checkpoint,
            precision,
            device,
            eps: 0.0,
            bytes: 0,
            packed_cols,
        };
        let tower = "model.vision_tower";
        let input = t.projection(&format!("{tower}.patch_embedder.input_proj.weight"))?;
        let table = t.dense(&format!("{tower}.patch_embedder.position_embedding_table"))?;
        let positions = [
            table.narrow(0, 0, 1)?.squeeze(0)?,
            table.narrow(0, 1, 1)?.squeeze(0)?,
        ];
        let clip = config.use_clipped_linears;
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                let p = format!("{tower}.encoder.layers.{i}");
                let a = format!("{p}.self_attn");
                Ok(Layer {
                    input_norm: t.dense(&format!("{p}.input_layernorm.weight"))?,
                    q: t.clipped(&format!("{a}.q_proj"), clip)?,
                    k: t.clipped(&format!("{a}.k_proj"), clip)?,
                    v: t.clipped(&format!("{a}.v_proj"), clip)?,
                    o: t.clipped(&format!("{a}.o_proj"), clip)?,
                    q_norm: t.dense(&format!("{a}.q_norm.weight"))?,
                    k_norm: t.dense(&format!("{a}.k_norm.weight"))?,
                    post_attention_norm: t
                        .dense(&format!("{p}.post_attention_layernorm.weight"))?,
                    pre_feedforward_norm: t
                        .dense(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                    gate: t.clipped(&format!("{p}.mlp.gate_proj"), clip)?,
                    up: t.clipped(&format!("{p}.mlp.up_proj"), clip)?,
                    down: t.clipped(&format!("{p}.mlp.down_proj"), clip)?,
                    post_feedforward_norm: t
                        .dense(&format!("{p}.post_feedforward_layernorm.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let standard = config
            .standardize
            .then(|| {
                Ok::<_, tur_core::TurError>((
                    t.dense(&format!("{tower}.std_bias"))?,
                    t.dense(&format!("{tower}.std_scale"))?,
                ))
            })
            .transpose()?;
        let projection = t.projection("model.embed_vision.embedding_projection.weight")?;
        let bytes = t.bytes;
        let dim = config.head_dim;
        let table = AxialTable::patches(dim, config.theta, dim / 2, 2);
        #[cfg(feature = "metal")]
        let fused = WholeAttention::new(dim, device)?
            .map(|whole| {
                let rotation = whole.kernels.axial_rotation(
                    &table.inv_freq,
                    &table.axis,
                    table.run,
                    table.runs,
                )?;
                Ok::<_, tur_core::TurError>((whole, rotation))
            })
            .transpose()?;
        let max_patches = tokens * config.pooling_kernel_size * config.pooling_kernel_size;
        Ok(Self {
            config,
            max_patches,
            input,
            positions,
            layers,
            standard,
            projection,
            unit: Tensor::ones(1, DType::F32, device)?,
            table,
            #[cfg(feature = "metal")]
            fused,
            device: device.clone(),
            bytes,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }

    fn target(&self, image: &Image) -> (usize, usize) {
        target_size(
            image.height(),
            image.width(),
            self.config.patch_size,
            self.config.pooling_kernel_size,
            self.max_patches,
        )
    }

    /// Soft tokens of `image`: one per pooled block.
    pub(super) fn soft_tokens(&self, image: &Image) -> usize {
        let (height, width) = self.target(image);
        let block = self.config.patch_size * self.config.pooling_kernel_size;
        height / block * (width / block)
    }

    /// Resize and cut `image` into patches, with values in 0..=1.
    pub(super) fn patches(&self, image: &Image) -> Result<Patches> {
        let (height, width) = self.target(image);
        let side = self.config.patch_size;
        let (rows, columns) = (height / side, width / side);
        let values = image.resized(height, width)?;
        let mut pixels = Vec::with_capacity(rows * columns * 3 * side * side);
        let mut positions = Vec::with_capacity(rows * columns * 2);
        for r in 0..rows {
            for c in 0..columns {
                positions.push(u32::try_from(c).map_err(|_| invalid("patch overflow"))?);
                positions.push(u32::try_from(r).map_err(|_| invalid("patch overflow"))?);
                for y in r * side..(r + 1) * side {
                    let row = &values[y * width * 3..(y + 1) * width * 3];
                    pixels.extend(
                        row[(c * side) * 3..((c + 1) * side) * 3]
                            .iter()
                            .map(|&v| f32::from(v) * RESCALE),
                    );
                }
            }
        }
        let shape = (rows * columns, 3 * side * side);
        Ok(Patches {
            pixels: Tensor::from_vec(pixels, shape, &self.device)?,
            positions,
            columns,
        })
    }

    /// `[patches, hidden]` projected patches with their learned positions.
    fn embedded(&self, patches: &Patches) -> Result<Tensor> {
        let scaled = ((&patches.pixels - 0.5)? * 2.0)?;
        let index = |axis: usize| -> Result<Tensor> {
            let values: Vec<u32> = patches
                .positions
                .as_chunks::<2>()
                .0
                .iter()
                .map(|at| at[axis])
                .collect();
            Ok(Tensor::from_vec(
                values,
                patches.positions.len() / 2,
                &self.device,
            )?)
        };
        let x = self.positions[0].index_select(&index(0)?, 0)?;
        let y = self.positions[1].index_select(&index(1)?, 0)?;
        Ok((self.input.forward(&scaled)? + x + y)?)
    }

    /// `[patches, hidden]` attention output of normalized rows.
    fn attention(&self, layer: &Layer, h: &Tensor, patches: &Patches) -> Result<Tensor> {
        let eps = self.config.rms_norm_eps;
        let rows = h.dim(0)?;
        let (count, dim) = (self.config.num_attention_heads, self.config.head_dim);
        let width = count * dim;
        let q = layer.q.forward(h)?;
        let k = layer.k.forward(h)?;
        let v = layer.v.forward(h)?.reshape((rows, count, dim))?;
        let v = math::rms(&v, &self.unit, eps)?.reshape((rows, width))?;
        #[cfg(feature = "metal")]
        if let Some((whole, rotation)) = &self.fused {
            use tur_metal::kernels::Heads;
            let positions = Tensor::from_slice(&patches.positions, (rows, 2), &self.device)?;
            let at = Heads {
                count,
                dim,
                head_stride: dim,
                col_offset: 0,
            };
            let rotate = |x: &Tensor, norm: &Tensor| -> Result<Tensor> {
                Ok(whole
                    .kernels
                    .axial_rope(x, at, Some((norm, eps)), &positions, rotation)?
                    .reshape((rows, width))?)
            };
            let (q, k) = (rotate(&q, &layer.q_norm)?, rotate(&k, &layer.k_norm)?);
            let kv = whole.whole(rows, rows)?;
            let attended = whole.attend(&q, &whole.pool(&k, &v)?, &kv, 1.0)?;
            return layer.o.forward(&attended);
        }
        let rotate = |x: &Tensor, norm: &Tensor| -> Result<Tensor> {
            let normed = math::rms(&x.reshape((rows, count, dim))?, norm, eps)?;
            Ok(self
                .table
                .rotate(&normed, &patches.positions, 2)?
                .reshape((rows, width))?)
        };
        let (q, k) = (rotate(&q, &layer.q_norm)?, rotate(&k, &layer.k_norm)?);
        layer
            .o
            .forward(&attend_whole((&q, &k, &v), count, HEAD_GROUP, 1.0)?)
    }

    /// Average each pooled block of patches, and scale as the pooler does.
    fn pooled(&self, x: &Tensor, patches: &Patches) -> Result<Tensor> {
        let k = self.config.pooling_kernel_size;
        let (rows, hidden) = x.dims2()?;
        let values = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let pooled_columns = patches.columns / k;
        let tokens = rows / (k * k);
        let mut out = vec![0f32; tokens * hidden];
        for (row, at) in values
            .chunks_exact(hidden)
            .zip(patches.positions.as_chunks::<2>().0)
        {
            let token = at[0] as usize / k + pooled_columns * (at[1] as usize / k);
            let target = out
                .get_mut(token * hidden..(token + 1) * hidden)
                .ok_or_else(|| invalid("patch outside its pooled grid"))?;
            for (value, &sample) in target.iter_mut().zip(row) {
                *value += sample;
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "widths are small")]
        let scale = (hidden as f32).sqrt() / (k * k) as f32;
        for value in &mut out {
            *value *= scale;
        }
        Ok(Tensor::from_vec(out, (tokens, hidden), &self.device)?)
    }

    /// `[soft tokens, hidden]` pooled states and `[soft tokens, text hidden]` projected rows.
    pub(super) fn forward(&self, patches: &Patches) -> Result<(Tensor, Tensor)> {
        let eps = self.config.rms_norm_eps;
        let mut x = self.embedded(patches)?;
        for layer in &self.layers {
            let h = math::rms(&x, &layer.input_norm, eps)?;
            let attended = self.attention(layer, &h, patches)?;
            x = (x + math::rms(&attended, &layer.post_attention_norm, eps)?)?;
            let h = math::rms(&x, &layer.pre_feedforward_norm, eps)?;
            let gated = (layer.gate.forward(&h)?.gelu()? * layer.up.forward(&h)?)?;
            let mlp = layer.down.forward(&gated)?;
            x = (&x + math::rms(&mlp, &layer.post_feedforward_norm, eps)?)?;
        }
        let mut pooled = self.pooled(&x, patches)?;
        if let Some((bias, scale)) = &self.standard {
            pooled = pooled.broadcast_sub(bias)?.broadcast_mul(scale)?;
        }
        let normed = math::rms(&pooled, &self.unit, eps)?;
        Ok((pooled, self.projection.forward(&normed)?))
    }
}

/// Names of one clipped projection's tensors, for the tower's shape list.
pub(in crate::models::gemma4) fn clipped_shapes(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    name: &str,
    rows: usize,
    cols: usize,
    clip: bool,
) {
    shapes.insert(format!("{name}.linear.weight"), vec![rows, cols]);
    if clip {
        for bound in clipped_tensor_names(name) {
            shapes.insert(bound, Vec::new());
        }
    }
}
