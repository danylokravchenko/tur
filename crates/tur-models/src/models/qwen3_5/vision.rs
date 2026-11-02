#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Qwen3.5's vision tower: patches, attention over each image's whole grid, and the merger
//! that turns each 2×2 block of patches into one soft token.
use super::config::{Vision, VisionConfig};
#[cfg(feature = "metal")]
use crate::models::encoder::WholeAttention;
use crate::{
    image::Image,
    models::{
        Precision,
        artifact::{Companion, Part},
        checkpoint::Checkpoint,
        encoder::{Linear, Loader, Norm, attend_whole, head_scale},
        math::AxialTable,
        weights::tiled,
    },
};
use candle_core::{DType, Device, Tensor};
use std::{collections::BTreeMap, path::Path};
use tur_core::{Result, execution::invalid};

/// `PyTorch`'s `LayerNorm` epsilon as the tower builds it.
const EPS: f32 = 1e-6;
/// Rotary base of the tower's axial positions, which the checkpoint leaves at its default.
const THETA: f64 = 10_000.0;
/// Heads attended at once on the Candle path: bounds the score buffer.
const HEAD_GROUP: usize = 4;
/// Most pixels an image is encoded at, against the checkpoint's 16.7 million: attention grows
/// with the square of the patches, and this size already reads small print. See the book's
/// Images chapter for its cost.
pub(crate) const MAX_PIXELS: usize = 1 << 20;
/// Longest ratio of an image's sides that the reference processor accepts.
const MAX_ASPECT: usize = 200;

/// Settings of the image processor (Transformers' `Qwen2VLImageProcessor`).
#[derive(Clone, Copy, Debug)]
struct Processor {
    min_pixels: usize,
    max_pixels: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl Processor {
    /// Read `preprocessor_config.json`, rejecting a layout the tower doesn't implement.
    /// Without the file, the checkpoint family's shipped values apply.
    fn load(dir: &Path, config: &VisionConfig) -> Result<Self> {
        let mut processor = Self {
            min_pixels: 65_536,
            max_pixels: MAX_PIXELS,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let raw = match std::fs::read_to_string(dir.join("preprocessor_config.json")) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(processor),
            Err(error) => return Err(error.into()),
        };
        let raw: serde_json::Value = serde_json::from_str(&raw)?;
        let edge = |key: &str| {
            raw["size"][key]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
        };
        if let Some(min) = edge("shortest_edge") {
            processor.min_pixels = min;
        }
        if let Some(max) = edge("longest_edge") {
            processor.max_pixels = max.min(MAX_PIXELS);
        }
        let triple = |key: &str, into: &mut [f32; 3]| {
            let Some(values) = raw[key].as_array() else {
                return;
            };
            for (slot, value) in into.iter_mut().zip(values) {
                #[allow(clippy::cast_possible_truncation, reason = "F32 by the reference")]
                if let Some(value) = value.as_f64() {
                    *slot = value as f32;
                }
            }
        };
        triple("image_mean", &mut processor.mean);
        triple("image_std", &mut processor.std);
        let number =
            |key: &str, expected: usize| raw[key].as_u64().is_none_or(|v| v == expected as u64);
        if !number("patch_size", config.patch_size)
            || !number("merge_size", config.spatial_merge_size)
            || !number("temporal_patch_size", config.temporal_patch_size)
            || processor.min_pixels > processor.max_pixels
            || processor.std.contains(&0.0)
        {
            return Err(invalid("unsupported Qwen3.5 image processor settings"));
        }
        Ok(processor)
    }
}

/// Transformers' `smart_resize`: each side a multiple of `factor`, the area inside the
/// bounds, and the aspect ratio kept as closely as that allows.
fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    bounds: (usize, usize),
) -> Result<(usize, usize)> {
    let (min_pixels, max_pixels) = bounds;
    if height.max(width) > MAX_ASPECT * height.min(width) {
        return Err(invalid(format!(
            "image aspect ratio must be under {MAX_ASPECT}; this one is {width}×{height}"
        )));
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "image sides are far below 2^52 and the results are non-negative"
    )]
    let sides = {
        let (h, w, f) = (height as f64, width as f64, factor as f64);
        // Python's `round` breaks ties to even.
        let (bar_h, bar_w) = (
            (h / f).round_ties_even() as usize * factor,
            (w / f).round_ties_even() as usize * factor,
        );
        if bar_h * bar_w > max_pixels {
            let beta = (h * w / max_pixels as f64).sqrt();
            (
                factor.max((h / beta / f).floor() as usize * factor),
                factor.max((w / beta / f).floor() as usize * factor),
            )
        } else if bar_h * bar_w < min_pixels {
            let beta = (min_pixels as f64 / (h * w)).sqrt();
            (
                (h * beta / f).ceil() as usize * factor,
                (w * beta / f).ceil() as usize * factor,
            )
        } else {
            (bar_h, bar_w)
        }
    };
    Ok(sides)
}

struct Block {
    norm1: Norm,
    qkv: Linear,
    proj: Linear,
    norm2: Norm,
    fc1: Linear,
    fc2: Linear,
}

/// The loaded tower and how it reads an image.
pub(super) struct VisionTower {
    config: VisionConfig,
    processor: Processor,
    /// `[hidden, 3 · patch²]`: the 3D patch projection summed over its two frames, since a
    /// still image repeats one frame.
    patch: Linear,
    /// Host `[positions, hidden]`: each image reads its own bilinear resampling.
    positions: Vec<f32>,
    blocks: Vec<Block>,
    merger_norm: Norm,
    merger_fc1: Linear,
    merger_fc2: Linear,
    table: AxialTable,
    #[cfg(feature = "metal")]
    fused: Option<(WholeAttention, tur_metal::kernels::AxialRotation)>,
    device: Device,
    bytes: usize,
}

/// A preprocessed image: patches in merge-block order, each `[C, H, W]` flattened.
pub(super) struct Patches {
    pub(super) pixels: Tensor,
    positions: Vec<u32>,
    rows: usize,
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

/// The tower's tensors as an artifact stores them: projections packed in `precision`, the
/// rest as the checkpoint has them.
pub(super) fn stored_tensors(vision: &Vision, precision: Precision) -> BTreeMap<String, Companion> {
    vision
        .weight_shapes()
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
        &[_, cols] if name.ends_with(".weight") && !name.ends_with("pos_embed.weight") => {
            Some(tiled(precision, cols)).filter(|p| p.ggml().is_some())
        }
        _ => None,
    }
}

/// Resident bytes of the tower in `precision`. The folded patch projection holds half of
/// its checkpoint tensor.
pub(super) fn planned_bytes(vision: &Vision, precision: Precision) -> Result<usize> {
    let mut total = 0usize;
    for (name, shape) in vision.weight_shapes() {
        let bytes = match stored_precision(&name, &shape, precision) {
            Some(packed) => packed.bytes(shape[0], shape[1])?,
            None if name.ends_with("patch_embed.proj.weight") => {
                shape.iter().product::<usize>() * 2
            }
            None => shape.iter().product::<usize>() * 4,
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("vision weight bytes overflow"))?;
    }
    Ok(total)
}

impl VisionTower {
    pub(super) fn load(
        dir: &Path,
        checkpoint: &Checkpoint,
        vision: &Vision,
        precision: Precision,
        device: &Device,
    ) -> Result<Self> {
        let config = vision.config.clone();
        let processor = Processor::load(dir, &config)?;
        let mut packed_cols = BTreeMap::new();
        for (name, shape) in vision.weight_shapes() {
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
            eps: EPS,
            bytes: 0,
            packed_cols,
        };
        let tower = "model.visual";
        let patch_side = config.patch_size;
        let folded = checkpoint
            .take(&format!("{tower}.patch_embed.proj.weight"))?
            .to_dtype(DType::F32)?
            .sum(2)?
            .reshape((config.hidden_size, 3 * patch_side * patch_side))?;
        let patch = Linear {
            weight: tur_metal::packed::PackedLinear::dense(&folded, device)?,
            bias: Some(t.dense(&format!("{tower}.patch_embed.proj.bias"))?),
        };
        t.bytes += folded.elem_count() * 4;
        let positions = checkpoint
            .take(&format!("{tower}.pos_embed.weight"))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        t.bytes += positions.len() * 4;
        let blocks = (0..config.depth)
            .map(|i| {
                let p = format!("{tower}.blocks.{i}");
                Ok(Block {
                    norm1: t.norm(&format!("{p}.norm1"))?,
                    qkv: t.linear(&format!("{p}.attn.qkv"), true)?,
                    proj: t.linear(&format!("{p}.attn.proj"), true)?,
                    norm2: t.norm(&format!("{p}.norm2"))?,
                    fc1: t.linear(&format!("{p}.mlp.linear_fc1"), true)?,
                    fc2: t.linear(&format!("{p}.mlp.linear_fc2"), true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let merger_norm = t.norm(&format!("{tower}.merger.norm"))?;
        let merger_fc1 = t.linear(&format!("{tower}.merger.linear_fc1"), true)?;
        let merger_fc2 = t.linear(&format!("{tower}.merger.linear_fc2"), true)?;
        let bytes = t.bytes;
        let head_dim = config.head_dim();
        let table = AxialTable::patches(head_dim, THETA, head_dim, 1);
        #[cfg(feature = "metal")]
        let fused = WholeAttention::new(head_dim, device)?
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
        Ok(Self {
            config,
            processor,
            patch,
            positions,
            blocks,
            merger_norm,
            merger_fc1,
            merger_fc2,
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

    /// Resized height and width of `image`, each a multiple of a merged block.
    fn target(&self, image: &Image) -> Result<(usize, usize)> {
        smart_resize(
            image.height(),
            image.width(),
            self.config.patch_size * self.config.spatial_merge_size,
            (self.processor.min_pixels, self.processor.max_pixels),
        )
    }

    /// `(rows, columns)` of `image`'s soft tokens.
    pub(super) fn grid(&self, image: &Image) -> Result<(usize, usize)> {
        let (height, width) = self.target(image)?;
        let block = self.config.patch_size * self.config.spatial_merge_size;
        Ok((height / block, width / block))
    }

    /// Resize, normalize and cut `image` into patches, grouped by the block they merge into.
    pub(super) fn patches(&self, image: &Image) -> Result<Patches> {
        let (height, width) = self.target(image)?;
        let side = self.config.patch_size;
        let merge = self.config.spatial_merge_size;
        let (rows, columns) = (height / side, width / side);
        let values = image.resized(height, width)?;
        let (mean, std) = (self.processor.mean, self.processor.std);
        let offset = mean.map(|m| m * 255.0);
        let scale = std.map(|s| 1.0 / (s * 255.0));
        let mut pixels = Vec::with_capacity(rows * columns * 3 * side * side);
        let mut positions = Vec::with_capacity(rows * columns * 2);
        for block_row in 0..rows / merge {
            for block_column in 0..columns / merge {
                for r in block_row * merge..(block_row + 1) * merge {
                    for c in block_column * merge..(block_column + 1) * merge {
                        positions.push(u32::try_from(r).map_err(|_| invalid("patch overflow"))?);
                        positions.push(u32::try_from(c).map_err(|_| invalid("patch overflow"))?);
                        for channel in 0..3 {
                            for y in r * side..(r + 1) * side {
                                let row = &values[y * width * 3..(y + 1) * width * 3];
                                let run = &row[(c * side) * 3..((c + 1) * side) * 3];
                                pixels.extend(
                                    run.iter().skip(channel).step_by(3).map(|&v| {
                                        (f32::from(v) - offset[channel]) * scale[channel]
                                    }),
                                );
                            }
                        }
                    }
                }
            }
        }
        let shape = (rows * columns, 3 * side * side);
        Ok(Patches {
            pixels: Tensor::from_vec(pixels, shape, &self.device)?,
            positions,
            rows,
            columns,
        })
    }

    /// `[patches, hidden]` learned positions, resampled bilinearly with aligned corners from
    /// the checkpoint's square table to this grid.
    fn resampled_positions(&self, patches: &Patches) -> Result<Tensor> {
        let hidden = self.config.hidden_size;
        let side = self.config.position_side();
        #[allow(clippy::cast_precision_loss, reason = "grids are small")]
        let axis = |index: u32, size: usize| -> (usize, usize, f32) {
            let at = index as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
            let floor = at.floor();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "`at` lies in 0..side"
            )]
            let low = (floor as usize).min(side - 1);
            (low, (low + 1).min(side - 1), at - floor)
        };
        let mut out = vec![0f32; patches.positions.len() / 2 * hidden];
        for (row, position) in out
            .chunks_exact_mut(hidden)
            .zip(patches.positions.as_chunks::<2>().0)
        {
            let (top, bottom, down) = axis(position[0], patches.rows);
            let (left, right, across) = axis(position[1], patches.columns);
            for (tap, weight) in [
                (top * side + left, (1.0 - down) * (1.0 - across)),
                (top * side + right, (1.0 - down) * across),
                (bottom * side + left, down * (1.0 - across)),
                (bottom * side + right, down * across),
            ] {
                let source = &self.positions[tap * hidden..(tap + 1) * hidden];
                for (value, &sample) in row.iter_mut().zip(source) {
                    *value = sample.mul_add(weight, *value);
                }
            }
        }
        Ok(Tensor::from_vec(
            out,
            (patches.positions.len() / 2, hidden),
            &self.device,
        )?)
    }

    /// `[patches, hidden]` attention output of `[patches, 3 · hidden]` `q | k | v` rows.
    fn attention(&self, qkv: &Tensor, patches: &Patches) -> Result<Tensor> {
        let hidden = self.config.hidden_size;
        let (count, dim) = (self.config.num_heads, self.config.head_dim());
        let rows = qkv.dim(0)?;
        #[cfg(feature = "metal")]
        if let Some((whole, rotation)) = &self.fused {
            use tur_metal::kernels::Heads;
            let positions = Tensor::from_slice(&patches.positions, (rows, 2), &self.device)?;
            let at = |col_offset| Heads {
                count,
                dim,
                head_stride: dim,
                col_offset,
            };
            let rotate = |offset| -> Result<Tensor> {
                Ok(whole
                    .kernels
                    .axial_rope(qkv, at(offset), None, &positions, rotation)?
                    .reshape((rows, hidden))?)
            };
            let (q, k) = (rotate(0)?, rotate(hidden)?);
            let v = qkv.narrow(1, 2 * hidden, hidden)?.contiguous()?;
            let kv = whole.whole(rows, rows)?;
            return whole.attend(&q, &whole.pool(&k, &v)?, &kv, head_scale(dim));
        }
        let rotate = |offset| -> Result<Tensor> {
            let rows3 = qkv
                .narrow(1, offset, hidden)?
                .contiguous()?
                .reshape((rows, count, dim))?;
            Ok(self
                .table
                .rotate(&rows3, &patches.positions, 2)?
                .reshape((rows, hidden))?)
        };
        let (q, k) = (rotate(0)?, rotate(hidden)?);
        let v = qkv.narrow(1, 2 * hidden, hidden)?.contiguous()?;
        attend_whole((&q, &k, &v), count, HEAD_GROUP, head_scale(dim))
    }

    /// `[patches, hidden]` final states and `[soft tokens, text hidden]` merged rows.
    pub(super) fn forward(&self, patches: &Patches) -> Result<(Tensor, Tensor)> {
        let mut x = self
            .patch
            .forward(&patches.pixels)?
            .broadcast_add(&self.resampled_positions(patches)?)?;
        for block in &self.blocks {
            let attended =
                self.attention(&block.qkv.forward(&block.norm1.forward(&x)?)?, patches)?;
            x = (x + block.proj.forward(&attended)?)?;
            let up = block.fc1.forward(&block.norm2.forward(&x)?)?.gelu()?;
            x = (&x + block.fc2.forward(&up)?)?;
        }
        let units = self.config.spatial_merge_size * self.config.spatial_merge_size;
        let merged = self
            .merger_norm
            .forward(&x)?
            .reshape((x.dim(0)? / units, self.config.hidden_size * units))?;
        let up = self.merger_fc1.forward(&merged)?.gelu_erf()?;
        Ok((x, self.merger_fc2.forward(&up)?))
    }
}
