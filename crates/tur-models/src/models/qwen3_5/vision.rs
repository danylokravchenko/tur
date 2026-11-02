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
    video::Video,
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
/// Most pixels across a video's frames, against the checkpoint's 25 million: 2048 soft tokens
/// at most, which with labels stays within a default context.
const MAX_VIDEO_PIXELS: usize = 1 << 22;
/// Most frames sampled from a video. With the budget above, this many frames are still 181
/// pixels on a side; longer videos are sampled more sparsely than the checkpoint's 2 per second.
const MAX_VIDEO_FRAMES: usize = 128;
/// Fewest frames sampled from a video, as the reference does.
const MIN_VIDEO_FRAMES: usize = 4;
/// Frames sampled per second of video, the reference's default.
const VIDEO_FPS: f64 = 2.0;
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

/// Pixel bounds across a video's frames from `video_preprocessor_config.json`, within Tur's
/// limit; the checkpoint family's shipped minimum without the file.
fn video_bounds(dir: &Path) -> Result<(usize, usize)> {
    let raw = match std::fs::read_to_string(dir.join("video_preprocessor_config.json")) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((4096, MAX_VIDEO_PIXELS));
        }
        Err(error) => return Err(error.into()),
    };
    let raw: serde_json::Value = serde_json::from_str(&raw)?;
    let edge = |key: &str| {
        raw["size"][key]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
    };
    Ok((
        edge("shortest_edge").unwrap_or(4096),
        edge("longest_edge").map_or(MAX_VIDEO_PIXELS, |max| max.min(MAX_VIDEO_PIXELS)),
    ))
}

/// Transformers' `smart_resize`: each side a multiple of `factor`, the area inside the
/// bounds, and the aspect ratio kept as closely as that allows.
fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    bounds: (usize, usize),
) -> Result<(usize, usize)> {
    resize_frames(1, height, width, factor, bounds)
}

/// `smart_resize` over `frames` frames, whose pixels the bounds count together. A video's frames
/// round up to pairs, as its temporal patches do. Sides under `factor` scale up first.
fn resize_frames(
    frames: usize,
    height: usize,
    width: usize,
    factor: usize,
    bounds: (usize, usize),
) -> Result<(usize, usize)> {
    let (min_pixels, max_pixels) = bounds;
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "image sides are far below 2^52 and the results are non-negative"
    )]
    let (height, width) = if frames > 1 && height.min(width) < factor {
        let scale = factor as f64 / height.min(width) as f64;
        (
            (height as f64 * scale) as usize,
            (width as f64 * scale) as usize,
        )
    } else {
        (height, width)
    };
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
        let count = frames as f64;
        let rounded = if frames > 1 {
            (count / 2.0).round_ties_even() as usize * 2
        } else {
            1
        };
        // Python's `round` breaks ties to even.
        let (bar_h, bar_w) = (
            (h / f).round_ties_even() as usize * factor,
            (w / f).round_ties_even() as usize * factor,
        );
        if rounded * bar_h * bar_w > max_pixels {
            let beta = (count * h * w / max_pixels as f64).sqrt();
            (
                factor.max((h / beta / f).floor() as usize * factor),
                factor.max((w / beta / f).floor() as usize * factor),
            )
        } else if rounded * bar_h * bar_w < min_pixels {
            let beta = (min_pixels as f64 / (count * h * w)).sqrt();
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
    /// Pixel bounds across a video's frames.
    video_pixels: (usize, usize),
    /// `[hidden, 3 · 2 · patch²]`: the 3D patch projection over a step's two frames.
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

/// Preprocessed frames: patches by step, then merge block, each `[C, T, H, W]` flattened.
pub(super) struct Patches {
    pub(super) pixels: Tensor,
    positions: Vec<u32>,
    rows: usize,
    columns: usize,
    /// Temporal steps. Attention stays within a step, so patches of two frames of a video
    /// never see each other until the text model reads them.
    steps: usize,
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

/// Resident bytes of the tower in `precision`.
pub(super) fn planned_bytes(vision: &Vision, precision: Precision) -> Result<usize> {
    let mut total = 0usize;
    for (name, shape) in vision.weight_shapes() {
        let bytes = match stored_precision(&name, &shape, precision) {
            Some(packed) => packed.bytes(shape[0], shape[1])?,
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
        let video_pixels = video_bounds(dir)?;
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
        let flat = checkpoint
            .take(&format!("{tower}.patch_embed.proj.weight"))?
            .to_dtype(DType::F32)?
            .reshape((config.hidden_size, 3 * 2 * patch_side * patch_side))?;
        let patch = Linear {
            weight: tur_metal::packed::PackedLinear::dense(&flat, device)?,
            bias: Some(t.dense(&format!("{tower}.patch_embed.proj.bias"))?),
        };
        t.bytes += flat.elem_count() * 4;
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
            video_pixels,
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

    /// Resize, normalize and cut `image` into patches: one step of two equal frames.
    pub(super) fn patches(&self, image: &Image) -> Result<Patches> {
        let (height, width) = self.target(image)?;
        let values = image.resized(height, width)?;
        self.frame_patches(&[&values, &values], height, width)
    }

    /// Normalize and cut resized frames, two per step, into patches grouped by step and the
    /// block they merge into.
    fn frame_patches(&self, frames: &[&[u8]], height: usize, width: usize) -> Result<Patches> {
        let side = self.config.patch_size;
        let merge = self.config.spatial_merge_size;
        let (rows, columns) = (height / side, width / side);
        let steps = frames.len() / 2;
        let offset = self.processor.mean.map(|m| m * 255.0);
        let scale = self.processor.std.map(|s| 1.0 / (s * 255.0));
        let patches = steps * rows * columns;
        let mut pixels = Vec::with_capacity(patches * 3 * 2 * side * side);
        let mut positions = Vec::with_capacity(patches * 2);
        for step in frames.as_chunks::<2>().0 {
            for block_row in 0..rows / merge {
                for block_column in 0..columns / merge {
                    for r in block_row * merge..(block_row + 1) * merge {
                        for c in block_column * merge..(block_column + 1) * merge {
                            positions
                                .push(u32::try_from(r).map_err(|_| invalid("patch overflow"))?);
                            positions
                                .push(u32::try_from(c).map_err(|_| invalid("patch overflow"))?);
                            for channel in 0..3 {
                                for frame in step {
                                    for y in r * side..(r + 1) * side {
                                        let row = &frame[y * width * 3..(y + 1) * width * 3];
                                        let run = &row[(c * side) * 3..((c + 1) * side) * 3];
                                        pixels.extend(run.iter().skip(channel).step_by(3).map(
                                            |&v| (f32::from(v) - offset[channel]) * scale[channel],
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Patches {
            pixels: Tensor::from_vec(pixels, (patches, 3 * 2 * side * side), &self.device)?,
            positions,
            rows,
            columns,
            steps,
        })
    }

    /// Frames sampled from a video: uniformly at the reference's rate, within the frame bounds.
    fn video_indices(video: &Video) -> Vec<usize> {
        let total = video.frame_count();
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "frame counts are bounded far below 2^52"
        )]
        let wanted = (total as f64 / video.fps() * VIDEO_FPS) as usize;
        let count = wanted.clamp(MIN_VIDEO_FRAMES, MAX_VIDEO_FRAMES).min(total);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "frame counts are bounded far below 2^52"
        )]
        (0..count)
            .map(|at| {
                let share = (total - 1) as f64 * at as f64 / (count - 1).max(1) as f64;
                share.round_ties_even() as usize
            })
            .collect()
    }

    fn video_target(&self, video: &Video, frames: usize) -> Result<(usize, usize)> {
        resize_frames(
            frames,
            video.height(),
            video.width(),
            self.config.patch_size * self.config.spatial_merge_size,
            self.video_pixels,
        )
    }

    /// Soft tokens of each step of `video`, in a grid, and the timestamp its text carries.
    pub(super) fn video_steps(&self, video: &Video) -> Result<(Vec<f64>, (usize, usize))> {
        let mut indices = Self::video_indices(video);
        let (height, width) = self.video_target(video, indices.len())?;
        if !indices.len().is_multiple_of(2) {
            indices.extend(indices.last().copied());
        }
        #[allow(clippy::cast_precision_loss, reason = "frame counts are small")]
        let stamps = indices
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0] as f64 + pair[1] as f64) / video.fps() / 2.0)
            .collect();
        let block = self.config.patch_size * self.config.spatial_merge_size;
        Ok((stamps, (height / block, width / block)))
    }

    /// Decode, resize and cut a video's sampled frames into patches.
    pub(super) fn video_patches(&self, video: &Video) -> Result<Patches> {
        let indices = Self::video_indices(video);
        let (height, width) = self.video_target(video, indices.len())?;
        let mut resized: Vec<Vec<u8>> = Vec::with_capacity(indices.len() + 1);
        video.each_frame(&indices, |frame| {
            resized.push(crate::image::resize::bicubic(
                &frame,
                (video.height(), video.width()),
                (height, width),
            ));
            Ok(())
        })?;
        // An odd frame count repeats the last frame to fill its step.
        if let Some(last) = resized.last().filter(|_| !indices.len().is_multiple_of(2)) {
            resized.push(last.clone());
        }
        let frames: Vec<&[u8]> = resized.iter().map(Vec::as_slice).collect();
        self.frame_patches(&frames, height, width)
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
            let per_step = rows / patches.steps;
            let kv = whole.whole(per_step, per_step)?;
            let attended = (0..patches.steps)
                .map(|step| {
                    let part = |x: &Tensor| -> Result<Tensor> {
                        Ok(x.narrow(0, step * per_step, per_step)?.contiguous()?)
                    };
                    whole.attend(
                        &part(&q)?,
                        &whole.pool(&part(&k)?, &part(&v)?)?,
                        &kv,
                        head_scale(dim),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Tensor::cat(&attended, 0)?);
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
        let per_step = rows / patches.steps;
        let attended = (0..patches.steps)
            .map(|step| {
                let part = |x: &Tensor| x.narrow(0, step * per_step, per_step);
                attend_whole(
                    (&part(&q)?, &part(&k)?, &part(&v)?),
                    count,
                    HEAD_GROUP,
                    head_scale(dim),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&attended, 0)?)
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
