#![forbid(unsafe_code)]
//! Projection weights kept in their packed storage format.

use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{GgmlDType, QMatMul, QStorage, QTensor},
};
use tur_core::{Result, execution::invalid};

enum Weights {
    /// `[out, in]`, multiplied through a transposed view; may share storage with another
    /// tensor (a tied embedding) because it is never copied.
    Dense(Tensor),
    Packed {
        matmul: QMatMul,
        format: GgmlDType,
        /// Handle to the same `MTLBuffer` the `QMatMul` owns, for the packed GEMV kernels.
        /// `matmul` keeps Candle's pooled `Arc` alive, so the pool cannot recycle it while
        /// this handle exists.
        #[cfg(feature = "metal")]
        buffer: Option<candle_metal_kernels::metal::Buffer>,
    },
    /// MXFP4 blocks then one scale byte per block, read only by the expert kernels. The
    /// `Arc` is what keeps Candle's pool from recycling the buffer.
    #[cfg(feature = "metal")]
    Mxfp4 {
        buffer: std::sync::Arc<candle_metal_kernels::metal::Buffer>,
        scales: usize,
    },
    /// GGML blocks in an arena the expert cache fills; the expert kernels read them in place.
    #[cfg(feature = "metal")]
    Arena {
        buffer: std::sync::Arc<candle_metal_kernels::metal::Buffer>,
        format: GgmlDType,
    },
}

/// A `[out, in]` projection whose weights stay packed on the device; never expanded.
pub struct PackedLinear {
    weights: Weights,
    out_features: usize,
    in_features: usize,
    resident_bytes: usize,
}

impl PackedLinear {
    /// Full-precision F32 weights for reference execution. An F32 tensor already on
    /// `device` is shared, not copied; one uploaded from the host is copied as just its
    /// own elements, since a transfer would copy a view's whole underlying storage.
    pub fn dense(weight: &Tensor, device: &Device) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        let weight = if weight.device().same_device(device) {
            weight.to_dtype(DType::F32)?
        } else {
            host_f32(weight)?.to_device(device)?
        };
        Ok(Self {
            weights: Weights::Dense(weight),
            out_features,
            in_features,
            resident_bytes: out_features * in_features * 4,
        })
    }

    /// Quantize on the host into GGML blocks of `format` and upload only the packed bytes.
    pub fn quantized(weight: &Tensor, format: GgmlDType, device: &Device) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        let values = host_f32(weight)?.flatten_all()?.to_vec1::<f32>()?;
        let packed = crate::quantize::quantize(&values, in_features, format, None)?;
        Self::from_blocks(&packed, format, (out_features, in_features), device)
    }

    /// Upload already packed GGML blocks of `format` for an `[out, in]` projection; the
    /// byte count must match the shape exactly.
    pub fn from_blocks(
        bytes: &[u8],
        format: GgmlDType,
        (out_features, in_features): (usize, usize),
        device: &Device,
    ) -> Result<Self> {
        if out_features == 0
            || bytes.len() != crate::quantize::packed_bytes(format, out_features, in_features)?
        {
            return Err(invalid(format!(
                "{} packed bytes do not form a [{out_features}, {in_features}] {format:?} projection",
                bytes.len()
            )));
        }
        let (qtensor, buffer) = upload_bytes(bytes, format, (out_features, in_features), device)?;
        #[cfg(not(feature = "metal"))]
        let () = buffer;
        Ok(Self {
            weights: Weights::Packed {
                matmul: QMatMul::from_qtensor(qtensor)?,
                format,
                #[cfg(feature = "metal")]
                buffer,
            },
            out_features,
            in_features,
            resident_bytes: bytes.len(),
        })
    }

    /// A `[rows, cols]` expert bank in MXFP4 on `device`: read in place by the expert
    /// kernels on Metal, and decoded to F32 for host reference execution.
    pub fn mxfp4(
        blocks: &[u8],
        scales: &[u8],
        (rows, cols): (usize, usize),
        device: &Device,
    ) -> Result<Self> {
        let count = rows
            .checked_mul(cols / crate::quantize::MXFP4_BLOCK_VALUES)
            .filter(|_| cols.is_multiple_of(crate::quantize::MXFP4_BLOCK_VALUES))
            .filter(|&count| count > 0 && scales.len() == count)
            .ok_or_else(|| invalid("MXFP4 scales do not form the projection"))?;
        if blocks.len() / crate::quantize::MXFP4_BLOCK_BYTES != count {
            return Err(invalid("MXFP4 blocks do not form the projection"));
        }
        match device {
            #[cfg(feature = "metal")]
            Device::Metal(metal) => {
                let buffer = metal.new_buffer_with_data(&[blocks, scales].concat())?;
                Ok(Self {
                    weights: Weights::Mxfp4 {
                        buffer,
                        scales: blocks.len(),
                    },
                    out_features: rows,
                    in_features: cols,
                    resident_bytes: blocks.len() + scales.len(),
                })
            }
            _ => Self::dense(
                &Tensor::from_vec(
                    crate::quantize::mxfp4_decode(blocks, scales)?,
                    (rows, cols),
                    &Device::Cpu,
                )?,
                device,
            ),
        }
    }

    /// Weights over an arena the expert cache fills after this returns.
    #[cfg(feature = "metal")]
    pub(crate) fn mxfp4_arena(
        buffer: std::sync::Arc<candle_metal_kernels::metal::Buffer>,
        scales: usize,
        (rows, cols): (usize, usize),
    ) -> Result<Self> {
        let resident_bytes = buffer.length();
        if rows == 0 || scales >= resident_bytes {
            return Err(invalid("expert arena has no room for its scales"));
        }
        Ok(Self {
            weights: Weights::Mxfp4 { buffer, scales },
            out_features: rows,
            in_features: cols,
            resident_bytes,
        })
    }

    /// GGML-block weights over an arena the expert cache fills after this returns.
    #[cfg(feature = "metal")]
    pub(crate) fn packed_arena(
        buffer: std::sync::Arc<candle_metal_kernels::metal::Buffer>,
        format: GgmlDType,
        (rows, cols): (usize, usize),
    ) -> Result<Self> {
        let resident_bytes = crate::quantize::packed_bytes(format, rows, cols)?;
        if rows == 0 || buffer.length() < resident_bytes {
            return Err(invalid("expert arena is smaller than its projection"));
        }
        Ok(Self {
            weights: Weights::Arena { buffer, format },
            out_features: rows,
            in_features: cols,
            resident_bytes,
        })
    }

    /// Another handle to the same arena, for each layer that reads it. Other weights have
    /// no shared storage to hand out.
    #[cfg(feature = "metal")]
    pub(crate) fn arena_handle(&self) -> Result<Self> {
        let weights = match &self.weights {
            Weights::Arena { buffer, format } => Weights::Arena {
                buffer: std::sync::Arc::clone(buffer),
                format: *format,
            },
            Weights::Mxfp4 { buffer, scales } => Weights::Mxfp4 {
                buffer: std::sync::Arc::clone(buffer),
                scales: *scales,
            },
            _ => return Err(invalid("only arena weights can be shared")),
        };
        Ok(Self {
            weights,
            out_features: self.out_features,
            in_features: self.in_features,
            resident_bytes: self.resident_bytes,
        })
    }

    /// Stack `[rows, in]` parts into one `[Σ rows, in]` projection, converting and
    /// quantizing one part at a time so the host never holds more than one full-precision
    /// part. GGML blocks never span rows, so the packed bytes equal quantizing the whole
    /// matrix. `format` `None` keeps dense F32 weights.
    pub fn stacked(
        parts: impl IntoIterator<Item = Result<Tensor>>,
        format: Option<GgmlDType>,
        device: &Device,
    ) -> Result<Self> {
        let (mut out_features, mut in_features) = (0, None);
        let (mut dense, mut packed) = (Vec::new(), Vec::new());
        for part in parts {
            let part = part?;
            let (rows, width) = part.dims2()?;
            if *in_features.get_or_insert(width) != width {
                return Err(invalid("stacked projection parts differ in input width"));
            }
            out_features += rows;
            let host = host_f32(&part)?;
            match format {
                None => dense.extend(host.flatten_all()?.to_vec1::<f32>()?),
                Some(format) => {
                    packed.extend_from_slice(&crate::quantize::quantize(
                        &host.flatten_all()?.to_vec1::<f32>()?,
                        width,
                        format,
                        None,
                    )?);
                }
            }
        }
        let in_features = in_features.ok_or_else(|| invalid("empty stacked projection"))?;
        let Some(format) = format else {
            return Self::dense(
                &Tensor::from_vec(dense, (out_features, in_features), device)?,
                device,
            );
        };
        Self::from_blocks(&packed, format, (out_features, in_features), device)
    }

    /// A copy of the first `rows` output rows as their own projection, in the same format.
    /// GGML blocks never span rows, so the packed bytes are a prefix of this projection's.
    pub fn leading_rows(&self, rows: usize) -> Result<Self> {
        if rows == 0 || rows > self.out_features {
            return Err(invalid("leading projection rows outside weights"));
        }
        match &self.weights {
            Weights::Dense(weight) => {
                let device = weight.device().clone();
                Self::dense(&weight.narrow(0, 0, rows)?.copy()?, &device)
            }
            Weights::Packed {
                matmul: QMatMul::QTensor(packed),
                format,
                ..
            } => {
                let bytes = crate::quantize::packed_bytes(*format, rows, self.in_features)?;
                let data = packed.data()?;
                let prefix = data
                    .get(..bytes)
                    .ok_or_else(|| invalid("packed rows outside weights"))?;
                Self::from_blocks(prefix, *format, (rows, self.in_features), &packed.device())
            }
            _ => Err(invalid("weights have no host blocks")),
        }
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Device bytes held by the weights.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Packed GGML format, or `None` for dense weights.
    pub fn format(&self) -> Option<GgmlDType> {
        match &self.weights {
            Weights::Packed { format, .. } => Some(*format),
            #[cfg(feature = "metal")]
            Weights::Arena { format, .. } => Some(*format),
            _ => None,
        }
    }

    /// Candle execution: dense matmul, GGML GEMV for one row, tiled GGML GEMM otherwise.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match &self.weights {
            Weights::Dense(weight) => x.matmul(&weight.t()?)?,
            Weights::Packed { matmul, .. } => matmul.forward(x)?,
            #[cfg(feature = "metal")]
            Weights::Mxfp4 { .. } | Weights::Arena { .. } => return Err(host_read()),
        })
    }

    /// Reference projection through a contiguous output-row range. Dense weights and
    /// host-packed weights multiply only those rows; Metal-packed weights (a test oracle,
    /// whose bytes stay on the device) multiply every row and select the outputs.
    /// Production routed execution uses the grouped device kernel instead.
    pub fn forward_rows(&self, x: &Tensor, start: usize, rows: usize) -> Result<Tensor> {
        if rows == 0
            || start
                .checked_add(rows)
                .is_none_or(|end| end > self.out_features)
        {
            return Err(invalid("projection row range outside weights"));
        }
        Ok(match &self.weights {
            Weights::Dense(weight) => x.matmul(&weight.narrow(0, start, rows)?.t()?)?,
            Weights::Packed {
                matmul: QMatMul::QTensor(packed),
                format,
                ..
            } if packed.device().is_cpu() => {
                let row = self.in_features / format.block_size() * format.type_size();
                let data = packed.data()?;
                let bytes = data
                    .get(start * row..(start + rows) * row)
                    .ok_or_else(|| invalid("packed rows outside weights"))?;
                let (slice, _) =
                    upload_bytes(bytes, *format, (rows, self.in_features), &Device::Cpu)?;
                QMatMul::from_qtensor(slice)?.forward(x)?
            }
            Weights::Packed { matmul, .. } => matmul.forward(x)?.narrow(1, start, rows)?,
            #[cfg(feature = "metal")]
            Weights::Mxfp4 { .. } | Weights::Arena { .. } => return Err(host_read()),
        })
    }

    /// F32 rows `ids` (`[n]` U32) as `[n, in]`: an embedding lookup over the weights, so a
    /// tied head serves as the input embedding without a second table. Packed rows are
    /// dequantized by GGML's own row kernels and equal the matmul's operand values.
    pub fn rows(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(match &self.weights {
            Weights::Dense(weight) => weight.index_select(ids, 0)?,
            Weights::Packed {
                matmul: QMatMul::QTensor(packed),
                ..
            } => packed.embedding(ids)?,
            _ => return Err(host_read()),
        })
    }

    /// The whole matrix in F32 on its device, for calibration and sensitivity analysis.
    pub fn dequantize(&self) -> Result<Tensor> {
        Ok(match &self.weights {
            Weights::Dense(weight) => weight.clone(),
            Weights::Packed {
                matmul: QMatMul::QTensor(packed),
                ..
            } => packed.dequantize(&packed.device())?,
            _ => return Err(host_read()),
        })
    }

    /// Host copy of the packed GGML blocks, or `None` for dense weights.
    pub fn blocks(&self) -> Result<Option<Vec<u8>>> {
        Ok(match &self.weights {
            Weights::Dense(_) => None,
            Weights::Packed {
                matmul: QMatMul::QTensor(packed),
                ..
            } => Some(packed.data()?.into_owned()),
            _ => return Err(host_read()),
        })
    }

    #[cfg(feature = "metal")]
    pub(crate) fn dense_tensor(&self) -> Option<&Tensor> {
        match &self.weights {
            Weights::Dense(weight) => Some(weight),
            _ => None,
        }
    }

    #[cfg(feature = "metal")]
    pub(crate) fn packed_buffer(
        &self,
    ) -> Option<(&candle_metal_kernels::metal::Buffer, GgmlDType)> {
        match &self.weights {
            Weights::Packed {
                buffer: Some(buffer),
                format,
                ..
            } => Some((buffer, *format)),
            Weights::Arena { buffer, format } => Some((buffer.as_ref(), *format)),
            _ => None,
        }
    }
}

#[cfg(feature = "metal")]
impl PackedLinear {
    /// MXFP4 buffer and the byte offset of its scales, when these weights are in MXFP4.
    pub(crate) fn mxfp4_buffer(&self) -> Option<(&candle_metal_kernels::metal::Buffer, usize)> {
        match &self.weights {
            Weights::Mxfp4 { buffer, scales } => Some((buffer.as_ref(), *scales)),
            _ => None,
        }
    }
}

fn host_read() -> tur_core::TurError {
    invalid("these weights have no host blocks; only the device kernels read them")
}

/// A host F32 copy that owns its storage from offset zero: GGML quantization and device
/// transfers read the storage buffer, not the layout, so a row-range view would quantize
/// the wrong rows or upload the whole buffer.
fn host_f32(weight: &Tensor) -> Result<Tensor> {
    let host = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let owned = {
        let (storage, layout) = host.storage_and_layout();
        layout.start_offset() == 0
            && layout.is_contiguous()
            && matches!(&*storage, candle_core::Storage::Cpu(candle_core::CpuStorage::F32(v))
                if v.len() == host.elem_count())
    };
    Ok(if owned {
        host
    } else {
        host.force_contiguous()?
    })
}

#[cfg(feature = "metal")]
type Handle = Option<candle_metal_kernels::metal::Buffer>;
#[cfg(not(feature = "metal"))]
type Handle = ();

/// Packed GGML blocks of `format` and `shape` as a tensor on `device`.
fn upload_bytes(
    bytes: &[u8],
    format: GgmlDType,
    shape: impl Into<candle_core::Shape>,
    device: &Device,
) -> Result<(QTensor, Handle)> {
    let storage = QStorage::from_data(std::borrow::Cow::Borrowed(bytes), device, format)?;
    match &storage {
        QStorage::Cpu(_) => Ok((QTensor::new(storage, shape)?, Handle::default())),
        #[cfg(feature = "metal")]
        QStorage::Metal(metal) => {
            let buffer = metal.buffer().clone();
            Ok((QTensor::new(storage, shape)?, Some(buffer)))
        }
        _ => Err(invalid("packed weights support CPU and Metal devices")),
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn weights(rows: usize, cols: usize) -> Result<Tensor> {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
            .collect();
        Ok(Tensor::from_vec(values, (rows, cols), &Device::Cpu)?)
    }

    #[test]
    fn test_stacked_parts_and_row_ranges_match_the_whole_matrix() -> Result<()> {
        let (rows, cols) = (24, 256);
        let whole = weights(rows, cols)?;
        let x = weights(3, cols)?;
        for format in [None, Some(GgmlDType::Q8_0), Some(GgmlDType::Q4K)] {
            let single = match format {
                Some(format) => PackedLinear::quantized(&whole, format, &Device::Cpu)?,
                None => PackedLinear::dense(&whole, &Device::Cpu)?,
            };
            let stacked = PackedLinear::stacked(
                (0..3).map(|part| Ok(whole.narrow(0, part * 8, 8)?)),
                format,
                &Device::Cpu,
            )?;
            assert_eq!(stacked.resident_bytes(), single.resident_bytes());
            assert_eq!(
                stacked.forward(&x)?.to_vec2::<f32>()?,
                single.forward(&x)?.to_vec2::<f32>()?
            );
            // Only the requested rows are multiplied, with the same packed blocks.
            let full = single.forward(&x)?.narrow(1, 8, 8)?.to_vec2::<f32>()?;
            let rows = single.forward_rows(&x, 8, 8)?.to_vec2::<f32>()?;
            for (a, b) in full.iter().flatten().zip(rows.iter().flatten()) {
                assert!(
                    (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                    "{format:?}: {a} {b}"
                );
            }
            assert!(single.forward_rows(&x, 20, 8).is_err());
            let leading = single.leading_rows(10)?;
            assert_eq!(leading.out_features(), 10);
            assert_eq!(
                leading.forward(&x)?.to_vec2::<f32>()?,
                single.forward_rows(&x, 0, 10)?.to_vec2::<f32>()?
            );
            assert!(single.leading_rows(0).is_err() && single.leading_rows(25).is_err());
        }
        assert!(
            PackedLinear::stacked([weights(2, 64), weights(2, 32)], None, &Device::Cpu).is_err()
        );
        Ok(())
    }

    #[test]
    fn test_gathered_rows_equal_the_dequantized_matrix_and_blocks_round_trip() -> Result<()> {
        let (rows, cols) = (12, 256);
        let whole = weights(rows, cols)?;
        let ids = Tensor::new(&[3u32, 0, 11, 3], &Device::Cpu)?;
        for format in [
            None,
            Some(GgmlDType::Q8_0),
            Some(GgmlDType::Q4K),
            Some(GgmlDType::Q6K),
        ] {
            let layer = match format {
                Some(format) => PackedLinear::quantized(&whole, format, &Device::Cpu)?,
                None => PackedLinear::dense(&whole, &Device::Cpu)?,
            };
            assert_eq!(
                layer.rows(&ids)?.to_vec2::<f32>()?,
                layer
                    .dequantize()?
                    .index_select(&ids, 0)?
                    .to_vec2::<f32>()?
            );
            let Some(format) = format else {
                assert!(layer.blocks()?.is_none());
                continue;
            };
            let blocks = layer.blocks()?.unwrap_or_default();
            let copy = PackedLinear::from_blocks(&blocks, format, (rows, cols), &Device::Cpu)?;
            assert_eq!(copy.resident_bytes(), layer.resident_bytes());
            assert_eq!(copy.blocks()?, Some(blocks.clone()));
            assert!(
                PackedLinear::from_blocks(&blocks[1..], format, (rows, cols), &Device::Cpu)
                    .is_err()
            );
            assert!(
                PackedLinear::from_blocks(&blocks, format, (rows + 1, cols), &Device::Cpu).is_err()
            );
        }
        Ok(())
    }
}
