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
        /// Handle to the same MTLBuffer the `QMatMul` owns, for the packed GEMV kernels.
        /// `matmul` keeps Candle's pooled `Arc` alive, so the pool cannot recycle it while
        /// this handle exists.
        #[cfg(feature = "metal")]
        buffer: Option<candle_metal_kernels::metal::Buffer>,
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
    /// `device` is shared, not copied.
    pub fn dense(weight: &Tensor, device: &Device) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        let weight = weight.to_dtype(DType::F32)?.to_device(device)?;
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
        if !in_features.is_multiple_of(format.block_size()) {
            return Err(invalid(format!(
                "input width {in_features} is not a multiple of the {format:?} block size"
            )));
        }
        let host = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let packed = QTensor::quantize(&host, format)?;
        let resident_bytes = packed.storage_size_in_bytes();
        let (qtensor, buffer) = upload(packed, device)?;
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
            resident_bytes,
        })
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
            Weights::Dense(_) => None,
            Weights::Packed { format, .. } => Some(*format),
        }
    }

    /// Candle execution: dense matmul, GGML GEMV for one row, tiled GGML GEMM otherwise.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match &self.weights {
            Weights::Dense(weight) => x.matmul(&weight.t()?)?,
            Weights::Packed { matmul, .. } => matmul.forward(x)?,
        })
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
            _ => None,
        }
    }
}

#[cfg(feature = "metal")]
type Handle = Option<candle_metal_kernels::metal::Buffer>;
#[cfg(not(feature = "metal"))]
type Handle = ();

/// Move the exact host-packed blocks to `device`, returning a buffer handle on Metal.
fn upload(packed: QTensor, device: &Device) -> Result<(QTensor, Handle)> {
    match device {
        Device::Cpu => Ok((packed, Handle::default())),
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            let storage = QStorage::from_data(packed.data()?, device, packed.dtype())?;
            let QStorage::Metal(metal) = &storage else {
                return Err(invalid("packed upload did not produce Metal storage"));
            };
            let buffer = metal.buffer().clone();
            Ok((QTensor::new(storage, packed.shape().clone())?, Some(buffer)))
        }
        _ => Err(invalid("packed weights support CPU and Metal devices")),
    }
}
