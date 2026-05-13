use std::fmt::Debug;

use candle_core::{DType, Result, Tensor};
use candle_nn::{LayerNorm, Module, RmsNorm, var_builder::Shard};
use either::Either;

use crate::weights::VarBuilderX;

#[derive(Clone)]
pub struct NormX {
    norm: Either<RmsNorm, LayerNorm>,
    dtype: DType,
}

impl Debug for NormX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.norm {
            Either::Left(_) => write!(f, "RmsNorm"),
            Either::Right(_) => write!(f, "LayerNorm"),
        }
    }
}

impl NormX {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        if xs.dtype() != self.dtype {
            let converted = xs.to_dtype(self.dtype)?;
            let out = match &self.norm {
                Either::Left(norm) => norm.forward(&converted)?,
                Either::Right(norm) => norm.forward(&converted)?,
            };
            out.to_dtype(in_dtype)
        } else {
            let out = match &self.norm {
                Either::Left(norm) => norm.forward(xs)?,
                Either::Right(norm) => norm.forward(xs)?,
            };
            Ok(out)
        }
    }
}

pub fn rms_norm(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    is_gemma: bool,
) -> Result<NormX> {
    rms_norm_sharded(size, eps, vb, dtype, is_gemma, Shard::default())
}

pub fn rms_norm_sharded(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    is_gemma: bool,
    shard: Shard,
) -> Result<NormX> {
    let (weight, norm_dtype) = match &vb.0 {
        Either::Left(inner_vb) => {
            let ws = inner_vb.get_with_hints(size, "weight", shard)?;
            if ws.dtype() != dtype {
                (ws.to_dtype(dtype)?, dtype)
            } else {
                (ws, dtype)
            }
        }
        Either::Right(inner_vb) => {
            // // Dequantize and convert to target dtype (BF16 on GPU/Metal, F32 on CPU)
            // let w = inner_vb
            //     .get(size, "weight")?
            //     .dequantize(inner_vb.device())?;
            // let w = if w.dtype() != dtype {
            //     w.to_dtype(dtype)?
            // } else {
            //     w
            // };
            // (w, dtype)
            // GGUF: Dequantize to F32 (QMatMul also outputs F32)
            let w = inner_vb
                .get(size, "weight")?
                .dequantize(inner_vb.device())?;
            (w, DType::F32)
        }
    };

    let weight = if is_gemma { (weight + 1.0)? } else { weight };
    Ok(NormX {
        norm: Either::Left(RmsNorm::new(weight, eps)),
        dtype: norm_dtype,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{LayerNorm, RmsNorm};
    use either::Either;

    #[test]
    fn test_normx_forward_same_dtype() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a simple RmsNorm
        let weight = Tensor::ones((4,), dtype, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);
        let normx = NormX {
            norm: Either::Left(norm),
            dtype,
        };

        // Create input tensor with same dtype
        let input = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();

        // Forward pass should succeed
        let result = normx.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), &[2, 4]);
    }

    #[test]
    fn test_normx_forward_different_dtype() {
        let device = Device::Cpu;
        let norm_dtype = DType::F32;
        let input_dtype = DType::F16;

        // Create RmsNorm with F32
        let weight = Tensor::ones((4,), norm_dtype, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);
        let normx = NormX {
            norm: Either::Left(norm),
            dtype: norm_dtype,
        };

        // Create input tensor with F16
        let input = Tensor::randn(0f32, 1f32, (2, 4), &device)
            .unwrap()
            .to_dtype(input_dtype)
            .unwrap();

        // Forward pass should convert to F32, apply norm, then convert back to F16
        let result = normx.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dtype(), input_dtype); // Should match input dtype
        assert_eq!(output.dims(), &[2, 4]);
    }

    #[test]
    fn test_normx_forward_layer_norm() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create LayerNorm instead of RmsNorm
        let weight = Tensor::ones((4,), dtype, &device).unwrap();
        let bias = Tensor::zeros((4,), dtype, &device).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5);
        let normx = NormX {
            norm: Either::Right(norm),
            dtype,
        };

        let input = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();

        let result = normx.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), &[2, 4]);
    }

    // Property-based tests
    #[test]
    fn test_normx_preserves_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let weight = Tensor::ones((8,), dtype, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);
        let normx = NormX {
            norm: Either::Left(norm),
            dtype,
        };

        // Test various input shapes
        let shapes = vec![vec![1, 8], vec![4, 8], vec![2, 3, 8], vec![1, 1, 1, 8]];

        for shape in shapes {
            let input = Tensor::randn(0f32, 1f32, shape.clone(), &device).unwrap();
            let output = normx.forward(&input).unwrap();
            assert_eq!(output.dims(), shape.as_slice());
        }
    }

    #[test]
    fn test_normx_numerical_stability() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let size = 4;
        let weight = Tensor::ones((size,), dtype, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);
        let normx = NormX {
            norm: Either::Left(norm),
            dtype,
        };

        // Test with extreme values
        let large_input = Tensor::ones((2, size), dtype, &device)
            .unwrap()
            .affine(1000.0, 0.0)
            .unwrap();

        let result = normx.forward(&large_input);
        assert!(result.is_ok());

        // Output should be normalized (not NaN or Inf)
        let output = result.unwrap();
        let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(output_vec.iter().all(|&x| x.is_finite()));
    }
}
