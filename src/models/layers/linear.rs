use candle_core::{Result, Tensor, quantized::QMatMul};
use candle_nn::{Linear, Module};
use either::Either;

use crate::VarBuilderX;

/// A linear layer that supports both quantized and non-quantized weights
#[derive(Clone)]
pub enum LinearX {
    /// Standard linear layer with dequantized weights
    Standard(Linear),
    /// Quantized linear layer using QMatMul
    Quantized {
        qmatmul: QMatMul,
        bias: Option<Tensor>,
    },
}

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(linear) => linear.forward(x),
            Self::Quantized { qmatmul, bias } => {
                let out = qmatmul.forward(x)?;
                match bias {
                    Some(b) => out.broadcast_add(b),
                    None => Ok(out),
                }
            }
        }
    }
}

impl LinearX {
    /// Returns the weight tensor for standard (non-quantized) layers.
    /// Returns `None` for quantized layers — the fused kernel cannot be used
    /// with quantized weights.
    pub fn weight(&self) -> Option<&Tensor> {
        match self {
            Self::Standard(linear) => Some(linear.weight()),
            Self::Quantized { .. } => None,
        }
    }
}

impl std::fmt::Debug for LinearX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard(_) => write!(f, "LinearX::Standard"),
            Self::Quantized { .. } => write!(f, "LinearX::Quantized"),
        }
    }
}

/// Create a linear layer from VarBuilderX
///
/// Strategy:
/// - SafeTensors: Always use full precision
/// - GGUF on CPU: Keep weights quantized (QMatMul) for memory efficiency
/// - GGUF on GPU/Metal: Dequantize to BF16/F32 for better performance
///   (GPU quantized ops are not well optimized yet)
pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilderX) -> Result<LinearX> {
    //let target_dtype = vb.dtype();
    match &vb.0 {
        Either::Left(inner_vb) => {
            // SafeTensors: use standard linear with full precision
            let weight = inner_vb.get((out_dim, in_dim), "weight")?;
            let bias = inner_vb.get(out_dim, "bias").ok();
            Ok(LinearX::Standard(Linear::new(weight, bias)))
        }
        Either::Right(inner_vb) => {
            // GGUF: keep weights quantized, use QMatMul
            // QMatMul dequantizes on-the-fly during forward pass
            let qweight = inner_vb.get((out_dim, in_dim), "weight")?;
            let bias = inner_vb
                .get(out_dim, "bias")
                .ok()
                .map(|qtensor| qtensor.dequantize(inner_vb.device()))
                .transpose()?;

            let qmatmul = QMatMul::from_arc(qweight)?;
            Ok(LinearX::Quantized { qmatmul, bias })
            // let device = inner_vb.device();

            // // Only use quantized matmul on CPU where it's beneficial
            // // On GPU/Metal, dequantize for better performance
            // if device.is_cpu() {
            //     // CPU: keep weights quantized for memory efficiency
            //     let qweight = inner_vb.get((out_dim, in_dim), "weight")?;
            //     let bias = inner_vb
            //         .get(out_dim, "bias")
            //         .ok()
            //         .map(|qtensor| qtensor.dequantize(device))
            //         .transpose()?;

            //     let qmatmul = QMatMul::from_arc(qweight)?;
            //     Ok(LinearX::Quantized { qmatmul, bias })
            // } else {
            //     // GPU/Metal: dequantize for better performance
            //     let qweight = inner_vb.get((out_dim, in_dim), "weight")?;
            //     let weight = qweight.dequantize(device)?;

            //     // Convert to appropriate dtype (BF16 on GPU/Metal, F32 on CPU)
            //     let weight = if weight.dtype() != target_dtype {
            //         weight.to_dtype(target_dtype)?
            //     } else {
            //         weight
            //     };

            //     let bias = inner_vb
            //         .get(out_dim, "bias")
            //         .ok()
            //         .map(|qtensor| {
            //             let b = qtensor.dequantize(device)?;
            //             if b.dtype() != target_dtype {
            //                 b.to_dtype(target_dtype)
            //             } else {
            //                 Ok(b)
            //             }
            //         })
            //         .transpose()?;

            //     Ok(LinearX::Standard(Linear::new(weight, bias)))
            // }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;

    #[test]
    fn test_linearx_standard_forward() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a standard linear layer: y = Wx + b
        // Input: (batch=2, in_dim=3), Weight: (out_dim=4, in_dim=3), Bias: (out_dim=4)
        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device).unwrap();
        let bias = Tensor::randn(0f32, 0.1f32, (4,), &device).unwrap();
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        // Create input tensor
        let input = Tensor::randn(0f32, 1f32, (2, 3), &device).unwrap();

        // Forward pass
        let result = linearx.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dims(), &[2, 4]); // (batch, out_dim)
        assert_eq!(output.dtype(), dtype);
    }

    #[test]
    fn test_linearx_standard_forward_no_bias() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create linear layer without bias
        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device).unwrap();
        let linear = Linear::new(weight, None);
        let linearx = LinearX::Standard(linear);

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device).unwrap();

        let result = linearx.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dims(), &[2, 4]);
        assert_eq!(output.dtype(), dtype);
    }

    #[test]
    fn test_linearx_standard_preserves_batch_shape() {
        let device = Device::Cpu;

        let weight = Tensor::randn(0f32, 1f32, (8, 4), &device).unwrap();
        let bias = Tensor::randn(0f32, 0.1f32, (8,), &device).unwrap();
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        // Test various batch shapes
        let test_cases = vec![
            (vec![1, 4], vec![1, 8]),             // Single sample
            (vec![5, 4], vec![5, 8]),             // Batch
            (vec![2, 3, 4], vec![2, 3, 8]),       // 3D input
            (vec![1, 2, 3, 4], vec![1, 2, 3, 8]), // 4D input
        ];

        for (input_shape, expected_output_shape) in test_cases {
            let input = Tensor::randn(0f32, 1f32, input_shape.clone(), &device).unwrap();
            let output = linearx.forward(&input).unwrap();
            assert_eq!(
                output.dims(),
                expected_output_shape.as_slice(),
                "Failed for input shape {:?}",
                input_shape
            );
        }
    }

    #[test]
    fn test_linearx_standard_numerical_correctness() {
        let device = Device::Cpu;

        // Simple test case: identity-like transformation
        // Weight = [[1, 0], [0, 1]], Bias = [0, 0]
        let weight = Tensor::new(&[[1f32, 0f32], [0f32, 1f32]], &device).unwrap();
        let bias = Tensor::new(&[0f32, 0f32], &device).unwrap();
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        let input = Tensor::new(&[[2f32, 3f32]], &device).unwrap();
        let output = linearx.forward(&input).unwrap();

        let output_vec = output.to_vec2::<f32>().unwrap();
        assert_eq!(output_vec, vec![vec![2f32, 3f32]]);
    }

    #[test]
    fn test_linearx_clone() {
        let device = Device::Cpu;

        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device).unwrap();
        let bias = Tensor::randn(0f32, 0.1f32, (4,), &device).unwrap();
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        // Clone the layer
        let cloned = linearx.clone();

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device).unwrap();

        // Both should work independently
        assert!(linearx.forward(&input).is_ok());
        assert!(cloned.forward(&input).is_ok());
    }

    #[test]
    fn test_linearx_different_dtypes() {
        let device = Device::Cpu;

        // Test with F16
        let weight_f32 = Tensor::randn(0f32, 1f32, (4, 3), &device).unwrap();
        let weight_f16 = weight_f32.to_dtype(DType::F16).unwrap();
        let linear = Linear::new(weight_f16, None);
        let linearx = LinearX::Standard(linear);

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let result = linearx.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(output.dims(), &[2, 4]);
    }

    #[test]
    fn test_linearx_zero_input() {
        let device = Device::Cpu;

        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device).unwrap();
        let bias = Tensor::ones((4,), DType::F32, &device).unwrap();
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        // Zero input should give us just the bias
        let input = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
        let output = linearx.forward(&input).unwrap();

        // Output should be close to bias (broadcasted)
        assert_eq!(output.dims(), &[2, 4]);
        let output_vec = output.to_vec2::<f32>().unwrap();
        for row in output_vec {
            for &val in &row {
                assert!((val - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", val);
            }
        }
    }

    #[test]
    fn test_linearx_large_dimensions() {
        let device = Device::Cpu;

        // Test with larger dimensions to ensure no overflow/underflow
        let weight = Tensor::randn(0f32, 0.01f32, (128, 64), &device).unwrap();
        let bias = Tensor::zeros((128,), DType::F32, &device).unwrap();
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        let input = Tensor::randn(0f32, 1f32, (4, 64), &device).unwrap();
        let output = linearx.forward(&input).unwrap();

        assert_eq!(output.dims(), &[4, 128]);

        // Check for numerical stability (no NaN or Inf)
        let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(output_vec.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_module_trait_implementation() {
        // Verify that LinearX properly implements the Module trait
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device).unwrap();
        let linear = Linear::new(weight, None);
        let linearx = LinearX::Standard(linear);

        // Should be able to use as Module
        fn use_as_module(module: &impl Module, input: &Tensor) -> Result<Tensor> {
            module.forward(input)
        }

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device).unwrap();
        let result = use_as_module(&linearx, &input);
        assert!(result.is_ok());
    }
}
