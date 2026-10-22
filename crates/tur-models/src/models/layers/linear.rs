use candle_core::{Result, Tensor, quantized::QMatMul};
use candle_nn::{Linear, Module};

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

impl std::fmt::Debug for LinearX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard(_) => write!(f, "LinearX::Standard"),
            Self::Quantized { .. } => write!(f, "LinearX::Quantized"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;

    #[test]
    fn test_linearx_standard_forward() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a standard linear layer: y = Wx + b
        // Input: (batch=2, in_dim=3), Weight: (out_dim=4, in_dim=3), Bias: (out_dim=4)
        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device)?;
        let bias = Tensor::randn(0f32, 0.1f32, (4,), &device)?;
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        // Create input tensor
        let input = Tensor::randn(0f32, 1f32, (2, 3), &device)?;

        // Forward pass
        let result = linearx.forward(&input);
        assert!(result.is_ok());

        let output = result?;
        assert_eq!(output.dims(), &[2, 4]); // (batch, out_dim)
        assert_eq!(output.dtype(), dtype);
        Ok(())
    }

    #[test]
    fn test_linearx_standard_forward_no_bias() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create linear layer without bias
        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device)?;
        let linear = Linear::new(weight, None);
        let linearx = LinearX::Standard(linear);

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device)?;

        let result = linearx.forward(&input);
        assert!(result.is_ok());

        let output = result?;
        assert_eq!(output.dims(), &[2, 4]);
        assert_eq!(output.dtype(), dtype);
        Ok(())
    }

    #[test]
    fn test_linearx_standard_preserves_batch_shape() -> Result<()> {
        let device = Device::Cpu;

        let weight = Tensor::randn(0f32, 1f32, (8, 4), &device)?;
        let bias = Tensor::randn(0f32, 0.1f32, (8,), &device)?;
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
            let input = Tensor::randn(0f32, 1f32, input_shape.clone(), &device)?;
            let output = linearx.forward(&input)?;
            assert_eq!(
                output.dims(),
                expected_output_shape.as_slice(),
                "Failed for input shape {:?}",
                input_shape
            );
        }
        Ok(())
    }

    #[test]
    fn test_linearx_standard_numerical_correctness() -> Result<()> {
        let device = Device::Cpu;

        // Simple test case: identity-like transformation
        // Weight = [[1, 0], [0, 1]], Bias = [0, 0]
        let weight = Tensor::new(&[[1f32, 0f32], [0f32, 1f32]], &device)?;
        let bias = Tensor::new(&[0f32, 0f32], &device)?;
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        let input = Tensor::new(&[[2f32, 3f32]], &device)?;
        let output = linearx.forward(&input)?;

        let output_vec = output.to_vec2::<f32>()?;
        assert_eq!(output_vec, vec![vec![2f32, 3f32]]);
        Ok(())
    }

    #[test]
    fn test_linearx_clone() -> Result<()> {
        let device = Device::Cpu;

        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device)?;
        let bias = Tensor::randn(0f32, 0.1f32, (4,), &device)?;
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        // Clone the layer
        let cloned = linearx.clone();

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device)?;

        // Both should work independently
        assert!(linearx.forward(&input).is_ok());
        assert!(cloned.forward(&input).is_ok());
        Ok(())
    }

    #[test]
    fn test_linearx_different_dtypes() -> Result<()> {
        let device = Device::Cpu;

        // Test with F16
        let weight_f32 = Tensor::randn(0f32, 1f32, (4, 3), &device)?;
        let weight_f16 = weight_f32.to_dtype(DType::F16)?;
        let linear = Linear::new(weight_f16, None);
        let linearx = LinearX::Standard(linear);

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device)?.to_dtype(DType::F16)?;

        let result = linearx.forward(&input);
        assert!(result.is_ok());

        let output = result?;
        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(output.dims(), &[2, 4]);
        Ok(())
    }

    #[test]
    fn test_linearx_zero_input() -> Result<()> {
        let device = Device::Cpu;

        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device)?;
        let bias = Tensor::ones((4,), DType::F32, &device)?;
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        // Zero input should give us just the bias
        let input = Tensor::zeros((2, 3), DType::F32, &device)?;
        let output = linearx.forward(&input)?;

        // Output should be close to bias (broadcasted)
        assert_eq!(output.dims(), &[2, 4]);
        let output_vec = output.to_vec2::<f32>()?;
        for row in output_vec {
            for &val in &row {
                assert!((val - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", val);
            }
        }
        Ok(())
    }

    #[test]
    fn test_linearx_large_dimensions() -> Result<()> {
        let device = Device::Cpu;

        // Test with larger dimensions to ensure no overflow/underflow
        let weight = Tensor::randn(0f32, 0.01f32, (128, 64), &device)?;
        let bias = Tensor::zeros((128,), DType::F32, &device)?;
        let linear = Linear::new(weight, Some(bias));
        let linearx = LinearX::Standard(linear);

        let input = Tensor::randn(0f32, 1f32, (4, 64), &device)?;
        let output = linearx.forward(&input)?;

        assert_eq!(output.dims(), &[4, 128]);

        // Check for numerical stability (no NaN or Inf)
        let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
        assert!(output_vec.iter().all(|&x| x.is_finite()));
        Ok(())
    }

    #[test]
    fn test_module_trait_implementation() -> Result<()> {
        // Verify that LinearX properly implements the Module trait
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (4, 3), &device)?;
        let linear = Linear::new(weight, None);
        let linearx = LinearX::Standard(linear);

        // Should be able to use as Module
        fn use_as_module(module: &impl Module, input: &Tensor) -> Result<Tensor> {
            module.forward(input)
        }

        let input = Tensor::randn(0f32, 1f32, (2, 3), &device)?;
        let result = use_as_module(&linearx, &input);
        assert!(result.is_ok());
        Ok(())
    }
}
