use candle_core::{DType, Result};
use candle_nn::Embedding;
use either::Either;

use crate::weights::VarBuilderX;

pub mod fused;
pub mod linear;
pub use linear::{LinearX, linear};
pub mod norm;

pub fn embedding(
    vocab_size: Option<usize>,
    hidden_size: usize,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<(Embedding, usize)> {
    let (embeddings, vocab_size) = match &vb.0 {
        Either::Left(inner_vb) => {
            assert!(
                vocab_size.is_some(),
                "vocab_size must be specified for safetensor models"
            );
            (
                inner_vb
                    .get((vocab_size.unwrap(), hidden_size), "weight")?
                    .to_dtype(dtype)?,
                vocab_size.unwrap(),
            )
        }
        Either::Right(inner_vb) => {
            // GGUF: Dequantize embeddings to F32 (QMatMul also outputs F32)
            let weight = if let Some(size) = vocab_size {
                inner_vb.get((size, hidden_size), "weight")?
            } else {
                inner_vb.get_no_shape("weight")?
            }
            .dequantize(inner_vb.device())?;

            // // Convert to target dtype (BF16 on GPU/Metal, F32 on CPU)
            // let weight = if weight.dtype() != dtype {
            //     weight.to_dtype(dtype)?
            // } else {
            //     weight
            // };

            let vocab_size = vocab_size.unwrap_or(weight.dim(0)?);
            (weight, vocab_size)
        }
    };
    Ok((Embedding::new(embeddings, hidden_size), vocab_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Module;
    use std::{collections::HashMap, path::PathBuf, time::SystemTime};

    fn write_tensor_to_file(name: &str, tensor: &Tensor) -> Result<PathBuf> {
        let mut path = std::env::temp_dir();
        path.push(name);
        let weight = tensor.to_device(&Device::Cpu)?.contiguous()?;
        let flat: Vec<f32> = weight.flatten_all()?.to_vec1::<f32>()?;
        let bytes: &[u8] = bytemuck::cast_slice(&flat);

        let shape = tensor.shape().dims().to_vec();

        let tensor_view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, bytes)?;

        let mut map = HashMap::new();
        map.insert("weight".to_string(), tensor_view);

        safetensors::serialize_to_file(&map, None, &path)?;

        Ok(path)
    }

    // Helper to create a mock VarBuilderX with SafeTensors backend
    fn create_safetensor_varbuilder(
        vocab_size: usize,
        hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilderX<'static>> {
        let weight = Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), device)?;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = write_tensor_to_file(&format!("weight.safetensors_{timestamp}"), &weight)?;

        // Load via SafeTensors backend
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                std::slice::from_ref(&path),
                dtype,
                device,
            )?
        };

        Ok(VarBuilderX(Either::Left(vb), String::new(), None))
    }

    #[test]
    fn test_embedding_with_safetensors() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 1000;
        let hidden_size = 128;

        let vb = create_safetensor_varbuilder(vocab_size, hidden_size, dtype, &device).unwrap();

        let result = embedding(Some(vocab_size), hidden_size, vb, dtype);
        assert!(result.is_ok());

        let (emb, returned_vocab_size) = result.unwrap();
        assert_eq!(returned_vocab_size, vocab_size);

        // Test forward pass with valid token IDs
        let token_ids = Tensor::new(&[0u32, 1u32, 2u32], &device).unwrap();
        let output = emb.forward(&token_ids);
        assert!(output.is_ok());

        let output_tensor = output.unwrap();
        assert_eq!(output_tensor.dims(), &[3, hidden_size]);
        assert_eq!(output_tensor.dtype(), dtype);
    }

    #[test]
    fn test_embedding_preserves_batch_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 100;
        let hidden_size = 64;

        let vb = create_safetensor_varbuilder(vocab_size, hidden_size, dtype, &device).unwrap();
        let (emb, _) = embedding(Some(vocab_size), hidden_size, vb, dtype).unwrap();

        // Test various input shapes
        let test_cases = vec![
            (vec![5], vec![5, hidden_size]),             // 1D: sequence
            (vec![2, 10], vec![2, 10, hidden_size]),     // 2D: batch x sequence
            (vec![2, 3, 5], vec![2, 3, 5, hidden_size]), // 3D: batch x beam x sequence
        ];

        for (input_shape, expected_output_shape) in test_cases {
            let token_ids = Tensor::zeros(input_shape.clone(), DType::U32, &device).unwrap();
            let output = emb.forward(&token_ids).unwrap();
            assert_eq!(
                output.dims(),
                expected_output_shape.as_slice(),
                "Failed for input shape {:?}",
                input_shape
            );
        }
    }

    #[test]
    fn test_embedding_different_dtypes() {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;

        // Test with F16
        let dtype = DType::F16;
        let vb =
            create_safetensor_varbuilder(vocab_size, hidden_size, DType::F32, &device).unwrap();
        let (emb, _) = embedding(Some(vocab_size), hidden_size, vb, dtype).unwrap();

        let token_ids = Tensor::new(&[0u32, 1u32], &device).unwrap();
        let output = emb.forward(&token_ids).unwrap();

        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), &[2, hidden_size]);
    }

    #[test]
    fn test_embedding_numerical_correctness() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 3;
        let hidden_size = 2;

        // Create embedding with known values
        let weight = Tensor::new(
            &[[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]],
            &device,
        )
        .unwrap();

        let path = write_tensor_to_file("test_embedding_numerical_correctness", &weight).unwrap();

        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                std::slice::from_ref(&path),
                dtype,
                &device,
            )
            .unwrap()
        };
        let vb = VarBuilderX(Either::Left(vb), String::new(), None);

        let (emb, _) = embedding(Some(vocab_size), hidden_size, vb, dtype).unwrap();

        // Look up token 0 and token 2
        let token_ids = Tensor::new(&[0u32, 2u32], &device).unwrap();
        let output = emb.forward(&token_ids).unwrap();

        let output_vec = output.to_vec2::<f32>().unwrap();
        assert_eq!(output_vec[0], vec![1.0, 2.0]); // Token 0
        assert_eq!(output_vec[1], vec![5.0, 6.0]); // Token 2
    }

    #[test]
    fn test_embedding_large_vocab() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 50000; // Large vocabulary
        let hidden_size = 256;

        let vb = create_safetensor_varbuilder(vocab_size, hidden_size, dtype, &device).unwrap();
        let result = embedding(Some(vocab_size), hidden_size, vb, dtype);
        assert!(result.is_ok());

        let (emb, returned_vocab_size) = result.unwrap();
        assert_eq!(returned_vocab_size, vocab_size);

        // Test with valid token IDs
        let token_ids = Tensor::new(&[0u32, 100u32, 49999u32], &device).unwrap();
        let output = emb.forward(&token_ids).unwrap();
        assert_eq!(output.dims(), &[3, hidden_size]);
    }

    #[test]
    fn test_embedding_returns_correct_vocab_size() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 500;
        let hidden_size = 128;

        let vb = create_safetensor_varbuilder(vocab_size, hidden_size, dtype, &device).unwrap();
        let (_, returned_vocab_size) = embedding(Some(vocab_size), hidden_size, vb, dtype).unwrap();

        assert_eq!(returned_vocab_size, vocab_size);
    }

    #[test]
    fn test_embedding_zero_token_id() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 100;
        let hidden_size = 64;

        let vb = create_safetensor_varbuilder(vocab_size, hidden_size, dtype, &device).unwrap();
        let (emb, _) = embedding(Some(vocab_size), hidden_size, vb, dtype).unwrap();

        // Token ID 0 should be valid
        let token_ids = Tensor::new(&[0u32], &device).unwrap();
        let output = emb.forward(&token_ids);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().dims(), &[1, hidden_size]);
    }

    #[test]
    fn test_embedding_sequential_tokens() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 100;
        let hidden_size = 64;

        let vb = create_safetensor_varbuilder(vocab_size, hidden_size, dtype, &device).unwrap();
        let (emb, _) = embedding(Some(vocab_size), hidden_size, vb, dtype).unwrap();

        // Test with sequential token IDs
        let token_ids = Tensor::new(&[0u32, 1u32, 2u32, 3u32, 4u32], &device).unwrap();
        let output = emb.forward(&token_ids).unwrap();

        assert_eq!(output.dims(), &[5, hidden_size]);

        // Verify output is finite (no NaN or Inf)
        let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(output_vec.iter().all(|&x| x.is_finite()));
    }

    #[test]
    #[should_panic(expected = "vocab_size must be specified for safetensor models")]
    fn test_embedding_panics_without_vocab_size_for_safetensors() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vocab_size = 100;
        let hidden_size = 64;

        let vb = create_safetensor_varbuilder(vocab_size, hidden_size, dtype, &device).unwrap();

        // Should panic because vocab_size is None for SafeTensors
        let _ = embedding(None, hidden_size, vb, dtype);
    }
}
