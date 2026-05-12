use candle_core::{DType, Result};
use candle_nn::Embedding;
use either::Either;

use crate::weights::VarBuilderX;

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
            let weight = if vocab_size.is_some() {
                inner_vb.get((vocab_size.unwrap(), hidden_size), "weight")?
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
