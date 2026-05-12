use candle_core::{DType, Result};
use candle_nn::{Embedding, Linear};
use either::Either;

use crate::weights::VarBuilderX;

// pub mod distributed;
// pub mod linear;
pub mod mamba;
pub mod norm;

/// Create a linear layer from VarBuilderX
pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilderX) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias").ok();
    Ok(Linear::new(weight, bias))
}

pub fn embedding(
    vocab_size: Option<usize>,
    hidden_size: usize,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<(Embedding, usize)> {
    let (embeddings, vocab_size) = match &vb.0 {
        Either::Left(vb) => {
            assert!(
                vocab_size.is_some(),
                "vocab_size must be specified for safetensor models"
            );
            (
                vb.get((vocab_size.unwrap(), hidden_size), "weight")?
                    .to_dtype(dtype)?,
                vocab_size.unwrap(),
            )
        }
        Either::Right(vb) => {
            let weight = if vocab_size.is_some() {
                vb.get((vocab_size.unwrap(), hidden_size), "weight")?
            } else {
                vb.get_no_shape("weight")?
            }
            .dequantize(vb.device())?;
            let vocab_size = vocab_size.unwrap_or(weight.dim(0)?);
            (weight, vocab_size)
        }
    };
    Ok((Embedding::new(embeddings, hidden_size), vocab_size))
}
