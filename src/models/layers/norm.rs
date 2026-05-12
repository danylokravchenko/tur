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
