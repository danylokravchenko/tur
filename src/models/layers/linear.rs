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
