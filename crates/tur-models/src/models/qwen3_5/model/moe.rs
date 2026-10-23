//! Qwen3.5 `MoE` feed-forward block: normalized top-k and sigmoid-gated shared expert.

use super::{DenseMlp, Tensors, WeightFormat, f32_on, linear};
use crate::models::qwen3_5::config::MoeConfig;
use candle_core::{Device, Tensor};
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

pub(super) struct Moe {
    router: Tensor,
    shared_gate: Tensor,
    shared: DenseMlp,
    gate_up: PackedLinear,
    down: PackedLinear,
    settings: MoeConfig,
}

impl Moe {
    pub(super) fn load(
        tensors: &mut Tensors,
        prefix: &str,
        settings: MoeConfig,
        format: WeightFormat,
        device: &Device,
    ) -> Result<Self> {
        let gate_up = tensors
            .take(&format!("{prefix}.experts.gate_up_proj"))?
            .flatten_to(1)?;
        let down = tensors
            .take(&format!("{prefix}.experts.down_proj"))?
            .flatten_to(1)?;
        Ok(Self {
            router: f32_on(&tensors.take(&format!("{prefix}.gate.weight"))?, device)?,
            shared_gate: f32_on(
                &tensors.take(&format!("{prefix}.shared_expert_gate.weight"))?,
                device,
            )?,
            shared: DenseMlp::load(tensors, &format!("{prefix}.shared_expert"), format, device)?,
            gate_up: linear(&gate_up, format, device)?,
            down: linear(&down, format, device)?,
            settings,
        })
    }

    pub(super) fn projection_bytes(&self) -> usize {
        self.shared.bytes() + self.gate_up.resident_bytes() + self.down.resident_bytes()
    }

    pub(super) fn router_bytes(&self) -> usize {
        (self.router.elem_count() + self.shared_gate.elem_count()) * 4
    }

    /// Slow numerical oracle; only routing decisions cross to the host. Expert weights
    /// remain in their declared storage format. Fused Metal dispatch never uses this path.
    pub(super) fn reference(&self, h: &Tensor) -> Result<Tensor> {
        let logits = h.matmul(&self.router.t()?)?.to_vec2::<f32>()?;
        let width = h.dim(1)?;
        let inner = self.settings.moe_intermediate_size;
        let mut outputs = Vec::with_capacity(logits.len());
        for (token, logits) in logits.iter().enumerate() {
            let (experts, weights) = route(logits, self.settings.num_experts_per_tok)?;
            let row = h.narrow(0, token, 1)?;
            let mut output = Tensor::zeros((1, width), h.dtype(), h.device())?;
            // Accumulate in expert order, as the independent Transformers oracle does.
            let mut selected: Vec<_> = experts.into_iter().zip(weights).collect();
            selected.sort_unstable_by_key(|&(expert, _)| expert);
            for (expert, weight) in selected {
                let gate_up = self
                    .gate_up
                    .forward_rows(&row, expert * 2 * inner, 2 * inner)?;
                let act = (candle_nn::ops::silu(&gate_up.narrow(1, 0, inner)?)?
                    * gate_up.narrow(1, inner, inner)?)?;
                output = (output
                    + (self.down.forward_rows(&act, expert * width, width)? * f64::from(weight))?)?;
            }
            outputs.push(output);
        }
        let shared = self.shared.forward(h)?;
        let gate = candle_nn::ops::sigmoid(&h.matmul(&self.shared_gate.t()?)?)?;
        Ok((Tensor::cat(&outputs, 0)? + shared.broadcast_mul(&gate)?)?)
    }

    #[cfg(feature = "metal")]
    pub(super) fn fused(&self, k: &tur_metal::kernels::Kernels, h: &Tensor) -> Result<Tensor> {
        let routes = k.route(
            &h.matmul(&self.router.t()?)?,
            self.settings.num_experts_per_tok,
        )?;
        let gathered = k.expert_gather(h, &routes)?;
        let act = k.swiglu(&k.expert_linear(&self.gate_up, &gathered, &routes)?)?;
        let routed = k.expert_scatter(&k.expert_linear(&self.down, &act, &routes)?, &routes)?;
        let shared = k.linear(
            &self.shared.down,
            &k.swiglu(&k.linear(&self.shared.gate_up, h)?)?,
        )?;
        let gate = candle_nn::ops::sigmoid(&h.matmul(&self.shared_gate.t()?)?)?;
        Ok((routed + shared.broadcast_mul(&gate)?)?)
    }
}

/// F32 top-k probabilities, normalized over the selected experts. Softmax over all
/// experts cancels in the renormalization; subtract the selected maximum for stability.
#[allow(
    clippy::float_cmp,
    reason = "routing ties include both signs of zero, as on the device"
)]
fn route(logits: &[f32], top_k: usize) -> Result<(Vec<usize>, Vec<f32>)> {
    if top_k == 0 || top_k > logits.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid MoE router scores or top-k"));
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        if logits[a] == logits[b] {
            a.cmp(&b)
        } else {
            logits[b].total_cmp(&logits[a])
        }
    });
    indices.truncate(top_k);
    let max = logits[indices[0]];
    let mut weights: Vec<f32> = indices.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok((indices, weights))
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    #[test]
    fn test_top_k_preserves_precision_ties_and_normalized_probabilities() -> Result<()> {
        let (indices, weights) = route(&[1000.0, 999.0, -1000.0, 1000.0], 3)?;
        assert_eq!(indices, [0, 3, 1]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-7);
        assert!((weights[2] / weights[0] - (-1.0f32).exp()).abs() < 1e-7);
        assert!(route(&[f32::NAN], 1).is_err());
        assert!(route(&[1.0], 2).is_err());
        assert_eq!(route(&[-0.0, 0.0], 1)?.0, [0]);
        Ok(())
    }
}
