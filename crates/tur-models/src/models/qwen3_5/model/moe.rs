//! Qwen3.5 `MoE` feed-forward block: normalized top-k and sigmoid-gated shared expert.

use super::{DenseMlp, Tensors, WeightFormat, f32_on};
use crate::models::qwen3_5::config::MoeConfig;
use candle_core::{Device, Tensor, quantized::GgmlDType};
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
        let packed = match format {
            WeightFormat::F32 => None,
            WeightFormat::Q8_0 => Some(GgmlDType::Q8_0),
            WeightFormat::Q4K => Some(GgmlDType::Q4K),
        };
        // Experts stream from the checkpoint and pack one at a time: the host never holds
        // a full-precision bank.
        let mut bank = |name: &str| {
            PackedLinear::stacked(
                tensors.split(&format!("{prefix}.experts.{name}"))?,
                packed,
                device,
            )
        };
        let (gate_up, down) = (bank("gate_up_proj")?, bank("down_proj")?);
        Ok(Self {
            router: f32_on(&tensors.take(&format!("{prefix}.gate.weight"))?, device)?,
            shared_gate: f32_on(
                &tensors.take(&format!("{prefix}.shared_expert_gate.weight"))?,
                device,
            )?,
            shared: DenseMlp::load(tensors, &format!("{prefix}.shared_expert"), format, device)?,
            gate_up,
            down,
            settings,
        })
    }

    pub(super) fn projection_bytes(&self) -> usize {
        self.shared.bytes() + self.gate_up.resident_bytes() + self.down.resident_bytes()
    }

    pub(super) fn router_bytes(&self) -> usize {
        (self.router.elem_count() + self.shared_gate.elem_count()) * 4
    }

    /// Numerical oracle; only routing decisions cross to the host. Each selected expert
    /// projects all of its tokens at once through only its own rows, in the declared
    /// storage format, and outputs accumulate in expert-ID order. Fused Metal dispatch
    /// never uses this path.
    pub(super) fn reference(&self, h: &Tensor) -> Result<Tensor> {
        let logits = h.matmul(&self.router.t()?)?.to_vec2::<f32>()?;
        let width = h.dim(1)?;
        let inner = self.settings.moe_intermediate_size;
        let mut groups = vec![(Vec::new(), Vec::new()); self.settings.num_experts];
        for (token, logits) in logits.iter().enumerate() {
            let (experts, weights) = route(logits, self.settings.num_experts_per_tok)?;
            let token = u32::try_from(token).map_err(|_| invalid("too many MoE tokens"))?;
            for (expert, weight) in experts.into_iter().zip(weights) {
                let (tokens, scales) = groups
                    .get_mut(expert)
                    .ok_or_else(|| invalid("MoE expert outside the bank"))?;
                tokens.push(token);
                scales.push(weight);
            }
        }
        let mut output = Tensor::zeros(h.dims(), h.dtype(), h.device())?;
        for (expert, (tokens, scales)) in groups.into_iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let count = tokens.len();
            let ids = Tensor::from_vec(tokens, count, h.device())?;
            let rows = h.index_select(&ids, 0)?;
            let gate_up = self
                .gate_up
                .forward_rows(&rows, expert * 2 * inner, 2 * inner)?;
            let act = (candle_nn::ops::silu(&gate_up.narrow(1, 0, inner)?)?
                * gate_up.narrow(1, inner, inner)?)?;
            let scales = Tensor::from_vec(scales, (count, 1), h.device())?;
            let routed = self
                .down
                .forward_rows(&act, expert * width, width)?
                .broadcast_mul(&scales)?;
            output = output.index_add(&ids, &routed, 0)?;
        }
        let shared = self.shared.forward(h)?;
        let gate = candle_nn::ops::sigmoid(&h.matmul(&self.shared_gate.t()?)?)?;
        Ok((output + shared.broadcast_mul(&gate)?)?)
    }

    /// Device routing; gate/up reads each token through the routing permutation, and the
    /// weighted scatter adds the gated shared expert in the same pass.
    #[cfg(feature = "metal")]
    pub(super) fn fused(&self, k: &tur_metal::kernels::Kernels, h: &Tensor) -> Result<Tensor> {
        let routes = k.route(
            &h.matmul(&self.router.t()?)?,
            self.settings.num_experts_per_tok,
        )?;
        let act = k.swiglu(&k.expert_linear_routed(&self.gate_up, h, &routes)?)?;
        let routed = k.expert_linear(&self.down, &act, &routes)?;
        let shared = k.linear(
            &self.shared.down,
            &k.swiglu(&k.linear(&self.shared.gate_up, h)?)?,
        )?;
        let gate = h.matmul(&self.shared_gate.t()?)?;
        k.expert_combine(
            &routed,
            &routes,
            tur_metal::kernels::SharedExpert {
                output: &shared,
                gate: &gate,
            },
        )
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
