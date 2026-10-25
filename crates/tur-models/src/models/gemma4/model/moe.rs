//! Gemma's independently normalized dense and routed feed-forward branches.
use super::{Gemma4, Tensors};
use crate::models::math;
use candle_core::{DType, Tensor};
use tur_core::{Result, execution::invalid};
use tur_metal::packed::PackedLinear;

pub(super) struct Moe {
    router: Tensor,
    scale: Tensor,
    expert_scale: Tensor,
    pre: Tensor,
    post_dense: Tensor,
    post_routed: Tensor,
    gate_up: PackedLinear,
    down: PackedLinear,
    experts: usize,
    top_k: usize,
}
impl Moe {
    pub(super) fn load(t: &Tensors, p: &str, experts: usize, top_k: usize) -> Result<Self> {
        let tensor = |name: &str| -> Result<Tensor> {
            Ok(t.take(&format!("{p}.{name}"))?
                .to_dtype(DType::F32)?
                .to_device(&t.device)?)
        };
        Ok(Self {
            router: tensor("router.proj.weight")?,
            scale: tensor("router.scale")?,
            expert_scale: tensor("router.per_expert_scale")?,
            pre: tensor("pre_feedforward_layernorm_2.weight")?,
            post_dense: tensor("post_feedforward_layernorm_1.weight")?,
            post_routed: tensor("post_feedforward_layernorm_2.weight")?,
            gate_up: t.projection(&format!("{p}.experts.gate_up_proj"))?,
            down: t.projection(&format!("{p}.experts.down_proj"))?,
            experts,
            top_k,
        })
    }
    pub(super) fn bytes(&self) -> usize {
        [
            &self.router,
            &self.scale,
            &self.expert_scale,
            &self.pre,
            &self.post_dense,
            &self.post_routed,
        ]
        .iter()
        .map(|t| t.elem_count() * 4)
        .sum::<usize>()
            + self.gate_up.resident_bytes()
            + self.down.resident_bytes()
    }
    pub(super) fn forward(
        &self,
        model: &Gemma4,
        residual: &Tensor,
        dense: &Tensor,
    ) -> Result<Tensor> {
        let width = residual.dim(1)?;
        let ones = Tensor::ones(width, DType::F32, residual.device())?;
        let norm = model.rms(residual, &ones)?.broadcast_mul(&self.scale)?;
        let logits = (norm / (width as f64).sqrt())?.matmul(&self.router.t()?)?;
        let input = model.rms(residual, &self.pre)?;
        #[cfg(feature = "metal")]
        if let Some(k) = &model.kernels {
            let routes = k.route_scaled(&logits, self.top_k, &self.expert_scale)?;
            let gate_up = k.expert_linear_routed(&self.gate_up, &input, &routes)?;
            let inner = gate_up.dim(1)? / 2;
            let act = (gate_up.narrow(1, 0, inner)?.gelu()? * gate_up.narrow(1, inner, inner)?)?;
            let routed = k.expert_scatter(&k.expert_linear(&self.down, &act, &routes)?, &routes)?;
            return Ok(
                (model.rms(dense, &self.post_dense)? + model.rms(&routed, &self.post_routed)?)?
            );
        }
        let routed = self.reference(&input, &logits)?;
        Ok((model.rms(dense, &self.post_dense)? + model.rms(&routed, &self.post_routed)?)?)
    }
    fn reference(&self, input: &Tensor, logits: &Tensor) -> Result<Tensor> {
        let scores = logits.to_vec2::<f32>()?;
        let scales = self.expert_scale.to_vec1::<f32>()?;
        let mut groups = vec![(Vec::new(), Vec::new()); self.experts];
        for (token, scores) in scores.iter().enumerate() {
            let (experts, weights) = math::route(scores, self.top_k)?;
            for (e, w) in experts.into_iter().zip(weights) {
                groups[e]
                    .0
                    .push(u32::try_from(token).map_err(|_| invalid("too many expert tokens"))?);
                groups[e].1.push(w * scales[e]);
            }
        }
        let width = input.dim(1)?;
        let inner = self.down.in_features();
        let mut output = Tensor::zeros(input.shape(), DType::F32, input.device())?;
        for (e, (tokens, scales)) in groups.into_iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let count = tokens.len();
            let ids = Tensor::new(tokens.as_slice(), input.device())?;
            let h = input.index_select(&ids, 0)?;
            let gate_up = self.gate_up.forward_rows(&h, e * 2 * inner, 2 * inner)?;
            let act = (gate_up.narrow(1, 0, inner)?.gelu()? * gate_up.narrow(1, inner, inner)?)?;
            let result = self
                .down
                .forward_rows(&act, e * width, width)?
                .broadcast_mul(&Tensor::from_vec(scales, (count, 1), input.device())?)?;
            output = output.index_add(&ids, &result, 0)?;
        }
        Ok(output)
    }
}
