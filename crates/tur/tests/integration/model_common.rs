//! Shared fixtures for Qwen3.5 engine tests.
#![allow(dead_code, reason = "some fixtures serve only Metal-gated tests")]

use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tur_core::{Result, execution::invalid};
use tur_engine::hybrid::{
    Boundaries, HybridEngine, HybridModel, ModelState, PackedForward, PoolConfig, Reservation,
    SamplingOptions, SequenceForward, SequenceStep, StatePool,
};
use tur_models::models::qwen3_5::{Config, Qwen35};

/// Deterministic values in `(-scale, scale)`.
pub(crate) fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0, -1.0) * scale
        })
        .collect()
}

/// A random BF16 checkpoint directory, removed on drop.
pub(crate) struct Checkpoint(pub PathBuf);

impl Drop for Checkpoint {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// Write `config` and deterministic random weights of every declared tensor; norms use
/// `norm_scale` and everything else `scale`.
pub(crate) fn checkpoint(config: &str, scale: f32, norm_scale: f32) -> Result<Checkpoint> {
    let dir = std::env::temp_dir().join(format!("tur-qwen35-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("config.json"), config)?;
    let parsed = Config::from_json(config)?;
    let mut tensors = HashMap::new();
    for (seed, (name, shape)) in parsed.weight_shapes()?.into_iter().enumerate() {
        let count = shape.iter().product();
        let scale = if name.contains("norm") {
            norm_scale
        } else {
            scale
        };
        let data = values(count, seed as u64 + 1, scale);
        tensors.insert(
            name,
            Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(DType::BF16)?,
        );
    }
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors"))?;
    Ok(Checkpoint(dir))
}

/// Largest absolute difference relative to the largest reference magnitude.
pub(crate) fn max_relative(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    // Zip ignores missing tails, and max can discard NaN differences.
    assert_eq!(a.len(), b.len(), "numerical comparison row count");
    for (index, (actual, expected)) in a.iter().zip(b).enumerate() {
        assert_eq!(actual.len(), expected.len(), "row {index} width");
        assert!(
            actual.iter().chain(expected).all(|value| value.is_finite()),
            "nonfinite numerical comparison in row {index}"
        );
    }
    let scale = b
        .iter()
        .flatten()
        .fold(0f32, |m, v| m.max(v.abs()))
        .max(1e-6);
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// A state pool for `model` with `pages` K/V pages and `slots` recurrent slots.
pub(crate) fn pool(model: &Qwen35, pages: usize, slots: usize) -> Result<StatePool> {
    StatePool::new(
        Arc::new(model.description()),
        model.device(),
        PoolConfig {
            pages,
            recurrent_slots: slots,
        },
    )
}

/// One forward pass of `(state, new tokens, boundaries)` sequences sharing a pool, with
/// logits at every new token. The reservations hold each pass's writes until committed.
pub(crate) fn forward(
    model: &impl HybridModel,
    sequences: &[(&ModelState, &[u32], Boundaries)],
) -> Result<(Vec<Result<SequenceStep>>, Vec<Reservation>)> {
    let pool = sequences
        .first()
        .map(|(state, _, _)| state.pool().clone())
        .ok_or_else(|| invalid("empty pass"))?;
    let mut reservations = sequences
        .iter()
        .map(|(state, tokens, boundaries)| state.reserve(tokens.len(), *boundaries))
        .collect::<Result<Vec<_>>>()?;
    for reservation in &mut reservations {
        reservation.prepare()?;
    }
    let bindings: Vec<_> = reservations.iter().map(Reservation::binding).collect();
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut ranges = Vec::new();
    for (state, new, _) in sequences {
        let start = tokens.len();
        tokens.extend_from_slice(new);
        positions.extend(state.position()..state.position() + new.len());
        ranges.push(start..tokens.len());
    }
    let logits: Vec<Vec<usize>> = ranges.iter().map(|r| r.clone().collect()).collect();
    let forwards: Vec<SequenceForward<'_>> = bindings
        .iter()
        .zip(&ranges)
        .zip(&logits)
        .map(|((state, range), logits)| SequenceForward {
            tokens: range.clone(),
            state,
            logits,
            hidden: false,
        })
        .collect();
    let steps = model.forward(&PackedForward {
        tokens: &tokens,
        positions: &positions,
        sequences: &forwards,
        pool: &pool,
    })?;
    Ok((steps, reservations))
}

/// Logits rows of a tensor on any device.
pub(crate) fn rows(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(tensor.to_device(&Device::Cpu)?.to_vec2::<f32>()?)
}

/// Feed `tokens` in `chunk`-sized pieces from `state`; returns every position's logits
/// and the final state. Released pages are reclaimed after each synchronized pass.
pub(crate) fn prefill(
    model: &impl HybridModel,
    state: ModelState,
    tokens: &[u32],
    chunk: usize,
) -> Result<(Vec<Vec<f32>>, ModelState)> {
    let mut state = state;
    let mut out = Vec::new();
    for piece in tokens.chunks(chunk) {
        let (mut steps, reservations) = forward(model, &[(&state, piece, Boundaries::Final)])?;
        let step = steps.pop().ok_or_else(|| invalid("missing step"))??;
        out.extend(rows(&step.logits)?);
        let reservation = reservations
            .first()
            .ok_or_else(|| invalid("missing reservation"))?;
        state = reservation.commit(piece.len())?;
        drop(reservations);
        state.pool().device().synchronize()?;
        state.pool().reclaim();
    }
    Ok((out, state))
}

pub(crate) fn generate<M: HybridModel>(
    model: Arc<M>,
    device: Device,
    pool: PoolConfig,
    prompt: &[u32],
    stops: &[u32],
    limit: usize,
) -> Result<Vec<u32>> {
    let mut engine = HybridEngine::new(model, device, pool)?;
    let h = engine.allocate(0, SamplingOptions::default(), None)?;
    let mut out = Vec::new();
    let mut next = prompt.to_vec();
    while out.len() < limit {
        let meta = engine.metadata(h.request)?;
        let n = next.len();
        engine.submit(PackedExecution {
            positions: (0..n).map(|i| meta.position + i).collect(),
            tokens: std::mem::take(&mut next),
            sequences: vec![SequenceDescriptor {
                state: meta.handle,
                tokens: 0..n,
                outputs: vec![n - 1],
                speculative: false,
            }],
        })?;
        let token = engine
            .finish()?
            .pop()
            .ok_or_else(|| invalid("missing result"))?
            .outcome?
            .outputs
            .first()
            .map(|o| o.token)
            .ok_or_else(|| invalid("no output"))?;
        out.push(token);
        if stops.contains(&token) {
            break;
        }
        next = vec![token];
    }
    Ok(out)
}
