//! Shared fixtures for Qwen3.5 engine tests.
#![allow(dead_code)]

use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tur_core::{Result, execution::invalid};
use tur_engine::hybrid::{
    Boundaries, HybridModel, ModelState, PackedForward, PoolConfig, Reservation, SequenceForward,
    SequenceStep, StatePool,
};
use tur_models::models::qwen3_5::{Config, Qwen35};

/// Deterministic values in `(-scale, scale)`.
pub fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// A random BF16 checkpoint directory, removed on drop.
pub struct Checkpoint(pub PathBuf);

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write `config` and deterministic random weights of every declared tensor; norms use
/// `norm_scale` and everything else `scale`.
pub fn checkpoint(config: &str, scale: f32, norm_scale: f32) -> Result<Checkpoint> {
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
pub fn max_relative(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
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
pub fn pool(model: &Qwen35, pages: usize, slots: usize) -> Result<StatePool> {
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
pub fn forward(
    model: &Qwen35,
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
pub fn rows(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(tensor.to_device(&Device::Cpu)?.to_vec2::<f32>()?)
}

/// Feed `tokens` in `chunk`-sized pieces from `state`; returns every position's logits
/// and the final state. Released pages are reclaimed after each synchronized pass.
pub fn prefill(
    model: &Qwen35,
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
        model.device().synchronize()?;
        state.pool().reclaim();
    }
    Ok((out, state))
}
