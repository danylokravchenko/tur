//! Shared fixtures for Qwen3.5 engine tests.
#![allow(dead_code)]

use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf};
use tur_core::Result;
use tur_models::models::qwen3_5::Config;

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
