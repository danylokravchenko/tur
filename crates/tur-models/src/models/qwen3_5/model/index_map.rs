#![forbid(unsafe_code)]
//! Training-free latent sparse attention index maps: calibration keys read back from the
//! state pool, and the leading principal subspace of their position-free dims.

use super::{Mixer, Qwen35};
use crate::hybrid::{Boundaries, HybridModel, PackedForward, SequenceForward};
use candle_core::{DType, Device};
use std::sync::Arc;
use tur_core::{Result, execution::invalid};
use tur_metal::{
    pool::{INDEX_DIM, PoolConfig, StatePool},
    state::{LayerSnapshot, ModelState},
};

/// Prompt tokens per pass while collecting calibration keys.
const FIT_CHUNK: usize = 2048;
/// Subspace iterations of a fit; the map only needs the leading subspace, not its basis.
const FIT_ITERATIONS: usize = 100;

impl Qwen35 {
    /// Each full attention layer's index input rows over `document`: the position-free
    /// dims of every K/V head, concatenated and normalized, at most `rows` per layer spread
    /// evenly across the document, as the reference samples its PCA rows.
    pub fn index_rows(&self, document: &[u32], rows: usize) -> Result<Vec<(usize, Vec<f32>)>> {
        let pool = StatePool::new(
            Arc::new(self.description.clone()),
            &self.device,
            PoolConfig {
                pages: self.description.pages_for(document.len()).max(1),
                window_pages: 0,
                recurrent_slots: 2,
                index_blocks: 0,
            },
        )?;
        let mut state = ModelState::new(&pool);
        for chunk in document.chunks(FIT_CHUNK) {
            let mut reservation = state.reserve(chunk.len(), Boundaries::Final)?;
            reservation.prepare()?;
            let binding = reservation.binding();
            let positions: Vec<usize> =
                (state.position()..state.position() + chunk.len()).collect();
            let sequences = [SequenceForward {
                tokens: 0..chunk.len(),
                state: &binding,
                logits: &[],
                hidden: false,
            }];
            let batch = PackedForward {
                embeddings: &[],
                axial: &[],
                tokens: chunk,
                positions: &positions,
                sequences: &sequences,
                pool: &pool,
            };
            for step in self.forward(&batch)? {
                step?;
            }
            state = reservation.commit(chunk.len())?;
            drop(reservation);
            self.device.synchronize()?;
            pool.reclaim();
        }
        let d = self.dims;
        let layers = state.read_layers()?;
        let mut out = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let (Mixer::Attention(_), Some(LayerSnapshot::Attention { key, .. })) =
                (&layer.mixer, layers.get(index))
            else {
                continue;
            };
            let free = key
                .narrow(2, d.rotary_dim, d.head_dim - d.rotary_dim)?
                .to_device(&Device::Cpu)?;
            let free = free
                .to_dtype(DType::F32)?
                .reshape((document.len(), d.kv_heads * (d.head_dim - d.rotary_dim)))?;
            let all = free.to_vec2::<f32>()?;
            let stride = (all.len() / rows.max(1)).max(1);
            let sampled = all.iter().step_by(stride).take(rows).flat_map(|row| {
                let norm = row
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12);
                row.iter().map(move |v| (f64::from(*v) / norm) as f32)
            });
            out.push((index, sampled.collect()));
        }
        Ok(out)
    }
}

/// A training-free index map `[dim, 64]` from `rows` (`dim` wide, unit length): an
/// orthonormal basis of their centred covariance's leading 64-dimensional subspace, as the
/// reference's PCA map takes its leading right singular vectors. Index scores and clusters
/// depend only on that subspace, so any basis of it maps identically.
pub fn principal_map(rows: &[f32], dim: usize) -> Result<Vec<f32>> {
    let count = rows
        .len()
        .checked_div(dim)
        .filter(|&n| n > INDEX_DIM && n * dim == rows.len());
    let Some(count) = count.filter(|_| dim >= INDEX_DIM) else {
        return Err(invalid(
            "a principal map needs more than 64 whole rows of at least 64 dims",
        ));
    };
    let mut mean = vec![0f64; dim];
    for row in rows.chunks_exact(dim) {
        mean.iter_mut()
            .zip(row)
            .for_each(|(m, v)| *m += f64::from(*v) / count as f64);
    }
    let mut covariance = vec![0f64; dim * dim];
    for row in rows.chunks_exact(dim) {
        let centred: Vec<f64> = row
            .iter()
            .zip(&mean)
            .map(|(v, m)| f64::from(*v) - m)
            .collect();
        for (i, &a) in centred.iter().enumerate() {
            let line = &mut covariance[i * dim..(i + 1) * dim];
            line.iter_mut()
                .zip(&centred)
                .for_each(|(c, &b)| *c = a.mul_add(b, *c));
        }
    }
    // Deterministic start: column j leans on dimension j, so it is never orthogonal to the
    // leading subspace by construction.
    let mut basis: Vec<f64> = (0..dim * INDEX_DIM)
        .map(|k| {
            if k / INDEX_DIM == k % INDEX_DIM {
                1.0
            } else {
                1e-3 * ((k * 7919 % 1000) as f64 / 1000.0 - 0.5)
            }
        })
        .collect();
    orthonormalize(&mut basis, dim)?;
    for _ in 0..FIT_ITERATIONS {
        let mut next = vec![0f64; dim * INDEX_DIM];
        for i in 0..dim {
            let line = &covariance[i * dim..(i + 1) * dim];
            let out = &mut next[i * INDEX_DIM..(i + 1) * INDEX_DIM];
            for (k, &c) in line.iter().enumerate() {
                let column = &basis[k * INDEX_DIM..(k + 1) * INDEX_DIM];
                out.iter_mut()
                    .zip(column)
                    .for_each(|(o, &b)| *o = c.mul_add(b, *o));
            }
        }
        basis = next;
        orthonormalize(&mut basis, dim)?;
    }
    Ok(basis.into_iter().map(|v| v as f32).collect())
}

/// Modified Gram-Schmidt over the 64 columns of a row-major `[dim, 64]` matrix.
fn orthonormalize(basis: &mut [f64], dim: usize) -> Result<()> {
    for j in 0..INDEX_DIM {
        for p in 0..j {
            let dot: f64 = (0..dim)
                .map(|i| basis[i * INDEX_DIM + j] * basis[i * INDEX_DIM + p])
                .sum();
            (0..dim).for_each(|i| {
                basis[i * INDEX_DIM + j] =
                    dot.mul_add(-basis[i * INDEX_DIM + p], basis[i * INDEX_DIM + j]);
            });
        }
        let norm = (0..dim)
            .map(|i| basis[i * INDEX_DIM + j].powi(2))
            .sum::<f64>()
            .sqrt();
        if norm < 1e-12 {
            return Err(invalid("calibration keys span fewer than 64 directions"));
        }
        (0..dim).for_each(|i| basis[i * INDEX_DIM + j] /= norm);
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_principal_map_spans_the_rows_leading_subspace() -> Result<()> {
        let dim = 128;
        // Rows spread over the first 64 dims, with faint noise in the rest.
        let rows: Vec<f32> = (0..400 * dim)
            .map(|k| {
                let wave = ((k * 2_654_435_761) % 1_000_003) as f32 / 1_000_003.0 - 0.5;
                if k % dim < INDEX_DIM {
                    wave
                } else {
                    wave * 1e-3
                }
            })
            .collect();
        let map = principal_map(&rows, dim)?;
        for j in 0..INDEX_DIM {
            let inside: f32 = (0..INDEX_DIM).map(|i| map[i * INDEX_DIM + j].powi(2)).sum();
            assert!(
                inside > 0.999,
                "column {j} keeps {inside} inside the subspace"
            );
        }
        assert!(principal_map(&rows[..64 * dim], dim).is_err());
        Ok(())
    }
}
