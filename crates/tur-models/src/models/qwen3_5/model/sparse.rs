#![forbid(unsafe_code)]
//! Qwen3.5's index maps for latent sparse attention; [`crate::models::sparse`] runs it.

use super::{Mixer, Qwen35};
use crate::models::{SparseAttention, sparse::Sparse};
use tur_core::{Result, TurError, execution::invalid};
use tur_metal::kernels::IndexMap;

impl Qwen35 {
    /// Read attention sparsely as `sparse` describes, through the index map file it names:
    /// one `[kv_heads * (head_dim - rotary), 64]` F32 tensor `layer.{i}` per full attention
    /// layer `i`.
    pub fn sparse_attention(&mut self, sparse: &SparseAttention) -> Result<()> {
        let d = self.dims;
        let tensors = candle_core::safetensors::load(&sparse.map, &self.device)?;
        let mut maps = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            maps.push(match layer.mixer {
                Mixer::Attention(_) => {
                    let map = tensors.get(&format!("layer.{index}")).ok_or_else(|| {
                        TurError::Other(format!("index map lacks attention layer {index}"))
                    })?;
                    Some(IndexMap::new(
                        map.clone(),
                        d.kv_heads,
                        d.head_dim,
                        d.rotary_dim,
                    )?)
                }
                Mixer::Delta(_) => None,
            });
        }
        if tensors.len() != maps.iter().flatten().count() {
            return Err(TurError::Other(
                "index map holds layers this model's attention lacks".into(),
            ));
        }
        let attached = Sparse::new(sparse, maps)?;
        let fused = self
            .fused
            .as_mut()
            .ok_or_else(|| invalid("latent sparse attention needs fused Metal execution"))?;
        fused.sparse = Some(attached);
        Ok(())
    }
}
