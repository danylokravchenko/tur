#![forbid(unsafe_code)]
//! Gemma 4's index maps and pass plans for latent sparse attention; the shared
//! [`crate::models::sparse`] runs it. Full attention layers that own K/V cluster it, and the
//! layers sharing it select from the owner's index through the owner's map.

use super::{Gemma4, Pass, Span};
use crate::models::{
    SparseAttention,
    paging::Tables,
    sparse::{self, PassTable, SpanRows, Sparse, SparsePass},
};
use tur_core::{Result, TurError, execution::invalid};
use tur_metal::kernels::{IndexMap, Kernels};

impl Gemma4 {
    /// Read full attention sparsely as `sparse` describes, through the index map file it
    /// names: one `[kv_heads * position-free dims, 64]` F32 tensor `layer.{i}` per full
    /// attention layer `i` that owns its K/V. Proportional rotation leaves each half of a
    /// head unrotated past its rotated pairs.
    pub fn sparse_attention(&mut self, sparse: &SparseAttention) -> Result<()> {
        let tensors = candle_core::safetensors::load(&sparse.map, &self.device)?;
        let config = &self.config;
        let mut maps: Vec<Option<IndexMap>> =
            (0..self.description.layers.len()).map(|_| None).collect();
        let mut used = 0;
        for index in 0..config.layers() {
            if config.windowed(index) || config.owners[index] != index {
                continue;
            }
            let map = tensors.get(&format!("layer.{index}")).ok_or_else(|| {
                TurError::Other(format!("index map lacks full attention layer {index}"))
            })?;
            let dim = config.dim(index);
            let pairs = super::rotated_pairs(dim, config.rope(index));
            let slot = maps
                .get_mut(config.state_layer(index))
                .ok_or_else(|| invalid("full attention layer outside the state pool"))?;
            *slot = Some(IndexMap::split_rotary(
                map.clone(),
                config.kv_heads(index),
                dim,
                pairs,
            )?);
            used += 1;
        }
        if tensors.len() != used {
            return Err(TurError::Other(
                "index map holds layers this model's full attention lacks".into(),
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

    /// The sparse plan of `pass` over its full attention page table, for every row or, from
    /// the first K/V-sharing layer, for the `kept` rows, laid out as `Paging` lays out each.
    pub(super) fn sparse_plan(
        &self,
        k: &Kernels,
        sparse: &Sparse,
        pass: &Pass<'_, '_>,
        kept: Option<&[u32]>,
    ) -> Result<SparsePass> {
        let mut tables = Tables::default();
        let mut spans = Vec::with_capacity(pass.spans.len());
        for s in pass.spans {
            let queries = queries(s, kept);
            if queries.is_empty() {
                continue;
            }
            spans.push(SpanRows {
                page_base: tables.push(&s.addresses)?,
                indexes: &s.addresses.indexes,
                past: s.past,
                len: s.len,
                queries,
            });
        }
        let first = (0..self.config.layers())
            .find(|&i| !self.config.windowed(i))
            .ok_or_else(|| invalid("model has no full attention layer"))?;
        let table = PassTable {
            pages: &tables.full,
            page: self.description.kv.page_tokens,
            slots: pass
                .pool
                .attention(self.config.state_layer(first))?
                .key
                .data
                .dim(0)?,
        };
        sparse::plan(k, sparse, pass.pool, &table, &spans)
    }
}

/// Positions of `s`'s query rows: every new token's, or those of its rows in `kept`.
fn queries(s: &Span<'_>, kept: Option<&[u32]>) -> Vec<usize> {
    let rows = s.start..s.start + s.len;
    kept.map_or_else(
        || (s.past..s.past + s.len).collect(),
        |kept| {
            kept.iter()
                .map(|&r| r as usize)
                .filter(|r| rows.contains(r))
                .map(|r| s.past + r - s.start)
                .collect()
        },
    )
}
