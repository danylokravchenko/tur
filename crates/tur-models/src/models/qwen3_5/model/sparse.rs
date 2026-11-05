#![forbid(unsafe_code)]
//! Latent sparse attention on the fused path. Every pass clusters, in each full attention
//! layer, the blocks of 512 keys it closes; queries of decoding and verifying sequences past
//! the crossover then read only the keys their layer's index selects. Prefill chunks read
//! densely, and a pass mixing both splits its rows in two dispatches.

use super::{Mixer, Pass, Qwen35};
use crate::models::SparseAttention;
use candle_core::Tensor;
use tur_core::{Result, TurError, execution::invalid};
use tur_metal::{
    kernels::{ClusterJob, Heads, IndexMap, IndexTable, Kernels, PagedKv, SparseBudget},
    pool::{IndexPool, StatePool, index_blocks_for},
};

/// Longest span that reads sparsely: decoding and verification windows; longer spans are
/// prefill chunks, whose queries keep the tiled dense kernel.
const SPARSE_SPAN: usize = 16;

/// A model's index maps, by layer, its budget and its crossover.
pub(super) struct Sparse {
    settings: SparseAttention,
    maps: Vec<Option<IndexMap>>,
    budget: SparseBudget,
    crossover: usize,
}

impl Sparse {
    /// Index map of `layer`; `None` for recurrent layers and the draft head.
    pub(super) fn map(&self, layer: usize) -> Option<&IndexMap> {
        self.maps.get(layer).and_then(Option::as_ref)
    }

    /// Scratch of one attention layer's selection over `n` rows ending at `contexts`.
    pub(super) fn workspace(&self, n: usize, contexts: &[usize]) -> Result<usize> {
        self.settings
            .workspace_bytes(n, contexts)
            .ok_or_else(|| invalid("sparse workspace overflow"))
    }
}

/// What a pass's attention layers share: the blocks it clusters, and which rows read
/// sparsely.
pub(super) struct SparsePass {
    jobs: Vec<ClusterJob>,
    rows: Option<SparseRows>,
}

/// Rows that read sparsely, with their index addressing; `split` is set when other rows
/// of the pass read densely.
struct SparseRows {
    table: IndexTable,
    split: Option<Split>,
}

/// A pass whose rows read both ways: each group's rows and page addressing, and the order
/// that puts `dense | sparse` outputs back in packed row order.
struct Split {
    dense: Tensor,
    sparse: Tensor,
    order: Tensor,
    dense_kv: PagedKv,
    sparse_kv: PagedKv,
}

/// Per-row addressing of one pass, in packed order.
#[derive(Default)]
struct Rows {
    page_first: Vec<u32>,
    index_first: Vec<u32>,
    visible: Vec<u32>,
    sparse: Vec<bool>,
}

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
        let budget = SparseBudget {
            top_k: sparse.top_k,
            top_p: sparse.top_p,
            window: sparse.window,
            sink: sparse.sink,
        };
        budget.width()?;
        let fused = self
            .fused
            .as_mut()
            .ok_or_else(|| invalid("latent sparse attention needs fused Metal execution"))?;
        fused.sparse = Some(Sparse {
            settings: sparse.clone(),
            maps,
            budget,
            crossover: sparse.crossover,
        });
        Ok(())
    }
}

/// Clustering jobs and sparse rows of `pass`, whose spans' page tables lie in the pass's
/// table in span order, as `Fused::paging` lays them out.
pub(super) fn plan(
    k: &Kernels,
    sparse: &Sparse,
    pass: &Pass<'_, '_>,
    page: usize,
    slots: usize,
) -> Result<SparsePass> {
    if !pass.pool.indexed() {
        return Err(invalid(
            "latent sparse attention needs a pool with index blocks",
        ));
    }
    let (mut pages, mut indexes, mut jobs, mut rows) =
        (Vec::new(), Vec::new(), Vec::new(), Rows::default());
    for span in pass.spans {
        let page_base = to_u32(pages.len())?;
        let index_base = to_u32(indexes.len())?;
        pages.extend_from_slice(&span.addresses.pages);
        indexes.extend_from_slice(&span.addresses.indexes);
        let end = span.past + span.len;
        for block in index_blocks_for(span.past)..index_blocks_for(end) {
            let slot = span
                .addresses
                .indexes
                .get(block)
                .copied()
                .ok_or_else(|| invalid("a pass closes a block it holds no index block for"))?;
            jobs.push(ClusterJob {
                first_block: page_base,
                block: to_u32(block)?,
                slot,
            });
        }
        for t in 0..span.len {
            let keys = span.past + t + 1;
            rows.page_first.push(page_base);
            rows.index_first.push(index_base);
            rows.visible.push(to_u32(keys)?);
            rows.sparse
                .push(span.len <= SPARSE_SPAN && keys > sparse.crossover);
        }
    }
    let blocks = first_index(sparse, pass)?.blocks();
    Ok(SparsePass {
        jobs,
        rows: sparse_rows(k, &rows, &pages, &indexes, page, slots, blocks)?,
    })
}

fn sparse_rows(
    k: &Kernels,
    rows: &Rows,
    pages: &[u32],
    indexes: &[u32],
    page: usize,
    slots: usize,
    blocks: usize,
) -> Result<Option<SparseRows>> {
    let picked: Vec<usize> = (0..rows.sparse.len()).filter(|&r| rows.sparse[r]).collect();
    if picked.is_empty() {
        return Ok(None);
    }
    let pick =
        |source: &[u32], which: &[usize]| which.iter().map(|&r| source[r]).collect::<Vec<_>>();
    let table = k.index_table(
        indexes,
        &pick(&rows.index_first, &picked),
        &pick(&rows.visible, &picked),
        blocks,
    )?;
    if picked.len() == rows.sparse.len() {
        return Ok(Some(SparseRows { table, split: None }));
    }
    let dense: Vec<usize> = (0..rows.sparse.len())
        .filter(|&r| !rows.sparse[r])
        .collect();
    let mut order = vec![0u32; rows.sparse.len()];
    for (at, &r) in dense.iter().chain(&picked).enumerate() {
        order[r] = to_u32(at)?;
    }
    let device = k.device();
    let index = |which: &[usize]| -> Result<Tensor> {
        let which: Vec<u32> = which.iter().map(|&r| to_u32(r)).collect::<Result<_>>()?;
        Ok(Tensor::new(which.as_slice(), device)?)
    };
    let kv = |which: &[usize]| {
        k.paged_kv(
            page,
            slots,
            pages,
            &pick(&rows.page_first, which),
            &pick(&rows.visible, which),
        )
    };
    Ok(Some(SparseRows {
        table,
        split: Some(Split {
            dense: index(&dense)?,
            sparse: index(&picked)?,
            order: Tensor::new(order.as_slice(), device)?,
            dense_kv: kv(&dense)?,
            sparse_kv: kv(&picked)?,
        }),
    }))
}

fn first_index<'p>(sparse: &Sparse, pass: &'p Pass<'_, '_>) -> Result<&'p IndexPool> {
    let layer = sparse
        .maps
        .iter()
        .position(Option::is_some)
        .ok_or_else(|| invalid("index map covers no layer"))?;
    pass.pool.index(layer)
}

fn to_u32(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid("sparse addressing overflow"))
}

/// One layer's attention under `plan`: cluster the blocks the pass closes, then read the
/// selected keys for its sparse rows and every key for the rest. `None` when the pass has
/// no sparse rows, so the caller reads densely.
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per input the attention dispatches bind"
)]
pub(super) fn attend(
    k: &Kernels,
    sparse: &Sparse,
    plan: &SparsePass,
    layer: usize,
    (q, q_gate, gate): (&Tensor, &Tensor, Heads),
    states: &StatePool,
    paging: &PagedKv,
) -> Result<Option<Tensor>> {
    let Some(map) = sparse.map(layer) else {
        return Ok(None);
    };
    let (pool, index) = (states.attention(layer)?, states.index(layer)?);
    k.lsa_cluster(pool, paging, map, index, &plan.jobs)?;
    let Some(rows) = &plan.rows else {
        return Ok(None);
    };
    let Some(split) = &rows.split else {
        let keys = k.lsa_select(q, map, index, &rows.table, sparse.budget)?;
        return Ok(Some(
            k.attention_sparse(q, pool, paging, q_gate, gate, &keys)?,
        ));
    };
    let (qs, gs) = (
        q.index_select(&split.sparse, 0)?,
        q_gate.index_select(&split.sparse, 0)?,
    );
    let keys = k.lsa_select(&qs, map, index, &rows.table, sparse.budget)?;
    let read = k.attention_sparse(&qs, pool, &split.sparse_kv, &gs, gate, &keys)?;
    let (qd, gd) = (
        q.index_select(&split.dense, 0)?,
        q_gate.index_select(&split.dense, 0)?,
    );
    let dense = k.attention(&qd, pool, &split.dense_kv, &gd, gate)?;
    Ok(Some(
        Tensor::cat(&[dense, read], 0)?.index_select(&split.order, 0)?,
    ))
}
