#![forbid(unsafe_code)]
//! Latent sparse attention on the fused path, shared by the models that support it. Every
//! pass clusters, in each full attention layer that writes K/V, the blocks of 512 keys it
//! closes; past the crossover, each query of a decoding or verifying sequence then reads
//! only the keys its layer's index selects for it, and with sparse prefill each query tile
//! of a prefill chunk reads the keys selected for the tile. A pass mixing kinds of read
//! dispatches each kind's rows on their own.

use crate::models::SparseAttention;
use candle_core::Tensor;
use tur_core::{Result, execution::invalid};
use tur_metal::{
    kernels::{ClusterJob, Heads, IndexMap, IndexTable, Kernels, PagedKv, SparseBudget},
    pool::{IndexPool, StatePool, index_blocks_for},
};

/// Longest span that reads per query: decoding and verification windows. Longer spans are
/// prefill chunks, whose queries read densely or, with sparse prefill, per query tile.
const SPARSE_SPAN: usize = 16;

/// A model's index maps, by pool layer, its budget and its crossover.
pub(crate) struct Sparse {
    settings: SparseAttention,
    maps: Vec<Option<IndexMap>>,
    budget: SparseBudget,
}

impl Sparse {
    /// `settings` over `maps`, one per pool layer: `None` for layers that read densely.
    pub(crate) fn new(settings: &SparseAttention, maps: Vec<Option<IndexMap>>) -> Result<Self> {
        if maps.iter().all(Option::is_none) {
            return Err(invalid("index map covers no layer"));
        }
        let budget = SparseBudget {
            top_k: settings.top_k,
            top_p: settings.top_p,
            window: settings.window,
            sink: settings.sink,
        };
        budget.width()?;
        Ok(Self {
            settings: settings.clone(),
            maps,
            budget,
        })
    }

    /// Index map of pool layer `layer`; `None` for layers that read densely.
    pub(crate) fn map(&self, layer: usize) -> Option<&IndexMap> {
        self.maps.get(layer).and_then(Option::as_ref)
    }

    /// Scratch of one attention layer's selection over `n` rows ending at `contexts`.
    pub(crate) fn workspace(&self, n: usize, contexts: &[usize]) -> Result<usize> {
        self.settings
            .workspace_bytes(n, contexts)
            .ok_or_else(|| invalid("sparse workspace overflow"))
    }
}

/// One sequence of a pass as the plan sees it: where its pages start in the pass's page
/// table, its index blocks, its committed tokens, the tokens it brings, and the positions
/// of its queries in packed order (every new token's, or the rows a layer keeps).
pub(crate) struct SpanRows<'a> {
    pub(crate) page_base: u32,
    pub(crate) indexes: &'a [u32],
    pub(crate) past: usize,
    pub(crate) len: usize,
    pub(crate) queries: Vec<usize>,
}

/// The pass's page table, laid out as the layer addressing it plans for, and the K/V pool
/// it addresses.
pub(crate) struct PassTable<'a> {
    pub(crate) pages: &'a [u32],
    pub(crate) page: usize,
    pub(crate) slots: usize,
}

/// What a pass's attention layers share: the blocks it clusters, and how its rows read.
pub(crate) struct SparsePass {
    jobs: Vec<ClusterJob>,
    rows: Option<SparseRows>,
}

/// How a row reads attention: every key, its own selected keys, or the keys selected for
/// its query tile.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Read {
    Dense,
    Query,
    Tile,
}

/// A pass with sparse rows: each kind of read's rows and addressing, in `dense`, `query`,
/// `tile` order, and the order that puts their outputs back in packed row order (`None`
/// when one kind holds every row).
struct SparseRows {
    groups: Vec<Group>,
    order: Option<Tensor>,
}

/// Rows of one kind of read, sparse ones with their index addressing. `rows` and `kv` are
/// `None` when the group holds every row of the pass, which then reads through the pass's
/// own addressing.
struct Group {
    rows: Option<Tensor>,
    kv: Option<PagedKv>,
    table: Option<IndexTable>,
}

/// Per-row addressing of one pass, in packed order.
#[derive(Default)]
struct Rows {
    page_first: Vec<u32>,
    index_first: Vec<u32>,
    visible: Vec<u32>,
    read: Vec<Read>,
}

/// Clustering jobs and sparse rows of a pass over `spans`, in packed row order.
pub(crate) fn plan(
    k: &Kernels,
    sparse: &Sparse,
    pool: &StatePool,
    table: &PassTable<'_>,
    spans: &[SpanRows<'_>],
) -> Result<SparsePass> {
    if !pool.indexed() {
        return Err(invalid(
            "latent sparse attention needs a pool with index blocks",
        ));
    }
    let (mut indexes, mut jobs, mut rows) = (Vec::new(), Vec::new(), Rows::default());
    for span in spans {
        let index_base = to_u32(indexes.len())?;
        indexes.extend_from_slice(span.indexes);
        for block in index_blocks_for(span.past)..index_blocks_for(span.past + span.len) {
            let slot = span
                .indexes
                .get(block)
                .copied()
                .ok_or_else(|| invalid("a pass closes a block it holds no index block for"))?;
            jobs.push(ClusterJob {
                first_block: span.page_base,
                block: to_u32(block)?,
                slot,
            });
        }
        let sparse_read = if span.len <= SPARSE_SPAN {
            Some(Read::Query)
        } else {
            sparse.settings.prefill.then_some(Read::Tile)
        };
        for &position in &span.queries {
            let keys = position + 1;
            rows.page_first.push(span.page_base);
            rows.index_first.push(index_base);
            rows.visible.push(to_u32(keys)?);
            rows.read.push(match sparse_read {
                Some(read) if keys > sparse.settings.crossover => read,
                _ => Read::Dense,
            });
        }
    }
    let addressing = Addressing {
        table,
        indexes: &indexes,
        blocks: first_index(sparse, pool)?.blocks(),
    };
    Ok(SparsePass {
        jobs,
        rows: sparse_rows(k, &rows, &addressing)?,
    })
}

/// The pass's page and index tables and the pools they address.
struct Addressing<'a> {
    table: &'a PassTable<'a>,
    indexes: &'a [u32],
    blocks: usize,
}

fn sparse_rows(k: &Kernels, rows: &Rows, at: &Addressing<'_>) -> Result<Option<SparseRows>> {
    if rows.read.iter().all(|&r| r == Read::Dense) {
        return Ok(None);
    }
    let kinds = [Read::Dense, Read::Query, Read::Tile];
    let members: Vec<Vec<usize>> = kinds
        .iter()
        .map(|&kind| {
            (0..rows.read.len())
                .filter(|&r| rows.read[r] == kind)
                .collect()
        })
        .collect();
    let whole = members.iter().any(|m| m.len() == rows.read.len());
    let mut order = vec![0u32; rows.read.len()];
    for (at, &r) in members.iter().flatten().enumerate() {
        order[r] = to_u32(at)?;
    }
    let device = k.device();
    let mut groups = Vec::new();
    for (&read, which) in kinds.iter().zip(&members) {
        if which.is_empty() {
            continue;
        }
        let pick = |source: &[u32]| which.iter().map(|&r| source[r]).collect::<Vec<_>>();
        let (index_first, visible) = (pick(&rows.index_first), pick(&rows.visible));
        let kv = k.paged_kv(
            at.table.page,
            at.table.slots,
            at.table.pages,
            &pick(&rows.page_first),
            &visible,
        )?;
        let table = match read {
            Read::Dense => None,
            Read::Query => Some(k.index_table(at.indexes, &index_first, &visible, at.blocks)?),
            Read::Tile => {
                Some(k.index_table_tiles(at.indexes, &index_first, &visible, &kv, at.blocks)?)
            }
        };
        let indices: Vec<u32> = which.iter().map(|&r| to_u32(r)).collect::<Result<_>>()?;
        groups.push(Group {
            rows: (!whole)
                .then(|| Tensor::new(indices.as_slice(), device))
                .transpose()?,
            kv: (!whole || read == Read::Tile).then_some(kv),
            table,
        });
    }
    Ok(Some(SparseRows {
        groups,
        order: (!whole)
            .then(|| Tensor::new(order.as_slice(), device))
            .transpose()?,
    }))
}

fn first_index<'p>(sparse: &Sparse, pool: &'p StatePool) -> Result<&'p IndexPool> {
    let layer = sparse
        .maps
        .iter()
        .position(Option::is_some)
        .ok_or_else(|| invalid("index map covers no layer"))?;
    pool.index(layer)
}

fn to_u32(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid("sparse addressing overflow"))
}

/// How a layer's attention output is finished: through a sigmoid gate read from a
/// projection, or ungated with an explicit score scale.
#[derive(Clone, Copy)]
pub(crate) enum Output<'a> {
    Gated(&'a Tensor, Heads),
    Ungated(f64),
}

/// One layer's attention under `plan`, over pool layer `layer`: cluster the blocks the pass
/// closes when `owner` (the layer writes the K/V it reads), then read each group's rows its
/// own way. `None` when the layer has no map or the pass no sparse rows, so the caller reads
/// densely.
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per input the attention dispatches bind"
)]
pub(crate) fn attend(
    k: &Kernels,
    sparse: &Sparse,
    plan: &SparsePass,
    (layer, owner): (usize, bool),
    q: &Tensor,
    output: Output<'_>,
    states: &StatePool,
    paging: &PagedKv,
) -> Result<Option<Tensor>> {
    let Some(map) = sparse.map(layer) else {
        return Ok(None);
    };
    let (pool, index) = (states.attention(layer)?, states.index(layer)?);
    if owner {
        k.lsa_cluster(pool, paging, map, index, &plan.jobs)?;
    }
    let Some(rows) = &plan.rows else {
        return Ok(None);
    };
    let mut outputs = Vec::with_capacity(rows.groups.len());
    for group in &rows.groups {
        let pick = |t: &Tensor| {
            group
                .rows
                .as_ref()
                .map_or_else(|| Ok(t.clone()), |which| t.index_select(which, 0))
        };
        let q = pick(q)?;
        let kv = group.kv.as_ref().unwrap_or(paging);
        let keys = group
            .table
            .as_ref()
            .map(|table| k.lsa_select(&q, map, index, table, sparse.budget))
            .transpose()?;
        outputs.push(match (output, &keys) {
            (Output::Gated(source, gate), None) => {
                k.attention(&q, pool, kv, &pick(source)?, gate)?
            }
            (Output::Gated(source, gate), Some(keys)) => {
                k.attention_sparse(&q, pool, kv, &pick(source)?, gate, keys)?
            }
            (Output::Ungated(scale), None) => k.attention_ungated(&q, pool, kv, None, scale)?,
            (Output::Ungated(scale), Some(keys)) => {
                k.attention_sparse_ungated(&q, pool, kv, scale, keys)?
            }
        });
    }
    match &rows.order {
        None => outputs
            .pop()
            .map(Some)
            .ok_or_else(|| invalid("a sparse pass has no rows")),
        Some(order) => Ok(Some(Tensor::cat(&outputs, 0)?.index_select(order, 0)?)),
    }
}
