//! Latent sparse attention index: Ward buckets over closed blocks of keys, and the keys a
//! decoding query reads. Follows the flat design of Prefrontal-AI's reference implementation;
//! see the book's latent sparse attention chapter.
use super::{Kernels, PagedKv, runtime::to_u32};
use crate::pool::AttentionPool;
use candle_core::{DType, Device, Tensor};
use tur_core::{
    Result,
    execution::{KvFormat, invalid},
};

/// Keys in one index block (a cluster block): clustered once, when its last key is written.
pub const INDEX_BLOCK_TOKENS: usize = 512;
/// Buckets of one index block.
pub const INDEX_BUCKETS: usize = 64;
/// Dimensions of index keys and queries.
pub const INDEX_DIM: usize = 64;
/// Pools of four consecutive keys that one bucket holds at most (16 keys).
pub const BUCKET_POOLS: usize = 4;
const POOLS: usize = 128;
const MAX_TOP_P: usize = 1024;
const MAX_FREE_DIMS: usize = 2048;
const SCORE_CHUNK: usize = 256;
/// Key positions travel as F32, exact below 2^24.
const MAX_POSITION: usize = 1 << 24;

/// One attention layer's index: for each index block, 64 unit centroids `[blocks, 64 * 64]`
/// and each bucket's pools in ascending order `[blocks, 64 * 4]`, as F32 integers with -1
/// past a bucket's size. Pool `i` of block `b` holds keys `512 b + 4 i ..= 512 b + 4 i + 3`.
pub struct IndexPool {
    centroids: Tensor,
    members: Tensor,
    blocks: usize,
}

impl IndexPool {
    /// A zeroed pool of `blocks` index blocks on `device`.
    pub fn new(device: &Device, blocks: usize) -> Result<Self> {
        if blocks == 0 {
            return Err(invalid("an index pool needs at least one block"));
        }
        Ok(Self {
            centroids: Tensor::zeros((blocks, INDEX_BUCKETS * INDEX_DIM), DType::F32, device)?,
            members: Tensor::zeros((blocks, INDEX_BUCKETS * BUCKET_POOLS), DType::F32, device)?,
            blocks,
        })
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub fn centroids(&self) -> &Tensor {
        &self.centroids
    }

    pub fn members(&self) -> &Tensor {
        &self.members
    }
}

/// A layer's map `[kv_heads * (head_dim - rotary), 64]` from the position-free key dims of
/// every K/V head, concatenated, to index space. Rotation covers each head's first `rotary`
/// dims, so the rest score the same before and after it.
pub struct IndexMap {
    map: Tensor,
    kv_heads: usize,
    head_dim: usize,
    rotary: usize,
}

impl IndexMap {
    /// Validate `map` against heads of `head_dim` whose first `rotary` dims rotate.
    pub fn new(map: Tensor, kv_heads: usize, head_dim: usize, rotary: usize) -> Result<Self> {
        let free = head_dim.checked_sub(rotary).filter(|&f| f > 0);
        let rows = free.and_then(|f| f.checked_mul(kv_heads));
        let valid = match (free, rows) {
            (Some(free), Some(rows)) => {
                free.is_multiple_of(32)
                    && rotary.is_multiple_of(32)
                    && rows <= MAX_FREE_DIMS
                    && map.dims() == [rows, INDEX_DIM]
                    && map.dtype() == DType::F32
            }
            _ => false,
        };
        if kv_heads == 0 || !valid {
            return Err(invalid(
                "an index map is F32 [kv_heads * position-free dims, 64], with both dim counts multiples of 32 and at most 2048 rows",
            ));
        }
        Ok(Self {
            map,
            kv_heads,
            head_dim,
            rotary,
        })
    }

    fn free_dims(&self) -> usize {
        self.kv_heads * (self.head_dim - self.rotary)
    }
}

/// Cluster logical index block `block` of the sequence whose page table starts at entry
/// `first_block` of the pass's table into index block `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterJob {
    pub first_block: u32,
    pub block: u32,
    pub slot: u32,
}

/// What a query reads: up to `top_k` keys from its `top_p` best buckets, its `window` most
/// recent keys (its own included) and the first `sink` keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseBudget {
    pub top_k: usize,
    pub top_p: usize,
    pub window: usize,
    pub sink: usize,
}

impl SparseBudget {
    /// The paper's budget for Qwen3.8-27B.
    pub const PAPER: Self = Self {
        top_k: 4096,
        top_p: 1024,
        window: INDEX_BLOCK_TOKENS,
        sink: 4,
    };

    /// Keys one query reads at most. The window must cover a query's open block, whose
    /// keys no bucket holds.
    pub fn width(&self) -> Result<usize> {
        let width = self
            .top_k
            .checked_add(self.window)
            .and_then(|w| w.checked_add(self.sink));
        match width {
            Some(width)
                if self.top_k > 0
                    && (1..=MAX_TOP_P).contains(&self.top_p)
                    && self.window >= INDEX_BLOCK_TOKENS
                    && width < MAX_POSITION =>
            {
                Ok(width)
            }
            _ => Err(invalid(
                "a sparse budget needs top_k > 0, top_p in 1..=1024 and a window of at least 512 keys",
            )),
        }
    }
}

/// Validated index addressing of a pass: token `t` sees the closed blocks before its own
/// position `visible[t] - 1`, logical block `b` at index block `table[first[t] + b]`.
pub struct IndexTable {
    table: Tensor,
    first: Tensor,
    visible: Tensor,
    tokens: usize,
    max_buckets: usize,
    /// Index blocks of the pool the table was validated against.
    blocks: usize,
}

/// Keys each token of a pass reads: positions `[tokens, width]` as F32, -1 past each row's
/// length `[tokens]`. A row lists the expanded buckets best first, then the window in
/// order, then the sinks.
pub struct SparseKeys {
    keys: Tensor,
    lengths: Tensor,
    width: usize,
}

impl SparseKeys {
    pub fn keys(&self) -> &Tensor {
        &self.keys
    }

    pub fn lengths(&self) -> &Tensor {
        &self.lengths
    }

    pub fn width(&self) -> usize {
        self.width
    }
}

#[repr(C)]
struct ClusterParams {
    jobs: u32,
    kv_heads: u32,
    dim: u32,
    rotary: u32,
    page: u32,
    group: u32,
    free_dims: u32,
}

#[repr(C)]
struct ScoreParams {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    rotary: u32,
    free_dims: u32,
    score_stride: u32,
}

#[repr(C)]
struct SelectParams {
    tokens: u32,
    score_stride: u32,
    top_k: u32,
    top_p: u32,
    window: u32,
    sink: u32,
    width: u32,
}

impl Kernels {
    /// Upload index addressing for a pass of `visible.len()` tokens over a pool of `blocks`
    /// index blocks.
    pub fn index_table(
        &self,
        table: &[u32],
        first: &[u32],
        visible: &[u32],
        blocks: usize,
    ) -> Result<IndexTable> {
        if visible.is_empty() || first.len() != visible.len() {
            return Err(invalid("index addressing needs one first entry per token"));
        }
        if let Some(&slot) = table.iter().find(|&&slot| slot as usize >= blocks) {
            return Err(invalid(format!(
                "index block {slot} outside a pool of {blocks}"
            )));
        }
        let mut max_buckets = 0;
        for (&start, &keys) in first.iter().zip(visible) {
            let keys = keys as usize;
            if keys == 0 || keys > MAX_POSITION {
                return Err(invalid("index addressing needs 1..=2^24 visible keys"));
            }
            let closed = (keys - 1) / INDEX_BLOCK_TOKENS;
            if start as usize + closed > table.len() {
                return Err(invalid("a token's closed blocks exceed its index table"));
            }
            max_buckets = max_buckets.max(closed * INDEX_BUCKETS);
        }
        // Every bound buffer must exist, so an empty table uploads one unread entry.
        let tokens = visible.len();
        let table = if table.is_empty() { &[0][..] } else { table };
        let [table, first, visible] = self.runtime.indices_packed([table, first, visible])?;
        Ok(IndexTable {
            table,
            first,
            visible,
            tokens,
            max_buckets,
            blocks,
        })
    }

    /// Cluster each job's closed block of keys from `pool` into its index block. Keys are
    /// read through `kv`'s page table, which must hold every page of each job's block.
    pub fn lsa_cluster(
        &self,
        pool: &AttentionPool,
        kv: &PagedKv,
        map: &IndexMap,
        index: &IndexPool,
        jobs: &[ClusterJob],
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        Self::check_cluster_jobs(pool, kv, map, index, jobs)?;
        let (k_data, k_scales) = self.kv_views(
            &pool.key,
            pool.format,
            kv.slots,
            pool.kv_heads,
            pool.head_dim,
        )?;
        let table = self.runtime.view(&kv.table, DType::U32)?;
        let flat: Vec<u32> = jobs
            .iter()
            .flat_map(|job| [job.first_block, job.block, job.slot])
            .collect();
        let jobs_tensor = self.runtime.indices(&flat)?;
        let job_view = self.runtime.view(&jobs_tensor, DType::U32)?;
        let map_view = self.view(&map.map)?;
        map_view.contiguous("index map")?;
        let centroids = self.view(&index.centroids)?;
        let members = self.view(&index.members)?;
        let params = ClusterParams {
            jobs: to_u32(jobs.len(), "cluster jobs")?,
            kv_heads: to_u32(pool.kv_heads, "kv heads")?,
            dim: to_u32(pool.head_dim, "head dim")?,
            rotary: to_u32(map.rotary, "rotary dims")?,
            page: to_u32(kv.page, "page")?,
            group: to_u32(pool.format.group().unwrap_or(32), "group")?,
            free_dims: to_u32(map.free_dims(), "position-free dims")?,
        };
        let scratch = self.output(jobs.len() * POOLS, INDEX_DIM)?;
        let name = match pool.format {
            KvFormat::F32 => "lsa_cluster_f32",
            KvFormat::F16 => "lsa_cluster_f16",
            KvFormat::Q8 { .. } => "lsa_cluster_q8",
        };
        let pipeline = self.pipeline(name)?;
        let threads = pipeline.threads(self.capabilities().max_threads_per_threadgroup);
        self.runtime
            .dispatch(pipeline, "tur_lsa_cluster", jobs.len(), threads, |e| {
                k_data.bind(e, 0)?;
                // Unscaled formats bind the data again; every declared buffer must be bound.
                k_scales.as_ref().unwrap_or(&k_data).bind(e, 1)?;
                table.bind(e, 2)?;
                job_view.bind(e, 3)?;
                map_view.bind(e, 4)?;
                scratch.bind(e, 5);
                centroids.bind_mut(e, 6)?;
                members.bind_mut(e, 7)?;
                e.set_bytes(8, &params);
                Ok(())
            })
    }

    fn check_cluster_jobs(
        pool: &AttentionPool,
        kv: &PagedKv,
        map: &IndexMap,
        index: &IndexPool,
        jobs: &[ClusterJob],
    ) -> Result<()> {
        if map.kv_heads != pool.kv_heads
            || map.head_dim != pool.head_dim
            || kv.page != pool.page_tokens
            || kv.page == 0
            || !INDEX_BLOCK_TOKENS.is_multiple_of(kv.page)
        {
            return Err(invalid(
                "the index map or pages do not match the attention pool",
            ));
        }
        let table_len = kv.table.dims1()?;
        let pages = INDEX_BLOCK_TOKENS / kv.page;
        let mut slots: Vec<u32> = jobs.iter().map(|job| job.slot).collect();
        slots.sort_unstable();
        if slots.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid("two cluster jobs write one index block"));
        }
        for job in jobs {
            let end_page = (job.block as usize + 1)
                .checked_mul(pages)
                .and_then(|p| p.checked_add(job.first_block as usize));
            let end_key = (job.block as usize + 1).checked_mul(INDEX_BLOCK_TOKENS);
            if job.slot as usize >= index.blocks
                || end_page.is_none_or(|end| end > table_len)
                || end_key.is_none_or(|end| end > MAX_POSITION)
            {
                return Err(invalid(
                    "a cluster job's block lies outside its page table or the index pool",
                ));
            }
        }
        Ok(())
    }

    /// Choose the keys each query of `q` (`[tokens, heads, head_dim]`, rotated or not)
    /// reads from the closed blocks `table` addresses in `index`, under `budget`.
    pub fn lsa_select(
        &self,
        q: &Tensor,
        map: &IndexMap,
        index: &IndexPool,
        table: &IndexTable,
        budget: SparseBudget,
    ) -> Result<SparseKeys> {
        let width = budget.width()?;
        let (tokens, heads, dim) = q.dims3()?;
        if tokens != table.tokens
            || dim != map.head_dim
            || !heads.is_multiple_of(map.kv_heads)
            || table.blocks > index.blocks
        {
            return Err(invalid("sparse selection shapes do not match"));
        }
        let qv = self.view(q)?;
        qv.contiguous("queries")?;
        let map_view = self.view(&map.map)?;
        map_view.contiguous("index map")?;
        let index_table = self.runtime.view(&table.table, DType::U32)?;
        let first = self.runtime.view(&table.first, DType::U32)?;
        let visible = self.runtime.view(&table.visible, DType::U32)?;
        let centroids = self.view(&index.centroids)?;
        let members = self.view(&index.members)?;
        let stride = table.max_buckets.max(1);
        let scores = self.output(tokens, stride)?;
        if table.max_buckets > 0 {
            let params = ScoreParams {
                tokens: to_u32(tokens, "tokens")?,
                heads: to_u32(heads, "heads")?,
                kv_heads: to_u32(map.kv_heads, "kv heads")?,
                dim: to_u32(dim, "head dim")?,
                rotary: to_u32(map.rotary, "rotary dims")?,
                free_dims: to_u32(map.free_dims(), "position-free dims")?,
                score_stride: to_u32(stride, "score stride")?,
            };
            let pipeline = self.pipeline("lsa_score_f32")?;
            let grid = (table.max_buckets.div_ceil(SCORE_CHUNK), tokens);
            let threads = pipeline.threads(SCORE_CHUNK);
            self.runtime
                .dispatch_grid(pipeline, "tur_lsa_score", grid, threads, |e| {
                    qv.bind(e, 0)?;
                    map_view.bind(e, 1)?;
                    centroids.bind(e, 2)?;
                    index_table.bind(e, 3)?;
                    first.bind(e, 4)?;
                    visible.bind(e, 5)?;
                    scores.bind(e, 6);
                    e.set_bytes(7, &params);
                    Ok(())
                })?;
        }
        let params = SelectParams {
            tokens: to_u32(tokens, "tokens")?,
            score_stride: to_u32(stride, "score stride")?,
            top_k: to_u32(budget.top_k, "top-k")?,
            top_p: to_u32(budget.top_p, "top-p")?,
            window: to_u32(budget.window, "window")?,
            sink: to_u32(budget.sink, "sinks")?,
            width: to_u32(width, "width")?,
        };
        let keys = self.output(tokens, width)?;
        let lengths = self.output(tokens, 1)?;
        let pipeline = self.pipeline("lsa_select_f32")?;
        let threads = pipeline.threads(self.capabilities().max_threads_per_threadgroup);
        self.runtime
            .dispatch(pipeline, "tur_lsa_select", tokens, threads, |e| {
                e.set_input_buffer(0, Some(scores.buffer()), 0);
                members.bind(e, 1)?;
                index_table.bind(e, 2)?;
                first.bind(e, 3)?;
                visible.bind(e, 4)?;
                keys.bind(e, 5);
                lengths.bind(e, 6);
                e.set_bytes(7, &params);
                Ok(())
            })?;
        Ok(SparseKeys {
            keys: self.finish(keys, (tokens, width))?,
            lengths: self.finish(lengths, tokens)?,
            width,
        })
    }
}
