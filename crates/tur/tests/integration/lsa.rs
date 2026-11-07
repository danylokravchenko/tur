#![cfg(feature = "metal")]
//! Latent sparse attention index kernels against the pinned reference implementation's
//! buckets and key lists (`scripts/lsa_reference.py golden`), and against an independent F64
//! scalar reference over every K/V format, more heads, another head size and several
//! sequences in one pass.

use crate::kernels::{read, relative_error, setup, sigmoid, stored};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use tur_core::{Result, TurError, execution::KvFormat};
use tur_metal::{
    kernels::{ClusterJob, IndexMap, IndexTable, Kernels, PagedKv, SparseBudget, Tuning},
    pool::{AttentionPool, BUCKET_POOLS, INDEX_BLOCK_TOKENS, INDEX_BUCKETS, INDEX_DIM, IndexPool},
};

const PAGE: usize = 16;
const POOL: usize = 4;
const POOLS: usize = INDEX_BLOCK_TOKENS / POOL;
const BUCKET_CAP: usize = 16;

/// Values in [-1, 1) on a 2^-23 grid from a 32-bit hash of (seed, index), as the golden
/// script's `unit`, so both sides build bit-identical inputs.
fn unit(seed: u32, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let mut x = seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u32);
            x ^= x >> 16;
            x = x.wrapping_mul(0x7FEB_352D);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846C_A68B);
            x ^= x >> 16;
            (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// Synthetic keys leaning toward one of `topics` directions per run of `segment` tokens,
/// and queries leaning toward topic `7 p mod topics` at position `p`.
struct Case {
    kv_heads: usize,
    heads: usize,
    dim: usize,
    rotary: usize,
    /// Whether `rotary` counts rotated pairs `(i, i + dim / 2)`, as proportional rotation
    /// rotates them, rather than a rotated prefix of each head.
    split: bool,
    segment: usize,
    topics: usize,
    /// Tokens of each sequence and the positions whose queries select.
    sequences: Vec<(usize, Vec<usize>)>,
}

impl Case {
    /// The case `scripts/lsa_reference.py golden` writes.
    fn golden() -> Self {
        Self {
            kv_heads: 2,
            heads: 4,
            dim: 256,
            rotary: 64,
            split: false,
            segment: 23,
            topics: 6,
            sequences: vec![(1700, vec![511, 512, 1023, 1024, 1300, 1535, 1536, 1699])],
        }
    }

    fn free(&self) -> usize {
        if self.split {
            self.dim - 2 * self.rotary
        } else {
            self.dim - self.rotary
        }
    }

    /// Head dim of position-free dim `e`.
    fn free_dim(&self, e: usize) -> usize {
        let run = self.dim / 2 - self.rotary;
        if self.split && e >= run {
            self.dim / 2 + self.rotary + (e - run)
        } else {
            self.rotary + e
        }
    }

    /// Keys `[tokens, kv_heads, dim]` of sequence `s`; the first matches the golden script.
    fn keys(&self, s: usize) -> Vec<f32> {
        let (tokens, row) = (self.sequences[s].0, self.kv_heads * self.dim);
        let topics = unit(11, self.topics * row);
        let noise = unit(12 + 100 * s as u32, tokens * row);
        (0..tokens * row)
            .map(|i| {
                let topic = (i / row / self.segment) % self.topics;
                // Halving is exact, so fusing rounds once, as the script's add does.
                noise[i].mul_add(0.5, topics[topic * row + i % row])
            })
            .collect()
    }

    /// Queries `[positions, heads, dim]` of sequence `s`.
    fn queries(&self, s: usize) -> Vec<f32> {
        let positions = &self.sequences[s].1;
        let (row, per) = (self.kv_heads * self.dim, self.heads / self.kv_heads);
        let topics = unit(11, self.topics * row);
        let noise = unit(14 + 100 * s as u32, positions.len() * self.heads * self.dim);
        let mut out = Vec::with_capacity(noise.len());
        for (i, &position) in positions.iter().enumerate() {
            let topic = (position * 7) % self.topics;
            for head in 0..self.heads {
                for d in 0..self.dim {
                    let lean = topics[topic * row + (head / per) * self.dim + d];
                    out.push(noise[(i * self.heads + head) * self.dim + d].mul_add(0.5, lean));
                }
            }
        }
        out
    }

    fn map(&self) -> Vec<f32> {
        unit(13, self.kv_heads * self.free() * INDEX_DIM)
    }

    /// Closed blocks of each sequence, numbered across the pass.
    fn blocks(&self) -> Vec<(usize, usize)> {
        self.sequences
            .iter()
            .enumerate()
            .flat_map(|(s, (tokens, _))| (0..tokens / INDEX_BLOCK_TOKENS).map(move |b| (s, b)))
            .collect()
    }
}

/// A case's keys stored in a shuffled paged pool, with its index addressing.
struct Built {
    pool: AttentionPool,
    /// Dequantized key and value rows `[slots, kv_heads, dim]`, what a correct kernel reads.
    exact: Vec<f32>,
    values: Vec<f32>,
    /// Slots of the K/V pool.
    pool_slots: usize,
    kv: PagedKv,
    page_table: Vec<u32>,
    /// Where each sequence's pages start in `page_table`.
    page_first: Vec<usize>,
    /// Index slot of each closed block, in `Case::blocks` order.
    slots: Vec<u32>,
    map: IndexMap,
}

fn build(case: &Case, device: &Device, kernels: &Kernels, format: KvFormat) -> Result<Built> {
    let row = case.kv_heads * case.dim;
    let pages: Vec<usize> = case
        .sequences
        .iter()
        .map(|(t, _)| t.div_ceil(PAGE))
        .collect();
    let total = pages.iter().sum::<usize>() + 2;
    // Reversed physical pages exercise the indirection; unused rows hold NaN poison.
    let physical: Vec<u32> = (0..total as u32).rev().collect();
    let slots = total * PAGE;
    let pool_slots = slots;
    let mut rows = vec![f32::NAN; slots * row];
    let mut value_rows = rows.clone();
    let (mut page_table, mut page_first) = (Vec::new(), Vec::new());
    for (s, &count) in pages.iter().enumerate() {
        let base = page_table.len();
        page_first.push(base);
        page_table.extend_from_slice(&physical[base..base + count]);
        let values = unit(15 + 100 * s as u32, case.sequences[s].0 * row);
        for (j, (key, value)) in case
            .keys(s)
            .chunks_exact(row)
            .zip(values.chunks_exact(row))
            .enumerate()
        {
            let slot = page_table[base + j / PAGE] as usize * PAGE + j % PAGE;
            rows[slot * row..(slot + 1) * row].copy_from_slice(key);
            value_rows[slot * row..(slot + 1) * row].copy_from_slice(value);
        }
    }
    let (key, exact) = stored(device, &rows, slots, case.kv_heads, case.dim, format)?;
    let (value, values) = stored(device, &value_rows, slots, case.kv_heads, case.dim, format)?;
    let pool = AttentionPool {
        key,
        value,
        format,
        page_tokens: PAGE,
        kv_heads: case.kv_heads,
        head_dim: case.dim,
    };
    let last = case.sequences[0].0 as u32;
    let kv = kernels.paged_kv(PAGE, slots, &page_table, &[0], &[last])?;
    // Index slots in reverse, leaving slots 0 and 1 unused.
    let blocks = case.blocks().len() as u32;
    let slots: Vec<u32> = (0..blocks).map(|i| blocks + 1 - i).collect();
    let rows = case.kv_heads * case.free();
    let map = Tensor::from_vec(case.map(), (rows, INDEX_DIM), device)?;
    let map = if case.split {
        IndexMap::split_rotary(map, case.kv_heads, case.dim, case.rotary)?
    } else {
        IndexMap::new(map, case.kv_heads, case.dim, case.rotary)?
    };
    Ok(Built {
        pool,
        exact,
        values,
        pool_slots,
        kv,
        page_table,
        page_first,
        slots,
        map,
    })
}

/// Cluster every closed block of the case into a fresh index pool.
fn cluster(case: &Case, built: &Built, kernels: &Kernels, device: &Device) -> Result<IndexPool> {
    let index = IndexPool::new(device, built.slots.len() + 2)?;
    let jobs: Vec<ClusterJob> = case
        .blocks()
        .iter()
        .zip(&built.slots)
        .map(|(&(s, block), &slot)| ClusterJob {
            first_block: built.page_first[s] as u32,
            block: block as u32,
            slot,
        })
        .collect();
    kernels.lsa_cluster(&built.pool, &built.kv, &built.map, &index, &jobs)?;
    Ok(index)
}

/// Index addressing and queries `[tokens, heads, dim]` of every selecting position.
fn selections(
    case: &Case,
    built: &Built,
    kernels: &Kernels,
    device: &Device,
    index: &IndexPool,
) -> Result<(IndexTable, Tensor)> {
    let inputs = selection_inputs(case, built, device)?;
    let table = kernels.index_table(
        &inputs.table,
        &inputs.first,
        &inputs.visible,
        index.blocks(),
    )?;
    Ok((table, inputs.q))
}

/// What selection reads: index table entries, each query's first entry and visible keys,
/// and the queries.
struct Inputs {
    table: Vec<u32>,
    first: Vec<u32>,
    visible: Vec<u32>,
    q: Tensor,
}

fn selection_inputs(case: &Case, built: &Built, device: &Device) -> Result<Inputs> {
    let (mut table, mut first, mut visible, mut queries) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let blocks = case.blocks();
    for (s, (_, positions)) in case.sequences.iter().enumerate() {
        let start = table.len() as u32;
        table.extend(
            blocks
                .iter()
                .zip(&built.slots)
                .filter(|((seq, _), _)| *seq == s)
                .map(|(_, &slot)| slot),
        );
        for &position in positions {
            first.push(start);
            visible.push((position + 1) as u32);
        }
        queries.extend(case.queries(s));
    }
    let q = Tensor::from_vec(queries, (visible.len(), case.heads, case.dim), device)?;
    Ok(Inputs {
        table,
        first,
        visible,
        q,
    })
}

/// Key lists, one per query or tile.
type Lists = Vec<Vec<i64>>;

/// Each row's keys up to its length.
fn rows_of(keys: &[f32], lengths: &[f32], width: usize) -> Vec<Vec<i64>> {
    lengths
        .iter()
        .enumerate()
        .map(|(r, &n)| {
            keys[r * width..r * width + n as usize]
                .iter()
                .map(|&k| k as i64)
                .collect()
        })
        .collect()
}

fn golden_budgets() -> [(&'static str, SparseBudget); 2] {
    [
        (
            "cut",
            SparseBudget {
                top_k: 200,
                top_p: 32,
                window: 512,
                sink: 4,
            },
        ),
        (
            "all",
            SparseBudget {
                top_k: 2048,
                top_p: 1024,
                window: 640,
                sink: 4,
            },
        ),
    ]
}

#[test]
fn test_lsa_index_matches_the_pinned_reference_buckets_and_key_lists() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let golden = candle_core::safetensors::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lsa/golden.safetensors"),
        &Device::Cpu,
    )?;
    let get = |name: &str| {
        golden
            .get(name)
            .ok_or_else(|| TurError::Other(format!("golden file lacks {name}")))
    };
    let case = Case::golden();
    let built = build(&case, &device, &kernels, KvFormat::F32)?;
    let index = cluster(&case, &built, &kernels, &device)?;
    let members = read(index.members())?;
    let centroids = read(index.centroids())?;
    let expected_members = get("members")?.flatten_all()?.to_vec1::<f32>()?;
    let expected_centroids = get("centroids")?.flatten_all()?.to_vec1::<f32>()?;
    let (row_m, row_c) = (INDEX_BUCKETS * BUCKET_POOLS, INDEX_BUCKETS * INDEX_DIM);
    for (block, &slot) in built.slots.iter().enumerate() {
        let slot = slot as usize;
        assert_eq!(
            members[slot * row_m..(slot + 1) * row_m],
            expected_members[block * row_m..(block + 1) * row_m],
            "buckets of block {block}"
        );
        let error = centroids[slot * row_c..(slot + 1) * row_c]
            .iter()
            .zip(&expected_centroids[block * row_c..(block + 1) * row_c])
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        // Unit vectors summed in a different order differ by F32 rounding alone.
        assert!(error < 1e-5, "centroids of block {block}: {error}");
    }
    let (table, q) = selections(&case, &built, &kernels, &device, &index)?;
    for (name, budget) in golden_budgets() {
        let selected = kernels.lsa_select(&q, &built.map, &index, &table, budget)?;
        let lengths = read(selected.lengths())?;
        let rows = rows_of(&read(selected.keys())?, &lengths, selected.width());
        let expected_lengths = get(&format!("lengths_{name}"))?.to_vec1::<i64>()?;
        let mut flat = get(&format!("keys_{name}"))?.to_vec1::<i64>()?.into_iter();
        for (r, (row, &length)) in rows.iter().zip(&expected_lengths).enumerate() {
            let expected: Vec<i64> = flat.by_ref().take(length as usize).collect();
            assert_eq!(row, &expected, "{name} budget, query {r}");
        }
    }
    Ok(())
}

/// F64 index keys of closed block `block` of sequence `s`: the position-free dims of every
/// K/V head, concatenated, carried through the map and normalized.
fn index_keys(case: &Case, built: &Built, s: usize, block: usize) -> Vec<Vec<f64>> {
    let (row, free) = (case.kv_heads * case.dim, case.free());
    let map = case.map();
    (0..INDEX_BLOCK_TOKENS)
        .map(|r| {
            let j = block * INDEX_BLOCK_TOKENS + r;
            let slot = built.page_table[built.page_first[s] + j / PAGE] as usize * PAGE + j % PAGE;
            let x: Vec<f64> = (0..case.kv_heads * free)
                .map(|d| {
                    f64::from(
                        built.exact[slot * row + (d / free) * case.dim + case.free_dim(d % free)],
                    )
                })
                .collect();
            project(&x, &map)
        })
        .collect()
}

/// `normalize(map^T x)`.
fn project(x: &[f64], map: &[f32]) -> Vec<f64> {
    let mut y = vec![0.0f64; INDEX_DIM];
    for (d, &xd) in x.iter().enumerate() {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj = xd.mul_add(f64::from(map[d * INDEX_DIM + j]), *yj);
        }
    }
    let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    y.iter().map(|v| v / norm).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Each alive bucket's cheapest allowed partner and its cost, ties to the lower index.
fn partners(sums: &[Vec<f64>], sizes: &[usize], alive: &[bool]) -> Vec<Option<(usize, f64)>> {
    let squares: Vec<f64> = sums.iter().map(|s| dot(s, s)).collect();
    (0..POOLS)
        .map(|i| {
            let mut best: Option<(usize, f64)> = None;
            for j in (0..POOLS)
                .filter(|&j| alive[i] && alive[j] && j != i && sizes[i] + sizes[j] <= BUCKET_CAP)
            {
                let merged = 2.0f64
                    .mul_add(dot(&sums[i], &sums[j]), squares[i] + squares[j])
                    .max(0.0)
                    .sqrt();
                let cost = squares[i].max(0.0).sqrt() + squares[j].max(0.0).sqrt() - merged;
                if best.is_none_or(|(_, c)| cost < c) {
                    best = Some((j, cost));
                }
            }
            best
        })
        .collect()
}

/// Ward rounds over a block's pools as `lsa.core.ward.ward_rounds`: returns each pool's
/// bucket (numbered by first pool) and each bucket's key sum.
fn ward(pools: Vec<Vec<f64>>) -> (Vec<usize>, Vec<Vec<f64>>) {
    let mut sums = pools;
    let mut sizes = vec![POOL; POOLS];
    let mut alive = vec![true; POOLS];
    let mut parent: Vec<usize> = (0..POOLS).collect();
    loop {
        let count = alive.iter().filter(|&&a| a).count();
        if count <= INDEX_BUCKETS {
            break;
        }
        let best = partners(&sums, &sizes, &alive);
        let mut mutual: Vec<(f64, usize, usize)> = (0..POOLS)
            .filter_map(|i| {
                let (j, cost) = best[i]?;
                (j > i && best[j].map(|(k, _)| k) == Some(i)).then_some((cost, i, j))
            })
            .collect();
        mutual.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        mutual.truncate(count - INDEX_BUCKETS);
        if mutual.is_empty() {
            break;
        }
        for (_, i, j) in mutual {
            let absorbed = std::mem::take(&mut sums[j]);
            sums[i].iter_mut().zip(absorbed).for_each(|(a, b)| *a += b);
            sizes[i] += sizes[j];
            alive[j] = false;
            parent[j] = i;
        }
    }
    let root = |mut i: usize| {
        while parent[i] != i {
            i = parent[i];
        }
        i
    };
    let label = |r: usize| alive[..r].iter().filter(|&&a| a).count();
    let labels = (0..POOLS).map(|i| label(root(i))).collect();
    let buckets = (0..POOLS)
        .filter(|&i| alive[i])
        .map(|i| sums[i].clone())
        .collect();
    (labels, buckets)
}

/// One block's reference buckets: pools per bucket (-1 padded) and unit centroids.
type Block = (Vec<f32>, Vec<Vec<f64>>);

fn reference_block(keys: &[Vec<f64>]) -> Block {
    let pools = keys
        .as_chunks::<POOL>()
        .0
        .iter()
        .map(|run| {
            (0..INDEX_DIM)
                .map(|d| run.iter().map(|k| k[d]).sum())
                .collect()
        })
        .collect();
    let (labels, sums) = ward(pools);
    let mut members = vec![-1.0f32; INDEX_BUCKETS * BUCKET_POOLS];
    let mut filled = vec![0usize; INDEX_BUCKETS];
    for (pool, &label) in labels.iter().enumerate() {
        members[label * BUCKET_POOLS + filled[label]] = pool as f32;
        filled[label] += 1;
    }
    let centroids = sums
        .iter()
        .map(|s| {
            let norm = dot(s, s).sqrt().max(1e-30);
            s.iter().map(|v| v / norm).collect()
        })
        .collect();
    (members, centroids)
}

/// Reference key list of one query, as `lsa.core.reference.read`.
fn reference_read(
    query: &[f64],
    position: usize,
    blocks: &[Block],
    budget: SparseBudget,
) -> Vec<i64> {
    let window_start = (position + 1).saturating_sub(budget.window);
    let mut keys = reference_expanded(query, position, blocks, budget);
    keys.extend((window_start..=position).map(|k| k as i64));
    keys.extend((0..budget.sink.min(window_start).min(position + 1)).map(|k| k as i64));
    keys
}

/// Reference key list of a query tile whose first row is at `first` and last at `last`,
/// scored by `query`: expanded buckets, sinks, then the window run to the last row.
fn reference_tile_read(
    query: &[f64],
    (first, last): (usize, usize),
    blocks: &[Block],
    budget: SparseBudget,
) -> Vec<i64> {
    let window_start = (first + 1).saturating_sub(budget.window);
    let mut keys = reference_expanded(query, first, blocks, budget);
    keys.extend((0..budget.sink.min(window_start).min(first + 1)).map(|k| k as i64));
    keys.extend((window_start..=last).map(|k| k as i64));
    keys
}

/// Keys of the best buckets before `position`'s block, outside its sinks and window.
fn reference_expanded(
    query: &[f64],
    position: usize,
    blocks: &[Block],
    budget: SparseBudget,
) -> Vec<i64> {
    let visible = (position / INDEX_BLOCK_TOKENS) * INDEX_BUCKETS;
    let mut order: Vec<(f64, usize)> = (0..visible)
        .map(|b| {
            (
                dot(query, &blocks[b / INDEX_BUCKETS].1[b % INDEX_BUCKETS]),
                b,
            )
        })
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let window_start = (position + 1).saturating_sub(budget.window);
    let mut keys: Vec<i64> = Vec::new();
    for &(_, bucket) in order.iter().take(budget.top_p) {
        let (block, local) = (bucket / INDEX_BUCKETS, bucket % INDEX_BUCKETS);
        let pools = &blocks[block].0[local * BUCKET_POOLS..(local + 1) * BUCKET_POOLS];
        for &pool in pools.iter().take_while(|&&p| p >= 0.0) {
            let start = block * INDEX_BLOCK_TOKENS + pool as usize * POOL;
            keys.extend(
                (start..start + POOL)
                    .filter(|&k| k >= budget.sink && k < window_start)
                    .map(|k| k as i64),
            );
        }
    }
    keys.truncate(budget.top_k);
    keys
}

/// F64 index query: query heads summed per K/V group over the position-free dims.
fn index_query(case: &Case, q: &[f32]) -> Vec<f64> {
    let (free, per) = (case.free(), case.heads / case.kv_heads);
    let x: Vec<f64> = (0..case.kv_heads * free)
        .map(|d| {
            let (h, e) = (d / free, case.free_dim(d % free));
            (0..per)
                .map(|g| f64::from(q[(h * per + g) * case.dim + e]))
                .sum()
        })
        .collect();
    project(&x, &case.map())
}

fn check_against_scalar(case: &Case, format: KvFormat) -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let built = build(case, &device, &kernels, format)?;
    let index = cluster(case, &built, &kernels, &device)?;
    let (members, centroids) = (read(index.members())?, read(index.centroids())?);
    let (row_m, row_c) = (INDEX_BUCKETS * BUCKET_POOLS, INDEX_BUCKETS * INDEX_DIM);
    let mut per_sequence: Vec<Vec<Block>> = vec![Vec::new(); case.sequences.len()];
    for (&(s, block), &slot) in case.blocks().iter().zip(&built.slots) {
        let (expected_members, expected_centroids) =
            reference_block(&index_keys(case, &built, s, block));
        let slot = slot as usize;
        assert_eq!(
            members[slot * row_m..(slot + 1) * row_m],
            expected_members[..],
            "{format:?} sequence {s} block {block} buckets"
        );
        let error = centroids[slot * row_c..(slot + 1) * row_c]
            .iter()
            .zip(expected_centroids.iter().flatten())
            .map(|(a, e)| (f64::from(*a) - e).abs())
            .fold(0.0, f64::max);
        assert!(
            error < 1e-5,
            "{format:?} sequence {s} block {block} centroids: {error}"
        );
        per_sequence[s].push((expected_members, expected_centroids));
    }
    let (table, q) = selections(case, &built, &kernels, &device, &index)?;
    let queries = read(&q)?;
    let budget = SparseBudget {
        top_k: 300,
        top_p: 48,
        window: 600,
        sink: 4,
    };
    let selected = kernels.lsa_select(&q, &built.map, &index, &table, budget)?;
    let rows = rows_of(
        &read(selected.keys())?,
        &read(selected.lengths())?,
        selected.width(),
    );
    let row = case.heads * case.dim;
    let mut r = 0;
    for (s, (_, positions)) in case.sequences.iter().enumerate() {
        for &position in positions {
            let query = index_query(case, &queries[r * row..(r + 1) * row]);
            let expected = reference_read(&query, position, &per_sequence[s], budget);
            assert_eq!(
                rows[r], expected,
                "{format:?} sequence {s} position {position}"
            );
            r += 1;
        }
    }
    Ok(())
}

#[test]
fn test_lsa_index_matches_scalar_reference_for_every_format_sequence_and_rotation() -> Result<()> {
    let wide = Case {
        kv_heads: 4,
        heads: 8,
        dim: 256,
        rotary: 64,
        split: false,
        segment: 37,
        topics: 5,
        sequences: vec![(1100, vec![1099, 700]), (2100, vec![2099, 1536, 1600])],
    };
    let narrow = Case {
        kv_heads: 1,
        heads: 2,
        dim: 128,
        rotary: 32,
        split: false,
        segment: 19,
        topics: 7,
        sequences: vec![(1600, vec![1599, 1024])],
    };
    // Gemma 4's full attention heads: 512 dims, 64 rotated pairs, two runs of 192 free.
    let split = Case {
        kv_heads: 1,
        heads: 4,
        dim: 512,
        rotary: 64,
        split: true,
        segment: 31,
        topics: 5,
        sequences: vec![(1100, vec![1099, 1030])],
    };
    for format in [KvFormat::F32, KvFormat::F16, KvFormat::Q8 { group: 32 }] {
        check_against_scalar(&wide, format)?;
    }
    check_against_scalar(&narrow, KvFormat::F16)?;
    check_against_scalar(&split, KvFormat::F16)
}

/// F64 gated attention of each selecting token over exactly the keys of its list.
fn sparse_reference(
    case: &Case,
    built: &Built,
    queries: &[f32],
    gate: &[f32],
    rows: &[Vec<i64>],
) -> Vec<f64> {
    let (heads, dim, per) = (case.heads, case.dim, case.heads / case.kv_heads);
    let row = case.kv_heads * dim;
    let owners: Vec<usize> = case
        .sequences
        .iter()
        .enumerate()
        .flat_map(|(s, (_, positions))| std::iter::repeat_n(s, positions.len()))
        .collect();
    let mut out = Vec::with_capacity(rows.len() * heads * dim);
    for (t, keys) in rows.iter().enumerate() {
        let slots: Vec<usize> = keys
            .iter()
            .map(|&j| {
                let j = j as usize;
                built.page_table[built.page_first[owners[t]] + j / PAGE] as usize * PAGE + j % PAGE
            })
            .collect();
        for h in 0..heads {
            let g = h / per;
            let qh = &queries[(t * heads + h) * dim..][..dim];
            let scores: Vec<f64> = slots
                .iter()
                .map(|s| {
                    let k = &built.exact[s * row + g * dim..][..dim];
                    qh.iter()
                        .zip(k)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for i in 0..dim {
                let value: f64 = slots
                    .iter()
                    .zip(&weights)
                    .map(|(s, w)| w * f64::from(built.values[s * row + g * dim + i]))
                    .sum();
                out.push(
                    value / total * sigmoid(f64::from(gate[(t * heads + h) * 2 * dim + dim + i])),
                );
            }
        }
    }
    out
}

#[test]
fn test_sparse_attention_matches_scalar_reference_over_selected_keys() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let split = kernels.clone().with_tuning(Tuning {
        min_split_keys: 8,
        ..kernels.tuning()
    })?;
    let whole = kernels.clone().with_tuning(Tuning {
        attention_parallelism: 1,
        ..kernels.tuning()
    })?;
    let case = Case {
        kv_heads: 2,
        heads: 8,
        dim: 256,
        rotary: 64,
        split: false,
        segment: 29,
        topics: 6,
        sequences: vec![(1300, vec![1299, 600]), (2100, vec![2099, 1536])],
    };
    let budget = SparseBudget {
        top_k: 300,
        top_p: 48,
        window: 512,
        sink: 4,
    };
    for format in [KvFormat::F32, KvFormat::F16, KvFormat::Q8 { group: 32 }] {
        let built = build(&case, &device, &kernels, format)?;
        let index = cluster(&case, &built, &kernels, &device)?;
        let (table, q) = selections(&case, &built, &kernels, &device, &index)?;
        let selected = kernels.lsa_select(&q, &built.map, &index, &table, budget)?;
        let rows = rows_of(
            &read(selected.keys())?,
            &read(selected.lengths())?,
            selected.width(),
        );
        let (mut first, mut visible) = (Vec::new(), Vec::new());
        for (s, (_, positions)) in case.sequences.iter().enumerate() {
            for &position in positions {
                first.push(built.page_first[s] as u32);
                visible.push(position as u32 + 1);
            }
        }
        let kv = kernels.paged_kv(PAGE, built.pool_slots, &built.page_table, &first, &visible)?;
        let tokens = visible.len();
        let gate = unit(16, tokens * case.heads * 2 * case.dim);
        let gate_tensor =
            Tensor::from_vec(gate.clone(), (tokens, case.heads * 2 * case.dim), &device)?;
        let heads = tur_metal::kernels::Heads {
            count: case.heads,
            dim: case.dim,
            head_stride: 2 * case.dim,
            col_offset: case.dim,
        };
        let expected = sparse_reference(&case, &built, &read(&q)?, &gate, &rows);
        for (label, tuned) in [("split", &split), ("whole", &whole), ("default", &kernels)] {
            let out =
                tuned.attention_sparse(&q, &built.pool, &kv, &gate_tensor, heads, &selected)?;
            let error = relative_error(&read(&out)?, &expected);
            assert!(error < 1e-5, "{format:?} {label}: {error}");
        }
    }
    Ok(())
}

/// Expected lists of each query tile of `kv` and, per query row, the prefix of its tile's
/// list that row sees: everything but the run past its own position.
fn tile_references(
    case: &Case,
    built: &Built,
    index: &IndexPool,
    queries: &[f32],
    kv: &PagedKv,
    budget: SparseBudget,
) -> Result<(Lists, Lists)> {
    let per_sequence = reference_blocks(case, built, index)?;
    let owners: Vec<(usize, usize)> = case
        .sequences
        .iter()
        .enumerate()
        .flat_map(|(s, (_, positions))| positions.iter().map(move |&p| (s, p)))
        .collect();
    let row = case.heads * case.dim;
    let (mut tiles, mut rows) = (Vec::new(), Vec::new());
    for tile in kv
        .tile_rows()
        .ok_or_else(|| TurError::Other("no tiles".into()))?
        .as_chunks::<2>()
        .0
    {
        let (t0, n) = (tile[0] as usize, tile[1] as usize);
        let (s, first) = owners[t0];
        let last = owners[t0 + n - 1].1;
        let mut query = vec![0.0f64; INDEX_DIM];
        for t in t0..t0 + n {
            let y = index_query(case, &queries[t * row..(t + 1) * row]);
            query.iter_mut().zip(y).for_each(|(a, b)| *a += b);
        }
        let list = reference_tile_read(&query, (first, last), &per_sequence[s], budget);
        for t in t0..t0 + n {
            rows.push(list[..list.len() - (last - owners[t].1)].to_vec());
        }
        tiles.push(list);
    }
    Ok((tiles, rows))
}

/// Reference buckets of every closed block, by sequence, from the case's stored keys.
fn reference_blocks(case: &Case, built: &Built, index: &IndexPool) -> Result<Vec<Vec<Block>>> {
    let mut per_sequence: Vec<Vec<Block>> = vec![Vec::new(); case.sequences.len()];
    let members = read(index.members())?;
    let row_m = INDEX_BUCKETS * BUCKET_POOLS;
    for (&(s, block), &slot) in case.blocks().iter().zip(&built.slots) {
        let (expected, centroids) = reference_block(&index_keys(case, built, s, block));
        let slot = slot as usize;
        assert_eq!(members[slot * row_m..(slot + 1) * row_m], expected[..]);
        per_sequence[s].push((expected, centroids));
    }
    Ok(per_sequence)
}

#[test]
fn test_tiled_sparse_attention_reads_each_tiles_shared_list() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    // Runs of consecutive prefill rows: 36 rows make a full and a partial tile, and the
    // second run crosses a block boundary inside its tile. F32 pools stage as F16 do.
    let case = Case {
        kv_heads: 2,
        heads: 8,
        dim: 256,
        rotary: 64,
        split: false,
        segment: 29,
        topics: 6,
        sequences: vec![
            (2100, (2064..2100).collect()),
            (1100, (1020..1030).collect()),
        ],
    };
    let budget = SparseBudget {
        top_k: 300,
        top_p: 48,
        window: 512,
        sink: 4,
    };
    for format in [KvFormat::F16, KvFormat::Q8 { group: 32 }] {
        let built = build(&case, &device, &kernels, format)?;
        let index = cluster(&case, &built, &kernels, &device)?;
        let Inputs {
            table: entries,
            first: index_first,
            visible,
            q,
        } = selection_inputs(&case, &built, &device)?;
        let page_first: Vec<u32> = case
            .sequences
            .iter()
            .enumerate()
            .flat_map(|(s, (_, positions))| {
                std::iter::repeat_n(built.page_first[s] as u32, positions.len())
            })
            .collect();
        let kv = kernels.paged_kv(
            PAGE,
            built.pool_slots,
            &built.page_table,
            &page_first,
            &visible,
        )?;
        let table =
            kernels.index_table_tiles(&entries, &index_first, &visible, &kv, index.blocks())?;
        let selected = kernels.lsa_select(&q, &built.map, &index, &table, budget)?;
        assert!(selected.tiled());
        let queries = read(&q)?;
        let (tiles, rows) = tile_references(&case, &built, &index, &queries, &kv, budget)?;
        let lists = rows_of(
            &read(selected.keys())?,
            &read(selected.lengths())?,
            selected.width(),
        );
        assert_eq!(lists, tiles, "{format:?} tile lists");
        let tokens = visible.len();
        let gate = unit(16, tokens * case.heads * 2 * case.dim);
        let gate_tensor =
            Tensor::from_vec(gate.clone(), (tokens, case.heads * 2 * case.dim), &device)?;
        let heads = tur_metal::kernels::Heads {
            count: case.heads,
            dim: case.dim,
            head_stride: 2 * case.dim,
            col_offset: case.dim,
        };
        let out = kernels.attention_sparse(&q, &built.pool, &kv, &gate_tensor, heads, &selected)?;
        let expected = sparse_reference(&case, &built, &queries, &gate, &rows);
        let error = relative_error(&read(&out)?, &expected);
        assert!(error < 1e-5, "{format:?}: {error}");
    }
    Ok(())
}

#[test]
fn test_lsa_rejects_invalid_maps_jobs_tables_and_budgets() -> Result<()> {
    let (device, kernels, _gpu) = setup()?;
    let rejected = |result: Result<()>| matches!(result, Err(TurError::InvalidExecution(_)));
    let map = |rows: usize| Tensor::zeros((rows, INDEX_DIM), DType::F32, &device);
    // 192 position-free dims per head: 384 rows for two heads, not 383, and rotary a multiple of 32.
    assert!(rejected(IndexMap::new(map(383)?, 2, 256, 64).map(drop)));
    assert!(rejected(IndexMap::new(map(400)?, 2, 256, 56).map(drop)));
    // Split rotation: 512-dim heads with 64 rotated pairs keep 384 free dims each; 40 pairs
    // would keep 432, not a multiple of 32, and 256 pairs none.
    assert!(IndexMap::split_rotary(map(768)?, 2, 512, 64).is_ok());
    assert!(rejected(
        IndexMap::split_rotary(map(864)?, 2, 512, 40).map(drop)
    ));
    assert!(rejected(
        IndexMap::split_rotary(map(64)?, 2, 512, 256).map(drop)
    ));
    let case = Case::golden();
    let built = build(&case, &device, &kernels, KvFormat::F32)?;
    let index = IndexPool::new(&device, 4)?;
    let job = |first_block, block, slot| ClusterJob {
        first_block,
        block,
        slot,
    };
    // The sequence's 107 pages hold three closed blocks of 32 pages; a fourth would need 128.
    assert!(rejected(kernels.lsa_cluster(
        &built.pool,
        &built.kv,
        &built.map,
        &index,
        &[job(0, 3, 0)]
    )));
    assert!(rejected(kernels.lsa_cluster(
        &built.pool,
        &built.kv,
        &built.map,
        &index,
        &[job(0, 0, 4)]
    )));
    assert!(rejected(kernels.lsa_cluster(
        &built.pool,
        &built.kv,
        &built.map,
        &index,
        &[job(0, 0, 1), job(0, 1, 1)]
    )));
    // A token at position 1100 sees two closed blocks; the table gives it one.
    assert!(rejected(
        kernels.index_table(&[0], &[0], &[1101], 4).map(drop)
    ));
    assert!(rejected(
        kernels.index_table(&[4], &[0], &[600], 4).map(drop)
    ));
    let table = kernels.index_table(&[0], &[0], &[600], 4)?;
    let q = Tensor::zeros((1, case.heads, case.dim), DType::F32, &device)?;
    let budget = SparseBudget {
        top_k: 64,
        top_p: 1025,
        window: 512,
        sink: 4,
    };
    assert!(rejected(
        kernels
            .lsa_select(&q, &built.map, &index, &table, budget)
            .map(drop)
    ));
    let short = SparseBudget {
        top_p: 8,
        window: 511,
        ..budget
    };
    assert!(rejected(
        kernels
            .lsa_select(&q, &built.map, &index, &table, short)
            .map(drop)
    ));
    let small = IndexPool::new(&device, 2)?;
    let valid = SparseBudget { top_p: 8, ..short };
    let valid = SparseBudget {
        window: 512,
        ..valid
    };
    assert!(rejected(
        kernels
            .lsa_select(&q, &built.map, &small, &table, valid)
            .map(drop)
    ));
    Ok(())
}
