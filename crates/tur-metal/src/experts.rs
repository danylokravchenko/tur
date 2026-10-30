#![forbid(unsafe_code)]
//! Bounded SSD-backed residency for MXFP4 expert banks.
//!
//! Experts live in a fixed arena of slots; a routed expert absent from it is read from the
//! checkpoint into a slot chosen by the policy in [`ExpertCache::route`]. Nothing is ever written
//! to disk: files open read-only, evicted slots are clean and simply overwritten, and reads
//! bypass the page cache so the arena really bounds what experts occupy in memory.
//!
//! Slot lifetime needs no tracking. Reading a layer's routes waits for all earlier GPU work,
//! so no kernel can be using a slot while it is overwritten, and the copies complete before
//! the layer's own kernels are encoded. Candle's buffers are untracked, so this ordering is
//! the only thing that makes the overwrite safe.

use crate::{
    kernels::{ExpertRoutes, Kernels},
    packed::PackedLinear,
    quantize::{MXFP4_BLOCK_BYTES, MXFP4_BLOCK_VALUES},
};
use candle_core::{DType, Device, MetalDevice, Tensor};
use candle_metal_kernels::metal::Buffer;
use parking_lot::Mutex;
use std::{
    collections::BTreeMap,
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tur_core::{
    Result,
    execution::{checked_elements, invalid},
};

/// Experts read and copied between waits, which bounds host staging.
const READ_BATCH: usize = 8;
/// Concurrent reads of one batch.
const READ_THREADS: usize = 4;
/// Memory left for the state pool, scratch, staging and everything else on the machine
/// beyond the arena. Measured on a 16 GB M1: with less than this to spare, 4 GiB arenas
/// swapped out 700 to 830 MB in a short run and a 6 GiB arena 2.4 GB, while 2 GiB with
/// about 6 GiB to spare swapped nothing.
const RESERVE_BYTES: usize = 3584 << 20;
/// Arenas below this cannot push other memory into swap, and skip the spare-memory check.
const NEGLIGIBLE_BYTES: usize = 256 << 20;
/// Marks an expert that holds no slot.
const ABSENT: u32 = u32::MAX;

/// Rows of one expert and input columns of one bank; columns are whole MXFP4 blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankShape {
    pub rows: usize,
    pub cols: usize,
}

impl BankShape {
    fn blocks(self) -> Result<usize> {
        if self.rows == 0 || self.cols == 0 || !self.cols.is_multiple_of(MXFP4_BLOCK_VALUES) {
            return Err(invalid("expert banks need rows and whole MXFP4 blocks"));
        }
        checked_elements(&[self.rows, self.cols / MXFP4_BLOCK_VALUES])
    }

    /// Scale bytes of one expert, which is also its block count.
    fn scale_bytes(self) -> Result<usize> {
        self.blocks()
    }

    fn block_bytes(self) -> Result<usize> {
        checked_elements(&[self.blocks()?, MXFP4_BLOCK_BYTES])
    }

    fn expert_bytes(self) -> Result<usize> {
        self.block_bytes()?
            .checked_add(self.scale_bytes()?)
            .ok_or_else(|| invalid("expert size overflow"))
    }
}

/// Where one layer's bank is stored: byte offsets of expert 0's blocks and scales.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankSource {
    pub file: PathBuf,
    pub blocks: u64,
    pub scales: u64,
}

/// Every layer's expert storage in a checkpoint.
#[derive(Clone, Debug)]
pub struct ExpertSources {
    pub experts: usize,
    /// Experts one token selects in each layer.
    pub top_k: usize,
    pub gate_up: BankShape,
    pub down: BankShape,
    /// `[gate_up, down]` of each layer.
    pub layers: Vec<[BankSource; 2]>,
}

/// What the cache has done since it was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertStats {
    /// Routed experts of a layer found in a slot.
    pub hits: u64,
    /// Routed experts read from the checkpoint.
    pub misses: u64,
    /// Bytes read from the checkpoint. Everything else about the cache is memory.
    pub bytes_read: u64,
    /// Misses that overwrote another expert.
    pub evictions: u64,
    /// Time the execution owner spent waiting for reads and copies.
    pub wait: Duration,
}

struct Slots {
    /// Slot of each `layer * experts + expert`.
    map: Vec<u32>,
    /// Owner key and last use of each slot.
    owners: Vec<Option<usize>>,
    used: Vec<u64>,
    tick: u64,
    stats: ExpertStats,
}

/// One bank's arena: blocks of every slot, then scales of every slot, as the kernels read.
struct Arena {
    weights: PackedLinear,
    buffer: Arc<Buffer>,
    scales: usize,
    shape: BankShape,
}

impl Arena {
    fn new(metal: &MetalDevice, shape: BankShape, slots: usize) -> Result<Self> {
        let blocks = checked_elements(&[slots, shape.block_bytes()?])?;
        let total = checked_elements(&[slots, shape.expert_bytes()?])?;
        // Outside Candle's pool, which rounds every size up to a power of two: a 4 GiB
        // arena plus one byte would otherwise cost 8 GiB.
        let buffer = metal.new_private_buffer(total, DType::U8, "tur_expert_arena")?;
        let rows = checked_elements(&[slots, shape.rows])?;
        Ok(Self {
            weights: PackedLinear::mxfp4_arena(Arc::clone(&buffer), blocks, (rows, shape.cols))?,
            buffer,
            scales: blocks,
            shape,
        })
    }
}

/// A fixed set of expert slots filled on demand. Owned by one model; its lock is only
/// taken by the execution owner between kernels and never across an await.
pub struct ExpertCache {
    device: Device,
    metal: MetalDevice,
    files: Vec<File>,
    /// File index and stored offsets of each layer's banks.
    sources: Vec<[(usize, u64, u64); 2]>,
    experts: usize,
    /// Fewer slots than one token touches: LRU would evict each just before its next use.
    cyclic: bool,
    gate_up: Arena,
    down: Arena,
    slots: usize,
    state: Mutex<Slots>,
}

/// Bytes macOS can hand out without swapping, from `vm_stat`'s report: free and purgeable
/// pages, and file-backed ones among those it would reclaim first. Inactive anonymous pages
/// do not count: reclaiming them is exactly the swap this check exists to prevent.
fn reclaimable_bytes(report: &str) -> Option<usize> {
    let page: usize = report
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let count = |key: &str| -> Option<usize> {
        let line = report
            .lines()
            .find(|line| line.trim_start().starts_with(key))?;
        line.rsplit(' ').next()?.trim_end_matches('.').parse().ok()
    };
    let reclaimable = count("Pages inactive:")?.checked_add(count("Pages speculative:")?)?;
    count("Pages free:")?
        .checked_add(count("Pages purgeable:")?)?
        .checked_add(reclaimable.min(count("File-backed pages:")?))?
        .checked_mul(page)
}

/// Refuse an arena that would push other memory into swap, the only way this cache could
/// cost SSD writes. Called with the model's other weights already resident.
fn check_spare_memory(arena: usize) -> Result<()> {
    if arena < NEGLIGIBLE_BYTES {
        return Ok(());
    }
    let output = std::process::Command::new("vm_stat").output()?;
    let report = String::from_utf8_lossy(&output.stdout);
    let Some(spare) = reclaimable_bytes(&report) else {
        return Err(invalid("cannot read spare memory from vm_stat"));
    };
    if arena.saturating_add(RESERVE_BYTES) > spare {
        return Err(tur_core::TurError::MemoryAllocation(format!(
            "an expert cache of {} MiB would swap: macOS can spare {} MiB and {} MiB stays \
             in reserve. Use a smaller --expert-cache-mb or close other applications",
            arena >> 20,
            spare >> 20,
            RESERVE_BYTES >> 20,
        )));
    }
    Ok(())
}

/// Open read-only and ask the kernel not to cache: the arena is the cache.
fn open(path: &Path) -> Result<File> {
    let file = File::open(path)?;
    rustix::fs::fcntl_nocache(&file, true).map_err(|e| {
        invalid(format!(
            "{}: cannot bypass the page cache: {e}",
            path.display()
        ))
    })?;
    Ok(file)
}

impl ExpertCache {
    /// Bytes one expert occupies across both banks.
    pub fn expert_bytes(gate_up: BankShape, down: BankShape) -> Result<usize> {
        gate_up
            .expert_bytes()?
            .checked_add(down.expert_bytes()?)
            .ok_or_else(|| invalid("expert size overflow"))
    }

    /// Slots that fit `budget` bytes, which must hold every expert of one layer: one pass
    /// may route to all of them, and they stay pinned while that layer's kernels run.
    pub fn planned_slots(
        budget: usize,
        experts: usize,
        gate_up: BankShape,
        down: BankShape,
    ) -> Result<usize> {
        let expert = Self::expert_bytes(gate_up, down)?;
        let slots = budget / expert;
        if slots < experts {
            let floor = checked_elements(&[experts, expert])?;
            return Err(invalid(format!(
                "an expert cache of {budget} bytes holds {slots} experts; one layer can route \
                 to all {experts}, which needs {floor} bytes"
            )));
        }
        Ok(slots)
    }

    /// Host and upload memory the largest read batch can hold beside the arena.
    pub fn staging_bytes(gate_up: BankShape, down: BankShape) -> Result<usize> {
        checked_elements(&[2, READ_BATCH, Self::expert_bytes(gate_up, down)?])
    }

    /// Allocate the arena and open every source file.
    pub fn new(sources: &ExpertSources, budget: usize, device: &Device) -> Result<Self> {
        let Device::Metal(metal) = device else {
            return Err(invalid("the expert cache runs on a Metal device"));
        };
        let (gate_up, down) = (sources.gate_up, sources.down);
        let slots = Self::planned_slots(budget, sources.experts, gate_up, down)?;
        check_spare_memory(checked_elements(&[
            slots,
            Self::expert_bytes(gate_up, down)?,
        ])?)?;
        if u32::try_from(slots).is_err() {
            return Err(invalid("too many expert slots"));
        }
        let keys = checked_elements(&[sources.layers.len(), sources.experts])?;
        let mut paths: BTreeMap<&Path, usize> = BTreeMap::new();
        let mut files = Vec::new();
        let mut layers = Vec::with_capacity(sources.layers.len());
        for banks in &sources.layers {
            let mut located = [(0, 0, 0); 2];
            for (entry, bank) in located.iter_mut().zip(banks) {
                let index = if let Some(index) = paths.get(bank.file.as_path()) {
                    *index
                } else {
                    files.push(open(&bank.file)?);
                    paths.insert(&bank.file, files.len() - 1);
                    files.len() - 1
                };
                *entry = (index, bank.blocks, bank.scales);
            }
            layers.push(located);
        }
        Ok(Self {
            device: device.clone(),
            metal: metal.clone(),
            files,
            sources: layers,
            experts: sources.experts,
            cyclic: slots < checked_elements(&[sources.layers.len(), sources.top_k])?,
            gate_up: Arena::new(metal, gate_up, slots)?,
            down: Arena::new(metal, down, slots)?,
            slots,
            state: Mutex::new(Slots {
                map: vec![ABSENT; keys],
                owners: vec![None; slots],
                used: vec![0; slots],
                tick: 0,
                stats: ExpertStats::default(),
            }),
        })
    }

    /// The gate|up arena, addressed by the slots of routes from [`Self::route`].
    pub fn gate_up(&self) -> &PackedLinear {
        &self.gate_up.weights
    }

    /// The down arena, addressed by the slots of routes from [`Self::route`].
    pub fn down(&self) -> &PackedLinear {
        &self.down.weights
    }

    /// Device bytes the arenas hold.
    pub fn resident_bytes(&self) -> usize {
        self.gate_up.weights.resident_bytes() + self.down.weights.resident_bytes()
    }

    /// Experts the arena holds at once.
    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn stats(&self) -> ExpertStats {
        self.state.lock().stats
    }

    /// Make every expert `routes` selects for `layer` resident and address it by slot.
    /// Waits for all earlier GPU work, and returns only once the copies have completed. On
    /// an I/O failure the affected slots are left ownerless, so a later pass reads them again.
    pub fn route(&self, k: &Kernels, layer: usize, routes: ExpertRoutes) -> Result<ExpertRoutes> {
        if layer >= self.sources.len() {
            return Err(invalid("expert cache has no such layer"));
        }
        // The readback is the wait the slot lifetime rule relies on.
        let mut tiles = routes.tile_values()?;
        let started = Instant::now();
        let mut state = self.state.lock();
        state.tick += 1;
        let tick = state.tick;
        let base = layer * self.experts;
        let mut misses = Vec::new();
        let mut seen = vec![false; self.experts];
        for tile in tiles.as_chunks::<4>().0 {
            if tile[0] < 0.0 {
                continue;
            }
            let expert = tile[0] as usize;
            let mark = seen
                .get_mut(expert)
                .ok_or_else(|| invalid("routed expert outside the bank"))?;
            if std::mem::replace(mark, true) {
                continue;
            }
            match state.map[base + expert] {
                ABSENT => misses.push(expert),
                slot => {
                    state.used[slot as usize] = tick;
                    state.stats.hits += 1;
                }
            }
        }
        let wrote = !misses.is_empty();
        for batch in misses.chunks(READ_BATCH) {
            self.fill(&mut state, layer, batch)?;
        }
        for tile in tiles.as_chunks_mut::<4>().0 {
            if tile[0] >= 0.0 {
                tile[0] = state.map[base + tile[0] as usize] as f32;
            }
        }
        if wrote {
            state.stats.wait += started.elapsed();
        }
        drop(state);
        k.with_slots(routes, tiles, self.slots)
    }

    /// Read `experts` of `layer` into free or evicted slots and wait for the
    /// copies. A slot loses its owner before it is touched, so a failure cannot leave one
    /// claiming contents it does not hold.
    fn fill(&self, state: &mut Slots, layer: usize, experts: &[usize]) -> Result<()> {
        let tick = state.tick;
        let base = layer * self.experts;
        let layers = self.sources.len();
        let mut chosen = Vec::with_capacity(experts.len());
        for &expert in experts {
            // Slots stamped this tick belong to this layer's own routes: the budget holds a
            // whole layer, so a victim always exists. Passes visit layers in order, so when a
            // token's experts exceed the slots, LRU hits nothing (0% against 34% at 81 slots
            // of 96) and the victim is what runs furthest ahead. With more slots LRU keeps
            // what routing repeats (79% against 65% at 324 slots).
            let slot = (0..self.slots)
                .filter(|&s| state.used[s] != tick)
                .max_by_key(|&s| match state.owners[s] {
                    None => (usize::MAX, 0),
                    Some(owner) if self.cyclic => {
                        let owner_layer = owner / self.experts;
                        (
                            (owner_layer + layers - layer - 1) % layers,
                            u64::MAX - state.used[s],
                        )
                    }
                    Some(_) => (0, u64::MAX - state.used[s]),
                })
                .ok_or_else(|| invalid("every expert slot is pinned"))?;
            if let Some(owner) = state.owners[slot].take() {
                state.map[owner] = ABSENT;
                state.stats.evictions += 1;
            }
            state.used[slot] = tick;
            chosen.push((expert, slot));
        }
        let [gate_up, down] = self.sources[layer];
        let mut reads = Vec::with_capacity(4 * chosen.len());
        for &(expert, slot) in &chosen {
            for (arena, (file, blocks, scales)) in [(&self.gate_up, gate_up), (&self.down, down)] {
                let block_bytes = arena.shape.block_bytes()?;
                let scale_bytes = arena.shape.scale_bytes()?;
                reads.push(Read::new(
                    file,
                    blocks,
                    expert,
                    block_bytes,
                    &arena.buffer,
                    slot * block_bytes,
                )?);
                reads.push(Read::new(
                    file,
                    scales,
                    expert,
                    scale_bytes,
                    &arena.buffer,
                    arena.scales + slot * scale_bytes,
                )?);
            }
        }
        self.read_all(&mut reads)?;
        // Keep every staging buffer alive until its copy has completed.
        let mut staging = Vec::with_capacity(reads.len());
        for read in &reads {
            staging.push(self.metal.new_buffer_with_data(&read.bytes)?);
        }
        {
            let mut blit = self.metal.blit_command_encoder()?;
            blit.set_label("tur_expert_fill");
            for (read, source) in reads.iter().zip(&staging) {
                blit.copy_from_buffer(source, 0, read.target, read.at, read.bytes.len());
            }
        }
        self.device.synchronize()?;
        for &(expert, slot) in &chosen {
            state.map[base + expert] =
                u32::try_from(slot).map_err(|_| invalid("too many expert slots"))?;
            state.owners[slot] = Some(base + expert);
            state.stats.misses += 1;
        }
        state.stats.bytes_read += reads.iter().map(|r| r.bytes.len() as u64).sum::<u64>();
        Ok(())
    }

    /// Run every read on scoped threads, so none can outlive this call or its buffers.
    fn read_all(&self, reads: &mut [Read<'_>]) -> Result<()> {
        let mut shares: Vec<Vec<&mut Read<'_>>> = (0..READ_THREADS).map(|_| Vec::new()).collect();
        for (i, read) in reads.iter_mut().enumerate() {
            shares[i % READ_THREADS].push(read);
        }
        let files = &self.files;
        std::thread::scope(|scope| {
            let handles: Vec<_> = shares
                .into_iter()
                .map(|share| {
                    scope.spawn(move || -> Result<()> {
                        for read in share {
                            let file = files
                                .get(read.file)
                                .ok_or_else(|| invalid("expert read has no file"))?;
                            file.read_exact_at(&mut read.bytes, read.offset)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .try_for_each(|h| h.join().map_err(|_| invalid("expert read thread failed"))?)
        })
    }
}

/// One contiguous read from a checkpoint and where it lands in an arena.
struct Read<'a> {
    file: usize,
    offset: u64,
    bytes: Vec<u8>,
    target: &'a Buffer,
    at: usize,
}

impl<'a> Read<'a> {
    fn new(
        file: usize,
        start: u64,
        expert: usize,
        size: usize,
        target: &'a Buffer,
        at: usize,
    ) -> Result<Self> {
        let offset = (expert as u64)
            .checked_mul(size as u64)
            .and_then(|skip| start.checked_add(skip))
            .ok_or_else(|| invalid("expert offset overflow"))?;
        Ok(Self {
            file,
            offset,
            bytes: vec![0; size],
            target,
            at,
        })
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        let stats = self.state.get_mut().stats;
        tracing::info!(
            slots = self.slots,
            hits = stats.hits,
            misses = stats.misses,
            evictions = stats.evictions,
            bytes_read = stats.bytes_read,
            wait_ms = stats.wait.as_millis() as u64,
            "Expert cache"
        );
    }
}

/// F32 tile rows `[expert, first row, rows, wide]`, as the kernels read them.
pub(crate) fn tile_tensor(values: Vec<f32>, device: &Device) -> Result<Tensor> {
    let rows = values.len() / 4;
    Ok(Tensor::from_vec(values, (rows, 4), device)?)
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_spare_memory_counts_only_pages_macos_reclaims_without_swapping() {
        let report = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
            Pages free:                                  10.\n\
            Pages active:                              1000.\n\
            Pages inactive:                              20.\n\
            Pages speculative:                            3.\n\
            Pages wired down:                          1000.\n\
            Pages purgeable:                              7.\n\
            File-backed pages:                            5.\n";
        // Free and purgeable, and at most the file-backed share of inactive pages.
        assert_eq!(reclaimable_bytes(report), Some(22 * 16384));
        assert_eq!(reclaimable_bytes("no page size here"), None);
    }

    #[test]
    fn test_budgets_below_one_layer_of_experts_are_rejected() -> Result<()> {
        let bank = BankShape { rows: 2, cols: 32 };
        let expert = ExpertCache::expert_bytes(bank, bank)?;
        assert_eq!(expert, 68);
        assert_eq!(ExpertCache::planned_slots(68 * 5 + 1, 4, bank, bank)?, 5);
        assert!(ExpertCache::planned_slots(68 * 3, 4, bank, bank).is_err());
        assert!(BankShape { rows: 2, cols: 31 }.blocks().is_err());
        Ok(())
    }
}
