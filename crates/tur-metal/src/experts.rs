#![forbid(unsafe_code)]
//! Bounded SSD-backed residency for expert banks stored as the kernels read them: MXFP4
//! checkpoints as published, or GGML blocks from a prepacked artifact or a GGUF file.
//!
//! Experts live in fixed arenas of slots, one per distinct pair of bank formats, because a
//! kernel reads one format at one stride; GGUF files mix formats across layers. A routed expert
//! absent from its arena is read from the checkpoint into a slot chosen by the policy in
//! [`ExpertCache::route`]. Nothing is ever written
//! to disk: files open read-only, evicted slots are clean and simply overwritten, and reads
//! bypass the page cache so the arena really bounds what experts occupy in memory.
//!
//! Slot lifetime needs no tracking. Reading a layer's routes waits for all earlier GPU work,
//! so no kernel can be using a slot while it is overwritten, and the copies complete before
//! the layer's own kernels are encoded. Candle's buffers are untracked, so this ordering is
//! the only thing that makes the overwrite safe.

use crate::{
    blocks::BlockFormat,
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
/// Batches staged at once: what a finished prefetch left, a demand read, and the next prefetch.
const STAGED_BATCHES: usize = 3;
/// Concurrent reads of one batch.
const READ_THREADS: usize = 4;
/// Memory left for the state pool, scratch, staging and everything else on the machine
/// beyond the arena. Measured on a 16 GB M1: 4 GiB arenas leaving less swapped out 700 to
/// 830 MB in a short run and a 6 GiB arena 2.4 GB; 1 GiB leaving this much swapped nothing.
const RESERVE_BYTES: usize = 2560 << 20;
/// Arenas below this cannot push other memory into swap, and skip the spare-memory check.
const NEGLIGIBLE_BYTES: usize = 256 << 20;
/// Marks an expert that holds no slot.
const ABSENT: u32 = u32::MAX;

/// How a bank is stored on disk and in its arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankFormat {
    /// 16-byte blocks of 32 values, with one scale byte per block stored apart.
    Mxfp4,
    /// Blocks with their scales inside them, every row in one tensor.
    Packed(BlockFormat),
    /// The same with each expert's first half of rows in one tensor and the rest in another,
    /// as GGUF stores gate and up.
    Split(BlockFormat),
}

/// Rows of one expert, input columns and storage of one bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankShape {
    pub rows: usize,
    pub cols: usize,
    pub format: BankFormat,
}

impl BankShape {
    fn mxfp4_blocks(self) -> Result<usize> {
        if self.rows == 0 || self.cols == 0 || !self.cols.is_multiple_of(MXFP4_BLOCK_VALUES) {
            return Err(invalid("expert banks need rows and whole MXFP4 blocks"));
        }
        checked_elements(&[self.rows, self.cols / MXFP4_BLOCK_VALUES])
    }

    /// Scale bytes one expert stores apart from its blocks.
    fn scale_bytes(self) -> Result<usize> {
        match self.format {
            BankFormat::Mxfp4 => self.mxfp4_blocks(),
            BankFormat::Packed(_) | BankFormat::Split(_) => Ok(0),
        }
    }

    fn block_bytes(self) -> Result<usize> {
        match self.format {
            BankFormat::Mxfp4 => checked_elements(&[self.mxfp4_blocks()?, MXFP4_BLOCK_BYTES]),
            BankFormat::Split(_) if !self.rows.is_multiple_of(2) => {
                Err(invalid("a split expert bank needs an even row count"))
            }
            BankFormat::Packed(format) | BankFormat::Split(format) => {
                // The kernels read 32 values at a time.
                if !format.tiles(self.cols) {
                    return Err(invalid(format!(
                        "the expert kernels cannot read {} rows of {} values",
                        format.name(),
                        self.cols
                    )));
                }
                format.bytes(self.rows, self.cols)
            }
        }
    }

    fn expert_bytes(self) -> Result<usize> {
        self.block_bytes()?
            .checked_add(self.scale_bytes()?)
            .ok_or_else(|| invalid("expert size overflow"))
    }

    /// Arena offset and size of each stored part of the expert in `slot`, in the order
    /// [`BankSource::parts`] lists them, for an arena whose scales start at `scales`.
    fn parts(self, slot: usize, scales: usize) -> Result<Vec<(usize, usize)>> {
        let block = self.block_bytes()?;
        let start = slot
            .checked_mul(block)
            .ok_or_else(|| invalid("expert slot offset overflow"))?;
        Ok(match self.format {
            BankFormat::Mxfp4 => {
                let scale = self.scale_bytes()?;
                vec![(start, block), (scales + slot * scale, scale)]
            }
            BankFormat::Packed(_) => vec![(start, block)],
            BankFormat::Split(_) => vec![(start, block / 2), (start + block / 2, block / 2)],
        })
    }
}

/// Where one layer's bank is stored, and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankSource {
    pub shape: BankShape,
    /// File and byte offset of expert 0 in each stored part: blocks then scales in MXFP4,
    /// gate rows then up rows when split, otherwise the one tensor.
    pub parts: Vec<(PathBuf, u64)>,
}

/// Every layer's expert storage in a checkpoint.
#[derive(Clone, Debug)]
pub struct ExpertSources {
    pub experts: usize,
    /// Experts one token selects in each layer.
    pub top_k: usize,
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
    /// Of `wait`: reads and staging uploads not already finished ahead of time, and the
    /// copies' GPU wait.
    pub read: Duration,
    pub copy: Duration,
    /// Time spent waiting for earlier GPU work when reading a layer's routes.
    pub gpu: Duration,
    /// Experts read ahead of their layer, and how many of those it then routed to.
    pub prefetched: u64,
    pub prefetch_used: u64,
}

/// One expert's bytes uploaded to staging buffers, ready to copy into any slot.
struct Staged {
    layer: usize,
    expert: usize,
    /// Every stored part of each bank, in order.
    parts: Vec<Arc<Buffer>>,
}

/// What reading needs, shared with the prefetch thread.
struct Source {
    metal: MetalDevice,
    files: Vec<File>,
    layers: Vec<SourceLayer>,
}

/// One layer's arena class and banks.
struct SourceLayer {
    class: usize,
    banks: [SourceBank; 2],
}

struct SourceBank {
    shape: BankShape,
    /// File index and offset of expert 0 in each stored part.
    parts: Vec<(usize, u64)>,
}

/// One read's position in its batch and its staging buffer.
type StagedPart = (usize, Arc<Buffer>);

/// What a prefetch thread returns: the read buffers it borrowed, and what it staged.
type Prefetched = (Vec<Vec<u8>>, Result<Vec<Staged>>);

struct Slots {
    /// Slot, counted across all classes, of each `layer * experts + expert`.
    map: Vec<u32>,
    /// Owner key and last use of each slot.
    owners: Vec<Option<usize>>,
    used: Vec<u64>,
    tick: u64,
    stats: ExpertStats,
    /// Host read buffers, one per read thread, kept so reads do not fault fresh pages in.
    buffers: Vec<Vec<u8>>,
    /// The same for prefetch, which runs while demand reads do.
    prefetch_buffers: Vec<Vec<u8>>,
    /// Reads started when the previous layer routed.
    pending: Option<std::thread::JoinHandle<Prefetched>>,
}

/// The slots of layers storing the same pair of bank formats.
struct Class {
    /// First slot of this class in the cache-wide numbering.
    first: usize,
    slots: usize,
    /// Fewer slots than one token touches: LRU would evict each just before its next use.
    cyclic: bool,
    gate_up: Arena,
    down: Arena,
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
            weights: match shape.format {
                BankFormat::Mxfp4 => {
                    PackedLinear::mxfp4_arena(Arc::clone(&buffer), blocks, (rows, shape.cols))?
                }
                BankFormat::Packed(format) | BankFormat::Split(format) => {
                    PackedLinear::packed_arena(Arc::clone(&buffer), format, (rows, shape.cols))?
                }
            },
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
    source: Arc<Source>,
    experts: usize,
    top_k: usize,
    classes: Vec<Class>,
    slots: usize,
    state: Mutex<Slots>,
}

/// Refuse an arena that would push other memory into swap, the only way this cache could
/// cost SSD writes. Called with the model's other weights already resident.
fn check_spare_memory(arena: usize) -> Result<()> {
    if arena < NEGLIGIBLE_BYTES {
        return Ok(());
    }
    let spare = crate::system::SystemMemory::read()?.spare_bytes;
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

/// Distinct bank pairs of `layers` in first-use order, and the class each layer reads.
fn classes(layers: &[[BankShape; 2]]) -> (Vec<[BankShape; 2]>, Vec<usize>) {
    let mut kinds: Vec<[BankShape; 2]> = Vec::new();
    let of_layer = layers
        .iter()
        .map(|layer| {
            kinds
                .iter()
                .position(|kind| kind == layer)
                .unwrap_or_else(|| {
                    kinds.push(*layer);
                    kinds.len() - 1
                })
        })
        .collect();
    (kinds, of_layer)
}

/// Slots of classes sized `(layers, bytes per expert)`, whose floors fit `budget`. A class whose
/// share would fall below `experts` takes exactly that, and the others share what remains.
fn share_slots(budget: usize, experts: usize, classes: &[(usize, usize)]) -> Vec<usize> {
    let mut floored = vec![false; classes.len()];
    loop {
        let (mut left, mut weight) = (budget as u128, 0u128);
        for (&(layers, expert), &floor) in classes.iter().zip(&floored) {
            if floor {
                left -= (experts * expert) as u128;
            } else {
                weight += (layers * expert) as u128;
            }
        }
        let share = |layers: usize| (left * layers as u128 / weight.max(1)) as usize;
        let short: Vec<usize> = (0..classes.len())
            .filter(|&c| !floored[c] && share(classes[c].0) < experts)
            .collect();
        if short.is_empty() {
            return classes
                .iter()
                .zip(&floored)
                .map(|(&(layers, _), &floor)| if floor { experts } else { share(layers) })
                .collect();
        }
        for class in short {
            floored[class] = true;
        }
    }
}

/// Open every file `sources` names once, and locate each layer's parts in them.
fn open_sources(
    sources: &ExpertSources,
    of_layer: &[usize],
) -> Result<(Vec<File>, Vec<SourceLayer>)> {
    let mut paths: BTreeMap<&Path, usize> = BTreeMap::new();
    let mut files = Vec::new();
    let mut layers = Vec::with_capacity(sources.layers.len());
    for (banks, &class) in sources.layers.iter().zip(of_layer) {
        let mut located = Vec::with_capacity(2);
        for bank in banks {
            if bank.parts.len() != bank.shape.parts(0, 0)?.len() {
                return Err(invalid("expert bank source does not match its format"));
            }
            let mut parts = Vec::with_capacity(bank.parts.len());
            for (path, offset) in &bank.parts {
                let index = if let Some(index) = paths.get(path.as_path()) {
                    *index
                } else {
                    files.push(open(path)?);
                    paths.insert(path, files.len() - 1);
                    files.len() - 1
                };
                parts.push((index, *offset));
            }
            located.push(SourceBank {
                shape: bank.shape,
                parts,
            });
        }
        let banks: [SourceBank; 2] = located
            .try_into()
            .map_err(|_| invalid("an expert layer has two banks"))?;
        layers.push(SourceLayer { class, banks });
    }
    Ok((files, layers))
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

    /// Slots of each class of `layers` (see [`classes`]) that fit `budget` bytes. Each class
    /// must hold every expert of one layer: one pass may route to all of them, and they stay
    /// pinned while that layer's kernels run. Beyond that, every layer gets the same share of
    /// slots, so a few layers in a wider format do not get a worse hit rate than the rest.
    pub fn planned_slots(
        budget: usize,
        experts: usize,
        layers: &[[BankShape; 2]],
    ) -> Result<Vec<usize>> {
        let (classes, of_layer) = classes(layers);
        let mut sized = Vec::with_capacity(classes.len());
        for (class, &[gate_up, down]) in classes.iter().enumerate() {
            let count = of_layer.iter().filter(|&&c| c == class).count();
            sized.push((count, Self::expert_bytes(gate_up, down)?));
        }
        let mut floor = 0usize;
        for &(_, expert) in &sized {
            floor = checked_elements(&[experts, expert])?
                .checked_add(floor)
                .ok_or_else(|| invalid("expert cache size overflow"))?;
        }
        if budget < floor {
            return Err(invalid(format!(
                "an expert cache of {budget} bytes cannot hold every expert of one layer in each \
                 of its {} bank formats; that needs {floor} bytes",
                sized.len()
            )));
        }
        Ok(share_slots(budget, experts, &sized))
    }

    /// Device bytes of the arenas [`Self::planned_slots`] allows.
    pub fn planned_bytes(
        budget: usize,
        experts: usize,
        layers: &[[BankShape; 2]],
    ) -> Result<usize> {
        let (classes, _) = classes(layers);
        let mut total = 0usize;
        for (slots, [gate_up, down]) in Self::planned_slots(budget, experts, layers)?
            .into_iter()
            .zip(classes)
        {
            total = checked_elements(&[slots, Self::expert_bytes(gate_up, down)?])?
                .checked_add(total)
                .ok_or_else(|| invalid("expert cache size overflow"))?;
        }
        Ok(total)
    }

    /// Host and upload memory the largest read batch can hold beside the arenas.
    pub fn staging_bytes(layers: &[[BankShape; 2]]) -> Result<usize> {
        let mut widest = 0;
        for &[gate_up, down] in layers {
            widest = widest.max(Self::expert_bytes(gate_up, down)?);
        }
        checked_elements(&[STAGED_BATCHES, READ_BATCH, widest])
    }

    /// Allocate the arenas and open every source file.
    ///
    /// `unloaded` is what the model's other weights will still take, since the spare-memory
    /// check can only see what is already resident.
    pub fn new(
        sources: &ExpertSources,
        budget: usize,
        unloaded: usize,
        device: &Device,
    ) -> Result<Self> {
        let Device::Metal(metal) = device else {
            return Err(invalid("the expert cache runs on a Metal device"));
        };
        let shapes: Vec<[BankShape; 2]> = sources
            .layers
            .iter()
            .map(|[gate_up, down]| [gate_up.shape, down.shape])
            .collect();
        let experts = sources.experts;
        let planned = Self::planned_slots(budget, experts, &shapes)?;
        check_spare_memory(
            Self::planned_bytes(budget, experts, &shapes)?
                .checked_add(unloaded)
                .ok_or_else(|| invalid("expert cache size overflow"))?,
        )?;
        let slots = planned.iter().sum::<usize>();
        if u32::try_from(slots).is_err() {
            return Err(invalid("too many expert slots"));
        }
        let keys = checked_elements(&[sources.layers.len(), experts])?;
        let (kinds, of_layer) = classes(&shapes);
        let mut classes = Vec::with_capacity(kinds.len());
        let mut first = 0;
        for (index, (&[gate_up, down], &count)) in kinds.iter().zip(&planned).enumerate() {
            let layers = of_layer.iter().filter(|&&c| c == index).count();
            classes.push(Class {
                first,
                slots: count,
                cyclic: count < checked_elements(&[layers, sources.top_k])?,
                gate_up: Arena::new(metal, gate_up, count)?,
                down: Arena::new(metal, down, count)?,
            });
            first += count;
        }
        let (files, layers) = open_sources(sources, &of_layer)?;
        Ok(Self {
            device: device.clone(),
            metal: metal.clone(),
            source: Arc::new(Source {
                metal: metal.clone(),
                files,
                layers,
            }),
            experts,
            top_k: sources.top_k,
            classes,
            slots,
            state: Mutex::new(Slots {
                map: vec![ABSENT; keys],
                owners: vec![None; slots],
                used: vec![0; slots],
                tick: 0,
                stats: ExpertStats::default(),
                buffers: (0..READ_THREADS).map(|_| Vec::new()).collect(),
                prefetch_buffers: (0..READ_THREADS).map(|_| Vec::new()).collect(),
                pending: None,
            }),
        })
    }

    /// The class `layer` reads.
    fn class(&self, layer: usize) -> Result<&Class> {
        self.source
            .layers
            .get(layer)
            .and_then(|source| self.classes.get(source.class))
            .ok_or_else(|| invalid("expert cache has no such layer"))
    }

    /// The `[gate_up, down]` arenas of `layer`, addressed by the slots of routes from
    /// [`Self::route`].
    pub fn arenas(&self, layer: usize) -> Result<[&PackedLinear; 2]> {
        let class = self.class(layer)?;
        Ok([&class.gate_up.weights, &class.down.weights])
    }

    /// Handles to the arenas for `layer` to read: `[gate_up, down]`.
    pub fn arena_handles(&self, layer: usize) -> Result<[PackedLinear; 2]> {
        let class = self.class(layer)?;
        Ok([
            class.gate_up.weights.arena_handle()?,
            class.down.weights.arena_handle()?,
        ])
    }

    /// Device bytes the arenas hold.
    pub fn resident_bytes(&self) -> usize {
        self.classes
            .iter()
            .map(|class| {
                class.gate_up.weights.resident_bytes() + class.down.weights.resident_bytes()
            })
            .sum()
    }

    /// Experts the arenas hold at once.
    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn stats(&self) -> ExpertStats {
        self.state.lock().stats
    }

    /// Make every expert `routes` selects for `layer` resident and address it by slot.
    /// Waits for all earlier GPU work, and returns only once the copies have completed. On
    /// an I/O failure the affected slots are left ownerless, so a later pass reads them again.
    ///
    /// `next_logits` are the following layer's router scores on this layer's input, already
    /// encoded. They predict what that layer will route to, and reading those experts starts
    /// now so that the SSD works while the GPU runs this layer.
    pub fn route(
        &self,
        k: &Kernels,
        layer: usize,
        routes: ExpertRoutes,
        next_logits: Option<&Tensor>,
    ) -> Result<ExpertRoutes> {
        let class = self.class(layer)?;
        // The readback is the wait the slot lifetime rule relies on.
        let waited = Instant::now();
        let mut tiles = routes.tile_values()?;
        let gpu = waited.elapsed();
        let started = Instant::now();
        let mut state = self.state.lock();
        state.stats.gpu += gpu;
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
        tracing::trace!(
            target: "tur_expert_trace",
            layer,
            experts = ?seen
                .iter()
                .enumerate()
                .filter_map(|(expert, &routed)| routed.then_some(expert))
                .collect::<Vec<_>>(),
        );
        let wrote = !misses.is_empty();
        let mut ready = Self::finish_prefetch(&mut state);
        let claimed = self.claim(&mut state, layer, &misses)?;
        // Started before this layer's reads and copies so that the SSD works through their
        // waits too, and after the claim so that it sees which experts those just evicted.
        if let Some(logits) = next_logits {
            self.start_prefetch(&mut state, layer + 1, logits)?;
        }
        for batch in claimed.chunks(READ_BATCH) {
            self.fill(&mut state, layer, batch, &mut ready)?;
        }
        for tile in tiles.as_chunks_mut::<4>().0 {
            if tile[0] >= 0.0 {
                tile[0] = (state.map[base + tile[0] as usize] as usize - class.first) as f32;
            }
        }
        if wrote {
            state.stats.wait += started.elapsed();
        }
        drop(state);
        k.with_slots(routes, tiles, class.slots)
    }

    /// Experts the finished prefetch staged. Its read errors are dropped on purpose: a read
    /// the pass needs is retried on demand, and reports its own failure there.
    fn finish_prefetch(state: &mut Slots) -> Vec<Staged> {
        let Some(handle) = state.pending.take() else {
            return Vec::new();
        };
        let Ok((buffers, staged)) = handle.join() else {
            state.prefetch_buffers = (0..READ_THREADS).map(|_| Vec::new()).collect();
            return Vec::new();
        };
        state.prefetch_buffers = buffers;
        staged.unwrap_or_default()
    }

    /// Begin reading the absent experts that `logits` rank highest for `layer`, on its own
    /// thread. Nothing it does touches a slot, so it needs no ordering with the GPU, and
    /// [`Self::finish_prefetch`] or the drop of this cache always joins it.
    fn start_prefetch(&self, state: &mut Slots, layer: usize, logits: &Tensor) -> Result<()> {
        let base = layer * self.experts;
        let mut votes = vec![0u32; self.experts];
        for row in logits.to_vec2::<f32>()? {
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
            for &expert in order.iter().take(self.top_k) {
                if let Some(vote) = votes.get_mut(expert) {
                    *vote += 1;
                }
            }
        }
        let mut wanted: Vec<usize> = (0..self.experts)
            .filter(|&e| votes[e] > 0 && state.map[base + e] == ABSENT)
            .collect();
        wanted.sort_by_key(|&e| std::cmp::Reverse(votes[e]));
        wanted.truncate(READ_BATCH);
        if wanted.is_empty() {
            return Ok(());
        }
        state.stats.prefetched += wanted.len() as u64;
        state.stats.bytes_read += self.source.read_bytes(layer)? * wanted.len() as u64;
        let source = Arc::clone(&self.source);
        let mut buffers = std::mem::take(&mut state.prefetch_buffers);
        state.pending = Some(std::thread::spawn(move || {
            let staged = source.stage(layer, &wanted, &mut buffers);
            (buffers, staged)
        }));
        Ok(())
    }

    /// Choose a free or evicted slot for each of `experts`. A slot loses its owner here, before
    /// it is touched, so a failure cannot leave one claiming contents it does not hold.
    fn claim(
        &self,
        state: &mut Slots,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<(usize, usize)>> {
        let tick = state.tick;
        let layers = self.source.layers.len();
        let class = self.class(layer)?;
        let mut chosen = Vec::with_capacity(experts.len());
        for &expert in experts {
            // Slots stamped this tick belong to this layer's own routes: the budget holds a
            // whole layer, so a victim always exists. Passes visit layers in order, so when a
            // token's experts exceed the slots, LRU hits nothing (0% against 34% at 81 slots
            // of 96) and the victim is what runs furthest ahead. With more slots LRU keeps
            // what routing repeats (79% against 65% at 324 slots). Replaying a real trace,
            // LFU and LRU-2 gained at most two points over this.
            let slot = (class.first..class.first + class.slots)
                .filter(|&s| state.used[s] != tick)
                .max_by_key(|&s| match state.owners[s] {
                    None => (usize::MAX, 0),
                    Some(owner) if class.cyclic => {
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
        Ok(chosen)
    }

    /// Bring `chosen` experts of `layer` into their claimed slots, from `ready` where prefetch
    /// already staged them, and wait for the copies.
    fn fill(
        &self,
        state: &mut Slots,
        layer: usize,
        chosen: &[(usize, usize)],
        ready: &mut Vec<Staged>,
    ) -> Result<()> {
        let base = layer * self.experts;
        let mut staged = Vec::with_capacity(chosen.len());
        let mut missing = Vec::new();
        for &(expert, _) in chosen {
            match ready
                .iter()
                .position(|r| r.layer == layer && r.expert == expert)
            {
                Some(found) => {
                    staged.push(ready.swap_remove(found));
                    state.stats.prefetch_used += 1;
                }
                None => missing.push(expert),
            }
        }
        if !missing.is_empty() {
            let step = Instant::now();
            let mut buffers = std::mem::take(&mut state.buffers);
            let read = self.source.stage(layer, &missing, &mut buffers);
            state.buffers = buffers;
            staged.extend(read?);
            state.stats.read += step.elapsed();
            state.stats.bytes_read += self.source.read_bytes(layer)? * missing.len() as u64;
        }
        let step = Instant::now();
        let class = self.class(layer)?;
        {
            let mut blit = self.metal.blit_command_encoder()?;
            blit.set_label("tur_expert_fill");
            for &(expert, slot) in chosen {
                let parts = staged
                    .iter()
                    .find(|r| r.expert == expert)
                    .ok_or_else(|| invalid("expert was not staged"))?;
                Self::copy(&mut blit, class, &parts.parts, slot - class.first)?;
            }
        }
        // Staging buffers stay alive in `staged` until their copies have completed.
        self.device.synchronize()?;
        state.stats.copy += step.elapsed();
        for &(expert, slot) in chosen {
            state.map[base + expert] =
                u32::try_from(slot).map_err(|_| invalid("too many expert slots"))?;
            state.owners[slot] = Some(base + expert);
            state.stats.misses += 1;
        }
        Ok(())
    }

    /// Encode one expert's copies from staging into `slot` of its class's arenas.
    fn copy(
        blit: &mut candle_metal_kernels::metal::BlitCommandsGuard<'_>,
        class: &Class,
        parts: &[Arc<Buffer>],
        slot: usize,
    ) -> Result<()> {
        let mut parts = parts.iter();
        for arena in [&class.gate_up, &class.down] {
            for (offset, size) in arena.shape.parts(slot, arena.scales)? {
                let part = parts
                    .next()
                    .ok_or_else(|| invalid("staged expert lacks a part"))?;
                blit.copy_from_buffer(part, 0, &arena.buffer, offset, size);
            }
        }
        Ok(())
    }
}

impl Source {
    /// Bytes one expert of `layer` reads.
    fn read_bytes(&self, layer: usize) -> Result<u64> {
        let [gate_up, down] = &self
            .layers
            .get(layer)
            .ok_or_else(|| invalid("expert cache has no such layer"))?
            .banks;
        Ok(ExpertCache::expert_bytes(gate_up.shape, down.shape)? as u64)
    }

    /// Read `experts` of `layer` and upload them to staging buffers, on scoped threads so
    /// none outlives this call. Uploads run in the read threads: one at a time they took
    /// almost as long as the reads themselves (24 s against 33 s for 160 tokens).
    fn stage(
        &self,
        layer: usize,
        experts: &[usize],
        buffers: &mut [Vec<u8>],
    ) -> Result<Vec<Staged>> {
        let banks = &self
            .layers
            .get(layer)
            .ok_or_else(|| invalid("expert cache has no such layer"))?
            .banks;
        let per_expert = banks.iter().map(|bank| bank.parts.len()).sum::<usize>();
        let mut parts = Vec::with_capacity(per_expert * experts.len());
        for &expert in experts {
            for bank in banks {
                for (&(file, start), (_, size)) in bank.parts.iter().zip(bank.shape.parts(0, 0)?) {
                    parts.push((file, part_offset(start, expert, size)?, size));
                }
            }
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (parts, next, this) = (&parts, &next, self);
        let uploaded = std::thread::scope(|scope| -> Result<Vec<Vec<StagedPart>>> {
            let handles: Vec<_> = buffers
                .iter_mut()
                .map(|buffer| scope.spawn(move || this.stage_parts(parts, next, buffer)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().map_err(|_| invalid("expert read thread failed"))?)
                .collect()
        })?;
        let mut slots: Vec<Option<Arc<Buffer>>> = vec![None; parts.len()];
        for (at, buffer) in uploaded.into_iter().flatten() {
            *slots
                .get_mut(at)
                .ok_or_else(|| invalid("expert read outside its batch"))? = Some(buffer);
        }
        let mut slots = slots.into_iter();
        experts
            .iter()
            .map(|&expert| {
                let parts = slots
                    .by_ref()
                    .take(per_expert)
                    .map(|part| part.ok_or_else(|| invalid("expert part was not read")))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Staged {
                    layer,
                    expert,
                    parts,
                })
            })
            .collect()
    }

    /// One thread's share of a batch: claim parts until none remain.
    fn stage_parts(
        &self,
        parts: &[(usize, u64, usize)],
        next: &std::sync::atomic::AtomicUsize,
        buffer: &mut Vec<u8>,
    ) -> Result<Vec<StagedPart>> {
        let mut done = Vec::new();
        loop {
            let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&(file, offset, size)) = parts.get(at) else {
                return Ok(done);
            };
            let file = self
                .files
                .get(file)
                .ok_or_else(|| invalid("expert read has no file"))?;
            buffer.resize(size, 0);
            file.read_exact_at(buffer, offset)?;
            done.push((at, self.metal.new_buffer_with_data(buffer.as_slice())?));
        }
    }
}

/// Stored offset of `expert`'s part of `size` bytes, from the offset of expert 0's.
fn part_offset(start: u64, expert: usize, size: usize) -> Result<u64> {
    (expert as u64)
        .checked_mul(size as u64)
        .and_then(|skip| start.checked_add(skip))
        .ok_or_else(|| invalid("expert offset overflow"))
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        if let Some(pending) = state.pending.take() {
            // Only its completion matters: a prefetch nobody will use has nothing to report.
            drop(pending.join());
        }
        let stats = state.stats;
        tracing::info!(
            slots = self.slots,
            hits = stats.hits,
            misses = stats.misses,
            evictions = stats.evictions,
            bytes_read = stats.bytes_read,
            wait_ms = stats.wait.as_millis() as u64,
            read_ms = stats.read.as_millis() as u64,
            prefetched = stats.prefetched,
            prefetch_used = stats.prefetch_used,
            copy_ms = stats.copy.as_millis() as u64,
            gpu_ms = stats.gpu.as_millis() as u64,
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
    fn test_budgets_below_one_layer_of_experts_are_rejected() -> Result<()> {
        let bank = BankShape {
            rows: 2,
            cols: 32,
            format: BankFormat::Mxfp4,
        };
        let expert = ExpertCache::expert_bytes(bank, bank)?;
        assert_eq!(expert, 68);
        let layers = [[bank, bank]];
        assert_eq!(ExpertCache::planned_slots(68 * 5 + 1, 4, &layers)?, [5]);
        assert!(ExpertCache::planned_slots(68 * 3, 4, &layers).is_err());
        assert!(
            BankShape {
                rows: 2,
                cols: 31,
                format: BankFormat::Mxfp4
            }
            .block_bytes()
            .is_err()
        );
        let q4k = BankShape {
            rows: 2,
            cols: 256,
            format: BankFormat::Packed(BlockFormat::Q4K),
        };
        assert_eq!((q4k.block_bytes()?, q4k.scale_bytes()?), (288, 0));
        Ok(())
    }

    #[test]
    fn test_split_banks_place_their_parts_inside_one_slot() -> Result<()> {
        let split = BankShape {
            rows: 4,
            cols: 256,
            format: BankFormat::Split(BlockFormat::Q4K),
        };
        assert_eq!(split.parts(2, 0)?, [(1152, 288), (1440, 288)]);
        let odd = BankShape { rows: 3, ..split };
        assert!(odd.block_bytes().is_err());
        Ok(())
    }

    /// Layers share slots evenly whatever their format, and a class too small for its floor
    /// takes the floor out of the others' share rather than exceeding the budget.
    #[test]
    fn test_budget_is_shared_per_layer_across_format_classes() -> Result<()> {
        let bank = |format| BankShape {
            rows: 2,
            cols: 256,
            format: BankFormat::Packed(format),
        };
        let narrow = [bank(BlockFormat::Q4K), bank(BlockFormat::Q4K)];
        let wide = [bank(BlockFormat::Q8_0), bank(BlockFormat::Q8_0)];
        let mut layers = vec![narrow; 9];
        layers.push(wide);
        // One wide layer's even share is under its floor of 40 experts, so it takes the floor.
        let expert = |shape: [BankShape; 2]| ExpertCache::expert_bytes(shape[0], shape[1]);
        let budget = 200_000;
        let slots = ExpertCache::planned_slots(budget, 40, &layers)?;
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[1], 40);
        let total = slots[0] * expert(narrow)? + slots[1] * expert(wide)?;
        assert_eq!(total, ExpertCache::planned_bytes(budget, 40, &layers)?);
        assert!(total <= budget && budget - total < expert(narrow)?);
        // Both floors must fit, not just the larger.
        let floor = 40 * (expert(narrow)? + expert(wide)?);
        assert!(ExpertCache::planned_slots(floor, 40, &layers).is_ok());
        assert!(ExpertCache::planned_slots(floor - 1, 40, &layers).is_err());
        Ok(())
    }
}
