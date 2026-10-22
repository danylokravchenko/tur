#![forbid(unsafe_code)]
//! Persistent paged K/V and recurrent state pools with generation-tagged leases.
//!
//! A page id spans every attention layer: page `p` owns slots `p * page_tokens ..` in each
//! layer's key and value tensors, so one page table addresses the whole model. A recurrent
//! slot id likewise spans every recurrent layer. Slot 0 is a permanent zero state shared by
//! every fresh sequence and never written.
//!
//! Leases are reference counted. Dropping the last holder retires the id; retired ids
//! return to the free list only through [`StatePool::reclaim`], which the execution owner
//! calls when no unfenced device work can reference them. Reclaiming bumps the id's
//! generation, so an id captured before reuse no longer resolves.

use candle_core::{DType, Device, Tensor};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tur_core::{
    Result, TurError,
    execution::{KvFormat, LayerStateKind, ModelDescription, invalid},
};

/// Pool capacities: K/V pages and usable recurrent slots (the zero slot is extra).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolConfig {
    pub pages: usize,
    pub recurrent_slots: usize,
}

impl PoolConfig {
    /// Fit `bytes`: two recurrent slots per concurrent sequence (its committed state and one
    /// in-flight state), then as many K/V pages as remain. Fails if either kind of state the
    /// model needs gets no capacity.
    pub fn for_budget(
        description: &ModelDescription,
        bytes: usize,
        sequences: usize,
    ) -> Result<Self> {
        let recurrent_slots = if description.has_recurrent() {
            sequences
                .checked_mul(2)
                .ok_or_else(|| invalid("recurrent slot count overflow"))?
        } else {
            0
        };
        let slot_bytes = recurrent_slots
            .checked_mul(description.recurrent_slot_bytes()?)
            .ok_or_else(|| invalid("recurrent pool size overflow"))?;
        let remaining = bytes.checked_sub(slot_bytes).ok_or_else(|| {
            TurError::MemoryAllocation(format!(
                "state budget of {bytes} bytes cannot hold {sequences} sequences' recurrent \
                 state ({slot_bytes} bytes)"
            ))
        })?;
        let pages = if description.has_attention() {
            remaining / description.page_bytes()?.max(1)
        } else {
            0
        };
        let config = Self {
            pages,
            recurrent_slots,
        };
        config.validate(description)?;
        Ok(config)
    }

    /// Device bytes of pools with this capacity, including the zero slot.
    pub fn bytes(&self, description: &ModelDescription) -> Result<usize> {
        let slots = if description.has_recurrent() {
            self.recurrent_slots + 1
        } else {
            0
        };
        self.pages
            .checked_mul(description.page_bytes()?)
            .zip(slots.checked_mul(description.recurrent_slot_bytes()?))
            .and_then(|(p, s)| p.checked_add(s))
            .ok_or_else(|| invalid("pool size overflow"))
    }

    fn validate(&self, description: &ModelDescription) -> Result<()> {
        if description.has_attention() && self.pages == 0 {
            return Err(TurError::MemoryAllocation(
                "state budget leaves no K/V pages".into(),
            ));
        }
        if description.has_recurrent() && self.recurrent_slots == 0 {
            return Err(TurError::MemoryAllocation(
                "state budget leaves no recurrent slots".into(),
            ));
        }
        let slots = self
            .pages
            .checked_mul(description.kv.page_tokens)
            .ok_or_else(|| invalid("K/V slot count overflow"))?;
        u32::try_from(slots).map_err(|_| invalid("K/V pool exceeds u32 slot indices"))?;
        u32::try_from(self.recurrent_slots + 1)
            .map_err(|_| invalid("recurrent pool exceeds u32 slot indices"))?;
        Ok(())
    }
}

/// Occupancy of one id space. Free ids are `capacity - leased - retired`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub capacity: usize,
    /// Held by live states, checkpoints or in-flight reservations.
    pub leased: usize,
    /// Released but not yet reclaimed after a completion fence.
    pub retired: usize,
    /// Most ids leased at once.
    pub peak: usize,
}

impl Usage {
    /// Ids that can be leased now.
    pub fn free(&self) -> usize {
        self.capacity - self.leased - self.retired
    }
}

/// Occupancy and unit sizes of both pools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolUsage {
    pub pages: Usage,
    pub slots: Usage,
    pub page_bytes: usize,
    pub slot_bytes: usize,
}

impl PoolUsage {
    /// Device bytes reserved by the pools, used or not, including the zero slot.
    pub fn resident_bytes(&self) -> usize {
        let slots = if self.slot_bytes > 0 {
            self.slots.capacity + 1
        } else {
            0
        };
        self.pages.capacity * self.page_bytes + slots * self.slot_bytes
    }

    /// Bytes held by leases.
    pub fn leased_bytes(&self) -> usize {
        self.pages.leased * self.page_bytes + self.slots.leased * self.slot_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Space {
    Pages,
    Slots,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Free,
    Leased,
    Retired,
}

struct Allocator {
    generations: Vec<u32>,
    status: Vec<Status>,
    free: Vec<u32>,
    retired: Vec<u32>,
    /// Ids below this are permanent (the zero slot) and never leased.
    reserved: usize,
    leased: usize,
    peak: usize,
}

impl Allocator {
    fn new(capacity: usize, reserved: usize) -> Self {
        let total = capacity + reserved;
        let mut status = vec![Status::Free; total];
        status
            .iter_mut()
            .take(reserved)
            .for_each(|s| *s = Status::Leased);
        Self {
            generations: vec![0; total],
            status,
            // Lowest ids first, so fresh pools fill pages in address order.
            free: (reserved as u32..total as u32).rev().collect(),
            retired: Vec::new(),
            reserved,
            leased: 0,
            peak: 0,
        }
    }

    fn usage(&self) -> Usage {
        Usage {
            capacity: self.status.len() - self.reserved,
            leased: self.leased,
            retired: self.retired.len(),
            peak: self.peak,
        }
    }

    fn lease(&mut self) -> Option<Id> {
        let index = self.free.pop()?;
        let slot = self.status.get_mut(index as usize)?;
        *slot = Status::Leased;
        self.leased += 1;
        self.peak = self.peak.max(self.leased);
        Some(Id {
            index,
            generation: self.generations.get(index as usize).copied()?,
        })
    }

    fn retire(&mut self, id: Id) {
        let index = id.index as usize;
        if index >= self.reserved
            && self.generations.get(index) == Some(&id.generation)
            && let Some(status @ Status::Leased) = self.status.get_mut(index)
        {
            *status = Status::Retired;
            self.leased -= 1;
            self.retired.push(id.index);
        }
    }

    fn reclaim(&mut self) -> usize {
        let count = self.retired.len();
        for index in self.retired.drain(..) {
            let i = index as usize;
            if let (Some(status), Some(generation)) =
                (self.status.get_mut(i), self.generations.get_mut(i))
            {
                *status = Status::Free;
                *generation = generation.wrapping_add(1);
                self.free.push(index);
            }
        }
        count
    }

    fn live(&self, id: Id) -> bool {
        self.status.get(id.index as usize) == Some(&Status::Leased)
            && self.generations.get(id.index as usize) == Some(&id.generation)
    }
}

/// A generation-tagged page or slot index. It resolves only while its lease is live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Id {
    pub index: u32,
    pub generation: u32,
}

/// One leased page or slot. Dropping the last reference retires it.
pub(crate) struct Lease {
    pool: Arc<Shared>,
    space: Space,
    id: Id,
}

impl Lease {
    pub(crate) fn id(&self) -> Id {
        self.id
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.allocator(self.space).lock().retire(self.id);
    }
}

/// Stored keys or values of one attention layer: `[slots, kv_heads, head_dim]` codes and,
/// for quantized formats, `[slots, kv_heads, head_dim / group]` F16 scales.
pub struct KvPool {
    pub data: Tensor,
    pub scales: Option<Tensor>,
}

/// One attention layer's paged storage.
pub struct AttentionPool {
    pub key: KvPool,
    pub value: KvPool,
    pub format: KvFormat,
    pub page_tokens: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// One recurrent layer's slots: `[slots, conv_width, conv_channels]` histories and
/// `[slots, heads, key_dim, value_dim]` matrices.
pub struct RecurrentPool {
    pub convolution: Tensor,
    pub state: Tensor,
}

enum LayerPool {
    Attention(AttentionPool),
    Recurrent(RecurrentPool),
}

struct Shared {
    id: u64,
    description: Arc<ModelDescription>,
    device: Device,
    layers: Vec<LayerPool>,
    pages: Mutex<Allocator>,
    slots: Mutex<Allocator>,
}

impl Shared {
    fn allocator(&self, space: Space) -> &Mutex<Allocator> {
        match space {
            Space::Pages => &self.pages,
            Space::Slots => &self.slots,
        }
    }
}

static POOL_IDS: AtomicU64 = AtomicU64::new(1);

/// The device's recommended working set in bytes, if it reports one (Metal). Allocations
/// beyond it rely on paging rather than resident memory.
pub fn working_set_bytes(device: &Device) -> Option<usize> {
    match device {
        #[cfg(feature = "metal")]
        Device::Metal(metal) => Some(metal.recommended_max_working_set_size()),
        _ => None,
    }
}

/// Device-resident state storage for one model on one device. Cloning shares the pool.
#[derive(Clone)]
pub struct StatePool(Arc<Shared>);

impl StatePool {
    /// Validate the layout and allocate zeroed pools of `config` capacity on `device`.
    pub fn new(
        description: Arc<ModelDescription>,
        device: &Device,
        config: PoolConfig,
    ) -> Result<Self> {
        description.validate()?;
        config.validate(&description)?;
        let kv = description.kv;
        let slots = config.pages * kv.page_tokens;
        let recurrent_slots = config.recurrent_slots + 1;
        let mut layers = Vec::with_capacity(description.layers.len());
        for kind in &description.layers {
            layers.push(match *kind {
                LayerStateKind::Attention { kv_heads, head_dim } => {
                    let store = || -> Result<KvPool> {
                        Ok(KvPool {
                            data: Tensor::zeros(
                                (slots, kv_heads, head_dim),
                                kv.format.data_dtype(),
                                device,
                            )?,
                            scales: kv
                                .format
                                .group()
                                .map(|g| {
                                    Tensor::zeros(
                                        (slots, kv_heads, head_dim / g),
                                        DType::F16,
                                        device,
                                    )
                                })
                                .transpose()?,
                        })
                    };
                    LayerPool::Attention(AttentionPool {
                        key: store()?,
                        value: store()?,
                        format: kv.format,
                        page_tokens: kv.page_tokens,
                        kv_heads,
                        head_dim,
                    })
                }
                LayerStateKind::Recurrent {
                    conv_channels,
                    conv_width,
                    heads,
                    key_dim,
                    value_dim,
                    dtype,
                } => LayerPool::Recurrent(RecurrentPool {
                    convolution: Tensor::zeros(
                        (recurrent_slots, conv_width, conv_channels),
                        dtype,
                        device,
                    )?,
                    state: Tensor::zeros(
                        (recurrent_slots, heads, key_dim, value_dim),
                        dtype,
                        device,
                    )?,
                }),
            });
        }
        let slot_capacity = if description.has_recurrent() {
            config.recurrent_slots
        } else {
            0
        };
        Ok(Self(Arc::new(Shared {
            id: POOL_IDS.fetch_add(1, Ordering::Relaxed),
            description,
            device: device.clone(),
            layers,
            pages: Mutex::new(Allocator::new(config.pages, 0)),
            slots: Mutex::new(Allocator::new(slot_capacity, 1)),
        })))
    }

    /// Model layout the pool was built for.
    pub fn description(&self) -> &ModelDescription {
        &self.0.description
    }

    /// Device holding every pool tensor.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// Whether both handles share one pool.
    pub fn same_pool(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }

    /// Current occupancy and unit sizes.
    pub fn usage(&self) -> Result<PoolUsage> {
        Ok(PoolUsage {
            pages: self.0.pages.lock().usage(),
            slots: self.0.slots.lock().usage(),
            page_bytes: self.0.description.page_bytes()?,
            slot_bytes: if self.0.description.has_recurrent() {
                self.0.description.recurrent_slot_bytes()?
            } else {
                0
            },
        })
    }

    /// Return retired ids to the free lists and bump their generations. Call only when no
    /// unfenced device work can reference them; returns the number reclaimed.
    pub fn reclaim(&self) -> usize {
        self.0.pages.lock().reclaim() + self.0.slots.lock().reclaim()
    }

    /// Storage of attention layer `layer`.
    pub fn attention(&self, layer: usize) -> Result<&AttentionPool> {
        match self.0.layers.get(layer) {
            Some(LayerPool::Attention(pool)) => Ok(pool),
            _ => Err(invalid(format!("layer {layer} has no attention pool"))),
        }
    }

    /// Storage of recurrent layer `layer`.
    pub fn recurrent(&self, layer: usize) -> Result<&RecurrentPool> {
        match self.0.layers.get(layer) {
            Some(LayerPool::Recurrent(pool)) => Ok(pool),
            _ => Err(invalid(format!("layer {layer} has no recurrent pool"))),
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.0.id
    }

    fn lease(&self, space: Space) -> Result<Arc<Lease>> {
        let id = self.0.allocator(space).lock().lease().ok_or_else(|| {
            TurError::MemoryAllocation(match space {
                Space::Pages => "K/V page pool exhausted".into(),
                Space::Slots => "recurrent state pool exhausted".into(),
            })
        })?;
        Ok(Arc::new(Lease {
            pool: Arc::clone(&self.0),
            space,
            id,
        }))
    }

    pub(crate) fn lease_page(&self) -> Result<Arc<Lease>> {
        self.lease(Space::Pages)
    }

    pub(crate) fn lease_slot(&self) -> Result<Arc<Lease>> {
        self.lease(Space::Slots)
    }

    /// Resolve page ids to indices, failing if any was released and reclaimed.
    pub(crate) fn resolve_pages(&self, ids: &[Id]) -> Result<Vec<u32>> {
        resolve(&self.0.pages, ids)
    }

    /// Resolve slot ids; the permanent zero slot (index 0) always resolves.
    pub(crate) fn resolve_slots(&self, ids: &[Id]) -> Result<Vec<u32>> {
        resolve(&self.0.slots, ids)
    }
}

fn resolve(allocator: &Mutex<Allocator>, ids: &[Id]) -> Result<Vec<u32>> {
    let allocator = allocator.lock();
    ids.iter()
        .map(|&id| {
            if allocator.live(id) {
                Ok(id.index)
            } else {
                Err(invalid(format!(
                    "stale state id {} (generation {})",
                    id.index, id.generation
                )))
            }
        })
        .collect()
}

/// Reference quantizer for `[tokens, kv_heads, head_dim]` F32 rows: stored codes and, for
/// Q8, F16 scales `[tokens, kv_heads, head_dim / group]`. The kernels apply the same
/// arithmetic: `scale = f16(max|x| · (1/127))`, `code = clamp(round(x / scale), ±127)`,
/// stored as a two's-complement byte in U8 (Candle has no signed 8-bit dtype).
pub fn quantize(format: KvFormat, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    match format {
        KvFormat::F32 => Ok((x, None)),
        KvFormat::F16 => Ok((x.to_dtype(DType::F16)?, None)),
        KvFormat::Q8 { group } => {
            let (n, heads, dim) = x.dims3()?;
            format.validate(dim)?;
            let grouped = x.reshape((n, heads, dim / group, group))?;
            let scales = (grouped.abs()?.max_keepdim(3)? / 127.0)?.to_dtype(DType::F16)?;
            // A scale that rounds to zero stores zero codes: the group is below F16 range.
            let divisor = scales.to_dtype(DType::F32)?.maximum(f32::MIN_POSITIVE)?;
            let codes = grouped
                .broadcast_div(&divisor)?
                .round()?
                .clamp(-127f32, 127f32)?;
            let codes = codes
                .lt(0f32)?
                .where_cond(&(&codes + 256.0)?, &codes)?
                .to_dtype(DType::U8)?
                .reshape((n, heads, dim))?;
            Ok((codes, Some(scales.reshape((n, heads, dim / group))?)))
        }
    }
}

/// Inverse of [`quantize`]: F32 `[tokens, kv_heads, head_dim]`.
pub fn dequantize(format: KvFormat, data: &Tensor, scales: Option<&Tensor>) -> Result<Tensor> {
    match (format, scales) {
        (KvFormat::F32, None) => Ok(data.clone()),
        (KvFormat::F16, None) => Ok(data.to_dtype(DType::F32)?),
        (KvFormat::Q8 { group }, Some(scales)) => {
            let (n, heads, dim) = data.dims3()?;
            let unsigned = data.to_dtype(DType::F32)?;
            let codes = unsigned
                .ge(128f32)?
                .where_cond(&(&unsigned - 256.0)?, &unsigned)?;
            Ok(codes
                .reshape((n, heads, dim / group, group))?
                .broadcast_mul(&scales.to_dtype(DType::F32)?.unsqueeze(3)?)?
                .reshape((n, heads, dim))?)
        }
        _ => Err(invalid("K/V scales do not match the format")),
    }
}

/// Contiguous runs `(first slot, first token, tokens)` covering tokens `start..end` of a
/// sequence whose page table is `pages`. Adjacent physical pages merge into one run.
fn runs(
    pages: &[u32],
    page: usize,
    start: usize,
    end: usize,
) -> Result<Vec<(usize, usize, usize)>> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut token = start;
    while token < end {
        let physical = *pages
            .get(token / page)
            .ok_or_else(|| invalid("token outside its page table"))?
            as usize;
        let slot = physical * page + token % page;
        let len = (page - token % page).min(end - token);
        match runs.last_mut() {
            Some((first, _, count)) if *first + *count == slot => *count += len,
            _ => runs.push((slot, token, len)),
        }
        token += len;
    }
    Ok(runs)
}

impl KvPool {
    fn gather(&self, runs: &[(usize, usize, usize)]) -> Result<(Tensor, Option<Tensor>)> {
        let take = |tensor: &Tensor| -> Result<Tensor> {
            let pieces = runs
                .iter()
                .map(|&(slot, _, len)| tensor.narrow(0, slot, len))
                .collect::<candle_core::Result<Vec<_>>>()?;
            Ok(Tensor::cat(&pieces, 0)?)
        };
        Ok((
            take(&self.data)?,
            self.scales.as_ref().map(take).transpose()?,
        ))
    }

    fn scatter(
        &self,
        runs: &[(usize, usize, usize)],
        start: usize,
        x: &(Tensor, Option<Tensor>),
    ) -> Result<()> {
        for &(slot, token, len) in runs {
            let offset = token - start;
            self.data
                .slice_set(&x.0.narrow(0, offset, len)?.contiguous()?, 0, slot)?;
            if let (Some(pool), Some(scales)) = (&self.scales, &x.1) {
                pool.slice_set(&scales.narrow(0, offset, len)?.contiguous()?, 0, slot)?;
            }
        }
        Ok(())
    }
}

impl AttentionPool {
    /// Reference read: F32 keys and values of tokens `0..count` of a sequence with page
    /// table `pages`, dequantized. Development and test path; kernels read pages in place.
    pub fn gather(&self, pages: &[u32], count: usize) -> Result<(Tensor, Tensor)> {
        if count == 0 {
            let empty = || {
                Tensor::zeros(
                    (0, self.kv_heads, self.head_dim),
                    DType::F32,
                    self.key.data.device(),
                )
            };
            return Ok((empty()?, empty()?));
        }
        let runs = runs(pages, self.page_tokens, 0, count)?;
        let read = |pool: &KvPool| -> Result<Tensor> {
            let (data, scales) = pool.gather(&runs)?;
            dequantize(self.format, &data, scales.as_ref())
        };
        Ok((read(&self.key)?, read(&self.value)?))
    }

    /// Reference write: quantize `[n, kv_heads, head_dim]` keys and values and store them as
    /// tokens `start..start + n` of the sequence with page table `pages`.
    pub fn scatter(&self, pages: &[u32], start: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        let n = key.dim(0)?;
        let shape = [n, self.kv_heads, self.head_dim];
        if key.dims() != shape || value.dims() != shape {
            return Err(invalid("K/V rows do not match the attention pool"));
        }
        let runs = runs(pages, self.page_tokens, start, start + n)?;
        self.key
            .scatter(&runs, start, &quantize(self.format, key)?)?;
        self.value
            .scatter(&runs, start, &quantize(self.format, value)?)
    }
}

impl RecurrentPool {
    /// Zero-copy `[conv_width, conv_channels]` and `[heads, key_dim, value_dim]` views of
    /// `slot`. Callers must not write through them.
    pub fn read(&self, slot: u32) -> Result<(Tensor, Tensor)> {
        Ok((
            self.convolution.get(slot as usize)?,
            self.state.get(slot as usize)?,
        ))
    }

    /// Reference write of boundary `b` of `[boundaries, ...]` outputs into `slot`.
    pub fn write(&self, slot: u32, convolution: &Tensor, state: &Tensor, b: usize) -> Result<()> {
        let slot = slot as usize;
        self.convolution
            .slice_set(&convolution.narrow(0, b, 1)?.contiguous()?, 0, slot)?;
        self.state
            .slice_set(&state.narrow(0, b, 1)?.contiguous()?, 0, slot)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tur_core::execution::{KvLayout, Modality, ModelIdentity};

    pub(crate) fn description(page_tokens: usize, format: KvFormat) -> Arc<ModelDescription> {
        Arc::new(ModelDescription {
            identity: ModelIdentity {
                revision: "pool-fixture".into(),
                state_format: 1,
            },
            layers: vec![
                LayerStateKind::Attention {
                    kv_heads: 2,
                    head_dim: 64,
                },
                LayerStateKind::Recurrent {
                    conv_channels: 3,
                    conv_width: 2,
                    heads: 1,
                    key_dim: 2,
                    value_dim: 2,
                    dtype: DType::F32,
                },
            ],
            kv: KvLayout::new(page_tokens, format),
            modalities: vec![Modality::Text],
            vocab_size: 8,
            max_position: 64,
        })
    }

    fn rows(n: usize, seed: u64) -> Result<Tensor> {
        let values: Vec<f32> = (0..n * 128)
            .map(|i| (((i as u64 * 2654435761 + seed * 97) % 2001) as f32 / 1000.0 - 1.0) * 3.0)
            .collect();
        Ok(Tensor::from_vec(values, (n, 2, 64), &Device::Cpu)?)
    }

    #[test]
    fn test_budget_splits_slots_then_pages_and_rejects_starved_pools() -> Result<()> {
        let model = description(4, KvFormat::F32);
        let page = model.page_bytes()?;
        let slot = model.recurrent_slot_bytes()?;
        let config = PoolConfig::for_budget(&model, 3 * page + 4 * slot + 7, 2)?;
        assert_eq!(
            config,
            PoolConfig {
                pages: 3,
                recurrent_slots: 4
            }
        );
        assert_eq!(config.bytes(&model)?, 3 * page + 5 * slot);
        assert!(PoolConfig::for_budget(&model, 4 * slot, 2).is_err());
        assert!(matches!(
            PoolConfig::for_budget(&model, slot, 2),
            Err(TurError::MemoryAllocation(_))
        ));
        Ok(())
    }

    #[test]
    fn test_leases_retire_on_drop_and_reclaim_bumps_generations() -> Result<()> {
        let pool = StatePool::new(
            description(4, KvFormat::F32),
            &Device::Cpu,
            PoolConfig {
                pages: 2,
                recurrent_slots: 1,
            },
        )?;
        let a = pool.lease_page()?;
        let b = pool.lease_page()?;
        assert_eq!((a.id().index, b.id().index), (0, 1));
        assert!(matches!(
            pool.lease_page(),
            Err(TurError::MemoryAllocation(_))
        ));
        let stale = a.id();
        drop(a);
        let usage = pool.usage()?.pages;
        assert_eq!(
            (usage.leased, usage.retired, usage.free(), usage.peak),
            (1, 1, 0, 2)
        );
        // Retired ids still resolve to nothing new until reclaimed, and never to a live lease.
        assert!(pool.resolve_pages(&[stale]).is_err());
        assert!(pool.lease_page().is_err());
        assert_eq!(pool.reclaim(), 1);
        let c = pool.lease_page()?;
        assert_eq!(c.id().index, stale.index);
        assert_ne!(c.id().generation, stale.generation);
        assert!(pool.resolve_pages(&[stale]).is_err());
        assert_eq!(pool.resolve_pages(&[c.id(), b.id()])?, vec![0, 1]);
        // The zero slot is permanent: one usable slot exists besides it.
        let slot = pool.lease_slot()?;
        assert_eq!(slot.id().index, 1);
        assert!(pool.lease_slot().is_err());
        assert_eq!(
            pool.resolve_slots(&[Id {
                index: 0,
                generation: 0
            }])?,
            vec![0]
        );
        Ok(())
    }

    #[test]
    fn test_q8_roundtrip_is_within_half_a_step_and_f16_is_exact_to_half() -> Result<()> {
        let x = rows(5, 1)?;
        let values = x.flatten_all()?.to_vec1::<f32>()?;
        for group in [32, 64] {
            let format = KvFormat::Q8 { group };
            let (codes, scales) = quantize(format, &x)?;
            let scales = scales.ok_or_else(|| invalid("missing scales"))?;
            assert_eq!(codes.dtype(), DType::U8);
            assert_eq!(scales.dims(), [5, 2, 64 / group]);
            let restored = dequantize(format, &codes, Some(&scales))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let steps = scales
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (i, (a, b)) in values.iter().zip(&restored).enumerate() {
                let step = steps[i / group];
                assert!(
                    (a - b).abs() <= 0.5 * step * (1.0 + 1e-3),
                    "group {group} element {i}"
                );
            }
        }
        let (half, _) = quantize(KvFormat::F16, &x)?;
        assert_eq!(half.dtype(), DType::F16);
        let restored = dequantize(KvFormat::F16, &half, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (a, b) in values.iter().zip(&restored) {
            assert!((a - b).abs() <= a.abs() * 2f32.powi(-11));
        }
        // A zero group stores zero codes and a zero scale rather than NaN.
        let zeros = Tensor::zeros((1, 2, 64), DType::F32, &Device::Cpu)?;
        let (codes, scales) = quantize(KvFormat::Q8, &zeros)?;
        let restored = dequantize(KvFormat::Q8, &codes, scales.as_ref())?;
        assert!(
            restored
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| *v == 0.0)
        );
        Ok(())
    }

    #[test]
    fn test_scatter_and_gather_follow_the_page_table_for_every_format() -> Result<()> {
        for format in [KvFormat::F32, KvFormat::F16, KvFormat::Q8] {
            let pool = StatePool::new(
                description(4, format),
                &Device::Cpu,
                PoolConfig {
                    pages: 5,
                    recurrent_slots: 1,
                },
            )?;
            let attention = pool.attention(0)?;
            // Non-adjacent, reversed physical pages; tokens 3..9 straddle three pages.
            let pages = [4u32, 1, 2];
            let (k, v) = (rows(9, 2)?, rows(9, 3)?);
            attention.scatter(&pages, 0, &k.narrow(0, 0, 3)?, &v.narrow(0, 0, 3)?)?;
            attention.scatter(&pages, 3, &k.narrow(0, 3, 6)?, &v.narrow(0, 3, 6)?)?;
            let (gk, gv) = attention.gather(&pages, 9)?;
            let (qk, sk) = quantize(format, &k)?;
            let (qv, sv) = quantize(format, &v)?;
            let expect_k = dequantize(format, &qk, sk.as_ref())?;
            let expect_v = dequantize(format, &qv, sv.as_ref())?;
            let flat = |t: &Tensor| t.flatten_all().and_then(|t| t.to_vec1::<f32>());
            assert_eq!(flat(&gk)?, flat(&expect_k)?, "{format}");
            assert_eq!(flat(&gv)?, flat(&expect_v)?, "{format}");
            // Token 4 lives in physical page 1, slot 1 * 4 + 0.
            let raw = attention.key.data.get(4)?.to_dtype(DType::F32)?;
            assert_eq!(flat(&raw)?, flat(&qk.get(4)?.to_dtype(DType::F32)?)?);
            assert!(attention.gather(&pages, 13).is_err());
        }
        Ok(())
    }
}
