#![forbid(unsafe_code)]
//! Immutable state snapshots over persistent pools, with reservations for one pass.
//!
//! A [`ModelState`] is a page table, a window page table and one recurrent slot at an
//! exact token boundary. Cloning shares every lease. A pass reserves the pages it opens and
//! fresh recurrent slots; the model writes new keys, values and recurrent states straight
//! into them, and [`Reservation::commit`] publishes the accepted boundary without copying
//! anything.
//!
//! Tokens visible to some snapshot are never rewritten. A sequence appends into its
//! partially filled tail page in place only when no other snapshot holds that page;
//! otherwise the tail's valid prefix is copied to a fresh page first (copy-on-write).
//! Recurrent state is never updated in place: every pass writes new slots, so rollback is
//! selecting an older boundary and no snapshot is ever truncated.
//!
//! Windowed attention layers store their keys in window pages, indexed by the same
//! logical page numbers. A committed state releases the window pages wholly before its
//! window ([`ModelDescription::first_window_page`]); released entries address a
//! placeholder page that no query reads.

use crate::pool::{Id, Lease, PoolUsage, StatePool};
use candle_core::{DType, Tensor};
use std::sync::Arc;
use tur_core::{
    Result,
    execution::{LayerStateKind, ModelDescription, invalid},
};

/// Recurrent states a model must write for one sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundaries {
    /// Only the state after the last token.
    Final,
    /// The state after every token, so any accepted prefix can be published.
    EveryToken,
}

/// Recurrent slot of a snapshot: the shared zero state, or a leased slot.
#[derive(Clone)]
enum Slot {
    Zero,
    Leased(Arc<Lease>),
}

impl Slot {
    fn id(&self) -> Id {
        match self {
            Self::Zero => Id {
                index: 0,
                generation: 0,
            },
            Self::Leased(lease) => lease.id(),
        }
    }
}

/// A window page table: logical page `i` is leased, or released behind the window.
type Windows = Vec<Option<Arc<Lease>>>;

/// An immutable, whole-model state at exactly one token boundary.
#[derive(Clone)]
pub struct ModelState {
    pool: StatePool,
    position: usize,
    pages: Vec<Arc<Lease>>,
    windows: Windows,
    slot: Slot,
}

/// Pool ids a pass would lease.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demand {
    pub pages: usize,
    pub windows: usize,
    pub slots: usize,
}

impl std::ops::Add for Demand {
    type Output = Self;

    /// Combined demand of two passes.
    fn add(self, other: Self) -> Self {
        Self {
            pages: self.pages + other.pages,
            windows: self.windows + other.windows,
            slots: self.slots + other.slots,
        }
    }
}

impl std::ops::Sub for Demand {
    type Output = Self;

    /// Demand left after removing `other`, which it contains.
    fn sub(self, other: Self) -> Self {
        Self {
            pages: self.pages.saturating_sub(other.pages),
            windows: self.windows.saturating_sub(other.windows),
            slots: self.slots.saturating_sub(other.slots),
        }
    }
}

impl Demand {
    /// Whether the free ids in `usage` cover this demand.
    pub fn fits(&self, usage: &PoolUsage) -> bool {
        self.pages <= usage.pages.free()
            && self.windows <= usage.windows.free()
            && self.slots <= usage.slots.free()
    }

    /// Bytes this demand leases.
    pub fn bytes(&self, usage: &PoolUsage) -> usize {
        self.pages * usage.page_bytes
            + self.windows * usage.window_bytes
            + self.slots * usage.slot_bytes
    }
}

/// Detached F32 copy of one layer's committed state, for inspection.
pub enum LayerSnapshot {
    /// `[position, kv_heads, head_dim]` keys and values, dequantized. A windowed layer's
    /// released keys read as zeros.
    Attention { key: Tensor, value: Tensor },
    /// `[conv_width, conv_channels]` history and `[heads, key_dim, value_dim]` matrix.
    Recurrent {
        convolution: Tensor,
        recurrent: Tensor,
    },
}

/// Whether appending to a table whose tail is `tail` must first copy that shared tail.
fn shared_tail(tail: Option<&Arc<Lease>>, position: usize, page: usize, tokens: usize) -> bool {
    tokens > 0 && !position.is_multiple_of(page) && tail.is_some_and(|t| Arc::strong_count(t) > 1)
}

impl ModelState {
    /// An empty sequence: no pages and the shared zero recurrent state. Allocates nothing.
    pub fn new(pool: &StatePool) -> Self {
        Self {
            pool: pool.clone(),
            position: 0,
            pages: Vec::new(),
            windows: Vec::new(),
            slot: Slot::Zero,
        }
    }

    /// Pool holding this state.
    pub fn pool(&self) -> &StatePool {
        &self.pool
    }

    /// Exact token boundary of every layer in this snapshot.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Complete identity and layout used to validate restore.
    pub fn description(&self) -> &ModelDescription {
        self.pool.description()
    }

    /// Full-attention pages this snapshot holds.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Window pages this snapshot still holds.
    pub fn window_count(&self) -> usize {
        self.windows.iter().flatten().count()
    }

    /// Pages, window pages and slot that only this snapshot holds: what dropping it would
    /// release.
    pub fn exclusive(&self) -> Demand {
        Demand {
            pages: self
                .pages
                .iter()
                .filter(|page| Arc::strong_count(page) == 1)
                .count(),
            windows: self
                .windows
                .iter()
                .flatten()
                .filter(|page| Arc::strong_count(page) == 1)
                .count(),
            slots: usize::from(
                matches!(&self.slot, Slot::Leased(slot) if Arc::strong_count(slot) == 1),
            ),
        }
    }

    /// Whether appending must first copy a shared, partially filled tail page, of the
    /// page table and of the window table.
    fn copies_tail(&self, tokens: usize) -> (bool, bool) {
        let page = self.description().kv.page_tokens;
        (
            shared_tail(self.pages.last(), self.position, page, tokens),
            shared_tail(
                self.windows.last().and_then(Option::as_ref),
                self.position,
                page,
                tokens,
            ),
        )
    }

    fn end(&self, tokens: usize) -> Result<usize> {
        let end = self
            .position
            .checked_add(tokens)
            .ok_or_else(|| invalid("position overflow"))?;
        if tokens == 0 || end > self.description().max_position {
            return Err(invalid("invalid pass length or position"));
        }
        Ok(end)
    }

    /// Pages, window pages and slots a pass of `tokens` would lease, including
    /// copy-on-write tails.
    pub fn demand(&self, tokens: usize, boundaries: Boundaries) -> Result<Demand> {
        let description = self.description();
        let end = self.end(tokens)?;
        let opened = description.pages_for(end) - description.pages_for(self.position);
        let (page_tail, window_tail) = self.copies_tail(tokens);
        let pages = if description.has_full_attention() {
            opened + usize::from(page_tail)
        } else {
            0
        };
        let windows = if description.window().is_some() {
            opened + usize::from(window_tail)
        } else {
            0
        };
        let slots = match (description.has_recurrent(), boundaries) {
            (false, _) => 0,
            (true, Boundaries::Final) => 1,
            (true, Boundaries::EveryToken) => tokens,
        };
        Ok(Demand {
            pages,
            windows,
            slots,
        })
    }

    /// Lease the pages and recurrent slots for `tokens` new tokens. Nothing is written
    /// until [`Reservation::prepare`]; dropping the reservation releases everything.
    pub fn reserve(&self, tokens: usize, boundaries: Boundaries) -> Result<Reservation> {
        let description = self.description();
        let end = self.end(tokens)?;
        let page = description.kv.page_tokens;
        let logical = description.pages_for(end);
        // Decide before cloning the tables, which raises every page's reference count.
        let (page_tail, window_tail) = self.copies_tail(tokens);
        let mut pages = self.pages.clone();
        let mut copy = None;
        if description.has_full_attention() {
            if page_tail {
                let fresh = self.pool.lease_page()?;
                if let Some(tail) = pages.last_mut() {
                    copy = Some((tail.id(), fresh.id(), self.position % page));
                    *tail = fresh;
                }
            }
            while pages.len() < logical {
                pages.push(self.pool.lease_page()?);
            }
        }
        let mut windows = self.windows.clone();
        let mut window_copy = None;
        if description.window().is_some() {
            if window_tail {
                let fresh = self.pool.lease_window()?;
                if let Some(Some(tail)) = windows.last_mut() {
                    window_copy = Some((tail.id(), fresh.id(), self.position % page));
                    *tail = fresh;
                }
            }
            while windows.len() < logical {
                windows.push(Some(self.pool.lease_window()?));
            }
        }
        let count = match (description.has_recurrent(), boundaries) {
            (false, _) => 0,
            (true, Boundaries::Final) => 1,
            (true, Boundaries::EveryToken) => tokens,
        };
        let slots = (0..count)
            .map(|_| self.pool.lease_slot())
            .collect::<Result<Vec<_>>>()?;
        Ok(Reservation {
            base: self.clone(),
            new_pages: pages.len() - self.pages.len() + usize::from(copy.is_some()),
            new_windows: windows.len() - self.windows.len() + usize::from(window_copy.is_some()),
            pages,
            windows,
            slots,
            copy,
            window_copy,
            tokens,
            boundaries,
        })
    }

    /// Addressing of committed positions `start..start + tokens` for a side pass, such as a
    /// draft head, that writes only layers no ordinary pass writes. It leases nothing and
    /// writes no recurrent slot. The pages must be exclusively owned: a page shared with a
    /// snapshot would see the side layer change under it.
    pub fn side(&self, start: usize, tokens: usize) -> Result<SequenceState> {
        let end = start
            .checked_add(tokens)
            .ok_or_else(|| invalid("position overflow"))?;
        if tokens == 0 || end > self.position {
            return Err(invalid("side pass outside committed positions"));
        }
        let description = self.description();
        let first = start / description.kv.page_tokens;
        if description.has_full_attention()
            && self
                .pages
                .get(first..description.pages_for(end))
                .is_none_or(|pages| pages.iter().any(|page| Arc::strong_count(page) > 1))
        {
            return Err(invalid("side pass needs exclusively owned pages"));
        }
        Ok(SequenceState {
            pool: self.pool.id(),
            past: start,
            tokens,
            page_tokens: description.kv.page_tokens,
            pages: ids(&self.pages),
            windows: window_ids(&self.windows),
            read: self.slot.id(),
            writes: Vec::new(),
            boundaries: Boundaries::Final,
        })
    }

    /// Detached F32 copies of every layer for inspection. Waits for the copies to finish,
    /// so no unfenced work references these pages if the snapshot is released afterwards.
    pub fn read_layers(&self) -> Result<Vec<LayerSnapshot>> {
        let pages = self.pool.resolve_pages(&ids(&self.pages))?;
        let windows = self.pool.resolve_windows(&window_ids(&self.windows))?;
        let slot = self.pool.resolve_slots(&[self.slot.id()])?;
        let slot = slot.first().copied().unwrap_or(0);
        let mut layers = Vec::with_capacity(self.description().layers.len());
        for (index, kind) in self.description().layers.iter().enumerate() {
            layers.push(match kind {
                LayerStateKind::Attention { window, .. } => {
                    let table = if window.is_some() { &windows } else { &pages };
                    let (key, value) = self.pool.attention(index)?.gather(table, self.position)?;
                    LayerSnapshot::Attention {
                        key: key.force_contiguous()?,
                        value: value.force_contiguous()?,
                    }
                }
                LayerStateKind::Recurrent { .. } => {
                    let (convolution, recurrent) = self.pool.recurrent(index)?.read(slot)?;
                    LayerSnapshot::Recurrent {
                        convolution: convolution.to_dtype(DType::F32)?.force_contiguous()?,
                        recurrent: recurrent.to_dtype(DType::F32)?.force_contiguous()?,
                    }
                }
            });
        }
        self.pool.device().synchronize()?;
        Ok(layers)
    }
}

fn ids(leases: &[Arc<Lease>]) -> Vec<Id> {
    leases.iter().map(|lease| lease.id()).collect()
}

fn window_ids(windows: &[Option<Arc<Lease>>]) -> Vec<Option<Id>> {
    windows
        .iter()
        .map(|lease| lease.as_ref().map(|l| l.id()))
        .collect()
}

/// Pool memory leased for one pass of one sequence. Holding it keeps every page and slot
/// the pass reads or writes alive; dropping it before commit releases the new ones.
pub struct Reservation {
    base: ModelState,
    /// Page table after the pass: committed pages (tail possibly replaced) plus new ones.
    pages: Vec<Arc<Lease>>,
    /// Window page table after the pass, likewise.
    windows: Windows,
    /// Recurrent slots written by the pass: one per boundary.
    slots: Vec<Arc<Lease>>,
    /// Pending copy-on-write of a shared tail: `(from, to, valid tokens)`.
    copy: Option<(Id, Id, usize)>,
    window_copy: Option<(Id, Id, usize)>,
    /// Pages leased by this reservation: opened pages plus a copied tail.
    new_pages: usize,
    new_windows: usize,
    tokens: usize,
    boundaries: Boundaries,
}

impl Reservation {
    /// The committed state this pass starts from.
    pub fn base(&self) -> &ModelState {
        &self.base
    }

    /// Bytes newly leased: opened pages and window pages, copied tails and recurrent slots.
    pub fn bytes(&self) -> Result<usize> {
        let description = self.base.description();
        let slots = self.slots.len();
        self.new_pages
            .checked_mul(description.page_bytes()?)
            .zip(
                self.new_windows
                    .checked_mul(description.window_page_bytes()?),
            )
            .zip(slots.checked_mul(if slots > 0 {
                description.recurrent_slot_bytes()?
            } else {
                0
            }))
            .and_then(|((p, w), s)| p.checked_add(w)?.checked_add(s))
            .ok_or_else(|| invalid("reservation size overflow"))
    }

    /// Enqueue the copy-on-write of shared tail pages in every attention layer. Call after
    /// every reservation of a batch succeeded, before the model reads the new tables.
    pub fn prepare(&mut self) -> Result<()> {
        let pool = &self.base.pool;
        let page = pool.description().kv.page_tokens;
        for (windowed, copy) in [(false, self.copy.take()), (true, self.window_copy.take())] {
            let Some((from, to, count)) = copy else {
                continue;
            };
            let resolved = if windowed {
                pool.resolve_windows(&[Some(from), Some(to)])?
            } else {
                pool.resolve_pages(&[from, to])?
            };
            let &[from, to] = resolved.as_slice() else {
                return Err(invalid("copy-on-write pages did not resolve"));
            };
            for (index, kind) in pool.description().layers.iter().enumerate() {
                if !matches!(kind, LayerStateKind::Attention { .. }) || kind.windowed() != windowed
                {
                    continue;
                }
                let layer = pool.attention(index)?;
                for kv in [&layer.key, &layer.value] {
                    for tensor in std::iter::once(&kv.data).chain(&kv.scales) {
                        // Copy through a temporary: a tensor cannot be both source and target.
                        // `Tensor::copy` shares storage on Metal; this allocates.
                        let source = tensor
                            .narrow(0, from as usize * page, count)?
                            .force_contiguous()?;
                        tensor.slice_set(&source, 0, to as usize * page)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Generation-tagged addressing for the model's pass.
    pub fn binding(&self) -> SequenceState {
        SequenceState {
            pool: self.base.pool.id(),
            past: self.base.position,
            tokens: self.tokens,
            page_tokens: self.base.description().kv.page_tokens,
            pages: ids(&self.pages),
            windows: window_ids(&self.windows),
            read: self.base.slot.id(),
            writes: ids(&self.slots),
            boundaries: self.boundaries,
        }
    }

    /// The snapshot after the first `accepted` tokens of this pass; the base is unchanged.
    /// A partial accept requires [`Boundaries::EveryToken`], whose boundary is selected
    /// exactly: recurrent state is never truncated. Window pages wholly behind the new
    /// boundary's window are released.
    pub fn commit(&self, accepted: usize) -> Result<ModelState> {
        if accepted == 0 {
            return Ok(self.base.clone());
        }
        if self.copy.is_some() || self.window_copy.is_some() {
            return Err(invalid("reservation was never prepared"));
        }
        let partial = accepted < self.tokens;
        if accepted > self.tokens || (partial && self.boundaries == Boundaries::Final) {
            return Err(invalid("accepted prefix has no saved boundary"));
        }
        let position = self
            .base
            .position
            .checked_add(accepted)
            .ok_or_else(|| invalid("position overflow"))?;
        let description = self.base.description();
        let keep = description.pages_for(position);
        let slot = match self.boundaries {
            Boundaries::Final => self.slots.first(),
            Boundaries::EveryToken => self.slots.get(accepted - 1),
        };
        let first = description.first_window_page(position);
        let windows = self
            .windows
            .iter()
            .take(keep)
            .enumerate()
            .map(|(i, page)| page.as_ref().filter(|_| i >= first).cloned())
            .collect();
        Ok(ModelState {
            pool: self.base.pool.clone(),
            position,
            pages: self.pages.get(..keep).unwrap_or(&self.pages).to_vec(),
            windows,
            slot: slot.map_or(Slot::Zero, |s| Slot::Leased(Arc::clone(s))),
        })
    }
}

/// Physical addressing of one sequence's pass. Ids are generation tagged: once the
/// reservation that produced them is released and reclaimed, they no longer resolve.
#[derive(Clone, Debug)]
pub struct SequenceState {
    pool: u64,
    past: usize,
    tokens: usize,
    page_tokens: usize,
    pages: Vec<Id>,
    windows: Vec<Option<Id>>,
    read: Id,
    writes: Vec<Id>,
    boundaries: Boundaries,
}

/// Resolved indices of a [`SequenceState`] in its pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Addresses {
    /// Physical page of each logical page, covering `past + tokens` tokens.
    pub pages: Vec<u32>,
    /// Physical window page of each logical page; released pages address the placeholder.
    pub windows: Vec<u32>,
    /// Recurrent slot holding the committed state.
    pub read: u32,
    /// Recurrent slots receiving the state after the last token, or after every token.
    pub writes: Vec<u32>,
}

impl SequenceState {
    /// Committed tokens before this pass.
    pub fn past(&self) -> usize {
        self.past
    }

    /// New tokens in this pass.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Recurrent boundaries the pass writes.
    pub fn boundaries(&self) -> Boundaries {
        self.boundaries
    }

    /// Page capacity in tokens.
    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    /// Resolve every id against `pool`, failing on a foreign pool or a stale id.
    pub fn resolve(&self, pool: &StatePool) -> Result<Addresses> {
        if pool.id() != self.pool {
            return Err(invalid("sequence state belongs to another pool"));
        }
        let read = pool.resolve_slots(&[self.read])?;
        Ok(Addresses {
            pages: pool.resolve_pages(&self.pages)?,
            windows: pool.resolve_windows(&self.windows)?,
            read: read.first().copied().unwrap_or(0),
            writes: pool.resolve_slots(&self.writes)?,
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use crate::pool::PoolConfig;
    use candle_core::Device;
    use tur_core::{
        TurError,
        execution::{KvFormat, KvLayout, Modality, ModelIdentity},
    };

    fn pool(pages: usize, slots: usize) -> Result<StatePool> {
        StatePool::new(
            Arc::new(ModelDescription {
                identity: ModelIdentity {
                    revision: "state-fixture".into(),
                    state_format: 1,
                },
                layers: vec![
                    LayerStateKind::Attention {
                        kv_heads: 1,
                        head_dim: 32,
                        window: None,
                    },
                    LayerStateKind::Recurrent {
                        conv_channels: 1,
                        conv_width: 1,
                        heads: 1,
                        key_dim: 1,
                        value_dim: 1,
                        dtype: DType::F32,
                    },
                ],
                kv: KvLayout::new(2, KvFormat::Q8),
                modalities: vec![Modality::Text],
                vocab_size: 8,
                max_position: 16,
            }),
            &Device::Cpu,
            PoolConfig {
                pages,
                window_pages: pages,
                recurrent_slots: slots,
            },
        )
    }

    /// Run one pass that writes token values `values` and recurrent boundary values.
    fn pass(
        state: &ModelState,
        values: &[f32],
        boundaries: Boundaries,
        accepted: usize,
    ) -> Result<ModelState> {
        let mut reservation = state.reserve(values.len(), boundaries)?;
        reservation.prepare()?;
        let addresses = reservation.binding().resolve(state.pool())?;
        let n = values.len();
        let rows: Vec<f32> = values.iter().flat_map(|v| [*v; 32]).collect();
        let rows = Tensor::from_vec(rows, (n, 1, 32), &Device::Cpu)?;
        state.pool().attention(0)?.scatter(
            &addresses.pages,
            state.position(),
            &rows,
            &(rows.clone() * 2.0)?,
        )?;
        let recurrent = state.pool().recurrent(1)?;
        for (b, &slot) in addresses.writes.iter().enumerate() {
            let value = values[if addresses.writes.len() == 1 {
                n - 1
            } else {
                b
            }];
            let t = Tensor::full(value, (1, 1, 1), &Device::Cpu)?;
            recurrent.write(slot, &t, &t.unsqueeze(3)?, 0)?;
        }
        reservation.commit(accepted)
    }

    fn keys(state: &ModelState) -> Result<(Vec<f32>, f32)> {
        let mut keys = Vec::new();
        let mut recurrent = f32::NAN;
        for layer in state.read_layers()? {
            match layer {
                LayerSnapshot::Attention { key, .. } => {
                    keys = key.mean(2)?.flatten_all()?.to_vec1::<f32>()?;
                }
                LayerSnapshot::Recurrent { recurrent: r, .. } => {
                    recurrent = r.flatten_all()?.to_vec1::<f32>()?[0];
                }
            }
        }
        Ok((keys, recurrent))
    }

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 0.02, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn test_appends_fill_the_tail_page_in_place_when_exclusive() -> Result<()> {
        let pool = pool(4, 2)?;
        let one = pass(&ModelState::new(&pool), &[1.0], Boundaries::Final, 1)?;
        assert_eq!(
            one.demand(1, Boundaries::Final)?,
            Demand {
                pages: 0,
                windows: 0,
                slots: 1
            }
        );
        let three = pass(&one, &[2.0, 3.0], Boundaries::Final, 2)?;
        drop(one);
        assert_eq!(three.page_count(), 2);
        assert_eq!(pool.usage()?.pages.leased, 2);
        let (k, r) = keys(&three)?;
        close(&k, &[1.0, 2.0, 3.0]);
        assert_eq!(r, 3.0);
        // The superseded recurrent slot is retired, not reused before reclaim.
        assert_eq!(pool.usage()?.slots.retired, 1);
        Ok(())
    }

    #[test]
    fn test_shared_tail_is_copied_before_append_and_snapshots_stay_exact() -> Result<()> {
        let pool = pool(6, 4)?;
        let saved = pass(
            &ModelState::new(&pool),
            &[1.0, 2.0, 3.0],
            Boundaries::Final,
            3,
        )?;
        // Two forks of a mid-page boundary each copy the shared tail.
        let fork = saved.clone();
        assert_eq!(
            saved.demand(1, Boundaries::Final)?,
            Demand {
                pages: 1,
                windows: 0,
                slots: 1
            }
        );
        let a = pass(&saved, &[4.0], Boundaries::Final, 1)?;
        let b = pass(&fork, &[9.0, 8.0], Boundaries::Final, 2)?;
        close(&keys(&saved)?.0, &[1.0, 2.0, 3.0]);
        close(&keys(&a)?.0, &[1.0, 2.0, 3.0, 4.0]);
        close(&keys(&b)?.0, &[1.0, 2.0, 3.0, 9.0, 8.0]);
        assert_eq!(keys(&a)?.1, 4.0);
        assert_eq!(keys(&saved)?.1, 3.0);
        // Full pages stay shared: 2 committed + 2 copied tails + 1 page opened by b.
        assert_eq!(pool.usage()?.pages.leased, 5);
        Ok(())
    }

    #[test]
    fn test_partial_accept_selects_the_exact_boundary_and_releases_the_rest() -> Result<()> {
        let pool = pool(4, 4)?;
        let base = ModelState::new(&pool);
        let two = pass(&base, &[1.0, 2.0, 3.0], Boundaries::EveryToken, 2)?;
        assert_eq!(two.position(), 2);
        let (k, r) = keys(&two)?;
        close(&k, &[1.0, 2.0]);
        assert_eq!(r, 2.0);
        assert_eq!(two.page_count(), 1);
        let usage = pool.usage()?;
        assert_eq!((usage.pages.leased, usage.slots.leased), (1, 1));
        assert_eq!(pass(&base, &[5.0], Boundaries::Final, 0)?.position(), 0);
        pool.reclaim();
        let mut reservation = base.reserve(2, Boundaries::Final)?;
        reservation.prepare()?;
        assert!(reservation.commit(1).is_err());
        assert!(reservation.commit(3).is_err());
        Ok(())
    }

    #[test]
    fn test_exhaustion_is_an_explicit_error_and_reclaim_restores_capacity() -> Result<()> {
        let pool = pool(2, 2)?;
        let full = pass(
            &ModelState::new(&pool),
            &[1.0, 2.0, 3.0, 4.0],
            Boundaries::Final,
            4,
        )?;
        assert!(matches!(
            full.reserve(1, Boundaries::Final),
            Err(TurError::MemoryAllocation(_))
        ));
        assert_eq!(full.demand(1, Boundaries::Final)?.pages, 1);
        drop(full);
        let usage = pool.usage()?;
        assert_eq!((usage.pages.leased, usage.pages.retired), (0, 2));
        assert!(
            !Demand {
                pages: 1,
                windows: 0,
                slots: 0
            }
            .fits(&usage)
        );
        assert_eq!(pool.reclaim(), 3);
        assert!(
            Demand {
                pages: 2,
                windows: 0,
                slots: 2
            }
            .fits(&pool.usage()?)
        );
        Ok(())
    }

    #[test]
    fn test_released_bindings_become_stale_after_reclaim() -> Result<()> {
        let pool = pool(2, 2)?;
        let other = self::pool(2, 2)?;
        let state = ModelState::new(&pool);
        let reservation = state.reserve(3, Boundaries::Final)?;
        let binding = reservation.binding();
        assert_eq!(binding.resolve(&pool)?.pages, vec![0, 1]);
        assert!(binding.resolve(&other).is_err());
        drop(reservation);
        assert!(binding.resolve(&pool).is_err());
        pool.reclaim();
        let _reused = state.reserve(3, Boundaries::Final)?;
        assert!(binding.resolve(&pool).is_err());
        Ok(())
    }

    #[test]
    fn test_reservations_validate_lengths_and_count_new_bytes() -> Result<()> {
        let pool = pool(8, 4)?;
        let state = ModelState::new(&pool);
        assert!(state.reserve(0, Boundaries::Final).is_err());
        assert!(state.reserve(17, Boundaries::Final).is_err());
        let usage = pool.usage()?;
        let reservation = state.reserve(3, Boundaries::EveryToken)?;
        assert_eq!(
            reservation.bytes()?,
            2 * usage.page_bytes + 3 * usage.slot_bytes
        );
        assert_eq!(
            state.demand(3, Boundaries::EveryToken)?.bytes(&usage),
            reservation.bytes()?
        );
        Ok(())
    }

    #[test]
    fn test_exclusive_counts_only_what_dropping_a_snapshot_releases() -> Result<()> {
        let pool = pool(8, 4)?;
        let prefix = pass(&ModelState::new(&pool), &[1.0; 4], Boundaries::Final, 4)?;
        assert_eq!(
            prefix.exclusive(),
            Demand {
                pages: 2,
                windows: 0,
                slots: 1
            }
        );
        // A continuation shares both full pages; each keeps its own slot.
        let longer = pass(&prefix, &[2.0; 3], Boundaries::Final, 3)?;
        assert_eq!(
            prefix.exclusive(),
            Demand {
                pages: 0,
                windows: 0,
                slots: 1
            }
        );
        assert_eq!(
            longer.exclusive(),
            Demand {
                pages: 2,
                windows: 0,
                slots: 1
            }
        );
        let copy = longer.clone();
        assert_eq!(longer.exclusive(), Demand::default());
        drop((longer, copy));
        assert_eq!(prefix.exclusive().pages, 2);
        assert_eq!(ModelState::new(&pool).exclusive(), Demand::default());
        Ok(())
    }

    /// A pool with one full and one windowed attention layer (window 3, pages of 2).
    fn windowed(pages: usize, windows: usize) -> Result<StatePool> {
        let layer = |window| LayerStateKind::Attention {
            kv_heads: 1,
            head_dim: 32,
            window,
        };
        StatePool::new(
            Arc::new(ModelDescription {
                identity: ModelIdentity {
                    revision: "window-fixture".into(),
                    state_format: 1,
                },
                layers: vec![layer(None), layer(Some(3))],
                kv: KvLayout::new(2, KvFormat::F32),
                modalities: vec![Modality::Text],
                vocab_size: 8,
                max_position: 32,
            }),
            &Device::Cpu,
            PoolConfig {
                pages,
                window_pages: windows,
                recurrent_slots: 0,
            },
        )
    }

    /// Append `values` as keys of both layers; returns the committed state.
    fn append(state: &ModelState, values: &[f32]) -> Result<ModelState> {
        let mut reservation = state.reserve(values.len(), Boundaries::Final)?;
        reservation.prepare()?;
        let addresses = reservation.binding().resolve(state.pool())?;
        let rows: Vec<f32> = values.iter().flat_map(|v| [*v; 32]).collect();
        let rows = Tensor::from_vec(rows, (values.len(), 1, 32), &Device::Cpu)?;
        for (layer, table) in [(0, &addresses.pages), (1, &addresses.windows)] {
            state
                .pool()
                .attention(layer)?
                .scatter(table, state.position(), &rows, &rows)?;
        }
        reservation.commit(values.len())
    }

    fn layer_keys(state: &ModelState) -> Result<Vec<Vec<f32>>> {
        state
            .read_layers()?
            .into_iter()
            .map(|layer| match layer {
                LayerSnapshot::Attention { key, .. } => {
                    Ok(key.mean(2)?.flatten_all()?.to_vec1::<f32>()?)
                }
                LayerSnapshot::Recurrent { .. } => Err(invalid("unexpected recurrent layer")),
            })
            .collect()
    }

    #[test]
    fn test_windowed_layers_hold_only_their_window_and_snapshots_keep_their_own() -> Result<()> {
        let pool = windowed(16, 6)?;
        let values: Vec<f32> = (1..=11).map(|v| v as f32).collect();
        let mut state = ModelState::new(&pool);
        for chunk in values.chunks(2) {
            state = append(&state, chunk)?;
            pool.reclaim();
        }
        // Keys 8..11 cover the window of 3 behind position 11: logical pages 4 and 5.
        assert_eq!((state.page_count(), state.window_count()), (6, 2));
        assert_eq!(pool.usage()?.windows.leased, 2);
        let keys = layer_keys(&state)?;
        assert_eq!(keys[0], values);
        assert_eq!(&keys[1][8..], &values[8..]);
        assert!(
            keys[1][..8].iter().all(|&k| k == 0.0),
            "released keys read as zero"
        );
        // A pass demands window pages like pages, and a fork copies its shared tail.
        let fork = state.clone();
        assert_eq!(
            fork.demand(1, Boundaries::Final)?,
            Demand {
                pages: 1,
                windows: 1,
                slots: 0
            }
        );
        let saved = state.clone();
        let longer = append(&fork, &[12.0, 13.0, 14.0])?;
        assert_eq!(&layer_keys(&longer)?[1][11..], &[12.0, 13.0, 14.0]);
        assert_eq!(&layer_keys(&saved)?[1][8..], &values[8..]);
        assert_eq!(saved.exclusive().windows, 0, "the window tail stays shared");
        drop((state, fork, saved));
        assert_eq!(longer.window_count(), 2);
        assert_eq!(longer.exclusive().windows, 2);
        Ok(())
    }

    #[test]
    fn test_window_pool_exhaustion_is_an_explicit_error() -> Result<()> {
        let pool = windowed(16, 2)?;
        let state = append(&ModelState::new(&pool), &[1.0, 2.0, 3.0, 4.0])?;
        assert!(matches!(
            state.reserve(1, Boundaries::Final),
            Err(TurError::MemoryAllocation(_))
        ));
        assert_eq!(state.demand(1, Boundaries::Final)?.windows, 1);
        Ok(())
    }
}
