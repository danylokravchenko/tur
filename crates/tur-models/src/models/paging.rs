//! Page addressing of one packed pass, shared by architectures with paged attention.
use crate::hybrid::{Addresses, StatePool};
use tur_core::{
    Result,
    execution::{LayerStateKind, ModelDescription, invalid},
};
use tur_metal::kernels::{Kernels, PagedKv};

/// Paged addressing of one pass, for the full and the windowed attention layers.
pub(crate) struct Paging {
    full: Option<PagedKv>,
    windowed: Option<PagedKv>,
}

impl Paging {
    /// Each sequence's tables, and for each new token its first table entry and visible
    /// key count. `spans` yields each sequence's addresses, committed tokens and new tokens.
    pub(crate) fn new<'a>(
        k: &Kernels,
        description: &ModelDescription,
        pool: &StatePool,
        spans: impl Iterator<Item = (&'a Addresses, usize, usize)>,
        tokens: usize,
    ) -> Result<Self> {
        Self::queries(
            k,
            description,
            pool,
            spans.map(|(addresses, past, len)| (addresses, past..past + len)),
            tokens,
        )
    }

    /// [`Self::new`] for queries at chosen positions: `spans` yields each sequence's
    /// addresses and the increasing positions of its queries in this pass.
    pub(crate) fn queries<'a, P: IntoIterator<Item = usize>>(
        k: &Kernels,
        description: &ModelDescription,
        pool: &StatePool,
        spans: impl Iterator<Item = (&'a Addresses, P)>,
        tokens: usize,
    ) -> Result<Self> {
        let mut first = Vec::with_capacity(tokens);
        let mut visible = Vec::with_capacity(tokens);
        let mut tables = Tables::default();
        for (addresses, positions) in spans {
            let base = tables.push(addresses)?;
            for p in positions {
                let keys = p
                    .checked_add(1)
                    .ok_or_else(|| invalid("context overflow"))?;
                first.push(base);
                visible.push(u32::try_from(keys).map_err(|_| invalid("context overflow"))?);
            }
        }
        let (full, windows) = (&tables.full, &tables.windows);
        Self::from_tables(k, description, pool, (full, windows), &first, &visible)
    }

    /// Addressing from page tables already laid out for the pass.
    pub(crate) fn from_tables(
        k: &Kernels,
        description: &ModelDescription,
        pool: &StatePool,
        (full, windows): (&[u32], &[u32]),
        first: &[u32],
        visible: &[u32],
    ) -> Result<Self> {
        let rows = |windowed: bool| -> Result<Option<usize>> {
            description
                .layers
                .iter()
                .position(|kind| {
                    matches!(kind, LayerStateKind::Attention { .. }) && kind.windowed() == windowed
                })
                .map(|layer| Ok(pool.attention(layer)?.key.data.dim(0)?))
                .transpose()
        };
        let page = description.kv.page_tokens;
        Ok(Self {
            full: rows(false)?
                .map(|slots| k.paged_kv(page, slots, full, first, visible))
                .transpose()?,
            windowed: rows(true)?
                .map(|slots| k.paged_kv(page, slots, windows, first, visible))
                .transpose()?,
        })
    }

    pub(crate) fn layer(&self, windowed: bool) -> Result<&PagedKv> {
        if windowed { &self.windowed } else { &self.full }
            .as_ref()
            .ok_or_else(|| invalid("no page addressing for this attention layer"))
    }
}

/// The full and window page tables of a pass, laid out as [`Paging`] lays them out: each
/// sequence's tables at the same base in both, the shorter padded.
#[derive(Default)]
pub(crate) struct Tables {
    pub(crate) full: Vec<u32>,
    pub(crate) windows: Vec<u32>,
}

impl Tables {
    /// Append one sequence's tables and return their base.
    pub(crate) fn push(&mut self, addresses: &Addresses) -> Result<u32> {
        let start = self.full.len().max(self.windows.len());
        let base = u32::try_from(start).map_err(|_| invalid("page table overflow"))?;
        // Both tables index the same logical pages; pad the shorter so bases agree.
        self.full.extend_from_slice(&addresses.pages);
        self.windows.extend_from_slice(&addresses.windows);
        let end = start + addresses.pages.len().max(addresses.windows.len());
        self.full.resize(end, 0);
        self.windows.resize(end, 0);
        Ok(base)
    }
}
