#![forbid(unsafe_code)]
//! Prefix reuse over exact token boundaries.
//!
//! An entry is a committed [`ModelState`] after some request's first `n` tokens: its page
//! table and the recurrent slot written at exactly that boundary. A recurrent state
//! exists only where a pass ended, so entries sit at pass ends:
//!
//! - every multiple of the spacing, because the pipeline never lets a prefill chunk cross
//!   one;
//! - the end of each prompt, with the logits of its last token, so an identical prompt
//!   samples its first output without a model pass;
//! - the last consumed token of each finished request, so a conversation's next turn can
//!   resume after the reply.
//!
//! Entries share leases with the request that produced them, so a snapshot copies
//! nothing; it keeps one recurrent slot and its pages alive. A snapshot inside a page
//! makes its first appender copy that one tail page.
//!
//! Keys chain a per-cache keyed hash over tokens, seeded by a namespace (tenant or cache
//! salt), so equal prefixes in different namespaces never meet. A hash hit is used only
//! after comparing every token. The pool pins model identity and state layout, so entries
//! cannot cross models, quantizations or positional settings.
//!
//! A lookup probes only the lengths at which entries exist, so its cost follows the
//! number of entries, not the prompt's length.
//!
//! Eviction removes the entry that saves the fewest tokens per retained byte: the tokens
//! it covers beyond the deepest retained entry it extends, times how often its prefix was
//! recently computed or restored, over its recurrent slot and the pages only it holds.
//! Counts are halved periodically, so prefixes that stop recurring lose their protection.
//! A new entry is always kept. Pages and slots return to the pool only when no request
//! still shares them and no unfenced work references them (see [`tur_metal::pool`]).

use candle_core::Tensor;
use std::{
    collections::{BTreeMap, HashMap},
    hash::{BuildHasher, RandomState},
    sync::Arc,
};
use tur_core::{Result, execution::invalid};
use tur_metal::state::ModelState;

/// Snapshot spacing and capacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefixConfig {
    /// Tokens between snapshots; a multiple of the K/V page size.
    pub spacing: usize,
    /// Most snapshots retained; each holds one recurrent slot. Zero disables reuse.
    pub entries: usize,
    /// Most of the state pool's bytes snapshots may take; fewer are retained when
    /// `entries` slots would exceed it. `None` retains exactly `entries`.
    pub state_share: Option<f64>,
    /// Waiting requests with a restorable snapshot take free running slots ahead of older
    /// requests without one, until the oldest waiting request reaches the queue-age limit.
    /// It trades cold latency and throughput for warm latency, so it is off by default.
    pub restorable_first: bool,
}

impl PrefixConfig {
    /// Default snapshot spacing in tokens, the best of a 64–512 sweep on shared-prompt traffic.
    pub const DEFAULT_SPACING: usize = 256;

    /// Disabled: no snapshots and no chunk alignment.
    pub const DISABLED: Self = Self {
        spacing: Self::DEFAULT_SPACING,
        entries: 0,
        state_share: None,
        restorable_first: false,
    };

    /// Default share of state memory for snapshots.
    pub const DEFAULT_STATE_SHARE: f64 = 0.125;

    pub fn enabled(&self) -> bool {
        self.entries > 0
    }

    /// Reject spacings that would leave snapshots inside a page.
    pub fn validate(&self, page_tokens: usize) -> Result<()> {
        if self.enabled() && (self.spacing == 0 || !self.spacing.is_multiple_of(page_tokens)) {
            return Err(invalid(format!(
                "prefix spacing {} must be a positive multiple of the {page_tokens}-token page",
                self.spacing
            )));
        }
        Ok(())
    }
}

/// Cumulative counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixStats {
    pub lookups: u64,
    pub hits: u64,
    /// Prompt tokens restored from snapshots instead of recomputed.
    pub hit_tokens: u64,
    pub inserted: u64,
    pub evicted: u64,
    /// Hash matches whose tokens differed; rejected.
    pub collisions: u64,
    /// Entries held now.
    pub entries: usize,
}

struct Entry {
    tokens: Arc<[u32]>,
    /// Chain keys of every prefix of `tokens`, to find the entries it extends.
    keys: Box<[u64]>,
    state: ModelState,
    /// `[1, vocab]` model logits after the last token, for prompt-end snapshots.
    logits: Option<Tensor>,
    last_use: u64,
}

/// Events (insertions and hits) per entry of capacity between halvings of the reuse
/// counts; at least [`MIN_AGING_EVENTS`].
const AGING_EVENTS_PER_ENTRY: u64 = 32;
const MIN_AGING_EVENTS: u64 = 256;

/// Per-token chain hashes of one request's tokens, extended as it generates.
#[derive(Clone, Debug, Default)]
pub struct PrefixKeys {
    /// Hash of the namespace, which seeds the chain.
    seed: u64,
    /// `keys[i]` identifies the first `i + 1` tokens.
    keys: Vec<u64>,
}

impl PrefixKeys {
    /// Tokens covered.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Key of the first `length` tokens, if covered.
    pub fn get(&self, length: usize) -> Option<u64> {
        self.keys.get(length.checked_sub(1)?).copied()
    }

    /// Longest prefix both key chains cover with equal keys. Chained keys agree on every
    /// shorter prefix, so this is a binary search; equal keys still need their tokens
    /// compared before use.
    pub fn shared(&self, other: &Self) -> usize {
        let (mut low, mut high) = (0, self.keys.len().min(other.keys.len()));
        while low < high {
            let middle = low + (high - low) / 2;
            if self.keys[middle] == other.keys[middle] {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        low
    }
}

/// A verified snapshot of a request's first `length` tokens.
pub struct Restored {
    pub length: usize,
    pub state: ModelState,
    /// The logits after the last of those tokens, when the snapshot kept them.
    pub logits: Option<Tensor>,
}

/// Bounded snapshots of committed prefixes for one engine's pool.
pub struct PrefixCache {
    config: PrefixConfig,
    hasher: RandomState,
    map: HashMap<u64, Entry>,
    /// Entries per token length, so lookups probe only lengths that exist.
    lengths: BTreeMap<usize, usize>,
    /// Recent insertions and hits per key, including evicted keys: a prefix computed
    /// again after its eviction counts as reused.
    reuse: HashMap<u64, u32>,
    events: u64,
    /// Bytes of one K/V page and one recurrent slot, from the first snapshot's pool.
    unit: Option<(usize, usize)>,
    tick: u64,
    stats: PrefixStats,
}

impl PrefixCache {
    pub fn new(config: PrefixConfig) -> Self {
        Self {
            config,
            hasher: RandomState::new(),
            map: HashMap::with_capacity(config.entries),
            lengths: BTreeMap::new(),
            reuse: HashMap::new(),
            events: 0,
            unit: None,
            tick: 0,
            stats: PrefixStats::default(),
        }
    }

    pub fn config(&self) -> &PrefixConfig {
        &self.config
    }

    pub fn stats(&self) -> PrefixStats {
        PrefixStats {
            entries: self.map.len(),
            ..self.stats
        }
    }

    /// Chain hashes of every prefix of `tokens` within `namespace`.
    pub fn keys(&self, namespace: Option<&str>, tokens: &[u32]) -> PrefixKeys {
        let mut keys = PrefixKeys {
            seed: self.hasher.hash_one(namespace),
            keys: Vec::new(),
        };
        self.extend(&mut keys, tokens);
        keys
    }

    /// Hash the tokens `keys` does not cover yet; `tokens` extends the sequence `keys`
    /// was built from.
    pub fn extend(&self, keys: &mut PrefixKeys, tokens: &[u32]) {
        if !self.config.enabled() {
            return;
        }
        let mut key = keys.keys.last().copied().unwrap_or(keys.seed);
        for &token in tokens.get(keys.keys.len()..).unwrap_or_default() {
            key = self.hasher.hash_one((key, token));
            keys.keys.push(key);
        }
    }

    /// Next aligned position after `position`, where a prefill chunk must end.
    pub fn next_boundary(&self, position: usize) -> Option<usize> {
        self.config
            .enabled()
            .then(|| (position / self.config.spacing + 1) * self.config.spacing)
    }

    /// Whether a commit at `position` is an aligned snapshot boundary.
    pub fn is_boundary(&self, position: usize) -> bool {
        self.config.enabled() && position > 0 && position.is_multiple_of(self.config.spacing)
    }

    /// Deepest snapshot a request computing a `prompt`-token prompt will insert within the
    /// first `shared` tokens of its sequence: its prompt end, or an aligned boundary.
    pub fn reachable(&self, shared: usize, prompt: usize) -> usize {
        if !self.config.enabled() {
            0
        } else if shared >= prompt {
            prompt
        } else {
            shared - shared % self.config.spacing
        }
    }

    /// Key and length of the deepest verified entry for a prefix of `tokens` whose length
    /// `accept` admits, counting rejected hash matches. A snapshot of all of `tokens`
    /// needs its logits, since an output must come from them.
    fn find(
        &self,
        keys: &PrefixKeys,
        tokens: &[u32],
        accept: impl Fn(usize) -> bool,
        collisions: &mut u64,
    ) -> Option<(u64, usize)> {
        let longest = tokens.len().min(keys.keys.len());
        for (&length, _) in self.lengths.range(1..=longest).rev() {
            let key = keys.keys[length - 1];
            let Some(entry) = self.map.get(&key) else {
                continue;
            };
            if tokens.get(..length) != Some(&entry.tokens[..]) {
                *collisions += 1;
                continue;
            }
            if accept(length) && (length < tokens.len() || entry.logits.is_some()) {
                return Some((key, length));
            }
        }
        None
    }

    /// Tokens of `tokens` a lookup could restore now, without counting it as one.
    pub fn depth(&self, keys: &PrefixKeys, tokens: &[u32]) -> usize {
        if !self.config.enabled() {
            return 0;
        }
        self.find(keys, tokens, |_| true, &mut 0)
            .map_or(0, |(_, length)| length)
    }

    /// Deepest verified snapshot of a prefix of `tokens` whose length `accept` admits. A
    /// snapshot of all of `tokens` needs its logits, since an output must come from them.
    pub fn lookup(
        &mut self,
        keys: &PrefixKeys,
        tokens: &[u32],
        accept: impl Fn(usize) -> bool,
    ) -> Option<Restored> {
        if !self.config.enabled() || self.map.is_empty() {
            return None;
        }
        self.stats.lookups += 1;
        let mut collisions = 0;
        let found = self.find(keys, tokens, accept, &mut collisions);
        self.stats.collisions += collisions;
        let (key, length) = found?;
        self.tick += 1;
        self.observe(key);
        let entry = self.map.get_mut(&key)?;
        entry.last_use = self.tick;
        self.stats.hits += 1;
        self.stats.hit_tokens += length as u64;
        Some(Restored {
            length,
            state: entry.state.clone(),
            logits: entry.logits.clone(),
        })
    }

    /// Count one insertion or hit of `key`, halving every count once enough accumulated.
    fn observe(&mut self, key: u64) {
        let count = self.reuse.entry(key).or_default();
        *count = count.saturating_add(1);
        self.events += 1;
        let period = (self.config.entries as u64)
            .saturating_mul(AGING_EVENTS_PER_ENTRY)
            .max(MIN_AGING_EVENTS);
        if self.events >= period {
            self.events = 0;
            self.reuse.retain(|_, count| {
                *count /= 2;
                *count > 0
            });
        }
    }

    /// Retain `state`, committed after exactly `tokens` (covered by `keys`), with the
    /// logits after its last token if known. An entry for the same tokens is refreshed and
    /// gains logits it lacked; a colliding one is kept.
    pub fn insert(
        &mut self,
        keys: &PrefixKeys,
        tokens: &[u32],
        state: &ModelState,
        logits: Option<&Tensor>,
    ) -> Result<()> {
        if !self.config.enabled() {
            return Ok(());
        }
        if tokens.is_empty() || state.position() != tokens.len() {
            return Err(invalid("prefix snapshots must sit at committed boundaries"));
        }
        let Some(&key) = keys.keys.get(tokens.len() - 1) else {
            return Err(invalid("prefix keys do not cover the snapshot"));
        };
        if self.unit.is_none() {
            let description = state.description();
            self.unit = Some((
                description.page_bytes()?,
                if description.has_recurrent() {
                    description.recurrent_slot_bytes()?
                } else {
                    0
                },
            ));
        }
        self.tick += 1;
        self.observe(key);
        if let Some(entry) = self.map.get_mut(&key) {
            if entry.tokens[..] == *tokens {
                entry.last_use = self.tick;
                if entry.logits.is_none() {
                    entry.logits = logits.cloned();
                }
            } else {
                self.stats.collisions += 1;
            }
            return Ok(());
        }
        while self.map.len() >= self.config.entries {
            if !self.evict() {
                break;
            }
        }
        self.map.insert(
            key,
            Entry {
                tokens: tokens.into(),
                keys: keys.keys[..tokens.len()].into(),
                state: state.clone(),
                logits: logits.cloned(),
                last_use: self.tick,
            },
        );
        *self.lengths.entry(tokens.len()).or_default() += 1;
        self.stats.inserted += 1;
        Ok(())
    }

    /// Recently saved tokens per retained byte. A request matching the entry would
    /// otherwise restore the deepest retained entry it extends, so it saves the tokens
    /// beyond that one, once per recent insertion or hit of its prefix; the bytes are its
    /// slot and the pages only it holds. Ancestors are found by key, without comparing
    /// tokens: a collision only misjudges an eviction.
    fn value(&self, key: u64, entry: &Entry) -> f64 {
        let (page_bytes, slot_bytes) = self.unit.unwrap_or((1, 1));
        let length = entry.tokens.len();
        let ancestor = self
            .lengths
            .range(1..length)
            .rev()
            .map(|(&shorter, _)| shorter)
            .find(|&shorter| self.map.contains_key(&entry.keys[shorter - 1]))
            .unwrap_or(0);
        let reuse = self.reuse.get(&key).copied().unwrap_or(0).max(1);
        let bytes = entry.state.exclusive().pages * page_bytes + slot_bytes;
        (length - ancestor) as f64 * f64::from(reuse) / bytes.max(1) as f64
    }

    /// Drop the entry saving the fewest tokens per retained byte, the least recently used
    /// among equals. Returns whether one was dropped.
    pub fn evict(&mut self) -> bool {
        let Some(key) = self
            .map
            .iter()
            .map(|(key, entry)| (self.value(*key, entry), entry.last_use, *key))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .map(|(_, _, key)| key)
        else {
            return false;
        };
        if let Some(entry) = self.map.remove(&key)
            && let Some(count) = self.lengths.get_mut(&entry.tokens.len())
        {
            *count -= 1;
            if *count == 0 {
                self.lengths.remove(&entry.tokens.len());
            }
        }
        self.stats.evicted += 1;
        true
    }

    /// Drop every entry and reuse count.
    pub fn clear(&mut self) {
        self.stats.evicted += self.map.len() as u64;
        self.map.clear();
        self.lengths.clear();
        self.reuse.clear();
        self.events = 0;
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use tur_core::execution::{
        KvFormat, KvLayout, LayerStateKind, Modality, ModelDescription, ModelIdentity,
    };
    use tur_metal::{
        pool::{PoolConfig, StatePool},
        state::Boundaries,
    };

    fn pool() -> Result<StatePool> {
        StatePool::new(
            Arc::new(ModelDescription {
                identity: ModelIdentity {
                    revision: "prefix-fixture".into(),
                    state_format: 1,
                },
                layers: vec![
                    LayerStateKind::Attention {
                        kv_heads: 1,
                        head_dim: 4,
                    },
                    LayerStateKind::Recurrent {
                        conv_channels: 1,
                        conv_width: 2,
                        heads: 1,
                        key_dim: 1,
                        value_dim: 1,
                        dtype: DType::F32,
                    },
                ],
                kv: KvLayout::new(2, KvFormat::F32),
                modalities: vec![Modality::Text],
                vocab_size: 16,
                max_position: 32,
            }),
            &Device::Cpu,
            PoolConfig {
                pages: 16,
                recurrent_slots: 8,
            },
        )
    }

    /// A committed state after `tokens` tokens.
    fn state(pool: &StatePool, tokens: usize) -> Result<ModelState> {
        let mut reservation = ModelState::new(pool).reserve(tokens, Boundaries::Final)?;
        reservation.prepare()?;
        reservation.commit(tokens)
    }

    fn cache(entries: usize) -> PrefixCache {
        PrefixCache::new(PrefixConfig {
            spacing: 4,
            entries,
            state_share: None,
            restorable_first: false,
        })
    }

    fn length(hit: Option<Restored>) -> Option<usize> {
        hit.map(|hit| hit.length)
    }

    #[test]
    fn test_lookup_returns_the_deepest_verified_admissible_prefix() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(4);
        let prompt: Vec<u32> = (0..11).collect();
        let mut keys = cache.keys(None, &prompt[..5]);
        assert_eq!(keys.len(), 5);
        cache.extend(&mut keys, &prompt);
        assert_eq!(keys.keys, cache.keys(None, &prompt).keys);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None)?;
        cache.insert(&keys, &prompt[..7], &state(&pool, 7)?, None)?;
        assert!(
            cache
                .insert(&keys, &prompt[..6], &state(&pool, 5)?, None)
                .is_err()
        );
        let hit = cache
            .lookup(&keys, &prompt, |_| true)
            .ok_or_else(|| invalid("miss"))?;
        assert_eq!((hit.length, hit.state.position()), (7, 7));
        // The caller may restrict admissible lengths, e.g. to its own pass boundaries.
        assert_eq!(length(cache.lookup(&keys, &prompt, |n| n != 7)), Some(4));
        // A different continuation shares only the first four tokens.
        let mut other = prompt.clone();
        other[5] = 15;
        let other_keys = cache.keys(None, &other);
        assert_eq!(length(cache.lookup(&other_keys, &other, |_| true)), Some(4));
        // Namespaces never share entries.
        let salted = cache.keys(Some("tenant"), &prompt);
        assert!(cache.lookup(&salted, &prompt, |_| true).is_none());
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.hit_tokens, stats.lookups), (3, 15, 4));
        Ok(())
    }

    #[test]
    fn test_a_whole_prompt_hit_needs_the_snapshot_logits() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(4);
        let prompt: Vec<u32> = (0..6).collect();
        let keys = cache.keys(None, &prompt);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None)?;
        cache.insert(&keys, &prompt, &state(&pool, 6)?, None)?;
        // Without logits a repeat must recompute its last token.
        assert_eq!(length(cache.lookup(&keys, &prompt, |_| true)), Some(4));
        let logits = Tensor::zeros((1, 16), DType::F32, &Device::Cpu)?;
        cache.insert(&keys, &prompt, &state(&pool, 6)?, Some(&logits))?;
        let hit = cache
            .lookup(&keys, &prompt, |_| true)
            .ok_or_else(|| invalid("miss"))?;
        assert_eq!(hit.length, 6);
        assert!(hit.logits.is_some());
        // A longer prompt uses the same snapshot without needing its logits.
        let longer: Vec<u32> = (0..9).collect();
        let longer_keys = cache.keys(None, &longer);
        assert_eq!(
            length(cache.lookup(&longer_keys, &longer, |_| true)),
            Some(6)
        );
        Ok(())
    }

    #[test]
    fn test_hash_matches_are_verified_against_tokens() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let prompt: Vec<u32> = (0..8).collect();
        let keys = cache.keys(None, &prompt);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None)?;
        // Forge a collision: different tokens presented with the stored key.
        let forged: Vec<u32> = (8..16).collect();
        assert!(cache.lookup(&keys, &forged[..5], |_| true).is_none());
        cache.insert(&keys, &forged[..4], &state(&pool, 4)?, None)?;
        assert_eq!(cache.stats().collisions, 2);
        assert_eq!(length(cache.lookup(&keys, &prompt, |_| true)), Some(4));
        Ok(())
    }

    #[test]
    fn test_eviction_prefers_unused_entries_and_releases_leases() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let prompts: Vec<Vec<u32>> = (0..3u32)
            .map(|p| (0..5).map(|t| t + p * 5).collect())
            .collect();
        let keys: Vec<_> = prompts.iter().map(|p| cache.keys(None, p)).collect();
        cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None)?;
        cache.insert(&keys[1], &prompts[1][..4], &state(&pool, 4)?, None)?;
        assert!(cache.lookup(&keys[0], &prompts[0], |_| true).is_some());
        // Inserting a third evicts the unused second entry, not the older hit one.
        cache.insert(&keys[2], &prompts[2][..4], &state(&pool, 4)?, None)?;
        assert!(cache.lookup(&keys[0], &prompts[0], |_| true).is_some());
        assert!(cache.lookup(&keys[1], &prompts[1], |_| true).is_none());
        assert_eq!(pool.usage()?.slots.leased, 2);
        cache.clear();
        let usage = pool.usage()?;
        assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
        assert!(!cache.evict());
        Ok(())
    }

    #[test]
    fn test_lookups_probe_only_retained_lengths_and_depth_counts_nothing() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(3);
        let prompt: Vec<u32> = (0..12).collect();
        let keys = cache.keys(None, &prompt);
        for length in [4, 8, 9] {
            cache.insert(&keys, &prompt[..length], &state(&pool, length)?, None)?;
        }
        assert_eq!(cache.depth(&keys, &prompt), 9);
        assert_eq!(cache.depth(&keys, &prompt[..8]), 4);
        assert_eq!(cache.stats().lookups, 0);
        // Evicting the 9-token entry (it saves one token beyond the 8-token one) leaves
        // the length index consistent.
        assert!(cache.evict());
        assert_eq!(length(cache.lookup(&keys, &prompt, |_| true)), Some(8));
        cache.clear();
        assert_eq!(cache.depth(&keys, &prompt), 0);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None)?;
        assert_eq!(length(cache.lookup(&keys, &prompt, |_| true)), Some(4));
        Ok(())
    }

    #[test]
    fn test_shared_keys_and_reachable_snapshots() {
        let cache = cache(2);
        let prompt: Vec<u32> = (0..10).collect();
        let mut other = prompt.clone();
        other[6] = 99;
        let (keys, other_keys) = (cache.keys(None, &prompt), cache.keys(None, &other));
        assert_eq!(keys.shared(&other_keys), 6);
        assert_eq!(keys.shared(&cache.keys(None, &prompt[..3])), 3);
        assert_eq!(keys.shared(&cache.keys(Some("tenant"), &prompt)), 0);
        assert_eq!(keys.get(6), other_keys.get(6));
        assert_eq!(keys.get(0), None);
        // A producer inserts aligned snapshots and its prompt end.
        assert_eq!(cache.reachable(6, 10), 4);
        assert_eq!(cache.reachable(10, 10), 10);
        assert_eq!(cache.reachable(12, 7), 7);
        assert_eq!(cache.reachable(3, 10), 0);
        assert_eq!(PrefixCache::new(PrefixConfig::DISABLED).reachable(8, 8), 0);
    }

    #[test]
    fn test_eviction_keeps_entries_saving_the_most_recurring_work() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let prompts: Vec<Vec<u32>> = (0..4u32)
            .map(|p| (0..8).map(|t| t + p * 8).collect())
            .collect();
        let keys: Vec<_> = prompts.iter().map(|p| cache.keys(None, p)).collect();
        // The first prefix is computed, evicted by two single-use prefixes, then computed
        // again: the repeat outranks newer single-use entries.
        cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None)?;
        cache.insert(&keys[1], &prompts[1][..4], &state(&pool, 4)?, None)?;
        cache.insert(&keys[2], &prompts[2][..4], &state(&pool, 4)?, None)?;
        assert!(cache.lookup(&keys[0], &prompts[0], |_| true).is_none());
        cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None)?;
        cache.insert(&keys[3], &prompts[3][..4], &state(&pool, 4)?, None)?;
        assert_eq!(
            length(cache.lookup(&keys[0], &prompts[0], |_| true)),
            Some(4)
        );
        // An extension saves only its tokens beyond the entry it extends. A new entry is
        // always kept; at the next insertion the extension goes before either full
        // prefix, although it is newer and longer than the one it extends.
        drop(cache);
        pool.reclaim();
        let mut wider = self::cache(3);
        wider.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None)?;
        wider.insert(&keys[0], &prompts[0][..6], &state(&pool, 6)?, None)?;
        wider.insert(&keys[1], &prompts[1][..4], &state(&pool, 4)?, None)?;
        wider.insert(&keys[2], &prompts[2][..4], &state(&pool, 4)?, None)?;
        assert_eq!(
            length(wider.lookup(&keys[0], &prompts[0], |_| true)),
            Some(4)
        );
        assert_eq!(
            length(wider.lookup(&keys[1], &prompts[1], |_| true)),
            Some(4)
        );
        assert_eq!(wider.stats().evicted, 1);
        Ok(())
    }

    #[test]
    fn test_reuse_counts_age_away() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let stale: Vec<u32> = (0..8).collect();
        let stale_keys = cache.keys(None, &stale);
        cache.insert(&stale_keys, &stale[..4], &state(&pool, 4)?, None)?;
        for _ in 0..20 {
            assert!(cache.lookup(&stale_keys, &stale, |_| true).is_some());
        }
        // A stream of single-use prefixes halves the stale count (21) every aging period
        // until it ties them and, as the least recently used, goes.
        let mut evicted_at = None;
        for salt in 1..=(16 * MIN_AGING_EVENTS as u32) {
            pool.reclaim();
            let fresh: Vec<u32> = (0..4).map(|t| t + salt * 8).collect();
            let keys = cache.keys(None, &fresh);
            cache.insert(&keys, &fresh, &state(&pool, 4)?, None)?;
            if cache.depth(&stale_keys, &stale) == 0 {
                evicted_at = Some(salt);
                break;
            }
        }
        assert!(
            evicted_at.is_some_and(|salt| salt > 3 * MIN_AGING_EVENTS as u32),
            "evicted after {evicted_at:?} fresh prefixes"
        );
        Ok(())
    }

    #[test]
    fn test_disabled_cache_never_aligns_or_stores() -> Result<()> {
        let pool = pool()?;
        let mut cache = PrefixCache::new(PrefixConfig::DISABLED);
        assert_eq!(cache.next_boundary(3), None);
        assert!(!cache.is_boundary(256));
        let keys = cache.keys(None, &[1; 6]);
        assert!(keys.is_empty());
        cache.insert(&keys, &[1; 4], &state(&pool, 4)?, None)?;
        assert_eq!(cache.stats().entries, 0);
        assert!(cache.lookup(&keys, &[1; 6], |_| true).is_none());
        assert!(
            PrefixConfig {
                spacing: 24,
                entries: 1,
                state_share: None,
                restorable_first: false,
            }
            .validate(16)
            .is_err()
        );
        assert!(PrefixConfig::DISABLED.validate(96).is_ok());
        Ok(())
    }
}
