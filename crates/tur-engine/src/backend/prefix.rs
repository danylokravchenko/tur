#![forbid(unsafe_code)]
//! Prefix reuse over exact token boundaries.
//!
//! An entry is a committed [`ModelState`] after some request's first `n` tokens: its page
//! table and the recurrent slot written at exactly that boundary. A recurrent state
//! exists only where a pass ended, so entries sit at pass ends:
//!
//! - every multiple of the spacing, where a prefill pass keeps the recurrent state as a
//!   checkpoint if it doesn't end there;
//! - the end of each prompt, with the logits of its last token, so an identical prompt
//!   samples its first output without a model pass;
//! - a predicted template-divergence boundary in prompts containing assistant history,
//!   so the next user turn can reuse the tokens before rewritten reasoning blocks;
//! - the last consumed token of each finished request, so a conversation's next turn can
//!   resume after the reply.
//!
//! Entries share leases with the request that produced them, so a snapshot copies
//! nothing; it keeps one recurrent slot and its pages alive. A snapshot inside a page
//! makes its first appender copy that one tail page.
//!
//! A draft head that writes its own keys and values (Qwen3.5's MTP head) reads the token
//! after each position, so its entry at a snapshot's last position depends on the token
//! that followed it. A snapshot records that token and the final hidden state there. A
//! request that needs the head and continues with the same token uses the entry as it is.
//! One that continues differently recomputes it from the hidden state, which needs the
//! snapshot to end inside a page: the request's first pass then copies that page, so the
//! entry is rewritten in a page no other holder sees. Chat templates rewrite history right
//! after a turn boundary, so those snapshots are always continued differently.
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
//! Entries on hold (see [`PrefixCache::hold`]) go only once no other entry is left.
//! A new entry is always kept. Pages and slots return to the pool only when no request
//! still shares them and no unfenced work references them (see [`tur_metal::pool`]).

use candle_core::Tensor;
use std::{
    collections::{BTreeMap, HashMap},
    hash::{BuildHasher, RandomState},
    sync::Arc,
    time::{Duration, Instant},
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
    /// How long a reply that ended in tool calls keeps its snapshots ahead of other entries
    /// for eviction, while the caller runs the tools. Zero holds nothing.
    pub tool_hold: Duration,
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
        tool_hold: Duration::ZERO,
    };

    /// Default share of state memory for snapshots.
    pub const DEFAULT_STATE_SHARE: f64 = 0.125;

    /// Default hold on a tool-calling reply's snapshots.
    pub const DEFAULT_TOOL_HOLD: Duration = Duration::from_secs(60);

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
    /// The draft head's entry at the last position; `None` when the head wrote none.
    head: Option<HeadEntry>,
    last_use: u64,
    /// Until when eviction takes this entry only after every entry not on hold.
    held_until: Option<Instant>,
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

/// The draft head's entry at a snapshot's last position (see the module documentation).
#[derive(Clone, Debug)]
pub struct HeadEntry {
    /// The token after the snapshot that the entry read.
    pub next: u32,
    /// `[1, hidden]` final hidden state at the last position, to recompute the entry for
    /// another next token.
    pub hidden: Option<Tensor>,
}

impl HeadEntry {
    /// Whether a request restoring the snapshot of `length` tokens, with `page_tokens`-token
    /// pages, can recompute this entry for its own next token.
    pub fn recomputable(&self, length: usize, page_tokens: usize) -> bool {
        self.hidden.is_some() && !length.is_multiple_of(page_tokens)
    }
}

/// A verified snapshot of a request's first `length` tokens.
pub struct Restored {
    pub length: usize,
    pub state: ModelState,
    /// The logits after the last of those tokens, when the snapshot kept them.
    pub logits: Option<Tensor>,
    /// The draft head's entry at the last of them, when it wrote its entries.
    pub head: Option<HeadEntry>,
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

    /// Snapshot positions strictly inside a pass of `tokens` tokens after `position`, as
    /// numbers of the pass's tokens: aligned positions and `turn`, a predicted
    /// template-rewrite boundary. The pass keeps the recurrent state at each, so a snapshot
    /// can sit there without the pass ending.
    pub fn checkpoints(&self, position: usize, tokens: usize, turn: Option<usize>) -> Vec<usize> {
        if !self.config.enabled() {
            return Vec::new();
        }
        (1..tokens)
            .filter(|&offset| {
                let at = position + offset;
                self.is_boundary(at) || turn == Some(at)
            })
            .collect()
    }

    /// Next aligned position after `position`. A speculative pass, which keeps no
    /// checkpoints, ends there.
    pub fn next_boundary(&self, position: usize) -> Option<usize> {
        self.config
            .enabled()
            .then(|| (position / self.config.spacing + 1) * self.config.spacing)
    }

    /// Whether a commit at `position` is an aligned snapshot boundary.
    pub fn is_boundary(&self, position: usize) -> bool {
        self.config.enabled() && position > 0 && position.is_multiple_of(self.config.spacing)
    }

    /// Deepest aligned or prompt-end snapshot a request computing a `prompt`-token
    /// prompt will insert within the first `shared` tokens of its sequence.
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
    /// needs its logits, since an output must come from them. With `head`, the draft
    /// head's entries must hold up to the snapshot, its last one usable or recomputable
    /// for the token after it.
    fn find(
        &self,
        keys: &PrefixKeys,
        tokens: &[u32],
        head: bool,
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
            let head_holds = !head
                || entry.head.as_ref().is_some_and(|entry_head| {
                    tokens
                        .get(length)
                        .is_none_or(|&token| token == entry_head.next)
                        || entry_head.recomputable(length, entry.state.description().kv.page_tokens)
                });
            if accept(length) && head_holds && (length < tokens.len() || entry.logits.is_some()) {
                return Some((key, length));
            }
        }
        None
    }

    /// Tokens of `tokens` a lookup could restore now, without counting it as one.
    pub fn depth(&self, keys: &PrefixKeys, tokens: &[u32], head: bool) -> usize {
        if !self.config.enabled() {
            return 0;
        }
        self.find(keys, tokens, head, |_| true, &mut 0)
            .map_or(0, |(_, length)| length)
    }

    /// Deepest verified snapshot of a prefix of `tokens` whose length `accept` admits. A
    /// snapshot of all of `tokens` needs its logits, since an output must come from them.
    /// With `head`, the draft head's entries must also hold for `tokens` (see the module
    /// documentation); a snapshot of all of `tokens` holds them as they are only if the
    /// output sampled from its logits is the one [`Restored::head`] read.
    pub fn lookup(
        &mut self,
        keys: &PrefixKeys,
        tokens: &[u32],
        head: bool,
        accept: impl Fn(usize) -> bool,
    ) -> Option<Restored> {
        if !self.config.enabled() || self.map.is_empty() {
            return None;
        }
        self.stats.lookups += 1;
        let mut collisions = 0;
        let found = self.find(keys, tokens, head, accept, &mut collisions);
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
            head: entry.head.clone(),
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
    /// logits after its last token if known and the draft head's entry at its last position
    /// if the head wrote its entries up to there. An entry for the same tokens is refreshed
    /// and gains logits it lacked, and takes this state if only this one has the head's
    /// entries; a colliding one is kept.
    pub fn insert(
        &mut self,
        keys: &PrefixKeys,
        tokens: &[u32],
        state: &ModelState,
        logits: Option<&Tensor>,
        head: Option<HeadEntry>,
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
                if entry.head.is_none() && head.is_some() {
                    entry.state = state.clone();
                    entry.head = head;
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
                head,
                last_use: self.tick,
                held_until: None,
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

    /// Keep the retained snapshots of `tokens`' first `lengths` tokens, most wanted first,
    /// ahead of other entries for eviction until `until`. At most half the capacity is on
    /// hold, because other traffic needs the rest: with three of four entries held, a shared
    /// system prompt lost its snapshots and its users took 1.7× as long. Older holds end first
    /// to make room. A hold reserves nothing, so memory pressure can still take held entries.
    pub fn hold(&mut self, keys: &PrefixKeys, tokens: &[u32], lengths: &[usize], until: Instant) {
        let budget = (self.config.entries / 2).max(1);
        let mut chosen: Vec<u64> = Vec::with_capacity(budget);
        for &length in lengths {
            if chosen.len() < budget
                && let Some(key) = keys.get(length)
                && !chosen.contains(&key)
                && self
                    .map
                    .get(&key)
                    .is_some_and(|entry| tokens.get(..length) == Some(&entry.tokens[..]))
            {
                chosen.push(key);
            }
        }
        let now = Instant::now();
        let mut others: Vec<(Instant, u64)> = self
            .map
            .iter()
            .filter(|(key, _)| !chosen.contains(key))
            .filter_map(|(key, entry)| {
                let held = entry.held_until.filter(|&held| held > now)?;
                Some((held, *key))
            })
            .collect();
        others.sort_unstable_by(|a, b| b.cmp(a));
        for (_, key) in others.iter().skip(budget - chosen.len()) {
            if let Some(entry) = self.map.get_mut(key) {
                entry.held_until = None;
            }
        }
        for key in chosen {
            if let Some(entry) = self.map.get_mut(&key) {
                entry.held_until = Some(until);
            }
        }
    }

    /// Drop the entry saving the fewest tokens per retained byte, the least recently used
    /// among equals. Entries on hold go only when no other remains, soonest-ending hold
    /// first: a newer tool call is likelier to return than an older one of the same loop.
    /// Returns whether one was dropped.
    pub fn evict(&mut self) -> bool {
        let now = Instant::now();
        let Some(key) = self
            .map
            .iter()
            .map(|(key, entry)| {
                let held = entry.held_until.filter(|&until| until > now);
                (held, self.value(*key, entry), entry.last_use, *key)
            })
            .min_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)))
            .map(|(_, _, _, key)| key)
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
                        window: None,
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
                window_pages: 16,
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
            tool_hold: Duration::ZERO,
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
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None, None)?;
        cache.insert(&keys, &prompt[..7], &state(&pool, 7)?, None, None)?;
        assert!(
            cache
                .insert(&keys, &prompt[..6], &state(&pool, 5)?, None, None)
                .is_err()
        );
        let hit = cache
            .lookup(&keys, &prompt, false, |_| true)
            .ok_or_else(|| invalid("miss"))?;
        assert_eq!((hit.length, hit.state.position()), (7, 7));
        // The caller may restrict admissible lengths, e.g. to its own pass boundaries.
        assert_eq!(
            length(cache.lookup(&keys, &prompt, false, |n| n != 7)),
            Some(4)
        );
        // A different continuation shares only the first four tokens.
        let mut other = prompt.clone();
        other[5] = 15;
        let other_keys = cache.keys(None, &other);
        assert_eq!(
            length(cache.lookup(&other_keys, &other, false, |_| true)),
            Some(4)
        );
        // Namespaces never share entries.
        let salted = cache.keys(Some("tenant"), &prompt);
        assert!(cache.lookup(&salted, &prompt, false, |_| true).is_none());
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
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None, None)?;
        cache.insert(&keys, &prompt, &state(&pool, 6)?, None, None)?;
        // Without logits a repeat must recompute its last token.
        assert_eq!(
            length(cache.lookup(&keys, &prompt, false, |_| true)),
            Some(4)
        );
        let logits = Tensor::zeros((1, 16), DType::F32, &Device::Cpu)?;
        cache.insert(&keys, &prompt, &state(&pool, 6)?, Some(&logits), None)?;
        let hit = cache
            .lookup(&keys, &prompt, false, |_| true)
            .ok_or_else(|| invalid("miss"))?;
        assert_eq!(hit.length, 6);
        assert!(hit.logits.is_some());
        // A longer prompt uses the same snapshot without needing its logits.
        let longer: Vec<u32> = (0..9).collect();
        let longer_keys = cache.keys(None, &longer);
        assert_eq!(
            length(cache.lookup(&longer_keys, &longer, false, |_| true)),
            Some(6)
        );
        Ok(())
    }

    fn head(next: u32, hidden: bool) -> Result<Option<HeadEntry>> {
        Ok(Some(HeadEntry {
            next,
            hidden: hidden
                .then(|| Tensor::zeros((1, 4), DType::F32, &Device::Cpu))
                .transpose()?,
        }))
    }

    #[test]
    fn test_a_draft_head_restores_entries_it_can_use_or_recompute() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(4);
        let prompt: Vec<u32> = (0..7).collect();
        let keys = cache.keys(None, &prompt);
        cache.insert(&keys, &prompt[..2], &state(&pool, 2)?, None, head(2, true)?)?;
        // Pages hold two tokens: an entry ending on a page boundary can't be recomputed.
        cache.insert(
            &keys,
            &prompt[..6],
            &state(&pool, 6)?,
            None,
            head(15, true)?,
        )?;
        assert_eq!(
            length(cache.lookup(&keys, &prompt, true, |_| true)),
            Some(2)
        );
        // A request without the head takes the deeper snapshot.
        assert_eq!(cache.depth(&keys, &prompt, false), 6);
        // Inside a page, the hidden state lets the request recompute the entry.
        cache.insert(
            &keys,
            &prompt[..5],
            &state(&pool, 5)?,
            None,
            head(15, true)?,
        )?;
        assert_eq!(cache.depth(&keys, &prompt, true), 5);
        let other: Vec<u32> = (20..26).collect();
        let other_keys = cache.keys(None, &other);
        cache.insert(
            &other_keys,
            &other[..3],
            &state(&pool, 3)?,
            None,
            head(15, false)?,
        )?;
        assert_eq!(cache.depth(&other_keys, &other, true), 0);
        // A whole-prompt hit is restored before its output is sampled.
        let logits = Tensor::zeros((1, 16), DType::F32, &Device::Cpu)?;
        cache.insert(&keys, &prompt[..2], &state(&pool, 2)?, Some(&logits), None)?;
        let hit = cache
            .lookup(&keys, &prompt[..2], true, |_| true)
            .ok_or_else(|| invalid("miss"))?;
        assert_eq!(hit.length, 2);
        assert_eq!(hit.head.map(|entry| entry.next), Some(2));
        // An entry without the head's entries serves only requests without the head, until
        // a snapshot of the same tokens that has them replaces its state.
        cache.clear();
        let headless = state(&pool, 2)?;
        cache.insert(&other_keys, &other[..2], &headless, None, None)?;
        assert_eq!(cache.depth(&other_keys, &other, true), 0);
        assert_eq!(cache.depth(&other_keys, &other, false), 2);
        assert_eq!(headless.exclusive().pages, 0);
        cache.insert(
            &other_keys,
            &other[..2],
            &state(&pool, 2)?,
            None,
            head(22, false)?,
        )?;
        assert_eq!(cache.depth(&other_keys, &other, true), 2);
        // The cache let go of the state without the head's entries.
        assert_eq!(headless.exclusive().pages, 1);
        Ok(())
    }

    #[test]
    fn test_hash_matches_are_verified_against_tokens() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let prompt: Vec<u32> = (0..8).collect();
        let keys = cache.keys(None, &prompt);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None, None)?;
        // Forge a collision: different tokens presented with the stored key.
        let forged: Vec<u32> = (8..16).collect();
        assert!(cache.lookup(&keys, &forged[..5], false, |_| true).is_none());
        cache.insert(&keys, &forged[..4], &state(&pool, 4)?, None, None)?;
        assert_eq!(cache.stats().collisions, 2);
        assert_eq!(
            length(cache.lookup(&keys, &prompt, false, |_| true)),
            Some(4)
        );
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
        cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None, None)?;
        cache.insert(&keys[1], &prompts[1][..4], &state(&pool, 4)?, None, None)?;
        assert!(
            cache
                .lookup(&keys[0], &prompts[0], false, |_| true)
                .is_some()
        );
        // Inserting a third evicts the unused second entry, not the older hit one.
        cache.insert(&keys[2], &prompts[2][..4], &state(&pool, 4)?, None, None)?;
        assert!(
            cache
                .lookup(&keys[0], &prompts[0], false, |_| true)
                .is_some()
        );
        assert!(
            cache
                .lookup(&keys[1], &prompts[1], false, |_| true)
                .is_none()
        );
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
            cache.insert(&keys, &prompt[..length], &state(&pool, length)?, None, None)?;
        }
        assert_eq!(cache.depth(&keys, &prompt, false), 9);
        assert_eq!(cache.depth(&keys, &prompt[..8], false), 4);
        assert_eq!(cache.stats().lookups, 0);
        // Evicting the 9-token entry (it saves one token beyond the 8-token one) leaves
        // the length index consistent.
        assert!(cache.evict());
        assert_eq!(
            length(cache.lookup(&keys, &prompt, false, |_| true)),
            Some(8)
        );
        cache.clear();
        assert_eq!(cache.depth(&keys, &prompt, false), 0);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None, None)?;
        assert_eq!(
            length(cache.lookup(&keys, &prompt, false, |_| true)),
            Some(4)
        );
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
        cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None, None)?;
        cache.insert(&keys[1], &prompts[1][..4], &state(&pool, 4)?, None, None)?;
        cache.insert(&keys[2], &prompts[2][..4], &state(&pool, 4)?, None, None)?;
        assert!(
            cache
                .lookup(&keys[0], &prompts[0], false, |_| true)
                .is_none()
        );
        cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None, None)?;
        cache.insert(&keys[3], &prompts[3][..4], &state(&pool, 4)?, None, None)?;
        assert_eq!(
            length(cache.lookup(&keys[0], &prompts[0], false, |_| true)),
            Some(4)
        );
        // An extension saves only its tokens beyond the entry it extends. A new entry is
        // always kept; at the next insertion the extension goes before either full
        // prefix, although it is newer and longer than the one it extends.
        drop(cache);
        pool.reclaim();
        let mut wider = self::cache(3);
        wider.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None, None)?;
        wider.insert(&keys[0], &prompts[0][..6], &state(&pool, 6)?, None, None)?;
        wider.insert(&keys[1], &prompts[1][..4], &state(&pool, 4)?, None, None)?;
        wider.insert(&keys[2], &prompts[2][..4], &state(&pool, 4)?, None, None)?;
        assert_eq!(
            length(wider.lookup(&keys[0], &prompts[0], false, |_| true)),
            Some(4)
        );
        assert_eq!(
            length(wider.lookup(&keys[1], &prompts[1], false, |_| true)),
            Some(4)
        );
        assert_eq!(wider.stats().evicted, 1);
        Ok(())
    }

    /// Four distinct eight-token prompts with their keys, in a cache of four.
    fn held_fixture() -> (PrefixCache, Vec<Vec<u32>>, Vec<PrefixKeys>) {
        let cache = cache(4);
        let prompts: Vec<Vec<u32>> = (0..5u32)
            .map(|p| (0..8).map(|t| t + p * 8).collect())
            .collect();
        let keys = prompts.iter().map(|p| cache.keys(None, p)).collect();
        (cache, prompts, keys)
    }

    #[test]
    fn test_held_entries_are_evicted_last_and_only_until_their_hold_ends() -> Result<()> {
        let pool = pool()?;
        for (until, held_survives) in [
            (Instant::now() + Duration::from_secs(600), true),
            (Instant::now(), false),
        ] {
            let (mut cache, prompts, keys) = held_fixture();
            cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?, None, None)?;
            // The others cover more tokens, so without the hold the first entry goes.
            for p in 1..4 {
                cache.insert(&keys[p], &prompts[p][..6], &state(&pool, 6)?, None, None)?;
            }
            // A length with no entry holds nothing.
            cache.hold(&keys[0], &prompts[0], &[7, 4], until);
            cache.insert(&keys[4], &prompts[4][..6], &state(&pool, 6)?, None, None)?;
            assert_eq!(
                cache.depth(&keys[0], &prompts[0], false) == 4,
                held_survives
            );
            // Once only held entries remain, pressure still takes them.
            while cache.evict() {}
            assert_eq!(cache.stats().entries, 0);
            pool.reclaim();
        }
        Ok(())
    }

    #[test]
    fn test_holds_take_half_the_capacity_and_the_oldest_ends_first() -> Result<()> {
        let pool = pool()?;
        let (mut cache, prompts, keys) = held_fixture();
        for p in 0..4 {
            cache.insert(&keys[p], &prompts[p][..4], &state(&pool, 4)?, None, None)?;
        }
        let soon = Instant::now() + Duration::from_secs(600);
        let later = soon + Duration::from_secs(600);
        cache.hold(&keys[0], &prompts[0], &[4], soon);
        cache.hold(&keys[1], &prompts[1], &[4], soon + Duration::from_secs(1));
        // The third hold ends the first one's, which leaves the two newest.
        cache.hold(&keys[2], &prompts[2], &[4], later);
        assert!(cache.evict());
        assert!(cache.evict());
        assert_eq!(cache.depth(&keys[0], &prompts[0], false), 0);
        assert_eq!(cache.depth(&keys[3], &prompts[3], false), 0);
        // Among holds, the one ending soonest goes first.
        assert!(cache.evict());
        assert_eq!(cache.depth(&keys[1], &prompts[1], false), 0);
        assert_eq!(cache.depth(&keys[2], &prompts[2], false), 4);
        // One call holds only its most wanted lengths within the budget.
        cache.clear();
        pool.reclaim();
        let prompt: Vec<u32> = (0..8).collect();
        let chain = cache.keys(None, &prompt);
        for length in [2, 4, 6] {
            cache.insert(
                &chain,
                &prompt[..length],
                &state(&pool, length)?,
                None,
                None,
            )?;
        }
        cache.insert(&keys[3], &prompts[3][..6], &state(&pool, 6)?, None, None)?;
        cache.hold(&chain, &prompt, &[6, 2, 4], later);
        assert!(cache.evict());
        assert!(cache.evict());
        assert_eq!(cache.depth(&chain, &prompt, false), 6);
        assert_eq!(cache.depth(&chain, &prompt[..4], false), 2);
        Ok(())
    }

    #[test]
    fn test_a_hold_applies_only_to_verified_tokens() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let prompt: Vec<u32> = (0..8).collect();
        let forged: Vec<u32> = (8..16).collect();
        let keys = cache.keys(None, &prompt);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?, None, None)?;
        // A key match over different tokens must not hold the entry.
        cache.hold(
            &keys,
            &forged,
            &[4],
            Instant::now() + Duration::from_secs(600),
        );
        let other = cache.keys(None, &forged);
        cache.insert(&other, &forged[..6], &state(&pool, 6)?, None, None)?;
        let third: Vec<u32> = (16..24).collect();
        let third_keys = cache.keys(None, &third);
        cache.insert(&third_keys, &third[..6], &state(&pool, 6)?, None, None)?;
        assert_eq!(cache.depth(&keys, &prompt, false), 0);
        Ok(())
    }

    #[test]
    fn test_reuse_counts_age_away() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let stale: Vec<u32> = (0..8).collect();
        let stale_keys = cache.keys(None, &stale);
        cache.insert(&stale_keys, &stale[..4], &state(&pool, 4)?, None, None)?;
        for _ in 0..20 {
            assert!(cache.lookup(&stale_keys, &stale, false, |_| true).is_some());
        }
        // A stream of single-use prefixes halves the stale count (21) every aging period
        // until it ties them and, as the least recently used, goes.
        let mut evicted_at = None;
        for salt in 1..=(16 * MIN_AGING_EVENTS as u32) {
            pool.reclaim();
            let fresh: Vec<u32> = (0..4).map(|t| t + salt * 8).collect();
            let keys = cache.keys(None, &fresh);
            cache.insert(&keys, &fresh, &state(&pool, 4)?, None, None)?;
            if cache.depth(&stale_keys, &stale, false) == 0 {
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
        cache.insert(&keys, &[1; 4], &state(&pool, 4)?, None, None)?;
        assert_eq!(cache.stats().entries, 0);
        assert!(cache.lookup(&keys, &[1; 6], false, |_| true).is_none());
        assert!(
            PrefixConfig {
                spacing: 24,
                entries: 1,
                state_share: None,
                restorable_first: false,
                tool_hold: Duration::ZERO,
            }
            .validate(16)
            .is_err()
        );
        assert!(PrefixConfig::DISABLED.validate(96).is_ok());
        Ok(())
    }
}
