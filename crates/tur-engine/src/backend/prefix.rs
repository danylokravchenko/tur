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
//! Eviction removes the least recently used entry, never-hit entries first. Pages and
//! slots return to the pool only when no request still shares them and no unfenced work
//! references them (see [`tur_metal::pool`]).

use candle_core::Tensor;
use std::{
    collections::HashMap,
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
}

impl PrefixConfig {
    /// Default snapshot spacing in tokens; see the P7 report for the measurement.
    pub const DEFAULT_SPACING: usize = 256;

    /// Disabled: no snapshots and no chunk alignment.
    pub const DISABLED: Self = Self {
        spacing: Self::DEFAULT_SPACING,
        entries: 0,
        state_share: None,
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
    state: ModelState,
    /// `[1, vocab]` model logits after the last token, for prompt-end snapshots.
    logits: Option<Tensor>,
    last_use: u64,
    hits: u64,
}

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
    tick: u64,
    stats: PrefixStats,
}

impl PrefixCache {
    pub fn new(config: PrefixConfig) -> Self {
        Self {
            config,
            hasher: RandomState::new(),
            map: HashMap::with_capacity(config.entries),
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
        let longest = tokens.len().min(keys.keys.len());
        for length in (1..=longest).rev() {
            let Some(entry) = self.map.get_mut(&keys.keys[length - 1]) else {
                continue;
            };
            if tokens.get(..length) != Some(&entry.tokens[..]) {
                self.stats.collisions += 1;
                continue;
            }
            if !accept(length) || (length == tokens.len() && entry.logits.is_none()) {
                continue;
            }
            self.tick += 1;
            entry.last_use = self.tick;
            entry.hits += 1;
            self.stats.hits += 1;
            self.stats.hit_tokens += length as u64;
            return Some(Restored {
                length,
                state: entry.state.clone(),
                logits: entry.logits.clone(),
            });
        }
        None
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
        self.tick += 1;
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
                state: state.clone(),
                logits: logits.cloned(),
                last_use: self.tick,
                hits: 0,
            },
        );
        self.stats.inserted += 1;
        Ok(())
    }

    /// Drop the least valuable entry: never-hit before hit, then least recently used.
    /// Returns whether one was dropped.
    pub fn evict(&mut self) -> bool {
        let Some(key) = self
            .map
            .iter()
            .min_by_key(|(_, e)| (e.hits > 0, e.last_use))
            .map(|(key, _)| *key)
        else {
            return false;
        };
        self.map.remove(&key);
        self.stats.evicted += 1;
        true
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.stats.evicted += self.map.len() as u64;
        self.map.clear();
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
            }
            .validate(16)
            .is_err()
        );
        assert!(PrefixConfig::DISABLED.validate(96).is_ok());
        Ok(())
    }
}
