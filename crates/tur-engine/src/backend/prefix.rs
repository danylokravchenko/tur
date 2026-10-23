#![forbid(unsafe_code)]
//! Prefix reuse over exact token boundaries.
//!
//! An entry is a committed [`ModelState`] after the first `n × spacing` tokens of some
//! request: its page table and the recurrent slot written at exactly that boundary. A
//! recurrent state exists only where a pass ended, so the pipeline never lets a prefill
//! chunk cross a multiple of the spacing, and every aligned position of every request is
//! a valid snapshot. Entries share leases with the request that produced them, so a
//! snapshot copies nothing; it keeps one recurrent slot and its pages alive. The spacing
//! is a multiple of the page size, so an adopted snapshot never forces a copy-on-write
//! tail.
//!
//! Keys chain a per-cache keyed hash over aligned blocks, seeded by a namespace (tenant
//! or cache salt), so equal prefixes in different namespaces never meet. A hash hit is
//! used only after comparing every token. The pool pins model identity and state
//! layout, so entries cannot cross models, quantizations or positional settings.
//!
//! Eviction removes the least recently used entry, never-hit entries first. Pages and
//! slots return to the pool only when no request still shares them and no unfenced work
//! references them (see [`tur_metal::pool`]).

use std::{
    collections::HashMap,
    hash::{BuildHasher, RandomState},
    sync::Arc,
};
use tur_core::{Result, execution::invalid};
use tur_metal::state::ModelState;

/// Snapshot spacing and capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixConfig {
    /// Tokens between snapshots; a multiple of the K/V page size.
    pub spacing: usize,
    /// Most snapshots retained; each holds one recurrent slot. Zero disables reuse.
    pub entries: usize,
}

impl PrefixConfig {
    /// Default snapshot spacing in tokens; see the P7 report for the measurement.
    pub const DEFAULT_SPACING: usize = 256;

    /// Disabled: no snapshots and no chunk alignment.
    pub const DISABLED: Self = Self {
        spacing: Self::DEFAULT_SPACING,
        entries: 0,
    };

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
    last_use: u64,
    hits: u64,
}

/// Aligned-block hashes of one request's tokens, extended as it generates.
#[derive(Clone, Debug, Default)]
pub struct PrefixKeys {
    /// Hash of the namespace, which seeds the chain.
    seed: u64,
    /// `keys[i]` identifies the first `(i + 1) × spacing` tokens.
    keys: Vec<u64>,
}

impl PrefixKeys {
    /// Aligned prefixes covered.
    pub fn blocks(&self) -> usize {
        self.keys.len()
    }
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

    /// Chain hashes of every aligned prefix of `tokens` within `namespace`.
    pub fn keys(&self, namespace: Option<&str>, tokens: &[u32]) -> PrefixKeys {
        let mut keys = PrefixKeys {
            seed: self.hasher.hash_one(namespace),
            keys: Vec::new(),
        };
        self.extend(&mut keys, tokens);
        keys
    }

    /// Hash the aligned blocks of `tokens` that `keys` does not cover yet; `tokens`
    /// extends the sequence `keys` was built from.
    pub fn extend(&self, keys: &mut PrefixKeys, tokens: &[u32]) {
        if !self.config.enabled() {
            return;
        }
        let spacing = self.config.spacing;
        let mut key = keys.keys.last().copied().unwrap_or(keys.seed);
        for block in tokens.chunks_exact(spacing).skip(keys.keys.len()) {
            key = self.hasher.hash_one((key, block));
            keys.keys.push(key);
        }
    }

    /// Next aligned position after `position`, where a prefill chunk must end.
    pub fn next_boundary(&self, position: usize) -> Option<usize> {
        self.config
            .enabled()
            .then(|| (position / self.config.spacing + 1) * self.config.spacing)
    }

    /// Whether a commit at `position` is a snapshot boundary.
    pub fn is_boundary(&self, position: usize) -> bool {
        self.config.enabled() && position > 0 && position.is_multiple_of(self.config.spacing)
    }

    /// Deepest verified snapshot of at most `limit` tokens of `tokens`.
    pub fn lookup(
        &mut self,
        keys: &PrefixKeys,
        tokens: &[u32],
        limit: usize,
    ) -> Option<(usize, ModelState)> {
        if !self.config.enabled() {
            return None;
        }
        self.stats.lookups += 1;
        let blocks = (limit / self.config.spacing).min(keys.keys.len());
        for block in (0..blocks).rev() {
            let length = (block + 1) * self.config.spacing;
            let Some(entry) = self.map.get_mut(&keys.keys[block]) else {
                continue;
            };
            if tokens.get(..length) != Some(&entry.tokens[..]) {
                self.stats.collisions += 1;
                continue;
            }
            self.tick += 1;
            entry.last_use = self.tick;
            entry.hits += 1;
            self.stats.hits += 1;
            self.stats.hit_tokens += length as u64;
            return Some((length, entry.state.clone()));
        }
        None
    }

    /// Retain `state`, committed after exactly `tokens`, an aligned prefix whose key is
    /// `keys` block `tokens.len() / spacing - 1`. An existing entry for the same tokens is
    /// refreshed; a colliding one is kept.
    pub fn insert(&mut self, keys: &PrefixKeys, tokens: &[u32], state: &ModelState) -> Result<()> {
        if !self.is_boundary(tokens.len()) || state.position() != tokens.len() {
            return Err(invalid(
                "prefix snapshots must sit at aligned committed boundaries",
            ));
        }
        let Some(&key) = keys.keys.get(tokens.len() / self.config.spacing - 1) else {
            return Err(invalid("prefix keys do not cover the snapshot"));
        };
        self.tick += 1;
        if let Some(entry) = self.map.get_mut(&key) {
            if entry.tokens[..] == *tokens {
                entry.last_use = self.tick;
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
        })
    }

    #[test]
    fn test_lookup_returns_the_deepest_verified_aligned_prefix() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(4);
        let prompt: Vec<u32> = (0..11).collect();
        let mut keys = cache.keys(None, &prompt[..5]);
        assert_eq!(keys.blocks(), 1);
        cache.extend(&mut keys, &prompt);
        assert_eq!(keys.blocks(), 2);
        assert_eq!(keys.keys, cache.keys(None, &prompt).keys);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?)?;
        cache.insert(&keys, &prompt[..8], &state(&pool, 8)?)?;
        assert!(
            cache
                .insert(&keys, &prompt[..6], &state(&pool, 6)?)
                .is_err()
        );
        let (length, hit) = cache
            .lookup(&keys, &prompt, 10)
            .ok_or_else(|| invalid("miss"))?;
        assert_eq!((length, hit.position()), (8, 8));
        // At least one token must remain to produce logits.
        assert_eq!(cache.lookup(&keys, &prompt, 7).map(|h| h.0), Some(4));
        // A different continuation shares only the first block.
        let mut other = prompt.clone();
        other[5] = 15;
        let other_keys = cache.keys(None, &other);
        assert_eq!(cache.lookup(&other_keys, &other, 10).map(|h| h.0), Some(4));
        // Namespaces never share entries.
        let salted = cache.keys(Some("tenant"), &prompt);
        assert!(cache.lookup(&salted, &prompt, 10).is_none());
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.hit_tokens, stats.lookups), (3, 16, 4));
        Ok(())
    }

    #[test]
    fn test_hash_matches_are_verified_against_tokens() -> Result<()> {
        let pool = pool()?;
        let mut cache = cache(2);
        let prompt: Vec<u32> = (0..8).collect();
        let keys = cache.keys(None, &prompt);
        cache.insert(&keys, &prompt[..4], &state(&pool, 4)?)?;
        // Forge a collision: different tokens presented with the stored key.
        let forged: Vec<u32> = (8..16).collect();
        assert!(cache.lookup(&keys, &forged, 7).is_none());
        cache.insert(&keys, &forged[..4], &state(&pool, 4)?)?;
        assert_eq!(cache.stats().collisions, 2);
        assert_eq!(cache.lookup(&keys, &prompt, 7).map(|h| h.0), Some(4));
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
        cache.insert(&keys[0], &prompts[0][..4], &state(&pool, 4)?)?;
        cache.insert(&keys[1], &prompts[1][..4], &state(&pool, 4)?)?;
        assert!(cache.lookup(&keys[0], &prompts[0], 4).is_some());
        // Inserting a third evicts the unused second entry, not the older hit one.
        cache.insert(&keys[2], &prompts[2][..4], &state(&pool, 4)?)?;
        assert!(cache.lookup(&keys[0], &prompts[0], 4).is_some());
        assert!(cache.lookup(&keys[1], &prompts[1], 4).is_none());
        assert_eq!(pool.usage()?.slots.leased, 2);
        cache.clear();
        let usage = pool.usage()?;
        assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
        assert!(!cache.evict());
        Ok(())
    }

    #[test]
    fn test_disabled_cache_never_aligns_or_stores() {
        let mut cache = PrefixCache::new(PrefixConfig::DISABLED);
        assert_eq!(cache.next_boundary(3), None);
        assert!(!cache.is_boundary(256));
        assert_eq!(cache.keys(None, &[1; 600]).blocks(), 0);
        assert!(
            cache
                .lookup(&PrefixKeys::default(), &[1; 600], 599)
                .is_none()
        );
        assert!(
            PrefixConfig {
                spacing: 24,
                entries: 1
            }
            .validate(16)
            .is_err()
        );
        assert!(PrefixConfig::DISABLED.validate(96).is_ok());
    }
}
