//! Prefix cache implementation for accelerating repeated prefill operations
//!
//! This module provides a caching mechanism that stores KV cache states from
//! common prompt prefixes, allowing subsequent requests with matching prefixes
//! to skip redundant computation.

use ahash::AHashMap;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

/// Statistics for prefix cache operations
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    /// Total tokens currently held in the cache
    pub total_tokens_cached: usize,
    /// Total tokens whose recomputation was skipped due to cache hits
    pub total_tokens_reused: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// A single entry in the prefix cache
#[derive(Debug, Clone)]
pub struct PrefixCacheEntry {
    pub tokens: Vec<u32>,
    pub kv_states: Vec<(Tensor, Tensor)>,
    pub device: Device,
    pub dtype: DType,
}

impl PrefixCacheEntry {
    pub fn new(
        tokens: Vec<u32>,
        kv_states: Vec<(Tensor, Tensor)>,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            tokens,
            kv_states,
            device,
            dtype,
        }
    }

    pub fn is_compatible(&self, device: &Device, dtype: DType) -> bool {
        self.device.same_device(device) && self.dtype == dtype
    }
}

/// Prefix cache for storing and reusing KV cache states
#[derive(Debug)]
pub struct PrefixCache {
    entries: AHashMap<u64, PrefixCacheEntry>,
    /// Position in this queue = LRU rank; front = least recently used
    lru_queue: VecDeque<u64>,
    max_entries: usize,
    max_token_length: usize,
    stats: CacheStats,
    /// Per-instance hasher seed so keys are consistent within a cache but
    /// not predictable across processes (hash-flooding resistance).
    hasher_state: ahash::RandomState,
}

impl PrefixCache {
    pub fn new(max_entries: usize, max_token_length: usize) -> Self {
        Self {
            entries: AHashMap::new(),
            lru_queue: VecDeque::new(),
            max_entries,
            max_token_length,
            stats: CacheStats::default(),
            hasher_state: ahash::RandomState::new(),
        }
    }

    fn compute_key(&self, tokens: &[u32]) -> u64 {
        let mut h = self.hasher_state.build_hasher();
        tokens.hash(&mut h);
        h.finish()
    }

    /// Find the longest cached prefix of `tokens`.
    ///
    /// Returns `Some((key, match_len))` on a hit, `None` on a miss.
    pub fn find_longest_prefix(&self, tokens: &[u32]) -> Option<(u64, usize)> {
        for len in (1..=tokens.len().min(self.max_token_length)).rev() {
            let prefix = &tokens[..len];
            let key = self.compute_key(prefix);
            if let Some(entry) = self.entries.get(&key) {
                if entry.tokens == prefix {
                    return Some((key, len));
                }
            }
        }
        None
    }

    /// Get a cache entry by key and promote it to the MRU position.
    pub fn get(&mut self, key: u64) -> Option<&PrefixCacheEntry> {
        if !self.entries.contains_key(&key) {
            return None;
        }
        self.update_lru(key);
        self.entries.get(&key)
    }

    /// Get a mutable cache entry without updating LRU order.
    pub fn get_mut(&mut self, key: u64) -> Option<&mut PrefixCacheEntry> {
        self.entries.get_mut(&key)
    }

    /// Insert or update a cache entry.
    ///
    /// Re-inserting an existing key updates the entry in-place and promotes it
    /// to MRU, avoiding ghost entries in the LRU queue and inflated statistics.
    ///
    /// Returns `false` if the entry exceeds `max_token_length` and was rejected.
    pub fn insert(&mut self, entry: PrefixCacheEntry) -> bool {
        if entry.tokens.len() > self.max_token_length {
            return false;
        }

        let key = self.compute_key(&entry.tokens);

        if let Some(old) = self.entries.get(&key) {
            // Update-in-place: correct the token count delta and promote LRU.
            self.stats.total_tokens_cached = self
                .stats
                .total_tokens_cached
                .saturating_sub(old.tokens.len())
                + entry.tokens.len();
            self.entries.insert(key, entry);
            self.update_lru(key);
            return true;
        }

        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        self.stats.total_tokens_cached += entry.tokens.len();
        self.entries.insert(key, entry);
        self.lru_queue.push_back(key);
        true
    }

    /// Evict the least recently used live entry, skipping any ghost keys.
    fn evict_lru(&mut self) {
        while let Some(key) = self.lru_queue.pop_front() {
            if let Some(entry) = self.entries.remove(&key) {
                self.stats.evictions += 1;
                self.stats.total_tokens_cached = self
                    .stats
                    .total_tokens_cached
                    .saturating_sub(entry.tokens.len());
                return;
            }
            // Ghost key (left by a previous duplicate insert) — discard and retry.
        }
    }

    fn update_lru(&mut self, key: u64) {
        if let Some(pos) = self.lru_queue.iter().position(|&k| k == key) {
            self.lru_queue.remove(pos);
        }
        self.lru_queue.push_back(key);
    }

    pub fn record_hit(&mut self, tokens_reused: usize) {
        self.stats.hits += 1;
        self.stats.total_tokens_reused += tokens_reused;
    }

    pub fn record_miss(&mut self) {
        self.stats.misses += 1;
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn stats_mut(&mut self) -> &mut CacheStats {
        &mut self.stats
    }

    /// Clear all entries. Historical hit/miss/eviction counters are preserved;
    /// only `total_tokens_cached` is reset to match the now-empty state.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_queue.clear();
        self.stats.total_tokens_cached = 0;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn max_token_length(&self) -> usize {
        self.max_token_length
    }
}

/// Thread-safe wrapper for PrefixCache
pub type SharedPrefixCache = Arc<RwLock<PrefixCache>>;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Result;

    #[test]
    fn test_cache_key_computation() {
        let cache = PrefixCache::new(10, 100);
        let tokens1 = vec![1u32, 2, 3, 4, 5];
        let tokens2 = vec![1u32, 2, 3, 4, 5];
        let tokens3 = vec![1u32, 2, 3, 4, 6];

        let key1 = cache.compute_key(&tokens1);
        let key2 = cache.compute_key(&tokens2);
        let key3 = cache.compute_key(&tokens3);

        assert_eq!(key1, key2, "same tokens → same key");
        assert_ne!(key1, key3, "different tokens → different keys");
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.hits = 7;
        stats.misses = 3;
        assert_eq!(stats.hit_rate(), 0.7);

        stats.reset();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_prefix_cache_basic() {
        let cache = PrefixCache::new(10, 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.find_longest_prefix(&[1, 2, 3]).is_none());
    }

    #[test]
    fn test_cache_length_constraint() {
        let mut cache = PrefixCache::new(10, 5);
        let device = Device::Cpu;
        let dtype = DType::F32;

        let entry =
            PrefixCacheEntry::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![], device.clone(), dtype);
        assert!(
            !cache.insert(entry),
            "should reject entry over max_token_length"
        );

        let entry = PrefixCacheEntry::new(vec![1, 2, 3], vec![], device, dtype);
        assert!(cache.insert(entry));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PrefixCache::new(2, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        cache.insert(PrefixCacheEntry::new(
            vec![1],
            vec![],
            device.clone(),
            dtype,
        ));
        cache.insert(PrefixCacheEntry::new(
            vec![2],
            vec![],
            device.clone(),
            dtype,
        ));
        assert_eq!(cache.len(), 2);

        cache.insert(PrefixCacheEntry::new(
            vec![3],
            vec![],
            device.clone(),
            dtype,
        ));
        assert_eq!(cache.len(), 2, "should maintain max_entries");
        assert_eq!(cache.stats().evictions, 1);

        assert!(
            cache.find_longest_prefix(&[1]).is_none(),
            "LRU entry should be evicted"
        );
        assert_eq!(cache.find_longest_prefix(&[2]).map(|(_, l)| l), Some(1));
        assert_eq!(cache.find_longest_prefix(&[3]).map(|(_, l)| l), Some(1));
    }

    #[test]
    fn test_duplicate_insert_no_ghost() {
        // Re-inserting the same token sequence must not create ghost LRU entries
        // or inflate total_tokens_cached.
        let mut cache = PrefixCache::new(2, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;
        let tokens = vec![1u32, 2, 3];

        for _ in 0..5 {
            cache.insert(PrefixCacheEntry::new(
                tokens.clone(),
                vec![],
                device.clone(),
                dtype,
            ));
        }

        assert_eq!(cache.len(), 1, "only one logical entry");
        assert_eq!(cache.lru_queue.len(), 1, "no ghost keys in LRU queue");
        assert_eq!(cache.stats().total_tokens_cached, tokens.len());
    }

    #[test]
    fn test_duplicate_insert_does_not_prevent_eviction() {
        // After repeated re-inserts, the cache should still be able to evict
        // to make room for a genuinely new entry.
        let mut cache = PrefixCache::new(2, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        cache.insert(PrefixCacheEntry::new(
            vec![1],
            vec![],
            device.clone(),
            dtype,
        ));
        cache.insert(PrefixCacheEntry::new(
            vec![2],
            vec![],
            device.clone(),
            dtype,
        ));
        // Re-insert [1] multiple times — must not overflow capacity
        for _ in 0..3 {
            cache.insert(PrefixCacheEntry::new(
                vec![1],
                vec![],
                device.clone(),
                dtype,
            ));
        }
        // Adding a new entry [3] should evict [2] (now the LRU)
        cache.insert(PrefixCacheEntry::new(
            vec![3],
            vec![],
            device.clone(),
            dtype,
        ));

        assert_eq!(cache.len(), 2);
        assert!(
            cache.find_longest_prefix(&[2]).is_none(),
            "[2] should be evicted"
        );
        assert!(cache.find_longest_prefix(&[1]).is_some());
        assert!(cache.find_longest_prefix(&[3]).is_some());
    }

    #[test]
    fn test_prefix_cache_basic_operations() -> Result<()> {
        let mut cache = PrefixCache::new(10, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        let k = Tensor::zeros((1, 2, 5, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 5, 64), dtype, &device)?;
        let tokens = vec![1u32, 2, 3, 4, 5];
        let entry = PrefixCacheEntry::new(tokens.clone(), vec![(k, v)], device.clone(), dtype);
        assert!(cache.insert(entry));
        assert_eq!(cache.len(), 1);

        let (_, match_len) = cache.find_longest_prefix(&tokens).expect("exact match");
        assert_eq!(match_len, tokens.len());

        Ok(())
    }

    #[test]
    fn test_prefix_cache_longest_prefix_matching() -> Result<()> {
        let mut cache = PrefixCache::new(10, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        let prefix = vec![1u32, 2, 3];
        let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        cache.insert(PrefixCacheEntry::new(
            prefix.clone(),
            vec![(k, v)],
            device.clone(),
            dtype,
        ));

        assert_eq!(cache.find_longest_prefix(&prefix).map(|(_, l)| l), Some(3));
        assert_eq!(
            cache.find_longest_prefix(&[1, 2, 3, 4, 5]).map(|(_, l)| l),
            Some(3)
        );
        assert!(cache.find_longest_prefix(&[10, 20, 30]).is_none());

        Ok(())
    }

    #[test]
    fn test_prefix_cache_lru_eviction() -> Result<()> {
        let mut cache = PrefixCache::new(2, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        for i in 0..3u32 {
            let tokens = vec![i * 10, i * 10 + 1, i * 10 + 2];
            let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            cache.insert(PrefixCacheEntry::new(
                tokens,
                vec![(k, v)],
                device.clone(),
                dtype,
            ));
        }

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
        assert!(
            cache.find_longest_prefix(&[0, 1, 2]).is_none(),
            "first entry evicted"
        );
        assert!(cache.find_longest_prefix(&[10, 11, 12]).is_some());
        assert!(cache.find_longest_prefix(&[20, 21, 22]).is_some());

        Ok(())
    }

    #[test]
    fn test_prefix_cache_length_constraint() -> Result<()> {
        let mut cache = PrefixCache::new(10, 5);
        let device = Device::Cpu;
        let dtype = DType::F32;

        let k = Tensor::zeros((1, 2, 8, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 8, 64), dtype, &device)?;
        let entry = PrefixCacheEntry::new(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![(k, v)],
            device.clone(),
            dtype,
        );
        assert!(!cache.insert(entry));
        assert_eq!(cache.len(), 0);

        let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let entry = PrefixCacheEntry::new(vec![1, 2, 3], vec![(k, v)], device, dtype);
        assert!(cache.insert(entry));
        assert_eq!(cache.len(), 1);

        Ok(())
    }

    #[test]
    fn test_prefix_cache_statistics() {
        let mut cache = PrefixCache::new(10, 100);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().hit_rate(), 0.0);

        cache.record_miss();
        cache.record_hit(5);
        cache.record_hit(10);
        cache.record_miss();

        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hit_rate(), 0.5);
        assert_eq!(cache.stats().total_tokens_reused, 15);
    }

    #[test]
    fn test_prefix_cache_entry_compatibility() -> Result<()> {
        let device = Device::Cpu;
        let k = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        let entry = PrefixCacheEntry::new(vec![1, 2, 3], vec![(k, v)], device.clone(), DType::F32);

        assert!(entry.is_compatible(&device, DType::F32));
        assert!(!entry.is_compatible(&device, DType::F16));

        Ok(())
    }

    #[test]
    fn test_prefix_cache_shared_access() -> Result<()> {
        let shared = Arc::new(RwLock::new(PrefixCache::new(10, 100)));
        let device = Device::Cpu;
        let dtype = DType::F32;

        {
            let mut cache = shared.write();
            let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            cache.insert(PrefixCacheEntry::new(
                vec![1, 2, 3],
                vec![(k, v)],
                device.clone(),
                dtype,
            ));
        }

        {
            let cache = shared.read();
            assert_eq!(
                cache.find_longest_prefix(&[1, 2, 3]).map(|(_, l)| l),
                Some(3)
            );
            assert_eq!(cache.len(), 1);
        }

        Ok(())
    }

    #[test]
    fn test_prefix_cache_clear() -> Result<()> {
        let mut cache = PrefixCache::new(10, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        for i in 0..3u32 {
            let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            cache.insert(PrefixCacheEntry::new(
                vec![i, i + 1, i + 2],
                vec![(k, v)],
                device.clone(),
                dtype,
            ));
        }

        assert_eq!(cache.len(), 3);
        assert!(cache.stats().total_tokens_cached > 0);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_tokens_cached, 0);
        // Historical stats (hits, misses, evictions) are intentionally preserved across clear().

        Ok(())
    }
}
