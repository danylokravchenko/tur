//! Prefix cache implementation for accelerating repeated prefill operations
//!
//! This module provides a caching mechanism that stores KV cache states from
//! common prompt prefixes, allowing subsequent requests with matching prefixes
//! to skip redundant computation.

use ahash::AHashMap;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

/// Statistics for prefix cache operations
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of cache evictions
    pub evictions: usize,
    /// Total tokens stored in cache
    pub total_tokens_cached: usize,
    /// Total tokens reused from cache
    pub total_tokens_reused: usize,
}

impl CacheStats {
    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// A single entry in the prefix cache
#[derive(Debug, Clone)]
pub struct PrefixCacheEntry {
    /// Token sequence that was cached
    pub tokens: Vec<u32>,
    /// KV cache state per layer: Vec<(K, V)> where index = layer_id
    pub kv_states: Vec<(Tensor, Tensor)>,
    /// When this entry was created
    pub created_at: Instant,
    /// When this entry was last accessed
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed
    pub access_count: usize,
    /// Device where tensors are stored
    pub device: Device,
    /// Data type of tensors
    pub dtype: DType,
}

impl PrefixCacheEntry {
    /// Create a new cache entry
    pub fn new(
        tokens: Vec<u32>,
        kv_states: Vec<(Tensor, Tensor)>,
        device: Device,
        dtype: DType,
    ) -> Self {
        let now = Instant::now();
        Self {
            tokens,
            kv_states,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            device,
            dtype,
        }
    }

    /// Check if this entry is compatible with the given device and dtype
    pub fn is_compatible(&self, device: &Device, dtype: DType) -> bool {
        self.device.same_device(device) && self.dtype == dtype
    }

    /// Update access statistics
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Prefix cache for storing and reusing KV cache states
#[derive(Debug)]
pub struct PrefixCache {
    /// Cache storage: hash(tokens) -> entry
    entries: AHashMap<u64, PrefixCacheEntry>,
    /// LRU tracking queue
    lru_queue: VecDeque<u64>,
    /// Maximum number of entries to store
    max_entries: usize,
    /// Maximum token length to cache
    max_token_length: usize,
    /// Cache statistics
    stats: CacheStats,
}

impl PrefixCache {
    /// Create a new prefix cache
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of cache entries
    /// * `max_token_length` - Maximum token sequence length to cache
    pub fn new(max_entries: usize, max_token_length: usize) -> Self {
        Self {
            entries: AHashMap::new(),
            lru_queue: VecDeque::new(),
            max_entries,
            max_token_length,
            stats: CacheStats::default(),
        }
    }

    /// Compute cache key from token sequence
    fn compute_key(tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Find the longest matching prefix in the cache
    ///
    /// Returns (cache_key, match_length) where match_length is the number of
    /// tokens that match. Returns (0, 0) if no match found.
    pub fn find_longest_prefix(&self, tokens: &[u32]) -> (u64, usize) {
        let mut best_match = (0u64, 0usize);

        // Try progressively shorter prefixes
        for len in (1..=tokens.len().min(self.max_token_length)).rev() {
            let prefix = &tokens[..len];
            let key = Self::compute_key(prefix);

            if let Some(entry) = self.entries.get(&key) {
                // Verify the tokens actually match (hash collision check)
                if entry.tokens == prefix {
                    best_match = (key, len);
                    break;
                }
            }
        }

        best_match
    }

    /// Get a cache entry by key and update access statistics
    pub fn get(&mut self, key: u64) -> Option<&PrefixCacheEntry> {
        if self.entries.contains_key(&key) {
            // Update LRU first
            self.update_lru(key);
            // Then get and update entry
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.mark_accessed();
            }
            // Return immutable reference
            self.entries.get(&key)
        } else {
            None
        }
    }

    /// Get a mutable cache entry by key without updating access statistics
    /// Use this when you need to modify the entry directly
    pub fn get_mut(&mut self, key: u64) -> Option<&mut PrefixCacheEntry> {
        self.entries.get_mut(&key)
    }

    /// Insert a new cache entry
    ///
    /// Returns true if the entry was inserted, false if it was rejected
    /// (e.g., due to length constraints)
    pub fn insert(&mut self, entry: PrefixCacheEntry) -> bool {
        // Check length constraint
        if entry.tokens.len() > self.max_token_length {
            return false;
        }

        let key = Self::compute_key(&entry.tokens);

        // Evict if at capacity and this is a new entry
        if !self.entries.contains_key(&key) && self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        // Update statistics
        self.stats.total_tokens_cached += entry.tokens.len();

        // Insert entry
        self.entries.insert(key, entry);
        self.lru_queue.push_back(key);

        true
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(key) = self.lru_queue.pop_front()
            && let Some(entry) = self.entries.remove(&key)
        {
            self.stats.evictions += 1;
            self.stats.total_tokens_cached = self
                .stats
                .total_tokens_cached
                .saturating_sub(entry.tokens.len());
        }
    }

    /// Update LRU queue for a key
    fn update_lru(&mut self, key: u64) {
        // Remove key from its current position
        if let Some(pos) = self.lru_queue.iter().position(|&k| k == key) {
            self.lru_queue.remove(pos);
        }
        // Add to back (most recently used)
        self.lru_queue.push_back(key);
    }

    /// Record a cache hit
    pub fn record_hit(&mut self, tokens_reused: usize) {
        self.stats.hits += 1;
        self.stats.total_tokens_reused += tokens_reused;
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.stats.misses += 1;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get mutable cache statistics
    pub fn stats_mut(&mut self) -> &mut CacheStats {
        &mut self.stats
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_queue.clear();
        self.stats.total_tokens_cached = 0;
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the maximum number of entries
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Get the maximum token length
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
        let tokens1 = vec![1, 2, 3, 4, 5];
        let tokens2 = vec![1, 2, 3, 4, 5];
        let tokens3 = vec![1, 2, 3, 4, 6];

        let key1 = PrefixCache::compute_key(&tokens1);
        let key2 = PrefixCache::compute_key(&tokens2);
        let key3 = PrefixCache::compute_key(&tokens3);

        assert_eq!(key1, key2, "Same tokens should produce same key");
        assert_ne!(key1, key3, "Different tokens should produce different keys");
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

        // Test find on empty cache
        let (key, len) = cache.find_longest_prefix(&[1, 2, 3]);
        assert_eq!(len, 0);
        assert_eq!(key, 0);
    }

    #[test]
    fn test_cache_length_constraint() {
        let mut cache = PrefixCache::new(10, 5);

        let device = Device::Cpu;
        let dtype = DType::F32;

        // Try to insert entry that's too long
        let long_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let entry = PrefixCacheEntry::new(long_tokens, vec![], device.clone(), dtype);
        assert!(!cache.insert(entry), "Should reject entry that's too long");

        // Insert valid entry
        let valid_tokens = vec![1, 2, 3];
        let entry = PrefixCacheEntry::new(valid_tokens, vec![], device, dtype);
        assert!(cache.insert(entry), "Should accept valid entry");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PrefixCache::new(2, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Insert 3 entries (should evict the first one)
        let entry1 = PrefixCacheEntry::new(vec![1], vec![], device.clone(), dtype);
        let entry2 = PrefixCacheEntry::new(vec![2], vec![], device.clone(), dtype);
        let entry3 = PrefixCacheEntry::new(vec![3], vec![], device.clone(), dtype);

        cache.insert(entry1);
        cache.insert(entry2);
        assert_eq!(cache.len(), 2);

        cache.insert(entry3);
        assert_eq!(cache.len(), 2, "Should maintain max size");
        assert_eq!(cache.stats().evictions, 1);

        // First entry should be evicted
        let (_, len) = cache.find_longest_prefix(&[1]);
        assert_eq!(len, 0, "First entry should be evicted");

        // Second and third should still be there
        let (_, len) = cache.find_longest_prefix(&[2]);
        assert_eq!(len, 1);
        let (_, len) = cache.find_longest_prefix(&[3]);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_prefix_cache_basic_operations() -> Result<()> {
        let mut cache = PrefixCache::new(10, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create dummy KV states
        let k = Tensor::zeros((1, 2, 5, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 5, 64), dtype, &device)?;
        let kv_states = vec![(k, v)];

        // Test insertion
        let tokens = vec![1, 2, 3, 4, 5];
        let entry = PrefixCacheEntry::new(tokens.clone(), kv_states, device.clone(), dtype);
        assert!(cache.insert(entry), "Should insert valid entry");
        assert_eq!(cache.len(), 1);

        // Test lookup
        let (key, match_len) = cache.find_longest_prefix(&tokens);
        assert_eq!(match_len, tokens.len(), "Should find exact match");
        assert_ne!(key, 0, "Should return valid key");

        Ok(())
    }

    #[test]
    fn test_prefix_cache_longest_prefix_matching() -> Result<()> {
        let mut cache = PrefixCache::new(10, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Cache a prefix
        let prefix = vec![1, 2, 3];
        let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let entry = PrefixCacheEntry::new(prefix.clone(), vec![(k, v)], device.clone(), dtype);
        cache.insert(entry);

        // Test exact match
        let (_, len) = cache.find_longest_prefix(&prefix);
        assert_eq!(len, 3, "Should match entire prefix");

        // Test extended sequence
        let extended = vec![1, 2, 3, 4, 5];
        let (_, len) = cache.find_longest_prefix(&extended);
        assert_eq!(len, 3, "Should match prefix portion");

        // Test no match
        let different = vec![10, 20, 30];
        let (_, len) = cache.find_longest_prefix(&different);
        assert_eq!(len, 0, "Should not match different sequence");

        Ok(())
    }

    #[test]
    fn test_prefix_cache_lru_eviction() -> Result<()> {
        let mut cache = PrefixCache::new(2, 100); // Max 2 entries
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Insert 3 entries
        for i in 0..3 {
            let tokens = vec![i * 10, i * 10 + 1, i * 10 + 2];
            let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let entry = PrefixCacheEntry::new(tokens, vec![(k, v)], device.clone(), dtype);
            cache.insert(entry);
        }

        // Verify cache size is limited
        assert_eq!(cache.len(), 2, "Cache should maintain max size");
        assert_eq!(cache.stats().evictions, 1, "Should have evicted 1 entry");

        // First entry should be evicted
        let (_, len) = cache.find_longest_prefix(&[0, 1, 2]);
        assert_eq!(len, 0, "First entry should be evicted");

        // Last two should remain
        let (_, len) = cache.find_longest_prefix(&[10, 11, 12]);
        assert_eq!(len, 3, "Second entry should remain");

        let (_, len) = cache.find_longest_prefix(&[20, 21, 22]);
        assert_eq!(len, 3, "Third entry should remain");

        Ok(())
    }

    #[test]
    fn test_prefix_cache_length_constraint() -> Result<()> {
        let mut cache = PrefixCache::new(10, 5); // Max 5 tokens
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Try to insert sequence longer than max
        let long_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let k = Tensor::zeros((1, 2, 8, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 8, 64), dtype, &device)?;
        let entry = PrefixCacheEntry::new(long_tokens, vec![(k, v)], device.clone(), dtype);

        assert!(!cache.insert(entry), "Should reject long sequence");
        assert_eq!(cache.len(), 0, "Cache should be empty");

        // Insert valid length
        let short_tokens = vec![1, 2, 3];
        let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let entry = PrefixCacheEntry::new(short_tokens, vec![(k, v)], device, dtype);

        assert!(cache.insert(entry), "Should accept short sequence");
        assert_eq!(cache.len(), 1, "Cache should have 1 entry");

        Ok(())
    }

    #[test]
    fn test_prefix_cache_statistics() {
        let mut cache = PrefixCache::new(10, 100);

        // Initial stats
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().hit_rate(), 0.0);

        // Record operations
        cache.record_miss();
        cache.record_hit(5);
        cache.record_hit(10);
        cache.record_miss();

        // Verify stats
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hit_rate(), 0.5);
        assert_eq!(cache.stats().total_tokens_reused, 15);
    }

    #[test]
    fn test_prefix_cache_entry_compatibility() -> Result<()> {
        let device_cpu = Device::Cpu;
        let dtype_f32 = DType::F32;
        let dtype_f16 = DType::F16;

        let k = Tensor::zeros((1, 2, 3, 64), dtype_f32, &device_cpu)?;
        let v = Tensor::zeros((1, 2, 3, 64), dtype_f32, &device_cpu)?;
        let entry =
            PrefixCacheEntry::new(vec![1, 2, 3], vec![(k, v)], device_cpu.clone(), dtype_f32);

        // Test compatibility checks
        assert!(
            entry.is_compatible(&device_cpu, dtype_f32),
            "Should be compatible with same device/dtype"
        );
        assert!(
            !entry.is_compatible(&device_cpu, dtype_f16),
            "Should not be compatible with different dtype"
        );

        Ok(())
    }

    #[test]
    fn test_prefix_cache_shared_access() -> Result<()> {
        let cache = Arc::new(RwLock::new(PrefixCache::new(10, 100)));
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Simulate multiple threads/engines accessing cache
        {
            let mut cache_guard = cache.write();
            let tokens = vec![1, 2, 3];
            let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let entry = PrefixCacheEntry::new(tokens, vec![(k, v)], device.clone(), dtype);
            cache_guard.insert(entry);
        }

        // Read from another "thread"
        {
            let cache_guard = cache.read();
            let (_, len) = cache_guard.find_longest_prefix(&[1, 2, 3]);
            assert_eq!(len, 3, "Should find cached entry");
        }

        // Verify statistics
        {
            let cache_guard = cache.read();
            assert_eq!(cache_guard.len(), 1);
        }

        Ok(())
    }

    #[test]
    fn test_prefix_cache_clear() -> Result<()> {
        let mut cache = PrefixCache::new(10, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Add some entries
        for i in 0..3 {
            let tokens = vec![i, i + 1, i + 2];
            let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
            let entry = PrefixCacheEntry::new(tokens, vec![(k, v)], device.clone(), dtype);
            cache.insert(entry);
        }

        assert_eq!(cache.len(), 3);
        assert!(cache.stats().total_tokens_cached > 0);

        // Clear cache
        cache.clear();

        assert_eq!(cache.len(), 0, "Cache should be empty");
        assert!(cache.is_empty());
        assert_eq!(
            cache.stats().total_tokens_cached,
            0,
            "Token count should be reset"
        );

        Ok(())
    }

    #[test]
    fn test_prefix_cache_access_tracking() -> Result<()> {
        let mut cache = PrefixCache::new(10, 100);
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tokens = vec![1, 2, 3];
        let k = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), dtype, &device)?;
        let entry = PrefixCacheEntry::new(tokens.clone(), vec![(k, v)], device, dtype);
        cache.insert(entry);

        // Get the key
        let (key, _) = cache.find_longest_prefix(&tokens);

        // Access the entry multiple times
        for _ in 0..3 {
            if let Some(entry) = cache.get(key) {
                assert!(entry.access_count > 0, "Access count should increase");
            }
        }

        Ok(())
    }
}
