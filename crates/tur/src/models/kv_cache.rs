//! KV Cache implementations for transformer models
//!
//! This module provides two KV cache implementations:
//! - `SimpleKvCache`: Contiguous memory, optimal for single requests
//! - `PagedKvCache`: Block-based memory, optimal for batching and prefix sharing

use ahash::AHashMap;
use candle_core::{Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use parking_lot::RwLock;
use std::sync::Arc;

/// Cache strategy for KV cache management
pub enum CacheStrategy {
    /// Simple contiguous cache - model owns the cache (single request mode)
    /// The model maintains a SimpleKvCache for accumulating KV state
    Simple(SimpleKvCache),
    /// Paged cache - requests own block tables, model uses shared allocator (batching mode)
    /// The model receives block tables as parameters and uses the shared allocator
    Paged(Arc<RwLock<BlockAllocator>>),
}

impl std::fmt::Debug for CacheStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple(_) => f.debug_tuple("Simple").field(&"SimpleKvCache").finish(),
            Self::Paged(_) => f
                .debug_tuple("Paged")
                .field(&"Arc<RwLock<BlockAllocator>>")
                .finish(),
        }
    }
}

impl Clone for CacheStrategy {
    fn clone(&self) -> Self {
        match self {
            Self::Simple(cache) => Self::Simple(cache.clone()),
            Self::Paged(allocator) => Self::Paged(allocator.clone()),
        }
    }
}

/// Trait for KV cache implementations
pub trait KvCacheImpl: Send + Sync {
    /// Append new K and V tensors to the cache
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>;

    /// Get the current accumulated KV cache state
    fn get_state(&self) -> Option<(Tensor, Tensor)>;

    /// Restore KV cache state from previously saved tensors
    fn set_state(&mut self, k: Tensor, v: Tensor) -> Result<()>;

    /// Reset the cache, clearing all accumulated state
    fn reset(&mut self);

    /// Check if the cache has any accumulated state
    fn is_empty(&self) -> bool;
}

/// Simple KV cache using contiguous memory
///
/// This is a wrapper around candle's ConcatKvCache that tracks accumulated KV state
/// for prefix caching support. Optimal for single-request inference.
#[derive(Debug, Clone)]
pub struct SimpleKvCache {
    /// The underlying candle KV cache
    inner: ConcatKvCache,
    /// Cached K tensor (accumulated across all appends)
    cached_k: Option<Tensor>,
    /// Cached V tensor (accumulated across all appends)
    cached_v: Option<Tensor>,
}

impl SimpleKvCache {
    /// Create a new simple KV cache
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate (typically 2 for sequence dimension)
    pub fn new(dim: usize) -> Self {
        Self {
            inner: ConcatKvCache::new(dim),
            cached_k: None,
            cached_v: None,
        }
    }
}

impl KvCacheImpl for SimpleKvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Append to inner cache
        let (k_out, v_out) = self.inner.append(k, v)?;

        // Update our tracked state - Tensor::clone is cheap (Arc-based)
        self.cached_k = Some(k_out.clone());
        self.cached_v = Some(v_out.clone());

        Ok((k_out, v_out))
    }

    fn get_state(&self) -> Option<(Tensor, Tensor)> {
        match (&self.cached_k, &self.cached_v) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }

    fn set_state(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        // Reset the inner cache
        self.inner.reset();

        // Append the restored state to the inner cache
        self.inner.append(&k, &v)?;

        // Store the tensors
        self.cached_k = Some(k);
        self.cached_v = Some(v);

        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.cached_k = None;
        self.cached_v = None;
    }

    fn is_empty(&self) -> bool {
        self.cached_k.is_none() && self.cached_v.is_none()
    }
}

// Type alias for backward compatibility
pub type KvCache = SimpleKvCache;

/// Unique identifier for a memory block
pub type BlockId = usize;

/// Physical memory block storing KV tensors for a fixed number of tokens
///
/// Tensors are `None` until the first write so that the block can be created
/// without knowing the per-layer shape (batch, num_heads, block_size, head_dim).
/// The first call to `PagedKvCache::append` that touches this block will
/// allocate tensors with the correct shape — avoiding the shape-mismatch
/// re-initialization that occurred when a global `num_heads`/`head_dim` was
/// baked in at construction time.
#[derive(Debug, Clone)]
pub struct KvBlock {
    /// Unique block identifier
    pub id: BlockId,
    /// K tensor: [batch, num_heads, block_size, head_dim] — None until first write
    pub k: Option<Tensor>,
    /// V tensor: [batch, num_heads, block_size, head_dim] — None until first write
    pub v: Option<Tensor>,
    /// Reference count for copy-on-write sharing
    pub ref_count: usize,
}

impl KvBlock {
    /// Create a new KV block. Tensors are allocated lazily on the first write.
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            k: None,
            v: None,
            ref_count: 1,
        }
    }
}

/// Block allocator managing physical memory blocks
#[derive(Debug)]
pub struct BlockAllocator {
    /// Pool of free block IDs
    free_blocks: Vec<BlockId>,
    /// Map of allocated blocks
    allocated_blocks: AHashMap<BlockId, KvBlock>,
    /// Next block ID to allocate
    next_block_id: BlockId,
    /// Block size (tokens per block)
    block_size: usize,

    /// Total number of blocks in pool
    total_blocks: usize,
}

impl BlockAllocator {
    /// Create a new block allocator
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            free_blocks: Vec::with_capacity(total_blocks),
            allocated_blocks: AHashMap::new(),
            next_block_id: 0,
            block_size,
            total_blocks,
        }
    }

    /// Allocate a new block
    pub fn allocate(&mut self) -> Result<BlockId> {
        // Try to reuse a free block first
        if let Some(block_id) = self.free_blocks.pop()
            && let Some(block) = self.allocated_blocks.get_mut(&block_id)
        {
            block.ref_count = 1;
            // Clear stale tensors so a reused block doesn't carry data from a
            // previous request into a new one.
            block.k = None;
            block.v = None;
            return Ok(block_id);
        }

        // No free blocks, create a new one if under limit
        if self.allocated_blocks.len() >= self.total_blocks {
            candle_core::bail!("Out of memory: all {} blocks allocated", self.total_blocks);
        }

        let block_id = self.next_block_id;
        self.next_block_id += 1;

        self.allocated_blocks
            .insert(block_id, KvBlock::new(block_id));
        Ok(block_id)
    }

    /// Free a block (decrement ref count, add to free list if reaches 0)
    pub fn free(&mut self, block_id: BlockId) -> Result<()> {
        if let Some(block) = self.allocated_blocks.get_mut(&block_id) {
            if block.ref_count > 0 {
                block.ref_count -= 1;
                if block.ref_count == 0 {
                    self.free_blocks.push(block_id);
                }
            }
            Ok(())
        } else {
            candle_core::bail!("Block {} not found", block_id);
        }
    }

    /// Increment reference count for a block (for copy-on-write)
    pub fn increment_ref(&mut self, block_id: BlockId) -> Result<()> {
        if let Some(block) = self.allocated_blocks.get_mut(&block_id) {
            block.ref_count += 1;
            Ok(())
        } else {
            candle_core::bail!("Block {} not found", block_id);
        }
    }

    /// Get a reference to a block
    pub fn get_block(&self, block_id: BlockId) -> Result<&KvBlock> {
        self.allocated_blocks
            .get(&block_id)
            .ok_or_else(|| candle_core::Error::Msg(format!("Block {} not found", block_id)))
    }

    /// Get a mutable reference to a block
    pub fn get_block_mut(&mut self, block_id: BlockId) -> Result<&mut KvBlock> {
        self.allocated_blocks
            .get_mut(&block_id)
            .ok_or_else(|| candle_core::Error::Msg(format!("Block {} not found", block_id)))
    }

    /// Get number of allocated blocks
    pub fn num_allocated(&self) -> usize {
        self.allocated_blocks.len() - self.free_blocks.len()
    }

    /// Get number of free blocks
    pub fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Paged KV cache with block-based memory management
///
/// This implementation uses fixed-size blocks for memory allocation, enabling:
/// - Efficient memory usage with variable-length sequences
/// - Copy-on-write for prefix sharing
/// - Better support for batched inference
pub struct PagedKvCache {
    /// Shared block allocator
    allocator: Arc<RwLock<BlockAllocator>>,
    /// Block table: maps logical token positions to physical blocks
    /// Blocks table is intentionally not shared, it has a copy-on-write semantics
    block_table: Vec<BlockId>,
    /// Current sequence length
    seq_len: usize,
    /// Block size (tokens per block)
    block_size: usize,
    /// Concatenation dimension (typically 2 for sequence dimension)
    concat_dim: usize,
    /// Running accumulated K tensor — grown by a single `cat` per `append` call.
    /// Avoids the O(n_blocks) re-concatenation that `get_state` previously performed
    /// on every decode step (which caused ~67 s/token regressions at 1500+ tokens).
    k_full: Option<Tensor>,
    /// Running accumulated V tensor — same rationale as `k_full`.
    v_full: Option<Tensor>,
}

impl PagedKvCache {
    /// Create a new paged KV cache
    pub fn new(allocator: Arc<RwLock<BlockAllocator>>, concat_dim: usize) -> Self {
        let block_size = allocator.read().block_size();
        Self {
            allocator,
            block_table: Vec::new(),
            seq_len: 0,
            block_size,
            concat_dim,
            k_full: None,
            v_full: None,
        }
    }

    /// Write `new_tokens` from `k`/`v` into the physical block table.
    /// Called by `append` to keep the block table in sync for prefix-cache
    /// restore; the hot path for attention uses `k_full`/`v_full` directly.
    fn write_to_blocks(&mut self, k: &Tensor, v: &Tensor, new_tokens: usize) -> Result<()> {
        let (b, h, _, d) = k.dims4()?;
        let mut token_offset = 0;
        while token_offset < new_tokens {
            let current_pos = self.seq_len + token_offset;
            let block_idx = current_pos / self.block_size;
            let pos_in_block = current_pos % self.block_size;
            let tokens_in_block = (self.block_size - pos_in_block).min(new_tokens - token_offset);

            let k_slice = k.narrow(self.concat_dim, token_offset, tokens_in_block)?;
            let v_slice = v.narrow(self.concat_dim, token_offset, tokens_in_block)?;

            // Copy-on-write: if this block is shared, fork it before writing.
            let block_id = {
                let allocator = self.allocator.read();
                let current_block_id = self.block_table[block_idx];
                let ref_count = allocator.get_block(current_block_id)?.ref_count;
                if ref_count > 1 {
                    drop(allocator);
                    let mut allocator = self.allocator.write();
                    let old_block = allocator.get_block(current_block_id)?;
                    let old_k = old_block.k.clone();
                    let old_v = old_block.v.clone();
                    let new_block_id = allocator.allocate()?;
                    let new_block = allocator.get_block_mut(new_block_id)?;
                    new_block.k = old_k;
                    new_block.v = old_v;
                    allocator.free(current_block_id)?;
                    self.block_table[block_idx] = new_block_id;
                    new_block_id
                } else {
                    current_block_id
                }
            };

            let mut allocator = self.allocator.write();
            let block = allocator.get_block_mut(block_id)?;

            let expected_shape = [b, h, self.block_size, d];
            if block
                .k
                .as_ref()
                .map(|t| t.dims() != expected_shape)
                .unwrap_or(true)
            {
                block.k = Some(Tensor::zeros(&expected_shape, k.dtype(), k.device())?);
                block.v = Some(Tensor::zeros(&expected_shape, v.dtype(), v.device())?);
            }

            let block_k = block.k.as_ref().unwrap();
            let block_v = block.v.as_ref().unwrap();

            let mut k_parts = Vec::new();
            if pos_in_block > 0 {
                k_parts.push(block_k.narrow(self.concat_dim, 0, pos_in_block)?);
            }
            k_parts.push(k_slice.clone());
            if pos_in_block + tokens_in_block < self.block_size {
                k_parts.push(block_k.narrow(
                    self.concat_dim,
                    pos_in_block + tokens_in_block,
                    self.block_size - pos_in_block - tokens_in_block,
                )?);
            }
            block.k = Some(Tensor::cat(&k_parts, self.concat_dim)?);

            let mut v_parts = Vec::new();
            if pos_in_block > 0 {
                v_parts.push(block_v.narrow(self.concat_dim, 0, pos_in_block)?);
            }
            v_parts.push(v_slice.clone());
            if pos_in_block + tokens_in_block < self.block_size {
                v_parts.push(block_v.narrow(
                    self.concat_dim,
                    pos_in_block + tokens_in_block,
                    self.block_size - pos_in_block - tokens_in_block,
                )?);
            }
            block.v = Some(Tensor::cat(&v_parts, self.concat_dim)?);

            token_offset += tokens_in_block;
        }
        Ok(())
    }

    /// Allocate blocks for new tokens
    fn allocate_blocks(&mut self, num_tokens: usize) -> Result<()> {
        let total_tokens = self.seq_len + num_tokens;
        let blocks_needed = total_tokens.div_ceil(self.block_size);

        let mut allocator = self.allocator.write();
        while self.block_table.len() < blocks_needed {
            let block_id = allocator.allocate()?;
            self.block_table.push(block_id);
        }

        Ok(())
    }

    /// Fork this cache for copy-on-write (for prefix sharing)
    pub fn fork(&self) -> Result<Self> {
        let mut allocator = self.allocator.write();

        // Increment reference counts for all blocks
        for &block_id in &self.block_table {
            allocator.increment_ref(block_id)?;
        }

        Ok(Self {
            allocator: self.allocator.clone(),
            // cloning is acceptable
            // For 2048 tokens with block_size=64: 32 blocks × 8 bytes = 256 bytes per fork
            // For 8192 tokens: 128 blocks × 8 bytes = 1 KB per fork
            // Physical blocks are NOT duplicated - they're shared via reference counting in the allocator
            block_table: self.block_table.clone(),
            seq_len: self.seq_len,
            block_size: self.block_size,
            concat_dim: self.concat_dim,
            // Tensor::clone is Arc-based — no GPU memory is copied here.
            k_full: self.k_full.clone(),
            v_full: self.v_full.clone(),
        })
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get block table (for prefix caching)
    pub fn block_table(&self) -> &[BlockId] {
        &self.block_table
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.block_table.len()
    }
}

impl KvCacheImpl for PagedKvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, new_tokens, _) = k.dims4()?;

        // Update block table for prefix-sharing metadata (lightweight bookkeeping).
        self.allocate_blocks(new_tokens)?;
        // Write new tokens into their physical blocks so prefix-cache restore works.
        self.write_to_blocks(k, v, new_tokens)?;

        // Grow the running full-sequence tensor with a single cat.
        // This replaces the previous O(n_blocks) reassembly in get_state that ran
        // every decode step and caused ~67 s/token at 1500+ token sequences.
        let k_out = match &self.k_full {
            None => k.clone(),
            Some(prev) => Tensor::cat(&[prev, k], self.concat_dim)?,
        };
        let v_out = match &self.v_full {
            None => v.clone(),
            Some(prev) => Tensor::cat(&[prev, v], self.concat_dim)?,
        };

        self.k_full = Some(k_out.clone());
        self.v_full = Some(v_out.clone());
        self.seq_len += new_tokens;

        Ok((k_out, v_out))
    }

    fn get_state(&self) -> Option<(Tensor, Tensor)> {
        Some((self.k_full.clone()?, self.v_full.clone()?))
    }

    fn set_state(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        self.reset();
        self.append(&k, &v)?;
        Ok(())
    }

    fn reset(&mut self) {
        let mut allocator = self.allocator.write();
        for &block_id in &self.block_table {
            let _ = allocator.free(block_id);
        }
        self.block_table.clear();
        self.seq_len = 0;
        self.k_full = None;
        self.v_full = None;
    }

    fn is_empty(&self) -> bool {
        self.block_table.is_empty()
    }
}

impl Drop for PagedKvCache {
    fn drop(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    #[test]
    fn test_simple_kv_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = SimpleKvCache::new(2);

        assert!(cache.is_empty());
        assert!(cache.get_state().is_none());

        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;

        let (k_out, v_out) = cache.append(&k1, &v1)?;
        assert_eq!(k_out.dims(), &[1, 2, 3, 4]);
        assert_eq!(v_out.dims(), &[1, 2, 3, 4]);

        assert!(!cache.is_empty());
        assert!(cache.get_state().is_some());

        Ok(())
    }

    #[test]
    fn test_block_allocator_basic() -> Result<()> {
        let mut allocator = BlockAllocator::new(10, 64);

        let block_id = allocator.allocate()?;
        assert_eq!(allocator.num_allocated(), 1);

        allocator.free(block_id)?;
        assert_eq!(allocator.num_free(), 1);

        let block_id2 = allocator.allocate()?;
        assert_eq!(block_id, block_id2);

        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_basic() -> Result<()> {
        let allocator = Arc::new(RwLock::new(BlockAllocator::new(10, 64)));

        let cache = PagedKvCache::new(allocator, 2);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);

        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_fork_independent_growth() -> Result<()> {
        let device = Device::Cpu;
        let allocator = Arc::new(RwLock::new(BlockAllocator::new(10, 4)));

        // Create initial cache and add prefix
        let mut cache1 = PagedKvCache::new(allocator.clone(), 2);
        let k_prefix = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let v_prefix = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        cache1.append(&k_prefix, &v_prefix)?;

        assert_eq!(cache1.num_blocks(), 1);
        assert_eq!(cache1.seq_len(), 3);

        // Fork the cache
        let mut cache2 = cache1.fork()?;
        assert_eq!(cache2.num_blocks(), 1);
        assert_eq!(cache2.seq_len(), 3);

        // Verify blocks are initially shared (same block ID)
        let original_block_id = cache1.block_table()[0];
        assert_eq!(
            cache1.block_table()[0],
            cache2.block_table()[0],
            "Blocks should be shared immediately after fork"
        );

        // Verify ref counts increased
        {
            let alloc = allocator.read();
            let block = alloc.get_block(original_block_id)?;
            assert_eq!(
                block.ref_count, 2,
                "Block should have ref_count=2 after fork"
            );
        }

        // Append to cache1 - should trigger copy-on-write for the shared block
        let k_new1 = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;
        let v_new1 = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;
        cache1.append(&k_new1, &v_new1)?;

        assert_eq!(cache1.num_blocks(), 2, "cache1 should have 2 blocks");
        assert_eq!(cache1.seq_len(), 5, "cache1 should have 5 tokens");
        assert_eq!(cache2.num_blocks(), 1, "cache2 should still have 1 block");
        assert_eq!(cache2.seq_len(), 3, "cache2 should still have 3 tokens");

        // Verify copy-on-write happened: cache1's first block should be different now
        assert_ne!(
            cache1.block_table()[0],
            original_block_id,
            "cache1's first block should be different after copy-on-write"
        );

        // Verify cache2 still has the original block (hasn't been modified yet)
        assert_eq!(
            cache2.block_table()[0],
            original_block_id,
            "cache2 should still have the original shared block"
        );

        // Verify ref count on original block decreased back to 1
        {
            let alloc = allocator.read();
            let block = alloc.get_block(original_block_id)?;
            assert_eq!(
                block.ref_count, 1,
                "Original block should have ref_count=1 after cache1's copy-on-write"
            );
        }

        // Now append to cache2 - should also trigger copy-on-write
        let k_new2 = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let v_new2 = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        cache2.append(&k_new2, &v_new2)?;

        assert_eq!(cache1.num_blocks(), 2, "cache1 should still have 2 blocks");
        assert_eq!(cache1.seq_len(), 5, "cache1 should still have 5 tokens");
        assert_eq!(cache2.num_blocks(), 2, "cache2 should now have 2 blocks");
        assert_eq!(cache2.seq_len(), 6, "cache2 should have 6 tokens");

        // Verify cache2 kept the original block (no copy-on-write needed since ref_count=1)
        assert_eq!(
            cache2.block_table()[0],
            original_block_id,
            "cache2 should keep the original block (it's the sole owner after cache1's copy-on-write)"
        );

        // Verify the caches have different first blocks
        assert_ne!(
            cache1.block_table()[0],
            cache2.block_table()[0],
            "First blocks should be different (cache1 did copy-on-write)"
        );

        // Verify the caches have different second blocks (independently allocated)
        assert_ne!(
            cache1.block_table()[1],
            cache2.block_table()[1],
            "Second blocks should be different (independently allocated)"
        );

        // Verify the original block is still owned by cache2
        {
            let alloc = allocator.read();
            let block = alloc.get_block(original_block_id)?;
            assert_eq!(
                block.ref_count, 1,
                "Original block should still have ref_count=1 (owned by cache2)"
            );
        }

        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_fork_ref_counting() -> Result<()> {
        let device = Device::Cpu;
        let allocator = Arc::new(RwLock::new(BlockAllocator::new(10, 4)));

        // Create and populate cache
        let mut cache1 = PagedKvCache::new(allocator.clone(), 2);
        let k = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        cache1.append(&k, &v)?;

        let block_id = cache1.block_table()[0];

        // Verify initial ref count
        {
            let alloc = allocator.read();
            assert_eq!(alloc.get_block(block_id)?.ref_count, 1);
        }

        // Fork increases ref count
        let cache2 = cache1.fork()?;
        {
            let alloc = allocator.read();
            assert_eq!(alloc.get_block(block_id)?.ref_count, 2);
        }

        // Drop cache1, ref count should decrease
        drop(cache1);
        {
            let alloc = allocator.read();
            assert_eq!(alloc.get_block(block_id)?.ref_count, 1);
        }

        // Drop cache2, block should be freed
        drop(cache2);
        {
            let alloc = allocator.read();
            assert_eq!(alloc.num_free(), 1, "Block should be in free list");
        }

        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_fork_memory_sharing() -> Result<()> {
        let device = Device::Cpu;
        let allocator = Arc::new(RwLock::new(BlockAllocator::new(10, 4)));

        // Create cache with prefix
        let mut cache1 = PagedKvCache::new(allocator.clone(), 2);
        let k_prefix = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let v_prefix = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        cache1.append(&k_prefix, &v_prefix)?;

        let initial_allocated = allocator.read().num_allocated();

        // Fork multiple times
        let cache2 = cache1.fork()?;
        let cache3 = cache1.fork()?;
        let cache4 = cache2.fork()?;

        // All forks share the same physical block
        let final_allocated = allocator.read().num_allocated();
        assert_eq!(
            initial_allocated, final_allocated,
            "No new blocks should be allocated during fork"
        );

        // Verify all caches point to same block
        let block_id = cache1.block_table()[0];
        assert_eq!(cache2.block_table()[0], block_id);
        assert_eq!(cache3.block_table()[0], block_id);
        assert_eq!(cache4.block_table()[0], block_id);

        // Verify ref count: cache1 + cache2 + cache3 + cache4 = 4 references
        {
            let alloc = allocator.read();
            assert_eq!(
                alloc.get_block(block_id)?.ref_count,
                4,
                "Block should have ref_count=4 (cache1, cache2, cache3, cache4)"
            );
        }

        Ok(())
    }

    #[test]
    fn test_block_allocator_ref_counting() -> Result<()> {
        let mut allocator = BlockAllocator::new(10, 64);

        let block_id = allocator.allocate()?;
        assert_eq!(allocator.get_block(block_id)?.ref_count, 1);

        // Increment ref count
        allocator.increment_ref(block_id)?;
        assert_eq!(allocator.get_block(block_id)?.ref_count, 2);

        allocator.increment_ref(block_id)?;
        assert_eq!(allocator.get_block(block_id)?.ref_count, 3);

        // Free should decrement
        allocator.free(block_id)?;
        assert_eq!(allocator.get_block(block_id)?.ref_count, 2);
        assert_eq!(
            allocator.num_free(),
            0,
            "Block should not be in free list yet"
        );

        allocator.free(block_id)?;
        assert_eq!(allocator.get_block(block_id)?.ref_count, 1);
        assert_eq!(allocator.num_free(), 0);

        // Final free should add to free list
        allocator.free(block_id)?;
        assert_eq!(allocator.get_block(block_id)?.ref_count, 0);
        assert_eq!(allocator.num_free(), 1, "Block should be in free list");

        Ok(())
    }
}
