//! Memory pool for managing GPU/CPU memory allocation
//!
//! This module provides memory management for paged attention, including:
//! - Block-based memory tracking
//! - Admission control based on memory availability
//! - Memory pressure detection

/// Memory pool for managing block allocation
pub struct MemoryPool {
    /// Total memory capacity (in blocks)
    total_blocks: usize,

    /// Currently allocated blocks
    allocated_blocks: usize,

    /// Block size (tokens per block)
    pub block_size: usize,

    /// Memory usage threshold for admission control (0.0-1.0).
    /// New requests are only admitted if usage < admission_threshold.
    admission_threshold: f32,

    /// Memory usage threshold for eviction (0.0-1.0).
    /// Eviction is triggered when usage > eviction_threshold.
    eviction_threshold: f32,

    /// Precomputed integer block limit for admission (avoids repeated f32 conversion on hot path)
    admission_limit: usize,

    /// Precomputed integer block limit for eviction
    eviction_limit: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    ///
    /// # Arguments
    /// * `total_memory_bytes` - Total memory available in bytes
    /// * `block_size` - Number of tokens per block
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    /// * `bytes_per_element` - Bytes per tensor element (e.g., 2 for BF16, 4 for F32)
    pub fn new(
        total_memory_bytes: usize,
        block_size: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        bytes_per_element: usize,
    ) -> Self {
        // Each block stores K and V for all layers: 2 × num_layers × num_heads × block_size × head_dim × bytes_per_element
        // Use checked_mul to avoid silent overflow on pathological inputs.
        let bytes_per_block = 2usize
            .checked_mul(num_layers)
            .and_then(|v| v.checked_mul(num_heads))
            .and_then(|v| v.checked_mul(block_size))
            .and_then(|v| v.checked_mul(head_dim))
            .and_then(|v| v.checked_mul(bytes_per_element))
            .unwrap_or(0);

        let total_blocks = if bytes_per_block > 0 {
            total_memory_bytes / bytes_per_block
        } else {
            0
        };

        const DEFAULT_ADMISSION: f32 = 0.9;
        const DEFAULT_EVICTION: f32 = 0.95;

        Self {
            total_blocks,
            allocated_blocks: 0,
            block_size,
            admission_threshold: DEFAULT_ADMISSION,
            eviction_threshold: DEFAULT_EVICTION,
            admission_limit: Self::compute_limit(total_blocks, DEFAULT_ADMISSION),
            eviction_limit: Self::compute_limit(total_blocks, DEFAULT_EVICTION),
        }
    }

    /// Create a memory pool with custom thresholds.
    ///
    /// `admission_threshold` must be strictly less than `eviction_threshold`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_thresholds(
        total_memory_bytes: usize,
        block_size: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        bytes_per_element: usize,
        admission_threshold: f32,
        eviction_threshold: f32,
    ) -> Self {
        let admission = admission_threshold.clamp(0.0, 1.0);
        let eviction = eviction_threshold.clamp(0.0, 1.0);
        debug_assert!(
            admission < eviction,
            "admission_threshold ({admission}) must be strictly less than eviction_threshold ({eviction})"
        );
        let mut pool = Self::new(
            total_memory_bytes,
            block_size,
            num_layers,
            num_heads,
            head_dim,
            bytes_per_element,
        );
        pool.admission_threshold = admission;
        pool.eviction_threshold = eviction;
        pool.admission_limit = Self::compute_limit(pool.total_blocks, admission);
        pool.eviction_limit = Self::compute_limit(pool.total_blocks, eviction);
        pool
    }

    /// Check if N blocks can be allocated without exceeding the admission threshold
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        if self.total_blocks == 0 {
            return false;
        }
        self.allocated_blocks.saturating_add(num_blocks) <= self.admission_limit
    }

    /// Allocate blocks (update tracking).
    ///
    /// Returns an error if allocation would exceed the admission threshold.
    pub fn allocate(&mut self, num_blocks: usize) -> Result<(), String> {
        if !self.can_allocate(num_blocks) {
            return Err(format!(
                "Cannot allocate {} blocks: {} / {} already allocated (threshold: {})",
                num_blocks, self.allocated_blocks, self.total_blocks, self.admission_threshold
            ));
        }
        self.allocated_blocks = self.allocated_blocks.saturating_add(num_blocks);
        Ok(())
    }

    /// Free blocks (update tracking)
    pub fn free(&mut self, num_blocks: usize) {
        self.allocated_blocks = self.allocated_blocks.saturating_sub(num_blocks);
    }

    /// Check if eviction is needed
    pub fn needs_eviction(&self) -> bool {
        if self.total_blocks == 0 {
            return true;
        }
        self.allocated_blocks > self.eviction_limit
    }

    /// Get current memory usage (0.0–1.0)
    pub fn usage(&self) -> f32 {
        if self.total_blocks == 0 {
            return 1.0;
        }
        self.allocated_blocks as f32 / self.total_blocks as f32
    }

    /// Get number of allocated blocks
    pub fn allocated_blocks(&self) -> usize {
        self.allocated_blocks
    }

    /// Get total blocks
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Get number of free blocks
    pub fn free_blocks(&self) -> usize {
        self.total_blocks.saturating_sub(self.allocated_blocks)
    }

    /// Set admission threshold, updating the precomputed block limit
    pub fn set_admission_threshold(&mut self, threshold: f32) {
        self.admission_threshold = threshold.clamp(0.0, 1.0);
        self.admission_limit = Self::compute_limit(self.total_blocks, self.admission_threshold);
    }

    /// Set eviction threshold, updating the precomputed block limit
    pub fn set_eviction_threshold(&mut self, threshold: f32) {
        self.eviction_threshold = threshold.clamp(0.0, 1.0);
        self.eviction_limit = Self::compute_limit(self.total_blocks, self.eviction_threshold);
    }

    /// Get admission threshold
    pub fn admission_threshold(&self) -> f32 {
        self.admission_threshold
    }

    /// Get eviction threshold
    pub fn eviction_threshold(&self) -> f32 {
        self.eviction_threshold
    }

    /// Compute an integer block limit from a fractional threshold
    fn compute_limit(total_blocks: usize, threshold: f32) -> usize {
        (total_blocks as f32 * threshold) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(total_memory_bytes: usize) -> MemoryPool {
        // 64 tokens/block, 32 layers, 32 heads, 128 head_dim, 2 bytes (BF16)
        MemoryPool::new(total_memory_bytes, 64, 32, 32, 128, 2)
    }

    #[test]
    fn test_memory_pool_basic() {
        let pool = make_pool(1024 * 1024 * 1024);
        assert!(pool.total_blocks() > 0);
        assert_eq!(pool.allocated_blocks(), 0);
        assert_eq!(pool.usage(), 0.0);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = make_pool(1024 * 1024 * 1024);
        let total = pool.total_blocks();

        // Should be able to allocate up to admission threshold
        let allocate_count = (total as f32 * 0.85) as usize;
        assert!(pool.allocate(allocate_count).is_ok());
        assert_eq!(pool.allocated_blocks(), allocate_count);

        // Should not be able to allocate more (would exceed threshold)
        let remaining = total - allocate_count;
        assert!(pool.allocate(remaining).is_err());
    }

    #[test]
    fn test_memory_pool_free() {
        let mut pool = make_pool(1024 * 1024 * 1024);
        let total = pool.total_blocks();
        let allocate_count = (total / 4).min(100);

        pool.allocate(allocate_count).unwrap();
        assert_eq!(pool.allocated_blocks(), allocate_count);

        pool.free(allocate_count / 2);
        assert_eq!(pool.allocated_blocks(), allocate_count / 2);

        pool.free(allocate_count); // Free more than allocated — must not underflow
        assert_eq!(pool.allocated_blocks(), 0);
    }

    #[test]
    fn test_memory_pool_eviction_detection() {
        let mut pool = make_pool(1024 * 1024 * 1024);
        let total = pool.total_blocks();

        // Allocate below eviction threshold (0.95)
        let below_threshold = (total as f32 * 0.85) as usize;
        pool.allocate(below_threshold).unwrap();
        assert!(!pool.needs_eviction());

        pool.free(below_threshold);

        // Manually set above eviction threshold (96%)
        let above_eviction = (total as f32 * 0.96).ceil() as usize;
        pool.allocated_blocks = above_eviction.min(total);

        let usage = pool.usage();
        let eviction_threshold = pool.eviction_threshold();
        assert!(
            usage > eviction_threshold,
            "Usage {} should be > eviction threshold {}. Total blocks: {}, Allocated: {}",
            usage,
            eviction_threshold,
            total,
            pool.allocated_blocks
        );
        assert!(pool.needs_eviction());
    }

    #[test]
    fn test_memory_pool_custom_thresholds() {
        let pool = MemoryPool::with_thresholds(
            1024 * 1024 * 1024,
            64,
            32,
            32,
            128,
            2,
            0.8, // admission
            0.9, // eviction
        );

        assert_eq!(pool.admission_threshold(), 0.8);
        assert_eq!(pool.eviction_threshold(), 0.9);
    }

    #[test]
    fn test_memory_pool_usage() {
        let mut pool = make_pool(1024 * 1024 * 1024);
        let total = pool.total_blocks();

        pool.allocate(total / 2).unwrap();
        assert!((pool.usage() - 0.5).abs() < 0.01);

        pool.allocate(total / 4).unwrap();
        assert!((pool.usage() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_memory_pool_zero_capacity() {
        // bytes_per_block == 0 (zero layers) → total_blocks == 0
        let pool = MemoryPool::new(1024 * 1024, 64, 0, 32, 128, 2);
        assert_eq!(pool.total_blocks(), 0);
        assert!(!pool.can_allocate(0)); // zero-capacity pool rejects everything
        assert!(!pool.can_allocate(1));
        assert!(pool.needs_eviction()); // zero-capacity is always full
        assert_eq!(pool.usage(), 1.0);
    }

    #[test]
    fn test_memory_pool_set_thresholds_updates_limits() {
        let mut pool = make_pool(1024 * 1024 * 1024);
        let total = pool.total_blocks();

        // Narrow the admission window to 50%
        pool.set_admission_threshold(0.5);
        let half = (total as f32 * 0.5) as usize;
        assert!(pool.allocate(half).is_ok());
        assert!(pool.allocate(1).is_err()); // just over the new limit
    }
}
