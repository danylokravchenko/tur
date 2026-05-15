//! Memory pool for managing GPU/CPU memory allocation
//!
//! This module provides memory management for paged attention, including:
//! - Block-based memory tracking
//! - Admission control based on memory availability
//! - Memory pressure detection

use candle_core::Device;

/// Memory pool for managing block allocation
pub struct MemoryPool {
    /// Total memory capacity (in blocks)
    total_blocks: usize,

    /// Currently allocated blocks
    allocated_blocks: usize,

    /// Block size (tokens per block)
    pub block_size: usize,

    /// Device
    device: Device,

    /// Memory usage threshold for admission control (0.0-1.0)
    /// New requests are only admitted if usage < admission_threshold
    admission_threshold: f32,

    /// Memory usage threshold for eviction (0.0-1.0)
    /// Eviction is triggered when usage > eviction_threshold
    eviction_threshold: f32,
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
    /// * `device` - Device for memory allocation
    pub fn new(
        total_memory_bytes: usize,
        block_size: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        bytes_per_element: usize,
        device: Device,
    ) -> Self {
        // Calculate memory per block:
        // Each block stores K and V tensors for all layers
        // K/V shape per layer: [num_heads, block_size, head_dim]
        // Total per block: 2 (K+V) × num_layers × num_heads × block_size × head_dim × bytes_per_element
        let bytes_per_block =
            2 * num_layers * num_heads * block_size * head_dim * bytes_per_element;

        let total_blocks = if bytes_per_block > 0 {
            total_memory_bytes.checked_div(bytes_per_block).unwrap_or(0)
        } else {
            0
        };

        Self {
            total_blocks,
            allocated_blocks: 0,
            block_size,
            device,
            admission_threshold: 0.9,
            eviction_threshold: 0.95,
        }
    }

    /// Create a memory pool with custom thresholds
    #[allow(clippy::too_many_arguments)]
    pub fn with_thresholds(
        total_memory_bytes: usize,
        block_size: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        bytes_per_element: usize,
        device: Device,
        admission_threshold: f32,
        eviction_threshold: f32,
    ) -> Self {
        let mut pool = Self::new(
            total_memory_bytes,
            block_size,
            num_layers,
            num_heads,
            head_dim,
            bytes_per_element,
            device,
        );
        pool.admission_threshold = admission_threshold.clamp(0.0, 1.0);
        pool.eviction_threshold = eviction_threshold.clamp(0.0, 1.0);
        pool
    }

    /// Check if N blocks can be allocated
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        let new_allocated = self.allocated_blocks + num_blocks;
        let usage = new_allocated as f32 / self.total_blocks as f32;
        usage < self.admission_threshold
    }

    /// Allocate blocks (update tracking)
    ///
    /// Returns error if allocation would exceed capacity
    pub fn allocate(&mut self, num_blocks: usize) -> Result<(), String> {
        if !self.can_allocate(num_blocks) {
            return Err(format!(
                "Cannot allocate {} blocks: {} / {} already allocated (threshold: {})",
                num_blocks, self.allocated_blocks, self.total_blocks, self.admission_threshold
            ));
        }

        self.allocated_blocks += num_blocks;
        Ok(())
    }

    /// Free blocks (update tracking)
    pub fn free(&mut self, num_blocks: usize) {
        self.allocated_blocks = self.allocated_blocks.saturating_sub(num_blocks);
    }

    /// Check if eviction is needed
    pub fn needs_eviction(&self) -> bool {
        let usage = self.allocated_blocks as f32 / self.total_blocks as f32;
        usage > self.eviction_threshold
    }

    /// Get current memory usage (0.0-1.0)
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

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Set admission threshold
    pub fn set_admission_threshold(&mut self, threshold: f32) {
        self.admission_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set eviction threshold
    pub fn set_eviction_threshold(&mut self, threshold: f32) {
        self.eviction_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get admission threshold
    pub fn admission_threshold(&self) -> f32 {
        self.admission_threshold
    }

    /// Get eviction threshold
    pub fn eviction_threshold(&self) -> f32 {
        self.eviction_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let device = Device::Cpu;
        // 1 GB memory, 64 tokens/block, 32 layers, 32 heads, 128 head_dim, 2 bytes (BF16)
        let pool = MemoryPool::new(
            1024 * 1024 * 1024, // 1 GB
            64,
            32,
            32,
            128,
            2,
            device,
        );

        assert!(pool.total_blocks() > 0);
        assert_eq!(pool.allocated_blocks(), 0);
        assert_eq!(pool.usage(), 0.0);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let device = Device::Cpu;
        let mut pool = MemoryPool::new(1024 * 1024 * 1024, 64, 32, 32, 128, 2, device);

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
        let device = Device::Cpu;
        let mut pool = MemoryPool::new(1024 * 1024 * 1024, 64, 32, 32, 128, 2, device);

        let total = pool.total_blocks();
        let allocate_count = (total / 4).min(100);

        pool.allocate(allocate_count).unwrap();
        assert_eq!(pool.allocated_blocks(), allocate_count);

        pool.free(allocate_count / 2);
        assert_eq!(pool.allocated_blocks(), allocate_count / 2);

        pool.free(allocate_count); // Free more than allocated
        assert_eq!(pool.allocated_blocks(), 0);
    }

    #[test]
    fn test_memory_pool_eviction_detection() {
        let device = Device::Cpu;
        let mut pool = MemoryPool::new(1024 * 1024 * 1024, 64, 32, 32, 128, 2, device);

        let total = pool.total_blocks();

        // Allocate below eviction threshold (0.95)
        let below_threshold = (total as f32 * 0.85) as usize;
        pool.allocate(below_threshold).unwrap();
        assert!(!pool.needs_eviction());

        // Free and manually set allocated blocks above eviction threshold
        pool.free(below_threshold);

        // Set allocated blocks to exactly 96% of total (above 0.95 eviction threshold)
        let above_eviction = (total as f32 * 0.96).ceil() as usize;
        pool.allocated_blocks = above_eviction.min(total);

        // Verify the usage is above eviction threshold
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
        let device = Device::Cpu;
        let pool = MemoryPool::with_thresholds(
            1024 * 1024 * 1024,
            64,
            32,
            32,
            128,
            2,
            device,
            0.8, // admission
            0.9, // eviction
        );

        assert_eq!(pool.admission_threshold(), 0.8);
        assert_eq!(pool.eviction_threshold(), 0.9);
    }

    #[test]
    fn test_memory_pool_usage() {
        let device = Device::Cpu;
        let mut pool = MemoryPool::new(1024 * 1024 * 1024, 64, 32, 32, 128, 2, device);

        let total = pool.total_blocks();

        pool.allocate(total / 2).unwrap();
        assert!((pool.usage() - 0.5).abs() < 0.01);

        pool.allocate(total / 4).unwrap();
        assert!((pool.usage() - 0.75).abs() < 0.01);
    }
}
