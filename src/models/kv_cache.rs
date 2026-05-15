//! Wrapper around candle's ConcatKvCache to enable state extraction and restoration
//! for prefix caching support.

use candle_core::{Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;

/// Wrapper around ConcatKvCache that tracks accumulated KV state
/// This enables extracting and restoring cache state for prefix caching
#[derive(Debug, Clone)]
pub struct KvCache {
    /// The underlying candle KV cache
    inner: ConcatKvCache,
    /// Cached K tensor (accumulated across all appends)
    cached_k: Option<Tensor>,
    /// Cached V tensor (accumulated across all appends)
    cached_v: Option<Tensor>,
}

impl KvCache {
    /// Create a new KV cache wrapper
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

    /// Append new K and V tensors to the cache
    ///
    /// This delegates to the underlying ConcatKvCache and also tracks the accumulated state
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Append to inner cache
        let (k_out, v_out) = self.inner.append(k, v)?;

        // Update our tracked state - Tensor::clone is cheap (Arc-based)
        self.cached_k = Some(k_out.clone());
        self.cached_v = Some(v_out.clone());

        Ok((k_out, v_out))
    }

    /// Get the current accumulated KV cache state
    ///
    /// Returns None if no state has been accumulated yet
    ///
    /// Note: Tensor clones are cheap (Arc-based), no deep copy occurs
    pub fn get_state(&self) -> Option<(Tensor, Tensor)> {
        match (&self.cached_k, &self.cached_v) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }

    /// Restore KV cache state from previously saved tensors
    ///
    /// This sets the internal state and resets the underlying cache to use these tensors
    pub fn set_state(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        // Reset the inner cache
        self.inner.reset();

        // Append the restored state to the inner cache
        // This ensures the cache behaves as if it had accumulated this state naturally
        self.inner.append(&k, &v)?;

        // Store the tensors
        self.cached_k = Some(k);
        self.cached_v = Some(v);

        Ok(())
    }

    /// Reset the cache, clearing all accumulated state
    pub fn reset(&mut self) {
        self.inner.reset();
        self.cached_k = None;
        self.cached_v = None;
    }

    /// Check if the cache has any accumulated state
    pub fn is_empty(&self) -> bool {
        self.cached_k.is_none() && self.cached_v.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_kv_cache_wrapper_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(2);

        assert!(cache.is_empty());
        assert!(cache.get_state().is_none());

        // Create some test tensors (B=1, H=2, S=3, D=4)
        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;

        // Append to cache
        let (k_out, v_out) = cache.append(&k1, &v1)?;
        assert_eq!(k_out.dims(), &[1, 2, 3, 4]);
        assert_eq!(v_out.dims(), &[1, 2, 3, 4]);

        assert!(!cache.is_empty());
        assert!(cache.get_state().is_some());

        Ok(())
    }

    #[test]
    fn test_kv_cache_state_restoration() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(2);

        // Create and append initial state
        let k1 = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let v1 = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        cache.append(&k1, &v1)?;

        // Get the state
        let (saved_k, saved_v) = cache.get_state().unwrap();

        // Reset cache
        cache.reset();
        assert!(cache.is_empty());

        // Restore state
        cache.set_state(saved_k.clone(), saved_v.clone())?;
        assert!(!cache.is_empty());

        // Verify restored state matches
        let (restored_k, restored_v) = cache.get_state().unwrap();
        assert_eq!(restored_k.dims(), saved_k.dims());
        assert_eq!(restored_v.dims(), saved_v.dims());

        Ok(())
    }

    #[test]
    fn test_kv_cache_accumulation() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(2);

        // Append first batch (seq_len=3)
        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;
        let (k_out1, _) = cache.append(&k1, &v1)?;
        assert_eq!(k_out1.dims()[2], 3); // seq_len = 3

        // Append second batch (seq_len=2)
        let k2 = Tensor::zeros((1, 2, 2, 4), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 2, 2, 4), DType::F32, &device)?;
        let (k_out2, _) = cache.append(&k2, &v2)?;
        assert_eq!(k_out2.dims()[2], 5); // seq_len = 3 + 2 = 5

        Ok(())
    }
}

// Made with Bob
