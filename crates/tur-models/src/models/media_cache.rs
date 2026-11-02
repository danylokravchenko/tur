//! Encoded rows of recent clips, images and videos, by content.
use crate::media::Digest;
use candle_core::Tensor;
use parking_lot::Mutex;

/// Device bytes a model with encoders sets aside for encoded rows, reserved in its memory plan
/// whether or not it fills them. A 1024-token image at 0.8B holds 4 MiB, and at Qwen3.8's width
/// 21 MiB, so this keeps several of the largest images.
pub(crate) const RESERVED_BYTES: usize = 64 << 20;

/// A least-recently-used store bounded by [`RESERVED_BYTES`]. Weights never change under a
/// loaded model and the key is a hash of the content, so nothing invalidates an entry; only
/// capacity evicts one. Rows are immutable, and a request that holds them keeps them alive
/// past eviction, so the commands using them are unaffected.
#[derive(Default)]
pub(crate) struct MediaCache {
    /// Least recently used first; a handful of entries, so a scan is cheap.
    entries: Mutex<Vec<(Digest, Tensor)>>,
}

fn bytes(rows: &Tensor) -> usize {
    rows.elem_count() * rows.dtype().size_in_bytes()
}

impl MediaCache {
    pub(crate) fn get(&self, key: &Digest) -> Option<Tensor> {
        let mut entries = self.entries.lock();
        let at = entries.iter().position(|(digest, _)| digest == key)?;
        let entry = entries.remove(at);
        let rows = entry.1.clone();
        entries.push(entry);
        Some(rows)
    }

    /// Keep `rows`, evicting the oldest as needed. Rows beyond the whole budget aren't kept.
    pub(crate) fn insert(&self, key: Digest, rows: &Tensor) {
        if bytes(rows) > RESERVED_BYTES {
            return;
        }
        let mut entries = self.entries.lock();
        entries.retain(|(digest, _)| *digest != key);
        entries.push((key, rows.clone()));
        let mut held: usize = entries.iter().map(|(_, rows)| bytes(rows)).sum();
        while held > RESERVED_BYTES {
            let (_, oldest) = entries.remove(0);
            held -= bytes(&oldest);
        }
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
    use tur_core::Result;

    fn rows(megabytes: usize) -> Result<Tensor> {
        Ok(Tensor::zeros(
            (megabytes << 18, 1),
            DType::F32,
            &Device::Cpu,
        )?)
    }

    #[test]
    fn test_capacity_evicts_the_least_recently_used_and_skips_oversized_rows() -> Result<()> {
        let cache = MediaCache::default();
        let key = |byte| [byte; 32];
        cache.insert(key(1), &rows(30)?);
        cache.insert(key(2), &rows(30)?);
        // Touching the first makes the second the oldest.
        assert!(cache.get(&key(1)).is_some());
        cache.insert(key(3), &rows(30)?);
        assert!(cache.get(&key(2)).is_none());
        assert!(cache.get(&key(1)).is_some() && cache.get(&key(3)).is_some());
        cache.insert(key(4), &rows(65)?);
        assert!(cache.get(&key(4)).is_none());
        assert!(cache.get(&key(1)).is_some());
        Ok(())
    }
}
