#![forbid(unsafe_code)]
//! Peak scratch planning for one forward pass.

use candle_core::DType;
use tur_core::{Result, execution::invalid};

/// A live allocation in a [`ScratchPlan`].
#[derive(Debug, PartialEq, Eq)]
#[must_use = "release a region when its buffer dies, or the plan overstates later peaks"]
pub struct Region(usize);

/// Tracks bytes live at each point of a pass and their peak. Regions are rounded to the
/// next power of two because Candle's pooled allocator reserves buffers that way, so the
/// peak bounds what the pool actually holds rather than the nominal tensor sizes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScratchPlan {
    live: usize,
    peak: usize,
}

impl ScratchPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve `elements` of `dtype`; zero-sized regions reserve nothing.
    pub fn reserve(&mut self, elements: usize, dtype: DType) -> Result<Region> {
        let bytes = elements
            .checked_mul(dtype.size_in_bytes())
            .and_then(|b| {
                if b == 0 {
                    Some(0)
                } else {
                    b.checked_next_power_of_two()
                }
            })
            .ok_or_else(|| invalid("scratch region overflows"))?;
        self.live = self
            .live
            .checked_add(bytes)
            .ok_or_else(|| invalid("scratch plan overflows"))?;
        self.peak = self.peak.max(self.live);
        Ok(Region(bytes))
    }

    /// Reserve a region and release it immediately: a transient that only raises the peak.
    pub fn transient(&mut self, elements: usize, dtype: DType) -> Result<()> {
        let region = self.reserve(elements, dtype)?;
        self.release(region);
        Ok(())
    }

    pub fn release(&mut self, region: Region) {
        self.live = self.live.saturating_sub(region.0);
    }

    /// Bytes live now.
    pub fn live_bytes(&self) -> usize {
        self.live
    }

    /// Largest number of bytes live at once.
    pub fn peak_bytes(&self) -> usize {
        self.peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_tracks_overlapping_regions_with_pool_rounding() -> Result<()> {
        let mut plan = ScratchPlan::new();
        let a = plan.reserve(100, DType::F32)?;
        assert_eq!(plan.live_bytes(), 512);
        let b = plan.reserve(10, DType::U32)?;
        assert_eq!(plan.peak_bytes(), 512 + 64);
        plan.release(a);
        plan.transient(64, DType::F32)?;
        assert_eq!(plan.peak_bytes(), 576);
        plan.release(b);
        assert_eq!(plan.live_bytes(), 0);
        Ok(())
    }

    #[test]
    fn test_empty_regions_reserve_nothing_and_overflow_is_rejected() -> Result<()> {
        let mut plan = ScratchPlan::new();
        plan.transient(0, DType::F32)?;
        assert_eq!(plan.peak_bytes(), 0);
        assert!(plan.reserve(usize::MAX, DType::F32).is_err());
        let _big = plan.reserve(usize::MAX / 2, DType::U8)?;
        assert!(plan.reserve(usize::MAX / 2, DType::U8).is_err());
        Ok(())
    }
}
