pub mod cpu_flash;
pub use cpu_flash::flash_attn;

use candle_core::Tensor;

#[derive(Debug, Clone, Default)]
pub enum AttnMask {
    #[default]
    None,
    Causal {
        kv_offset: usize,
    },
    Mask(Tensor),
}

impl AttnMask {
    #[inline]
    pub fn causal() -> Self {
        AttnMask::Causal { kv_offset: 0 }
    }

    #[inline]
    pub fn causal_with_offset(kv_offset: usize) -> Self {
        AttnMask::Causal { kv_offset }
    }

    #[inline]
    pub fn is_causal(&self) -> bool {
        matches!(self, AttnMask::Causal { .. })
    }

    #[inline]
    pub fn kv_offset(&self) -> usize {
        match self {
            AttnMask::Causal { kv_offset } => *kv_offset,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_attn_mask_default() {
        let mask = AttnMask::default();
        assert!(matches!(mask, AttnMask::None));
    }

    #[test]
    fn test_attn_mask_none_is_not_causal() {
        let mask = AttnMask::None;
        assert!(!mask.is_causal());
    }

    #[test]
    fn test_attn_mask_none_kv_offset_is_zero() {
        let mask = AttnMask::None;
        assert_eq!(mask.kv_offset(), 0);
    }

    #[test]
    fn test_causal_constructor() {
        let mask = AttnMask::causal();
        assert!(mask.is_causal());
        assert_eq!(mask.kv_offset(), 0);
    }

    #[test]
    fn test_causal_with_offset_zero() {
        let mask = AttnMask::causal_with_offset(0);
        assert!(mask.is_causal());
        assert_eq!(mask.kv_offset(), 0);
    }

    #[test]
    fn test_causal_with_offset_nonzero() {
        let mask = AttnMask::causal_with_offset(42);
        assert!(mask.is_causal());
        assert_eq!(mask.kv_offset(), 42);
    }

    #[test]
    fn test_causal_with_large_offset() {
        let mask = AttnMask::causal_with_offset(1000);
        assert!(mask.is_causal());
        assert_eq!(mask.kv_offset(), 1000);
    }

    #[test]
    fn test_mask_variant_is_not_causal() {
        let tensor = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let mask = AttnMask::Mask(tensor);
        assert!(!mask.is_causal());
    }

    #[test]
    fn test_mask_variant_kv_offset_is_zero() {
        let tensor = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let mask = AttnMask::Mask(tensor);
        assert_eq!(mask.kv_offset(), 0);
    }

    #[test]
    fn test_attn_mask_clone() {
        let mask1 = AttnMask::causal_with_offset(10);
        let mask2 = mask1.clone();
        assert!(mask2.is_causal());
        assert_eq!(mask2.kv_offset(), 10);
    }

    #[test]
    fn test_multiple_causal_masks_independent() {
        let mask1 = AttnMask::causal_with_offset(10);
        let mask2 = AttnMask::causal_with_offset(20);
        assert_eq!(mask1.kv_offset(), 10);
        assert_eq!(mask2.kv_offset(), 20);
    }
}
