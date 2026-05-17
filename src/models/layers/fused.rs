/// Fused RMSNorm + Linear projection, Metal-only.
///
/// On non-Metal devices or quantized-weight paths the caller must fall back to
/// the separate `NormX::forward` → `LinearX::forward` sequence.
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};

// ---------------------------------------------------------------------------
// Pipeline cache (Metal only)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
mod metal_impl {
    use super::*;
    use candle_core::{MetalDevice, MetalStorage, backend::BackendStorage};
    use candle_metal_kernels::metal::ComputePipeline;
    use objc2_metal::MTLSize;
    use parking_lot::Mutex;
    use std::sync::OnceLock;

    /// Metal source embedded at compile time. Compiled by Metal at first use
    /// and cached from that point on (Metal also persists its binary archive to
    /// ~/Library/Caches/com.apple.metal across process restarts).
    const METAL_SOURCE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/kernels/rms_norm_linear.metal"
    ));

    pub(super) const BLOCK_SIZE: usize = 256;
    pub(super) const MAX_SMEM_BYTES: usize = 32 * 1024; // 32 KB (Apple Silicon M1+)

    struct PipelinePair {
        f32: ComputePipeline,
        bf16: ComputePipeline,
    }
    // ComputePipeline already declares Send + Sync in candle-metal-kernels.
    unsafe impl Send for PipelinePair {}
    unsafe impl Sync for PipelinePair {}

    static PIPELINES: OnceLock<Mutex<Option<PipelinePair>>> = OnceLock::new();

    fn get_pipeline(device: &MetalDevice, dtype: DType) -> Result<ComputePipeline> {
        let mutex = PIPELINES.get_or_init(|| Mutex::new(None));
        let mut guard = mutex.lock();

        if guard.is_none() {
            let lib = device
                .new_library_with_source(METAL_SOURCE, None)
                .map_err(|e| candle_core::Error::Msg(format!("Metal compile: {e}")))?;

            let f32_func = lib
                .get_function("rms_norm_linear_f32", None)
                .map_err(|e| candle_core::Error::Msg(format!("Metal fn f32: {e}")))?;
            let bf16_func = lib
                .get_function("rms_norm_linear_bf16", None)
                .map_err(|e| candle_core::Error::Msg(format!("Metal fn bf16: {e}")))?;

            let f32_pl = device
                .new_compute_pipeline_state_with_function(&f32_func)
                .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline f32: {e}")))?;
            let bf16_pl = device
                .new_compute_pipeline_state_with_function(&bf16_func)
                .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline bf16: {e}")))?;

            *guard = Some(PipelinePair {
                f32: f32_pl,
                bf16: bf16_pl,
            });
        }

        let pair = guard.as_ref().expect("pipelines just initialised");
        Ok(match dtype {
            DType::F32 => pair.f32.clone(),
            DType::BF16 => pair.bf16.clone(),
            dt => candle_core::bail!("fused_rms_norm_linear: unsupported dtype {dt:?}"),
        })
    }

    pub(super) fn dispatch(
        x_s: &MetalStorage,
        x_l: &Layout,
        nw_s: &MetalStorage,
        nw_l: &Layout,
        pw_s: &MetalStorage,
        pw_l: &Layout,
        eps: f32,
    ) -> Result<(MetalStorage, Shape)> {
        let dtype = x_s.dtype();
        if nw_s.dtype() != dtype || pw_s.dtype() != dtype {
            candle_core::bail!(
                "fused_rms_norm_linear: dtype mismatch x={dtype:?} norm_w={:?} proj_w={:?}",
                nw_s.dtype(),
                pw_s.dtype(),
            );
        }

        // x:[N,H]  norm_weight:[H]  proj_weight:[K,H]
        let (n, h) = x_l.shape().dims2()?;
        let (k, _) = pw_l.shape().dims2()?;

        let smem_bytes = (h + BLOCK_SIZE) * std::mem::size_of::<f32>();
        if smem_bytes > MAX_SMEM_BYTES {
            candle_core::bail!(
                "fused_rms_norm_linear: H={h} needs {smem_bytes}B threadgroup memory \
                 (limit {MAX_SMEM_BYTES}B); use the unfused path"
            );
        }

        let device = x_s.device();
        let pipeline = get_pipeline(device, dtype)?;

        let elem_count = n * k;
        let out_buf = device.new_buffer(elem_count, dtype, "fused_rms_norm_linear")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("fused_rms_norm_linear");
        encoder.set_compute_pipeline_state(&pipeline);

        let elem_bytes = dtype.size_in_bytes();
        encoder.set_buffer(0, Some(x_s.buffer()), x_l.start_offset() * elem_bytes);
        encoder.set_buffer(1, Some(nw_s.buffer()), nw_l.start_offset() * elem_bytes);
        encoder.set_buffer(2, Some(pw_s.buffer()), pw_l.start_offset() * elem_bytes);
        encoder.set_buffer(3, Some(&out_buf), 0);

        let h_u32 = h as u32;
        let k_u32 = k as u32;
        encoder.set_bytes(4, &h_u32);
        encoder.set_bytes(5, &k_u32);
        encoder.set_bytes(6, &eps);

        encoder.set_threadgroup_memory_length(0, smem_bytes);

        encoder.dispatch_thread_groups(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: BLOCK_SIZE,
                height: 1,
                depth: 1,
            },
        );

        let out_shape = Shape::from_dims(&[n, k]);
        let out_storage = MetalStorage::new(out_buf, device.clone(), elem_count, dtype);
        Ok((out_storage, out_shape))
    }
}

// ---------------------------------------------------------------------------
// CustomOp3 implementation
// ---------------------------------------------------------------------------

pub struct FusedRmsNormLinear {
    pub eps: f32,
}

impl candle_core::CustomOp3 for FusedRmsNormLinear {
    fn name(&self) -> &'static str {
        "fused_rms_norm_linear"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused_rms_norm_linear is Metal-only; use the unfused path on CPU")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal_impl::dispatch(s1, l1, s2, l2, s3, l3, self.eps)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fused RMSNorm + linear projection (Metal only).
///
/// Computes `output = linear(rms_norm(x, norm_weight, eps), proj_weight)` in a
/// single GPU pass, eliminating the intermediate normalised-activation buffer.
///
/// # Shape contract
/// - `x`:           `[N, H]`          — token activations (contiguous)
/// - `norm_weight`: `[H]`             — RMSNorm γ / scale (same dtype as x)
/// - `proj_weight`: `[K, H]`          — linear weight, candle row-major
/// - returns:       `[N, K]`
///
/// Falls back gracefully: call sites should check `device.is_metal()` and that
/// the linear weight is non-quantized before invoking this function.
pub fn rms_norm_linear(
    x: &Tensor,
    norm_weight: &Tensor,
    proj_weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    x.apply_op3_no_bwd(norm_weight, proj_weight, &FusedRmsNormLinear { eps })
}
