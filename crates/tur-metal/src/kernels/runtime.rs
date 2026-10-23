//! Device capabilities, compiled pipelines, checked buffer bindings and dispatch.
//!
//! Every dispatch goes through Candle's command encoder for the same device, so custom
//! kernels share Candle's command ordering, hazard barriers, pooled-buffer reuse rules and
//! the [`crate::completion::Completion`] fence. No separate queue or synchronization exists.

use candle_core::{
    DType, Device, MetalDevice, Shape, Storage, Tensor, metal_backend::MetalStorage,
};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState as _, MTLDevice as _, MTLGPUFamily, MTLMathMode,
    MTLSize,
};
use std::{collections::HashMap, sync::Arc, sync::RwLockReadGuard};
use tur_core::{Result, TurError, execution::invalid};

/// Version of the buffer layouts the kernels assume. Bump when any binding layout changes.
/// 2: paged K/V pools in F32/F16/Q8 and slot-addressed recurrent state (P5).
/// 3: tiled prefill attention over query tiles.
pub const LAYOUT_VERSION: u32 = 3;

// Runtime Metal compilation has no local include search path, so embed the families as
// one source string with shared helpers first.
const SOURCE: &str = concat!(
    include_str!("common.metal"),
    "\n",
    include_str!("normalization.metal"),
    "\n",
    include_str!("delta.metal"),
    "\n",
    include_str!("attention.metal"),
    "\n",
    include_str!("quantized.metal"),
    "\n",
    include_str!("sampling.metal"),
);
const SIMD_WIDTH: usize = 32;

/// Every function in [`SOURCE`]. Template instantiations encode their shape family
/// (`r4` = 4 registers per lane, `b3` = 3 batch rows) and dtype or K/V format in the name.
const FUNCTIONS: &[&str] = &[
    "rms_norm_f32",
    "add_rms_norm_f32",
    "qk_norm_rope_f32",
    "swiglu_f32",
    "gated_rms_f32",
    "delta_conv_f32",
    "kv_write_r2_f32",
    "kv_write_r2_f16",
    "kv_write_r2_q8",
    "kv_write_r4_f32",
    "kv_write_r4_f16",
    "kv_write_r4_q8",
    "kv_write_r8_f32",
    "kv_write_r8_f16",
    "kv_write_r8_q8",
    "attention_r2_f32",
    "attention_r2_f16",
    "attention_r2_q8",
    "attention_r4_f32",
    "attention_r4_f16",
    "attention_r4_q8",
    "attention_r8_f32",
    "attention_r8_f16",
    "attention_r8_q8",
    "attention_tiled_r2_f32",
    "attention_tiled_r2_f16",
    "attention_tiled_r2_q8",
    "attention_tiled_r4_f32",
    "attention_tiled_r4_f16",
    "attention_tiled_r4_q8",
    "attention_tiled_r8_f32",
    "attention_tiled_r8_f16",
    "attention_tiled_r8_q8",
    "attention_reduce_r2_f32",
    "attention_reduce_r4_f32",
    "attention_reduce_r8_f32",
    "qmv_q8_0_b1_f32",
    "qmv_q8_0_b2_f32",
    "qmv_q8_0_b3_f32",
    "qmv_q8_0_b4_f32",
    "qmv_q8_0_b5_f32",
    "qmv_q8_0_b6_f32",
    "qmv_q8_0_b7_f32",
    "qmv_q8_0_b8_f32",
    "qmm_q8_0_t1_f32",
    "qmm_q8_0_t2_f32",
    "delta_scan_r2_f32",
    "delta_scan_r2_f16",
    "delta_scan_r2_bf16",
    "delta_scan_r4_f32",
    "delta_scan_r4_f16",
    "delta_scan_r4_bf16",
    "delta_scan_r8_f32",
    "delta_scan_r8_f16",
    "delta_scan_r8_bf16",
    "delta_step_d64_f32",
    "delta_step_d64_f16",
    "delta_step_d64_bf16",
    "delta_step_d128_f32",
    "delta_step_d128_f16",
    "delta_step_d128_bf16",
    "delta_step_d256_f32",
    "delta_step_d256_f16",
    "delta_step_d256_bf16",
    "argmax_f32",
    "penalize_f32",
    "mask_f32",
    "sample_f32",
    "qmv_q4k_b1_f32",
    "qmv_q4k_b2_f32",
    "qmv_q4k_b3_f32",
    "qmv_q4k_b4_f32",
    "qmv_q4k_b5_f32",
    "qmv_q4k_b6_f32",
    "qmv_q4k_b7_f32",
    "qmv_q4k_b8_f32",
];

pub(super) fn device_error(error: impl std::fmt::Display) -> TurError {
    TurError::Other(format!("Metal runtime: {error}"))
}

pub(super) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{what} {value} exceeds u32")))
}

/// Measured properties of one Metal device. Family support does not prove identical
/// numerics or throughput across chips; it gates features, and tuning starts from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub name: String,
    pub architecture: String,
    /// Highest supported `MTLGPUFamilyAppleN`, or 0.
    pub apple_family: u8,
    pub metal3: bool,
    pub metal4: bool,
    pub unified_memory: bool,
    pub max_threads_per_threadgroup: usize,
    pub threadgroup_memory_bytes: usize,
    pub max_buffer_bytes: usize,
    pub recommended_working_set_bytes: usize,
}

impl Capabilities {
    fn query(device: &MetalDevice) -> Self {
        let raw = device.metal_device().as_ref();
        let families = [
            (MTLGPUFamily::Apple10, 10),
            (MTLGPUFamily::Apple9, 9),
            (MTLGPUFamily::Apple8, 8),
            (MTLGPUFamily::Apple7, 7),
            (MTLGPUFamily::Apple6, 6),
        ];
        let apple_family = families
            .iter()
            .find(|(family, _)| raw.supportsFamily(*family))
            .map_or(0, |(_, n)| *n);
        let threads = raw.maxThreadsPerThreadgroup();
        Self {
            name: raw.name().to_string(),
            architecture: device.metal_device().architecture_name(),
            apple_family,
            metal3: raw.supportsFamily(MTLGPUFamily::Metal3),
            metal4: raw.supportsFamily(MTLGPUFamily::Metal4),
            unified_memory: raw.hasUnifiedMemory(),
            max_threads_per_threadgroup: threads.width,
            threadgroup_memory_bytes: raw.maxThreadgroupMemoryLength(),
            max_buffer_bytes: raw.maxBufferLength(),
            recommended_working_set_bytes: device.recommended_max_working_set_size(),
        }
    }
}

/// Launch shapes derived from capabilities. Each value is clamped again per pipeline, so
/// no single threadgroup configuration is assumed for every device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// Threads per threadgroup for one-row-per-threadgroup reductions.
    pub row_threads: usize,
    /// Simdgroups per threadgroup for one-item-per-simdgroup kernels.
    pub simdgroups: usize,
    /// Simdgroups per delta-scan threadgroup. Reduced automatically when the staged state
    /// tile exceeds threadgroup memory.
    pub scan_simdgroups: usize,
    /// Threads per threadgroup for elementwise kernels.
    pub elementwise_threads: usize,
    /// Attention simdgroups below which keys are split across threadgroups.
    pub attention_parallelism: usize,
    /// Smallest key range of one attention split.
    pub min_split_keys: usize,
    /// Threadgroups (query tiles × heads) from which the tiled matrix-unit kernel replaces
    /// the per-query kernel when tiles average at least 16 query rows; below it, splitting
    /// keys fills the GPU better. 0 always tiles eligible passes; `usize::MAX` never tiles.
    pub attention_tiled_groups: usize,
    /// Inclusive row counts served by the packed `Q8_0` and Q4K GEMVs; other counts use
    /// Candle's one-row GEMV or tiled GEMM.
    pub gemv_q8_0_rows: (usize, usize),
    pub gemv_q4k_rows: (usize, usize),
    /// Inclusive row counts served by the small-batch `Q8_0` matrix-unit kernel, which
    /// takes precedence over the GEMV range; at most 16.
    pub qmm_q8_0_rows: (usize, usize),
    /// Single-token recurrent sequences use the coalesced step kernel instead of the scan.
    pub delta_step: bool,
}

impl Tuning {
    /// Starting point from the device; benchmarks may justify per-device overrides.
    pub fn for_device(capabilities: &Capabilities) -> Self {
        let limit = capabilities.max_threads_per_threadgroup.max(SIMD_WIDTH);
        Self {
            row_threads: limit.min(256),
            simdgroups: (limit / SIMD_WIDTH).clamp(1, 4),
            // The M1 sweep (P4 kernel artifact) is flat within ~5%; 2 was best most often.
            scan_simdgroups: (limit / SIMD_WIDTH).clamp(1, 2),
            elementwise_threads: limit.min(256),
            attention_parallelism: 1024,
            min_split_keys: 64,
            // The M1 attention sweep crosses over at 64 queries × 8 heads (16 tiles).
            attention_tiled_groups: 16,
            // Measured crossovers on M1 (P4 and P6 kernel artifacts): Candle's one-row GEMV
            // wins at one row. For Q8_0 the custom GEMV wins at 2–3 rows and the
            // matrix-unit kernel from 4 (1.3–2.6× at 5–8 rows, F32 parity with Candle's
            // F16-tile GEMM at 16). Q4K uses Candle's tiled GEMM from 5 rows.
            gemv_q8_0_rows: (2, 3),
            gemv_q4k_rows: (2, 4),
            qmm_q8_0_rows: (4, 16),
            delta_step: true,
        }
    }
}

/// A compiled pipeline with its device limits.
pub(super) struct Pipeline {
    pipeline: ComputePipeline,
    width: usize,
    max_threads: usize,
}

impl Pipeline {
    /// Largest multiple of the execution width not above `desired` or the pipeline limit.
    pub(super) fn threads(&self, desired: usize) -> usize {
        let width = self.width.max(1);
        let capped = desired.min(self.max_threads).max(width);
        (capped / width) * width
    }

    /// Simdgroups per threadgroup for one-item-per-simdgroup kernels.
    pub(super) fn simdgroups(&self, desired: usize) -> usize {
        self.threads(desired.saturating_mul(SIMD_WIDTH)) / SIMD_WIDTH
    }
}

/// One device's compiled library and pipelines. Created once per model load.
pub(super) struct Runtime {
    pub(super) device: Device,
    pub(super) metal: MetalDevice,
    pub(super) capabilities: Capabilities,
    pub(super) tuning: Tuning,
    pipelines: HashMap<&'static str, Pipeline>,
}

impl Runtime {
    /// Compile every kernel up front so no request pays pipeline creation latency.
    pub(super) fn new(device: &Device) -> Result<Self> {
        let Device::Metal(metal) = device else {
            return Err(invalid("Metal kernels require a Metal device"));
        };
        let capabilities = Capabilities::query(metal);
        if SIMD_WIDTH > capabilities.max_threads_per_threadgroup {
            return Err(device_error("threadgroups smaller than one simdgroup"));
        }
        let options = MTLCompileOptions::new();
        // Safe math keeps IEEE semantics (infinities, exact exp/log edge cases) that the
        // reference comparison and online softmax rely on.
        options.setMathMode(MTLMathMode::Safe);
        let library = metal
            .metal_device()
            .new_library_with_source(SOURCE, Some(&options))
            .map_err(device_error)?;
        let mut pipelines = HashMap::with_capacity(FUNCTIONS.len());
        for &name in FUNCTIONS {
            let function = library.get_function(name, None).map_err(device_error)?;
            let pipeline = metal
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(device_error)?;
            let raw = pipeline.as_ref();
            let width = raw.threadExecutionWidth();
            if width != SIMD_WIDTH {
                return Err(device_error(format!(
                    "{name} executes {width} threads per SIMD group; kernels assume {SIMD_WIDTH}"
                )));
            }
            let max_threads = raw.maxTotalThreadsPerThreadgroup();
            pipelines.insert(
                name,
                Pipeline {
                    pipeline,
                    width,
                    max_threads,
                },
            );
        }
        Ok(Self {
            tuning: Tuning::for_device(&capabilities),
            device: device.clone(),
            metal: metal.clone(),
            capabilities,
            pipelines,
        })
    }

    pub(super) fn pipeline(&self, name: &str) -> Result<&Pipeline> {
        self.pipelines
            .get(name)
            .ok_or_else(|| device_error(format!("kernel {name} is not compiled")))
    }

    /// Encode one dispatch of `groups` threadgroups of `threads` threads.
    pub(super) fn dispatch(
        &self,
        pipeline: &Pipeline,
        label: &str,
        groups: usize,
        threads: usize,
        bind: impl FnOnce(&ComputeCommandEncoder) -> Result<()>,
    ) -> Result<()> {
        if groups == 0 || threads == 0 || threads > pipeline.max_threads {
            return Err(invalid(format!("invalid {label} launch shape")));
        }
        let guard = self.metal.command_encoder()?;
        guard.set_label(label);
        guard.set_compute_pipeline_state(&pipeline.pipeline);
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        bind(encoder)?;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Validate `tensor` as F32/U32 rows on this device. Rank 1 is one row; rank 2 may
    /// have any row stride with unit column stride; higher ranks must be contiguous and
    /// flatten to `[dims[0], rest]`.
    pub(super) fn view<'a>(&self, tensor: &'a Tensor, dtype: DType) -> Result<View<'a>> {
        if tensor.dtype() != dtype {
            return Err(invalid(format!(
                "kernel input is {:?}, expected {dtype:?}",
                tensor.dtype()
            )));
        }
        if !tensor.device().same_device(&self.device) {
            return Err(invalid("kernel input is on a different device"));
        }
        let (guard, layout) = tensor.storage_and_layout();
        let dims = layout.dims();
        let stride = layout.stride();
        let (rows, cols, row_stride) = match (dims, stride) {
            ([cols], [1]) => (1, *cols, *cols),
            ([rows, cols], [row_stride, 1]) => (*rows, *cols, *row_stride),
            ([rows, ..], _) if layout.is_contiguous() => {
                let cols = dims[1..].iter().product::<usize>();
                (*rows, cols, cols)
            }
            _ => return Err(invalid("kernel input needs unit column stride")),
        };
        let Storage::Metal(storage) = &*guard else {
            return Err(invalid("kernel input is not Metal storage"));
        };
        let size = dtype.size_in_bytes();
        let elements = storage.buffer().length() / size;
        let end = rows
            .checked_sub(1)
            .map_or(Some(0), |last| {
                last.checked_mul(row_stride)
                    .and_then(|r| r.checked_add(cols))
                    .and_then(|r| r.checked_add(layout.start_offset()))
            })
            .ok_or_else(|| invalid("kernel input extent overflows"))?;
        if end > elements || (rows > 1 && row_stride < cols) {
            return Err(invalid("kernel input exceeds its buffer or overlaps rows"));
        }
        Ok(View {
            offset: layout.start_offset() * size,
            guard,
            rows,
            cols,
            row_stride,
        })
    }

    /// Fresh pooled output of `rows × cols` F32 elements.
    pub(super) fn output(&self, rows: usize, cols: usize) -> Result<Output> {
        let count = rows
            .checked_mul(cols)
            .filter(|&c| c > 0)
            .ok_or_else(|| invalid("empty or overflowing kernel output"))?;
        let buffer = self.metal.new_buffer(count, DType::F32, "tur_kernel")?;
        Ok(Output { buffer, rows, cols })
    }

    /// Upload host indices; validated by the caller before reaching the GPU.
    pub(super) fn indices(&self, values: &[u32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(values, values.len(), &self.device)?)
    }
}

/// A validated, read-locked kernel input.
pub(super) struct View<'a> {
    guard: RwLockReadGuard<'a, Storage>,
    offset: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_stride: usize,
}

impl View<'_> {
    pub(super) fn buffer(&self) -> Result<&Buffer> {
        match &*self.guard {
            Storage::Metal(storage) => Ok(storage.buffer()),
            _ => Err(invalid("kernel input is not Metal storage")),
        }
    }

    /// Byte offset of the first element in its buffer.
    pub(super) fn offset(&self) -> usize {
        self.offset
    }

    pub(super) fn bind(&self, encoder: &ComputeCommandEncoder, index: usize) -> Result<()> {
        encoder.set_input_buffer(index, Some(self.buffer()?), self.offset);
        Ok(())
    }

    /// Bind as a kernel output, so later readers of this buffer wait for the writes.
    /// Used for persistent pools that kernels update in place.
    pub(super) fn bind_mut(&self, encoder: &ComputeCommandEncoder, index: usize) -> Result<()> {
        encoder.set_output_buffer(index, Some(self.buffer()?), self.offset);
        Ok(())
    }

    pub(super) fn expect(&self, rows: usize, cols: usize, what: &str) -> Result<()> {
        if self.rows != rows || self.cols != cols {
            return Err(invalid(format!(
                "{what} is {}×{}, expected {rows}×{cols}",
                self.rows, self.cols
            )));
        }
        Ok(())
    }

    pub(super) fn contiguous(&self, what: &str) -> Result<()> {
        if self.rows > 1 && self.row_stride != self.cols {
            return Err(invalid(format!("{what} must be contiguous")));
        }
        Ok(())
    }
}

/// A kernel output that one or more dispatches fill before it becomes a tensor.
pub struct Output {
    buffer: Arc<Buffer>,
    rows: usize,
    cols: usize,
}

impl Output {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub(super) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub(super) fn bind(&self, encoder: &ComputeCommandEncoder, index: usize) {
        encoder.set_output_buffer(index, Some(&self.buffer), 0);
    }

    /// Wrap as an F32 tensor of `shape`, which must have exactly `rows × cols` elements.
    pub(super) fn into_tensor(
        self,
        metal: &MetalDevice,
        shape: impl Into<Shape>,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let count = self.rows * self.cols;
        if shape.elem_count() != count {
            return Err(invalid("kernel output shape does not match its size"));
        }
        let storage = MetalStorage::new(self.buffer, metal.clone(), count, DType::F32);
        Ok(Tensor::from((Storage::Metal(storage), shape)))
    }
}
