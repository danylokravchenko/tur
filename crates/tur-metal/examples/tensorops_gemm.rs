#![allow(
    clippy::print_stdout,
    reason = "this probe emits its JSON lines on stdout"
)]
#![forbid(unsafe_code)]
//! Metal 4 `TensorOps` capability and F32 GEMM throughput against Candle's F32 GEMM on the
//! same device (P10). Kernels build `tensor_inline` views over existing buffers in the
//! shader, so the host needs no `MTLTensor` objects, whose buffer-backed constructor is
//! unsafe in objc2-metal. Prints one JSON line per measurement.
//!
//! `cargo run --release -p tur-metal --example tensorops_gemm`

use candle_core::{DType, Device, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLSize};
use serde_json::json;
use std::time::Instant;
use tur_core::{Result, TurError};

const REPS: usize = 10;

/// `C[m×n] = A[m×k] · B[k×n]`, row-major, one `tm × tn` output tile per threadgroup.
fn source(m: usize, n: usize, k: usize, tm: usize, tn: usize, simdgroups: usize) -> String {
    format!(
        r"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemm(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                 device float *c [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]]) {{
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>({k}, {m}));
    auto B = tensor<device float, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>({n}, {k}));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>({n}, {m}));
    constexpr auto desc = matmul2d_descriptor({tm}, {tn}, dynamic_length_v<int>, false, false, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<{simdgroups}>> op;
    auto a_tile = A.slice(0, tg.y * {tm});
    auto b_tile = B.slice(tg.x * {tn}, 0);
    auto c_tile = C.slice(tg.x * {tn}, tg.y * {tm});
    op.run(a_tile, b_tile, c_tile);
}}
"
    )
}

fn error(e: impl std::fmt::Display) -> TurError {
    TurError::Other(format!("TensorOps: {e}"))
}

fn seconds(device: &Device, mut op: impl FnMut() -> Result<()>) -> Result<f64> {
    op()?;
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..REPS {
        op()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() / REPS as f64)
}

fn dispatch(
    device: &Device,
    pipeline: &ComputePipeline,
    tensors: [&Tensor; 3],
    groups: (usize, usize),
    threads: usize,
) -> Result<()> {
    let Device::Metal(metal) = device else {
        return Err(error("needs a Metal device"));
    };
    let [a, b, c] = tensors.map(Tensor::storage_and_layout);
    let (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c)) = (&*a.0, &*b.0, &*c.0) else {
        return Err(error("inputs are not Metal storage"));
    };
    let guard = metal.command_encoder()?;
    guard.set_compute_pipeline_state(pipeline);
    let encoder: &ComputeCommandEncoder = guard.as_ref();
    encoder.set_input_buffer(0, Some(a.buffer()), 0);
    encoder.set_input_buffer(1, Some(b.buffer()), 0);
    encoder.set_output_buffer(2, Some(c.buffer()), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups.0,
            height: groups.1,
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

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let Device::Metal(metal) = &device else {
        return Err(error("needs a Metal device"));
    };
    let raw = metal.metal_device();
    // Prefill projections of the 0.8B checkpoint: in_proj-like, a short chunk, down.
    for (m, n, k) in [
        (2048usize, 8192usize, 1024usize),
        (128, 8192, 1024),
        (2048, 1024, 3584),
    ] {
        let a = Tensor::randn(0f32, 1.0, (m, k), &device)?;
        let b = Tensor::randn(0f32, 1.0, (k, n), &device)?;
        let reference = a.matmul(&b)?;
        let flops = 2.0 * (m * n * k) as f64;
        let candle = seconds(&device, || {
            a.matmul(&b)?;
            Ok(())
        })?;
        println!(
            "{}",
            json!({"m": m, "n": n, "k": k, "kernel": "candle_f32", "seconds": candle,
                   "tflops": flops / candle / 1e12})
        );
        for (tm, tn, simdgroups) in [
            (32usize, 32usize, 4usize),
            (64, 32, 4),
            (64, 64, 4),
            (128, 64, 8),
        ] {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let library = raw
                .new_library_with_source(&source(m, n, k, tm, tn, simdgroups), Some(&options))
                .map_err(error)?;
            let function = library.get_function("gemm", None).map_err(error)?;
            let pipeline = raw
                .new_compute_pipeline_state_with_function(&function)
                .map_err(error)?;
            let c = Tensor::zeros((m, n), DType::F32, &device)?;
            let run = || {
                dispatch(
                    &device,
                    &pipeline,
                    [&a, &b, &c],
                    (n / tn, m / tm),
                    32 * simdgroups,
                )
            };
            let tensorops = seconds(&device, run)?;
            let max_error = (&c - &reference)?.abs()?.max_all()?.to_scalar::<f32>()?;
            println!(
                "{}",
                json!({"m": m, "n": n, "k": k, "kernel": "tensorops_matmul2d_f32",
                       "tile": [tm, tn], "simdgroups": simdgroups, "seconds": tensorops,
                       "tflops": flops / tensorops / 1e12, "speedup_vs_candle": candle / tensorops,
                       "max_abs_error_vs_candle": max_error})
            );
        }
    }
    Ok(())
}
