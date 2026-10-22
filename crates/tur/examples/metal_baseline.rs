#![allow(
    clippy::print_stdout,
    reason = "this recorder emits its JSON artifact on stdout"
)]
#![forbid(unsafe_code)]

use candle_core::{DType, Device, Tensor};
use serde_json::{Value, json};
use std::{error::Error, io, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const WARMUP: usize = 3;
const SAMPLES: usize = 20;

fn validate(actual: &Tensor, reference: &Tensor) -> Result<f32> {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let reference = reference.flatten_all()?.to_vec1::<f32>()?;
    if actual.len() != reference.len() {
        return Err(io::Error::other("readback length differs from CPU reference").into());
    }
    let mut max_error = 0.0_f32;
    for (actual, reference) in actual.iter().zip(reference.iter()) {
        let error = (actual - reference).abs();
        if !actual.is_finite()
            || !reference.is_finite()
            || error > 1e-4f32.mul_add(reference.abs(), 1e-4)
        {
            return Err(io::Error::other(format!(
                "Metal readback {actual} differs from CPU reference {reference}"
            ))
            .into());
        }
        max_error = max_error.max(error);
    }
    Ok(max_error)
}

fn measure(
    device: &Device,
    reference: &Tensor,
    mut operation: impl FnMut() -> candle_core::Result<Tensor>,
) -> Result<Value> {
    for _ in 0..WARMUP {
        let output = operation()?;
        device.synchronize()?;
        validate(&output, reference)?;
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut max_error = 0.0_f32;
    for _ in 0..SAMPLES {
        device.synchronize()?;
        let start = Instant::now();
        let output = operation()?;
        // Retain the output until the GPU finishes; enqueue time alone is not a measurement.
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64());
        max_error = max_error.max(validate(&output, reference)?);
    }
    Ok(json!({"samples_seconds": samples, "max_abs_error": max_error}))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn custom_dispatch_probe(device: &Device) -> Result<Value> {
    use candle_core::{Storage, metal_backend::MetalStorage};
    use candle_metal_kernels::metal::{ComputePipeline, Library};
    use objc2_foundation::NSString;
    use objc2_metal::{MTLDevice as _, MTLGPUFamily, MTLSize};
    use std::sync::Arc;

    let Device::Metal(native) = device else {
        return Err(io::Error::other("custom MSL probe requires Metal").into());
    };
    let raw = native.metal_device().as_ref();
    // Call the fallible bindings directly: Candle's compilation convenience methods unwrap errors.
    let invalid_shader_rejected = raw
        .newLibraryWithSource_options_error(&NSString::from_str("kernel void invalid("), None)
        .is_err();
    if !invalid_shader_rejected {
        return Err(io::Error::other("invalid MSL unexpectedly compiled").into());
    }
    let start = Instant::now();
    let library = raw
        .newLibraryWithSource_options_error(
            &NSString::from_str(include_str!("metal_probe.metal")),
            None,
        )
        .map(Library::new)
        .map_err(|error| io::Error::other(error.to_string()))?;
    let missing_function_rejected = library.get_function("missing_probe", None).is_err();
    if !missing_function_rejected {
        return Err(io::Error::other("missing MSL function unexpectedly resolved").into());
    }
    let function = library.get_function("write_indices", None)?;
    let pipeline = raw
        .newComputePipelineStateWithFunction_error(function.as_ref())
        .map(ComputePipeline::new)
        .map_err(|error| io::Error::other(error.to_string()))?;
    let compile_seconds = start.elapsed().as_secs_f64();
    let width = pipeline.max_total_threads_per_threadgroup().min(32);
    if width == 0 {
        return Err(io::Error::other("zero maximum Metal threadgroup size").into());
    }
    let mut pending = Vec::new();
    for count in [1_u32, 31, 257, 1025] {
        let elements = usize::try_from(count)?
            .checked_add(2)
            .ok_or_else(|| io::Error::other("probe buffer size overflow"))?;
        let buffer = native.new_buffer_with_data(&vec![u32::MAX; elements])?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| io::Error::other("probe byte size overflow"))?;
        if buffer.length() < bytes {
            return Err(io::Error::other("probe allocation is too small").into());
        }
        {
            let guard = native.command_encoder()?;
            guard.set_label("tur_p0_write_indices");
            guard.set_compute_pipeline_state(&pipeline);
            // The guard keeps the encoder locked while binding; outputs are tracked for barriers.
            let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
            encoder.set_output_buffer(0, Some(&buffer), std::mem::size_of::<u32>());
            encoder.set_bytes(1, &count);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: usize::try_from(count)?.div_ceil(width),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        }
        pending.push((count, elements, buffer));
    }
    // Hold every buffer and the compiled pipeline across completion, then use Candle's safe readback.
    native.wait_until_completed()?;
    for (count, elements, buffer) in pending {
        let storage = MetalStorage::new(Arc::clone(&buffer), native.clone(), elements, DType::U32);
        let output = Tensor::from((Storage::Metal(storage), elements)).to_vec1::<u32>()?;
        if output.first() != Some(&u32::MAX) || output.last() != Some(&u32::MAX) {
            return Err(io::Error::other("custom MSL overwrote a guard element").into());
        }
        let values = output
            .get(1..elements - 1)
            .ok_or_else(|| io::Error::other("missing custom MSL output range"))?;
        if !values.iter().copied().eq(0..count) {
            return Err(
                io::Error::other("custom MSL output differs from reference indices").into(),
            );
        }
    }
    Ok(json!({
        "backend": "Candle-native buffers and command ownership; fallible objc2 compilation",
        "compile_seconds": compile_seconds,
        "dispatch_completed": true, "readback_validated": true,
        "invalid_shader_rejected": invalid_shader_rejected,
        "missing_function_rejected": missing_function_rejected,
        "tested_counts": [1, 31, 257, 1025], "guard_elements": 2, "buffer_offset_bytes": 4,
        "supports_apple7": raw.supportsFamily(MTLGPUFamily::Apple7),
        "supports_metal3": raw.supportsFamily(MTLGPUFamily::Metal3),
        "supports_metal4": raw.supportsFamily(MTLGPUFamily::Metal4),
        "recommended_working_set_bytes": native.recommended_max_working_set_size(),
    }))
}

fn main() -> Result<()> {
    #[cfg(feature = "metal")]
    if metal::Device::all().is_empty() {
        return Err(io::Error::other("no accessible Metal devices; GPU access is required").into());
    }
    let device = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let input: Vec<f32> = (0..257).map(|i| (i % 17) as f32 / 16.0).collect();
    let input_cpu = Tensor::from_vec(input, 257, &cpu)?;
    let input_gpu = input_cpu.to_device(&device)?;
    let reference = input_cpu.affine(0.5, 0.25)?;
    let mut cases = vec![json!({
        "operation": "affine", "shape": [257],
        "measurement": measure(&device, &reference, || input_gpu.affine(0.5, 0.25))?,
    })];
    for rows in [1, 128, 2048] {
        let lhs: Vec<f32> = (0..rows * 256).map(|i| (i % 17) as f32 / 16.0).collect();
        let rhs: Vec<f32> = (0..256 * 256).map(|i| (i % 13) as f32 / 16.0).collect();
        let lhs = Tensor::from_vec(lhs, (rows, 256), &cpu)?;
        let rhs = Tensor::from_vec(rhs, (256, 256), &cpu)?;
        let reference = lhs.matmul(&rhs)?;
        let lhs = lhs.to_device(&device)?;
        let rhs = rhs.to_device(&device)?;
        cases.push(json!({
            "operation": "matmul", "m": rows, "k": 256, "n": 256,
            "measurement": measure(&device, &reference, || lhs.matmul(&rhs))?,
        }));
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let probe = custom_dispatch_probe(&device)?;
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let probe = json!({"error": "Metal feature disabled"});
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "workload": "synthetic Candle primitives; not model or serving throughput",
            "dtype": DType::F32.as_str(),
            "fixture": "modular nonnegative integers divided by 16; no RNG",
            "checkpoint": null, "tokenizer": null, "template": null, "quantization": null,
            "prompt_tokens": 0, "output_tokens": 0, "seed": null,
            "cold_model_load_seconds": null,
            "warmup_iterations": WARMUP, "measured_iterations": SAMPLES,
            "timing": "host wall time through Device::synchronize; excludes readback and CPU validation",
            "tolerance": {"absolute": 1e-4, "relative": 1e-4},
            "custom_msl": probe, "cases": cases,
        }))?
    );
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_rejects_nonfinite_and_incorrect_values() -> Result<()> {
        let reference = Tensor::new(&[1.0_f32, 2.0], &Device::Cpu)?;
        for values in [[f32::NAN, 2.0], [f32::INFINITY, 2.0], [1.0, 2.01]] {
            let actual = Tensor::new(&values, &Device::Cpu)?;
            assert!(validate(&actual, &reference).is_err());
        }
        Ok(())
    }

    #[test]
    fn test_validation_checks_length_and_accepts_reference() -> Result<()> {
        let reference = Tensor::new(&[1.0_f32, 2.0], &Device::Cpu)?;
        assert_eq!(validate(&reference, &reference)?, 0.0);
        let shorter = Tensor::new(&[1.0_f32], &Device::Cpu)?;
        assert!(validate(&shorter, &reference).is_err());
        Ok(())
    }
}
