//! Independent `MoE` conformance and packed execution over real hybrid state.

use crate::model_common::{checkpoint, max_relative};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, invalid},
};
#[cfg(feature = "metal")]
use tur_engine::hybrid::{Boundaries, ModelState};
use tur_models::models::qwen3_5::{Config, Qwen35};
use tur_models::models::{Execution, LoadOptions, WeightFormat};

const CONFIG: &str = include_str!("../fixtures/qwen3_5_moe_config.json");
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen3_5_moe_golden.safetensors"
);

fn load(path: &Path, device: &Device, format: WeightFormat) -> Result<Qwen35> {
    Qwen35::load(
        path,
        &LoadOptions {
            revision: "moe-fixture-v1".into(),
            weights: format.into(),
            device: device.clone(),
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F32),
            recurrent: DType::F32,
            mtp: None,
        },
    )
}

fn golden() -> Result<(Vec<u32>, Vec<Tensor>, Tensor)> {
    let mut data = candle_core::safetensors::load(GOLDEN, &Device::Cpu)?;
    let ids = data
        .remove("ids")
        .ok_or_else(|| invalid("missing ids"))?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| invalid("invalid golden ID")))
        .collect::<Result<_>>()?;
    let layers = data
        .remove("hidden")
        .ok_or_else(|| invalid("missing hidden"))?;
    let layers = (0..layers.dim(0)?)
        .map(|i| Ok(layers.get(i)?))
        .collect::<Result<_>>()?;
    Ok((
        ids,
        layers,
        data.remove("logits")
            .ok_or_else(|| invalid("missing logits"))?,
    ))
}

fn compare(model: &Qwen35) -> Result<()> {
    let (ids, expected_layers, logits) = golden()?;
    let (layers, actual) = model.trace(&ids, &(0..ids.len()).collect::<Vec<_>>())?;
    assert_eq!(layers.len(), expected_layers.len(), "hidden state count");
    for (index, (actual, expected)) in layers.iter().zip(expected_layers).enumerate() {
        let error = max_relative(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
        assert!(error < 1e-4, "layer {index}: {error}");
    }
    let error = max_relative(&actual.to_vec2::<f32>()?, &logits.to_vec2::<f32>()?);
    assert!(error < 1e-4, "logits: {error}");
    Ok(())
}

#[test]
fn test_moe_reference_matches_independent_transformers_layers_and_logits() -> Result<()> {
    let fixture = checkpoint(CONFIG, 0.03, 0.1)?;
    compare(&load(&fixture.0, &Device::Cpu, WeightFormat::F32)?)
}

#[test]
fn test_moe_config_and_weight_mapping_match_the_pinned_checkpoint() -> Result<()> {
    let config = Config::from_json(include_str!("../fixtures/checkpoints/qwen3.6-config.json"))?;
    let expected = config.weight_shapes()?;
    let index: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/checkpoints/qwen3.6-model.safetensors.index.json"
    ))?;
    let names = index["weight_map"]
        .as_object()
        .ok_or_else(|| invalid("missing weight_map"))?;
    for name in expected.keys() {
        assert!(names.contains_key(name), "missing {name}");
    }
    let headers: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/checkpoints/qwen3.6-text-tensors.json"
    ))?;
    for (name, shape) in &expected {
        assert_eq!(
            serde_json::to_value(shape)?,
            headers["tensors"][name]["shape"],
            "{name}"
        );
    }
    assert_eq!(
        expected["model.language_model.layers.0.mlp.experts.gate_up_proj"],
        [256, 1024, 2048]
    );
    assert_eq!(
        expected["model.language_model.layers.0.mlp.experts.down_proj"],
        [256, 2048, 512]
    );
    for (key, value) in [
        ("num_experts", 0),
        ("num_experts_per_tok", 257),
        ("shared_expert_intermediate_size", 0),
    ] {
        let mut raw: serde_json::Value = serde_json::from_str(CONFIG)?;
        raw[key] = serde_json::json!(value);
        assert!(Config::from_json(&raw.to_string()).is_err());
    }
    assert!(
        Config::from_json(
            r#"{"model_type":"gemma4","architectures":["Gemma4ForConditionalGeneration"]}"#
        )
        .is_err()
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_moe_fused_packed_prefill_decode_and_rollback_match_reference() -> Result<()> {
    let fixture = checkpoint(CONFIG, 0.03, 0.1)?;
    let device = Device::new_metal(0)?;
    compare(&load(&fixture.0, &device, WeightFormat::F32)?)?;
    let (ids, _, _) = golden()?;
    for format in [WeightFormat::F32, WeightFormat::Q8_0, WeightFormat::Q4K] {
        let target = load(&fixture.0, &device, format)?;
        let fused = load(&fixture.0, &device, format)?;
        let pool = crate::model_common::pool(&fused, 16, 8)?;
        let (_, state) =
            crate::model_common::prefill(&fused, ModelState::new(&pool), &ids[..13], 7)?;
        // Every token has a recurrent checkpoint. Commit just two, rejecting the rest.
        let (steps, reservations) = crate::model_common::forward(
            &fused,
            &[(&state, &ids[13..16], Boundaries::EveryToken)],
        )?;
        for step in steps {
            step?;
        }
        let accepted = reservations
            .first()
            .ok_or_else(|| invalid("missing reservation"))?
            .commit(2)?;
        drop(reservations);
        let (continued, _) = crate::model_common::prefill(&fused, accepted, &ids[15..], 1)?;
        let control_pool = crate::model_common::pool(&target, 16, 8)?;
        let (_, control) =
            crate::model_common::prefill(&target, ModelState::new(&control_pool), &ids[..13], 7)?;
        let (_, control) = crate::model_common::prefill(&target, control, &ids[13..15], 1)?;
        let (expected, _) = crate::model_common::prefill(&target, control, &ids[15..], 1)?;
        let error = max_relative(&continued, &expected);
        assert!(error < 2e-4, "{format:?}: {error}");
        let (_, planned, _) = Qwen35::memory_requirements(
            fused.config(),
            &LoadOptions {
                revision: "plan".into(),
                weights: format.into(),
                device: Device::Cpu,
                execution: Execution::Reference,
                kv: KvLayout::new(16, KvFormat::F32),
                recurrent: DType::F32,
                mtp: None,
            },
            32,
            &[32],
        )?;
        assert_eq!(planned, fused.weight_bytes());
    }
    Ok(())
}

/// Verification logits of a `window`-token draft against decoding the same tokens one at
/// a time, inside and beyond the Q4K GEMV range.
#[cfg(feature = "metal")]
#[test]
fn test_verification_windows_bound_drift_from_sequential_decoding() -> Result<()> {
    let fixture = checkpoint(CONFIG, 0.03, 0.1)?;
    let device = Device::new_metal(0)?;
    let (ids, _, _) = golden()?;
    for format in [WeightFormat::F32, WeightFormat::Q8_0, WeightFormat::Q4K] {
        let model = load(&fixture.0, &device, format)?;
        let start = ids
            .len()
            .checked_sub(8)
            .filter(|&start| start > 0)
            .ok_or_else(|| invalid("golden too short"))?;
        for window in [3, 8] {
            let draft = &ids[start..start + window];
            let pool = crate::model_common::pool(&model, 16, 2 + window)?;
            let (_, base) =
                crate::model_common::prefill(&model, ModelState::new(&pool), &ids[..start], 7)?;
            let (mut steps, reservations) =
                crate::model_common::forward(&model, &[(&base, draft, Boundaries::EveryToken)])?;
            let verified = crate::model_common::rows(
                &steps.pop().ok_or_else(|| invalid("missing step"))??.logits,
            )?;
            drop(reservations);
            let (sequential, _) = crate::model_common::prefill(&model, base, draft, 1)?;
            let error = max_relative(&verified, &sequential);
            // Every row count up to 16 stays on F32 GEMV or matrix-unit kernels, so a
            // verification window computes what one-at-a-time decoding does. (Before Q4K had
            // a matrix-unit kernel, 5–16 rows used Candle's F16-staged tiles: 1.1e-2 here.)
            let bound = 2e-4;
            assert!(error < bound, "{format:?} window {window}: {error}");
        }
    }
    Ok(())
}
