//! gpt-oss conformance against independent Transformers F32 layers and logits.

use crate::model_common::max_relative;
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, invalid},
};
use tur_models::models::gpt_oss::{Config, GptOss};
use tur_models::models::{Execution, LoadOptions, WeightFormat};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gpt_oss")
}

fn load(device: &Device, format: WeightFormat, execution: Execution) -> Result<GptOss> {
    GptOss::load(
        &fixture(),
        &LoadOptions {
            revision: "gpt-oss-fixture-v1".into(),
            weights: format.into(),
            device: device.clone(),
            execution,
            kv: KvLayout::new(16, KvFormat::F32),
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            expert_cache: None,
        },
    )
}

struct Golden {
    ids: Vec<u32>,
    layers: Vec<Tensor>,
    logits: Tensor,
}

fn golden() -> Result<Golden> {
    let mut data =
        candle_core::safetensors::load(fixture().join("golden.safetensors"), &Device::Cpu)?;
    let mut take = |name: &str| {
        data.remove(name)
            .ok_or_else(|| invalid(format!("missing {name}")))
    };
    let ids = take("tokens")?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| invalid("invalid golden ID")))
        .collect::<Result<_>>()?;
    let layers = (0..4)
        .map(|i| take(&format!("layer.{i}")))
        .collect::<Result<_>>()?;
    Ok(Golden {
        ids,
        layers,
        logits: take("logits")?,
    })
}

fn compare(model: &GptOss, bound: f32) -> Result<()> {
    let golden = golden()?;
    let rows: Vec<usize> = (0..golden.ids.len()).collect();
    let (layers, logits) = model.trace(&golden.ids, &rows)?;
    assert_eq!(layers.len(), golden.layers.len());
    for (index, (actual, expected)) in layers.iter().zip(&golden.layers).enumerate() {
        let error = max_relative(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
        assert!(error < bound, "layer {index}: {error}");
    }
    let error = max_relative(&logits.to_vec2::<f32>()?, &golden.logits.to_vec2::<f32>()?);
    assert!(error < bound, "logits: {error}");
    Ok(())
}

#[test]
fn test_gpt_oss_reference_matches_independent_transformers_layers_and_logits() -> Result<()> {
    compare(
        &load(&Device::Cpu, WeightFormat::F32, Execution::Reference)?,
        1e-4,
    )
}

#[cfg(feature = "metal")]
#[test]
fn test_gpt_oss_fused_f32_experts_match_transformers() -> Result<()> {
    let device = Device::new_metal(0)?;
    compare(&load(&device, WeightFormat::F32, Execution::Fused)?, 1e-4)
}

#[cfg(feature = "metal")]
#[test]
fn test_gpt_oss_fused_mxfp4_experts_match_reference_and_plan() -> Result<()> {
    let device = Device::new_metal(0)?;
    let golden = golden()?;
    let rows: Vec<usize> = (0..golden.ids.len()).collect();
    let packed = load(&device, WeightFormat::Q8_0, Execution::Fused)?;
    let reference = load(&device, WeightFormat::Q8_0, Execution::Reference)?;
    let (_, actual) = packed.trace(&golden.ids, &rows)?;
    let (_, expected) = reference.trace(&golden.ids, &rows)?;
    let error = max_relative(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
    assert!(error < 2e-4, "{error}");
    let (_, planned, _) = GptOss::memory_requirements(
        packed.config(),
        &LoadOptions {
            revision: "plan".into(),
            weights: WeightFormat::Q8_0.into(),
            device,
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F32),
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            expert_cache: None,
        },
        32,
        &[32],
    )?;
    assert_eq!(planned, packed.weight_bytes());
    Ok(())
}

/// A load whose experts stay on disk behind a cache of `slots` experts.
#[cfg(feature = "metal")]
fn cached(dir: &Path, device: &Device, slots: usize) -> Result<(GptOss, LoadOptions)> {
    // gate|up: 128 rows of 2 blocks; down: 64 rows of 2 blocks; 17 bytes per block.
    let expert = (128 + 64) * 2 * 17;
    let options = LoadOptions {
        revision: "gpt-oss-fixture-v1".into(),
        weights: WeightFormat::Q8_0.into(),
        device: device.clone(),
        execution: Execution::Fused,
        kv: KvLayout::new(16, KvFormat::F32),
        recurrent: DType::F32,
        mtp: None,
        audio: false,
        expert_cache: Some(slots * expert),
    };
    Ok((GptOss::load(dir, &options)?, options))
}

#[cfg(feature = "metal")]
fn logits(model: &GptOss, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    let rows: Vec<usize> = (0..ids.len()).collect();
    Ok(model.trace(ids, &rows)?.1.to_vec2::<f32>()?)
}

/// The cache changes where weights live, never what the kernels compute: evicting between
/// and within passes leaves every logit bit-identical to keeping all experts resident.
#[cfg(feature = "metal")]
#[test]
fn test_expert_cache_logits_are_bit_identical_to_resident_under_eviction() -> Result<()> {
    let device = Device::new_metal(0)?;
    let golden = golden()?;
    let resident = logits(
        &load(&device, WeightFormat::Q8_0, Execution::Fused)?,
        &golden.ids,
    )?;
    for slots in [4, 7, 16] {
        let (model, options) = cached(&fixture(), &device, slots)?;
        for pass in 0..3 {
            let actual = logits(&model, &golden.ids)?;
            for (a, r) in actual.iter().flatten().zip(resident.iter().flatten()) {
                assert_eq!(a.to_bits(), r.to_bits(), "{slots} slots, pass {pass}");
            }
        }
        let stats = model
            .expert_stats()
            .ok_or_else(|| invalid("missing expert stats"))?;
        assert!(stats.misses > 0, "{slots} slots read nothing");
        // A full cache keeps everything after the first pass; a small one keeps reading.
        if slots == 16 {
            assert_eq!(stats.misses, 16);
            assert_eq!(stats.evictions, 0);
            assert!(stats.hits > 0);
        } else {
            assert!(stats.evictions > 0, "{slots} slots never evicted");
        }
        let (_, planned, _) = GptOss::memory_requirements(model.config(), &options, 32, &[32])?;
        assert_eq!(planned, model.weight_bytes());
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_expert_cache_rejects_a_budget_below_one_layer_of_experts() -> Result<()> {
    let device = Device::new_metal(0)?;
    assert!(cached(&fixture(), &device, 3).is_err());
    let reference = LoadOptions {
        revision: "gpt-oss-fixture-v1".into(),
        weights: WeightFormat::Q8_0.into(),
        device,
        execution: Execution::Reference,
        kv: KvLayout::new(16, KvFormat::F32),
        recurrent: DType::F32,
        mtp: None,
        audio: false,
        expert_cache: Some(1 << 20),
    };
    assert!(GptOss::load(&fixture(), &reference).is_err());
    Ok(())
}

/// A read failure fails that pass, leaves no slot claiming data it does not hold, and the
/// file is only ever read: its bytes and modification time do not change.
#[cfg(feature = "metal")]
#[test]
fn test_expert_cache_recovers_from_read_failures_and_never_writes() -> Result<()> {
    let device = Device::new_metal(0)?;
    let golden = golden()?;
    let dir = std::env::temp_dir().join(format!("tur-gpt-oss-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir)?;
    let result = (|| -> Result<()> {
        let weights = dir.join("model.safetensors");
        std::fs::copy(fixture().join("config.json"), dir.join("config.json"))?;
        let original = std::fs::read(fixture().join("model.safetensors"))?;
        std::fs::write(&weights, &original)?;
        let (model, _) = cached(&dir, &device, 4)?;
        let expected = logits(
            &load(&device, WeightFormat::Q8_0, Execution::Fused)?,
            &golden.ids,
        )?;
        let modified = std::fs::metadata(&weights)?.modified()?;
        assert_eq!(logits(&model, &golden.ids)?, expected);
        assert_eq!(std::fs::metadata(&weights)?.modified()?, modified);
        assert_eq!(std::fs::read(&weights)?, original);
        // Cutting the file makes the next miss fail; restoring it must fully recover.
        std::fs::write(&weights, &original[..original.len() / 4])?;
        assert!(logits(&model, &golden.ids).is_err());
        std::fs::write(&weights, &original)?;
        assert_eq!(logits(&model, &golden.ids)?, expected);
        Ok(())
    })();
    std::fs::remove_dir_all(&dir)?;
    result
}

/// Hiding experts from the routers routes exactly as removing them does: a checkpoint pruned
/// to a searched plan, copied byte for byte, computes bit-identical logits.
#[cfg(feature = "metal")]
#[test]
fn test_pruned_checkpoint_equals_hiding_its_removed_experts() -> Result<()> {
    use tur_models::models::gpt_oss::{Pruning, plan_pruning, prune};
    let device = Device::new_metal(0)?;
    let golden = golden()?;
    let (model, _) = cached(&fixture(), &device, 4)?;
    let probe = model.expert_probe(std::slice::from_ref(&golden.ids))?;
    assert_eq!(probe.layers.len(), 4);
    let kept: Vec<Vec<usize>> = plan_pruning(&probe, 3)?
        .into_iter()
        .map(|plan| {
            let mut kept = plan.kept;
            kept.sort_unstable();
            kept
        })
        .collect();
    assert!(kept.iter().all(|layer| layer.len() == 3));
    let dir = std::env::temp_dir().join(format!("tur-gpt-oss-{}", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let pruning = |kept: Vec<Vec<usize>>| Pruning {
            source: "gpt-oss-fixture-v1".into(),
            calibration: "golden tokens".into(),
            kept,
        };
        prune(&fixture(), &dir, &pruning(kept.clone()))?;
        let pruned = GptOss::load(
            &dir,
            &LoadOptions {
                revision: "pruned".into(),
                weights: WeightFormat::Q8_0.into(),
                device: device.clone(),
                execution: Execution::Fused,
                kv: KvLayout::new(16, KvFormat::F32),
                recurrent: DType::F32,
                mtp: None,
                audio: false,
                expert_cache: None,
            },
        )?;
        model.hide_experts(Some(&kept))?;
        assert_eq!(logits(&pruned, &golden.ids)?, logits(&model, &golden.ids)?);
        model.hide_experts(None)?;
        assert!(prune(&fixture(), &dir, &pruning(kept.clone())).is_err());
        let mut uneven = kept.clone();
        uneven[0].pop();
        assert!(prune(&fixture(), &dir.join("uneven"), &pruning(uneven)).is_err());
        Ok(())
    })();
    drop(std::fs::remove_dir_all(&dir));
    result
}

#[test]
fn test_gpt_oss_config_and_tensors_match_the_pinned_checkpoint() -> Result<()> {
    let config = Config::from_json(include_str!("../fixtures/gpt_oss/20b-config.json"))?;
    let index: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/gpt_oss/20b-model.safetensors.index.json"
    ))?;
    let names = index["weight_map"]
        .as_object()
        .ok_or_else(|| invalid("missing weight_map"))?;
    let expected = config.weight_shapes();
    assert_eq!(expected.len(), names.len());
    for name in expected.keys() {
        assert!(names.contains_key(name), "missing {name}");
    }
    assert_eq!(
        expected["model.layers.0.mlp.experts.gate_up_proj_blocks"],
        [32, 5760, 90, 16]
    );
    assert_eq!(
        expected["model.layers.23.mlp.experts.down_proj_scales"],
        [32, 2880, 90]
    );
    assert_eq!(config.max_position(), 131_072);
    Ok(())
}

/// Real weights against Transformers' own layers, with F32 projections so that only
/// execution can differ. Needs `TUR_GPT_OSS_20B` and `TUR_GPT_OSS_GOLDEN`.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the 13 GB gpt-oss-20b checkpoint and its golden"]
fn test_gpt_oss_20b_matches_transformers_through_the_expert_cache() -> Result<()> {
    let read = |name: &str| std::env::var(name).map_err(|_| invalid(format!("{name} is not set")));
    let dir = PathBuf::from(read("TUR_GPT_OSS_20B")?);
    let mut data = candle_core::safetensors::load(read("TUR_GPT_OSS_GOLDEN")?, &Device::Cpu)?;
    let mut take = |name: &str| {
        data.remove(name)
            .ok_or_else(|| invalid(format!("missing {name}")))
    };
    let ids: Vec<u32> = take("tokens")?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| invalid("invalid golden ID")))
        .collect::<Result<_>>()?;
    let device = Device::new_metal(0)?;
    let model = GptOss::load(
        &dir,
        &LoadOptions {
            revision: "gpt-oss-20b-conformance".into(),
            weights: WeightFormat::F32.into(),
            device,
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F32),
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            expert_cache: Some(1 << 30),
        },
    )?;
    let rows: Vec<usize> = (0..ids.len()).collect();
    let (layers, logits) = model.trace(&ids, &rows)?;
    for (index, actual) in layers.iter().enumerate() {
        let expected = take(&format!("layer.{index}"))?;
        let error = max_relative(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
        assert!(error < 1e-4, "layer {index}: {error}");
    }
    let expected = take("logits")?;
    let error = max_relative(&logits.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
    assert!(error < 1e-4, "logits: {error}");
    Ok(())
}

#[test]
fn test_gpt_oss_config_rejects_settings_tur_does_not_execute() -> Result<()> {
    let source = std::fs::read_to_string(fixture().join("config.json"))?;
    assert!(Config::from_json(&source).is_ok());
    for (key, value) in [
        ("attention_bias", serde_json::json!(false)),
        ("hidden_act", serde_json::json!("gelu")),
        ("hidden_size", serde_json::json!(48)),
        ("head_dim", serde_json::json!(96)),
        ("num_experts_per_tok", serde_json::json!(5)),
        ("num_local_experts", serde_json::json!(512)),
        ("tie_word_embeddings", serde_json::json!(true)),
    ] {
        let mut raw: serde_json::Value = serde_json::from_str(&source)?;
        raw[key] = value;
        assert!(Config::from_json(&raw.to_string()).is_err(), "{key}");
    }
    let mut raw: serde_json::Value = serde_json::from_str(&source)?;
    raw["quantization_config"]["quant_method"] = serde_json::json!("gptq");
    assert!(Config::from_json(&raw.to_string()).is_err());
    Ok(())
}
