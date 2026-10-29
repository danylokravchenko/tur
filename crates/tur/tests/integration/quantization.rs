//! Prepacked weight artifacts, calibration statistics, GPTQ and byte-budget allocation on
//! small random checkpoints.
use crate::model_common as common;
use tur_models::models::{
    Embedding, Execution, LoadOptions, Method, Packer, Precision, WeightFormat, WeightPlan, Weights,
};

use crate::model_common::{Checkpoint, checkpoint};
use candle_core::{DType, Device};
use std::{path::Path, sync::Arc};
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout},
};
use tur_engine::hybrid::HybridModel;
use tur_models::models::qwen3_5::{
    Config, Qwen35, Reference, allocate, sensitivities, write_artifact, write_calibrated_artifact,
};
use tur_models::models::{Conversion, Manifest};

/// Two recurrent and two attention layers, every width a multiple of 256.
const SMALL: &str = r#"{
  "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
  "hidden_size": 256, "intermediate_size": 512, "vocab_size": 160, "num_hidden_layers": 4,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
  "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 64,
  "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attn_output_gate": true,
  "tie_word_embeddings": false, "eos_token_id": 1,
  "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
}"#;
const MOE: &str = include_str!("../fixtures/qwen3_5_moe_config.json");

static GPU: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn small(tied: bool) -> Result<Checkpoint> {
    checkpoint(
        &SMALL.replace(
            "\"tie_word_embeddings\": false",
            &format!("\"tie_word_embeddings\": {tied}"),
        ),
        0.08,
        0.1,
    )
}

fn options(weights: Weights, device: &Device) -> LoadOptions {
    LoadOptions {
        revision: "prepacked-fixture".into(),
        weights,
        device: device.clone(),
        execution: Execution::Fused,
        kv: KvLayout::new(16, KvFormat::F32),
        recurrent: DType::F32,
        mtp: None,
    }
}

fn load(dir: &Path, weights: Weights, device: &Device) -> Result<Qwen35> {
    Qwen35::load(dir, &options(weights, device))
}

fn samples(count: usize, len: usize, vocab: u32) -> Vec<Vec<u32>> {
    (0..count)
        .map(|s| {
            (0..len)
                .map(|i| ((i * 37 + s * 101 + 11) as u32 % (vocab - 2)) + 2)
                .collect()
        })
        .collect()
}

/// An artifact directory beside `source`, removed with it.
fn artifact(
    source: &Checkpoint,
    plan: &WeightPlan,
    packer: &Packer<'_>,
) -> Result<(Checkpoint, Manifest)> {
    let output = Checkpoint(
        source
            .0
            .with_extension(format!("tur-{}", uuid::Uuid::new_v4())),
    );
    let manifest = write_artifact(
        &Conversion {
            source: &source.0,
            identity: "fixture@0",
            output: &output.0,
            calibration: "synthetic token sequences",
        },
        plan,
        packer,
    )?;
    Ok((output, manifest))
}

/// Uniform Q4K, then the first half of the projections in `Q8_0` with each embedding
/// storage the checkpoint allows.
fn plans(config: &Config) -> Result<Vec<WeightPlan>> {
    let mut plans = vec![WeightPlan::uniform(config, WeightFormat::Q4K)?];
    let projections = config.projections()?;
    let mut mixed = WeightPlan::uniform(config, WeightFormat::Q4K)?;
    for p in &projections[..projections.len() / 2] {
        mixed.projections.insert(p.name.clone(), Precision::Q8_0);
    }
    for embedding in [
        Embedding::Table(Precision::Q8_0),
        Embedding::Table(Precision::Q6k),
        Embedding::Head,
    ] {
        if embedding != Embedding::Head || config.tied_embeddings() {
            plans.push(WeightPlan {
                embedding,
                ..mixed.clone()
            });
        }
    }
    Ok(plans)
}

#[test]
fn test_prepacked_artifacts_load_exactly_what_conversion_computes() -> Result<()> {
    let _gpu = GPU.lock();
    let tokens = samples(1, 40, 160).remove(0);
    let rows: Vec<usize> = (0..tokens.len()).collect();
    for tied in [false, true] {
        let source = small(tied)?;
        let config = Config::from_json(&std::fs::read_to_string(source.0.join("config.json"))?)?;
        for plan in plans(&config)? {
            let (dir, manifest) = artifact(&source, &plan, &Packer::RTN)?;
            assert_eq!(Manifest::read(&dir.0)?, manifest);
            assert_eq!(manifest.plan, plan);
            let prepacked = Weights::Prepacked(Arc::new(manifest));
            for device in [Device::Cpu, Device::new_metal(0)?] {
                let converted = load(&source.0, Weights::Plan(Arc::new(plan.clone())), &device)?;
                let stored = load(&dir.0, prepacked.clone(), &device)?;
                let (_, expected) = converted.trace(&tokens, &rows)?;
                let (_, actual) = stored.trace(&tokens, &rows)?;
                assert_eq!(
                    common::rows(&actual)?,
                    common::rows(&expected)?,
                    "{:?} tied {tied} {device:?}",
                    plan.embedding
                );
                let (description, bytes, _) = Qwen35::memory_requirements(
                    &config,
                    &options(prepacked.clone(), &device),
                    64,
                    &[64],
                )?;
                assert_eq!(bytes, stored.weight_bytes());
                assert_eq!(bytes, converted.weight_bytes());
                assert_eq!(description, stored.description());
                assert_ne!(
                    stored.description().identity,
                    converted.description().identity
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_shared_head_embeddings_hold_no_second_table() -> Result<()> {
    let source = small(true)?;
    let config = Config::from_json(&std::fs::read_to_string(source.0.join("config.json"))?)?;
    let table = WeightPlan::uniform(&config, WeightFormat::Q8_0)?;
    let head = WeightPlan {
        embedding: Embedding::Head,
        ..table.clone()
    };
    let with_table = load(&source.0, Weights::Plan(Arc::new(table)), &Device::Cpu)?;
    let with_head = load(&source.0, Weights::Plan(Arc::new(head)), &Device::Cpu)?;
    assert_eq!(
        with_table.weight_bytes() - with_head.weight_bytes(),
        config.vocab_size() * config.hidden_size() * 2
    );
    Ok(())
}

#[test]
fn test_artifact_validation_rejects_changed_configs_stale_manifests_and_conversion() -> Result<()> {
    let source = small(false)?;
    let config = Config::from_json(&std::fs::read_to_string(source.0.join("config.json"))?)?;
    let plan = WeightPlan::uniform(&config, WeightFormat::Q4K)?;
    let (dir, manifest) = artifact(&source, &plan, &Packer::RTN)?;
    // Conversion formats never read an artifact as if it were a checkpoint.
    assert!(load(&dir.0, WeightFormat::Q4K.into(), &Device::Cpu).is_err());
    // A manifest read before the artifact changed no longer describes it.
    let mut stale = manifest;
    stale.plan = WeightPlan::uniform(&config, WeightFormat::Q8_0)?;
    assert!(load(&dir.0, Weights::Prepacked(Arc::new(stale)), &Device::Cpu).is_err());
    // An existing artifact is never overwritten.
    assert!(
        write_artifact(
            &Conversion {
                source: &source.0,
                identity: "fixture@0",
                output: &dir.0,
                calibration: "",
            },
            &plan,
            &Packer::RTN,
        )
        .is_err()
    );
    // Any config change invalidates the artifact before weights are read.
    let config_path = dir.0.join("config.json");
    let original = std::fs::read_to_string(&config_path)?;
    std::fs::write(&config_path, original.replace("1e-6", "1e-5"))?;
    assert!(Manifest::read(&dir.0).is_err());
    std::fs::write(&config_path, &original)?;
    assert!(Manifest::read(&dir.0).is_ok());
    // A truncated payload no longer matches its header.
    let weights = dir.0.join(tur_models::models::ARTIFACT_FILE);
    let bytes = std::fs::read(&weights)?;
    std::fs::write(&weights, &bytes[..bytes.len() - 1])?;
    assert!(Manifest::read(&dir.0).is_err());
    Ok(())
}

#[test]
fn test_calibration_records_every_projection_input() -> Result<()> {
    for (json, experts) in [(SMALL, None), (MOE, Some((8, 2)))] {
        let source = checkpoint(json, 0.05, 0.1)?;
        let model = load(&source.0, WeightFormat::F32.into(), &Device::Cpu)?;
        let calibration = model.calibrate(&samples(2, 24, 96), true)?;
        assert_eq!((calibration.samples, calibration.tokens), (2, 48));
        let projections = model.config().projections()?;
        assert_eq!(calibration.sites.len(), projections.len());
        for p in &projections {
            let site = &calibration.sites[&p.name];
            assert_eq!(site.tokens.len(), p.experts);
            match (experts, p.experts) {
                (Some((_, top_k)), n) if n > 1 => {
                    assert_eq!(site.tokens.iter().sum::<usize>(), 48 * top_k, "{}", p.name);
                    assert!(site.hessian.is_none());
                }
                _ => {
                    assert_eq!(site.tokens, vec![48], "{}", p.name);
                    let hessian = site.hessian.as_ref().map_or(&[][..], Vec::as_slice);
                    assert_eq!(hessian.len(), p.cols * p.cols);
                    for (c, &square) in site.squares[0].iter().enumerate() {
                        let diagonal = hessian[c * p.cols + c];
                        assert!(
                            (diagonal - square).abs() <= 1e-4 * square.abs().max(1e-3),
                            "{} {c}: {diagonal} {square}",
                            p.name
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_streamed_calibration_packs_the_same_weights_as_resident_calibration() -> Result<()> {
    let _gpu = GPU.lock();
    for (json, tied) in [(SMALL, false), (SMALL, true), (MOE, false)] {
        let json = json.replace(
            "\"tie_word_embeddings\": false",
            &format!("\"tie_word_embeddings\": {tied}"),
        );
        let source = checkpoint(&json, 0.05, 0.1)?;
        let config = Config::from_json(&json)?;
        let plan = WeightPlan {
            embedding: if tied {
                Embedding::Head
            } else {
                Embedding::Table(Precision::Bf16)
            },
            ..WeightPlan::uniform(&config, WeightFormat::Q4K)?
        };
        let data = vec![vec![2, 8, 2], vec![5], vec![7, 11, 3, 2, 7]];
        for device in [Device::Cpu, Device::new_metal(0)?] {
            let model = load(&source.0, WeightFormat::F32.into(), &device)?;
            for method in [Method::Imatrix, Method::Gptq] {
                let calibration = model.calibrate(&data, method == Method::Gptq)?;
                let packer = Packer {
                    method,
                    damp: 0.01,
                    calibration: Some(&calibration),
                };
                let (expected, manifest) = artifact(&source, &plan, &packer)?;
                let output = Checkpoint(
                    source
                        .0
                        .with_extension(format!("stream-{}", uuid::Uuid::new_v4())),
                );
                let actual = write_calibrated_artifact(
                    &Conversion {
                        source: &source.0,
                        identity: "fixture@0",
                        output: &output.0,
                        calibration: "synthetic token sequences",
                    },
                    &plan,
                    &Packer {
                        calibration: None,
                        ..packer
                    },
                    &data,
                    &device,
                )?;
                assert_eq!(actual, manifest);
                // Compare stored payloads, independently of loading and inference.
                let expected = candle_core::safetensors::load(
                    expected.0.join(tur_models::models::ARTIFACT_FILE),
                    &Device::Cpu,
                )?;
                let actual = candle_core::safetensors::load(
                    output.0.join(tur_models::models::ARTIFACT_FILE),
                    &Device::Cpu,
                )?;
                assert_eq!(actual.len(), expected.len());
                for (name, expected) in expected {
                    let actual = &actual[&name];
                    assert_eq!(actual.dims(), expected.dims(), "{name}");
                    assert_eq!(actual.dtype(), expected.dtype(), "{name}");
                    let actual = actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let expected = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(actual == expected, "{name} {method:?} {device:?}");
                }
                assert!(
                    !std::fs::read_dir(&output.0)?.any(|entry| entry.is_ok_and(|e| e
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".tur-calibration-")))
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_streamed_packing_failure_cleans_activations_without_publishing_an_artifact() -> Result<()> {
    let source = small(false)?;
    let config = Config::from_json(SMALL)?;
    let plan = WeightPlan::uniform(&config, WeightFormat::Q4K)?;
    let output = Checkpoint(source.0.with_extension("failed-stream"));
    let result = write_calibrated_artifact(
        &Conversion {
            source: &source.0,
            identity: "fixture@0",
            output: &output.0,
            calibration: "synthetic token sequences",
        },
        &plan,
        &Packer {
            method: Method::Gptq,
            damp: 0.0,
            calibration: None,
        },
        &[vec![2, 3, 2]],
        &Device::Cpu,
    );
    assert!(matches!(
        result,
        Err(tur_core::TurError::InvalidExecution(_))
    ));
    assert!(!output.0.join(tur_models::models::ARTIFACT_FILE).exists());
    for entry in std::fs::read_dir(&output.0)? {
        assert!(
            !entry?
                .file_name()
                .to_string_lossy()
                .starts_with(".tur-calibration-")
        );
    }
    Ok(())
}

#[test]
fn test_calibrated_artifacts_reduce_divergence_on_their_calibration_data() -> Result<()> {
    let source = small(false)?;
    let model = load(&source.0, WeightFormat::F32.into(), &Device::Cpu)?;
    let data = samples(4, 64, 160);
    let calibration = model.calibrate(&data, true)?;
    let reference = Reference::new(&model, data)?;
    let plan = WeightPlan::uniform(model.config(), WeightFormat::Q4K)?;
    let mut divergences = Vec::new();
    for method in [Method::Rtn, Method::Imatrix, Method::Gptq] {
        let packer = Packer {
            method,
            damp: 0.01,
            calibration: Some(&calibration),
        };
        let (dir, manifest) = artifact(&source, &plan, &packer)?;
        assert_eq!(manifest.quantizer.method, method);
        let info = manifest
            .calibration
            .as_ref()
            .ok_or_else(|| tur_core::execution::invalid("missing calibration"))?;
        assert_eq!((info.samples, info.tokens), (4, 256));
        let quantized = load(&dir.0, Weights::Prepacked(Arc::new(manifest)), &Device::Cpu)?;
        divergences.push(reference.divergence(&quantized)?);
    }
    let (rtn, imatrix, gptq) = (divergences[0], divergences[1], divergences[2]);
    assert!(rtn > 0.0 && gptq < rtn && imatrix < rtn, "{divergences:?}");
    Ok(())
}

#[test]
fn test_divergence_is_zero_for_identical_weights_on_metal() -> Result<()> {
    let _gpu = GPU.lock();
    let source = small(false)?;
    let device = Device::new_metal(0)?;
    let model = load(&source.0, WeightFormat::F32.into(), &device)?;
    let reference = Reference::new(&model, samples(2, 32, 160))?;
    assert!(reference.divergence(&model)?.abs() < 1e-6);
    let quantized = load(&source.0, WeightFormat::Q4K.into(), &device)?;
    assert!(reference.divergence(&quantized)? > 0.0);
    Ok(())
}

#[test]
fn test_sensitivity_allocation_fits_its_budget_and_restores_the_model() -> Result<()> {
    let source = small(false)?;
    let mut model = load(&source.0, WeightFormat::F32.into(), &Device::Cpu)?;
    let config = model.config().clone();
    let reference = Reference::new(&model, samples(2, 48, 160))?;
    let measured = sensitivities(
        &mut model,
        &reference,
        &[Precision::Q4k, Precision::Q8_0],
        &Packer::RTN,
        |_| {},
    )?;
    assert_eq!(measured.len(), 2 * config.projections()?.len());
    assert!(measured.iter().all(|s| s.kl.is_finite() && s.kl >= -1e-9));
    // Every swap was undone.
    assert!(reference.divergence(&model)?.abs() < 1e-9);
    let first = config.projections()?.remove(0);
    let wrong = tur_metal::packed::PackedLinear::dense(
        &candle_core::Tensor::zeros((first.rows, first.cols + 1), DType::F32, &Device::Cpu)?,
        &Device::Cpu,
    )?;
    assert!(model.replace_projection(&first.name, wrong).is_err());
    let embedding = Embedding::Table(Precision::Bf16);
    let q4k = WeightPlan::uniform(&config, WeightFormat::Q4K)?;
    let q8 = WeightPlan::uniform(&config, WeightFormat::Q8_0)?;
    let budget = usize::midpoint(q4k.bytes(&config)?, q8.bytes(&config)?);
    let plan = allocate(&config, &measured, budget, Precision::Q4k, embedding)?;
    assert!(plan.bytes(&config)? <= budget);
    let divergence = |plan: &WeightPlan| -> Result<f64> {
        reference.divergence(&load(
            &source.0,
            Weights::Plan(Arc::new(plan.clone())),
            &Device::Cpu,
        )?)
    };
    let (mixed, uniform) = (divergence(&plan)?, divergence(&q4k)?);
    assert!(mixed < uniform, "mixed {mixed}, uniform q4k {uniform}");
    Ok(())
}
