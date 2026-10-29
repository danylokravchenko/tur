//! Independent Gemma text conformance and paged-state lifecycle coverage.
#[cfg(feature = "metal")]
use crate::model_common as common;
use crate::model_common::{max_relative, rows};
use candle_core::{DType, Device};
use std::path::{Path, PathBuf};
#[cfg(feature = "metal")]
use std::sync::Arc;
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, invalid},
};
#[cfg(feature = "metal")]
use tur_metal::{pool::PoolConfig, state::ModelState};
#[cfg(feature = "metal")]
use tur_models::hybrid::{Boundaries, HybridModel, StatePool};
use tur_models::models::{
    Execution, LoadOptions, WeightFormat,
    gemma4::{Config, Gemma4},
};

fn fixture(kind: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gemma4")
        .join(kind)
}
fn options(
    device: Device,
    execution: Execution,
    format: WeightFormat,
    kv: KvFormat,
) -> LoadOptions {
    LoadOptions {
        revision: "gemma4-transformers-5.20.2".into(),
        weights: format.into(),
        device,
        execution,
        kv: KvLayout::new(4, kv),
        recurrent: DType::F32,
        mtp: None,
    }
}
#[cfg(feature = "metal")]
fn pool(model: &Gemma4) -> Result<StatePool> {
    StatePool::new(
        Arc::new(model.description()),
        model.device(),
        PoolConfig {
            pages: 256,
            window_pages: 256,
            recurrent_slots: 1,
        },
    )
}

fn conformance(device: &Device, execution: Execution) -> Result<()> {
    for kind in ["dense", "moe", "unified"] {
        let dir = fixture(kind);
        let options = options(device.clone(), execution, WeightFormat::F32, KvFormat::F32);
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let (_, expected_bytes, _) = Gemma4::memory_requirements(&config, &options, 9, &[9])?;
        let model = Gemma4::load(&dir, &options)?;
        assert_eq!(model.weight_bytes(), expected_bytes);
        let golden = candle_core::safetensors::load(dir.join("golden.safetensors"), &Device::Cpu)?;
        let tokens = golden
            .get("tokens")
            .ok_or_else(|| invalid("missing tokens"))?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let requested: Vec<usize> = (0..tokens.len()).collect();
        let (trace, logits) = model.trace(&tokens, &requested)?;
        assert_eq!(trace.len(), 4);
        for (i, hidden) in trace.iter().enumerate() {
            let reference = golden
                .get(&format!("layer.{i}"))
                .ok_or_else(|| invalid("missing layer reference"))?;
            let error = max_relative(&rows(hidden)?, &rows(reference)?);
            assert!(error < 1e-4, "{kind} layer {i}: {error}");
        }
        let reference = golden
            .get("logits")
            .ok_or_else(|| invalid("missing reference logits"))?;
        let error = max_relative(&rows(&logits)?, &rows(reference)?);
        assert!(error < 1e-4, "{kind} logits: {error}");
    }
    Ok(())
}
#[test]
fn test_gemma4_reference_matches_transformers_per_layer_and_logits() -> Result<()> {
    conformance(&Device::Cpu, Execution::Reference)
}
#[test]
#[cfg(feature = "metal")]
fn test_gemma4_metal_matches_transformers_per_layer_and_logits() -> Result<()> {
    conformance(&Device::new_metal(0)?, Execution::Fused)
}

#[test]
#[cfg(feature = "metal")]
fn test_gemma4_chunking_batching_and_forks_preserve_window_and_shared_kv() -> Result<()> {
    let device = Device::new_metal(0)?;
    let tokens: Vec<u32> = (0..41).map(|i| (i * 17 % 50 + 2) as u32).collect();
    for format in [KvFormat::F32] {
        let model = Gemma4::load(
            &fixture("dense"),
            &options(device.clone(), Execution::Fused, WeightFormat::F32, format),
        )?;
        let pool = pool(&model)?;
        assert_eq!(
            model.description().layers.len(),
            2,
            "sharing suffix must not allocate K/V layers"
        );
        let (whole, _) = common::prefill(&model, ModelState::new(&pool), &tokens, 41)?;
        for chunk in [1, 3, 7, 16] {
            let (chunked, _) = common::prefill(&model, ModelState::new(&pool), &tokens, chunk)?;
            let error = max_relative(&chunked, &whole);
            assert!(error < 1e-4, "{format} chunk {chunk}: {error}");
        }
        let (_, prefix) = common::prefill(&model, ModelState::new(&pool), &tokens[..13], 5)?;
        let other = ModelState::new(&pool);
        let mut request = vec![
            (&prefix, &tokens[13..20], Boundaries::EveryToken),
            (&other, &tokens[20..25], Boundaries::Final),
        ];
        let (steps, reservations) = common::forward(&model, &request)?;
        let (other_solo, _) = common::prefill(&model, ModelState::new(&pool), &tokens[20..25], 5)?;
        for (step, expected) in steps
            .into_iter()
            .zip([&whole[13..20], other_solo.as_slice()])
        {
            let step = step?;
            assert!(
                max_relative(&rows(&step.logits)?, expected) < 1e-4,
                "mixed batch isolation with {format}"
            );
        }
        let restored = reservations[0].commit(3)?;
        drop(reservations);
        request.clear();
        device.synchronize()?;
        pool.reclaim();
        let (continued, _) = common::prefill(&model, restored, &tokens[16..], 4)?;
        assert!(
            max_relative(&continued, &whole[16..]) < 1e-4,
            "rollback with {format}"
        );
    }
    Ok(())
}

#[test]
fn test_gemma4_factory_runs_through_shared_pipeline_and_accounts_weights() -> Result<()> {
    let checkpoint = crate::common::Checkpoint::new()?;
    for name in ["model.safetensors", "config.json"] {
        std::fs::copy(fixture("dense").join(name), checkpoint.0.join(name))?;
    }
    let factory = checkpoint.factory();
    let mut pipeline = tur::InferencePipeline::new(
        &factory,
        tur::backend::pipeline::PipelineConfig::reproducible(3),
    )?;
    let golden =
        candle_core::safetensors::load(fixture("dense").join("golden.safetensors"), &Device::Cpu)?;
    let tokens = golden
        .get("tokens")
        .ok_or_else(|| invalid("missing tokens"))?
        .flatten_all()?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;
    let logits = golden
        .get("logits")
        .ok_or_else(|| invalid("missing logits"))?
        .to_vec2::<f32>()?;
    let last = logits.last().ok_or_else(|| invalid("empty logits"))?;
    let expected = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(id, _)| format!("t{id}"))
        .ok_or_else(|| invalid("empty logit row"))?;
    let prompt = tokens
        .iter()
        .map(|t| format!("t{t}"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut request = tur::backend::pipeline::GenerationRequest::new(prompt, 1);
    request.sampling.temperature = Some(0.0);
    let result = pipeline.run(&request, |_| {})?;
    assert_eq!(result.generated_tokens, 1);
    assert_eq!(result.generated_text, expected);
    Ok(())
}

#[test]
#[ignore = "requires pinned Gemma 4 E2B checkpoint and Transformers golden"]
#[cfg(feature = "metal")]
fn test_gemma4_e2b_checkpoint_logits_match_transformers() -> Result<()> {
    let required = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .ok_or_else(|| invalid(format!("set {name}")))
    };
    let dir = required("TUR_GEMMA4_E2B")?;
    let golden = candle_core::safetensors::load(required("TUR_GEMMA4_GOLDEN")?, &Device::Cpu)?;
    let tokens = golden
        .get("tokens")
        .ok_or_else(|| invalid("missing tokens"))?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let model = Gemma4::load(
        &dir,
        &options(
            Device::new_metal(0)?,
            Execution::Fused,
            WeightFormat::Q8_0,
            KvFormat::F32,
        ),
    )?;
    let model = Arc::new(model);
    let requested: Vec<usize> = (0..tokens.len()).collect();
    let (_, logits) = model.trace(&tokens, &requested)?;
    let reference = golden
        .get("logits")
        .ok_or_else(|| invalid("missing reference logits"))?;
    let actual = logits
        .argmax(1)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    let expected = reference.argmax(1)?.to_vec1::<u32>()?;
    assert_eq!(actual.len(), expected.len());
    let agreement =
        actual.iter().zip(&expected).filter(|(a, b)| a == b).count() as f64 / expected.len() as f64;
    // Quantized weights are judged by the existing 95% next-token quality floor, rather
    // than an F32 arithmetic error bound; the offline F32 oracles gate the equations.
    assert!(
        agreement >= 0.95,
        "E2B Q8 top-1 agreement {agreement}; actual {actual:?}; expected {expected:?}"
    );
    let greedy = golden
        .get("greedy")
        .ok_or_else(|| invalid("regenerate the Gemma golden with greedy continuation"))?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;
    let stops = [1, 106, 50];
    let output = common::generate(
        Arc::clone(&model),
        model.device().clone(),
        PoolConfig {
            pages: 64,
            window_pages: 64,
            recurrent_slots: 1,
        },
        &tokens,
        &stops,
        64,
    )?;
    let prefix = output
        .iter()
        .zip(&greedy)
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        2 * prefix >= greedy.len(),
        "E2B greedy prefix {prefix}/{}: {output:?}",
        greedy.len()
    );
    assert!(
        output.last().is_some_and(|t| stops.contains(t)),
        "E2B did not terminate: {output:?}"
    );

    Ok(())
}

#[test]
fn test_gemma4_e2b_shapes_match_pinned_checkpoint_headers_and_reject_unsupported_math() -> Result<()>
{
    let dir = fixture("e2b");
    let json = std::fs::read_to_string(dir.join("config.json"))?;
    let config = Config::from_json(&json)?;
    let headers: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("headers.json"))?)?;
    for (name, shape) in config.weight_shapes()? {
        let actual: Vec<usize> = serde_json::from_value(headers[&name]["shape"].clone())?;
        assert_eq!(actual, shape, "{name}");
    }
    let defaults = config.generation_defaults(None)?;
    assert_eq!(defaults.eos_tokens, vec![1, 106]);
    for (field, value) in [
        ("attention_bias", serde_json::json!(true)),
        ("num_kv_shared_layers", serde_json::json!(35)),
        ("use_bidirectional_attention", serde_json::json!("all")),
        ("global_head_dim", serde_json::json!(0)),
    ] {
        let mut raw: serde_json::Value = serde_json::from_str(&json)?;
        raw["text_config"][field] = value;
        assert!(Config::from_json(&raw.to_string()).is_err(), "{field}");
    }
    let mut raw: serde_json::Value = serde_json::from_str(&json)?;
    raw["text_config"]["rope_parameters"]["full_attention"]["factor"] = serde_json::json!(2.0);
    assert!(Config::from_json(&raw.to_string()).is_err());
    Ok(())
}

#[test]
fn test_gemma4_chat_template_matches_pinned_transformers_bytes() -> Result<()> {
    use tur_models::backend::chat_template::ChatTemplate;
    let template = ChatTemplate::from_model_dir(&fixture("e2b"))?;
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../fixtures/gemma4/templates.json"))?;
    for case in cases {
        let messages = case["messages"]
            .as_array()
            .ok_or_else(|| invalid("missing messages"))?
            .iter()
            .map(crate::reference::message)
            .collect::<Result<Vec<_>>>()?;
        let tools = crate::reference::tools(&case["tools"]);
        let rendered = template.render(&messages, &tools, &crate::reference::options(&case))?;
        assert_eq!(
            rendered,
            case["text"]
                .as_str()
                .ok_or_else(|| invalid("missing rendered template"))?,
            "{}",
            case["name"]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "metal")]
fn test_gemma4_prepacked_artifacts_match_load_time_conversion_exactly() -> Result<()> {
    use tur_models::models::{
        Conversion, Manifest, Method, Packer, WeightPlan, Weights, gemma4::write_artifact,
    };
    let device = Device::new_metal(0)?;
    for kind in ["dense", "moe", "unified"] {
        let source = fixture(kind);
        let config = Config::from_json(&std::fs::read_to_string(source.join("config.json"))?)?;
        let golden =
            candle_core::safetensors::load(source.join("golden.safetensors"), &Device::Cpu)?;
        let tokens = golden
            .get("tokens")
            .ok_or_else(|| invalid("missing tokens"))?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let requested: Vec<usize> = (0..tokens.len()).collect();
        for format in [WeightFormat::F32, WeightFormat::Q8_0] {
            let output = common::Checkpoint(
                std::env::temp_dir().join(format!("tur-gemma4-{kind}-{}", uuid::Uuid::new_v4())),
            );
            let conversion = Conversion {
                source: &source,
                identity: "gemma4-fixture@0",
                output: &output.0,
                calibration: "",
            };
            let plan = WeightPlan::uniform(&config, format)?;
            let manifest = write_artifact(&conversion, &plan, &Packer::RTN)?;
            assert_eq!(Manifest::read(&output.0)?, manifest);
            let converted = Gemma4::load(
                &source,
                &options(device.clone(), Execution::Fused, format, KvFormat::F16),
            )?;
            let stored = Gemma4::load(
                &output.0,
                &LoadOptions {
                    weights: Weights::Prepacked(Arc::new(manifest)),
                    ..options(device.clone(), Execution::Fused, format, KvFormat::F16)
                },
            )?;
            assert_eq!(stored.weight_bytes(), converted.weight_bytes());
            let (_, expected) = converted.trace(&tokens, &requested)?;
            let (_, actual) = stored.trace(&tokens, &requested)?;
            assert_eq!(rows(&actual)?, rows(&expected)?, "{kind} {format:?}");
            // An artifact directory only loads as prepacked weights.
            assert!(
                Gemma4::load(
                    &output.0,
                    &options(device.clone(), Execution::Fused, format, KvFormat::F16)
                )
                .is_err()
            );
        }
        let calibrated = Packer {
            method: Method::Gptq,
            ..Packer::RTN
        };
        let output = common::Checkpoint(
            std::env::temp_dir().join(format!("tur-gemma4-{kind}-{}", uuid::Uuid::new_v4())),
        );
        let conversion = Conversion {
            source: &source,
            identity: "gemma4-fixture@0",
            output: &output.0,
            calibration: "",
        };
        let plan = WeightPlan::uniform(&config, WeightFormat::Q8_0)?;
        assert!(write_artifact(&conversion, &plan, &calibrated).is_err());
    }
    Ok(())
}

/// Teacher-forced assistant drafts over the dense target: the target's final hidden state
/// at `p` and the token at `p + 1` draft the token after, against Transformers.
#[cfg(feature = "metal")]
fn assistant_drafts(device: &Device, execution: Execution) -> Result<()> {
    use tur_models::hybrid::{DraftForward, DraftSequence, PackedForward, SequenceForward};
    use tur_models::models::DraftHead;
    let golden = candle_core::safetensors::load(
        fixture("assistant").join("golden.safetensors"),
        &Device::Cpu,
    )?;
    let get = |name: &str| {
        golden
            .get(name)
            .ok_or_else(|| invalid(format!("missing {name}")))
    };
    let tokens = get("tokens")?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let expected = rows(get("logits")?)?;
    let model = Gemma4::load(
        &fixture("dense"),
        &LoadOptions {
            mtp: Some(DraftHead {
                vocab: None,
                model: Some(fixture("assistant")),
            }),
            ..options(device.clone(), execution, WeightFormat::F32, KvFormat::F32)
        },
    )?;
    let pool = pool(&model)?;
    // Decode one token at a time and draft at each committed end, as the engine does: a
    // committed state no longer holds window pages behind its window.
    let mut state = ModelState::new(&pool);
    for (p, expected) in expected.iter().enumerate() {
        let mut reservation = state.reserve(1, Boundaries::Final)?;
        reservation.prepare()?;
        let binding = reservation.binding();
        let step = model
            .forward(&PackedForward {
                tokens: &tokens[p..=p],
                positions: &[p],
                sequences: &[SequenceForward {
                    tokens: 0..1,
                    state: &binding,
                    logits: &[],
                    hidden: true,
                }],
                pool: &pool,
            })?
            .pop()
            .ok_or_else(|| invalid("missing step"))??;
        let hidden = step
            .hidden
            .ok_or_else(|| invalid("missing hidden states"))?;
        state = reservation.commit(1)?;
        drop(reservation);
        let side = state.side(p, 1)?;
        let drafted = model
            .draft(&DraftForward {
                tokens: &tokens[p + 1..p + 2],
                positions: &[p],
                sequences: &[DraftSequence {
                    tokens: 0..1,
                    state: &side,
                    hidden: &hidden,
                    logits: true,
                }],
                pool: &pool,
            })?
            .pop()
            .ok_or_else(|| invalid("missing draft"))??
            .ok_or_else(|| invalid("draft returned no logits"))?;
        let actual = rows(&drafted)?;
        let actual = actual.first().ok_or_else(|| invalid("empty draft"))?;
        let masked = |row: &[f32]| row.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
        assert_eq!(masked(actual), masked(expected), "position {p} candidates");
        let finite = |row: &[f32]| {
            row.iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>()
        };
        let error = max_relative(&[finite(actual)], &[finite(expected)]);
        assert!(error < 1e-4, "{execution:?} position {p}: {error}");
    }
    Ok(())
}

#[test]
#[cfg(feature = "metal")]
fn test_gemma4_assistant_drafts_match_transformers_on_metal_and_cpu() -> Result<()> {
    assistant_drafts(&Device::new_metal(0)?, Execution::Fused)?;
    assistant_drafts(&Device::Cpu, Execution::Reference)
}

#[test]
#[ignore = "requires pinned Gemma 4 E2B and assistant checkpoints and a Transformers golden"]
#[cfg(feature = "metal")]
fn test_gemma4_e2b_assistant_drafts_agree_with_transformers_past_the_window() -> Result<()> {
    use tur_models::hybrid::{DraftForward, DraftSequence, PackedForward, SequenceForward};
    use tur_models::models::DraftHead;
    let required = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .ok_or_else(|| invalid(format!("set {name}")))
    };
    let golden =
        candle_core::safetensors::load(required("TUR_GEMMA4_ASSISTANT_GOLDEN")?, &Device::Cpu)?;
    let get = |name: &str| {
        golden
            .get(name)
            .ok_or_else(|| invalid(format!("missing {name}")))
    };
    let tokens = get("tokens")?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let positions = get("positions")?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let expected = get("logits")?.argmax(1)?.to_vec1::<u32>()?;
    let model = Gemma4::load(
        &required("TUR_GEMMA4_E2B")?,
        &LoadOptions {
            mtp: Some(DraftHead {
                vocab: None,
                model: Some(required("TUR_GEMMA4_ASSISTANT")?),
            }),
            ..options(
                Device::new_metal(0)?,
                Execution::Fused,
                WeightFormat::Q8_0,
                KvFormat::F16,
            )
        },
    )?;
    let pool = pool(&model)?;
    let mut state = ModelState::new(&pool);
    let mut actual = Vec::with_capacity(positions.len());
    // Prefill up to each reference position and draft at that committed end.
    for &p in &positions {
        let p = p as usize;
        let piece = &tokens[state.position()..=p];
        let mut reservation = state.reserve(piece.len(), Boundaries::Final)?;
        reservation.prepare()?;
        let binding = reservation.binding();
        let packed: Vec<usize> = (state.position()..=p).collect();
        let step = model
            .forward(&PackedForward {
                tokens: piece,
                positions: &packed,
                sequences: &[SequenceForward {
                    tokens: 0..piece.len(),
                    state: &binding,
                    logits: &[],
                    hidden: true,
                }],
                pool: &pool,
            })?
            .pop()
            .ok_or_else(|| invalid("missing step"))??;
        let hidden = step
            .hidden
            .ok_or_else(|| invalid("missing hidden states"))?;
        state = reservation.commit(piece.len())?;
        drop(reservation);
        let side = state.side(p, 1)?;
        let drafted = model
            .draft(&DraftForward {
                tokens: &tokens[p + 1..p + 2],
                positions: &[p],
                sequences: &[DraftSequence {
                    tokens: 0..1,
                    state: &side,
                    hidden: &hidden.narrow(0, piece.len() - 1, 1)?,
                    logits: true,
                }],
                pool: &pool,
            })?
            .pop()
            .ok_or_else(|| invalid("missing draft"))??
            .ok_or_else(|| invalid("draft returned no logits"))?;
        actual.push(
            drafted
                .argmax(1)?
                .to_device(&Device::Cpu)?
                .to_vec1::<u32>()?[0],
        );
        model.device().synchronize()?;
        pool.reclaim();
    }
    let agreement =
        actual.iter().zip(&expected).filter(|(a, b)| a == b).count() as f64 / expected.len() as f64;
    // A Q8_0 target and F32 reference differ in hidden states, so drafts are judged by
    // agreement; the offline fixture gates the drafter's equations at 1e-4.
    assert!(
        agreement >= 0.9,
        "assistant draft agreement {agreement}; actual {actual:?}; expected {expected:?}"
    );
    assert!(
        state.window_count() < state.page_count(),
        "a prompt past the window releases sliding pages"
    );
    Ok(())
}
