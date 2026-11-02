//! Independent `MoE` conformance and packed execution over real hybrid state.

#[cfg(feature = "metal")]
use crate::common::Checkpoint;
use crate::model_common::{checkpoint, max_relative};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
#[cfg(feature = "metal")]
use tur::{
    InferencePipeline,
    backend::pipeline::{GenerationRequest, PipelineConfig},
};
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
            audio: false,
            vision: false,
            expert_cache: None,
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
                audio: false,
                vision: false,
                expert_cache: None,
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

/// Expert banks behind a cache read the artifact's own blocks: under eviction, between and
/// within passes, every logit is bit-identical to keeping all of them resident.
#[cfg(feature = "metal")]
#[test]
fn test_expert_cache_over_artifacts_is_bit_identical_to_resident() -> Result<()> {
    use tur_models::models::{
        Conversion, Manifest, Packer, WeightPlan, Weights,
        qwen3_5::{artifact_config, write_artifact},
    };
    let fixture = checkpoint(CONFIG, 0.03, 0.1)?;
    let device = Device::new_metal(0)?;
    let (ids, _, _) = golden()?;
    let rows: Vec<usize> = (0..ids.len()).collect();
    let raw: serde_json::Value = serde_json::from_str(CONFIG)?;
    let number = |key: &str| -> Result<usize> {
        raw[key]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| invalid(format!("fixture has no {key}")))
    };
    let (experts, inner, hidden) = (
        number("num_experts")?,
        number("moe_intermediate_size")?,
        number("hidden_size")?,
    );
    for format in [WeightFormat::Q8_0, WeightFormat::Q4K] {
        let output = fixture.0.with_extension(format!("{format:?}"));
        write_artifact(
            &Conversion {
                source: &fixture.0,
                identity: "moe-fixture-v1",
                output: &output,
                calibration: "",
            },
            &WeightPlan::uniform(&artifact_config(&fixture.0)?, format)?,
            &Packer::RTN,
        )?;
        let result = (|| -> Result<()> {
            let manifest = std::sync::Arc::new(Manifest::read(&output)?);
            let options = |expert_cache: Option<usize>| LoadOptions {
                revision: "moe-fixture-v1".into(),
                weights: Weights::Prepacked(std::sync::Arc::clone(&manifest)),
                device: device.clone(),
                execution: Execution::Fused,
                kv: KvLayout::new(16, KvFormat::F32),
                recurrent: DType::F32,
                mtp: None,
                audio: false,
                vision: false,
                expert_cache,
            };
            let resident = Qwen35::load(&output, &options(None))?;
            let expected = resident.trace(&ids, &rows)?.1.to_vec2::<f32>()?;
            let bytes = |rows: usize, cols: usize| -> Result<usize> {
                tur_models::models::Precision::from(format).bytes(rows, cols)
            };
            let expert = bytes(2 * inner, hidden)? + bytes(hidden, inner)?;
            assert!(Qwen35::load(&output, &options(Some((experts - 1) * expert))).is_err());
            for slots in [experts, experts + 3] {
                let opts = options(Some(slots * expert));
                let model = Qwen35::load(&output, &opts)?;
                for pass in 0..2 {
                    let actual = model.trace(&ids, &rows)?.1.to_vec2::<f32>()?;
                    for (a, e) in actual.iter().flatten().zip(expected.iter().flatten()) {
                        assert_eq!(a.to_bits(), e.to_bits(), "{format:?} {slots} {pass}");
                    }
                }
                let stats = model
                    .expert_stats()
                    .ok_or_else(|| invalid("missing expert stats"))?;
                assert!(
                    stats.misses > 0 && stats.evictions > 0,
                    "{format:?} {slots}"
                );
                let (_, planned, _) =
                    Qwen35::memory_requirements(model.config(), &opts, 32, &[32])?;
                assert_eq!(planned, model.weight_bytes(), "{format:?}");
                assert!(model.weight_bytes() < resident.weight_bytes());
            }
            Ok(())
        })();
        drop(std::fs::remove_dir_all(&output));
        result?;
    }
    // Banks converted at load time are not on disk in the form the kernels read.
    assert!(
        Qwen35::load(
            &fixture.0,
            &LoadOptions {
                revision: "moe-fixture-v1".into(),
                weights: WeightFormat::Q8_0.into(),
                device,
                execution: Execution::Fused,
                kv: KvLayout::new(16, KvFormat::F32),
                recurrent: DType::F32,
                mtp: None,
                audio: false,
                vision: false,
                expert_cache: Some(1 << 20),
            },
        )
        .is_err()
    );
    Ok(())
}

/// A `Q8_0` artifact of the `MoE` fixture beside a tokenizer, for a pipeline behind a cache.
#[cfg(feature = "metal")]
fn cached_artifact() -> Result<(Checkpoint, tur_models::backend::factory::ModelFactory)> {
    use tur_models::models::{
        Conversion, Packer, WeightPlan,
        qwen3_5::{artifact_config, write_artifact},
    };
    let served = Checkpoint::new()?;
    let weights = checkpoint(CONFIG, 0.03, 0.1)?;
    std::fs::write(served.0.join("config.json"), CONFIG)?;
    std::fs::copy(
        weights.0.join("model.safetensors"),
        served.0.join("model.safetensors"),
    )?;
    let source = served.0.with_extension("source");
    std::fs::rename(&served.0, &source)?;
    write_artifact(
        &Conversion {
            source: &source,
            identity: "moe-fixture-v1",
            output: &served.0,
            calibration: "",
        },
        &WeightPlan::uniform(&artifact_config(&source)?, WeightFormat::Q8_0)?,
        &Packer::RTN,
    )?;
    drop(std::fs::remove_dir_all(&source));
    let factory = tur_models::backend::factory::ModelFactory {
        format: None,
        device: Device::new_metal(0)?,
        // Ten of the fixture's 16 experts.
        expert_cache: Some(2 << 20),
        ..served.factory()
    };
    Ok((served, factory))
}

/// A layered pass computes what one pass does, and leaves the same recurrent state and keys:
/// a token decoded after it gives bit-identical logits.
#[cfg(feature = "metal")]
#[test]
fn test_layered_pass_over_recurrent_layers_matches_one_pass() -> Result<()> {
    use crate::model_common::{pool, prefill, rows};
    use tur_engine::hybrid::{HybridModel, LayerStep, PackedForward, SequenceForward};
    use tur_models::models::{Manifest, Weights};
    let (served, factory) = cached_artifact()?;
    let (ids, _, _) = golden()?;
    let (prefix, next) = ids.split_at(ids.len() - 1);
    let model = Qwen35::load(
        &served.0,
        &LoadOptions {
            revision: "moe-fixture-v1".into(),
            weights: Weights::Prepacked(std::sync::Arc::new(Manifest::read(&served.0)?)),
            device: factory.device.clone(),
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F32),
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            vision: false,
            expert_cache: factory.expert_cache,
        },
    )?;
    let layers = model
        .pausable_layers()
        .ok_or_else(|| invalid("a cached model pauses between layers"))?;
    let control = pool(&model, 16, 8)?;
    let (whole, state) = prefill(&model, ModelState::new(&control), prefix, prefix.len())?;
    let (decoded, _) = prefill(&model, state, next, 1)?;
    let split = pool(&model, 16, 8)?;
    let mut reservation = ModelState::new(&split).reserve(prefix.len(), Boundaries::Final)?;
    reservation.prepare()?;
    let binding = reservation.binding();
    let positions: Vec<usize> = (0..prefix.len()).collect();
    let sequences = [SequenceForward {
        tokens: 0..prefix.len(),
        state: &binding,
        logits: &positions,
        hidden: false,
    }];
    let batch = PackedForward {
        embeddings: &[],
        axial: &[],
        tokens: prefix,
        positions: &positions,
        sequences: &sequences,
        pool: &split,
    };
    let LayerStep::Paused(carry) = model.forward_layers(&batch, None, 1)? else {
        return Err(invalid("a group before the last layer must pause"));
    };
    assert_eq!(carry.next, 1);
    factory.device.synchronize()?;
    let LayerStep::Done(step) = model.forward_layers(&batch, Some(carry), layers)? else {
        return Err(invalid("the last layer must finish"));
    };
    let actual = rows(&step.logits)?;
    let (after, _) = prefill(&model, reservation.commit(prefix.len())?, next, 1)?;
    for (a, e) in actual
        .iter()
        .chain(&after)
        .flatten()
        .zip(whole.iter().chain(&decoded).flatten())
    {
        assert_eq!(a.to_bits(), e.to_bits());
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn long_prompt(tokens: usize) -> String {
    (0..tokens)
        .map(|i| format!("t{}", 10 + i % 80))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Runs `short` until it is decoding, submits `long`, and drains both. Returns their texts
/// and how many tokens `short` streamed before `long`'s first.
#[cfg(feature = "metal")]
fn stream_beside_prompt(pipeline: &mut InferencePipeline) -> Result<(String, String, usize)> {
    let short = pipeline.submit_request(&GenerationRequest::new("t10 t11 t12".into(), 30))?;
    let mut streamed = 0;
    while streamed < 2 {
        let step = pipeline.step_streaming()?;
        streamed += step.sampled_tokens.iter().filter(|t| t.0 == short).count();
    }
    let long = pipeline.submit_request(&GenerationRequest::new(long_prompt(300), 4))?;
    let (mut before, mut started) = (0, false);
    let mut texts = std::collections::HashMap::new();
    while pipeline.active_request_count() > 0 {
        let step = pipeline.step_streaming()?;
        for (id, _, _) in &step.sampled_tokens {
            started |= *id == long;
            before += usize::from(*id == short && !started);
        }
        for (id, result) in step.completed {
            texts.insert(id, result?.generated_text);
        }
    }
    let mut text = |id| {
        texts
            .remove(&id)
            .ok_or_else(|| invalid("missing completion"))
    };
    Ok((text(short)?, text(long)?, before))
}

#[cfg(feature = "metal")]
fn solo_texts(factory: &tur::ModelFactory, config: PipelineConfig) -> Result<(String, String)> {
    let mut solo = InferencePipeline::new(factory, config)?;
    solo.clear_prefix_cache();
    let short = solo.run(&GenerationRequest::new("t10 t11 t12".into(), 30), |_| {})?;
    solo.clear_prefix_cache();
    let long = solo.run(&GenerationRequest::new(long_prompt(300), 4), |_| {})?;
    Ok((short.generated_text, long.generated_text))
}

/// A stream runs between layer groups of a hybrid model's prompt, every output is what a
/// solo run produces, and the chunk's snapshot, with its recurrent state, serves a repeat.
#[cfg(feature = "metal")]
#[test]
fn test_layered_prefill_over_recurrent_layers_keeps_outputs_and_snapshots() -> Result<()> {
    let (_served, factory) = cached_artifact()?;
    let config = || PipelineConfig::new(2, 256);
    let (short, long) = solo_texts(&factory, config())?;
    let mut pipeline = InferencePipeline::new(&factory, config())?;
    let (short_text, long_text, before) = stream_beside_prompt(&mut pipeline)?;
    assert_eq!(short_text, short);
    assert_eq!(long_text, long);
    // Two groups of the first chunk: one decode between them and one after.
    assert!(before >= 2, "{before} tokens before the prompt's first");
    let repeat = pipeline.run(&GenerationRequest::new(long_prompt(300), 4), |_| {})?;
    assert_eq!(repeat.generated_text, long);
    assert!(repeat.cached_tokens >= 256, "{}", repeat.cached_tokens);
    // Switched off, nothing pauses and the outputs are the same.
    let mut off = config();
    off.scheduler.layered_prefill = false;
    let mut pipeline = InferencePipeline::new(&factory, off)?;
    assert!(!pipeline.scheduler().layered_prefill());
    let (short_text, long_text, _) = stream_beside_prompt(&mut pipeline)?;
    assert_eq!(short_text, short);
    assert_eq!(long_text, long);
    Ok(())
}

/// Cancelling between groups returns the chunk's pages and recurrent slot.
#[cfg(feature = "metal")]
#[test]
fn test_cancelling_between_layer_groups_releases_recurrent_slots() -> Result<()> {
    let (_served, factory) = cached_artifact()?;
    let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::new(2, 256))?;
    let short = pipeline.submit_request(&GenerationRequest::new("t10 t11 t12".into(), 12))?;
    pipeline.step_streaming()?;
    pipeline.step_streaming()?;
    let idle = pipeline.memory()?;
    let long = pipeline.submit_request(&GenerationRequest::new(long_prompt(256), 4))?;
    pipeline.step_streaming()?;
    let paused = pipeline.memory()?;
    assert!(paused.pages.leased >= idle.pages.leased + 16);
    assert!(paused.slots.leased > idle.slots.leased);
    pipeline.cancel(long)?;
    let mut finished = false;
    while pipeline.active_request_count() > 0 {
        for (id, result) in pipeline.step_streaming()?.completed {
            finished |= id == short && result.is_ok();
        }
    }
    assert!(finished);
    pipeline.clear_prefix_cache();
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

fn head_fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_5_moe_mtp")
}

/// One draft per position over `ids`, as serving accumulates the head's keys and values.
fn head_logits(model: &Qwen35, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
    use std::sync::Arc;
    use tur_engine::hybrid::{
        Boundaries, DraftForward, DraftSequence, HybridModel, ModelState, PackedForward,
        PoolConfig, SequenceForward, StatePool,
    };
    let count = ids.len() - 1;
    let description = model.description();
    let pool = StatePool::new(
        Arc::new(description.clone()),
        model.device(),
        PoolConfig {
            pages: description.pages_for(count),
            window_pages: description.pages_for(count),
            recurrent_slots: 2,
        },
    )?;
    let mut reservation = ModelState::new(&pool).reserve(count, Boundaries::Final)?;
    reservation.prepare()?;
    let binding = reservation.binding();
    let positions: Vec<usize> = (0..count).collect();
    let hidden = model
        .forward(&PackedForward {
            embeddings: &[],
            axial: &[],
            tokens: &ids[..count],
            positions: &positions,
            sequences: &[SequenceForward {
                tokens: 0..count,
                state: &binding,
                logits: &[],
                hidden: true,
            }],
            pool: &pool,
        })?
        .pop()
        .ok_or_else(|| invalid("missing step"))??
        .hidden
        .ok_or_else(|| invalid("missing hidden states"))?;
    let state = reservation.commit(count)?;
    drop(reservation);
    let mut rows = Vec::with_capacity(count);
    for row in 0..count {
        let side = state.side(row, 1)?;
        let logits = model
            .draft(&DraftForward {
                tokens: &ids[row + 1..row + 2],
                positions: &positions[row..=row],
                sequences: &[DraftSequence {
                    tokens: 0..1,
                    state: &side,
                    hidden: &hidden.narrow(0, row, 1)?,
                    drafts: 1,
                }],
                pool: &pool,
            })?
            .pop()
            .ok_or_else(|| invalid("missing draft"))??
            .ok_or_else(|| invalid("missing draft logits"))?;
        rows.push(logits.to_vec2::<f32>()?.remove(0));
    }
    Ok(rows)
}

fn head_options(device: &Device, execution: Execution) -> LoadOptions {
    LoadOptions {
        revision: "moe-mtp-fixture-v1".into(),
        weights: WeightFormat::F32.into(),
        device: device.clone(),
        execution,
        kv: KvLayout::new(16, KvFormat::F32),
        recurrent: DType::F32,
        mtp: Some(tur_models::models::DraftHead::default()),
        audio: false,
        vision: false,
        expert_cache: None,
    }
}

/// The expert head against Transformers' own `MoE` decoder layer, on the host and on the
/// kernels; the kernels stay F32 here so that only execution can differ.
#[test]
fn test_mtp_head_over_experts_matches_transformers() -> Result<()> {
    let mut golden =
        candle_core::safetensors::load(head_fixture().join("golden.safetensors"), &Device::Cpu)?;
    let ids: Vec<u32> = golden
        .remove("tokens")
        .ok_or_else(|| invalid("missing tokens"))?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| invalid("invalid golden ID")))
        .collect::<Result<_>>()?;
    let expected = golden
        .remove("logits")
        .ok_or_else(|| invalid("missing logits"))?
        .to_vec2::<f32>()?;
    #[cfg_attr(
        not(feature = "metal"),
        allow(unused_mut, reason = "only Metal adds a second device")
    )]
    let mut devices = vec![(Device::Cpu, Execution::Reference)];
    #[cfg(feature = "metal")]
    devices.push((Device::new_metal(0)?, Execution::Fused));
    for (device, execution) in devices {
        let model = Qwen35::load(&head_fixture(), &head_options(&device, execution))?;
        let error = max_relative(&head_logits(&model, &ids)?, &expected);
        assert!(error < 2e-4, "{execution:?}: {error}");
    }
    Ok(())
}

/// A prepacked artifact carries the head's bank, and a cache serves it as one more layer:
/// bit-identical drafts with and without eviction, and an exact memory plan.
#[cfg(feature = "metal")]
#[test]
fn test_mtp_head_reads_its_experts_through_the_cache() -> Result<()> {
    use tur_models::models::{
        Conversion, Manifest, Packer, WeightPlan, Weights,
        qwen3_5::{artifact_config, write_artifact},
    };
    let device = Device::new_metal(0)?;
    let output = std::env::temp_dir().join(format!("tur-moe-mtp-{}", uuid::Uuid::new_v4()));
    write_artifact(
        &Conversion {
            source: &head_fixture(),
            identity: "moe-mtp-fixture-v1",
            output: &output,
            calibration: "",
        },
        &WeightPlan::uniform(&artifact_config(&head_fixture())?, WeightFormat::Q8_0)?,
        &Packer::RTN,
    )?;
    let result = (|| -> Result<()> {
        let manifest = std::sync::Arc::new(Manifest::read(&output)?);
        let options = |expert_cache| LoadOptions {
            weights: Weights::Prepacked(std::sync::Arc::clone(&manifest)),
            expert_cache,
            ..head_options(&device, Execution::Fused)
        };
        let ids: Vec<u32> = (0..18).map(|i| 2 + (i * 37 + 11) % 88).collect();
        let expected = head_logits(&Qwen35::load(&output, &options(None))?, &ids)?;
        // Q8_0: gate|up 128 rows and down 64 rows, two 34-byte blocks per row.
        let expert = (128 + 64) * 2 * 34;
        for slots in [4, 9] {
            let opts = options(Some(slots * expert));
            let model = Qwen35::load(&output, &opts)?;
            let actual = head_logits(&model, &ids)?;
            for (a, e) in actual.iter().flatten().zip(expected.iter().flatten()) {
                assert_eq!(a.to_bits(), e.to_bits(), "{slots} slots");
            }
            let stats = model
                .expert_stats()
                .ok_or_else(|| invalid("missing expert stats"))?;
            assert!(stats.evictions > 0, "{slots} slots");
            let (_, planned, _) = Qwen35::memory_requirements(model.config(), &opts, 32, &[32])?;
            assert_eq!(planned, model.weight_bytes());
        }
        Ok(())
    })();
    drop(std::fs::remove_dir_all(&output));
    result
}

#[test]
fn test_mtp_expert_tensors_match_the_pinned_checkpoint() -> Result<()> {
    let config = Config::from_json(include_str!("../fixtures/checkpoints/qwen3.6-config.json"))?;
    assert!(config.supports_mtp());
    let expected = config.with_mtp()?.weight_shapes()?;
    let headers: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/checkpoints/qwen3.6-text-tensors.json"
    ))?;
    let mut checked = 0;
    for (name, shape) in expected.iter().filter(|(n, _)| n.starts_with("mtp.")) {
        assert_eq!(
            serde_json::to_value(shape)?,
            headers["tensors"][name]["shape"],
            "{name}"
        );
        checked += 1;
    }
    assert!(checked > 15);
    Ok(())
}
