//! Inkling 2 conformance: the published checkpoint layout, Transformers F32 goldens for
//! every layer application, the logits and cached decoding, and the expert kernels for its
//! NVFP4 and MXFP8 banks and sigmoid routing.

use crate::model_common::{forward, generate, max_relative, prefill, rows as logit_rows};
use candle_core::{DType, Device, Tensor};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tur_core::{
    Result,
    execution::{KvFormat, KvLayout, invalid},
};
use tur_engine::hybrid::{Boundaries, HybridModel, ModelState, PoolConfig, StatePool};
use tur_models::models::{
    Execution, LoadOptions, WeightFormat,
    inkling2::{Config, Inkling2},
};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2")
}

fn options(device: &Device, format: WeightFormat, execution: Execution) -> LoadOptions {
    LoadOptions {
        revision: "inkling2-fixture-v1".into(),
        weights: format.into(),
        device: device.clone(),
        execution,
        kv: KvLayout::new(16, KvFormat::F32),
        recurrent: DType::F32,
        mtp: None,
        audio: false,
        vision: false,
        video: false,
        expert_cache: None,
    }
}

fn load(device: &Device, format: WeightFormat, execution: Execution) -> Result<Inkling2> {
    Inkling2::load(&fixture(), &options(device, format, execution))
}

struct Golden {
    ids: Vec<u32>,
    layers: Vec<Tensor>,
    logits: Tensor,
    /// Logits from Transformers' cache: the last prefilled token's, then one per decoded
    /// token.
    decoded: Tensor,
    prefill: usize,
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
    // Six layers, the two looped ones three times each.
    let layers = (0..10)
        .map(|i| take(&format!("layer.{i}")))
        .collect::<Result<_>>()?;
    Ok(Golden {
        ids,
        layers,
        logits: take("logits")?,
        decoded: take("decode.logits")?,
        prefill: 13,
    })
}

fn compare(model: &Inkling2, bound: f32) -> Result<()> {
    let golden = golden()?;
    let rows: Vec<usize> = (0..golden.ids.len()).collect();
    let (layers, logits) = model.trace(&golden.ids, &rows)?;
    assert_eq!(layers.len(), golden.layers.len());
    for (index, (actual, expected)) in layers.iter().zip(&golden.layers).enumerate() {
        let actual = actual.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let error = max_relative(&actual, &expected.to_vec2::<f32>()?);
        assert!(error < bound, "layer application {index}: {error}");
    }
    let logits = logits.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let error = max_relative(&logits, &golden.logits.to_vec2::<f32>()?);
    assert!(error < bound, "logits: {error}");
    Ok(())
}

fn pool(model: &Inkling2) -> Result<StatePool> {
    StatePool::new(
        Arc::new(model.description()),
        model.device(),
        PoolConfig {
            pages: 8,
            window_pages: 8,
            recurrent_slots: 4,
            index_blocks: 0,
        },
    )
}

/// Prefill as Transformers did, then decode one token at a time through the pools: the
/// convolution histories, sliding windows and paged keys carry the state.
fn compare_decoding(model: &Inkling2, bound: f32) -> Result<()> {
    let golden = golden()?;
    let pool = pool(model)?;
    let (head, tail) = golden.ids.split_at(golden.prefill);
    let (prefilled, state) = prefill(model, ModelState::new(&pool), head, head.len())?;
    let (decoded, _) = prefill(model, state, &tail[..tail.len() - 1], 1)?;
    let mut actual = vec![prefilled[golden.prefill - 1].clone()];
    actual.extend(decoded);
    let error = max_relative(&actual, &golden.decoded.to_vec2::<f32>()?);
    assert!(error < bound, "decode logits: {error}");
    Ok(())
}

/// The engine's greedy decoding through its scheduler and pools picks, at each step, the
/// argmax of a fresh pass over everything before it, or a token within `tolerance` of it
/// (relative to the row's largest logit) where packed decode and prefill kernels round
/// differently.
fn check_engine_greedy(model: Inkling2, device: Device, tolerance: f32) -> Result<()> {
    let model = Arc::new(model);
    let golden = golden()?;
    let prompt = &golden.ids[..10];
    let pool = PoolConfig {
        pages: 8,
        window_pages: 8,
        recurrent_slots: 4,
        index_blocks: 0,
    };
    let generated = generate(Arc::clone(&model), device, pool, prompt, &[], 9)?;
    assert_eq!(generated.len(), 9);
    let mut all = prompt.to_vec();
    all.extend(&generated);
    let rows: Vec<usize> = (prompt.len() - 1..all.len() - 1).collect();
    let (_, logits) = model.trace(&all[..all.len() - 1], &rows)?;
    for (step, row) in logits
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?
        .iter()
        .enumerate()
    {
        let best = row.iter().copied().fold(f32::MIN, f32::max);
        let scale = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let chosen = row[generated[step] as usize];
        assert!(
            best - chosen <= tolerance * scale,
            "step {step}: {chosen} below {best}"
        );
    }
    Ok(())
}

#[test]
fn test_inkling2_engine_greedy_decoding_matches_full_passes() -> Result<()> {
    check_engine_greedy(
        load(&Device::Cpu, WeightFormat::F32, Execution::Reference)?,
        Device::Cpu,
        0.0,
    )
}

#[cfg(feature = "metal")]
#[test]
fn test_inkling2_fused_engine_greedy_decoding_matches_full_passes() -> Result<()> {
    let device = Device::new_metal(0)?;
    check_engine_greedy(
        load(&device, WeightFormat::Q8_0, Execution::Fused)?,
        device,
        1e-3,
    )
}

/// Sequences packed into one pass, one decoding after a prefix and one starting, give the
/// logits each gives alone: packing shares no state between them.
fn check_packed_sequences(model: &Inkling2, bound: f32) -> Result<()> {
    let golden = golden()?;
    let pool = pool(model)?;
    let (_, decoding) = prefill(model, ModelState::new(&pool), &golden.ids[..11], 11)?;
    let (alone, _) = prefill(model, decoding.clone(), &golden.ids[11..14], 3)?;
    let fresh = &golden.ids[4..20];
    let (fresh_alone, _) = prefill(model, ModelState::new(&pool), fresh, fresh.len())?;
    let (mut steps, reservations) = forward(
        model,
        &[
            (&decoding, &golden.ids[11..14], Boundaries::Final),
            (&ModelState::new(&pool), fresh, Boundaries::Final),
        ],
    )?;
    drop(reservations);
    let second = logit_rows(&steps.pop().ok_or_else(|| invalid("missing step"))??.logits)?;
    let first = logit_rows(&steps.pop().ok_or_else(|| invalid("missing step"))??.logits)?;
    assert!(max_relative(&first, &alone) < bound, "decoding sequence");
    assert!(
        max_relative(&second, &fresh_alone) < bound,
        "fresh sequence"
    );
    Ok(())
}

#[test]
fn test_inkling2_packed_sequences_match_separate_passes() -> Result<()> {
    check_packed_sequences(
        &load(&Device::Cpu, WeightFormat::F32, Execution::Reference)?,
        1e-5,
    )
}

#[cfg(feature = "metal")]
#[test]
fn test_inkling2_fused_packed_sequences_match_separate_passes() -> Result<()> {
    let device = Device::new_metal(0)?;
    // F32 projections: packed Q8_0 projections switch kernels with the pass's row count.
    check_packed_sequences(&load(&device, WeightFormat::F32, Execution::Fused)?, 1e-4)
}

#[test]
fn test_inkling2_reference_matches_independent_transformers_layers_and_logits() -> Result<()> {
    compare(
        &load(&Device::Cpu, WeightFormat::F32, Execution::Reference)?,
        1e-4,
    )
}

#[test]
fn test_inkling2_reference_decodes_like_transformers_cache() -> Result<()> {
    compare_decoding(
        &load(&Device::Cpu, WeightFormat::F32, Execution::Reference)?,
        1e-4,
    )
}

#[test]
fn test_inkling2_config_and_tensor_table_match_the_published_checkpoint() -> Result<()> {
    let dir = fixture();
    let read = |name: &str| std::fs::read_to_string(dir.join(name));
    let config = Config::from_json(&read("mini-config.json")?)?
        .with_quantization(&read("mini-hf_quant_config.json")?)?;
    assert_eq!(config.vocab_size(), 200_058);
    assert_eq!(config.max_position(), 1 << 20);
    let header: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&read("mini-header.json")?)?;
    let text: BTreeMap<&str, (&str, Vec<usize>)> = header
        .iter()
        .filter(|(name, _)| !name.starts_with("model.visual."))
        .map(|(name, entry)| {
            let shape = entry["shape"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_u64)
                .map(|d| d as usize)
                .collect();
            (
                name.as_str(),
                (entry["dtype"].as_str().unwrap_or(""), shape),
            )
        })
        .collect();
    let table = config.weight_shapes()?;
    assert_eq!(table.len(), text.len(), "text tensor count");
    for (name, (shape, stored)) in &table {
        let (dtype, expected) = text
            .get(name.as_str())
            .ok_or_else(|| invalid(format!("{name} is not in the checkpoint")))?;
        assert_eq!(shape, expected, "{name}");
        let kind = match *dtype {
            "BF16" | "F32" => "float",
            "U8" => "codes",
            "F8_E4M3" => "fp8",
            other => other,
        };
        assert_eq!(stored.name(), kind, "{name}");
    }
    // Six layers before the loop, twelve looped layers three times, six after: each
    // application has paged K/V and a slot of convolution histories.
    let (description, _, _) = Inkling2::memory_requirements(
        &config,
        &options(&Device::Cpu, WeightFormat::F32, Execution::Reference),
        1,
        &[1],
    )?;
    assert_eq!(description.layers.len(), 2 * 48);
    assert_eq!(description.window(), Some(1024));
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_inkling2_fused_f32_matches_transformers() -> Result<()> {
    let device = Device::new_metal(0)?;
    compare(&load(&device, WeightFormat::F32, Execution::Fused)?, 1e-4)
}

#[cfg(feature = "metal")]
#[test]
fn test_inkling2_fused_decodes_like_transformers_cache() -> Result<()> {
    let device = Device::new_metal(0)?;
    compare_decoding(&load(&device, WeightFormat::F32, Execution::Fused)?, 1e-4)
}

/// Packed NVFP4 and MXFP8 experts on the kernels, with each NVFP4 expert's factor applied
/// in the activation, against the reference over the same projections with decoded experts;
/// the plan made before loading holds exactly the loaded bytes.
#[cfg(feature = "metal")]
#[test]
fn test_inkling2_packed_experts_match_reference_and_plan() -> Result<()> {
    let device = Device::new_metal(0)?;
    let golden = golden()?;
    let rows: Vec<usize> = (0..golden.ids.len()).collect();
    let packed = load(&device, WeightFormat::Q8_0, Execution::Fused)?;
    let reference = load(&device, WeightFormat::Q8_0, Execution::Reference)?;
    let (_, actual) = packed.trace(&golden.ids, &rows)?;
    let (_, expected) = reference.trace(&golden.ids, &rows)?;
    let error = max_relative(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
    assert!(error < 2e-4, "{error}");
    let options = options(&device, WeightFormat::Q8_0, Execution::Fused);
    let (_, planned, _) = Inkling2::memory_requirements(packed.config(), &options, 32, &[32])?;
    assert_eq!(planned, packed.weight_bytes());
    Ok(())
}

/// The block drafter after a committed prefix against an F32 transcription of vLLM's Inkling
/// 2 `DSpark` drafter (`scripts/inkling2_reference.py --tiny-mtp`): each committed position's
/// context keys and values from its four target features, the anchor-and-masks block over
/// them, and the recurrent head's seven chained, softcapped logit rows. F32 projections, so
/// only execution can differ.
#[cfg(feature = "metal")]
#[test]
fn test_inkling2_drafter_matches_reference_block() -> Result<()> {
    use crate::model_common::forward_rows;
    use tur_models::{
        hybrid::{DraftForward, DraftSequence},
        models::DraftHead,
    };
    let device = Device::new_metal(0)?;
    let mut options = options(&device, WeightFormat::F32, Execution::Fused);
    options.mtp = Some(DraftHead::default());
    let model = Inkling2::load(&fixture(), &options)?;
    let mut data =
        candle_core::safetensors::load(fixture().join("draft-golden.safetensors"), &Device::Cpu)?;
    let expected = data
        .remove("logits")
        .ok_or_else(|| invalid("missing draft logits"))?;
    let committed = 16;
    let ids = golden()?.ids;
    let pool = pool(&model)?;
    let (mut steps, reservations) = forward_rows(
        &model,
        &[(
            &ModelState::new(&pool),
            &ids[..committed],
            Boundaries::Final,
            &[committed - 1],
            true,
        )],
    )?;
    let features = steps
        .pop()
        .ok_or_else(|| invalid("missing step"))??
        .hidden
        .ok_or_else(|| invalid("missing features"))?;
    let state = reservations
        .first()
        .ok_or_else(|| invalid("missing reservation"))?
        .commit(committed)?;
    drop(reservations);
    let side = state.side(0, committed, &[])?;
    let positions: Vec<usize> = (0..committed).collect();
    let sequences = [DraftSequence {
        tokens: 0..committed,
        state: &side,
        hidden: &features,
        drafts: 7,
    }];
    let actual = model
        .draft(&DraftForward {
            tokens: &ids[1..=committed],
            positions: &positions,
            sequences: &sequences,
            pool: &pool,
        })?
        .pop()
        .ok_or_else(|| invalid("missing draft"))??
        .ok_or_else(|| invalid("no drafted logits"))?;
    let error = max_relative(&logit_rows(&actual)?, &expected.to_vec2::<f32>()?);
    assert!(error < 1e-4, "drafted logits: {error}");
    // Drafts past the depth the head offers are refused.
    let deeper = [DraftSequence {
        tokens: 0..committed,
        state: &side,
        hidden: &features,
        drafts: 8,
    }];
    let refused = model.draft(&DraftForward {
        tokens: &ids[1..=committed],
        positions: &positions,
        sequences: &deeper,
        pool: &pool,
    })?;
    assert!(refused.first().is_some_and(Result::is_err));
    Ok(())
}

/// Drafts of the block drafter verified by the target, through the serving pipeline: whatever
/// the (random) drafter proposes, greedy and sampled outputs equal ordinary decoding, the
/// drafter's chained drafts are verified, and every page and slot is returned.
#[cfg(feature = "metal")]
#[test]
fn test_inkling2_drafter_preserves_target_sampling() -> Result<()> {
    let fixture = crate::common::Checkpoint::inkling2()?;
    let mut factory = fixture.factory();
    factory.revision = "inkling2-fixture-v1".into();
    factory.device = Device::new_metal(0)?;
    factory.kv = KvLayout::new(16, KvFormat::F32);
    crate::speculation::assert_mtp_preserves_outputs_with(
        &factory,
        crate::speculation::mtp_config(4, 1),
    )
}

#[cfg(feature = "metal")]
mod kernels {
    use crate::kernels::{
        AttentionCase, paged_pool, read, relative_error, setup, stored, upload, values,
    };
    use candle_core::{Device, Tensor};
    use tur_core::{Result, execution::KvFormat};
    use tur_metal::{
        kernels::{Clipped, LogScale, Relative, Tuning},
        packed::PackedLinear,
        pool::AttentionPool,
        quantize::Microscale,
    };

    /// Keys split across threadgroups, their partials combined with each token's sink logit
    /// and gated afterwards, give what one pass over every key gives, for full and windowed
    /// layers over each pool format. Splits of 2 and 8 keys cut through pages, windows and
    /// the relative bias's extent; unused pool slots hold NaN, so a stray read shows.
    #[test]
    fn test_relative_attention_splits_match_one_pass() -> Result<()> {
        let (device, kernels, _gpu) = setup()?;
        let case = AttentionCase {
            heads: 6,
            kv_heads: 2,
            dim: 128,
            page: 4,
            sequences: vec![(37, 3), (0, 9)],
        };
        let (heads, dim, rel_dim, extent) = (case.heads, case.dim, 16, 7);
        let (key, value, slots, table, first, visible) = paged_pool(&case, 41);
        let tokens = visible.len();
        // Queries, then relative states, as one projection row holds them; scales and
        // thresholds that clip some heads, and log scaling past position 8.
        let width = heads * (dim + rel_dim);
        let rows = upload(&device, &values(tokens * width, 1, 0.3), &[tokens, width])?;
        let q_scale = upload(&device, &values(heads * dim, 5, 2.0), &[heads * dim])?;
        let rel_scale = upload(
            &device,
            &values(heads * rel_dim, 6, 2.0),
            &[heads * rel_dim],
        )?;
        let positions: Vec<u32> = visible.iter().map(|&keys| keys - 1).collect();
        let positions = Tensor::new(positions.as_slice(), &device)?;
        let proj = upload(
            &device,
            &values(rel_dim * extent, 3, 1.0),
            &[rel_dim, extent],
        )?;
        // Each row holds the sink logits, then the gates.
        let width = heads + heads * dim;
        let side = upload(&device, &values(tokens * width, 4, 2.0), &[tokens, width])?;
        let paged = kernels.paged_kv(case.page, slots, &table, &first, &visible)?;
        for format in [KvFormat::F32, KvFormat::F16, KvFormat::Q8] {
            let (key, _) = stored(&device, &key, slots, case.kv_heads, dim, format)?;
            let (value, _) = stored(&device, &value, slots, case.kv_heads, dim, format)?;
            let pool = AttentionPool {
                key,
                value,
                format,
                page_tokens: case.page,
                kv_heads: case.kv_heads,
                head_dim: dim,
            };
            for window in [None, Some(5), Some(24)] {
                let run = |minimum: usize| -> Result<Vec<f32>> {
                    let tuned = kernels.clone().with_tuning(Tuning {
                        min_split_keys: minimum,
                        ..kernels.tuning()
                    })?;
                    let relative = Relative {
                        rel: Clipped {
                            rows: &rows,
                            col: heads * dim,
                            scale: &rel_scale,
                            threshold: 1.5,
                        },
                        log_scale: Some(LogScale {
                            positions: &positions,
                            alpha: 0.1,
                            floor: 8.0,
                        }),
                        proj: &proj,
                        sinks: (&side, 0),
                        gate: (&side, heads),
                        window,
                        scale: 0.09,
                    };
                    let q = Clipped {
                        rows: &rows,
                        col: 0,
                        scale: &q_scale,
                        threshold: 2.0,
                    };
                    read(&tuned.relative_attention((q, heads), &pool, &paged, relative)?)
                };
                let whole: Vec<f64> = run(1024)?.into_iter().map(f64::from).collect();
                for minimum in [2, 8] {
                    let error = relative_error(&run(minimum)?, &whole);
                    assert!(
                        error < 1e-5,
                        "{format} window {window:?} split {minimum}: {error}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Bytes of a microscaled bank: random values (never an E4M3 NaN) and moderate scales.
    fn bank(format: Microscale, rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let values = (0..format.value_bytes(rows * cols))
            .map(|_| {
                let byte = (next() >> 24) as u8;
                if format == Microscale::Mxfp8 && byte & 0x7F == 0x7F {
                    byte ^ 1
                } else {
                    byte
                }
            })
            .collect();
        let scales = (0..rows * cols / format.block_values())
            .map(|_| match format {
                // E4M3 exponents 6..=9: scales near 0.25..4.
                Microscale::Nvfp4 => (((next() % 4 + 6) << 3) | (next() % 8)) as u8,
                _ => 118 + (next() % 5) as u8,
            })
            .collect();
        (values, scales)
    }

    /// Every tier of the grouped expert kernels (GEMV for up to three rows, matrix units up
    /// to sixteen, F16-staged wide tiles above) reads NVFP4 and MXFP8 banks as the host
    /// decodes them.
    #[test]
    fn test_microscaled_expert_tiers_match_host_decoded_weights() -> Result<()> {
        let (device, kernels, _gpu) = setup()?;
        let (experts, out, width) = (4, 40, 96);
        for format in [Microscale::Nvfp4, Microscale::Mxfp8] {
            let (values, scales) = bank(format, experts * out, width, 7);
            let layer = PackedLinear::microscaled(
                format,
                &values,
                &scales,
                (experts * out, width),
                &device,
            )?;
            let dense = format.decode(&values, &scales)?;
            for (tokens, top_k) in [(2, 1), (9, 2), (60, 2)] {
                let x: Vec<f32> = (0..tokens * width)
                    .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
                    .collect();
                let x = Tensor::from_vec(x.clone(), (tokens, width), &device)?;
                // Every token prefers expert 1, so its group spans every tier as tokens grow.
                let scores: Vec<f32> = (0..tokens)
                    .flat_map(|t| {
                        (0..experts).map(move |e| if e == 1 { 9.0 } else { ((t + e) % 3) as f32 })
                    })
                    .collect();
                let routes = kernels.route(
                    &Tensor::from_vec(scores, (tokens, experts), &device)?,
                    top_k,
                )?;
                let grouped = read(&kernels.expert_linear_routed(&layer, &x, &routes)?)?;
                let host_x = x.to_vec2::<f32>()?;
                let mut expected = Vec::new();
                for (token, expert) in routes.assignment_rows()? {
                    for o in 0..out {
                        let row = &dense[(expert * out + o) * width..][..width];
                        expected.push(
                            row.iter()
                                .zip(&host_x[token])
                                .map(|(w, x)| f64::from(*w) * f64::from(*x))
                                .sum::<f64>(),
                        );
                    }
                }
                let error = relative_error(&grouped, &expected);
                assert!(error < 1e-5, "{format:?}, {tokens} tokens: {error}");
            }
        }
        Ok(())
    }

    /// Selection by sigmoid score plus bias, weights from the sigmoid scores alone,
    /// normalized over the selection and scaled.
    #[test]
    fn test_sigmoid_routing_selects_with_bias_and_weights_without_it() -> Result<()> {
        let (device, kernels, _gpu) = setup()?;
        let (tokens, experts, top_k, scale) = (5, 12, 3, 2.5f32);
        let logits: Vec<f32> = (0..tokens * experts)
            .map(|i| ((i * 29 % 23) as f32 - 11.0) / 4.0)
            .collect();
        let bias: Vec<f32> = (0..experts)
            .map(|e| if e % 4 == 0 { 0.8 } else { -0.1 })
            .collect();
        let routes = kernels.route_sigmoid(
            &Tensor::from_vec(logits.clone(), (tokens, experts), &device)?,
            top_k,
            &Tensor::from_vec(bias.clone(), experts, &device)?,
            scale,
        )?;
        // One-hot rows make the scatter return each token's weight per expert.
        let rows = routes.assignment_rows()?;
        let mut onehot = vec![0f32; rows.len() * experts];
        for (row, (_, expert)) in rows.iter().enumerate() {
            onehot[row * experts + expert] = 1.0;
        }
        let x = Tensor::from_vec(onehot, (rows.len(), experts), &device)?;
        let weights = kernels
            .expert_scatter(&x, &routes)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        for (t, row) in weights.iter().enumerate() {
            let sigmoid = |e: usize| 1.0 / (1.0 + (-f64::from(logits[t * experts + e])).exp());
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| {
                (sigmoid(b) + f64::from(bias[b]))
                    .total_cmp(&(sigmoid(a) + f64::from(bias[a])))
                    .then(a.cmp(&b))
            });
            let chosen = &order[..top_k];
            let total: f64 = chosen.iter().map(|&e| sigmoid(e)).sum();
            for (e, weight) in row.iter().enumerate() {
                let expected = if chosen.contains(&e) {
                    sigmoid(e) / total * f64::from(scale)
                } else {
                    0.0
                };
                assert!(
                    (f64::from(*weight) - expected).abs() < 1e-5,
                    "token {t}, expert {e}"
                );
            }
        }
        Ok(())
    }
}

/// The published template renders as Transformers' `apply_chat_template` does: the effort
/// message before the first non-system turn, replayed reasoning, and no generation prompt,
/// since the model opens its own message.
#[test]
fn test_inkling2_template_renders_like_transformers() -> Result<()> {
    use tur_models::backend::{
        chat_template::{ChatMessage, ChatTemplate, RenderOptions, Role, ToolCall},
        tools::ToolDefinition,
    };
    let dir = fixture();
    let template = ChatTemplate::from_template(std::fs::read_to_string(
        dir.join("mini-chat_template.jinja"),
    )?)?;
    let cases: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("mini-templates.json"))?)?;
    for (name, case) in &cases {
        let messages = case["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|turn| {
                let role = match turn["role"].as_str() {
                    Some("system") => Role::System,
                    Some("assistant") => Role::Assistant,
                    Some("tool") => Role::Tool,
                    _ => Role::User,
                };
                let mut message =
                    ChatMessage::new(role, turn["content"].as_str().unwrap_or_default());
                message.reasoning_content = turn["reasoning_content"].as_str().map(str::to_owned);
                message.tool_calls = turn["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|call| ToolCall {
                        name: call["function"]["name"]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned(),
                        arguments: call["function"]["arguments"]
                            .as_object()
                            .cloned()
                            .unwrap_or_default(),
                    })
                    .collect();
                message
            })
            .collect::<Vec<_>>();
        let options = RenderOptions {
            add_generation_prompt: true,
            reasoning_effort: case["kwargs"]["reasoning_effort"]
                .as_str()
                .map(str::to_owned),
            ..RenderOptions::default()
        };
        let tools: Vec<ToolDefinition> = case["kwargs"]["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| {
                let f = &tool["function"];
                ToolDefinition::new(
                    f["name"].as_str().unwrap_or_default(),
                    f["description"].as_str().unwrap_or_default(),
                    f["parameters"].clone(),
                )
            })
            .collect();
        let rendered = template.render(&messages, &tools, &options)?;
        assert_eq!(
            rendered,
            case["text"].as_str().unwrap_or_default(),
            "{name}"
        );
    }
    Ok(())
}

/// Real weights against Transformers' own layers, one decoder layer at a time
/// (`scripts/inkling2_reference.py --model`), with F32 projections and the experts packed as
/// published, so only execution can differ. About 22 GB resident. Needs `TUR_INKLING2_MINI`
/// and `TUR_INKLING2_GOLDEN`.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the 20 GB Inkling-2-Mini checkpoint and its golden"]
fn test_inkling2_mini_matches_transformers_layers_and_logits() -> Result<()> {
    let read = |name: &str| std::env::var(name).map_err(|_| invalid(format!("{name} is not set")));
    let dir = PathBuf::from(read("TUR_INKLING2_MINI")?);
    let mut data = candle_core::safetensors::load(read("TUR_INKLING2_GOLDEN")?, &Device::Cpu)?;
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
    let mut options = options(&device, WeightFormat::F32, Execution::Fused);
    options.revision = "inkling-2-mini-conformance".into();
    let model = Inkling2::load(&dir, &options)?;
    let rows: Vec<usize> = (0..ids.len()).collect();
    let (layers, logits) = model.trace(&ids, &rows)?;
    assert_eq!(layers.len(), 48);
    for (index, actual) in layers.iter().enumerate() {
        let expected = take(&format!("layer.{index}"))?;
        let actual = actual.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let error = max_relative(&actual, &expected.to_vec2::<f32>()?);
        assert!(error < 1e-3, "layer application {index}: {error}");
    }
    let expected = take("logits")?;
    let logits = logits.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let error = max_relative(&logits, &expected.to_vec2::<f32>()?);
    assert!(error < 1e-3, "logits: {error}");
    Ok(())
}
