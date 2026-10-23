#![allow(
    clippy::print_stderr,
    reason = "checkpoint comparisons report numerical evidence to the test runner"
)]
#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![cfg(feature = "metal")]
#![forbid(unsafe_code)]
//! Fused Metal execution against the F32 reference on identical weights: per-layer and
//! logit agreement, chunked/incremental prefill, batch isolation, speculative boundaries,
//! packed weight and K/V formats, page sizes, exact prefix forks and restores, memory
//! pressure, workspace and allocation bounds. Needs host GPU access and fails, rather
//! than skips, without it. Real-checkpoint tests are `#[ignore]` and need
//! `TUR_QWEN35_0_8B`; the reference itself is validated against Transformers in P3.

mod common;

use candle_core::{DType, Device, Tensor};
use common::{Checkpoint, forward, max_relative, prefill, rows};
use std::{path::Path, sync::Arc};
use tur_core::{
    Result, TurError,
    execution::{KvFormat, KvLayout, PackedExecution, SequenceDescriptor, StateHandle, invalid},
};
use tur_engine::hybrid::{
    Boundaries, HybridEngine, HybridModel, ModelState, PackedForward, PoolConfig, SamplingOptions,
    SequenceForward, StatePool,
};
use tur_metal::state::LayerSnapshot;
use tur_models::models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat};

/// Kernel-supported head sizes, two attention layers and projection widths that are
/// multiples of 256 so Q4K applies everywhere. No modality placeholder: engine tests feed
/// greedy outputs back as inputs.
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

/// Exact storage for fused-versus-reference agreement.
const F32_KV: KvLayout = KvLayout {
    page_tokens: 16,
    format: KvFormat::F32,
    version: tur_core::execution::KV_LAYOUT_VERSION,
};

/// Device-wide allocation counters are shared by every test in the process.
static GPU: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn metal() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

fn small() -> Result<Checkpoint> {
    common::checkpoint(SMALL, 0.08, 0.1)
}

fn load_kv(
    dir: &Path,
    format: WeightFormat,
    device: &Device,
    execution: Execution,
    kv: KvLayout,
) -> Result<Qwen35> {
    load_state(dir, format, device, execution, kv, DType::F32)
}

fn load_state(
    dir: &Path,
    format: WeightFormat,
    device: &Device,
    execution: Execution,
    kv: KvLayout,
    recurrent: DType,
) -> Result<Qwen35> {
    Qwen35::load(
        dir,
        &LoadOptions {
            revision: "small-fixture".into(),
            format,
            device: device.clone(),
            execution,
            kv,
            recurrent,
        },
    )
}

fn load(dir: &Path, format: WeightFormat, device: &Device, execution: Execution) -> Result<Qwen35> {
    load_kv(dir, format, device, execution, F32_KV)
}

/// Room for several full contexts of the small fixture.
fn pool(model: &Qwen35) -> Result<StatePool> {
    common::pool(model, 256, 16)
}

fn prompt(len: usize, salt: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((i * 37 + 11 + salt * 13) % 148 + 2) as u32)
        .collect()
}

fn flat(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(vec![
        tensor
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?,
    ])
}

/// Every layer's committed state as flat rows, for comparing snapshots.
fn snapshot(state: &ModelState) -> Result<Vec<Vec<Vec<f32>>>> {
    state
        .read_layers()?
        .iter()
        .map(|layer| {
            let (a, b) = match layer {
                LayerSnapshot::Attention { key, value } => (key, value),
                LayerSnapshot::Recurrent {
                    convolution,
                    recurrent,
                } => (convolution, recurrent),
            };
            Ok(vec![flat(a)?.concat(), flat(b)?.concat()])
        })
        .collect()
}

fn snapshot_error(a: &ModelState, b: &ModelState) -> Result<f32> {
    let (a, b) = (snapshot(a)?, snapshot(b)?);
    Ok(a.iter()
        .zip(&b)
        .map(|(x, y)| max_relative(x, y))
        .fold(0.0, f32::max))
}

#[test]
fn test_fused_layers_and_logits_match_reference_across_page_and_chunk_boundaries() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let fused = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    assert_eq!(fused.execution(), Execution::Fused);
    // 513 tokens reach the default tiled-attention dispatch and leave a partial query
    // tile, while crossing 16-token pages and the reference's 64-token delta chunks.
    let tokens = prompt(513, 0);
    let all: Vec<usize> = (0..tokens.len()).collect();
    let (fused_layers, fused_logits) = fused.trace(&tokens, &all)?;
    let reference = fused.with_execution(Execution::Reference)?;
    assert_eq!(reference.execution(), Execution::Reference);
    let (layers, logits) = reference.trace(&tokens, &all)?;
    for (i, (a, b)) in fused_layers.iter().zip(&layers).enumerate() {
        let error = max_relative(&rows(a)?, &rows(b)?);
        assert!(error < 1e-4, "layer {i}: {error}");
    }
    let error = max_relative(&rows(&fused_logits)?, &rows(&logits)?);
    assert!(error < 1e-4, "logits: {error}");
    Ok(())
}

#[test]
fn test_fused_chunked_and_incremental_prefill_match_reference_whole_prefill() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let reference = load(
        &checkpoint.0,
        WeightFormat::F32,
        &device,
        Execution::Reference,
    )?;
    let fused = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    let tokens = prompt(150, 1);
    let (whole, _) = prefill(
        &reference,
        ModelState::new(&pool(&reference)?),
        &tokens,
        150,
    )?;
    let pool = pool(&fused)?;
    for chunk in [150, 65, 37, 16, 1] {
        let (logits, state) = prefill(&fused, ModelState::new(&pool), &tokens, chunk)?;
        let error = max_relative(&logits, &whole);
        assert!(error < 1e-4, "chunk {chunk}: {error}");
        assert_eq!(state.position(), 150);
    }
    Ok(())
}

#[test]
fn test_fused_batch_matches_each_sequence_alone() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let model = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    let pool = pool(&model)?;
    // Different committed lengths (including a partial page and an empty state).
    let (_, a) = prefill(&model, ModelState::new(&pool), &prompt(37, 2), 37)?;
    let (_, b) = prefill(&model, ModelState::new(&pool), &prompt(16, 3), 16)?;
    let c = ModelState::new(&pool);
    let new = [prompt(1, 4), prompt(5, 5), prompt(20, 6)];
    let batch: Vec<(&ModelState, &[u32], Boundaries)> = [&a, &b, &c]
        .into_iter()
        .zip(&new)
        .map(|(state, tokens)| (state, tokens.as_slice(), Boundaries::Final))
        .collect();
    let (together, packed) = forward(&model, &batch)?;
    for ((step, reservation), single) in together.into_iter().zip(&packed).zip(&batch) {
        let (mut alone, reservations) = forward(&model, std::slice::from_ref(single))?;
        let alone = alone.pop().ok_or_else(|| invalid("missing step"))??;
        let error = max_relative(&rows(&step?.logits)?, &rows(&alone.logits)?);
        assert!(error < 1e-5, "logits {error}");
        // K/V pages and recurrent slots written in the batch equal those written alone.
        let n = single.1.len();
        let solo = reservations
            .first()
            .ok_or_else(|| invalid("missing reservation"))?
            .commit(n)?;
        let error = snapshot_error(&reservation.commit(n)?, &solo)?;
        assert!(error < 1e-5, "state {error}");
    }
    Ok(())
}

#[test]
fn test_fused_every_token_boundaries_match_reference() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let fused = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    let reference = load(
        &checkpoint.0,
        WeightFormat::F32,
        &device,
        Execution::Reference,
    )?;
    let pool = pool(&reference)?;
    let (_, state) = prefill(&reference, ModelState::new(&pool), &prompt(20, 7), 20)?;
    let draft = prompt(5, 8);
    let (mut a, fused_reservations) = forward(&fused, &[(&state, &draft, Boundaries::EveryToken)])?;
    let (mut b, reference_reservations) =
        forward(&reference, &[(&state, &draft, Boundaries::EveryToken)])?;
    let (a, b) = (
        a.pop().ok_or_else(|| invalid("missing step"))??,
        b.pop().ok_or_else(|| invalid("missing step"))??,
    );
    assert!(max_relative(&rows(&a.logits)?, &rows(&b.logits)?) < 1e-4);
    // Every accepted prefix publishes the exact saved boundary.
    for accepted in 1..=draft.len() {
        let x = fused_reservations[0].commit(accepted)?;
        let y = reference_reservations[0].commit(accepted)?;
        assert_eq!(x.position(), 20 + accepted);
        let error = snapshot_error(&x, &y)?;
        assert!(error < 1e-4, "accepted {accepted}: {error}");
    }
    Ok(())
}

#[test]
fn test_fused_packed_formats_track_reference() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    for format in [WeightFormat::Q8_0, WeightFormat::Q4K] {
        let fused = load(&checkpoint.0, format, &device, Execution::Fused)?;
        // Packed storage only: no projection is expanded to F32.
        let f32_bytes = load(
            &checkpoint.0,
            WeightFormat::F32,
            &Device::Cpu,
            Execution::Reference,
        )?
        .projection_bytes();
        assert!(
            fused.projection_bytes() * 3 < f32_bytes,
            "{format:?} expanded"
        );
        let reference = load(&checkpoint.0, format, &device, Execution::Reference)?;
        // Two batches of up to twelve sequences and their one-sequence references each write
        // a slot per sequence before the batch is reclaimed.
        let pool = common::pool(&reference, 256, 64)?;
        let (_, long_state) = prefill(&reference, ModelState::new(&pool), &prompt(30, 9), 30)?;
        let (_, short_state) = prefill(&reference, ModelState::new(&pool), &prompt(7, 10), 7)?;
        // Batches of single-token sequences: three take the small-batch GEMV, six and twelve
        // the Q8_0 matrix-unit kernel (one and two row tiles), and every recurrent layer
        // the decode step kernel. The reference run per sequence uses Candle's one-row GEMV
        // (F32 dequantization); run as a batch it uses Candle's tiled GEMM, which stages
        // weight tiles as F16.
        for count in [3usize, 6, 12] {
            let new: Vec<Vec<u32>> = (0..count).map(|i| prompt(1, 11 + i)).collect();
            let batch: Vec<(&ModelState, &[u32], Boundaries)> = new
                .iter()
                .enumerate()
                .map(|(i, tokens)| {
                    let state = if i % 2 == 0 {
                        &long_state
                    } else {
                        &short_state
                    };
                    (state, tokens.as_slice(), Boundaries::Final)
                })
                .collect();
            let (tiled, tiled_held) = forward(&reference, &batch)?;
            let (fused_steps, fused_steps_held) = forward(&fused, &batch)?;
            for ((fused, tiled), single) in fused_steps.into_iter().zip(tiled).zip(&batch) {
                let (mut alone, _held) = forward(&reference, std::slice::from_ref(single))?;
                let alone = alone.pop().ok_or_else(|| invalid("missing step"))??;
                let (fused_logits, single_logits, tiled_logits) = (
                    rows(&fused?.logits)?,
                    rows(&alone.logits)?,
                    rows(&tiled?.logits)?,
                );
                // Q4K has no matrix-unit kernel: above its GEMV range it uses F16 tiles.
                let bound = if format == WeightFormat::Q4K && count > 4 {
                    2e-2
                } else {
                    1e-4
                };
                let error = max_relative(&fused_logits, &single_logits);
                assert!(
                    error < bound,
                    "{format:?} decode of {count} vs one-row GEMV {error}"
                );
                let error = max_relative(&tiled_logits, &single_logits);
                assert!(
                    error < 2e-2,
                    "{format:?} F16-tiled GEMM vs one-row GEMV {error}"
                );
            }
            // Retire this batch's slots before the next one leases its own.
            drop((fused_steps_held, tiled_held));
            device.synchronize()?;
            pool.reclaim();
        }
        let tokens = prompt(100, 14);
        let error = max_relative(
            &prefill(&fused, ModelState::new(&pool), &tokens, 100)?.0,
            &prefill(&reference, ModelState::new(&pool), &tokens, 100)?.0,
        );
        assert!(error < 1e-4, "{format:?} prefill {error}");
    }
    Ok(())
}

#[test]
fn test_kv_formats_and_page_sizes_on_the_fused_path() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let tokens = prompt(150, 20);
    let run = |execution: Execution, kv: KvLayout| -> Result<Vec<Vec<f32>>> {
        let model = load_kv(&checkpoint.0, WeightFormat::F32, &device, execution, kv)?;
        let pool = pool(&model)?;
        // A whole prefill, then chunks that cross pages and write into partial tails.
        let (_, state) = prefill(&model, ModelState::new(&pool), &tokens[..100], 100)?;
        Ok(prefill(&model, state, &tokens[100..], 7)?.0)
    };
    let exact = run(Execution::Fused, F32_KV)?;
    // Page size changes addressing only, never values.
    for page in [1, 7, 32, 64, 128] {
        let error = max_relative(
            &run(
                Execution::Fused,
                KvLayout {
                    page_tokens: page,
                    ..F32_KV
                },
            )?,
            &exact,
        );
        assert!(error < 1e-5, "page {page}: {error}");
    }
    for (format, bound) in [
        (KvFormat::F16, 1e-3),
        (KvFormat::Q8, 2e-2),
        (KvFormat::Q8 { group: 64 }, 2e-2),
    ] {
        let kv = KvLayout { format, ..F32_KV };
        let fused = run(Execution::Fused, kv)?;
        let reference = run(Execution::Reference, kv)?;
        // The kernels read exactly what the reference quantizer stores.
        let error = max_relative(&fused, &reference);
        assert!(error < 1e-3, "{format} fused vs reference: {error}");
        let error = max_relative(&fused, &exact);
        eprintln!("{format}: logits vs F32 K/V {error:.2e}");
        assert!(error < bound, "{format} vs F32 K/V: {error}");
    }
    Ok(())
}

#[test]
fn test_decode_reads_pages_in_place_without_rebuilding_the_cache() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let Device::Metal(raw) = &device else {
        return Err(TurError::Other("expected Metal".into()));
    };
    let model = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    let pool = pool(&model)?;
    let (_, short) = prefill(&model, ModelState::new(&pool), &prompt(40, 21), 40)?;
    let (_, long) = prefill(&model, ModelState::new(&pool), &prompt(1500, 22), 500)?;
    // Synchronizing drops every unused pooled buffer, and buffers a pass frees stay
    // allocated until the next synchronization: growth across one encoded pass counts all
    // the memory it touched.
    let pass = |state: &ModelState| -> Result<usize> {
        device.synchronize()?;
        let before = raw.current_allocated_size();
        let new = prompt(1, 23);
        let (steps, reservations) = forward(&model, &[(state, &new, Boundaries::Final)])?;
        let grown = raw.current_allocated_size().saturating_sub(before);
        for step in steps {
            step?;
        }
        drop(reservations);
        device.synchronize()?;
        pool.reclaim();
        Ok(grown)
    };
    let short_pass = pass(&short)?;
    let long_pass = pass(&long)?;
    // Rebuilding the cache would copy every committed key and value of both attention
    // layers each pass; reading pages in place adds only page addressing and split-K
    // partials with context.
    let rebuilt = 2 * 1500 * 2 * 2 * 64 * 4;
    let extra = long_pass.saturating_sub(short_pass);
    eprintln!("decode pass: {short_pass} bytes at 40 tokens, {long_pass} at 1500");
    assert!(
        extra * 16 < rebuilt,
        "decode at 1500 tokens allocated {extra} more bytes"
    );
    Ok(())
}

#[test]
fn test_execution_selection_and_declared_workspace() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let cpu = load(
        &checkpoint.0,
        WeightFormat::F32,
        &Device::Cpu,
        Execution::Fused,
    )?;
    assert_eq!(cpu.execution(), Execution::Reference);
    let device = metal()?;
    let model = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    let pool = pool(&model)?;
    let state = ModelState::new(&pool);
    let workspace = |n: usize, model: &Qwen35| -> Result<usize> {
        let tokens = prompt(n, 15);
        let positions: Vec<usize> = (0..n).collect();
        let logits = [n - 1];
        let reservation = state.reserve(n, Boundaries::Final)?;
        let binding = reservation.binding();
        model.workspace_bytes(&PackedForward {
            tokens: &tokens,
            positions: &positions,
            sequences: &[SequenceForward {
                tokens: 0..n,
                state: &binding,
                logits: &logits,
            }],
            pool: &pool,
        })
    };
    let (one, many) = (workspace(1, &model)?, workspace(512, &model)?);
    assert!(one > 0 && many > 100 * one, "{one} {many}");
    assert_eq!(model.planned_workspace_bytes(512, &[512])?, many);
    // The plan bounds the device memory one pass adds beyond its logits: pools are
    // allocated up front and the pass writes state into them.
    let Device::Metal(raw) = &device else {
        return Err(TurError::Other("expected Metal".into()));
    };
    device.synchronize()?;
    let before = raw.current_allocated_size();
    let (mut steps, _held) = forward(&model, &[(&state, &prompt(512, 15), Boundaries::Final)])?;
    let step = steps.pop().ok_or_else(|| invalid("missing step"))??;
    device.synchronize()?;
    let grown = raw.current_allocated_size().saturating_sub(before);
    let logits = (step.logits.elem_count() * 4).next_power_of_two();
    assert!(
        grown <= many + logits,
        "grew {grown}, plan {many} + logits {logits}"
    );
    let reference = model.with_execution(Execution::Reference)?;
    assert_eq!(workspace(512, &reference)?, 0);
    Ok(())
}

#[test]
fn test_f16_recurrent_state_halves_slot_memory_and_tracks_f32_state() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let exact = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    let narrow = load_state(
        &checkpoint.0,
        WeightFormat::F32,
        &device,
        Execution::Fused,
        F32_KV,
        DType::F16,
    )?;
    let (f32_slot, f16_slot) = (
        exact.description().recurrent_slot_bytes()?,
        narrow.description().recurrent_slot_bytes()?,
    );
    assert!(
        f16_slot * 2 > f32_slot && f16_slot * 3 < f32_slot * 2,
        "{f16_slot} vs {f32_slot}"
    );
    // A chunked prefill, then token-by-token decode: every decode step stores the state.
    let tokens = prompt(120, 21);
    let mut logits = Vec::new();
    for model in [&exact, &narrow] {
        let pool = pool(model)?;
        let (_, state) = prefill(model, ModelState::new(&pool), &tokens[..80], 40)?;
        logits.push(prefill(model, state, &tokens[80..], 1)?.0);
    }
    let error = max_relative(&logits[1], &logits[0]);
    assert!(error > 0.0 && error < 1e-2, "F16 state decode {error}");
    Ok(())
}

#[test]
fn test_fused_decode_reaches_a_stable_allocation_bound() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let model = load_kv(
        &checkpoint.0,
        WeightFormat::Q8_0,
        &device,
        Execution::Fused,
        KvLayout::new(16, KvFormat::Q8),
    )?;
    let pool = pool(&model)?;
    let (_, mut a) = prefill(&model, ModelState::new(&pool), &prompt(40, 16), 40)?;
    let (_, mut b) = prefill(&model, ModelState::new(&pool), &prompt(9, 17), 9)?;
    let Device::Metal(raw) = &device else {
        return Err(TurError::Other("expected Metal".into()));
    };
    // Committed decode: pages fill and open, slots turn over, all inside the pool.
    let mut decode = |step: usize| -> Result<()> {
        let new = [prompt(1, 18 + step), prompt(1, 19 + step)];
        let (steps, reservations) = forward(
            &model,
            &[
                (&a, new[0].as_slice(), Boundaries::Final),
                (&b, new[1].as_slice(), Boundaries::Final),
            ],
        )?;
        for step in steps {
            step?;
        }
        a = reservations[0].commit(1)?;
        b = reservations[1].commit(1)?;
        drop(reservations);
        device.synchronize()?;
        pool.reclaim();
        Ok(())
    };
    for step in 0..3 {
        decode(step)?;
    }
    let settled = raw.current_allocated_size();
    for step in 3..40 {
        decode(step)?;
    }
    assert_eq!(raw.current_allocated_size(), settled);
    let usage = pool.usage()?;
    assert_eq!(
        usage.pages.leased,
        (40 + 40usize).div_ceil(16) + (9 + 40usize).div_ceil(16)
    );
    assert_eq!(usage.slots.leased, 2);
    Ok(())
}

fn greedy(engine: &mut HybridEngine<Qwen35>) -> Result<StateHandle> {
    engine.allocate(0, SamplingOptions::default(), None)
}

/// Feed `tokens` to one request and return the sampled token after the last.
fn feed(engine: &mut HybridEngine<Qwen35>, request: StateHandle, tokens: &[u32]) -> Result<u32> {
    let meta = engine.metadata(request.request)?;
    let n = tokens.len();
    engine.submit(PackedExecution {
        positions: (meta.position..meta.position + n).collect(),
        tokens: tokens.to_vec(),
        sequences: vec![SequenceDescriptor {
            state: meta.handle,
            tokens: 0..n,
            outputs: vec![n - 1],
            speculative: false,
        }],
    })?;
    engine
        .finish()?
        .pop()
        .ok_or_else(|| invalid("missing result"))?
        .outcome?
        .outputs
        .first()
        .map(|o| o.token)
        .ok_or_else(|| invalid("no output"))
}

/// Greedy continuation of `steps` tokens after the request's committed state.
fn continue_greedy(
    engine: &mut HybridEngine<Qwen35>,
    request: StateHandle,
    first: u32,
    steps: usize,
) -> Result<Vec<u32>> {
    let mut out = vec![first];
    while out.len() < steps {
        let last = *out.last().ok_or_else(|| invalid("empty"))?;
        out.push(feed(engine, request, &[last])?);
    }
    Ok(out)
}

#[test]
fn test_full_and_partial_prefix_forks_and_restores_continue_exactly() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let model = Arc::new(load_kv(
        &checkpoint.0,
        WeightFormat::F32,
        &device,
        Execution::Fused,
        KvLayout::new(16, KvFormat::Q8),
    )?);
    let config = PoolConfig {
        pages: 64,
        recurrent_slots: 16,
    };
    let mut engine = HybridEngine::new(Arc::clone(&model), device, config)?;
    // Prefixes ending mid-page (37) and on a page boundary (48).
    for length in [37usize, 48] {
        let tokens = prompt(length + 3, 24);
        let (prefix, suffix) = tokens.split_at(length);
        let a = greedy(&mut engine)?;
        let first = feed(&mut engine, a, prefix)?;
        let saved = engine.checkpoint(engine.metadata(a.request)?.handle)?;
        let saved_state = snapshot(saved.model_state())?;
        // Branch 1 continues greedily; branch 2 takes another suffix through a fork.
        let expected = continue_greedy(&mut engine, a, first, 6)?;
        let b = engine.fork(&saved)?;
        let fork_next = feed(&mut engine, b, suffix)?;
        // Fresh requests computing the same full sequences without any sharing.
        let fresh = greedy(&mut engine)?;
        let fresh_first = feed(&mut engine, fresh, prefix)?;
        assert_eq!(fresh_first, first);
        assert_eq!(
            continue_greedy(&mut engine, fresh, fresh_first, 6)?,
            expected
        );
        let fresh_b = greedy(&mut engine)?;
        assert_eq!(feed(&mut engine, fresh_b, &tokens)?, fork_next);
        // The checkpoint never changed, and restoring it replays exactly.
        assert_eq!(snapshot(saved.model_state())?, saved_state);
        let a = engine.restore(engine.metadata(a.request)?.handle, &saved)?;
        assert_eq!(continue_greedy(&mut engine, a, first, 6)?, expected);
        // A speculative pass accepting a partial prefix publishes that exact boundary:
        // drafts e1, e2 match and the corrupted third is replaced by the sampled e3.
        let c = engine.fork(&saved)?;
        let mut drafts = vec![first, expected[1], expected[2]];
        drafts.push((expected[3] + 1) % 148 + 2);
        let meta = engine.metadata(c.request)?;
        let n = drafts.len();
        engine.submit(PackedExecution {
            positions: (meta.position..meta.position + n).collect(),
            tokens: drafts,
            sequences: vec![SequenceDescriptor {
                state: meta.handle,
                tokens: 0..n,
                outputs: (0..n).collect(),
                speculative: true,
            }],
        })?;
        let commit = engine
            .finish()?
            .pop()
            .ok_or_else(|| invalid("missing result"))?
            .outcome?;
        assert_eq!(commit.accepted_tokens, 3);
        let outputs: Vec<u32> = commit.outputs.iter().map(|o| o.token).collect();
        assert_eq!(outputs, expected[1..4]);
        let next = *outputs.last().ok_or_else(|| invalid("no output"))?;
        assert_eq!(continue_greedy(&mut engine, c, next, 3)?, expected[3..6]);
        for handle in [a, b, c, fresh, fresh_b] {
            engine.cancel(engine.metadata(handle.request)?.handle)?;
        }
    }
    let usage = engine.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_memory_pressure_fails_new_work_without_corrupting_live_requests() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let model = Arc::new(load_kv(
        &checkpoint.0,
        WeightFormat::Q8_0,
        &device,
        Execution::Fused,
        KvLayout::new(16, KvFormat::Q8),
    )?);
    let tokens = prompt(60, 25);
    let mut control = HybridEngine::new(
        Arc::clone(&model),
        device.clone(),
        PoolConfig {
            pages: 64,
            recurrent_slots: 8,
        },
    )?;
    let c = greedy(&mut control)?;
    let first = feed(&mut control, c, &tokens)?;
    let expected = continue_greedy(&mut control, c, first, 20)?;
    // Seven pages: request a holds 4 to 5 of them (60 to 80 tokens), so b's 64-token
    // prefill (4 pages) never fits beside it.
    let mut engine = HybridEngine::new(
        Arc::clone(&model),
        device,
        PoolConfig {
            pages: 7,
            recurrent_slots: 4,
        },
    )?;
    let a = greedy(&mut engine)?;
    let mut out = vec![feed(&mut engine, a, &tokens)?];
    let b = greedy(&mut engine)?;
    let mut rejected = 0;
    while out.len() < expected.len() {
        // b keeps asking for a 64-token prefill, which never fits beside a.
        if engine.metadata(b.request)?.position == 0 {
            match feed(&mut engine, b, &prompt(64, 26)) {
                Err(TurError::MemoryAllocation(_)) => rejected += 1,
                other => return Err(invalid(format!("b was admitted: {other:?}"))),
            }
        }
        let last = *out.last().ok_or_else(|| invalid("empty"))?;
        out.push(feed(&mut engine, a, &[last])?);
    }
    assert_eq!(out, expected);
    assert!(rejected > 0);
    assert_eq!(engine.metadata(b.request)?.position, 0);
    // Releasing a admits b; the pool never grew.
    engine.cancel(engine.metadata(a.request)?.handle)?;
    feed(&mut engine, b, &prompt(64, 26))?;
    let usage = engine.memory()?;
    assert_eq!(usage.pages.capacity, 7);
    assert!(usage.pages.peak <= 7);
    Ok(())
}

fn env_path(name: &str) -> Result<std::path::PathBuf> {
    std::env::var_os(name)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| TurError::Other(format!("set {name}")))
}

fn real(format: WeightFormat, execution: Execution, kv: KvLayout) -> Result<Qwen35> {
    Qwen35::load(
        &env_path("TUR_QWEN35_0_8B")?,
        &LoadOptions {
            revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
            format,
            device: metal()?,
            execution,
            kv,
            recurrent: DType::F32,
        },
    )
}

/// A fixed 97-token chat prompt: six 16-token pages, crossing one 64-token delta chunk.
fn real_prompt() -> Result<Vec<u32>> {
    let tokenizer =
        tokenizers::Tokenizer::from_file(env_path("TUR_QWEN35_0_8B")?.join("tokenizer.json"))
            .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let text = "<|im_start|>user\nExplain, step by step and with one small numeric example, how a \
                gated delta rule updates a recurrent memory matrix, why the decay term keeps \
                old associations from dominating, how chunked prefill computes the same state \
                as token-by-token recurrence, and what has to be stored so that a speculative \
                decoder can roll back rejected draft tokens without recomputing the whole \
                prompt. Keep the answer under two hundred words and finish with a one-line \
                summary.<|im_end|>\n<|im_start|>assistant\n";
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| TurError::Tokenizer(e.to_string()))?
        .get_ids()
        .to_vec())
}

fn argmax(rows: &[Vec<f32>]) -> Vec<usize> {
    rows.iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |m, (i, &v)| if v > m.1 { (i, v) } else { m },
                )
                .0
        })
        .collect()
}

#[test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint and host GPU access"]
fn test_real_checkpoint_fused_matches_reference() -> Result<()> {
    let _gpu = GPU.lock();
    let tokens = real_prompt()?;
    let all: Vec<usize> = (0..tokens.len()).collect();
    // F32: every layer and all logits within the P3 reference gate.
    let fused = real(WeightFormat::F32, Execution::Fused, F32_KV)?;
    let (fused_layers, fused_logits) = fused.trace(&tokens, &all)?;
    let reference = fused.with_execution(Execution::Reference)?;
    let (layers, logits) = reference.trace(&tokens, &all)?;
    drop(reference);
    let mut worst = 0f32;
    for (a, b) in fused_layers.iter().zip(&layers) {
        worst = worst.max(max_relative(&rows(a)?, &rows(b)?));
    }
    let (a, b) = (rows(&fused_logits)?, rows(&logits)?);
    let logit_error = max_relative(&a, &b);
    let agree = argmax(&a)
        .iter()
        .zip(argmax(&b))
        .filter(|(x, y)| **x == *y)
        .count();
    eprintln!(
        "F32 {} tokens: worst layer {worst:.2e}, logits {logit_error:.2e}, argmax {agree}/{}",
        tokens.len(),
        tokens.len()
    );
    assert!(worst < 1e-4 && logit_error < 1e-4);
    assert_eq!(agree, tokens.len());
    // Packed formats: fused GEMV/kernels against Candle's packed kernels on equal weights.
    for format in [WeightFormat::Q8_0, WeightFormat::Q4K] {
        let fused = real(format, Execution::Fused, F32_KV)?;
        let (_, x) = fused.trace(&tokens, &all)?;
        let reference = fused.with_execution(Execution::Reference)?;
        let (_, y) = reference.trace(&tokens, &all)?;
        let (x, y) = (rows(&x)?, rows(&y)?);
        let error = max_relative(&x, &y);
        let agree = argmax(&x)
            .iter()
            .zip(argmax(&y))
            .filter(|(p, q)| **p == *q)
            .count();
        eprintln!(
            "{format:?}: logits {error:.2e}, argmax {agree}/{}",
            tokens.len()
        );
        assert!(error < 1e-3, "{format:?} {error}");
    }
    Ok(())
}

fn generate(
    model: Arc<Qwen35>,
    prompt: &[u32],
    limit: usize,
    batch: usize,
) -> Result<Vec<Vec<u32>>> {
    let device = model.device().clone();
    let config = PoolConfig {
        pages: 64,
        recurrent_slots: 2 * batch,
    };
    let mut engine = HybridEngine::new(model, device, config)?;
    let handles = (0..batch)
        .map(|_| greedy(&mut engine))
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = vec![Vec::new(); batch];
    let mut next = vec![prompt.to_vec(); batch];
    for _ in 0..limit {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut sequences = Vec::new();
        for (h, pending) in handles.iter().zip(&mut next) {
            let meta = engine.metadata(h.request)?;
            let start = tokens.len();
            positions.extend(meta.position..meta.position + pending.len());
            tokens.append(pending);
            sequences.push(SequenceDescriptor {
                state: meta.handle,
                tokens: start..tokens.len(),
                outputs: vec![tokens.len() - 1],
                speculative: false,
            });
        }
        engine.submit(PackedExecution {
            tokens,
            positions,
            sequences,
        })?;
        for ((result, out), pending) in engine
            .finish()?
            .into_iter()
            .zip(&mut outputs)
            .zip(&mut next)
        {
            let token = result
                .outcome?
                .outputs
                .first()
                .map(|o| o.token)
                .ok_or_else(|| invalid("no output"))?;
            out.push(token);
            pending.push(token);
        }
    }
    Ok(outputs)
}

#[test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint and host GPU access"]
fn test_real_checkpoint_fused_greedy_generation_matches_reference() -> Result<()> {
    let _gpu = GPU.lock();
    let tokens = real_prompt()?;
    let expected = generate(
        Arc::new(real(WeightFormat::F32, Execution::Reference, F32_KV)?),
        &tokens,
        32,
        1,
    )?;
    // Three identical requests decode as one packed batch through the fused kernels.
    let actual = generate(
        Arc::new(real(WeightFormat::F32, Execution::Fused, F32_KV)?),
        &tokens,
        32,
        3,
    )?;
    for output in &actual {
        assert_eq!(output, &expected[0]);
    }
    Ok(())
}

#[test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint and host GPU access"]
fn test_real_checkpoint_kv_formats_keep_attention_quality() -> Result<()> {
    let _gpu = GPU.lock();
    let tokens = real_prompt()?;
    let all: Vec<usize> = (0..tokens.len()).collect();
    let (_, exact) = real(WeightFormat::F32, Execution::Fused, F32_KV)?.trace(&tokens, &all)?;
    let exact = rows(&exact)?;
    let greedy_exact = generate(
        Arc::new(real(WeightFormat::F32, Execution::Fused, F32_KV)?),
        &tokens,
        64,
        1,
    )?;
    // Predeclared budgets: teacher-forced top-1 agreement over the prompt and the
    // greedy prefix that must match F32 K/V before any divergence.
    for (format, agreement, prefix) in [
        (KvFormat::F16, 1.0, 64),
        (KvFormat::Q8, 0.97, 16),
        (KvFormat::Q8 { group: 128 }, 0.95, 8),
    ] {
        let kv = KvLayout { format, ..F32_KV };
        let (_, logits) = real(WeightFormat::F32, Execution::Fused, kv)?.trace(&tokens, &all)?;
        let logits = rows(&logits)?;
        let error = max_relative(&logits, &exact);
        let agree = argmax(&logits)
            .iter()
            .zip(argmax(&exact))
            .filter(|(a, b)| **a == *b)
            .count() as f64
            / tokens.len() as f64;
        let output = generate(
            Arc::new(real(WeightFormat::F32, Execution::Fused, kv)?),
            &tokens,
            64,
            1,
        )?;
        let same = output[0]
            .iter()
            .zip(&greedy_exact[0])
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!("{format}: logits {error:.2e}, top-1 {agree:.3}, greedy prefix {same}/64");
        assert!(agree >= agreement, "{format}: agreement {agree}");
        assert!(same >= prefix, "{format}: greedy prefix {same}");
    }
    Ok(())
}
