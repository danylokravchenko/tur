#![cfg(feature = "metal")]
#![forbid(unsafe_code)]
//! Fused Metal execution against the F32 reference on identical weights: per-layer and
//! logit agreement, chunked/incremental prefill, batch isolation, speculative boundaries,
//! packed formats, workspace and allocation bounds. Needs host GPU access and fails,
//! rather than skips, without it. Real-checkpoint tests are `#[ignore]` and need
//! `TUR_QWEN35_0_8B`; the reference itself is validated against Transformers in P3.

mod common;

use candle_core::{Device, Tensor};
use common::{Checkpoint, max_relative};
use std::{path::Path, sync::Arc};
use tur_core::{
    Result, TurError,
    execution::{PackedExecution, SequenceDescriptor, invalid},
};
use tur_engine::hybrid::{
    Boundaries, HybridEngine, HybridModel, PackedForward, SamplingOptions, SequenceForward,
    SequenceStep,
};
use tur_metal::state::{LayerUpdate, ModelState};
use tur_models::models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat};

/// Kernel-supported head sizes, two attention layers and projection widths that are
/// multiples of 256 so Q4K applies everywhere.
const SMALL: &str = r#"{
  "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
  "hidden_size": 256, "intermediate_size": 512, "vocab_size": 160, "num_hidden_layers": 4,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
  "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 64,
  "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attn_output_gate": true,
  "tie_word_embeddings": false, "eos_token_id": 1, "image_token_id": 159,
  "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
}"#;

/// Device-wide allocation counters are shared by every test in the process.
static GPU: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn metal() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

fn small() -> Result<Checkpoint> {
    common::checkpoint(SMALL, 0.08, 0.1)
}

fn load(dir: &Path, format: WeightFormat, device: &Device, execution: Execution) -> Result<Qwen35> {
    Qwen35::load(
        dir,
        &LoadOptions {
            revision: "small-fixture".into(),
            format,
            device: device.clone(),
            execution,
        },
    )
}

fn prompt(len: usize, salt: usize) -> Vec<u32> {
    // Token 159 is the image placeholder; prompts stay within 2..150.
    (0..len)
        .map(|i| ((i * 37 + 11 + salt * 13) % 148 + 2) as u32)
        .collect()
}

fn rows(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(tensor.to_device(&Device::Cpu)?.to_vec2::<f32>()?)
}

fn flat(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(vec![
        tensor
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?,
    ])
}

/// One forward of `sequences` (each `(state, new tokens, boundaries)`), all logits rows.
fn step(
    model: &Qwen35,
    sequences: &[(&ModelState, &[u32], Boundaries)],
) -> Result<Vec<Result<SequenceStep>>> {
    let reads: Vec<_> = sequences.iter().map(|(s, _, _)| s.bind_layers()).collect();
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut ranges = Vec::new();
    for (state, new, _) in sequences {
        let start = tokens.len();
        tokens.extend_from_slice(new);
        positions.extend(state.position()..state.position() + new.len());
        ranges.push(start..tokens.len());
    }
    let logits: Vec<Vec<usize>> = ranges.iter().map(|r| r.clone().collect()).collect();
    let forwards: Vec<SequenceForward<'_>> = sequences
        .iter()
        .zip(&reads)
        .zip(&ranges)
        .zip(&logits)
        .map(
            |((((_, _, boundaries), state), range), logits)| SequenceForward {
                tokens: range.clone(),
                state,
                logits,
                boundaries: *boundaries,
            },
        )
        .collect();
    model.forward(&PackedForward {
        tokens: &tokens,
        positions: &positions,
        sequences: &forwards,
    })
}

/// Prefill `tokens` in `chunk`-sized pieces, returning every position's logits and the state.
fn prefill(model: &Qwen35, tokens: &[u32], chunk: usize) -> Result<(Vec<Vec<f32>>, ModelState)> {
    let mut state = ModelState::allocate(Arc::new(model.description()), model.device())?;
    let mut out = Vec::new();
    for piece in tokens.chunks(chunk) {
        let result = step(model, &[(&state, piece, Boundaries::Final)])?
            .pop()
            .ok_or_else(|| invalid("missing step"))??;
        out.extend(rows(&result.logits)?);
        let staged = state.stage(result.layers, piece.len(), Boundaries::Final)?;
        state = state.apply(&staged, piece.len())?;
    }
    Ok((out, state))
}

#[test]
fn test_fused_layers_and_logits_match_reference_across_page_and_chunk_boundaries() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let fused = load(&checkpoint.0, WeightFormat::F32, &device, Execution::Fused)?;
    assert_eq!(fused.execution(), Execution::Fused);
    // 150 tokens cross 16-token pages and the reference's 64-token delta chunks.
    let tokens = prompt(150, 0);
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
    let (whole, _) = prefill(&reference, &tokens, 150)?;
    for chunk in [150, 65, 37, 16, 1] {
        let (logits, state) = prefill(&fused, &tokens, chunk)?;
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
    // Different committed lengths (including a partial page and an empty state).
    let (_, a) = prefill(&model, &prompt(37, 2), 37)?;
    let (_, b) = prefill(&model, &prompt(16, 3), 16)?;
    let c = ModelState::allocate(Arc::new(model.description()), model.device())?;
    let new = [prompt(1, 4), prompt(5, 5), prompt(20, 6)];
    let batch: Vec<(&ModelState, &[u32], Boundaries)> = [&a, &b, &c]
        .into_iter()
        .zip(&new)
        .map(|(state, tokens)| (state, tokens.as_slice(), Boundaries::Final))
        .collect();
    let together = step(&model, &batch)?;
    for (packed, single) in together.into_iter().zip(&batch) {
        let packed = packed?;
        let alone = step(&model, std::slice::from_ref(single))?
            .pop()
            .ok_or_else(|| invalid("missing step"))??;
        let error = max_relative(&rows(&packed.logits)?, &rows(&alone.logits)?);
        assert!(error < 1e-5, "logits {error}");
        for (x, y) in packed.layers.iter().zip(&alone.layers) {
            let pairs = match (x, y) {
                (
                    LayerUpdate::Attention { key, value },
                    LayerUpdate::Attention { key: k2, value: v2 },
                ) => [(key, k2), (value, v2)],
                (
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    },
                    LayerUpdate::Recurrent {
                        convolution: c2,
                        recurrent: r2,
                    },
                ) => [(convolution, c2), (recurrent, r2)],
                _ => return Err(invalid("update kinds differ")),
            };
            for (p, q) in pairs {
                assert!(max_relative(&flat(p)?, &flat(q)?) < 1e-5);
            }
        }
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
    let (_, state) = prefill(&reference, &prompt(20, 7), 20)?;
    let draft = prompt(5, 8);
    let run = |model: &Qwen35| -> Result<SequenceStep> {
        step(model, &[(&state, &draft, Boundaries::EveryToken)])?
            .pop()
            .ok_or_else(|| invalid("missing step"))?
    };
    let (a, b) = (run(&fused)?, run(&reference)?);
    assert!(max_relative(&rows(&a.logits)?, &rows(&b.logits)?) < 1e-4);
    for (x, y) in a.layers.iter().zip(&b.layers) {
        if let (
            LayerUpdate::Recurrent {
                convolution,
                recurrent,
            },
            LayerUpdate::Recurrent {
                convolution: c2,
                recurrent: r2,
            },
        ) = (x, y)
        {
            assert_eq!(recurrent.dim(0)?, 5);
            assert_eq!(convolution.dims(), c2.dims());
            assert!(max_relative(&flat(convolution)?, &flat(c2)?) < 1e-6);
            assert!(max_relative(&flat(recurrent)?, &flat(r2)?) < 1e-4);
        }
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
        let (_, a) = prefill(&reference, &prompt(30, 9), 30)?;
        let (_, b) = prefill(&reference, &prompt(7, 10), 7)?;
        // Three single-token sequences: the fused path takes the small-batch GEMV. The
        // reference run per sequence uses Candle's one-row GEMV (F32 dequantization); run
        // as a batch it uses Candle's tiled GEMM, which stages weight tiles as F16.
        let new = [prompt(1, 11), prompt(1, 12), prompt(1, 13)];
        let batch: Vec<(&ModelState, &[u32], Boundaries)> = [&a, &b, &a]
            .into_iter()
            .zip(&new)
            .map(|(state, tokens)| (state, tokens.as_slice(), Boundaries::Final))
            .collect();
        let tiled = step(&reference, &batch)?;
        for ((fused, tiled), single) in step(&fused, &batch)?.into_iter().zip(tiled).zip(&batch) {
            let alone = step(&reference, std::slice::from_ref(single))?
                .pop()
                .ok_or_else(|| invalid("missing step"))??;
            let (x, y, z) = (
                rows(&fused?.logits)?,
                rows(&alone.logits)?,
                rows(&tiled?.logits)?,
            );
            let error = max_relative(&x, &y);
            assert!(error < 1e-4, "{format:?} decode vs one-row GEMV {error}");
            let error = max_relative(&z, &y);
            assert!(
                error < 2e-2,
                "{format:?} F16-tiled GEMM vs one-row GEMV {error}"
            );
        }
        let tokens = prompt(100, 14);
        let error = max_relative(
            &prefill(&fused, &tokens, 100)?.0,
            &prefill(&reference, &tokens, 100)?.0,
        );
        assert!(error < 1e-4, "{format:?} prefill {error}");
    }
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
    let state = ModelState::allocate(Arc::new(model.description()), &device)?;
    let reads = state.bind_layers();
    let workspace = |n: usize, model: &Qwen35| -> Result<usize> {
        let tokens = prompt(n, 15);
        let positions: Vec<usize> = (0..n).collect();
        let logits = [n - 1];
        model.workspace_bytes(&PackedForward {
            tokens: &tokens,
            positions: &positions,
            sequences: &[SequenceForward {
                tokens: 0..n,
                state: &reads,
                logits: &logits,
                boundaries: Boundaries::Final,
            }],
        })
    };
    let (one, many) = (workspace(1, &model)?, workspace(512, &model)?);
    assert!(one > 0 && many > 100 * one, "{one} {many}");
    // The plan bounds the device memory one pass actually adds beyond its outputs.
    let Device::Metal(raw) = &device else {
        return Err(TurError::Other("expected Metal".into()));
    };
    device.synchronize()?;
    let before = raw.current_allocated_size();
    let result = step(&model, &[(&state, &prompt(512, 15), Boundaries::Final)])?
        .pop()
        .ok_or_else(|| invalid("missing step"))??;
    device.synchronize()?;
    let grown = raw.current_allocated_size().saturating_sub(before);
    let mut outputs = result.logits.elem_count() * 4;
    for update in &result.layers {
        let (a, b) = match update {
            LayerUpdate::Attention { key, value } => (key, value),
            LayerUpdate::Recurrent {
                convolution,
                recurrent,
            } => (convolution, recurrent),
        };
        outputs +=
            (a.elem_count() * 4).next_power_of_two() + (b.elem_count() * 4).next_power_of_two();
    }
    assert!(
        grown <= many + outputs,
        "grew {grown}, plan {many} + outputs {outputs}"
    );
    let reference = model.with_execution(Execution::Reference)?;
    assert_eq!(workspace(512, &reference)?, 0);
    Ok(())
}

#[test]
fn test_fused_decode_reaches_a_stable_allocation_bound() -> Result<()> {
    let _gpu = GPU.lock();
    let checkpoint = small()?;
    let device = metal()?;
    let model = load(&checkpoint.0, WeightFormat::Q8_0, &device, Execution::Fused)?;
    let (_, a) = prefill(&model, &prompt(40, 16), 40)?;
    let (_, b) = prefill(&model, &prompt(9, 17), 9)?;
    let new = [prompt(1, 18), prompt(1, 19)];
    let Device::Metal(raw) = &device else {
        return Err(TurError::Other("expected Metal".into()));
    };
    let decode = || -> Result<()> {
        for result in step(
            &model,
            &[
                (&a, new[0].as_slice(), Boundaries::Final),
                (&b, new[1].as_slice(), Boundaries::Final),
            ],
        )? {
            result?;
        }
        Ok(device.synchronize()?)
    };
    for _ in 0..3 {
        decode()?;
    }
    let settled = raw.current_allocated_size();
    for _ in 0..20 {
        decode()?;
    }
    assert_eq!(raw.current_allocated_size(), settled);
    Ok(())
}

fn env_path(name: &str) -> Result<std::path::PathBuf> {
    std::env::var_os(name)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| TurError::Other(format!("set {name}")))
}

fn real(format: WeightFormat, execution: Execution) -> Result<Qwen35> {
    Qwen35::load(
        &env_path("TUR_QWEN35_0_8B")?,
        &LoadOptions {
            revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
            format,
            device: metal()?,
            execution,
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
    let fused = real(WeightFormat::F32, Execution::Fused)?;
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
    println!(
        "F32 {} tokens: worst layer {worst:.2e}, logits {logit_error:.2e}, argmax {agree}/{}",
        tokens.len(),
        tokens.len()
    );
    assert!(worst < 1e-4 && logit_error < 1e-4);
    assert_eq!(agree, tokens.len());
    // Packed formats: fused GEMV/kernels against Candle's packed kernels on equal weights.
    for format in [WeightFormat::Q8_0, WeightFormat::Q4K] {
        let fused = real(format, Execution::Fused)?;
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
        println!(
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
    let mut engine = HybridEngine::new(model, device)?;
    let handles = (0..batch)
        .map(|_| engine.allocate(0, SamplingOptions::default(), None))
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
        Arc::new(real(WeightFormat::F32, Execution::Reference)?),
        &tokens,
        32,
        1,
    )?;
    // Three identical requests decode as one packed batch through the fused kernels.
    let actual = generate(
        Arc::new(real(WeightFormat::F32, Execution::Fused)?),
        &tokens,
        32,
        3,
    )?;
    for output in &actual {
        assert_eq!(output, &expected[0]);
    }
    Ok(())
}
