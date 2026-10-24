#![allow(
    clippy::print_stderr,
    reason = "checkpoint comparisons report numerical evidence to the test runner"
)]
#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![forbid(unsafe_code)]
//! Qwen3.5 through `HybridEngine`: prefill/chunk/decode agreement, batch isolation,
//! speculation and quantization on a tiny random checkpoint, plus real-checkpoint
//! generation (`#[ignore]`; needs `TUR_QWEN35_0_8B` and `TUR_QWEN35_GOLDEN`).

mod common;

use candle_core::{Device, Tensor};
use common::{Checkpoint, max_relative};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tur_core::{
    Result, TurError,
    execution::{KvFormat, KvLayout, PackedExecution, SequenceDescriptor, StateHandle, invalid},
};
use tur_engine::hybrid::{HybridEngine, HybridModel, ModelState, PoolConfig, SamplingOptions};
use tur_models::models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat};

/// Exact K/V storage: these tests compare chunking, batching and Transformers outputs.
const EXACT_KV: KvLayout = KvLayout {
    page_tokens: 16,
    format: KvFormat::F32,
    version: tur_core::execution::KV_LAYOUT_VERSION,
};

/// Room for a few full contexts of the tiny model.
const POOL: PoolConfig = PoolConfig {
    pages: 128,
    recurrent_slots: 8,
};

const TINY: &str = r#"{
  "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
  "hidden_size": 64, "intermediate_size": 128, "vocab_size": 96, "num_hidden_layers": 4,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
  "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
  "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "max_position_embeddings": 512,
  "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attn_output_gate": true,
  "tie_word_embeddings": true, "eos_token_id": 1, "image_token_id": 95,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25}
}"#;

fn tiny_checkpoint() -> Result<Checkpoint> {
    common::checkpoint(TINY, 0.3, 0.1)
}

fn load(dir: &Path, format: WeightFormat, device: Device) -> Result<Arc<Qwen35>> {
    Ok(Arc::new(Qwen35::load(
        dir,
        &LoadOptions {
            revision: "tiny-fixture".into(),
            weights: format.into(),
            device,
            execution: Execution::Fused,
            kv: EXACT_KV,
            recurrent: candle_core::DType::F32,
            mtp: false,
            draft_vocab: None,
        },
    )?))
}

fn prompt(len: usize) -> Vec<u32> {
    // Token 95 is the tiny config's image placeholder; prompts stay within 2..90.
    (0..len).map(|i| ((i * 37 + 11) % 88 + 2) as u32).collect()
}

/// Logits at every position, feeding `tokens` in pieces of `chunk` through `forward`.
fn logits_by_chunks(model: &Qwen35, tokens: &[u32], chunk: usize) -> Result<Vec<Vec<f32>>> {
    let pool = common::pool(model, 16, 2)?;
    Ok(common::prefill(model, ModelState::new(&pool), tokens, chunk)?.0)
}

#[test]
fn test_whole_chunked_and_incremental_prefill_agree() -> Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = load(&checkpoint.0, WeightFormat::F32, Device::Cpu)?;
    let tokens = prompt(150);
    let whole = logits_by_chunks(&model, &tokens, 150)?;
    for chunk in [1, 37, 64, 65] {
        let other = logits_by_chunks(&model, &tokens, chunk)?;
        let error = max_relative(&other, &whole);
        assert!(error < 1e-4, "chunk {chunk}: relative error {error}");
    }
    Ok(())
}

struct Runner {
    engine: HybridEngine<Qwen35>,
}

impl Runner {
    fn new(model: Arc<Qwen35>) -> Result<Self> {
        Ok(Self {
            engine: HybridEngine::new(model, Device::Cpu, POOL)?,
        })
    }

    fn handle(&self, h: StateHandle) -> Result<StateHandle> {
        Ok(self.engine.metadata(h.request)?.handle)
    }

    /// Submit `(request, tokens)` pairs with an output at each last token; returns tokens.
    fn step(&mut self, items: &[(StateHandle, Vec<u32>)]) -> Result<Vec<u32>> {
        let mut plan = PackedExecution {
            tokens: Vec::new(),
            positions: Vec::new(),
            sequences: Vec::new(),
        };
        for (handle, tokens) in items {
            let meta = self.engine.metadata(handle.request)?;
            let start = plan.tokens.len();
            plan.tokens.extend_from_slice(tokens);
            plan.positions
                .extend((0..tokens.len()).map(|i| meta.position + i));
            plan.sequences.push(SequenceDescriptor {
                state: meta.handle,
                tokens: start..plan.tokens.len(),
                outputs: vec![plan.tokens.len() - 1],
                speculative: false,
            });
        }
        self.engine.submit(plan)?;
        self.engine
            .finish()?
            .into_iter()
            .map(|r| {
                r.outcome?
                    .outputs
                    .first()
                    .map(|o| o.token)
                    .ok_or_else(|| invalid("no output"))
            })
            .collect()
    }

    fn generate(&mut self, prompt: &[u32], steps: usize) -> Result<Vec<u32>> {
        let h = self.engine.allocate(0, SamplingOptions::default(), None)?;
        let mut out = self.step(&[(h, prompt.to_vec())])?;
        while out.len() < steps {
            let last = out.last().copied().ok_or_else(|| invalid("empty"))?;
            out.extend(self.step(&[(self.handle(h)?, vec![last])])?);
        }
        Ok(out)
    }
}

#[test]
fn test_batch_members_entering_and_leaving_do_not_leak_state() -> Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = load(&checkpoint.0, WeightFormat::F32, Device::Cpu)?;
    let a_prompt = prompt(20);
    let b_prompt: Vec<u32> = prompt(31).into_iter().rev().collect();
    let a_alone = Runner::new(Arc::clone(&model))?.generate(&a_prompt, 10)?;
    let b_alone = Runner::new(Arc::clone(&model))?.generate(&b_prompt, 4)?;

    let mut runner = Runner::new(model)?;
    let a = runner
        .engine
        .allocate(0, SamplingOptions::default(), None)?;
    let mut a_out = runner.step(&[(a, a_prompt)])?;
    let mut b = None;
    let mut b_out = Vec::new();
    while a_out.len() < 10 {
        let mut items = vec![(
            runner.handle(a)?,
            vec![*a_out.last().ok_or_else(|| invalid("a"))?],
        )];
        match (a_out.len(), b) {
            // B joins with its whole prompt, decodes, then leaves before A finishes.
            (3, None) => {
                let handle = runner
                    .engine
                    .allocate(0, SamplingOptions::default(), None)?;
                b = Some(handle);
                items.push((handle, b_prompt.clone()));
            }
            (4..=6, Some(handle)) => {
                items.insert(
                    0,
                    (
                        runner.handle(handle)?,
                        vec![*b_out.last().ok_or_else(|| invalid("b"))?],
                    ),
                );
            }
            (7, Some(handle)) => {
                runner.engine.cancel(runner.handle(handle)?)?;
                b = None;
            }
            _ => {}
        }
        let tokens = runner.step(&items)?;
        let (a_token, b_token) = match items.len() {
            1 => (tokens[0], None),
            _ if items[0].0.request == a.request => (tokens[0], Some(tokens[1])),
            _ => (tokens[1], Some(tokens[0])),
        };
        a_out.push(a_token);
        b_out.extend(b_token);
    }
    assert_eq!(a_out, a_alone);
    assert_eq!(b_out, b_alone);
    Ok(())
}

#[test]
fn test_speculative_verification_reproduces_greedy_decode() -> Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = load(&checkpoint.0, WeightFormat::F32, Device::Cpu)?;
    let tokens = prompt(12);
    let reference = Runner::new(Arc::clone(&model))?.generate(&tokens, 9)?;
    let mut runner = Runner::new(model)?;
    let h = runner
        .engine
        .allocate(0, SamplingOptions::default(), None)?;
    let mut out = runner.step(&[(h, tokens)])?;
    while out.len() < reference.len() {
        // Drafts: the next two reference tokens, with the second corrupted every other round.
        let at = out.len();
        let mut drafts = vec![*out.last().ok_or_else(|| invalid("empty"))?];
        drafts.extend(reference.iter().skip(at).take(2).copied());
        if at % 2 == 0 && drafts.len() == 3 {
            drafts[2] = (drafts[2] + 1) % 88 + 2;
        }
        let meta = runner.engine.metadata(h.request)?;
        let n = drafts.len();
        runner.engine.submit(PackedExecution {
            positions: (0..n).map(|i| meta.position + i).collect(),
            tokens: drafts,
            sequences: vec![SequenceDescriptor {
                state: meta.handle,
                tokens: 0..n,
                outputs: (0..n).collect(),
                speculative: true,
            }],
        })?;
        let commit = runner
            .engine
            .finish()?
            .pop()
            .ok_or_else(|| invalid("missing result"))?
            .outcome?;
        out.extend(commit.outputs.iter().map(|o| o.token));
    }
    assert_eq!(&out[..reference.len()], reference.as_slice());
    Ok(())
}

#[test]
fn test_vision_placeholders_fail_only_their_request() -> Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = load(&checkpoint.0, WeightFormat::F32, Device::Cpu)?;
    let mut runner = Runner::new(model)?;
    let good = runner
        .engine
        .allocate(0, SamplingOptions::default(), None)?;
    let bad = runner
        .engine
        .allocate(0, SamplingOptions::default(), None)?;
    let plan = PackedExecution {
        tokens: vec![5, 6, 95, 7],
        positions: vec![0, 1, 0, 1],
        sequences: vec![
            SequenceDescriptor {
                state: good,
                tokens: 0..2,
                outputs: vec![1],
                speculative: false,
            },
            SequenceDescriptor {
                state: bad,
                tokens: 2..4,
                outputs: vec![3],
                speculative: false,
            },
        ],
    };
    runner.engine.submit(plan)?;
    let results = runner.engine.finish()?;
    assert!(
        results
            .iter()
            .any(|r| r.request == good.request && r.outcome.is_ok())
    );
    assert!(
        results
            .iter()
            .any(|r| r.request == bad.request && r.outcome.is_err())
    );
    assert_eq!(runner.engine.metadata(bad.request)?.position, 0);
    Ok(())
}

#[test]
fn test_q8_projection_storage_tracks_f32() -> Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let reference = load(&checkpoint.0, WeightFormat::F32, Device::Cpu)?;
    let quantized = load(&checkpoint.0, WeightFormat::Q8_0, Device::Cpu)?;
    assert_ne!(
        reference.description().identity,
        quantized.description().identity
    );
    let tokens = prompt(40);
    let q = logits_by_chunks(&quantized, &tokens, 40)?;
    let f = logits_by_chunks(&reference, &tokens, 40)?;
    let argmax = |row: &Vec<f32>| {
        row.iter()
            .enumerate()
            .fold(
                (0, f32::MIN),
                |best, (i, &v)| if v > best.1 { (i, v) } else { best },
            )
            .0
    };
    let agree = q
        .iter()
        .zip(&f)
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();
    let error = max_relative(&q, &f);
    eprintln!(
        "Q8_0 tiny: relative logit error {error:.3e}; top-1 {agree}/{}",
        f.len()
    );
    // Random weights amplify per-element error through every layer; the decision matters.
    assert!(
        agree * 10 >= f.len() * 9,
        "Q8_0 top-1 agreement {agree}/{}",
        f.len()
    );
    assert!(error < 0.1, "Q8_0 relative logit error {error}");
    Ok(())
}

fn env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| TurError::Other(format!("set {name}")))
}

fn device() -> Result<Device> {
    match std::env::var("TUR_DEVICE").as_deref() {
        Ok("metal") => Ok(Device::new_metal(0)?),
        _ => Ok(Device::Cpu),
    }
}

fn i64_ids(tensor: &Tensor) -> Result<Vec<u32>> {
    tensor
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| invalid("id")))
        .collect()
}

struct Real {
    model: Arc<Qwen35>,
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    argmax: Vec<u32>,
    stops: Vec<u32>,
}

fn real(format: WeightFormat) -> Result<Real> {
    let dir = env_path("TUR_QWEN35_0_8B")?;
    let golden = candle_core::safetensors::load(env_path("TUR_QWEN35_GOLDEN")?, &Device::Cpu)?;
    let get = |n: &str| golden.get(n).ok_or_else(|| invalid(format!("golden {n}")));
    let model = Arc::new(Qwen35::load(
        &dir,
        &LoadOptions {
            revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
            weights: format.into(),
            device: device()?,
            execution: Execution::Fused,
            kv: EXACT_KV,
            recurrent: candle_core::DType::F32,
            mtp: false,
            draft_vocab: None,
        },
    )?);
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    let mut stops = model
        .config()
        .generation_defaults(generation.as_deref())?
        .eos_tokens;
    // The tokenizer's EOS (<|im_end|>) ends a chat turn.
    stops.push(248_046);
    Ok(Real {
        model,
        prompt: i64_ids(get("input_ids")?)?,
        greedy: i64_ids(get("greedy")?)?,
        argmax: i64_ids(get("argmax")?)?,
        stops,
    })
}

fn until_stop(tokens: &[u32], stops: &[u32]) -> Vec<u32> {
    let end = tokens
        .iter()
        .position(|t| stops.contains(t))
        .map_or(tokens.len(), |i| i + 1);
    tokens[..end].to_vec()
}

fn engine_generate(
    model: Arc<Qwen35>,
    prompt: &[u32],
    stops: &[u32],
    limit: usize,
) -> Result<Vec<u32>> {
    let device = model.device().clone();
    let mut engine = HybridEngine::new(model, device, POOL)?;
    let h = engine.allocate(0, SamplingOptions::default(), None)?;
    let mut out = Vec::new();
    let mut next = prompt.to_vec();
    while out.len() < limit {
        let meta = engine.metadata(h.request)?;
        let n = next.len();
        engine.submit(PackedExecution {
            positions: (0..n).map(|i| meta.position + i).collect(),
            tokens: std::mem::take(&mut next),
            sequences: vec![SequenceDescriptor {
                state: meta.handle,
                tokens: 0..n,
                outputs: vec![n - 1],
                speculative: false,
            }],
        })?;
        let token = engine
            .finish()?
            .pop()
            .ok_or_else(|| invalid("missing result"))?
            .outcome?
            .outputs
            .first()
            .map(|o| o.token)
            .ok_or_else(|| invalid("no output"))?;
        out.push(token);
        if stops.contains(&token) {
            break;
        }
        next = vec![token];
    }
    Ok(out)
}

#[test]
#[ignore = "needs TUR_QWEN35_0_8B and TUR_QWEN35_GOLDEN"]
fn test_f32_greedy_generation_matches_transformers() -> Result<()> {
    let Real {
        model,
        prompt,
        greedy,
        stops,
        ..
    } = real(WeightFormat::F32)?;
    let expected = until_stop(&greedy, &stops);
    let actual = engine_generate(model, &prompt, &stops, expected.len())?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
#[ignore = "needs TUR_QWEN35_0_8B and TUR_QWEN35_GOLDEN"]
fn test_quantized_generation_is_coherent() -> Result<()> {
    let dir = env_path("TUR_QWEN35_0_8B")?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    // Q8_0 has a predeclared 0.95 top-1 floor. Q4K has no quality budget yet (measured
    // 0.821 on 2026-10-22); this requires only coherent, terminating generation from it.
    for (format, floor) in [(WeightFormat::Q8_0, Some(0.95)), (WeightFormat::Q4K, None)] {
        let Real {
            model,
            prompt,
            greedy,
            argmax,
            stops,
        } = real(format)?;
        // Teacher-forced next-token agreement with the F32 reference over the prompt.
        let all: Vec<usize> = (0..prompt.len()).collect();
        let (_, logits) = model.trace(&prompt, &all)?;
        let top = logits
            .argmax(1)?
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let agree =
            top.iter().zip(&argmax).filter(|(a, b)| a == b).count() as f64 / argmax.len() as f64;
        let expected = until_stop(&greedy, &stops);
        let output = engine_generate(model, &prompt, &stops, 64)?;
        let prefix = output
            .iter()
            .zip(&expected)
            .take_while(|(a, b)| a == b)
            .count();
        let text = tokenizer
            .decode(&output, true)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?;
        eprintln!(
            "{format:?}: top-1 agreement {agree:.3}; greedy prefix {prefix}/{}; {text:?}",
            expected.len()
        );
        if let Some(floor) = floor {
            assert!(agree >= floor, "{format:?} agreement {agree}");
        }
        assert!(
            output.last().is_some_and(|t| stops.contains(t)),
            "{format:?} did not stop"
        );
        assert!(
            2 * prefix >= expected.len(),
            "{format:?} diverged after {prefix} tokens"
        );
    }
    Ok(())
}

/// The native MTP head against `scripts/qwen3_5_mtp_reference.py`: the backbone's final
/// hidden states of a greedy continuation, then the head over consecutive segments ending at
/// each golden row, so its own keys and values accumulate exactly as they do while serving.
#[test]
#[ignore = "needs TUR_QWEN35_0_8B and TUR_QWEN35_MTP_GOLDEN"]
fn test_mtp_head_matches_transformers_reference() -> Result<()> {
    use tur_models::hybrid::{
        Boundaries, DraftForward, DraftSequence, PackedForward, SequenceForward, StatePool,
    };
    let dir = env_path("TUR_QWEN35_0_8B")?;
    let golden = candle_core::safetensors::load(env_path("TUR_QWEN35_MTP_GOLDEN")?, &Device::Cpu)?;
    let get = |n: &str| golden.get(n).ok_or_else(|| invalid(format!("golden {n}")));
    let tokens = i64_ids(get("tokens")?)?;
    let rows: Vec<usize> = i64_ids(get("logit_rows")?)?
        .into_iter()
        .map(|r| r as usize)
        .collect();
    let expected = get("logits")?.to_vec2::<f32>()?;
    let argmax = i64_ids(get("argmax")?)?;
    let count = tokens.len() - 1;
    for execution in [Execution::Fused, Execution::Reference] {
        let model = Qwen35::load(
            &dir,
            &LoadOptions {
                revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
                weights: WeightFormat::F32.into(),
                device: device()?,
                execution,
                kv: EXACT_KV,
                recurrent: candle_core::DType::F32,
                mtp: true,
                draft_vocab: None,
            },
        )?;
        let description = model.description();
        let pool = StatePool::new(
            Arc::new(description.clone()),
            model.device(),
            PoolConfig {
                pages: description.pages_for(count),
                recurrent_slots: 2,
            },
        )?;
        let mut reservation = ModelState::new(&pool).reserve(count, Boundaries::Final)?;
        reservation.prepare()?;
        let binding = reservation.binding();
        let positions: Vec<usize> = (0..count).collect();
        let mut steps = model.forward(&PackedForward {
            tokens: &tokens[..count],
            positions: &positions,
            sequences: &[SequenceForward {
                tokens: 0..count,
                state: &binding,
                logits: &[],
                hidden: true,
            }],
            pool: &pool,
        })?;
        let hidden = steps
            .pop()
            .ok_or_else(|| invalid("missing step"))??
            .hidden
            .ok_or_else(|| invalid("missing hidden states"))?;
        let state = reservation.commit(count)?;
        drop(reservation);
        let (mut actual, mut start) = (Vec::new(), 0);
        for &row in &rows {
            let len = row + 1 - start;
            let side = state.side(start, len)?;
            let logits = model
                .draft(&DraftForward {
                    tokens: &tokens[start + 1..row + 2],
                    positions: &positions[start..=row],
                    sequences: &[DraftSequence {
                        tokens: 0..len,
                        state: &side,
                        hidden: &hidden.narrow(0, start, len)?,
                        logits: true,
                    }],
                    pool: &pool,
                })?
                .pop()
                .ok_or_else(|| invalid("missing draft"))??
                .ok_or_else(|| invalid("missing draft logits"))?;
            let row_logits = logits.to_vec2::<f32>()?.remove(0);
            let best = row_logits
                .iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0;
            assert_eq!(best as u32, argmax[row], "{execution:?} row {row} argmax");
            actual.push(row_logits);
            start = row + 1;
        }
        let error = max_relative(&actual, &expected);
        eprintln!(
            "MTP {execution:?}: max relative logit error {error:.2e} over {} rows",
            rows.len()
        );
        assert!(error < 1e-4, "{execution:?}: {error}");
    }
    Ok(())
}
