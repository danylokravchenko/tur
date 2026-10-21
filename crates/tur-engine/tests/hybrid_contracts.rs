#![forbid(unsafe_code)]

use candle_core::{Device, Tensor};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tur_core::{
    Result, TurError,
    execution::{
        LayerStateKind, ModelDescription, ModelIdentity, PackedExecution, SequenceDescriptor,
        StateHandle, StateMetadata, invalid,
    },
};
use tur_engine::{
    backend::guidance::{GuidanceControl, TopLevelGrammar, build_llg_factory},
    hybrid::{Checkpoint, HybridEngine, SamplingOptions, TokenOutput},
};
use tur_metal::state::{LayerRead, LayerUpdate};
use tur_models::hybrid::{HybridModel, TokenStep};

struct Fixture {
    calls: Arc<AtomicUsize>,
    revision: &'static str,
    fault: Option<u32>,
    invalid_update: bool,
}

impl HybridModel for Fixture {
    fn description(&self) -> ModelDescription {
        ModelDescription {
            identity: ModelIdentity {
                revision: self.revision.into(),
                state_format: 1,
            },
            layers: vec![
                LayerStateKind::Attention {
                    kv_heads: 1,
                    head_dim: 1,
                    page_tokens: 2,
                },
                LayerStateKind::Recurrent {
                    conv_channels: 1,
                    conv_width: 2,
                    heads: 1,
                    key_dim: 1,
                    value_dim: 1,
                },
                LayerStateKind::Attention {
                    kv_heads: 1,
                    head_dim: 1,
                    page_tokens: 3,
                },
                LayerStateKind::Recurrent {
                    conv_channels: 1,
                    conv_width: 2,
                    heads: 1,
                    key_dim: 1,
                    value_dim: 1,
                },
            ],
            vocab_size: 8,
            max_position: 16,
        }
    }

    fn step(&self, token: u32, position: usize, state: &[LayerRead]) -> Result<TokenStep> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fault == Some(token) && !self.invalid_update {
            return Err(invalid("injected execution failure"));
        }
        let mut layers = Vec::new();
        for (index, layer) in state.iter().enumerate() {
            layers.push(match layer {
                LayerRead::Attention { pages } => {
                    let mut sum = token as f32 + position as f32;
                    for (key, _) in pages {
                        sum += key.flatten_all()?.to_vec1::<f32>()?.iter().sum::<f32>();
                    }
                    LayerUpdate::Attention {
                        key: Tensor::new(&[[sum]], &Device::Cpu)?,
                        value: Tensor::new(&[[sum + index as f32]], &Device::Cpu)?,
                    }
                }
                LayerRead::Recurrent {
                    convolution,
                    recurrent,
                } => {
                    let history = convolution.flatten_all()?.to_vec1::<f32>()?;
                    let last = history
                        .last()
                        .copied()
                        .ok_or_else(|| invalid("missing convolution history"))?;
                    let convolution = Tensor::new(&[[last], [token as f32]], &Device::Cpu)?;
                    let recurrent = if self.invalid_update && self.fault == Some(token) {
                        Tensor::new(&[1.0f32], &Device::Cpu)?
                    } else {
                        recurrent.affine(2.0, f64::from(token) + f64::from(last))?
                    };
                    LayerUpdate::Recurrent {
                        convolution,
                        recurrent,
                    }
                }
            });
        }
        let logits = Tensor::new(&[0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], &Device::Cpu)?;
        Ok(TokenStep { layers, logits })
    }
}

fn engine(
    fault: Option<u32>,
    invalid_update: bool,
) -> Result<(HybridEngine<Fixture>, Arc<AtomicUsize>)> {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(Fixture {
        calls: Arc::clone(&calls),
        revision: "fixture-v1",
        fault,
        invalid_update,
    });
    Ok((HybridEngine::new(model, Device::Cpu)?, calls))
}

fn allocate(engine: &mut HybridEngine<Fixture>, seed: u64) -> Result<StateHandle> {
    engine.allocate(
        seed,
        SamplingOptions {
            temperature: Some(0.8),
            top_p: Some(0.9),
        },
        None,
    )
}

fn plan(
    engine: &HybridEngine<Fixture>,
    sequences: &[(StateHandle, &[u32])],
) -> Result<PackedExecution> {
    let mut packed = PackedExecution {
        tokens: Vec::new(),
        positions: Vec::new(),
        sequences: Vec::new(),
    };
    for &(handle, tokens) in sequences {
        let metadata = engine.metadata(handle.request)?;
        let start = packed.tokens.len();
        packed.tokens.extend_from_slice(tokens);
        for offset in 0..tokens.len() {
            packed.positions.push(
                metadata
                    .position
                    .checked_add(offset)
                    .ok_or_else(|| invalid("fixture position overflow"))?,
            );
        }
        let end = packed.tokens.len();
        packed.sequences.push(SequenceDescriptor {
            state: handle,
            tokens: start..end,
            outputs: (start..end).collect(),
        });
    }
    Ok(packed)
}

fn run(
    engine: &mut HybridEngine<Fixture>,
    sequences: &[(StateHandle, &[u32])],
) -> Result<Vec<TokenOutput>> {
    let input = plan(engine, sequences)?;
    engine.submit(input)?;
    engine.finish()
}

fn checkpoint(engine: &HybridEngine<Fixture>, handle: StateHandle) -> Result<Checkpoint> {
    engine.checkpoint(engine.metadata(handle.request)?.handle)
}

fn handle(engine: &HybridEngine<Fixture>, original: StateHandle) -> Result<StateHandle> {
    Ok(engine.metadata(original.request)?.handle)
}

fn values(checkpoint: &Checkpoint) -> Result<Vec<Vec<f32>>> {
    let mut values = Vec::new();
    for layer in checkpoint.model_state().read_layers()? {
        match layer {
            LayerRead::Attention { pages } => {
                for (key, value) in pages {
                    values.push(key.flatten_all()?.to_vec1::<f32>()?);
                    values.push(value.flatten_all()?.to_vec1::<f32>()?);
                }
            }
            LayerRead::Recurrent {
                convolution,
                recurrent,
            } => {
                values.push(convolution.flatten_all()?.to_vec1::<f32>()?);
                values.push(recurrent.flatten_all()?.to_vec1::<f32>()?);
            }
        }
    }
    Ok(values)
}

fn equivalent(left: &Checkpoint, right: &Checkpoint) -> Result<()> {
    assert_eq!(left.position(), right.position());
    assert_eq!(left.rng_draws(), right.rng_draws());
    assert_eq!(left.sampled_tokens(), right.sampled_tokens());
    assert_eq!(values(left)?, values(right)?);
    Ok(())
}

#[test]
fn test_independent_requests_match_single_execution_after_batch_reordering() -> Result<()> {
    let (mut batched, _) = engine(None, false)?;
    let (mut single, _) = engine(None, false)?;
    let a = allocate(&mut batched, 123)?;
    let b = allocate(&mut batched, 456)?;
    let control_a = allocate(&mut single, 123)?;
    let control_b = allocate(&mut single, 456)?;
    run(&mut batched, &[(b, &[4, 3]), (a, &[1, 2, 3])])?;
    run(&mut single, &[(control_a, &[1, 2, 3])])?;
    run(&mut single, &[(control_b, &[4, 3])])?;
    equivalent(&checkpoint(&batched, a)?, &checkpoint(&single, control_a)?)?;
    equivalent(&checkpoint(&batched, b)?, &checkpoint(&single, control_b)?)?;
    let ah = handle(&batched, a)?;
    let bh = handle(&batched, b)?;
    run(&mut batched, &[(ah, &[2]), (bh, &[5])])?;
    let ca = handle(&single, control_a)?;
    let cb = handle(&single, control_b)?;
    run(&mut single, &[(cb, &[5]), (ca, &[2])])?;
    equivalent(&checkpoint(&batched, a)?, &checkpoint(&single, control_a)?)?;
    equivalent(&checkpoint(&batched, b)?, &checkpoint(&single, control_b)?)
}

#[test]
fn test_checkpoint_fork_restore_crosses_partial_attention_pages_and_all_recurrent_layers()
-> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    let a = allocate(&mut engine, 1)?;
    run(&mut engine, &[(a, &[1, 2, 3])])?;
    let saved = checkpoint(&engine, a)?;
    let original = values(&saved)?;
    assert_eq!(
        original,
        vec![
            vec![1.0, 4.0],
            vec![1.0, 4.0],
            vec![10.0],
            vec![10.0],
            vec![2.0, 3.0],
            vec![15.0],
            vec![1.0, 4.0, 10.0],
            vec![3.0, 6.0, 12.0],
            vec![2.0, 3.0],
            vec![15.0],
        ]
    );
    let b = engine.fork(&saved)?;
    let a = handle(&engine, a)?;
    let first = run(&mut engine, &[(a, &[4, 5]), (b, &[6])])?;
    assert_eq!(values(&saved)?, original);
    assert_ne!(
        values(&checkpoint(&engine, a)?)?,
        values(&checkpoint(&engine, b)?)?
    );
    let a = handle(&engine, a)?;
    let restored = engine.restore(a, &saved)?;
    equivalent(&checkpoint(&engine, restored)?, &saved)?;
    let replay = run(&mut engine, &[(restored, &[4, 5])])?;
    assert_eq!(
        first
            .iter()
            .filter(|o| o.request == a.request)
            .map(|o| o.token)
            .collect::<Vec<_>>(),
        replay.iter().map(|o| o.token).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn test_speculative_prefix_commit_restores_rng_and_every_layer() -> Result<()> {
    let (mut speculative, _) = engine(None, false)?;
    let (mut direct, _) = engine(None, false)?;
    let a = allocate(&mut speculative, 777)?;
    let b = allocate(&mut direct, 777)?;
    let input = plan(&speculative, &[(a, &[1, 2, 3, 4])])?;
    speculative.submit(input)?;
    assert_eq!(checkpoint(&speculative, a)?.position(), 0);
    let outputs = speculative.commit(&[(a.request, 1)])?;
    assert_eq!(outputs.len(), 1);
    run(&mut direct, &[(b, &[1])])?;
    equivalent(&checkpoint(&speculative, a)?, &checkpoint(&direct, b)?)?;
    let ah = handle(&speculative, a)?;
    let bh = handle(&direct, b)?;
    run(&mut speculative, &[(ah, &[2, 3, 4])])?;
    run(&mut direct, &[(bh, &[2, 3, 4])])?;
    equivalent(&checkpoint(&speculative, a)?, &checkpoint(&direct, b)?)?;
    assert!(speculative.finish().is_err());
    Ok(())
}

#[test]
fn test_zero_acceptance_rolls_back_and_invalidates_old_descriptors() -> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    let a = allocate(&mut engine, 99)?;
    let before = checkpoint(&engine, a)?;
    let input = plan(&engine, &[(a, &[1, 2])])?;
    engine.submit(input.clone())?;
    assert!(engine.commit(&[(a.request, 0)])?.is_empty());
    equivalent(&checkpoint(&engine, a)?, &before)?;
    assert!(engine.submit(input).is_err());
    Ok(())
}

#[test]
fn test_invalid_acceptance_keeps_pending_state_for_correction() -> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    let a = allocate(&mut engine, 99)?;
    let input = plan(&engine, &[(a, &[1, 2])])?;
    engine.submit(input)?;
    assert!(engine.commit(&[]).is_err());
    assert!(engine.commit(&[(a.request, 3)]).is_err());
    assert!(engine.commit(&[(a.request, 1), (a.request, 1)]).is_err());
    assert_eq!(checkpoint(&engine, a)?.position(), 0);
    assert_eq!(engine.commit(&[(a.request, 1)])?.len(), 1);
    assert_eq!(checkpoint(&engine, a)?.position(), 1);
    Ok(())
}

#[test]
fn test_cancellation_retains_inflight_state_and_does_not_publish_outputs() -> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    let a = allocate(&mut engine, 11)?;
    let b = allocate(&mut engine, 12)?;
    let saved = checkpoint(&engine, b)?;
    let input = plan(&engine, &[(a, &[1, 2]), (b, &[3, 4])])?;
    engine.submit(input)?;
    assert!(engine.restore(a, &saved).is_err());
    engine.cancel(b)?;
    assert!(engine.metadata(b.request).is_err());
    let output = engine.finish()?;
    assert_eq!(output.len(), 2);
    assert!(output.iter().all(|o| o.request == a.request));
    assert_eq!(checkpoint(&engine, a)?.position(), 2);
    assert_eq!(saved.position(), 0);
    assert!(engine.metadata(b.request).is_err());
    let c = engine.fork(&saved)?;
    engine.cancel(c)?;
    assert!(engine.metadata(c.request).is_err());
    Ok(())
}

#[test]
fn test_execution_error_discards_all_candidates_and_allows_retry() -> Result<()> {
    let (mut engine, _) = engine(Some(7), false)?;
    let a = allocate(&mut engine, 11)?;
    let b = allocate(&mut engine, 12)?;
    let before_a = checkpoint(&engine, a)?;
    let before_b = checkpoint(&engine, b)?;
    let input = plan(&engine, &[(a, &[1, 2]), (b, &[3, 7])])?;
    engine.submit(input)?;
    assert!(engine.finish().is_err());
    equivalent(&checkpoint(&engine, a)?, &before_a)?;
    equivalent(&checkpoint(&engine, b)?, &before_b)?;
    run(&mut engine, &[(a, &[1]), (b, &[2])])?;
    Ok(())
}

#[test]
fn test_invalid_layer_update_never_publishes_partial_state() -> Result<()> {
    let (mut engine, _) = engine(Some(6), true)?;
    let a = allocate(&mut engine, 11)?;
    let before = checkpoint(&engine, a)?;
    let input = plan(&engine, &[(a, &[1, 6])])?;
    engine.submit(input)?;
    assert!(engine.finish().is_err());
    equivalent(&checkpoint(&engine, a)?, &before)
}

#[test]
fn test_checkpoint_tensor_inspection_cannot_mutate_shared_state() -> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    let a = allocate(&mut engine, 1)?;
    run(&mut engine, &[(a, &[1, 2, 3])])?;
    let saved = checkpoint(&engine, a)?;
    let before = values(&saved)?;
    for layer in saved.model_state().read_layers()? {
        match layer {
            LayerRead::Attention { pages } => {
                for (key, value) in pages {
                    key.zero_set()?;
                    value.zero_set()?;
                }
            }
            LayerRead::Recurrent {
                convolution,
                recurrent,
            } => {
                convolution.zero_set()?;
                recurrent.zero_set()?;
            }
        }
    }
    assert_eq!(values(&saved)?, before);
    equivalent(&checkpoint(&engine, a)?, &saved)
}

#[test]
fn test_incompatible_restore_and_foreign_handles_leave_state_unchanged() -> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    let a = allocate(&mut engine, 1)?;
    let before = checkpoint(&engine, a)?;
    let other_model = Arc::new(Fixture {
        calls: Arc::new(AtomicUsize::new(0)),
        revision: "different-model",
        fault: None,
        invalid_update: false,
    });
    let mut other = HybridEngine::new(other_model, Device::Cpu)?;
    let foreign = allocate(&mut other, 1)?;
    let saved = checkpoint(&other, foreign)?;
    assert!(engine.restore(a, &saved).is_err());
    assert!(engine.fork(&saved).is_err());
    let mut input = plan(&engine, &[(a, &[1])])?;
    input
        .sequences
        .first_mut()
        .ok_or_else(|| invalid("missing fixture descriptor"))?
        .state = foreign;
    assert!(engine.submit(input).is_err());
    equivalent(&checkpoint(&engine, a)?, &before)
}

#[test]
fn test_invalid_descriptors_are_rejected_before_model_execution() -> Result<()> {
    let (mut engine, calls) = engine(None, false)?;
    let a = allocate(&mut engine, 1)?;
    let b = allocate(&mut engine, 2)?;
    let input = plan(&engine, &[(a, &[1, 2]), (b, &[3])])?;
    let mut cases = Vec::new();
    let mut value = input.clone();
    value.positions.pop();
    cases.push(value);
    let mut value = input.clone();
    value.tokens[0] = 8;
    cases.push(value);
    let mut value = input.clone();
    value.positions[1] = 4;
    cases.push(value);
    let mut value = input.clone();
    value.sequences[1].state = a;
    cases.push(value);
    let mut value = input.clone();
    value.sequences[0].tokens = 0..usize::MAX;
    cases.push(value);
    let mut value = input.clone();
    value.sequences[0].tokens = 1..2;
    cases.push(value);
    let mut value = input.clone();
    value.sequences[0].outputs = vec![2];
    cases.push(value);
    let mut value = input.clone();
    value.sequences[0].outputs = vec![0, 0];
    cases.push(value);
    let mut value = input.clone();
    value.sequences[0].state.revision = 1;
    cases.push(value);
    let mut value = input.clone();
    value.sequences.pop();
    cases.push(value);
    let mut value = input.clone();
    value.sequences[0].tokens = 0..0;
    cases.push(value);
    let mut value = input.clone();
    value.tokens.clear();
    cases.push(value);
    for invalid in cases {
        assert!(engine.submit(invalid).is_err());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(checkpoint(&engine, a)?.position(), 0);
    Ok(())
}

#[test]
fn test_position_overflow_and_limit_are_checked() -> Result<()> {
    let (mut engine, calls) = engine(None, false)?;
    let a = allocate(&mut engine, 1)?;
    let model = Fixture {
        calls,
        revision: "fixture-v1",
        fault: None,
        invalid_update: false,
    }
    .description();
    let input = PackedExecution {
        tokens: vec![1],
        positions: vec![usize::MAX],
        sequences: vec![SequenceDescriptor {
            state: a,
            tokens: 0..1,
            outputs: vec![0],
        }],
    };
    assert!(
        input
            .validate(
                &model,
                &[StateMetadata {
                    handle: a,
                    position: usize::MAX
                }]
            )
            .is_err()
    );
    let tokens = [1u32; 17];
    let input = plan(&engine, &[(a, &tokens)])?;
    assert!(engine.submit(input).is_err());
    Ok(())
}

fn grammar() -> Result<GuidanceControl> {
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
        "pre_tokenizer": null, "post_processor": null,
        "decoder": {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":false, "use_regex":false},
        "added_tokens": [{"id":0,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "model": {"type":"BPE", "merges":[], "vocab":{"<eos>":0,"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7},"unk_token":"<eos>"}
    });
    let tokenizer = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&tokenizer)?)
        .map_err(|error| TurError::Tokenizer(error.to_string()))?;
    let mut guidance = GuidanceControl::new(build_llg_factory(tokenizer, Some(8))?);
    guidance.activate(TopLevelGrammar::from_regex("abc"))?;
    Ok(guidance)
}

#[test]
fn test_grammar_parser_fork_rollback_restore_and_terminal_state_are_atomic() -> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    let a = engine.allocate(42, SamplingOptions::default(), Some(grammar()?))?;
    assert_eq!(
        run(&mut engine, &[(a, &[1])])?.first().map(|o| o.token),
        Some(1)
    );
    let saved = checkpoint(&engine, a)?;
    let b = engine.fork(&saved)?;
    let a = handle(&engine, a)?;
    let input = plan(&engine, &[(a, &[2, 3])])?;
    engine.submit(input)?;
    assert_eq!(
        engine.commit(&[(a.request, 1)])?.first().map(|o| o.token),
        Some(2)
    );
    let a = handle(&engine, a)?;
    assert_eq!(
        run(&mut engine, &[(a, &[3])])?.first().map(|o| o.token),
        Some(3)
    );
    let output = run(&mut engine, &[(b, &[2, 3])])?;
    assert_eq!(
        output.iter().map(|o| o.token).collect::<Vec<_>>(),
        vec![2, 3]
    );
    equivalent(&checkpoint(&engine, a)?, &checkpoint(&engine, b)?)?;
    let terminal = checkpoint(&engine, a)?;
    let a = handle(&engine, a)?;
    assert!(run(&mut engine, &[(a, &[4])]).is_err());
    equivalent(&checkpoint(&engine, a)?, &terminal)?;
    let restored = engine.restore(a, &saved)?;
    assert_eq!(
        run(&mut engine, &[(restored, &[2, 3])])?
            .iter()
            .map(|o| o.token)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    Ok(())
}

#[test]
fn test_no_output_positions_advance_only_model_state() -> Result<()> {
    let (mut engine, _) = engine(None, false)?;
    assert!(grammar()?.validate_token_contract(7).is_err());
    let a = engine.allocate(42, SamplingOptions::default(), Some(grammar()?))?;
    let mut input = plan(&engine, &[(a, &[1, 2])])?;
    input
        .sequences
        .first_mut()
        .ok_or_else(|| invalid("missing descriptor"))?
        .outputs
        .clear();
    engine.submit(input)?;
    assert!(engine.finish()?.is_empty());
    let saved = checkpoint(&engine, a)?;
    assert_eq!(saved.position(), 2);
    assert_eq!(saved.rng_draws(), 0);
    assert!(saved.sampled_tokens().is_empty());
    let a = handle(&engine, a)?;
    assert_eq!(
        run(&mut engine, &[(a, &[3])])?.first().map(|o| o.token),
        Some(1)
    );
    Ok(())
}

#[test]
fn test_cancellation_holds_shared_model_until_completion() -> Result<()> {
    let model = Arc::new(Fixture {
        calls: Arc::new(AtomicUsize::new(0)),
        revision: "fixture-v1",
        fault: None,
        invalid_update: false,
    });
    let mut engine = HybridEngine::new(Arc::clone(&model), Device::Cpu)?;
    let a = allocate(&mut engine, 1)?;
    assert_eq!(Arc::strong_count(&model), 2);
    let input = plan(&engine, &[(a, &[1, 2])])?;
    engine.submit(input)?;
    assert_eq!(Arc::strong_count(&model), 3);
    engine.cancel(a)?;
    assert_eq!(Arc::strong_count(&model), 3);
    assert!(engine.finish()?.is_empty());
    assert_eq!(Arc::strong_count(&model), 2);
    drop(engine);
    assert_eq!(Arc::strong_count(&model), 1);
    Ok(())
}
