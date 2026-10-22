#![forbid(unsafe_code)]

use candle_core::{DType, Device, Tensor};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tur_core::{
    Result, TurError,
    execution::{
        KvFormat, KvLayout, LayerStateKind, Modality, ModelDescription, ModelIdentity,
        PackedExecution, SequenceDescriptor, StateHandle, StateMetadata, invalid,
    },
};
use tur_engine::{
    backend::guidance::{GuidanceControl, TopLevelGrammar, build_llg_factory},
    hybrid::{
        Boundaries, Checkpoint, HybridEngine, HybridModel, PackedForward, PoolConfig,
        SamplingOptions, SequenceCommit, SequenceForward, SequenceResult, SequenceStep, StatePool,
        TokenOutput,
    },
};
use tur_metal::state::LayerSnapshot;

const LOGITS: [f32; 8] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4];

#[derive(Default)]
struct Fixture {
    calls: AtomicUsize,
    boundaries: Mutex<Vec<Boundaries>>,
    revision: &'static str,
    /// Fails only the sequence containing this token.
    fault: Option<u32>,
    /// Fails the whole forward pass when any sequence contains this token.
    batch_fault: Option<u32>,
    /// Turns `fault` into malformed logits, after the sequence wrote its state.
    invalid_logits: bool,
}

fn fixture() -> Fixture {
    Fixture {
        revision: "fixture-v1",
        ..Default::default()
    }
}

fn attention() -> LayerStateKind {
    LayerStateKind::Attention {
        kv_heads: 1,
        head_dim: 1,
    }
}

/// Ample capacity: 8 pages of 2 tokens hold the 16-position limit; 8 recurrent slots.
const POOL: PoolConfig = PoolConfig {
    pages: 8,
    recurrent_slots: 8,
};

fn recurrent() -> LayerStateKind {
    LayerStateKind::Recurrent {
        conv_channels: 1,
        conv_width: 2,
        heads: 1,
        key_dim: 1,
        value_dim: 1,
        dtype: DType::F32,
    }
}

fn scalars(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

impl Fixture {
    /// Sequential reference math: each attention key adds the token, its position and all
    /// prior keys; recurrent state doubles and adds the token and previous token. State is
    /// written into the reserved pages and slots before any injected logits fault.
    fn sequence(
        &self,
        pool: &StatePool,
        tokens: &[u32],
        positions: &[usize],
        sequence: &SequenceForward<'_>,
    ) -> Result<SequenceStep> {
        let faulty = self.fault.is_some_and(|f| tokens.contains(&f));
        if faulty && !self.invalid_logits {
            return Err(invalid("injected sequence failure"));
        }
        let state = sequence.state;
        let addresses = state.resolve(pool)?;
        let n = tokens.len();
        for (index, kind) in self.description().layers.iter().enumerate() {
            match kind {
                LayerStateKind::Attention { .. } => {
                    let store = pool.attention(index)?;
                    let (past, _) = store.gather(&addresses.pages, state.past())?;
                    let mut sum: f32 = scalars(&past)?.iter().sum();
                    let mut keys = Vec::with_capacity(n);
                    for (&token, &position) in tokens.iter().zip(positions) {
                        let key = token as f32 + position as f32 + sum;
                        sum += key;
                        keys.push(key);
                    }
                    let values: Vec<f32> = keys.iter().map(|k| k + index as f32).collect();
                    store.scatter(
                        &addresses.pages,
                        state.past(),
                        &Tensor::from_vec(keys, (n, 1, 1), &Device::Cpu)?,
                        &Tensor::from_vec(values, (n, 1, 1), &Device::Cpu)?,
                    )?;
                }
                LayerStateKind::Recurrent { .. } => {
                    let store = pool.recurrent(index)?;
                    let (convolution, recurrent) = store.read(addresses.read)?;
                    let mut last = scalars(&convolution)?
                        .last()
                        .copied()
                        .ok_or_else(|| invalid("missing convolution history"))?;
                    let mut value = scalars(&recurrent)?
                        .first()
                        .copied()
                        .ok_or_else(|| invalid("missing recurrent state"))?;
                    let mut history = Vec::new();
                    let mut matrices = Vec::new();
                    for &token in tokens {
                        value = 2.0 * value + token as f32 + last;
                        history.extend([last, token as f32]);
                        matrices.push(value);
                        last = token as f32;
                    }
                    let keep = match state.boundaries() {
                        Boundaries::Final => 1,
                        Boundaries::EveryToken => n,
                    };
                    let history = Tensor::from_vec(
                        history.split_off(2 * (n - keep)),
                        (keep, 2, 1),
                        &Device::Cpu,
                    )?;
                    let matrices = Tensor::from_vec(
                        matrices.split_off(n - keep),
                        (keep, 1, 1, 1),
                        &Device::Cpu,
                    )?;
                    for (b, &slot) in addresses.writes.iter().enumerate() {
                        store.write(slot, &history, &matrices, b)?;
                    }
                }
            }
        }
        let rows = sequence.logits.len();
        let logits = if faulty {
            Tensor::new(&[1.0f32], &Device::Cpu)?
        } else {
            Tensor::from_vec(LOGITS.repeat(rows), (rows, LOGITS.len()), &Device::Cpu)?
        };
        Ok(SequenceStep { logits })
    }
}

impl HybridModel for Fixture {
    fn description(&self) -> ModelDescription {
        ModelDescription {
            identity: ModelIdentity {
                revision: self.revision.into(),
                state_format: 1,
            },
            layers: vec![attention(), recurrent(), attention(), recurrent()],
            kv: KvLayout::new(2, KvFormat::F32),
            modalities: vec![Modality::Text],
            vocab_size: 8,
            max_position: 16,
        }
    }

    fn workspace_bytes(&self, batch: &PackedForward<'_>) -> Result<usize> {
        Ok(batch.tokens.len() * 16)
    }

    fn forward(&self, batch: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.batch_fault.is_some_and(|f| batch.tokens.contains(&f)) {
            return Err(invalid("injected batch failure"));
        }
        let mut recorded = self.boundaries.lock();
        Ok(batch
            .sequences
            .iter()
            .map(|sequence| {
                recorded.push(sequence.state.boundaries());
                let range = sequence.tokens.clone();
                self.sequence(
                    batch.pool,
                    batch.tokens.get(range.clone()).unwrap_or_default(),
                    batch.positions.get(range).unwrap_or_default(),
                    sequence,
                )
            })
            .collect())
    }
}

fn engine(model: Fixture) -> Result<(HybridEngine<Fixture>, Arc<Fixture>)> {
    sized(model, POOL)
}

fn sized(model: Fixture, pool: PoolConfig) -> Result<(HybridEngine<Fixture>, Arc<Fixture>)> {
    let model = Arc::new(model);
    Ok((
        HybridEngine::new(Arc::clone(&model), Device::Cpu, pool)?,
        model,
    ))
}

fn allocate(engine: &mut HybridEngine<Fixture>, seed: u64) -> Result<StateHandle> {
    engine.allocate(
        seed,
        SamplingOptions {
            temperature: Some(0.8),
            top_p: Some(0.9),
            ..Default::default()
        },
        None,
    )
}

fn greedy(engine: &mut HybridEngine<Fixture>) -> Result<StateHandle> {
    engine.allocate(1, SamplingOptions::default(), None)
}

fn packed(
    engine: &HybridEngine<Fixture>,
    sequences: &[(StateHandle, &[u32])],
    speculative: bool,
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
        packed
            .positions
            .extend((0..tokens.len()).map(|offset| metadata.position + offset));
        let end = packed.tokens.len();
        packed.sequences.push(SequenceDescriptor {
            state: metadata.handle,
            tokens: start..end,
            outputs: (start..end).collect(),
            speculative,
        });
    }
    Ok(packed)
}

fn plan(
    engine: &HybridEngine<Fixture>,
    sequences: &[(StateHandle, &[u32])],
) -> Result<PackedExecution> {
    packed(engine, sequences, false)
}

fn submit_all(
    engine: &mut HybridEngine<Fixture>,
    input: PackedExecution,
) -> Result<Vec<SequenceResult>> {
    engine.submit(input)?;
    engine.finish()
}

fn commits(results: Vec<SequenceResult>) -> Result<Vec<SequenceCommit>> {
    results.into_iter().map(|r| r.outcome).collect()
}

fn run(
    engine: &mut HybridEngine<Fixture>,
    sequences: &[(StateHandle, &[u32])],
) -> Result<Vec<TokenOutput>> {
    let input = plan(engine, sequences)?;
    Ok(commits(submit_all(engine, input)?)?
        .into_iter()
        .flat_map(|c| c.outputs)
        .collect())
}

fn outcome(results: &[SequenceResult], handle: StateHandle) -> Result<&Result<SequenceCommit>> {
    results
        .iter()
        .find(|r| r.request == handle.request)
        .map(|r| &r.outcome)
        .ok_or_else(|| invalid("missing request result"))
}

fn tokens(outputs: &[TokenOutput]) -> Vec<u32> {
    outputs.iter().map(|o| o.token).collect()
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
        let (a, b) = match &layer {
            LayerSnapshot::Attention { key, value } => (key, value),
            LayerSnapshot::Recurrent {
                convolution,
                recurrent,
            } => (convolution, recurrent),
        };
        values.push(scalars(a)?);
        values.push(scalars(b)?);
    }
    Ok(values)
}

fn equivalent(left: &Checkpoint, right: &Checkpoint) -> Result<()> {
    assert_eq!(left.position(), right.position());
    assert_eq!(left.rng_draws(), right.rng_draws());
    assert_eq!(left.sampled_tokens(), right.sampled_tokens());
    assert_eq!(left.finished(), right.finished());
    assert_eq!(values(left)?, values(right)?);
    Ok(())
}

#[test]
fn test_independent_requests_match_single_execution_after_batch_reordering() -> Result<()> {
    let (mut batched, _) = engine(fixture())?;
    let (mut single, _) = engine(fixture())?;
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
fn test_packed_prefill_matches_incremental_decode() -> Result<()> {
    let (mut packed_engine, model) = engine(fixture())?;
    let (mut incremental, _) = engine(fixture())?;
    let a = allocate(&mut packed_engine, 5)?;
    let b = allocate(&mut incremental, 5)?;
    run(&mut packed_engine, &[(a, &[1, 2, 3, 4, 5])])?;
    assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    for token in 1..=5 {
        let current = handle(&incremental, b)?;
        run(&mut incremental, &[(current, &[token])])?;
    }
    equivalent(
        &checkpoint(&packed_engine, a)?,
        &checkpoint(&incremental, b)?,
    )
}

#[test]
fn test_checkpoint_fork_restore_crosses_partial_attention_pages_and_all_recurrent_layers()
-> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    let a = allocate(&mut engine, 1)?;
    run(&mut engine, &[(a, &[1, 2, 3])])?;
    let saved = checkpoint(&engine, a)?;
    let original = values(&saved)?;
    assert_eq!(
        original,
        vec![
            vec![1.0, 4.0, 10.0],
            vec![1.0, 4.0, 10.0],
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
        tokens(
            &first
                .into_iter()
                .filter(|o| o.request == a.request)
                .collect::<Vec<_>>()
        ),
        tokens(&replay)
    );
    Ok(())
}

#[test]
fn test_ordinary_execution_retains_only_final_recurrent_boundary() -> Result<()> {
    let (mut engine, model) = engine(fixture())?;
    let a = greedy(&mut engine)?;
    let ordinary = engine.submit(plan(&engine, &[(a, &[1, 2, 3])])?)?;
    engine.finish()?;
    let speculative = engine.submit(packed(&engine, &[(a, &[7, 7, 7])], true)?)?;
    engine.finish()?;
    assert_eq!(
        *model.boundaries.lock(),
        vec![Boundaries::Final, Boundaries::EveryToken]
    );
    // Pages hold 2 tokens of 8 K/V bytes in each of two attention layers (32 bytes); a
    // recurrent slot holds 12 bytes in each of two layers. Three new tokens open two pages
    // and write one slot; three speculative tokens after position 3 fill the exclusive
    // tail in place, open one page and write a slot per token.
    assert_eq!(ordinary.state_update_bytes, 2 * 32 + 24);
    assert_eq!(speculative.state_update_bytes, 32 + 3 * 24);
    assert_eq!(ordinary.logits_bytes, 3 * 8 * 4);
    assert_eq!(ordinary.workspace_bytes, 48);
    assert_eq!(
        ordinary.total_bytes()?,
        ordinary.packed_input_bytes + 88 + 96 + 48
    );
    Ok(())
}

#[test]
fn test_speculative_acceptance_follows_sampled_tokens() -> Result<()> {
    let (mut speculative, _) = engine(fixture())?;
    let (mut direct, _) = engine(fixture())?;
    let a = greedy(&mut speculative)?;
    let b = greedy(&mut direct)?;
    // Greedy target always samples 7: drafts 7, 7 are accepted, 3 is rejected.
    let input = packed(&speculative, &[(a, &[1, 7, 7, 3])], true)?;
    let commit = commits(submit_all(&mut speculative, input)?)?
        .pop()
        .ok_or_else(|| invalid("missing commit"))?;
    assert_eq!(commit.accepted_tokens, 3);
    assert_eq!(tokens(&commit.outputs), vec![7, 7, 7]);
    run(&mut direct, &[(b, &[1, 7, 7])])?;
    equivalent(&checkpoint(&speculative, a)?, &checkpoint(&direct, b)?)?;

    // Every draft matches: the final output is the bonus token.
    let input = packed(&speculative, &[(a, &[7, 7])], true)?;
    let commit = commits(submit_all(&mut speculative, input)?)?
        .pop()
        .ok_or_else(|| invalid("missing commit"))?;
    assert_eq!(commit.accepted_tokens, 2);
    assert_eq!(tokens(&commit.outputs), vec![7, 7]);
    let bh = handle(&direct, b)?;
    run(&mut direct, &[(bh, &[7, 7])])?;
    equivalent(&checkpoint(&speculative, a)?, &checkpoint(&direct, b)?)
}

#[test]
fn test_speculative_rejection_keeps_only_the_verified_token() -> Result<()> {
    let (mut speculative, _) = engine(fixture())?;
    let (mut direct, _) = engine(fixture())?;
    let a = greedy(&mut speculative)?;
    let b = greedy(&mut direct)?;
    let input = packed(&speculative, &[(a, &[1, 2, 3])], true)?;
    let commit = commits(submit_all(&mut speculative, input)?)?
        .pop()
        .ok_or_else(|| invalid("missing commit"))?;
    assert_eq!(commit.accepted_tokens, 1);
    assert_eq!(tokens(&commit.outputs), vec![7]);
    run(&mut direct, &[(b, &[1])])?;
    equivalent(&checkpoint(&speculative, a)?, &checkpoint(&direct, b)?)
}

#[test]
fn test_published_descriptors_are_invalidated() -> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    let a = allocate(&mut engine, 99)?;
    let input = plan(&engine, &[(a, &[1, 2])])?;
    submit_all(&mut engine, input.clone())?;
    assert!(engine.submit(input).is_err());
    Ok(())
}

#[test]
fn test_cancellation_retains_inflight_state_and_does_not_publish_outputs() -> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    let a = allocate(&mut engine, 11)?;
    let b = allocate(&mut engine, 12)?;
    let saved = checkpoint(&engine, b)?;
    let input = plan(&engine, &[(a, &[1, 2]), (b, &[3, 4])])?;
    engine.submit(input)?;
    assert!(engine.restore(a, &saved).is_err());
    engine.cancel(b)?;
    assert!(engine.metadata(b.request).is_err());
    let results = commits(engine.finish()?)?;
    assert_eq!(results.len(), 1);
    assert!(results.iter().all(|c| c.handle.request == a.request));
    assert_eq!(results.iter().map(|c| c.outputs.len()).sum::<usize>(), 2);
    assert_eq!(checkpoint(&engine, a)?.position(), 2);
    assert_eq!(saved.position(), 0);
    assert!(engine.metadata(b.request).is_err());
    let c = engine.fork(&saved)?;
    engine.cancel(c)?;
    assert!(engine.metadata(c.request).is_err());
    Ok(())
}

#[test]
fn test_one_request_failure_does_not_affect_others() -> Result<()> {
    let (mut engine, _) = engine(Fixture {
        fault: Some(7),
        ..fixture()
    })?;
    let a = allocate(&mut engine, 11)?;
    let b = allocate(&mut engine, 12)?;
    let before_b = checkpoint(&engine, b)?;
    let input = plan(&engine, &[(a, &[1, 2]), (b, &[3, 7])])?;
    let results = submit_all(&mut engine, input)?;
    let a_commit = outcome(&results, a)?
        .as_ref()
        .map_err(|e| invalid(e.to_string()))?;
    assert_eq!(a_commit.outputs.len(), 2);
    assert!(outcome(&results, b)?.is_err());
    assert_eq!(checkpoint(&engine, a)?.position(), 2);
    equivalent(&checkpoint(&engine, b)?, &before_b)?;
    let a = handle(&engine, a)?;
    run(&mut engine, &[(a, &[1]), (b, &[2])])?;
    Ok(())
}

#[test]
fn test_whole_batch_model_error_leaves_every_request_unchanged() -> Result<()> {
    let (mut engine, _) = engine(Fixture {
        batch_fault: Some(6),
        ..fixture()
    })?;
    let a = allocate(&mut engine, 11)?;
    let b = allocate(&mut engine, 12)?;
    let before_a = checkpoint(&engine, a)?;
    let before_b = checkpoint(&engine, b)?;
    let input = plan(&engine, &[(a, &[1, 2]), (b, &[3, 6])])?;
    engine.submit(input)?;
    assert!(engine.finish().is_err());
    assert!(engine.finish().is_err());
    equivalent(&checkpoint(&engine, a)?, &before_a)?;
    equivalent(&checkpoint(&engine, b)?, &before_b)?;
    run(&mut engine, &[(a, &[1]), (b, &[2])])?;
    Ok(())
}

#[test]
fn test_invalid_output_is_isolated_and_its_state_writes_are_never_published() -> Result<()> {
    let (mut engine, _) = engine(Fixture {
        fault: Some(6),
        invalid_logits: true,
        ..fixture()
    })?;
    let a = allocate(&mut engine, 11)?;
    let b = allocate(&mut engine, 12)?;
    run(&mut engine, &[(a, &[1])])?;
    let a = handle(&engine, a)?;
    let before = checkpoint(&engine, a)?;
    // `a` wrote keys into its exclusive tail page and a fresh slot, then failed.
    let input = plan(&engine, &[(a, &[1, 6]), (b, &[2])])?;
    let results = submit_all(&mut engine, input)?;
    assert!(outcome(&results, a)?.is_err());
    assert!(outcome(&results, b)?.is_ok());
    equivalent(&checkpoint(&engine, a)?, &before)?;
    let a = handle(&engine, a)?;
    run(&mut engine, &[(a, &[3])])?;
    let (mut control, _) = engine_control()?;
    let c = allocate(&mut control, 11)?;
    run(&mut control, &[(c, &[1])])?;
    let c = handle(&control, c)?;
    run(&mut control, &[(c, &[3])])?;
    equivalent(&checkpoint(&engine, a)?, &checkpoint(&control, c)?)
}

fn engine_control() -> Result<(HybridEngine<Fixture>, Arc<Fixture>)> {
    engine(fixture())
}

#[test]
fn test_pool_exhaustion_fails_the_pass_before_execution_and_leaves_state_unchanged() -> Result<()> {
    // Three pages of two tokens and four recurrent slots.
    let (mut engine, model) = sized(
        fixture(),
        PoolConfig {
            pages: 3,
            recurrent_slots: 4,
        },
    )?;
    let a = allocate(&mut engine, 1)?;
    let b = allocate(&mut engine, 2)?;
    run(&mut engine, &[(a, &[1, 2, 3, 4])])?;
    let a = handle(&engine, a)?;
    let before = checkpoint(&engine, a)?;
    let calls = model.calls.load(Ordering::SeqCst);
    // a needs one more page and b two: four pages exceed three.
    let input = plan(&engine, &[(a, &[5]), (b, &[1, 2, 3])])?;
    assert!(matches!(
        engine.submit(input),
        Err(TurError::MemoryAllocation(_))
    ));
    assert_eq!(model.calls.load(Ordering::SeqCst), calls);
    equivalent(&checkpoint(&engine, a)?, &before)?;
    assert_eq!(engine.metadata(b.request)?.position, 0);
    let usage = engine.memory()?;
    assert_eq!((usage.pages.leased, usage.pages.free()), (2, 1));
    // A retained checkpoint keeps a's pages leased after a is cancelled; releasing it
    // frees them for b once nothing is in flight.
    engine.cancel(a)?;
    assert_eq!(engine.memory()?.pages.leased, 2);
    drop(before);
    run(&mut engine, &[(b, &[1, 2, 3])])?;
    assert_eq!(engine.metadata(b.request)?.position, 3);
    Ok(())
}

#[test]
fn test_released_pages_and_slots_are_reclaimed_after_completion() -> Result<()> {
    let (mut engine, _) = sized(
        fixture(),
        PoolConfig {
            pages: 4,
            recurrent_slots: 3,
        },
    )?;
    // Far more passes than capacity: every cycle's pages and slots come back.
    for round in 0..20u32 {
        let a = allocate(&mut engine, u64::from(round))?;
        run(&mut engine, &[(a, &[1, 2, 3])])?;
        let saved = checkpoint(&engine, a)?;
        let b = engine.fork(&saved)?;
        let a = handle(&engine, a)?;
        run(&mut engine, &[(a, &[4])])?;
        // An in-flight cancel keeps its reservation until the fence, then releases it.
        engine.submit(plan(&engine, &[(b, &[5])])?)?;
        engine.cancel(b)?;
        assert!(engine.finish()?.is_empty());
        engine.cancel(handle(&engine, a)?)?;
        drop(saved);
        let usage = engine.memory()?;
        assert_eq!(
            (usage.pages.leased, usage.slots.leased),
            (0, 0),
            "round {round}"
        );
        assert_eq!(usage.pages.retired + usage.slots.retired, 0);
    }
    let usage = engine.memory()?;
    assert!(usage.pages.peak <= 4 && usage.slots.peak <= 3);
    Ok(())
}

#[test]
fn test_checkpoint_tensor_inspection_cannot_mutate_shared_state() -> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    let a = allocate(&mut engine, 1)?;
    run(&mut engine, &[(a, &[1, 2, 3])])?;
    let saved = checkpoint(&engine, a)?;
    let before = values(&saved)?;
    for layer in saved.model_state().read_layers()? {
        let (a, b) = match &layer {
            LayerSnapshot::Attention { key, value } => (key, value),
            LayerSnapshot::Recurrent {
                convolution,
                recurrent,
            } => (convolution, recurrent),
        };
        a.zero_set()?;
        b.zero_set()?;
    }
    assert_eq!(values(&saved)?, before);
    equivalent(&checkpoint(&engine, a)?, &saved)
}

#[test]
fn test_incompatible_restore_and_foreign_handles_leave_state_unchanged() -> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    let a = allocate(&mut engine, 1)?;
    let before = checkpoint(&engine, a)?;
    let (mut other, _) = self::engine(Fixture {
        revision: "different-model",
        ..Default::default()
    })?;
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
    let (mut engine, model) = engine(fixture())?;
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
    value.sequences[0].outputs = vec![1, 0];
    cases.push(value);
    let mut value = input.clone();
    value.sequences[0].speculative = true;
    value.sequences[0].outputs = vec![1];
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
    assert_eq!(model.calls.load(Ordering::SeqCst), 0);
    assert_eq!(checkpoint(&engine, a)?.position(), 0);
    Ok(())
}

#[test]
fn test_position_overflow_and_limit_are_checked() -> Result<()> {
    let (mut engine, model) = engine(fixture())?;
    let a = allocate(&mut engine, 1)?;
    let input = PackedExecution {
        tokens: vec![1],
        positions: vec![usize::MAX],
        sequences: vec![SequenceDescriptor {
            state: a,
            tokens: 0..1,
            outputs: vec![0],
            speculative: false,
        }],
    };
    assert!(
        input
            .validate(
                &model.description(),
                &[StateMetadata {
                    handle: a,
                    position: usize::MAX,
                    finished: false,
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
fn test_grammar_parser_fork_speculation_restore_and_terminal_state_are_atomic() -> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    let a = engine.allocate(42, SamplingOptions::default(), Some(grammar()?))?;
    assert_eq!(tokens(&run(&mut engine, &[(a, &[1])])?), vec![1]);
    let saved = checkpoint(&engine, a)?;
    let b = engine.fork(&saved)?;
    // The grammar forces 'b' (2), so the draft 3 is rejected and never reaches the parser.
    let input = packed(&engine, &[(a, &[2, 3])], true)?;
    let commit = commits(submit_all(&mut engine, input)?)?
        .pop()
        .ok_or_else(|| invalid("missing commit"))?;
    assert_eq!(commit.accepted_tokens, 1);
    assert_eq!(tokens(&commit.outputs), vec![2]);
    let a = handle(&engine, a)?;
    assert_eq!(tokens(&run(&mut engine, &[(a, &[3])])?), vec![3]);
    assert_eq!(tokens(&run(&mut engine, &[(b, &[2, 3])])?), vec![2, 3]);
    equivalent(&checkpoint(&engine, a)?, &checkpoint(&engine, b)?)?;
    let a = handle(&engine, a)?;
    let input = plan(&engine, &[(a, &[4])])?;
    let terminal = commits(submit_all(&mut engine, input)?)?;
    assert!(terminal.iter().all(|c| c.finished && c.outputs.is_empty()));
    assert!(engine.metadata(a.request)?.finished);
    let a = handle(&engine, a)?;
    let restored = engine.restore(a, &saved)?;
    assert!(!engine.metadata(restored.request)?.finished);
    assert_eq!(
        tokens(&run(&mut engine, &[(restored, &[2, 3])])?),
        vec![2, 3]
    );
    Ok(())
}

#[test]
fn test_terminal_grammar_does_not_fail_other_requests() -> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    let a = engine.allocate(42, SamplingOptions::default(), Some(grammar()?))?;
    run(&mut engine, &[(a, &[1])])?;
    let a = handle(&engine, a)?;
    run(&mut engine, &[(a, &[2, 3])])?;
    let a = handle(&engine, a)?;
    let b = greedy(&mut engine)?;
    let input = plan(&engine, &[(b, &[1]), (a, &[4])])?;
    let results = commits(submit_all(&mut engine, input)?)?;
    assert_eq!(results.len(), 2);
    let (finished, open): (Vec<_>, Vec<_>) = results.into_iter().partition(|c| c.finished);
    assert!(finished.iter().all(|c| c.outputs.is_empty()));
    assert_eq!(
        open.iter()
            .flat_map(|c| tokens(&c.outputs))
            .collect::<Vec<_>>(),
        vec![7]
    );
    assert_eq!(engine.metadata(b.request)?.position, 1);
    Ok(())
}

#[test]
fn test_no_output_positions_advance_only_model_state() -> Result<()> {
    let (mut engine, _) = engine(fixture())?;
    assert!(grammar()?.validate_token_contract(7).is_err());
    let a = engine.allocate(42, SamplingOptions::default(), Some(grammar()?))?;
    let mut input = plan(&engine, &[(a, &[1, 2])])?;
    input
        .sequences
        .first_mut()
        .ok_or_else(|| invalid("missing descriptor"))?
        .outputs
        .clear();
    let results = commits(submit_all(&mut engine, input)?)?;
    assert!(results.iter().all(|c| c.outputs.is_empty()));
    let saved = checkpoint(&engine, a)?;
    assert_eq!(saved.position(), 2);
    assert_eq!(saved.rng_draws(), 0);
    assert!(saved.sampled_tokens().is_empty());
    let a = handle(&engine, a)?;
    assert_eq!(tokens(&run(&mut engine, &[(a, &[3])])?), vec![1]);
    Ok(())
}

#[test]
fn test_cancellation_holds_shared_model_until_completion() -> Result<()> {
    let (mut engine, model) = engine(fixture())?;
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
