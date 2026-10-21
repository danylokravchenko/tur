#![forbid(unsafe_code)]
//! Single-owner hybrid request execution with completion-gated publication.

use crate::backend::guidance::GuidanceControl;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tur_core::{
    Result, TurError,
    execution::{
        ExecutionRequirements, ModelDescription, PackedExecution, SequenceDescriptor, StateHandle,
        StateMetadata, invalid,
    },
};
use tur_metal::{
    completion::{Completion, Quarantine},
    state::{ModelState, StagedUpdate},
};
pub use tur_models::hybrid::{
    Boundaries, HybridModel, LayerRead, PackedForward, SequenceForward, SequenceStep,
};
use uuid::Uuid;

/// Per-request sampling policy using the existing Candle sampler.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

struct SamplingState {
    seed: u64,
    draws: u64,
    options: SamplingOptions,
    guidance: Option<GuidanceControl>,
    tokens: Vec<u32>,
    finished: bool,
}

/// Bookkeeping needed to undo one request's sampling after a mid-sequence error.
struct SamplingMark {
    draws: u64,
    tokens: usize,
    finished: bool,
    guidance: Option<GuidanceControl>,
}

impl SamplingState {
    fn fork(&self) -> Self {
        Self {
            seed: self.seed,
            draws: self.draws,
            options: self.options,
            guidance: self.guidance.as_ref().map(GuidanceControl::fork),
            tokens: self.tokens.clone(),
            finished: self.finished,
        }
    }

    /// Only an active grammar needs a parser copy; history rolls back by truncation.
    fn mark(&self) -> SamplingMark {
        SamplingMark {
            draws: self.draws,
            tokens: self.tokens.len(),
            finished: self.finished,
            guidance: self
                .guidance
                .as_ref()
                .filter(|g| g.is_active())
                .map(GuidanceControl::fork),
        }
    }

    fn rollback(&mut self, mark: SamplingMark) {
        self.draws = mark.draws;
        self.tokens.truncate(mark.tokens);
        self.finished = mark.finished;
        if let Some(guidance) = mark.guidance {
            self.guidance = Some(guidance);
        }
    }

    /// Sample one output, or `None` once the grammar has reached a terminal state.
    fn sample(&mut self, logits: &[f32]) -> Result<Option<u32>> {
        if self.finished {
            return Ok(None);
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(invalid("nonfinite model logits"));
        }
        let next_draw = self
            .draws
            .checked_add(1)
            .ok_or_else(|| invalid("RNG counter overflow"))?;
        // SplitMix64's specified modular arithmetic derives a stable per-draw stream.
        // The request counter itself is checked and never wraps or depends on batch order.
        let mut seed = self
            .seed
            .wrapping_add(self.draws.wrapping_mul(0x9e3779b97f4a7c15));
        seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d049bb133111eb);
        seed ^= seed >> 31;
        let mut values = logits.to_vec();
        if let Some(guidance) = &mut self.guidance
            && guidance.apply_mask(&mut values)?
        {
            self.finished = true;
            return Ok(None);
        }
        if !values.iter().any(|v| v.is_finite()) {
            return Err(TurError::Guidance("grammar disallows every token".into()));
        }
        let vocab = values.len();
        let masked = Tensor::from_vec(values, vocab, &Device::Cpu)?;
        let mut sampler = LogitsProcessor::new(seed, self.options.temperature, self.options.top_p);
        let token = sampler.sample(&masked)?;
        if let Some(guidance) = &mut self.guidance
            && guidance.commit(token)?
        {
            self.finished = true;
        }
        self.draws = next_draw;
        self.tokens.push(token);
        Ok(Some(token))
    }
}

struct Snapshot {
    model: ModelState,
    sampling: SamplingState,
}

impl Snapshot {
    fn fork(&self) -> Self {
        Self {
            model: self.model.clone(),
            sampling: self.sampling.fork(),
        }
    }
}

/// Opaque atomic snapshot of all layers, sequence position, RNG and grammar parser.
/// Cloning uses independent parser state and shares private immutable model buffers.
pub struct Checkpoint(Snapshot);
impl Clone for Checkpoint {
    fn clone(&self) -> Self {
        Self(self.0.fork())
    }
}
impl Checkpoint {
    /// Exact committed position of this checkpoint.
    pub fn position(&self) -> usize {
        self.0.model.position()
    }
    /// Inspect detached layer tensors through the immutable model-state API.
    pub fn model_state(&self) -> &ModelState {
        &self.0.model
    }
    /// Number of committed sampling draws.
    pub fn rng_draws(&self) -> u64 {
        self.0.sampling.draws
    }
    /// Sampled token history, including grammar-committed tokens.
    pub fn sampled_tokens(&self) -> &[u32] {
        &self.0.sampling.tokens
    }
    /// Whether the grammar had reached a terminal state.
    pub fn finished(&self) -> bool {
        self.0.sampling.finished
    }
}

struct Request {
    handle: StateHandle,
    snapshot: Snapshot,
    cancelled: bool,
}

/// One sampled output, indexed by the original packed token position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenOutput {
    pub request: Uuid,
    pub packed_position: usize,
    pub token: u32,
}

/// What one request published from a completed batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceCommit {
    /// New handle; descriptors built from the previous revision are rejected.
    pub handle: StateHandle,
    /// Input tokens consumed into model state. Speculative sequences stop after the first
    /// draft that disagrees with the sampled target token.
    pub accepted_tokens: usize,
    pub outputs: Vec<TokenOutput>,
    /// The grammar reached a terminal state; later outputs will not be sampled.
    pub finished: bool,
}

/// Per-request outcome. An error leaves that request's committed state unchanged.
#[derive(Debug)]
pub struct SequenceResult {
    pub request: Uuid,
    pub outcome: Result<SequenceCommit>,
}

struct Batch<M> {
    // Weights, committed inputs and model outputs stay alive across the completion fence.
    model: Arc<M>,
    plan: PackedExecution,
    bases: Vec<ModelState>,
    inputs: Vec<Vec<LayerRead>>,
    staged: Vec<Result<(StagedUpdate, Tensor)>>,
    error: Option<TurError>,
}

/// Device work enqueued while publishing accepted boundaries reads these buffers.
type Trailing = (
    Vec<ModelState>,
    Vec<Vec<LayerRead>>,
    Vec<(StagedUpdate, Tensor)>,
);

/// One execution owner with at most one in-flight batch. No locks are held during execution.
/// `submit` validates and enqueues one packed pass; `finish` fences, samples and publishes.
pub struct HybridEngine<M: HybridModel> {
    model: Arc<M>,
    description: Arc<ModelDescription>,
    device: Device,
    quarantine: Quarantine,
    requests: HashMap<Uuid, Request>,
    pending: Option<Completion<Batch<M>>>,
    pending_handles: HashSet<Uuid>,
    trailing: Option<Completion<Trailing>>,
}

fn validate_logits(logits: &Tensor, rows: usize, vocab: usize, device: &Device) -> Result<()> {
    if logits.dims() != [rows, vocab]
        || logits.dtype() != DType::F32
        || !logits.device().same_device(device)
    {
        return Err(invalid("model logits must be F32 [outputs, vocabulary]"));
    }
    Ok(())
}

/// Sample in order and decide how many inputs are accepted.
fn sample_sequence(
    sampling: &mut SamplingState,
    sequence: &SequenceDescriptor,
    tokens: &[u32],
    rows: &[Vec<f32>],
) -> Result<(usize, Vec<TokenOutput>)> {
    let length = sequence.tokens.len();
    let mut outputs = Vec::with_capacity(sequence.outputs.len());
    for (index, (&position, row)) in sequence.outputs.iter().zip(rows).enumerate() {
        let Some(token) = sampling.sample(row)? else {
            return Ok((
                if sequence.speculative {
                    index + 1
                } else {
                    length
                },
                outputs,
            ));
        };
        outputs.push(TokenOutput {
            request: sequence.state.request,
            packed_position: position,
            token,
        });
        if sequence.speculative {
            let draft = tokens
                .get(position + 1)
                .filter(|_| position + 1 < sequence.tokens.end);
            if sampling.finished || draft != Some(&token) {
                return Ok((index + 1, outputs));
            }
        }
    }
    Ok((length, outputs))
}

fn boundaries(sequence: &SequenceDescriptor) -> Boundaries {
    if sequence.speculative {
        Boundaries::EveryToken
    } else {
        Boundaries::Final
    }
}

fn forwards<'a>(
    plan: &'a PackedExecution,
    inputs: &'a [Vec<LayerRead>],
) -> Vec<SequenceForward<'a>> {
    plan.sequences
        .iter()
        .zip(inputs)
        .map(|(sequence, state)| SequenceForward {
            tokens: sequence.tokens.clone(),
            state,
            logits: &sequence.outputs,
            boundaries: boundaries(sequence),
        })
        .collect()
}

/// Sample, select the accepted boundary and publish one request; on error nothing changes.
fn publish(
    request: &mut Request,
    sequence: &SequenceDescriptor,
    tokens: &[u32],
    base: &ModelState,
    (update, logits): &(StagedUpdate, Tensor),
) -> Result<SequenceCommit> {
    let rows: Vec<Vec<f32>> = if sequence.outputs.is_empty() {
        Vec::new()
    } else {
        logits.to_vec2()?
    };
    let revision = request
        .handle
        .revision
        .checked_add(1)
        .ok_or_else(|| invalid("request revision overflow"))?;
    let sampling = &mut request.snapshot.sampling;
    let mark = sampling.mark();
    let published = sample_sequence(sampling, sequence, tokens, &rows)
        .and_then(|(accepted, outputs)| Ok((base.apply(update, accepted)?, accepted, outputs)));
    let (model, accepted, outputs) = match published {
        Ok(published) => published,
        Err(error) => {
            sampling.rollback(mark);
            return Err(error);
        }
    };
    request.snapshot.model = model;
    request.handle.revision = revision;
    Ok(SequenceCommit {
        handle: request.handle,
        accepted_tokens: accepted,
        outputs,
        finished: request.snapshot.sampling.finished,
    })
}

impl<M: HybridModel> HybridEngine<M> {
    /// Validate the model description before allocating any request resources.
    /// Completion fences wait on `device`; give each engine its own device for isolation.
    pub fn new(model: Arc<M>, device: Device) -> Result<Self> {
        let description = model.description();
        description.validate()?;
        Ok(Self {
            model,
            description: Arc::new(description),
            device,
            quarantine: Quarantine::new(),
            requests: HashMap::new(),
            pending: None,
            pending_handles: HashSet::new(),
            trailing: None,
        })
    }

    /// Allocate request-owned layer state, independent RNG and an optional grammar parser.
    pub fn allocate(
        &mut self,
        seed: u64,
        options: SamplingOptions,
        guidance: Option<GuidanceControl>,
    ) -> Result<StateHandle> {
        if let Some(guidance) = &guidance {
            guidance.validate_token_contract(self.description.vocab_size)?;
        }
        if options
            .temperature
            .is_some_and(|t| !t.is_finite() || t < 0.0)
            || options
                .top_p
                .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(invalid("invalid sampling options"));
        }
        let snapshot = Snapshot {
            model: ModelState::allocate(Arc::clone(&self.description), &self.device)?,
            sampling: SamplingState {
                seed,
                draws: 0,
                options,
                guidance,
                tokens: Vec::new(),
                finished: false,
            },
        };
        self.insert(snapshot)
    }

    fn insert(&mut self, snapshot: Snapshot) -> Result<StateHandle> {
        let request = Uuid::new_v4();
        if self.requests.contains_key(&request) {
            return Err(invalid("request identity collision"));
        }
        let handle = StateHandle {
            request,
            revision: 0,
        };
        self.requests.insert(
            request,
            Request {
                handle,
                snapshot,
                cancelled: false,
            },
        );
        Ok(handle)
    }

    fn request(&self, handle: StateHandle) -> Result<&Request> {
        let request = self
            .requests
            .get(&handle.request)
            .ok_or_else(|| TurError::RequestNotFound(handle.request.to_string()))?;
        if request.handle != handle || request.cancelled {
            return Err(invalid("stale or cancelled request handle"));
        }
        Ok(request)
    }

    /// Current handle, sequence position and grammar status for descriptor construction.
    pub fn metadata(&self, request: Uuid) -> Result<StateMetadata> {
        let state = self
            .requests
            .get(&request)
            .ok_or_else(|| TurError::RequestNotFound(request.to_string()))?;
        if state.cancelled {
            return Err(invalid("request is cancelled"));
        }
        Ok(StateMetadata {
            handle: state.handle,
            position: state.snapshot.model.position(),
            finished: state.snapshot.sampling.finished,
        })
    }

    /// Snapshot a committed request. Pending candidate state is never visible.
    pub fn checkpoint(&self, handle: StateHandle) -> Result<Checkpoint> {
        Ok(Checkpoint(self.request(handle)?.snapshot.fork()))
    }

    fn validate_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        if checkpoint.0.model.description() != self.description.as_ref()
            || !checkpoint.0.model.device().same_device(&self.device)
        {
            return Err(invalid(
                "checkpoint model identity, layout or device mismatch",
            ));
        }
        Ok(())
    }

    /// Create an independent branch; immutable pages/snapshots are shared, parser state is copied.
    pub fn fork(&mut self, checkpoint: &Checkpoint) -> Result<StateHandle> {
        self.validate_checkpoint(checkpoint)?;
        self.insert(checkpoint.0.fork())
    }

    /// Restore all state atomically, invalidating every descriptor with the previous revision.
    pub fn restore(&mut self, handle: StateHandle, checkpoint: &Checkpoint) -> Result<StateHandle> {
        self.validate_checkpoint(checkpoint)?;
        self.request(handle)?;
        if self.pending_handles.contains(&handle.request) {
            return Err(invalid("cannot restore in-flight request"));
        }
        let revision = handle
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("request revision overflow"))?;
        let snapshot = checkpoint.0.fork();
        let request = self
            .requests
            .get_mut(&handle.request)
            .ok_or_else(|| invalid("request disappeared"))?;
        request.snapshot = snapshot;
        request.handle.revision = revision;
        Ok(request.handle)
    }

    /// Cancel immediately; in-flight resources remain retained until completion.
    pub fn cancel(&mut self, handle: StateHandle) -> Result<()> {
        self.request(handle)?;
        if self.pending_handles.contains(&handle.request) {
            if let Some(request) = self.requests.get_mut(&handle.request) {
                request.cancelled = true;
            }
        } else {
            self.requests.remove(&handle.request);
        }
        Ok(())
    }

    /// Retry this engine's failed fences; returns the number still quarantined.
    pub fn retry_quarantined(&self) -> usize {
        self.quarantine.retry()
    }

    /// Validate the complete packed descriptor, then enqueue one packed model pass.
    /// Returns the checked payload this submission retains, including model workspace.
    /// Per-request execution errors are reported by `finish`.
    pub fn submit(&mut self, plan: PackedExecution) -> Result<ExecutionRequirements> {
        if self.pending.is_some() {
            return Err(invalid("one batch is already in flight"));
        }
        if let Some(mut trailing) = self.trailing.take()
            && let Err(error) = trailing.wait()
        {
            self.trailing = Some(trailing);
            return Err(error);
        }
        if self.quarantine.retry() != 0 {
            return Err(invalid("retired submissions still await device completion"));
        }
        let states: Vec<_> = self
            .requests
            .values()
            .filter(|r| !r.cancelled)
            .map(|r| StateMetadata {
                handle: r.handle,
                position: r.snapshot.model.position(),
                finished: r.snapshot.sampling.finished,
            })
            .collect();
        let (plan, mut requirements) = plan.validate(&self.description, &states)?.into_parts();
        let mut bases = Vec::with_capacity(plan.sequences.len());
        for sequence in &plan.sequences {
            sequence
                .state
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("request revision overflow"))?;
            bases.push(self.request(sequence.state)?.snapshot.model.clone());
        }
        let inputs: Vec<_> = bases.iter().map(ModelState::bind_layers).collect();
        requirements.workspace_bytes = self.model.workspace_bytes(&PackedForward {
            tokens: &plan.tokens,
            positions: &plan.positions,
            sequences: &forwards(&plan, &inputs),
        })?;
        requirements.total_bytes()?;
        let mut completion = Completion::new(
            self.device.clone(),
            &self.quarantine,
            Batch {
                model: Arc::clone(&self.model),
                plan,
                bases,
                inputs,
                staged: Vec::new(),
                error: None,
            },
        );
        let batch = completion
            .resources_mut()
            .ok_or_else(|| invalid("missing submission resources"))?;
        let result = batch.model.forward(&PackedForward {
            tokens: &batch.plan.tokens,
            positions: &batch.plan.positions,
            sequences: &forwards(&batch.plan, &batch.inputs),
        });
        match result {
            Ok(steps) if steps.len() == batch.plan.sequences.len() => {
                for ((sequence, base), step) in
                    batch.plan.sequences.iter().zip(&batch.bases).zip(steps)
                {
                    batch.staged.push(step.and_then(|step| {
                        validate_logits(
                            &step.logits,
                            sequence.outputs.len(),
                            self.description.vocab_size,
                            &self.device,
                        )?;
                        let staged =
                            base.stage(step.layers, sequence.tokens.len(), boundaries(sequence))?;
                        Ok((staged, step.logits))
                    }));
                }
            }
            Ok(_) => batch.error = Some(invalid("model returned wrong number of sequences")),
            Err(error) => batch.error = Some(error),
        }
        self.pending_handles = batch
            .plan
            .sequences
            .iter()
            .map(|s| s.state.request)
            .collect();
        self.pending = Some(completion);
        Ok(requirements)
    }

    /// Fence the pending batch, then sample and publish each request independently.
    /// A fence error keeps the batch pending for retry; a whole-batch model error is
    /// returned once and leaves every request unchanged.
    pub fn finish(&mut self) -> Result<Vec<SequenceResult>> {
        let completion = self
            .pending
            .as_mut()
            .ok_or_else(|| invalid("no batch in flight"))?;
        completion.wait()?;
        let batch = completion
            .take()
            .ok_or_else(|| invalid("incomplete submission"))?;
        self.pending = None;
        self.pending_handles.clear();
        let Batch {
            plan,
            bases,
            inputs,
            staged,
            error,
            ..
        } = batch;
        if let Some(error) = error {
            self.requests.retain(|_, request| !request.cancelled);
            return Err(error);
        }
        let mut results = Vec::with_capacity(plan.sequences.len());
        let mut retained = Vec::with_capacity(staged.len());
        for ((sequence, base), staged) in plan.sequences.iter().zip(&bases).zip(staged) {
            let id = sequence.state.request;
            let live = self.requests.get_mut(&id).filter(|r| !r.cancelled);
            let outcome = match (live, staged) {
                (None, staged) => {
                    retained.extend(staged.ok());
                    continue;
                }
                (Some(_), Err(error)) => Err(error),
                (Some(request), Ok(staged)) => {
                    let outcome = publish(request, sequence, &plan.tokens, base, &staged);
                    retained.push(staged);
                    outcome
                }
            };
            results.push(SequenceResult {
                request: id,
                outcome,
            });
        }
        self.requests.retain(|_, request| !request.cancelled);
        self.trailing = Some(Completion::new(
            self.device.clone(),
            &self.quarantine,
            (bases, inputs, retained),
        ));
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tur_core::execution::{LayerStateKind, Modality, ModelIdentity, SequenceDescriptor};

    struct UnusedModel;
    impl HybridModel for UnusedModel {
        fn description(&self) -> ModelDescription {
            ModelDescription {
                identity: ModelIdentity {
                    revision: "unit-fixture".into(),
                    state_format: 1,
                },
                layers: vec![LayerStateKind::Attention {
                    kv_heads: 1,
                    head_dim: 1,
                    page_tokens: 2,
                    dtype: DType::F32,
                }],
                modalities: vec![Modality::Text],
                vocab_size: 2,
                max_position: 4,
            }
        }
        fn forward(&self, _: &PackedForward<'_>) -> Result<Vec<Result<SequenceStep>>> {
            Err(invalid("unexpected dispatch"))
        }
    }

    #[test]
    fn test_rng_counter_overflow_leaves_bookkeeping_unchanged() {
        let mut state = SamplingState {
            seed: 1,
            draws: u64::MAX,
            options: SamplingOptions::default(),
            guidance: None,
            tokens: vec![],
            finished: false,
        };
        assert!(state.sample(&[1.0, 2.0]).is_err());
        assert_eq!(state.draws, u64::MAX);
        assert!(state.tokens.is_empty());
    }

    #[test]
    fn test_revision_overflow_rejects_submission_and_restore_without_changes() -> Result<()> {
        let mut engine = HybridEngine::new(Arc::new(UnusedModel), Device::Cpu)?;
        let mut handle = engine.allocate(1, SamplingOptions::default(), None)?;
        let checkpoint = engine.checkpoint(handle)?;
        handle.revision = u64::MAX;
        engine
            .requests
            .get_mut(&handle.request)
            .ok_or_else(|| invalid("missing fixture request"))?
            .handle = handle;
        let plan = PackedExecution {
            tokens: vec![1],
            positions: vec![0],
            sequences: vec![SequenceDescriptor {
                state: handle,
                tokens: 0..1,
                outputs: vec![0],
                speculative: false,
            }],
        };
        assert!(engine.submit(plan).is_err());
        assert!(engine.restore(handle, &checkpoint).is_err());
        assert_eq!(engine.checkpoint(handle)?.position(), 0);
        assert_eq!(engine.metadata(handle.request)?.handle.revision, u64::MAX);
        assert!(engine.pending.is_none());
        Ok(())
    }

    #[test]
    fn test_invalid_sampling_options_do_not_allocate_requests() -> Result<()> {
        let mut engine = HybridEngine::new(Arc::new(UnusedModel), Device::Cpu)?;
        for options in [
            SamplingOptions {
                temperature: Some(f64::NAN),
                top_p: None,
            },
            SamplingOptions {
                temperature: Some(-1.0),
                top_p: None,
            },
            SamplingOptions {
                temperature: None,
                top_p: Some(0.0),
            },
            SamplingOptions {
                temperature: None,
                top_p: Some(1.1),
            },
        ] {
            assert!(engine.allocate(1, options, None).is_err());
        }
        assert!(engine.requests.is_empty());
        Ok(())
    }

    #[test]
    fn test_rollback_restores_history_and_draws() {
        let mut state = SamplingState {
            seed: 3,
            draws: 0,
            options: SamplingOptions::default(),
            guidance: None,
            tokens: vec![],
            finished: false,
        };
        let mark = state.mark();
        assert!(matches!(state.sample(&[0.0, 1.0]), Ok(Some(1))));
        assert!(state.sample(&[f32::NAN, 1.0]).is_err());
        state.rollback(mark);
        assert_eq!(state.draws, 0);
        assert!(state.tokens.is_empty());
    }
}
