#![forbid(unsafe_code)]
//! Single-owner hybrid request execution with completion-gated transactions.

use crate::backend::guidance::GuidanceControl;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tur_core::{
    Result, TurError,
    execution::{ModelDescription, PackedExecution, StateHandle, StateMetadata, invalid},
};
use tur_metal::{
    completion::{Completion, retry_retired},
    state::{LayerRead, ModelState},
};
pub use tur_models::hybrid::{HybridModel, TokenStep};
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
}

impl SamplingState {
    fn fork(&self) -> Self {
        Self {
            seed: self.seed,
            draws: self.draws,
            options: self.options,
            guidance: self.guidance.as_ref().map(GuidanceControl::fork),
            tokens: self.tokens.clone(),
        }
    }

    fn sample(&mut self, logits: &Tensor, vocab: usize) -> Result<u32> {
        if logits.dtype() != DType::F32 || logits.dims() != [vocab] {
            return Err(invalid("model logits must be F32 vocabulary vector"));
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
        let mut values = logits.to_vec1::<f32>()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("nonfinite model logits"));
        }
        if let Some(guidance) = &mut self.guidance {
            if guidance.apply_mask(&mut values)? {
                return Err(TurError::Guidance("grammar has stopped".into()));
            }
        }
        if !values.iter().any(|v| v.is_finite()) {
            return Err(TurError::Guidance("grammar disallows every token".into()));
        }
        let masked = Tensor::from_vec(values, vocab, logits.device())?;
        let mut sampler = LogitsProcessor::new(seed, self.options.temperature, self.options.top_p);
        let token = sampler.sample(&masked)?;
        if let Some(guidance) = &mut self.guidance {
            guidance.commit(token)?;
        }
        self.draws = next_draw;
        self.tokens.push(token);
        Ok(token)
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

struct Work {
    handle: StateHandle,
    base: Snapshot,
    boundaries: Vec<Snapshot>,
    outputs: Vec<(usize, TokenOutput)>,
}

struct StepResources {
    inputs: Vec<LayerRead>,
    output: Option<TokenStep>,
}

struct Batch<M> {
    model: Arc<M>,
    work: Vec<Work>,
    // Keep detached input and output bindings alive across the completion fence.
    scratch: Vec<StepResources>,
    candidates: Vec<ModelState>,
    error: Option<TurError>,
}

/// One execution owner with at most one in-flight batch. No locks are held during execution.
/// `submit` stages updates; `commit` fences and publishes accepted boundaries exactly once.
pub struct HybridEngine<M: HybridModel> {
    model: Arc<M>,
    description: Arc<ModelDescription>,
    device: Device,
    requests: HashMap<Uuid, Request>,
    pending: Option<Completion<Batch<M>>>,
    pending_handles: HashSet<Uuid>,
    pending_lengths: Vec<(Uuid, usize)>,
}

impl<M: HybridModel> HybridEngine<M> {
    /// Validate the model description before allocating any request resources.
    pub fn new(model: Arc<M>, device: Device) -> Result<Self> {
        let description = model.description();
        description.validate()?;
        Ok(Self {
            model,
            description: Arc::new(description),
            device,
            requests: HashMap::new(),
            pending: None,
            pending_handles: HashSet::new(),
            pending_lengths: Vec::new(),
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

    /// Current handle and sequence position for descriptor construction.
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

    /// Validate the complete packed descriptor before any model work, then stage private updates.
    /// Execution errors are reported by `commit` after the completion fence.
    pub fn submit(&mut self, plan: PackedExecution) -> Result<()> {
        if self.pending.is_some() {
            return Err(invalid("one batch is already in flight"));
        }
        if retry_retired() != 0 {
            return Err(invalid("retired submissions still await device completion"));
        }
        let states: Vec<_> = self
            .requests
            .values()
            .filter(|r| !r.cancelled)
            .map(|r| StateMetadata {
                handle: r.handle,
                position: r.snapshot.model.position(),
            })
            .collect();
        let validated = plan.validate(&self.description, &states)?;
        let plan = validated.packed();
        let mut work = Vec::with_capacity(plan.sequences.len());
        for sequence in &plan.sequences {
            sequence
                .state
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("request revision overflow"))?;
            work.push(Work {
                handle: sequence.state,
                base: self.request(sequence.state)?.snapshot.fork(),
                boundaries: Vec::new(),
                outputs: Vec::new(),
            });
        }
        let mut completion = Completion::new(
            self.device.clone(),
            Batch {
                model: Arc::clone(&self.model),
                work,
                scratch: Vec::new(),
                candidates: Vec::new(),
                error: None,
            },
        );
        let batch = completion
            .resources_mut()
            .ok_or_else(|| invalid("missing submission resources"))?;
        let result = (|| -> Result<()> {
            for (sequence, work) in plan.sequences.iter().zip(&mut batch.work) {
                let mut state = work.base.fork();
                for packed_position in sequence.tokens.clone() {
                    let token = *plan
                        .tokens
                        .get(packed_position)
                        .ok_or_else(|| invalid("packed token out of bounds"))?;
                    // Own inputs before calling the model: it may encode work and then fail.
                    batch.scratch.push(StepResources {
                        inputs: state.model.read_layers()?,
                        output: None,
                    });
                    let bindings = batch
                        .scratch
                        .last_mut()
                        .ok_or_else(|| invalid("missing step resources"))?;
                    bindings.output = Some(batch.model.step(
                        token,
                        state.model.position(),
                        &bindings.inputs,
                    )?);
                    let step = bindings
                        .output
                        .as_ref()
                        .ok_or_else(|| invalid("missing step output"))?;
                    state.model = state.model.advance(&step.layers)?;
                    batch.candidates.push(state.model.clone());
                    if !step.logits.device().same_device(&self.device)
                        || step.logits.dims() != [self.description.vocab_size]
                        || step.logits.dtype() != DType::F32
                    {
                        return Err(invalid("invalid output logits"));
                    }
                    if sequence.outputs.contains(&packed_position) {
                        let token = state
                            .sampling
                            .sample(&step.logits, self.description.vocab_size)?;
                        work.outputs.push((
                            work.boundaries.len(),
                            TokenOutput {
                                request: sequence.state.request,
                                packed_position,
                                token,
                            },
                        ));
                    }
                    work.boundaries.push(state.fork());
                }
            }
            Ok(())
        })();
        batch.error = result.err();
        self.pending_handles = plan.sequences.iter().map(|s| s.state.request).collect();
        self.pending_lengths = plan
            .sequences
            .iter()
            .map(|s| (s.state.request, s.tokens.len()))
            .collect();
        self.pending = Some(completion);
        Ok(())
    }

    /// Accept all staged tokens after completion, publishing each request atomically.
    pub fn finish(&mut self) -> Result<Vec<TokenOutput>> {
        let accepted = self.pending_lengths.clone();
        self.commit(&accepted)
    }

    /// Commit exactly one accepted token prefix per request, including zero for full rollback.
    /// Rejected suffixes roll back model state, position, RNG, sampled history and grammar together.
    /// A rejected recurrent suffix selects its exact saved boundary; it is never truncated.
    pub fn commit(&mut self, accepted: &[(Uuid, usize)]) -> Result<Vec<TokenOutput>> {
        if self.pending.is_none() {
            return Err(invalid("no batch in flight"));
        }
        let mut counts = HashMap::new();
        for &(request, count) in accepted {
            let length = self
                .pending_lengths
                .iter()
                .find(|(id, _)| *id == request)
                .map(|(_, len)| *len)
                .ok_or_else(|| invalid("unknown acceptance request"))?;
            if count > length || counts.insert(request, count).is_some() {
                return Err(invalid("invalid accepted prefix"));
            }
        }
        if counts.len() != self.pending_lengths.len() {
            return Err(invalid("missing accepted prefix"));
        }
        let completion = self
            .pending
            .as_mut()
            .ok_or_else(|| invalid("missing completion"))?;
        completion.wait()?;
        let batch = completion
            .take()
            .ok_or_else(|| invalid("incomplete submission"))?;
        self.pending = None;
        self.pending_handles.clear();
        self.pending_lengths.clear();
        if let Some(error) = batch.error {
            self.requests.retain(|_, request| !request.cancelled);
            return Err(error);
        }
        // Prepare all publications before mutating any live request.
        let mut publications = Vec::with_capacity(batch.work.len());
        let mut outputs = Vec::new();
        for work in batch.work {
            let count = *counts
                .get(&work.handle.request)
                .ok_or_else(|| invalid("missing acceptance count"))?;
            let request = self
                .requests
                .get(&work.handle.request)
                .ok_or_else(|| invalid("in-flight request disappeared"))?;
            if request.cancelled {
                continue;
            }
            let revision = request
                .handle
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("request revision overflow"))?;
            let snapshot = if count == 0 {
                work.base
            } else {
                work.boundaries
                    .into_iter()
                    .nth(count - 1)
                    .ok_or_else(|| invalid("missing accepted boundary"))?
            };
            outputs.extend(
                work.outputs
                    .into_iter()
                    .filter(|(boundary, _)| *boundary < count)
                    .map(|(_, output)| output),
            );
            publications.push((work.handle.request, revision, snapshot));
        }
        for (id, revision, snapshot) in publications {
            if let Some(request) = self.requests.get_mut(&id) {
                request.snapshot = snapshot;
                request.handle.revision = revision;
            }
        }
        self.requests.retain(|_, request| !request.cancelled);
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tur_core::execution::{LayerStateKind, ModelIdentity, SequenceDescriptor};

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
                }],
                vocab_size: 2,
                max_position: 4,
            }
        }
        fn step(&self, _: u32, _: usize, _: &[LayerRead]) -> Result<TokenStep> {
            Err(invalid("unexpected dispatch"))
        }
    }

    #[test]
    fn test_rng_counter_overflow_leaves_bookkeeping_unchanged() -> Result<()> {
        let mut state = SamplingState {
            seed: 1,
            draws: u64::MAX,
            options: SamplingOptions::default(),
            guidance: None,
            tokens: vec![],
        };
        assert!(
            state
                .sample(&Tensor::new(&[1.0f32, 2.0], &Device::Cpu)?, 2)
                .is_err()
        );
        assert_eq!(state.draws, u64::MAX);
        assert!(state.tokens.is_empty());
        Ok(())
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
}
