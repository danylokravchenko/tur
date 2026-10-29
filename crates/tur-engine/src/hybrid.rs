#![forbid(unsafe_code)]
//! Single-owner hybrid request execution with completion-gated publication.

use crate::backend::guidance::{GrammarMask, GuidanceControl};
use candle_core::{DType, Device, Tensor};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};
use tur_core::{
    Result, TurError,
    execution::{
        ExecutionRequirements, ModelDescription, PackedExecution, SequenceDescriptor, StateHandle,
        StateMetadata, invalid,
    },
};
use tur_metal::{
    completion::{Completion, Notifier, Quarantine},
    sampling::{self, PenaltyEntry, PenaltyParams, SampleParams, SampleStatus},
};
pub use tur_metal::{
    pool::{PoolConfig, PoolUsage, StatePool},
    state::{Demand, ModelState, Reservation},
};
pub use tur_models::hybrid::{
    Boundaries, DraftForward, DraftSequence, HybridModel, PackedForward, SequenceForward,
    SequenceState, SequenceStep,
};
use uuid::Uuid;

/// Per-request sampling policy. Unset fields disable their control.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingOptions {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    /// Divides positive and multiplies negative logits of every prompt or sampled token;
    /// above 1 discourages repetition.
    pub repetition_penalty: Option<f64>,
    /// Subtracted once from the logit of every token sampled so far.
    pub presence_penalty: Option<f64>,
    /// Subtracted per occurrence from the logit of every token sampled so far.
    pub frequency_penalty: Option<f64>,
}

impl SamplingOptions {
    /// Greedy below a temperature of 1e-7, ignoring top-k and top-p.
    pub fn greedy(&self) -> bool {
        self.temperature.is_none_or(|t| t < 1e-7)
    }

    fn penalties(&self) -> PenaltyParams {
        PenaltyParams {
            repetition: self.repetition_penalty.unwrap_or(1.0) as f32,
            presence: self.presence_penalty.unwrap_or(0.0) as f32,
            frequency: self.frequency_penalty.unwrap_or(0.0) as f32,
        }
    }

    fn params(&self, seed: u64, draw: u64) -> SampleParams {
        SampleParams {
            temperature: self.temperature.unwrap_or(1.0) as f32,
            top_k: self
                .top_k
                .map_or(0, |k| u32::try_from(k).unwrap_or(u32::MAX)),
            top_p: self.top_p.unwrap_or(1.0) as f32,
            seed,
            draw,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.temperature.is_some_and(|t| !t.is_finite() || t < 0.0)
            || self.top_k == Some(0)
            || self
                .top_p
                .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(invalid("invalid sampling options"));
        }
        self.penalties().validate()?;
        if !self.greedy() {
            self.params(0, 0).validate()?;
        }
        Ok(())
    }
}

/// Penalty parameters and distinct penalized tokens of one output row.
type Penalties = (PenaltyParams, Vec<PenaltyEntry>);

/// Context tokens and sampled counts of a penalized request.
#[derive(Clone, Default)]
struct History {
    counts: HashMap<u32, u32>,
    /// Prompt tokens, penalized by the repetition penalty besides sampled ones.
    context: HashSet<u32>,
}

impl History {
    /// Distinct penalized tokens of one output whose position follows `drafts`, tokens of
    /// this pass that count as sampled if the output is used (speculative drafts).
    fn entries(&self, penalties: &PenaltyParams, drafts: &[u32]) -> Vec<PenaltyEntry> {
        let mut counts = self.counts.clone();
        for &token in drafts {
            *counts.entry(token).or_default() += 1;
        }
        if penalties.uses_context() {
            for &token in &self.context {
                counts.entry(token).or_default();
            }
        }
        counts
            .into_iter()
            .map(|(token, count)| PenaltyEntry { token, count })
            .collect()
    }
}

struct SamplingState {
    seed: u64,
    draws: u64,
    options: SamplingOptions,
    guidance: Option<GuidanceControl>,
    tokens: Vec<u32>,
    finished: bool,
    /// Present while any penalty is active.
    history: Option<History>,
}

/// Bookkeeping needed to undo one request's sampling after a mid-sequence error.
struct SamplingMark {
    draws: u64,
    tokens: usize,
    finished: bool,
    guidance: Option<GuidanceControl>,
}

/// One output position's logits as `finish` delivers them.
enum Row<'a> {
    /// Model logits: the host applies penalties, the grammar mask and the sampler.
    Logits(&'a [f32]),
    /// Penalized on the device, whose rejection rounds did not settle the sample.
    Penalized(Vec<f32>),
    /// Decided on the device: argmax for greedy requests, otherwise the sampling kernel.
    Device { token: u32, status: SampleStatus },
    /// The grammar stopped before this output; nothing is sampled.
    Stopped,
}

/// How the device decides one output row: penalties, then an optional grammar mask, then
/// argmax (greedy) or the sampling kernel.
#[cfg_attr(
    not(feature = "metal"),
    allow(dead_code, reason = "without Metal every row stays on the host")
)]
struct DeviceRow {
    greedy: bool,
    penalties: Option<Penalties>,
    mask: Option<Vec<u32>>,
    params: SampleParams,
}

/// A row's device decision, or a grammar that already stopped.
enum DevicePlan {
    Stopped,
    Row(DeviceRow),
}

/// Penalize and mask `logits` (`[rows.len(), vocab]`) in place, then decide every row:
/// one dispatch per stage, greedy rows (which must come first) by argmax and the rest by
/// sampling. Returns `[rows, 2]` token/status pairs.
#[cfg(feature = "metal")]
fn decide_rows(
    kernels: &tur_metal::kernels::Kernels,
    logits: &Tensor,
    rows: &[DeviceRow],
) -> Result<Tensor> {
    let (mut penalties, mut entries, mut masked, mut bits, mut params) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (index, row) in rows.iter().enumerate() {
        let index = u32::try_from(index).map_err(|_| invalid("row exceeds u32"))?;
        match &row.penalties {
            Some((p, e)) => {
                penalties.push(*p);
                entries.extend(e.iter().map(|entry| (index, *entry)));
            }
            None => penalties.push(PenaltyParams::NONE),
        }
        if let Some(mask) = &row.mask {
            // A grammar's bitset may carry spare capacity beyond the vocabulary.
            let words = logits.dim(1)?.div_ceil(32);
            masked.push(index);
            bits.extend(mask.iter().copied().chain(std::iter::repeat(0)).take(words));
        }
        if !row.greedy {
            params.push(row.params);
        }
    }
    kernels.penalize(logits, &penalties, &entries)?;
    kernels.mask(logits, &masked, &bits)?;
    let greedy = rows.iter().take_while(|row| row.greedy).count();
    if rows[greedy..].iter().any(|row| row.greedy) {
        return Err(invalid("greedy device rows must come first"));
    }
    let mut pairs = Vec::with_capacity(2);
    if greedy > 0 {
        pairs.push(kernels.argmax(&logits.narrow(0, 0, greedy)?)?);
    }
    if rows.len() > greedy {
        pairs.push(kernels.sample(&logits.narrow(0, greedy, rows.len() - greedy)?, &params)?);
    }
    Ok(Tensor::cat(&pairs, 0)?)
}

/// First maximum of `logits`; −∞ marks a masked token only when `masked`. Chunked so the
/// finiteness check and maximum vectorize; only a chunk with a new maximum is searched for
/// its index.
fn first_max(logits: &[f32], masked: bool) -> Result<u32> {
    const CHUNK: usize = 64;
    let mut best: Option<(usize, f32)> = None;
    for (chunk_index, chunk) in logits.chunks(CHUNK).enumerate() {
        let (valid, max) = chunk
            .iter()
            .fold((true, f32::NEG_INFINITY), |(valid, max), &v| {
                // `v < +∞` also rejects NaN while admitting masked −∞.
                let admitted = if masked {
                    v < f32::INFINITY
                } else {
                    v.is_finite()
                };
                #[allow(clippy::needless_bitwise_bool, reason = "branch-free so it vectorizes")]
                (valid & admitted, max.max(v))
            });
        if !valid {
            return Err(invalid("nonfinite model logits"));
        }
        if max > f32::NEG_INFINITY && best.is_none_or(|(_, value)| max > value) {
            // Exact: `max` is one of the chunk's values.
            #[allow(
                clippy::float_cmp,
                reason = "locating the element equal to the maximum"
            )]
            let offset = chunk.iter().position(|&v| v == max).unwrap_or(0);
            best = Some((chunk_index * CHUNK + offset, max));
        }
    }
    let (index, _) = best.ok_or_else(|| TurError::Guidance("every token is masked".into()))?;
    u32::try_from(index).map_err(|_| invalid("token id exceeds u32"))
}

impl SamplingState {
    fn new(seed: u64, options: SamplingOptions, guidance: Option<GuidanceControl>) -> Self {
        Self {
            seed,
            draws: 0,
            history: options.penalties().active().then(History::default),
            options,
            guidance,
            tokens: Vec::new(),
            finished: false,
        }
    }

    /// The device decision for output `offset` of this pass after `drafts`. A grammar's
    /// mask is computed now, while the device runs the model, and committed at
    /// publication; callers use this only for a grammar's single output of a pass.
    fn device_plan(&mut self, drafts: &[u32], offset: u64) -> Result<DevicePlan> {
        let mask = match self
            .guidance
            .as_mut()
            .map(GuidanceControl::mask)
            .transpose()?
        {
            Some(Some(GrammarMask::Stop)) => return Ok(DevicePlan::Stopped),
            Some(Some(GrammarMask::Allowed(bits))) => Some(bits),
            Some(None) | None => None,
        };
        Ok(DevicePlan::Row(DeviceRow {
            greedy: self.options.greedy(),
            penalties: self.penalty_entries(drafts),
            mask,
            params: self
                .options
                .params(self.seed, self.draws.wrapping_add(offset)),
        }))
    }

    /// Penalties of an output after `drafts`, or `None` when inactive. They see the
    /// history committed before the pass, never outputs sampled within it, so the host and
    /// the device (which builds them at submission) agree.
    fn penalty_entries(&self, drafts: &[u32]) -> Option<Penalties> {
        let penalties = self.options.penalties();
        self.history
            .as_ref()
            .map(|history| (penalties, history.entries(&penalties, drafts)))
    }

    fn fork(&self) -> Self {
        Self {
            seed: self.seed,
            draws: self.draws,
            options: self.options,
            guidance: self.guidance.as_ref().map(GuidanceControl::fork),
            tokens: self.tokens.clone(),
            finished: self.finished,
            history: self.history.clone(),
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
        if let Some(history) = &mut self.history {
            for token in self.tokens.get(mark.tokens..).unwrap_or_default() {
                if let Some(count) = history.counts.get_mut(token) {
                    *count -= 1;
                    if *count == 0 {
                        history.counts.remove(token);
                    }
                }
            }
        }
        self.tokens.truncate(mark.tokens);
        self.finished = mark.finished;
        if let Some(guidance) = mark.guidance {
            self.guidance = Some(guidance);
        }
    }

    /// Sample one output with its precomputed penalties (see [`Self::penalty_entries`]),
    /// or `None` once the grammar has reached a terminal state.
    fn sample(&mut self, row: Row<'_>, penalties: Option<&Penalties>) -> Result<Option<u32>> {
        if self.finished {
            return Ok(None);
        }
        let next_draw = self
            .draws
            .checked_add(1)
            .ok_or_else(|| invalid("RNG counter overflow"))?;
        let token = match row {
            Row::Stopped => {
                if let Some(guidance) = &mut self.guidance {
                    guidance.stop();
                }
                self.finished = true;
                return Ok(None);
            }
            Row::Device { token, status } => match status {
                SampleStatus::Sampled => token,
                SampleStatus::Invalid if self.guidance.is_some() => {
                    return Err(TurError::Guidance(
                        "grammar disallows every token, or nonfinite logits".into(),
                    ));
                }
                SampleStatus::Invalid => return Err(invalid("nonfinite model logits")),
                SampleStatus::Unconverged => {
                    return Err(invalid("unsettled device sample without its row"));
                }
            },
            Row::Penalized(values) => {
                sampling::sample(&values, &self.options.params(self.seed, self.draws))?
            }
            Row::Logits(logits) => {
                let plain = self.guidance.is_none() && self.history.is_none();
                if plain && self.options.greedy() {
                    first_max(logits, false)?
                } else {
                    if logits.iter().any(|value| !value.is_finite()) {
                        return Err(invalid("nonfinite model logits"));
                    }
                    let mut values = logits.to_vec();
                    if let Some((params, entries)) = penalties {
                        sampling::penalize(&mut values, params, entries)?;
                    }
                    if let Some(guidance) = &mut self.guidance
                        && guidance.apply_mask(&mut values)?
                    {
                        self.finished = true;
                        return Ok(None);
                    }
                    if self.options.greedy() {
                        first_max(&values, true)?
                    } else {
                        sampling::sample(&values, &self.options.params(self.seed, self.draws))?
                    }
                }
            }
        };
        if let Some(guidance) = &mut self.guidance
            && guidance.commit(token)?
        {
            self.finished = true;
        }
        if let Some(history) = &mut self.history {
            *history.counts.entry(token).or_default() += 1;
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
/// Cloning uses independent parser state and shares the snapshot's pool pages and slot;
/// holding a checkpoint keeps them leased.
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
    /// Model logits of the sequence's last output, before penalties and masks; a view of
    /// the pass's output, so copy it to keep it beyond the result.
    pub logits: Option<Tensor>,
    /// Final hidden states of every input token of the pass, when the draft head is
    /// enabled ([`HybridEngine::enable_draft_head`]); the first `accepted_tokens` rows are
    /// the committed ones.
    pub hidden: Option<Tensor>,
}

/// One request's draft-head pass over its newest committed positions.
pub struct DraftRequest {
    pub handle: StateHandle,
    /// First position; the positions run to the request's committed position.
    pub start: usize,
    /// The token at each position + 1: committed inputs, then the newest sampled output.
    pub tokens: Vec<u32>,
    /// F32 `[tokens, hidden]` final hidden states of the positions.
    pub hidden: Tensor,
    /// Greedy drafts to return after the newest token, chained where the head can (a
    /// Gemma 4 assistant); a head that cannot returns at most one.
    pub drafts: usize,
}

/// The draft head's greedy draft.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draft {
    pub token: u32,
    /// The head's softmax probability of `token` over its vocabulary: a confidence to be
    /// calibrated against observed acceptance, not an acceptance probability itself.
    pub confidence: f32,
}

/// Output rows sampled since the engine was created, by where they were decided.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplingStats {
    /// Penalized, reduced or sampled on the device; only the token returns to the host.
    pub device_rows: u64,
    /// Copied in full for the host sampler (grammar-constrained requests, or no kernels).
    pub host_rows: u64,
    /// Device rows whose rejection rounds did not settle, resampled on the host.
    pub fallback_rows: u64,
}

/// Where a sequence's output rows went before the fence.
#[derive(Clone, Copy)]
enum Placement {
    /// No outputs, or the model failed this sequence.
    None,
    /// Full logit rows from this row of the host copy.
    Host(usize),
    /// The grammar stopped before the output; nothing is decided.
    Stopped,
    /// Token/status pairs from this row of the device-decided copy.
    #[cfg_attr(
        not(feature = "metal"),
        allow(dead_code, reason = "without Metal every row stays on the host")
    )]
    Device(usize),
}

struct Batch<M> {
    // Weights, every page and slot the pass reads or writes, and logits stay alive across
    // the completion fence.
    model: Arc<M>,
    plan: PackedExecution,
    reservations: Vec<Reservation>,
    logits: Vec<Result<Tensor>>,
    /// Final hidden states of every sequence's tokens, when the draft head is enabled.
    hidden: Vec<Option<Tensor>>,
    placement: Vec<Placement>,
    /// Full rows for the host sampler and token/status pairs decided on the device, each
    /// concatenated before the fence, so `finish` copies them to the host once.
    host: Option<Tensor>,
    pairs: Option<Tensor>,
    /// Penalized device rows, read back only for samples the device could not settle.
    device_rows: Option<Tensor>,
    error: Option<TurError>,
}

impl<M> Batch<M> {
    /// Host rows then device pairs, flattened into one tensor for a single host copy.
    fn readback(&mut self) -> Result<Option<Tensor>> {
        let parts = [&self.host, &self.pairs]
            .into_iter()
            .flatten()
            .map(Tensor::flatten_all)
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(match parts.len() {
            0 => None,
            1 => parts.into_iter().next(),
            _ => Some(Tensor::cat(&parts, 0)?),
        })
    }
}

/// One execution owner with at most one in-flight batch. No locks are held during execution.
/// `submit` validates, reserves pool memory and enqueues one packed pass; `finish` fences,
/// samples and publishes. Released pages and slots are reclaimed only while no batch is in
/// flight, so no unfenced work can reference a reused id.
pub struct HybridEngine<M: HybridModel> {
    model: Arc<M>,
    description: Arc<ModelDescription>,
    device: Device,
    pool: StatePool,
    quarantine: Quarantine,
    requests: HashMap<Uuid, Request>,
    pending: Option<Completion<Batch<M>>>,
    pending_handles: HashSet<Uuid>,
    /// Observes completion of the pending batch off the execution thread.
    notifier: Notifier,
    /// When the device finished the last completed batch.
    completed_at: Option<Instant>,
    stats: SamplingStats,
    /// Decide unconstrained rows on the device when kernels exist.
    device_sampling: bool,
    /// Return every sequence's final hidden states for the model's draft head.
    draft_head: bool,
    /// The model's kernels, for penalizing and sampling rows on the device.
    #[cfg(feature = "metal")]
    kernels: Option<tur_metal::kernels::Kernels>,
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

/// Speculative drafts at or before packed `position`: they count as sampled tokens for
/// that output's penalties, because the output is used only if every draft was accepted.
fn drafts<'a>(sequence: &SequenceDescriptor, tokens: &'a [u32], position: usize) -> &'a [u32] {
    if sequence.speculative {
        tokens
            .get(sequence.tokens.start + 1..=position)
            .unwrap_or_default()
    } else {
        &[]
    }
}

/// Sample in order from `rows` and decide how many inputs are accepted.
fn sample_sequence(
    sampling: &mut SamplingState,
    sequence: &SequenceDescriptor,
    tokens: &[u32],
    rows: Vec<Row<'_>>,
) -> Result<(usize, Vec<TokenOutput>)> {
    let length = sequence.tokens.len();
    let penalties: Vec<Option<Penalties>> = sequence
        .outputs
        .iter()
        .map(|&position| sampling.penalty_entries(drafts(sequence, tokens, position)))
        .collect();
    let mut outputs = Vec::with_capacity(sequence.outputs.len());
    for (index, ((&position, row), penalties)) in sequence
        .outputs
        .iter()
        .zip(rows)
        .zip(&penalties)
        .enumerate()
    {
        let Some(token) = sampling.sample(row, penalties.as_ref())? else {
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

fn boundaries(speculative: bool) -> Boundaries {
    if speculative {
        Boundaries::EveryToken
    } else {
        Boundaries::Final
    }
}

fn forwards<'a>(
    plan: &'a PackedExecution,
    states: &'a [SequenceState],
    hidden: bool,
) -> Vec<SequenceForward<'a>> {
    plan.sequences
        .iter()
        .zip(states)
        .map(|(sequence, state)| SequenceForward {
            tokens: sequence.tokens.clone(),
            state,
            logits: &sequence.outputs,
            hidden,
        })
        .collect()
}

/// Sample from `rows`, select the accepted boundary and publish one request; on error
/// nothing changes.
fn publish(
    request: &mut Request,
    sequence: &SequenceDescriptor,
    tokens: &[u32],
    reservation: &Reservation,
    rows: Vec<Row<'_>>,
) -> Result<SequenceCommit> {
    let revision = request
        .handle
        .revision
        .checked_add(1)
        .ok_or_else(|| invalid("request revision overflow"))?;
    let sampling = &mut request.snapshot.sampling;
    let mark = sampling.mark();
    let published = sample_sequence(sampling, sequence, tokens, rows)
        .and_then(|(accepted, outputs)| Ok((reservation.commit(accepted)?, accepted, outputs)));
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
    /// Validate the model description and allocate its state pools with `config` capacity.
    /// Completion fences wait on `device`; give each engine its own device for isolation.
    pub fn new(model: Arc<M>, device: Device, config: PoolConfig) -> Result<Self> {
        let description = Arc::new(model.description());
        let pool = StatePool::new(Arc::clone(&description), &device, config)?;
        Ok(Self {
            description,
            device,
            pool,
            quarantine: Quarantine::new(),
            requests: HashMap::new(),
            pending: None,
            pending_handles: HashSet::new(),
            notifier: Notifier::new()?,
            completed_at: None,
            stats: SamplingStats::default(),
            device_sampling: true,
            draft_head: false,
            #[cfg(feature = "metal")]
            kernels: model.kernels(),
            model,
        })
    }

    /// When the device finished the most recently completed batch.
    pub fn completed_at(&self) -> Option<Instant> {
        self.completed_at
    }

    /// Whether a submitted batch awaits `finish`.
    pub fn in_flight(&self) -> bool {
        self.pending.is_some()
    }

    /// Where output rows have been sampled so far.
    pub fn sampling_stats(&self) -> SamplingStats {
        self.stats
    }

    /// Persistent state storage shared by every request of this engine.
    pub fn pool(&self) -> &StatePool {
        &self.pool
    }

    /// Reclaim released pages and slots if no batch is in flight, then report occupancy.
    pub fn memory(&mut self) -> Result<PoolUsage> {
        if self.pending.is_none() && self.quarantine.retry() == 0 {
            self.pool.reclaim();
        }
        self.pool.usage()
    }

    /// Pool pages and slots that a pass of `tokens` new tokens for `handle` would lease.
    pub fn demand(&self, handle: StateHandle, tokens: usize, speculative: bool) -> Result<Demand> {
        self.request(handle)?
            .snapshot
            .model
            .demand(tokens, boundaries(speculative))
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
        options.validate()?;
        let snapshot = Snapshot {
            model: ModelState::new(&self.pool),
            sampling: SamplingState::new(seed, options, guidance),
        };
        self.insert(snapshot)
    }

    /// Record prompt tokens that the repetition penalty treats like sampled ones. Without
    /// that penalty nothing is stored.
    pub fn add_context(&mut self, handle: StateHandle, tokens: &[u32]) -> Result<()> {
        self.request(handle)?;
        let sampling = &mut self
            .requests
            .get_mut(&handle.request)
            .ok_or_else(|| invalid("request disappeared"))?
            .snapshot
            .sampling;
        if sampling.options.penalties().uses_context()
            && let Some(history) = &mut sampling.history
        {
            history.context.extend(tokens);
        }
        Ok(())
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
        // Pages live in one engine's pool; the pool pins model identity, layout and device.
        if !checkpoint.0.model.pool().same_pool(&self.pool) {
            return Err(invalid(
                "checkpoint belongs to another model, state layout or engine",
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

    /// The committed model state of a request, sharing its pages and slot; holding it
    /// keeps them leased. Pending candidate state is never visible.
    pub fn model_state(&self, handle: StateHandle) -> Result<ModelState> {
        Ok(self.request(handle)?.snapshot.model.clone())
    }

    /// Start a request that holds no tokens from a committed model state of this engine,
    /// such as a prefix-cache snapshot. RNG, sampled history and grammar are unchanged.
    /// Invalidates every descriptor with the previous revision.
    pub fn adopt(&mut self, handle: StateHandle, state: &ModelState) -> Result<StateHandle> {
        if !state.pool().same_pool(&self.pool) {
            return Err(invalid("model state belongs to another engine"));
        }
        if self.request(handle)?.snapshot.model.position() != 0
            || self.pending_handles.contains(&handle.request)
        {
            return Err(invalid("only an idle request without state can adopt one"));
        }
        let revision = handle
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("request revision overflow"))?;
        let request = self
            .requests
            .get_mut(&handle.request)
            .ok_or_else(|| invalid("request disappeared"))?;
        request.snapshot.model = state.clone();
        request.handle.revision = revision;
        Ok(request.handle)
    }

    /// Drop every layer's state for recompute preemption: the position returns to zero and
    /// the pages and slot are released, while RNG draws, sampled history and the grammar
    /// parser are kept, so recomputing the same tokens continues the same sample stream.
    /// Invalidates every descriptor with the previous revision.
    pub fn release_state(&mut self, handle: StateHandle) -> Result<StateHandle> {
        self.request(handle)?;
        if self.pending_handles.contains(&handle.request) {
            return Err(invalid("cannot release in-flight request state"));
        }
        let revision = handle
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("request revision overflow"))?;
        let request = self
            .requests
            .get_mut(&handle.request)
            .ok_or_else(|| invalid("request disappeared"))?;
        request.snapshot.model = ModelState::new(&self.pool);
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

    /// Validate the complete packed descriptor, reserve pool memory for every sequence,
    /// then enqueue one packed model pass. Returns the checked payload this submission
    /// retains, including model workspace. A pool that cannot hold the pass fails with
    /// [`TurError::MemoryAllocation`] before any work is enqueued, leaving every request
    /// unchanged. Per-request execution errors are reported by `finish`.
    pub fn submit(&mut self, plan: PackedExecution) -> Result<ExecutionRequirements> {
        if self.pending.is_some() {
            return Err(invalid("one batch is already in flight"));
        }
        if self.quarantine.retry() != 0 {
            return Err(invalid("retired submissions still await device completion"));
        }
        // Nothing is in flight: every released id is safe to reuse.
        self.pool.reclaim();
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
        let mut demand = Demand::default();
        for sequence in &plan.sequences {
            sequence
                .state
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("request revision overflow"))?;
            demand = demand
                + self.demand(sequence.state, sequence.tokens.len(), sequence.speculative)?;
        }
        let usage = self.pool.usage()?;
        if !demand.fits(&usage) {
            return Err(TurError::MemoryAllocation(format!(
                "pass needs {} K/V pages and {} recurrent slots; {} and {} are free",
                demand.pages,
                demand.slots,
                usage.pages.free(),
                usage.slots.free()
            )));
        }
        let mut reservations = Vec::with_capacity(plan.sequences.len());
        for sequence in &plan.sequences {
            reservations.push(
                self.request(sequence.state)?
                    .snapshot
                    .model
                    .reserve(sequence.tokens.len(), boundaries(sequence.speculative))?,
            );
        }
        requirements.state_update_bytes = reservations.iter().try_fold(0usize, |total, r| {
            total
                .checked_add(r.bytes()?)
                .ok_or_else(|| invalid("batch state update overflow"))
        })?;
        let bindings: Vec<_> = reservations.iter().map(Reservation::binding).collect();
        requirements.workspace_bytes = self.model.workspace_bytes(&PackedForward {
            tokens: &plan.tokens,
            positions: &plan.positions,
            sequences: &forwards(&plan, &bindings, self.draft_head),
            pool: &self.pool,
        })?;
        requirements.total_bytes()?;
        let mut completion = Completion::new(
            self.device.clone(),
            &self.quarantine,
            Batch {
                model: Arc::clone(&self.model),
                plan,
                reservations,
                logits: Vec::new(),
                hidden: Vec::new(),
                placement: Vec::new(),
                host: None,
                pairs: None,
                device_rows: None,
                error: None,
            },
        );
        let batch = completion
            .resources_mut()
            .ok_or_else(|| invalid("missing submission resources"))?;
        // Copy-on-write tails are the first work of the pass, after every lease succeeded.
        let prepared = batch
            .reservations
            .iter_mut()
            .try_for_each(Reservation::prepare);
        let result = prepared.and_then(|()| {
            batch.model.forward(&PackedForward {
                tokens: &batch.plan.tokens,
                positions: &batch.plan.positions,
                sequences: &forwards(&batch.plan, &bindings, self.draft_head),
                pool: &self.pool,
            })
        });
        match result {
            Ok(steps) if steps.len() == batch.plan.sequences.len() => {
                for (sequence, step) in batch.plan.sequences.iter().zip(steps) {
                    batch
                        .hidden
                        .push(step.as_ref().ok().and_then(|step| step.hidden.clone()));
                    batch.logits.push(step.and_then(|step| {
                        validate_logits(
                            &step.logits,
                            sequence.outputs.len(),
                            self.description.vocab_size,
                            &self.device,
                        )?;
                        Ok(step.logits)
                    }));
                }
            }
            Ok(_) => batch.error = Some(invalid("model returned wrong number of sequences")),
            Err(error) => batch.error = Some(error),
        }
        if batch.error.is_none()
            && let Err(error) = self.gather(batch)
        {
            batch.error = Some(error);
        }
        self.pending_handles = batch
            .plan
            .sequences
            .iter()
            .map(|s| s.state.request)
            .collect();
        // Commit the encoded tail now and copy the host rows and device decisions as
        // part of the fence; without a notifier `finish` fences and copies directly.
        let output = completion
            .resources_mut()
            .map(Batch::readback)
            .transpose()?
            .flatten();
        if let Err(error) = completion.notify(&self.notifier, output) {
            tracing::warn!("fencing on the execution thread: {error}");
        }
        self.pending = Some(completion);
        Ok(requirements)
    }

    /// Sample every row on the host instead (`false`), e.g. to compare paths. Both draw
    /// from the same distribution with the same per-request stream.
    pub fn set_device_sampling(&mut self, enabled: bool) {
        self.device_sampling = enabled;
    }

    /// Return every pass's final hidden states with its results, for the model's draft
    /// head ([`Self::draft`]).
    pub fn enable_draft_head(&mut self) {
        self.draft_head = true;
    }

    /// Run the model's draft head between passes over each request's newest committed
    /// positions, writing the head's own state layer, and return its chained greedy drafts
    /// where requested (empty otherwise). Requests fail independently; the pages a
    /// request's positions occupy must not be shared with a snapshot. Waits for the device
    /// only to read drafts.
    pub fn draft(&mut self, requests: &[DraftRequest]) -> Result<Vec<Result<Vec<Draft>>>> {
        if self.pending.is_some() {
            return Err(invalid("the draft head runs between passes"));
        }
        let mut results: Vec<Option<Result<Vec<Draft>>>> =
            requests.iter().map(|_| None).collect();
        let (mut tokens, mut positions, mut live, mut states) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (index, request) in requests.iter().enumerate() {
            let state = self.request(request.handle).and_then(|r| {
                let position = r.snapshot.model.position();
                if request.start.checked_add(request.tokens.len()) != Some(position) {
                    return Err(invalid(
                        "draft positions must end at the committed position",
                    ));
                }
                r.snapshot.model.side(request.start, request.tokens.len())
            });
            match state {
                Ok(state) => {
                    let first = tokens.len();
                    tokens.extend_from_slice(&request.tokens);
                    positions.extend(request.start..request.start + request.tokens.len());
                    live.push((index, first..tokens.len()));
                    states.push(state);
                }
                Err(error) => results[index] = Some(Err(error)),
            }
        }
        if !live.is_empty() {
            let sequences: Vec<_> = live
                .iter()
                .zip(&states)
                .map(|((index, range), state)| DraftSequence {
                    tokens: range.clone(),
                    state,
                    hidden: &requests[*index].hidden,
                    drafts: requests[*index].drafts,
                })
                .collect();
            let outputs = self.model.draft(&DraftForward {
                tokens: &tokens,
                positions: &positions,
                sequences: &sequences,
                pool: &self.pool,
            })?;
            if outputs.len() != live.len() {
                return Err(invalid("draft head returned wrong number of sequences"));
            }
            // Logit tensors, and how many rows they hold: a chain returns several.
            let (mut rows, mut total) = (Vec::new(), 0);
            let mut wanted = Vec::with_capacity(live.len());
            for ((index, _), output) in live.iter().zip(outputs) {
                match output {
                    Ok(Some(logits)) => {
                        let count = logits.dim(0)?;
                        if count > requests[*index].drafts {
                            return Err(invalid("draft head returned too many drafts"));
                        }
                        wanted.push((*index, total..total + count));
                        total += count;
                        rows.push(logits);
                    }
                    Ok(None) => wanted.push((*index, 0..0)),
                    Err(error) => results[*index] = Some(Err(error)),
                }
            }
            let (drafts, mass) = if rows.is_empty() {
                // Nothing waits: later passes, and the reuse of any page released meanwhile,
                // are encoded after these writes on the same queue.
                (Vec::new(), Vec::new())
            } else {
                let rows = Tensor::cat(&rows, 0)?;
                // Σ exp(logit - max): the draft's probability is its reciprocal.
                let mass = rows
                    .broadcast_sub(&rows.max_keepdim(1)?)?
                    .exp()?
                    .sum_keepdim(1)?;
                let drafts = self.argmax(&rows)?;
                (drafts, mass.flatten_all()?.to_vec1::<f32>()?)
            };
            for (index, range) in wanted {
                results[index] = Some(
                    range
                        .map(|row| match (drafts.get(row), mass.get(row)) {
                            (Some(&token), Some(&mass)) => Ok(Draft {
                                token,
                                confidence: mass.recip(),
                            }),
                            _ => Err(invalid("missing draft row")),
                        })
                        .collect(),
                );
            }
        }
        Ok(results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(invalid("draft request was not executed"))))
            .collect())
    }

    /// First maximum of each F32 row, on the device when kernels exist. Reading the result
    /// waits for all work enqueued before it.
    fn argmax(&self, rows: &Tensor) -> Result<Vec<u32>> {
        #[cfg(feature = "metal")]
        if let Some(kernels) = &self.kernels {
            let pairs = kernels.argmax(rows)?.to_vec2::<f32>()?;
            return pairs
                .iter()
                .map(|pair| match pair.as_slice() {
                    [token, valid] if *valid > 0.5 => Ok(*token as u32),
                    _ => Err(invalid("draft logits are not finite")),
                })
                .collect();
        }
        rows.argmax_keepdim(1)?
            .flatten_all()?
            .to_vec1::<u32>()
            .map_err(Into::into)
    }

    /// Whether rows can be penalized and sampled on the device.
    fn has_kernels(&self) -> bool {
        #[cfg(feature = "metal")]
        return self.device_sampling && self.kernels.is_some();
        #[cfg(not(feature = "metal"))]
        false
    }

    /// Enqueue, before the fence, one concatenation of every row the host samples and the
    /// device's decisions for every other row, so `finish` makes at most two host copies.
    /// The host samples without kernels, and a grammar's speculative or multi-output
    /// sequences, whose later masks depend on tokens sampled within the pass.
    fn gather(&mut self, batch: &mut Batch<M>) -> Result<()> {
        let mut host = Vec::new();
        let mut host_rows = 0;
        // Device-decided sequences: (index, logits, plans), greedy first.
        let mut device: Vec<(usize, Tensor, Vec<DeviceRow>)> = Vec::new();
        batch.placement = vec![Placement::None; batch.logits.len()];
        let kernels = self.has_kernels();
        for (index, sequence) in batch.plan.sequences.iter().enumerate() {
            let logits = match batch.logits.get(index) {
                Some(Ok(logits)) => logits.clone(),
                _ => continue,
            };
            let Some(request) = self.requests.get_mut(&sequence.state.request) else {
                continue;
            };
            if sequence.outputs.is_empty() {
                continue;
            }
            let sampling = &mut request.snapshot.sampling;
            let single = !sequence.speculative && sequence.outputs.len() == 1;
            if !kernels || (sampling.guidance.is_some() && !single) {
                batch.placement[index] = Placement::Host(host_rows);
                host_rows += sequence.outputs.len();
                host.push(logits);
                continue;
            }
            let mut plans = Vec::with_capacity(sequence.outputs.len());
            let mut stopped = false;
            for (offset, &position) in sequence.outputs.iter().enumerate() {
                let plan = sampling.device_plan(
                    drafts(sequence, &batch.plan.tokens, position),
                    offset as u64,
                );
                match plan {
                    Ok(DevicePlan::Row(row)) => plans.push(row),
                    Ok(DevicePlan::Stopped) => stopped = true,
                    Err(error) => {
                        // A grammar failure fails only its own request.
                        batch.logits[index] = Err(error);
                        stopped = false;
                        plans.clear();
                        break;
                    }
                }
            }
            if stopped {
                batch.placement[index] = Placement::Stopped;
            } else if !plans.is_empty() {
                device.push((index, logits, plans));
            }
        }
        if !host.is_empty() {
            batch.host = Some(Tensor::cat(&host, 0)?);
        }
        if !device.is_empty() {
            self.decide(batch, device)?;
        }
        let device_rows = batch.plan.sequences.iter().zip(&batch.placement);
        let device_rows = device_rows
            .filter(|(_, p)| matches!(p, Placement::Device(_)))
            .map(|(s, _)| s.outputs.len() as u64)
            .sum::<u64>();
        self.stats.device_rows += device_rows;
        self.stats.host_rows += host_rows as u64;
        Ok(())
    }

    /// Decide every device row of the pass with one penalty, one mask and one argmax or
    /// sampling dispatch each. Rows are ordered greedy first.
    #[cfg(feature = "metal")]
    fn decide(
        &self,
        batch: &mut Batch<M>,
        mut device: Vec<(usize, Tensor, Vec<DeviceRow>)>,
    ) -> Result<()> {
        let kernels = self
            .kernels
            .as_ref()
            .ok_or_else(|| invalid("device sampling without kernels"))?;
        device.sort_by_key(|(_, _, plans)| !plans.first().is_some_and(|row| row.greedy));
        let (mut tensors, mut rows) = (Vec::new(), Vec::new());
        for (index, logits, plans) in device {
            batch.placement[index] = Placement::Device(rows.len());
            rows.extend(plans);
            tensors.push(logits);
        }
        // Penalties and masks write in place: never into the model's own output, which
        // results still expose. (`Tensor::copy` shares storage on Metal.)
        let modifies = rows
            .iter()
            .any(|row| row.mask.is_some() || row.penalties.is_some());
        let all = match tensors.as_slice() {
            [only] if modifies => only.force_contiguous()?,
            _ => Tensor::cat(&tensors, 0)?,
        };
        batch.pairs = Some(decide_rows(kernels, &all, &rows)?);
        batch.device_rows = Some(all);
        Ok(())
    }

    #[cfg(not(feature = "metal"))]
    #[allow(
        clippy::unused_self,
        clippy::needless_pass_by_value,
        reason = "without Metal every row stays on the host"
    )]
    fn decide(&self, _: &mut Batch<M>, _: Vec<(usize, Tensor, Vec<DeviceRow>)>) -> Result<()> {
        Err(invalid("device sampling without kernels"))
    }

    /// Sample the next output of an idle request from `logits` (`[1, vocab]` F32 on this
    /// engine's device) as if a pass had produced them, with the request's penalties,
    /// grammar and RNG stream, and publish it without consuming input. A prefix snapshot
    /// taken at the end of a prompt keeps that prompt's last logits, so an exact repeat
    /// needs no model pass. Fences once; fails while a batch is in flight.
    pub fn sample_logits(
        &mut self,
        handle: StateHandle,
        logits: &Tensor,
    ) -> Result<SequenceCommit> {
        if self.pending.is_some() {
            return Err(invalid(
                "cannot sample cached logits while a batch is in flight",
            ));
        }
        self.request(handle)?;
        validate_logits(logits, 1, self.description.vocab_size, &self.device)?;
        let revision = handle
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("request revision overflow"))?;
        #[cfg(feature = "metal")]
        let kernels = self.kernels.clone().filter(|_| self.device_sampling);
        let request = self
            .requests
            .get_mut(&handle.request)
            .ok_or_else(|| invalid("request disappeared"))?;
        let sampling = &mut request.snapshot.sampling;
        let mark = sampling.mark();
        let penalties = sampling.penalty_entries(&[]);
        let host;
        #[cfg_attr(
            not(feature = "metal"),
            allow(unused_labels, reason = "without Metal every row stays on the host")
        )]
        let row = 'row: {
            #[cfg(feature = "metal")]
            if let Some(kernels) = &kernels {
                match sampling.device_plan(&[], 0) {
                    Ok(DevicePlan::Stopped) => break 'row Ok(Row::Stopped),
                    Ok(DevicePlan::Row(plan)) => {
                        let decided = (|| -> Result<Row<'static>> {
                            let all = logits.force_contiguous()?;
                            let pair = decide_rows(kernels, &all, &[plan])?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            let (token, status) = match pair.as_slice() {
                                &[token, word] => (token as u32, SampleStatus::from_word(word)),
                                _ => return Err(invalid("device decision without a pair")),
                            };
                            Ok(if status == SampleStatus::Unconverged {
                                Row::Penalized(all.flatten_all()?.to_vec1::<f32>()?)
                            } else {
                                Row::Device { token, status }
                            })
                        })();
                        break 'row decided;
                    }
                    Err(error) => break 'row Err(error),
                }
            }
            host = logits.flatten_all()?.to_vec1::<f32>()?;
            Ok(Row::Logits(&host))
        };
        let sampled = row.and_then(|row| sampling.sample(row, penalties.as_ref()));
        let token = match sampled {
            Ok(token) => token,
            Err(error) => {
                sampling.rollback(mark);
                return Err(error);
            }
        };
        request.handle.revision = revision;
        Ok(SequenceCommit {
            handle: request.handle,
            accepted_tokens: 0,
            outputs: token
                .map(|token| TokenOutput {
                    request: handle.request,
                    packed_position: 0,
                    token,
                })
                .into_iter()
                .collect(),
            finished: request.snapshot.sampling.finished,
        })
    }

    /// Fence the pending batch, then sample and publish each request independently.
    /// A fence error keeps the batch pending for retry; a whole-batch model error is
    /// returned once and leaves every request unchanged.
    pub fn finish(&mut self) -> Result<Vec<SequenceResult>> {
        let completion = self
            .pending
            .as_mut()
            .ok_or_else(|| invalid("no batch in flight"))?;
        if let Err(error) = completion.wait() {
            // A retried fence can succeed without the work having run correctly, so
            // the retry must not publish this batch's outputs.
            if let Some(batch) = completion.resources_mut() {
                batch
                    .error
                    .get_or_insert_with(|| invalid(format!("device completion failed: {error}")));
            }
            return Err(error);
        }
        self.completed_at = completion.completed_at().or_else(|| Some(Instant::now()));
        let copied = completion.take_copy();
        let batch = completion
            .take()
            .ok_or_else(|| invalid("incomplete submission"))?;
        self.pending = None;
        self.pending_handles.clear();
        let Batch {
            plan,
            reservations,
            logits,
            hidden,
            placement,
            host,
            pairs,
            device_rows,
            error,
            ..
        } = batch;
        if let Some(error) = error {
            self.requests.retain(|_, request| !request.cancelled);
            drop(reservations);
            self.pool.reclaim();
            return Err(error);
        }
        let vocab = self.description.vocab_size;
        // One host copy of the full rows then the device decisions, made with the fence or
        // here when no notifier ran.
        let host_len = host.as_ref().map_or(0, Tensor::elem_count);
        let mut full = match copied {
            Some(copied) => copied,
            None => [host, pairs]
                .into_iter()
                .flatten()
                .map(|t| Ok(t.flatten_all()?.to_vec1::<f32>()?))
                .collect::<Result<Vec<_>>>()?
                .concat(),
        };
        if full.len() < host_len {
            return Err(invalid("host copy is shorter than the logit rows"));
        }
        let pairs = full.split_off(host_len);
        let outside = || invalid("logit rows outside the copied batch");
        let mut fallbacks = 0;
        let mut results = Vec::with_capacity(plan.sequences.len());
        let mut hidden = hidden.into_iter();
        for (((sequence, reservation), logits), placement) in plan
            .sequences
            .iter()
            .zip(&reservations)
            .zip(logits)
            .zip(placement)
        {
            let hidden = hidden.next().flatten();
            let n = sequence.outputs.len();
            let last = logits
                .as_ref()
                .ok()
                .filter(|_| n > 0)
                .and_then(|t| t.narrow(0, n - 1, 1).ok());
            let rows = logits.and_then(|_| match placement {
                Placement::None => Ok(Vec::new()),
                Placement::Stopped => Ok((0..n).map(|_| Row::Stopped).collect()),
                Placement::Host(first) => Ok(full
                    .get(first * vocab..(first + n) * vocab)
                    .ok_or_else(outside)?
                    .chunks_exact(vocab.max(1))
                    .map(Row::Logits)
                    .collect()),
                Placement::Device(first) => pairs
                    .get(first * 2..(first + n) * 2)
                    .ok_or_else(outside)?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .enumerate()
                    .map(|(offset, &[token, word])| {
                        let status = SampleStatus::from_word(word);
                        if status != SampleStatus::Unconverged {
                            return Ok(Row::Device {
                                token: token as u32,
                                status,
                            });
                        }
                        let row = device_rows
                            .as_ref()
                            .ok_or_else(outside)?
                            .get(first + offset)?;
                        fallbacks += 1;
                        Ok(Row::Penalized(row.to_vec1::<f32>()?))
                    })
                    .collect(),
            });
            let id = sequence.state.request;
            let Some(request) = self.requests.get_mut(&id).filter(|r| !r.cancelled) else {
                continue;
            };
            let outcome =
                rows.and_then(|rows| publish(request, sequence, &plan.tokens, reservation, rows));
            results.push(SequenceResult {
                request: id,
                outcome,
                logits: last,
                hidden,
            });
        }
        self.stats.fallback_rows += fallbacks;
        self.requests.retain(|_, request| !request.cancelled);
        // Unaccepted pages, superseded slots and cancelled states retire here; their work
        // has completed, and publishing enqueued none.
        drop(reservations);
        self.pool.reclaim();
        Ok(results)
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use tur_core::execution::{
        KvFormat, KvLayout, LayerStateKind, Modality, ModelIdentity, SequenceDescriptor,
    };

    const POOL: PoolConfig = PoolConfig {
        pages: 2,
        window_pages: 2,
        recurrent_slots: 1,
    };

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
                    window: None,
                }],
                kv: KvLayout::new(2, KvFormat::F32),
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
        let mut state = SamplingState::new(1, SamplingOptions::default(), None);
        state.draws = u64::MAX;
        assert!(state.sample(Row::Logits(&[1.0, 2.0]), None).is_err());
        assert_eq!(state.draws, u64::MAX);
        assert_eq!(state.tokens, [] as [u32; 0]);
    }

    #[test]
    fn test_revision_overflow_rejects_submission_and_restore_without_changes() -> Result<()> {
        let mut engine = HybridEngine::new(Arc::new(UnusedModel), Device::Cpu, POOL)?;
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
        let mut engine = HybridEngine::new(Arc::new(UnusedModel), Device::Cpu, POOL)?;
        for options in [
            SamplingOptions {
                temperature: Some(f64::NAN),
                top_p: None,
                ..Default::default()
            },
            SamplingOptions {
                temperature: Some(-1.0),
                top_p: None,
                ..Default::default()
            },
            SamplingOptions {
                temperature: None,
                top_p: Some(0.0),
                ..Default::default()
            },
            SamplingOptions {
                temperature: None,
                top_p: Some(1.1),
                ..Default::default()
            },
        ] {
            assert!(engine.allocate(1, options, None).is_err());
        }
        assert!(engine.requests.is_empty());
        Ok(())
    }

    #[test]
    fn test_greedy_fast_path_keeps_the_first_maximum_and_rejects_nonfinite_rows() {
        let mut state = SamplingState::new(3, SamplingOptions::default(), None);
        state.draws = 0;
        assert!(matches!(
            state.sample(Row::Logits(&[1.0, 3.0, 3.0, -2.0]), None),
            Ok(Some(1))
        ));
        assert!(matches!(
            state.sample(Row::Logits(&[-5.0, -5.0]), None),
            Ok(Some(0))
        ));
        assert!(
            state
                .sample(Row::Logits(&[1.0, f32::INFINITY]), None)
                .is_err()
        );
        assert_eq!((state.draws, state.tokens.as_slice()), (2, &[1, 0][..]));
    }

    #[test]
    fn test_penalty_history_counts_samples_and_rolls_back() -> Result<()> {
        let options = SamplingOptions {
            frequency_penalty: Some(1.5),
            repetition_penalty: Some(2.0),
            ..Default::default()
        };
        let mut state = SamplingState::new(3, options, None);
        let row = [4.0, 3.9, -1.0];
        assert_eq!(
            state.sample(Row::Logits(&row), state.penalty_entries(&[]).as_ref())?,
            Some(0)
        );
        // Token 0 is now penalized twice over: 4 / 2 - 1.5 = 0.5 < 3.9.
        let mark = state.mark();
        assert_eq!(
            state.sample(Row::Logits(&row), state.penalty_entries(&[]).as_ref())?,
            Some(1)
        );
        let entries = |state: &SamplingState| {
            let mut e = state
                .penalty_entries(&[])
                .map(|(_, e)| e)
                .unwrap_or_default();
            e.sort_by_key(|e| e.token);
            e
        };
        assert_eq!(
            entries(&state),
            [
                PenaltyEntry { token: 0, count: 1 },
                PenaltyEntry { token: 1, count: 1 }
            ]
        );
        state.rollback(mark);
        assert_eq!(entries(&state), [PenaltyEntry { token: 0, count: 1 }]);
        // Drafts count as sampled for the output they precede; prompt context adds
        // unsampled entries for the repetition penalty.
        let drafted = state
            .penalty_entries(&[2, 2])
            .map(|(_, e)| e)
            .unwrap_or_default();
        assert!(drafted.contains(&PenaltyEntry { token: 2, count: 2 }));
        if let Some(history) = &mut state.history {
            history.context.insert(7);
        }
        assert!(entries(&state).contains(&PenaltyEntry { token: 7, count: 0 }));
        Ok(())
    }

    #[test]
    fn test_rollback_restores_history_and_draws() {
        let mut state = SamplingState::new(3, SamplingOptions::default(), None);
        state.draws = 0;
        let mark = state.mark();
        assert!(matches!(
            state.sample(Row::Logits(&[0.0, 1.0]), None),
            Ok(Some(1))
        ));
        assert!(state.sample(Row::Logits(&[f32::NAN, 1.0]), None).is_err());
        state.rollback(mark);
        assert_eq!(state.draws, 0);
        assert_eq!(state.tokens, [] as [u32; 0]);
    }
}
