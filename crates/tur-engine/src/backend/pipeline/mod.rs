#![forbid(unsafe_code)]
//! Chat admission and scheduled packed execution over the single-owner hybrid engine.

mod audio;
mod config;
mod drafting;
mod execution;
mod requests;
mod scheduling;

pub use config::PipelineConfig;
pub use requests::{BatchStepOutput, GenerationRequest, GenerationStats, PassTiming, TokenLogprob};

use super::{
    guidance::ParserFactory,
    harmony::Harmony,
    memory::{BudgetLease, MemoryPlan},
    prefix::{PrefixCache, PrefixKeys, PrefixStats},
    reasoning::ReasoningSplit,
    scheduler::{Candidate, Chunk, Scheduler},
    speculation::{
        Acceptance, Calibration, DraftCosts, DraftSource, HeadCosts, NgramIndex, SpeculationConfig,
        SpeculationStats,
    },
    stop::StopMatcher,
    tokenizer::TokenOutputStream,
};
use crate::{
    Result,
    hybrid::{Draft, HybridEngine, PoolUsage, SamplingStats},
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;
use tur_core::execution::{StateHandle, invalid};
use tur_models::models::GenerationDefaults;
use tur_models::{
    backend::{chat_template::ChatTemplate, tools::ToolDefinition},
    models::TextModel,
};
use uuid::Uuid;

/// A submitted pass awaiting `finish_step`.
struct InFlight {
    chunks: Vec<Chunk>,
    /// Speculative drafts appended to each chunk's single decode token, and their source.
    drafts: Vec<Vec<u32>>,
    sources: Vec<DraftSource>,
    /// Decoding sequences of the pass; when they are all its chunks, the pass's cost
    /// feeds the draft cost fit.
    decodes: usize,
    context: usize,
    started: Instant,
    submitted: Instant,
    timing: PassTiming,
}

struct Pending {
    handle: StateHandle,
    /// The prompt, then every sampled token; the first `consumed` are in model state.
    /// Audio positions hold clip identities (see [`audio`]).
    tokens: Vec<u32>,
    /// Encoded clips of the prompt, kept until completion so recomputation reuses them.
    audio: Vec<audio::AudioSpan>,
    prompt_tokens: usize,
    /// Predicted last shared token before the template rewrites this turn as history.
    turn_boundary: Option<usize>,
    consumed: usize,
    /// Original committed pass ends. Replay never splits or combines these chunks:
    /// doing so changes the rounding points of narrow recurrent state.
    boundaries: Vec<usize>,
    /// Tokens first restored from a prefix snapshot; no pass boundaries exist below.
    replay_floor: usize,
    cached_tokens: usize,
    keys: PrefixKeys,
    /// Snapshots inserted when a lookup last missed; until more arrive it would miss again.
    prefix_checked: Option<u64>,
    /// Holds model state. Preemption clears it and `consumed`, so the request recomputes.
    running: bool,
    stream: TokenOutputStream,
    /// Splits output by channel, for models that speak Harmony.
    harmony: Option<Harmony>,
    /// Ends the answer at the request's stop sequences.
    stop: StopMatcher,
    max_tokens: usize,
    arrival: Instant,
    deadline: Option<Instant>,
    first_token: Option<Duration>,
    last_run: u64,
    preemptions: usize,
    tools: Vec<ToolDefinition>,
    /// Names tokens for the log probabilities it reports, when the request asked for them.
    logprob_tokens: Option<Arc<ParserFactory>>,
    logprobs: Vec<TokenLogprob>,
    /// Separates reasoning from a structured answer, for models whose channels don't.
    split: Option<ReasoningSplit>,
    /// Greedy decoding: its drafts' acceptance is calibrated separately from sampling's.
    greedy: bool,
    ngrams: NgramIndex,
    acceptance: Acceptance,
    /// Positions whose draft-head keys and values are written, and the head's chained
    /// drafts for the tokens after the pending one, valid while `mtp_done == consumed`.
    mtp_done: usize,
    mtp_drafts: Vec<Draft>,
    mtp_acceptance: Acceptance,
    /// Confidences of the head's drafts verified by the pass in flight.
    mtp_verifying: Vec<f32>,
    drafted: usize,
    accepted_drafts: usize,
}

impl Pending {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_tokens
    }

    /// Scheduling view. Prefill chunks stop at aligned and predicted turn boundaries,
    /// so each snapshot has the recurrent state written at exactly its position.
    fn candidate(&self, id: Uuid, boundary: Option<usize>) -> Candidate {
        let boundary = boundary
            .into_iter()
            .chain(self.turn_boundary.filter(|&end| end > self.consumed))
            .min();
        let unconsumed = self.tokens.len() - self.consumed;
        let next = self.boundaries.partition_point(|&end| end <= self.consumed);
        let replay = self
            .boundaries
            .get(next)
            .filter(|_| self.consumed >= self.replay_floor)
            .map(|end| end - self.consumed);
        // Below the first restore, chunks also stop there, where replay boundaries begin.
        let end = boundary.map(|end| {
            if self.consumed < self.replay_floor {
                end.min(self.replay_floor)
            } else {
                end
            }
        });
        let aligned = end.map_or(unconsumed, |end| unconsumed.min(end - self.consumed));
        Candidate {
            id,
            pending: replay.unwrap_or(aligned),
            minimum: replay.unwrap_or(1),
            // Past the prompt, every unconsumed token is an output: the newest sampled one, or
            // a run its grammar forced.
            decoding: self.running && self.generated() > 0 && self.consumed >= self.prompt_tokens,
            running: self.running,
            context: self.consumed,
            arrival: self.arrival,
            last_run: self.last_run,
        }
    }
}

/// Serving adapter. Every iteration the scheduler selects one packed pass of decodes and
/// prefill chunks that fits the state pool; see [`super::scheduler`]. Under memory
/// pressure the newest running request is preempted and later recomputes its prompt and
/// generated tokens with its RNG and grammar state intact and its original rounding
/// boundaries preserved. Batch-dependent floating-point paths may still affect logits.
/// A request that cannot fit even alone fails with [`crate::TurError::MemoryAllocation`].
pub struct InferencePipeline {
    engine: HybridEngine<TextModel>,
    /// The engine's model, shared to encode prompt audio at submission.
    model: Arc<TextModel>,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    defaults: GenerationDefaults,
    scheduler: Scheduler,
    plan: MemoryPlan,
    device_sampling: bool,
    _lease: BudgetLease,
    pending: HashMap<Uuid, Pending>,
    prefix: PrefixCache,
    /// Waiting requests left unscheduled this pass for a producer of a shared prefix.
    awaiting: HashSet<Uuid>,
    in_flight: Option<InFlight>,
    /// Outcomes produced before or without a pass, returned by the next `finish_step`.
    staged: BatchStepOutput,
    /// Grammar parser factory over the tokenizer, built on the first constrained request.
    guidance: Option<Arc<ParserFactory>>,
    speculation: SpeculationConfig,
    draft_costs: DraftCosts,
    head_costs: HeadCosts,
    /// Consecutive draft-head passes whose drafts were skipped as unprofitable, and the
    /// count after which one explores anyway.
    head_skips: usize,
    head_explore: usize,
    /// Pipeline-wide acceptance, the starting estimate of new requests, per source.
    acceptance: Acceptance,
    mtp_acceptance: Acceptance,
    /// Draft-head acceptance by confidence, for sampled and greedy requests.
    calibration: [Calibration; 2],
    speculation_stats: SpeculationStats,
}

impl InferencePipeline {
    /// Requests with allocated sampling state, running or waiting.
    pub fn active_request_count(&self) -> usize {
        self.pending.len()
    }

    /// Requests waiting for admission or recomputation; they hold no pool memory.
    pub fn waiting_request_count(&self) -> usize {
        self.pending.values().filter(|p| !p.running).count()
    }

    /// Planned resident memory and the state pool it sized.
    pub fn memory_plan(&self) -> &MemoryPlan {
        &self.plan
    }

    /// Where output rows have been sampled: on the device or the host.
    pub fn sampling_stats(&self) -> SamplingStats {
        self.engine.sampling_stats()
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// The model behind this pipeline, for its identity and kernels.
    pub fn model(&self) -> &TextModel {
        &self.model
    }

    /// Width of the model's output.
    pub fn vocab_size(&self) -> usize {
        self.engine.pool().description().vocab_size
    }

    /// Whether unconstrained rows are sampled on the device when the kernels allow it.
    pub fn device_sampling(&self) -> bool {
        self.device_sampling
    }

    /// Whether requests may draft speculatively.
    pub fn speculation_enabled(&self) -> bool {
        self.speculation.enabled()
    }

    /// Prefix reuse counters.
    pub fn prefix_stats(&self) -> PrefixStats {
        self.prefix.stats()
    }

    /// Speculative decoding counters.
    pub fn speculation_stats(&self) -> SpeculationStats {
        self.speculation_stats
    }

    /// Measured device cost of decode-only passes against their drafted tokens.
    pub fn draft_costs(&self) -> &DraftCosts {
        &self.draft_costs
    }

    /// Drop every prefix snapshot; their pages and slots return once no request shares
    /// them.
    pub fn clear_prefix_cache(&mut self) {
        self.prefix.clear();
    }

    /// Current state pool occupancy, after reclaiming released pages and slots.
    pub fn memory(&mut self) -> Result<PoolUsage> {
        self.engine.memory()
    }

    /// Run one scheduled packed pass to completion: [`Self::begin_step`] then
    /// [`Self::finish_step`].
    pub fn step_streaming(&mut self) -> Result<BatchStepOutput> {
        self.begin_step()?;
        self.finish_step()
    }

    /// Run one request to completion and stream decoded text to a caller-owned callback.
    pub fn run(
        &mut self,
        request: &GenerationRequest,
        on_token: impl FnMut(&str),
    ) -> Result<GenerationStats> {
        self.run_channels(request, on_token, |_| {})
    }

    /// [`Self::run`], also streaming the reasoning of models that separate it.
    pub fn run_channels(
        &mut self,
        request: &GenerationRequest,
        mut on_token: impl FnMut(&str),
        mut on_reasoning: impl FnMut(&str),
    ) -> Result<GenerationStats> {
        if !self.pending.is_empty() {
            return Err(invalid("run requires an idle pipeline"));
        }
        let id = self.submit_request(request)?;
        loop {
            let step = match self.step_streaming() {
                Ok(step) => step,
                Err(error) => {
                    self.cancel(id)?;
                    return Err(error);
                }
            };
            for (_, text) in step.new_reasoning {
                on_reasoning(&text);
            }
            for (_, text) in step.new_tokens {
                on_token(&text);
            }
            for (finished, result) in step.completed {
                if finished == id {
                    return result;
                }
            }
        }
    }
}
