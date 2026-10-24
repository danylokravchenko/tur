#![forbid(unsafe_code)]
//! Chat admission and scheduled packed execution over the single-owner hybrid engine.

mod config;
mod drafting;
mod execution;
mod requests;
mod scheduling;

pub use config::PipelineConfig;
pub use requests::{BatchStepOutput, GenerationRequest, GenerationStats, PassTiming};

use super::{
    guidance::ParserFactory,
    memory::{BudgetLease, MemoryPlan},
    prefix::{PrefixCache, PrefixKeys, PrefixStats},
    scheduler::{Candidate, Chunk, Scheduler},
    speculation::{
        Acceptance, Calibration, DraftCosts, DraftSource, HeadCosts, NgramIndex, SpeculationConfig,
        SpeculationStats,
    },
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
use tur_models::{
    backend::{chat_template::ChatTemplate, tools::ToolDefinition},
    models::qwen3_5::{GenerationDefaults, Qwen35},
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
    tokens: Vec<u32>,
    prompt_tokens: usize,
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
    max_tokens: usize,
    arrival: Instant,
    deadline: Option<Instant>,
    first_token: Option<Duration>,
    last_run: u64,
    preemptions: usize,
    tools: Vec<ToolDefinition>,
    /// Grammar-constrained: never drafted, so masks stay on the device.
    constrained: bool,
    /// Greedy decoding: its drafts' acceptance is calibrated separately from sampling's.
    greedy: bool,
    ngrams: NgramIndex,
    acceptance: Acceptance,
    /// Positions whose draft-head keys and values are written, and the head's draft for
    /// the token after the pending one, valid while `mtp_done == consumed`.
    mtp_done: usize,
    mtp_draft: Option<Draft>,
    mtp_acceptance: Acceptance,
    /// Confidence of the head's draft in the pass in flight.
    mtp_verifying: Option<f32>,
    drafted: usize,
    accepted_drafts: usize,
}

impl Pending {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_tokens
    }

    /// Scheduling view. Prefill chunks end at or before `boundary` (the next snapshot
    /// position), so recurrent state exists at every aligned position.
    fn candidate(&self, id: Uuid, boundary: Option<usize>) -> Candidate {
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
            decoding: self.running && unconsumed == 1 && self.generated() > 0,
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
    engine: HybridEngine<Qwen35>,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    defaults: GenerationDefaults,
    max_position: usize,
    scheduler: Scheduler,
    plan: MemoryPlan,
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
        mut on_token: impl FnMut(&str),
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
