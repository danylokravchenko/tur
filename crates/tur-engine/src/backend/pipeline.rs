#![forbid(unsafe_code)]
//! Chat admission and scheduled packed execution over the single-owner hybrid engine.

use candle_core::Tensor;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;
use tur_core::execution::{PackedExecution, SequenceDescriptor, StateHandle, invalid};
use tur_models::{
    backend::{
        chat_template::{ChatMessage, ChatTemplate, RenderOptions, Role},
        factory::ModelFactory,
        tools::{ToolCall, ToolDefinition},
    },
    models::qwen3_5::{GenerationDefaults, Qwen35},
};
use uuid::Uuid;

use super::{
    guidance::{GuidanceControl, ParserFactory, TopLevelGrammar, build_llg_factory},
    memory::{BudgetLease, DeviceBudget, MemoryPlan, PlanInputs},
    prefix::{PrefixCache, PrefixConfig, PrefixKeys, PrefixStats},
    scheduler::{Candidate, Chunk, Decision, Scheduler, SchedulerConfig},
    speculation::{
        self, Acceptance, DraftCosts, DraftSource, HeadCosts, NgramIndex, Proposal,
        SpeculationConfig, SpeculationStats,
    },
    tokenizer::TokenOutputStream,
};
use crate::{
    Result, TurError,
    hybrid::{
        Demand, DraftRequest, HybridEngine, PoolUsage, SamplingOptions, SamplingStats,
        SequenceCommit,
    },
};

/// One text conversation. Unset sampling and thinking controls use checkpoint defaults.
pub struct GenerationRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: usize,
    pub render: RenderOptions,
    pub sampling: SamplingOptions,
    pub seed: u64,
    /// Fail the request if it has not finished this long after submission.
    pub timeout: Option<Duration>,
    /// Original submission time. The worker sets this before enqueueing so deadlines
    /// include intake waiting; direct callers may leave it unset.
    pub submitted_at: Option<Instant>,
    /// Prefix-reuse namespace: only requests with the same salt share snapshots.
    pub cache_salt: Option<String>,
    /// Constrain generation to this grammar; the request completes when it terminates.
    pub grammar: Option<TopLevelGrammar>,
}

impl GenerationRequest {
    /// Generate a reply to one user turn, using the checkpoint's generation opener.
    pub fn new(prompt: String, max_tokens: usize) -> Self {
        Self {
            messages: vec![ChatMessage::new(Role::User, prompt)],
            tools: Vec::new(),
            max_tokens,
            render: RenderOptions {
                add_generation_prompt: true,
                ..Default::default()
            },
            sampling: SamplingOptions::default(),
            seed: 299_792_458,
            timeout: None,
            submitted_at: None,
            cache_salt: None,
            grammar: None,
        }
    }
}

/// Completed generation, including EOS in the generated token count when sampled.
#[derive(Debug)]
pub struct GenerationStats {
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    /// Prompt tokens restored from a prefix snapshot when first admitted.
    pub cached_tokens: usize,
    /// From submission to completion.
    pub elapsed: Duration,
    /// From submission to the first sampled token.
    pub first_token: Duration,
    /// Times the request's state was released under memory pressure and recomputed.
    pub preemptions: usize,
    /// Speculative draft tokens verified, and how many of them the target accepted.
    pub draft_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub tool_calls: Vec<ToolCall>,
    pub generated_text: String,
    /// `stop` for EOS, `length` for the request limit.
    pub finish_reason: &'static str,
}

/// Incremental text and independent request outcomes from one packed model invocation.
#[derive(Default)]
pub struct BatchStepOutput {
    pub new_tokens: Vec<(Uuid, String)>,
    /// Sampled token IDs (including EOS) and publication times, independent of text
    /// decoding, which can buffer several tokens into one fragment.
    pub sampled_tokens: Vec<(Uuid, u32, Instant)>,
    pub completed: Vec<(Uuid, Result<GenerationStats>)>,
    /// Host and device time of the pass, if one ran.
    pub timing: Option<PassTiming>,
}

/// Where one pass's time went. Host phases are disjoint; `device` overlaps `encode`,
/// `overlap` and `wait`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassTiming {
    pub tokens: usize,
    pub sequences: usize,
    /// Speculative draft tokens among `tokens`.
    pub drafts: usize,
    /// Draft-head pass after publication, from submission to its drafts on the host.
    pub draft_head: Duration,
    /// Expiry, prefix restore, scheduling and packing.
    pub schedule: Duration,
    /// Validation, reservation and encoding.
    pub encode: Duration,
    /// From submission to `finish_step`: host work beside the running device.
    pub overlap: Duration,
    /// Blocked until the device completed.
    pub wait: Duration,
    /// After completion: logit copies, host sampling, publication and text decoding.
    pub publish: Duration,
    /// From the start of encoding to device completion.
    pub device: Duration,
    /// From the previous pass's completion to this pass's encoding: the device had no
    /// work queued. A lower bound on its idle time; work starts as buffers commit.
    pub idle_before: Option<Duration>,
}

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
    /// Greedy decoding, the regime where draft-head drafts were measured to pay.
    greedy: bool,
    ngrams: NgramIndex,
    acceptance: Acceptance,
    /// Positions whose draft-head keys and values are written, and the head's draft for
    /// the token after the pending one, valid while `mtp_done == consumed`.
    mtp_done: usize,
    mtp_draft: Option<u32>,
    mtp_acceptance: Acceptance,
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

/// Scheduling limits and the state memory budget of one pipeline.
#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    pub scheduler: SchedulerConfig,
    /// Prefix snapshots: spacing and how many are retained beside running requests.
    pub prefix: PrefixConfig,
    /// Penalize and sample unconstrained rows on the device (when the model runs on Metal
    /// kernels); otherwise every full row is copied to the host sampler.
    pub device_sampling: bool,
    /// Opt-in throughput mode using drafts from each request's own history. Tokens arrive
    /// in bursts; longer pauses can increase p95 inter-token latency. Disabled by default.
    pub speculation: SpeculationConfig,
    /// Bytes for K/V pages and recurrent state. `None` sizes the pool for
    /// `max_running` contexts of [`super::memory::DEFAULT_CONTEXT_TOKENS`] tokens, within
    /// what the device budget leaves.
    pub state_bytes: Option<usize>,
}

impl PipelineConfig {
    /// Default scheduling for `max_batch_size` running sequences and `prefill_chunk_size`
    /// prefill chunks, one retained prefix snapshot per running sequence, and a derived
    /// state budget.
    pub fn new(max_batch_size: usize, prefill_chunk_size: usize) -> Self {
        Self {
            scheduler: SchedulerConfig::new(max_batch_size, prefill_chunk_size),
            prefix: PrefixConfig {
                spacing: PrefixConfig::DEFAULT_SPACING,
                entries: max_batch_size,
                state_share: Some(PrefixConfig::DEFAULT_STATE_SHARE),
            },
            device_sampling: true,
            speculation: SpeculationConfig::DISABLED,
            state_bytes: None,
        }
    }

    /// Reproducibility mode: one request at a time in fixed `prefill_chunk_size` chunks,
    /// without prefix reuse or speculation, so a request's computation (and with its seed,
    /// its tokens) does not depend on other traffic. Batched passes can change logits in
    /// their last bits through batch-dependent kernels; this mode trades throughput to rule
    /// that out.
    pub fn reproducible(prefill_chunk_size: usize) -> Self {
        Self {
            prefix: PrefixConfig::DISABLED,
            speculation: SpeculationConfig::DISABLED,
            ..Self::new(1, prefill_chunk_size)
        }
    }
}

/// Serving adapter. Every iteration the scheduler selects one packed pass of decodes and
/// prefill chunks that fits the state pool; see [`super::scheduler`]. Under memory
/// pressure the newest running request is preempted and later recomputes its prompt and
/// generated tokens with its RNG and grammar state intact and its original rounding
/// boundaries preserved. Batch-dependent floating-point paths may still affect logits.
/// A request that cannot fit even alone fails with [`TurError::MemoryAllocation`].
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
    speculation_stats: SpeculationStats,
}

fn sampling(overrides: SamplingOptions, defaults: &GenerationDefaults) -> SamplingOptions {
    SamplingOptions {
        temperature: overrides.temperature.or(defaults.temperature),
        top_k: overrides.top_k.or(defaults.top_k),
        top_p: overrides.top_p.or(defaults.top_p),
        repetition_penalty: overrides.repetition_penalty.or(defaults.repetition_penalty),
        ..overrides
    }
}

impl InferencePipeline {
    /// Load the model as the only one on its device; see [`Self::with_budget`].
    pub fn new(factory: &ModelFactory, config: PipelineConfig) -> Result<Self> {
        Self::with_budget(factory, config, &DeviceBudget::for_device(&factory.device))
    }

    /// Load the validated model, plan its memory beside what other models leased from
    /// `budget`, and lease the plan before loading weights. Invalid limits and a plan
    /// beyond the remaining budget fail before reading weight payloads or uploading them.
    pub fn with_budget(
        factory: &ModelFactory,
        config: PipelineConfig,
        budget: &DeviceBudget,
    ) -> Result<Self> {
        let scheduler = Scheduler::new(config.scheduler)?;
        config.speculation.validate()?;
        // Narrow state rounds at ordinary pass boundaries; a multi-token verification
        // keeps F32 intermediates between drafts, so it is not the same target process.
        let config = PipelineConfig {
            speculation: if factory.recurrent == candle_core::DType::F32 {
                config.speculation
            } else {
                if config.speculation.enabled() {
                    tracing::warn!(
                        recurrent = ?factory.recurrent,
                        "speculative drafts need F32 recurrent state; using ordinary decoding"
                    );
                }
                SpeculationConfig::DISABLED
            },
            ..config
        };
        if config.speculation.mtp_enabled() && config.prefix.enabled() {
            // A snapshot at boundary B cannot hold the head's entry for B - 1: it depends on
            // the token sampled after B, which differs between requests sharing the snapshot.
            return Err(invalid(
                "native MTP drafts need prefix reuse disabled (PrefixConfig::DISABLED)",
            ));
        }
        let factory = &ModelFactory {
            mtp: config.speculation.mtp_enabled(),
            ..factory.clone()
        };
        let limits = config.scheduler;
        let draft_rows = config
            .speculation
            .max_pass_drafts
            .min(
                config
                    .speculation
                    .max_draft
                    .saturating_mul(limits.max_running),
            )
            .min(limits.max_batch_tokens.saturating_sub(1));
        let draft_slots = draft_rows;
        let (model, tokenizer, template, defaults, (plan, pool, lease)) = factory
            .create_model_with(|architecture, options| {
                let tokens = limits.max_batch_tokens;
                let sequences = limits.max_running.min(tokens);
                let (description, weights, scratch_peak) = Qwen35::memory_requirements(
                    architecture,
                    options,
                    tokens,
                    &vec![architecture.max_position(); sequences],
                )?;
                // Full model logits and their gathered copy coexist for sampled requests;
                // every drafted token adds an output row.
                let in_flight = description
                    .vocab_size
                    .checked_mul(8)
                    .and_then(|row| row.checked_add(16))
                    .and_then(|row| sequences.checked_add(draft_rows)?.checked_mul(row))
                    .and_then(|logits| logits.checked_add(tokens.checked_mul(12)?))
                    // Prompt-end snapshots keep one F32 logit row each.
                    .and_then(|bytes| {
                        bytes.checked_add(
                            description
                                .vocab_size
                                .checked_mul(4)?
                                .checked_mul(config.prefix.entries)?,
                        )
                    })
                    .ok_or_else(|| invalid("in-flight size overflow"))?;
                let in_flight = in_flight
                    .checked_add(
                        config
                            .speculation
                            .index_bytes(sequences, architecture.max_position())?,
                    )
                    .ok_or_else(|| invalid("draft index reservation overflow"))?;
                let (plan, pool) = MemoryPlan::new(
                    &description,
                    PlanInputs {
                        weights,
                        scratch_peak,
                        in_flight,
                        sequences: limits.max_running,
                        retained: config.prefix.entries,
                        retained_share: config.prefix.state_share,
                        draft_slots,
                        state_bytes: config.state_bytes,
                        limit: budget.limit(),
                        reserved: budget.leased(),
                    },
                )?;
                let lease = budget.lease(plan.resident())?;
                Ok((plan, pool, lease))
            })?;
        let max_position = model.config().max_position();
        if model.weight_bytes() != plan.weights {
            return Err(invalid(
                "loaded weights disagree with the pre-load memory plan",
            ));
        }
        let mut engine = HybridEngine::new(Arc::new(model), factory.device.clone(), pool)?;
        engine.set_device_sampling(config.device_sampling);
        if config.speculation.mtp_enabled() {
            engine.enable_draft_head();
        }
        config
            .prefix
            .validate(engine.pool().description().kv.page_tokens)?;
        Ok(Self {
            engine,
            tokenizer,
            template,
            defaults,
            max_position,
            scheduler,
            plan,
            _lease: lease,
            pending: HashMap::new(),
            prefix: PrefixCache::new(PrefixConfig {
                entries: plan.retained,
                ..config.prefix
            }),
            in_flight: None,
            staged: BatchStepOutput::default(),
            guidance: None,
            speculation: config.speculation,
            draft_costs: DraftCosts::default(),
            head_costs: HeadCosts::default(),
            head_skips: 0,
            head_explore: speculation::HEAD_EXPLORE_PASSES,
            acceptance: Acceptance::default(),
            mtp_acceptance: Acceptance::default(),
            speculation_stats: SpeculationStats::default(),
        })
    }

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

    /// Render, tokenize and queue one text conversation with independent sampling state.
    /// Fails if the waiting queue is full or the request could never fit the state pool.
    pub fn submit_request(&mut self, request: &GenerationRequest) -> Result<Uuid> {
        let arrival = request.submitted_at.unwrap_or_else(Instant::now);
        let deadline = request
            .timeout
            .map(|timeout| {
                arrival
                    .checked_add(timeout)
                    .ok_or_else(|| invalid("request deadline overflow"))
            })
            .transpose()?;
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(TurError::DeadlineExceeded);
        }
        if request.messages.is_empty() || request.max_tokens == 0 {
            return Err(invalid("messages and max_tokens must be nonempty/positive"));
        }
        let prompt = self
            .template
            .render(&request.messages, &request.tools, &request.render)?;
        let tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| TurError::Tokenizer(error.to_string()))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            return Err(invalid("chat template produced an empty prompt"));
        }
        // The final sampled token need not be consumed into model state.
        let positions = tokens
            .len()
            .checked_add(request.max_tokens - 1)
            .ok_or_else(|| invalid("request context length overflow"))?;
        if positions > self.max_position {
            return Err(invalid(
                "prompt and generation limit exceed the model context",
            ));
        }
        // A request that cannot fit the whole pool alone would never complete.
        let usage = self.engine.memory()?;
        let pages = self.engine.pool().description().pages_for(positions);
        if pages > usage.pages.capacity {
            return Err(TurError::MemoryAllocation(format!(
                "request needs {pages} K/V pages for {positions} positions; the state pool \
                 holds {}",
                usage.pages.capacity
            )));
        }
        let (waiting, waiting_tokens) = self
            .pending
            .values()
            .filter(|p| !p.running)
            .fold((0usize, 0usize), |(n, t), p| {
                (n + 1, t.saturating_add(p.tokens.len() - p.consumed))
            });
        self.scheduler
            .admit(waiting, waiting_tokens, tokens.len())?;
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(TurError::DeadlineExceeded);
        }
        let constrained = request.grammar.is_some();
        let options = sampling(request.sampling, &self.defaults);
        let guidance = request
            .grammar
            .clone()
            .map(|grammar| -> Result<GuidanceControl> {
                let factory = if let Some(factory) = &self.guidance {
                    Arc::clone(factory)
                } else {
                    let vocab = self.engine.pool().description().vocab_size;
                    let factory = build_llg_factory(self.tokenizer.clone(), Some(vocab))?;
                    self.guidance = Some(Arc::clone(&factory));
                    factory
                };
                let mut control = GuidanceControl::new(factory);
                control.activate(grammar)?;
                Ok(control)
            })
            .transpose()?;
        let handle = self.engine.allocate(request.seed, options, guidance)?;
        self.engine.add_context(handle, &tokens)?;
        let keys = self.prefix.keys(request.cache_salt.as_deref(), &tokens);
        self.pending.insert(
            handle.request,
            Pending {
                handle,
                prompt_tokens: tokens.len(),
                tokens,
                consumed: 0,
                boundaries: Vec::new(),
                replay_floor: 0,
                cached_tokens: 0,
                keys,
                prefix_checked: None,
                running: false,
                stream: TokenOutputStream::new(self.tokenizer.clone()),
                max_tokens: request.max_tokens,
                arrival,
                deadline,
                first_token: None,
                last_run: 0,
                preemptions: 0,
                tools: request.tools.clone(),
                constrained,
                greedy: options.greedy(),
                ngrams: NgramIndex::default(),
                acceptance: self.acceptance,
                mtp_done: 0,
                mtp_draft: None,
                mtp_acceptance: self.mtp_acceptance,
                drafted: 0,
                accepted_drafts: 0,
            },
        );
        Ok(handle.request)
    }

    /// Release all model state for a disconnected or cancelled request.
    pub fn cancel(&mut self, id: Uuid) -> Result<()> {
        let pending = self
            .pending
            .get(&id)
            .ok_or_else(|| TurError::RequestNotFound(id.to_string()))?;
        self.engine.cancel(pending.handle)?;
        self.pending.remove(&id);
        Ok(())
    }

    fn fail(&mut self, id: Uuid, error: TurError, output: &mut BatchStepOutput) -> Result<()> {
        self.cancel(id)?;
        output.new_tokens.retain(|(request, _)| *request != id);
        output
            .sampled_tokens
            .retain(|(request, _, _)| *request != id);
        output.completed.push((id, Err(error)));
        Ok(())
    }

    /// Release a running request's state; it recomputes its tokens when readmitted.
    fn preempt(&mut self, id: Uuid) -> Result<()> {
        let pending = self
            .pending
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown pipeline request"))?;
        pending.handle = self.engine.release_state(pending.handle)?;
        pending.consumed = 0;
        pending.running = false;
        pending.ngrams = NgramIndex::default();
        (pending.mtp_done, pending.mtp_draft) = (0, None);
        pending.preemptions += 1;
        tracing::debug!(request = %id, tokens = pending.tokens.len(), "preempted for recompute");
        Ok(())
    }

    /// Publish one request's commit. `logits` are the model logits of its last output,
    /// kept with a snapshot when the commit completes the prompt.
    fn publish(
        &mut self,
        id: Uuid,
        commit: SequenceCommit,
        logits: Option<Tensor>,
        output: &mut BatchStepOutput,
    ) -> Result<()> {
        let pending = self
            .pending
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown pipeline request"))?;
        pending.handle = commit.handle;
        if pending
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(TurError::DeadlineExceeded);
        }
        pending.consumed = pending
            .consumed
            .checked_add(commit.accepted_tokens)
            .ok_or_else(|| invalid("consumed position overflow"))?;
        if pending
            .boundaries
            .last()
            .is_none_or(|&last| pending.consumed > last)
        {
            pending.boundaries.push(pending.consumed);
        }
        // The prompt's own last logits (not a replay's) can serve an identical prompt.
        let prompt_end = commit.accepted_tokens > 0
            && pending.consumed == pending.prompt_tokens
            && pending.tokens.len() == pending.prompt_tokens;
        // Accepted drafts are consumed before they are known tokens: the outputs that
        // equal them are appended below.
        let mut finished = commit.finished;
        let mut reason = "stop";
        for token in commit.outputs {
            let sampled_at = Instant::now();
            output.sampled_tokens.push((id, token.token, sampled_at));
            pending.tokens.push(token.token);
            pending
                .first_token
                .get_or_insert_with(|| sampled_at.saturating_duration_since(pending.arrival));
            let eos = self.defaults.eos_tokens.contains(&token.token);
            if !eos && let Some(text) = pending.stream.next_token(token.token)? {
                output.new_tokens.push((id, text));
            }
            if eos || pending.generated() >= pending.max_tokens {
                finished = true;
                reason = if eos { "stop" } else { "length" };
                break;
            }
        }
        if (self.prefix.is_boundary(pending.consumed) || prompt_end)
            && let Some(prefix) = pending.tokens.get(..pending.consumed)
        {
            self.prefix.extend(&mut pending.keys, prefix);
            let state = self.engine.model_state(pending.handle)?;
            let logits = logits
                .filter(|_| prompt_end)
                // An allocated copy: `Tensor::copy` shares storage on Metal.
                .map(|logits| logits.force_contiguous())
                .transpose()?;
            self.prefix
                .insert(&pending.keys, prefix, &state, logits.as_ref())?;
        }
        if !finished {
            return Ok(());
        }
        if let Some(text) = pending.stream.decode_rest()? {
            output.new_tokens.push((id, text));
        }
        let generated_text = pending.stream.decode_all()?;
        let tool_calls = if pending.tools.is_empty() {
            Vec::new()
        } else {
            ToolCall::parse_from_output(&generated_text, &pending.tools)
        };
        let elapsed = pending.arrival.elapsed();
        if pending
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(TurError::DeadlineExceeded);
        }
        let stats = GenerationStats {
            generated_tokens: pending.generated(),
            prompt_tokens: pending.prompt_tokens,
            cached_tokens: pending.cached_tokens,
            elapsed,
            first_token: pending.first_token.unwrap_or(elapsed),
            preemptions: pending.preemptions,
            draft_tokens: pending.drafted,
            accepted_draft_tokens: pending.accepted_drafts,
            generated_text,
            tool_calls,
            finish_reason: reason,
        };
        // The reply's state lets a conversation's next turn resume after it.
        if pending.consumed > 0
            && self.prefix.config().enabled()
            && let Some(consumed) = pending.tokens.get(..pending.consumed)
        {
            self.prefix.extend(&mut pending.keys, consumed);
            let state = self.engine.model_state(pending.handle)?;
            self.prefix.insert(&pending.keys, consumed, &state, None)?;
        }
        self.engine.cancel(pending.handle)?;
        self.pending.remove(&id);
        output.completed.push((id, Ok(stats)));
        Ok(())
    }

    /// Fail every request whose deadline has passed.
    fn expire(&mut self, now: Instant, output: &mut BatchStepOutput) -> Result<()> {
        let expired: Vec<Uuid> = self
            .pending
            .iter()
            .filter(|(_, p)| p.deadline.is_some_and(|deadline| now >= deadline))
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            self.fail(id, TurError::DeadlineExceeded, output)?;
        }
        Ok(())
    }

    /// Restore the deepest verified snapshot for the oldest waiting requests that have no
    /// state, while running slots remain. They then run their remaining prompt; a snapshot
    /// of the whole prompt samples the first output from its logits without a pass.
    fn restore_prefixes(&mut self, output: &mut BatchStepOutput) -> Result<()> {
        if !self.prefix.config().enabled() {
            return Ok(());
        }
        let running = self.pending.values().filter(|p| p.running).count();
        let mut waiting: Vec<(Instant, Uuid)> = self
            .pending
            .iter()
            .filter(|(_, p)| !p.running && p.consumed == 0)
            .map(|(id, p)| (p.arrival, *id))
            .collect();
        waiting.sort_unstable();
        let free = self.scheduler.config().max_running.saturating_sub(running);
        let inserted = self.prefix.stats().inserted;
        for (_, id) in waiting.into_iter().take(free) {
            let pending = self
                .pending
                .get_mut(&id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            if pending.prefix_checked == Some(inserted) {
                continue;
            }
            // Recomputation keeps its original pass boundaries above the first restore.
            let (boundaries, floor) = (&pending.boundaries, pending.replay_floor);
            let accept = |length: usize| {
                boundaries.is_empty()
                    || length <= floor
                    || boundaries.binary_search(&length).is_ok()
            };
            let Some(hit) = self.prefix.lookup(&pending.keys, &pending.tokens, accept) else {
                pending.prefix_checked = Some(inserted);
                continue;
            };
            pending.handle = self.engine.adopt(pending.handle, &hit.state)?;
            pending.consumed = hit.length;
            pending.running = true;
            if pending.boundaries.is_empty() {
                pending.replay_floor = hit.length;
                pending.cached_tokens = hit.length;
            }
            tracing::debug!(request = %id, tokens = hit.length, "restored prefix snapshot");
            if hit.length < pending.tokens.len() {
                continue;
            }
            let handle = pending.handle;
            let first = hit
                .logits
                .ok_or_else(|| invalid("whole-prompt snapshot without logits"))
                .and_then(|logits| self.engine.sample_logits(handle, &logits))
                .and_then(|commit| self.publish(id, commit, None, output));
            if let Err(error) = first {
                self.fail(id, error, output)?;
            }
        }
        Ok(())
    }

    /// Evict snapshots while they keep the oldest admissible waiting request's next chunk
    /// (with one page of headroom per running sequence) from fitting.
    fn relieve_admission(&mut self, candidates: &[Candidate]) -> Result<()> {
        let running = candidates.iter().filter(|c| c.running).count();
        let Some(oldest) = candidates
            .iter()
            .filter(|c| !c.running && running < self.scheduler.config().max_running)
            .min_by_key(|c| (c.arrival, c.id))
        else {
            return Ok(());
        };
        let handle = self
            .pending
            .get(&oldest.id)
            .ok_or_else(|| invalid("missing pipeline request"))?
            .handle;
        let tokens = oldest.pending.min(self.scheduler.config().max_chunk);
        let headroom = Demand {
            pages: running,
            slots: 0,
        };
        let mut usage = self.engine.memory()?;
        while self.prefix.stats().entries > 0 {
            let demand = self.engine.demand(handle, tokens, false)? + headroom;
            if demand.fits(&usage) {
                break;
            }
            self.prefix.evict();
            let after = self.engine.memory()?;
            // Snapshots sharing pages with running requests free only their slot; stop
            // once evicting no longer helps a page shortage.
            if demand.slots <= after.slots.free() && after.pages.free() == usage.pages.free() {
                break;
            }
            usage = after;
        }
        Ok(())
    }

    /// Ask the scheduler for a pass, evicting prefix snapshots before preempting or
    /// failing requests, until one fits.
    fn select(&mut self, output: &mut BatchStepOutput) -> Result<Vec<Chunk>> {
        self.restore_prefixes(output)?;
        loop {
            let candidates: Vec<Candidate> = self
                .pending
                .iter()
                .map(|(id, p)| p.candidate(*id, self.prefix.next_boundary(p.consumed)))
                .collect();
            self.relieve_admission(&candidates)?;
            let usage = self.engine.memory()?;
            let running = candidates.iter().filter(|c| c.running).count();
            let (engine, pending) = (&self.engine, &self.pending);
            let mut demand = Demand::default();
            let decision = self.scheduler.plan(&candidates, Instant::now(), |chunk| {
                let request = pending
                    .get(&chunk.id)
                    .ok_or_else(|| invalid("missing pipeline request"))?;
                let needed = demand + engine.demand(request.handle, chunk.tokens, false)?;
                // Admission keeps one page of growth per running sequence free, so new
                // work does not immediately force running decodes out.
                let headroom = Demand {
                    pages: if chunk.admit { running } else { 0 },
                    slots: 0,
                };
                let fits = (needed + headroom).fits(&usage);
                if fits {
                    demand = needed;
                }
                Ok(fits)
            })?;
            match decision {
                Decision::Run(chunks) => return Ok(chunks),
                Decision::Preempt(_) | Decision::Stall(_) if self.prefix.evict() => {}
                Decision::Preempt(id) => self.preempt(id)?,
                Decision::Stall(id) => self.fail(
                    id,
                    TurError::MemoryAllocation(format!(
                        "request cannot fit the state pool ({} of {} K/V pages and {} of {} \
                         recurrent slots held)",
                        usage.pages.leased,
                        usage.pages.capacity,
                        usage.slots.leased,
                        usage.slots.capacity
                    )),
                    output,
                )?,
            }
        }
    }

    /// Drafts for each chunk of a pass, their sources, and how many chunks are decodes. Every
    /// decoding sequence may draft, also beside prefill chunks once draft costs are
    /// identified: the prefill's estimated cost then joins the pass base, so drafts are
    /// weighed against the whole pass they lengthen. Proposals come from each request's
    /// own history and are capped so that every output stays within its request's limit,
    /// the model context and the next prefix snapshot boundary; they never contain EOS.
    /// [`speculation::plan`] chooses how many to verify. Drafts take only what the
    /// scheduled pass left: pass tokens, and free pages and recurrent slots beside the
    /// pass's ordinary demand. No growth headroom is held back: rejected drafts return
    /// their leases when the pass completes, and accepted ones are progress a decode
    /// would have leased anyway.
    fn draft(&mut self, chunks: &[Chunk]) -> Result<(Vec<Vec<u32>>, Vec<DraftSource>, usize)> {
        let mut drafts = vec![Vec::new(); chunks.len()];
        let mut sources = vec![DraftSource::Lookup; chunks.len()];
        let decoding: Vec<bool> = chunks
            .iter()
            .map(|chunk| {
                self.pending.get(&chunk.id).is_some_and(|p| {
                    !chunk.admit
                        && chunk.tokens == 1
                        && p.running
                        && p.generated() > 0
                        && p.tokens.len() == p.consumed + 1
                })
            })
            .collect();
        let decodes = decoding.iter().filter(|&&d| d).count();
        let decode_only = decodes > 0 && decodes == chunks.len();
        if decodes == 0 || !self.speculation.enabled() {
            return Ok((drafts, sources, decodes));
        }
        let chunk_tokens = chunks.iter().map(|chunk| chunk.tokens).sum::<usize>();
        let room = self
            .scheduler
            .config()
            .max_batch_tokens
            .saturating_sub(chunk_tokens);
        // A mixed pass drafts only against identified costs, with its prefill's estimated
        // time added to the base; without either estimate it runs undrafted.
        let mixed = if decode_only {
            None
        } else {
            let context = chunks
                .iter()
                .filter_map(|chunk| {
                    let p = self.pending.get(&chunk.id)?;
                    Some(p.consumed + chunk.tokens)
                })
                .max()
                .unwrap_or(0);
            match self
                .scheduler
                .costs()
                .estimate(chunk_tokens - decodes, context)
            {
                Some(seconds) => Some(seconds),
                None => return Ok((drafts, sources, decodes)),
            }
        };
        let mut proposed = Vec::with_capacity(decodes);
        for (index, chunk) in chunks.iter().enumerate() {
            if !decoding[index] {
                continue;
            }
            let pending = self
                .pending
                .get_mut(&chunk.id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            if pending.constrained {
                continue;
            }
            // Outputs number drafts + 1 and occupy positions up to consumed + drafts.
            let remaining = pending.max_tokens.saturating_sub(pending.generated());
            let mut cap = self
                .speculation
                .max_draft
                .min(self.speculation.max_pass_drafts)
                .min(room)
                .min(remaining.saturating_sub(1))
                .min(
                    self.max_position
                        .saturating_sub(pending.consumed.saturating_add(1)),
                );
            if let Some(boundary) = self.prefix.next_boundary(pending.consumed) {
                cap = cap.min(boundary.saturating_sub(pending.consumed + 1));
            }
            let eos = &self.defaults.eos_tokens;
            let mut proposal: Vec<u32> = pending
                .ngrams
                .propose(&pending.tokens, &self.speculation, cap)
                .into_iter()
                .take_while(|token| !eos.contains(token))
                .collect();
            let mut source = DraftSource::Lookup;
            // The head's draft replaces a lookup proposal whose acceptance is lower.
            if (proposal.is_empty() || pending.mtp_acceptance.rate() > pending.acceptance.rate())
                && cap > 0
                && pending.mtp_done == pending.consumed
                && let Some(token) = pending.mtp_draft.filter(|token| !eos.contains(token))
            {
                (proposal, source) = (vec![token], DraftSource::Mtp);
            }
            if !proposal.is_empty() {
                let acceptance = match source {
                    DraftSource::Lookup => pending.acceptance,
                    DraftSource::Mtp => pending.mtp_acceptance,
                };
                proposed.push((index, proposal, acceptance, source));
            }
        }
        if proposed.is_empty() {
            return Ok((drafts, sources, decodes));
        }
        let proposals: Vec<Proposal> = proposed
            .iter()
            .map(|(_, tokens, acceptance, _)| Proposal {
                tokens: tokens.len(),
                acceptance: *acceptance,
            })
            .collect();
        let limit = self.speculation.max_pass_drafts.min(room);
        let counts = speculation::plan(&proposals, decodes, &self.draft_costs, limit, mixed);
        let usage = self.engine.memory()?;
        let mut demand = Demand::default();
        let mut plain = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let handle = self
                .pending
                .get(&chunk.id)
                .ok_or_else(|| invalid("missing pipeline request"))?
                .handle;
            let one = self.engine.demand(handle, chunk.tokens, false)?;
            demand = demand + one;
            plain.push(one);
        }
        for ((index, proposal, _, source), &count) in proposed.into_iter().zip(&counts) {
            let pending = self
                .pending
                .get_mut(&chunks[index].id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            let mut count = count;
            while count > 0 {
                let verify = self.engine.demand(pending.handle, count + 1, true)?;
                let needed = Demand {
                    pages: demand.pages - plain[index].pages + verify.pages,
                    slots: demand.slots - plain[index].slots + verify.slots,
                };
                if needed.fits(&usage) {
                    demand = needed;
                    break;
                }
                count -= 1;
            }
            if count == 0 {
                match source {
                    DraftSource::Lookup => pending.acceptance.recover(self.acceptance),
                    DraftSource::Mtp => pending.mtp_acceptance.recover(Acceptance::default()),
                }
            }
            drafts[index] = proposal.get(..count).unwrap_or_default().to_vec();
            sources[index] = source;
        }
        Ok((drafts, sources, decodes))
    }

    /// Pack the selected chunks: each sequence's next unconsumed tokens, with logits at its
    /// last known token. A drafted decode appends its drafts and needs logits at every
    /// position, for verification. Returns the plan and its longest resulting context.
    fn pack(&self, chunks: &[Chunk], drafts: &[Vec<u32>]) -> Result<(PackedExecution, usize)> {
        let mut plan = PackedExecution {
            tokens: Vec::new(),
            positions: Vec::new(),
            sequences: Vec::with_capacity(chunks.len()),
        };
        let mut context = 0;
        for (chunk, draft) in chunks.iter().zip(drafts) {
            let pending = self
                .pending
                .get(&chunk.id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            let end = pending
                .consumed
                .checked_add(chunk.tokens)
                .ok_or_else(|| invalid("chunk position overflow"))?;
            let input = pending
                .tokens
                .get(pending.consumed..end)
                .ok_or_else(|| invalid("chunk beyond the request's tokens"))?;
            let start = plan.tokens.len();
            plan.tokens.extend_from_slice(input);
            plan.tokens.extend_from_slice(draft);
            let end = end
                .checked_add(draft.len())
                .ok_or_else(|| invalid("draft position overflow"))?;
            plan.positions.extend(pending.consumed..end);
            let speculative = !draft.is_empty();
            plan.sequences.push(SequenceDescriptor {
                state: pending.handle,
                tokens: start..plan.tokens.len(),
                outputs: if speculative {
                    (start..plan.tokens.len()).collect()
                } else if end == pending.tokens.len() {
                    vec![plan.tokens.len() - 1]
                } else {
                    Vec::new()
                },
                speculative,
            });
            context = context.max(end);
        }
        Ok((plan, context))
    }

    /// Expire deadlines, select decodes and prefill chunks (preempting under memory
    /// pressure) and submit them as one packed pass without waiting for it. Returns whether
    /// a pass is in flight. Until [`Self::finish_step`], callers may submit and cancel
    /// requests while the device runs; the next pass cannot be selected before this one's
    /// tokens are known, so at most one is in flight.
    pub fn begin_step(&mut self) -> Result<bool> {
        if self.in_flight.is_some() {
            return Err(invalid("a pass is already in flight"));
        }
        let begin = Instant::now();
        let mut output = std::mem::take(&mut self.staged);
        let selected = self
            .expire(begin, &mut output)
            .and_then(|()| self.select(&mut output));
        let chunks = match selected {
            Ok(chunks) => chunks,
            Err(error) => {
                self.staged = output;
                return Err(error);
            }
        };
        self.staged = output;
        if chunks.is_empty() {
            return Ok(false);
        }
        let (drafts, sources, decodes) = self.draft(&chunks)?;
        let chunk_tokens = chunks.iter().map(|chunk| chunk.tokens).sum::<usize>();
        let (plan, context) = self.pack(&chunks, &drafts)?;
        let (tokens, sequences) = (plan.tokens.len(), plan.sequences.len());
        let started = Instant::now();
        self.engine.submit(plan)?;
        let submitted = Instant::now();
        self.in_flight = Some(InFlight {
            chunks,
            drafts,
            sources,
            decodes,
            context,
            started,
            submitted,
            timing: PassTiming {
                tokens,
                sequences,
                drafts: tokens - chunk_tokens,
                schedule: started.saturating_duration_since(begin),
                encode: submitted.saturating_duration_since(started),
                idle_before: self
                    .engine
                    .completed_at()
                    .map(|at| started.saturating_duration_since(at)),
                ..PassTiming::default()
            },
        });
        Ok(true)
    }

    /// Wait for the pass from [`Self::begin_step`], if any, then publish each request's
    /// outputs. A request's model or decode error is returned only to that request;
    /// requests cancelled while the pass ran publish nothing.
    pub fn finish_step(&mut self) -> Result<BatchStepOutput> {
        let mut output = std::mem::take(&mut self.staged);
        let Some(flight) = self.in_flight.take() else {
            return Ok(output);
        };
        let waiting = Instant::now();
        let results = self.engine.finish()?;
        let mut timing = flight.timing;
        timing.overlap = waiting.saturating_duration_since(flight.submitted);
        let completed = self.engine.completed_at().unwrap_or_else(Instant::now);
        timing.wait = completed.saturating_duration_since(waiting);
        timing.device = completed.saturating_duration_since(flight.started);
        // Host work after completion, from the engine's sampling on.
        let published = completed.max(waiting);
        // Pass costs exclude host work that overlapped the device. The scheduler sizes
        // prefill from undrafted token costs: a drafted pass contributes its undrafted
        // tokens and time, once the per-draft cost is known.
        let device = timing.device.as_secs_f64();
        let undrafted = match timing.drafts {
            0 => Some(device),
            drafts => self
                .draft_costs
                .line(flight.decodes)
                .map(|(_, per_draft)| per_draft.mul_add(-(drafts as f64), device))
                .filter(|&seconds| seconds > 0.0),
        };
        if let Some(seconds) = undrafted {
            self.scheduler.record(
                timing.tokens - timing.drafts,
                flight.context,
                Duration::from_secs_f64(seconds),
            );
        }
        if flight.decodes == flight.chunks.len() {
            self.draft_costs.record(
                flight.chunks.len(),
                timing.drafts,
                timing.device.as_secs_f64(),
            );
        }
        if timing.drafts > 0 {
            self.speculation_stats.passes += 1;
        }
        let drafted: HashMap<Uuid, (usize, DraftSource)> = flight
            .chunks
            .iter()
            .zip(&flight.drafts)
            .zip(&flight.sources)
            .filter(|((_, draft), _)| !draft.is_empty())
            .map(|((chunk, draft), &source)| (chunk.id, (draft.len(), source)))
            .collect();
        for chunk in &flight.chunks {
            if let Some(pending) = self.pending.get_mut(&chunk.id) {
                pending.running = true;
                pending.last_run = chunk.ticket;
            }
        }
        // Committed positions each request's draft head still needs: (id, first, hidden).
        let mut heads = Vec::new();
        for result in results {
            let id = result.request;
            let Some(before) = self.pending.get(&id).map(|p| p.consumed) else {
                continue;
            };
            if let (Some(&(drafted, source)), Ok(commit)) = (drafted.get(&id), &result.outcome) {
                // Every accepted input after the decode token is an accepted draft.
                let accepted = commit.accepted_tokens.saturating_sub(1).min(drafted);
                if let Some(pending) = self.pending.get_mut(&id) {
                    match source {
                        DraftSource::Lookup => pending.acceptance.observe(drafted, accepted),
                        DraftSource::Mtp => pending.mtp_acceptance.observe(drafted, accepted),
                    }
                    pending.drafted += drafted;
                    pending.accepted_drafts += accepted;
                }
                let stats = &mut self.speculation_stats;
                match source {
                    DraftSource::Lookup => self.acceptance.observe(drafted, accepted),
                    DraftSource::Mtp => {
                        self.mtp_acceptance.observe(drafted, accepted);
                        stats.mtp_drafted += drafted as u64;
                        stats.mtp_accepted += accepted as u64;
                    }
                }
                stats.sequences += 1;
                stats.drafted += drafted as u64;
                stats.accepted += accepted as u64;
            }
            // Only greedy, unconstrained requests ever take the head's drafts; it runs for
            // no other.
            let drafts_from_head = self
                .pending
                .get(&id)
                .is_some_and(|p| p.greedy && !p.constrained);
            if let (Some(hidden), Ok(commit)) = (&result.hidden, &result.outcome)
                && commit.accepted_tokens > 0
                && drafts_from_head
            {
                heads.push((id, before, hidden.narrow(0, 0, commit.accepted_tokens)?));
            }
            let logits = result.logits;
            let outcome = result
                .outcome
                .and_then(|commit| self.publish(id, commit, logits, &mut output));
            if let Err(error) = outcome {
                self.fail(id, error, &mut output)?;
            }
        }
        if !heads.is_empty() {
            timing.draft_head = self.draft_head(heads)?;
            self.speculation_stats.mtp_passes += 1;
            self.speculation_stats.mtp_seconds += timing.draft_head.as_secs_f64();
        }
        self.expire(Instant::now(), &mut output)?;
        timing.publish = published.elapsed();
        output.timing = Some(timing);
        Ok(output)
    }

    /// Run the draft head over each published request's newly committed positions, pairing
    /// each with the token that followed it, and keep the head's draft for a request whose
    /// next pass decodes, unless prompt lookup proposes with higher acceptance. A request whose head falls behind (it failed, or a position was
    /// published without hidden states) stops drafting from the head until it recomputes.
    fn draft_head(&mut self, heads: Vec<(Uuid, usize, Tensor)>) -> Result<Duration> {
        let mut requests = Vec::with_capacity(heads.len());
        let mut ids = Vec::with_capacity(heads.len());
        let mut rates = Vec::new();
        for (id, before, hidden) in heads {
            let Some(pending) = self.pending.get_mut(&id) else {
                continue;
            };
            pending.mtp_draft = None;
            let Some(tokens) = pending.tokens.get(before + 1..=pending.consumed) else {
                continue;
            };
            if pending.mtp_done != before || tokens.len() != hidden.dim(0)? {
                continue;
            }
            // Its LM-head projection costs about a third of a small model's decode: compute
            // a draft only when the next pass decodes, prompt lookup has no proposal or a less
            // accepted one, and the head's acceptance can repay it.
            let mut draft = pending.tokens.len() == pending.consumed + 1;
            if draft && pending.mtp_acceptance.rate() < speculation::MTP_MIN_RATE {
                // Towards the neutral prior, not the pipeline's rate, which may itself sit
                // below the threshold: a skipped head is retried after a few passes.
                pending.mtp_acceptance.recover(Acceptance::default());
                draft = false;
            }
            draft = draft
                && (pending.mtp_acceptance.rate() > pending.acceptance.rate()
                    || pending
                        .ngrams
                        .propose(&pending.tokens, &self.speculation, 1)
                        .is_empty());
            if draft {
                rates.push(pending.mtp_acceptance.rate());
            }
            requests.push(DraftRequest {
                handle: pending.handle,
                start: before,
                tokens: tokens.to_vec(),
                hidden,
                draft,
            });
            ids.push(id);
        }
        if requests.is_empty() {
            return Ok(Duration::ZERO);
        }
        // The head's drafts must repay its projection for the whole next pass. Beyond the
        // measured beneficial range its drafts are not computed; its keys and values still
        // are, so drafting resumes when fewer sequences decode.
        let decodes = self
            .pending
            .values()
            .filter(|p| p.running && p.tokens.len() == p.consumed + 1)
            .count();
        if decodes > speculation::MTP_MAX_DECODES {
            rates.clear();
            for request in &mut requests {
                request.draft = false;
            }
        }
        if !rates.is_empty() {
            if self
                .head_costs
                .worthwhile(&rates, decodes, &self.draft_costs)
            {
                (self.head_skips, self.head_explore) = (0, speculation::HEAD_EXPLORE_PASSES);
            } else if self.head_skips >= self.head_explore {
                self.head_skips = 0;
                self.head_explore =
                    (self.head_explore * 2).min(speculation::HEAD_EXPLORE_MAX_PASSES);
            } else {
                self.head_skips += 1;
                for request in &mut requests {
                    request.draft = false;
                }
            }
        }
        let drafts = requests.iter().filter(|r| r.draft).count();
        // Decode-sized: at most the drafted tokens plus one per sequence.
        let rows: usize = requests.iter().map(|r| r.tokens.len()).sum();
        let decode_sized = rows <= requests.len() * (self.speculation.max_draft + 1);
        let started = Instant::now();
        let outcomes = self.engine.draft(&requests)?;
        let elapsed = started.elapsed();
        if decode_sized {
            self.head_costs.record(drafts, elapsed.as_secs_f64());
        }
        for (id, outcome) in ids.into_iter().zip(outcomes) {
            let Some(pending) = self.pending.get_mut(&id) else {
                continue;
            };
            match outcome {
                Ok(draft) => {
                    pending.mtp_done = pending.consumed;
                    pending.mtp_draft = draft;
                }
                Err(error) => tracing::debug!(request = %id, "draft head failed: {error}"),
            }
        }
        Ok(elapsed)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_uses_checkpoint_defaults_and_explicit_greedy() {
        let defaults = GenerationDefaults {
            temperature: Some(0.7),
            top_k: Some(20),
            top_p: Some(0.8),
            repetition_penalty: Some(1.1),
            eos_tokens: vec![1],
        };
        let inherited = sampling(SamplingOptions::default(), &defaults);
        assert_eq!(inherited.temperature, Some(0.7));
        assert_eq!(inherited.top_k, Some(20));
        assert_eq!(inherited.top_p, Some(0.8));
        let explicit = sampling(
            SamplingOptions {
                temperature: Some(0.0),
                top_k: Some(1),
                top_p: Some(1.0),
                presence_penalty: Some(0.5),
                ..Default::default()
            },
            &defaults,
        );
        assert_eq!(explicit.temperature, Some(0.0));
        assert_eq!(explicit.top_k, Some(1));
        assert_eq!(explicit.top_p, Some(1.0));
        assert_eq!(explicit.repetition_penalty, Some(1.1));
        assert_eq!(explicit.presence_penalty, Some(0.5));
    }
}
