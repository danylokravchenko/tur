#![forbid(unsafe_code)]
//! Chat admission and scheduled packed execution over the single-owner hybrid engine.

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
    memory::{BudgetLease, DeviceBudget, MemoryPlan, PlanInputs},
    scheduler::{Candidate, Chunk, Decision, Scheduler, SchedulerConfig},
    tokenizer::TokenOutputStream,
};
use crate::{
    Result, TurError,
    hybrid::{Demand, HybridEngine, PoolUsage, SamplingOptions, SequenceCommit},
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
        }
    }
}

/// Completed generation, including EOS in the generated token count when sampled.
#[derive(Debug)]
pub struct GenerationStats {
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    /// From submission to completion.
    pub elapsed: Duration,
    /// From submission to the first sampled token.
    pub first_token: Duration,
    /// Times the request's state was released under memory pressure and recomputed.
    pub preemptions: usize,
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
}

impl Pending {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_tokens
    }

    fn candidate(&self, id: Uuid) -> Candidate {
        let unconsumed = self.tokens.len() - self.consumed;
        let next = self.boundaries.partition_point(|&end| end <= self.consumed);
        let replay = self.boundaries.get(next).map(|end| end - self.consumed);
        Candidate {
            id,
            pending: replay.unwrap_or(unconsumed),
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
    /// Bytes for K/V pages and recurrent state. `None` sizes the pool for
    /// `max_running` contexts of [`super::memory::DEFAULT_CONTEXT_TOKENS`] tokens, within
    /// what the device budget leaves.
    pub state_bytes: Option<usize>,
}

impl PipelineConfig {
    /// Default scheduling for `max_batch_size` running sequences and `prefill_chunk_size`
    /// prefill chunks, with a derived state budget.
    pub fn new(max_batch_size: usize, prefill_chunk_size: usize) -> Self {
        Self {
            scheduler: SchedulerConfig::new(max_batch_size, prefill_chunk_size),
            state_bytes: None,
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
}

fn sampling(overrides: SamplingOptions, defaults: &GenerationDefaults) -> SamplingOptions {
    SamplingOptions {
        temperature: overrides.temperature.or(defaults.temperature),
        top_k: overrides.top_k.or(defaults.top_k),
        top_p: overrides.top_p.or(defaults.top_p),
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
        let limits = config.scheduler;
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
                // Full model logits and their gathered copy coexist for sampled requests.
                let in_flight = description
                    .vocab_size
                    .checked_mul(8)
                    .and_then(|row| row.checked_add(16))
                    .and_then(|row| sequences.checked_mul(row))
                    .and_then(|logits| logits.checked_add(tokens.checked_mul(12)?))
                    .ok_or_else(|| invalid("in-flight size overflow"))?;
                let (plan, pool) = MemoryPlan::new(
                    &description,
                    PlanInputs {
                        weights,
                        scratch_peak,
                        in_flight,
                        sequences: limits.max_running,
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
        let engine = HybridEngine::new(Arc::new(model), factory.device.clone(), pool)?;
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

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
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
        let handle = self.engine.allocate(
            request.seed,
            sampling(request.sampling, &self.defaults),
            None,
        )?;
        self.pending.insert(
            handle.request,
            Pending {
                handle,
                prompt_tokens: tokens.len(),
                tokens,
                consumed: 0,
                boundaries: Vec::new(),
                running: false,
                stream: TokenOutputStream::new(self.tokenizer.clone()),
                max_tokens: request.max_tokens,
                arrival,
                deadline,
                first_token: None,
                last_run: 0,
                preemptions: 0,
                tools: request.tools.clone(),
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
        pending.preemptions += 1;
        tracing::debug!(request = %id, tokens = pending.tokens.len(), "preempted for recompute");
        Ok(())
    }

    fn publish(
        &mut self,
        id: Uuid,
        commit: SequenceCommit,
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
            }
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
            elapsed,
            first_token: pending.first_token.unwrap_or(elapsed),
            preemptions: pending.preemptions,
            generated_text,
            tool_calls,
            finish_reason: reason,
        };
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

    /// Ask the scheduler for a pass, preempting or failing requests until one fits.
    fn select(&mut self, output: &mut BatchStepOutput) -> Result<Vec<Chunk>> {
        loop {
            let usage = self.engine.memory()?;
            let candidates: Vec<Candidate> = self
                .pending
                .iter()
                .map(|(id, p)| p.candidate(*id))
                .collect();
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

    /// Pack the selected chunks: each sequence's next unconsumed tokens, with logits at its
    /// last known token. Returns the plan and its longest resulting context.
    fn pack(&self, chunks: &[Chunk]) -> Result<(PackedExecution, usize)> {
        let mut plan = PackedExecution {
            tokens: Vec::new(),
            positions: Vec::new(),
            sequences: Vec::with_capacity(chunks.len()),
        };
        let mut context = 0;
        for chunk in chunks {
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
            plan.positions.extend(pending.consumed..end);
            plan.sequences.push(SequenceDescriptor {
                state: pending.handle,
                tokens: start..plan.tokens.len(),
                outputs: if end == pending.tokens.len() {
                    vec![plan.tokens.len() - 1]
                } else {
                    Vec::new()
                },
                speculative: false,
            });
            context = context.max(end);
        }
        Ok((plan, context))
    }

    /// Run one scheduled packed pass: expire deadlines, select decodes and prefill chunks
    /// (preempting under memory pressure), fence, then publish. A request's model or
    /// decode error is returned only to that request.
    pub fn step_streaming(&mut self) -> Result<BatchStepOutput> {
        let mut output = BatchStepOutput::default();
        self.expire(Instant::now(), &mut output)?;
        let chunks = self.select(&mut output)?;
        if chunks.is_empty() {
            return Ok(output);
        }
        let (plan, context) = self.pack(&chunks)?;
        let tokens = plan.tokens.len();
        let started = Instant::now();
        self.engine.submit(plan)?;
        let results = self.engine.finish()?;
        self.scheduler.record(tokens, context, started.elapsed());
        for chunk in &chunks {
            if let Some(pending) = self.pending.get_mut(&chunk.id) {
                pending.running = true;
                pending.last_run = chunk.ticket;
            }
        }
        for result in results {
            let id = result.request;
            let outcome = result
                .outcome
                .and_then(|commit| self.publish(id, commit, &mut output));
            if let Err(error) = outcome {
                self.fail(id, error, &mut output)?;
            }
        }
        self.expire(Instant::now(), &mut output)?;
        Ok(output)
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
            },
            &defaults,
        );
        assert_eq!(explicit.temperature, Some(0.0));
        assert_eq!(explicit.top_k, Some(1));
        assert_eq!(explicit.top_p, Some(1.0));
    }
}
