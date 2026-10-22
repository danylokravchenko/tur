#![forbid(unsafe_code)]
//! Chat admission and packed prefill/decode over the single-owner hybrid engine.

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
    memory::{MemoryPlan, PlanInputs},
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
        }
    }
}

/// Completed generation, including EOS in the generated token count when sampled.
#[derive(Debug)]
pub struct GenerationStats {
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    pub elapsed: Duration,
    pub tool_calls: Vec<ToolCall>,
    pub generated_text: String,
    /// `stop` for EOS, `length` for the request limit.
    pub finish_reason: &'static str,
}

/// Incremental text and independent request outcomes from one packed model invocation.
#[derive(Default)]
pub struct BatchStepOutput {
    pub new_tokens: Vec<(Uuid, String)>,
    pub completed: Vec<(Uuid, Result<GenerationStats>)>,
}

struct Pending {
    handle: StateHandle,
    prompt: Vec<u32>,
    consumed: usize,
    next: Option<u32>,
    stream: TokenOutputStream,
    generated: usize,
    max_tokens: usize,
    started: Instant,
    tools: Vec<ToolDefinition>,
}

/// Admission limits and the state memory budget of one pipeline.
#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    pub max_batch_size: usize,
    pub prefill_chunk_size: usize,
    /// Bytes for K/V pages and recurrent state. `None` sizes the pool for
    /// `max_batch_size` contexts of [`super::memory::DEFAULT_CONTEXT_TOKENS`] tokens, within
    /// the device's working set.
    pub state_bytes: Option<usize>,
}

impl PipelineConfig {
    /// Limits with a derived state budget.
    pub fn new(max_batch_size: usize, prefill_chunk_size: usize) -> Self {
        Self {
            max_batch_size,
            prefill_chunk_size,
            state_bytes: None,
        }
    }
}

/// Thin serving adapter. Every iteration packs the admitted sequences whose next chunk
/// fits the state pool into one model pass; the rest wait. If nothing fits, the newest
/// request is failed with [`TurError::MemoryAllocation`] so older ones can progress.
/// Prefix reuse and production scheduling with preemption are later stages.
pub struct InferencePipeline {
    engine: HybridEngine<Qwen35>,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    defaults: GenerationDefaults,
    max_position: usize,
    max_batch_size: usize,
    prefill_chunk_size: usize,
    plan: MemoryPlan,
    pending: HashMap<Uuid, Pending>,
    order: Vec<Uuid>,
}

fn sampling(overrides: SamplingOptions, defaults: &GenerationDefaults) -> SamplingOptions {
    SamplingOptions {
        temperature: overrides.temperature.or(defaults.temperature),
        top_k: overrides.top_k.or(defaults.top_k),
        top_p: overrides.top_p.or(defaults.top_p),
    }
}

impl InferencePipeline {
    /// Load the validated model and size its state pool. Zero batch or chunk limits fail
    /// before loading weights; a plan beyond the device working set fails after.
    pub fn new(factory: &ModelFactory, config: PipelineConfig) -> Result<Self> {
        let PipelineConfig {
            max_batch_size,
            prefill_chunk_size,
            state_bytes,
        } = config;
        if max_batch_size == 0 || prefill_chunk_size == 0 {
            return Err(invalid("batch and prefill chunk sizes must be positive"));
        }
        let (model, tokenizer, template, defaults) = factory.create_model()?;
        let max_position = model.config().max_position();
        let description = tur_models::hybrid::HybridModel::description(&model);
        // The largest pass: every sequence prefills a whole chunk at the sizing context.
        let context = super::memory::DEFAULT_CONTEXT_TOKENS.min(max_position);
        let tokens = max_batch_size
            .checked_mul(prefill_chunk_size)
            .ok_or_else(|| invalid("pass size overflow"))?;
        let in_flight = max_batch_size
            .checked_mul(description.vocab_size * 4)
            .and_then(|logits| logits.checked_add(tokens * 12))
            .ok_or_else(|| invalid("in-flight size overflow"))?;
        let (plan, pool) = MemoryPlan::new(
            &description,
            PlanInputs {
                weights: model.weight_bytes(),
                scratch_peak: model
                    .planned_workspace_bytes(tokens, &vec![context; max_batch_size])?,
                in_flight,
                sequences: max_batch_size,
                state_bytes,
                limit: tur_metal::pool::working_set_bytes(&factory.device),
            },
        )?;
        let engine = HybridEngine::new(Arc::new(model), factory.device.clone(), pool)?;
        Ok(Self {
            engine,
            tokenizer,
            template,
            defaults,
            max_position,
            max_batch_size,
            prefill_chunk_size,
            plan,
            pending: HashMap::new(),
            order: Vec::new(),
        })
    }

    /// Number of requests with allocated model state.
    pub fn active_request_count(&self) -> usize {
        self.pending.len()
    }

    /// Planned resident memory and the state pool it sized.
    pub fn memory_plan(&self) -> &MemoryPlan {
        &self.plan
    }

    /// Current state pool occupancy, after reclaiming released pages and slots.
    pub fn memory(&mut self) -> Result<PoolUsage> {
        self.engine.memory()
    }

    /// Render, tokenize and allocate independent hybrid state for one text conversation.
    pub fn submit_request(&mut self, request: &GenerationRequest) -> Result<Uuid> {
        if self.pending.len() >= self.max_batch_size {
            return Err(invalid("pipeline admission limit reached"));
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
        let handle = self.engine.allocate(
            request.seed,
            sampling(request.sampling, &self.defaults),
            None,
        )?;
        self.pending.insert(
            handle.request,
            Pending {
                handle,
                prompt: tokens,
                consumed: 0,
                next: None,
                stream: TokenOutputStream::new(self.tokenizer.clone()),
                generated: 0,
                max_tokens: request.max_tokens,
                started: Instant::now(),
                tools: request.tools.clone(),
            },
        );
        self.order.push(handle.request);
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
        self.order.retain(|request| *request != id);
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
        if pending.consumed < pending.prompt.len() {
            pending.consumed = pending
                .consumed
                .checked_add(commit.accepted_tokens)
                .ok_or_else(|| invalid("prefill position overflow"))?;
        }
        let mut finished = commit.finished;
        let mut reason = "stop";
        for token in commit.outputs {
            pending.generated += 1;
            pending.next = Some(token.token);
            let eos = self.defaults.eos_tokens.contains(&token.token);
            if !eos && let Some(text) = pending.stream.next_token(token.token)? {
                output.new_tokens.push((id, text));
            }
            if eos || pending.generated >= pending.max_tokens {
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
        let stats = GenerationStats {
            generated_tokens: pending.generated,
            prompt_tokens: pending.prompt.len(),
            elapsed: pending.started.elapsed(),
            generated_text,
            tool_calls,
            finish_reason: reason,
        };
        self.engine.cancel(pending.handle)?;
        self.pending.remove(&id);
        output.completed.push((id, Ok(stats)));
        Ok(())
    }

    /// Pack the requests whose next chunk fits the state pool, in admission order.
    fn plan(&self, usage: &PoolUsage) -> Result<PackedExecution> {
        let mut plan = PackedExecution {
            tokens: Vec::new(),
            positions: Vec::new(),
            sequences: Vec::new(),
        };
        let mut demand = Demand::default();
        for id in &self.order {
            let pending = self
                .pending
                .get(id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            let meta = self.engine.metadata(*id)?;
            let (input, last) = if pending.consumed < pending.prompt.len() {
                let end = pending
                    .consumed
                    .saturating_add(self.prefill_chunk_size)
                    .min(pending.prompt.len());
                (
                    &pending.prompt[pending.consumed..end],
                    end == pending.prompt.len(),
                )
            } else {
                let next = pending
                    .next
                    .as_ref()
                    .ok_or_else(|| invalid("decode without a sampled token"))?;
                (std::slice::from_ref(next), true)
            };
            let needed = demand + self.engine.demand(pending.handle, input.len(), false)?;
            if !needed.fits(usage) {
                continue;
            }
            demand = needed;
            let start = plan.tokens.len();
            plan.tokens.extend_from_slice(input);
            let end_position = meta
                .position
                .checked_add(input.len())
                .ok_or_else(|| invalid("position overflow"))?;
            plan.positions.extend(meta.position..end_position);
            plan.sequences.push(SequenceDescriptor {
                state: pending.handle,
                tokens: start..plan.tokens.len(),
                outputs: if last {
                    vec![plan.tokens.len() - 1]
                } else {
                    Vec::new()
                },
                speculative: false,
            });
        }
        Ok(plan)
    }

    /// Pack one chunk per prefill request and one token per decode request, fence, then
    /// publish. Requests whose chunk does not fit the state pool wait; when nothing fits,
    /// the newest request fails with a memory error. A request's model or decode error is
    /// returned only to that request.
    pub fn step_streaming(&mut self) -> Result<BatchStepOutput> {
        let mut output = BatchStepOutput::default();
        let plan = loop {
            if self.pending.is_empty() {
                return Ok(output);
            }
            let usage = self.engine.memory()?;
            let plan = self.plan(&usage)?;
            if !plan.sequences.is_empty() {
                break plan;
            }
            let newest = *self
                .order
                .last()
                .ok_or_else(|| invalid("pending request missing from order"))?;
            self.cancel(newest)?;
            output.completed.push((
                newest,
                Err(TurError::MemoryAllocation(format!(
                    "state pool exhausted ({} of {} K/V pages and {} of {} recurrent slots \
                     held); the newest request was stopped",
                    usage.pages.leased,
                    usage.pages.capacity,
                    usage.slots.leased,
                    usage.slots.capacity
                ))),
            ));
        };
        self.engine.submit(plan)?;
        for result in self.engine.finish()? {
            let id = result.request;
            let outcome = result
                .outcome
                .and_then(|commit| self.publish(id, commit, &mut output));
            if let Err(error) = outcome {
                self.cancel(id)?;
                output.completed.push((id, Err(error)));
            }
        }
        self.order.retain(|id| self.pending.contains_key(id));
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
