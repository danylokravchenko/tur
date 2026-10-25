//! Request admission, cancellation and publication of committed outputs.

use super::super::{
    guidance::{GuidanceControl, TopLevelGrammar, build_llg_factory},
    speculation::NgramIndex,
    tokenizer::TokenOutputStream,
};
use super::{InferencePipeline, Pending};
use crate::{
    Result, TurError,
    hybrid::{SamplingOptions, SequenceCommit},
};
use candle_core::Tensor;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tur_core::execution::invalid;
use tur_models::{
    backend::{
        chat_template::{ChatMessage, RenderOptions, Role},
        tools::{ToolCall, ToolDefinition},
    },
    models::qwen3_5::GenerationDefaults,
};
use uuid::Uuid;

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
    /// Predict where a new user turn would first rewrite this prompt. Only token
    /// equality at lookup authorizes a restore; the probe just chooses a pass end.
    fn turn_boundary(&self, request: &GenerationRequest, tokens: &[u32]) -> Option<usize> {
        // A first turn has no demonstrated conversational reuse; splitting it adds a
        // pass to ordinary single-turn requests. Start predicting once history exists.
        if !self.prefix.config().enabled()
            || !request.render.add_generation_prompt
            || !request
                .messages
                .iter()
                .any(|message| message.role == Role::Assistant)
        {
            return None;
        }
        let mut messages = request.messages.clone();
        messages.extend([
            ChatMessage::new(Role::Assistant, ""),
            ChatMessage::new(Role::User, ""),
        ]);
        let options = RenderOptions {
            add_generation_prompt: false,
            ..request.render.clone()
        };
        // A template may reject synthetic empty turns. That makes the prediction
        // unavailable, without invalidating the actual request that already rendered.
        let rendered = self
            .template
            .render(&messages, &request.tools, &options)
            .ok()?;
        let future = self.tokenizer.encode(rendered, false).ok()?;
        let boundary = tokens
            .iter()
            .zip(future.get_ids())
            .take_while(|(current, future)| current == future)
            .count();
        (boundary > 0 && boundary < tokens.len() && !self.prefix.is_boundary(boundary))
            .then_some(boundary)
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
        let turn_boundary = self.turn_boundary(request, &tokens);
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
                turn_boundary,
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
                mtp_verifying: None,
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

    pub(super) fn fail(
        &mut self,
        id: Uuid,
        error: TurError,
        output: &mut BatchStepOutput,
    ) -> Result<()> {
        self.cancel(id)?;
        output.new_tokens.retain(|(request, _)| *request != id);
        output
            .sampled_tokens
            .retain(|(request, _, _)| *request != id);
        output.completed.push((id, Err(error)));
        Ok(())
    }

    /// Release a running request's state; it recomputes its tokens when readmitted.
    pub(super) fn preempt(&mut self, id: Uuid) -> Result<()> {
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
    pub(super) fn publish(
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
        if (self.prefix.is_boundary(pending.consumed)
            || prompt_end
            || pending.turn_boundary == Some(pending.consumed))
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
    pub(super) fn expire(&mut self, now: Instant, output: &mut BatchStepOutput) -> Result<()> {
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
