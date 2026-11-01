//! Request admission, cancellation and publication of committed outputs.

use super::super::{
    guidance::{GuidanceControl, ParserFactory, TopLevelGrammar, build_llg_factory, token_bytes},
    harmony::{Fragment, Harmony},
    reasoning::{ReasoningSplit, Route},
    speculation::NgramIndex,
    stop::StopMatcher,
    structured::{self, Preamble, StructuredOutput},
    tokenizer::TokenOutputStream,
};
use super::{InferencePipeline, Pending};
use crate::{
    Result, TurError,
    hybrid::{Forced, OutputLogprobs, SamplingOptions, SequenceCommit},
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
        tools::{ToolCall, ToolDefinition, ToolSyntax},
    },
    models::GenerationDefaults,
};
use uuid::Uuid;

/// One text conversation. Unset sampling and thinking controls use checkpoint defaults.
pub struct GenerationRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: usize,
    /// Shorten `max_tokens` to what the context leaves instead of failing: for a limit the
    /// caller did not choose. A prompt that alone exceeds the context still fails.
    pub fit_context: bool,
    /// End the answer before the first of these texts. It is left out of the output, and
    /// streams never show part of it. Reasoning is not searched.
    pub stop: Vec<String>,
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
    /// Force a JSON answer or a tool call in this model's format; excludes `grammar`. Qwen and
    /// Gemma run with thinking off unless `render` asks for it. Reasoning then stays free and
    /// is returned apart from the answer.
    pub structured: Option<StructuredOutput>,
    /// Soft tokens of every clip of `messages`, in order, when they were encoded before
    /// submission; empty encodes them at submission, on the calling thread.
    pub encoded_audio: Vec<Tensor>,
}

impl GenerationRequest {
    /// Generate a reply to one user turn, using the checkpoint's generation opener.
    pub fn new(prompt: String, max_tokens: usize) -> Self {
        Self {
            messages: vec![ChatMessage::new(Role::User, prompt)],
            tools: Vec::new(),
            max_tokens,
            fit_context: false,
            stop: Vec::new(),
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
            structured: None,
            encoded_audio: Vec::new(),
        }
    }

    /// Whether submission would have to encode audio.
    pub fn needs_audio_encoding(&self) -> bool {
        self.encoded_audio.is_empty()
            && self
                .messages
                .iter()
                .any(|m| m.content.audio().next().is_some())
    }
}

/// One generated token's log probability, or one of its likeliest alternatives.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    /// The token's text, lossy where it holds part of a character.
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
    /// Empty on the alternatives themselves.
    pub top: Vec<Self>,
}

impl TokenLogprob {
    fn new(factory: &ParserFactory, id: u32, logprob: f32, top: Vec<Self>) -> Self {
        let bytes = token_bytes(factory, id);
        Self {
            token: String::from_utf8_lossy(&bytes).into_owned(),
            bytes,
            logprob,
            top,
        }
    }

    fn of_output(factory: &ParserFactory, id: u32, output: OutputLogprobs) -> Self {
        let top = output
            .top
            .into_iter()
            .map(|(id, logprob)| Self::new(factory, id, logprob, Vec::new()))
            .collect();
        Self::new(factory, id, output.logprob, top)
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
    /// Reasoning the model separates from its answer; empty for models that don't.
    pub reasoning_text: String,
    /// Every generated token except EOS, in order, when the request asked for them. This
    /// includes reasoning and tokens a stop sequence cut from `generated_text`.
    pub logprobs: Vec<TokenLogprob>,
    /// `stop` for EOS, `length` for the request limit.
    pub finish_reason: &'static str,
}

/// Incremental text and independent request outcomes from one packed model invocation.
#[derive(Default)]
pub struct BatchStepOutput {
    pub new_tokens: Vec<(Uuid, String)>,
    /// Text of the reasoning channel, for models that separate it from their answer.
    pub new_reasoning: Vec<(Uuid, String)>,
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

impl Pending {
    /// Take one published output. Returns the finish reason if the request ends on it.
    fn accept(
        &mut self,
        id: Uuid,
        token: u32,
        logprobs: Option<OutputLogprobs>,
        eos_tokens: &[u32],
        output: &mut BatchStepOutput,
    ) -> Result<Option<&'static str>> {
        let sampled_at = Instant::now();
        output.sampled_tokens.push((id, token, sampled_at));
        self.tokens.push(token);
        self.first_token
            .get_or_insert_with(|| sampled_at.saturating_duration_since(self.arrival));
        let eos = eos_tokens.contains(&token);
        if !eos {
            self.decode_token(id, token, output)?;
            if let (Some(factory), Some(logprobs)) = (&self.logprob_tokens, logprobs) {
                self.logprobs
                    .push(TokenLogprob::of_output(factory, token, logprobs));
            }
        }
        let stopped = eos || self.stop.matched();
        Ok(if stopped {
            Some("stop")
        } else {
            (self.generated() >= self.max_tokens).then_some("length")
        })
    }

    /// Hand one request's newly decoded text to its audiences.
    fn publish_fragment(&mut self, id: Uuid, fragment: Fragment, output: &mut BatchStepOutput) {
        if let Some(text) = fragment.reasoning {
            output.new_reasoning.push((id, text));
        }
        if let Some(text) = fragment.content.and_then(|text| self.stop.push(text)) {
            output.new_tokens.push((id, text));
        }
    }

    /// Decode one sampled token and publish whatever text it completes.
    fn decode_token(&mut self, id: Uuid, token: u32, output: &mut BatchStepOutput) -> Result<()> {
        let route = match &mut self.split {
            Some(split) => split.push(token)?,
            None => Route::Answer,
        };
        let fragment = match (route, &mut self.harmony) {
            (Route::Taken(fragment), _) => fragment,
            (Route::Answer, Some(harmony)) => harmony.push(token)?,
            (Route::Answer, None) => Fragment {
                content: self.stream.next_token(token)?,
                reasoning: None,
            },
        };
        self.publish_fragment(id, fragment, output);
        Ok(())
    }

    /// Publish all text still held back and return the final answer, reasoning and calls.
    fn finish_text(
        &mut self,
        id: Uuid,
        output: &mut BatchStepOutput,
    ) -> Result<(String, String, Vec<ToolCall>)> {
        let fragment = match &mut self.harmony {
            Some(harmony) => harmony.finish()?,
            None => Fragment {
                content: self.stream.decode_rest()?,
                reasoning: self
                    .split
                    .as_mut()
                    .map(ReasoningSplit::take_rest)
                    .transpose()?
                    .flatten(),
            },
        };
        self.publish_fragment(id, fragment, output);
        if let Some(text) = self.stop.flush() {
            output.new_tokens.push((id, text));
        }
        let (mut content, reasoning, tool_calls) = if let Some(harmony) = &mut self.harmony {
            (
                harmony.content()?,
                harmony.reasoning()?,
                harmony.tool_calls(&self.tools)?,
            )
        } else {
            let tool_calls = if self.tools.is_empty() {
                Vec::new()
            } else {
                ToolCall::parse_from_output(&self.stream.decode_raw()?, &self.tools)
            };
            let reasoning = self
                .split
                .as_ref()
                .map_or_else(|| Ok(String::new()), ReasoningSplit::reasoning)?;
            (self.stream.decode_all()?, reasoning, tool_calls)
        };
        self.stop.truncate(&mut content);
        Ok((content, reasoning, tool_calls))
    }
}

impl InferencePipeline {
    /// Predict where a new user turn would first rewrite this prompt. Only token
    /// equality at lookup authorizes a restore; the probe just chooses a pass end.
    fn turn_boundary(
        &self,
        request: &GenerationRequest,
        render: &RenderOptions,
        tokens: &[u32],
    ) -> Option<usize> {
        // A first turn has no demonstrated conversational reuse; splitting it adds a
        // pass to ordinary single-turn requests. Start predicting once history exists.
        if !self.prefix.config().enabled()
            || !render.add_generation_prompt
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
            ..render.clone()
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

    /// The parser factory over the tokenizer, built on first use.
    fn parser_factory(&mut self) -> Result<Arc<ParserFactory>> {
        if let Some(factory) = &self.guidance {
            return Ok(Arc::clone(factory));
        }
        let vocab = self.engine.pool().description().vocab_size;
        let factory = build_llg_factory(
            self.tokenizer.clone(),
            Some(vocab),
            &self.defaults.eos_tokens,
        )?;
        self.guidance = Some(Arc::clone(&factory));
        Ok(factory)
    }

    /// The options a structured request renders with: thinking stays off unless asked for,
    /// since Harmony alone reasons by default and most answers don't need it.
    fn render_options(&self, request: &GenerationRequest) -> RenderOptions {
        let mut render = request.render.clone();
        if request.structured.is_some() && self.tool_syntax() != ToolSyntax::Harmony {
            render.enable_thinking.get_or_insert(false);
        }
        render
    }

    fn tool_syntax(&self) -> ToolSyntax {
        self.model.tool_syntax()
    }

    /// Whether structured output may follow reasoning, given the rendered prompt.
    fn preamble(&self, request: &GenerationRequest, prompt: &[u32]) -> Preamble {
        let thinking = request.render.enable_thinking;
        let reasons = match self.tool_syntax() {
            ToolSyntax::Harmony => thinking != Some(false),
            ToolSyntax::Gemma => thinking == Some(true),
            // A template that ignored the request hasn't opened anything to close.
            ToolSyntax::Xml => thinking == Some(true) && self.prompt_opens_thinking(prompt),
        };
        if reasons {
            Preamble::Reasoning
        } else {
            Preamble::None
        }
    }

    fn prompt_opens_thinking(&self, prompt: &[u32]) -> bool {
        let (Some(open), Some(close)) = (
            self.tokenizer.token_to_id("<think>"),
            self.tokenizer.token_to_id("</think>"),
        ) else {
            return false;
        };
        prompt
            .iter()
            .rev()
            .find(|token| **token == open || **token == close)
            == Some(&open)
    }

    /// How to tell a structured answer from the reasoning before it. Harmony's channels already
    /// do; otherwise only a grammar that admits reasoning can produce any.
    fn reasoning_split(&self, preamble: Preamble) -> Option<ReasoningSplit> {
        if preamble != Preamble::Reasoning {
            return None;
        }
        let id = |token: &str| self.tokenizer.token_to_id(token);
        match self.tool_syntax() {
            ToolSyntax::Harmony => None,
            ToolSyntax::Xml => Some(ReasoningSplit::opened(&self.tokenizer, id("</think>")?)),
            ToolSyntax::Gemma => Some(ReasoningSplit::channel(
                &self.tokenizer,
                id("<|channel>")?,
                id("<channel|>")?,
            )),
        }
    }

    /// The request's parser, started. A grammar that doesn't compile is the request's fault.
    fn request_guidance(
        &mut self,
        request: &GenerationRequest,
        preamble: Preamble,
    ) -> Result<Option<GuidanceControl>> {
        let grammar = match (&request.grammar, &request.structured) {
            (Some(_), Some(_)) => return Err(invalid("grammar excludes structured output")),
            (Some(grammar), None) => grammar.clone(),
            (None, Some(output)) => structured::grammar(
                output,
                self.tool_syntax(),
                &request.tools,
                &self.tokenizer,
                preamble,
                &self.defaults.eos_tokens,
            )?,
            (None, None) => return Ok(self.lazy_guidance(request)),
        };
        let mut control = GuidanceControl::new(self.parser_factory()?);
        control
            .activate(grammar)
            .map_err(|error| invalid(format!("invalid grammar: {error}")))?;
        Ok(Some(control))
    }

    /// A dormant grammar holding this request's tool calls to their schemas. It is best effort:
    /// a schema it can't enforce leaves the calls unconstrained, as they would be anyway, and
    /// must not fail a request that may never make one.
    fn lazy_guidance(&mut self, request: &GenerationRequest) -> Option<GuidanceControl> {
        let attempt = (|| -> Result<Option<GuidanceControl>> {
            let Some((grammar, trigger)) =
                structured::lazy_call(self.tool_syntax(), &request.tools, &self.tokenizer)?
            else {
                return Ok(None);
            };
            let mut control = GuidanceControl::new(self.parser_factory()?);
            control.arm(grammar, trigger)?;
            Ok(Some(control))
        })();
        attempt.unwrap_or_else(|error| {
            tracing::debug!(%error, "tool calls run without a grammar");
            None
        })
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
        let render = self.render_options(request);
        let prompt = self
            .template
            .render(&request.messages, &request.tools, &render)?;
        let tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| TurError::Tokenizer(error.to_string()))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            return Err(invalid("chat template produced an empty prompt"));
        }
        let (tokens, audio) = self.expand_audio(request, tokens)?;
        let limit = self.plan.context_tokens;
        let max_tokens = if request.fit_context {
            request
                .max_tokens
                .min(limit.saturating_add(1).saturating_sub(tokens.len()))
                .max(1)
        } else {
            request.max_tokens
        };
        // The final sampled token need not be consumed into model state.
        let positions = tokens
            .len()
            .checked_add(max_tokens - 1)
            .ok_or_else(|| invalid("request context length overflow"))?;
        if positions > limit {
            return Err(TurError::ContextExceeded(format!(
                "{} prompt tokens and {max_tokens} output tokens need {positions} positions; \
                 this server allows {limit}",
                tokens.len(),
            )));
        }
        let stop = StopMatcher::new(&request.stop)?;
        let (waiting, waiting_tokens) = self
            .pending
            .values()
            .filter(|p| !p.running)
            .fold((0usize, 0usize), |(n, t), p| {
                (n + 1, t.saturating_add(p.tokens.len() - p.consumed))
            });
        self.scheduler
            .admit(waiting, waiting_tokens, tokens.len())?;
        let turn_boundary = self.turn_boundary(request, &render, &tokens);
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(TurError::DeadlineExceeded);
        }
        let options = sampling(request.sampling, &self.defaults);
        let preamble = if request.structured.is_some() {
            self.preamble(request, &tokens)
        } else {
            Preamble::None
        };
        let guidance = self.request_guidance(request, preamble)?;
        let logprob_tokens = options
            .logprobs
            .map(|_| self.parser_factory())
            .transpose()?;
        let handle = self.engine.allocate(request.seed, options, guidance)?;
        // Audio positions are not tokens the repetition penalty could see repeated.
        let text: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|&t| !super::audio::is_identity(t))
            .collect();
        self.engine.add_context(handle, &text)?;
        let keys = self.prefix.keys(request.cache_salt.as_deref(), &tokens);
        self.pending.insert(
            handle.request,
            Pending {
                handle,
                prompt_tokens: tokens.len(),
                turn_boundary,
                tokens,
                audio,
                consumed: 0,
                boundaries: Vec::new(),
                replay_floor: 0,
                cached_tokens: 0,
                keys,
                prefix_checked: None,
                running: false,
                stream: TokenOutputStream::new(self.tokenizer.clone()),
                harmony: (self.tool_syntax() == ToolSyntax::Harmony)
                    .then(|| Harmony::new(&self.tokenizer))
                    .flatten(),
                stop,
                max_tokens,
                arrival,
                deadline,
                first_token: None,
                last_run: 0,
                preemptions: 0,
                tools: request.tools.clone(),
                logprob_tokens,
                logprobs: Vec::new(),
                split: self.reasoning_split(preamble),
                greedy: options.greedy(),
                ngrams: NgramIndex::default(),
                acceptance: self.acceptance,
                mtp_done: 0,
                mtp_drafts: Vec::new(),
                mtp_acceptance: self.mtp_acceptance,
                mtp_verifying: Vec::new(),
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
        output.new_reasoning.retain(|(request, _)| *request != id);
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
        pending.mtp_done = 0;
        pending.mtp_drafts.clear();
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
            if let Some(ended) = pending.accept(
                id,
                token.token,
                token.logprobs,
                &self.defaults.eos_tokens,
                output,
            )? {
                finished = true;
                reason = ended;
                break;
            }
        }
        // Tokens its grammar alone decides join the outputs here, and the next pass reads them
        // as inputs, as it reads a prompt's.
        while !finished {
            match self.engine.force_output(pending.handle)? {
                Forced::Sample => break,
                Forced::Stop => finished = true,
                Forced::Token(token) => {
                    if let Some(ended) =
                        pending.accept(id, token, None, &self.defaults.eos_tokens, output)?
                    {
                        finished = true;
                        reason = ended;
                    }
                }
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
        let (generated_text, reasoning_text, tool_calls) = pending.finish_text(id, output)?;
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
            reasoning_text,
            logprobs: std::mem::take(&mut pending.logprobs),
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
