use candle_core::{DType, Device};
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info, trace, warn};
use uuid::Uuid;

use crate::{
    ProgressReporter, Result, TurError,
    backend::{
        audio_encoder::AudioEncoder,
        chat_template::{ChatTemplate, Message},
        engine::InferenceEngine,
        factory::{ModelConstructor, ModelFactory},
        guidance::{ParserFactory, TopLevelGrammar},
        scheduler::{ContinuousBatchScheduler, SchedulingPolicy},
        tokenizer::TokenOutputStream,
        tools::{ToolCall, ToolDefinition},
    },
    models::kv_cache::BlockAllocator,
};

/// A single modality input to a generation request.
#[derive(Debug, Clone)]
pub enum ModalInput {
    /// Plain text — the normal prompt string.
    Text(String),
    /// Raw PCM audio samples (mono f32) at the given sample rate.
    Audio { pcm: Vec<f32>, sample_rate: u32 },
}

/// Request struct for generation, supporting text and audio inputs.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub id: Uuid,
    /// Ordered list of inputs.  Use [`GenerationRequest::new`] for text-only
    /// requests or [`with_audio`] to append audio inputs.
    pub inputs: Vec<ModalInput>,
    pub sample_len: usize,
    /// Optional grammar for guided (constrained) generation.
    /// When set, only tokens valid under the grammar are sampled.
    pub grammar: Option<TopLevelGrammar>,
    /// Tools available to the model.  When non-empty the pipeline injects them
    /// into the prompt via the model's `format_prompt_with_tools` and parses
    /// any [`ToolCall`]s from the generated text, returning them in
    /// [`GenerationStats::tool_calls`].
    pub tools: Vec<ToolDefinition>,
    /// Enable extended thinking (`/think` tag in Qwen3).  Only effective when
    /// tools are also set; for tool-free requests the caller is responsible for
    /// pre-formatting the prompt with the appropriate tag.
    pub thinking: bool,
    /// When `true`, skip chat-template formatting and pass the text prompt
    /// directly to the tokeniser.  Used by the OpenAI-compatible server after
    /// it has already formatted the full conversation via `format_messages`.
    pub raw: bool,
}

impl GenerationRequest {
    /// Create a text-only request with auto-generated UUID.
    pub fn new(prompt: String, sample_len: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            inputs: vec![ModalInput::Text(prompt)],
            sample_len,
            grammar: None,
            tools: Vec::new(),
            thinking: false,
            raw: false,
        }
    }

    /// Return the first text input, or an empty string when there is none.
    /// Used internally to extract the prompt for template formatting and tokenisation.
    pub fn text_prompt(&self) -> &str {
        self.inputs
            .iter()
            .find_map(|i| match i {
                ModalInput::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("")
    }

    /// Append a PCM audio input (mono f32 at `sample_rate` Hz).
    pub fn with_audio(mut self, pcm: Vec<f32>, sample_rate: u32) -> Self {
        self.inputs.push(ModalInput::Audio { pcm, sample_rate });
        self
    }

    /// Attach a grammar for guided generation.
    pub fn with_grammar(mut self, grammar: TopLevelGrammar) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Attach tool definitions.  The text prompt is treated as a raw user message
    /// and the pipeline formats it with the model's tool-calling template.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Enable extended thinking mode (Qwen3 `/think` tag).
    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    /// Skip chat-template formatting and pass the text prompt directly to the
    /// tokeniser.  Use this when the caller has already formatted the full
    /// conversation (e.g. the OpenAI-compatible server pre-formats multi-turn
    /// chats via [`InferencePipeline::format_messages`]).
    pub fn with_raw(mut self, raw: bool) -> Self {
        self.raw = raw;
        self
    }
}

/// Statistics for generation runs
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub generated_tokens: usize,
    pub elapsed: std::time::Duration,
    /// Tool calls parsed from the generated text.  Non-empty only when the
    /// request included tool definitions and the model emitted `<tool_call>`
    /// blocks in its output.
    pub tool_calls: Vec<ToolCall>,
}

/// Result handle for async request tracking
#[derive(Debug, Clone)]
pub struct RequestHandle {
    pub id: Uuid,
}

/// Completed generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub request_id: Uuid,
    pub prompt: String,
    pub generated_text: String,
    pub generated_tokens: Vec<u32>,
    pub stats: GenerationStats,
}

/// Components that are only present when batching is enabled.
struct BatchingComponents {
    scheduler: ContinuousBatchScheduler,
    tokenizer: Arc<Tokenizer>,
}

pub type OnTokenFn = Box<dyn FnMut(&str)>;

/// High-level generation pipeline with output handling, continuous batching, and multimodal support.
pub struct InferencePipeline<T: ModelConstructor> {
    engine: InferenceEngine<T>,
    tokenizer: TokenOutputStream,
    progress: Option<ProgressReporter>,
    on_token: Option<OnTokenFn>,
    batching: Option<BatchingComponents>,
    results: Arc<RwLock<HashMap<Uuid, GenerationResult>>>,
    /// Insertion-order tracking for bounded eviction of `results`.
    result_order: VecDeque<Uuid>,
    max_results: usize,
    /// Jinja2 chat template loaded from the model's `tokenizer_config.json`.
    /// When present, `format_prompt` and `format_prompt_with_tools` use it
    /// instead of the hardcoded static fallback on `ModelImpl`.
    chat_template: Option<ChatTemplate>,
    /// Optional audio encoder for multimodal requests.
    /// When present, [`ModalInput::Audio`] inputs are encoded before prefill.
    audio_encoder: Option<Box<dyn AudioEncoder>>,
}

/// Builder for [`InferencePipeline`].
pub struct InferencePipelineBuilder<'a, T: ModelConstructor> {
    factory: &'a ModelFactory<T>,
    device: Device,
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    progress: Option<ProgressReporter>,
    on_token: Option<OnTokenFn>,
    prefix_cache: Option<super::prefix_cache::SharedPrefixCache>,
    guidance_factory: Option<Arc<ParserFactory>>,
    audio_encoder: Option<Box<dyn AudioEncoder>>,
    // Batching configuration
    enable_batching: bool,
    max_batch_size: usize,
    max_prefill_batch: usize,
    max_decode_batch: usize,
    scheduling_policy: SchedulingPolicy,
    max_results: usize,
    prefill_chunk_size: Option<usize>,
}

impl<'a, T: ModelConstructor> InferencePipelineBuilder<'a, T> {
    pub fn new(factory: &'a ModelFactory<T>, device: Device) -> Self {
        Self {
            factory,
            device,
            seed: 299_792_458,
            temp: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            progress: None,
            on_token: None,
            prefix_cache: None,
            guidance_factory: None,
            audio_encoder: None,
            enable_batching: false,
            max_batch_size: 16,
            max_prefill_batch: 8,
            max_decode_batch: 16,
            scheduling_policy: SchedulingPolicy::FCFS,
            max_results: 10_000,
            prefill_chunk_size: None,
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temp = Some(temp);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = penalty;
        self
    }

    pub fn repeat_last_n(mut self, n: usize) -> Self {
        self.repeat_last_n = n;
        self
    }

    pub fn progress(mut self, progress: ProgressReporter) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn on_token<F: FnMut(&str) + 'static>(mut self, callback: F) -> Self {
        self.on_token = Some(Box::new(callback));
        self
    }

    pub fn with_prefix_cache(mut self, max_entries: usize, max_token_length: usize) -> Self {
        self.prefix_cache = Some(Arc::new(RwLock::new(
            super::prefix_cache::PrefixCache::new(max_entries, max_token_length),
        )));
        self
    }

    /// Use a shared prefix cache instance
    pub fn with_shared_prefix_cache(
        mut self,
        cache: super::prefix_cache::SharedPrefixCache,
    ) -> Self {
        self.prefix_cache = Some(cache);
        self
    }

    pub fn enable_batching(mut self, enable: bool) -> Self {
        self.enable_batching = enable;
        self
    }

    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn max_prefill_batch(mut self, size: usize) -> Self {
        self.max_prefill_batch = size;
        self
    }

    pub fn max_decode_batch(mut self, size: usize) -> Self {
        self.max_decode_batch = size;
        self
    }

    pub fn scheduling_policy(mut self, policy: SchedulingPolicy) -> Self {
        self.scheduling_policy = policy;
        self
    }

    /// Maximum number of completed results to retain before evicting oldest entries.
    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Split each prompt into chunks of `chunk_size` tokens during prefill.
    ///
    /// Each scheduler iteration processes at most `chunk_size` new prompt tokens
    /// per request.  This bounds per-iteration memory (attention is O(chunk²) not
    /// O(prompt²)) and lets decode-phase requests interleave with in-progress
    /// prefills instead of waiting for a full long-prompt prefill to complete.
    ///
    /// Only effective when batching is enabled (paged KV caches are required to
    /// hold intermediate KV state between chunks).  Setting `chunk_size` larger
    /// than any prompt degrades gracefully to the single-shot behaviour.
    pub fn prefill_chunk_size(mut self, chunk_size: usize) -> Self {
        self.prefill_chunk_size = Some(chunk_size);
        self
    }

    /// Enable guided (constrained) generation using the given parser factory.
    /// Build the factory with `guidance::build_llg_factory`.
    pub fn with_guidance_factory(mut self, factory: Arc<ParserFactory>) -> Self {
        self.guidance_factory = Some(factory);
        self
    }

    /// Attach an audio encoder for multimodal requests.
    /// When set, [`ModalInput::Audio`] inputs in each request are encoded to
    /// embeddings before prefill via [`InferenceEngine::prefill_with_audio`].
    pub fn with_audio_encoder(mut self, encoder: Box<dyn AudioEncoder>) -> Self {
        self.audio_encoder = Some(encoder);
        self
    }

    pub fn build(self) -> InferencePipeline<T> {
        debug!(
            seed = self.seed,
            temp = ?self.temp,
            top_p = ?self.top_p,
            repeat_penalty = self.repeat_penalty,
            repeat_last_n = self.repeat_last_n,
            has_on_token = self.on_token.is_some(),
            has_prefix_cache = self.prefix_cache.is_some(),
            enable_batching = self.enable_batching,
            max_batch_size = self.max_batch_size,
            max_prefill_batch = self.max_prefill_batch,
            max_decode_batch = self.max_decode_batch,
            scheduling_policy = ?self.scheduling_policy,
            prefill_chunk_size = ?self.prefill_chunk_size,
            "Building text generation pipeline with configuration"
        );

        let mut engine_builder = InferenceEngine::builder(self.factory, self.device.clone())
            .seed(self.seed)
            .repeat_penalty(self.repeat_penalty)
            .repeat_last_n(self.repeat_last_n);

        if let Some(temp) = self.temp {
            engine_builder = engine_builder.temperature(temp);
        }
        if let Some(top_p) = self.top_p {
            engine_builder = engine_builder.top_p(top_p);
        }
        if let Some(progress) = &self.progress {
            engine_builder = engine_builder.with_progress(progress.clone());
        }
        if let Some(prefix_cache) = &self.prefix_cache {
            engine_builder = engine_builder.with_shared_prefix_cache(prefix_cache.clone());
        }
        if let Some(guidance_factory) = self.guidance_factory {
            engine_builder = engine_builder.with_guidance_factory(guidance_factory);
        }

        let (batching, engine, tokenizer, chat_template) = if self.enable_batching {
            let block_size = 64;
            let num_heads = 32;
            let head_dim = 128;
            let dtype = DType::BF16;
            let max_decode_tokens = 4096;
            let max_prefill_tokens = 2048;
            let memory_pool_size_bytes = 8 * 1024 * 1024 * 1024;
            let total_blocks = 1024;

            let block_allocator =
                Arc::new(RwLock::new(BlockAllocator::new(total_blocks, block_size)));

            let (engine, tokenizer, chat_template) =
                engine_builder.build().expect("Failed to build engine");
            let tokenizer_arc = Arc::new(tokenizer.clone());

            // Derive num_layers and EOS tokens from the just-built model/tokenizer so
            // they always agree with the actual model, never with hardcoded constants.
            let num_layers = engine.model().num_layers();
            let model_eos_ids = engine.model().eos_token_ids();
            let (eos_token, im_end_token) = match model_eos_ids.as_slice() {
                [] => (
                    tokenizer_arc
                        .token_to_id("<|endoftext|>")
                        .or_else(|| tokenizer_arc.token_to_id("<|end_of_text|>"))
                        .expect("cannot find EOS token"),
                    tokenizer_arc
                        .token_to_id("<|im_end|>")
                        .or_else(|| tokenizer_arc.token_to_id("<|end_of_role|>"))
                        .or_else(|| tokenizer_arc.token_to_id("<|endoftext|>"))
                        .or_else(|| tokenizer_arc.token_to_id("<|end_of_text|>"))
                        .expect("cannot find secondary EOS token"),
                ),
                [a] => (*a, *a),
                [a, b, ..] => (*a, *b),
            };

            debug!(
                total_blocks,
                block_size,
                num_heads,
                head_dim,
                dtype = ?dtype,
                memory_pool_size_bytes,
                max_prefill_tokens,
                max_decode_tokens,
                num_layers,
                eos_token,
                im_end_token,
                "Initializing continuous batching components"
            );

            let memory_pool = Arc::new(RwLock::new(crate::backend::memory_pool::MemoryPool::new(
                memory_pool_size_bytes,
                block_size,
                32,
                num_heads,
                head_dim,
                2,
            )));

            let batch_config = crate::backend::scheduler::BatchConfig {
                max_prefill_batch: self.max_prefill_batch,
                max_decode_batch: self.max_decode_batch,
                max_prefill_tokens,
                max_decode_tokens,
                prefill_chunk_size: self.prefill_chunk_size,
            };

            let scheduler = ContinuousBatchScheduler::new(
                self.scheduling_policy,
                batch_config,
                memory_pool,
                block_allocator,
                num_layers,
                (eos_token, im_end_token),
            );

            let batching = Some(BatchingComponents {
                scheduler,
                tokenizer: tokenizer_arc,
            });
            (batching, engine, tokenizer, chat_template)
        } else {
            debug!("Batching disabled, using single-request mode");
            let (engine, tokenizer, chat_template) =
                engine_builder.build().expect("Failed to build engine");
            (None, engine, tokenizer, chat_template)
        };

        debug!("Text generation pipeline built successfully");

        InferencePipeline {
            engine,
            tokenizer: TokenOutputStream::new(tokenizer),
            progress: self.progress,
            on_token: self.on_token,
            batching,
            results: Arc::new(RwLock::new(HashMap::new())),
            result_order: VecDeque::new(),
            max_results: self.max_results,
            chat_template,
            audio_encoder: self.audio_encoder,
        }
    }
}

impl<T: ModelConstructor> InferencePipeline<T> {
    /// Create a builder for [`InferencePipeline`] from factory.
    pub fn builder(
        factory: &'_ ModelFactory<T>,
        device: Device,
    ) -> InferencePipelineBuilder<'_, T> {
        InferencePipelineBuilder::new(factory, device)
    }

    /// Access the underlying inference engine
    pub fn engine(&self) -> &InferenceEngine<T> {
        &self.engine
    }

    /// Access the underlying inference engine mutably
    pub fn engine_mut(&mut self) -> &mut InferenceEngine<T> {
        &mut self.engine
    }

    pub fn run(&mut self, request: &GenerationRequest) -> Result<GenerationStats> {
        if let Some(grammar) = request.grammar.clone() {
            self.engine.activate_grammar(grammar)?;
        }
        let result = self.run_inner(request);
        self.engine.deactivate_grammar();
        result
    }

    fn run_inner(&mut self, request: &GenerationRequest) -> Result<GenerationStats> {
        let sample_len = request.sample_len;
        let has_tools = !request.tools.is_empty();
        self.tokenizer.clear();

        let text = request.text_prompt();
        let prompt_str = if request.raw {
            text.to_string()
        } else if has_tools {
            self.format_prompt_with_tools(text, &request.tools, request.thinking)
        } else {
            self.format_prompt(text, request.thinking)
        };

        trace!(
            request_id = %request.id,
            prompt_chars = prompt_str.chars().count(),
            has_tools,
            sample_len,
            "Starting generation request",
        );
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt_str.as_str(), true)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        trace!(
            request_id = %request.id,
            prompt_tokens = tokens.len(),
            "Prompt encoded into tokens",
        );

        if let Some(ref progress) = self.progress {
            progress.init_generation(sample_len);
        }

        let mut generated_tokens = 0usize;
        // Accumulate text for tool-call parsing when tools are active.
        let mut generated_text: Option<String> = if has_tools { Some(String::new()) } else { None };
        let model_eos_ids = self.engine.model().eos_token_ids();
        let (eos_token, im_end_token) =
            InferenceEngine::<T>::resolve_eos_tokens(&model_eos_ids, &self.tokenizer)?;

        let start_gen = std::time::Instant::now();

        // Encode audio inputs when an encoder is available, then dispatch to the
        // appropriate prefill path.
        let audio_embeds = self.encode_audio_inputs(&request.inputs)?;
        let (first_token, _, cache_hit, cached_tokens) = if audio_embeds.is_empty() {
            self.engine.prefill(&tokens)?
        } else {
            self.engine.prefill_with_audio(&tokens, audio_embeds)?
        };
        trace!(
            request_id = %request.id,
            first_token,
            cache_hit,
            cached_tokens,
            "Engine prefetched prompt and produced first token",
        );
        tokens.push(first_token);
        generated_tokens += 1;

        if let Some(ref progress) = self.progress {
            progress.inc_generation();
        }

        if first_token != eos_token
            && first_token != im_end_token
            && let Some(t) = self.tokenizer.next_token(first_token)?
        {
            trace!(
                request_id = %request.id,
                text_chars = t.chars().count(),
                "Decoded first token into text chunk",
            );
            if let Some(ref mut buf) = generated_text {
                buf.push_str(&t);
            }
            if let Some(ref mut on_token) = self.on_token {
                on_token(&t);
            }
        }

        // Decode phase
        for _ in 1..sample_len {
            if tokens[tokens.len() - 1] == eos_token || tokens[tokens.len() - 1] == im_end_token {
                break;
            }

            let start_pos = tokens.len() - 1;
            let next_token = self.engine.decode_step(&tokens, start_pos)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if let Some(ref progress) = self.progress {
                progress.inc_generation();
            }

            if next_token == eos_token || next_token == im_end_token {
                trace!(
                    request_id = %request.id,
                    next_token,
                    generated_tokens,
                    "Decode loop reached EOS token",
                );
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                trace!(
                    request_id = %request.id,
                    next_token,
                    text_chars = t.chars().count(),
                    "Decoded token into text chunk",
                );
                if let Some(ref mut buf) = generated_text {
                    buf.push_str(&t);
                }
                if let Some(ref mut on_token) = self.on_token {
                    on_token(&t);
                }
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self
            .tokenizer
            .decode_rest()
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
        {
            trace!(
                request_id = %request.id,
                text_chars = rest.chars().count(),
                "Flushed remaining decoded text",
            );
            if let Some(ref mut buf) = generated_text {
                buf.push_str(&rest);
            }
            if let Some(ref mut on_token) = self.on_token {
                on_token(&rest);
            }
        }

        if let Some(ref progress) = self.progress {
            progress.flush_text();
        }

        trace!(
            request_id = %request.id,
            generated_tokens,
            elapsed_ms = dt.as_secs_f64() * 1000.0,
            "Generation request completed",
        );

        if let Some(ref progress) = self.progress {
            progress.finish_generation(generated_tokens, dt.as_secs_f64());
        } else {
            info!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }

        let tool_calls = generated_text
            .as_deref()
            .map(ToolCall::parse_from_output)
            .unwrap_or_default();

        Ok(GenerationStats {
            generated_tokens,
            elapsed: dt,
            tool_calls,
        })
    }

    /// Encode all [`ModalInput::Audio`] entries in `inputs` using the pipeline's
    /// audio encoder.  Returns an empty `Vec` when no encoder is configured or
    /// when there are no audio inputs, leaving the normal text-only path intact.
    fn encode_audio_inputs(&self, inputs: &[ModalInput]) -> Result<Vec<candle_core::Tensor>> {
        let Some(enc) = &self.audio_encoder else {
            return Ok(Vec::new());
        };
        inputs
            .iter()
            .filter_map(|i| match i {
                ModalInput::Audio { pcm, sample_rate } => Some(enc.encode(pcm, *sample_rate)),
                _ => None,
            })
            .collect()
    }

    /// Format a raw user message using the loaded Jinja2 chat template when
    /// available, or the model's static `format_prompt` implementation otherwise.
    pub fn format_prompt(&self, prompt: &str, thinking: bool) -> String {
        self.chat_template
            .as_ref()
            .and_then(|ct| {
                ct.format(&[Message::user(prompt)], None, true, thinking)
                    .map_err(|e| {
                        tracing::warn!(
                            "Chat template render error in format_prompt: {e}; \
                             falling back to static format"
                        );
                        e
                    })
                    .ok()
            })
            .unwrap_or_else(|| self.engine.model().format_prompt(prompt, thinking))
    }

    /// Format a full multi-turn conversation using the loaded Jinja2 chat
    /// template.  Falls back to formatting only the last user message through
    /// the model's static `format_prompt` when no template is available.
    ///
    /// Pair with [`GenerationRequest::with_raw`] so `run` does not re-wrap the
    /// already-formatted prompt in another user turn.
    pub fn format_messages(
        &self,
        messages: &[crate::backend::chat_template::Message],
        tools: Option<&[crate::backend::tools::ToolDefinition]>,
        thinking: bool,
    ) -> String {
        let non_empty_tools = tools.filter(|t| !t.is_empty());
        if let Some(ct) = &self.chat_template {
            match ct.format(messages, non_empty_tools, true, thinking) {
                Ok(s) => return s,
                Err(e) => {
                    tracing::warn!(
                        "Chat template render error in format_messages: {e}; falling back"
                    );
                }
            }
        }
        // Full ChatML fallback: format every message so context is preserved.
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    /// Format a raw user message with tool definitions using the loaded Jinja2
    /// chat template when available, or the static `format_prompt_with_tools`
    /// otherwise.  Useful for inspecting the prompt before submitting a request.
    pub fn format_prompt_with_tools(
        &self,
        prompt: &str,
        tools: &[ToolDefinition],
        thinking: bool,
    ) -> String {
        self.chat_template
            .as_ref()
            .and_then(|ct| {
                ct.format(&[Message::user(prompt)], Some(tools), true, thinking)
                    .map_err(|e| {
                        tracing::warn!(
                            "Chat template render error in format_prompt_with_tools: {e}; \
                             falling back to static format"
                        );
                        e
                    })
                    .ok()
            })
            .unwrap_or_else(|| {
                self.engine
                    .model()
                    .format_prompt_with_tools(prompt, tools, thinking)
            })
    }

    /// Submit a new generation request (batching mode only).
    /// Audio inputs are not supported in batching mode — only the text prompt is used.
    pub fn submit_request(&mut self, request: &GenerationRequest) -> Result<RequestHandle> {
        let batching = self.batching.as_mut().ok_or_else(|| {
            TurError::Other("Batching not enabled. Use builder.enable_batching(true)".to_string())
        })?;

        let prompt = request.text_prompt();
        let tokens = batching
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();

        trace!(
            request_id = %request.id,
            prompt_tokens = tokens.len(),
            sample_len = request.sample_len,
            "Submitting request to scheduler"
        );

        let request_state = crate::backend::batch_manager::RequestState::new(
            request.id,
            prompt.to_string(),
            tokens,
            request.sample_len,
            0,
        );

        batching.scheduler.enqueue_request(request_state);

        Ok(RequestHandle { id: request.id })
    }

    /// Run one iteration of continuous batching (batching mode only).
    /// Returns the number of active requests.
    pub fn step(&mut self) -> Result<usize> {
        let batching = self
            .batching
            .as_mut()
            .ok_or_else(|| TurError::Other("Batching not enabled".to_string()))?;

        let completed = batching.scheduler.schedule_iteration(&mut self.engine)?;

        let mut results = self.results.write();
        for (request_id, tokens, prompt, arrival_time) in completed {
            let generated_text = batching
                .tokenizer
                .decode(&tokens, true)
                .map_err(|e| TurError::Tokenizer(e.to_string()))?;

            let result = GenerationResult {
                request_id,
                prompt,
                generated_text,
                generated_tokens: tokens.clone(),
                stats: GenerationStats {
                    generated_tokens: tokens.len(),
                    elapsed: arrival_time.elapsed(),
                    tool_calls: Vec::new(),
                },
            };

            // Evict oldest result when at capacity.
            if self.result_order.len() >= self.max_results
                && let Some(oldest) = self.result_order.pop_front()
            {
                results.remove(&oldest);
                warn!(
                    evicted_request_id = %oldest,
                    max_results = self.max_results,
                    "Results map at capacity; evicted oldest completed result. \
                     Call clear_results() periodically to avoid this.",
                );
            }

            results.insert(request_id, result);
            self.result_order.push_back(request_id);

            trace!(
                request_id = %request_id,
                generated_tokens = tokens.len(),
                "Request completed and result stored"
            );
        }

        let active = self
            .batching
            .as_ref()
            .map(|b| b.scheduler.active_request_count())
            .unwrap_or(0);
        Ok(active)
    }

    /// Run continuous batching until all requests are completed (batching mode only)
    pub fn run_until_complete(&mut self) -> Result<()> {
        if self.batching.is_none() {
            return Err(TurError::Other("Batching not enabled".to_string()));
        }

        while self
            .batching
            .as_ref()
            .map(|b| b.scheduler.has_pending_work())
            .unwrap_or(false)
        {
            self.step()?;
        }
        Ok(())
    }

    /// Try to get a completed result (non-blocking, batching mode only)
    pub fn try_get_result(&self, handle: &RequestHandle) -> Option<GenerationResult> {
        self.results.read().get(&handle.id).cloned()
    }

    /// Wait for a result to complete (blocking, batching mode only)
    pub fn get_result(&mut self, handle: &RequestHandle) -> Result<GenerationResult> {
        if let Some(result) = self.try_get_result(handle) {
            return Ok(result);
        }

        if self.batching.is_none() {
            return Err(TurError::Other("Batching not enabled".to_string()));
        }

        loop {
            if let Some(result) = self.try_get_result(handle) {
                return Ok(result);
            }

            let has_work = self
                .batching
                .as_ref()
                .map(|b| b.scheduler.has_pending_work())
                .unwrap_or(false);

            if !has_work {
                return Err(TurError::Other(format!(
                    "Request {} not found and no active requests",
                    handle.id
                )));
            }

            self.step()?;
        }
    }

    /// Get all completed results (batching mode only)
    pub fn get_all_results(&self) -> HashMap<Uuid, GenerationResult> {
        self.results.read().clone()
    }

    /// Clear completed results (batching mode only)
    pub fn clear_results(&mut self) {
        self.results.write().clear();
        self.result_order.clear();
    }

    /// Get number of active requests (batching mode only)
    pub fn active_request_count(&self) -> usize {
        self.batching
            .as_ref()
            .map(|b| b.scheduler.active_request_count())
            .unwrap_or(0)
    }

    /// Get number of queued requests (batching mode only)
    pub fn queued_request_count(&self) -> usize {
        self.batching
            .as_ref()
            .map(|b| b.scheduler.queued_request_count())
            .unwrap_or(0)
    }

    /// Check if batching is enabled
    pub fn is_batching_enabled(&self) -> bool {
        self.batching.is_some()
    }
}
