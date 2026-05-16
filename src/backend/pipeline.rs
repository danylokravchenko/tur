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
        engine::InferenceEngine,
        factory::{ModelConstructor, ModelFactory},
        scheduler::{ContinuousBatchScheduler, SchedulingPolicy},
        tokenizer::TokenOutputStream,
    },
    models::kv_cache::BlockAllocator,
};

/// Request struct for text generation, suitable for scheduling and prefix-caching
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub id: Uuid,
    pub prompt: String,
    pub sample_len: usize,
}

impl GenerationRequest {
    /// Create a new request with auto-generated UUID
    pub fn new(prompt: String, sample_len: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            prompt,
            sample_len,
        }
    }
}

/// Statistics for generation runs
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub generated_tokens: usize,
    pub elapsed: std::time::Duration,
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

/// High-level text generation pipeline with output handling and continuous batching support
pub struct TextGeneration<T: ModelConstructor> {
    engine: InferenceEngine<T>,
    tokenizer: TokenOutputStream,
    progress: Option<ProgressReporter>,
    emit_output: bool,
    batching: Option<BatchingComponents>,
    results: Arc<RwLock<HashMap<Uuid, GenerationResult>>>,
    /// Insertion-order tracking for bounded eviction of `results`.
    result_order: VecDeque<Uuid>,
    max_results: usize,
}

/// Builder for TextGeneration
pub struct TextGenerationBuilder<'a, T: ModelConstructor> {
    factory: &'a ModelFactory<T>,
    device: Device,
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    progress: Option<ProgressReporter>,
    emit_output: bool,
    prefix_cache: Option<super::prefix_cache::SharedPrefixCache>,
    // Batching configuration
    enable_batching: bool,
    max_batch_size: usize,
    max_prefill_batch: usize,
    max_decode_batch: usize,
    scheduling_policy: SchedulingPolicy,
    max_results: usize,
}

impl<'a, T: ModelConstructor> TextGenerationBuilder<'a, T> {
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
            emit_output: true,
            prefix_cache: None,
            enable_batching: false,
            max_batch_size: 16,
            max_prefill_batch: 8,
            max_decode_batch: 16,
            scheduling_policy: SchedulingPolicy::FCFS,
            max_results: 10_000,
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

    pub fn emit_output(mut self, emit: bool) -> Self {
        self.emit_output = emit;
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

    pub fn build(self) -> TextGeneration<T> {
        debug!(
            seed = self.seed,
            temp = ?self.temp,
            top_p = ?self.top_p,
            repeat_penalty = self.repeat_penalty,
            repeat_last_n = self.repeat_last_n,
            emit_output = self.emit_output,
            has_prefix_cache = self.prefix_cache.is_some(),
            enable_batching = self.enable_batching,
            max_batch_size = self.max_batch_size,
            max_prefill_batch = self.max_prefill_batch,
            max_decode_batch = self.max_decode_batch,
            scheduling_policy = ?self.scheduling_policy,
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

        let (batching, engine, tokenizer) = if self.enable_batching {
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

            let (engine, tokenizer) = engine_builder.build().expect("Failed to build engine");
            let tokenizer_arc = Arc::new(tokenizer.clone());

            // Derive num_layers and EOS tokens from the just-built model/tokenizer so
            // they always agree with the actual model, never with hardcoded constants.
            let num_layers = engine.model().num_layers();
            let eos_token = tokenizer_arc
                .token_to_id("<|endoftext|>")
                .expect("tokenizer missing <|endoftext|>");
            let im_end_token = tokenizer_arc
                .token_to_id("<|im_end|>")
                .expect("tokenizer missing <|im_end|>");

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
            (batching, engine, tokenizer)
        } else {
            debug!("Batching disabled, using single-request mode");
            let (engine, tokenizer) = engine_builder.build().expect("Failed to build engine");
            (None, engine, tokenizer)
        };

        debug!("Text generation pipeline built successfully");

        TextGeneration {
            engine,
            tokenizer: TokenOutputStream::new(tokenizer),
            progress: self.progress,
            emit_output: self.emit_output,
            batching,
            results: Arc::new(RwLock::new(HashMap::new())),
            result_order: VecDeque::new(),
            max_results: self.max_results,
        }
    }
}

impl<T: ModelConstructor> TextGeneration<T> {
    /// Create a builder for TextGeneration from factory
    pub fn builder(factory: &'_ ModelFactory<T>, device: Device) -> TextGenerationBuilder<'_, T> {
        TextGenerationBuilder::new(factory, device)
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
        let sample_len = request.sample_len;
        self.tokenizer.clear();
        trace!(
            request_id = %request.id,
            prompt_chars = request.prompt.chars().count(),
            sample_len,
            "Starting text generation request",
        );
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(request.prompt.as_str(), true)
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
        let (eos_token, im_end_token) = InferenceEngine::<T>::get_eos_tokens(&self.tokenizer)?;

        let start_gen = std::time::Instant::now();

        // Prefill phase
        let (first_token, _, cache_hit, cached_tokens) = self.engine.prefill(&tokens)?;
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
            if let Some(ref progress) = self.progress {
                progress.print(&t);
            } else if self.emit_output {
                use std::io::Write;
                print!("{t}");
                std::io::stdout().flush()?;
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
                if let Some(ref progress) = self.progress {
                    progress.print(&t);
                } else if self.emit_output {
                    use std::io::Write;
                    print!("{t}");
                    std::io::stdout().flush()?;
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
            if let Some(ref progress) = self.progress {
                progress.print(&rest);
            } else if self.emit_output {
                print!("{rest}");
            }
        }

        if let Some(ref progress) = self.progress {
            progress.flush_text();
        } else if self.emit_output {
            use std::io::Write;
            std::io::stdout().flush()?;
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
        Ok(GenerationStats {
            generated_tokens,
            elapsed: dt,
        })
    }

    pub fn format_prompt(&self, prompt: &str, thinking: bool) -> String {
        T::format_prompt(prompt, thinking)
    }

    /// Submit a new generation request (batching mode only)
    pub fn submit_request(&mut self, request: &GenerationRequest) -> Result<RequestHandle> {
        let batching = self.batching.as_mut().ok_or_else(|| {
            TurError::Other("Batching not enabled. Use builder.enable_batching(true)".to_string())
        })?;

        let tokens = batching
            .tokenizer
            .encode(request.prompt.as_str(), true)
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
            request.prompt.clone(),
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
                },
            };

            // Evict oldest result when at capacity.
            if self.result_order.len() >= self.max_results {
                if let Some(oldest) = self.result_order.pop_front() {
                    results.remove(&oldest);
                    warn!(
                        evicted_request_id = %oldest,
                        max_results = self.max_results,
                        "Results map at capacity; evicted oldest completed result. \
                         Call clear_results() periodically to avoid this.",
                    );
                }
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
