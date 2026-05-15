use candle_core::{DType, Device};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{info, trace};
use uuid::Uuid;

use crate::{
    ProgressReporter, Result, TurError,
    backend::{
        engine::InferenceEngine,
        scheduler::{ContinuousBatchScheduler, SchedulingPolicy},
        tokenizer::TokenOutputStream,
    },
    models::{ModelImpl, kv_cache::BlockAllocator},
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

/// High-level text generation pipeline with output handling and continuous batching support
pub struct TextGeneration<T: ModelImpl> {
    engine: InferenceEngine<T>,
    tokenizer: TokenOutputStream,
    progress: Option<ProgressReporter>,
    emit_output: bool,
    // Batching support (optional)
    scheduler: Option<ContinuousBatchScheduler>,
    tokenizer_arc: Option<Arc<Tokenizer>>,
    results: Arc<RwLock<HashMap<Uuid, GenerationResult>>>,
}

/// Builder for TextGeneration
pub struct TextGenerationBuilder<T: ModelImpl> {
    model: T,
    tokenizer: Tokenizer,
    device: Device,
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    progress: Option<ProgressReporter>,
    emit_output: bool,
    // Batching configuration
    enable_batching: bool,
    max_batch_size: usize,
    max_prefill_batch: usize,
    max_decode_batch: usize,
    scheduling_policy: SchedulingPolicy,
}

impl<T: ModelImpl> TextGenerationBuilder<T> {
    pub fn new(model: T, tokenizer: Tokenizer, device: Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
            seed: 299_792_458,
            temp: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            progress: None,
            emit_output: true,
            enable_batching: false,
            max_batch_size: 16,
            max_prefill_batch: 8,
            max_decode_batch: 16,
            scheduling_policy: SchedulingPolicy::FCFS,
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

    pub fn build(self) -> TextGeneration<T> {
        // Initialize batching components if enabled
        let (scheduler, tokenizer_arc, _block_allocator) = if self.enable_batching {
            let tokenizer_arc = Arc::new(self.tokenizer.clone());

            // Create BlockAllocator for paged KV cache
            // TODO: Make these configurable - should match model architecture
            let total_blocks = 1024; // Number of blocks
            let block_size = 64; // Tokens per block
            let batch_size = 1; // Batch size for KV tensors
            let num_heads = 32; // Number of attention heads
            let head_dim = 128; // Head dimension
            let dtype = DType::BF16; // Data type

            let block_allocator = Arc::new(RwLock::new(BlockAllocator::new(
                total_blocks,
                block_size,
                batch_size,
                num_heads,
                head_dim,
                self.device.clone(),
                dtype,
            )));

            // Create a simple memory pool for batching
            // TODO: Make these configurable
            let memory_pool = Arc::new(RwLock::new(crate::backend::memory_pool::MemoryPool::new(
                8 * 1024 * 1024 * 1024, // 8GB
                64,
                32,
                32,
                128,
                2,
                self.device.clone(),
            )));

            let batch_config = crate::backend::scheduler::BatchConfig {
                max_prefill_batch: self.max_prefill_batch,
                max_decode_batch: self.max_decode_batch,
                max_prefill_tokens: 2048,
                max_decode_tokens: 4096,
            };

            let scheduler = ContinuousBatchScheduler::new(
                self.scheduling_policy,
                batch_config,
                memory_pool,
                tokenizer_arc.clone(),
            );

            (
                Some(scheduler),
                Some(tokenizer_arc),
                Some(block_allocator.clone()),
            )
        } else {
            (None, None, None)
        };

        // Build engine with model factory and optional block allocator
        let mut engine_builder = InferenceEngine::builder(self.model, self.device.clone())
            .seed(self.seed)
            .repeat_penalty(self.repeat_penalty)
            .repeat_last_n(self.repeat_last_n);

        if let Some(temp) = self.temp {
            engine_builder = engine_builder.temperature(temp);
        }
        if let Some(top_p) = self.top_p {
            engine_builder = engine_builder.top_p(top_p);
        }

        let engine = engine_builder.build();

        TextGeneration {
            engine,
            tokenizer: TokenOutputStream::new(self.tokenizer),
            progress: self.progress,
            emit_output: self.emit_output,
            scheduler,
            tokenizer_arc,
            results: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl<T: ModelImpl> TextGeneration<T> {
    /// Create a new builder for TextGeneration
    pub fn builder(model: T, tokenizer: Tokenizer, device: Device) -> TextGenerationBuilder<T> {
        TextGenerationBuilder::new(model, tokenizer, device)
    }

    /// Create from an existing inference engine (without batching)
    pub fn from_engine(
        engine: InferenceEngine<T>,
        tokenizer: Tokenizer,
        progress: Option<ProgressReporter>,
    ) -> Self {
        Self {
            engine,
            tokenizer: TokenOutputStream::new(tokenizer),
            progress,
            emit_output: true,
            scheduler: None,
            tokenizer_arc: None,
            results: Arc::new(RwLock::new(HashMap::new())),
        }
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
            "starting text generation request",
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
            "prompt encoded into tokens",
        );

        // Initialize generation progress bar
        if let Some(ref progress) = self.progress {
            progress.init_generation(sample_len);
        }

        let mut generated_tokens = 0usize;
        let (eos_token, eos_token2) = InferenceEngine::<T>::get_eos_tokens(&self.tokenizer)?;

        let start_gen = std::time::Instant::now();

        // Prefill phase
        let (first_token, _, cache_hit, cached_tokens) = self.engine.prefill(&tokens)?;
        trace!(
            request_id = %request.id,
            first_token,
            cache_hit,
            cached_tokens,
            "engine prefetched prompt and produced first token",
        );
        tokens.push(first_token);
        generated_tokens += 1;

        if let Some(ref progress) = self.progress {
            progress.inc_generation();
        }

        if first_token != eos_token
            && first_token != eos_token2
            && let Some(t) = self.tokenizer.next_token(first_token)?
        {
            trace!(
                request_id = %request.id,
                text_chars = t.chars().count(),
                "decoded first token into text chunk",
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
            if tokens[tokens.len() - 1] == eos_token || tokens[tokens.len() - 1] == eos_token2 {
                break;
            }

            let start_pos = tokens.len() - 1;
            let next_token = self.engine.decode_step(&tokens, start_pos)?;
            tokens.push(next_token);
            generated_tokens += 1;

            // Update progress bar
            if let Some(ref progress) = self.progress {
                progress.inc_generation();
            }

            if next_token == eos_token || next_token == eos_token2 {
                trace!(
                    request_id = %request.id,
                    next_token,
                    generated_tokens,
                    "decode loop reached EOS token",
                );
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                trace!(
                    request_id = %request.id,
                    next_token,
                    text_chars = t.chars().count(),
                    "decoded token into text chunk",
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
                "flushed remaining decoded text",
            );
            if let Some(ref progress) = self.progress {
                progress.print(&rest);
            } else if self.emit_output {
                print!("{rest}");
            }
        }

        // Flush any remaining buffered text
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
            "generation request completed",
        );

        // Finish generation progress bar
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

    pub fn with_output_enabled(mut self, emit_output: bool) -> Self {
        self.emit_output = emit_output;
        self
    }

    /// Submit a new generation request (batching mode only)
    pub fn submit_request(&mut self, request: &GenerationRequest) -> Result<RequestHandle> {
        let scheduler = self.scheduler.as_mut().ok_or_else(|| {
            TurError::Other("Batching not enabled. Use builder.enable_batching(true)".to_string())
        })?;

        let tokenizer = self
            .tokenizer_arc
            .as_ref()
            .ok_or_else(|| TurError::Other("Tokenizer not available for batching".to_string()))?;

        // Tokenize the prompt
        let tokens = tokenizer
            .encode(request.prompt.as_str(), true)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();

        trace!(
            request_id = %request.id,
            prompt_tokens = tokens.len(),
            sample_len = request.sample_len,
            "submitting request to scheduler"
        );

        // Create request state
        let request_state = crate::backend::batch_manager::RequestState::new(
            request.id,
            request.prompt.clone(),
            tokens,
            request.sample_len,
            0, // default priority
        );

        // Enqueue in scheduler
        scheduler.enqueue_request(request_state);

        Ok(RequestHandle { id: request.id })
    }

    /// Run one iteration of continuous batching (batching mode only)
    /// Returns the number of active requests
    pub fn step(&mut self) -> Result<usize> {
        let scheduler = self
            .scheduler
            .as_mut()
            .ok_or_else(|| TurError::Other("Batching not enabled".to_string()))?;

        // Run one scheduler iteration
        let completed = scheduler.schedule_iteration(&mut self.engine)?;

        let tokenizer = self
            .tokenizer_arc
            .as_ref()
            .ok_or_else(|| TurError::Other("Tokenizer not available".to_string()))?;

        // Process completed requests
        for (request_id, tokens) in completed {
            let generated_text = tokenizer
                .decode(&tokens, true)
                .map_err(|e| TurError::Tokenizer(e.to_string()))?;

            // Get request info from scheduler
            if let Some(state) = scheduler.get_request_state(&request_id) {
                let result = GenerationResult {
                    request_id,
                    prompt: state.prompt.clone(),
                    generated_text,
                    generated_tokens: tokens.clone(),
                    stats: GenerationStats {
                        generated_tokens: tokens.len(),
                        elapsed: state.arrival_time.elapsed(),
                    },
                };

                self.results.write().insert(request_id, result);

                trace!(
                    request_id = %request_id,
                    generated_tokens = tokens.len(),
                    "request completed"
                );
            }
        }

        Ok(scheduler.active_request_count())
    }

    /// Run continuous batching until all requests are completed (batching mode only)
    pub fn run_until_complete(&mut self) -> Result<()> {
        if self.scheduler.is_none() {
            return Err(TurError::Other("Batching not enabled".to_string()));
        }

        loop {
            let has_work = self
                .scheduler
                .as_ref()
                .map(|s| s.has_active_requests() || s.has_queued_requests())
                .unwrap_or(false);

            if !has_work {
                break;
            }

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
        // Check if already completed
        if let Some(result) = self.try_get_result(handle) {
            return Ok(result);
        }

        if self.scheduler.is_none() {
            return Err(TurError::Other("Batching not enabled".to_string()));
        }

        // Keep running until this request completes
        loop {
            if let Some(result) = self.try_get_result(handle) {
                return Ok(result);
            }

            let has_work = self
                .scheduler
                .as_ref()
                .map(|s| s.has_active_requests() || s.has_queued_requests())
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
    }

    /// Get number of active requests (batching mode only)
    pub fn active_request_count(&self) -> usize {
        self.scheduler
            .as_ref()
            .map(|s| s.active_request_count())
            .unwrap_or(0)
    }

    /// Get number of queued requests (batching mode only)
    pub fn queued_request_count(&self) -> usize {
        self.scheduler
            .as_ref()
            .map(|s| s.queued_request_count())
            .unwrap_or(0)
    }

    /// Check if batching is enabled
    pub fn is_batching_enabled(&self) -> bool {
        self.scheduler.is_some()
    }
}
