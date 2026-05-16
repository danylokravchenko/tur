use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use parking_lot::RwLock;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, trace};
use uuid::Uuid;

use crate::{
    ProgressReporter, Result, TurError,
    backend::{
        factory::{ModelConstructor, ModelFactory},
        prefix_cache::{PrefixCache, PrefixCacheEntry, SharedPrefixCache},
        tokenizer::{LogitsSampler, TokenOutputStream},
    },
    models::{ModelImpl, kv_cache::PagedKvCache},
};

/// Core inference engine without output handling
/// This is the reusable component for benchmarks and tests
pub struct InferenceEngine<T: ModelImpl> {
    model: T,
    device: Device,
    sampler: Box<dyn LogitsSampler>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    prefix_cache: Option<SharedPrefixCache>,
}

/// Builder for InferenceEngine
pub struct InferenceEngineBuilder<'a, T: ModelConstructor> {
    factory: &'a ModelFactory<T>,
    device: Device,
    sampler: Option<Box<dyn LogitsSampler>>,
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    prefix_cache: Option<SharedPrefixCache>,
    progress: Option<ProgressReporter>,
}

impl<'a, T: ModelConstructor> InferenceEngineBuilder<'a, T> {
    pub fn new(factory: &'a ModelFactory<T>, device: Device) -> Self {
        Self {
            factory,
            device,
            sampler: None,
            seed: 299_792_458, // Default seed
            temp: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            prefix_cache: None,
            progress: None,
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

    /// Enable prefix caching with specified configuration
    pub fn with_prefix_cache(mut self, max_entries: usize, max_token_length: usize) -> Self {
        self.prefix_cache = Some(Arc::new(RwLock::new(PrefixCache::new(
            max_entries,
            max_token_length,
        ))));
        self
    }

    /// Use a shared prefix cache instance
    pub fn with_shared_prefix_cache(mut self, cache: SharedPrefixCache) -> Self {
        self.prefix_cache = Some(cache);
        self
    }

    // Use a specific sampler
    pub fn with_sampler(mut self, sampler: Box<dyn LogitsSampler>) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Set progress reporter for model loading
    pub fn with_progress(mut self, progress: ProgressReporter) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn build(self) -> Result<(InferenceEngine<T>, Tokenizer)> {
        // Log engine configuration
        debug!(
            seed = self.seed,
            temp = ?self.temp,
            top_p = ?self.top_p,
            repeat_penalty = self.repeat_penalty,
            repeat_last_n = self.repeat_last_n,
            prefix_cache = ?self.prefix_cache,
            has_custom_sampler = self.sampler.is_some(),
            "Building inference engine with configuration"
        );

        // Create model using factory
        let (model, tokenizer) = self.factory.create_model(self.progress.as_ref())?;

        let sampler: Box<dyn LogitsSampler> = if let Some(sampler) = self.sampler {
            sampler
        } else {
            Box::new(LogitsProcessor::new(self.seed, self.temp, self.top_p))
        };
        debug!(
            model_name = model.name(),
            "Created inference engine for model",
        );
        let engine = InferenceEngine {
            model,
            device: self.device,
            sampler,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            prefix_cache: self.prefix_cache,
        };
        Ok((engine, tokenizer))
    }
}

impl<T: ModelConstructor> InferenceEngine<T> {
    /// Create a new builder for InferenceEngine from factory
    pub fn builder(factory: &'_ ModelFactory<T>, device: Device) -> InferenceEngineBuilder<'_, T> {
        InferenceEngineBuilder::new(factory, device)
    }

    /// Get EOS tokens from tokenizer
    pub fn get_eos_tokens(tokenizer: &TokenOutputStream) -> Result<(u32, u32)> {
        let eos_token = tokenizer
            .get_token("<|endoftext|>")
            .ok_or_else(|| TurError::Tokenizer("cannot find <|endoftext|> token".to_string()))?;
        let eos_token2 = tokenizer
            .get_token("<|im_end|>")
            .ok_or_else(|| TurError::Tokenizer("cannot find <|im_end|> token".to_string()))?;
        Ok((eos_token, eos_token2))
    }

    /// Apply repeat penalty and optional guidance constraints to logits
    fn process_logits(&mut self, logits: Tensor, tokens: &[u32]) -> Result<Tensor> {
        // Apply repeat penalty
        let logits = if self.repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        // Apply guidance constraints if any
        if let Some(constrained) = self.sampler.apply_constraints(&logits, tokens)? {
            Ok(constrained)
        } else {
            Ok(logits)
        }
    }

    /// Try to restore KV cache state from prefix cache
    /// Returns (cache_hit, cached_token_count)
    fn try_restore_from_cache(&mut self, tokens: &[u32]) -> Result<(bool, usize)> {
        let cache = match &self.prefix_cache {
            Some(c) => c,
            None => return Ok((false, 0)),
        };

        let mut cache = cache.write();

        // Find longest matching prefix
        let (key, match_len) = cache.find_longest_prefix(tokens);

        if match_len == 0 {
            cache.record_miss();
            return Ok((false, 0));
        }

        // Get the cached entry
        let entry = match cache.get(key) {
            Some(e) => e,
            None => {
                cache.record_miss();
                return Ok((false, 0));
            }
        };

        // Validate compatibility
        if !entry.is_compatible(
            &self.device,
            self.model
                .get_kv_cache_state()?
                .first()
                .map(|(k, _)| k.dtype())
                .unwrap_or(DType::F32),
        ) {
            cache.record_miss();
            return Ok((false, 0));
        }

        // Restore KV cache state
        self.model.set_kv_cache_state(entry.kv_states.clone())?;

        // Update stats
        cache.record_hit(match_len);

        Ok((true, match_len))
    }

    /// Store current KV cache state to prefix cache
    fn store_to_cache(&mut self, tokens: &[u32]) -> Result<()> {
        let cache = match &self.prefix_cache {
            Some(c) => c,
            None => return Ok(()),
        };

        // Extract current KV cache state
        let kv_states = self.model.get_kv_cache_state()?;

        // Don't cache if no state
        if kv_states.is_empty() {
            return Ok(());
        }

        // Get dtype from first tensor
        let dtype = kv_states
            .first()
            .map(|(k, _)| k.dtype())
            .unwrap_or(DType::F32);
        // Create cache entry
        let entry = PrefixCacheEntry::new(tokens.to_vec(), kv_states, self.device.clone(), dtype);

        // Insert into cache
        let mut cache = cache.write();
        cache.insert(entry);
        trace!(
            tokens = tokens.len(),
            "Stored prompt tokens in prefix cache",
        );

        Ok(())
    }

    /// Perform prefill: encode full prompt and get first token
    /// Returns (first_token, prefill_duration, cache_hit, cached_tokens)
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(u32, std::time::Duration, bool, usize)> {
        let start = std::time::Instant::now();

        // Try to restore from cache
        let (cache_hit, cached_len) = self.try_restore_from_cache(tokens)?;

        // If entire prompt is cached, we still need at least one token to process
        // to get the next token prediction
        let remaining_tokens = &tokens[cached_len..];

        // Ensure we have at least one token to process
        let (process_tokens, process_offset) = if remaining_tokens.is_empty() {
            // All tokens cached - process last token to get next prediction
            (
                &tokens[tokens.len().saturating_sub(1)..],
                tokens.len().saturating_sub(1),
            )
        } else {
            (remaining_tokens, cached_len)
        };

        let input = Tensor::new(process_tokens, &self.device)?.unsqueeze(0)?;
        trace!(
            processed_tokens = process_tokens.len(),
            process_offset, "Running prefill forward pass on tokens",
        );
        let logits = self.model.forward(&input, process_offset)?;

        // Store to cache if enabled and not already cached
        if !cache_hit && self.prefix_cache.is_some() {
            self.store_to_cache(tokens)?;
        }

        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = self.process_logits(logits, tokens)?;
        let next_token = self.sampler.sample(&logits)?;
        trace!(
            next_token,
            cache_hit,
            cached_tokens = cached_len,
            "Prefill sampled first token"
        );

        Ok((next_token, start.elapsed(), cache_hit, cached_len))
    }

    /// Perform single decode step with KV cache reuse
    /// Returns next token
    pub fn decode_step(&mut self, tokens: &[u32], start_pos: usize) -> Result<u32> {
        let context_size = 1;
        let pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[pos..];

        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        trace!(
            tokens = ctxt.len(),
            start_pos, "Decoding next token from context",
        );
        let logits = self.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = self.process_logits(logits, tokens)?;

        let next_token = self.sampler.sample(&logits)?;
        trace!(next_token, "Decode step sampled token");
        Ok(next_token)
    }

    /// Run separated inference: prefill + decode with detailed stats
    pub fn run_separated(
        &mut self,
        tokens: &[u32],
        sample_len: usize,
        eos_tokens: (u32, u32),
    ) -> Result<DetailedGenerationStats> {
        let prompt_tokens = tokens.len();
        let mut token_vec = tokens.to_vec();

        // Prefill phase
        let (first_token, prefill_duration, cache_hit, cached_tokens) = self.prefill(tokens)?;
        token_vec.push(first_token);
        let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;

        // Decode phase
        let mut generated_tokens = 1;
        let decode_start = std::time::Instant::now();

        for _ in 1..sample_len {
            let start_pos = token_vec.len() - 1;
            let next_token = self.decode_step(&token_vec, start_pos)?;
            token_vec.push(next_token);
            generated_tokens += 1;

            if next_token == eos_tokens.0 || next_token == eos_tokens.1 {
                break;
            }
        }

        let decode_elapsed = decode_start.elapsed();
        let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
        let decode_tokens_per_sec = generated_tokens as f64 / decode_elapsed.as_secs_f64();
        let latency_per_token_ms = decode_ms / generated_tokens as f64;

        Ok(DetailedGenerationStats {
            prefill_ms,
            prompt_tokens,
            decode_tokens_per_sec,
            latency_per_token_ms,
            generated_tokens,
            decode_ms,
            cache_hit,
            cached_tokens,
        })
    }

    /// Access the underlying model
    pub fn model(&self) -> &T {
        &self.model
    }

    /// Access the underlying model mutably
    pub fn model_mut(&mut self) -> &mut T {
        &mut self.model
    }

    /// Perform batched prefill: encode multiple prompts and get first tokens
    /// Returns vector of (first_token, request_id) pairs
    pub fn prefill_batch(
        &mut self,
        batch_tokens: &[(Uuid, Vec<u32>)],
        paged_caches: Option<&mut [Vec<PagedKvCache>]>,
    ) -> Result<Vec<(Uuid, u32)>> {
        if batch_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Clear KV cache only if not using paged cache
        if paged_caches.is_none() {
            self.model.clear_kv_cache();
        }

        // Find max length in batch
        let max_len = batch_tokens
            .iter()
            .map(|(_, tokens)| tokens.len())
            .max()
            .unwrap_or(0);

        // Prepare batched input
        let batch_size = batch_tokens.len();
        let mut input_data = Vec::with_capacity(batch_size * max_len);
        let positions = vec![0; batch_size]; // All start at position 0 for prefill

        // Pad sequences to max length
        for (_, tokens) in batch_tokens.iter() {
            input_data.extend_from_slice(tokens);
            // Pad with zeros if needed (will be masked in attention)
            input_data.extend(std::iter::repeat_n(0, max_len.saturating_sub(tokens.len())));
        }

        let input = Tensor::from_vec(input_data, (batch_size, max_len), &self.device)?;
        trace!(
            batch_size,
            max_len,
            has_paged_caches = paged_caches.is_some(),
            "Running batched prefill forward pass"
        );

        let logits = self.model.forward_batch(&input, &positions, paged_caches)?;

        // Sample next token for each request
        let mut results = Vec::with_capacity(batch_size);
        for (idx, (id, tokens)) in batch_tokens.iter().enumerate() {
            // Extract logits for this request: [1, 1, vocab_size]
            let request_logits = logits
                .narrow(0, idx, 1)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let processed_logits = self.process_logits(request_logits, tokens)?;
            let next_token = self.sampler.sample(&processed_logits)?;
            results.push((*id, next_token));
            trace!(request_id = ?id, next_token, "Batched prefill sampled token");
        }

        Ok(results)
    }

    /// Perform batched decode step: process one token per request
    /// Returns vector of (next_token, request_id) pairs
    pub fn decode_batch(
        &mut self,
        batch_data: &[(Uuid, Vec<u32>, usize)], // (id, all_tokens, position)
        paged_caches: Option<&mut [Vec<PagedKvCache>]>,
    ) -> Result<Vec<(Uuid, u32)>> {
        if batch_data.is_empty() {
            return Ok(Vec::new());
        }

        // Clear KV cache only if not using paged cache
        if paged_caches.is_none() {
            self.model.clear_kv_cache();
        }

        let batch_size = batch_data.len();

        // Prepare batched input - last token from each request
        let mut input_data = Vec::with_capacity(batch_size);
        let mut positions = Vec::with_capacity(batch_size);

        for (_, tokens, pos) in batch_data.iter() {
            input_data.push(*tokens.last().unwrap_or(&0));
            positions.push(*pos);
        }

        let input = Tensor::from_vec(input_data, (batch_size, 1), &self.device)?;
        trace!(
            batch_size,
            has_paged_caches = paged_caches.is_some(),
            "Running batched decode forward pass"
        );

        let logits = self.model.forward_batch(&input, &positions, paged_caches)?;

        // Sample next token for each request
        let mut results = Vec::with_capacity(batch_size);
        for (idx, (id, tokens, _)) in batch_data.iter().enumerate() {
            // Extract logits for this request: [1, 1, vocab_size]
            let request_logits = logits
                .narrow(0, idx, 1)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let processed_logits = self.process_logits(request_logits, tokens)?;
            let next_token = self.sampler.sample(&processed_logits)?;
            results.push((*id, next_token));
            trace!(request_id = ?id, next_token, "Batched decode sampled token");
        }

        Ok(results)
    }

    /// Run continuous batching loop with scheduler integration
    /// This is the main entry point for continuous batching execution
    pub fn run_continuous_batching(
        &mut self,
        scheduler: &mut crate::backend::scheduler::ContinuousBatchScheduler,
        eos_tokens: (u32, u32),
    ) -> Result<()> {
        // 1. Admit new requests from queue
        let admitted = scheduler.admit_requests()?;
        trace!(admitted_count = admitted.len(), "Admitted new requests");

        // 2. Form and execute prefill batch
        if let Some(prefill_batch) = scheduler.form_prefill_batch() {
            trace!(
                batch_size = prefill_batch.request_ids.len(),
                total_tokens = prefill_batch.total_tokens,
                "Executing prefill batch"
            );

            // Gather tokens for each request
            let mut batch_tokens = Vec::new();
            for id in &prefill_batch.request_ids {
                if let Some(request) = scheduler.get_request(id) {
                    batch_tokens.push((*id, request.prompt_tokens.clone()));
                }
            }

            // Get paged caches from scheduler
            let mut paged_caches = scheduler.get_paged_caches_mut(&prefill_batch.request_ids);

            // Execute batched prefill with paged caches
            let results = if !paged_caches.is_empty() {
                self.prefill_batch(&batch_tokens, Some(&mut paged_caches))?
            } else {
                self.prefill_batch(&batch_tokens, None)?
            };

            // Put paged caches back to scheduler
            scheduler.put_paged_caches(&prefill_batch.request_ids, paged_caches);

            // Update requests with first generated tokens and transition to decode
            for (id, next_token) in results {
                if let Some(request) = scheduler.get_request_mut(&id) {
                    request.generated_tokens.push(next_token);
                    request.position = request.seq_len();
                }
                scheduler.transition_to_decode(&id)?;
            }
        }

        // 3. Form and execute decode batch
        if let Some(decode_batch) = scheduler.form_decode_batch() {
            trace!(
                batch_size = decode_batch.request_ids.len(),
                total_tokens = decode_batch.total_tokens,
                "Executing decode batch"
            );

            // Gather data for each request
            let mut batch_data = Vec::new();
            let mut to_complete = Vec::new();

            for id in &decode_batch.request_ids {
                if let Some(request) = scheduler.get_request(id) {
                    let all_tokens = request.all_tokens();
                    batch_data.push((*id, all_tokens, request.position));
                }
            }

            // Get paged caches from scheduler
            let mut paged_caches = scheduler.get_paged_caches_mut(&decode_batch.request_ids);

            // Execute batched decode with paged caches
            let results = if !paged_caches.is_empty() {
                self.decode_batch(&batch_data, Some(&mut paged_caches))?
            } else {
                self.decode_batch(&batch_data, None)?
            };

            // Put paged caches back to scheduler
            scheduler.put_paged_caches(&decode_batch.request_ids, paged_caches);

            // Update requests with new tokens
            for (id, next_token) in results {
                if let Some(request) = scheduler.get_request_mut(&id) {
                    request.generated_tokens.push(next_token);
                    request.position = request.seq_len();

                    // Check if request should be completed
                    if next_token == eos_tokens.0
                        || next_token == eos_tokens.1
                        || request.should_stop()
                    {
                        to_complete.push(id);
                    }
                }
            }

            // Complete finished requests
            for id in to_complete {
                scheduler.complete_request(&id)?;
                trace!(request_id = ?id, "Completed request");
            }
        }

        Ok(())
    }
}

/// Detailed statistics separating prefill and decode phases
#[derive(Debug, Clone)]
pub struct DetailedGenerationStats {
    /// Time to encode prompt and perform first forward pass (prefill)
    pub prefill_ms: f64,
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Tokens generated per second during steady-state decode
    pub decode_tokens_per_sec: f64,
    /// Average latency per token during decode (ms)
    pub latency_per_token_ms: f64,
    /// Total tokens generated (excluding prompt)
    pub generated_tokens: usize,
    /// Total time for decode phase
    pub decode_ms: f64,
    /// Whether prefix cache was hit
    pub cache_hit: bool,
    /// Number of tokens restored from cache
    pub cached_tokens: usize,
}

impl DetailedGenerationStats {
    pub fn report(&self, name: &str) {
        println!("\n=== Generation Stats: {} ===", name);
        if self.cache_hit {
            println!("Prefix Cache: HIT ({} tokens restored)", self.cached_tokens);
        } else {
            println!("Prefix Cache: MISS");
        }
        println!("\nPrefill Phase:");
        println!("  Prompt tokens: {}", self.prompt_tokens);
        println!("  Prefill time: {:.2} ms", self.prefill_ms);
        println!(
            "  Prefill throughput: {:.2} tokens/sec",
            self.prompt_tokens as f64 / (self.prefill_ms / 1000.0)
        );
        println!("\nDecode Phase:");
        println!("  Generated tokens: {}", self.generated_tokens);
        println!("  Decode time: {:.2} ms", self.decode_ms);
        println!(
            "  Decode throughput: {:.2} tokens/sec",
            self.decode_tokens_per_sec
        );
        println!("  Latency per token: {:.2} ms", self.latency_per_token_ms);
        println!("========================================\n");
    }
}
