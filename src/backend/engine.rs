use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, trace};

use crate::{
    Result, TurError,
    backend::prefix_cache::{PrefixCache, PrefixCacheEntry, SharedPrefixCache},
    backend::tokenizer::{LogitsSampler, TokenOutputStream},
    models::ModelImpl,
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
pub struct InferenceEngineBuilder<T: ModelImpl> {
    model: T,
    device: Device,
    sampler: Option<Box<dyn LogitsSampler>>,
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    prefix_cache: Option<SharedPrefixCache>,
}

impl<T: ModelImpl> InferenceEngineBuilder<T> {
    pub fn new(model: T, device: Device) -> Self {
        Self {
            model,
            device,
            sampler: None,
            seed: 299_792_458, // Default seed
            temp: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            prefix_cache: None,
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

    pub fn build(self) -> InferenceEngine<T> {
        let sampler: Box<dyn LogitsSampler> = if let Some(sampler) = self.sampler {
            sampler
        } else {
            Box::new(LogitsProcessor::new(self.seed, self.temp, self.top_p))
        };
        debug!(
            model_name = self.model.name(),
            "Created a new inference engine for model",
        );
        InferenceEngine {
            model: self.model,
            device: self.device,
            sampler,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            prefix_cache: self.prefix_cache,
        }
    }
}

impl<T: ModelImpl> InferenceEngine<T> {
    /// Create a new builder for InferenceEngine
    pub fn builder(model: T, device: Device) -> InferenceEngineBuilder<T> {
        InferenceEngineBuilder::new(model, device)
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
            "stored prompt tokens in prefix cache",
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
            process_offset, "running prefill forward pass on tokens",
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
            "prefill sampled first token"
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
            start_pos, "decoding next token from context",
        );
        let logits = self.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = self.process_logits(logits, tokens)?;

        let next_token = self.sampler.sample(&logits)?;
        trace!(next_token, "decode step sampled token");
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
