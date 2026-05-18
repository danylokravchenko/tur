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
        guidance::{GuidanceControl, ParserFactory, TopLevelGrammar},
        prefix_cache::{PrefixCache, PrefixCacheEntry, SharedPrefixCache},
        tokenizer::{LogitsSampler, TokenOutputStream},
    },
    models::{
        ModelImpl, ModelInput,
        kv_cache::{KvCacheImpl, PagedKvCache},
    },
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
    /// Optional per-request grammar constraint for guided generation.
    guidance: Option<GuidanceControl>,
    /// EOS token IDs detected at build time; used by `sample_guided` to force
    /// termination when the grammar signals stop.
    eos_tokens: Option<(u32, u32)>,
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
    guidance_factory: Option<Arc<ParserFactory>>,
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
            guidance_factory: None,
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

    /// Enable guided generation with the given parser factory.
    /// The factory is typically built once via `guidance::build_llg_factory`.
    pub fn with_guidance_factory(mut self, factory: Arc<ParserFactory>) -> Self {
        self.guidance_factory = Some(factory);
        self
    }

    pub fn build(
        self,
    ) -> Result<(InferenceEngine<T>, Tokenizer, Option<crate::backend::chat_template::ChatTemplate>)>
    {
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

        // Create model using factory — weights are prepared once; chat template
        // piggybacks on the same ModelPaths so there is no second HF download.
        let (model, tokenizer, chat_template) =
            self.factory.create_model(self.progress.as_ref())?;

        let sampler: Box<dyn LogitsSampler> = if let Some(sampler) = self.sampler {
            sampler
        } else {
            Box::new(LogitsProcessor::new(self.seed, self.temp, self.top_p))
        };
        debug!(
            model_name = model.name(),
            "Created inference engine for model",
        );
        let guidance = self.guidance_factory.map(GuidanceControl::new);

        let eos_tokens = tokenizer
            .token_to_id("<|endoftext|>")
            .zip(tokenizer.token_to_id("<|im_end|>"));

        let engine = InferenceEngine {
            model,
            device: self.device,
            sampler,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            prefix_cache: self.prefix_cache,
            guidance,
            eos_tokens,
        };
        Ok((engine, tokenizer, chat_template))
    }
}

impl<T: ModelConstructor> InferenceEngine<T> {
    /// Create a new builder for InferenceEngine from factory
    pub fn builder(factory: &'_ ModelFactory<T>, device: Device) -> InferenceEngineBuilder<'_, T> {
        InferenceEngineBuilder::new(factory, device)
    }

    /// Activate guided generation for the next request. Requires a guidance factory
    /// configured on the builder (`with_guidance_factory`).
    pub fn activate_grammar(&mut self, grammar: TopLevelGrammar) -> Result<()> {
        self.guidance
            .as_mut()
            .ok_or_else(|| {
                TurError::Guidance(
                    "Guidance not configured. Use builder.with_guidance_factory()".to_string(),
                )
            })?
            .activate(grammar)
    }

    /// Deactivate guided generation. Call after each request completes or errors.
    pub fn deactivate_grammar(&mut self) {
        if let Some(g) = self.guidance.as_mut() {
            g.deactivate();
        }
    }

    /// Apply repeat penalty + optional guidance mask, then sample. Commits guidance state.
    fn sample_guided(&mut self, logits: Tensor, tokens: &[u32]) -> Result<u32> {
        // Apply repeat penalty and sampler constraints.
        let logits = self.process_logits(logits, tokens)?;

        // Apply grammar mask when guidance is active.
        let logits = if let Some(ref mut guidance) = self.guidance {
            if guidance.is_active() {
                let mut logits_vec = logits.to_vec1::<f32>()?;
                let stop = guidance.apply_mask(&mut logits_vec)?;
                if stop {
                    // Grammar reached a terminal state — force EOS so the pipeline
                    // decode loop terminates cleanly without additional sampling.
                    if let Some((eos, _)) = self.eos_tokens {
                        return Ok(eos);
                    }
                    // No known EOS token; fall through to unconstrained sampling
                    // and rely on the pipeline's EOS check.
                    return self.sampler.sample(&logits);
                }
                Tensor::from_vec(logits_vec, logits.shape().clone(), logits.device())?
            } else {
                logits
            }
        } else {
            logits
        };

        let token = self.sampler.sample(&logits)?;

        // Advance grammar state.
        if let Some(ref mut guidance) = self.guidance {
            guidance.commit(token)?;
        }

        Ok(token)
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

        Ok(self
            .sampler
            .apply_constraints(&logits, tokens)?
            .unwrap_or(logits))
    }

    /// Try to restore KV cache state from prefix cache
    /// Returns (cache_hit, cached_token_count)
    fn try_restore_from_cache(&mut self, tokens: &[u32]) -> Result<(bool, usize)> {
        let Some(cache) = &self.prefix_cache else {
            return Ok((false, 0));
        };
        let mut cache = cache.write();

        let Some((key, match_len)) = cache.find_longest_prefix(tokens) else {
            cache.record_miss();
            return Ok((false, 0));
        };

        let entry = match cache.get(key) {
            Some(e) => e,
            None => {
                cache.record_miss();
                return Ok((false, 0));
            }
        };

        if !entry.is_compatible(&self.device, self.model.dtype()) {
            cache.record_miss();
            return Ok((false, 0));
        }

        // Full cache hit: the stored K/V covers all N prompt tokens.  If we
        // restore all N entries and then re-process the last token in prefill,
        // the KV cache appends a duplicate entry at position N-1.  Trim to N-1
        // so prefill re-processes the final token against the correct history.
        // We still report match_len as the cached count — all N tokens were
        // available in cache; the trim is an implementation detail.
        if match_len == tokens.len() {
            if match_len == 1 {
                // Cannot narrow to zero-length tensors safely; treat as miss.
                cache.record_miss();
                return Ok((false, 0));
            }
            let trimmed = entry
                .kv_states
                .iter()
                .map(|(k, v)| {
                    let s = k.dim(2)?;
                    candle_core::Result::Ok((k.narrow(2, 0, s - 1)?, v.narrow(2, 0, s - 1)?))
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            self.model.set_kv_cache_state(trimmed)?;
            cache.record_hit(match_len);
            return Ok((true, match_len));
        }

        self.model.set_kv_cache_state(entry.kv_states.clone())?;
        cache.record_hit(match_len);
        Ok((true, match_len))
    }

    /// Store current KV cache state to prefix cache
    fn store_to_cache(&mut self, tokens: &[u32]) -> Result<()> {
        let Some(cache) = &self.prefix_cache else {
            return Ok(());
        };

        // Extract current KV cache state
        let kv_states = self.model.get_kv_cache_state()?;

        if kv_states.is_empty() {
            return Ok(());
        }

        let entry = PrefixCacheEntry::new(
            tokens.to_vec(),
            kv_states,
            self.device.clone(),
            self.model.dtype(),
        );

        // Insert into cache
        let mut cache = cache.write();
        cache.insert(entry);
        trace!(
            tokens = tokens.len(),
            "Stored prompt tokens in prefix cache",
        );

        Ok(())
    }

    /// Try to restore a prefix cache hit into per-layer paged KV caches.
    ///
    /// Mirrors `try_restore_from_cache` but writes the cached K/V tensors directly
    /// into the request's `PagedKvCache` layers via `set_state`, which resets each
    /// layer and re-populates it via `PagedKvCache::append`.  The same full-hit
    /// trimming logic applies: if the cached entry covers all N prompt tokens we
    /// trim it to N-1 so the final token is re-processed against the correct history.
    ///
    /// Returns (cache_hit, cached_token_count).
    fn try_restore_paged(
        &mut self,
        tokens: &[u32],
        req_caches: &mut [PagedKvCache],
    ) -> Result<(bool, usize)> {
        let Some(cache) = &self.prefix_cache else {
            return Ok((false, 0));
        };
        let mut cache = cache.write();

        let Some((key, match_len)) = cache.find_longest_prefix(tokens) else {
            cache.record_miss();
            return Ok((false, 0));
        };

        let entry = match cache.get(key) {
            Some(e) => e,
            None => {
                cache.record_miss();
                return Ok((false, 0));
            }
        };

        if !entry.is_compatible(&self.device, self.model.dtype()) {
            cache.record_miss();
            return Ok((false, 0));
        }

        if entry.kv_states.len() != req_caches.len() {
            // Layer count mismatch — cached entry is stale; treat as miss.
            cache.record_miss();
            return Ok((false, 0));
        }

        // Full hit: trim last K/V entry to avoid duplicating position N-1.
        let (states, effective_len) = if match_len == tokens.len() {
            if match_len == 1 {
                cache.record_miss();
                return Ok((false, 0));
            }
            let trimmed = entry
                .kv_states
                .iter()
                .map(|(k, v)| {
                    let s = k.dim(2)?;
                    candle_core::Result::Ok((k.narrow(2, 0, s - 1)?, v.narrow(2, 0, s - 1)?))
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            (trimmed, match_len - 1)
        } else {
            (entry.kv_states.clone(), match_len)
        };

        // Inject cached K/V into each layer's paged cache.
        for (layer_cache, (k, v)) in req_caches.iter_mut().zip(states.iter()) {
            layer_cache.set_state(k.clone(), v.clone())?;
        }

        cache.record_hit(effective_len);
        Ok((true, effective_len))
    }

    /// Extract per-layer K/V state from paged caches and store in the prefix cache.
    fn store_paged_to_cache(&mut self, tokens: &[u32], req_caches: &[PagedKvCache]) -> Result<()> {
        let Some(cache) = &self.prefix_cache else {
            return Ok(());
        };

        let kv_states: Vec<(Tensor, Tensor)> =
            req_caches.iter().filter_map(|c| c.get_state()).collect();

        // Only cache if every layer produced a state (partial state would be wrong).
        if kv_states.len() != req_caches.len() || kv_states.is_empty() {
            return Ok(());
        }

        let entry = PrefixCacheEntry::new(
            tokens.to_vec(),
            kv_states,
            self.device.clone(),
            self.model.dtype(),
        );
        cache.write().insert(entry);
        trace!(
            tokens = tokens.len(),
            "Stored batched prompt tokens in prefix cache"
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
        let next_token = self.sample_guided(logits, tokens)?;
        trace!(
            next_token,
            cache_hit,
            cached_tokens = cached_len,
            "Prefill sampled first token"
        );

        Ok((next_token, start.elapsed(), cache_hit, cached_len))
    }

    /// Prefill with audio embeddings merged alongside token IDs.
    ///
    /// Builds a [`ModelInput::Mixed`] and calls [`ModelImpl::forward_modal`].
    /// Models that only support text will return an error from `forward_modal`.
    /// Prefix caching is skipped on this path (audio inputs are not yet cacheable).
    ///
    /// Returns `(first_token, duration, cache_hit=false, cached_tokens=0)`.
    pub fn prefill_with_audio(
        &mut self,
        tokens: &[u32],
        audio_embeds: Vec<Tensor>,
    ) -> Result<(u32, std::time::Duration, bool, usize)> {
        let start = std::time::Instant::now();
        let token_tensor = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        trace!(
            token_count = tokens.len(),
            audio_chunks = audio_embeds.len(),
            "Running modal prefill forward pass",
        );
        let logits = self.model.forward_modal(ModelInput::Mixed {
            token_ids: token_tensor,
            audio_embeds,
            offset: 0,
        })?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = self.sample_guided(logits, tokens)?;
        trace!(next_token, "Modal prefill sampled first token");
        Ok((next_token, start.elapsed(), false, 0))
    }

    /// Perform single decode step with KV cache reuse
    /// Returns next token
    pub fn decode_step(&mut self, tokens: &[u32], start_pos: usize) -> Result<u32> {
        let ctxt = &tokens[tokens.len().saturating_sub(1)..];

        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        trace!(
            tokens = ctxt.len(),
            start_pos, "Decoding next token from context",
        );
        let logits = self.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = self.sample_guided(logits, tokens)?;
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

    /// Perform batched prefill: encode prompt tokens and return the next token.
    ///
    /// Each entry in `batch_tokens` is `(request_id, chunk_tokens, kv_start_pos)`:
    /// - `chunk_tokens`: the tokens to process this call (may be a subset of the
    ///   full prompt when chunked prefill is enabled).
    /// - `kv_start_pos`: the position in the KV cache where `chunk_tokens` begin.
    ///   Pass `0` for the first (or only) chunk; pass the number of tokens already
    ///   processed for subsequent chunks.
    ///
    /// The prefix cache is consulted only when `kv_start_pos == 0`; intermediate
    /// chunks skip it because their tokens cannot match a full-prompt cache entry.
    ///
    /// Returns `(request_id, sampled_token)` pairs.  For intermediate prefill chunks
    /// the caller should discard the sampled token — it is the model's prediction at
    /// the end of that chunk, not the first generated token.
    pub fn prefill_batch(
        &mut self,
        batch_tokens: &[(Uuid, Vec<u32>, usize)],
        paged_caches: Option<&mut [Vec<PagedKvCache>]>,
    ) -> Result<Vec<(Uuid, u32)>> {
        if batch_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // When paged KV caches are in use we must prefill each request individually.
        //
        // Batched prefill pads shorter sequences to max_len with zeros.  The paged
        // cache blindly stores all max_len K/V entries — including the garbage padding
        // tokens — and ModelForCausalLM::forward_batch extracts logits from the last
        // padded position rather than the last real token.  Both corrupt decode quality.
        //
        // Per-request prefill avoids padding entirely: each forward pass sees only the
        // real prompt tokens, stores exactly the right K/V entries, and extracts the
        // logit at the correct final position.
        if let Some(caches) = paged_caches {
            let mut results = Vec::with_capacity(batch_tokens.len());
            for ((id, tokens, kv_start_pos), req_caches) in
                batch_tokens.iter().zip(caches.iter_mut())
            {
                // Prefix-cache restore is only meaningful for the first chunk
                // (kv_start_pos == 0).  Intermediate chunks arrive with KV already
                // populated for earlier positions, so a prefix lookup would try to
                // overwrite already-written entries with mismatched state.
                let (cache_hit, cached_len) = if *kv_start_pos == 0 {
                    self.try_restore_paged(tokens, req_caches)?
                } else {
                    (false, 0)
                };

                let process_tokens = &tokens[cached_len..];
                let effective_pos = kv_start_pos + cached_len;
                let positions = [effective_pos];
                let input = Tensor::new(process_tokens, &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward_batch(
                    &input,
                    &positions,
                    Some(std::slice::from_mut(req_caches)),
                )?;

                if !cache_hit && self.prefix_cache.is_some() && *kv_start_pos == 0 {
                    self.store_paged_to_cache(tokens, req_caches)?;
                }

                // logits: [1, 1, vocab_size] — squeeze to [vocab_size]
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let logits = self.process_logits(logits, tokens)?;
                let next_token = self.sampler.sample(&logits)?;
                results.push((*id, next_token));
                trace!(
                    request_id = ?id,
                    next_token,
                    kv_start_pos,
                    cache_hit,
                    cached_tokens = cached_len,
                    "Batched prefill sampled token"
                );
            }
            return Ok(results);
        }

        // No paged caches: use the shared model KV cache and a single padded batch.
        // Chunked prefill is not supported on this path (it requires per-request
        // paged caches to hold intermediate KV state between chunks).
        self.model.clear_kv_cache();

        // Find max length in batch
        let max_len = batch_tokens
            .iter()
            .map(|(_, tokens, _)| tokens.len())
            .max()
            .unwrap_or(0);

        let batch_size = batch_tokens.len();
        let mut input_data = Vec::with_capacity(batch_size * max_len);
        let positions = vec![0usize; batch_size];

        for (_, tokens, _) in batch_tokens.iter() {
            input_data.extend_from_slice(tokens);
            input_data.extend(std::iter::repeat_n(0, max_len.saturating_sub(tokens.len())));
        }

        let input = Tensor::from_vec(input_data, (batch_size, max_len), &self.device)?;
        trace!(
            batch_size,
            max_len,
            has_paged_caches = false,
            "Running batched prefill forward pass"
        );

        let logits = self.model.forward_batch(&input, &positions, None)?;

        let mut results = Vec::with_capacity(batch_size);
        for (idx, (id, tokens, _)) in batch_tokens.iter().enumerate() {
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

        let batch_size = batch_data.len();

        // Non-paged decode relies on the model's shared SimpleKvCache accumulating state
        // from the preceding prefill — do NOT clear it here.  The shared cache cannot
        // serve multiple concurrent requests correctly, so only batch_size == 1 is valid
        // on this path; use paged caches for true multi-request batching.
        debug_assert!(
            paged_caches.is_some() || batch_size == 1,
            "non-paged batched decode requires batch_size == 1; use paged caches for multi-request batching"
        );

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
