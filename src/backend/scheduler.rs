//! Continuous batching scheduler for dynamic request management
//!
//! This module implements a continuous batching scheduler that:
//! - Manages request queues with different scheduling policies
//! - Performs memory-aware admission control
//! - Forms separate batches for prefill and decode phases
//! - Integrates with the memory pool for resource management

use crate::{
    Result, TurError,
    backend::{
        batch_manager::{BatchManager, RequestPhase, RequestState},
        factory::ModelConstructor,
        memory_pool::MemoryPool,
    },
    models::kv_cache::{BlockAllocator, PagedKvCache},
};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, trace, warn};
use uuid::Uuid;

/// Scheduling policy for request admission
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-Come-First-Served: admit requests in arrival order
    FCFS,
    /// Shortest-Job-First: prioritize requests with fewer tokens
    SJF,
    /// Priority-based: use request priority field
    Priority,
}

/// Batch configuration limits
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of requests in a prefill batch
    pub max_prefill_batch: usize,
    /// Maximum number of requests in a decode batch
    pub max_decode_batch: usize,
    /// Maximum total tokens in a prefill batch
    pub max_prefill_tokens: usize,
    /// Maximum total tokens in a decode batch
    pub max_decode_tokens: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_prefill_batch: 8,
            max_decode_batch: 64,
            max_prefill_tokens: 2048,
            max_decode_tokens: 4096,
        }
    }
}

/// Batch of requests ready for execution
#[derive(Debug, Clone)]
pub struct ExecutionBatch {
    /// Request IDs in this batch
    pub request_ids: Vec<Uuid>,
    /// Phase of this batch (all requests must be in same phase)
    pub phase: RequestPhase,
    /// Total number of tokens across all requests
    pub total_tokens: usize,
}

type ScheduleIterationOutput = (Uuid, Vec<u32>, String, std::time::Instant);

/// Continuous batch scheduler
pub struct ContinuousBatchScheduler {
    batch_manager: BatchManager,
    policy: SchedulingPolicy,
    config: BatchConfig,
    memory_pool: Arc<RwLock<MemoryPool>>,
    /// Track allocated blocks per request (request_id -> num_blocks)
    allocated_blocks: ahash::AHashMap<uuid::Uuid, usize>,
    /// Block allocator for paged KV cache (shared across all requests)
    block_allocator: Arc<RwLock<BlockAllocator>>,
    /// PagedKvCache instances per request (one per layer per request)
    /// Map: request_id -> Vec<PagedKvCache> (one cache per model layer)
    paged_caches: ahash::AHashMap<Uuid, Vec<PagedKvCache>>,
    /// Number of model layers (needed for PagedKvCache initialization)
    num_layers: usize,
    /// EOS token IDs derived from the model's tokenizer at construction time.
    eos_token: u32,
    im_end_token: u32,
}

impl ContinuousBatchScheduler {
    /// Create a new scheduler.
    ///
    /// `eos_tokens` should be `(eos_token_id, im_end_token_id)` looked up from the
    /// model's tokenizer by the caller so generation stops on the correct tokens for
    /// every model family, not just Qwen.
    pub fn new(
        policy: SchedulingPolicy,
        config: BatchConfig,
        memory_pool: Arc<RwLock<MemoryPool>>,
        allocator: Arc<RwLock<BlockAllocator>>,
        num_layers: usize,
        eos_tokens: (u32, u32),
    ) -> Self {
        debug!(
            policy = ?policy,
            max_prefill_batch = config.max_prefill_batch,
            max_decode_batch = config.max_decode_batch,
            max_prefill_tokens = config.max_prefill_tokens,
            max_decode_tokens = config.max_decode_tokens,
            eos_token = eos_tokens.0,
            im_end_token = eos_tokens.1,
            "Creating continuous batch scheduler"
        );

        // MemoryPool and BlockAllocator must agree on block_size; a mismatch means the
        // admission-control estimate and the actual physical allocation use different units.
        debug_assert_eq!(
            memory_pool.read().block_size,
            allocator.read().block_size(),
            "MemoryPool block_size must equal BlockAllocator block_size"
        );

        Self {
            batch_manager: BatchManager::new(config.max_prefill_batch, config.max_decode_batch),
            policy,
            config,
            memory_pool,
            allocated_blocks: ahash::AHashMap::new(),
            block_allocator: allocator,
            paged_caches: ahash::AHashMap::new(),
            num_layers,
            eos_token: eos_tokens.0,
            im_end_token: eos_tokens.1,
        }
    }

    /// Initialize PagedKvCache for a request (one cache per model layer)
    /// If a prefix_request_id is provided, fork from that request's caches for prefix sharing
    fn initialize_paged_cache(
        &mut self,
        request_id: &Uuid,
        prefix_request_id: Option<&Uuid>,
    ) -> Result<()> {
        let caches: Vec<PagedKvCache> = if let Some(prefix_id) = prefix_request_id {
            // Fork from existing request's caches for prefix sharing
            if let Some(prefix_caches) = self.paged_caches.get(prefix_id) {
                let forked_caches: candle_core::Result<Vec<_>> =
                    prefix_caches.iter().map(|cache| cache.fork()).collect();
                trace!(
                    request_id = %request_id,
                    prefix_request_id = %prefix_id,
                    num_layers = self.num_layers,
                    "Forked PagedKvCache from prefix request for sharing"
                );
                forked_caches?
            } else {
                // Prefix request not found, create new caches
                trace!(
                    request_id = %request_id,
                    prefix_request_id = %prefix_id,
                    "Prefix request not found, creating new caches"
                );
                (0..self.num_layers)
                    .map(|_| PagedKvCache::new(self.block_allocator.clone(), 2))
                    .collect()
            }
        } else {
            // No prefix sharing, create new caches
            trace!(request_id = %request_id, num_layers = self.num_layers, "Initialized new PagedKvCache for request");
            (0..self.num_layers)
                .map(|_| PagedKvCache::new(self.block_allocator.clone(), 2))
                .collect()
        };

        self.paged_caches.insert(*request_id, caches);
        Ok(())
    }

    /// Find a request with matching prefix for potential cache sharing
    /// Returns the request ID if a suitable prefix match is found
    fn find_prefix_match(&self, tokens: &[u32]) -> Option<Uuid> {
        // Simple prefix matching: find any active request that shares a prefix
        // For efficiency, we only check if the new request's tokens are a prefix of an existing request
        // or if an existing request's tokens are a prefix of the new request

        const MIN_PREFIX_LENGTH: usize = 10;
        if tokens.len() < MIN_PREFIX_LENGTH {
            return None;
        }

        let mut best_match: Option<(Uuid, usize)> = None;

        // Check all active requests with paged caches
        for (request_id, _) in &self.paged_caches {
            if let Some(request) = self.batch_manager.get_request(request_id) {
                let request_tokens = &request.prompt_tokens;

                // Find common prefix length
                let common_len = tokens
                    .iter()
                    .zip(request_tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                // Only consider if common prefix is significant
                if common_len >= MIN_PREFIX_LENGTH {
                    // Update best match if this is longer
                    if best_match.is_none_or(|(_, len)| common_len > len) {
                        best_match = Some((*request_id, common_len));
                    }
                }
            }
        }

        if let Some((id, len)) = best_match {
            trace!(
                prefix_request_id = %id,
                common_prefix_length = len,
                "Found prefix match for cache sharing"
            );
            Some(id)
        } else {
            None
        }
    }

    /// Get block tables for a batch of requests.
    ///
    /// Returns the first layer's block table for each request as a representative.
    /// Block IDs are sourced directly from the live `PagedKvCache` — not from
    /// `RequestState::block_table`, which is no longer populated.
    pub fn get_block_tables(
        &self,
        request_ids: &[Uuid],
    ) -> Option<Vec<Vec<crate::models::kv_cache::BlockId>>> {
        let mut block_tables = Vec::with_capacity(request_ids.len());
        for id in request_ids {
            if let Some(caches) = self.paged_caches.get(id) {
                let table = caches
                    .first()
                    .map(|c| c.block_table().to_vec())
                    .unwrap_or_default();
                block_tables.push(table);
            } else {
                return None;
            }
        }
        Some(block_tables)
    }

    /// Get mutable paged caches for a batch of requests
    /// Temporarily removes them from the scheduler - caller must put them back!
    pub fn get_paged_caches_mut(&mut self, request_ids: &[Uuid]) -> Vec<Vec<PagedKvCache>> {
        request_ids
            .iter()
            .filter_map(|id| self.paged_caches.remove(id))
            .collect()
    }

    /// Put paged caches back into the scheduler after use
    pub fn put_paged_caches(&mut self, request_ids: &[Uuid], mut caches: Vec<Vec<PagedKvCache>>) {
        for (id, cache) in request_ids.iter().zip(caches.drain(..)) {
            self.paged_caches.insert(*id, cache);
        }
    }

    /// Enqueue a new request.
    ///
    /// The request's priority is normalised based on the scheduler's policy before
    /// it is handed to `BatchManager`, which inserts it in priority order.  This
    /// means `BatchManager`'s sorted queue naturally implements all three policies:
    ///
    /// * `FCFS`     — all priorities forced to 0, so insertion order is preserved.
    /// * `Priority` — the caller-supplied `request.priority` is used as-is.
    /// * `SJF`      — shorter prompts receive a higher effective priority so they are admitted first.
    pub fn enqueue_request(&mut self, mut request: RequestState) {
        match self.policy {
            SchedulingPolicy::FCFS => {
                request.priority = 0;
            }
            SchedulingPolicy::Priority => {
                // Use the request's own priority as-is.
            }
            SchedulingPolicy::SJF => {
                // Shorter prompts → higher effective priority.
                // Saturating_sub prevents overflow for very long prompts.
                request.priority = u32::MAX.saturating_sub(request.prompt_tokens.len() as u32);
            }
        }
        self.batch_manager.enqueue_request(request)
    }

    /// Get a request by ID
    pub fn get_request(&self, request_id: &Uuid) -> Option<&RequestState> {
        self.batch_manager.get_request(request_id)
    }

    /// Get a mutable request by ID
    pub fn get_request_mut(&mut self, request_id: &Uuid) -> Option<&mut RequestState> {
        self.batch_manager.get_request_mut(request_id)
    }

    /// Admit queued requests based on memory availability and policy
    /// Returns list of admitted request IDs
    pub fn admit_requests(&mut self) -> Result<Vec<Uuid>> {
        let mut admitted = Vec::new();

        // Keep admitting while we have queued requests and memory
        while let Some(request) = self.batch_manager.peek_next_request() {
            let required_blocks = self.estimate_required_blocks(request)?;

            // Check if we have enough memory
            let mut pool = self.memory_pool.write();
            if pool.can_allocate(required_blocks) {
                // Allocate the blocks
                pool.allocate(required_blocks)
                    .map_err(TurError::MemoryAllocation)?;
                drop(pool); // Release lock before calling batch_manager

                // Admit the request (removes from queue, adds to active)
                if let Some(request_id) = self.batch_manager.admit_request() {
                    // Clone prompt tokens before calling find_prefix_match to avoid
                    // holding a reference into batch_manager across the &self borrow.
                    let prompt_tokens = self
                        .batch_manager
                        .get_request(&request_id)
                        .map(|r| r.prompt_tokens.clone())
                        .unwrap_or_default();

                    // Check for prefix match for potential cache sharing
                    let prefix_match = self.find_prefix_match(&prompt_tokens);

                    // Initialize PagedKvCache for this request (with optional prefix sharing via fork).
                    // PagedKvCache owns its own block table and allocates blocks lazily as tokens
                    // are written — do NOT pre-allocate blocks here as well (that would cause
                    // double allocation from the same BlockAllocator).
                    self.initialize_paged_cache(&request_id, prefix_match.as_ref())?;

                    // Track how many MemoryPool blocks were reserved for admission control.
                    self.allocated_blocks.insert(request_id, required_blocks);
                    trace!(
                        request_id = %request_id,
                        required_blocks,
                        "Reserved MemoryPool blocks and initialized PagedKvCache for request"
                    );
                    admitted.push(request_id);
                } else {
                    break;
                }
            } else {
                warn!("Memory exhausted, cannot admit more requests");
                break;
            }
        }

        if !admitted.is_empty() {
            debug!(
                admitted_count = admitted.len(),
                "Admitted requests to scheduler"
            );
        }

        Ok(admitted)
    }

    /// Form a prefill batch from admitted requests.
    ///
    /// Respects both `config.max_prefill_batch` (request count, enforced by
    /// `BatchManager`) and `config.max_prefill_tokens` (total token budget,
    /// enforced here by stopping early once the budget would be exceeded).
    pub fn form_prefill_batch(&self) -> Option<ExecutionBatch> {
        let candidates = self.batch_manager.form_prefill_batch();
        if candidates.is_empty() {
            return None;
        }

        let mut selected = Vec::with_capacity(candidates.len());
        let mut total_tokens = 0usize;

        for id in &candidates {
            if let Some(request) = self.batch_manager.get_request(id) {
                let req_tokens = request.seq_len();
                // Always include the first request even if it alone exceeds the budget;
                // refusing a single oversized request would stall the scheduler forever.
                if !selected.is_empty()
                    && total_tokens + req_tokens > self.config.max_prefill_tokens
                {
                    break;
                }
                total_tokens += req_tokens;
                selected.push(*id);
            }
        }

        if selected.is_empty() {
            return None;
        }

        trace!(
            batch_size = selected.len(),
            total_tokens,
            max_prefill_tokens = self.config.max_prefill_tokens,
            "Formed prefill batch"
        );

        Some(ExecutionBatch {
            request_ids: selected,
            phase: RequestPhase::Prefilling,
            total_tokens,
        })
    }

    /// Form a decode batch from decoding requests.
    ///
    /// Respects both `config.max_decode_batch` (request count) and
    /// `config.max_decode_tokens` (total token budget).
    pub fn form_decode_batch(&self) -> Option<ExecutionBatch> {
        let candidates = self.batch_manager.form_decode_batch();
        if candidates.is_empty() {
            return None;
        }

        let mut selected = Vec::with_capacity(candidates.len());
        let mut total_tokens = 0usize;

        for id in &candidates {
            if let Some(request) = self.batch_manager.get_request(id) {
                let req_tokens = request.seq_len();
                if !selected.is_empty() && total_tokens + req_tokens > self.config.max_decode_tokens
                {
                    break;
                }
                total_tokens += req_tokens;
                selected.push(*id);
            }
        }

        if selected.is_empty() {
            return None;
        }

        trace!(
            batch_size = selected.len(),
            total_tokens,
            max_decode_tokens = self.config.max_decode_tokens,
            "Formed decode batch"
        );

        Some(ExecutionBatch {
            request_ids: selected,
            phase: RequestPhase::Decoding,
            total_tokens,
        })
    }

    /// Transition a request from prefilling to decoding
    pub fn transition_to_decode(&mut self, request_id: &Uuid) -> Result<()> {
        self.batch_manager.transition_to_decode(request_id)
    }

    /// Mark a request as completed and free its memory
    pub fn complete_request(&mut self, request_id: &Uuid) -> Result<Vec<u32>> {
        // Get allocated MemoryPool blocks for this request
        let blocks_to_free = self.allocated_blocks.remove(request_id).unwrap_or(0);

        // Complete the request in batch manager
        let tokens = self.batch_manager.complete_request(request_id)?;

        // Free memory pool reservation
        if blocks_to_free > 0 {
            let mut pool = self.memory_pool.write();
            pool.free(blocks_to_free);
            trace!(
                request_id = %request_id,
                freed_blocks = blocks_to_free,
                "Freed MemoryPool reservation for completed request"
            );
        }

        // Drop PagedKvCache — its Drop impl calls reset() which frees all BlockAllocator blocks.
        self.paged_caches.remove(request_id);

        Ok(tokens)
    }

    /// Mark a request as failed and free its memory
    pub fn fail_request(&mut self, request_id: &Uuid, error: String) -> Result<()> {
        // Get allocated MemoryPool blocks for this request
        let blocks_to_free = self.allocated_blocks.remove(request_id).unwrap_or(0);

        // Fail the request in batch manager
        self.batch_manager.fail_request(request_id, error)?;

        // Free memory pool reservation
        if blocks_to_free > 0 {
            let mut pool = self.memory_pool.write();
            pool.free(blocks_to_free);
            trace!(
                request_id = %request_id,
                freed_blocks = blocks_to_free,
                "Freed MemoryPool reservation for failed request"
            );
        }

        // Drop PagedKvCache — its Drop impl calls reset() which frees all BlockAllocator blocks.
        self.paged_caches.remove(request_id);

        Ok(())
    }

    /// Get statistics about the scheduler state
    pub fn get_stats(&self) -> SchedulerStats {
        let batch_stats = self.batch_manager.stats();
        let pool = self.memory_pool.read();

        SchedulerStats {
            queued_requests: batch_stats.queued_requests,
            prefilling_requests: batch_stats.prefill_requests,
            decoding_requests: batch_stats.decode_requests,
            completed_requests: batch_stats.completed_requests,
            allocated_blocks: pool.allocated_blocks(),
            free_blocks: pool.free_blocks(),
            total_blocks: pool.total_blocks(),
        }
    }

    /// Estimate required blocks for a request
    fn estimate_required_blocks(&self, request: &RequestState) -> Result<usize> {
        // Use block size from memory pool
        let pool = self.memory_pool.read();
        let block_size = pool.block_size;
        drop(pool);

        let prompt_tokens = request.prompt_tokens.len();
        let max_tokens = request.max_tokens;
        let total_tokens = prompt_tokens + max_tokens;

        let blocks = total_tokens.div_ceil(block_size);
        Ok(blocks)
    }

    /// Check if there are any active requests
    pub fn has_active_requests(&self) -> bool {
        self.batch_manager.num_active_requests() > 0
    }

    /// Check if there are any queued requests
    pub fn has_queued_requests(&self) -> bool {
        self.batch_manager.num_queued_requests() > 0
    }

    /// True when there is any work to do (active or queued).
    pub fn has_pending_work(&self) -> bool {
        self.has_active_requests() || self.has_queued_requests()
    }

    /// Get number of active requests
    pub fn active_request_count(&self) -> usize {
        self.batch_manager.num_active_requests()
    }

    /// Get number of queued requests
    pub fn queued_request_count(&self) -> usize {
        self.batch_manager.num_queued_requests()
    }

    /// Get request state by ID
    pub fn get_request_state(&self, request_id: &Uuid) -> Option<&RequestState> {
        self.batch_manager.get_request(request_id)
    }

    /// Main scheduling iteration - admits requests, forms batches, and executes them
    /// Returns list of completed requests with their generated tokens, prompt, and arrival time
    pub fn schedule_iteration<T: ModelConstructor>(
        &mut self,
        engine: &mut crate::backend::engine::InferenceEngine<T>,
    ) -> Result<Vec<ScheduleIterationOutput>> {
        trace!("Starting scheduler iteration");

        // 1. Admit new requests from queue
        let admitted = self.admit_requests()?;
        if !admitted.is_empty() {
            debug!("Admitted {} new requests", admitted.len());
        }

        let mut completed = Vec::new();

        // 2. Form and execute prefill batch
        if let Some(prefill_batch) = self.form_prefill_batch() {
            debug!(
                "Executing prefill batch: {} requests, {} tokens",
                prefill_batch.request_ids.len(),
                prefill_batch.total_tokens
            );

            // Collect tokens for each request in the batch
            let batch_tokens: Vec<(Uuid, Vec<u32>)> = prefill_batch
                .request_ids
                .iter()
                .filter_map(|id| {
                    self.get_request(id)
                        .map(|req| (*id, req.prompt_tokens.clone()))
                })
                .collect();

            // Collect mutable references to paged caches for the batch
            let request_ids_vec: Vec<Uuid> = prefill_batch.request_ids.clone();
            let mut paged_caches_vec: Vec<Vec<PagedKvCache>> = request_ids_vec
                .iter()
                .filter_map(|id| self.paged_caches.remove(id))
                .collect();

            // Execute prefill batch
            let results = if !paged_caches_vec.is_empty() {
                engine.prefill_batch(&batch_tokens, Some(&mut paged_caches_vec))?
            } else {
                engine.prefill_batch(&batch_tokens, None)?
            };

            // Put paged caches back to scheduler (they were modified during forward pass)
            for (id, cache) in request_ids_vec.iter().zip(paged_caches_vec.drain(..)) {
                self.paged_caches.insert(*id, cache);
            }

            // Update request states with first generated token
            for (request_id, first_token) in results {
                if let Some(request) = self.get_request_mut(&request_id) {
                    request.generated_tokens.push(first_token);
                    request.position = request.prompt_tokens.len();
                }
                // Transition to decode phase
                self.transition_to_decode(&request_id)?;
            }
        }

        // 3. Form and execute decode batch
        if let Some(decode_batch) = self.form_decode_batch() {
            debug!(
                "Executing decode batch: {} requests",
                decode_batch.request_ids.len()
            );

            // Collect tokens and positions for each request
            let batch_data: Vec<(Uuid, Vec<u32>, usize)> = decode_batch
                .request_ids
                .iter()
                .filter_map(|id| {
                    self.get_request(id).map(|req| {
                        let all_tokens = req.all_tokens();
                        (*id, all_tokens, req.position)
                    })
                })
                .collect();

            // Collect mutable references to paged caches for the batch
            let request_ids_vec: Vec<Uuid> = decode_batch.request_ids.clone();
            let mut paged_caches_vec: Vec<Vec<PagedKvCache>> = request_ids_vec
                .iter()
                .filter_map(|id| self.paged_caches.remove(id))
                .collect();

            // Execute decode batch
            let results = if !paged_caches_vec.is_empty() {
                engine.decode_batch(&batch_data, Some(&mut paged_caches_vec))?
            } else {
                engine.decode_batch(&batch_data, None)?
            };

            // Put paged caches back to scheduler (they were modified during forward pass)
            for (id, cache) in request_ids_vec.iter().zip(paged_caches_vec.drain(..)) {
                self.paged_caches.insert(*id, cache);
            }

            // Update request states and check for completion
            for (request_id, next_token) in results {
                let should_complete = if let Some(request) = self.get_request_mut(&request_id) {
                    request.generated_tokens.push(next_token);
                    request.position += 1;

                    // Check if request should stop
                    request.should_stop()
                        || next_token == self.eos_token
                        || next_token == self.im_end_token
                } else {
                    false
                };

                // Complete request if needed
                if should_complete {
                    // Get request info BEFORE completing (which removes it)
                    let (prompt, arrival_time) = if let Some(req) = self.get_request(&request_id) {
                        (req.prompt.clone(), req.arrival_time)
                    } else {
                        continue;
                    };

                    let tokens = self.complete_request(&request_id)?;
                    completed.push((request_id, tokens, prompt, arrival_time));
                    debug!("Request {} completed", request_id);
                }
            }
        }

        Ok(completed)
    }
}

/// Statistics about scheduler state
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub queued_requests: usize,
    pub prefilling_requests: usize,
    pub decoding_requests: usize,
    pub completed_requests: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub total_blocks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn create_test_scheduler(policy: SchedulingPolicy) -> ContinuousBatchScheduler {
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(
            1024 * 1024 * 100, // 100MB
            16,
            32,
            32,
            128,
            2,
        )));
        let block_allocator = Arc::new(RwLock::new(BlockAllocator::new(1000, 16)));
        let config = BatchConfig::default();
        ContinuousBatchScheduler::new(policy, config, memory_pool, block_allocator, 32, (0, 0))
    }

    fn create_test_request(prompt_len: usize, max_tokens: usize, priority: u32) -> RequestState {
        let id = Uuid::new_v4();
        let prompt = "test prompt".to_string();
        let tokens = vec![1u32; prompt_len];
        RequestState::new(id, prompt, tokens, max_tokens, priority)
    }

    #[test]
    fn test_scheduler_enqueue() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);
        let request = create_test_request(10, 50, 0);

        scheduler.enqueue_request(request);

        // Request should be in queue, not active yet
        let stats = scheduler.get_stats();
        assert_eq!(stats.queued_requests, 1);
        assert_eq!(stats.prefilling_requests, 0);
    }

    #[test]
    fn test_scheduler_admit_requests() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        // Enqueue multiple requests
        let req1 = create_test_request(10, 50, 0);
        let req2 = create_test_request(20, 50, 0);
        let id1 = req1.id;
        let id2 = req2.id;

        scheduler.enqueue_request(req1);
        scheduler.enqueue_request(req2);

        // Admit requests
        let admitted = scheduler.admit_requests().unwrap();
        assert_eq!(admitted.len(), 2);
        assert!(admitted.contains(&id1));
        assert!(admitted.contains(&id2));

        // Check they're in prefilling phase
        assert_eq!(
            scheduler.get_request(&id1).unwrap().phase,
            RequestPhase::Prefilling
        );
        assert_eq!(
            scheduler.get_request(&id2).unwrap().phase,
            RequestPhase::Prefilling
        );
    }

    #[test]
    fn test_scheduler_form_prefill_batch() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        // Enqueue and admit requests
        let req1 = create_test_request(10, 50, 0);
        let req2 = create_test_request(20, 50, 0);
        let id1 = req1.id;
        let id2 = req2.id;

        scheduler.enqueue_request(req1);
        scheduler.enqueue_request(req2);
        scheduler.admit_requests().unwrap();

        // Form prefill batch
        let batch = scheduler.form_prefill_batch();
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.phase, RequestPhase::Prefilling);
        assert_eq!(batch.request_ids.len(), 2);
        assert!(batch.request_ids.contains(&id1));
        assert!(batch.request_ids.contains(&id2));
    }

    #[test]
    fn test_scheduler_transition_to_decode() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let request = create_test_request(10, 50, 0);
        let request_id = request.id;

        scheduler.enqueue_request(request);
        scheduler.admit_requests().unwrap();

        // Transition to decode
        scheduler.transition_to_decode(&request_id).unwrap();
        assert_eq!(
            scheduler.get_request(&request_id).unwrap().phase,
            RequestPhase::Decoding
        );
    }

    #[test]
    fn test_scheduler_form_decode_batch() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let req1 = create_test_request(10, 50, 0);
        let req2 = create_test_request(20, 50, 0);
        let id1 = req1.id;
        let id2 = req2.id;

        scheduler.enqueue_request(req1);
        scheduler.enqueue_request(req2);
        scheduler.admit_requests().unwrap();

        // Transition both to decode
        scheduler.transition_to_decode(&id1).unwrap();
        scheduler.transition_to_decode(&id2).unwrap();

        // Form decode batch
        let batch = scheduler.form_decode_batch();
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.phase, RequestPhase::Decoding);
        assert_eq!(batch.request_ids.len(), 2);
        assert!(batch.request_ids.contains(&id1));
        assert!(batch.request_ids.contains(&id2));
    }

    #[test]
    fn test_scheduler_complete_request() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let request = create_test_request(10, 50, 0);
        let request_id = request.id;

        scheduler.enqueue_request(request);
        scheduler.admit_requests().unwrap();

        // Complete request
        let tokens = scheduler.complete_request(&request_id).unwrap();
        assert!(!tokens.is_empty());

        // Request should no longer be active
        assert!(scheduler.get_request(&request_id).is_none());
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let req1 = create_test_request(10, 50, 0);
        let req2 = create_test_request(20, 50, 0);
        let id1 = req1.id;

        scheduler.enqueue_request(req1);
        scheduler.enqueue_request(req2);

        let stats = scheduler.get_stats();
        assert_eq!(stats.queued_requests, 2);
        assert_eq!(stats.prefilling_requests, 0);
        assert_eq!(stats.decoding_requests, 0);

        scheduler.admit_requests().unwrap();
        let stats = scheduler.get_stats();
        assert_eq!(stats.queued_requests, 0);
        assert_eq!(stats.prefilling_requests, 2);

        scheduler.transition_to_decode(&id1).unwrap();
        let stats = scheduler.get_stats();
        assert_eq!(stats.prefilling_requests, 1);
        assert_eq!(stats.decoding_requests, 1);
    }

    #[test]
    fn test_scheduler_memory_limit() {
        // Create scheduler with very limited memory
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(
            1024 * 10, // Very small: 10KB
            16,
            32,
            32,
            128,
            2,
        )));
        let block_allocator = Arc::new(RwLock::new(BlockAllocator::new(100, 16)));
        let config = BatchConfig::default();
        let mut scheduler = ContinuousBatchScheduler::new(
            SchedulingPolicy::FCFS,
            config,
            memory_pool,
            block_allocator,
            32,
            (0, 0),
        );

        // Enqueue many large requests
        for _ in 0..10 {
            let request = create_test_request(100, 100, 0);
            scheduler.enqueue_request(request);
        }

        // Should only admit requests that fit in memory
        let admitted = scheduler.admit_requests().unwrap();
        assert!(admitted.len() < 10); // Not all requests should be admitted

        let stats = scheduler.get_stats();
        assert!(stats.queued_requests > 0); // Some should remain queued
    }

    #[test]
    fn test_scheduler_fail_request() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let request = create_test_request(10, 50, 0);
        let request_id = request.id;

        scheduler.enqueue_request(request);
        scheduler.admit_requests().unwrap();

        // Fail request
        scheduler
            .fail_request(&request_id, "Test error".to_string())
            .unwrap();

        // Request must be removed from the active set after failure
        assert!(scheduler.get_request(&request_id).is_none());
        let stats = scheduler.get_stats();
        assert_eq!(stats.prefilling_requests + stats.decoding_requests, 0);
    }

    #[test]
    fn test_scheduler_memory_tracking() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let initial_stats = scheduler.get_stats();
        let initial_allocated = initial_stats.allocated_blocks;

        // Enqueue and admit a request
        let request = create_test_request(10, 50, 0);
        let request_id = request.id;
        scheduler.enqueue_request(request);
        scheduler.admit_requests().unwrap();

        // Memory should be allocated
        let after_admit_stats = scheduler.get_stats();
        assert!(after_admit_stats.allocated_blocks > initial_allocated);

        // Complete request
        scheduler.complete_request(&request_id).unwrap();

        // Memory should be freed
        let after_complete_stats = scheduler.get_stats();
        assert_eq!(after_complete_stats.allocated_blocks, initial_allocated);
    }

    #[test]
    fn test_scheduler_empty_batches() {
        let scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        // No requests - should return None
        assert!(scheduler.form_prefill_batch().is_none());
        assert!(scheduler.form_decode_batch().is_none());
    }

    #[test]
    fn test_scheduler_invalid_transition() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let request = create_test_request(10, 50, 0);
        let request_id = request.id;

        scheduler.enqueue_request(request);
        scheduler.admit_requests().unwrap();

        // Transition to decode
        scheduler.transition_to_decode(&request_id).unwrap();

        // Try to transition again - should fail
        let result = scheduler.transition_to_decode(&request_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_scheduler_request_not_found() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);
        let fake_id = Uuid::new_v4();

        // Try to complete non-existent request
        let result = scheduler.complete_request(&fake_id);
        assert!(result.is_err());

        // Try to fail non-existent request
        let result = scheduler.fail_request(&fake_id, "error".to_string());
        assert!(result.is_err());

        // Try to transition non-existent request
        let result = scheduler.transition_to_decode(&fake_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_scheduler_batch_limits() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        // Enqueue more requests than max batch size
        let mut ids = Vec::new();
        for _ in 0..20 {
            let request = create_test_request(10, 50, 0);
            ids.push(request.id);
            scheduler.enqueue_request(request);
        }

        scheduler.admit_requests().unwrap();

        // Prefill batch should respect max_prefill_batch limit
        let batch = scheduler.form_prefill_batch().unwrap();
        assert!(batch.request_ids.len() <= scheduler.config.max_prefill_batch);

        // Transition all to decode
        for id in &ids {
            if scheduler.get_request(id).is_some() {
                let _ = scheduler.transition_to_decode(id);
            }
        }

        // Decode batch should respect max_decode_batch limit
        let batch = scheduler.form_decode_batch().unwrap();
        assert!(batch.request_ids.len() <= scheduler.config.max_decode_batch);
    }

    #[test]
    fn test_scheduler_full_lifecycle() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        let request = create_test_request(10, 50, 0);
        let request_id = request.id;

        // 1. Enqueue
        scheduler.enqueue_request(request);
        assert_eq!(scheduler.get_stats().queued_requests, 1);

        // 2. Admit
        let admitted = scheduler.admit_requests().unwrap();
        assert_eq!(admitted.len(), 1);
        assert_eq!(scheduler.get_stats().prefilling_requests, 1);

        // 3. Form prefill batch
        let batch = scheduler.form_prefill_batch().unwrap();
        assert_eq!(batch.request_ids.len(), 1);
        assert_eq!(batch.phase, RequestPhase::Prefilling);

        // 4. Transition to decode
        scheduler.transition_to_decode(&request_id).unwrap();
        assert_eq!(scheduler.get_stats().decoding_requests, 1);

        // 5. Form decode batch
        let batch = scheduler.form_decode_batch().unwrap();
        assert_eq!(batch.request_ids.len(), 1);
        assert_eq!(batch.phase, RequestPhase::Decoding);

        // 6. Complete
        let tokens = scheduler.complete_request(&request_id).unwrap();
        assert!(!tokens.is_empty());
        assert!(scheduler.get_request(&request_id).is_none());
    }

    #[test]
    fn test_schedule_iteration() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        // Enqueue multiple requests
        let req1 = create_test_request(5, 10, 0);
        let req2 = create_test_request(8, 10, 0);
        let id1 = req1.id;
        let id2 = req2.id;

        scheduler.enqueue_request(req1);
        scheduler.enqueue_request(req2);

        // Verify initial state
        assert_eq!(scheduler.queued_request_count(), 2);
        assert_eq!(scheduler.active_request_count(), 0);

        // Note: We can't actually call schedule_iteration without a real model,
        // but we can test the individual components it uses:

        // 1. Test admission
        let admitted = scheduler.admit_requests().unwrap();
        assert_eq!(admitted.len(), 2);
        assert!(admitted.contains(&id1));
        assert!(admitted.contains(&id2));

        // 2. Test prefill batch formation
        let prefill_batch = scheduler.form_prefill_batch();
        assert!(prefill_batch.is_some());
        let batch = prefill_batch.unwrap();
        assert_eq!(batch.phase, RequestPhase::Prefilling);
        assert_eq!(batch.request_ids.len(), 2);

        // 3. Simulate prefill completion by transitioning to decode
        scheduler.transition_to_decode(&id1).unwrap();
        scheduler.transition_to_decode(&id2).unwrap();

        // 4. Test decode batch formation
        let decode_batch = scheduler.form_decode_batch();
        assert!(decode_batch.is_some());
        let batch = decode_batch.unwrap();
        assert_eq!(batch.phase, RequestPhase::Decoding);
        assert_eq!(batch.request_ids.len(), 2);

        // 5. Simulate completion
        let tokens1 = scheduler.complete_request(&id1).unwrap();
        let tokens2 = scheduler.complete_request(&id2).unwrap();
        assert!(!tokens1.is_empty());
        assert!(!tokens2.is_empty());

        // Verify final state
        assert_eq!(scheduler.active_request_count(), 0);
        assert_eq!(scheduler.queued_request_count(), 0);
    }

    #[test]
    fn test_schedule_iteration_workflow() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        // Create requests with different characteristics
        let short_req = create_test_request(3, 5, 0);
        let long_req = create_test_request(10, 20, 0);
        let short_id = short_req.id;
        let long_id = long_req.id;

        scheduler.enqueue_request(short_req);
        scheduler.enqueue_request(long_req);

        // Step 1: Admission
        let admitted = scheduler.admit_requests().unwrap();
        assert_eq!(admitted.len(), 2);

        // Step 2: Prefill phase
        let prefill_batch = scheduler.form_prefill_batch().unwrap();
        assert_eq!(prefill_batch.request_ids.len(), 2);
        assert_eq!(prefill_batch.phase, RequestPhase::Prefilling);

        // Verify requests are in prefilling state
        assert_eq!(
            scheduler.get_request(&short_id).unwrap().phase,
            RequestPhase::Prefilling
        );
        assert_eq!(
            scheduler.get_request(&long_id).unwrap().phase,
            RequestPhase::Prefilling
        );

        // Step 3: Transition to decode (simulating prefill completion)
        scheduler.transition_to_decode(&short_id).unwrap();
        scheduler.transition_to_decode(&long_id).unwrap();

        // Step 4: Decode phase
        let decode_batch = scheduler.form_decode_batch().unwrap();
        assert_eq!(decode_batch.request_ids.len(), 2);
        assert_eq!(decode_batch.phase, RequestPhase::Decoding);

        // Verify requests are in decoding state
        assert_eq!(
            scheduler.get_request(&short_id).unwrap().phase,
            RequestPhase::Decoding
        );
        assert_eq!(
            scheduler.get_request(&long_id).unwrap().phase,
            RequestPhase::Decoding
        );

        // Step 5: Complete short request first
        let short_tokens = scheduler.complete_request(&short_id).unwrap();
        assert_eq!(short_tokens.len(), 3); // Original prompt tokens

        // Long request should still be active
        assert!(scheduler.get_request(&long_id).is_some());
        assert_eq!(scheduler.active_request_count(), 1);

        // Step 6: Complete long request
        let long_tokens = scheduler.complete_request(&long_id).unwrap();
        assert_eq!(long_tokens.len(), 10);

        // All requests completed
        assert_eq!(scheduler.active_request_count(), 0);
    }

    #[test]
    fn test_schedule_iteration_mixed_phases() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::FCFS);

        // Create and admit first batch
        let req1 = create_test_request(5, 10, 0);
        let req2 = create_test_request(5, 10, 0);
        let id1 = req1.id;
        let id2 = req2.id;

        scheduler.enqueue_request(req1);
        scheduler.enqueue_request(req2);
        scheduler.admit_requests().unwrap();

        // Transition first request to decode
        scheduler.transition_to_decode(&id1).unwrap();

        // Now we have mixed phases: id1 in decode, id2 in prefill
        assert_eq!(
            scheduler.get_request(&id1).unwrap().phase,
            RequestPhase::Decoding
        );
        assert_eq!(
            scheduler.get_request(&id2).unwrap().phase,
            RequestPhase::Prefilling
        );

        // Add new request while others are active
        let req3 = create_test_request(5, 10, 0);
        let id3 = req3.id;
        scheduler.enqueue_request(req3);

        // Admit the new request
        let admitted = scheduler.admit_requests().unwrap();
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0], id3);

        // Form batches - should separate by phase
        let prefill_batch = scheduler.form_prefill_batch().unwrap();
        assert_eq!(prefill_batch.request_ids.len(), 2); // id2 and id3

        let decode_batch = scheduler.form_decode_batch().unwrap();
        assert_eq!(decode_batch.request_ids.len(), 1); // id1

        // Verify batch contents
        assert!(prefill_batch.request_ids.contains(&id2));
        assert!(prefill_batch.request_ids.contains(&id3));
        assert!(decode_batch.request_ids.contains(&id1));
    }

    #[test]
    fn test_schedule_iteration_memory_pressure() {
        // Create scheduler with limited memory
        // Use smaller model params and very limited memory to trigger pressure
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(
            5 * 1024 * 1024,
            16,
            8,
            8,
            64,
            2,
        )));
        let block_allocator = Arc::new(RwLock::new(BlockAllocator::new(500, 16)));
        let config = BatchConfig::default();
        let mut scheduler = ContinuousBatchScheduler::new(
            SchedulingPolicy::FCFS,
            config,
            memory_pool,
            block_allocator,
            32,
            (0, 0),
        );

        // Enqueue many requests with larger size to exceed memory
        let mut ids = Vec::new();
        for _ in 0..10 {
            let request = create_test_request(50, 100, 0); // Larger requests to trigger memory pressure
            ids.push(request.id);
            scheduler.enqueue_request(request);
        }

        // First admission should only admit what fits in memory
        let admitted1 = scheduler.admit_requests().unwrap();

        // Should admit at least one request
        assert!(!admitted1.is_empty(), "Should admit at least one request");

        // Should not admit all requests due to memory limit
        assert!(
            admitted1.len() < 10,
            "Should not admit all requests due to memory limit. Admitted: {}, Total blocks available: {}",
            admitted1.len(),
            scheduler.get_stats().total_blocks
        );

        let stats1 = scheduler.get_stats();
        assert!(
            stats1.queued_requests > 0,
            "Some requests should remain queued"
        );

        // Complete one request to free memory
        if let Some(id) = admitted1.first() {
            scheduler.complete_request(id).unwrap();
        }

        // Now we should be able to admit more
        let admitted2 = scheduler.admit_requests().unwrap();

        // After freeing memory, should be able to admit at least one more
        if stats1.queued_requests > 0 {
            assert!(
                !admitted2.is_empty(),
                "Should admit more requests after freeing memory"
            );
        }

        // Total admitted should still be less than total requests
        let total_admitted = admitted1.len() + admitted2.len() - 1; // -1 for completed
        assert!(total_admitted < 10, "Should not have admitted all requests");
    }
}
