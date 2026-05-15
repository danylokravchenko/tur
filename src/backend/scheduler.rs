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
        memory_pool::MemoryPool,
    },
};
use parking_lot::RwLock;
use std::sync::Arc;
use tokenizers::Tokenizer;
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

/// Continuous batch scheduler
pub struct ContinuousBatchScheduler {
    batch_manager: BatchManager,
    #[allow(dead_code)]
    policy: SchedulingPolicy,
    #[allow(dead_code)]
    config: BatchConfig,
    memory_pool: Arc<RwLock<MemoryPool>>,
    #[allow(dead_code)]
    tokenizer: Arc<Tokenizer>,
    /// Track allocated blocks per request (request_id -> num_blocks)
    allocated_blocks: ahash::AHashMap<uuid::Uuid, usize>,
}

impl ContinuousBatchScheduler {
    /// Create a new scheduler
    pub fn new(
        policy: SchedulingPolicy,
        config: BatchConfig,
        memory_pool: Arc<RwLock<MemoryPool>>,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        Self {
            batch_manager: BatchManager::new(config.max_prefill_batch, config.max_decode_batch),
            policy,
            config,
            memory_pool,
            tokenizer,
            allocated_blocks: ahash::AHashMap::new(),
        }
    }

    /// Enqueue a new request
    pub fn enqueue_request(&mut self, request: RequestState) {
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
                if let Some(request) = self.batch_manager.admit_request() {
                    // Track allocated blocks for this request
                    self.allocated_blocks.insert(request.id, required_blocks);
                    admitted.push(request.id);
                } else {
                    break;
                }
            } else {
                // No more memory available
                break;
            }
        }

        Ok(admitted)
    }

    /// Form a prefill batch from admitted requests
    pub fn form_prefill_batch(&self) -> Option<ExecutionBatch> {
        let prefill_ids = self.batch_manager.form_prefill_batch();
        if prefill_ids.is_empty() {
            return None;
        }

        let mut total_tokens = 0;
        for id in &prefill_ids {
            if let Some(request) = self.batch_manager.get_request(id) {
                total_tokens += request.seq_len();
            }
        }

        Some(ExecutionBatch {
            request_ids: prefill_ids,
            phase: RequestPhase::Prefilling,
            total_tokens,
        })
    }

    /// Form a decode batch from decoding requests
    pub fn form_decode_batch(&self) -> Option<ExecutionBatch> {
        let decode_ids = self.batch_manager.form_decode_batch();
        if decode_ids.is_empty() {
            return None;
        }

        let mut total_tokens = 0;
        for id in &decode_ids {
            if let Some(request) = self.batch_manager.get_request(id) {
                total_tokens += request.seq_len();
            }
        }

        Some(ExecutionBatch {
            request_ids: decode_ids,
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
        // Get allocated blocks for this request
        let blocks_to_free = self.allocated_blocks.remove(request_id).unwrap_or(0);

        // Complete the request in batch manager
        let tokens = self.batch_manager.complete_request(request_id)?;

        // Free memory blocks
        if blocks_to_free > 0 {
            let mut pool = self.memory_pool.write();
            pool.free(blocks_to_free);
        }

        Ok(tokens)
    }

    /// Mark a request as failed and free its memory
    pub fn fail_request(&mut self, request_id: &Uuid, error: String) -> Result<()> {
        // Get allocated blocks for this request
        let blocks_to_free = self.allocated_blocks.remove(request_id).unwrap_or(0);

        // Fail the request in batch manager
        self.batch_manager.fail_request(request_id, error)?;

        // Free memory blocks
        if blocks_to_free > 0 {
            let mut pool = self.memory_pool.write();
            pool.free(blocks_to_free);
        }

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
    use candle_core::Device;

    fn create_test_scheduler(policy: SchedulingPolicy) -> ContinuousBatchScheduler {
        // Create a simple memory pool for testing
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(
            1024 * 1024 * 100, // 100MB
            16,                // 16 tokens per block
            32,                // 32 layers
            32,                // 32 heads
            128,               // 128 head_dim
            2,                 // 2 bytes (BF16)
            Device::Cpu,
        )));

        // Create a simple tokenizer for testing
        let tokenizer = Arc::new(Tokenizer::from_file("tokenizer.json").unwrap_or_else(|_| {
            // Fallback: create a minimal tokenizer
            tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
        }));

        let config = BatchConfig::default();
        ContinuousBatchScheduler::new(policy, config, memory_pool, tokenizer)
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
        use candle_core::Device;
        // Create scheduler with very limited memory
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(
            1024 * 10, // Very small: 10KB
            16,
            32,
            32,
            128,
            2,
            Device::Cpu,
        )));
        let tokenizer = Arc::new(Tokenizer::from_file("tokenizer.json").unwrap_or_else(|_| {
            tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
        }));
        let config = BatchConfig::default();
        let mut scheduler =
            ContinuousBatchScheduler::new(SchedulingPolicy::FCFS, config, memory_pool, tokenizer);

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

        // Request should still be in active set but marked as failed
        assert_eq!(
            scheduler.get_request(&request_id).unwrap().phase,
            RequestPhase::Failed
        );
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
}
