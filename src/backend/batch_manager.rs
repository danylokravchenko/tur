//! Batch request manager for continuous batching
//!
//! This module provides infrastructure for managing multiple concurrent requests,
//! tracking their lifecycle, and forming batches for efficient execution.

use crate::{Result, TurError, models::kv_cache::BlockId};
use ahash::AHashMap;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, trace};
use uuid::Uuid;

/// Phase of request execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    /// Request is queued, waiting to be processed
    Queued,
    /// Request is in prefill phase (processing prompt)
    Prefilling,
    /// Request is in decode phase (generating tokens)
    Decoding,
    /// Request has completed successfully
    Completed,
    /// Request failed with an error
    Failed,
}

/// State of an active request
#[derive(Debug, Clone)]
pub struct RequestState {
    /// Unique request identifier
    pub id: Uuid,
    /// Original prompt text
    pub prompt: String,
    /// Tokenized prompt
    pub prompt_tokens: Vec<u32>,
    /// Generated tokens so far
    pub generated_tokens: Vec<u32>,
    /// Block table for paged KV cache
    pub block_table: Vec<BlockId>,
    /// Current execution phase
    pub phase: RequestPhase,
    /// Current position in sequence (for KV cache offset)
    pub position: usize,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// When the request arrived
    pub arrival_time: Instant,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Number of prompt tokens already written into the KV cache (chunked prefill cursor).
    /// Zero when chunking is disabled or before the first prefill chunk executes.
    pub prefill_offset: usize,
}

impl RequestState {
    /// Create a new request state
    pub fn new(
        id: Uuid,
        prompt: String,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: u32,
    ) -> Self {
        Self {
            id,
            prompt,
            prompt_tokens,
            generated_tokens: Vec::new(),
            block_table: Vec::new(),
            phase: RequestPhase::Queued,
            position: 0,
            arrival_time: Instant::now(),
            max_tokens,
            priority,
            prefill_offset: 0,
        }
    }

    /// Get total sequence length (prompt + generated)
    pub fn seq_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    /// Get all tokens (prompt + generated)
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(self.seq_len());
        tokens.extend_from_slice(&self.prompt_tokens);
        tokens.extend_from_slice(&self.generated_tokens);
        tokens
    }

    /// Check if request is finished
    pub fn is_finished(&self) -> bool {
        matches!(self.phase, RequestPhase::Completed | RequestPhase::Failed)
    }

    /// Check if request should stop (reached max tokens or finished)
    pub fn should_stop(&self) -> bool {
        self.is_finished() || self.generated_tokens.len() >= self.max_tokens
    }

    /// Get elapsed time since arrival
    pub fn elapsed(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }
}

/// Manages active and queued requests for continuous batching
pub struct BatchManager {
    /// Active requests being processed
    active_requests: AHashMap<Uuid, RequestState>,
    /// Queue of requests waiting to be admitted, maintained in priority order
    request_queue: VecDeque<RequestState>,
    /// Completed requests (stored temporarily for result retrieval)
    completed_requests: AHashMap<Uuid, Vec<u32>>,
    /// Failed requests: partial generated tokens + error message
    failed_requests: AHashMap<Uuid, (Vec<u32>, String)>,
    /// Maximum batch size for prefill
    max_prefill_batch: usize,
    /// Maximum batch size for decode
    max_decode_batch: usize,
}

impl BatchManager {
    /// Create a new batch manager
    pub fn new(max_prefill_batch: usize, max_decode_batch: usize) -> Self {
        debug!(
            max_prefill_batch,
            max_decode_batch, "Creating batch manager"
        );
        Self {
            active_requests: AHashMap::new(),
            request_queue: VecDeque::new(),
            completed_requests: AHashMap::new(),
            failed_requests: AHashMap::new(),
            max_prefill_batch,
            max_decode_batch,
        }
    }

    /// Enqueue a new request, inserting in priority order (higher priority closer to front).
    /// Requests with equal priority are ordered FIFO.
    pub fn enqueue_request(&mut self, request: RequestState) {
        trace!(
            request_id = %request.id,
            prompt_tokens = request.prompt_tokens.len(),
            max_tokens = request.max_tokens,
            priority = request.priority,
            queue_size = self.request_queue.len() + 1,
            "Enqueuing request"
        );
        // Insert before the first entry with strictly lower priority to preserve FIFO within a tier
        let insert_pos = self
            .request_queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(self.request_queue.len());
        self.request_queue.insert(insert_pos, request);
    }

    /// Get the next request from queue (without removing)
    pub fn peek_next_request(&self) -> Option<&RequestState> {
        self.request_queue.front()
    }

    /// Admit the highest-priority request from the queue to the active set.
    /// Returns the admitted request's `Uuid`; use `get_request` / `get_request_mut`
    /// to access or mutate it. Returning a `Uuid` instead of a cloned `RequestState`
    /// ensures there is exactly one copy of the request, preventing silent mutation loss.
    pub fn admit_request(&mut self) -> Option<Uuid> {
        if let Some(mut request) = self.request_queue.pop_front() {
            request.phase = RequestPhase::Prefilling;
            let id = request.id;
            debug!(
                request_id = %id,
                prompt_tokens = request.prompt_tokens.len(),
                active_requests = self.active_requests.len() + 1,
                queued_requests = self.request_queue.len(),
                "Admitted request to active set"
            );
            self.active_requests.insert(id, request);
            Some(id)
        } else {
            None
        }
    }

    /// Get a request by ID
    pub fn get_request(&self, id: &Uuid) -> Option<&RequestState> {
        self.active_requests.get(id)
    }

    /// Get a mutable request by ID
    pub fn get_request_mut(&mut self, id: &Uuid) -> Option<&mut RequestState> {
        self.active_requests.get_mut(id)
    }

    /// Form a prefill batch (up to max_prefill_batch requests), ordered by arrival time.
    pub fn form_prefill_batch(&self) -> Vec<Uuid> {
        let mut candidates: Vec<&RequestState> = self
            .active_requests
            .values()
            .filter(|r| r.phase == RequestPhase::Prefilling)
            .collect();
        candidates.sort_unstable_by_key(|r| r.arrival_time);
        let batch: Vec<Uuid> = candidates
            .into_iter()
            .take(self.max_prefill_batch)
            .map(|r| r.id)
            .collect();
        if !batch.is_empty() {
            trace!(
                batch_size = batch.len(),
                max_batch_size = self.max_prefill_batch,
                "Formed prefill batch"
            );
        }
        batch
    }

    /// Form a decode batch (up to max_decode_batch requests), ordered by arrival time.
    pub fn form_decode_batch(&self) -> Vec<Uuid> {
        let mut candidates: Vec<&RequestState> = self
            .active_requests
            .values()
            .filter(|r| r.phase == RequestPhase::Decoding)
            .collect();
        candidates.sort_unstable_by_key(|r| r.arrival_time);
        let batch: Vec<Uuid> = candidates
            .into_iter()
            .take(self.max_decode_batch)
            .map(|r| r.id)
            .collect();
        if !batch.is_empty() {
            trace!(
                batch_size = batch.len(),
                max_batch_size = self.max_decode_batch,
                "Formed decode batch"
            );
        }
        batch
    }

    /// Transition request from prefill to decode phase
    pub fn transition_to_decode(&mut self, id: &Uuid) -> Result<()> {
        if let Some(request) = self.active_requests.get_mut(id) {
            if request.phase != RequestPhase::Prefilling {
                return Err(TurError::InvalidPhase(format!(
                    "Request {} is not in prefilling phase (current: {:?})",
                    id, request.phase
                )));
            }
            request.phase = RequestPhase::Decoding;
            trace!(
                request_id = %id,
                generated_tokens = request.generated_tokens.len(),
                "Transitioned request to decode phase"
            );
            Ok(())
        } else {
            Err(TurError::RequestNotFound(id.to_string()))
        }
    }

    /// Mark request as completed
    pub fn complete_request(&mut self, id: &Uuid) -> Result<Vec<u32>> {
        if let Some(mut request) = self.active_requests.remove(id) {
            request.phase = RequestPhase::Completed;
            let generated = request.generated_tokens.clone();
            let elapsed = request.elapsed();
            debug!(
                request_id = %id,
                generated_tokens = generated.len(),
                elapsed_ms = elapsed.as_millis(),
                active_requests = self.active_requests.len(),
                "Completed request"
            );
            self.completed_requests.insert(*id, generated.clone());
            Ok(generated)
        } else {
            Err(TurError::RequestNotFound(id.to_string()))
        }
    }

    /// Mark request as failed, removing it from the active set.
    /// Partial generated tokens and the error reason are stored and retrievable
    /// via `get_failed_result` / `remove_failed_result`.
    pub fn fail_request(&mut self, id: &Uuid, error: String) -> Result<()> {
        if let Some(request) = self.active_requests.remove(id) {
            debug!(
                request_id = %id,
                error = %error,
                generated_tokens = request.generated_tokens.len(),
                "Request failed"
            );
            self.failed_requests
                .insert(*id, (request.generated_tokens, error));
            Ok(())
        } else {
            Err(TurError::RequestNotFound(id.to_string()))
        }
    }

    /// Get completed request result
    pub fn get_completed_result(&self, id: &Uuid) -> Option<&Vec<u32>> {
        self.completed_requests.get(id)
    }

    /// Remove completed request result
    pub fn remove_completed_result(&mut self, id: &Uuid) -> Option<Vec<u32>> {
        self.completed_requests.remove(id)
    }

    /// Get partial generated tokens and error reason for a failed request
    pub fn get_failed_result(&self, id: &Uuid) -> Option<(&Vec<u32>, &str)> {
        self.failed_requests
            .get(id)
            .map(|(tokens, err)| (tokens, err.as_str()))
    }

    /// Remove and return the failed request's partial tokens and error reason
    pub fn remove_failed_result(&mut self, id: &Uuid) -> Option<(Vec<u32>, String)> {
        self.failed_requests.remove(id)
    }

    /// Get number of active requests
    pub fn num_active_requests(&self) -> usize {
        self.active_requests.len()
    }

    /// Get number of queued requests
    pub fn num_queued_requests(&self) -> usize {
        self.request_queue.len()
    }

    /// Get number of requests in prefill phase
    pub fn num_prefill_requests(&self) -> usize {
        self.active_requests
            .values()
            .filter(|req| req.phase == RequestPhase::Prefilling)
            .count()
    }

    /// Get number of requests in decode phase
    pub fn num_decode_requests(&self) -> usize {
        self.active_requests
            .values()
            .filter(|req| req.phase == RequestPhase::Decoding)
            .count()
    }

    /// Clear all completed results (for cleanup)
    pub fn clear_completed_results(&mut self) {
        self.completed_requests.clear();
    }

    /// Get statistics in a single pass over active_requests
    pub fn stats(&self) -> BatchManagerStats {
        let mut prefill_requests = 0usize;
        let mut decode_requests = 0usize;
        for req in self.active_requests.values() {
            match req.phase {
                RequestPhase::Prefilling => prefill_requests += 1,
                RequestPhase::Decoding => decode_requests += 1,
                _ => {}
            }
        }
        BatchManagerStats {
            active_requests: self.active_requests.len(),
            queued_requests: self.request_queue.len(),
            prefill_requests,
            decode_requests,
            completed_requests: self.completed_requests.len(),
            failed_requests: self.failed_requests.len(),
        }
    }
}

/// Statistics for batch manager
#[derive(Debug, Clone)]
pub struct BatchManagerStats {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub prefill_requests: usize,
    pub decode_requests: usize,
    pub completed_requests: usize,
    pub failed_requests: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_state_creation() {
        let id = Uuid::new_v4();
        let request = RequestState::new(id, "Hello world".to_string(), vec![1, 2, 3], 10, 1);

        assert_eq!(request.id, id);
        assert_eq!(request.prompt, "Hello world");
        assert_eq!(request.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.priority, 1);
        assert_eq!(request.phase, RequestPhase::Queued);
        assert_eq!(request.seq_len(), 3);
        assert!(!request.is_finished());
    }

    #[test]
    fn test_request_state_tokens() {
        let mut request =
            RequestState::new(Uuid::new_v4(), "test".to_string(), vec![1, 2, 3], 10, 1);

        request.generated_tokens = vec![4, 5, 6];
        assert_eq!(request.seq_len(), 6);
        assert_eq!(request.all_tokens(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_batch_manager_enqueue() {
        let mut manager = BatchManager::new(4, 8);
        let request = RequestState::new(Uuid::new_v4(), "test".to_string(), vec![1, 2, 3], 10, 1);

        manager.enqueue_request(request.clone());
        assert_eq!(manager.num_queued_requests(), 1);
        assert_eq!(manager.num_active_requests(), 0);
    }

    #[test]
    fn test_batch_manager_enqueue_priority_order() {
        let mut manager = BatchManager::new(4, 8);

        let low = RequestState::new(Uuid::new_v4(), "low".to_string(), vec![1], 10, 1);
        let high = RequestState::new(Uuid::new_v4(), "high".to_string(), vec![2], 10, 5);
        let mid = RequestState::new(Uuid::new_v4(), "mid".to_string(), vec![3], 10, 3);

        manager.enqueue_request(low);
        manager.enqueue_request(high);
        manager.enqueue_request(mid);

        // Queue front should be the highest-priority request
        assert_eq!(manager.peek_next_request().unwrap().priority, 5);
        let id1 = manager.admit_request().unwrap();
        assert_eq!(manager.get_request(&id1).unwrap().priority, 5);
        let id2 = manager.admit_request().unwrap();
        assert_eq!(manager.get_request(&id2).unwrap().priority, 3);
        let id3 = manager.admit_request().unwrap();
        assert_eq!(manager.get_request(&id3).unwrap().priority, 1);
    }

    #[test]
    fn test_batch_manager_admit() {
        let mut manager = BatchManager::new(4, 8);
        let id = Uuid::new_v4();
        let request = RequestState::new(id, "test".to_string(), vec![1, 2, 3], 10, 1);

        manager.enqueue_request(request);
        let admitted_id = manager.admit_request().unwrap();

        assert_eq!(admitted_id, id);
        assert_eq!(
            manager.get_request(&admitted_id).unwrap().phase,
            RequestPhase::Prefilling
        );
        assert_eq!(manager.num_queued_requests(), 0);
        assert_eq!(manager.num_active_requests(), 1);
        assert_eq!(manager.num_prefill_requests(), 1);
    }

    #[test]
    fn test_batch_manager_phase_transition() {
        let mut manager = BatchManager::new(4, 8);
        let id = Uuid::new_v4();
        let request = RequestState::new(id, "test".to_string(), vec![1, 2, 3], 10, 1);

        manager.enqueue_request(request);
        manager.admit_request().unwrap();

        assert_eq!(manager.num_prefill_requests(), 1);
        assert_eq!(manager.num_decode_requests(), 0);

        manager.transition_to_decode(&id).unwrap();

        assert_eq!(manager.num_prefill_requests(), 0);
        assert_eq!(manager.num_decode_requests(), 1);
    }

    #[test]
    fn test_batch_manager_complete() {
        let mut manager = BatchManager::new(4, 8);
        let id = Uuid::new_v4();
        let mut request = RequestState::new(id, "test".to_string(), vec![1, 2, 3], 10, 1);
        request.generated_tokens = vec![4, 5, 6];

        manager.enqueue_request(request);
        manager.admit_request().unwrap();

        let tokens = manager.complete_request(&id).unwrap();
        assert_eq!(tokens, vec![4, 5, 6]);
        assert_eq!(manager.num_active_requests(), 0);
        assert_eq!(manager.completed_requests.len(), 1);
    }

    #[test]
    fn test_batch_manager_fail_request() {
        let mut manager = BatchManager::new(4, 8);
        let id = Uuid::new_v4();
        let mut request = RequestState::new(id, "test".to_string(), vec![1, 2, 3], 10, 1);
        request.generated_tokens = vec![4, 5];

        manager.enqueue_request(request);
        manager.admit_request().unwrap();
        assert_eq!(manager.num_active_requests(), 1);

        manager.fail_request(&id, "OOM".to_string()).unwrap();

        // Request must be removed from active set
        assert_eq!(manager.num_active_requests(), 0);

        // Partial tokens and error reason must be retrievable
        let (tokens, err) = manager.get_failed_result(&id).unwrap();
        assert_eq!(*tokens, vec![4, 5]);
        assert_eq!(err, "OOM");

        let (tokens, err) = manager.remove_failed_result(&id).unwrap();
        assert_eq!(tokens, vec![4, 5]);
        assert_eq!(err, "OOM");
        assert!(manager.get_failed_result(&id).is_none());
    }

    #[test]
    fn test_batch_manager_form_batches() {
        let mut manager = BatchManager::new(2, 4);

        // Add 3 prefill requests
        for i in 0..3 {
            let request =
                RequestState::new(Uuid::new_v4(), format!("test{}", i), vec![1, 2, 3], 10, 1);
            manager.enqueue_request(request);
            manager.admit_request().unwrap();
        }

        // Form prefill batch (should get 2, limited by max_prefill_batch)
        let prefill_batch = manager.form_prefill_batch();
        assert_eq!(prefill_batch.len(), 2);

        // Transition to decode
        for id in &prefill_batch {
            manager.transition_to_decode(id).unwrap();
        }

        // Form decode batch
        let decode_batch = manager.form_decode_batch();
        assert_eq!(decode_batch.len(), 2);
    }

    #[test]
    fn test_batch_manager_stats() {
        let mut manager = BatchManager::new(4, 8);

        // Add some requests
        for i in 0..3 {
            let request =
                RequestState::new(Uuid::new_v4(), format!("test{}", i), vec![1, 2, 3], 10, 1);
            manager.enqueue_request(request);
        }

        // Admit 2
        manager.admit_request().unwrap();
        let id2 = manager.admit_request().unwrap();

        // Transition one to decode
        manager.transition_to_decode(&id2).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.queued_requests, 1);
        assert_eq!(stats.active_requests, 2);
        assert_eq!(stats.prefill_requests, 1);
        assert_eq!(stats.decode_requests, 1);
        assert_eq!(stats.failed_requests, 0);
    }
}
