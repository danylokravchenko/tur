use parking_lot::RwLock;
use std::sync::Arc;
use tur::{
    backend::{
        engine::InferenceEngine,
        prefix_cache::{PrefixCache, SharedPrefixCache},
    },
    models::kv_cache::{BlockAllocator, PagedKvCache},
};
use uuid::Uuid;

mod common;
use common::create_test_factory;

/// Create per-request paged KV caches (one `PagedKvCache` per layer).
///
/// Qwen3-0.6B has 28 transformer layers; the paged cache depth must match.
const QWEN3_NUM_LAYERS: usize = 28;

fn make_paged_caches(
    allocator: &Arc<RwLock<BlockAllocator>>,
    num_requests: usize,
) -> Vec<Vec<PagedKvCache>> {
    (0..num_requests)
        .map(|_| {
            (0..QWEN3_NUM_LAYERS)
                .map(|_| PagedKvCache::new(allocator.clone(), 2))
                .collect()
        })
        .collect()
}

#[test]
fn test_inference_engine_prefill_batch() {
    let (factory, device, _) = create_test_factory();
    let (mut engine, _tokenizer) = InferenceEngine::builder(&factory, device).build().unwrap();

    // Create batch of 3 requests with different prompts
    let batch_tokens = vec![
        (Uuid::new_v4(), vec![1u32, 2, 3, 4]),
        (Uuid::new_v4(), vec![5u32, 6, 7]),
        (Uuid::new_v4(), vec![8u32, 9, 10, 11, 12]),
    ];

    let result = engine.prefill_batch(&batch_tokens, None);
    assert!(result.is_ok(), "Batched prefill failed: {:?}", result.err());

    let results = result.unwrap();
    assert_eq!(results.len(), 3, "Should return 3 results");

    // Verify each request got a token
    for (id, _token) in &results {
        assert!(
            batch_tokens.iter().any(|(req_id, _)| req_id == id),
            "Result ID should match input"
        );
        // Token can be any value including 0 (valid token ID)
    }
}

#[test]
fn test_inference_engine_decode_batch() {
    let (factory, device, _) = create_test_factory();
    let (mut engine, _tokenizer) = InferenceEngine::builder(&factory, device).build().unwrap();

    // First do prefill to populate KV cache
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    let prefill_batch = vec![
        (id1, vec![1u32, 2, 3]),
        (id2, vec![4u32, 5, 6, 7]),
        (id3, vec![8u32, 9]),
    ];

    let prefill_results = engine.prefill_batch(&prefill_batch, None).unwrap();

    // Now do decode with variable positions
    let mut decode_batch = Vec::new();
    for ((id, mut tokens), (_, next_token)) in prefill_batch.into_iter().zip(prefill_results.iter())
    {
        tokens.push(*next_token);
        let position = tokens.len() - 1;
        decode_batch.push((id, tokens, position));
    }

    let result = engine.decode_batch(&decode_batch, None);
    assert!(result.is_ok(), "Batched decode failed: {:?}", result.err());

    let results = result.unwrap();
    assert_eq!(results.len(), 3, "Should return 3 results");

    // Verify each request got a token
    for (id, _token) in &results {
        assert!(
            decode_batch.iter().any(|(req_id, _, _)| req_id == id),
            "Result ID should match input"
        );
        // Token can be any value including 0 (valid token ID)
    }
}

#[test]
fn test_inference_engine_batch_consistency() {
    let (factory, device, _) = create_test_factory();

    // Single request
    let tokens = vec![1u32, 2, 3, 4, 5];
    let (mut engine1, _tokenizer) = InferenceEngine::builder(&factory, device.clone())
        .build()
        .unwrap();
    let (single_token, _, _, _) = engine1.prefill(&tokens).unwrap();

    // Same request in batch
    let id = Uuid::new_v4();
    let (mut engine2, _tokenizer) = InferenceEngine::builder(&factory, device.clone())
        .build()
        .unwrap();
    let batch_results = engine2
        .prefill_batch(&[(id, tokens.clone())], None)
        .unwrap();

    assert_eq!(batch_results.len(), 1);
    let (result_id, batch_token) = batch_results[0];
    assert_eq!(result_id, id);

    // Tokens should be the same (deterministic with same seed)
    assert_eq!(
        single_token, batch_token,
        "Single and batched prefill should produce same token"
    );
}

#[test]
fn test_inference_engine_empty_batch() {
    let (factory, device, _) = create_test_factory();
    let (mut engine, _tokenizer) = InferenceEngine::builder(&factory, device).build().unwrap();

    // Empty batch should return empty results
    let result = engine.prefill_batch(&[], None);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);

    let result = engine.decode_batch(&[], None);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_inference_engine_large_batch() {
    let (factory, device, _) = create_test_factory();
    let (mut engine, _) = InferenceEngine::builder(&factory, device).build().unwrap();

    // Create larger batch (8 requests)
    let batch_size = 8;
    let batch_tokens: Vec<_> = (0..batch_size)
        .map(|i| {
            let tokens: Vec<u32> = (1..=5).map(|j| (i * 10 + j) as u32).collect();
            (Uuid::new_v4(), tokens)
        })
        .collect();

    let result = engine.prefill_batch(&batch_tokens, None);
    assert!(
        result.is_ok(),
        "Large batch prefill failed: {:?}",
        result.err()
    );

    let results = result.unwrap();
    assert_eq!(
        results.len(),
        batch_size,
        "Should return {} results",
        batch_size
    );
}

#[test]
fn test_inference_engine_variable_length_batch() {
    let (factory, device, _) = create_test_factory();
    let (mut engine, _) = InferenceEngine::builder(&factory, device).build().unwrap();

    // Create batch with very different sequence lengths
    let batch_tokens = vec![
        (Uuid::new_v4(), vec![1u32]),             // Length 1
        (Uuid::new_v4(), vec![2u32, 3, 4, 5, 6]), // Length 5
        (Uuid::new_v4(), vec![7u32, 8]),          // Length 2
        (
            Uuid::new_v4(),
            vec![9u32, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ), // Length 10
    ];

    let result = engine.prefill_batch(&batch_tokens, None);
    assert!(
        result.is_ok(),
        "Variable length batch failed: {:?}",
        result.err()
    );

    let results = result.unwrap();
    assert_eq!(results.len(), 4, "Should return 4 results");
}

// ---------------------------------------------------------------------------
// Prefix cache correctness tests
// ---------------------------------------------------------------------------

/// Single-request path: the second `prefill` with the same tokens must restore
/// KV state from the prefix cache and produce an identical next token.
#[test]
fn test_prefix_cache_single_request_correctness() {
    let (factory, device, _) = create_test_factory();
    let (mut engine, _) = InferenceEngine::builder(&factory, device)
        .with_prefix_cache(10, 512)
        .build()
        .unwrap();

    let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

    // Cold run: the cache is empty, no KV state is reused.
    let (token_cold, _, hit_cold, cached_cold) = engine.prefill(&tokens).unwrap();
    assert!(!hit_cold, "first prefill must be a cache miss");
    assert_eq!(
        cached_cold, 0,
        "no tokens should be cached on the first run"
    );

    // Warm run: the engine restores the stored K/V state and re-processes only
    // the final token, so the output must be identical.
    let (token_warm, _, hit_warm, cached_warm) = engine.prefill(&tokens).unwrap();
    assert!(hit_warm, "second prefill must be a cache hit");
    assert!(
        cached_warm > 0,
        "some tokens must be served from cache on second run"
    );

    assert_eq!(
        token_cold, token_warm,
        "prefix cache must reproduce the same next token as a cold run",
    );
}

/// Paged-batch path: the second `prefill_batch` with the same tokens must hit
/// the prefix cache and produce an identical next token for each request.
#[test]
fn test_prefix_cache_paged_batch_correctness() {
    let (factory, device, _) = create_test_factory();
    let shared_cache: SharedPrefixCache = Arc::new(RwLock::new(PrefixCache::new(10, 512)));
    let (mut engine, _) = InferenceEngine::builder(&factory, device)
        .with_shared_prefix_cache(shared_cache.clone())
        .build()
        .unwrap();

    // Use a generous allocator: 256 blocks × 16 tokens/block is enough for two
    // concurrent requests across all 28 layers.
    let allocator = Arc::new(RwLock::new(BlockAllocator::new(256, 16)));

    let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let id = Uuid::new_v4();

    // First paged prefill: cold run.  The engine stores the K/V state in the cache.
    let mut caches1 = make_paged_caches(&allocator, 1);
    let results1 = engine
        .prefill_batch(&[(id, tokens.clone())], Some(&mut caches1))
        .unwrap();
    let token_cold = results1[0].1;

    assert_eq!(
        shared_cache.read().len(),
        1,
        "prefix cache must hold one entry after the first paged prefill",
    );

    // Second paged prefill with fresh per-request caches: must hit the prefix cache.
    let id2 = Uuid::new_v4();
    let mut caches2 = make_paged_caches(&allocator, 1);
    let results2 = engine
        .prefill_batch(&[(id2, tokens.clone())], Some(&mut caches2))
        .unwrap();
    let token_warm = results2[0].1;

    assert!(
        shared_cache.read().stats().hits > 0,
        "prefix cache must record a hit on the second paged prefill",
    );
    assert_eq!(
        token_cold, token_warm,
        "paged prefill with prefix cache must reproduce the same next token as a cold run",
    );
}

/// Partial-prefix hit: after caching a short prompt, a longer prompt that shares
/// that prefix must reuse the cached K/V tail and produce a consistent token.
#[test]
fn test_prefix_cache_paged_partial_prefix_hit() {
    let (factory, device, _) = create_test_factory();
    let shared_cache: SharedPrefixCache = Arc::new(RwLock::new(PrefixCache::new(10, 512)));
    let (mut engine, _) = InferenceEngine::builder(&factory, device)
        .with_shared_prefix_cache(shared_cache.clone())
        .build()
        .unwrap();

    let allocator = Arc::new(RwLock::new(BlockAllocator::new(256, 16)));

    // Warm the cache with a short prefix [1, 2, 3, 4].
    let short_tokens = vec![1u32, 2, 3, 4];
    let id1 = Uuid::new_v4();
    let mut caches_warm = make_paged_caches(&allocator, 1);
    engine
        .prefill_batch(&[(id1, short_tokens)], Some(&mut caches_warm))
        .unwrap();

    // Cold run with a longer prompt that extends the cached prefix.
    // (The first 4 tokens come from the cache; tokens 5-8 are computed fresh.)
    let long_tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let id2 = Uuid::new_v4();
    let mut caches_cold = make_paged_caches(&allocator, 1);
    let results_cold = engine
        .prefill_batch(&[(id2, long_tokens.clone())], Some(&mut caches_cold))
        .unwrap();

    // Reset stats to isolate the hit below from the one above.
    shared_cache.write().stats_mut().reset();

    // Second run with the full long prompt: must hit the (now-stored) long entry.
    let id3 = Uuid::new_v4();
    let mut caches_hit = make_paged_caches(&allocator, 1);
    let results_hit = engine
        .prefill_batch(&[(id3, long_tokens)], Some(&mut caches_hit))
        .unwrap();

    assert!(
        shared_cache.read().stats().hits > 0,
        "second long-prompt prefill must hit the prefix cache",
    );
    assert_eq!(
        results_cold[0].1, results_hit[0].1,
        "partial-prefix cache hit must reproduce the same next token as the cold run",
    );
}
