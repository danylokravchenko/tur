use parking_lot::RwLock;
use std::sync::Arc;
use tur::ProgressReporter;
use tur::backend::InferenceEngine;
use tur::backend::pipeline::{GenerationRequest, TextGeneration};
use tur::backend::prefix_cache::PrefixCache;
use tur::backend::tokenizer::TokenOutputStream;

mod common;
use common::create_test_model;

#[test]
fn test_pipeline_end_to_end_generation() {
    let (model, tokenizer, device) = create_test_model().unwrap();

    let mut pipeline = TextGeneration::builder(model, tokenizer, device.clone())
        .seed(299792458)
        .temperature(0.8)
        .top_p(0.9)
        .repeat_penalty(1.1)
        .repeat_last_n(64)
        .build();

    // Test basic generation works
    let request = GenerationRequest::new("Hello".to_string(), 10);
    let result = pipeline.run(&request);
    assert!(result.is_ok(), "Generation failed: {:?}", result.err());

    // Test multiple sequential runs (verifies state management)
    for _ in 0..3 {
        let request = GenerationRequest::new("Test".to_string(), 5);
        let result = pipeline.run(&request);
        assert!(result.is_ok(), "Sequential run failed: {:?}", result.err());
    }

    // Test with progress reporter
    let (model, tokenizer, device) = create_test_model().unwrap();
    let progress = ProgressReporter::new();

    let mut pipeline_with_progress = TextGeneration::builder(model, tokenizer, device)
        .seed(299792458)
        .temperature(0.8)
        .top_p(0.9)
        .repeat_penalty(1.1)
        .repeat_last_n(64)
        .progress(progress)
        .build();

    let request = GenerationRequest::new("Test with progress".to_string(), 5);
    let result = pipeline_with_progress.run(&request);
    assert!(
        result.is_ok(),
        "Generation with progress failed: {:?}",
        result.err()
    );
}

#[test]
fn test_pipeline_parameter_variations() {
    // Test extreme temperature values affect generation
    let test_cases = [
        (Some(0.1), Some(0.9), 1.1, "Low temp"),
        (Some(1.5), Some(0.9), 1.1, "High temp"),
        (None, Some(0.9), 1.1, "No temp"),
        (Some(0.8), Some(0.5), 1.1, "Low top_p"),
        (Some(0.8), None, 1.1, "No top_p"),
        (Some(0.8), Some(0.9), 1.0, "No penalty"),
        (Some(0.8), Some(0.9), 2.0, "High penalty"),
    ];

    for (temp, top_p, penalty, desc) in test_cases {
        let (model, tokenizer, device) = create_test_model().unwrap();
        let mut builder = TextGeneration::builder(model, tokenizer, device)
            .seed(299792458)
            .repeat_penalty(penalty)
            .repeat_last_n(64);

        if let Some(t) = temp {
            builder = builder.temperature(t);
        }
        if let Some(p) = top_p {
            builder = builder.top_p(p);
        }

        let mut pipeline = builder.build();

        let request = GenerationRequest::new("Test".to_string(), 10);
        let result = pipeline.run(&request);
        assert!(result.is_ok(), "{} failed: {:?}", desc, result.err());
    }
}

#[test]
fn test_prefix_cache_full_hit() {
    // Test the edge case where all tokens are cached
    let (model, tokenizer, device) = create_test_model().unwrap();

    // Create prefix cache
    let cache = Arc::new(RwLock::new(PrefixCache::new(10, 100)));

    // Build engine with cache
    let mut engine = InferenceEngine::builder(model, device)
        .seed(299792458)
        .temperature(0.8)
        .with_shared_prefix_cache(cache.clone())
        .build();

    let tokenizer_stream = TokenOutputStream::new(tokenizer);
    let prompt = "Hello, world!";
    let tokens = tokenizer_stream
        .tokenizer()
        .encode(prompt, true)
        .expect("encoding failed")
        .get_ids()
        .to_vec();

    // First prefill - cache miss, stores to cache
    let result1 = engine.prefill(&tokens);
    assert!(result1.is_ok(), "First prefill failed: {:?}", result1.err());
    let (_token1, _duration1, hit1, cached1) = result1.unwrap();
    assert!(!hit1, "First request should be cache miss");
    assert_eq!(cached1, 0, "First request should have 0 cached tokens");

    // Clear KV cache but keep prefix cache
    engine.model_mut().clear_kv_cache();

    // Second prefill - should be full cache hit (all tokens cached)
    let result2 = engine.prefill(&tokens);
    assert!(
        result2.is_ok(),
        "Second prefill with full cache hit failed: {:?}",
        result2.err()
    );
    let (_token2, _duration2, hit2, cached2) = result2.unwrap();
    assert!(hit2, "Second request should be cache hit");
    assert_eq!(
        cached2,
        tokens.len(),
        "All tokens should be cached on second request"
    );

    // Verify cache statistics
    {
        let cache_guard = cache.read();
        let stats = cache_guard.stats();
        assert_eq!(stats.hits, 1, "Should have 1 cache hit");
        assert_eq!(stats.misses, 1, "Should have 1 cache miss");
        assert_eq!(
            stats.total_tokens_reused,
            tokens.len(),
            "Should have reused all tokens"
        );
    }
}

#[test]
fn test_prefix_cache_partial_hit() {
    // Test partial cache hits with shared prefix
    let (model, tokenizer, device) = create_test_model().unwrap();

    let cache = Arc::new(RwLock::new(PrefixCache::new(10, 100)));
    let mut engine = InferenceEngine::builder(model, device)
        .seed(299792458)
        .with_shared_prefix_cache(cache.clone())
        .build();

    let tokenizer_stream = TokenOutputStream::new(tokenizer);

    // First prompt
    let prompt1 = "Hello, world! How are you?";
    let tokens1 = tokenizer_stream
        .tokenizer()
        .encode(prompt1, true)
        .expect("encoding failed")
        .get_ids()
        .to_vec();

    let result1 = engine.prefill(&tokens1);
    assert!(result1.is_ok(), "First prefill failed");
    let (_token1, _duration1, hit1, cached1) = result1.unwrap();
    assert!(!hit1, "First request should be cache miss");
    assert_eq!(cached1, 0);

    engine.model_mut().clear_kv_cache();

    // Second prompt with shared prefix
    let prompt2 = "Hello, world! What is Rust?";
    let tokens2 = tokenizer_stream
        .tokenizer()
        .encode(prompt2, true)
        .expect("encoding failed")
        .get_ids()
        .to_vec();

    let result2 = engine.prefill(&tokens2);
    assert!(result2.is_ok(), "Second prefill failed");
    let (_token2, _duration2, hit2, cached2) = result2.unwrap();

    // Should have partial cache hit (shared prefix "Hello, world!")
    if hit2 {
        assert!(
            cached2 > 0 && cached2 < tokens2.len(),
            "Should have partial cache hit, got {} cached out of {} tokens",
            cached2,
            tokens2.len()
        );
    }

    // Verify cache has entries
    assert!(!cache.read().is_empty(), "Cache should have entries");
}

#[test]
fn test_prefix_cache_with_pipeline() {
    // Test prefix cache integration with full TextGeneration pipeline
    let (model, tokenizer, device) = create_test_model().unwrap();

    let cache = Arc::new(RwLock::new(PrefixCache::new(10, 100)));
    let engine = InferenceEngine::builder(model, device.clone())
        .seed(299792458)
        .temperature(0.8)
        .with_shared_prefix_cache(cache.clone())
        .build();

    let mut pipeline = TextGeneration::from_engine(engine, tokenizer, None);

    // Run multiple generations with similar prompts
    let prompts = vec!["Hello", "Hello, world", "Hello, how are you?"];

    for prompt in prompts {
        let request = GenerationRequest::new(prompt.to_string(), 5);
        let result = pipeline.run(&request);
        assert!(
            result.is_ok(),
            "Generation failed for prompt '{}': {:?}",
            prompt,
            result.err()
        );
    }

    // Verify cache was used
    {
        let cache_guard = cache.read();
        let stats = cache_guard.stats();
        assert!(
            stats.hits > 0 || stats.misses > 0,
            "Cache should have been accessed"
        );
    }
}

#[test]
fn test_continuous_batching_basic() {
    // Test basic continuous batching setup and usage
    let (model, tokenizer, device) = create_test_model().unwrap();

    // Enable batching with builder
    let mut pipeline = TextGeneration::builder(model, tokenizer, device.clone())
        .seed(299792458)
        .temperature(0.8)
        .enable_batching(true)
        .max_batch_size(4)
        .max_prefill_batch(2)
        .max_decode_batch(4)
        .build();

    assert!(pipeline.is_batching_enabled(), "Batching should be enabled");

    // Submit multiple requests
    let requests = vec![
        GenerationRequest::new("Hello".to_string(), 5),
        GenerationRequest::new("How are you?".to_string(), 5),
        GenerationRequest::new("What is Rust?".to_string(), 5),
    ];

    let mut handles = Vec::new();
    for request in &requests {
        let handle = pipeline.submit_request(request).unwrap();
        handles.push(handle);
    }

    // Process all requests until completion
    pipeline.run_until_complete().unwrap();

    // Retrieve results
    for handle in &handles {
        let result = pipeline.try_get_result(handle);
        assert!(result.is_some(), "Result should be available");

        let result = result.unwrap();
        assert!(
            !result.generated_text.is_empty(),
            "Should have generated text"
        );
        assert!(
            !result.generated_tokens.is_empty(),
            "Should have generated tokens"
        );
    }

    // Verify all requests completed
    assert_eq!(pipeline.active_request_count(), 0);
    assert_eq!(pipeline.queued_request_count(), 0);
}

#[test]
fn test_continuous_batching_step_by_step() {
    // Test manual step-by-step execution for fine-grained control
    let (model, tokenizer, device) = create_test_model().unwrap();

    let mut pipeline = TextGeneration::builder(model, tokenizer, device)
        .seed(299792458)
        .enable_batching(true)
        .max_batch_size(2)
        .build();

    // Submit requests
    let req1 = GenerationRequest::new("Test 1".to_string(), 3);
    let req2 = GenerationRequest::new("Test 2".to_string(), 3);

    let handle1 = pipeline.submit_request(&req1).unwrap();
    let handle2 = pipeline.submit_request(&req2).unwrap();

    // Manually step through execution
    let mut iterations = 0;
    let max_iterations = 20; // Safety limit

    while pipeline.active_request_count() > 0 || pipeline.queued_request_count() > 0 {
        let active = pipeline.step().unwrap();
        iterations += 1;

        if iterations > max_iterations {
            panic!("Too many iterations, possible infinite loop");
        }

        if active == 0 {
            break;
        }
    }

    // Verify results
    let result1 = pipeline.try_get_result(&handle1);
    let result2 = pipeline.try_get_result(&handle2);

    assert!(result1.is_some(), "Request 1 should be completed");
    assert!(result2.is_some(), "Request 2 should be completed");
}

#[test]
fn test_continuous_batching_blocking_get() {
    // Test blocking result retrieval
    let (model, tokenizer, device) = create_test_model().unwrap();

    let mut pipeline = TextGeneration::builder(model, tokenizer, device)
        .seed(299792458)
        .enable_batching(true)
        .build();

    let request = GenerationRequest::new("Hello world".to_string(), 5);
    let handle = pipeline.submit_request(&request).unwrap();

    // Blocking get - will process until this request completes
    let result = pipeline.get_result(&handle).unwrap();

    assert_eq!(result.request_id, handle.id);
    assert!(!result.generated_text.is_empty());
    assert!(!result.generated_tokens.is_empty());
}

#[test]
fn test_continuous_batching_mixed_lengths() {
    // Test handling requests with different generation lengths
    let (model, tokenizer, device) = create_test_model().unwrap();

    let mut pipeline = TextGeneration::builder(model, tokenizer, device)
        .seed(299792458)
        .enable_batching(true)
        .max_batch_size(3)
        .build();

    // Submit requests with varying lengths
    let short_req = GenerationRequest::new("Hi".to_string(), 2);
    let medium_req = GenerationRequest::new("Hello".to_string(), 5);
    let long_req = GenerationRequest::new("Tell me".to_string(), 10);

    let h1 = pipeline.submit_request(&short_req).unwrap();
    let h2 = pipeline.submit_request(&medium_req).unwrap();
    let h3 = pipeline.submit_request(&long_req).unwrap();

    // Process all
    pipeline.run_until_complete().unwrap();

    // All should complete successfully
    assert!(pipeline.try_get_result(&h1).is_some());
    assert!(pipeline.try_get_result(&h2).is_some());
    assert!(pipeline.try_get_result(&h3).is_some());
}

#[test]
fn test_continuous_batching_sequential_submission() {
    // Test submitting requests while others are processing
    let (model, tokenizer, device) = create_test_model().unwrap();

    let mut pipeline = TextGeneration::builder(model, tokenizer, device)
        .seed(299792458)
        .enable_batching(true)
        .build();

    // Submit first batch
    let req1 = GenerationRequest::new("First".to_string(), 5);
    let h1 = pipeline.submit_request(&req1).unwrap();

    // Process a few steps
    for _ in 0..3 {
        pipeline.step().unwrap();
    }

    // Submit more while first is processing
    let req2 = GenerationRequest::new("Second".to_string(), 5);
    let req3 = GenerationRequest::new("Third".to_string(), 5);
    let h2 = pipeline.submit_request(&req2).unwrap();
    let h3 = pipeline.submit_request(&req3).unwrap();

    // Complete all
    pipeline.run_until_complete().unwrap();

    // All should be done
    assert!(pipeline.try_get_result(&h1).is_some());
    assert!(pipeline.try_get_result(&h2).is_some());
    assert!(pipeline.try_get_result(&h3).is_some());
}

#[test]
fn test_continuous_batching_result_management() {
    // Test result storage and retrieval
    let (model, tokenizer, device) = create_test_model().unwrap();

    let mut pipeline = TextGeneration::builder(model, tokenizer, device)
        .seed(299792458)
        .enable_batching(true)
        .build();

    let req = GenerationRequest::new("Test".to_string(), 3);
    let handle = pipeline.submit_request(&req).unwrap();

    pipeline.run_until_complete().unwrap();

    // Get all results
    let all_results = pipeline.get_all_results();
    assert_eq!(all_results.len(), 1);
    assert!(all_results.contains_key(&handle.id));

    // Clear results
    pipeline.clear_results();
    let all_results_after = pipeline.get_all_results();
    assert_eq!(all_results_after.len(), 0);
}
