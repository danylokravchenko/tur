use tur::backend::engine::InferenceEngine;

use uuid::Uuid;

mod common;
use common::create_test_model;

#[test]
fn test_inference_engine_prefill_batch() {
    let (model, _tokenizer, device) = create_test_model().unwrap();
    let mut engine = InferenceEngine::builder(model, device).build();

    // Create batch of 3 requests with different prompts
    let batch_tokens = vec![
        (Uuid::new_v4(), vec![1u32, 2, 3, 4]),
        (Uuid::new_v4(), vec![5u32, 6, 7]),
        (Uuid::new_v4(), vec![8u32, 9, 10, 11, 12]),
    ];

    let result = engine.prefill_batch(&batch_tokens);
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
    let (model, _tokenizer, device) = create_test_model().unwrap();
    let mut engine = InferenceEngine::builder(model, device).build();

    // First do prefill to populate KV cache
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    let prefill_batch = vec![
        (id1, vec![1u32, 2, 3]),
        (id2, vec![4u32, 5, 6, 7]),
        (id3, vec![8u32, 9]),
    ];

    let prefill_results = engine.prefill_batch(&prefill_batch).unwrap();

    // Now do decode with variable positions
    let mut decode_batch = Vec::new();
    for ((id, mut tokens), (_, next_token)) in prefill_batch.into_iter().zip(prefill_results.iter())
    {
        tokens.push(*next_token);
        let position = tokens.len() - 1;
        decode_batch.push((id, tokens, position));
    }

    let result = engine.decode_batch(&decode_batch);
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
    let (model1, _tokenizer, device) = create_test_model().unwrap();

    // Single request
    let tokens = vec![1u32, 2, 3, 4, 5];
    let mut engine1 = InferenceEngine::builder(model1, device.clone()).build();
    let (single_token, _, _, _) = engine1.prefill(&tokens).unwrap();

    // Same request in batch
    let id = Uuid::new_v4();
    let (model2, _tokenizer, device) = create_test_model().unwrap();
    let mut engine2 = InferenceEngine::builder(model2, device).build();
    let batch_results = engine2.prefill_batch(&[(id, tokens.clone())]).unwrap();

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
    let (model, _tokenizer, device) = create_test_model().unwrap();
    let mut engine = InferenceEngine::builder(model, device).build();

    // Empty batch should return empty results
    let result = engine.prefill_batch(&[]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);

    let result = engine.decode_batch(&[]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_inference_engine_large_batch() {
    let (model, _tokenizer, device) = create_test_model().unwrap();
    let mut engine = InferenceEngine::builder(model, device).build();

    // Create larger batch (8 requests)
    let batch_size = 8;
    let batch_tokens: Vec<_> = (0..batch_size)
        .map(|i| {
            let tokens: Vec<u32> = (1..=5).map(|j| (i * 10 + j) as u32).collect();
            (Uuid::new_v4(), tokens)
        })
        .collect();

    let result = engine.prefill_batch(&batch_tokens);
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
    let (model, _tokenizer, device) = create_test_model().unwrap();
    let mut engine = InferenceEngine::builder(model, device).build();

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

    let result = engine.prefill_batch(&batch_tokens);
    assert!(
        result.is_ok(),
        "Variable length batch failed: {:?}",
        result.err()
    );

    let results = result.unwrap();
    assert_eq!(results.len(), 4, "Should return 4 results");
}
