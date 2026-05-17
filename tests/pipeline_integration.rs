use candle_core::Device;
use parking_lot::RwLock;
use std::sync::{Arc, Mutex};
use tur::ProgressReporter;
use tur::backend::InferenceEngine;
use tur::backend::guidance::TopLevelGrammar;
use tur::backend::pipeline::{GenerationRequest, TextGeneration};
use tur::backend::prefix_cache::PrefixCache;
use tur::backend::tokenizer::TokenOutputStream;
use tur::backend::tools::ToolDefinition;

mod common;
use common::create_test_factory;

/// Build a `ParserFactory` for guided-generation tests.
///
/// Loads the tokenizer via the engine builder (which also loads the model weights).
/// Callers should pass the same `factory` to the pipeline builder afterwards; the
/// model weights will be loaded a second time from the local HuggingFace cache,
/// which is fast.
fn build_guidance_factory(
    factory: &tur::ModelFactory<tur::models::Qwen35ModelForCausalLM>,
    device: Device,
) -> Arc<tur::backend::guidance::ParserFactory> {
    let (_, tokenizer) = InferenceEngine::builder(factory, device)
        .build()
        .expect("Failed to build engine to extract tokenizer");
    tur::backend::guidance::build_llg_factory(tokenizer, None)
        .expect("Failed to build guidance factory")
}

#[test]
fn test_pipeline_end_to_end_generation() {
    let (factory, device, _) = create_test_factory();

    let mut pipeline = TextGeneration::builder(&factory, device.clone())
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
    let progress = ProgressReporter::new();
    let mut pipeline_with_progress = TextGeneration::builder(&factory, device)
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
    let stats = result.unwrap();
    assert!(
        stats.tool_calls.is_empty(),
        "tool_calls must be empty for requests without tools"
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

    let (factory, device, _) = create_test_factory();

    for (temp, top_p, penalty, desc) in test_cases {
        let mut builder = TextGeneration::builder(&factory, device.clone())
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
    let (factory, device, _) = create_test_factory();

    // Create prefix cache
    let cache = Arc::new(RwLock::new(PrefixCache::new(10, 100)));

    // Build engine with cache
    let (mut engine, tokenizer) = InferenceEngine::builder(&factory, device)
        .seed(299792458)
        .temperature(0.8)
        .with_shared_prefix_cache(cache.clone())
        .build()
        .unwrap();

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
    let (factory, device, _) = create_test_factory();

    let cache = Arc::new(RwLock::new(PrefixCache::new(10, 100)));
    let (mut engine, tokenizer) = InferenceEngine::builder(&factory, device)
        .seed(299792458)
        .with_shared_prefix_cache(cache.clone())
        .build()
        .unwrap();

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
    let (factory, device, _) = create_test_factory();

    let cache = Arc::new(RwLock::new(PrefixCache::new(10, 100)));
    let mut pipeline = TextGeneration::builder(&factory, device)
        .seed(299792458)
        .temperature(0.8)
        .with_shared_prefix_cache(cache.clone())
        .build();

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
    tur::shared::init_tracing();
    let (factory, device, _) = create_test_factory();

    // Use greedy sampling (no .temperature() call → LogitsProcessor argmax) and
    // the default repeat_penalty=1.0 so process_logits is a no-op.  This makes
    // both the baseline and the chunked run fully deterministic: the generated
    // token sequence depends only on the model weights and the KV cache state,
    // neither of which is affected by how many tokens we process per forward pass.
    let make_pipeline = |chunk_size: Option<usize>| {
        let mut b = TextGeneration::builder(&factory, device.clone())
            .seed(299792458)
            .enable_batching(true)
            .max_batch_size(4)
            .max_prefill_batch(2)
            .max_decode_batch(4);
        if let Some(sz) = chunk_size {
            b = b.prefill_chunk_size(sz);
        }
        b.build()
    };

    let prompts = ["Hello", "How are you?", "What is Rust?"];
    let sample_len = 5;

    let make_requests = || {
        prompts
            .iter()
            .map(|p| GenerationRequest::new(p.to_string(), sample_len))
            .collect::<Vec<_>>()
    };

    // ── Baseline: single-shot prefill ─────────────────────────────────────
    let mut baseline = make_pipeline(None);
    assert!(baseline.is_batching_enabled(), "Batching should be enabled");

    for req in make_requests().iter() {
        baseline.submit_request(req).unwrap();
    }
    baseline.run_until_complete().unwrap();

    // Collect baseline results keyed by prompt for later comparison.
    let baseline_results: std::collections::HashMap<String, Vec<u32>> = baseline
        .get_all_results()
        .into_values()
        .map(|r| (r.prompt.clone(), r.generated_tokens))
        .collect();

    for prompt in prompts {
        let tokens = baseline_results
            .get(prompt)
            .unwrap_or_else(|| panic!("Baseline missing result for '{prompt}'"));
        assert!(
            !tokens.is_empty(),
            "Baseline '{prompt}': expected non-empty generated tokens"
        );
    }
    assert_eq!(baseline.active_request_count(), 0);
    assert_eq!(baseline.queued_request_count(), 0);

    // ── Chunked prefill (chunk_size = 2) ──────────────────────────────────
    // chunk_size = 2 guarantees multiple forward passes during prefill for
    // every prompt above (each is longer than 2 tokens after tokenisation).
    // The KV cache state at the end of prefill must be identical to the
    // single-shot run, so the decode phase produces the exact same tokens.
    let mut chunked = make_pipeline(Some(2));
    assert!(chunked.is_batching_enabled());

    for req in make_requests().iter() {
        chunked.submit_request(req).unwrap();
    }
    chunked.run_until_complete().unwrap();

    assert_eq!(chunked.active_request_count(), 0);
    assert_eq!(chunked.queued_request_count(), 0);

    for result in chunked.get_all_results().into_values() {
        assert!(
            !result.generated_text.is_empty(),
            "Chunked '{}': expected non-empty generated text",
            result.prompt
        );

        let baseline_tokens = baseline_results
            .get(&result.prompt)
            .unwrap_or_else(|| panic!("Chunked: no baseline for '{}'", result.prompt));

        assert_eq!(
            result.generated_tokens, *baseline_tokens,
            "Chunked prefill must produce identical tokens to single-shot for prompt '{}'",
            result.prompt
        );
    }
}

#[test]
fn test_continuous_batching_step_by_step() {
    // Test manual step-by-step execution for fine-grained control
    let (factory, device, _) = create_test_factory();
    let mut pipeline = TextGeneration::builder(&factory, device)
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
    let (factory, device, _) = create_test_factory();
    let mut pipeline = TextGeneration::builder(&factory, device)
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
    let (factory, device, _) = create_test_factory();
    let mut pipeline = TextGeneration::builder(&factory, device)
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
    let (factory, device, _) = create_test_factory();
    let mut pipeline = TextGeneration::builder(&factory, device)
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
    tur::shared::init_tracing();
    // Verify that result management, prefix caching, and chunked prefill
    // work correctly together in the continuous-batching pipeline.
    let (factory, device, _) = create_test_factory();

    // Shared prefix cache so we can inspect hit/miss stats across both pipeline runs.
    let shared_cache = Arc::new(RwLock::new(PrefixCache::new(10, 512)));

    // Build a pipeline with both features enabled.
    // chunk_size=2 forces multiple forward passes during prefill for any prompt
    // longer than 2 tokens, exercising the chunked-prefill code path.
    let make_pipeline = || {
        TextGeneration::builder(&factory, device.clone())
            .seed(299792458)
            .enable_batching(true)
            .with_shared_prefix_cache(shared_cache.clone())
            .prefill_chunk_size(2)
            .build()
    };

    let prompt = "Hello, world!";
    let sample_len = 5;

    // ── Cold run ──────────────────────────────────────────────────────────
    // Cache is empty; the engine computes all tokens and stores the first-chunk
    // KV state for future requests sharing this prompt prefix.
    let mut pipeline1 = make_pipeline();
    let handle1 = pipeline1
        .submit_request(&GenerationRequest::new(prompt.to_string(), sample_len))
        .unwrap();
    pipeline1.run_until_complete().unwrap();

    // Result is stored and accessible by handle.
    let all_results1 = pipeline1.get_all_results();
    assert_eq!(all_results1.len(), 1);
    assert!(all_results1.contains_key(&handle1.id));
    let cold_tokens = all_results1[&handle1.id].generated_tokens.clone();
    assert!(
        !cold_tokens.is_empty(),
        "cold run must generate at least one token"
    );

    // Cold run: exactly one miss, zero hits.
    {
        let guard = shared_cache.read();
        let stats = guard.stats();
        assert_eq!(stats.misses, 1, "cold run must register a cache miss");
        assert_eq!(stats.hits, 0, "cold run must not register any hits");
    }

    // Results must be empty after clear.
    pipeline1.clear_results();
    assert_eq!(pipeline1.get_all_results().len(), 0);

    // ── Warm run (same prompt, shared cache) ──────────────────────────────
    // The cache now holds the KV state from the first chunk of the cold run.
    // Submitting the same prompt must produce a cache hit for that chunk and
    // yield identical generated tokens (deterministic sampler, same KV state).
    let mut pipeline2 = make_pipeline();
    let handle2 = pipeline2
        .submit_request(&GenerationRequest::new(prompt.to_string(), sample_len))
        .unwrap();
    pipeline2.run_until_complete().unwrap();

    let all_results2 = pipeline2.get_all_results();
    assert_eq!(all_results2.len(), 1);
    assert!(all_results2.contains_key(&handle2.id));
    let warm_tokens = all_results2[&handle2.id].generated_tokens.clone();
    assert!(
        !warm_tokens.is_empty(),
        "warm run must generate at least one token"
    );

    // Prefix cache must record at least one hit on the warm run.
    assert!(
        shared_cache.read().stats().hits > 0,
        "warm run with the same prompt must hit the prefix cache",
    );

    // Chunked prefill + prefix cache must reproduce the exact same generated tokens.
    // This confirms that restoring cached KV state yields the same computation as
    // computing from scratch via chunked prefill.
    assert_eq!(
        cold_tokens, warm_tokens,
        "prefix cache + chunked prefill must reproduce identical tokens on the warm run",
    );

    // Results management API still works after the warm run.
    pipeline2.clear_results();
    assert_eq!(pipeline2.get_all_results().len(), 0);
}

// ── Guided generation ──────────────────────────────────────────────────────

/// Pipeline built without a guidance factory must return an error when a
/// request carries a grammar — not a panic or silent ignore.
#[test]
fn test_guided_no_factory_returns_error() {
    let (factory, device, _) = create_test_factory();

    let mut pipeline = TextGeneration::builder(&factory, device)
        .seed(299792458)
        .build();

    let request = GenerationRequest::new("Hello".to_string(), 5)
        .with_grammar(TopLevelGrammar::from_regex(r"[0-9]+"));

    let result = pipeline.run(&request);
    assert!(result.is_err(), "Expected error without guidance factory");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Guidance") || msg.contains("guidance"),
        "Error should mention guidance, got: {msg}"
    );
}

/// A regex grammar `[0-9]+` must restrict the sampled tokens so that every
/// character in the streamed output is an ASCII digit.
#[test]
fn test_guided_regex_output_is_only_digits() {
    let (factory, device, _) = create_test_factory();
    let guidance_factory = build_guidance_factory(&factory, device.clone());

    let output = Arc::new(Mutex::new(String::new()));
    let output_clone = output.clone();

    let mut pipeline = TextGeneration::builder(&factory, device)
        .seed(299792458)
        .with_guidance_factory(guidance_factory)
        .on_token(move |s| output_clone.lock().unwrap().push_str(s))
        .build();

    let request = GenerationRequest::new("Reply with one number only: ".to_string(), 20)
        .with_grammar(TopLevelGrammar::from_regex(r"[0-9]+"));

    let stats = pipeline
        .run(&request)
        .expect("Guided regex generation failed");
    assert!(
        stats.generated_tokens > 0,
        "Should have generated at least one token"
    );

    let generated = output.lock().unwrap().clone();
    assert!(
        generated.chars().all(|c| c.is_ascii_digit()),
        "All characters must be digits under [0-9]+ grammar, got: {generated:?}"
    );
}

/// A JSON-schema grammar must produce a complete, parse-able JSON object.
/// We use a tight schema so the model closes the object quickly.
#[test]
fn test_guided_json_schema_output_is_valid_json() {
    let (factory, device, _) = create_test_factory();
    let guidance_factory = build_guidance_factory(&factory, device.clone());

    let output = Arc::new(Mutex::new(String::new()));
    let output_clone = output.clone();

    let mut pipeline = TextGeneration::builder(&factory, device)
        .seed(299792458)
        .with_guidance_factory(guidance_factory)
        .on_token(move |s| output_clone.lock().unwrap().push_str(s))
        .build();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "integer" }
        },
        "required": ["value"]
    });

    // Give enough tokens for the grammar to close the JSON object.
    let request = GenerationRequest::new("Output JSON: ".to_string(), 60)
        .with_grammar(TopLevelGrammar::from_json_schema(schema));

    let stats = pipeline
        .run(&request)
        .expect("Guided JSON generation failed");
    assert!(stats.generated_tokens > 0);

    let generated = output.lock().unwrap().clone();
    let parsed = serde_json::from_str::<serde_json::Value>(&generated);
    assert!(
        parsed.is_ok(),
        "Output must be valid JSON under schema, got: {generated:?}"
    );
    let obj = parsed.unwrap();
    assert!(
        obj.get("value").is_some(),
        "JSON must contain required field 'value', got: {obj}"
    );
}

// ── Tool calling ──────────────────────────────────────────────────────────────

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        "Get the current weather for a given location.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. Paris, France"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }),
    )
}

/// `format_prompt_with_tools` must inject the tool schema into the prompt
/// string in the Qwen3 `<tools>` format — verifiable without running inference.
#[test]
fn test_tools_format_prompt_contains_schema() {
    let (factory, device, _) = create_test_factory();
    let pipeline = TextGeneration::builder(&factory, device).build();

    let formatted = pipeline.format_prompt_with_tools(
        "What is the weather in Paris?",
        &[weather_tool()],
        false,
    );

    assert!(
        formatted.contains("<tools>"),
        "Formatted prompt must contain <tools> block"
    );
    assert!(
        formatted.contains("get_weather"),
        "Formatted prompt must contain the tool name"
    );
    assert!(
        formatted.contains("location"),
        "Formatted prompt must contain the 'location' parameter"
    );
    assert!(
        formatted.contains("<tool_call>"),
        "Formatted prompt must contain the <tool_call> usage example"
    );
    assert!(
        formatted.contains("What is the weather in Paris?"),
        "Formatted prompt must preserve the user message"
    );
}

/// End-to-end tool-call test: with a clear weather question and greedy
/// sampling the model should emit a `<tool_call>` block that the pipeline
/// parses into `GenerationStats::tool_calls`.
#[test]
fn test_tools_model_emits_and_pipeline_parses_tool_call() {
    let (factory, device, _) = create_test_factory();

    let output = Arc::new(Mutex::new(String::new()));
    let output_clone = output.clone();

    // Greedy sampling (no temperature) for deterministic output.
    let mut pipeline = TextGeneration::builder(&factory, device)
        .seed(299792458)
        .on_token(move |s| output_clone.lock().unwrap().push_str(s))
        .build();

    // 200 tokens is enough for the model to close a <tool_call> block.
    let request = GenerationRequest::new(
        "What is the current weather in Paris, France?".to_string(),
        200,
    )
    .with_tools(vec![weather_tool()]);

    let stats = pipeline
        .run(&request)
        .expect("Tool-augmented generation failed");

    assert!(
        stats.generated_tokens > 0,
        "Should have generated at least one token"
    );

    // The generated text must reference the tool by name.
    let generated = output.lock().unwrap().clone();
    assert!(
        generated.contains("get_weather"),
        "Model output must reference the tool name 'get_weather', got: {generated:?}"
    );

    // The pipeline must have parsed at least one tool call.
    assert!(
        !stats.tool_calls.is_empty(),
        "stats.tool_calls must be non-empty when the model emits a <tool_call> block, \
         got output: {generated:?}"
    );

    let call = &stats.tool_calls[0];
    assert_eq!(
        call.name, "get_weather",
        "Parsed tool call must be 'get_weather', got: {:?}",
        call.name
    );
    assert!(
        call.arguments.get("location").is_some(),
        "Tool call arguments must include 'location', got: {}",
        call.arguments
    );
}

/// After a grammar-constrained request the constraint must be cleared so that
/// the next request (without a grammar) runs freely — no bleed-over and no
/// error.
#[test]
fn test_guided_grammar_deactivates_between_requests() {
    let (factory, device, _) = create_test_factory();
    let guidance_factory = build_guidance_factory(&factory, device.clone());

    let mut pipeline = TextGeneration::builder(&factory, device)
        .seed(299792458)
        .temperature(0.1)
        .with_guidance_factory(guidance_factory)
        .build();

    // First request — constrained to digits.
    let constrained = GenerationRequest::new("A number: ".to_string(), 10)
        .with_grammar(TopLevelGrammar::from_regex(r"[0-9]+"));
    pipeline
        .run(&constrained)
        .expect("Constrained request failed");

    // Second request — no grammar; must not inherit the previous constraint.
    let free = GenerationRequest::new("Say hello: ".to_string(), 10);
    pipeline
        .run(&free)
        .expect("Free request after grammar failed");
}
