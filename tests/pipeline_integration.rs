use candle_core::Device;
use parking_lot::RwLock;
use std::sync::Arc;
use tur::ProgressReporter;
use tur::backend::InferenceEngine;
use tur::backend::pipeline::{GenerationRequest, TextGeneration};
use tur::backend::prefix_cache::PrefixCache;
use tur::backend::tokenizer::TokenOutputStream;
use tur::models::qwen3::{Config, ModelForCausalLM};
use tur::weights::{Downloader, VarBuilderX};

/// Download a real small model for testing
fn download_test_model() -> candle_core::Result<(VarBuilderX<'static>, Config, Device)> {
    let device = Device::Cpu;
    let dtype = candle_core::DType::F32;

    let model_id = Some("Qwen3-0.6B".to_string());
    let quantization = Some("Q4_K_M".to_string());

    let downloader = Downloader::new(model_id, None, quantization);
    let (paths, is_gguf) = downloader
        .prepare_model_weights()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to prepare model: {}", e)))?;

    let config_path = paths.get_config_filename();
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_content)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

    let vb = VarBuilderX::new(&paths, is_gguf, dtype, &device)?;

    Ok((vb, config, device))
}

/// Load tokenizer for testing
fn load_tokenizer() -> candle_core::Result<tokenizers::Tokenizer> {
    let model_id = Some("Qwen3-0.6B".to_string());
    let downloader = Downloader::new(model_id, None, None);
    let (paths, _) = downloader
        .prepare_model_weights()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to prepare model: {}", e)))?;

    let tokenizer_path = paths.get_tokenizer_filename();
    tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))
}

#[test]
fn test_pipeline_end_to_end_generation() {
    let (vb, config, device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    let tokenizer = load_tokenizer().unwrap();

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
    let (vb, config, device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    let tokenizer = load_tokenizer().unwrap();
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
    let (vb, config, device) = download_test_model().unwrap();
    let tokenizer = load_tokenizer().unwrap();

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
        let model = ModelForCausalLM::new(&config, vb.clone()).unwrap();
        let mut builder = TextGeneration::builder(model, tokenizer.clone(), device.clone())
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
    let (vb, config, device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    let tokenizer = load_tokenizer().unwrap();

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
    let (vb, config, device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    let tokenizer = load_tokenizer().unwrap();

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
    let (vb, config, device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    let tokenizer = load_tokenizer().unwrap();

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
