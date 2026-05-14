use candle_core::Device;
use tur::ProgressReporter;
use tur::backend::pipeline::{GenerationRequest, TextGeneration};
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
