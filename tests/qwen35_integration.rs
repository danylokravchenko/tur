use candle_core::{DType, Device, Result, Tensor};
use tur::models::ModelImpl;
use tur::models::qwen35::{Config, ModelForCausalLM};
use tur::weights::{Downloader, VarBuilderX};

fn download_test_model() -> Result<(VarBuilderX<'static>, Config, Device)> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let model_id = Some("Qwen3.5-0.8B".to_string());
    let quantization = Some("Q4_K_M".to_string());

    let downloader = Downloader::new(model_id, None, quantization);
    let (paths, is_gguf) = downloader
        .prepare_model_weights()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to prepare model: {}", e)))?;

    let config_content = std::fs::read_to_string(paths.config_filename())?;
    let config: Config = serde_json::from_str(&config_content)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

    let vb = VarBuilderX::new(&paths, is_gguf, dtype, &device)?;

    Ok((vb, config, device))
}

#[test]
fn test_qwen35_model_creation() {
    let result = download_test_model();
    assert!(
        result.is_ok(),
        "Failed to download model: {:?}",
        result.err()
    );

    let (vb, config, _device) = result.unwrap();
    let result = ModelForCausalLM::new(&config, vb);
    assert!(
        result.is_ok(),
        "Failed to create Qwen3.5 model: {:?}",
        result.err()
    );
}

#[test]
fn test_qwen35_model_forward_pass() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32, 4u32, 5u32]], &device).unwrap();
    let result = model.forward(&input_ids, 0);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.dims(), &[1, 1, config.text_config.vocab_size]);
}

#[test]
fn test_qwen35_model_forward_with_offset() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let result1 = model.forward(&input_ids, 0);
    assert!(result1.is_ok());

    let next_token = Tensor::new(&[[4u32]], &device).unwrap();
    let result2 = model.forward(&next_token, 3);
    assert!(result2.is_ok());

    let output = result2.unwrap();
    assert_eq!(output.dims(), &[1, 1, config.text_config.vocab_size]);
}

#[test]
fn test_qwen35_model_clear_kv_cache() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let _ = model.forward(&input_ids, 0).unwrap();

    model.clear_kv_cache();

    let input_ids2 = Tensor::new(&[[4u32, 5u32]], &device).unwrap();
    let result = model.forward(&input_ids2, 0);
    assert!(
        result.is_ok(),
        "Forward after cache clear failed: {:?}",
        result.err()
    );
}

#[test]
fn test_qwen35_model_numerical_stability() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let output = model.forward(&input_ids, 0).unwrap();

    let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        output_vec.iter().all(|&x| x.is_finite()),
        "Output contains NaN or Inf values"
    );
}

#[test]
fn test_qwen35_model_forward_batch() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[10u32], [20u32], [30u32]], &device).unwrap();
    let positions = vec![5, 10, 15];

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(result.is_ok(), "Batched forward failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.dims(), &[3, 1, config.text_config.vocab_size]);
}

#[test]
fn test_qwen35_forward_batch_position_mismatch_error() {
    let (vb, _config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&_config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32], [2u32], [3u32]], &device).unwrap();
    let positions = vec![0, 5]; // wrong length

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_err(),
        "Should fail when positions length doesn't match batch size"
    );
}

#[test]
fn test_qwen35_autoregressive_generation() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let prompt = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let mut offset = 0;

    let output = model.forward(&prompt, offset).unwrap();
    assert_eq!(output.dims(), &[1, 1, config.text_config.vocab_size]);
    offset += 3;

    for _ in 0..3 {
        let next_token = Tensor::new(&[[10u32]], &device).unwrap();
        let output = model.forward(&next_token, offset).unwrap();
        assert_eq!(output.dims(), &[1, 1, config.text_config.vocab_size]);
        offset += 1;
    }
}

#[test]
fn test_qwen35_format_prompt() {
    let (vb, config, _device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();

    let prompt = model.format_prompt("Hello, world!", false);
    assert!(prompt.contains("Hello, world!"));
    assert!(prompt.contains("<|im_start|>"));
    assert!(prompt.contains("/no_think"));

    let prompt_think = model.format_prompt("Hello", true);
    assert!(prompt_think.contains("/think"));
}

#[test]
fn test_qwen35_model_name() {
    let (vb, config, _device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    assert_eq!(model.name(), "Qwen3.5");
}

#[test]
fn test_qwen35_num_layers() {
    let (vb, config, _device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    assert_eq!(model.num_layers(), config.text_config.num_hidden_layers);
}

#[test]
fn test_qwen35_kv_cache_state_roundtrip() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let _ = model.forward(&input_ids, 0).unwrap();

    let state = model.get_kv_cache_state().unwrap();
    // Qwen3.5 is a hybrid model: linear attention layers don't expose KV state,
    // so get_kv_cache_state returns an empty vec (prefix caching disabled).
    let _ = config.text_config.num_hidden_layers; // referenced to avoid unused warning

    model.clear_kv_cache();
    if !state.is_empty() {
        model.set_kv_cache_state(state).unwrap();
    }

    let next_token = Tensor::new(&[[4u32]], &device).unwrap();
    let result = model.forward(&next_token, 3);
    assert!(
        result.is_ok(),
        "Forward after state restore failed: {:?}",
        result.err()
    );
}

#[test]
fn test_qwen35_model_different_sequence_lengths() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    for seq_len in [1, 5, 10, 20] {
        model.clear_kv_cache();

        let input_data: Vec<u32> = (0..seq_len)
            .map(|i| (i % config.text_config.vocab_size) as u32)
            .collect();
        let input_ids = Tensor::from_vec(input_data, (1, seq_len), &device).unwrap();

        let result = model.forward(&input_ids, 0);
        assert!(
            result.is_ok(),
            "Failed for seq_len={}: {:?}",
            seq_len,
            result.err()
        );
        assert_eq!(
            result.unwrap().dims(),
            &[1, 1, config.text_config.vocab_size]
        );
    }
}

#[test]
fn test_qwen35_model_creation_safetensors() {
    let device = candle_core::Device::Cpu;
    let dtype = candle_core::DType::F32;

    let downloader = tur::weights::Downloader::new(
        Some("Qwen3.5-0.8B".to_string()),
        None,
        None, // no quantization → safetensors
    );
    let result = downloader.prepare_model_weights();
    assert!(result.is_ok(), "Failed to download safetensors model: {:?}", result.err());

    let (paths, is_gguf) = result.unwrap();
    assert!(!is_gguf, "Expected safetensors, got GGUF");

    let config_content = std::fs::read_to_string(paths.config_filename()).unwrap();
    let config: Config = serde_json::from_str(&config_content).unwrap();

    let vb = tur::weights::VarBuilderX::new(&paths, is_gguf, dtype, &device).unwrap();
    let result = ModelForCausalLM::new(&config, vb);
    assert!(result.is_ok(), "Failed to create model from safetensors: {:?}", result.err());

    let mut model = result.unwrap();
    let input_ids = candle_core::Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let fwd = model.forward(&input_ids, 0);
    assert!(fwd.is_ok(), "Forward pass failed: {:?}", fwd.err());
    assert_eq!(fwd.unwrap().dims(), &[1, 1, config.text_config.vocab_size]);
}
