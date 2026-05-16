use candle_core::{DType, Device, Result, Tensor};
use tur::models::ModelImpl;
use tur::models::qwen3::{Config, ModelForCausalLM};
use tur::weights::{Downloader, VarBuilderX};

/// Download a real small model for testing
/// Uses Qwen2.5-0.5B-Instruct with Q4_K_M quantization for fast downloads
fn download_test_model() -> Result<(VarBuilderX<'static>, Config, Device)> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    // Use a small quantized model for fast testing
    let model_id = Some("Qwen3-0.6B".to_string());
    let quantization = Some("Q4_K_M".to_string());

    let downloader = Downloader::new(model_id, None, quantization);
    let (paths, is_gguf) = downloader
        .prepare_model_weights()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to prepare model: {}", e)))?;

    // Load config
    let config_path = paths.get_config_filename();
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_content)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

    // Create VarBuilder
    let vb = VarBuilderX::new(&paths, is_gguf, dtype, &device)?;

    Ok((vb, config, device))
}

#[test]
fn test_varbuilderx_with_real_gguf_model() {
    let result = download_test_model();
    assert!(
        result.is_ok(),
        "Failed to download model: {:?}",
        result.err()
    );

    let (vb, _config, _device) = result.unwrap();

    // Test VarBuilderX properties with GGUF model
    assert!(vb.is_qvar_builder(), "Should be GGUF/QVarBuilder");
    assert!(!vb.is_var_builder(), "Should not be SafeTensors VarBuilder");
    assert!(vb.device().is_cpu());
    assert_eq!(vb.dtype(), DType::F32); // GGUF uses F32 for dequantized ops
}

#[test]
fn test_varbuilderx_operations_with_real_model() {
    let (vb, config, _device) = download_test_model().unwrap();

    // Test path building
    let vb_token = vb.pp("token_embd");
    assert_eq!(vb_token.module_path(), "token_embd");

    let vb_blk = vb.pp("blk");
    assert_eq!(vb_blk.module_path(), "blk");

    let vb_layer = vb_blk.pp("0");
    assert_eq!(vb_layer.module_path(), "blk.0");

    // Test key existence - GGUF models use "token_embd" for embeddings
    let vb_embed = vb.pp("token_embd");
    assert!(vb_embed.has_key("weight"), "Should have embedding weight");

    // GGUF models use "output_norm" for final norm
    let vb_norm = vb.pp("output_norm");
    assert!(vb_norm.has_key("weight"), "Should have norm weight");

    // Test non-existent key
    assert!(!vb.has_key("nonexistent_key"));

    // Test tensor retrieval - get embedding weights
    let result = vb_embed.get((config.vocab_size, config.hidden_size), "weight");
    assert!(
        result.is_ok(),
        "Failed to get embedding weight: {:?}",
        result.err()
    );

    let tensor = result.unwrap();
    assert_eq!(tensor.dims()[0], config.vocab_size);
    assert_eq!(tensor.dims()[1], config.hidden_size);
}

#[test]
fn test_qwen3_model_creation_with_real_gguf() {
    let (vb, config, _device) = download_test_model().unwrap();

    let result = ModelForCausalLM::new(&config, vb);
    assert!(
        result.is_ok(),
        "Failed to create Qwen3 model: {:?}",
        result.err()
    );
}

#[test]
fn test_qwen3_model_forward_pass_with_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Create input token IDs (batch_size=1, seq_len=5)
    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32, 4u32, 5u32]], &device).unwrap();

    // Forward pass with offset=0
    let result = model.forward(&input_ids, 0);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

    let output = result.unwrap();
    // Output should be (batch_size, 1, vocab_size) because we narrow to last token
    assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_qwen3_model_forward_with_offset_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // First forward pass
    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let result1 = model.forward(&input_ids, 0);
    assert!(result1.is_ok());

    // Second forward pass with offset (simulating autoregressive generation)
    let next_token = Tensor::new(&[[4u32]], &device).unwrap();
    let result2 = model.forward(&next_token, 3);
    assert!(result2.is_ok());

    let output = result2.unwrap();
    assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_qwen3_model_clear_kv_cache_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // First forward pass
    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let _ = model.forward(&input_ids, 0).unwrap();

    // Clear cache
    model.clear_kv_cache();

    // Should be able to do another forward pass from scratch
    let input_ids2 = Tensor::new(&[[4u32, 5u32]], &device).unwrap();
    let result = model.forward(&input_ids2, 0);
    assert!(result.is_ok());
}

#[test]
fn test_qwen3_model_batch_processing_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Batch of 2 sequences
    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32], [4u32, 5u32, 6u32]], &device).unwrap();

    let result = model.forward(&input_ids, 0);
    assert!(result.is_ok());

    let output = result.unwrap();
    // Output should be (batch_size=2, 1, vocab_size)
    assert_eq!(output.dims(), &[2, 1, config.vocab_size]);
}

#[test]
fn test_qwen3_model_different_sequence_lengths_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Test with different sequence lengths
    let test_cases = vec![1, 5, 10, 20];

    for seq_len in test_cases {
        model.clear_kv_cache();

        let input_data: Vec<u32> = (0..seq_len)
            .map(|i| (i % config.vocab_size) as u32)
            .collect();
        let input_ids = Tensor::from_vec(input_data, (1, seq_len), &device).unwrap();

        let result = model.forward(&input_ids, 0);
        assert!(
            result.is_ok(),
            "Failed for sequence length {}: {:?}",
            seq_len,
            result.err()
        );

        let output = result.unwrap();
        assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
    }
}

#[test]
fn test_qwen3_model_numerical_stability_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let output = model.forward(&input_ids, 0).unwrap();

    // Check that output contains no NaN or Inf values
    let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        output_vec.iter().all(|&x| x.is_finite()),
        "Output contains NaN or Inf values"
    );
}

#[test]
fn test_varbuilderx_module_path_tracking_real_model() {
    let (vb, _config, _device) = download_test_model().unwrap();

    // Test nested path building with GGUF naming
    let vb1 = vb.pp("blk");
    assert_eq!(vb1.module_path(), "blk");

    let vb2 = vb1.pp("0");
    assert_eq!(vb2.module_path(), "blk.0");

    let vb3 = vb2.pp("attn_q");
    assert_eq!(vb3.module_path(), "blk.0.attn_q");
}

#[test]
fn test_qwen3_autoregressive_generation_simulation_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Simulate autoregressive generation
    let prompt = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let mut offset = 0;

    // Process prompt
    let output = model.forward(&prompt, offset).unwrap();
    assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
    offset += 3;

    // Generate 3 tokens
    for _ in 0..3 {
        let next_token = Tensor::new(&[[10u32]], &device).unwrap();
        let output = model.forward(&next_token, offset).unwrap();
        assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
        offset += 1;
    }
}

#[test]
fn test_varbuilderx_all_keys_real_model() {
    let (vb, _config, _device) = download_test_model().unwrap();

    // GGUF models should expose all keys
    let all_keys = vb.all_keys();
    assert!(all_keys.is_some(), "GGUF models should expose all keys");

    let keys = all_keys.unwrap();
    assert!(!keys.is_empty(), "Should have at least some keys");

    // Check for expected GGUF keys
    assert!(
        keys.iter().any(|k| k.contains("token_embd")),
        "Should have token_embd key"
    );
    assert!(
        keys.iter().any(|k| k.contains("output_norm")),
        "Should have output_norm key"
    );
}

#[test]
fn test_qwen3_forward_batch_with_variable_positions() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Create a batch of 3 sequences with different positions
    // Simulating decode phase where each request is at a different position
    let input_ids = Tensor::new(&[[10u32], [20u32], [30u32]], &device).unwrap();
    let positions = vec![5, 10, 15]; // Each sequence at different position

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_ok(),
        "Batched forward with variable positions failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    // Output should be (batch_size=3, 1, vocab_size)
    assert_eq!(output.dims(), &[3, 1, config.vocab_size]);
}

#[test]
fn test_qwen3_forward_batch_consistency_with_single_forward() {
    let (vb, config, device) = download_test_model().unwrap();

    // Test that batched forward with same position produces similar results to single forward
    let input_ids_single = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let input_ids_batch = Tensor::new(&[[1u32, 2u32, 3u32], [1u32, 2u32, 3u32]], &device).unwrap();

    // Single forward
    let mut model1 = ModelForCausalLM::new(&config, vb.clone()).unwrap();
    let output_single = model1.forward(&input_ids_single, 0).unwrap();

    // Batched forward with same inputs
    let mut model2 = ModelForCausalLM::new(&config, vb).unwrap();
    let positions = vec![0, 0];
    let output_batch = model2
        .forward_batch(&input_ids_batch, &positions, None)
        .unwrap();

    // Check shapes
    assert_eq!(output_single.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(output_batch.dims(), &[2, 1, config.vocab_size]);

    // Extract first batch element and flatten to compare
    let output_batch_first = output_batch.narrow(0, 0, 1).unwrap();

    // Both outputs are [1, 1, vocab_size], flatten to [vocab_size] for comparison
    let single_flat = output_single.flatten_all().unwrap();
    let batch_flat = output_batch_first.flatten_all().unwrap();

    // Compare first few logits (outputs should be very similar)
    let single_vec = single_flat.to_vec1::<f32>().unwrap();
    let batch_vec = batch_flat.to_vec1::<f32>().unwrap();

    let max_diff = single_vec
        .iter()
        .zip(batch_vec.iter())
        .take(10) // Compare first 10 logits
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, x| acc.max(x));

    assert!(
        max_diff < 1e-2,
        "Outputs differ too much: max_diff = {}",
        max_diff
    );
}

#[test]
fn test_qwen3_forward_batch_prefill_phase() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Simulate prefill phase with multiple sequences of same length
    let input_ids = Tensor::new(
        &[
            [1u32, 2u32, 3u32, 4u32],
            [5u32, 6u32, 7u32, 8u32],
            [9u32, 10u32, 11u32, 12u32],
        ],
        &device,
    )
    .unwrap();

    // All starting from position 0 (prefill phase)
    let positions = vec![0, 0, 0];

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(result.is_ok(), "Batched prefill failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.dims(), &[3, 1, config.vocab_size]);
}

#[test]
fn test_qwen3_forward_batch_decode_phase() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // First do prefill for 3 sequences
    let prefill_ids = Tensor::new(
        &[[1u32, 2u32, 3u32], [4u32, 5u32, 6u32], [7u32, 8u32, 9u32]],
        &device,
    )
    .unwrap();
    let prefill_positions = vec![0, 0, 0];
    let _ = model
        .forward_batch(&prefill_ids, &prefill_positions, None)
        .unwrap();

    // Now simulate decode phase - each sequence generates next token
    let decode_ids = Tensor::new(&[[10u32], [11u32], [12u32]], &device).unwrap();
    let decode_positions = vec![3, 3, 3]; // All at position 3 after prefill

    let result = model.forward_batch(&decode_ids, &decode_positions, None);
    assert!(result.is_ok(), "Batched decode failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.dims(), &[3, 1, config.vocab_size]);
}

#[test]
fn test_qwen3_forward_batch_mixed_positions() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Simulate continuous batching scenario:
    // - Request 1: at position 10 (been generating for a while)
    // - Request 2: at position 3 (just started)
    // - Request 3: at position 20 (almost done)
    let input_ids = Tensor::new(&[[100u32], [200u32], [300u32]], &device).unwrap();
    let positions = vec![10, 3, 20];

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_ok(),
        "Batched forward with mixed positions failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_eq!(output.dims(), &[3, 1, config.vocab_size]);
}

#[test]
fn test_qwen3_forward_batch_position_mismatch_error() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32], [2u32], [3u32]], &device).unwrap();
    let positions = vec![0, 5]; // Wrong length - should be 3, not 2

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_err(),
        "Should fail when positions length doesn't match batch size"
    );
}

#[test]
fn test_qwen3_forward_batch_large_batch() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    // Test with larger batch size (8 sequences)
    let batch_size = 8;
    let input_data: Vec<Vec<u32>> = (0..batch_size).map(|i| vec![(i + 1) as u32]).collect();
    let positions: Vec<usize> = (0..batch_size).map(|i| i * 2).collect();

    // Convert to flat array for Tensor::new
    let flat_data: Vec<u32> = input_data.iter().flatten().copied().collect();
    let input_ids = Tensor::from_vec(flat_data, (batch_size, 1), &device).unwrap();

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_ok(),
        "Large batch forward failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_eq!(output.dims(), &[batch_size, 1, config.vocab_size]);
}
