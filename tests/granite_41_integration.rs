use candle_core::{DType, Device, Result, Tensor};
use tur::ModelSource;
use tur::backend::pipeline::InferencePipeline;
use tur::backend::tools::ToolDefinition;
use tur::models::ModelImpl;
use tur::models::granite_41::{Config, ModelForCausalLM};
use tur::weights::{Downloader, VarBuilderX};

/// Download the smallest available Granite 4.1 model for testing.
/// Uses ibm-granite/granite-4.1-3b; weights are cast to F32 because candle's
/// CPU backend does not support BF16 matmul.
fn download_test_model() -> Result<(VarBuilderX<'static>, Config, Device)> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let model_id = Some("ibm-granite/granite-4.1-3b".to_string());

    let downloader = Downloader::new(model_id, None, None);
    let (paths, is_gguf) = downloader
        .prepare_model_weights()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to prepare model: {}", e)))?;

    let config_path = paths.config_filename();
    let config_content = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config_content)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

    let vb = VarBuilderX::new(&paths, is_gguf, dtype, &device)?;

    Ok((vb, config, device))
}

#[test]
fn test_varbuilderx_with_real_safetensors_model() {
    let result = download_test_model();
    assert!(
        result.is_ok(),
        "Failed to download model: {:?}",
        result.err()
    );

    let (vb, _config, _device) = result.unwrap();

    assert!(
        !vb.is_qvar_builder(),
        "SafeTensors model should not be QVarBuilder"
    );
    assert!(vb.is_var_builder(), "Should be SafeTensors VarBuilder");
    assert!(vb.device().is_cpu());
}

#[test]
fn test_varbuilderx_operations_with_real_model() {
    let (vb, config, _device) = download_test_model().unwrap();

    let vb_model = vb.pp("model");
    assert_eq!(vb_model.module_path(), "model");

    let vb_layers = vb_model.pp("layers");
    assert_eq!(vb_layers.module_path(), "model.layers");

    let vb_layer0 = vb_layers.pp("0");
    assert_eq!(vb_layer0.module_path(), "model.layers.0");

    // SafeTensors models use "model.embed_tokens"
    let vb_embed = vb.pp("model.embed_tokens");
    assert!(vb_embed.has_key("weight"), "Should have embedding weight");

    // SafeTensors models use "model.norm"
    let vb_norm = vb.pp("model.norm");
    assert!(vb_norm.has_key("weight"), "Should have norm weight");

    assert!(!vb.has_key("nonexistent_key"));

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
fn test_granite_config_values() {
    let (_vb, config, _device) = download_test_model().unwrap();

    assert_eq!(config.vocab_size, 100352);
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_hidden_layers, 40);
    assert_eq!(config.num_attention_heads, 40);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim(), 64);
    assert!(config.tie_word_embeddings);
    assert_eq!(config.eos_token_id, 100257);
}

#[test]
fn test_granite_model_creation_with_real_safetensors() {
    let (vb, config, _device) = download_test_model().unwrap();

    let result = ModelForCausalLM::new(&config, vb);
    assert!(
        result.is_ok(),
        "Failed to create Granite 4.1 model: {:?}",
        result.err()
    );
}

#[test]
fn test_granite_model_eos_token_ids() {
    let (vb, config, _device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();

    let ids = model.eos_token_ids();
    assert_eq!(ids, vec![100257], "Granite EOS token ID should be 100257");
}

#[test]
fn test_granite_model_name() {
    let (vb, config, _device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    assert_eq!(model.name(), "Granite4.1");
}

#[test]
fn test_granite_model_num_layers() {
    let (vb, config, _device) = download_test_model().unwrap();
    let model = ModelForCausalLM::new(&config, vb).unwrap();
    assert_eq!(model.num_layers(), config.num_hidden_layers);
}

#[test]
fn test_granite_model_forward_pass_with_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32, 4u32, 5u32]], &device).unwrap();

    let result = model.forward(&input_ids, 0);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

    let output = result.unwrap();
    // Output should be (batch_size=1, 1, vocab_size)
    assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_granite_model_forward_with_offset_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let result1 = model.forward(&input_ids, 0);
    assert!(result1.is_ok());

    let next_token = Tensor::new(&[[4u32]], &device).unwrap();
    let result2 = model.forward(&next_token, 3);
    assert!(result2.is_ok());

    let output = result2.unwrap();
    assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_granite_model_clear_kv_cache_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let _ = model.forward(&input_ids, 0).unwrap();

    model.clear_kv_cache();

    let input_ids2 = Tensor::new(&[[4u32, 5u32]], &device).unwrap();
    let result = model.forward(&input_ids2, 0);
    assert!(result.is_ok());
}

#[test]
fn test_granite_model_batch_processing_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32], [4u32, 5u32, 6u32]], &device).unwrap();

    let result = model.forward(&input_ids, 0);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.dims(), &[2, 1, config.vocab_size]);
}

#[test]
fn test_granite_model_different_sequence_lengths_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

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
fn test_granite_model_numerical_stability_real_model() {
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
fn test_granite_autoregressive_generation_simulation_real_model() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let prompt = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let mut offset = 0;

    let output = model.forward(&prompt, offset).unwrap();
    assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
    offset += 3;

    for _ in 0..3 {
        let next_token = Tensor::new(&[[10u32]], &device).unwrap();
        let output = model.forward(&next_token, offset).unwrap();
        assert_eq!(output.dims(), &[1, 1, config.vocab_size]);
        offset += 1;
    }
}

#[test]
fn test_granite_forward_batch_with_variable_positions() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[10u32], [20u32], [30u32]], &device).unwrap();
    let positions = vec![5, 10, 15];

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_ok(),
        "Batched forward with variable positions failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_eq!(output.dims(), &[3, 1, config.vocab_size]);
}

#[test]
fn test_granite_forward_batch_consistency_with_single_forward() {
    let (vb, config, device) = download_test_model().unwrap();

    let input_ids_single = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let input_ids_batch = Tensor::new(&[[1u32, 2u32, 3u32], [1u32, 2u32, 3u32]], &device).unwrap();

    let mut model1 = ModelForCausalLM::new(&config, vb.clone()).unwrap();
    let output_single = model1.forward(&input_ids_single, 0).unwrap();

    let mut model2 = ModelForCausalLM::new(&config, vb).unwrap();
    let positions = vec![0, 0];
    let output_batch = model2
        .forward_batch(&input_ids_batch, &positions, None)
        .unwrap();

    assert_eq!(output_single.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(output_batch.dims(), &[2, 1, config.vocab_size]);

    let output_batch_first = output_batch.narrow(0, 0, 1).unwrap();

    let single_flat = output_single
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let batch_flat = output_batch_first
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let max_diff = single_flat
        .iter()
        .zip(batch_flat.iter())
        .take(10)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, x| acc.max(x));

    assert!(
        max_diff < 1e-2,
        "Outputs differ too much: max_diff = {}",
        max_diff
    );
}

#[test]
fn test_granite_forward_batch_prefill_phase() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(
        &[
            [1u32, 2u32, 3u32, 4u32],
            [5u32, 6u32, 7u32, 8u32],
            [9u32, 10u32, 11u32, 12u32],
        ],
        &device,
    )
    .unwrap();
    let positions = vec![0, 0, 0];

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(result.is_ok(), "Batched prefill failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.dims(), &[3, 1, config.vocab_size]);
}

#[test]
fn test_granite_forward_batch_decode_phase() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let prefill_ids = Tensor::new(
        &[[1u32, 2u32, 3u32], [4u32, 5u32, 6u32], [7u32, 8u32, 9u32]],
        &device,
    )
    .unwrap();
    let prefill_positions = vec![0, 0, 0];
    let _ = model
        .forward_batch(&prefill_ids, &prefill_positions, None)
        .unwrap();

    let decode_ids = Tensor::new(&[[10u32], [11u32], [12u32]], &device).unwrap();
    let decode_positions = vec![3, 3, 3];

    let result = model.forward_batch(&decode_ids, &decode_positions, None);
    assert!(result.is_ok(), "Batched decode failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.dims(), &[3, 1, config.vocab_size]);
}

#[test]
fn test_granite_forward_batch_mixed_positions() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

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
fn test_granite_forward_batch_position_mismatch_error() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32], [2u32], [3u32]], &device).unwrap();
    let positions = vec![0, 5]; // Wrong length — should be 3

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_err(),
        "Should fail when positions length doesn't match batch size"
    );
}

#[test]
fn test_granite_forward_batch_large_batch() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let batch_size = 8;
    let input_data: Vec<u32> = (1..=(batch_size as u32)).collect();
    let positions: Vec<usize> = (0..batch_size).map(|i| i * 2).collect();
    let input_ids = Tensor::from_vec(input_data, (batch_size, 1), &device).unwrap();

    let result = model.forward_batch(&input_ids, &positions, None);
    assert!(
        result.is_ok(),
        "Large batch forward failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_eq!(output.dims(), &[batch_size, 1, config.vocab_size]);
}

#[test]
fn test_granite_kv_cache_state_save_and_restore() {
    let (vb, config, device) = download_test_model().unwrap();
    let mut model = ModelForCausalLM::new(&config, vb).unwrap();

    let input_ids = Tensor::new(&[[1u32, 2u32, 3u32]], &device).unwrap();
    let _ = model.forward(&input_ids, 0).unwrap();

    // Save state
    let state = model.get_kv_cache_state().unwrap();
    assert_eq!(state.len(), config.num_hidden_layers);

    // Advance one more token
    let next = Tensor::new(&[[4u32]], &device).unwrap();
    let _ = model.forward(&next, 3).unwrap();

    // Restore to saved state
    model.set_kv_cache_state(state).unwrap();

    // Should reproduce the same output as before the extra token
    let result = model.forward(&Tensor::new(&[[4u32]], &device).unwrap(), 3);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().dims(), &[1, 1, config.vocab_size]);
}

// ── Chat-template prompt formatting ─────────────────────────────────────────

fn make_granite_pipeline() -> InferencePipeline<tur::models::Granite41ModelForCausalLM> {
    let factory = tur::ModelFactory::<tur::models::Granite41ModelForCausalLM>::new(
        ModelSource::HuggingFace("ibm-granite/granite-4.1-3b".to_string()),
        None,
        Device::Cpu,
        DType::F32,
    );
    InferencePipeline::builder(&factory, Device::Cpu).build()
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get current weather for a location".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City and country" },
                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["location"]
        }),
    }
}

/// Verify that the Granite chat template is loaded and correctly injects tools.
/// Checks structure without running inference.
#[test]
fn test_granite_chat_template_tool_formatting() {
    let pipeline = make_granite_pipeline();

    let formatted = pipeline.format_prompt_with_tools(
        "What is the weather in Paris?",
        &[weather_tool()],
        false,
    );

    // Granite's template wraps tools in a system message with <tools> XML tags
    assert!(
        formatted.contains("<|start_of_role|>system<|end_of_role|>"),
        "should open with a system role: {formatted}"
    );
    assert!(
        formatted.contains("<tools>"),
        "should contain <tools> XML block: {formatted}"
    );
    assert!(
        formatted.contains("get_weather"),
        "should inject the tool name: {formatted}"
    );
    assert!(
        formatted.contains("location"),
        "should inject the 'location' parameter: {formatted}"
    );
    assert!(
        formatted.contains("<tool_call>"),
        "should describe the <tool_call> response format: {formatted}"
    );
    assert!(
        formatted.contains("What is the weather in Paris?"),
        "should preserve the user message: {formatted}"
    );
    assert!(
        formatted.contains("<|start_of_role|>assistant<|end_of_role|>"),
        "should end with the assistant generation prompt: {formatted}"
    );
}

/// Verify that `chat_template.jinja` was actually loaded and used — not the
/// static offline fallback in `granite_41::ModelForCausalLM`.
///
/// The Jinja2 template appends a sentence after the `</tool_call>` example
/// ("If a tool does not exist…") that the static fallback omits.  Its presence
/// proves the Jinja2 render path fired.
#[test]
fn test_granite_chat_template_was_loaded() {
    let pipeline = make_granite_pipeline();

    let formatted = pipeline.format_prompt_with_tools(
        "What is the weather in Paris?",
        &[weather_tool()],
        false,
    );

    assert!(
        formatted.contains("If a tool does not exist"),
        "sentence only present in chat_template.jinja, not in the static fallback: {formatted}"
    );
}

/// Verify that `--thinking` is a no-op for Granite: the flag must not alter
/// the prompt in any way because Granite's chat template has no thinking toggle.
#[test]
fn test_granite_thinking_flag_is_noop() {
    let pipeline = make_granite_pipeline();

    let without_thinking = pipeline.format_prompt("Hello", false);
    let with_thinking = pipeline.format_prompt("Hello", true);

    assert_eq!(
        without_thinking, with_thinking,
        "`--thinking` should not change the Granite prompt: \
         without={without_thinking:?}  with={with_thinking:?}"
    );
}
