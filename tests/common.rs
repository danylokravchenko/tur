use candle_core::{DType, Device};
/// Create a test model and tokenizer using the ModelFactory
use tur::{ModelFactory, models::qwen3::ModelForCausalLM};

pub fn create_test_model() -> candle_core::Result<(ModelForCausalLM, tokenizers::Tokenizer, Device)>
{
    let device = Device::Cpu;
    let dtype = DType::F32;

    let model_id = Some("Qwen3-0.6B".to_string());
    let quantization = Some("Q4_K_M".to_string());

    let factory = ModelFactory::new(model_id, None, quantization, device.clone(), dtype);
    let (model, tokenizer) = factory
        .create_model(None)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create model: {}", e)))?;

    Ok((model, tokenizer, device))
}
