use candle_core::{DType, Device};
/// Create a test model factory
use tur::{ModelFactory, models::qwen3::ModelForCausalLM};

pub fn create_test_factory() -> (ModelFactory<ModelForCausalLM>, Device, DType) {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let model_id = Some("Qwen3-0.6B".to_string());
    let quantization = Some("Q4_K_M".to_string());

    let factory =
        ModelFactory::<ModelForCausalLM>::new(model_id, None, quantization, device.clone(), dtype);
    (factory, device, dtype)
}
