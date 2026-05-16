use candle_core::{DType, Device};
use std::sync::Arc;
use tur::{
    ModelFactory,
    backend::{InferenceEngine, guidance::{self, ParserFactory}},
    models::qwen3::ModelForCausalLM,
};

pub fn create_test_factory() -> (ModelFactory<ModelForCausalLM>, Device, DType) {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let model_id = "Qwen3-0.6B".to_string();
    let quantization = Some("Q4_K_M".to_string());

    let factory = ModelFactory::<ModelForCausalLM>::new(
        tur::ModelSource::HuggingFace(model_id),
        quantization,
        device.clone(),
        dtype,
    );
    (factory, device, dtype)
}

/// Build a `ParserFactory` for guided-generation tests.
///
/// Loads the tokenizer via the engine builder (which also loads the model weights).
/// Callers should pass the same `factory` to the pipeline builder afterwards; the
/// model weights will be loaded a second time from the local HuggingFace cache,
/// which is fast.
pub fn build_guidance_factory(
    factory: &ModelFactory<ModelForCausalLM>,
    device: Device,
) -> Arc<ParserFactory> {
    let (_, tokenizer) = InferenceEngine::builder(factory, device)
        .build()
        .expect("Failed to build engine to extract tokenizer");
    guidance::build_llg_factory(tokenizer, None).expect("Failed to build guidance factory")
}
