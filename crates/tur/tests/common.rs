#![forbid(unsafe_code)]
#![allow(dead_code)]
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf};
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};
use tur::{
    ModelFactory, ModelSource, Result, TurError,
    core::execution::{KvFormat, KvLayout},
    models::qwen3_5::{Config, WeightFormat},
};

const TINY: &str = r#"{
  "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
  "hidden_size": 64, "intermediate_size": 128, "vocab_size": 96, "num_hidden_layers": 4,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
  "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
  "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "max_position_embeddings": 512,
  "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attn_output_gate": true,
  "tie_word_embeddings": true, "eos_token_id": 1, "image_token_id": 94,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25}
}"#;

pub struct Checkpoint(pub PathBuf);
impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl Checkpoint {
    /// With zero decoder updates and tied embeddings, token 5 is the unique greedy winner.
    pub fn new() -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("tur-adapter-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir)?;
        let fixture = Self(dir);
        std::fs::write(fixture.0.join("config.json"), TINY)?;
        let config = Config::from_json(TINY)?;
        let mut weights = HashMap::new();
        for (name, shape) in config.weight_shapes()? {
            let mut values = vec![0f32; shape.iter().product()];
            if name.ends_with("embed_tokens.weight") {
                values.fill(1.0);
                for value in values.iter_mut().skip(5 * 64).take(64) {
                    *value = 2.0;
                }
            }
            weights.insert(
                name,
                Tensor::from_vec(values, shape, &Device::Cpu)?.to_dtype(DType::BF16)?,
            );
        }
        candle_core::safetensors::save(&weights, fixture.0.join("model.safetensors"))?;
        let wordlevel = WordLevel::builder()
            .vocab((0..96u32).map(|id| (format!("t{id}"), id)).collect())
            .unk_token("t0".into())
            .build()
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(wordlevel);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .save(fixture.0.join("tokenizer.json"), false)
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        std::fs::write(
            fixture.0.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}"
            })
            .to_string(),
        )?;
        std::fs::write(
            fixture.0.join("generation_config.json"),
            r#"{"do_sample":false}"#,
        )?;
        Ok(fixture)
    }

    pub fn factory(&self) -> ModelFactory {
        ModelFactory {
            source: ModelSource::LocalPath(self.0.clone()),
            revision: "tiny-adapter-fixture-v1".into(),
            format: WeightFormat::F32,
            device: Device::Cpu,
            // 16-dimension heads take unscaled formats only.
            kv: KvLayout::new(16, KvFormat::F32),
        }
    }
}
