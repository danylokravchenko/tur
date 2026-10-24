#![allow(dead_code, reason = "some fixtures serve only Metal-gated tests")]
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

pub(crate) struct Checkpoint(pub PathBuf);
impl Drop for Checkpoint {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// Deterministic values in `(-scale, scale)`.
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0, -1.0) * scale
        })
        .collect()
}

impl Checkpoint {
    /// With zero decoder updates and tied embeddings, token 5 is the unique greedy winner.
    pub(crate) fn new() -> Result<Self> {
        Self::write(TINY, |name, count, _| {
            let mut values = vec![0f32; count];
            if name.ends_with("embed_tokens.weight") {
                values.fill(1.0);
                for value in values.iter_mut().skip(5 * 64).take(64) {
                    *value = 2.0;
                }
            }
            values
        })
    }

    /// Random weights without modality placeholders: every output depends on the whole
    /// context and on the request's own recurrent and attention state, so leaked or
    /// misplaced state changes generated text.
    pub(crate) fn random() -> Result<Self> {
        let config = TINY.replace(r#", "image_token_id": 94"#, "");
        Self::random_config(&config)
    }

    /// Random fixture using head sizes supported by the fused Metal kernels.
    #[cfg(feature = "metal")]
    pub(crate) fn fused() -> Result<Self> {
        let mut config: serde_json::Value = serde_json::from_str(TINY)?;
        for (name, value) in [
            ("hidden_size", 256),
            ("intermediate_size", 512),
            ("head_dim", 64),
            ("linear_key_head_dim", 64),
            ("linear_value_head_dim", 64),
        ] {
            config[name] = serde_json::json!(value);
        }
        if let Some(object) = config.as_object_mut() {
            object.remove("image_token_id");
        }
        Self::random_config(&config.to_string())
    }

    /// [`Self::random`] with a random native multi-token prediction head.
    pub(crate) fn random_mtp() -> Result<Self> {
        let config = TINY.replace(r#", "image_token_id": 94"#, "").replace(
            r#""num_hidden_layers": 4,"#,
            r#""num_hidden_layers": 4, "mtp_num_hidden_layers": 1,"#,
        );
        Self::random_config(&config)
    }

    /// [`Self::fused`] with a random native multi-token prediction head.
    #[cfg(feature = "metal")]
    pub(crate) fn fused_mtp() -> Result<Self> {
        let fixture = Self::fused()?;
        let path = fixture.0.join("config.json");
        let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        config["mtp_num_hidden_layers"] = serde_json::json!(1);
        drop(fixture);
        Self::random_config(&config.to_string())
    }

    fn random_config(config: &str) -> Result<Self> {
        Self::write(config, |name, count, seed| {
            let scale = if name.contains("norm") { 0.1 } else { 0.5 };
            values(count, seed, scale)
        })
    }

    fn write(config: &str, weights: impl Fn(&str, usize, u64) -> Vec<f32>) -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("tur-adapter-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir)?;
        let fixture = Self(dir);
        std::fs::write(fixture.0.join("config.json"), config)?;
        let parsed = Config::from_json(config)?;
        let parsed = if parsed.supports_mtp() {
            parsed.with_mtp()?
        } else {
            parsed
        };
        let mut tensors = HashMap::new();
        for (seed, (name, shape)) in parsed.weight_shapes()?.into_iter().enumerate() {
            let values = weights(&name, shape.iter().product(), seed as u64 + 1);
            tensors.insert(
                name,
                Tensor::from_vec(values, shape, &Device::Cpu)?.to_dtype(DType::BF16)?,
            );
        }
        candle_core::safetensors::save(&tensors, fixture.0.join("model.safetensors"))?;
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

    pub(crate) fn factory(&self) -> ModelFactory {
        ModelFactory {
            source: ModelSource::LocalPath(self.0.clone()),
            revision: "tiny-adapter-fixture-v1".into(),
            format: Some(WeightFormat::F32),
            device: Device::Cpu,
            // 16-dimension heads take unscaled formats only.
            kv: KvLayout::new(16, KvFormat::F32),
            recurrent: DType::F32,
        }
    }
}
