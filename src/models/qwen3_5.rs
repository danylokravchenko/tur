//! Clean Qwen 3.5 Model Implementation
//!
//! A simplified yet functional implementation of Qwen 3.5 causal language model.
//! This module provides the core components for inference:
//! - Configuration management
//! - Token embedding layer
//! - Decoder layers with attention and MLP
//! - Output normalization and LM head
//! - Forward pass for token generation

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding, linear};
use serde::Deserialize;
use std::sync::Arc;

/// Qwen 3.5 Model Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of decoder layers
    pub num_hidden_layers: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// Activation function type
    pub hidden_act: String,
    /// RMS normalization epsilon
    pub rms_norm_eps: f64,
    /// Rope theta for rotary embeddings
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
}

fn default_rope_theta() -> f64 {
    1000000.0
}

impl Default for Qwen3_5Config {
    fn default() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 2048,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            intermediate_size: 5504,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_position_embeddings: 32768,
        }
    }
}

/// RMS Layer Normalization
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((size,), "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = ((x * x)?.mean_keepdim(2)? + self.eps)?;
        let norm_x = norm_x.sqrt()?;
        (x / norm_x)? * &self.weight
    }
}

/// Multi-Head Self-Attention Layer
pub struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let q_proj = linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;

        // Project queries, keys, values
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Transpose to (batch, heads, seq_len, head_dim)
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;

        // Compute attention scores
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores / ((self.head_dim as f64).sqrt()))?;

        // Apply softmax (simplified - in practice use proper softmax)
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, hidden_size)
        let attn_output = attn_output.permute((0, 2, 1, 3))?;
        let attn_output = attn_output.reshape((batch_size, seq_len, hidden_size))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }
}

/// Feed-Forward Network (MLP)
pub struct MLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl MLP {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU activation: x * sigmoid(gate(x)) * up(x)
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::sigmoid(&gate)?;
        let up = self.up_proj.forward(x)?;

        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

/// Decoder Layer combining attention and MLP
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(config: &Qwen3_5Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(
            config.hidden_size,
            config.num_attention_heads,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(config.hidden_size, config.intermediate_size, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Apply input layer norm
        let normalized = self.input_layernorm.forward(hidden_states)?;

        // Self-attention with residual connection
        let attn_output = self.self_attn.forward(&normalized)?;
        let hidden_states = (hidden_states + attn_output)?;

        // Apply post-attention layer norm
        let normalized = self.post_attention_layernorm.forward(&hidden_states)?;

        // MLP with residual connection
        let mlp_output = self.mlp.forward(&normalized)?;
        hidden_states + mlp_output
    }
}

/// Qwen 3.5 Causal Language Model
pub struct Qwen3_5ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    config: Qwen3_5Config,
    device: Device,
}

impl Qwen3_5ForCausalLM {
    /// Create a new Qwen 3.5 model with the given configuration
    pub fn new(config: Qwen3_5Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = DecoderLayer::new(&config, vb.pp(format!("model.layers.{}", i)))?;
            layers.push(layer);
        }

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            device: device.clone(),
        })
    }

    /// Forward pass for token predictions
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Embed input tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through decoder layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Apply final normalization
        let hidden_states = self.norm.forward(&hidden_states)?;

        // Project to vocabulary size
        self.lm_head.forward(&hidden_states)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Qwen3_5Config::default();
        assert_eq!(config.vocab_size, 152064);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 24);
    }
}
