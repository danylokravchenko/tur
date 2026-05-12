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
use candle_nn::{Embedding, Module};
use serde::Deserialize;
use std::{collections::HashMap, fmt::Debug};
use tracing::{debug, error, trace};

use crate::{
    models::layers::{
        embedding, linear,
        mamba::MambaLinearAttention,
        norm::{NormX, rms_norm},
    },
    weights::VarBuilderX,
};

/// Qwen 3.5 Model Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Head dimension (if not specified, derived from hidden_size / num_attention_heads)
    #[serde(default)]
    pub head_dim: Option<usize>,
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
    /// Attention output gating
    #[serde(default = "default_attn_output_gate")]
    pub attn_output_gate: Option<bool>,
    /// Whether to tie word embeddings
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

fn default_attn_output_gate() -> Option<bool> {
    Some(true)
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
            num_key_value_heads: None,
            head_dim: None,
            num_hidden_layers: 24,
            intermediate_size: 5504,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_position_embeddings: 32768,
            attn_output_gate: Some(true),
            tie_word_embeddings: Some(false),
        }
    }
}

/// Attention mechanism variants
pub enum AttentionMechanism {
    /// Standard multi-head self-attention
    SelfAttention(Attention),
    /// Mamba linear attention (selective state space model)
    MambaLinearAttention(MambaLinearAttention),
}

impl Debug for AttentionMechanism {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionMechanism::SelfAttention(attn) => write!(f, "SelfAttention({:?})", attn),
            AttentionMechanism::MambaLinearAttention(attn) => {
                write!(f, "MambaLinearAttention({:?})", attn)
            }
        }
    }
}

impl AttentionMechanism {
    /// Forward pass through the attention mechanism
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            AttentionMechanism::SelfAttention(attn) => attn.forward(x),
            AttentionMechanism::MambaLinearAttention(attn) => attn.forward(x),
        }
    }
}

/// Multi-Head Self-Attention Layer with Attention Output Gating
#[derive(Debug)]
pub struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    q_proj_size: usize,
    kv_proj_size: usize,
    attn_output_gate: bool,
}

impl Attention {
    pub fn new(config: &Qwen3_5Config, vb: VarBuilderX) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);

        // Use explicit head_dim from config, or derive it
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_heads);

        // Qwen3.5 uses attention output gating by default
        let attn_output_gate = config.attn_output_gate.unwrap_or(true);

        // Detect the actual projection sizes from weight shapes
        let q_proj_size = if let Some(shape) = vb.pp("q_proj").tensor_shape("weight") {
            // For GGUF: shape is [out_dim, in_dim]
            shape[0]
        } else {
            // For SafeTensors: use the expected size based on config
            num_heads * head_dim * if attn_output_gate { 2 } else { 1 }
        };

        let kv_proj_size = if let Some(shape) = vb.pp("k_proj").tensor_shape("weight") {
            // For GGUF: shape is [out_dim, in_dim]
            shape[0]
        } else {
            // For SafeTensors: k and v use num_kv_heads, not num_heads
            num_kv_heads * head_dim
        };

        // Create the linear layers with the detected projection sizes
        let q_proj = linear(hidden_size, q_proj_size, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, kv_proj_size, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, kv_proj_size, vb.pp("v_proj"))?;

        // o_proj input size is always num_heads * head_dim (after gating is applied)
        let o_proj_in_size = num_heads * head_dim;
        let o_proj = linear(o_proj_in_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            q_proj_size,
            kv_proj_size,
            attn_output_gate,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = x.dims3()?;

        // Project queries, keys, values
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Handle attention output gating for queries
        let (q, q_gate) = if self.attn_output_gate {
            // Split q into query and gate: [batch, seq, num_heads * head_dim * 2]
            // -> [batch, seq, num_heads * head_dim] and [batch, seq, num_heads * head_dim]
            let half_size = self.q_proj_size / 2;
            let q_actual = q.narrow(2, 0, half_size)?;
            let q_gate = q.narrow(2, half_size, half_size)?;
            (q_actual, Some(q_gate))
        } else {
            (q, None)
        };

        // Calculate number of KV heads from kv_proj_size
        let num_kv_heads = self.kv_proj_size / self.head_dim;

        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, num_kv_heads, self.head_dim))?;

        // Transpose to (batch, heads, seq_len, head_dim)
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;

        // Handle GQA: repeat KV heads to match Q heads if needed
        let (k, v) = if num_kv_heads < self.num_heads {
            let repeat_factor = self.num_heads / num_kv_heads;
            // Repeat each KV head repeat_factor times
            // Shape: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
            // We need to repeat each head repeat_factor times: [h1, h1, h1, h2, h2, h2, ...]
            let k = k
                .unsqueeze(2)? // (batch, num_kv_heads, 1, seq_len, head_dim)
                .repeat(&[1, 1, repeat_factor, 1, 1])? // (batch, num_kv_heads, repeat_factor, seq_len, head_dim)
                .reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
            let v = v
                .unsqueeze(2)? // (batch, num_kv_heads, 1, seq_len, head_dim)
                .repeat(&[1, 1, repeat_factor, 1, 1])? // (batch, num_kv_heads, repeat_factor, seq_len, head_dim)
                .reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
            (k, v)
        } else {
            (k, v)
        };

        // Compute attention scores
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores / ((self.head_dim as f64).sqrt()))?;

        // Apply softmax
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, num_heads * head_dim)
        let attn_output = attn_output.permute((0, 2, 1, 3))?;
        let mut attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Apply gating if enabled
        if let Some(gate) = q_gate {
            // Reshape gate to match attention output dimensions
            // gate shape: [batch, seq, half_size] -> [batch, seq, num_heads * head_dim]
            let gate = if gate.dim(2)? != self.num_heads * self.head_dim {
                // If gate size doesn't match, reshape it properly
                gate.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
                    .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?
            } else {
                gate
            };

            // Apply sigmoid to gate and multiply with attention output
            let gate = candle_nn::ops::sigmoid(&gate)?;
            attn_output = (attn_output * gate)?;
        }

        // Output projection
        self.o_proj.forward(&attn_output)
    }
}

/// Feed-Forward Network (MLP)
pub struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilderX) -> Result<Self> {
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
    attention: AttentionMechanism,
    mlp: Mlp,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl std::fmt::Debug for DecoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderLayer")
            .field("attention", &self.attention)
            .finish()
    }
}

impl DecoderLayer {
    pub fn new(config: &Qwen3_5Config, vb: VarBuilderX) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        // Determine which attention mechanism to use based on available keys
        let attention = if vb.has_key("linear_attn.A_log") {
            // Mamba linear attention
            let linear_attn = MambaLinearAttention::new(config.hidden_size, vb.pp("linear_attn"))?;
            AttentionMechanism::MambaLinearAttention(linear_attn)
        } else {
            // Standard self-attention
            let self_attn = Attention::new(config, vb.pp("self_attn"))?;
            AttentionMechanism::SelfAttention(self_attn)
        };

        let mlp = Mlp::new(
            config.hidden_size,
            config.intermediate_size,
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("mlp").clone()
            },
        )?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("attn_norm").clone()
            } else {
                vb.pp("input_layernorm").clone()
            },
            DType::F32,
            !is_qvar_builder,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("post_attention_norm").clone()
            } else {
                vb.pp("post_attention_layernorm").clone()
            },
            DType::F32,
            !is_qvar_builder,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Apply input layer norm
        let normalized = self.input_layernorm.forward(hidden_states)?;

        // Apply attention mechanism with residual connection
        let attn_output = self.attention.forward(&normalized)?;
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
    norm: NormX,
    lm_head: candle_nn::Linear,
    config: Qwen3_5Config,
    device: Device,
}

impl Qwen3_5ForCausalLM {
    /// Create a new Qwen 3.5 model with the given configuration
    pub fn new(
        config: Qwen3_5Config,
        vb: VarBuilderX,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_prefix(config, vb, dtype, device, None)
    }

    pub fn new_with_prefix(
        config: Qwen3_5Config,
        vb: VarBuilderX,
        dtype: DType,
        device: &Device,
        prefix: Option<String>,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let (prefix, gguf_prefix) = Self::determine_prefix(&vb, prefix, is_qvar_builder);
        let key_map = Self::create_key_map();
        let tie_word_embeddings = Self::should_tie_embeddings(
            &config,
            &vb,
            &prefix,
            &gguf_prefix,
            &key_map,
            is_qvar_builder,
        );

        let embed_tokens = Self::create_embeddings(
            &config,
            &vb,
            &prefix,
            &gguf_prefix,
            &key_map,
            dtype,
            is_qvar_builder,
        )?;
        let layers = Self::create_layers(&config, &vb, &prefix)?;
        let norm = Self::create_norm(
            &config,
            &vb,
            &prefix,
            &gguf_prefix,
            &key_map,
            is_qvar_builder,
        )?;
        let lm_head = Self::create_lm_head(
            &config,
            &vb,
            &prefix,
            &gguf_prefix,
            &key_map,
            tie_word_embeddings,
            is_qvar_builder,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            device: device.clone(),
        })
    }

    /// Determine the correct prefix for weight keys
    fn determine_prefix(
        vb: &VarBuilderX,
        prefix: Option<String>,
        is_qvar_builder: bool,
    ) -> (String, String) {
        let has_prefix = prefix.is_some();

        let prefix = if !is_qvar_builder {
            if vb.has_key("model.language_model.embed_tokens.weight") {
                trace!(
                    "Found 'model.language_model.embed_tokens.weight' - using prefix 'model.language_model.'"
                );
                "model.language_model.".to_string()
            } else if vb.has_key("model.embed_tokens.weight") {
                trace!("Found 'model.embed_tokens.weight' - using prefix 'model.'");
                "model.".to_string()
            } else if vb.has_key("embed_tokens.weight") {
                trace!("Found 'embed_tokens.weight' - using no prefix");
                "".to_string()
            } else {
                trace!("No standard keys found - using default prefix");
                prefix.unwrap_or("model.".to_string())
            }
        } else {
            prefix.unwrap_or_default()
        };

        debug!("Using prefix: '{}'", prefix);
        let gguf_prefix = if has_prefix {
            prefix.clone()
        } else {
            "".to_string()
        };

        (prefix, gguf_prefix)
    }

    /// Create key mapping for GGUF format
    fn create_key_map() -> HashMap<&'static str, &'static str> {
        [
            ("embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("norm", "output_norm"),
            ("layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect()
    }

    /// Determine if word embeddings should be tied
    fn should_tie_embeddings(
        config: &Qwen3_5Config,
        vb: &VarBuilderX,
        prefix: &str,
        gguf_prefix: &str,
        key_map: &HashMap<&str, &str>,
        is_qvar_builder: bool,
    ) -> bool {
        if let Some(tie) = config.tie_word_embeddings {
            return tie;
        }

        if !is_qvar_builder
            && vb.has_key("embed_tokens.weight")
            && !vb.has_key(&format!("{}embed_tokens.weight", prefix))
        {
            error!("This model does not support decoding!");
            return true;
        }

        if is_qvar_builder {
            !vb.has_key(&format!("{}{}.weight", gguf_prefix, key_map["lm_head"]))
        } else {
            !vb.has_key("lm_head.weight") && !vb.has_key(&format!("{}lm_head.weight", prefix))
        }
    }

    /// Create token embeddings
    fn create_embeddings(
        config: &Qwen3_5Config,
        vb: &VarBuilderX,
        prefix: &str,
        gguf_prefix: &str,
        key_map: &HashMap<&str, &str>,
        dtype: DType,
        is_qvar_builder: bool,
    ) -> Result<Embedding> {
        let (embed_tokens, _vocab_size) = embedding(
            Some(config.vocab_size),
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            dtype,
        )?;
        Ok(embed_tokens)
    }

    /// Create decoder layers
    fn create_layers(
        config: &Qwen3_5Config,
        vb: &VarBuilderX,
        prefix: &str,
    ) -> Result<Vec<DecoderLayer>> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DecoderLayer::new(config, vb.pp(&format!("{}layers.{}", prefix, i)))?;
            layers.push(layer);
        }
        Ok(layers)
    }

    /// Create output normalization layer
    fn create_norm(
        config: &Qwen3_5Config,
        vb: &VarBuilderX,
        prefix: &str,
        gguf_prefix: &str,
        key_map: &HashMap<&str, &str>,
        is_qvar_builder: bool,
    ) -> Result<NormX> {
        rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["norm"]))
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            DType::F32,
            !is_qvar_builder,
        )
    }

    /// Create language model head
    fn create_lm_head(
        config: &Qwen3_5Config,
        vb: &VarBuilderX,
        prefix: &str,
        gguf_prefix: &str,
        key_map: &HashMap<&str, &str>,
        tie_word_embeddings: bool,
        is_qvar_builder: bool,
    ) -> Result<candle_nn::Linear> {
        linear(
            config.hidden_size,
            config.vocab_size,
            if tie_word_embeddings {
                if is_qvar_builder {
                    vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
                } else {
                    vb.pp(&format!("{}embed_tokens", prefix))
                }
            } else {
                if is_qvar_builder {
                    vb.pp(key_map["lm_head"])
                } else {
                    vb.pp("lm_head")
                }
            },
        )
    }

    /// Forward pass for token predictions
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Embed input tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through decoder layers
        for layer in &self.layers {
            // trace!("Forward pass through layer: {:?}", layer);
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

    /// Get the device the model is on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model configuration
    pub fn config(&self) -> &Qwen3_5Config {
        &self.config
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
