//! Validated configuration for the dense, text-only Qwen3.5 architecture.

use std::collections::BTreeMap;

use serde::Deserialize;
use candle_core::DType;
use tur_core::{Result, execution::{LayerStateKind, Modality, ModelDescription, ModelIdentity, checked_elements, invalid}};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum LayerKind { LinearAttention, FullAttention }

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Rope {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    pub attn_output_gate: bool,
    #[serde(default)]
    pub output_gate_type: Option<String>,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<LayerKind>,
    pub rope_parameters: Rope,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    pub eos_token_id: serde_json::Value,
    #[serde(default)]
    pub mamba_ssm_dtype: Option<String>,
}

/// Architecture-specific settings. Construction validates the complete text layout.
/// Vision and MTP tensors may exist in the checkpoint but are never executed.
#[derive(Clone, Debug)]
pub struct Config {
    pub(super) text: TextConfig,
    pub(super) prefix: String,
    pub(super) forbidden_tokens: Vec<u32>,
    eos: Vec<u32>,
}

pub(super) fn product(dims: &[usize]) -> Result<usize> { checked_elements(dims) }

impl Config {
    /// Parse explicit architecture identifiers and nested text settings, never a model name.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        let (text, prefix) = match (raw["model_type"].as_str(), raw["architectures"].as_array()) {
            (Some("qwen3_5"), Some(a)) if a == &[serde_json::json!("Qwen3_5ForConditionalGeneration")] =>
                (raw.get("text_config").ok_or_else(|| invalid("qwen3_5 requires text_config"))?, "model.language_model"),
            (Some("qwen3_5_text"), Some(a)) if a == &[serde_json::json!("Qwen3_5ForCausalLM")] => (&raw, "model"),
            _ => return Err(invalid("unsupported architecture: expected dense Qwen3_5 conditional generation or text causal LM")),
        };
        for settings in [&raw, text] {
            for key in ["quantization_config", "rope_scaling", "num_experts", "moe_intermediate_size", "sliding_window"] {
                if settings.get(key).is_some_and(|v| !v.is_null()) {
                    return Err(invalid(format!("unsupported Qwen3.5 setting: {key}")));
                }
            }
            if settings.get("mlp_only_layers").is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty())) {
                return Err(invalid("MLP-only layers are unsupported"));
            }
        }
        let text: TextConfig = serde_json::from_value(text.clone())?;
        if text.model_type != "qwen3_5_text" || text.hidden_act != "silu" || text.attention_bias
            || text.attention_dropout != 0.0 || !text.attn_output_gate
            || text.output_gate_type.as_deref().is_some_and(|s| s != "swish" && s != "silu")
            || text.mamba_ssm_dtype.as_deref().is_some_and(|s| s != "float32") {
            return Err(invalid("unsupported Qwen3.5 activation, gate, bias, dropout or state dtype"));
        }
        if raw.get("tie_word_embeddings").and_then(|v| v.as_bool()).is_some_and(|t| t != text.tie_word_embeddings) {
            return Err(invalid("conflicting embedding tying settings"));
        }
        let dims = [text.hidden_size, text.intermediate_size, text.vocab_size, text.num_hidden_layers,
            text.num_attention_heads, text.num_key_value_heads, text.head_dim, text.linear_num_key_heads,
            text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim,
            text.linear_conv_kernel_dim, text.max_position_embeddings];
        if dims.contains(&0) || text.layer_types.len() != text.num_hidden_layers
            || text.num_attention_heads % text.num_key_value_heads != 0
            || text.linear_num_value_heads % text.linear_num_key_heads != 0
            || !text.rms_norm_eps.is_finite() || text.rms_norm_eps <= 0.0
            || text.vocab_size > u32::MAX as usize || text.max_position_embeddings > u32::MAX as usize {
            return Err(invalid("invalid Qwen3.5 dimensions, heads, layer list or normalization"));
        }
        let rope = &text.rope_parameters;
        let rotary = text.head_dim as f64 * rope.partial_rotary_factor;
        if rope.rope_type != "default" || !rope.rope_theta.is_finite() || rope.rope_theta <= 0.0
            || !rotary.is_finite() || rotary < 2.0 || rotary > text.head_dim as f64
            || rotary.fract() != 0.0 || rotary as usize % 2 != 0
            || text.partial_rotary_factor.is_some_and(|f| f != rope.partial_rotary_factor) {
            return Err(invalid("unsupported or invalid partial RoPE configuration"));
        }
        if !rope.mrope_section.is_empty() {
            let sum = rope.mrope_section.iter().try_fold(0usize, |s, &n| s.checked_add(n).ok_or_else(|| invalid("mrope section overflow")))?;
            if !rope.mrope_interleaved || rope.mrope_section.len() != 3 || sum != rotary as usize / 2 {
                return Err(invalid("unsupported M-RoPE sections"));
            }
        }
        let eos = parse_eos(&text.eos_token_id, text.vocab_size)?;
        let mut forbidden_tokens = Vec::new();
        for field in ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id", "audio_token_id"] {
            if let Some(id) = raw.get(field).and_then(|v| v.as_u64()) {
                forbidden_tokens.push(u32::try_from(id).map_err(|_| invalid("modality token overflow"))?);
            }
        }
        let config = Self { text, prefix: prefix.into(), forbidden_tokens, eos };
        config.description("validation".into())?.validate()?;
        // Validate every weight product before any file payload is read.
        config.weight_shapes()?;
        Ok(config)
    }

    /// Exclusive text positional limit.
    pub fn max_position(&self) -> usize { self.text.max_position_embeddings }
    /// Padded projection vocabulary size (may exceed tokenizer vocabulary).
    pub fn vocab_size(&self) -> usize { self.text.vocab_size }
    /// EOS IDs declared by the architecture; generation config may override these.
    pub fn eos_tokens(&self) -> &[u32] { &self.eos }

    pub(super) fn key_width(&self) -> Result<usize> { product(&[self.text.linear_num_key_heads, self.text.linear_key_head_dim]) }
    pub(super) fn value_width(&self) -> Result<usize> { product(&[self.text.linear_num_value_heads, self.text.linear_value_head_dim]) }
    pub(super) fn conv_width(&self) -> Result<usize> {
        self.key_width()?.checked_mul(2).and_then(|n| n.checked_add(self.value_width().ok()?)).ok_or_else(|| invalid("convolution channels overflow"))
    }
    pub(super) fn rotary_dim(&self) -> usize { (self.text.head_dim as f64 * self.text.rope_parameters.partial_rotary_factor) as usize }

    pub(super) fn description(&self, revision: String) -> Result<ModelDescription> {
        let t = &self.text;
        let layers = t.layer_types.iter().map(|k| Ok(match k {
            LayerKind::FullAttention => LayerStateKind::Attention { kv_heads: t.num_key_value_heads, head_dim: t.head_dim, page_tokens: 16, dtype: DType::F32 },
            LayerKind::LinearAttention => LayerStateKind::Recurrent { conv_channels: self.conv_width()?, conv_width: t.linear_conv_kernel_dim,
                heads: t.linear_num_value_heads, key_dim: t.linear_key_head_dim, value_dim: t.linear_value_head_dim, dtype: DType::F32 },
        })).collect::<Result<_>>()?;
        Ok(ModelDescription { identity: ModelIdentity { revision, state_format: 1 }, layers, modalities: vec![Modality::Text], vocab_size: t.vocab_size, max_position: t.max_position_embeddings })
    }

    /// Exact SafeTensors weight names and dimensions required by the text tower.
    pub fn weight_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        let t = &self.text;
        let mut shapes = BTreeMap::new();
        shapes.insert(format!("{}.embed_tokens.weight", self.prefix), vec![t.vocab_size, t.hidden_size]);
        shapes.insert(format!("{}.norm.weight", self.prefix), vec![t.hidden_size]);
        if !t.tie_word_embeddings { shapes.insert("lm_head.weight".into(), vec![t.vocab_size, t.hidden_size]); }
        for (i, kind) in t.layer_types.iter().enumerate() {
            let p = format!("{}.layers.{i}", self.prefix);
            for n in ["input_layernorm", "post_attention_layernorm"] { shapes.insert(format!("{p}.{n}.weight"), vec![t.hidden_size]); }
            for n in ["gate_proj", "up_proj"] { shapes.insert(format!("{p}.mlp.{n}.weight"), vec![t.intermediate_size, t.hidden_size]); }
            shapes.insert(format!("{p}.mlp.down_proj.weight"), vec![t.hidden_size, t.intermediate_size]);
            match kind {
                LayerKind::FullAttention => {
                    let q = product(&[t.num_attention_heads, t.head_dim])?;
                    let kv = product(&[t.num_key_value_heads, t.head_dim])?;
                    for (n, shape) in [("q_proj.weight", vec![product(&[2, q])?, t.hidden_size]), ("k_proj.weight", vec![kv, t.hidden_size]),
                        ("v_proj.weight", vec![kv, t.hidden_size]), ("o_proj.weight", vec![t.hidden_size, q]),
                        ("q_norm.weight", vec![t.head_dim]), ("k_norm.weight", vec![t.head_dim])] {
                        shapes.insert(format!("{p}.self_attn.{n}"), shape);
                    }
                }
                LayerKind::LinearAttention => {
                    for (n, shape) in [("in_proj_qkv.weight", vec![self.conv_width()?, t.hidden_size]),
                        ("in_proj_z.weight", vec![self.value_width()?, t.hidden_size]),
                        ("in_proj_a.weight", vec![t.linear_num_value_heads, t.hidden_size]),
                        ("in_proj_b.weight", vec![t.linear_num_value_heads, t.hidden_size]),
                        ("out_proj.weight", vec![t.hidden_size, self.value_width()?]),
                        ("conv1d.weight", vec![self.conv_width()?, 1, t.linear_conv_kernel_dim]),
                        ("A_log", vec![t.linear_num_value_heads]), ("dt_bias", vec![t.linear_num_value_heads]),
                        ("norm.weight", vec![t.linear_value_head_dim])] {
                        shapes.insert(format!("{p}.linear_attn.{n}"), shape);
                    }
                }
            }
        }
        for shape in shapes.values() { product(shape)?.checked_mul(4).ok_or_else(|| invalid("weight bytes overflow"))?; }
        Ok(shapes)
    }
}

pub(super) fn parse_eos(raw: &serde_json::Value, vocab: usize) -> Result<Vec<u32>> {
    let values = if let Some(a) = raw.as_array() { a.clone() } else { vec![raw.clone()] };
    if values.is_empty() { return Err(invalid("empty EOS list")); }
    values.iter().map(|v| {
        let id = v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| invalid("invalid EOS ID"))?;
        if id as usize >= vocab { return Err(invalid("EOS outside vocabulary")); }
        Ok(id)
    }).collect()
}
