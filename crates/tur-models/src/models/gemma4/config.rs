//! Validated Gemma 4 text architecture and checkpoint tensor contract.
use crate::models::weights::{Projection, WeightLayout};
use candle_core::DType;
use serde::Deserialize;
use std::collections::BTreeMap;
use tur_core::{
    Result,
    execution::{
        KvLayout, LayerStateKind, Modality, ModelDescription, ModelIdentity, checked_elements,
        invalid,
    },
};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum LayerKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Rope {
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}
fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "fields mirror the checkpoint architecture"
)]
pub(super) struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_global_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_activation: String,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attention_k_eq_v: bool,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<LayerKind>,
    pub rope_parameters: BTreeMap<String, Rope>,
    pub sliding_window: usize,
    pub num_kv_shared_layers: usize,
    pub hidden_size_per_layer_input: usize,
    pub vocab_size_per_layer_input: usize,
    pub use_double_wide_mlp: bool,
    pub final_logit_softcapping: Option<f64>,
    pub use_bidirectional_attention: Option<String>,
    pub enable_moe_block: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
}

/// Settings belonging to the Gemma 4 text tower, validated before weights are read.
#[derive(Clone, Debug)]
pub struct Config {
    pub(super) text: TextConfig,
    pub(super) prefix: String,
    pub(super) forbidden: Vec<u32>,
    pub(super) eos: Vec<u32>,
    /// Each decoder layer's physical K/V owner; sharing layers allocate no extra state.
    pub(super) owners: Vec<usize>,
}

impl Config {
    /// Parse explicit Gemma 4 architecture identifiers and validate the implemented text equations.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        let (text, prefix, architecture) = match raw["model_type"].as_str() {
            Some("gemma4") => (
                &raw["text_config"],
                "model.language_model",
                "Gemma4ForConditionalGeneration",
            ),
            Some("gemma4_unified") => (
                &raw["text_config"],
                "model.language_model",
                "Gemma4UnifiedForConditionalGeneration",
            ),
            Some("gemma4_unified_text") => (&raw, "model", "Gemma4UnifiedForCausalLM"),
            Some("gemma4_text") => (&raw, "model", "Gemma4ForCausalLM"),
            _ => return Err(invalid("unsupported Gemma 4 architecture")),
        };
        if raw["architectures"]
            .as_array()
            .is_none_or(|a| a.len() != 1 || a[0].as_str() != Some(architecture))
        {
            return Err(invalid(
                "Gemma 4 architecture identifier disagrees with model_type",
            ));
        }
        for section in [&raw, text] {
            for key in ["quantization_config", "rope_scaling", "per_layer_config"] {
                if section.get(key).is_some_and(|v| !v.is_null()) {
                    return Err(invalid(format!("unsupported Gemma 4 {key}")));
                }
            }
        }
        let settings: TextConfig = serde_json::from_value(text.clone())?;
        let expected_type = if raw["model_type"]
            .as_str()
            .is_some_and(|s| s.starts_with("gemma4_unified"))
        {
            "gemma4_unified_text"
        } else {
            "gemma4_text"
        };
        if settings.model_type != expected_type {
            return Err(invalid(
                "Gemma 4 text type disagrees with outer architecture",
            ));
        }
        validate(&settings)?;
        if settings.model_type == "gemma4_unified_text"
            && (settings.hidden_size_per_layer_input > 0 || settings.enable_moe_block)
        {
            return Err(invalid(
                "unsupported Unified per-layer embedding or expert configuration",
            ));
        }
        if raw
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .is_some_and(|t| t != settings.tie_word_embeddings)
        {
            return Err(invalid("conflicting embedding tying settings"));
        }
        let eos =
            crate::models::generation::parse_tokens(&text["eos_token_id"], settings.vocab_size)?;
        let mut eos = eos;
        if let Some(ids) = raw.get("eos_token_id") {
            for id in crate::models::generation::parse_tokens(ids, settings.vocab_size)? {
                if !eos.contains(&id) {
                    eos.push(id);
                }
            }
        }
        let mut forbidden = Vec::new();
        for (key, value) in raw.as_object().into_iter().flatten() {
            if key.ends_with("_token_id")
                && ["image", "audio", "video", "boi", "eoi", "boa", "eoa"]
                    .iter()
                    .any(|m| key.starts_with(m))
            {
                forbidden.extend(crate::models::generation::parse_tokens(
                    value,
                    settings.vocab_size,
                )?);
            }
        }
        let first_shared = settings.num_hidden_layers - settings.num_kv_shared_layers;
        let mut owners = Vec::with_capacity(settings.num_hidden_layers);
        for (i, kind) in settings.layer_types.iter().enumerate() {
            let owner = if i < first_shared {
                i
            } else {
                settings.layer_types[..first_shared]
                    .iter()
                    .rposition(|k| k == kind)
                    .ok_or_else(|| invalid("shared K/V has no preceding layer of the same kind"))?
            };
            owners.push(owner);
        }
        let config = Self {
            text: settings,
            prefix: prefix.into(),
            forbidden,
            eos,
            owners,
        };
        config.weight_shapes()?;
        Ok(config)
    }

    /// Merge generation metadata with Gemma stop tokens.
    pub fn generation_defaults(
        &self,
        json: Option<&str>,
    ) -> Result<crate::models::GenerationDefaults> {
        crate::models::generation::defaults(&self.eos, self.vocab_size(), json)
    }

    /// Exclusive text position limit.
    pub fn max_position(&self) -> usize {
        self.text.max_position_embeddings
    }
    /// Padded output vocabulary width.
    pub fn vocab_size(&self) -> usize {
        self.text.vocab_size
    }
    /// Residual stream width.
    pub fn hidden_size(&self) -> usize {
        self.text.hidden_size
    }
    pub(super) fn dim(&self, i: usize) -> usize {
        if self.text.layer_types[i] == LayerKind::FullAttention {
            self.text.global_head_dim
        } else {
            self.text.head_dim
        }
    }
    pub(super) fn kv_heads(&self, i: usize) -> usize {
        if self.unified_kv(i) {
            self.text
                .num_global_key_value_heads
                .unwrap_or(self.text.num_key_value_heads)
        } else {
            self.text.num_key_value_heads
        }
    }
    pub(super) fn unified_kv(&self, i: usize) -> bool {
        self.text.attention_k_eq_v && self.text.layer_types[i] == LayerKind::FullAttention
    }
    pub(super) fn intermediate(&self, i: usize) -> usize {
        self.text.intermediate_size
            * if self.text.use_double_wide_mlp && self.owners[i] != i {
                2
            } else {
                1
            }
    }
    pub(super) fn rope(&self, i: usize) -> &Rope {
        &self.text.rope_parameters[if self.text.layer_types[i] == LayerKind::FullAttention {
            "full_attention"
        } else {
            "sliding_attention"
        }]
    }
    pub(super) fn state_layer(&self, i: usize) -> usize {
        self.owners[..self.owners[i]]
            .iter()
            .enumerate()
            .filter(|(j, owner)| **owner == *j)
            .count()
    }
    pub(super) fn description(&self, revision: String, kv: KvLayout) -> Result<ModelDescription> {
        let description = ModelDescription {
            identity: ModelIdentity {
                revision,
                state_format: 1,
            },
            layers: self
                .owners
                .iter()
                .enumerate()
                .filter(|(i, owner)| **owner == *i)
                .map(|(i, _)| LayerStateKind::Attention {
                    kv_heads: self.kv_heads(i),
                    head_dim: self.dim(i),
                })
                .collect(),
            kv,
            modalities: vec![Modality::Text],
            vocab_size: self.vocab_size(),
            max_position: self.max_position(),
        };
        description.validate()?;
        Ok(description)
    }

    /// Required text tensors and exact shapes; multimodal towers are not loaded.
    pub fn weight_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        let t = &self.text;
        let mut shapes = BTreeMap::new();
        shapes.insert(
            format!("{}.embed_tokens.weight", self.prefix),
            vec![t.vocab_size, t.hidden_size],
        );
        shapes.insert(format!("{}.norm.weight", self.prefix), vec![t.hidden_size]);
        if !t.tie_word_embeddings {
            shapes.insert("lm_head.weight".into(), vec![t.vocab_size, t.hidden_size]);
        }
        if t.hidden_size_per_layer_input > 0 {
            let width = checked_elements(&[t.num_hidden_layers, t.hidden_size_per_layer_input])?;
            shapes.insert(
                format!("{}.embed_tokens_per_layer.weight", self.prefix),
                vec![t.vocab_size_per_layer_input, width],
            );
            shapes.insert(
                format!("{}.per_layer_model_projection.weight", self.prefix),
                vec![width, t.hidden_size],
            );
            shapes.insert(
                format!("{}.per_layer_projection_norm.weight", self.prefix),
                vec![t.hidden_size_per_layer_input],
            );
        }
        for i in 0..t.num_hidden_layers {
            let p = format!("{}.layers.{i}", self.prefix);
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                shapes.insert(format!("{p}.{name}.weight"), vec![t.hidden_size]);
            }
            shapes.insert(format!("{p}.layer_scalar"), vec![1]);
            shapes.insert(format!("{p}.self_attn.q_norm.weight"), vec![self.dim(i)]);
            if self.owners[i] == i {
                shapes.insert(format!("{p}.self_attn.k_norm.weight"), vec![self.dim(i)]);
            }
            if t.hidden_size_per_layer_input > 0 {
                shapes.insert(
                    format!("{p}.post_per_layer_input_norm.weight"),
                    vec![t.hidden_size],
                );
            }
            if t.enable_moe_block {
                for name in [
                    "pre_feedforward_layernorm_2",
                    "post_feedforward_layernorm_1",
                    "post_feedforward_layernorm_2",
                    "router.scale",
                ] {
                    let name = if name == "router.scale" {
                        format!("{p}.{name}")
                    } else {
                        format!("{p}.{name}.weight")
                    };
                    shapes.insert(name, vec![t.hidden_size]);
                }
                shapes.insert(
                    format!("{p}.router.per_expert_scale"),
                    vec![t.num_experts.unwrap_or(0)],
                );
                shapes.insert(
                    format!("{p}.router.proj.weight"),
                    vec![t.num_experts.unwrap_or(0), t.hidden_size],
                );
            }
        }
        for p in self.projection_layout()? {
            for (name, rows) in p.parts {
                let shape = if p.experts == 1 {
                    vec![rows, p.cols]
                } else {
                    vec![p.experts, rows / p.experts, p.cols]
                };
                shapes.insert(name, shape);
            }
        }
        // A tied head's projection refers to the embedding's original tensor.
        for shape in shapes.values() {
            checked_elements(shape)?
                .checked_mul(DType::F32.size_in_bytes())
                .ok_or_else(|| invalid("weight bytes overflow"))?;
        }
        Ok(shapes)
    }
}

impl WeightLayout for Config {
    fn embeddings_tied(&self) -> bool {
        self.text.tie_word_embeddings
    }
    fn vocabulary(&self) -> usize {
        self.vocab_size()
    }
    fn residual_width(&self) -> usize {
        self.hidden_size()
    }
    fn projection_layout(&self) -> Result<Vec<Projection>> {
        let t = &self.text;
        let mut projections = Vec::new();
        let mut add = |name: String, rows: usize, cols: usize| {
            projections.push(Projection::new(
                name.trim_end_matches(".weight").into(),
                vec![(name, rows)],
                cols,
            ));
        };
        for i in 0..t.num_hidden_layers {
            let p = format!("{}.layers.{i}", self.prefix);
            add(
                format!("{p}.self_attn.q_proj.weight"),
                checked_elements(&[t.num_attention_heads, self.dim(i)])?,
                t.hidden_size,
            );
            add(
                format!("{p}.self_attn.o_proj.weight"),
                t.hidden_size,
                checked_elements(&[t.num_attention_heads, self.dim(i)])?,
            );
            if self.owners[i] == i {
                add(
                    format!("{p}.self_attn.k_proj.weight"),
                    checked_elements(&[self.kv_heads(i), self.dim(i)])?,
                    t.hidden_size,
                );
                if !self.unified_kv(i) {
                    add(
                        format!("{p}.self_attn.v_proj.weight"),
                        checked_elements(&[self.kv_heads(i), self.dim(i)])?,
                        t.hidden_size,
                    );
                }
            }
            for name in ["gate_proj", "up_proj"] {
                add(
                    format!("{p}.mlp.{name}.weight"),
                    self.intermediate(i),
                    t.hidden_size,
                );
            }
            add(
                format!("{p}.mlp.down_proj.weight"),
                t.hidden_size,
                self.intermediate(i),
            );
            if t.hidden_size_per_layer_input > 0 {
                add(
                    format!("{p}.per_layer_input_gate.weight"),
                    t.hidden_size_per_layer_input,
                    t.hidden_size,
                );
                add(
                    format!("{p}.per_layer_projection.weight"),
                    t.hidden_size,
                    t.hidden_size_per_layer_input,
                );
            }
        }
        if t.hidden_size_per_layer_input > 0 {
            add(
                format!("{}.per_layer_model_projection.weight", self.prefix),
                checked_elements(&[t.num_hidden_layers, t.hidden_size_per_layer_input])?,
                t.hidden_size,
            );
        }
        let head = if t.tie_word_embeddings {
            format!("{}.embed_tokens.weight", self.prefix)
        } else {
            "lm_head.weight".into()
        };
        projections.push(Projection::new(
            "lm_head".into(),
            vec![(head, t.vocab_size)],
            t.hidden_size,
        ));
        if t.enable_moe_block {
            let experts = t.num_experts.ok_or_else(|| invalid("missing experts"))?;
            let width = t
                .moe_intermediate_size
                .ok_or_else(|| invalid("missing expert width"))?;
            for i in 0..t.num_hidden_layers {
                let p = format!("{}.layers.{i}.experts", self.prefix);
                projections.push(Projection::bank(
                    format!("{p}.gate_up_proj"),
                    experts,
                    checked_elements(&[2, width])?,
                    t.hidden_size,
                ));
                projections.push(Projection::bank(
                    format!("{p}.down_proj"),
                    experts,
                    t.hidden_size,
                    width,
                ));
            }
        }
        Ok(projections)
    }
}

fn validate(t: &TextConfig) -> Result<()> {
    if [
        t.hidden_size,
        t.intermediate_size,
        t.vocab_size,
        t.num_hidden_layers,
        t.num_attention_heads,
        t.num_key_value_heads,
        t.head_dim,
        t.global_head_dim,
        t.max_position_embeddings,
        t.sliding_window,
    ]
    .contains(&0)
        || !matches!(t.model_type.as_str(), "gemma4_text" | "gemma4_unified_text")
        || t.layer_types.len() != t.num_hidden_layers
        || t.num_kv_shared_layers >= t.num_hidden_layers
        || !t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
        || t.num_global_key_value_heads
            .is_some_and(|h| h == 0 || !t.num_attention_heads.is_multiple_of(h))
        || !t.rms_norm_eps.is_finite()
        || t.rms_norm_eps <= 0.0
        || t.hidden_activation != "gelu_pytorch_tanh"
        || t.attention_bias
        || t.attention_dropout != 0.0
        || t.use_bidirectional_attention
            .as_deref()
            .is_some_and(|s| s != "vision")
        || t.final_logit_softcapping
            .is_some_and(|c| !c.is_finite() || c <= 0.0)
        || !t.head_dim.is_multiple_of(2)
        || !t.global_head_dim.is_multiple_of(2)
        || u32::try_from(t.vocab_size).is_err()
        || u32::try_from(t.max_position_embeddings).is_err()
        || (t.hidden_size_per_layer_input > 0 && t.vocab_size_per_layer_input < t.vocab_size)
    {
        return Err(invalid("unsupported or invalid Gemma 4 text settings"));
    }
    for kind in ["sliding_attention", "full_attention"] {
        let rope = t
            .rope_parameters
            .get(kind)
            .ok_or_else(|| invalid("missing Gemma 4 RoPE settings"))?;
        let dim = if kind == "full_attention" {
            t.global_head_dim
        } else {
            t.head_dim
        };
        let rotated = dim as f64 * rope.partial_rotary_factor;
        if !rope.rope_theta.is_finite()
            || rope.rope_theta <= 0.0
            || !rope.partial_rotary_factor.is_finite()
            || !(0.0..=1.0).contains(&rope.partial_rotary_factor)
            || rotated < 2.0
            || rotated.fract() != 0.0
            || !(rotated as usize).is_multiple_of(2)
            || (kind == "sliding_attention"
                && (rope.rope_type != "default" || rope.partial_rotary_factor < 1.0))
            || (kind == "full_attention" && rope.rope_type != "proportional")
        {
            return Err(invalid("unsupported Gemma 4 RoPE settings"));
        }
    }
    if t.enable_moe_block {
        if t.num_experts.is_none_or(|e| e == 0 || e > 256)
            || t.top_k_experts
                .is_none_or(|k| k == 0 || k > t.num_experts.unwrap_or(0))
            || t.moe_intermediate_size.is_none_or(|i| i == 0)
        {
            return Err(invalid("invalid Gemma 4 expert geometry"));
        }
    } else if [t.num_experts, t.top_k_experts, t.moe_intermediate_size]
        .iter()
        .any(Option::is_some)
    {
        return Err(invalid("expert settings on dense Gemma 4"));
    }
    checked_elements(&[t.intermediate_size, 2])?;
    Ok(())
}
