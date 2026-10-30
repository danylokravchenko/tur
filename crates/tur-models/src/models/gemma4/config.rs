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
    /// The embedding row Transformers looks up at multimodal positions (default 0).
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
}

/// The USM-style conformer audio encoder of Gemma 4 E2B and E4B.
#[derive(Clone, Debug, Deserialize)]
#[allow(
    clippy::struct_field_names,
    reason = "fields mirror the checkpoint architecture"
)]
pub(super) struct AudioConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub subsampling_conv_channels: Vec<usize>,
    pub output_proj_dims: usize,
    pub conv_kernel_size: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    pub attention_logit_cap: f64,
    pub attention_invalid_logits_value: f64,
    pub residual_weight: f64,
    pub gradient_clipping: f64,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub use_clipped_linears: bool,
}

/// The audio tower's settings and the prompt tokens that frame its soft tokens.
#[derive(Clone, Debug)]
pub(super) struct Audio {
    pub config: AudioConfig,
    /// `<|audio|>`: one per soft token.
    pub token: u32,
    /// `<|audio>` and `<audio|>` around a clip's soft tokens.
    pub begin: u32,
    pub end: u32,
}

impl Audio {
    /// Mel bins of the feature extractor (Transformers' `Gemma4AudioFeatureExtractor`).
    pub(super) const MELS: usize = 128;
    /// Soft tokens of one clip at most: 30 s at 40 ms each (`audio_seq_length`).
    pub(super) const MAX_TOKENS: usize = 750;

    fn parse(raw: &serde_json::Value, vocab: usize) -> Result<Option<Self>> {
        let Some(section) = raw.get("audio_config").filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        let config: AudioConfig = serde_json::from_value(section.clone())?;
        let heads = config.num_attention_heads;
        if section["model_type"].as_str() != Some("gemma4_audio")
            || config.hidden_act != "silu"
            || heads == 0
            || config.hidden_size == 0
            || !config.hidden_size.is_multiple_of(heads)
            || config.num_hidden_layers == 0
            || config.subsampling_conv_channels.len() != 2
            || config.subsampling_conv_channels.contains(&0)
            || config.conv_kernel_size == 0
            || config.output_proj_dims == 0
            // Relative positions then index exactly the causal window (see `audio.rs`).
            || config.attention_context_right != 0
            || config.attention_context_left < 2
            || config.attention_chunk_size + 1 != config.attention_context_left
            || [config.attention_logit_cap, config.rms_norm_eps, config.gradient_clipping]
                .iter()
                .any(|v| v.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater))
            || !config.residual_weight.is_finite()
        {
            return Err(invalid("unsupported Gemma 4 audio encoder configuration"));
        }
        let token = |key: &str| -> Result<u32> {
            let id = raw[key]
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| (id as usize) < vocab)
                .ok_or_else(|| invalid(format!("Gemma 4 audio needs a valid {key}")))?;
            Ok(id)
        };
        Ok(Some(Self {
            config,
            token: token("audio_token_id")?,
            begin: token("boa_token_id")?,
            end: raw
                .get("eoa_token_id")
                .filter(|v| !v.is_null())
                .map_or_else(|| token("eoa_token_index"), |_| token("eoa_token_id"))?,
        }))
    }

    /// Head width of the encoder's attention.
    pub(super) fn head_dim(&self) -> usize {
        self.config.hidden_size / self.config.num_attention_heads
    }

    /// Checkpoint tensors of the tower and its projector into the text model, with shapes.
    #[allow(
        clippy::too_many_lines,
        reason = "one entry per tensor of the encoder's fixed layout"
    )]
    pub(super) fn weight_shapes(&self, text_hidden: usize) -> BTreeMap<String, Vec<usize>> {
        let c = &self.config;
        let (h, [c0, c1]) = (
            c.hidden_size,
            [
                c.subsampling_conv_channels[0],
                c.subsampling_conv_channels[1],
            ],
        );
        let mut shapes = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            shapes.insert(name, shape);
        };
        let tower = "model.audio_tower";
        let sub = format!("{tower}.subsample_conv_projection");
        add(format!("{sub}.layer0.conv.weight"), vec![c0, 1, 3, 3]);
        add(format!("{sub}.layer0.norm.weight"), vec![c0]);
        add(format!("{sub}.layer1.conv.weight"), vec![c1, c0, 3, 3]);
        add(format!("{sub}.layer1.norm.weight"), vec![c1]);
        add(
            format!("{sub}.input_proj_linear.weight"),
            vec![h, Self::MELS / 4 * c1],
        );
        let clipped = |add: &mut dyn FnMut(String, Vec<usize>), name: String, rows, cols| {
            add(format!("{name}.linear.weight"), vec![rows, cols]);
            if c.use_clipped_linears {
                for bound in ["input_min", "input_max", "output_min", "output_max"] {
                    add(format!("{name}.{bound}"), Vec::new());
                }
            }
        };
        for i in 0..c.num_hidden_layers {
            let layer = format!("{tower}.layers.{i}");
            for ff in ["feed_forward1", "feed_forward2"] {
                clipped(&mut add, format!("{layer}.{ff}.ffw_layer_1"), 4 * h, h);
                clipped(&mut add, format!("{layer}.{ff}.ffw_layer_2"), h, 4 * h);
                add(format!("{layer}.{ff}.pre_layer_norm.weight"), vec![h]);
                add(format!("{layer}.{ff}.post_layer_norm.weight"), vec![h]);
            }
            for p in ["q_proj", "k_proj", "v_proj", "post"] {
                clipped(&mut add, format!("{layer}.self_attn.{p}"), h, h);
            }
            add(
                format!("{layer}.self_attn.relative_k_proj.weight"),
                vec![h, h],
            );
            add(
                format!("{layer}.self_attn.per_dim_scale"),
                vec![self.head_dim()],
            );
            clipped(&mut add, format!("{layer}.lconv1d.linear_start"), 2 * h, h);
            clipped(&mut add, format!("{layer}.lconv1d.linear_end"), h, h);
            add(
                format!("{layer}.lconv1d.depthwise_conv1d.weight"),
                vec![h, 1, c.conv_kernel_size],
            );
            for norm in ["lconv1d.pre_layer_norm", "lconv1d.conv_norm"] {
                add(format!("{layer}.{norm}.weight"), vec![h]);
            }
            for norm in ["norm_pre_attn", "norm_post_attn", "norm_out"] {
                add(format!("{layer}.{norm}.weight"), vec![h]);
            }
        }
        add(
            format!("{tower}.output_proj.weight"),
            vec![c.output_proj_dims, h],
        );
        add(
            format!("{tower}.output_proj.bias"),
            vec![c.output_proj_dims],
        );
        add(
            "model.embed_audio.embedding_projection.weight".into(),
            vec![text_hidden, c.output_proj_dims],
        );
        shapes
    }
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
    /// The audio encoder, when the checkpoint declares one.
    pub(super) audio: Option<Audio>,
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
        let audio = Audio::parse(&raw, settings.vocab_size)?;
        let config = Self {
            text: settings,
            prefix: prefix.into(),
            forbidden,
            eos,
            owners,
            audio,
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

    /// Whether the checkpoint declares an audio encoder.
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
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
    /// Whether layer `i` reads a sliding window, and so stores keys in window pages.
    pub(super) fn windowed(&self, i: usize) -> bool {
        self.text.layer_types[i] == LayerKind::SlidingAttention
    }
    pub(super) fn state_layer(&self, i: usize) -> usize {
        self.owners[..self.owners[i]]
            .iter()
            .enumerate()
            .filter(|(j, owner)| **owner == *j)
            .count()
    }
    /// The model's state and limits; `audio` declares that the encoder is loaded.
    pub(super) fn description(
        &self,
        revision: String,
        kv: KvLayout,
        audio: bool,
    ) -> Result<ModelDescription> {
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
                    // One key beyond the window: the assistant drafts from a sliding
                    // window of `sliding_window + 1` target keys.
                    window: self.windowed(i).then_some(self.text.sliding_window + 1),
                })
                .collect(),
            kv,
            modalities: if audio {
                vec![Modality::Text, Modality::Audio]
            } else {
                vec![Modality::Text]
            },
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
        let single = |name: String, rows: usize, cols: usize| {
            Projection::new(
                name.trim_end_matches(".weight").into(),
                vec![(name, rows)],
                cols,
            )
        };
        for i in 0..t.num_hidden_layers {
            let p = format!("{}.layers.{i}", self.prefix);
            let a = format!("{p}.self_attn");
            let query = checked_elements(&[t.num_attention_heads, self.dim(i)])?;
            let kv = checked_elements(&[self.kv_heads(i), self.dim(i)])?;
            // K/V owners project `q | k | v` (or `q | k` when values reuse keys) in one
            // matrix; sharing layers project queries only.
            let mut parts = vec![(format!("{a}.q_proj.weight"), query)];
            if self.owners[i] == i {
                parts.push((format!("{a}.k_proj.weight"), kv));
                if !self.unified_kv(i) {
                    parts.push((format!("{a}.v_proj.weight"), kv));
                }
            }
            let name = if parts.len() == 1 {
                "q_proj"
            } else {
                "qkv_proj"
            };
            projections.push(Projection::new(format!("{a}.{name}"), parts, t.hidden_size));
            projections.push(single(format!("{a}.o_proj.weight"), t.hidden_size, query));
            projections.push(Projection::new(
                format!("{p}.mlp.gate_up_proj"),
                vec![
                    (format!("{p}.mlp.gate_proj.weight"), self.intermediate(i)),
                    (format!("{p}.mlp.up_proj.weight"), self.intermediate(i)),
                ],
                t.hidden_size,
            ));
            projections.push(single(
                format!("{p}.mlp.down_proj.weight"),
                t.hidden_size,
                self.intermediate(i),
            ));
            if t.hidden_size_per_layer_input > 0 {
                projections.push(single(
                    format!("{p}.per_layer_input_gate.weight"),
                    t.hidden_size_per_layer_input,
                    t.hidden_size,
                ));
                projections.push(single(
                    format!("{p}.per_layer_projection.weight"),
                    t.hidden_size,
                    t.hidden_size_per_layer_input,
                ));
            }
        }
        if t.hidden_size_per_layer_input > 0 {
            projections.push(single(
                format!("{}.per_layer_model_projection.weight", self.prefix),
                checked_elements(&[t.num_hidden_layers, t.hidden_size_per_layer_input])?,
                t.hidden_size,
            ));
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
    if t.num_kv_shared_layers >= t.num_hidden_layers {
        return Err(invalid("Gemma 4 shares K/V of every layer"));
    }
    validate_equations(t)
}

/// The implemented layer equations, independent of which layers own K/V.
pub(super) fn validate_equations(t: &TextConfig) -> Result<()> {
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
        || u32::try_from(t.sliding_window).is_err()
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
