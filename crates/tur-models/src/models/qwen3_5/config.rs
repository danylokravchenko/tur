//! Validated text configuration for dense and expert Qwen3.5 architectures (3.8/3.6).

use candle_core::DType;
use serde::Deserialize;
use std::collections::BTreeMap;
use tur_core::{
    Result,
    execution::{
        KvFormat, KvLayout, LayerStateKind, Modality, ModelDescription, ModelIdentity,
        checked_elements, invalid,
    },
};

/// One loaded projection: checkpoint matrices stacked along output rows into the matrix
/// the model multiplies by. Names are stable keys for weight plans, calibration
/// statistics and prepacked artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// Checkpoint module path of the loaded matrix, e.g. `….layers.0.self_attn.kv_proj`.
    pub name: String,
    /// Checkpoint tensors in stacking order, with the output rows each contributes.
    pub parts: Vec<(String, usize)>,
    pub rows: usize,
    pub cols: usize,
    /// Experts of a stacked expert bank, each owning `rows / experts` rows; 1 otherwise.
    pub experts: usize,
}

impl Projection {
    fn new(name: String, parts: Vec<(String, usize)>, cols: usize) -> Self {
        let rows = parts.iter().map(|(_, rows)| rows).sum();
        Self {
            name,
            parts,
            rows,
            cols,
            experts: 1,
        }
    }

    fn bank(name: String, experts: usize, rows: usize, cols: usize) -> Self {
        Self {
            parts: vec![(name.clone(), experts * rows)],
            name,
            rows: experts * rows,
            cols,
            experts,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum LayerKind {
    LinearAttention,
    FullAttention,
}

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
#[allow(
    clippy::struct_excessive_bools,
    reason = "mirrors the checkpoint's JSON settings field for field"
)]
pub(super) struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    #[serde(default)]
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
    /// Native multi-token prediction layers shipped beside the text tower.
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    pub layer_types: Vec<LayerKind>,
    pub rope_parameters: Rope,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    pub eos_token_id: serde_json::Value,
    #[serde(default)]
    pub mamba_ssm_dtype: Option<String>,
}

/// Sampling defaults and stop tokens from `generation_config.json`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationDefaults {
    pub eos_tokens: Vec<u32>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
}

#[derive(Deserialize)]
struct RawGeneration {
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    do_sample: Option<bool>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    repetition_penalty: Option<f64>,
}

/// Architecture-specific settings. Construction validates the complete text layout.
/// Vision and MTP tensors may exist in the checkpoint but are never executed.
#[derive(Clone, Debug)]
pub struct Config {
    pub(super) text: TextConfig,
    pub(super) moe: Option<MoeConfig>,
    /// Load and run the native multi-token prediction head ([`Config::with_mtp`]).
    pub(super) mtp: bool,
    pub(super) prefix: String,
    forbidden_tokens: Vec<u32>,
    eos: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(super) struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

impl Config {
    /// Parse explicit architecture identifiers and nested text settings, never a model name.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        let (text, prefix) = text_section(&raw)?;
        let is_moe = matches!(
            raw["model_type"].as_str(),
            Some("qwen3_5_moe" | "qwen3_5_moe_text")
        );
        let moe = moe_settings(text, is_moe)?;
        reject_unsupported_settings(&raw, text, is_moe)?;
        let mut text: TextConfig = serde_json::from_value(text.clone())?;
        if let Some(moe) = &moe {
            text.intermediate_size = moe
                .moe_intermediate_size
                .max(moe.shared_expert_intermediate_size);
        }
        if text.model_type
            != if is_moe {
                "qwen3_5_moe_text"
            } else {
                "qwen3_5_text"
            }
            || text.hidden_act != "silu"
            || text.attention_bias
            || text.attention_dropout != 0.0
            || !text.attn_output_gate
            || text
                .output_gate_type
                .as_deref()
                .is_some_and(|s| s != "swish" && s != "silu")
            || text
                .mamba_ssm_dtype
                .as_deref()
                .is_some_and(|s| s != "float32")
        {
            return Err(invalid(
                "unsupported Qwen3.5 activation, gate, bias, dropout or state dtype",
            ));
        }
        if raw
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .is_some_and(|t| t != text.tie_word_embeddings)
        {
            return Err(invalid("conflicting embedding tying settings"));
        }
        let dims = [
            text.hidden_size,
            text.intermediate_size,
            text.vocab_size,
            text.num_hidden_layers,
            text.num_attention_heads,
            text.num_key_value_heads,
            text.head_dim,
            text.linear_num_key_heads,
            text.linear_num_value_heads,
            text.linear_key_head_dim,
            text.linear_value_head_dim,
            text.max_position_embeddings,
        ];
        if dims.contains(&0)
            || text.linear_conv_kernel_dim < 2
            || text.layer_types.len() != text.num_hidden_layers
            || !text
                .num_attention_heads
                .is_multiple_of(text.num_key_value_heads)
            || !text
                .linear_num_value_heads
                .is_multiple_of(text.linear_num_key_heads)
            || !text.rms_norm_eps.is_finite()
            || text.rms_norm_eps <= 0.0
            || u32::try_from(text.vocab_size).is_err()
        {
            return Err(invalid(
                "invalid Qwen3.5 dimensions, heads, layer list or normalization",
            ));
        }
        validate_rope(&text)?;
        let eos = parse_tokens(&text.eos_token_id, text.vocab_size)?;
        let forbidden_tokens = modality_tokens(&raw)?;
        let config = Self {
            text,
            moe,
            mtp: false,
            prefix: prefix.into(),
            forbidden_tokens,
            eos,
        };
        config
            .description(
                "validation".into(),
                KvLayout::new(1, KvFormat::F32),
                DType::F32,
            )?
            .validate()?;
        // Validate every weight product before any file payload is read.
        config.weight_shapes()?;
        Ok(config)
    }

    /// Exclusive text positional limit.
    pub fn max_position(&self) -> usize {
        self.text.max_position_embeddings
    }

    /// Residual stream width.
    pub fn hidden_size(&self) -> usize {
        self.text.hidden_size
    }

    /// Padded projection vocabulary size (may exceed tokenizer vocabulary).
    pub fn vocab_size(&self) -> usize {
        self.text.vocab_size
    }

    /// EOS IDs declared by the architecture; generation config may add more.
    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos
    }

    /// Image/video/audio placeholder tokens that text-only execution must reject.
    pub fn forbidden_tokens(&self) -> &[u32] {
        &self.forbidden_tokens
    }

    /// Add placeholders the tokenizer declares in `extra_special_tokens` (for example audio
    /// tokens that only some checkpoints define) when `tokenizer.json` assigns them IDs.
    pub fn forbid_tokenizer_modalities(
        &mut self,
        tokenizer_config: &str,
        tokenizer: &str,
    ) -> Result<()> {
        let config: serde_json::Value = serde_json::from_str(tokenizer_config)?;
        let tokenizer: serde_json::Value = serde_json::from_str(tokenizer)?;
        let Some(extra) = config["extra_special_tokens"].as_object() else {
            return Ok(());
        };
        let names: Vec<&str> = extra
            .iter()
            .filter(|(key, _)| {
                ["audio", "image", "video", "vision"]
                    .iter()
                    .any(|m| key.starts_with(m))
            })
            .filter_map(|(_, value)| value.as_str())
            .collect();
        for added in tokenizer["added_tokens"].as_array().into_iter().flatten() {
            let (Some(content), Some(id)) = (added["content"].as_str(), added["id"].as_u64())
            else {
                continue;
            };
            let id = u32::try_from(id).map_err(|_| invalid("added token overflow"))?;
            if names.contains(&content) && !self.forbidden_tokens.contains(&id) {
                self.forbidden_tokens.push(id);
            }
        }
        Ok(())
    }

    /// Parse `generation_config.json`, merging its EOS IDs after the architecture's.
    /// Sampling values are ignored when `do_sample` is false.
    pub fn generation_defaults(&self, json: Option<&str>) -> Result<GenerationDefaults> {
        let mut defaults = GenerationDefaults {
            eos_tokens: self.eos.clone(),
            ..Default::default()
        };
        let Some(json) = json else {
            return Ok(defaults);
        };
        let raw: RawGeneration = serde_json::from_str(json)?;
        if let Some(eos) = &raw.eos_token_id {
            for id in parse_tokens(eos, self.text.vocab_size)? {
                if !defaults.eos_tokens.contains(&id) {
                    defaults.eos_tokens.push(id);
                }
            }
        }
        if raw.do_sample != Some(false) {
            if raw.temperature.is_some_and(|t| !t.is_finite() || t < 0.0)
                || raw
                    .top_p
                    .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
                || raw.top_k == Some(0)
                || raw
                    .repetition_penalty
                    .is_some_and(|r| !r.is_finite() || r <= 0.0)
            {
                return Err(invalid("invalid generation sampling defaults"));
            }
            defaults.temperature = raw.temperature;
            defaults.top_k = raw.top_k;
            defaults.top_p = raw.top_p;
            defaults.repetition_penalty = raw.repetition_penalty;
        }
        Ok(defaults)
    }

    pub(super) fn layer_kinds(&self) -> &[LayerKind] {
        &self.text.layer_types
    }

    /// Whether the checkpoint declares a native multi-token prediction head this
    /// implementation can run: one dense full-attention layer sharing the embedding.
    pub fn supports_mtp(&self) -> bool {
        self.text.mtp_num_hidden_layers == 1
            && !self.text.mtp_use_dedicated_embeddings
            && self.moe.is_none()
    }

    /// Also load the `mtp.*` head, whose attention keys and values add one state layer.
    pub fn with_mtp(self) -> Result<Self> {
        if !self.supports_mtp() {
            return Err(invalid(
                "checkpoint has no supported MTP head (one dense layer, shared embedding)",
            ));
        }
        Ok(Self { mtp: true, ..self })
    }

    /// Whether the multi-token prediction head is loaded.
    pub fn mtp(&self) -> bool {
        self.mtp
    }

    pub(super) fn key_width(&self) -> Result<usize> {
        checked_elements(&[
            self.text.linear_num_key_heads,
            self.text.linear_key_head_dim,
        ])
    }

    pub(super) fn value_width(&self) -> Result<usize> {
        checked_elements(&[
            self.text.linear_num_value_heads,
            self.text.linear_value_head_dim,
        ])
    }

    /// Depthwise convolution channels: query and key widths, then value width.
    pub(super) fn conv_channels(&self) -> Result<usize> {
        self.key_width()?
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.value_width().ok()?))
            .ok_or_else(|| invalid("convolution channels overflow"))
    }

    /// Inputs retained between steps; the newest input completes the kernel window.
    pub(super) fn conv_history(&self) -> usize {
        self.text.linear_conv_kernel_dim - 1
    }

    pub(super) fn rotary_dim(&self) -> usize {
        (self.text.head_dim as f64 * self.text.rope_parameters.partial_rotary_factor) as usize
    }

    /// State layout with K/V stored as `kv`. F32 recurrent state matches the reference's
    /// `mamba_ssm_dtype` and FP32 accumulation.
    /// State layout with K/V stored per `kv` and recurrent matrices stored as `recurrent`.
    pub(super) fn description(
        &self,
        revision: String,
        kv: KvLayout,
        recurrent: DType,
    ) -> Result<ModelDescription> {
        let t = &self.text;
        let layers = t
            .layer_types
            .iter()
            .map(|kind| {
                Ok(match kind {
                    LayerKind::FullAttention => LayerStateKind::Attention {
                        kv_heads: t.num_key_value_heads,
                        head_dim: t.head_dim,
                    },
                    LayerKind::LinearAttention => LayerStateKind::Recurrent {
                        conv_channels: self.conv_channels()?,
                        conv_width: self.conv_history(),
                        heads: t.linear_num_value_heads,
                        key_dim: t.linear_key_head_dim,
                        value_dim: t.linear_value_head_dim,
                        dtype: recurrent,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut layers = layers;
        if self.mtp {
            // The head's own attention keys and values, after every text layer.
            layers.push(LayerStateKind::Attention {
                kv_heads: t.num_key_value_heads,
                head_dim: t.head_dim,
            });
        }
        Ok(ModelDescription {
            identity: ModelIdentity {
                revision: if self.mtp {
                    format!("{revision}+mtp")
                } else {
                    revision
                },
                state_format: 1,
            },
            layers,
            kv,
            modalities: vec![Modality::Text],
            vocab_size: t.vocab_size,
            max_position: t.max_position_embeddings,
        })
    }

    /// Exact `SafeTensors` weight names and dimensions required by the text tower.
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
        let q = checked_elements(&[t.num_attention_heads, t.head_dim])?;
        let kv = checked_elements(&[t.num_key_value_heads, t.head_dim])?;
        for (i, kind) in t.layer_types.iter().enumerate() {
            let p = format!("{}.layers.{i}", self.prefix);
            for n in ["input_layernorm", "post_attention_layernorm"] {
                shapes.insert(format!("{p}.{n}.weight"), vec![t.hidden_size]);
            }
            if let Some(moe) = &self.moe {
                shapes.insert(
                    format!("{p}.mlp.gate.weight"),
                    vec![moe.num_experts, t.hidden_size],
                );
                shapes.insert(
                    format!("{p}.mlp.experts.gate_up_proj"),
                    vec![
                        moe.num_experts,
                        checked_elements(&[2, moe.moe_intermediate_size])?,
                        t.hidden_size,
                    ],
                );
                shapes.insert(
                    format!("{p}.mlp.experts.down_proj"),
                    vec![moe.num_experts, t.hidden_size, moe.moe_intermediate_size],
                );
                shapes.insert(
                    format!("{p}.mlp.shared_expert_gate.weight"),
                    vec![1, t.hidden_size],
                );
                for n in ["gate_proj", "up_proj"] {
                    shapes.insert(
                        format!("{p}.mlp.shared_expert.{n}.weight"),
                        vec![moe.shared_expert_intermediate_size, t.hidden_size],
                    );
                }
                shapes.insert(
                    format!("{p}.mlp.shared_expert.down_proj.weight"),
                    vec![t.hidden_size, moe.shared_expert_intermediate_size],
                );
            } else {
                for n in ["gate_proj", "up_proj"] {
                    shapes.insert(
                        format!("{p}.mlp.{n}.weight"),
                        vec![t.intermediate_size, t.hidden_size],
                    );
                }
                shapes.insert(
                    format!("{p}.mlp.down_proj.weight"),
                    vec![t.hidden_size, t.intermediate_size],
                );
            }
            let entries = match kind {
                LayerKind::FullAttention => vec![
                    (
                        "q_proj.weight",
                        vec![checked_elements(&[2, q])?, t.hidden_size],
                    ),
                    ("k_proj.weight", vec![kv, t.hidden_size]),
                    ("v_proj.weight", vec![kv, t.hidden_size]),
                    ("o_proj.weight", vec![t.hidden_size, q]),
                    ("q_norm.weight", vec![t.head_dim]),
                    ("k_norm.weight", vec![t.head_dim]),
                ],
                LayerKind::LinearAttention => vec![
                    (
                        "in_proj_qkv.weight",
                        vec![self.conv_channels()?, t.hidden_size],
                    ),
                    ("in_proj_z.weight", vec![self.value_width()?, t.hidden_size]),
                    (
                        "in_proj_a.weight",
                        vec![t.linear_num_value_heads, t.hidden_size],
                    ),
                    (
                        "in_proj_b.weight",
                        vec![t.linear_num_value_heads, t.hidden_size],
                    ),
                    ("out_proj.weight", vec![t.hidden_size, self.value_width()?]),
                    (
                        "conv1d.weight",
                        vec![self.conv_channels()?, 1, t.linear_conv_kernel_dim],
                    ),
                    ("A_log", vec![t.linear_num_value_heads]),
                    ("dt_bias", vec![t.linear_num_value_heads]),
                    ("norm.weight", vec![t.linear_value_head_dim]),
                ],
            };
            let module = match kind {
                LayerKind::FullAttention => "self_attn",
                LayerKind::LinearAttention => "linear_attn",
            };
            for (n, shape) in entries {
                shapes.insert(format!("{p}.{module}.{n}"), shape);
            }
        }
        if self.mtp {
            let h = t.hidden_size;
            let p = "mtp.layers.0";
            for (name, shape) in [
                (
                    "mtp.fc.weight".to_owned(),
                    vec![h, checked_elements(&[2, h])?],
                ),
                ("mtp.pre_fc_norm_embedding.weight".into(), vec![h]),
                ("mtp.pre_fc_norm_hidden.weight".into(), vec![h]),
                ("mtp.norm.weight".into(), vec![h]),
                (format!("{p}.input_layernorm.weight"), vec![h]),
                (format!("{p}.post_attention_layernorm.weight"), vec![h]),
                (
                    format!("{p}.self_attn.q_proj.weight"),
                    vec![checked_elements(&[2, q])?, h],
                ),
                (format!("{p}.self_attn.k_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.v_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.o_proj.weight"), vec![h, q]),
                (format!("{p}.self_attn.q_norm.weight"), vec![t.head_dim]),
                (format!("{p}.self_attn.k_norm.weight"), vec![t.head_dim]),
                (
                    format!("{p}.mlp.gate_proj.weight"),
                    vec![t.intermediate_size, h],
                ),
                (
                    format!("{p}.mlp.up_proj.weight"),
                    vec![t.intermediate_size, h],
                ),
                (
                    format!("{p}.mlp.down_proj.weight"),
                    vec![h, t.intermediate_size],
                ),
            ] {
                shapes.insert(name, shape);
            }
        }
        for shape in shapes.values() {
            checked_elements(shape)?
                .checked_mul(4)
                .ok_or_else(|| invalid("weight bytes overflow"))?;
        }
        Ok(shapes)
    }

    /// Every projection the model multiplies by, in load order: each layer's mixer then
    /// feed-forward projections, the MTP head's when loaded, then the LM head (stacked
    /// from the embedding when tied).
    pub fn projections(&self) -> Result<Vec<Projection>> {
        let t = &self.text;
        let hidden = t.hidden_size;
        let query = checked_elements(&[t.num_attention_heads, t.head_dim])?;
        let kv = checked_elements(&[t.num_key_value_heads, t.head_dim])?;
        let part = |name: String, rows: usize| (format!("{name}.weight"), rows);
        let attention = |prefix: &str| {
            let a = format!("{prefix}.self_attn");
            vec![
                Projection::new(
                    format!("{a}.q_proj"),
                    vec![part(format!("{a}.q_proj"), 2 * query)],
                    hidden,
                ),
                Projection::new(
                    format!("{a}.kv_proj"),
                    vec![
                        part(format!("{a}.k_proj"), kv),
                        part(format!("{a}.v_proj"), kv),
                    ],
                    hidden,
                ),
                Projection::new(
                    format!("{a}.o_proj"),
                    vec![part(format!("{a}.o_proj"), hidden)],
                    query,
                ),
            ]
        };
        let dense = |m: String, inner: usize| {
            vec![
                Projection::new(
                    format!("{m}.gate_up_proj"),
                    vec![
                        part(format!("{m}.gate_proj"), inner),
                        part(format!("{m}.up_proj"), inner),
                    ],
                    hidden,
                ),
                Projection::new(
                    format!("{m}.down_proj"),
                    vec![part(format!("{m}.down_proj"), hidden)],
                    inner,
                ),
            ]
        };
        let mut projections = Vec::new();
        for (i, kind) in t.layer_types.iter().enumerate() {
            let layer = format!("{}.layers.{i}", self.prefix);
            match kind {
                LayerKind::FullAttention => projections.extend(attention(&layer)),
                LayerKind::LinearAttention => {
                    let l = format!("{layer}.linear_attn");
                    let heads = t.linear_num_value_heads;
                    projections.push(Projection::new(
                        format!("{l}.in_proj"),
                        vec![
                            part(format!("{l}.in_proj_qkv"), self.conv_channels()?),
                            part(format!("{l}.in_proj_z"), self.value_width()?),
                            part(format!("{l}.in_proj_b"), heads),
                            part(format!("{l}.in_proj_a"), heads),
                        ],
                        hidden,
                    ));
                    projections.push(Projection::new(
                        format!("{l}.out_proj"),
                        vec![part(format!("{l}.out_proj"), hidden)],
                        self.value_width()?,
                    ));
                }
            }
            match &self.moe {
                Some(moe) => {
                    let e = format!("{layer}.mlp.experts");
                    let inner = moe.moe_intermediate_size;
                    projections.push(Projection::bank(
                        format!("{e}.gate_up_proj"),
                        moe.num_experts,
                        checked_elements(&[2, inner])?,
                        hidden,
                    ));
                    projections.push(Projection::bank(
                        format!("{e}.down_proj"),
                        moe.num_experts,
                        hidden,
                        inner,
                    ));
                    projections.extend(dense(
                        format!("{layer}.mlp.shared_expert"),
                        moe.shared_expert_intermediate_size,
                    ));
                }
                None => projections.extend(dense(format!("{layer}.mlp"), t.intermediate_size)),
            }
        }
        if self.mtp {
            projections.push(Projection::new(
                "mtp.fc".into(),
                vec![part("mtp.fc".into(), hidden)],
                checked_elements(&[2, hidden])?,
            ));
            projections.extend(attention("mtp.layers.0"));
            projections.extend(dense("mtp.layers.0.mlp".into(), t.intermediate_size));
        }
        let head = if t.tie_word_embeddings {
            self.embedding_name()
        } else {
            "lm_head.weight".into()
        };
        projections.push(Projection::new(
            "lm_head".into(),
            vec![(head, t.vocab_size)],
            hidden,
        ));
        Ok(projections)
    }

    /// Checkpoint name of the input embedding table.
    pub fn embedding_name(&self) -> String {
        format!("{}.embed_tokens.weight", self.prefix)
    }

    /// Checkpoint name prefix owned by the text tower; other prefixes are ignored.
    pub(super) fn text_prefix(&self) -> &str {
        &self.prefix
    }

    /// Whether the LM head reuses the input embedding matrix.
    pub fn tied_embeddings(&self) -> bool {
        self.text.tie_word_embeddings
    }
}

/// The text settings and weight prefix of a supported architecture, chosen by its explicit
/// identifiers.
fn text_section(raw: &serde_json::Value) -> Result<(&serde_json::Value, &'static str)> {
    let architectures = raw["architectures"].as_array();
    let section = match (raw["model_type"].as_str(), architectures) {
        (Some("qwen3_5"), Some(a))
            if a == &[serde_json::json!("Qwen3_5ForConditionalGeneration")] =>
        {
            (
                raw.get("text_config")
                    .ok_or_else(|| invalid("qwen3_5 requires text_config"))?,
                "model.language_model",
            )
        }
        (Some("qwen3_5_text"), Some(a)) if a == &[serde_json::json!("Qwen3_5ForCausalLM")] => {
            (raw, "model")
        }
        (Some("qwen3_5_moe"), Some(a))
            if a == &[serde_json::json!("Qwen3_5MoeForConditionalGeneration")] =>
        {
            (
                raw.get("text_config")
                    .ok_or_else(|| invalid("qwen3_5_moe requires text_config"))?,
                "model.language_model",
            )
        }
        (Some("qwen3_5_moe_text"), Some(a))
            if a == &[serde_json::json!("Qwen3_5MoeForCausalLM")] =>
        {
            (raw, "model")
        }
        _ => {
            return Err(invalid(
                "unsupported architecture: expected Qwen3_5 dense or MoE conditional generation or text causal LM",
            ));
        }
    };
    Ok(section)
}

fn moe_settings(text: &serde_json::Value, is_moe: bool) -> Result<Option<MoeConfig>> {
    Ok(if is_moe {
        let moe: MoeConfig = serde_json::from_value(text.clone())?;
        if moe.num_experts == 0
            || moe.num_experts > 256
            || moe.num_experts_per_tok == 0
            || moe.num_experts_per_tok > moe.num_experts
            || moe.moe_intermediate_size == 0
            || moe.shared_expert_intermediate_size == 0
        {
            return Err(invalid("invalid Qwen3.5 MoE expert count, top-k or width"));
        }
        Some(moe)
    } else {
        None
    })
}

/// Reject quantization, scaling, windowing and `MoE` routing settings Tur does not implement.
fn reject_unsupported_settings(
    raw: &serde_json::Value,
    text: &serde_json::Value,
    is_moe: bool,
) -> Result<()> {
    for settings in [raw, text] {
        for key in ["quantization_config", "rope_scaling", "sliding_window"] {
            if settings.get(key).is_some_and(|v| !v.is_null()) {
                return Err(invalid(format!("unsupported Qwen3.5 setting: {key}")));
            }
        }
        if !is_moe
            && [
                "num_experts",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "num_experts_per_tok",
            ]
            .iter()
            .any(|key| settings.get(key).is_some_and(|v| !v.is_null()))
        {
            return Err(invalid("MoE settings require an explicit MoE architecture"));
        }
        if settings.get("norm_topk_prob").is_some_and(|v| v != true)
            || settings.get("decoder_sparse_step").is_some_and(|v| v != 1)
        {
            return Err(invalid(
                "unsupported MoE routing normalization or sparse layer interval",
            ));
        }
        let mlp_only = settings.get("mlp_only_layers").filter(|v| !v.is_null());
        if mlp_only.is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty())) {
            return Err(invalid("MLP-only layers are unsupported"));
        }
    }
    Ok(())
}

#[allow(
    clippy::float_cmp,
    reason = "duplicate configuration fields must agree exactly"
)]
fn validate_rope(text: &TextConfig) -> Result<()> {
    let rope = &text.rope_parameters;
    let rotary = text.head_dim as f64 * rope.partial_rotary_factor;
    if rope.rope_type != "default"
        || !rope.rope_theta.is_finite()
        || rope.rope_theta <= 0.0
        || !rotary.is_finite()
        || rotary < 2.0
        || rotary > text.head_dim as f64
        || rotary.fract() != 0.0
        || !(rotary as usize).is_multiple_of(2)
        || text
            .partial_rotary_factor
            .is_some_and(|f| f != rope.partial_rotary_factor)
    {
        return Err(invalid("unsupported or invalid partial RoPE configuration"));
    }
    // Text-only positions are equal on all three M-RoPE axes, so interleaved sections
    // reduce to ordinary partial rotary; any other layout would need real 3-D positions.
    if !rope.mrope_section.is_empty() {
        let sum = rope
            .mrope_section
            .iter()
            .try_fold(0usize, |s, &n| s.checked_add(n))
            .ok_or_else(|| invalid("mrope section overflow"))?;
        if !rope.mrope_interleaved || rope.mrope_section.len() != 3 || sum != rotary as usize / 2 {
            return Err(invalid("unsupported M-RoPE sections"));
        }
    }
    Ok(())
}

/// Image, video and audio token IDs, which text-only execution must never accept.
fn modality_tokens(raw: &serde_json::Value) -> Result<Vec<u32>> {
    let mut forbidden_tokens = Vec::new();
    for field in [
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "audio_token_id",
    ] {
        if let Some(id) = raw.get(field).and_then(serde_json::Value::as_u64) {
            forbidden_tokens
                .push(u32::try_from(id).map_err(|_| invalid("modality token overflow"))?);
        }
    }
    Ok(forbidden_tokens)
}

fn parse_tokens(raw: &serde_json::Value, vocab: usize) -> Result<Vec<u32>> {
    let values = raw
        .as_array()
        .map_or_else(|| vec![raw.clone()], Clone::clone);
    if values.is_empty() {
        return Err(invalid("empty EOS list"));
    }
    values
        .iter()
        .map(|v| {
            let id = v
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid("invalid EOS ID"))?;
            if id as usize >= vocab {
                return Err(invalid("EOS outside vocabulary"));
            }
            Ok(id)
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    const SMALL: &str =
        include_str!("../../../../tur/tests/fixtures/checkpoints/small-config.json");
    const FLAGSHIP: &str =
        include_str!("../../../../tur/tests/fixtures/checkpoints/qwen3.8-config.json");
    const FLAGSHIP_GENERATION: &str =
        include_str!("../../../../tur/tests/fixtures/checkpoints/qwen3.8-generation_config.json");
    const FLAGSHIP_TOKENIZER_CONFIG: &str =
        include_str!("../../../../tur/tests/fixtures/checkpoints/qwen3.8-tokenizer_config.json");
    const FLAGSHIP_TENSORS: &str =
        include_str!("../../../../tur/tests/fixtures/checkpoints/qwen3.8-text-tensors.json");

    #[test]
    fn test_pinned_configs_parse_with_hybrid_layout() -> Result<()> {
        for (json, layers, attention) in [(SMALL, 24, 6), (FLAGSHIP, 64, 16)] {
            let config = Config::from_json(json)?;
            let description = config.description(
                "fixture".into(),
                KvLayout::new(16, KvFormat::Q8),
                DType::F32,
            )?;
            description.validate()?;
            assert_eq!(description.layers.len(), layers);
            let full = description
                .layers
                .iter()
                .filter(|l| matches!(l, LayerStateKind::Attention { .. }))
                .count();
            assert_eq!(full, attention);
            assert_eq!(config.rotary_dim(), 64);
            assert_eq!(config.conv_history(), 3);
            assert_eq!(
                config.forbidden_tokens(),
                &[248_056, 248_057, 248_053, 248_054]
            );
        }
        Ok(())
    }

    #[test]
    fn test_flagship_weight_shapes_match_pinned_checkpoint_headers() -> Result<()> {
        let config = Config::from_json(FLAGSHIP)?;
        let pinned: serde_json::Value = serde_json::from_str(FLAGSHIP_TENSORS)?;
        let tensors = pinned["text_tensors"]
            .as_object()
            .ok_or_else(|| invalid("missing pinned tensors"))?;
        let expected = config.weight_shapes()?;
        assert_eq!(expected.len(), tensors.len());
        for (name, shape) in &expected {
            let header = tensors
                .get(name)
                .ok_or_else(|| invalid(format!("missing {name}")))?;
            let actual: Vec<usize> = serde_json::from_value(header["shape"].clone())?;
            assert_eq!(&actual, shape, "{name}");
        }
        Ok(())
    }

    #[test]
    fn test_projections_cover_every_matrix_once_with_matching_shapes() -> Result<()> {
        let moe = include_str!("../../../../tur/tests/fixtures/checkpoints/qwen3.6-config.json");
        for (json, mtp) in [(SMALL, true), (FLAGSHIP, false), (moe, false)] {
            let mut config = Config::from_json(json)?;
            if mtp && config.supports_mtp() {
                config = config.with_mtp()?;
            }
            let shapes = config.weight_shapes()?;
            let mut seen = std::collections::BTreeSet::new();
            for projection in config.projections()? {
                let mut rows = 0;
                for (name, part_rows) in &projection.parts {
                    let shape = shapes
                        .get(name)
                        .ok_or_else(|| invalid(format!("{name} is not a checkpoint tensor")))?;
                    let (part, cols) = match shape.as_slice() {
                        [r, c] => (*r, *c),
                        [e, r, c] => (e * r, *c),
                        _ => return Err(invalid(format!("{name} is not a matrix"))),
                    };
                    assert_eq!((part, cols), (*part_rows, projection.cols), "{name}");
                    rows += part;
                    assert!(seen.insert(name.clone()) || config.tied_embeddings());
                }
                assert_eq!(rows, projection.rows);
                assert!(projection.rows.is_multiple_of(projection.experts));
            }
            // Everything else is a vector, a convolution, a router or the embedding.
            for (name, shape) in &shapes {
                if !seen.contains(name) {
                    assert!(
                        shape.len() != 2
                            || name.ends_with("embed_tokens.weight")
                            || name.ends_with("gate.weight"),
                        "{name} {shape:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_tokenizer_modality_placeholders_are_forbidden() -> Result<()> {
        let mut config = Config::from_json(FLAGSHIP)?;
        let tokenizer = r#"{"added_tokens": [
            {"id": 248046, "content": "<|im_end|>"},
            {"id": 248056, "content": "<|image_pad|>"},
            {"id": 248070, "content": "<|audio_start|>"},
            {"id": 248076, "content": "<|audio_pad|>"}
        ]}"#;
        config.forbid_tokenizer_modalities(FLAGSHIP_TOKENIZER_CONFIG, tokenizer)?;
        assert_eq!(
            config.forbidden_tokens(),
            &[248_056, 248_057, 248_053, 248_054, 248_070, 248_076]
        );
        Ok(())
    }

    #[test]
    fn test_generation_defaults_merge_eos_and_sampling() -> Result<()> {
        let config = Config::from_json(FLAGSHIP)?;
        let defaults = config.generation_defaults(Some(FLAGSHIP_GENERATION))?;
        assert_eq!(defaults.eos_tokens, vec![248_044, 248_046]);
        assert_eq!(defaults.top_k, Some(20));
        assert_eq!(defaults.top_p, Some(0.95));
        assert_eq!(defaults.temperature, Some(1.0));
        let greedy = config.generation_defaults(Some(r#"{"do_sample": false, "top_k": 0}"#))?;
        assert_eq!(greedy.top_k, None);
        Ok(())
    }

    #[test]
    fn test_unsupported_layouts_are_rejected() -> Result<()> {
        let mut raw: serde_json::Value = serde_json::from_str(SMALL)?;
        raw["text_config"]["attention_bias"] = true.into();
        assert!(Config::from_json(&raw.to_string()).is_err());
        let mut raw: serde_json::Value = serde_json::from_str(SMALL)?;
        raw["text_config"]["mlp_only_layers"] = serde_json::json!([1]);
        assert!(Config::from_json(&raw.to_string()).is_err());
        let mut raw: serde_json::Value = serde_json::from_str(SMALL)?;
        raw["text_config"]["mlp_only_layers"] = serde_json::Value::Null;
        assert!(Config::from_json(&raw.to_string()).is_ok());
        let mut raw: serde_json::Value = serde_json::from_str(SMALL)?;
        raw["model_type"] = "qwen3".into();
        assert!(Config::from_json(&raw.to_string()).is_err());
        let mut raw: serde_json::Value = serde_json::from_str(SMALL)?;
        raw["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([10, 11, 10]);
        assert!(Config::from_json(&raw.to_string()).is_err());
        Ok(())
    }

    #[test]
    fn test_mtp_head_adds_its_tensors_and_one_attention_state_layer() -> Result<()> {
        let config = Config::from_json(SMALL)?;
        assert!(config.supports_mtp() && !config.mtp());
        let plain = config.description("r".into(), KvLayout::new(16, KvFormat::F16), DType::F32)?;
        let with = config.clone().with_mtp()?;
        let description =
            with.description("r".into(), KvLayout::new(16, KvFormat::F16), DType::F32)?;
        assert_eq!(description.layers.len(), plain.layers.len() + 1);
        assert!(matches!(
            description.layers.last(),
            Some(LayerStateKind::Attention { .. })
        ));
        assert_ne!(description.identity, plain.identity);
        let shapes = with.weight_shapes()?;
        assert_eq!(shapes.keys().filter(|n| n.starts_with("mtp.")).count(), 15);
        assert!(
            !config
                .weight_shapes()?
                .keys()
                .any(|n| n.starts_with("mtp."))
        );
        // No head declared, or dedicated embeddings: no MTP.
        let mut raw: serde_json::Value = serde_json::from_str(SMALL)?;
        raw["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(0);
        assert!(Config::from_json(&raw.to_string())?.with_mtp().is_err());
        raw["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        raw["text_config"]["mtp_use_dedicated_embeddings"] = serde_json::json!(true);
        assert!(Config::from_json(&raw.to_string())?.with_mtp().is_err());
        Ok(())
    }
}
