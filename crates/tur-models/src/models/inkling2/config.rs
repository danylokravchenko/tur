//! Validated Inkling 2 settings, selected from explicit architecture fields only.
use crate::models::{Projection, generation, weights::WeightLayout};
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

#[derive(Clone, Copy, Debug, Deserialize)]
pub(super) struct Loop {
    pub(super) core_start: usize,
    pub(super) core_end: usize,
    pub(super) num_passes: usize,
    pub(super) gate_rank: usize,
    is_varloop: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "mirrors the checkpoint's boolean architecture fields"
)]
pub(super) struct Text {
    model_type: String,
    pub(super) hidden_size: usize,
    pub(super) num_hidden_layers: usize,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) head_dim: usize,
    swa_head_dim: usize,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: usize,
    pub(super) sliding_window_size: usize,
    pub(super) d_rel: usize,
    pub(super) rel_extent: usize,
    q_bias: bool,
    o_bias: bool,
    pub(super) log_scaling_n_floor: usize,
    pub(super) log_scaling_alpha: f64,
    pub(super) rms_norm_eps: f64,
    use_value_norm_no_gamma: bool,
    pub(super) qk_l2_clip_threshold: f64,
    pub(super) rel_l2_clip_threshold: f64,
    local_layer_ids: Vec<usize>,
    /// Transformers derives layer kinds itself and ignores `local_layer_ids`; when a
    /// config names them, they must agree.
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    pub(super) sconv_kernel_size: usize,
    pub(super) final_logit_softcapping: f64,
    pub(super) n_layers_per_attn_res_block: usize,
    attn_res_normalize_values: bool,
    pub(super) n_routed_experts: usize,
    pub(super) num_experts_per_tok: usize,
    n_shared_experts: usize,
    pub(super) intermediate_size: usize,
    pub(super) mlp_gate_softcap: f64,
    pub(super) mlp_up_softcap: f64,
    pub(super) route_scale: f64,
    use_gate_bias: bool,
    pub(super) loop_config: Loop,
    model_max_length: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct Published {
    model_type: String,
    architectures: Vec<String>,
    eos_token_id: serde_json::Value,
    text_config: Text,
}

/// How one layer's routed experts are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertFormat {
    /// E2M1 codes with an E4M3 scale per 16 values and a factor per expert.
    Nvfp4,
    /// E4M3 values with a power-of-two scale per 32 values.
    Mxfp8,
}

impl ExpertFormat {
    pub(super) fn microscale(self) -> tur_metal::quantize::Microscale {
        match self {
            Self::Nvfp4 => tur_metal::quantize::Microscale::Nvfp4,
            Self::Mxfp8 => tur_metal::quantize::Microscale::Mxfp8,
        }
    }
}

/// One application of a decoder layer: looped layers apply once per pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Application {
    pub(super) layer: usize,
    pub(super) pass: usize,
}

/// Columns of one application's convolution history: keys, values, the attention output
/// and the expert output, each `taps - 1` rows deep.
#[derive(Clone, Copy, Debug)]
pub(super) struct ConvColumns {
    pub(super) key: usize,
    pub(super) value: usize,
    pub(super) attention: usize,
    pub(super) mlp: usize,
    pub(super) width: usize,
}

/// Inkling 2 settings. Construction validates the complete text layout; expert formats come
/// from the checkpoint's quantization file ([`Self::with_quantization`]).
#[derive(Clone, Debug)]
pub struct Config {
    pub(super) text: Text,
    eos: Vec<u32>,
    experts: Option<Vec<ExpertFormat>>,
}

/// Most states one attention residual reads: the bound of the Metal residual kernel.
const RESIDUAL_STATES: usize = 8;

/// Prefix of every text tensor in the published checkpoint.
pub(super) const PREFIX: &str = "model.llm";

impl Config {
    /// Parse explicit architecture identifiers, never a model name.
    pub fn from_json(json: &str) -> Result<Self> {
        let published: Published = serde_json::from_str(json)?;
        let t = &published.text_config;
        if published.model_type != "inkling2_loop"
            || published.architectures != ["Inkling2LoopForConditionalGeneration"]
            || t.model_type != "inkling2_loop_text"
            || t.q_bias
            || t.o_bias
            || !t.use_value_norm_no_gamma
            || !t.attn_res_normalize_values
            || !t.use_gate_bias
            || t.n_shared_experts != 0
            || t.loop_config.is_varloop
            || (
                t.swa_head_dim,
                t.swa_num_attention_heads,
                t.swa_num_key_value_heads,
            ) != (t.head_dim, t.num_attention_heads, t.num_key_value_heads)
        {
            return Err(invalid("unsupported Inkling 2 architecture settings"));
        }
        Self::validate(t)?;
        let eos = generation::parse_tokens(&published.eos_token_id, t.unpadded_vocab_size)?;
        Ok(Self {
            text: published.text_config,
            eos,
            experts: None,
        })
    }

    fn validate(t: &Text) -> Result<()> {
        let l = &t.loop_config;
        let block = t.n_layers_per_attn_res_block;
        let floats = [
            t.log_scaling_alpha,
            t.rms_norm_eps,
            t.qk_l2_clip_threshold,
            t.rel_l2_clip_threshold,
            t.final_logit_softcapping,
            t.mlp_gate_softcap,
            t.mlp_up_softcap,
            t.route_scale,
        ];
        // Kernels: head dims of the paged attention kernels, 32-value expert rows,
        // relative states held one component per lane, routing over at most 256 experts.
        if t.hidden_size == 0
            || !t.hidden_size.is_multiple_of(32)
            || !t.intermediate_size.is_multiple_of(32)
            || t.intermediate_size == 0
            || !matches!(t.head_dim, 64 | 128 | 256)
            || t.num_key_value_heads == 0
            || !t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
            || t.num_attention_heads == 0
            || t.d_rel == 0
            || t.d_rel > 32
            || t.rel_extent == 0
            || t.sliding_window_size == 0
            || t.sconv_kernel_size < 2
            || t.n_routed_experts == 0
            || t.n_routed_experts > 256
            || t.num_experts_per_tok == 0
            || t.num_experts_per_tok > t.n_routed_experts
            || t.unpadded_vocab_size == 0
            || t.unpadded_vocab_size > t.vocab_size
            || t.model_max_length == 0
            || t.log_scaling_n_floor == 0
            || l.gate_rank == 0
            || floats.iter().any(|v| !v.is_finite() || *v < 0.0)
            || [
                t.qk_l2_clip_threshold,
                t.rel_l2_clip_threshold,
                t.final_logit_softcapping,
                t.mlp_gate_softcap,
                t.mlp_up_softcap,
            ]
            .contains(&0.0)
        {
            return Err(invalid("invalid Inkling 2 dimensions or settings"));
        }
        // Transformers' own checks: looped layers start after the first block and span
        // whole blocks.
        if block == 0
            || l.num_passes == 0
            || l.core_start == 0
            || l.core_start >= l.core_end
            || l.core_end > t.num_hidden_layers
            || !l.core_start.is_multiple_of(block)
            || !l.core_end.is_multiple_of(block)
            || t.local_layer_ids.iter().any(|&i| i >= t.num_hidden_layers)
        {
            return Err(invalid(
                "invalid Inkling 2 loop or attention residual layout",
            ));
        }
        // The residual kernel reads at most 8 states: one per block plus the partial one.
        if t.num_hidden_layers.div_ceil(block) + 1 > RESIDUAL_STATES {
            return Err(invalid("too many attention residual blocks"));
        }
        if let Some(kinds) = &t.layer_types {
            let derived = (0..t.num_hidden_layers).map(|i| {
                if t.local_layer_ids.contains(&i) {
                    "hybrid_sliding"
                } else {
                    "hybrid"
                }
            });
            if kinds.len() != t.num_hidden_layers || !kinds.iter().map(String::as_str).eq(derived) {
                return Err(invalid("layer_types disagree with local_layer_ids"));
            }
        }
        Ok(())
    }

    /// Read each layer's expert format from the checkpoint's `hf_quant_config.json`, which
    /// names every expert tensor under its NVFP4 or MXFP8 parameter group.
    pub fn with_quantization(self, json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        let groups = raw["model"]["parameter_groups"]
            .as_array()
            .ok_or_else(|| invalid("quantization config has no parameter groups"))?;
        let mut formats = vec![None; self.layers()];
        for group in groups {
            let format = match group["compute_dtype"].as_str() {
                Some("nvfp4") => ExpertFormat::Nvfp4,
                Some("mxfp8") => ExpertFormat::Mxfp8,
                _ => continue,
            };
            for name in group["tensors"].as_array().into_iter().flatten() {
                let name = name
                    .as_str()
                    .ok_or_else(|| invalid("quantized tensor name is not a string"))?;
                let layer = (0..self.layers())
                    .find(|&i| {
                        [
                            Self::expert_name(i, "w13_weight"),
                            Self::expert_name(i, "w2_weight"),
                        ]
                        .contains(&name.to_owned())
                    })
                    .ok_or_else(|| invalid(format!("unexpected quantized tensor {name}")))?;
                if formats[layer].is_some_and(|f| f != format) {
                    return Err(invalid(format!("layer {layer} mixes expert formats")));
                }
                formats[layer] = Some(format);
            }
        }
        let experts = formats
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("quantization config omits a layer's experts"))?;
        Ok(Self {
            experts: Some(experts),
            ..self
        })
    }

    /// Exclusive text position limit.
    pub fn max_position(&self) -> usize {
        self.text.model_max_length
    }

    /// Output vocabulary width: the unembedding's real rows, without its padding.
    #[allow(
        clippy::misnamed_getters,
        reason = "the checkpoint's `vocab_size` counts padding rows the model never outputs"
    )]
    pub fn vocab_size(&self) -> usize {
        self.text.unpadded_vocab_size
    }

    /// Rows of the stored embedding and unembedding, padding included.
    pub(super) fn padded_vocab(&self) -> usize {
        self.text.vocab_size
    }

    /// Merge generation metadata with the checkpoint's EOS token.
    pub fn generation_defaults(
        &self,
        json: Option<&str>,
    ) -> Result<crate::models::GenerationDefaults> {
        generation::defaults(&self.eos, self.vocab_size(), json)
    }

    pub(super) fn layers(&self) -> usize {
        self.text.num_hidden_layers
    }

    pub(super) fn sliding(&self, layer: usize) -> bool {
        self.text.local_layer_ids.contains(&layer)
    }

    pub(super) fn looped(&self, layer: usize) -> bool {
        (self.text.loop_config.core_start..self.text.loop_config.core_end).contains(&layer)
    }

    /// A layer whose input becomes a block state and whose residual stream restarts.
    pub(super) fn boundary(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.text.n_layers_per_attn_res_block)
    }

    /// Bias extent of a layer's relative positions: the window for sliding layers.
    pub(super) fn extent(&self, layer: usize) -> usize {
        if self.sliding(layer) {
            self.text.sliding_window_size
        } else {
            self.text.rel_extent
        }
    }

    pub(super) fn expert_formats(&self) -> Result<&[ExpertFormat]> {
        self.experts.as_deref().ok_or_else(|| {
            invalid("Inkling 2 expert formats come from the checkpoint's hf_quant_config.json")
        })
    }

    /// Every layer application in execution order: layers before the loop, each pass of
    /// the looped layers, then the rest.
    pub(super) fn applications(&self) -> Vec<Application> {
        let l = &self.text.loop_config;
        let once = |layer| Application { layer, pass: 0 };
        (0..l.core_start)
            .map(once)
            .chain((0..l.num_passes).flat_map(|pass| {
                (l.core_start..l.core_end).map(move |layer| Application { layer, pass })
            }))
            .chain((l.core_end..self.layers()).map(once))
            .collect()
    }

    pub(super) fn conv_columns(&self) -> ConvColumns {
        let t = &self.text;
        let kv = t.num_key_value_heads * t.head_dim;
        ConvColumns {
            key: 0,
            value: kv,
            attention: 2 * kv,
            mlp: 2 * kv + t.hidden_size,
            width: 2 * kv + 2 * t.hidden_size,
        }
    }

    /// State of application `a` lives at layers `2a` (paged K/V) and `2a + 1` (the
    /// convolution histories, a recurrent slot).
    pub(super) fn description(&self, revision: String, kv: KvLayout) -> Result<ModelDescription> {
        let t = &self.text;
        let columns = self.conv_columns();
        let layers = self
            .applications()
            .iter()
            .flat_map(|a| {
                [
                    LayerStateKind::Attention {
                        kv_heads: t.num_key_value_heads,
                        head_dim: t.head_dim,
                        window: self.sliding(a.layer).then_some(t.sliding_window_size),
                    },
                    // The pools hold convolution histories beside a recurrent matrix; these
                    // layers have none, so each slot keeps a one-element placeholder.
                    LayerStateKind::Recurrent {
                        conv_channels: columns.width,
                        conv_width: t.sconv_kernel_size - 1,
                        heads: 1,
                        key_dim: 1,
                        value_dim: 1,
                        dtype: DType::F32,
                    },
                ]
            })
            .collect();
        let description = ModelDescription {
            identity: ModelIdentity {
                revision,
                state_format: 1,
            },
            layers,
            kv,
            modalities: vec![Modality::Text],
            vocab_size: self.vocab_size(),
            max_position: self.max_position(),
        };
        description.validate()?;
        Ok(description)
    }

    pub(super) fn layer_name(layer: usize) -> String {
        format!("{PREFIX}.layers.{layer}")
    }

    pub(super) fn expert_name(layer: usize, name: &str) -> String {
        format!("{PREFIX}.layers.{layer}.mlp.experts.{name}")
    }

    /// Every text tensor this architecture reads, with its stored shape and kind. Needs the
    /// expert formats.
    pub fn weight_shapes(&self) -> Result<BTreeMap<String, (Vec<usize>, Stored)>> {
        let t = &self.text;
        let (hidden, heads, dim) = (t.hidden_size, t.num_attention_heads, t.head_dim);
        let kv = t.num_key_value_heads * dim;
        let (taps, rank) = (t.sconv_kernel_size, t.loop_config.gate_rank);
        let loops = t.loop_config.core_end - t.loop_config.core_start;
        let mut shapes = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>, kind| {
            shapes.insert(name, (shape, kind));
        };
        let float = Stored::Float;
        add(
            format!("{PREFIX}.embed.weight"),
            vec![self.padded_vocab(), hidden],
            float,
        );
        add(
            format!("{PREFIX}.unembed.weight"),
            vec![self.padded_vocab(), hidden],
            float,
        );
        for name in ["embed_norm.weight", "norm.weight", "attn_res.output_query"] {
            add(format!("{PREFIX}.{name}"), vec![hidden], float);
        }
        for name in [
            "middle_cycle_cross_pass_queries",
            "middle_cycle_cross_pass_kv_queries",
        ] {
            add(format!("{PREFIX}.{name}"), vec![2 * loops, hidden], float);
        }
        for (i, format) in self.expert_formats()?.iter().enumerate() {
            let p = Self::layer_name(i);
            let a = format!("{p}.attn");
            let sliding = self.sliding(i);
            if i > 0 {
                let shape = if sliding {
                    vec![hidden]
                } else {
                    vec![3, hidden]
                };
                add(
                    format!("{PREFIX}.attn_res.attn_queries.{}", i - 1),
                    shape,
                    float,
                );
            }
            add(
                format!("{PREFIX}.attn_res.mlp_queries.{i}"),
                vec![hidden],
                float,
            );
            for (name, rows, cols) in [
                ("wq_du", heads * dim, hidden),
                ("gate_attn", heads * dim, hidden),
                ("wk_dv", kv, hidden),
                ("wv_dv", kv, hidden),
                ("wo_ud", hidden, heads * dim),
                ("wr_du", heads * t.d_rel, hidden),
                ("attn_sink_proj", heads, hidden),
            ] {
                add(format!("{a}.{name}.weight"), vec![rows, cols], float);
            }
            add(
                format!("{a}.rel_logits_proj.proj"),
                vec![t.d_rel, self.extent(i)],
                float,
            );
            add(format!("{a}.q_l2_clip.scale"), vec![heads, dim], float);
            add(
                format!("{a}.rel_l2_clip.scale"),
                vec![heads, t.d_rel],
                float,
            );
            if sliding {
                add(format!("{p}.attn_norm.weight"), vec![hidden], float);
                for x in ["k", "v"] {
                    add(format!("{a}.{x}_sconv.weight"), vec![kv, 1, taps], float);
                }
            } else {
                for x in ["q", "k", "v"] {
                    add(format!("{p}.{x}_input_norm.weight"), vec![hidden], float);
                }
                for x in ["k", "v"] {
                    add(format!("{a}.{x}_sconv.static_w"), vec![taps, kv], float);
                    let rows = t.num_key_value_heads * taps;
                    add(
                        format!("{a}.{x}_sconv.weight_proj"),
                        vec![rows, hidden],
                        float,
                    );
                }
            }
            for name in ["attn_sconv", "mlp_sconv"] {
                add(format!("{p}.{name}.weight"), vec![hidden, 1, taps], float);
            }
            add(format!("{p}.mlp_norm.weight"), vec![hidden], float);
            add(
                format!("{p}.mlp.gate.weight"),
                vec![t.n_routed_experts, hidden],
                float,
            );
            add(
                format!("{p}.mlp.gate.bias"),
                vec![t.n_routed_experts],
                float,
            );
            let banks = [
                ("w13_weight", 2 * t.intermediate_size, hidden),
                ("w2_weight", hidden, t.intermediate_size),
            ];
            for (name, rows, cols) in banks {
                let e = Self::expert_name(i, name);
                let experts = t.n_routed_experts;
                match format {
                    ExpertFormat::Nvfp4 => {
                        add(e.clone(), vec![experts, rows, cols / 2], Stored::Codes);
                        add(
                            format!("{e}_scale"),
                            vec![experts, rows, cols / 16],
                            Stored::Fp8,
                        );
                        add(format!("{e}_amax"), vec![experts], float);
                        add(format!("{e}_outer_scale_denominator"), vec![1], float);
                    }
                    ExpertFormat::Mxfp8 => {
                        add(e.clone(), vec![experts, rows, cols], Stored::Fp8);
                        add(
                            format!("{e}_scale"),
                            vec![experts, rows, cols / 32],
                            Stored::Codes,
                        );
                    }
                }
            }
            if self.looped(i) {
                for gate in ["attn", "k", "v", "mlp"] {
                    let c = format!("{p}.controllers.{gate}");
                    add(format!("{c}.down_weight"), vec![rank, 2 * hidden], float);
                    add(format!("{c}.forget_weight"), vec![hidden, rank], float);
                    add(format!("{c}.gain_weight"), vec![hidden, rank], float);
                }
            }
        }
        Ok(shapes)
    }

    /// F32 elements the model keeps besides projections, the embedding and expert
    /// banks: norms, scales, convolution taps, routers, residual queries, message gates and
    /// two factors per expert of every layer.
    pub(super) fn resident_f32_elements(&self) -> Result<usize> {
        let projected: std::collections::BTreeSet<String> = self
            .projection_layout()?
            .into_iter()
            .flat_map(|p| p.parts.into_iter().map(|(name, _)| name))
            .collect();
        let mut total = 0usize;
        for (name, (shape, kind)) in self.weight_shapes()? {
            let expert_part = name.contains(".mlp.experts.");
            if kind != Stored::Float
                || expert_part
                || projected.contains(&name)
                || name == format!("{PREFIX}.embed.weight")
            {
                continue;
            }
            total = total
                .checked_add(checked_elements(&shape)?)
                .ok_or_else(|| invalid("Inkling 2 weight size overflow"))?;
        }
        total
            .checked_add(checked_elements(&[
                self.layers(),
                self.text.n_routed_experts,
                2,
            ])?)
            .ok_or_else(|| invalid("Inkling 2 weight size overflow"))
    }
}

/// What a checkpoint tensor holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stored {
    /// BF16, F16 or F32 values.
    Float,
    /// Bytes: packed E2M1 codes or E8M0 scales.
    Codes,
    /// E4M3 values or scales.
    Fp8,
}

impl Stored {
    /// `float`, `codes` or `fp8`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Codes => "codes",
            Self::Fp8 => "fp8",
        }
    }
}

impl WeightLayout for Config {
    fn projection_layout(&self) -> Result<Vec<Projection>> {
        let t = &self.text;
        let (hidden, heads, dim) = (t.hidden_size, t.num_attention_heads, t.head_dim);
        let kv = t.num_key_value_heads * dim;
        let dynamic = t.num_key_value_heads * t.sconv_kernel_size;
        let mut projections = Vec::with_capacity(4 * self.layers() + 1);
        for i in 0..self.layers() {
            let a = format!("{}.attn", Self::layer_name(i));
            let sliding = self.sliding(i);
            // Everything the attention reads from its query input, in one matrix.
            projections.push(Projection::new(
                format!("{a}.query"),
                vec![
                    (format!("{a}.wq_du.weight"), heads * dim),
                    (format!("{a}.gate_attn.weight"), heads * dim),
                    (format!("{a}.wr_du.weight"), heads * t.d_rel),
                    (format!("{a}.attn_sink_proj.weight"), heads),
                ],
                hidden,
            ));
            for (x, name) in [("k", "wk_dv"), ("v", "wv_dv")] {
                let mut parts = vec![(format!("{a}.{name}.weight"), kv)];
                if !sliding {
                    parts.push((format!("{a}.{x}_sconv.weight_proj"), dynamic));
                }
                projections.push(Projection::new(format!("{a}.{x}"), parts, hidden));
            }
            projections.push(Projection::new(
                format!("{a}.output"),
                vec![(format!("{a}.wo_ud.weight"), hidden)],
                heads * dim,
            ));
        }
        projections.push(Projection::new(
            "lm_head".to_owned(),
            vec![(format!("{PREFIX}.unembed.weight"), self.padded_vocab())],
            hidden,
        ));
        Ok(projections)
    }
    fn embeddings_tied(&self) -> bool {
        false
    }
    fn vocabulary(&self) -> usize {
        self.padded_vocab()
    }
    fn residual_width(&self) -> usize {
        self.text.hidden_size
    }
    fn embedding_tensor_name(&self) -> String {
        format!("{PREFIX}.embed.weight")
    }
}
