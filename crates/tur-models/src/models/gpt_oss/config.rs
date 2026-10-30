//! Validated gpt-oss settings, selected from explicit architecture fields only.
use crate::models::{Projection, generation, weights::WeightLayout};
use serde::Deserialize;
use std::collections::BTreeMap;
use tur_core::{
    Result,
    execution::{KvLayout, LayerStateKind, Modality, ModelDescription, ModelIdentity, invalid},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum LayerKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct Yarn {
    rope_type: YarnKind,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    original_max_position_embeddings: usize,
    truncate: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum YarnKind {
    Yarn,
}

#[derive(Clone, Debug, Deserialize)]
struct Quantization {
    quant_method: String,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Text {
    model_type: String,
    architectures: Vec<String>,
    attention_bias: bool,
    attention_dropout: f64,
    eos_token_id: serde_json::Value,
    pub(super) head_dim: usize,
    hidden_act: String,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) layer_types: Vec<LayerKind>,
    max_position_embeddings: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_experts_per_tok: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) num_local_experts: usize,
    pub(super) rms_norm_eps: f64,
    pub(super) sliding_window: usize,
    pub(super) swiglu_limit: f32,
    tie_word_embeddings: bool,
    vocab_size: usize,
    rope_theta: f64,
    rope_scaling: Yarn,
    quantization_config: Quantization,
}

/// gpt-oss settings. Construction validates the complete text layout.
#[derive(Clone, Debug)]
pub struct Config {
    pub(super) text: Text,
    eos: Vec<u32>,
}

/// Transformers' default for the gate's sigmoid steepness; the checkpoint omits it.
pub(super) const SWIGLU_ALPHA: f32 = 1.702;

impl Config {
    /// Parse explicit architecture identifiers, never a model name.
    pub fn from_json(json: &str) -> Result<Self> {
        let text: Text = serde_json::from_str(json)?;
        let (hidden, inner) = (text.hidden_size, text.intermediate_size);
        if text.model_type != "gpt_oss"
            || text.architectures != ["GptOssForCausalLM"]
            || text.hidden_act != "silu"
            || !text.attention_bias
            || text.attention_dropout != 0.0
            || text.tie_word_embeddings
            || text.quantization_config.quant_method != "mxfp4"
        {
            return Err(invalid("unsupported gpt-oss architecture settings"));
        }
        // The expert kernels read whole 32-value MXFP4 blocks of each row.
        if hidden == 0
            || inner == 0
            || !hidden.is_multiple_of(32)
            || !inner.is_multiple_of(32)
            || !matches!(text.head_dim, 64 | 128 | 256)
            || text.num_key_value_heads == 0
            || text.num_attention_heads == 0
            || !text
                .num_attention_heads
                .is_multiple_of(text.num_key_value_heads)
            || text.num_local_experts == 0
            || text.num_local_experts > 256
            || text.num_experts_per_tok == 0
            || text.num_experts_per_tok > text.num_local_experts
            || text.layer_types.is_empty()
            || text.sliding_window == 0
            || text.vocab_size == 0
            || text.max_position_embeddings == 0
            || !text.swiglu_limit.is_finite()
            || text.swiglu_limit <= 0.0
            || text.rope_theta <= 1.0
            || text.rope_scaling.factor < 1.0
            || text.rope_scaling.original_max_position_embeddings == 0
            || text.rope_scaling.beta_fast <= 0.0
            || text.rope_scaling.beta_slow <= 0.0
        {
            return Err(invalid(
                "invalid gpt-oss dimensions, routing or rotary settings",
            ));
        }
        let YarnKind::Yarn = text.rope_scaling.rope_type;
        let eos = generation::parse_tokens(&text.eos_token_id, text.vocab_size)?;
        Ok(Self { text, eos })
    }

    /// Exclusive text position limit.
    pub fn max_position(&self) -> usize {
        self.text.max_position_embeddings
    }

    /// Output vocabulary width.
    pub fn vocab_size(&self) -> usize {
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
        self.text.layer_types.len()
    }

    pub(super) fn windowed(&self, layer: usize) -> bool {
        self.text.layer_types[layer] == LayerKind::SlidingAttention
    }

    /// Inverse `YaRN` frequencies and the amplitude on `cos` and `sin`, in F32 like the
    /// reference: a one-ulp frequency difference grows linearly with position, so the
    /// blend stays unfused to round exactly as it does.
    #[allow(
        clippy::suboptimal_flops,
        reason = "a fused multiply-add rounds differently from the reference"
    )]
    pub(super) fn yarn(&self) -> (Vec<f32>, f32) {
        let yarn = self.text.rope_scaling;
        let dim = self.text.head_dim;
        let (base, factor) = (self.text.rope_theta, yarn.factor);
        let original = yarn.original_max_position_embeddings as f64;
        let correction = |rotations: f64| {
            dim as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let (mut low, mut high) = (correction(yarn.beta_fast), correction(yarn.beta_slow));
        if yarn.truncate {
            low = low.floor();
            high = high.ceil();
        }
        let low = low.max(0.0) as f32;
        let mut high = high.min(dim as f64 - 1.0) as f32;
        if low.to_bits() == high.to_bits() {
            high += 0.001;
        }
        let frequencies = (0..dim / 2)
            .map(|i| {
                let position = (base as f32).powf((2 * i) as f32 / dim as f32);
                let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                (1.0 / (factor as f32 * position)) * ramp + (1.0 / position) * (1.0 - ramp)
            })
            .collect();
        let amplitude = if factor <= 1.0 {
            1.0
        } else {
            (0.1 * factor.ln() + 1.0) as f32
        };
        (frequencies, amplitude)
    }

    pub(super) fn description(&self, revision: String, kv: KvLayout) -> Result<ModelDescription> {
        let description = ModelDescription {
            identity: ModelIdentity {
                revision,
                state_format: 1,
            },
            layers: (0..self.layers())
                .map(|i| LayerStateKind::Attention {
                    kv_heads: self.text.num_key_value_heads,
                    head_dim: self.text.head_dim,
                    window: self.windowed(i).then_some(self.text.sliding_window),
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

    /// Every text tensor this architecture reads, with its exact stored shape.
    #[allow(
        clippy::many_single_char_names,
        reason = "short module prefixes keep the tensor table readable"
    )]
    pub fn weight_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let t = &self.text;
        let (hidden, inner, experts) = (t.hidden_size, t.intermediate_size, t.num_local_experts);
        let query = t.num_attention_heads * t.head_dim;
        let kv = t.num_key_value_heads * t.head_dim;
        let mut shapes = BTreeMap::new();
        shapes.insert(
            "model.embed_tokens.weight".to_owned(),
            vec![t.vocab_size, hidden],
        );
        shapes.insert("model.norm.weight".to_owned(), vec![hidden]);
        shapes.insert("lm_head.weight".to_owned(), vec![t.vocab_size, hidden]);
        for i in 0..self.layers() {
            let p = format!("model.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                shapes.insert(format!("{p}.{name}.weight"), vec![hidden]);
            }
            let a = format!("{p}.self_attn");
            shapes.insert(format!("{a}.q_proj.weight"), vec![query, hidden]);
            shapes.insert(format!("{a}.q_proj.bias"), vec![query]);
            shapes.insert(format!("{a}.k_proj.weight"), vec![kv, hidden]);
            shapes.insert(format!("{a}.k_proj.bias"), vec![kv]);
            shapes.insert(format!("{a}.v_proj.weight"), vec![kv, hidden]);
            shapes.insert(format!("{a}.v_proj.bias"), vec![kv]);
            shapes.insert(format!("{a}.o_proj.weight"), vec![hidden, query]);
            shapes.insert(format!("{a}.o_proj.bias"), vec![hidden]);
            shapes.insert(format!("{a}.sinks"), vec![t.num_attention_heads]);
            let m = format!("{p}.mlp");
            shapes.insert(format!("{m}.router.weight"), vec![experts, hidden]);
            shapes.insert(format!("{m}.router.bias"), vec![experts]);
            for (name, rows, cols) in [
                ("gate_up_proj", 2 * inner, hidden),
                ("down_proj", hidden, inner),
            ] {
                let e = format!("{m}.experts.{name}");
                shapes.insert(format!("{e}_blocks"), vec![experts, rows, cols / 32, 16]);
                shapes.insert(format!("{e}_scales"), vec![experts, rows, cols / 32]);
                shapes.insert(format!("{e}_bias"), vec![experts, rows]);
            }
        }
        shapes
    }

    /// Experts per layer and the shapes of one expert's `gate_up` and `down` banks.
    #[cfg(feature = "metal")]
    pub(super) fn bank_shapes(
        &self,
    ) -> (
        usize,
        tur_metal::experts::BankShape,
        tur_metal::experts::BankShape,
    ) {
        use tur_metal::experts::BankShape;
        let t = &self.text;
        (
            t.num_local_experts,
            BankShape {
                rows: 2 * t.intermediate_size,
                cols: t.hidden_size,
            },
            BankShape {
                rows: t.hidden_size,
                cols: t.intermediate_size,
            },
        )
    }

    /// Where a validated checkpoint stores every layer's expert banks.
    #[cfg(feature = "metal")]
    pub(super) fn expert_sources(
        &self,
        checkpoint: &crate::models::checkpoint::Checkpoint,
    ) -> Result<tur_metal::experts::ExpertSources> {
        use tur_metal::experts::{BankSource, ExpertSources};
        let (experts, gate_up, down) = self.bank_shapes();
        let bank = |layer: usize, name: &str| -> Result<BankSource> {
            let e = format!("model.layers.{layer}.mlp.experts.{name}");
            let (file, blocks) = checkpoint.location(&format!("{e}_blocks"))?;
            let (scales_file, scales) = checkpoint.location(&format!("{e}_scales"))?;
            if file != scales_file {
                return Err(invalid(format!("{e} spans checkpoint files")));
            }
            Ok(BankSource {
                file: file.to_path_buf(),
                blocks,
                scales,
            })
        };
        Ok(ExpertSources {
            experts,
            top_k: self.text.num_experts_per_tok,
            gate_up,
            down,
            layers: (0..self.layers())
                .map(|i| Ok([bank(i, "gate_up_proj")?, bank(i, "down_proj")?]))
                .collect::<Result<_>>()?,
        })
    }

    /// Whether `name` is stored as packed MXFP4 bytes rather than floats.
    pub(super) fn is_packed(name: &str) -> bool {
        name.ends_with("_blocks") || name.ends_with("_scales")
    }
}

impl WeightLayout for Config {
    fn projection_layout(&self) -> Result<Vec<Projection>> {
        let t = &self.text;
        let (hidden, dim) = (t.hidden_size, t.head_dim);
        let query = t.num_attention_heads * dim;
        let kv = t.num_key_value_heads * dim;
        let mut projections = Vec::with_capacity(3 * self.layers() + 1);
        for i in 0..self.layers() {
            let a = format!("model.layers.{i}.self_attn");
            projections.push(Projection::new(
                format!("{a}.q_proj"),
                vec![(format!("{a}.q_proj.weight"), query)],
                hidden,
            ));
            projections.push(Projection::new(
                format!("{a}.kv_proj"),
                vec![
                    (format!("{a}.k_proj.weight"), kv),
                    (format!("{a}.v_proj.weight"), kv),
                ],
                hidden,
            ));
            projections.push(Projection::new(
                format!("{a}.o_proj"),
                vec![(format!("{a}.o_proj.weight"), hidden)],
                query,
            ));
        }
        projections.push(Projection::new(
            "lm_head".to_owned(),
            vec![("lm_head.weight".to_owned(), t.vocab_size)],
            hidden,
        ));
        Ok(projections)
    }
    fn embeddings_tied(&self) -> bool {
        false
    }
    fn vocabulary(&self) -> usize {
        self.vocab_size()
    }
    fn residual_width(&self) -> usize {
        self.text.hidden_size
    }
}
