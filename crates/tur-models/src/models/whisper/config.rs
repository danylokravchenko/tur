//! Validated Whisper architecture and decoding settings.
use crate::audio::{WHISPER_FRAMES, WHISPER_WINDOW};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};
use tur_core::{Result, execution::invalid};

/// Encoder-decoder shape from `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
    pub activation_function: String,
    #[serde(default)]
    pub scale_embedding: bool,
    pub decoder_start_token_id: u32,
}

impl Config {
    /// Parse and check the equations implemented: GELU layers, one 30 s window of
    /// features halved by the second convolution, and head widths that divide the model.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        if raw["model_type"].as_str() != Some("whisper")
            || raw["architectures"]
                .as_array()
                .is_none_or(|a| a.len() != 1 || a[0] != "WhisperForConditionalGeneration")
        {
            return Err(invalid("not a Whisper speech-recognition checkpoint"));
        }
        let config: Self = serde_json::from_value(raw)?;
        if config.activation_function != "gelu"
            || config.d_model == 0
            || [
                config.encoder_attention_heads,
                config.decoder_attention_heads,
            ]
            .iter()
            .any(|&h| h == 0 || !config.d_model.is_multiple_of(h))
            || config.encoder_layers == 0
            || config.decoder_layers == 0
            || config.max_source_positions * 2 != WHISPER_FRAMES
            || config.max_target_positions < 8
            || config.decoder_start_token_id as usize >= config.vocab_size
        {
            return Err(invalid("unsupported Whisper configuration"));
        }
        Ok(config)
    }

    /// Checkpoint tensors and their exact shapes. The output projection is the token
    /// embedding, so no separate `proj_out` is read.
    pub fn weight_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let d = self.d_model;
        let mut shapes = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            shapes.insert(name, shape);
        };
        let encoder = "model.encoder";
        add(
            format!("{encoder}.conv1.weight"),
            vec![d, self.num_mel_bins, 3],
        );
        add(format!("{encoder}.conv1.bias"), vec![d]);
        add(format!("{encoder}.conv2.weight"), vec![d, d, 3]);
        add(format!("{encoder}.conv2.bias"), vec![d]);
        add(
            format!("{encoder}.embed_positions.weight"),
            vec![self.max_source_positions, d],
        );
        let decoder = "model.decoder";
        add(
            format!("{decoder}.embed_tokens.weight"),
            vec![self.vocab_size, d],
        );
        add(
            format!("{decoder}.embed_positions.weight"),
            vec![self.max_target_positions, d],
        );
        let attention = |add: &mut dyn FnMut(String, Vec<usize>), p: String| {
            for projection in ["q_proj", "v_proj", "out_proj"] {
                add(format!("{p}.{projection}.weight"), vec![d, d]);
                add(format!("{p}.{projection}.bias"), vec![d]);
            }
            // Whisper's keys carry no bias.
            add(format!("{p}.k_proj.weight"), vec![d, d]);
        };
        let norm = |add: &mut dyn FnMut(String, Vec<usize>), p: String| {
            add(format!("{p}.weight"), vec![d]);
            add(format!("{p}.bias"), vec![d]);
        };
        for (stack, layers, ffn, cross) in [
            (encoder, self.encoder_layers, self.encoder_ffn_dim, false),
            (decoder, self.decoder_layers, self.decoder_ffn_dim, true),
        ] {
            for i in 0..layers {
                let p = format!("{stack}.layers.{i}");
                attention(&mut add, format!("{p}.self_attn"));
                norm(&mut add, format!("{p}.self_attn_layer_norm"));
                if cross {
                    attention(&mut add, format!("{p}.encoder_attn"));
                    norm(&mut add, format!("{p}.encoder_attn_layer_norm"));
                }
                add(format!("{p}.fc1.weight"), vec![ffn, d]);
                add(format!("{p}.fc1.bias"), vec![ffn]);
                add(format!("{p}.fc2.weight"), vec![d, ffn]);
                add(format!("{p}.fc2.bias"), vec![d]);
                norm(&mut add, format!("{p}.final_layer_norm"));
            }
            norm(&mut add, format!("{stack}.layer_norm"));
        }
        shapes
    }
}

/// Decoding controls from `generation_config.json`, as Transformers' Whisper `generate`
/// applies them.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Generation {
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub no_timestamps_token_id: Option<u32>,
    /// `<|en|>`-style language tokens.
    #[serde(default)]
    pub lang_to_id: BTreeMap<String, u32>,
    #[serde(default)]
    pub task_to_id: BTreeMap<String, u32>,
    /// Never sampled.
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
    /// Never sampled as the first token after the prompt.
    #[serde(default)]
    pub begin_suppress_tokens: Vec<u32>,
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub is_multilingual: Option<bool>,
}

impl Generation {
    /// Read `generation_config.json` beside the checkpoint; Whisper needs its special
    /// tokens to build a prompt.
    pub fn from_dir(dir: &Path, config: &Config) -> Result<Self> {
        let generation: Self = serde_json::from_str(&std::fs::read_to_string(
            dir.join("generation_config.json"),
        )?)?;
        let vocab = config.vocab_size;
        let ids = generation
            .lang_to_id
            .values()
            .chain(generation.task_to_id.values())
            .chain(&generation.suppress_tokens)
            .chain(&generation.begin_suppress_tokens)
            .chain(generation.eos_token_id.iter())
            .chain(generation.no_timestamps_token_id.iter());
        if generation.eos_token_id.is_none()
            || generation.no_timestamps_token_id.is_none()
            || !generation.task_to_id.contains_key("transcribe")
            || ids.into_iter().any(|&id| id as usize >= vocab)
        {
            return Err(invalid("incomplete Whisper generation config"));
        }
        Ok(generation)
    }
}

/// Validate `preprocessor_config.json`, when present, against the implemented features.
pub(super) fn validate_preprocessor(dir: &Path, config: &Config) -> Result<()> {
    let raw = match std::fs::read_to_string(dir.join("preprocessor_config.json")) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let pre: serde_json::Value = serde_json::from_str(&raw)?;
    let is =
        |key: &str, value: usize| pre[key].is_null() || pre[key].as_u64() == Some(value as u64);
    if !(is("feature_size", config.num_mel_bins)
        && is("sampling_rate", crate::audio::SAMPLE_RATE as usize)
        && is("n_fft", 400)
        && is("hop_length", 160)
        && is("n_samples", WHISPER_WINDOW)
        && is("nb_max_frames", WHISPER_FRAMES))
    {
        return Err(invalid("unsupported Whisper feature extractor settings"));
    }
    Ok(())
}
