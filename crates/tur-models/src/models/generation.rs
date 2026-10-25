//! Stop-token and generation metadata shared by all text architectures.
use serde::Deserialize;
use tur_core::{Result, execution::invalid};

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

pub(crate) fn defaults(
    eos: &[u32],
    vocab: usize,
    json: Option<&str>,
) -> Result<GenerationDefaults> {
    let mut defaults = GenerationDefaults {
        eos_tokens: eos.to_vec(),
        ..Default::default()
    };
    let Some(json) = json else {
        return Ok(defaults);
    };
    let raw: RawGeneration = serde_json::from_str(json)?;
    if let Some(eos) = &raw.eos_token_id {
        for id in parse_tokens(eos, vocab)? {
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

pub(crate) fn parse_tokens(raw: &serde_json::Value, vocab: usize) -> Result<Vec<u32>> {
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
