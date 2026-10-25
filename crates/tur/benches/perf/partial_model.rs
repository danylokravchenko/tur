//! Calibration rankings for dense partial-model experiments. Layer ranking uses
//! residual cosine distance; neuron ranking uses activation-weighted output energy.
//! `just bench layer_influence|neuron_influence <checkpoint> <revision> <calibration.txt> [tokens=256]`.

use candle_core::{DType, Device, Tensor};
use serde_json::json;
use std::path::Path;
use tur_core::{Result, execution::invalid};
use tur_models::models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat};

fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

/// Mean per-token cosine distance in F64, avoiding device reduction roundoff in ranks.
fn influence(before: &Tensor, after: &Tensor) -> Result<f64> {
    let (tokens, width) = before.dims2()?;
    if tokens == 0 || after.dims() != before.dims() {
        return Err(invalid(
            "layer influence requires matching nonempty residuals",
        ));
    }
    let before = before.flatten_all()?.to_vec1::<f32>()?;
    let after = after.flatten_all()?.to_vec1::<f32>()?;
    let mut total = 0.0;
    for (a, b) in before.chunks_exact(width).zip(after.chunks_exact(width)) {
        let (mut dot, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (&a, &b) in a.iter().zip(b) {
            let (a, b) = (f64::from(a), f64::from(b));
            dot = a.mul_add(b, dot);
            aa = a.mul_add(a, aa);
            bb = b.mul_add(b, bb);
        }
        if aa == 0.0 || bb == 0.0 || !dot.is_finite() || !aa.is_finite() || !bb.is_finite() {
            return Err(invalid("nonfinite or zero residual in layer influence"));
        }
        total += 1.0 - (dot / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0);
    }
    Ok(total / tokens as f64)
}

fn neuron_scores(model: &Qwen35, tokens: &[u32]) -> Result<Vec<serde_json::Value>> {
    let calibration = model.calibrate(&[tokens.to_vec()], false)?;
    let mut scores = Vec::new();
    for projection in model.config().projections()? {
        if !projection.name.ends_with(".mlp.down_proj") {
            continue;
        }
        let activity = calibration
            .sites
            .get(&projection.name)
            .and_then(|site| site.importance(0))
            .ok_or_else(|| invalid("missing dense MLP activation statistics"))?;
        let energy = model
            .projection(&projection.name)?
            .dequantize()?
            .sqr()?
            .sum(0)?
            .to_vec1::<f32>()?;
        if activity.len() != energy.len() {
            return Err(invalid("MLP activation/weight dimensions disagree"));
        }
        let importance: Vec<f64> = activity
            .iter()
            .zip(energy)
            .map(|(&activity, energy)| f64::from(activity) * f64::from(energy))
            .collect();
        if importance
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
        {
            return Err(invalid("nonfinite or negative neuron importance"));
        }
        scores.push(json!({"layer": scores.len(), "importance": importance}));
    }
    Ok(scores)
}

pub(crate) fn run(args: &[String], width: bool) -> Result<()> {
    let [directory, revision, calibration, rest @ ..] = args else {
        return Err(invalid(
            "layer_influence|neuron_influence <checkpoint> <revision> <calibration.txt> [tokens]",
        ));
    };
    let limit = match rest {
        [] => 256,
        [tokens] => tokens
            .parse::<usize>()
            .map_err(|_| invalid("token limit"))?,
        _ => return Err(invalid("too many layer_influence arguments")),
    };
    if limit == 0 {
        return Err(invalid("token limit must be positive"));
    }
    let directory = Path::new(directory);
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let text = config.get("text_config").unwrap_or(&config);
    let kinds = text["layer_types"]
        .as_array()
        .ok_or_else(|| invalid("layer_types"))?;
    if text["model_type"] != "qwen3_5_text" {
        return Err(invalid(
            "layer influence pilot requires a dense Qwen3.5 checkpoint",
        ));
    }
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|error| invalid(error.to_string()))?;
    let calibration_text = std::fs::read_to_string(calibration)?;
    let encoding = tokenizer
        .encode(calibration_text.as_str(), false)
        .map_err(|error| invalid(error.to_string()))?;
    let tokens = &encoding.get_ids()[..limit.min(encoding.len())];
    if tokens.is_empty() {
        return Err(invalid("empty calibration"));
    }
    let model = Qwen35::load(
        directory,
        &LoadOptions {
            revision: revision.clone(),
            weights: if width {
                WeightFormat::F32
            } else {
                WeightFormat::Q8_0
            }
            .into(),
            device: Device::new_metal(0)?,
            execution: Execution::Fused,
            kv: tur_core::execution::KvLayout::new(16, tur_core::execution::KvFormat::F32),
            recurrent: DType::F32,
            mtp: None,
        },
    )?;
    let scores = if width {
        neuron_scores(&model, tokens)?
    } else {
        layer_scores(&model, tokens, kinds)?
    };
    let ids: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    let report = json!({
        "schema_version": 1, "source_revision": revision, "config": config,
        "axis": if width { "mlp_width" } else { "depth" },
        "weights": if width { "f32" } else { "q8_0" }, "kv": "f32", "recurrent": "f32",
        "calibration": {"file": calibration, "tokens": tokens.len(),
            "text_fnv1a": fingerprint(calibration_text.as_bytes()), "tokens_fnv1a": fingerprint(&ids)},
        "metric": if width { "E[post-SwiGLU activation^2] * squared L2 norm of matching down-projection column; ignores cross-neuron correlations" }
            else { "mean per-token 1 - cosine(layer input, layer output); layer 0 protected" },
        "scores": scores,
    });
    std::io::Write::write_all(
        &mut std::io::stdout(),
        serde_json::to_string_pretty(&report)?.as_bytes(),
    )?;
    Ok(())
}

fn layer_scores(
    model: &Qwen35,
    tokens: &[u32],
    kinds: &[serde_json::Value],
) -> Result<Vec<serde_json::Value>> {
    let (hidden, _) = model.trace(tokens, &[])?;
    model.device().synchronize()?;
    if hidden.len() != kinds.len() {
        return Err(invalid("trace layer count disagrees with config"));
    }
    let mut scores = Vec::new();
    for (index, pair) in hidden.windows(2).enumerate() {
        let layer = index + 1;
        scores.push(json!({
            "layer": layer, "kind": kinds[layer], "influence": influence(&pair[0], &pair[1])?,
            "protected": layer + 1 == hidden.len() || kinds[layer] == "full_attention",
        }));
    }
    Ok(scores)
}
