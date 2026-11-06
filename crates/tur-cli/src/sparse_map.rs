//! Fit a Qwen3.5-family model's training-free latent sparse attention index map: each full
//! attention layer's leading 64 principal directions of its normalized position-free keys
//! over calibration text, as the reference's PCA map takes them.
use crate::{args::checkpoint_dir, quantize::encode_samples, ui::Console};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tur::{
    Result, TurError,
    core::progress::Stage,
    models::{
        DEFAULT_KV_LAYOUT, Execution, LoadOptions, Manifest, WeightFormat, Weights, is_artifact,
        qwen3_5::{Qwen35, principal_map},
    },
};

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Qwen3.5-family checkpoint, dense or `MoE`: a local directory, a prepacked artifact
    /// directory, or a Hugging Face repository at `--revision`.
    #[arg(long)]
    model: PathBuf,
    /// Exact identity of the checkpoint.
    #[arg(long)]
    revision: String,
    /// Index map file to write, for `--sparse-map`.
    #[arg(long)]
    output: PathBuf,
    /// Calibration text files, one long document each, in the mix the model will serve:
    /// never the held-out text it is judged on. The reference fits on eight.
    #[arg(long, value_delimiter = ',', required = true)]
    calibration: Vec<PathBuf>,
    /// Tokens of each calibration document, its first, prefilled as one sequence.
    #[arg(long, default_value_t = 32_768)]
    context: usize,
    /// Key rows each document contributes per layer, spread evenly across it.
    #[arg(long, default_value_t = 8_192)]
    rows: usize,
    /// Weights to run the calibration text through, as the model will serve them (default:
    /// `q8_0`). A prepacked artifact loads as stored.
    #[arg(long)]
    weight_format: Option<WeightFormat>,
    /// Keep routed experts on disk and read them into this many MiB, for `MoE` models
    /// larger than the GPU's working set (default: every expert resident).
    #[arg(long)]
    expert_cache_mb: Option<usize>,
}

/// The weights `args` names: an artifact as stored, or a checkpoint converted at load.
fn weights(args: &Args) -> Result<Weights> {
    match (is_artifact(&args.model), args.weight_format) {
        (true, None) => Ok(Weights::Prepacked(Arc::new(Manifest::read(&args.model)?))),
        (true, Some(_)) => Err(TurError::Other(
            "checkpoint is a prepacked artifact; omit --weight-format".into(),
        )),
        (false, format) => Ok(format.unwrap_or(WeightFormat::Q8_0).into()),
    }
}

pub(crate) fn run(args: &mut Args, console: &Console) -> Result<()> {
    args.model = checkpoint_dir(&args.model, &args.revision)?;
    let mut documents = Vec::with_capacity(args.calibration.len());
    for path in &args.calibration {
        let first = encode_samples(
            &args.model,
            std::slice::from_ref(path),
            args.context,
            args.context,
        )?;
        match first.into_iter().next() {
            Some(document) if document.len() >= 512 => documents.push(document),
            _ => {
                return Err(TurError::Other(format!(
                    "{} holds fewer than 512 tokens, one block of keys",
                    path.display()
                )));
            }
        }
    }
    let model = Qwen35::load(
        &args.model,
        &LoadOptions {
            revision: args.revision.clone(),
            weights: weights(args)?,
            device: tur::shared::inference_device(false)?,
            execution: Execution::Fused,
            kv: DEFAULT_KV_LAYOUT,
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            vision: false,
            video: false,
            expert_cache: args
                .expert_cache_mb
                .map(|mb| {
                    mb.checked_mul(1 << 20)
                        .ok_or_else(|| TurError::Other("expert cache size overflow".into()))
                })
                .transpose()?,
        },
    )?;
    let mut stage = Stage::begin("Collecting keys", documents.len() as u64, "documents");
    let mut rows: Vec<(usize, Vec<f32>)> = Vec::new();
    for document in &documents {
        for (layer, sampled) in model.index_rows(document, args.rows)? {
            match rows.iter_mut().find(|(l, _)| *l == layer) {
                Some((_, all)) => all.extend(sampled),
                None => rows.push((layer, sampled)),
            }
        }
        stage.advance(1);
    }
    stage.finish();
    drop(model);
    let fitting = Stage::begin("Fitting index maps", 0, "");
    let mut maps = HashMap::new();
    for (layer, rows) in &rows {
        let dim = rows.len() / sampled_rows(&documents, args.rows).max(1);
        let map = principal_map(rows, dim)?;
        maps.insert(
            format!("layer.{layer}"),
            Tensor::from_vec(map, (dim, 64), &Device::Cpu)?,
        );
    }
    fitting.finish();
    candle_core::safetensors::save(&maps, &args.output)?;
    console.line(&format!(
        "  {} wrote {}",
        console.paint("32", "✓"),
        args.output.display()
    ));
    console.rows(&[
        ("Layers", maps.len().to_string()),
        (
            "Calibration",
            format!(
                "{} documents, {} tokens",
                documents.len(),
                documents.iter().map(Vec::len).sum::<usize>()
            ),
        ),
    ]);
    Ok(())
}

/// Rows each layer collects over `documents`, at most `rows` from each.
fn sampled_rows(documents: &[Vec<u32>], rows: usize) -> usize {
    documents
        .iter()
        .map(|d| {
            d.iter()
                .step_by((d.len() / rows.max(1)).max(1))
                .take(rows)
                .count()
        })
        .sum()
}
