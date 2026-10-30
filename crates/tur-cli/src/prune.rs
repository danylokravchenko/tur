//! Measure which experts calibration text routes to and write a smaller checkpoint without
//! the rest. The source stays read-only and is read through the bounded expert cache.
use crate::quantize::encode_samples;
use candle_core::DType;
use std::{io::Write, path::PathBuf};
use tur::{
    Result, TurError,
    models::{
        DEFAULT_KV_LAYOUT, Execution, LoadOptions, WeightFormat,
        gpt_oss::{GptOss, Pruning, divergence, plan_pruning, prune, refine},
    },
};

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Source gpt-oss `SafeTensors` checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// Exact identity of the source checkpoint, recorded in the output.
    #[arg(long)]
    revision: String,
    /// New checkpoint directory.
    #[arg(long)]
    output: PathBuf,
    /// Experts to keep in every layer.
    #[arg(long)]
    keep: usize,
    /// Calibration text files, in the mix the pruned model should serve. Never the
    /// held-out text it is judged on.
    #[arg(long, value_delimiter = ',', required = true)]
    calibration: Vec<PathBuf>,
    /// Calibration tokens, split into independent samples.
    #[arg(long, default_value_t = 4_096)]
    calibration_tokens: usize,
    #[arg(long, default_value_t = 512)]
    sample_tokens: usize,
    /// End-to-end KL evaluations spent swapping experts after the local search. On
    /// gpt-oss-20b, 48 lowered calibration KL by 28% and held-out KL not at all.
    #[arg(long, default_value_t = 0)]
    search_evaluations: usize,
    /// Calibration samples each of those evaluations runs, from the first.
    #[arg(long, default_value_t = 4)]
    search_samples: usize,
    /// Write only the kept experts as JSON to this path, and no checkpoint.
    #[arg(long)]
    plan_only: Option<PathBuf>,
    /// Device memory for experts while measuring.
    #[arg(long, default_value_t = 2_048)]
    expert_cache_mb: usize,
}

pub(crate) fn run(args: &Args) -> Result<()> {
    let samples = encode_samples(
        &args.model,
        &args.calibration,
        args.calibration_tokens,
        args.sample_tokens,
    )?;
    if samples.is_empty() {
        return Err(TurError::Other("calibration text is empty".into()));
    }
    let model = GptOss::load(
        &args.model,
        &LoadOptions {
            revision: args.revision.clone(),
            weights: WeightFormat::Q8_0.into(),
            device: tur::shared::inference_device(false)?,
            execution: Execution::Fused,
            kv: DEFAULT_KV_LAYOUT,
            recurrent: DType::F32,
            mtp: None,
            audio: false,
            expert_cache: Some(args.expert_cache_mb << 20),
        },
    )?;
    let probe = model.expert_probe(&samples)?;
    let mut plans = plan_pruning(&probe, args.keep)?;
    drop(probe);
    let judged = &samples[..args.search_samples.min(samples.len())];
    let reference = judged
        .iter()
        .map(|s| model.log_probs(s))
        .collect::<Result<Vec<_>>>()?;
    let (local, searched) = refine(&mut plans, args.search_evaluations, |kept| {
        model.hide_experts(Some(kept))?;
        let candidate = judged
            .iter()
            .map(|s| model.log_probs(s))
            .collect::<Result<Vec<_>>>()?;
        Ok(divergence(&reference, &candidate))
    })?;
    drop(model);
    let pruning = Pruning {
        source: args.revision.clone(),
        calibration: args
            .calibration
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        kept: plans
            .into_iter()
            .map(|plan| {
                let mut kept = plan.kept;
                kept.sort_unstable();
                kept
            })
            .collect(),
    };
    match &args.plan_only {
        Some(path) => std::fs::write(path, serde_json::to_string_pretty(&pruning)?)?,
        None => prune(&args.model, &args.output, &pruning)?,
    }
    writeln!(
        std::io::stdout(),
        "{}",
        serde_json::json!({
            "output": args.plan_only.as_ref().unwrap_or(&args.output),
            "samples": samples.len(),
            "tokens": samples.iter().map(Vec::len).sum::<usize>(),
            "kept_per_layer": args.keep,
            "calibration_kl_local": local,
            "calibration_kl_searched": searched,
        })
    )?;
    Ok(())
}
