//! Measure which experts calibration text routes to and write a smaller checkpoint without
//! the rest. The source stays read-only and is read through the bounded expert cache.
use crate::{args::checkpoint_dir, quantize::encode_samples, ui::Console};
use candle_core::DType;
use std::{io::Write, path::PathBuf};
use tur::{
    Result, TurError,
    core::progress::Stage,
    models::{
        DEFAULT_KV_LAYOUT, Execution, LoadOptions, WeightFormat,
        gpt_oss::{GptOss, Pruning, divergence, plan_pruning, prune, refine},
    },
};

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Source gpt-oss `SafeTensors` checkpoint: a local directory, or a Hugging Face repository
    /// at `--revision`.
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
    /// Print the outcome as JSON on stdout.
    #[arg(long)]
    json: bool,
}

pub(crate) fn run(args: &mut Args, console: &Console) -> Result<()> {
    args.model = checkpoint_dir(&args.model, &args.revision)?;
    let expert_cache = args
        .expert_cache_mb
        .checked_mul(1 << 20)
        .ok_or_else(|| TurError::Other("expert cache size overflow".into()))?;
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
            expert_cache: Some(expert_cache),
        },
    )?;
    let probe = model.expert_probe(&samples)?;
    let planning = Stage::begin("Planning which experts to keep", 0, "");
    let mut plans = plan_pruning(&probe, args.keep)?;
    planning.finish();
    drop(probe);
    let judged = &samples[..args.search_samples.min(samples.len())];
    let mut measuring = Stage::begin(
        "Recording reference outputs",
        judged.len() as u64,
        "samples",
    );
    let reference = judged
        .iter()
        .map(|s| {
            let probs = model.log_probs(s);
            measuring.advance(1);
            probs
        })
        .collect::<Result<Vec<_>>>()?;
    measuring.finish();
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
    let output = args.plan_only.as_ref().unwrap_or(&args.output);
    if args.json {
        writeln!(
            std::io::stdout(),
            "{}",
            serde_json::json!({
                "output": output,
                "samples": samples.len(),
                "tokens": samples.iter().map(Vec::len).sum::<usize>(),
                "kept_per_layer": args.keep,
                "calibration_kl_local": local,
                "calibration_kl_searched": searched,
            })
        )?;
    }
    console.line(&format!(
        "  {} wrote {}",
        console.paint("32", "✓"),
        output.display()
    ));
    console.rows(&[
        ("Kept", format!("{} experts per layer", args.keep)),
        ("Calibration KL", format!("{local:.4} → {searched:.4} nats")),
    ]);
    Ok(())
}
