//! Offline weight preparation: calibrate on text, optionally allocate per-projection
//! precision under a byte budget from measured sensitivities, and write a prepacked
//! artifact that `run` and `serve` load as stored.
use std::{io::Write, path::PathBuf, time::Instant};
use tur::{
    Result, TurError,
    core::execution::{KvFormat, KvLayout},
    models::qwen3_5::{
        Config, Conversion, DEFAULT_RECURRENT_STATE, Embedding, Execution, LoadOptions, Method,
        Packer, Precision, Qwen35, Reference, WeightFormat, WeightPlan, allocate, artifact_config,
        sensitivities, write_artifact, write_calibrated_artifact,
    },
    quantization::Calibration,
};

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Source `SafeTensors` checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// Exact identity of the source checkpoint, recorded in the artifact.
    #[arg(long)]
    revision: String,
    /// New artifact directory.
    #[arg(long)]
    output: PathBuf,
    /// Rounding: rtn, imatrix (importance-weighted K-quant scales) or gptq.
    #[arg(long, default_value = "gptq")]
    method: Method,
    /// Every projection's precision, or the narrowest one when `--budget-bits` allocates.
    #[arg(long, default_value = "q4k")]
    weight_format: WeightFormat,
    /// Average bits per projection weight (embedding table excluded): measure each
    /// projection's sensitivity and upgrade the most sensitive from `--weight-format` to
    /// `q8_0`. Requires the complete F32 model in memory.
    #[arg(long)]
    budget_bits: Option<f64>,
    /// Embedding storage: head (tied checkpoints: gather rows of the packed LM head),
    /// or a table in f32, bf16, `q8_0`, q6k, q5k or q4k. Default: head when tied, else bf16.
    #[arg(long)]
    embedding: Option<String>,
    /// Calibration text files. Never the held-out evaluation text.
    #[arg(long, value_delimiter = ',')]
    calibration: Vec<PathBuf>,
    /// Calibration tokens, split into independent samples.
    #[arg(long, default_value_t = 16_384)]
    calibration_tokens: usize,
    #[arg(long, default_value_t = 512)]
    sample_tokens: usize,
    /// Calibration tokens over which sensitivities are measured, from the first samples.
    #[arg(long, default_value_t = 1024)]
    sensitivity_tokens: usize,
    /// GPTQ damping relative to the mean Hessian diagonal.
    #[arg(long, default_value_t = 0.01)]
    damp: f64,
    /// Write calibration, sensitivity and plan details as JSON here.
    #[arg(long)]
    report: Option<PathBuf>,
}

fn samples(args: &Args) -> Result<Vec<Vec<u32>>> {
    let tokenizer = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let mut samples = Vec::new();
    let mut budget = args.calibration_tokens;
    for path in &args.calibration {
        let text = std::fs::read_to_string(path)?;
        let ids = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?;
        for sample in ids.get_ids().chunks(args.sample_tokens.max(1)) {
            if budget == 0 {
                return Ok(samples);
            }
            let sample = &sample[..sample.len().min(budget)];
            budget -= sample.len();
            samples.push(sample.to_vec());
        }
    }
    Ok(samples)
}

pub(crate) fn quantize(args: &Args) -> Result<()> {
    let config = artifact_config(&args.model)?;
    let embedding = match args.embedding.as_deref() {
        None if config.tied_embeddings() => Embedding::Head,
        None => Embedding::Table(Precision::Bf16),
        Some("head") => Embedding::Head,
        Some(name) => Embedding::Table(name.parse().map_err(TurError::Other)?),
    };
    let base = WeightPlan {
        embedding,
        ..WeightPlan::uniform(&config, args.weight_format)?
    };
    let calibrated = args.method != Method::Rtn || args.budget_bits.is_some();
    let mut report = serde_json::json!({ "source": args.revision, "method": args.method });
    let started = Instant::now();
    let streaming = calibrated && args.budget_bits.is_none();
    let data = if calibrated {
        samples(args)?
    } else {
        Vec::new()
    };
    if calibrated && data.is_empty() {
        return Err(TurError::Other(
            "calibration needs --calibration text".into(),
        ));
    }
    let (calibration, plan) = match args.budget_bits {
        Some(bits) => {
            let (calibration, plan) =
                mixed_plan(args, &config, embedding, &data, bits, &mut report)?;
            (Some(calibration), plan)
        }
        None => (None, base),
    };
    let packer = Packer {
        method: args.method,
        damp: args.damp,
        calibration: calibration.as_ref(),
    };
    let packing = Instant::now();
    let calibration_description = args
        .calibration
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let conversion = Conversion {
        source: &args.model,
        identity: &args.revision,
        output: &args.output,
        calibration: &calibration_description,
    };
    let manifest = if streaming {
        let device = tur::shared::inference_device(false)?;
        write_calibrated_artifact(&conversion, &plan, &packer, &data, &device)?
    } else {
        write_artifact(&conversion, &plan, &packer)?
    };
    report["plan_bytes"] = plan.bytes(&config)?.into();
    report["packing_seconds"] = packing.elapsed().as_secs_f64().into();
    report["total_seconds"] = started.elapsed().as_secs_f64().into();
    report["manifest"] = serde_json::to_value(&manifest)?;
    let text = serde_json::to_string_pretty(&report)?;
    match &args.report {
        Some(path) => std::fs::write(path, text)?,
        None => writeln!(std::io::stdout(), "{text}")?,
    }
    Ok(())
}

fn mixed_plan(
    args: &Args,
    config: &Config,
    embedding: Embedding,
    samples: &[Vec<u32>],
    bits: f64,
    report: &mut serde_json::Value,
) -> Result<(Calibration, WeightPlan)> {
    let device = tur::shared::inference_device(false)?;
    let mut model = Qwen35::load(
        &args.model,
        &LoadOptions {
            revision: args.revision.clone(),
            weights: WeightFormat::F32.into(),
            device,
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F32),
            recurrent: DEFAULT_RECURRENT_STATE,
            mtp: None,
        },
    )?;
    let calibration = model.calibrate(samples, args.method == Method::Gptq)?;
    tracing::info!(
        tokens = calibration.tokens,
        samples = calibration.samples,
        "Calibrated"
    );
    let mut taken = 0;
    let measured: Vec<Vec<u32>> = samples
        .iter()
        .take_while(|s| {
            let take = taken < args.sensitivity_tokens;
            taken += s.len();
            take
        })
        .cloned()
        .collect();
    let reference = Reference::new(&model, measured)?;
    let packer = Packer {
        method: args.method,
        damp: args.damp,
        calibration: Some(&calibration),
    };
    let measured = sensitivities(
        &mut model,
        &reference,
        &[Precision::from(args.weight_format), Precision::Q8_0],
        &packer,
        |s| {
            tracing::info!(
                name = s.name,
                precision = s.precision.name(),
                kl = s.kl,
                "Sensitivity"
            );
        },
    )?;
    let weights: f64 = config
        .projections()?
        .iter()
        .map(|p| (p.rows * p.cols) as f64)
        .sum();
    let embedding_bytes = match embedding {
        Embedding::Head => 0,
        Embedding::Table(p) => p.bytes(config.vocab_size(), config.hidden_size())?,
    };
    let budget = embedding_bytes + (weights * bits / 8.0) as usize;
    let plan = allocate(
        config,
        &measured,
        budget,
        Precision::from(args.weight_format),
        embedding,
    )?;
    report["sensitivities"] = serde_json::to_value(&measured)?;
    report["budget_bytes"] = budget.into();
    Ok((calibration, plan))
}
