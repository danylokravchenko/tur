use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use candle_core::{DType, Device};
use clap::Parser;
use tracing::{debug, info};
use tur::{
    AutoModelFactory, ModelSource, Result, TurError,
    server::{
        AppState, run_server,
        worker::{PipelineWorker, WorkerConfig},
    },
};

/// A single model entry in the `--models` list.
///
/// Format accepted by clap: `<id>:<hf-model-id-or-path>[:<quantization>]`
///
/// Examples:
///   qwen3-0.6b:Qwen3-0.6B
///   granite:ibm-granite/granite-4.1-3b:Q4_K_M
#[derive(Debug, Clone)]
struct ModelSpec {
    /// Key used in API requests (the `model` field).
    id: String,
    /// HuggingFace model ID or local path.
    source: String,
    /// Optional GGUF quantization level (e.g. Q4_K_M).
    quantization: Option<String>,
}

impl std::str::FromStr for ModelSpec {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(3, ':').collect();
        match parts.as_slice() {
            [id, source] => Ok(Self {
                id: id.to_string(),
                source: source.to_string(),
                quantization: None,
            }),
            [id, source, quant] => Ok(Self {
                id: id.to_string(),
                source: source.to_string(),
                quantization: Some(quant.to_string()),
            }),
            _ => Err(format!(
                "invalid model spec '{s}'; expected <id>:<source>[:<quantization>]"
            )),
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Tur - OpenAI-compatible multi-model inference server",
    long_about = None
)]
struct Args {
    /// Models to serve.  Repeat the flag for each model.
    /// Format: <api-id>:<hf-model-id-or-local-path>[:<quantization>]
    ///
    /// Examples:
    ///   --model qwen3-0.6b:Qwen3-0.6B
    ///   --model granite:ibm-granite/granite-4.1-3b:Q4_K_M
    ///
    /// When omitted, the server starts with the two default models.
    #[arg(long = "model", value_name = "SPEC")]
    models: Vec<ModelSpec>,

    /// Host address to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value = "0.7")]
    temperature: Option<f64>,

    /// Nucleus sampling cutoff.
    #[arg(long, default_value = "0.95")]
    top_p: Option<f64>,

    /// Random seed.
    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,

    /// Repetition penalty (1.0 = none).
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Context window for repetition penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Maximum number of requests processed in parallel by each model's
    /// continuous batch scheduler.
    #[arg(long, default_value_t = 8, env = "MAX_BATCH_SIZE")]
    max_batch_size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tur::shared::init_tracing();
    let args = Args::parse();

    let device = Device::new_metal(0)?;
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };
    debug!("Device: {:?}, DType: {:?}", device, dtype);

    // Default to the two bundled models when the user doesn't specify any.
    let specs: Vec<ModelSpec> = if args.models.is_empty() {
        vec![
            ModelSpec {
                id: "qwen3-0.6b".to_string(),
                source: "Qwen3-0.6B".to_string(),
                quantization: None,
            },
            // ModelSpec {
            //     id: "granite4.1-3b".to_string(),
            //     source: "ibm-granite/granite-4.1-3b".to_string(),
            //     quantization: None,
            // },
        ]
    } else {
        args.models.clone()
    };

    if specs.is_empty() {
        return Err(TurError::Other(
            "No models specified. Use --model <id>:<source>[:<quant>]".to_string(),
        ));
    }

    let default_model = specs[0].id.clone();

    let mut workers: HashMap<String, PipelineWorker> = HashMap::new();

    for spec in &specs {
        let source = if spec.source.starts_with('/') || spec.source.starts_with('.') {
            ModelSource::LocalPath(spec.source.clone())
        } else {
            ModelSource::HuggingFace(spec.source.clone())
        };

        info!(
            model_id = %spec.id,
            source = %spec.source,
            quantization = ?spec.quantization,
            max_batch_size = args.max_batch_size,
            "Loading model…"
        );

        let factory = Arc::new(AutoModelFactory::new(
            source,
            spec.quantization.clone(),
            device.clone(),
            dtype,
        ));

        let worker = PipelineWorker::spawn(WorkerConfig {
            factory,
            seed: args.seed,
            temperature: args.temperature,
            top_p: args.top_p,
            repeat_penalty: args.repeat_penalty,
            repeat_last_n: args.repeat_last_n,
            max_batch_size: args.max_batch_size,
        })?;

        info!(model_id = %spec.id, "Worker ready");
        workers.insert(spec.id.clone(), worker);
    }

    let state = AppState {
        workers,
        default_model,
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|e| TurError::Other(format!("Invalid address: {e}")))?;

    run_server(state, addr).await
}
