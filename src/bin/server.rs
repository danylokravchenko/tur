use std::{net::SocketAddr, sync::Arc};

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

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Tur - OpenAI-compatible inference server",
    long_about = None
)]
struct Args {
    /// HuggingFace model ID (e.g. "Qwen3-0.6B").
    #[arg(long, env = "HF_MODEL_ID")]
    model_id: Option<String>,

    /// Local path to model weights (safetensors or GGUF).
    #[arg(long, env = "MODEL_WEIGHT_PATH")]
    weight_path: Option<String>,

    /// GGUF quantization level (e.g. Q4_K_M).  Downloads from unsloth repo.
    #[arg(long, short = 'q', env = "QUANTIZATION")]
    quantization: Option<String>,

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
}

#[tokio::main]
async fn main() -> Result<()> {
    tur::shared::init_tracing();
    let args = Args::parse();

    let source = match (args.model_id.clone(), args.weight_path) {
        (Some(id), _) => ModelSource::HuggingFace(id),
        (None, Some(p)) => ModelSource::LocalPath(p),
        (None, None) => {
            return Err(TurError::Other(
                "Provide --model-id or --weight-path".to_string(),
            ));
        }
    };

    let model_id = match &source {
        ModelSource::HuggingFace(id) => id.clone(),
        ModelSource::LocalPath(p) => p.clone(),
    };

    let device = Device::new_metal(0)?;
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };

    debug!("Device: {:?}, DType: {:?}", device, dtype);

    let factory = Arc::new(AutoModelFactory::new(
        source,
        args.quantization,
        device,
        dtype,
    ));

    info!("Spawning pipeline worker (model loading happens here)…");
    let worker = PipelineWorker::spawn(WorkerConfig {
        factory,
        seed: args.seed,
        temperature: args.temperature,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
    })?;

    let state = AppState { worker, model_id };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|e| TurError::Other(format!("Invalid address: {e}")))?;

    run_server(state, addr).await
}
