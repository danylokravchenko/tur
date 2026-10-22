#![forbid(unsafe_code)]
use clap::Parser;
use std::{collections::HashMap, net::SocketAddr, path::PathBuf};
use tur::{
    ModelFactory, ModelSource, Result, TurError,
    models::qwen3_5::WeightFormat,
    server::{
        AppState, run_server,
        worker::{PipelineWorker, WorkerConfig},
    },
};

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Tur hybrid text inference server (one resident model)"
)]
struct Args {
    /// Fully qualified Hugging Face repository; requires an immutable --revision.
    #[arg(
        long,
        env = "HF_MODEL_ID",
        conflicts_with = "weight_path",
        required_unless_present = "weight_path"
    )]
    model_id: Option<String>,
    /// Local SafeTensors checkpoint directory.
    #[arg(long, env = "MODEL_WEIGHT_PATH", conflicts_with = "model_id")]
    weight_path: Option<PathBuf>,
    /// Remote commit or explicit immutable identity for local weights.
    #[arg(long)]
    revision: String,
    /// Explicit SafeTensors conversion to f32, q8_0 or q4k resident projections.
    #[arg(long, default_value = "q8_0")]
    weight_format: WeightFormat,
    /// Model ID accepted by /v1/chat/completions and listed by /v1/models.
    #[arg(long, default_value = "qwen-hybrid")]
    served_model_name: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value_t = 8)]
    max_batch_size: usize,
    #[arg(long, default_value_t = 256)]
    prefill_chunk_size: usize,
    /// CPU numerical reference for development checks.
    #[arg(long)]
    cpu_reference: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tur::shared::init_tracing()?;
    let args = Args::parse();
    if args.served_model_name.trim().is_empty() {
        return Err(TurError::Other(
            "served model name must not be empty".into(),
        ));
    }
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|error| TurError::Other(format!("Invalid address: {error}")))?;
    let source = match (args.model_id, args.weight_path) {
        (Some(id), None) => ModelSource::HuggingFace(id),
        (None, Some(path)) => ModelSource::LocalPath(path),
        _ => {
            return Err(TurError::Other(
                "provide exactly one checkpoint source".into(),
            ));
        }
    };
    // Load before binding the listener so readiness means a validated, usable checkpoint.
    let worker = PipelineWorker::spawn(WorkerConfig {
        factory: ModelFactory {
            source,
            revision: args.revision,
            format: args.weight_format,
            device: tur::shared::inference_device(args.cpu_reference)?,
        },
        max_batch_size: args.max_batch_size,
        prefill_chunk_size: args.prefill_chunk_size,
    })?;
    let default_model = args.served_model_name;
    let workers = HashMap::from([(default_model.clone(), worker)]);
    run_server(
        AppState {
            workers,
            default_model,
        },
        addr,
    )
    .await
}
