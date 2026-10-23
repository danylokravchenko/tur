#![forbid(unsafe_code)]
use clap::Parser;
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, time::Duration};
use tur::{
    ModelFactory, ModelSource, Result, TurError,
    backend::{memory::DeviceBudget, pipeline::PipelineConfig, prefix::PrefixConfig},
    core::execution::{KvFormat, KvLayout},
    models::qwen3_5::{
        DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat, parse_recurrent_state,
    },
    server::{
        AppState, run_server,
        worker::{PipelineWorker, StreamConfig, WorkerConfig},
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
    /// Local `SafeTensors` checkpoint directory.
    #[arg(long, env = "MODEL_WEIGHT_PATH", conflicts_with = "model_id")]
    weight_path: Option<PathBuf>,
    /// Remote commit or explicit immutable identity for local weights.
    #[arg(long)]
    revision: String,
    /// Explicit `SafeTensors` conversion to f32, `q8_0` or q4k resident projections.
    #[arg(long, default_value = "q8_0")]
    weight_format: WeightFormat,
    /// Model ID accepted by /v1/chat/completions and listed by /v1/models.
    #[arg(long, default_value = "qwen-hybrid")]
    served_model_name: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Requests holding model state at once.
    #[arg(long, default_value_t = 8)]
    max_batch_size: usize,
    /// Largest prefill chunk of one request in one pass.
    #[arg(long, default_value_t = 256)]
    prefill_chunk_size: usize,
    /// Tokens in one packed pass. Default: one prefill chunk beside every decode.
    #[arg(long)]
    max_batch_tokens: Option<usize>,
    /// Requests queued for admission before new ones are rejected.
    #[arg(long, default_value_t = 256)]
    max_waiting: usize,
    /// Pass duration the adaptive prefill budget aims for, in milliseconds.
    #[arg(long, default_value_t = 60)]
    target_pass_ms: u64,
    /// Fail requests not finished this many seconds after arrival.
    #[arg(long)]
    request_timeout_secs: Option<u64>,
    /// K/V storage: f32, f16, q8 (one scale per 32 dimensions, half of F16's memory) or
    /// q8_<group>.
    #[arg(long, default_value_t = DEFAULT_KV_LAYOUT.format)]
    kv_format: KvFormat,
    /// Tokens per K/V page.
    #[arg(long, default_value_t = DEFAULT_KV_LAYOUT.page_tokens)]
    kv_page_tokens: usize,
    /// Recurrent matrix state storage: f32 (default) or f16, which halves recurrent memory
    /// and traffic with stochastic rounding but adds error that grows with decode length.
    #[arg(long, value_parser = ["f32", "f16"])]
    recurrent_state: Option<String>,
    /// Memory for K/V pages and recurrent state, in MiB. Default: room for each concurrent
    /// request's 8K-token context within the device's working set.
    #[arg(long)]
    state_memory_mb: Option<usize>,
    /// Prefix snapshots retained beside running requests; each holds one recurrent
    /// state. Default: one per --max-batch-size; 0 disables prefix reuse.
    #[arg(long)]
    prefix_snapshots: Option<usize>,
    /// Tokens between prefix snapshots; a multiple of --kv-page-tokens.
    #[arg(long, default_value_t = PrefixConfig::DEFAULT_SPACING)]
    prefix_spacing: usize,
    /// Streamed messages a client may leave unread before text is buffered.
    #[arg(long, default_value_t = 64)]
    stream_channel_capacity: usize,
    /// Undelivered text, in KiB, beyond which a stalled streaming client is cancelled.
    #[arg(long, default_value_t = 16)]
    stream_buffer_kb: usize,
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
    let mut pipeline = PipelineConfig {
        state_bytes: args.state_memory_mb.map(|mb| mb << 20),
        ..PipelineConfig::new(args.max_batch_size, args.prefill_chunk_size)
    };
    pipeline.prefix = PrefixConfig {
        spacing: args.prefix_spacing,
        entries: args.prefix_snapshots.unwrap_or(args.max_batch_size),
    };
    let scheduler = &mut pipeline.scheduler;
    scheduler.max_batch_tokens = args.max_batch_tokens.unwrap_or(scheduler.max_batch_tokens);
    scheduler.max_waiting = args.max_waiting;
    scheduler.target_pass = Duration::from_millis(args.target_pass_ms);
    let device = tur::shared::inference_device(args.cpu_reference)?;
    // Load before binding the listener so readiness means a validated, usable checkpoint.
    let worker = PipelineWorker::spawn(WorkerConfig {
        budget: DeviceBudget::for_device(&device),
        factory: ModelFactory {
            source,
            revision: args.revision,
            format: args.weight_format,
            device,
            kv: KvLayout::new(args.kv_page_tokens, args.kv_format),
            recurrent: args
                .recurrent_state
                .as_deref()
                .map_or(Ok(DEFAULT_RECURRENT_STATE), parse_recurrent_state)
                .map_err(TurError::Other)?,
        },
        pipeline,
        request_timeout: args.request_timeout_secs.map(Duration::from_secs),
        stream: StreamConfig {
            channel_capacity: args.stream_channel_capacity,
            max_buffered_bytes: args.stream_buffer_kb << 10,
        },
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
