use std::{collections::HashMap, net::SocketAddr, time::Duration};

use tur::{
    Result, TurError,
    backend::{memory::DeviceBudget, prefix::PrefixConfig},
    server::{
        AppState, run_server,
        worker::{PipelineWorker, StreamConfig, WorkerConfig},
    },
};

use crate::args::{ModelArgs, PipelineArgs};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    pipeline: PipelineArgs,
    /// Model ID accepted by /v1/chat/completions and listed by /v1/models.
    #[arg(long, default_value = "qwen-hybrid")]
    served_model_name: String,
    /// Address the HTTP listener binds once the model has loaded.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    /// Requests holding model state at once.
    #[arg(long, default_value_t = 8)]
    max_batch_size: usize,
    /// Tokens in one packed pass. Default: 2048, or more to fit one prefill chunk beside
    /// every decode.
    #[arg(long)]
    max_batch_tokens: Option<usize>,
    /// Requests queued for admission before new ones are rejected.
    #[arg(long, default_value_t = 256)]
    max_waiting: usize,
    /// Fail requests not finished this many seconds after arrival.
    #[arg(long)]
    request_timeout_secs: Option<u64>,
    /// Prefix snapshots retained beside running requests; each holds one recurrent
    /// state. Default: one per --max-batch-size, scaled down for large states; 0 disables
    /// prefix reuse.
    #[arg(long)]
    prefix_snapshots: Option<usize>,
}

pub async fn serve(args: Args) -> Result<()> {
    if args.served_model_name.trim().is_empty() {
        return Err(TurError::Other(
            "served model name must not be empty".into(),
        ));
    }
    let mut pipeline = args.pipeline.config(args.max_batch_size);
    if let Some(entries) = args.prefix_snapshots {
        pipeline.prefix = PrefixConfig {
            entries,
            state_share: None,
            ..pipeline.prefix
        };
    }
    let scheduler = &mut pipeline.scheduler;
    scheduler.max_batch_tokens = args.max_batch_tokens.unwrap_or(scheduler.max_batch_tokens);
    scheduler.max_waiting = args.max_waiting;
    let factory = args.model.factory()?;
    // Load before binding the listener so readiness means a validated, usable checkpoint.
    let worker = PipelineWorker::spawn(WorkerConfig {
        budget: DeviceBudget::for_device(&factory.device),
        factory,
        pipeline,
        request_timeout: args.request_timeout_secs.map(Duration::from_secs),
        stream: StreamConfig::default(),
    })?;
    let default_model = args.served_model_name;
    let workers = HashMap::from([(default_model.clone(), worker)]);
    run_server(
        AppState {
            workers,
            default_model,
        },
        args.listen,
    )
    .await
}
