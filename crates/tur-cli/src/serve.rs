use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{net::TcpListener, sync::mpsc};

use tur::{
    Result, TurError,
    backend::{memory::DeviceBudget, pipeline::PipelineConfig},
    server::{
        AppState, ServerStatus, forward_signals, serve_until,
        transcription::{TranscriptionConfig, TranscriptionWorker},
        worker::{PipelineWorker, StreamConfig, WorkerConfig},
    },
};

use crate::{
    args::{ModelArgs, PipelineArgs, WhisperArgs},
    config::inherit,
    monitor,
    ui::Console,
};

#[derive(Debug, Default, serde::Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Args {
    #[command(flatten)]
    #[serde(skip)]
    model: ModelArgs,
    #[command(flatten)]
    #[serde(skip)]
    pipeline: PipelineArgs,
    #[command(flatten)]
    #[serde(skip)]
    whisper: WhisperArgs,
    /// Model ID accepted by /v1/chat/completions and listed by /v1/models (default: qwen-hybrid).
    #[arg(long)]
    served_model_name: Option<String>,
    /// Model ID of `--whisper` at /v1/audio/transcriptions and /v1/models (default:
    /// whisper-1, the name `OpenAI` clients send).
    #[arg(long)]
    transcription_model_name: Option<String>,
    /// Clips queued for transcription before new ones are rejected (default: 16).
    #[arg(long)]
    max_waiting_transcriptions: Option<usize>,
    /// Address the HTTP listener binds once the model has loaded (default: 127.0.0.1:8080).
    #[arg(long)]
    listen: Option<SocketAddr>,
    /// Requests holding model state at once (default: 8).
    #[arg(long)]
    max_batch_size: Option<usize>,
    /// Tokens in one packed pass. Default: 2048, or more to fit one prefill chunk beside
    /// every decode.
    #[arg(long)]
    max_batch_tokens: Option<usize>,
    /// Requests queued for admission before new ones are rejected (default: 256).
    #[arg(long)]
    max_waiting: Option<usize>,
    /// Fail requests not finished this many seconds after arrival.
    #[arg(long)]
    request_timeout_secs: Option<u64>,
    /// Prefix snapshots retained beside running requests; each holds one recurrent
    /// state. Default: one per --max-batch-size, scaled down for large states; 0 disables
    /// prefix reuse.
    #[arg(long)]
    prefix_snapshots: Option<usize>,
    /// Seconds accepted requests may take to finish after SIGINT or SIGTERM. A second signal
    /// ends the wait (default: 30).
    #[arg(long)]
    shutdown_timeout_secs: Option<u64>,
    /// Show the live dashboard in this terminal once serving; logs move into it (default: true
    /// in a terminal). `q` closes it and the server keeps running.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    dashboard: Option<bool>,
    #[arg(skip)]
    stream_channel_capacity: Option<usize>,
    #[arg(skip)]
    stream_max_buffered_bytes: Option<usize>,
}

impl Args {
    inherit!(
        served_model_name,
        transcription_model_name,
        max_waiting_transcriptions,
        listen,
        max_batch_size,
        max_batch_tokens,
        max_waiting,
        request_timeout_secs,
        prefix_snapshots,
        shutdown_timeout_secs,
        dashboard,
        stream_channel_capacity,
        stream_max_buffered_bytes
    );

    pub(crate) fn configure(&mut self, settings: crate::config::FileSettings) {
        self.model.inherit(settings.model);
        self.pipeline.inherit(settings.engine);
        self.whisper.inherit(settings.whisper);
        self.inherit(settings.server);
    }

    /// Where the listener binds.
    pub(crate) fn listen_address(&self) -> SocketAddr {
        self.listen
            .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 8080)))
    }

    fn pipeline_config(&self) -> Result<PipelineConfig> {
        let mut pipeline = self.pipeline.config(self.max_batch_size.unwrap_or(8))?;
        if let Some(entries) = self.prefix_snapshots {
            pipeline.prefix.entries = entries;
            pipeline.prefix.state_share = None;
        }
        let scheduler = &mut pipeline.scheduler;
        scheduler.max_batch_tokens = self.max_batch_tokens.unwrap_or(scheduler.max_batch_tokens);
        scheduler.max_waiting = self.max_waiting.unwrap_or(scheduler.max_waiting);
        scheduler.validate()?;
        Ok(pipeline)
    }

    fn stream_config(&self) -> Result<StreamConfig> {
        let defaults = StreamConfig::default();
        let stream = StreamConfig {
            channel_capacity: self
                .stream_channel_capacity
                .unwrap_or(defaults.channel_capacity),
            max_buffered_bytes: self
                .stream_max_buffered_bytes
                .unwrap_or(defaults.max_buffered_bytes),
        };
        if stream.channel_capacity == 0
            || stream.channel_capacity > tokio::sync::Semaphore::MAX_PERMITS
            || stream.max_buffered_bytes == 0
        {
            return Err(TurError::Other(
                "server streaming limits must be positive and channel capacity must fit Tokio semaphore limits".into(),
            ));
        }
        Ok(stream)
    }
}

pub(crate) async fn serve(mut args: Args, console: Console) -> Result<()> {
    if args.model.caches_experts() {
        args.pipeline.prefer_large_passes();
    }
    let address = args.listen_address();
    let stream = args.stream_config()?;
    let pipeline = args.pipeline_config()?;
    let default_model = args
        .served_model_name
        .unwrap_or_else(|| "qwen-hybrid".into());
    if default_model.trim().is_empty() {
        return Err(TurError::Other(
            "served model name must not be empty".into(),
        ));
    }
    let factory = args.model.factory()?;
    pipeline.prefix.validate(factory.kv.page_tokens)?;
    let budget = DeviceBudget::for_device(&factory.device);
    // Whisper leases first, so the text state pool is sized from what remains.
    let mut transcribers = HashMap::new();
    let transcription_name = args
        .transcription_model_name
        .unwrap_or_else(|| "whisper-1".into());
    // A device of its own has its own queue, so text passes interleave with a transcription.
    let whisper_device = tur::shared::inference_device(factory.device.is_cpu())?;
    if let Some(whisper) = args.whisper.factory(&whisper_device)? {
        let worker = TranscriptionWorker::spawn(TranscriptionConfig {
            factory: whisper,
            budget: budget.clone(),
            max_waiting: args.max_waiting_transcriptions.unwrap_or(16),
        })?;
        transcribers.insert(transcription_name.clone(), worker);
    }
    let default_transcriber = (!transcribers.is_empty()).then_some(transcription_name);
    // Load before binding the listener so readiness means a validated, usable checkpoint.
    let worker = PipelineWorker::spawn(WorkerConfig {
        budget,
        factory,
        pipeline,
        request_timeout: args.request_timeout_secs.map(Duration::from_secs),
        stream,
    })?;
    let workers = HashMap::from([(default_model.clone(), worker)]);
    let watch = Watch {
        console,
        address,
        dashboard: args.dashboard.unwrap_or(true),
        drain_seconds: args.shutdown_timeout_secs.unwrap_or(30),
    };
    serve_watched(
        AppState {
            workers,
            transcribers,
            default_transcriber,
            default_model,
            status: ServerStatus::shared(),
        },
        watch,
    )
    .await
}

/// How a running server is presented and stopped.
struct Watch {
    console: Console,
    address: SocketAddr,
    dashboard: bool,
    drain_seconds: u64,
}

/// Open the listener and serve, with the display on a thread of its own: it blocks on the
/// terminal, and its failures must never reach the server.
async fn serve_watched(state: AppState, watch: Watch) -> Result<()> {
    let address = watch.address;
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| TurError::Other(format!("cannot bind {address}: {error}")))?;
    tracing::info!("Listening on http://{address}");
    // Beside the signals, a dashboard holds raw-mode Ctrl-C and asks for a drain through this.
    let (drain, signals) = mpsc::channel(2);
    forward_signals(drain.clone())?;
    let stop = Arc::new(AtomicBool::new(false));
    let attachment = monitor::Attachment {
        console: watch.console,
        server: monitor::Server::listening_on(address),
        dashboard: watch.dashboard,
        drain_seconds: watch.drain_seconds,
        drain,
        stop: Arc::clone(&stop),
    };
    let display = tokio::task::spawn_blocking(move || monitor::attach(&attachment));
    let served = serve_until(
        state,
        listener,
        signals,
        Duration::from_secs(watch.drain_seconds),
    )
    .await;
    stop.store(true, Ordering::Relaxed);
    if let Err(error) = display.await {
        tracing::warn!("server display failed: {error}");
    }
    served
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible server configuration"
)]
mod tests {
    use super::*;
    use crate::{Command, config::FileSettings, parse_cli};

    fn arguments(values: &[&str]) -> Result<Args> {
        match parse_cli(values)?.command {
            Some(Command::Serve(args)) => Ok(args),
            _ => Err(TurError::Other("expected serve arguments".into())),
        }
    }

    #[test]
    fn test_server_file_limits_and_cli_overrides_reach_pipeline() -> Result<()> {
        let settings: FileSettings = toml::from_str(
            r#"
            [engine]
            prefill_chunk_size = 512
            target_pass_ms = 80
            max_queue_wait_ms = 350
            min_prefill_tokens = 16
            max_waiting_tokens = 10000
            prefix_spacing = 512
            device_sampling = false
            speculative_draft = 4
            mtp = true
            [server]
            listen = "127.0.0.1:9000"
            max_batch_size = 16
            max_batch_tokens = 4096
            max_waiting = 30
            prefix_snapshots = 20
            request_timeout_secs = 60
            stream_channel_capacity = 10
            stream_max_buffered_bytes = 2048
        "#,
        )
        .map_err(|error| TurError::Other(error.to_string()))?;
        let mut args = arguments(&[
            "tur",
            "serve",
            "--max-batch-size",
            "2",
            "--prefix-snapshots",
            "0",
            "--speculative-draft",
            "0",
            "--mtp",
            "false",
        ])?;
        args.configure(settings);
        let pipeline = args.pipeline_config()?;
        let scheduler = pipeline.scheduler;
        assert_eq!(scheduler.max_running, 2);
        assert_eq!(scheduler.max_chunk, 512);
        assert_eq!(scheduler.max_batch_tokens, 4096);
        assert_eq!(scheduler.max_waiting, 30);
        assert_eq!(scheduler.max_waiting_tokens, 10000);
        assert_eq!(scheduler.min_prefill_tokens, 16);
        assert_eq!(scheduler.target_pass, Duration::from_millis(80));
        assert_eq!(scheduler.max_queue_wait, Duration::from_millis(350));
        assert!(!pipeline.device_sampling);
        assert!(!pipeline.speculation.enabled());
        assert_eq!(pipeline.prefix.entries, 0);
        assert_eq!(pipeline.prefix.spacing, 512);
        assert_eq!(
            args.stream_config()?,
            StreamConfig {
                channel_capacity: 10,
                max_buffered_bytes: 2048
            }
        );
        assert_eq!(args.listen, Some(SocketAddr::from(([127, 0, 0, 1], 9000))));
        assert_eq!(args.request_timeout_secs, Some(60));
        Ok(())
    }

    #[test]
    fn test_server_derived_defaults_use_resolved_inputs() -> Result<()> {
        let mut args = arguments(&["tur", "serve"])?;
        args.configure(
            toml::from_str("[engine]\nprefill_chunk_size = 4096\n[server]\nmax_batch_size = 3")
                .map_err(|error| TurError::Other(error.to_string()))?,
        );
        let pipeline = args.pipeline_config()?;
        assert_eq!(pipeline.scheduler.max_batch_tokens, 4099);
        assert_eq!(pipeline.prefix.entries, 3);
        assert_eq!(pipeline.prefix.state_share, Some(0.125));
        assert_eq!(args.stream_config()?, StreamConfig::default());
        Ok(())
    }

    #[test]
    fn test_invalid_server_limits_fail_before_loading() -> Result<()> {
        for text in [
            "[engine]\ntarget_pass_ms = 0",
            "[engine]\nmin_prefill_tokens = 257",
            "[server]\nmax_batch_size = 0",
            "[server]\nmax_batch_tokens = 1",
            "[server]\nstream_channel_capacity = 0",
            "[server]\nstream_max_buffered_bytes = 0",
        ] {
            let mut args = arguments(&["tur", "serve"])?;
            args.configure(
                toml::from_str(text).map_err(|error| TurError::Other(error.to_string()))?,
            );
            assert!(
                args.pipeline_config().is_err() || args.stream_config().is_err(),
                "accepted {text}"
            );
        }
        Ok(())
    }
}
