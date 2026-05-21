use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};
use tracing::debug;

use crate::{
    AutoModelFactory, Result, TurError,
    backend::{
        chat_template::Message,
        pipeline::{GenerationRequest, GenerationStats, InferencePipeline},
        tools::ToolDefinition,
    },
};

/// A single generation job sent from the async HTTP layer to the worker thread.
pub struct WorkerMsg {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: usize,
    pub thinking: bool,
    /// Token stream — each generated token is sent here.  The channel is
    /// closed when generation completes so the consumer detects end-of-stream.
    pub token_tx: mpsc::UnboundedSender<String>,
    pub result_tx: oneshot::Sender<Result<GenerationStats>>,
}

/// Configuration used to build the pipeline inside the worker thread.
pub struct WorkerConfig {
    pub factory: Arc<AutoModelFactory>,
    pub seed: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

/// Handle to the dedicated pipeline worker thread.  Cheap to clone.
#[derive(Clone)]
pub struct PipelineWorker {
    tx: mpsc::Sender<WorkerMsg>,
}

impl PipelineWorker {
    /// Spawn the worker thread and build the pipeline inside it.
    ///
    /// Building on the dedicated thread ensures Metal/CUDA device state never
    /// crosses thread boundaries.  `format_messages` and `run` both execute on
    /// the same thread that owns the pipeline.
    pub fn spawn(config: WorkerConfig) -> Result<Self> {
        let (tx, mut rx) = mpsc::channel::<WorkerMsg>(4);

        let token_sink: Arc<Mutex<Option<mpsc::UnboundedSender<String>>>> =
            Arc::new(Mutex::new(None));
        let sink_for_callback = Arc::clone(&token_sink);

        std::thread::Builder::new()
            .name("pipeline-worker".to_string())
            .spawn(move || {
                let WorkerConfig {
                    factory,
                    seed,
                    temperature,
                    top_p,
                    repeat_penalty,
                    repeat_last_n,
                } = config;

                let device = factory.device().clone();

                let mut builder = InferencePipeline::builder(&*factory, device)
                    .seed(seed)
                    .repeat_penalty(repeat_penalty)
                    .repeat_last_n(repeat_last_n)
                    .on_token(move |t| {
                        if let Some(ref tx) = *sink_for_callback.lock() {
                            let _ = tx.send(t.to_string());
                        }
                    });

                if let Some(temp) = temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(top_p) = top_p {
                    builder = builder.top_p(top_p);
                }

                let mut pipeline = builder.build();
                debug!("Pipeline worker ready");

                while let Some(msg) = rx.blocking_recv() {
                    let tools_opt = if msg.tools.is_empty() {
                        None
                    } else {
                        Some(msg.tools.as_slice())
                    };

                    let prompt = pipeline.format_messages(&msg.messages, tools_opt, msg.thinking);

                    let mut req = GenerationRequest::new(prompt, msg.max_tokens).with_raw(true);
                    if !msg.tools.is_empty() {
                        req = req.with_tools(msg.tools);
                    }
                    if msg.thinking {
                        req = req.with_thinking(true);
                    }

                    *token_sink.lock() = Some(msg.token_tx);
                    let result = pipeline.run(&req);
                    *token_sink.lock() = None;
                    let _ = msg.result_tx.send(result);
                }

                debug!("Pipeline worker shut down");
            })
            .map_err(|e| TurError::Other(format!("Failed to spawn pipeline worker: {e}")))?;

        Ok(Self { tx })
    }

    /// Build a `PipelineWorker` from an already-created channel sender.
    ///
    /// Useful in tests: the caller spawns a Tokio task that receives
    /// `WorkerMsg` values and responds with predetermined output, then wraps
    /// its `Sender` here to obtain a usable `PipelineWorker`.
    pub fn from_sender(tx: mpsc::Sender<WorkerMsg>) -> Self {
        Self { tx }
    }

    pub async fn send(&self, msg: WorkerMsg) -> Result<()> {
        self.tx
            .send(msg)
            .await
            .map_err(|_| TurError::Other("Pipeline worker unavailable".to_string()))
    }
}
