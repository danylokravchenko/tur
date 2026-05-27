use std::{collections::HashMap, sync::Arc};

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::{
    AutoModelFactory, Result, TurError,
    backend::{
        chat_template::Message,
        pipeline::{GenerationRequest, GenerationStats, InferencePipeline},
        tools::{ToolCall, ToolDefinition},
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
    /// Maximum number of requests to process simultaneously in the decode step.
    pub max_batch_size: usize,
}

/// Per-request state kept in the worker while a request is in-flight.
struct PendingRequest {
    token_tx: mpsc::UnboundedSender<String>,
    result_tx: oneshot::Sender<Result<GenerationStats>>,
    /// Whether the original request included tool definitions.
    /// When true the worker parses tool calls from the generated text at
    /// completion time.
    has_tools: bool,
}

/// Handle to the dedicated pipeline worker thread.  Cheap to clone.
#[derive(Clone)]
pub struct PipelineWorker {
    tx: mpsc::Sender<WorkerMsg>,
}

impl PipelineWorker {
    /// Spawn the worker thread and build the pipeline inside it.
    ///
    /// The pipeline is built with continuous batching enabled so multiple
    /// in-flight requests share every GPU forward pass.  Metal/CUDA device
    /// state never crosses thread boundaries — all model operations happen on
    /// the single dedicated OS thread.
    pub fn spawn(config: WorkerConfig) -> Result<Self> {
        // Larger channel so many concurrent HTTP handlers can enqueue without
        // back-pressure.  The scheduler's own memory-based admission control is
        // the real throttle.
        let (tx, rx) = mpsc::channel::<WorkerMsg>(256);

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
                    max_batch_size,
                } = config;

                let device = factory.device().clone();

                let mut builder = InferencePipeline::builder(&*factory, device)
                    .seed(seed)
                    .repeat_penalty(repeat_penalty)
                    .repeat_last_n(repeat_last_n)
                    .enable_batching(true)
                    .max_batch_size(max_batch_size)
                    .max_prefill_batch(max_batch_size)
                    .max_decode_batch(max_batch_size);

                if let Some(temp) = temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(tp) = top_p {
                    builder = builder.top_p(tp);
                }

                let mut pipeline = builder.build();
                debug!("Pipeline worker ready (continuous batching enabled, max_batch={max_batch_size})");

                // Per-request routing table: uuid → (token_tx, result_tx, has_tools)
                let mut pending: HashMap<Uuid, PendingRequest> = HashMap::new();

                // We need a blocking handle to the tokio receiver.  The channel
                // is created outside (in async context) but we receive on it here
                // from a plain OS thread using `blocking_recv`.
                let mut rx = rx;

                loop {
                    // ── 1. Intake new requests ──────────────────────────────
                    if pending.is_empty() {
                        // Nothing in flight: block until the next message
                        // arrives so we don't busy-spin.
                        let msg = match rx.blocking_recv() {
                            Some(m) => m,
                            None => {
                                debug!("Pipeline worker channel closed, shutting down");
                                return;
                            }
                        };
                        Self::admit_msg(msg, &mut pipeline, &mut pending);
                    }

                    // Drain any additional messages that arrived without blocking.
                    loop {
                        match rx.try_recv() {
                            Ok(msg) => Self::admit_msg(msg, &mut pipeline, &mut pending),
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                                debug!("Pipeline worker channel disconnected");
                                // Drain remaining pending requests with an error.
                                for (_, pr) in pending.drain() {
                                    let _ = pr.result_tx.send(Err(TurError::Other(
                                        "Worker shutting down".to_string(),
                                    )));
                                }
                                return;
                            }
                        }
                    }

                    // ── 2. Run one scheduling iteration ────────────────────
                    match pipeline.step_streaming() {
                        Ok(output) => {
                            // Route streaming tokens to their HTTP handlers.
                            for (id, text) in output.new_tokens {
                                if let Some(pr) = pending.get(&id) {
                                    let _ = pr.token_tx.send(text);
                                }
                            }

                            // Complete finished requests.
                            for (id, mut result) in output.completed {
                                if let Some(pr) = pending.remove(&id) {
                                    // Parse tool calls from the full generated
                                    // text when the request included tools.
                                    if pr.has_tools {
                                        result.stats.tool_calls =
                                            ToolCall::parse_from_output(&result.generated_text);
                                    }
                                    // Dropping token_tx closes the stream so the
                                    // HTTP handler detects end-of-tokens.
                                    drop(pr.token_tx);
                                    let _ = pr.result_tx.send(Ok(result.stats));
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Scheduler step failed: {e}");
                            // Propagate the error to every waiting request and
                            // reset — the pipeline may be in an undefined state.
                            for (_, pr) in pending.drain() {
                                drop(pr.token_tx);
                                let _ = pr.result_tx.send(Err(TurError::Other(e.to_string())));
                            }
                            return;
                        }
                    }
                }
            })
            .map_err(|e| TurError::Other(format!("Failed to spawn pipeline worker: {e}")))?;

        Ok(Self { tx })
    }

    /// Format the incoming message and enqueue it with the scheduler.
    fn admit_msg(
        msg: WorkerMsg,
        pipeline: &mut InferencePipeline<crate::backend::AnyModel>,
        pending: &mut HashMap<Uuid, PendingRequest>,
    ) {
        let tools_opt = if msg.tools.is_empty() {
            None
        } else {
            Some(msg.tools.as_slice())
        };
        let prompt = pipeline.format_messages(&msg.messages, tools_opt, msg.thinking);

        let mut req = GenerationRequest::new(prompt, msg.max_tokens).with_raw(true);
        if !msg.tools.is_empty() {
            req = req.with_tools(msg.tools.clone());
        }
        if msg.thinking {
            req = req.with_thinking(true);
        }

        let has_tools = !msg.tools.is_empty();
        match pipeline.submit_request(&req) {
            Ok(handle) => {
                pending.insert(
                    handle.id,
                    PendingRequest {
                        token_tx: msg.token_tx,
                        result_tx: msg.result_tx,
                        has_tools,
                    },
                );
            }
            Err(e) => {
                drop(msg.token_tx);
                let _ = msg.result_tx.send(Err(e));
            }
        }
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
