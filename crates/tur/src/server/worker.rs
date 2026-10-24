#![forbid(unsafe_code)]
//! A dedicated execution owner between async HTTP handlers and hybrid inference.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::{
    ModelFactory, Result, TurError,
    backend::{
        memory::DeviceBudget,
        pipeline::{
            BatchStepOutput, GenerationRequest, GenerationStats, InferencePipeline, PipelineConfig,
        },
    },
};

/// A validated text conversation and its independent response channels. Only streaming
/// requests carry a text channel; it is bounded (see [`StreamConfig`]).
pub struct WorkerMsg {
    pub request: GenerationRequest,
    pub token_tx: Option<mpsc::Sender<String>>,
    pub result_tx: oneshot::Sender<Result<GenerationStats>>,
}

/// Bounded delivery of streamed text. Each pass sends at most one coalesced message per
/// request; text the client has not taken waits in a per-request buffer. A buffer beyond
/// `max_buffered_bytes` means the client stalled: the request is cancelled with
/// [`TurError::SlowClient`] and its state released, so a stalled client holds at most
/// `channel_capacity + 1` messages of at most that size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamConfig {
    pub channel_capacity: usize,
    pub max_buffered_bytes: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 64,
            max_buffered_bytes: 16 << 10,
        }
    }
}

/// Checkpoint, scheduling limits and state budget for one worker. Model operations stay on
/// its OS thread.
pub struct WorkerConfig {
    pub factory: ModelFactory,
    pub pipeline: PipelineConfig,
    /// Device memory shared with every other worker on the same device.
    pub budget: DeviceBudget,
    /// Deadline for requests that set none.
    pub request_timeout: Option<Duration>,
    pub stream: StreamConfig,
}

/// Why streamed text could not be handed over.
enum Undelivered {
    Disconnected,
    Stalled(usize),
}

/// A streaming client's channel and the text it has not taken yet.
struct Stream {
    tx: mpsc::Sender<String>,
    buffer: String,
}

impl Stream {
    /// Send everything buffered as one message, keeping it if the channel is full.
    fn flush(&mut self, limit: usize) -> std::result::Result<(), Undelivered> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        match self.tx.try_send(std::mem::take(&mut self.buffer)) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(text)) => {
                self.buffer = text;
                if self.buffer.len() > limit {
                    Err(Undelivered::Stalled(self.buffer.len()))
                } else {
                    Ok(())
                }
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(Undelivered::Disconnected),
        }
    }
}

struct PendingRequest {
    stream: Option<Stream>,
    result_tx: oneshot::Sender<Result<GenerationStats>>,
}

impl PendingRequest {
    fn disconnected(&self) -> bool {
        self.result_tx.is_closed() || self.stream.as_ref().is_some_and(|s| s.tx.is_closed())
    }
}

/// Handle to a single-owner packed hybrid worker with a bounded intake queue.
#[derive(Clone)]
pub struct PipelineWorker {
    tx: mpsc::Sender<WorkerMsg>,
    timeout: Option<Duration>,
    stream: StreamConfig,
}

const MAX_ADMISSIONS_PER_PASS: usize = 32;

fn drain_intake<T>(rx: &mut mpsc::Receiver<T>, mut admit: impl FnMut(T)) {
    // A producer can refill a bounded channel while it is drained. Always return to
    // execution, cancellation and expiration after a bounded amount of intake work.
    let started = Instant::now();
    for _ in 0..MAX_ADMISSIONS_PER_PASS {
        let Ok(msg) = rx.try_recv() else { break };
        admit(msg);
        if started.elapsed() >= Duration::from_millis(2) {
            break;
        }
    }
}

fn admit(
    msg: WorkerMsg,
    pipeline: &mut InferencePipeline,
    pending: &mut HashMap<Uuid, PendingRequest>,
) {
    if msg.result_tx.is_closed() || msg.token_tx.as_ref().is_some_and(mpsc::Sender::is_closed) {
        return;
    }
    match pipeline.submit_request(&msg.request) {
        Ok(id) => {
            pending.insert(
                id,
                PendingRequest {
                    stream: msg.token_tx.map(|tx| Stream {
                        tx,
                        buffer: String::new(),
                    }),
                    result_tx: msg.result_tx,
                },
            );
        }
        Err(error) => {
            let _ = msg.result_tx.send(Err(error));
        }
    }
}

/// Release every request whose client went away.
fn release_disconnected(
    pipeline: &mut InferencePipeline,
    pending: &mut HashMap<Uuid, PendingRequest>,
) -> Result<()> {
    let gone: Vec<Uuid> = pending
        .iter()
        .filter(|(_, request)| request.disconnected())
        .map(|(id, _)| *id)
        .collect();
    for id in gone {
        pipeline.cancel(id)?;
        pending.remove(&id);
    }
    Ok(())
}

/// Hand one pass's text and completions to their clients: one coalesced message per
/// streaming request, never waiting on a client.
fn deliver(
    step: BatchStepOutput,
    pipeline: &mut InferencePipeline,
    pending: &mut HashMap<Uuid, PendingRequest>,
    config: StreamConfig,
) -> Result<()> {
    for (id, text) in step.new_tokens {
        if let Some(stream) = pending.get_mut(&id).and_then(|r| r.stream.as_mut()) {
            stream.buffer.push_str(&text);
        }
    }
    for (id, result) in step.completed {
        if let Some(mut request) = pending.remove(&id) {
            // A remainder the channel cannot take is recovered by the adapter from the
            // final text, so completion never waits on the client.
            if let Some(stream) = &mut request.stream {
                let _ = stream.flush(usize::MAX);
            }
            drop(request.stream);
            let _ = request.result_tx.send(result);
        }
    }
    let mut cancelled = Vec::new();
    for (id, request) in pending.iter_mut() {
        if let Some(stream) = &mut request.stream {
            match stream.flush(config.max_buffered_bytes) {
                Ok(()) => {}
                Err(Undelivered::Disconnected) => cancelled.push((*id, None)),
                Err(Undelivered::Stalled(bytes)) => cancelled.push((*id, Some(bytes))),
            }
        }
    }
    for (id, stalled) in cancelled {
        pipeline.cancel(id)?;
        if let (Some(request), Some(bytes)) = (pending.remove(&id), stalled) {
            tracing::warn!(request = %id, bytes, "cancelled a stalled stream");
            let _ = request.result_tx.send(Err(TurError::SlowClient(format!(
                "{bytes} bytes undelivered beyond the {}-byte bound",
                config.max_buffered_bytes
            ))));
        }
    }
    Ok(())
}

fn serve(
    mut rx: mpsc::Receiver<WorkerMsg>,
    mut pipeline: InferencePipeline,
    config: StreamConfig,
) -> Result<()> {
    let mut pending: HashMap<Uuid, PendingRequest> = HashMap::new();
    loop {
        if pending.is_empty() {
            let Some(msg) = rx.blocking_recv() else {
                return Ok(());
            };
            admit(msg, &mut pipeline, &mut pending);
            drain_intake(&mut rx, |msg| admit(msg, &mut pipeline, &mut pending));
        }
        // Intake (rendering and tokenization) and disconnects run while the device
        // executes the submitted pass; the next pass needs its sampled tokens.
        let step = pipeline.begin_step().and_then(|_| {
            drain_intake(&mut rx, |msg| admit(msg, &mut pipeline, &mut pending));
            release_disconnected(&mut pipeline, &mut pending)?;
            pipeline.finish_step()
        });
        let step = match step {
            Ok(step) => step,
            Err(error) => {
                for (_, request) in pending.drain() {
                    let _ = request
                        .result_tx
                        .send(Err(TurError::Other(error.to_string())));
                }
                return Err(error);
            }
        };
        deliver(step, &mut pipeline, &mut pending, config)?;
    }
}

impl PipelineWorker {
    /// Load and validate the checkpoint before reporting readiness. Startup errors propagate.
    pub fn spawn(config: WorkerConfig) -> Result<Self> {
        let timeout = config.request_timeout;
        let stream = config.stream;
        let (tx, rx) = mpsc::channel(256);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("hybrid-worker".into())
            .spawn(move || {
                match InferencePipeline::with_budget(
                    &config.factory,
                    config.pipeline,
                    &config.budget,
                ) {
                    Ok(pipeline) => {
                        let plan = pipeline.memory_plan();
                        tracing::info!(
                            resident_bytes = plan.resident(),
                            weights = plan.weights,
                            state_pool = plan.state_pool,
                            scratch_peak = plan.scratch_peak,
                            limit = ?plan.limit,
                            "Memory plan"
                        );
                        if ready_tx.send(Ok(())).is_err() {
                            return;
                        }
                        if let Err(error) = serve(rx, pipeline, stream) {
                            tracing::warn!("Hybrid worker stopped: {error}");
                        }
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })?;
        ready_rx
            .recv()
            .map_err(|error| TurError::Other(format!("Worker startup failed: {error}")))??;
        Ok(Self {
            tx,
            timeout,
            stream,
        })
    }

    /// Wrap a caller-provided sender, allowing HTTP tests to provide deterministic workers.
    pub fn from_sender(tx: mpsc::Sender<WorkerMsg>) -> Self {
        Self {
            tx,
            timeout: None,
            stream: StreamConfig::default(),
        }
    }

    /// Bounded text channel for one streaming request.
    pub fn stream_channel(&self) -> (mpsc::Sender<String>, mpsc::Receiver<String>) {
        mpsc::channel(self.stream.channel_capacity.max(1))
    }

    /// Enqueue with the original arrival time. A full intake queue fails immediately;
    /// senders never form another unbounded queue outside the worker.
    pub fn send(&self, mut msg: WorkerMsg) -> impl Future<Output = Result<()>> {
        msg.request.submitted_at.get_or_insert_with(Instant::now);
        msg.request.timeout = msg.request.timeout.or(self.timeout);
        std::future::ready(self.tx.try_send(msg).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => {
                TurError::QueueFull("worker intake is full".into())
            }
            mpsc::error::TrySendError::Closed(_) => TurError::WorkerUnavailable,
        }))
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use crate::{
        ModelSource,
        models::qwen3_5::{DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat},
    };
    use candle_core::Device;

    #[test]
    fn test_refilled_intake_returns_to_execution_within_its_limit() -> Result<()> {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(())
            .map_err(|e| TurError::Other(e.to_string()))?;
        let mut admitted = 0;
        let mut refill_error = None;
        drain_intake(&mut rx, |()| {
            admitted += 1;
            refill_error = tx.try_send(()).err();
        });
        assert!(refill_error.is_none());
        assert!((1..=MAX_ADMISSIONS_PER_PASS).contains(&admitted));
        assert_eq!(rx.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_intake_rejects_overload_and_records_arrival_before_queueing() -> Result<()> {
        let (tx, mut rx) = mpsc::channel(1);
        let worker = PipelineWorker {
            tx,
            timeout: Some(Duration::from_secs(1)),
            stream: StreamConfig::default(),
        };
        let message = || {
            let (token_tx, token_rx) = mpsc::channel(1);
            let (result_tx, result_rx) = oneshot::channel();
            (
                WorkerMsg {
                    request: GenerationRequest::new("t10".into(), 1),
                    token_tx: Some(token_tx),
                    result_tx,
                },
                token_rx,
                result_rx,
            )
        };
        let before = Instant::now();
        let (msg, _tokens, _result) = message();
        worker.send(msg).await?;
        let queued_at = Instant::now();
        let (msg, _tokens2, _result2) = message();
        assert!(matches!(
            worker.send(msg).await,
            Err(TurError::QueueFull(_))
        ));
        let received = rx.recv().await.ok_or(TurError::WorkerUnavailable)?;
        assert!(
            received
                .request
                .submitted_at
                .is_some_and(|at| at >= before && at <= queued_at)
        );
        assert_eq!(received.request.timeout, Some(Duration::from_secs(1)));
        drop(rx);
        let (msg, _tokens3, _result3) = message();
        assert!(matches!(
            worker.send(msg).await,
            Err(TurError::WorkerUnavailable)
        ));
        Ok(())
    }

    #[test]
    fn test_startup_failure_is_returned_before_worker_is_ready() {
        let result = PipelineWorker::spawn(WorkerConfig {
            factory: ModelFactory {
                source: ModelSource::LocalPath("/missing/tur-checkpoint".into()),
                revision: "fixture".into(),
                format: Some(WeightFormat::F32),
                device: Device::Cpu,
                kv: DEFAULT_KV_LAYOUT,
                recurrent: DEFAULT_RECURRENT_STATE,
                mtp: false,
                draft_vocab: None,
            },
            pipeline: PipelineConfig::new(1, 64),
            budget: DeviceBudget::new(None),
            request_timeout: None,
            stream: StreamConfig::default(),
        });
        assert!(result.is_err());
    }
}
