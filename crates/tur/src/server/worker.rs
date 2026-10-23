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
        pipeline::{GenerationRequest, GenerationStats, InferencePipeline, PipelineConfig},
    },
};

/// A validated text conversation and its independent response channels.
pub struct WorkerMsg {
    pub request: GenerationRequest,
    pub token_tx: mpsc::UnboundedSender<String>,
    pub result_tx: oneshot::Sender<Result<GenerationStats>>,
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
}

struct PendingRequest {
    token_tx: mpsc::UnboundedSender<String>,
    result_tx: oneshot::Sender<Result<GenerationStats>>,
}

/// Handle to a single-owner packed hybrid worker with a bounded intake queue.
#[derive(Clone)]
pub struct PipelineWorker {
    tx: mpsc::Sender<WorkerMsg>,
    timeout: Option<Duration>,
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
    if msg.result_tx.is_closed() || msg.token_tx.is_closed() {
        return;
    }
    match pipeline.submit_request(&msg.request) {
        Ok(id) => {
            pending.insert(
                id,
                PendingRequest {
                    token_tx: msg.token_tx,
                    result_tx: msg.result_tx,
                },
            );
        }
        Err(error) => {
            let _ = msg.result_tx.send(Err(error));
        }
    }
}

fn serve(mut rx: mpsc::Receiver<WorkerMsg>, mut pipeline: InferencePipeline) -> Result<()> {
    let mut pending: HashMap<Uuid, PendingRequest> = HashMap::new();
    loop {
        if pending.is_empty() {
            let Some(msg) = rx.blocking_recv() else {
                return Ok(());
            };
            admit(msg, &mut pipeline, &mut pending);
        }
        drain_intake(&mut rx, |msg| admit(msg, &mut pipeline, &mut pending));
        let cancelled: Vec<Uuid> = pending
            .iter()
            .filter(|(_, request)| request.result_tx.is_closed() || request.token_tx.is_closed())
            .map(|(id, _)| *id)
            .collect();
        for id in cancelled {
            pipeline.cancel(id)?;
            pending.remove(&id);
        }
        let step = match pipeline.step_streaming() {
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
        for (id, text) in step.new_tokens {
            if let Some(request) = pending.get(&id) {
                let _ = request.token_tx.send(text);
            }
        }
        for (id, result) in step.completed {
            if let Some(request) = pending.remove(&id) {
                drop(request.token_tx);
                let _ = request.result_tx.send(result);
            }
        }
    }
}

impl PipelineWorker {
    /// Load and validate the checkpoint before reporting readiness. Startup errors propagate.
    pub fn spawn(config: WorkerConfig) -> Result<Self> {
        let timeout = config.request_timeout;
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
                        if let Err(error) = serve(rx, pipeline) {
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
        Ok(Self { tx, timeout })
    }

    /// Wrap a caller-provided sender, allowing HTTP tests to provide deterministic workers.
    pub fn from_sender(tx: mpsc::Sender<WorkerMsg>) -> Self {
        Self { tx, timeout: None }
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
        };
        let message = || {
            let (token_tx, token_rx) = mpsc::unbounded_channel();
            let (result_tx, result_rx) = oneshot::channel();
            (
                WorkerMsg {
                    request: GenerationRequest::new("t10".into(), 1),
                    token_tx,
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
                format: WeightFormat::F32,
                device: Device::Cpu,
                kv: DEFAULT_KV_LAYOUT,
                recurrent: DEFAULT_RECURRENT_STATE,
            },
            pipeline: PipelineConfig::new(1, 64),
            budget: DeviceBudget::new(None),
            request_timeout: None,
        });
        assert!(result.is_err());
    }
}
