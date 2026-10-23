#![forbid(unsafe_code)]
//! A dedicated execution owner between async HTTP handlers and hybrid inference.

use std::{collections::HashMap, time::Duration};
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
}

fn admit(
    mut msg: WorkerMsg,
    pipeline: &mut InferencePipeline,
    pending: &mut HashMap<Uuid, PendingRequest>,
    timeout: Option<Duration>,
) {
    if msg.result_tx.is_closed() || msg.token_tx.is_closed() {
        return;
    }
    msg.request.timeout = msg.request.timeout.or(timeout);
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

fn serve(
    mut rx: mpsc::Receiver<WorkerMsg>,
    mut pipeline: InferencePipeline,
    timeout: Option<Duration>,
) -> Result<()> {
    let mut pending: HashMap<Uuid, PendingRequest> = HashMap::new();
    loop {
        if pending.is_empty() {
            let Some(msg) = rx.blocking_recv() else {
                return Ok(());
            };
            admit(msg, &mut pipeline, &mut pending, timeout);
        }
        // Drain intake every iteration; the pipeline's waiting queue bounds what it accepts
        // and rejects the rest, so arrivals join the next pass instead of queueing here.
        while let Ok(msg) = rx.try_recv() {
            admit(msg, &mut pipeline, &mut pending, timeout);
        }
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
                        if let Err(error) = serve(rx, pipeline, config.request_timeout) {
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
        Ok(Self { tx })
    }

    /// Wrap a caller-provided sender, allowing HTTP tests to provide deterministic workers.
    pub fn from_sender(tx: mpsc::Sender<WorkerMsg>) -> Self {
        Self { tx }
    }

    /// Enqueue a conversation with backpressure when the intake queue is full.
    pub async fn send(&self, msg: WorkerMsg) -> Result<()> {
        self.tx
            .send(msg)
            .await
            .map_err(|_| TurError::Other("Hybrid worker unavailable".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ModelSource,
        models::qwen3_5::{DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat},
    };
    use candle_core::Device;

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
