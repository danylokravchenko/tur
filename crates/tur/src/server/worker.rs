#![forbid(unsafe_code)]
//! A dedicated execution owner between async HTTP handlers and hybrid inference.

use std::{
    collections::HashMap,
    sync::{Arc, mpsc as std_mpsc},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, watch};
use uuid::Uuid;

use crate::{
    ModelFactory, ModelSource, Result, TurError,
    audio::Audio,
    backend::{
        memory::DeviceBudget,
        pipeline::{
            BatchStepOutput, GenerationRequest, GenerationStats, InferencePipeline, PipelineConfig,
        },
    },
    models::TextModel,
    runtime::pool::allocated_bytes,
    server::telemetry::{ModelReport, PassReading, WorkerTelemetry},
};

/// A piece of generated text and who it is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextDelta {
    Content(String),
    Reasoning(String),
}

/// A validated text conversation and its independent response channels. Only streaming
/// requests carry a text channel; it is bounded (see [`StreamConfig`]).
pub struct WorkerMsg {
    pub request: GenerationRequest,
    pub token_tx: Option<mpsc::Sender<TextDelta>>,
    pub result_tx: oneshot::Sender<Result<GenerationStats>>,
}

impl WorkerMsg {
    /// Whether the client has gone away.
    fn is_abandoned(&self) -> bool {
        self.result_tx.is_closed() || self.token_tx.as_ref().is_some_and(mpsc::Sender::is_closed)
    }
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
    tx: mpsc::Sender<TextDelta>,
    content: String,
    reasoning: String,
}

impl Stream {
    /// Send everything buffered, reasoning first as the model writes it, keeping what the
    /// channel has no room for.
    fn flush(&mut self, limit: usize) -> std::result::Result<(), Undelivered> {
        for (buffer, wrap) in [
            (
                &mut self.reasoning,
                TextDelta::Reasoning as fn(String) -> TextDelta,
            ),
            (&mut self.content, TextDelta::Content),
        ] {
            if buffer.is_empty() {
                continue;
            }
            match self.tx.try_send(wrap(std::mem::take(buffer))) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(
                    TextDelta::Content(text) | TextDelta::Reasoning(text),
                )) => *buffer = text,
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(Undelivered::Disconnected);
                }
            }
        }
        let held = self.content.len() + self.reasoning.len();
        if held > limit {
            return Err(Undelivered::Stalled(held));
        }
        Ok(())
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
    report: Option<Arc<ModelReport>>,
    telemetry: Arc<WorkerTelemetry>,
    exit: watch::Receiver<()>,
}

const MAX_ADMISSIONS_PER_PASS: usize = 32;
/// Requests waiting for their clips to be encoded before new ones are rejected.
const MAX_ENCODING: usize = 8;

/// Encodes chat clips on its own OS thread, so text passes keep running meanwhile. A request
/// returns through the worker's intake once its clips are encoded. The thread stops when
/// the worker drops this handle; the weak sender lets the worker's intake close regardless.
struct ClipEncoder {
    tx: std_mpsc::SyncSender<WorkerMsg>,
}

/// The loaded model's device capabilities and launch tuning, when it runs on Metal kernels.
#[cfg(feature = "metal")]
fn kernel_report(model: &TextModel) -> serde_json::Value {
    use tur_models::hybrid::HybridModel as _;
    let Some(kernels) = model.kernels() else {
        return serde_json::Value::Null;
    };
    let device = kernels.capabilities();
    serde_json::json!({
        "device": device.name,
        "architecture": device.architecture,
        "apple_family": device.apple_family,
        "metal3": device.metal3,
        "metal4": device.metal4,
        "unified_memory": device.unified_memory,
        "max_threads_per_threadgroup": device.max_threads_per_threadgroup,
        "threadgroup_memory_bytes": device.threadgroup_memory_bytes,
        "max_buffer_bytes": device.max_buffer_bytes,
        "recommended_working_set_bytes": device.recommended_working_set_bytes,
        "tuning": format!("{:?}", kernels.tuning()),
    })
}

#[cfg(not(feature = "metal"))]
fn kernel_report(_model: &TextModel) -> serde_json::Value {
    serde_json::Value::Null
}

/// Facts to report for the model `pipeline` loaded from `factory`.
fn model_report(factory: &ModelFactory, pipeline: &InferencePipeline) -> ModelReport {
    let model = pipeline.model();
    ModelReport {
        source: match &factory.source {
            ModelSource::HuggingFace(id) => id.clone(),
            ModelSource::LocalPath(_) => "local".into(),
        },
        revision: factory.revision.clone(),
        weight_format: factory.format.map_or_else(
            || "as stored".into(),
            |format| format!("{format:?}").to_lowercase(),
        ),
        kv: format!("{}/{}", factory.kv.format, factory.kv.page_tokens),
        recurrent_state: format!("{:?}", factory.recurrent).to_lowercase(),
        max_position: model.max_position(),
        vocab_size: pipeline.vocab_size(),
        plan: *pipeline.memory_plan(),
        scheduler: *pipeline.scheduler().config(),
        device_sampling: pipeline.device_sampling(),
        speculation: pipeline.speculation_enabled(),
        kernels: kernel_report(model),
    }
}

/// Publish what the pass changed. `timing` is absent when no pass ran.
fn publish_pass(
    telemetry: &WorkerTelemetry,
    pipeline: &mut InferencePipeline,
    device: &candle_core::Device,
    timing: Option<crate::backend::pipeline::PassTiming>,
) -> Result<()> {
    telemetry.publish_pass(&PassReading {
        active: pipeline.active_request_count(),
        waiting: pipeline.waiting_request_count(),
        pool: pipeline.memory()?,
        prefix: pipeline.prefix_stats(),
        sampling: pipeline.sampling_stats(),
        speculation: pipeline.speculation_stats(),
        allocated: allocated_bytes(device),
        timing,
    });
    Ok(())
}

impl ClipEncoder {
    fn spawn(
        model: Arc<TextModel>,
        intake: mpsc::WeakSender<WorkerMsg>,
        telemetry: Arc<WorkerTelemetry>,
    ) -> Result<Self> {
        let (tx, rx) = std_mpsc::sync_channel::<WorkerMsg>(MAX_ENCODING);
        std::thread::Builder::new()
            .name("clip-encoder".into())
            .spawn(move || {
                while let Ok(msg) = rx.recv() {
                    encode_clips(&model, msg, &intake, &telemetry);
                }
            })?;
        Ok(Self { tx })
    }
}

fn encode_clips(
    model: &TextModel,
    mut msg: WorkerMsg,
    intake: &mpsc::WeakSender<WorkerMsg>,
    telemetry: &WorkerTelemetry,
) {
    if msg.is_abandoned() {
        telemetry.record_cancelled();
        return;
    }
    let clips: Vec<&Audio> = msg
        .request
        .messages
        .iter()
        .flat_map(|m| m.content.audio())
        .collect();
    match clips.iter().map(|clip| model.encode_audio(clip)).collect() {
        Ok(rows) => msg.request.encoded_audio = rows,
        Err(error) => return reply(telemetry, msg.result_tx, Err(error)),
    }
    // A worker that stopped drops the request, which tells its client.
    if let Some(intake) = intake.upgrade() {
        drop(intake.blocking_send(msg));
    }
}

/// Count a request's outcome and hand it over. A client that disconnected has dropped the
/// receiver, and nobody is left to tell.
fn reply(
    telemetry: &WorkerTelemetry,
    tx: oneshot::Sender<Result<GenerationStats>>,
    result: Result<GenerationStats>,
) {
    telemetry.record(&result);
    drop(tx.send(result));
}

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
    encoder: Option<&ClipEncoder>,
    pending: &mut HashMap<Uuid, PendingRequest>,
    telemetry: &WorkerTelemetry,
) {
    if msg.is_abandoned() {
        telemetry.record_cancelled();
        return;
    }
    if let Some(encoder) = encoder.filter(|_| msg.request.needs_audio_encoding()) {
        match encoder.tx.try_send(msg) {
            Ok(()) => {}
            Err(std_mpsc::TrySendError::Full(msg)) => reply(
                telemetry,
                msg.result_tx,
                Err(TurError::QueueFull("audio encoding queue is full".into())),
            ),
            Err(std_mpsc::TrySendError::Disconnected(msg)) => {
                reply(telemetry, msg.result_tx, Err(TurError::WorkerUnavailable));
            }
        }
        return;
    }
    match pipeline.submit_request(&msg.request) {
        Ok(id) => {
            pending.insert(
                id,
                PendingRequest {
                    stream: msg.token_tx.map(|tx| Stream {
                        tx,
                        content: String::new(),
                        reasoning: String::new(),
                    }),
                    result_tx: msg.result_tx,
                },
            );
        }
        Err(error) => {
            reply(telemetry, msg.result_tx, Err(error));
        }
    }
}

/// Release every request whose client went away.
fn release_disconnected(
    pipeline: &mut InferencePipeline,
    pending: &mut HashMap<Uuid, PendingRequest>,
    telemetry: &WorkerTelemetry,
) -> Result<()> {
    let gone: Vec<Uuid> = pending
        .iter()
        .filter(|(_, request)| request.disconnected())
        .map(|(id, _)| *id)
        .collect();
    for id in gone {
        pipeline.cancel(id)?;
        pending.remove(&id);
        telemetry.record_cancelled();
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
    telemetry: &WorkerTelemetry,
) -> Result<()> {
    for (id, text) in step.new_reasoning {
        if let Some(stream) = pending.get_mut(&id).and_then(|r| r.stream.as_mut()) {
            stream.reasoning.push_str(&text);
        }
    }
    for (id, text) in step.new_tokens {
        if let Some(stream) = pending.get_mut(&id).and_then(|r| r.stream.as_mut()) {
            stream.content.push_str(&text);
        }
    }
    for (id, result) in step.completed {
        if let Some(mut request) = pending.remove(&id) {
            // A remainder the channel cannot take is recovered by the adapter from the
            // final text, so completion never waits on the client.
            if let Some(stream) = &mut request.stream {
                stream.flush(usize::MAX).ok();
            }
            drop(request.stream);
            reply(telemetry, request.result_tx, result);
        }
    }
    flush_streams(pipeline, pending, config, telemetry)
}

/// Send each stream its buffered text, cancelling those that left or stalled.
fn flush_streams(
    pipeline: &mut InferencePipeline,
    pending: &mut HashMap<Uuid, PendingRequest>,
    config: StreamConfig,
    telemetry: &WorkerTelemetry,
) -> Result<()> {
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
        let request = pending.remove(&id);
        if stalled.is_none() {
            telemetry.record_cancelled();
        }
        if let (Some(request), Some(bytes)) = (request, stalled) {
            tracing::warn!(request = %id, bytes, "cancelled a stalled stream");
            reply(
                telemetry,
                request.result_tx,
                Err(TurError::SlowClient(format!(
                    "{bytes} bytes undelivered beyond the {}-byte bound",
                    config.max_buffered_bytes
                ))),
            );
        }
    }
    Ok(())
}

/// Fail everything the worker holds, between passes, for a shutdown that ran out of time.
fn abandon(
    pipeline: &mut InferencePipeline,
    pending: &mut HashMap<Uuid, PendingRequest>,
    telemetry: &WorkerTelemetry,
) -> Result<()> {
    for (id, request) in pending.drain() {
        pipeline.cancel(id)?;
        reply(
            telemetry,
            request.result_tx,
            Err(TurError::WorkerUnavailable),
        );
    }
    Ok(())
}

/// Run one pass. Intake (rendering and tokenization) and disconnects run while the device
/// executes it; the next pass needs its sampled tokens.
fn run_pass(
    rx: &mut mpsc::Receiver<WorkerMsg>,
    pipeline: &mut InferencePipeline,
    encoder: Option<&ClipEncoder>,
    pending: &mut HashMap<Uuid, PendingRequest>,
    telemetry: &WorkerTelemetry,
) -> Result<BatchStepOutput> {
    pipeline.begin_step()?;
    drain_intake(rx, |msg| admit(msg, pipeline, encoder, pending, telemetry));
    release_disconnected(pipeline, pending, telemetry)?;
    pipeline.finish_step()
}

/// A whole pass failed: every pending request gets the error.
fn fail_pending(
    pending: &mut HashMap<Uuid, PendingRequest>,
    telemetry: &WorkerTelemetry,
    error: &TurError,
) {
    for (_, request) in pending.drain() {
        reply(
            telemetry,
            request.result_tx,
            Err(TurError::Other(error.to_string())),
        );
    }
}

fn serve(
    mut rx: mpsc::Receiver<WorkerMsg>,
    intake: mpsc::WeakSender<WorkerMsg>,
    mut pipeline: InferencePipeline,
    config: StreamConfig,
    telemetry: &Arc<WorkerTelemetry>,
    device: &candle_core::Device,
) -> Result<()> {
    let encoder = pipeline
        .audio_model()
        .map(|model| ClipEncoder::spawn(model, intake, Arc::clone(telemetry)))
        .transpose()?;
    let encoder = encoder.as_ref();
    let telemetry = telemetry.as_ref();
    let mut pending: HashMap<Uuid, PendingRequest> = HashMap::new();
    publish_pass(telemetry, &mut pipeline, device, None)?;
    loop {
        if telemetry.stop_requested() {
            return abandon(&mut pipeline, &mut pending, telemetry);
        }
        if pending.is_empty() {
            let Some(msg) = rx.blocking_recv() else {
                return Ok(());
            };
            admit(msg, &mut pipeline, encoder, &mut pending, telemetry);
            drain_intake(&mut rx, |msg| {
                admit(msg, &mut pipeline, encoder, &mut pending, telemetry);
            });
        }
        let step = run_pass(&mut rx, &mut pipeline, encoder, &mut pending, telemetry);
        let step = match step {
            Ok(step) => step,
            Err(error) => {
                fail_pending(&mut pending, telemetry, &error);
                return Err(error);
            }
        };
        let timing = step.timing;
        deliver(step, &mut pipeline, &mut pending, config, telemetry)?;
        publish_pass(telemetry, &mut pipeline, device, timing)?;
    }
}

impl PipelineWorker {
    /// Load and validate the checkpoint before reporting readiness. Startup errors propagate.
    pub fn spawn(config: WorkerConfig) -> Result<Self> {
        let timeout = config.request_timeout;
        let stream = config.stream;
        let (tx, rx) = mpsc::channel(256);
        let intake = tx.downgrade();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let telemetry = Arc::new(WorkerTelemetry::default());
        let worker_telemetry = Arc::clone(&telemetry);
        let (exit_tx, exit) = watch::channel(());
        std::thread::Builder::new()
            .name("hybrid-worker".into())
            .spawn(move || {
                match InferencePipeline::with_budget(
                    &config.factory,
                    config.pipeline,
                    &config.budget,
                ) {
                    Ok(pipeline) => {
                        let report = model_report(&config.factory, &pipeline);
                        tracing::info!(
                            source = %report.source,
                            revision = %report.revision,
                            kernels = %report.kernels,
                            "Model loaded"
                        );
                        if ready_tx.send(Ok(report)).is_err() {
                            return;
                        }
                        if let Err(error) = serve(
                            rx,
                            intake,
                            pipeline,
                            stream,
                            &worker_telemetry,
                            &config.factory.device,
                        ) {
                            tracing::warn!("Hybrid worker stopped: {error}");
                        }
                    }
                    Err(error) => {
                        // The caller waits on this channel until the worker reports.
                        drop(ready_tx.send(Err(error)));
                    }
                }
                // Dropped last: the pipeline and its device memory are gone by now.
                drop(exit_tx);
            })?;
        let report = ready_rx
            .recv()
            .map_err(|error| TurError::Other(format!("Worker startup failed: {error}")))??;
        Ok(Self {
            tx,
            timeout,
            stream,
            report: Some(Arc::new(report)),
            telemetry,
            exit,
        })
    }

    /// Wrap a caller-provided sender, allowing HTTP tests to provide deterministic workers.
    pub fn from_sender(tx: mpsc::Sender<WorkerMsg>) -> Self {
        Self {
            tx,
            timeout: None,
            stream: StreamConfig::default(),
            report: None,
            telemetry: Arc::default(),
            exit: watch::channel(()).1,
        }
    }

    /// Whether the worker still takes requests. A stopped one never comes back.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }

    /// The loaded model, absent for a test double.
    pub fn report(&self) -> Option<&ModelReport> {
        self.report.as_deref()
    }

    /// Counters the worker publishes every pass.
    pub fn telemetry(&self) -> &WorkerTelemetry {
        &self.telemetry
    }

    /// A receiver whose sender drops when the worker has released its model.
    pub(crate) fn exit(&self) -> watch::Receiver<()> {
        self.exit.clone()
    }

    /// Bounded text channel for one streaming request.
    pub fn stream_channel(&self) -> (mpsc::Sender<TextDelta>, mpsc::Receiver<TextDelta>) {
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
    use candle_core::Device;
    use tur_models::models::{DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat};

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
            report: None,
            telemetry: Arc::default(),
            exit: watch::channel(()).1,
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
                audio: false,
                expert_cache: None,
            },
            pipeline: PipelineConfig::new(1, 64),
            budget: DeviceBudget::new(None),
            request_timeout: None,
            stream: StreamConfig::default(),
        });
        assert!(result.is_err());
    }
}
