#![forbid(unsafe_code)]
//! A dedicated Whisper execution owner behind the transcription endpoints.
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, watch};

use crate::{
    Result, TurError,
    audio::Audio,
    backend::{factory::WhisperFactory, memory::DeviceBudget},
    models::whisper::{TranscribeOptions, Transcriber, Transcription},
};

/// One clip and where its outcome goes.
pub struct TranscriptionJob {
    pub audio: Audio,
    pub options: TranscribeOptions,
    pub reply: oneshot::Sender<Result<Transcription>>,
}

/// Checkpoint, intake bound and device budget of the transcription worker.
pub struct TranscriptionConfig {
    pub factory: WhisperFactory,
    /// Device memory shared with the text workers; the model's weights and one encoder
    /// pass are leased before loading.
    pub budget: DeviceBudget,
    /// Clips queued before new ones are rejected.
    pub max_waiting: usize,
}

/// Handle to the single-owner Whisper worker with a bounded intake queue. Clips are
/// transcribed one at a time, in arrival order.
#[derive(Clone)]
pub struct TranscriptionWorker {
    tx: mpsc::Sender<TranscriptionJob>,
    revision: Option<Arc<str>>,
    leased_bytes: usize,
    exit: watch::Receiver<()>,
}

impl TranscriptionWorker {
    /// Lease memory, load the model on a new OS thread and wait until it is ready.
    pub fn spawn(config: TranscriptionConfig) -> Result<Self> {
        if config.max_waiting == 0 {
            return Err(TurError::Other(
                "transcription queue must hold at least one clip".into(),
            ));
        }
        let (weights, scratch) = config.factory.memory_requirements()?;
        let revision: Arc<str> = config.factory.revision.as_str().into();
        let (exit_tx, exit) = watch::channel(());
        let lease = config.budget.lease(
            weights
                .checked_add(scratch)
                .ok_or_else(|| TurError::Other("Whisper memory overflow".into()))?,
        )?;
        let leased_bytes = lease.bytes();
        let (tx, mut rx) = mpsc::channel::<TranscriptionJob>(config.max_waiting);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("whisper-worker".into())
            .spawn(move || {
                let transcriber = match config.factory.create() {
                    Ok(transcriber) => transcriber,
                    Err(error) => {
                        // The caller waits on this channel until the worker reports.
                        drop(ready_tx.send(Err(error)));
                        return;
                    }
                };
                tracing::info!(
                    weights = transcriber.model().weight_bytes(),
                    leased = lease.bytes(),
                    "Whisper loaded"
                );
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                serve(&mut rx, &transcriber);
                drop(lease);
                drop(exit_tx);
            })?;
        ready_rx
            .recv()
            .map_err(|error| TurError::Other(format!("Whisper startup failed: {error}")))??;
        Ok(Self {
            tx,
            revision: Some(revision),
            leased_bytes,
            exit,
        })
    }

    /// Wrap a caller-provided sender, allowing HTTP tests to provide deterministic workers.
    pub fn from_sender(tx: mpsc::Sender<TranscriptionJob>) -> Self {
        Self {
            tx,
            revision: None,
            leased_bytes: 0,
            exit: watch::channel(()).1,
        }
    }

    /// Whether the worker still takes clips.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }

    /// The loaded checkpoint's immutable revision.
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    /// Device memory the model holds.
    pub fn leased_bytes(&self) -> usize {
        self.leased_bytes
    }

    /// A receiver whose sender drops when the worker has released its model.
    pub(crate) fn exit(&self) -> watch::Receiver<()> {
        self.exit.clone()
    }

    /// Queue a clip and wait for its transcription. A full queue fails immediately.
    pub async fn transcribe(
        &self,
        audio: Audio,
        options: TranscribeOptions,
    ) -> Result<Transcription> {
        let (reply, result) = oneshot::channel();
        self.tx
            .try_send(TranscriptionJob {
                audio,
                options,
                reply,
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => {
                    TurError::QueueFull("transcription queue is full".into())
                }
                mpsc::error::TrySendError::Closed(_) => TurError::WorkerUnavailable,
            })?;
        result.await.map_err(|_| TurError::WorkerUnavailable)?
    }
}

fn serve(rx: &mut mpsc::Receiver<TranscriptionJob>, transcriber: &Transcriber) {
    while let Some(job) = rx.blocking_recv() {
        // A client that disconnected while queued needs no work.
        if job.reply.is_closed() {
            continue;
        }
        let result = transcriber.transcribe(&job.audio, &job.options);
        // A client that disconnected meanwhile has dropped the receiver.
        drop(job.reply.send(result));
    }
}
