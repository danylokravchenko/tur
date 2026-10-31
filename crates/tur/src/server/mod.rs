pub mod media;
pub mod routes;
pub mod telemetry;
pub mod transcription;
pub mod types;
pub mod worker;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, MatchedPath, Request, State},
    http::{Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{
    net::TcpListener,
    signal::unix::{SignalKind, signal},
    sync::{mpsc, watch},
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, warn};

use std::collections::HashMap;
pub use telemetry::ServerStatus;
use transcription::TranscriptionWorker;
use types::ApiErrorResponse;
use worker::PipelineWorker;

/// Largest request body: room for a few minutes of 48 kHz stereo WAV, base64-encoded
/// in a chat request or uploaded to the transcription endpoints.
pub const MAX_BODY_BYTES: usize = 64 << 20;

/// How long workers may take to exit after the HTTP side finished.
const WORKER_EXIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Shared state injected into every request handler.
#[derive(Clone)]
pub struct AppState {
    /// Workers keyed by the configured hybrid model ID.
    pub workers: HashMap<String, PipelineWorker>,
    /// The model used when the client omits the `model` field.
    pub default_model: String,
    /// Speech-to-text workers keyed by their served name.
    pub transcribers: HashMap<String, TranscriptionWorker>,
    /// The transcriber used when a form omits `model`.
    pub default_transcriber: Option<String>,
    /// Draining state and server-wide counters.
    pub status: Arc<ServerStatus>,
}

/// Count every exchange by route, and refuse new work while draining. Connections opened
/// before the drain can still send requests, which is why the closed listener is not enough.
async fn track(State(status): State<Arc<ServerStatus>>, request: Request, next: Next) -> Response {
    let path = request
        .extensions()
        .get::<MatchedPath>()
        .map_or("unmatched", MatchedPath::as_str)
        .to_owned();
    let response = if status.draining() && request.method() == Method::POST {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiErrorResponse::shutting_down()),
        )
            .into_response()
    } else {
        next.run(request).await
    };
    status.record_http(&path, response.status().as_u16());
    response
}

pub fn build_router(state: AppState) -> Router {
    let status = Arc::clone(&state.status);
    Router::new()
        .route("/health", get(routes::ops::health))
        .route("/ready", get(routes::ops::ready))
        .route("/metrics", get(routes::ops::metrics))
        .route("/diagnostics", get(routes::ops::diagnostics))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route(
            "/v1/audio/transcriptions",
            post(routes::audio::transcriptions),
        )
        .route("/v1/audio/translations", post(routes::audio::translations))
        .layer(middleware::from_fn_with_state(status, track))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Forward each SIGINT and SIGTERM until the receiver goes away.
pub fn forward_signals(tx: mpsc::Sender<()>) -> crate::Result<()> {
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = terminate.recv() => {}
                _ = interrupt.recv() => {}
            }
            if tx.send(()).await.is_err() {
                return;
            }
        }
    });
    Ok(())
}

/// Serve on `addr` until a signal, then drain for at most `drain`; see [`serve_until`].
pub async fn run_server(state: AppState, addr: SocketAddr, drain: Duration) -> crate::Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| crate::TurError::Other(format!("Failed to bind to {addr}: {e}")))?;
    info!("Listening on http://{addr}");
    let (tx, signals) = mpsc::channel(2);
    forward_signals(tx)?;
    serve_until(state, listener, signals, drain).await
}

/// Resolves when a drain should stop waiting for accepted requests: `limit` after the first
/// message on `signals`, or at the second. Tells `draining` as the drain begins.
async fn drain_finished(
    status: &ServerStatus,
    draining: &watch::Sender<bool>,
    signals: &mut mpsc::Receiver<()>,
    limit: Duration,
) {
    next_signal(signals).await;
    info!(
        seconds = limit.as_secs(),
        "Draining: finishing accepted requests"
    );
    status.begin_drain();
    draining.send_replace(true);
    let reason = match first_of(tokio::time::sleep(limit), next_signal(signals)).await {
        Some(()) => "Drain timed out",
        None => "Second signal",
    };
    warn!("{reason}; failing remaining requests");
}

/// The next message on `signals`. A closed channel never delivers one.
async fn next_signal(signals: &mut mpsc::Receiver<()>) {
    if signals.recv().await.is_none() {
        std::future::pending::<()>().await;
    }
}

/// The output of `work`, or `None` if `give_up` finishes first.
async fn first_of<T>(
    work: impl Future<Output = T>,
    give_up: impl Future<Output = ()>,
) -> Option<T> {
    tokio::select! {
        output = work => Some(output),
        () = give_up => None,
    }
}

/// Drop the state's senders and wait, within a bound, for workers to release the device. A
/// worker exits once no handler holds a sender.
async fn wait_for_workers(state: AppState) {
    let chat = state.workers.values().map(PipelineWorker::exit);
    let speech = state.transcribers.values().map(TranscriptionWorker::exit);
    let exits: Vec<_> = chat.chain(speech).collect();
    drop(state);
    let exited = tokio::time::timeout(WORKER_EXIT_TIMEOUT, async move {
        for mut exit in exits {
            // Only the sender's drop, at the worker's exit, ends this.
            while exit.changed().await.is_ok() {}
        }
    })
    .await;
    if exited.is_err() {
        warn!("A worker did not exit in time; leaving it to process exit");
    }
}

/// Wait for the first message on `signals`, then close the listener, refuse new requests and
/// let accepted ones finish. After `drain`, or at a second message, workers fail what they
/// still hold. Returns once workers have released the device or a further bound passed.
pub async fn serve_until(
    state: AppState,
    listener: TcpListener,
    mut signals: mpsc::Receiver<()>,
    drain: Duration,
) -> crate::Result<()> {
    let status = Arc::clone(&state.status);
    let monitor = tokio::spawn(telemetry::monitor_system(Arc::clone(&status)));
    let (draining, mut begun) = watch::channel(false);
    let server =
        axum::serve(listener, build_router(state.clone())).with_graceful_shutdown(async move {
            // A closed sender means the server is being dropped anyway.
            drop(begun.wait_for(|draining| *draining).await);
        });
    let finished = first_of(
        async { server.await },
        drain_finished(&status, &draining, &mut signals, drain),
    )
    .await;
    monitor.abort();
    status.begin_drain();
    if finished.is_none() {
        for worker in state.workers.values() {
            worker.telemetry().request_stop();
        }
    }
    wait_for_workers(state).await;
    info!("Server stopped");
    finished
        .transpose()
        .map_err(|e| crate::TurError::Other(format!("Server error: {e}")))
        .map(drop)
}
