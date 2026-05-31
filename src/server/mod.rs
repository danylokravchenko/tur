pub mod routes;
pub mod types;
pub mod worker;

use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use std::collections::HashMap;
use worker::PipelineWorker;

/// Shared state injected into every request handler.
#[derive(Clone)]
pub struct AppState {
    /// Workers keyed by model ID (e.g. "qwen3-0.6b", "granite4.1-3b").
    pub workers: HashMap<String, PipelineWorker>,
    /// The model used when the client omits the `model` field.
    pub default_model: String,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(state: AppState, addr: SocketAddr) -> crate::Result<()> {
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| crate::TurError::Other(format!("Failed to bind to {addr}: {e}")))?;

    info!("Listening on http://{addr}");
    axum::serve(listener, router)
        .await
        .map_err(|e| crate::TurError::Other(format!("Server error: {e}")))
}
