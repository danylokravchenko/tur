//! Endpoints for whoever operates the server: health, readiness, metrics and diagnostics.

use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

use crate::{
    hybrid::PoolUsage,
    runtime::pool::Usage,
    server::{
        AppState,
        telemetry::{self, ModelReport, WorkerCounters},
    },
};

/// Names of workers that have stopped. Nothing restarts one, so the process needs replacing.
fn stopped_workers(state: &AppState) -> Vec<&str> {
    let chat = state
        .workers
        .iter()
        .filter(|(_, worker)| !worker.is_alive())
        .map(|(name, _)| name.as_str());
    let speech = state
        .transcribers
        .iter()
        .filter(|(_, worker)| !worker.is_alive())
        .map(|(name, _)| name.as_str());
    let mut stopped: Vec<&str> = chat.chain(speech).collect();
    stopped.sort_unstable();
    stopped
}

/// Liveness: the process answers and every worker runs. A supervisor should restart a
/// server that fails this.
pub async fn health(State(state): State<AppState>) -> Response {
    let stopped = stopped_workers(&state);
    if stopped.is_empty() {
        (StatusCode::OK, Json(json!({"status": "ok"}))).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "worker_stopped", "workers": stopped})),
        )
            .into_response()
    }
}

/// Readiness: new requests would be served. The listener opens only after every model
/// loaded, so this fails only for a stopped worker or a server that is draining.
pub async fn ready(State(state): State<AppState>) -> Response {
    let stopped = stopped_workers(&state);
    let status = if !stopped.is_empty() {
        "worker_stopped"
    } else if state.status.draining() {
        "draining"
    } else {
        return (StatusCode::OK, Json(json!({"status": "ready"}))).into_response();
    };
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"status": status, "workers": stopped})),
    )
        .into_response()
}

fn render_metrics(state: &AppState) -> Result<String, std::fmt::Error> {
    let mut out = String::new();
    telemetry::render_server(&mut out, &state.status)?;
    let mut names: Vec<&String> = state.workers.keys().collect();
    names.sort_unstable();
    for name in names {
        if let Some(worker) = state.workers.get(name) {
            telemetry::render_model(
                &mut out,
                name,
                worker.report(),
                &worker.telemetry().counters(),
                worker.is_alive(),
            )?;
        }
    }
    Ok(out)
}

/// Prometheus text exposition of the server's counters.
pub async fn metrics(State(state): State<AppState>) -> Response {
    let Ok(out) = render_metrics(&state) else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
        .into_response()
}

fn usage_json(usage: Usage, unit_bytes: usize) -> Value {
    json!({
        "capacity": usage.capacity,
        "leased": usage.leased,
        "retired": usage.retired,
        "peak": usage.peak,
        "unit_bytes": unit_bytes,
    })
}

fn pool_json(pool: &PoolUsage) -> Value {
    json!({
        "resident_bytes": pool.resident_bytes(),
        "leased_bytes": pool.leased_bytes(),
        "kv_pages": usage_json(pool.pages, pool.page_bytes),
        "window_pages": usage_json(pool.windows, pool.window_bytes),
        "recurrent_slots": usage_json(pool.slots, pool.slot_bytes),
    })
}

/// The loaded checkpoint and what it reserved.
fn report_json(report: &ModelReport) -> Value {
    let plan = &report.plan;
    json!({
        "source": report.source,
        "revision": report.revision,
        "weight_format": report.weight_format,
        "kv": report.kv,
        "recurrent_state": report.recurrent_state,
        "vocab_size": report.vocab_size,
        "max_position": report.max_position,
        "context": {
            "limit_tokens": plan.context_tokens,
            "full_contexts": plan.full_contexts,
            "sized_for_tokens": plan.sized_context,
            "max_running": report.scheduler.max_running,
        },
        "memory_plan": {
            "resident_bytes": plan.resident(),
            "weights": plan.weights,
            "state_pool": plan.state_pool,
            "scratch_peak": plan.scratch_peak,
            "in_flight": plan.in_flight,
            "overhead": plan.overhead,
            "other_models": plan.reserved,
            "device_limit": plan.limit,
            "headroom": plan.headroom(),
            "snapshots": plan.retained,
            "draft_slots": plan.draft_slots,
        },
        "scheduler": {
            "max_running": report.scheduler.max_running,
            "max_chunk": report.scheduler.max_chunk,
            "max_batch_tokens": report.scheduler.max_batch_tokens,
            "max_waiting": report.scheduler.max_waiting,
            "max_waiting_tokens": report.scheduler.max_waiting_tokens,
        },
        "device_sampling": report.device_sampling,
        "speculation": report.speculation,
        "kernels": report.kernels,
    })
}

/// What the worker has done and holds now.
fn counters_json(counters: &WorkerCounters) -> Value {
    let seconds = counters.pass_seconds;
    json!({
        "running": counters.running,
        "waiting": counters.waiting,
        "requests": counters.outcomes,
        "tokens": {
            "prompt": counters.prompt_tokens,
            "cached_prompt": counters.cached_tokens,
            "generated": counters.generated_tokens,
        },
        "preemptions": counters.preemptions,
        "latency": {
            "request_seconds": counters.request_seconds.json(),
            "first_token_seconds": counters.first_token_seconds.json(),
        },
        "pool": pool_json(&counters.pool),
        "device_allocated_bytes": counters.device_allocated,
        "device_allocated_peak_bytes": counters.device_allocated_peak,
        "passes": {
            "count": counters.passes,
            "tokens": counters.pass_tokens,
            "seconds": {
                "schedule": seconds.schedule,
                "encode": seconds.encode,
                "overlap": seconds.overlap,
                "wait": seconds.wait,
                "publish": seconds.publish,
                "device": seconds.device,
            },
        },
        "sampling": {
            "device_rows": counters.sampling.device_rows,
            "host_rows": counters.sampling.host_rows,
            "fallback_rows": counters.sampling.fallback_rows,
        },
        "prefix": {
            "entries": counters.prefix.entries,
            "lookups": counters.prefix.lookups,
            "hits": counters.prefix.hits,
            "hit_tokens": counters.prefix.hit_tokens,
            "inserted": counters.prefix.inserted,
            "evicted": counters.prefix.evicted,
            "collisions": counters.prefix.collisions,
        },
        "speculation": {
            "drafted": counters.speculation.drafted,
            "accepted": counters.speculation.accepted,
            "mtp_drafted": counters.speculation.mtp_drafted,
            "mtp_accepted": counters.speculation.mtp_accepted,
        },
    })
}

/// Everything known about the server, for a person debugging it. Prompts and paths never
/// appear here, but model details do, so keep the port away from untrusted clients.
pub async fn diagnostics(State(state): State<AppState>) -> Json<Value> {
    let chat: serde_json::Map<String, Value> = state
        .workers
        .iter()
        .map(|(name, worker)| {
            (
                name.clone(),
                json!({
                    "alive": worker.is_alive(),
                    "checkpoint": worker.report().map(report_json),
                    "activity": counters_json(&worker.telemetry().counters()),
                }),
            )
        })
        .collect();
    let speech: serde_json::Map<String, Value> = state
        .transcribers
        .iter()
        .map(|(name, worker)| {
            (
                name.clone(),
                json!({
                    "alive": worker.is_alive(),
                    "revision": worker.revision(),
                    "leased_bytes": worker.leased_bytes(),
                }),
            )
        })
        .collect();
    let system = state.status.system().map(|system| {
        json!({
            "spare_bytes": system.spare_bytes,
            "swapouts_since_start": system.swapouts,
            "process_resident_bytes": system.process_resident_bytes,
        })
    });
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "pid": std::process::id(),
        "uptime_seconds": state.status.uptime().as_secs_f64(),
        "draining": state.status.draining(),
        "system": system,
        "models": chat,
        "transcription": speech,
    }))
}
