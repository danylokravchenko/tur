use axum::{Json, extract::State};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::server::{
    AppState,
    types::{ModelList, ModelObject},
};

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let chat = state
        .workers
        .iter()
        .map(|(id, worker)| (id, worker.report().map(|report| report.revision.clone())));
    let speech = state
        .transcribers
        .iter()
        .map(|(id, worker)| (id, worker.revision().map(str::to_owned)));
    let mut models: Vec<_> = chat.chain(speech).collect();
    models.sort();
    let data = models
        .into_iter()
        .map(|(id, revision)| ModelObject {
            id: id.clone(),
            object: "model",
            created,
            owned_by: "tur",
            revision,
        })
        .collect();

    Json(ModelList {
        object: "list",
        data,
    })
}
