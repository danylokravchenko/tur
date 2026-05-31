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

    let mut ids: Vec<&String> = state.workers.keys().collect();
    ids.sort();
    let data = ids
        .into_iter()
        .map(|id| ModelObject {
            id: id.clone(),
            object: "model",
            created,
            owned_by: "tur",
        })
        .collect();

    Json(ModelList {
        object: "list",
        data,
    })
}
