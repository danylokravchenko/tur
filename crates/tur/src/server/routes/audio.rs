//! `OpenAI`-compatible speech-to-text: `/v1/audio/transcriptions` and `/translations`.
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::{
    TurError,
    audio::Audio,
    core::execution::invalid,
    models::whisper::{Task, TranscribeOptions},
    server::{
        AppState, media,
        routes::chat::execution_error,
        types::{ApiErrorResponse, TranscriptionResponse},
    },
};

/// The fields of a transcription form that Tur reads.
struct Form {
    audio: Audio,
    model: String,
    language: Option<String>,
    text: bool,
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiErrorResponse::bad_request(message)),
    )
        .into_response()
}

fn text(data: &[u8], name: &str) -> crate::Result<String> {
    String::from_utf8(data.to_vec()).map_err(|_| invalid(format!("field {name} must be UTF-8")))
}

/// Parse and decode the form. Runs on a blocking thread: WAV decoding and resampling
/// are CPU work.
fn parse(content_type: &str, body: &[u8]) -> crate::Result<Form> {
    let boundary = media::boundary(content_type)?;
    let (mut audio, mut model, mut language, mut format) = (None, String::new(), None, None);
    for field in media::multipart(body, &boundary)? {
        match field.name.as_str() {
            "file" => {
                let declared = field.content_type.as_deref().unwrap_or("audio/wav");
                let named_wav = field
                    .filename
                    .as_deref()
                    .is_none_or(|name| name.to_ascii_lowercase().ends_with(".wav"));
                if !declared.starts_with("audio/") && declared != "application/octet-stream" {
                    return Err(invalid("the file field must hold audio"));
                }
                if !named_wav && !declared.contains("wav") {
                    return Err(invalid("only WAV audio is supported"));
                }
                audio = Some(Audio::from_wav(field.data)?);
            }
            "model" => model = text(field.data, "model")?,
            "language" => language = Some(text(field.data, "language")?).filter(|l| !l.is_empty()),
            "response_format" => format = Some(text(field.data, "response_format")?),
            "temperature" => {
                let value: f64 = text(field.data, "temperature")?
                    .trim()
                    .parse()
                    .map_err(|_| invalid("temperature must be a number"))?;
                if value != 0.0 {
                    return Err(invalid("only greedy decoding (temperature 0) is supported"));
                }
            }
            "prompt" if field.data.is_empty() => {}
            "prompt" | "timestamp_granularities[]" | "timestamp_granularities" => {
                return Err(invalid(format!("{} is not supported", field.name)));
            }
            // Unknown fields are ignored, as `OpenAI` adds optional ones over time.
            _ => {}
        }
    }
    let text = match format.as_deref() {
        None | Some("json") => false,
        Some("text") => true,
        Some(other) => {
            return Err(invalid(format!(
                "response_format '{other}' is not supported; use json or text"
            )));
        }
    };
    Ok(Form {
        audio: audio.ok_or_else(|| invalid("the form needs a file field"))?,
        model,
        language,
        text,
    })
}

async fn speech(state: AppState, headers: HeaderMap, body: Bytes, task: Task) -> Response {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let form = match tokio::task::spawn_blocking(move || parse(&content_type, &body)).await {
        Ok(Ok(form)) => form,
        Ok(Err(error)) => return bad_request(error.to_string()),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiErrorResponse::internal("form decoding failed")),
            )
                .into_response();
        }
    };
    let name = if form.model.is_empty() {
        state.default_transcriber.clone()
    } else {
        Some(form.model.clone())
    };
    let Some(worker) = name.as_ref().and_then(|n| state.transcribers.get(n)) else {
        let available: Vec<_> = state.transcribers.keys().cloned().collect();
        return (
            StatusCode::NOT_FOUND,
            Json(ApiErrorResponse::bad_request(format!(
                "transcription model '{}' not found; available: {}",
                name.unwrap_or_default(),
                available.join(", ")
            ))),
        )
            .into_response();
    };
    let options = TranscribeOptions {
        language: form.language,
        task,
    };
    match worker.transcribe(form.audio, options).await {
        Ok(result) if form.text => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text,
        )
            .into_response(),
        Ok(result) => Json(TranscriptionResponse { text: result.text }).into_response(),
        Err(error) => {
            let (status, body) = execution_error(&error);
            if matches!(error, TurError::InvalidExecution(_)) {
                return bad_request(error.to_string());
            }
            (status, Json(body)).into_response()
        }
    }
}

/// Transcribe an uploaded WAV file in its spoken language (or `language`).
pub async fn transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    speech(state, headers, body, Task::Transcribe).await
}

/// Translate an uploaded WAV file into English text.
pub async fn translations(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    speech(state, headers, body, Task::Translate).await
}
