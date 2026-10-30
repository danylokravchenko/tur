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
    models::whisper::{Fallback, Task, TranscribeOptions, Transcription},
    server::{
        AppState, media,
        routes::chat::execution_error,
        types::{
            ApiErrorResponse, TranscriptionResponse, TranscriptionSegment, TranscriptionWord,
            VerboseTranscription,
        },
    },
};

/// How a transcription is returned.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl Format {
    fn parse(name: Option<&str>) -> crate::Result<Self> {
        match name {
            None | Some("json") => Ok(Self::Json),
            Some("text") => Ok(Self::Text),
            Some("verbose_json") => Ok(Self::VerboseJson),
            Some("srt") => Ok(Self::Srt),
            Some("vtt") => Ok(Self::Vtt),
            Some(other) => Err(invalid(format!(
                "response_format '{other}' is not supported; use json, text, verbose_json, srt or vtt"
            ))),
        }
    }

    fn timed(self) -> bool {
        matches!(self, Self::VerboseJson | Self::Srt | Self::Vtt)
    }
}

/// The fields of a transcription form that Tur reads.
struct Form {
    audio: Audio,
    model: String,
    language: Option<String>,
    format: Format,
    prompt: Option<String>,
    temperature: f32,
    words: bool,
    condition_on_previous_text: bool,
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
    let (mut prompt, mut temperature, mut words, mut condition) = (None, 0.0, false, false);
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
                temperature = text(field.data, "temperature")?
                    .trim()
                    .parse()
                    .map_err(|_| invalid("temperature must be a number"))?;
                if !(0.0..=1.0).contains(&temperature) {
                    return Err(invalid("temperature must be between 0 and 1"));
                }
            }
            "prompt" => prompt = Some(text(field.data, "prompt")?).filter(|p| !p.is_empty()),
            "condition_on_previous_text" => {
                condition = text(field.data, "condition_on_previous_text")? == "true";
            }
            "timestamp_granularities[]" | "timestamp_granularities" => {
                match text(field.data, &field.name)?.as_str() {
                    "word" => words = true,
                    "segment" => {}
                    other => {
                        return Err(invalid(format!("unknown timestamp granularity '{other}'")));
                    }
                }
            }
            // Unknown fields are ignored, as `OpenAI` adds optional ones over time.
            _ => {}
        }
    }
    let format = Format::parse(format.as_deref())?;
    if words && format != Format::VerboseJson {
        return Err(invalid("word timestamps need response_format verbose_json"));
    }
    Ok(Form {
        audio: audio.ok_or_else(|| invalid("the form needs a file field"))?,
        model,
        language,
        format,
        prompt,
        temperature,
        words,
        condition_on_previous_text: condition,
    })
}

fn respond(result: Transcription, format: Format, task: Task, duration: f64) -> Response {
    let plain = |body: String, content_type: &'static str| {
        ([(header::CONTENT_TYPE, content_type)], body).into_response()
    };
    match format {
        Format::Json => Json(TranscriptionResponse { text: result.text }).into_response(),
        Format::Text => plain(result.text, "text/plain; charset=utf-8"),
        Format::Srt => plain(result.srt(), "application/x-subrip"),
        Format::Vtt => plain(result.vtt(), "text/vtt; charset=utf-8"),
        Format::VerboseJson => Json(VerboseTranscription {
            task: match task {
                Task::Transcribe => "transcribe",
                Task::Translate => "translate",
            },
            language: result.language,
            duration,
            text: result.text,
            words: result
                .segments
                .iter()
                .flat_map(|segment| &segment.words)
                .map(|word| TranscriptionWord {
                    word: word.word.clone(),
                    start: word.start,
                    end: word.end,
                })
                .collect(),
            segments: result
                .segments
                .into_iter()
                .enumerate()
                .map(|(id, segment)| TranscriptionSegment {
                    id,
                    seek: segment.seek,
                    start: segment.start,
                    end: segment.end,
                    text: segment.text,
                    tokens: segment.tokens,
                })
                .collect(),
        })
        .into_response(),
    }
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
        timestamps: form.format.timed(),
        word_timestamps: form.words,
        prompt: form.prompt,
        condition_on_previous_text: form.condition_on_previous_text,
        // As `OpenAI`'s API reads it: 0 retries doubtful windows, anything else is fixed.
        fallback: Some(if form.temperature > 0.0 {
            Fallback::fixed(form.temperature)
        } else {
            Fallback::default()
        }),
    };
    let duration = form.audio.duration().as_secs_f64();
    match worker.transcribe(form.audio, options).await {
        Ok(result) => respond(result, form.format, task, duration),
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
