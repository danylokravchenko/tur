//! `OpenAI`'s Decisions API, served by Intern-Decision checkpoints. Every question of a
//! request becomes a field of one prompt (see [`decision`]); its answer is the softmax of its
//! option symbols' logits just before its marker, tempered by the checkpoint's calibration.
//! Nothing is generated.

use std::collections::HashSet;

use crate::backend::decision::{self, Field};
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse, response::Response};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::{
    backend::{
        chat_template::RenderOptions,
        pipeline::{GenerationRequest, GenerationStats, Scoring},
    },
    hybrid::SamplingOptions,
    server::{
        AppState,
        routes::chat::{execution_error, image},
        types::{
            ApiErrorResponse, ChoiceProbability, ChoiceValue, DecisionAnswer, DecisionContent,
            DecisionInput, DecisionInputDetails, DecisionOutputDetails, DecisionPart,
            DecisionQuestion, DecisionRequest, DecisionResponse, DecisionUsage, LevelProbability,
        },
        worker::WorkerMsg,
    },
};

/// `OpenAI` caps a score question at ten levels.
const MAX_LEVELS: usize = 10;

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiErrorResponse::bad_request(message)),
    )
        .into_response()
}

fn failure(error: &crate::TurError) -> Response {
    let (status, body) = execution_error(error);
    (status, Json(body)).into_response()
}

fn question_name(question: &DecisionQuestion) -> Option<&str> {
    match question {
        DecisionQuestion::Predicate { name, .. }
        | DecisionQuestion::Choice { name, .. }
        | DecisionQuestion::Score { name, .. } => name.as_deref(),
    }
}

/// A choice value as its option's text: booleans read `true` and `false`.
fn value_text(value: &ChoiceValue) -> String {
    match value {
        ChoiceValue::Bool(value) => value.to_string(),
        ChoiceValue::Text(text) => text.clone(),
    }
}

fn choice_options(
    name: &str,
    choices: &[crate::server::types::DecisionChoice],
) -> Result<Vec<(String, String)>, String> {
    let options: Vec<(String, String)> = choices
        .iter()
        .map(|c| {
            (
                value_text(&c.value),
                c.description.clone().unwrap_or_default(),
            )
        })
        .collect();
    // `"true"` and `true` would read the same to the model.
    let distinct: HashSet<&str> = options.iter().map(|(value, _)| value.as_str()).collect();
    if !(2..=decision::SYMBOLS.len()).contains(&options.len()) || distinct.len() != options.len() {
        return Err(format!(
            "question {name:?} needs 2 to {} distinct choices",
            decision::SYMBOLS.len()
        ));
    }
    Ok(options)
}

/// Levels as options valued by their index, each described by its label and description.
fn level_options(
    name: &str,
    levels: &[crate::server::types::DecisionLevel],
) -> Result<Vec<(String, String)>, String> {
    if !(2..=MAX_LEVELS).contains(&levels.len()) {
        return Err(format!("question {name:?} needs 2 to {MAX_LEVELS} levels"));
    }
    Ok(levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let text = level
                .description
                .as_deref()
                .filter(|d| !d.is_empty())
                .map_or_else(
                    || level.label.clone(),
                    |description| format!("{}: {description}", level.label),
                );
            (index.to_string(), text)
        })
        .collect())
}

/// One question as a field. Unnamed questions are called `q1`, `q2`, ... by position.
fn field(index: usize, question: &DecisionQuestion) -> Result<Field, String> {
    let name = question_name(question).map_or_else(|| format!("q{}", index + 1), str::to_owned);
    let (instructions, options) = match question {
        DecisionQuestion::Predicate { instructions, .. } => (
            instructions,
            vec![
                ("no".to_owned(), decision::NO.to_owned()),
                ("yes".to_owned(), decision::YES.to_owned()),
            ],
        ),
        DecisionQuestion::Choice {
            instructions,
            choices,
            ..
        } => (instructions, choice_options(&name, choices)?),
        DecisionQuestion::Score {
            instructions,
            levels,
            ..
        } => (instructions, level_options(&name, levels)?),
    };
    Ok(Field {
        name,
        instructions: instructions.clone(),
        options,
    })
}

fn fields(questions: &[DecisionQuestion]) -> Result<Vec<Field>, String> {
    if !(1..=decision::MAX_FIELDS).contains(&questions.len()) {
        return Err(format!(
            "questions must hold 1 to {} items",
            decision::MAX_FIELDS
        ));
    }
    let fields = questions
        .iter()
        .enumerate()
        .map(|(index, question)| field(index, question))
        .collect::<Result<Vec<_>, _>>()?;
    let names: HashSet<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    if names.len() != fields.len() {
        return Err("question names must be unique, and unnamed ones are q1, q2, ...".into());
    }
    Ok(fields)
}

/// A part's value in the state; images are passed to the model apart from it.
fn part_value(part: &DecisionPart) -> Option<Value> {
    match part {
        DecisionPart::InputText { text } | DecisionPart::OutputText { text } => {
            Some(Value::String(text.clone()))
        }
        DecisionPart::InputJson { data } => Some(data.clone()),
        DecisionPart::InputImage { .. } => None,
    }
}

/// A message's parts as text, one per line; JSON parts are written compactly.
fn content_text(content: &DecisionContent) -> String {
    let parts = match content {
        DecisionContent::Text(text) => return text.clone(),
        DecisionContent::Parts(parts) => parts,
    };
    let texts: Vec<String> = parts
        .iter()
        .filter_map(part_value)
        .map(|value| match value {
            Value::String(text) => text,
            value => value.to_string(),
        })
        .collect();
    texts.join("\n")
}

/// The state the questions are about. A string stays a string, and a lone user message's
/// lone part besides its images its own value; anything else becomes a list of
/// `{role, content}` turns.
fn decision_state(input: &DecisionInput) -> Value {
    let messages = match input {
        DecisionInput::Text(text) => return Value::String(text.clone()),
        DecisionInput::Messages(messages) => messages,
    };
    if let [message] = messages.as_slice()
        && message.role == "user"
    {
        let values: Vec<Value> = match &message.content {
            DecisionContent::Text(text) => vec![Value::String(text.clone())],
            DecisionContent::Parts(parts) => parts.iter().filter_map(part_value).collect(),
        };
        match <[Value; 1]>::try_from(values) {
            Ok([value]) => return value,
            Err(values) if values.is_empty() => return Value::String(String::new()),
            Err(_) => {}
        }
    }
    messages
        .iter()
        .map(|message| {
            serde_json::json!({
                "role": message.role,
                "content": content_text(&message.content),
            })
        })
        .collect()
}

/// The input's image URLs, in order.
fn image_urls(input: &DecisionInput) -> Vec<String> {
    let DecisionInput::Messages(messages) = input else {
        return Vec::new();
    };
    messages
        .iter()
        .filter_map(|message| match &message.content {
            DecisionContent::Parts(parts) => Some(parts),
            DecisionContent::Text(_) => None,
        })
        .flatten()
        .filter_map(|part| match part {
            DecisionPart::InputImage { image_url } => Some(image_url.clone()),
            _ => None,
        })
        .collect()
}

/// One request reading every field; images are decoded here, so callers run this off the
/// async executor.
fn request(
    state: &Value,
    fields: &[Field],
    images: &[String],
    cache_salt: Option<String>,
) -> crate::Result<GenerationRequest> {
    let images = images
        .iter()
        .map(|url| image(url))
        .collect::<crate::Result<Vec<_>>>()?;
    Ok(GenerationRequest {
        messages: decision::conversation(state, fields, images)?,
        sampling: SamplingOptions {
            temperature: Some(0.0),
            ..SamplingOptions::default()
        },
        render: RenderOptions {
            add_generation_prompt: false,
            enable_thinking: Some(false),
            add_vision_id: Some(true),
            ..RenderOptions::default()
        },
        cache_salt,
        scoring: Some(Scoring {
            marker: decision::MARKER.into(),
            candidates: decision::symbols(fields),
        }),
        ..GenerationRequest::new(String::new(), 1)
    })
}

/// Calibrated probabilities of one field: `softmax(logits / temperature)`, which equals the
/// checkpoint's `softmax(log(softmax(logits)) / temperature)`.
fn probabilities(logits: &[f32], temperature: f64) -> crate::Result<Vec<f32>> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(crate::core::execution::invalid("nonfinite option logits"));
    }
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&value| f64::from(value) / temperature)
        .collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|&value| (value - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "probabilities are reported as f32"
    )]
    Ok(exp.into_iter().map(|value| (value / sum) as f32).collect())
}

/// The likeliest option, breaking ties by the smallest value text as the checkpoint does.
fn likeliest(field: &Field, probabilities: &[f32]) -> usize {
    (0..probabilities.len())
        .min_by(|&a, &b| {
            probabilities[b]
                .total_cmp(&probabilities[a])
                .then_with(|| field.options[a].0.cmp(&field.options[b].0))
        })
        .unwrap_or_default()
}

/// The answer a question's calibrated probabilities give. `confidence` is the probability
/// of the likeliest option.
fn answer(question: DecisionQuestion, field: &Field, probabilities: &[f32]) -> DecisionAnswer {
    let best = likeliest(field, probabilities);
    let confidence = probabilities.get(best).copied().unwrap_or_default();
    match question {
        DecisionQuestion::Predicate { name, .. } => DecisionAnswer::Predicate {
            name,
            probability: probabilities.get(1).copied().unwrap_or_default(),
        },
        DecisionQuestion::Choice { name, choices, .. } => {
            let probabilities: Vec<ChoiceProbability> = choices
                .into_iter()
                .zip(probabilities)
                .map(|(choice, &probability)| ChoiceProbability {
                    value: choice.value,
                    probability,
                })
                .collect();
            DecisionAnswer::Choice {
                name,
                choice: probabilities
                    .get(best)
                    .map_or(ChoiceValue::Bool(false), |p| p.value.clone()),
                confidence,
                probabilities,
            }
        }
        DecisionQuestion::Score { name, levels, .. } => DecisionAnswer::Score {
            name,
            #[allow(clippy::cast_precision_loss, reason = "a level index is at most 9")]
            score: probabilities
                .iter()
                .enumerate()
                .map(|(index, p)| index as f32 * p)
                .sum(),
            confidence,
            probabilities: levels
                .into_iter()
                .zip(probabilities)
                .enumerate()
                .map(|(value, (level, &probability))| LevelProbability {
                    value,
                    label: level.label,
                    probability,
                })
                .collect(),
        },
    }
}

fn usage(stats: &GenerationStats) -> DecisionUsage {
    let tokens = |count: usize| u32::try_from(count).unwrap_or(u32::MAX);
    DecisionUsage {
        input_tokens: tokens(stats.prompt_tokens),
        output_tokens: 0,
        total_tokens: tokens(stats.prompt_tokens),
        input_tokens_details: DecisionInputDetails {
            cached_tokens: tokens(stats.cached_tokens),
            cache_write_tokens: 0,
        },
        output_tokens_details: DecisionOutputDetails {
            reasoning_tokens: 0,
        },
    }
}

/// Run one decision request on the worker.
async fn run(
    worker: &crate::server::worker::PipelineWorker,
    request: GenerationRequest,
) -> crate::Result<GenerationStats> {
    let (result_tx, result_rx) = oneshot::channel();
    worker
        .send(WorkerMsg {
            request,
            token_tx: None,
            result_tx,
        })
        .await?;
    result_rx
        .await
        .map_err(|_| crate::TurError::WorkerUnavailable)?
}

/// `POST /v1/decisions`.
pub async fn decisions(
    State(state): State<AppState>,
    Json(req): Json<DecisionRequest>,
) -> Response {
    let model_id = if req.model.is_empty() {
        state.default_model.clone()
    } else {
        req.model
    };
    let Some(worker) = state.workers.get(&model_id).cloned() else {
        return (
            StatusCode::NOT_FOUND,
            Json(ApiErrorResponse::bad_request(format!(
                "model '{model_id}' not found"
            ))),
        )
            .into_response();
    };
    let Some(temperature) = worker.decision_temperature() else {
        return bad_request(format!("model '{model_id}' is not a decision checkpoint"));
    };
    let fields = match fields(&req.questions) {
        Ok(fields) => fields,
        Err(message) => return bad_request(message),
    };
    let (input, images) = (decision_state(&req.input), image_urls(&req.input));
    let salt = req.prompt_cache_key;
    let shared = fields.clone();
    let built = tokio::task::spawn_blocking(move || request(&input, &shared, &images, salt)).await;
    let request = match built {
        Ok(Ok(request)) => request,
        Ok(Err(error)) => return bad_request(error.to_string()),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiErrorResponse::internal("input decoding failed")),
            )
                .into_response();
        }
    };
    let stats = match run(&worker, request).await {
        Ok(stats) => stats,
        Err(error) => return failure(&error),
    };
    if stats.scores.len() != fields.len() {
        return failure(&crate::core::execution::invalid(
            "a question went unanswered",
        ));
    }
    let mut answers = Vec::with_capacity(fields.len());
    for ((question, field), logits) in req.questions.into_iter().zip(&fields).zip(&stats.scores) {
        match probabilities(logits, temperature) {
            Ok(probabilities) => answers.push(answer(question, field, &probabilities)),
            Err(error) => return failure(&error),
        }
    }
    Json(DecisionResponse {
        model: model_id,
        answers,
        usage: usage(&stats),
    })
    .into_response()
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use crate::server::types::{DecisionChoice, DecisionMessage};

    fn choice(name: Option<&str>, values: &[&str]) -> DecisionQuestion {
        DecisionQuestion::Choice {
            instructions: "Which?".into(),
            name: name.map(str::to_owned),
            choices: values
                .iter()
                .map(|value| DecisionChoice {
                    value: ChoiceValue::Text((*value).into()),
                    description: None,
                })
                .collect(),
        }
    }

    #[test]
    fn test_fields_reject_shapes_the_checkpoint_cannot_answer() {
        assert!(fields(&[]).is_err());
        assert!(fields(&[choice(None, &["only"])]).is_err());
        assert!(fields(&[choice(None, &["a", "a"])]).is_err());
        // An unnamed first question is q1.
        assert!(fields(&[choice(None, &["a", "b"]), choice(Some("q1"), &["a", "b"])]).is_err());
        let mixed = DecisionQuestion::Choice {
            instructions: "?".into(),
            name: None,
            choices: [ChoiceValue::Bool(true), ChoiceValue::Text("true".into())]
                .map(|value| DecisionChoice {
                    value,
                    description: None,
                })
                .into(),
        };
        assert!(fields(&[mixed]).is_err());
    }

    #[test]
    fn test_state_keeps_a_lone_value_and_lists_conversations() {
        let message = |role: &str, content: &str| DecisionMessage {
            role: role.into(),
            content: DecisionContent::Text(content.into()),
        };
        let json = DecisionInput::Messages(vec![DecisionMessage {
            role: "user".into(),
            content: DecisionContent::Parts(vec![DecisionPart::InputJson {
                data: serde_json::json!({"days_late": 9}),
            }]),
        }]);
        assert_eq!(decision_state(&json), serde_json::json!({"days_late": 9}));
        let turns =
            DecisionInput::Messages(vec![message("user", "hi"), message("assistant", "yes")]);
        assert_eq!(
            decision_state(&turns),
            serde_json::json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yes"}
            ])
        );
        // Images reach the model apart from the state.
        let photo = DecisionInput::Messages(vec![DecisionMessage {
            role: "user".into(),
            content: DecisionContent::Parts(vec![
                DecisionPart::InputImage {
                    image_url: "data:image/png;base64,AA==".into(),
                },
                DecisionPart::InputText {
                    text: "Returned item".into(),
                },
            ]),
        }]);
        assert_eq!(
            decision_state(&photo),
            Value::String("Returned item".into())
        );
        assert_eq!(image_urls(&photo), ["data:image/png;base64,AA=="]);
    }

    #[test]
    fn test_calibration_flattens_without_moving_the_argmax() -> crate::Result<()> {
        let raw = probabilities(&[2.0, 0.0], 1.0)?;
        let tempered = probabilities(&[2.0, 0.0], 2.0)?;
        assert!(tempered[0] < raw[0] && tempered[0] > tempered[1]);
        assert!((raw[0] - 1.0 / (1.0 + (-2f32).exp())).abs() < 1e-6);
        Ok(())
    }
}
