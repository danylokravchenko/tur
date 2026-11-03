//! Prompts of Intern-Decision checkpoints, which answer typed questions about a state with a
//! probability per option. One prompt holds every question: the user turn lists each field's
//! options under single-token symbols, and the assistant turn is a JSON object whose values
//! are `<decision>` markers. The logits just before each marker, restricted to that field's
//! symbols, are its answer, so all fields are read in one causal pass and nothing is generated.
//! The layout follows `compile_row` of the checkpoint's `inference.py`.

use super::chat_template::{ChatMessage, Content, ContentPart, Role};
use super::factory::ModelSource;
use crate::{Result, TurError};
use serde::Serialize;
use serde_json::{Value, ser::PrettyFormatter};

/// The added token each field's answer is read before.
pub const MARKER: &str = "<decision>";
/// Option symbols in order, each one token.
pub const SYMBOLS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
/// Questions one prompt may hold, as `inference.py` validates.
pub const MAX_FIELDS: usize = 16;
/// Images one prompt may hold, as `inference.py` validates.
pub const MAX_IMAGES: usize = 8;

const SYSTEM_PROMPT: &str = "You are a careful decision assistant. Use the state and decision \
schema in the user message to make the requested decisions. For every field, choose exactly one \
answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object \
mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. \
Do not include explanations, Markdown, or extra text.";

/// Descriptions of a yes/no question's options when the caller gives none.
pub const NO: &str = "The answer is no (negative, or disagree with the claim).";
pub const YES: &str = "The answer is yes (affirmative, or align with the claim).";

/// Candidate temperatures fitted for each published size, from its `inference.py`
/// (`DEFAULT_TEMPERATURE`); they flatten the raw candidate probabilities.
const TEMPERATURES: &[(&str, f64)] = &[("internlm/Intern-Decision-0.8B", 2.747_760_550_703)];

/// One question: its name, its instructions, and its options as (value, description).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub instructions: String,
    pub options: Vec<(String, String)>,
}

/// `value` as Python's `json.dumps(value, ensure_ascii=False, indent)` writes it.
fn pretty(value: &impl Serialize, indent: &[u8]) -> Result<String> {
    let mut out = Vec::new();
    let mut serializer =
        serde_json::Serializer::with_formatter(&mut out, PrettyFormatter::with_indent(indent));
    value.serialize(&mut serializer)?;
    String::from_utf8(out).map_err(|error| TurError::Other(error.to_string()))
}

/// The conversation of one decision request: system, user and the assistant's skeleton,
/// rendered with the generation prompt off, thinking off and `add_vision_id` on. `images`
/// open the user turn, in order. Fails when a field has no options or more than [`SYMBOLS`]
/// holds, when there are more than [`MAX_IMAGES`] images, or when the input already
/// contains [`MARKER`].
pub fn conversation(
    state: &Value,
    fields: &[Field],
    images: Vec<ContentPart>,
) -> Result<Vec<ChatMessage>> {
    let invalid = |message: String| TurError::InvalidExecution(message);
    if !(1..=MAX_FIELDS).contains(&fields.len()) {
        return Err(invalid(format!(
            "a decision takes 1 to {MAX_FIELDS} questions"
        )));
    }
    if images.len() > MAX_IMAGES
        || images
            .iter()
            .any(|part| !matches!(part, ContentPart::Image(_)))
    {
        return Err(invalid(format!(
            "a decision takes at most {MAX_IMAGES} images"
        )));
    }
    let mut schema = Vec::new();
    for field in fields {
        if !(1..=SYMBOLS.len()).contains(&field.options.len()) {
            return Err(invalid(format!(
                "question {:?} needs 1 to {} options",
                field.name,
                SYMBOLS.len()
            )));
        }
        schema.push(format!("{}: {}", field.name, field.instructions));
        for (symbol, (value, description)) in SYMBOLS.chars().zip(&field.options) {
            schema.push(format!("    {symbol} = {value}: {description}"));
        }
    }
    let user = format!(
        "Return one answer for every field using the supplied answer symbols.\n\n## State\n{}\n\
         ## Decision schema\n{}",
        pretty(state, b"  ")?,
        schema.join("\n")
    );
    if user.contains(MARKER) {
        return Err(invalid(format!("the input must not contain {MARKER}")));
    }
    let skeleton: serde_json::Map<String, Value> = fields
        .iter()
        .map(|field| (field.name.clone(), Value::String(MARKER.into())))
        .collect();
    let user = if images.is_empty() {
        ChatMessage::new(Role::User, user)
    } else {
        let mut parts = images;
        parts.push(ContentPart::Text(user));
        ChatMessage {
            role: Role::User,
            content: Content::Parts(parts),
            reasoning_content: None,
            tool_calls: Vec::new(),
        }
    };
    Ok(vec![
        ChatMessage::new(Role::System, SYSTEM_PROMPT),
        user,
        ChatMessage::new(Role::Assistant, pretty(&skeleton, b"    ")?),
    ])
}

/// The option symbols of every field, in field order.
pub fn symbols(fields: &[Field]) -> Vec<Vec<String>> {
    fields
        .iter()
        .map(|field| {
            SYMBOLS
                .chars()
                .take(field.options.len())
                .map(String::from)
                .collect()
        })
        .collect()
}

/// The candidate temperature of a published checkpoint; other sources, including local
/// copies, read uncalibrated at 1.
pub fn temperature(source: &ModelSource) -> f64 {
    match source {
        ModelSource::HuggingFace(id) => TEMPERATURES
            .iter()
            .find(|(name, _)| name == id)
            .map_or(1.0, |&(_, temperature)| temperature),
        ModelSource::LocalPath(_) => 1.0,
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn field(name: &str, options: usize) -> Field {
        Field {
            name: name.into(),
            instructions: "Which?".into(),
            options: (0..options)
                .map(|i| (i.to_string(), String::new()))
                .collect(),
        }
    }

    #[test]
    fn test_skeleton_holds_one_marker_per_field_in_order() -> Result<()> {
        let messages = conversation(&Value::Null, &[field("b", 2), field("a", 3)], Vec::new())?;
        let skeleton = messages[2].content.as_text().unwrap_or_default();
        assert_eq!(
            skeleton,
            "{\n    \"b\": \"<decision>\",\n    \"a\": \"<decision>\"\n}"
        );
        assert_eq!(symbols(&[field("b", 2)]), [["A", "B"]]);
        Ok(())
    }

    #[test]
    fn test_inputs_the_prompt_cannot_express_are_rejected() {
        assert!(conversation(&Value::Null, &[], Vec::new()).is_err());
        assert!(conversation(&Value::Null, &[field("x", 63)], Vec::new()).is_err());
        let state = Value::String(format!("say {MARKER}"));
        assert!(conversation(&state, &[field("x", 2)], Vec::new()).is_err());
    }
}
