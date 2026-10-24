#![allow(
    clippy::print_stderr,
    reason = "checkpoint comparisons report numerical evidence to the test runner"
)]
#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![forbid(unsafe_code)]
//! Qwen3.5/3.8 template fidelity and real-checkpoint reference agreement.
//!
//! Checkpoint tests are `#[ignore]` and need `TUR_QWEN35_0_8B` (snapshot directory of
//! `Qwen/Qwen3.5-0.8B@2fc06364`) and `TUR_QWEN35_GOLDEN` (output of
//! `scripts/qwen3_5_reference.py --golden`). They fail, rather than skip, without them.

use candle_core::{Device, Tensor};
use std::path::PathBuf;
use tur_core::{Result, TurError, execution::invalid};
use tur_models::{
    backend::{
        chat_template::{ChatMessage, ChatTemplate, RenderOptions, Role, ToolCall},
        tools::ToolDefinition,
    },
    models::qwen3_5::{Execution, LoadOptions, Qwen35, WeightFormat},
};

const FIXTURES: &str = include_str!("fixtures/qwen3_5_templates.json");

fn templates(name: &str) -> Result<ChatTemplate> {
    let file = match name {
        "qwen3.5-0.8b" => "small-chat_template.jinja",
        "qwen3.8-27b" => "qwen3.8-chat_template.jinja",
        _ => return Err(invalid(format!("unknown template {name}"))),
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/references/p3")
        .join(file);
    ChatTemplate::from_template(std::fs::read_to_string(path)?)
}

fn message(raw: &serde_json::Value) -> Result<ChatMessage> {
    let role = match raw["role"].as_str() {
        Some("system") => Role::System,
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("tool") => Role::Tool,
        _ => return Err(invalid("fixture role")),
    };
    let mut message = ChatMessage::new(role, raw["content"].as_str().unwrap_or_default());
    message.reasoning_content = raw["reasoning_content"].as_str().map(String::from);
    for call in raw["tool_calls"].as_array().into_iter().flatten() {
        let function = &call["function"];
        message.tool_calls.push(ToolCall {
            name: function["name"].as_str().unwrap_or_default().into(),
            arguments: function["arguments"]
                .as_object()
                .cloned()
                .unwrap_or_default(),
        });
    }
    Ok(message)
}

fn tools(raw: &serde_json::Value) -> Vec<ToolDefinition> {
    raw.as_array()
        .into_iter()
        .flatten()
        .map(|tool| {
            let f = &tool["function"];
            ToolDefinition::new(
                f["name"].as_str().unwrap_or_default(),
                f["description"].as_str().unwrap_or_default(),
                f["parameters"].clone(),
            )
        })
        .collect()
}

fn options(case: &serde_json::Value) -> RenderOptions {
    let o = &case["options"];
    RenderOptions {
        add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap_or(true),
        enable_thinking: o["enable_thinking"].as_bool(),
        reasoning_effort: o["reasoning_effort"].as_str().map(String::from),
        preserve_thinking: o["preserve_thinking"].as_bool(),
    }
}

fn cases() -> Result<Vec<serde_json::Value>> {
    let fixtures: serde_json::Value = serde_json::from_str(FIXTURES)?;
    fixtures["cases"]
        .as_array()
        .cloned()
        .ok_or_else(|| invalid("fixture cases"))
}

#[test]
fn test_templates_render_exactly_like_transformers() -> Result<()> {
    let cases = cases()?;
    assert!(cases.len() >= 17);
    for case in cases {
        let template = templates(case["template"].as_str().unwrap_or_default())?;
        let messages = case["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .map(message)
            .collect::<Result<Vec<_>>>()?;
        let rendered = template.render(&messages, &tools(&case["tools"]), &options(&case))?;
        assert_eq!(
            rendered,
            case["text"].as_str().unwrap_or_default(),
            "{} / {}",
            case["template"],
            case["name"]
        );
    }
    Ok(())
}

#[test]
fn test_template_rejects_invalid_reasoning_effort() -> Result<()> {
    let template = templates("qwen3.8-27b")?;
    let options = RenderOptions {
        add_generation_prompt: true,
        reasoning_effort: Some("extreme".into()),
        ..Default::default()
    };
    assert!(
        template
            .render(&[ChatMessage::new(Role::User, "hi")], &[], &options)
            .is_err()
    );
    Ok(())
}

fn env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| TurError::Other(format!("set {name}; see the module documentation")))
}

fn device() -> Result<Device> {
    match std::env::var("TUR_DEVICE").as_deref() {
        Ok("metal") => Ok(Device::new_metal(0)?),
        _ => Ok(Device::Cpu),
    }
}

fn golden() -> Result<std::collections::HashMap<String, Tensor>> {
    Ok(candle_core::safetensors::load(
        env_path("TUR_QWEN35_GOLDEN")?,
        &Device::Cpu,
    )?)
}

fn ids(tensor: &Tensor) -> Result<Vec<u32>> {
    tensor
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| invalid("token id")))
        .collect()
}

fn relative(actual: &Tensor, expected: &Tensor) -> Result<f32> {
    let actual = actual.to_device(&Device::Cpu)?;
    let diff = (actual - expected)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let scale = expected.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(diff / scale.max(1e-6))
}

#[test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint"]
fn test_tokenizer_matches_reference_ids() -> Result<()> {
    let dir = env_path("TUR_QWEN35_0_8B")?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    for case in cases()? {
        let text = case["text"].as_str().unwrap_or_default();
        let encoded = tokenizer
            .encode(text, false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?;
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone())?;
        assert_eq!(encoded.get_ids(), expected.as_slice(), "{}", case["name"]);
    }
    Ok(())
}

/// F32 reference tolerances, relative to each tensor's largest magnitude.
const LAYER_TOLERANCE: f32 = 1e-4;
const LOGIT_TOLERANCE: f32 = 1e-4;

#[test]
#[ignore = "needs TUR_QWEN35_0_8B checkpoint and TUR_QWEN35_GOLDEN"]
fn test_f32_layers_and_logits_match_transformers() -> Result<()> {
    let golden = golden()?;
    let get = |name: &str| {
        golden
            .get(name)
            .ok_or_else(|| invalid(format!("golden {name}")))
    };
    let model = Qwen35::load(
        &env_path("TUR_QWEN35_0_8B")?,
        &LoadOptions {
            revision: "Qwen/Qwen3.5-0.8B@2fc06364".into(),
            weights: WeightFormat::F32.into(),
            device: device()?,
            execution: Execution::Fused,
            // Exact K/V storage: the Transformers reference keeps F32 keys and values.
            kv: tur_core::execution::KvLayout::new(16, tur_core::execution::KvFormat::F32),
            recurrent: candle_core::DType::F32,
            mtp: false,
        },
    )?;
    let tokens = ids(get("input_ids")?)?;
    let rows: Vec<usize> = get("logit_rows")?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|r| usize::try_from(r).map_err(|_| invalid("row")))
        .collect::<Result<_>>()?;
    let all: Vec<usize> = (0..tokens.len()).collect();
    let (layers, logits) = model.trace(&tokens, &all)?;
    let hidden = get("hidden")?;
    let mut worst = 0f32;
    for (index, layer) in layers.iter().enumerate() {
        let error = relative(layer, &hidden.get(index)?)?;
        worst = worst.max(error);
        assert!(
            error < LAYER_TOLERANCE,
            "layer {index}: relative error {error}"
        );
    }
    let selected = logits.index_select(
        &Tensor::from_vec(
            rows.iter().map(|&r| r as u32).collect::<Vec<_>>(),
            rows.len(),
            logits.device(),
        )?,
        0,
    )?;
    let logit_error = relative(&selected, get("logits")?)?;
    let argmax = ids(&logits
        .argmax(1)?
        .to_dtype(candle_core::DType::I64)?
        .to_device(&Device::Cpu)?)?;
    let expected = ids(get("argmax")?)?;
    let agree = argmax.iter().zip(&expected).filter(|(a, b)| a == b).count();
    eprintln!(
        "worst layer relative error {worst:.2e}; logits {logit_error:.2e}; argmax {agree}/{}",
        expected.len()
    );
    assert!(logit_error < LOGIT_TOLERANCE);
    assert_eq!(argmax, expected);
    Ok(())
}
