//! Image input against pinned Transformers references: preprocessing, the Qwen3.5 and
//! Gemma 4 vision towers, and the text model reading their rows.
use crate::model_common::{max_relative, rows};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf};
use tur_core::{
    Result,
    execution::{EmbeddedInput, KvFormat, KvLayout, invalid},
};
use tur_models::{
    image::Image,
    models::{Execution, LoadOptions, WeightFormat, gemma4::Gemma4, qwen3_5::Qwen35},
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vision")
        .join(name)
}

fn golden(name: &str) -> Result<HashMap<String, Tensor>> {
    Ok(candle_core::safetensors::load(
        fixture(name).join("golden.safetensors"),
        &Device::Cpu,
    )?)
}

fn tensor<'a>(golden: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    golden
        .get(name)
        .ok_or_else(|| invalid(format!("missing golden {name}")))
}

fn image(name: &str) -> Result<Image> {
    Image::decode(&std::fs::read(fixture(name))?)
}

fn options(device: &Device, execution: Execution) -> LoadOptions {
    LoadOptions {
        revision: "vision-transformers-5.20.2".into(),
        weights: WeightFormat::F32.into(),
        device: device.clone(),
        execution,
        kv: KvLayout::new(4, KvFormat::F32),
        recurrent: DType::F32,
        mtp: None,
        audio: false,
        vision: true,
        expert_cache: None,
    }
}

fn u32s(tensor: &Tensor) -> Result<Vec<u32>> {
    Ok(tensor.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?)
}

/// Largest absolute difference of two equally shaped tensors.
fn max_absolute(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let b = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    assert_eq!(a.dims(), b.dims());
    Ok((a - b)?.abs()?.max_all()?.to_scalar::<f32>()?)
}

/// The golden's patches keep the two frames of a still image; Tur folds them into one.
fn first_frame(pixels: &Tensor) -> Result<Tensor> {
    let (patches, width) = pixels.dims2()?;
    let per_frame = width / 6;
    Ok(pixels
        .reshape((patches, 3, 2, per_frame))?
        .narrow(2, 0, 1)?
        .reshape((patches, width / 2))?)
}

/// Preprocessed pixels are exact up to F32 rounding of the normalization.
const PIXEL_GATE: f32 = 1e-5;

fn qwen_conformance(device: &Device, execution: Execution) -> Result<()> {
    let golden = golden("qwen3_5")?;
    let model = Qwen35::load(&fixture("qwen3_5"), &options(device, execution))?;
    let picture = image("small.png")?;
    let [pixels, hidden, soft] = model.trace_image(&picture)?;
    assert!(max_absolute(&pixels, &first_frame(tensor(&golden, "small.pixels")?)?)? <= PIXEL_GATE);
    let error = max_relative(&rows(&hidden)?, &rows(tensor(&golden, "tower")?)?);
    assert!(error < 1e-4, "tower: {error}");
    let error = max_relative(&rows(&soft)?, &rows(tensor(&golden, "soft_tokens")?)?);
    assert!(error < 1e-4, "soft tokens: {error}");
    let tokens = u32s(tensor(&golden, "tokens")?)?;
    let prompt = model
        .image_prompt()
        .ok_or_else(|| invalid("vision tower not loaded"))?;
    let first = tokens
        .iter()
        .position(|&t| t == prompt.placeholder)
        .ok_or_else(|| invalid("no image placeholder"))?;
    let axial: Vec<[u32; 3]> = u32s(tensor(&golden, "rotary")?)?
        .as_chunks::<3>()
        .0
        .to_vec();
    let all: Vec<usize> = (0..tokens.len()).collect();
    let embedded = [EmbeddedInput {
        tokens: first..first + soft.dim(0)?,
        rows: soft.to_device(&Device::Cpu)?.to_device(device)?,
    }];
    let (trace, logits) = model.trace_media(&tokens, &all, &embedded, &axial)?;
    for (i, state) in trace.iter().enumerate() {
        let error = max_relative(
            &rows(state)?,
            &rows(tensor(&golden, &format!("layer.{i}"))?)?,
        );
        assert!(error < 1e-4, "layer {i}: {error}");
    }
    let error = max_relative(&rows(&logits)?, &rows(tensor(&golden, "logits")?)?);
    assert!(error < 1e-4, "logits: {error}");
    Ok(())
}

#[test]
fn test_qwen_vision_reference_matches_transformers() -> Result<()> {
    qwen_conformance(&Device::Cpu, Execution::Reference)
}

#[test]
fn test_qwen_vision_metal_matches_transformers() -> Result<()> {
    qwen_conformance(&Device::new_metal(0)?, Execution::Fused)
}

#[test]
fn test_downscaling_matches_the_references_antialiased_resampling() -> Result<()> {
    let picture = image("large.png")?;
    let qwen_golden = golden("qwen3_5")?;
    let qwen = Qwen35::load(
        &fixture("qwen3_5"),
        &options(&Device::Cpu, Execution::Reference),
    )?;
    let [pixels, ..] = qwen.trace_image(&picture)?;
    let expected = first_frame(tensor(&qwen_golden, "large.pixels")?)?;
    assert!(max_absolute(&pixels, &expected)? <= PIXEL_GATE);
    let gemma_golden = golden("gemma4")?;
    let gemma = Gemma4::load(
        &fixture("gemma4"),
        &options(&Device::Cpu, Execution::Reference),
    )?;
    let [pixels, ..] = gemma.trace_image(&picture)?;
    assert!(max_absolute(&pixels, tensor(&gemma_golden, "large.pixels")?)? <= PIXEL_GATE);
    Ok(())
}

/// Text states on this fixture have an RMS of 1e-3 to 4e-2, so F32 rounding that differs
/// between CPU (2.7e-4) and Metal (6.2e-4) is what remains; the towers meet 1e-6.
const TEXT_GATE: f32 = 1e-3;

fn gemma_conformance(device: &Device, execution: Execution) -> Result<()> {
    let golden = golden("gemma4")?;
    let model = Gemma4::load(&fixture("gemma4"), &options(device, execution))?;
    let picture = image("small.png")?;
    assert_eq!(
        model.image_tokens(&picture)?,
        tensor(&golden, "soft_tokens")?.dim(0)?
    );
    let [pixels, pooled, soft] = model.trace_image(&picture)?;
    assert!(max_absolute(&pixels, tensor(&golden, "small.pixels")?)? <= PIXEL_GATE);
    let error = max_relative(&rows(&pooled)?, &rows(tensor(&golden, "tower")?)?);
    assert!(error < 1e-4, "tower: {error}");
    let error = max_relative(&rows(&soft)?, &rows(tensor(&golden, "soft_tokens")?)?);
    assert!(error < 1e-4, "soft tokens: {error}");
    let prompt = model
        .image_prompt()
        .ok_or_else(|| invalid("vision tower not loaded"))?;
    let slot = model
        .media_slot()
        .ok_or_else(|| invalid("vision tower not loaded"))?;
    let reference = u32s(tensor(&golden, "tokens")?)?;
    let first = reference
        .iter()
        .position(|&t| t == prompt.placeholder)
        .ok_or_else(|| invalid("no image placeholder"))?;
    assert_eq!(Some(reference[first - 1]), prompt.begin);
    assert_eq!(Some(reference[first + soft.dim(0)?]), prompt.end);
    let tokens: Vec<u32> = reference
        .iter()
        .map(|&t| if t == prompt.placeholder { slot } else { t })
        .collect();
    let all: Vec<usize> = (0..tokens.len()).collect();
    let embedded = [EmbeddedInput {
        tokens: first..first + soft.dim(0)?,
        rows: soft.to_device(&Device::Cpu)?.to_device(device)?,
    }];
    let (trace, logits) = model.trace(&tokens, &all, &embedded)?;
    for (i, state) in trace.iter().enumerate() {
        let error = max_relative(
            &rows(state)?,
            &rows(tensor(&golden, &format!("layer.{i}"))?)?,
        );
        assert!(error < TEXT_GATE, "layer {i}: {error}");
    }
    let error = max_relative(&rows(&logits)?, &rows(tensor(&golden, "logits")?)?);
    assert!(error < TEXT_GATE, "logits: {error}");
    Ok(())
}

#[test]
fn test_gemma4_vision_reference_matches_transformers() -> Result<()> {
    gemma_conformance(&Device::Cpu, Execution::Reference)
}

#[test]
fn test_gemma4_vision_metal_matches_transformers() -> Result<()> {
    gemma_conformance(&Device::new_metal(0)?, Execution::Fused)
}

/// A word-level tokenizer over the tiny checkpoints' 96 tokens, with `names` as their special
/// tokens, and a template rendering typed parts the way the real one does.
fn chat_checkpoint(
    model: &str,
    names: &[(u32, &str)],
    image: &str,
) -> Result<crate::common::Checkpoint> {
    use tokenizers::{
        AddedToken, Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };
    let dir = std::env::temp_dir().join(format!("tur-vision-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir)?;
    let checkpoint = crate::common::Checkpoint(dir);
    for entry in std::fs::read_dir(fixture(model))? {
        let entry = entry?;
        if entry.file_name() != "golden.safetensors" {
            std::fs::copy(entry.path(), checkpoint.0.join(entry.file_name()))?;
        }
    }
    let name = |id: u32| {
        names
            .iter()
            .find(|(token, _)| *token == id)
            .map_or_else(|| format!("t{id}"), |(_, name)| (*name).to_owned())
    };
    let word = WordLevel::builder()
        .vocab((0..96u32).map(|id| (name(id), id)).collect())
        .unk_token("t0".into())
        .build()
        .map_err(|e| tur::TurError::Tokenizer(e.to_string()))?;
    let mut tokenizer = Tokenizer::new(word);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    let special: Vec<AddedToken> = names
        .iter()
        .map(|(_, name)| AddedToken::from(*name, true))
        .collect();
    tokenizer.add_special_tokens(&special);
    tokenizer
        .save(checkpoint.0.join("tokenizer.json"), false)
        .map_err(|e| tur::TurError::Tokenizer(e.to_string()))?;
    let template = format!(
        "{{% for m in messages %}}{{% if m.content is string %}}{{{{ m.content }}}}\
         {{% else %}}{{% for p in m.content %}}{{% if p.type == 'image' %}}{image}{{% else %}}\
         {{{{ p.text }}}}{{% endif %}}{{% endfor %}}{{% endif %}}{{% endfor %}}"
    );
    std::fs::write(
        checkpoint.0.join("tokenizer_config.json"),
        serde_json::json!({ "chat_template": template }).to_string(),
    )?;
    std::fs::write(
        checkpoint.0.join("generation_config.json"),
        r#"{"do_sample":false}"#,
    )?;
    Ok(checkpoint)
}

fn qwen_chat() -> Result<crate::common::Checkpoint> {
    chat_checkpoint(
        "qwen3_5",
        &[
            (90, "<|image_pad|>"),
            (92, "<|vision_start|>"),
            (93, "<|vision_end|>"),
        ],
        "<|vision_start|><|image_pad|><|vision_end|>",
    )
}

fn factory(checkpoint: &crate::common::Checkpoint, device: &Device) -> tur::ModelFactory {
    tur::ModelFactory {
        source: tur::ModelSource::LocalPath(checkpoint.0.clone()),
        revision: "tiny-vision-chat-v1".into(),
        format: Some(WeightFormat::F32),
        device: device.clone(),
        kv: KvLayout::new(4, KvFormat::F32),
        recurrent: DType::F32,
        audio: false,
        vision: true,
        expert_cache: None,
        gguf: None,
    }
}

fn image_request(
    picture: Image,
    (before, after): (&str, &str),
    max_tokens: usize,
) -> tur::backend::pipeline::GenerationRequest {
    use tur::backend::chat_template::{ChatMessage, Content, ContentPart, Role};
    let mut request = tur::backend::pipeline::GenerationRequest::new(String::new(), max_tokens);
    request.render.add_generation_prompt = false;
    request.messages = vec![ChatMessage {
        content: Content::Parts(vec![
            ContentPart::Text(before.into()),
            ContentPart::Image(picture),
            ContentPart::Text(after.into()),
        ]),
        ..ChatMessage::new(Role::User, "")
    }];
    request
}

/// The golden prompt once expanded: `t2 t7`, the image in its frame, `t9 t3 t5 t6`.
fn qwen_request(picture: Image, max_tokens: usize) -> tur::backend::pipeline::GenerationRequest {
    image_request(picture, ("t2 t7 ", " t9 t3 t5 t6"), max_tokens)
}

/// `small.png` with every value inverted: the same size and grid, with different content.
fn inverted() -> Result<Image> {
    let picture = image("small.png")?;
    Image::from_rgb(
        u32::try_from(picture.width()).map_err(|_| invalid("width"))?,
        u32::try_from(picture.height()).map_err(|_| invalid("height"))?,
        picture.pixels().iter().map(|v| 255 - v).collect(),
    )
}

#[test]
fn test_qwen_image_chat_matches_transformers_and_isolates_images() -> Result<()> {
    use tur::{InferencePipeline, backend::pipeline::PipelineConfig};
    let golden = golden("qwen3_5")?;
    let logits = tensor(&golden, "logits")?;
    let first = logits
        .get(logits.dim(0)? - 1)?
        .argmax(0)?
        .to_scalar::<u32>()?;
    let checkpoint = qwen_chat()?;
    let device = Device::new_metal(0)?;
    let mut pipeline =
        InferencePipeline::new(&factory(&checkpoint, &device), PipelineConfig::new(2, 256))?;
    let stats = pipeline.run(&qwen_request(image("small.png")?, 3), |_| {})?;
    assert_eq!(stats.prompt_tokens, tensor(&golden, "tokens")?.dim(0)?);
    let words: Vec<&str> = stats.generated_text.split_whitespace().collect();
    assert_eq!(words.first().copied(), Some(format!("t{first}").as_str()));
    // Chunks that split the grid change nothing.
    let mut chunked =
        InferencePipeline::new(&factory(&checkpoint, &device), PipelineConfig::new(1, 4))?;
    assert_eq!(
        chunked
            .run(&qwen_request(image("small.png")?, 3), |_| {})?
            .generated_text,
        stats.generated_text
    );
    Ok(())
}

#[test]
fn test_prefix_reuse_restores_an_image_and_tells_images_apart() -> Result<()> {
    use tur::{
        InferencePipeline,
        backend::{pipeline::PipelineConfig, prefix::PrefixConfig},
    };
    let (before, after) = ("t4 ".repeat(70), " t8".repeat(70));
    let request = |picture| image_request(picture, (&before, &after), 2);
    let checkpoint = qwen_chat()?;
    let device = Device::new_metal(0)?;
    let new = || {
        InferencePipeline::new(
            &factory(&checkpoint, &device),
            PipelineConfig {
                prefix: PrefixConfig {
                    spacing: 16,
                    entries: 2,
                    state_share: None,
                    restorable_first: false,
                    tool_hold: std::time::Duration::ZERO,
                },
                ..PipelineConfig::new(2, 64)
            },
        )
    };
    let mut pipeline = new()?;
    let stats = pipeline.run(&request(image("small.png")?), |_| {})?;
    // A repeat restores past the image, and continues as it did.
    let repeat = pipeline.run(&request(image("small.png")?), |_| {})?;
    assert_eq!(repeat.generated_text, stats.generated_text);
    assert!(
        repeat.cached_tokens > 70 + 6,
        "{} cached",
        repeat.cached_tokens
    );
    // Another image of the same size shares no state past the text before it.
    let shared = pipeline.run(&request(inverted()?), |_| {})?;
    assert!(
        shared.cached_tokens <= 71,
        "{} cached",
        shared.cached_tokens
    );
    let alone = new()?.run(&request(inverted()?), |_| {})?;
    assert_eq!(shared.generated_text, alone.generated_text);
    Ok(())
}

/// A text request packed beside an image's rows moves to the axial kernel with them, and its
/// answer must not notice; the image's answer must not notice the text either.
#[test]
fn test_requests_beside_an_image_answer_as_they_do_alone() -> Result<()> {
    use tur::{
        InferencePipeline,
        backend::pipeline::{GenerationRequest, PipelineConfig},
    };
    let checkpoint = qwen_chat()?;
    let device = Device::new_metal(0)?;
    let new = |slots| {
        InferencePipeline::new(
            &factory(&checkpoint, &device),
            PipelineConfig::new(slots, 256),
        )
    };
    let text = GenerationRequest::new("t4 t8 t15 t16 t23".into(), 6);
    let text_alone = new(1)?.run(&text, |_| {})?.generated_text;
    let image_alone = new(1)?
        .run(&qwen_request(image("small.png")?, 6), |_| {})?
        .generated_text;
    let mut both = new(2)?;
    let first = both.submit_request(&qwen_request(image("small.png")?, 6))?;
    let second = both.submit_request(&text)?;
    let (mut image_text, mut plain_text) = (None, None);
    while image_text.is_none() || plain_text.is_none() {
        for (id, result) in both.step_streaming()?.completed {
            let answer = result?.generated_text;
            if id == first {
                image_text = Some(answer);
            } else if id == second {
                plain_text = Some(answer);
            }
        }
    }
    assert_eq!(image_text.as_deref(), Some(image_alone.as_str()));
    assert_eq!(plain_text.as_deref(), Some(text_alone.as_str()));
    Ok(())
}

#[test]
fn test_images_need_a_tower_and_one_placeholder_each() -> Result<()> {
    use tur::{
        InferencePipeline, TurError,
        backend::pipeline::{GenerationRequest, PipelineConfig},
    };
    let checkpoint = qwen_chat()?;
    let device = Device::Cpu;
    let mut pipeline =
        InferencePipeline::new(&factory(&checkpoint, &device), PipelineConfig::new(1, 256))?;
    // A placeholder typed as text has no image behind it.
    let typed = GenerationRequest::new("t2 <|image_pad|> t3".into(), 2);
    assert!(matches!(
        pipeline.submit_request(&typed),
        Err(TurError::InvalidExecution(message)) if message.contains("placeholders")
    ));
    // Nor is one allowed in through a model without the tower.
    let without = tur::ModelFactory {
        vision: false,
        ..factory(&checkpoint, &device)
    };
    let mut text = InferencePipeline::new(&without, PipelineConfig::new(1, 256))?;
    assert!(matches!(
        text.submit_request(&qwen_request(image("small.png")?, 2)),
        Err(TurError::InvalidExecution(message)) if message.contains("no image")
    ));
    assert!(matches!(
        text.run(&typed, |_| {}),
        Err(TurError::InvalidExecution(_))
    ));
    assert_eq!(text.active_request_count(), 0);
    Ok(())
}

#[test]
fn test_gemma4_image_chat_frames_its_soft_tokens_as_transformers_does() -> Result<()> {
    use tur::{InferencePipeline, backend::pipeline::PipelineConfig};
    let golden = golden("gemma4")?;
    let logits = tensor(&golden, "logits")?;
    let first = logits
        .get(logits.dim(0)? - 1)?
        .argmax(0)?
        .to_scalar::<u32>()?;
    let checkpoint = chat_checkpoint("gemma4", &[(90, "<|image|>")], "<|image|>")?;
    let mut pipeline = InferencePipeline::new(
        &factory(&checkpoint, &Device::new_metal(0)?),
        PipelineConfig::new(1, 256),
    )?;
    let stats = pipeline.run(
        &image_request(image("small.png")?, ("t2 t7 ", " t9 t3"), 2),
        |_| {},
    )?;
    assert_eq!(stats.prompt_tokens, tensor(&golden, "tokens")?.dim(0)?);
    let words: Vec<&str> = stats.generated_text.split_whitespace().collect();
    assert_eq!(words.first().copied(), Some(format!("t{first}").as_str()));
    Ok(())
}

#[tokio::test]
async fn test_worker_encodes_images_off_its_thread_and_answers_identically() -> Result<()> {
    use tokio::sync::oneshot;
    use tur::{
        InferencePipeline, TurError,
        backend::{memory::DeviceBudget, pipeline::PipelineConfig},
        server::worker::{PipelineWorker, StreamConfig, WorkerConfig, WorkerMsg},
    };
    let checkpoint = qwen_chat()?;
    let device = Device::Cpu;
    let expected =
        InferencePipeline::new(&factory(&checkpoint, &device), PipelineConfig::new(1, 256))?
            .run(&qwen_request(image("small.png")?, 3), |_| {})?
            .generated_text;
    let worker = PipelineWorker::spawn(WorkerConfig {
        factory: factory(&checkpoint, &device),
        pipeline: PipelineConfig::new(1, 256),
        budget: DeviceBudget::for_device(&device),
        request_timeout: None,
        stream: StreamConfig::default(),
    })?;
    let (result_tx, result) = oneshot::channel();
    worker
        .send(WorkerMsg {
            request: qwen_request(image("small.png")?, 3),
            token_tx: None,
            result_tx,
        })
        .await?;
    let stats = result.await.map_err(|_| TurError::WorkerUnavailable)??;
    assert_eq!(stats.generated_text, expected);
    Ok(())
}

/// Artifacts carry the tower packed, and a load from one reads an image as a conversion does.
#[test]
fn test_artifacts_carry_the_vision_tower_packed() -> Result<()> {
    use std::sync::Arc;
    use tur_models::models::{
        Conversion, Packer, WeightPlan, Weights, qwen3_5::Config, qwen3_5::write_artifact,
    };
    let source = fixture("qwen3_5");
    let config = Config::from_json(&std::fs::read_to_string(source.join("config.json"))?)?;
    let picture = image("small.png")?;
    for format in [WeightFormat::F32, WeightFormat::Q8_0] {
        let output = crate::common::Checkpoint(
            std::env::temp_dir().join(format!("tur-qwen-vision-{}", uuid::Uuid::new_v4())),
        );
        let manifest = write_artifact(
            &Conversion {
                source: &source,
                identity: "qwen-vision-fixture@0",
                output: &output.0,
                calibration: "",
            },
            &WeightPlan::uniform(&config, format)?,
            &Packer::RTN,
        )?;
        assert!(manifest.vision);
        let converted = LoadOptions {
            weights: format.into(),
            ..options(&Device::Cpu, Execution::Reference)
        };
        let prepacked = |vision| LoadOptions {
            weights: Weights::Prepacked(Arc::new(manifest.clone())),
            vision,
            ..options(&Device::Cpu, Execution::Reference)
        };
        let stored = Qwen35::load(&output.0, &prepacked(true))?;
        assert_eq!(
            rows(&stored.encode_image(&picture)?)?,
            rows(&Qwen35::load(&source, &converted)?.encode_image(&picture)?)?,
            "{format:?}"
        );
        assert_eq!(
            stored.weight_bytes(),
            Qwen35::memory_requirements(&config, &prepacked(true), 1, &[1])?.1
        );
        // The tower is optional at load, and its tensors do not get in the way.
        let text = Qwen35::load(&output.0, &prepacked(false))?;
        assert!(text.image_prompt().is_none());
    }
    Ok(())
}

fn required(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| invalid(format!("set {name}")))
}

#[test]
#[ignore = "requires Qwen3.5-0.8B, an image and its Transformers golden"]
fn test_qwen_0_8b_image_answer_matches_transformers() -> Result<()> {
    use tur::{InferencePipeline, backend::pipeline::PipelineConfig};
    let golden = candle_core::safetensors::load(required("TUR_QWEN_IMAGE_GOLDEN")?, &Device::Cpu)?;
    let picture = Image::decode(&std::fs::read(required("TUR_IMAGE")?)?)?;
    let dir = required("TUR_QWEN35_0_8B")?;
    let device = Device::new_metal(0)?;
    let f32_load = LoadOptions {
        kv: KvLayout::new(16, KvFormat::F32),
        ..options(&device, Execution::Fused)
    };
    let model = Qwen35::load(&dir, &f32_load)?;
    let [pixels, _, soft] = model.trace_image(&picture)?;
    assert!(max_absolute(&pixels, &first_frame(tensor(&golden, "pixels")?)?)? <= PIXEL_GATE);
    let error = max_relative(&rows(&soft)?, &rows(tensor(&golden, "soft_tokens")?)?);
    assert!(error < 1e-3, "soft tokens: {error}");
    drop(model);
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| invalid(e.to_string()))?;
    let expected = tokenizer
        .decode(&u32s(tensor(&golden, "greedy")?)?, true)
        .map_err(|e| invalid(e.to_string()))?;
    let factory = tur::ModelFactory {
        source: tur::ModelSource::LocalPath(dir),
        revision: "qwen-image-0.8b".into(),
        format: Some(WeightFormat::F32),
        device,
        kv: KvLayout::new(16, KvFormat::F32),
        recurrent: DType::F32,
        audio: false,
        vision: true,
        expert_cache: None,
        gguf: None,
    };
    let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::new(1, 512))?;
    let mut request = image_request(picture, ("", "Describe this image in one sentence."), 48);
    request.render.add_generation_prompt = true;
    request.render.enable_thinking = Some(false);
    request.sampling.temperature = Some(0.0);
    let stats = pipeline.run(&request, |_| {})?;
    assert_eq!(stats.prompt_tokens, tensor(&golden, "tokens")?.dim(0)?);
    assert_eq!(stats.generated_text.trim(), expected.trim());
    Ok(())
}

#[test]
#[ignore = "requires the pinned Gemma 4 E2B checkpoint, an image and its Transformers golden"]
fn test_gemma4_e2b_vision_tower_matches_transformers() -> Result<()> {
    let golden =
        candle_core::safetensors::load(required("TUR_GEMMA4_IMAGE_GOLDEN")?, &Device::Cpu)?;
    let picture = Image::decode(&std::fs::read(required("TUR_IMAGE")?)?)?;
    let model = Gemma4::load(
        &required("TUR_GEMMA4_E2B")?,
        &LoadOptions {
            weights: WeightFormat::Q8_0.into(),
            kv: KvLayout::new(16, KvFormat::F16),
            ..options(&Device::new_metal(0)?, Execution::Fused)
        },
    )?;
    let soft = model.encode_image(&picture)?.to_device(&Device::Cpu)?;
    let expected = tensor(&golden, "soft_tokens")?;
    assert_eq!(soft.dims(), expected.dims());
    let dot = (&soft * expected)?.sum(1)?;
    let norm = (soft.sqr()?.sum(1)? * expected.sqr()?.sum(1)?)?.sqrt()?;
    let cosines = (dot / norm)?.to_vec1::<f32>()?;
    let min = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(min > 0.99, "minimum cosine {min}");
    Ok(())
}
