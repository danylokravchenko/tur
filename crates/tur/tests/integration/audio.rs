//! Audio input against pinned Transformers references: Gemma 4's encoder and prompts.
use crate::model_common::{max_relative, rows};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf};
use tur_core::{
    Result,
    execution::{EmbeddedInput, KvFormat, KvLayout, invalid},
};
use tur_models::{
    audio::Audio,
    models::{Execution, LoadOptions, WeightFormat, gemma4::Gemma4},
};

pub(crate) fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/audio")
        .join(name)
}

pub(crate) fn golden(name: &str) -> Result<HashMap<String, Tensor>> {
    Ok(candle_core::safetensors::load(
        fixture(name).join("golden.safetensors"),
        &Device::Cpu,
    )?)
}

pub(crate) fn tensor<'a>(golden: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    golden
        .get(name)
        .ok_or_else(|| invalid(format!("missing golden {name}")))
}

pub(crate) fn tone() -> Result<Audio> {
    Audio::from_wav(&std::fs::read(fixture("tone.wav"))?)
}

fn gemma(device: &Device, execution: Execution, audio: bool) -> Result<Gemma4> {
    Gemma4::load(
        &fixture("gemma4"),
        &LoadOptions {
            revision: "gemma4-audio-transformers-5.20.2".into(),
            weights: WeightFormat::F32.into(),
            device: device.clone(),
            execution,
            kv: KvLayout::new(4, KvFormat::F32),
            recurrent: DType::F32,
            mtp: None,
            audio,
        },
    )
}

fn conformance(device: &Device, execution: Execution) -> Result<()> {
    let golden = golden("gemma4")?;
    let model = gemma(device, execution, true)?;
    let prompt = model
        .audio_prompt()
        .ok_or_else(|| invalid("audio encoder not loaded"))?;
    let clip = tone()?;
    let soft = model.encode_audio(&clip)?;
    let expected = tensor(&golden, "soft_tokens")?;
    assert_eq!(model.audio_tokens(&clip)?, expected.dim(0)?);
    let error = max_relative(&rows(&soft)?, &rows(expected)?);
    assert!(error < 1e-4, "soft tokens: {error}");
    // The reference prompt holds audio placeholders; the model reads the slot token there.
    let reference: Vec<u32> = tensor(&golden, "tokens")?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1()?;
    let first = reference
        .iter()
        .position(|&t| t == prompt.placeholder)
        .ok_or_else(|| invalid("no audio placeholder"))?;
    assert_eq!(reference[first - 1], prompt.begin);
    assert_eq!(reference[first + soft.dim(0)?], prompt.end);
    let tokens: Vec<u32> = reference
        .iter()
        .map(|&t| {
            if t == prompt.placeholder {
                prompt.slot
            } else {
                t
            }
        })
        .collect();
    let all: Vec<usize> = (0..tokens.len()).collect();
    let embedded = [EmbeddedInput {
        tokens: first..first + soft.dim(0)?,
        rows: soft,
    }];
    let (trace, logits) = model.trace(&tokens, &all, &embedded)?;
    for (i, hidden) in trace.iter().enumerate() {
        let error = max_relative(
            &rows(hidden)?,
            &rows(tensor(&golden, &format!("layer.{i}"))?)?,
        );
        assert!(error < 1e-4, "layer {i}: {error}");
    }
    let error = max_relative(&rows(&logits)?, &rows(tensor(&golden, "logits")?)?);
    assert!(error < 1e-4, "logits: {error}");
    Ok(())
}

#[test]
fn test_gemma4_audio_reference_matches_transformers() -> Result<()> {
    conformance(&Device::Cpu, Execution::Reference)
}

#[test]
#[cfg(feature = "metal")]
fn test_gemma4_audio_metal_matches_transformers() -> Result<()> {
    conformance(&Device::new_metal(0)?, Execution::Fused)
}

#[test]
fn test_gemma4_audio_is_optional_and_bounded() -> Result<()> {
    let text = gemma(&Device::Cpu, Execution::Reference, false)?;
    assert!(text.audio_prompt().is_none());
    assert!(text.encode_audio(&tone()?).is_err());
    let model = gemma(&Device::Cpu, Execution::Reference, true)?;
    let long = Audio::from_samples(&vec![0.0; 16_000 * 31], 16_000)?;
    assert!(model.audio_tokens(&long).is_err());
    assert!(
        model
            .audio_tokens(&Audio::from_samples(&[0.0; 100], 16_000)?)
            .is_err()
    );
    assert_eq!(
        model.weight_bytes(),
        Gemma4::memory_requirements(
            model.config(),
            &LoadOptions {
                revision: String::new(),
                weights: WeightFormat::F32.into(),
                device: Device::Cpu,
                execution: Execution::Reference,
                kv: KvLayout::new(4, KvFormat::F32),
                recurrent: DType::F32,
                mtp: None,
                audio: true,
            },
            1,
            &[1],
        )?
        .1
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn required(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| invalid(format!("set {name}")))
}

/// Mean and minimum cosine similarity of matching rows.
#[cfg(feature = "metal")]
fn cosines(actual: &Tensor, expected: &Tensor) -> Result<(f64, f64)> {
    let (a, b) = (rows(actual)?, rows(expected)?);
    assert_eq!(a.len(), b.len());
    let mut values = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(&b) {
        let dot: f64 = x
            .iter()
            .zip(y)
            .map(|(p, q)| f64::from(*p) * f64::from(*q))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|p| f64::from(*p).powi(2)).sum::<f64>().sqrt();
        values.push(dot / (norm(x) * norm(y)));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Ok((mean, values.iter().copied().fold(f64::INFINITY, f64::min)))
}

#[test]
#[ignore = "requires the pinned Gemma 4 E2B checkpoint, a speech WAV and its Transformers golden"]
#[cfg(feature = "metal")]
fn test_gemma4_e2b_audio_encoder_matches_transformers() -> Result<()> {
    let golden =
        candle_core::safetensors::load(required("TUR_GEMMA4_AUDIO_GOLDEN")?, &Device::Cpu)?;
    let clip = Audio::from_wav(&std::fs::read(required("TUR_AUDIO_SPEECH")?)?)?;
    let model = Gemma4::load(
        &required("TUR_GEMMA4_E2B")?,
        &LoadOptions {
            revision: "google/gemma-4-E2B-it@3e22461f".into(),
            weights: WeightFormat::Q8_0.into(),
            device: Device::new_metal(0)?,
            execution: Execution::Fused,
            kv: KvLayout::new(16, KvFormat::F16),
            recurrent: DType::F32,
            mtp: None,
            audio: true,
        },
    )?;
    let expected = tensor(&golden, "soft_tokens")?;
    let soft = model.encode_audio(&clip)?.to_device(&Device::Cpu)?;
    // Measured: mean 0.99994, minimum 0.99965 against the F32 Transformers tower.
    let (mean, min) = cosines(&soft, expected)?;
    assert_eq!(soft.dims(), expected.dims());
    assert!(mean > 0.999 && min > 0.99, "cosine mean {mean} min {min}");
    Ok(())
}

/// The tiny audio Gemma with a word-level tokenizer (`t0`..`t95`, `<|audio|>` at 90) and a
/// template rendering typed parts the way Gemma's does.
fn gemma_chat() -> Result<crate::common::Checkpoint> {
    use tokenizers::{
        AddedToken, Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };
    let dir = std::env::temp_dir().join(format!("tur-audio-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir)?;
    let fixture = crate::common::Checkpoint(dir);
    for file in ["config.json", "model.safetensors"] {
        std::fs::copy(fixture_file("gemma4", file), fixture.0.join(file))?;
    }
    let vocab = (0..96u32)
        .map(|id| match id {
            90 => ("<|audio|>".to_owned(), id),
            _ => (format!("t{id}"), id),
        })
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".into())
        .build()
        .map_err(|e| tur::TurError::Tokenizer(e.to_string()))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.add_special_tokens(&[AddedToken::from("<|audio|>", true)]);
    tokenizer
        .save(fixture.0.join("tokenizer.json"), false)
        .map_err(|e| tur::TurError::Tokenizer(e.to_string()))?;
    let template = "{% for m in messages %}{% if m.content is string %}{{ m.content }}\
        {% else %}{% for p in m.content %}{% if p.type == 'audio' %}<|audio|>{% else %}\
        {{ p.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}";
    std::fs::write(
        fixture.0.join("tokenizer_config.json"),
        serde_json::json!({ "chat_template": template }).to_string(),
    )?;
    std::fs::write(
        fixture.0.join("generation_config.json"),
        r#"{"do_sample":false}"#,
    )?;
    Ok(fixture)
}

fn fixture_file(model: &str, file: &str) -> PathBuf {
    fixture(model).join(file)
}

fn factory(fixture: &crate::common::Checkpoint) -> tur::ModelFactory {
    tur::ModelFactory {
        source: tur::ModelSource::LocalPath(fixture.0.clone()),
        revision: "tiny-audio-chat-v1".into(),
        format: Some(WeightFormat::F32),
        device: Device::Cpu,
        kv: KvLayout::new(4, KvFormat::F32),
        recurrent: DType::F32,
        audio: true,
    }
}

/// `t2 t7 t5`, the clip, then `t9 t3`: the golden prompt once the clip is expanded.
fn audio_request(clip: Audio, max_tokens: usize) -> tur::backend::pipeline::GenerationRequest {
    use tur::backend::chat_template::{ChatMessage, Content, ContentPart, Role};
    let mut request = tur::backend::pipeline::GenerationRequest::new(String::new(), max_tokens);
    request.render.add_generation_prompt = false;
    request.messages = vec![ChatMessage {
        content: Content::Parts(vec![
            ContentPart::Text("t2 t7 t5 ".into()),
            ContentPart::Audio(clip),
            ContentPart::Text(" t9 t3".into()),
        ]),
        ..ChatMessage::new(Role::User, "")
    }];
    request
}

#[test]
fn test_gemma4_audio_chat_matches_transformers_and_isolates_clips() -> Result<()> {
    use tur::{InferencePipeline, backend::pipeline::PipelineConfig};
    let golden = golden("gemma4")?;
    let logits = tensor(&golden, "logits")?;
    let last = logits.dim(0)? - 1;
    let first = logits.get(last)?.argmax(0)?.to_scalar::<u32>()?;
    let fixture = gemma_chat()?;
    let mut pipeline = InferencePipeline::new(&factory(&fixture), PipelineConfig::new(2, 256))?;
    let stats = pipeline.run(&audio_request(tone()?, 3), |_| {})?;
    assert_eq!(stats.prompt_tokens, tensor(&golden, "tokens")?.dim(1)?);
    let words: Vec<&str> = stats.generated_text.split_whitespace().collect();
    assert_eq!(words.first().copied(), Some(format!("t{first}").as_str()));
    // A repeat restores the whole prompt; chunks that split the clip change nothing.
    let repeat = pipeline.run(&audio_request(tone()?, 3), |_| {})?;
    assert_eq!(repeat.generated_text, stats.generated_text);
    assert_eq!(repeat.cached_tokens, stats.prompt_tokens);
    let mut chunked = InferencePipeline::new(&factory(&fixture), PipelineConfig::new(1, 5))?;
    assert_eq!(
        chunked
            .run(&audio_request(tone()?, 3), |_| {})?
            .generated_text,
        stats.generated_text
    );
    // Another clip of the same length shares no state past the text before it.
    let quieter: Vec<f32> = tone()?.samples().iter().map(|s| s * 0.25).collect();
    let other = Audio::from_samples(&quieter, 16_000)?;
    let shared = pipeline.run(&audio_request(other.clone(), 3), |_| {})?;
    assert!(shared.cached_tokens <= 4, "{} cached", shared.cached_tokens);
    let mut fresh = InferencePipeline::new(&factory(&fixture), PipelineConfig::new(1, 256))?;
    let alone = fresh.run(&audio_request(other, 3), |_| {})?;
    assert_eq!(shared.generated_text, alone.generated_text);
    Ok(())
}

#[test]
fn test_audio_requests_need_an_encoder_and_one_placeholder_per_clip() -> Result<()> {
    use tur::{
        InferencePipeline, TurError,
        backend::pipeline::{GenerationRequest, PipelineConfig},
    };
    let fixture = gemma_chat()?;
    let mut pipeline = InferencePipeline::new(&factory(&fixture), PipelineConfig::new(1, 256))?;
    // A placeholder typed as text has no clip behind it.
    let typed = GenerationRequest::new("t2 <|audio|> t3".into(), 2);
    assert!(matches!(
        pipeline.submit_request(&typed),
        Err(TurError::InvalidExecution(message)) if message.contains("placeholders")
    ));
    let text = crate::common::Checkpoint::new()?;
    let mut text_model = InferencePipeline::new(&text.factory(), PipelineConfig::new(1, 256))?;
    assert!(matches!(
        text_model.submit_request(&audio_request(tone()?, 2)),
        Err(TurError::InvalidExecution(message)) if message.contains("no audio")
    ));
    assert_eq!(pipeline.active_request_count(), 0);
    Ok(())
}

fn whisper_conformance(device: &Device, format: WeightFormat, gate: f32) -> Result<()> {
    use tur_models::models::whisper::Whisper;
    let golden = golden("whisper")?;
    let model = Whisper::load(&fixture("whisper"), format, device)?;
    assert_eq!(
        model.weight_bytes(),
        Whisper::planned_bytes(model.config(), format)?
    );
    let encoded = model.encode(&model.features(tone()?.samples())?)?;
    let error = max_relative(&rows(&encoded)?, &rows(tensor(&golden, "encoded")?)?);
    assert!(error < gate, "encoder: {error}");
    let tokens: Vec<u32> = tensor(&golden, "tokens")?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1()?;
    let logits = model.decoder(&encoded)?.step(&tokens)?;
    let error = max_relative(&rows(&logits)?, &rows(tensor(&golden, "logits")?)?);
    assert!(error < gate, "decoder: {error}");
    // Greedy continuation through the self-attention cache, one token per step.
    let expected: Vec<u32> = tensor(&golden, "greedy")?.to_dtype(DType::U32)?.to_vec1()?;
    let mut decoder = model.decoder(&encoded)?;
    let mut greedy = expected[..2].to_vec();
    let mut logits = decoder.step(&greedy)?;
    while greedy.len() < expected.len() {
        let last = logits.dim(0)? - 1;
        let token = logits.get(last)?.argmax(0)?.to_scalar::<u32>()?;
        greedy.push(token);
        logits = decoder.step(&[token])?;
    }
    assert_eq!(greedy, expected);
    Ok(())
}

#[test]
fn test_whisper_reference_matches_transformers() -> Result<()> {
    whisper_conformance(&Device::Cpu, WeightFormat::F32, 1e-4)
}

#[test]
#[cfg(feature = "metal")]
fn test_whisper_metal_matches_transformers() -> Result<()> {
    whisper_conformance(&Device::new_metal(0)?, WeightFormat::F32, 1e-4)
}

#[test]
#[ignore = "requires the pinned Whisper large-v3-turbo checkpoint, a speech WAV and its Transformers golden"]
#[cfg(feature = "metal")]
fn test_whisper_turbo_transcription_matches_transformers() -> Result<()> {
    use tur_models::models::whisper::{TranscribeOptions, Transcriber};
    let golden_path = required("TUR_WHISPER_GOLDEN")?;
    let golden = candle_core::safetensors::load(&golden_path, &Device::Cpu)?;
    let expected: Vec<u32> = tensor(&golden, "greedy")?.to_dtype(DType::U32)?.to_vec1()?;
    let clip = Audio::from_wav(&std::fs::read(required("TUR_AUDIO_SPEECH")?)?)?;
    let dir = required("TUR_WHISPER")?;
    let options = TranscribeOptions {
        language: Some("en".into()),
        ..Default::default()
    };
    for format in [WeightFormat::F32, WeightFormat::Q8_0] {
        let transcriber = Transcriber::load(&dir, format, &Device::new_metal(0)?)?;
        let result = transcriber.transcribe(&clip, &options)?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| tur::TurError::Tokenizer(e.to_string()))?;
        let reference = tokenizer
            .decode(&expected, true)
            .map_err(|e| tur::TurError::Tokenizer(e.to_string()))?;
        assert_eq!(result.text, reference.trim(), "{format:?}");
        assert_eq!(result.language.as_deref(), Some("en"));
    }
    Ok(())
}

mod http {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt as _;
    use tokio::sync::mpsc;
    use tower::ServiceExt as _;
    use tur::{
        TurError,
        backend::pipeline::GenerationStats,
        models::whisper::{Task, Transcription},
        server::{
            AppState, build_router,
            transcription::{TranscriptionJob, TranscriptionWorker},
            worker::{PipelineWorker, WorkerMsg},
        },
    };

    fn base64(bytes: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let word = chunk
                .iter()
                .enumerate()
                .fold(0u32, |w, (i, &b)| w | u32::from(b) << (16 - 8 * i));
            for i in 0..4 {
                out.push(if i <= chunk.len() {
                    char::from(TABLE[(word >> (18 - 6 * i) & 63) as usize])
                } else {
                    '='
                });
            }
        }
        out
    }

    /// Answers each chat request with the durations of its clips, in milliseconds.
    fn chat_worker() -> PipelineWorker {
        let (tx, mut rx) = mpsc::channel::<WorkerMsg>(8);
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let clips: Vec<String> = msg
                    .request
                    .messages
                    .iter()
                    .flat_map(|m| m.content.audio())
                    .map(|clip| clip.duration().as_millis().to_string())
                    .collect();
                drop(msg.result_tx.send(Ok(GenerationStats {
                    finish_reason: "stop",
                    generated_tokens: 1,
                    prompt_tokens: 0,
                    cached_tokens: 0,
                    elapsed: std::time::Duration::ZERO,
                    first_token: std::time::Duration::ZERO,
                    preemptions: 0,
                    draft_tokens: 0,
                    accepted_draft_tokens: 0,
                    tool_calls: vec![],
                    generated_text: clips.join(","),
                })));
            }
        });
        PipelineWorker::from_sender(tx)
    }

    /// Answers each clip with its task, language and duration.
    fn whisper_worker() -> TranscriptionWorker {
        let (tx, mut rx) = mpsc::channel::<TranscriptionJob>(4);
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let text = format!(
                    "{:?} {} {}",
                    job.options.task,
                    job.options.language.as_deref().unwrap_or("auto"),
                    job.audio.duration().as_millis()
                );
                drop(job.reply.send(Ok(Transcription {
                    text,
                    language: None,
                    tokens: 1,
                    windows: 1,
                })));
            }
        });
        TranscriptionWorker::from_sender(tx)
    }

    fn router() -> axum::Router {
        build_router(AppState {
            workers: HashMap::from([("gemma".to_owned(), chat_worker())]),
            default_model: "gemma".into(),
            transcribers: HashMap::from([("whisper-1".to_owned(), whisper_worker())]),
            default_transcriber: Some("whisper-1".into()),
        })
    }

    async fn send(request: Request<Body>) -> Result<(u16, String)> {
        let response = router()
            .oneshot(request)
            .await
            .map_err(|e| TurError::Other(e.to_string()))?;
        let status = response.status().as_u16();
        let bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|e| TurError::Other(e.to_string()))?
            .to_bytes();
        Ok((status, String::from_utf8_lossy(&bytes).into_owned()))
    }

    fn chat(part: &serde_json::Value) -> Result<Request<Body>> {
        let body = serde_json::json!({"messages": [{"role": "user", "content": [
            part, {"type": "text", "text": "What was said?"}
        ]}]});
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .map_err(|e| TurError::Other(e.to_string()))
    }

    fn form(fields: &[(&str, &str)], file: Option<&[u8]>) -> Result<Request<Body>> {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend(
                format!(
                    "--b\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .bytes(),
            );
        }
        if let Some(file) = file {
            body.extend(
                b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
Content-Type: audio/wav\r\n\r\n",
            );
            body.extend_from_slice(file);
            body.extend(b"\r\n");
        }
        body.extend(b"--b--\r\n");
        Request::post("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=b")
            .body(Body::from(body))
            .map_err(|e| TurError::Other(e.to_string()))
    }

    #[tokio::test]
    async fn test_chat_audio_parts_reach_the_worker_decoded() -> Result<()> {
        let wav = std::fs::read(fixture("tone.wav"))?;
        let (status, body) = send(chat(&serde_json::json!({
            "type": "input_audio", "input_audio": {"data": base64(&wav), "format": "wav"}
        }))?)
        .await?;
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("\"1234\""), "{body}");
        let url = format!("data:audio/wav;base64,{}", base64(&wav));
        let (status, body) = send(chat(&serde_json::json!({
            "type": "audio_url", "audio_url": {"url": url}
        }))?)
        .await?;
        assert_eq!(status, 200, "{body}");
        for bad in [
            serde_json::json!({"type": "input_audio", "input_audio": {"data": base64(&wav), "format": "mp3"}}),
            serde_json::json!({"type": "input_audio", "input_audio": {"data": "not base64!"}}),
            serde_json::json!({"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}),
        ] {
            assert_eq!(send(chat(&bad)?).await?.0, 400, "{bad}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_transcription_form_fields_formats_and_errors() -> Result<()> {
        let wav = std::fs::read(fixture("tone.wav"))?;
        let (status, body) = send(form(&[("model", "whisper-1")], Some(&wav))?).await?;
        assert_eq!(
            (status, body.as_str()),
            (200, r#"{"text":"Transcribe auto 1234"}"#)
        );
        let (status, body) = send(form(
            &[
                ("language", "uk"),
                ("response_format", "text"),
                ("temperature", "0"),
            ],
            Some(&wav),
        )?)
        .await?;
        assert_eq!((status, body.as_str()), (200, "Transcribe uk 1234"));
        let mut translation = form(&[], Some(&wav))?;
        *translation.uri_mut() = "/v1/audio/translations"
            .parse()
            .map_err(|_| TurError::Other("uri".into()))?;
        assert!(
            send(translation)
                .await?
                .1
                .contains(&format!("{:?}", Task::Translate))
        );
        for (fields, file, status) in [
            (vec![], None, 400),
            (vec![("temperature", "0.4")], Some(&wav[..]), 400),
            (vec![("response_format", "srt")], Some(&wav[..]), 400),
            (vec![("prompt", "earlier words")], Some(&wav[..]), 400),
            (vec![], Some(&b"OggS"[..]), 400),
            (vec![("model", "whisper-2")], Some(&wav[..]), 404),
        ] {
            assert_eq!(send(form(&fields, file)?).await?.0, status, "{fields:?}");
        }
        let models = send(
            Request::get("/v1/models")
                .body(Body::empty())
                .map_err(|e| TurError::Other(e.to_string()))?,
        )
        .await?;
        assert!(models.1.contains("whisper-1") && models.1.contains("gemma"));
        Ok(())
    }
}
