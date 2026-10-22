#![forbid(unsafe_code)]
mod common;
use common::Checkpoint;
use tur::{InferencePipeline, Result, backend::pipeline::GenerationRequest};

#[test]
fn test_streamed_text_matches_full_decode_and_chunked_prefill() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let request = GenerationRequest::new("t10 t11 t12 t13 t14".into(), 4);
    let mut full = InferencePipeline::new(&fixture.factory(), 1, 256)?;
    let mut text = String::new();
    let stats = full.run(&request, |token| text.push_str(token))?;
    assert_eq!(stats.generated_text, "t5 t5 t5 t5");
    assert_eq!(text, stats.generated_text);
    assert_eq!(stats.prompt_tokens, 5);
    assert_eq!(stats.generated_tokens, 4);
    assert_eq!(stats.finish_reason, "length");
    assert_eq!(full.active_request_count(), 0);
    let mut chunked = InferencePipeline::new(&fixture.factory(), 1, 2)?;
    let chunked_stats = chunked.run(&request, |_| {})?;
    assert_eq!(chunked_stats.generated_text, stats.generated_text);
    assert_eq!(chunked_stats.generated_tokens, stats.generated_tokens);
    Ok(())
}

#[test]
fn test_generation_config_eos_stops_and_is_not_streamed() -> Result<()> {
    let fixture = Checkpoint::new()?;
    std::fs::write(
        fixture.0.join("generation_config.json"),
        r#"{"do_sample":false,"eos_token_id":5}"#,
    )?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), 1, 2)?;
    let mut text = String::new();
    let stats = pipeline.run(&GenerationRequest::new("t10 t11 t12".into(), 20), |t| {
        text.push_str(t)
    })?;
    assert_eq!(stats.generated_tokens, 1);
    assert_eq!(stats.finish_reason, "stop");
    assert!(stats.generated_text.is_empty());
    assert!(text.is_empty());
    assert_eq!(pipeline.active_request_count(), 0);
    Ok(())
}

#[test]
fn test_packed_requests_can_join_cancel_and_fail_independently() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), 3, 2)?;
    let first =
        pipeline.submit_request(&GenerationRequest::new("t10 t11 t12 t13 t14".into(), 3))?;
    assert!(pipeline.step_streaming()?.completed.is_empty());
    let cancelled = pipeline.submit_request(&GenerationRequest::new("t20".into(), 5))?;
    pipeline.cancel(cancelled)?;
    let invalid = pipeline.submit_request(&GenerationRequest::new("t94".into(), 3))?;
    let second = pipeline.submit_request(&GenerationRequest::new("t30".into(), 2))?;
    let mut results = std::collections::HashMap::new();
    while pipeline.active_request_count() > 0 {
        for (id, result) in pipeline.step_streaming()?.completed {
            results.insert(id, result);
        }
    }
    assert_eq!(results.len(), 3);
    assert!(
        results
            .remove(&invalid)
            .is_some_and(|result| result.is_err())
    );
    for (id, expected) in [(first, "t5 t5 t5"), (second, "t5 t5")] {
        let stats = results
            .remove(&id)
            .ok_or_else(|| tur::TurError::Other("missing completion".into()))??;
        assert_eq!(stats.generated_text, expected);
    }
    Ok(())
}

#[test]
fn test_admission_validates_sampling_context_and_limits_without_leaking_slots() -> Result<()> {
    let fixture = Checkpoint::new()?;
    assert!(InferencePipeline::new(&fixture.factory(), 0, 2).is_err());
    assert!(InferencePipeline::new(&fixture.factory(), 1, 0).is_err());
    let mut pipeline = InferencePipeline::new(&fixture.factory(), 1, 2)?;
    for limit in [0, 513, usize::MAX] {
        assert!(
            pipeline
                .submit_request(&GenerationRequest::new("t10 t11".into(), limit))
                .is_err()
        );
    }
    let mut request = GenerationRequest::new("t10".into(), 1);
    request.sampling.top_k = Some(0);
    assert!(pipeline.submit_request(&request).is_err());
    request.sampling.top_k = None;
    let id = pipeline.submit_request(&request)?;
    assert!(pipeline.submit_request(&request).is_err());
    pipeline.cancel(id)?;
    assert_eq!(pipeline.active_request_count(), 0);
    assert!(pipeline.cancel(id).is_err());
    assert_eq!(pipeline.run(&request, |_| {})?.generated_tokens, 1);
    Ok(())
}

#[test]
fn test_checkpoint_template_controls_thinking_without_prompt_heuristics() -> Result<()> {
    let fixture = Checkpoint::new()?;
    std::fs::write(
        fixture.0.join("chat_template.jinja"),
        "{{ messages[0].content }}{% if enable_thinking is not defined %} t70 t71{% elif enable_thinking %} t72{% endif %}",
    )?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), 1, 2)?;
    let mut request = GenerationRequest::new("t10".into(), 1);
    assert_eq!(pipeline.run(&request, |_| {})?.prompt_tokens, 3);
    request.render.enable_thinking = Some(true);
    assert_eq!(pipeline.run(&request, |_| {})?.prompt_tokens, 2);
    request.render.enable_thinking = Some(false);
    assert_eq!(pipeline.run(&request, |_| {})?.prompt_tokens, 1);
    Ok(())
}

#[test]
fn test_factory_rejects_legacy_architectures_and_invalid_templates() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let original = std::fs::read_to_string(fixture.0.join("config.json"))?;
    for architecture in ["qwen3", "granite"] {
        std::fs::write(
            fixture.0.join("config.json"),
            serde_json::json!({"model_type":architecture}).to_string(),
        )?;
        assert!(fixture.factory().create_model().is_err());
    }
    std::fs::write(fixture.0.join("config.json"), original)?;
    std::fs::write(fixture.0.join("chat_template.jinja"), "{% invalid %}")?;
    assert!(fixture.factory().create_model().is_err());
    Ok(())
}
