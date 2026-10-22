#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![forbid(unsafe_code)]
mod common;
use common::Checkpoint;
use tur::{
    InferencePipeline, Result, TurError,
    backend::pipeline::{GenerationRequest, PipelineConfig},
};

#[test]
fn test_streamed_text_matches_full_decode_and_chunked_prefill() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let request = GenerationRequest::new("t10 t11 t12 t13 t14".into(), 4);
    let mut full = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 256))?;
    let mut text = String::new();
    let stats = full.run(&request, |token| text.push_str(token))?;
    assert_eq!(stats.generated_text, "t5 t5 t5 t5");
    assert_eq!(text, stats.generated_text);
    assert_eq!(stats.prompt_tokens, 5);
    assert_eq!(stats.generated_tokens, 4);
    assert_eq!(stats.finish_reason, "length");
    assert_eq!(full.active_request_count(), 0);
    let mut chunked = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 2))?;
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
    let mut pipeline = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 2))?;
    let mut text = String::new();
    let stats = pipeline.run(&GenerationRequest::new("t10 t11 t12".into(), 20), |t| {
        text.push_str(t);
    })?;
    assert_eq!(stats.generated_tokens, 1);
    assert_eq!(stats.finish_reason, "stop");
    assert_eq!(stats.generated_text, "");
    assert_eq!(text, "");
    assert_eq!(pipeline.active_request_count(), 0);
    Ok(())
}

#[test]
fn test_packed_requests_can_join_cancel_and_fail_independently() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(3, 2))?;
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
            .ok_or_else(|| TurError::Other("missing completion".into()))??;
        assert_eq!(stats.generated_text, expected);
    }
    Ok(())
}

#[test]
fn test_admission_validates_sampling_context_and_limits_without_leaking_slots() -> Result<()> {
    let fixture = Checkpoint::new()?;
    assert!(InferencePipeline::new(&fixture.factory(), PipelineConfig::new(0, 2)).is_err());
    assert!(InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 0)).is_err());
    let mut config = PipelineConfig::new(1, 2);
    config.scheduler.max_waiting = 1;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config)?;
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
    // The waiting queue holds one request.
    assert!(pipeline.submit_request(&request).is_err());
    assert_eq!(pipeline.waiting_request_count(), 1);
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
    let mut pipeline = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 2))?;
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

/// A pipeline whose state pool holds `pages` K/V pages beside its recurrent slots.
fn budgeted(fixture: &Checkpoint, batch: usize, pages: usize) -> Result<InferencePipeline> {
    // Unit sizes come from an unconstrained pipeline over the same checkpoint.
    let usage =
        InferencePipeline::new(&fixture.factory(), PipelineConfig::new(batch, 2))?.memory()?;
    let budget = (2 * batch + 1) * usage.slot_bytes + pages * usage.page_bytes;
    let mut pipeline = InferencePipeline::new(
        &fixture.factory(),
        PipelineConfig {
            state_bytes: Some(budget),
            ..PipelineConfig::new(batch, 2)
        },
    )?;
    assert_eq!(pipeline.memory()?.pages.capacity, pages);
    assert!(pipeline.memory_plan().state_pool <= budget);
    Ok(pipeline)
}

fn drain(
    pipeline: &mut InferencePipeline,
) -> Result<std::collections::HashMap<uuid::Uuid, Result<String>>> {
    let mut results = std::collections::HashMap::new();
    while pipeline.active_request_count() > 0 {
        for (id, result) in pipeline.step_streaming()?.completed {
            results.insert(id, result.map(|stats| stats.generated_text));
        }
    }
    Ok(results)
}

#[test]
fn test_requests_beyond_the_state_pool_are_rejected_at_admission() -> Result<()> {
    let fixture = Checkpoint::new()?;
    // One 16-token page: 5 prompt tokens plus 12 generated fit, plus 13 do not.
    let mut pipeline = budgeted(&fixture, 2, 1)?;
    let planned = pipeline.memory_plan().state_pool;
    assert_eq!(planned, pipeline.memory()?.resident_bytes());
    assert!(matches!(
        pipeline.submit_request(&GenerationRequest::new("t10 t11 t12 t13 t14".into(), 13)),
        Err(TurError::MemoryAllocation(_))
    ));
    let id = pipeline.submit_request(&GenerationRequest::new("t10 t11 t12 t13 t14".into(), 12))?;
    let mut results = drain(&mut pipeline)?;
    assert_eq!(
        results.remove(&id).transpose()?.as_deref(),
        Some(["t5"; 12].join(" ").as_str())
    );
    Ok(())
}

#[test]
fn test_requests_wait_for_pages_and_complete_in_turn() -> Result<()> {
    let fixture = Checkpoint::new()?;
    // Each request fits the single page alone; the second waits for the first.
    let mut pipeline = budgeted(&fixture, 2, 1)?;
    let request = GenerationRequest::new("t10 t11 t12 t13 t14".into(), 8);
    let first = pipeline.submit_request(&request)?;
    let second = pipeline.submit_request(&request)?;
    let mut results = drain(&mut pipeline)?;
    for id in [first, second] {
        assert_eq!(
            results.remove(&id).transpose()?.as_deref(),
            Some(["t5"; 8].join(" ").as_str())
        );
    }
    assert_eq!(pipeline.memory()?.pages.leased, 0);
    Ok(())
}

#[test]
fn test_exhausted_pool_preempts_the_newest_request_and_both_finish() -> Result<()> {
    let fixture = Checkpoint::new()?;
    // Two pages; both requests start, and each needs a second page at position 16. The
    // newer one is preempted, recomputes after the older finishes, and completes too.
    let mut pipeline = budgeted(&fixture, 2, 2)?;
    let request = GenerationRequest::new("t10 t11 t12 t13 t14".into(), 20);
    let older = pipeline.submit_request(&request)?;
    let newer = pipeline.submit_request(&request)?;
    let mut preempted = std::collections::HashMap::new();
    while pipeline.active_request_count() > 0 {
        for (id, result) in pipeline.step_streaming()?.completed {
            let stats = result?;
            assert_eq!(stats.generated_text, ["t5"; 20].join(" "));
            preempted.insert(id, stats.preemptions);
        }
    }
    assert_eq!(preempted.get(&older), Some(&0));
    assert!(preempted.get(&newer).is_some_and(|&n| n > 0));
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}
