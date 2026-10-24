//! End-to-end prompt-lookup verification through admission, sampling and publication.

use crate::common::Checkpoint;
use tur::{
    Result,
    backend::{
        guidance::TopLevelGrammar,
        pipeline::{GenerationRequest, InferencePipeline, PipelineConfig},
        prefix::PrefixConfig,
        speculation::SpeculationConfig,
    },
    core::execution::invalid,
    hybrid::SamplingOptions,
    models::qwen3_5::DraftHead,
};

fn config(drafts: usize, batch: usize) -> PipelineConfig {
    PipelineConfig {
        prefix: PrefixConfig::DISABLED,
        speculation: SpeculationConfig::new(drafts),
        ..PipelineConfig::new(batch, 16)
    }
}

#[test]
fn test_lookup_verification_matches_target_sampling_and_request_limits() -> Result<()> {
    let fixture = Checkpoint::new()?;
    for sampling in [
        SamplingOptions::default(),
        SamplingOptions {
            temperature: Some(0.8),
            top_k: Some(8),
            top_p: Some(0.95),
            repetition_penalty: Some(1.2),
            presence_penalty: Some(0.2),
            frequency_penalty: Some(0.1),
        },
    ] {
        let mut plain = InferencePipeline::new(&fixture.factory(), config(0, 1))?;
        let mut speculative = InferencePipeline::new(&fixture.factory(), config(4, 1))?;
        for limit in [1, 2, 7, 31] {
            let mut request = GenerationRequest::new("t5 t5 t5 t5 t5 t5 t5 t5".into(), limit);
            request.sampling = sampling;
            request.seed = limit as u64;
            let expected = plain.run(&request, |_| {})?;
            let actual = speculative.run(&request, |_| {})?;
            assert_eq!(actual.generated_text, expected.generated_text);
            assert_eq!(actual.generated_tokens, limit);
            assert_eq!(actual.finish_reason, "length");
        }
        assert!(speculative.speculation_stats().drafted > 0);
        assert_eq!(speculative.memory()?.slots.leased, 0);
    }
    Ok(())
}

#[test]
fn test_batch_drafts_share_a_bound_and_cancel_without_leaks() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let mut cfg = config(4, 3);
    cfg.speculation.max_pass_drafts = 5;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cfg)?;
    assert_eq!(pipeline.memory_plan().draft_slots, 5);
    let request = GenerationRequest::new("t5 t5 t5 t5 t5 t5 t5 t5".into(), 31);
    let ids = (0..3)
        .map(|_| pipeline.submit_request(&request))
        .collect::<Result<Vec<_>>>()?;
    let mut saw_drafts = false;
    while pipeline.active_request_count() > 0 {
        pipeline.begin_step()?;
        let output = pipeline.finish_step()?;
        if let Some(timing) = output.timing {
            assert!(timing.drafts <= 5);
            saw_drafts |= timing.drafts > 0;
        }
        for (_, result) in output.completed {
            result?;
        }
        if saw_drafts && pipeline.active_request_count() > 0 {
            pipeline.begin_step()?;
            for &id in &ids {
                pipeline.cancel(id)?;
            }
            let output = pipeline.finish_step()?;
            assert_eq!(output.sampled_tokens.len(), 0);
            break;
        }
    }
    assert!(saw_drafts);
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_constraints_and_eos_do_not_publish_extra_draft_outputs() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let tokenizer_path = fixture.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_path)?)?;
    tokenizer["decoder"] = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    std::fs::write(tokenizer_path, tokenizer.to_string())?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config(4, 1))?;
    let mut request = GenerationRequest::new("t5 t5 t5 t5".into(), 16);
    request.grammar = Some(TopLevelGrammar::from_regex("(t2[0-9]){3}t9"));
    pipeline.run(&request, |_| {})?;
    assert_eq!(pipeline.speculation_stats().drafted, 0);
    drop(pipeline);
    let path = fixture.0.join("config.json");
    let mut json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    json["eos_token_id"] = serde_json::json!(5);
    std::fs::write(path, json.to_string())?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config(4, 1))?;
    request.grammar = None;
    let stats = pipeline.run(&request, |_| {})?;
    assert_eq!((stats.generated_tokens, stats.finish_reason), (1, "stop"));
    Ok(())
}

#[test]
fn test_prefix_snapshots_remain_at_the_accepted_boundary() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let mut cfg = config(4, 1);
    cfg.prefix = PrefixConfig {
        entries: 8,
        spacing: 16,
        state_share: None,
    };
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cfg)?;
    let request = GenerationRequest::new("t5 t5 t5 t5 t5 t5 t5 t5".into(), 40);
    let first = pipeline.run(&request, |_| {})?;
    let second = pipeline.run(&request, |_| {})?;
    assert_eq!(first.generated_text, second.generated_text);
    assert_eq!(second.generated_tokens, 40);
    assert_eq!(second.cached_tokens, second.prompt_tokens);
    pipeline.clear_prefix_cache();
    assert_eq!(pipeline.memory()?.slots.leased, 0);
    Ok(())
}

#[test]
fn test_draft_reservation_does_not_overcommit_an_explicit_state_budget() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let plain = InferencePipeline::new(&fixture.factory(), config(0, 1))?;
    let bytes = plain.memory_plan().state_pool;
    drop(plain);
    let mut cfg = config(8, 1);
    cfg.state_bytes = Some(1);
    assert!(InferencePipeline::new(&fixture.factory(), cfg).is_err());
    cfg.state_bytes = Some(bytes);
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cfg)?;
    let stats = pipeline.run(&GenerationRequest::new("t5 t5 t5 t5".into(), 24), |_| {})?;
    assert_eq!(stats.generated_tokens, 24);
    if pipeline.memory_plan().state_pool > bytes {
        return Err(invalid("state pool exceeded explicit budget"));
    }
    Ok(())
}

#[test]
fn test_narrow_recurrent_state_automatically_disables_speculation() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let mut factory = fixture.factory();
    factory.recurrent = candle_core::DType::F16;
    let mut pipeline = InferencePipeline::new(&factory, config(4, 1))?;
    assert_eq!(pipeline.memory_plan().draft_slots, 0);
    let stats = pipeline.run(&GenerationRequest::new("t5 t5 t5 t5".into(), 16), |_| {})?;
    assert_eq!(stats.draft_tokens, 0);
    assert_eq!(stats.generated_tokens, 16);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_speculative_device_sampling_preserves_seeded_penalties_and_outputs() -> Result<()> {
    let fixture = Checkpoint::fused()?;
    let mut factory = fixture.factory();
    factory.device = candle_core::Device::new_metal(0)?;
    factory.format = Some(tur::models::qwen3_5::WeightFormat::Q8_0);
    let mut plain = InferencePipeline::new(&factory, config(0, 1))?;
    let mut speculative = InferencePipeline::new(&factory, config(3, 1))?;
    let mut request = GenerationRequest::new(
        (2..90)
            .map(|t| format!("t{t}"))
            .collect::<Vec<_>>()
            .join(" "),
        40,
    );
    request.sampling = SamplingOptions {
        temperature: Some(0.8),
        top_k: Some(40),
        top_p: Some(0.95),
        repetition_penalty: Some(1.1),
        presence_penalty: Some(0.2),
        frequency_penalty: Some(0.05),
    };
    for seed in 0..3 {
        request.seed = seed;
        assert_eq!(
            plain.run(&request, |_| {})?.generated_text,
            speculative.run(&request, |_| {})?.generated_text
        );
    }
    assert!(speculative.speculation_stats().drafted > 0);
    assert!(speculative.sampling_stats().device_rows > 0);
    let usage = speculative.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_preempted_drafting_requests_rebuild_their_lookup_and_match_plain_decoding() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let request = GenerationRequest::new("t10 t11 t12 t13 t14".into(), 20);
    let expected = InferencePipeline::new(&fixture.factory(), config(0, 1))?
        .run(&request, |_| {})?
        .generated_text;
    // Two K/V pages beside every recurrent slot (two per sequence plus the drafts'): both
    // requests start, each needs a second page at position 16, and the newer is preempted.
    let mut cfg = config(4, 2);
    let usage = InferencePipeline::new(&fixture.factory(), cfg)?.memory()?;
    cfg.state_bytes = Some((usage.slots.capacity + 1) * usage.slot_bytes + 2 * usage.page_bytes);
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cfg)?;
    for _ in 0..2 {
        pipeline.submit_request(&request)?;
    }
    let (mut preemptions, mut drafted) = (0, 0);
    while pipeline.active_request_count() > 0 {
        for (_, result) in pipeline.step_streaming()?.completed {
            let stats = result?;
            assert_eq!(stats.generated_text, expected);
            preemptions += stats.preemptions;
            drafted += stats.draft_tokens;
        }
    }
    assert!(preemptions > 0, "the pool never forced a preemption");
    assert!(drafted > 0);
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_deadline_expiring_during_a_drafted_pass_fails_only_that_request() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config(4, 1))?;
    let mut request = GenerationRequest::new("t5 t5 t5 t5 t5 t5 t5 t5".into(), 400);
    request.timeout = Some(std::time::Duration::from_secs(5));
    let id = pipeline.submit_request(&request)?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut outcome = None;
    while outcome.is_none() {
        pipeline.begin_step()?;
        if pipeline.speculation_stats().passes > 0 {
            // A drafted pass has completed before; let the deadline pass under this one.
            std::thread::sleep(deadline.saturating_duration_since(std::time::Instant::now()));
        }
        for (finished, result) in pipeline.finish_step()?.completed {
            assert_eq!(finished, id);
            outcome = Some(result);
        }
    }
    assert!(matches!(
        outcome,
        Some(Err(tur::TurError::DeadlineExceeded))
    ));
    assert!(pipeline.speculation_stats().drafted > 0);
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_decodes_draft_beside_a_prefill_and_match_plain_decoding() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let repeat = GenerationRequest::new("t5 t5 t5 t5 t5 t5 t5 t5".into(), 48);
    let long = GenerationRequest::new(
        (2..90)
            .map(|t| format!("t{t}"))
            .collect::<Vec<_>>()
            .join(" "),
        8,
    );
    let mut plain = InferencePipeline::new(&fixture.factory(), config(0, 1))?;
    let expected = [
        plain.run(&repeat, |_| {})?.generated_text,
        plain.run(&long, |_| {})?.generated_text,
    ];
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config(4, 2))?;
    let first = pipeline.submit_request(&repeat)?;
    let mut second = None;
    let mut mixed_drafts = false;
    let mut texts = std::collections::HashMap::new();
    while pipeline.active_request_count() > 0 {
        // Admit the long prompt once the first request's costs are identified, so its
        // prefill chunks share passes with drafted decodes.
        if second.is_none() && pipeline.draft_costs().line(1).is_some() {
            second = Some(pipeline.submit_request(&long)?);
        }
        let output = pipeline.step_streaming()?;
        if let Some(timing) = output.timing {
            mixed_drafts |= timing.drafts > 0 && timing.tokens - timing.drafts > timing.sequences;
        }
        for (id, result) in output.completed {
            texts.insert(id, result?.generated_text);
        }
    }
    let second = second.ok_or_else(|| invalid("draft costs were never identified"))?;
    assert_eq!(texts.get(&first), Some(&expected[0]));
    assert_eq!(texts.get(&second), Some(&expected[1]));
    assert!(mixed_drafts, "no pass drafted beside a prefill");
    Ok(())
}

/// Lookup needs eight-token matches, which the random fixtures' text rarely repeats, so the
/// head drafts. The fixtures' 96 tokens are below the default draft vocabulary.
fn mtp_config(drafts: usize, batch: usize) -> PipelineConfig {
    PipelineConfig {
        prefix: PrefixConfig::DISABLED,
        speculation: SpeculationConfig {
            min_ngram: 8,
            max_ngram: 8,
            ..SpeculationConfig::with_mtp(drafts)
        },
        ..PipelineConfig::new(batch, 16)
    }
}

/// Multi-token prediction drafts verified by the target: whatever the (random) head
/// predicts, from the whole vocabulary or its leading token IDs, outputs equal ordinary
/// decoding, and every page and slot is returned.
fn assert_mtp_preserves_outputs(factory: &tur::ModelFactory) -> Result<()> {
    for vocab in [None, Some(40)] {
        let mut config = mtp_config(4, 1);
        config.speculation.mtp = Some(DraftHead { vocab });
        assert_mtp_preserves_outputs_with(factory, config)?;
    }
    Ok(())
}

fn assert_mtp_preserves_outputs_with(
    factory: &tur::ModelFactory,
    mtp: PipelineConfig,
) -> Result<()> {
    let mut plain = InferencePipeline::new(factory, config(0, 1))?;
    let mut drafted = InferencePipeline::new(factory, mtp)?;
    // Novel token sequences give prompt lookup nothing, so the head drafts.
    let prompt = (2..40)
        .map(|t| format!("t{t}"))
        .collect::<Vec<_>>()
        .join(" ");
    for sampling in [
        SamplingOptions::default(),
        SamplingOptions {
            temperature: Some(0.8),
            top_k: Some(8),
            top_p: Some(0.95),
            repetition_penalty: Some(1.2),
            ..SamplingOptions::default()
        },
    ] {
        for (seed, limit) in [(0, 1), (1, 7), (2, 33)] {
            let mut request = GenerationRequest::new(prompt.clone(), limit);
            request.sampling = sampling;
            request.seed = seed;
            assert_eq!(
                drafted.run(&request, |_| {})?.generated_text,
                plain.run(&request, |_| {})?.generated_text,
                "seed {seed}, limit {limit}"
            );
        }
    }
    let stats = drafted.speculation_stats();
    assert!(stats.mtp_passes > 0 && stats.mtp_drafted > 0, "{stats:?}");
    // Sampled requests draft too, and every verified draft is calibrated by confidence.
    assert_eq!(
        stats.confidence_drafted.iter().sum::<u64>(),
        stats.mtp_drafted,
        "{stats:?}"
    );
    let usage = drafted.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_mtp_drafts_preserve_target_sampling_and_release_state() -> Result<()> {
    let fixture = Checkpoint::random_mtp()?;
    assert_mtp_preserves_outputs(&fixture.factory())
}

#[cfg(feature = "metal")]
#[test]
fn test_fused_mtp_drafts_preserve_target_sampling() -> Result<()> {
    let fixture = Checkpoint::fused_mtp()?;
    let mut factory = fixture.factory();
    factory.device = candle_core::Device::new_metal(0)?;
    factory.format = Some(tur::models::qwen3_5::WeightFormat::Q8_0);
    assert_mtp_preserves_outputs(&factory)
}

#[test]
fn test_mtp_needs_a_head_and_disables_prefix_reuse() -> Result<()> {
    let with_head = Checkpoint::random_mtp()?;
    let without_head = Checkpoint::random()?;
    assert!(InferencePipeline::new(&without_head.factory(), mtp_config(4, 1)).is_err());
    // The head's keys and values are one more state layer of the plan.
    let plain = InferencePipeline::new(&with_head.factory(), config(4, 1))?;
    let drafted = InferencePipeline::new(&with_head.factory(), mtp_config(4, 1))?;
    // Requested prefix reuse is dropped, retaining no snapshots or their logit rows.
    let mut prefix = mtp_config(4, 1);
    prefix.prefix = PipelineConfig::new(1, 16).prefix;
    let prefix = InferencePipeline::new(&with_head.factory(), prefix)?;
    assert_eq!(
        prefix.memory_plan().resident(),
        drafted.memory_plan().resident()
    );
    assert!(drafted.memory_plan().weights > plain.memory_plan().weights);
    // A reduced draft vocabulary is a copy of that many F32 LM-head rows of 64 values; the
    // pipeline checks the plan against the loaded weights.
    let mut reduced = mtp_config(4, 1);
    reduced.speculation.mtp = Some(DraftHead { vocab: Some(40) });
    let reduced = InferencePipeline::new(&with_head.factory(), reduced)?;
    assert_eq!(
        reduced.memory_plan().weights - drafted.memory_plan().weights,
        40 * 64 * 4
    );
    let mut empty = mtp_config(4, 1);
    empty.speculation.mtp = Some(DraftHead { vocab: Some(0) });
    assert!(InferencePipeline::new(&with_head.factory(), empty).is_err());
    Ok(())
}

#[test]
fn test_preempted_mtp_requests_recompute_the_head_and_match_plain_decoding() -> Result<()> {
    let fixture = Checkpoint::random_mtp()?;
    let request = GenerationRequest::new("t10 t11 t12 t13 t14".into(), 20);
    let expected = InferencePipeline::new(&fixture.factory(), config(0, 1))?
        .run(&request, |_| {})?
        .generated_text;
    let mut cfg = mtp_config(4, 2);
    let usage = InferencePipeline::new(&fixture.factory(), cfg)?.memory()?;
    cfg.state_bytes = Some((usage.slots.capacity + 1) * usage.slot_bytes + 2 * usage.page_bytes);
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cfg)?;
    for _ in 0..2 {
        pipeline.submit_request(&request)?;
    }
    let mut preemptions = 0;
    while pipeline.active_request_count() > 0 {
        for (_, result) in pipeline.step_streaming()?.completed {
            let stats = result?;
            assert_eq!(stats.generated_text, expected);
            preemptions += stats.preemptions;
        }
    }
    assert!(preemptions > 0, "the pool never forced a preemption");
    assert!(pipeline.speculation_stats().mtp_passes > 0);
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}
