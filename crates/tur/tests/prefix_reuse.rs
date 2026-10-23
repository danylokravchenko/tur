#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![forbid(unsafe_code)]
//! Prefix reuse through the scheduled pipeline on random weights, where restoring the
//! wrong state changes generated text: warmed prefixes continue like cold runs, save
//! their prefill, stay inside their namespace, yield memory before running requests do,
//! and leave no state behind.
mod common;
use common::Checkpoint;
use std::collections::HashMap;
use tur::{
    InferencePipeline, Result, TurError,
    backend::{
        pipeline::{GenerationRequest, GenerationStats, PipelineConfig},
        prefix::PrefixConfig,
    },
};

/// `shared` tokens common to every prompt, then a tail distinct per `salt`.
fn prompt(shared: usize, tail: usize, salt: usize) -> String {
    (0..shared)
        .map(|t| 2 + (t * 17 + 5) % 90)
        .chain((0..tail).map(|t| 2 + (t * 7 + salt * 31 + 11) % 90))
        .map(|t| format!("t{t}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn request(text: String, seed: u64) -> GenerationRequest {
    let mut request = GenerationRequest::new(text, 10);
    request.seed = seed;
    request.sampling.temperature = Some(0.8);
    request.sampling.top_k = Some(40);
    request
}

/// Snapshots every 16 tokens (one page), chunks of up to 64 tokens.
fn cached(entries: usize) -> PipelineConfig {
    PipelineConfig {
        prefix: PrefixConfig {
            spacing: 16,
            entries,
        },
        ..PipelineConfig::new(4, 64)
    }
}

fn cold(fixture: &Checkpoint, requests: &[GenerationRequest]) -> Result<Vec<String>> {
    let mut pipeline = InferencePipeline::new(
        &fixture.factory(),
        PipelineConfig {
            prefix: PrefixConfig::DISABLED,
            ..PipelineConfig::new(1, 64)
        },
    )?;
    let texts = requests
        .iter()
        .map(|request| Ok(pipeline.run(request, |_| {})?.generated_text))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(pipeline.prefix_stats().lookups, 0);
    Ok(texts)
}

#[test]
fn test_warmed_prefixes_continue_like_cold_runs_and_skip_their_prefill() -> Result<()> {
    let fixture = Checkpoint::random()?;
    // A 40-token conversation, its exact repeat, and two prompts sharing 36 tokens of it.
    let requests = [
        request(prompt(40, 0, 0), 1),
        request(prompt(40, 0, 0), 1),
        request(prompt(36, 9, 1), 2),
        request(prompt(36, 3, 2), 3),
    ];
    let expected = cold(&fixture, &requests)?;
    assert_ne!(expected[2], expected[3]);
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(8))?;
    let mut stats: Vec<GenerationStats> = Vec::new();
    for request in &requests {
        stats.push(pipeline.run(request, |_| {})?);
    }
    for (index, (stats, expected)) in stats.iter().zip(&expected).enumerate() {
        assert_eq!(&stats.generated_text, expected, "request {index}");
    }
    // The first prompt left snapshots at 16 and 32 tokens; the repeat restores 32 (one
    // token must remain to produce logits), as do the 36-token shares.
    let cached_tokens: Vec<usize> = stats.iter().map(|s| s.cached_tokens).collect();
    assert_eq!(cached_tokens, [0, 32, 32, 32]);
    let prefix = pipeline.prefix_stats();
    assert_eq!(
        (prefix.hits, prefix.hit_tokens, prefix.collisions),
        (3, 96, 0)
    );
    // Snapshots hold state after every request finished; clearing them releases it all.
    assert!(pipeline.memory()?.slots.leased > 0);
    pipeline.clear_prefix_cache();
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_cache_salts_isolate_namespaces() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(8))?;
    let text = prompt(40, 0, 0);
    let mut tenant = request(text.clone(), 1);
    tenant.cache_salt = Some("tenant-a".into());
    assert_eq!(pipeline.run(&tenant, |_| {})?.cached_tokens, 0);
    assert_eq!(
        pipeline
            .run(&request(text.clone(), 1), |_| {})?
            .cached_tokens,
        0
    );
    assert_eq!(pipeline.run(&tenant, |_| {})?.cached_tokens, 32);
    let mut other = request(text, 1);
    other.cache_salt = Some("tenant-b".into());
    assert_eq!(pipeline.run(&other, |_| {})?.cached_tokens, 0);
    Ok(())
}

#[test]
fn test_snapshots_yield_memory_and_concurrent_reuse_matches_cold_runs() -> Result<()> {
    let fixture = Checkpoint::random()?;
    // Eight requests over one 48-token shared prefix with tails of 1–15 tokens.
    // Outputs of 24 tokens cross a page boundary while other requests decode.
    let requests: Vec<_> = (0..8)
        .map(|i| {
            let mut request = request(prompt(48, 1 + i * 2, i), 10 + i as u64);
            request.max_tokens = 24;
            request
        })
        .collect();
    let expected = cold(&fixture, &requests)?;
    // Seven pages and four running slots: the shared prefix holds three, each request
    // needs up to five, so concurrent requests evict snapshots and still get preempted.
    let config = cached(6);
    let usage = InferencePipeline::new(&fixture.factory(), config)?.memory()?;
    let slots = 2 * config.scheduler.max_running + config.prefix.entries + 1;
    let mut pipeline = InferencePipeline::new(
        &fixture.factory(),
        PipelineConfig {
            state_bytes: Some(slots * usage.slot_bytes + 7 * usage.page_bytes),
            ..config
        },
    )?;
    assert_eq!(pipeline.memory()?.pages.capacity, 7);
    // Warm the prefix, then submit the rest at once.
    let warm = pipeline.run(&requests[0], |_| {})?;
    assert_eq!(warm.generated_text, expected[0]);
    let mut ids = HashMap::new();
    for (index, request) in requests.iter().enumerate().skip(1) {
        ids.insert(pipeline.submit_request(request)?, index);
    }
    let mut outcomes = HashMap::new();
    for _ in 0..10_000 {
        if pipeline.active_request_count() == 0 {
            break;
        }
        for (id, result) in pipeline.step_streaming()?.completed {
            outcomes.insert(ids[&id], result?);
        }
    }
    let mut preemptions = 0;
    for (index, expected) in expected.iter().enumerate().skip(1) {
        let stats = outcomes
            .remove(&index)
            .ok_or_else(|| TurError::Other(format!("request {index} did not finish")))?;
        assert_eq!(&stats.generated_text, expected, "request {index}");
        preemptions += stats.preemptions;
    }
    let prefix = pipeline.prefix_stats();
    assert!(prefix.hits >= 4, "{prefix:?}");
    assert!(
        prefix.evicted > 0,
        "pressure never evicted a snapshot: {prefix:?}"
    );
    assert!(preemptions > 0, "the pool never forced a preemption");
    pipeline.clear_prefix_cache();
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}
