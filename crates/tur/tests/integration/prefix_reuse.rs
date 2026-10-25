//! Prefix reuse through the scheduled pipeline on random weights, where restoring the
//! wrong state changes generated text: warmed prefixes continue like cold runs, save
//! their prefill, stay inside their namespace, yield memory before running requests do,
//! and leave no state behind. Requests reaching a cold shared prefix together wait for
//! the oldest to compute it, and restorable requests pass older cold ones only when asked.
use crate::common::Checkpoint;
use std::{collections::HashMap, time::Duration};
use tur::{
    InferencePipeline, Result, TurError,
    backend::{
        chat_template::{ChatMessage, Role},
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
            state_share: None,
            restorable_first: false,
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
    let mut passes = Vec::new();
    for request in &requests {
        let before = pipeline.scheduler().iteration();
        stats.push(pipeline.run(request, |_| {})?);
        passes.push(pipeline.scheduler().iteration() - before);
    }
    for (index, (stats, expected)) in stats.iter().zip(&expected).enumerate() {
        assert_eq!(&stats.generated_text, expected, "request {index}");
    }
    // The first prompt left snapshots at 16 and 32 tokens and at its end, with its last
    // logits. The repeat restores all 40 tokens and samples its first output from them,
    // so its outputs take one pass fewer than their count, against three aligned prefill
    // passes (16 + 16 + 8 tokens) plus the remaining decodes for the cold run; the
    // 36-token shares restore 32.
    let cached_tokens: Vec<usize> = stats.iter().map(|s| s.cached_tokens).collect();
    assert_eq!(cached_tokens, [0, 40, 32, 32]);
    assert_eq!(
        passes[..2],
        [
            stats[0].generated_tokens as u64 + 2,
            stats[1].generated_tokens as u64 - 1
        ]
    );
    let prefix = pipeline.prefix_stats();
    assert_eq!(
        (prefix.hits, prefix.hit_tokens, prefix.collisions),
        (3, 104, 0)
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
    // Room for both namespaces' snapshots, so none is evicted.
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(16))?;
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
    assert_eq!(pipeline.run(&tenant, |_| {})?.cached_tokens, 40);
    let mut other = request(text, 1);
    other.cache_salt = Some("tenant-b".into());
    assert_eq!(pipeline.run(&other, |_| {})?.cached_tokens, 0);
    Ok(())
}

#[test]
fn test_a_next_turn_resumes_after_the_previous_reply() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(8))?;
    // The fixture's template joins message contents with spaces, and its word-level
    // tokenizer re-encodes a reply exactly, so the next turn's prompt starts with the
    // first turn's prompt and reply.
    let first = request(prompt(20, 0, 0), 1);
    let reply = pipeline.run(&first, |_| {})?;
    let prompt_tokens = reply.prompt_tokens;
    let mut next = first;
    next.messages.push(ChatMessage::new(
        Role::Assistant,
        reply.generated_text.clone(),
    ));
    next.messages
        .push(ChatMessage::new(Role::User, prompt(0, 7, 3)));
    next.seed = 2;
    let expected = cold(&fixture, std::slice::from_ref(&next))?;
    let stats = pipeline.run(&next, |_| {})?;
    // The reply's last token was sampled but never consumed.
    assert_eq!(
        stats.cached_tokens,
        prompt_tokens + reply.generated_tokens - 1
    );
    assert_eq!(stats.generated_text, expected[0]);
    Ok(())
}

#[test]
#[cfg(feature = "metal")]
fn test_template_divergence_restores_the_last_shared_token() -> Result<()> {
    let fixture = Checkpoint::fused()?;
    // Keep the pinned template's history rules, spelling its delimiters with tokens
    // from the tiny fixture's vocabulary so distinct delimiters cannot collapse to UNK.
    let template = include_str!("../fixtures/checkpoints/small-chat_template.jinja")
        .replace("<|im_start|>", " t90 ")
        .replace("<|im_end|>", " t91 ")
        .replace("<think>", " t92 ")
        .replace("</think>", " t93 ");
    std::fs::write(fixture.0.join("chat_template.jinja"), template)?;
    let mut factory = fixture.factory();
    factory.device = candle_core::Device::new_metal(0)?;
    let mut pipeline = InferencePipeline::new(&factory, cached(16))?;
    let single = request(prompt(22, 0, 0), 1);
    let stats = pipeline.run(&single, |_| {})?;
    // A single-turn request keeps its two ordinary prefill passes, without a third
    // pass to save a speculative future turn.
    assert_eq!(
        pipeline.scheduler().iteration(),
        stats.generated_tokens as u64 + 1
    );
    pipeline.clear_prefix_cache();
    let mut first = request(String::new(), 1);
    first.messages = vec![
        ChatMessage::new(Role::User, "t2"),
        ChatMessage::new(Role::Assistant, "t3"),
        ChatMessage::new(Role::User, prompt(14, 0, 0)),
    ];
    let reply = pipeline.run(&first, |_| {})?;
    // The earlier exchange adds eight tokens, the last user wrapper and assistant
    // header add five, and the empty think block adds two. A new user drops that block.
    assert_eq!(reply.prompt_tokens, 29);
    let repeated = pipeline.run(&first, |_| {})?;
    assert_eq!(repeated.cached_tokens, 29);
    assert_eq!(repeated.generated_text, reply.generated_text);
    let mut next = request(String::new(), 2);
    next.messages.clone_from(&first.messages);
    next.messages
        .push(ChatMessage::new(Role::Assistant, reply.generated_text));
    next.messages
        .push(ChatMessage::new(Role::User, prompt(0, 7, 3)));
    let expected = cold(&fixture, std::slice::from_ref(&next))?;
    let stats = pipeline.run(&next, |_| {})?;
    assert_eq!(stats.generated_text, expected[0]);
    assert_eq!(stats.cached_tokens, 27);
    pipeline.clear_prefix_cache();
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));

    // A concurrent follower must wait for the same unaligned boundary instead of
    // restoring the shallower aligned snapshot before the producer reaches it.
    let outcomes = burst(&mut pipeline, &[first, next], |_, _, _| Ok(()))?;
    let follower = outcomes
        .get(&1)
        .ok_or_else(|| TurError::Other("next turn did not finish".into()))?
        .as_ref()
        .map_err(|error| TurError::Other(error.to_string()))?;
    assert_eq!(follower.generated_text, expected[0]);
    assert_eq!(follower.cached_tokens, 27);
    Ok(())
}

#[test]
fn test_templates_rejecting_empty_turns_still_accept_valid_requests() -> Result<()> {
    let fixture = Checkpoint::random()?;
    std::fs::write(
        fixture.0.join("chat_template.jinja"),
        "{% for m in messages %}{% if not m.content %}{{ raise_exception('empty turn') }}\
         {% endif %}{{ m.content }} {% endfor %}\
         {% if add_generation_prompt %}t90 t91{% endif %}",
    )?;
    let mut request = request(String::new(), 1);
    request.messages = vec![
        ChatMessage::new(Role::User, "t2"),
        ChatMessage::new(Role::Assistant, "t3"),
        ChatMessage::new(Role::User, prompt(20, 0, 0)),
    ];
    let expected = cold(&fixture, std::slice::from_ref(&request))?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(8))?;
    let stats = pipeline.run(&request, |_| {})?;
    assert_eq!(stats.generated_text, expected[0]);
    assert_eq!(stats.cached_tokens, 0);
    let repeated = pipeline.run(&request, |_| {})?;
    assert_eq!(repeated.generated_text, expected[0]);
    assert_eq!(repeated.cached_tokens, 24);
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

/// Submit every request at once and step until all finish; `before_step(pass, pipeline,
/// ids)` runs before each pass. Returns each request's outcome by index.
fn burst(
    pipeline: &mut InferencePipeline,
    requests: &[GenerationRequest],
    mut before_step: impl FnMut(
        usize,
        &mut InferencePipeline,
        &HashMap<uuid::Uuid, usize>,
    ) -> Result<()>,
) -> Result<HashMap<usize, Result<GenerationStats>>> {
    let mut ids = HashMap::new();
    for (index, request) in requests.iter().enumerate() {
        ids.insert(pipeline.submit_request(request)?, index);
    }
    let mut outcomes = HashMap::new();
    for pass in 0..10_000 {
        if pipeline.active_request_count() == 0 {
            break;
        }
        before_step(pass, pipeline, &ids)?;
        for (id, result) in pipeline.step_streaming()?.completed {
            outcomes.insert(ids[&id], result);
        }
    }
    Ok(outcomes)
}

/// Four prompts sharing 48 tokens, with distinct tails.
fn shared_burst() -> Vec<GenerationRequest> {
    (0..4)
        .map(|i| request(prompt(48, 3 + i, i), 20 + i as u64))
        .collect()
}

#[test]
fn test_a_cold_shared_prefix_is_computed_once_and_waiters_restore_it() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let requests = shared_burst();
    let expected = cold(&fixture, &requests)?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(16))?;
    let mut waiting = Vec::new();
    let outcomes = burst(&mut pipeline, &requests, |pass, pipeline, _| {
        if pass == 1 {
            waiting.push(pipeline.waiting_request_count());
        }
        Ok(())
    })?;
    // After the first pass only the oldest request runs; the others wait for its
    // snapshot at the deepest aligned boundary they share, then restore it.
    assert_eq!(waiting, [3]);
    let mut cached = Vec::new();
    for (index, expected) in expected.iter().enumerate() {
        let stats = outcomes
            .get(&index)
            .ok_or_else(|| TurError::Other(format!("request {index} did not finish")))?
            .as_ref()
            .map_err(|error| TurError::Other(error.to_string()))?;
        assert_eq!(&stats.generated_text, expected, "request {index}");
        cached.push(stats.cached_tokens);
    }
    assert_eq!(cached[0], 0);
    assert!(cached[1..].iter().all(|&tokens| tokens >= 48), "{cached:?}");
    assert_eq!(pipeline.prefix_stats().hits, 3);
    pipeline.clear_prefix_cache();
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_waiters_proceed_when_their_producer_is_cancelled() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let requests = shared_burst();
    let expected = cold(&fixture, &requests)?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(16))?;
    let outcomes = burst(&mut pipeline, &requests, |pass, pipeline, ids| {
        if pass == 1
            && let Some((&producer, _)) = ids.iter().find(|(_, index)| **index == 0)
        {
            pipeline.cancel(producer)?;
        }
        Ok(())
    })?;
    assert!(!outcomes.contains_key(&0));
    for (index, expected) in expected.iter().enumerate().skip(1) {
        let stats = outcomes
            .get(&index)
            .ok_or_else(|| TurError::Other(format!("request {index} did not finish")))?
            .as_ref()
            .map_err(|error| TurError::Other(error.to_string()))?;
        assert_eq!(&stats.generated_text, expected, "request {index}");
    }
    pipeline.clear_prefix_cache();
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_aged_requests_and_other_namespaces_never_wait() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let mut config = cached(16);
    config.scheduler.max_queue_wait = Duration::ZERO;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config)?;
    let outcomes = burst(&mut pipeline, &shared_burst(), |_, _, _| Ok(()))?;
    assert_eq!(outcomes.len(), 4);
    for outcome in outcomes.values() {
        let stats = outcome
            .as_ref()
            .map_err(|error| TurError::Other(error.to_string()))?;
        assert_eq!(stats.cached_tokens, 0);
    }
    let mut pipeline = InferencePipeline::new(&fixture.factory(), cached(16))?;
    let salted: Vec<GenerationRequest> = shared_burst()
        .into_iter()
        .enumerate()
        .map(|(index, mut request)| {
            request.cache_salt = Some(format!("tenant-{index}"));
            request
        })
        .collect();
    for outcome in burst(&mut pipeline, &salted, |_, _, _| Ok(()))?.values() {
        let stats = outcome
            .as_ref()
            .map_err(|error| TurError::Other(error.to_string()))?;
        assert_eq!(stats.cached_tokens, 0);
    }
    Ok(())
}

#[test]
fn test_restorable_requests_pass_older_cold_ones_only_when_enabled() -> Result<()> {
    let fixture = Checkpoint::random()?;
    // Three cold prompts sharing no aligned prefix with the warmed one, then a request that
    // restores 48 warmed tokens.
    let requests = [
        request(prompt(0, 40, 0), 40),
        request(prompt(0, 40, 1), 41),
        request(prompt(0, 40, 2), 42),
        request(prompt(48, 3, 3), 43),
    ];
    let expected = cold(&fixture, &requests)?;
    for restorable_first in [false, true] {
        let mut config = PipelineConfig {
            prefix: PrefixConfig {
                spacing: 16,
                entries: 16,
                state_share: None,
                restorable_first,
            },
            ..PipelineConfig::new(2, 64)
        };
        config.scheduler.max_queue_wait = Duration::from_secs(60);
        let mut pipeline = InferencePipeline::new(&fixture.factory(), config)?;
        pipeline.run(&request(prompt(48, 5, 9), 44), |_| {})?;
        let mut ids = HashMap::new();
        for (index, request) in requests.iter().enumerate() {
            ids.insert(pipeline.submit_request(request)?, index);
        }
        // Pass at which each request finished.
        let mut finished = HashMap::new();
        for pass in 0..10_000 {
            if pipeline.active_request_count() == 0 {
                break;
            }
            for (id, result) in pipeline.step_streaming()?.completed {
                let index = ids[&id];
                let stats = result?;
                assert_eq!(&stats.generated_text, &expected[index], "request {index}");
                assert_eq!(stats.cached_tokens >= 48, index == 3, "request {index}");
                finished.insert(index, pass);
            }
        }
        assert_eq!(finished.len(), requests.len());
        // Two slots: by default the restorable request waits for a cold one to finish;
        // restorable first, it takes the second slot ahead of the second cold request.
        let (warm, second) = (finished.get(&3), finished.get(&1));
        assert_eq!(warm < second, restorable_first, "{finished:?}");
    }
    Ok(())
}
