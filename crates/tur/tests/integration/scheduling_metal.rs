#![cfg(feature = "metal")]
//! Packed serving pressure and device-budget arbitration on an actual Metal device.
use crate::common;

use candle_core::{DType, Device};
use std::collections::HashMap;
use tur::{
    InferencePipeline, Result, TurError,
    backend::{
        memory::DeviceBudget,
        pipeline::{GenerationRequest, PipelineConfig},
        prefix::PrefixConfig,
    },
};

#[test]
fn test_f16_packed_recompute_and_loaded_model_budget_on_metal() -> Result<()> {
    let checkpoint = common::Checkpoint::fused()?;
    let mut factory = checkpoint.factory();
    factory.device = Device::new_metal(0)?;
    factory.recurrent = DType::F16;
    let mut config = PipelineConfig::new(3, 8);
    config.prefix = PrefixConfig::DISABLED;
    config.scheduler.max_batch_tokens = 24;
    config.scheduler.min_prefill_tokens = 2;
    let probe = InferencePipeline::new(&factory, config.clone())?;
    let usage = {
        let mut probe = probe;
        probe.memory()?
    };
    config.state_bytes = Some(7 * usage.slot_bytes + 6 * usage.page_bytes);
    let requests: Vec<_> = (0..3)
        .map(|i| {
            let prompt = (0..33 + i)
                .map(|t| format!("t{}", 2 + (t * 7 + i) % 80))
                .collect::<Vec<_>>()
                .join(" ");
            GenerationRequest::new(prompt, 24)
        })
        .collect();
    let mut solo = InferencePipeline::new(&factory, config.clone())?;
    let expected = requests
        .iter()
        .map(|request| Ok(solo.run(request, |_| {})?.generated_text))
        .collect::<Result<Vec<_>>>()?;
    // The probe's overhead uses the physical working set; the shared test budget
    // recomputes its own overhead from this much smaller limit.
    let bytes = solo
        .memory_plan()
        .resident()
        .saturating_sub(solo.memory_plan().overhead);
    drop(solo);
    let budget = DeviceBudget::new(Some(bytes * 3));
    let mut packed = InferencePipeline::with_budget(&factory, config.clone(), &budget)?;
    let other = InferencePipeline::with_budget(&factory, config.clone(), &budget)?;
    assert_eq!(
        budget.leased(),
        packed.memory_plan().resident() + other.memory_plan().resident()
    );
    let rejected = InferencePipeline::with_budget(&factory, config, &budget);
    assert!(
        matches!(&rejected, Err(TurError::MemoryAllocation(_))),
        "third model: {:?}; base {bytes}, leased {}, limit {:?}, first {}, second {}",
        rejected.as_ref().err(),
        budget.leased(),
        budget.limit(),
        packed.memory_plan().resident(),
        other.memory_plan().resident()
    );
    let ids = requests
        .iter()
        .enumerate()
        .map(|(i, request)| Ok((packed.submit_request(request)?, i)))
        .collect::<Result<HashMap<_, _>>>()?;
    let (mut finished, mut preemptions) = (0, 0);
    for _ in 0..1000 {
        if packed.active_request_count() == 0 {
            break;
        }
        for (id, result) in packed.step_streaming()?.completed {
            let stats = result?;
            let index = ids
                .get(&id)
                .ok_or_else(|| TurError::Other("unknown completion".into()))?;
            assert_eq!(stats.generated_text, expected[*index]);
            preemptions += stats.preemptions;
            finished += 1;
        }
    }
    assert_eq!(finished, requests.len());
    assert!(preemptions > 0);
    let usage = packed.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    drop(packed);
    drop(other);
    assert_eq!(budget.leased(), 0);
    Ok(())
}

#[test]
fn test_device_sampling_with_penalties_is_independent_of_batch_order_on_metal() -> Result<()> {
    let checkpoint = common::Checkpoint::fused()?;
    let mut factory = checkpoint.factory();
    factory.device = Device::new_metal(0)?;
    let requests: Vec<_> = (0..6usize)
        .map(|i| {
            let prompt = (0..5 + i * 4)
                .map(|t| format!("t{}", 2 + (t * 11 + i * 3) % 90))
                .collect::<Vec<_>>()
                .join(" ");
            let mut request = GenerationRequest::new(prompt, 12);
            request.seed = 100 + i as u64;
            request.sampling.temperature = Some(0.9);
            request.sampling.top_k = (i % 2 == 0).then_some(20);
            request.sampling.top_p = Some(0.9);
            request.sampling.presence_penalty = Some(0.6);
            request.sampling.repetition_penalty = (i % 3 == 0).then_some(1.3);
            request
        })
        .collect();
    let mut config = PipelineConfig::new(6, 8);
    config.scheduler.max_batch_tokens = 20;
    config.scheduler.min_prefill_tokens = 2;
    let mut solo = InferencePipeline::new(&factory, config.clone())?;
    let expected = requests
        .iter()
        .map(|request| Ok(solo.run(request, |_| {})?.generated_text))
        .collect::<Result<Vec<_>>>()?;
    assert!(
        expected
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 3
    );
    for order in [vec![0, 1, 2, 3, 4, 5], vec![5, 3, 1, 4, 2, 0]] {
        let mut batched = InferencePipeline::new(&factory, config.clone())?;
        let ids = order
            .iter()
            .map(|&i| Ok((batched.submit_request(&requests[i])?, i)))
            .collect::<Result<HashMap<_, _>>>()?;
        let (mut finished, mut generated) = (0, 0);
        while batched.active_request_count() > 0 {
            for (id, result) in batched.step_streaming()?.completed {
                let index = ids
                    .get(&id)
                    .ok_or_else(|| TurError::Other("unknown completion".into()))?;
                let stats = result?;
                assert_eq!(stats.generated_text, expected[*index], "request {index}");
                generated += stats.generated_tokens as u64;
                finished += 1;
            }
        }
        assert_eq!(finished, requests.len());
        // Every sampled token, EOS included, was decided on the device.
        let stats = batched.sampling_stats();
        assert_eq!(
            (stats.device_rows, stats.host_rows),
            (generated, 0),
            "{stats:?}"
        );
    }
    Ok(())
}

/// Six sampled requests of mixed lengths on the fused Metal fixture.
fn sampled_requests() -> Vec<GenerationRequest> {
    (0..6usize)
        .map(|i| {
            let prompt = (0..3 + i * 7)
                .map(|t| format!("t{}", 2 + (t * 13 + i * 5) % 90))
                .collect::<Vec<_>>()
                .join(" ");
            let mut request = GenerationRequest::new(prompt, 16);
            request.seed = 40 + i as u64;
            request.sampling.temperature = Some(1.0);
            request.sampling.top_p = Some(0.95);
            request
        })
        .collect()
}

/// Submit request `i` before pass `arrivals[i]` and run until idle; returns texts by index.
fn drive(
    pipeline: &mut InferencePipeline,
    requests: &[GenerationRequest],
    arrivals: &[usize],
) -> Result<HashMap<usize, String>> {
    let mut ids = HashMap::new();
    let mut texts = HashMap::new();
    for pass in 0.. {
        for (index, _) in arrivals.iter().enumerate().filter(|(_, a)| **a == pass) {
            ids.insert(pipeline.submit_request(&requests[index])?, index);
        }
        if texts.len() == requests.len() {
            break;
        }
        for (id, result) in pipeline.step_streaming()?.completed {
            texts.insert(ids[&id], result?.generated_text);
        }
    }
    Ok(texts)
}

#[test]
fn test_batched_runs_repeat_exactly_and_reproducible_mode_ignores_other_traffic() -> Result<()> {
    let checkpoint = common::Checkpoint::fused()?;
    let mut factory = checkpoint.factory();
    factory.device = Device::new_metal(0)?;
    let requests = sampled_requests();
    let arrivals = [0, 0, 1, 1, 2, 3];
    // Run-to-run determinism of one batched schedule: identical arrivals give identical
    // tokens in a fresh pipeline.
    let mut config = PipelineConfig::new(6, 8);
    config.scheduler.max_batch_tokens = 24;
    config.scheduler.min_prefill_tokens = 2;
    let first = drive(
        &mut InferencePipeline::new(&factory, config.clone())?,
        &requests,
        &arrivals,
    )?;
    let second = drive(
        &mut InferencePipeline::new(&factory, config)?,
        &requests,
        &arrivals,
    )?;
    assert_eq!(first, second);
    // Reproducibility mode: each request's tokens are those it produces alone, whatever
    // else is queued and in whatever order requests arrive.
    let alone = requests
        .iter()
        .map(|request| {
            let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::reproducible(8))?;
            Ok(pipeline.run(request, |_| {})?.generated_text)
        })
        .collect::<Result<Vec<_>>>()?;
    for order in [arrivals.to_vec(), vec![3, 2, 1, 0, 0, 0]] {
        let mut pipeline = InferencePipeline::new(&factory, PipelineConfig::reproducible(8))?;
        let texts = drive(&mut pipeline, &requests, &order)?;
        for (index, expected) in alone.iter().enumerate() {
            assert_eq!(&texts[&index], expected, "request {index}");
        }
    }
    Ok(())
}

#[test]
fn test_grammar_requests_are_sampled_on_the_device_and_batch_independently() -> Result<()> {
    use tur::backend::guidance::TopLevelGrammar;
    let checkpoint = common::Checkpoint::fused()?;
    let path = checkpoint.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    tokenizer["decoder"] = serde_json::json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false
    });
    std::fs::write(&path, tokenizer.to_string())?;
    let mut factory = checkpoint.factory();
    factory.device = Device::new_metal(0)?;
    let mut requests = sampled_requests();
    for (index, request) in requests.iter_mut().enumerate() {
        if index % 2 == 0 {
            request.grammar = Some(TopLevelGrammar::from_regex("(t2[0-9]){3}t9"));
            request.max_tokens = 40;
        }
    }
    let mut config = PipelineConfig::new(6, 8);
    config.scheduler.max_batch_tokens = 24;
    config.scheduler.min_prefill_tokens = 2;
    let mut solo = InferencePipeline::new(&factory, config.clone())?;
    let expected = requests
        .iter()
        .map(|request| Ok(solo.run(request, |_| {})?.generated_text))
        .collect::<Result<Vec<_>>>()?;
    // The fixture's single-digit tokens can open a pair that no later token completes, so
    // a text may stop early; it must still be a prefix of the grammar's language.
    let mut complete = 0;
    for (index, text) in expected.iter().enumerate().step_by(2) {
        let mut rest = text.as_str();
        for _ in 0..3 {
            let pair = rest.strip_prefix("t2").map(str::chars);
            match pair
                .and_then(|mut chars| chars.next().filter(char::is_ascii_digit).map(|_| chars))
            {
                Some(chars) => rest = chars.as_str(),
                None => break,
            }
        }
        assert!(["", "t", "t2", "t9"].contains(&rest), "{index}: {text}");
        complete += usize::from(rest == "t9");
    }
    assert!(complete >= 1, "{expected:?}");
    let mut batched = InferencePipeline::new(&factory, config)?;
    let texts = drive(&mut batched, &requests, &[0, 0, 1, 1, 2, 3])?;
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(&texts[&index], expected, "request {index}");
    }
    assert_eq!(batched.sampling_stats().host_rows, 0);
    assert_eq!(solo.sampling_stats().host_rows, 0);
    Ok(())
}

#[test]
fn test_logprobs_report_the_model_distribution_without_changing_the_tokens() -> Result<()> {
    let checkpoint = common::Checkpoint::fused()?;
    let path = checkpoint.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    tokenizer["decoder"] = serde_json::json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false
    });
    std::fs::write(&path, tokenizer.to_string())?;
    let mut factory = checkpoint.factory();
    factory.device = Device::new_metal(0)?;
    let mut config = PipelineConfig::new(6, 8);
    config.scheduler.max_batch_tokens = 24;
    config.scheduler.min_prefill_tokens = 2;
    let plain = sampled_requests();
    let mut asking = sampled_requests();
    for request in &mut asking {
        request.sampling.logprobs = Some(3);
    }
    let mut pipeline = InferencePipeline::new(&factory, config)?;
    for (plain, asking) in plain.iter().zip(&asking) {
        let expected = pipeline.run(plain, |_| {})?;
        // The second run may restore the prompt's last logits instead of computing them.
        let reported = pipeline.run(asking, |_| {})?;
        assert_eq!(reported.generated_text, expected.generated_text);
        assert_eq!(expected.logprobs, []);
        assert_ne!(reported.logprobs, []);
        assert!(reported.logprobs.len() <= reported.generated_tokens);
        for entry in &reported.logprobs {
            assert_eq!(entry.top.len(), 3);
            assert!(entry.top.windows(2).all(|w| w[0].logprob >= w[1].logprob));
            assert!(entry.top.iter().all(|top| top.logprob <= 0.0));
            // Nothing is likelier than the likeliest token, up to f32 rounding.
            assert!(entry.logprob <= entry.top[0].logprob + 1e-5);
            assert!(entry.top.iter().map(|top| top.logprob.exp()).sum::<f32>() <= 1.0 + 1e-4);
        }
    }
    Ok(())
}

/// Run `request` alone and count the passes it took.
fn run_counting_passes(
    pipeline: &mut InferencePipeline,
    request: &GenerationRequest,
) -> Result<(usize, tur::backend::pipeline::GenerationStats)> {
    let id = pipeline.submit_request(request)?;
    let mut passes = 0;
    loop {
        passes += usize::from(pipeline.begin_step()?);
        let step = pipeline.finish_step()?;
        for (done, stats) in step.completed {
            if done == id {
                return Ok((passes, stats?));
            }
        }
    }
}

#[test]
fn test_forced_tokens_skip_passes_and_leave_same_seed_outputs_unchanged() -> Result<()> {
    use tur::backend::guidance::TopLevelGrammar;
    let checkpoint = common::Checkpoint::fused()?;
    // Forcing tokenizes the forced text, which the fixture's whole-word tokenizer can't do.
    // Letters as tokens can, at the same vocabulary size.
    let letters: Vec<String> = (b'a'..=b'z').map(|c| (c as char).to_string()).collect();
    let mut vocab = serde_json::Map::new();
    vocab.insert("<eos>".into(), 0.into());
    let words = letters.iter().cloned().chain(
        letters
            .iter()
            .flat_map(|a| letters.iter().map(move |b| format!("{a}{b}"))),
    );
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(checkpoint.0.join("config.json"))?)?;
    let size = config["vocab_size"]
        .as_u64()
        .or_else(|| config["text_config"]["vocab_size"].as_u64())
        .ok_or_else(|| TurError::Other("fixture without vocab_size".into()))?;
    for (id, word) in words.take(size as usize - 1).enumerate() {
        vocab.insert(word, (id + 1).into());
    }
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
        "pre_tokenizer": null, "post_processor": null,
        "decoder": {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":false, "use_regex":false},
        "added_tokens": [{"id":0,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "model": {"type":"BPE", "merges":[], "vocab": vocab, "unk_token":"<eos>"}
    });
    std::fs::write(checkpoint.0.join("tokenizer.json"), tokenizer.to_string())?;
    let mut factory = checkpoint.factory();
    factory.device = Device::new_metal(0)?;
    let mut config = PipelineConfig::new(2, 8);
    config.prefix = PrefixConfig::DISABLED;
    let mut pipeline = InferencePipeline::new(&factory, config)?;
    for seed in 0..8 {
        let mut request = GenerationRequest::new("abc".into(), 40);
        request.seed = seed;
        request.sampling.temperature = Some(1.0);
        // The fixture's EOS ids are the lowest, so the grammar starts above them.
        // Free choices between forced runs: later draws must land where they always did.
        request.grammar = Some(TopLevelGrammar::from_regex("cdef[g-j]klmn[o-r]stuv"));
        let (forced_passes, forced) = run_counting_passes(&mut pipeline, &request)?;
        // Log probabilities need every position's logits, so this run samples each token.
        request.sampling.logprobs = Some(0);
        let (sampled_passes, sampled) = run_counting_passes(&mut pipeline, &request)?;
        assert_eq!(forced.generated_text, sampled.generated_text, "seed {seed}");
        assert_eq!(forced.generated_tokens, sampled.generated_tokens);
        assert!(
            forced_passes < sampled_passes,
            "seed {seed}: {forced_passes} forced passes, {sampled_passes} sampled"
        );
    }
    Ok(())
}
