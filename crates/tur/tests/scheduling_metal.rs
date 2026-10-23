#![cfg(feature = "metal")]
#![forbid(unsafe_code)]
#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
//! Packed serving pressure and device-budget arbitration on an actual Metal device.
mod common;

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
    let probe = InferencePipeline::new(&factory, config)?;
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
    let mut solo = InferencePipeline::new(&factory, config)?;
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
    let mut packed = InferencePipeline::with_budget(&factory, config, &budget)?;
    let other = InferencePipeline::with_budget(&factory, config, &budget)?;
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
    let mut solo = InferencePipeline::new(&factory, config)?;
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
        let mut batched = InferencePipeline::new(&factory, config)?;
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
        &mut InferencePipeline::new(&factory, config)?,
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
