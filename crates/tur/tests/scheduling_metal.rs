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
    },
};

#[test]
fn test_f16_packed_recompute_and_loaded_model_budget_on_metal() -> Result<()> {
    let checkpoint = common::Checkpoint::fused()?;
    let mut factory = checkpoint.factory();
    factory.device = Device::new_metal(0)?;
    factory.recurrent = DType::F16;
    let mut config = PipelineConfig::new(3, 8);
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
