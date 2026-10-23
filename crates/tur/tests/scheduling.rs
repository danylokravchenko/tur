#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
#![forbid(unsafe_code)]
//! Concurrency stress tests of the scheduled pipeline: mixed prompt and context lengths,
//! staggered and reshuffled arrivals, cancellation, memory pressure with recompute
//! preemption, fairness bounds, queue limits and deadlines. Outputs are compared against
//! each request run alone on random weights, where leaked or misplaced state changes text.
mod common;
use common::Checkpoint;
use std::{collections::HashMap, time::Duration};
use tur::{
    InferencePipeline, Result, TurError,
    backend::{
        pipeline::{GenerationRequest, GenerationStats, PipelineConfig},
        prefix::PrefixConfig,
    },
};
use uuid::Uuid;

/// Twelve requests with prompts of 1–37 tokens and limits of 1–23 tokens; odd ones sample
/// with temperature and nucleus truncation from their own seed.
fn requests() -> Vec<GenerationRequest> {
    (0..12usize)
        .map(|i| {
            let prompt = (0..=(i * 7 % 37))
                .map(|t| format!("t{}", 2 + (i * 13 + t * 29) % 90))
                .collect::<Vec<_>>()
                .join(" ");
            let mut request = GenerationRequest::new(prompt, 1 + i * 5 % 23);
            request.seed = i as u64;
            if i % 2 == 1 {
                request.sampling.temperature = Some(0.9);
                request.sampling.top_p = Some(0.95);
            }
            request
        })
        .collect()
}

fn solo(fixture: &Checkpoint, requests: &[GenerationRequest]) -> Result<Vec<String>> {
    let mut pipeline = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 3))?;
    requests
        .iter()
        .map(|request| Ok(pipeline.run(request, |_| {})?.generated_text))
        .collect()
}

/// Submit request `i` before step `arrivals[i]`, cancel `cancels[i]` before that step, and
/// step until idle. Returns every outcome and the steps at which each request emitted text.
struct Run {
    outcomes: HashMap<usize, Result<GenerationStats>>,
    emitted: HashMap<usize, Vec<usize>>,
    /// Requests that were live when cancelled.
    cancelled: Vec<usize>,
    steps: usize,
}

fn drive(
    pipeline: &mut InferencePipeline,
    requests: &[GenerationRequest],
    arrivals: &[usize],
    cancels: &HashMap<usize, usize>,
) -> Result<Run> {
    let mut ids: HashMap<Uuid, usize> = HashMap::new();
    let mut live: HashMap<usize, Uuid> = HashMap::new();
    let mut run = Run {
        outcomes: HashMap::new(),
        emitted: HashMap::new(),
        cancelled: Vec::new(),
        steps: 0,
    };
    let last_arrival = arrivals.iter().copied().max().unwrap_or(0);
    for step in 0.. {
        for (index, _) in arrivals.iter().enumerate().filter(|(_, a)| **a == step) {
            let id = pipeline.submit_request(&requests[index])?;
            ids.insert(id, index);
            live.insert(index, id);
        }
        for (&index, _) in cancels.iter().filter(|(_, at)| **at == step) {
            if let Some(id) = live.remove(&index) {
                pipeline.cancel(id)?;
                run.cancelled.push(index);
            }
        }
        if step > last_arrival && pipeline.active_request_count() == 0 {
            run.steps = step;
            break;
        }
        let output = pipeline.step_streaming()?;
        for (id, _) in &output.new_tokens {
            if let Some(&index) = ids.get(id) {
                run.emitted.entry(index).or_default().push(step);
            }
        }
        for (id, outcome) in output.completed {
            if let Some(&index) = ids.get(&id) {
                live.remove(&index);
                run.outcomes.insert(index, outcome);
            }
        }
    }
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(run)
}

fn text(run: &mut Run, index: usize) -> Result<String> {
    Ok(run
        .outcomes
        .remove(&index)
        .ok_or_else(|| TurError::Other(format!("request {index} did not finish")))??
        .generated_text)
}

/// Four running sequences in passes of nine tokens: decodes rotate and prefill advances
/// in chunks of at most five beside them.
fn tight() -> PipelineConfig {
    let mut config = PipelineConfig::new(4, 5);
    config.scheduler.max_batch_tokens = 9;
    config.scheduler.min_prefill_tokens = 3;
    config
}

#[test]
fn test_staggered_and_reshuffled_mixed_requests_match_solo_runs() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let requests = requests();
    let expected = solo(&fixture, &requests)?;
    // Generated text varies across requests, so equality below is not vacuous.
    assert!(expected.iter().filter(|t| !t.is_empty()).count() >= 8);
    let staggered: Vec<usize> = (0..requests.len()).map(|i| i / 2).collect();
    let reversed: Vec<usize> = (0..requests.len()).rev().map(|i| i / 3).collect();
    for arrivals in [vec![0; requests.len()], staggered, reversed] {
        let mut pipeline = InferencePipeline::new(&fixture.factory(), tight())?;
        let mut run = drive(&mut pipeline, &requests, &arrivals, &HashMap::new())?;
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(&text(&mut run, index)?, expected, "request {index}");
        }
        assert_eq!(pipeline.waiting_request_count(), 0);
    }
    Ok(())
}

/// A pipeline whose state pool holds `pages` K/V pages beside its recurrent slots.
fn budgeted(
    fixture: &Checkpoint,
    config: PipelineConfig,
    pages: usize,
) -> Result<InferencePipeline> {
    let usage = InferencePipeline::new(&fixture.factory(), config)?.memory()?;
    let budget =
        (2 * config.scheduler.max_running + 1) * usage.slot_bytes + pages * usage.page_bytes;
    let mut pipeline = InferencePipeline::new(
        &fixture.factory(),
        PipelineConfig {
            state_bytes: Some(budget),
            prefix: PrefixConfig::DISABLED,
            ..config
        },
    )?;
    assert_eq!(pipeline.memory()?.pages.capacity, pages);
    Ok(pipeline)
}

#[test]
fn test_preemption_and_cancellation_under_pool_pressure_preserve_survivors() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let requests = requests();
    let expected = solo(&fixture, &requests)?;
    // Four 16-token pages: the longest request needs three alone, so concurrent requests
    // repeatedly exhaust the pool and the newest are preempted and recomputed.
    let mut pipeline = budgeted(&fixture, tight(), 4)?;
    let arrivals: Vec<usize> = (0..requests.len()).map(|i| i / 4).collect();
    // Cancel one request on arrival, one mid-prefill and one mid-decode.
    let cancels = HashMap::from([(11, 2), (5, 3), (2, 6)]);
    let mut run = drive(&mut pipeline, &requests, &arrivals, &cancels)?;
    assert_eq!(run.cancelled.len(), cancels.len());
    let mut preemptions = 0;
    for (index, expected) in expected.iter().enumerate() {
        if cancels.contains_key(&index) {
            assert!(
                !run.outcomes.contains_key(&index),
                "cancelled {index} finished"
            );
            continue;
        }
        let stats = run
            .outcomes
            .remove(&index)
            .ok_or_else(|| TurError::Other(format!("request {index} did not finish")))??;
        assert_eq!(&stats.generated_text, expected, "request {index}");
        preemptions += stats.preemptions;
    }
    assert!(preemptions > 0, "the pool never forced a preemption");
    Ok(())
}

#[test]
fn test_decode_gaps_and_prefill_progress_are_bounded() -> Result<()> {
    // The degenerate fixture never samples EOS, so every request decodes to its limit.
    let fixture = Checkpoint::new()?;
    // Nine running slots: the long prompt is admitted beside eight decodes.
    let mut config = PipelineConfig::new(9, 6);
    config.scheduler.max_batch_tokens = 6;
    config.scheduler.min_prefill_tokens = 2;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config)?;
    let mut requests: Vec<_> = (0..8)
        .map(|i| GenerationRequest::new(format!("t{} t{}", 10 + i, 20 + i), 40))
        .collect();
    let long = (0..40).map(|t| format!("t{}", 2 + t)).collect::<Vec<_>>();
    requests.push(GenerationRequest::new(long.join(" "), 2));
    // The long prompt arrives once all eight short requests are decoding.
    let mut arrivals = vec![0; 8];
    arrivals.push(6);
    let run = drive(&mut pipeline, &requests, &arrivals, &HashMap::new())?;
    for index in 0..8 {
        let steps = &run.emitted[&index];
        assert_eq!(steps.len(), 40);
        // Eight decodes share four (with prefill waiting) to six decode tokens per pass:
        // no request waits more than two passes between tokens.
        let gap = steps.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        assert!(gap <= 2, "request {index} waited {gap} passes");
    }
    // The 40-token prompt advances by at least two tokens a pass while decodes run.
    let first = run.emitted[&8][0];
    assert!(first - 6 <= 20, "long prompt took {} passes", first - 6);
    assert!(run.steps > 40);
    Ok(())
}

#[test]
fn test_queue_limits_and_deadlines_fail_requests_and_release_state() -> Result<()> {
    // The degenerate fixture never samples EOS, so only the deadline ends the late request.
    let fixture = Checkpoint::new()?;
    let mut config = tight();
    config.scheduler.max_waiting = 2;
    config.scheduler.max_waiting_tokens = 60;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), config)?;
    let requests = requests();
    let first = pipeline.submit_request(&requests[3])?;
    pipeline.submit_request(&requests[4])?;
    assert!(
        pipeline.submit_request(&requests[0]).is_err(),
        "third waiting request"
    );
    pipeline.cancel(first)?;
    // 29 + 36 waiting prompt tokens would exceed the limit of 60.
    assert!(pipeline.submit_request(&requests[5]).is_err());
    // An expired request fails on the next step, after it already holds pool state.
    let mut late = GenerationRequest::new("t10 t11 t12".into(), 20);
    late.timeout = Some(Duration::from_millis(50));
    let late = pipeline.submit_request(&late)?;
    let mut outcome = None;
    while pipeline.active_request_count() > 0 {
        std::thread::sleep(Duration::from_millis(5));
        for (id, result) in pipeline.step_streaming()?.completed {
            if id == late {
                outcome = Some(result);
            }
        }
    }
    assert!(matches!(outcome, Some(Err(TurError::DeadlineExceeded))));
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_deadlines_include_intake_and_never_publish_a_late_completion() -> Result<()> {
    let fixture = Checkpoint::new()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 256))?;
    let mut request = GenerationRequest::new("t10".into(), 1);
    request.timeout = Some(Duration::from_secs(1));
    request.submitted_at = Some(
        std::time::Instant::now()
            .checked_sub(Duration::from_secs(2))
            .ok_or_else(|| TurError::Other("fixture arrival underflow".into()))?,
    );
    assert!(matches!(
        pipeline.submit_request(&request),
        Err(TurError::DeadlineExceeded)
    ));
    assert_eq!(pipeline.active_request_count(), 0);

    request = GenerationRequest::new(vec!["t10"; 256].join(" "), 1);
    let timeout = Duration::from_millis(1);
    request.timeout = Some(timeout);
    let mut streamed = String::new();
    match pipeline.run(&request, |text| streamed.push_str(text)) {
        Ok(stats) => assert!(
            stats.elapsed < timeout,
            "late completion: {:?}",
            stats.elapsed
        ),
        Err(TurError::DeadlineExceeded) => assert_eq!(streamed, ""),
        Err(error) => return Err(error),
    }
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    Ok(())
}

#[test]
fn test_sample_timestamps_include_tokens_that_text_decoding_buffers() -> Result<()> {
    let fixture = Checkpoint::new()?;
    // A punctuation token is buffered until completion by TokenOutputStream.
    let path = fixture.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let vocab = tokenizer["model"]["vocab"]
        .as_object_mut()
        .ok_or_else(|| TurError::Other("missing fixture vocabulary".into()))?;
    vocab.remove("t5");
    vocab.insert("!".into(), serde_json::json!(5));
    std::fs::write(path, tokenizer.to_string())?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 16))?;
    pipeline.submit_request(&GenerationRequest::new("t10".into(), 4))?;
    let (mut sampled, mut fragments) = (Vec::new(), 0);
    while pipeline.active_request_count() > 0 {
        let output = pipeline.step_streaming()?;
        fragments += output.new_tokens.len();
        sampled.extend(output.sampled_tokens);
        for (_, result) in output.completed {
            assert_eq!(result?.generated_tokens, 4);
        }
    }
    assert_eq!(sampled.len(), 4);
    assert_eq!(fragments, 1);
    assert!(sampled.iter().all(|(_, token, _)| *token == 5));
    assert!(sampled.windows(2).all(|pair| pair[0].2 <= pair[1].2));
    Ok(())
}

#[test]
fn test_loaded_models_share_a_budget_and_failed_loads_return_reservations() -> Result<()> {
    use tur::backend::memory::DeviceBudget;
    let fixture = Checkpoint::new()?;
    let mut config = PipelineConfig::new(1, 16);
    let probe = InferencePipeline::new(&fixture.factory(), config)?;
    config.state_bytes = Some(probe.memory_plan().state_pool);
    let budget = DeviceBudget::new(Some(probe.memory_plan().resident() * 3));
    drop(probe);
    let first = InferencePipeline::with_budget(&fixture.factory(), config, &budget)?;
    let second = InferencePipeline::with_budget(&fixture.factory(), config, &budget)?;
    assert_eq!(
        budget.leased(),
        first.memory_plan().resident() + second.memory_plan().resident()
    );
    assert!(matches!(
        InferencePipeline::with_budget(&fixture.factory(), config, &budget),
        Err(TurError::MemoryAllocation(_))
    ));
    drop(first);
    drop(second);
    assert_eq!(budget.leased(), 0);
    // A failed weight load releases the reservation made before touching its payload.
    std::fs::write(fixture.0.join("model.safetensors"), [])?;
    assert!(InferencePipeline::with_budget(&fixture.factory(), config, &budget).is_err());
    assert_eq!(budget.leased(), 0);
    // With no budget, memory admission must win even over the corrupt weight file.
    let empty = DeviceBudget::new(Some(0));
    assert!(matches!(
        InferencePipeline::with_budget(&fixture.factory(), config, &empty),
        Err(TurError::MemoryAllocation(_))
    ));
    assert_eq!(empty.leased(), 0);
    Ok(())
}

#[test]
fn test_f16_recompute_under_pressure_preserves_sampled_continuations() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let mut factory = fixture.factory();
    factory.recurrent = candle_core::DType::F16;
    // Each prompt alone needs three pages. Two prompts fill the four-page pool
    // before either can finish prefill, forcing preemption independently of EOS.
    let requests: Vec<_> = (0..3)
        .map(|i| {
            let prompt = (0..33 + i)
                .map(|t| format!("t{}", 2 + (t * 7 + i) % 80))
                .collect::<Vec<_>>()
                .join(" ");
            let mut request = GenerationRequest::new(prompt, 8);
            request.seed = i as u64;
            if i % 2 == 1 {
                request.sampling.temperature = Some(0.9);
            }
            request
        })
        .collect();
    let expected = {
        let mut solo = InferencePipeline::new(&factory, tight())?;
        requests
            .iter()
            .map(|request| Ok(solo.run(request, |_| {})?.generated_text))
            .collect::<Result<Vec<_>>>()?
    };
    let config = tight();
    let usage = InferencePipeline::new(&factory, config)?.memory()?;
    let bytes = (2 * config.scheduler.max_running + 1) * usage.slot_bytes + 4 * usage.page_bytes;
    let mut pipeline = InferencePipeline::new(
        &factory,
        PipelineConfig {
            state_bytes: Some(bytes),
            prefix: PrefixConfig::DISABLED,
            ..config
        },
    )?;
    let mut run = drive(
        &mut pipeline,
        &requests,
        &vec![0; requests.len()],
        &HashMap::new(),
    )?;
    let mut preemptions = 0;
    for (index, expected) in expected.iter().enumerate() {
        let stats = run
            .outcomes
            .remove(&index)
            .ok_or_else(|| TurError::Other("missing completion".into()))??;
        assert_eq!(&stats.generated_text, expected, "request {index}");
        preemptions += stats.preemptions;
    }
    assert!(preemptions > 0);
    Ok(())
}

#[test]
fn test_presence_penalty_steers_greedy_decoding_away_from_sampled_tokens() -> Result<()> {
    // Without penalties the degenerate fixture repeats token 5 forever.
    let fixture = Checkpoint::new()?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), PipelineConfig::new(1, 16))?;
    let plain = pipeline.run(&GenerationRequest::new("t10".into(), 6), |_| {})?;
    assert_eq!(plain.generated_text, ["t5"; 6].join(" "));
    let mut request = GenerationRequest::new("t10".into(), 6);
    request.sampling.presence_penalty = Some(100.0);
    request.sampling.temperature = Some(0.0);
    // Every other logit ties, so once token 5 is penalized the first maximum is token 0,
    // then token 1, the fixture's EOS.
    let stats = pipeline.run(&request, |_| {})?;
    assert_eq!(stats.generated_text, "t5 t0");
    assert_eq!((stats.generated_tokens, stats.finish_reason), (3, "stop"));
    assert!(pipeline.sampling_stats().host_rows > 0);
    Ok(())
}

#[test]
fn test_cancellation_while_a_pass_is_in_flight_releases_state_and_spares_survivors() -> Result<()> {
    let fixture = Checkpoint::random()?;
    let requests = requests();
    let expected = solo(&fixture, &requests)?;
    let mut pipeline = InferencePipeline::new(&fixture.factory(), tight())?;
    let mut ids = HashMap::new();
    for (index, request) in requests.iter().enumerate() {
        ids.insert(pipeline.submit_request(request)?, index);
    }
    let victims: Vec<Uuid> = ids
        .iter()
        .filter(|(_, index)| **index % 3 == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut outcomes = HashMap::new();
    let mut passes = 0;
    while pipeline.active_request_count() > 0 {
        let in_flight = pipeline.begin_step()?;
        // Between submission and completion: cancel some requests, admit nothing new.
        if passes == 3 {
            assert!(in_flight);
            for id in &victims {
                match pipeline.cancel(*id) {
                    Ok(()) | Err(TurError::RequestNotFound(_)) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        let output = pipeline.finish_step()?;
        let timing = output
            .timing
            .ok_or_else(|| TurError::Other("no pass".into()))?;
        assert!(timing.tokens > 0 && timing.device >= timing.encode);
        for (id, result) in output.completed {
            outcomes.insert(ids[&id], result?.generated_text);
        }
        passes += 1;
    }
    for (index, expected) in expected.iter().enumerate() {
        if index % 3 == 0 {
            // Short requests may finish before the cancellation.
            if let Some(text) = outcomes.get(&index) {
                assert_eq!(text, expected);
            }
            continue;
        }
        assert_eq!(outcomes.get(&index), Some(expected), "request {index}");
    }
    let usage = pipeline.memory()?;
    assert_eq!((usage.pages.leased, usage.slots.leased), (0, 0));
    assert!(pipeline.begin_step().is_ok_and(|ran| !ran));
    assert!(pipeline.finish_step()?.timing.is_none());
    Ok(())
}
