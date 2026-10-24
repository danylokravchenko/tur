#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmarks emit JSON artifacts and timing summaries"
)]
#![forbid(unsafe_code)]
//! Serving benchmark on a real checkpoint: bursts of concurrent chat requests through
//! `InferencePipeline` on Metal, reporting aggregate throughput, time to first token,
//! inter-token latency and pass counts. Every pass ends at device completion, so a step's
//! wall time includes its GPU work. Prints one JSON document.
//!
//! `TUR_QWEN35_0_8B=<snapshot> cargo bench --offline -p tur-engine --bench serving > out.json`
//! narrows with `TUR_BENCH_CONCURRENCY=1,8` and `TUR_BENCH_OUTPUT=64`.
//! `TUR_BENCH_SAMPLING=0.7,20,0.95` samples instead of greedy decoding,
//! `TUR_BENCH_DEVICE_SAMPLING=0` samples on the host, and
//! `TUR_BENCH_WORKLOADS=prefix TUR_BENCH_SPACING=0,256` runs the prefix-reuse workloads;
//! `TUR_BENCH_WORKLOADS=arrivals TUR_BENCH_RATE=2,4` runs open-loop Poisson arrivals, and
//! `TUR_BENCH_WORKLOADS=grammar` grammar-constrained requests.
//! `TUR_BENCH_WORKLOADS=speculation TUR_BENCH_DRAFT=0,4` compares adaptive prompt lookup
//! (`TUR_BENCH_MTP=1` adds the native MTP head to drafted configurations)
//! with ordinary decoding, including acceptance counters, memory and exact token IDs.

use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, time::Instant};
use tokenizers::Tokenizer;
use tur_core::{Result, TurError, execution::invalid};
use tur_engine::{
    backend::{
        chat_template::{ChatMessage, Role},
        factory::{ModelFactory, ModelSource},
        guidance::TopLevelGrammar,
        pipeline::{GenerationRequest, InferencePipeline, PipelineConfig},
        prefix::PrefixConfig,
        speculation::SpeculationConfig,
    },
    hybrid::SamplingOptions,
};
use tur_models::models::qwen3_5::{
    DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat, parse_recurrent_state,
};

const CHUNK: usize = 256;

fn env_sizes(name: &str, default: &str) -> Result<Vec<usize>> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|v| v.parse().map_err(|_| invalid(name.to_owned())))
        .collect()
}

fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// A prompt of at least `tokens` tokens, distinct per `salt`.
fn prompt(tokenizer: &Tokenizer, tokens: usize, salt: usize) -> Result<String> {
    let words = [
        "river", "lantern", "copper", "meadow", "signal", "harbor", "violet", "engine", "quartz",
        "summit", "orchard", "beacon", "glacier", "thunder", "saffron", "compass",
    ];
    let mut text = format!("Request {salt}. Continue the list:");
    let mut i = salt;
    loop {
        let count = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .len();
        if count >= tokens {
            return Ok(text);
        }
        for _ in 0..(tokens - count).div_ceil(2) {
            text.push(' ');
            text.push_str(words[i % words.len()]);
            i = i.wrapping_mul(31).wrapping_add(7);
        }
    }
}

struct Burst {
    prompts: Vec<String>,
    output: usize,
}

/// Sampling of every benchmark request: `greedy` or `temperature,top_k,top_p`.
fn sampling() -> Result<SamplingOptions> {
    let spec = std::env::var("TUR_BENCH_SAMPLING").unwrap_or_else(|_| "greedy".into());
    if spec == "greedy" {
        return Ok(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        });
    }
    let parts: Vec<&str> = spec.split(',').collect();
    let [t, k, p] = parts[..] else {
        return Err(invalid("TUR_BENCH_SAMPLING is greedy or T,K,P"));
    };
    let parse = |v: &str| v.parse::<f64>().map_err(|_| invalid("TUR_BENCH_SAMPLING"));
    Ok(SamplingOptions {
        temperature: Some(parse(t)?),
        top_k: Some(parse(k)? as usize),
        top_p: Some(parse(p)?),
        ..Default::default()
    })
}

fn summary(samples: &[f64]) -> Value {
    json!({"p50": percentile(samples, 0.5), "p95": percentile(samples, 0.95), "p99": percentile(samples, 0.99), "sum": samples.iter().sum::<f64>()})
}

/// Submit every request at once and step until all finish, timing each pass's phases.
fn run_requests(pipeline: &mut InferencePipeline, requests: &[GenerationRequest]) -> Result<Value> {
    let start = Instant::now();
    let mut ids = HashMap::new();
    for (index, request) in requests.iter().enumerate() {
        ids.insert(pipeline.submit_request(request)?, index);
    }
    let mut emitted: Vec<Vec<f64>> = vec![Vec::new(); ids.len()];
    let mut bursts: Vec<Vec<f64>> = vec![Vec::new(); ids.len()];
    let mut token_ids: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
    let mut ttft = vec![0.0; ids.len()];
    let mut cached = 0;
    let (mut drafted, mut accepted, mut draft_passes, mut draft_seconds) = (0, 0, 0, 0.0);
    let (mut generated, mut prompt_tokens, mut passes, mut packed) = (0, 0, 0, 0);
    let mut phases: HashMap<&str, Vec<f64>> = HashMap::new();
    while pipeline.active_request_count() > 0 {
        let output = pipeline.step_streaming()?;
        if let Some(t) = output.timing {
            passes += 1;
            packed += t.tokens;
            if t.drafts > 0 {
                draft_passes += 1;
                draft_seconds += t.device.as_secs_f64();
            }
            for (name, value) in [
                ("schedule", t.schedule),
                ("encode", t.encode),
                ("wait", t.wait),
                ("publish", t.publish),
                ("device", t.device),
                (
                    "pass",
                    t.schedule + t.encode + t.overlap + t.wait + t.publish,
                ),
            ] {
                phases.entry(name).or_default().push(value.as_secs_f64());
            }
            if let Some(idle) = t.idle_before {
                phases
                    .entry("idle_before")
                    .or_default()
                    .push(idle.as_secs_f64());
            }
        }
        let mut burst_started = vec![false; ids.len()];
        for (id, token, sampled_at) in &output.sampled_tokens {
            if let Some(&index) = ids.get(id) {
                let at = sampled_at.saturating_duration_since(start).as_secs_f64();
                emitted[index].push(at);
                token_ids[index].push(*token);
                // One burst per request and pass: speculative passes publish several.
                if !std::mem::replace(&mut burst_started[index], true) {
                    bursts[index].push(at);
                }
            }
        }
        for (id, result) in output.completed {
            let stats = result?;
            if let Some(&index) = ids.get(&id) {
                ttft[index] = stats.first_token.as_secs_f64();
            }
            generated += stats.generated_tokens;
            prompt_tokens += stats.prompt_tokens;
            cached += stats.cached_tokens;
            drafted += stats.draft_tokens;
            accepted += stats.accepted_draft_tokens;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let observed: usize = emitted.iter().map(Vec::len).sum();
    if observed != generated {
        return Err(invalid(
            "sample timestamps do not cover every generated token",
        ));
    }
    let itl: Vec<f64> = emitted
        .iter()
        .flat_map(|t| t.windows(2).map(|w| w[1] - w[0]))
        .collect();
    let burst_gaps: Vec<f64> = bursts
        .iter()
        .flat_map(|t| t.windows(2).map(|w| w[1] - w[0]))
        .collect();
    let burst_count: usize = bursts.iter().map(Vec::len).sum();
    let idle: f64 = phases.get("idle_before").map_or(0.0, |v| v.iter().sum());
    Ok(json!({
        "requests": ids.len(), "prompt_tokens": prompt_tokens, "cached_tokens": cached,
        "packed_tokens": packed, "generated_tokens": generated,
        "wall_seconds": wall, "output_tokens_per_second": generated as f64 / wall,
        "total_tokens_per_second": (generated + prompt_tokens) as f64 / wall,
        "ttft_seconds": {"p50": percentile(&ttft, 0.5), "p95": percentile(&ttft, 0.95), "max": percentile(&ttft, 1.0)},
        "itl_seconds": {"p50": percentile(&itl, 0.5), "p95": percentile(&itl, 0.95), "p99": percentile(&itl, 0.99)},
        "burst_gap_seconds": {"p50": percentile(&burst_gaps, 0.5), "p95": percentile(&burst_gaps, 0.95), "p99": percentile(&burst_gaps, 0.99)},
        "tokens_per_burst": generated as f64 / burst_count.max(1) as f64,
        "passes": passes,
        "phase_seconds": phases.iter().map(|(k, v)| (k.to_string(), summary(v))).collect::<serde_json::Map<_, _>>(),
        "device_idle_fraction_lower_bound": idle / wall,
        "sampling": format!("{:?}", pipeline.sampling_stats()),
        "speculation": {"drafted": drafted, "accepted": accepted, "passes": draft_passes, "device_seconds": draft_seconds},
        "token_ids": token_ids,
    }))
}

fn burst(
    pipeline: &mut InferencePipeline,
    case: &Burst,
    options: SamplingOptions,
) -> Result<Value> {
    let requests: Vec<GenerationRequest> = case
        .prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let mut request = GenerationRequest::new(prompt.clone(), case.output);
            request.seed = index as u64;
            request.sampling = options;
            request
        })
        .collect();
    run_requests(pipeline, &requests)
}

fn report(label: &str, result: &Value) {
    let ms = |path: &[&str]| {
        path.iter()
            .fold(result, |v, k| &v[*k])
            .as_f64()
            .unwrap_or(0.0)
            * 1e3
    };
    eprintln!(
        "{label}: {:.1} out tok/s, TTFT p50 {:.0} ms, ITL p50/p95 {:.1}/{:.1} ms, burst gap p50/p95 {:.1}/{:.1} ms ({:.2} tok/burst), pass p50 {:.1} ms (schedule {:.2}, encode {:.2}, wait {:.1}, publish {:.2}, idle {:.2}), cached {}",
        result["output_tokens_per_second"].as_f64().unwrap_or(0.0),
        ms(&["ttft_seconds", "p50"]),
        ms(&["itl_seconds", "p50"]),
        ms(&["itl_seconds", "p95"]),
        ms(&["burst_gap_seconds", "p50"]),
        ms(&["burst_gap_seconds", "p95"]),
        result["tokens_per_burst"].as_f64().unwrap_or(0.0),
        ms(&["phase_seconds", "pass", "p50"]),
        ms(&["phase_seconds", "schedule", "p50"]),
        ms(&["phase_seconds", "encode", "p50"]),
        ms(&["phase_seconds", "wait", "p50"]),
        ms(&["phase_seconds", "publish", "p50"]),
        ms(&["phase_seconds", "idle_before", "p50"]),
        result["cached_tokens"],
    );
}

/// Prefix workloads at one snapshot spacing (0 disables reuse): requests sharing a long
/// system prompt, and multi-turn conversations whose next prompt re-renders the history.
fn prefix_runs(
    factory: &ModelFactory,
    tokenizer: &Tokenizer,
    spacing: usize,
    options: SamplingOptions,
) -> Result<Vec<Value>> {
    let mut config = PipelineConfig::new(8, CHUNK);
    config.prefix = if spacing == 0 {
        PrefixConfig::DISABLED
    } else {
        PrefixConfig {
            spacing,
            entries: 16,
            state_share: None,
        }
    };
    let mut pipeline = InferencePipeline::new(factory, config)?;
    let slot_bytes = pipeline.memory()?.slot_bytes;
    let warm = Burst {
        prompts: vec![prompt(tokenizer, 64, 99)?],
        output: 8,
    };
    burst(&mut pipeline, &warm, options)?;
    pipeline.clear_prefix_cache();
    let mut runs = Vec::new();
    // Eight users behind one 1.5K-token system prompt, after one request warmed it.
    let system = prompt(tokenizer, 1536, 7)?;
    let shared = |user: usize| -> Result<GenerationRequest> {
        let mut request = GenerationRequest::new(prompt(tokenizer, 48, 1000 + user)?, 32);
        request
            .messages
            .insert(0, ChatMessage::new(Role::System, system.clone()));
        request.seed = user as u64;
        request.sampling = options;
        Ok(request)
    };
    run_requests(&mut pipeline, &[shared(0)?])?;
    let users = (1..=8).map(shared).collect::<Result<Vec<_>>>()?;
    let result = run_requests(&mut pipeline, &users)?;
    report(&format!("shared system prompt, spacing {spacing}"), &result);
    runs.push(json!({"workload": "shared_system_prompt", "spacing": spacing, "system_tokens": 1536, "result": result}));
    // Four conversations of six turns, advanced together turn by turn.
    pipeline.clear_prefix_cache();
    let mut conversations: Vec<Vec<ChatMessage>> = (0..4)
        .map(|c| -> Result<Vec<ChatMessage>> {
            Ok(vec![ChatMessage::new(
                Role::User,
                prompt(tokenizer, 256, 2000 + c)?,
            )])
        })
        .collect::<Result<_>>()?;
    let mut turns = Vec::new();
    for turn in 0..6 {
        let requests: Vec<GenerationRequest> = conversations
            .iter()
            .enumerate()
            .map(|(c, messages)| {
                let mut request = GenerationRequest::new(String::new(), 48);
                request.messages.clone_from(messages);
                request.render.enable_thinking = Some(false);
                request.seed = (c * 100 + turn) as u64;
                request.sampling = options;
                request
            })
            .collect();
        let start = Instant::now();
        let mut ids = HashMap::new();
        for (c, request) in requests.iter().enumerate() {
            ids.insert(pipeline.submit_request(request)?, c);
        }
        let (mut ttft, mut prompt_tokens, mut cached, mut replies) =
            (Vec::new(), 0, 0, vec![String::new(); requests.len()]);
        while pipeline.active_request_count() > 0 {
            for (id, result) in pipeline.step_streaming()?.completed {
                let stats = result?;
                ttft.push(stats.first_token.as_secs_f64());
                prompt_tokens += stats.prompt_tokens;
                cached += stats.cached_tokens;
                replies[ids[&id]] = stats.generated_text;
            }
        }
        for (c, reply) in replies.into_iter().enumerate() {
            conversations[c].push(ChatMessage::new(Role::Assistant, reply));
            conversations[c].push(ChatMessage::new(
                Role::User,
                prompt(tokenizer, 64, 3000 + c * 10 + turn)?,
            ));
        }
        turns.push(
            json!({"turn": turn, "prompt_tokens": prompt_tokens, "cached_tokens": cached,
            "ttft_seconds": {"p50": percentile(&ttft, 0.5), "max": percentile(&ttft, 1.0)},
            "wall_seconds": start.elapsed().as_secs_f64()}),
        );
    }
    let (prompt_total, cached_total) = turns.iter().fold((0, 0), |(p, c), t| {
        (
            p + t["prompt_tokens"].as_u64().unwrap_or(0),
            c + t["cached_tokens"].as_u64().unwrap_or(0),
        )
    });
    eprintln!(
        "multi-turn, spacing {spacing}: {cached_total} of {prompt_total} prompt tokens cached; last-turn TTFT p50 {:.0} ms",
        turns
            .last()
            .map_or(0.0, |t| t["ttft_seconds"]["p50"].as_f64().unwrap_or(0.0))
            * 1e3
    );
    let prefix = pipeline.prefix_stats();
    runs.push(
        json!({"workload": "multi_turn", "spacing": spacing, "conversations": 4, "turns": turns,
        "prefix": format!("{prefix:?}"), "snapshot_slot_bytes": slot_bytes}),
    );
    Ok(runs)
}

/// Open-loop Poisson arrivals of 512-token prompts at `rate` requests per second. Each
/// request is rendered and tokenized when it arrives: before the next pass (`serial`), or
/// while the device runs the current one (`overlapped`). TTFT starts at the scheduled
/// arrival, so intake delay counts.
fn arrivals(
    factory: &ModelFactory,
    tokenizer: &Tokenizer,
    rate: f64,
    count: usize,
    overlapped: bool,
    options: SamplingOptions,
) -> Result<Value> {
    let mut pipeline = InferencePipeline::new(factory, PipelineConfig::new(16, CHUNK))?;
    burst(
        &mut pipeline,
        &Burst {
            prompts: vec![prompt(tokenizer, 64, 99)?],
            output: 8,
        },
        options,
    )?;
    let prompts: Vec<String> = (0..count)
        .map(|i| prompt(tokenizer, 512, 5000 + i))
        .collect::<Result<_>>()?;
    // Deterministic exponential gaps.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut at = 0.0;
    let schedule: Vec<f64> = (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            at += -u.ln() / rate;
            at
        })
        .collect();
    let start = Instant::now();
    let (mut next, mut ids) = (0, HashMap::new());
    // Render and tokenize every request due by now; returns the seconds spent.
    let submit_due = |pipeline: &mut InferencePipeline,
                      ids: &mut HashMap<_, _>,
                      next: &mut usize|
     -> Result<f64> {
        let begin = Instant::now();
        while *next < count && start.elapsed().as_secs_f64() >= schedule[*next] {
            let mut request = GenerationRequest::new(prompts[*next].clone(), 64);
            request.seed = *next as u64;
            request.sampling = options;
            request.submitted_at =
                start.checked_add(std::time::Duration::from_secs_f64(schedule[*next]));
            ids.insert(pipeline.submit_request(&request)?, *next);
            *next += 1;
        }
        Ok(begin.elapsed().as_secs_f64())
    };
    let (mut ttft, mut emitted, mut intake, mut idle) =
        (Vec::new(), vec![Vec::new(); count], Vec::new(), 0.0);
    let mut generated = 0;
    while next < count || pipeline.active_request_count() > 0 {
        if pipeline.active_request_count() == 0 {
            let wait = schedule[next] - start.elapsed().as_secs_f64();
            if wait > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(wait));
            }
        }
        if !overlapped || pipeline.active_request_count() == 0 {
            intake.push(submit_due(&mut pipeline, &mut ids, &mut next)?);
        }
        pipeline.begin_step()?;
        if overlapped {
            intake.push(submit_due(&mut pipeline, &mut ids, &mut next)?);
        }
        let output = pipeline.finish_step()?;
        if let Some(t) = output.timing {
            idle += t.idle_before.map_or(0.0, |d| d.as_secs_f64());
        }
        for (id, _, at) in &output.sampled_tokens {
            if let Some(&index) = ids.get(id) {
                emitted[index].push(at.saturating_duration_since(start).as_secs_f64());
            }
        }
        for (_, result) in output.completed {
            let stats = result?;
            ttft.push(stats.first_token.as_secs_f64());
            generated += stats.generated_tokens;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let itl: Vec<f64> = emitted
        .iter()
        .flat_map(|t| t.windows(2).map(|w| w[1] - w[0]))
        .collect();
    intake.retain(|&seconds| seconds > 0.0);
    Ok(json!({
        "rate": rate, "requests": count, "overlapped": overlapped, "wall_seconds": wall,
        "output_tokens_per_second": generated as f64 / wall,
        "ttft_seconds": {"p50": percentile(&ttft, 0.5), "p95": percentile(&ttft, 0.95), "max": percentile(&ttft, 1.0)},
        "itl_seconds": {"p50": percentile(&itl, 0.5), "p95": percentile(&itl, 0.95), "p99": percentile(&itl, 0.99)},
        "intake_seconds": summary(&intake),
        "device_idle_fraction_lower_bound": idle / wall,
    }))
}

fn main() -> Result<()> {
    let dir = std::env::var("TUR_QWEN35_0_8B")
        .map(PathBuf::from)
        .map_err(|_| invalid("set TUR_QWEN35_0_8B to a Qwen3.5-0.8B snapshot"))?;
    let concurrency = env_sizes("TUR_BENCH_CONCURRENCY", "1,2,4,8,16")?;
    let output = env_sizes("TUR_BENCH_OUTPUT", "64")?
        .first()
        .copied()
        .unwrap_or(64);
    let workloads = std::env::var("TUR_BENCH_WORKLOADS").unwrap_or_else(|_| "decode,mixed".into());
    let device_sampling = std::env::var("TUR_BENCH_DEVICE_SAMPLING").map_or(true, |v| v != "0");
    let options = sampling()?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let factory = ModelFactory {
        source: ModelSource::LocalPath(dir.clone()),
        revision: "bench".into(),
        format: Some(WeightFormat::Q8_0),
        device: candle_core::Device::new_metal(0)?,
        kv: DEFAULT_KV_LAYOUT,
        recurrent: std::env::var("TUR_BENCH_STATE")
            .map_or(Ok(DEFAULT_RECURRENT_STATE), |s| parse_recurrent_state(&s))
            .map_err(TurError::Other)?,
        mtp: false,
    };
    let max = concurrency.iter().copied().max().unwrap_or(1);
    let mut runs = Vec::new();
    if workloads.contains("decode") || workloads.contains("mixed") {
        let mut config = PipelineConfig::new(max, CHUNK);
        config.device_sampling = device_sampling;
        let mut pipeline = InferencePipeline::new(&factory, config)?;
        // Warm pipelines and kernels.
        let warm = Burst {
            prompts: vec![prompt(&tokenizer, 64, 99)?],
            output: 8,
        };
        burst(&mut pipeline, &warm, options)?;
        if workloads.contains("decode") {
            for &requests in &concurrency {
                // Distinct prompts per burst, so no burst restores another's snapshots.
                let short = Burst {
                    prompts: (0..requests)
                        .map(|i| prompt(&tokenizer, 128, requests * 100 + i))
                        .collect::<Result<_>>()?,
                    output,
                };
                let result = burst(&mut pipeline, &short, options)?;
                report(&format!("decode-heavy c={requests}"), &result);
                runs.push(json!({"workload": "decode_heavy", "prompt": 128, "concurrency": requests, "result": result}));
            }
        }
        if workloads.contains("mixed") {
            // Mixed: half the requests carry 2K-token prompts, arriving with short ones.
            let mixed = Burst {
                prompts: (0..max)
                    .map(|i| prompt(&tokenizer, if i % 2 == 0 { 128 } else { 2048 }, 10_000 + i))
                    .collect::<Result<_>>()?,
                output,
            };
            let result = burst(&mut pipeline, &mixed, options)?;
            report(&format!("mixed c={max}"), &result);
            runs.push(
                json!({"workload": "mixed", "prompt": [128, 2048], "concurrency": max, "result": result}),
            );
        }
    }
    if workloads.contains("grammar") {
        // JSON-like lists that the grammar terminates; every token is masked.
        let grammar =
            TopLevelGrammar::from_regex(r#"\{"items": \[("[a-z]{2,10}", ){3,6}"[a-z]{2,10}"\]\}"#);
        let mut config = PipelineConfig::new(max, CHUNK);
        config.device_sampling = device_sampling;
        // Repeated prompts would otherwise restore from prompt-end snapshots.
        config.prefix = PrefixConfig::DISABLED;
        let mut pipeline = InferencePipeline::new(&factory, config)?;
        for &concurrency in &concurrency {
            let requests: Vec<GenerationRequest> = (0..concurrency)
                .map(|i| {
                    let mut request = GenerationRequest::new(
                        format!("Request {i}. List a few animals as JSON."),
                        output,
                    );
                    request.seed = i as u64;
                    request.sampling = options;
                    request.render.enable_thinking = Some(false);
                    request.grammar = Some(grammar.clone());
                    request
                })
                .collect();
            // The first run builds the parser factory over the tokenizer; time the second.
            run_requests(&mut pipeline, &requests)?;
            let result = run_requests(&mut pipeline, &requests)?;
            report(&format!("grammar c={concurrency}"), &result);
            runs.push(json!({"workload": "grammar", "concurrency": concurrency, "result": result}));
        }
    }
    if workloads.contains("arrivals") {
        let count = env_sizes("TUR_BENCH_REQUESTS", "48")?
            .first()
            .copied()
            .unwrap_or(48);
        for rate in env_sizes("TUR_BENCH_RATE", "2,4")? {
            for overlapped in [false, true] {
                let result = arrivals(
                    &factory,
                    &tokenizer,
                    rate as f64,
                    count,
                    overlapped,
                    options,
                )?;
                eprintln!(
                    "arrivals {rate}/s, {}: {:.1} out tok/s, TTFT p50/p95 {:.0}/{:.0} ms, ITL p50/p95/p99 {:.1}/{:.1}/{:.1} ms, intake p50 {:.2} ms (sum {:.0} ms), idle {:.2}%",
                    if overlapped { "overlapped" } else { "serial" },
                    result["output_tokens_per_second"].as_f64().unwrap_or(0.0),
                    result["ttft_seconds"]["p50"].as_f64().unwrap_or(0.0) * 1e3,
                    result["ttft_seconds"]["p95"].as_f64().unwrap_or(0.0) * 1e3,
                    result["itl_seconds"]["p50"].as_f64().unwrap_or(0.0) * 1e3,
                    result["itl_seconds"]["p95"].as_f64().unwrap_or(0.0) * 1e3,
                    result["itl_seconds"]["p99"].as_f64().unwrap_or(0.0) * 1e3,
                    result["intake_seconds"]["p50"].as_f64().unwrap_or(0.0) * 1e3,
                    result["intake_seconds"]["sum"].as_f64().unwrap_or(0.0) * 1e3,
                    result["device_idle_fraction_lower_bound"]
                        .as_f64()
                        .unwrap_or(0.0)
                        * 100.0,
                );
                runs.push(json!({"workload": "arrivals", "result": result}));
            }
        }
    }
    if workloads.contains("prefix") {
        for spacing in env_sizes("TUR_BENCH_SPACING", "0,64,128,256,512")? {
            runs.extend(prefix_runs(&factory, &tokenizer, spacing, options)?);
        }
    }
    if workloads.contains("speculation") {
        let repeats = env_sizes("TUR_BENCH_REPEATS", "3")?
            .first()
            .copied()
            .unwrap_or(3);
        let mtp = std::env::var("TUR_BENCH_MTP").is_ok_and(|v| v == "1");
        for draft in env_sizes("TUR_BENCH_DRAFT", "0,4")? {
            for &count in &concurrency {
                let mut config = PipelineConfig::new(count, CHUNK);
                config.prefix = PrefixConfig::DISABLED;
                config.device_sampling = device_sampling;
                let with_mtp = mtp && draft > 0;
                config.speculation = if with_mtp {
                    SpeculationConfig::with_mtp(draft)
                } else {
                    SpeculationConfig::new(draft)
                };
                let mut pipeline = InferencePipeline::new(&factory, config)?;
                for workload in ["repeat", "novel"] {
                    let requests: Vec<_> = (0..count).map(|i| {
                        let text = if workload == "repeat" {
                            format!("Request {i}. Continue this pattern exactly for 200 words, without an introduction:\n{}", "red green blue yellow orange purple ".repeat(12))
                        } else { format!("Request {i}. Explain how a database recovers from a power failure during a transaction. Be specific about write-ahead logs, checkpoints and isolation.") };
                        let mut request = GenerationRequest::new(text, output);
                        request.seed = i as u64;
                        request.render.enable_thinking = Some(false);
                        request.sampling = options;
                        request
                    }).collect();
                    // Warm execution separately; adaptive draft selection remains active.
                    run_requests(&mut pipeline, &requests)?;
                    for repeat in 0..repeats {
                        let result = run_requests(&mut pipeline, &requests)?;
                        report(
                            &format!(
                                "speculation {workload} c={count} draft={draft}{}",
                                if with_mtp { "+mtp" } else { "" }
                            ),
                            &result,
                        );
                        let memory = pipeline.memory_plan();
                        let stats = pipeline.speculation_stats();
                        runs.push(json!({"workload": workload, "concurrency": count, "draft": draft, "repeat": repeat,
                            "mtp": with_mtp, "speculation_totals": format!("{stats:?}"),
                            "resident_bytes": memory.resident(), "state_pool_bytes": memory.state_pool, "draft_slots": memory.draft_slots,
                            "cost_fit_seconds": pipeline.draft_costs().line(count),
                            "result": result}));
                    }
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "benchmark": "serving", "checkpoint": dir, "weights": "q8_0",
            "kv": format!("{:?}", DEFAULT_KV_LAYOUT), "prefill_chunk": CHUNK,
            "output_tokens": output, "workload": "closed-loop bursts submitted at once",
            "sampling": format!("{options:?}"), "device_sampling": device_sampling,
            "latency_basis": "sampled tokens including EOS; TTFT from request submission; ITL between token publication times (zero inside a speculative burst); burst gaps between the first publications of successive passes; pass phases from PassTiming",
            "recurrent_state": format!("{:?}", factory.recurrent),
            "runs": runs,
        }))?
    );
    Ok(())
}
