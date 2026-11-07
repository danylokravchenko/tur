//! Offline bound on table-based drafters: records each request's tokens and its model's
//! likeliest next tokens at every position, then replays drafters over those traces through
//! the engine's own `NgramIndex`, `Acceptance` and `plan`. Replay needs no kernel work because
//! target matching publishes the same tokens whatever is drafted. Pass times come from the
//! cost line measured on this Mac, and real runs of lookup and lookup plus MTP sit beside it.
//!
//! `TUR_BENCH_MODEL=<snapshot> cargo bench --offline -p tur --bench perf -- drafter > out.json`
//! narrows with `TUR_BENCH_CONCURRENCY=1,4` and `TUR_BENCH_OUTPUT=256`, and samples with
//! `TUR_BENCH_SAMPLING=0.7,20,0.95`.
//!
//! Drafters: `lookup` alone, then lookup beside a table of `bigram` followers of published
//! tokens, `ngram` followers of the last two to four tokens (as llama.cpp's `ngram-mod`), or
//! `recycling` (Token Recycling's top candidates, which need logits on the host). `warm`
//! tables persist across requests, as one tenant's would; the others start empty per request.

use super::serving::{env_sizes, percentile, run_requests, sampling};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    path::PathBuf,
};
use tokenizers::Tokenizer;
use tur_core::{Result, TurError, execution::invalid};
use tur_engine::{
    backend::{
        factory::{ModelFactory, ModelSource},
        pipeline::{GenerationRequest, InferencePipeline, PipelineConfig},
        prefix::PrefixConfig,
        speculation::{self, Acceptance, DraftCosts, NgramIndex, Proposal, SpeculationConfig},
    },
    hybrid::SamplingOptions,
};
use tur_models::models::{DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat};

const CHUNK: usize = 256;
const MAX_DRAFT: usize = 4;
const CANDIDATES: usize = 8;

/// One tenant's requests in order, with domains interleaved as a shared server would see them.
const PROMPTS: [(&str, &str); 16] = [
    (
        "prose",
        "Explain how a database recovers from a power failure during a transaction.",
    ),
    (
        "code",
        "Write a Rust function that parses an INI file into a map of sections to key-value pairs. Only output code.",
    ),
    (
        "json",
        "Return a JSON array of five fictional employees with name, title, department, start_date and manager fields. Only output JSON.",
    ),
    (
        "chat",
        "My sourdough starter smells like acetone and stopped rising. What should I do this week?",
    ),
    (
        "code",
        "Write a Python class implementing an LRU cache with get, put and a thread lock. Only output code.",
    ),
    (
        "prose",
        "Describe how the Roman aqueducts moved water across long distances and how they were maintained.",
    ),
    (
        "chat",
        "I have two days in Lisbon with a toddler. Suggest a relaxed plan with food recommendations.",
    ),
    (
        "json",
        "Return a JSON object describing a weather API with endpoints, parameters, and example responses. Only output JSON.",
    ),
    (
        "prose",
        "Explain how vaccines train the immune system, including memory cells and booster doses.",
    ),
    (
        "json",
        "Return a JSON array of six products with sku, name, price, currency, stock and tags fields. Only output JSON.",
    ),
    (
        "code",
        "Write a TypeScript function that debounces async calls and cancels stale results. Only output code.",
    ),
    (
        "chat",
        "How should I prepare for a salary negotiation after a promotion without a raise?",
    ),
    (
        "code",
        "Write a C function that reverses a singly linked list and a small test in main. Only output code.",
    ),
    (
        "prose",
        "Explain how a compiler turns source code into machine code, stage by stage.",
    ),
    (
        "chat",
        "My tomato plants have yellow lower leaves and curling tops. What is likely wrong?",
    ),
    (
        "json",
        "Return a JSON object for a three-step onboarding checklist with titles, owners, due dates and nested subtasks. Only output JSON.",
    ),
];

/// One request's published tokens and the model's likeliest tokens at each of them.
struct Trace {
    domain: &'static str,
    prompt: Vec<u32>,
    tokens: Vec<u32>,
    top: Vec<Vec<u32>>,
}

/// Where a drafter's table learns from, and which suffix lengths key it.
#[derive(Clone, Copy, PartialEq)]
enum Learns {
    Nothing,
    /// Followers in published text, including prompts, for suffixes of these lengths.
    Published(usize, usize),
    /// The model's top candidates at verified positions, after each input token.
    Logits,
}

#[derive(Clone, Copy)]
struct Drafter {
    name: &'static str,
    learns: Learns,
    warm: bool,
    /// Charge a logit row's readback per verified position; off bounds a cheaper path.
    readback: bool,
}

const DRAFTERS: [Drafter; 7] = [
    Drafter {
        name: "recycling-warm-free-readback",
        learns: Learns::Logits,
        warm: true,
        readback: false,
    },
    Drafter {
        name: "lookup",
        learns: Learns::Nothing,
        warm: false,
        readback: true,
    },
    Drafter {
        name: "bigram",
        learns: Learns::Published(1, 1),
        warm: false,
        readback: true,
    },
    Drafter {
        name: "bigram-warm",
        learns: Learns::Published(1, 1),
        warm: true,
        readback: true,
    },
    Drafter {
        name: "ngram-warm",
        learns: Learns::Published(2, 4),
        warm: true,
        readback: true,
    },
    Drafter {
        name: "recycling",
        learns: Learns::Logits,
        warm: false,
        readback: true,
    },
    Drafter {
        name: "recycling-warm",
        learns: Learns::Logits,
        warm: true,
        readback: true,
    },
];

/// Measured pass cost: seconds of a pass's base and per draft, and of reading back a row's
/// logits for the drafters that learn from them.
#[derive(Clone, Copy)]
struct PassCost {
    base: f64,
    per_draft: f64,
    per_row_readback: f64,
}

fn key(suffix: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    suffix.hash(&mut hasher);
    hasher.finish()
}

/// Suffix hash → candidate followers, likeliest first.
#[derive(Default)]
struct Table {
    followers: HashMap<u64, Vec<u32>>,
}

impl Table {
    /// Chain up to `limit` drafts, feeding each back as the newest token.
    fn propose(&self, history: &[u32], lengths: (usize, usize), limit: usize) -> Vec<u32> {
        let mut context = history[history.len().saturating_sub(lengths.1)..].to_vec();
        let mut chain = Vec::with_capacity(limit);
        while chain.len() < limit {
            let next = (lengths.0..=lengths.1)
                .rev()
                .filter(|&n| n <= context.len())
                .find_map(|n| self.followers.get(&key(&context[context.len() - n..])))
                .and_then(|followers| followers.first());
            let Some(&next) = next else { break };
            chain.push(next);
            context.push(next);
            if context.len() > lengths.1 {
                context.remove(0);
            }
        }
        chain
    }

    fn learn_published(&mut self, text: &[u32], lengths: (usize, usize), from: usize) {
        for end in from.max(1)..text.len() {
            for n in (lengths.0..=lengths.1).filter(|&n| n <= end) {
                self.followers
                    .insert(key(&text[end - n..end]), vec![text[end]]);
            }
        }
    }

    fn top_hit(&self, token: u32, next: u32) -> bool {
        self.followers
            .get(&key(&[token]))
            .is_some_and(|candidates| candidates.contains(&next))
    }
}

#[derive(Default)]
struct Totals {
    tokens: usize,
    passes: usize,
    seconds: f64,
    baseline_seconds: f64,
    lookup_drafted: usize,
    lookup_accepted: usize,
    table_drafted: usize,
    table_accepted: usize,
    /// Positions the table's top candidates could have covered, for a draft tree's ceiling.
    tree_positions: usize,
    tree_hits: usize,
}

impl Totals {
    fn value(&self) -> Value {
        let rate = |a: usize, d: usize| if d == 0 { 0.0 } else { a as f64 / d as f64 };
        json!({
            "speedup": self.baseline_seconds / self.seconds.max(f64::MIN_POSITIVE),
            "tokens_per_pass": rate(self.tokens, self.passes),
            "lookup_drafted": self.lookup_drafted,
            "lookup_acceptance": rate(self.lookup_accepted, self.lookup_drafted),
            "table_drafted": self.table_drafted,
            "table_acceptance": rate(self.table_accepted, self.table_drafted),
            "top8_hit_rate": rate(self.tree_hits, self.tree_positions),
        })
    }
}

/// One decoding request during replay; `next` indexes the token its pass draws.
struct Replay<'a> {
    trace: &'a Trace,
    next: usize,
    history: Vec<u32>,
    index: NgramIndex,
    lookup: Acceptance,
    table: Acceptance,
}

/// Run one drafter over `group`, requests decoding in lockstep, updating `table` as passes
/// complete. Stops with the shortest trace, so every pass has the same number of decodes.
fn replay(
    group: &[&Trace],
    drafter: Drafter,
    table: &mut Table,
    cost: PassCost,
    totals: &mut Totals,
) {
    let config = SpeculationConfig::new(MAX_DRAFT);
    let end = group.iter().map(|t| t.tokens.len()).min().unwrap_or(0);
    let mut replays: Vec<Replay<'_>> = group
        .iter()
        .map(|&trace| Replay {
            trace,
            next: 1,
            history: trace
                .prompt
                .iter()
                .chain(trace.tokens.first())
                .copied()
                .collect(),
            index: NgramIndex::default(),
            lookup: Acceptance::default(),
            table: Acceptance::default(),
        })
        .collect();
    if let Learns::Published(min, max) = drafter.learns {
        for replay in &replays {
            table.learn_published(&replay.history, (min, max), 0);
        }
    }
    let lengths = match drafter.learns {
        Learns::Published(min, max) => (min, max),
        Learns::Nothing | Learns::Logits => (1, 1),
    };
    let readback = if drafter.learns == Learns::Logits && drafter.readback {
        cost.per_row_readback
    } else {
        0.0
    };
    let decodes = group.len();
    let mut costs = DraftCosts::default();
    while replays.iter().all(|r| r.next < end) {
        let mut chosen = Vec::with_capacity(decodes);
        for r in &mut replays {
            let cap = MAX_DRAFT.min(end - r.next - 1);
            let found = r.index.propose(&r.history, &config, cap);
            let learned = if drafter.learns == Learns::Nothing {
                Vec::new()
            } else {
                table.propose(&r.history, lengths, cap)
            };
            let use_table =
                !learned.is_empty() && (found.is_empty() || r.table.rate() > r.lookup.rate());
            chosen.push(if use_table {
                (learned, true)
            } else {
                (found, false)
            });
        }
        let proposals: Vec<Proposal> = chosen
            .iter()
            .zip(&replays)
            .map(|((tokens, from_table), r)| Proposal {
                tokens: tokens.len(),
                acceptance: if *from_table { r.table } else { r.lookup },
            })
            .collect();
        let counts = speculation::plan(&proposals, decodes, &costs, MAX_DRAFT * 4, None);
        let mut drafted_total = 0;
        for ((r, (tokens, from_table)), &count) in replays.iter_mut().zip(&chosen).zip(&counts) {
            let actual = &r.trace.tokens;
            let accepted = tokens[..count]
                .iter()
                .zip(&actual[r.next..])
                .take_while(|(draft, token)| draft == token)
                .count();
            drafted_total += count;
            let (drafted_stat, accepted_stat, acceptance) = if *from_table {
                (
                    &mut totals.table_drafted,
                    &mut totals.table_accepted,
                    &mut r.table,
                )
            } else {
                (
                    &mut totals.lookup_drafted,
                    &mut totals.lookup_accepted,
                    &mut r.lookup,
                )
            };
            *drafted_stat += count;
            *accepted_stat += accepted;
            if count > 0 {
                acceptance.observe(count, accepted);
            } else if !tokens.is_empty() {
                acceptance.recover(Acceptance::default());
            }
            let published = accepted + 1;
            if drafter.learns == Learns::Logits {
                for position in r.next..r.next + published {
                    let input = actual[position - 1];
                    totals.tree_positions += 1;
                    totals.tree_hits += usize::from(table.top_hit(input, actual[position]));
                    table
                        .followers
                        .insert(key(&[input]), r.trace.top[position].clone());
                }
            }
            r.history
                .extend_from_slice(&actual[r.next..r.next + published]);
            if let Learns::Published(min, max) = drafter.learns {
                table.learn_published(&r.history, (min, max), r.history.len() - published);
            }
            r.next += published;
            totals.tokens += published;
        }
        let pass = readback.mul_add(
            (decodes + drafted_total) as f64,
            cost.per_draft.mul_add(drafted_total as f64, cost.base),
        );
        costs.record(decodes, drafted_total, pass);
        totals.passes += 1;
        totals.seconds += pass;
    }
    let published: usize = replays.iter().map(|r| r.next - 1).sum();
    totals.baseline_seconds =
        (published as f64 / decodes as f64).mul_add(cost.base, totals.baseline_seconds);
}

fn request(prompt: &str, seed: u64, output: usize, options: SamplingOptions) -> GenerationRequest {
    let mut request = GenerationRequest::new(prompt.to_owned(), output);
    request.seed = seed;
    request.sampling = options;
    request.render.enable_thinking = Some(false);
    request
}

fn pipeline(
    factory: &ModelFactory,
    count: usize,
    speculation: SpeculationConfig,
) -> Result<InferencePipeline> {
    let mut config = PipelineConfig::new(count, CHUNK);
    config.prefix = PrefixConfig::DISABLED;
    config.speculation = speculation;
    InferencePipeline::new(factory, config)
}

/// Each token's decoded text → its ID. Logprobs report text, so candidates are mapped back;
/// IDs sharing lossy text (partial characters) resolve to the first, which only loses drafts.
fn ids_by_text(tokenizer: &Tokenizer) -> Result<HashMap<String, u32>> {
    let mut ids = HashMap::new();
    for id in 0..tokenizer.get_vocab_size(true) as u32 {
        let text = tokenizer
            .decode(&[id], false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?;
        ids.entry(text).or_insert(id);
    }
    Ok(ids)
}

/// Generate every prompt alone, reporting logprobs, and record its trace. Also returns the
/// share of sampled tokens whose reported text maps back to their own ID.
fn record(
    factory: &ModelFactory,
    tokenizer: &Tokenizer,
    output: usize,
    options: SamplingOptions,
) -> Result<(Vec<Trace>, f64)> {
    let ids = ids_by_text(tokenizer)?;
    let mut pipeline = pipeline(factory, 1, SpeculationConfig::DISABLED)?;
    let mut traces = Vec::with_capacity(PROMPTS.len());
    let (mut mapped, mut checked) = (0usize, 0usize);
    for (seed, (domain, text)) in PROMPTS.iter().enumerate() {
        let mut generation = request(text, seed as u64, output, options);
        generation.sampling.logprobs = Some(CANDIDATES);
        let id = pipeline.submit_request(&generation)?;
        let mut tokens = Vec::with_capacity(output);
        let mut logprobs = Vec::new();
        while pipeline.active_request_count() > 0 {
            let step = pipeline.step_streaming()?;
            tokens.extend(
                step.sampled_tokens
                    .iter()
                    .filter(|t| t.0 == id)
                    .map(|t| t.1),
            );
            for (_, result) in step.completed {
                logprobs = result?.logprobs;
            }
        }
        // Logprobs leave out EOS, which the replay then never reaches.
        tokens.truncate(logprobs.len());
        for (token, logprob) in tokens.iter().zip(&logprobs) {
            checked += 1;
            mapped += usize::from(ids.get(&logprob.token) == Some(token));
        }
        let top = logprobs
            .iter()
            .map(|logprob| {
                logprob
                    .top
                    .iter()
                    .filter_map(|c| ids.get(&c.token).copied())
                    .collect()
            })
            .collect();
        let prompt = tokenizer
            .encode(*text, false)
            .map_err(|e| TurError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        traces.push(Trace {
            domain,
            prompt,
            tokens,
            top,
        });
        eprintln!(
            "recorded {seed}: {domain}, {} tokens",
            traces[seed].tokens.len()
        );
    }
    Ok((traces, mapped as f64 / checked.max(1) as f64))
}

/// Median seconds per output token of `count` requests decoding, with these logprobs.
fn seconds_per_token(
    factory: &ModelFactory,
    count: usize,
    output: usize,
    options: SamplingOptions,
    logprobs: Option<usize>,
) -> Result<f64> {
    let mut pipeline = pipeline(factory, count, SpeculationConfig::DISABLED)?;
    let mut samples = Vec::with_capacity(4);
    for repeat in 0..4 {
        let requests: Vec<_> = PROMPTS[..count]
            .iter()
            .enumerate()
            .map(|(seed, (_, text))| {
                let mut generation = request(text, seed as u64, output, options);
                generation.sampling.logprobs = logprobs;
                generation
            })
            .collect();
        let result = run_requests(&mut pipeline, &requests)?;
        // The first run warms execution.
        if repeat > 0 {
            samples.push(
                1.0 / result["output_tokens_per_second"]
                    .as_f64()
                    .unwrap_or(f64::MAX),
            );
        }
    }
    Ok(percentile(&samples, 0.5))
}

/// The base and per-draft line of `count` decoding requests, measured on repetitive output
/// where prompt lookup drafts often enough to separate them.
fn measured_line(factory: &ModelFactory, count: usize, output: usize) -> Result<(f64, f64)> {
    let mut pipeline = pipeline(factory, count, SpeculationConfig::new(MAX_DRAFT))?;
    let text = "red green blue yellow orange purple ".repeat(12);
    let requests: Vec<_> = (0..count)
        .map(|i| {
            let prompt = format!("Request {i}. Continue this pattern exactly:\n{text}");
            request(
                &prompt,
                i as u64,
                output,
                SamplingOptions {
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
        })
        .collect();
    for _ in 0..3 {
        run_requests(&mut pipeline, &requests)?;
    }
    pipeline
        .draft_costs()
        .line(count)
        .ok_or_else(|| invalid("draft costs were not identified"))
}

/// One round of every prompt in groups of `count`: tokens per second and each group's tokens.
fn real_round(
    pipeline: &mut InferencePipeline,
    count: usize,
    output: usize,
    options: SamplingOptions,
) -> Result<(f64, Vec<Value>)> {
    let (mut generated, mut wall) = (0.0, 0.0);
    let mut tokens = Vec::new();
    for (group, prompts) in PROMPTS.chunks_exact(count).enumerate() {
        let requests: Vec<_> = prompts
            .iter()
            .enumerate()
            .map(|(i, (_, text))| request(text, (group * count + i) as u64, output, options))
            .collect();
        let result = run_requests(pipeline, &requests)?;
        generated += result["generated_tokens"].as_f64().unwrap_or(0.0);
        wall += result["wall_seconds"].as_f64().unwrap_or(0.0);
        tokens.push(result["token_ids"].clone());
    }
    Ok((generated / wall, tokens))
}

/// Real lookup, and lookup with the checkpoint's head, against normal decoding, in rotation
/// so that thermal drift falls on all: each mode's ratio per round, after a warming round,
/// and whether its tokens matched.
fn real_comparison(
    factory: &ModelFactory,
    count: usize,
    output: usize,
    options: SamplingOptions,
) -> Result<Value> {
    let mut normal = pipeline(factory, count, SpeculationConfig::DISABLED)?;
    let mut modes = [
        (
            "lookup",
            pipeline(factory, count, SpeculationConfig::new(MAX_DRAFT))?,
        ),
        (
            "lookup+mtp",
            pipeline(factory, count, SpeculationConfig::with_mtp(MAX_DRAFT))?,
        ),
    ];
    let mut ratios = [Vec::new(), Vec::new()];
    let mut identical = [true; 2];
    for round in 0..4 {
        let (plain, plain_tokens) = real_round(&mut normal, count, output, options)?;
        for (index, (name, drafting)) in modes.iter_mut().enumerate() {
            let (rate, tokens) = real_round(drafting, count, output, options)?;
            identical[index] &= tokens == plain_tokens;
            if round > 0 {
                ratios[index].push(rate / plain);
                eprintln!("c={count} real {name}: {:.3}x", rate / plain);
            }
        }
    }
    Ok(json!({
        "lookup": {"ratios": ratios[0], "identical": identical[0]},
        "lookup+mtp": {"ratios": ratios[1], "identical": identical[1]},
    }))
}

/// Replay every drafter over `traces` in groups of `count`, overall and per domain.
fn replays(traces: &[Trace], count: usize, cost: PassCost) -> Value {
    let mut results = serde_json::Map::new();
    for drafter in DRAFTERS {
        let mut overall = Totals::default();
        let mut domains: HashMap<&str, Totals> = HashMap::new();
        let mut table = Table::default();
        for group in traces.chunks_exact(count) {
            if !drafter.warm {
                table = Table::default();
            }
            let refs: Vec<&Trace> = group.iter().collect();
            let mut totals = Totals::default();
            replay(&refs, drafter, &mut table, cost, &mut totals);
            for into in [
                Some(&mut overall),
                (count == 1).then(|| domains.entry(group[0].domain).or_default()),
            ]
            .into_iter()
            .flatten()
            {
                into.tokens += totals.tokens;
                into.passes += totals.passes;
                into.seconds += totals.seconds;
                into.baseline_seconds += totals.baseline_seconds;
                into.lookup_drafted += totals.lookup_drafted;
                into.lookup_accepted += totals.lookup_accepted;
                into.table_drafted += totals.table_drafted;
                into.table_accepted += totals.table_accepted;
                into.tree_positions += totals.tree_positions;
                into.tree_hits += totals.tree_hits;
            }
        }
        let mut sorted: Vec<_> = domains
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.value()))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let line = overall.value();
        eprintln!(
            "c={count} {}: modeled {:.3}x, {:.2} tok/pass, lookup {:.2}, table {:.2} over {} drafts, top8 {:.2}",
            drafter.name,
            line["speedup"].as_f64().unwrap_or(0.0),
            line["tokens_per_pass"].as_f64().unwrap_or(0.0),
            line["lookup_acceptance"].as_f64().unwrap_or(0.0),
            line["table_acceptance"].as_f64().unwrap_or(0.0),
            overall.table_drafted,
            line["top8_hit_rate"].as_f64().unwrap_or(0.0),
        );
        results.insert(
            drafter.name.into(),
            json!({"overall": line, "domains": serde_json::Map::from_iter(sorted)}),
        );
    }
    results.into()
}

pub(crate) fn run() -> Result<()> {
    let dir = std::env::var("TUR_BENCH_MODEL")
        .map(PathBuf::from)
        .map_err(|_| invalid("set TUR_BENCH_MODEL to a validated text checkpoint"))?;
    let output = env_sizes("TUR_BENCH_OUTPUT", "256")?
        .first()
        .copied()
        .unwrap_or(256);
    let options = sampling()?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| TurError::Tokenizer(e.to_string()))?;
    let factory = ModelFactory {
        source: ModelSource::LocalPath(dir.clone()),
        revision: "bench".into(),
        format: Some(WeightFormat::Q8_0),
        device: candle_core::Device::new_metal(0)?,
        kv: DEFAULT_KV_LAYOUT,
        recurrent: DEFAULT_RECURRENT_STATE,
        audio: false,
        vision: false,
        video: false,
        expert_cache: None,
        gguf: None,
        sparse: None,
    };
    let (traces, mapped) = record(&factory, &tokenizer, output, options)?;
    eprintln!("candidate text mapped back to its id for {mapped:.4} of tokens");
    let mut runs = Vec::new();
    for count in env_sizes("TUR_BENCH_CONCURRENCY", "1,4")? {
        let (base, per_draft) = measured_line(&factory, count, 128)?;
        let plain = seconds_per_token(&factory, count, 64, options, None)?;
        let reported = seconds_per_token(&factory, count, 64, options, Some(CANDIDATES))?;
        // Throughput counts every request's tokens, so one row's copy is the difference per token.
        let per_row_readback = (reported - plain).max(0.0) * count as f64;
        eprintln!(
            "c={count}: base {:.2} ms, draft {:.2} ms, readback {:.2} ms per row",
            base * 1e3,
            per_draft * 1e3,
            per_row_readback * 1e3
        );
        let cost = PassCost {
            base,
            per_draft,
            per_row_readback,
        };
        let modeled = replays(&traces, count, cost);
        let real = real_comparison(&factory, count, output, options)?;
        eprintln!("c={count} real: {real}");
        runs.push(json!({
            "concurrency": count, "base_seconds": base, "per_draft_seconds": per_draft,
            "per_row_readback_seconds": per_row_readback, "real": real,
            "modeled": modeled,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "benchmark": "drafter", "checkpoint": dir, "weights": "q8_0",
            "output_tokens": output, "sampling": format!("{options:?}"),
            "candidate_text_mapped": mapped, "runs": runs,
        }))?
    );
    Ok(())
}
