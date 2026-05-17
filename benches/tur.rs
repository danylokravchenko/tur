use candle_core::{DType, Device};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use parking_lot::RwLock;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tur::ModelFactory;
use tur::backend::InferenceEngine;
use tur::backend::prefix_cache::PrefixCache;
use tur::backend::tokenizer::TokenOutputStream;
use tur::models::kv_cache::{BlockAllocator, PagedKvCache};
use tur::models::{ModelImpl, Qwen35ModelForCausalLM};
use uuid::Uuid;

const MODEL_ID: &str = "Qwen3-0.6B";
const QUANTIZATION: &str = "Q4_K_M";
const BENCHMARK_SEED: u64 = 299_792_458;
const SAMPLE_LEN: usize = 64;
const TEMPERATURE: f64 = 0.0;
const REPEAT_PENALTY: f32 = 1.0;
const REPEAT_LAST_N: usize = 64;
const QWEN3_NUM_LAYERS: usize = 28;
const BATCH_SIZES: [usize; 3] = [1, 2, 4];

const BENCHMARK_PROMPTS: [(&str, &str); 4] = [
    (
        "short",
        "Describe Rust token generation in exactly three short sentences. Mention prompt encoding, \
         cached decoding, and EOS termination.",
    ),
    (
        "medium",
        "Explain how an autoregressive Rust inference pipeline works. In exactly five short \
         sentences, mention loading model weights, tokenizer encoding, prompt prefill, KV-cache \
         reuse during decoding, and token sampling. Do not use bullet points.",
    ),
    (
        "long",
        "You are evaluating inference performance for a causal language model implemented in Rust. \
         Provide a concise plain-text explanation in exactly six short sentences. Cover these \
         topics in order: loading weights, tokenizer setup, prompt encoding, first-pass prefill \
         over the full context, iterative one-token decoding with cache reuse, and stopping when \
         an EOS-style token is produced. Do not use markdown, bullet points, or dialogue.",
    ),
    (
        "structured",
        "Return exactly four numbered lines. Line 1 must summarize model loading. Line 2 must \
         summarize tokenization. Line 3 must summarize cached autoregressive decoding. Line 4 must \
         summarize EOS-based stopping.",
    ),
];

const PREFIX_CACHE_PROMPTS: [(&str, &str); 3] = [
    (
        "base",
        "You are a helpful AI assistant. Please answer the following question: What is Rust?",
    ),
    (
        "variant1",
        "You are a helpful AI assistant. Please answer the following question: What is Python?",
    ),
    (
        "variant2",
        "You are a helpful AI assistant. Please answer the following question: What is JavaScript?",
    ),
];

// --- Helpers ---

fn create_benchmark_factory() -> ModelFactory<Qwen35ModelForCausalLM> {
    ModelFactory::new(
        tur::ModelSource::HuggingFace(MODEL_ID.to_string()),
        Some(QUANTIZATION.to_string()),
        Device::Cpu,
        DType::F32,
    )
}

fn build_engine() -> (InferenceEngine<Qwen35ModelForCausalLM>, TokenOutputStream) {
    build_engine_inner(None)
}

fn build_engine_with_cache(
    cache: Arc<RwLock<PrefixCache>>,
) -> (InferenceEngine<Qwen35ModelForCausalLM>, TokenOutputStream) {
    build_engine_inner(Some(cache))
}

fn build_engine_inner(
    cache: Option<Arc<RwLock<PrefixCache>>>,
) -> (InferenceEngine<Qwen35ModelForCausalLM>, TokenOutputStream) {
    let factory = create_benchmark_factory();
    let mut builder = InferenceEngine::builder(&factory, factory.device().clone())
        .seed(BENCHMARK_SEED)
        .temperature(TEMPERATURE)
        .repeat_penalty(REPEAT_PENALTY)
        .repeat_last_n(REPEAT_LAST_N);
    if let Some(c) = cache {
        builder = builder.with_shared_prefix_cache(c);
    }
    let (engine, tokenizer, _) = builder.build().expect("failed to build engine");
    (engine, TokenOutputStream::new(tokenizer))
}

fn encode_tokens(ts: &TokenOutputStream, text: &str) -> Vec<u32> {
    ts.tokenizer()
        .encode(text, true)
        .expect("encoding failed")
        .get_ids()
        .to_vec()
}

fn format_prompt(prompt: &str) -> String {
    Qwen35ModelForCausalLM::format_prompt(prompt, false)
}

fn configure_group(
    group: &mut BenchmarkGroup<WallTime>,
    samples: usize,
    measure_secs: u64,
    warmup_secs: u64,
) {
    group.sample_size(samples);
    group.measurement_time(Duration::from_secs(measure_secs));
    group.warm_up_time(Duration::from_secs(warmup_secs));
}

fn make_paged_caches(
    allocator: &Arc<RwLock<BlockAllocator>>,
    num_requests: usize,
) -> Vec<Vec<PagedKvCache>> {
    (0..num_requests)
        .map(|_| {
            (0..QWEN3_NUM_LAYERS)
                .map(|_| PagedKvCache::new(allocator.clone(), 2))
                .collect()
        })
        .collect()
}

// --- Benchmarks ---

fn bench_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill");
    configure_group(&mut group, 10, 30, 3);

    for (prompt_name, prompt_body) in BENCHMARK_PROMPTS {
        let prompt = format_prompt(prompt_body);
        let (mut engine, ts) = build_engine();
        let eos_tokens = InferenceEngine::<Qwen35ModelForCausalLM>::get_eos_tokens(&ts)
            .expect("failed to get EOS tokens");

        let cold_start = Instant::now();
        println!(
            "Cold start for '{}': {:.2} ms",
            prompt_name,
            cold_start.elapsed().as_secs_f64() * 1000.0
        );

        // Warmup
        let warmup_tokens = encode_tokens(&ts, &prompt);
        let warmup_stats = engine
            .run_separated(&warmup_tokens, SAMPLE_LEN, eos_tokens)
            .expect("warmup failed");
        warmup_stats.report(&format!("{}_warmup", prompt_name));
        group.throughput(Throughput::Elements(warmup_stats.prompt_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("prefill", prompt_name),
            &prompt,
            |b, prompt| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let tokens = encode_tokens(&ts, black_box(prompt.as_str()));
                        let start = Instant::now();
                        engine.prefill(&tokens).expect("prefill failed");
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    configure_group(&mut group, 10, 40, 3);

    for (prompt_name, prompt_body) in BENCHMARK_PROMPTS {
        let prompt = format_prompt(prompt_body);
        let (mut engine, ts) = build_engine();
        let eos_tokens = InferenceEngine::<Qwen35ModelForCausalLM>::get_eos_tokens(&ts)
            .expect("failed to get EOS tokens");

        let tokens = encode_tokens(&ts, &prompt);
        let warmup_stats = engine
            .run_separated(&tokens, SAMPLE_LEN, eos_tokens)
            .expect("warmup failed");
        group.throughput(Throughput::Elements(warmup_stats.generated_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("decode", prompt_name),
            &prompt,
            |b, prompt| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let tokens = encode_tokens(&ts, black_box(prompt.as_str()));
                        let stats = engine
                            .run_separated(&tokens, black_box(SAMPLE_LEN), eos_tokens)
                            .expect("generation failed");
                        total += Duration::from_secs_f64(stats.decode_ms / 1000.0);
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    configure_group(&mut group, 10, 50, 5);

    for (prompt_name, prompt_body) in BENCHMARK_PROMPTS {
        let prompt = format_prompt(prompt_body);

        group.bench_with_input(
            BenchmarkId::new("full", prompt_name),
            &prompt,
            |b, prompt| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();

                        let factory = create_benchmark_factory();
                        let (mut engine, tokenizer, _) =
                            InferenceEngine::builder(&factory, factory.device().clone())
                                .seed(BENCHMARK_SEED)
                                .temperature(TEMPERATURE)
                                .repeat_penalty(REPEAT_PENALTY)
                                .repeat_last_n(REPEAT_LAST_N)
                                .build()
                                .expect("failed to build engine");
                        let ts = TokenOutputStream::new(tokenizer);
                        let tokens = encode_tokens(&ts, black_box(prompt.as_str()));
                        let eos_tokens =
                            InferenceEngine::<Qwen35ModelForCausalLM>::get_eos_tokens(&ts)
                                .expect("failed to get EOS tokens");

                        engine
                            .run_separated(&tokens, black_box(SAMPLE_LEN), eos_tokens)
                            .expect("generation failed");

                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_prefix_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache");
    configure_group(&mut group, 10, 40, 3);

    // No-cache baseline
    {
        let (mut engine, ts) = build_engine();
        let avg_tokens: usize = PREFIX_CACHE_PROMPTS
            .iter()
            .map(|(_, prompt)| encode_tokens(&ts, &format_prompt(prompt)).len())
            .sum::<usize>()
            / PREFIX_CACHE_PROMPTS.len();
        group.throughput(Throughput::Elements(avg_tokens as u64));

        group.bench_function(BenchmarkId::new("no_cache", "repeated_prefix"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                let mut total_tokens = 0usize;

                for i in 0..iters {
                    let (_, prompt) =
                        PREFIX_CACHE_PROMPTS[(i % PREFIX_CACHE_PROMPTS.len() as u64) as usize];
                    let tokens = encode_tokens(&ts, black_box(&format_prompt(prompt)));
                    total_tokens += tokens.len();

                    let start = Instant::now();
                    engine.prefill(&tokens).expect("prefill failed");
                    total += start.elapsed();
                    engine.model_mut().clear_kv_cache();
                }

                if iters > 0 && total.as_secs_f64() > 0.0 {
                    println!(
                        "\nNo cache: {:.2} tokens/sec ({} tokens in {:.3}s)",
                        total_tokens as f64 / total.as_secs_f64(),
                        total_tokens,
                        total.as_secs_f64()
                    );
                }
                total
            });
        });
    }

    // With prefix cache
    {
        let cache = Arc::new(RwLock::new(PrefixCache::new(100, 2048)));
        let (mut engine, ts) = build_engine_with_cache(cache.clone());

        group.bench_function(BenchmarkId::new("with_cache", "repeated_prefix"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                let mut cache_hits = 0u64;
                let mut total_cached_tokens = 0usize;
                let mut total_tokens = 0usize;

                for i in 0..iters {
                    let (_, prompt) =
                        PREFIX_CACHE_PROMPTS[(i % PREFIX_CACHE_PROMPTS.len() as u64) as usize];
                    let tokens = encode_tokens(&ts, black_box(&format_prompt(prompt)));
                    total_tokens += tokens.len();

                    let start = Instant::now();
                    let (_token, _duration, hit, cached_tokens) =
                        engine.prefill(&tokens).expect("prefill failed");
                    total += start.elapsed();

                    if hit {
                        cache_hits += 1;
                        total_cached_tokens += cached_tokens;
                    }
                    engine.model_mut().clear_kv_cache();
                }

                if iters > 0 {
                    let hit_rate = (cache_hits as f64 / iters as f64) * 100.0;
                    let avg_cached = if cache_hits > 0 {
                        total_cached_tokens as f64 / cache_hits as f64
                    } else {
                        0.0
                    };
                    println!(
                        "\nWith cache: {:.2} tokens/sec ({} tokens in {:.3}s)",
                        total_tokens as f64 / total.as_secs_f64(),
                        total_tokens,
                        total.as_secs_f64()
                    );
                    println!(
                        "Prefix cache stats: {:.1}% hit rate, avg {:.1} tokens cached per hit",
                        hit_rate, avg_cached
                    );
                    let stats = cache.read().stats().clone();
                    println!(
                        "Cache: {} hits, {} misses, {} tokens reused",
                        stats.hits, stats.misses, stats.total_tokens_reused
                    );
                }
                total
            });
        });
    }

    group.finish();
}

fn bench_prefix_cache_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache_lengths");
    configure_group(&mut group, 10, 30, 3);

    for &prefix_len in &[10usize, 25, 50, 75, 100] {
        let cache = Arc::new(RwLock::new(PrefixCache::new(100, 2048)));
        let (mut engine, ts) = build_engine_with_cache(cache);

        let base = "You are a helpful AI assistant. ".repeat(prefix_len / 10);
        let p1 = format_prompt(&format!("{}Question: What is Rust?", base));
        let p2 = format_prompt(&format!("{}Question: What is Python?", base));

        group.bench_with_input(
            BenchmarkId::new("cache_hit", prefix_len),
            &(p1, p2),
            |b, (p1, p2)| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;

                    // Prime the cache
                    let tokens1 = encode_tokens(&ts, p1);
                    engine.prefill(&tokens1).expect("prefill failed");
                    engine.model_mut().clear_kv_cache();

                    for i in 0..iters {
                        let prompt = if i % 2 == 0 { p1 } else { p2 };
                        let tokens = encode_tokens(&ts, black_box(prompt.as_str()));
                        let start = Instant::now();
                        engine.prefill(&tokens).expect("prefill failed");
                        total += start.elapsed();
                        engine.model_mut().clear_kv_cache();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch prefill: multiple requests processed together, with and without prefix cache.
///
/// The no-cache path uses the shared model KV cache (non-paged). The with-cache path
/// uses per-request paged KV caches so each request can independently restore cached
/// prefix state without interference.
fn bench_batch_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_prefill");
    configure_group(&mut group, 10, 40, 3);

    for &batch_size in &BATCH_SIZES {
        // Without prefix cache: shared model KV cache, cleared each iteration
        {
            let (mut engine, ts) = build_engine();

            let batch_token_count: usize = (0..batch_size)
                .map(|i| {
                    let (_, prompt) = PREFIX_CACHE_PROMPTS[i % PREFIX_CACHE_PROMPTS.len()];
                    encode_tokens(&ts, &format_prompt(prompt)).len()
                })
                .sum();
            group.throughput(Throughput::Elements(batch_token_count as u64));

            group.bench_with_input(
                BenchmarkId::new("no_cache", batch_size),
                &batch_size,
                |b, &batch_size| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        let mut total_tokens = 0usize;
                        for _ in 0..iters {
                            let batch: Vec<(Uuid, Vec<u32>, usize)> = (0..batch_size)
                                .map(|i| {
                                    let (_, prompt) =
                                        PREFIX_CACHE_PROMPTS[i % PREFIX_CACHE_PROMPTS.len()];
                                    let text = format_prompt(prompt);
                                    (Uuid::new_v4(), encode_tokens(&ts, black_box(&text)), 0)
                                })
                                .collect();
                            total_tokens += batch.iter().map(|(_, t, _)| t.len()).sum::<usize>();

                            let start = Instant::now();
                            engine
                                .prefill_batch(&batch, None)
                                .expect("prefill_batch failed");
                            total += start.elapsed();
                            engine.model_mut().clear_kv_cache();
                        }
                        if iters > 0 && total.as_secs_f64() > 0.0 {
                            println!(
                                "\nBatch {} no_cache: {:.2} tokens/sec ({} tokens in {:.3}s)",
                                batch_size,
                                total_tokens as f64 / total.as_secs_f64(),
                                total_tokens,
                                total.as_secs_f64()
                            );
                        }
                        total
                    });
                },
            );
        }

        // With prefix cache: paged KV caches per request, prefix cache warmed before measurement
        {
            let cache = Arc::new(RwLock::new(PrefixCache::new(100, 2048)));
            let (mut engine, ts) = build_engine_with_cache(cache.clone());
            // Each request needs up to ~100 tokens / 16 per block = ~7 blocks * 28 layers * max_batch
            let allocator = Arc::new(RwLock::new(BlockAllocator::new(1000, 16)));

            // Warm up the prefix cache with all prompt variants
            for (_, prompt) in &PREFIX_CACHE_PROMPTS {
                let text = format_prompt(prompt);
                let tokens = encode_tokens(&ts, &text);
                let mut caches = make_paged_caches(&allocator, 1);
                engine
                    .prefill_batch(&[(Uuid::new_v4(), tokens, 0)], Some(&mut caches))
                    .expect("cache warmup failed");
            }

            let batch_token_count: usize = (0..batch_size)
                .map(|i| {
                    let (_, prompt) = PREFIX_CACHE_PROMPTS[i % PREFIX_CACHE_PROMPTS.len()];
                    encode_tokens(&ts, &format_prompt(prompt)).len()
                })
                .sum();
            group.throughput(Throughput::Elements(batch_token_count as u64));

            group.bench_with_input(
                BenchmarkId::new("with_cache", batch_size),
                &batch_size,
                |b, &batch_size| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        let mut total_tokens = 0usize;
                        for _ in 0..iters {
                            let batch: Vec<(Uuid, Vec<u32>, usize)> = (0..batch_size)
                                .map(|i| {
                                    let (_, prompt) = PREFIX_CACHE_PROMPTS[i % PREFIX_CACHE_PROMPTS.len()];
                                    let text = format_prompt(prompt);
                                    (Uuid::new_v4(), encode_tokens(&ts, black_box(&text)), 0)
                                })
                                .collect();
                            total_tokens += batch.iter().map(|(_, t, _)| t.len()).sum::<usize>();

                            let mut paged = make_paged_caches(&allocator, batch_size);
                            let start = Instant::now();
                            engine
                                .prefill_batch(&batch, Some(&mut paged))
                                .expect("prefill_batch failed");
                            total += start.elapsed();
                            // paged caches drop here, freeing blocks back to allocator
                        }
                        if iters > 0 && total.as_secs_f64() > 0.0 {
                            let total_cached = cache.read().stats().total_tokens_reused;
                            println!(
                                "\nBatch {} with_cache: {:.2} tokens/sec ({} tokens in {:.3}s, {} reused from cache)",
                                batch_size,
                                total_tokens as f64 / total.as_secs_f64(),
                                total_tokens,
                                total.as_secs_f64(),
                                total_cached,
                            );
                        }
                        total
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark chunked prefill against single-shot prefill.
///
/// For each prompt, this benchmark measures:
/// - `single_shot`: one `prefill_batch` call with all prompt tokens at once.
/// - `chunk_N`: N tokens per forward pass, multiple calls to complete the prompt.
///
/// The meaningful comparison is **total time to finish the full prefill**.
/// Chunking adds a small dispatch overhead (O(prompt_len/chunk_size) extra
/// calls) but caps the per-iteration attention matrix at O(chunk²) instead of
/// O(prompt²), which reduces peak memory and lets decode requests interleave.
///
/// Expect total chunked time to be within ~5–15 % of single-shot; if it is,
/// chunked prefill is a safe trade-off for the batching workload.
fn bench_chunked_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_prefill");
    configure_group(&mut group, 10, 40, 3);

    // Chunk sizes to compare.  Values smaller than the prompt force real chunking;
    // a value larger than any prompt is equivalent to single-shot and acts as a
    // sanity check that overhead is negligible at large chunk sizes.
    const CHUNK_SIZES: [usize; 3] = [16, 32, 64];

    for (prompt_name, prompt_body) in BENCHMARK_PROMPTS {
        let factory = create_benchmark_factory();
        let (mut engine, tokenizer, _) = InferenceEngine::builder(&factory, factory.device().clone())
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE)
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .build()
            .expect("failed to build engine");
        let ts = TokenOutputStream::new(tokenizer);
        let allocator = Arc::new(RwLock::new(BlockAllocator::new(4096, 16)));

        let prompt = format_prompt(prompt_body);
        let tokens = encode_tokens(&ts, &prompt);
        let prompt_len = tokens.len();

        // Warmup: one full run so model internals reach steady state.
        {
            let mut caches = make_paged_caches(&allocator, 1);
            engine
                .prefill_batch(&[(Uuid::new_v4(), tokens.clone(), 0)], Some(&mut caches))
                .expect("warmup failed");
        }

        group.throughput(Throughput::Elements(prompt_len as u64));

        // ── Single-shot baseline ─────────────────────────────────────────────
        group.bench_with_input(
            BenchmarkId::new(prompt_name, "single_shot"),
            &tokens,
            |b, tokens| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut caches = make_paged_caches(&allocator, 1);
                        let start = Instant::now();
                        engine
                            .prefill_batch(
                                &[(Uuid::new_v4(), tokens.clone(), 0)],
                                Some(&mut caches),
                            )
                            .expect("single-shot prefill failed");
                        total += start.elapsed();
                        // caches drop here, freeing paged blocks back to allocator
                    }
                    total
                });
            },
        );

        // ── Chunked variants ─────────────────────────────────────────────────
        for &chunk_size in &CHUNK_SIZES {
            let n_chunks = prompt_len.div_ceil(chunk_size);
            let label = format!("chunk_{chunk_size}");

            group.bench_with_input(
                BenchmarkId::new(prompt_name, &label),
                tokens.as_slice(),
                |b, tokens| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let id = Uuid::new_v4();
                            let mut caches = make_paged_caches(&allocator, 1);
                            let start = Instant::now();
                            let mut offset = 0;
                            while offset < tokens.len() {
                                let end = (offset + chunk_size).min(tokens.len());
                                let chunk = tokens[offset..end].to_vec();
                                engine
                                    .prefill_batch(&[(id, chunk, offset)], Some(&mut caches))
                                    .expect("chunk prefill failed");
                                offset = end;
                            }
                            total += start.elapsed();
                        }

                        // Print overhead summary on last call to help read results.
                        if iters > 0 && total.as_secs_f64() > 0.0 {
                            println!(
                                "\n{prompt_name} chunk_{chunk_size}: \
                                 {n_chunks} chunks × {chunk_size} tok, \
                                 {:.2} ms/iter",
                                total.as_secs_f64() * 1000.0 / iters as f64,
                            );
                        }
                        total
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_prefill,
    bench_decode,
    bench_full_pipeline,
    bench_prefix_cache,
    bench_prefix_cache_lengths,
    bench_batch_prefill,
    bench_chunked_prefill,
);
criterion_main!(benches);
