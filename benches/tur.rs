use candle_core::{DType, Device};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parking_lot::RwLock;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tur::Downloader;
use tur::ModelFactory;
use tur::backend::InferenceEngine;
use tur::backend::prefix_cache::PrefixCache;
use tur::backend::tokenizer::TokenOutputStream;
use tur::models::{ModelImpl, Qwen35ModelForCausalLM};

const MODEL_ID: &str = "Qwen3-0.6B";
const QUANTIZATION: &str = "Q4_K_M";
const BENCHMARK_SEED: u64 = 299_792_458;
const SAMPLE_LEN: usize = 64;
const TEMPERATURE: Option<f64> = Some(0.0);
const TOP_P: Option<f64> = None;
const REPEAT_PENALTY: f32 = 1.0;
const REPEAT_LAST_N: usize = 64;

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

// Prefix cache benchmark prompts - shared prefix with variations
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

fn benchmark_prompt<T: ModelImpl>(prompt: &str) -> String {
    T::format_prompt(prompt, false)
}

/// Create a model factory for benchmarking
fn create_benchmark_factory() -> ModelFactory<Qwen35ModelForCausalLM> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    ModelFactory::new(
        tur::ModelSource::HuggingFace(MODEL_ID.to_string()),
        Some(QUANTIZATION.to_string()),
        device,
        dtype,
    )
}

/// Benchmark prefill phase: prompt encoding + first forward pass
fn bench_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));

    for (prompt_name, prompt_body) in BENCHMARK_PROMPTS {
        let prompt = benchmark_prompt::<Qwen35ModelForCausalLM>(prompt_body);

        // Measure cold start (model loading)
        let cold_start = Instant::now();
        let factory = create_benchmark_factory();
        let (mut engine, tokenizer) = InferenceEngine::builder(&factory, factory.device().clone())
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE.unwrap_or(0.0))
            .top_p(TOP_P.unwrap_or(1.0))
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .build()
            .expect("failed to build engine");
        let cold_start_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
        println!("Cold start for '{}': {:.2} ms", prompt_name, cold_start_ms);

        let tokenizer_stream = TokenOutputStream::new(tokenizer);

        // Warmup run
        let tokens = tokenizer_stream
            .tokenizer()
            .encode(prompt.as_str(), true)
            .expect("encoding failed")
            .get_ids()
            .to_vec();

        let eos_tokens =
            InferenceEngine::<Qwen35ModelForCausalLM>::get_eos_tokens(&tokenizer_stream)
                .expect("failed to get EOS tokens");

        let warmup_stats = engine
            .run_separated(&tokens, SAMPLE_LEN, eos_tokens)
            .expect("benchmark warmup failed");
        warmup_stats.report(&format!("{}_warmup", prompt_name));

        let throughput = u64::try_from(warmup_stats.prompt_tokens)
            .expect("prompt token count does not fit into u64");
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("prefill", prompt_name),
            &prompt,
            |b, prompt| {
                b.iter_custom(|iters| {
                    let mut total_elapsed = Duration::ZERO;
                    for _ in 0..iters {
                        let tokens = tokenizer_stream
                            .tokenizer()
                            .encode(black_box(prompt.as_str()), true)
                            .expect("encoding failed")
                            .get_ids()
                            .to_vec();

                        let start = Instant::now();
                        let (_first_token, _duration, _, _) =
                            engine.prefill(&tokens).expect("prefill failed");
                        total_elapsed += start.elapsed();
                    }
                    total_elapsed
                });
            },
        );
    }

    group.finish();
}

/// Benchmark decode phase: steady-state token generation with KV cache
fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));
    group.warm_up_time(Duration::from_secs(3));

    for (prompt_name, prompt_body) in BENCHMARK_PROMPTS {
        let prompt = benchmark_prompt::<Qwen35ModelForCausalLM>(prompt_body);

        let factory = create_benchmark_factory();
        let (mut engine, tokenizer) = InferenceEngine::builder(&factory, factory.device().clone())
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE.unwrap_or(0.0))
            .top_p(TOP_P.unwrap_or(1.0))
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .build()
            .expect("failed to build engine");

        let tokenizer_stream = TokenOutputStream::new(tokenizer);
        let tokens = tokenizer_stream
            .tokenizer()
            .encode(prompt.as_str(), true)
            .expect("encoding failed")
            .get_ids()
            .to_vec();

        let eos_tokens =
            InferenceEngine::<Qwen35ModelForCausalLM>::get_eos_tokens(&tokenizer_stream)
                .expect("failed to get EOS tokens");

        // Warmup run
        let warmup_stats = engine
            .run_separated(&tokens, SAMPLE_LEN, eos_tokens)
            .expect("benchmark warmup failed");

        let throughput = u64::try_from(warmup_stats.generated_tokens)
            .expect("generated token count does not fit into u64");
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("decode", prompt_name),
            &prompt,
            |b, prompt| {
                b.iter_custom(|iters| {
                    let mut total_elapsed = Duration::ZERO;
                    for _ in 0..iters {
                        let tokens = tokenizer_stream
                            .tokenizer()
                            .encode(black_box(prompt.as_str()), true)
                            .expect("encoding failed")
                            .get_ids()
                            .to_vec();

                        let stats = engine
                            .run_separated(&tokens, black_box(SAMPLE_LEN), eos_tokens)
                            .expect("benchmark generation failed");
                        // Only measure decode time, not prefill
                        total_elapsed += Duration::from_secs_f64(stats.decode_ms / 1000.0);
                    }
                    total_elapsed
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full pipeline: cold start + prefill + decode
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(50));
    group.warm_up_time(Duration::from_secs(5));

    for (prompt_name, prompt_body) in BENCHMARK_PROMPTS {
        let prompt = benchmark_prompt::<Qwen35ModelForCausalLM>(prompt_body);

        group.bench_with_input(
            BenchmarkId::new("full", prompt_name),
            &prompt,
            |b, prompt| {
                b.iter_custom(|iters| {
                    let mut total_elapsed = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();

                        // Cold start
                        let factory = create_benchmark_factory();
                        let (mut engine, tokenizer) =
                            InferenceEngine::builder(&factory, factory.device().clone())
                                .seed(BENCHMARK_SEED)
                                .temperature(TEMPERATURE.unwrap_or(0.0))
                                .top_p(TOP_P.unwrap_or(1.0))
                                .repeat_penalty(REPEAT_PENALTY)
                                .repeat_last_n(REPEAT_LAST_N)
                                .build()
                                .expect("failed to build engine");

                        let tokenizer_stream = TokenOutputStream::new(tokenizer);
                        let tokens = tokenizer_stream
                            .tokenizer()
                            .encode(black_box(prompt.as_str()), true)
                            .expect("encoding failed")
                            .get_ids()
                            .to_vec();

                        let eos_tokens = InferenceEngine::<Qwen35ModelForCausalLM>::get_eos_tokens(
                            &tokenizer_stream,
                        )
                        .expect("failed to get EOS tokens");

                        // Prefill + Decode
                        let _stats = engine
                            .run_separated(&tokens, black_box(SAMPLE_LEN), eos_tokens)
                            .expect("benchmark generation failed");

                        total_elapsed += start.elapsed();
                    }
                    total_elapsed
                });
            },
        );
    }

    group.finish();
}

/// Benchmark prefix cache: measure cache hit performance vs cache miss
fn bench_prefix_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));
    group.warm_up_time(Duration::from_secs(3));

    // Calculate average tokens per prompt for throughput reporting
    let tokenizer = Tokenizer::from_file(
        Downloader::new(
            Some(MODEL_ID.to_string()),
            None,
            Some(QUANTIZATION.to_string()),
        )
        .prepare_model_weights()
        .expect("failed to prepare model weights")
        .0
        .tokenizer_filename(),
    )
    .expect("failed to load tokenizer");

    let avg_tokens: usize = PREFIX_CACHE_PROMPTS
        .iter()
        .map(|(_, prompt)| {
            let formatted = benchmark_prompt::<Qwen35ModelForCausalLM>(prompt);
            tokenizer
                .encode(formatted.as_str(), true)
                .expect("encoding failed")
                .get_ids()
                .len()
        })
        .sum::<usize>()
        / PREFIX_CACHE_PROMPTS.len();

    let throughput = u64::try_from(avg_tokens).expect("token count does not fit into u64");
    group.throughput(Throughput::Elements(throughput));

    // Benchmark without cache (baseline)
    {
        let factory = create_benchmark_factory();
        let (mut engine, tokenizer) = InferenceEngine::builder(&factory, factory.device().clone())
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE.unwrap_or(0.0))
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .build()
            .expect("failed to build engine");

        let tokenizer_stream = TokenOutputStream::new(tokenizer);

        group.bench_function(BenchmarkId::new("no_cache", "repeated_prefix"), |b| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::ZERO;
                let mut total_tokens = 0usize;

                for i in 0..iters {
                    // Alternate between prompts with shared prefix
                    let prompt_idx = (i % PREFIX_CACHE_PROMPTS.len() as u64) as usize;
                    let prompt = benchmark_prompt::<Qwen35ModelForCausalLM>(
                        PREFIX_CACHE_PROMPTS[prompt_idx].1,
                    );

                    let tokens = tokenizer_stream
                        .tokenizer()
                        .encode(black_box(prompt.as_str()), true)
                        .expect("encoding failed")
                        .get_ids()
                        .to_vec();

                    total_tokens += tokens.len();

                    let start = Instant::now();
                    let (_token, _duration, _hit, _cached) =
                        engine.prefill(&tokens).expect("prefill failed");
                    total_elapsed += start.elapsed();

                    // Clear cache to simulate no caching
                    engine.model_mut().clear_kv_cache();
                }

                // Print throughput statistics
                if iters > 0 && total_elapsed.as_secs_f64() > 0.0 {
                    let tokens_per_sec = total_tokens as f64 / total_elapsed.as_secs_f64();
                    println!(
                        "\nNo cache: {:.2} tokens/sec ({} tokens in {:.3}s)",
                        tokens_per_sec,
                        total_tokens,
                        total_elapsed.as_secs_f64()
                    );
                }

                total_elapsed
            });
        });
    }

    // Benchmark with cache (optimized)
    {
        let factory = create_benchmark_factory();
        let cache = Arc::new(RwLock::new(PrefixCache::new(100, 2048)));
        let (mut engine, tokenizer) = InferenceEngine::builder(&factory, factory.device().clone())
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE.unwrap_or(0.0))
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .with_shared_prefix_cache(cache.clone())
            .build()
            .expect("failed to build engine");

        let tokenizer_stream = TokenOutputStream::new(tokenizer);

        group.bench_function(BenchmarkId::new("with_cache", "repeated_prefix"), |b| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::ZERO;
                let mut cache_hits = 0u64;
                let mut total_cached_tokens = 0usize;
                let mut total_tokens = 0usize;

                for i in 0..iters {
                    // Alternate between prompts with shared prefix
                    let prompt_idx = (i % PREFIX_CACHE_PROMPTS.len() as u64) as usize;
                    let prompt = benchmark_prompt::<Qwen35ModelForCausalLM>(
                        PREFIX_CACHE_PROMPTS[prompt_idx].1,
                    );

                    let tokens = tokenizer_stream
                        .tokenizer()
                        .encode(black_box(prompt.as_str()), true)
                        .expect("encoding failed")
                        .get_ids()
                        .to_vec();

                    total_tokens += tokens.len();

                    let start = Instant::now();
                    let (_token, _duration, hit, cached_tokens) =
                        engine.prefill(&tokens).expect("prefill failed");
                    total_elapsed += start.elapsed();

                    if hit {
                        cache_hits += 1;
                        total_cached_tokens += cached_tokens;
                    }

                    // Clear KV cache but keep prefix cache
                    engine.model_mut().clear_kv_cache();
                }

                // Print cache statistics
                if iters > 0 {
                    let hit_rate = (cache_hits as f64 / iters as f64) * 100.0;
                    let avg_cached = if cache_hits > 0 {
                        total_cached_tokens as f64 / cache_hits as f64
                    } else {
                        0.0
                    };

                    // Calculate and report token throughput
                    let tokens_per_sec = if total_elapsed.as_secs_f64() > 0.0 {
                        total_tokens as f64 / total_elapsed.as_secs_f64()
                    } else {
                        0.0
                    };

                    println!(
                        "\nWith cache: {:.2} tokens/sec ({} tokens in {:.3}s)",
                        tokens_per_sec,
                        total_tokens,
                        total_elapsed.as_secs_f64()
                    );
                    println!(
                        "Prefix cache stats: {:.1}% hit rate, avg {:.1} tokens cached per hit",
                        hit_rate, avg_cached
                    );

                    {
                        let cache_guard = cache.read();
                        let stats = cache_guard.stats();
                        println!(
                            "Cache: {} hits, {} misses, {} tokens reused",
                            stats.hits, stats.misses, stats.total_tokens_reused
                        );
                    }
                }

                total_elapsed
            });
        });
    }

    group.finish();
}

/// Benchmark prefix cache with varying prefix lengths
fn bench_prefix_cache_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache_lengths");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));

    let prefix_lengths = [10, 25, 50, 75, 100];

    for &prefix_len in &prefix_lengths {
        let factory = create_benchmark_factory();
        let cache = Arc::new(RwLock::new(PrefixCache::new(100, 2048)));
        let (mut engine, tokenizer) = InferenceEngine::builder(&factory, factory.device().clone())
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE.unwrap_or(0.0))
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .with_shared_prefix_cache(cache)
            .build()
            .expect("failed to build engine");

        let tokenizer_stream = TokenOutputStream::new(tokenizer);

        // Create a prompt with specific prefix length
        let base_prompt = "You are a helpful AI assistant. ".repeat(prefix_len / 10);
        let prompt1 = format!("{}Question: What is Rust?", base_prompt);
        let prompt2 = format!("{}Question: What is Python?", base_prompt);

        let formatted1 = benchmark_prompt::<Qwen35ModelForCausalLM>(&prompt1);
        let formatted2 = benchmark_prompt::<Qwen35ModelForCausalLM>(&prompt2);

        group.bench_with_input(
            BenchmarkId::new("cache_hit", prefix_len),
            &(formatted1, formatted2),
            |b, (p1, p2)| {
                b.iter_custom(|iters| {
                    let mut total_elapsed = Duration::ZERO;

                    // First request - cache miss
                    let tokens1 = tokenizer_stream
                        .tokenizer()
                        .encode(p1.as_str(), true)
                        .expect("encoding failed")
                        .get_ids()
                        .to_vec();
                    let _ = engine.prefill(&tokens1).expect("prefill failed");
                    engine.model_mut().clear_kv_cache();

                    // Subsequent requests - cache hits
                    for i in 0..iters {
                        let prompt = if i % 2 == 0 { p1 } else { p2 };
                        let tokens = tokenizer_stream
                            .tokenizer()
                            .encode(black_box(prompt.as_str()), true)
                            .expect("encoding failed")
                            .get_ids()
                            .to_vec();

                        let start = Instant::now();
                        let (_token, _duration, _hit, _cached) =
                            engine.prefill(&tokens).expect("prefill failed");
                        total_elapsed += start.elapsed();

                        engine.model_mut().clear_kv_cache();
                    }

                    total_elapsed
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_prefill,
    bench_decode,
    bench_full_pipeline,
    bench_prefix_cache,
    bench_prefix_cache_lengths
);
criterion_main!(benches);
