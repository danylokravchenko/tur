use candle_core::{DType, Device};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tur::Downloader;
use tur::backend::pipeline::InferenceEngine;
use tur::backend::tokenizer::TokenOutputStream;
use tur::models::{ModelImpl, Qwen35ModelForCausalLM};
use tur::weights::VarBuilderX;

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

fn benchmark_prompt<T: ModelImpl>(prompt: &str) -> String {
    T::format_prompt(prompt, false)
}

/// Load model and tokenizer (cold start measurement)
fn load_model_and_tokenizer() -> (Qwen35ModelForCausalLM, Tokenizer, Device) {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let downloader = Downloader::new(
        Some(MODEL_ID.to_string()),
        None,
        Some(QUANTIZATION.to_string()),
    );
    let (paths, gguf) = downloader
        .prepare_model_weights()
        .expect("failed to prepare model weights for benchmark");

    let config_path = paths.get_config_filename();
    let config_content =
        std::fs::read_to_string(&config_path).expect("failed to read benchmark model config");
    let config: tur::models::qwen3::Config =
        serde_json::from_str(&config_content).expect("failed to parse benchmark model config");

    let tokenizer_path = paths.get_tokenizer_filename();
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).expect("failed to load benchmark tokenizer");

    let vb = VarBuilderX::new(&paths, gguf, dtype, &device)
        .expect("failed to create benchmark var builder");
    let model =
        Qwen35ModelForCausalLM::new(&config, vb).expect("failed to initialize benchmark model");

    (model, tokenizer, device)
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
        let (model, tokenizer, device) = load_model_and_tokenizer();
        let cold_start_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
        println!("Cold start for '{}': {:.2} ms", prompt_name, cold_start_ms);

        let mut engine = InferenceEngine::builder(model, device)
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE.unwrap_or(0.0))
            .top_p(TOP_P.unwrap_or(1.0))
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .build();

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
                        let (_first_token, _duration) =
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

        let (model, tokenizer, device) = load_model_and_tokenizer();
        let mut engine = InferenceEngine::builder(model, device)
            .seed(BENCHMARK_SEED)
            .temperature(TEMPERATURE.unwrap_or(0.0))
            .top_p(TOP_P.unwrap_or(1.0))
            .repeat_penalty(REPEAT_PENALTY)
            .repeat_last_n(REPEAT_LAST_N)
            .build();

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
                        let (model, tokenizer, device) = load_model_and_tokenizer();
                        let mut engine = InferenceEngine::builder(model, device)
                            .seed(BENCHMARK_SEED)
                            .temperature(TEMPERATURE.unwrap_or(0.0))
                            .top_p(TOP_P.unwrap_or(1.0))
                            .repeat_penalty(REPEAT_PENALTY)
                            .repeat_last_n(REPEAT_LAST_N)
                            .build();

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

criterion_group!(benches, bench_prefill, bench_decode, bench_full_pipeline);
criterion_main!(benches);
