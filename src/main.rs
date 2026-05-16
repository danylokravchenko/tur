use candle_core::{DType, Device};
use clap::Parser;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::level_filters::LevelFilter;
use tracing::{debug, info, trace};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, fmt};
use tur::backend::pipeline::GenerationRequest;
use tur::backend::prefix_cache::PrefixCache;
use tur::{ProgressReporter, Result, TextGeneration, TurError};

const DEFAULT_PROMPT: &str = "Who are you?";

#[allow(clippy::print_stdout)]
fn print_bunner() {
    let banner = match std::fs::read_to_string("assets/banner.txt") {
        Ok(banner) => banner,
        Err(_) => return,
    };

    // Electric Blue Neon
    // let colors = [
    //     "\x1b[38;5;21m",
    //     "\x1b[38;5;27m",
    //     "\x1b[38;5;33m",
    //     "\x1b[38;5;39m",
    //     "\x1b[38;5;45m",
    //     "\x1b[38;5;51m",
    //     "\x1b[38;5;87m",
    //     "\x1b[38;5;123m",
    //     "\x1b[38;5;159m",
    //     "\x1b[38;5;195m",
    //     "\x1b[38;5;231m",
    //     "\x1b[38;5;195m",
    //     "\x1b[38;5;159m",
    //     "\x1b[38;5;123m",
    //     "\x1b[38;5;87m",
    //     "\x1b[38;5;51m",
    //     "\x1b[38;5;45m",
    //     "\x1b[38;5;39m",
    //     "\x1b[38;5;33m",
    //     "\x1b[38;5;27m",
    // ];

    // Blade Runner (Orange→Purple→Blue)
    let colors = [
        "\x1b[38;5;208m",
        "\x1b[38;5;214m",
        "\x1b[38;5;220m",
        "\x1b[38;5;226m",
        "\x1b[38;5;220m",
        "\x1b[38;5;214m",
        "\x1b[38;5;208m",
        "\x1b[38;5;202m",
        "\x1b[38;5;196m",
        "\x1b[38;5;197m",
        "\x1b[38;5;198m",
        "\x1b[38;5;165m",
        "\x1b[38;5;129m",
        "\x1b[38;5;93m",
        "\x1b[38;5;57m",
        "\x1b[38;5;21m",
        "\x1b[38;5;27m",
        "\x1b[38;5;33m",
        "\x1b[38;5;39m",
        "\x1b[38;5;45m",
    ];
    let reset = "\x1b[0m";
    let lines: Vec<&str> = banner.lines().collect();

    for line in lines {
        let chars: Vec<char> = line.chars().collect();
        let line_len = chars.len();

        for (i, ch) in chars.iter().enumerate() {
            // Calculate color index based on position
            let color_idx = (i * (colors.len() - 1)) / line_len;
            print!("{}{}", colors[color_idx], ch);
        }
        println!("{reset}");
    }
}

fn init_tracing() {
    let registry = tracing_subscriber::registry();

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(LevelFilter::TRACE.to_string()))
        .add_directive("ureq=error".parse().unwrap())
        .add_directive("tokenizers=error".parse().unwrap())
        .add_directive("rustls=error".parse().unwrap());

    let console_layer = fmt::layer()
        .compact()
        .with_file(false)
        .with_line_number(false)
        .with_thread_names(true)
        .with_thread_ids(true)
        .with_target(true)
        .with_filter(env_filter.clone());

    let subscriber = registry.with(console_layer);

    subscriber.try_init().unwrap();
}

#[derive(Debug, Parser)]
#[command(author, version, about = "Qwen 3.5 Model - Clean Implementation", long_about = None)]
struct Args {
    /// Simplified model ID (e.g., "Qwen3-0.6B" or full "Qwen/Qwen3-0.6B").
    /// Config and tokenizer are always downloaded from the main repo.
    /// If --quantization is specified, GGUF weights are downloaded from unsloth/<model>-GGUF
    #[arg(long, env = "HF_MODEL_ID")]
    model_id: Option<String>,

    /// Local path to a directory containing model weights (safetensors or GGUF)
    #[arg(long, env = "MODEL_WEIGHT_PATH")]
    weight_path: Option<String>,

    /// Quantization level for GGUF models (e.g., Q4_K_M, Q5_K_M, Q8_0).
    /// When specified, downloads GGUF weights from unsloth repo instead of SafeTensors.
    /// Example: --model-id Qwen3-0.6B --quantization Q4_K_M
    #[arg(long, short = 'q', env = "QUANTIZATION")]
    quantization: Option<String>,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, default_value = "0.7")]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value = "0.95")]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long, default_value = "32")]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Enable thinking/reasoning mode (allows model to show its reasoning process)
    #[arg(long)]
    thinking: bool,

    /// Enable prefix cache optimization
    #[arg(long)]
    prefix_cache: bool,

    /// Maximum number of cached prefixes (default: 100)
    #[arg(long, default_value_t = 100)]
    cache_max_entries: usize,

    /// Maximum token length for cached prefixes (default: 2048)
    #[arg(long, default_value_t = 2048)]
    cache_max_tokens: usize,
}

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    print_bunner();

    info!("Qwen 3.5 Model - Clean Implementation");

    let device = Device::new_metal(0)?;

    // DType for non-quantized operations (embeddings, norms, activations)
    // When using GGUF quantized models, linear layer weights remain quantized
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };

    debug!("Device: {:?}", device);
    debug!("DType for non-quantized ops: {:?}", dtype);
    debug!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    if args.model_id.is_none() && args.weight_path.is_none() {
        return Err(TurError::Other(
            "Please provide a weight source:\n\
             Examples:\n\
             - Full precision SafeTensors:\n\
               --model-id Qwen3-0.6B\n\
             - Quantized GGUF (auto-downloads from unsloth):\n\
               --model-id Qwen3-0.6B --quantization Q4_K_M\n\
             - Local weights:\n\
               --weight-path /path/to/model"
                .to_string(),
        ));
    }

    // Create progress reporter
    let progress = ProgressReporter::new();

    // Use ModelFactory to create the model and tokenizer
    let factory = tur::ModelFactory::new(
        args.model_id,
        args.weight_path,
        args.quantization,
        device.clone(),
        dtype,
    );
    let (model, tokenizer) = factory.create_model(Some(&progress))?;

    // Build inference engine with all parameters
    let mut engine_builder = tur::backend::InferenceEngine::builder(model, device.clone())
        .seed(args.seed)
        .repeat_penalty(args.repeat_penalty)
        .repeat_last_n(args.repeat_last_n);

    if let Some(temp) = args.temperature {
        engine_builder = engine_builder.temperature(temp);
    }
    if let Some(top_p) = args.top_p {
        engine_builder = engine_builder.top_p(top_p);
    }

    // Enable prefix cache if requested
    if args.prefix_cache {
        let cache = Arc::new(RwLock::new(PrefixCache::new(
            args.cache_max_entries,
            args.cache_max_tokens,
        )));
        info!(
            "✓ Prefix cache enabled (max_entries: {}, max_tokens: {})",
            args.cache_max_entries, args.cache_max_tokens
        );
        engine_builder = engine_builder.with_shared_prefix_cache(cache);
    }

    let engine = engine_builder.build();

    // Create text generation pipeline from engine
    let mut pipeline = TextGeneration::from_engine(engine, tokenizer, Some(progress));
    debug!("✓ Model is initialized and ready for inference");

    let prompt = pipeline.format_prompt(DEFAULT_PROMPT, args.thinking);
    trace!("formatted prompt: {}", &prompt);

    let request = GenerationRequest::new(prompt, args.sample_len);
    pipeline.run(&request)?;

    Ok(())
}
