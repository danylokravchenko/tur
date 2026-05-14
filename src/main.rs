use candle_core::{DType, Device};
use clap::Parser;
use tokenizers::Tokenizer;
use tracing::level_filters::LevelFilter;
use tracing::{debug, info, trace};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, fmt};
use tur::models::Qwen35ModelForCausalLM;
use tur::weights::VarBuilderX;
use tur::{Downloader, ProgressReporter, Result, TextGeneration, TurError};

const DEFAULT_PROMPT: &str = "Who are you?";

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
}

fn format_prompt(prompt: &str, thinking: bool) -> String {
    let think_tag = if thinking { " /think" } else { " /no_think" };
    format!("<|im_start|>user\n{prompt}{think_tag}<|im_end|>\n<|im_start|>assistant\n")
}

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    info!("Qwen 3.5 Model - Clean Implementation");

    let device = Device::new_metal(0)?;
    //let device = Device::Cpu;
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

    // Log download strategy
    if let Some(ref model_id) = args.model_id {
        let model_name = model_id.split('/').next_back().unwrap_or(model_id);
        if let Some(ref quant) = args.quantization {
            info!("Downloading quantized GGUF model:");
            info!("  - Model: {}", model_name);
            info!("  - Quantization: {}", quant);
            info!("  - Config/tokenizer from: Qwen/{}", model_name);
            info!("  - GGUF weights from: unsloth/{}-GGUF", model_name);
        } else {
            info!("Downloading full-precision SafeTensors model:");
            info!("  - Model: {}", model_name);
            info!("  - Repo: Qwen/{}", model_name);
        }
    }

    let downloader = Downloader::new(args.model_id, args.weight_path, args.quantization);
    let (paths, gguf) = downloader.prepare_model_weights()?;

    // Load config from downloaded config.json
    let config_path = paths.get_config_filename();
    trace!("Reading config file: {}", config_path.display());
    let config_content = std::fs::read_to_string(&config_path)?;

    // Parse the JSON and extract text_config
    let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
    let config: tur::models::qwen3::Config = serde_json::from_value(config_json.clone())?;

    debug!("Model Config: {:?}", config);

    let weight_files = paths.get_weight_filenames();
    if gguf {
        debug!("Loading GGUF quantized model from: {:?}", weight_files);
        if device.is_cpu() {
            debug!(
                "CPU mode: Linear layers will use quantized weights (QMatMul) for memory efficiency"
            );
        } else {
            debug!(
                "GPU/Metal mode: Dequantizing linear layers to {:?} for better performance",
                dtype
            );
        }
        debug!("Embeddings and norms will use {:?}", dtype);
    } else {
        debug!(
            "Loading full-precision SafeTensors model from: {:?}",
            weight_files
        );
        debug!("All operations will use {:?}", dtype);
    }

    let tokenizer_path = paths.get_tokenizer_filename();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    debug!("Loaded Tokenizer from: {:?}", tokenizer_path);

    // Create progress reporter
    let progress = ProgressReporter::new();

    // VarBuilderX automatically handles both quantized (GGUF) and full-precision (SafeTensors)
    let vb = VarBuilderX::new(&paths, gguf, dtype, &device)?;
    let model = Qwen35ModelForCausalLM::new_with_progress(&config, vb, Some(&progress))?;

    if gguf {
        debug!("✓ Loaded quantized Qwen 3.5 model (GGUF format)");
    } else {
        debug!(
            "✓ Loaded full-precision Qwen 3.5 with {} safetensor shard(s)",
            weight_files.len()
        );
    }
    let mut builder = TextGeneration::builder(model, tokenizer, device.clone())
        .seed(args.seed)
        .repeat_penalty(args.repeat_penalty)
        .repeat_last_n(args.repeat_last_n)
        .progress(progress);

    if let Some(temp) = args.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(top_p) = args.top_p {
        builder = builder.top_p(top_p);
    }

    let mut pipeline = builder.build();
    debug!("✓ Model is initialized and ready for inference");

    let prompt = format_prompt(DEFAULT_PROMPT, args.thinking);
    trace!("formatted prompt: {}", &prompt);

    pipeline.run(&prompt, args.sample_len)?;

    Ok(())
}
