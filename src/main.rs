use candle_core::{DType, Device};
use clap::Parser;
use tracing::{debug, info};
use tur::backend::pipeline::GenerationRequest;
use tur::backend::tools::ToolDefinition;
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

    /// Path to a JSON file containing tool definitions.
    /// The file must be a JSON array of objects with "name", "description",
    /// and "parameters" (JSON Schema) fields.
    ///
    /// Example file content:
    ///   [{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}]
    #[arg(long, value_name = "FILE")]
    tools: Option<String>,

    /// Enable prefix cache optimization
    #[arg(long)]
    prefix_cache: bool,

    /// Maximum number of cached prefixes (default: 100)
    #[arg(long, default_value_t = 100)]
    cache_max_entries: usize,

    /// Maximum token length for cached prefixes (default: 2048)
    #[arg(long, default_value_t = 2048)]
    cache_max_tokens: usize,

    /// Enable continuous batching for concurrent request processing
    #[arg(long)]
    enable_batching: bool,

    /// Maximum batch size for concurrent requests (default: 16)
    #[arg(long, default_value_t = 16)]
    max_batch_size: usize,

    /// Maximum prefill batch size (default: 8)
    #[arg(long, default_value_t = 8)]
    max_prefill_batch: usize,

    /// Maximum decode batch size (default: 16)
    #[arg(long, default_value_t = 16)]
    max_decode_batch: usize,

    /// Scheduling policy: fcfs, priority, or shortest_job_first (default: fcfs)
    #[arg(long, default_value = "fcfs")]
    scheduling_policy: String,

    /// Split each prompt into chunks of this many tokens during prefill.
    /// Caps per-iteration attention memory from O(prompt²) to O(chunk²) and
    /// allows decode requests to interleave with in-progress prefills.
    /// Only effective when --enable-batching is set.
    /// Omit to process the full prompt in one shot (default behaviour).
    #[arg(long)]
    prefill_chunk_size: Option<usize>,
}

fn main() -> Result<()> {
    tur::shared::init_tracing();
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

    let source = match (args.model_id, args.weight_path) {
        (Some(model_id), _) => tur::ModelSource::HuggingFace(model_id),
        (None, Some(path)) => tur::ModelSource::LocalPath(path),
        (None, None) => {
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
    };

    // Create progress reporter
    let progress = ProgressReporter::new();

    // Use ModelFactory to create the model and tokenizer
    let factory = tur::ModelFactory::<tur::models::Qwen35ModelForCausalLM>::new(
        source,
        args.quantization,
        device.clone(),
        dtype,
    );

    // Build inference engine with all parameters.
    // on_token streams each decoded token; here we route it through ProgressReporter so
    // text is printed above the progress bar. Replace this closure to send tokens
    // elsewhere (e.g. an HTTP stream, a channel, a file).
    let progress_for_tokens = progress.clone();
    let mut pipeline_builder = TextGeneration::builder(&factory, device.clone())
        .progress(progress.clone())
        .on_token(move |t| progress_for_tokens.print(t))
        .seed(args.seed)
        .repeat_penalty(args.repeat_penalty)
        .repeat_last_n(args.repeat_last_n);

    if let Some(temp) = args.temperature {
        pipeline_builder = pipeline_builder.temperature(temp);
    }
    if let Some(top_p) = args.top_p {
        pipeline_builder = pipeline_builder.top_p(top_p);
    }

    // Enable prefix cache if requested
    if args.prefix_cache {
        info!(
            "✓ Prefix cache enabled (max_entries: {}, max_tokens: {})",
            args.cache_max_entries, args.cache_max_tokens
        );
        pipeline_builder =
            pipeline_builder.with_prefix_cache(args.cache_max_entries, args.cache_max_tokens);
    }

    // Enable batching if requested
    if args.enable_batching {
        use tur::backend::scheduler::SchedulingPolicy;

        let policy = match args.scheduling_policy.to_lowercase().as_str() {
            "fcfs" => SchedulingPolicy::FCFS,
            "priority" => SchedulingPolicy::Priority,
            "shortest_job_first" | "sjf" => SchedulingPolicy::SJF,
            _ => {
                return Err(TurError::Other(format!(
                    "Invalid scheduling policy: {}. Use 'fcfs', 'priority', or 'sjf'",
                    args.scheduling_policy
                )));
            }
        };

        info!(
            "✓ Continuous batching enabled (max_batch: {}, prefill: {}, decode: {}, policy: {:?})",
            args.max_batch_size, args.max_prefill_batch, args.max_decode_batch, policy
        );

        pipeline_builder = pipeline_builder
            .enable_batching(true)
            .max_batch_size(args.max_batch_size)
            .max_prefill_batch(args.max_prefill_batch)
            .max_decode_batch(args.max_decode_batch)
            .scheduling_policy(policy);

        if let Some(chunk_size) = args.prefill_chunk_size {
            info!("✓ Chunked prefill enabled (chunk_size: {chunk_size} tokens)");
            pipeline_builder = pipeline_builder.prefill_chunk_size(chunk_size);
        }
    }

    let mut pipeline = pipeline_builder.build();
    debug!("✓ Model is initialized and ready for inference");

    // Load tool definitions from file if provided.
    let tools = match args.tools {
        Some(ref path) => {
            let json = std::fs::read_to_string(path)
                .map_err(|e| TurError::Other(format!("Failed to read tools file '{path}': {e}")))?;
            let defs: Vec<ToolDefinition> = serde_json::from_str(&json).map_err(|e| {
                TurError::Other(format!("Failed to parse tools file '{path}': {e}"))
            })?;
            info!("✓ Loaded {} tool(s) from {path}", defs.len());
            defs
        }
        None => Vec::new(),
    };

    // Build the request.
    let mut request = GenerationRequest::new(DEFAULT_PROMPT.to_string(), args.sample_len);
    if args.thinking {
        request = request.with_thinking(args.thinking);
    }
    if !tools.is_empty() {
        request = request.with_tools(tools);
    }

    let stats = pipeline.run(&request)?;

    if !stats.tool_calls.is_empty() {
        info!("\n--- Tool calls ({}) ---", stats.tool_calls.len());
        for call in &stats.tool_calls {
            info!(
                "  {} ({})",
                call.name,
                serde_json::to_string(&call.arguments).unwrap_or_default()
            );
        }
    }

    Ok(())
}
