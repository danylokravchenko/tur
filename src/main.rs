use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use tracing::level_filters::LevelFilter;
use tracing::{debug, info};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, fmt};
use tur::{Downloader, Qwen3_5ForCausalLM, VarBuilderX};

fn init_tracing() {
    let registry = tracing_subscriber::registry();

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(LevelFilter::TRACE.to_string()))
        .add_directive("ureq=error".parse().unwrap())
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
    /// Hugging Face model ID for downloading weights
    #[arg(long, env = "HF_MODEL_ID")]
    model_id: Option<String>,

    /// Local path to a directory containing safetensors weights
    #[arg(long, env = "MODEL_WEIGHT_PATH")]
    weight_path: Option<String>,

    /// Optional weight filename inside the local path
    #[arg(long, env = "MODEL_WEIGHT_FILE")]
    weight_file: Option<String>,
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();

    info!("Qwen 3.5 Model - Clean Implementation");

    let device = Device::Cpu;
    debug!("Device: {:?}", device);

    if args.model_id.is_none() && args.weight_path.is_none() {
        anyhow::bail!(
            "Please provide a weight source: `--weight-path <path>` for local safetensors or `--model-id <hf-model>` for Hugging Face downloads."
        );
    }

    let downloader = Downloader::new(args.model_id, args.weight_path, args.weight_file);
    let (paths, gguf) = downloader.prepare_model_weights(None, None)?;

    if gguf {
        anyhow::bail!("GGUF model loading is not implemented in this example.");
    }

    // Load config from downloaded config.json
    let config_path = paths.get_config_filename();
    let config_content = std::fs::read_to_string(&config_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read config file {}: {}",
            config_path.display(),
            e
        )
    })?;

    // Parse the JSON and extract text_config
    let config_json: serde_json::Value = serde_json::from_str(&config_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse config JSON: {}", e))?;

    let text_config = config_json
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("Config missing text_config field"))?;

    let mut config: tur::models::qwen3_5::Qwen3_5Config =
        serde_json::from_value(text_config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse text_config: {}", e))?;

    // Extract rope_theta from rope_parameters if present
    if let Some(rope_params) = text_config.get("rope_parameters") {
        if let Some(rope_theta) = rope_params.get("rope_theta").and_then(|v| v.as_f64()) {
            config.rope_theta = rope_theta;
        }
    }
    debug!("Model Config: {:?}", config);

    let safetensors = paths.get_weight_filenames();
    debug!("Loaded weight paths: {:?}", safetensors);

    let vb = VarBuilderX::new(&paths, gguf, DType::F32, &device)?;
    let model = Qwen3_5ForCausalLM::new(config, vb, DType::F32, &device)?;

    debug!(
        "\n✓ Loaded Qwen 3.5 with {} safetensor shard(s)",
        safetensors.len()
    );
    debug!("✓ Model is initialized and ready for inference");

    // Run a simple forward pass with dummy token IDs
    info!("Running forward pass...");

    // Create a simple input: batch_size=1, seq_len=5 with token IDs [1, 2, 3, 4, 5]
    let input_ids = Tensor::new(&[1u32, 2u32, 3u32, 4u32, 5u32], &device)?.reshape((1, 5))?; // Shape: [batch_size, seq_len]

    debug!("Input shape: {:?}", input_ids.shape());

    // Forward pass
    let logits = model.forward(&input_ids)?;

    debug!("Output logits shape: {:?}", logits.shape());
    debug!(
        "Expected shape: [batch_size=1, seq_len=5, vocab_size={}]",
        model.vocab_size()
    );

    // Verify output shape
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    assert_eq!(batch_size, 1, "Batch size mismatch");
    assert_eq!(seq_len, 5, "Sequence length mismatch");
    assert_eq!(vocab_size, model.vocab_size(), "Vocab size mismatch");

    info!("✓ Forward pass successful!");
    info!("  Input shape: [1, 5]");
    info!(
        "  Output shape: [{}, {}, {}]",
        batch_size, seq_len, vocab_size
    );

    // Get the last token's logits for next token prediction
    let last_token_logits = logits.i((0, seq_len - 1))?;
    debug!("Last token logits shape: {:?}", last_token_logits.shape());

    // Find the token with highest probability (argmax)
    let next_token = last_token_logits.argmax(0)?;
    let next_token_id = next_token.to_scalar::<u32>()?;
    info!("  Predicted next token ID: {}", next_token_id);

    drop(model);
    Ok(())
}
