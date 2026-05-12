use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use tur::{Downloader, Qwen3_5ForCausalLM};

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
    let args = Args::parse();

    println!("Qwen 3.5 Model - Clean Implementation");
    println!("=====================================\n");

    let device = Device::Cpu;
    println!("Device: {:?}\n", device);

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

    println!("Model Config:");
    println!("  - Vocabulary size: {}", config.vocab_size);
    println!("  - Hidden size: {}", config.hidden_size);
    println!("  - Number of heads: {}", config.num_attention_heads);
    println!("  - Number of layers: {}", config.num_hidden_layers);
    println!("  - Intermediate size: {}\n", config.intermediate_size);

    let safetensors = paths.get_weight_filenames();
    println!("Loaded weight paths:");
    for path in &safetensors {
        println!("  - {}", path.display());
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors, DType::F32, &device)? };
    let model = Qwen3_5ForCausalLM::new(config, vb, &device)?;

    println!(
        "\n✓ Loaded Qwen 3.5 with {} safetensor shard(s)",
        safetensors.len()
    );
    println!("✓ Model is initialized and ready for inference");

    // TODO: run a forward pass with token IDs once the input pipeline is wired.

    drop(model);
    Ok(())
}
