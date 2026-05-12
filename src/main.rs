use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use tokenizers::Tokenizer;
use tracing::level_filters::LevelFilter;
use tracing::{debug, info, trace};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, fmt};
use tur::Downloader;
use tur::backend::tokenizer::TokenOutputStream;
use tur::models::Gwen35ModelForCausalLM;

use anyhow::Result;
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

struct TextGeneration {
    model: Gwen35ModelForCausalLM,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Gwen35ModelForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eos_token2 {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
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
}

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    info!("Qwen 3.5 Model - Clean Implementation");

    let device = Device::new_metal(0)?;
    //let device = Device::Cpu;
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };
    debug!("Device: {:?}", device);
    debug!("DType: {:?}", dtype);
    debug!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

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
    trace!("Reading config file: {}", config_path.display());
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

    debug!("Model Config value: {:?}", config_json);

    // let text_config = config_json
    //     .get("text_config")
    //     .ok_or_else(|| anyhow::anyhow!("Config missing text_config field"))?;

    let config: tur::models::qwen3_5::Config = serde_json::from_value(config_json.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse text_config: {}", e))?;

    // // Extract rope_theta from rope_parameters if present
    // if let Some(rope_params) = text_config.get("rope_parameters") {
    //     if let Some(rope_theta) = rope_params.get("rope_theta").and_then(|v| v.as_f64()) {
    //         config.rope_theta = rope_theta;
    //     }
    // }
    debug!("Model Config: {:?}", config);

    let safetensors = paths.get_weight_filenames();
    debug!("Loaded weight paths: {:?}", safetensors);

    let tokenizer_path = paths.get_tokenizer_filename();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    debug!("Loaded Tokenizer from: {:?}", tokenizer_path);

    // let vb = VarBuilderX::new(&paths, gguf, DType::F32, &device)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors, dtype, &device)? };
    let model = Gwen35ModelForCausalLM::new(&config, vb)?;

    debug!(
        "✓ Loaded Qwen 3.5 with {} safetensor shard(s)",
        safetensors.len()
    );
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    debug!("✓ Model is initialized and ready for inference");

    let prompt_str =
        format!("<|im_start|>user\n{DEFAULT_PROMPT}<|im_end|>\n<|im_start|>assistant\n");
    trace!("formatted prompt: {}", &prompt_str);

    pipeline.run(&prompt_str, args.sample_len)?;

    // let tokens = tos
    //     .tokenizer()
    //     .encode(prompt_str, true)
    //     .map_err(anyhow::Error::msg)?;

    // let tokens = tokens.get_ids();

    // let mut all_tokens = vec![];

    // let mut logits_processor = {
    //     let temperature = args.temperature;
    //     let sampling = if temperature <= 0. {
    //         Sampling::ArgMax
    //     } else {
    //         match (args.top_k, args.top_p) {
    //             (None, None) => Sampling::All { temperature },
    //             (Some(k), None) => Sampling::TopK { k, temperature },
    //             (None, Some(p)) => Sampling::TopP { p, temperature },
    //             (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
    //         }
    //     };
    //     trace!("Using sampling: {:?}", sampling);
    //     LogitsProcessor::from_sampling(args.seed, sampling)
    // };

    // let start_prompt_processing = std::time::Instant::now();

    // let mut next_token = {
    //     let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    //     let logits = model.forward(&input)?;
    //     let logits = logits.i((0, tokens.len() - 1))?;
    //     logits_processor.sample(&logits)?
    // };

    // let prompt_dt = start_prompt_processing.elapsed();
    // trace!(
    //     "Prompt was processed in {:?}. Next token: {}",
    //     prompt_dt, next_token
    // );

    // let eos_token = *tos
    //     .tokenizer()
    //     .get_vocab(true)
    //     .get("<|im_end|>")
    //     .ok_or_else(|| anyhow::anyhow!("Tokenizer missing <|im_end|> token"))?;

    // let start_post_prompt = std::time::Instant::now();
    // let mut sampled = 0usize;

    // while sampled < args.sample_len && next_token != eos_token {
    //     if let Some(t) = tos.next_token(next_token)? {
    //         trace!("{t}");
    //     }

    //     let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
    //     let logits = model.forward(&input)?;
    //     let logits = logits.i((0, 0))?;
    //     next_token = logits_processor.sample(&logits)?;
    //     sampled += 1;
    //     all_tokens.push(next_token);
    // }

    // if next_token == eos_token {
    //     debug!(
    //         "Encountered EOS token after generating {} token(s)",
    //         sampled
    //     );
    // }

    // if let Some(rest) = tos.decode_rest()? {
    //     trace!("{rest}");
    // }

    // let dt = start_post_prompt.elapsed();
    // info!(
    //     "\n{:4} prompt tokens processed: {:.2} token/s",
    //     tokens.len(),
    //     tokens.len() as f64 / prompt_dt.as_secs_f64(),
    // );
    // info!(
    //     "{sampled:4} tokens generated: {:.2} token/s",
    //     sampled as f64 / dt.as_secs_f64(),
    // );

    // // Run a simple forward pass with dummy token IDs
    // info!("Running forward pass...");

    // // Create a simple input: batch_size=1, seq_len=5 with token IDs [1, 2, 3, 4, 5]
    // let input_ids = Tensor::new(&[1u32, 2u32, 3u32, 4u32, 5u32], &device)?.reshape((1, 5))?; // Shape: [batch_size, seq_len]

    // debug!("Input shape: {:?}", input_ids.shape());

    // // Forward pass
    // let logits = model.forward(&input_ids)?;

    // debug!("Output logits shape: {:?}", logits.shape());
    // debug!(
    //     "Expected shape: [batch_size=1, seq_len=5, vocab_size={}]",
    //     model.vocab_size()
    // );

    // // Verify output shape
    // let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    // assert_eq!(batch_size, 1, "Batch size mismatch");
    // assert_eq!(seq_len, 5, "Sequence length mismatch");
    // assert_eq!(vocab_size, model.vocab_size(), "Vocab size mismatch");

    // info!("✓ Forward pass successful!");
    // info!("  Input shape: [1, 5]");
    // info!(
    //     "  Output shape: [{}, {}, {}]",
    //     batch_size, seq_len, vocab_size
    // );

    // // Get the last token's logits for next token prediction
    // let last_token_logits = logits.i((0, seq_len - 1))?;
    // debug!("Last token logits shape: {:?}", last_token_logits.shape());

    // // Find the token with highest probability (argmax)
    // let next_token = last_token_logits.argmax(0)?;
    // let next_token_id = next_token.to_scalar::<u32>()?;
    // info!("  Predicted next token ID: {}", next_token_id);

    //drop(model);
    Ok(())
}
