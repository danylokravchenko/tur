#![forbid(unsafe_code)]
use clap::Parser;
use std::{io::Write, path::PathBuf};
use tur::{
    ModelFactory, ModelSource, Result,
    backend::{
        pipeline::{GenerationRequest, InferencePipeline, PipelineConfig},
        prefix::PrefixConfig,
        speculation::SpeculationConfig,
    },
    core::execution::{KvFormat, KvLayout},
    models::qwen3_5::{
        DEFAULT_KV_LAYOUT, DEFAULT_RECURRENT_STATE, WeightFormat, parse_recurrent_state,
    },
};

#[derive(Debug, Parser)]
#[command(author, version, about = "Tur hybrid text generation")]
struct Args {
    /// Fully qualified Hugging Face repository; requires an immutable --revision.
    #[arg(
        long,
        env = "HF_MODEL_ID",
        conflicts_with = "weight_path",
        required_unless_present = "weight_path"
    )]
    model_id: Option<String>,
    /// Local directory with `SafeTensors`, config, tokenizer and chat template.
    #[arg(long, env = "MODEL_WEIGHT_PATH", conflicts_with = "model_id")]
    weight_path: Option<PathBuf>,
    /// Checkpoint commit (remote) or explicit immutable identity (local).
    #[arg(long)]
    revision: String,
    /// Convert `SafeTensors` projections into this resident format: f32, `q8_0`, q4k.
    /// Omitted, a prepacked artifact (`tur-quantize`) loads as stored and any other
    /// checkpoint converts to `q8_0`.
    #[arg(long)]
    weight_format: Option<WeightFormat>,
    #[arg(long, default_value = "Who are you?")]
    prompt: String,
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,
    /// Unset sampling controls inherit `generation_config.json`.
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long)]
    top_k: Option<usize>,
    /// Above 1 discourages every prompt or generated token (divides positive logits).
    #[arg(long)]
    repetition_penalty: Option<f64>,
    /// Subtracted once from the logit of every generated token.
    #[arg(long)]
    presence_penalty: Option<f64>,
    /// Subtracted per occurrence from the logit of every generated token.
    #[arg(long)]
    frequency_penalty: Option<f64>,
    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,
    /// Explicit true/false; omission preserves the checkpoint's thinking default.
    #[arg(long)]
    thinking: Option<bool>,
    #[arg(long, value_parser = ["low", "medium", "xhigh"])]
    reasoning_effort: Option<String>,
    #[arg(long)]
    preserve_thinking: Option<bool>,
    /// JSON array of function definitions (name, description, parameters).
    #[arg(long)]
    tools: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    prefill_chunk_size: usize,
    /// Maximum prompt-lookup drafts per decode. Opt-in throughput mode: tokens arrive in
    /// bursts and pauses can grow. Zero disables it; draft length adapts to measured cost.
    /// Requires F32 recurrent state; narrower state automatically uses ordinary decoding.
    #[arg(long, default_value_t = 0)]
    speculative_draft: usize,
    /// Also draft one token per decode from the checkpoint's native multi-token prediction
    /// (MTP) head when prompt lookup has none. Implies at least one draft; disables prefix
    /// reuse. Requires F32 recurrent state.
    #[arg(long)]
    mtp: bool,
    /// K/V storage: f32, f16, q8 (one scale per 32 dimensions, half of F16's memory) or
    /// q8_<group>.
    #[arg(long, default_value_t = DEFAULT_KV_LAYOUT.format)]
    kv_format: KvFormat,
    /// Tokens per K/V page.
    #[arg(long, default_value_t = DEFAULT_KV_LAYOUT.page_tokens)]
    kv_page_tokens: usize,
    /// Recurrent matrix state storage: f32 (default) or f16, which halves recurrent memory
    /// and traffic with stochastic rounding but adds error that grows with decode length.
    #[arg(long, value_parser = ["f32", "f16"])]
    recurrent_state: Option<String>,
    /// Memory for K/V pages and recurrent state, in MiB. Default: room for each concurrent
    /// request's 8K-token context within the device's working set.
    #[arg(long)]
    state_memory_mb: Option<usize>,
    /// Use the CPU numerical reference for development checks.
    #[arg(long)]
    cpu_reference: bool,
}

fn main() -> Result<()> {
    tur::shared::init_tracing()?;
    let args = Args::parse();
    let device = tur::shared::inference_device(args.cpu_reference)?;
    let source = match (args.model_id, args.weight_path) {
        (Some(id), None) => ModelSource::HuggingFace(id),
        (None, Some(path)) => ModelSource::LocalPath(path),
        _ => {
            return Err(tur::TurError::Other(
                "provide exactly one checkpoint source".into(),
            ));
        }
    };
    let factory = ModelFactory {
        source,
        revision: args.revision,
        format: args.weight_format,
        device,
        kv: KvLayout::new(args.kv_page_tokens, args.kv_format),
        recurrent: args
            .recurrent_state
            .as_deref()
            .map_or(Ok(DEFAULT_RECURRENT_STATE), parse_recurrent_state)
            .map_err(tur::TurError::Other)?,
        mtp: false,
    };
    let mut pipeline = InferencePipeline::new(
        &factory,
        PipelineConfig {
            speculation: if args.mtp {
                SpeculationConfig::with_mtp(args.speculative_draft.max(1))
            } else {
                SpeculationConfig::new(args.speculative_draft)
            },
            state_bytes: args.state_memory_mb.map(|mb| mb << 20),
            prefix: if args.mtp {
                PrefixConfig::DISABLED
            } else {
                PipelineConfig::new(1, args.prefill_chunk_size).prefix
            },
            ..PipelineConfig::new(1, args.prefill_chunk_size)
        },
    )?;
    let plan = pipeline.memory_plan();
    tracing::info!(
        resident_bytes = plan.resident(),
        weights = plan.weights,
        state_pool = plan.state_pool,
        scratch_peak = plan.scratch_peak,
        "Memory plan"
    );
    let mut request = GenerationRequest::new(args.prompt, args.max_tokens);
    request.seed = args.seed;
    request.sampling = tur::hybrid::SamplingOptions {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
    };
    request.render.enable_thinking = args.thinking;
    request.render.reasoning_effort = args.reasoning_effort;
    request.render.preserve_thinking = args.preserve_thinking;
    if let Some(path) = args.tools {
        request.tools = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    }
    let mut stdout = std::io::stdout().lock();
    let mut output_error = None;
    let stats = pipeline.run(&request, |text| {
        if output_error.is_none() {
            output_error = stdout
                .write_all(text.as_bytes())
                .and_then(|()| stdout.flush())
                .err();
        }
    })?;
    if let Some(error) = output_error {
        return Err(error.into());
    }
    writeln!(stdout)?;
    tracing::info!(prompt_tokens = stats.prompt_tokens, generated_tokens = stats.generated_tokens,
        elapsed = ?stats.elapsed, finish_reason = stats.finish_reason, "Generation complete");
    Ok(())
}
