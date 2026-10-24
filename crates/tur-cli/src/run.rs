use std::{io::Write, path::PathBuf};

use tur::{
    Result,
    backend::pipeline::{GenerationRequest, InferencePipeline},
    hybrid::SamplingOptions,
};

use crate::args::{ModelArgs, PipelineArgs};

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    pipeline: PipelineArgs,
    #[arg(long, default_value = "Who are you?")]
    prompt: String,
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,
    /// Unset sampling controls inherit `generation_config.json`; zero is greedy.
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
}

pub(crate) fn run(args: Args) -> Result<()> {
    let mut pipeline = InferencePipeline::new(&args.model.factory()?, args.pipeline.config(1)?)?;
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
    request.sampling = SamplingOptions {
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
