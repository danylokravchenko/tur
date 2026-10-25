use std::{io::Write, path::PathBuf};

use tur::{
    Result,
    backend::pipeline::{GenerationRequest, InferencePipeline},
    hybrid::SamplingOptions,
};

use crate::{
    args::{ModelArgs, PipelineArgs},
    config::inherit,
};

#[derive(Debug, Default, serde::Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Args {
    #[command(flatten)]
    #[serde(skip)]
    model: ModelArgs,
    #[command(flatten)]
    #[serde(skip)]
    pipeline: PipelineArgs,
    #[arg(long, default_value = "Who are you?")]
    #[serde(skip)]
    prompt: String,
    /// Generated-token limit, including EOS (default: 128).
    #[arg(short = 'n', long)]
    max_tokens: Option<usize>,
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
    /// Seed of the request random stream (default: 299792458).
    #[arg(long)]
    seed: Option<u64>,
    /// Explicit true/false; omission preserves the checkpoint's thinking default.
    #[arg(long)]
    thinking: Option<bool>,
    #[arg(long, value_parser = crate::config::reasoning_effort)]
    #[serde(deserialize_with = "crate::config::optional_reasoning_effort")]
    reasoning_effort: Option<String>,
    #[arg(long)]
    preserve_thinking: Option<bool>,
    /// JSON array of function definitions (name, description, parameters).
    #[arg(long)]
    #[serde(skip)]
    tools: Option<PathBuf>,
}

impl Args {
    inherit!(
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        seed,
        thinking,
        reasoning_effort,
        preserve_thinking
    );

    pub(crate) fn configure(&mut self, settings: crate::config::FileSettings) {
        self.model.inherit(settings.model);
        self.pipeline.inherit(settings.engine);
        self.inherit(settings.generation);
    }

    fn request(self) -> Result<GenerationRequest> {
        let mut request = GenerationRequest::new(self.prompt, self.max_tokens.unwrap_or(128));
        request.seed = self.seed.unwrap_or(299_792_458);
        request.sampling = SamplingOptions {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
        };
        request.render.enable_thinking = self.thinking;
        request.render.reasoning_effort = self.reasoning_effort;
        request.render.preserve_thinking = self.preserve_thinking;
        if let Some(path) = self.tools {
            request.tools = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        }
        Ok(request)
    }
}

pub(crate) fn run(args: Args) -> Result<()> {
    let factory = args.model.factory()?;
    let config = args.pipeline.config(1)?;
    config.scheduler.validate()?;
    config.prefix.validate(factory.kv.page_tokens)?;
    let mut pipeline = InferencePipeline::new(&factory, config)?;
    let plan = pipeline.memory_plan();
    tracing::info!(
        resident_bytes = plan.resident(),
        weights = plan.weights,
        state_pool = plan.state_pool,
        scratch_peak = plan.scratch_peak,
        "Memory plan"
    );
    let request = args.request()?;
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

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible generation configuration"
)]
mod tests {
    use super::*;
    use crate::{Command, TurError, config::FileSettings, parse_cli};

    fn arguments(values: &[&str]) -> Result<Args> {
        match parse_cli(values)?.command {
            Command::Run(args) => Ok(args),
            _ => Err(TurError::Other("expected run arguments".into())),
        }
    }

    #[test]
    fn test_generation_defaults_preserve_checkpoint_settings() -> Result<()> {
        let request = arguments(&["tur", "run"])?.request()?;
        assert_eq!(request.max_tokens, 128);
        assert_eq!(request.seed, 299_792_458);
        assert!(request.sampling.temperature.is_none());
        assert!(request.render.enable_thinking.is_none());
        Ok(())
    }

    #[test]
    fn test_cli_overrides_file_generation_defaults_including_false_and_zero() -> Result<()> {
        let settings: FileSettings = toml::from_str(
            r#"
            [engine]
            prefill_chunk_size = 64
            [generation]
            max_tokens = 512
            temperature = 0.8
            top_k = 12
            seed = 123
            thinking = true
            reasoning_effort = "low"
            preserve_thinking = true
        "#,
        )
        .map_err(|error| TurError::Other(error.to_string()))?;
        let mut args = arguments(&[
            "tur",
            "run",
            "--prompt",
            "hello",
            "-n",
            "32",
            "--temperature",
            "0",
            "--thinking",
            "false",
            "--seed",
            "0",
        ])?;
        args.configure(settings);
        assert_eq!(args.pipeline.config(1)?.scheduler.max_chunk, 64);
        let request = args.request()?;
        assert_eq!(request.max_tokens, 32);
        assert_eq!(request.seed, 0);
        assert_eq!(request.sampling.temperature, Some(0.0));
        assert_eq!(request.sampling.top_k, Some(12));
        assert_eq!(request.render.enable_thinking, Some(false));
        assert_eq!(request.render.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(request.render.preserve_thinking, Some(true));
        Ok(())
    }
}
