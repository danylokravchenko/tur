use std::{
    io::{IsTerminal as _, Read as _, Write},
    path::PathBuf,
};

use tur::{
    Result, TurError,
    audio::Audio,
    backend::{
        chat_template::{ChatMessage, Content, ContentPart, Role},
        pipeline::{GenerationRequest, InferencePipeline},
    },
    hybrid::SamplingOptions,
    image::Image,
    video::Video,
};

use crate::{
    args::{ModelArgs, PipelineArgs},
    config::inherit,
    ui::{Console, describe, seconds},
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
    /// One user turn, for one answer. Without it, piped input is the prompt and a terminal
    /// opens a chat.
    #[arg(value_name = "PROMPT", conflicts_with = "prompt")]
    #[serde(skip)]
    words: Vec<String>,
    /// PROMPT as a flag, for when tur.toml selects the model.
    #[arg(long)]
    #[serde(skip)]
    prompt: Option<String>,
    /// WAV clip placed before the prompt, for models with an audio encoder (Gemma 4 E2B
    /// and E4B).
    #[arg(long)]
    #[serde(skip)]
    audio: Option<PathBuf>,
    /// Image placed before the prompt, for models with a vision encoder (Qwen3.5 and
    /// Gemma 4): PNG, JPEG, WebP or GIF. Repeat for several.
    #[arg(long)]
    #[serde(skip)]
    image: Vec<PathBuf>,
    /// Video placed before the prompt, for models with a vision encoder (Qwen3.5 and
    /// Gemma 4): H.264 in an MP4 file. Repeat for several.
    #[arg(long)]
    #[serde(skip)]
    video: Vec<PathBuf>,
    /// Generated-token limit, including EOS. Default: 1024, or what the context leaves.
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
        self.model.select(settings.model);
        self.pipeline.inherit(settings.engine);
        self.inherit(settings.generation);
    }

    /// What this invocation asks. `None` is a chat, which needs a terminal to read turns from.
    fn prompt(&mut self) -> Result<Option<String>> {
        if !self.words.is_empty() {
            return Ok(Some(self.words.join(" ")));
        }
        if let Some(prompt) = self.prompt.take() {
            return Ok(Some(prompt));
        }
        if std::io::stdin().is_terminal() {
            if self.audio.is_some() || !self.image.is_empty() || !self.video.is_empty() {
                return Err(TurError::Other(
                    "--audio, --image and --video need a prompt".into(),
                ));
            }
            return Ok(None);
        }
        let mut prompt = String::new();
        std::io::stdin().read_to_string(&mut prompt)?;
        if prompt.trim().is_empty() {
            return Err(TurError::Other("no prompt on standard input".into()));
        }
        Ok(Some(prompt))
    }

    fn request(self, prompt: String) -> Result<GenerationRequest> {
        let mut request = GenerationRequest::new(prompt.clone(), self.max_tokens.unwrap_or(1024));
        // A limit nobody chose must not fail a long conversation.
        request.fit_context = self.max_tokens.is_none();
        let mut parts = Vec::new();
        if let Some(path) = &self.audio {
            parts.push(ContentPart::Audio(Audio::from_wav(&std::fs::read(path)?)?));
        }
        for path in &self.image {
            parts.push(ContentPart::Image(Image::decode(&std::fs::read(path)?)?));
        }
        for path in &self.video {
            parts.push(ContentPart::Video(Video::decode(&std::fs::read(path)?)?));
        }
        if !parts.is_empty() {
            parts.push(ContentPart::Text(prompt));
            request.messages[0].content = Content::Parts(parts);
        }
        request.seed = self.seed.unwrap_or(299_792_458);
        request.sampling = SamplingOptions {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            logprobs: None,
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

pub(crate) fn run(mut args: Args, console: &Console) -> Result<()> {
    if args.model.caches_experts() {
        args.pipeline.prefer_large_passes();
    }
    let factory = args.model.factory()?;
    let prompt = args.prompt()?;
    let config = args.pipeline.config(1)?;
    config.scheduler.validate()?;
    config.prefix.validate(factory.kv.page_tokens)?;
    let mut pipeline = InferencePipeline::new(&factory, config)?;
    match prompt {
        Some(prompt) => {
            let request = args.request(prompt)?;
            generate(&mut pipeline, &request, console).map(drop)
        }
        None => chat(&mut pipeline, args.request(String::new())?, console),
    }
}

/// Answer turns from the terminal until `/bye` or the end of input. Each turn sends the whole
/// conversation, and prefix reuse keeps that from recomputing what the last turn already did.
fn chat(
    pipeline: &mut InferencePipeline,
    mut request: GenerationRequest,
    console: &Console,
) -> Result<()> {
    console.line("  /clear forgets the conversation; /bye or Ctrl-D leaves.");
    request.messages.clear();
    loop {
        let mut stdout = std::io::stdout();
        write!(stdout, "{}", console.paint("1", ">>> "))?;
        stdout.flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            writeln!(stdout)?;
            return Ok(());
        }
        match line.trim() {
            "" => continue,
            "/bye" => return Ok(()),
            "/clear" => {
                request.messages.clear();
                continue;
            }
            turn => request.messages.push(ChatMessage::new(Role::User, turn)),
        }
        match generate(pipeline, &request, console) {
            Ok(answer) => request
                .messages
                .push(ChatMessage::new(Role::Assistant, answer)),
            Err(error) => {
                // The conversation stays usable: the failed turn is simply not part of it.
                request.messages.pop();
                let label = console.paint("1;31", "error:");
                console.line(&format!("{label} {}", describe(&error)));
            }
        }
        request.seed = request.seed.wrapping_add(1);
    }
}

/// Stream one answer to stdout, and any reasoning to stderr, then return the answer's text.
fn generate(
    pipeline: &mut InferencePipeline,
    request: &GenerationRequest,
    console: &Console,
) -> Result<String> {
    let mut stdout = std::io::stdout().lock();
    let mut output_error = None;
    let mut reasoned = false;
    let mut answer = String::new();
    let stats = pipeline.run_channels(
        request,
        |text| {
            answer.push_str(text);
            if output_error.is_none() {
                output_error = stdout
                    .write_all(text.as_bytes())
                    .and_then(|()| stdout.flush())
                    .err();
            }
        },
        |text| {
            // Reasoning goes to stderr so that stdout holds only the answer. The handle is
            // locked per write: a held lock would stall log lines from other threads. Nothing
            // depends on progress text reaching a closed stderr.
            reasoned = true;
            drop(std::io::stderr().write_all(console.paint("2", text).as_bytes()));
        },
    )?;
    if reasoned {
        drop(writeln!(std::io::stderr()));
    }
    if let Some(error) = output_error {
        return Err(error.into());
    }
    writeln!(stdout)?;
    tracing::info!(prompt_tokens = stats.prompt_tokens, generated_tokens = stats.generated_tokens,
        elapsed = ?stats.elapsed, finish_reason = stats.finish_reason, "Generation complete");
    if console.interactive() {
        let elapsed = stats.elapsed.as_secs_f64();
        console.line(&console.paint(
            "2",
            &format!(
                "  {} tokens at {:.1}/s · first token {} · {} total · {}",
                stats.generated_tokens,
                stats.generated_tokens as f64 / elapsed.max(f64::MIN_POSITIVE),
                seconds(stats.first_token.as_secs_f64()),
                seconds(elapsed),
                stats.finish_reason,
            ),
        ));
    }
    Ok(answer)
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible generation configuration"
)]
mod tests {
    use super::*;
    use crate::{Command, config::FileSettings, parse_cli};

    fn arguments(values: &[&str]) -> Result<Args> {
        match parse_cli(values)?.command {
            Some(Command::Run(args)) => Ok(args),
            _ => Err(TurError::Other("expected run arguments".into())),
        }
    }

    #[test]
    fn test_generation_defaults_preserve_checkpoint_settings() -> Result<()> {
        let request = arguments(&["tur", "run"])?.request("hello".into())?;
        assert_eq!(request.max_tokens, 1024);
        assert!(request.fit_context);
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
            "qwen",
            "hello",
            "world",
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
        assert_eq!(args.prompt()?.as_deref(), Some("hello world"));
        let request = args.request("hello world".into())?;
        assert_eq!(request.max_tokens, 32);
        assert!(!request.fit_context);
        assert_eq!(request.seed, 0);
        assert_eq!(request.sampling.temperature, Some(0.0));
        assert_eq!(request.sampling.top_k, Some(12));
        assert_eq!(request.render.enable_thinking, Some(false));
        assert_eq!(request.render.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(request.render.preserve_thinking, Some(true));
        Ok(())
    }

    #[test]
    fn test_prompt_flag_serves_a_model_from_the_file_and_excludes_words() -> Result<()> {
        let mut args = arguments(&["tur", "run", "--prompt", "hello"])?;
        assert_eq!(args.prompt()?.as_deref(), Some("hello"));
        assert!(parse_cli(&["tur", "run", "qwen", "hello", "--prompt", "hi"]).is_err());
        Ok(())
    }
}
