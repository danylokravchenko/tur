//! `tur transcribe`: speech to text from a WAV file with the configured Whisper model.
use std::{io::Write, path::PathBuf};

use tur::{
    Result, TurError,
    audio::Audio,
    models::whisper::{Task, TranscribeOptions},
};

use crate::args::WhisperArgs;

#[derive(Debug, Default, clap::Args)]
pub(crate) struct Args {
    #[command(flatten)]
    whisper: WhisperArgs,
    /// WAV file to transcribe (integer PCM or float, any sample rate and channel count).
    file: PathBuf,
    /// Spoken language as an ISO 639-1 code such as `en`; detected when omitted.
    #[arg(long)]
    language: Option<String>,
    /// Write an English translation instead of a transcript.
    #[arg(long)]
    translate: bool,
    /// Use the CPU numerical reference for development checks.
    #[arg(long)]
    cpu_reference: bool,
}

impl Args {
    pub(crate) fn configure(&mut self, settings: crate::config::FileSettings) {
        self.whisper.inherit(settings.whisper);
    }
}

pub(crate) fn transcribe(args: &Args) -> Result<()> {
    let device = tur::shared::inference_device(args.cpu_reference)?;
    let factory = args
        .whisper
        .factory(&device)?
        .ok_or_else(|| TurError::Other("set --whisper (or [whisper].source in tur.toml)".into()))?;
    let audio = Audio::from_wav(&std::fs::read(&args.file)?)?;
    let transcriber = factory.create()?;
    let started = std::time::Instant::now();
    let result = transcriber.transcribe(
        &audio,
        &TranscribeOptions {
            language: args.language.clone(),
            task: if args.translate {
                Task::Translate
            } else {
                Task::Transcribe
            },
        },
    )?;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{}", result.text)?;
    let elapsed = started.elapsed();
    tracing::info!(
        audio_secs = audio.duration().as_secs_f64(),
        elapsed_secs = elapsed.as_secs_f64(),
        language = result.language.as_deref().unwrap_or("-"),
        tokens = result.tokens,
        "Transcribed"
    );
    Ok(())
}
