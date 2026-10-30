//! `tur transcribe`: speech to text from a WAV file with the configured Whisper model.
use std::{io::Write, path::PathBuf};

use tur::{
    Result, TurError,
    audio::Audio,
    models::whisper::{Fallback, Task, TranscribeOptions},
};

use crate::args::WhisperArgs;

/// What `tur transcribe` writes.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum Format {
    #[default]
    Text,
    Srt,
    Vtt,
    /// One `start<TAB>end<TAB>word` line per word.
    Words,
}

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
    /// Output format; `srt` and `vtt` carry timestamps.
    #[arg(long, value_enum, default_value_t)]
    format: Format,
    /// Text the decoder reads before every window, for names and style.
    #[arg(long)]
    prompt: Option<String>,
    /// Also read the text decoded so far; follows context better, repeats errors more easily.
    #[arg(long)]
    condition_on_previous_text: bool,
    /// Sample at this fixed temperature; by default, doubtful windows retry at rising ones.
    #[arg(long)]
    temperature: Option<f32>,
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
            timestamps: !matches!(args.format, Format::Text),
            word_timestamps: matches!(args.format, Format::Words),
            prompt: args.prompt.clone(),
            condition_on_previous_text: args.condition_on_previous_text,
            fallback: Some(
                args.temperature
                    .map_or_else(Fallback::default, Fallback::fixed),
            ),
        },
    )?;
    let mut stdout = std::io::stdout().lock();
    match args.format {
        Format::Text => writeln!(stdout, "{}", result.text)?,
        Format::Srt => write!(stdout, "{}", result.srt())?,
        Format::Vtt => write!(stdout, "{}", result.vtt())?,
        Format::Words => {
            for word in result.segments.iter().flat_map(|segment| &segment.words) {
                writeln!(
                    stdout,
                    "{:.2}\t{:.2}\t{}",
                    word.start,
                    word.end,
                    word.word.trim()
                )?;
            }
        }
    }
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
