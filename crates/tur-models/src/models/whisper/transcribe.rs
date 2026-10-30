//! Greedy Whisper transcription, as Transformers' `generate` decodes and seeks.
use super::{
    config::{Config, Generation},
    model::{Decoder, Whisper},
    timestamps::{TIME_PRECISION, TimestampRules},
};
use crate::{
    audio::{Audio, WHISPER_FRAMES, WHISPER_WINDOW},
    models::WeightFormat,
};
use candle_core::{Device, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;
use tur_core::{Result, TurError, execution::invalid};

/// What the decoder writes: the spoken language, or an English translation of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Task {
    #[default]
    Transcribe,
    Translate,
}

impl Task {
    fn key(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
        }
    }
}

/// Controls of one transcription.
#[derive(Clone, Debug, Default)]
pub struct TranscribeOptions {
    /// ISO 639-1 code such as `en`; `None` detects it from the first window.
    pub language: Option<String>,
    pub task: Task,
    /// Decode timestamps and report segments. Clips beyond 30 s always do, because seeking
    /// between windows depends on them.
    pub timestamps: bool,
}

/// A stretch of speech with its timing.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// Seconds from the start of the clip.
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Feature frame at which this segment's window started.
    pub seek: usize,
    /// Text and timestamp tokens.
    pub tokens: Vec<u32>,
}

/// A finished transcription.
#[derive(Clone, Debug)]
pub struct Transcription {
    pub text: String,
    /// The language code the decoder was prompted with, when the model is multilingual.
    pub language: Option<String>,
    /// Text tokens decoded across all windows.
    pub tokens: usize,
    /// Windows of up to 30 s that were decoded.
    pub windows: usize,
    /// Timed segments; empty when the decoding carried no timestamps.
    pub segments: Vec<Segment>,
}

impl Transcription {
    /// `SubRip` subtitles of the segments.
    pub fn srt(&self) -> String {
        let cues: Vec<String> = self
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| {
                let (start, end) = (clock(segment.start, ','), clock(segment.end, ','));
                let text = segment.text.trim();
                format!("{}\n{start} --> {end}\n{text}\n\n", index + 1)
            })
            .collect();
        cues.concat()
    }

    /// `WebVTT` subtitles of the segments.
    pub fn vtt(&self) -> String {
        let cues: Vec<String> = self
            .segments
            .iter()
            .map(|segment| {
                let (start, end) = (clock(segment.start, '.'), clock(segment.end, '.'));
                format!("{start} --> {end}\n{}\n\n", segment.text.trim())
            })
            .collect();
        format!("WEBVTT\n\n{}", cues.concat())
    }
}

/// `HH:MM:SS` and milliseconds, after the subtitle format's `separator`.
fn clock(seconds: f64, separator: char) -> String {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clip times are short and non-negative"
    )]
    let millis = (seconds.max(0.0) * 1000.0).round() as u64;
    let (hours, minutes) = (millis / 3_600_000, millis / 60_000 % 60);
    let (secs, rest) = (millis / 1000 % 60, millis % 1000);
    format!("{hours:02}:{minutes:02}:{secs:02}{separator}{rest:03}")
}

/// What a transcription accumulates across windows.
#[derive(Default)]
struct Progress {
    language: Option<u32>,
    text: String,
    tokens: usize,
    windows: usize,
    segments: Vec<Segment>,
}

/// A Whisper checkpoint with its tokenizer and decoding settings.
pub struct Transcriber {
    model: Whisper,
    tokenizer: Tokenizer,
    generation: Generation,
}

impl Transcriber {
    /// Load a local checkpoint directory: weights, `tokenizer.json` and
    /// `generation_config.json`.
    pub fn load(dir: &Path, format: WeightFormat, device: &Device) -> Result<Self> {
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let generation = Generation::from_dir(dir, &config)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&id| id as usize >= config.vocab_size)
        {
            return Err(TurError::Tokenizer(
                "tokenizer IDs exceed the Whisper vocabulary".into(),
            ));
        }
        Ok(Self {
            model: Whisper::load(dir, format, device)?,
            tokenizer,
            generation,
        })
    }

    pub fn model(&self) -> &Whisper {
        &self.model
    }

    /// Language codes the model can be prompted with, such as `en`.
    pub fn languages(&self) -> impl Iterator<Item = &str> {
        self.generation
            .lang_to_id
            .keys()
            .filter_map(|key| key.strip_prefix("<|")?.strip_suffix("|>"))
    }

    /// Transcribe (or translate) a clip. A clip of at most 30 s is one window. A longer one
    /// is decoded with timestamps, each window starting where the last closed segment of
    /// the one before it ended, so no word is cut at a window's edge.
    pub fn transcribe(&self, clip: &Audio, options: &TranscribeOptions) -> Result<Transcription> {
        let mut progress = Progress {
            language: options
                .language
                .as_deref()
                .map(|code| self.language_token(code))
                .transpose()?,
            ..Progress::default()
        };
        if clip.samples().len() > WHISPER_WINDOW {
            self.seek_through(clip.samples(), options, &mut progress)?;
        } else if options.timestamps {
            let mut padded = clip.samples().to_vec();
            padded.resize(WHISPER_WINDOW, 0.0);
            self.seek_through(&padded, options, &mut progress)?;
        } else {
            let features = self.model.features(clip.samples())?;
            self.plain_window(&features, options, &mut progress)?;
        }
        let language = progress.language.and_then(|token| {
            self.generation
                .lang_to_id
                .iter()
                .find(|(_, id)| **id == token)
                .and_then(|(key, _)| key.strip_prefix("<|")?.strip_suffix("|>"))
                .map(str::to_owned)
        });
        Ok(Transcription {
            text: progress.text.trim().to_owned(),
            language,
            tokens: progress.tokens,
            windows: progress.windows,
            segments: progress.segments,
        })
    }

    /// Decode with timestamps window by window, as Transformers' `generate` seeks. `samples`
    /// shorter than a window arrive padded to one, as its feature extractor pads them.
    fn seek_through(
        &self,
        samples: &[f32],
        options: &TranscribeOptions,
        progress: &mut Progress,
    ) -> Result<()> {
        let features = self.model.clip_features(samples)?;
        let mut seek = 0;
        while seek < features.frames() {
            let frames = (features.frames() - seek).min(WHISPER_FRAMES);
            let window = features.window(seek, frames, self.model.device())?;
            seek += self.timed_window(&window, (seek, frames), options, progress)?;
        }
        Ok(())
    }

    /// One window without timestamps.
    fn plain_window(
        &self,
        features: &Tensor,
        options: &TranscribeOptions,
        progress: &mut Progress,
    ) -> Result<()> {
        let encoded = self.model.encode(features)?;
        let mut decoder = self.model.decoder(&encoded)?;
        let logits = self.prompt(&mut decoder, options, progress, true)?;
        let decoded = self.greedy(&mut decoder, &logits, None)?;
        progress.tokens += decoded.len();
        progress.windows += 1;
        progress.text.push_str(&self.decode_text(&decoded)?);
        Ok(())
    }

    /// One window of `frames` frames from frame `seek`, with timestamps; returns the frames
    /// to advance.
    fn timed_window(
        &self,
        features: &Tensor,
        (seek, frames): (usize, usize),
        options: &TranscribeOptions,
        progress: &mut Progress,
    ) -> Result<usize> {
        let rules = self.timestamp_rules()?;
        let encoded = self.model.encode(features)?;
        let mut decoder = self.model.decoder(&encoded)?;
        let logits = self.prompt(&mut decoder, options, progress, false)?;
        let sequence = self.greedy(&mut decoder, &logits, Some(&rules))?;
        let (segments, advance) = rules.split(&sequence, frames);
        #[allow(clippy::cast_precision_loss, reason = "clip frames are far below 2^52")]
        let offset = seek as f64 * 0.01;
        #[allow(clippy::cast_precision_loss, reason = "timestamp steps are small")]
        for segment in segments {
            let text_tokens = segment.tokens.iter().filter(|&&t| t < rules.begin).count();
            if text_tokens == 0 {
                continue;
            }
            let text = self.decode_text(&segment.tokens)?;
            progress.tokens += text_tokens;
            progress.text.push_str(&text);
            progress.segments.push(Segment {
                start: (segment.start as f64).mul_add(TIME_PRECISION, offset),
                end: (segment.end as f64).mul_add(TIME_PRECISION, offset),
                text,
                seek,
                tokens: segment.tokens,
            });
        }
        progress.windows += 1;
        Ok(advance)
    }

    /// Feed the decoder its prompt and return the logits after it. A multilingual model
    /// reads the language first, detecting it on the first window of a clip without one.
    fn prompt(
        &self,
        decoder: &mut Decoder<'_>,
        options: &TranscribeOptions,
        progress: &mut Progress,
        no_timestamps: bool,
    ) -> Result<Tensor> {
        let start = self.model.config().decoder_start_token_id;
        let mut suffix = Vec::with_capacity(3);
        if self.generation.lang_to_id.is_empty() {
            suffix.push(start);
            if no_timestamps {
                suffix.push(self.no_timestamps()?);
            }
            return decoder.step(&suffix);
        }
        let first = decoder.step(&[start])?;
        let language = match progress.language {
            Some(token) => token,
            None => *progress.language.insert(self.detect(&first)?),
        };
        let task = self
            .generation
            .task_to_id
            .get(options.task.key())
            .copied()
            .ok_or_else(|| invalid(format!("this Whisper model cannot {}", options.task.key())))?;
        suffix.extend([language, task]);
        if no_timestamps {
            suffix.push(self.no_timestamps()?);
        }
        decoder.step(&suffix)
    }

    fn timestamp_rules(&self) -> Result<TimestampRules> {
        let no_timestamps = self.no_timestamps()?;
        Ok(TimestampRules {
            begin: no_timestamps + 1,
            no_timestamps,
            eos: self
                .generation
                .eos_token_id
                .ok_or_else(|| invalid("missing Whisper end-of-text token"))?,
            max_initial: self.generation.max_initial_timestamp_index,
        })
    }

    /// Text of `tokens`, without timestamps or special tokens.
    fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        let limit = self.no_timestamps()?;
        let text: Vec<u32> = tokens.iter().copied().filter(|&t| t < limit).collect();
        self.tokenizer
            .decode(&text, true)
            .map_err(|error| TurError::Tokenizer(error.to_string()))
    }

    fn no_timestamps(&self) -> Result<u32> {
        self.generation
            .no_timestamps_token_id
            .ok_or_else(|| invalid("missing no-timestamps token"))
    }

    fn language_token(&self, code: &str) -> Result<u32> {
        self.generation
            .lang_to_id
            .get(&format!("<|{}|>", code.to_ascii_lowercase()))
            .copied()
            .ok_or_else(|| invalid(format!("unsupported Whisper language '{code}'")))
    }

    /// The most likely language token after the start token, as Transformers'
    /// `detect_language` picks it.
    fn detect(&self, logits: &Tensor) -> Result<u32> {
        let row = last_row(logits)?;
        self.generation
            .lang_to_id
            .values()
            .copied()
            .max_by(|&a, &b| row[a as usize].total_cmp(&row[b as usize]))
            .ok_or_else(|| invalid("Whisper model has no language tokens"))
    }

    /// Greedy tokens after the prompt until end of text or the decoder's length limit.
    /// Suppressions run before the timestamp rules, the order Transformers applies, so the
    /// timestamps' total mass is weighed against the text tokens that remain.
    fn greedy(
        &self,
        decoder: &mut Decoder<'_>,
        prompt: &Tensor,
        timestamps: Option<&TimestampRules>,
    ) -> Result<Vec<u32>> {
        let eos = self
            .generation
            .eos_token_id
            .ok_or_else(|| invalid("missing Whisper end-of-text token"))?;
        let limit = self
            .generation
            .max_length
            .unwrap_or(usize::MAX)
            .min(self.model.config().max_target_positions);
        let mut row = last_row(prompt)?;
        let mut out = Vec::new();
        loop {
            for &id in &self.generation.suppress_tokens {
                row[id as usize] = f32::NEG_INFINITY;
            }
            if out.is_empty() {
                for &id in &self.generation.begin_suppress_tokens {
                    row[id as usize] = f32::NEG_INFINITY;
                }
            }
            if let Some(rules) = timestamps {
                rules.constrain(&mut row, &out);
            }
            let token = argmax(&row)?;
            if token == eos {
                return Ok(out);
            }
            out.push(token);
            if decoder.position() + 1 >= limit {
                return Ok(out);
            }
            row = last_row(&decoder.step(&[token])?)?;
        }
    }
}

fn last_row(logits: &Tensor) -> Result<Vec<f32>> {
    let rows = logits.dim(0)?;
    Ok(logits
        .get(rows.checked_sub(1).ok_or_else(|| invalid("no logits"))?)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?)
}

/// Index of the largest value; the first on ties, as `torch.argmax`.
fn argmax(row: &[f32]) -> Result<u32> {
    let (index, _) = row
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        });
    u32::try_from(index).map_err(|_| invalid("vocabulary index overflow"))
}
