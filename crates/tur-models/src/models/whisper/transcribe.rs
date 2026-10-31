//! Greedy Whisper transcription, as Transformers' `generate` decodes and seeks.
use super::{
    alignment::{self, Word},
    config::{Config, Generation},
    fallback::{Fallback, Sampler, logprob},
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
use tur_core::{Result, TurError, execution::invalid, progress::Stage};

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
    /// Time each word from cross-attention; implies `timestamps`.
    pub word_timestamps: bool,
    /// Text the decoder reads before every window, for names and style.
    pub prompt: Option<String>,
    /// Also read the text decoded so far, up to half the decoder's positions.
    pub condition_on_previous_text: bool,
    /// Retry doubtful windows at higher temperatures; `None` decodes each once, greedily.
    pub fallback: Option<Fallback>,
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
    /// Empty unless word timestamps were requested.
    pub words: Vec<Word>,
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
    /// Tokens a conditioned decoder may read, oldest first.
    context: Vec<u32>,
    /// Whether the last window decoded calmly enough for the next to read it.
    condition: bool,
}

/// How one transcription's windows decode.
struct WindowPlan<'a> {
    options: &'a TranscribeOptions,
    fallback: Fallback,
    /// `<|startofprev|>` and the prompt's tokens.
    prompt: Vec<u32>,
    rules: Option<TimestampRules>,
    /// Feature frames holding the clip, without padding.
    speech_frames: usize,
}

/// One accepted decoding of a window.
struct Decoded {
    /// Empty for a window of silence.
    tokens: Vec<u32>,
    silent: bool,
    alignment: Vec<Vec<Vec<f32>>>,
}

/// Samples per feature frame.
const HOP: usize = 160;

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
        let loading = Stage::begin("Loading Whisper", 0, "");
        let model = Whisper::load(dir, format, device)?;
        loading.finish();
        Ok(Self {
            model,
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
        let samples = clip.samples();
        let timed = options.timestamps || options.word_timestamps || samples.len() > WHISPER_WINDOW;
        if options.word_timestamps && self.generation.alignment_heads.is_empty() {
            return Err(invalid("this Whisper model has no alignment heads"));
        }
        let plan = WindowPlan {
            options,
            fallback: options.fallback.clone().unwrap_or_else(|| Fallback {
                no_speech_threshold: None,
                silence_rms: None,
                ..Fallback::fixed(0.0)
            }),
            prompt: self.prompt_ids(options.prompt.as_deref())?,
            rules: timed.then(|| self.timestamp_rules()).transpose()?,
            speech_frames: samples.len() / HOP,
        };
        let mut progress = Progress {
            language: options
                .language
                .as_deref()
                .map(|code| self.language_token(code))
                .transpose()?,
            context: plan.prompt.iter().skip(1).copied().collect(),
            condition: true,
            ..Progress::default()
        };
        if timed && samples.len() > WHISPER_WINDOW {
            self.seek_through(samples, &plan, &mut progress)?;
        } else if timed {
            let mut padded = samples.to_vec();
            padded.resize(WHISPER_WINDOW, 0.0);
            self.seek_through(&padded, &plan, &mut progress)?;
        } else if !plan.fallback.is_quiet(samples) {
            let decoding = Stage::begin("Transcribing", 0, "");
            let encoded = self.model.encode(&self.model.features(samples)?)?;
            let decoded = self.decode_window(&encoded, 0, &plan, &mut progress)?;
            decoding.finish();
            progress.tokens += decoded.tokens.len();
            progress.windows += 1;
            progress.text.push_str(&self.decode_text(&decoded.tokens)?);
        }
        let language = self.language_code(progress.language);
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
        plan: &WindowPlan<'_>,
        progress: &mut Progress,
    ) -> Result<()> {
        let features = self.model.clip_features(samples)?;
        // Audio seconds, not windows: a window advances only as far as its last closed segment.
        let mut stage = Stage::begin(
            "Transcribing",
            (features.frames() * HOP / crate::audio::SAMPLE_RATE as usize) as u64,
            "audio seconds",
        );
        let mut seek = 0;
        let mut seconds = 0;
        while seek < features.frames() {
            let frames = (features.frames() - seek).min(WHISPER_FRAMES);
            if plan
                .fallback
                .is_quiet(&samples[seek * HOP..(seek + frames) * HOP])
            {
                progress.windows += 1;
                seek += frames;
                continue;
            }
            let window = features.window(seek, frames, self.model.device())?;
            seek += self.timed_window(&window, (seek, frames), plan, progress)?;
            let reached = seek.min(features.frames()) * HOP / crate::audio::SAMPLE_RATE as usize;
            stage.advance((reached - seconds) as u64);
            seconds = reached;
        }
        stage.finish();
        Ok(())
    }

    /// One window of `frames` frames from frame `seek`, with timestamps; returns the frames
    /// to advance.
    fn timed_window(
        &self,
        features: &Tensor,
        (seek, frames): (usize, usize),
        plan: &WindowPlan<'_>,
        progress: &mut Progress,
    ) -> Result<usize> {
        let rules = plan
            .rules
            .ok_or_else(|| invalid("timed window without rules"))?;
        let encoded = self.model.encode(features)?;
        let decoded = self.decode_window(&encoded, seek, plan, progress)?;
        if decoded.silent {
            progress.windows += 1;
            return Ok(frames);
        }
        let (segments, advance) = rules.split(&decoded.tokens, frames);
        #[allow(clippy::cast_precision_loss, reason = "clip frames are far below 2^52")]
        let offset = seek as f64 * 0.01;
        let times: Vec<f64> = alignment::token_times(
            &decoded.alignment,
            plan.speech_frames.saturating_sub(seek).max(2),
            self.model.config().median_filter_width,
            decoded.tokens.len(),
        )
        .into_iter()
        .map(|time| time + offset)
        .collect();
        let language = self.language_code(progress.language);
        let mut from = 0;
        for segment in segments {
            let to = from + segment.tokens.len();
            let closes_twice =
                segment.tokens.len() > 2 && segment.tokens[segment.tokens.len() - 2] >= rules.begin;
            let kept = segment.tokens.len() - usize::from(closes_twice);
            progress.context.extend_from_slice(&segment.tokens[..kept]);
            let text_at: Vec<usize> = (0..segment.tokens.len())
                .filter(|&i| segment.tokens[i] < rules.begin)
                .collect();
            if !text_at.is_empty() {
                let words = if plan.options.word_timestamps {
                    alignment::words(
                        &segment.tokens,
                        &text_at,
                        &times[from..to],
                        language.as_deref(),
                        &|tokens| self.decode_text(tokens),
                    )?
                } else {
                    Vec::new()
                };
                let text = self.decode_text(&segment.tokens)?;
                progress.tokens += text_at.len();
                progress.text.push_str(&text);
                #[allow(clippy::cast_precision_loss, reason = "timestamp steps are small")]
                progress.segments.push(Segment {
                    start: (segment.start as f64).mul_add(TIME_PRECISION, offset),
                    end: (segment.end as f64).mul_add(TIME_PRECISION, offset),
                    text,
                    seek,
                    tokens: segment.tokens,
                    words,
                });
            }
            from = to;
        }
        progress.windows += 1;
        Ok(advance)
    }

    /// Decode one window, moving up the fallback's temperatures until an attempt passes.
    fn decode_window(
        &self,
        encoded: &Tensor,
        seek: usize,
        plan: &WindowPlan<'_>,
        progress: &mut Progress,
    ) -> Result<Decoded> {
        self.ensure_language(encoded, progress)?;
        let prefix = self.decoder_prefix(plan, progress);
        let suffix = self.prompt_tokens(plan, progress)?;
        let vocab = self.model.config().vocab_size;
        let attempts = plan.fallback.temperatures.len();
        for (attempt, &temperature) in plan.fallback.temperatures.iter().enumerate() {
            let mut decoder = self.model.decoder(encoded)?;
            let logits = decoder.step(&[prefix.as_slice(), suffix.as_slice()].concat())?;
            let no_speech = if plan.fallback.no_speech_threshold.is_some() {
                // Read where the decoder has just seen its start token, as `OpenAI` does.
                let row = logits.get(prefix.len())?.to_vec1::<f32>()?;
                logprob(&row, self.no_timestamps()? - 1).exp()
            } else {
                0.0
            };
            if plan.options.word_timestamps {
                decoder.record_alignment(&self.generation.alignment_heads);
            }
            let mut sampler = Sampler::new((seek as u64) << 8 | attempt as u64);
            let run = self.generate(
                &mut decoder,
                &logits,
                plan.rules.as_ref(),
                temperature,
                &mut sampler,
            )?;
            if plan.fallback.is_silence(no_speech, run.avg_logprob) {
                return Ok(Decoded {
                    tokens: Vec::new(),
                    silent: true,
                    alignment: Vec::new(),
                });
            }
            if attempt + 1 == attempts
                || !plan.fallback.rejects(&run.scored, vocab, run.avg_logprob)?
            {
                progress.condition = temperature < 0.5;
                return Ok(Decoded {
                    tokens: run.tokens,
                    silent: false,
                    alignment: decoder.alignment(),
                });
            }
        }
        Err(invalid("a fallback needs at least one temperature"))
    }

    /// The tokens a decoder reads before its start token: the prompt, or with conditioning
    /// `<|startofprev|>` and the newest tokens that fit.
    fn decoder_prefix(&self, plan: &WindowPlan<'_>, progress: &Progress) -> Vec<u32> {
        let previous = self.generation.prev_sot_token_id;
        match previous {
            Some(previous)
                if plan.options.condition_on_previous_text
                    && progress.condition
                    && !progress.context.is_empty() =>
            {
                let limit = self.model.config().max_target_positions / 2 - 1;
                let newest = progress.context.len().saturating_sub(limit);
                std::iter::once(previous)
                    .chain(progress.context[newest..].iter().copied())
                    .collect()
            }
            _ => plan.prompt.clone(),
        }
    }

    /// `<|startofprev|>` and the tokens of ` {text}`, as `get_prompt_ids` encodes them.
    fn prompt_ids(&self, text: Option<&str>) -> Result<Vec<u32>> {
        let Some(text) = text.map(str::trim).filter(|text| !text.is_empty()) else {
            return Ok(Vec::new());
        };
        let previous = self
            .generation
            .prev_sot_token_id
            .ok_or_else(|| invalid("this Whisper model takes no prompt"))?;
        let eos = self
            .generation
            .eos_token_id
            .ok_or_else(|| invalid("missing Whisper end-of-text token"))?;
        let encoded = self
            .tokenizer
            .encode(format!(" {text}"), false)
            .map_err(|error| TurError::Tokenizer(error.to_string()))?;
        let tokens = encoded.get_ids();
        if tokens.iter().any(|&token| token >= eos) {
            return Err(invalid("the prompt holds a special token"));
        }
        if tokens.len() > self.model.config().max_target_positions / 2 - 1 {
            return Err(invalid("the prompt is too long for the decoder"));
        }
        Ok(std::iter::once(previous)
            .chain(tokens.iter().copied())
            .collect())
    }

    /// Detect the language from the first window of a clip without one.
    fn ensure_language(&self, encoded: &Tensor, progress: &mut Progress) -> Result<()> {
        if progress.language.is_none() && !self.generation.lang_to_id.is_empty() {
            let first = self
                .model
                .decoder(encoded)?
                .step(&[self.model.config().decoder_start_token_id])?;
            progress.language = Some(self.detect(&first)?);
        }
        Ok(())
    }

    /// Start token, language and task, and `<|notimestamps|>` when decoding without them.
    fn prompt_tokens(&self, plan: &WindowPlan<'_>, progress: &Progress) -> Result<Vec<u32>> {
        let mut tokens = vec![self.model.config().decoder_start_token_id];
        if let Some(language) = progress.language {
            let key = plan.options.task.key();
            let task = self
                .generation
                .task_to_id
                .get(key)
                .copied()
                .ok_or_else(|| invalid(format!("this Whisper model cannot {key}")))?;
            tokens.extend([language, task]);
        }
        if plan.rules.is_none() {
            tokens.push(self.no_timestamps()?);
        }
        Ok(tokens)
    }

    fn language_code(&self, token: Option<u32>) -> Option<String> {
        let token = token?;
        self.generation
            .lang_to_id
            .iter()
            .find(|(_, id)| **id == token)
            .and_then(|(key, _)| key.strip_prefix("<|")?.strip_suffix("|>"))
            .map(str::to_owned)
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

    /// Tokens after the prompt until end of text or the decoder's length limit: greedy at
    /// temperature 0, sampled above it. Suppressions run before the timestamp rules, the
    /// order Transformers applies, so the timestamps' total mass is weighed against the
    /// text tokens that remain.
    fn generate(
        &self,
        decoder: &mut Decoder<'_>,
        prompt: &Tensor,
        timestamps: Option<&TimestampRules>,
        temperature: f32,
        sampler: &mut Sampler,
    ) -> Result<Run> {
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
        let mut total = 0.0;
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
            let token = if temperature > 0.0 {
                sampler.draw(&row, temperature)
            } else {
                argmax(&row)?
            };
            total += logprob(&row, token);
            if token == eos {
                return Ok(Run::finished(out, total, Some(eos)));
            }
            out.push(token);
            if decoder.position() + 1 >= limit {
                return Ok(Run::finished(out, total, None));
            }
            row = last_row(&decoder.step(&[token])?)?;
        }
    }
}

/// One attempt at a window.
struct Run {
    /// Generated tokens without end of text.
    tokens: Vec<u32>,
    /// The same with end of text, which the fallback's measures count.
    scored: Vec<u32>,
    avg_logprob: f64,
}

impl Run {
    fn finished(tokens: Vec<u32>, total: f64, eos: Option<u32>) -> Self {
        let mut scored = tokens.clone();
        scored.extend(eos);
        #[allow(clippy::cast_precision_loss, reason = "token counts are small")]
        let avg_logprob = total / scored.len().max(1) as f64;
        Self {
            tokens,
            scored,
            avg_logprob,
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
