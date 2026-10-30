//! Greedy Whisper transcription, window by window, as Transformers' `generate` decodes.
use super::{
    config::{Config, Generation},
    model::{Decoder, Whisper},
};
use crate::{
    audio::{Audio, WHISPER_WINDOW},
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

    /// Transcribe (or translate) a clip. Clips beyond 30 s are decoded in consecutive
    /// 30 s windows, so a word spanning a boundary may be cut; Whisper's timestamp-guided
    /// seeking is not implemented.
    pub fn transcribe(&self, clip: &Audio, options: &TranscribeOptions) -> Result<Transcription> {
        let mut language = options
            .language
            .as_deref()
            .map(|code| self.language_token(code))
            .transpose()?;
        let mut text = String::new();
        let (mut tokens, mut windows) = (0, 0);
        for window in clip.samples().chunks(WHISPER_WINDOW) {
            let encoded = self.model.encode(&self.model.features(window)?)?;
            let mut decoder = self.model.decoder(&encoded)?;
            let start = self.model.config().decoder_start_token_id;
            let logits = if self.generation.lang_to_id.is_empty() {
                decoder.step(&[start, self.no_timestamps()?])?
            } else {
                let first = decoder.step(&[start])?;
                let token = match language {
                    Some(token) => token,
                    None => *language.insert(self.detect(&first)?),
                };
                let task = self
                    .generation
                    .task_to_id
                    .get(options.task.key())
                    .copied()
                    .ok_or_else(|| {
                        invalid(format!("this Whisper model cannot {}", options.task.key()))
                    })?;
                decoder.step(&[token, task, self.no_timestamps()?])?
            };
            let decoded = self.greedy(&mut decoder, &logits)?;
            tokens += decoded.len();
            windows += 1;
            text.push_str(
                &self
                    .tokenizer
                    .decode(&decoded, true)
                    .map_err(|error| TurError::Tokenizer(error.to_string()))?,
            );
        }
        let language = language.and_then(|token| {
            self.generation
                .lang_to_id
                .iter()
                .find(|(_, id)| **id == token)
                .and_then(|(key, _)| key.strip_prefix("<|")?.strip_suffix("|>"))
                .map(str::to_owned)
        });
        Ok(Transcription {
            text: text.trim().to_owned(),
            language,
            tokens,
            windows,
        })
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

    /// Greedy tokens after the prompt until end of text or the decoder's position limit.
    fn greedy(&self, decoder: &mut Decoder<'_>, prompt: &Tensor) -> Result<Vec<u32>> {
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
            let token = argmax(&row)?;
            if token == eos {
                return Ok(out);
            }
            out.push(token);
            if decoder.position() >= limit {
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
