//! Audio input: WAV decoding, resampling to the 16 kHz model rate and log-mel features.
mod features;
mod fft;
mod resample;
mod wav;

pub(crate) use features::{
    GemmaFeatures, WHISPER_FRAMES, WHISPER_WINDOW, WhisperClip, WhisperFeatures, gemma_frames,
};
use std::{sync::Arc, time::Duration};
use tur_core::{Result, execution::invalid};

/// Sample rate every audio model here consumes.
pub const SAMPLE_RATE: u32 = 16_000;

/// Longest clip decoded from one WAV file: 30 minutes. A transcription runs window by
/// window, so this bounds host memory, not model work per pass.
pub const MAX_DURATION: Duration = Duration::from_mins(30);

/// One mono clip at [`SAMPLE_RATE`], shared cheaply between a request and its model.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    samples: Arc<[f32]>,
}

impl Audio {
    /// Decode a RIFF/WAVE file of integer PCM (8 to 32 bits) or float samples, mix its
    /// channels to mono and resample it to 16 kHz.
    pub fn from_wav(bytes: &[u8]) -> Result<Self> {
        // Bound decoded frames at the highest common rate before resampling.
        let limit = usize::try_from(MAX_DURATION.as_secs() * 192_000)
            .map_err(|_| invalid("audio limit overflow"))?;
        let decoded = wav::decode(bytes, limit)?;
        Self::from_samples(&decoded.samples, decoded.rate)
    }

    /// A mono clip of `samples` recorded at `rate` Hz.
    pub fn from_samples(samples: &[f32], rate: u32) -> Result<Self> {
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(invalid("audio samples must be finite"));
        }
        let samples = resample::resample(samples, rate, SAMPLE_RATE)?;
        if samples.is_empty() {
            return Err(invalid("audio clip is empty"));
        }
        let clip = Self {
            samples: samples.into(),
        };
        if clip.duration() > MAX_DURATION {
            return Err(invalid(format!(
                "audio is longer than {} minutes",
                MAX_DURATION.as_secs() / 60
            )));
        }
        Ok(clip)
    }

    /// Mono samples at [`SAMPLE_RATE`].
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.samples.len() as f64 / f64::from(SAMPLE_RATE))
    }

    /// Identifies the samples: distinct clips are told apart by content, as prefix reuse
    /// requires, not by how they were encoded.
    pub fn digest(&self) -> crate::media::Digest {
        let bytes: Vec<u8> = self
            .samples
            .iter()
            .flat_map(|sample| sample.to_bits().to_le_bytes())
            .collect();
        crate::media::digest(&[b"audio", &bytes])
    }
}
