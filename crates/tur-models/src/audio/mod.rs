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

/// How a model frames one clip in its prompt. The chat template renders `placeholder`
/// once per audio part; the clip then occupies `begin`, one position per soft token and
/// `end`. Each soft-token position feeds the model `slot`, whose input embedding the
/// encoder's row replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioPrompt {
    pub placeholder: u32,
    pub begin: u32,
    pub end: u32,
    pub slot: u32,
}

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

    /// 64-bit FNV-1a of the samples' bits: distinct clips are told apart by content, as
    /// prefix reuse requires, not by how they were encoded.
    pub fn fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for sample in self.samples.iter() {
            for byte in sample.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        hash
    }
}
