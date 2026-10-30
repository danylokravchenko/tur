//! Retrying a window at higher temperatures when its decoding looks wrong, as `OpenAI`'s
//! Whisper does and Transformers' `generate` offers: repetitive text compresses too well,
//! and a guess has a low average log-probability.
use flate2::{Compression, write::ZlibEncoder};
use std::io::Write as _;
use tur_core::Result;

/// The temperatures a window may decode at and when to try the next one. The last attempt
/// stands whatever it looks like.
#[derive(Clone, Debug, PartialEq)]
pub struct Fallback {
    /// Tried in order; 0 is greedy.
    pub temperatures: Vec<f32>,
    /// Retry above this ratio of raw to compressed token bytes.
    pub max_compression_ratio: f64,
    /// Retry below this mean log-probability of the chosen tokens.
    pub min_avg_logprob: f64,
    /// Above this probability of `<|nospeech|>`, a window that also falls below
    /// `min_avg_logprob` is silence: it yields no text and no retry.
    pub no_speech_threshold: Option<f64>,
    /// A window whose loudest 20 ms stays below this RMS is not decoded at all. The model
    /// can confidently invent text over silence, which `no_speech_threshold` lets through.
    pub silence_rms: Option<f32>,
}

impl Default for Fallback {
    /// `OpenAI` Whisper's defaults.
    fn default() -> Self {
        Self {
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            max_compression_ratio: 2.4,
            min_avg_logprob: -1.0,
            no_speech_threshold: Some(0.6),
            // About -50 dBFS, under the noise floor of most speech recordings.
            silence_rms: Some(0.003),
        }
    }
}

impl Fallback {
    /// One attempt at `temperature`, with nothing to retry.
    pub fn fixed(temperature: f32) -> Self {
        Self {
            temperatures: vec![temperature],
            ..Self::default()
        }
    }

    /// Whether a window with these measures holds no speech.
    pub(super) fn is_silence(&self, no_speech: f64, avg_logprob: f64) -> bool {
        self.no_speech_threshold
            .is_some_and(|threshold| no_speech > threshold && avg_logprob < self.min_avg_logprob)
    }

    /// Whether `samples` at 16 kHz never rise above the silence level.
    pub(super) fn is_quiet(&self, samples: &[f32]) -> bool {
        self.silence_rms.is_some_and(|limit| {
            samples.chunks(320).all(|frame| {
                let energy: f32 = frame.iter().map(|s| s * s).sum();
                #[allow(clippy::cast_precision_loss, reason = "frames are 320 samples")]
                let mean = energy / frame.len() as f32;
                mean.sqrt() < limit
            })
        })
    }

    pub(super) fn rejects(&self, tokens: &[u32], vocab: usize, avg_logprob: f64) -> Result<bool> {
        Ok(
            compression_ratio(tokens, vocab)? > self.max_compression_ratio
                || avg_logprob < self.min_avg_logprob,
        )
    }
}

/// Raw over zlib-compressed bytes of `tokens`, each stored as Transformers'
/// `_retrieve_compression_ratio` stores them.
pub(super) fn compression_ratio(tokens: &[u32], vocab: usize) -> Result<f64> {
    if tokens.is_empty() {
        return Ok(0.0);
    }
    let width = (vocab.max(1).ilog2() as usize / 8 + 1).min(4);
    let raw: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes().into_iter().take(width))
        .collect();
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;
    #[allow(clippy::cast_precision_loss, reason = "token byte counts are small")]
    Ok(raw.len() as f64 / compressed.len() as f64)
}

/// `ln(softmax(row)[token])`.
pub(super) fn logprob(row: &[f32], token: u32) -> f64 {
    let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = row.iter().map(|&v| f64::from(v - top).exp()).sum();
    f64::from(row[token as usize] - top) - total.ln()
}

/// A deterministic generator for fallback sampling: a window's retries repeat exactly.
pub(super) struct Sampler(u64);

impl Sampler {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// `SplitMix64`, as a uniform value in `[0, 1)`.
    #[allow(clippy::cast_precision_loss, reason = "53 bits of a uniform draw")]
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A token from `softmax(row / temperature)`; masked entries are never drawn.
    pub(super) fn draw(&mut self, row: &[f32], temperature: f32) -> u32 {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = f64::from(temperature);
        let weight = |v: f32| (f64::from(v - top) / scale).exp();
        let total: f64 = row.iter().map(|&v| weight(v)).sum();
        let mut left = self.uniform() * total;
        let mut chosen = 0;
        for (index, &value) in row.iter().enumerate() {
            let w = weight(value);
            if w > 0.0 {
                chosen = index;
                left -= w;
                if left <= 0.0 {
                    break;
                }
            }
        }
        u32::try_from(chosen).unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_loops_compress_past_the_threshold_and_speech_does_not() -> Result<()> {
        let fallback = Fallback::default();
        let looped: Vec<u32> = [440, 1702, 6292].repeat(100);
        assert!(fallback.rejects(&looped, 51866, -0.1)?);
        let varied: Vec<u32> = (0..60).map(|i| 300 + i * 797 % 40_000).collect();
        assert!(compression_ratio(&varied, 51866)? < 1.6);
        assert!(!fallback.rejects(&varied, 51866, -0.1)?);
        assert!(fallback.rejects(&varied, 51866, -1.5)?);
        Ok(())
    }

    #[test]
    fn test_silence_needs_both_measures() {
        let fallback = Fallback::default();
        assert!(fallback.is_silence(0.9, -1.5));
        // Confident text over a likely no-speech token, and unclear speech, are both kept.
        assert!(!fallback.is_silence(0.9, -0.2));
        assert!(!fallback.is_silence(0.3, -1.5));
        let off = Fallback {
            no_speech_threshold: None,
            ..Fallback::default()
        };
        assert!(!off.is_silence(0.9, -1.5));
    }

    #[test]
    fn test_quiet_windows_have_no_loud_frame() {
        let fallback = Fallback::default();
        let mut samples = vec![0.0005; 16_000];
        assert!(fallback.is_quiet(&samples));
        // One 20 ms burst anywhere makes the window worth decoding.
        samples[8_000..8_320].fill(0.05);
        assert!(!fallback.is_quiet(&samples));
        let off = Fallback {
            silence_rms: None,
            ..Fallback::default()
        };
        assert!(!off.is_quiet(&[0.0; 320]));
    }

    #[test]
    fn test_logprob_is_the_softmax_of_the_chosen_token() {
        let row = [0.0f32.ln(), 1.0f32.ln(), 3.0f32.ln()];
        assert!((logprob(&row, 2) - 0.75f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_draws_follow_the_temperature_and_skip_masked_tokens() {
        let row = [f32::NEG_INFINITY, 0.0, 2.0, f32::NEG_INFINITY];
        let share = |temperature: f32| {
            let mut sampler = Sampler::new(7);
            let draws: Vec<u32> = (0..20_000)
                .map(|_| sampler.draw(&row, temperature))
                .collect();
            assert!(draws.iter().all(|&t| t == 1 || t == 2));
            draws.iter().filter(|&&t| t == 2).count() as f64 / 20_000.0
        };
        // softmax([0, 2]) and softmax([0, 2] / 0.5), with a 4-sigma allowance.
        assert!((share(1.0) - 0.8808).abs() < 0.01);
        assert!((share(0.5) - 0.9820).abs() < 0.005);
    }
}
