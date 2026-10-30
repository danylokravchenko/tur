//! Log-mel spectrograms in the exact framing of each model's Transformers extractor,
//! computed on the host and uploaded once per clip.
use super::{SAMPLE_RATE, fft::Fft};
use candle_core::{Device, Tensor};
use tur_core::{Result, execution::invalid};

/// Mel frequency scale of a filterbank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MelScale {
    /// `2595 · log10(1 + f / 700)`, unnormalized filters (Gemma 4).
    Htk,
    /// Linear below 1 kHz and logarithmic above, area-normalized filters (Whisper).
    Slaney,
}

impl MelScale {
    fn mel(self, hz: f64) -> f64 {
        match self {
            Self::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
            Self::Slaney if hz >= 1000.0 => 15.0 + (hz / 1000.0).ln() * 27.0 / 6.4f64.ln(),
            Self::Slaney => 3.0 * hz / 200.0,
        }
    }
    fn hz(self, mel: f64) -> f64 {
        match self {
            Self::Htk => 700.0 * (10f64.powf(mel / 2595.0) - 1.0),
            Self::Slaney if mel >= 15.0 => 1000.0 * (6.4f64.ln() / 27.0 * (mel - 15.0)).exp(),
            Self::Slaney => 200.0 * mel / 3.0,
        }
    }
}

/// Triangular `[bins, mels]` filters between 0 Hz and `max_hz`, as Transformers'
/// `mel_filter_bank`, which follows librosa and torchaudio.
#[allow(
    clippy::cast_precision_loss,
    reason = "bin and filter counts are small integers"
)]
fn filterbank(bins: usize, mels: usize, max_hz: f64, scale: MelScale) -> Vec<f64> {
    let (low, high) = (scale.mel(0.0), scale.mel(max_hz));
    let edges: Vec<f64> = (0..mels + 2)
        .map(|i| scale.hz(low + (high - low) * i as f64 / (mels + 1) as f64))
        .collect();
    let nyquist = f64::from(SAMPLE_RATE / 2);
    let mut filters = vec![0.0; bins * mels];
    for bin in 0..bins {
        let hz = nyquist * bin as f64 / (bins - 1) as f64;
        for m in 0..mels {
            let down = (hz - edges[m]) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
            let mut weight = down.min(up).max(0.0);
            if scale == MelScale::Slaney {
                weight *= 2.0 / (edges[m + 2] - edges[m]);
            }
            filters[bin * mels + m] = weight;
        }
    }
    filters
}

/// Short-time mel energies: frames of `window` samples every `hop`, a periodic Hann
/// window, an `fft`-point transform (zero-padded past the window) and a filterbank.
///
/// Each windowed frame is rounded to F32 before an F64 transform, as Transformers
/// computes it. Near-silent bins carry `ln(mel + 0.001)` an error 1000 times their
/// energy's, and an F32 transform moved the Gemma reference's features by 1.1e-4 of their
/// range, where this order matches it to 5e-8.
pub(crate) struct Spectrogram {
    window: Vec<f32>,
    hop: usize,
    bins: usize,
    mels: usize,
    fft: Fft,
    /// Each frequency bin's nonzero `(mel, weight)` filter entries: at most two.
    filters: Vec<Vec<(usize, f64)>>,
    /// Square the magnitude (power spectrum) instead of taking its root.
    power: bool,
}

impl Spectrogram {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "window indices are small; the window is stored in f32 as Transformers does"
    )]
    fn new(
        (window, fft, hop): (usize, usize, usize),
        mels: usize,
        scale: MelScale,
        power: bool,
    ) -> Result<Self> {
        if window == 0 || window > fft || hop == 0 || mels == 0 {
            return Err(invalid("invalid spectrogram framing"));
        }
        let bins = fft / 2 + 1;
        let dense = filterbank(bins, mels, 8000.0, scale);
        let filters = (0..bins)
            .map(|bin| {
                (0..mels)
                    .map(|m| (m, dense[bin * mels + m]))
                    .filter(|(_, w)| *w != 0.0)
                    .collect()
            })
            .collect();
        Ok(Self {
            window: (0..window)
                .map(|i| {
                    0.5f64.mul_add(
                        -(std::f64::consts::TAU * i as f64 / window as f64).cos(),
                        0.5,
                    ) as f32
                })
                .collect(),
            hop,
            bins,
            mels,
            fft: Fft::new(fft)?,
            filters,
            power,
        })
    }

    /// Row-major `[frames, mels]` energies of `frames` frames of `samples`, the first at
    /// offset 0.
    fn mel(&self, samples: &[f32], frames: usize) -> Result<Vec<f64>> {
        let width = self.window.len();
        let needed = frames
            .checked_sub(1)
            .and_then(|f| f.checked_mul(self.hop))
            .and_then(|start| start.checked_add(width))
            .ok_or_else(|| invalid("audio is too short for one frame"))?;
        if needed > samples.len() {
            return Err(invalid("spectrogram frames exceed the samples"));
        }
        let mut energies = vec![0.0f64; frames * self.mels];
        let mut frame = vec![0.0f64; width];
        for (index, row) in energies.chunks_exact_mut(self.mels).enumerate() {
            let start = index * self.hop;
            for ((slot, &x), &w) in frame
                .iter_mut()
                .zip(&samples[start..start + width])
                .zip(&self.window)
            {
                *slot = f64::from(x * w);
            }
            for (bin, (re, im)) in self.fft.real(&frame, self.bins).into_iter().enumerate() {
                let power = re.mul_add(re, im * im);
                let value = if self.power { power } else { power.sqrt() };
                for &(m, weight) in &self.filters[bin] {
                    row[m] = value.mul_add(weight, row[m]);
                }
            }
        }
        Ok(energies)
    }
}

/// Gemma 4's semicausal USM features: 20 ms frames every 10 ms, a 512-point transform,
/// 128 HTK mel bins and `ln(mel + 0.001)`.
pub(crate) struct GemmaFeatures {
    spectrogram: Spectrogram,
    device: Device,
}

/// Frames a Gemma clip of `samples` produces, every one valid; see [`GemmaFeatures::extract`].
pub(crate) fn gemma_frames(samples: usize) -> usize {
    // Frame `i` ends at sample `160 · i + 160` of the unpadded clip and is valid while
    // that end is inside it.
    samples.saturating_sub(1) / 160
}

impl GemmaFeatures {
    pub(crate) fn new(mels: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            spectrogram: Spectrogram::new((320, 512, 160), mels, MelScale::Htk, false)?,
            device: device.clone(),
        })
    }

    /// `[frames, mels]` features of the valid frames only, on the model device.
    /// Transformers pads the clip to 128 samples and masks the frames that reach the
    /// padding; every later layer only looks back or is masked, so dropping those frames
    /// here is exact.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "features are F32 model inputs"
    )]
    pub(crate) fn extract(&self, samples: &[f32]) -> Result<Tensor> {
        let frames = gemma_frames(samples.len());
        if frames == 0 {
            return Err(invalid("audio clip is too short"));
        }
        // Half a window of leading silence centres the first frame on the first sample.
        let mut padded = vec![0.0f32; 160];
        padded.extend_from_slice(samples);
        padded.resize(padded.len().max((frames - 1) * 160 + 320), 0.0);
        let features: Vec<f32> = self
            .spectrogram
            .mel(&padded, frames)?
            .into_iter()
            .map(|mel| (mel + 1e-3).ln() as f32)
            .collect();
        Ok(Tensor::from_vec(
            features,
            (frames, self.spectrogram.mels),
            &self.device,
        )?)
    }
}

/// Whisper's features: a 30 s window, 25 ms frames every 10 ms with reflected edges,
/// a power spectrum, Slaney mel bins, `log10` clamped to 8 below its peak and scaled.
pub(crate) struct WhisperFeatures {
    spectrogram: Spectrogram,
    device: Device,
}

/// Samples in one Whisper window: 30 s.
pub(crate) const WHISPER_WINDOW: usize = 480_000;
/// Feature frames in one Whisper window.
pub(crate) const WHISPER_FRAMES: usize = 3000;

impl WhisperFeatures {
    pub(crate) fn new(mels: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            spectrogram: Spectrogram::new((400, 400, 160), mels, MelScale::Slaney, true)?,
            device: device.clone(),
        })
    }

    /// `[mels, 3000]` features of at most one window, zero-padded to 30 s.
    pub(crate) fn extract(&self, samples: &[f32]) -> Result<Tensor> {
        if samples.len() > WHISPER_WINDOW {
            return Err(invalid("a Whisper window holds at most 30 s"));
        }
        let mut clip = samples.to_vec();
        clip.resize(WHISPER_WINDOW, 0.0);
        self.clip(&clip)?.window(0, WHISPER_FRAMES, &self.device)
    }

    /// Features of a whole clip of at least one frame, normalized by its loudest frame.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "features are F32 model inputs"
    )]
    pub(crate) fn clip(&self, samples: &[f32]) -> Result<WhisperClip> {
        const REFLECTED: usize = 200;
        if samples.len() <= REFLECTED {
            return Err(invalid("audio is too short for a Whisper frame"));
        }
        // Centre frames by reflecting 200 samples at each edge, as `torch.stft`.
        let mut padded = Vec::with_capacity(samples.len() + 2 * REFLECTED);
        padded.extend(samples[1..=REFLECTED].iter().rev());
        padded.extend_from_slice(samples);
        padded.extend(
            samples[samples.len() - REFLECTED - 1..samples.len() - 1]
                .iter()
                .rev(),
        );
        // The transform yields one frame more than Whisper keeps.
        let frames = samples.len() / self.spectrogram.hop;
        let log: Vec<f64> = self
            .spectrogram
            .mel(&padded, frames)?
            .into_iter()
            .map(|mel| mel.max(1e-10).log10())
            .collect();
        let floor = log.iter().copied().fold(f64::NEG_INFINITY, f64::max) - 8.0;
        let mels = self.spectrogram.mels;
        let mut features = vec![0.0f32; mels * frames];
        for (frame, row) in log.chunks_exact(mels).enumerate() {
            for (m, &value) in row.iter().enumerate() {
                features[m * frames + frame] = ((value.max(floor) + 4.0) / 4.0) as f32;
            }
        }
        Ok(WhisperClip {
            features,
            mels,
            frames,
        })
    }
}

/// Row-major `[mels, frames]` features of a whole clip, on the host.
pub(crate) struct WhisperClip {
    features: Vec<f32>,
    mels: usize,
    frames: usize,
}

impl WhisperClip {
    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    /// `[mels, 3000]` features from frame `seek`: `frames` of them, then zeros, as
    /// Transformers pads a clip's last window.
    pub(crate) fn window(&self, seek: usize, frames: usize, device: &Device) -> Result<Tensor> {
        let end = seek
            .checked_add(frames)
            .filter(|&end| end <= self.frames && frames <= WHISPER_FRAMES)
            .ok_or_else(|| invalid("Whisper window outside the clip"))?;
        let mut window = vec![0.0f32; self.mels * WHISPER_FRAMES];
        for (row, source) in window
            .as_chunks_mut::<WHISPER_FRAMES>()
            .0
            .iter_mut()
            .zip(self.features.chunks_exact(self.frames))
        {
            row[..frames].copy_from_slice(&source[seek..end]);
        }
        Ok(Tensor::from_vec(
            window,
            (self.mels, WHISPER_FRAMES),
            device,
        )?)
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert filterbank values after fallible construction"
)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_scales_round_trip_and_filters_tile_the_band() {
        for scale in [MelScale::Htk, MelScale::Slaney] {
            for hz in [0.0, 440.0, 1000.0, 7999.0] {
                assert!((scale.hz(scale.mel(hz)) - hz).abs() < 1e-6);
            }
        }
        // Unnormalized HTK filters overlap so that adjacent triangles sum to one.
        let htk = filterbank(257, 128, 8000.0, MelScale::Htk);
        let sum: f64 = (0..128).map(|m| htk[100 * 128 + m]).sum();
        assert!((sum - 1.0).abs() < 1e-9, "{sum}");
        assert_eq!(gemma_frames(19_744), 123);
        assert_eq!(gemma_frames(160), 0);
    }

    fn golden(model: &str) -> Result<std::collections::HashMap<String, Tensor>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tur/tests/fixtures/audio")
            .join(model)
            .join("golden.safetensors");
        Ok(candle_core::safetensors::load(path, &Device::Cpu)?)
    }

    fn tone() -> Result<Vec<f32>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tur/tests/fixtures/audio/tone.wav");
        Ok(super::super::Audio::from_wav(&std::fs::read(path)?)?
            .samples()
            .to_vec())
    }

    /// Largest absolute difference, relative to the reference's largest magnitude.
    fn relative(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let scale = expected.abs()?.max_all()?.to_scalar::<f32>()?;
        let diff = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        Ok(diff / scale)
    }

    #[test]
    fn test_gemma_and_whisper_features_match_transformers() -> Result<()> {
        let samples = tone()?;
        let gemma = golden("gemma4")?;
        let expected = gemma.get("features").ok_or_else(|| invalid("features"))?;
        let actual = GemmaFeatures::new(128, &Device::Cpu)?.extract(&samples)?;
        assert_eq!(actual.dims(), expected.dims());
        let error = relative(&actual, expected)?;
        assert!(error < 1e-5, "gemma features: {error}");
        let whisper = golden("whisper")?;
        let expected = whisper.get("features").ok_or_else(|| invalid("features"))?;
        let actual = WhisperFeatures::new(128, &Device::Cpu)?.extract(&samples)?;
        assert_eq!(actual.dims(), expected.dims());
        // The reference runs torch's F32 STFT, whose rounding alone is about 3e-5.
        let error = relative(&actual, expected)?;
        assert!(error < 1e-4, "whisper features: {error}");
        Ok(())
    }
}
