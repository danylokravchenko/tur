//! Band-limited sample-rate conversion by a Hann-windowed sinc kernel.
use tur_core::{Result, execution::invalid};

/// Zero crossings of the sinc kernel on each side of an output sample. Thirty-two keeps
/// the stopband below -60 dB for speech-band conversions from 8 to 96 kHz.
const ZEROS: f64 = 32.0;
/// Passband edge as a share of the lower Nyquist frequency, leaving room for the
/// transition band so content near it does not alias.
const ROLLOFF: f64 = 0.94;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Kernel weights for every fractional phase: output sample `n` sits at input time
/// `n · from / to`, whose fraction takes one of `to / gcd(from, to)` values, so one row
/// per phase replaces a sine and cosine per tap.
fn table(from: u64, to: u64, reach: usize, cutoff: f64, half: f64) -> Vec<f32> {
    let step = gcd(from, to);
    let phases = to / step;
    let taps = 2 * reach + 1;
    let mut weights = Vec::with_capacity(usize::try_from(phases).unwrap_or(0) * taps);
    for phase in 0..phases {
        #[allow(
            clippy::cast_precision_loss,
            reason = "phase fractions and tap offsets are small"
        )]
        let fraction = (phase * step) as f64 / to as f64;
        for tap in 0..taps {
            #[allow(clippy::cast_precision_loss, reason = "tap offsets are small")]
            let t = tap as f64 - reach as f64 - fraction;
            let weight = if t.abs() >= half {
                0.0
            } else {
                let window = 0.5f64.mul_add((std::f64::consts::PI * t / half).cos(), 0.5);
                let argument = std::f64::consts::PI * cutoff * t;
                let sinc = if argument == 0.0 {
                    1.0
                } else {
                    argument.sin() / argument
                };
                cutoff * sinc * window
            };
            #[allow(
                clippy::cast_possible_truncation,
                reason = "kernel weights are stored in f32"
            )]
            weights.push(weight as f32);
        }
    }
    weights
}

/// Convert `input` from `from` Hz to `to` Hz. Output sample positions are computed
/// exactly in integers, so long inputs do not drift; samples beyond the input are zero.
pub(super) fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == 0 || to == 0 {
        return Err(invalid("sample rates must be positive"));
    }
    if from == to {
        return Ok(input.to_vec());
    }
    let (from64, to64) = (u64::from(from), u64::from(to));
    let length = u64::try_from(input.len())
        .ok()
        .and_then(|n| n.checked_mul(to64))
        .map(|n| n.div_ceil(from64))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("resampled length overflow"))?;
    let cutoff = ROLLOFF * (f64::from(to) / f64::from(from)).min(1.0);
    let half = ZEROS / cutoff;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the kernel half-width is a small positive number of samples"
    )]
    let reach = half.ceil() as usize;
    let taps = 2 * reach + 1;
    let weights = table(from64, to64, reach, cutoff, half);
    let step = gcd(from64, to64);
    // Zero padding on both sides lets every output read a full kernel window.
    let mut padded = vec![0.0f32; reach];
    padded.extend_from_slice(input);
    padded.resize(padded.len() + reach + 1, 0.0);
    let mut output = Vec::with_capacity(length);
    for n in 0..length as u64 {
        let position = n * from64;
        let base = usize::try_from(position / to64).map_err(|_| invalid("sample index"))?;
        let phase = usize::try_from((position % to64) / step).map_err(|_| invalid("phase"))?;
        let kernel = weights
            .get(phase * taps..(phase + 1) * taps)
            .ok_or_else(|| invalid("resampling phase out of range"))?;
        let window = padded
            .get(base..base + taps)
            .ok_or_else(|| invalid("resampling window out of range"))?;
        output.push(kernel.iter().zip(window).map(|(w, x)| w * x).sum());
    }
    Ok(output)
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert resampled signals after fallible conversion"
)]
mod tests {
    use super::*;

    fn sine(rate: u32, hz: f64, seconds: f64) -> Vec<f32> {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "test signal synthesis"
        )]
        (0..(f64::from(rate) * seconds) as usize)
            .map(|i| (std::f64::consts::TAU * hz * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    #[test]
    fn test_resampling_keeps_passband_tones_and_removes_aliases() -> Result<()> {
        let down = resample(&sine(44_100, 1000.0, 0.5), 44_100, 16_000)?;
        assert_eq!(down.len(), 8000);
        let expected = sine(16_000, 1000.0, 0.5);
        // Skip the edges, where the kernel runs past the signal.
        let error = down[200..7800]
            .iter()
            .zip(&expected[200..7800])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 2e-3, "passband error {error}");
        // 12 kHz is above the 8 kHz output Nyquist frequency and must not fold back.
        let alias = resample(&sine(48_000, 12_000.0, 0.5), 48_000, 16_000)?;
        let peak = alias[200..7800]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        assert!(peak < 1e-3, "alias peak {peak}");
        let up = resample(&sine(8000, 440.0, 0.25), 8000, 16_000)?;
        assert_eq!(up.len(), 4000);
        assert!(resample(&[0.0], 0, 16_000).is_err());
        Ok(())
    }
}
