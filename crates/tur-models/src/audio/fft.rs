#![allow(
    clippy::many_single_char_names,
    reason = "tensor names follow the reference equations"
)]
//! Mixed-radix F64 discrete Fourier transform for sizes whose prime factors are 2, 3 or 5.
use tur_core::{Result, execution::invalid};

type Complex = (f64, f64);

/// A transform of one size with its twiddle factors `e^{-2πij/n}` precomputed, so the
/// recursion evaluates no trigonometry.
pub(super) struct Fft {
    size: usize,
    twiddles: Vec<Complex>,
}

fn factor(n: usize) -> Option<usize> {
    [2, 3, 5].into_iter().find(|p| n.is_multiple_of(*p))
}

impl Fft {
    pub(super) fn new(size: usize) -> Result<Self> {
        let mut rest = size;
        while rest > 1 {
            rest /= factor(rest).ok_or_else(|| invalid("FFT sizes must factor into 2, 3 and 5"))?;
        }
        if size == 0 {
            return Err(invalid("empty FFT"));
        }
        #[allow(clippy::cast_precision_loss, reason = "FFT sizes are small")]
        let twiddles = (0..size)
            .map(|j| {
                let angle = -std::f64::consts::TAU * j as f64 / size as f64;
                (angle.cos(), angle.sin())
            })
            .collect();
        Ok(Self { size, twiddles })
    }

    /// The first `bins` coefficients of the transform of `frame`, zero-padded to the size.
    pub(super) fn real(&self, frame: &[f64], bins: usize) -> Vec<Complex> {
        let mut input = vec![(0.0, 0.0); self.size];
        for (slot, &x) in input.iter_mut().zip(frame) {
            slot.0 = x;
        }
        let mut output = self.transform(&input);
        output.truncate(bins);
        output
    }

    /// Decimation in time: split into `p` interleaved subsequences, transform each and
    /// combine them with twiddles `W_n^{r·k}`, read from the full table at stride
    /// `size / n`.
    fn transform(&self, x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        let Some(p) = factor(n) else {
            return x.to_vec();
        };
        let m = n / p;
        let parts: Vec<Vec<Complex>> = (0..p)
            .map(|r| {
                let sub: Vec<Complex> = x.iter().skip(r).step_by(p).copied().collect();
                self.transform(&sub)
            })
            .collect();
        let stride = self.size / n;
        let mut out = vec![(0.0, 0.0); n];
        for (index, slot) in out.iter_mut().enumerate() {
            let k = index % m;
            let (mut re, mut im) = (0.0, 0.0);
            for (r, part) in parts.iter().enumerate() {
                let (wr, wi) = self.twiddles[(r * index) % n * stride];
                let (xr, xi) = part[k];
                re += wr.mul_add(xr, -(wi * xi));
                im += wr.mul_add(xi, wi * xr);
            }
            *slot = (re, im);
        }
        out
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert transform values after fallible construction"
)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_radix_transform_matches_direct_sums() -> Result<()> {
        for size in [400, 512, 30] {
            let fft = Fft::new(size)?;
            #[allow(clippy::cast_precision_loss, reason = "test signal")]
            let frame: Vec<f64> = (0..size).map(|i| ((i * 7 % 13) as f64).sin()).collect();
            let fast = fft.real(&frame, size / 2 + 1);
            for (k, &(re, im)) in fast.iter().enumerate() {
                #[allow(clippy::cast_precision_loss, reason = "test indices")]
                let (dr, di) = frame.iter().enumerate().fold((0.0, 0.0), |(a, b), (i, x)| {
                    let angle = -std::f64::consts::TAU * (i * k) as f64 / size as f64;
                    (a + x * angle.cos(), b + x * angle.sin())
                });
                assert!(
                    (re - dr).abs() < 1e-9 && (im - di).abs() < 1e-9,
                    "{size} bin {k}"
                );
            }
        }
        assert!(Fft::new(7).is_err());
        Ok(())
    }
}
