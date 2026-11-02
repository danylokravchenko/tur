//! Antialiased bicubic resampling of 8-bit images, as `torchvision` computes it on a tensor:
//! Pillow's scheme of fixed-point filters along rows, then columns, with each pass rounded
//! to 8 bits. Tur's output equals it byte for byte.

/// Cubic convolution with `a = -0.5`, the filter of `antialias=True`.
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        (A + 2.0).mul_add(x, -(A + 3.0)).mul_add(x * x, 1.0)
    } else if x < 2.0 {
        (x - 5.0).mul_add(x, 8.0).mul_add(x, -4.0) * A
    } else {
        0.0
    }
}

/// One output sample's first input sample and the range of its weights.
struct Tap {
    first: usize,
    weights: std::ops::Range<usize>,
}

/// An axis's filters in fixed point with `precision` fractional bits.
struct Filter {
    taps: Vec<Tap>,
    weights: Vec<i32>,
    precision: u32,
}

impl Filter {
    /// Downscaling widens the filter over the input samples an output covers, which is what
    /// antialiases. The precision is the largest that keeps every weight in 16 bits.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "sides are far below 2^52, and truncation toward zero is the reference's rounding"
    )]
    fn new(input: usize, output: usize) -> Self {
        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        let support = 2.0 * filter_scale;
        let mut taps = Vec::with_capacity(output);
        let mut raw = Vec::with_capacity(output * (support as usize * 2 + 1));
        for index in 0..output {
            let center = scale * (index as f64 + 0.5);
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let start = raw.len();
            raw.extend((first..end).map(|at| cubic((at as f64 - center + 0.5) / filter_scale)));
            let total: f64 = raw[start..].iter().sum();
            for weight in &mut raw[start..] {
                *weight /= total;
            }
            taps.push(Tap {
                first,
                weights: start..raw.len(),
            });
        }
        let largest = raw.iter().copied().fold(0.0, f64::max);
        let mut precision = 0;
        while precision < 22 && largest.mul_add(f64::from(1u32 << (precision + 1)), 0.5) < 32_768.0
        {
            precision += 1;
        }
        let unit = f64::from(1u32 << precision);
        let weights = raw
            .iter()
            .map(|&w| {
                if w < 0.0 {
                    w.mul_add(unit, -0.5) as i32
                } else {
                    w.mul_add(unit, 0.5) as i32
                }
            })
            .collect();
        Self {
            taps,
            weights,
            precision,
        }
    }

    fn weights(&self, tap: &Tap) -> &[i32] {
        &self.weights[tap.weights.clone()]
    }

    /// A summed sample back in 8 bits.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0..=255 first"
    )]
    fn settle(&self, sum: i32) -> u8 {
        ((sum + (1 << (self.precision - 1))) >> self.precision).clamp(0, 255) as u8
    }
}

/// `pixels` of interleaved RGB at `from` = (height, width), resampled to `to`.
pub(crate) fn bicubic(pixels: &[u8], from: (usize, usize), to: (usize, usize)) -> Vec<u8> {
    let (in_height, in_width) = from;
    let (out_height, out_width) = to;
    let across = Filter::new(in_width, out_width);
    let mut rows = vec![0u8; in_height * out_width * 3];
    for (source, target) in pixels
        .chunks_exact(in_width * 3)
        .zip(rows.chunks_exact_mut(out_width * 3))
    {
        for (tap, out) in across.taps.iter().zip(target.as_chunks_mut::<3>().0) {
            let samples = source[tap.first * 3..].as_chunks::<3>().0;
            let mut sum = [0i32; 3];
            for (rgb, &weight) in samples.iter().zip(across.weights(tap)) {
                for (total, &sample) in sum.iter_mut().zip(rgb) {
                    *total += i32::from(sample) * weight;
                }
            }
            for (value, &total) in out.iter_mut().zip(&sum) {
                *value = across.settle(total);
            }
        }
    }
    let down = Filter::new(in_height, out_height);
    let stride = out_width * 3;
    let mut sums = vec![0i32; stride];
    let mut out = vec![0u8; out_height * stride];
    for (tap, target) in down.taps.iter().zip(out.chunks_exact_mut(stride)) {
        sums.fill(0);
        let sources = rows[tap.first * stride..].chunks_exact(stride);
        for (source, &weight) in sources.zip(down.weights(tap)) {
            for (total, &sample) in sums.iter_mut().zip(source) {
                *total += i32::from(sample) * weight;
            }
        }
        for (value, &total) in target.iter_mut().zip(&sums) {
            *value = down.settle(total);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filters_stay_inside_the_input_and_fit_16_bits() {
        for (input, output) in [(7, 64), (64, 64), (800, 96), (3, 2), (1, 5), (4000, 32)] {
            let filter = Filter::new(input, output);
            assert_eq!(filter.taps.len(), output);
            let unit = 1i32 << filter.precision;
            for tap in &filter.taps {
                let weights = filter.weights(tap);
                assert!(tap.first + weights.len() <= input);
                assert!(weights.iter().all(|w| i16::try_from(*w).is_ok()));
                // Rounding each weight moves the sum by at most half a step per weight.
                let error = (weights.iter().sum::<i32>() - unit).abs();
                assert!(error as usize <= weights.len());
            }
        }
    }

    #[test]
    fn test_flat_images_stay_flat_at_any_size() {
        let pixels = [200u8, 100, 50].repeat(9 * 13);
        let out = bicubic(&pixels, (9, 13), (32, 4));
        assert_eq!(out.len(), 32 * 4 * 3);
        assert!(
            out.as_chunks::<3>()
                .0
                .iter()
                .all(|rgb| rgb == &[200, 100, 50])
        );
    }
}
