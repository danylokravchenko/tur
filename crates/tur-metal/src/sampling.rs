#![forbid(unsafe_code)]
//! Token sampling: penalties, temperature, top-k/top-p truncation and a counter-based RNG.
//!
//! This host reference and the `penalize_f32`/`sample_f32` kernels implement one algorithm,
//! so a request draws from the same distribution on either path. Weights are
//! `w = exp((logit - max) / temperature)`. A token `j` survives truncation when fewer than
//! `top_k` tokens have a strictly larger weight and their mass is below `top_p` of the
//! total: top-k, then top-p measured on the whole distribution, as Candle's sampler does.
//!
//! Sampling never sorts. Each round inverts the CDF of the weights above a pivot, in token
//! order, with uniform `u(seed, draw, round)`. An accepted token is returned; a rejected one
//! lies outside the truncated set, as does every token not more probable than it, so the
//! pivot rises to its weight and the next round resamples from the strictly heavier rest.
//! This is exact rejection sampling from the renormalized truncated distribution. The
//! heaviest token is always accepted, so the rounds terminate; after [`MAX_ROUNDS`] the
//! host samples the truncated set explicitly.
//!
//! Draws depend only on the request seed and its draw counter, never on batch order.
//! The host inverts in token order; the device in a fixed permutation of it (each thread's
//! strided tokens in turn, for coalesced passes). Both sample the same distribution with
//! the same truncation, but the same uniform can select different tokens on the two paths.

use std::{cmp::Reverse, collections::BinaryHeap};
use tur_core::{Result, TurError, execution::invalid};

/// Rejection rounds a kernel runs before reporting [`SampleStatus::Unconverged`].
pub const MAX_ROUNDS: u32 = 64;

/// Truncation and RNG coordinates of one sampled row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
    /// Positive and finite.
    pub temperature: f32,
    /// 0 disables top-k.
    pub top_k: u32,
    /// In `(0, 1]`; 1 disables top-p.
    pub top_p: f32,
    pub seed: u64,
    /// Index of this sample in the request's stream.
    pub draw: u64,
}

impl SampleParams {
    /// Reject parameters the kernel cannot interpret.
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || self.temperature <= 0.0
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
        {
            return Err(invalid(
                "sampling needs a positive temperature and top-p in (0, 1]",
            ));
        }
        Ok(())
    }

    fn truncates(&self, vocab: usize) -> bool {
        (self.top_k > 0 && (self.top_k as usize) < vocab) || self.top_p < 1.0
    }

    /// Words uploaded per row: temperature and top-p bits, top-k, seed and draw halves.
    pub fn words(&self) -> [u32; 7] {
        [
            self.temperature.to_bits(),
            self.top_k,
            self.top_p.to_bits(),
            self.seed as u32,
            (self.seed >> 32) as u32,
            self.draw as u32,
            (self.draw >> 32) as u32,
        ]
    }
}

/// Repetition (multiplicative on context tokens), presence and frequency (subtractive on
/// sampled tokens) penalties of one row. The identity is `(1, 0, 0)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenaltyParams {
    pub repetition: f32,
    pub presence: f32,
    pub frequency: f32,
}

impl PenaltyParams {
    pub const NONE: Self = Self {
        repetition: 1.0,
        presence: 0.0,
        frequency: 0.0,
    };

    /// Whether any penalty changes logits.
    #[allow(clippy::float_cmp, reason = "exact identity values disable penalties")]
    pub fn active(&self) -> bool {
        self.repetition != 1.0 || self.presence != 0.0 || self.frequency != 0.0
    }

    /// Whether tokens that were never sampled still change, through the repetition penalty.
    #[allow(
        clippy::float_cmp,
        reason = "exact identity value disables the penalty"
    )]
    pub fn uses_context(&self) -> bool {
        self.repetition != 1.0
    }

    /// Reject non-finite penalties and a non-positive repetition factor.
    pub fn validate(&self) -> Result<()> {
        if !self.repetition.is_finite()
            || self.repetition <= 0.0
            || !self.presence.is_finite()
            || !self.frequency.is_finite()
        {
            return Err(invalid(
                "penalties must be finite with a positive repetition",
            ));
        }
        Ok(())
    }

    /// Penalized logit of a context token sampled `count` times, exactly as the kernel
    /// computes it.
    pub fn apply(&self, logit: f32, count: u32) -> f32 {
        let scaled = if logit > 0.0 {
            logit / self.repetition
        } else {
            logit * self.repetition
        };
        let count = count as f32;
        let presence = if count > 0.0 { self.presence } else { 0.0 };
        scaled - self.frequency.mul_add(count, presence)
    }
}

/// One distinct context token of a row and how often the request sampled it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PenaltyEntry {
    pub token: u32,
    pub count: u32,
}

/// Penalize `logits` in place. Entries must be distinct tokens inside the row.
pub fn penalize(
    logits: &mut [f32],
    params: &PenaltyParams,
    entries: &[PenaltyEntry],
) -> Result<()> {
    for entry in entries {
        let logit = logits
            .get_mut(entry.token as usize)
            .ok_or_else(|| invalid("penalized token outside the vocabulary"))?;
        *logit = params.apply(*logit, entry.count);
    }
    Ok(())
}

/// Outcome of a device-sampled row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleStatus {
    /// NaN or +∞ in the row, or no token with a positive weight.
    Invalid,
    Sampled,
    /// Rejection did not accept within [`MAX_ROUNDS`]; resample on the host.
    Unconverged,
}

impl SampleStatus {
    /// Decode the kernel's status word.
    pub fn from_word(word: f32) -> Self {
        match word as u32 {
            1 => Self::Sampled,
            2 => Self::Unconverged,
            _ => Self::Invalid,
        }
    }
}

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Uniform in `[0, 1)` with 24 bits, from `SplitMix64` over the request seed, its draw
/// counter and the rejection round. Identical to `uniform24` in the MSL.
pub fn uniform(seed: u64, draw: u64, round: u32) -> f32 {
    let z = seed
        .wrapping_add(draw.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(u64::from(round).wrapping_mul(0xd1b5_4a32_d192_ed03));
    (mix64(z) >> 40) as f32 * (1.0 / 16_777_216.0)
}

/// `exp(x)` for `x ≤ 0` within about 2 ulp, and exactly 0 below `-87.3` (where the result
/// is no longer a normal F32) or at −∞. Branch-free, so loops over rows vectorize: range
/// reduction by `ln 2` in two parts, then a degree-6 polynomial on `|r| ≤ ln 2 / 2`.
fn exp_nonpositive(x: f32) -> f32 {
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    let clamped = x.clamp(-87.3, 0.0);
    let n = (clamped * std::f32::consts::LOG2_E).round();
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, clamped));
    let p = r.mul_add(1.0 / 720.0, 1.0 / 120.0);
    let p = r.mul_add(p, 1.0 / 24.0);
    let p = r.mul_add(p, 1.0 / 6.0);
    let p = r.mul_add(p, 0.5);
    let p = r.mul_add(p, 1.0);
    let p = r.mul_add(p, 1.0);
    // 2^n for n in -126..=0, built from the exponent bits.
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    if x < -87.3 { 0.0 } else { p * scale }
}

/// Maximum logit; fails on NaN, +∞ or a row without any finite logit. −∞ marks masked
/// tokens.
fn row_max(logits: &[f32]) -> Result<f32> {
    // A fold without early exit or short-circuiting vectorizes; NaN or +∞ shows up in
    // the flag.
    #[allow(
        clippy::needless_bitwise_bool,
        reason = "branch-free so the fold vectorizes"
    )]
    let (max, invalid_seen) =
        logits
            .iter()
            .fold((f32::NEG_INFINITY, false), |(max, bad), &value| {
                (
                    max.max(value),
                    bad | value.is_nan() | (value == f32::INFINITY),
                )
            });
    if invalid_seen {
        return Err(invalid("nonfinite model logits"));
    }
    if max == f32::NEG_INFINITY {
        return Err(TurError::Guidance("every token is masked".into()));
    }
    Ok(max)
}

/// Sample one token from `logits` (penalties and masks already applied).
pub fn sample(logits: &[f32], params: &SampleParams) -> Result<u32> {
    params.validate()?;
    let max = row_max(logits)?;
    let inv = 1.0 / params.temperature;
    let weights: Vec<f32> = logits
        .iter()
        .map(|&l| exp_nonpositive((l - max) * inv))
        .collect();
    let truncates = params.truncates(weights.len());
    let mut low = 0.0f32;
    let mut mass: f32 = weights.iter().filter(|&&w| w > low).sum();
    let total = mass;
    let k = params.top_k as usize;
    if k > 0 && k < weights.len() {
        // The truncated set lies within the k heaviest tokens (and their ties), so an
        // O(vocabulary) selection replaces the rejection passes on the host.
        // Non-negative weights order like their bit patterns.
        let mut heaviest = BinaryHeap::with_capacity(k + 1);
        for &w in &weights {
            if heaviest.len() < k {
                heaviest.push(Reverse(w.to_bits()));
            } else if heaviest
                .peek()
                .is_some_and(|&Reverse(min)| w.to_bits() > min)
            {
                heaviest.pop();
                heaviest.push(Reverse(w.to_bits()));
            }
        }
        let kth = heaviest
            .peek()
            .map_or(0.0, |&Reverse(bits)| f32::from_bits(bits));
        let candidates = (0..weights.len()).filter(|&i| weights[i] >= kth && weights[i] > 0.0);
        return sample_truncated(&weights, candidates.collect(), params, total, 0);
    }
    for round in 0..MAX_ROUNDS {
        let target = uniform(params.seed, params.draw, round) * mass;
        let (token, weight) = locate(&weights, low, target)?;
        if !truncates {
            return to_token(token);
        }
        let (count, above) = weights
            .iter()
            .filter(|&&w| w > weight)
            .fold((0usize, 0f32), |(n, s), &w| (n + 1, s + w));
        if (params.top_k == 0 || count < params.top_k as usize) && above < params.top_p * total {
            return to_token(token);
        }
        low = weight;
        mass = above;
    }
    let candidates = (0..weights.len()).filter(|&i| weights[i] > 0.0).collect();
    sample_truncated(&weights, candidates, params, total, MAX_ROUNDS)
}

/// First token whose cumulative weight above `low` exceeds `target`, or the last such
/// token when rounding leaves the target beyond the sum.
fn locate(weights: &[f32], low: f32, target: f32) -> Result<(usize, f32)> {
    let mut sum = 0f32;
    let mut last = None;
    for (index, &w) in weights.iter().enumerate() {
        if w > low {
            sum += w;
            last = Some((index, w));
            if sum > target {
                return Ok((index, w));
            }
        }
    }
    last.ok_or_else(|| invalid("no token above the sampling pivot"))
}

/// Sample the truncated set explicitly from `candidates`, a superset of it, with the
/// uniform of `round`.
fn sample_truncated(
    weights: &[f32],
    mut order: Vec<usize>,
    params: &SampleParams,
    total: f32,
    round: u32,
) -> Result<u32> {
    order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
    let (mut kept, mut above, mut start) = (Vec::new(), 0f32, 0);
    while start < order.len() {
        // Tokens of equal weight share their strictly heavier count and mass.
        let weight = weights[order[start]];
        let end = start
            + order[start..]
                .iter()
                .take_while(|&&i| weights[i].to_bits() == weight.to_bits())
                .count();
        if (params.top_k != 0 && start >= params.top_k as usize) || above >= params.top_p * total {
            break;
        }
        kept.extend_from_slice(&order[start..end]);
        above = weight.mul_add((end - start) as f32, above);
        start = end;
    }
    kept.sort_unstable();
    let mass: f32 = kept.iter().map(|&i| weights[i]).sum();
    let target = uniform(params.seed, params.draw, round) * mass;
    let mut sum = 0f32;
    for &index in &kept {
        sum += weights[index];
        if sum > target {
            return to_token(index);
        }
    }
    kept.last()
        .map_or_else(|| Err(invalid("empty truncated set")), |&i| to_token(i))
}

fn to_token(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| invalid("token id exceeds u32"))
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn params(temperature: f32, top_k: u32, top_p: f32) -> SampleParams {
        SampleParams {
            temperature,
            top_k,
            top_p,
            seed: 7,
            draw: 0,
        }
    }

    /// Exact truncated probabilities by sorting, as Candle's top-k-then-top-p defines them.
    fn expected(logits: &[f32], p: &SampleParams) -> Vec<f64> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f64> = logits
            .iter()
            .map(|&l| f64::from((l - max) / p.temperature).exp())
            .collect();
        let total: f64 = w.iter().sum();
        let keep: Vec<bool> = (0..w.len())
            .map(|j| {
                let above: Vec<f64> = w.iter().copied().filter(|&x| x > w[j]).collect();
                (p.top_k == 0 || above.len() < p.top_k as usize)
                    && above.iter().sum::<f64>() < f64::from(p.top_p) * total
            })
            .collect();
        let kept: f64 = (0..w.len()).filter(|&i| keep[i]).map(|i| w[i]).sum();
        (0..w.len())
            .map(|i| if keep[i] { w[i] / kept } else { 0.0 })
            .collect()
    }

    /// Pearson chi-square of `draws` samples against `expected` over its support.
    fn chi_square(logits: &[f32], p: SampleParams, draws: u64) -> Result<(f64, usize)> {
        let probabilities = expected(logits, &p);
        let mut counts = vec![0u64; logits.len()];
        for draw in 0..draws {
            let token = sample(logits, &SampleParams { draw, ..p })? as usize;
            assert!(
                probabilities[token] > 0.0,
                "sampled truncated token {token}"
            );
            counts[token] += 1;
        }
        let support = probabilities.iter().filter(|&&q| q > 0.0).count();
        let statistic = probabilities
            .iter()
            .zip(&counts)
            .filter(|(q, _)| **q > 0.0)
            .map(|(q, &n)| {
                let e = q * draws as f64;
                (n as f64 - e).powi(2) / e
            })
            .sum();
        Ok((statistic, support - 1))
    }

    #[test]
    fn test_vectorizable_exponential_matches_the_standard_one() {
        for i in 0..=100_000 {
            let x = -(i as f32) * 0.000_873;
            let (fast, exact) = (exp_nonpositive(x), x.exp());
            assert!(
                (fast - exact).abs() <= exact * 3e-7,
                "{x}: {fast} vs {exact}"
            );
        }
        assert_eq!(exp_nonpositive(0.0), 1.0);
        assert_eq!(exp_nonpositive(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp_nonpositive(-90.0), 0.0);
    }

    #[test]
    fn test_uniform_is_reproducible_and_covers_the_unit_interval() {
        assert_eq!(uniform(1, 2, 3).to_bits(), uniform(1, 2, 3).to_bits());
        assert_ne!(uniform(1, 2, 3).to_bits(), uniform(1, 3, 3).to_bits());
        let samples: Vec<f32> = (0..10_000).map(|d| uniform(42, d, 0)).collect();
        assert!(samples.iter().all(|u| (0.0..1.0).contains(u)));
        let mean = samples.iter().map(|&u| f64::from(u)).sum::<f64>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.01, "{mean}");
    }

    #[test]
    fn test_categorical_draws_match_truncated_distributions() -> Result<()> {
        let logits: Vec<f32> = (0..12)
            .map(|i| (((i * 7) % 12) as f32).mul_add(0.4, -2.0))
            .collect();
        // 99.9th percentiles of chi-square with the support's degrees of freedom.
        let critical = |df: usize| {
            [
                0.0, 10.8, 13.8, 16.3, 18.5, 20.5, 22.5, 24.3, 26.1, 27.9, 29.6, 31.3,
            ][df]
        };
        for p in [
            params(1.0, 0, 1.0),
            params(0.7, 5, 1.0),
            params(1.3, 0, 0.8),
            params(1.0, 6, 0.9),
            params(0.5, 1, 1.0),
        ] {
            let (statistic, df) = chi_square(&logits, p, 40_000)?;
            if df == 0 {
                assert!(statistic.abs() < 1e-9);
                continue;
            }
            assert!(
                statistic < critical(df),
                "{p:?}: χ²={statistic:.1}, df={df}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_ties_masks_and_invalid_rows() -> Result<()> {
        // Tied weights share their heavier count: top-1 keeps both maxima.
        let tied = [3.0, 1.0, 3.0, -1.0];
        let tokens: Vec<u32> = (0..200)
            .map(|draw| {
                sample(
                    &tied,
                    &SampleParams {
                        draw,
                        ..params(1.0, 1, 1.0)
                    },
                )
            })
            .collect::<Result<_>>()?;
        assert!(tokens.iter().all(|&t| t == 0 || t == 2));
        assert!(tokens.contains(&0) && tokens.contains(&2));
        let masked = [f32::NEG_INFINITY, 0.5, f32::NEG_INFINITY];
        assert_eq!(sample(&masked, &params(1.0, 0, 1.0))?, 1);
        assert!(sample(&[f32::NAN, 1.0], &params(1.0, 0, 1.0)).is_err());
        assert!(sample(&[f32::INFINITY, 1.0], &params(1.0, 0, 1.0)).is_err());
        assert!(sample(&[f32::NEG_INFINITY; 3], &params(1.0, 0, 1.0)).is_err());
        assert!(sample(&[1.0], &params(0.0, 0, 1.0)).is_err());
        assert!(sample(&[1.0], &params(1.0, 0, 0.0)).is_err());
        Ok(())
    }

    #[test]
    fn test_explicit_fallback_samples_the_same_truncated_set() -> Result<()> {
        let logits: Vec<f32> = (0..9).map(|i| (i as f32 * 1.3).sin() * 3.0).collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        let p = params(1.0, 3, 0.95);
        let truth = expected(&logits, &p);
        for draw in 0..500 {
            let all = (0..weights.len()).collect();
            let token =
                sample_truncated(&weights, all, &SampleParams { draw, ..p }, total, 0)? as usize;
            assert!(truth[token] > 0.0);
        }
        Ok(())
    }

    #[test]
    fn test_penalties_follow_repetition_presence_and_frequency_rules() -> Result<()> {
        let p = PenaltyParams {
            repetition: 2.0,
            presence: 0.5,
            frequency: 0.25,
        };
        let mut logits = [4.0, -4.0, 1.0, 3.0];
        penalize(
            &mut logits,
            &p,
            &[
                PenaltyEntry { token: 0, count: 0 },
                PenaltyEntry { token: 1, count: 2 },
                PenaltyEntry { token: 2, count: 1 },
            ],
        )?;
        assert_eq!(logits, [2.0, -9.0, -0.25, 3.0]);
        assert!(!PenaltyParams::NONE.active());
        assert!(p.active() && p.uses_context());
        assert!(
            PenaltyParams {
                repetition: 0.0,
                ..PenaltyParams::NONE
            }
            .validate()
            .is_err()
        );
        assert!(penalize(&mut logits, &p, &[PenaltyEntry { token: 9, count: 0 }]).is_err());
        Ok(())
    }
}
