#![forbid(unsafe_code)]
//! Draft proposals and draft-length control for speculative decoding.
//!
//! Proposals come from a request's own history (prompt lookup): the longest suffix n-gram
//! that occurred earlier proposes the tokens that followed its latest occurrence. The
//! engine verifies every drafted position in one pass and accepts drafts while they equal
//! the target's own samples, which keeps the target distribution exactly for these
//! deterministic proposals (see [`crate::hybrid`]).
//!
//! Drafting is worth it only while the expected accepted tokens outpace the extra
//! verification cost. Each request keeps an online estimate of its per-draft acceptance,
//! and [`DraftCosts`] fits measured device time against drafted tokens separately for
//! each count of decoding sequences. [`plan`] then drafts the most likely tokens first
//! and stops where the expected tokens per second of the pass would fall, so batching
//! that makes verification slower disables speculation by itself.

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    ops::RangeInclusive,
};
use tur_core::{Result, execution::invalid};
use tur_models::models::DraftHead;

/// Prune periodically to retain at most two windows of lookup history.
const INDEX_WINDOW: usize = 8192;

/// Token IDs the draft head drafts from by default. Byte-level BPE numbers tokens in merge
/// order, so these are the most frequent: on Qwen3.5-0.8B they kept the head's acceptance
/// on prose and code at a seventh of its projection cost; 16K lost prose drafts and 64K
/// added cost.
pub const MTP_VOCAB: usize = 32_768;

/// Speculative decoding limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeculationConfig {
    /// Most drafted tokens per sequence and pass. Zero disables speculation.
    pub max_draft: usize,
    /// Most drafted tokens per pass over all sequences.
    pub max_pass_drafts: usize,
    /// Shortest suffix n-gram matched against the history.
    pub min_ngram: usize,
    /// Longest suffix n-gram matched against the history; longer matches are tried first.
    pub max_ngram: usize,
    /// Also draft one token from the checkpoint's native multi-token prediction head when
    /// prompt lookup has no proposal or a less accepted one. Drafts are computed for
    /// unconstrained requests in passes of at most [`MTP_MAX_DECODES`] decodes, and each
    /// is weighed by its calibrated confidence ([`Calibration`]). Needs prefix reuse
    /// disabled.
    pub mtp: Option<DraftHead>,
}

impl SpeculationConfig {
    /// Disable proposal generation and its additional memory reservation.
    pub const DISABLED: Self = Self {
        max_draft: 0,
        max_pass_drafts: 0,
        min_ngram: 1,
        max_ngram: 1,
        mtp: None,
    };

    /// Up to `max_draft` drafts per sequence and four sequences' worth per pass, from
    /// suffix matches of one to four tokens.
    pub const fn new(max_draft: usize) -> Self {
        Self {
            max_draft,
            max_pass_drafts: max_draft.saturating_mul(4),
            min_ngram: 1,
            max_ngram: 4,
            mtp: None,
        }
    }

    /// [`Self::new`] plus one-token drafts from the native multi-token prediction head,
    /// over the leading [`MTP_VOCAB`] token IDs.
    pub fn with_mtp(max_draft: usize) -> Self {
        Self {
            mtp: Some(DraftHead {
                vocab: Some(MTP_VOCAB),
                model: None,
            }),
            ..Self::new(max_draft)
        }
    }

    /// Whether the multi-token prediction head drafts.
    pub fn mtp_enabled(&self) -> bool {
        self.enabled() && self.mtp.is_some()
    }

    /// Whether proposals and their memory reservation are enabled.
    pub fn enabled(&self) -> bool {
        self.max_draft > 0 && self.max_pass_drafts > 0
    }

    /// Reject invalid suffix lengths before admitting requests.
    pub fn validate(&self) -> Result<()> {
        if self.enabled()
            && (self.min_ngram == 0 || self.min_ngram > self.max_ngram || self.max_ngram > 16)
        {
            return Err(invalid(
                "speculation n-gram lengths must satisfy 0 < min ≤ max ≤ 16",
            ));
        }
        Ok(())
    }

    fn lengths(&self) -> RangeInclusive<usize> {
        self.min_ngram..=self.max_ngram
    }

    pub(crate) fn index_bytes(&self, sequences: usize, context: usize) -> Result<usize> {
        if !self.enabled() {
            return Ok(0);
        }
        // Include hash-table control bytes and capacity rounding, not just keys/values.
        sequences
            .checked_mul(context.min(2 * INDEX_WINDOW))
            .and_then(|n| n.checked_mul(self.max_ngram))
            .and_then(|n| n.checked_mul(64))
            .ok_or_else(|| invalid("draft index size overflow"))
    }
}

/// Keys are already n-gram hashes; hashing them again would only cost time.
#[derive(Clone, Copy, Debug, Default)]
struct Prehashed(u64);

impl Hasher for Prehashed {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(8)) ^ u64::from(byte);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

/// Latest earlier occurrence of every short n-gram of one request's tokens.
#[derive(Clone, Debug, Default)]
pub struct NgramIndex {
    /// N-gram hash → end (exclusive) of its latest occurrence that has a follower.
    ends: HashMap<u64, usize, BuildHasherDefault<Prehashed>>,
    /// Occurrences ending before this position are indexed.
    next_end: usize,
}

fn ngram_hash(tokens: &[u32]) -> u64 {
    tokens.iter().fold(tokens.len() as u64, |hash, &token| {
        (hash.rotate_left(5) ^ u64::from(token)).wrapping_mul(0x517c_c1b7_2722_0a95)
    })
}

impl NgramIndex {
    /// Index the n-grams of `tokens` that end before its last token, so every indexed
    /// occurrence has a known follower. `tokens` only ever grows.
    fn update(&mut self, tokens: &[u32], lengths: RangeInclusive<usize>) {
        let start = tokens.len().saturating_sub(INDEX_WINDOW);
        if self.next_end / INDEX_WINDOW != tokens.len() / INDEX_WINDOW {
            self.ends.retain(|_, end| *end >= start);
        }
        for end in self.next_end.max(start).max(1)..tokens.len() {
            for n in lengths.clone().filter(|&n| n <= end) {
                self.ends.insert(ngram_hash(&tokens[end - n..end]), end);
            }
        }
        self.next_end = self.next_end.max(tokens.len());
    }

    /// Up to `limit` tokens that followed the latest earlier occurrence of the longest
    /// matching suffix of `tokens`; empty without a match or for disabled/invalid limits.
    /// A continuation that reaches the end of the history repeats with the period between
    /// the two occurrences, so a short cycle still fills the whole limit.
    pub fn propose(
        &mut self,
        tokens: &[u32],
        config: &SpeculationConfig,
        limit: usize,
    ) -> Vec<u32> {
        if !config.enabled() || config.validate().is_err() {
            return Vec::new();
        }
        self.update(tokens, config.lengths());
        if limit == 0 {
            return Vec::new();
        }
        for n in config.lengths().rev() {
            let Some(suffix) = tokens
                .len()
                .checked_sub(n)
                .and_then(|start| tokens.get(start..))
            else {
                continue;
            };
            let Some(&end) = self.ends.get(&ngram_hash(suffix)) else {
                continue;
            };
            // Hash hits are verified; only occurrences with a follower are indexed.
            if end >= n && tokens.get(end - n..end) == Some(suffix) {
                let Some(cycle) = tokens.get(end..) else {
                    continue;
                };
                return cycle.iter().copied().cycle().take(limit).collect();
            }
        }
        Vec::new()
    }
}

/// Weight of one observation in an acceptance estimate.
const ACCEPTANCE_SMOOTHING: f64 = 0.2;
/// How far an idle estimate moves back towards the pipeline-wide rate per pass without
/// drafts, so a request whose early proposals failed is eventually retried.
const ACCEPTANCE_RECOVERY: f64 = 0.05;

/// Online estimate of the probability that one more draft is accepted, given that the
/// drafts before it were.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Acceptance(f64);

impl Default for Acceptance {
    fn default() -> Self {
        Self(0.5)
    }
}

impl Acceptance {
    /// Estimated conditional probability of accepting the next draft.
    pub fn rate(self) -> f64 {
        self.0
    }

    /// Fold in one verification: `accepted` successes, then a rejection unless every one
    /// of `drafted` drafts was accepted.
    pub fn observe(&mut self, drafted: usize, accepted: usize) {
        let mut fold =
            |outcome: f64| self.0 = ACCEPTANCE_SMOOTHING.mul_add(outcome - self.0, self.0);
        for _ in 0..accepted.min(drafted) {
            fold(1.0);
        }
        if accepted < drafted {
            fold(0.0);
        }
    }

    /// Fold in a draft expected to be accepted with probability `rate`.
    pub fn expect(&mut self, rate: f64) {
        self.0 = ACCEPTANCE_SMOOTHING.mul_add(rate.clamp(0.0, 1.0) - self.0, self.0);
    }

    /// Move towards `prior` after a pass in which a proposal existed but was not drafted.
    pub fn recover(&mut self, prior: Self) {
        self.0 = ACCEPTANCE_RECOVERY.mul_add(prior.0 - self.0, self.0);
    }
}

/// Weight kept by older samples of a cost fit per new sample (a window of about twenty).
const COST_DECAY: f64 = 0.95;
/// Variance of drafted tokens below which a fit cannot separate base and draft costs.
/// Passes alternating zero and one draft (single-token proposals) reach 0.25.
const MIN_DRAFT_VARIANCE: f64 = 0.1;
/// Conditional acceptance below which a multi-token prediction draft cannot repay the
/// head's LM-head projection: its logits are then skipped (its keys and values are still
/// written) and the estimate recovers towards the pipeline-wide rate.
pub const MTP_MIN_RATE: f64 = 0.3;
/// Most decoding sequences of a pass for which draft-head drafts are computed. On the M1
/// with Qwen3.5-0.8B they paid at one to four sequences and did not at eight, where a
/// verified draft costs about three quarters of a decode token's share of the pass even
/// with a reduced draft vocabulary.
pub const MTP_MAX_DECODES: usize = 4;
/// Decode counts above this share one fit.
const COST_BUCKETS: usize = 64;

/// Exponentially weighted least squares of `seconds = base + per_draft × drafts`.
#[derive(Clone, Copy, Debug, Default)]
struct CostFit {
    weight: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
    /// Per-draft cost of the latest identified fit. When recent passes stop varying their
    /// drafts (planning chose none), the base still tracks recent passes and this slope
    /// keeps the line identified, so no exploratory full-draft pass is forced.
    slope: Option<f64>,
}

impl CostFit {
    fn record(&mut self, drafts: f64, seconds: f64) {
        for sum in [
            &mut self.weight,
            &mut self.x,
            &mut self.y,
            &mut self.xx,
            &mut self.xy,
        ] {
            *sum *= COST_DECAY;
        }
        self.weight += 1.0;
        self.x += drafts;
        self.y += seconds;
        self.xx = drafts.mul_add(drafts, self.xx);
        self.xy = drafts.mul_add(seconds, self.xy);
        if let Some((_, slope)) = self.line() {
            self.slope = Some(slope);
        }
    }

    fn mean_drafts(&self) -> f64 {
        if self.weight > 0.0 {
            self.x / self.weight
        } else {
            0.0
        }
    }

    /// `(base, per_draft)` seconds once both are identified. A negative per-draft slope
    /// from noise counts as free drafts only up to a microsecond.
    fn line(&self) -> Option<(f64, f64)> {
        if self.weight < 2.0 {
            return None;
        }
        let mean_x = self.x / self.weight;
        let mean_y = self.y / self.weight;
        let variance = mean_x.mul_add(-mean_x, self.xx / self.weight);
        let slope = if variance < MIN_DRAFT_VARIANCE {
            self.slope?
        } else {
            (mean_x.mul_add(-mean_y, self.xy / self.weight) / variance).max(1e-6)
        };
        let base = slope.mul_add(-mean_x, mean_y);
        (base > 0.0).then_some((base, slope))
    }
}

/// Measured device time of decode-only passes against their drafted tokens, fitted
/// separately for each count of decoding sequences.
#[derive(Clone, Debug, Default)]
pub struct DraftCosts {
    fits: Vec<CostFit>,
}

impl DraftCosts {
    fn bucket(decodes: usize) -> usize {
        decodes.min(COST_BUCKETS)
    }

    /// Record a completed decode-only pass of `decodes` sequences and `drafts` drafted
    /// tokens. Non-finite or negative samples are ignored.
    pub fn record(&mut self, decodes: usize, drafts: usize, seconds: f64) {
        if decodes == 0 || !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        let bucket = Self::bucket(decodes);
        if self.fits.len() <= bucket {
            self.fits.resize(bucket + 1, CostFit::default());
        }
        self.fits[bucket].record(drafts as f64, seconds);
    }

    /// `(base, per_draft)` seconds for passes of `decodes` sequences, once measured.
    pub fn line(&self, decodes: usize) -> Option<(f64, f64)> {
        self.fits.get(Self::bucket(decodes))?.line()
    }

    /// Whether a never-identified fit should next sample a pass without drafts: it
    /// alternates with drafted passes until base and per-draft costs first separate.
    fn explore_without(&self, decodes: usize) -> bool {
        self.fits
            .get(Self::bucket(decodes))
            .is_some_and(|fit| fit.mean_drafts() > 0.5)
    }
}

/// One decoding sequence's proposal for this pass.
#[derive(Clone, Copy, Debug)]
pub struct Proposal {
    /// Proposed tokens, already capped by every per-sequence limit.
    pub tokens: usize,
    pub acceptance: Acceptance,
}

/// Drafts per proposal for a decode-only pass of `decodes` sequences, at most `limit` in
/// total. Depth `i` of a proposal is expected to add `rate^i` tokens; depths are taken
/// most likely first while the expected tokens per second of the pass rise. Until the
/// costs are first identified, passes alternate between drafting everything and nothing;
/// afterwards the last identified per-draft cost stands in for missing variation.
///
/// `prefill` is the estimated extra time of a mixed pass's prefill tokens: it joins the
/// base cost, and a mixed pass never explores unidentified costs.
pub fn plan(
    proposals: &[Proposal],
    decodes: usize,
    costs: &DraftCosts,
    limit: usize,
    prefill: Option<f64>,
) -> Vec<usize> {
    let mut depths: Vec<(f64, usize, usize)> = proposals
        .iter()
        .enumerate()
        .flat_map(|(index, p)| {
            let rate = p.acceptance.rate().clamp(0.0, 1.0);
            (1..=p.tokens).map(move |depth| (rate.powi(depth as i32), depth, index))
        })
        .collect();
    // Most likely first; a shallower depth of the same proposal never sorts after a deeper.
    depths.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    depths.truncate(limit);
    let line = costs
        .line(decodes)
        .map(|(base, per_draft)| (base + prefill.unwrap_or(0.0), per_draft));
    let take = match line {
        Some((base, per_draft)) => {
            let (mut best, mut best_rate, mut gain) = (0, decodes as f64 / base, 0.0);
            for (taken, (expected, _, _)) in depths.iter().enumerate() {
                gain += expected;
                let rate = (decodes as f64 + gain) / per_draft.mul_add((taken + 1) as f64, base);
                if rate > best_rate {
                    (best, best_rate) = (taken + 1, rate);
                }
            }
            best
        }
        None if prefill.is_some() || costs.explore_without(decodes) => 0,
        None => depths.len(),
    };
    let mut counts = vec![0; proposals.len()];
    for &(_, _, index) in &depths[..take] {
        counts[index] += 1;
    }
    counts
}

/// Confidence bins of a [`Calibration`].
const CONFIDENCE_BINS: usize = 10;
/// Observations a bin's prior, its own confidence, counts as.
const CONFIDENCE_PRIOR: f64 = 4.0;
/// Weight kept by a bin's older observations per new one (a window of about 200).
const CONFIDENCE_DECAY: f64 = 0.995;

/// Observed acceptance of draft-head drafts by the head's own probability of its draft,
/// so a draft is verified on its own expected acceptance rather than its request's
/// history. Each bin starts at its confidence and learns from verified drafts.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Decayed `(accepted, drafted)` per confidence bin.
    bins: [(f64, f64); CONFIDENCE_BINS],
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            bins: [(0.0, 0.0); CONFIDENCE_BINS],
        }
    }
}

impl Calibration {
    fn bin(confidence: f32) -> usize {
        let scaled = f64::from(confidence).clamp(0.0, 1.0) * CONFIDENCE_BINS as f64;
        (scaled as usize).min(CONFIDENCE_BINS - 1)
    }

    /// Expected acceptance of a draft the head gave probability `confidence`.
    pub fn acceptance(&self, confidence: f32) -> Acceptance {
        let bin = Self::bin(confidence);
        let (accepted, drafted) = self.bins[bin];
        let prior = (bin as f64 + 0.5) / CONFIDENCE_BINS as f64;
        Acceptance(CONFIDENCE_PRIOR.mul_add(prior, accepted) / (drafted + CONFIDENCE_PRIOR))
    }

    /// Record whether a draft of `confidence` was accepted.
    pub fn observe(&mut self, confidence: f32, accepted: bool) {
        let (hits, total) = &mut self.bins[Self::bin(confidence)];
        *hits = CONFIDENCE_DECAY.mul_add(*hits, f64::from(u8::from(accepted)));
        *total = CONFIDENCE_DECAY.mul_add(*total, 1.0);
    }
}

/// Weight of one observation in a draft-head cost estimate.
const HEAD_SMOOTHING: f64 = 0.2;
/// Draft-head passes skipped as unprofitable before one computes drafts anyway, so the
/// acceptance and cost estimates that decided it stay current. The interval doubles while
/// exploring keeps finding no gain, up to the maximum.
pub const HEAD_EXPLORE_PASSES: usize = 16;
pub const HEAD_EXPLORE_MAX_PASSES: usize = 256;
/// Predicted throughput gain head drafts must exceed. Within it, the draft cost fit (often
/// learned from multi-token lookup drafts) cannot tell a gain from a loss.
const HEAD_MARGIN: f64 = 1.05;

/// Measured device seconds of draft-head passes that computed `n` drafts, by `n`. The
/// head's LM-head projection is paid before any draft is verified, so its cost decides
/// whether head drafts can raise throughput at all.
#[derive(Clone, Debug, Default)]
pub struct HeadCosts {
    seconds: Vec<Option<f64>>,
}

impl HeadCosts {
    /// Record a decode-sized head pass (one or a few positions per sequence) that computed
    /// `drafts` drafts in `seconds`. Passes that also cover prefill positions cost more
    /// and are not recorded.
    pub fn record(&mut self, drafts: usize, seconds: f64) {
        if drafts == 0 || !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        let bucket = drafts.min(COST_BUCKETS);
        if self.seconds.len() <= bucket {
            self.seconds.resize(bucket + 1, None);
        }
        let cell = &mut self.seconds[bucket];
        *cell = Some(cell.map_or(seconds, |mean| HEAD_SMOOTHING.mul_add(seconds - mean, mean)));
    }

    /// Whether computing drafts for sequences with acceptance `rates` raises the expected
    /// tokens per second of a pass of `decodes` decoding sequences by more than the margin,
    /// given its drafted cost line. Unmeasured costs are worth measuring.
    pub fn worthwhile(&self, rates: &[f64], decodes: usize, costs: &DraftCosts) -> bool {
        let n = rates.len();
        let head = self.seconds.get(n.min(COST_BUCKETS)).copied().flatten();
        let (Some(head), Some((base, per_draft))) = (head, costs.line(decodes)) else {
            return n > 0;
        };
        let gain: f64 = rates.iter().map(|r| r.clamp(0.0, 1.0)).sum();
        let decodes = decodes as f64;
        (decodes + gain) / per_draft.mul_add(n as f64, base + head) > HEAD_MARGIN * decodes / base
    }
}

/// Where a sequence's drafts came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftSource {
    /// The request's own history (prompt lookup).
    Lookup,
    /// The checkpoint's native multi-token prediction head.
    Mtp,
}

/// Cumulative speculation counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeculationStats {
    /// Passes that verified at least one draft.
    pub passes: u64,
    /// Sequence verifications with at least one draft.
    pub sequences: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Of `drafted`/`accepted`, those from the multi-token prediction head.
    pub mtp_drafted: u64,
    pub mtp_accepted: u64,
    /// Draft-head passes run between ordinary passes, and their device seconds.
    pub mtp_passes: u64,
    pub mtp_seconds: f64,
    /// Prompt-lookup drafts verified and accepted at each depth (the last counts deeper
    /// ones too), for per-position acceptance.
    pub depth_drafted: [u64; STAT_DEPTHS],
    pub depth_accepted: [u64; STAT_DEPTHS],
    /// Draft-head drafts verified and accepted per confidence decile.
    pub confidence_drafted: [u64; CONFIDENCE_BINS],
    pub confidence_accepted: [u64; CONFIDENCE_BINS],
}

/// Depths [`SpeculationStats`] distinguishes.
pub const STAT_DEPTHS: usize = 8;

impl SpeculationStats {
    /// Count a prompt-lookup verification of `drafted` drafts, `accepted` of them accepted.
    pub fn observe_depths(&mut self, drafted: usize, accepted: usize) {
        for depth in 0..drafted {
            let bin = depth.min(STAT_DEPTHS - 1);
            self.depth_drafted[bin] += 1;
            self.depth_accepted[bin] += u64::from(depth < accepted);
        }
    }

    /// Count a verified draft-head draft of `confidence`.
    pub fn observe_confidence(&mut self, confidence: f32, accepted: bool) {
        let bin = Calibration::bin(confidence);
        self.confidence_drafted[bin] += 1;
        self.confidence_accepted[bin] += u64::from(accepted);
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn propose(tokens: &[u32], limit: usize) -> Vec<u32> {
        let mut index = NgramIndex::default();
        index.propose(tokens, &SpeculationConfig::new(8), limit)
    }

    #[test]
    fn test_longest_latest_suffix_match_proposes_its_continuation() {
        // "1 2 3" occurred twice; the latest occurrence continues with 9 8.
        let tokens = [1, 2, 3, 4, 5, 1, 2, 3, 9, 8, 7, 1, 2, 3];
        assert_eq!(propose(&tokens, 2), [9, 8]);
        // A longer suffix outranks a later one-token match: "5 6" → 7, not "6" → 1.
        assert_eq!(propose(&[5, 6, 7, 6, 1, 5, 6], 3), [7, 6, 1]);
        // Proposals stop at the end of the history and need a match.
        // A continuation reaching the end repeats with its period.
        assert_eq!(propose(&[4, 4], 5), [4; 5]);
        assert_eq!(propose(&[1, 2, 3, 1, 2, 3], 5), [1, 2, 3, 1, 2]);
        assert_eq!(propose(&[1, 2, 3], 4), [] as [u32; 0]);
        assert_eq!(propose(&[1, 2, 1], 0), [] as [u32; 0]);
        assert_eq!(propose(&[], 2), [] as [u32; 0]);
    }

    #[test]
    fn test_incremental_index_matches_rebuilt_index() {
        let tokens: Vec<u32> = (0..200u32).map(|i| (i * 7 + i / 13) % 11).collect();
        let config = SpeculationConfig::new(4);
        let mut incremental = NgramIndex::default();
        for len in 0..=tokens.len() {
            let grown = incremental.propose(&tokens[..len], &config, 4);
            let mut rebuilt = NgramIndex::default();
            assert_eq!(grown, rebuilt.propose(&tokens[..len], &config, 4), "{len}");
        }
    }

    #[test]
    fn test_acceptance_rises_with_accepted_drafts_and_recovers_towards_the_prior() {
        let mut acceptance = Acceptance::default();
        acceptance.observe(4, 4);
        assert!(acceptance.rate() > 0.75);
        let high = acceptance.rate();
        acceptance.observe(4, 0);
        assert!(acceptance.rate() < high);
        let mut low = Acceptance(0.0);
        low.recover(Acceptance(0.8));
        assert!((low.rate() - 0.04).abs() < 1e-12);
        let mut expected = Acceptance(0.5);
        expected.expect(1.0);
        assert!((expected.rate() - 0.6).abs() < 1e-12);
    }

    fn costs(decodes: usize, base: f64, per_draft: f64) -> DraftCosts {
        let mut costs = DraftCosts::default();
        for drafts in [0, 4, 0, 2, 8, 0, 1] {
            costs.record(decodes, drafts, per_draft.mul_add(drafts as f64, base));
        }
        costs
    }

    #[test]
    fn test_cost_fit_separates_base_and_draft_costs() -> Result<()> {
        let costs = costs(3, 0.020, 0.001);
        let (base, per_draft) = costs.line(3).ok_or_else(|| invalid("unfitted"))?;
        assert!((base - 0.020).abs() < 1e-9 && (per_draft - 0.001).abs() < 1e-9);
        assert!(costs.line(2).is_none());
        // Without varied drafts the costs are not separable.
        let mut flat = DraftCosts::default();
        for _ in 0..10 {
            flat.record(1, 0, 0.02);
        }
        assert!(flat.line(1).is_none());
        Ok(())
    }

    #[test]
    fn test_plan_drafts_while_expected_throughput_rises() {
        let certain = Proposal {
            tokens: 4,
            acceptance: Acceptance(1.0),
        };
        let unlikely = Proposal {
            tokens: 4,
            acceptance: Acceptance(0.05),
        };
        // Cheap drafts: every certain depth pays; 5% depths do not beside a 20 ms pass.
        let cheap = costs(2, 0.020, 0.001);
        assert_eq!(plan(&[certain, unlikely], 2, &cheap, 16, None), [4, 0]);
        // The pass limit keeps the most likely depths.
        assert_eq!(plan(&[unlikely, certain], 2, &cheap, 3, None), [0, 3]);
        // Drafts that cost as much as a sequence's share of the pass are not worth 50%.
        let even = Proposal {
            tokens: 4,
            acceptance: Acceptance(0.5),
        };
        let dear = costs(2, 0.020, 0.010);
        assert_eq!(plan(&[even, even], 2, &dear, 16, None), [0, 0]);
        assert_eq!(plan(&[even, even], 2, &cheap, 16, None), [2, 2]);
        // A mixed pass's prefill time lengthens the base, so drafts become relatively
        // cheaper; without identified costs a mixed pass never drafts.
        assert_eq!(plan(&[even, even], 2, &dear, 16, Some(0.200)), [2, 2]);
        assert_eq!(
            plan(&[certain], 2, &DraftCosts::default(), 16, Some(0.1)),
            [0]
        );
    }

    #[test]
    fn test_identified_costs_survive_passes_without_drafts() -> Result<()> {
        let mut costs = costs(2, 0.020, 0.001);
        let (_, slope) = costs.line(2).ok_or_else(|| invalid("unfitted"))?;
        // Planning chose no drafts for a long time: the drafted samples decay away, but the
        // last slope keeps the line identified while its base follows recent passes.
        for _ in 0..500 {
            costs.record(2, 0, 0.020);
        }
        let (base, kept) = costs.line(2).ok_or_else(|| invalid("forgot the fit"))?;
        assert!((kept - slope).abs() < 1e-9 && (base - 0.020).abs() < 1e-9);
        assert!(!costs.explore_without(2));
        let certain = Proposal {
            tokens: 4,
            acceptance: Acceptance(1.0),
        };
        assert_eq!(plan(&[certain], 2, &costs, 16, None), [4]);
        Ok(())
    }

    #[test]
    fn test_unmeasured_costs_alternate_between_drafting_and_not() {
        let proposal = Proposal {
            tokens: 3,
            acceptance: Acceptance(0.5),
        };
        let mut costs = DraftCosts::default();
        assert_eq!(plan(&[proposal], 1, &costs, 8, None), [3]);
        costs.record(1, 3, 0.02);
        assert_eq!(plan(&[proposal], 1, &costs, 8, None), [0]);
        costs.record(1, 0, 0.018);
        costs.record(1, 0, 0.018);
        assert!(costs.line(1).is_some());
    }

    #[test]
    fn test_head_drafts_only_when_their_cost_repays_them() {
        let costs = costs(4, 0.020, 0.0005);
        let mut head = HeadCosts::default();
        // Unmeasured: measure it.
        assert!(head.worthwhile(&[0.7; 4], 4, &costs));
        // A 6 ms head beside a 20 ms pass: 70% acceptance across four sequences pays.
        head.record(4, 0.006);
        assert!(head.worthwhile(&[0.7; 4], 4, &costs));
        // The same head for a single unlikely draft does not.
        head.record(1, 0.006);
        assert!(!head.worthwhile(&[0.2], 4, &costs));
        assert!(!head.worthwhile(&[], 4, &costs));
    }

    #[test]
    fn test_calibration_starts_at_confidence_and_learns_observed_acceptance() {
        let mut calibration = Calibration::default();
        assert!((calibration.acceptance(0.93).rate() - 0.95).abs() < 1e-12);
        assert!((calibration.acceptance(0.0).rate() - 0.05).abs() < 1e-12);
        // An overconfident bin: drafts at 0.92 are accepted half the time.
        for i in 0..400 {
            calibration.observe(0.92, i % 2 == 0);
        }
        assert!((calibration.acceptance(0.95).rate() - 0.5).abs() < 0.02);
        // Other bins keep their prior.
        assert!((calibration.acceptance(0.45).rate() - 0.45).abs() < 1e-12);
    }

    #[test]
    fn test_stats_count_acceptance_per_depth() {
        let mut stats = SpeculationStats::default();
        stats.observe_depths(3, 1);
        stats.observe_depths(10, 10);
        assert_eq!(stats.depth_drafted[..3], [2, 2, 2]);
        assert_eq!(stats.depth_accepted[..3], [2, 1, 1]);
        assert_eq!(stats.depth_drafted[STAT_DEPTHS - 1], 3);
    }

    #[test]
    fn test_config_validation() {
        assert!(SpeculationConfig::DISABLED.validate().is_ok());
        assert!(!SpeculationConfig::DISABLED.enabled());
        assert!(SpeculationConfig::new(4).validate().is_ok());
        let broken = SpeculationConfig {
            min_ngram: 3,
            max_ngram: 2,
            ..SpeculationConfig::new(4)
        };
        assert!(broken.validate().is_err());
    }

    #[test]
    fn test_long_history_lookup_has_a_bounded_index() {
        let config = SpeculationConfig::new(4);
        let mut index = NgramIndex::default();
        let tokens: Vec<u32> = (0..4 * INDEX_WINDOW).map(|i| i as u32).collect();
        for end in [
            INDEX_WINDOW,
            2 * INDEX_WINDOW - 1,
            2 * INDEX_WINDOW,
            tokens.len(),
        ] {
            index.propose(&tokens[..end], &config, 4);
            assert!(index.ends.len() <= 2 * INDEX_WINDOW * config.max_ngram);
        }
    }
}
