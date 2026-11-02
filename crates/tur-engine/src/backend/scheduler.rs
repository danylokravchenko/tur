#![forbid(unsafe_code)]
//! Iteration-level scheduling of packed hybrid passes.
//!
//! Each iteration selects one packed pass. Running decodes come first, one token each,
//! rotated by when they last ran when the token budget cannot hold them all. Prefill chunks
//! follow in arrival order within a budget sized from measured pass costs. While prefill
//! work fits every pass reserves at least `min_prefill_tokens`, or an indivisible replay
//! chunk. The oldest prefill beyond `max_queue_wait` ignores the cost target within the
//! token cap and may reclaim resources from younger running requests. The caller checks memory
//! per chunk: a decode that does not fit preempts the newest running request, which later
//! recomputes its tokens. Waiting behind older work is not bounded by the queue-age cap.

use std::time::{Duration, Instant};
use tur_core::{Result, TurError, execution::invalid};
use uuid::Uuid;

/// Admission limits and pass shaping for one pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Requests holding model state at once.
    pub max_running: usize,
    /// Submitted requests not yet running; further submissions are rejected.
    pub max_waiting: usize,
    /// Prompt tokens of waiting requests; further submissions are rejected.
    pub max_waiting_tokens: usize,
    /// Tokens in one packed pass.
    pub max_batch_tokens: usize,
    /// Largest prefill chunk of one sequence in one pass.
    pub max_chunk: usize,
    /// Prefill tokens every pass grants while prefill work exists, whatever its cost.
    pub min_prefill_tokens: usize,
    /// Pass duration the adaptive prefill budget aims for while decodes run. A pass may
    /// also take up to twice its decode-only estimate, so prefill is never starved by a
    /// target below the cost of the decodes themselves.
    pub target_pass: Duration,
    /// Queue age after which a request's next chunk ignores the cost target.
    pub max_queue_wait: Duration,
}

impl SchedulerConfig {
    /// Defaults for `max_running` sequences and `max_chunk`-token prefill chunks: a pass
    /// holds up to 2048 tokens, and at least one full chunk beside every decode.
    pub fn new(max_running: usize, max_chunk: usize) -> Self {
        Self {
            max_running,
            max_waiting: 256,
            max_waiting_tokens: 1 << 20,
            max_batch_tokens: max_chunk.saturating_add(max_running).max(2048),
            max_chunk,
            min_prefill_tokens: max_chunk.min(32),
            target_pass: Duration::from_millis(60),
            max_queue_wait: Duration::from_secs(2),
        }
    }

    /// Positive limits, with room in every pass for one decode beside the prefill minimum.
    pub fn validate(&self) -> Result<()> {
        if self.max_running == 0
            || self.max_chunk == 0
            || self.min_prefill_tokens == 0
            || self.min_prefill_tokens > self.max_chunk
            || self.max_chunk > self.max_batch_tokens
            || self.min_prefill_tokens >= self.max_batch_tokens
            || self.target_pass.is_zero()
        {
            return Err(invalid(
                "scheduler limits must be positive, with min prefill ≤ chunk ≤ batch tokens \
                 and room for a decode beside the prefill minimum",
            ));
        }
        Ok(())
    }
}

/// One request the scheduler may run this iteration.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub id: Uuid,
    /// Tokens available for the next chunk: one for a decode, or more when a grammar forced
    /// some, remaining prompt tokens, or the next original chunk during recomputation.
    pub pending: usize,
    /// Minimum indivisible chunk. Recompute uses the original pass boundary so narrow
    /// recurrent state is rounded at the same positions as before preemption.
    pub minimum: usize,
    /// Its pending tokens are outputs, which run with the decodes instead of waiting for
    /// prefill budget.
    pub decoding: bool,
    /// Holds model state; otherwise admitting it counts against `max_running`.
    pub running: bool,
    /// Tokens already in model state.
    pub context: usize,
    pub arrival: Instant,
    /// Ticket of the chunk this request last ran; zero if never.
    pub last_run: u64,
}

/// New tokens of one request in the selected pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub id: Uuid,
    pub tokens: usize,
    /// The request is not running yet and is admitted by this pass.
    pub admit: bool,
    /// Increases with every scheduled chunk; the caller stores it as the request's
    /// `last_run`, so skipped decodes go first next time in strict rotation.
    pub ticket: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Execute these chunks as one packed pass; empty when nothing is pending.
    Run(Vec<Chunk>),
    /// Release this running request's state; it recomputes its tokens later.
    Preempt(Uuid),
    /// This request cannot make progress even alone; fail it.
    Stall(Uuid),
}

const TOKEN_BUCKETS: usize = 16;
const CONTEXT_BUCKETS: usize = 8;
/// Weight of the newest sample in each bucket's moving average.
const SMOOTHING: f64 = 0.25;

/// Measured seconds per token, as moving averages over (packed tokens, context) buckets of
/// powers of two. Unmeasured shapes are estimated from the nearest measured token bucket
/// of the nearest measured context bucket, scaled linearly by tokens.
#[derive(Clone, Debug, Default)]
pub struct CostModel {
    cells: [[Option<f64>; CONTEXT_BUCKETS]; TOKEN_BUCKETS],
}

fn token_bucket(tokens: usize) -> usize {
    (tokens.max(1).ilog2() as usize).min(TOKEN_BUCKETS - 1)
}

fn context_bucket(context: usize) -> usize {
    ((context / 1024 + 1).ilog2() as usize).min(CONTEXT_BUCKETS - 1)
}

impl CostModel {
    /// Fold one completed pass into its bucket; non-finite or negative samples are ignored.
    pub fn record(&mut self, tokens: usize, context: usize, seconds: f64) {
        if tokens == 0 || !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        let cell = &mut self.cells[token_bucket(tokens)][context_bucket(context)];
        let per_token = seconds / tokens as f64;
        *cell = Some(cell.map_or(per_token, |mean| SMOOTHING.mul_add(per_token - mean, mean)));
    }

    /// Estimated seconds of a pass of `tokens` packed tokens at `context`, if measured.
    pub fn estimate(&self, tokens: usize, context: usize) -> Option<f64> {
        let (column, bucket) = (context_bucket(context), token_bucket(tokens));
        let (_, _, mean) = (0..TOKEN_BUCKETS)
            .flat_map(|b| (0..CONTEXT_BUCKETS).map(move |c| (b, c)))
            .filter_map(|(b, c)| self.cells[b][c].map(|mean| (c, b, mean)))
            .min_by_key(|(c, b, _)| (c.abs_diff(column), b.abs_diff(bucket)))?;
        Some(mean * tokens as f64)
    }
}

/// Decode-priority scheduler with bounded, cost-adaptive prefill.
#[derive(Clone, Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    costs: CostModel,
    /// Prefill budget of the previous pass; each pass moves at most a factor of two.
    budget: usize,
    iteration: u64,
    tickets: u64,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            budget: config.max_chunk,
            config,
            costs: CostModel::default(),
            iteration: 0,
            tickets: 0,
        })
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn costs(&self) -> &CostModel {
        &self.costs
    }

    /// Number of passes selected so far.
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Reject a submission that would exceed the waiting queue's request or token limits.
    pub fn admit(&self, waiting: usize, waiting_tokens: usize, tokens: usize) -> Result<()> {
        if waiting >= self.config.max_waiting
            || waiting_tokens.saturating_add(tokens) > self.config.max_waiting_tokens
        {
            return Err(TurError::QueueFull(format!(
                "{waiting} requests, {waiting_tokens} tokens waiting"
            )));
        }
        Ok(())
    }

    /// Record a completed pass of `tokens` packed tokens whose longest sequence ended at
    /// `context`.
    pub fn record(&mut self, tokens: usize, context: usize, elapsed: Duration) {
        self.costs.record(tokens, context, elapsed.as_secs_f64());
    }

    /// Largest prefill budget within `cap` whose estimated pass meets the target, moved at
    /// most a factor of two from the previous pass and never below the prefill minimum.
    /// Without decodes no latency is at stake, and prefill takes the whole budget.
    fn prefill_budget(&mut self, decodes: usize, context: usize, cap: usize) -> usize {
        if decodes == 0 {
            self.budget = cap;
            return cap;
        }
        let floor = self.config.min_prefill_tokens.min(cap);
        let decode_only = self.costs.estimate(decodes, context).unwrap_or(0.0);
        let target = self.config.target_pass.as_secs_f64().max(2.0 * decode_only);
        let mut wanted = cap;
        while wanted > floor
            && self
                .costs
                .estimate(decodes + wanted, context)
                .is_some_and(|seconds| seconds > target)
        {
            wanted /= 2;
        }
        let low = (self.budget / 2).max(floor);
        let high = self.budget.saturating_mul(2).max(low);
        self.budget = wanted.clamp(low, high).min(cap).max(floor);
        self.budget
    }

    /// Select the next pass. `fits(chunk)` reports whether the chunk fits memory beside
    /// the chunks already accepted, and accounts for it when it does.
    pub fn plan(
        &mut self,
        candidates: &[Candidate],
        now: Instant,
        mut fits: impl FnMut(&Chunk) -> Result<bool>,
    ) -> Result<Decision> {
        if candidates.is_empty() {
            return Ok(Decision::Run(Vec::new()));
        }
        if candidates
            .iter()
            .any(|c| c.minimum == 0 || c.minimum > c.pending || c.minimum > self.config.max_chunk)
        {
            return Err(invalid(
                "candidate minimum must fit its pending tokens and the chunk limit",
            ));
        }
        let newest_running = candidates
            .iter()
            .filter(|c| c.running)
            .max_by_key(|c| (c.arrival, c.id));
        let running = candidates.iter().filter(|c| c.running).count();
        let mut decodes: Vec<&Candidate> = candidates
            .iter()
            .filter(|c| c.running && c.decoding)
            .collect();
        decodes.sort_by_key(|c| (c.last_run, c.arrival, c.id));
        let mut prefills: Vec<&Candidate> = candidates
            .iter()
            .filter(|c| !c.decoding && c.pending > 0)
            .collect();
        prefills.sort_by_key(|c| (c.arrival, c.id));
        let aged = prefills.first().copied().filter(|oldest| {
            now.saturating_duration_since(oldest.arrival) >= self.config.max_queue_wait
        });
        let younger_victim = aged.and_then(|oldest| {
            newest_running
                .filter(|victim| (victim.arrival, victim.id) > (oldest.arrival, oldest.id))
        });
        if let (Some(oldest), Some(victim)) = (aged, younger_victim)
            && !oldest.running
            && running >= self.config.max_running
        {
            return Ok(Decision::Preempt(victim.id));
        }

        // Hold back the prefill minimum only for prefill that can run: a waiting request
        // with no free running slot leaves the whole pass to decodes.
        let eligible = prefills
            .iter()
            .find(|p| p.running || running < self.config.max_running);
        let reserve = if let Some(prefill) = eligible {
            self.config.min_prefill_tokens.max(prefill.minimum)
        } else {
            0
        };
        let mut chunks = Vec::new();
        let room = self.config.max_batch_tokens - reserve;
        let mut used = 0usize;
        for (index, decode) in decodes.iter().enumerate().take(room) {
            // Later decodes keep their one token each. A recomputed run too long for what's
            // left waits for a later pass, so it stays on its original boundaries.
            let spare = room - used - (decodes.len() - index - 1).min(room - used - 1);
            let tokens = decode.pending.min(self.config.max_chunk).min(spare);
            if tokens < decode.minimum {
                continue;
            }
            let chunk = Chunk {
                id: decode.id,
                tokens,
                admit: false,
                ticket: 0,
            };
            if !fits(&chunk)? {
                return Ok(match newest_running {
                    Some(victim) if running > 1 => Decision::Preempt(victim.id),
                    _ => Decision::Stall(decode.id),
                });
            }
            used += tokens;
            chunks.push(chunk);
            if used == room {
                break;
            }
        }

        if let Some(oldest) = prefills.first() {
            let cap = self.config.max_batch_tokens - used;
            let context = candidates
                .iter()
                .map(|c| {
                    c.context
                        .saturating_add(c.pending.min(self.config.max_chunk))
                })
                .max()
                .unwrap_or(0);
            let mut budget = self.prefill_budget(used, context, cap);
            budget = budget.max(oldest.minimum.min(cap));
            if now.saturating_duration_since(oldest.arrival) >= self.config.max_queue_wait {
                budget = budget.max(self.config.max_chunk.min(cap));
            }
            let mut admitted = running;
            for prefill in &prefills {
                if budget == 0 {
                    break;
                }
                if !prefill.running && admitted >= self.config.max_running {
                    continue;
                }
                let chunk = Chunk {
                    id: prefill.id,
                    tokens: prefill.pending.min(self.config.max_chunk).min(budget),
                    admit: !prefill.running,
                    ticket: 0,
                };
                if chunk.tokens < prefill.minimum {
                    continue;
                }
                if fits(&chunk)? {
                    budget -= chunk.tokens;
                    admitted += usize::from(chunk.admit);
                    chunks.push(chunk);
                } else if aged.is_some_and(|oldest| oldest.id == prefill.id)
                    && let Some(victim) = younger_victim
                {
                    // An age guard on cost alone cannot unblock a prefill whose pages
                    // are held by younger decodes. Discard this pass and reclaim them.
                    return Ok(Decision::Preempt(victim.id));
                }
            }
        }

        if chunks.is_empty() {
            // Nothing fits: free memory held by the newest running request, or fail the
            // oldest request if it cannot run with the pool to itself.
            return Ok(match newest_running {
                Some(victim) if running > 1 => Decision::Preempt(victim.id),
                Some(victim) => Decision::Stall(victim.id),
                None => prefills
                    .first()
                    .copied()
                    .or(candidates.first())
                    .map_or_else(
                        || Decision::Run(Vec::new()),
                        |oldest| Decision::Stall(oldest.id),
                    ),
            });
        }
        self.iteration += 1;
        for chunk in &mut chunks {
            self.tickets += 1;
            chunk.ticket = self.tickets;
        }
        Ok(Decision::Run(chunks))
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    fn config() -> SchedulerConfig {
        SchedulerConfig {
            max_running: 4,
            max_waiting: 8,
            max_waiting_tokens: 1000,
            max_batch_tokens: 20,
            max_chunk: 16,
            min_prefill_tokens: 4,
            target_pass: Duration::from_millis(10),
            max_queue_wait: Duration::from_secs(1),
        }
    }

    fn candidate(index: u128, start: Instant, pending: usize, running: bool) -> Candidate {
        Candidate {
            id: Uuid::from_u128(index),
            pending,
            minimum: 1,
            decoding: running && pending == 1,
            running,
            context: 10,
            arrival: start + Duration::from_millis(index as u64),
            last_run: 0,
        }
    }

    fn chunks(decision: Decision) -> Result<Vec<Chunk>> {
        match decision {
            Decision::Run(chunks) => Ok(chunks),
            other => Err(invalid(format!("expected a pass, got {other:?}"))),
        }
    }

    fn run(decision: Decision) -> Result<Vec<(u128, usize)>> {
        Ok(chunks(decision)?
            .iter()
            .map(|c| (c.id.as_u128(), c.tokens))
            .collect())
    }

    #[test]
    fn test_config_validation_rejects_passes_without_decode_room() {
        assert!(SchedulerConfig::new(8, 256).validate().is_ok());
        assert!(SchedulerConfig::new(1, 1).validate().is_ok());
        for broken in [
            SchedulerConfig {
                max_running: 0,
                ..config()
            },
            SchedulerConfig {
                min_prefill_tokens: 20,
                max_chunk: 20,
                ..config()
            },
            SchedulerConfig {
                max_chunk: 21,
                ..config()
            },
            SchedulerConfig {
                target_pass: Duration::ZERO,
                ..config()
            },
        ] {
            assert!(Scheduler::new(broken).is_err());
        }
    }

    #[test]
    fn test_decodes_run_first_and_prefill_keeps_its_minimum() -> Result<()> {
        let start = Instant::now();
        let mut scheduler = Scheduler::new(config())?;
        let mut candidates: Vec<_> = (1..=4).map(|i| candidate(i, start, 1, true)).collect();
        candidates.push(candidate(5, start, 100, false));
        let chunks = run(scheduler.plan(&candidates, start, |_| Ok(true))?)?;
        assert_eq!(&chunks[..4], &[(1, 1), (2, 1), (3, 1), (4, 1)]);
        // Four running requests fill max_running, so the waiting prefill is not admitted.
        assert_eq!(chunks.len(), 4);
        // With room to admit it, the prefill gets the rest of the budget up to a chunk.
        candidates.remove(3);
        let chunks = run(scheduler.plan(&candidates, start, |_| Ok(true))?)?;
        assert_eq!(chunks.last(), Some(&(5, 16)));
        // A prefill that cannot be admitted reserves nothing from the decodes.
        let mut full = candidates.clone();
        full.push(candidate(6, start, 1, true));
        full.extend((7..=25).map(|i| candidate(i, start, 1, true)));
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_running: 23,
            ..config()
        })?;
        let chunks = run(scheduler.plan(&full, start, |_| Ok(true))?)?;
        assert_eq!(chunks.len(), 20);
        assert!(chunks.iter().all(|&(_, tokens)| tokens == 1));
        // Twenty decodes leave the minimum prefill budget: 16 decodes and 4 prefill tokens.
        let mut config = config();
        config.max_running = 32;
        let mut scheduler = Scheduler::new(config)?;
        let mut many: Vec<_> = (1..=20).map(|i| candidate(i, start, 1, true)).collect();
        many.push(candidate(30, start, 100, false));
        let chunks = run(scheduler.plan(&many, start, |_| Ok(true))?)?;
        assert_eq!(chunks.iter().filter(|(_, n)| *n == 1).count(), 16);
        assert_eq!(chunks.last(), Some(&(30, 4)));
        Ok(())
    }

    #[test]
    fn test_forced_runs_decode_whole_and_leave_each_later_decode_a_token() -> Result<()> {
        let start = Instant::now();
        let mut config = SchedulerConfig::new(4, 8);
        config.max_batch_tokens = 8;
        config.min_prefill_tokens = 2;
        let mut scheduler = Scheduler::new(config)?;
        let forced = |index, pending| Candidate {
            decoding: true,
            ..candidate(index, start, pending, true)
        };
        let run = |scheduler: &mut Scheduler, candidates: &[Candidate]| {
            chunks(scheduler.plan(candidates, start, |_| Ok(true))?)
        };
        let lengths = |chunks: Vec<Chunk>| chunks.iter().map(|c| c.tokens).collect::<Vec<_>>();
        // A short run goes whole, in the same pass as a plain decode.
        assert_eq!(
            lengths(run(&mut scheduler, &[forced(1, 3), forced(2, 1)])?),
            [3, 1]
        );
        // A long one stops where the batch ends, one token short for the decode after it.
        assert_eq!(
            lengths(run(&mut scheduler, &[forced(1, 100), forced(2, 1)])?),
            [7, 1]
        );
        // Prefill keeps its reserve beside a run.
        let prefill = candidate(3, start, 50, false);
        let selected = run(&mut scheduler, &[forced(1, 100), prefill])?;
        assert_eq!(selected[0].tokens, 6);
        assert_eq!(selected[1].id, prefill.id);
        Ok(())
    }

    #[test]
    fn test_skipped_decodes_rotate_to_the_front() -> Result<()> {
        let start = Instant::now();
        let mut config = config();
        config.max_running = 32;
        let mut scheduler = Scheduler::new(config)?;
        let mut candidates: Vec<_> = (1..=30).map(|i| candidate(i, start, 1, true)).collect();
        let mut runs = vec![0usize; 31];
        for _ in 0..6 {
            let selected = chunks(scheduler.plan(&candidates, start, |_| Ok(true))?)?;
            assert_eq!(selected.len(), 20);
            for chunk in selected {
                let id = chunk.id.as_u128() as usize;
                runs[id] += 1;
                candidates[id - 1].last_run = chunk.ticket;
            }
        }
        // 120 decode slots over 30 requests: every request ran exactly four times.
        assert!(runs[1..].iter().all(|&n| n == 4), "{runs:?}");
        Ok(())
    }

    #[test]
    fn test_measured_costs_shrink_prefill_by_bounded_steps() -> Result<()> {
        let start = Instant::now();
        let mut config = config();
        config.max_batch_tokens = 1024;
        config.max_chunk = 1024;
        let mut scheduler = Scheduler::new(config)?;
        // One millisecond per token: the 10 ms target allows about 8 tokens.
        for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            scheduler.record(tokens, 10, Duration::from_millis(tokens as u64));
        }
        let estimate = scheduler.costs().estimate(100, 10).unwrap_or_default();
        assert!((estimate - 0.1).abs() < 1e-3, "{estimate}");
        // Longer contexts without measurements borrow the nearest measured context.
        let far = scheduler.costs().estimate(100, 50_000).unwrap_or_default();
        assert!((far - 0.1).abs() < 1e-3, "{far}");
        assert!(CostModel::default().estimate(1, 1).is_none());
        // Without decodes prefill takes the whole pass.
        let alone = [candidate(1, start, 5000, false)];
        assert_eq!(
            run(scheduler.plan(&alone, start, |_| Ok(true))?)?[0].1,
            1024
        );
        let prefill = [
            candidate(1, start, 5000, false),
            candidate(2, start, 1, true),
        ];
        let mut budgets = Vec::new();
        for _ in 0..10 {
            let chunks = run(scheduler.plan(&prefill, start, |_| Ok(true))?)?;
            assert_eq!(chunks[0], (2, 1));
            budgets.push(chunks[1].1);
        }
        // Halving each pass from the full budget down to the cost target: one decode and
        // seven prefill tokens take 8 ms of the 10 ms target.
        assert_eq!(budgets, [512, 256, 128, 64, 32, 16, 8, 7, 7, 7]);
        // A single noisy sample cannot drop the budget below half, nor below the minimum.
        scheduler.record(9, 10, Duration::from_secs(10));
        let chunk = run(scheduler.plan(&prefill, start, |_| Ok(true))?)?[1].1;
        assert!(chunk >= 4);
        // An old request ignores the cost target and gets a full chunk.
        let late = start + Duration::from_secs(2);
        assert_eq!(
            run(scheduler.plan(&prefill, late, |_| Ok(true))?)?[1].1,
            1023
        );
        // Decodes slower than the target let prefill take up to their own cost again.
        let mut slow = Scheduler::new(SchedulerConfig {
            max_running: 16,
            ..config
        })?;
        for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            slow.record(tokens, 10, Duration::from_millis(40 + tokens as u64));
        }
        let decodes: Vec<_> = (2..=9).map(|i| candidate(i, start, 1, true)).collect();
        let mut mixed = vec![candidate(1, start, 5000, false)];
        mixed.extend(decodes);
        for _ in 0..12 {
            run(slow.plan(&mixed, start, |_| Ok(true))?)?;
        }
        // 8 decodes cost 48 ms; the pass may take 96 ms, which about 31 prefill tokens fill.
        let budget = run(slow.plan(&mixed, start, |_| Ok(true))?)?
            .last()
            .map_or(0, |c| c.1);
        assert!((24..=64).contains(&budget), "{budget}");
        Ok(())
    }

    #[test]
    fn test_memory_shortage_preempts_the_newest_running_request() -> Result<()> {
        let start = Instant::now();
        let mut scheduler = Scheduler::new(config())?;
        let candidates = [
            candidate(1, start, 1, true),
            candidate(2, start, 1, true),
            candidate(3, start, 9, true),
        ];
        // The second decode does not fit: the newest running request (a prefill) yields.
        let mut accepted = 0;
        let decision = scheduler.plan(&candidates, start, |_| {
            accepted += 1;
            Ok(accepted < 2)
        })?;
        assert_eq!(decision, Decision::Preempt(Uuid::from_u128(3)));
        // Prefills that do not fit wait; when nothing fits, the newest yields.
        let prefills = [candidate(1, start, 9, true), candidate(2, start, 9, true)];
        let decision = scheduler.plan(&prefills, start, |_| Ok(false))?;
        assert_eq!(decision, Decision::Preempt(Uuid::from_u128(2)));
        // Alone, a request that cannot fit is failed rather than preempted forever.
        let decision = scheduler.plan(&candidates[..1], start, |_| Ok(false))?;
        assert_eq!(decision, Decision::Stall(Uuid::from_u128(1)));
        let decision = scheduler.plan(&[candidate(7, start, 9, false)], start, |_| Ok(false))?;
        assert_eq!(decision, Decision::Stall(Uuid::from_u128(7)));
        Ok(())
    }

    #[test]
    fn test_waiting_queue_limits_requests_and_tokens() -> Result<()> {
        let mut scheduler = Scheduler::new(config())?;
        assert!(scheduler.admit(7, 0, 1000).is_ok());
        assert!(scheduler.admit(8, 0, 1).is_err());
        assert!(scheduler.admit(0, 900, 101).is_err());
        assert!(scheduler.admit(0, usize::MAX, 1).is_err());
        assert_eq!(
            scheduler.plan(&[], Instant::now(), |_| Ok(true))?,
            Decision::Run(vec![])
        );
        Ok(())
    }

    #[test]
    fn test_cost_samples_are_normalized_for_ragged_token_counts() {
        let mut costs = CostModel::default();
        costs.record(7, 10, 0.070);
        assert!((costs.estimate(7, 10).unwrap_or_default() - 0.070).abs() < 1e-12);
        costs.record(5, 10, 0.050);
        assert!((costs.estimate(6, 10).unwrap_or_default() - 0.060).abs() < 1e-12);
        costs.record(0, 10, 1.0);
        costs.record(6, 10, f64::NAN);
        assert!((costs.estimate(6, 10).unwrap_or_default() - 0.060).abs() < 1e-12);
    }

    #[test]
    fn test_aged_prefill_reclaims_younger_state_and_running_slots() -> Result<()> {
        let start = Instant::now();
        let mut scheduler = Scheduler::new(config())?;
        let mut older = candidate(1, start, 16, true);
        older.decoding = false;
        let younger = candidate(2, start, 1, true);
        let now = start + Duration::from_secs(2);
        assert_eq!(
            scheduler.plan(&[older, younger], now, |c| Ok(c.id == younger.id))?,
            Decision::Preempt(younger.id)
        );
        older.running = false;
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_running: 1,
            ..config()
        })?;
        assert_eq!(
            scheduler.plan(&[older, younger], now, |_| Ok(true))?,
            Decision::Preempt(younger.id)
        );
        // Younger waiting work cannot evict the older running request.
        let mut waiting = candidate(3, start, 16, false);
        waiting.arrival = start + Duration::from_millis(3);
        let selected = chunks(scheduler.plan(&[younger, waiting], now, |_| Ok(true))?)?;
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, younger.id);
        Ok(())
    }

    #[test]
    fn test_replay_chunks_remain_indivisible_under_cost_and_token_pressure() -> Result<()> {
        let start = Instant::now();
        let mut scheduler = Scheduler::new(config())?;
        scheduler.record(16, 10, Duration::from_secs(1));
        let mut replay = candidate(1, start, 16, true);
        replay.minimum = 16;
        replay.decoding = false;
        let mut candidates = vec![replay];
        candidates.extend((2..=9).map(|id| candidate(id, start, 1, true)));
        let selected = chunks(scheduler.plan(&candidates, start, |_| Ok(true))?)?;
        assert_eq!(selected.iter().map(|c| c.tokens).sum::<usize>(), 20);
        assert_eq!(
            selected.last().map(|c| (c.id, c.tokens)),
            Some((replay.id, 16))
        );
        // A replay waiting for a running slot must not reserve its large chunk at
        // the expense of decodes when only a different, ordinary prefill can run.
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_running: 20,
            ..config()
        })?;
        replay.running = false;
        let mut candidates = vec![replay];
        candidates.extend((2..=20).map(|id| candidate(id, start, 1, true)));
        candidates.push(candidate(21, start, 16, true));
        let selected = chunks(scheduler.plan(&candidates, start, |_| Ok(true))?)?;
        assert_eq!(selected.len(), 17);
        assert_eq!(selected.last().map(|c| c.tokens), Some(4));
        Ok(())
    }
}
