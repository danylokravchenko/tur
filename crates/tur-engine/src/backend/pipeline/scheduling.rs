//! Prefix restoration and pass selection under state-pool pressure.

use super::super::{
    prefix::HeadEntry,
    scheduler::{Candidate, Chunk, Decision},
};
use super::{BatchStepOutput, InferencePipeline, Pending};
use crate::{Result, TurError, hybrid::Demand};
use std::time::Instant;
use tur_core::execution::invalid;
use uuid::Uuid;

impl InferencePipeline {
    /// Whether waiting request `waiter` should wait for one of `producers` (running or
    /// admitted by this pass) to compute a prefix both share, instead of computing it
    /// concurrently: a producer has not yet reached the deepest snapshot it will insert
    /// within the shared tokens, and the cache holds nothing as deep. Only first
    /// admissions younger than the queue-age limit wait.
    fn awaits(&self, id: Uuid, waiter: &Pending, producers: &[Uuid], now: Instant) -> bool {
        if !waiter.boundaries.is_empty()
            || now.saturating_duration_since(waiter.arrival)
                >= self.scheduler.config().max_queue_wait
        {
            return false;
        }
        let head = self.needs_head(id);
        let mut cached = None;
        producers.iter().any(|id| {
            self.pending.get(id).is_some_and(|producer| {
                let shared = waiter.keys.shared(&producer.keys).min(waiter.prompt_tokens);
                let depth = self.prefix.reachable(shared, producer.prompt_tokens);
                let depth = producer
                    .turn_boundary
                    .filter(|&boundary| boundary <= shared)
                    .map_or(depth, |boundary| depth.max(boundary));
                depth > producer.consumed
                    && depth
                        > *cached.get_or_insert_with(|| {
                            self.prefix.depth(&waiter.keys, &waiter.tokens, head)
                        })
                    && waiter.tokens.get(..depth) == producer.tokens.get(..depth)
            })
        })
    }

    /// Waiting requests without state in prefill order, counting the tokens a snapshot would
    /// save, and the earliest arrival among them. Only ordering reads that count, so a replay
    /// that could not use the snapshot merely ranks too early.
    fn waiting_order(&self, now: Instant) -> (Vec<Uuid>, Option<Instant>) {
        let shortest = self.scheduler.config().shortest_prompt_first;
        let mut ranked = Vec::new();
        let mut oldest: Option<Instant> = None;
        for (id, pending) in &self.pending {
            if pending.running || pending.consumed > 0 {
                continue;
            }
            let cached = if shortest {
                self.prefix
                    .depth(&pending.keys, &pending.tokens, self.needs_head(*id))
            } else {
                0
            };
            let remaining = pending.tokens.len() - cached;
            let rank = self.scheduler.prefill_rank(remaining, pending.arrival, now);
            ranked.push((rank, *id));
            oldest = Some(oldest.map_or(pending.arrival, |o| o.min(pending.arrival)));
        }
        ranked.sort_unstable();
        (ranked.into_iter().map(|(_, id)| id).collect(), oldest)
    }

    /// For the first waiting requests without state in prefill order, while running slots
    /// remain, restore their deepest verified
    /// snapshot, or let them wait for a producer (see [`Self::awaits`]); a waiting one takes
    /// no slot. With
    /// [`PrefixConfig::restorable_first`](crate::backend::prefix::PrefixConfig) and the
    /// oldest waiting request younger than the queue-age limit, only restored requests take
    /// slots here, so they pass older misses. Restored requests run their remaining prompt;
    /// a snapshot of the whole prompt samples the first output from its logits without a
    /// pass.
    fn restore_prefixes(&mut self, output: &mut BatchStepOutput) -> Result<()> {
        if !self.prefix.config().enabled() {
            return Ok(());
        }
        let now = Instant::now();
        self.awaiting.clear();
        let mut producers = Vec::with_capacity(self.scheduler.config().max_running);
        producers.extend(
            self.pending
                .iter()
                .filter(|(_, p)| p.running)
                .map(|(id, _)| *id),
        );
        let (waiting, oldest) = self.waiting_order(now);
        let mut free = self
            .scheduler
            .config()
            .max_running
            .saturating_sub(producers.len());
        let restorable_first = self.prefix.config().restorable_first
            && oldest.is_some_and(|arrival| {
                now.saturating_duration_since(arrival) < self.scheduler.config().max_queue_wait
            });
        let inserted = self.prefix.stats().inserted;
        for id in waiting {
            if free == 0 {
                break;
            }
            let pending = self
                .pending
                .get(&id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            if self.awaits(id, pending, &producers, now) {
                self.awaiting.insert(id);
                continue;
            }
            producers.push(id);
            if !restorable_first {
                free -= 1;
            }
            let (head, keyed) = (self.needs_head(id), self.keyed_head());
            let page = self.engine.pool().description().kv.page_tokens;
            let pending = self
                .pending
                .get_mut(&id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            if pending.prefix_checked == Some(inserted) {
                continue;
            }
            // Recomputation keeps its original pass boundaries above the first restore.
            let (boundaries, floor) = (&pending.boundaries, pending.replay_floor);
            let limit = pending
                .scores
                .as_ref()
                .map_or(usize::MAX, super::scoring::Scores::restore_limit);
            let accept = |length: usize| {
                length <= limit
                    && (boundaries.is_empty()
                        || length <= floor
                        || boundaries.binary_search(&length).is_ok())
            };
            let Some(hit) = self
                .prefix
                .lookup(&pending.keys, &pending.tokens, head, accept)
            else {
                pending.prefix_checked = Some(inserted);
                continue;
            };
            if restorable_first {
                free -= 1;
            }
            pending.handle = self.engine.adopt(pending.handle, &hit.state)?;
            pending.consumed = hit.length;
            pending.running = true;
            resume_head(pending, hit.length, hit.head.as_ref(), keyed, page);
            if pending.boundaries.is_empty() {
                pending.replay_floor = hit.length;
                pending.cached_tokens = hit.length;
            }
            tracing::debug!(request = %id, tokens = hit.length, "restored prefix snapshot");
            if hit.length < pending.tokens.len() {
                continue;
            }
            let handle = pending.handle;
            let first = hit
                .logits
                .ok_or_else(|| invalid("whole-prompt snapshot without logits"))
                .and_then(|logits| self.engine.sample_logits(handle, &logits))
                .and_then(|commit| self.publish(id, commit, None, None, output));
            if let Err(error) = first {
                self.fail(id, error, output)?;
            }
            // The sampled output decides whether the head's last entry holds.
            if let Some(pending) = self
                .pending
                .get_mut(&id)
                .or_else(|| self.finishing.get_mut(&id).map(|f| &mut f.pending))
            {
                resume_head(pending, hit.length, hit.head.as_ref(), keyed, page);
            }
        }
        self.settle()
    }

    /// Whether `id` runs a draft head that keeps keys and values, so a snapshot it restores
    /// must hold the head's entries (see [`super::super::prefix`]).
    fn needs_head(&self, id: Uuid) -> bool {
        self.keyed_head() && !self.engine.constraining(id)
    }

    /// Evict snapshots while they keep the first admissible waiting request's next chunk
    /// (with one page of headroom per running sequence) from fitting.
    fn relieve_admission(&mut self, candidates: &[Candidate]) -> Result<()> {
        let running = candidates.iter().filter(|c| c.running).count();
        let now = Instant::now();
        let Some(first) = candidates
            .iter()
            .filter(|c| !c.running && running < self.scheduler.config().max_running)
            .min_by_key(|c| {
                let rank = self.scheduler.prefill_rank(c.remaining, c.arrival, now);
                (rank, c.id)
            })
        else {
            return Ok(());
        };
        let handle = self
            .pending
            .get(&first.id)
            .ok_or_else(|| invalid("missing pipeline request"))?
            .handle;
        let tokens = first.pending.min(self.scheduler.config().max_chunk);
        let headroom = growth(self.engine.pool().description(), running);
        let mut usage = self.engine.memory()?;
        while self.prefix.stats().entries > 0 {
            let demand = self.engine.demand(handle, tokens, false)? + headroom;
            if demand.fits(&usage) {
                break;
            }
            self.prefix.evict();
            let after = self.engine.memory()?;
            // Snapshots sharing pages with running requests free only their slot; stop
            // once evicting no longer helps a page shortage.
            if demand.slots <= after.slots.free()
                && after.pages.free() == usage.pages.free()
                && after.windows.free() == usage.windows.free()
            {
                break;
            }
            usage = after;
        }
        Ok(())
    }

    /// Ask the scheduler for a pass, evicting prefix snapshots before preempting or
    /// failing requests, until one fits.
    pub(super) fn select(&mut self, output: &mut BatchStepOutput) -> Result<Vec<Chunk>> {
        self.restore_prefixes(output)?;
        loop {
            let candidates: Vec<Candidate> = self
                .pending
                .iter()
                .filter(|(id, _)| !self.awaiting.contains(*id))
                // A paused chunk holds the pass's prefill work until it finishes.
                .filter(|(_, p)| self.layered.is_none() || p.decoding())
                .map(|(id, p)| p.candidate(*id, self.prefix.next_boundary(p.consumed)))
                .collect();
            self.relieve_admission(&candidates)?;
            let usage = self.engine.memory()?;
            let running = candidates.iter().filter(|c| c.running).count();
            let (engine, pending) = (&self.engine, &self.pending);
            let mut demand = Demand::default();
            let decision = self.scheduler.plan(&candidates, Instant::now(), |chunk| {
                let request = pending
                    .get(&chunk.id)
                    .ok_or_else(|| invalid("missing pipeline request"))?;
                let needed = demand + engine.demand(request.handle, chunk.tokens, false)?;
                // Admission keeps one page of growth per running sequence free, so new
                // work does not immediately force running decodes out.
                let headroom = growth(
                    engine.pool().description(),
                    usize::from(chunk.admit) * running,
                );
                let fits = (needed + headroom).fits(&usage);
                if fits {
                    demand = needed;
                }
                Ok(fits)
            })?;
            match decision {
                Decision::Run(chunks) => return Ok(chunks),
                Decision::Preempt(_) | Decision::Stall(_) if self.prefix.evict() => {}
                // Its pages come back without costing anyone their committed work.
                Decision::Preempt(_) | Decision::Stall(_) if self.abort_layered() => {}
                Decision::Preempt(id) => self.preempt(id)?,
                Decision::Stall(id) => self.fail(
                    id,
                    TurError::MemoryAllocation(format!(
                        "request cannot fit the state pool ({} of {} K/V pages and {} of {} \
                         recurrent slots held)",
                        usage.pages.leased,
                        usage.pages.capacity,
                        usage.slots.leased,
                        usage.slots.capacity
                    )),
                    output,
                )?,
            }
        }
    }
}

/// One page of growth for each of `sequences`, in every page space the model uses.
fn growth(description: &tur_core::execution::ModelDescription, sequences: usize) -> Demand {
    Demand {
        pages: if description.has_full_attention() {
            sequences
        } else {
            0
        },
        windows: if description.window().is_some() {
            sequences
        } else {
            0
        },
        slots: 0,
    }
}

/// Where the draft head of `pending`, restored to a snapshot of `length` tokens whose head
/// entry is `head`, continues. A head without keys and values always continues there. The
/// snapshot's last entry holds if the token after it is the one it read, or is not known
/// yet; otherwise the head recomputes it from the stored hidden state where it can, and
/// stays behind where it can't.
fn resume_head(
    pending: &mut Pending,
    length: usize,
    head: Option<&HeadEntry>,
    keyed: bool,
    page_tokens: usize,
) {
    pending.head_seed = None;
    pending.mtp_done = match (keyed, head) {
        (false, _) => length,
        (true, None) => 0,
        (true, Some(head)) => match pending.tokens.get(length) {
            Some(&next) if next != head.next => {
                if head.recomputable(length, page_tokens) {
                    pending.head_seed.clone_from(&head.hidden);
                    length - 1
                } else {
                    0
                }
            }
            _ => length,
        },
    };
}
