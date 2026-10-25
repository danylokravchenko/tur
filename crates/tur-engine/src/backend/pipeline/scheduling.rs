//! Prefix restoration and pass selection under state-pool pressure.

use super::super::scheduler::{Candidate, Chunk, Decision};
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
    fn awaits(&self, waiter: &Pending, producers: &[Uuid], now: Instant) -> bool {
        if !waiter.boundaries.is_empty()
            || now.saturating_duration_since(waiter.arrival)
                >= self.scheduler.config().max_queue_wait
        {
            return false;
        }
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
                        > *cached
                            .get_or_insert_with(|| self.prefix.depth(&waiter.keys, &waiter.tokens))
                    && waiter.tokens.get(..depth) == producer.tokens.get(..depth)
            })
        })
    }

    /// For the oldest waiting requests without state, while running slots remain, restore
    /// their deepest verified snapshot, or let them wait for a producer (see
    /// [`Self::awaits`]); a waiting one takes no slot. With
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
        let mut waiting = Vec::new();
        for (id, pending) in &self.pending {
            if pending.running {
                producers.push(*id);
            } else if pending.consumed == 0 {
                waiting.push((pending.arrival, *id));
            }
        }
        waiting.sort_unstable();
        let mut free = self
            .scheduler
            .config()
            .max_running
            .saturating_sub(producers.len());
        let restorable_first = self.prefix.config().restorable_first
            && waiting.first().is_some_and(|(arrival, _)| {
                now.saturating_duration_since(*arrival) < self.scheduler.config().max_queue_wait
            });
        let inserted = self.prefix.stats().inserted;
        for (_, id) in waiting {
            if free == 0 {
                break;
            }
            let pending = self
                .pending
                .get(&id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            if self.awaits(pending, &producers, now) {
                self.awaiting.insert(id);
                continue;
            }
            producers.push(id);
            if !restorable_first {
                free -= 1;
            }
            let pending = self
                .pending
                .get_mut(&id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            if pending.prefix_checked == Some(inserted) {
                continue;
            }
            // Recomputation keeps its original pass boundaries above the first restore.
            let (boundaries, floor) = (&pending.boundaries, pending.replay_floor);
            let accept = |length: usize| {
                boundaries.is_empty()
                    || length <= floor
                    || boundaries.binary_search(&length).is_ok()
            };
            let Some(hit) = self.prefix.lookup(&pending.keys, &pending.tokens, accept) else {
                pending.prefix_checked = Some(inserted);
                continue;
            };
            if restorable_first {
                free -= 1;
            }
            pending.handle = self.engine.adopt(pending.handle, &hit.state)?;
            pending.consumed = hit.length;
            pending.running = true;
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
                .and_then(|commit| self.publish(id, commit, None, output));
            if let Err(error) = first {
                self.fail(id, error, output)?;
            }
        }
        Ok(())
    }

    /// Evict snapshots while they keep the oldest admissible waiting request's next chunk
    /// (with one page of headroom per running sequence) from fitting.
    fn relieve_admission(&mut self, candidates: &[Candidate]) -> Result<()> {
        let running = candidates.iter().filter(|c| c.running).count();
        let Some(oldest) = candidates
            .iter()
            .filter(|c| !c.running && running < self.scheduler.config().max_running)
            .min_by_key(|c| (c.arrival, c.id))
        else {
            return Ok(());
        };
        let handle = self
            .pending
            .get(&oldest.id)
            .ok_or_else(|| invalid("missing pipeline request"))?
            .handle;
        let tokens = oldest.pending.min(self.scheduler.config().max_chunk);
        let headroom = Demand {
            pages: running,
            slots: 0,
        };
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
            if demand.slots <= after.slots.free() && after.pages.free() == usage.pages.free() {
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
                let headroom = Demand {
                    pages: if chunk.admit { running } else { 0 },
                    slots: 0,
                };
                let fits = (needed + headroom).fits(&usage);
                if fits {
                    demand = needed;
                }
                Ok(fits)
            })?;
            match decision {
                Decision::Run(chunks) => return Ok(chunks),
                Decision::Preempt(_) | Decision::Stall(_) if self.prefix.evict() => {}
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
