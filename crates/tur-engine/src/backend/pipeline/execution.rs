//! Packing, submitting and finishing one scheduled model pass.

use super::super::{scheduler::Chunk, speculation::DraftSource};
use super::{BatchStepOutput, InFlight, InferencePipeline, PassTiming};
use crate::Result;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tur_core::execution::{PackedExecution, SequenceDescriptor, invalid};
use uuid::Uuid;

impl InferencePipeline {
    /// Pack the selected chunks: each sequence's next unconsumed tokens, with logits at its
    /// last known token. A drafted decode appends its drafts and needs logits at every
    /// position, for verification. Returns the plan and its longest resulting context.
    fn pack(&self, chunks: &[Chunk], drafts: &[Vec<u32>]) -> Result<(PackedExecution, usize)> {
        let mut plan = PackedExecution {
            tokens: Vec::new(),
            positions: Vec::new(),
            sequences: Vec::with_capacity(chunks.len()),
        };
        let mut context = 0;
        for (chunk, draft) in chunks.iter().zip(drafts) {
            let pending = self
                .pending
                .get(&chunk.id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            let end = pending
                .consumed
                .checked_add(chunk.tokens)
                .ok_or_else(|| invalid("chunk position overflow"))?;
            let input = pending
                .tokens
                .get(pending.consumed..end)
                .ok_or_else(|| invalid("chunk beyond the request's tokens"))?;
            let start = plan.tokens.len();
            plan.tokens.extend_from_slice(input);
            plan.tokens.extend_from_slice(draft);
            let end = end
                .checked_add(draft.len())
                .ok_or_else(|| invalid("draft position overflow"))?;
            plan.positions.extend(pending.consumed..end);
            let speculative = !draft.is_empty();
            plan.sequences.push(SequenceDescriptor {
                state: pending.handle,
                tokens: start..plan.tokens.len(),
                outputs: if speculative {
                    (start..plan.tokens.len()).collect()
                } else if end == pending.tokens.len() {
                    vec![plan.tokens.len() - 1]
                } else {
                    Vec::new()
                },
                speculative,
            });
            context = context.max(end);
        }
        Ok((plan, context))
    }

    /// Expire deadlines, select decodes and prefill chunks (preempting under memory
    /// pressure) and submit them as one packed pass without waiting for it. Returns whether
    /// a pass is in flight. Until [`Self::finish_step`], callers may submit and cancel
    /// requests while the device runs; the next pass cannot be selected before this one's
    /// tokens are known, so at most one is in flight.
    pub fn begin_step(&mut self) -> Result<bool> {
        if self.in_flight.is_some() {
            return Err(invalid("a pass is already in flight"));
        }
        let begin = Instant::now();
        let mut output = std::mem::take(&mut self.staged);
        let selected = self
            .expire(begin, &mut output)
            .and_then(|()| self.select(&mut output));
        let chunks = match selected {
            Ok(chunks) => chunks,
            Err(error) => {
                self.staged = output;
                return Err(error);
            }
        };
        self.staged = output;
        if chunks.is_empty() {
            return Ok(false);
        }
        let (drafts, sources, decodes) = self.draft(&chunks)?;
        let chunk_tokens = chunks.iter().map(|chunk| chunk.tokens).sum::<usize>();
        let (plan, context) = self.pack(&chunks, &drafts)?;
        let (tokens, sequences) = (plan.tokens.len(), plan.sequences.len());
        let started = Instant::now();
        self.engine.submit(plan)?;
        let submitted = Instant::now();
        self.in_flight = Some(InFlight {
            chunks,
            drafts,
            sources,
            decodes,
            context,
            started,
            submitted,
            timing: PassTiming {
                tokens,
                sequences,
                drafts: tokens - chunk_tokens,
                schedule: started.saturating_duration_since(begin),
                encode: submitted.saturating_duration_since(started),
                idle_before: self
                    .engine
                    .completed_at()
                    .map(|at| started.saturating_duration_since(at)),
                ..PassTiming::default()
            },
        });
        Ok(true)
    }

    /// Wait for the pass from [`Self::begin_step`], if any, then publish each request's
    /// outputs. A request's model or decode error is returned only to that request;
    /// requests cancelled while the pass ran publish nothing.
    pub fn finish_step(&mut self) -> Result<BatchStepOutput> {
        let mut output = std::mem::take(&mut self.staged);
        let Some(flight) = self.in_flight.take() else {
            return Ok(output);
        };
        let waiting = Instant::now();
        let results = self.engine.finish()?;
        let mut timing = flight.timing;
        timing.overlap = waiting.saturating_duration_since(flight.submitted);
        let completed = self.engine.completed_at().unwrap_or_else(Instant::now);
        timing.wait = completed.saturating_duration_since(waiting);
        timing.device = completed.saturating_duration_since(flight.started);
        // Host work after completion, from the engine's sampling on.
        let published = completed.max(waiting);
        // Pass costs exclude host work that overlapped the device. The scheduler sizes
        // prefill from undrafted token costs: a drafted pass contributes its undrafted
        // tokens and time, once the per-draft cost is known.
        let device = timing.device.as_secs_f64();
        let undrafted = match timing.drafts {
            0 => Some(device),
            drafts => self
                .draft_costs
                .line(flight.decodes)
                .map(|(_, per_draft)| per_draft.mul_add(-(drafts as f64), device))
                .filter(|&seconds| seconds > 0.0),
        };
        if let Some(seconds) = undrafted {
            self.scheduler.record(
                timing.tokens - timing.drafts,
                flight.context,
                Duration::from_secs_f64(seconds),
            );
        }
        if flight.decodes == flight.chunks.len() {
            self.draft_costs.record(
                flight.chunks.len(),
                timing.drafts,
                timing.device.as_secs_f64(),
            );
        }
        if timing.drafts > 0 {
            self.speculation_stats.passes += 1;
        }
        let drafted: HashMap<Uuid, (usize, DraftSource)> = flight
            .chunks
            .iter()
            .zip(&flight.drafts)
            .zip(&flight.sources)
            .filter(|((_, draft), _)| !draft.is_empty())
            .map(|((chunk, draft), &source)| (chunk.id, (draft.len(), source)))
            .collect();
        for chunk in &flight.chunks {
            if let Some(pending) = self.pending.get_mut(&chunk.id) {
                pending.running = true;
                pending.last_run = chunk.ticket;
            }
        }
        // Committed positions each request's draft head still needs: (id, first, hidden).
        let mut heads = Vec::new();
        for result in results {
            let id = result.request;
            let Some(before) = self.pending.get(&id).map(|p| p.consumed) else {
                continue;
            };
            if let (Some(&(drafted, source)), Ok(commit)) = (drafted.get(&id), &result.outcome) {
                // Every accepted input after the decode token is an accepted draft.
                let accepted = commit.accepted_tokens.saturating_sub(1).min(drafted);
                // The head's estimate is of its first draft, which decides whether it runs:
                // deeper chained drafts are accepted less often, are weighed by their own
                // calibrated confidence, and folded in they switched the head off for
                // requests at four E2B sequences.
                let first = (drafted.min(1), accepted.min(1));
                if let Some(pending) = self.pending.get_mut(&id) {
                    // A sampled request's head estimate follows its drafts' calibrated
                    // confidence instead (see `draft_head`).
                    match source {
                        DraftSource::Lookup => pending.acceptance.observe(drafted, accepted),
                        DraftSource::Mtp if pending.greedy => {
                            pending.mtp_acceptance.observe(first.0, first.1);
                        }
                        DraftSource::Mtp => {}
                    }
                    pending.drafted += drafted;
                    pending.accepted_drafts += accepted;
                }
                let (confidences, greedy) = self
                    .pending
                    .get_mut(&id)
                    .map_or((Vec::new(), false), |p| {
                        (std::mem::take(&mut p.mtp_verifying), p.greedy)
                    });
                let stats = &mut self.speculation_stats;
                match source {
                    DraftSource::Lookup => {
                        self.acceptance.observe(drafted, accepted);
                        stats.observe_depths(drafted, accepted);
                    }
                    DraftSource::Mtp => {
                        if greedy {
                            self.mtp_acceptance.observe(first.0, first.1);
                        }
                        stats.mtp_drafted += drafted as u64;
                        stats.mtp_accepted += accepted as u64;
                        // Each draft the verification reached: those after a rejection
                        // were never tested.
                        for (depth, &confidence) in
                            confidences.iter().enumerate().take(drafted).take(accepted + 1)
                        {
                            let hit = depth < accepted;
                            self.calibration[usize::from(greedy)].observe(confidence, hit);
                            stats.observe_confidence(confidence, hit);
                        }
                    }
                }
                stats.sequences += 1;
                stats.drafted += drafted as u64;
                stats.accepted += accepted as u64;
            }
            // Constrained requests never take the head's drafts; it runs for no other.
            let drafts_from_head = self.pending.get(&id).is_some_and(|p| !p.constrained);
            if let (Some(hidden), Ok(commit)) = (&result.hidden, &result.outcome)
                && commit.accepted_tokens > 0
                && drafts_from_head
            {
                heads.push((id, before, hidden.narrow(0, 0, commit.accepted_tokens)?));
            }
            let logits = result.logits;
            let outcome = result
                .outcome
                .and_then(|commit| self.publish(id, commit, logits, &mut output));
            if let Err(error) = outcome {
                self.fail(id, error, &mut output)?;
            }
        }
        if !heads.is_empty() {
            timing.draft_head = self.draft_head(heads)?;
            self.speculation_stats.mtp_passes += 1;
            self.speculation_stats.mtp_seconds += timing.draft_head.as_secs_f64();
        }
        self.expire(Instant::now(), &mut output)?;
        timing.publish = published.elapsed();
        output.timing = Some(timing);
        Ok(output)
    }
}
