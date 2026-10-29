//! Draft proposals and native MTP head execution for a scheduled pass.

use super::super::{
    scheduler::Chunk,
    speculation::{self, Acceptance, DraftSource, Proposal},
};
use super::InferencePipeline;
use crate::{
    Result,
    hybrid::{Demand, DraftRequest},
};
use candle_core::Tensor;
use std::time::{Duration, Instant};
use tur_core::execution::invalid;
use uuid::Uuid;

impl InferencePipeline {
    /// Drafts for each chunk of a pass, their sources, and how many chunks are decodes. Every
    /// decoding sequence may draft, also beside prefill chunks once draft costs are
    /// identified: the prefill's estimated cost then joins the pass base, so drafts are
    /// weighed against the whole pass they lengthen. Proposals come from each request's
    /// own history and are capped so that every output stays within its request's limit,
    /// the model context and the next prefix snapshot boundary; they never contain EOS.
    /// [`speculation::plan`] chooses how many to verify. Drafts take only what the
    /// scheduled pass left: pass tokens, and free pages and recurrent slots beside the
    /// pass's ordinary demand. No growth headroom is held back: rejected drafts return
    /// their leases when the pass completes, and accepted ones are progress a decode
    /// would have leased anyway.
    pub(super) fn draft(
        &mut self,
        chunks: &[Chunk],
    ) -> Result<(Vec<Vec<u32>>, Vec<DraftSource>, usize)> {
        let mut drafts = vec![Vec::new(); chunks.len()];
        let mut sources = vec![DraftSource::Lookup; chunks.len()];
        let decoding: Vec<bool> = chunks
            .iter()
            .map(|chunk| {
                self.pending.get(&chunk.id).is_some_and(|p| {
                    !chunk.admit
                        && chunk.tokens == 1
                        && p.running
                        && p.generated() > 0
                        && p.tokens.len() == p.consumed + 1
                })
            })
            .collect();
        let decodes = decoding.iter().filter(|&&d| d).count();
        let decode_only = decodes > 0 && decodes == chunks.len();
        if decodes == 0 || !self.speculation.enabled() {
            return Ok((drafts, sources, decodes));
        }
        let chunk_tokens = chunks.iter().map(|chunk| chunk.tokens).sum::<usize>();
        let room = self
            .scheduler
            .config()
            .max_batch_tokens
            .saturating_sub(chunk_tokens);
        // A mixed pass drafts only against identified costs, with its prefill's estimated
        // time added to the base; without either estimate it runs undrafted.
        let mixed = if decode_only {
            None
        } else {
            let context = chunks
                .iter()
                .filter_map(|chunk| {
                    let p = self.pending.get(&chunk.id)?;
                    Some(p.consumed + chunk.tokens)
                })
                .max()
                .unwrap_or(0);
            match self
                .scheduler
                .costs()
                .estimate(chunk_tokens - decodes, context)
            {
                Some(seconds) => Some(seconds),
                None => return Ok((drafts, sources, decodes)),
            }
        };
        let mut proposed = Vec::with_capacity(decodes);
        for (index, chunk) in chunks.iter().enumerate() {
            if !decoding[index] {
                continue;
            }
            let pending = self
                .pending
                .get_mut(&chunk.id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            if pending.constrained {
                continue;
            }
            // Outputs number drafts + 1 and occupy positions up to consumed + drafts.
            let remaining = pending.max_tokens.saturating_sub(pending.generated());
            let mut cap = self
                .speculation
                .max_draft
                .min(self.speculation.max_pass_drafts)
                .min(room)
                .min(remaining.saturating_sub(1))
                .min(
                    self.max_position
                        .saturating_sub(pending.consumed.saturating_add(1)),
                );
            if let Some(boundary) = self.prefix.next_boundary(pending.consumed) {
                cap = cap.min(boundary.saturating_sub(pending.consumed + 1));
            }
            let eos = &self.defaults.eos_tokens;
            let mut proposal: Vec<u32> = pending
                .ngrams
                .propose(&pending.tokens, &self.speculation, cap)
                .into_iter()
                .take_while(|token| !eos.contains(token))
                .collect();
            let mut source = DraftSource::Lookup;
            let mut acceptance = pending.acceptance;
            pending.mtp_verifying = None;
            if let Some(draft) = pending
                .mtp_draft
                .filter(|draft| !eos.contains(&draft.token))
                .filter(|_| cap > 0 && pending.mtp_done == pending.consumed)
            {
                let head =
                    self.calibration[usize::from(pending.greedy)].acceptance(draft.confidence);
                // The head's draft replaces a lookup proposal whose acceptance is lower.
                if proposal.is_empty() || head.rate() > acceptance.rate() {
                    (proposal, source, acceptance) = (vec![draft.token], DraftSource::Mtp, head);
                    pending.mtp_verifying = Some(draft.confidence);
                }
            }
            if !proposal.is_empty() {
                proposed.push((index, proposal, acceptance, source));
            }
        }
        if proposed.is_empty() {
            return Ok((drafts, sources, decodes));
        }
        let proposals: Vec<Proposal> = proposed
            .iter()
            .map(|(_, tokens, acceptance, _)| Proposal {
                tokens: tokens.len(),
                acceptance: *acceptance,
            })
            .collect();
        let limit = self.speculation.max_pass_drafts.min(room);
        let counts = speculation::plan(&proposals, decodes, &self.draft_costs, limit, mixed);
        let usage = self.engine.memory()?;
        let mut demand = Demand::default();
        let mut plain = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let handle = self
                .pending
                .get(&chunk.id)
                .ok_or_else(|| invalid("missing pipeline request"))?
                .handle;
            let one = self.engine.demand(handle, chunk.tokens, false)?;
            demand = demand + one;
            plain.push(one);
        }
        for ((index, proposal, _, source), &count) in proposed.into_iter().zip(&counts) {
            let pending = self
                .pending
                .get_mut(&chunks[index].id)
                .ok_or_else(|| invalid("missing pipeline request"))?;
            let mut count = count;
            while count > 0 {
                let verify = self.engine.demand(pending.handle, count + 1, true)?;
                // The verified draft replaces this sequence's plain demand.
                let needed = demand - plain[index] + verify;
                if needed.fits(&usage) {
                    demand = needed;
                    break;
                }
                count -= 1;
            }
            if count == 0 {
                match source {
                    DraftSource::Lookup => pending.acceptance.recover(self.acceptance),
                    DraftSource::Mtp => pending.mtp_acceptance.recover(Acceptance::default()),
                }
            }
            drafts[index] = proposal.get(..count).unwrap_or_default().to_vec();
            sources[index] = source;
        }
        Ok((drafts, sources, decodes))
    }

    /// Whether the head's drafts, at these expected acceptance rates, repay its projection
    /// for the whole next pass. Beyond the measured beneficial range its drafts are not
    /// computed; its keys and values still are, so drafting resumes when fewer sequences
    /// decode. A losing head is retried after exponentially growing skips.
    fn head_drafts_pay(&mut self, rates: &[f64]) -> bool {
        let decodes = self
            .pending
            .values()
            .filter(|p| p.running && p.tokens.len() == p.consumed + 1)
            .count();
        if decodes > speculation::MTP_MAX_DECODES {
            return false;
        }
        if rates.is_empty() {
            return true;
        }
        if self
            .head_costs
            .worthwhile(rates, decodes, &self.draft_costs)
        {
            (self.head_skips, self.head_explore) = (0, speculation::HEAD_EXPLORE_PASSES);
            return true;
        }
        if self.head_skips >= self.head_explore {
            self.head_skips = 0;
            self.head_explore = (self.head_explore * 2).min(speculation::HEAD_EXPLORE_MAX_PASSES);
            return true;
        }
        self.head_skips += 1;
        false
    }

    /// Run the draft head over each published request's newly committed positions, pairing
    /// each with the token that followed it, and keep the head's draft for a request whose
    /// next pass decodes, unless prompt lookup proposes with higher acceptance. A request whose head falls behind (it failed, or a position was
    /// published without hidden states) stops drafting from the head until it recomputes.
    pub(super) fn draft_head(&mut self, heads: Vec<(Uuid, usize, Tensor)>) -> Result<Duration> {
        let mut requests = Vec::with_capacity(heads.len());
        let mut ids = Vec::with_capacity(heads.len());
        let mut rates = Vec::new();
        for (id, before, hidden) in heads {
            let Some(pending) = self.pending.get_mut(&id) else {
                continue;
            };
            pending.mtp_draft = None;
            let Some(tokens) = pending.tokens.get(before + 1..=pending.consumed) else {
                continue;
            };
            if pending.mtp_done != before || tokens.len() != hidden.dim(0)? {
                continue;
            }
            // Its LM-head projection costs about a third of a small model's decode: compute
            // a draft only when the next pass decodes, prompt lookup has no proposal or a less
            // accepted one, and the head's acceptance can repay it.
            let mut draft = pending.tokens.len() == pending.consumed + 1;
            if draft && pending.mtp_acceptance.rate() < speculation::MTP_MIN_RATE {
                // Towards the neutral prior, not the pipeline's rate, which may itself sit
                // below the threshold: a skipped head is retried after a few passes.
                pending.mtp_acceptance.recover(Acceptance::default());
                draft = false;
            }
            draft = draft
                && (pending.mtp_acceptance.rate() > pending.acceptance.rate()
                    || pending
                        .ngrams
                        .propose(&pending.tokens, &self.speculation, 1)
                        .is_empty());
            if draft {
                rates.push(pending.mtp_acceptance.rate());
            }
            requests.push(DraftRequest {
                handle: pending.handle,
                start: before,
                tokens: tokens.to_vec(),
                hidden,
                draft,
            });
            ids.push(id);
        }
        if requests.is_empty() {
            return Ok(Duration::ZERO);
        }
        if !self.head_drafts_pay(&rates) {
            for request in &mut requests {
                request.draft = false;
            }
        }
        let drafts = requests.iter().filter(|r| r.draft).count();
        // Decode-sized: at most the drafted tokens plus one per sequence.
        let rows: usize = requests.iter().map(|r| r.tokens.len()).sum();
        let decode_sized = rows <= requests.len() * (self.speculation.max_draft + 1);
        let started = Instant::now();
        let outcomes = self.engine.draft(&requests)?;
        let elapsed = started.elapsed();
        if decode_sized {
            self.head_costs.record(drafts, elapsed.as_secs_f64());
        }
        for (id, outcome) in ids.into_iter().zip(outcomes) {
            let Some(pending) = self.pending.get_mut(&id) else {
                continue;
            };
            match outcome {
                Ok(draft) => {
                    pending.mtp_done = pending.consumed;
                    pending.mtp_draft = draft;
                    // Whether the head is worth running for a sampled request follows the
                    // expected acceptance of its drafts. Its verification outcomes also count
                    // the unlikely drafts planning chose to verify, whose rejections stopped
                    // the head computing its likely ones; greedy requests, whose drafts are
                    // rarely unlikely, did better on verification outcomes.
                    if let Some(draft) = draft.filter(|_| !pending.greedy) {
                        let expected = self.calibration[0].acceptance(draft.confidence).rate();
                        pending.mtp_acceptance.expect(expected);
                        self.mtp_acceptance.expect(expected);
                    }
                }
                Err(error) => tracing::debug!(request = %id, "draft head failed: {error}"),
            }
        }
        Ok(elapsed)
    }
}
