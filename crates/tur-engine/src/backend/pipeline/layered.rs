//! Layered prefill: a prefill chunk beside decodes runs a group of layers at a time, with a
//! decode pass after each, so a stream's gap is one group rather than the whole chunk. The
//! chunk still commits once, at its ordinary boundary, so snapshots and recomputation see
//! the same passes as ever.

use super::super::scheduler::Chunk;
use super::InferencePipeline;
use crate::{Result, hybrid::HybridModel};
use uuid::Uuid;

/// Prompt tokens per layer group. On gpt-oss-20b behind a 1 GiB cache, a 1,328-token prompt
/// beside a stream cut its worst gap from 14.4 s to 5.7 s, for 14% more prefill time.
const GROUP_TOKENS: usize = 256;
/// Shorter chunks run beside decodes as usual: they stall nothing worth a second read of
/// the experts that decodes evict between groups.
const MIN_TOKENS: usize = 128;

pub(super) struct LayeredChunk {
    chunk: Chunk,
    /// Whether the request held state before this chunk, restored if it is abandoned.
    was_running: bool,
    groups: usize,
    /// Groups already run.
    done: usize,
}

impl LayeredChunk {
    pub(super) fn id(&self) -> Uuid {
        self.chunk.id
    }
}

impl InferencePipeline {
    /// Split the selected pass's prefill by layers while decodes run. Returns the chunks to
    /// pack, and whether they are a paused chunk's last group, submitted alone because its
    /// rows join no other request's layers. Other prefill waits for a paused chunk.
    pub(super) fn layer_prefill(&mut self, chunks: Vec<Chunk>) -> Result<(Vec<Chunk>, bool)> {
        let layers = self.model.pausable_layers();
        let Some(layers) = layers.filter(|_| self.scheduler.layered_prefill()) else {
            return Ok((chunks, false));
        };
        let decoding = |chunk: &&Chunk| {
            self.pending
                .get(&chunk.id)
                .is_some_and(super::Pending::decoding)
        };
        let decodes: Vec<Chunk> = chunks.iter().filter(decoding).copied().collect();
        // A chunk's last group ran without them, and the next chunk's first must not follow.
        if std::mem::take(&mut self.decode_next) && !decodes.is_empty() {
            return Ok((decodes, false));
        }
        if self.layered.is_none() {
            let long = chunks
                .iter()
                .find(|chunk| !decodes.contains(chunk) && chunk.tokens >= MIN_TOKENS)
                .copied();
            let Some(chunk) = long.filter(|_| !decodes.is_empty()) else {
                return Ok((chunks, false));
            };
            self.start_layered(chunk, layers)?;
        }
        let Some(layered) = self.layered.as_ref() else {
            return Ok((chunks, false));
        };
        if decodes.is_empty() || layered.done + 1 >= layered.groups {
            let chunk = layered.chunk;
            self.layered = None;
            self.decode_next = true;
            return Ok((vec![chunk], true));
        }
        let end = layers * (layered.done + 1) / layered.groups;
        self.engine.advance_layered(end)?;
        if let Some(layered) = self.layered.as_mut() {
            layered.done += 1;
        }
        Ok((decodes, false))
    }

    fn start_layered(&mut self, chunk: Chunk, layers: usize) -> Result<()> {
        let (plan, _) = self.pack(&[chunk], &[Vec::new()])?;
        self.engine.begin_layered(plan)?;
        let pending = self
            .pending
            .get_mut(&chunk.id)
            .ok_or_else(|| tur_core::execution::invalid("missing pipeline request"))?;
        let was_running = pending.running;
        // It holds pages now: no restore or admission may treat it as waiting.
        pending.running = true;
        self.layered = Some(LayeredChunk {
            chunk,
            was_running,
            groups: (chunk.tokens / GROUP_TOKENS).clamp(2, layers.max(2)),
            done: 0,
        });
        Ok(())
    }

    /// Give up the paused chunk for its pages; it runs again later. Whether there was one.
    pub(super) fn abort_layered(&mut self) -> bool {
        let Some(layered) = self.layered.take() else {
            return false;
        };
        self.engine.abort_layered();
        if let Some(pending) = self.pending.get_mut(&layered.chunk.id) {
            pending.running = layered.was_running;
        }
        true
    }
}
