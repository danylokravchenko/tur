//! Loading, effective configuration and the pre-load memory reservation.

use super::super::{
    memory::{DeviceBudget, MemoryPlan, PlanInputs},
    prefix::{PrefixCache, PrefixConfig},
    scheduler::{Scheduler, SchedulerConfig},
    speculation::{
        self, Acceptance, Calibration, DraftCosts, HeadCosts, SpeculationConfig, SpeculationStats,
    },
};
use super::{BatchStepOutput, InferencePipeline};
use crate::{Result, hybrid::HybridEngine};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tur_core::execution::invalid;
use tur_models::{backend::factory::ModelFactory, models::qwen3_5::Qwen35};

/// Scheduling limits and the state memory budget of one pipeline.
#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    pub scheduler: SchedulerConfig,
    /// Prefix snapshots: spacing and how many are retained beside running requests.
    pub prefix: PrefixConfig,
    /// Penalize and sample unconstrained rows on the device (when the model runs on Metal
    /// kernels); otherwise every full row is copied to the host sampler.
    pub device_sampling: bool,
    /// Opt-in throughput mode using drafts from each request's own history. Tokens arrive
    /// in bursts; longer pauses can increase p95 inter-token latency. Disabled by default.
    pub speculation: SpeculationConfig,
    /// Bytes for K/V pages and recurrent state. `None` sizes the pool for
    /// `max_running` contexts of [`super::super::memory::DEFAULT_CONTEXT_TOKENS`] tokens, within
    /// what the device budget leaves.
    pub state_bytes: Option<usize>,
}

impl PipelineConfig {
    /// Default scheduling for `max_batch_size` running sequences and `prefill_chunk_size`
    /// prefill chunks, one retained prefix snapshot per running sequence, and a derived
    /// state budget.
    pub fn new(max_batch_size: usize, prefill_chunk_size: usize) -> Self {
        Self {
            scheduler: SchedulerConfig::new(max_batch_size, prefill_chunk_size),
            prefix: PrefixConfig {
                spacing: PrefixConfig::DEFAULT_SPACING,
                entries: max_batch_size,
                state_share: Some(PrefixConfig::DEFAULT_STATE_SHARE),
            },
            device_sampling: true,
            speculation: SpeculationConfig::DISABLED,
            state_bytes: None,
        }
    }

    /// Reproducibility mode: one request at a time in fixed `prefill_chunk_size` chunks,
    /// without prefix reuse or speculation, so a request's computation (and with its seed,
    /// its tokens) does not depend on other traffic. Batched passes can change logits in
    /// their last bits through batch-dependent kernels; this mode trades throughput to rule
    /// that out.
    pub fn reproducible(prefill_chunk_size: usize) -> Self {
        Self {
            prefix: PrefixConfig::DISABLED,
            speculation: SpeculationConfig::DISABLED,
            ..Self::new(1, prefill_chunk_size)
        }
    }

    fn resolve(mut self, recurrent: candle_core::DType) -> Result<Self> {
        self.speculation.validate()?;
        // Narrow state rounds at ordinary pass boundaries; verification keeps F32
        // intermediates between drafts, so it would change the target process.
        if recurrent != candle_core::DType::F32 {
            if self.speculation.enabled() {
                tracing::warn!(
                    ?recurrent,
                    "speculative drafts need F32 recurrent state; using ordinary decoding"
                );
            }
            self.speculation = SpeculationConfig::DISABLED;
        }
        // A snapshot at B cannot hold the head's entry for B - 1: it depends on the
        // token sampled after B, which differs between requests sharing the snapshot.
        if self.speculation.mtp_enabled() && self.prefix.enabled() {
            tracing::info!("native MTP drafts disable prefix reuse");
            self.prefix = PrefixConfig::DISABLED;
        }
        Ok(self)
    }
}

impl InferencePipeline {
    /// Load the model as the only one on its device; see [`Self::with_budget`].
    pub fn new(factory: &ModelFactory, config: PipelineConfig) -> Result<Self> {
        Self::with_budget(factory, config, &DeviceBudget::for_device(&factory.device))
    }

    /// Load the validated model, plan its memory beside what other models leased from
    /// `budget`, and lease the plan before loading weights. Invalid limits and a plan
    /// beyond the remaining budget fail before reading weight payloads or uploading them.
    pub fn with_budget(
        factory: &ModelFactory,
        config: PipelineConfig,
        budget: &DeviceBudget,
    ) -> Result<Self> {
        let scheduler = Scheduler::new(config.scheduler)?;
        let config = config.resolve(factory.recurrent)?;
        let draft = config
            .speculation
            .mtp
            .filter(|_| config.speculation.enabled());
        let limits = config.scheduler;
        let draft_rows = config
            .speculation
            .max_pass_drafts
            .min(
                config
                    .speculation
                    .max_draft
                    .saturating_mul(limits.max_running),
            )
            .min(limits.max_batch_tokens.saturating_sub(1));
        let (model, tokenizer, template, defaults, (plan, pool, lease)) = factory
            .create_model_with(draft, |architecture, options| {
                let tokens = limits.max_batch_tokens;
                let sequences = limits.max_running.min(tokens);
                let (description, weights, scratch_peak) = Qwen35::memory_requirements(
                    architecture,
                    options,
                    tokens,
                    &vec![architecture.max_position(); sequences],
                )?;
                config.prefix.validate(description.kv.page_tokens)?;
                // Full model logits and their gathered copy coexist for sampled requests;
                // every drafted token adds an output row.
                let in_flight = description
                    .vocab_size
                    .checked_mul(8)
                    .and_then(|row| row.checked_add(16))
                    .and_then(|row| sequences.checked_add(draft_rows)?.checked_mul(row))
                    .and_then(|logits| logits.checked_add(tokens.checked_mul(12)?))
                    // Prompt-end snapshots keep one F32 logit row each.
                    .and_then(|bytes| {
                        bytes.checked_add(
                            description
                                .vocab_size
                                .checked_mul(4)?
                                .checked_mul(config.prefix.entries)?,
                        )
                    })
                    .ok_or_else(|| invalid("in-flight size overflow"))?;
                let in_flight = in_flight
                    .checked_add(
                        config
                            .speculation
                            .index_bytes(sequences, architecture.max_position())?,
                    )
                    .ok_or_else(|| invalid("draft index reservation overflow"))?;
                let (plan, pool) = MemoryPlan::new(
                    &description,
                    PlanInputs {
                        weights,
                        scratch_peak,
                        in_flight,
                        sequences: limits.max_running,
                        retained: config.prefix.entries,
                        retained_share: config.prefix.state_share,
                        draft_slots: draft_rows,
                        state_bytes: config.state_bytes,
                        limit: budget.limit(),
                        reserved: budget.leased(),
                    },
                )?;
                let lease = budget.lease(plan.resident())?;
                Ok((plan, pool, lease))
            })?;
        let max_position = model.config().max_position();
        if model.weight_bytes() != plan.weights {
            return Err(invalid(
                "loaded weights disagree with the pre-load memory plan",
            ));
        }
        let mut engine = HybridEngine::new(Arc::new(model), factory.device.clone(), pool)?;
        engine.set_device_sampling(config.device_sampling);
        if config.speculation.mtp_enabled() {
            engine.enable_draft_head();
        }
        Ok(Self {
            engine,
            tokenizer,
            template,
            defaults,
            max_position,
            scheduler,
            plan,
            _lease: lease,
            pending: HashMap::new(),
            prefix: PrefixCache::new(PrefixConfig {
                entries: plan.retained,
                ..config.prefix
            }),
            awaiting: HashSet::new(),
            in_flight: None,
            staged: BatchStepOutput::default(),
            guidance: None,
            speculation: config.speculation,
            draft_costs: DraftCosts::default(),
            head_costs: HeadCosts::default(),
            head_skips: 0,
            head_explore: speculation::HEAD_EXPLORE_PASSES,
            acceptance: Acceptance::default(),
            mtp_acceptance: Acceptance::default(),
            calibration: [Calibration::default(); 2],
            speculation_stats: SpeculationStats::default(),
        })
    }
}
