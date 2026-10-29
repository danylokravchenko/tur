#![forbid(unsafe_code)]
//! Resident memory planning for one model on one device.

use candle_core::Device;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tur_core::{
    Result, TurError,
    execution::{ModelDescription, invalid},
};
use tur_metal::pool::{PoolConfig, working_set_bytes};

/// Context each concurrent sequence is sized for when no state budget is given.
pub const DEFAULT_CONTEXT_TOKENS: usize = 8192;

/// Device memory of one resident model:
///
/// ```text
/// resident = packed weights + attention KV + recurrent state
///          + retained checkpoints + activation/scratch peak
///          + speculative state + in-flight buffers + runtime overhead
/// ```
///
/// Attention K/V, recurrent state, retained checkpoints and speculative boundaries all
/// live in one state pool (pages and slots), so `state_pool` bounds their sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryPlan {
    pub weights: usize,
    pub state_pool: usize,
    /// Kernel scratch of the largest planned pass.
    pub scratch_peak: usize,
    /// Packed inputs and logits of the largest pass, plus bounded host proposal indices.
    pub in_flight: usize,
    /// Reserve for the runtime, allocator rounding and the OS.
    pub overhead: usize,
    /// Device limit the plan fits in, if the device reports one.
    pub limit: Option<usize>,
    /// Bytes other resident models had leased from the device budget when planning.
    pub reserved: usize,
    /// Snapshot slots the pool holds beyond live sequences, after the share cap.
    pub retained: usize,
    /// Recurrent slots held beyond live sequences and snapshots for speculative drafts.
    pub draft_slots: usize,
}

/// What the plan must hold besides the state pool.
#[derive(Clone, Copy, Debug)]
pub struct PlanInputs {
    pub weights: usize,
    pub scratch_peak: usize,
    pub in_flight: usize,
    /// Concurrent sequences, each needing a committed and an in-flight recurrent slot.
    pub sequences: usize,
    /// Recurrent snapshots retained beyond live sequences, one slot each (prefix reuse).
    pub retained: usize,
    /// Most of the state pool's bytes that retained snapshots may take; `None` keeps
    /// `retained` as given.
    pub retained_share: Option<f64>,
    /// Recurrent slots for the per-token boundaries of speculative drafts, one per
    /// drafted token, so one sequence can verify its longest draft beside full snapshots.
    pub draft_slots: usize,
    /// Most tokens one sequence adds in a pass (a prefill chunk or a verified draft):
    /// windowed layers hold their window plus these pages.
    pub pass_tokens: usize,
    /// Explicit state pool budget in bytes; otherwise sized for [`DEFAULT_CONTEXT_TOKENS`]
    /// per sequence within the device limit.
    pub state_bytes: Option<usize>,
    /// The device's recommended working set.
    pub limit: Option<usize>,
    /// Bytes already leased by other models on the device.
    pub reserved: usize,
}

impl MemoryPlan {
    /// Total planned resident bytes.
    pub fn resident(&self) -> usize {
        self.weights
            .saturating_add(self.state_pool)
            .saturating_add(self.scratch_peak)
            .saturating_add(self.in_flight)
            .saturating_add(self.overhead)
    }

    /// Size the state pool and check the plan, beside memory other models reserved, against
    /// the device limit. An explicit budget that would exceed the limit is rejected rather
    /// than left to swap.
    pub fn new(description: &ModelDescription, inputs: PlanInputs) -> Result<(Self, PoolConfig)> {
        let overhead = inputs.limit.map_or(0, |limit| limit / 10);
        let fixed = inputs
            .weights
            .checked_add(inputs.scratch_peak)
            .and_then(|b| b.checked_add(inputs.in_flight))
            .and_then(|b| b.checked_add(overhead))
            .and_then(|b| b.checked_add(inputs.reserved))
            .ok_or_else(|| invalid("memory plan overflow"))?;
        let available = inputs.limit.map(|limit| limit.saturating_sub(fixed));
        let bytes = if let Some(bytes) = inputs.state_bytes {
            bytes
        } else {
            let context = DEFAULT_CONTEXT_TOKENS.min(description.max_position);
            let window_pages = match description.window() {
                Some(window) => inputs
                    .sequences
                    .checked_add(inputs.retained)
                    .and_then(|holders| {
                        let held = window.checked_add(inputs.pass_tokens)?;
                        holders.checked_mul(description.pages_for(held).checked_add(1)?)
                    })
                    .ok_or_else(|| invalid("window page count overflow"))?,
                None => 0,
            };
            let wanted = PoolConfig {
                pages: description
                    .full_pages_for(context)
                    .checked_mul(inputs.sequences)
                    .ok_or_else(|| invalid("page count overflow"))?,
                window_pages,
                recurrent_slots: inputs
                    .sequences
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_add(inputs.retained))
                    .and_then(|slots| slots.checked_add(inputs.draft_slots))
                    .ok_or_else(|| invalid("recurrent slot count overflow"))?,
            }
            .bytes(description)?;
            available.map_or(wanted, |available| wanted.min(available))
        };
        // A snapshot costs a whole recurrent state (157 MB on Qwen3.8-27B against 20 MB on
        // 0.8B), so large models on small budgets retain proportionally fewer.
        let retained = match inputs.retained_share {
            Some(share) if description.has_recurrent() => {
                let slot = description.recurrent_slot_bytes()?.max(1);
                inputs
                    .retained
                    .min((bytes as f64 * share.clamp(0.0, 1.0)) as usize / slot)
            }
            _ => inputs.retained,
        };
        let draft_slots = if description.has_recurrent() {
            inputs.draft_slots
        } else {
            0
        };
        let extra = retained
            .checked_add(draft_slots)
            .ok_or_else(|| invalid("recurrent slot count overflow"))?;
        let config = PoolConfig::for_budget(
            description,
            bytes,
            inputs.sequences,
            extra,
            inputs.pass_tokens,
        )?;
        let plan = Self {
            weights: inputs.weights,
            state_pool: config.bytes(description)?,
            scratch_peak: inputs.scratch_peak,
            in_flight: inputs.in_flight,
            overhead,
            limit: inputs.limit,
            reserved: inputs.reserved,
            retained,
            draft_slots,
        };
        if let Some(limit) = inputs.limit
            && plan.resident().saturating_add(plan.reserved) > limit
        {
            return Err(TurError::MemoryAllocation(format!(
                "planned resident memory {} bytes beside {} bytes of other models exceeds the \
                 device working set of {limit} bytes (state pool {}, weights {})",
                plan.resident(),
                plan.reserved,
                plan.state_pool,
                plan.weights
            )));
        }
        Ok((plan, config))
    }
}

#[derive(Debug)]
struct Ledger {
    limit: Option<usize>,
    leased: AtomicUsize,
}

/// Device memory shared by every model resident on one device. Each pipeline plans
/// against what the others left and leases its planned resident bytes, so two models
/// cannot both promise the same free memory. Cloning shares the budget.
#[derive(Clone, Debug)]
pub struct DeviceBudget(Arc<Ledger>);

/// Bytes leased from a [`DeviceBudget`], returned when dropped.
#[derive(Debug)]
pub struct BudgetLease {
    ledger: Arc<Ledger>,
    bytes: usize,
}

impl Drop for BudgetLease {
    fn drop(&mut self) {
        self.ledger.leased.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

impl BudgetLease {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl DeviceBudget {
    /// A budget of `limit` bytes; `None` is unbounded (the CPU reference).
    pub fn new(limit: Option<usize>) -> Self {
        Self(Arc::new(Ledger {
            limit,
            leased: AtomicUsize::new(0),
        }))
    }

    /// The device's recommended working set.
    pub fn for_device(device: &Device) -> Self {
        Self::new(working_set_bytes(device))
    }

    pub fn limit(&self) -> Option<usize> {
        self.0.limit
    }

    /// Bytes currently leased by resident models.
    pub fn leased(&self) -> usize {
        self.0.leased.load(Ordering::Relaxed)
    }

    /// Lease `bytes`, or fail without leasing if they exceed what remains. The counter
    /// guards no other data, so relaxed ordering suffices.
    pub fn lease(&self, bytes: usize) -> Result<BudgetLease> {
        let limit = self.0.limit;
        self.0
            .leased
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |leased| {
                leased
                    .checked_add(bytes)
                    .filter(|total| limit.is_none_or(|limit| *total <= limit))
            })
            .map_err(|leased| {
                TurError::MemoryAllocation(format!(
                    "{bytes} bytes exceed the {} bytes left of the device budget",
                    limit.map_or(0, |limit| limit.saturating_sub(leased))
                ))
            })?;
        Ok(BudgetLease {
            ledger: Arc::clone(&self.0),
            bytes,
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use candle_core::DType;
    use tur_core::execution::{KvFormat, KvLayout, LayerStateKind, Modality, ModelIdentity};

    fn description() -> ModelDescription {
        ModelDescription {
            identity: ModelIdentity {
                revision: "plan-fixture".into(),
                state_format: 1,
            },
            layers: vec![
                LayerStateKind::Attention {
                    kv_heads: 1,
                    head_dim: 32,
                    window: None,
                },
                LayerStateKind::Recurrent {
                    conv_channels: 4,
                    conv_width: 2,
                    heads: 1,
                    key_dim: 4,
                    value_dim: 4,
                    dtype: DType::F32,
                },
            ],
            kv: KvLayout::new(16, KvFormat::F32),
            modalities: vec![Modality::Text],
            vocab_size: 8,
            max_position: 1 << 20,
        }
    }

    fn inputs(state_bytes: Option<usize>, limit: Option<usize>) -> PlanInputs {
        PlanInputs {
            weights: 1000,
            scratch_peak: 500,
            in_flight: 100,
            sequences: 2,
            retained: 0,
            retained_share: None,
            draft_slots: 0,
            pass_tokens: 1,
            state_bytes,
            limit,
            reserved: 0,
        }
    }

    #[test]
    fn test_default_budget_sizes_contexts_and_fits_the_device_limit() -> Result<()> {
        let model = description();
        let page = model.page_bytes()?;
        let slot = model.recurrent_slot_bytes()?;
        // Unlimited (CPU): two default contexts of pages plus four recurrent slots.
        let (plan, config) = MemoryPlan::new(&model, inputs(None, None))?;
        assert_eq!(config.pages, 2 * DEFAULT_CONTEXT_TOKENS / 16);
        assert_eq!(config.recurrent_slots, 4);
        assert_eq!(plan.state_pool, config.pages * page + 5 * slot);
        assert_eq!(plan.resident(), 1600 + plan.state_pool);
        // Retained snapshots add one recurrent slot each.
        let retained = PlanInputs {
            retained: 3,
            ..inputs(None, None)
        };
        let (plan, with_snapshots) = MemoryPlan::new(&model, retained)?;
        assert_eq!((with_snapshots.recurrent_slots, plan.retained), (7, 3));
        // Draft slots add one each, beside snapshots and outside their share cap.
        let drafts = PlanInputs {
            draft_slots: 4,
            retained_share: Some(0.0),
            ..retained
        };
        let (plan, with_drafts) = MemoryPlan::new(&model, drafts)?;
        assert_eq!(
            (with_drafts.recurrent_slots, plan.retained, plan.draft_slots),
            (8, 0, 4)
        );
        // A share cap keeps snapshots to that fraction of the pool's bytes.
        let budget = 4 * page + 40 * slot;
        let cap = budget / 8 / slot;
        let capped = PlanInputs {
            retained: cap + 10,
            retained_share: Some(0.125),
            ..inputs(Some(budget), None)
        };
        let (plan, config) = MemoryPlan::new(&model, capped)?;
        assert_eq!((plan.retained, config.recurrent_slots), (cap, 4 + cap));
        assert!(plan.retained * slot <= budget / 8);
        // A device limit shrinks the pool to what remains after weights, scratch,
        // in-flight buffers and a tenth for overhead.
        let limit = 1600 + 10 * page + 5 * slot;
        let limit = limit + limit / 9;
        let (plan, config) = MemoryPlan::new(&model, inputs(None, Some(limit)))?;
        assert!(config.pages >= 9 && config.pages <= 10, "{config:?}");
        assert!(plan.resident() <= limit);
        Ok(())
    }

    #[test]
    fn test_explicit_budget_beyond_the_device_limit_is_rejected() -> Result<()> {
        let model = description();
        let (plan, _) = MemoryPlan::new(&model, inputs(Some(1 << 20), None))?;
        assert!(plan.state_pool <= 1 << 20);
        assert!(matches!(
            MemoryPlan::new(&model, inputs(Some(1 << 20), Some(1 << 20))),
            Err(TurError::MemoryAllocation(_))
        ));
        assert!(MemoryPlan::new(&model, inputs(Some(0), None)).is_err());
        Ok(())
    }

    #[test]
    fn test_budget_boundaries_fit_without_overallocating_the_zero_slot() -> Result<()> {
        let model = description();
        let page = model.page_bytes()?;
        let slot = model.recurrent_slot_bytes()?;
        let minimum = page + 5 * slot;
        assert!(MemoryPlan::new(&model, inputs(Some(minimum - 1), None)).is_err());
        for budget in [minimum, minimum + page - 1, minimum + page] {
            let (plan, _) = MemoryPlan::new(&model, inputs(Some(budget), None))?;
            assert!(plan.state_pool <= budget);
        }
        // Leave exactly ten pages plus usable slots after fixed allocations and overhead.
        // The zero slot must reduce page capacity instead of exceeding the working set.
        let limit = (1600 + 10 * page + 4 * slot) * 10 / 9;
        let (plan, config) = MemoryPlan::new(&model, inputs(None, Some(limit)))?;
        assert_eq!(config.pages, 9);
        assert!(plan.resident() <= limit);
        let mut overflow = inputs(None, None);
        overflow.sequences = usize::MAX;
        assert!(MemoryPlan::new(&model, overflow).is_err());
        Ok(())
    }

    #[test]
    fn test_models_sharing_a_device_budget_cannot_overcommit() -> Result<()> {
        let model = description();
        let page = model.page_bytes()?;
        let slot = model.recurrent_slot_bytes()?;
        let limit = (1600 + 20 * page + 5 * slot) * 10 / 9;
        let budget = DeviceBudget::new(Some(limit));
        // The first model sizes its pool into the whole budget and leases it.
        let (first, _) = MemoryPlan::new(
            &model,
            PlanInputs {
                reserved: budget.leased(),
                ..inputs(None, budget.limit())
            },
        )?;
        let lease = budget.lease(first.resident())?;
        // A second plan sees what is left: too little for its fixed allocations.
        assert!(matches!(
            MemoryPlan::new(
                &model,
                PlanInputs {
                    reserved: budget.leased(),
                    ..inputs(None, budget.limit())
                },
            ),
            Err(TurError::MemoryAllocation(_))
        ));
        assert!(budget.lease(limit - lease.bytes() + 1).is_err());
        assert_eq!(budget.leased(), lease.bytes());
        // Unloading the first model returns its memory.
        drop(lease);
        assert_eq!(budget.leased(), 0);
        let (second, _) = MemoryPlan::new(
            &model,
            PlanInputs {
                reserved: budget.leased(),
                ..inputs(None, budget.limit())
            },
        )?;
        assert_eq!(second, first);
        let unbounded = DeviceBudget::new(None);
        assert!(unbounded.lease(usize::MAX).is_ok());
        assert!(unbounded.lease(usize::MAX).is_ok());
        Ok(())
    }

    #[test]
    fn test_concurrent_leases_never_exceed_the_limit() -> Result<()> {
        let budget = DeviceBudget::new(Some(1000));
        let leases: Vec<BudgetLease> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    let budget = budget.clone();
                    scope.spawn(move || {
                        (0..100)
                            .filter_map(|_| budget.lease(7).ok())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let leases = workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .map_err(|_| invalid("budget lease worker panicked"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok::<Vec<BudgetLease>, TurError>(leases.into_iter().flatten().collect())
        })?;
        assert_eq!(leases.len(), 142);
        assert_eq!(budget.leased(), 994);
        drop(leases);
        assert_eq!(budget.leased(), 0);
        Ok(())
    }
}
