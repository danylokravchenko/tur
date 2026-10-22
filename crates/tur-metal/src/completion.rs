#![forbid(unsafe_code)]
//! Completion ownership around Candle's safe device synchronization.
//!
//! The fence is `Device::synchronize`, which waits for all work submitted through that
//! device handle and its clones. Engines that need independent completion must own
//! separate devices.

use candle_core::Device;
use parking_lot::Mutex;
use std::sync::Arc;
use tur_core::Result;

#[derive(Clone)]
enum Fence {
    Device(Device),
    #[cfg(test)]
    Controlled(Arc<std::sync::atomic::AtomicUsize>),
}

impl Fence {
    fn wait(&self) -> Result<()> {
        match self {
            Self::Device(device) => Ok(device.synchronize()?),
            #[cfg(test)]
            Self::Controlled(failures) => {
                use std::sync::atomic::Ordering;
                if failures
                    .try_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
                        count.checked_sub(1)
                    })
                    .is_ok()
                {
                    Err(tur_core::TurError::Other("injected fence failure".into()))
                } else {
                    Ok(())
                }
            }
        }
    }
}

struct Retired {
    fence: Fence,
    _resources: Box<dyn Send>,
}

/// Resources whose completion fence failed on drop. A device error must never return
/// possibly in-flight resources to an allocator, so they wait here for a successful retry.
/// Each execution owner has its own quarantine; one device failure cannot block other owners.
#[derive(Clone, Default)]
pub struct Quarantine(Arc<Mutex<Vec<Retired>>>);

impl Quarantine {
    /// Create an empty quarantine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retry every retained fence and release completed resources; returns entries still held.
    pub fn retry(&self) -> usize {
        let entries = std::mem::take(&mut *self.0.lock());
        // Device waits and arbitrary resource destructors run outside the lock.
        let pending: Vec<_> = entries
            .into_iter()
            .filter(|entry| entry.fence.wait().is_err())
            .collect();
        let mut retired = self.0.lock();
        retired.extend(pending);
        retired.len()
    }
}

/// Retains weights, state, outputs and scratch until a successful completion fence.
/// Dropping an unfinished completion waits; a failed fence quarantines its resources.
pub struct Completion<T: Send + 'static> {
    fence: Fence,
    quarantine: Quarantine,
    resources: Option<T>,
    completed: bool,
}

impl<T: Send + 'static> Completion<T> {
    /// Begin ownership before encoding any work that references these resources.
    pub fn new(device: Device, quarantine: &Quarantine, resources: T) -> Self {
        Self {
            fence: Fence::Device(device),
            quarantine: quarantine.clone(),
            resources: Some(resources),
            completed: false,
        }
    }

    /// Access retained resources while encoding. The caller must keep every live binding here.
    pub fn resources_mut(&mut self) -> Option<&mut T> {
        self.completed = false;
        self.resources.as_mut()
    }

    /// Wait for device completion. On error retain everything for a later retry.
    pub fn wait(&mut self) -> Result<()> {
        self.completed = false;
        self.fence.wait()?;
        self.completed = true;
        Ok(())
    }

    /// Transfer completed resources; pending work cannot be extracted.
    pub fn take(&mut self) -> Option<T> {
        if self.completed {
            self.resources.take()
        } else {
            None
        }
    }
}

impl<T: Send + 'static> Drop for Completion<T> {
    fn drop(&mut self) {
        if !self.completed
            && self.fence.wait().is_err()
            && let Some(resources) = self.resources.take()
        {
            self.quarantine.0.lock().push(Retired {
                fence: self.fence.clone(),
                _resources: Box::new(resources),
            });
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn controlled<T: Send + 'static>(
        failures: usize,
        quarantine: &Quarantine,
        resources: T,
    ) -> Completion<T> {
        Completion {
            fence: Fence::Controlled(Arc::new(AtomicUsize::new(failures))),
            quarantine: quarantine.clone(),
            resources: Some(resources),
            completed: false,
        }
    }

    #[test]
    fn test_completion_retains_resources_until_fence() -> Result<()> {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut completion = Completion::new(
            Device::Cpu,
            &Quarantine::new(),
            Resource(Arc::clone(&drops)),
        );
        assert!(completion.take().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        completion.wait()?;
        drop(completion.take());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn test_dropped_completion_releases_after_wait() {
        let drops = Arc::new(AtomicUsize::new(0));
        drop(Completion::new(
            Device::Cpu,
            &Quarantine::new(),
            Resource(Arc::clone(&drops)),
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_failed_wait_keeps_resources_until_successful_retry() -> Result<()> {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut completion = controlled(1, &Quarantine::new(), Resource(Arc::clone(&drops)));
        assert!(completion.wait().is_err());
        assert!(completion.take().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        completion.wait()?;
        drop(completion.take());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn test_failed_drop_quarantines_resources_until_fence_retry() {
        let drops = Arc::new(AtomicUsize::new(0));
        let quarantine = Quarantine::new();
        drop(controlled(2, &quarantine, Resource(Arc::clone(&drops))));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(quarantine.retry(), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(quarantine.retry(), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_quarantine_is_isolated_per_owner() {
        let failed = Quarantine::new();
        let healthy = Quarantine::new();
        drop(controlled(usize::MAX, &failed, 0u8));
        assert_eq!(failed.retry(), 1);
        assert_eq!(healthy.retry(), 0);
    }

    #[test]
    fn test_mutation_after_fence_requires_another_wait() -> Result<()> {
        let mut completion = Completion::new(Device::Cpu, &Quarantine::new(), 1u32);
        completion.wait()?;
        if let Some(value) = completion.resources_mut() {
            *value = 2;
        }
        assert!(completion.take().is_none());
        completion.wait()?;
        assert_eq!(completion.take(), Some(2));
        Ok(())
    }
}
