#![forbid(unsafe_code)]
//! Completion ownership around Candle's safe device synchronization.

use candle_core::Device;
use std::sync::Mutex;
use tur_core::Result;

#[derive(Clone)]
enum Fence {
    Device(Device),
    #[cfg(test)]
    Controlled(std::sync::Arc<std::sync::atomic::AtomicUsize>),
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

// Only failed synchronization on drop uses this queue. A device error must never
// return possibly in-flight resources to an allocator. Retry on the next submission.
static RETIRED: Mutex<Vec<Retired>> = Mutex::new(Vec::new());

/// Retry fences for abandoned submissions. Failed entries remain retained.
pub fn retry_retired() -> usize {
    let entries = {
        let mut retired = match RETIRED.lock() {
            Ok(guard) => guard,
            Err(error) => error.into_inner(),
        };
        std::mem::take(&mut *retired)
    };
    // Device waits and arbitrary resource destructors must run outside the queue lock.
    let pending = entries
        .into_iter()
        .filter(|entry| entry.fence.wait().is_err());
    let pending: Vec<_> = pending.collect();
    let mut retired = match RETIRED.lock() {
        Ok(guard) => guard,
        Err(error) => error.into_inner(),
    };
    retired.extend(pending);
    retired.len()
}

/// Retains weights, state, outputs and scratch until a successful completion fence.
/// Dropping an unfinished completion waits; a failed fence quarantines its resources.
pub struct Completion<T: Send + 'static> {
    fence: Fence,
    resources: Option<T>,
    completed: bool,
}

impl<T: Send + 'static> Completion<T> {
    /// Begin ownership before encoding any work that references these resources.
    pub fn new(device: Device, resources: T) -> Self {
        Self {
            fence: Fence::Device(device),
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
        if !self.completed && self.fence.wait().is_err() {
            if let Some(resources) = self.resources.take() {
                let mut retired = match RETIRED.lock() {
                    Ok(guard) => guard,
                    Err(error) => error.into_inner(),
                };
                retired.push(Retired {
                    fence: self.fence.clone(),
                    _resources: Box::new(resources),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_completion_retains_resources_until_fence() -> Result<()> {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut completion = Completion::new(Device::Cpu, Resource(Arc::clone(&drops)));
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
        drop(Completion::new(Device::Cpu, Resource(Arc::clone(&drops))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_failed_wait_keeps_resources_until_successful_retry() -> Result<()> {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut completion = Completion {
            fence: Fence::Controlled(Arc::new(AtomicUsize::new(1))),
            resources: Some(Resource(Arc::clone(&drops))),
            completed: false,
        };
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
        let failures = Arc::new(AtomicUsize::new(1));
        drop(Completion {
            fence: Fence::Controlled(Arc::clone(&failures)),
            resources: Some(Resource(Arc::clone(&drops))),
            completed: false,
        });
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        retry_retired();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_mutation_after_fence_requires_another_wait() -> Result<()> {
        let mut completion = Completion::new(Device::Cpu, 1u32);
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
