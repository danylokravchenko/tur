#![forbid(unsafe_code)]
//! Completion ownership around Candle's safe device synchronization.
//!
//! The fence is `Device::synchronize`, which waits for all work submitted through that
//! device handle and its clones. Engines that need independent completion must own
//! separate devices.
//!
//! Candle commits a command buffer only every few dozen encoders or at synchronization,
//! so the tail of a pass would wait for the fence. [`Notifier`] runs the fence on its own
//! thread: [`Completion::notify`] hands it over right after encoding, which commits the
//! tail at once, and the owner keeps doing host work until it consumes the completion
//! event in [`Completion::wait`]. Only that thread waits meanwhile; two concurrent waits
//! on one device could otherwise return before the other's work completed. A host copy
//! queued after the work doubles as its fence, so outputs return without another round
//! trip to the device.

use candle_core::{Device, Tensor};
use parking_lot::Mutex;
use std::{
    sync::{Arc, mpsc},
    time::Instant,
};
use tur_core::{Result, TurError};

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
                    Err(TurError::Other("injected fence failure".into()))
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

/// Completion time and the host copy of the requested output.
type Done = Result<(Instant, Option<Vec<f32>>)>;
type Event = mpsc::Receiver<Done>;
type Request = (Fence, Option<Tensor>, mpsc::SyncSender<Done>);

/// A thread that waits for device completion on request and reports when it happened.
/// One per execution owner; it exits when the owner drops it.
pub struct Notifier {
    requests: mpsc::Sender<Request>,
}

/// Copy `output` to the host (which waits for every earlier command), then fence.
fn complete(fence: &Fence, output: Option<Tensor>) -> Done {
    let copied = output
        .map(|tensor| -> Result<Vec<f32>> { Ok(tensor.flatten_all()?.to_vec1::<f32>()?) })
        .transpose()?;
    fence.wait()?;
    Ok((Instant::now(), copied))
}

impl Notifier {
    pub fn new() -> Result<Self> {
        let (requests, rx) = mpsc::channel::<Request>();
        std::thread::Builder::new()
            .name("tur-fence".into())
            .spawn(move || {
                for (fence, output, reply) in rx {
                    // Fails only when the requester stopped waiting; nobody needs the result.
                    drop(reply.send(complete(&fence, output)));
                }
            })?;
        Ok(Self { requests })
    }

    fn start(&self, fence: Fence, output: Option<Tensor>) -> Result<Event> {
        let (reply, event) = mpsc::sync_channel(1);
        self.requests
            .send((fence, output, reply))
            .map_err(|_| TurError::Other("completion notifier stopped".into()))?;
        Ok(event)
    }
}

/// Retains weights, state, outputs and scratch until a successful completion fence.
/// Dropping an unfinished completion waits; a failed fence quarantines its resources.
pub struct Completion<T: Send + 'static> {
    fence: Fence,
    quarantine: Quarantine,
    resources: Option<T>,
    completed: bool,
    /// Completion event of a fence running on a [`Notifier`].
    event: Option<Event>,
    /// When the device finished, if a notifier observed it.
    completed_at: Option<Instant>,
    /// Host copy of the output handed to the notifier.
    copied: Option<Vec<f32>>,
}

impl<T: Send + 'static> Completion<T> {
    /// Begin ownership before encoding any work that references these resources.
    pub fn new(device: Device, quarantine: &Quarantine, resources: T) -> Self {
        Self {
            fence: Fence::Device(device),
            quarantine: quarantine.clone(),
            resources: Some(resources),
            completed: false,
            event: None,
            completed_at: None,
            copied: None,
        }
    }

    /// Access retained resources while encoding. The caller must keep every live binding here.
    /// Waits for a fence already handed to a notifier, which covers only earlier work.
    pub fn resources_mut(&mut self) -> Option<&mut T> {
        if let Some(event) = self.event.take() {
            // Only orders this access after earlier work: with `completed` reset, the next
            // `wait` checks the fence again and reports its outcome.
            drop(event.recv());
        }
        self.completed = false;
        self.resources.as_mut()
    }

    /// Hand the fence to `notifier` after the last encoding, so the device starts every
    /// pending command now and completion is observed off this thread. `output`, an F32
    /// tensor written by this work, is copied to the host as part of the fence; see
    /// [`Self::take_copy`].
    pub fn notify(&mut self, notifier: &Notifier, output: Option<Tensor>) -> Result<()> {
        if self.event.is_none() {
            self.event = Some(notifier.start(self.fence.clone(), output)?);
        }
        Ok(())
    }

    /// The host copy made by the notifier, once completed.
    pub fn take_copy(&mut self) -> Option<Vec<f32>> {
        self.copied.take()
    }

    /// Wait for device completion, consuming a notifier's event when one was started. On
    /// error retain everything for a later retry.
    pub fn wait(&mut self) -> Result<()> {
        self.completed = false;
        match self.event.take() {
            Some(event) => {
                let (at, copied) = event
                    .recv()
                    .map_err(|_| TurError::Other("completion notifier stopped".into()))??;
                self.completed_at = Some(at);
                self.copied = copied;
            }
            None => self.fence.wait()?,
        }
        self.completed = true;
        Ok(())
    }

    /// When a notifier saw the device finish this work.
    pub fn completed_at(&self) -> Option<Instant> {
        self.completed_at
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
        let notified = self.event.take().map(|event| event.recv());
        if !self.completed
            && !matches!(notified, Some(Ok(Ok(_))))
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
            event: None,
            completed_at: None,
            copied: None,
        }
    }

    #[test]
    fn test_notified_completion_reports_its_time_and_failures() -> Result<()> {
        let notifier = Notifier::new()?;
        let drops = Arc::new(AtomicUsize::new(0));
        let before = Instant::now();
        let mut completion = Completion::new(
            Device::Cpu,
            &Quarantine::new(),
            Resource(Arc::clone(&drops)),
        );
        let output = Tensor::new(&[1f32, 2.0], &Device::Cpu)?;
        completion.notify(&notifier, Some(output))?;
        assert!(completion.take().is_none());
        completion.wait()?;
        assert!(completion.completed_at().is_some_and(|at| at >= before));
        assert_eq!(completion.take_copy(), Some(vec![1.0, 2.0]));
        drop(completion.take());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        // A failed notified fence keeps resources; a later direct wait succeeds.
        let mut failing = controlled(1, &Quarantine::new(), Resource(Arc::clone(&drops)));
        failing.notify(&notifier, None)?;
        assert!(failing.wait().is_err());
        assert!(failing.take().is_none());
        failing.wait()?;
        drop(failing.take());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        // Dropping with an outstanding event waits for it before releasing.
        let mut dropped = Completion::new(
            Device::Cpu,
            &Quarantine::new(),
            Resource(Arc::clone(&drops)),
        );
        dropped.notify(&notifier, None)?;
        drop(dropped);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        Ok(())
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
