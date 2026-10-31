#![forbid(unsafe_code)]
//! Progress of long work such as downloads, weight loading and quantization. Stages publish
//! structured `tracing` events, so no signature grows a callback and work nobody watches pays
//! for a filtered event only. A front end renders them by subscribing to [`TARGET`].

/// The `tracing` target of every progress event.
pub const TARGET: &str = "tur_progress";

/// [`Stage`] unit that a front end displays as a byte size.
pub const BYTES: &str = "bytes";

/// One piece of long work. Dropping it unfinished reports a failure, so the `?` that abandons
/// the work also ends its progress.
#[derive(Debug)]
pub struct Stage {
    name: String,
    unit: &'static str,
    done: u64,
    total: u64,
    ended: bool,
}

impl Stage {
    /// Start a stage of `total` `unit`s. A `total` of zero means the size is unknown.
    #[must_use]
    pub fn begin(name: impl Into<String>, total: u64, unit: &'static str) -> Self {
        let stage = Self {
            name: name.into(),
            unit,
            done: 0,
            total,
            ended: false,
        };
        stage.emit("begin");
        stage
    }

    /// Count `by` more units done.
    pub fn advance(&mut self, by: u64) {
        self.done = self.done.saturating_add(by);
        self.emit("advance");
    }

    /// End the stage successfully.
    pub fn finish(mut self) {
        self.ended = true;
        self.emit("finish");
    }

    fn emit(&self, state: &'static str) {
        tracing::debug!(
            target: TARGET,
            stage = self.name.as_str(),
            state,
            unit = self.unit,
            done = self.done,
            total = self.total,
        );
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if !self.ended {
            self.emit("fail");
        }
    }
}
