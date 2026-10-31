#![forbid(unsafe_code)]
//! What the server reports about itself: worker counters published once per pass, a
//! periodic reading of system memory, and the Prometheus text of `/metrics`.

use parking_lot::Mutex;
use std::{
    collections::BTreeMap,
    fmt::{self, Write as _},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use crate::{
    Result,
    backend::{
        memory::MemoryPlan,
        pipeline::{GenerationStats, PassTiming},
        prefix::PrefixStats,
        scheduler::SchedulerConfig,
        speculation::SpeculationStats,
    },
    hybrid::{PoolUsage, SamplingStats},
    runtime::system::SystemMemory,
    server::routes::chat::execution_error,
};

/// How often system memory is read. Each reading spawns `vm_stat` and `ps`.
const SYSTEM_INTERVAL: Duration = Duration::from_secs(10);
/// Spare memory below which the server warns that concurrent work may start to swap.
const LOW_SPARE_BYTES: usize = 1 << 30;

/// Upper bounds, in seconds, of the latency histograms' buckets.
const LATENCY_BOUNDS: [f64; 12] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// A cumulative latency histogram.
#[derive(Clone, Copy, Debug, Default)]
pub struct Histogram {
    buckets: [u64; LATENCY_BOUNDS.len()],
    count: u64,
    sum: f64,
}

impl Histogram {
    fn observe(&mut self, value: Duration) {
        let seconds = value.as_secs_f64();
        self.count += 1;
        self.sum += seconds;
        for (bound, bucket) in LATENCY_BOUNDS.iter().zip(&mut self.buckets) {
            if seconds <= *bound {
                *bucket += 1;
            }
        }
    }

    fn render(&self, out: &mut String, name: &str, labels: &str) -> fmt::Result {
        for (bound, bucket) in LATENCY_BOUNDS.iter().zip(&self.buckets) {
            writeln!(out, "{name}_bucket{{{labels},le=\"{bound}\"}} {bucket}")?;
        }
        writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {}", self.count)?;
        writeln!(out, "{name}_sum{{{labels}}} {}", self.sum)?;
        writeln!(out, "{name}_count{{{labels}}} {}", self.count)?;
        Ok(())
    }
}

/// Immutable facts about a loaded chat model.
#[derive(Clone, Debug)]
pub struct ModelReport {
    /// The Hugging Face repository, or `local`: a local path says nothing a client needs.
    pub source: String,
    pub revision: String,
    pub weight_format: String,
    pub kv: String,
    pub recurrent_state: String,
    pub max_position: usize,
    pub vocab_size: usize,
    pub plan: MemoryPlan,
    pub scheduler: SchedulerConfig,
    pub device_sampling: bool,
    pub speculation: bool,
    /// Device capabilities and launch tuning; null on the CPU reference.
    pub kernels: serde_json::Value,
}

/// Seconds of pass phases, summed; see [`PassTiming`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PassSeconds {
    pub schedule: f64,
    pub encode: f64,
    pub overlap: f64,
    pub wait: f64,
    pub publish: f64,
    pub device: f64,
}

/// A chat worker's counters. The worker writes them; readers copy them.
#[derive(Clone, Debug, Default)]
pub struct WorkerCounters {
    pub running: usize,
    pub waiting: usize,
    pub pool: PoolUsage,
    pub prefix: PrefixStats,
    pub sampling: SamplingStats,
    pub speculation: SpeculationStats,
    pub passes: u64,
    pub pass_tokens: u64,
    pub pass_seconds: PassSeconds,
    /// Device allocations now and at their highest, read after each pass.
    pub device_allocated: Option<usize>,
    pub device_allocated_peak: usize,
    /// Finished requests by outcome: `ok`, `cancelled` or the error code a client saw.
    pub outcomes: BTreeMap<&'static str, u64>,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub preemptions: u64,
    pub request_seconds: Histogram,
    pub first_token_seconds: Histogram,
}

/// Shared between a chat worker and whoever observes or stops it.
#[derive(Debug, Default)]
pub struct WorkerTelemetry {
    counters: Mutex<WorkerCounters>,
    stop: AtomicBool,
}

impl WorkerTelemetry {
    /// A copy of the current counters.
    pub fn counters(&self) -> WorkerCounters {
        self.counters.lock().clone()
    }

    /// Ask the worker to fail what it holds and exit after its current pass.
    pub(crate) fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    pub(crate) fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Count one request's outcome.
    pub(crate) fn record(&self, result: &Result<GenerationStats>) {
        let mut counters = self.counters.lock();
        match result {
            Ok(stats) => {
                *counters.outcomes.entry("ok").or_default() += 1;
                counters.prompt_tokens += stats.prompt_tokens as u64;
                counters.cached_tokens += stats.cached_tokens as u64;
                counters.generated_tokens += stats.generated_tokens as u64;
                counters.preemptions += stats.preemptions as u64;
                counters.request_seconds.observe(stats.elapsed);
                counters.first_token_seconds.observe(stats.first_token);
            }
            Err(error) => {
                let code = execution_error(error)
                    .1
                    .error
                    .code
                    .unwrap_or("internal_error");
                *counters.outcomes.entry(code).or_default() += 1;
            }
        }
    }

    /// Count a request whose client went away.
    pub(crate) fn record_cancelled(&self) {
        *self
            .counters
            .lock()
            .outcomes
            .entry("cancelled")
            .or_default() += 1;
    }

    /// Publish the state after one pass.
    pub(crate) fn publish_pass(&self, pass: &PassReading) {
        let mut counters = self.counters.lock();
        counters.running = pass.active - pass.waiting;
        counters.waiting = pass.waiting;
        counters.pool = pass.pool;
        counters.prefix = pass.prefix;
        counters.sampling = pass.sampling;
        counters.speculation = pass.speculation;
        counters.device_allocated = pass.allocated;
        counters.device_allocated_peak = counters
            .device_allocated_peak
            .max(pass.allocated.unwrap_or(0));
        if let Some(timing) = pass.timing {
            counters.passes += 1;
            counters.pass_tokens += timing.tokens as u64;
            let seconds = &mut counters.pass_seconds;
            seconds.schedule += timing.schedule.as_secs_f64();
            seconds.encode += timing.encode.as_secs_f64();
            seconds.overlap += timing.overlap.as_secs_f64();
            seconds.wait += timing.wait.as_secs_f64();
            seconds.publish += timing.publish.as_secs_f64();
            seconds.device += timing.device.as_secs_f64();
        }
    }
}

/// One pass's view of the pipeline, for [`WorkerTelemetry::publish_pass`].
pub(crate) struct PassReading {
    pub(crate) active: usize,
    pub(crate) waiting: usize,
    pub(crate) pool: PoolUsage,
    pub(crate) prefix: PrefixStats,
    pub(crate) sampling: SamplingStats,
    pub(crate) speculation: SpeculationStats,
    pub(crate) allocated: Option<usize>,
    pub(crate) timing: Option<PassTiming>,
}

/// The latest reading of system memory.
#[derive(Clone, Copy, Debug)]
pub struct SystemReading {
    pub spare_bytes: usize,
    /// Pages written to swap since this server started.
    pub swapouts: u64,
    pub process_resident_bytes: Option<u64>,
}

/// Server-wide state: when it started, whether it is draining, and what it counted.
#[derive(Debug)]
pub struct ServerStatus {
    started: Instant,
    draining: AtomicBool,
    http: Mutex<BTreeMap<(String, u16), u64>>,
    system: Mutex<Option<SystemReading>>,
}

impl Default for ServerStatus {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            draining: AtomicBool::new(false),
            http: Mutex::new(BTreeMap::new()),
            system: Mutex::new(None),
        }
    }
}

impl ServerStatus {
    /// A fresh status to share among a router's handlers.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Whether the server has stopped taking new work.
    pub fn draining(&self) -> bool {
        self.draining.load(Ordering::Relaxed)
    }

    /// Stop taking new work; requests already accepted keep running.
    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::Relaxed);
    }

    /// Time since the server started.
    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// Count one finished HTTP exchange. Paths are route patterns, so the map stays small.
    pub(crate) fn record_http(&self, path: &str, status: u16) {
        *self
            .http
            .lock()
            .entry((path.to_owned(), status))
            .or_default() += 1;
    }

    /// The latest reading of system memory, if one succeeded.
    pub fn system(&self) -> Option<SystemReading> {
        *self.system.lock()
    }
}

/// This process's resident memory, from `ps`.
fn process_resident_bytes() -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kib: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    kib.checked_mul(1024)
}

/// Warn when spare memory runs low or the machine starts swapping, once per episode. All
/// model memory was reserved at startup, so this reports pressure from elsewhere; there is
/// nothing the server could free that would relieve it.
#[derive(Default)]
struct PressureWatch {
    baseline: Option<u64>,
    last_swapouts: u64,
    low: bool,
    swapping: bool,
}

impl PressureWatch {
    fn observe(&mut self, system: SystemMemory) -> u64 {
        let baseline = *self.baseline.get_or_insert(system.swapouts);
        let low = system.spare_bytes < LOW_SPARE_BYTES;
        if low && !self.low {
            tracing::warn!(
                spare_mib = system.spare_bytes >> 20,
                "system memory is low; other applications may push this server into swap"
            );
        }
        self.low = low;
        let swapping = system.swapouts > self.last_swapouts && self.last_swapouts >= baseline;
        if swapping && !self.swapping {
            tracing::warn!("macOS is writing memory to swap while this server runs");
        }
        self.swapping = swapping;
        self.last_swapouts = system.swapouts;
        system.swapouts.saturating_sub(baseline)
    }
}

/// Keep `status`'s system reading current until the server drains or this task is aborted.
pub(crate) async fn monitor_system(status: Arc<ServerStatus>) {
    let mut watch = PressureWatch::default();
    let mut ticks = tokio::time::interval(SYSTEM_INTERVAL);
    while !status.draining() {
        ticks.tick().await;
        let reading = tokio::task::spawn_blocking(|| {
            SystemMemory::read().map(|system| (system, process_resident_bytes()))
        })
        .await;
        match reading {
            Ok(Ok((system, process_resident_bytes))) => {
                let swapouts = watch.observe(system);
                *status.system.lock() = Some(SystemReading {
                    spare_bytes: system.spare_bytes,
                    swapouts,
                    process_resident_bytes,
                });
            }
            Ok(Err(error)) => tracing::debug!("cannot read system memory: {error}"),
            Err(error) => tracing::debug!("system memory reader failed: {error}"),
        }
    }
}

/// Quote a Prometheus label value.
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// One metric's header and its unlabeled sample.
fn single(
    out: &mut String,
    name: &str,
    kind: &str,
    help: &str,
    value: impl fmt::Display,
) -> fmt::Result {
    writeln!(out, "# HELP {name} {help}")?;
    writeln!(out, "# TYPE {name} {kind}")?;
    writeln!(out, "{name} {value}")?;
    Ok(())
}

/// One metric's header.
fn header(out: &mut String, name: &str, kind: &str, help: &str) -> fmt::Result {
    writeln!(out, "# HELP {name} {help}")?;
    writeln!(out, "# TYPE {name} {kind}")?;
    Ok(())
}

/// Render the server-wide metrics of `/metrics`.
pub(crate) fn render_server(out: &mut String, status: &ServerStatus) -> fmt::Result {
    single(
        out,
        "tur_uptime_seconds",
        "gauge",
        "Seconds since the server started.",
        status.uptime().as_secs_f64(),
    )?;
    single(
        out,
        "tur_draining",
        "gauge",
        "1 while the server finishes accepted requests before exiting.",
        u8::from(status.draining()),
    )?;
    header(
        out,
        "tur_http_requests_total",
        "counter",
        "Finished HTTP exchanges by route and status. A failure mid-stream keeps status 200; see tur_requests_total.",
    )?;
    for ((path, code), count) in status.http.lock().iter() {
        writeln!(
            out,
            "tur_http_requests_total{{path=\"{}\",status=\"{code}\"}} {count}",
            label(path)
        )?;
    }
    if let Some(system) = status.system() {
        single(
            out,
            "tur_system_spare_memory_bytes",
            "gauge",
            "Bytes macOS can hand out without compressing or swapping.",
            system.spare_bytes,
        )?;
        single(
            out,
            "tur_system_swapouts_total",
            "counter",
            "Pages macOS wrote to swap since this server started.",
            system.swapouts,
        )?;
        if let Some(bytes) = system.process_resident_bytes {
            single(
                out,
                "tur_process_resident_bytes",
                "gauge",
                "Resident memory of this process.",
                bytes,
            )?;
        }
    }
    Ok(())
}

/// One metric: name, type, help and value.
type Sample = (&'static str, &'static str, &'static str, f64);

/// What requests and passes did.
fn activity_samples(c: &WorkerCounters) -> [Sample; 10] {
    [
        (
            "tur_requests_running",
            "gauge",
            "Requests holding model state.",
            c.running as f64,
        ),
        (
            "tur_requests_waiting",
            "gauge",
            "Requests waiting for admission or recomputation.",
            c.waiting as f64,
        ),
        (
            "tur_prompt_tokens_total",
            "counter",
            "Prompt tokens of finished requests.",
            c.prompt_tokens as f64,
        ),
        (
            "tur_cached_prompt_tokens_total",
            "counter",
            "Prompt tokens restored from snapshots.",
            c.cached_tokens as f64,
        ),
        (
            "tur_generated_tokens_total",
            "counter",
            "Tokens generated by finished requests.",
            c.generated_tokens as f64,
        ),
        (
            "tur_preemptions_total",
            "counter",
            "Times finished requests were preempted and recomputed.",
            c.preemptions as f64,
        ),
        (
            "tur_passes_total",
            "counter",
            "Packed passes run.",
            c.passes as f64,
        ),
        (
            "tur_pass_tokens_total",
            "counter",
            "Tokens computed in packed passes.",
            c.pass_tokens as f64,
        ),
        (
            "tur_device_seconds_total",
            "counter",
            "Seconds from encoding to device completion.",
            c.pass_seconds.device,
        ),
        (
            "tur_device_allocated_peak_bytes",
            "gauge",
            "Most bytes the Metal device had allocated.",
            c.device_allocated_peak as f64,
        ),
    ]
}

/// What the pool holds, and how its caches and samplers behaved.
fn state_samples(c: &WorkerCounters) -> [Sample; 13] {
    [
        (
            "tur_kv_pages_capacity",
            "gauge",
            "K/V pages in the state pool.",
            c.pool.pages.capacity as f64,
        ),
        (
            "tur_kv_pages_leased",
            "gauge",
            "K/V pages held by requests and snapshots.",
            c.pool.pages.leased as f64,
        ),
        (
            "tur_kv_pages_peak",
            "gauge",
            "Most K/V pages ever leased.",
            c.pool.pages.peak as f64,
        ),
        (
            "tur_recurrent_slots_capacity",
            "gauge",
            "Recurrent slots in the state pool.",
            c.pool.slots.capacity as f64,
        ),
        (
            "tur_recurrent_slots_leased",
            "gauge",
            "Recurrent slots held by requests and snapshots.",
            c.pool.slots.leased as f64,
        ),
        (
            "tur_prefix_snapshots",
            "gauge",
            "Prefix snapshots held.",
            c.prefix.entries as f64,
        ),
        (
            "tur_prefix_lookups_total",
            "counter",
            "Prefix cache lookups.",
            c.prefix.lookups as f64,
        ),
        (
            "tur_prefix_hits_total",
            "counter",
            "Prefix cache hits.",
            c.prefix.hits as f64,
        ),
        (
            "tur_prefix_evictions_total",
            "counter",
            "Prefix snapshots evicted.",
            c.prefix.evicted as f64,
        ),
        (
            "tur_sampled_device_rows_total",
            "counter",
            "Output rows sampled on the device.",
            c.sampling.device_rows as f64,
        ),
        (
            "tur_sampled_host_rows_total",
            "counter",
            "Output rows sampled on the host.",
            c.sampling.host_rows as f64,
        ),
        (
            "tur_draft_tokens_total",
            "counter",
            "Speculative draft tokens verified.",
            c.speculation.drafted as f64,
        ),
        (
            "tur_draft_tokens_accepted_total",
            "counter",
            "Speculative draft tokens accepted.",
            c.speculation.accepted as f64,
        ),
    ]
}

/// What startup planned.
fn plan_samples(plan: &MemoryPlan) -> [Sample; 6] {
    [
        (
            "tur_context_tokens",
            "gauge",
            "Most positions one request may use.",
            plan.context_tokens as f64,
        ),
        (
            "tur_full_contexts",
            "gauge",
            "Requests the pool holds at once at the sized context.",
            plan.full_contexts as f64,
        ),
        (
            "tur_memory_plan_resident_bytes",
            "gauge",
            "Device memory reserved for this model.",
            plan.resident() as f64,
        ),
        (
            "tur_memory_plan_weight_bytes",
            "gauge",
            "Planned weight bytes.",
            plan.weights as f64,
        ),
        (
            "tur_memory_plan_state_pool_bytes",
            "gauge",
            "Planned state pool bytes.",
            plan.state_pool as f64,
        ),
        (
            "tur_memory_plan_scratch_bytes",
            "gauge",
            "Planned kernel scratch bytes.",
            plan.scratch_peak as f64,
        ),
    ]
}

/// Every per-model metric that has one value.
fn model_samples(report: Option<&ModelReport>, counters: &WorkerCounters) -> Vec<Sample> {
    let mut samples = Vec::with_capacity(30);
    samples.extend(activity_samples(counters));
    samples.extend(state_samples(counters));
    if let Some(allocated) = counters.device_allocated {
        samples.push((
            "tur_device_allocated_bytes",
            "gauge",
            "Bytes the Metal device has allocated.",
            allocated as f64,
        ));
    }
    if let Some(report) = report {
        samples.extend(plan_samples(&report.plan));
    }
    samples
}

/// Render one chat model's metrics, labeled with its served name.
pub(crate) fn render_model(
    out: &mut String,
    model: &str,
    report: Option<&ModelReport>,
    counters: &WorkerCounters,
    alive: bool,
) -> fmt::Result {
    let labels = format!("model=\"{}\"", label(model));
    header(
        out,
        "tur_worker_up",
        "gauge",
        "1 while the model's worker accepts requests.",
    )?;
    writeln!(out, "tur_worker_up{{{labels}}} {}", u8::from(alive))?;
    if let Some(report) = report {
        header(
            out,
            "tur_model_info",
            "gauge",
            "The served checkpoint, as labels.",
        )?;
        writeln!(
            out,
            "tur_model_info{{{labels},source=\"{}\",revision=\"{}\"}} 1",
            label(&report.source),
            label(&report.revision)
        )?;
    }
    for (name, kind, help, value) in model_samples(report, counters) {
        header(out, name, kind, help)?;
        writeln!(out, "{name}{{{labels}}} {value}")?;
    }
    header(
        out,
        "tur_requests_total",
        "counter",
        "Finished requests by outcome.",
    )?;
    for (outcome, count) in &counters.outcomes {
        writeln!(
            out,
            "tur_requests_total{{{labels},outcome=\"{outcome}\"}} {count}"
        )?;
    }
    header(
        out,
        "tur_pass_phase_seconds_total",
        "counter",
        "Host seconds of packed passes by phase.",
    )?;
    let seconds = counters.pass_seconds;
    for (phase, value) in [
        ("schedule", seconds.schedule),
        ("encode", seconds.encode),
        ("overlap", seconds.overlap),
        ("wait", seconds.wait),
        ("publish", seconds.publish),
    ] {
        writeln!(
            out,
            "tur_pass_phase_seconds_total{{{labels},phase=\"{phase}\"}} {value}"
        )?;
    }
    header(
        out,
        "tur_request_duration_seconds",
        "histogram",
        "Time from arrival to completion of successful requests.",
    )?;
    counters
        .request_seconds
        .render(out, "tur_request_duration_seconds", &labels)?;
    header(
        out,
        "tur_time_to_first_token_seconds",
        "histogram",
        "Time from arrival to the first sampled token.",
    )?;
    counters
        .first_token_seconds
        .render(out, "tur_time_to_first_token_seconds", &labels)?;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_buckets_are_cumulative_and_end_at_the_count() -> fmt::Result {
        let mut histogram = Histogram::default();
        histogram.observe(Duration::from_millis(40));
        histogram.observe(Duration::from_secs(1));
        histogram.observe(Duration::from_secs(1000));
        let mut out = String::new();
        histogram.render(&mut out, "m", "model=\"x\"")?;
        assert!(
            out.contains("m_bucket{model=\"x\",le=\"0.05\"} 1\n"),
            "{out}"
        );
        assert!(out.contains("m_bucket{model=\"x\",le=\"1\"} 2\n"), "{out}");
        assert!(
            out.contains("m_bucket{model=\"x\",le=\"300\"} 2\n"),
            "{out}"
        );
        assert!(
            out.contains("m_bucket{model=\"x\",le=\"+Inf\"} 3\n"),
            "{out}"
        );
        assert!(out.contains("m_count{model=\"x\"} 3\n"), "{out}");
        Ok(())
    }

    #[test]
    fn test_pressure_warnings_count_swap_since_start_only() {
        let reading = |spare_bytes, swapouts| SystemMemory {
            spare_bytes,
            swapouts,
        };
        let mut watch = PressureWatch::default();
        // Swap from before this server started is not counted against it.
        assert_eq!(watch.observe(reading(1 << 32, 1000)), 0);
        assert!(!watch.swapping);
        assert_eq!(watch.observe(reading(1 << 32, 1010)), 10);
        assert!(watch.swapping);
        assert_eq!(watch.observe(reading(1 << 20, 1010)), 10);
        assert!(!watch.swapping);
        assert!(watch.low);
    }

    #[test]
    fn test_label_values_are_escaped() {
        assert_eq!(label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
