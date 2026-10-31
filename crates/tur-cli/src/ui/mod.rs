//! How the CLI talks to a person: progress bars, logs that do not break them, and summaries.
//! All of it goes to stderr, so stdout stays exactly the command's result.
mod banner;
mod format;

pub(crate) use banner::welcome;
pub(crate) use format::{bytes, count, seconds};

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use parking_lot::Mutex;
use std::{
    collections::{HashMap, VecDeque},
    io::{IsTerminal as _, Write as _},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tracing::field::{Field, Visit};
use tracing_subscriber::{
    Layer, filter::filter_fn, fmt::MakeWriter, fmt::time::Uptime, layer::SubscriberExt as _,
    util::SubscriberInitExt as _,
};
use tur::{
    Result, TurError,
    core::progress::{BYTES, TARGET},
};

/// Log lines a dashboard keeps, and replays when it closes.
const CAPTURED_LINES: usize = 500;
/// How often a stage reports when nobody watches a live bar.
const PLAIN_INTERVAL: Duration = Duration::from_secs(10);

/// One running stage.
struct Live {
    bar: ProgressBar,
    unit: String,
    started: Instant,
    reported: Instant,
}

struct Inner {
    multi: MultiProgress,
    interactive: bool,
    color: bool,
    quiet: bool,
    /// Off once a server runs: its requests are not startup work to narrate.
    progress: AtomicBool,
    stages: Mutex<HashMap<String, Live>>,
    captured: Mutex<Option<VecDeque<String>>>,
}

/// The terminal side of the process; cheap to clone.
#[derive(Clone)]
pub(crate) struct Console(Arc<Inner>);

impl Console {
    fn new(quiet: bool) -> Self {
        let interactive = std::io::stderr().is_terminal()
            && std::env::var_os("TERM").is_none_or(|term| term != "dumb");
        let target = if interactive && !quiet {
            ProgressDrawTarget::stderr()
        } else {
            ProgressDrawTarget::hidden()
        };
        Self(Arc::new(Inner {
            multi: MultiProgress::with_draw_target(target),
            interactive,
            color: interactive && std::env::var_os("NO_COLOR").is_none(),
            quiet,
            progress: AtomicBool::new(!quiet),
            stages: Mutex::new(HashMap::new()),
            captured: Mutex::new(None),
        }))
    }

    /// Whether a person is watching stderr and may see live output.
    pub(crate) fn interactive(&self) -> bool {
        self.0.interactive && !self.0.quiet
    }

    /// Whether output may carry color.
    pub(crate) fn colored(&self) -> bool {
        self.0.color
    }

    /// `text` in the SGR style `code`, when color is on.
    pub(crate) fn paint(&self, code: &str, text: &str) -> String {
        if self.0.color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    /// Print a line above any live bars, unless `--quiet`.
    pub(crate) fn line(&self, text: &str) {
        if !self.0.quiet {
            self.write(text.as_bytes(), true);
        }
    }

    /// Print aligned `label  value` rows, unless `--quiet`.
    pub(crate) fn rows(&self, rows: &[(&str, String)]) {
        let width = rows.iter().map(|row| row.0.len()).max().unwrap_or(0);
        for (label, value) in rows {
            let label = self.paint("2", &format!("{label:width$}"));
            self.line(&format!("  {label}  {value}"));
        }
    }

    /// Stop drawing progress, for a server that has finished starting.
    pub(crate) fn stop_progress(&self) {
        self.0.progress.store(false, Ordering::Relaxed);
    }

    /// Hold log lines for a dashboard instead of printing them over its screen.
    pub(crate) fn capture_logs(&self) {
        *self.0.captured.lock() = Some(VecDeque::with_capacity(CAPTURED_LINES));
    }

    /// The newest `count` captured lines.
    pub(crate) fn recent_logs(&self, count: usize) -> Vec<String> {
        self.0
            .captured
            .lock()
            .as_ref()
            .map_or_else(Vec::new, |lines| {
                let skip = lines.len().saturating_sub(count);
                lines.iter().skip(skip).cloned().collect()
            })
    }

    /// Print logs directly again, after replaying what was captured.
    pub(crate) fn release_logs(&self) {
        let captured = self.0.captured.lock().take();
        for line in captured.into_iter().flatten() {
            self.write(line.as_bytes(), true);
        }
    }

    fn log(&self, text: &[u8]) {
        {
            let mut captured = self.0.captured.lock();
            if let Some(lines) = captured.as_mut() {
                for line in String::from_utf8_lossy(text).lines() {
                    if lines.len() == CAPTURED_LINES {
                        lines.pop_front();
                    }
                    lines.push_back(line.to_owned());
                }
                return;
            }
        }
        self.write(text, false);
    }

    fn write(&self, text: &[u8], newline: bool) {
        self.0.multi.suspend(|| {
            let mut stderr = std::io::stderr();
            // Nothing depends on messages reaching a closed stderr.
            drop(stderr.write_all(text));
            if newline {
                drop(stderr.write_all(b"\n"));
            }
        });
    }

    fn begin(&self, event: &Event) {
        let bar = if self.interactive() {
            let bar = self
                .0
                .multi
                .add(ProgressBar::new(event.total).with_style(style(event)));
            bar.set_message(event.stage.clone());
            bar.enable_steady_tick(Duration::from_millis(100));
            bar
        } else {
            ProgressBar::hidden()
        };
        let now = Instant::now();
        self.0.stages.lock().insert(
            event.stage.clone(),
            Live {
                bar,
                unit: event.unit.clone(),
                started: now,
                reported: now,
            },
        );
    }

    fn advance(&self, event: &Event) {
        let mut plain = None;
        if let Some(live) = self.0.stages.lock().get_mut(&event.stage) {
            live.bar.set_position(event.done);
            if !self.0.interactive && live.reported.elapsed() >= PLAIN_INTERVAL {
                live.reported = Instant::now();
                plain = Some(format!(
                    "{}: {}",
                    event.stage,
                    amount(event.done, event.total, &event.unit)
                ));
            }
        }
        if let Some(text) = plain {
            self.line(&text);
        }
    }

    fn end(&self, event: &Event, succeeded: bool) {
        let Some(live) = self.0.stages.lock().remove(&event.stage) else {
            return;
        };
        live.bar.finish_and_clear();
        let mark = if succeeded {
            self.paint("32", "✓")
        } else {
            self.paint("31", "✗")
        };
        let done = amount(event.done, event.total, &live.unit);
        let elapsed = seconds(live.started.elapsed().as_secs_f64());
        let detail = if done.is_empty() {
            format!("in {elapsed}")
        } else if succeeded {
            format!("{done} in {elapsed}")
        } else {
            format!("stopped at {done}")
        };
        self.line(&format!(
            "  {mark} {}  {}",
            event.stage,
            self.paint("2", &detail)
        ));
    }

    fn progress(&self, event: &Event) {
        if !self.0.progress.load(Ordering::Relaxed) {
            return;
        }
        match event.state.as_str() {
            "begin" => self.begin(event),
            "advance" => self.advance(event),
            "finish" => self.end(event, true),
            _ => self.end(event, false),
        }
    }
}

/// `3/12 layers`, a byte size, or nothing for a stage of unknown size.
fn amount(done: u64, total: u64, unit: &str) -> String {
    if unit == BYTES {
        format!("{} / {}", bytes(done as f64), bytes(total as f64))
    } else if total == 0 {
        String::new()
    } else {
        format!("{done}/{total} {unit}")
    }
}

fn style(event: &Event) -> ProgressStyle {
    let tail = if event.unit == BYTES {
        "{bytes}/{total_bytes}  {binary_bytes_per_sec}  eta {eta}".to_owned()
    } else {
        format!("{{pos}}/{{len}} {}  {{elapsed}}  eta {{eta}}", event.unit)
    };
    let template = if event.total == 0 {
        "  {spinner:.cyan} {msg}  {elapsed}".to_owned()
    } else {
        format!("  {{spinner:.cyan}} {{msg}}  {{wide_bar:.cyan/238}}  {tail}")
    };
    ProgressStyle::with_template(&template)
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("━╸─")
}

/// One progress event's fields.
#[derive(Default)]
struct Event {
    stage: String,
    state: String,
    unit: String,
    done: u64,
    total: u64,
}

impl Visit for Event {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "stage" => value.clone_into(&mut self.stage),
            "state" => value.clone_into(&mut self.state),
            "unit" => value.clone_into(&mut self.unit),
            _ => {}
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "done" => self.done = value,
            "total" => self.total = value,
            _ => {}
        }
    }

    fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}
}

impl<S: tracing::Subscriber> Layer<S> for Console {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let mut fields = Event::default();
        event.record(&mut fields);
        self.progress(&fields);
    }
}

/// Routes `tracing`'s formatted lines through the console.
pub(crate) struct LogWriter(Console);

impl std::io::Write for LogWriter {
    fn write(&mut self, text: &[u8]) -> std::io::Result<usize> {
        self.0.log(text);
        Ok(text.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Console {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> LogWriter {
        LogWriter(self.clone())
    }
}

/// Install logging and progress. In a terminal, logs are short and start at `level`; piped
/// output keeps timestamps, threads and targets for whoever collects it, at `info`.
pub(crate) fn init(quiet: bool, level: &str) -> Result<Console> {
    let console = Console::new(quiet);
    let logs = if console.0.interactive {
        let filter = tur::shared::log_filter(if quiet { "warn" } else { level })?;
        tracing_subscriber::fmt::layer()
            .compact()
            // A dashboard shows these as plain text, so no color escapes.
            .with_ansi(false)
            .with_timer(Uptime::default())
            .with_target(false)
            .with_writer(console.clone())
            .with_filter(filter)
            .boxed()
    } else {
        let filter = tur::shared::log_filter(if quiet { "warn" } else { "info" })?;
        tracing_subscriber::fmt::layer()
            .compact()
            .with_ansi(false)
            .with_thread_names(true)
            .with_thread_ids(true)
            .with_writer(console.clone())
            .with_filter(filter)
            .boxed()
    };
    let progress = console
        .clone()
        .with_filter(filter_fn(|metadata| metadata.target() == TARGET));
    tracing_subscriber::registry()
        .with(logs)
        .with(progress)
        .try_init()
        .map_err(|error| TurError::Other(format!("Cannot initialize logging: {error}")))?;
    Ok(console)
}

/// An error as one readable sentence, with the cause `TurError`'s own message leaves out.
pub(crate) fn describe(error: &TurError) -> String {
    let message = match error {
        TurError::Other(message) => message.clone(),
        other => other.to_string(),
    };
    match std::error::Error::source(error) {
        Some(source) if !message.contains(&source.to_string()) => format!("{message}: {source}"),
        _ => message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::Context;
    use tur::core::progress::Stage;

    /// Records what a stage reports.
    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<(String, u64, u64)>>>);

    impl<S: tracing::Subscriber> Layer<S> for Recorder {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            let mut fields = Event::default();
            event.record(&mut fields);
            self.0
                .lock()
                .push((fields.state, fields.done, fields.total));
        }
    }

    fn recorded(work: impl FnOnce()) -> Vec<(String, u64, u64)> {
        let recorder = Recorder::default();
        let subscriber = tracing_subscriber::registry().with(recorder.clone());
        tracing::subscriber::with_default(subscriber, work);
        recorder.0.lock().clone()
    }

    #[test]
    fn test_finished_stage_reports_its_counts_and_an_abandoned_one_fails() {
        let finished = recorded(|| {
            let mut stage = Stage::begin("work", 2, "layers");
            stage.advance(2);
            stage.finish();
        });
        let states: Vec<_> = finished.iter().map(|event| event.0.as_str()).collect();
        assert_eq!(states, ["begin", "advance", "finish"]);
        assert_eq!((finished[2].1, finished[2].2), (2, 2));
        let abandoned = recorded(|| drop(Stage::begin("work", 2, "layers")));
        assert_eq!(abandoned.last().map(|event| event.0.as_str()), Some("fail"));
    }

    #[test]
    fn test_amounts_read_naturally_for_each_kind_of_stage() {
        assert_eq!(amount(3, 12, "layers"), "3/12 layers");
        assert_eq!(amount(1024, 2048, BYTES), "1.0 KiB / 2.0 KiB");
        assert_eq!(amount(0, 0, ""), "");
    }

    #[test]
    fn test_describe_keeps_the_cause_io_errors_hide() {
        let io = TurError::from(std::io::Error::other("disk full"));
        assert_eq!(describe(&io), "IO failure: disk full");
        assert_eq!(describe(&TurError::Other("plain".into())), "plain");
    }
}
