//! A server's `/diagnostics` as sections a person reads. `tur status` prints them and the
//! dashboard draws them, so both always say the same things.
use crate::ui::{bytes, count, seconds};
use serde_json::Value;
use std::{collections::VecDeque, time::Instant};

/// Throughput readings the dashboard's chart keeps.
const HISTORY: usize = 240;

fn number(value: &Value, pointer: &str) -> f64 {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

fn text<'a>(value: &'a Value, pointer: &str) -> &'a str {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or("-")
}

/// One line of a section; `meter` is how full the value is, from 0 to 1.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Row {
    pub(super) label: &'static str,
    pub(super) value: String,
    pub(super) meter: Option<f64>,
    pub(super) alert: bool,
}

impl Row {
    fn new(label: &'static str, value: String) -> Self {
        Self {
            label,
            value,
            meter: None,
            alert: false,
        }
    }

    fn meter(label: &'static str, used: f64, total: f64, value: String) -> Self {
        Self {
            meter: Some(if total > 0.0 { used / total } else { 0.0 }),
            ..Self::new(label, value)
        }
    }

    fn alert(mut self, alert: bool) -> Self {
        self.alert = alert;
        self
    }
}

/// A titled group of rows.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Section {
    pub(super) title: String,
    pub(super) rows: Vec<Row>,
}

/// The server's overall condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Health {
    Ready,
    Draining,
    WorkerStopped,
}

/// One parsed `/diagnostics` response. Holds the first chat model: the CLI serves one.
#[derive(Clone, Debug)]
pub(super) struct Snapshot {
    raw: Value,
    model: Option<String>,
}

impl Health {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::WorkerStopped => "worker stopped",
        }
    }
}

impl Snapshot {
    pub(super) fn parse(body: &str) -> tur::Result<Self> {
        let raw: Value = serde_json::from_str(body)?;
        let model = raw
            .get("models")
            .and_then(Value::as_object)
            .and_then(|models| models.keys().min().cloned());
        Ok(Self { raw, model })
    }

    fn model(&self) -> Option<&Value> {
        self.raw.get("models")?.get(self.model.as_ref()?)
    }

    fn activity(&self) -> Option<&Value> {
        self.model()?.get("activity")
    }

    pub(super) fn health(&self) -> Health {
        let stopped = |group: &str| {
            self.raw
                .get(group)
                .and_then(Value::as_object)
                .is_some_and(|workers| {
                    workers
                        .values()
                        .any(|worker| worker["alive"].as_bool() == Some(false))
                })
        };
        if stopped("models") || stopped("transcription") {
            Health::WorkerStopped
        } else if self.raw["draining"].as_bool() == Some(true) {
            Health::Draining
        } else {
            Health::Ready
        }
    }

    pub(super) fn version(&self) -> &str {
        text(&self.raw, "/version")
    }

    pub(super) fn uptime(&self) -> String {
        seconds(number(&self.raw, "/uptime_seconds"))
    }

    /// Everything to show, with live rates when a [`Rates`] has watched for a while.
    pub(super) fn sections(&self, rates: Option<&Rates>) -> Vec<Section> {
        let mut sections = Vec::new();
        if let (Some(name), Some(model)) = (&self.model, self.model()) {
            sections.push(model_section(name, model));
            let activity = &model["activity"];
            sections.push(traffic_section(model, activity, rates));
            sections.push(memory_section(model, activity, &self.raw));
            sections.push(efficiency_section(model, activity));
        }
        if let Some(section) = self.speech_section() {
            sections.push(section);
        }
        sections
    }

    fn speech_section(&self) -> Option<Section> {
        let workers = self.raw.get("transcription")?.as_object()?;
        let rows: Vec<Row> = workers
            .iter()
            .map(|(name, worker)| {
                let alive = worker["alive"].as_bool() == Some(true);
                Row::new(
                    "Whisper",
                    format!(
                        "{name} · {} · {}",
                        bytes(number(worker, "/leased_bytes")),
                        if alive { "running" } else { "stopped" }
                    ),
                )
                .alert(!alive)
            })
            .collect();
        (!rows.is_empty()).then(|| Section {
            title: "Speech".into(),
            rows,
        })
    }
}

fn model_section(name: &str, model: &Value) -> Section {
    let checkpoint = &model["checkpoint"];
    let revision: String = text(checkpoint, "/revision").chars().take(10).collect();
    Section {
        title: format!("Model {name}"),
        rows: vec![
            Row::new(
                "Checkpoint",
                format!("{} @ {revision}", text(checkpoint, "/source")),
            ),
            Row::new(
                "Storage",
                format!(
                    "weights {} · K/V {} · recurrent {}",
                    text(checkpoint, "/weight_format"),
                    text(checkpoint, "/kv"),
                    text(checkpoint, "/recurrent_state"),
                ),
            ),
            Row::new(
                "Capacity",
                format!(
                    "{} running · {} waiting · {} tokens per request",
                    number(checkpoint, "/scheduler/max_running"),
                    number(checkpoint, "/scheduler/max_waiting"),
                    count(number(checkpoint, "/context/limit_tokens")),
                ),
            ),
        ],
    }
}

/// The bound of the bucket holding quantile `q` of a cumulative histogram: an upper bound on
/// that latency, the most its buckets can tell.
fn quantile(histogram: &Value, q: f64) -> Option<f64> {
    let total = histogram["count"].as_f64().filter(|&total| total > 0.0)?;
    let bounds = histogram["bounds"].as_array()?;
    let buckets = histogram["buckets"].as_array()?;
    bounds
        .iter()
        .zip(buckets)
        .find(|(_, bucket)| bucket.as_f64().is_some_and(|b| b >= q * total))
        .and_then(|(bound, _)| bound.as_f64())
}

fn latency(histogram: &Value) -> String {
    let at =
        |q| quantile(histogram, q).map_or_else(|| "-".to_owned(), |b| format!("≤{}", seconds(b)));
    format!("p50 {} · p95 {}", at(0.5), at(0.95))
}

fn outcomes(activity: &Value) -> Row {
    let outcomes = activity["requests"].as_object();
    let tally = |wanted: fn(&str) -> bool| -> f64 {
        outcomes.map_or(0.0, |outcomes| {
            outcomes
                .iter()
                .filter(|(name, _)| wanted(name))
                .filter_map(|(_, value)| value.as_f64())
                // `sum` of nothing is negative zero, which prints as -0.
                .fold(0.0, |total, value| total + value)
        })
    };
    let failed = tally(|name| name != "ok" && name != "cancelled");
    Row::new(
        "Finished",
        format!(
            "{} ok · {} cancelled · {} failed",
            count(tally(|name| name == "ok")),
            count(tally(|name| name == "cancelled")),
            count(failed),
        ),
    )
    .alert(failed > 0.0)
}

fn traffic_section(model: &Value, activity: &Value, rates: Option<&Rates>) -> Section {
    let mut rows = Vec::new();
    if let Some(rates) = rates {
        rows.push(Row::new(
            "Throughput",
            format!(
                "{} tokens/s through the model",
                count(rates.tokens_per_second)
            ),
        ));
        rows.push(Row::meter(
            "Device busy",
            rates.device_busy,
            1.0,
            format!("{:.0}%", rates.device_busy * 100.0),
        ));
    }
    rows.push(Row::new(
        "Requests",
        format!(
            "{}/{} running · {} waiting",
            number(activity, "/running"),
            number(model, "/checkpoint/scheduler/max_running"),
            number(activity, "/waiting"),
        ),
    ));
    rows.push(outcomes(activity));
    let prompt = number(activity, "/tokens/prompt");
    let cached = number(activity, "/tokens/cached_prompt");
    rows.push(Row::new(
        "Tokens",
        format!(
            "{} prompt ({:.0}% reused) · {} generated",
            count(prompt),
            if prompt > 0.0 {
                cached / prompt * 100.0
            } else {
                0.0
            },
            count(number(activity, "/tokens/generated")),
        ),
    ));
    rows.push(Row::new(
        "First token",
        latency(&activity["latency"]["first_token_seconds"]),
    ));
    rows.push(Row::new(
        "Request",
        latency(&activity["latency"]["request_seconds"]),
    ));
    let passes = number(activity, "/passes/count");
    rows.push(Row::new(
        "Passes",
        format!(
            "{} · {:.0} tokens each · {} preempted",
            count(passes),
            if passes > 0.0 {
                number(activity, "/passes/tokens") / passes
            } else {
                0.0
            },
            count(number(activity, "/preemptions")),
        ),
    ));
    Section {
        title: "Traffic".into(),
        rows,
    }
}

fn usage(label: &'static str, usage: &Value) -> Row {
    let (leased, capacity) = (number(usage, "/leased"), number(usage, "/capacity"));
    Row::meter(
        label,
        leased,
        capacity,
        format!(
            "{}/{} · peak {}",
            count(leased),
            count(capacity),
            count(number(usage, "/peak"))
        ),
    )
}

fn memory_section(model: &Value, activity: &Value, raw: &Value) -> Section {
    let limit = number(model, "/checkpoint/memory_plan/device_limit");
    let allocated = number(activity, "/device_allocated_bytes");
    let mut rows = vec![
        Row::meter(
            "State pool",
            number(activity, "/pool/leased_bytes"),
            number(activity, "/pool/resident_bytes"),
            format!(
                "{} / {}",
                bytes(number(activity, "/pool/leased_bytes")),
                bytes(number(activity, "/pool/resident_bytes")),
            ),
        ),
        usage("K/V pages", &activity["pool"]["kv_pages"]),
        usage("Recurrent", &activity["pool"]["recurrent_slots"]),
        Row::meter(
            "Device",
            allocated,
            limit,
            format!(
                "{} / {} · peak {}",
                bytes(allocated),
                bytes(limit),
                bytes(number(activity, "/device_allocated_peak_bytes")),
            ),
        ),
    ];
    if let Some(system) = raw.get("system").filter(|system| !system.is_null()) {
        let swapped = number(system, "/swapouts_since_start");
        rows.push(
            Row::new(
                "System",
                format!(
                    "{} spare · process {} · {} pages swapped",
                    bytes(number(system, "/spare_bytes")),
                    bytes(number(system, "/process_resident_bytes")),
                    count(swapped),
                ),
            )
            .alert(swapped > 0.0),
        );
    }
    Section {
        title: "Memory".into(),
        rows,
    }
}

fn percent(part: f64, whole: f64) -> String {
    if whole > 0.0 {
        format!("{:.0}%", part / whole * 100.0)
    } else {
        "-".into()
    }
}

fn efficiency_section(model: &Value, activity: &Value) -> Section {
    let sampled = number(activity, "/sampling/device_rows")
        + number(activity, "/sampling/host_rows")
        + number(activity, "/sampling/fallback_rows");
    let mut rows = vec![
        Row::new(
            "Prefix reuse",
            format!(
                "{} of lookups · {} tokens restored · {} snapshots",
                percent(
                    number(activity, "/prefix/hits"),
                    number(activity, "/prefix/lookups")
                ),
                count(number(activity, "/prefix/hit_tokens")),
                number(activity, "/prefix/entries"),
            ),
        ),
        Row::new(
            "Sampling",
            format!(
                "{} on device",
                percent(number(activity, "/sampling/device_rows"), sampled)
            ),
        ),
    ];
    if model["checkpoint"]["speculation"].as_bool() == Some(true) {
        rows.push(Row::new(
            "Speculation",
            format!(
                "{} of drafts accepted",
                percent(
                    number(activity, "/speculation/accepted"),
                    number(activity, "/speculation/drafted"),
                ),
            ),
        ));
    }
    Section {
        title: "Efficiency".into(),
        rows,
    }
}

/// Rates between successive snapshots. Counters only tell totals; what a person watching
/// wants is what the server does now.
#[derive(Debug, Default)]
pub(super) struct Rates {
    previous: Option<(Instant, f64, f64)>,
    pub(super) tokens_per_second: f64,
    pub(super) device_busy: f64,
    pub(super) history: VecDeque<u64>,
}

impl Rates {
    /// Generated tokens only count once a request finishes, so throughput reads tokens
    /// through passes, which move during a long request.
    pub(super) fn observe(&mut self, snapshot: &Snapshot, now: Instant) {
        let Some(activity) = snapshot.activity() else {
            return;
        };
        let tokens = number(activity, "/passes/tokens");
        let device = number(activity, "/passes/seconds/device");
        if let Some((then, last_tokens, last_device)) = self.previous {
            let elapsed = now.duration_since(then).as_secs_f64();
            if elapsed > 0.0 {
                self.tokens_per_second = (tokens - last_tokens).max(0.0) / elapsed;
                self.device_busy = ((device - last_device).max(0.0) / elapsed).min(1.0);
                if self.history.len() == HISTORY {
                    self.history.pop_front();
                }
                self.history.push_back(self.tokens_per_second as u64);
            }
        }
        self.previous = Some((now, tokens, device));
    }
}

#[cfg(test)]
pub(super) const SAMPLE: &str = r#"{
  "version": "1.2.3", "uptime_seconds": 125.0, "draining": false,
  "system": {"spare_bytes": 4294967296, "swapouts_since_start": 3, "process_resident_bytes": 1073741824},
  "models": {"tiny": {"alive": true,
    "checkpoint": {"source": "org/model", "revision": "0123456789abcdef", "weight_format": "q8_0",
      "kv": "f16/16", "recurrent_state": "f32", "speculation": true,
      "context": {"limit_tokens": 8192},
      "scheduler": {"max_running": 8, "max_waiting": 256},
      "memory_plan": {"device_limit": 8589934592}},
    "activity": {"running": 2, "waiting": 1, "requests": {"ok": 10, "cancelled": 1, "timeout": 2},
      "tokens": {"prompt": 1000, "cached_prompt": 250, "generated": 500}, "preemptions": 0,
      "latency": {
        "first_token_seconds": {"bounds": [0.1, 1.0], "buckets": [5, 10], "count": 10, "sum": 3.0},
        "request_seconds": {"bounds": [0.1, 1.0], "buckets": [0, 0], "count": 0, "sum": 0.0}},
      "pool": {"resident_bytes": 2147483648, "leased_bytes": 1073741824,
        "kv_pages": {"capacity": 100, "leased": 25, "peak": 30},
        "recurrent_slots": {"capacity": 8, "leased": 2, "peak": 3}},
      "device_allocated_bytes": 4294967296, "device_allocated_peak_bytes": 4294967296,
      "passes": {"count": 10, "tokens": 500, "seconds": {"device": 2.0}},
      "sampling": {"device_rows": 9, "host_rows": 1, "fallback_rows": 0},
      "prefix": {"entries": 2, "lookups": 4, "hits": 1, "hit_tokens": 64},
      "speculation": {"drafted": 10, "accepted": 4}}}},
  "transcription": {"whisper-1": {"alive": false, "leased_bytes": 1048576}}
}"#;

#[cfg(test)]
#[allow(clippy::panic_in_result_fn, reason = "tests assert parsed values")]
mod tests {
    use super::*;
    use std::time::Duration;
    use tur::Result;

    fn find<'a>(sections: &'a [Section], label: &str) -> Option<&'a Row> {
        sections
            .iter()
            .flat_map(|s| &s.rows)
            .find(|r| r.label == label)
    }

    #[test]
    fn test_sections_summarize_diagnostics() -> Result<()> {
        let snapshot = Snapshot::parse(SAMPLE)?;
        let sections = snapshot.sections(None);
        assert_eq!(snapshot.version(), "1.2.3");
        assert_eq!(snapshot.uptime(), "2m 05s");
        let requests = find(&sections, "Requests").map(|row| row.value.as_str());
        assert_eq!(requests, Some("2/8 running · 1 waiting"));
        let finished = find(&sections, "Finished").map(|row| (row.value.as_str(), row.alert));
        assert_eq!(finished, Some(("10 ok · 1 cancelled · 2 failed", true)));
        let tokens = find(&sections, "Tokens").map(|row| row.value.as_str());
        assert_eq!(tokens, Some("1000 prompt (25% reused) · 500 generated"));
        let first = find(&sections, "First token").map(|row| row.value.as_str());
        assert_eq!(first, Some("p50 ≤100 ms · p95 ≤1.0 s"));
        let unobserved = find(&sections, "Request").map(|row| row.value.as_str());
        assert_eq!(unobserved, Some("p50 - · p95 -"));
        assert_eq!(
            find(&sections, "K/V pages").and_then(|r| r.meter),
            Some(0.25)
        );
        assert!(find(&sections, "System").is_some_and(|row| row.alert));
        assert!(find(&sections, "Throughput").is_none());
        Ok(())
    }

    #[test]
    fn test_no_finished_requests_read_as_plain_zeros() -> Result<()> {
        let snapshot = Snapshot::parse(r#"{"models": {"m": {"activity": {"requests": {}}}}}"#)?;
        let sections = snapshot.sections(None);
        let finished = find(&sections, "Finished").map(|row| row.value.as_str());
        assert_eq!(finished, Some("0 ok · 0 cancelled · 0 failed"));
        Ok(())
    }

    #[test]
    fn test_a_stopped_worker_outranks_draining() -> Result<()> {
        assert_eq!(Snapshot::parse(SAMPLE)?.health(), Health::WorkerStopped);
        let draining = Snapshot::parse(r#"{"draining": true, "models": {}}"#)?;
        assert_eq!(draining.health(), Health::Draining);
        let ready = Snapshot::parse(r#"{"draining": false}"#)?;
        assert_eq!(ready.health(), Health::Ready);
        Ok(())
    }

    #[test]
    fn test_rates_read_counter_deltas_and_cap_device_busy() -> Result<()> {
        let body = |tokens: u64, device: f64| {
            format!(
                r#"{{"models": {{"m": {{"activity": {{"passes": {{"tokens": {tokens}, "seconds": {{"device": {device}}}}}}}}}}}}}"#
            )
        };
        let start = Instant::now();
        let mut rates = Rates::default();
        rates.observe(&Snapshot::parse(&body(100, 1.0))?, start);
        assert!(rates.history.is_empty());
        let later = start + Duration::from_secs(2);
        rates.observe(&Snapshot::parse(&body(300, 2.0))?, later);
        assert!((rates.tokens_per_second - 100.0).abs() < 1e-9);
        assert!((rates.device_busy - 0.5).abs() < 1e-9);
        let end = later + Duration::from_secs(1);
        rates.observe(&Snapshot::parse(&body(300, 9.0))?, end);
        assert!((rates.device_busy - 1.0).abs() < 1e-9);
        assert_eq!(rates.history, [100, 0]);
        Ok(())
    }
}
