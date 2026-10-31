//! `tur status`, `tur top`, and the dashboard `tur serve` opens beside its server. All read
//! the same `/diagnostics` a remote operator would, so the view never disagrees with the API.
mod client;
mod snapshot;
mod stop;
mod view;

pub(crate) use client::Server;
pub(crate) use stop::{StopArgs, stop};

use crate::ui::Console;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    crossterm::{
        cursor::{Hide, Show},
        event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
};
use snapshot::{Rates, Section, Snapshot};
use std::{
    io::{IsTerminal as _, Stdout, Write as _},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tur::{Result, TurError};

/// How long one wait for a key lasts, which bounds how late a stop or redraw can be.
const KEY_POLL: Duration = Duration::from_millis(100);
/// Log lines the dashboard asks for; more than any screen shows.
const LOG_LINES: usize = 200;

#[derive(Debug, clap::Args)]
pub(crate) struct StatusArgs {
    /// Server address (default: `[server].listen` from tur.toml, or 127.0.0.1:8080).
    #[arg(long, env = "TUR_URL")]
    url: Option<String>,
    /// Print the raw `/diagnostics` JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TopArgs {
    /// Server address (default: `[server].listen` from tur.toml, or 127.0.0.1:8080).
    #[arg(long, env = "TUR_URL")]
    url: Option<String>,
    /// Seconds between readings.
    #[arg(long, default_value_t = 1.0)]
    interval: f64,
}

impl StatusArgs {
    pub(crate) fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }
}

impl TopArgs {
    pub(crate) fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }
}

fn read(server: &Server) -> Result<Snapshot> {
    let (status, body) = server.get("/diagnostics")?;
    if status != 200 {
        return Err(TurError::Other(format!(
            "{} answered /diagnostics with HTTP {status}",
            server.url()
        )));
    }
    Snapshot::parse(&body)
}

fn plain(sections: &[Section]) -> String {
    let mut out = String::new();
    for section in sections {
        out.push('\n');
        out.push_str(&section.title);
        out.push('\n');
        for row in &section.rows {
            let line = format!("  {:12} {}\n", row.label, row.value);
            out.push_str(&line);
        }
    }
    out
}

/// Print one reading. Fails unless the server would serve new requests, so scripts can gate
/// on the exit code.
pub(crate) fn status(args: &StatusArgs, server: &Server) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    if args.json {
        let (_, body) = server.get("/diagnostics")?;
        writeln!(stdout, "{body}")?;
    }
    let snapshot = read(server)?;
    let health = snapshot.health();
    if !args.json {
        writeln!(
            stdout,
            "tur {} at {}: {}, up {}",
            snapshot.version(),
            server.url(),
            health.label(),
            snapshot.uptime()
        )?;
        write!(stdout, "{}", plain(&snapshot.sections(None)))?;
    }
    match health {
        snapshot::Health::Ready => Ok(()),
        other => Err(TurError::Other(format!(
            "server is not ready: {}",
            other.label()
        ))),
    }
}

/// Raw mode on an alternate screen, restored however the dashboard ends.
struct Screen(Terminal<CrosstermBackend<Stdout>>);

impl Screen {
    fn open() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide)?;
        Ok(Self(Terminal::new(CrosstermBackend::new(stdout))?))
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        // Nothing more can be done about a terminal that refuses to be restored.
        drop(execute!(self.0.backend_mut(), LeaveAlternateScreen, Show));
        drop(disable_raw_mode());
    }
}

/// What the dashboard is attached to.
struct Session<'a> {
    server: &'a Server,
    interval: Duration,
    keys: &'a str,
    /// In `tur serve`: its console's logs, a stop flag and the way to ask for a drain.
    embedded: Option<Embedded<'a>>,
}

struct Embedded<'a> {
    console: &'a Console,
    stop: &'a AtomicBool,
    drain: &'a mpsc::Sender<()>,
}

/// Why a key ended the dashboard, if one did.
fn pressed(embedded: Option<&Embedded<'_>>) -> Result<bool> {
    if !event::poll(KEY_POLL)? {
        return Ok(false);
    }
    let Event::Key(key) = event::read()? else {
        return Ok(false);
    };
    if key.kind == KeyEventKind::Release {
        return Ok(false);
    }
    let interrupt = key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
    match embedded {
        // Raw mode turned Ctrl-C into this key, so the signal it stands for is sent by hand.
        // A full channel already holds both signals a drain listens for.
        Some(embedded) if interrupt => {
            if embedded.drain.try_send(()).is_err() {
                tracing::debug!("drain already requested twice");
            }
            Ok(false)
        }
        None if interrupt => Ok(true),
        _ => Ok(matches!(key.code, KeyCode::Char('q' | 'Q') | KeyCode::Esc)),
    }
}

/// Draw until a key closes the dashboard or the embedded server stops.
fn dashboard(session: &Session<'_>) -> Result<()> {
    let mut screen = Screen::open()?;
    let url = session.server.url();
    let mut rates = Rates::default();
    let (mut snapshot, mut problem) = (None, None::<String>);
    let mut due = Instant::now();
    loop {
        if session
            .embedded
            .as_ref()
            .is_some_and(|embedded| embedded.stop.load(Ordering::Relaxed))
        {
            return Ok(());
        }
        let now = Instant::now();
        if now >= due {
            due = now + session.interval;
            match read(session.server) {
                Ok(reading) => {
                    rates.observe(&reading, now);
                    snapshot = Some(reading);
                    problem = None;
                }
                Err(error) => problem = Some(crate::ui::describe(&error)),
            }
        }
        let logs = session
            .embedded
            .as_ref()
            .map(|embedded| embedded.console.recent_logs(LOG_LINES));
        screen.0.draw(|frame| {
            view::draw(
                frame,
                &view::View {
                    url: &url,
                    snapshot: snapshot.as_ref(),
                    rates: &rates,
                    problem: problem.as_deref(),
                    logs: logs.as_deref(),
                    keys: session.keys,
                },
            );
        })?;
        if pressed(session.embedded.as_ref())? {
            return Ok(());
        }
    }
}

fn all_terminals() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

/// Watch a server from another terminal.
pub(crate) fn top(args: &TopArgs, server: &Server) -> Result<()> {
    if !all_terminals() {
        return Err(TurError::Other(
            "tur top needs a terminal; tur status --json suits scripts".into(),
        ));
    }
    if !(args.interval.is_finite() && args.interval >= 0.1) {
        return Err(TurError::Other("--interval must be at least 0.1".into()));
    }
    // Fail plainly before taking over the screen when nothing is listening.
    read(server)?;
    dashboard(&Session {
        server,
        interval: Duration::from_secs_f64(args.interval),
        keys: "q quit",
        embedded: None,
    })
}

/// What `tur serve` shows once its listener is open.
pub(crate) struct Attachment {
    pub(crate) console: Console,
    pub(crate) server: Server,
    pub(crate) dashboard: bool,
    pub(crate) drain_seconds: u64,
    pub(crate) drain: mpsc::Sender<()>,
    pub(crate) stop: Arc<AtomicBool>,
}

fn banner(attachment: &Attachment) {
    let console = &attachment.console;
    let url = attachment.server.url();
    console.line("");
    console.line(&format!(
        "  {} {}",
        console.paint("32", "●"),
        console.paint("1", &format!("Tur is serving on {url}"))
    ));
    console.rows(&[
        ("API", format!("POST {url}/v1/chat/completions")),
        (
            "Operate",
            format!("{url}/health /ready /metrics /diagnostics"),
        ),
        ("Watch", "tur status · tur top · tur stop".to_owned()),
        (
            "Stop",
            format!(
                "Ctrl-C finishes requests for up to {} s; a second one stops now",
                attachment.drain_seconds
            ),
        ),
    ]);
    console.line("");
}

/// Announce the server, then show the dashboard until it closes or the server stops. Blocks,
/// and only ever affects what is displayed: the server runs whatever happens here.
pub(crate) fn attach(attachment: &Attachment) {
    let console = &attachment.console;
    console.stop_progress();
    if !console.interactive() {
        return;
    }
    banner(attachment);
    if !attachment.dashboard {
        return;
    }
    if !all_terminals() {
        tracing::warn!("no dashboard: stdin and stdout are not both a terminal");
        return;
    }
    console.capture_logs();
    let shown = dashboard(&Session {
        server: &attachment.server,
        interval: Duration::from_secs(1),
        keys: "q close dashboard, server keeps running · Ctrl-C drain and stop",
        embedded: Some(Embedded {
            console,
            stop: &attachment.stop,
            drain: &attachment.drain,
        }),
    });
    console.release_logs();
    match shown {
        Ok(()) if attachment.stop.load(Ordering::Relaxed) => {}
        Ok(()) => console.line("  Dashboard closed; logs follow here. Ctrl-C stops the server."),
        Err(error) => tracing::warn!("dashboard failed: {}", crate::ui::describe(&error)),
    }
}
