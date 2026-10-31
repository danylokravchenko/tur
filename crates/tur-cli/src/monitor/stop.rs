//! `tur stop`: signal a local server's process, which drains exactly as Ctrl-C does. A signal
//! rather than an HTTP route, because the API port has no authentication and allows any origin:
//! a shutdown route would let any web page stop the server.
use super::client::Server;
use crate::ui::Console;
use serde_json::Value;
use std::{path::Path, process::Command, time::Duration};
use tur::{Result, TurError, core::progress::Stage};

/// How often to check whether the server has exited.
const EXIT_POLL: Duration = Duration::from_millis(100);

#[derive(Debug, clap::Args)]
pub(crate) struct StopArgs {
    /// Server address (default: `[server].listen` from tur.toml, or 127.0.0.1:8080).
    #[arg(long, env = "TUR_URL")]
    url: Option<String>,
    /// Fail running requests and stop at once, instead of letting them finish.
    #[arg(long)]
    force: bool,
}

impl StopArgs {
    pub(crate) fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }
}

/// The process a `/diagnostics` body names. Rejects what `kill` would read as a process group
/// or every process.
fn server_pid(body: &str) -> Result<u32> {
    let diagnostics: Value = serde_json::from_str(body)?;
    diagnostics["pid"]
        .as_u64()
        .and_then(|pid| u32::try_from(pid).ok())
        .filter(|&pid| pid > 1)
        .ok_or_else(|| TurError::Other("the server did not report a usable process ID".into()))
}

/// Whether `command`, as `ps` reports it, is a Tur binary.
fn is_tur(command: &str) -> bool {
    Path::new(command.trim())
        .file_name()
        .is_some_and(|name| name == "tur")
}

/// Refuse a PID that is not a Tur server here. A forwarded port can make a server in a
/// container or over a tunnel look local, and its PID then means something else on this machine.
fn verify(pid: u32) -> Result<()> {
    let output = Command::new("ps")
        .args(["-o", "comm=", "-p", &pid.to_string()])
        .output()?;
    if is_tur(&String::from_utf8_lossy(&output.stdout)) {
        Ok(())
    } else {
        Err(TurError::Other(format!(
            "process {pid} is not a tur process on this machine; is the port forwarded?"
        )))
    }
}

fn signal(pid: u32) -> Result<()> {
    let output = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(TurError::Other(format!(
            "cannot signal process {pid}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

fn alive(pid: u32) -> Result<bool> {
    Ok(Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()?
        .success())
}

/// Stop the server and wait for its process to exit, which is when its device memory is free.
/// The listener closes as a drain begins, so only the process can say when it has finished.
/// The server bounds its own drain, so this wait ends too.
pub(crate) fn stop(args: &StopArgs, server: &Server, console: &Console) -> Result<()> {
    if !server.is_local()? {
        return Err(TurError::Other(format!(
            "tur stop only reaches a server on this machine, not {}",
            server.url()
        )));
    }
    let (_, body) = server.get("/diagnostics")?;
    let pid = server_pid(&body)?;
    verify(pid)?;
    signal(pid)?;
    // The second signal is what ends a drain early, as a second Ctrl-C does.
    if args.force {
        signal(pid)?;
    }
    let waiting = Stage::begin(
        if args.force {
            "Stopping"
        } else {
            "Finishing requests"
        },
        0,
        "",
    );
    while alive(pid)? {
        std::thread::sleep(EXIT_POLL);
    }
    waiting.finish();
    console.line(&format!(
        "  {} stopped the server at {}",
        console.paint("32", "✓"),
        server.url()
    ));
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible process control"
)]
mod tests {
    use super::*;

    #[test]
    fn test_only_an_ordinary_process_id_is_accepted() -> Result<()> {
        assert_eq!(server_pid(r#"{"pid": 4242}"#)?, 4242);
        // `kill` reads 0 and negatives as process groups, and -1 as every process.
        for body in [
            r#"{"pid": 0}"#,
            r#"{"pid": 1}"#,
            r#"{"pid": -1}"#,
            r#"{"pid": 99999999999}"#,
            r#"{"pid": "12"}"#,
            "{}",
        ] {
            assert!(server_pid(body).is_err(), "accepted {body}");
        }
        Ok(())
    }

    #[test]
    fn test_only_a_tur_binary_passes_for_a_server() {
        assert!(is_tur("tur\n"));
        assert!(is_tur("/usr/local/bin/tur"));
        assert!(!is_tur("target/release/nginx"));
        assert!(!is_tur("turbo"));
        assert!(!is_tur(""));
    }

    #[test]
    fn test_signal_ends_a_process_and_refuses_a_foreign_pid() -> Result<()> {
        let mut child = Command::new("sleep").arg("30").spawn()?;
        let pid = child.id();
        assert!(alive(pid)?);
        assert!(verify(pid).is_err(), "sleep passed for a tur server");
        signal(pid)?;
        child.wait()?;
        assert!(!alive(pid)?);
        assert!(signal(pid).is_err());
        Ok(())
    }
}
