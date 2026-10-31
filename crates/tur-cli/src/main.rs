#![forbid(unsafe_code)]
//! The `tur` command: generation, the HTTP server and its monitor, and weight preparation.
mod args;
mod config;
mod monitor;
mod prune;
mod quantize;
mod run;
mod serve;
mod transcribe;
mod ui;

use clap::{Parser, Subcommand};
use std::{io::Write as _, path::PathBuf, process::ExitCode};
use tur::{Result, TurError};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Tur hybrid text inference for Apple Silicon",
    after_help = "Examples:\n  \
        tur run --model ./model --revision local --prompt 'Hello'\n  \
        tur serve --model ./model --revision local\n  \
        tur top                 watch a running server\n  \
        tur status --json       one reading, for scripts"
)]
struct Cli {
    /// Read this TOML file instead of ./tur.toml (not for quantize and prune).
    #[arg(long, global = true, conflicts_with = "no_config")]
    config: Option<PathBuf>,
    /// Ignore ./tur.toml and use only flags, environment and built-in defaults.
    #[arg(long, global = true)]
    no_config: bool,
    /// Show only warnings, errors and the result: no progress or summaries.
    #[arg(long, short, global = true)]
    quiet: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate one completion: the answer to stdout, any separate reasoning to stderr.
    Run(run::Args),
    /// Serve one resident model over an OpenAI-style HTTP API, with a live dashboard.
    Serve(serve::Args),
    /// Print a running server's state once. Exits nonzero unless it is ready.
    Status(monitor::StatusArgs),
    /// Watch a running server's traffic, memory and efficiency live.
    Top(monitor::TopArgs),
    /// Stop a server on this machine, finishing its requests first, and wait until it exits.
    Stop(monitor::StopArgs),
    /// Prepack weights offline, optionally calibrated, for direct loading.
    Quantize(quantize::Args),
    /// Write a gpt-oss checkpoint with fewer experts, keeping those each layer can least lose.
    Prune(prune::Args),
    /// Transcribe a WAV file with a Whisper checkpoint and print the text.
    Transcribe(transcribe::Args),
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().collect();
    let cli = match Cli::try_parse_from(&arguments) {
        Ok(cli) => cli,
        Err(error) => return parse_failure(&error, &arguments),
    };
    let console = match ui::init(
        cli.quiet,
        cli.command.as_ref().map_or("warn", Command::log_level),
    ) {
        Ok(console) => console,
        Err(error) => {
            // Logging is what failed, so stderr is all there is.
            report(&format!("error: {}", ui::describe(&error)));
            return ExitCode::FAILURE;
        }
    };
    match execute(cli, &console) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let label = console.paint("1;31", "error:");
            report(&format!("{label} {}", ui::describe(&error)));
            ExitCode::FAILURE
        }
    }
}

/// Print clap's message, which is also how help and version arrive, with the way forward when
/// flags were given without the command they belong to.
fn parse_failure(error: &clap::Error, arguments: &[String]) -> ExitCode {
    // Nothing depends on messages reaching a closed terminal.
    drop(error.print());
    if !error.use_stderr() {
        return ExitCode::SUCCESS;
    }
    if lacks_command(error, arguments) {
        report(
            "\nThese flags belong to a command, for example:\n  \
             tur run --model <dir or org/name> --revision <rev> --prompt 'Hello'\n  \
             tur serve --model <dir or org/name> --revision <rev>\n\n\
             Run `tur` to list every command.",
        );
    }
    ExitCode::FAILURE
}

fn lacks_command(error: &clap::Error, arguments: &[String]) -> bool {
    use clap::CommandFactory as _;

    let command = Cli::command();
    error.kind() == clap::error::ErrorKind::UnknownArgument
        && !arguments.iter().skip(1).any(|argument| {
            command
                .get_subcommands()
                .any(|subcommand| subcommand.get_name() == argument)
        })
}

fn report(message: &str) {
    // The failure is already being reported by the exit code if stderr is closed.
    drop(writeln!(std::io::stderr(), "{message}"));
}

/// The server a monitoring command talks to: `url`, else what tur.toml has it listen on.
fn monitored(cli_url: Option<&str>, settings: &config::FileSettings) -> Result<monitor::Server> {
    cli_url.map_or_else(
        || {
            Ok(monitor::Server::listening_on(
                settings.server.listen_address(),
            ))
        },
        monitor::Server::parse,
    )
}

fn execute(cli: Cli, console: &ui::Console) -> Result<()> {
    let settings = || config::FileSettings::load(cli.config.as_deref(), cli.no_config);
    let Some(command) = cli.command else {
        return introduce(console);
    };
    match command {
        Command::Run(mut args) => {
            args.configure(settings()?);
            run::run(args, console)
        }
        Command::Serve(mut args) => {
            args.configure(settings()?);
            ui::welcome(console, &tagline());
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(serve::serve(args, console.clone()))
        }
        Command::Status(args) => monitor::status(&args, &monitored(args.url(), &settings()?)?),
        Command::Top(args) => monitor::top(&args, &monitored(args.url(), &settings()?)?),
        Command::Stop(args) => monitor::stop(&args, &monitored(args.url(), &settings()?)?, console),
        Command::Transcribe(mut args) => {
            args.configure(settings()?);
            transcribe::transcribe(&args, console)
        }
        Command::Prune(mut args) => {
            job_arguments_only(cli.config.as_deref(), "prune")?;
            ui::welcome(console, &tagline());
            prune::run(&mut args, console)
        }
        Command::Quantize(mut args) => {
            job_arguments_only(cli.config.as_deref(), "quantize")?;
            ui::welcome(console, &tagline());
            quantize::quantize(&mut args, console)
        }
    }
}

fn tagline() -> String {
    format!(
        "{} · inference for Apple Silicon",
        env!("CARGO_PKG_VERSION")
    )
}

/// Bare `tur`: the welcome and everything it can do. Piped, the plain help is more useful.
fn introduce(console: &ui::Console) -> Result<()> {
    use clap::CommandFactory as _;

    let mut help = Cli::command();
    if !console.interactive() {
        help.print_help()?;
        return Ok(());
    }
    ui::welcome(console, &tagline());
    let rows: Vec<(String, String)> = help
        .get_subcommands()
        .filter(|command| command.get_name() != "help")
        .map(|command| {
            let about = command.get_about().map(ToString::to_string);
            (command.get_name().to_owned(), about.unwrap_or_default())
        })
        .collect();
    let rows: Vec<(&str, String)> = rows.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    console.rows(&rows);
    console.line("\n  tur <command> --help prints every flag.");
    Ok(())
}

fn job_arguments_only(config: Option<&std::path::Path>, command: &str) -> Result<()> {
    if config.is_some() {
        return Err(TurError::Other(format!(
            "--config does not apply to {command}, which takes explicit job arguments"
        )));
    }
    Ok(())
}

impl Command {
    /// Where logging starts in a terminal. Progress and summaries already narrate one-off
    /// work, so only a server's log is interesting at `info`.
    fn log_level(&self) -> &'static str {
        if matches!(self, Self::Serve(_)) {
            "info"
        } else {
            "warn"
        }
    }
}

#[cfg(test)]
fn parse_cli(arguments: &[&str]) -> Result<Cli> {
    use clap::{CommandFactory as _, FromArgMatches as _};

    // Tests must not inherit the developer's selected checkpoint.
    let mut command = Cli::command();
    for name in ["run", "serve"] {
        command = command.mut_subcommand(name, |command| {
            command
                .mut_arg("model", |arg| arg.env(None::<&str>))
                .mut_arg("revision", |arg| arg.env(None::<&str>))
        });
    }
    for name in ["status", "top", "stop"] {
        command = command.mut_subcommand(name, |command| {
            command.mut_arg("url", |arg| arg.env(None::<&str>))
        });
    }
    let matches = command
        .try_get_matches_from(arguments)
        .map_err(|error| TurError::Other(error.to_string()))?;
    Cli::from_arg_matches(&matches).map_err(|error| TurError::Other(error.to_string()))
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert fallible CLI parsing"
)]
mod tests {
    use super::*;

    #[test]
    fn test_config_flags_before_and_after_subcommand() -> Result<()> {
        for arguments in [
            vec!["tur", "--config", "custom.toml", "serve"],
            vec!["tur", "serve", "--config", "custom.toml"],
        ] {
            assert_eq!(
                parse_cli(&arguments)?.config,
                Some(PathBuf::from("custom.toml"))
            );
        }
        assert!(parse_cli(&["tur", "run", "--config", "x", "--no-config"]).is_err());
        Ok(())
    }

    #[test]
    fn test_monitor_address_comes_from_flag_then_file_then_default() -> Result<()> {
        let default = config::FileSettings::default();
        assert_eq!(monitored(None, &default)?.url(), "http://127.0.0.1:8080");
        let settings: config::FileSettings = toml::from_str("[server]\nlisten = '0.0.0.0:9000'")
            .map_err(|error| TurError::Other(error.to_string()))?;
        assert_eq!(monitored(None, &settings)?.url(), "http://127.0.0.1:9000");
        let settings = config::FileSettings::default();
        assert_eq!(monitored(Some("h:1"), &settings)?.url(), "http://h:1");
        let Some(Command::Top(args)) = parse_cli(&["tur", "top", "--url", "h:1"])?.command else {
            return Err(TurError::Other("expected top".into()));
        };
        assert_eq!(args.url(), Some("h:1"));
        assert!(parse_cli(&["tur", "-q", "status", "--json"])?.quiet);
        assert!(matches!(
            parse_cli(&["tur", "stop", "--force"])?.command,
            Some(Command::Stop(_))
        ));
        Ok(())
    }

    #[test]
    fn test_flags_without_a_command_get_directions() -> Result<()> {
        let failure = |values: &[&str]| -> Result<bool> {
            let arguments: Vec<String> = values.iter().map(ToString::to_string).collect();
            let error = Cli::try_parse_from(&arguments)
                .err()
                .ok_or_else(|| TurError::Other("arguments were accepted".into()))?;
            Ok(lacks_command(&error, &arguments))
        };
        assert!(failure(&["tur", "--model", "m", "--revision", "r"])?);
        assert!(!failure(&["tur", "serve", "--nonsense"])?);
        Ok(())
    }

    #[test]
    fn test_quantize_still_requires_explicit_job_arguments() -> Result<()> {
        assert!(parse_cli(&["tur", "quantize"]).is_err());
        assert!(parse_cli(&["tur"])?.command.is_none());
        assert!(matches!(
            parse_cli(&[
                "tur",
                "quantize",
                "--model",
                "source",
                "--revision",
                "id",
                "--output",
                "dest"
            ])?
            .command,
            Some(Command::Quantize(_))
        ));
        Ok(())
    }
}
