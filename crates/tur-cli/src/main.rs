#![forbid(unsafe_code)]
//! The `tur` command: one-shot generation, the HTTP server and offline weight preparation.
mod args;
mod config;
mod quantize;
mod run;
mod serve;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tur::{Result, TurError};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Debug, Parser)]
#[command(author, version, about = "Tur hybrid text inference for Apple Silicon")]
struct Cli {
    /// Read this TOML file instead of ./tur.toml (run and serve only).
    #[arg(long, global = true, conflicts_with = "no_config")]
    config: Option<PathBuf>,
    /// Ignore ./tur.toml and use only flags, environment and built-in defaults.
    #[arg(long, global = true)]
    no_config: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate one completion and stream it to stdout.
    Run(run::Args),
    /// Serve one resident model over an OpenAI-style HTTP API.
    Serve(serve::Args),
    /// Prepack weights offline, optionally calibrated, for direct loading.
    Quantize(quantize::Args),
}

fn main() -> Result<()> {
    tur::shared::init_tracing()?;
    let cli = Cli::parse();
    match cli.command {
        Command::Run(mut args) => {
            args.configure(config::FileSettings::load(
                cli.config.as_deref(),
                cli.no_config,
            )?);
            run::run(args)
        }
        Command::Serve(mut args) => {
            args.configure(config::FileSettings::load(
                cli.config.as_deref(),
                cli.no_config,
            )?);
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(serve::serve(args))
        }
        Command::Quantize(args) => {
            if cli.config.is_some() {
                return Err(TurError::Other(
                    "--config applies only to run and serve; quantize uses explicit job arguments"
                        .into(),
                ));
            }
            quantize::quantize(&args)
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
    fn test_quantize_still_requires_explicit_job_arguments() -> Result<()> {
        assert!(parse_cli(&["tur", "quantize"]).is_err());
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
            Command::Quantize(_)
        ));
        Ok(())
    }
}
