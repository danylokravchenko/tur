#![forbid(unsafe_code)]
//! The `tur` command: one-shot generation, the HTTP server and offline weight preparation.
mod args;
mod quantize;
mod run;
mod serve;

use clap::{Parser, Subcommand};
use tur::Result;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Debug, Parser)]
#[command(author, version, about = "Tur hybrid text inference for Apple Silicon")]
struct Cli {
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
    match Cli::parse().command {
        Command::Run(args) => run::run(args),
        Command::Serve(args) => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(serve::serve(args)),
        Command::Quantize(args) => quantize::quantize(&args),
    }
}
