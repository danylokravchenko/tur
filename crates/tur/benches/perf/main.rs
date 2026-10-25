//! Every Tur benchmark and measurement tool in one binary; the first argument names the
//! case: `cargo bench --offline -p tur --bench perf -- <case> [args]`. Each case prints
//! JSON on stdout; see its module for the environment variables it reads.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmarks emit JSON artifacts and timing summaries"
)]

mod kernels;
mod kv_quality;
mod load_weights;
mod model;
mod partial_model;
mod prefix;
mod serving;
mod tensorops;

use tur_core::{Result, TurError};

const CASES: &str = "kernels | model | serving | prefix | kv_quality | layer_influence | neuron_influence | tensorops | load_weights <dir> <variant>";

fn main() -> Result<()> {
    // `cargo bench` passes `--bench`; every other argument belongs to the case.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench")
        .collect();
    let Some((case, rest)) = args.split_first() else {
        eprintln!("usage: perf <case>, one of: {CASES}");
        return Ok(());
    };
    match case.as_str() {
        "kernels" => kernels::run(),
        "model" => model::run(),
        "serving" => serving::run(),
        "prefix" => prefix::run(),
        "kv_quality" => kv_quality::run(),
        "layer_influence" => partial_model::run(rest, false),
        "neuron_influence" => partial_model::run(rest, true),
        "tensorops" => tensorops::run(),
        "load_weights" => load_weights::run(rest),
        _ => Err(TurError::Other(format!(
            "unknown case '{case}'; expected one of: {CASES}"
        ))),
    }
}
