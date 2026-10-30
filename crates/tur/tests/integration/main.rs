//! Workspace integration tests, linked into one binary: kernels, models and the engine
//! on small fixtures, then the pipeline, scheduler and HTTP server through the facade.
#![allow(
    clippy::panic_in_result_fn,
    reason = "tests assert outcomes after fallible setup"
)]

mod audio;
mod common;
mod contracts;
mod engine;
mod fused;
mod gemma4;
mod kernels;
mod model_common;
mod moe;
mod pipeline;
mod prefix_reuse;
mod prepacked;
mod quantization;
mod reference;
mod scheduling;
mod scheduling_metal;
mod server;
mod serving;
mod speculation;
