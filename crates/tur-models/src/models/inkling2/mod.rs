//! Inkling 2: a looped hybrid decoder. The middle layers run several passes with shared
//! weights, each later pass reading messages from the previous one; attention residuals over
//! block states replace plain residual sums; attention uses learned relative position bias,
//! short convolutions and per-token sinks; routed experts are stored in NVFP4 and MXFP8.
mod config;
mod model;
pub use config::{Config, Stored};
pub use model::Inkling2;
