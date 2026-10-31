#![forbid(unsafe_code)]
//! Shared errors and validated execution descriptors.
pub mod errors;
pub mod execution;
pub mod progress;
pub use errors::{Result, TurError};
