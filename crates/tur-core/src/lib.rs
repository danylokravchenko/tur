#![forbid(unsafe_code)]
//! Shared errors and validated execution descriptors.
pub mod errors;
pub mod execution;
pub use errors::{Result, TurError};
