//! Device resource ownership, attention storage, packed weights and Metal kernels.
pub mod completion;
#[cfg(feature = "metal")]
pub mod kernels;
pub mod packed;
pub mod scratch;
pub mod state;
