//! Device resource ownership, paged state pools, packed weights and Metal kernels.
pub mod completion;
#[cfg(feature = "metal")]
pub mod kernels;
pub mod packed;
pub mod pool;
pub mod sampling;
pub mod scratch;
pub mod state;
