//! Metal compute backend probe for the compressed linear operator.
//!
//! macOS only; on every other target this crate is an empty library.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "macos")]
mod metal;

#[cfg(target_os = "macos")]
pub use metal::{Job, MetalEngine};
