//! Zero-copy mapped weights, persistent CPU compressed-linear workers, the
//! Metal compressed-linear kernel (macOS only) and layer-major main-model inference.
#![deny(unsafe_op_in_unsafe_fn)]

mod kernel;
mod mapped;
#[cfg(target_os = "macos")]
mod metal;
mod model;
mod pool;

pub use kernel::Kernel;
pub use mapped::{MappedModel, Matrix};
#[cfg(target_os = "macos")]
pub use metal::{Job, MetalEngine};
pub use model::{Backend, Model, PassTiming};
pub use pool::Engine;
pub use yskn_ref::{Error, Result};
