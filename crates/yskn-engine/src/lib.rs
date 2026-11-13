//! Zero-copy mapped weights and persistent CPU compressed-linear workers.
#![deny(unsafe_op_in_unsafe_fn)]

mod kernel;
mod mapped;
mod pool;

pub use kernel::Kernel;
pub use mapped::{MappedModel, Matrix};
pub use pool::Engine;
pub use yskn_ref::{Error, Result};
