//! Zero-copy mapped weights, persistent CPU compressed-linear workers, the
//! Metal compressed-linear kernel (macOS only) and layer-major main-model inference.
#![deny(unsafe_op_in_unsafe_fn)]

mod cpu_stages;
mod draft;
mod draft_math;
mod kernel;
pub mod machine;
mod mapped;
#[cfg(target_os = "macos")]
mod metal;
mod model;
mod pool;

pub use cpu_stages::CpuStages;
pub use draft::{DraftTiming, FastDraft};
pub use kernel::Kernel;
pub use machine::Machine;
pub use mapped::{MappedModel, Matrix};
#[cfg(target_os = "macos")]
pub use metal::{Job, MetalEngine};
pub use model::{
    Backend, CpuShare, LogitRows, MetalOptions, Model, PassTiming, SessionMemory, Snapshot,
};
pub use pool::Engine;
pub use yskn_ref::{Error, KvMode, Result};
