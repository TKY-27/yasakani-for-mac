//! Scalar implementation of [chapters 01–06](../../../docs/spec/01_MODEL_ARCHITECTURE.md).
#![forbid(unsafe_code)]

pub mod attention;
pub mod compressed_linear;
pub mod gated_deltanet;
pub mod model;
pub mod output_head_sampler;
pub mod portable_math;
pub mod residual_norm_mlp;

use std::fmt;

pub use model::{Model, ModelDimensions};
pub use output_head_sampler::{Sampler, SamplingParameters, SamplingVersion};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Format(yskn_format::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidModel(String),
    InvalidInput(String),
    NonFinite(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::InvalidModel(message) => write!(formatter, "invalid model: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid input: {message}"),
            Self::NonFinite(stage) => write!(formatter, "non-finite value in {stage}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<yskn_format::Error> for Error {
    fn from(error: yskn_format::Error) -> Self {
        Self::Format(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub(crate) fn require_finite(values: &[f32], stage: &'static str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFinite(stage));
    }
    Ok(())
}
