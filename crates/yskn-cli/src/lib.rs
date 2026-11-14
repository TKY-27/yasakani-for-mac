//! Chat and local HTTP product layer. Engines stay on their owning thread.
#![forbid(unsafe_code)]

pub mod chat;
pub mod cli;
pub mod engine;
pub mod loopguard;
pub mod request;
pub mod server;
pub mod stub;

use std::fmt;

#[derive(Debug, Clone)]
pub struct Error {
    pub message: String,
    pub status: u16,
    pub kind: &'static str,
    pub code: Option<&'static str>,
}
impl Error {
    pub fn request(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: 400,
            kind: "invalid_request_error",
            code: None,
        }
    }
    pub fn engine(message: impl fmt::Display) -> Self {
        Self {
            message: message.to_string(),
            status: 500,
            kind: "server_error",
            code: None,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
impl From<yskn_text::Error> for Error {
    fn from(value: yskn_text::Error) -> Self {
        Self::engine(value.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;
