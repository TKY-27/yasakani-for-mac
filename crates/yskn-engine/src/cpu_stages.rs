//! Exact CPU stages. Disable independently of the projection backend with YSKN_CPU_STAGES=reference.
use std::sync::OnceLock;

fn selection() -> &'static Result<bool, &'static str> {
    static SELECTION: OnceLock<Result<bool, &'static str>> = OnceLock::new();
    SELECTION.get_or_init(|| match std::env::var("YSKN_CPU_STAGES").as_deref() {
        Ok("reference") => Ok(false),
        Ok("auto") | Err(std::env::VarError::NotPresent) => Ok(neon_available()),
        _ => Err("YSKN_CPU_STAGES must be auto or reference"),
    })
}

pub(crate) fn validate() -> crate::Result<()> {
    selection()
        .map(|_| ())
        .map_err(|message| crate::Error::InvalidInput(message.into()))
}

pub(crate) fn enabled() -> bool {
    // Model construction rejects invalid configuration before opening weights.
    selection().as_ref().copied().unwrap_or(false)
}

fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub(crate) mod attention;
pub(crate) mod recurrence;
pub(crate) mod rest;
