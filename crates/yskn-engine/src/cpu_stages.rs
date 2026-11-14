//! Exact CPU stages. Disable independently of the projection backend with
//! [`CpuStages::Reference`] (`Model::set_cpu_stages`) or, process-wide for the
//! engine's examples and explicit constructors, `YSKN_CPU_STAGES=reference` (D20).
use std::sync::OnceLock;

/// Whether the exact NEON fast paths around the projections run (D20). Both
/// choices give identical bits; `Reference` only runs the scalar stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuStages {
    /// The fast paths when run-time NEON detection succeeds, else the reference.
    Auto,
    Reference,
}

impl CpuStages {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Reference => "reference",
        }
    }
    /// Whether the fast paths run on this CPU.
    pub fn fast(self) -> bool {
        self == Self::Auto && neon_available()
    }
}

impl std::str::FromStr for CpuStages {
    type Err = crate::Error;
    fn from_str(value: &str) -> crate::Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "reference" => Ok(Self::Reference),
            _ => Err(crate::Error::InvalidInput(
                "CPU stages must be auto or reference".into(),
            )),
        }
    }
}

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

/// The process-wide default for a newly constructed model.
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

pub(crate) mod kv_codec;

pub(crate) mod math;
