//! Start-up checks and the effective run configuration on this machine (D23).
//! Every check only reads facts, never changes a system setting, and fails
//! closed: a fact that cannot be read is an error, not a pass.
use std::{fs, sync::Once};

use serde_json::{Value, json};
use yskn_engine::{Backend, CpuShare, CpuStages, Kernel, Machine, MetalOptions};

use crate::{Error, Result, real_engine::RealEngineConfig};

/// The oldest supported macOS major version.
pub const MIN_MACOS: u32 = 26;

/// Apple silicon (macOS on aarch64) and macOS 26 or later; `macos` is
/// `kern.osproductversion` (`None` when it cannot be read).
pub fn check_platform(arch: &str, os: &str, macos: Option<&str>) -> Result<()> {
    if os != "macos" || arch != "aarch64" {
        return Err(Error::engine(format!(
            "yskn requires a Mac with Apple silicon (macOS on aarch64); this program runs on {os} {arch}"
        )));
    }
    let Some(version) = macos else {
        return Err(Error::engine(format!(
            "yskn requires macOS {MIN_MACOS} or later and cannot read this Mac's version (sysctl kern.osproductversion)"
        )));
    };
    let major = version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
        .ok_or_else(|| {
            Error::engine(format!(
                "yskn requires macOS {MIN_MACOS} or later and cannot read the version {version:?}"
            ))
        })?;
    // macOS 26 can report 16.x to programs built with an older SDK (as macOS 11
    // reported 10.16); no macOS 16 exists, so 16 is read as 26.
    let major = if major == 16 { 26 } else { major };
    if major < MIN_MACOS {
        return Err(Error::engine(format!(
            "yskn requires macOS {MIN_MACOS} or later; this Mac runs macOS {version}"
        )));
    }
    Ok(())
}

/// The checks for this process: its architecture and the running macOS.
pub fn check_this_platform() -> Result<()> {
    check_platform(
        std::env::consts::ARCH,
        std::env::consts::OS,
        Machine::detect().macos_version.as_deref(),
    )
}

/// Every file the configuration reads, before anything is opened or mapped;
/// one error lists all missing ones and which option needs each.
pub fn check_folder(config: &RealEngineConfig) -> Result<()> {
    let folder = &config.model_dir;
    let manifest_path = folder.join("yasakani.json");
    let manifest: Value = match fs::read(&manifest_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| Error::engine(format!("{}: {e}", manifest_path.display())))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
        Err(e) => return Err(Error::engine(format!("{}: {e}", manifest_path.display()))),
    };
    // The same defaults as yskn_ref's manifest resolution; it validates the names.
    let file = |key: &str, default: &'static str| {
        manifest["files"][key]
            .as_str()
            .unwrap_or(default)
            .to_owned()
    };
    let mut needed = vec![
        ("yasakani.json".to_owned(), "the model manifest".to_owned()),
        (
            file("weights", "qwen38_27b_e3x.yskn"),
            "files.weights, the main model".to_owned(),
        ),
    ];
    if config.kv != yskn_engine::KvMode::Int8 {
        needed.push((
            file("kv_profile", "kvq_u5.bin"),
            format!(
                "files.kv_profile, for --kv {}; --kv int8 runs without it",
                config.kv_name()
            ),
        ));
    }
    if config.draft.neural() {
        let why = format!(
            "for --draft {}; --draft ngram or none runs without it",
            config.draft.name()
        );
        needed.push((
            file("draft", "bdraft2_k8.yskn"),
            format!("files.draft, the neural draft, {why}"),
        ));
        needed.push((
            file("draft_head", "bdraft2_k8.shead"),
            format!("files.draft_head, the draft head, {why}"),
        ));
    }
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        needed.push((name.to_owned(), "the tokenizer".to_owned()));
    }
    let missing: Vec<String> = needed
        .into_iter()
        .filter(|(name, _)| !folder.join(name).is_file())
        .map(|(name, why)| format!("{name} ({why})"))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Error::engine(format!(
            "model folder {} is incomplete: missing {}",
            folder.display(),
            missing.join(", ")
        )))
    }
}

/// The effective switches of one run (`yskn info` shows them).
#[derive(Debug, Clone)]
pub struct RunChoice {
    pub backend: Backend,
    /// The Metal device name, or the CPU kernel and worker count.
    pub device: String,
    /// Why the default selection fell back to the CPU backend.
    pub fallback: Option<String>,
    pub kernel: Kernel,
    pub threads: usize,
    pub cpu_stages: CpuStages,
    pub metal: MetalOptions,
    /// Whether `CpuShare::Auto` runs the calibrated table on this machine.
    pub share_table: bool,
    pub max_buffer_length: Option<usize>,
    pub recommended_working_set: Option<u64>,
    pub machine: Machine,
    /// Where the adaptive depth's cost table comes from (D21).
    pub draft_cost_table: String,
}

/// What a working Metal engine reports, or why there is none.
struct Probe {
    name: String,
    share_table: bool,
    max_buffer_length: usize,
    working_set: u64,
}

fn probe_metal(threads: usize) -> std::result::Result<Probe, String> {
    #[cfg(target_os = "macos")]
    {
        match yskn_engine::MetalEngine::new() {
            Ok(Some(metal)) => Ok(Probe {
                name: metal.device_name(),
                share_table: metal.share_table_applies(threads),
                max_buffer_length: metal.max_buffer_length(),
                working_set: metal.recommended_working_set(),
            }),
            Ok(None) => Err("no Metal device".into()),
            Err(error) => Err(error.to_string()),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = threads;
        Err("Metal requires macOS".into())
    }
}

/// The one stderr line of a default selection without Metal; the cost is the
/// CPU kernel's relative to Metal, measured on the M5 (D12).
fn fallback_notice(why: &str) -> String {
    format!(
        "yskn: Metal is unavailable ({why}); using the CPU backend, whose compressed projections take about 6 times as long (646 instead of 111 ms per pass at one row on an Apple M5, D12). --backend metal makes this an error."
    )
}

impl RunChoice {
    /// Explicit choices, else the detected ones. Without `--backend`, Metal is
    /// used when its device, shader and required features work, else the CPU
    /// backend with one stderr line saying why; an explicit Metal is an error.
    pub fn select(config: &RealEngineConfig) -> Result<Self> {
        Self::select_with(config, probe_metal)
    }

    fn select_with(
        config: &RealEngineConfig,
        probe: impl FnOnce(usize) -> std::result::Result<Probe, String>,
    ) -> Result<Self> {
        let kernel = config.kernel.unwrap_or_else(Kernel::fastest);
        if !kernel.available() {
            return Err(Error::engine(format!(
                "--kernel {} is not available on this CPU (available: {})",
                kernel.name(),
                Kernel::available_variants()
                    .iter()
                    .map(|k| k.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let threads = config
            .threads
            .unwrap_or_else(yskn_engine::Engine::default_threads);
        let cpu_device = format!("{} x{threads}", kernel.name());
        let machine = Machine::detect();
        let chip = machine.chip.as_deref().unwrap_or("unknown chip");
        let draft_cost_table = if matches!(
            config.depth_policy(),
            yskn_ref::speculation::DepthPolicy::Fixed
        ) {
            "unused (no adaptive neural draft)".to_owned()
        } else if chip == "Apple M5" {
            "measured on an Apple M5 (D21)".to_owned()
        } else {
            format!(
                "measured on an Apple M5 (D21); on this {chip} it can only change speed, never output"
            )
        };
        let mut choice = Self {
            backend: Backend::Cpu,
            device: cpu_device,
            fallback: None,
            kernel,
            threads,
            cpu_stages: config.cpu_stages,
            metal: config.metal,
            share_table: false,
            max_buffer_length: None,
            recommended_working_set: None,
            machine,
            draft_cost_table,
        };
        if config.backend == Some(Backend::Cpu) {
            return Ok(choice);
        }
        match probe(threads) {
            Ok(probe) => {
                choice.backend = Backend::Metal;
                choice.device = probe.name;
                choice.share_table = probe.share_table;
                choice.max_buffer_length = Some(probe.max_buffer_length);
                choice.recommended_working_set = Some(probe.working_set);
            }
            Err(why) if config.backend == Some(Backend::Metal) => {
                return Err(Error::engine(format!(
                    "--backend metal was requested, but Metal is unavailable: {why}"
                )));
            }
            Err(why) => {
                static NOTICE: Once = Once::new();
                NOTICE.call_once(|| eprintln!("{}", fallback_notice(&why)));
                choice.fallback = Some(why);
            }
        }
        Ok(choice)
    }

    fn cpu_stages_text(&self) -> String {
        match (self.cpu_stages, self.cpu_stages.fast()) {
            (CpuStages::Reference, _) => "reference".into(),
            (CpuStages::Auto, true) => "auto (NEON fast paths)".into(),
            (CpuStages::Auto, false) => "auto (no NEON detected: reference stages)".into(),
        }
    }

    fn cpu_share_text(&self) -> String {
        match self.metal.cpu_share {
            CpuShare::Off => "off".into(),
            CpuShare::Fixed(share) => format!("fixed {share}"),
            CpuShare::Auto if self.share_table => {
                "auto: calibrated table (Apple M5, 10 CPU workers)".into()
            }
            CpuShare::Auto => format!(
                "auto: off, the table was measured on an Apple M5 with 10 CPU workers and this is {} with {}; measure with bench_metal --shares and pass --cpu-share",
                self.device, self.threads
            ),
        }
    }

    /// Fields of `yskn info`.
    pub fn to_json(&self) -> Value {
        let metal = self.backend == Backend::Metal;
        let unused = || json!("unused (CPU backend)");
        json!({
            "kernel": self.kernel.name(),
            "cpu_stages": self.cpu_stages_text(),
            "metal_mac": if metal { json!(if self.metal.float_mac { "float" } else { "integer" }) } else { unused() },
            "cpu_share": if metal { json!(self.cpu_share_text()) } else { unused() },
            "backend_fallback": self.fallback,
            "draft_cost_table": self.draft_cost_table,
            "machine": {
                "chip": self.machine.chip,
                "macos": self.machine.macos_version,
                "architecture": std::env::consts::ARCH,
                "cores_per_performance_level": self.machine.core_levels,
                "page_size": self.machine.page_size,
                "metal_max_buffer_length": self.max_buffer_length,
                "metal_recommended_working_set": self.recommended_working_set,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shader that does not compile or a missing feature (MetalEngine::new
    /// fails) or no device: the default selection runs on the CPU and says why;
    /// an explicit Metal backend is an error; a working probe selects Metal.
    #[test]
    fn default_backend_falls_back_to_the_cpu_and_explicit_metal_fails() {
        let failing = |_| Err("Metal shader compilation: expected failure".to_owned());
        let config = RealEngineConfig::default();
        let choice = RunChoice::select_with(&config, failing).unwrap();
        assert_eq!(choice.backend, Backend::Cpu);
        assert_eq!(
            choice.device,
            format!("{} x{}", choice.kernel.name(), choice.threads)
        );
        assert_eq!(
            choice.fallback.as_deref(),
            Some("Metal shader compilation: expected failure")
        );
        let notice = fallback_notice("no Metal device");
        assert!(
            notice
                .starts_with("yskn: Metal is unavailable (no Metal device); using the CPU backend")
        );
        assert!(notice.contains("6 times as long") && !notice.contains('\n'));
        let metal = RealEngineConfig {
            backend: Some(Backend::Metal),
            ..RealEngineConfig::default()
        };
        let error = RunChoice::select_with(&metal, failing)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "--backend metal was requested, but Metal is unavailable: Metal shader compilation: expected failure"
        );
        let working = |threads| {
            Ok(Probe {
                name: "Apple M9".into(),
                share_table: threads == 10,
                max_buffer_length: 1 << 34,
                working_set: 1 << 33,
            })
        };
        let choice = RunChoice::select_with(&config, working).unwrap();
        assert_eq!(
            (choice.backend, choice.device.as_str()),
            (Backend::Metal, "Apple M9")
        );
        assert!(choice.fallback.is_none());
        let cpu = RealEngineConfig {
            backend: Some(Backend::Cpu),
            ..RealEngineConfig::default()
        };
        let choice = RunChoice::select_with(&cpu, |_| panic!("CPU must not probe Metal")).unwrap();
        assert_eq!(choice.backend, Backend::Cpu);
        assert!(choice.fallback.is_none());
    }

    /// Off-machine CPU share: `auto` says it is off and how to calibrate.
    #[test]
    fn cpu_share_text_names_the_calibrated_machine_and_how_to_calibrate() {
        let config = RealEngineConfig {
            threads: Some(8),
            ..RealEngineConfig::default()
        };
        let probe = |threads| {
            Ok(Probe {
                name: "Apple M4 Pro".into(),
                share_table: threads == 10,
                max_buffer_length: 1 << 34,
                working_set: 1 << 33,
            })
        };
        let choice = RunChoice::select_with(&config, probe).unwrap();
        assert_eq!(
            choice.to_json()["cpu_share"],
            "auto: off, the table was measured on an Apple M5 with 10 CPU workers and this is Apple M4 Pro with 8; measure with bench_metal --shares and pass --cpu-share"
        );
    }
}
