//! Metadata-only start-up accounting; never maps or reads a weight payload.
use std::{fs, process::Command};

use serde_json::{Value, json};
use yskn_format::{Container, Header, Role};
use yskn_ref::{KvMode, draft::DraftDimensions};

use crate::{
    Error, Result,
    real_engine::{ModelInfo, RealEngineConfig},
};

#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub weights_bytes: u64,
    pub draft_bytes: u64,
    /// One committed recurrent state, including convolution history.
    pub recurrent_bytes: u64,
    pub cache_bytes: u64,
    /// One recurrent checkpoint, prefix IDs, logits and optional draft ring.
    pub snapshot_bytes: u64,
    /// Working recurrence, fixed replay, draft ring/captures and operator scratch.
    pub working_bytes: u64,
    /// 15% of accounted allocations plus 1 GiB for the process, allocator and OS.
    pub margin_bytes: u64,
    pub needed_bytes: u64,
    pub physical_bytes: u64,
}
impl MemoryEstimate {
    pub fn enforce(&self, physical_bytes: u64) -> Result<()> {
        if self.needed_bytes > physical_bytes {
            return Err(Error::engine(format!(
                "configuration needs {} bytes ({:.2} GiB), including {} bytes margin; physical memory is {} bytes ({:.2} GiB). Lower --ctx, use --kv int8, or use --draft ngram/none",
                self.needed_bytes,
                self.needed_bytes as f64 / 1_073_741_824.0,
                self.margin_bytes,
                physical_bytes,
                physical_bytes as f64 / 1_073_741_824.0,
            )));
        }
        Ok(())
    }
    pub fn to_json(&self) -> Value {
        json!({"weights_bytes":self.weights_bytes,"draft_bytes":self.draft_bytes,
            "recurrent_bytes":self.recurrent_bytes,"cache_bytes":self.cache_bytes,
            "snapshot_bytes":self.snapshot_bytes,"working_bytes":self.working_bytes,
            "margin_bytes":self.margin_bytes,"needed_bytes":self.needed_bytes,
            "physical_bytes":self.physical_bytes})
    }
}
fn overflow() -> Error {
    Error::engine("memory estimate arithmetic overflow")
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn dim(model: &Value, key: &str) -> Result<usize> {
    model
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::engine(format!("model.{key} must be positive")))
}
fn vector(header: &Header, name: &str) -> Result<u64> {
    header
        .vectors
        .iter()
        .find(|v| v.name == name)
        .map(|v| v.len as u64)
        .ok_or_else(|| Error::engine(format!("missing {name}")))
}
fn codes(header: &Header) -> Result<u64> {
    header
        .tensors
        .iter()
        .filter(|t| t.role != Role::Embedding)
        .try_fold(0, |n, t| add(n, t.stream_bytes))
}
/// Physical memory only; no settings are changed. Failure is explicit, never a
/// silent bypass of the start-up fit check.
pub fn physical_memory() -> Result<u64> {
    if cfg!(target_os = "macos") {
        let primary = Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output();
        if let Ok(result) = &primary {
            if result.status.success() {
                if let Ok(bytes) = String::from_utf8_lossy(&result.stdout)
                    .trim()
                    .parse::<u64>()
                {
                    if bytes > 0 {
                        return Ok(bytes);
                    }
                }
            }
        }
        // Some macOS sandboxes deny sysctl's format lookup. Foundation exposes
        // the same physical-memory fact without changing any system setting.
        // This script is fixed: no model path or request input is interpolated.
        let fallback = Command::new("/usr/bin/osascript")
            .args([
                "-l",
                "JavaScript",
                "-e",
                "ObjC.import('Foundation'); $.NSProcessInfo.processInfo.physicalMemory.toString()",
            ])
            .output();
        if let Ok(result) = &fallback {
            if result.status.success() {
                if let Ok(bytes) = String::from_utf8_lossy(&result.stdout)
                    .trim()
                    .parse::<u64>()
                {
                    if bytes > 0 {
                        return Ok(bytes);
                    }
                }
            }
        }
        let explain = |result: std::io::Result<std::process::Output>| match result {
            Ok(output) => String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            Err(error) => error.to_string(),
        };
        return Err(Error::engine(format!(
            "cannot read physical memory: sysctl hw.memsize: {}; Foundation NSProcessInfo.physicalMemory: {}",
            explain(primary),
            explain(fallback)
        )));
    }

    let meminfo = fs::read_to_string("/proc/meminfo").map_err(Error::engine)?;
    let kb = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .ok_or_else(|| Error::engine("cannot read physical memory"))?;
    mul(kb, 1024)
}

pub(crate) fn inspect(config: &RealEngineConfig) -> Result<ModelInfo> {
    config.validate()?;
    let (path, rounded_max, profile) =
        yskn_ref::model::model_location_with_kv(&config.model_dir, config.kv)
            .map_err(Error::engine)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(config.model_dir.join("yasakani.json")).map_err(Error::engine)?,
    )
    .map_err(Error::engine)?;
    let max_context = manifest
        .get("max_ctx")
        .map(|_| dim(&manifest, "max_ctx"))
        .transpose()?
        .unwrap_or(rounded_max);
    if config.context > max_context {
        return Err(Error::engine(format!(
            "--ctx {} exceeds model max_ctx {max_context}",
            config.context
        )));
    }
    let container = Container::open(&path).map_err(Error::engine)?;
    let h = container.header();
    let hidden = dim(&h.model, "hidden")?;
    let ffn = dim(&h.model, "ffn")?;
    let vocab = dim(&h.model, "vocab")?;
    let layers = dim(&h.model, "layers")?;
    let kv_heads = dim(&h.model, "kv_heads")? as u64;
    let head_width = dim(&h.model, "head_dim")? as u64;
    let kinds = h
        .model
        .get("layer_types")
        .and_then(Value::as_array)
        .filter(|v| v.len() == layers)
        .ok_or_else(|| Error::engine("invalid model.layer_types"))?;
    let mut recurrent = 0;
    let mut replay_per_row = 0;
    let mut cache_per_token = 0;
    let mut attention = 0;
    for (index, kind) in kinds.iter().enumerate() {
        if kind.as_str() == Some("full_attention") {
            let mut row_bytes = mul(2, mul(kv_heads, add(32, head_width)?)?)?;
            if config.kv == KvMode::Kv5 {
                let p = profile
                    .as_ref()
                    .ok_or_else(|| Error::engine("missing KV profile"))?;
                p.check_dimensions(
                    kinds
                        .iter()
                        .filter(|k| k.as_str() == Some("full_attention"))
                        .count(),
                    kv_heads as usize,
                    head_width as usize,
                )
                .map_err(Error::engine)?;
                for side in 0..2 {
                    if p.packable(attention, side) {
                        // D13: 32-byte enum + 544-byte packed row + packed u32 words.
                        let words = mul(4, mul(head_width, p.packed_bits() as u64)?.div_ceil(32))?;
                        let packed = add(576, words)?;
                        row_bytes = add(row_bytes, mul(kv_heads, packed - (32 + head_width))?)?;
                    }
                }
            }
            cache_per_token = add(cache_per_token, add(48, row_bytes)?)?;
            attention += 1;
        } else {
            let prefix = format!("layers.{index}.linear_attn");
            let hd = vector(h, &format!("{prefix}.norm"))?;
            let vh = vector(h, &format!("{prefix}.A_log"))?;
            let qkv = h
                .tensors
                .iter()
                .find(|t| t.name == format!("{prefix}.in_proj_qkv"))
                .ok_or_else(|| Error::engine("missing recurrent qkv"))?
                .rows as u64;
            let conv = vector(h, &format!("{prefix}.conv1d"))?;
            let out = mul(vh, hd)?;
            let qk = qkv.checked_sub(out).ok_or_else(overflow)?;
            let hist = conv.checked_sub(qkv).ok_or_else(overflow)?;
            recurrent = add(recurrent, mul(4, add(mul(vh, mul(hd, hd)?)?, hist)?)?)?;
            replay_per_row = add(
                replay_per_row,
                mul(4, add(add(mul(2, qkv)?, out)?, add(qk, mul(2, vh)?)?)?)?,
            )?;
        }
    }
    let width = config
        .prefill_width
        .max(config.k.checked_add(1).ok_or_else(overflow)?);
    // Cache Vecs grow geometrically. Account for their largest capacity, not
    // merely configured length, at non-power-of-two context limits.
    let cache_capacity = config
        .context
        .checked_next_power_of_two()
        .ok_or_else(overflow)? as u64;
    let cache = mul(cache_per_token, cache_capacity)?;
    let weights = h.file_bytes;
    let dense = h
        .vectors
        .iter()
        .try_fold(0, |n, v| add(n, mul(v.len as u64, 4)?))?;
    let mut draft_bytes = 0;
    let mut draft_codes = 0;
    let mut ring = 0;
    let mut captures = 0;
    if config.draft.neural() {
        let (draft_path, head_path) =
            yskn_ref::draft::model_paths(&config.model_dir).map_err(Error::engine)?;
        let d = Container::open(draft_path).map_err(Error::engine)?;
        let dims = DraftDimensions::read(d.header()).map_err(Error::engine)?;
        // The scalar draft owns codes and expanded vectors/signs. Twice its
        // container size is a conservative bound for those owned structures.
        draft_bytes = add(
            mul(d.header().file_bytes, 2)?,
            fs::metadata(&head_path).map_err(Error::engine)?.len(),
        )?;
        let layout = yskn_ref::draft_head::HeadLayout::read(
            &mut fs::File::open(head_path).map_err(Error::engine)?,
        )
        .map_err(Error::engine)?;
        // Seven generated draft rows each scan approximate V*rank int8
        // weights and rescore precisely shortlist*hidden int8 weights. These
        // are logical code reads, not a hardware/DRAM bandwidth measurement.
        draft_codes = add(
            codes(d.header())?,
            reduced_head_code_reads(&layout, 512.min(layout.vocabulary))?,
        )?;
        ring = mul(
            4,
            mul(
                dims.layers as u64,
                mul(dims.kv_heads as u64, mul(dims.window as u64, 128)?)?,
            )?,
        )?;
        captures = mul(4, mul(width as u64, mul(dims.features as u64, 2)?)?)?;
    }
    let checkpoint = add(
        add(recurrent, mul(config.context as u64, 4)?)?,
        add(mul(vocab as u64, 4)?, ring)?,
    )?;
    // Conservative scratch: 64 f32 hidden/FFN vectors per row, plus all row
    // logits, dense loaded vectors, token history, replay and the working state.
    let scratch = mul(256, mul(width as u64, add(hidden as u64, ffn as u64)?)?)?;
    let working = add(
        add(
            add(recurrent, mul(width as u64, replay_per_row)?)?,
            add(dense, scratch)?,
        )?,
        add(
            add(ring, captures)?,
            add(
                mul(config.context as u64, 8)?,
                mul(width as u64, mul(vocab as u64, 4)?)?,
            )?,
        )?,
    )?;
    let subtotal = [weights, draft_bytes, recurrent, cache, checkpoint, working]
        .into_iter()
        .try_fold(0, add)?;
    let margin = add(mul(subtotal, 15)?.div_ceil(100), 1_073_741_824)?;
    let needed = add(subtotal, margin)?;
    let physical = physical_memory()?;
    let backend = config
        .backend
        .map_or_else(yskn_engine::Backend::from_env, Ok)
        .map_err(Error::engine)?;
    let kernel = std::env::var("YSKN_KERNEL")
        .map_or_else(
            |_| Ok(yskn_engine::Kernel::fastest()),
            |s| s.parse::<yskn_engine::Kernel>(),
        )
        .map_err(Error::engine)?;
    let device = match backend {
        yskn_engine::Backend::Cpu => format!(
            "{} x{}",
            kernel.name(),
            config
                .threads
                .unwrap_or_else(yskn_engine::Engine::default_threads)
        ),
        yskn_engine::Backend::Metal => metal_device()?,
    };
    Ok(ModelInfo {
        model_file: path,
        model_bytes: weights,
        code_bytes_per_pass: codes(h)?,
        draft_code_bytes_per_pass: draft_codes,
        layers,
        hidden_width: hidden,
        feed_forward_width: ffn,
        vocabulary: vocab,
        context: config.context,
        model_max_context: max_context,
        threads: config
            .threads
            .unwrap_or_else(yskn_engine::Engine::default_threads),
        kv: config.kv_name().into(),
        backend: backend.name().into(),
        device,
        draft: config.draft.name().into(),
        k: config.k,
        memory: MemoryEstimate {
            weights_bytes: weights,
            draft_bytes,
            recurrent_bytes: recurrent,
            cache_bytes: cache,
            snapshot_bytes: checkpoint,
            working_bytes: working,
            margin_bytes: margin,
            needed_bytes: needed,
            physical_bytes: physical,
        },
    })
}

fn metal_device() -> Result<String> {
    #[cfg(target_os = "macos")]
    {
        yskn_engine::MetalEngine::new()
            .map_err(Error::engine)?
            .map(|m| m.device_name())
            .ok_or_else(|| Error::engine("Metal backend requires a Metal device"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(Error::engine(
            "Metal backend is unavailable on this platform",
        ))
    }
}

pub(crate) fn reduced_head_code_reads(
    layout: &yskn_ref::draft_head::HeadLayout,
    shortlist: usize,
) -> Result<u64> {
    mul(
        (yskn_ref::draft::BLOCK_ROWS - 1) as u64,
        add(
            mul(layout.vocabulary as u64, layout.rank as u64)?,
            mul(shortlist as u64, layout.hidden as u64)?,
        )?,
    )
}
