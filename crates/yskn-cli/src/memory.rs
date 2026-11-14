//! Metadata-only start-up accounting; never maps or reads a weight payload.
use std::{fs, process::Command};

use serde_json::{Value, json};
use yskn_format::{Container, Header, Role};
use yskn_ref::{KvMode, draft::DraftDimensions};

use crate::{
    Error, Result,
    real_engine::{ModelInfo, RealEngineConfig},
};

const GIB: f64 = 1_073_741_824.0;
/// The smallest context `enforce` suggests (D23).
const SMALLEST_SUGGESTED_CONTEXT: usize = 4096;

/// Header-derived sizes behind the start-up estimate (D19). They do not depend
/// on the context length, pass width or physical memory, so one set of facts
/// prices every `--ctx` and, with or without `draft`, `--draft none`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFacts {
    pub weights_bytes: u64,
    /// One committed recurrent state, including convolution history.
    pub recurrent_bytes: u64,
    pub replay_bytes_per_row: u64,
    /// KV cache bytes per token in the chosen mode, and in `int8`.
    pub cache_bytes_per_token: u64,
    pub int8_cache_bytes_per_token: u64,
    /// All dense vectors expanded to f32.
    pub dense_bytes: u64,
    pub hidden: u64,
    pub feed_forward: u64,
    pub vocabulary: u64,
    /// The fast neural draft (`--draft neural|both`), if the configuration runs it.
    pub draft: Option<DraftFacts>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftFacts {
    /// Mapped draft files plus the fast draft's owned head and vectors.
    pub bytes: u64,
    /// The D17 ring of past key/value rows.
    pub ring_bytes: u64,
    /// Captured main-model features per row.
    pub features: u64,
}

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
    /// What this estimate was computed from, to price the alternatives.
    pub facts: MemoryFacts,
    pub context: usize,
    pub width: usize,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

impl MemoryFacts {
    /// The D19 formula for `context` tokens and pass width `width` (max of the
    /// prefill width and K + 1):
    ///
    /// ```text
    /// subtotal = weights + draft + recurrent + cache + snapshot + working
    /// cache    = cache_per_token * next_power_of_two(context)
    /// snapshot = recurrent + 4*context + 4*vocabulary + draft_ring
    /// working  = recurrent + width*replay_row + dense + 256*width*(hidden+ffn)
    ///          + draft_ring + 8*width*features + 8*context + 4*width*vocabulary
    /// needed   = subtotal + ceil(0.15 * subtotal) + 1 GiB
    /// ```
    pub fn estimate(&self, context: usize, width: usize, physical: u64) -> Result<MemoryEstimate> {
        let (draft_bytes, ring, features) = self
            .draft
            .as_ref()
            .map_or((0, 0, 0), |d| (d.bytes, d.ring_bytes, d.features));
        let width_u = width as u64;
        // Cache Vecs grow geometrically. Account for their largest capacity, not
        // merely configured length, at non-power-of-two context limits.
        let cache_capacity = context.checked_next_power_of_two().ok_or_else(overflow)? as u64;
        let cache = mul(self.cache_bytes_per_token, cache_capacity)?;
        let captures = mul(4, mul(width_u, mul(features, 2)?)?)?;
        let checkpoint = add(
            add(self.recurrent_bytes, mul(context as u64, 4)?)?,
            add(mul(self.vocabulary, 4)?, ring)?,
        )?;
        // Conservative scratch: 64 f32 hidden/FFN vectors per row, plus all row
        // logits, dense loaded vectors, token history, replay and the working state.
        let scratch = mul(256, mul(width_u, add(self.hidden, self.feed_forward)?)?)?;
        let working = add(
            add(
                add(
                    self.recurrent_bytes,
                    mul(width_u, self.replay_bytes_per_row)?,
                )?,
                add(self.dense_bytes, scratch)?,
            )?,
            add(
                add(ring, captures)?,
                add(
                    mul(context as u64, 8)?,
                    mul(width_u, mul(self.vocabulary, 4)?)?,
                )?,
            )?,
        )?;
        let subtotal = [
            self.weights_bytes,
            draft_bytes,
            self.recurrent_bytes,
            cache,
            checkpoint,
            working,
        ]
        .into_iter()
        .try_fold(0, add)?;
        let margin = add(mul(subtotal, 15)?.div_ceil(100), 1_073_741_824)?;
        Ok(MemoryEstimate {
            weights_bytes: self.weights_bytes,
            draft_bytes,
            recurrent_bytes: self.recurrent_bytes,
            cache_bytes: cache,
            snapshot_bytes: checkpoint,
            working_bytes: working,
            margin_bytes: margin,
            needed_bytes: add(subtotal, margin)?,
            physical_bytes: physical,
            facts: self.clone(),
            context,
            width,
        })
    }
}

impl MemoryEstimate {
    pub fn fits(&self, physical_bytes: u64) -> bool {
        self.needed_bytes <= physical_bytes
    }

    /// Fails closed when the estimate exceeds `physical_bytes`, naming what the
    /// configuration needs, what each part costs, and what each option that
    /// reduces it saves (D23).
    pub fn enforce(&self, physical_bytes: u64) -> Result<()> {
        if self.fits(physical_bytes) {
            return Ok(());
        }
        Err(Error::engine(self.shortfall(physical_bytes)?))
    }

    fn shortfall(&self, physical: u64) -> Result<String> {
        let mut text = format!(
            "this configuration needs {} bytes ({:.2} GiB), including {} bytes margin; physical memory is {} bytes ({:.2} GiB), {} bytes short.\nneeded: model weights {}, draft {}, KV cache for --ctx {} {}, recurrent state {}, prompt checkpoint {}, working memory {}, margin {}.\noptions that lower the need (bytes saved, bytes then needed):",
            self.needed_bytes,
            gib(self.needed_bytes),
            self.margin_bytes,
            physical,
            gib(physical),
            self.needed_bytes.saturating_sub(physical),
            self.weights_bytes,
            self.draft_bytes,
            self.context,
            self.cache_bytes,
            self.recurrent_bytes,
            self.snapshot_bytes,
            self.working_bytes,
            self.margin_bytes,
        );
        let mut option = |flag: String, other: &MemoryEstimate| {
            text += &format!(
                "\n  {flag}: saves {} bytes, needs {} bytes ({:.2} GiB){}",
                self.needed_bytes.saturating_sub(other.needed_bytes),
                other.needed_bytes,
                gib(other.needed_bytes),
                if other.fits(physical) { ", fits" } else { "" },
            );
        };
        let without_draft = MemoryFacts {
            draft: None,
            ..self.facts.clone()
        };
        if self.facts.draft.is_some() {
            option(
                "--draft none".into(),
                &without_draft.estimate(self.context, self.width, physical)?,
            );
        }
        for context in smaller_contexts(self.context) {
            option(
                format!("--ctx {context}"),
                &self.facts.estimate(context, self.width, physical)?,
            );
        }
        let int8 = MemoryFacts {
            cache_bytes_per_token: self.facts.int8_cache_bytes_per_token,
            ..self.facts.clone()
        };
        if int8 != self.facts {
            option(
                "--kv int8".into(),
                &int8.estimate(self.context, self.width, physical)?,
            );
        }
        // The largest context that fits for each distinct draft/cache choice.
        let least_facts = MemoryFacts {
            draft: None,
            ..int8.clone()
        };
        let mut variants: Vec<(&MemoryFacts, &str)> = Vec::new();
        for (facts, flag) in [
            (&self.facts, ""),
            (&without_draft, " --draft none"),
            (&int8, " --kv int8"),
            (&least_facts, " --draft none --kv int8"),
        ] {
            if variants.iter().all(|(seen, _)| *seen != facts) {
                variants.push((facts, flag));
            }
        }
        let mut fitting = Vec::new();
        for (facts, flag) in variants {
            for context in std::iter::once(self.context).chain(smaller_contexts(self.context)) {
                if facts
                    .estimate(context, self.width, physical)?
                    .fits(physical)
                {
                    fitting.push(format!("--ctx {context}{flag}"));
                    break;
                }
            }
        }
        if fitting.is_empty() {
            let least = least_facts.estimate(
                SMALLEST_SUGGESTED_CONTEXT.min(self.context),
                self.width,
                physical,
            )?;
            text += &format!(
                "\nnothing listed fits: even --ctx {} --draft none --kv int8 needs {} bytes ({:.2} GiB), so this Mac does not have enough memory for this model.",
                least.context,
                least.needed_bytes,
                gib(least.needed_bytes)
            );
        } else {
            text += &format!("\nlargest that fits: {}", fitting.join("; "));
        }
        Ok(text)
    }

    pub fn to_json(&self) -> Value {
        let f = &self.facts;
        json!({"weights_bytes":self.weights_bytes,"draft_bytes":self.draft_bytes,
            "recurrent_bytes":self.recurrent_bytes,"cache_bytes":self.cache_bytes,
            "snapshot_bytes":self.snapshot_bytes,"working_bytes":self.working_bytes,
            "margin_bytes":self.margin_bytes,"needed_bytes":self.needed_bytes,
            "physical_bytes":self.physical_bytes,"fits":self.fits(self.physical_bytes),
            "check":(!self.fits(self.physical_bytes)).then(|| self.shortfall(self.physical_bytes).ok()).flatten(),
            "facts":{"replay_bytes_per_row":f.replay_bytes_per_row,
                "cache_bytes_per_token":f.cache_bytes_per_token,
                "int8_cache_bytes_per_token":f.int8_cache_bytes_per_token,
                "dense_bytes":f.dense_bytes,"hidden":f.hidden,"feed_forward":f.feed_forward,
                "vocabulary":f.vocabulary,"width":self.width,
                "draft_ring_bytes":f.draft.as_ref().map(|d| d.ring_bytes),
                "draft_features":f.draft.as_ref().map(|d| d.features)}})
    }
}

/// Powers of two below `context`, from the largest down to 4096.
fn smaller_contexts(context: usize) -> impl Iterator<Item = usize> {
    let largest = context
        .saturating_sub(1)
        .checked_next_power_of_two()
        .unwrap_or(0)
        / 2;
    std::iter::successors(Some(largest), |&n| Some(n / 2))
        .take_while(|&n| n >= SMALLEST_SUGGESTED_CONTEXT)
        .filter(move |&n| n < context)
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
    let mut int8_cache_per_token = 0;
    let mut attention = 0;
    for (index, kind) in kinds.iter().enumerate() {
        if kind.as_str() == Some("full_attention") {
            let int8_row = mul(2, mul(kv_heads, add(32, head_width)?)?)?;
            let mut row_bytes = int8_row;
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
            int8_cache_per_token = add(int8_cache_per_token, add(48, int8_row)?)?;
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
    let weights = h.file_bytes;
    let dense = h
        .vectors
        .iter()
        .try_fold(0, |n, v| add(n, mul(v.len as u64, 4)?))?;
    let mut draft = None;
    let mut proposal_codes = 0;
    let mut update_codes = 0;
    if config.draft.neural() {
        let (draft_path, head_path) =
            yskn_ref::draft::model_paths(&config.model_dir).map_err(Error::engine)?;
        let d = Container::open(draft_path).map_err(Error::engine)?;
        let dims = DraftDimensions::read(d.header()).map_err(Error::engine)?;
        let layout = yskn_ref::draft_head::HeadLayout::read(
            &mut fs::File::open(&head_path).map_err(Error::engine)?,
        )
        .map_err(Error::engine)?;
        // Code streams and selector pred/succ stay mapped. Only the norms,
        // layer vectors and selector hp are expanded to f32 (FastDraft).
        let len = |range: &std::ops::Range<u64>| range.end - range.start;
        let owned_head = [0, 1, 2, 4]
            .into_iter()
            .try_fold(0, |n, i| add(n, len(&layout.arrays[i])))?;
        let mut vector_elements = 0;
        for name in ["bdraft2.hidden_norm", "bdraft2.norm", "bdraft2.sel.hp"] {
            vector_elements = add(vector_elements, vector(d.header(), name)?)?;
        }
        for i in 0..dims.layers {
            for name in ["ln1", "ln2", "q_norm", "k_norm", "abase", "mbase"] {
                vector_elements = add(
                    vector_elements,
                    vector(d.header(), &format!("bdraft2.layers.{i}.{name}"))?,
                )?;
            }
        }
        let vectors = mul(vector_elements, 4)?;
        let bytes = [
            d.header().file_bytes,
            layout.file_bytes,
            owned_head,
            vectors,
        ]
        .into_iter()
        .try_fold(0, add)?;
        // Logical code reads, not a hardware/DRAM bandwidth measurement. A
        // proposal runs every block matrix once for its eight rows, and seven
        // generated rows each scan approximate V*rank int8 weights and rescore
        // shortlist*hidden int8 weights. A history update (one push of
        // consecutive committed rows, at most 256 per read) runs the FC input
        // projection and every layer's key/value projections.
        let mut fc = None;
        for t in &d.header().tensors {
            let kv = t.name.starts_with("bdraft2.layers.")
                && (t.name.ends_with(".k") || t.name.ends_with(".v"));
            if t.name == "bdraft2.fc" {
                fc = Some(t.stream_bytes);
            }
            if t.name == "bdraft2.fc" || kv {
                update_codes = add(update_codes, t.stream_bytes)?;
            }
        }
        let fc = fc.ok_or_else(|| Error::engine("missing bdraft2.fc"))?;
        proposal_codes = add(
            codes(d.header())?.checked_sub(fc).ok_or_else(overflow)?,
            reduced_head_code_reads(&layout, 512.min(layout.vocabulary))?,
        )?;
        let ring_bytes = mul(
            4,
            mul(
                dims.layers as u64,
                mul(dims.kv_heads as u64, mul(dims.window as u64, 128)?)?,
            )?,
        )?;
        draft = Some(DraftFacts {
            bytes,
            ring_bytes,
            features: dims.features as u64,
        });
    }
    let facts = MemoryFacts {
        weights_bytes: weights,
        recurrent_bytes: recurrent,
        replay_bytes_per_row: replay_per_row,
        cache_bytes_per_token: cache_per_token,
        int8_cache_bytes_per_token: int8_cache_per_token,
        dense_bytes: dense,
        hidden: hidden as u64,
        feed_forward: ffn as u64,
        vocabulary: vocab as u64,
        draft,
    };
    let memory = facts.estimate(config.context, width, physical_memory()?)?;
    let run = crate::platform::RunChoice::select(config)?;
    Ok(ModelInfo {
        model_file: path,
        model_bytes: weights,
        code_bytes_per_pass: codes(h)?,
        draft_proposal_code_bytes: proposal_codes,
        draft_update_code_bytes: update_codes,
        layers,
        hidden_width: hidden,
        feed_forward_width: ffn,
        vocabulary: vocab,
        context: config.context,
        model_max_context: max_context,
        threads: run.threads,
        kv: config.kv_name().into(),
        backend: run.backend.name().into(),
        device: run.device.clone(),
        draft: config.draft.name().into(),
        k: config.k,
        depth: match config.depth_policy() {
            yskn_ref::speculation::DepthPolicy::Fixed => "fixed",
            yskn_ref::speculation::DepthPolicy::Adaptive(_) => "adaptive",
        }
        .into(),
        memory,
        run,
    })
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
