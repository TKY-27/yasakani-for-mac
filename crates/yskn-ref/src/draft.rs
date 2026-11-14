//! One-shot scalar block drafter (chapter 07; D15).
//! No main weights are opened: the caller supplies frontier/mask embeddings.
use std::fs;
use std::io::{Read, Seek};
use std::path::{Component, Path, PathBuf};

use serde_json::Value;
use yskn_format::{Container, Header, VectorType};

use crate::compressed_linear::{CompressedLinear, quantize_activation};
use crate::draft_head::{Candidate, DraftChoice, HeadLayout, ReducedHead, choose};
use crate::portable_math::{bfloat, exp, exp2, rsqrt, sincos};
use crate::{Error, Result, require_finite};

pub const DEFAULT_CAPTURE_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];
pub const DEFAULT_MASK_TOKEN: u32 = 248070;
pub const BLOCK_ROWS: usize = 8;
const HD: usize = 128;
const SCALE: f32 = 0.088_388_346;
fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidModel(message.into())
}
fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("draft dimension overflow"))
}
fn dimension(model: &Value, name: &str) -> Result<usize> {
    model
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid(format!("draft model.{name} must be positive")))
}
#[derive(Clone, Debug)]
pub struct DraftOptions {
    pub mask_token: u32,
    pub capture_layers: Vec<usize>,
    pub shortlist: usize,
}
impl Default for DraftOptions {
    fn default() -> Self {
        Self {
            mask_token: DEFAULT_MASK_TOKEN,
            capture_layers: DEFAULT_CAPTURE_LAYERS.to_vec(),
            shortlist: 512,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct DraftDimensions {
    pub hidden: usize,
    pub features: usize,
    pub feed_forward: usize,
    pub vocabulary: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub window: usize,
    pub correction_rank: usize,
    pub epsilon: f32,
}
impl DraftDimensions {
    pub fn read(header: &Header) -> Result<Self> {
        let m = &header.model;
        if m.get("arch").and_then(Value::as_str) != Some("bdraft2") {
            return Err(invalid("requires arch=bdraft2"));
        }
        let hidden = dimension(m, "hidden")?;
        let features = header
            .tensors
            .iter()
            .find(|t| t.name == "bdraft2.fc")
            .ok_or_else(|| invalid("missing bdraft2.fc"))?
            .cols;
        let result = Self {
            hidden,
            features,
            feed_forward: dimension(m, "ffn")?,
            vocabulary: dimension(m, "vocab")?,
            layers: dimension(m, "layers")?,
            query_heads: dimension(m, "heads")?,
            kv_heads: dimension(m, "kv_heads")?,
            window: dimension(m, "window")?,
            correction_rank: dimension(m, "rank")?,
            epsilon: m
                .get("eps")
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("draft eps missing"))? as f32,
        };
        if hidden % 512 != 0
            || result.feed_forward % 512 != 0
            || result.query_heads % result.kv_heads != 0
            || product(result.query_heads, HD)? % 512 != 0
            || features % hidden != 0
            || result.vocabulary < 16
            || result.vocabulary > i32::MAX as usize
            || result.window < 2
            || result.window > 2048
            || !result.window.is_power_of_two()
            || result.correction_rank % 32 != 0
            || !result.epsilon.is_finite()
            || result.epsilon <= 0.0
            || dimension(m, "head_dim")? != HD
            || dimension(m, "block")? != BLOCK_ROWS
            || dimension(m, "top_k")? != 16
        {
            return Err(invalid(
                "unsupported draft dimensions (128 channels, block 8, top16 required)",
            ));
        }
        // Bound all composite extents before loading/allocating them.
        for (a, b) in [
            (hidden, 4),
            (hidden, result.correction_rank),
            (result.vocabulary, result.correction_rank),
            (
                result.layers,
                product(result.window, product(result.kv_heads, HD)?)?,
            ),
        ] {
            product(a, b)?;
        }
        Ok(result)
    }
}

/// One committed main-model row, concatenated [capture_slot, hidden_channel].
/// Slots default to zero-based layers 5,19,33,47,61. Tap AFTER the layer's MLP
/// down residual addition, BEFORE the next layer/final normalization. Values
/// are BF16 residuals expanded exactly to f32: bf(old_residual + bf(down)).
/// No extra gain/norm/rotation. Source: engine35.cpp:798–799,830–833,1650;
/// q35_kernels.cu:1437–1444. Batch capture [slot,row,channel] must be gathered
/// into this [slot,channel] order (q35_bdraft.cu:249–252,490–499).
pub struct ContextFeatures<'a> {
    pub values: &'a [f32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSnapshot {
    dimensions: (usize, usize, usize),
    position: usize,
    keys: Vec<Vec<u16>>,
    values: Vec<Vec<u16>>,
}
impl DraftSnapshot {
    /// A zeroed ring at position 0: `layers` arrays of [kv_head, slot, channel] BF16 bits.
    pub fn empty(layers: usize, kv_heads: usize, window: usize) -> Self {
        let n = kv_heads * window * HD;
        Self {
            dimensions: (layers, kv_heads, window),
            position: 0,
            keys: vec![vec![0; n]; layers],
            values: vec![vec![0; n]; layers],
        }
    }
    pub fn position(&self) -> usize {
        self.position
    }
    /// (layers, kv_heads, window); restore requires equal shapes.
    pub fn shape(&self) -> (usize, usize, usize) {
        self.dimensions
    }
    /// Zero every slot and continue at `position` (reset, or reseeding a full ring suffix).
    pub fn clear(&mut self, position: usize) {
        self.position = position;
        for row in self.keys.iter_mut().chain(&mut self.values) {
            row.fill(0);
        }
    }
    /// Publish one complete context row: each layer's [kv_head, channel] key and value
    /// BF16 bits go to slot `position % window`, then the position advances.
    pub fn push_row(&mut self, layers: &[(Vec<u16>, Vec<u16>)]) {
        let (_, kv_heads, window) = self.dimensions;
        for (layer, (key, value)) in layers.iter().enumerate() {
            for head in 0..kv_heads {
                let offset = (head * window + self.position % window) * HD;
                self.keys[layer][offset..offset + HD]
                    .copy_from_slice(&key[head * HD..(head + 1) * HD]);
                self.values[layer][offset..offset + HD]
                    .copy_from_slice(&value[head * HD..(head + 1) * HD]);
            }
        }
        self.position += 1;
    }
    /// Head-major BF16 bits [kv_head,slot,channel] for each layer.
    pub fn keys(&self) -> &[Vec<u16>] {
        &self.keys
    }
    pub fn values(&self) -> &[Vec<u16>] {
        &self.values
    }
}
struct Layer {
    q: CompressedLinear,
    k: CompressedLinear,
    v: CompressedLinear,
    o: CompressedLinear,
    gate: CompressedLinear,
    up: CompressedLinear,
    down: CompressedLinear,
    akp: CompressedLinear,
    mkp: CompressedLinear,
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    abase: Vec<f32>,
    mbase: Vec<f32>,
}
pub struct Draft {
    dimensions: DraftDimensions,
    options: DraftOptions,
    signs: Vec<i8>,
    fc: CompressedLinear,
    hidden_norm: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    hp: Vec<f32>,
    pred: Vec<f32>,
    succ: Vec<f32>,
    head: ReducedHead,
    ring: DraftSnapshot,
}
#[derive(Clone, Debug, PartialEq)]
pub struct DraftProposal {
    pub tokens: Vec<u32>,
    /// Pre-correction top16, in rescore order.
    pub candidates: Vec<Vec<Candidate>>,
    /// Same IDs/order, with scores corrected using the selected predecessor.
    pub corrected: Vec<Vec<Candidate>>,
}
/// Options must match the container and head dimensions (shared with fast drafts).
pub fn validate_options(
    d: &DraftDimensions,
    options: &DraftOptions,
    head: &HeadLayout,
) -> Result<()> {
    if options.capture_layers.is_empty()
        || product(options.capture_layers.len(), d.hidden)? != d.features
        || options.capture_layers.windows(2).any(|w| w[0] >= w[1])
        || options.mask_token as usize >= d.vocabulary
        || options.shortlist < 16
        || options.shortlist > 2048
        || options.shortlist % 8 != 0
        || options.shortlist > d.vocabulary
        || head.hidden != d.hidden
        || head.vocabulary != d.vocabulary
    {
        return Err(invalid("draft options/head incompatible with dimensions"));
    }
    Ok(())
}
/// Every projection uses a prefix of `bdraft2.fc`'s input signs.
pub fn validate_signs(d: &DraftDimensions, signs: &[i8]) -> Result<()> {
    if [d.hidden, d.feed_forward, product(d.query_heads, HD)?]
        .iter()
        .any(|&n| n > signs.len())
    {
        return Err(invalid("bdraft2.fc input signs too short"));
    }
    Ok(())
}
/// Manifest defaults and canonical containment check, including symlinks.
/// Only draft files are resolved; files.weights is never opened.
pub fn model_paths(folder: impl AsRef<Path>) -> Result<(PathBuf, PathBuf)> {
    let root = folder.as_ref().canonicalize()?;
    let manifest = match fs::read(root.join("yasakani.json")) {
        Ok(b) => serde_json::from_slice::<Value>(&b)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(e) => return Err(e.into()),
    };
    if !manifest.is_object() || manifest.get("files").is_some_and(|v| !v.is_object()) {
        return Err(invalid("invalid draft manifest"));
    }
    let resolve = |key: &str, default: &str| -> Result<PathBuf> {
        let name = match manifest.get("files").and_then(|f| f.get(key)) {
            Some(v) => v
                .as_str()
                .ok_or_else(|| invalid("draft filename must be a string"))?,
            None => default,
        };
        let path = Path::new(name);
        if name.is_empty()
            || path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(invalid("draft path must stay inside model folder"));
        }
        let path = root.join(path).canonicalize()?;
        if !path.starts_with(&root) {
            return Err(invalid("draft symlink escapes model folder"));
        }
        Ok(path)
    };
    Ok((
        resolve("draft", "bdraft2_k8.yskn")?,
        resolve("draft_head", "bdraft2_k8.shead")?,
    ))
}
impl Draft {
    pub fn load(folder: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(folder, DraftOptions::default())
    }
    pub fn load_with_options(folder: impl AsRef<Path>, options: DraftOptions) -> Result<Self> {
        let (draft, head) = model_paths(folder)?;
        Self::from_container(
            &mut Container::open(draft)?,
            ReducedHead::open(head)?,
            options,
        )
    }
    pub fn from_container<R: Read + Seek>(
        container: &mut Container<R>,
        head: ReducedHead,
        options: DraftOptions,
    ) -> Result<Self> {
        let d = DraftDimensions::read(container.header())?;
        validate_options(&d, &options, head.layout())?;
        let fc = matrix(container, "bdraft2.fc", d.hidden, d.features)?;
        let signs = fc.input_signs().to_vec();
        validate_signs(&d, &signs)?;
        let hidden_norm = vector(container, "bdraft2.hidden_norm", d.hidden)?;
        let final_norm = vector(container, "bdraft2.norm", d.hidden)?;
        let mut layers = Vec::new();
        for i in 0..d.layers {
            let p = format!("bdraft2.layers.{i}.");
            layers.push(Layer {
                q: matrix(
                    container,
                    &format!("{p}q"),
                    product(d.query_heads, HD)?,
                    d.hidden,
                )?,
                k: matrix(
                    container,
                    &format!("{p}k"),
                    product(d.kv_heads, HD)?,
                    d.hidden,
                )?,
                v: matrix(
                    container,
                    &format!("{p}v"),
                    product(d.kv_heads, HD)?,
                    d.hidden,
                )?,
                o: matrix(
                    container,
                    &format!("{p}o"),
                    d.hidden,
                    product(d.query_heads, HD)?,
                )?,
                gate: matrix(container, &format!("{p}gate"), d.feed_forward, d.hidden)?,
                up: matrix(container, &format!("{p}up"), d.feed_forward, d.hidden)?,
                down: matrix(container, &format!("{p}down"), d.hidden, d.feed_forward)?,
                akp: matrix(container, &format!("{p}akp"), d.hidden / 4, d.hidden)?,
                mkp: matrix(container, &format!("{p}mkp"), d.hidden / 4, d.hidden)?,
                ln1: vector(container, &format!("{p}ln1"), d.hidden)?,
                ln2: vector(container, &format!("{p}ln2"), d.hidden)?,
                qn: vector(container, &format!("{p}q_norm"), HD)?,
                kn: vector(container, &format!("{p}k_norm"), HD)?,
                abase: vector(container, &format!("{p}abase"), 4 * d.hidden)?,
                mbase: vector(container, &format!("{p}mbase"), 4 * d.hidden)?,
            });
        }
        let hp = vector(
            container,
            "bdraft2.sel.hp",
            product(d.correction_rank, d.hidden)?,
        )?;
        let pred = vector(
            container,
            "bdraft2.sel.pred",
            product(d.vocabulary, d.correction_rank)?,
        )?;
        let succ = vector(
            container,
            "bdraft2.sel.succ",
            product(d.vocabulary, d.correction_rank)?,
        )?;
        product(product(d.kv_heads, d.window)?, HD)?;
        let ring = DraftSnapshot::empty(d.layers, d.kv_heads, d.window);
        Ok(Self {
            dimensions: d,
            options,
            signs,
            fc,
            hidden_norm,
            final_norm,
            layers,
            hp,
            pred,
            succ,
            head,
            ring,
        })
    }
    pub fn dimensions(&self) -> &DraftDimensions {
        &self.dimensions
    }
    pub fn options(&self) -> &DraftOptions {
        &self.options
    }
    pub fn position(&self) -> usize {
        self.ring.position
    }
    pub fn snapshot(&self) -> DraftSnapshot {
        self.ring.clone()
    }
    pub fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        if self.ring.dimensions != snapshot.dimensions {
            return Err(Error::InvalidInput("incompatible draft snapshot".into()));
        }
        self.ring = snapshot.clone();
        Ok(())
    }
    pub fn reset(&mut self) {
        self.ring.clear(0);
    }
    // Used only when a long prefill retained a complete ring-capacity suffix.
    pub(crate) fn reset_at(&mut self, position: usize) {
        self.ring.clear(position);
    }
    pub fn push_context(&mut self, position: usize, features: ContextFeatures<'_>) -> Result<()> {
        let d = &self.dimensions;
        if position != self.position() || features.values.len() != d.features {
            return Err(Error::InvalidInput(
                "draft context position/shape mismatch".into(),
            ));
        }
        position
            .checked_add(BLOCK_ROWS)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| Error::InvalidInput("draft position overflow".into()))?;
        require_finite(features.values, "draft context features")?;
        if features
            .values
            .iter()
            .any(|&x| bfloat(x).to_bits() != x.to_bits())
        {
            return Err(Error::InvalidInput(
                "context residual must be BF16 expanded to f32".into(),
            ));
        }
        let context = norm(
            &linear(&self.fc, features.values, &self.signs)?,
            &self.hidden_norm,
            d.epsilon,
        )?;
        // All layers project the SAME context; not sequential blocks (engine35.cpp:1778–1782).
        let mut pending = Vec::with_capacity(d.layers);
        for l in &self.layers {
            let mut k = linear(&l.k, &context, &self.signs)?;
            let v = linear(&l.v, &context, &self.signs)?;
            rope(&mut k, &l.kn, position, d.epsilon)?;
            let key: Vec<_> = k.into_iter().map(yskn_format::f32_to_bf16).collect();
            let value: Vec<_> = v.into_iter().map(yskn_format::f32_to_bf16).collect();
            if key
                .iter()
                .chain(&value)
                .any(|&b| !yskn_format::bf16_to_f32(b).is_finite())
            {
                return Err(Error::NonFinite("draft ring BF16"));
            }
            pending.push((key, value));
        }
        self.ring.push_row(&pending);
        Ok(())
    }
    /// Row-major [row,capture_slot,channel]; exactly the same per-row operations.
    /// Completed rows stay committed if a later row fails.
    pub fn push_context_rows(&mut self, position: usize, features: &[f32]) -> Result<()> {
        if features.len() % self.dimensions.features != 0 || position != self.position() {
            return Err(Error::InvalidInput(
                "draft context batch shape/position".into(),
            ));
        }
        for row in features.chunks_exact(self.dimensions.features) {
            self.push_context(self.position(), ContextFeatures { values: row })?;
        }
        Ok(())
    }
    /// Embedding provider returns BF16-expanded main embedding rows for frontier/mask.
    pub fn propose(
        &self,
        frontier: u32,
        mut embedding: impl FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        if frontier as usize >= self.dimensions.vocabulary {
            return Err(Error::InvalidInput(
                "draft frontier outside vocabulary".into(),
            ));
        }
        let mut rows = Vec::with_capacity(BLOCK_ROWS);
        for row in 0..BLOCK_ROWS {
            rows.push(embedding(if row == 0 {
                frontier
            } else {
                self.options.mask_token
            })?);
        }
        let hidden = self.block_hidden(&rows)?;
        let mut result = DraftProposal {
            tokens: Vec::new(),
            candidates: Vec::new(),
            corrected: Vec::new(),
        };
        let mut predecessor = frontier;
        for (j, x) in hidden.chunks_exact(self.dimensions.hidden).enumerate() {
            let candidates = self.head.candidates(x, self.options.shortlist)?;
            let corrected = self.correct(x, predecessor, &candidates)?;
            let token = choose(&corrected, choice, (self.position() + 1 + j) as u64)?;
            result.tokens.push(token);
            result.candidates.push(candidates);
            result.corrected.push(corrected);
            predecessor = token;
        }
        Ok(result)
    }
    /// Execute eight explicit embedding rows; returns normalized rows 1..7.
    /// Exposed for fast-path conformance and non-causal block experiments.
    pub fn block_hidden(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        let d = &self.dimensions;
        if embeddings.len() != BLOCK_ROWS || embeddings.iter().any(|r| r.len() != d.hidden) {
            return Err(Error::InvalidInput(
                "draft block needs eight hidden-width embedding rows".into(),
            ));
        }
        if self
            .position()
            .checked_add(BLOCK_ROWS)
            .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(Error::InvalidInput("draft position overflow".into()));
        }
        let mut hidden: Vec<_> = embeddings.iter().flatten().copied().collect();
        require_finite(&hidden, "draft embeddings")?;
        if hidden.iter().any(|&x| bfloat(x).to_bits() != x.to_bits()) {
            return Err(Error::InvalidInput(
                "draft embeddings must be BF16 expanded to f32".into(),
            ));
        }
        for (i, l) in self.layers.iter().enumerate() {
            let normalized = norm_rows(&hidden, &l.ln1, d.epsilon)?;
            let dynamic = linear_rows(&l.akp, &normalized, &self.signs)?;
            let filtered = dynamic_filter(&normalized, &dynamic, &l.abase, d.hidden, 0);
            let mut q = linear_rows(&l.q, &filtered, &self.signs)?;
            let mut k = linear_rows(&l.k, &filtered, &self.signs)?;
            let v = linear_rows(&l.v, &filtered, &self.signs)?;
            for row in 0..BLOCK_ROWS {
                rope(
                    &mut q[row * d.query_heads * HD..(row + 1) * d.query_heads * HD],
                    &l.qn,
                    self.position() + row,
                    d.epsilon,
                )?;
                rope(
                    &mut k[row * d.kv_heads * HD..(row + 1) * d.kv_heads * HD],
                    &l.kn,
                    self.position() + row,
                    d.epsilon,
                )?;
            }
            let attended = attention(
                &q,
                &k,
                &v,
                &self.ring.keys[i],
                &self.ring.values[i],
                self.position(),
                d,
            )?;
            let projected = linear_rows(&l.o, &attended, &self.signs)?;
            add(
                &mut hidden,
                &dynamic_filter(&projected, &dynamic, &l.abase, d.hidden, 1),
            )?;
            let normalized = norm_rows(&hidden, &l.ln2, d.epsilon)?;
            let dynamic = linear_rows(&l.mkp, &normalized, &self.signs)?;
            let filtered = dynamic_filter(&normalized, &dynamic, &l.mbase, d.hidden, 0);
            let gate = linear_rows(&l.gate, &filtered, &self.signs)?;
            let up = linear_rows(&l.up, &filtered, &self.signs)?;
            let projected = linear_rows(&l.down, &glu(&gate, &up), &self.signs)?;
            add(
                &mut hidden,
                &dynamic_filter(&projected, &dynamic, &l.mbase, d.hidden, 1),
            )?;
        }
        norm_rows(&hidden[d.hidden..], &self.final_norm, d.epsilon)
    }
    fn correct(
        &self,
        x: &[f32],
        predecessor: u32,
        candidates: &[Candidate],
    ) -> Result<Vec<Candidate>> {
        let r = self.dimensions.correction_rank;
        let feature = correction_feature(x, &self.hp, r)?;
        let pred = &self.pred[predecessor as usize * r..(predecessor as usize + 1) * r];
        let succ: Vec<_> = candidates
            .iter()
            .map(|c| &self.succ[c.token as usize * r..(c.token as usize + 1) * r])
            .collect();
        correct_candidates(&feature, pred, &succ, candidates)
    }
}

/// The block-draft operations used by `DraftProposer`. Implemented by this scalar
/// reference and by fast drafts, which must reproduce it bit for bit.
pub trait BlockDraft {
    fn dimensions(&self) -> &DraftDimensions;
    fn options(&self) -> &DraftOptions;
    fn position(&self) -> usize;
    fn snapshot(&self) -> DraftSnapshot;
    fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()>;
    fn reset(&mut self);
    /// Zero the ring and continue at `position`; only for reseeding from a complete
    /// ring-capacity suffix of committed rows.
    fn reset_at(&mut self, position: usize);
    /// `push_context` for consecutive rows starting at `position`; rows before a
    /// failing row stay committed.
    fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()>;
    fn propose(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal>;
}
impl BlockDraft for Draft {
    fn dimensions(&self) -> &DraftDimensions {
        Draft::dimensions(self)
    }
    fn options(&self) -> &DraftOptions {
        Draft::options(self)
    }
    fn position(&self) -> usize {
        Draft::position(self)
    }
    fn snapshot(&self) -> DraftSnapshot {
        Draft::snapshot(self)
    }
    fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        Draft::restore(self, snapshot)
    }
    fn reset(&mut self) {
        Draft::reset(self)
    }
    fn reset_at(&mut self, position: usize) {
        Draft::reset_at(self, position)
    }
    fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        for (i, &values) in rows.iter().enumerate() {
            self.push_context(position + i, ContextFeatures { values })?;
        }
        Ok(())
    }
    fn propose(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        Draft::propose(self, frontier, embedding, choice)
    }
}

/// Dense BF16 hidden-to-correction projection `hp` ([rank, hidden], row-major).
pub fn correction_feature(x: &[f32], hp: &[f32], rank: usize) -> Result<Vec<f32>> {
    let hidden = x.len();
    let mut feature = vec![0.0; rank];
    for (rank, value) in feature.iter_mut().enumerate() {
        let mut partial = [0.0_f32; 32];
        for (lane, acc) in partial.iter_mut().enumerate() {
            for i in (lane..hidden).step_by(32) {
                // q35_bdraft.cu:303: dense BF16, NOT compressed.
                *acc = x[i].mul_add(hp[rank * hidden + i], *acc);
            }
        }
        *value = xor_sum32(&partial);
    }
    require_finite(&feature, "draft correction feature")?;
    Ok(feature)
}

/// Predecessor correction: `pred` is the predecessor's row, `succ[i]` candidate i's row.
pub fn correct_candidates(
    feature: &[f32],
    pred: &[f32],
    succ: &[&[f32]],
    candidates: &[Candidate],
) -> Result<Vec<Candidate>> {
    let rank = feature.len();
    let product: Vec<_> = feature
        .iter()
        .enumerate()
        .map(|(r, &h)| pred[r] * h)
        .collect();
    require_finite(&product, "draft predecessor product")?;
    let mut output = Vec::with_capacity(16);
    for (c, succ) in candidates.iter().zip(succ) {
        let mut partial = [0.0_f32; 32];
        for (lane, acc) in partial.iter_mut().enumerate() {
            for r in (lane..rank).step_by(32) {
                // q35_bdraft.cu:409 (also legacy selector :334).
                *acc = product[r].mul_add(succ[r], *acc);
            }
        }
        output.push(Candidate {
            token: c.token,
            score: c.score + xor_sum32(&partial),
        });
    }
    if output.iter().any(|c| !c.score.is_finite()) {
        return Err(Error::NonFinite("draft corrected score"));
    }
    Ok(output)
}
fn matrix<R: Read + Seek>(
    c: &mut Container<R>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<CompressedLinear> {
    let t = c
        .header()
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if t.rows != rows || t.cols != cols {
        return Err(invalid(format!("{name}: draft matrix shape mismatch")));
    }
    let t = c.load_tensor(name)?;
    require_finite(t.table(), "draft weight table")?;
    require_finite(t.scales(), "draft weight scales")?;
    // engine35.cpp:1638 and q35_bdraft.cu:474: common input signs, no output transform.
    CompressedLinear::new(t, false)
}
fn vector<R: Read + Seek>(c: &mut Container<R>, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = c
        .header()
        .vectors
        .iter()
        .find(|v| v.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if v.len != n || v.dtype != VectorType::Bf16 {
        return Err(invalid(format!("{name}: draft BF16 vector shape mismatch")));
    }
    let v = c.load_vector(name)?.values;
    require_finite(&v, "draft vector")?;
    Ok(v)
}
fn linear(w: &CompressedLinear, x: &[f32], signs: &[i8]) -> Result<Vec<f32>> {
    w.project(&quantize_activation(x, &signs[..x.len()])?, None)
}
fn linear_rows(w: &CompressedLinear, x: &[f32], signs: &[i8]) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    for row in x.chunks_exact(w.cols()) {
        output.extend(linear(w, row, signs)?);
    }
    Ok(output)
}
pub fn norm(x: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    let mut partial = [0.0_f32; 256];
    for (lane, acc) in partial.iter_mut().enumerate() {
        for i in (lane..x.len()).step_by(256) {
            // q35_bdraft.cu:465 (unfused launch equivalent :34).
            *acc = x[i].mul_add(x[i], *acc);
        }
    }
    let mut total = 0.0;
    for warp in partial.chunks_exact(32) {
        total += xor_sum32(warp);
    }
    if !total.is_finite() {
        return Err(Error::NonFinite("draft norm squares"));
    }
    let r = rsqrt(total / x.len() as f32 + epsilon);
    let y: Vec<_> = x.iter().zip(weight).map(|(&x, &w)| (r * x) * w).collect();
    require_finite(&y, "draft norm output")?;
    Ok(y)
}
pub fn norm_rows(x: &[f32], w: &[f32], e: f32) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    for row in x.chunks_exact(w.len()) {
        out.extend(norm(row, w, e)?);
    }
    Ok(out)
}
pub fn rope(x: &mut [f32], w: &[f32], position: usize, epsilon: f32) -> Result<()> {
    for head in x.chunks_exact_mut(HD) {
        let squares: Vec<_> = head.iter().map(|&v| v * v).collect();
        let g: Vec<_> = squares.chunks_exact(32).map(xor_sum32).collect();
        let sum = ((g[0] + g[1]) + g[2]) + g[3];
        if !sum.is_finite() {
            return Err(Error::NonFinite("draft RoPE squares"));
        }
        let r = rsqrt(sum / 128.0 + epsilon);
        let n: [f32; HD] = std::array::from_fn(|i| (r * head[i]) * w[i]);
        require_finite(&n, "draft head norm")?;
        for i in 0..64 {
            let frequency = exp2((-((2 * i) as f32) / 128.0) * 23.253_496_f32);
            let (cs, sn) = sincos(position as f32 * frequency)?;
            head[i] = n[i] * cs - n[i + 64] * sn;
            head[i + 64] = n[i + 64] * cs + n[i] * sn;
        }
    }
    require_finite(x, "draft RoPE")
}
pub fn dynamic_filter(
    x: &[f32],
    dynamic: &[f32],
    base: &[f32],
    width: usize,
    branch: usize,
) -> Vec<f32> {
    let groups = width / 16;
    let dw = 4 * groups;
    let mut y = vec![0.0; x.len()];
    for row in 0..x.len() / width {
        for c in 0..width {
            let g = c / 16;
            let d = row * dw + branch * 2 * groups;
            let mut value = (base[2 * branch * width + c] + dynamic[d + g]) * x[row * width + c];
            if row > 0 {
                value += (base[(2 * branch + 1) * width + c] + dynamic[d + groups + g])
                    * x[(row - 1) * width + c];
            }
            y[row * width + c] = value;
        }
    }
    y
}
/// SiLU-gated product, elementwise: `(g / (1 + exp(-g))) * u`.
pub fn glu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| (g / (1.0 + exp(-g))) * u)
        .collect()
}
/// The reference's 32-lane XOR reduction tree, shared with fast drafts.
pub fn xor_sum32(values: &[f32]) -> f32 {
    crate::gated_deltanet::xor_sum32(values)
}
pub fn add(x: &mut [f32], y: &[f32]) -> Result<()> {
    for (x, &y) in x.iter_mut().zip(y) {
        *x += y;
    }
    require_finite(x, "draft residual")
}

#[derive(Clone)]
struct Partial {
    maximum: f32,
    denominator: f32,
    output: [f32; HD],
}
impl Partial {
    fn empty() -> Self {
        Self {
            maximum: f32::NEG_INFINITY,
            denominator: 0.0,
            output: [0.0; HD],
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rk: &[u16],
    rv: &[u16],
    p: usize,
    d: &DraftDimensions,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0; BLOCK_ROWS * d.query_heads * HD];
    let start = p.saturating_sub(d.window - 1);
    let count = p - start;
    let chunk = count.div_ceil(32);
    for row in 0..BLOCK_ROWS {
        for head in 0..d.query_heads {
            let kv = head / (d.query_heads / d.kv_heads);
            let query =
                &q[(row * d.query_heads + head) * HD..(row * d.query_heads + head + 1) * HD];
            let mut parts = vec![Partial::empty(); 33];
            for (split, part) in parts[..32].iter_mut().enumerate() {
                let first = split * chunk;
                let last = count.min(first + chunk);
                let n = last.saturating_sub(first);
                let mut scores = [f32::NEG_INFINITY; 64];
                for (j, score) in scores[..n].iter_mut().enumerate() {
                    let absolute = start + first + j;
                    if absolute < (p + row).saturating_sub(d.window - 1) {
                        continue;
                    }
                    let offset = (kv * d.window + absolute % d.window) * HD;
                    let mut acc = 0.0_f32;
                    for channel in 0..HD {
                        // Nested pair FMAs execute even then odd: q35_bdraft.cu:175.
                        acc = query[channel]
                            .mul_add(yskn_format::bf16_to_f32(rk[offset + channel]), acc);
                    }
                    *score = acc * SCALE;
                    if !score.is_finite() {
                        return Err(Error::NonFinite("draft history score"));
                    }
                }
                // Max is exact for finite inputs; its pair/XOR tree cannot change it.
                part.maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if part.maximum == f32::NEG_INFINITY {
                    continue;
                }
                let weights: [f32; 64] = std::array::from_fn(|i| {
                    if scores[i] == f32::NEG_INFINITY {
                        0.0
                    } else {
                        exp(scores[i] - part.maximum)
                    }
                });
                let pairs: [f32; 32] = std::array::from_fn(|i| weights[i] + weights[i + 32]);
                part.denominator = xor_sum32(&pairs);
                for (j, &weight) in weights[..n].iter().enumerate() {
                    let offset = (kv * d.window + (start + first + j) % d.window) * HD;
                    for (c, output) in part.output.iter_mut().enumerate() {
                        // q35_bdraft.cu:205.
                        *output = weight.mul_add(yskn_format::bf16_to_f32(rv[offset + c]), *output);
                    }
                }
            }
            let current = &mut parts[32];
            let scaled: [f32; HD] = std::array::from_fn(|c| query[c] * SCALE);
            for j in 0..BLOCK_ROWS {
                let offset = (j * d.kv_heads + kv) * HD;
                let mut lanes = [0.0; 32];
                for (lane, local) in lanes.iter_mut().enumerate() {
                    let c = 4 * lane;
                    // Three explicit nested FMAs, last product separate: q35_bdraft.cu:141.
                    *local = scaled[c].mul_add(
                        k[offset + c],
                        scaled[c + 1].mul_add(
                            k[offset + c + 1],
                            scaled[c + 2]
                                .mul_add(k[offset + c + 2], scaled[c + 3] * k[offset + c + 3]),
                        ),
                    );
                }
                let score = xor_sum32(&lanes);
                if !score.is_finite() {
                    return Err(Error::NonFinite("draft block score"));
                }
                let maximum = current.maximum.max(score);
                let carry = exp(current.maximum - maximum);
                let weight = exp(score - maximum);
                // q35_bdraft.cu:146,148.
                current.denominator = current.denominator.mul_add(carry, weight);
                for (c, output) in current.output.iter_mut().enumerate() {
                    *output = output.mul_add(carry, weight * v[offset + c]);
                }
                current.maximum = maximum;
            }
            let maximum = parts
                .iter()
                .map(|p| p.maximum)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            let mut output = [0.0; HD];
            for part in parts {
                if part.maximum == f32::NEG_INFINITY {
                    continue;
                }
                let weight = exp(part.maximum - maximum);
                // q35_bdraft.cu:579 (unfused-output equivalent :226).
                denominator = part.denominator.mul_add(weight, denominator);
                for (c, o) in output.iter_mut().enumerate() {
                    *o = part.output[c].mul_add(weight, *o);
                }
            }
            for (c, &o) in output.iter().enumerate() {
                out[(row * d.query_heads + head) * HD + c] = o / denominator;
            }
        }
    }
    require_finite(&out, "draft attention output")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dims() -> DraftDimensions {
        DraftDimensions {
            hidden: 512,
            features: 2560,
            feed_forward: 512,
            vocabulary: 32,
            layers: 1,
            query_heads: 4,
            kv_heads: 4,
            window: 8,
            correction_rank: 32,
            epsilon: 1e-6,
        }
    }
    #[test]
    fn draft_attention_every_row_sees_future_block_rows_and_sliding_history() {
        let d = dims();
        let q = vec![0.0; 8 * 4 * HD];
        let k = q.clone();
        let mut v = q.clone();
        for h in 0..4 {
            v[(7 * 4 + h) * HD..(7 * 4 + h + 1) * HD].fill(8.0);
        }
        let rk = vec![0; 4 * 8 * HD];
        let mut rv = rk.clone();
        for h in 0..4 {
            for slot in 0..8 {
                rv[(h * 8 + slot) * HD..(h * 8 + slot + 1) * HD]
                    .fill(yskn_format::f32_to_bf16(2.0));
            }
        }
        let out = attention(&q, &k, &v, &rk, &rv, 0, &d).unwrap();
        // Uniform weights: future row 7 contributes 8/8 to ALL rows, including 0.
        assert!(out.iter().all(|&x| x == 1.0));
        let out = attention(&q, &k, &v, &rk, &rv, 9, &d).unwrap();
        for row in 0..8 {
            let history = 7 - row;
            let expected = (8.0 + 2.0 * history as f32) / (8 + history) as f32;
            assert!(
                out[row * 4 * HD..(row + 1) * 4 * HD]
                    .iter()
                    .all(|&x| x.to_bits() == expected.to_bits())
            );
        }
    }
    #[test]
    fn draft_rope_position_zero_preserves_head_channels_after_direct_weight_norm() {
        let mut x = vec![1.0; HD];
        let w = vec![2.0; HD];
        rope(&mut x, &w, 0, 1e-6).unwrap();
        let expected = 2.0 * rsqrt(1.0 + 1e-6);
        assert!(x.iter().all(|&v| v.to_bits() == expected.to_bits()));
    }
    #[test]
    fn draft_rope_nonzero_position_matches_hand_rotation_and_head_weights() {
        let mut x = vec![1.0; HD];
        let mut weights = vec![0.0; HD];
        weights[0] = 2.0;
        weights[64] = 3.0;
        weights[32] = 4.0;
        weights[96] = 5.0;
        rope(&mut x, &weights, 1, 1e-6).unwrap();
        // RMS(x)=1; r=1/sqrt(1+1e-6). For pair 0, angle=1:
        // r*(2*cos(1)-3*sin(1), 3*cos(1)+2*sin(1)). Pair 32 has
        // angle=10^(-7/2), with direct head weights (4,5). Constants are
        // independently evaluated in binary64 from those two equations.
        for (channel, expected) in [
            (0, -1.443_807_620_783_779_8_f64),
            (64, 3.303_847_235_297_007_4_f64),
            (32, 3.998_416_661_988_438_5_f64),
            (96, 5.001_262_160_412_533_f64),
        ] {
            assert!(
                (f64::from(x[channel]) - expected).abs() < 3e-6,
                "channel={channel} got={} expected={expected}",
                x[channel]
            );
        }
        assert!(
            x.iter()
                .enumerate()
                .all(|(channel, &value)| { [0, 32, 64, 96].contains(&channel) || value == 0.0 })
        );
    }

    #[test]
    fn draft_grouped_attention_nonuniform_scores_use_inverse_sqrt_head_scaling() {
        let mut d = dims();
        d.query_heads = 8;
        d.kv_heads = 2;
        let mut q = vec![0.0; BLOCK_ROWS * d.query_heads * HD];
        for row in 0..BLOCK_ROWS {
            for head in 0..d.query_heads {
                q[(row * d.query_heads + head) * HD] = (head + 1) as f32;
            }
        }
        let mut k = vec![0.0; BLOCK_ROWS * d.kv_heads * HD];
        let mut v = k.clone();
        let mut rk = vec![0; d.kv_heads * d.window * HD];
        let mut rv = rk.clone();
        for kv in 0..d.kv_heads {
            k[kv * HD] = (kv + 1) as f32;
            v[kv * HD] = (16 + kv * 32) as f32;
            rk[kv * d.window * HD] = yskn_format::f32_to_bf16(-((kv + 1) as f32));
            rv[kv * d.window * HD] = yskn_format::f32_to_bf16((8 + kv * 32) as f32);
        }
        let out = attention(&q, &k, &v, &rk, &rv, 1, &d).unwrap();
        for row in 0..BLOCK_ROWS {
            for head in 0..d.query_heads {
                // Eight query heads share two KV heads in contiguous groups of four.
                // Among current rows only key 0 has score s=q*k/sqrt(128),
                // the other seven scores are zero. History position 0 has -s
                // and is in the sliding window until block row 7.
                let kv = head / 4;
                let score = ((head + 1) * (kv + 1)) as f64 / 128.0_f64.sqrt();
                let current_weight = score.exp();
                let history_weight = if row < 7 { (-score).exp() } else { 0.0 };
                let expected = (current_weight * (16 + kv * 32) as f64
                    + history_weight * (8 + kv * 32) as f64)
                    / (current_weight + 7.0 + history_weight);
                let offset = (row * d.query_heads + head) * HD;
                assert!(
                    (f64::from(out[offset]) - expected).abs() < 8e-6,
                    "row={row} head={head}: {} != {expected}",
                    out[offset]
                );
                assert!(out[offset + 1..offset + HD].iter().all(|&x| x == 0.0));
            }
        }
    }

    #[test]
    fn draft_filter_uses_current_dynamic_previous_unmodified_row_and_branch_bases() {
        let x: Vec<_> = (0..3).flat_map(|r| vec![(r + 1) as f32; 512]).collect();
        let mut dynamic = vec![0.0; 3 * 128];
        dynamic[128 + 32] = 3.0;
        dynamic[2 * 128 + 64] = 4.0;
        let base: Vec<_> = [1.0, 2.0, 5.0, 7.0]
            .into_iter()
            .flat_map(|n| vec![n; 512])
            .collect();
        let pre = dynamic_filter(&x, &dynamic, &base, 512, 0);
        assert_eq!((pre[0], pre[512], pre[1024]), (1.0, 7.0, 7.0));
        let post = dynamic_filter(&x, &dynamic, &base, 512, 1);
        assert_eq!((post[0], post[512], post[1024]), (5.0, 17.0, 41.0));
    }
}
