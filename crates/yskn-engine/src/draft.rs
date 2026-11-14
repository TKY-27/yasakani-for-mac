//! Fast block draft (D18): the scalar reference draft (`yskn_ref::draft`, D15)
//! with compressed projections on Metal or the CPU kernel, parallel and vectorized
//! attention and head stages, and mapped code streams. Ring contents, proposals,
//! candidates and scores equal the reference bit for bit.
use std::{
    fs::File,
    io::Cursor,
    ops::Range,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use memmap2::Mmap;
use yskn_format::{VectorType, bf16_to_f32, f32_to_bf16};
use yskn_ref::compressed_linear::{PreparedActivation, quantize_activation, restore_projection};
use yskn_ref::draft::{
    self as reference, BLOCK_ROWS, BlockDraft, ContextFeatures, DraftDimensions, DraftOptions,
    DraftProposal, DraftSnapshot, correct_candidates, model_paths, validate_options,
    validate_signs,
};
use yskn_ref::draft_head::{Candidate, DraftChoice, HeadLayout, choose, shortlist, top16};
use yskn_ref::portable_math::bfloat;

use crate::draft_math::{self, HD, RANK, Shape, TILE};
use crate::{Backend, Engine, Error, Kernel, MappedModel, Matrix, Result, model::Compute};

/// Rows per push batch: bounds activation and output scratch for long prefills.
const PUSH_BATCH: usize = 256;
/// Approximate-score tiles per pool task (2,048 tokens).
const SCORE_TILES: usize = 256;
/// Rank rows per projection task.
const RANK_TASK: usize = 32;
/// Shortlist candidates per rescoring task.
const RESCORE_TASK: usize = 64;

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidModel(message.into())
}

/// Wall time of the last `propose`, by stage. Stages do not overlap.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct DraftTiming {
    pub total: Duration,
    /// Activation quantization, compressed kernel and restoration (45 matrices).
    pub projections: Duration,
    /// The compressed kernel calls alone (Metal command buffers or CPU pool), included above.
    pub projection_kernel: Duration,
    pub attention: Duration,
    /// Norms, RoPE, dynamic filters, GLU and residual adds.
    pub block_other: Duration,
    /// Rank-512 projection of the seven rows.
    pub head_projection: Duration,
    /// Approximate scores for the whole vocabulary.
    pub head_scores: Duration,
    pub head_shortlist: Duration,
    /// Full-width rescoring and top 16.
    pub head_rescore: Duration,
    /// Correction feature, predecessor correction and choice.
    pub head_select: Duration,
}
impl DraftTiming {
    pub fn head(&self) -> Duration {
        self.head_projection
            + self.head_scores
            + self.head_shortlist
            + self.head_rescore
            + self.head_select
    }
    /// Embedding calls, validation and assembly outside the timed stages.
    pub fn rest(&self) -> Duration {
        self.total
            .saturating_sub(self.projections + self.attention + self.block_other + self.head())
    }
}

struct Layer {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    akp: Matrix,
    mkp: Matrix,
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    abase: Vec<f32>,
    mbase: Vec<f32>,
}

/// A mapped BF16 matrix read one row at a time (`bdraft2.sel.pred/succ`).
struct Bf16Rows {
    mapping: Arc<Mmap>,
    range: Range<usize>,
    width: usize,
}
impl Bf16Rows {
    fn row(&self, index: usize) -> Vec<f32> {
        let start = self.range.start + index * self.width * 2;
        self.mapping[start..start + self.width * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
            .collect()
    }
}

/// The SHEAD file, mapped. Floats are copied (and checked finite), the int8 rescoring
/// rows stay mapped, and the approximate rows are repacked into token tiles.
struct Head {
    mapping: Mmap,
    layout: HeadLayout,
    projection: Vec<f32>,
    approximate: Vec<i8>,
    approximate_scale: Vec<f32>,
    rescore_scale: Vec<f32>,
}
impl Head {
    /// # Safety
    /// The file must not be modified or truncated until the head is dropped.
    unsafe fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the caller guarantees file stability while the mapping lives.
        let mapping = unsafe { Mmap::map(&file)? };
        // The reference reader validates magic, dimensions, every extent and EOF.
        let layout = HeadLayout::read(&mut Cursor::new(&mapping[..]))?;
        let floats = |range: &Range<u64>| -> Result<Vec<f32>> {
            let values: Vec<f32> = mapping[range.start as usize..range.end as usize]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::NonFinite("SHEAD weights/scales"));
            }
            Ok(values)
        };
        let projection = floats(&layout.arrays[0])?;
        let approximate_scale = floats(&layout.arrays[2])?;
        let rescore_scale = floats(&layout.arrays[4])?;
        // Touch every page of the mapped rescoring rows once, so proposals do not
        // pay first-access page faults for newly shortlisted tokens.
        let r = &layout.arrays[3];
        let touched = mapping[r.start as usize..r.end as usize]
            .iter()
            .step_by(4096)
            .fold(0_u8, |acc, &b| acc ^ b);
        std::hint::black_box(touched);
        let a = &layout.arrays[1];
        let approximate = draft_math::pack_approximate(
            &mapping[a.start as usize..a.end as usize],
            layout.vocabulary,
        );
        Ok(Self {
            mapping,
            layout,
            projection,
            approximate,
            approximate_scale,
            rescore_scale,
        })
    }
    fn rescore_row(&self, token: usize) -> &[u8] {
        let start = self.layout.arrays[3].start as usize + token * self.layout.hidden;
        &self.mapping[start..start + self.layout.hidden]
    }
}

pub struct FastDraft {
    dimensions: DraftDimensions,
    options: DraftOptions,
    signs: Vec<i8>,
    fc: Matrix,
    hidden_norm: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    hp: Vec<f32>,
    pred: Bf16Rows,
    succ: Bf16Rows,
    head: Head,
    ring: DraftSnapshot,
    compute: Compute,
    simd: bool,
    timing: DraftTiming,
}

impl FastDraft {
    /// Engine and backend from `YSKN_KERNEL`/`YSKN_BACKEND`, and the Metal
    /// options from `YSKN_METAL_MAC`/`YSKN_METAL_CPU_SHARE` (as `Model::load`),
    /// default options.
    /// # Safety
    /// Both draft files must remain unmodified and untruncated until this draft is dropped.
    pub unsafe fn load(folder: impl AsRef<Path>) -> Result<Self> {
        // SAFETY: forwarded file-stability contract.
        let mut draft = unsafe {
            Self::with_options(
                folder,
                Engine::from_env()?,
                Backend::from_env()?,
                DraftOptions::default(),
            )?
        };
        draft.compute.configure_metal_from_env()?;
        Ok(draft)
    }

    /// Explicit pool, projection backend and options. Never reads the environment.
    /// The float stages use NEON unless `engine` uses the portable kernel.
    /// # Safety
    /// Both draft files must remain unmodified and untruncated until this draft is dropped.
    pub unsafe fn with_options(
        folder: impl AsRef<Path>,
        engine: Engine,
        backend: Backend,
        options: DraftOptions,
    ) -> Result<Self> {
        let (draft_path, head_path) = model_paths(folder)?;
        // SAFETY: the caller guarantees file stability for both mappings.
        let mapping = unsafe { MappedModel::open(draft_path)? };
        // SAFETY: as above.
        let head = unsafe { Head::open(&head_path)? };
        let d = DraftDimensions::read(mapping.header())?;
        validate_options(&d, &options, &head.layout)?;
        let fc = matrix(&mapping, "bdraft2.fc", d.hidden, d.features)?;
        let signs = fc.input_signs().to_vec();
        validate_signs(&d, &signs)?;
        let mut reader = mapping.reader()?;
        let mut vector = |name: &str, n: usize| -> Result<Vec<f32>> {
            vector_descriptor(&mapping, name, n)?;
            let values = reader.load_vector(name)?.values;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::NonFinite("draft vector"));
            }
            Ok(values)
        };
        let hidden_norm = vector("bdraft2.hidden_norm", d.hidden)?;
        let final_norm = vector("bdraft2.norm", d.hidden)?;
        let (qn, kvn) = (d.query_heads * HD, d.kv_heads * HD);
        let mut layers = Vec::new();
        for i in 0..d.layers {
            let p = format!("bdraft2.layers.{i}.");
            let m = |name: &str, rows, cols| matrix(&mapping, &format!("{p}{name}"), rows, cols);
            let (q, k, v, o) = (
                m("q", qn, d.hidden)?,
                m("k", kvn, d.hidden)?,
                m("v", kvn, d.hidden)?,
                m("o", d.hidden, qn)?,
            );
            let (gate, up, down) = (
                m("gate", d.feed_forward, d.hidden)?,
                m("up", d.feed_forward, d.hidden)?,
                m("down", d.hidden, d.feed_forward)?,
            );
            let (akp, mkp) = (
                m("akp", d.hidden / 4, d.hidden)?,
                m("mkp", d.hidden / 4, d.hidden)?,
            );
            layers.push(Layer {
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                akp,
                mkp,
                ln1: vector(&format!("{p}ln1"), d.hidden)?,
                ln2: vector(&format!("{p}ln2"), d.hidden)?,
                qn: vector(&format!("{p}q_norm"), HD)?,
                kn: vector(&format!("{p}k_norm"), HD)?,
                abase: vector(&format!("{p}abase"), 4 * d.hidden)?,
                mbase: vector(&format!("{p}mbase"), 4 * d.hidden)?,
            });
        }
        let hp = vector("bdraft2.sel.hp", d.correction_rank * d.hidden)?;
        let pred = bf16_rows(
            &mapping,
            "bdraft2.sel.pred",
            d.vocabulary,
            d.correction_rank,
        )?;
        let succ = bf16_rows(
            &mapping,
            "bdraft2.sel.succ",
            d.vocabulary,
            d.correction_rank,
        )?;
        let compute = Compute::new(engine, backend)?;
        Ok(Self {
            ring: DraftSnapshot::empty(d.layers, d.kv_heads, d.window),
            simd: cfg!(target_arch = "aarch64") && compute.kernel() != Kernel::Portable,
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
            compute,
            timing: DraftTiming::default(),
        })
    }

    pub fn dimensions(&self) -> &DraftDimensions {
        &self.dimensions
    }
    pub fn options(&self) -> &DraftOptions {
        &self.options
    }
    pub fn position(&self) -> usize {
        self.ring.position()
    }
    pub fn backend(&self) -> Backend {
        self.compute.backend()
    }
    /// Explicit Metal arithmetic and CPU share, as `Model::set_metal_options`.
    pub fn set_metal_options(&mut self, options: crate::MetalOptions) {
        self.compute.set_metal_options(options);
    }
    /// The projection device: Metal device name, or CPU kernel and threads.
    pub fn device(&self) -> String {
        self.compute.device()
    }
    /// Whether attention/head float stages use NEON (else the portable loops).
    pub fn vectorized(&self) -> bool {
        self.simd
    }
    /// Stage times of the last `propose`.
    pub fn timing(&self) -> &DraftTiming {
        &self.timing
    }
    pub fn snapshot(&self) -> DraftSnapshot {
        self.ring.clone()
    }
    pub fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        if self.ring.shape() != snapshot.shape() {
            return Err(Error::InvalidInput("incompatible draft snapshot".into()));
        }
        self.ring = snapshot.clone();
        Ok(())
    }
    pub fn reset(&mut self) {
        self.ring.clear(0);
    }

    pub fn push_context(&mut self, position: usize, features: ContextFeatures<'_>) -> Result<()> {
        self.push_rows(position, &[features.values])
    }
    /// Row-major [row, capture_slot, channel], as the reference's batch push.
    pub fn push_context_rows(&mut self, position: usize, features: &[f32]) -> Result<()> {
        let width = self.dimensions.features;
        if !features.len().is_multiple_of(width) || position != self.position() {
            return Err(Error::InvalidInput(
                "draft context batch shape/position".into(),
            ));
        }
        let rows: Vec<_> = features.chunks_exact(width).collect();
        self.push_rows(position, &rows)
    }

    /// Consecutive committed rows from `position`, batched per dispatch. Exactly the
    /// reference's per-row operations; rows before a failing row stay committed.
    pub fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        let mut next = position;
        for batch in rows.chunks(PUSH_BATCH) {
            self.push_batch(next, batch)?;
            next += batch.len();
        }
        Ok(())
    }

    fn push_batch(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        let d = &self.dimensions;
        let mut limit = RowLimit::new(rows.len());
        // The reference's per-row validation, in its order.
        for (i, values) in rows.iter().enumerate() {
            if let Err(error) = validate_context(position + i, self.position() + i, values, d) {
                limit.fail(i, error);
                break;
            }
        }
        let rows = &rows[..limit.rows];
        let signs = &self.signs;
        let prepared = limit.take(self.compute.map_range(0..rows.len(), |i| {
            Ok(quantize_activation(rows[i], &signs[..rows[i].len()]))
        })?);
        let context = if prepared.is_empty() {
            Vec::new()
        } else {
            let [sums] = weighted_sums(&mut self.compute, [&self.fc], &prepared)?;
            let (fc, norm, epsilon) = (&self.fc, &self.hidden_norm, d.epsilon);
            limit.take(self.compute.map_range(0..prepared.len(), |i| {
                Ok(restore(fc, &sums, i, &prepared[i])
                    .and_then(|y| reference::norm(&y, norm, epsilon)))
            })?)
        };
        let hidden = d.hidden;
        let prepared = limit.take(self.compute.map_range(0..context.len(), |i| {
            Ok(quantize_activation(&context[i], &signs[..hidden]))
        })?);
        let rows = prepared.len();
        if rows > 0 {
            // All layers project the SAME context; not sequential blocks (engine35.cpp:1778-1782).
            let matrices: Vec<&Matrix> = self.layers.iter().flat_map(|l| [&l.k, &l.v]).collect();
            let sums = weighted_sums_dyn(&mut self.compute, &matrices, &prepared)?;
            let layers = &self.layers;
            let base = self.ring.position();
            let entries = self.compute.map_range(0..rows * layers.len(), |index| {
                let (i, layer) = (index / layers.len(), index % layers.len());
                Ok(ring_entry(
                    &layers[layer],
                    &sums[2 * layer],
                    &sums[2 * layer + 1],
                    i,
                    &prepared[i],
                    base + i,
                    d.epsilon,
                ))
            })?;
            let mut entries = entries.into_iter();
            let mut ready = Vec::with_capacity(rows);
            for i in 0..rows {
                let row: Vec<_> = entries.by_ref().take(self.layers.len()).collect();
                match row.into_iter().collect::<Result<Vec<_>>>() {
                    Ok(row) => ready.push(row),
                    Err(error) => {
                        limit.fail(i, error);
                        break;
                    }
                }
            }
            for row in ready {
                self.ring.push_row(&row);
            }
        }
        limit.finish()
    }

    /// Embedding provider returns BF16-expanded main embedding rows for frontier/mask.
    pub fn propose(
        &mut self,
        frontier: u32,
        mut embedding: impl FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        let started = Instant::now();
        self.timing = DraftTiming::default();
        let result = self.propose_untimed(frontier, &mut embedding, choice);
        self.timing.total = started.elapsed();
        result
    }

    fn propose_untimed(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
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
        self.select(&hidden, frontier, choice)
    }

    /// Execute eight explicit embedding rows; returns normalized rows 1..7.
    pub fn block_hidden(&mut self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        let d = self.dimensions.clone();
        if embeddings.len() != BLOCK_ROWS || embeddings.iter().any(|r| r.len() != d.hidden) {
            return Err(Error::InvalidInput(
                "draft block needs eight hidden-width embedding rows".into(),
            ));
        }
        let position = self.position();
        if position
            .checked_add(BLOCK_ROWS)
            .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(Error::InvalidInput("draft position overflow".into()));
        }
        let mut hidden: Vec<_> = embeddings.iter().flatten().copied().collect();
        if hidden.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite("draft embeddings"));
        }
        if hidden.iter().any(|&x| bfloat(x).to_bits() != x.to_bits()) {
            return Err(Error::InvalidInput(
                "draft embeddings must be BF16 expanded to f32".into(),
            ));
        }
        let shape = Shape {
            query_heads: d.query_heads,
            kv_heads: d.kv_heads,
            window: d.window,
            position,
        };
        let (eps, h) = (d.epsilon, d.hidden);
        for i in 0..self.layers.len() {
            let mut clock = Instant::now();
            let l = &self.layers[i];
            let normalized = reference::norm_rows(&hidden, &l.ln1, eps)?;
            self.timing.block_other += lap(&mut clock);
            let [dynamic] = project(
                &mut self.compute,
                [&l.akp],
                &normalized,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            let filtered = reference::dynamic_filter(&normalized, &dynamic, &l.abase, h, 0);
            self.timing.block_other += lap(&mut clock);
            let [mut q, mut k, v] = project(
                &mut self.compute,
                [&l.q, &l.k, &l.v],
                &filtered,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            rope_rows(
                &mut self.compute,
                &mut q,
                &mut k,
                &l.qn,
                &l.kn,
                &d,
                position,
            )?;
            self.timing.block_other += lap(&mut clock);
            let attended = attention(
                &mut self.compute,
                &q,
                &k,
                &v,
                &self.ring,
                i,
                shape,
                self.simd,
            )?;
            self.timing.attention += lap(&mut clock);
            let [projected] = project(
                &mut self.compute,
                [&l.o],
                &attended,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            reference::add(
                &mut hidden,
                &reference::dynamic_filter(&projected, &dynamic, &l.abase, h, 1),
            )?;
            let normalized = reference::norm_rows(&hidden, &l.ln2, eps)?;
            self.timing.block_other += lap(&mut clock);
            let [dynamic] = project(
                &mut self.compute,
                [&l.mkp],
                &normalized,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            let filtered = reference::dynamic_filter(&normalized, &dynamic, &l.mbase, h, 0);
            self.timing.block_other += lap(&mut clock);
            let [gate, up] = project(
                &mut self.compute,
                [&l.gate, &l.up],
                &filtered,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            let width = d.feed_forward;
            let glu: Vec<f32> = self
                .compute
                .map_range(0..BLOCK_ROWS, |r| {
                    Ok(reference::glu(
                        &gate[r * width..(r + 1) * width],
                        &up[r * width..(r + 1) * width],
                    ))
                })?
                .concat();
            self.timing.block_other += lap(&mut clock);
            let [projected] = project(
                &mut self.compute,
                [&l.down],
                &glu,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            reference::add(
                &mut hidden,
                &reference::dynamic_filter(&projected, &dynamic, &l.mbase, h, 1),
            )?;
            self.timing.block_other += lap(&mut clock);
        }
        let mut clock = Instant::now();
        let result = reference::norm_rows(&hidden[h..], &self.final_norm, eps);
        self.timing.block_other += lap(&mut clock);
        result
    }

    /// Reduced head for the seven rows, then sequential correction and choice.
    /// Per-row failures are reported in the reference's row/stage order.
    fn select(
        &mut self,
        hidden: &[f32],
        frontier: u32,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        let (h, v, count) = (
            self.dimensions.hidden,
            self.dimensions.vocabulary,
            self.options.shortlist,
        );
        let rows: Vec<&[f32]> = hidden.chunks_exact(h).collect();
        let n = rows.len();
        let simd = self.simd;
        let mut errors: Vec<Option<Error>> = rows
            .iter()
            .map(|x| {
                x.iter()
                    .any(|v| !v.is_finite())
                    .then_some(Error::NonFinite("draft head input"))
            })
            .collect();
        let mut clock = Instant::now();
        let deinterleaved: Vec<_> = rows.iter().map(|x| draft_math::deinterleave(x)).collect();
        let projection = &self.head.projection;
        let chunks = self.compute.map_range(0..RANK / RANK_TASK, |chunk| {
            let mut out = vec![0.0; n * RANK_TASK];
            for k in 0..RANK_TASK {
                let weights = &projection[(chunk * RANK_TASK + k) * h..][..h];
                for (r, x) in rows.iter().enumerate() {
                    out[r * RANK_TASK + k] =
                        draft_math::low_rank(x, &deinterleaved[r], weights, simd);
                }
            }
            Ok(out)
        })?;
        let mut low = vec![0.0; n * RANK];
        for (chunk, values) in chunks.iter().enumerate() {
            for r in 0..n {
                low[r * RANK + chunk * RANK_TASK..][..RANK_TASK]
                    .copy_from_slice(&values[r * RANK_TASK..(r + 1) * RANK_TASK]);
            }
        }
        for (r, error) in errors.iter_mut().enumerate() {
            if error.is_none() && low[r * RANK..(r + 1) * RANK].iter().any(|v| !v.is_finite()) {
                *error = Some(Error::NonFinite("draft low-rank projection"));
            }
        }
        self.timing.head_projection += lap(&mut clock);
        let tiles = v.div_ceil(TILE);
        let (packed, scales) = (&self.head.approximate, &self.head.approximate_scale);
        let low = &low;
        let parts = self
            .compute
            .map_range(0..tiles.div_ceil(SCORE_TILES), |task| {
                let range = task * SCORE_TILES..((task + 1) * SCORE_TILES).min(tiles);
                Ok(draft_math::approximate_scores(
                    low, packed, scales, range, simd,
                ))
            })?;
        let mut scores = vec![Vec::with_capacity(v); n];
        for part in &parts {
            let tokens = part.len() / n;
            for (r, scores) in scores.iter_mut().enumerate() {
                scores.extend_from_slice(&part[r * tokens..(r + 1) * tokens]);
            }
        }
        drop(parts);
        self.timing.head_scores += lap(&mut clock);
        let scores = &scores;
        let shortlists = self
            .compute
            .map_range(0..n, |r| Ok(shortlist(&scores[r], count)))?;
        let mut ids = Vec::with_capacity(n);
        for (r, list) in shortlists.into_iter().enumerate() {
            match list {
                Ok(list) if errors[r].is_none() => ids.push(list),
                Ok(_) => ids.push(Vec::new()),
                Err(error) => {
                    errors[r].get_or_insert(error);
                    ids.push(Vec::new());
                }
            }
        }
        self.timing.head_shortlist += lap(&mut clock);
        let (head, ids) = (&self.head, &ids);
        let tasks = count.div_ceil(RESCORE_TASK);
        let rescored = self.compute.map_range(0..n * tasks, |index| {
            let (r, task) = (index / tasks, index % tasks);
            let list = ids[r].get(task * RESCORE_TASK..).unwrap_or(&[]);
            Ok(list
                .iter()
                .take(RESCORE_TASK)
                .map(|&token| Candidate {
                    token,
                    score: draft_math::rescore(
                        rows[r],
                        &deinterleaved[r],
                        head.rescore_row(token as usize),
                        simd,
                    ) * head.rescore_scale[token as usize],
                })
                .collect::<Vec<_>>())
        })?;
        let mut top = Vec::with_capacity(n);
        for (r, chunk) in rescored.chunks(tasks).enumerate() {
            if errors[r].is_some() {
                top.push(Vec::new());
                continue;
            }
            match top16(chunk.concat()) {
                Ok(list) => top.push(list),
                Err(error) => {
                    errors[r] = Some(error);
                    top.push(Vec::new());
                }
            }
        }
        self.timing.head_rescore += lap(&mut clock);
        let (hp, rank) = (&self.hp, self.dimensions.correction_rank);
        let features = self.compute.map_range(0..n, |r| {
            Ok(draft_math::correction_feature(rows[r], hp, rank))
        })?;
        let mut result = DraftProposal {
            tokens: Vec::new(),
            candidates: Vec::new(),
            corrected: Vec::new(),
        };
        let mut predecessor = frontier;
        let position = self.position();
        for (j, ((error, candidates), feature)) in
            errors.into_iter().zip(top).zip(features).enumerate()
        {
            if let Some(error) = error {
                return Err(error);
            }
            let feature = feature?;
            let pred = self.pred.row(predecessor as usize);
            let succ: Vec<_> = candidates
                .iter()
                .map(|c| self.succ.row(c.token as usize))
                .collect();
            let succ: Vec<&[f32]> = succ.iter().map(Vec::as_slice).collect();
            let corrected = correct_candidates(&feature, &pred, &succ, &candidates)?;
            let token = choose(&corrected, choice, (position + 1 + j) as u64)?;
            result.tokens.push(token);
            result.candidates.push(candidates);
            result.corrected.push(corrected);
            predecessor = token;
        }
        self.timing.head_select += lap(&mut clock);
        Ok(result)
    }
}

impl BlockDraft for FastDraft {
    fn dimensions(&self) -> &DraftDimensions {
        FastDraft::dimensions(self)
    }
    fn options(&self) -> &DraftOptions {
        FastDraft::options(self)
    }
    fn position(&self) -> usize {
        FastDraft::position(self)
    }
    fn snapshot(&self) -> DraftSnapshot {
        FastDraft::snapshot(self)
    }
    fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        FastDraft::restore(self, snapshot)
    }
    fn reset(&mut self) {
        FastDraft::reset(self)
    }
    fn reset_at(&mut self, position: usize) {
        self.ring.clear(position);
    }
    fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        FastDraft::push_rows(self, position, rows)
    }
    fn propose(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        FastDraft::propose(self, frontier, embedding, choice)
    }
}

fn lap(clock: &mut Instant) -> Duration {
    let now = Instant::now();
    let elapsed = now - *clock;
    *clock = now;
    elapsed
}

/// The rows of a batch still valid; the first failure (lowest row) is kept.
struct RowLimit {
    rows: usize,
    error: Option<Error>,
}
impl RowLimit {
    fn new(rows: usize) -> Self {
        Self { rows, error: None }
    }
    fn fail(&mut self, row: usize, error: Error) {
        if row < self.rows {
            self.rows = row;
            self.error = Some(error);
        }
    }
    /// Keep the leading successful results; a failure truncates later rows.
    fn take<T>(&mut self, results: Vec<Result<T>>) -> Vec<T> {
        let mut kept = Vec::with_capacity(results.len());
        for (row, result) in results.into_iter().enumerate().take(self.rows) {
            match result {
                Ok(value) => kept.push(value),
                Err(error) => {
                    self.fail(row, error);
                    break;
                }
            }
        }
        kept
    }
    fn finish(self) -> Result<()> {
        self.error.map_or(Ok(()), Err)
    }
}

/// The reference `push_context` checks for the row that will land at `expected`.
fn validate_context(
    position: usize,
    expected: usize,
    values: &[f32],
    d: &DraftDimensions,
) -> Result<()> {
    if position != expected || values.len() != d.features {
        return Err(Error::InvalidInput(
            "draft context position/shape mismatch".into(),
        ));
    }
    position
        .checked_add(BLOCK_ROWS)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| Error::InvalidInput("draft position overflow".into()))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite("draft context features"));
    }
    if values.iter().any(|&x| bfloat(x).to_bits() != x.to_bits()) {
        return Err(Error::InvalidInput(
            "context residual must be BF16 expanded to f32".into(),
        ));
    }
    Ok(())
}

/// One layer's ring key/value BF16 bits for context row `i` at `position`.
fn ring_entry(
    layer: &Layer,
    key_sums: &[i64],
    value_sums: &[i64],
    i: usize,
    prepared: &PreparedActivation,
    position: usize,
    epsilon: f32,
) -> Result<(Vec<u16>, Vec<u16>)> {
    let mut k = restore(&layer.k, key_sums, i, prepared)?;
    let v = restore(&layer.v, value_sums, i, prepared)?;
    reference::rope(&mut k, &layer.kn, position, epsilon)?;
    let key: Vec<_> = k.into_iter().map(f32_to_bf16).collect();
    let value: Vec<_> = v.into_iter().map(f32_to_bf16).collect();
    if key
        .iter()
        .chain(&value)
        .any(|&b| !bf16_to_f32(b).is_finite())
    {
        return Err(Error::NonFinite("draft ring BF16"));
    }
    Ok((key, value))
}

/// Row-wise RoPE of the block's queries and keys at positions `position + row`.
fn rope_rows(
    compute: &mut Compute,
    q: &mut [f32],
    k: &mut [f32],
    qn: &[f32],
    kn: &[f32],
    d: &DraftDimensions,
    position: usize,
) -> Result<()> {
    let (qw, kw) = (d.query_heads * HD, d.kv_heads * HD);
    let (q_in, k_in): (&[f32], &[f32]) = (q, k);
    // Reference order per row: queries then keys; errors surface in row order.
    let rotated = compute.map_range(0..2 * BLOCK_ROWS, |index| {
        let (row, keys) = (index / 2, index % 2 == 1);
        let (source, width, weights) = if keys { (k_in, kw, kn) } else { (q_in, qw, qn) };
        let mut x = source[row * width..(row + 1) * width].to_vec();
        reference::rope(&mut x, weights, position + row, d.epsilon)?;
        Ok(x)
    })?;
    for (index, x) in rotated.into_iter().enumerate() {
        let (row, keys) = (index / 2, index % 2 == 1);
        if keys {
            k[row * kw..(row + 1) * kw].copy_from_slice(&x);
        } else {
            q[row * qw..(row + 1) * qw].copy_from_slice(&x);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &mut Compute,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    ring: &DraftSnapshot,
    layer: usize,
    shape: Shape,
    simd: bool,
) -> Result<Vec<f32>> {
    let (keys, values) = (&ring.keys()[layer], &ring.values()[layer]);
    let parts = compute.map_range(0..shape.kv_heads * 32, |index| {
        Ok(draft_math::history_part(
            q,
            keys,
            values,
            shape,
            index / 32,
            index % 32,
            simd,
        ))
    })?;
    let heads = shape.query_heads;
    let group = heads / shape.kv_heads;
    let out = compute.map_range(0..BLOCK_ROWS * heads, |index| {
        let (row, head) = (index / heads, index % heads);
        let kv = head / group;
        draft_math::finish(q, k, v, &parts[kv * 32..(kv + 1) * 32], shape, row, head)
    })?;
    let out = out.concat();
    if out.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite("draft attention output"));
    }
    Ok(out)
}

fn restore(
    matrix: &Matrix,
    sums: &[i64],
    row: usize,
    prepared: &PreparedActivation,
) -> Result<Vec<f32>> {
    let d = matrix.descriptor();
    let n = d.physical_rows;
    restore_projection(
        &sums[row * n..(row + 1) * n],
        prepared,
        None,
        matrix.effective_row_scale(),
        None,
        d.rows,
    )
}

fn weighted_sums<const N: usize>(
    compute: &mut Compute,
    matrices: [&Matrix; N],
    prepared: &[PreparedActivation],
) -> Result<[Vec<i64>; N]> {
    let sums = weighted_sums_dyn(compute, &matrices, prepared)?;
    Ok(sums.try_into().expect("one sum array per matrix"))
}

/// Exact sums of matrices sharing one prepared input: one Metal command buffer.
fn weighted_sums_dyn(
    compute: &mut Compute,
    matrices: &[&Matrix],
    prepared: &[PreparedActivation],
) -> Result<Vec<Vec<i64>>> {
    let activation: Vec<i8> = prepared
        .iter()
        .flat_map(|p| p.activation8().iter().copied())
        .collect();
    let scales: Vec<i32> = prepared
        .iter()
        .flat_map(|p| p.integer_scale().iter().copied())
        .collect();
    compute.weighted_sums(matrices, &activation, &scales)
}

/// The reference's `linear_rows` for matrices sharing one input: quantization,
/// exact sums and restoration per row; each matrix's rows are concatenated.
fn project<const N: usize>(
    compute: &mut Compute,
    matrices: [&Matrix; N],
    input: &[f32],
    signs: &[i8],
    kernel: &mut Duration,
) -> Result<[Vec<f32>; N]> {
    let cols = matrices[0].descriptor().cols;
    let m = input.len() / cols;
    let prepared = compute.map_range(0..m, |i| {
        quantize_activation(&input[i * cols..(i + 1) * cols], &signs[..cols])
    })?;
    let start = Instant::now();
    let sums = weighted_sums(compute, matrices, &prepared)?;
    *kernel += start.elapsed();
    let rows = compute.map_range(0..N * m, |index| {
        let (j, i) = (index / m, index % m);
        restore(matrices[j], &sums[j], i, &prepared[i])
    })?;
    let mut rows = rows.into_iter();
    Ok(std::array::from_fn(|_| {
        rows.by_ref().take(m).flatten().collect()
    }))
}

fn matrix(mapping: &MappedModel, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
    let t = mapping
        .header()
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if t.rows != rows || t.cols != cols {
        return Err(invalid(format!("{name}: draft matrix shape mismatch")));
    }
    // Quantizing the table checks the table and scales are finite (D10); common
    // input signs, no output transform (engine35.cpp:1638, q35_bdraft.cu:474).
    mapping.matrix(name)
}

fn vector_descriptor<'a>(
    mapping: &'a MappedModel,
    name: &str,
    n: usize,
) -> Result<&'a yskn_format::VectorDescriptor> {
    let v = mapping
        .header()
        .vectors
        .iter()
        .find(|v| v.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if v.len != n || v.dtype != VectorType::Bf16 {
        return Err(invalid(format!("{name}: draft BF16 vector shape mismatch")));
    }
    Ok(v)
}

/// A large BF16 vector kept mapped, after the reference's finite check.
fn bf16_rows(mapping: &MappedModel, name: &str, rows: usize, width: usize) -> Result<Bf16Rows> {
    let n = rows
        .checked_mul(width)
        .ok_or_else(|| invalid("draft dimension overflow"))?;
    let v = vector_descriptor(mapping, name, n)?;
    let bytes = mapping.mapping();
    let range = crate::mapped::region(bytes, v.offset, 2 * n as u64)?;
    if bytes[range.clone()]
        .as_chunks::<2>()
        .0
        .iter()
        .any(|b| u16::from_le_bytes(*b) & 0x7f80 == 0x7f80)
    {
        return Err(Error::NonFinite("draft vector"));
    }
    Ok(Bf16Rows {
        mapping: Arc::clone(bytes),
        range,
        width,
    })
}
