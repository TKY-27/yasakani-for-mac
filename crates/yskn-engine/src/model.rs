//! Layer-major inference over mapped code streams and the reference's scalar primitives.
#[cfg(target_os = "macos")]
use crate::MetalEngine;
use crate::{Engine, Error, MappedModel, Matrix, Result, mapped::MappedEmbeddingTensor};
use std::{
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};
use yskn_ref::{
    KvMode, ModelDimensions,
    compressed_linear::{PreparedActivation, quantize_activation, restore_projection},
    gated_deltanet::{GatedDeltaNet, GatedDeltaNetState, round_projection},
    model::{Branch, Layer, LayerState, Weights, load_weights_with_kv, model_location_with_kv},
    residual_norm_mlp::{NormalizedInput, gated_product, prepare_normalization, residual_add},
};

pub struct Linear {
    matrix: Matrix,
    output_rotation: bool,
}
impl Linear {
    fn input_signs(&self) -> &[i8] {
        self.matrix.input_signs()
    }
}

/// Wall times cover nonoverlapping stages of a complete layer-major pass.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct PassTiming {
    pub position: usize,
    pub rows: usize,
    pub total: Duration,
    pub kernel: Duration,
    pub attention: Duration,
    /// Attention row preparation, including KV encoding (also norm/RoPE/gating).
    pub kv_encode: Duration,
    pub recurrence: Duration,
}
impl PassTiming {
    pub fn rest(&self) -> Duration {
        self.total
            .saturating_sub(self.kernel + self.attention + self.kv_encode + self.recurrence)
    }
}

pub struct Model {
    weights: Weights<Linear, MappedEmbeddingTensor>,
    engine: Compute,
    states: Vec<LayerState>,
    position: usize,
    context_capacity: usize,
    rows_per_pass: usize,
    kv_mode: KvMode,
    logits: Vec<Vec<f32>>,
    timings: Vec<PassTiming>,
}
impl Model {
    /// Default kernel and pool (`YSKN_KERNEL`), backend from `YSKN_BACKEND`
    /// (default Metal when a device exists, CPU otherwise), `YSKN_KV`
    /// (`int8`, `int8-profile`, `kv5`; default `int8`), 16 rows per pass.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mode = std::env::var_os("YSKN_KV")
            .map(|value| {
                yskn_ref::quality::parse_kv_mode(&value).map_err(|_| {
                    Error::InvalidInput("YSKN_KV expects int8|int8-profile|kv5".into())
                })
            })
            .transpose()?
            .unwrap_or_default();
        // SAFETY: forwarded file-stability contract is required of this method's caller.
        unsafe { Self::with_options(path, Engine::from_env()?, Backend::from_env()?, 16, mode) }
    }

    /// The CPU backend with an explicit kernel/pool and pass width (positive, with
    /// no implicit upper clamp). Never reads the environment.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_engine(
        path: impl AsRef<Path>,
        engine: Engine,
        rows_per_pass: usize,
    ) -> Result<Self> {
        // SAFETY: forwarded file-stability contract is required of this method's caller.
        unsafe { Self::with_backend(path, engine, Backend::Cpu, rows_per_pass) }
    }

    /// An explicit projection backend. `engine` runs all other row/head work, and
    /// the projections too when `backend` is [`Backend::Cpu`]. Never reads the
    /// environment. Metal fails without a device.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_backend(
        path: impl AsRef<Path>,
        engine: Engine,
        backend: Backend,
        rows_per_pass: usize,
    ) -> Result<Self> {
        // SAFETY: forwarded file-stability contract is required of this method's caller.
        unsafe { Self::with_options(path, engine, backend, rows_per_pass, KvMode::Int8) }
    }

    /// Explicit backend, pool, pass width and KV mode. Never reads the environment.
    /// Profile modes require a model folder containing the manifest's KVQP profile.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_options(
        path: impl AsRef<Path>,
        engine: Engine,
        backend: Backend,
        rows_per_pass: usize,
        kv_mode: KvMode,
    ) -> Result<Self> {
        if rows_per_pass == 0 {
            return Err(Error::InvalidInput("rows_per_pass must be positive".into()));
        }
        let engine = Compute::new(engine, backend)?;
        let (path, context_capacity, profile) = model_location_with_kv(path.as_ref(), kv_mode)?;
        // SAFETY: caller establishes file stability through model drop. Every weight view
        // shares this mapping, so none can outlive that external-file contract.
        let mapping = unsafe { MappedModel::open(path)? };
        let weights = load_weights_with_kv(
            &mut mapping.reader()?,
            kv_mode,
            profile,
            |_, name, rows, cols, output_rotation| {
                let matrix = mapping.matrix(name)?;
                let d = matrix.descriptor();
                if d.rows != rows || d.cols != cols {
                    return Err(Error::InvalidModel(format!(
                        "{name} must have shape [{rows}, {cols}]"
                    )));
                }
                if output_rotation
                    && (d.physical_rows % 512 != 0 || matrix.output_signs().is_none())
                {
                    return Err(Error::InvalidModel(
                        "body projection requires complete output rotations".into(),
                    ));
                }
                Ok(Linear {
                    matrix,
                    output_rotation,
                })
            },
            |_, name| mapping.embedding_tensor(name),
        )?;
        let states = weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        Ok(Self {
            weights,
            engine,
            states,
            position: 0,
            context_capacity,
            rows_per_pass,
            kv_mode,
            logits: Vec::new(),
            timings: Vec::new(),
        })
    }
    pub fn dimensions(&self) -> &ModelDimensions {
        &self.weights.dimensions
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }
    pub fn rows_per_pass(&self) -> usize {
        self.rows_per_pass
    }
    pub fn kv_mode(&self) -> KvMode {
        self.kv_mode
    }
    pub fn backend(&self) -> Backend {
        self.engine.backend()
    }
    /// The projection device: the Metal device name, or the CPU kernel and threads.
    pub fn device(&self) -> String {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.engine.metal {
            return metal.device_name();
        }
        let cpu = &self.engine.engine;
        format!("{} x{}", cpu.kernel().name(), cpu.threads())
    }
    pub fn row_logits(&self) -> &[Vec<f32>] {
        &self.logits
    }
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.last().map(Vec::as_slice)
    }
    pub fn pass_timings(&self) -> &[PassTiming] {
        &self.timings
    }
    pub fn reset(&mut self) -> Result<()> {
        let states = self
            .weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        self.states = states;
        self.position = 0;
        self.logits.clear();
        self.timings.clear();
        Ok(())
    }
    /// Validate the entire request, then publish state/logits only after all passes succeed.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        if self
            .position
            .checked_add(tokens.len())
            .is_none_or(|end| end > self.context_capacity)
        {
            return Err(Error::InvalidInput(
                "pass exceeds the session context capacity".into(),
            ));
        }
        if tokens
            .iter()
            .any(|&t| t as usize >= self.dimensions().vocabulary_size)
        {
            return Err(Error::InvalidInput("token ID exceeds vocabulary".into()));
        }
        let mut states = self.states.clone();
        let mut logits = Vec::with_capacity(tokens.len());
        let mut timings = Vec::new();
        for (pass, tokens) in tokens.chunks(self.rows_per_pass).enumerate() {
            let position = self.position + pass * self.rows_per_pass;
            let start = Instant::now();
            let mut timing = PassTiming {
                position,
                rows: tokens.len(),
                ..Default::default()
            };
            let mut hidden = self
                .engine
                .map_range(0..tokens.len(), |i| self.weights.embeddings.row(tokens[i]))?;
            for (layer, state) in self.weights.layers.iter().zip(&mut states) {
                hidden = forward(
                    &mut self.engine,
                    layer,
                    state,
                    position,
                    &hidden,
                    self.weights.dimensions.epsilon,
                    &mut timing,
                )?;
            }
            let input = normalize(
                &mut self.engine,
                &hidden,
                &self.weights.final_norm,
                &self.weights.head,
                self.weights.dimensions.epsilon,
            )?;
            let [head] =
                project_normalized(&mut self.engine, [&self.weights.head], &input, &mut timing)?;
            logits.extend(head);
            timing.total = start.elapsed();
            timings.push(timing);
        }
        self.states = states;
        self.position += tokens.len();
        self.logits = logits;
        self.timings = timings;
        Ok(())
    }
}

/// Where compressed projections run (their exact i64 sums are identical);
/// everything else runs on the CPU pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
}
impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
    /// `YSKN_BACKEND` (`cpu` or `metal`) when set; otherwise Metal when this
    /// process has a Metal device, else CPU.
    pub fn from_env() -> Result<Self> {
        match std::env::var("YSKN_BACKEND") {
            Ok(value) => value.parse(),
            Err(std::env::VarError::NotPresent) => Ok(Self::detected()),
            Err(std::env::VarError::NotUnicode(_)) => Err(Error::InvalidInput(
                "YSKN_BACKEND must be cpu or metal".into(),
            )),
        }
    }
    fn detected() -> Self {
        #[cfg(target_os = "macos")]
        if MetalEngine::device_exists() {
            return Self::Metal;
        }
        Self::Cpu
    }
}
impl std::str::FromStr for Backend {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            _ => Err(Error::InvalidInput(format!(
                "unknown backend {value}; expected cpu or metal"
            ))),
        }
    }
}

/// The CPU pool (all row/head work and the CPU kernel) plus the selected projector.
struct Compute {
    engine: Engine,
    #[cfg(target_os = "macos")]
    metal: Option<MetalEngine>,
}
impl Compute {
    fn new(engine: Engine, backend: Backend) -> Result<Self> {
        #[cfg(not(target_os = "macos"))]
        if backend == Backend::Metal {
            return Err(Error::InvalidInput(
                "the Metal backend requires macOS".into(),
            ));
        }
        Ok(Self {
            #[cfg(target_os = "macos")]
            metal: match backend {
                Backend::Cpu => None,
                Backend::Metal => Some(
                    MetalEngine::new()?
                        .ok_or_else(|| Error::InvalidInput("no Metal device".into()))?,
                ),
            },
            engine,
        })
    }
    fn backend(&self) -> Backend {
        #[cfg(target_os = "macos")]
        if self.metal.is_some() {
            return Backend::Metal;
        }
        Backend::Cpu
    }
    fn map_range<T: Send, F: Fn(usize) -> Result<T> + Sync>(
        &mut self,
        range: std::ops::Range<usize>,
        function: F,
    ) -> Result<Vec<T>> {
        self.engine.map_range(range, function)
    }
    /// Each matrix's M×N sums for one shared input: one Metal command buffer,
    /// or one CPU kernel call per matrix.
    fn weighted_sums(
        &mut self,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &mut self.metal {
            return metal.weighted_sums_shared(matrices, activation, scales);
        }
        matrices
            .iter()
            .map(|matrix| self.engine.weighted_sums(matrix, activation, scales))
            .collect()
    }
}

fn normalize(
    engine: &mut Compute,
    rows: &[Vec<f32>],
    norm: &[f32],
    linear: &Linear,
    epsilon: f32,
) -> Result<Vec<NormalizedInput>> {
    engine.map_range(0..rows.len(), |i| {
        prepare_normalization(&rows[i], norm, linear.input_signs(), epsilon)
    })
}
/// Projections that share one normalized input (and its first matrix's signs).
fn project_normalized<const N: usize>(
    engine: &mut Compute,
    linears: [&Linear; N],
    input: &[NormalizedInput],
    timing: &mut PassTiming,
) -> Result<[Vec<Vec<f32>>; N]> {
    let input: Vec<_> = input
        .iter()
        .map(|x| (&x.prepared, Some(x.inverse_rms)))
        .collect();
    project(engine, linears, &input, timing)
}
fn project<const N: usize>(
    engine: &mut Compute,
    linears: [&Linear; N],
    inputs: &[(&PreparedActivation, Option<f32>)],
    timing: &mut PassTiming,
) -> Result<[Vec<Vec<f32>>; N]> {
    let activation: Vec<_> = inputs
        .iter()
        .flat_map(|(p, _)| p.activation8().iter().copied())
        .collect();
    let scales: Vec<_> = inputs
        .iter()
        .flat_map(|(p, _)| p.integer_scale().iter().copied())
        .collect();
    let matrices = linears.map(|linear| &linear.matrix);
    let start = Instant::now();
    let sums = engine.weighted_sums(&matrices, &activation, &scales)?;
    timing.kernel += start.elapsed();
    let m = inputs.len();
    let mut rows = engine
        .map_range(0..N * m, |index| {
            let (linear, sums, i) = (linears[index / m], &sums[index / m], index % m);
            let n = linear.matrix.descriptor().physical_rows;
            restore_projection(
                &sums[i * n..(i + 1) * n],
                inputs[i].0,
                inputs[i].1,
                linear.matrix.effective_row_scale(),
                if linear.output_rotation {
                    linear.matrix.output_signs()
                } else {
                    None
                },
                linear.matrix.descriptor().rows,
            )
        })?
        .into_iter();
    Ok(std::array::from_fn(|_| rows.by_ref().take(m).collect()))
}
fn project_rows(
    engine: &mut Compute,
    linear: &Linear,
    rows: &[Vec<f32>],
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let prepared = engine.map_range(0..rows.len(), |i| {
        quantize_activation(&rows[i], linear.input_signs())
    })?;
    let inputs: Vec<_> = prepared.iter().map(|p| (p, None)).collect();
    let [projected] = project(engine, [linear], &inputs, timing)?;
    Ok(projected)
}
/// The reference's `GatedDeltaNet::step` stages for a pass of rows: rows are
/// prepared in parallel (they read only earlier rows' rounded projections), then
/// each value head runs its rows in ascending order on one worker. On error the
/// state is partly updated; `prefill` discards its state copy.
fn recurrence(
    engine: &mut Compute,
    layer: &GatedDeltaNet,
    state: &mut GatedDeltaNetState,
    qkv: &[Vec<f32>],
    z: &[Vec<f32>],
    input: &[NormalizedInput],
) -> Result<Vec<Vec<f32>>> {
    let rounded: Vec<_> = qkv.iter().map(|row| round_projection(row)).collect();
    let rows = {
        let taps: Vec<&[f32]> = state
            .history_rows()
            .chain(rounded.iter().map(Vec::as_slice))
            .collect();
        let width = taps.len() - rounded.len();
        engine.map_range(0..qkv.len(), |i| {
            layer.prepare(
                &taps[i..i + width],
                &qkv[i],
                &z[i],
                &input[i].weighted_hidden_bf16,
                input[i].inverse_rms,
            )
        })?
    };
    let heads: Vec<Mutex<&mut [f32]>> = state.recurrent_heads_mut().map(Mutex::new).collect();
    let outputs = engine.map_range(0..heads.len(), |head| {
        let mut head_state = heads[head]
            .lock()
            .map_err(|_| Error::InvalidInput("recurrent head lock poisoned".into()))?;
        rows.iter()
            .map(|row| layer.recur_head(&mut head_state, row, head))
            .collect::<Result<Vec<_>>>()
    })?;
    drop(heads);
    for row in &rows {
        layer.advance_history(state, row)?;
    }
    Ok((0..rows.len())
        .map(|i| {
            outputs
                .iter()
                .flat_map(|head| head[i].iter().copied())
                .collect()
        })
        .collect())
}
fn forward(
    engine: &mut Compute,
    layer: &Layer<Linear>,
    state: &mut LayerState,
    position: usize,
    hidden: &[Vec<f32>],
    epsilon: f32,
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let input_projection = match &layer.branch {
        Branch::Attention(l) => &l.query_gate,
        Branch::Recurrent(l) => &l.query_key_value,
    };
    let input = normalize(engine, hidden, &layer.input_norm, input_projection, epsilon)?;
    let projected = match (&layer.branch, state) {
        (Branch::Recurrent(l), LayerState::Recurrent(state)) => {
            let [qkv, z] =
                project_normalized(engine, [&l.query_key_value, &l.output_gate], &input, timing)?;
            let start = Instant::now();
            let gated = recurrence(engine, &l.recurrence, state, &qkv, &z, &input)?;
            timing.recurrence += start.elapsed();
            project_rows(engine, &l.output, &gated, timing)?
        }
        (Branch::Attention(l), LayerState::Attention(state)) => {
            let [q, k, v] =
                project_normalized(engine, [&l.query_gate, &l.key, &l.value], &input, timing)?;
            let start = Instant::now();
            let prepared = engine.map_range(0..hidden.len(), |i| {
                l.attention.prepare_row(position + i, &q[i], &k[i], &v[i])
            })?;
            timing.kv_encode += start.elapsed();
            let start = Instant::now();
            for (i, p) in prepared.iter().enumerate() {
                l.attention.append_prepared(state, position + i, p)?;
            }
            let heads = l.attention.query_heads();
            let outputs = engine.map_range(0..hidden.len() * heads, |i| {
                l.attention.output_head(
                    state,
                    position + i / heads,
                    &prepared[i / heads],
                    i % heads,
                )
            })?;
            let gated: Vec<_> = outputs
                .chunks(heads)
                .map(|heads| heads.iter().flatten().copied().collect())
                .collect();
            timing.attention += start.elapsed();
            project_rows(engine, &l.output, &gated, timing)?
        }
        _ => return Err(Error::InvalidModel("layer/state types differ".into())),
    };
    let hidden = engine.map_range(0..hidden.len(), |i| residual_add(&hidden[i], &projected[i]))?;
    let input = normalize(
        engine,
        &hidden,
        &layer.post_attention_norm,
        &layer.gate,
        epsilon,
    )?;
    let [gate, up] = project_normalized(engine, [&layer.gate, &layer.up], &input, timing)?;
    let product = engine.map_range(0..hidden.len(), |i| gated_product(&gate[i], &up[i]))?;
    let down = project_rows(engine, &layer.down, &product, timing)?;
    engine.map_range(0..hidden.len(), |i| residual_add(&hidden[i], &down[i]))
}

impl yskn_ref::quality::ScoringModel for Model {
    fn vocabulary_size(&self) -> usize {
        self.dimensions().vocabulary_size
    }
    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill(tokens)
    }
    fn row_logits(&self) -> &[Vec<f32>] {
        self.row_logits()
    }
}
