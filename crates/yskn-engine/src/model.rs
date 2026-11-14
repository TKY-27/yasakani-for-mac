//! Layer-major inference over mapped code streams and the reference's scalar primitives.
#[cfg(target_os = "macos")]
use crate::MetalEngine;
use crate::cpu_stages::rest::{
    gated_product, prepare_normalization, quantize_activation, residual_add, restore_projection,
};
use crate::{Engine, Error, MappedModel, Matrix, Result, mapped::MappedEmbeddingTensor};
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use yskn_ref::{
    KvMode, ModelDimensions,
    compressed_linear::PreparedActivation,
    gated_deltanet::{GatedDeltaNet, GatedDeltaNetRow, GatedDeltaNetState, round_projection},
    model::{
        Branch, CaptureOptions, CapturedRow, Layer, LayerState, Weights, load_weights_with_kv,
        model_location_with_kv,
    },
    residual_norm_mlp::NormalizedInput,
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

/// Vocabulary-head work requested by a committed prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitRows {
    All,
    Last,
    None,
}

/// An owned prompt checkpoint. Attention remains in the owning session.
pub struct Snapshot {
    owner: Arc<()>,
    position: usize,
    tokens: Vec<u32>,
    recurrence: Vec<Option<GatedDeltaNetState>>,
    committed_logits: Option<Vec<f32>>,
    captures: Vec<CapturedRow>,
    capture_options: CaptureOptions,
}

impl Snapshot {
    pub fn capture_bytes(&self) -> usize {
        yskn_ref::model::capture_allocated_bytes(&self.captures, self.captures.capacity())
    }
}

/// Persistent storage only; transient projection/attention work is not included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionMemory {
    pub capture_bytes: usize,
    pub recurrent_state_bytes: usize,
    pub replay_bytes: usize,
    pub replay_capacity_rows: usize,
    pub attention_row_capacities: Vec<usize>,
    pub pending_token_capacity: usize,
    pub history_capacity: usize,
}

pub struct Model {
    weights: Weights<Linear, MappedEmbeddingTensor>,
    engine: Compute,
    states: Vec<LayerState>,
    working: Vec<LayerState>,
    replay: Vec<Vec<GatedDeltaNetRow>>,
    pending_tokens: Vec<u32>,
    pending: bool,
    working_complete: bool,
    context_identity: Arc<()>,
    capture_options: CaptureOptions,
    captures: Vec<CapturedRow>,
    pending_captures: Vec<CapturedRow>,
    committed_logits: Option<Vec<f32>>,
    tokens: Vec<u32>,
    owner: Arc<()>,
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
    /// no implicit upper clamp). The process-wide CPU-stage diagnostic override still applies.
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
    /// the projections too when `backend` is [`Backend::Cpu`]. Backend selection is
    /// explicit; the CPU-stage diagnostic override still applies. Metal fails without a device.
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

    /// Explicit backend, pool, pass width and KV mode. Only the process-wide
    /// `YSKN_CPU_STAGES` diagnostic override is read from the environment.
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
        crate::cpu_stages::validate()?;
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
        let working = weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        let replay = weights
            .layers
            .iter()
            .map(|layer| match &layer.branch {
                Branch::Attention(_) => Ok(Vec::new()),
                Branch::Recurrent(layer) => (0..rows_per_pass)
                    .map(|_| layer.recurrence.new_row())
                    .collect::<Result<Vec<_>>>(),
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            working,
            replay,
            pending_tokens: Vec::with_capacity(rows_per_pass),
            pending: false,
            working_complete: false,
            context_identity: Arc::new(()),
            capture_options: CaptureOptions::default(),
            captures: Vec::new(),
            pending_captures: Vec::new(),
            committed_logits: None,
            tokens: Vec::new(),
            owner: Arc::new(()),
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
    pub fn has_provisional_pass(&self) -> bool {
        self.pending
    }

    /// Configure observation before any input has been committed.
    pub fn set_capture(&mut self, options: CaptureOptions) -> Result<()> {
        self.require_no_pending_pass()?;
        if self.position != 0 {
            return Err(Error::InvalidInput(
                "capture must be configured before prefill".into(),
            ));
        }
        options.validate(self.weights.layers.len())?;
        self.capture_options = options;
        self.captures.clear();
        self.pending_captures.clear();
        Ok(())
    }
    pub fn capture_options(&self) -> &CaptureOptions {
        &self.capture_options
    }
    pub fn context_identity(&self) -> Arc<()> {
        Arc::clone(&self.context_identity)
    }
    pub fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        std::mem::take(&mut self.captures)
    }
    pub fn captured_rows(&self) -> &[CapturedRow] {
        &self.captures
    }
    pub fn provisional_captures(&self) -> &[CapturedRow] {
        &self.pending_captures
    }
    pub fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        self.weights.embeddings.row(token)
    }
    fn retain_captures(&mut self, rows: Vec<CapturedRow>) {
        let excess =
            (self.captures.len() + rows.len()).saturating_sub(self.capture_options.max_rows);
        self.captures.drain(..excess.min(self.captures.len()));
        self.captures.extend(rows);
    }

    pub fn session_memory(&self) -> SessionMemory {
        SessionMemory {
            capture_bytes: yskn_ref::model::capture_allocated_bytes(
                &self.captures,
                self.captures.capacity(),
            ) + yskn_ref::model::capture_allocated_bytes(
                &self.pending_captures,
                self.pending_captures.capacity(),
            ),
            recurrent_state_bytes: self
                .states
                .iter()
                .chain(&self.working)
                .map(|state| match state {
                    LayerState::Recurrent(s) => s.allocated_bytes(),
                    LayerState::Attention(_) => 0,
                })
                .sum(),
            replay_bytes: self
                .replay
                .iter()
                .flatten()
                .map(GatedDeltaNetRow::allocated_bytes)
                .sum(),
            replay_capacity_rows: self.rows_per_pass,
            attention_row_capacities: self
                .states
                .iter()
                .filter_map(|state| match state {
                    LayerState::Attention(s) => Some(s.capacity()),
                    _ => None,
                })
                .collect(),
            pending_token_capacity: self.pending_tokens.capacity(),
            history_capacity: self.tokens.capacity(),
        }
    }

    /// Digest the committed state, including cache codes, history and logit bits.
    pub fn state_digest(&self) -> u64 {
        let mut digest = 0xcbf29ce484222325;
        for byte in (self.position as u64).to_le_bytes() {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        for state in &self.states {
            match state {
                LayerState::Attention(state) => {
                    digest = digest.wrapping_mul(0x100000001b3);
                    state.update_prefix_digest(&mut digest, self.position);
                }
                LayerState::Recurrent(_) => state.update_digest(&mut digest),
            }
        }
        for byte in self
            .committed_logits
            .iter()
            .flatten()
            .flat_map(|v| v.to_bits().to_le_bytes())
        {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        digest
    }

    pub fn reset(&mut self) -> Result<()> {
        self.states = self
            .weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        self.position = 0;
        self.pending = false;
        self.working_complete = false;
        self.captures.clear();
        self.pending_captures.clear();
        self.pending_tokens.clear();
        self.tokens.clear();
        self.committed_logits = None;
        self.logits.clear();
        self.timings.clear();
        self.owner = Arc::new(());
        self.context_identity = Arc::new(());
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        self.require_no_pending_pass()?;
        Ok(Snapshot {
            owner: Arc::clone(&self.owner),
            position: self.position,
            tokens: self.tokens.clone(),
            recurrence: self
                .states
                .iter()
                .map(|state| match state {
                    LayerState::Attention(_) => None,
                    LayerState::Recurrent(state) => Some(state.clone()),
                })
                .collect(),
            committed_logits: self.committed_logits.clone(),
            captures: self.captures.clone(),
            capture_options: self.capture_options.clone(),
        })
    }

    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.require_no_pending_pass()?;
        if !Arc::ptr_eq(&snapshot.owner, &self.owner)
            || snapshot.position > self.position
            || !self.tokens.starts_with(&snapshot.tokens)
        {
            return Err(Error::InvalidInput(
                "snapshot is not a retained prefix of this session".into(),
            ));
        }
        if snapshot.capture_options != self.capture_options {
            return Err(Error::InvalidInput(
                "snapshot capture configuration changed".into(),
            ));
        }
        // Validate every cache before publishing any part of the checkpoint.
        for state in &self.states {
            if let LayerState::Attention(state) = state
                && state.len() < snapshot.position
            {
                return Err(Error::InvalidInput(
                    "snapshot attention prefix is unavailable".into(),
                ));
            }
        }
        for (state, saved) in self.states.iter_mut().zip(&snapshot.recurrence) {
            match (state, saved) {
                (LayerState::Attention(state), None) => state.truncate(snapshot.position)?,
                (LayerState::Recurrent(state), Some(saved)) => state.copy_from(saved),
                _ => unreachable!("snapshot belongs to the same model"),
            }
        }
        self.context_identity = Arc::new(());
        self.captures.clone_from(&snapshot.captures);
        self.position = snapshot.position;
        self.tokens.truncate(snapshot.position);
        self.committed_logits.clone_from(&snapshot.committed_logits);
        self.logits = self.committed_logits.iter().cloned().collect();
        self.timings.clear();
        Ok(())
    }

    fn require_no_pending_pass(&self) -> Result<()> {
        if self.pending {
            Err(Error::InvalidInput(
                "commit or discard the current provisional pass first".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
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
        Ok(())
    }

    fn prepare_working(&mut self) {
        for (work, committed) in self.working.iter_mut().zip(&self.states) {
            if let (LayerState::Recurrent(work), LayerState::Recurrent(committed)) =
                (work, committed)
            {
                if crate::cpu_stages::enabled() {
                    work.copy_history_from(committed);
                } else {
                    work.copy_from(committed);
                }
            }
        }
    }

    fn publish_recurrence(&mut self) {
        for (work, committed) in self.working.iter_mut().zip(&mut self.states) {
            if matches!(committed, LayerState::Recurrent(_)) {
                std::mem::swap(work, committed);
            }
        }
    }

    fn truncate_attention(&mut self, position: usize) {
        for state in &mut self.states {
            if let LayerState::Attention(state) = state {
                state
                    .truncate(position)
                    .expect("rollback only removes appended rows");
            }
        }
    }

    /// Scoring retains all rows; prompt prefill can instead request Last or None.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill_with_logits(tokens, LogitRows::All)
    }

    /// Atomically commit the complete prompt, keeping only requested vocabulary heads.
    /// None clears predictions; Last computes only the final row's vocabulary head.
    pub fn prefill_with_logits(&mut self, tokens: &[u32], requested: LogitRows) -> Result<()> {
        self.require_no_pending_pass()?;
        self.validate_tokens(tokens)?;
        if tokens.is_empty() {
            return Ok(());
        }
        let copy_start = Instant::now();
        self.prepare_working();
        let copy_time = copy_start.elapsed();
        let result = (|| {
            let mut logits = Vec::new();
            let mut timings = Vec::new();
            let mut captures = Vec::new();
            for (pass, chunk) in tokens.chunks(self.rows_per_pass).enumerate() {
                let position = self.position + pass * self.rows_per_pass;
                let rows = match requested {
                    LogitRows::Last if position + chunk.len() != self.position + tokens.len() => {
                        LogitRows::None
                    }
                    rows => rows,
                };
                let (head, mut timing, captured) = self.compute_pass(
                    chunk,
                    position,
                    rows,
                    false,
                    (self.position + tokens.len()).saturating_sub(self.capture_options.max_rows),
                )?;
                if pass == 0 {
                    timing.recurrence += copy_time;
                    timing.total += copy_time;
                }
                captures.extend(captured);
                let excess = captures.len().saturating_sub(self.capture_options.max_rows);
                captures.drain(..excess);
                logits.extend(head);
                timings.push(timing);
            }
            Ok::<_, Error>((logits, timings, captures))
        })();
        match result {
            Ok((logits, timings, captures)) => {
                self.retain_captures(captures);
                self.publish_recurrence();
                self.position += tokens.len();
                self.tokens.extend_from_slice(tokens);
                self.committed_logits = logits.last().cloned();
                self.logits = logits;
                self.timings = timings;
                Ok(())
            }
            Err(error) => {
                self.truncate_attention(self.position);
                Err(error)
            }
        }
    }

    /// A single bounded causal pass. Attention is appended in place; committed
    /// recurrence stays untouched while the preallocated working state advances.
    pub fn run_provisional(&mut self, tokens: &[u32]) -> Result<&[Vec<f32>]> {
        self.require_no_pending_pass()?;
        if tokens.is_empty() || tokens.len() > self.rows_per_pass {
            return Err(Error::InvalidInput(
                "provisional width must be in 1..=rows_per_pass".into(),
            ));
        }
        self.validate_tokens(tokens)?;
        let copy_start = Instant::now();
        self.prepare_working();
        let copy_time = copy_start.elapsed();
        match self.compute_pass(tokens, self.position, LogitRows::All, true, 0) {
            Ok((logits, mut timing, captures)) => {
                timing.recurrence += copy_time;
                timing.total += copy_time;
                self.pending_captures = captures;
                self.working_complete = true;
                self.pending_tokens.clear();
                self.pending_tokens.extend_from_slice(tokens);
                self.pending = true;
                self.logits = logits;
                self.timings = vec![timing];
                Ok(&self.logits)
            }
            Err(error) => {
                self.truncate_attention(self.position);
                Err(error)
            }
        }
    }

    /// Commit exactly k input rows. Partial acceptance replays recurrence only;
    /// full acceptance simply publishes the already computed working state.
    pub fn commit(&mut self, k: usize) -> Result<()> {
        if !self.pending || k > self.pending_tokens.len() {
            return Err(Error::InvalidInput(
                "accepted rows exceed the pending pass or no pass exists".into(),
            ));
        }
        if k == 0 {
            self.discard_provisional();
            return Ok(());
        }
        if k < self.pending_tokens.len() || !self.working_complete {
            self.working_complete = false;
            self.prepare_working();
            let result = self
                .weights
                .layers
                .iter()
                .zip(&mut self.working)
                .zip(&self.replay)
                .zip(&self.states)
                .try_for_each(|(((layer, state), rows), source)| {
                    if let (Branch::Recurrent(layer), LayerState::Recurrent(state)) =
                        (&layer.branch, state)
                    {
                        let source = if crate::cpu_stages::enabled() {
                            if let LayerState::Recurrent(source) = source {
                                Some(source)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        replay_recurrence(
                            &mut self.engine,
                            &layer.recurrence,
                            state,
                            source,
                            &rows[..k],
                        )?;
                    }
                    Ok::<_, Error>(())
                });
            result?;
        }
        let accepted = self
            .pending_captures
            .drain(..k.min(self.pending_captures.len()))
            .collect::<Vec<_>>();
        self.retain_captures(accepted);
        self.pending_captures.clear();
        self.publish_recurrence();
        self.position += k;
        self.tokens.extend_from_slice(&self.pending_tokens[..k]);
        self.truncate_attention(self.position);
        self.logits.truncate(k);
        self.committed_logits = self.logits.last().cloned();
        self.pending_tokens.clear();
        self.pending = false;
        Ok(())
    }

    pub fn discard_provisional(&mut self) {
        if self.pending {
            self.truncate_attention(self.position);
            self.pending_tokens.clear();
            self.pending_captures.clear();
            self.pending = false;
            self.logits = self.committed_logits.iter().cloned().collect();
        }
    }

    fn compute_pass(
        &mut self,
        tokens: &[u32],
        position: usize,
        requested: LogitRows,
        cache_replay: bool,
        capture_from: usize,
    ) -> Result<(Vec<Vec<f32>>, PassTiming, Vec<CapturedRow>)> {
        let start = Instant::now();
        let mut timing = PassTiming {
            position,
            rows: tokens.len(),
            ..Default::default()
        };
        let mut hidden = self
            .engine
            .map_range(0..tokens.len(), |i| self.weights.embeddings.row(tokens[i]))?;
        let capture_start = if cache_replay {
            0
        } else {
            tokens.len().saturating_sub(self.capture_options.max_rows)
        }
        .max(capture_from.saturating_sub(position).min(tokens.len()));
        let mut captures = if self.capture_options.layers.is_empty() {
            Vec::new()
        } else {
            (capture_start..tokens.len())
                .map(|row| CapturedRow {
                    position: position + row,
                    values: Vec::with_capacity(
                        self.capture_options.layers.len() * self.weights.dimensions.hidden_width,
                    ),
                })
                .collect::<Vec<_>>()
        };
        for (i, layer) in self.weights.layers.iter().enumerate() {
            let (state, source) = match layer.branch {
                Branch::Attention(_) => (&mut self.states[i], None),
                Branch::Recurrent(_) => (
                    &mut self.working[i],
                    if crate::cpu_stages::enabled() && position == self.position {
                        if let LayerState::Recurrent(source) = &self.states[i] {
                            Some(source)
                        } else {
                            None
                        }
                    } else {
                        None
                    },
                ),
            };
            hidden = forward(
                &mut self.engine,
                layer,
                state,
                source,
                &hidden,
                self.weights.dimensions.epsilon,
                &mut timing,
                if cache_replay {
                    Some(&mut self.replay[i])
                } else {
                    None
                },
            )?;
            if self.capture_options.layers.binary_search(&i).is_ok() {
                for (capture, row) in captures.iter_mut().zip(&hidden[capture_start..]) {
                    capture.values.extend_from_slice(row);
                }
            }
        }
        let selected = match requested {
            LogitRows::All => hidden.as_slice(),
            LogitRows::Last => &hidden[hidden.len() - 1..],
            LogitRows::None => &[],
        };
        let head = if selected.is_empty() {
            Vec::new()
        } else {
            let input = normalize(
                &mut self.engine,
                selected,
                &self.weights.final_norm,
                &self.weights.head,
                self.weights.dimensions.epsilon,
            )?;
            let [head] =
                project_normalized(&mut self.engine, [&self.weights.head], &input, &mut timing)?;
            head
        };
        timing.total = start.elapsed();
        Ok((head, timing, captures))
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
/// state is partly updated; the committed recurrent state remains untouched.
#[allow(clippy::too_many_arguments)]
fn recurrence(
    engine: &mut Compute,
    layer: &GatedDeltaNet,
    state: &mut GatedDeltaNetState,
    source: Option<&GatedDeltaNetState>,
    qkv: &[Vec<f32>],
    z: &[Vec<f32>],
    input: &[NormalizedInput],
    cache: Option<&mut [GatedDeltaNetRow]>,
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
    if let Some(cache) = cache {
        for (slot, row) in cache.iter_mut().zip(&rows) {
            slot.copy_from(row);
        }
    }
    replay_recurrence(engine, layer, state, source, &rows)
}

fn replay_recurrence(
    engine: &mut Compute,
    layer: &GatedDeltaNet,
    state: &mut GatedDeltaNetState,
    source: Option<&GatedDeltaNetState>,
    rows: &[GatedDeltaNetRow],
) -> Result<Vec<Vec<f32>>> {
    let heads: Vec<Mutex<&mut [f32]>> = state.recurrent_heads_mut().map(Mutex::new).collect();
    let sources = source.map(|s| s.recurrent_heads().collect::<Vec<_>>());
    let outputs = engine.map_range(0..heads.len(), |head| {
        let mut head_state = heads[head]
            .lock()
            .map_err(|_| Error::InvalidInput("recurrent head lock poisoned".into()))?;
        rows.iter()
            .enumerate()
            .map(|(i, row)| {
                crate::cpu_stages::recurrence::recur_head(
                    layer,
                    &mut head_state,
                    if i == 0 {
                        sources.as_ref().map(|s| s[head])
                    } else {
                        None
                    },
                    row,
                    head,
                )
            })
            .collect::<Result<Vec<_>>>()
    })?;
    drop(heads);
    for row in rows {
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
/// Prepare, append and attend a batch of projected rows using the engine pool.
fn attention_pass(
    engine: &mut Compute,
    attention: &yskn_ref::attention::Attention,
    state: &mut yskn_ref::attention::AttentionState,
    [q, k, v]: [&[Vec<f32>]; 3],
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let position = timing.position;
    let start = Instant::now();
    let prepared = if crate::cpu_stages::enabled() {
        let parallel = (|| {
            let sides_per_row = attention.kv_heads() * 2;
            let sides = engine.map_range(0..q.len() * sides_per_row, |index| {
                let row = index / sides_per_row;
                let side = index % 2;
                attention.prepare_cache_side(
                    position + row,
                    index % sides_per_row / 2,
                    side,
                    if side == 0 { &k[row] } else { &v[row] },
                )
            })?;
            let queries = engine.map_range(0..q.len(), |i| {
                attention.prepare_queries(position + i, &q[i], &k[i], &v[i])
            })?;
            let mut sides = sides.into_iter();
            queries
                .into_iter()
                .map(|query| {
                    attention.finish_prepared(query, sides.by_ref().take(sides_per_row).collect())
                })
                .collect::<Result<Vec<_>>>()
        })();
        match parallel {
            Ok(prepared) => prepared,
            Err(error) => {
                // On failure, replay in scalar row/stage order before any cache append.
                // Success retains independent KV-side tasks and never repeats preparation.
                for i in 0..q.len() {
                    attention.prepare_row(position + i, &q[i], &k[i], &v[i])?;
                }
                return Err(error);
            }
        }
    } else {
        engine.map_range(0..q.len(), |i| {
            attention.prepare_row(position + i, &q[i], &k[i], &v[i])
        })?
    };
    timing.kv_encode += start.elapsed();
    let start = Instant::now();
    for (i, p) in prepared.iter().enumerate() {
        attention.append_prepared(state, position + i, p)?;
    }
    let heads = attention.query_heads();
    let outputs = engine.map_range(0..q.len() * heads, |i| {
        crate::cpu_stages::attention::output_head(
            attention,
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
    Ok(gated)
}

#[allow(clippy::too_many_arguments)]
fn forward(
    engine: &mut Compute,
    layer: &Layer<Linear>,
    state: &mut LayerState,
    source: Option<&GatedDeltaNetState>,
    hidden: &[Vec<f32>],
    epsilon: f32,
    timing: &mut PassTiming,
    cache: Option<&mut [GatedDeltaNetRow]>,
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
            let gated = recurrence(
                engine,
                &l.recurrence,
                state,
                source,
                &qkv,
                &z,
                &input,
                cache,
            )?;
            timing.recurrence += start.elapsed();
            project_rows(engine, &l.output, &gated, timing)?
        }
        (Branch::Attention(l), LayerState::Attention(state)) => {
            let [q, k, v] =
                project_normalized(engine, [&l.query_gate, &l.key, &l.value], &input, timing)?;
            let gated = attention_pass(engine, &l.attention, state, [&q, &k, &v], timing)?;
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

impl yskn_ref::speculation::Verifier for Model {
    type Error = Error;
    fn context_identity(&self) -> Option<Arc<()>> {
        Some(self.context_identity())
    }
    fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        self.take_captured_rows()
    }

    fn dimensions(&self) -> Option<&ModelDimensions> {
        Some(self.dimensions())
    }
    fn position(&self) -> usize {
        self.position()
    }
    fn context_capacity(&self) -> usize {
        self.context_capacity()
    }
    fn max_pass_width(&self) -> usize {
        self.rows_per_pass()
    }
    fn run_provisional(&mut self, tokens: &[u32]) -> Result<()> {
        self.run_provisional(tokens).map(|_| ())
    }
    fn row_logits(&self, row: usize) -> Option<&[f32]> {
        self.row_logits().get(row).map(Vec::as_slice)
    }
    fn logits(&self) -> Option<&[f32]> {
        self.logits()
    }
    fn commit(&mut self, rows: usize) -> Result<()> {
        self.commit(rows)
    }
    fn capture_options(&self) -> Option<&CaptureOptions> {
        Some(self.capture_options())
    }
    fn captured_rows(&self) -> &[CapturedRow] {
        self.captured_rows()
    }
    fn provisional_captures(&self) -> &[CapturedRow] {
        self.provisional_captures()
    }
    fn embedding(&self, token: u32) -> yskn_ref::Result<Vec<f32>> {
        self.embedding(token)
            .map_err(|error| yskn_ref::Error::InvalidInput(error.to_string()))
    }
    fn discard(&mut self) {
        self.discard_provisional();
    }
}

impl yskn_ref::speculation::CheckpointVerifier for Model {
    type Snapshot = Snapshot;
    fn snapshot(&self) -> Result<Snapshot> {
        self.snapshot()
    }
    fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.restore(snapshot)
    }
}

#[cfg(test)]
#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
pub(crate) mod session_test_support;

#[cfg(test)]
mod session_tests {
    use super::*;
    use crate::Kernel;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn attention_pass_errors_follow_scalar_row_and_stage_order() {
        use yskn_ref::attention::{Attention, AttentionDimensions};
        let layer = Attention::new(
            AttentionDimensions {
                query_heads: 2,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
            },
            vec![0.0; 256],
            vec![0.0; 256],
            1e-6,
        )
        .unwrap();
        for threads in [1, 4] {
            let mut engine = Compute::new(
                Engine::new(Kernel::Portable, threads).unwrap(),
                Backend::Cpu,
            )
            .unwrap();
            // Same-row query before value, earlier query before later value,
            // earlier value before later query, and final gate check after cache encoding.
            for (rows, query_row, value_row, bad_gate, expected_context) in [
                (1, 0, 0, false, "attention head sum of squares"),
                (2, 0, 1, false, "attention head sum of squares"),
                (2, 1, 0, false, "attention cache input"),
                (1, 0, 0, true, "attention cache input"),
            ] {
                let mut q = vec![vec![0.25; 1024]; rows];
                let k = vec![vec![0.25; 512]; rows];
                let mut v = vec![vec![0.25; 512]; rows];
                if bad_gate {
                    q[query_row][256..512].fill(f32::MAX);
                } else {
                    q[query_row][..256].fill(1e20);
                }
                v[value_row][..256].fill(2e38);
                let expected = (0..rows)
                    .map(|i| layer.prepare_row(i, &q[i], &k[i], &v[i]))
                    .collect::<Result<Vec<_>>>()
                    .err()
                    .unwrap();
                assert!(
                    matches!(expected, Error::NonFinite(context) if context == expected_context)
                );
                let mut state = layer.new_state().unwrap();
                let before = state.clone();
                let actual = attention_pass(
                    &mut engine,
                    &layer,
                    &mut state,
                    [&q, &k, &v],
                    &mut PassTiming::default(),
                )
                .unwrap_err();
                assert_eq!(
                    actual.to_string(),
                    expected.to_string(),
                    "threads={threads} rows={rows} query={query_row} value={value_row} gate={bad_gate}"
                );
                assert_eq!(state, before, "failed preparation appended cache rows");
                // The same pool/state can still complete the actual attention pass.
                let q = vec![vec![0.25; 1024]; rows];
                let v = vec![vec![0.25; 512]; rows];
                let actual = attention_pass(
                    &mut engine,
                    &layer,
                    &mut state,
                    [&q, &k, &v],
                    &mut PassTiming::default(),
                )
                .unwrap();
                let mut reference = layer.new_state().unwrap();
                for i in 0..rows {
                    let expected = layer.step(&mut reference, i, &q[i], &k[i], &v[i]).unwrap();
                    assert_eq!(
                        actual[i].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                }
                assert_eq!(state, reference);
            }
        }
    }

    fn fixture_path() -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yskn-rollback-{}-{}.yskn",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let bytes = super::session_test_support::synthetic_model(
            187,
            &["full_attention", "linear_attention"],
        );
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn failed_later_layer_truncates_already_appended_attention() {
        let path = fixture_path();
        let mut reference = yskn_ref::Model::load(&path).unwrap();
        // SAFETY: this test never modifies the backing file while the mapping is alive.
        let mut engine =
            unsafe { Model::with_engine(&path, Engine::new(Kernel::Portable, 2).unwrap(), 4) }
                .unwrap();
        engine
            .set_capture(CaptureOptions {
                layers: vec![0, 1],
                max_rows: 3,
            })
            .unwrap();
        engine.prefill(&[1, 2]).unwrap();
        reference.prefill(&[1, 2]).unwrap();
        let digest = engine.state_digest();
        let logits = engine.row_logits().to_vec();
        let captures = engine.captured_rows().to_vec();
        // Owned norm data can be poisoned without mutating any mapped weight bytes.
        // Layer zero has appended both attention rows before layer one reads this.
        let saved = engine.weights.layers[1].input_norm[0];
        engine.weights.layers[1].input_norm[0] = f32::NAN;
        assert!(matches!(
            engine.run_provisional(&[3, 4]),
            Err(Error::NonFinite(_))
        ));
        engine.weights.layers[1].input_norm[0] = saved;
        assert_eq!(engine.state_digest(), digest);
        assert_eq!(engine.row_logits(), logits);
        assert!(!engine.has_provisional_pass());
        assert_eq!(engine.captured_rows(), captures);
        assert!(engine.provisional_captures().is_empty());
        // The digest only includes the committed prefix: a successful probe is
        // essential to detect an untruncated, otherwise invisible cache suffix.
        engine.run_provisional(&[5, 6]).unwrap();
        reference.run_provisional(&[5, 6]).unwrap();
        assert_eq!(engine.row_logits(), reference.row_logits());
        engine.commit(2).unwrap();
        reference.commit(2).unwrap();
        assert_eq!(engine.state_digest(), reference.state_digest());
        drop(engine);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_partial_replay_retains_pending_pass_and_full_retry_replays_correctly() {
        use yskn_ref::gated_deltanet::{GatedDeltaNetDimensions, GatedDeltaNetWeights};
        let path = fixture_path();
        let mut reference = yskn_ref::Model::load(&path).unwrap();
        // SAFETY: backing file remains immutable until model drop.
        let mut engine =
            unsafe { Model::with_engine(&path, Engine::new(Kernel::Portable, 2).unwrap(), 4) }
                .unwrap();
        engine.prefill(&[1, 2]).unwrap();
        reference.prefill(&[1, 2]).unwrap();
        engine.run_provisional(&[3, 4, 5]).unwrap();
        reference.run_provisional(&[3, 4, 5]).unwrap();
        let digest = engine.state_digest();
        let logits = engine.row_logits().to_vec();
        // An incompatible cached row forces the recurrence replay error path.
        // The actual row is restored before retrying the still-pending pass.
        let other = GatedDeltaNet::new(
            GatedDeltaNetDimensions {
                hidden_width: 512,
                query_key_heads: 1,
                value_heads: 1,
                head_dim: 128,
                convolution_width: 4,
            },
            GatedDeltaNetWeights {
                projection_a: vec![0.0; 512],
                projection_b: vec![0.0; 512],
                convolution_weight: vec![0.0; 384 * 4],
                log_decay_weight: vec![0.0],
                time_bias: vec![0.0],
                normalization_weight: vec![1.0; 128],
            },
            1e-6,
        )
        .unwrap();
        let saved = std::mem::replace(&mut engine.replay[1][0], other.new_row().unwrap());
        assert!(engine.commit(1).is_err());
        assert!(engine.has_provisional_pass());
        assert_eq!(engine.state_digest(), digest);
        assert_eq!(engine.row_logits(), logits);
        assert!(engine.snapshot().is_err());
        engine.replay[1][0] = saved;
        engine.commit(3).unwrap();
        reference.commit(3).unwrap();
        assert_eq!(engine.state_digest(), reference.state_digest());
        assert_eq!(engine.row_logits(), reference.row_logits());
        drop(engine);
        std::fs::remove_file(path).unwrap();
    }
}
