//! [Chapter 01: main-model architecture](../../../docs/spec/01_MODEL_ARCHITECTURE.md).

use std::fs;
use std::io::{Read, Seek};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use serde_json::Value;
use yskn_format::{Container, VectorType};

use crate::attention::{Attention, AttentionDimensions, AttentionState};
use crate::compressed_linear::{CompressedLinear, Embedding, EmbeddingTensor, quantize_activation};
use crate::gated_deltanet::{
    GatedDeltaNet, GatedDeltaNetDimensions, GatedDeltaNetState, GatedDeltaNetWeights,
};
use crate::kv_profile::KvProfile;
use crate::output_head_sampler::output_head;
use crate::residual_norm_mlp::{gated_product, prepare_normalization, residual_add};
use crate::{Error, KvMode, Result, require_finite};

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDimensions {
    pub hidden_width: usize,
    pub feed_forward_width: usize,
    pub vocabulary_size: usize,
    pub layer_count: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub attention_head_width: usize,
    pub epsilon: f32,
}

impl ModelDimensions {
    fn read(model: &Value) -> Result<Self> {
        if model.get("arch").and_then(Value::as_str) != Some("qwen3_5") {
            return Err(Error::InvalidModel(
                "main reference requires arch=qwen3_5".into(),
            ));
        }
        let dimensions = Self {
            hidden_width: positive_dimension(model, "hidden")?,
            feed_forward_width: positive_dimension(model, "ffn")?,
            vocabulary_size: positive_dimension(model, "vocab")?,
            layer_count: positive_dimension(model, "layers")?,
            query_heads: positive_dimension(model, "heads")?,
            kv_heads: positive_dimension(model, "kv_heads")?,
            attention_head_width: positive_dimension(model, "head_dim")?,
            epsilon: model
                .get("eps")
                .and_then(Value::as_f64)
                .ok_or_else(|| Error::InvalidModel("model.eps must be a number".into()))?
                as f32,
        };
        if dimensions.hidden_width % 512 != 0 || dimensions.feed_forward_width % 512 != 0 {
            return Err(Error::InvalidModel(
                "hidden and feed-forward widths must be multiples of 512".into(),
            ));
        }
        if dimensions.vocabulary_size > u32::MAX as usize {
            return Err(Error::InvalidModel(
                "vocabulary exceeds u32 token IDs".into(),
            ));
        }
        if dimensions.query_heads % dimensions.kv_heads != 0 {
            return Err(Error::InvalidModel(
                "query heads must be divisible by KV heads".into(),
            ));
        }
        if !dimensions.epsilon.is_finite() || dimensions.epsilon <= 0.0 {
            return Err(Error::InvalidModel(
                "model.eps must be finite and positive".into(),
            ));
        }
        Ok(dimensions)
    }
}

pub enum Embeddings<T = yskn_format::Tensor> {
    Coded(Box<Embedding<T>>),
    Plain {
        values: Vec<f32>,
        hidden_width: usize,
    },
}

impl<T: EmbeddingTensor> Embeddings<T> {
    pub fn row(&self, token: u32) -> Result<Vec<f32>> {
        match self {
            Self::Coded(embedding) => embedding.row(token as usize),
            Self::Plain {
                values,
                hidden_width,
            } => {
                let start = (token as usize)
                    .checked_mul(*hidden_width)
                    .ok_or_else(|| Error::InvalidInput("embedding token offset overflow".into()))?;
                let end = start
                    .checked_add(*hidden_width)
                    .ok_or_else(|| Error::InvalidInput("embedding token extent overflow".into()))?;
                Ok(values
                    .get(start..end)
                    .ok_or_else(|| {
                        Error::InvalidInput("embedding token is outside the vocabulary".into())
                    })?
                    .to_vec())
            }
        }
    }
}

pub struct AttentionLayer<L = CompressedLinear> {
    pub query_gate: L,
    pub key: L,
    pub value: L,
    pub output: L,
    pub attention: Attention,
}

pub struct RecurrentLayer<L = CompressedLinear> {
    pub query_key_value: L,
    pub output_gate: L,
    pub output: L,
    pub recurrence: GatedDeltaNet,
}

pub enum Branch<L = CompressedLinear> {
    Attention(Box<AttentionLayer<L>>),
    Recurrent(Box<RecurrentLayer<L>>),
}

pub struct Layer<L = CompressedLinear> {
    pub branch: Branch<L>,
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub gate: L,
    pub up: L,
    pub down: L,
}

#[derive(Clone)]
pub enum LayerState {
    Attention(AttentionState),
    Recurrent(GatedDeltaNetState),
}

impl LayerState {
    /// Add all persistent state bytes to a backend-independent FNV-1a digest.
    pub fn update_digest(&self, digest: &mut u64) {
        let tag = match self {
            Self::Attention(_) => 0,
            Self::Recurrent(_) => 1,
        };
        digest_bytes(digest, &[tag]);
        match self {
            Self::Attention(state) => state.update_digest(digest),
            Self::Recurrent(state) => state.update_digest(digest),
        }
    }
}

fn digest_bytes(digest: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *digest = (*digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
}

impl<L> Layer<L> {
    pub fn new_state(&self) -> Result<LayerState> {
        match &self.branch {
            Branch::Attention(layer) => Ok(LayerState::Attention(layer.attention.new_state()?)),
            Branch::Recurrent(layer) => Ok(LayerState::Recurrent(layer.recurrence.new_state()?)),
        }
    }
}

impl Layer {
    fn forward(
        &self,
        hidden: &[f32],
        state: &mut LayerState,
        position: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        let input_signs = match &self.branch {
            Branch::Attention(layer) => layer.query_gate.input_signs(),
            Branch::Recurrent(layer) => layer.query_key_value.input_signs(),
        };
        let input = prepare_normalization(hidden, &self.input_norm, input_signs, epsilon)?;
        let projected = match (&self.branch, state) {
            (Branch::Recurrent(layer), LayerState::Recurrent(state)) => {
                // qkv and z share qkv's prepared signs, even when their metadata differs.
                let query_key_value = layer
                    .query_key_value
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let output_gate = layer
                    .output_gate
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let gated = layer.recurrence.step(
                    state,
                    &query_key_value,
                    &output_gate,
                    &input.weighted_hidden_bf16,
                    input.inverse_rms,
                )?;
                let prepared = quantize_activation(&gated, layer.output.input_signs())?;
                layer.output.project(&prepared, None)?
            }
            (Branch::Attention(layer), LayerState::Attention(state)) => {
                // All three projections reuse q's input signs and the same integer scales.
                let query_gate = layer
                    .query_gate
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let key = layer
                    .key
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let value = layer
                    .value
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let gated = layer
                    .attention
                    .step(state, position, &query_gate, &key, &value)?;
                let prepared = quantize_activation(&gated, layer.output.input_signs())?;
                layer.output.project(&prepared, None)?
            }
            _ => {
                return Err(Error::InvalidModel(
                    "layer and persistent state types differ".into(),
                ));
            }
        };
        let hidden = residual_add(hidden, &projected)?;
        let input = prepare_normalization(
            &hidden,
            &self.post_attention_norm,
            self.gate.input_signs(),
            epsilon,
        )?;
        let gate = self
            .gate
            .project(&input.prepared, Some(input.inverse_rms))?;
        let up = self.up.project(&input.prepared, Some(input.inverse_rms))?;
        let product = gated_product(&gate, &up)?;
        let prepared = quantize_activation(&product, self.down.input_signs())?;
        residual_add(&hidden, &self.down.project(&prepared, None)?)
    }
}

pub struct Weights<L = CompressedLinear, T = yskn_format::Tensor> {
    pub dimensions: ModelDimensions,
    pub embeddings: Embeddings<T>,
    pub layers: Vec<Layer<L>>,
    pub final_norm: Vec<f32>,
    pub head: L,
}

/// Optional post-MLP residual taps. Empty layers disable observation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureOptions {
    pub layers: Vec<usize>,
    /// Maximum committed rows retained, normally the draft ring capacity.
    pub max_rows: usize,
}
impl CaptureOptions {
    pub fn validate(&self, layer_count: usize) -> Result<()> {
        if !self.layers.is_empty()
            && (self.max_rows == 0
                || self.layers.iter().any(|&layer| layer >= layer_count)
                || self.layers.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(Error::InvalidInput(
                "invalid capture layers or row capacity".into(),
            ));
        }
        Ok(())
    }
}

/// One absolute input position, in [capture_slot, hidden_channel] order.
#[derive(Clone, Debug, PartialEq)]
pub struct CapturedRow {
    pub position: usize,
    pub values: Vec<f32>,
}

/// Allocated capture storage, including row descriptors and FP32 payloads.
pub fn capture_allocated_bytes(rows: &[CapturedRow], capacity: usize) -> usize {
    capacity * std::mem::size_of::<CapturedRow>()
        + rows
            .iter()
            .map(|row| row.values.capacity() * 4)
            .sum::<usize>()
}

/// An owned committed checkpoint. Attention rows stay in the originating session.
///
/// A checkpoint can only restore its own session, before reset, while its exact
/// token prefix is still present. It contains no attention cache copies.
pub struct Snapshot {
    session: Rc<()>,
    position: usize,
    tokens: Vec<u32>,
    recurrent_states: Vec<Option<GatedDeltaNetState>>,
    committed_logits: Option<Vec<f32>>,
    capture_options: CaptureOptions,
    captures: Vec<CapturedRow>,
}

impl Snapshot {
    pub fn capture_bytes(&self) -> usize {
        capture_allocated_bytes(&self.captures, self.captures.capacity())
    }
}

/// A single sequence. Immutable weights are shared only through a single-threaded Rc.
pub struct Model {
    weights: Rc<Weights>,
    layer_states: Vec<LayerState>,
    position: usize,
    context_capacity: usize,
    committed_logits: Option<Vec<f32>>,
    last_pass_logits: Vec<Vec<f32>>,
    pending_tokens: Option<Vec<u32>>,
    committed_tokens: Vec<u32>,
    session: Rc<()>,
    context_identity: Arc<()>,
    capture_options: CaptureOptions,
    captures: Vec<CapturedRow>,
    provisional_captures: Vec<CapturedRow>,
}

type ComputedRows = (Vec<LayerState>, Vec<Vec<f32>>, Vec<CapturedRow>);

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_container(Container::open(path)?)
    }

    /// Read only the main weights from chapter 10's optional package manifest.
    ///
    /// `folder` may also be a plain `.yskn` weights path (any non-directory):
    /// that file is loaded directly with the default context capacity of 262,144.
    pub fn load_folder(folder: impl AsRef<Path>) -> Result<Self> {
        Self::load_folder_with(folder, KvMode::Int8)
    }

    /// Profile modes require files.kv_profile (default kvq_u5.bin) inside the folder.
    pub fn load_folder_with(folder: impl AsRef<Path>, mode: KvMode) -> Result<Self> {
        let (path, capacity, profile) = model_location_with_kv(folder.as_ref(), mode)?;
        let mut model = Self::from_container_with_kv(Container::open(path)?, mode, profile)?;
        model.context_capacity = capacity;
        Ok(model)
    }

    pub fn from_container<R: Read + Seek>(container: Container<R>) -> Result<Self> {
        Self::from_container_with_kv(container, KvMode::Int8, None)
    }

    pub fn from_container_with_kv<R: Read + Seek>(
        mut container: Container<R>,
        mode: KvMode,
        profile: Option<Arc<KvProfile>>,
    ) -> Result<Self> {
        let Weights {
            dimensions,
            embeddings,
            layers,
            final_norm,
            head,
        } = load_weights_with_kv(&mut container, mode, profile, load_linear, |c, name| {
            Ok(c.load_tensor(name)?)
        })?;
        let layer_states = layers.iter().map(Layer::new_state).collect::<Result<_>>()?;
        Ok(Self {
            weights: Rc::new(Weights {
                dimensions,
                embeddings,
                layers,
                final_norm,
                head,
            }),
            layer_states,
            position: 0,
            // This is a session limit from chapter 10, not a model tensor dimension.
            context_capacity: 262_144,
            committed_logits: None,
            last_pass_logits: Vec::new(),
            pending_tokens: None,
            committed_tokens: Vec::new(),
            session: Rc::new(()),
            context_identity: Arc::new(()),
            capture_options: CaptureOptions::default(),
            captures: Vec::new(),
            provisional_captures: Vec::new(),
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
    pub fn logits(&self) -> Option<&[f32]> {
        self.last_pass_logits.last().map(Vec::as_slice)
    }
    pub fn row_logits(&self) -> &[Vec<f32>] {
        &self.last_pass_logits
    }
    pub fn has_provisional_pass(&self) -> bool {
        self.pending_tokens.is_some()
    }

    /// Configure before consuming any rows; changing taps mid-prefix loses context.
    pub fn set_capture(&mut self, options: CaptureOptions) -> Result<()> {
        self.require_no_pending_pass()?;
        options.validate(self.dimensions().layer_count)?;
        if self.position != 0 {
            return Err(Error::InvalidInput(
                "configure capture before prefill".into(),
            ));
        }
        self.capture_options = options;
        self.captures.clear();
        Ok(())
    }
    pub fn capture_options(&self) -> &CaptureOptions {
        &self.capture_options
    }
    pub fn context_identity(&self) -> Arc<()> {
        Arc::clone(&self.context_identity)
    }
    /// Move undelivered rows to the consumer; snapshots now omit delivered rows.
    pub fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        std::mem::take(&mut self.captures)
    }
    pub fn capture_bytes(&self) -> usize {
        capture_allocated_bytes(&self.captures, self.captures.capacity())
            + capture_allocated_bytes(
                &self.provisional_captures,
                self.provisional_captures.capacity(),
            )
    }
    pub fn captured_rows(&self) -> &[CapturedRow] {
        &self.captures
    }
    pub fn provisional_captures(&self) -> &[CapturedRow] {
        &self.provisional_captures
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

    pub fn new_session(&self) -> Result<Self> {
        Ok(Self {
            weights: Rc::clone(&self.weights),
            layer_states: self
                .weights
                .layers
                .iter()
                .map(Layer::new_state)
                .collect::<Result<_>>()?,
            position: 0,
            context_capacity: self.context_capacity,
            committed_logits: None,
            last_pass_logits: Vec::new(),
            pending_tokens: None,
            committed_tokens: Vec::new(),
            session: Rc::new(()),
            context_identity: Arc::new(()),
            capture_options: self.capture_options.clone(),
            captures: Vec::new(),
            provisional_captures: Vec::new(),
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        let states = self
            .weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        self.layer_states = states;
        self.position = 0;
        self.committed_logits = None;
        self.last_pass_logits.clear();
        self.pending_tokens = None;
        self.committed_tokens.clear();
        self.captures.clear();
        self.provisional_captures.clear();
        self.session = Rc::new(());
        self.context_identity = Arc::new(());
        Ok(())
    }

    /// Consume all tokens atomically. Every row's logits remain available until the next pass.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.require_no_pending_pass()?;
        if tokens.is_empty() {
            return Ok(());
        }
        let (states, logits, captures) =
            self.compute_rows(tokens, self.capture_options.max_rows)?;
        self.layer_states = states;
        self.position += tokens.len();
        self.committed_tokens.extend_from_slice(tokens);
        self.committed_logits = logits.last().cloned();
        self.last_pass_logits = logits;
        self.retain_captures(captures);
        Ok(())
    }

    /// Compute a straight causal candidate chain without changing committed state or position.
    pub fn run_provisional(&mut self, tokens: &[u32]) -> Result<&[Vec<f32>]> {
        self.require_no_pending_pass()?;
        if tokens.is_empty() {
            return Err(Error::InvalidInput(
                "a provisional pass must contain at least one row".into(),
            ));
        }
        let (_, logits, captures) = self.compute_rows(tokens, tokens.len())?;
        self.pending_tokens = Some(tokens.to_vec());
        self.provisional_captures = captures;
        self.last_pass_logits = logits;
        Ok(&self.last_pass_logits)
    }

    /// Retain exactly `accepted_rows` input rows, including row zero. No implicit +1.
    pub fn commit(&mut self, accepted_rows: usize) -> Result<()> {
        let tokens = self
            .pending_tokens
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("there is no provisional pass to commit".into()))?;
        if accepted_rows > tokens.len() {
            return Err(Error::InvalidInput(
                "accepted rows exceed the provisional pass".into(),
            ));
        }
        if accepted_rows == 0 {
            self.discard_provisional();
            return Ok(());
        }
        // Replay from the still-committed state. The exact same scalar row path
        // makes commit independent of candidate count and avoids M state snapshots.
        let (states, logits, captures) =
            self.compute_rows(&tokens[..accepted_rows], self.capture_options.max_rows)?;
        self.layer_states = states;
        self.position += accepted_rows;
        self.committed_tokens
            .extend_from_slice(&tokens[..accepted_rows]);
        self.committed_logits = logits.last().cloned();
        self.last_pass_logits = logits;
        self.pending_tokens = None;
        self.provisional_captures.clear();
        self.retain_captures(captures);
        Ok(())
    }

    pub fn discard_provisional(&mut self) {
        if self.pending_tokens.take().is_some() {
            self.provisional_captures.clear();
            self.last_pass_logits = self.committed_logits.iter().cloned().collect();
        }
    }

    /// Capture a committed checkpoint without copying any attention cache rows.
    pub fn snapshot(&self) -> Result<Snapshot> {
        self.require_no_pending_pass()?;
        Ok(Snapshot {
            session: Rc::clone(&self.session),
            position: self.position,
            tokens: self.committed_tokens.clone(),
            recurrent_states: self
                .layer_states
                .iter()
                .map(|state| match state {
                    LayerState::Attention(_) => None,
                    LayerState::Recurrent(state) => Some(state.clone()),
                })
                .collect(),
            committed_logits: self.committed_logits.clone(),
            capture_options: self.capture_options.clone(),
            captures: self.captures.clone(),
        })
    }

    /// Return to a checkpoint on this session's retained causal prefix.
    /// Invalid checkpoints leave all observable state unchanged.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.require_no_pending_pass()?;
        if !Rc::ptr_eq(&self.session, &snapshot.session) {
            return Err(Error::InvalidInput(
                "snapshot belongs to a different session or reset".into(),
            ));
        }
        if snapshot.capture_options != self.capture_options {
            return Err(Error::InvalidInput(
                "snapshot capture configuration changed".into(),
            ));
        }
        if snapshot.position > self.position {
            return Err(Error::InvalidInput(
                "snapshot is beyond the current position".into(),
            ));
        }
        if self.committed_tokens[..snapshot.position] != snapshot.tokens {
            return Err(Error::InvalidInput(
                "snapshot token prefix is no longer retained".into(),
            ));
        }
        // Validate every cache before mutating any layer, so restore remains atomic.
        for state in &self.layer_states {
            if let LayerState::Attention(state) = state {
                if state.len() < snapshot.position {
                    return Err(Error::InvalidInput(
                        "snapshot attention prefix is unavailable".into(),
                    ));
                }
            }
        }
        for (state, recurrent) in self.layer_states.iter_mut().zip(&snapshot.recurrent_states) {
            match (state, recurrent) {
                (LayerState::Attention(state), None) => state.truncate(snapshot.position)?,
                (LayerState::Recurrent(state), Some(saved)) => state.copy_from(saved),
                _ => unreachable!("checkpoint layer kinds are fixed within a session"),
            }
        }
        self.context_identity = Arc::new(());
        self.captures.clone_from(&snapshot.captures);
        self.provisional_captures.clear();
        self.position = snapshot.position;
        self.committed_tokens.truncate(snapshot.position);
        self.committed_logits.clone_from(&snapshot.committed_logits);
        self.last_pass_logits.clear();
        self.last_pass_logits
            .extend(self.committed_logits.iter().cloned());
        Ok(())
    }

    /// Digest the committed numeric state, independent of provisional rows.
    pub fn state_digest(&self) -> u64 {
        let mut digest = 0xcbf29ce484222325;
        digest_bytes(&mut digest, &(self.position as u64).to_le_bytes());
        for state in &self.layer_states {
            state.update_digest(&mut digest);
        }
        if let Some(logits) = &self.committed_logits {
            for value in logits {
                digest_bytes(&mut digest, &value.to_bits().to_le_bytes());
            }
        }
        digest
    }

    fn require_no_pending_pass(&self) -> Result<()> {
        if self.pending_tokens.is_some() {
            Err(Error::InvalidInput(
                "commit or discard the current provisional pass first".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn compute_rows(&self, tokens: &[u32], capture_limit: usize) -> Result<ComputedRows> {
        let end_position = self
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| Error::InvalidInput("position overflow".into()))?;
        if end_position > self.context_capacity {
            return Err(Error::InvalidInput(
                "pass exceeds the session context capacity".into(),
            ));
        }
        if tokens
            .iter()
            .any(|&token| token as usize >= self.weights.dimensions.vocabulary_size)
        {
            return Err(Error::InvalidInput("token ID exceeds vocabulary".into()));
        }
        let mut states = self.layer_states.clone();
        let mut logits = Vec::with_capacity(tokens.len());
        let mut captures = Vec::new();
        let capture_start = tokens.len().saturating_sub(capture_limit);
        // A canonical row traversal defines batch/chunk invariance. Kernel dispatch
        // and upstream one-row precommit shortcuts are not part of this reference.
        for (row, &token) in tokens.iter().enumerate() {
            let mut hidden = self.weights.embeddings.row(token)?;
            let mut values = Vec::new();
            for (index, (layer, state)) in self.weights.layers.iter().zip(&mut states).enumerate() {
                hidden = layer.forward(
                    &hidden,
                    state,
                    self.position + row,
                    self.weights.dimensions.epsilon,
                )?;
                if row >= capture_start && self.capture_options.layers.contains(&index) {
                    values.extend_from_slice(&hidden);
                }
            }
            if !values.is_empty() {
                captures.push(CapturedRow {
                    position: self.position + row,
                    values,
                });
            }
            let input = prepare_normalization(
                &hidden,
                &self.weights.final_norm,
                self.weights.head.input_signs(),
                self.weights.dimensions.epsilon,
            )?;
            logits.push(output_head(&self.weights.head, &input)?);
        }
        Ok((states, logits, captures))
    }
}

/// Shared architecture validation/loading; backends choose storage for each tensor.
pub fn load_weights<R: Read + Seek, L, T: EmbeddingTensor>(
    container: &mut Container<R>,
    linear: impl FnMut(&mut Container<R>, &str, usize, usize, bool) -> Result<L>,
    tensor: impl FnMut(&mut Container<R>, &str) -> Result<T>,
) -> Result<Weights<L, T>> {
    load_weights_with_kv(container, KvMode::Int8, None, linear, tensor)
}

/// Backend-independent loader; the profile is shared by immutable attention layers.
pub fn load_weights_with_kv<R: Read + Seek, L, T: EmbeddingTensor>(
    container: &mut Container<R>,
    mode: KvMode,
    profile: Option<Arc<KvProfile>>,
    mut linear: impl FnMut(&mut Container<R>, &str, usize, usize, bool) -> Result<L>,
    mut tensor: impl FnMut(&mut Container<R>, &str) -> Result<T>,
) -> Result<Weights<L, T>> {
    let header = container.header().model.clone();
    let dimensions = ModelDimensions::read(&header)?;
    let rotary_dim = rotary_width_from_header(&header)?;
    let layer_types = header
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::InvalidModel("model.layer_types must be an array".into()))?;
    if layer_types.len() != dimensions.layer_count
        || layer_types.iter().any(|value| !value.is_string())
    {
        return Err(Error::InvalidModel(
            "layer_types must contain one string per layer".into(),
        ));
    }
    if mode != KvMode::Int8 {
        profile
            .as_ref()
            .ok_or_else(|| Error::InvalidModel("profile KV mode requires a KVQP profile".into()))?
            .check_dimensions(
                layer_types
                    .iter()
                    .filter(|kind| kind.as_str() == Some("full_attention"))
                    .count(),
                dimensions.kv_heads,
                dimensions.attention_head_width,
            )?;
    }
    let embeddings = if container
        .header()
        .tensors
        .iter()
        .any(|tensor| tensor.name == "embed_tokens")
    {
        let main = tensor(container, "embed_tokens")?;
        if main.rows() != dimensions.vocabulary_size || main.cols() != dimensions.hidden_width {
            return Err(Error::InvalidModel(
                "embedding dimensions do not match model header".into(),
            ));
        }
        let residual = optional_tensor(container, "embed_tokens.res", &mut tensor)?;
        let residual2 = optional_tensor(container, "embed_tokens.res2", &mut tensor)?;
        Embeddings::Coded(Box::new(Embedding::new(main, residual, residual2)?))
    } else {
        let count = product(dimensions.vocabulary_size, dimensions.hidden_width)?;
        Embeddings::Plain {
            values: load_vector(container, "embed_tokens", count)?,
            hidden_width: dimensions.hidden_width,
        }
    };

    let mut layers = Vec::new();
    let mut attention_ordinal = 0;
    for (index, kind) in layer_types.iter().enumerate() {
        let prefix = format!("layers.{index}");
        let branch = if kind.as_str() == Some("full_attention") {
            let query_width = product(dimensions.query_heads, dimensions.attention_head_width)?;
            let key_width = product(dimensions.kv_heads, dimensions.attention_head_width)?;
            let attention = Attention::new(
                AttentionDimensions {
                    query_heads: dimensions.query_heads,
                    kv_heads: dimensions.kv_heads,
                    head_dim: dimensions.attention_head_width,
                    rotary_dim,
                },
                load_vector(
                    container,
                    &format!("{prefix}.self_attn.q_norm"),
                    dimensions.attention_head_width,
                )?,
                load_vector(
                    container,
                    &format!("{prefix}.self_attn.k_norm"),
                    dimensions.attention_head_width,
                )?,
                dimensions.epsilon,
            )?
            .with_kv(mode, profile.clone(), attention_ordinal)?;
            attention_ordinal += 1;
            Branch::Attention(Box::new(AttentionLayer {
                query_gate: linear(
                    container,
                    &format!("{prefix}.self_attn.q_proj"),
                    product(2, query_width)?,
                    dimensions.hidden_width,
                    true,
                )?,
                key: linear(
                    container,
                    &format!("{prefix}.self_attn.k_proj"),
                    key_width,
                    dimensions.hidden_width,
                    true,
                )?,
                value: linear(
                    container,
                    &format!("{prefix}.self_attn.v_proj"),
                    key_width,
                    dimensions.hidden_width,
                    true,
                )?,
                output: linear(
                    container,
                    &format!("{prefix}.self_attn.o_proj"),
                    dimensions.hidden_width,
                    query_width,
                    true,
                )?,
                attention,
            }))
        } else {
            // The upstream dispatch treats every other layer-type string as GDN.
            Branch::Recurrent(Box::new(load_recurrent_layer(
                container,
                &prefix,
                &dimensions,
                &mut linear,
            )?))
        };
        layers.push(Layer {
            branch,
            input_norm: load_vector(
                container,
                &format!("{prefix}.input_layernorm"),
                dimensions.hidden_width,
            )?,
            post_attention_norm: load_vector(
                container,
                &format!("{prefix}.post_attention_layernorm"),
                dimensions.hidden_width,
            )?,
            gate: linear(
                container,
                &format!("{prefix}.mlp.gate_proj"),
                dimensions.feed_forward_width,
                dimensions.hidden_width,
                true,
            )?,
            up: linear(
                container,
                &format!("{prefix}.mlp.up_proj"),
                dimensions.feed_forward_width,
                dimensions.hidden_width,
                true,
            )?,
            down: linear(
                container,
                &format!("{prefix}.mlp.down_proj"),
                dimensions.hidden_width,
                dimensions.feed_forward_width,
                true,
            )?,
        });
    }
    let final_norm = load_vector(container, "norm", dimensions.hidden_width)?;
    let head = linear(
        container,
        "lm_head",
        dimensions.vocabulary_size,
        dimensions.hidden_width,
        false,
    )?;
    Ok(Weights {
        dimensions,
        embeddings,
        layers,
        final_norm,
        head,
    })
}

fn load_recurrent_layer<R: Read + Seek, L>(
    container: &mut Container<R>,
    prefix: &str,
    model: &ModelDimensions,
    linear: &mut impl FnMut(&mut Container<R>, &str, usize, usize, bool) -> Result<L>,
) -> Result<RecurrentLayer<L>> {
    let name = |suffix: &str| format!("{prefix}.linear_attn.{suffix}");
    let head_dim = vector_length(container, &name("norm"))?;
    let value_heads = vector_length(container, &name("A_log"))?;
    let qkv_rows = container
        .header()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name("in_proj_qkv"))
        .ok_or_else(|| Error::InvalidModel(format!("missing {}", name("in_proj_qkv"))))?
        .rows;
    let output_width = product(value_heads, head_dim)?;
    let query_key_width = qkv_rows.checked_sub(output_width).ok_or_else(|| {
        Error::InvalidModel("GDN qkv rows are smaller than its value width".into())
    })?;
    let two_heads = product(2, head_dim)?;
    if query_key_width % two_heads != 0 || query_key_width == 0 {
        return Err(Error::InvalidModel(
            "GDN qkv shape does not give integral query/key heads".into(),
        ));
    }
    let convolution_elements = vector_length(container, &name("conv1d"))?;
    if convolution_elements % qkv_rows != 0 {
        return Err(Error::InvalidModel(
            "GDN convolution shape does not give integral taps".into(),
        ));
    }
    // Chapter 03's shape equations determine these dimensions from header descriptors.
    let dimensions = GatedDeltaNetDimensions {
        hidden_width: model.hidden_width,
        query_key_heads: query_key_width / two_heads,
        value_heads,
        head_dim,
        convolution_width: convolution_elements / qkv_rows,
    };
    let dense_elements = product(value_heads, model.hidden_width)?;
    let recurrence = GatedDeltaNet::new(
        dimensions,
        GatedDeltaNetWeights {
            projection_a: load_vector(container, &name("in_proj_a"), dense_elements)?,
            projection_b: load_vector(container, &name("in_proj_b"), dense_elements)?,
            convolution_weight: load_vector(container, &name("conv1d"), convolution_elements)?,
            log_decay_weight: load_vector(container, &name("A_log"), value_heads)?,
            time_bias: load_vector(container, &name("dt_bias"), value_heads)?,
            normalization_weight: load_vector(container, &name("norm"), head_dim)?,
        },
        model.epsilon,
    )?;
    Ok(RecurrentLayer {
        query_key_value: linear(
            container,
            &name("in_proj_qkv"),
            qkv_rows,
            model.hidden_width,
            true,
        )?,
        output_gate: linear(
            container,
            &name("in_proj_z"),
            output_width,
            model.hidden_width,
            true,
        )?,
        output: linear(
            container,
            &name("out_proj"),
            model.hidden_width,
            output_width,
            true,
        )?,
        recurrence,
    })
}

fn load_linear<R: Read + Seek>(
    container: &mut Container<R>,
    name: &str,
    rows: usize,
    cols: usize,
    output_rotation: bool,
) -> Result<CompressedLinear> {
    let tensor = container.load_tensor(name)?;
    if tensor.rows() != rows || tensor.cols() != cols {
        return Err(Error::InvalidModel(format!(
            "{name} must have shape [{rows}, {cols}]"
        )));
    }
    CompressedLinear::new(tensor, output_rotation)
}

fn optional_tensor<R: Read + Seek, T>(
    container: &mut Container<R>,
    name: &str,
    tensor: &mut impl FnMut(&mut Container<R>, &str) -> Result<T>,
) -> Result<Option<T>> {
    if container
        .header()
        .tensors
        .iter()
        .any(|tensor| tensor.name == name)
    {
        Ok(Some(tensor(container, name)?))
    } else {
        Ok(None)
    }
}

fn vector_length<R: Read + Seek>(container: &Container<R>, name: &str) -> Result<usize> {
    container
        .header()
        .vectors
        .iter()
        .find(|vector| vector.name == name)
        .map(|vector| vector.len)
        .filter(|&len| len > 0)
        .ok_or_else(|| Error::InvalidModel(format!("missing or empty vector {name}")))
}

fn load_vector<R: Read + Seek>(
    container: &mut Container<R>,
    name: &str,
    length: usize,
) -> Result<Vec<f32>> {
    let descriptor = container
        .header()
        .vectors
        .iter()
        .find(|vector| vector.name == name)
        .ok_or_else(|| Error::InvalidModel(format!("missing vector {name}")))?;
    if descriptor.dtype != VectorType::Bf16 || descriptor.len != length {
        return Err(Error::InvalidModel(format!(
            "{name} must contain {length} BF16 values"
        )));
    }
    let vector = container.load_vector(name)?;
    require_finite(&vector.values, "model vector")?;
    Ok(vector.values)
}

fn positive_dimension(model: &Value, name: &str) -> Result<usize> {
    model
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| {
            Error::InvalidModel(format!(
                "model.{name} must be a positive addressable integer"
            ))
        })
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::InvalidModel("model dimension product overflow".into()))
}

fn rotary_width_from_header(header: &Value) -> Result<usize> {
    if header.get("arch").and_then(Value::as_str) != Some("qwen3_5") {
        return Err(Error::InvalidModel(
            "main reference requires arch=qwen3_5".into(),
        ));
    }
    // D8 is an explicit exception to D5's header-derived dimensions.
    if header.get("head_dim").and_then(Value::as_u64) != Some(256) {
        return Err(Error::InvalidModel(
            "qwen3_5 requires model.head_dim=256 (D8)".into(),
        ));
    }
    Ok(64)
}

/// Resolve a folder manifest (or a direct weights path) without opening model weights.
pub fn model_location(folder: &Path) -> Result<(PathBuf, usize)> {
    let (path, capacity, _) = model_location_with_kv(folder, KvMode::Int8)?;
    Ok((path, capacity))
}

/// Resolve weights/capacity and load only the optional profile, never weight payloads.
/// Mapped backends can reuse this with load_weights_with_kv.
pub fn model_location_with_kv(
    folder: &Path,
    mode: KvMode,
) -> Result<(PathBuf, usize, Option<Arc<KvProfile>>)> {
    if !folder.is_dir() {
        if mode != KvMode::Int8 {
            return Err(Error::InvalidModel(
                "profile KV mode requires a model folder or an explicit profile".into(),
            ));
        }
        return Ok((folder.to_path_buf(), 262_144, None));
    }
    let manifest_path = folder.join("yasakani.json");
    let manifest: Value = match fs::read(manifest_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Value::Null,
        Err(error) => return Err(error.into()),
    };
    let weights = manifest_file(folder, &manifest, "weights", "qwen38_27b_e3x.yskn")?;
    let mut context_capacity = 262_144;
    if let Some(capacity) = manifest.get("max_ctx") {
        let capacity = capacity
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                Error::InvalidModel("max_ctx must be a positive addressable integer".into())
            })?;
        context_capacity = capacity
            .checked_add(63)
            .map(|rounded| rounded / 64 * 64)
            .ok_or_else(|| Error::InvalidModel("rounded context capacity overflows".into()))?;
    }
    let profile = if mode == KvMode::Int8 {
        None
    } else {
        let path = manifest_file(folder, &manifest, "kv_profile", "kvq_u5.bin")?;
        Some(Arc::new(KvProfile::parse(&fs::read(path)?)?))
    };
    Ok((weights, context_capacity, profile))
}

fn manifest_file(folder: &Path, manifest: &Value, key: &str, default: &str) -> Result<PathBuf> {
    let filename = match manifest.get("files").and_then(|files| files.get(key)) {
        Some(Value::String(filename)) => filename.as_str(),
        None => default,
        _ => return Err(Error::InvalidModel(format!("files.{key} must be a string"))),
    };
    if filename.is_empty()
        || Path::new(filename)
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return Err(Error::InvalidModel(format!(
            "files.{key} must be relative to the model folder"
        )));
    }
    let path = folder.join(filename);
    // Apply the same containment rule to both files, including existing symlink targets.
    if path.exists() && !fs::canonicalize(&path)?.starts_with(fs::canonicalize(folder)?) {
        return Err(Error::InvalidModel(format!(
            "files.{key} must stay inside the model folder"
        )));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rotary_width_checks_architecture_and_head_width_itself() {
        assert_eq!(
            rotary_width_from_header(&json!({"arch": "qwen3_5", "head_dim": 256})).unwrap(),
            64
        );
        for (header, expected) in [
            (
                json!({"arch": "other", "head_dim": 256}),
                "main reference requires arch=qwen3_5",
            ),
            (
                json!({"arch": "qwen3_5", "head_dim": 128}),
                "qwen3_5 requires model.head_dim=256 (D8)",
            ),
        ] {
            assert!(
                matches!(rotary_width_from_header(&header), Err(Error::InvalidModel(message)) if message == expected)
            );
        }
    }
}

#[cfg(test)]
mod snapshot_capacity_tests {
    use super::*;

    #[test]
    fn repeated_owned_checkpoints_do_not_grow_session_buffers() {
        let mut model = Model::from_container(
            Container::from_bytes(crate::test_support::synthetic_model(
                139,
                &["linear_attention", "full_attention"],
            ))
            .unwrap(),
        )
        .unwrap();
        model.prefill(&[1, 2]).unwrap();
        let capacities = |model: &Model| {
            (
                model.layer_states.capacity(),
                model.committed_tokens.capacity(),
                model.committed_logits.as_ref().unwrap().capacity(),
                model.last_pass_logits.capacity(),
                model
                    .layer_states
                    .iter()
                    .map(|state| match state {
                        LayerState::Attention(state) => state.capacity(),
                        LayerState::Recurrent(state) => state.allocated_bytes(),
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let expected = capacities(&model);
        let saved = model.snapshot().unwrap();
        let snapshot_capacities = (
            saved.tokens.capacity(),
            saved.recurrent_states.capacity(),
            saved.committed_logits.as_ref().unwrap().capacity(),
        );
        for _ in 0..64 {
            let snapshot = model.snapshot().unwrap();
            assert_eq!(capacities(&model), expected);
            assert_eq!(
                (
                    snapshot.tokens.capacity(),
                    snapshot.recurrent_states.capacity(),
                    snapshot.committed_logits.as_ref().unwrap().capacity(),
                ),
                snapshot_capacities
            );
            model.restore(&saved).unwrap();
            assert_eq!(capacities(&model), expected);
        }
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    use crate::portable_math::bfloat;

    #[test]
    fn captured_two_layer_residual_is_independent_post_mlp_bf16_sum() {
        let mut bytes =
            crate::test_support::synthetic_model(317, &["full_attention", "full_attention"]);
        let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&bytes[32..32 + n]).unwrap();
        // Hand-set sizeable MLP scales so both residual boundaries are observable.
        for tensor in header["tensors"].as_array().unwrap() {
            if tensor["name"].as_str().unwrap().contains(".mlp.") {
                let start = tensor["scale_off"].as_u64().unwrap() as usize;
                let count = tensor["physical_rows"].as_u64().unwrap() as usize;
                for pair in bytes[start..start + 2 * count].chunks_exact_mut(2) {
                    pair.copy_from_slice(&yskn_format::f32_to_f16(0.25).to_le_bytes());
                }
            }
        }
        let mut model = Model::from_container(Container::from_bytes(bytes).unwrap()).unwrap();
        let weights = Rc::get_mut(&mut model.weights).unwrap();
        weights.embeddings = Embeddings::Plain {
            values: (0..128 * 512)
                .map(|i| bfloat((i % 71) as f32 / 32.0 - 1.0))
                .collect(),
            hidden_width: 512,
        };
        for layer in &mut weights.layers {
            layer.input_norm.fill(2.0);
            layer.post_attention_norm.fill(3.0);
        }
        model
            .set_capture(CaptureOptions {
                layers: vec![0, 1],
                max_rows: 4,
            })
            .unwrap();
        let mut hidden = model.embedding(3).unwrap();
        let mut expected = Vec::new();
        for layer in &model.weights.layers {
            let Branch::Attention(branch) = &layer.branch else {
                unreachable!()
            };
            let mut state = branch.attention.new_state().unwrap();
            let norm = prepare_normalization(
                &hidden,
                &layer.input_norm,
                branch.query_gate.input_signs(),
                model.dimensions().epsilon,
            )
            .unwrap();
            let q = branch
                .query_gate
                .project(&norm.prepared, Some(norm.inverse_rms))
                .unwrap();
            let k = branch
                .key
                .project(&norm.prepared, Some(norm.inverse_rms))
                .unwrap();
            let v = branch
                .value
                .project(&norm.prepared, Some(norm.inverse_rms))
                .unwrap();
            let output = branch.attention.step(&mut state, 0, &q, &k, &v).unwrap();
            let prepared = quantize_activation(&output, branch.output.input_signs()).unwrap();
            let projected = branch.output.project(&prepared, None).unwrap();
            let old: Vec<_> = hidden
                .iter()
                .zip(projected)
                .map(|(&h, p)| bfloat(h + bfloat(p)))
                .collect();
            let norm = prepare_normalization(
                &old,
                &layer.post_attention_norm,
                layer.gate.input_signs(),
                model.dimensions().epsilon,
            )
            .unwrap();
            let gate = layer
                .gate
                .project(&norm.prepared, Some(norm.inverse_rms))
                .unwrap();
            let up = layer
                .up
                .project(&norm.prepared, Some(norm.inverse_rms))
                .unwrap();
            let product = gated_product(&gate, &up).unwrap();
            let prepared = quantize_activation(&product, layer.down.input_signs()).unwrap();
            let down = layer.down.project(&prepared, None).unwrap();
            hidden = old
                .iter()
                .zip(down)
                .map(|(&h, p)| bfloat(h + bfloat(p)))
                .collect();
            assert_ne!(hidden, old, "fixture must detect a tap before the MLP add");
            assert_ne!(hidden, norm.weighted_hidden_bf16);
            expected.extend_from_slice(&hidden);
        }
        model.prefill(&[3]).unwrap();
        assert_eq!(model.captured_rows()[0].position, 0);
        assert_eq!(
            model.captured_rows()[0]
                .values
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_ne!(&expected[..512], &expected[512..]);
    }

    #[test]
    fn capture_chunks_commit_restore_and_bounded_prefill_observe_without_changing_logits() {
        let bytes =
            crate::test_support::synthetic_model(319, &["linear_attention", "full_attention"]);
        let build =
            || Model::from_container(Container::from_bytes(bytes.clone()).unwrap()).unwrap();
        let mut whole = build();
        let mut single = build();
        let mut uneven = build();
        let mut plain = build();
        for model in [&mut whole, &mut single, &mut uneven] {
            model
                .set_capture(CaptureOptions {
                    layers: vec![0, 1],
                    max_rows: 16,
                })
                .unwrap();
        }
        let tokens = [1, 4, 2, 9, 3, 8, 7, 6, 5];
        whole.prefill(&tokens).unwrap();
        plain.prefill(&tokens).unwrap();
        assert_eq!(whole.row_logits(), plain.row_logits());
        assert_eq!(whole.state_digest(), plain.state_digest());
        for token in tokens {
            single.prefill(&[token]).unwrap();
        }
        for chunk in [&tokens[..2], &tokens[2..7], &tokens[7..]] {
            uneven.prefill(chunk).unwrap();
        }
        assert_eq!(whole.captured_rows(), single.captured_rows());
        assert_eq!(whole.captured_rows(), uneven.captured_rows());
        let snapshot = whole.snapshot().unwrap();
        whole.run_provisional(&[10, 11, 12]).unwrap();
        let provisional = whole.provisional_captures().to_vec();
        assert_eq!(
            provisional.iter().map(|r| r.position).collect::<Vec<_>>(),
            [9, 10, 11]
        );
        whole.commit(2).unwrap();
        assert_eq!(&whole.captured_rows()[9..], &provisional[..2]);
        whole.restore(&snapshot).unwrap();
        assert_eq!(whole.captured_rows(), single.captured_rows());
        whole.run_provisional(&[11]).unwrap();
        whole.discard_provisional();
        assert!(whole.provisional_captures().is_empty());
        let mut bounded = build();
        bounded
            .set_capture(CaptureOptions {
                layers: vec![0, 1],
                max_rows: 3,
            })
            .unwrap();
        bounded.prefill(&tokens).unwrap();
        assert_eq!(bounded.captured_rows(), &single.captured_rows()[6..]);
    }
}
