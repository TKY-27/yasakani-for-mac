//! Layer-major inference over mapped code streams and the reference's scalar primitives.
use crate::{Engine, Error, MappedModel, Matrix, Result, mapped::MappedEmbeddingTensor};
use std::{
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};
use yskn_ref::{
    ModelDimensions,
    compressed_linear::{PreparedActivation, quantize_activation, restore_projection},
    gated_deltanet::{GatedDeltaNet, GatedDeltaNetState, round_projection},
    model::{Branch, Layer, LayerState, Weights, load_weights, model_location},
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
pub struct PassTiming {
    pub position: usize,
    pub rows: usize,
    pub total: Duration,
    pub kernel: Duration,
    pub attention: Duration,
    pub recurrence: Duration,
}
impl PassTiming {
    pub fn rest(&self) -> Duration {
        self.total
            .saturating_sub(self.kernel + self.attention + self.recurrence)
    }
}

pub struct Model {
    weights: Weights<Linear, MappedEmbeddingTensor>,
    engine: Engine,
    states: Vec<LayerState>,
    position: usize,
    context_capacity: usize,
    rows_per_pass: usize,
    logits: Vec<Vec<f32>>,
    timings: Vec<PassTiming>,
}
impl Model {
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        // SAFETY: forwarded file-stability contract is required of this method's caller.
        unsafe { Self::with_engine(path, Engine::from_env()?, 16) }
    }

    /// Use an explicit kernel/pool and pass width (positive, with no implicit upper clamp).
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_engine(
        path: impl AsRef<Path>,
        engine: Engine,
        rows_per_pass: usize,
    ) -> Result<Self> {
        if rows_per_pass == 0 {
            return Err(Error::InvalidInput("rows_per_pass must be positive".into()));
        }
        let (path, context_capacity) = model_location(path.as_ref())?;
        // SAFETY: caller establishes file stability through model drop. Every weight view
        // shares this mapping, so none can outlive that external-file contract.
        let mapping = unsafe { MappedModel::open(path)? };
        let weights = load_weights(
            &mut mapping.reader()?,
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
            logits.extend(project_normalized(
                &mut self.engine,
                &self.weights.head,
                &input,
                &mut timing,
            )?);
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

fn normalize(
    engine: &mut Engine,
    rows: &[Vec<f32>],
    norm: &[f32],
    linear: &Linear,
    epsilon: f32,
) -> Result<Vec<NormalizedInput>> {
    engine.map_range(0..rows.len(), |i| {
        prepare_normalization(&rows[i], norm, linear.input_signs(), epsilon)
    })
}
fn project_normalized(
    engine: &mut Engine,
    linear: &Linear,
    input: &[NormalizedInput],
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let input: Vec<_> = input
        .iter()
        .map(|x| (&x.prepared, Some(x.inverse_rms)))
        .collect();
    project(engine, linear, &input, timing)
}
fn project(
    engine: &mut Engine,
    linear: &Linear,
    inputs: &[(&PreparedActivation, Option<f32>)],
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let activation: Vec<_> = inputs
        .iter()
        .flat_map(|(p, _)| p.activation8().iter().copied())
        .collect();
    let scales: Vec<_> = inputs
        .iter()
        .flat_map(|(p, _)| p.integer_scale().iter().copied())
        .collect();
    let start = Instant::now();
    let sums = engine.weighted_sums(&linear.matrix, &activation, &scales)?;
    timing.kernel += start.elapsed();
    let n = linear.matrix.descriptor().physical_rows;
    engine.map_range(0..inputs.len(), |i| {
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
    })
}
fn project_rows(
    engine: &mut Engine,
    linear: &Linear,
    rows: &[Vec<f32>],
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let prepared = engine.map_range(0..rows.len(), |i| {
        quantize_activation(&rows[i], linear.input_signs())
    })?;
    let inputs: Vec<_> = prepared.iter().map(|p| (p, None)).collect();
    project(engine, linear, &inputs, timing)
}
/// The reference's `GatedDeltaNet::step` stages for a pass of rows: rows are
/// prepared in parallel (they read only earlier rows' rounded projections), then
/// each value head runs its rows in ascending order on one worker. On error the
/// state is partly updated; `prefill` discards its state copy.
fn recurrence(
    engine: &mut Engine,
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
    engine: &mut Engine,
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
            let qkv = project_normalized(engine, &l.query_key_value, &input, timing)?;
            let z = project_normalized(engine, &l.output_gate, &input, timing)?;
            let start = Instant::now();
            let gated = recurrence(engine, &l.recurrence, state, &qkv, &z, &input)?;
            timing.recurrence += start.elapsed();
            project_rows(engine, &l.output, &gated, timing)?
        }
        (Branch::Attention(l), LayerState::Attention(state)) => {
            let q = project_normalized(engine, &l.query_gate, &input, timing)?;
            let k = project_normalized(engine, &l.key, &input, timing)?;
            let v = project_normalized(engine, &l.value, &input, timing)?;
            let start = Instant::now();
            let prepared = engine.map_range(0..hidden.len(), |i| {
                l.attention.prepare_row(position + i, &q[i], &k[i], &v[i])
            })?;
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
    let gate = project_normalized(engine, &layer.gate, &input, timing)?;
    let up = project_normalized(engine, &layer.up, &input, timing)?;
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
