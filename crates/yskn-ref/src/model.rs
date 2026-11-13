//! [Chapter 01: main-model architecture](../../../docs/spec/01_MODEL_ARCHITECTURE.md).

use std::fs;
use std::io::{Read, Seek};
use std::path::{Component, Path};
use std::rc::Rc;

use serde_json::Value;
use yskn_format::{Container, VectorType};

use crate::attention::{Attention, AttentionDimensions, AttentionState};
use crate::compressed_linear::{CompressedLinear, Embedding, quantize_activation};
use crate::gated_deltanet::{
    GatedDeltaNet, GatedDeltaNetDimensions, GatedDeltaNetState, GatedDeltaNetWeights,
};
use crate::output_head_sampler::output_head;
use crate::residual_norm_mlp::{gated_product, prepare_normalization, residual_add};
use crate::{Error, Result, require_finite};

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

enum Embeddings {
    Coded(Box<Embedding>),
    Plain {
        values: Vec<f32>,
        hidden_width: usize,
    },
}

impl Embeddings {
    fn row(&self, token: u32) -> Result<Vec<f32>> {
        match self {
            Self::Coded(embedding) => embedding.row(token as usize),
            Self::Plain {
                values,
                hidden_width,
            } => {
                let start = token as usize * hidden_width;
                Ok(values[start..start + hidden_width].to_vec())
            }
        }
    }
}

struct AttentionLayer {
    query_gate: CompressedLinear,
    key: CompressedLinear,
    value: CompressedLinear,
    output: CompressedLinear,
    attention: Attention,
}

struct RecurrentLayer {
    query_key_value: CompressedLinear,
    output_gate: CompressedLinear,
    output: CompressedLinear,
    recurrence: GatedDeltaNet,
}

enum Branch {
    Attention(Box<AttentionLayer>),
    Recurrent(Box<RecurrentLayer>),
}

struct Layer {
    branch: Branch,
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    gate: CompressedLinear,
    up: CompressedLinear,
    down: CompressedLinear,
}

#[derive(Clone)]
enum LayerState {
    Attention(AttentionState),
    Recurrent(GatedDeltaNetState),
}

impl Layer {
    fn new_state(&self) -> Result<LayerState> {
        match &self.branch {
            Branch::Attention(layer) => Ok(LayerState::Attention(layer.attention.new_state()?)),
            Branch::Recurrent(layer) => Ok(LayerState::Recurrent(layer.recurrence.new_state()?)),
        }
    }

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

struct Weights {
    dimensions: ModelDimensions,
    embeddings: Embeddings,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    head: CompressedLinear,
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
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_container(Container::open(path)?)
    }

    /// Read only the main weights from chapter 10's optional package manifest.
    pub fn load_folder(folder: impl AsRef<Path>) -> Result<Self> {
        let folder = folder.as_ref();
        let manifest_path = folder.join("yasakani.json");
        let manifest: Value = match fs::read(manifest_path) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(error) => return Err(error.into()),
        };
        let filename = match manifest.pointer("/files/weights") {
            Some(Value::String(filename)) => filename.as_str(),
            None => "qwen38_27b_e3x.yskn",
            _ => return Err(Error::InvalidModel("files.weights must be a string".into())),
        };
        if Path::new(filename)
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
        {
            return Err(Error::InvalidModel(
                "files.weights must be relative to the model folder".into(),
            ));
        }
        let mut model = Self::load(folder.join(filename))?;
        if let Some(capacity) = manifest.get("max_ctx") {
            let capacity = capacity
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    Error::InvalidModel("max_ctx must be a positive addressable integer".into())
                })?;
            model.context_capacity = capacity
                .checked_add(63)
                .map(|rounded| rounded / 64 * 64)
                .ok_or_else(|| Error::InvalidModel("rounded context capacity overflows".into()))?;
        }
        Ok(model)
    }

    pub fn from_container<R: Read + Seek>(mut container: Container<R>) -> Result<Self> {
        let header = container.header().model.clone();
        let dimensions = ModelDimensions::read(&header)?;
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
        if layer_types
            .iter()
            .any(|kind| kind.as_str() == Some("full_attention"))
        {
            require_rotary_width_spec()?;
        }

        let embeddings = if container
            .header()
            .tensors
            .iter()
            .any(|tensor| tensor.name == "embed_tokens")
        {
            let main = container.load_tensor("embed_tokens")?;
            if main.rows() != dimensions.vocabulary_size || main.cols() != dimensions.hidden_width {
                return Err(Error::InvalidModel(
                    "embedding dimensions do not match model header".into(),
                ));
            }
            let residual = load_optional_tensor(&mut container, "embed_tokens.res")?;
            let residual2 = load_optional_tensor(&mut container, "embed_tokens.res2")?;
            Embeddings::Coded(Box::new(Embedding::new(main, residual, residual2)?))
        } else {
            let count = product(dimensions.vocabulary_size, dimensions.hidden_width)?;
            Embeddings::Plain {
                values: load_vector(&mut container, "embed_tokens", count)?,
                hidden_width: dimensions.hidden_width,
            }
        };

        let mut layers = Vec::new();
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
                        rotary_dim: rotary_width_from_header(&header),
                    },
                    load_vector(
                        &mut container,
                        &format!("{prefix}.self_attn.q_norm"),
                        dimensions.attention_head_width,
                    )?,
                    load_vector(
                        &mut container,
                        &format!("{prefix}.self_attn.k_norm"),
                        dimensions.attention_head_width,
                    )?,
                    dimensions.epsilon,
                )?;
                Branch::Attention(Box::new(AttentionLayer {
                    query_gate: load_linear(
                        &mut container,
                        &format!("{prefix}.self_attn.q_proj"),
                        product(2, query_width)?,
                        dimensions.hidden_width,
                        true,
                    )?,
                    key: load_linear(
                        &mut container,
                        &format!("{prefix}.self_attn.k_proj"),
                        key_width,
                        dimensions.hidden_width,
                        true,
                    )?,
                    value: load_linear(
                        &mut container,
                        &format!("{prefix}.self_attn.v_proj"),
                        key_width,
                        dimensions.hidden_width,
                        true,
                    )?,
                    output: load_linear(
                        &mut container,
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
                    &mut container,
                    &prefix,
                    &dimensions,
                )?))
            };
            layers.push(Layer {
                branch,
                input_norm: load_vector(
                    &mut container,
                    &format!("{prefix}.input_layernorm"),
                    dimensions.hidden_width,
                )?,
                post_attention_norm: load_vector(
                    &mut container,
                    &format!("{prefix}.post_attention_layernorm"),
                    dimensions.hidden_width,
                )?,
                gate: load_linear(
                    &mut container,
                    &format!("{prefix}.mlp.gate_proj"),
                    dimensions.feed_forward_width,
                    dimensions.hidden_width,
                    true,
                )?,
                up: load_linear(
                    &mut container,
                    &format!("{prefix}.mlp.up_proj"),
                    dimensions.feed_forward_width,
                    dimensions.hidden_width,
                    true,
                )?,
                down: load_linear(
                    &mut container,
                    &format!("{prefix}.mlp.down_proj"),
                    dimensions.hidden_width,
                    dimensions.feed_forward_width,
                    true,
                )?,
            });
        }
        let final_norm = load_vector(&mut container, "norm", dimensions.hidden_width)?;
        let head = load_linear(
            &mut container,
            "lm_head",
            dimensions.vocabulary_size,
            dimensions.hidden_width,
            false,
        )?;
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
        Ok(())
    }

    /// Consume all tokens atomically. Every row's logits remain available until the next pass.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.require_no_pending_pass()?;
        if tokens.is_empty() {
            return Ok(());
        }
        let (states, logits) = self.compute_rows(tokens)?;
        self.layer_states = states;
        self.position += tokens.len();
        self.committed_logits = logits.last().cloned();
        self.last_pass_logits = logits;
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
        let (_, logits) = self.compute_rows(tokens)?;
        self.pending_tokens = Some(tokens.to_vec());
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
        let (states, logits) = self.compute_rows(&tokens[..accepted_rows])?;
        self.layer_states = states;
        self.position += accepted_rows;
        self.committed_logits = logits.last().cloned();
        self.last_pass_logits = logits;
        self.pending_tokens = None;
        Ok(())
    }

    pub fn discard_provisional(&mut self) {
        if self.pending_tokens.take().is_some() {
            self.last_pass_logits = self.committed_logits.iter().cloned().collect();
        }
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

    fn compute_rows(&self, tokens: &[u32]) -> Result<(Vec<LayerState>, Vec<Vec<f32>>)> {
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
        // A canonical row traversal defines batch/chunk invariance. Kernel dispatch
        // and upstream one-row precommit shortcuts are not part of this reference.
        for (row, &token) in tokens.iter().enumerate() {
            let mut hidden = self.weights.embeddings.row(token)?;
            for (layer, state) in self.weights.layers.iter().zip(&mut states) {
                hidden = layer.forward(
                    &hidden,
                    state,
                    self.position + row,
                    self.weights.dimensions.epsilon,
                )?;
            }
            let input = prepare_normalization(
                &hidden,
                &self.weights.final_norm,
                self.weights.head.input_signs(),
                self.weights.dimensions.epsilon,
            )?;
            logits.push(output_head(&self.weights.head, &input)?);
        }
        Ok((states, logits))
    }
}

fn load_recurrent_layer<R: Read + Seek>(
    container: &mut Container<R>,
    prefix: &str,
    model: &ModelDimensions,
) -> Result<RecurrentLayer> {
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
        query_key_value: load_linear(
            container,
            &name("in_proj_qkv"),
            qkv_rows,
            model.hidden_width,
            true,
        )?,
        output_gate: load_linear(
            container,
            &name("in_proj_z"),
            output_width,
            model.hidden_width,
            true,
        )?,
        output: load_linear(
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

fn load_optional_tensor<R: Read + Seek>(
    container: &mut Container<R>,
    name: &str,
) -> Result<Option<yskn_format::Tensor>> {
    if container
        .header()
        .tensors
        .iter()
        .any(|tensor| tensor.name == name)
    {
        Ok(Some(container.load_tensor(name)?))
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

fn require_rotary_width_spec() -> Result<()> {
    Err(Error::SpecGap("ARCH-01"))
}

fn rotary_width_from_header(_header: &Value) -> usize {
    todo!(
        "docs/port/SPEC_GAPS.md#arch-01: the header field or derivation for rotary width is unspecified"
    )
}
