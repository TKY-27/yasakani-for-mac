//! Scalar recurrence from [chapter 03](../../../docs/spec/03_GATED_DELTANET.md).

use crate::portable_math::{bfloat, exp, rsqrt, sigmoid, silu, softplus};
use crate::{Error, Result};

/// Dimensions read from the model and its tensor descriptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedDeltaNetDimensions {
    pub hidden_width: usize,
    pub query_key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub convolution_width: usize,
}

/// Dense parameters; compressed projections belong to the enclosing model layer.
#[derive(Clone, Debug)]
pub struct GatedDeltaNetWeights {
    pub projection_a: Vec<f32>,
    pub projection_b: Vec<f32>,
    pub convolution_weight: Vec<f32>,
    pub log_decay_weight: Vec<f32>,
    pub time_bias: Vec<f32>,
    pub normalization_weight: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct GatedDeltaNet {
    dimensions: GatedDeltaNetDimensions,
    weights: GatedDeltaNetWeights,
    epsilon: f32,
    query_key_width: usize,
    output_width: usize,
    convolution_channels: usize,
    state_elements: usize,
}

/// FP32 recurrent matrices and BF16-rounded raw convolution history.
#[derive(Clone, Debug, PartialEq)]
pub struct GatedDeltaNetState {
    dimensions: GatedDeltaNetDimensions,
    recurrent_state: Vec<f32>,
    history: Vec<f32>,
}

impl GatedDeltaNetState {
    /// Copy into an already allocated state with the same dimensions.
    pub fn copy_from(&mut self, source: &Self) {
        assert_eq!(self.dimensions, source.dimensions);
        self.recurrent_state
            .copy_from_slice(&source.recurrent_state);
        self.history.copy_from_slice(&source.history);
    }

    /// Copy only history when an engine writes the recurrent matrix out of place.
    pub fn copy_history_from(&mut self, source: &Self) {
        assert_eq!(self.dimensions, source.dimensions);
        self.history.copy_from_slice(&source.history);
    }

    pub fn recurrent_heads(&self) -> std::slice::ChunksExact<'_, f32> {
        self.recurrent_state
            .chunks_exact(self.dimensions.head_dim * self.dimensions.head_dim)
    }

    pub fn allocated_bytes(&self) -> usize {
        (self.recurrent_state.capacity() + self.history.capacity()) * std::mem::size_of::<f32>()
    }

    pub fn update_digest(&self, digest: &mut u64) {
        for values in [&self.recurrent_state, &self.history] {
            for byte in (values.len() as u64).to_le_bytes().into_iter().chain(
                values
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes()),
            ) {
                *digest = (*digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
    }

    /// Rounded projections of the previous rows, oldest first.
    pub fn history_rows(&self) -> std::slice::ChunksExact<'_, f32> {
        let rows = self.dimensions.convolution_width - 1;
        self.history.chunks_exact(self.history.len() / rows)
    }

    /// Each value head's `head_dim × head_dim` recurrent block, in head order.
    pub fn recurrent_heads_mut(&mut self) -> std::slice::ChunksExactMut<'_, f32> {
        let head_dim = self.dimensions.head_dim;
        self.recurrent_state.chunks_exact_mut(head_dim * head_dim)
    }
}

/// One row after [`GatedDeltaNet::prepare`]: BF16 projections, convolution,
/// normalized query/key and dense gates. It does not depend on the recurrent state.
#[derive(Clone, Debug)]
pub struct GatedDeltaNetRow {
    projected: Vec<f32>,
    z: Vec<f32>,
    convolved: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    gates_a: Vec<f32>,
    gates_b: Vec<f32>,
}

impl GatedDeltaNetRow {
    /// Retain a prepared row in a fixed-width replay slot without allocating.
    pub fn copy_from(&mut self, source: &Self) {
        self.projected.copy_from_slice(&source.projected);
        self.z.copy_from_slice(&source.z);
        self.convolved.copy_from_slice(&source.convolved);
        self.query.copy_from_slice(&source.query);
        self.key.copy_from_slice(&source.key);
        self.gates_a.copy_from_slice(&source.gates_a);
        self.gates_b.copy_from_slice(&source.gates_b);
    }

    pub fn allocated_bytes(&self) -> usize {
        [
            &self.projected,
            &self.z,
            &self.convolved,
            &self.query,
            &self.key,
            &self.gates_a,
            &self.gates_b,
        ]
        .iter()
        .map(|v| v.capacity())
        .sum::<usize>()
            * std::mem::size_of::<f32>()
    }
}

/// The BF16 projection boundary; the result is also what the convolution history keeps.
pub fn round_projection(projected_qkv: &[f32]) -> Vec<f32> {
    projected_qkv.iter().map(|&value| bfloat(value)).collect()
}

fn dimension_mismatch() -> Error {
    Error::InvalidModel("DeltaNet row or state dimensions do not match the layer".into())
}

impl GatedDeltaNet {
    pub fn new(
        dimensions: GatedDeltaNetDimensions,
        weights: GatedDeltaNetWeights,
        epsilon: f32,
    ) -> Result<Self> {
        if dimensions.hidden_width == 0
            || dimensions.query_key_heads == 0
            || dimensions.value_heads == 0
            || dimensions.value_heads % dimensions.query_key_heads != 0
        {
            return Err(Error::InvalidModel(
                "DeltaNet requires positive widths and an integral value/key head ratio".into(),
            ));
        }
        if dimensions.head_dim != 128 || dimensions.convolution_width != 4 {
            return Err(Error::InvalidModel(
                "chapter 03 specifies only 128-channel heads and four convolution taps".into(),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidModel(
                "DeltaNet epsilon must be finite and positive".into(),
            ));
        }
        let product = |left: usize, right: usize| {
            left.checked_mul(right)
                .ok_or_else(|| Error::InvalidModel("DeltaNet dimension overflow".into()))
        };
        let query_key_width = product(dimensions.query_key_heads, dimensions.head_dim)?;
        let output_width = product(dimensions.value_heads, dimensions.head_dim)?;
        let convolution_channels = query_key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(output_width))
            .ok_or_else(|| Error::InvalidModel("DeltaNet channel count overflow".into()))?;
        let state_elements = product(output_width, dimensions.head_dim)?;
        let dense_elements = product(dimensions.value_heads, dimensions.hidden_width)?;
        let convolution_elements = product(convolution_channels, dimensions.convolution_width)?;
        for (name, values, expected) in [
            ("projection_a", &weights.projection_a, dense_elements),
            ("projection_b", &weights.projection_b, dense_elements),
            (
                "convolution_weight",
                &weights.convolution_weight,
                convolution_elements,
            ),
            (
                "log_decay_weight",
                &weights.log_decay_weight,
                dimensions.value_heads,
            ),
            ("time_bias", &weights.time_bias, dimensions.value_heads),
            (
                "normalization_weight",
                &weights.normalization_weight,
                dimensions.head_dim,
            ),
        ] {
            if values.len() != expected {
                return Err(Error::InvalidModel(format!(
                    "DeltaNet {name} has {} elements; expected {expected}",
                    values.len()
                )));
            }
            finite(values, "DeltaNet weights")?;
        }
        Ok(Self {
            dimensions,
            weights,
            epsilon,
            query_key_width,
            output_width,
            convolution_channels,
            state_elements,
        })
    }

    /// One preallocated slot for recurrence-only replay at commit.
    pub fn new_row(&self) -> Result<GatedDeltaNetRow> {
        Ok(GatedDeltaNetRow {
            projected: zeros(self.convolution_channels)?,
            z: zeros(self.output_width)?,
            convolved: zeros(self.convolution_channels)?,
            query: zeros(self.query_key_width)?,
            key: zeros(self.query_key_width)?,
            gates_a: zeros(self.dimensions.value_heads)?,
            gates_b: zeros(self.dimensions.value_heads)?,
        })
    }

    pub fn new_state(&self) -> Result<GatedDeltaNetState> {
        let history_elements = self
            .convolution_channels
            .checked_mul(self.dimensions.convolution_width - 1)
            .ok_or_else(|| Error::InvalidModel("DeltaNet history size overflow".into()))?;
        Ok(GatedDeltaNetState {
            dimensions: self.dimensions,
            recurrent_state: zeros(self.state_elements)?,
            history: zeros(history_elements)?,
        })
    }

    /// Consume one causal row and return the input to the compressed output projection.
    ///
    /// The row runs through the public stages in order: [`prepare`](Self::prepare),
    /// [`recur_head`](Self::recur_head) for each value head, then
    /// [`advance_history`](Self::advance_history). Heads do not interact, so a
    /// caller may instead run each head's chain of rows independently.
    pub fn step(
        &self,
        state: &mut GatedDeltaNetState,
        projected_qkv: &[f32],
        projected_z: &[f32],
        weighted_hidden_bf16: &[f32],
        inverse_rms: f32,
    ) -> Result<Vec<f32>> {
        if state.dimensions != self.dimensions {
            return Err(dimension_mismatch());
        }
        let history: Vec<&[f32]> = state.history_rows().collect();
        let row = self.prepare(
            &history,
            projected_qkv,
            projected_z,
            weighted_hidden_bf16,
            inverse_rms,
        )?;
        // Work on a copy so a later finite-value error leaves the caller's state intact.
        let mut next_recurrent_state = state.recurrent_state.clone();
        let mut gated = zeros(self.output_width)?;
        let head_dim = self.dimensions.head_dim;
        for (value_head, (head_state, output)) in next_recurrent_state
            .chunks_exact_mut(head_dim * head_dim)
            .zip(gated.chunks_exact_mut(head_dim))
            .enumerate()
        {
            output.copy_from_slice(&self.recur_head(head_state, &row, value_head)?);
        }
        state.recurrent_state = next_recurrent_state;
        self.advance_history(state, &row)?;
        Ok(gated)
    }

    /// Stage 1 of [`step`](Self::step): everything that does not read the recurrent
    /// state. `history` holds the previous `convolution_width - 1` rows' rounded
    /// projections ([`round_projection`]), oldest first, as in
    /// [`GatedDeltaNetState::history_rows`] followed by any earlier rows of a pass.
    pub fn prepare(
        &self,
        history: &[&[f32]],
        projected_qkv: &[f32],
        projected_z: &[f32],
        weighted_hidden_bf16: &[f32],
        inverse_rms: f32,
    ) -> Result<GatedDeltaNetRow> {
        self.prepare_with_gates(
            history,
            projected_qkv,
            projected_z,
            weighted_hidden_bf16,
            inverse_rms,
            |weights, hidden, rms| self.dense_gates(weights, hidden, rms),
        )
    }

    /// Shared preparation with independent dense-output work supplied by the engine.
    pub fn prepare_with_gates(
        &self,
        history: &[&[f32]],
        projected_qkv: &[f32],
        projected_z: &[f32],
        weighted_hidden_bf16: &[f32],
        inverse_rms: f32,
        mut dense: impl FnMut(&[f32], &[f32], f32) -> Result<Vec<f32>>,
    ) -> Result<GatedDeltaNetRow> {
        if history.len() != self.dimensions.convolution_width - 1
            || history
                .iter()
                .any(|row| row.len() != self.convolution_channels)
            || projected_qkv.len() != self.convolution_channels
            || projected_z.len() != self.output_width
            || weighted_hidden_bf16.len() != self.dimensions.hidden_width
        {
            return Err(dimension_mismatch());
        }
        finite(projected_qkv, "DeltaNet QKV projection")?;
        finite(projected_z, "DeltaNet Z projection")?;
        finite(weighted_hidden_bf16, "DeltaNet dense gate input")?;
        finite(&[inverse_rms], "DeltaNet inverse RMS")?;

        // The projection boundary is BF16, including the history retained for later rows.
        let projected = round_projection(projected_qkv);
        let z: Vec<f32> = projected_z.iter().map(|&value| bfloat(value)).collect();
        finite(&projected, "DeltaNet rounded QKV projection")?;
        finite(&z, "DeltaNet rounded Z projection")?;

        let mut convolved = zeros(self.convolution_channels)?;
        for (channel, output) in convolved.iter_mut().enumerate() {
            let weight_start = channel * self.dimensions.convolution_width;
            let mut value = self.weights.convolution_weight[weight_start] * history[0][channel];
            // Oldest tap first, with no leading zero addition and no fused multiply-add.
            for tap in 1..self.dimensions.convolution_width - 1 {
                value +=
                    self.weights.convolution_weight[weight_start + tap] * history[tap][channel];
            }
            value += self.weights.convolution_weight
                [weight_start + self.dimensions.convolution_width - 1]
                * projected[channel];
            finite(&[value], "DeltaNet convolution")?;
            // Convolution is rounded before SiLU and again after SiLU.
            *output = bfloat(silu(bfloat(value)));
        }
        finite(&convolved, "DeltaNet convolved activation")?;

        let mut query = zeros(self.query_key_width)?;
        let mut key = zeros(self.query_key_width)?;
        for head in 0..self.dimensions.query_key_heads {
            let start = head * self.dimensions.head_dim;
            let end = start + self.dimensions.head_dim;
            let query_head = &convolved[start..end];
            let key_head = &convolved[self.query_key_width + start..self.query_key_width + end];
            let query_square_sum = squares128(query_head);
            let key_square_sum = squares128(key_head);
            finite(
                &[query_square_sum, key_square_sum],
                "DeltaNet query/key norm",
            )?;
            let query_inverse_length = rsqrt(query_square_sum + 1.0e-6);
            let key_inverse_length = rsqrt(key_square_sum + 1.0e-6);
            for channel in 0..self.dimensions.head_dim {
                // The documented factor is the rounded 1/sqrt(128); no BF16 follows.
                query[start + channel] =
                    (query_head[channel] * query_inverse_length) * 0.088_388_346;
                key[start + channel] = key_head[channel] * key_inverse_length;
            }
        }
        finite(&query, "DeltaNet query")?;
        finite(&key, "DeltaNet key")?;

        let gates_a = dense(
            &self.weights.projection_a,
            weighted_hidden_bf16,
            inverse_rms,
        )?;
        let gates_b = dense(
            &self.weights.projection_b,
            weighted_hidden_bf16,
            inverse_rms,
        )?;
        if gates_a.len() != self.dimensions.value_heads
            || gates_b.len() != self.dimensions.value_heads
        {
            return Err(dimension_mismatch());
        }
        finite(&gates_a, "DeltaNet dense gates")?;
        finite(&gates_b, "DeltaNet dense gates")?;
        Ok(GatedDeltaNetRow {
            projected,
            z,
            convolved,
            query,
            key,
            gates_a,
            gates_b,
        })
    }

    /// Stage 2 of [`step`](Self::step): one value head's recurrence for one row.
    /// `head_state` is that head's `head_dim × head_dim` block of the recurrent
    /// state (see [`GatedDeltaNetState::recurrent_heads_mut`]); it is updated in
    /// place, so callers keep a copy if an error must leave it unchanged. Returns
    /// the head's `head_dim` gated outputs. No other head is read or written.
    pub fn recur_head(
        &self,
        head_state: &mut [f32],
        row: &GatedDeltaNetRow,
        value_head: usize,
    ) -> Result<Vec<f32>> {
        self.recur_head_with(
            head_state,
            row,
            value_head,
            |head_state, query, key, value, decay, update_gate| {
                let head_dim = self.dimensions.head_dim;
                let mut core = zeros(head_dim)?;
                for (value_channel, core_value) in core.iter_mut().enumerate() {
                    let mut prediction_part = [0.0; 4];
                    for (group, partial) in prediction_part.iter_mut().enumerate() {
                        // Four ordered 32-term sums are distinct from XOR reduction.
                        for offset in 0..32 {
                            let key_channel = group * 32 + offset;
                            let index = key_channel * head_dim + value_channel;
                            head_state[index] *= decay;
                            *partial += head_state[index] * key[key_channel];
                        }
                    }
                    let prediction = sum4(prediction_part);
                    let value = value[value_channel];
                    let correction = (value - prediction) * update_gate;
                    finite(&[prediction, correction], "DeltaNet recurrent prediction")?;
                    let mut output_part = [0.0; 4];
                    for (group, partial) in output_part.iter_mut().enumerate() {
                        for offset in 0..32 {
                            let key_channel = group * 32 + offset;
                            let index = key_channel * head_dim + value_channel;
                            head_state[index] += key[key_channel] * correction;
                            *partial += head_state[index] * query[key_channel];
                        }
                    }
                    // The recurrent output is rounded before per-head normalization.
                    *core_value = bfloat(sum4(output_part));
                }
                Ok(core)
            },
        )
    }

    /// Engine core hook; validation, decay, normalization and BF16 gating retain the scalar contract.
    pub fn recur_head_with(
        &self,
        head_state: &mut [f32],
        row: &GatedDeltaNetRow,
        value_head: usize,
        core_step: impl FnOnce(&mut [f32], &[f32], &[f32], &[f32], f32, f32) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let head_dim = self.dimensions.head_dim;
        if head_state.len() != head_dim * head_dim
            || value_head >= self.dimensions.value_heads
            || row.convolved.len() != self.convolution_channels
            || row.z.len() != self.output_width
        {
            return Err(dimension_mismatch());
        }
        let GatedDeltaNetRow {
            z,
            convolved,
            query,
            key,
            gates_a,
            gates_b,
            ..
        } = row;
        let value_heads_per_key_head =
            self.dimensions.value_heads / self.dimensions.query_key_heads;
        let key_head = value_head / value_heads_per_key_head;
        let key_start = key_head * head_dim;
        let output_start = value_head * head_dim;
        // Beta has a BF16 boundary; log_decay and its exponential remain FP32.
        let update_gate = bfloat(sigmoid(gates_b[value_head]));
        let decay_weight = exp(self.weights.log_decay_weight[value_head]);
        let biased_gate = gates_a[value_head] + self.weights.time_bias[value_head];
        let log_decay = -decay_weight * softplus(biased_gate);
        finite(
            &[update_gate, decay_weight, biased_gate, log_decay],
            "DeltaNet decay",
        )?;
        let decay = exp(log_decay);
        finite(&[decay], "DeltaNet decay factor")?;
        let core = core_step(
            head_state,
            &query[key_start..key_start + head_dim],
            &key[key_start..key_start + head_dim],
            &convolved[2 * self.query_key_width + output_start
                ..2 * self.query_key_width + output_start + head_dim],
            decay,
            update_gate,
        )?;
        if core.len() != head_dim {
            return Err(dimension_mismatch());
        }
        finite(&core, "DeltaNet recurrent output")?;
        let square_sum = squares128(&core);
        finite(&[square_sum], "DeltaNet output norm")?;
        let normalization_argument = square_sum / head_dim as f32 + self.epsilon;
        finite(&[normalization_argument], "DeltaNet output norm argument")?;
        let output_inverse_rms = rsqrt(normalization_argument);
        let mut gated = zeros(head_dim)?;
        for (channel, output) in gated.iter_mut().enumerate() {
            // Chapter 03 requires a round after RMS, after gain, and after gating.
            let normalized = bfloat(
                self.weights.normalization_weight[channel]
                    * bfloat(core[channel] * output_inverse_rms),
            );
            *output = bfloat(normalized * silu(z[output_start + channel]));
        }
        finite(head_state, "DeltaNet recurrent state")?;
        finite(&gated, "DeltaNet gated output")?;
        Ok(gated)
    }

    /// Stage 3 of [`step`](Self::step): append the row's rounded projection to the
    /// convolution history, after every head of that row has succeeded.
    pub fn advance_history(
        &self,
        state: &mut GatedDeltaNetState,
        row: &GatedDeltaNetRow,
    ) -> Result<()> {
        if state.dimensions != self.dimensions || row.projected.len() != self.convolution_channels {
            return Err(dimension_mismatch());
        }
        state.history.copy_within(self.convolution_channels.., 0);
        let newest = state.history.len() - self.convolution_channels;
        state.history[newest..].copy_from_slice(&row.projected);
        Ok(())
    }

    fn dense_gates(
        &self,
        weights: &[f32],
        weighted_hidden_bf16: &[f32],
        inverse_rms: f32,
    ) -> Result<Vec<f32>> {
        let mut gates = zeros(self.dimensions.value_heads)?;
        for (head, gate) in gates.iter_mut().enumerate() {
            let weight = &weights
                [head * self.dimensions.hidden_width..(head + 1) * self.dimensions.hidden_width];
            *gate = dense_gate(weight, weighted_hidden_bf16, inverse_rms)?;
        }
        finite(&gates, "DeltaNet dense gates")?;
        Ok(gates)
    }
}

/// One dense output, retaining the scalar partition and XOR reduction order.
pub fn dense_gate(weight: &[f32], hidden: &[f32], inverse_rms: f32) -> Result<f32> {
    dense_gate_with_partials(weight, hidden, inverse_rms, |weight, hidden| {
        let mut partial = [0.0; 128];
        for (partition, sum) in partial.iter_mut().enumerate() {
            for chunk in 0..hidden.len().div_ceil(1024) {
                for element in 0..8 {
                    let index = 8 * partition + 1024 * chunk + element;
                    if index < hidden.len() {
                        *sum += weight[index] * hidden[index];
                    }
                }
            }
        }
        partial
    })
}

/// Independent partial accumulators; the reference owns validation and reduction.
pub fn dense_gate_with_partials(
    weight: &[f32],
    hidden: &[f32],
    inverse_rms: f32,
    partials: impl FnOnce(&[f32], &[f32]) -> [f32; 128],
) -> Result<f32> {
    if weight.len() != hidden.len() {
        return Err(dimension_mismatch());
    }
    let partial = partials(weight, hidden);
    finite(&partial, "DeltaNet dense gate partial")?;
    let grouped = std::array::from_fn(|group| xor_sum32(&partial[group * 32..(group + 1) * 32]));
    let gate = bfloat(sum4(grouped) * inverse_rms);
    finite(&[gate], "DeltaNet dense gates")?;
    Ok(gate)
}

pub(crate) fn xor_sum32(values: &[f32]) -> f32 {
    debug_assert_eq!(values.len(), 32);
    let mut lanes = [0.0; 32];
    lanes.copy_from_slice(values);
    // Simultaneous XOR stages preserve the documented GPU reduction topology.
    for distance in [16, 8, 4, 2, 1] {
        let previous = lanes;
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = previous[lane] + previous[lane ^ distance];
        }
    }
    lanes[0]
}

fn sum4(values: [f32; 4]) -> f32 {
    // There is no leading addition to zero in the chapter's four-group fold.
    ((values[0] + values[1]) + values[2]) + values[3]
}

fn squares128(values: &[f32]) -> f32 {
    debug_assert_eq!(values.len(), 128);
    let grouped = std::array::from_fn(|group| {
        let products: [f32; 32] =
            std::array::from_fn(|lane| values[group * 32 + lane] * values[group * 32 + lane]);
        xor_sum32(&products)
    });
    sum4(grouped)
}

fn finite(values: &[f32], context: &'static str) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite(context))
    }
}

fn zeros(length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::InvalidModel("DeltaNet allocation exceeds available memory".into()))?;
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer() -> GatedDeltaNet {
        let dimensions = GatedDeltaNetDimensions {
            hidden_width: 512,
            query_key_heads: 2,
            value_heads: 4,
            head_dim: 128,
            convolution_width: 4,
        };
        let convolution_channels =
            (2 * dimensions.query_key_heads + dimensions.value_heads) * dimensions.head_dim;
        GatedDeltaNet::new(
            dimensions,
            GatedDeltaNetWeights {
                projection_a: vec![0.0; dimensions.value_heads * dimensions.hidden_width],
                projection_b: vec![0.0; dimensions.value_heads * dimensions.hidden_width],
                convolution_weight: vec![0.25; convolution_channels * dimensions.convolution_width],
                log_decay_weight: vec![-2.0; dimensions.value_heads],
                time_bias: vec![0.0; dimensions.value_heads],
                normalization_weight: vec![1.0; dimensions.head_dim],
            },
            1.0e-6,
        )
        .unwrap()
    }

    #[test]
    fn history_retains_raw_projected_bfloat_values() {
        let layer = layer();
        let mut state = layer.new_state().unwrap();
        let z = vec![0.125; layer.output_width];
        let weighted_hidden = vec![0.0; layer.dimensions.hidden_width];
        for input in [0.1, -0.2, 0.3, 0.4] {
            layer
                .step(
                    &mut state,
                    &vec![input; layer.convolution_channels],
                    &z,
                    &weighted_hidden,
                    1.0,
                )
                .unwrap();
        }
        for (age, input) in [-0.2, 0.3, 0.4].into_iter().enumerate() {
            for value in &state.history
                [age * layer.convolution_channels..(age + 1) * layer.convolution_channels]
            {
                assert_eq!(value.to_bits(), bfloat(input).to_bits());
            }
        }
    }

    #[test]
    fn repeated_rows_and_cloned_state_have_identical_bits() {
        let layer = layer();
        let mut committed = layer.new_state().unwrap();
        let mut provisional = committed.clone();
        let qkv: Vec<f32> = (0..layer.convolution_channels)
            .map(|index| (index % 17) as f32 / 128.0 - 0.0625)
            .collect();
        let z = vec![0.5; layer.output_width];
        let weighted_hidden = vec![0.125; layer.dimensions.hidden_width];
        for _ in 0..4 {
            let left = layer
                .step(&mut committed, &qkv, &z, &weighted_hidden, 0.75)
                .unwrap();
            let right = layer
                .step(&mut provisional, &qkv, &z, &weighted_hidden, 0.75)
                .unwrap();
            assert_eq!(
                left.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                right
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(committed, provisional);
    }

    #[test]
    fn head_major_stages_equal_row_major_steps() {
        let mut layer = layer();
        for (index, weight) in layer.weights.projection_a.iter_mut().enumerate() {
            *weight = (index % 13) as f32 / 64.0 - 0.09;
        }
        let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..6)
            .map(|row| {
                let qkv = (0..layer.convolution_channels)
                    .map(|i| ((i * 7 + row * 5) % 23) as f32 / 32.0 - 0.3)
                    .collect();
                let z = (0..layer.output_width)
                    .map(|i| ((i + row) % 9) as f32 / 8.0 - 0.5)
                    .collect();
                (qkv, z)
            })
            .collect();
        let hidden = vec![0.125; layer.dimensions.hidden_width];
        let mut stepped = layer.new_state().unwrap();
        let expected: Vec<Vec<f32>> = rows
            .iter()
            .map(|(qkv, z)| layer.step(&mut stepped, qkv, z, &hidden, 0.75).unwrap())
            .collect();

        let mut state = layer.new_state().unwrap();
        let rounded: Vec<_> = rows.iter().map(|(qkv, _)| round_projection(qkv)).collect();
        let taps: Vec<&[f32]> = state
            .history_rows()
            .chain(rounded.iter().map(Vec::as_slice))
            .collect();
        let prepared: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(i, (qkv, z))| {
                layer
                    .prepare(&taps[i..i + 3], qkv, z, &hidden, 0.75)
                    .unwrap()
            })
            .collect();
        drop(taps);
        let mut actual = vec![Vec::new(); rows.len()];
        // Heads in reverse order: no head may depend on another.
        for (head, head_state) in state.recurrent_heads_mut().enumerate().rev() {
            for (row, output) in prepared.iter().zip(&mut actual) {
                let mut values = layer.recur_head(head_state, row, head).unwrap();
                values.extend_from_slice(output);
                *output = values;
            }
        }
        for row in &prepared {
            layer.advance_history(&mut state, row).unwrap();
        }
        let bits = |rows: &[Vec<f32>]| -> Vec<Vec<u32>> {
            rows.iter()
                .map(|row| row.iter().map(|v| v.to_bits()).collect())
                .collect()
        };
        assert_eq!(bits(&actual), bits(&expected));
        assert_eq!(state, stepped);
    }

    #[test]
    fn non_finite_input_does_not_mutate_state() {
        let layer = layer();
        let mut state = layer.new_state().unwrap();
        let original = state.clone();
        let mut qkv = vec![0.0; layer.convolution_channels];
        qkv[0] = f32::NAN;
        assert!(matches!(
            layer.step(
                &mut state,
                &qkv,
                &vec![0.0; layer.output_width],
                &vec![0.0; layer.dimensions.hidden_width],
                1.0,
            ),
            Err(Error::NonFinite(_))
        ));
        assert_eq!(state, original);
    }

    #[test]
    fn dense_gates_cover_a_small_hidden_width_once() {
        let mut layer = layer();
        layer.weights.projection_a.fill(1.0);
        let gates = layer
            .dense_gates(&layer.weights.projection_a, &vec![0.5; 512], 0.25)
            .unwrap();
        assert!(gates.iter().all(|&gate| gate == 64.0));
    }
}
