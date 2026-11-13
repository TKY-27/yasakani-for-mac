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
    pub fn step(
        &self,
        state: &mut GatedDeltaNetState,
        projected_qkv: &[f32],
        projected_z: &[f32],
        weighted_hidden_bf16: &[f32],
        inverse_rms: f32,
    ) -> Result<Vec<f32>> {
        if state.dimensions != self.dimensions
            || projected_qkv.len() != self.convolution_channels
            || projected_z.len() != self.output_width
            || weighted_hidden_bf16.len() != self.dimensions.hidden_width
        {
            return Err(Error::InvalidModel(
                "DeltaNet row or state dimensions do not match the layer".into(),
            ));
        }
        finite(projected_qkv, "DeltaNet QKV projection")?;
        finite(projected_z, "DeltaNet Z projection")?;
        finite(weighted_hidden_bf16, "DeltaNet dense gate input")?;
        finite(&[inverse_rms], "DeltaNet inverse RMS")?;

        // The projection boundary is BF16, including the history retained for later rows.
        let projected: Vec<f32> = projected_qkv.iter().map(|&value| bfloat(value)).collect();
        let z: Vec<f32> = projected_z.iter().map(|&value| bfloat(value)).collect();
        finite(&projected, "DeltaNet rounded QKV projection")?;
        finite(&z, "DeltaNet rounded Z projection")?;

        let mut convolved = zeros(self.convolution_channels)?;
        for (channel, output) in convolved.iter_mut().enumerate() {
            let weight_start = channel * self.dimensions.convolution_width;
            let mut value = self.weights.convolution_weight[weight_start] * state.history[channel];
            // Oldest tap first, with no leading zero addition and no fused multiply-add.
            for tap in 1..self.dimensions.convolution_width - 1 {
                value += self.weights.convolution_weight[weight_start + tap]
                    * state.history[tap * self.convolution_channels + channel];
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

        let gates_a = self.dense_gates(
            &self.weights.projection_a,
            weighted_hidden_bf16,
            inverse_rms,
        )?;
        let gates_b = self.dense_gates(
            &self.weights.projection_b,
            weighted_hidden_bf16,
            inverse_rms,
        )?;
        // Work on a copy so a later finite-value error leaves the caller's state intact.
        let mut next_recurrent_state = state.recurrent_state.clone();
        let mut gated = zeros(self.output_width)?;
        let value_heads_per_key_head =
            self.dimensions.value_heads / self.dimensions.query_key_heads;
        for value_head in 0..self.dimensions.value_heads {
            let key_head = value_head / value_heads_per_key_head;
            let key_start = key_head * self.dimensions.head_dim;
            let output_start = value_head * self.dimensions.head_dim;
            let matrix_start = output_start * self.dimensions.head_dim;
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
            let mut core = zeros(self.dimensions.head_dim)?;
            for (value_channel, core_value) in core.iter_mut().enumerate() {
                let mut prediction_part = [0.0; 4];
                for (group, partial) in prediction_part.iter_mut().enumerate() {
                    // Four ordered 32-term sums are distinct from XOR reduction.
                    for offset in 0..32 {
                        let key_channel = group * 32 + offset;
                        let index =
                            matrix_start + key_channel * self.dimensions.head_dim + value_channel;
                        next_recurrent_state[index] *= decay;
                        *partial += next_recurrent_state[index] * key[key_start + key_channel];
                    }
                }
                let prediction = sum4(prediction_part);
                let value = convolved[2 * self.query_key_width + output_start + value_channel];
                let correction = (value - prediction) * update_gate;
                finite(&[prediction, correction], "DeltaNet recurrent prediction")?;
                let mut output_part = [0.0; 4];
                for (group, partial) in output_part.iter_mut().enumerate() {
                    for offset in 0..32 {
                        let key_channel = group * 32 + offset;
                        let index =
                            matrix_start + key_channel * self.dimensions.head_dim + value_channel;
                        next_recurrent_state[index] += key[key_start + key_channel] * correction;
                        *partial += next_recurrent_state[index] * query[key_start + key_channel];
                    }
                }
                // The recurrent output is rounded before per-head normalization.
                *core_value = bfloat(sum4(output_part));
            }
            finite(&core, "DeltaNet recurrent output")?;
            let square_sum = squares128(&core);
            finite(&[square_sum], "DeltaNet output norm")?;
            let normalization_argument =
                square_sum / self.dimensions.head_dim as f32 + self.epsilon;
            finite(&[normalization_argument], "DeltaNet output norm argument")?;
            let output_inverse_rms = rsqrt(normalization_argument);
            for channel in 0..self.dimensions.head_dim {
                // Chapter 03 requires a round after RMS, after gain, and after gating.
                let normalized = bfloat(
                    self.weights.normalization_weight[channel]
                        * bfloat(core[channel] * output_inverse_rms),
                );
                gated[output_start + channel] =
                    bfloat(normalized * silu(z[output_start + channel]));
            }
        }
        finite(&next_recurrent_state, "DeltaNet recurrent state")?;
        finite(&gated, "DeltaNet gated output")?;
        state.recurrent_state = next_recurrent_state;
        state.history.copy_within(self.convolution_channels.., 0);
        let newest = state.history.len() - self.convolution_channels;
        state.history[newest..].copy_from_slice(&projected);
        Ok(gated)
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
            let mut partial = [0.0; 128];
            for (partition, partial_sum) in partial.iter_mut().enumerate() {
                // Keep the chapter's partition and chunk order as hidden width varies.
                // Empty partitions contribute zero; every real input is visited once.
                for chunk in 0..self.dimensions.hidden_width.div_ceil(1024) {
                    for element in 0..8 {
                        let index = 8 * partition + 1024 * chunk + element;
                        if index < self.dimensions.hidden_width {
                            *partial_sum += weight[index] * weighted_hidden_bf16[index];
                        }
                    }
                }
            }
            finite(&partial, "DeltaNet dense gate partial")?;
            let grouped =
                std::array::from_fn(|group| xor_sum32(&partial[group * 32..(group + 1) * 32]));
            // The dense projection is BF16 only after the deferred inverse RMS.
            *gate = bfloat(sum4(grouped) * inverse_rms);
        }
        finite(&gates, "DeltaNet dense gates")?;
        Ok(gates)
    }
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
