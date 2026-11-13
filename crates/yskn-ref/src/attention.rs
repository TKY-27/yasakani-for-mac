//! Scalar int8 attention from [chapter 04](../../../docs/spec/04_ATTENTION.md).

use crate::gated_deltanet::xor_sum32;
use crate::portable_math::{bfloat, exp, exp2, rsqrt, sigmoid, sincos};
use crate::{Error, Result};

const FREQUENCY_BITS: [u32; 32] = [
    0x3f80_0000,
    0x3f1a_b32b,
    0x3eba_f81b,
    0x3e61_f835,
    0x3e08_8d77,
    0x3da5_0956,
    0x3d47_763f,
    0x3cf1_1177,
    0x3c91_ad39,
    0x3c30_1052,
    0x3bd4_ca15,
    0x3b80_967d,
    0x3b1b_690d,
    0x3abb_d3ed,
    0x3a63_01e2,
    0x3a09_2e02,
    0x39a5_cb60,
    0x3948_60c1,
    0x38f2_2ce2,
    0x3892_587e,
    0x3830_df52,
    0x37d5_c441,
    0x3781_2dab,
    0x371c_1fc4,
    0x36bc_b0c1,
    0x3664_0cc6,
    0x3609_cf4a,
    0x35a6_8e4c,
    0x3549_4c57,
    0x34f3_499d,
    0x3493_048e,
    0x3431_af44,
];
const LOG2_E: f32 = std::f32::consts::LOG2_E;
const EMPTY_MAXIMUM: f32 = -1.0e30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionDimensions {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

#[derive(Clone, Debug)]
pub struct Attention {
    dimensions: AttentionDimensions,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    epsilon: f32,
    query_width: usize,
    kv_width: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct Int8CacheRow {
    integers: Vec<i8>,
    scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct CacheToken {
    keys: Vec<Int8CacheRow>,
    values: Vec<Int8CacheRow>,
}

/// Only the visible causal prefix is retained; a model pass owns its provisional copy.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionState {
    dimensions: AttentionDimensions,
    rows: Vec<CacheToken>,
}

impl AttentionState {
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

struct PreparedRow {
    queries: Vec<f32>,
    gates: Vec<f32>,
    cache: CacheToken,
}

struct AttentionSplit {
    maximum: f32,
    denominator: f32,
    numerator: Vec<f32>,
}

impl Attention {
    pub fn new(
        dimensions: AttentionDimensions,
        query_norm: Vec<f32>,
        key_norm: Vec<f32>,
        epsilon: f32,
    ) -> Result<Self> {
        if dimensions.query_heads == 0
            || dimensions.kv_heads == 0
            || dimensions.query_heads % dimensions.kv_heads != 0
        {
            return Err(Error::InvalidModel(
                "attention requires positive heads and an integral query/KV head ratio".into(),
            ));
        }
        if dimensions.head_dim != 256 || dimensions.rotary_dim != 64 {
            return Err(Error::InvalidModel(
                "chapter 04 specifies only 256-channel heads and 64 rotary channels".into(),
            ));
        }
        if query_norm.len() != dimensions.head_dim || key_norm.len() != dimensions.head_dim {
            return Err(Error::InvalidModel(
                "attention head normalization weights have the wrong length".into(),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidModel(
                "attention epsilon must be finite and positive".into(),
            ));
        }
        finite(&query_norm, "attention query normalization weights")?;
        finite(&key_norm, "attention key normalization weights")?;
        let query_width = dimensions
            .query_heads
            .checked_mul(dimensions.head_dim)
            .ok_or_else(|| Error::InvalidModel("attention query width overflow".into()))?;
        query_width
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidModel("attention query/gate width overflow".into()))?;
        let kv_width = dimensions
            .kv_heads
            .checked_mul(dimensions.head_dim)
            .ok_or_else(|| Error::InvalidModel("attention KV width overflow".into()))?;
        Ok(Self {
            dimensions,
            query_norm,
            key_norm,
            epsilon,
            query_width,
            kv_width,
        })
    }

    pub fn new_state(&self) -> Result<AttentionState> {
        Ok(AttentionState {
            dimensions: self.dimensions,
            rows: Vec::new(),
        })
    }

    /// Consume one causal row and return the input to the compressed output projection.
    pub fn step(
        &self,
        state: &mut AttentionState,
        position: usize,
        projected_query_gate: &[f32],
        projected_key: &[f32],
        projected_value: &[f32],
    ) -> Result<Vec<f32>> {
        if state.dimensions != self.dimensions || state.rows.len() != position {
            return Err(Error::InvalidModel(
                "attention position or state dimensions do not match the causal prefix".into(),
            ));
        }
        self.validate_projected_rows(projected_query_gate, projected_key, projected_value)?;
        let prepared = self.prepare_row(
            position,
            projected_query_gate,
            projected_key,
            projected_value,
        )?;
        let mut output = Vec::with_capacity(self.query_width);
        for query_head in 0..self.dimensions.query_heads {
            let start = query_head * self.dimensions.head_dim;
            let attended = self.attend_head(
                &prepared.queries[start..start + self.dimensions.head_dim],
                query_head,
                &state.rows,
                &prepared.cache,
            )?;
            let unrotated = inverse_rotate_cache(&attended);
            finite(&unrotated, "attention inverse cache rotation")?;
            for (channel, value) in unrotated.into_iter().enumerate() {
                // The inverse rotation, sigmoid, and product have separate BF16 boundaries.
                let rounded_attended = bfloat(value);
                let gate = bfloat(sigmoid(prepared.gates[start + channel]));
                output.push(bfloat(rounded_attended * gate));
            }
        }
        finite(&output, "attention gated output")?;
        state
            .rows
            .try_reserve(1)
            .map_err(|_| Error::InvalidModel("attention cache exceeds available memory".into()))?;
        state.rows.push(prepared.cache);
        Ok(output)
    }

    fn validate_projected_rows(
        &self,
        projected_query_gate: &[f32],
        projected_key: &[f32],
        projected_value: &[f32],
    ) -> Result<()> {
        if projected_query_gate.len() != 2 * self.query_width
            || projected_key.len() != self.kv_width
            || projected_value.len() != self.kv_width
        {
            return Err(Error::InvalidModel(
                "attention projected row dimensions do not match the layer".into(),
            ));
        }
        finite(projected_query_gate, "attention query/gate projection")?;
        finite(projected_key, "attention key projection")?;
        finite(projected_value, "attention value projection")
    }

    fn prepare_row(
        &self,
        position: usize,
        projected_query_gate: &[f32],
        projected_key: &[f32],
        projected_value: &[f32],
    ) -> Result<PreparedRow> {
        self.validate_projected_rows(projected_query_gate, projected_key, projected_value)?;
        let mut queries = Vec::with_capacity(self.query_width);
        let mut gates = Vec::with_capacity(self.query_width);
        for head in 0..self.dimensions.query_heads {
            let start = 2 * head * self.dimensions.head_dim;
            // Each head stores its query followed immediately by that head's raw gate.
            let query: Vec<f32> = projected_query_gate[start..start + self.dimensions.head_dim]
                .iter()
                .map(|&value| bfloat(value))
                .collect();
            let normalized = self.normalize_head(&query, &self.query_norm)?;
            let rotated = rotate_cache(&self.rope(&normalized, position)?);
            // Only queries have a BF16 boundary after the cache rotation.
            queries.extend(rotated.into_iter().map(bfloat));
            gates.extend(
                projected_query_gate
                    [start + self.dimensions.head_dim..start + 2 * self.dimensions.head_dim]
                    .iter()
                    .map(|&value| bfloat(value)),
            );
        }
        let mut keys = Vec::with_capacity(self.dimensions.kv_heads);
        let mut values = Vec::with_capacity(self.dimensions.kv_heads);
        for head in 0..self.dimensions.kv_heads {
            let start = head * self.dimensions.head_dim;
            let end = start + self.dimensions.head_dim;
            // The compressed projection boundary is BF16, before head norm or rotation.
            let key: Vec<f32> = projected_key[start..end]
                .iter()
                .map(|&value| bfloat(value))
                .collect();
            let value: Vec<f32> = projected_value[start..end]
                .iter()
                .map(|&value| bfloat(value))
                .collect();
            let normalized_key = self.normalize_head(&key, &self.key_norm)?;
            let rotated_key = rotate_cache(&self.rope(&normalized_key, position)?);
            keys.push(encode_int8(&rotated_key)?);
            // Values receive neither head normalization nor RoPE.
            values.push(encode_int8(&rotate_cache(&value))?);
        }
        finite(&queries, "attention rounded rotated query")?;
        finite(&gates, "attention rounded gate")?;
        Ok(PreparedRow {
            queries,
            gates,
            cache: CacheToken { keys, values },
        })
    }

    fn normalize_head(&self, values: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        finite(values, "attention head input")?;
        let mut square_sum = 0.0;
        // Each consecutive 32-channel group uses XOR reduction; groups fold from zero.
        for group in 0..self.dimensions.head_dim / 32 {
            let products: [f32; 32] = std::array::from_fn(|lane| {
                let value = values[group * 32 + lane];
                value * value
            });
            square_sum += xor_sum32(&products);
        }
        finite(&[square_sum], "attention head sum of squares")?;
        let normalization_argument = square_sum / self.dimensions.head_dim as f32 + self.epsilon;
        finite(&[normalization_argument], "attention head norm argument")?;
        let inverse_rms = rsqrt(normalization_argument);
        // Norm rounding follows both multiplications, not the intermediate RMS product.
        let normalized: Vec<f32> = values
            .iter()
            .zip(weight)
            .map(|(&value, &gain)| bfloat((value * inverse_rms) * (1.0 + gain)))
            .collect();
        finite(&normalized, "attention normalized head")?;
        Ok(normalized)
    }

    fn rope(&self, normalized: &[f32], position: usize) -> Result<Vec<f32>> {
        let mut rotated = normalized.to_vec();
        let pairs = self.dimensions.rotary_dim / 2;
        for pair in 0..pairs {
            // These are chapter 04's exact frequency bits, never regenerated from theta.
            let angle = position as f32 * f32::from_bits(FREQUENCY_BITS[pair]);
            let (cosine, sine) = sincos(angle)?;
            // The cached cosine/sine and both products are individually rounded to BF16.
            let cosine = bfloat(cosine);
            let sine = bfloat(sine);
            rotated[pair] = bfloat(
                bfloat(normalized[pair] * cosine) + bfloat(-normalized[pair + pairs] * sine),
            );
            rotated[pair + pairs] =
                bfloat(bfloat(normalized[pair + pairs] * cosine) + bfloat(normalized[pair] * sine));
        }
        finite(&rotated, "attention rotary head")?;
        Ok(rotated)
    }

    fn attend_head(
        &self,
        query: &[f32],
        query_head: usize,
        prefix: &[CacheToken],
        current: &CacheToken,
    ) -> Result<Vec<f32>> {
        let length = prefix
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::InvalidModel("attention context length overflow".into()))?;
        let kv_head = query_head / (self.dimensions.query_heads / self.dimensions.kv_heads);
        // These are the chapter's split policy, not model dimensions.
        let split_count = if length < 4096 { 31 } else { 42 };
        let per_split = length.div_ceil(split_count);
        let padded_count = if split_count <= 32 { 32 } else { 64 };
        let mut splits = Vec::with_capacity(padded_count);
        for split_index in 0..padded_count {
            let mut split = AttentionSplit {
                maximum: EMPTY_MAXIMUM,
                denominator: 0.0,
                numerator: vec![0.0; self.dimensions.head_dim],
            };
            if split_index < split_count {
                let first = split_index * per_split;
                let last = first.saturating_add(per_split).min(length);
                let prefix_end = first.max(last.min(length - 1));
                // D9: visit only absolute 32-slot tiles intersecting this row's prefix.
                if first < prefix_end {
                    let mut tile_start = first / 32 * 32;
                    while tile_start < prefix_end {
                        self.prefix_tile(
                            &mut split,
                            query,
                            kv_head,
                            prefix,
                            tile_start,
                            first..prefix_end,
                        )?;
                        tile_start += 32;
                    }
                }
                if first < length && length - 1 < first.saturating_add(per_split) {
                    self.add_self(
                        &mut split,
                        query,
                        &current.keys[kv_head],
                        &current.values[kv_head],
                    )?;
                }
            }
            splits.push(split);
        }
        let largest_maximum = splits
            .iter()
            .map(|split| split.maximum)
            .fold(EMPTY_MAXIMUM, f32::max);
        let mut scales = Vec::with_capacity(splits.len());
        for split in &splits {
            if split.maximum > -1.0e29 {
                let difference = split.maximum - largest_maximum;
                finite(&[difference], "attention split maximum difference")?;
                scales.push(exp(difference));
            } else {
                scales.push(0.0);
            }
        }
        let mut denominator = 0.0;
        // Splits, including zero padding, combine in ascending order with separate products.
        for (scale, split) in scales.iter().zip(&splits) {
            denominator += *scale * split.denominator;
        }
        finite(&[denominator], "attention combined denominator")?;
        if denominator <= 0.0 {
            return Err(Error::InvalidModel(
                "attention combined denominator is not positive".into(),
            ));
        }
        let mut attended = vec![0.0; self.dimensions.head_dim];
        for (channel, output) in attended.iter_mut().enumerate() {
            let mut numerator = 0.0;
            for (scale, split) in scales.iter().zip(&splits) {
                numerator += *scale * split.numerator[channel];
            }
            *output = numerator / denominator;
        }
        finite(&attended, "attention combined output")?;
        Ok(attended)
    }

    fn prefix_tile(
        &self,
        split: &mut AttentionSplit,
        query: &[f32],
        kv_head: usize,
        prefix: &[CacheToken],
        tile_start: usize,
        visible: std::ops::Range<usize>,
    ) -> Result<()> {
        let mut scores = [f32::NEG_INFINITY; 32];
        for (tile_index, score) in scores.iter_mut().enumerate() {
            let position = tile_start + tile_index;
            if visible.contains(&position) {
                let key = &prefix[position].keys[kv_head];
                let mut accumulator = 0.0;
                for chunk in 0..self.dimensions.head_dim / 16 {
                    // D3: provisional mma16 order is ascending scalar multiply then add.
                    for channel in chunk * 16..(chunk + 1) * 16 {
                        accumulator += query[channel] * f32::from(key.integers[channel]);
                    }
                }
                *score = accumulator * (key.scale * 0.0625);
                finite(&[*score], "attention prefix score")?;
            }
        }
        let new_maximum = scores.iter().copied().fold(split.maximum, f32::max);
        let rescale_argument = (split.maximum - new_maximum) * LOG2_E;
        let scaled_maximum = new_maximum * LOG2_E;
        finite(
            &[rescale_argument, scaled_maximum],
            "attention prefix exponent argument",
        )?;
        let rescale = exp2(rescale_argument);
        split.maximum = new_maximum;
        let mut probabilities = [0.0; 32];
        for (index, probability) in probabilities.iter_mut().enumerate() {
            if scores[index] != f32::NEG_INFINITY {
                // The score and maximum are multiplied separately before subtraction.
                let argument = scores[index] * LOG2_E - scaled_maximum;
                finite(&[argument], "attention prefix probability argument")?;
                *probability = exp2(argument);
            }
            // The causal mask denotes an exact zero probability, not a model activation.
        }
        finite(&probabilities, "attention prefix probability")?;
        let mut partial = [0.0; 4];
        for (lane, sum) in partial.iter_mut().enumerate() {
            // Preserve the paired probabilities and four interleaved partial sums.
            for group in 0..4 {
                let key = 8 * group + 2 * lane;
                *sum += probabilities[key] + probabilities[key + 1];
            }
        }
        let probability_sum = xor_sum4(partial);
        split.denominator = split.denominator * rescale + probability_sum;
        for value in &mut split.numerator {
            *value *= rescale;
        }
        let mut value_weights = [0.0; 32];
        for (tile_index, weight) in value_weights.iter_mut().enumerate() {
            let position = tile_start + tile_index;
            if visible.contains(&position) {
                let value_scale = prefix[position].values[kv_head].scale;
                // Prefix probability times the row scale rounds to BF16 before the value dot.
                *weight = bfloat(probabilities[tile_index] * value_scale);
            }
        }
        for half in 0..2 {
            for channel in 0..self.dimensions.head_dim {
                // D3: each mma16 consumes its 16 keys in ascending scalar order.
                for (offset, &value_weight) in
                    value_weights[half * 16..(half + 1) * 16].iter().enumerate()
                {
                    let position = tile_start + half * 16 + offset;
                    // D9 skips masked operations entirely, including zero-valued products.
                    if visible.contains(&position) {
                        let value = f32::from(prefix[position].values[kv_head].integers[channel]);
                        split.numerator[channel] += value_weight * value;
                    }
                }
            }
        }
        finite(&[split.denominator], "attention prefix denominator")?;
        finite(&split.numerator, "attention prefix numerator")
    }

    fn add_self(
        &self,
        split: &mut AttentionSplit,
        query: &[f32],
        key: &Int8CacheRow,
        value: &Int8CacheRow,
    ) -> Result<()> {
        let mut partial = [0.0; 4];
        for (lane, sum) in partial.iter_mut().enumerate() {
            for chunk in 0..self.dimensions.head_dim / 16 {
                // Self has its own interleaved FP32 order; it is not the prefix mma16.
                for offset in [2 * lane, 2 * lane + 1, 8 + 2 * lane, 9 + 2 * lane] {
                    let channel = 16 * chunk + offset;
                    *sum += query[channel] * f32::from(key.integers[channel]);
                }
            }
        }
        let score = xor_sum4(partial) * (key.scale * 0.0625);
        finite(&[score], "attention self score")?;
        let new_maximum = split.maximum.max(score);
        let rescale_argument = (split.maximum - new_maximum) * LOG2_E;
        // Self subtracts before multiplying by LOG2_E, unlike prefix probabilities.
        let probability_argument = (score - new_maximum) * LOG2_E;
        finite(
            &[rescale_argument, probability_argument],
            "attention self exponent argument",
        )?;
        let rescale = exp2(rescale_argument);
        let probability = exp2(probability_argument);
        split.maximum = new_maximum;
        split.denominator = split.denominator * rescale + probability;
        // No BF16 rounding is permitted for the separate self value weight.
        let value_weight = probability * value.scale;
        for (channel, numerator) in split.numerator.iter_mut().enumerate() {
            *numerator = *numerator * rescale + value_weight * f32::from(value.integers[channel]);
        }
        finite(&[split.denominator], "attention self denominator")?;
        finite(&split.numerator, "attention self numerator")
    }
}

fn cache_sign(channel: usize) -> f32 {
    let mut value = channel as u32 + 1;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    if value & 1 == 0 { 1.0 } else { -1.0 }
}

fn hadamard(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut distance = 1;
    // Ascending butterfly distance is observable after every binary32 add/subtract.
    while distance < values.len() {
        for block in (0..values.len()).step_by(2 * distance) {
            for offset in 0..distance {
                let first = values[block + offset];
                let second = values[block + offset + distance];
                values[block + offset] = first + second;
                values[block + offset + distance] = first - second;
            }
        }
        distance *= 2;
    }
}

fn rotate_cache(values: &[f32]) -> Vec<f32> {
    let mut rotated: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(channel, &value)| value * cache_sign(channel))
        .collect();
    hadamard(&mut rotated);
    for value in &mut rotated {
        // The supported 256-channel rotation has exactly representable normalization 1/16.
        *value *= 0.0625;
    }
    rotated
}

fn inverse_rotate_cache(values: &[f32]) -> Vec<f32> {
    let mut rotated = values.to_vec();
    hadamard(&mut rotated);
    for (channel, value) in rotated.iter_mut().enumerate() {
        *value = (*value * 0.0625) * cache_sign(channel);
    }
    rotated
}

fn encode_int8(values: &[f32]) -> Result<Int8CacheRow> {
    finite(values, "attention cache input")?;
    let maximum = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
    let scale = maximum / 127.0;
    let integers = values
        .iter()
        .map(|&value| {
            if scale > 0.0 {
                // Cache quantization is nearest-even, then saturation to the symmetric range.
                (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8
            } else {
                0
            }
        })
        .collect();
    Ok(Int8CacheRow { integers, scale })
}

fn xor_sum4(mut partial: [f32; 4]) -> f32 {
    // Self scores and prefix denominators use simultaneous XOR distances 1 then 2.
    for distance in [1, 2] {
        let previous = partial;
        for (lane, value) in partial.iter_mut().enumerate() {
            *value = previous[lane] + previous[lane ^ distance];
        }
    }
    partial[0]
}

fn finite(values: &[f32], context: &'static str) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite(context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer() -> Attention {
        Attention::new(
            AttentionDimensions {
                query_heads: 2,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
            },
            vec![0.0; 256],
            vec![0.0; 256],
            1.0e-6,
        )
        .unwrap()
    }

    #[test]
    fn int8_encoding_handles_zero_and_halfway_ties() {
        let zero = encode_int8(&[0.0; 256]).unwrap();
        assert_eq!(zero.scale.to_bits(), 0.0_f32.to_bits());
        assert!(zero.integers.iter().all(|&value| value == 0));
        let encoded = encode_int8(&[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]).unwrap();
        assert_eq!(encoded.scale, 1.0);
        assert_eq!(encoded.integers, [127, 0, 2, 2, 0, -2, -2]);
    }

    #[test]
    fn cache_rotation_roundtrip_preserves_small_integer_vectors() {
        // Nonzero integers avoid asserting that inverse signs preserve the sign of zero.
        let values: Vec<f32> = (0..256)
            .map(|index| (index % 23) as f32 * 2.0 - 21.0)
            .collect();
        let restored = inverse_rotate_cache(&rotate_cache(&values));
        assert_eq!(
            restored
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn query_projection_keeps_each_heads_gate_interleaved() {
        let layer = layer();
        let mut query_gate = vec![0.0; 1024];
        query_gate[256..512].fill(0.25);
        query_gate[768..1024].fill(-0.5);
        let prepared = layer
            .prepare_row(0, &query_gate, &[0.0; 512], &[0.0; 512])
            .unwrap();
        assert!(prepared.gates[..256].iter().all(|&value| value == 0.25));
        assert!(prepared.gates[256..].iter().all(|&value| value == -0.5));
        assert!(prepared.queries.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn normalization_rejects_infinite_argument_before_rsqrt() {
        let mut layer = layer();
        layer.epsilon = f32::MAX;
        assert!(matches!(
            layer.normalize_head(&[1.0e18; 256], &[0.0; 256]),
            Err(Error::NonFinite(_))
        ));
    }

    #[test]
    fn successful_step_appends_cache_and_failed_step_preserves_it() {
        let layer = layer();
        let mut state = layer.new_state().unwrap();
        let output = layer
            .step(&mut state, 0, &[0.0; 1024], &[0.0; 512], &[0.0; 512])
            .unwrap();
        assert_eq!(state.len(), 1);
        assert!(output.iter().all(|value| value.is_finite()));
        let original = state.clone();
        assert!(matches!(
            layer.step(
                &mut state,
                1,
                &[0.0; 1024],
                &[f32::INFINITY; 512],
                &[0.0; 512]
            ),
            Err(Error::NonFinite(_))
        ));
        assert_eq!(state, original);
    }

    #[test]
    fn masked_prefix_cache_contents_cannot_change_output_bits() {
        let layer = layer();
        let mut prefix = vec![
            CacheToken {
                keys: vec![
                    Int8CacheRow {
                        integers: vec![2; 256],
                        scale: 0.25
                    };
                    2
                ],
                values: vec![
                    Int8CacheRow {
                        integers: vec![-3; 256],
                        scale: 0.125
                    };
                    2
                ],
            };
            64
        ];
        let visible = 35..38;
        let evaluate = |prefix: &[CacheToken]| {
            let mut split = AttentionSplit {
                maximum: EMPTY_MAXIMUM,
                denominator: 0.0,
                numerator: vec![0.0; 256],
            };
            layer
                .prefix_tile(&mut split, &[0.25; 256], 0, prefix, 32, visible.clone())
                .unwrap();
            (
                split.maximum.to_bits(),
                split.denominator.to_bits(),
                split
                    .numerator
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            )
        };
        let expected = evaluate(&prefix);
        assert_eq!(expected.0, 2.0_f32.to_bits());
        assert_eq!(expected.1, 3.0_f32.to_bits());
        assert_eq!(expected.2, vec![(-1.125_f32).to_bits(); 256]);
        for (position, token) in prefix.iter_mut().enumerate() {
            if !visible.contains(&position) {
                for row in token.keys.iter_mut().chain(&mut token.values) {
                    row.scale = f32::NAN;
                    row.integers.fill(-127);
                }
            }
        }
        assert_eq!(evaluate(&prefix), expected);
        // Neither future slots nor padding beyond the causal cache need to exist.
        prefix.truncate(visible.end);
        assert_eq!(evaluate(&prefix), expected);

        // Substituting +0 operands for masked slots would erase these negative zeros.
        for token in &mut prefix[visible.clone()] {
            token.keys[0].integers.fill(0);
            token.values[0].scale = f32::from_bits(32);
            token.values[0].integers.fill(-127);
        }
        let mut split = AttentionSplit {
            maximum: 0.0,
            denominator: 0.0,
            numerator: vec![-0.0; 256],
        };
        layer
            .prefix_tile(&mut split, &[0.0; 256], 0, &prefix, 32, visible)
            .unwrap();
        assert!(
            split
                .numerator
                .iter()
                .all(|value| value.to_bits() == (-0.0_f32).to_bits())
        );
    }

    #[test]
    fn non_finite_cache_input_is_an_error() {
        assert!(matches!(
            encode_int8(&[f32::INFINITY]),
            Err(Error::NonFinite(_))
        ));
    }
}
