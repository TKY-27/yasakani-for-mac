//! Scalar selectable-cache attention from [chapter 04](../../../docs/spec/04_ATTENTION.md).

use std::sync::Arc;

use crate::gated_deltanet::xor_sum32;
use crate::kv_codec::{CacheRow, Int8Row, encode_packed, requantize_int8};
use crate::kv_profile::KvProfile;
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

/// Cache arithmetic contract; plain Int8 remains the loader default (D13).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvMode {
    #[default]
    Int8,
    Int8Profile,
    Kv5,
}

impl KvMode {
    /// Stable spelling used by command-line options and result files.
    pub fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int8Profile => "int8-profile",
            Self::Kv5 => "kv5",
        }
    }
}

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
    kv_mode: KvMode,
    kv_profile: Option<Arc<KvProfile>>,
    attention_ordinal: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct CacheToken {
    keys: Vec<CacheRow>,
    values: Vec<CacheRow>,
}

/// Causal cache rows. Engines may append provisionally and roll back by truncation.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionState {
    dimensions: AttentionDimensions,
    rows: Vec<CacheToken>,
}

impl AttentionState {
    /// Remove a provisional suffix without copying or reallocating the prefix.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.rows.len() {
            return Err(Error::InvalidInput(
                "cannot extend an attention cache by truncation".into(),
            ));
        }
        self.rows.truncate(len);
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.rows.capacity()
    }

    /// Diagnostic FNV-1a over every stored code and scale, independent of allocation.
    pub fn update_digest(&self, digest: &mut u64) {
        self.update_prefix_digest(digest, self.rows.len());
    }

    /// Digest only the committed prefix while later rows are provisional.
    pub fn update_prefix_digest(&self, digest: &mut u64, len: usize) {
        assert!(len <= self.rows.len());
        fn bytes(digest: &mut u64, values: &[u8]) {
            for &byte in values {
                *digest = (*digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        bytes(digest, &(len as u64).to_le_bytes());
        for token in &self.rows[..len] {
            for side in [&token.keys, &token.values] {
                bytes(digest, &(side.len() as u64).to_le_bytes());
                for row in side {
                    bytes(digest, &row.scale().to_bits().to_le_bytes());
                    match row {
                        CacheRow::Int8(row) => {
                            bytes(digest, &[0]);
                            for &value in row.integers() {
                                bytes(digest, &[value as u8]);
                            }
                        }
                        CacheRow::Packed(row) => {
                            bytes(digest, &[row.bits()]);
                            for value in row.words() {
                                bytes(digest, &value.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

pub struct PreparedRow {
    dimensions: AttentionDimensions,
    position: usize,
    queries: Vec<f32>,
    gates: Vec<f32>,
    cache: CacheToken,
}

struct AttentionSplit {
    maximum: f32,
    denominator: f32,
    numerator: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SplitPlan {
    tiles: Vec<(usize, std::ops::Range<usize>)>,
    include_self: bool,
}

/// Chapter 04's 31/42 splits with D9's absolute, per-row prefix tiles.
/// `length` includes self and is therefore positive.
fn prefix_schedule(length: usize) -> Vec<SplitPlan> {
    let split_count = if length < 4096 { 31 } else { 42 };
    let per_split = length.div_ceil(split_count);
    (0..split_count)
        .map(|split_index| {
            let first = split_index * per_split;
            let last = first.saturating_add(per_split).min(length);
            let prefix_end = first.max(last.min(length - 1));
            let mut tiles = Vec::new();
            if first < prefix_end {
                let mut tile_start = first / 32 * 32;
                while tile_start < prefix_end {
                    tiles.push((
                        tile_start,
                        first.max(tile_start)..prefix_end.min(tile_start + 32),
                    ));
                    tile_start += 32;
                }
            }
            SplitPlan {
                tiles,
                include_self: first < length && length - 1 < first.saturating_add(per_split),
            }
        })
        .collect()
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
            kv_mode: KvMode::Int8,
            kv_profile: None,
            attention_ordinal: 0,
        })
    }

    /// Configure before creating sequence state. Arc keeps engine stage calls shareable.
    pub fn with_kv(
        mut self,
        mode: KvMode,
        profile: Option<Arc<KvProfile>>,
        attention_ordinal: usize,
    ) -> Result<Self> {
        if mode != KvMode::Int8 {
            let profile = profile.as_ref().ok_or_else(|| {
                Error::InvalidModel("profile KV mode requires a KVQP profile".into())
            })?;
            if profile.kv_heads() != self.dimensions.kv_heads
                || attention_ordinal >= profile.attention_layers()
            {
                return Err(Error::InvalidModel(
                    "attention/profile dimensions differ".into(),
                ));
            }
        }
        self.kv_mode = mode;
        self.kv_profile = if mode == KvMode::Int8 { None } else { profile };
        self.attention_ordinal = attention_ordinal;
        Ok(self)
    }

    fn group_index(&self, head: usize, side: usize) -> usize {
        // src/kvq_atxpk.hpp:65-68; ordinal counts attention layers only.
        (self.attention_ordinal * self.dimensions.kv_heads + head) * 2 + side
    }

    fn encode_row(
        &self,
        head: usize,
        side: usize,
        position: usize,
        row: &[f32],
    ) -> Result<CacheRow> {
        if self.kv_mode == KvMode::Int8 {
            return encode_int8(row);
        }
        let profile = self.kv_profile.as_ref().expect("validated KV profile");
        let group = self.group_index(head, side);
        let row = row.try_into().expect("256-channel cache row");
        // src/kvq_fast.cuh:78,189,218-234: guard uses the absolute cache position.
        match self.kv_mode {
            KvMode::Int8 => unreachable!(),
            KvMode::Int8Profile => Ok(CacheRow::Int8(requantize_int8(
                profile, group, position, row,
            )?)),
            KvMode::Kv5 => encode_packed(profile, group, position, row),
        }
    }

    fn value_mean<'a>(&'a self, head: usize, row: &CacheRow) -> Option<&'a [f32; 256]> {
        match row {
            CacheRow::Int8(_) => None,
            CacheRow::Packed(_) => {
                let profile = self.kv_profile.as_ref().expect("validated KV profile");
                let group = &profile.groups()[self.group_index(head, 1)];
                // Missing packed means still add +0, preserving signed-zero operations.
                Some(group.mean.as_ref().unwrap_or(&[0.0; 256]))
            }
        }
    }

    fn add_value_mean(split: &mut AttentionSplit, mean: Option<&[f32; 256]>) -> Result<()> {
        if let Some(mean) = mean {
            for (numerator, &mean) in split.numerator.iter_mut().zip(mean) {
                // After self; denominator is FP32 including self, not BF16 weight sums.
                // src/q35_atxpk3.cuh:443-475; src/q35_atxpk4.cuh:385-417.
                *numerator += split.denominator * mean;
            }
            finite(&split.numerator, "attention split value mean")?;
        }
        // Int8 fallback already restored mean/basis (src/kvq_fast.cuh:214-218,234).
        Ok(())
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
        let output = self.output_row(state, position, &prepared)?;
        self.append_prepared(state, position, &prepared)?;
        Ok(output)
    }

    /// Append cache rows in ascending position before running independent attention queries.
    pub fn append_prepared(
        &self,
        state: &mut AttentionState,
        position: usize,
        prepared: &PreparedRow,
    ) -> Result<()> {
        if state.dimensions != self.dimensions
            || state.rows.len() != position
            || prepared.dimensions != self.dimensions
            || prepared.position != position
        {
            return Err(Error::InvalidInput("invalid attention cache append".into()));
        }
        state
            .rows
            .try_reserve(1)
            .map_err(|_| Error::InvalidModel("attention cache exceeds available memory".into()))?;
        state.rows.push(prepared.cache.clone());
        Ok(())
    }

    pub fn query_heads(&self) -> usize {
        self.dimensions.query_heads
    }

    pub fn output_row(
        &self,
        state: &AttentionState,
        position: usize,
        prepared: &PreparedRow,
    ) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(self.query_width);
        for query_head in 0..self.dimensions.query_heads {
            output.extend(self.output_head(state, position, prepared, query_head)?);
        }
        finite(&output, "attention gated output")?;
        Ok(output)
    }

    /// Attend one head using only this row's causal prefix, even if later cache rows exist.
    pub fn output_head(
        &self,
        state: &AttentionState,
        position: usize,
        prepared: &PreparedRow,
        query_head: usize,
    ) -> Result<Vec<f32>> {
        if state.dimensions != self.dimensions
            || position > state.rows.len()
            || query_head >= self.dimensions.query_heads
            || prepared.dimensions != self.dimensions
            || prepared.position != position
        {
            return Err(Error::InvalidInput(
                "invalid attention prefix or head".into(),
            ));
        }
        let mut output = Vec::with_capacity(self.dimensions.head_dim);
        {
            let start = query_head * self.dimensions.head_dim;
            let attended = self.attend_head(
                &prepared.queries[start..start + self.dimensions.head_dim],
                query_head,
                &state.rows[..position],
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

    pub fn prepare_row(
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
            keys.push(self.encode_row(head, 0, position, &rotated_key)?);
            // Values receive neither head normalization nor RoPE.
            values.push(self.encode_row(head, 1, position, &rotate_cache(&value))?);
        }
        finite(&queries, "attention rounded rotated query")?;
        finite(&gates, "attention rounded gate")?;
        Ok(PreparedRow {
            dimensions: self.dimensions,
            position,
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
        let mean = self.value_mean(kv_head, &current.values[kv_head]);
        let schedule = prefix_schedule(length);
        let padded_count = schedule.len().next_power_of_two();
        let mut splits = Vec::with_capacity(padded_count);
        for plan in schedule {
            let mut split = AttentionSplit {
                maximum: EMPTY_MAXIMUM,
                denominator: 0.0,
                numerator: vec![0.0; self.dimensions.head_dim],
            };
            for (tile_start, visible) in plan.tiles {
                self.prefix_tile(&mut split, query, kv_head, prefix, tile_start, visible)?;
            }
            if plan.include_self {
                self.add_self(
                    &mut split,
                    query,
                    &current.keys[kv_head],
                    &current.values[kv_head],
                )?;
            }
            Self::add_value_mean(&mut split, mean)?;
            splits.push(split);
        }
        splits.resize_with(padded_count, || AttentionSplit {
            maximum: EMPTY_MAXIMUM,
            denominator: 0.0,
            numerator: vec![0.0; self.dimensions.head_dim],
        });
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
        let mut key_scratch = [0.0; 256];
        let mut scores = [f32::NEG_INFINITY; 32];
        for (tile_index, score) in scores.iter_mut().enumerate() {
            let position = tile_start + tile_index;
            if visible.contains(&position) {
                let key = &prefix[position].keys[kv_head];
                let bases = key.bases(&mut key_scratch);
                let mut accumulator = 0.0;
                for chunk in 0..self.dimensions.head_dim / 16 {
                    // D3: provisional mma16 order is ascending scalar multiply then add.
                    for (offset, &query_value) in
                        query[chunk * 16..(chunk + 1) * 16].iter().enumerate()
                    {
                        accumulator += query_value * bases[chunk * 16 + offset];
                    }
                }
                // Shipping QK8=false: no table-int8 scale, only the packed row scale.
                // src/q35_atxpk3.cuh:379; src/q35_atxpk4.cuh:325.
                *score = accumulator * (key.scale() * 0.0625);
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
        let mut value_scratch = [[0.0; 256]; 32];
        let mut value_bases = [None; 32];
        for (tile_index, scratch) in value_scratch.iter_mut().enumerate() {
            let position = tile_start + tile_index;
            if visible.contains(&position) {
                value_bases[tile_index] = Some(prefix[position].values[kv_head].bases(scratch));
            }
        }
        let mut value_weights = [0.0; 32];
        for (tile_index, weight) in value_weights.iter_mut().enumerate() {
            let position = tile_start + tile_index;
            if visible.contains(&position) {
                let value_scale = prefix[position].values[kv_head].scale();
                // Round probability * scale, not the value: src/q35_atxpk3.cuh:405-406;
                // src/q35_atxpk4.cuh:348-349; src/q35_atxpk.cu:55-56.
                *weight = bfloat(probabilities[tile_index] * value_scale);
            }
        }
        for half in 0..2 {
            for (channel, numerator) in split.numerator.iter_mut().enumerate() {
                // D3: each mma16 consumes its 16 keys in ascending scalar order.
                for (offset, &value_weight) in
                    value_weights[half * 16..(half + 1) * 16].iter().enumerate()
                {
                    // D9 skips masked operations entirely, including zero-valued products.
                    if let Some(bases) = value_bases[half * 16 + offset] {
                        *numerator += value_weight * bases[channel];
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
        key: &CacheRow,
        value: &CacheRow,
    ) -> Result<()> {
        let mut key_scratch = [0.0; 256];
        let key_bases = key.bases(&mut key_scratch);
        let mut value_scratch = [0.0; 256];
        let value_bases = value.bases(&mut value_scratch);
        let mut partial = [0.0; 4];
        for (lane, sum) in partial.iter_mut().enumerate() {
            for chunk in 0..self.dimensions.head_dim / 16 {
                // Self has its own interleaved FP32 order; it is not the prefix mma16.
                for offset in [2 * lane, 2 * lane + 1, 8 + 2 * lane, 9 + 2 * lane] {
                    let channel = 16 * chunk + offset;
                    *sum += query[channel] * key_bases[channel];
                }
            }
        }
        let score = xor_sum4(partial) * (key.scale() * 0.0625);
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
        // FP32 self weight and own encoded row: src/q35_atxpk3.cuh:421-458;
        // src/q35_atxpk4.cuh:363-400. Packed self still uses the BF16 table base.
        let value_weight = probability * value.scale();
        for (channel, numerator) in split.numerator.iter_mut().enumerate() {
            *numerator = *numerator * rescale + value_weight * value_bases[channel];
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

fn encode_int8(values: &[f32]) -> Result<CacheRow> {
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
    Ok(CacheRow::Int8(Int8Row { integers, scale }))
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
use crate::test_support::kv_profile as synthetic_kv_profile;

#[cfg(test)]
#[path = "attention_kv_tests.rs"]
mod kv_tests;

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

    fn plan(tiles: &[(usize, std::ops::Range<usize>)], include_self: bool) -> SplitPlan {
        SplitPlan {
            tiles: tiles.to_vec(),
            include_self,
        }
    }

    #[test]
    fn prefix_schedule_empty_and_short_prefixes() {
        // Chapter 04 assigns one slot per split at these lengths; D9 aligns tiles to zero.
        let only_self = prefix_schedule(1);
        assert_eq!(only_self.len(), 31);
        assert_eq!(only_self[0], plan(&[], true));
        assert_eq!(only_self[1..], vec![plan(&[], false); 30]);

        let short = prefix_schedule(5);
        assert_eq!(short.len(), 31);
        assert_eq!(
            short[..5],
            [
                plan(&[(0, 0..1)], false),
                plan(&[(0, 1..2)], false),
                plan(&[(0, 2..3)], false),
                plan(&[(0, 3..4)], false),
                plan(&[], true),
            ]
        );
        assert_eq!(short[5..], vec![plan(&[], false); 26]);
    }

    #[test]
    fn prefix_schedule_33_stops_before_the_self_tile() {
        let schedule = prefix_schedule(33);
        assert_eq!(schedule.len(), 31);
        assert_eq!(
            schedule[..17],
            [
                plan(&[(0, 0..2)], false),
                plan(&[(0, 2..4)], false),
                plan(&[(0, 4..6)], false),
                plan(&[(0, 6..8)], false),
                plan(&[(0, 8..10)], false),
                plan(&[(0, 10..12)], false),
                plan(&[(0, 12..14)], false),
                plan(&[(0, 14..16)], false),
                plan(&[(0, 16..18)], false),
                plan(&[(0, 18..20)], false),
                plan(&[(0, 20..22)], false),
                plan(&[(0, 22..24)], false),
                plan(&[(0, 24..26)], false),
                plan(&[(0, 26..28)], false),
                plan(&[(0, 28..30)], false),
                plan(&[(0, 30..32)], false),
                plan(&[], true),
            ]
        );
        assert_eq!(schedule[17..], vec![plan(&[], false); 14]);
    }

    #[test]
    fn prefix_schedule_76_splits_across_absolute_tile_boundaries() {
        // From chapter 04: 31 splits, three slots each; self is slot 75 in split 25.
        let schedule = prefix_schedule(76);
        assert_eq!(schedule.len(), 31);
        assert_eq!(
            schedule[..26],
            [
                plan(&[(0, 0..3)], false),
                plan(&[(0, 3..6)], false),
                plan(&[(0, 6..9)], false),
                plan(&[(0, 9..12)], false),
                plan(&[(0, 12..15)], false),
                plan(&[(0, 15..18)], false),
                plan(&[(0, 18..21)], false),
                plan(&[(0, 21..24)], false),
                plan(&[(0, 24..27)], false),
                plan(&[(0, 27..30)], false),
                plan(&[(0, 30..32), (32, 32..33)], false),
                plan(&[(32, 33..36)], false),
                plan(&[(32, 36..39)], false),
                plan(&[(32, 39..42)], false),
                plan(&[(32, 42..45)], false),
                plan(&[(32, 45..48)], false),
                plan(&[(32, 48..51)], false),
                plan(&[(32, 51..54)], false),
                plan(&[(32, 54..57)], false),
                plan(&[(32, 57..60)], false),
                plan(&[(32, 60..63)], false),
                plan(&[(32, 63..64), (64, 64..66)], false),
                plan(&[(64, 66..69)], false),
                plan(&[(64, 69..72)], false),
                plan(&[(64, 72..75)], false),
                plan(&[], true),
            ]
        );
        assert_eq!(schedule[26..], vec![plan(&[], false); 5]);
    }

    #[test]
    fn prefix_schedule_4096_uses_42_splits_and_covers_each_prefix_slot_once() {
        let schedule = prefix_schedule(4096);
        assert_eq!(schedule.len(), 42);
        assert_eq!(
            schedule[0],
            plan(
                &[(0, 0..32), (32, 32..64), (64, 64..96), (96, 96..98)],
                false
            )
        );
        assert_eq!(
            schedule[41],
            plan(
                &[(4000, 4018..4032), (4032, 4032..4064), (4064, 4064..4095)],
                true
            )
        );
        let mut visits = [0; 4095];
        for split in &schedule {
            for (tile_start, visible) in &split.tiles {
                assert_eq!(tile_start % 32, 0);
                assert!(!visible.is_empty());
                assert!(visible.start >= *tile_start && visible.end <= tile_start + 32);
                for slot in visible.clone() {
                    visits[slot] += 1;
                }
            }
        }
        assert_eq!(visits, [1; 4095]);
        assert_eq!(
            schedule.iter().filter(|split| split.include_self).count(),
            1
        );
    }

    #[test]
    fn length_76_attention_pins_absolute_tile_output_bits() {
        let layer = layer();
        let query: Vec<f32> = (0..256)
            .map(|channel| ((channel * 7 % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let rows: Vec<CacheToken> = (0..76)
            .map(|position| CacheToken {
                keys: vec![
                    CacheRow::Int8(Int8Row {
                        integers: (0..256)
                            .map(
                                |channel| (((position * 13 + channel * 7) % 255) as i16 - 127)
                                    as i8
                            )
                            .collect(),
                        scale: (position % 7 + 1) as f32 * 0.0078125,
                    });
                    2
                ],
                values: vec![
                    CacheRow::Int8(Int8Row {
                        integers: (0..256)
                            .map(
                                |channel| (((position * 19 + (channel % 16) * 11) % 255) as i16
                                    - 127) as i8
                            )
                            .collect(),
                        scale: (position % 11 + 1) as f32 * 0.015625,
                    });
                    2
                ],
            })
            .collect();
        let output = layer
            .attend_head(&query, 0, &rows[..75], &rows[75])
            .unwrap();
        // Captured from the pre-extraction D9 implementation; values repeat every 16 channels.
        let expected = [
            0xbede_440b,
            0xbeac_aee9,
            0x3ce1_ab7d,
            0xbdb7_14de,
            0xbd93_8526,
            0x3ef5_474a,
            0x3e3f_be6c,
            0x3f4c_2f7e,
            0x3f26_e245,
            0x3f81_1d5b,
            0x3f09_e0ce,
            0x3f0e_8ea2,
            0xbde6_e0e7,
            0xbeb1_494b,
            0xbccd_e3de,
            0xbf2b_79d6,
        ];
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected.repeat(16)
        );
    }

    #[test]
    fn int8_encoding_handles_zero_and_halfway_ties() {
        let CacheRow::Int8(zero) = encode_int8(&[0.0; 256]).unwrap() else {
            panic!("int8");
        };
        assert_eq!(zero.scale.to_bits(), 0.0_f32.to_bits());
        assert!(zero.integers.iter().all(|&value| value == 0));
        let CacheRow::Int8(encoded) =
            encode_int8(&[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]).unwrap()
        else {
            panic!("int8");
        };
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
                    CacheRow::Int8(Int8Row {
                        integers: vec![2; 256],
                        scale: 0.25
                    });
                    2
                ],
                values: vec![
                    CacheRow::Int8(Int8Row {
                        integers: vec![-3; 256],
                        scale: 0.125
                    });
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
                    if let CacheRow::Int8(row) = row {
                        row.scale = f32::NAN;
                        row.integers.fill(-127);
                    }
                }
            }
        }
        assert_eq!(evaluate(&prefix), expected);
        // Neither future slots nor padding beyond the causal cache need to exist.
        prefix.truncate(visible.end);
        assert_eq!(evaluate(&prefix), expected);

        // Substituting +0 operands for masked slots would erase these negative zeros.
        for token in &mut prefix[visible.clone()] {
            if let CacheRow::Int8(row) = &mut token.keys[0] {
                row.integers.fill(0);
            }
            if let CacheRow::Int8(row) = &mut token.values[0] {
                row.scale = f32::from_bits(32);
                row.integers.fill(-127);
            }
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
