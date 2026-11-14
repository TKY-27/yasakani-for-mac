//! D3 SIMD across keys for scores, across channels for values. No persistent cache change.
use crate::Result;
use yskn_ref::{
    Error,
    attention::{Attention, AttentionOperands, AttentionState, PreparedRow},
    kv_codec::CacheRow,
    portable_math::{bfloat, exp, exp2},
};
const LOG2_E: f32 = std::f32::consts::LOG2_E;
const EMPTY_MAXIMUM: f32 = -1.0e30;

pub(crate) fn output_head(
    layer: &Attention,
    state: &AttentionState,
    position: usize,
    prepared: &PreparedRow,
    head: usize,
) -> Result<Vec<f32>> {
    if super::enabled() {
        layer.output_head_with_math::<super::math::FastMath>(
            state,
            position,
            prepared,
            head,
            attend_head,
        )
    } else {
        layer.output_head(state, position, prepared, head)
    }
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

fn attend_head(operands: AttentionOperands<'_>) -> Result<Vec<f32>> {
    let AttentionOperands {
        query,
        kv_head,
        prefix,
        current,
        value_mean: mean,
    } = operands;
    let length = prefix
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidModel("attention context length overflow".into()))?;
    let schedule = prefix_schedule(length);
    let padded_count = schedule.len().next_power_of_two();
    let mut splits = Vec::with_capacity(padded_count);
    for plan in schedule {
        let mut split = AttentionSplit {
            maximum: EMPTY_MAXIMUM,
            denominator: 0.0,
            numerator: vec![0.0; 256],
        };
        for (tile_start, visible) in plan.tiles {
            prefix_tile(&mut split, query, tile_start, visible, |position, side| {
                let row = if side == 0 {
                    prefix[position].key(kv_head)
                } else {
                    prefix[position].value(kv_head)
                };
                (row.bases(), row.scale())
            })?;
        }
        if plan.include_self {
            add_self(
                &mut split,
                query,
                current.key(kv_head),
                current.value(kv_head),
            )?;
        }
        add_value_mean(&mut split, mean)?;
        splits.push(split);
    }
    splits.resize_with(padded_count, || AttentionSplit {
        maximum: EMPTY_MAXIMUM,
        denominator: 0.0,
        numerator: vec![0.0; 256],
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
    let mut attended = vec![0.0; 256];
    for (scale, split) in scales.iter().zip(&splits) {
        add_scaled(&mut attended, &split.numerator, *scale);
    }
    for value in &mut attended {
        *value /= denominator;
    }
    finite(&attended, "attention combined output")?;
    Ok(attended)
}

fn prefix_tile(
    split: &mut AttentionSplit,
    query: &[f32],
    tile_start: usize,
    visible: std::ops::Range<usize>,
    row: impl Fn(usize, usize) -> ([f32; 256], f32),
) -> Result<()> {
    let mut scores = [f32::NEG_INFINITY; 32];
    for first in (visible.start..visible.end).step_by(4) {
        let count = (visible.end - first).min(4);
        let mut bases = [[0.0; 256]; 4];
        let mut scales = [0.0; 4];
        for lane in 0..count {
            (bases[lane], scales[lane]) = row(first + lane, 0);
        }
        let dots = dots4(query, &bases);
        for lane in 0..count {
            let score = dots[lane] * (scales[lane] * 0.0625);
            finite(&[score], "attention prefix score")?;
            scores[first + lane - tile_start] = score;
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
    // Only visible rows are read. Each channel still consumes ascending tile keys.
    for position in visible {
        let (bases, scale) = row(position, 1);
        let weight = bfloat(probabilities[position - tile_start] * scale);
        add_scaled(&mut split.numerator, &bases, weight);
    }
    finite(&[split.denominator], "attention prefix denominator")?;
    finite(&split.numerator, "attention prefix numerator")
}

fn add_self(
    split: &mut AttentionSplit,
    query: &[f32],
    key: &CacheRow,
    value: &CacheRow,
) -> Result<()> {
    let key_bases = key.bases();
    let value_bases = value.bases();
    let mut partial = [0.0; 4];
    for (lane, sum) in partial.iter_mut().enumerate() {
        for chunk in 0..256 / 16 {
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

fn dots4(query: &[f32], bases: &[[f32; 256]; 4]) -> [f32; 4] {
    assert_eq!(query.len(), 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: output_head checks enabled() (runtime NEON detection); direct kernel
        // tests check neon_available() before calling.
        // query and every base row have exactly 256 channels.
        unsafe { dots4_neon(query, bases) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        std::array::from_fn(|lane| {
            let mut sum = 0.0;
            for i in 0..256 {
                sum += query[i] * bases[lane][i];
            }
            sum
        })
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dots4_neon(query: &[f32], bases: &[[f32; 256]; 4]) -> [f32; 4] {
    use std::arch::aarch64::*;
    // Transpose four rows into channel-major scratch. Each key's chain keeps D3 order.
    let transposed: [[f32; 4]; 256] =
        std::array::from_fn(|i| std::array::from_fn(|lane| bases[lane][i]));
    let mut out = [0.0; 4];
    // SAFETY: each transposed row and out hold four f32 values, all indices are in range.
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        for i in 0..256 {
            sum = vaddq_f32(
                sum,
                vmulq_n_f32(vld1q_f32(transposed[i].as_ptr()), query[i]),
            );
        }
        vst1q_f32(out.as_mut_ptr(), sum);
    }
    out
}

fn add_scaled(output: &mut [f32], bases: &[f32], weight: f32) {
    assert_eq!(output.len(), 256);
    assert_eq!(bases.len(), 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: output_head checks enabled() (runtime NEON detection); direct kernel
        // tests check neon_available(). Both checked slices have 256 elements.
        unsafe {
            add_scaled_neon(output, bases, weight);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (out, base) in output.iter_mut().zip(bases) {
        *out += weight * base;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_scaled_neon(output: &mut [f32], bases: &[f32], weight: f32) {
    use std::arch::aarch64::*;
    // SAFETY: both slices contain 256 elements; each load/store covers i..i+4.
    unsafe {
        let output_ptr = output.as_mut_ptr();
        for i in (0..256).step_by(4) {
            let sum = vaddq_f32(
                vld1q_f32(output_ptr.add(i)),
                vmulq_n_f32(vld1q_f32(bases.as_ptr().add(i)), weight),
            );
            vst1q_f32(output_ptr.add(i), sum);
        }
    }
}

#[cfg(test)]
use crate::model::session_test_support::kv_profile as profile_fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use yskn_ref::{KvMode, attention::AttentionDimensions, kv_profile::KvProfile};
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn attention_matches_reference_at_tile_and_split_boundaries_with_mixed_sides() {
        if !super::super::neon_available() {
            return;
        }
        let profile = Arc::new(KvProfile::parse(&profile_fixture::mixed_profile()).unwrap());
        let mut seed = 7919u32;
        let mut sample = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            match seed % 16 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1),
                3 => f32::from_bits(0x80000001),
                4 => f32::from_bits(0x3f808000),
                _ => seed as i32 as f32 / i32::MAX as f32 * 3.0,
            }
        };
        for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
            for ordinal in 0..2 {
                let layer = Attention::new(
                    AttentionDimensions {
                        query_heads: 4,
                        kv_heads: 2,
                        head_dim: 256,
                        rotary_dim: 64,
                    },
                    vec![0.0; 256],
                    vec![0.0; 256],
                    1e-6,
                )
                .unwrap()
                .with_kv(mode, Some(profile.clone()), ordinal)
                .unwrap();
                let mut state = layer.new_state().unwrap();
                for position in 0..4107 {
                    let q: Vec<_> = (0..2048).map(|_| sample()).collect();
                    let k: Vec<_> = (0..512).map(|_| sample()).collect();
                    let v: Vec<_> = (0..512).map(|_| sample()).collect();
                    let row = layer.prepare_row(position, &q, &k, &v).unwrap();
                    if [
                        0, 1, 30, 31, 32, 33, 61, 62, 63, 64, 2000, 4094, 4095, 4096, 4106,
                    ]
                    .contains(&position)
                    {
                        for head in 0..4 {
                            let expected = layer.output_head(&state, position, &row, head).unwrap();
                            let actual = layer
                                .output_head_with(&state, position, &row, head, attend_head)
                                .unwrap();
                            assert_eq!(
                                bits(&actual),
                                bits(&expected),
                                "{mode:?} ordinal={ordinal} position={position} head={head}"
                            );
                        }
                    }
                    layer.append_prepared(&mut state, position, &row).unwrap();
                    // Future/self cache contents must not participate as prefix operands.
                    if position == 32 {
                        assert_eq!(
                            bits(
                                &layer
                                    .output_head_with(&state, position, &row, 0, attend_head)
                                    .unwrap()
                            ),
                            bits(&layer.output_head(&state, position, &row, 0).unwrap())
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn masked_prefix_slots_preserve_negative_zero_on_vector_path() {
        if !super::super::neon_available() {
            return;
        }
        // A split can begin or end inside an absolute 32-slot tile.
        for visible in [35..64, 32..38] {
            for poisoned in [false, true] {
                // Key/value operands are read lazily, just as they are from CacheToken.
                let mut rows = vec![([127.0; 256], 0.125); 128];
                for position in 0..64 {
                    if visible.contains(&position) {
                        rows[2 * position] = ([0.0; 256], 0.25);
                        rows[2 * position + 1] = ([-127.0; 256], f32::from_bits(32));
                    } else if poisoned {
                        rows[2 * position] = ([-128.0; 256], f32::NAN);
                        rows[2 * position + 1] = ([127.0; 256], f32::NAN);
                    }
                }
                // Future padding need not even exist; masked scales must never be read.
                for rows in [&rows[..], &rows[..2 * visible.end]] {
                    let mut split = AttentionSplit {
                        maximum: 0.0,
                        denominator: 0.0,
                        numerator: vec![-0.0; 256],
                    };
                    prefix_tile(
                        &mut split,
                        &[0.0; 256],
                        32,
                        visible.clone(),
                        |position, side| rows[2 * position + side],
                    )
                    .unwrap();
                    assert_eq!(split.maximum.to_bits(), 0.0_f32.to_bits());
                    assert_eq!(
                        split.denominator.to_bits(),
                        (visible.len() as f32).to_bits()
                    );
                    assert_eq!(
                        bits(&split.numerator),
                        vec![(-0.0_f32).to_bits(); 256],
                        "visible={visible:?} poisoned={poisoned}"
                    );
                }
            }
        }
    }

    #[test]
    fn score_lanes_preserve_the_full_ascending_chain_for_extreme_int8_and_signed_zero() {
        if !super::super::neon_available() {
            return;
        }
        let query: Vec<_> = (0..256)
            .map(|i| match i % 5 {
                0 => -0.0,
                1 => f32::from_bits(1),
                2 => f32::from_bits(0x3f808000),
                _ => i as f32 / 32.0 - 4.0,
            })
            .collect();
        let bases = std::array::from_fn(|lane| {
            std::array::from_fn(|i| ((i + lane * 31) as u8 as i8) as f32)
        });
        let actual = dots4(&query, &bases);
        for lane in 0..4 {
            let mut expected = 0.0;
            for i in 0..256 {
                expected += query[i] * bases[lane][i];
            }
            assert_eq!(actual[lane].to_bits(), expected.to_bits());
        }
    }
}
