//! Share widening across every query/row, with disjoint split-range workers.
use super::*;
use crate::model::Compute;
use std::ops::Range;
use yskn_ref::attention::CacheToken;

const PARTS: usize = 8;

struct Tile {
    keys: [[f32; 32]; 256],
    values: [[f32; 256]; 32],
    key_scales: [f32; 32],
    value_scales: [f32; 32],
}
impl Tile {
    fn load(prefix: &[CacheToken], head: usize, start: usize) -> Box<Self> {
        let mut tile = Box::new(Self {
            keys: [[0.0; 32]; 256],
            values: [[0.0; 256]; 32],
            key_scales: [0.0; 32],
            value_scales: [0.0; 32],
        });
        for (lane, row) in prefix[start..prefix.len().min(start + 32)]
            .iter()
            .enumerate()
        {
            let key = row.key(head);
            for (column, base) in tile.keys.iter_mut().zip(key.bases()) {
                column[lane] = base;
            }
            let value = row.value(head);
            tile.values[lane] = value.bases();
            tile.key_scales[lane] = key.scale();
            tile.value_scales[lane] = value.scale();
        }
        tile
    }
}

#[derive(Clone, Copy)]
struct Schedule {
    count: usize,
    width: usize,
    prefix: usize,
}
impl Schedule {
    fn new(prefix: usize) -> Self {
        let length = prefix + 1;
        let count = if length < 4096 { 31 } else { 42 };
        Self {
            count,
            width: length.div_ceil(count),
            prefix,
        }
    }
    fn splits(self, part: usize) -> Range<usize> {
        self.count * part / PARTS..self.count * (part + 1) / PARTS
    }
    fn tiles(self, part: usize) -> Range<usize> {
        let splits = self.splits(part);
        let first = (splits.start * self.width).min(self.prefix);
        let last = (splits.end * self.width).min(self.prefix);
        if first == last {
            0..0
        } else {
            first / 32..last.div_ceil(32)
        }
    }
}

struct Head<'a> {
    queries: Vec<usize>,
    prefix: &'a [CacheToken],
    ranges: [Range<usize>; PARTS],
    shared: Vec<Option<Box<Tile>>>,
}

pub(super) struct Batch<'a> {
    operands: Vec<AttentionOperands<'a>>,
    schedules: Vec<Schedule>,
    heads: Vec<Head<'a>>,
}

type Partial = Vec<(usize, usize, AttentionSplit)>;

impl<'a> Batch<'a> {
    pub(super) fn new(operands: Vec<AttentionOperands<'a>>, kv_heads: usize) -> Self {
        let schedules: Vec<_> = operands
            .iter()
            .map(|q| Schedule::new(q.prefix.len()))
            .collect();
        let heads = (0..kv_heads)
            .map(|head| {
                let queries: Vec<_> = (0..operands.len())
                    .filter(|&i| operands[i].kv_head == head)
                    .collect();
                let prefix = queries
                    .iter()
                    .map(|&i| operands[i].prefix)
                    .max_by_key(|p| p.len())
                    .unwrap_or(&[]);
                let ranges = std::array::from_fn(|part| {
                    let ranges: Vec<_> = queries
                        .iter()
                        .map(|&i| schedules[i].tiles(part))
                        .filter(|r| !r.is_empty())
                        .collect();
                    ranges.iter().map(|r| r.start).min().unwrap_or(0)
                        ..ranges.iter().map(|r| r.end).max().unwrap_or(0)
                });
                // Only partition-boundary tiles have multiple owners. Decode those once;
                // workers share the immutable temporary copy, never reread the cache.
                let shared = (0..prefix.len().div_ceil(32))
                    .map(|tile| {
                        if ranges.iter().filter(|r| r.contains(&tile)).count() > 1 {
                            Some(Tile::load(prefix, head, tile * 32))
                        } else {
                            None
                        }
                    })
                    .collect();
                Head {
                    queries,
                    prefix,
                    ranges,
                    shared,
                }
            })
            .collect();
        Self {
            operands,
            schedules,
            heads,
        }
    }
    pub(super) fn tasks(&self) -> usize {
        self.heads.len() * PARTS
    }

    pub(super) fn run(&self, task: usize) -> Result<Partial> {
        let head_index = task / PARTS;
        let part = task % PARTS;
        let head = &self.heads[head_index];
        let mut states: Vec<_> = head
            .queries
            .iter()
            .map(|&q| {
                let splits = self.schedules[q].splits(part);
                (
                    q,
                    splits.clone(),
                    splits.map(|_| empty_split()).collect::<Vec<_>>(),
                )
            })
            .collect();
        for tile_index in head.ranges[part].clone() {
            let owned;
            let tile = if let Some(tile) = &head.shared[tile_index] {
                tile.as_ref()
            } else {
                owned = Tile::load(head.prefix, head_index, tile_index * 32);
                &owned
            };
            let tile_start = tile_index * 32;
            for (q, splits, states) in &mut states {
                let schedule = self.schedules[*q];
                let end = (tile_start + 32).min(schedule.prefix);
                if tile_start >= end {
                    continue;
                }
                let first = (tile_start / schedule.width).max(splits.start);
                let last = end.div_ceil(schedule.width).min(splits.end);
                for index in first..last {
                    let visible = (index * schedule.width).max(tile_start)
                        ..((index + 1) * schedule.width).min(end);
                    grouped_tile(
                        &mut states[index - splits.start],
                        self.operands[*q].query,
                        tile,
                        tile_start,
                        visible,
                    )?;
                }
            }
        }
        let mut result = Vec::new();
        for (q, indices, states) in states {
            let op = &self.operands[q];
            let schedule = self.schedules[q];
            for (index, mut split) in indices.zip(states) {
                if schedule.prefix / schedule.width == index {
                    add_self(
                        &mut split,
                        op.query,
                        op.current.key(head_index),
                        op.current.value(head_index),
                    )?;
                }
                add_value_mean(&mut split, op.value_mean)?;
                result.push((q, index, split));
            }
        }
        Ok(result)
    }
    pub(super) fn collect(&self, results: Vec<Partial>) -> Vec<Vec<AttentionSplit>> {
        let mut splits: Vec<Vec<Option<AttentionSplit>>> = self
            .schedules
            .iter()
            .map(|s| (0..s.count.next_power_of_two()).map(|_| None).collect())
            .collect();
        for result in results {
            for (q, index, split) in result {
                splits[q][index] = Some(split);
            }
        }
        splits
            .into_iter()
            .map(|splits| {
                splits
                    .into_iter()
                    .map(|s| s.unwrap_or_else(empty_split))
                    .collect()
            })
            .collect()
    }
    #[cfg(test)]
    pub(super) fn finish(&self, results: Vec<Partial>) -> Result<Vec<Vec<f32>>> {
        self.collect(results)
            .iter()
            .map(|splits| combine_splits(splits))
            .collect()
    }
}

fn empty_split() -> AttentionSplit {
    AttentionSplit {
        maximum: EMPTY_MAXIMUM,
        denominator: 0.0,
        numerator: vec![0.0; 256],
    }
}

fn grouped_tile(
    split: &mut AttentionSplit,
    query: &[f32],
    tile: &Tile,
    start: usize,
    visible: Range<usize>,
) -> Result<()> {
    let lanes = visible.start - start..visible.end - start;
    let dots = dots(query, &tile.keys, lanes.clone());
    let mut scores = [f32::NEG_INFINITY; 32];
    for lane in lanes.clone() {
        let score = dots[lane] * (tile.key_scales[lane] * 0.0625);
        finite(&[score], "attention prefix score")?;
        scores[lane] = score;
    }
    let probabilities = prefix_probabilities(split, &scores)?;
    let mut weights = [0.0; 32];
    for lane in lanes.clone() {
        weights[lane] = bfloat(probabilities[lane] * tile.value_scales[lane]);
    }
    add_values(&mut split.numerator, &tile.values, &weights, lanes);
    finite(&[split.denominator], "attention prefix denominator")?;
    finite(&split.numerator, "attention prefix numerator")
}

fn add_values(
    output: &mut [f32],
    values: &[[f32; 256]; 32],
    weights: &[f32; 32],
    visible: Range<usize>,
) {
    assert_eq!(output.len(), 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers require NEON; shapes are fixed and output has 256 elements.
        unsafe {
            add_values_neon(output, values, weights, visible);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (channel, out) in output.iter_mut().enumerate() {
        for lane in visible.clone() {
            *out += weights[lane] * values[lane][channel];
        }
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_values_neon(
    output: &mut [f32],
    values: &[[f32; 256]; 32],
    weights: &[f32; 32],
    visible: Range<usize>,
) {
    use std::arch::aarch64::*;
    // SAFETY: each block covers 32 of 256 independent output channels. All
    // mutable reads/writes use this one exclusive pointer, never a shared alias.
    unsafe {
        let ptr = output.as_mut_ptr();
        for channel in (0..256).step_by(32) {
            let mut sums: [_; 8] =
                std::array::from_fn(|group| vld1q_f32(ptr.add(channel + group * 4)));
            // Keep channel accumulators in registers for the complete key chain.
            for lane in visible.clone() {
                for (group, sum) in sums.iter_mut().enumerate() {
                    *sum = vaddq_f32(
                        *sum,
                        vmulq_n_f32(
                            vld1q_f32(values[lane].as_ptr().add(channel + group * 4)),
                            weights[lane],
                        ),
                    );
                }
            }
            for (group, sum) in sums.into_iter().enumerate() {
                vst1q_f32(ptr.add(channel + group * 4), sum);
            }
        }
    }
}

fn dots(query: &[f32], keys: &[[f32; 32]; 256], visible: Range<usize>) -> [f32; 32] {
    assert_eq!(query.len(), 256);
    let mut output = [0.0; 32];
    #[cfg(target_arch = "aarch64")]
    {
        let first = visible.start.div_ceil(4) * 4;
        let last = visible.end / 4 * 4;
        // SAFETY: runtime NEON is required; each specialization accesses only
        // complete four-key groups inside the visible range and 256 channels.
        unsafe {
            match (last.saturating_sub(first)) / 4 {
                1 => dots_neon::<1>(query, keys, first, &mut output),
                2 => dots_neon::<2>(query, keys, first, &mut output),
                3 => dots_neon::<3>(query, keys, first, &mut output),
                4 => dots_neon::<4>(query, keys, first, &mut output),
                5 => dots_neon::<5>(query, keys, first, &mut output),
                6 => dots_neon::<6>(query, keys, first, &mut output),
                7 => dots_neon::<7>(query, keys, first, &mut output),
                8 => dots_neon::<8>(query, keys, first, &mut output),
                _ => {}
            }
        }
        // Scalar edge keys skip masked lanes altogether, including their arithmetic.
        for lane in visible.filter(|&lane| lane < first || lane >= last) {
            for channel in 0..256 {
                output[lane] += query[channel] * keys[channel][lane];
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for lane in visible {
        for channel in 0..256 {
            output[lane] += query[channel] * keys[channel][lane];
        }
    }
    output
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dots_neon<const GROUPS: usize>(
    query: &[f32],
    keys: &[[f32; 32]; 256],
    first: usize,
    output: &mut [f32; 32],
) {
    use std::arch::aarch64::*;
    let mut sums = [vdupq_n_f32(0.0); GROUPS];
    // SAFETY: caller bounds first + 4*GROUPS by 32; each load is four
    // independent keys. Output stores derive from one exclusive mutable pointer.
    unsafe {
        for channel in 0..256 {
            for (group, sum) in sums.iter_mut().enumerate() {
                *sum = vaddq_f32(
                    *sum,
                    vmulq_n_f32(
                        vld1q_f32(keys[channel].as_ptr().add(first + group * 4)),
                        query[channel],
                    ),
                );
            }
        }
        let ptr = output.as_mut_ptr();
        for (group, sum) in sums.into_iter().enumerate() {
            vst1q_f32(ptr.add(first + group * 4), sum);
        }
    }
}

pub(crate) fn output_rows(
    engine: &mut Compute,
    layer: &Attention,
    state: &AttentionState,
    position: usize,
    prepared: &[PreparedRow],
) -> Result<Vec<Vec<f32>>> {
    let heads = layer.query_heads();
    if !super::super::enabled() {
        return engine.map_range(0..prepared.len() * heads, |i| {
            layer.output_head(state, position + i / heads, &prepared[i / heads], i % heads)
        });
    }
    let attempt = (|| {
        let operands = (0..prepared.len() * heads)
            .map(|i| {
                layer.head_operands(state, position + i / heads, &prepared[i / heads], i % heads)
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = Batch::new(operands, layer.kv_heads());
        let results = engine.map_range(0..batch.tasks(), |task| batch.run(task))?;
        let splits = batch.collect(results);
        engine.map_range(0..splits.len(), |i| {
            layer.output_head_with_math::<super::super::math::FastMath>(
                state,
                position + i / heads,
                &prepared[i / heads],
                i % heads,
                |_| combine_splits(&splits[i]),
            )
        })
    })();
    match attempt {
        Ok(outputs) => Ok(outputs),
        Err(error) => {
            // Task order is not scalar row/head/stage order. On failure replay
            // read-only heads, preserving the first error without touching state.
            for (row, prepared) in prepared.iter().enumerate() {
                for head in 0..heads {
                    layer.output_head(state, position + row, prepared, head)?;
                }
            }
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_numeric_failure_among_good_rows_matches_scalar_and_keeps_state() {
        use yskn_ref::attention::AttentionDimensions;
        let layer = Attention::new(
            AttentionDimensions {
                query_heads: 12,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
            },
            vec![1e37; 256],
            vec![1e37; 256],
            1e-6,
        )
        .unwrap();
        let good = vec![0.0; 12 * 512];
        let keys = vec![1e-20; 512];
        let values = vec![0.25; 512];
        let mut state = layer.new_state().unwrap();
        let first = layer.prepare_row(0, &good, &keys, &values).unwrap();
        layer.output_row(&state, 0, &first).unwrap();
        layer.append_prepared(&mut state, 0, &first).unwrap();
        let mut prepared = Vec::new();
        for row in 0..3 {
            let mut query = good.clone();
            if row == 1 {
                query[6 * 512..6 * 512 + 256].fill(1e-20);
            }
            // All preparation succeeds: overflow happens in the attention score.
            let p = layer.prepare_row(row + 1, &query, &keys, &values).unwrap();
            layer.append_prepared(&mut state, row + 1, &p).unwrap();
            prepared.push(p);
        }
        layer.output_row(&state, 1, &prepared[0]).unwrap();
        layer.output_row(&state, 3, &prepared[2]).unwrap();
        let expected = layer.output_row(&state, 2, &prepared[1]).unwrap_err();
        assert!(matches!(
            expected,
            Error::NonFinite("attention prefix score")
        ));
        let before = state.clone();
        for threads in [1, 4, 10] {
            let mut engine = Compute::new(
                crate::Engine::new(crate::Kernel::Portable, threads).unwrap(),
                crate::Backend::Cpu,
            )
            .unwrap();
            let actual = output_rows(&mut engine, &layer, &state, 1, &prepared).unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert_eq!(state, before);
        }
    }

    #[test]
    fn grouped_masks_skip_poisoned_scales_and_preserve_negative_zero() {
        if !super::super::super::neon_available() {
            return;
        }
        for visible in [3..32, 0..6, 1..2, 0..32] {
            let mut tile = Tile {
                keys: [[-128.0; 32]; 256],
                values: [[127.0; 256]; 32],
                key_scales: [f32::NAN; 32],
                value_scales: [f32::NAN; 32],
            };
            for lane in visible.clone() {
                for column in &mut tile.keys {
                    column[lane] = 0.0;
                }
                tile.values[lane] = [-127.0; 256];
                tile.key_scales[lane] = 0.25;
                tile.value_scales[lane] = f32::from_bits(32);
            }
            let mut split = AttentionSplit {
                maximum: 0.0,
                denominator: 0.0,
                numerator: vec![-0.0; 256],
            };
            grouped_tile(
                &mut split,
                &[0.0; 256],
                &tile,
                32,
                32 + visible.start..32 + visible.end,
            )
            .unwrap();
            assert_eq!(split.maximum.to_bits(), 0.0_f32.to_bits());
            assert_eq!(
                split.denominator.to_bits(),
                (visible.len() as f32).to_bits()
            );
            assert!(
                split
                    .numerator
                    .iter()
                    .all(|n| n.to_bits() == (-0.0_f32).to_bits())
            );
        }
    }

    #[test]
    fn grouped_scores_keep_ascending_products_for_extreme_operands() {
        if !super::super::super::neon_available() {
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
        let keys = std::array::from_fn(|i| {
            std::array::from_fn(|lane| ((i + lane * 31) as u8 as i8) as f32)
        });
        for visible in [0..1, 1..4, 3..8, 5..16, 7..17, 0..31, 0..32] {
            let actual = dots(&query, &keys, visible.clone());
            for lane in visible {
                let mut expected = 0.0;
                for channel in 0..256 {
                    expected += query[channel] * keys[channel][lane];
                }
                assert_eq!(actual[lane].to_bits(), expected.to_bits());
            }
        }
    }
}
