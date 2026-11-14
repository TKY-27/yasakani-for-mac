//! Share compact tiles across queries; split-range tasks retain scalar reduction order.
use super::*;
use crate::model::Compute;
use std::{ops::Range, sync::OnceLock};
use yskn_ref::attention::CacheToken;

const PARTS: usize = 32;

// Int8 integers are exactly representable in BF16, as are packed bases.
// Widen into registers at consumption, keeping the reusable tile near 32 KiB.
struct Tile {
    keys: [[u16; 32]; 256],
    values: [[u16; 256]; 32],
    key_scales: [f32; 32],
    value_scales: [f32; 32],
}
impl Tile {
    fn empty() -> Box<Self> {
        Box::new(Self {
            keys: [[0; 32]; 256],
            values: [[0; 256]; 32],
            key_scales: [0.0; 32],
            value_scales: [0.0; 32],
        })
    }
    fn load(prefix: &[CacheToken], head: usize, start: usize) -> Box<Self> {
        let mut tile = Self::empty();
        tile.fill(prefix, head, start);
        tile
    }
    fn fill(&mut self, prefix: &[CacheToken], head: usize, start: usize) {
        let tile = self;
        let rows = &prefix[start..prefix.len().min(start + 32)];
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: fast attention requires NEON; rows has at most 32 entries,
            // each CacheRow exposes exactly 256 bases. The tile is exclusively owned.
            unsafe {
                use yskn_ref::kv_codec::BaseView;
                // Resolve the row pointers together before consuming their bases.
                // Each hint stays within the bases of a live cache row.
                for row in rows {
                    for view in [row.key(head).base_view(), row.value(head).base_view()] {
                        let (ptr, bytes) = match view {
                            BaseView::Bfloat16(base) => (base.as_ptr().cast::<u8>(), 512),
                            BaseView::Int8(base) => (base.as_ptr().cast::<u8>(), 256),
                        };
                        for offset in (0..bytes).step_by(128) {
                            std::arch::asm!("prfm pldl1keep, [{address}]", address = in(reg) ptr.add(offset), options(readonly, nostack, preserves_flags));
                        }
                    }
                }
                tile.load_keys_neon(rows, head)
            };
        }
        for (lane, row) in rows.iter().enumerate() {
            let key = row.key(head);
            #[cfg(not(target_arch = "aarch64"))]
            for (column, base) in tile.keys.iter_mut().zip(key.bases()) {
                column[lane] = (base.to_bits() >> 16) as u16;
            }
            let value = row.value(head);
            match value.base_view() {
                yskn_ref::kv_codec::BaseView::Bfloat16(base) => {
                    tile.values[lane].copy_from_slice(base)
                }
                yskn_ref::kv_codec::BaseView::Int8(base) => {
                    for (out, &value) in tile.values[lane].iter_mut().zip(base) {
                        *out = ((value as f32).to_bits() >> 16) as u16;
                    }
                }
            }
            tile.key_scales[lane] = key.scale();
            tile.value_scales[lane] = value.scale();
        }
    }
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn load_keys_neon(&mut self, rows: &[CacheToken], head: usize) {
        use std::arch::aarch64::*;
        use yskn_ref::kv_codec::BaseView;
        // SAFETY: every input view contains 256 bases. Each four-channel load
        // stays in bounds; all stores derive from this one mutable tile pointer.
        unsafe {
            let output = self.keys.as_mut_ptr().cast::<u16>();
            for (group, rows) in rows.as_chunks::<4>().0.iter().enumerate() {
                let views: [_; 4] = std::array::from_fn(|i| rows[i].key(head).base_view());
                for channel in (0..256).step_by(4) {
                    let bases: [_; 4] = std::array::from_fn(|i| match views[i] {
                        BaseView::Bfloat16(base) => vld1_u16(base.as_ptr().add(channel)),
                        BaseView::Int8(base) => {
                            let bytes =
                                std::ptr::read_unaligned(base.as_ptr().add(channel).cast::<i32>());
                            let floats = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(
                                vreinterpret_s8_s32(vdup_n_s32(bytes)),
                            ))));
                            vshrn_n_u32::<16>(vreinterpretq_u32_f32(floats))
                        }
                    });
                    let ab0 = vtrn1_u16(bases[0], bases[1]);
                    let ab1 = vtrn2_u16(bases[0], bases[1]);
                    let cd0 = vtrn1_u16(bases[2], bases[3]);
                    let cd1 = vtrn2_u16(bases[2], bases[3]);
                    let columns = [
                        vtrn1_u32(vreinterpret_u32_u16(ab0), vreinterpret_u32_u16(cd0)),
                        vtrn1_u32(vreinterpret_u32_u16(ab1), vreinterpret_u32_u16(cd1)),
                        vtrn2_u32(vreinterpret_u32_u16(ab0), vreinterpret_u32_u16(cd0)),
                        vtrn2_u32(vreinterpret_u32_u16(ab1), vreinterpret_u32_u16(cd1)),
                    ];
                    for (i, column) in columns.into_iter().enumerate() {
                        vst1_u16(
                            output.add((channel + i) * 32 + group * 4),
                            vreinterpret_u16_u32(column),
                        );
                    }
                }
            }
            for (lane, row) in rows.iter().enumerate().skip(rows.len() / 4 * 4) {
                for (channel, base) in row.key(head).bases().into_iter().enumerate() {
                    *output.add(channel * 32 + lane) = (base.to_bits() >> 16) as u16;
                }
            }
        }
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
    shared: Vec<Option<OnceLock<Box<Tile>>>>,
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
                // Partition-boundary tiles have multiple owners. The first worker
                // loads each once; other owners borrow its immutable temporary copy.
                let shared = (0..prefix.len().div_ceil(32))
                    .map(|tile| {
                        if ranges.iter().filter(|r| r.contains(&tile)).count() > 1 {
                            Some(OnceLock::new())
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
        let mut owned = None;
        for tile_index in head.ranges[part].clone() {
            let tile = if let Some(tile) = &head.shared[tile_index] {
                tile.get_or_init(|| Tile::load(head.prefix, head_index, tile_index * 32))
                    .as_ref()
            } else {
                let tile = owned.get_or_insert_with(Tile::empty);
                tile.fill(head.prefix, head_index, tile_index * 32);
                &**tile
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
    values: &[[u16; 256]; 32],
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
            *out += weights[lane] * f32::from_bits(u32::from(values[lane][channel]) << 16);
        }
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_values_neon(
    output: &mut [f32],
    values: &[[u16; 256]; 32],
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
                            vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(
                                values[lane].as_ptr().add(channel + group * 4),
                            ))),
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

fn dots(query: &[f32], keys: &[[u16; 32]; 256], visible: Range<usize>) -> [f32; 32] {
    assert_eq!(query.len(), 256);
    let mut output = [0.0; 32];
    #[cfg(target_arch = "aarch64")]
    {
        let first = visible.start / 4 * 4;
        let last = visible.end.div_ceil(4) * 4;
        // SAFETY: runtime NEON is required; each specialization accesses only
        // complete four-key groups inside the initialized tile and 256 channels.
        // Unused edge lanes never enter scores, validation, or value accumulation.
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
    }
    #[cfg(not(target_arch = "aarch64"))]
    for lane in visible {
        for channel in 0..256 {
            output[lane] += query[channel] * f32::from_bits(u32::from(keys[channel][lane]) << 16);
        }
    }
    output
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dots_neon<const GROUPS: usize>(
    query: &[f32],
    keys: &[[u16; 32]; 256],
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
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(
                            keys[channel].as_ptr().add(first + group * 4),
                        ))),
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
    fn grouped_error_priority_follows_scalar_rows_before_task_order() {
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
        // A zero KV head 1 prefix lets row 0 reach its overflowing self score.
        let mut prefix_keys = keys.clone();
        prefix_keys[256..].fill(0.0);
        let first = layer.prepare_row(0, &good, &prefix_keys, &values).unwrap();
        layer.output_row(&state, 0, &first).unwrap();
        layer.append_prepared(&mut state, 0, &first).unwrap();
        let mut prepared = Vec::new();
        for (row, head) in [6, 0].into_iter().enumerate() {
            let mut query = good.clone();
            query[head * 512..head * 512 + 256].fill(1e-20);
            let p = layer.prepare_row(row + 1, &query, &keys, &values).unwrap();
            layer.append_prepared(&mut state, row + 1, &p).unwrap();
            prepared.push(p);
        }
        // Row 1's KV head 0 tasks precede row 0's KV head 1 tasks.
        assert!(matches!(
            layer.output_row(&state, 2, &prepared[1]),
            Err(Error::NonFinite("attention prefix score"))
        ));
        let expected = prepared
            .iter()
            .enumerate()
            .map(|(row, p)| layer.output_row(&state, row + 1, p))
            .collect::<Result<Vec<_>>>()
            .unwrap_err();
        assert!(matches!(expected, Error::NonFinite("attention self score")));
        let before = state.clone();
        for threads in [1, 4, 10] {
            let mut engine = Compute::new(
                crate::Engine::new(crate::Kernel::Portable, threads).unwrap(),
                crate::Backend::Cpu,
            )
            .unwrap();
            let actual = output_rows(&mut engine, &layer, &state, 1, &prepared).unwrap_err();
            assert_eq!(
                actual.to_string(),
                expected.to_string(),
                "{threads} workers"
            );
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
                keys: [[(f32::NAN.to_bits() >> 16) as u16; 32]; 256],
                values: [[(f32::NAN.to_bits() >> 16) as u16; 256]; 32],
                key_scales: [f32::NAN; 32],
                value_scales: [f32::NAN; 32],
            };
            for lane in visible.clone() {
                for column in &mut tile.keys {
                    column[lane] = 0;
                }
                tile.values[lane] = [((-127.0_f32).to_bits() >> 16) as u16; 256];
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
            std::array::from_fn(|lane| {
                ((((i + lane * 31) as u8 as i8) as f32).to_bits() >> 16) as u16
            })
        });
        for visible in [0..1, 1..4, 3..8, 5..16, 7..17, 0..31, 0..32] {
            let actual = dots(&query, &keys, visible.clone());
            for lane in visible {
                let mut expected = 0.0;
                for channel in 0..256 {
                    expected +=
                        query[channel] * f32::from_bits(u32::from(keys[channel][lane]) << 16);
                }
                assert_eq!(actual[lane].to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
#[ignore = "CPU arithmetic throughput diagnostic; no model files"]
fn bench_attention_core_rates() {
    if !super::super::neon_available() {
        return;
    }
    use std::{hint::black_box, time::Instant};
    let query = vec![0.25; 256];
    let keys = [[0x3e80; 32]; 256];
    let values = [[0x3e80; 256]; 32];
    let weights = [0.25; 32];
    for width in [4, 16, 32] {
        let mut rates = Vec::new();
        for pass in 0..6 {
            let iterations = 50_000;
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(dots(black_box(&query), black_box(&keys), 0..width));
            }
            let seconds = start.elapsed().as_secs_f64();
            let rate = 2.0 * 256.0 * width as f64 * iterations as f64 / seconds / 1e9;
            if pass > 0 {
                rates.push(rate);
            }
        }
        rates.sort_by(f64::total_cmp);
        println!(
            "score width={width} GFLOPs_median={:.3} GFLOPs_max={:.3}",
            rates[2], rates[4]
        );
    }
    let mut rates = Vec::new();
    for pass in 0..6 {
        let mut output = vec![0.0; 256];
        let iterations = 50_000;
        let start = Instant::now();
        for _ in 0..iterations {
            add_values(
                black_box(&mut output),
                black_box(&values),
                black_box(&weights),
                0..32,
            );
        }
        black_box(output);
        let seconds = start.elapsed().as_secs_f64();
        let rate = 2.0 * 256.0 * 32.0 * iterations as f64 / seconds / 1e9;
        if pass > 0 {
            rates.push(rate);
        }
    }
    rates.sort_by(f64::total_cmp);
    println!(
        "value width=32 GFLOPs_median={:.3} GFLOPs_max={:.3}",
        rates[2], rates[4]
    );
}
