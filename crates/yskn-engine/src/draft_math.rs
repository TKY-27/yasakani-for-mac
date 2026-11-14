//! Draft attention and reduced-head kernels (D18).
//!
//! Every output value runs exactly the scalar reference's floating operations in
//! its order (`yskn_ref::draft`, `yskn_ref::draft_head`). Vectors hold only
//! independent outputs: queries, reduction lanes, tokens or channels. NEON is part
//! of the AArch64 baseline; callers select it unless the engine kernel is
//! `Portable`, whose scalar loops are the fallback on every architecture.
use yskn_format::bf16_to_f32;
use yskn_ref::draft::{BLOCK_ROWS, xor_sum32};
use yskn_ref::portable_math::exp;

use crate::{Error, Result};

pub(crate) const HD: usize = 128;
/// The reference's 1/sqrt(128) literal (yskn_ref::draft).
const SCALE: f32 = 0.088_388_346;
/// Tokens per packed approximate-weight tile.
pub(crate) const TILE: usize = 8;
/// SHEAD rank (validated by `HeadLayout::read`).
pub(crate) const RANK: usize = 512;

/// Attention geometry for one block at ring position `position`.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub window: usize,
    pub position: usize,
}
impl Shape {
    fn group(&self) -> usize {
        self.query_heads / self.kv_heads
    }
    /// Queries per kv head: query index = row * group + head-within-group.
    pub fn queries(&self) -> usize {
        BLOCK_ROWS * self.group()
    }
    fn start(&self) -> usize {
        self.position.saturating_sub(self.window - 1)
    }
    fn chunk(&self) -> usize {
        (self.position - self.start()).div_ceil(32)
    }
}

/// One history split's partials for every query of one kv head.
pub(crate) struct HistoryPart {
    maximum: Vec<f32>,
    denominator: Vec<f32>,
    output: Vec<f32>,
    /// An unmasked score of this query was not finite (the reference's error).
    failed: Vec<bool>,
}

/// History split `split` of kv head `kv`: scores, local softmax and value sums.
pub(crate) fn history_part(
    q: &[f32],
    ring_keys: &[u16],
    ring_values: &[u16],
    s: Shape,
    kv: usize,
    split: usize,
    simd: bool,
) -> HistoryPart {
    let (group, nq) = (s.group(), s.queries());
    let mut part = HistoryPart {
        maximum: vec![f32::NEG_INFINITY; nq],
        denominator: vec![0.0; nq],
        output: vec![0.0; nq * HD],
        failed: vec![false; nq],
    };
    let (start, chunk) = (s.start(), s.chunk());
    let count = s.position - start;
    let first = split * chunk;
    let n = count.min(first + chunk).saturating_sub(first);
    if n == 0 {
        return part;
    }
    // Keys padded to a multiple of four with zeros; padded dots are never read.
    let padded = n.next_multiple_of(4);
    let mut keys = vec![0.0; padded * HD];
    let mut values = vec![0.0; n * HD];
    for j in 0..n {
        let offset = (kv * s.window + (start + first + j) % s.window) * HD;
        for c in 0..HD {
            keys[j * HD + c] = bf16_to_f32(ring_keys[offset + c]);
            values[j * HD + c] = bf16_to_f32(ring_values[offset + c]);
        }
    }
    let mut transposed = vec![0.0; HD * nq];
    for row in 0..BLOCK_ROWS {
        for g in 0..group {
            let query = &q[(row * s.query_heads + kv * group + g) * HD..][..HD];
            for (c, &value) in query.iter().enumerate() {
                transposed[c * nq + row * group + g] = value;
            }
        }
    }
    let mut dots = vec![0.0; padded * nq];
    key_dots(&transposed, nq, &keys, padded, &mut dots, simd);
    let mut weights = vec![0.0; nq * 64];
    for qi in 0..nq {
        let visible = (s.position + qi / group).saturating_sub(s.window - 1);
        let mut scores = [f32::NEG_INFINITY; 64];
        for (j, score) in scores[..n].iter_mut().enumerate() {
            if start + first + j < visible {
                continue;
            }
            *score = dots[j * nq + qi] * SCALE;
            part.failed[qi] |= !score.is_finite();
        }
        // Max is exact for finite inputs; its pair/XOR tree cannot change it.
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        part.maximum[qi] = maximum;
        if maximum == f32::NEG_INFINITY {
            continue;
        }
        let w = &mut weights[qi * 64..(qi + 1) * 64];
        for (w, &score) in w.iter_mut().zip(&scores) {
            *w = if score == f32::NEG_INFINITY {
                0.0
            } else {
                exp(score - maximum)
            };
        }
        let pairs: [f32; 32] = std::array::from_fn(|i| w[i] + w[i + 32]);
        part.denominator[qi] = xor_sum32(&pairs);
    }
    value_sums(&weights, &values, n, nq, &mut part.output, simd);
    part
}

/// `dots[j][qi]`: serial channel-ascending FMA chain of query qi with key j.
fn key_dots(transposed: &[f32], nq: usize, keys: &[f32], n: usize, dots: &mut [f32], simd: bool) {
    #[cfg(target_arch = "aarch64")]
    if simd {
        // SAFETY: `simd` is set only when the engine kernel was runtime-detected with NEON.
        return unsafe { neon::key_dots(transposed, nq, keys, n, dots) };
    }
    let _ = simd;
    for j in 0..n {
        let key = &keys[j * HD..(j + 1) * HD];
        let acc = &mut dots[j * nq..(j + 1) * nq];
        for (c, &k) in key.iter().enumerate() {
            for (acc, &q) in acc.iter_mut().zip(&transposed[c * nq..(c + 1) * nq]) {
                // Nested pair FMAs execute even then odd: q35_bdraft.cu:175.
                *acc = q.mul_add(k, *acc);
            }
        }
    }
}

/// `output[qi][c]`: key-ascending FMA chain over the split's values (q35_bdraft.cu:205).
fn value_sums(
    weights: &[f32],
    values: &[f32],
    n: usize,
    nq: usize,
    output: &mut [f32],
    simd: bool,
) {
    #[cfg(target_arch = "aarch64")]
    if simd {
        // SAFETY: `simd` is set only when the engine kernel was runtime-detected with NEON.
        return unsafe { neon::value_sums(weights, values, n, nq, output) };
    }
    let _ = simd;
    for qi in 0..nq {
        let out = &mut output[qi * HD..(qi + 1) * HD];
        for j in 0..n {
            let weight = weights[qi * 64 + j];
            for (o, &v) in out.iter_mut().zip(&values[j * HD..(j + 1) * HD]) {
                *o = weight.mul_add(v, *o);
            }
        }
    }
}

/// One (row, head): the block partial, then the 33-part combination. `parts` are
/// the head's kv-head history splits in ascending order.
pub(crate) fn finish(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    parts: &[HistoryPart],
    s: Shape,
    row: usize,
    head: usize,
) -> Result<[f32; HD]> {
    let group = s.group();
    let kv = head / group;
    let qi = row * group + head % group;
    if parts.iter().any(|p| p.failed[qi]) {
        return Err(Error::NonFinite("draft history score"));
    }
    let query = &q[(row * s.query_heads + head) * HD..][..HD];
    let scaled: [f32; HD] = std::array::from_fn(|c| query[c] * SCALE);
    let mut maximum = f32::NEG_INFINITY;
    let mut denominator = 0.0_f32;
    let mut current = [0.0_f32; HD];
    for j in 0..BLOCK_ROWS {
        let offset = (j * s.kv_heads + kv) * HD;
        let key = &k[offset..offset + HD];
        let lanes: [f32; 32] = std::array::from_fn(|lane| {
            let c = 4 * lane;
            // Three explicit nested FMAs, last product separate: q35_bdraft.cu:141.
            scaled[c].mul_add(
                key[c],
                scaled[c + 1].mul_add(
                    key[c + 1],
                    scaled[c + 2].mul_add(key[c + 2], scaled[c + 3] * key[c + 3]),
                ),
            )
        });
        let score = xor_sum32(&lanes);
        if !score.is_finite() {
            return Err(Error::NonFinite("draft block score"));
        }
        let next = maximum.max(score);
        let carry = exp(maximum - next);
        let weight = exp(score - next);
        // q35_bdraft.cu:146,148.
        denominator = denominator.mul_add(carry, weight);
        for (o, &value) in current.iter_mut().zip(&v[offset..offset + HD]) {
            *o = o.mul_add(carry, weight * value);
        }
        maximum = next;
    }
    let overall = parts
        .iter()
        .map(|p| p.maximum[qi])
        .chain([maximum])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f32;
    let mut output = [0.0_f32; HD];
    let history = parts.iter().map(|p| {
        (
            p.maximum[qi],
            p.denominator[qi],
            &p.output[qi * HD..(qi + 1) * HD],
        )
    });
    for (part_maximum, part_denominator, part_output) in
        history.chain([(maximum, denominator, &current[..])])
    {
        if part_maximum == f32::NEG_INFINITY {
            continue;
        }
        let weight = exp(part_maximum - overall);
        // q35_bdraft.cu:579 (unfused-output equivalent :226).
        total = part_denominator.mul_add(weight, total);
        for (o, &p) in output.iter_mut().zip(part_output) {
            *o = p.mul_add(weight, *o);
        }
    }
    Ok(output.map(|o| o / total))
}

/// `x[128t + 4lane + comp]` stored as `[t][comp][lane]` for lane-vector loads.
pub(crate) fn deinterleave(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (t, block) in x.as_chunks::<128>().0.iter().enumerate() {
        for lane in 0..32 {
            for comp in 0..4 {
                out[(t * 4 + comp) * 32 + lane] = block[4 * lane + comp];
            }
        }
    }
    out
}

/// Rank projection for one rank row (q35_bdraft.cu:786): lane chains run over
/// 512-blocks, then u, then component. `x` is the row, `xd` its deinterleaving.
pub(crate) fn low_rank(x: &[f32], xd: &[f32], weights: &[f32], simd: bool) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if simd {
        // SAFETY: `simd` is set only when the engine kernel was runtime-detected with NEON.
        return xor_sum32(&unsafe { neon::lane_dots_f32(xd, weights) });
    }
    let _ = (simd, xd);
    let mut partial = [0.0_f32; 32];
    for block in (0..x.len()).step_by(512) {
        for u in 0..4 {
            for (lane, acc) in partial.iter_mut().enumerate() {
                for component in 0..4 {
                    let i = block + 4 * (lane + 32 * u) + component;
                    *acc = x[i].mul_add(weights[i], *acc);
                }
            }
        }
    }
    xor_sum32(&partial)
}

/// Full-width int8 rescoring of one token row (q35_bdraft.cu:920-921), before its scale.
pub(crate) fn rescore(x: &[f32], xd: &[f32], weights: &[u8], simd: bool) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if simd {
        // SAFETY: `simd` is set only when the engine kernel was runtime-detected with NEON.
        return xor_sum32(&unsafe { neon::lane_dots_i8(xd, weights) });
    }
    let _ = (simd, xd);
    let mut partial = [0.0_f32; 32];
    for t in 0..x.len() / 128 {
        for (lane, acc) in partial.iter_mut().enumerate() {
            for component in 0..4 {
                let i = 4 * (lane + 32 * t) + component;
                *acc = x[i].mul_add(weights[i] as i8 as f32, *acc);
            }
        }
    }
    xor_sum32(&partial)
}

/// Repack `[token][512]` approximate weights into `[tile][512][TILE]`, zero-padded.
pub(crate) fn pack_approximate(weights: &[u8], vocabulary: usize) -> Vec<i8> {
    let tiles = vocabulary.div_ceil(TILE);
    let mut packed = vec![0_i8; tiles * RANK * TILE];
    for (token, row) in weights.as_chunks::<RANK>().0.iter().enumerate() {
        let (tile, t) = (token / TILE, token % TILE);
        for (i, &w) in row.iter().enumerate() {
            packed[(tile * RANK + i) * TILE + t] = w as i8;
        }
    }
    packed
}

/// Approximate scores (q35_bdraft.cu:839) of tiles `tiles` for the seven
/// rank-projected rows `low` ([row][512]): `[row][tile token]`, times each scale.
pub(crate) fn approximate_scores(
    low: &[f32],
    packed: &[i8],
    scales: &[f32],
    tiles: std::ops::Range<usize>,
    simd: bool,
) -> Vec<f32> {
    const ROWS: usize = BLOCK_ROWS - 1;
    let tokens = (tiles.len() * TILE).min(scales.len() - tiles.start * TILE);
    let mut out = vec![0.0; ROWS * tokens];
    // partial[row][lane][t] for one tile.
    let mut partial = [[[0.0_f32; TILE]; 32]; ROWS];
    for (local, tile) in tiles.clone().enumerate() {
        let weights = &packed[tile * RANK * TILE..(tile + 1) * RANK * TILE];
        #[cfg(target_arch = "aarch64")]
        let vectorized = simd;
        #[cfg(not(target_arch = "aarch64"))]
        let vectorized = false;
        if vectorized {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: `simd` is set only when the engine kernel was runtime-detected with NEON.
            unsafe {
                neon::approximate_tile(low, weights, &mut partial)
            };
        } else {
            for (row, partial) in partial.iter_mut().enumerate() {
                for (lane, acc) in partial.iter_mut().enumerate() {
                    *acc = [0.0; TILE];
                    for c in 0..16 {
                        let i = 16 * lane + c;
                        let x = low[row * RANK + i];
                        for (acc, &w) in acc.iter_mut().zip(&weights[i * TILE..(i + 1) * TILE]) {
                            *acc = x.mul_add(f32::from(w), *acc);
                        }
                    }
                }
            }
        }
        for (row, partial) in partial.iter_mut().enumerate() {
            // xor_sum32's tree, for TILE tokens at once.
            for half in [16, 8, 4, 2, 1] {
                for lane in 0..half {
                    for t in 0..TILE {
                        partial[lane][t] += partial[lane + half][t];
                    }
                }
            }
            for (t, &sum) in partial[0].iter().enumerate() {
                let token = local * TILE + t;
                if token < tokens {
                    out[row * tokens + token] = sum * scales[tiles.start * TILE + token];
                }
            }
        }
    }
    out
}

/// Correction feature (q35_bdraft.cu:303) with each lane's chain unchanged; the
/// lane loop is innermost so its 32 independent chains vectorize.
pub(crate) fn correction_feature(x: &[f32], hp: &[f32], rank: usize) -> Result<Vec<f32>> {
    let hidden = x.len();
    let mut feature = vec![0.0; rank];
    for (r, value) in feature.iter_mut().enumerate() {
        let weights = &hp[r * hidden..(r + 1) * hidden];
        let mut partial = [0.0_f32; 32];
        for (x, w) in x
            .as_chunks::<32>()
            .0
            .iter()
            .zip(weights.as_chunks::<32>().0)
        {
            for ((acc, &x), &w) in partial.iter_mut().zip(x).zip(w) {
                *acc = x.mul_add(w, *acc);
            }
        }
        *value = xor_sum32(&partial);
    }
    if feature.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite("draft correction feature"));
    }
    Ok(feature)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! Base AArch64 NEON (always present on this architecture). Loads go through
    //! bounds-checked slices; `vfmaq*` is one fused multiply-add per element.
    use super::{HD, RANK, TILE};
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    fn ld(s: &[f32], i: usize) -> float32x4_t {
        let s = &s[i..i + 4];
        // SAFETY: `s` holds exactly four readable f32 values.
        unsafe { vld1q_f32(s.as_ptr()) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn st(s: &mut [f32], i: usize, v: float32x4_t) {
        let s = &mut s[i..i + 4];
        // SAFETY: `s` holds exactly four writable f32 values.
        unsafe { vst1q_f32(s.as_mut_ptr(), v) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn ld4_f32(s: &[f32], i: usize) -> float32x4x4_t {
        let s = &s[i..i + 16];
        // SAFETY: `s` holds exactly sixteen readable f32 values.
        unsafe { vld4q_f32(s.as_ptr()) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn ld_s8(s: &[i8], i: usize) -> int8x16_t {
        let s = &s[i..i + 16];
        // SAFETY: `s` holds exactly sixteen readable bytes.
        unsafe { vld1q_s8(s.as_ptr()) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn ld4_u8_as_s8(s: &[u8], i: usize) -> int8x16x4_t {
        let s = &s[i..i + 64];
        // SAFETY: `s` holds exactly 64 readable bytes; i8 and u8 share layout.
        unsafe { vld4q_s8(s.as_ptr().cast::<i8>()) }
    }
    /// Exact int8 -> f32 for 16 values, in order, as four vectors.
    #[inline]
    #[target_feature(enable = "neon")]
    fn widen(x: int8x16_t) -> [float32x4_t; 4] {
        let low = vmovl_s8(vget_low_s8(x));
        let high = vmovl_high_s8(x);
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
            vcvtq_f32_s32(vmovl_high_s16(low)),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
            vcvtq_f32_s32(vmovl_high_s16(high)),
        ]
    }

    /// Query lanes: tiles of 8 queries x 4 keys, channel-ascending per chain.
    #[target_feature(enable = "neon")]
    pub(super) fn key_dots(t: &[f32], nq: usize, keys: &[f32], n: usize, dots: &mut [f32]) {
        assert!(nq.is_multiple_of(8) && n.is_multiple_of(4));
        for qb in (0..nq).step_by(8) {
            for jb in (0..n).step_by(4) {
                let mut acc = [[vdupq_n_f32(0.0); 2]; 4];
                for c in (0..HD).step_by(4) {
                    let k: [float32x4_t; 4] = std::array::from_fn(|j| ld(keys, (jb + j) * HD + c));
                    macro_rules! channel {
                        ($e:literal) => {{
                            let q0 = ld(t, (c + $e) * nq + qb);
                            let q1 = ld(t, (c + $e) * nq + qb + 4);
                            for j in 0..4 {
                                acc[j][0] = vfmaq_laneq_f32::<$e>(acc[j][0], q0, k[j]);
                                acc[j][1] = vfmaq_laneq_f32::<$e>(acc[j][1], q1, k[j]);
                            }
                        }};
                    }
                    channel!(0);
                    channel!(1);
                    channel!(2);
                    channel!(3);
                }
                for (j, acc) in acc.iter().enumerate() {
                    st(dots, (jb + j) * nq + qb, acc[0]);
                    st(dots, (jb + j) * nq + qb + 4, acc[1]);
                }
            }
        }
    }

    /// Channel lanes: two queries x 32 channels per tile, key-ascending per chain.
    #[target_feature(enable = "neon")]
    pub(super) fn value_sums(w: &[f32], values: &[f32], n: usize, nq: usize, out: &mut [f32]) {
        assert!(nq.is_multiple_of(2));
        for qp in (0..nq).step_by(2) {
            for cb in (0..HD).step_by(32) {
                let mut acc = [[vdupq_n_f32(0.0); 8]; 2];
                for j in 0..n {
                    let w0 = vdupq_n_f32(w[qp * 64 + j]);
                    let w1 = vdupq_n_f32(w[(qp + 1) * 64 + j]);
                    for b in 0..8 {
                        let v = ld(values, j * HD + cb + 4 * b);
                        acc[0][b] = vfmaq_f32(acc[0][b], w0, v);
                        acc[1][b] = vfmaq_f32(acc[1][b], w1, v);
                    }
                }
                for (q, acc) in acc.iter().enumerate() {
                    for (b, &a) in acc.iter().enumerate() {
                        st(out, (qp + q) * HD + cb + 4 * b, a);
                    }
                }
            }
        }
    }

    /// Lanes of `x[128t + 4lane + comp] * w[...]`, chains over t then comp.
    #[target_feature(enable = "neon")]
    pub(super) fn lane_dots_f32(xd: &[f32], w: &[f32]) -> [f32; 32] {
        let mut acc = [vdupq_n_f32(0.0); 8];
        for t in 0..w.len() / 128 {
            for (g, acc) in acc.iter_mut().enumerate() {
                let w = ld4_f32(w, 128 * t + 16 * g);
                for (comp, w) in [w.0, w.1, w.2, w.3].into_iter().enumerate() {
                    *acc = vfmaq_f32(*acc, ld(xd, (t * 4 + comp) * 32 + 4 * g), w);
                }
            }
        }
        lanes(acc)
    }

    /// As `lane_dots_f32`, with int8 weights converted exactly to f32.
    #[target_feature(enable = "neon")]
    pub(super) fn lane_dots_i8(xd: &[f32], w: &[u8]) -> [f32; 32] {
        let mut acc = [vdupq_n_f32(0.0); 8];
        for t in 0..w.len() / 128 {
            for half in 0..2 {
                let w = ld4_u8_as_s8(w, 128 * t + 64 * half);
                for (comp, w) in [w.0, w.1, w.2, w.3].into_iter().enumerate() {
                    for (quarter, w) in widen(w).into_iter().enumerate() {
                        let g = 4 * half + quarter;
                        acc[g] = vfmaq_f32(acc[g], ld(xd, (t * 4 + comp) * 32 + 4 * g), w);
                    }
                }
            }
        }
        lanes(acc)
    }

    #[target_feature(enable = "neon")]
    fn lanes(acc: [float32x4_t; 8]) -> [f32; 32] {
        let mut out = [0.0; 32];
        for (g, a) in acc.into_iter().enumerate() {
            st(&mut out, 4 * g, a);
        }
        out
    }

    /// One packed tile for the seven rows: token lanes, component-ascending chains.
    #[allow(clippy::needless_range_loop)] // `lane` also addresses the weights.
    #[target_feature(enable = "neon")]
    pub(super) fn approximate_tile(
        low: &[f32],
        w: &[i8],
        partial: &mut [[[f32; TILE]; 32]; super::BLOCK_ROWS - 1],
    ) {
        const ROWS: usize = super::BLOCK_ROWS - 1;
        for lane in 0..32 {
            let mut acc = [[vdupq_n_f32(0.0); 2]; ROWS];
            for quad in 0..4 {
                let i = 16 * lane + 4 * quad;
                let pair0 = widen(ld_s8(w, i * TILE));
                let pair1 = widen(ld_s8(w, (i + 2) * TILE));
                // weights[e] = components i+e for tokens 0..4 and 4..8.
                let weights = [
                    [pair0[0], pair0[1]],
                    [pair0[2], pair0[3]],
                    [pair1[0], pair1[1]],
                    [pair1[2], pair1[3]],
                ];
                for (row, acc) in acc.iter_mut().enumerate() {
                    let x = ld(low, row * RANK + i);
                    macro_rules! component {
                        ($e:literal) => {{
                            acc[0] = vfmaq_laneq_f32::<$e>(acc[0], weights[$e][0], x);
                            acc[1] = vfmaq_laneq_f32::<$e>(acc[1], weights[$e][1], x);
                        }};
                    }
                    component!(0);
                    component!(1);
                    component!(2);
                    component!(3);
                }
            }
            for (row, acc) in acc.iter().enumerate() {
                st(&mut partial[row][lane], 0, acc[0]);
                st(&mut partial[row][lane], 4, acc[1]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use yskn_format::f32_to_bf16;
    use yskn_ref::draft::DraftDimensions;
    use yskn_ref::draft_head::ReducedHead;

    struct Random(u64);
    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Uniform in [-scale, scale), BF16-representable.
        fn value(&mut self, scale: f32) -> f32 {
            let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
            yskn_ref::portable_math::bfloat((2.0 * unit - 1.0) * scale)
        }
        fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.value(scale)).collect()
        }
    }

    fn variants() -> Vec<bool> {
        let mut variants = vec![false];
        if cfg!(target_arch = "aarch64") && crate::Kernel::Neon.available() {
            variants.push(true);
        }
        variants
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn fast_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rk: &[u16],
        rv: &[u16],
        s: Shape,
        simd: bool,
    ) -> Result<Vec<f32>> {
        let parts: Vec<_> = (0..s.kv_heads * 32)
            .map(|i| history_part(q, rk, rv, s, i / 32, i % 32, simd))
            .collect();
        let group = s.query_heads / s.kv_heads;
        let mut out = Vec::new();
        for row in 0..BLOCK_ROWS {
            for head in 0..s.query_heads {
                let kv = head / group;
                out.extend(finish(
                    q,
                    k,
                    v,
                    &parts[kv * 32..(kv + 1) * 32],
                    s,
                    row,
                    head,
                )?);
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite("draft attention output"));
        }
        Ok(out)
    }

    #[test]
    fn history_splits_and_block_combination_match_reference_attention_bits() {
        let mut random = Random(0x5eed_1234_abcd_0001);
        // (query heads, kv heads, window, positions): grouped heads, a wrapped
        // ring, every split length 0..64, padded key tiles and per-row masks.
        let cases: [(usize, usize, usize, &[usize]); 4] = [
            (4, 4, 8, &[0, 1, 5, 7, 8, 9, 20]),
            (8, 2, 2048, &[0, 3, 37, 100, 2047, 2048, 2100]),
            (16, 4, 256, &[255, 300, 513]),
            (8, 8, 64, &[33, 70]),
        ];
        for (query_heads, kv_heads, window, positions) in cases {
            let d = DraftDimensions {
                hidden: 512,
                features: 2560,
                feed_forward: 512,
                vocabulary: 32,
                layers: 1,
                query_heads,
                kv_heads,
                window,
                correction_rank: 32,
                epsilon: 1e-6,
            };
            let q = random.values(BLOCK_ROWS * query_heads * HD, 2.0);
            let k = random.values(BLOCK_ROWS * kv_heads * HD, 2.0);
            let v = random.values(BLOCK_ROWS * kv_heads * HD, 2.0);
            let rk: Vec<_> = (0..kv_heads * window * HD)
                .map(|_| f32_to_bf16(random.value(2.0)))
                .collect();
            let rv: Vec<_> = (0..kv_heads * window * HD)
                .map(|_| f32_to_bf16(random.value(2.0)))
                .collect();
            for &position in positions {
                let expected =
                    yskn_ref::draft::attention(&q, &k, &v, &rk, &rv, position, &d).unwrap();
                let s = Shape {
                    query_heads,
                    kv_heads,
                    window,
                    position,
                };
                for simd in variants() {
                    let actual = fast_attention(&q, &k, &v, &rk, &rv, s, simd).unwrap();
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "{query_heads}/{kv_heads} w={window} p={position} simd={simd}"
                    );
                }
            }
            // An overflowing visible history score fails exactly like the reference.
            let mut huge = rk.clone();
            huge[..HD].fill(f32_to_bf16(3.0e38));
            let mut hot = q.clone();
            hot[..HD].fill(1.0e3);
            let position = positions[positions.len() - 1].min(window - 1).max(1);
            let expected =
                yskn_ref::draft::attention(&hot, &k, &v, &huge, &rv, position, &d).unwrap_err();
            for simd in variants() {
                let s = Shape {
                    query_heads,
                    kv_heads,
                    window,
                    position,
                };
                let actual = fast_attention(&hot, &k, &v, &huge, &rv, s, simd).unwrap_err();
                assert_eq!(actual.to_string(), expected.to_string());
            }
        }
    }

    fn shead(random: &mut Random, vocabulary: usize, hidden: usize) -> Vec<u8> {
        let mut b = b"SHEAD001".to_vec();
        for n in [vocabulary as i32, hidden as i32, 512, 0] {
            b.extend_from_slice(&n.to_le_bytes());
        }
        for _ in 0..RANK * hidden {
            b.extend_from_slice(&(random.value(1.0) * 0.0625).to_le_bytes());
        }
        for _ in 0..vocabulary * RANK {
            b.push(random.next() as u8);
        }
        for _ in 0..vocabulary {
            b.extend_from_slice(&(random.value(1.0) * 0.125).to_le_bytes());
        }
        for _ in 0..vocabulary * hidden {
            b.push(random.next() as u8);
        }
        for _ in 0..vocabulary {
            b.extend_from_slice(&(random.value(1.0) / 1024.0).to_le_bytes());
        }
        b
    }

    #[test]
    fn head_stages_and_correction_feature_match_reference_bits() {
        let mut random = Random(0x0dd_ba11_cafe_f00d);
        for (vocabulary, hidden) in [(37, 1024), (100, 512), (16, 1536)] {
            let bytes = shead(&mut random, vocabulary, hidden);
            let head = ReducedHead::read(&mut Cursor::new(&bytes)).unwrap();
            let layout = head.layout().clone();
            let range = |i: usize| layout.arrays[i].start as usize..layout.arrays[i].end as usize;
            let floats = |i: usize| -> Vec<f32> {
                bytes[range(i)]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            };
            let (projection, approximate_scale, rescore_scale) = (floats(0), floats(2), floats(4));
            let packed = pack_approximate(&bytes[range(1)], vocabulary);
            let rows: Vec<Vec<f32>> = (0..BLOCK_ROWS - 1)
                .map(|_| random.values(hidden, 1.5))
                .collect();
            let tiles = vocabulary.div_ceil(TILE);
            for simd in variants() {
                let mut low = Vec::new();
                for x in &rows {
                    let xd = deinterleave(x);
                    let actual: Vec<_> = (0..RANK)
                        .map(|k| low_rank(x, &xd, &projection[k * hidden..(k + 1) * hidden], simd))
                        .collect();
                    let expected = head.low_rank(x).unwrap();
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "low rank V={vocabulary} simd={simd}"
                    );
                    low.extend(expected);
                    let ids: Vec<u32> = (0..vocabulary as u32).collect();
                    let expected = head.rescore(x, &ids);
                    for c in &expected {
                        let t = c.token as usize;
                        let weights = &bytes[range(3)][t * hidden..(t + 1) * hidden];
                        let score = rescore(x, &xd, weights, simd) * rescore_scale[t];
                        assert_eq!(
                            score.to_bits(),
                            c.score.to_bits(),
                            "rescore token {t} simd={simd}"
                        );
                    }
                }
                // Whole range, and a split that leaves a partial last tile.
                let whole = approximate_scores(&low, &packed, &approximate_scale, 0..tiles, simd);
                let split = tiles / 2;
                let first = approximate_scores(&low, &packed, &approximate_scale, 0..split, simd);
                let second =
                    approximate_scores(&low, &packed, &approximate_scale, split..tiles, simd);
                let (a, b) = (
                    first.len() / (BLOCK_ROWS - 1),
                    second.len() / (BLOCK_ROWS - 1),
                );
                for r in 0..BLOCK_ROWS - 1 {
                    let expected = head.approximate_scores(&low[r * RANK..(r + 1) * RANK]);
                    assert_eq!(
                        bits(&whole[r * vocabulary..(r + 1) * vocabulary]),
                        bits(&expected)
                    );
                    let joined: Vec<_> = first[r * a..(r + 1) * a]
                        .iter()
                        .chain(&second[r * b..(r + 1) * b])
                        .copied()
                        .collect();
                    assert_eq!(
                        bits(&joined),
                        bits(&expected),
                        "split approximate V={vocabulary}"
                    );
                }
            }
            let rank = 64;
            let hp = random.values(rank * hidden, 0.25);
            for x in &rows {
                assert_eq!(
                    bits(&correction_feature(x, &hp, rank).unwrap()),
                    bits(&yskn_ref::draft::correction_feature(x, &hp, rank).unwrap())
                );
            }
        }
    }
}
