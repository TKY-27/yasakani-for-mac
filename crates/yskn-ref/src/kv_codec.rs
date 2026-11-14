//! Scalar source-order KV row codec. All floating operations are separate RN
//! binary32 operations; the encoder sources contain no explicit FMA.
use crate::kv_profile::{CodeTable, Group, KvProfile, invalid};
use crate::portable_math::{bfloat, half, sqrt_rn};
use crate::{Result, require_finite};

#[derive(Clone, Debug, PartialEq)]
pub struct Int8Row {
    pub(crate) integers: Vec<i8>,
    pub(crate) scale: f32,
}
impl Int8Row {
    pub fn integers(&self) -> &[i8] {
        &self.integers
    }
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackedRow {
    words: Vec<u32>,
    bits: u8,
    scale: f32,
    bases: Box<[f32; 256]>,
}
impl PackedRow {
    pub fn words(&self) -> &[u32] {
        &self.words
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CacheRow {
    Int8(Int8Row),
    Packed(PackedRow),
}
impl CacheRow {
    /// Attention reads only validated, immutable operands. Int8 conversion is
    /// exact; packed operands were BF16-rounded once when the row was encoded.
    pub(crate) fn bases<'a>(&'a self, scratch: &'a mut [f32; 256]) -> &'a [f32; 256] {
        match self {
            Self::Int8(row) => {
                for (base, &integer) in scratch.iter_mut().zip(&row.integers) {
                    *base = f32::from(integer);
                }
                scratch
            }
            Self::Packed(row) => &row.bases,
        }
    }

    pub fn scale(&self) -> f32 {
        match self {
            Self::Int8(r) => r.scale,
            Self::Packed(r) => r.scale,
        }
    }
    /// Base operand, without row scale or mean. Packed table entries are BF16.
    pub fn key_base(&self, profile: &KvProfile, group: usize, channel: usize) -> Result<f32> {
        if channel >= 256 {
            return Err(invalid("channel outside row"));
        }
        match self {
            Self::Int8(r) => Ok(f32::from(r.integers[channel])),
            Self::Packed(r) => {
                let g = validate_packed(profile, group, r)?;
                let state = packed_state(r, channel);
                Ok(bfloat(profile.tables()[g.table].reconstruction(state)))
            }
        }
    }
    pub fn value_base(&self, profile: &KvProfile, group: usize, channel: usize) -> Result<f32> {
        self.key_base(profile, group, channel)
    }
    /// Add denominator * this mean once per attention split, after self.
    pub fn value_mean(&self, profile: &KvProfile, group: usize, channel: usize) -> Result<f32> {
        if channel >= 256 {
            return Err(invalid("channel outside row"));
        }
        match self {
            Self::Int8(_) => Ok(0.0),
            Self::Packed(r) => Ok(validate_packed(profile, group, r)?
                .mean
                .as_ref()
                .map_or(0.0, |m| m[channel])),
        }
    }
}

fn add(a: i32, b: i32) -> Result<i32> {
    a.checked_add(b)
        .ok_or_else(|| invalid("i32 trellis overflow"))
}
fn sub(a: i32, b: i32) -> Result<i32> {
    a.checked_sub(b)
        .ok_or_else(|| invalid("i32 trellis overflow"))
}
fn mul(a: i32, b: i32) -> Result<i32> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("i32 trellis overflow"))
}

/// src/kvq_fast.cuh:108-135. Lowest predecessor j, then lowest final state.
fn general_chain(target: &[i32; 64], table: &CodeTable, bits: u8) -> Result<[u8; 64]> {
    let fields = 1usize << bits;
    let stride = 256 >> bits;
    let mut cost = [0x4000_0000; 256];
    cost[0] = 0;
    let mut bp = [[0u8; 32]; 64];
    for (t, &x) in target.iter().enumerate() {
        let mut next = [0; 256];
        for u in 0..stride {
            let mut best = cost[u];
            let mut choice = 0;
            for j in 1..fields {
                if cost[j * stride + u] < best {
                    best = cost[j * stride + u];
                    choice = j;
                }
            }
            bp[t][u] = choice as u8;
            for j in 0..fields {
                let state = (u << bits) | j;
                let dx = sub(x, i32::from(table.codes[state]))?;
                next[state] = add(mul(dx, dx)?, best)?;
            }
        }
        cost = next;
    }
    let mut state = (0..256).min_by_key(|&s| (cost[s] as u32, s)).unwrap();
    let mut path = [0; 64];
    for t in (0..64).rev() {
        path[t] = state as u8;
        state = (usize::from(bp[t][state >> bits]) << (8 - bits)) | (state >> bits);
    }
    Ok(path)
}

/// Compressed cost recurrence, including every intermediate checked i32
/// product/add/subtract (src/kvq_vit2.cuh:64-88,143-192).
fn vit2_chain(target: &[i32; 64], table: &CodeTable, bits: u8) -> Result<[u8; 64]> {
    let fields = 1usize << bits;
    let stride = 256 >> bits;
    let f = fields as i32;
    let mut ac = [0; 256];
    let mut bc = [0; 256];
    for d in 0..256 {
        let c = i32::from(table.codes[d]);
        ac[d] = add(mul(mul(f, c)?, c)?, (d / stride) as i32)?;
        bc[d] = mul(mul(-2, f)?, c)?;
    }
    let mut cost = vec![f << 24; stride];
    cost[0] = 0;
    let mut bp = [[0u8; 16]; 64];
    let mut final_key = (i32::MAX, usize::MAX);
    for (t, &x) in target.iter().enumerate() {
        let mut next = vec![0; stride];
        let square = mul(mul(x, x)?, f)?; // checked multiplication preserves the source's << KK result
        for g in 0..stride {
            let mut best = i32::MAX;
            // KK=5 pairs the two Ac+Bc*X keys BEFORE adding their shared
            // reduced cost; KK=4 adds the cost directly (src/kvq_vit2.cuh:67-75).
            let step = if bits == 5 { 2 } else { 1 };
            for j in (0..fields).step_by(step) {
                let d = stride * j + g;
                let mut emission = add(ac[d], mul(bc[d], x)?)?;
                if bits == 5 {
                    let other = d + stride;
                    emission = emission.min(add(ac[other], mul(bc[other], x)?)?);
                }
                best = best.min(add(emission, cost[d >> bits])?);
            }
            let k = add(best, square)?;
            if t == 63 {
                let state = stride * (k as usize & (fields - 1)) + g;
                final_key = final_key.min((k >> bits, state));
            } else {
                let k = sub(k, cost[0])?;
                next[g] = k & !(f - 1);
                bp[t + 1][g] = (k & (f - 1)) as u8;
            }
        }
        if t < 63 {
            cost = next;
        }
    }
    let mut state = final_key.1;
    let mut path = [0; 64];
    for t in (1..64).rev() {
        path[t] = state as u8;
        state = (usize::from(bp[t][state >> bits]) << (8 - bits)) | (state >> bits);
    }
    path[0] = state as u8;
    Ok(path)
}

/// Warp XOR reductions descend 16,8,4,2,1; eight warp sums fold from +0
/// (src/kvq_fast.cuh:33-41). This is also the scale-search error fold.
fn warp_sum(mut v: [f32; 32]) -> f32 {
    for offset in [16, 8, 4, 2, 1] {
        let old = v;
        for i in 0..32 {
            v[i] = old[i] + old[i ^ offset];
        }
    }
    v[0]
}
fn sum256(v: &[f32; 256]) -> f32 {
    let mut total = 0.0;
    for chunk in v.chunks_exact(32) {
        total += warp_sum(chunk.try_into().unwrap());
    }
    total
}

fn initial_int8(row: &[f32; 256]) -> Result<Int8Row> {
    require_finite(row, "KV row")?;
    // src/kvx_codec.cuh:50-69, also used for the final int8 store.
    let maximum = row.iter().map(|x| x.abs()).fold(0.0, f32::max);
    let scale = maximum / 127.0;
    let integers = row
        .iter()
        .map(|&x| {
            if scale > 0.0 {
                (x / scale).round_ties_even().clamp(-127.0, 127.0) as i8
            } else {
                0
            }
        })
        .collect();
    Ok(Int8Row { integers, scale })
}

struct Selection {
    reconstruction: [f32; 256],
    states: [u8; 256],
    scale: f32,
}

fn select(
    profile: &KvProfile,
    group: usize,
    position: usize,
    row: &[f32; 256],
) -> Result<Selection> {
    let g = profile.group(group)?;
    let initial = initial_int8(row)?;
    // src/kvq_fast.cuh:78: profile input is dequantized initial int8.
    let y = std::array::from_fn(|i| f32::from(initial.integers[i]) * initial.scale);
    if position < usize::from(g.guard) {
        return Ok(Selection {
            reconstruction: y,
            states: [0; 256],
            scale: 1.0,
        });
    }
    let mut residual = y;
    if let Some(mean) = &g.mean {
        for i in 0..256 {
            residual[i] -= mean[i];
        }
    }
    let mut coefficients = [0.0; 2];
    // Sequential residual projection; coefficients rounded to half before use
    // (src/kvq_fast.cuh:89-93; src/kvq_gpu.cuh:72-81).
    for (j, basis) in g.basis.iter().enumerate() {
        coefficients[j] = half(sum256(&std::array::from_fn(|i| residual[i] * basis[i])));
        for i in 0..256 {
            residual[i] -= coefficients[j] * basis[i];
        }
    }
    require_finite(&residual, "KV residual preparation")?;
    let mut rms = sqrt_rn(sum256(&residual.map(|x| x * x)) / 256.0);
    if rms == 0.0 {
        rms = 1.0;
    }
    if !rms.is_finite() {
        return Err(invalid("non-finite KV RMS"));
    }
    let mut selected = Selection {
        reconstruction: residual,
        states: [0; 256],
        scale: 1.0,
    };
    match g.mode {
        0 => {
            let table = &profile.tables()[g.table];
            let vit2 = profile.vit2_eligible() && matches!(g.bits, 4 | 5);
            let mut best_error = None;
            for &multiplier in profile.multipliers() {
                // src/kvq_fast.cuh:165-168: no gain in integer target conversion.
                let mut scale = half(rms * multiplier);
                if scale == 0.0 {
                    scale = 1.0;
                }
                if !scale.is_finite() {
                    return Err(invalid("non-finite KV candidate scale"));
                }
                let targets = residual.map(|x| {
                    ((x / scale) * 256.0)
                        .round_ties_even()
                        .clamp(-2048.0, 2048.0) as i32
                });
                let mut states = [0; 256];
                for c in 0..4 {
                    let target = targets[c * 64..c * 64 + 64].try_into().unwrap();
                    let path = if vit2 {
                        vit2_chain(target, table, g.bits)?
                    } else {
                        general_chain(target, table, g.bits)?
                    };
                    states[c * 64..c * 64 + 64].copy_from_slice(&path);
                }
                let rec = states.map(|s| table.reconstruction(s) * scale);
                let error = sum256(&std::array::from_fn(|i| {
                    let d = residual[i] - rec[i];
                    d * d
                }));
                if !error.is_finite() {
                    return Err(invalid("non-finite KV reconstruction error"));
                }
                // First strict minimum in multiplier order (src/kvq_fast.cuh:198-211).
                if best_error.is_none_or(|e| error < e) {
                    best_error = Some(error);
                    selected = Selection {
                        reconstruction: rec,
                        states,
                        scale,
                    };
                }
            }
        }
        1 => {
            // src/kvq_fast.cuh:192-196: scale is FP32, not rounded to half.
            let mut scale = residual.iter().map(|x| x.abs()).fold(0.0, f32::max) / 7.0;
            if scale == 0.0 {
                scale = 1.0;
            }
            selected.reconstruction =
                residual.map(|x| (x / scale).round_ties_even().clamp(-7.0, 7.0) * scale);
            selected.scale = scale;
        }
        _ => {} // src/kvq_fast.cuh:191-217: unchanged residual, then restore.
    }
    for (j, basis) in g.basis.iter().enumerate() {
        for (i, x) in selected.reconstruction.iter_mut().enumerate() {
            *x += coefficients[j] * basis[i];
        }
    }
    if let Some(mean) = &g.mean {
        for (x, m) in selected.reconstruction.iter_mut().zip(mean) {
            *x += m;
        }
    }
    require_finite(&selected.reconstruction, "KV reconstruction")?;
    Ok(selected)
}

/// Upstream TB_CAP=0: initial int8, profile quantization, final int8.
pub fn requantize_int8(
    profile: &KvProfile,
    group: usize,
    position: usize,
    row: &[f32; 256],
) -> Result<Int8Row> {
    initial_int8(&select(profile, group, position, row)?.reconstruction)
}

/// Upstream TB_CAP=1: a whole layer/side must be packable; otherwise int8.
pub fn encode_packed(
    profile: &KvProfile,
    group: usize,
    position: usize,
    row: &[f32; 256],
) -> Result<CacheRow> {
    let g = profile.group(group)?;
    let selected = select(profile, group, position, row)?;
    if !profile.packable(group / (profile.kv_heads() * 2), group % 2) {
        return Ok(CacheRow::Int8(initial_int8(&selected.reconstruction)?));
    }
    let mask = (1 << g.bits) - 1;
    let fields = selected.states.map(|s| s & mask);
    let mut packed = PackedRow {
        words: pack(&fields, g.bits),
        bits: g.bits,
        scale: selected.scale,
        bases: Box::new([0.0; 256]),
    };
    *packed.bases = decode_packed(profile, group, &packed)?;
    Ok(CacheRow::Packed(packed))
}

// src/kvq_fast.cuh:226-229: MSB-first fields in native (LE) u32 words.
fn pack(fields: &[u8; 256], bits: u8) -> Vec<u32> {
    let mut words = vec![0; 8 * usize::from(bits)];
    for (e, &field) in fields.iter().enumerate() {
        for j in 0..usize::from(bits) {
            let bit = e * usize::from(bits) + j;
            words[bit / 32] |=
                u32::from((field >> (usize::from(bits) - 1 - j)) & 1) << (31 - bit % 32);
        }
    }
    words
}
fn field(row: &PackedRow, channel: usize) -> u8 {
    let mut result = 0;
    for j in 0..usize::from(row.bits) {
        let bit = channel * usize::from(row.bits) + j;
        result = (result << 1) | ((row.words[bit / 32] >> (31 - bit % 32)) & 1) as u8;
    }
    result
}
fn packed_state(row: &PackedRow, channel: usize) -> u8 {
    // src/q35_atxpk.cu:69-74: only the last eight stream bits, reset each chain.
    let first = (channel / 64 * 64).max(channel.saturating_sub(7 / usize::from(row.bits)));
    let mut state = 0u8;
    for i in first..=channel {
        state = (state << row.bits) | field(row, i);
    }
    state
}
fn validate_packed<'a>(profile: &'a KvProfile, group: usize, row: &PackedRow) -> Result<&'a Group> {
    let g = profile.group(group)?;
    if !profile.packable(group / (profile.kv_heads() * 2), group % 2)
        || row.bits != g.bits
        || row.words.len() != usize::from(row.bits) * 8
    {
        return Err(invalid("packed row/profile mismatch"));
    }
    Ok(g)
}

/// Per-channel BF16 base operands, without scale and without mean, as read by
/// attention (src/kvq_atxpk.hpp:39-42; src/q35_atxpk3.cuh:427-430,454).
pub fn decode_packed(profile: &KvProfile, group: usize, row: &PackedRow) -> Result<[f32; 256]> {
    let g = validate_packed(profile, group, row)?;
    let table = &profile.tables()[g.table];
    Ok(std::array::from_fn(|i| {
        bfloat(table.reconstruction(packed_state(row, i)))
    }))
}

#[cfg(test)]
#[path = "kv_codec_tests.rs"]
mod tests;
