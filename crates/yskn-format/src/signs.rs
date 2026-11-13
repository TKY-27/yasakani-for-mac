//! Rotation sign streams from [FORMAT.md §4.5](../../../docs/FORMAT.md#45-signs).

use crate::{Result, allocate};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignRule {
    SplitMix64,
    Mulberry32,
}

/// Generate signs using the format's wrapping integer arithmetic.
pub fn signs(rule: SignRule, seed: u64, offset: u64, count: usize) -> Result<Vec<i8>> {
    let mut result = allocate(count)?;
    for index in 0..count {
        let position = offset.wrapping_add(index as u64).wrapping_add(1);
        let positive = match rule {
            SignRule::SplitMix64 => {
                let mut value = seed.wrapping_add(position.wrapping_mul(0x9e37_79b9_7f4a_7c15));
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^= value >> 31;
                value & (1_u64 << 63) != 0
            }
            SignRule::Mulberry32 => {
                let mut initial = (seed as u32).wrapping_mul(7919).wrapping_add(13);
                if initial == 0 {
                    initial = 0x9e37_79b9;
                }
                let mut value = initial.wrapping_add((position as u32).wrapping_mul(0x6d2b_79f5));
                value = (value ^ (value >> 15)).wrapping_mul(value | 1);
                value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
                value ^= value >> 14;
                value >= 1_u32 << 31
            }
        };
        result.push(if positive { 1 } else { -1 });
    }
    Ok(result)
}
