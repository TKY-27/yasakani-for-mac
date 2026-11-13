//! IEEE storage conversions for [FORMAT.md §3](../../../docs/FORMAT.md#3-the-header).

/// Widen binary16 exactly, including subnormal values and signed zero.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let widened = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((113 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(widened)
}

/// Round binary32 to binary16, nearest with ties to even.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;
    if exponent == 255 {
        return sign | 0x7c00 | if fraction == 0 { 0 } else { 0x0200 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent < -10 {
        return sign;
    }
    if half_exponent <= 0 {
        let significand = fraction | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        // Keep the discarded bits so a subnormal midpoint chooses an even code.
        return sign | round_shift_even(significand, shift) as u16;
    }
    // Rounding the combined exponent/significand lets a carry promote the exponent.
    let magnitude = ((half_exponent as u32) << 10) + round_shift_even(fraction, 13);
    sign | magnitude as u16
}

fn round_shift_even(value: u32, shift: u32) -> u32 {
    let retained = value >> shift;
    let discarded = value & ((1_u32 << shift) - 1);
    let midpoint = 1_u32 << (shift - 1);
    retained + u32::from(discarded > midpoint || (discarded == midpoint && retained & 1 != 0))
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Round binary32 to bfloat16, nearest with ties to even.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    // The retained low bit distinguishes the two possible midpoint directions.
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_finite_binary16_value_roundtrips() {
        for bits in 0..=u16::MAX {
            if bits & 0x7c00 != 0x7c00 {
                assert_eq!(f32_to_f16(f16_to_f32(bits)), bits);
            }
        }
    }

    #[test]
    fn every_finite_bfloat16_value_roundtrips() {
        for bits in 0..=u16::MAX {
            if bits & 0x7f80 != 0x7f80 {
                assert_eq!(f32_to_bf16(bf16_to_f32(bits)), bits);
            }
        }
    }

    #[test]
    fn binary16_midpoints_choose_even_including_subnormal_boundary() {
        for lower in 0..0x7bff {
            let midpoint = (f16_to_f32(lower) + f16_to_f32(lower + 1)) * 0.5;
            let expected = lower + (lower & 1);
            assert_eq!(f32_to_f16(midpoint), expected);
            assert_eq!(f32_to_f16(-midpoint), expected | 0x8000);
        }
        assert_eq!(f32_to_f16(65_520.0), 0x7c00);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
    }

    #[test]
    fn bfloat16_midpoints_choose_even() {
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x0000_8000)), 0);
        assert_eq!(f32_to_bf16(f32::from_bits(0x0001_8000)), 2);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
    }

    #[test]
    fn infinities_and_nans_keep_their_categories() {
        for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let half = f16_to_f32(f32_to_f16(value));
            let bfloat = bf16_to_f32(f32_to_bf16(value));
            if value.is_nan() {
                assert!(half.is_nan() && bfloat.is_nan());
            } else {
                assert_eq!(half, value);
                assert_eq!(bfloat, value);
            }
        }
    }
}
