// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//! Portable arithmetic for [chapters 02–06](../../../docs/spec/02_COMPRESSED_LINEAR.md).
//!
//! The polynomial operations and bit constants follow `src/pm_math.cuh`.
//! Separate Rust arithmetic operations preserve each `_rn` rounding boundary;
//! do not contract multiply/add pairs or replace these functions with libm.

use crate::{Error, Result};

/// Round to binary16 and widen; activation quantization observes this rounding.
pub fn half(value: f32) -> f32 {
    yskn_format::f16_to_f32(yskn_format::f32_to_f16(value))
}

/// Round to BF16 and widen at the model's explicit storage boundaries.
pub fn bfloat(value: f32) -> f32 {
    yskn_format::bf16_to_f32(yskn_format::f32_to_bf16(value))
}

/// Elementwise engine hooks. Default bodies are the scalar numeric contract.
/// Implementations may only reorder work between independent elements.
pub trait Elementwise {
    fn bfloat(values: &mut [f32]) {
        for value in values {
            *value = bfloat(*value);
        }
    }
    fn sigmoid(values: &mut [f32]) {
        for value in values {
            *value = sigmoid(*value);
        }
    }
    fn silu(values: &mut [f32]) {
        for value in values {
            *value = silu(*value);
        }
    }
    fn scale_half(values: &mut [f32], scale: f32) {
        for value in values {
            *value = half(*value * scale);
        }
    }
    fn finite(values: &[f32], context: &'static str) -> Result<()> {
        crate::require_finite(values, context)
    }
}

pub struct ScalarMath;
impl Elementwise for ScalarMath {}

const fn single(bits: u32) -> f32 {
    f32::from_bits(bits)
}

const fn double(high: u32, low: u32) -> f64 {
    f64::from_bits(((high as u64) << 32) | low as u64)
}

/// The correctly rounded binary32 square-root primitive used by `pm_rsqrt`.
pub fn sqrt_rn(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff == 0 || bits == 0x7f80_0000 {
        return value;
    }
    if bits & 0x8000_0000 != 0 || bits & 0x7f80_0000 == 0x7f80_0000 {
        return f32::NAN;
    }

    let exponent_bits = ((bits >> 23) & 255) as i32;
    let (significand, exponent) = if exponent_bits == 0 {
        (u64::from(bits & 0x007f_ffff), -149)
    } else {
        (
            u64::from((bits & 0x007f_ffff) | 0x0080_0000),
            exponent_bits - 150,
        )
    };
    let leading_exponent = 63 - significand.leading_zeros() as i32 + exponent;
    let mut result_exponent = leading_exponent.div_euclid(2);
    let shift = exponent - 2 * result_exponent + 46;
    let radicand = significand << shift;

    // The normalized radicand has at most 48 bits. Integer comparisons make
    // the square-root rounding independent of a platform math implementation.
    let mut lower = 1_u64 << 23;
    let mut upper = 1_u64 << 24;
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2;
        if middle * middle <= radicand {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    // (lower + 1/2)^2 = lower^2 + lower + 1/4. An integer radicand
    // cannot be exactly halfway, so this is the exact nearest-even decision.
    if radicand - lower * lower > lower {
        lower += 1;
    }
    if lower == 1 << 24 {
        lower >>= 1;
        result_exponent += 1;
    }
    single((((result_exponent + 127) as u32) << 23) | (lower as u32 & 0x007f_ffff))
}

pub fn rsqrt(value: f32) -> f32 {
    // Upstream rounds the square root before a separately rounded division.
    1.0 / sqrt_rn(value)
}

pub fn exp(value: f32) -> f32 {
    let bits = value.to_bits();
    let absolute_bits = bits & 0x7fff_ffff;
    let negative = bits >> 31 != 0;
    if absolute_bits > 0x7f80_0000 {
        return value + value;
    }
    if !negative && absolute_bits >= 0x42b1_7218 {
        return f32::INFINITY;
    }
    if bits > 0xc2cf_f1b5 {
        return 0.0;
    }
    let ln2_high = single(0x3f31_7200);
    let ln2_low = single(0x35bf_be8e);
    let mut high = 0.0;
    let mut low = 0.0;
    let mut reduced = value;
    let mut exponent = 0;
    if absolute_bits > 0x3eb1_7218 {
        if absolute_bits < 0x3f85_1592 {
            high = value - if negative { -ln2_high } else { ln2_high };
            low = if negative { -ln2_low } else { ln2_low };
            exponent = if negative { -1 } else { 1 };
        } else {
            exponent = (single(0x3fb8_aa3b) * value + if negative { -0.5 } else { 0.5 }) as i32;
            let exponent_float = exponent as f32;
            high = value - exponent_float * ln2_high;
            low = exponent_float * ln2_low;
        }
        reduced = high - low;
    } else if absolute_bits < 0x3900_0000 {
        return 1.0 + value;
    }
    let square = reduced * reduced;
    let correction = reduced - square * (single(0x3e2a_aa8f) + square * single(0xbb35_5215));
    if exponent == 0 {
        return 1.0 - ((reduced * correction) / (correction - 2.0) - reduced);
    }
    let result = 1.0 - ((low - (reduced * correction) / (2.0 - correction)) - high);
    if exponent >= -125 {
        if exponent == 128 {
            return (result * 2.0) * single(0x7f00_0000);
        }
        return result * single(((127 + exponent) as u32) << 23);
    }
    (result * single(((127 + exponent + 100) as u32) << 23)) * single(0x0d80_0000)
}

/// `pm_logf`; its caller must supply a positive finite value.
pub fn log(mut value: f32) -> f32 {
    let mut bits = value.to_bits();
    let mut exponent = 0;
    if bits < 0x0080_0000 {
        value *= 33_554_432.0;
        exponent = -25;
        bits = value.to_bits();
    }
    exponent += (bits >> 23) as i32 - 127;
    bits &= 0x007f_ffff;
    let adjustment = (bits + 0x004a_fb20) & 0x0080_0000;
    let mantissa = single(bits | (adjustment ^ 0x3f80_0000));
    exponent += (adjustment >> 23) as i32;
    let fraction = mantissa - 1.0;
    let ratio = fraction / (2.0 + fraction);
    let square = ratio * ratio;
    let fourth = square * square;
    let even = fourth * (single(0x3ecc_ce13) + fourth * single(0x3e78_9e26));
    let odd = square * (single(0x3f2a_aaaa) + fourth * single(0x3e91_e9ee));
    let remainder = odd + even;
    let half_square = (0.5 * fraction) * fraction;
    let exponent_float = exponent as f32;
    exponent_float * single(0x3f31_7180)
        - ((half_square
            - (ratio * (half_square + remainder) + exponent_float * single(0x3717_f7d1)))
            - fraction)
}

/// `pm_log1p_pos`; only the nonnegative domain is specified upstream.
pub fn log1p(value: f32) -> f32 {
    let shifted = 1.0 + value;
    let correction = if value <= 1.0 {
        value - (shifted - 1.0)
    } else {
        1.0 - (shifted - value)
    };
    log(shifted) + correction / shifted
}

pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + exp(-value))
}

pub fn silu(value: f32) -> f32 {
    value / (1.0 + exp(-value))
}

pub fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        log1p(exp(value))
    }
}

/// D7: round the product with binary32 ln(2), then use the portable exp.
pub fn exp2(value: f32) -> f32 {
    let argument = value * f32::from_bits(0x3f31_7218);
    exp(argument)
}

pub fn exp_d(value: f64) -> f64 {
    let bits = value.to_bits();
    let high_bits = (bits >> 32) as u32;
    let low_bits = bits as u32;
    let negative = high_bits >> 31 != 0;
    let absolute_high = high_bits & 0x7fff_ffff;
    if absolute_high >= 0x4086_2e42 {
        if absolute_high >= 0x7ff0_0000 {
            if (absolute_high & 0x000f_ffff) | low_bits != 0 {
                return value + value;
            }
            return if negative { 0.0 } else { value };
        }
        if value > double(0x4086_2e42, 0xfefa_39ef) {
            return f64::INFINITY;
        }
        if value < double(0xc087_4910, 0xd52d_3051) {
            return 0.0;
        }
    }
    let ln2_high = double(0x3fe6_2e42, 0xfee0_0000);
    let ln2_low = double(0x3dea_39ef, 0x3579_3c76);
    let mut high = 0.0;
    let mut low = 0.0;
    let mut reduced = value;
    let mut exponent = 0;
    if absolute_high > 0x3fd6_2e42 {
        if absolute_high < 0x3ff0_a2b2 {
            high = value - if negative { -ln2_high } else { ln2_high };
            low = if negative { -ln2_low } else { ln2_low };
            exponent = if negative { -1 } else { 1 };
        } else {
            exponent = (double(0x3ff7_1547, 0x652b_82fe) * value
                + if negative { -0.5 } else { 0.5 }) as i32;
            let exponent_float = exponent as f64;
            high = value - exponent_float * ln2_high;
            low = exponent_float * ln2_low;
        }
        reduced = high - low;
    } else if absolute_high < 0x3e30_0000 {
        return 1.0 + value;
    }
    let square = reduced * reduced;
    // Keep this Horner nesting: each product and sum has its own RN boundary.
    let polynomial = double(0x3fc5_5555, 0x5555_553e)
        + square
            * (double(0xbf66_c16c, 0x16be_bd93)
                + square
                    * (double(0x3f11_566a, 0xaf25_de2c)
                        + square
                            * (double(0xbebb_bd41, 0xc5d2_6bf1)
                                + square * double(0x3e66_3769, 0x72be_a4d0))));
    let correction = reduced - square * polynomial;
    if exponent == 0 {
        return 1.0 - ((reduced * correction) / (correction - 2.0) - reduced);
    }
    let result = 1.0 - ((low - (reduced * correction) / (2.0 - correction)) - high);
    if exponent >= -1021 {
        return scale_double_exponent(result, exponent);
    }
    scale_double_exponent(result, exponent + 1000) * double(0x0170_0000, 0)
}

fn scale_double_exponent(value: f64, adjustment: i32) -> f64 {
    let bits = value.to_bits();
    let high = (bits >> 32) as u32;
    // This is upstream's high-word exponent adjustment, not a floating multiply.
    let adjusted_high = high.wrapping_add((adjustment as u32).wrapping_mul(1_048_576));
    double(adjusted_high, bits as u32)
}

/// `pm_log_d`, including its source-defined bit operations at zero.
///
/// The source has no mathematical zero special case: +0 and -0 yield finite
/// artifacts. Keep that behavior for the sampler's rare unit-uniform endpoint.
pub fn log_d(mut value: f64) -> f64 {
    let mut high = (value.to_bits() >> 32) as i32;
    let mut exponent = 0;
    if high < 0x0010_0000 {
        value *= double(0x4350_0000, 0);
        exponent = -54;
        high = (value.to_bits() >> 32) as i32;
    }
    exponent += (high >> 20) - 1023;
    high &= 0x000f_ffff;
    let adjustment = (high + 0x0009_5f64) & 0x0010_0000;
    value = double(
        (high | (adjustment ^ 0x3ff0_0000)) as u32,
        value.to_bits() as u32,
    );
    exponent += adjustment >> 20;
    let ln2_high = double(0x3fe6_2e42, 0xfee0_0000);
    let ln2_low = double(0x3dea_39ef, 0x3579_3c76);
    let fraction = value - 1.0;
    if (0x000f_ffff & (2 + high)) < 3 {
        if fraction == 0.0 {
            if exponent == 0 {
                return 0.0;
            }
            let exponent_float = exponent as f64;
            return exponent_float * ln2_high + exponent_float * ln2_low;
        }
        let remainder = (fraction * fraction) * (0.5 - double(0x3fd5_5555, 0x5555_5555) * fraction);
        if exponent == 0 {
            return fraction - remainder;
        }
        let exponent_float = exponent as f64;
        return exponent_float * ln2_high - ((remainder - exponent_float * ln2_low) - fraction);
    }
    let ratio = fraction / (2.0 + fraction);
    let exponent_float = exponent as f64;
    let square = ratio * ratio;
    let fourth = square * square;
    let interval = (high - 0x0006_147a) | (0x0006_b851 - high);
    let even = fourth
        * (double(0x3fd9_9999, 0x9997_fa04)
            + fourth
                * (double(0x3fcc_71c5, 0x1d8e_78af) + fourth * double(0x3fc3_9a09, 0xd078_c69f)));
    let odd = square
        * (double(0x3fe5_5555, 0x5555_5593)
            + fourth
                * (double(0x3fd2_4924, 0x9422_9359)
                    + fourth
                        * (double(0x3fc7_4664, 0x96cb_03de)
                            + fourth * double(0x3fc2_f112, 0xdf3e_5244))));
    let remainder = odd + even;
    if interval > 0 {
        let half_square = (0.5 * fraction) * fraction;
        if exponent == 0 {
            return fraction - (half_square - ratio * (half_square + remainder));
        }
        return exponent_float * ln2_high
            - ((half_square - (ratio * (half_square + remainder) + exponent_float * ln2_low))
                - fraction);
    }
    if exponent == 0 {
        return fraction - ratio * (fraction - remainder);
    }
    exponent_float * ln2_high
        - ((ratio * (fraction - remainder) - exponent_float * ln2_low) - fraction)
}

fn kernel_sin(reduced: f64) -> f64 {
    let square = reduced * reduced;
    let cube = square * reduced;
    let remainder = double(0x3f81_1111, 0x1110_f8a6)
        + square
            * (double(0xbf2a_01a0, 0x19c1_61d5)
                + square
                    * (double(0x3ec7_1de3, 0x57b1_fe7d)
                        + square
                            * (double(0xbe5a_e5e6, 0x8a2b_9ceb)
                                + square * double(0x3de5_d93a, 0x5acf_d57c))));
    reduced + cube * (double(0xbfc5_5555, 0x5555_5549) + square * remainder)
}

fn kernel_cos(reduced: f64) -> f64 {
    let square = reduced * reduced;
    let remainder = square
        * (double(0x3fa5_5555, 0x5555_554c)
            + square
                * (double(0xbf56_c16c, 0x16c1_5177)
                    + square
                        * (double(0x3efa_01a0, 0x19cb_1590)
                            + square
                                * (double(0xbe92_7e4f, 0x809c_52ad)
                                    + square
                                        * (double(0x3e21_ee9e, 0xbdb4_b1c4)
                                            + square * double(0xbda8_fae9, 0xbe88_38d4))))));
    let absolute_high = (reduced.to_bits() >> 32) as u32 & 0x7fff_ffff;
    let correction = square * remainder;
    if absolute_high < 0x3fd3_3333 {
        return 1.0 - (0.5 * square - correction);
    }
    let quarter = if absolute_high > 0x3fe9_0000 {
        0.28125
    } else {
        double(absolute_high - 0x0020_0000, 0)
    };
    (1.0 - quarter) - ((0.5 * square - quarter) - correction)
}

/// Return `(cosine, sine)`, preserving `pm_sincos_f` argument reduction.
pub fn sincos(angle: f32) -> Result<(f32, f32)> {
    if !angle.is_finite() {
        return Err(Error::NonFinite("rotary angle"));
    }
    let value = f64::from(angle);
    let quadrant_float = value * double(0x3fe4_5f30, 0x6dc9_c883) + 0.5;
    if !(-2_147_483_648.0..2_147_483_648.0).contains(&quadrant_float) {
        return Err(Error::InvalidModel(
            "rotary angle exceeds the portable reduction's i32 domain".into(),
        ));
    }
    let quadrant = quadrant_float as i32;
    let quadrant_float = f64::from(quadrant);
    let reduced = ((value - quadrant_float * double(0x3ff9_21fb, 0x5440_0000))
        - quadrant_float * double(0x3dd0_b461, 0x1a60_0000))
        - quadrant_float * double(0x3ba3_198a, 0x2e03_7073);
    let sine = kernel_sin(reduced);
    let cosine = kernel_cos(reduced);
    let (cosine, sine) = match quadrant & 3 {
        0 => (cosine, sine),
        1 => (-sine, cosine),
        2 => (-cosine, -sine),
        _ => (sine, -cosine),
    };
    // Upstream narrows only after the complete binary64 polynomial/reduction.
    Ok((cosine as f32, sine as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp2_pins_output_bits() {
        // Captured from the current D7 implementation, including subnormal rounding.
        for (value, expected) in [
            (1.0, 0x4000_0000),
            (-1.0, 0x3f00_0000),
            (0.5, 0x3fb5_04f3),
            (-30.0, 0x3080_0008),
            (-126.0, 0x007f_ffe6),
            (-150.0, 0x0000_0001),
            (-1.44e30, 0x0000_0000),
            (f32::NEG_INFINITY, 0x0000_0000),
        ] {
            assert_eq!(exp2(value).to_bits(), expected, "input {value}");
        }
        assert!(exp2(f32::NAN).is_nan());
    }

    #[test]
    fn exp2_uses_d7_binary32_product_and_portable_exp() {
        for value in [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.0e-7, -1.0e-7, 30.0, -30.0, -100.0,
        ] {
            let expected = exp(value * f32::from_bits(0x3f31_7218));
            assert_eq!(exp2(value).to_bits(), expected.to_bits(), "input {value}");
        }
        assert_eq!(exp2(0.0).to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn square_root_handles_normal_subnormal_and_rounding_boundaries() {
        let cases = [
            (0x0000_0000, 0x0000_0000),
            (0x8000_0000, 0x8000_0000),
            (0x3f80_0000, 0x3f80_0000),
            (0x4000_0000, 0x3fb5_04f3),
            (0x4080_0000, 0x4000_0000),
            (0x0000_0001, 0x1a35_04f3),
            (0x0080_0000, 0x2000_0000),
            (0x7f7f_ffff, 0x5f7f_ffff),
        ];
        for (input, expected) in cases {
            assert_eq!(sqrt_rn(single(input)).to_bits(), expected);
        }
        assert_eq!(rsqrt(4.0).to_bits(), 0.5_f32.to_bits());
        assert!(sqrt_rn(-1.0).is_nan());
    }

    #[test]
    fn portable_math_has_exact_identity_points() {
        assert_eq!(exp(0.0).to_bits(), 1.0_f32.to_bits());
        assert_eq!(log(1.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(log1p(0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(sigmoid(0.0).to_bits(), 0.5_f32.to_bits());
        assert_eq!(silu(0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(softplus(21.0).to_bits(), 21.0_f32.to_bits());
        assert_eq!(exp_d(0.0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(log_d(1.0).to_bits(), 0.0_f64.to_bits());
        let (cosine, sine) = sincos(0.0).unwrap();
        assert_eq!(cosine.to_bits(), 1.0_f32.to_bits());
        assert_eq!(sine.to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn portable_exp_preserves_underflow_and_overflow_rules() {
        assert_eq!(exp(-104.0).to_bits(), 0);
        assert_eq!(exp(89.0), f32::INFINITY);
        assert!(exp(-103.0) > 0.0);
        assert_eq!(exp_d(-746.0).to_bits(), 0);
        assert_eq!(exp_d(710.0), f64::INFINITY);
        assert!(sincos(f32::MAX).is_err());
    }

    #[test]
    fn polynomial_bit_patterns_match_the_portable_source() {
        // Evaluated from the unchanged permitted pm_math.cuh with scalar RN
        // primitives and contraction disabled; these are not CUDA golden bits.
        let cases = [
            (
                0.5,
                [
                    0x3fd3_094c,
                    0xbf31_7218,
                    0x3ecf_991f,
                    0x3f1f_597f,
                    0x3e9f_597f,
                    0x3f79_5d1c,
                ],
            ),
            (
                1.0,
                [
                    0x402d_f854,
                    0x0000_0000,
                    0x3f31_7218,
                    0x3f3b_26a8,
                    0x3f3b_26a8,
                    0x3fa8_18f5,
                ],
            ),
            (
                10.0,
                [
                    0x46ac_14ee,
                    0x4013_5d8e,
                    0x4019_771e,
                    0x3f7f_fd06,
                    0x411f_fe24,
                    0x4120_0030,
                ],
            ),
        ];
        for (input, expected) in cases {
            let actual = [
                exp(input),
                log(input),
                log1p(input),
                sigmoid(input),
                silu(input),
                softplus(input),
            ];
            assert_eq!(actual.map(f32::to_bits), expected);
        }
        assert_eq!(exp(-102.0).to_bits(), 4);
        let double_cases = [
            (0.5, 0x3ffa_6129_8e1e_069c, 0xbfe6_2e42_fefa_39ef),
            (1.0, 0x4005_bf0a_8b14_576a, 0),
            (10.0, 0x40d5_829d_cf95_0560, 0x4002_6bb1_bbb5_5516),
        ];
        for (input, exponential, logarithm) in double_cases {
            assert_eq!(exp_d(input).to_bits(), exponential);
            assert_eq!(log_d(input).to_bits(), logarithm);
        }
        for (input, cosine, sine) in [
            (1.0, 0x3f0a_5140, 0x3f57_6aa4),
            (20.0, 0x3ed0_f022, 0x3f69_b6c8),
            (1_000_000.0, 0x3f6f_cefd, 0xbeb3_3259),
        ] {
            let (actual_cosine, actual_sine) = sincos(input).unwrap();
            assert_eq!(actual_cosine.to_bits(), cosine);
            assert_eq!(actual_sine.to_bits(), sine);
        }
    }

    #[test]
    fn portable_log_keeps_the_upstream_zero_endpoint_behavior() {
        assert_eq!(log_d(0.0).to_bits(), 0xc087_5427_f6ac_ad6f);
        assert_eq!(log_d(-0.0).to_bits(), 0xc0a0_ec2b_7d28_4853);
    }
}
