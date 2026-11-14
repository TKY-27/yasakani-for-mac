//! Portable algorithms across independent lanes; no native transcendental approximations.
use yskn_ref::portable_math::{Elementwise, ScalarMath};
pub(crate) struct FastMath;
impl Elementwise for FastMath {
    fn bfloat(values: &mut [f32]) {
        map(values, Operation::Bfloat);
    }
    fn sigmoid(values: &mut [f32]) {
        map(values, Operation::Sigmoid);
    }
    fn silu(values: &mut [f32]) {
        map(values, Operation::Silu);
    }
    fn scale_half(values: &mut [f32], scale: f32) {
        map(values, Operation::Half(scale));
    }
    fn finite(values: &[f32], context: &'static str) -> crate::Result<()> {
        if values.len() < 16 {
            return ScalarMath::finite(values, context);
        }
        #[cfg(target_arch = "aarch64")]
        if super::enabled() {
            // SAFETY: runtime NEON detection; the kernel handles the scalar tail.
            if unsafe { all_finite(values) } {
                return Ok(());
            }
            return Err(crate::Error::NonFinite(context));
        }
        ScalarMath::finite(values, context)
    }
}
pub(crate) fn gated_product4(gate: [f32; 4], up: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: gated_product calls this only after runtime NEON detection;
        // both independent input arrays and the result contain four floats.
        unsafe { gated_product4_neon(gate, up) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        use yskn_ref::portable_math::{bfloat, silu};
        std::array::from_fn(|i| bfloat(bfloat(silu(bfloat(gate[i]))) * bfloat(up[i])))
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gated_product4_neon(gate: [f32; 4], up: [f32; 4]) -> [f32; 4] {
    let mut output = [0.0; 4];
    // SAFETY: four-element input/output arrays; no multiply-add contraction.
    unsafe {
        let g = bfloat4(vld1q_f32(gate.as_ptr()));
        let activation = bfloat4(vdivq_f32(
            g,
            vaddq_f32(vdupq_n_f32(1.0), exp4(vnegq_f32(g))),
        ));
        let value = bfloat4(vld1q_f32(up.as_ptr()));
        vst1q_f32(output.as_mut_ptr(), bfloat4(vmulq_f32(activation, value)));
    }
    output
}

pub(crate) fn exp2(values: &mut [f32]) {
    map(values, Operation::Exp2);
}

#[derive(Clone, Copy)]
enum Operation {
    Bfloat,
    Silu,
    Sigmoid,
    Exp2,
    Half(f32),
}
fn map(values: &mut [f32], operation: Operation) {
    #[cfg(target_arch = "aarch64")]
    if super::enabled() {
        // SAFETY: runtime NEON detection and exclusive ownership of the slice.
        unsafe { map_neon(values, operation) };
        return;
    }
    scalar(values, operation);
}
fn scalar(values: &mut [f32], operation: Operation) {
    match operation {
        Operation::Bfloat => ScalarMath::bfloat(values),
        Operation::Silu => ScalarMath::silu(values),
        Operation::Sigmoid => ScalarMath::sigmoid(values),
        Operation::Exp2 => {
            for value in values {
                *value = yskn_ref::portable_math::exp2(*value);
            }
        }
        Operation::Half(scale) => ScalarMath::scale_half(values, scale),
    }
}
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn all_finite(values: &[f32]) -> bool {
    let mut bad = vdupq_n_u32(0);
    // SAFETY: chunks and tail are within the input slice.
    unsafe {
        for chunk in values.as_chunks::<4>().0 {
            let bits = vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(chunk.as_ptr())),
                vdupq_n_u32(0x7fffffff),
            );
            bad = vorrq_u32(bad, vcgeq_u32(bits, vdupq_n_u32(0x7f800000)));
        }
    }
    vmaxvq_u32(bad) == 0 && values[values.len() / 4 * 4..].iter().all(|x| x.is_finite())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn bfloat4(value: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_u32_f32(value);
    let rounded = vandq_u32(
        vaddq_u32(
            bits,
            vaddq_u32(
                vdupq_n_u32(0x7fff),
                vandq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(1)),
            ),
        ),
        vdupq_n_u32(0xffff0000),
    );
    let nan = vcgtq_u32(
        vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
        vdupq_n_u32(0x7f800000),
    );
    let quiet = vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0xffff0000)),
        vdupq_n_u32(0x00400000),
    );
    vreinterpretq_f32_u32(vbslq_u32(nan, quiet, rounded))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn half4(value: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_u32_f32(value);
    let sign = vandq_u32(bits, vdupq_n_u32(0x80000000));
    let exponent = vandq_u32(vshrq_n_u32::<23>(bits), vdupq_n_u32(255));
    // Normal half inputs only need 13-bit significand rounding; keep the
    // full integer conversion below for mixed/subnormal/exceptional vectors.
    if vminvq_u32(exponent) >= 113 && vmaxvq_u32(exponent) <= 142 {
        let magnitude = vandq_u32(bits, vdupq_n_u32(0x7fffffff));
        let bias = vaddq_u32(
            vdupq_n_u32(0xfff),
            vandq_u32(vshrq_n_u32::<13>(magnitude), vdupq_n_u32(1)),
        );
        let rounded = vandq_u32(vaddq_u32(magnitude, bias), vdupq_n_u32(0xffffe000));
        let rounded = vbslq_u32(
            vcgeq_u32(rounded, vdupq_n_u32(0x47800000)),
            vdupq_n_u32(0x7f800000),
            rounded,
        );
        return vreinterpretq_f32_u32(vorrq_u32(sign, rounded));
    }
    let fraction = vandq_u32(bits, vdupq_n_u32(0x007fffff));
    let rounded = vshrq_n_u32::<13>(vaddq_u32(
        fraction,
        vaddq_u32(
            vdupq_n_u32(0xfff),
            vandq_u32(vshrq_n_u32::<13>(fraction), vdupq_n_u32(1)),
        ),
    ));
    let normal = vaddq_u32(
        vshlq_n_u32::<10>(vsubq_u32(exponent, vdupq_n_u32(112))),
        rounded,
    );
    let shift = vminq_s32(
        vmaxq_s32(
            vsubq_s32(vdupq_n_s32(126), vreinterpretq_s32_u32(exponent)),
            vdupq_n_s32(14),
        ),
        vdupq_n_s32(24),
    );
    let sig = vorrq_u32(fraction, vdupq_n_u32(0x800000));
    let retained = vshlq_u32(sig, vnegq_s32(shift));
    let bias = vsubq_u32(
        vshlq_u32(vdupq_n_u32(1), vsubq_s32(shift, vdupq_n_s32(1))),
        vdupq_n_u32(1),
    );
    let subnormal = vshlq_u32(
        vaddq_u32(sig, vaddq_u32(bias, vandq_u32(retained, vdupq_n_u32(1)))),
        vnegq_s32(shift),
    );
    let mut half = vbslq_u32(vcleq_u32(exponent, vdupq_n_u32(112)), subnormal, normal);
    half = vbslq_u32(vcltq_u32(exponent, vdupq_n_u32(102)), vdupq_n_u32(0), half);
    half = vbslq_u32(
        vcgeq_u32(exponent, vdupq_n_u32(143)),
        vdupq_n_u32(0x7c00),
        half,
    );
    let nan = vcgtq_u32(
        vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
        vdupq_n_u32(0x7f800000),
    );
    half = vbslq_u32(nan, vdupq_n_u32(0x7e00), half);
    let he = vshrq_n_u32::<10>(half);
    let hf = vandq_u32(half, vdupq_n_u32(1023));
    let normal = vorrq_u32(
        vshlq_n_u32::<23>(vaddq_u32(he, vdupq_n_u32(112))),
        vshlq_n_u32::<13>(hf),
    );
    let shift = vsubq_u32(vclzq_u32(hf), vdupq_n_u32(21));
    let subnormal = vorrq_u32(
        vshlq_n_u32::<23>(vsubq_u32(vdupq_n_u32(113), shift)),
        vshlq_n_u32::<13>(vandq_u32(
            vshlq_u32(hf, vreinterpretq_s32_u32(shift)),
            vdupq_n_u32(1023),
        )),
    );
    let subnormal = vbslq_u32(vceqq_u32(hf, vdupq_n_u32(0)), vdupq_n_u32(0), subnormal);
    let wide = vbslq_u32(vceqq_u32(he, vdupq_n_u32(0)), subnormal, normal);
    let wide = vbslq_u32(
        vceqq_u32(he, vdupq_n_u32(31)),
        vorrq_u32(vdupq_n_u32(0x7f800000), vshlq_n_u32::<13>(hf)),
        wide,
    );
    vreinterpretq_f32_u32(vorrq_u32(sign, wide))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp4(value: float32x4_t) -> float32x4_t {
    let c = |bits| vdupq_n_f32(f32::from_bits(bits));
    let bits = vreinterpretq_u32_f32(value);
    let abs = vandq_u32(bits, vdupq_n_u32(0x7fffffff));
    let sign = vandq_u32(bits, vdupq_n_u32(0x80000000));
    let large = vcgeq_u32(abs, vdupq_n_u32(0x3f851592));
    let reduce = vcgtq_u32(abs, vdupq_n_u32(0x3eb17218));
    let signed = |v: float32x4_t| vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v), sign));
    let long_k = vcvtq_s32_f32(vaddq_f32(
        vmulq_f32(c(0x3fb8aa3b), value),
        signed(vdupq_n_f32(0.5)),
    ));
    let near_k = vcvtq_s32_f32(signed(vdupq_n_f32(1.0)));
    let k = vbslq_s32(reduce, vbslq_s32(large, long_k, near_k), vdupq_n_s32(0));
    let kf = vcvtq_f32_s32(k);
    let hi = vbslq_f32(
        reduce,
        vsubq_f32(
            value,
            vbslq_f32(large, vmulq_f32(kf, c(0x3f317200)), signed(c(0x3f317200))),
        ),
        vdupq_n_f32(0.0),
    );
    let lo = vbslq_f32(
        reduce,
        vbslq_f32(large, vmulq_f32(kf, c(0x35bfbe8e)), signed(c(0x35bfbe8e))),
        vdupq_n_f32(0.0),
    );
    let reduced = vbslq_f32(reduce, vsubq_f32(hi, lo), value);
    let square = vmulq_f32(reduced, reduced);
    let correction = vsubq_f32(
        reduced,
        vmulq_f32(
            square,
            vaddq_f32(c(0x3e2aaa8f), vmulq_f32(square, c(0xbb355215))),
        ),
    );
    let product = vmulq_f32(reduced, correction);
    let direct = vsubq_f32(
        vdupq_n_f32(1.0),
        vsubq_f32(
            vdivq_f32(product, vsubq_f32(correction, vdupq_n_f32(2.0))),
            reduced,
        ),
    );
    let result = vsubq_f32(
        vdupq_n_f32(1.0),
        vsubq_f32(
            vsubq_f32(
                lo,
                vdivq_f32(product, vsubq_f32(vdupq_n_f32(2.0), correction)),
            ),
            hi,
        ),
    );
    let scale = |offset| {
        vreinterpretq_f32_u32(vshlq_n_u32::<23>(vreinterpretq_u32_s32(vaddq_s32(
            k,
            vdupq_n_s32(offset),
        ))))
    };
    let normal = vmulq_f32(result, scale(127));
    let tiny = vmulq_f32(vmulq_f32(result, scale(227)), c(0x0d800000));
    let huge = vmulq_f32(vmulq_n_f32(result, 2.0), c(0x7f000000));
    let out = vbslq_f32(
        vcgeq_s32(k, vdupq_n_s32(-125)),
        vbslq_f32(vceqq_s32(k, vdupq_n_s32(128)), huge, normal),
        tiny,
    );
    let out = vbslq_f32(vceqq_s32(k, vdupq_n_s32(0)), direct, out);
    let out = vbslq_f32(
        vcltq_u32(abs, vdupq_n_u32(0x39000000)),
        vaddq_f32(vdupq_n_f32(1.0), value),
        out,
    );
    // Rare range-limit/NaN lanes keep the scalar branch and NaN payload behavior.
    let mut input = [0.0; 4];
    let mut output = [0.0; 4];
    // SAFETY: local arrays have exactly four elements.
    unsafe {
        vst1q_f32(input.as_mut_ptr(), value);
        vst1q_f32(output.as_mut_ptr(), out);
    }
    for i in 0..4 {
        if !(-103.0..88.0).contains(&input[i]) {
            output[i] = yskn_ref::portable_math::exp(input[i]);
        }
    }
    // SAFETY: array has four elements.
    unsafe { vld1q_f32(output.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn map_neon(values: &mut [f32], operation: Operation) {
    let end = values.len() / 4 * 4;
    // SAFETY: all mutable reads/writes derive from this one exclusive pointer.
    unsafe {
        let ptr = values.as_mut_ptr();
        for i in (0..end).step_by(4) {
            let value = vld1q_f32(ptr.add(i));
            let result = match operation {
                Operation::Bfloat => bfloat4(value),
                Operation::Half(scale) => half4(vmulq_n_f32(value, scale)),
                Operation::Exp2 => exp4(vmulq_n_f32(value, f32::from_bits(0x3f317218))),
                Operation::Sigmoid => vdivq_f32(
                    vdupq_n_f32(1.0),
                    vaddq_f32(vdupq_n_f32(1.0), exp4(vnegq_f32(value))),
                ),
                Operation::Silu => {
                    vdivq_f32(value, vaddq_f32(vdupq_n_f32(1.0), exp4(vnegq_f32(value))))
                }
            };
            vst1q_f32(ptr.add(i), result);
        }
    }
    scalar(&mut values[end..], operation);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(values: &[f32]) {
        for operation in [
            Operation::Bfloat,
            Operation::Half(1.0),
            Operation::Half(f32::from_bits(0x3d3504f3)),
            Operation::Silu,
            Operation::Sigmoid,
            Operation::Exp2,
        ] {
            let mut expected = values.to_vec();
            let mut actual = values.to_vec();
            scalar(&mut expected, operation);
            map(&mut actual, operation);
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "input={:08x} index={i}",
                    values[i].to_bits()
                );
            }
        }
        assert_eq!(
            FastMath::finite(values, "test").is_ok(),
            ScalarMath::finite(values, "test").is_ok()
        );
    }
    #[test]
    fn elementwise_matches_scalar_at_rounding_and_exp_boundaries() {
        let mut values = vec![
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0xff800001),
            f32::from_bits(1),
            f32::from_bits(0x80000001),
            f32::MAX,
            -f32::MAX,
        ];
        for bits in [
            0x39000000u32,
            0x3eb17218,
            0x3f851592,
            0x42b17218,
            0x42cff1b5,
            0x00800000,
            0x477ff000, // half overflow midpoint (65520)
            0x38800000, // half normal/subnormal boundary
        ] {
            for delta in -8..=8 {
                let x = f32::from_bits(bits.wrapping_add_signed(delta));
                values.extend([x, -x]);
            }
        }
        for half in 0..0x7c00u16 {
            let x = yskn_format::f16_to_f32(half);
            let midpoint = (x + yskn_format::f16_to_f32(half + 1)) * 0.5;
            values.extend([
                x,
                -x,
                midpoint,
                -midpoint,
                f32::from_bits(midpoint.to_bits().saturating_sub(1)),
                f32::from_bits(midpoint.to_bits() + 1),
            ]);
        }
        for b in 0..=u16::MAX {
            let bits = u32::from(b) << 16;
            values.extend([
                f32::from_bits(bits),
                f32::from_bits(bits | 0x7fff),
                f32::from_bits(bits | 0x8000),
                f32::from_bits(bits | 0x8001),
            ]);
        }
        let mut seed = 23u32;
        for _ in 0..100_003 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            values.push(f32::from_bits(seed));
        }
        check(&values);
        for len in 0..9 {
            check(&values[..len]);
        }
        #[cfg(target_arch = "aarch64")]
        if super::super::neon_available() {
            for chunk in values.as_chunks::<4>().0 {
                let mut result = [0.0; 4];
                // SAFETY: runtime NEON and exact four-element arrays.
                unsafe {
                    vst1q_f32(result.as_mut_ptr(), exp4(vld1q_f32(chunk.as_ptr())));
                }
                for i in 0..4 {
                    assert_eq!(
                        result[i].to_bits(),
                        yskn_ref::portable_math::exp(chunk[i]).to_bits(),
                        "exp input={:08x}",
                        chunk[i].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn finite_detects_invalid_values_in_every_lane_and_tail() {
        for len in 1..20 {
            for pos in 0..len {
                for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut values = vec![0.0; len];
                    values[pos] = value;
                    assert_eq!(
                        FastMath::finite(&values, "test").unwrap_err().to_string(),
                        ScalarMath::finite(&values, "test").unwrap_err().to_string()
                    );
                }
            }
        }
    }
}
