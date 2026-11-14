//! Independent butterflies and elementwise operations; reductions retain chapter 05 order.
use crate::{Error, Result};
use yskn_ref::{
    compressed_linear::{self, PreparedActivation},
    portable_math::{Elementwise, bfloat, rsqrt, silu},
    residual_norm_mlp::NormalizedInput,
};

fn require_finite(values: &[f32], context: &'static str) -> Result<()> {
    super::math::FastMath::finite(values, context)
}

pub(crate) fn quantize_activation(input: &[f32], signs: &[i8]) -> Result<PreparedActivation> {
    if super::enabled() {
        compressed_linear::quantize_activation_with_math::<super::math::FastMath>(
            input,
            signs,
            hadamard512,
        )
    } else {
        compressed_linear::quantize_activation(input, signs)
    }
}
pub(crate) fn restore_projection(
    sums: &[i64],
    prepared: &PreparedActivation,
    inverse_rms: Option<f32>,
    scales: &[f32],
    signs: Option<&[i8]>,
    rows: usize,
) -> Result<Vec<f32>> {
    if super::enabled() {
        compressed_linear::restore_projection_with_transform(
            sums,
            prepared,
            inverse_rms,
            scales,
            signs,
            rows,
            hadamard512,
        )
    } else {
        compressed_linear::restore_projection(sums, prepared, inverse_rms, scales, signs, rows)
    }
}
fn hadamard512(values: &mut [f32; 512]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: quantize_activation/restore_projection check enabled() (runtime
        // NEON detection); direct tests check neon_available(). The array has 512 elements.
        unsafe {
            hadamard_neon(values);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    compressed_linear::hadamard512(values);
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hadamard_neon(values: &mut [f32; 512]) {
    use std::arch::aarch64::*;
    // SAFETY: each initial chunk holds four floats. Later paired vectors lie within
    // a 2*stride block of the 512-element array; both operands load before either store.
    unsafe {
        let values_ptr = values.as_mut_ptr();
        for i in (0..512).step_by(4) {
            let v = vld1q_f32(values_ptr.add(i));
            let even = vuzp1q_f32(v, v);
            let odd = vuzp2q_f32(v, v);
            let first = vzip1q_f32(vaddq_f32(even, odd), vsubq_f32(even, odd));
            let lo = vget_low_f32(first);
            let hi = vget_high_f32(first);
            vst1q_f32(
                values_ptr.add(i),
                vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi)),
            );
        }
        let mut stride = 4;
        while stride < 512 {
            for base in (0..512).step_by(stride * 2) {
                for off in (0..stride).step_by(4) {
                    let lo = vld1q_f32(values_ptr.add(base + off));
                    let hi = vld1q_f32(values_ptr.add(base + off + stride));
                    vst1q_f32(values_ptr.add(base + off), vaddq_f32(lo, hi));
                    vst1q_f32(values_ptr.add(base + off + stride), vsubq_f32(lo, hi));
                }
            }
            stride *= 2;
        }
    }
}

fn adjusted4(hidden: [f32; 4], weight: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: prepare_normalization checks enabled() (runtime NEON detection)
        // before calling; both arrays have four elements.
        unsafe { adjusted_neon(hidden, weight) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        std::array::from_fn(|i| hidden[i] * (1.0 + weight[i]))
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn adjusted_neon(hidden: [f32; 4], weight: [f32; 4]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mut out = [0.0; 4];
    // SAFETY: all three arrays contain four floats.
    unsafe {
        vst1q_f32(
            out.as_mut_ptr(),
            vmulq_f32(
                vld1q_f32(hidden.as_ptr()),
                vaddq_f32(vdupq_n_f32(1.0), vld1q_f32(weight.as_ptr())),
            ),
        );
    }
    out
}

pub(crate) fn prepare_normalization(
    hidden: &[f32],
    normalization_weight: &[f32],
    input_signs: &[i8],
    epsilon: f32,
) -> Result<NormalizedInput> {
    if !super::enabled() {
        return yskn_ref::residual_norm_mlp::prepare_normalization(
            hidden,
            normalization_weight,
            input_signs,
            epsilon,
        );
    }
    if hidden.is_empty() || !hidden.len().is_multiple_of(512) {
        return Err(Error::InvalidInput(
            "hidden width must be a positive multiple of 512".into(),
        ));
    }
    if normalization_weight.len() != hidden.len() || input_signs.len() != hidden.len() {
        return Err(Error::InvalidInput(
            "normalization dimensions do not match".into(),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidModel(
            "epsilon must be finite and positive".into(),
        ));
    }
    require_finite(hidden, "normalization input")?;
    require_finite(normalization_weight, "normalization weights")?;

    // Chapter 05 fixes the block fold from +0; combining blocks differently changes bits.
    let mut total = 0.0_f32;
    for block in hidden.as_chunks::<512>().0 {
        total += block_sum_squares(block);
    }
    require_finite(&[total], "normalization sum of squares")?;
    let norm_argument = total / hidden.len() as f32 + epsilon;
    require_finite(&[norm_argument], "normalization argument")?;
    let inverse_rms = rsqrt(norm_argument);
    require_finite(&[inverse_rms], "normalization inverse RMS")?;

    let mut unnormalized = vec![0.0; hidden.len()];
    let mut weighted_hidden_bf16 = vec![0.0; hidden.len()];
    for i in (0..hidden.len()).step_by(4) {
        let h: [f32; 4] = hidden[i..i + 4].try_into().unwrap();
        let w: [f32; 4] = normalization_weight[i..i + 4].try_into().unwrap();
        let adjusted = adjusted4(h, w);
        unnormalized[i..i + 4].copy_from_slice(&adjusted);
        weighted_hidden_bf16[i..i + 4].copy_from_slice(&adjusted);
    }
    super::math::FastMath::bfloat(&mut weighted_hidden_bf16);
    require_finite(&weighted_hidden_bf16, "weighted hidden BF16")?;
    Ok(NormalizedInput {
        prepared: quantize_activation(&unnormalized, input_signs)?,
        inverse_rms,
        weighted_hidden_bf16,
    })
}

fn block_sum_squares(values: &[f32]) -> f32 {
    let mut partial = [0.0_f32; 128];
    // Four strided terms per partial, then simultaneous XOR stages: chapter 05's order.
    for (index, sum) in partial.iter_mut().enumerate() {
        for group in 0..4 {
            let value = values[index + 128 * group];
            *sum += value * value;
        }
    }
    let mut result = 0.0_f32;
    for group in partial.as_chunks::<32>().0 {
        let mut lanes = [0.0; 32];
        lanes.copy_from_slice(group);
        for offset in [16, 8, 4, 2, 1] {
            let previous = lanes;
            for lane in 0..32 {
                lanes[lane] = previous[lane] + previous[lane ^ offset];
            }
        }
        result += lanes[0];
    }
    result
}

pub(crate) fn residual_add(hidden: &[f32], projection: &[f32]) -> Result<Vec<f32>> {
    if !super::enabled() {
        return yskn_ref::residual_norm_mlp::residual_add(hidden, projection);
    }
    if hidden.len() != projection.len() {
        return Err(Error::InvalidInput(
            "residual dimensions do not match".into(),
        ));
    }
    require_finite(hidden, "residual input")?;
    require_finite(projection, "residual projection")?;
    let mut result = vec![0.0; hidden.len()];
    let end = hidden.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        let a = hidden[i..i + 4].try_into().unwrap();
        let b = projection[i..i + 4].try_into().unwrap();
        result[i..i + 4].copy_from_slice(&rounded_binary4(a, b, false));
    }
    for i in end..hidden.len() {
        result[i] = bfloat(hidden[i] + bfloat(projection[i]));
    }
    require_finite(&result, "residual output")?;
    Ok(result)
}

pub(crate) fn gated_product(gate_projection: &[f32], up_projection: &[f32]) -> Result<Vec<f32>> {
    if !super::enabled() {
        return yskn_ref::residual_norm_mlp::gated_product(gate_projection, up_projection);
    }
    if gate_projection.len() != up_projection.len() {
        return Err(Error::InvalidInput(
            "MLP branch dimensions do not match".into(),
        ));
    }
    require_finite(gate_projection, "MLP gate projection")?;
    require_finite(up_projection, "MLP up projection")?;
    let mut output = vec![0.0; gate_projection.len()];
    let end = output.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        let mut activation: [f32; 4] = gate_projection[i..i + 4].try_into().unwrap();
        super::math::FastMath::bfloat(&mut activation);
        super::math::FastMath::silu(&mut activation);
        super::math::FastMath::bfloat(&mut activation);
        let up = up_projection[i..i + 4].try_into().unwrap();
        output[i..i + 4].copy_from_slice(&rounded_binary4(activation, up, true));
    }
    for i in end..output.len() {
        output[i] = bfloat(bfloat(silu(bfloat(gate_projection[i]))) * bfloat(up_projection[i]));
    }
    require_finite(&output, "MLP product")?;
    Ok(output)
}

fn rounded_binary4(a: [f32; 4], b: [f32; 4], multiply: bool) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: residual_add/gated_product check enabled() (runtime NEON detection)
        // before calling; both arrays have four elements.
        unsafe { rounded_binary_neon(a, b, multiply) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        std::array::from_fn(|i| {
            bfloat(if multiply {
                a[i] * bfloat(b[i])
            } else {
                a[i] + bfloat(b[i])
            })
        })
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rounded_binary_neon(a: [f32; 4], b: [f32; 4], multiply: bool) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mut out = [0.0; 4];
    // SAFETY: loads/stores cover precisely four floats; integer rounding matches
    // f32_to_bf16 including quieting NaNs and preserving the sign of zero.
    unsafe {
        let round = |v: float32x4_t| {
            let bits = vreinterpretq_u32_f32(v);
            let bias = vaddq_u32(
                vdupq_n_u32(0x7fff),
                vandq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(1)),
            );
            let rounded = vandq_u32(vaddq_u32(bits, bias), vdupq_n_u32(0xffff0000));
            let nan = vcgtq_u32(
                vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                vdupq_n_u32(0x7f800000),
            );
            let quiet = vorrq_u32(
                vandq_u32(bits, vdupq_n_u32(0xffff0000)),
                vdupq_n_u32(0x00400000),
            );
            vreinterpretq_f32_u32(vbslq_u32(nan, quiet, rounded))
        };
        let a = vld1q_f32(a.as_ptr());
        let b = round(vld1q_f32(b.as_ptr()));
        let result = if multiply {
            vmulq_f32(a, b)
        } else {
            vaddq_f32(a, b)
        };
        vst1q_f32(out.as_mut_ptr(), round(result));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use yskn_ref::residual_norm_mlp as reference;
    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }
    fn same_prepared(a: &PreparedActivation, b: &PreparedActivation) {
        assert_eq!(a.activation8(), b.activation8());
        assert_eq!(a.integer_scale(), b.integer_scale());
        assert_eq!(bits(a.activation_scale()), bits(b.activation_scale()));
        assert_eq!(a.row_quantum().to_bits(), b.row_quantum().to_bits());
    }
    fn same_result(a: Result<Vec<f32>>, b: Result<Vec<f32>>) {
        match (a, b) {
            (Ok(a), Ok(b)) => assert_eq!(bits(&a), bits(&b)),
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            _ => panic!("result kinds differ"),
        }
    }
    #[test]
    fn rotations_quantization_norm_and_elementwise_boundaries_equal_reference() {
        let mut seed = 91u32;
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
                5 => f32::from_bits(0x3f818000),
                _ => seed as i32 as f32 / i32::MAX as f32 * 2.0,
            }
        };
        for iteration in 0..32 {
            let width = if iteration % 2 == 0 { 512 } else { 1024 };
            let mut hidden: Vec<_> = (0..width).map(|_| sample()).collect();
            if iteration == 0 {
                hidden.fill(-0.0);
            }
            if iteration == 1 {
                hidden.fill(f32::from_bits(1));
            }
            let other: Vec<_> = (0..width).map(|_| sample()).collect();
            let signs: Vec<_> = (0..width)
                .map(|i| if i % 3 == 0 { -1 } else { 1 })
                .collect();
            if super::super::neon_available() {
                let mut a: [f32; 512] = hidden[..512].try_into().unwrap();
                let mut b = a;
                hadamard512(&mut a);
                compressed_linear::hadamard512(&mut b);
                assert_eq!(bits(&a), bits(&b));
            }
            let a = quantize_activation(&hidden, &signs).unwrap();
            let b = compressed_linear::quantize_activation(&hidden, &signs).unwrap();
            same_prepared(&a, &b);
            let a = prepare_normalization(&hidden, &other, &signs, 1e-6).unwrap();
            let b = reference::prepare_normalization(&hidden, &other, &signs, 1e-6).unwrap();
            same_prepared(&a.prepared, &b.prepared);
            assert_eq!(a.inverse_rms.to_bits(), b.inverse_rms.to_bits());
            assert_eq!(bits(&a.weighted_hidden_bf16), bits(&b.weighted_hidden_bf16));
            let sums: Vec<_> = (0..width)
                .map(|i| match i % 4 {
                    0 => i64::MIN,
                    1 => i64::MAX,
                    _ => (i as i64 - 256) * 7919,
                })
                .collect();
            for rotation in [None, Some(signs.as_slice())] {
                for inverse in [None, Some(0.71)] {
                    same_result(
                        restore_projection(
                            &sums,
                            &a.prepared,
                            inverse,
                            &other,
                            rotation,
                            width - 7,
                        ),
                        compressed_linear::restore_projection(
                            &sums,
                            &b.prepared,
                            inverse,
                            &other,
                            rotation,
                            width - 7,
                        ),
                    );
                }
            }
            // Include non-vector tails as well as complete vectors.
            for length in [0, 1, 3, 4, 7, width] {
                same_result(
                    residual_add(&hidden[..length], &other[..length]),
                    reference::residual_add(&hidden[..length], &other[..length]),
                );
                same_result(
                    gated_product(&hidden[..length], &other[..length]),
                    reference::gated_product(&hidden[..length], &other[..length]),
                );
            }
        }
    }
    #[test]
    fn validation_and_nonfinite_errors_remain_identical() {
        let signs = vec![1; 512];
        let weights = vec![0.0; 512];
        for bad in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            -f32::MAX,
        ] {
            let mut input = vec![1.0; 512];
            input[17] = bad;
            assert_eq!(
                quantize_activation(&input, &signs)
                    .err()
                    .map(|e| e.to_string()),
                compressed_linear::quantize_activation(&input, &signs)
                    .err()
                    .map(|e| e.to_string())
            );
            assert_eq!(
                prepare_normalization(&input, &weights, &signs, 1e-6)
                    .err()
                    .map(|e| e.to_string()),
                reference::prepare_normalization(&input, &weights, &signs, 1e-6)
                    .err()
                    .map(|e| e.to_string())
            );
            same_result(
                residual_add(&input, &weights),
                reference::residual_add(&input, &weights),
            );
            same_result(
                gated_product(&input, &weights),
                reference::gated_product(&input, &weights),
            );
        }
        let input = vec![1.0; 512];
        let invalid_signs = vec![0; 512];
        for signs in [&invalid_signs[..], &signs[..511]] {
            assert_eq!(
                quantize_activation(&input, signs).unwrap_err().to_string(),
                compressed_linear::quantize_activation(&input, signs)
                    .unwrap_err()
                    .to_string()
            );
        }
        let prepared = quantize_activation(&input, &vec![1; 512]).unwrap();
        for inverse in [Some(f32::NAN), Some(f32::INFINITY)] {
            same_result(
                restore_projection(&[1; 512], &prepared, inverse, &input, None, 512),
                compressed_linear::restore_projection(
                    &[1; 512], &prepared, inverse, &input, None, 512,
                ),
            );
        }
        same_result(
            residual_add(&[1.0], &[]),
            reference::residual_add(&[1.0], &[]),
        );
        same_result(
            gated_product(&[1.0], &[]),
            reference::gated_product(&[1.0], &[]),
        );
    }
}
