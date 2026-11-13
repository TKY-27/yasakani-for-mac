//! [Chapter 05: residual, normalization and MLP](../../../docs/spec/05_RESIDUAL_NORM_MLP.md).

use crate::compressed_linear::{PreparedActivation, quantize_activation};
use crate::portable_math::{bfloat, rsqrt, silu};
use crate::{Error, Result, require_finite};

pub struct NormalizedInput {
    pub prepared: PreparedActivation,
    pub inverse_rms: f32,
    pub weighted_hidden_bf16: Vec<f32>,
}

pub fn prepare_normalization(
    hidden: &[f32],
    normalization_weight: &[f32],
    input_signs: &[i8],
    epsilon: f32,
) -> Result<NormalizedInput> {
    if hidden.is_empty() || hidden.len() % 512 != 0 {
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
    for block in hidden.chunks_exact(512) {
        total += block_sum_squares(block);
    }
    require_finite(&[total], "normalization sum of squares")?;
    let norm_argument = total / hidden.len() as f32 + epsilon;
    require_finite(&[norm_argument], "normalization argument")?;
    let inverse_rms = rsqrt(norm_argument);
    require_finite(&[inverse_rms], "normalization inverse RMS")?;

    let mut unnormalized = Vec::with_capacity(hidden.len());
    let mut weighted_hidden_bf16 = Vec::with_capacity(hidden.len());
    for (&value, &weight) in hidden.iter().zip(normalization_weight) {
        let adjusted = value * (1.0 + weight);
        unnormalized.push(adjusted);
        // Dense GDN gates receive a BF16 copy; the compressed input must remain FP32.
        weighted_hidden_bf16.push(bfloat(adjusted));
    }
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
    for group in partial.chunks_exact(32) {
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

pub fn residual_add(hidden: &[f32], projection: &[f32]) -> Result<Vec<f32>> {
    if hidden.len() != projection.len() {
        return Err(Error::InvalidInput(
            "residual dimensions do not match".into(),
        ));
    }
    require_finite(hidden, "residual input")?;
    require_finite(projection, "residual projection")?;
    let result: Vec<_> = hidden
        .iter()
        .zip(projection)
        // The projection rounds before addition; the sum has a second BF16 boundary.
        .map(|(&hidden, &projection)| bfloat(hidden + bfloat(projection)))
        .collect();
    require_finite(&result, "residual output")?;
    Ok(result)
}

pub fn gated_product(gate_projection: &[f32], up_projection: &[f32]) -> Result<Vec<f32>> {
    if gate_projection.len() != up_projection.len() {
        return Err(Error::InvalidInput(
            "MLP branch dimensions do not match".into(),
        ));
    }
    require_finite(gate_projection, "MLP gate projection")?;
    require_finite(up_projection, "MLP up projection")?;
    let mut output = Vec::with_capacity(gate_projection.len());
    for (&gate, &up) in gate_projection.iter().zip(up_projection) {
        // Chapter 05 rounds both projections, SiLU, then the product separately to BF16.
        let gate = bfloat(gate);
        let up = bfloat(up);
        let activation = bfloat(silu(gate));
        output.push(bfloat(activation * up));
    }
    require_finite(&output, "MLP product")?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_rounds_projection_before_sum() {
        // 1.00390625 is the midpoint between BF16 1 and 1.0078125.
        // Rounding it before cancellation yields zero; rounding only the sum would not.
        let output = residual_add(&[-1.0], &[f32::from_bits(0x3f80_8000)]).unwrap();
        assert_eq!(output[0].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn normalization_rejects_nonfinite_and_square_overflow() {
        let mut hidden = vec![1.0; 512];
        let weights = vec![0.0; 512];
        let signs = vec![1; 512];
        hidden[19] = f32::NAN;
        assert!(matches!(
            prepare_normalization(&hidden, &weights, &signs, 1e-6),
            Err(Error::NonFinite(_))
        ));
        hidden[19] = f32::MAX;
        assert!(matches!(
            prepare_normalization(&hidden, &weights, &signs, 1e-6),
            Err(Error::NonFinite(_))
        ));
        assert!(matches!(
            prepare_normalization(&[1e17; 512], &[-1.0; 512], &signs, f32::MAX),
            Err(Error::NonFinite(_))
        ));
    }
}
