//! Scalar implementation of [chapter 02](../../../docs/spec/02_COMPRESSED_LINEAR.md).

use yskn_format::Tensor;

use crate::portable_math::bfloat;
use crate::{Error, Result, require_finite};

pub const ROTATION_BLOCK: usize = 512;
const HADAMARD_NORMALIZATION: f32 = f32::from_bits(0x3d35_04f3);
const INTEGER_SCALE_RANGE: f32 = 16_777_216.0;
const INTEGER_SCALE_QUANTUM: f32 = f32::from_bits(0x3380_0000);

/// Ascending-stride butterflies; every sum/difference rounds separately to f32.
pub fn hadamard512(values: &mut [f32; ROTATION_BLOCK]) {
    // Both operands are saved before either write. Reordering the stages
    // changes intermediate rounding and therefore the reference's result.
    let mut stride = 1;
    while stride < ROTATION_BLOCK {
        for base in (0..ROTATION_BLOCK).step_by(2 * stride) {
            for offset in 0..stride {
                let lower = values[base + offset];
                let upper = values[base + offset + stride];
                values[base + offset] = lower + upper;
                values[base + offset + stride] = lower - upper;
            }
        }
        stride *= 2;
    }
}

#[derive(Clone, Debug)]
pub struct PreparedActivation {
    activation8: Vec<i8>,
    activation_scale: Vec<f32>,
    integer_scale: Vec<i32>,
    row_quantum: f32,
}

impl PreparedActivation {
    pub fn activation8(&self) -> &[i8] {
        &self.activation8
    }

    pub fn activation_scale(&self) -> &[f32] {
        &self.activation_scale
    }

    pub fn integer_scale(&self) -> &[i32] {
        &self.integer_scale
    }

    pub fn row_quantum(&self) -> f32 {
        self.row_quantum
    }
}

pub fn quantize_activation(input: &[f32], input_signs: &[i8]) -> Result<PreparedActivation> {
    quantize_activation_with_transform(input, input_signs, hadamard512)
}

/// Engine transform hook; every scale, conversion and validation stays scalar.
pub fn quantize_activation_with_transform(
    input: &[f32],
    input_signs: &[i8],
    transform: impl Fn(&mut [f32; ROTATION_BLOCK]),
) -> Result<PreparedActivation> {
    quantize_activation_with_math::<crate::portable_math::ScalarMath>(input, input_signs, transform)
}

/// Shared quantizer with independent transform and elementwise conversion kernels.
pub fn quantize_activation_with_math<M: crate::portable_math::Elementwise>(
    input: &[f32],
    input_signs: &[i8],
    transform: impl Fn(&mut [f32; ROTATION_BLOCK]),
) -> Result<PreparedActivation> {
    if input.is_empty() || input.len() % ROTATION_BLOCK != 0 || input_signs.len() != input.len() {
        return Err(Error::InvalidModel(
            "activation and input signs must have matching, nonzero multiples of 512 elements"
                .into(),
        ));
    }
    M::finite(input, "pre-rotation activation")?;
    if input_signs.iter().any(|&sign| sign != -1 && sign != 1) {
        return Err(Error::InvalidModel(
            "rotation signs must be -1 or +1".into(),
        ));
    }
    let groups = input.len() / ROTATION_BLOCK;
    let mut activation8 = vec![0; input.len()];
    let mut activation_scale = Vec::with_capacity(groups);
    for group in 0..groups {
        let start = group * ROTATION_BLOCK;
        let mut rotated = [0.0; ROTATION_BLOCK];
        for (channel, value) in rotated.iter_mut().enumerate() {
            *value = input[start + channel] * f32::from(input_signs[start + channel]);
        }
        transform(&mut rotated);
        // FP16 rounding precedes the finite check, maximum and int8 quantization.
        M::scale_half(&mut rotated, HADAMARD_NORMALIZATION);
        M::finite(&rotated, "FP16-rounded rotated activation")?;
        let maximum = rotated
            .iter()
            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
        let scale = maximum / 127.0;
        activation_scale.push(scale);
        for (channel, &value) in rotated.iter().enumerate() {
            let ratio = if scale > 0.0 { value / scale } else { 0.0 };
            // The specified integer conversion is nearest-even, not Rust's
            // truncating cast. Clamping never emits the unused code -128.
            activation8[start + channel] = ratio.round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
    let maximum_scale = activation_scale.iter().copied().fold(0.0_f32, f32::max);
    let scale_multiplier = if maximum_scale > 0.0 {
        INTEGER_SCALE_RANGE / maximum_scale
    } else {
        0.0
    };
    // Multiplication by exactly 2^-24 implements ldexp(maximum_scale, -24)
    // with one binary32 rounding, including gradual underflow.
    let row_quantum = maximum_scale * INTEGER_SCALE_QUANTUM;
    let integer_scale = activation_scale
        .iter()
        .map(|&scale| {
            // Quantize scales onto a common 24-bit integer grid before any dot.
            (scale * scale_multiplier).round_ties_even() as i32
        })
        .collect();
    Ok(PreparedActivation {
        activation8,
        activation_scale,
        integer_scale,
        row_quantum,
    })
}

/// Quantize a lookup table and fold its scale into the physical row scales.
pub fn quantize_weight_table(table: &[f32], scales: &[f32]) -> Result<(Vec<i8>, Vec<f32>)> {
    require_finite(table, "weight lookup table")?;
    require_finite(scales, "weight row scales")?;
    let table_maximum = table
        .iter()
        .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
    if table_maximum == 0.0 {
        return Err(Error::InvalidModel(
            "a compressed projection cannot use an all-zero lookup table (LIN-01)".into(),
        ));
    }
    // D1 fixes the upstream host's unspecified rounding mode to RNE.
    let table_scale = table_maximum / 127.0;
    let weight_table8 = table
        .iter()
        .map(|&value| (value / table_scale).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect();
    let effective_row_scale: Vec<_> = scales.iter().map(|&scale| scale * table_scale).collect();
    require_finite(&effective_row_scale, "effective weight row scales")?;
    Ok((weight_table8, effective_row_scale))
}

/// A compressed matrix keeps its codes; it never expands the dense weights.
#[derive(Debug)]
pub struct CompressedLinear {
    tensor: Tensor,
    weight_table8: Vec<i8>,
    effective_row_scale: Vec<f32>,
    output_rotation: bool,
}

impl CompressedLinear {
    /// `output_rotation` follows the consumer: body=true, vocabulary head=false.
    pub fn new(tensor: Tensor, output_rotation: bool) -> Result<Self> {
        let (weight_table8, effective_row_scale) =
            quantize_weight_table(tensor.table(), tensor.scales())?;
        if output_rotation
            && (tensor.physical_rows() % ROTATION_BLOCK != 0 || tensor.output_signs().is_none())
        {
            return Err(Error::InvalidModel(
                "a body projection requires complete 512-row output rotations".into(),
            ));
        }
        Ok(Self {
            tensor,
            weight_table8,
            effective_row_scale,
            output_rotation,
        })
    }

    pub fn rows(&self) -> usize {
        self.tensor.rows()
    }

    pub fn cols(&self) -> usize {
        self.tensor.cols()
    }

    pub fn input_signs(&self) -> &[i8] {
        self.tensor.input_signs()
    }

    /// Integer weighted sums for the first `rows` physical output rows.
    ///
    /// `activation8` contains one input row and `integer_scale` contains one
    /// scale per 512 input columns, as produced by [`quantize_activation`].
    pub fn weighted_sums(
        &self,
        activation8: &[i8],
        integer_scale: &[i32],
        rows: usize,
    ) -> Result<Vec<i64>> {
        if activation8.len() != self.cols() {
            return Err(Error::InvalidModel(
                "projection input width does not match the prepared activation".into(),
            ));
        }
        if integer_scale.len() != self.cols() / ROTATION_BLOCK {
            return Err(Error::InvalidModel(
                "projection integer scales must match the number of 512-column groups".into(),
            ));
        }
        if rows > self.tensor.physical_rows() {
            return Err(Error::InvalidModel(
                "requested projection rows exceed the physical row count".into(),
            ));
        }
        let mut output = vec![0; rows];
        for (row, sum) in output.iter_mut().enumerate() {
            let mut weighted_sum = 0_i64;
            for (group, &integer_scale) in integer_scale.iter().enumerate() {
                let mut dot = 0_i32;
                // Two adjacent 256-weight sequences form one 512-term dot.
                // Reset only here; scaling each sequence separately differs.
                for sequence_in_group in 0..2 {
                    let sequence = group * 2 + sequence_in_group;
                    for step in 0..128 {
                        let state = usize::from(self.tensor.state(row, sequence, step)?);
                        for component in 0..2 {
                            let column = sequence * 256 + step * 2 + component;
                            dot += i32::from(self.weight_table8[state * 2 + component])
                                * i32::from(activation8[column]);
                        }
                    }
                }
                let weighted_dot = i64::from(integer_scale) * i64::from(dot);
                weighted_sum = weighted_sum.checked_add(weighted_dot).ok_or_else(|| {
                    Error::InvalidModel("compressed integer accumulation exceeds i64".into())
                })?;
            }
            *sum = weighted_sum;
        }
        Ok(output)
    }

    pub fn project(
        &self,
        prepared: &PreparedActivation,
        inverse_rms: Option<f32>,
    ) -> Result<Vec<f32>> {
        if prepared.activation8.len() != self.cols() {
            return Err(Error::InvalidModel(
                "projection input width does not match the prepared activation".into(),
            ));
        }
        if inverse_rms.is_some_and(|value| !value.is_finite()) {
            return Err(Error::NonFinite("deferred inverse RMS"));
        }
        let weighted_sums = self.weighted_sums(
            &prepared.activation8,
            &prepared.integer_scale,
            self.tensor.physical_rows(),
        )?;
        restore_projection(
            &weighted_sums,
            prepared,
            inverse_rms,
            &self.effective_row_scale,
            if self.output_rotation {
                self.tensor.output_signs()
            } else {
                None
            },
            self.rows(),
        )
    }
}

/// Restore a projection's integer sums using the canonical per-row scale/rotation order.
pub fn restore_projection(
    weighted_sums: &[i64],
    prepared: &PreparedActivation,
    inverse_rms: Option<f32>,
    effective_row_scale: &[f32],
    output_signs: Option<&[i8]>,
    rows: usize,
) -> Result<Vec<f32>> {
    restore_projection_with_transform(
        weighted_sums,
        prepared,
        inverse_rms,
        effective_row_scale,
        output_signs,
        rows,
        hadamard512,
    )
}

/// Engine inverse-transform hook with the reference scale and conversion order.
pub fn restore_projection_with_transform(
    weighted_sums: &[i64],
    prepared: &PreparedActivation,
    inverse_rms: Option<f32>,
    effective_row_scale: &[f32],
    output_signs: Option<&[i8]>,
    rows: usize,
    transform: impl Fn(&mut [f32; ROTATION_BLOCK]),
) -> Result<Vec<f32>> {
    if weighted_sums.len() != effective_row_scale.len()
        || rows > weighted_sums.len()
        || output_signs.is_some_and(|s| {
            s.len() != weighted_sums.len()
                || s.len() % ROTATION_BLOCK != 0
                || s.iter().any(|&x| x != -1 && x != 1)
        })
    {
        return Err(Error::InvalidInput(
            "invalid projection restore dimensions or signs".into(),
        ));
    }
    if inverse_rms.is_some_and(|x| !x.is_finite()) {
        return Err(Error::NonFinite("deferred inverse RMS"));
    }
    let mut output = vec![0.0; weighted_sums.len()];
    for (row, linear) in output.iter_mut().enumerate() {
        // There is exactly one i64->f32 conversion. Restore scales in the
        // specified order; neither reassociation nor an FMA is equivalent.
        *linear = weighted_sums[row] as f32 * prepared.row_quantum();
        if let Some(inverse_rms) = inverse_rms {
            *linear *= inverse_rms;
        }
        *linear *= effective_row_scale[row];
    }
    require_finite(&output, "scaled compressed projection")?;
    if let Some(signs) = output_signs {
        inverse_rotation_with_transform(&mut output, signs, transform)?;
    }
    output.truncate(rows);
    Ok(output)
}

fn inverse_rotation(values: &mut [f32], signs: &[i8]) -> Result<()> {
    inverse_rotation_with_transform(values, signs, hadamard512)
}

fn inverse_rotation_with_transform(
    values: &mut [f32],
    signs: &[i8],
    transform: impl Fn(&mut [f32; ROTATION_BLOCK]),
) -> Result<()> {
    for (block_index, block) in values.chunks_exact_mut(ROTATION_BLOCK).enumerate() {
        let mut rotated = [0.0; ROTATION_BLOCK];
        rotated.copy_from_slice(block);
        transform(&mut rotated);
        for (channel, value) in block.iter_mut().enumerate() {
            // Inverse rotation applies the signs after the normalized transform.
            *value = (rotated[channel] * HADAMARD_NORMALIZATION)
                * f32::from(signs[block_index * ROTATION_BLOCK + channel]);
        }
    }
    require_finite(values, "inverse-rotated projection")
}

/// Stored floating embedding components, including mapped code streams.
pub trait EmbeddingTensor {
    fn descriptor(&self) -> &yskn_format::TensorDescriptor;
    fn table(&self) -> &[f32];
    fn scales(&self) -> &[f32];
    fn input_signs(&self) -> &[i8];
    fn rotated_row(&self, row: usize) -> Result<Vec<f32>>;
    fn rows(&self) -> usize {
        self.descriptor().rows
    }
    fn cols(&self) -> usize {
        self.descriptor().cols
    }
}

impl EmbeddingTensor for Tensor {
    fn descriptor(&self) -> &yskn_format::TensorDescriptor {
        self.descriptor()
    }
    fn table(&self) -> &[f32] {
        self.table()
    }
    fn scales(&self) -> &[f32] {
        self.scales()
    }
    fn input_signs(&self) -> &[i8] {
        self.input_signs()
    }
    fn rotated_row(&self, row: usize) -> Result<Vec<f32>> {
        Ok(self.rotated_row(row)?)
    }
}

#[derive(Debug)]
pub struct Embedding<T = Tensor> {
    main: T,
    residual: Option<T>,
    residual2: Option<T>,
}

impl<T: EmbeddingTensor> Embedding<T> {
    pub fn new(main: T, residual: Option<T>, residual2: Option<T>) -> Result<Self> {
        if residual2.is_some() && residual.is_none() {
            return Err(Error::InvalidModel(
                "a third embedding component requires the second".into(),
            ));
        }
        for component in std::iter::once(&main)
            .chain(residual.iter())
            .chain(residual2.iter())
        {
            if component.rows() != main.rows()
                || component.cols() != main.cols()
                || component.descriptor().output_rotation.is_some()
                || component.descriptor().input_rotation.sign_offset
                    != main.descriptor().input_rotation.sign_offset
            {
                return Err(Error::InvalidModel("embedding components must share dimensions and input sign offset, without output rotation".into()));
            }
            require_finite(component.table(), "embedding table")?;
            require_finite(component.scales(), "embedding scales")?;
        }
        Ok(Self {
            main,
            residual,
            residual2,
        })
    }

    pub fn rows(&self) -> usize {
        self.main.rows()
    }

    pub fn cols(&self) -> usize {
        self.main.cols()
    }

    pub fn row(&self, token: usize) -> Result<Vec<f32>> {
        if token >= self.rows() {
            return Err(Error::InvalidModel(
                "embedding token is outside the vocabulary".into(),
            ));
        }
        let mut values = self.main.rotated_row(token)?;
        // Add residual components in stored rotated coordinates and in order;
        // rounding each component to BF16 here would change the embedding.
        for component in self.residual.iter().chain(self.residual2.iter()) {
            let residual = component.rotated_row(token)?;
            for (value, addition) in values.iter_mut().zip(residual) {
                *value += addition;
            }
        }
        require_finite(&values, "summed embedding components")?;
        inverse_rotation(&mut values, self.main.input_signs())?;
        for value in &mut values {
            // The main embedding signs apply to every component (LIN-02).
            // BF16 rounding occurs once, after the complete inverse rotation.
            *value = bfloat(*value);
        }
        require_finite(&values, "BF16 embedding")?;
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_quantization_uses_half_hook_before_finite_check() {
        use crate::portable_math::{Elementwise, ScalarMath};
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        struct CountingMath;
        impl Elementwise for CountingMath {
            fn scale_half(values: &mut [f32], scale: f32) {
                CALLS.fetch_add(1, Ordering::Relaxed);
                ScalarMath::scale_half(values, scale);
            }
        }
        let input: Vec<_> = (0..1024).map(|i| (i % 17) as f32 - 8.0).collect();
        let actual =
            quantize_activation_with_math::<CountingMath>(&input, &[1; 1024], hadamard512).unwrap();
        let expected = quantize_activation(&input, &[1; 1024]).unwrap();
        assert_eq!(CALLS.load(Ordering::Relaxed), 2);
        assert_eq!(actual.activation8, expected.activation8);
        assert_eq!(actual.integer_scale, expected.integer_scale);
        assert_eq!(actual.activation_scale, expected.activation_scale);
        assert_eq!(actual.row_quantum.to_bits(), expected.row_quantum.to_bits());

        // Input validation must precede the hook; overflow must stop before the next block.
        for (value, context, calls) in [
            (f32::NAN, "pre-rotation activation", 2),
            (f32::MAX, "FP16-rounded rotated activation", 3),
        ] {
            let mut input = vec![0.0; 1024];
            input[0] = value;
            let result =
                quantize_activation_with_math::<CountingMath>(&input, &[1; 1024], hadamard512);
            assert!(matches!(result, Err(Error::NonFinite(actual)) if actual == context));
            assert_eq!(CALLS.load(Ordering::Relaxed), calls);
        }
    }

    #[test]
    fn hadamard_inverse_is_exact_for_small_integer_inputs() {
        let original: [f32; ROTATION_BLOCK] =
            std::array::from_fn(|index| (index % 11) as f32 - 5.0);
        let mut values = original;
        hadamard512(&mut values);
        hadamard512(&mut values);
        for (actual, expected) in values.into_iter().zip(original) {
            assert_eq!(actual.to_bits(), (expected * 512.0).to_bits());
        }
    }

    #[test]
    fn zero_activations_have_zero_scales_and_codes() {
        let prepared = quantize_activation(&[0.0; 1024], &[1; 1024]).unwrap();
        assert!(prepared.activation8.iter().all(|&value| value == 0));
        assert!(prepared.integer_scale.iter().all(|&value| value == 0));
        assert_eq!(prepared.row_quantum.to_bits(), 0);
    }

    #[test]
    fn per_group_quantization_preserves_the_common_scale_grid() {
        let input: Vec<_> = (0..1024).map(|index| (index % 17) as f32 - 8.0).collect();
        let prepared = quantize_activation(&input, &[1; 1024]).unwrap();
        assert!(prepared.activation8.iter().all(|&value| value != -128));
        assert!(
            prepared
                .integer_scale
                .iter()
                .all(|&value| (0..=16_777_216).contains(&value))
        );
        assert!(prepared.integer_scale.contains(&16_777_216));
        assert!(prepared.row_quantum > 0.0);
    }

    #[test]
    fn invalid_activations_fail_before_integer_conversion() {
        let mut input = [0.0; ROTATION_BLOCK];
        input[0] = f32::NAN;
        assert!(matches!(
            quantize_activation(&input, &[1; ROTATION_BLOCK]),
            Err(Error::NonFinite(_))
        ));
        input[0] = f32::MAX;
        assert!(matches!(
            quantize_activation(&input, &[1; ROTATION_BLOCK]),
            Err(Error::NonFinite(_))
        ));
        assert!(quantize_activation(&[0.0; 2], &[1; 2]).is_err());
    }
}
