//! NEON lanes are independent value channels; each keeps four ordered 32-term sums.
use crate::Result;
use yskn_ref::gated_deltanet::{GatedDeltaNet, GatedDeltaNetRow};

pub(crate) fn recur_head(
    layer: &GatedDeltaNet,
    state: &mut [f32],
    source: Option<&[f32]>,
    row: &GatedDeltaNetRow,
    head: usize,
) -> Result<Vec<f32>> {
    #[cfg(target_arch = "aarch64")]
    if super::enabled() {
        return layer.recur_head_with(state, row, head, |state, query, key, value, decay, gate| {
            if let Some(source) = source {
                assert_eq!(source.len(), state.len());
            }
            // SAFETY: runtime NEON detection; the reference validates 128x128 state and
            // 128-element operands. Source, if supplied, has the same extent and cannot alias state.
            unsafe { core(state, source, query, key, value, decay, gate) }
        });
    }
    if let Some(source) = source {
        state.copy_from_slice(source);
    }
    layer.recur_head(state, row, head)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn core(
    state: &mut [f32],
    source: Option<&[f32]>,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    decay: f32,
    gate: f32,
) -> Result<Vec<f32>> {
    use std::arch::aarch64::*;
    use yskn_ref::portable_math::bfloat;
    let mut output = vec![0.0; 128];
    // SAFETY: callers validate all extents; j steps by four through 128 columns,
    // and k visits 128 rows. Loads/stores use those exact bounds. No fused operations.
    unsafe {
        // All state accesses derive from one exclusive pointer, including in-place reads.
        let state_ptr = state.as_mut_ptr();
        let old = source.map_or(state_ptr.cast_const(), |s| s.as_ptr());
        for j in (0..128).step_by(4) {
            let mut parts = [vdupq_n_f32(0.0); 4];
            for (group, part) in parts.iter_mut().enumerate() {
                for offset in 0..32 {
                    let k = group * 32 + offset;
                    let index = k * 128 + j;
                    let decayed = vmulq_n_f32(vld1q_f32(old.add(index)), decay);
                    vst1q_f32(state_ptr.add(index), decayed);
                    *part = vaddq_f32(*part, vmulq_n_f32(decayed, key[k]));
                }
            }
            let prediction =
                vaddq_f32(vaddq_f32(vaddq_f32(parts[0], parts[1]), parts[2]), parts[3]);
            let correction = vmulq_n_f32(
                vsubq_f32(vld1q_f32(value.as_ptr().add(j)), prediction),
                gate,
            );
            let mut check = [0.0; 8];
            vst1q_f32(check.as_mut_ptr(), prediction);
            vst1q_f32(check.as_mut_ptr().add(4), correction);
            if !check.iter().all(|x| x.is_finite()) {
                return Err(crate::Error::NonFinite("DeltaNet recurrent prediction"));
            }
            let mut parts = [vdupq_n_f32(0.0); 4];
            for (group, part) in parts.iter_mut().enumerate() {
                for offset in 0..32 {
                    let k = group * 32 + offset;
                    let index = k * 128 + j;
                    let updated = vaddq_f32(
                        vld1q_f32(state_ptr.add(index)),
                        vmulq_n_f32(correction, key[k]),
                    );
                    vst1q_f32(state_ptr.add(index), updated);
                    *part = vaddq_f32(*part, vmulq_n_f32(updated, query[k]));
                }
            }
            vst1q_f32(
                output.as_mut_ptr().add(j),
                vaddq_f32(vaddq_f32(vaddq_f32(parts[0], parts[1]), parts[2]), parts[3]),
            );
        }
    }
    for x in &mut output {
        *x = bfloat(*x);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yskn_ref::gated_deltanet::{GatedDeltaNetDimensions, GatedDeltaNetWeights};
    #[test]
    fn independent_value_lanes_and_out_of_place_state_equal_scalar_bits() {
        let dimensions = GatedDeltaNetDimensions {
            hidden_width: 512,
            query_key_heads: 1,
            value_heads: 2,
            head_dim: 128,
            convolution_width: 4,
        };
        let layer = GatedDeltaNet::new(
            dimensions,
            GatedDeltaNetWeights {
                projection_a: vec![0.001; 1024],
                projection_b: vec![-0.002; 1024],
                convolution_weight: vec![0.25; 2048],
                log_decay_weight: vec![-2.0; 2],
                time_bias: vec![0.0; 2],
                normalization_weight: vec![1.0; 128],
            },
            1e-6,
        )
        .unwrap();
        let mut seed = 17u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            match seed % 16 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1),
                3 => f32::from_bits(0x80000001),
                4 => f32::from_bits(0x3f808000),
                _ => seed as i32 as f32 / i32::MAX as f32,
            }
        };
        let mut source: Vec<_> = (0..128 * 128).map(|_| next()).collect();
        for iteration in 0..20 {
            let qkv: Vec<_> = (0..512).map(|_| next()).collect();
            let z: Vec<_> = (0..256).map(|_| next()).collect();
            let hidden: Vec<_> = (0..512).map(|_| next()).collect();
            let history = vec![0.0; 512];
            let row = layer
                .prepare(&[&history, &history, &history], &qkv, &z, &hidden, 1.0)
                .unwrap();
            let mut expected = source.clone();
            let expected_output = layer
                .recur_head(&mut expected, &row, iteration % 2)
                .unwrap();
            let mut actual = vec![f32::NAN; 128 * 128];
            let actual_output =
                recur_head(&layer, &mut actual, Some(&source), &row, iteration % 2).unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(bits(&actual_output), bits(&expected_output));
            let mut inplace = source.clone();
            assert_eq!(
                bits(&recur_head(&layer, &mut inplace, None, &row, iteration % 2).unwrap()),
                bits(&expected_output)
            );
            assert_eq!(bits(&inplace), bits(&expected));
            source = actual;
        }
        let zero = vec![0.0; 512];
        let z = vec![0.0; 256];
        let row = layer
            .prepare(&[&zero, &zero, &zero], &zero, &z, &zero, 1.0)
            .unwrap();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut a = vec![0.0; 128 * 128];
            a[100] = invalid;
            let mut b = a.clone();
            assert_eq!(
                recur_head(&layer, &mut a, None, &row, 0)
                    .unwrap_err()
                    .to_string(),
                layer.recur_head(&mut b, &row, 0).unwrap_err().to_string()
            );
        }
    }
}
