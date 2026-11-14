//! VIT2 lanes follow independent reduced states; cost keys retain predecessor ties.
use crate::Result;
use yskn_ref::{kv_codec, kv_profile::CodeTable};

pub(crate) fn search_chain(
    target: &[i32; 64],
    table: &CodeTable,
    bits: u8,
    vit2: bool,
) -> Result<[u8; 64]> {
    #[cfg(target_arch = "aarch64")]
    if super::enabled()
        && vit2
        && matches!(bits, 4 | 5)
        && target.iter().all(|x| (-2048..=2048).contains(x))
        && table.codes.iter().all(|c| (-1100..=1100).contains(c))
    {
        // SAFETY: NEON detected, dimensions fixed, D13's reduced-cost bound
        // (< 923,638,528 in magnitude) makes every i32 intermediate exact.
        return Ok(unsafe { vit2_neon(target, table, bits) });
    }
    kv_codec::search_chain(target, table, bits, vit2)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vit2_neon(target: &[i32; 64], table: &CodeTable, bits: u8) -> [u8; 64] {
    use std::arch::aarch64::*;
    let fields = 1usize << bits;
    let stride = 256 >> bits;
    let f = fields as i32;
    let ac: [i32; 256] = std::array::from_fn(|d| {
        let c = i32::from(table.codes[d]);
        f * c * c + (d / stride) as i32
    });
    let bc: [i32; 256] = table.codes.map(|c| -2 * f * i32::from(c));
    let mut cost = [f << 24; 16];
    cost[0] = 0;
    let mut bp = [[0u8; 16]; 64];
    let mut final_key = (i32::MAX, usize::MAX);
    // SAFETY: g+4 <= stride <= 16; d+stride+4 <= 256 for the paired case.
    // Read-only cost and separately owned next cannot alias.
    unsafe {
        for (t, &x) in target.iter().enumerate() {
            let mut next = [0; 16];
            let square = x * x * f;
            for g in (0..stride).step_by(4) {
                let mut best = vdupq_n_s32(i32::MAX);
                let step = if bits == 5 { 2 } else { 1 };
                for j in (0..fields).step_by(step) {
                    let d = stride * j + g;
                    let mut emission = vaddq_s32(
                        vld1q_s32(ac.as_ptr().add(d)),
                        vmulq_n_s32(vld1q_s32(bc.as_ptr().add(d)), x),
                    );
                    if bits == 5 {
                        let other = d + stride;
                        emission = vminq_s32(
                            emission,
                            vaddq_s32(
                                vld1q_s32(ac.as_ptr().add(other)),
                                vmulq_n_s32(vld1q_s32(bc.as_ptr().add(other)), x),
                            ),
                        );
                    }
                    best = vminq_s32(best, vaddq_s32(emission, vdupq_n_s32(cost[d >> bits])));
                }
                let mut keys = [0; 4];
                vst1q_s32(keys.as_mut_ptr(), vaddq_s32(best, vdupq_n_s32(square)));
                for (lane, k) in keys.into_iter().enumerate() {
                    if t == 63 {
                        let state = stride * (k as usize & (fields - 1)) + g + lane;
                        final_key = final_key.min((k >> bits, state));
                    } else {
                        let k = k - cost[0];
                        next[g + lane] = k & !(f - 1);
                        bp[t + 1][g + lane] = (k & (f - 1)) as u8;
                    }
                }
            }
            if t < 63 {
                cost = next;
            }
        }
    }
    let mut state = final_key.1;
    let mut path = [0; 64];
    for t in (1..64).rev() {
        path[t] = state as u8;
        state = (usize::from(bp[t][state >> bits]) << (8 - bits)) | (state >> bits);
    }
    path[0] = state as u8;
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "one-core timing, run explicitly on a quiet machine"]
    fn bench_search_chain() {
        let table = CodeTable {
            gain: 1.0,
            codes: std::array::from_fn(|i| ((i * 73 + 19) % 2049) as i16 - 1024),
        };
        let target = std::array::from_fn(|i| ((i * 137) % 4097) as i32 - 2048);
        for (name, search) in [
            ("reference", kv_codec::search_chain as fn(_, _, _, _) -> _),
            ("auto", search_chain),
        ] {
            let mut times = Vec::new();
            for _ in 0..9 {
                let start = std::time::Instant::now();
                for _ in 0..1000 {
                    std::hint::black_box(
                        search(std::hint::black_box(&target), &table, 5, true).unwrap(),
                    );
                }
                times.push(start.elapsed().as_secs_f64() / 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!("{name} chain min_s={} median_s={}", times[0], times[4]);
        }
    }

    #[test]
    fn exact_paths_include_ties_clamps_and_checked_fallback() {
        let mut seed = 71u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for bits in [3, 4, 5] {
            for case in 0..120 {
                let table = CodeTable {
                    gain: 1.0,
                    codes: std::array::from_fn(|_| match case {
                        0..=3 => 0,
                        4 => -1100,
                        5 => 1100,
                        6 => i16::MAX,
                        _ => (random() % 2201) as i16 - 1100,
                    }),
                };
                let target = std::array::from_fn(|_| match case {
                    0 | 4 => -2048,
                    1 | 5 => 2048,
                    2 => 0,
                    _ => (random() % 4097) as i32 - 2048,
                });
                let vit2 = bits != 3;
                let expected = kv_codec::search_chain(&target, &table, bits, vit2);
                let actual = search_chain(&target, &table, bits, vit2);
                match (actual, expected) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b, "bits={bits} case={case}"),
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    other => panic!("{other:?}"),
                }
            }
        }
    }

    #[test]
    fn selected_bytes_and_scale_match_scalar_on_random_and_tied_rows() {
        use yskn_ref::kv_profile::KvProfile;
        let profile =
            KvProfile::parse(&crate::model::session_test_support::kv_profile::mixed_profile())
                .unwrap();
        let mut seed = 123u32;
        for case in 0..40 {
            let row = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                match case {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::from_bits(1),
                    3 => 1.0,
                    _ => seed as i32 as f32 / i32::MAX as f32 * 4.0,
                }
            });
            for group in 0..8 {
                for position in [0, 3] {
                    let expected =
                        kv_codec::encode_packed(&profile, group, position, &row).unwrap();
                    let actual = kv_codec::encode_packed_with_search(
                        &profile,
                        group,
                        position,
                        &row,
                        search_chain,
                    )
                    .unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(actual.scale().to_bits(), expected.scale().to_bits());
                    assert_eq!(
                        kv_codec::requantize_int8_with_search(
                            &profile,
                            group,
                            position,
                            &row,
                            search_chain
                        )
                        .unwrap(),
                        kv_codec::requantize_int8(&profile, group, position, &row).unwrap()
                    );
                }
            }
        }
    }
}
