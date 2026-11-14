use crate::kv_codec::requantize_int8;
use super::synthetic_kv_profile::{Profile, mixed_profile};
use super::*;

fn layer(profile: Profile) -> Attention {
    configured(Arc::new(KvProfile::parse(&profile.bytes()).unwrap()), 0)
}
fn configured(profile: Arc<KvProfile>, ordinal: usize) -> Attention {
    Attention::new(
        AttentionDimensions {
            query_heads: 2,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
        },
        vec![0.0; 256],
        vec![0.0; 256],
        1.0e-6,
    )
    .unwrap()
    .with_kv(KvMode::Kv5, Some(profile), ordinal)
    .unwrap()
}
fn split() -> AttentionSplit {
    AttentionSplit {
        maximum: EMPTY_MAXIMUM,
        denominator: 0.0,
        numerator: vec![0.0; 256],
    }
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn token(layer: &Attention, position: usize, value: f32) -> CacheToken {
    CacheToken {
        keys: (0..2)
            .map(|head| layer.encode_row(head, 0, position, &[0.0; 256]).unwrap())
            .collect(),
        values: (0..2)
            .map(|head| layer.encode_row(head, 1, position, &[value; 256]).unwrap())
            .collect(),
    }
}

#[test]
fn packed_zero_codebook_gives_exact_mean_in_prefix_and_self_splits() {
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(0);
    for group in [1, 3] {
        profile.groups[group].mean = Some([0.25; 256]);
    }
    let layer = layer(profile);
    // Zero codebook, equal scores => denominator 3, numerator 3 * 1/4.
    // Unequal scales make using a sum of rounded/scaled weights observably wrong.
    let prefix = vec![token(&layer, 0, 0.0), token(&layer, 1, 8.0)];
    let current = token(&layer, 2, 2.0);
    let mut s = split();
    layer
        .prefix_tile(&mut s, &[0.0; 256], 0, &prefix, 0, 0..2)
        .unwrap();
    layer
        .add_self(&mut s, &[0.0; 256], &current.keys[0], &current.values[0])
        .unwrap();
    assert_eq!(s.denominator.to_bits(), 3.0_f32.to_bits());
    assert_eq!(bits(&s.numerator), vec![0; 256]);
    Attention::add_value_mean(&mut s, layer.value_mean(0, &current.values[0]).unwrap()).unwrap();
    assert_eq!(bits(&s.numerator), vec![0.75_f32.to_bits(); 256]);
    // 73 prefix entries also exercise distinct splits, including self and empty splits.
    let prefix = vec![prefix[0].clone(); 73];
    assert_eq!(
        bits(
            &layer
                .attend_head(&[0.0; 256], 0, &prefix, &current)
                .unwrap()
        ),
        vec![0.25_f32.to_bits(); 256]
    );
}

#[test]
fn packed_key_mean_is_ignored_and_bf16_base_and_row_scale_define_scores() {
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(257);
    profile.gain = 1.0;
    let plain = layer(profile);
    let mut cache = token(&plain, 0, 0.0);
    cache.keys[0] = plain.encode_row(0, 0, 0, &[2.0; 256]).unwrap();
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(257);
    profile.gain = 1.0;
    profile.groups[0].mean = Some([64.0; 256]);
    let mean = layer(profile);
    for layer in [&plain, &mean] {
        // 257/256 ties to BF16 1.0; scale=2; dot=256*1/4; score=8.
        let mut prefix = split();
        layer
            .prefix_tile(
                &mut prefix,
                &[0.25; 256],
                0,
                std::slice::from_ref(&cache),
                0,
                0..1,
            )
            .unwrap();
        assert_eq!(prefix.maximum.to_bits(), 8.0_f32.to_bits());
        let mut own = split();
        layer
            .add_self(&mut own, &[0.25; 256], &cache.keys[0], &cache.values[0])
            .unwrap();
        assert_eq!(own.maximum.to_bits(), 8.0_f32.to_bits());
    }
}

#[test]
fn prefix_rounds_probability_times_scale_but_self_weight_stays_f32() {
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(256);
    profile.gain = 1.0;
    let layer = layer(profile);
    // Constant unit codebook; RMS=257/256 is representable in half but ties to BF16 1.
    let row = token(&layer, 0, 257.0 / 256.0);
    assert_eq!(
        row.values[0].scale().to_bits(),
        (257.0_f32 / 256.0).to_bits()
    );
    let mut prefix = split();
    layer
        .prefix_tile(
            &mut prefix,
            &[0.0; 256],
            0,
            std::slice::from_ref(&row),
            0,
            0..1,
        )
        .unwrap();
    assert_eq!(bits(&prefix.numerator), vec![1.0_f32.to_bits(); 256]);
    let mut own = split();
    layer
        .add_self(&mut own, &[0.0; 256], &row.keys[0], &row.values[0])
        .unwrap();
    assert_eq!(
        bits(&own.numerator),
        vec![(257.0_f32 / 256.0).to_bits(); 256]
    );
}

#[test]
fn masked_packed_slots_are_never_read_and_do_not_change_output_bits() {
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(256);
    profile.gain = 1.0;
    let layer = layer(profile);
    let mut prefix = vec![token(&layer, 0, 1.0); 64];
    let evaluate = |prefix: &[CacheToken]| {
        let mut s = split();
        layer
            .prefix_tile(&mut s, &[0.0; 256], 0, prefix, 32, 35..38)
            .unwrap();
        (
            s.maximum.to_bits(),
            s.denominator.to_bits(),
            bits(&s.numerator),
        )
    };
    let expected = (0_u32, 3.0_f32.to_bits(), vec![3.0_f32.to_bits(); 256]);
    assert_eq!(evaluate(&prefix), expected);
    for (position, token) in prefix.iter_mut().enumerate() {
        if !(35..38).contains(&position) {
            // Empty head arrays would panic if a masked key OR value were fetched.
            token.keys.clear();
            token.values.clear();
        }
    }
    assert_eq!(evaluate(&prefix), expected);
    prefix.truncate(38);
    assert_eq!(evaluate(&prefix), expected);
}

#[test]
fn prepared_rows_store_packed_sides_fallbacks_and_absolute_guard_positions() {
    let profile = Arc::new(KvProfile::parse(&mixed_profile()).unwrap());
    let packed = configured(Arc::clone(&profile), 0);
    let fallback = configured(Arc::clone(&profile), 1);
    let first = packed
        .prepare_row(0, &[0.0; 1024], &[0.0; 512], &[0.5; 512])
        .unwrap();
    assert!(
        first
            .cache
            .keys
            .iter()
            .chain(&first.cache.values)
            .all(|r| matches!(r, CacheRow::Packed(_)))
    );
    for row in first.cache.keys.iter().chain(&first.cache.values) {
        let CacheRow::Packed(row) = row else {
            unreachable!()
        };
        assert_eq!(row.bits(), 5);
        assert_eq!(row.words().len(), 40);
    }
    for mode in [KvMode::Int8Profile, KvMode::Kv5] {
        let fallback = fallback
            .clone()
            .with_kv(mode, Some(Arc::clone(&profile)), 1)
            .unwrap();
        for position in [0, 3] {
            let prepared = fallback
                .prepare_row(position, &[0.0; 1024], &[0.0; 512], &[0.5; 512])
                .unwrap();
            assert!(
                prepared
                    .cache
                    .values
                    .iter()
                    .all(|r| matches!(r, CacheRow::Int8(_)))
            );
            let rotated = rotate_cache(&[0.5; 256]);
            let expected = requantize_int8(
                &profile,
                5,
                position,
                rotated.as_slice().try_into().unwrap(),
            )
            .unwrap();
            assert_eq!(prepared.cache.values[0], CacheRow::Int8(expected));
            if mode == KvMode::Kv5 {
                assert!(matches!(prepared.cache.keys[0], CacheRow::Packed(_)));
            } else {
                assert!(matches!(prepared.cache.keys[0], CacheRow::Int8(_)));
            }
        }
    }
    let mut state = packed.new_state().unwrap();
    packed.append_prepared(&mut state, 0, &first).unwrap();
    assert_eq!(state.rows[0], first.cache);
}

#[test]
fn encoded_self_is_prepared_row_and_future_cache_rows_do_not_affect_stages() {
    let layer = configured(Arc::new(KvProfile::parse(&mixed_profile()).unwrap()), 0);
    let mut sequential = layer.new_state().unwrap();
    let mut staged = layer.new_state().unwrap();
    let mut prepared = Vec::new();
    let mut expected = Vec::new();
    for position in 0..76 {
        let values = vec![(position % 7) as f32 * 0.125; 512];
        expected.push(bits(
            &layer
                .step(
                    &mut sequential,
                    position,
                    &[0.25; 1024],
                    &[0.5; 512],
                    &values,
                )
                .unwrap(),
        ));
        let row = layer
            .prepare_row(position, &[0.25; 1024], &[0.5; 512], &values)
            .unwrap();
        layer.append_prepared(&mut staged, position, &row).unwrap();
        prepared.push(row);
    }
    for (position, row) in prepared.iter().enumerate() {
        assert_eq!(
            bits(&layer.output_row(&staged, position, row).unwrap()),
            expected[position]
        );
    }
    // Even an unusable cached self entry is irrelevant: self comes from PreparedRow.
    staged.rows[75].keys.clear();
    staged.rows[75].values.clear();
    assert_eq!(
        bits(&layer.output_row(&staged, 75, &prepared[75]).unwrap()),
        expected[75]
    );
}

#[test]
fn codec_failure_after_a_successful_row_preserves_attention_state() {
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(i16::MAX); // general trellis overflow once guard bypass ends
    for group in &mut profile.groups {
        group.guard = 1;
    }
    for mode in [KvMode::Int8Profile, KvMode::Kv5] {
        let layer = layer(Profile::new(1, 2))
            .with_kv(
                mode,
                Some(Arc::new(KvProfile::parse(&profile.bytes()).unwrap())),
                0,
            )
            .unwrap();
        let mut state = layer.new_state().unwrap();
        layer
            .step(&mut state, 0, &[0.0; 1024], &[0.0; 512], &[0.0; 512])
            .unwrap();
        let before = state.clone();
        assert!(
            layer
                .step(&mut state, 1, &[0.0; 1024], &[0.0; 512], &[0.0; 512])
                .is_err()
        );
        assert_eq!(state, before);
    }
}

#[test]
fn prefix_rounds_nonunit_probability_times_scale_before_value_multiply() {
    let mut profile = Profile::new(1, 2);
    profile.codes.fill(256);
    profile.gain = 1.0;
    let layer = layer(profile);
    let mut prefix = vec![token(&layer, 0, 257.0 / 256.0), token(&layer, 1, 1.0)];
    // Query=1, key bases=-1/0, scale=1/16: scores -1 and 0 exactly.
    for (row, integer) in prefix.iter_mut().zip([-1, 0]) {
        row.keys[0] = CacheRow::Int8(Int8Row {
            integers: vec![integer; 256],
            scale: 1.0 / 16.0,
        });
    }
    let mut s = split();
    layer
        .prefix_tile(&mut s, &[1.0; 256], 0, &prefix, 0, 0..2)
        .unwrap();
    // q35_atxpk3.cuh:402-406: p=exp2(-log2(e))=exp(-1), ~0.36787944.
    // p*(257/256) ~0.36931646 rounds to BF16 189/512=0.369140625.
    // Both value bases are 1: numerator=189/512+1=701/512 (0x3faf4000).
    // Moving rounding onto scale instead gives p*1+1 ~1.36787944, different bits.
    assert_eq!(s.maximum.to_bits(), 0);
    assert_eq!(bits(&s.numerator), vec![0x3faf_4000; 256]);
}

#[test]
fn value_guard_alone_disables_packing_and_preserves_initial_int8_until_guard() {
    let mut synthetic = Profile::new(1, 2);
    synthetic.codes.fill(0);
    synthetic.gain = 1.0;
    synthetic.groups[1].guard = 2;
    let profile = Arc::new(KvProfile::parse(&synthetic.bytes()).unwrap());
    assert!(profile.packable(0, 0));
    assert!(!profile.packable(0, 1));
    let layer = configured(profile, 0);
    // Cache-domain values: max=127, initial scale=127/127=1.
    // RN-even gives [-127,-64,0,32]. With no mean/basis and codebook=0,
    // every post-guard reconstruction is 0: final int8 bytes=0, scale=0.
    let row: [f32; 256] = std::array::from_fn(|i| [-127.0, -63.5, 0.0, 31.75][i % 4]);
    let initial: Vec<i8> = (0..256).map(|i| [-127, -64, 0, 32][i % 4]).collect();
    for position in [0, 1, 2, 3] {
        for head in 0..2 {
            let encoded = layer.encode_row(head, 1, position, &row).unwrap();
            let CacheRow::Int8(encoded) = encoded else {
                panic!("guard must disable packing for the entire value side")
            };
            if head == 0 && position < 2 {
                assert_eq!(encoded.integers(), initial);
                assert_eq!(encoded.scale().to_bits(), 1.0_f32.to_bits());
            } else {
                assert_eq!(encoded.integers(), &[0; 256]);
                assert_eq!(encoded.scale().to_bits(), 0);
            }
            assert!(matches!(
                layer.encode_row(head, 0, position, &row).unwrap(),
                CacheRow::Packed(_)
            ));
        }
    }
}

#[test]
fn packed_prepared_row_used_by_plain_int8_output_head_returns_input_error() {
    let packed = layer(Profile::new(1, 2));
    let plain = Attention::new(packed.dimensions, vec![0.0; 256], vec![0.0; 256], 1.0e-6).unwrap();
    let prepared = packed
        .prepare_row(0, &[0.25; 1024], &[0.5; 512], &[0.75; 512])
        .unwrap();
    let state = plain.new_state().unwrap();
    assert!(matches!(
        plain.output_head(&state, 0, &prepared, 0),
        Err(Error::InvalidInput(message)) if message == "packed attention row requires a profile"
    ));
}
