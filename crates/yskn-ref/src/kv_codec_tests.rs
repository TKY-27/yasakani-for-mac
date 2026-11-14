use super::*;

fn put16(b: &mut [u8], o: usize, n: u16) {
    b[o..o + 2].copy_from_slice(&n.to_le_bytes());
}
fn put32(b: &mut [u8], o: usize, n: u32) {
    b[o..o + 4].copy_from_slice(&n.to_le_bytes());
}
fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}

// Synthetic bytes only: never embed or copy the model's profile/table payload.
fn fixture(layers: u16, heads: u16, bits: u8) -> Vec<u8> {
    let ng = usize::from(layers) * usize::from(heads) * 2;
    let mut b = vec![0; 100 + 12 + 4 + 4 * ng];
    b[..4].copy_from_slice(b"KVQP");
    for (o, v) in [
        (4, 1),
        (6, 96),
        (16, 256),
        (18, layers),
        (20, heads),
        (22, 1),
        (24, 3),
    ] {
        put16(&mut b, o, v);
    }
    put32(&mut b, 28, ng as u32);
    for (o, v) in [(32, 100), (36, 112), (40, 116)] {
        put32(&mut b, o, v);
    }
    for (i, m) in [0.9f32, 1.0, 1.1].iter().enumerate() {
        put32(&mut b, 100 + 4 * i, m.to_bits());
    }
    let to = b.len();
    put32(&mut b, 112, to as u32);
    b.resize(to + 524, 0);
    b[to] = 8;
    b[to + 1] = bits;
    b[to + 2] = 64;
    b[to + 3] = 8;
    put32(&mut b, to + 4, 1.0f32.to_bits());
    for s in 0..256 {
        put16(
            &mut b,
            to + 12 + 2 * s,
            (((s * 73 + 19) % 2049) as i16 - 1024) as u16,
        );
    }
    for g in 0..ng {
        let go = b.len();
        put32(&mut b, 116 + 4 * g, go as u32);
        b.extend_from_slice(&[0, bits, 0, 0, 0, 0, 0, 0]);
    }
    let n = b.len();
    put32(&mut b, 8, n as u32);
    b
}
fn replace_group(b: &mut Vec<u8>, group: usize, mode: u8, guard: u8, basis: u8, mean: bool) {
    let og = u32_at(b, 40);
    let old = u32_at(b, og + 4 * group);
    let mut header: [u8; 8] = b[old..old + 8].try_into().unwrap();
    header[0] = mode;
    header[3] = guard;
    header[4] = basis;
    header[5] = u8::from(mean);
    let go = b.len();
    put32(b, og + 4 * group, go as u32);
    b.extend_from_slice(&header);
    for vector in 0..usize::from(basis.min(2)) + usize::from(mean) {
        for i in 0..256 {
            let v = if mean && vector == 0 {
                (i as f32 - 128.0) / 1024.0
            } else {
                0.0625 * if (i >> vector) & 1 == 0 { 1.0 } else { -1.0 }
            };
            b.extend_from_slice(&yskn_format::f32_to_f16(v).to_le_bytes());
        }
    }
    let n = b.len();
    put32(b, 8, n as u32);
}
fn rng(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}
fn row(seed: &mut u32) -> [f32; 256] {
    std::array::from_fn(|_| {
        let mut x = 0.0;
        for _ in 0..12 {
            x += (rng(seed) >> 8) as f32 / 16777216.0;
        }
        (x - 6.0) * 0.75
    })
}

#[test]
fn kv_profile_valid_shapes_and_dimension_contract() {
    for (l, h) in [(1, 1), (3, 2)] {
        let mut b = fixture(l, h, 5);
        b[12..16].fill(0xff);
        b[26..28].fill(0xff);
        b[44..96].fill(0xff);
        let p = KvProfile::parse(&b).unwrap();
        assert_eq!(p.groups().len(), usize::from(l * h * 2));
        p.check_dimensions(l.into(), h.into(), 256).unwrap();
        assert!(p.check_dimensions(l.into(), h.into(), 128).is_err());
        assert!(p.check_dimensions(99, h.into(), 256).is_err());
        assert!(p.check_dimensions(l.into(), 99, 256).is_err());
        assert!(p.packable(0, 0));
        assert!(!p.packable(l.into(), 0));
        assert!(!p.packable(0, 2));
    }
}

#[test]
fn kv_profile_malformed_files_are_errors_never_panics() {
    let b = fixture(1, 1, 5);
    for length in 0..b.len() {
        assert!(
            KvProfile::parse(&b[..length]).is_err(),
            "truncated {length}"
        );
        if length >= 12 {
            let mut short = b[..length].to_vec();
            put32(&mut short, 8, length as u32);
            assert!(
                KvProfile::parse(&short).is_err(),
                "truncated payload {length}"
            );
        }
    }
    for (offset, value, width) in [
        (0, 0, 4),
        (4, 2, 2),
        (6, 100, 2),
        (8, 0, 4),
        (16, 128, 2),
        (18, 0, 2),
        (20, 0, 2),
        (22, 2, 2),
        (24, 0, 2),
        (24, 17, 2),
        (28, 3, 4),
        (32, u32::MAX, 4),
        (36, u32::MAX, 4),
        (40, u32::MAX, 4),
        (32, 0, 4),
        (36, 100, 4),
        (112, u32::MAX, 4),
        (116, u32::MAX, 4),
    ] {
        let mut bad = b.clone();
        if width == 2 {
            put16(&mut bad, offset, value as u16);
        } else {
            put32(&mut bad, offset, value);
        }
        assert!(
            KvProfile::parse(&bad).is_err(),
            "offset {offset} value {value}"
        );
    }
    let to = u32_at(&b, 112);
    let go = u32_at(&b, 116);
    for (o, v) in [
        (to, 7),
        (to + 2, 63),
        (to + 3, 7),
        (go + 1, 2),
        (go + 2, 1),
        (go + 4, 2),
        (go + 5, 1),
    ] {
        let mut bad = b.clone();
        bad[o] = v;
        assert!(KvProfile::parse(&bad).is_err(), "offset {o}");
    }
    for (o, bits) in [
        (100, f32::NAN.to_bits()),
        (104, f32::NEG_INFINITY.to_bits()),
        (to + 4, f32::INFINITY.to_bits()),
    ] {
        let mut bad = b.clone();
        put32(&mut bad, o, bits);
        assert!(KvProfile::parse(&bad).is_err());
    }
}

// Independent oracle: forward edges from reachable states, unbounded i64 costs.
// No compressed recurrence, finite infinity, inverse-state indexing or subject helpers.
fn oracle(x: &[i32; 64], table: &CodeTable, bits: u8) -> i64 {
    let mut costs = [i64::MAX; 256];
    costs[0] = 0;
    for &target in x {
        let mut next = [i64::MAX; 256];
        for (previous, &cost) in costs.iter().enumerate() {
            if cost == i64::MAX {
                continue;
            }
            for symbol in 0..1usize << bits {
                let s = ((previous << bits) | symbol) & 255;
                let delta = i64::from(target) - i64::from(table.codes[s]);
                next[s] = next[s].min(cost + delta * delta);
            }
        }
        costs = next;
    }
    *costs.iter().min().unwrap()
}
fn path_cost(path: &[u8; 64], x: &[i32; 64], table: &CodeTable, bits: u8) -> i64 {
    let mut previous = 0;
    let mut cost = 0;
    for (&s, &x) in path.iter().zip(x) {
        assert_eq!(
            (usize::from(previous) << bits | (usize::from(s) & ((1 << bits) - 1))) & 255,
            usize::from(s)
        );
        let d = i64::from(x) - i64::from(table.codes[usize::from(s)]);
        cost += d * d;
        previous = s;
    }
    cost
}

#[test]
fn kv_trellis_general_and_vit2_match_independent_i64_optimum() {
    let mut seed = 0x31415926;
    for bits in [3, 4, 5] {
        let p = KvProfile::parse(&fixture(1, 1, bits)).unwrap();
        let table = &p.tables()[0];
        for case in 0..24 {
            let x = std::array::from_fn(|i| match case {
                0 => -2048,
                1 => 2048,
                2 => {
                    if i % 2 == 0 {
                        -2048
                    } else {
                        2048
                    }
                }
                _ => (rng(&mut seed) % 4097) as i32 - 2048,
            });
            let optimum = oracle(&x, table, bits);
            let general = general_chain(&x, table, bits).unwrap();
            assert_eq!(path_cost(&general, &x, table, bits), optimum);
            if bits != 3 {
                let vit2 = vit2_chain(&x, table, bits).unwrap();
                assert_eq!(path_cost(&vit2, &x, table, bits), optimum);
                assert_eq!(general, vit2, "tie path bits={bits} case={case}");
            }
        }
    }
}

#[test]
fn kv_tie_rules_and_overflow_errors() {
    let zeros = CodeTable {
        gain: 1.0,
        codes: [0; 256],
    };
    for bits in [4, 5] {
        assert_eq!(general_chain(&[0; 64], &zeros, bits).unwrap(), [0; 64]);
        assert_eq!(vit2_chain(&[0; 64], &zeros, bits).unwrap(), [0; 64]);
    }
    let extreme = CodeTable {
        gain: 1.0,
        codes: [i16::MIN; 256],
    };
    assert!(general_chain(&[2048; 64], &extreme, 5).is_err());
    assert!(vit2_chain(&[2048; 64], &extreme, 5).is_err());
    let mut b = fixture(1, 1, 5);
    let to = u32_at(&b, 112);
    put16(&mut b, to + 12, 1101);
    assert!(!KvProfile::parse(&b).unwrap().vit2_eligible());
    let p3 = KvProfile::parse(&fixture(1, 1, 3)).unwrap();
    assert!(!p3.vit2_eligible());
}

#[test]
fn kv_packing_roundtrip_and_selected_reconstruction() {
    let mut seed = 0x12345678;
    for bits in [4, 5] {
        let fields = std::array::from_fn(|i| (i as u8) & ((1 << bits) - 1));
        let packed = PackedRow {
            words: pack(&fields, bits),
            bits,
            scale: 1.0,
            bases: [0; 256],
        };
        for (i, &expected) in fields.iter().enumerate() {
            assert_eq!(field(&packed, i), expected);
        }
        assert_eq!(
            pack(&std::array::from_fn(|i| field(&packed, i)), bits),
            packed.words
        );
        assert_eq!(
            packed.words[0],
            if bits == 4 { 0x01234567 } else { 0x00443214 }
        );
        let p = KvProfile::parse(&fixture(1, 1, bits)).unwrap();
        for _ in 0..4 {
            let x = row(&mut seed);
            let s = select(&p, 0, 0, &x).unwrap();
            let encoded = encode_packed(&p, 0, 0, &x).unwrap();
            let CacheRow::Packed(ref pk) = encoded else {
                panic!()
            };
            let decoded = decode_packed(&p, 0, pk).unwrap();
            let cached = encoded.bases();
            assert_eq!(pk.scale.to_bits(), s.scale.to_bits());
            for i in 0..256 {
                assert_eq!(packed_state(pk, i), s.states[i]);
                let code = p.tables()[0].reconstruction(s.states[i]);
                assert_eq!(decoded[i].to_bits(), bfloat(code).to_bits());
                assert_eq!(cached[i].to_bits(), decoded[i].to_bits());
                assert_eq!(s.reconstruction[i].to_bits(), (code * pk.scale).to_bits());
                assert_eq!(encoded.key_base(&p, 0, i).unwrap(), decoded[i]);
                assert_eq!(encoded.value_base(&p, 0, i).unwrap(), decoded[i]);
            }
        }
    }
}

#[test]
fn kv_guard_basis_mean_flat_fallback_and_row_independence() {
    let mut b = fixture(1, 2, 5);
    replace_group(&mut b, 1, 1, 2, 2, true);
    replace_group(&mut b, 3, 2, 0, 1, true);
    let p = KvProfile::parse(&b).unwrap();
    assert!(!p.packable(0, 1));
    let mut seed = 777;
    let x = row(&mut seed);
    let guarded = requantize_int8(&p, 1, 1, &x).unwrap();
    let initial = initial_int8(&x).unwrap();
    let y = std::array::from_fn(|i| f32::from(initial.integers[i]) * initial.scale);
    assert_eq!(guarded, initial_int8(&y).unwrap());
    // Independent formula for mean + sequential half-rounded basis projection,
    // flat 4-bit residual quantization and final int8 store.
    for group in [1, 3] {
        let g = &p.groups()[group];
        let mut residual = y;
        for (r, m) in residual.iter_mut().zip(g.mean.unwrap()) {
            *r -= m;
        }
        let mut coefficients = Vec::new();
        for basis in &g.basis {
            // Independent scalar dot; these fixtures keep its final half rounding stable.
            let dot = residual
                .iter()
                .zip(basis)
                .fold(0.0, |sum, (r, u)| sum + r * u);
            let c = half(dot);
            coefficients.push(c);
            for (r, u) in residual.iter_mut().zip(basis) {
                *r -= c * u;
            }
        }
        let mut rec = residual;
        if g.mode == 1 {
            let scale = rec.iter().map(|x| x.abs()).fold(0.0, f32::max) / 7.0;
            rec = rec.map(|r| (r / scale).round_ties_even().clamp(-7.0, 7.0) * scale);
        }
        for (basis, c) in g.basis.iter().zip(coefficients) {
            for (r, u) in rec.iter_mut().zip(basis) {
                *r += c * u;
            }
        }
        for (r, m) in rec.iter_mut().zip(g.mean.unwrap()) {
            *r += m;
        }
        assert_eq!(
            requantize_int8(&p, group, 2, &x).unwrap(),
            initial_int8(&rec).unwrap()
        );
        assert!(matches!(
            encode_packed(&p, group, 2, &x).unwrap(),
            CacheRow::Int8(_)
        ));
    }
    let expected = encode_packed(&p, 0, 100, &x).unwrap();
    let expected_int8 = requantize_int8(&p, 1, 3, &x).unwrap();
    for _ in 0..6 {
        let other = row(&mut seed);
        encode_packed(&p, 0, 0, &other).unwrap();
        requantize_int8(&p, 1, 3, &other).unwrap();
    }
    assert_eq!(encode_packed(&p, 0, 100, &x).unwrap(), expected);
    assert_eq!(requantize_int8(&p, 1, 3, &x).unwrap(), expected_int8);
    assert!(encode_packed(&p, 99, 0, &x).is_err());
    assert!(encode_packed(&p, 0, 0, &[f32::NAN; 256]).is_err());
    assert!(encode_packed(&p, 0, 0, &[f32::MAX; 256]).is_err());
    assert!(
        requantize_int8(&p, 0, 0, &[0.0; 256])
            .unwrap()
            .scale
            .is_finite()
    );
}

use sha2::{Digest, Sha256};

fn encoded_bytes(encoded: &CacheRow) -> Vec<u8> {
    let mut bytes = Vec::new();
    match encoded {
        CacheRow::Int8(r) => {
            bytes.push(8);
            bytes.extend_from_slice(&r.scale.to_bits().to_le_bytes());
            bytes.extend(r.integers.iter().map(|&i| i as u8));
        }
        CacheRow::Packed(r) => {
            bytes.push(r.bits);
            bytes.extend_from_slice(&r.scale.to_bits().to_le_bytes());
            for w in &r.words {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    bytes
}
#[test]
fn kv_frozen_sha256_synthetic_outputs() {
    assert_eq!(
        format!("{:x}", Sha256::digest(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    for (bits, expected) in [
        (
            4,
            "03ff805b36bd010e21ec2c5b722fa806811e21fd85e0710f2cf59c40d4263d0b",
        ),
        (
            5,
            "1935bdce14986fc445e26499fe2b942aa0ebd0ef00b7eed6ac47e48402b60bb7",
        ),
    ] {
        let mut b = fixture(2, 2, bits);
        replace_group(&mut b, 1, 1, 1, 2, true);
        replace_group(&mut b, 3, 0, 0, 0, true);
        replace_group(&mut b, 5, 0, 0, 1, true);
        replace_group(&mut b, 7, 2, 0, 0, false);
        let p = KvProfile::parse(&b).unwrap();
        let mut seed = 0x05a0c0de;
        let mut bytes = Vec::new();
        for _ in 0..4 {
            let x = row(&mut seed);
            for group in 0..8 {
                for pos in [0, 7] {
                    bytes.extend(encoded_bytes(&encode_packed(&p, group, pos, &x).unwrap()));
                    bytes.extend(encoded_bytes(&CacheRow::Int8(
                        requantize_int8(&p, group, pos, &x).unwrap(),
                    )));
                }
            }
        }
        let actual = format!("{:x}", Sha256::digest(&bytes));
        println!("frozen {bits} {actual}");
        assert_eq!(actual, expected);
    }
}

#[test]
fn kv_first_strict_scale_minimum_and_bfloat_decode_boundary() {
    let mut b = fixture(1, 1, 5);
    let to = u32_at(&b, 112);
    for i in 0..256 {
        put16(&mut b, to + 12 + 2 * i, 0);
    }
    let p = KvProfile::parse(&b).unwrap();
    let selected = select(&p, 0, 0, &[0.0; 256]).unwrap();
    assert_eq!(selected.scale.to_bits(), half(0.9).to_bits());
    assert_eq!(selected.states, [0; 256]);
    let mut b = fixture(1, 1, 4);
    let to = u32_at(&b, 112);
    put32(&mut b, to + 4, 1.003f32.to_bits());
    let p = KvProfile::parse(&b).unwrap();
    let mut seed = 567;
    let encoded = encode_packed(&p, 0, 0, &row(&mut seed)).unwrap();
    let CacheRow::Packed(pk) = encoded else {
        panic!()
    };
    let decoded = decode_packed(&p, 0, &pk).unwrap();
    assert!(
        decoded
            .iter()
            .enumerate()
            .any(|(i, &x)| x.to_bits()
                != p.tables()[0].reconstruction(packed_state(&pk, i)).to_bits())
    );
    assert!(decode_packed(&KvProfile::parse(&fixture(1, 1, 5)).unwrap(), 0, &pk).is_err());
}

// For all clamped targets, bound each source expression, not just sampled paths.
// For bits 4/5, d=U*j+g gives d>>bits independent of g. Thus the spread of
// reduced costs is bounded by the spread of emission costs at fixed j.
fn prove_real_overflow_bounds(table: &CodeTable) {
    let c = table
        .codes
        .iter()
        .map(|&c| i64::from(c).abs())
        .max()
        .unwrap();
    let difference = 2048 + c;
    let general_bound = 0x4000_0000i64 + 64 * difference * difference;
    assert!(general_bound <= i64::from(i32::MAX));
    for bits in [4, 5] {
        let f = 1i64 << bits;
        let u = 256usize >> bits;
        let mut spread = 0i64;
        for j in 0..f as usize {
            for g in 0..u {
                for h in 0..u {
                    let a = i64::from(table.codes[u * j + g]);
                    let b = i64::from(table.codes[u * j + h]);
                    for x in [-2048i64, 2048] {
                        spread = spread.max((f * (a * a - b * b - 2 * x * (a - b))).abs());
                    }
                }
            }
        }
        // At initialization M[0]=0, others=F<<24. After any step:
        // M[g] = min_j(F*(X-c)^2 + M[previous] + j) - M[0], low bits cleared.
        // min(M)-M[0] >= -spread; choosing a j with previous=0 gives upper
        // bound F*difference^2. This is invariant for every target sequence.
        let ac = f * c * c + (f - 1);
        let bcx = 2 * f * c * 2048;
        let initial = f << 24;
        let upper = initial.max(f * difference * difference);
        let intermediate = ac + bcx + upper + f * 2048 * 2048;
        let subtract = intermediate + spread; // M[0] may be negative
        assert!(subtract <= i64::from(i32::MAX));
        assert!(-spread - bcx - upper >= i64::from(i32::MIN));
        println!("overflow bound bits={bits}: general={general_bound} vit2_abs<={subtract}");
    }
}

#[test]
fn kv_real_profile_facts_quality_and_overflow() {
    let Some(dir) = std::env::var_os("YSKN_MODEL_DIR") else {
        println!("skipped: YSKN_MODEL_DIR is not set");
        return;
    };
    let b = std::fs::read(std::path::Path::new(&dir).join("kvq_u5.bin"))
        .expect("YSKN_MODEL_DIR profile must be readable");
    assert_eq!(b.len(), 17432);
    let p = KvProfile::parse(&b).unwrap();
    p.check_dimensions(16, 4, 256).unwrap();
    assert_eq!(p.groups().len(), 128);
    assert_eq!(p.tables().len(), 2);
    assert_eq!(
        p.multipliers(),
        &[0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.07, 1.1]
    );
    assert!(p.groups().iter().all(|g| g.mode == 0 && g.bits == 5));
    assert!(p.groups().iter().enumerate().all(|(i, g)| g.table == i % 2));
    assert_eq!(p.packed_bits(), 5);
    assert!(p.vit2_eligible());
    assert_eq!(
        (0..16).filter(|&l| p.packable(l, 0)).collect::<Vec<_>>(),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(
        (0..16).filter(|&l| !p.packable(l, 1)).collect::<Vec<_>>(),
        vec![10, 14, 15]
    );
    assert_eq!(
        p.groups()
            .iter()
            .enumerate()
            .filter(|(_, g)| g.guard != 0)
            .map(|(i, g)| (i, g.guard))
            .collect::<Vec<_>>(),
        vec![(83, 1)]
    );
    assert_eq!(
        p.groups()
            .iter()
            .enumerate()
            .filter(|(_, g)| g.basis_count != 0)
            .map(|(i, g)| (i, g.basis_count))
            .collect::<Vec<_>>(),
        vec![(115, 1), (121, 2), (123, 1), (125, 1), (127, 2)]
    );
    assert_eq!(
        p.groups()
            .iter()
            .enumerate()
            .filter(|(_, g)| g.mean.is_some())
            .map(|(i, _)| i)
            .collect::<Vec<_>>(),
        vec![
            3, 5, 7, 13, 63, 71, 77, 95, 98, 99, 103, 106, 109, 113, 115, 117, 119, 121, 123, 125,
            127
        ]
    );
    for (i, t) in p.tables().iter().enumerate() {
        assert_eq!(t.codes.iter().min(), Some(&-886));
        assert_eq!(t.codes.iter().max(), Some(&874));
        assert_eq!(t.gain, if i == 0 { 1.0015934 } else { 1.0 });
        prove_real_overflow_bounds(t);
        let mut seed = 0x0badbeef;
        for case in 0..8 {
            let x = std::array::from_fn(|j| match case {
                0 => -2048,
                1 => 2048,
                2 => {
                    if j % 2 == 0 {
                        -2048
                    } else {
                        2048
                    }
                }
                _ => (rng(&mut seed) % 4097) as i32 - 2048,
            });
            let optimum = oracle(&x, t, 5);
            assert_eq!(
                path_cost(&general_chain(&x, t, 5).unwrap(), &x, t, 5),
                optimum
            );
            assert_eq!(path_cost(&vit2_chain(&x, t, 5).unwrap(), &x, t, 5), optimum);
        }
    }
    // Same mean/basis preparation, same initial int8 row, but mode 1 residual.
    let mut flat_bytes = b;
    let og = u32_at(&flat_bytes, 40);
    for group in 0..128 {
        let go = u32_at(&flat_bytes, og + 4 * group);
        flat_bytes[go] = 1;
        flat_bytes[go + 1] = 4;
    }
    let flat = KvProfile::parse(&flat_bytes).unwrap();
    let mut seed = 0xdecafbad;
    let mut e5 = 0.0f64;
    let mut e4 = 0.0f64;
    for group in 0..128 {
        let x = row(&mut seed);
        let five = select(&p, group, 100, &x).unwrap();
        let four = select(&flat, group, 100, &x).unwrap();
        let mse = |r: &[f32; 256]| {
            x.iter()
                .zip(r)
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum::<f64>()
                / 256.0
        };
        let a = mse(&five.reconstruction);
        let b = mse(&four.reconstruction);
        assert!(a < b, "group {group}: five={a} four={b}");
        e5 += a;
        e4 += b;
        let encoded = encode_packed(&p, group, 100, &x).unwrap();
        if let CacheRow::Packed(ref pk) = encoded {
            let bases = decode_packed(&p, group, pk).unwrap();
            assert_eq!(
                encoded.value_mean(&p, group, 0).unwrap(),
                p.groups()[group].mean.as_ref().map_or(0.0, |m| m[0])
            );
            assert_eq!(encoded.key_base(&p, group, 0).unwrap(), bases[0]);
        }
    }
    println!(
        "real seeded rows: mean MSE 5-bit={} flat4={}",
        e5 / 128.0,
        e4 / 128.0
    );
}

#[test]
fn kv_initial_int8_rounding_and_packed_mean_stay_separate() {
    let mut x = [0.0; 256];
    x[..7].copy_from_slice(&[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]);
    let initial = initial_int8(&x).unwrap();
    assert_eq!(initial.scale, 1.0);
    assert_eq!(&initial.integers[..7], &[127, 0, 2, 2, 0, -2, -2]);
    let mut bytes = fixture(1, 1, 5);
    replace_group(&mut bytes, 0, 0, 0, 0, true);
    let p = KvProfile::parse(&bytes).unwrap();
    assert!(p.packable(0, 0));
    let mut seed = 123;
    let x = row(&mut seed);
    let selected = select(&p, 0, 0, &x).unwrap();
    let encoded = encode_packed(&p, 0, 0, &x).unwrap();
    let mean = p.groups()[0].mean.unwrap();
    for (i, &m) in mean.iter().enumerate() {
        let code = p.tables()[0].reconstruction(selected.states[i]);
        assert_eq!(
            selected.reconstruction[i].to_bits(),
            (code * selected.scale + m).to_bits()
        );
        assert_eq!(encoded.key_base(&p, 0, i).unwrap(), bfloat(code));
        assert_eq!(encoded.value_mean(&p, 0, i).unwrap(), m);
    }
    assert!(encoded.key_base(&p, 0, 256).is_err());
}

#[test]
fn kv_zero_and_negative_multipliers_follow_source_scale_rules() {
    let mut bytes = fixture(1, 1, 5);
    put32(&mut bytes, 100, 0.0_f32.to_bits());
    put32(&mut bytes, 104, (-1.0_f32).to_bits());
    let p = KvProfile::parse(&bytes).unwrap();
    let mut seed = 42;
    let selected = select(&p, 0, 0, &row(&mut seed)).unwrap();
    assert!(selected.scale.is_finite());
    // Zero RMS is replaced by one, then zero candidate scale by one.
    let to = u32_at(&bytes, 112);
    for i in 0..256 {
        put16(&mut bytes, to + 12 + 2 * i, 0);
    }
    let p = KvProfile::parse(&bytes).unwrap();
    assert_eq!(select(&p, 0, 0, &[0.0; 256]).unwrap().scale, 1.0);
}
