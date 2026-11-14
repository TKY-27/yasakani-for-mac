mod support;
use std::io::Cursor;
use support::draft as fixture;
use yskn_format::Container;
use yskn_ref::draft::{ContextFeatures, Draft, DraftOptions, model_paths};
use yskn_ref::draft_head::{
    Candidate, DraftChoice, HeadLayout, ReducedHead, choose, shortlist, top16,
};
fn draft() -> Draft {
    Draft::from_container(
        &mut Container::from_bytes(fixture::container(23)).unwrap(),
        ReducedHead::read(&mut Cursor::new(fixture::shead(37))).unwrap(),
        fixture::options(),
    )
    .unwrap()
}
fn digest(proposal: &yskn_ref::draft::DraftProposal) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for bits in proposal
        .tokens
        .iter()
        .copied()
        .chain(
            proposal
                .candidates
                .iter()
                .flatten()
                .flat_map(|c| [c.token, c.score.to_bits()]),
        )
        .chain(
            proposal
                .corrected
                .iter()
                .flatten()
                .flat_map(|c| [c.token, c.score.to_bits()]),
        )
    {
        for b in bits.to_le_bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    hash
}
#[test]
fn draft_ring_batching_overwrite_snapshot_reset_and_failed_row_are_exact() {
    let mut a = draft();
    let initial = a.snapshot();
    let rows: Vec<_> = (0..11).flat_map(fixture::features).collect();
    a.push_context_rows(0, &rows).unwrap();
    let snapshot = a.snapshot();
    assert_eq!(snapshot.position(), 11);
    assert_ne!(initial.keys(), snapshot.keys());
    assert_ne!(initial.values(), snapshot.values());
    let mut b = draft();
    for row in 0..11 {
        b.push_context(
            row,
            ContextFeatures {
                values: &fixture::features(row),
            },
        )
        .unwrap();
    }
    assert_eq!(snapshot, b.snapshot());
    // At position 8 exactly slot 0 is overwritten, all other head-major slots survive.
    b.reset();
    b.push_context_rows(0, &rows[..8 * 2560]).unwrap();
    let old = b.snapshot();
    b.push_context(
        8,
        ContextFeatures {
            values: &fixture::features(8),
        },
    )
    .unwrap();
    let new = b.snapshot();
    for layer in 0..2 {
        for head in 0..4 {
            let base = head * 8 * 128;
            assert_ne!(
                &old.keys()[layer][base..base + 128],
                &new.keys()[layer][base..base + 128]
            );
            assert_eq!(
                &old.keys()[layer][base + 128..base + 8 * 128],
                &new.keys()[layer][base + 128..base + 8 * 128]
            );
            assert_eq!(
                &old.values()[layer][base + 128..base + 8 * 128],
                &new.values()[layer][base + 128..base + 8 * 128]
            );
        }
    }
    b.restore(&snapshot).unwrap();
    assert_eq!(snapshot, b.snapshot());
    let expected = a
        .propose(3, fixture::embedding, DraftChoice::Greedy)
        .unwrap();
    assert_eq!(
        expected,
        b.propose(3, fixture::embedding, DraftChoice::Greedy)
            .unwrap()
    );
    assert!(
        b.push_context(
            10,
            ContextFeatures {
                values: &fixture::features(0)
            }
        )
        .is_err()
    );
    let mut bad = fixture::features(0);
    bad[0] = f32::NAN;
    assert!(
        b.push_context(11, ContextFeatures { values: &bad })
            .is_err()
    );
    assert_eq!(snapshot, b.snapshot());
    b.reset();
    assert_eq!(initial, b.snapshot());
}
#[test]
fn draft_block_future_perturbation_changes_earlier_proposals_without_mutating_ring() {
    let d = draft();
    let state = d.snapshot();
    let mut rows: Vec<_> = (0..8).map(|i| fixture::embedding(i).unwrap()).collect();
    let original = d.block_hidden(&rows).unwrap();
    rows[7] = fixture::embedding(19).unwrap();
    let changed = d.block_hidden(&rows).unwrap();
    // Returned row 0 is input row 1, which sees input row 7 through noncausal attention.
    assert_ne!(&original[..512], &changed[..512]);
    assert_eq!(state, d.snapshot());
}
#[test]
fn draft_proposals_repeat_bits_and_freeze_synthetic_scores() {
    let mut d = draft();
    d.push_context_rows(0, &fixture::features(0)).unwrap();
    for (choice, expected) in [
        (DraftChoice::Greedy, 3266815268263694244_u64),
        (DraftChoice::sampled(123), 6053741631618019484_u64),
    ] {
        let a = d.propose(3, fixture::embedding, choice).unwrap();
        let b = d.propose(3, fixture::embedding, choice).unwrap();
        assert_eq!(digest(&a), digest(&b));
        assert_eq!(a.tokens.len(), 7);
        assert_eq!(
            a.tokens,
            if choice == DraftChoice::Greedy {
                vec![20; 7]
            } else {
                vec![31, 31, 3, 3, 3, 3, 31]
            }
        );
        assert!(a.candidates.iter().all(|c| c.len() == 16));
        assert!(a.corrected.iter().zip(&a.candidates).any(|(a, b)| {
            a.iter()
                .zip(b)
                .any(|(a, b)| a.score.to_bits() != b.score.to_bits())
        }));
        assert_eq!(
            digest(&a),
            expected,
            "choice={choice:?} tokens={:?}",
            a.tokens
        );
    }
}
#[test]
fn draft_shortlist_cutoff_is_prefix_based_with_ascending_id_ties() {
    let mut scores = vec![1.0; 32];
    scores[31] = 2.0;
    // 1+small mantissa increments share the cutoff prefix; exact sort would favor high IDs.
    for (i, s) in scores[..31].iter_mut().enumerate() {
        *s = f32::from_bits(1.0_f32.to_bits() + i as u32);
    }
    let mut expected = vec![31];
    expected.extend(0..15);
    assert_eq!(shortlist(&scores, 16).unwrap(), expected);
    scores[..16].fill(-3.0);
    scores[16..24].fill(-2.0);
    scores[24..].fill(-1.0);
    assert_eq!(
        shortlist(&scores, 16).unwrap(),
        (24..32).chain(16..24).collect::<Vec<_>>()
    );
    assert_eq!(
        shortlist(&[0.0; 32], 24).unwrap(),
        (0..24).collect::<Vec<_>>()
    );
    assert!(shortlist(&[0.0; 32], 17).is_err());
    assert!(shortlist(&[f32::NAN; 32], 16).is_err());
}
#[test]
fn draft_top16_and_choice_have_distinct_tie_rules_and_sampled_top_p() {
    let candidates: Vec<_> = (0..32)
        .rev()
        .map(|i| Candidate {
            token: i,
            score: (i % 2) as f32,
        })
        .collect();
    let result = top16(candidates).unwrap();
    assert_eq!(
        result.iter().map(|c| c.token).collect::<Vec<_>>(),
        (0..16).map(|i| 2 * i + 1).collect::<Vec<_>>()
    );
    let ties: Vec<_> = (0..16)
        .rev()
        .map(|i| Candidate {
            token: i,
            score: 0.0,
        })
        .collect();
    assert_eq!(choose(&ties, DraftChoice::Greedy, 8).unwrap(), 15);
    // A dominant candidate makes top-p exclude every other token regardless of seed.
    let mut dominant = ties.clone();
    dominant[7].score = 100.0;
    for seed in 0..8 {
        assert_eq!(
            choose(&dominant, DraftChoice::sampled(seed), 8).unwrap(),
            dominant[7].token
        );
    }
    // Token-keyed sampling must not depend on candidate-list order (ties rank by token).
    let a = choose(&ties, DraftChoice::sampled(314), 8).unwrap();
    let mut reversed = ties.clone();
    reversed.reverse();
    assert_eq!(choose(&reversed, DraftChoice::sampled(314), 8).unwrap(), a);
    assert!(
        choose(
            &ties,
            DraftChoice::Sampled {
                seed: 0,
                temperature_thousandths: 0
            },
            0
        )
        .is_err()
    );
}
fn mutate_header(bytes: &mut Vec<u8>, f: impl FnOnce(&mut serde_json::Value)) {
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let old_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let mut original: serde_json::Value = serde_json::from_slice(&bytes[32..32 + n]).unwrap();
    f(&mut original);
    let mut offset = old_offset;
    loop {
        let mut header = original.clone();
        let shift = (offset - old_offset) as u64;
        for t in header["tensors"].as_array_mut().unwrap() {
            for field in ["stream_off", "lut_off", "scale_off"] {
                t[field] = serde_json::json!(t[field].as_u64().unwrap().saturating_add(shift));
            }
        }
        for v in header["vectors"].as_array_mut().unwrap() {
            v["off"] = serde_json::json!(v["off"].as_u64().unwrap().saturating_add(shift));
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let required = (32 + encoded.len() + 255) & !255;
        if required > offset {
            offset = required;
            continue;
        }
        let mut result = vec![0; offset];
        result[..32].copy_from_slice(&bytes[..32]);
        result[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
        result[16..24].copy_from_slice(&(offset as u64).to_le_bytes());
        result[24..32].copy_from_slice(&((bytes.len() + offset - old_offset) as u64).to_le_bytes());
        result[32..32 + encoded.len()].copy_from_slice(&encoded);
        result.extend_from_slice(&bytes[old_offset..]);
        *bytes = result;
        return;
    }
}

#[test]
fn draft_shead_malformed_headers_lengths_and_nonfinite_arrays_are_errors() {
    let good = fixture::shead(1);
    let layout = HeadLayout::read(&mut Cursor::new(&good)).unwrap();
    assert_eq!(layout.reserved, 12345);
    for length in [0, 7, 8, 23, 24, good.len() - 1]
        .into_iter()
        .chain(layout.arrays.iter().map(|a| a.end as usize - 1))
    {
        assert!(
            ReducedHead::read(&mut Cursor::new(&good[..length])).is_err(),
            "length {length}"
        );
    }
    for offset in [0, 8, 12, 16] {
        let mut b = good.clone();
        b[offset..offset + 4].copy_from_slice(&(-1_i32).to_le_bytes());
        assert!(ReducedHead::read(&mut Cursor::new(b)).is_err());
    }
    let mut b = good.clone();
    for offset in [8, 12, 16] {
        b[offset..offset + 4].copy_from_slice(&i32::MAX.to_le_bytes());
    }
    assert!(ReducedHead::read(&mut Cursor::new(b)).is_err());
    let mut b = good.clone();
    b.push(0);
    assert!(ReducedHead::read(&mut Cursor::new(b)).is_err());
    for array in [0, 2, 4] {
        let mut b = good.clone();
        let i = layout.arrays[array].start as usize;
        b[i..i + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(ReducedHead::read(&mut Cursor::new(b)).is_err());
    }
}
#[test]
fn draft_container_malformed_descriptors_shapes_options_and_weights_are_errors() {
    let good = fixture::container(23);
    for len in [0, 3, 31, good.len() - 1] {
        assert!(Container::from_bytes(good[..len].to_vec()).is_err());
    }
    let mutations: Vec<fn(&mut serde_json::Value)> = vec![
        |h| h["model"]["arch"] = serde_json::json!("wrong"),
        |h| h["model"]["hidden"] = serde_json::json!(0),
        |h| h["model"]["heads"] = serde_json::json!(3),
        |h| h["model"]["window"] = serde_json::json!(4096),
        |h| h["model"]["block"] = serde_json::json!(16),
        |h| h["model"]["rank"] = serde_json::json!(1),
        |h| h["model"]["ffn"] = serde_json::json!(4096),
        |h| h["model"]["eps"] = serde_json::json!(-1),
        |h| h["model"]["layers"] = serde_json::json!(3),
        |h| h["vectors"][0]["n"] = serde_json::json!(511),
        |h| h["vectors"][0]["dtype"] = serde_json::json!("f16"),
        |h| h["tensors"][1]["stream_off"] = serde_json::json!(u64::MAX),
        |h| h["tensors"][1]["cols"] = serde_json::json!(1),
        |h| h["tensors"][1]["name"] = serde_json::json!("missing"),
    ];
    for (i, m) in mutations.into_iter().enumerate() {
        let mut b = fixture::container(23);
        mutate_header(&mut b, m);
        let result = Container::from_bytes(b)
            .map_err(yskn_ref::Error::from)
            .and_then(|mut c| {
                Draft::from_container(
                    &mut c,
                    ReducedHead::read(&mut Cursor::new(fixture::shead(37)))?,
                    fixture::options(),
                )
            });
        assert!(result.is_err(), "mutation {i}");
    }
    let mut b = fixture::container(23);
    let c = Container::from_bytes(b.clone()).unwrap();
    let i = c.header().vectors[0].offset as usize;
    b[i..i + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    assert!(
        Draft::from_container(
            &mut Container::from_bytes(b).unwrap(),
            ReducedHead::read(&mut Cursor::new(fixture::shead(37))).unwrap(),
            fixture::options()
        )
        .is_err()
    );
    for options in [
        DraftOptions {
            mask_token: 32,
            ..fixture::options()
        },
        DraftOptions {
            capture_layers: vec![1],
            ..fixture::options()
        },
        DraftOptions {
            shortlist: 40,
            ..fixture::options()
        },
    ] {
        assert!(
            Draft::from_container(
                &mut Container::from_bytes(fixture::container(23)).unwrap(),
                ReducedHead::read(&mut Cursor::new(fixture::shead(37))).unwrap(),
                options
            )
            .is_err()
        );
    }
}
#[test]
fn draft_folder_loading_rejects_traversal_absolute_and_symlink_escapes() {
    let root = std::env::temp_dir().join(format!("yskn-draft-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("bdraft2_k8.yskn"), fixture::container(23)).unwrap();
    std::fs::write(root.join("bdraft2_k8.shead"), fixture::shead(37)).unwrap();
    assert!(Draft::load_with_options(&root, fixture::options()).is_ok());
    for name in ["../outside", "/tmp/outside", ""] {
        std::fs::write(
            root.join("yasakani.json"),
            serde_json::to_vec(&serde_json::json!({"files":{"draft":name}})).unwrap(),
        )
        .unwrap();
        assert!(model_paths(&root).is_err());
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(std::env::temp_dir(), root.join("escape")).unwrap();
        std::fs::write(
            root.join("yasakani.json"),
            r#"{"files":{"draft":"escape"}}"#,
        )
        .unwrap();
        assert!(model_paths(&root).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn draft_real_files_load_and_propose_deterministically() {
    let Some(folder) = std::env::var_os("YSKN_MODEL_DIR") else {
        println!("skipped: YSKN_MODEL_DIR is not set");
        return;
    };
    let start = std::time::Instant::now();
    let mut d = Draft::load(folder).unwrap();
    println!(
        "draft load {:?}; dimensions={:?}",
        start.elapsed(),
        d.dimensions()
    );
    let width = d.dimensions().hidden;
    let features: Vec<_> = (0..d.dimensions().features)
        .map(|i| yskn_ref::portable_math::bfloat((i % 31) as f32 / 32.0 - 0.5))
        .collect();
    d.push_context(0, ContextFeatures { values: &features })
        .unwrap();
    let embedding = |token: u32| {
        Ok((0..width)
            .map(|i| {
                yskn_ref::portable_math::bfloat(((i * 7 + token as usize) % 29) as f32 / 32.0 - 0.5)
            })
            .collect())
    };
    let start = std::time::Instant::now();
    let a = d.propose(3, embedding, DraftChoice::Greedy).unwrap();
    println!(
        "draft real propose_seconds={:.6} tokens={:?} digest={:016x}",
        start.elapsed().as_secs_f64(),
        a.tokens,
        digest(&a)
    );
    let start = std::time::Instant::now();
    let b = d.propose(3, embedding, DraftChoice::Greedy).unwrap();
    println!(
        "draft repeated propose_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    assert_eq!(digest(&a), digest(&b));
    assert_eq!(a, b);
}

#[test]
fn draft_reduced_head_sparse_weights_have_independent_known_scores() {
    let mut bytes = fixture::shead(7);
    let layout = HeadLayout::read(&mut Cursor::new(&bytes)).unwrap();
    for region in &layout.arrays {
        bytes[region.start as usize..region.end as usize].fill(0);
    }
    // Hidden channel 511, projected to rank component 511: catches row/stride mistakes.
    let offset = layout.arrays[0].start as usize + 4 * (511 * 512 + 511);
    bytes[offset..offset + 4].copy_from_slice(&2.0_f32.to_le_bytes());
    for v in 0..32 {
        bytes[layout.arrays[1].start as usize + v * 512 + 511] = (v as i8 - 16) as u8;
        bytes[layout.arrays[3].start as usize + v * 512 + 511] = (v as i8 - 24) as u8;
        for region in [2, 4] {
            let offset = layout.arrays[region].start as usize + 4 * v;
            bytes[offset..offset + 4].copy_from_slice(&1.0_f32.to_le_bytes());
        }
    }
    let head = ReducedHead::read(&mut Cursor::new(bytes)).unwrap();
    let mut hidden = vec![0.0; 512];
    hidden[511] = 0.5;
    let candidates = head.candidates(&hidden, 16).unwrap();
    let expected: Vec<_> = (16..32)
        .rev()
        .map(|v| Candidate {
            token: v,
            score: (v as f32 - 24.0) * 0.5,
        })
        .collect();
    assert_eq!(candidates, expected);
}

#[test]
fn draft_uses_fc_sign_prefix_and_ignores_per_matrix_output_rotation() {
    let mut bytes = fixture::container(23);
    mutate_header(&mut bytes, |h| {
        h["tensors"][1]["rot_in"]["sign_seed"] = serde_json::json!(7);
        h["tensors"][1]["rot_out"] = serde_json::json!({"kind":"H512","m":1,"sign_seed":91,"sign_offset":512,"rule":"mulberry32_7919_13","qm_off":0});
    });
    let mut a = draft();
    let mut b = Draft::from_container(
        &mut Container::from_bytes(bytes).unwrap(),
        ReducedHead::read(&mut Cursor::new(fixture::shead(37))).unwrap(),
        fixture::options(),
    )
    .unwrap();
    for d in [&mut a, &mut b] {
        d.push_context_rows(0, &fixture::features(0)).unwrap();
    }
    assert_eq!(a.snapshot(), b.snapshot());
    assert_eq!(
        a.propose(3, fixture::embedding, DraftChoice::Greedy)
            .unwrap(),
        b.propose(3, fixture::embedding, DraftChoice::Greedy)
            .unwrap()
    );
}
