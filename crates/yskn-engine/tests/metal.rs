#[cfg(not(target_os = "macos"))]
#[test]
fn metal_matches_scalar_reference() {
    println!("skipped: no Metal device (not macOS)");
}

#[cfg(target_os = "macos")]
mod fixture;

#[cfg(target_os = "macos")]
mod macos {
    use super::fixture::{DATA_OFFSET, TempFile, assert_sums, change_header, random, synthetic};
    use serde_json::json;
    use yskn_engine::MetalEngine;
    use yskn_engine::{Engine, Kernel, MappedModel};
    use yskn_format::Container;
    use yskn_ref::compressed_linear::CompressedLinear;

    /// A missing device is reported on stdout and is the only way to skip.
    fn engine() -> Option<MetalEngine> {
        let engine = MetalEngine::new().unwrap();
        if engine.is_none() {
            println!("skipped: no Metal device");
        }
        engine
    }

    fn reference_sums(
        reference: &CompressedLinear,
        activation: &[i8],
        scales: &[i32],
        n: usize,
    ) -> Vec<i64> {
        let k = reference.cols();
        let groups = k / 512;
        (0..activation.len() / k)
            .flat_map(|m| {
                reference
                    .weighted_sums(
                        &activation[m * k..(m + 1) * k],
                        &scales[m * groups..(m + 1) * groups],
                        n,
                    )
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn metal_matches_scalar_reference_for_layouts_bits_shapes_and_batches() {
        let Some(mut engine) = engine() else { return };
        for bits in 4..=8 {
            for layout in ["row", "warp"] {
                let mut shapes = vec![
                    (17, 32, 512),
                    (65, 96, 1024),
                    (17, 32, 5120),
                    (13, 16, 512),
                    (100, 128, 1024),
                ];
                if layout == "row" {
                    shapes.push((9, 11, 512));
                }
                for (rows, physical_rows, cols) in shapes {
                    let bytes = synthetic(bits, layout, rows, physical_rows, cols, false);
                    let file = TempFile::new(&bytes);
                    // SAFETY: the private file is never changed while this model,
                    // its matrix, or the engine's no-copy buffer exists; the guard
                    // is declared first, so it is dropped (and the file removed) last.
                    let model = unsafe { MappedModel::open(&file.0) }.unwrap();
                    let matrix = model.matrix("linear").unwrap();
                    let mut reader = Container::from_bytes(bytes).unwrap();
                    let reference =
                        CompressedLinear::new(reader.load_tensor("linear").unwrap(), false)
                            .unwrap();
                    let groups = cols / 512;
                    let mut seed = 0x51ab_9001_7c3e_d24f ^ (bits as u64 * 131 + cols as u64);
                    // Extreme rows first: all +127, all -127, all -128, zero, alternating.
                    let activation: Vec<i8> = (0..40)
                        .flat_map(|m| {
                            let mut row: Vec<i8> = match m {
                                0 => vec![127; cols],
                                1 => vec![-127; cols],
                                2 => vec![-128; cols],
                                3 => vec![0; cols],
                                4 => (0..cols)
                                    .map(|c| if c % 2 == 0 { 127 } else { -128 })
                                    .collect(),
                                _ => (0..cols).map(|_| random(&mut seed) as i8).collect(),
                            };
                            if m > 4 {
                                row[0] = m as i8;
                            }
                            row
                        })
                        .collect();
                    let scales: Vec<i32> = (0..40)
                        .flat_map(|m| {
                            let mut row: Vec<i32> =
                                (0..groups).map(|_| random(&mut seed) as i32).collect();
                            match m {
                                0 => row.fill(i32::MAX),
                                2 => row.fill(i32::MIN),
                                _ => row[0] = (row[0] & !63) | m,
                            }
                            row
                        })
                        .collect();
                    let expected = reference_sums(&reference, &activation, &scales, physical_rows);
                    assert!(
                        expected[physical_rows..2 * physical_rows]
                            .iter()
                            .any(|&s| s != 0)
                    );
                    assert!(
                        expected[3 * physical_rows..4 * physical_rows]
                            .iter()
                            .all(|&s| s == 0)
                    );
                    for float_mac in [false, true] {
                        engine.set_float_mac(float_mac);
                        let mut jobs = Vec::new();
                        for m in [1, 2, 3, 8, 16, 17, 40] {
                            let context = format!(
                                "{layout} bits={bits} K={cols} N={rows}/{physical_rows} M={m} float={float_mac}"
                            );
                            let result = engine
                                .weighted_sums(
                                    &matrix,
                                    &activation[..m * cols],
                                    &scales[..m * groups],
                                )
                                .unwrap();
                            assert_sums(&result, &expected[..m * physical_rows], &context);
                            jobs.push(
                                engine
                                    .job(
                                        &matrix,
                                        &activation[..m * cols],
                                        &scales[..m * groups],
                                        physical_rows,
                                    )
                                    .unwrap(),
                            );
                            if m == 3 {
                                for n in [1, rows, physical_rows - 1] {
                                    let prefix: Vec<_> = (0..m)
                                        .flat_map(|i| {
                                            expected[i * physical_rows..i * physical_rows + n]
                                                .iter()
                                                .copied()
                                        })
                                        .collect();
                                    let result = engine
                                        .weighted_sums_prefix(
                                            &matrix,
                                            &activation[..m * cols],
                                            &scales[..m * groups],
                                            n,
                                        )
                                        .unwrap();
                                    assert_sums(&result, &prefix, &format!("{context} prefix={n}"));
                                }
                            }
                        }
                        // All M variants again, encoded into one command buffer.
                        engine.run(&mut jobs).unwrap();
                        for job in &jobs {
                            let m = job.output().len() / physical_rows;
                            assert_sums(
                                &job.output(),
                                &expected[..m * physical_rows],
                                &format!(
                                    "batched {layout} bits={bits} K={cols} M={m} float={float_mac}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    /// All weights ±127 with activations at -128/+127 and |scales| up to 2^31, at one
    /// group and at the maximum of 512 groups (1024 sequences): the largest partial
    /// sums either kernel can see. Both multiply-accumulate modes must equal the
    /// CPU kernel.
    #[test]
    fn extreme_values_at_maximum_sequence_counts_match_cpu_kernel() {
        let Some(mut engine) = engine() else { return };
        let mut cpu = Engine::new(Kernel::fastest(), 4).unwrap();
        let mut seed = 0x0e57_7a11_c0de_0127_u64;
        for bits in 4..=8 {
            for layout in ["row", "warp"] {
                for (cols, signs) in [(512, false), (512, true), (262_144, false), (262_144, true)]
                {
                    let mut bytes = synthetic(bits, layout, 32, 32, cols, false);
                    let header_length =
                        u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
                    let header: serde_json::Value =
                        serde_json::from_slice(&bytes[32..32 + header_length]).unwrap();
                    let table = header["tensors"][0]["lut_off"].as_u64().unwrap() as usize;
                    // Each entry ±1.0: every weight quantizes to ±127 (all +127 unless `signs`).
                    for pair in bytes[table..table + 16_384].as_chunks_mut::<2>().0 {
                        let negative = signs && !random(&mut seed).is_multiple_of(2);
                        *pair = yskn_format::f32_to_f16(if negative { -1.0 } else { 1.0 })
                            .to_le_bytes();
                    }
                    let file = TempFile::new(&bytes);
                    // SAFETY: the private file is never modified and outlives all mapping owners.
                    let model = unsafe { MappedModel::open(&file.0) }.unwrap();
                    let matrix = model.matrix("linear").unwrap();
                    let groups = cols / 512;
                    let m = 17;
                    let activation: Vec<i8> = (0..m)
                        .flat_map(|row| {
                            let mut seed = seed ^ row as u64;
                            (0..cols).map(move |c| match row {
                                0 | 1 => -128,
                                2 => 127,
                                3 => {
                                    if c % 2 == 0 {
                                        -128
                                    } else {
                                        127
                                    }
                                }
                                4..=9 => {
                                    if random(&mut seed).is_multiple_of(2) {
                                        -128
                                    } else {
                                        127
                                    }
                                }
                                _ => random(&mut seed) as i8,
                            })
                        })
                        .collect();
                    let scales: Vec<i32> = (0..m)
                        .flat_map(|row| {
                            let mut seed = seed ^ (row as u64) << 32;
                            (0..groups).map(move |_| match row {
                                0 | 2 => i32::MAX,
                                1 | 3 => i32::MIN,
                                _ => random(&mut seed) as i32,
                            })
                        })
                        .collect();
                    seed = seed.wrapping_add(1);
                    let expected = cpu.weighted_sums(&matrix, &activation, &scales).unwrap();
                    // All +127: row 1 reaches the bound 512 * 127 * 128 * 2^31 per group
                    // (99.2 % of the i64 limit at 512 groups).
                    let largest = expected.iter().map(|s| s.unsigned_abs()).max().unwrap();
                    assert!(signs || largest == (groups as u64 * 512 * 127 * 128) << 31);
                    for float_mac in [false, true] {
                        engine.set_float_mac(float_mac);
                        for rows in [1, 5, 16, 17] {
                            let result = engine
                                .weighted_sums(
                                    &matrix,
                                    &activation[..rows * cols],
                                    &scales[..rows * groups],
                                )
                                .unwrap();
                            assert_sums(
                                &result,
                                &expected[..rows * 32],
                                &format!(
                                    "{layout} bits={bits} K={cols} signs={signs} M={rows} float={float_mac}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Rows shared between the GPU and the CPU kernel (several matrices per call,
    /// padded and partial tiles) equal the GPU-only and CPU-only sums.
    #[test]
    fn cpu_and_gpu_row_split_matches_either_device_alone() {
        let Some(mut engine) = engine() else { return };
        let mut cpu = Engine::new(Kernel::fastest(), 4).unwrap();
        let shapes = [
            (300, 320, "warp", 6),
            (70, 75, "row", 5),
            (17, 32, "row", 8),
        ];
        let files: Vec<_> = shapes
            .iter()
            .map(|&(rows, physical, layout, bits)| {
                TempFile::new(&synthetic(bits, layout, rows, physical, 5120, false))
            })
            .collect();
        // SAFETY: the private files are never modified and outlive all mapping owners.
        let models: Vec<_> = files
            .iter()
            .map(|file| unsafe { MappedModel::open(&file.0) }.unwrap())
            .collect();
        let matrices: Vec<_> = models.iter().map(|m| m.matrix("linear").unwrap()).collect();
        let refs: Vec<_> = matrices.iter().collect();
        let mut seed = 0x5917_c0de_0000_0001;
        for m in [1, 3, 16, 20] {
            let activation: Vec<i8> = (0..m * 5120).map(|_| random(&mut seed) as i8).collect();
            let scales: Vec<i32> = (0..m * 10).map(|_| random(&mut seed) as i32).collect();
            let expected: Vec<_> = matrices
                .iter()
                .map(|matrix| cpu.weighted_sums(matrix, &activation, &scales).unwrap())
                .collect();
            for share in [None, Some(0.0), Some(0.3), Some(0.9)] {
                engine.set_cpu_share(share).unwrap();
                let split = engine
                    .weighted_sums_split(&mut cpu, &refs, &activation, &scales)
                    .unwrap();
                let gpu = engine
                    .weighted_sums_shared(&refs, &activation, &scales)
                    .unwrap();
                for (i, expected) in expected.iter().enumerate() {
                    let context = format!("matrix={i} M={m} share={share:?}");
                    assert_sums(&split[i], expected, &format!("split {context}"));
                    assert_sums(&gpu[i], expected, &format!("gpu {context}"));
                }
            }
        }
        assert!(engine.set_cpu_share(Some(0.95)).is_err());
        assert!(engine.set_cpu_share(Some(-0.1)).is_err());
        engine.set_cpu_share(Some(0.5)).unwrap();
        assert_eq!(engine.cpu_share(1), 0.5);
        engine.set_cpu_share(Some(0.0)).unwrap();
        assert_eq!(engine.cpu_share(16), 0.0);
        engine.set_cpu_share(None).unwrap();
        let calibrated: Vec<_> = [1, 2, 3, 4, 8, 12, 16, 40]
            .map(|m| engine.cpu_share(m))
            .to_vec();
        let table = [0.08, 0.08, 0.085, 0.09, 0.12, 0.135, 0.15, 0.15];
        assert!(
            calibrated
                .iter()
                .zip(table)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn unaligned_streams_and_split_counts_match_scalar_reference() {
        let Some(mut engine) = engine() else { return };
        for bits in 4..=8 {
            for layout in ["row", "warp"] {
                for shift in [1, 4, 8] {
                    let mut bytes = synthetic(bits, layout, 13, 64, 5120, false);
                    bytes.splice(DATA_OFFSET..DATA_OFFSET, vec![0; shift]);
                    let file_bytes = bytes.len() as u64;
                    bytes[16..24].copy_from_slice(&((DATA_OFFSET + shift) as u64).to_le_bytes());
                    bytes[24..32].copy_from_slice(&file_bytes.to_le_bytes());
                    let bytes = change_header(&bytes, |header| {
                        for field in ["stream_off", "lut_off", "scale_off"] {
                            let offset = header["tensors"][0][field].as_u64().unwrap();
                            header["tensors"][0][field] = json!(offset + shift as u64);
                        }
                    });
                    let file = TempFile::new(&bytes);
                    // SAFETY: as above, the private file outlives every mapping owner.
                    let model = unsafe { MappedModel::open(&file.0) }.unwrap();
                    let matrix = model.matrix("linear").unwrap();
                    assert_eq!(matrix.descriptor().stream_offset % 16, shift as u64 % 16);
                    let mut reader = Container::from_bytes(bytes).unwrap();
                    let reference =
                        CompressedLinear::new(reader.load_tensor("linear").unwrap(), false)
                            .unwrap();
                    let mut seed = 0x77ed_31a0_9bc1_521d;
                    let activation: Vec<_> =
                        (0..5 * 5120).map(|_| random(&mut seed) as i8).collect();
                    let scales: Vec<_> = (0..50).map(|_| random(&mut seed) as i32).collect();
                    let expected = reference_sums(&reference, &activation, &scales, 64);
                    for splits in [1, 3, 8, 20, 32] {
                        engine.set_splits(splits).unwrap();
                        let result = match engine.weighted_sums(&matrix, &activation, &scales) {
                            Ok(result) => result,
                            // Some devices cap a threadgroup below 32 x splits (CI's
                            // paravirtual GPU allows 448 threads); the default of 4 must work.
                            Err(error)
                                if splits > 8
                                    && error.to_string().contains("threads per threadgroup") =>
                            {
                                eprintln!("skipped: splits={splits}: {error}");
                                continue;
                            }
                            Err(error) => panic!("splits={splits}: {error}"),
                        };
                        assert_sums(
                            &result,
                            &expected,
                            &format!("{layout} bits={bits} shift={shift} splits={splits}"),
                        );
                    }
                    engine.set_splits(8).unwrap();
                }
            }
        }
    }

    #[test]
    fn tensors_sharing_codes_keep_their_own_tables() {
        let Some(mut engine) = engine() else { return };
        // Tensor "linear" (A) keeps the fixture's codes; B points at the same codes
        // but has its own lookup table appended at the end of the file.
        let mut bytes = synthetic(4, "row", 32, 32, 512, false);
        let header_length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[32..32 + header_length]).unwrap();
        let table_a = header["tensors"][0]["lut_off"].as_u64().unwrap() as usize;
        let table_b = bytes.len();
        let one = yskn_format::f32_to_f16(1.0).to_le_bytes();
        let minus_one = yskn_format::f32_to_f16(-1.0).to_le_bytes();
        for pair in bytes[table_a..table_a + 16_384].as_chunks_mut::<2>().0 {
            *pair = one;
        }
        for _ in 0..8192 {
            bytes.extend_from_slice(&minus_one);
        }
        let file_bytes = bytes.len() as u64;
        bytes[24..32].copy_from_slice(&file_bytes.to_le_bytes());
        let bytes = change_header(&bytes, |header| {
            let mut b = header["tensors"][0].clone();
            b["name"] = json!("b");
            b["lut_off"] = json!(table_b);
            header["tensors"].as_array_mut().unwrap().push(b);
        });
        let file = TempFile::new(&bytes);
        // SAFETY: the private file is never modified and outlives all mapping owners.
        let model = unsafe { MappedModel::open(&file.0) }.unwrap();
        let a = model.matrix("linear").unwrap();
        let b = model.matrix("b").unwrap();
        assert_eq!(a.code_range(), b.code_range());
        // Every weight quantizes to ±127, so each row is ±512 * 127 whatever the codes.
        let a_sums = engine.weighted_sums(&a, &[1; 512], &[1]).unwrap();
        let b_sums = engine.weighted_sums(&b, &[1; 512], &[1]).unwrap();
        assert_eq!(a_sums, vec![65_024; 32]);
        assert_eq!(b_sums, vec![-65_024; 32]);
    }

    #[test]
    fn invalid_shapes_are_rejected_before_dispatch() {
        let Some(mut engine) = engine() else { return };
        let bytes = synthetic(6, "warp", 65, 96, 1024, false);
        let file = TempFile::new(&bytes);
        // SAFETY: the private file is never modified and outlives all mapping owners.
        let model = unsafe { MappedModel::open(&file.0) }.unwrap();
        let matrix = model.matrix("linear").unwrap();
        for activation in [vec![], vec![1; 1023], vec![1; 1025]] {
            assert!(engine.weighted_sums(&matrix, &activation, &[1, 1]).is_err());
        }
        for scales in [vec![], vec![1], vec![1; 3]] {
            assert!(engine.weighted_sums(&matrix, &[1; 1024], &scales).is_err());
        }
        for n in [0, 97, usize::MAX] {
            assert!(
                engine
                    .weighted_sums_prefix(&matrix, &[1; 1024], &[1, 1], n)
                    .is_err()
            );
        }
        assert!(engine.set_splits(0).is_err());
        assert!(engine.set_splits(33).is_err());
        assert_eq!(
            engine.weighted_sums(&matrix, &[0; 1024], &[1, -1]).unwrap(),
            vec![0; 96]
        );
        assert!(engine.round_trip().unwrap() > std::time::Duration::ZERO);
    }
}
