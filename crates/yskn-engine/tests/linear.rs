mod fixture;

use fixture::{DATA_OFFSET, TempFile, assert_sums, change_header, random, synthetic};
use serde_json::json;
use yskn_engine::{Engine, Kernel, MappedModel};
use yskn_format::Container;
use yskn_ref::compressed_linear::CompressedLinear;

#[test]
fn every_kernel_matches_scalar_for_layout_bits_batches_padding_and_signed_scales() {
    for bits in 4..=8 {
        for layout in ["row", "warp"] {
            let mut shapes = vec![(17, 32, 512), (65, 96, 1024), (17, 32, 5120), (13, 16, 512)];
            if layout == "row" {
                shapes.push((9, 11, 512));
            }
            for (rows, physical_rows, cols) in shapes {
                let bytes = synthetic(bits, layout, rows, physical_rows, cols, false);
                // This guard is declared first, so every mapping and worker job
                // is dropped before its private file is removed, even on panic.
                let file = TempFile::new(&bytes);
                // SAFETY: the private file is complete and never changed while
                // this model, its matrix, or an engine job can retain a mapping.
                let model = unsafe { MappedModel::open(&file.0) }.unwrap();
                let matrix = model.matrix("linear").unwrap();
                let mut reader = Container::from_bytes(bytes).unwrap();
                let reference =
                    CompressedLinear::new(reader.load_tensor("linear").unwrap(), false).unwrap();
                let mut engines: Vec<_> = Kernel::available_variants()
                    .into_iter()
                    .flat_map(|kernel| {
                        [1, 4]
                            .into_iter()
                            .map(move |threads| Engine::new(kernel, threads).unwrap())
                    })
                    .collect();
                let groups = cols / 512;
                let mut seed = 0xe79a_170f_88ab_203d;
                let patterns: Vec<Vec<i8>> = (0..40)
                    .map(|input| match input {
                        1 => vec![127; cols],
                        2 => vec![-127; cols],
                        3 => vec![0; cols],
                        _ => {
                            let mut row: Vec<_> =
                                (0..cols).map(|_| random(&mut seed) as i8).collect();
                            row[0] = input as i8;
                            row
                        }
                    })
                    .collect();
                let scales: Vec<Vec<i32>> = (0..40)
                    .map(|input| {
                        let mut row: Vec<_> =
                            (0..groups).map(|_| random(&mut seed) as i32).collect();
                        // Distinct low bits guarantee every row's scales differ,
                        // including across the kernel's 16-row chunk boundary.
                        row[0] = (row[0] & !63) | input;
                        row
                    })
                    .collect();
                let expected: Vec<_> = patterns
                    .iter()
                    .zip(&scales)
                    .map(|(activation, scales)| {
                        reference
                            .weighted_sums(activation, scales, physical_rows)
                            .unwrap()
                    })
                    .collect();
                assert!(expected[0].iter().any(|&sum| sum != 0));
                assert!(expected[3].iter().all(|&sum| sum == 0));
                let activation: Vec<_> = patterns.iter().flatten().copied().collect();
                let scale: Vec<_> = scales.iter().flatten().copied().collect();
                for m in [1, 2, 3, 8, 16, 17, 40] {
                    let oracle: Vec<_> = expected[..m].iter().flatten().copied().collect();
                    for engine in &mut engines {
                        let context = format!(
                            "{} threads={} {layout} bits={bits} K={cols} N={rows}/{physical_rows} M={m}",
                            engine.kernel().name(),
                            engine.threads()
                        );
                        let result = engine
                            .weighted_sums(&matrix, &activation[..m * cols], &scale[..m * groups])
                            .unwrap();
                        assert_sums(&result, &oracle, &context);
                        if m == 3 {
                            for n in [1, rows, physical_rows - 1] {
                                let prefix: Vec<_> = (0..m)
                                    .flat_map(|input| expected[input][..n].iter().copied())
                                    .collect();
                                let result = engine
                                    .weighted_sums_prefix(
                                        &matrix,
                                        &activation[..m * cols],
                                        &scale[..m * groups],
                                        n,
                                    )
                                    .unwrap();
                                assert_sums(&result, &prefix, &format!("{context} prefix={n}"));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn unaligned_mapped_streams_match_scalar_in_both_layouts() {
    for bits in 4..=8 {
        for layout in ["row", "warp"] {
            let mut bytes = synthetic(bits, layout, 13, 16, 512, false);
            bytes.insert(DATA_OFFSET, 0);
            let file_bytes = bytes.len() as u64;
            bytes[16..24].copy_from_slice(&((DATA_OFFSET + 1) as u64).to_le_bytes());
            bytes[24..32].copy_from_slice(&file_bytes.to_le_bytes());
            let bytes = change_header(&bytes, |header| {
                for field in ["stream_off", "lut_off", "scale_off"] {
                    let offset = header["tensors"][0][field].as_u64().unwrap();
                    header["tensors"][0][field] = json!(offset + 1);
                }
            });
            let file = TempFile::new(&bytes);
            // SAFETY: this uniquely created private file is immutable and its
            // guard is dropped after every mapping and persistent worker.
            let model = unsafe { MappedModel::open(&file.0) }.unwrap();
            let matrix = model.matrix("linear").unwrap();
            assert_eq!(matrix.descriptor().stream_offset % 4, 1);
            let mut reader = Container::from_bytes(bytes).unwrap();
            let reference =
                CompressedLinear::new(reader.load_tensor("linear").unwrap(), false).unwrap();
            let mut seed = 0x77ed_31a0_9bc1_521d;
            let activation: Vec<_> = (0..17 * 512).map(|_| random(&mut seed) as i8).collect();
            let scales: Vec<_> = (0..17).map(|_| random(&mut seed) as i32).collect();
            let expected: Vec<_> = activation
                .as_chunks::<512>()
                .0
                .iter()
                .zip(&scales)
                .flat_map(|(row, &scale)| reference.weighted_sums(row, &[scale], 16).unwrap())
                .collect();
            for kernel in Kernel::available_variants() {
                for threads in [1, 4] {
                    let mut engine = Engine::new(kernel, threads).unwrap();
                    let result = engine.weighted_sums(&matrix, &activation, &scales).unwrap();
                    assert_sums(
                        &result,
                        &expected,
                        &format!(
                            "unaligned {layout} bits={bits} {} threads={threads}",
                            kernel.name()
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn mapped_payload_and_derived_parameters_match_the_existing_reader() {
    for layout in ["row", "warp"] {
        let bytes = synthetic(6, layout, 512, 512, 512, true);
        let file = TempFile::new(&bytes);
        // SAFETY: this private fixture is immutable, and the guard outlives
        // both model and matrix mappings, including the cloned matrix below.
        let model = unsafe { MappedModel::open(&file.0) }.unwrap();
        let matrix = model.matrix("linear").unwrap();
        let mut reader = Container::from_bytes(bytes.clone()).unwrap();
        let tensor = reader.load_tensor("linear").unwrap();
        let descriptor = tensor.descriptor();
        assert_eq!(model.header().file_bytes, reader.header().file_bytes);
        assert_eq!(matrix.descriptor().layout, descriptor.layout);
        assert_eq!(matrix.input_signs(), tensor.input_signs());
        assert_eq!(matrix.output_signs(), tensor.output_signs());
        let start = descriptor.stream_offset as usize;
        let end = start + descriptor.stream_bytes as usize;
        assert_eq!(matrix.code_words(), &bytes[start..end]);
        assert_eq!(
            matrix.code_words().as_ptr(),
            model.matrix("linear").unwrap().code_words().as_ptr(),
            "separate matrix handles must share the mapped code stream"
        );
        // Compute the specified quantization independently of the shared
        // production helper, using the reader's decoded FP16 values.
        let maximum = tensor
            .table()
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let table_scale = maximum / 127.0;
        let table: Vec<_> = tensor
            .table()
            .iter()
            .map(|&v| (v / table_scale).round_ties_even().clamp(-127.0, 127.0) as i8)
            .collect();
        let scales: Vec<_> = tensor.scales().iter().map(|&v| v * table_scale).collect();
        assert_eq!(&matrix.weight_table8()[..], table);
        assert_eq!(matrix.effective_row_scale(), scales);
        assert!(model.matrix("missing").is_err());
        let retained = matrix.clone();
        drop(matrix);
        drop(model);
        assert_eq!(retained.code_words(), &bytes[start..end]);
        assert_eq!(retained.input_signs(), tensor.input_signs());
    }
}

#[test]
fn mapped_open_rejects_truncation_corrupt_headers_and_out_of_bounds_payloads() {
    let valid = synthetic(8, "warp", 17, 32, 512, false);
    assert!(Container::from_bytes(valid.clone()).is_ok());
    let mut malformed: Vec<_> = [0, 4, 31, 32, DATA_OFFSET, valid.len() - 1]
        .into_iter()
        .map(|length| valid[..length].to_vec())
        .collect();
    for (offset, value) in [(8, u64::MAX), (16, 0), (16, u64::MAX), (24, 32)] {
        let mut bytes = valid.clone();
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        malformed.push(bytes);
    }
    for (offset, value) in [(0, b'X'), (4, 2), (32, b'!'), (DATA_OFFSET - 1, 1)] {
        let mut bytes = valid.clone();
        bytes[offset] = value;
        malformed.push(bytes);
    }
    for field in ["stream_off", "lut_off", "scale_off"] {
        for offset in [0, valid.len() as u64, u64::MAX] {
            malformed.push(change_header(&valid, |header| {
                header["tensors"][0][field] = json!(offset);
            }));
        }
    }
    for (index, bytes) in malformed.iter().enumerate() {
        assert!(
            Container::from_bytes(bytes.clone()).is_err(),
            "reader case {index}"
        );
        let file = TempFile::new(bytes);
        // SAFETY: each malformed fixture is stable for the duration of open;
        // corrupt contents are explicitly allowed by the mapped-open contract.
        let result = unsafe { MappedModel::open(&file.0) };
        assert!(result.is_err(), "mapped case {index}");
    }
}

#[test]
fn kernel_rejects_invalid_shapes_without_poisoning_the_worker_pool() {
    let bytes = synthetic(6, "warp", 65, 96, 1024, false);
    let file = TempFile::new(&bytes);
    // SAFETY: the private file is never modified and outlives all mapping owners.
    let model = unsafe { MappedModel::open(&file.0) }.unwrap();
    let matrix = model.matrix("linear").unwrap();
    assert!(Engine::new(Kernel::Portable, 0).is_err());
    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
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
    assert_eq!(
        engine.weighted_sums(&matrix, &[0; 1024], &[1, -1]).unwrap(),
        vec![0; 96]
    );
}

#[test]
fn worker_tasks_cover_large_matrices_and_the_final_partial_task() {
    let bytes = synthetic(6, "warp", 1025, 1056, 512, false);
    let file = TempFile::new(&bytes);
    // SAFETY: this private fixture remains unchanged until all mapping owners drop.
    let model = unsafe { MappedModel::open(&file.0) }.unwrap();
    let matrix = model.matrix("linear").unwrap();
    let mut reader = Container::from_bytes(bytes).unwrap();
    let reference = CompressedLinear::new(reader.load_tensor("linear").unwrap(), false).unwrap();
    let mut seed = 932173;
    let activation: Vec<_> = (0..3 * 512).map(|_| random(&mut seed) as i8).collect();
    let scales = [16_777_216, -12345, 777];
    let expected: Vec<_> = (0..3)
        .flat_map(|m| {
            reference
                .weighted_sums(&activation[m * 512..(m + 1) * 512], &scales[m..m + 1], 1056)
                .unwrap()
        })
        .collect();
    for kernel in Kernel::available_variants() {
        for threads in [1, 4, 10] {
            let mut engine = Engine::new(kernel, threads).unwrap();
            assert_sums(
                &engine.weighted_sums(&matrix, &activation, &scales).unwrap(),
                &expected,
                &format!("large matrix {} threads={threads}", kernel.name()),
            );
        }
    }
}
