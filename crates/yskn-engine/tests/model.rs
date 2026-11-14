// Shared writer retains yskn-ref's Rust 1.85 syntax.
#[allow(clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
mod support;

use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use yskn_engine::{Backend, Engine, Kernel, KvMode, Model};
use yskn_format::Container;

struct Fixture(PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yskn-model-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("weights.yskn"), bytes).unwrap();
        fs::write(
            path.join("yasakani.json"),
            r#"{"files":{"weights":"weights.yskn"},"max_ctx":257}"#,
        )
        .unwrap();
        Self(path)
    }
    fn model(&self, kernel: Kernel, threads: usize, rows: usize) -> Model {
        // SAFETY: fixture contents are immutable until every model is dropped before Fixture.
        unsafe { Model::with_engine(&self.0, Engine::new(kernel, threads).unwrap(), rows) }.unwrap()
    }
    fn backend_model(&self, backend: Backend, threads: usize, rows: usize) -> Model {
        let engine = Engine::new(Kernel::fastest(), threads).unwrap();
        // SAFETY: fixture contents are immutable until every model is dropped before Fixture.
        unsafe { Model::with_backend(&self.0, engine, backend, rows) }.unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn same(actual: &[Vec<f32>], expected: &[Vec<f32>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (row, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.len(), b.len(), "{context} row={row}");
        for (v, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{context} row={row} vocabulary={v}"
            );
        }
    }
}
fn logit_bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .collect()
}
fn digest(rows: &[Vec<f32>]) -> String {
    let mut hash = Sha256::new();
    for row in rows {
        for value in row {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}
fn compare(bytes: Vec<u8>, count: usize, baseline: &str) {
    compare_mode(bytes, count, baseline, KvMode::Int8);
}
fn compare_mode(bytes: Vec<u8>, count: usize, baseline: &str, mode: KvMode) {
    let fixture = Fixture::new(&bytes);
    if mode != KvMode::Int8 {
        fs::write(
            fixture.0.join("kvq_u5.bin"),
            support::kv_profile::mixed_profile(),
        )
        .unwrap();
    }
    let mut reference = yskn_ref::Model::load_folder_with(&fixture.0, mode).unwrap();
    reference.prefill(&[1, 2, 3]).unwrap();
    let initial = reference.row_logits().to_vec();
    let tokens: Vec<_> = (0..count).map(|i| ((i * 7 + 11) % 65) as u32).collect();
    reference.prefill(&tokens).unwrap();
    let expected = reference.row_logits().to_vec();
    let int8_engine_logits = if mode != KvMode::Int8 {
        let mut int8_reference =
            yskn_ref::Model::load_folder_with(&fixture.0, KvMode::Int8).unwrap();
        int8_reference.prefill(&[1, 2, 3]).unwrap();
        int8_reference.prefill(&tokens).unwrap();
        assert_ne!(
            logit_bits(&expected),
            logit_bits(int8_reference.row_logits()),
            "reference ignored {mode:?}"
        );
        let mut int8_engine = fixture.model(Kernel::Portable, 1, 5);
        int8_engine.prefill(&[1, 2, 3]).unwrap();
        int8_engine.prefill(&tokens).unwrap();
        Some(logit_bits(int8_engine.row_logits()))
    } else {
        None
    };
    assert_eq!(
        digest(&expected),
        baseline,
        "frozen reference logit bits changed"
    );
    reference.prefill(&[17, 21]).unwrap();
    let continuation = reference.row_logits().to_vec();
    reference.reset().unwrap();
    reference.prefill(&[5, 13, 7, 1]).unwrap();
    let after_reset = reference.row_logits().to_vec();
    let check = |context: &str, mut model: Model| {
        assert_eq!(model.context_capacity(), 320);
        assert_eq!(model.kv_mode(), mode);
        model.prefill(&[1, 2, 3]).unwrap();
        same(model.row_logits(), &initial, context);
        model.prefill(&tokens).unwrap();
        same(model.row_logits(), &expected, context);
        if let Some(int8_logits) = &int8_engine_logits {
            assert_ne!(
                &logit_bits(model.row_logits()),
                int8_logits,
                "engine ignored {mode:?}: {context}"
            );
        }
        assert_eq!(model.position(), 3 + count);
        assert_eq!(model.logits().unwrap().len(), expected[0].len());
        model.reset().unwrap();
        assert_eq!(model.position(), 0);
        assert!(model.logits().is_none());
        model.prefill(&[5, 13, 7, 1]).unwrap();
        same(model.row_logits(), &after_reset, context);
        model.reset().unwrap();
        model.prefill(&[1, 2, 3]).unwrap();
        let mut actual = Vec::new();
        let mut first = 0;
        for chunk in [1, 3, 7, 2, 19, 5, 11, 23, 100] {
            if first == tokens.len() {
                break;
            }
            let end = (first + chunk).min(tokens.len());
            model.prefill(&tokens[first..end]).unwrap();
            actual.extend_from_slice(model.row_logits());
            first = end;
        }
        same(&actual, &expected, context);
        let last = model.row_logits().to_vec();
        model.prefill(&[]).unwrap();
        same(model.row_logits(), &last, "empty prefill");
        // Continuing the sequence proves the persistent cache/recurrence too.
        model.prefill(&[17, 21]).unwrap();
        let continued = model.row_logits().to_vec();
        same(&continued, &continuation, context);
    };
    let mut backends = vec![Backend::Cpu];
    if metal_device() {
        backends.push(Backend::Metal);
    }
    for backend in backends {
        let kernels = if mode == KvMode::Int8 {
            Kernel::available_variants()
        } else {
            vec![Kernel::fastest()]
        };
        let widths: &[usize] = if mode == KvMode::Int8 {
            &[1, 2, 5, 16, 17, 40]
        } else {
            &[1, 5, 16, 17]
        };
        for kernel in kernels {
            // Metal ignores the projection kernel; only run that backend once per width/pool.
            if backend == Backend::Metal && kernel != Kernel::fastest() {
                continue;
            }
            for threads in [1, 4] {
                for &rows in widths {
                    // Pass grouping is above Compute::weighted_sums/map_range. The
                    // linear suite keeps the full kernel × pool × M product; here
                    // cover every kernel/pool at width 17 and every width at x4.
                    if rows != 17 && (kernel != Kernel::fastest() || threads != 4) {
                        continue;
                    }
                    let context = format!(
                        "{mode:?} {backend:?} {} threads={threads} rows_per_pass={rows}",
                        kernel.name()
                    );
                    let model = if mode == KvMode::Int8 && backend == Backend::Cpu {
                        fixture.model(kernel, threads, rows)
                    } else {
                        // SAFETY: fixture contents stay immutable until this model is dropped.
                        unsafe {
                            Model::with_options(
                                &fixture.0,
                                Engine::new(kernel, threads).unwrap(),
                                backend,
                                rows,
                                mode,
                            )
                        }
                        .unwrap()
                    };
                    assert_eq!(model.backend(), backend);
                    check(&context, model);
                    println!("ran: {context}");
                }
            }
        }
    }
}
/// A missing device is reported on stdout and is the only way to skip Metal cases.
fn metal_device() -> bool {
    #[cfg(target_os = "macos")]
    if yskn_engine::MetalEngine::device_exists() {
        return true;
    }
    println!("skipped: no Metal device");
    false
}
#[test]
fn recurrent_model_is_identical_for_all_kernels_threads_pass_widths_and_chunks() {
    compare(
        support::synthetic_model(19, &["linear_attention", "linear_attention"]),
        9,
        "efd528122d8df0a2b6d0162f1acd4ccbefb699c6d4074f0981d6ac290183cc40",
    );
}
#[test]
fn mixed_model_crosses_attention_tiles_from_position_three_without_bit_changes() {
    compare(
        support::synthetic_model(
            19,
            &[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
        ),
        73,
        "3cf62a9fb84c5f8d5ff3cdd434f061014b0baafc773fa38166c3091a0142a8ba",
    );
}
#[test]
fn int8_profile_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation() {
    compare_mode(
        support::synthetic_model(
            107,
            &["linear_attention", "full_attention", "full_attention"],
        ),
        73,
        "bd759b4b86ab911371bf726f34e0234c321c9d6292e31133e7f0d2f5c6d53b8c",
        KvMode::Int8Profile,
    );
}
#[test]
fn kv5_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation() {
    compare_mode(
        support::synthetic_model(
            107,
            &["linear_attention", "full_attention", "full_attention"],
        ),
        73,
        "af825f48eb7eb1a8337f52b31b7018ab568907e996724d81139c1594b0e5fc26",
        KvMode::Kv5,
    );
}
#[test]
fn other_hidden_ffn_widths_and_padded_vocabulary_are_identical() {
    compare(
        support::synthetic_model_with_dimensions(21, &["linear_attention"], 1024, 1024, 65),
        9,
        "24b68b1e25f9e0fc3c75f82816989d90bcfda5f6621ba222fac1296078a7d553",
    );
}
#[test]
fn dense_embeddings_are_identical() {
    compare(
        support::synthetic_model_with_dense_embeddings(21, &["linear_attention"]),
        9,
        "2621a671d18e0c7e4bfc3f8525aabb2c3b92949a0c7e6ef34b7e7b634dccf5fb",
    );
}
#[test]
fn invalid_requests_preserve_logits_position_and_future_sequence_state() {
    let bytes = support::synthetic_model(17, &["linear_attention", "full_attention"]);
    let fixture = Fixture::new(&bytes);
    fs::write(
        fixture.0.join("yasakani.json"),
        r#"{"files":{"weights":"weights.yskn"},"max_ctx":1}"#,
    )
    .unwrap();
    let mut reference = yskn_ref::Model::load_folder(&fixture.0).unwrap();
    let mut model = fixture.model(Kernel::fastest(), 4, 5);
    assert_eq!(model.context_capacity(), 64);
    reference.prefill(&[1, 2, 3]).unwrap();
    model.prefill(&[1, 2, 3]).unwrap();
    let before = model.row_logits().to_vec();
    for tokens in [vec![1, 2, 128], vec![1; 62]] {
        assert!(matches!(
            model.prefill(&tokens),
            Err(yskn_engine::Error::InvalidInput(_))
        ));
        assert!(matches!(
            reference.prefill(&tokens),
            Err(yskn_ref::Error::InvalidInput(_))
        ));
        assert_eq!(model.position(), 3);
        same(model.row_logits(), &before, "failed pass");
        same(reference.row_logits(), &before, "reference failed pass");
    }
    model.prefill(&[9; 61]).unwrap();
    reference.prefill(&[9; 61]).unwrap();
    same(
        model.row_logits(),
        reference.row_logits(),
        "continuation after rejected pass",
    );
    assert!(model.prefill(&[1]).is_err());
    assert_eq!(model.position(), 64);
    // SAFETY: fixture contents remain stable until this rejected construction completes.
    assert!(
        unsafe {
            Model::with_engine(
                fixture.0.join("weights.yskn"),
                Engine::new(Kernel::Portable, 1).unwrap(),
                0,
            )
        }
        .is_err()
    );
}
#[test]
fn failure_after_a_completed_pass_preserves_logits_position_and_state() {
    // Token 100's dense embedding is finite but its squares overflow, so the
    // first layer norm fails in whichever pass contains it.
    let mut bytes = support::synthetic_model_with_dense_embeddings(21, &["linear_attention"]);
    let embedding = Container::from_bytes(bytes.clone())
        .unwrap()
        .header()
        .vectors
        .iter()
        .find(|v| v.name == "embed_tokens")
        .unwrap()
        .clone();
    let huge = yskn_format::f32_to_bf16(1.0e38).to_le_bytes();
    let row = embedding.offset as usize + 100 * 512 * 2;
    for pair in bytes[row..row + 1024].as_chunks_mut::<2>().0 {
        *pair = huge;
    }
    let fixture = Fixture::new(&bytes);
    let mut reference = yskn_ref::Model::load_folder(&fixture.0).unwrap();
    reference.prefill(&[1, 2, 3]).unwrap();
    assert!(reference.prefill(&[4, 5, 6, 7, 8, 100, 9]).is_err());
    reference.prefill(&[4, 5, 6, 7, 8]).unwrap();
    let continuation = reference.row_logits().to_vec();
    let mut backends = vec![Backend::Cpu];
    if metal_device() {
        backends.push(Backend::Metal);
    }
    for backend in backends {
        let context = backend.name();
        let mut model = fixture.backend_model(backend, 4, 2);
        model.prefill(&[1, 2, 3]).unwrap();
        let before = model.row_logits().to_vec();
        let timings = model.pass_timings().len();
        // Passes [4, 5] and [6, 7] complete before [8, 100] fails.
        assert!(matches!(
            model.prefill(&[4, 5, 6, 7, 8, 100, 9]),
            Err(yskn_engine::Error::NonFinite(_))
        ));
        assert_eq!(model.position(), 3, "{context}");
        assert_eq!(model.pass_timings().len(), timings, "{context}");
        same(model.row_logits(), &before, context);
        // Unchanged recurrent state: continuing equals the reference continuing.
        model.prefill(&[4, 5, 6, 7, 8]).unwrap();
        same(model.row_logits(), &continuation, context);
    }
}
#[test]
fn persistent_range_pool_preserves_order_borrows_and_recovers_after_errors() {
    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
    let input: Vec<_> = (0..101).map(|i| i * 3).collect();
    assert_eq!(
        engine.map_range(3..99, |i| Ok(input[i] + 1)).unwrap(),
        input[3..99].iter().map(|x| x + 1).collect::<Vec<_>>()
    );
    assert!(
        engine
            .map_range(0..101, |i| if i == 8 {
                Err(yskn_engine::Error::InvalidInput("expected".into()))
            } else {
                Ok(input[i])
            })
            .is_err()
    );
    assert!(
        engine
            .map_range(0..101, |i| -> yskn_engine::Result<usize> {
                if i == 8 {
                    panic!("expected range closure panic");
                }
                Ok(i)
            })
            .is_err()
    );
    // Several failures, including a panic after the first: the lowest index wins.
    for _ in 0..20 {
        let error = engine
            .map_range(0..101, |i| match i {
                97 => panic!("expected range closure panic"),
                _ if i % 10 == 3 => Err(yskn_engine::Error::InvalidInput(format!("index {i}"))),
                _ => Ok(i),
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "invalid input: index 3");
    }
    assert_eq!(engine.map_range(0..101, |i| Ok(input[i])).unwrap(), input);
    assert!(engine.map_range(0..0, |_| Ok(())).unwrap().is_empty());
}

#[test]
fn load_selects_env_kv_and_explicit_constructors_keep_int8_default() {
    const CHILD: &str = "YSKN_TEST_KV_FOLDER";
    if let Some(folder) = std::env::var_os(CHILD) {
        let value = std::env::var_os("YSKN_KV");
        let expected = value
            .as_deref()
            .map(yskn_ref::quality::parse_kv_mode)
            .transpose();
        // SAFETY: the parent keeps this fixture immutable until this child exits.
        let loaded = unsafe { Model::load(&folder) };
        match expected {
            Ok(mode) => assert_eq!(loaded.unwrap().kv_mode(), mode.unwrap_or_default()),
            Err(_) => assert!(
                matches!(loaded, Err(yskn_engine::Error::InvalidInput(message))
                if message.contains("YSKN_KV") && !message.contains("--kv"))
            ),
        }
        let fixture_engine = || Engine::new(Kernel::Portable, 1).unwrap();
        // SAFETY: the parent keeps this fixture immutable until this child exits.
        let cpu = unsafe { Model::with_engine(&folder, fixture_engine(), 5) }.unwrap();
        // SAFETY: same immutable fixture contract.
        let backend =
            unsafe { Model::with_backend(&folder, fixture_engine(), Backend::Cpu, 5) }.unwrap();
        assert_eq!(cpu.kv_mode(), KvMode::Int8);
        assert_eq!(backend.kv_mode(), KvMode::Int8);
        return;
    }
    let fixture = Fixture::new(&support::synthetic_model(
        107,
        &["linear_attention", "full_attention", "full_attention"],
    ));
    fs::write(
        fixture.0.join("kvq_u5.bin"),
        support::kv_profile::mixed_profile(),
    )
    .unwrap();
    for value in [
        None,
        Some("int8"),
        Some("int8-profile"),
        Some("kv5"),
        Some("invalid"),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "load_selects_env_kv_and_explicit_constructors_keep_int8_default",
            ])
            .env(CHILD, &fixture.0)
            .env("YSKN_BACKEND", "cpu")
            .env("YSKN_KERNEL", "portable")
            .env_remove("YSKN_KV");
        if let Some(value) = value {
            child.env("YSKN_KV", value);
        }
        let output = child.output().unwrap();
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child did not run the test: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(
            output.status.success(),
            "kv={value:?}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// D14: explicit constructors read no Metal option from the environment (binary32,
/// calibrated share); `Model::load` applies `YSKN_METAL_MAC` and
/// `YSKN_METAL_CPU_SHARE` on Metal, rejects unknown values, and reads neither on CPU.
#[test]
fn metal_options_come_from_the_environment_only_when_loading() {
    const CHILD: &str = "YSKN_TEST_METAL_FOLDER";
    const FAILS: &str = "YSKN_TEST_METAL_LOAD_FAILS";
    if let Some(folder) = std::env::var_os(CHILD) {
        let engine = || Engine::new(Kernel::fastest(), 2).unwrap();
        // SAFETY: the parent keeps this fixture immutable until this child exits.
        let mut explicit =
            unsafe { Model::with_backend(&folder, engine(), Backend::Metal, 5) }.unwrap();
        // SAFETY: same immutable fixture contract.
        let loaded = unsafe { Model::load(&folder) };
        if std::env::var_os(FAILS).is_some() {
            assert!(
                matches!(&loaded, Err(yskn_engine::Error::InvalidInput(message))
                    if message.contains("YSKN_METAL_")),
                "{:?}",
                loaded.err()
            );
            return;
        }
        let mut loaded = loaded.unwrap();
        // The environment's mode and share cannot change a bit either.
        let tokens = [3, 17, 42, 5, 99, 0, 7];
        explicit.prefill(&tokens).unwrap();
        loaded.prefill(&tokens).unwrap();
        same(
            loaded.row_logits(),
            explicit.row_logits(),
            "load vs explicit",
        );
        return;
    }
    if !metal_device() {
        return;
    }
    let fixture = Fixture::new(&support::synthetic_model(
        107,
        &["linear_attention", "full_attention"],
    ));
    for (backend, mac, share, fails) in [
        ("metal", None, None, false),
        ("metal", Some("integer"), Some("off"), false),
        ("metal", Some("float"), Some("0.5"), false),
        ("metal", Some("invalid"), None, true),
        ("metal", None, Some("invalid"), true),
        ("metal", None, Some("0.95"), true),
        ("cpu", Some("invalid"), Some("invalid"), false),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "metal_options_come_from_the_environment_only_when_loading",
            ])
            .env(CHILD, &fixture.0)
            .env("YSKN_BACKEND", backend)
            .env_remove("YSKN_KERNEL")
            .env_remove("YSKN_KV")
            .env_remove("YSKN_METAL_MAC")
            .env_remove("YSKN_METAL_CPU_SHARE")
            .env_remove(FAILS);
        for (name, value) in [("YSKN_METAL_MAC", mac), ("YSKN_METAL_CPU_SHARE", share)] {
            if let Some(value) = value {
                child.env(name, value);
            }
        }
        if fails {
            child.env(FAILS, "1");
        }
        let output = child.output().unwrap();
        let context = format!("backend={backend} mac={mac:?} share={share:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "{context}: child did not run the test: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(
            output.status.success(),
            "{context}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    println!("ran: Metal options from the environment");
}

/// D23: the explicit per-model CPU-stage choice (the product's `--cpu-stages`)
/// overrides the process default without an environment variable; both
/// choices give the scalar reference's bits, also through a provisional pass
/// and partial commit, and the choice cannot change while a pass is pending.
#[test]
fn explicit_cpu_stage_choice_overrides_the_process_default_with_identical_bits() {
    let fixture = Fixture::new(&support::synthetic_model(
        107,
        &["linear_attention", "full_attention"],
    ));
    let tokens = [3, 17, 42, 5, 99, 0, 7, 64, 11];
    let mut reference = yskn_ref::Model::load_folder(&fixture.0).unwrap();
    reference.prefill(&tokens).unwrap();
    for stages in [
        yskn_engine::CpuStages::Reference,
        yskn_engine::CpuStages::Auto,
    ] {
        let mut model = fixture.model(Kernel::fastest(), 3, 4);
        model.set_cpu_stages(stages).unwrap();
        assert_eq!(model.cpu_stages(), stages.fast(), "{stages:?}");
        model.prefill(&tokens[..5]).unwrap();
        model.run_provisional(&tokens[5..]).unwrap();
        assert!(model.set_cpu_stages(stages).is_err(), "{stages:?}");
        model.commit(2).unwrap();
        model.prefill(&tokens[7..]).unwrap();
        same(
            &model.row_logits()[model.row_logits().len() - 1..],
            &reference.row_logits()[reference.row_logits().len() - 1..],
            &format!("{stages:?}"),
        );
        assert_eq!(model.state_digest(), reference.state_digest(), "{stages:?}");
    }
}
