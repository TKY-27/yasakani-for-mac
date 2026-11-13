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
use yskn_engine::{Engine, Kernel, Model};
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
    let fixture = Fixture::new(&bytes);
    let mut reference =
        yskn_ref::Model::from_container(Container::from_bytes(bytes).unwrap()).unwrap();
    reference.prefill(&[1, 2, 3]).unwrap();
    let initial = reference.row_logits().to_vec();
    let tokens: Vec<_> = (0..count).map(|i| ((i * 7 + 11) % 65) as u32).collect();
    reference.prefill(&tokens).unwrap();
    let expected = reference.row_logits().to_vec();
    assert_eq!(
        digest(&expected),
        baseline,
        "reference bits changed from pre-change HEAD"
    );
    reference.prefill(&[17, 21]).unwrap();
    let continuation = reference.row_logits().to_vec();
    reference.reset().unwrap();
    reference.prefill(&[5, 13, 7, 1]).unwrap();
    let after_reset = reference.row_logits().to_vec();
    for kernel in Kernel::available_variants() {
        for threads in [1, 4] {
            for rows in [1, 2, 5, 16, 17, 40] {
                let context = format!("{} threads={threads} rows_per_pass={rows}", kernel.name());
                let mut model = fixture.model(kernel, threads, rows);
                assert_eq!(model.context_capacity(), 320);
                model.prefill(&[1, 2, 3]).unwrap();
                same(model.row_logits(), &initial, &context);
                model.prefill(&tokens).unwrap();
                same(model.row_logits(), &expected, &context);
                assert_eq!(model.position(), 3 + count);
                assert_eq!(model.logits().unwrap().len(), expected[0].len());
                model.reset().unwrap();
                assert_eq!(model.position(), 0);
                assert!(model.logits().is_none());
                model.prefill(&[5, 13, 7, 1]).unwrap();
                same(model.row_logits(), &after_reset, &context);
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
                same(&actual, &expected, &context);
                let last = model.row_logits().to_vec();
                model.prefill(&[]).unwrap();
                same(model.row_logits(), &last, "empty prefill");
                // Continuing the sequence proves the persistent cache/recurrence too.
                model.prefill(&[17, 21]).unwrap();
                let continued = model.row_logits().to_vec();
                same(&continued, &continuation, &context);
            }
        }
    }
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
    let mut model = fixture.model(Kernel::fastest(), 4, 2);
    reference.prefill(&[1, 2, 3]).unwrap();
    model.prefill(&[1, 2, 3]).unwrap();
    let before = model.row_logits().to_vec();
    let timings = model.pass_timings().len();
    // Passes [4, 5] and [6, 7] complete before [8, 100] fails.
    assert!(matches!(
        model.prefill(&[4, 5, 6, 7, 8, 100, 9]),
        Err(yskn_engine::Error::NonFinite(_))
    ));
    assert!(reference.prefill(&[4, 5, 6, 7, 8, 100, 9]).is_err());
    assert_eq!(model.position(), 3);
    assert_eq!(model.pass_timings().len(), timings);
    same(model.row_logits(), &before, "failed later pass");
    // Unchanged recurrent state: continuing equals the reference continuing.
    model.prefill(&[4, 5, 6, 7, 8]).unwrap();
    reference.prefill(&[4, 5, 6, 7, 8]).unwrap();
    same(
        model.row_logits(),
        reference.row_logits(),
        "continuation after a failed later pass",
    );
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
