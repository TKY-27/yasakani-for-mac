//! The fast draft against the scalar reference draft, bit for bit (D18).
#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
mod support;

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use support::draft::{self as fixture, Shape};
use yskn_engine::{Backend, Engine, FastDraft, Kernel};
use yskn_ref::draft::{ContextFeatures, Draft, DraftOptions, DraftProposal};
use yskn_ref::draft_head::DraftChoice;

struct Folder(PathBuf);
impl Folder {
    fn new(seed: u64, shape: &Shape) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yskn-fast-draft-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(
            path.join("bdraft2_k8.yskn"),
            fixture::container_shape(seed, shape),
        )
        .unwrap();
        fs::write(
            path.join("bdraft2_k8.shead"),
            fixture::shead_shape(seed + 1, shape.vocab, shape.hidden),
        )
        .unwrap();
        Self(path)
    }
}
impl Drop for Folder {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn options(shape: &Shape) -> DraftOptions {
    DraftOptions {
        mask_token: shape.vocab as u32 - 1,
        capture_layers: (0..shape.slots).collect(),
        shortlist: 24,
    }
}

/// CPU portable/fastest kernels with one and four workers, plus Metal when present.
fn configurations() -> Vec<(String, Kernel, usize, Backend)> {
    let mut list = vec![
        (
            "cpu portable x1".to_owned(),
            Kernel::Portable,
            1,
            Backend::Cpu,
        ),
        (
            "cpu fastest x1".to_owned(),
            Kernel::fastest(),
            1,
            Backend::Cpu,
        ),
        (
            "cpu fastest x4".to_owned(),
            Kernel::fastest(),
            4,
            Backend::Cpu,
        ),
    ];
    #[cfg(target_os = "macos")]
    if yskn_engine::MetalEngine::device_exists() {
        list.push(("metal x4".to_owned(), Kernel::fastest(), 4, Backend::Metal));
    }
    if list.len() == 3 {
        println!("skipped: no Metal device");
    }
    list
}

fn fast(
    folder: &Folder,
    shape: &Shape,
    kernel: Kernel,
    threads: usize,
    backend: Backend,
) -> FastDraft {
    // SAFETY: the fixture files are never modified while any draft is alive.
    unsafe {
        FastDraft::with_options(
            &folder.0,
            Engine::new(kernel, threads).unwrap(),
            backend,
            options(shape),
        )
    }
    .unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn assert_proposals(actual: &DraftProposal, expected: &DraftProposal, context: &str) {
    assert_eq!(actual.tokens, expected.tokens, "{context}: tokens");
    for (name, a, e) in [
        ("candidates", &actual.candidates, &expected.candidates),
        ("corrected", &actual.corrected, &expected.corrected),
    ] {
        let flat = |c: &Vec<Vec<yskn_ref::draft_head::Candidate>>| -> Vec<(u32, u32)> {
            c.iter()
                .flatten()
                .map(|c| (c.token, c.score.to_bits()))
                .collect()
        };
        assert_eq!(a.len(), e.len(), "{context}: {name} positions");
        assert_eq!(flat(a), flat(e), "{context}: {name} ids/scores");
    }
}

/// (frontier, choice) cases and one explicit block, evaluated on any draft.
fn cases(shape: &Shape) -> Vec<(u32, DraftChoice)> {
    let last = shape.vocab as u32 - 2;
    vec![
        (3, DraftChoice::Greedy),
        (3, DraftChoice::sampled(17)),
        (last, DraftChoice::Greedy),
        (last, DraftChoice::sampled(0xabcdef)),
    ]
}
fn block_rows(shape: &Shape) -> Vec<Vec<f32>> {
    (0..8)
        .map(|t| fixture::embedding_width(t * 3 + 1, shape.hidden).unwrap())
        .collect()
}
struct Expected {
    proposals: Vec<DraftProposal>,
    block: Vec<u32>,
}
fn expected(reference: &Draft, shape: &Shape) -> Expected {
    let embedding = |token| fixture::embedding_width(token, shape.hidden);
    Expected {
        proposals: cases(shape)
            .into_iter()
            .map(|(frontier, choice)| reference.propose(frontier, embedding, choice).unwrap())
            .collect(),
        block: bits(&reference.block_hidden(&block_rows(shape)).unwrap()),
    }
}
fn compare_proposals(fast: &mut FastDraft, expected: &Expected, shape: &Shape, context: &str) {
    let embedding = |token| fixture::embedding_width(token, shape.hidden);
    for ((frontier, choice), expected) in cases(shape).into_iter().zip(&expected.proposals) {
        let actual = fast.propose(frontier, embedding, choice).unwrap();
        assert_proposals(
            &actual,
            expected,
            &format!("{context} f={frontier} {choice:?}"),
        );
    }
    assert_eq!(
        bits(&fast.block_hidden(&block_rows(shape)).unwrap()),
        expected.block,
        "{context}: block hidden"
    );
}

fn shapes() -> Vec<(u64, Shape, usize)> {
    vec![
        // The original fixture: equal heads, ring of 8 wrapped twice.
        (23, Shape::default(), 21),
        // Grouped heads, two 512-blocks, a partial vocabulary tile, wider rank.
        (
            31,
            Shape {
                hidden: 1024,
                ffn: 1024,
                query_heads: 8,
                kv_heads: 2,
                window: 16,
                layers: 1,
                vocab: 40,
                rank: 64,
                slots: 2,
            },
            37,
        ),
        // A ring of 256: multi-key history splits and a push batch above 256 rows.
        (
            41,
            Shape {
                query_heads: 8,
                kv_heads: 2,
                window: 256,
                layers: 1,
                ..Shape::default()
            },
            300,
        ),
    ]
}

#[test]
fn rings_proposals_and_top16_match_reference_on_every_backend_and_thread_count() {
    for (seed, shape, total) in shapes() {
        let folder = Folder::new(seed, &shape);
        let width = shape.slots * shape.hidden;
        let rows: Vec<Vec<f32>> = (0..total)
            .map(|r| fixture::features_width(r, width))
            .collect();
        // Uneven chunks: single rows, small batches, and one batch above the ring.
        let mut chunks = vec![1, 3, 1, shape.window + 1, 2];
        chunks.push(total - chunks.iter().sum::<usize>());
        // The scalar reference runs once per shape; every configuration replays it.
        let mut reference = Draft::load_with_options(&folder.0, options(&shape)).unwrap();
        let checkpoints = [0, 2, chunks.len()];
        let mut rings = Vec::new();
        let mut results = vec![expected(&reference, &shape)];
        let mut start = 0;
        for (i, &n) in chunks.iter().enumerate() {
            reference
                .push_context_rows(start, &rows[start..start + n].concat())
                .unwrap();
            start += n;
            rings.push(reference.snapshot());
            if checkpoints.contains(&(i + 1)) {
                results.push(expected(&reference, &shape));
            }
        }
        drop(reference);
        for (name, kernel, threads, backend) in configurations() {
            let mut fast = fast(&folder, &shape, kernel, threads, backend);
            assert_eq!(fast.backend(), backend);
            let context = format!("seed {seed} {name}");
            let mut results = results.iter();
            compare_proposals(
                &mut fast,
                results.next().unwrap(),
                &shape,
                &format!("{context} empty ring"),
            );
            let mut start = 0;
            for (i, &n) in chunks.iter().enumerate() {
                let batch = &rows[start..start + n];
                let flat: Vec<f32> = batch.concat();
                match i % 3 {
                    0 if n == 1 => fast
                        .push_context(start, ContextFeatures { values: &batch[0] })
                        .unwrap(),
                    1 => fast.push_context_rows(start, &flat).unwrap(),
                    _ => {
                        let slices: Vec<&[f32]> = batch.iter().map(Vec::as_slice).collect();
                        fast.push_rows(start, &slices).unwrap()
                    }
                }
                start += n;
                assert_eq!(
                    fast.snapshot(),
                    rings[i],
                    "{context} ring after {start} rows"
                );
                if checkpoints.contains(&(i + 1)) {
                    let context = format!("{context} position {start}");
                    compare_proposals(&mut fast, results.next().unwrap(), &shape, &context);
                }
            }
            assert!(start > shape.window, "ring has wrapped");
            eprintln!("{context}: {total} rows, rings/proposals/top16/block bits equal");
        }
    }
}

#[test]
fn failed_rows_snapshot_restore_and_reset_match_reference() {
    let shape = Shape::default();
    let folder = Folder::new(23, &shape);
    let good: Vec<f32> = (0..7)
        .flat_map(|r| fixture::features_width(r, 2560))
        .collect();
    let mut nan = fixture::features_width(9, 2560);
    nan[17] = f32::NAN;
    let mut not_bf16 = fixture::features_width(9, 2560);
    not_bf16[3] = 1.0 + f32::EPSILON;
    for (name, kernel, threads, backend) in configurations() {
        for (label, bad) in [("nan", &nan), ("not bf16", &not_bf16)] {
            let mut reference = Draft::load_with_options(&folder.0, options(&shape)).unwrap();
            let mut fast = fast(&folder, &shape, kernel, threads, backend);
            // Rows 0..4 commit, row 4 fails, rows after it never run.
            let mut flat = good[..4 * 2560].to_vec();
            flat.extend_from_slice(bad);
            flat.extend_from_slice(&good[4 * 2560..]);
            let reference_error = reference.push_context_rows(0, &flat).unwrap_err();
            let actual = fast.push_context_rows(0, &flat).unwrap_err();
            assert_eq!(
                actual.to_string(),
                reference_error.to_string(),
                "{name} {label}"
            );
            assert_eq!(fast.snapshot(), reference.snapshot(), "{name} {label}");
            assert_eq!(fast.position(), 4);
            // Wrong position and wrong width are rejected without changes.
            let row = &good[..2560];
            assert_eq!(
                fast.push_context(5, ContextFeatures { values: row })
                    .unwrap_err()
                    .to_string(),
                reference
                    .push_context(5, ContextFeatures { values: row })
                    .unwrap_err()
                    .to_string()
            );
            assert_eq!(
                fast.push_context(
                    4,
                    ContextFeatures {
                        values: &row[..512]
                    }
                )
                .unwrap_err()
                .to_string(),
                reference
                    .push_context(
                        4,
                        ContextFeatures {
                            values: &row[..512]
                        }
                    )
                    .unwrap_err()
                    .to_string()
            );
            assert_eq!(fast.snapshot(), reference.snapshot());
            // Restore a reference ring, propose from it, then reset both.
            reference.push_context_rows(4, &good).unwrap();
            fast.restore(&reference.snapshot()).unwrap();
            let context = format!("{name} restored");
            compare_proposals(&mut fast, &expected(&reference, &shape), &shape, &context);
            fast.reset();
            reference.reset();
            assert_eq!(fast.snapshot(), reference.snapshot());
            assert_eq!(fast.position(), 0);
        }
        let other = Shape {
            window: 16,
            ..Shape::default()
        };
        let folder_other = Folder::new(5, &other);
        let wrong = Draft::load_with_options(&folder_other.0, options(&other)).unwrap();
        let mut fast = fast(&folder, &shape, kernel, threads, backend);
        assert!(fast.restore(&wrong.snapshot()).is_err(), "{name}");
    }
}

#[test]
fn loading_rejects_the_reference_errors() {
    let shape = Shape::default();
    let folder = Folder::new(23, &shape);
    let load = |options: DraftOptions| {
        // SAFETY: fixture files are not modified while the draft is alive.
        unsafe {
            FastDraft::with_options(
                &folder.0,
                Engine::new(Kernel::Portable, 1).unwrap(),
                Backend::Cpu,
                options,
            )
        }
    };
    let bad = DraftOptions {
        shortlist: 20,
        ..options(&shape)
    };
    assert!(load(bad.clone()).is_err());
    assert!(Draft::load_with_options(&folder.0, bad).is_err());
    assert!(
        load(DraftOptions {
            capture_layers: vec![0, 1],
            ..options(&shape)
        })
        .is_err()
    );
    fs::write(
        folder.0.join("bdraft2_k8.shead"),
        fixture::shead_shape(3, 32, 1024),
    )
    .unwrap();
    assert!(load(options(&shape)).is_err());
}
