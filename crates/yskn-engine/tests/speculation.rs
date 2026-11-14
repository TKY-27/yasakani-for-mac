#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
mod support;

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use yskn_engine::{Backend, Engine, Kernel, KvMode, LogitRows, Model};
use yskn_format::Container;
use yskn_ref::speculation::{
    GenerationLimits, GenerationResult, NgramProposer, NoProposer, Proposer, ScriptProposer,
    StopReason, Verifier, generate,
};
use yskn_ref::{Sampler, SamplingParameters};

struct Fixture(PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yskn-speculation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("weights.yskn"), bytes).unwrap();
        fs::write(
            path.join("yasakani.json"),
            r#"{"files":{"weights":"weights.yskn"},"max_ctx":1024}"#,
        )
        .unwrap();
        fs::write(
            path.join("kvq_u5.bin"),
            support::kv_profile::mixed_profile(),
        )
        .unwrap();
        Self(path)
    }
    fn engine(&self, backend: Backend, mode: KvMode) -> Model {
        self.engine_width(backend, mode, 9)
    }
    fn engine_width(&self, backend: Backend, mode: KvMode, width: usize) -> Model {
        // SAFETY: fixture files are immutable and outlive every model using them.
        unsafe {
            Model::with_options(
                &self.0,
                Engine::new(Kernel::fastest(), 2).unwrap(),
                backend,
                width,
                mode,
            )
        }
        .unwrap()
    }
    fn reference(&self, mode: KvMode) -> yskn_ref::Model {
        yskn_ref::Model::load_folder_with(&self.0, mode).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn backends() -> Vec<Backend> {
    let mut result = vec![Backend::Cpu];
    #[cfg(target_os = "macos")]
    if yskn_engine::MetalEngine::device_exists() {
        result.push(Backend::Metal);
        return result;
    }
    println!("skipped: no Metal device");
    result
}
fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .collect()
}
fn random(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as usize
}
fn mixed() -> Fixture {
    let fixture = Fixture::new(&support::synthetic_model(
        163,
        &["linear_attention", "full_attention"],
    ));
    fs::write(
        fixture.0.join("kvq_u5.bin"),
        support::kv_profile::compact_mixed_profile(),
    )
    .unwrap();
    fixture
}

fn check_partial_commits(mode: KvMode) {
    let fixture = Fixture::new(&support::synthetic_model(
        163,
        &["linear_attention", "full_attention", "full_attention"],
    ));
    for backend in backends() {
        let mut engine = fixture.engine(backend, mode);
        let mut reference = fixture.reference(mode);
        engine.prefill(&[8, 21, 5]).unwrap();
        reference.prefill(&[8, 21, 5]).unwrap();
        let mut seed = 0x4567_89ab_cdef_u64;
        let mut pairs = [[false; 10]; 10];
        let mut rolled_back = [false; 2];
        let mut recrossed = [false; 2];
        // Compute across each tile edge, reject back before it, then cross again.
        let boundaries = [(29, 6, 1), (30, 6, 4), (61, 6, 1), (62, 6, 4)]
            .into_iter()
            .map(|(start, width, accepted)| (Some(start), width, accepted));
        let sweep = (1..=9).flat_map(|width| (0..=width).map(move |k| (None, width, k)));
        for (pass, (start, width, accepted)) in boundaries.chain(sweep).enumerate() {
            if let Some(start) = start {
                assert!(engine.position() <= start);
                let padding: Vec<_> = (engine.position()..start)
                    .map(|_| (random(&mut seed) % 128) as u32)
                    .collect();
                engine.prefill(&padding).unwrap();
                reference.prefill(&padding).unwrap();
                assert_eq!(engine.position(), start);
                assert_eq!(engine.state_digest(), reference.state_digest());
                assert_eq!(bits(engine.row_logits()), bits(reference.row_logits()));
            }
            let start = engine.position();
            pairs[width][accepted] = true;
            let tokens: Vec<_> = (0..width)
                .map(|_| (random(&mut seed) % 128) as u32)
                .collect();
            engine.run_provisional(&tokens).unwrap();
            reference.run_provisional(&tokens).unwrap();
            assert_eq!(
                bits(engine.row_logits()),
                bits(reference.row_logits()),
                "{backend:?} {mode:?} pass {pass}"
            );
            assert!(engine.run_provisional(&[1]).is_err());
            assert!(engine.commit(width + 1).is_err());
            engine.commit(accepted).unwrap();
            reference.commit(accepted).unwrap();
            for (i, boundary) in [32, 64].into_iter().enumerate() {
                if start < boundary && start + width > boundary && engine.position() < boundary {
                    rolled_back[i] = true;
                } else if rolled_back[i] && start < boundary && engine.position() > boundary {
                    recrossed[i] = true;
                }
            }
            assert_eq!(engine.position(), reference.position());
            assert_eq!(
                engine.state_digest(),
                reference.state_digest(),
                "{backend:?} {mode:?} pass {pass} accepted {accepted}"
            );
            // A future row checks actual cache/recurrence semantics in addition to the digest.
            engine.run_provisional(&[17]).unwrap();
            reference.run_provisional(&[17]).unwrap();
            assert_eq!(bits(engine.row_logits()), bits(reference.row_logits()));
            engine.discard_provisional();
            reference.discard_provisional();
            let digest = engine.state_digest();
            let logits = bits(engine.row_logits());
            assert!(engine.run_provisional(&[1, 128]).is_err());
            assert!(engine.run_provisional(&[]).is_err());
            assert_eq!(engine.state_digest(), digest);
            assert_eq!(bits(engine.row_logits()), logits);
            assert!(!engine.has_provisional_pass());
        }
        for (width, accepted) in pairs.iter().enumerate().skip(1) {
            assert!(accepted[..=width].iter().all(|&seen| seen), "width {width}");
        }
        assert!(
            rolled_back.into_iter().all(|seen| seen),
            "rollback before 32 and 64"
        );
        assert!(
            recrossed.into_iter().all(|seen| seen),
            "re-cross 32 and 64 after rollback"
        );
    }
}

#[test]
fn randomized_partial_commits_and_discards_match_reference_int8_every_backend() {
    check_partial_commits(KvMode::Int8);
}

#[test]
fn randomized_partial_commits_and_discards_match_reference_int8_profile_every_backend() {
    check_partial_commits(KvMode::Int8Profile);
}

#[test]
fn randomized_partial_commits_and_discards_match_reference_kv5_every_backend() {
    check_partial_commits(KvMode::Kv5);
}

#[test]
fn failed_numeric_pass_preserves_committed_state_and_future_rows() {
    let mut bytes = support::synthetic_model_with_dense_embeddings(
        167,
        &["linear_attention", "full_attention"],
    );
    let descriptor = Container::from_bytes(bytes.clone())
        .unwrap()
        .header()
        .vectors
        .iter()
        .find(|v| v.name == "embed_tokens")
        .unwrap()
        .clone();
    let start = descriptor.offset as usize + 100 * 512 * 2;
    for pair in bytes[start..start + 1024].as_chunks_mut::<2>().0 {
        *pair = yskn_format::f32_to_bf16(1.0e38).to_le_bytes();
    }
    let fixture = Fixture::new(&bytes);
    for backend in backends() {
        let mut engine = fixture.engine(backend, KvMode::Int8);
        let mut reference = fixture.reference(KvMode::Int8);
        engine.prefill(&[1, 2]).unwrap();
        reference.prefill(&[1, 2]).unwrap();
        let digest = engine.state_digest();
        let logits = bits(engine.row_logits());
        assert!(matches!(
            engine.run_provisional(&[4, 100]),
            Err(yskn_engine::Error::NonFinite(_))
        ));
        assert!(reference.run_provisional(&[4, 100]).is_err());
        assert_eq!(engine.state_digest(), digest);
        assert_eq!(bits(engine.row_logits()), logits);
        assert!(!engine.has_provisional_pass());
        assert!(
            engine
                .prefill(&[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100])
                .is_err()
        );
        assert_eq!(engine.state_digest(), digest);
        assert_eq!(bits(engine.row_logits()), logits);
        engine.prefill(&[4, 5]).unwrap();
        reference.prefill(&[4, 5]).unwrap();
        assert_eq!(bits(engine.row_logits()), bits(reference.row_logits()));
        assert_eq!(engine.state_digest(), reference.state_digest());
    }
}

#[test]
fn snapshots_restore_prefix_without_growing_persistent_storage_and_reject_misuse() {
    let fixture = mixed();
    for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
        let mut reference = fixture.reference(mode);
        reference.prefill(&[1, 2, 3, 8, 9, 10]).unwrap();
        let expected_digest = reference.state_digest();
        let expected = bits(reference.row_logits())[3..].to_vec();
        for backend in backends() {
            let mut engine = fixture.engine(backend, mode);
            let empty = engine.snapshot().unwrap();
            engine.prefill(&[1, 2, 3]).unwrap();
            let prefix = engine.snapshot().unwrap();
            engine.prefill(&[4, 5]).unwrap();
            let future = engine.snapshot().unwrap();
            engine.restore(&prefix).unwrap();
            let digest = engine.state_digest();
            assert!(engine.restore(&future).is_err());
            assert_eq!(engine.state_digest(), digest);
            engine.run_provisional(&[4]).unwrap();
            assert!(engine.snapshot().is_err());
            assert!(engine.restore(&prefix).is_err());
            engine.discard_provisional();
            let mut other = fixture.engine(backend, mode);
            other.prefill(&[1, 2, 3]).unwrap();
            assert!(other.restore(&prefix).is_err());
            // Warm enough suffix/cache capacity once before measuring reuse.
            engine.prefill(&[4, 5, 6, 7]).unwrap();
            engine.restore(&prefix).unwrap();
            let memory = engine.session_memory();
            for _ in 0..4 {
                engine.prefill(&[11, 12]).unwrap();
                engine.restore(&prefix).unwrap();
                engine.prefill(&[8, 9, 10]).unwrap();
                assert_eq!(engine.state_digest(), expected_digest);
                assert_eq!(bits(engine.row_logits()), expected);
                engine.restore(&prefix).unwrap();
                assert_eq!(engine.session_memory(), memory);
            }
            engine.restore(&empty).unwrap();
            engine.prefill(&[1, 7, 3]).unwrap();
            let digest = engine.state_digest();
            assert!(engine.restore(&prefix).is_err());
            assert_eq!(engine.state_digest(), digest);
            engine.reset().unwrap();
            engine.prefill(&[1, 2, 3]).unwrap();
            assert!(engine.restore(&prefix).is_err());
            assert!(engine.restore(&empty).is_err());
        }
    }
}

#[test]
fn all_last_and_none_head_requests_preserve_the_same_future_numeric_state() {
    let fixture = mixed();
    for backend in backends() {
        for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
            let mut all = fixture.engine(backend, mode);
            all.prefill(&[1, 2]).unwrap();
            let snapshot = all.snapshot().unwrap();
            let prompt: Vec<_> = (3..16).collect();
            all.prefill_with_logits(&prompt, LogitRows::All).unwrap();
            let rows = bits(all.row_logits());
            all.prefill(&[17, 18]).unwrap();
            let expected = bits(all.row_logits());
            let digest = all.state_digest();
            for requested in [LogitRows::Last, LogitRows::None] {
                all.restore(&snapshot).unwrap();
                all.prefill_with_logits(&prompt, requested).unwrap();
                if requested == LogitRows::Last {
                    assert_eq!(bits(all.row_logits()), vec![rows.last().unwrap().clone()]);
                } else {
                    assert!(all.row_logits().is_empty());
                    assert!(all.logits().is_none());
                }
                all.prefill(&[17, 18]).unwrap();
                assert_eq!(bits(all.row_logits()), expected);
                assert_eq!(all.state_digest(), digest);
            }
        }
    }
}

struct Guided {
    script: Vec<u32>,
    initial_history: usize,
    kind: usize,
    random: u64,
}
impl Proposer for Guided {
    fn propose(&mut self, history: &[u32], _last: u32, max: usize) -> Vec<u32> {
        let offset = history.len() - self.initial_history;
        let good = match self.kind {
            0 => max,
            1 => 0,
            _ => random(&mut self.random) % (max + 1),
        };
        self.script
            .iter()
            .skip(offset)
            .take(max)
            .enumerate()
            .map(|(depth, &token)| {
                if depth < good {
                    token
                } else {
                    (token + 1) % 128
                }
            })
            .collect()
    }
}
enum Draft {
    None(NoProposer),
    Guided(Guided),
    Eos(ScriptProposer),
    Ngram(NgramProposer),
}
impl Proposer for Draft {
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        match self {
            Self::None(p) => p.propose(history, last, max),
            Self::Guided(p) => p.propose(history, last, max),
            Self::Eos(p) => p.propose(history, last, max),
            Self::Ngram(p) => p.propose(history, last, max),
        }
    }
}
fn draft(kind: usize, script: &[u32], initial_history: usize, k: usize) -> Draft {
    match kind {
        0 => Draft::None(NoProposer),
        1..=3 => Draft::Guided(Guided {
            script: script.to_vec(),
            initial_history,
            kind: kind - 1,
            random: 0x1234_5678,
        }),
        4 => Draft::Eos(ScriptProposer::new(vec![script[3]; k])),
        _ => Draft::Ngram(NgramProposer::new(2, k).unwrap()),
    }
}
#[allow(clippy::too_many_arguments)] // Keep every generation boundary explicit at each matrix call.
fn run_generation<V: Verifier>(
    model: &mut V,
    sampler: &Sampler,
    proposer: &mut Draft,
    prompt: &[u32],
    k: usize,
    capacity: usize,
    eos: &[u32],
    cancel_after: Option<usize>,
) -> GenerationResult
where
    V::Error: std::fmt::Debug,
{
    let mut seen = Vec::new();
    let mut callback = |token| {
        seen.push(token);
        cancel_after != Some(seen.len())
    };
    let mut limits = GenerationLimits::new(8, capacity);
    limits.max_draft_tokens = k;
    limits.eos_ids = eos;
    limits.on_token = Some(&mut callback);
    let result = generate(model, proposer, &mut sampler.clone(), prompt, limits).unwrap();
    assert_eq!(seen, result.tokens);
    assert_eq!(
        result
            .statistics
            .tokens_confirmed_per_pass
            .iter()
            .sum::<usize>(),
        result.tokens.len()
    );
    result
}

fn check_generation(sampler: Sampler) {
    let fixture = Fixture::new(&support::synthetic_model(
        173,
        &["linear_attention", "full_attention"],
    ));
    let prompt = [1, 2, 3, 1, 2, 3, 1];
    let mut reference = fixture.reference(KvMode::Int8);
    reference.prefill(&prompt[..prompt.len() - 1]).unwrap();
    let reference_start = reference.snapshot().unwrap();
    let baseline = run_generation(
        &mut reference,
        &sampler,
        &mut Draft::None(NoProposer),
        &prompt,
        0,
        64,
        &[],
        None,
    );
    let baseline_digest = reference.state_digest();
    assert_eq!(baseline.tokens.len(), 8);
    let mut expected = Vec::new();
    for k in [1, 2, 3, 7] {
        for kind in 0..6 {
            // NoProposer always returns zero guesses, independent of the K cap.
            if kind == 0 {
                expected.push((k, kind, baseline.clone(), baseline_digest));
                continue;
            }
            reference.restore(&reference_start).unwrap();
            let result = run_generation(
                &mut reference,
                &sampler,
                &mut draft(kind, &baseline.tokens, prompt.len() - 1, k),
                &prompt,
                k,
                64,
                &[],
                None,
            );
            expected.push((k, kind, result, reference.state_digest()));
        }
    }
    let mut stops = Vec::new();
    for (capacity, eos, cancel, reason) in [
        (prompt.len() + 2, vec![], None, StopReason::ContextCapacity),
        (64, vec![], Some(3), StopReason::Cancelled),
        (
            64,
            vec![baseline.tokens[3]],
            None,
            StopReason::EndOfSequence,
        ),
    ] {
        reference.restore(&reference_start).unwrap();
        let result = run_generation(
            &mut reference,
            &sampler,
            &mut Draft::None(NoProposer),
            &prompt,
            0,
            capacity,
            &eos,
            cancel,
        );
        stops.push((
            capacity,
            eos,
            cancel,
            reason,
            result,
            reference.state_digest(),
        ));
    }
    for backend in backends() {
        let mut engine = fixture.engine(backend, KvMode::Int8);
        engine.prefill(&prompt[..prompt.len() - 1]).unwrap();
        let engine_start = engine.snapshot().unwrap();
        for &(k, kind, ref actual_ref, reference_digest) in &expected {
            let mut previous_stats = None;
            for _ in 0..2 {
                engine.restore(&engine_start).unwrap();
                let actual_engine = run_generation(
                    &mut engine,
                    &sampler,
                    &mut draft(kind, &baseline.tokens, prompt.len() - 1, k),
                    &prompt,
                    k,
                    64,
                    &[],
                    None,
                );
                assert_eq!(
                    actual_ref.tokens, baseline.tokens,
                    "reference kind={kind} k={k}"
                );
                assert_eq!(
                    actual_engine.tokens, baseline.tokens,
                    "{backend:?} kind={kind} k={k}"
                );
                assert_eq!(reference_digest, baseline_digest);
                assert_eq!(engine.state_digest(), baseline_digest);
                assert_eq!(
                    actual_engine.statistics.tokens_confirmed_per_pass,
                    actual_ref.statistics.tokens_confirmed_per_pass
                );
                if let Some(previous) = previous_stats {
                    assert_eq!(actual_engine.statistics.tokens_confirmed_per_pass, previous);
                }
                previous_stats = Some(actual_engine.statistics.tokens_confirmed_per_pass.clone());
                if kind == 1 {
                    assert_eq!(
                        actual_engine.statistics.guesses_offered,
                        actual_engine.statistics.guesses_accepted
                    );
                    assert!(actual_engine.statistics.passes < 8);
                }
                if kind == 2 {
                    assert_eq!(actual_engine.statistics.guesses_accepted, 0);
                }
                if kind == 5 {
                    assert!(actual_engine.statistics.guesses_offered > 0);
                }
            }
        }
        // Every stop boundary retains the scalar no-draft committed state.
        for (capacity, eos, cancel, expected_reason, expected, digest) in &stops {
            for k in [1, 2, 3, 7] {
                for kind in [1, 4] {
                    engine.restore(&engine_start).unwrap();
                    let actual = run_generation(
                        &mut engine,
                        &sampler,
                        &mut draft(kind, &baseline.tokens, prompt.len() - 1, k),
                        &prompt,
                        k,
                        *capacity,
                        eos,
                        *cancel,
                    );
                    assert_eq!(actual.tokens, expected.tokens);
                    assert_eq!(actual.stop_reason, *expected_reason);
                    assert_eq!(engine.state_digest(), *digest);
                }
            }
        }
    }
}

#[test]
fn greedy_generation_is_lossless_repeatable_for_every_proposer() {
    check_generation(Sampler::Greedy);
}
#[test]
fn seed17_generation_is_lossless_repeatable_for_every_proposer() {
    check_generation(Sampler::Sampled(SamplingParameters {
        seed: 17,
        ..SamplingParameters::default()
    }));
}
#[test]
fn seedabcdef_generation_is_lossless_repeatable_for_every_proposer() {
    check_generation(Sampler::Sampled(SamplingParameters {
        seed: 0xabcdef,
        ..SamplingParameters::default()
    }));
}

#[test]
fn snapshot_after_partial_commit_restores_exact_prefix_and_future_state() {
    let fixture = mixed();
    for backend in backends() {
        for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
            let mut engine = fixture.engine(backend, mode);
            engine.prefill(&[1, 2, 3]).unwrap();
            engine.run_provisional(&[4, 5, 6]).unwrap();
            engine.commit(2).unwrap();
            let snapshot = engine.snapshot().unwrap();
            engine.prefill(&[11, 12, 13]).unwrap();
            engine.restore(&snapshot).unwrap();
            engine.prefill(&[7, 8]).unwrap();
            let mut reference = fixture.reference(mode);
            reference.prefill(&[1, 2, 3, 4, 5, 7, 8]).unwrap();
            assert_eq!(engine.state_digest(), reference.state_digest());
            assert_eq!(bits(engine.row_logits()), bits(reference.row_logits())[5..]);
        }
    }
}

fn capture_bits(rows: &[yskn_ref::model::CapturedRow]) -> Vec<(usize, Vec<u32>)> {
    rows.iter()
        .map(|row| {
            (
                row.position,
                row.values.iter().map(|v| v.to_bits()).collect(),
            )
        })
        .collect()
}

#[test]
fn residual_capture_matches_reference_across_widths_kv_backends_and_session_operations() {
    use yskn_ref::model::CaptureOptions;
    // Capture slots 0 and 2 require the original three-layer fixture.
    let fixture = Fixture::new(&support::synthetic_model(
        163,
        &["linear_attention", "full_attention", "full_attention"],
    ));
    let options = CaptureOptions {
        layers: vec![0, 2],
        max_rows: 5,
    };
    for backend in backends() {
        for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
            for width in [1, 3, 9] {
                let mut reference = fixture.reference(mode);
                let empty_reference = reference.snapshot().unwrap();
                reference.set_capture(options.clone()).unwrap();
                let mut engine = fixture.engine_width(backend, mode, width);
                let empty_engine = engine.snapshot().unwrap();
                engine.set_capture(options.clone()).unwrap();
                assert!(engine.restore(&empty_engine).is_err());
                assert!(reference.restore(&empty_reference).is_err());
                assert_eq!(engine.capture_options(), &options);
                assert_eq!(engine.position(), 0);
                let mut unobserved = fixture.engine_width(backend, mode, width);
                engine
                    .prefill(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                    .unwrap();
                // Uneven reference calls deliberately differ from the engine's grouping.
                for chunk in [&[1, 2][..], &[3, 4, 5, 6, 7], &[8, 9, 10, 11, 12]] {
                    reference.prefill(chunk).unwrap();
                }
                unobserved
                    .prefill(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                    .unwrap();
                assert_eq!(engine.state_digest(), unobserved.state_digest());
                assert_eq!(bits(engine.row_logits()), bits(unobserved.row_logits()));
                assert!(unobserved.captured_rows().is_empty());
                assert_eq!(
                    capture_bits(engine.captured_rows()),
                    capture_bits(reference.captured_rows()),
                    "{backend:?} {mode:?} width={width}"
                );
                assert_eq!(
                    engine
                        .captured_rows()
                        .iter()
                        .map(|row| row.position)
                        .collect::<Vec<_>>(),
                    vec![7, 8, 9, 10, 11]
                );
                assert!(
                    engine
                        .captured_rows()
                        .iter()
                        .all(|row| row.values.len() == 1024)
                );
                assert_eq!(
                    engine.embedding(2).unwrap(),
                    reference.embedding(2).unwrap()
                );
                assert!(engine.embedding(128).is_err());
                assert!(engine.set_capture(options.clone()).is_err());
                let original = capture_bits(engine.captured_rows());
                let tokens = &[13, 14, 15][..width.min(3)];
                engine.run_provisional(tokens).unwrap();
                reference.run_provisional(tokens).unwrap();
                assert_eq!(capture_bits(engine.captured_rows()), original);
                assert_eq!(
                    capture_bits(engine.provisional_captures()),
                    capture_bits(reference.provisional_captures())
                );
                assert!(engine.set_capture(options.clone()).is_err());
                engine.discard_provisional();
                reference.discard_provisional();
                assert!(engine.provisional_captures().is_empty());
                assert_eq!(capture_bits(engine.captured_rows()), original);
                engine.run_provisional(tokens).unwrap();
                reference.run_provisional(tokens).unwrap();
                engine.commit(tokens.len().min(2)).unwrap();
                reference.commit(tokens.len().min(2)).unwrap();
                assert!(engine.provisional_captures().is_empty());
                assert_eq!(
                    capture_bits(engine.captured_rows()),
                    capture_bits(reference.captured_rows())
                );
                assert_eq!(engine.state_digest(), reference.state_digest());
                let snapshot = engine.snapshot().unwrap();
                let ref_snapshot = reference.snapshot().unwrap();
                engine.prefill(&[16, 17, 18, 19, 20, 21]).unwrap();
                reference.prefill(&[16, 17, 18, 19, 20, 21]).unwrap();
                engine.restore(&snapshot).unwrap();
                reference.restore(&ref_snapshot).unwrap();
                assert_eq!(
                    capture_bits(engine.captured_rows()),
                    capture_bits(reference.captured_rows())
                );
                engine.prefill(&[22, 23]).unwrap();
                reference.prefill(&[22, 23]).unwrap();
                assert_eq!(
                    capture_bits(engine.captured_rows()),
                    capture_bits(reference.captured_rows())
                );
                assert_eq!(engine.state_digest(), reference.state_digest());
                engine.reset().unwrap();
                reference.reset().unwrap();
                assert!(engine.captured_rows().is_empty());
                assert!(engine.provisional_captures().is_empty());
                assert_eq!(engine.capture_options(), &options);
                engine.prefill(&[24]).unwrap();
                reference.prefill(&[24]).unwrap();
                assert_eq!(
                    capture_bits(engine.captured_rows()),
                    capture_bits(reference.captured_rows())
                );
            }
        }
    }
}
