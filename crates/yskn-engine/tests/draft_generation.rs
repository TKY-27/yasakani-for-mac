#[path = "../../yskn-ref/tests/support/proposals.rs"]
mod proposals;

#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../yskn-ref/tests/support/mod.rs"]
mod support;

use std::{
    fmt::Debug,
    fs,
    io::Cursor,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use yskn_engine::{Backend, Engine, Kernel, KvMode, Model};
use yskn_format::Container;
use yskn_ref::draft::{Draft, DraftOptions};
use yskn_ref::draft_head::{DraftChoice, ReducedHead};
use yskn_ref::speculation::{
    DraftProposer, GenerationLimits, NoProposer, Proposer, Verifier, generate,
};
use yskn_ref::{Sampler, SamplingParameters};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yskn-neural-generation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(
            path.join("weights.yskn"),
            support::synthetic_model_with_dimensions(
                73,
                &["linear_attention", "full_attention"],
                512,
                512,
                32,
            ),
        )
        .unwrap();
        fs::write(
            path.join("yasakani.json"),
            r#"{"files":{"weights":"weights.yskn"},"max_ctx":64}"#,
        )
        .unwrap();
        fs::write(
            path.join("kvq_u5.bin"),
            support::kv_profile::compact_mixed_profile(),
        )
        .unwrap();
        Self(path)
    }
    fn engine(&self, backend: Backend, mode: KvMode) -> Model {
        // SAFETY: immutable fixture files outlive every mapped model in this test.
        unsafe {
            Model::with_options(
                &self.0,
                Engine::new(Kernel::fastest(), 2).unwrap(),
                backend,
                8,
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

fn draft_artifact() -> Vec<u8> {
    // Keep the codec/payload unchanged; use two capture slots and one draft layer.
    let mut bytes = support::draft::container(23);
    let length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let data = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[32..32 + length]).unwrap();
    header["model"]["layers"] = serde_json::json!(1);
    let fc = &mut header["tensors"][0];
    assert_eq!(fc["name"], "bdraft2.fc");
    fc["cols"] = serde_json::json!(1024);
    fc["rot_in"]["m"] = serde_json::json!(2);
    fc["stream_bytes"] = serde_json::json!(512 * (1024 / 256) * fc["bits"].as_u64().unwrap() * 16);
    let encoded = serde_json::to_vec(&header).unwrap();
    assert!(32 + encoded.len() <= data);
    bytes[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
    bytes[32..data].fill(0);
    bytes[32..32 + encoded.len()].copy_from_slice(&encoded);
    bytes
}

fn proposer(artifact: &[u8], head: &[u8], k: usize, choice: DraftChoice) -> DraftProposer {
    let draft = Draft::from_container(
        &mut Container::from_bytes(artifact.to_vec()).unwrap(),
        ReducedHead::read(&mut Cursor::new(head)).unwrap(),
        DraftOptions {
            capture_layers: vec![0, 1],
            ..support::draft::options()
        },
    )
    .unwrap();
    DraftProposer::new(draft, k, choice).unwrap()
}

fn decode<V: Verifier>(
    main: &mut V,
    proposer: &mut impl Proposer,
    sampler: &Sampler,
    prompt: &[u32],
    k: usize,
    count: usize,
) -> Vec<u32>
where
    V::Error: Debug,
{
    let mut limits = GenerationLimits::new(count, main.context_capacity());
    limits.max_draft_tokens = k;
    let mut observed = proposals::Observed::new(proposer);
    let result = generate(main, &mut observed, &mut sampler.clone(), prompt, limits).unwrap();
    if k >= 3 && count == 5 {
        assert!(
            observed.max_offered > 1,
            "request must offer multiple guesses in one pass"
        );
    }
    assert_eq!(result.tokens.len(), count);
    if k != 0 {
        assert!(result.statistics.guesses_offered > 0);
    }
    result.tokens
}

fn engine_decode(
    main: &mut Model,
    proposer: &mut impl Proposer,
    sampler: &Sampler,
    prompt: &[u32],
    k: usize,
    count: usize,
) -> (Vec<u32>, u64) {
    let tokens = decode(main, proposer, sampler, prompt, k, count);
    assert!(!main.has_provisional_pass());
    (tokens, main.state_digest())
}

fn check_backend(backend: Backend, mode: KvMode, seed: Option<u64>) {
    let fixture = Fixture::new();
    let artifact = draft_artifact();
    let head = support::draft::shead(37);
    let prompt = [3, 7, 11, 9];
    let second_prompt = [3, 7, 11, 13, 5, 17];

    let sampler = seed.map_or(Sampler::Greedy, |seed| {
        Sampler::Sampled(SamplingParameters {
            seed,
            ..SamplingParameters::default()
        })
    });
    let choice = seed.map_or(DraftChoice::Greedy, DraftChoice::sampled);
    let mut reference = fixture.reference(mode);
    reference.prefill(&prompt[..3]).unwrap();
    let first_tokens = decode(&mut reference, &mut NoProposer, &sampler, &prompt, 0, 5);
    let last = *first_tokens.last().unwrap();
    let first_expected = (first_tokens, reference.state_digest());
    let continued_tokens = decode(&mut reference, &mut NoProposer, &sampler, &[last], 0, 8);
    let continued_expected = (continued_tokens, reference.state_digest());
    let mut fresh_reference = fixture.reference(mode);
    fresh_reference
        .prefill(&second_prompt[..second_prompt.len() - 1])
        .unwrap();
    let second_tokens = decode(
        &mut fresh_reference,
        &mut NoProposer,
        &sampler,
        &second_prompt,
        0,
        5,
    );
    let second_expected = (second_tokens, fresh_reference.state_digest());

    // Establish engine K=0 against the scalar baseline on this backend/KV mode.
    let mut sequential = fixture.engine(backend, mode);
    sequential.prefill(&prompt[..3]).unwrap();
    let prefix = sequential.snapshot().unwrap();
    assert_eq!(
        engine_decode(&mut sequential, &mut NoProposer, &sampler, &prompt, 0, 5),
        first_expected
    );
    assert_eq!(
        engine_decode(&mut sequential, &mut NoProposer, &sampler, &[last], 0, 8),
        continued_expected
    );
    sequential.restore(&prefix).unwrap();
    sequential
        .prefill(&second_prompt[3..second_prompt.len() - 1])
        .unwrap();
    assert_eq!(
        engine_decode(
            &mut sequential,
            &mut NoProposer,
            &sampler,
            &second_prompt,
            0,
            5,
        ),
        second_expected
    );

    for k in [1, 3, 7] {
        let mut proposer = proposer(&artifact, &head, k, choice);
        let mut main = fixture.engine(backend, mode);
        main.set_capture(proposer.capture_options()).unwrap();
        main.prefill(&prompt[..3]).unwrap();
        let prefix = proposer.snapshot_session(&mut main).unwrap();
        assert_eq!(
            engine_decode(&mut main, &mut proposer, &sampler, &prompt, k, 5),
            first_expected,
            "{backend:?} {mode:?} seed={seed:?} K={k} first request"
        );
        let checkpoint = proposer.snapshot_session(&mut main).unwrap();
        let ring = proposer.draft().snapshot();
        assert_eq!(main.position(), 8);
        assert_eq!(ring.position(), proposer.draft().dimensions().window);
        assert_eq!(
            engine_decode(&mut main, &mut proposer, &sampler, &[last], k, 8),
            continued_expected,
            "{backend:?} {mode:?} seed={seed:?} K={k} continuation"
        );
        assert!(
            main.position() > proposer.draft().dimensions().window,
            "continuation must wrap the ring"
        );
        assert_eq!(main.position(), 16);
        proposer.restore_session(&mut main, &checkpoint).unwrap();
        assert_eq!(proposer.draft().snapshot(), ring);
        assert_eq!(
            engine_decode(&mut main, &mut proposer, &sampler, &[last], k, 8),
            continued_expected,
            "{backend:?} {mode:?} seed={seed:?} K={k} restored continuation"
        );
        proposer.restore_session(&mut main, &prefix).unwrap();
        main.prefill(&second_prompt[3..second_prompt.len() - 1])
            .unwrap();
        assert_eq!(
            engine_decode(&mut main, &mut proposer, &sampler, &second_prompt, k, 5),
            second_expected,
            "{backend:?} {mode:?} seed={seed:?} K={k} reused prefix"
        );
        assert_eq!(main.position(), 10);
        assert!(
            proposer.draft().position() > proposer.draft().dimensions().window,
            "reused prefix must wrap the ring"
        );
        assert_eq!(proposer.draft().position(), main.position());
    }
}

macro_rules! generation_case {
    ($name:ident, $mode:expr, $seed:expr) => {
        #[test]
        fn $name() {
            check_backend(Backend::Cpu, $mode, $seed);
            #[cfg(target_os = "macos")]
            if yskn_engine::MetalEngine::device_exists() {
                check_backend(Backend::Metal, $mode, $seed);
                return;
            }
            println!("skipped: no Metal device");
        }
    };
}
generation_case!(
    neural_int8_greedy_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Int8,
    None
);
generation_case!(
    neural_int8_seed17_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Int8,
    Some(17)
);
generation_case!(
    neural_int8_seedabcdef_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Int8,
    Some(0xabcdef)
);
generation_case!(
    neural_int8_profile_greedy_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Int8Profile,
    None
);
generation_case!(
    neural_int8_profile_seed17_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Int8Profile,
    Some(17)
);
generation_case!(
    neural_int8_profile_seedabcdef_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Int8Profile,
    Some(0xabcdef)
);
generation_case!(
    neural_kv5_greedy_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Kv5,
    None
);
generation_case!(
    neural_kv5_seed17_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Kv5,
    Some(17)
);
generation_case!(
    neural_kv5_seedabcdef_is_lossless_across_joint_restore_and_reused_prefix,
    KvMode::Kv5,
    Some(0xabcdef)
);

#[test]
fn engine_handover_counts_memory_reseeds_long_suffix_and_detects_main_only_restore() {
    let fixture = Fixture::new();
    let artifact = draft_artifact();
    let head = support::draft::shead(37);
    let mut main = fixture.engine(Backend::Cpu, KvMode::Int8);
    let mut reference = fixture.reference(KvMode::Int8);
    let mut neural = proposer(&artifact, &head, 3, DraftChoice::Greedy);
    reference.set_capture(neural.capture_options()).unwrap();
    main.set_capture(neural.capture_options()).unwrap();
    main.prefill(&[3; 12]).unwrap();
    reference.prefill(&[3; 12]).unwrap();
    assert_eq!(main.captured_rows(), reference.captured_rows());
    assert!(main.session_memory().capture_bytes >= 8 * 2 * 512 * 4);
    assert!(main.snapshot().unwrap().capture_bytes() >= 8 * 2 * 512 * 4);
    let prefix = neural.snapshot_session(&mut main).unwrap();
    assert_eq!(main.session_memory().capture_bytes, 0);
    assert_eq!(main.snapshot().unwrap().capture_bytes(), 0);
    main.prefill(&[7; 17]).unwrap();
    reference.prefill(&[7; 17]).unwrap();
    assert_eq!(main.captured_rows(), reference.captured_rows());
    neural.synchronize(&mut main).unwrap();
    assert_eq!(neural.draft().position(), 29);
    assert_eq!(main.session_memory().capture_bytes, 0);
    neural.restore_session(&mut main, &prefix).unwrap();
    assert_eq!(main.snapshot().unwrap().capture_bytes(), 0);
    let main_only = main.snapshot().unwrap();
    main.prefill(&[11; 3]).unwrap();
    neural.synchronize(&mut main).unwrap();
    main.restore(&main_only).unwrap();
    main.prefill(&[13; 5]).unwrap();
    assert!(neural.synchronize(&mut main).is_err());
    neural.reset();
    main.reset().unwrap();
    main.prefill(&[3; 3]).unwrap();
    neural.synchronize(&mut main).unwrap();
    main.reset().unwrap();
    main.prefill(&[7; 5]).unwrap();
    assert!(neural.synchronize(&mut main).is_err());
}

/// One generation of 9 tokens after a 3-token prefill; returns tokens, accepted
/// guesses and the main state digest.
fn draft_run<D: yskn_ref::draft::BlockDraft>(
    fixture: &Fixture,
    backend: Backend,
    proposer: &mut DraftProposer<D>,
    sampler: &Sampler,
    prompt: &[u32],
    k: usize,
) -> (Vec<u32>, usize, u64) {
    let mut main = fixture.engine(backend, KvMode::Int8);
    main.set_capture(proposer.capture_options()).unwrap();
    main.prefill(&prompt[..3]).unwrap();
    let mut limits = GenerationLimits::new(9, main.context_capacity());
    limits.max_draft_tokens = k;
    let result = generate(&mut main, proposer, &mut sampler.clone(), prompt, limits).unwrap();
    assert!(result.statistics.guesses_offered > 0);
    assert!(!main.has_provisional_pass());
    let accepted = result.statistics.guesses_accepted;
    (result.tokens, accepted, main.state_digest())
}

/// The fast draft drives the same lossless loop as the reference draft: equal
/// tokens, main state, accepted guesses and final draft ring, and both equal no draft.
fn check_fast_draft(backend: Backend) {
    use yskn_engine::FastDraft;
    let fixture = Fixture::new();
    let artifact = draft_artifact();
    let head = support::draft::shead(37);
    fs::write(fixture.0.join("bdraft2_k8.yskn"), &artifact).unwrap();
    fs::write(fixture.0.join("bdraft2_k8.shead"), &head).unwrap();
    let prompt = [3, 7, 11, 9];
    for seed in [None, Some(17), Some(0xabcdef)] {
        let sampler = seed.map_or(Sampler::Greedy, |seed| {
            Sampler::Sampled(SamplingParameters {
                seed,
                ..SamplingParameters::default()
            })
        });
        let choice = seed.map_or(DraftChoice::Greedy, DraftChoice::sampled);
        let mut main = fixture.engine(backend, KvMode::Int8);
        main.prefill(&prompt[..3]).unwrap();
        let limits = GenerationLimits::new(9, main.context_capacity());
        let none = generate(
            &mut main,
            &mut NoProposer,
            &mut sampler.clone(),
            &prompt,
            limits,
        )
        .unwrap()
        .tokens;
        let none_digest = main.state_digest();
        for k in [1, 3, 7] {
            let mut reference = proposer(&artifact, &head, k, choice);
            let expected = draft_run(&fixture, backend, &mut reference, &sampler, &prompt, k);
            assert_eq!(
                (&expected.0, expected.2),
                (&none, none_digest),
                "reference K={k}"
            );
            // SAFETY: the fixture files are not modified while the draft is alive.
            let draft = unsafe {
                FastDraft::with_options(
                    &fixture.0,
                    Engine::new(Kernel::fastest(), 3).unwrap(),
                    backend,
                    DraftOptions {
                        capture_layers: vec![0, 1],
                        ..support::draft::options()
                    },
                )
            }
            .unwrap();
            let mut fast = DraftProposer::new(draft, k, choice).unwrap();
            let actual = draft_run(&fixture, backend, &mut fast, &sampler, &prompt, k);
            assert_eq!(actual, expected, "{backend:?} seed={seed:?} K={k}");
            assert_eq!(fast.draft().snapshot(), reference.draft().snapshot());
        }
    }
    eprintln!(
        "fast draft generation {backend:?}: greedy + 2 seeds, K=1/3/7 equal reference and no draft"
    );
}

#[test]
fn cpu_fast_draft_generation_equals_reference_draft_and_no_draft() {
    check_fast_draft(Backend::Cpu);
}

#[test]
fn metal_fast_draft_generation_equals_reference_draft_and_no_draft() {
    #[cfg(target_os = "macos")]
    if yskn_engine::MetalEngine::device_exists() {
        check_fast_draft(Backend::Metal);
        return;
    }
    println!("skipped: no Metal device");
}
