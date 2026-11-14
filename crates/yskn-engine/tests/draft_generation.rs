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
                &["linear_attention", "full_attention", "full_attention"],
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
            support::kv_profile::mixed_profile(),
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
    let result = generate(main, proposer, &mut sampler.clone(), prompt, limits).unwrap();
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

fn check_backend(backend: Backend) {
    let fixture = Fixture::new();
    let artifact = draft_artifact();
    let head = support::draft::shead(37);
    let prompt = [3, 7, 11, 9];
    let second_prompt = [3, 7, 11, 13, 5, 17];
    for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
        for seed in [None, Some(17), Some(0xabcdef)] {
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
                let prefix = proposer.snapshot_session(&main).unwrap();
                assert_eq!(
                    engine_decode(&mut main, &mut proposer, &sampler, &prompt, k, 5),
                    first_expected,
                    "{backend:?} {mode:?} seed={seed:?} K={k} first request"
                );
                let checkpoint = proposer.snapshot_session(&main).unwrap();
                let ring = proposer.draft().snapshot();
                assert_eq!(
                    engine_decode(&mut main, &mut proposer, &sampler, &[last], k, 8),
                    continued_expected,
                    "{backend:?} {mode:?} seed={seed:?} K={k} continuation"
                );
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
                // generate flushes every final committed capture into the ring.
                assert_eq!(proposer.draft().position(), main.position());
            }
        }
        eprintln!("neural generation {backend:?} {mode:?}: greedy + 2 seeds, K=0/1/3/7 passed");
    }
}

#[test]
fn cpu_neural_generation_is_lossless_across_joint_restore_and_reused_prefix() {
    check_backend(Backend::Cpu);
}

#[test]
fn metal_neural_generation_is_lossless_across_joint_restore_and_reused_prefix() {
    #[cfg(target_os = "macos")]
    if yskn_engine::MetalEngine::device_exists() {
        check_backend(Backend::Metal);
        return;
    }
    println!("skipped: no Metal device");
}
