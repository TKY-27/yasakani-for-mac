mod product_support;

#[global_allocator]
static ALLOCATOR: product_support::allocation::Counting = product_support::allocation::Counting;

use product_support::{Folder, greedy, sampled, scalar};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::process::Command;
use yskn::{
    engine::{Engine, Finish, Generation, Sampling},
    real_engine::DraftMode,
};

fn modes() -> [DraftMode; 4] {
    [
        DraftMode::None,
        DraftMode::Ngram,
        DraftMode::Neural,
        DraftMode::Both,
    ]
}
fn run(engine: &mut dyn Engine, prompt: &[u32], n: usize, sampling: &Sampling) -> Generation {
    engine
        .generate(prompt, n, sampling, None, &mut |_| Ok(true))
        .unwrap()
}

#[test]
fn real_product_matches_independent_scalar_loop_for_greedy_and_seeded_sampling_in_every_draft_mode()
{
    let folder = Folder::new();
    let prompt = [1, 2, 1, 2, 1, 2, 3];
    for sampling in [greedy(), sampled()] {
        let expected = scalar(&folder, &prompt, 7, &sampling);
        for mode in modes() {
            let mut engine = folder.engine(mode);
            let result = run(&mut engine, &prompt, 7, &sampling);
            assert_eq!(
                result.ids, expected,
                "draft={mode:?}, sampling={sampling:?}"
            );
            assert_eq!(result.finish, Finish::Length);
            assert_eq!(result.cached_tokens, 0);
            assert!(result.passes > 0);
            assert!(result.rows_verified >= result.passes);
            assert!(result.guesses_accepted <= result.guesses_offered);
        }
    }
}

#[test]
fn committed_prefix_snapshot_and_reset_reuse_preserve_scalar_tokens_and_cache_counts() {
    let folder = Folder::new();
    let first = [1, 2, 3, 4];
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let initial = run(&mut engine, &first, 2, &greedy());
        let mut continued = first.to_vec();
        continued.extend(&initial.ids);
        continued.push(5);
        let result = run(&mut engine, &continued, 2, &greedy());
        assert_eq!(
            result.ids,
            scalar(&folder, &continued, 2, &greedy()),
            "{mode:?}"
        );
        // The final returned token is pending; every earlier input is committed.
        assert_eq!(
            result.cached_tokens,
            first.len() + initial.ids.len() - 1,
            "{mode:?}"
        );
        assert_eq!(result.snap_mode, 1);
        // The new suffix differs after the snapshot saved before the final
        // prompt token; it cannot reuse the complete committed history.
        let mut branch = continued[..continued.len() - 1].to_vec();
        branch.push(6);
        let result = run(&mut engine, &branch, 2, &greedy());
        assert_eq!(
            result.ids,
            scalar(&folder, &branch, 2, &greedy()),
            "{mode:?}"
        );
        assert_eq!(result.cached_tokens, continued.len() - 1, "{mode:?}");
        assert_eq!(result.snap_mode, 2);
        let unrelated = [7, 8, 9];
        let result = run(&mut engine, &unrelated, 2, &greedy());
        assert_eq!(
            result.ids,
            scalar(&folder, &unrelated, 2, &greedy()),
            "{mode:?}"
        );
        assert_eq!((result.cached_tokens, result.snap_mode), (0, 0));
    }
}

#[test]
fn repeated_snapshot_replacement_retains_bounded_live_allocations_for_every_draft_mode() {
    const CHILD: &str = "YSKN_SNAPSHOT_ALLOCATION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Isolate process-wide allocation levels from parallel tests and their
        // worker threads. The allocator is installed in this test binary only.
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "repeated_snapshot_replacement_retains_bounded_live_allocations_for_every_draft_mode",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let folder = Folder::new();
    // Two unrelated prefixes, each with 27 growing branches. Every shared
    // request restores the previous prefix and then replaces its checkpoint;
    // a leak cleared only by reset must also be caught before the next reset.
    let prompts: Vec<Vec<u32>> = [1, 5]
        .into_iter()
        .flat_map(|prefix| {
            (4..31).map(move |len| {
                let mut prompt = vec![prefix; len];
                prompt[len - 1] = prefix + 1 + (len % 2) as u32;
                prompt
            })
        })
        .collect();
    let expected: Vec<_> = prompts
        .iter()
        .map(|p| scalar(&folder, p, 2, &greedy()))
        .collect();
    for mode in modes() {
        let mut engine = folder.engine(mode);
        // Warm scratch/KV/draft capacities beyond the largest measured prompt
        // while retaining just one checkpoint, rather than warming a leak into
        // the baseline by repeatedly replacing checkpoints first.
        for _ in 0..3 {
            drop(run(&mut engine, &[9; 96], 2, &greedy()));
        }
        let warmed = product_support::allocation::live_bytes();
        for (index, prompt) in prompts.iter().enumerate() {
            let result = run(&mut engine, prompt, 2, &greedy());
            assert_eq!(result.ids, expected[index], "{mode:?}, prompt {index}");
            if index % 27 == 0 {
                assert_eq!((result.cached_tokens, result.snap_mode), (0, 0));
            } else {
                assert_eq!(
                    (result.cached_tokens, result.snap_mode),
                    (prompts[index - 1].len() - 1, 2)
                );
            }
            drop(result);
            let live = product_support::allocation::live_bytes();
            // Small allocator/runtime bookkeeping is tolerated; even one
            // retained recurrent checkpoint exceeds this allowance.
            assert!(
                live <= warmed + 16 * 1024,
                "{mode:?}, prompt {index}: live {live}, warmed {warmed}"
            );
        }
    }
}

#[test]
fn cancellation_and_callback_errors_quiesce_prefill_and_generation_then_allow_next_request() {
    let folder = Folder::new();
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let mut empty_callbacks = 0;
        let result = engine
            .generate(&[1; 12], 8, &greedy(), None, &mut |ids| {
                assert!(ids.is_empty(), "cancellation must happen during prefill");
                empty_callbacks += 1;
                Ok(empty_callbacks < 2)
            })
            .unwrap();
        assert_eq!(result.finish, Finish::Cancelled);
        assert!(result.ids.is_empty());
        assert_eq!(result.passes, 0);
        assert_eq!(empty_callbacks, 2);
        assert_eq!(
            run(&mut engine, &[2, 3], 2, &greedy()).ids,
            scalar(&folder, &[2, 3], 2, &greedy())
        );
        engine.reset().unwrap();
        let result = engine
            .generate(&[1, 2], 8, &greedy(), None, &mut |ids| Ok(ids.is_empty()))
            .unwrap();
        assert_eq!(result.finish, Finish::Cancelled);
        assert_eq!(result.ids.len(), 1);
        assert_eq!(
            run(&mut engine, &[2, 3], 2, &greedy()).ids,
            scalar(&folder, &[2, 3], 2, &greedy())
        );
        engine.reset().unwrap();
        let failure = engine
            .generate(&[1, 2], 8, &greedy(), None, &mut |ids| {
                if ids.is_empty() {
                    Ok(true)
                } else {
                    Err(yskn::Error::engine("callback-sentinel"))
                }
            })
            .unwrap_err();
        assert!(failure.to_string().contains("callback-sentinel"));
        assert_eq!(
            run(&mut engine, &[2, 3], 2, &greedy()).ids,
            scalar(&folder, &[2, 3], 2, &greedy())
        );
    }
}

#[test]
fn interrupted_prefill_reuses_shared_prefix_after_cancellation_or_callback_error() {
    let folder = Folder::new();
    let prompt = [1; 12];
    let resumed = [vec![1; 8], vec![2, 3]].concat();
    let expected = scalar(&folder, &resumed, 4, &greedy());
    for mode in modes() {
        for callback_error in [false, true] {
            let mut engine = folder.engine(mode);
            let mut polls = 0;
            let interrupted = engine.generate(&prompt, 4, &greedy(), None, &mut |ids| {
                assert!(ids.is_empty());
                polls += 1;
                if polls < 2 {
                    Ok(true)
                } else if callback_error {
                    Err(yskn::Error::engine("prefill-callback-sentinel"))
                } else {
                    Ok(false)
                }
            });
            if callback_error {
                assert!(
                    interrupted
                        .unwrap_err()
                        .to_string()
                        .contains("prefill-callback-sentinel")
                );
            } else {
                let result = interrupted.unwrap();
                assert_eq!(result.finish, Finish::Cancelled);
                assert!(result.ids.is_empty());
                assert_eq!(result.passes, 0);
            }
            assert_eq!(polls, 2);
            let result = run(&mut engine, &resumed, 4, &greedy());
            // Width two committed exactly two rows before the interruption.
            // Require live reuse, so an unrelated/reset path cannot mask damage.
            assert_eq!(
                (result.cached_tokens, result.snap_mode),
                (2, 1),
                "{mode:?}, error={callback_error}"
            );
            assert_eq!(result.ids, expected, "{mode:?}, error={callback_error}");
        }
    }
}

#[test]
fn callback_error_during_generation_preserves_shared_prefix_for_next_request() {
    let folder = Folder::new();
    let resumed = [1, 2, 3, 4];
    let expected = scalar(&folder, &resumed, 4, &greedy());
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let error = engine
            .generate(&[1, 2], 8, &greedy(), None, &mut |ids| {
                if ids.is_empty() {
                    Ok(true)
                } else {
                    Err(yskn::Error::engine("generation-callback-sentinel"))
                }
            })
            .unwrap_err();
        assert!(error.to_string().contains("generation-callback-sentinel"));
        let result = run(&mut engine, &resumed, 4, &greedy());
        assert_eq!((result.cached_tokens, result.snap_mode), (2, 1), "{mode:?}");
        assert_eq!(result.ids, expected, "{mode:?}");
    }
}

#[test]
fn prefix_and_snapshot_reuse_apply_each_requests_seed_and_sampling_settings() {
    let folder = Folder::new();
    let first = [1, 2, 3, 4];
    let settings = [
        sampled(),
        Sampling {
            temperature: 1.2345,
            top_p: 0.8724,
            top_k: 7,
            min_p: 0.0314,
            seed: 19,
        },
        Sampling {
            temperature: 0.4174,
            top_p: 0.7734,
            top_k: 3,
            min_p: 0.1424,
            seed: 1234,
        },
    ];
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let initial = run(&mut engine, &first, 4, &settings[0]);
        assert_eq!(initial.ids, scalar(&folder, &first, 4, &settings[0]));
        let continued = [first.to_vec(), initial.ids.clone(), vec![5]].concat();
        let result = run(&mut engine, &continued, 4, &settings[1]);
        assert_eq!(
            (result.cached_tokens, result.snap_mode),
            (first.len() + initial.ids.len() - 1, 1),
            "{mode:?}"
        );
        assert_eq!(
            result.ids,
            scalar(&folder, &continued, 4, &settings[1]),
            "{mode:?}"
        );
        let mut branch = continued[..continued.len() - 1].to_vec();
        branch.push(6);
        // Alternate settings repeatedly on the same saved prefix, including
        // returning to the original seed after different sampled requests.
        for sampling in [&settings[2], &settings[0], &settings[1]] {
            let result = run(&mut engine, &branch, 4, sampling);
            assert_eq!(
                (result.cached_tokens, result.snap_mode),
                (branch.len() - 1, 2),
                "{mode:?}, {sampling:?}"
            );
            assert_eq!(
                result.ids,
                scalar(&folder, &branch, 4, sampling),
                "{mode:?}, {sampling:?}"
            );
        }
    }
}

#[test]
fn context_boundary_reserves_margin_clamps_output_and_rejects_prompts_without_room() {
    let folder = Folder::new();
    const CONTEXT: usize = 80;
    // Engine's documented 64-row margin and the unconsumed final prediction
    // (D19 / spec 08): at most 16 committed rows plus one pending output.
    const SEQUENCE_LIMIT: usize = CONTEXT - 64 + 1;
    let prompt = [1, 2, 3, 4, 1, 2, 3, 4];
    for mode in modes() {
        let mut engine = folder.engine_with_context(mode, CONTEXT);
        assert_eq!(engine.context_capacity(), CONTEXT);
        // Check both the usable sequence boundary and the raw configured
        // context. Raw-context requests must clamp to preserve the margin.
        for requested_total in [
            SEQUENCE_LIMIT - 1,
            SEQUENCE_LIMIT,
            SEQUENCE_LIMIT + 1,
            CONTEXT - 1,
            CONTEXT,
            CONTEXT + 1,
        ] {
            let requested = requested_total - prompt.len();
            let count = requested.min(SEQUENCE_LIMIT - prompt.len());
            let expected = scalar(&folder, &prompt, count, &greedy());
            let result = run(&mut engine, &prompt, requested, &greedy());
            assert_eq!(
                result.ids.len(),
                count,
                "{mode:?}, requested total {requested_total}"
            );
            assert_eq!(
                result.ids, expected,
                "{mode:?}, requested total {requested_total}"
            );
            assert_eq!(result.finish, Finish::Length);
        }
        let last_valid = vec![1; CONTEXT - 64 - 1];
        let result = run(&mut engine, &last_valid, 3, &greedy());
        assert_eq!(
            result.ids,
            scalar(&folder, &last_valid, 2, &greedy()),
            "{mode:?}"
        );
        for len in [CONTEXT - 64, CONTEXT - 64 + 1, CONTEXT, CONTEXT + 1] {
            let mut called = false;
            let error = engine
                .generate(&vec![1; len], 1, &greedy(), None, &mut |_| {
                    called = true;
                    Ok(true)
                })
                .unwrap_err();
            assert!(
                error.to_string().contains("prompt leaves no room"),
                "{mode:?}, len={len}: {error}"
            );
            assert!(!called, "oversize prompt reached engine callback");
        }
    }
}

#[test]
fn stop_id_is_included_and_stops_at_the_first_matching_prediction() {
    let folder = Folder::new();
    let expected = scalar(&folder, &[1, 2], 3, &greedy());
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let result = engine
            .generate(&[1, 2], 8, &greedy(), Some(&[expected[0]]), &mut |_| {
                Ok(true)
            })
            .unwrap();
        assert_eq!(result.ids, expected[..1]);
        assert_eq!(result.finish, Finish::Stop);
    }
}

fn fast_draft_engine(
    folder: &Folder,
    mode: DraftMode,
    depth: yskn::real_engine::DepthMode,
    k: usize,
) -> yskn::real_engine::RealEngine {
    let mut config = folder.config(mode);
    config.depth = depth;
    config.k = k;
    let mut options = product_support::model::draft::options();
    options.capture_layers = vec![0];
    yskn::real_engine::RealEngine::load_with_draft_options(config, options).unwrap()
}

#[test]
fn fast_draft_output_equals_no_draft_output_for_adaptive_and_fixed_depths() {
    use yskn::real_engine::DepthMode;
    let folder = Folder::new();
    let prompt = [1, 2, 1, 2, 1, 2, 3];
    for sampling in [greedy(), sampled()] {
        let expected = run(&mut folder.engine(DraftMode::None), &prompt, 24, &sampling);
        assert_eq!(expected.ids, scalar(&folder, &prompt, 24, &sampling));
        assert_eq!((expected.draft_proposals, expected.draft_updates), (0, 0));
        for mode in [DraftMode::Neural, DraftMode::Both] {
            for depth in [DepthMode::Adaptive, DepthMode::Fixed] {
                for k in [1, 3, 7] {
                    let mut engine = fast_draft_engine(&folder, mode, depth, k);
                    let first = run(&mut engine, &prompt, 24, &sampling);
                    let case = format!("{mode:?} {depth:?} k={k} {sampling:?}");
                    assert_eq!(first.ids, expected.ids, "{case}");
                    engine.reset().unwrap();
                    let again = run(&mut engine, &prompt, 24, &sampling);
                    assert_eq!(again.ids, expected.ids, "{case}");
                    assert_eq!(
                        (again.passes, again.rows_verified, again.draft_proposals),
                        (first.passes, first.rows_verified, first.draft_proposals),
                        "{case}: depths must repeat"
                    );
                    assert!(first.draft_updates > 0, "{case}");
                    if depth == DepthMode::Fixed && mode == DraftMode::Neural {
                        assert!(first.draft_proposals > 0, "{case}");
                        assert!(first.guesses_offered > 0, "{case}");
                    }
                }
            }
        }
    }
}

#[test]
fn product_forwards_free_ngram_copy_in_full_before_adaptive_neural_depth() {
    let folder = Folder::new();
    let mut engine = fast_draft_engine(
        &folder,
        DraftMode::Both,
        yskn::real_engine::DepthMode::Adaptive,
        7,
    );
    let result = engine
        .generate(
            &[1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
            8,
            &greedy(),
            None,
            &mut |ids| Ok(ids.is_empty()),
        )
        .unwrap();
    assert_eq!(result.finish, Finish::Cancelled);
    assert_eq!(result.ids.len(), 1);
    assert_eq!(result.passes, 1);
    assert_eq!(result.rows_verified, 8);
    assert_eq!(result.guesses_offered, 7);
    assert_eq!(result.draft_proposals, 0);
}

#[test]
fn draft_work_counts_history_updates_even_without_a_proposal() {
    use yskn::real_engine::DepthMode;
    let folder = Folder::new();
    for mode in [DraftMode::Neural, DraftMode::Both] {
        let mut engine = fast_draft_engine(&folder, mode, DepthMode::Fixed, 3);
        assert!(engine.info().draft_proposal_code_bytes > 0);
        assert!(engine.info().draft_update_code_bytes > 0);
        // One token: the only pass has no room for a guess, so no proposal
        // runs, but committing that pass still updates the draft history.
        let one = run(&mut engine, &[1, 2, 3, 4], 1, &greedy());
        assert_eq!((one.ids.len(), one.passes), (1, 1));
        assert_eq!(one.draft_proposals, 0, "{mode:?}");
        assert_eq!(one.draft_updates, 1, "{mode:?}");
        engine.reset().unwrap();
        let more = run(&mut engine, &[7, 8, 9], 6, &greedy());
        // Every pass commits one run of rows; the pass that ends the request
        // still updates the history.
        assert_eq!(more.draft_updates, more.passes, "{mode:?}");
        if mode == DraftMode::Neural {
            assert_eq!(more.draft_proposals, more.passes - 1, "{mode:?}");
        }
    }
    for mode in [DraftMode::None, DraftMode::Ngram] {
        let mut engine = folder.engine(mode);
        let result = run(&mut engine, &[1, 2, 3, 4], 6, &greedy());
        assert_eq!((result.draft_proposals, result.draft_updates), (0, 0));
        assert_eq!(engine.info().draft_proposal_code_bytes, 0);
    }
}

#[test]
fn synthetic_fast_draft_memory_counts_owned_vectors_and_mapped_selectors_once() {
    let folder = Folder::new();
    let memory = folder.config(DraftMode::Neural).inspect().unwrap().memory;
    let mapped = std::fs::metadata(folder.0.join("draft.yskn"))
        .unwrap()
        .len()
        + std::fs::metadata(folder.0.join("draft.shead"))
            .unwrap()
            .len();
    // h=512, layers=2, correction rank=32, V=32, head rank=512.
    // Two global norms, six vectors/layer, hp; pred/succ have no f32 owner.
    let vectors = 4 * (2 * 512 + 2 * (10 * 512 + 2 * 128) + 32 * 512);
    let head = 4 * 512 * 512 + 32 * 512 + 2 * 32 * 4;
    assert_eq!(memory.draft_bytes, mapped + vectors + head);
    let subtotal = memory.weights_bytes
        + memory.draft_bytes
        + memory.recurrent_bytes
        + memory.cache_bytes
        + memory.snapshot_bytes
        + memory.working_bytes;
    assert_eq!(
        memory.margin_bytes,
        (subtotal * 15).div_ceil(100) + 1_073_741_824
    );
    assert_eq!(memory.needed_bytes, subtotal + memory.margin_bytes);
    assert!(memory.enforce(memory.needed_bytes - 1).is_err());
}

#[test]
fn memory_check_reports_needed_available_and_configuration_remedies() {
    let folder = Folder::new();
    let info = folder.config(DraftMode::Both).inspect().unwrap();
    let error = info.memory.enforce(1).unwrap_err().to_string();
    for expected in [
        "need",
        "physical memory is 1 bytes",
        "--ctx",
        "--kv",
        "--draft",
    ] {
        assert!(error.contains(expected), "missing {expected:?}: {error}");
    }
    assert!(
        error.contains(&info.memory.needed_bytes.to_string()),
        "{error}"
    );
}

#[test]
fn requested_context_cannot_exceed_unrounded_model_limit() {
    let folder = Folder::new();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(folder.0.join("yasakani.json")).unwrap()).unwrap();
    manifest["max_ctx"] = json!(97);
    folder.json("yasakani.json", manifest);
    let mut config = folder.config(DraftMode::None);
    config.context = 100;
    let error = config.inspect().unwrap_err().to_string();
    assert!(
        error.contains("100") && error.contains("97") && error.contains("max_ctx"),
        "{error}"
    );
}

#[test]
fn model_generation_defaults_reach_request_sampling_without_overriding_explicit_values() {
    let folder = Folder::new();
    let sampling = folder.config(DraftMode::None).sampling_defaults().unwrap();
    assert_eq!(
        sampling,
        Sampling {
            temperature: 0.75,
            top_p: 0.9,
            top_k: 11,
            min_p: 0.05,
            seed: 0
        }
    );
    let config = yskn::server::Config {
        defaults: yskn::request::RequestDefaults {
            sampling,
            ..Default::default()
        },
        ..Default::default()
    };
    let messages = json!([{"role":"user","content":"t1"}]);
    let defaults =
        yskn::request::parse_chat_request(&json!({"messages":messages}), &config.defaults).unwrap();
    assert_eq!(
        (
            defaults.temperature,
            defaults.top_p,
            defaults.top_k,
            defaults.min_p,
            defaults.seed
        ),
        (0.75, 0.9, 11, 0.05, None)
    );
    let explicit = yskn::request::parse_chat_request(
        &json!({"messages":messages,"temperature":0.5,"top_p":0.8,"top_k":3,"min_p":0.1,"seed":99}),
        &config.defaults,
    )
    .unwrap();
    assert_eq!(
        (
            explicit.temperature,
            explicit.top_p,
            explicit.top_k,
            explicit.min_p,
            explicit.seed
        ),
        (0.5, 0.8, 3, 0.1, Some(99))
    );
}

#[test]
fn synthetic_tokenizer_streams_and_decodes_every_model_id_the_same_on_every_load() {
    // The socket tests compare server text with a separately loaded decoder;
    // that holds only if the tokenizer, like the real one, is deterministic.
    let folder = Folder::new();
    let ids: Vec<u32> = (0..32).collect();
    let first = yskn_text::ChatFormatter::load(&folder.0)
        .unwrap()
        .decode(&ids)
        .unwrap();
    for _ in 0..4 {
        let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
        assert_eq!(formatter.decode(&ids).unwrap(), first);
        let mut stream = yskn_text::OutputStream::new(&formatter, false);
        let mut text = String::new();
        for n in 1..=ids.len() {
            for (kind, piece) in stream.feed(&ids[..n], n == ids.len()).unwrap() {
                assert_eq!(kind, "content");
                text += &piece;
            }
        }
        assert_eq!(text, first);
    }
}

fn digest(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in ids {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn selfcheck_reference(folder: &Folder) -> Value {
    let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
    let prompts = [
        ("short-answer",formatter.prompt(&json!([{"role":"user","content":"Say hello in one short sentence."}]),None,&json!({"enable_thinking":false})).unwrap().0,48),
        ("thinking",formatter.prompt(&json!([{"role":"user","content":"What is 17 times 23? Think it through step by step."}]),None,&json!({})).unwrap().0,160),
        ("plain-text",formatter.encode("The capital of France is").unwrap(),32),
    ];
    let cases: Vec<_> = prompts.into_iter().map(|(name,prompt,max_new)| {
        let ids = scalar(folder,&prompt,max_new,&greedy());
        json!({"name":name,"prompt_tokens":prompt.len(),"prompt_sha256":digest(&prompt),"tokens":ids.len(),"sha256":digest(&ids),"head":&ids[..ids.len().min(12)],"ids":ids})
    }).collect();
    json!({"format":1,"cases":cases})
}
fn selfcheck(folder: &Folder) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_yskn"))
        .args(["selfcheck", "--model-dir"])
        .arg(&folder.0)
        .args(["--backend", "cpu", "--ctx", "512", "--draft", "none"])
        .env_remove("YASAKANI_API_KEY")
        .env_remove("YASAKANI_MODEL_DIR")
        .output()
        .unwrap()
}

fn product_command(folder: &Folder, subcommand: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_yskn"));
    command
        .args([subcommand, "--model-dir"])
        .arg(&folder.0)
        .args(["--backend", "cpu", "--ctx", "512", "--threads", "1"]);
    for key in [
        "YASAKANI_API_KEY",
        "YASAKANI_MODEL_DIR",
        "YASAKANI_THINKING_BUDGET",
    ] {
        command.env_remove(key);
    }
    command
}

#[test]
fn neural_benchmark_json_counts_one_history_update_without_a_proposal() {
    let folder = Folder::new();
    folder.neural_bench_fixture();
    let path = folder.0.join("neural-bench.json");
    let output = product_command(&folder, "bench")
        .args(["--draft", "neural", "--n", "1", "--json"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(report["tokens_per_task"], 1);
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 4);
    for run in runs {
        let m = &run["metrics"];
        assert_eq!(m["generated_tokens"], 1);
        assert_eq!(m["passes"], 1);
        assert_eq!(m["neural_proposals"], 0);
        assert_eq!(m["draft_updates"], 1);
        let main = m["main_code_bytes_per_pass"].as_u64().unwrap();
        let update = m["draft_code_bytes_per_update"].as_u64().unwrap();
        assert!(update > 0);
        assert_eq!(
            m["bytes_read_per_pass"].as_f64().unwrap(),
            (main + update) as f64
        );
    }
}

#[test]
fn synthetic_product_cli_info_bench_comparison_json_and_terminal_chat_work() {
    use std::io::Write;
    use std::process::Stdio;
    let folder = Folder::new();
    let info = product_command(&folder, "info")
        .args(["--draft", "none"])
        .output()
        .unwrap();
    assert!(
        info.status.success(),
        "{}",
        String::from_utf8_lossy(&info.stderr)
    );
    let facts = serde_json::Deserializer::from_slice(&info.stdout)
        .into_iter::<Value>()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(facts["layers"], 2);
    assert_eq!(facts["hidden_width"], 512);
    assert_eq!(facts["context"], 512);
    assert_eq!(facts["kv"], "kv5");
    assert_eq!(facts["backend"], "cpu");
    assert!(facts["memory"]["needed_bytes"].as_u64().unwrap() > 0);
    assert!(facts["memory"]["physical_bytes"].as_u64().unwrap() > 0);
    let output_path = folder.0.join("bench.json");
    let bench = product_command(&folder, "bench")
        .args([
            "--n", "2", "--draft", "none", "--draft", "ngram", "--k", "1", "--k", "3", "--json",
        ])
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        bench.status.success(),
        "{}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let report: Value = serde_json::from_slice(&std::fs::read(output_path).unwrap()).unwrap();
    assert_eq!(report["tokens_per_task"], 2);
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 16);
    let mut seen = std::collections::BTreeSet::new();
    for run in runs {
        let config = &run["configuration"];
        let metrics = &run["metrics"];
        seen.insert((
            config["draft"].as_str().unwrap(),
            config["k"].as_u64().unwrap(),
            metrics["task"].as_str().unwrap(),
        ));
        assert_eq!(config["context"], 512);
        assert_eq!(config["backend"], "cpu");
        assert_eq!(config["threads"], 1);
        assert_eq!(config["kv"], "kv5");
        assert_eq!(metrics["generated_tokens"], 2);
        for name in [
            "prompt_tokens",
            "prefill_ms",
            "prefill_tokens_per_s",
            "passes",
            "rows_verified",
            "generation_ms",
            "milliseconds_per_pass",
            "tokens_confirmed_per_pass",
            "tokens_per_s",
            "bytes_read_per_pass",
        ] {
            assert!(
                metrics[name]
                    .as_f64()
                    .is_some_and(|n| n.is_finite() && n > 0.0),
                "{name}: {run}"
            );
        }
        assert_eq!(
            metrics["bytes_read_per_pass"].as_f64().unwrap(),
            metrics["main_code_bytes_per_pass"].as_f64().unwrap()
        );
        assert_eq!(metrics["neural_proposals"], 0);
        assert!(
            metrics["guesses_accepted"].as_u64().unwrap()
                <= metrics["guesses_offered"].as_u64().unwrap()
        );
    }
    assert_eq!(seen.len(), 16);
    let mut chat = product_command(&folder, "chat")
        .args([
            "--draft",
            "ngram",
            "--k",
            "3",
            "--max-tokens",
            "2",
            "--no-think",
            "--temperature",
            "0",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    chat.stdin.take().unwrap().write_all(b"1 2\n").unwrap();
    let chat = chat.wait_with_output().unwrap();
    assert!(
        chat.status.success(),
        "{}",
        String::from_utf8_lossy(&chat.stderr)
    );
    let text = String::from_utf8(chat.stdout).unwrap();
    assert!(text.contains("[2 tokens,"), "{text}");
    assert!(text.contains("tokens/s"), "{text}");
}
#[test]
fn selfcheck_cli_accepts_scalar_reference_and_reports_mismatching_token_ids() {
    let folder = Folder::new();
    let mut reference = selfcheck_reference(&folder);
    folder.json("selfcheck.json", reference.clone());
    let output = selfcheck(&folder);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{text}");
    for name in ["short-answer", "thinking", "plain-text"] {
        assert!(text.contains(name), "{text}");
    }
    assert_eq!(text.matches("PASS").count(), 3, "{text}");
    let got = reference["cases"][0]["ids"][0].as_u64().unwrap();
    let want = (got + 1) % 32;
    reference["cases"][0]["ids"][0] = json!(want);
    reference["cases"][0]["head"][0] = json!(want);
    reference["cases"][0]["sha256"] = json!("0".repeat(64));
    folder.json("selfcheck.json", reference);
    let output = selfcheck(&folder);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("FAIL"), "{text}");
    assert!(
        text.contains(&format!(
            "first different token #0 (zero-based): expected {want}, got {got}"
        )),
        "{text}"
    );
}

/// D23: every forced-off switch the product honours gives the tokens of the
/// default configuration (detected backend, kernel, CPU stages, binary32 Metal
/// arithmetic, automatic CPU share, one worker per core), greedy and seeded,
/// with the fast draft and the adaptive depth running; `info` shows each choice.
#[test]
fn forced_off_switches_give_the_default_tokens_greedy_and_seeded() {
    use yskn::real_engine::{RealEngine, RealEngineConfig};
    use yskn_engine::{Backend, CpuShare, CpuStages, Kernel, MetalOptions};
    let folder = Folder::new();
    let prompt = [1, 2, 1, 2, 1, 2, 3];
    let base = || RealEngineConfig {
        backend: None,
        threads: None,
        k: 7,
        ..folder.config(DraftMode::Both)
    };
    let load = |config: RealEngineConfig| {
        let mut options = product_support::model::draft::options();
        options.capture_layers = vec![0];
        RealEngine::load_with_draft_options(config, options).unwrap()
    };
    let metal = |share| MetalOptions {
        float_mac: true,
        cpu_share: share,
    };
    let cases = [
        (
            "CPU backend",
            RealEngineConfig {
                backend: Some(Backend::Cpu),
                ..base()
            },
            ("backend", "cpu"),
        ),
        (
            "portable kernel",
            RealEngineConfig {
                kernel: Some(Kernel::Portable),
                ..base()
            },
            ("kernel", "portable"),
        ),
        (
            "reference CPU stages",
            RealEngineConfig {
                cpu_stages: CpuStages::Reference,
                ..base()
            },
            ("cpu_stages", "reference"),
        ),
        (
            "integer Metal arithmetic",
            RealEngineConfig {
                metal: MetalOptions {
                    float_mac: false,
                    cpu_share: CpuShare::Auto,
                },
                ..base()
            },
            ("metal_mac", "integer"),
        ),
        (
            "CPU share off",
            RealEngineConfig {
                metal: metal(CpuShare::Off),
                ..base()
            },
            ("cpu_share", "off"),
        ),
        (
            // Gives the CPU rows of every 512-row matrix on any Metal device.
            "CPU share 0.5",
            RealEngineConfig {
                metal: metal(CpuShare::Fixed(0.5)),
                ..base()
            },
            ("cpu_share", "fixed 0.5"),
        ),
        (
            "one thread",
            RealEngineConfig {
                threads: Some(1),
                ..base()
            },
            ("threads", "1"),
        ),
    ];
    let default = load(base());
    let on_metal = default.info().backend == "metal";
    let facts = default.info().to_json();
    assert_eq!(facts["cpu_stages"], "auto (NEON fast paths)");
    assert_eq!(facts["threads"], yskn_engine::Engine::default_threads());
    if !on_metal {
        println!("no Metal device: the Metal switches ran on the CPU backend");
    }
    for sampling in [greedy(), sampled()] {
        let expected = run(&mut load(base()), &prompt, 24, &sampling).ids;
        assert_eq!(expected, scalar(&folder, &prompt, 24, &sampling));
        for (name, config, (key, shown)) in &cases {
            let mut engine = load(config.clone());
            let info = engine.info().to_json();
            let metal_only = matches!(*key, "metal_mac" | "cpu_share");
            if !metal_only || on_metal {
                assert_eq!(
                    info[key].to_string().trim_matches('"'),
                    *shown,
                    "{name}: {info}"
                );
            }
            let ids = run(&mut engine, &prompt, 24, &sampling).ids;
            assert_eq!(ids, expected, "{name} {sampling:?}");
        }
    }
}

/// D23 start-up checks: Apple silicon and macOS 26 or later, failing closed
/// with the requirement and the value found.
#[test]
fn platform_check_requires_apple_silicon_and_macos_26_and_names_the_value_found() {
    use yskn::platform::check_platform;
    for version in ["26.0", "26.4.1", "27.2.1", "30.1"] {
        check_platform("aarch64", "macos", Some(version)).unwrap();
    }
    // macOS 26 reports 16.x to programs built with an older SDK.
    check_platform("aarch64", "macos", Some("16.0")).unwrap();
    for (arch, os, version, expected) in [
        (
            "aarch64",
            "macos",
            Some("15.7.1"),
            "macOS 26 or later; this Mac runs macOS 15.7.1",
        ),
        ("aarch64", "macos", Some("14.0"), "this Mac runs macOS 14.0"),
        ("aarch64", "macos", None, "cannot read this Mac's version"),
        (
            "aarch64",
            "macos",
            Some("unknown"),
            "cannot read the version \"unknown\"",
        ),
        (
            "x86_64",
            "macos",
            Some("27.0"),
            "Apple silicon (macOS on aarch64); this program runs on macos x86_64",
        ),
        (
            "aarch64",
            "linux",
            Some("27.0"),
            "this program runs on linux aarch64",
        ),
    ] {
        let error = check_platform(arch, os, version).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
    yskn::platform::check_this_platform().unwrap();
}

/// D23: before anything is opened, a missing model file is reported by name,
/// with the option that needs it; files a configuration does not read are
/// not required.
#[test]
fn incomplete_model_folder_lists_every_missing_file_and_the_option_needing_it() {
    let folder = Folder::new();
    std::fs::remove_file(folder.0.join("tokenizer.json")).unwrap();
    std::fs::remove_file(folder.0.join("draft.shead")).unwrap();
    std::fs::remove_file(folder.0.join("profile.bin")).unwrap();
    let error = folder
        .config(DraftMode::Both)
        .inspect()
        .unwrap_err()
        .to_string();
    for expected in [
        "is incomplete: missing",
        "profile.bin (files.kv_profile, for --kv kv5; --kv int8 runs without it)",
        "draft.shead (files.draft_head, the draft head, for --draft both; --draft ngram or none runs without it)",
        "tokenizer.json (the tokenizer)",
    ] {
        assert!(error.contains(expected), "{expected}: {error}");
    }
    assert!(
        !error.contains("weights.yskn") && !error.contains("draft.yskn"),
        "{error}"
    );
    let mut config = folder.config(DraftMode::None);
    config.kv = yskn_engine::KvMode::Int8;
    let error = config.inspect().unwrap_err().to_string();
    assert!(
        error.ends_with("missing tokenizer.json (the tokenizer)"),
        "{error}"
    );
    let output = product_command(&folder, "info")
        .args(["--draft", "none", "--kv", "int8"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing tokenizer.json"));
}

/// Header facts of the published model (`yskn info --model-dir ~/models/qwen38-speed`,
/// memory.facts, D23), so the README's memory table is tested without it.
fn published_model() -> yskn::memory::MemoryFacts {
    yskn::memory::MemoryFacts {
        weights_bytes: 10_311_344_128,
        recurrent_bytes: 156_893_184,
        replay_bytes_per_row: 5_916_672,
        cache_bytes_per_token: 89_600,
        int8_cache_bytes_per_token: 37_632,
        dense_bytes: 104_953_856,
        hidden: 5120,
        feed_forward: 17_408,
        vocabulary: 248_320,
        draft: Some(yskn::memory::DraftFacts {
            bytes: 2_721_905_176,
            ring_bytes: 41_943_040,
            features: 25_600,
        }),
    }
}

/// The README table "What a smaller Mac can run": needed bytes per context
/// for kv5/int8 with and without the draft (pass width 16, the defaults), the
/// largest context each memory size accepts, and what the check prints.
#[test]
fn published_model_memory_table_matches_the_readme() {
    use yskn::memory::MemoryFacts;
    const GIB: u64 = 1 << 30;
    let kv5 = published_model();
    let variants = [
        kv5.clone(),
        MemoryFacts {
            draft: None,
            ..kv5.clone()
        },
        MemoryFacts {
            cache_bytes_per_token: kv5.int8_cache_bytes_per_token,
            ..kv5.clone()
        },
        MemoryFacts {
            cache_bytes_per_token: kv5.int8_cache_bytes_per_token,
            draft: None,
            ..kv5.clone()
        },
    ];
    // --ctx: kv5 both, kv5 none, int8 both, int8 none.
    let needed: [(usize, [u64; 4]); 7] = [
        (
            4096,
            [
                17_480_703_900,
                14_250_275_636,
                17_235_913_833,
                14_005_485_568,
            ],
        ),
        (
            8192,
            [
                17_902_812_265,
                14_672_384_000,
                17_413_232_130,
                14_182_803_866,
            ],
        ),
        (
            16384,
            [
                18_747_028_994,
                15_516_600_730,
                17_767_868_726,
                14_537_440_461,
            ],
        ),
        (
            32768,
            [
                20_435_462_454,
                17_205_034_189,
                18_477_141_916,
                15_246_713_652,
            ],
        ),
        (
            65536,
            [
                23_812_329_372,
                20_581_901_108,
                19_895_688_297,
                16_665_260_032,
            ],
        ),
        (
            131072,
            [
                30_566_063_209,
                27_335_634_944,
                22_732_781_058,
                19_502_352_794,
            ],
        ),
        (
            262144,
            [
                44_073_530_882,
                40_843_102_618,
                28_406_966_582,
                25_176_538_317,
            ],
        ),
    ];
    for (context, row) in needed {
        for (facts, bytes) in variants.iter().zip(row) {
            assert_eq!(
                facts.estimate(context, 16, 0).unwrap().needed_bytes,
                bytes,
                "{context}"
            );
        }
    }
    // Largest power-of-two --ctx that fits (0: none from 4096 up).
    let largest: [(u64, [usize; 4]); 4] = [
        (16, [0, 16384, 0, 65536]),
        (24, [65536, 65536, 131072, 262144]),
        (32, [131072, 131072, 262144, 262144]),
        (48, [262144, 262144, 262144, 262144]),
    ];
    for (gib, row) in largest {
        for (facts, expected) in variants.iter().zip(row) {
            let fits = needed
                .iter()
                .map(|&(context, _)| context)
                .filter(|&context| {
                    facts
                        .estimate(context, 16, gib * GIB)
                        .unwrap()
                        .fits(gib * GIB)
                })
                .max()
                .unwrap_or(0);
            assert_eq!(fits, expected, "{gib} GB");
        }
    }
    // What a 16 GB Mac prints for the defaults (README).
    let message = kv5
        .estimate(32768, 16, 16 * GIB)
        .unwrap()
        .enforce(16 * GIB)
        .unwrap_err()
        .to_string();
    for line in [
        "this configuration needs 20435462454 bytes (19.03 GiB), including 3599183646 bytes margin; physical memory is 17179869184 bytes (16.00 GiB), 3255593270 bytes short.",
        "needed: model weights 10311344128, draft 2721905176, KV cache for --ctx 32768 2936012800, recurrent state 156893184, prompt checkpoint 199960576, working memory 510162944, margin 3599183646.",
        "  --draft none: saves 3230428265 bytes, needs 17205034189 bytes (16.02 GiB)",
        "  --ctx 16384: saves 1688433460 bytes, needs 18747028994 bytes (17.46 GiB)",
        "  --ctx 4096: saves 2954758554 bytes, needs 17480703900 bytes (16.28 GiB)",
        "  --kv int8: saves 1958320538 bytes, needs 18477141916 bytes (17.21 GiB)",
        "largest that fits: --ctx 16384 --draft none; --ctx 32768 --draft none --kv int8",
    ] {
        assert!(message.lines().any(|l| l == line), "{line}\n{message}");
    }
    let message = kv5
        .estimate(32768, 16, 8 * GIB)
        .unwrap()
        .enforce(8 * GIB)
        .unwrap_err()
        .to_string();
    assert!(
        message.ends_with("nothing listed fits: even --ctx 4096 --draft none --kv int8 needs 14005485568 bytes (13.04 GiB), so this Mac does not have enough memory for this model."),
        "{message}"
    );
    kv5.estimate(32768, 16, 24 * GIB)
        .unwrap()
        .enforce(24 * GIB)
        .unwrap();
}

/// D23: the product's switches, as options or environment variables (an
/// option wins; an invalid value from either is an error naming its source),
/// and `yskn info` shows the effective choices.
#[test]
fn product_switches_are_validated_and_info_shows_the_effective_choices() {
    let folder = Folder::new();
    let info = |args: &[&str], env: &[(&str, &str)]| {
        let mut command = product_command(&folder, "info");
        for key in [
            "YSKN_BACKEND",
            "YSKN_KERNEL",
            "YSKN_CPU_STAGES",
            "YSKN_METAL_MAC",
            "YSKN_METAL_CPU_SHARE",
        ] {
            command.env_remove(key);
        }
        command
            .args(["--draft", "none"])
            .args(args)
            .envs(env.iter().copied());
        command.output().unwrap()
    };
    let facts = |output: &std::process::Output| -> Value {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::Deserializer::from_slice(&output.stdout)
            .into_iter::<Value>()
            .next()
            .unwrap()
            .unwrap()
    };
    // product_command passes --backend cpu --threads 1.
    let cpu = facts(&info(
        &["--kernel", "portable", "--cpu-stages", "reference"],
        &[],
    ));
    assert_eq!(cpu["backend"], "cpu");
    assert_eq!(cpu["device"], "portable x1");
    assert_eq!(cpu["kernel"], "portable");
    assert_eq!(cpu["threads"], 1);
    assert_eq!(cpu["cpu_stages"], "reference");
    assert_eq!(cpu["metal_mac"], "unused (CPU backend)");
    assert_eq!(cpu["cpu_share"], "unused (CPU backend)");
    assert!(cpu["machine"]["macos"].is_string(), "{cpu}");
    // The environment applies when the option is absent; the option wins.
    let env = facts(&info(
        &[],
        &[
            ("YSKN_KERNEL", "portable"),
            ("YSKN_CPU_STAGES", "reference"),
        ],
    ));
    assert_eq!(
        (env["kernel"].as_str(), env["cpu_stages"].as_str()),
        (Some("portable"), Some("reference"))
    );
    let both = facts(&info(
        &["--cpu-stages", "auto"],
        &[("YSKN_CPU_STAGES", "reference")],
    ));
    assert_eq!(both["cpu_stages"], "auto (NEON fast paths)");
    for (args, env, expected) in [
        (&["--kernel", "fast"][..], &[][..], "--kernel=fast"),
        (&["--cpu-stages", "on"], &[], "--cpu-stages=on"),
        (
            &["--metal-mac", "half"],
            &[],
            "--metal-mac=half: expected float or integer",
        ),
        (&["--cpu-share", "0.95"], &[], "--cpu-share=0.95"),
        (&["--cpu-share", "-1"], &[], "--cpu-share=-1"),
        (
            &[],
            &[("YSKN_METAL_CPU_SHARE", "most")],
            "YSKN_METAL_CPU_SHARE=most",
        ),
        // An invalid environment value is an error even when an option overrides it.
        (
            &["--kernel", "portable"],
            &[("YSKN_KERNEL", "fast")],
            "YSKN_KERNEL=fast",
        ),
        (&[], &[("YSKN_BACKEND", "gpu")], "YSKN_BACKEND=gpu"),
    ] {
        let output = info(args, env);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{args:?} {env:?}");
        assert!(stderr.contains(expected), "{expected}: {stderr}");
    }
    let mut command = product_command(&folder, "info");
    command.env_remove("YSKN_BACKEND");
    let metal = command
        .args([
            "--draft",
            "none",
            "--backend",
            "metal",
            "--metal-mac",
            "integer",
            "--cpu-share",
            "off",
        ])
        .output()
        .unwrap();
    #[cfg(target_os = "macos")]
    let device = yskn_engine::MetalEngine::device_exists();
    #[cfg(not(target_os = "macos"))]
    let device = false;
    if device {
        let metal = facts(&metal);
        assert_eq!(metal["backend"], "metal");
        assert_eq!(metal["metal_mac"], "integer");
        assert_eq!(metal["cpu_share"], "off");
        assert!(
            metal["machine"]["metal_max_buffer_length"]
                .as_u64()
                .unwrap()
                > 0
        );
    } else {
        assert!(!metal.status.success());
        assert!(String::from_utf8_lossy(&metal.stderr).contains("--backend metal was requested"));
    }
    let help = Command::new(env!("CARGO_BIN_EXE_yskn"))
        .arg("--help")
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    for switch in [
        "--backend metal|cpu",
        "YSKN_KERNEL",
        "--cpu-stages auto|reference",
        "--metal-mac float|integer",
        "--cpu-share auto|off|F",
        "--threads N",
    ] {
        assert!(help.contains(switch), "{switch}");
    }
}

mod sockets {
    use super::*;
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::{Request, client::conn::http1};
    use hyper_util::rt::TokioIo;
    use std::{
        net::SocketAddr,
        sync::{Arc, Mutex, mpsc},
        time::{Duration, Instant},
    };
    use tokio::{io::AsyncWriteExt, net::TcpStream, sync::oneshot};
    use yskn::{
        chat::ChatRunner,
        real_engine::RealEngine,
        server::{Config, Server},
    };

    const PATIENCE: Duration = Duration::from_secs(30);
    /// A cancelled call: prompt length, its empty (prefill) callbacks, ids returned.
    #[derive(Debug)]
    struct Cancelled {
        prompt: usize,
        prefill_callbacks: usize,
        ids: usize,
    }
    #[derive(Default)]
    struct Observation {
        gate: Mutex<Option<PrefillGate>>,
        cancelled: Mutex<Vec<Cancelled>>,
    }
    struct PrefillGate {
        reached: oneshot::Sender<()>,
        disconnected: mpsc::Receiver<()>,
    }
    impl Observation {
        fn pausing_prefill() -> (Arc<Self>, oneshot::Receiver<()>, mpsc::SyncSender<()>) {
            let (reached, boundary) = oneshot::channel();
            let (disconnected, resume) = mpsc::sync_channel(1);
            let observation = Arc::new(Self {
                gate: Mutex::new(Some(PrefillGate {
                    reached,
                    disconnected: resume,
                })),
                ..Default::default()
            });
            (observation, boundary, disconnected)
        }
    }
    impl PrefillGate {
        fn wait_for_disconnect(
            self,
            ids: &[u32],
            cb: &mut dyn FnMut(&[u32]) -> yskn::Result<bool>,
        ) -> yskn::Result<bool> {
            self.reached
                .send(())
                .expect("test left before prefill boundary");
            self.disconnected
                .recv_timeout(PATIENCE)
                .expect("client was not disconnected");
            // Closing the client and Hyper dropping its response body happen
            // on different threads. Poll cancellation at this same boundary
            // until the real callback observes it; never advance another row.
            let deadline = Instant::now() + PATIENCE;
            while cb(ids)? {
                assert!(
                    Instant::now() < deadline,
                    "disconnect did not cancel prefill"
                );
                std::thread::yield_now();
            }
            Ok(false)
        }
    }
    struct Observed {
        inner: RealEngine,
        observation: Arc<Observation>,
    }
    impl Engine for Observed {
        fn context_capacity(&self) -> usize {
            self.inner.context_capacity()
        }
        fn reset(&mut self) -> yskn::Result<()> {
            self.inner.reset()
        }
        fn generate(
            &mut self,
            p: &[u32],
            n: usize,
            s: &Sampling,
            stop: Option<&[u32]>,
            cb: &mut dyn FnMut(&[u32]) -> yskn::Result<bool>,
        ) -> yskn::Result<Generation> {
            let mut prefill_callbacks = 0;
            let result = self.inner.generate(p, n, s, stop, &mut |ids| {
                if ids.is_empty() {
                    prefill_callbacks += 1;
                    if prefill_callbacks == 2 {
                        let gate = self.observation.gate.lock().unwrap().take();
                        if let Some(gate) = gate {
                            return gate.wait_for_disconnect(ids, cb);
                        }
                    }
                }
                cb(ids)
            })?;
            if result.finish == Finish::Cancelled {
                self.observation.cancelled.lock().unwrap().push(Cancelled {
                    prompt: p.len(),
                    prefill_callbacks,
                    ids: result.ids.len(),
                });
            }
            Ok(result)
        }
    }
    async fn request(addr: SocketAddr, body: Value) -> (u16, String) {
        tokio::time::timeout(PATIENCE, async {
            let stream = TcpStream::connect(addr).await.unwrap();
            let (mut sender, conn) = http1::handshake(TokioIo::new(stream)).await.unwrap();
            tokio::spawn(async move {
                let _ = conn.await;
            });
            let response = sender
                .send_request(
                    Request::post("/v1/chat/completions")
                        .header("host", addr.to_string())
                        .header("connection", "close")
                        .body(Full::new(Bytes::from(body.to_string())))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status().as_u16();
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(bytes.to_vec()).unwrap())
        })
        .await
        .expect("real-engine request timed out")
    }
    async fn start(
        folder: &Folder,
        observation: Arc<Observation>,
    ) -> (
        SocketAddr,
        oneshot::Sender<()>,
        tokio::task::JoinHandle<yskn::Result<()>>,
    ) {
        let config = Config {
            port: 0,
            defaults: yskn::request::RequestDefaults {
                thinking_budget: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine_config = folder.config(DraftMode::None);
        engine_config.prefill_width = 1;
        let path = folder.0.clone();
        let server = Server::bind(config, move || {
            ChatRunner::new(
                Box::new(Observed {
                    inner: RealEngine::load(engine_config)?,
                    observation,
                }),
                yskn_text::ChatFormatter::load(path)?,
                1,
            )
        })
        .await
        .unwrap();
        let deadline = tokio::time::Instant::now() + PATIENCE;
        while !server.is_ready() {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let addr = server.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        let task = tokio::spawn(server.run(async {
            let _ = rx.await;
        }));
        (addr, tx, task)
    }
    #[tokio::test]
    async fn observed_prefill_gate_holds_real_engine_until_disconnect_then_cancels() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (observation, boundary, disconnected) = Observation::pausing_prefill();
        let cancelled = Arc::new(AtomicBool::new(false));
        let signal = cancelled.clone();
        let worker_observation = observation.clone();
        let worker = tokio::task::spawn_blocking(move || {
            let folder = Folder::new();
            let mut engine = Observed {
                inner: folder.engine(DraftMode::None),
                observation: worker_observation,
            };
            engine
                .generate(&[1; 12], 2, &greedy(), None, &mut |_| {
                    Ok(!signal.load(Ordering::Acquire))
                })
                .unwrap()
        });
        tokio::time::timeout(PATIENCE, boundary)
            .await
            .unwrap()
            .unwrap();
        assert!(
            !worker.is_finished(),
            "engine crossed the held prefill boundary"
        );
        assert!(observation.cancelled.lock().unwrap().is_empty());
        cancelled.store(true, Ordering::Release);
        disconnected.send(()).unwrap();
        let result = tokio::time::timeout(PATIENCE, worker)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.finish, Finish::Cancelled);
        assert!(result.ids.is_empty());
        let calls = observation.cancelled.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].prefill_callbacks, 2);
    }
    #[tokio::test]
    async fn real_socket_streaming_and_nonstreaming_use_scalar_tokens_and_exact_usage() {
        let folder = Folder::new();
        let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
        let messages = json!([{"role":"user","content":"1 2"}]);
        let prompt = formatter
            .prompt(&messages, None, &json!({"enable_thinking":false}))
            .unwrap()
            .0;
        let expected_ids = scalar(&folder, &prompt, 4, &greedy());
        assert_eq!(expected_ids.len(), 4);
        let expected_text = formatter.decode(&expected_ids).unwrap();
        // The server must report usage for each request, as D16 defines it.
        let check_usage = |usage: &Value, cached: usize| {
            assert_eq!(usage["prompt_tokens"], prompt.len(), "{usage}");
            assert_eq!(usage["completion_tokens"], expected_ids.len(), "{usage}");
            assert_eq!(
                usage["total_tokens"],
                prompt.len() + expected_ids.len(),
                "{usage}"
            );
            assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], cached);
        };
        let (addr, shutdown, task) = start(&folder, Arc::default()).await;
        let body =
            json!({"messages":messages,"temperature":0,"max_tokens":4,"enable_thinking":false});
        let (status, text) = request(addr, body.clone()).await;
        assert_eq!(status, 200, "{text}");
        let response: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(response["choices"][0]["message"]["content"], expected_text);
        assert_eq!(response["choices"][0]["finish_reason"], "length");
        // A fresh engine has nothing cached.
        check_usage(&response["usage"], 0);
        let mut stream_body = body;
        stream_body["stream"] = json!(true);
        stream_body["stream_options"] = json!({"include_usage":true});
        let (status, text) = request(addr, stream_body).await;
        assert_eq!(status, 200, "{text}");
        assert!(text.contains("data: [DONE]"), "{text}");
        let chunks: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let generated: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(generated, expected_text);
        let finishes: Vec<_> = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["finish_reason"].as_str())
            .collect();
        assert_eq!(finishes, ["length"], "{text}");
        let usage = chunks
            .iter()
            .find_map(|chunk| chunk.get("usage").filter(|v| v.is_object()))
            .expect("streamed usage");
        // The same prompt again restores the snapshot taken before its final,
        // pending token: everything but that token is reused.
        check_usage(usage, prompt.len() - 1);
        shutdown.send(()).unwrap();
        task.await.unwrap().unwrap();
    }
    #[tokio::test]
    async fn real_socket_disconnect_during_prefill_cancels_and_releases_engine_for_next_chat() {
        let folder = Folder::new();
        let (observation, boundary, disconnected) = Observation::pausing_prefill();
        let (addr, shutdown, task) = start(&folder, observation.clone()).await;
        let long_messages = json!([{"role":"user","content":"1 ".repeat(256)}]);
        let body=json!({"messages":long_messages,"stream":true,"max_tokens":4,"temperature":0,"enable_thinking":false}).to_string();
        let mut client = TcpStream::connect(addr).await.unwrap();
        client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
        tokio::time::timeout(PATIENCE, boundary)
            .await
            .expect("prefill did not reach the held boundary")
            .expect("engine left before prefill boundary");
        drop(client);
        disconnected.send(()).unwrap();
        let messages = json!([{"role":"user","content":"2"}]);
        let (status, text) = request(
            addr,
            json!({"messages":messages,"max_tokens":2,"temperature":0,"enable_thinking":false}),
        )
        .await;
        assert_eq!(status, 200, "next request did not acquire engine: {text}");
        let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
        let long_prompt = formatter
            .prompt(&long_messages, None, &json!({"enable_thinking":false}))
            .unwrap()
            .0;
        let cancelled = std::mem::take(&mut *observation.cancelled.lock().unwrap());
        assert_eq!(cancelled.len(), 1, "{cancelled:?}");
        let cancelled = &cancelled[0];
        assert_eq!(cancelled.prompt, long_prompt.len(), "{cancelled:?}");
        // Width-1 prefill calls back once before and once after each of the
        // prompt.len() - 1 rows; cancellation must land strictly inside.
        assert_eq!(cancelled.ids, 0, "cancelled after prefill: {cancelled:?}");
        assert_eq!(cancelled.prefill_callbacks, 2, "{cancelled:?}");
        // The released engine answers the next chat exactly as a fresh one.
        let prompt = formatter
            .prompt(&messages, None, &json!({"enable_thinking":false}))
            .unwrap()
            .0;
        let expected = scalar(&folder, &prompt, 2, &greedy());
        let response: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            response["choices"][0]["message"]["content"],
            formatter.decode(&expected).unwrap()
        );
        assert_eq!(response["usage"]["prompt_tokens"], prompt.len());
        assert_eq!(response["usage"]["completion_tokens"], expected.len());
        assert_eq!(
            response["usage"]["prompt_tokens_details"]["cached_tokens"],
            0
        );
        shutdown.send(()).unwrap();
        task.await.unwrap().unwrap();
    }
}
