mod support;

use std::io::Cursor;

use serde_json::{Value, json};
use yskn_format::{Container, f32_to_bf16};
use yskn_ref::draft::{ContextFeatures, Draft, DraftOptions};
use yskn_ref::draft_head::{DraftChoice, ReducedHead};
use yskn_ref::model::CaptureOptions;
use yskn_ref::speculation::{DraftProposer, GenerationLimits, NoProposer, Proposer, generate};
use yskn_ref::{Model, Sampler, SamplingParameters};

// Shorten the ordinary seeded artifact to two capture slots and one draft layer.
// The removed stream suffix/descriptors remain inert bytes, not a different codec.
fn edit_header(bytes: &mut [u8], edit: impl FnOnce(&mut Value)) {
    let length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let data = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&bytes[32..32 + length]).unwrap();
    edit(&mut header);
    let encoded = serde_json::to_vec(&header).unwrap();
    assert!(32 + encoded.len() <= data);
    bytes[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
    bytes[32..data].fill(0);
    bytes[32..32 + encoded.len()].copy_from_slice(&encoded);
}

fn draft_bytes() -> Vec<u8> {
    let mut bytes = support::draft::container(23);
    edit_header(&mut bytes, |header| {
        header["model"]["layers"] = json!(1);
        let fc = &mut header["tensors"][0];
        assert_eq!(fc["name"], "bdraft2.fc");
        fc["cols"] = json!(1024);
        fc["rot_in"]["m"] = json!(2);
        fc["stream_bytes"] = json!(512 * (1024 / 256) * fc["bits"].as_u64().unwrap() * 16);
    });
    bytes
}

fn draft_options() -> DraftOptions {
    DraftOptions {
        capture_layers: vec![0, 1],
        ..support::draft::options()
    }
}

fn draft_from(bytes: Vec<u8>, head: Vec<u8>) -> Draft {
    Draft::from_container(
        &mut Container::from_bytes(bytes).unwrap(),
        ReducedHead::read(&mut Cursor::new(head)).unwrap(),
        draft_options(),
    )
    .unwrap()
}

fn main_bytes() -> Vec<u8> {
    support::synthetic_model_with_dimensions(
        73,
        &["linear_attention", "full_attention"],
        512,
        512,
        32,
    )
}

fn model(bytes: Vec<u8>) -> Model {
    Model::from_container(Container::from_bytes(bytes).unwrap()).unwrap()
}

fn capture_options() -> CaptureOptions {
    CaptureOptions {
        layers: vec![0, 1],
        max_rows: 8,
    }
}

fn header(bytes: &[u8]) -> Value {
    let length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[32..32 + length]).unwrap()
}

fn zero_corrections(bytes: &mut [u8]) {
    for vector in header(bytes)["vectors"].as_array().unwrap() {
        if vector["name"].as_str().unwrap().starts_with("bdraft2.sel.") {
            let start = vector["off"].as_u64().unwrap() as usize;
            let length = vector["n"].as_u64().unwrap() as usize * 2;
            bytes[start..start + length].fill(0);
        }
    }
}

fn blank_head() -> Vec<u8> {
    let mut bytes = support::draft::shead(37);
    bytes[24..].fill(0);
    bytes
}

fn write_vector(bytes: &mut [u8], name: &str, index: usize, value: f32) {
    let header = header(bytes);
    let vector = header["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|vector| vector["name"] == name)
        .unwrap();
    assert!(index < vector["n"].as_u64().unwrap() as usize);
    let offset = vector["off"].as_u64().unwrap() as usize + index * 2;
    bytes[offset..offset + 2].copy_from_slice(&f32_to_bf16(value).to_le_bytes());
}

// Minimum-norm interpolation of a finite synthetic training set. The label is
// independent of the draft: main's all-zero output head always selects token 0.
// Fitting the exported artifact avoids a test-only production prediction hook.
fn separator(examples: &[(Vec<f32>, f64)]) -> Vec<f32> {
    let n = examples.len();
    let mut matrix = vec![vec![0.0_f64; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = examples[i]
                .0
                .iter()
                .zip(&examples[j].0)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
        }
        matrix[i][i] += 1e-10;
        matrix[i][n] = examples[i].1;
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| matrix[a][col].abs().total_cmp(&matrix[b][col].abs()))
            .unwrap();
        matrix.swap(col, pivot);
        let divisor = matrix[col][col];
        assert!(divisor.abs() > 1e-16);
        for value in &mut matrix[col][col..=n] {
            *value /= divisor;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            for j in col..=n {
                matrix[row][j] -= factor * matrix[col][j];
            }
        }
    }
    (0..512)
        .map(|channel| {
            examples
                .iter()
                .enumerate()
                .map(|(i, (row, _))| matrix[i][n] * f64::from(row[channel]))
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn synthetic_teacher_accepts_correct_captures_and_rejects_lagged_or_swapped_slots() {
    let mut main_artifact = main_bytes();
    let head = header(&main_artifact)["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tensor| tensor["name"] == "lm_head")
        .unwrap()
        .clone();
    let offset = head["scale_off"].as_u64().unwrap() as usize;
    let rows = head["physical_rows"].as_u64().unwrap() as usize;
    main_artifact[offset..offset + rows * 2].fill(0);
    let mut main = model(main_artifact);
    main.set_capture(capture_options()).unwrap();
    main.prefill(&[3, 7, 11, 0]).unwrap();
    assert_eq!(
        yskn_ref::output_head_sampler::greedy(main.logits().unwrap()).unwrap(),
        0
    );
    let captured = main.captured_rows().to_vec();
    assert_eq!(captured.len(), 4);
    assert_ne!(&captured[1].values[..512], &captured[1].values[512..]);

    let mut base_bytes = draft_bytes();
    zero_corrections(&mut base_bytes);
    let mut hidden_sets = Vec::new();
    let mut states = Vec::new();
    for wiring in 0..3 {
        let mut draft = draft_from(base_bytes.clone(), blank_head());
        for position in 0_usize..4 {
            let source = if wiring == 1 {
                position.saturating_sub(1)
            } else {
                position
            };
            let mut values = captured[source].values.clone();
            if wiring == 2 {
                values.rotate_left(512);
            }
            draft
                .push_context(position, ContextFeatures { values: &values })
                .unwrap();
        }
        let embeddings: Vec<_> = (0..8)
            .map(|row| main.embedding(if row == 0 { 0 } else { 31 }).unwrap())
            .collect();
        hidden_sets.push(draft.block_hidden(&embeddings).unwrap());
        states.push(draft.snapshot());
    }
    assert_ne!(states[0], states[1]);
    assert_ne!(states[0], states[2]);
    let examples: Vec<_> = hidden_sets
        .iter()
        .enumerate()
        .flat_map(|(wiring, hidden)| {
            hidden
                .chunks_exact(512)
                .map(move |row| (row.to_vec(), if wiring == 0 { 1.0 } else { -1.0 }))
        })
        .collect();
    let weights = separator(&examples);
    for (channel, &weight) in weights.iter().enumerate() {
        write_vector(&mut base_bytes, "bdraft2.sel.hp", channel, weight);
    }
    for token in 0..32 {
        write_vector(&mut base_bytes, "bdraft2.sel.pred", token * 32, 1.0);
    }
    write_vector(&mut base_bytes, "bdraft2.sel.succ", 0, 1.0);
    write_vector(&mut base_bytes, "bdraft2.sel.succ", 32, -1.0);
    // Independently ask the actual main head for every teacher-forced choice.
    main.run_provisional(&[0; 8]).unwrap();
    let teacher: Vec<_> = main
        .row_logits()
        .iter()
        .map(|logits| yskn_ref::output_head_sampler::greedy(logits).unwrap())
        .collect();
    main.discard_provisional();
    assert_eq!(teacher, vec![0; 8]);
    let prompt_checkpoint = main.snapshot().unwrap();
    let mut acceptance = Vec::new();
    for (wiring, state) in states.iter().enumerate() {
        main.restore(&prompt_checkpoint).unwrap();
        let mut draft = draft_from(base_bytes.clone(), blank_head());
        // Correct wiring is exercised through the production capture hook.
        // Deliberately corrupted rings start at the same committed position.
        if wiring != 0 {
            draft.restore(state).unwrap();
        }
        let mut proposer = DraftProposer::new(draft, 7, DraftChoice::Greedy).unwrap();
        proposer.synchronize(&mut main).unwrap();
        assert_eq!(proposer.draft().snapshot(), *state);
        let proposal = proposer
            .propose_with(&mut main, &[3, 7, 11, 0], 0, 7)
            .unwrap();
        acceptance.push(
            proposal
                .iter()
                .zip(&teacher)
                .take_while(|(guess, expected)| guess == expected)
                .count(),
        );
        eprintln!("teacher wiring={wiring} tokens={proposal:?}");
    }
    eprintln!("teacher accepted prefix correct/lag-one/swapped = {acceptance:?} / 7");
    assert_eq!(acceptance, [7, 0, 0]);
    main.restore(&prompt_checkpoint).unwrap();
    let mut proposer =
        DraftProposer::new(draft_from(base_bytes, blank_head()), 7, DraftChoice::Greedy).unwrap();
    let mut limits = GenerationLimits::new(8, main.context_capacity());
    limits.max_draft_tokens = 7;
    let generated = generate(&mut main, &mut proposer, &mut Sampler::Greedy, &[0], limits).unwrap();
    assert_eq!(generated.tokens, vec![0; 8]);
    assert_eq!(generated.statistics.guesses_offered, 7);
    assert_eq!(generated.statistics.guesses_accepted, 7);
    assert_eq!(generated.statistics.tokens_confirmed_per_pass, [8]);
    assert_eq!(proposer.draft().position(), main.position());
}

fn decode(
    main: &mut Model,
    proposer: &mut impl Proposer,
    sampler: &Sampler,
    prompt: &[u32],
    k: usize,
    count: usize,
) -> (Vec<u32>, u64) {
    let mut limits = GenerationLimits::new(count, main.context_capacity());
    limits.max_draft_tokens = k;
    let result = generate(main, proposer, &mut sampler.clone(), prompt, limits).unwrap();
    assert_eq!(result.tokens.len(), count);
    if k != 0 {
        assert!(result.statistics.guesses_offered > 0);
    }
    (result.tokens, main.state_digest())
}

#[test]
fn reference_neural_generation_matches_k0_across_joint_restore_and_reused_prefix() {
    let base = model(main_bytes());
    let draft_artifact = draft_bytes();
    let head_artifact = support::draft::shead(37);
    let prompt = [3, 7, 11, 9];
    let second_prompt = [3, 7, 11, 13, 5, 17];
    for seed in [None, Some(17), Some(0xabcdef)] {
        let sampler = seed.map_or(Sampler::Greedy, |seed| {
            Sampler::Sampled(SamplingParameters {
                seed,
                ..SamplingParameters::default()
            })
        });
        let choice = seed.map_or(DraftChoice::Greedy, DraftChoice::sampled);
        let mut sequential = base.new_session().unwrap();
        sequential.prefill(&prompt[..3]).unwrap();
        let first_expected = decode(&mut sequential, &mut NoProposer, &sampler, &prompt, 0, 5);
        let last = *first_expected.0.last().unwrap();
        let continued_expected = decode(&mut sequential, &mut NoProposer, &sampler, &[last], 0, 8);
        // Prefix reuse is checked against a fresh, independently prefetched session.
        let mut fresh_second = base.new_session().unwrap();
        fresh_second
            .prefill(&second_prompt[..second_prompt.len() - 1])
            .unwrap();
        let second_expected = decode(
            &mut fresh_second,
            &mut NoProposer,
            &sampler,
            &second_prompt,
            0,
            5,
        );
        for k in [1, 3, 7] {
            let mut proposer = DraftProposer::new(
                draft_from(draft_artifact.clone(), head_artifact.clone()),
                k,
                choice,
            )
            .unwrap();
            let mut main = base.new_session().unwrap();
            main.set_capture(proposer.capture_options()).unwrap();
            main.prefill(&prompt[..3]).unwrap();
            let prefix = proposer.snapshot_session(&mut main).unwrap();
            assert_eq!(
                proposer
                    .propose_with(&mut main, &prompt[..3], prompt[3], k)
                    .unwrap()
                    .len(),
                k
            );
            assert_eq!(
                decode(&mut main, &mut proposer, &sampler, &prompt, k, 5),
                first_expected,
                "seed={seed:?} K={k} first request"
            );
            let checkpoint = proposer.snapshot_session(&mut main).unwrap();
            let ring = proposer.draft().snapshot();
            assert_eq!(
                decode(&mut main, &mut proposer, &sampler, &[last], k, 8),
                continued_expected,
                "seed={seed:?} K={k} continuation"
            );
            proposer.restore_session(&mut main, &checkpoint).unwrap();
            assert_eq!(proposer.draft().snapshot(), ring);
            assert_eq!(
                decode(&mut main, &mut proposer, &sampler, &[last], k, 8),
                continued_expected,
                "seed={seed:?} K={k} restored continuation"
            );
            proposer.restore_session(&mut main, &prefix).unwrap();
            main.prefill(&second_prompt[3..second_prompt.len() - 1])
                .unwrap();
            assert_eq!(
                decode(&mut main, &mut proposer, &sampler, &second_prompt, k, 5),
                second_expected,
                "seed={seed:?} K={k} reused prefix"
            );
            assert_eq!(proposer.draft().position(), main.position());
        }
    }
}

#[test]
fn reused_and_restored_sessions_reseed_after_a_long_new_prompt() {
    for restore in [false, true] {
        let mut main = model(main_bytes());
        let mut proposer = DraftProposer::new(
            draft_from(draft_bytes(), blank_head()),
            3,
            DraftChoice::Greedy,
        )
        .unwrap();
        main.set_capture(proposer.capture_options()).unwrap();
        main.prefill(&[3, 7, 11]).unwrap();
        let prefix = proposer.snapshot_session(&mut main).unwrap();
        decode(&mut main, &mut proposer, &Sampler::Greedy, &[9], 3, 2);
        if restore {
            proposer.restore_session(&mut main, &prefix).unwrap();
        }
        let mut expected = draft_from(draft_bytes(), blank_head());
        main.prefill(&[13; 12]).unwrap();
        // A fresh ring built from the retained suffix is bit-identical to the
        // upstream ring after every row older than the window is overwritten.
        let rows = main.captured_rows().to_vec();
        for position in 0..main.position() {
            let values = rows
                .iter()
                .find(|r| r.position == position)
                .map_or_else(|| vec![0.0; 1024], |r| r.values.clone());
            expected
                .push_context(position, ContextFeatures { values: &values })
                .unwrap();
        }
        proposer.synchronize(&mut main).unwrap();
        assert_eq!(proposer.draft().snapshot(), expected.snapshot());
        decode(&mut main, &mut proposer, &Sampler::Greedy, &[17], 3, 2);
    }
}

#[test]
fn main_only_restore_or_reset_is_detected_even_after_advancing_past_the_draft() {
    for reset in [false, true] {
        let mut main = model(main_bytes());
        let mut proposer = DraftProposer::new(
            draft_from(draft_bytes(), blank_head()),
            3,
            DraftChoice::Greedy,
        )
        .unwrap();
        main.set_capture(proposer.capture_options()).unwrap();
        main.prefill(&[3; 3]).unwrap();
        let prefix = main.snapshot().unwrap();
        main.prefill(&[7; 3]).unwrap();
        proposer.synchronize(&mut main).unwrap();
        if reset {
            main.reset().unwrap();
        } else {
            main.restore(&prefix).unwrap();
        }
        main.prefill(&[11; 7]).unwrap();
        assert!(
            proposer.synchronize(&mut main).is_err(),
            "main-only reset={reset} must invalidate the old draft context"
        );
    }
}

#[test]
fn delivered_captures_are_released_and_joint_snapshots_do_not_retain_them() {
    let mut main = model(main_bytes());
    let mut proposer = DraftProposer::new(
        draft_from(draft_bytes(), blank_head()),
        3,
        DraftChoice::Greedy,
    )
    .unwrap();
    main.set_capture(proposer.capture_options()).unwrap();
    main.prefill(&[3; 12]).unwrap();
    assert_eq!(main.captured_rows().len(), 8);
    let checkpoint = proposer.snapshot_session(&mut main).unwrap();
    assert!(main.captured_rows().is_empty());
    assert_eq!(main.capture_bytes(), 0);
    assert_eq!(main.snapshot().unwrap().capture_bytes(), 0);
    decode(&mut main, &mut proposer, &Sampler::Greedy, &[9], 3, 2);
    assert!(main.captured_rows().is_empty());
    proposer.restore_session(&mut main, &checkpoint).unwrap();
    assert!(main.captured_rows().is_empty());
    assert_eq!(main.snapshot().unwrap().capture_bytes(), 0);
}

#[test]
fn depth_dependent_teacher_uses_block_successors_and_the_distinct_frontier() {
    let mut main = model(main_bytes());
    main.set_capture(capture_options()).unwrap();
    main.prefill(&[3, 7, 11, 0]).unwrap();
    let mut artifact = draft_bytes();
    zero_corrections(&mut artifact);
    let mut draft = draft_from(artifact.clone(), blank_head());
    for row in main.captured_rows() {
        draft
            .push_context(
                row.position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    // The teacher is next(input) = input + 1. History ends in 0, frontier is 5,
    // so the seven independently specified targets are 6..=12.
    let hidden: Vec<_> = [5, 0]
        .into_iter()
        .map(|frontier| {
            let embeddings: Vec<_> = (0..8)
                .map(|j| main.embedding(if j == 0 { frontier } else { 31 }).unwrap())
                .collect();
            draft.block_hidden(&embeddings).unwrap()
        })
        .collect();
    for depth in 0..7 {
        let examples: Vec<_> = hidden
            .iter()
            .enumerate()
            .flat_map(|(wiring, hidden)| {
                hidden.chunks_exact(512).enumerate().map(move |(row, x)| {
                    (
                        x.to_vec(),
                        if wiring == 0 && row == depth {
                            1.0
                        } else {
                            -1.0
                        },
                    )
                })
            })
            .collect();
        for (channel, weight) in separator(&examples).into_iter().enumerate() {
            write_vector(
                &mut artifact,
                "bdraft2.sel.hp",
                depth * 512 + channel,
                weight * 32.0,
            );
        }
        for token in 0..32 {
            write_vector(&mut artifact, "bdraft2.sel.pred", token * 32 + depth, 1.0);
        }
        write_vector(
            &mut artifact,
            "bdraft2.sel.succ",
            (6 + depth) * 32 + depth,
            1.0,
        );
    }
    let mut proposer =
        DraftProposer::new(draft_from(artifact, blank_head()), 7, DraftChoice::Greedy).unwrap();
    let expected: Vec<_> = (6..=12).collect();
    assert_eq!(
        proposer
            .propose_with(&mut main, &[3, 7, 11, 0], 5, 7)
            .unwrap(),
        expected
    );
    assert_ne!(
        proposer
            .propose_with(&mut main, &[3, 7, 11, 0], 0, 7)
            .unwrap(),
        expected
    );
}

#[test]
fn sampled_hook_matches_direct_draft_with_frontier_seed_and_absolute_positions() {
    let mut main = model(main_bytes());
    main.set_capture(capture_options()).unwrap();
    main.prefill(&[3, 7, 11, 0, 9]).unwrap();
    let mut direct = draft_from(draft_bytes(), support::draft::shead(37));
    for row in main.captured_rows() {
        direct
            .push_context(
                row.position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    assert_eq!(direct.position(), 5);
    let checkpoint = main.snapshot().unwrap();
    let mut expected_runs = Vec::new();
    let greedy = direct
        .propose(13, |token| main.embedding(token), DraftChoice::Greedy)
        .unwrap()
        .tokens;
    for seed in [17, 0xabcdef] {
        main.restore(&checkpoint).unwrap();
        let choice = DraftChoice::sampled(seed);
        let direct_proposal = direct
            .propose(13, |token| main.embedding(token), choice)
            .unwrap();
        let expected = direct_proposal.tokens;
        let positioned: Vec<_> = direct_proposal
            .corrected
            .iter()
            .enumerate()
            .map(|(depth, candidates)| {
                yskn_ref::draft_head::choose(candidates, choice, 6 + depth as u64).unwrap()
            })
            .collect();
        assert_eq!(
            expected, positioned,
            "frontier 5 predicts at absolute positions 6..=12"
        );
        let mut proposer = DraftProposer::new(
            draft_from(draft_bytes(), support::draft::shead(37)),
            7,
            choice,
        )
        .unwrap();
        assert_eq!(
            proposer
                .propose_with(&mut main, &[3, 7, 11, 0, 9], 13, 7)
                .unwrap(),
            expected
        );
        expected_runs.push(expected);
    }
    assert_ne!(expected_runs[0], greedy);
    assert_ne!(expected_runs[0], expected_runs[1]);
}
