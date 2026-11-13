mod support;

use support::{
    synthetic_model, synthetic_model_with_dense_embeddings, synthetic_model_with_dimensions,
};
use yskn_format::Container;
use yskn_ref::{Error, Model, Sampler, SamplingParameters, SamplingVersion};

fn recurrent_model(seed: u64) -> Model {
    Model::from_container(
        Container::from_bytes(synthetic_model(
            seed,
            &["linear_attention", "linear_attention"],
        ))
        .unwrap(),
    )
    .unwrap()
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn multirow_pass_matches_single_rows_bit_for_bit() {
    let mut batched = recurrent_model(17);
    let mut individual = batched.new_session().unwrap();
    let tokens = [3, 17, 9, 1, 23];
    batched.run_provisional(&tokens).unwrap();
    let expected = bits(batched.row_logits());
    assert_eq!(batched.position(), 0);
    let mut actual = Vec::new();
    for token in tokens {
        individual.prefill(&[token]).unwrap();
        actual.extend(bits(individual.row_logits()));
    }
    assert_eq!(expected, actual);
    // Nonconstant, nonzero logits prevent a disconnected/all-zero implementation passing.
    assert!(expected[0].windows(2).any(|pair| pair[0] != pair[1]));
    assert!(expected.windows(2).any(|pair| pair[0] != pair[1]));
    batched.commit(tokens.len()).unwrap();
    batched.prefill(&[31]).unwrap();
    individual.prefill(&[31]).unwrap();
    assert_eq!(bits(batched.row_logits()), bits(individual.row_logits()));
}

#[test]
fn prefill_chunk_boundaries_do_not_change_logits_or_future_state() {
    let mut whole = recurrent_model(29);
    let mut chunks = whole.new_session().unwrap();
    let tokens = [11, 6, 30, 19, 2, 43, 7];
    whole.prefill(&tokens).unwrap();
    let expected = bits(whole.row_logits());
    let mut actual = Vec::new();
    for chunk in [&tokens[..2], &tokens[2..3], &tokens[3..]] {
        chunks.prefill(chunk).unwrap();
        actual.extend(bits(chunks.row_logits()));
    }
    assert_eq!(expected, actual);
    whole.prefill(&[5, 12]).unwrap();
    chunks.prefill(&[5, 12]).unwrap();
    assert_eq!(bits(whole.row_logits()), bits(chunks.row_logits()));
}

#[test]
fn provisional_prefix_commit_matches_committing_pass() {
    let mut provisional = recurrent_model(41);
    let mut direct = provisional.new_session().unwrap();
    provisional.prefill(&[8, 21]).unwrap();
    direct.prefill(&[8, 21]).unwrap();
    let tokens = [7, 11, 5, 27, 3];
    provisional.run_provisional(&tokens).unwrap();
    assert_eq!(provisional.position(), 2);
    assert!(provisional.prefill(&[1]).is_err());
    assert!(provisional.commit(tokens.len() + 1).is_err());
    assert_eq!(provisional.position(), 2);
    provisional.commit(3).unwrap();
    direct.prefill(&tokens[..3]).unwrap();
    assert_eq!(provisional.position(), 5);
    assert_eq!(bits(provisional.row_logits()), bits(direct.row_logits()));
    provisional.prefill(&[10, 14]).unwrap();
    direct.prefill(&[10, 14]).unwrap();
    assert_eq!(bits(provisional.row_logits()), bits(direct.row_logits()));
    let committed = bits(provisional.row_logits());
    provisional.run_provisional(&[4, 19]).unwrap();
    provisional.commit(0).unwrap();
    assert_eq!(provisional.position(), 7);
    assert_eq!(
        bits(provisional.row_logits()),
        vec![committed.last().unwrap().clone()]
    );
    provisional.prefill(&[17]).unwrap();
    direct.prefill(&[17]).unwrap();
    assert_eq!(bits(provisional.row_logits()), bits(direct.row_logits()));
}

#[test]
fn synthetic_logits_sample_repeatably_by_seed_and_absolute_position() {
    let mut model = recurrent_model(53);
    model.prefill(&[8, 7, 6]).unwrap();
    for version in [SamplingVersion::Gumbel, SamplingVersion::InverseCumulative] {
        let sampler = Sampler::Sampled(SamplingParameters {
            seed: 0x1234_5678_9abc_def0,
            version,
            ..SamplingParameters::default()
        });
        let first: Vec<_> = model
            .row_logits()
            .iter()
            .enumerate()
            .map(|(row, logits)| sampler.sample(logits, row as u64 + 1).unwrap())
            .collect();
        let second: Vec<_> = model
            .row_logits()
            .iter()
            .enumerate()
            .rev()
            .map(|(row, logits)| sampler.sample(logits, row as u64 + 1).unwrap())
            .rev()
            .collect();
        assert_eq!(first, second);
    }
}

#[test]
fn invalid_token_pass_is_atomic_and_reset_reproduces_the_initial_state() {
    let mut model = recurrent_model(67);
    model.prefill(&[3]).unwrap();
    let expected = bits(model.row_logits());
    assert!(
        model
            .prefill(&[4, model.dimensions().vocabulary_size as u32])
            .is_err()
    );
    assert_eq!(model.position(), 1);
    assert_eq!(bits(model.row_logits()), expected);
    assert!(!model.has_provisional_pass());
    model.reset().unwrap();
    assert_eq!(model.position(), 0);
    assert!(model.logits().is_none());
    model.prefill(&[3]).unwrap();
    assert_eq!(bits(model.row_logits()), expected);
}

#[test]
fn mixed_model_reports_the_documented_gap() {
    let bytes = synthetic_model(
        79,
        &[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    );
    let result = Model::from_container(Container::from_bytes(bytes).unwrap());
    assert!(matches!(result, Err(Error::SpecGap("ARCH-01"))));
}

#[test]
fn header_dimensions_and_padded_vocabulary_are_used_in_execution() {
    let bytes = synthetic_model_with_dimensions(89, &["linear_attention"], 1024, 1024, 65);
    let mut model = Model::from_container(Container::from_bytes(bytes).unwrap()).unwrap();
    assert_eq!(model.dimensions().hidden_width, 1024);
    assert_eq!(model.dimensions().feed_forward_width, 1024);
    assert_eq!(model.dimensions().vocabulary_size, 65);
    model.prefill(&[64, 0]).unwrap();
    assert_eq!(model.row_logits().len(), 2);
    assert!(
        model
            .row_logits()
            .iter()
            .all(|row| row.len() == 65 && row.iter().all(|value| value.is_finite()))
    );
    assert!(model.prefill(&[65]).is_err());
    assert_eq!(model.position(), 2);
}

#[test]
fn plain_bfloat_embeddings_use_the_same_model_path() {
    let bytes = synthetic_model_with_dense_embeddings(97, &["linear_attention"]);
    let mut model = Model::from_container(Container::from_bytes(bytes).unwrap()).unwrap();
    let mut split = model.new_session().unwrap();
    model.prefill(&[0, 127]).unwrap();
    let expected = bits(model.row_logits());
    split.prefill(&[0]).unwrap();
    assert_eq!(bits(split.row_logits()), expected[..1]);
    split.prefill(&[127]).unwrap();
    assert_eq!(bits(split.row_logits()), expected[1..]);
}

#[test]
fn model_vectors_reject_nonfinite_values_before_execution() {
    let mut bytes = synthetic_model(101, &["linear_attention"]);
    let container = Container::from_bytes(bytes.clone()).unwrap();
    let offset = container
        .header()
        .vectors
        .iter()
        .find(|vector| vector.name == "layers.0.input_layernorm")
        .unwrap()
        .offset as usize;
    bytes[offset..offset + 2].copy_from_slice(&0x7fc1_u16.to_le_bytes());
    let result = Model::from_container(Container::from_bytes(bytes).unwrap());
    assert!(matches!(result, Err(Error::NonFinite("model vector"))));
}

#[test]
fn model_folder_manifest_rounds_context_capacity_and_selects_weights() {
    let folder = std::env::temp_dir().join(format!("yskn-ref-folder-{}", std::process::id()));
    std::fs::create_dir(&folder).unwrap();
    let weights_path = folder.join("fixture.yskn");
    std::fs::write(&weights_path, synthetic_model(103, &["linear_attention"])).unwrap();
    std::fs::write(
        folder.join("yasakani.json"),
        r#"{"files":{"weights":"fixture.yskn"},"max_ctx":1}"#,
    )
    .unwrap();
    let result = Model::load_folder(&folder);
    // Loading owns its compressed bytes, so cleanup also checks no borrowed file storage.
    std::fs::remove_dir_all(&folder).unwrap();
    let mut model = result.unwrap();
    assert_eq!(model.context_capacity(), 64);
    assert!(model.prefill(&[0; 65]).is_err());
    assert_eq!(model.position(), 0);
    model.prefill(&[0]).unwrap();
    assert_eq!(model.position(), 1);
}

#[test]
#[ignore = "ARCH-01, MATH-01 and ATT-MASK-01: rotary-width header rule, portable exp2 and masked operands are unspecified"]
fn mixed_model_row_and_chunk_invariants() {
    let bytes = synthetic_model(
        83,
        &[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    );
    let mut batched = Model::from_container(Container::from_bytes(bytes).unwrap()).unwrap();
    let mut individual = batched.new_session().unwrap();
    let tokens = [4, 3, 7, 2, 6];
    batched.run_provisional(&tokens).unwrap();
    let expected = bits(batched.row_logits());
    let mut actual = Vec::new();
    for token in tokens {
        individual.prefill(&[token]).unwrap();
        actual.extend(bits(individual.row_logits()));
    }
    assert_eq!(expected, actual);
    batched.commit(3).unwrap();
    individual.reset().unwrap();
    individual.prefill(&tokens[..3]).unwrap();
    batched.prefill(&[19]).unwrap();
    individual.prefill(&[19]).unwrap();
    assert_eq!(bits(batched.row_logits()), bits(individual.row_logits()));
}
