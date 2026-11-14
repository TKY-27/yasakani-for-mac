mod support;

use support::synthetic_model;
use yskn_format::Container;
use yskn_ref::Model;

fn mixed_model() -> Model {
    Model::from_container(
        Container::from_bytes(synthetic_model(
            137,
            &["linear_attention", "full_attention"],
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
fn restoring_prefix_then_suffix_matches_fresh_session_bit_for_bit() {
    let mut restored = mixed_model();
    let mut fresh = restored.new_session().unwrap();
    let prefix = [8, 21, 5];
    restored.prefill(&prefix).unwrap();
    let snapshot = restored.snapshot().unwrap();
    let prefix_digest = restored.state_digest();
    let prefix_logits = bits(restored.row_logits());
    for suffix in [&[7, 9][..], &[3, 4, 6][..], &[11][..]] {
        restored.prefill(&[19, 20, 22]).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.position(), prefix.len());
        assert_eq!(restored.state_digest(), prefix_digest);
        assert_eq!(bits(restored.row_logits()), vec![prefix_logits[2].clone()]);
        restored.prefill(suffix).unwrap();
        fresh.reset().unwrap();
        fresh.prefill(&prefix).unwrap();
        fresh.prefill(suffix).unwrap();
        assert_eq!(bits(restored.row_logits()), bits(fresh.row_logits()));
        assert_eq!(restored.state_digest(), fresh.state_digest());
        restored.restore(&snapshot).unwrap();
    }
}

#[test]
fn snapshots_reject_pending_other_sessions_reset_future_and_changed_prefix() {
    let mut model = mixed_model();
    let empty = model.snapshot().unwrap();
    model.prefill(&[1, 2]).unwrap();
    let prefix = model.snapshot().unwrap();
    model.run_provisional(&[3, 4]).unwrap();
    let pending_logits = bits(model.row_logits());
    let committed_digest = model.state_digest();
    assert!(model.snapshot().is_err());
    assert!(model.restore(&prefix).is_err());
    assert!(model.has_provisional_pass());
    assert_eq!(model.state_digest(), committed_digest);
    assert_eq!(bits(model.row_logits()), pending_logits);
    model.commit(1).unwrap();
    let later = model.snapshot().unwrap();
    model.restore(&prefix).unwrap();
    let digest = model.state_digest();
    assert!(model.restore(&later).is_err());
    assert_eq!(model.state_digest(), digest);
    let mut other = model.new_session().unwrap();
    other.prefill(&[1, 2]).unwrap();
    let digest = other.state_digest();
    assert!(other.restore(&prefix).is_err());
    assert_eq!(other.state_digest(), digest);
    model.restore(&empty).unwrap();
    assert_eq!(model.position(), 0);
    assert!(model.logits().is_none());
    model.prefill(&[1, 7, 3]).unwrap();
    let digest = model.state_digest();
    assert!(model.restore(&prefix).is_err());
    assert!(model.restore(&later).is_err());
    assert_eq!(model.state_digest(), digest);
    model.reset().unwrap();
    model.prefill(&[1, 2]).unwrap();
    let digest = model.state_digest();
    assert!(model.restore(&prefix).is_err());
    assert!(model.restore(&empty).is_err());
    assert_eq!(model.state_digest(), digest);
}

#[test]
fn snapshot_includes_only_committed_tokens_after_partial_commit() {
    let mut model = mixed_model();
    let mut fresh = model.new_session().unwrap();
    model.prefill(&[5]).unwrap();
    model.run_provisional(&[6, 7, 8]).unwrap();
    model.commit(2).unwrap();
    let snapshot = model.snapshot().unwrap();
    model.prefill(&[9, 10]).unwrap();
    model.restore(&snapshot).unwrap();
    model.prefill(&[11]).unwrap();
    fresh.prefill(&[5, 6, 7, 11]).unwrap();
    assert_eq!(model.state_digest(), fresh.state_digest());
    assert_eq!(
        bits(model.row_logits()),
        vec![bits(fresh.row_logits())[3].clone()]
    );
}
