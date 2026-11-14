mod support;

use std::io::Cursor;
use yskn_format::Container;
use yskn_ref::draft::{ContextFeatures, Draft};
use yskn_ref::draft_head::{DraftChoice, ReducedHead};
use yskn_ref::model::{CaptureOptions, CapturedRow, ModelDimensions};
use yskn_ref::speculation::{
    DraftProposer, GenerationError, GenerationLimits, Greedy, NoProposer, Proposer, Verifier,
    generate,
};
use yskn_ref::{Error, Result};

fn draft() -> Draft {
    let mut options = support::draft::options();
    options.capture_layers = vec![0, 1, 2, 3, 4];
    Draft::from_container(
        &mut Container::from_bytes(support::draft::container(23)).unwrap(),
        ReducedHead::read(&mut Cursor::new(support::draft::shead(37))).unwrap(),
        options,
    )
    .unwrap()
}
fn features(position: usize, token: u32) -> CapturedRow {
    CapturedRow {
        position,
        values: support::draft::features(position * 13 + token as usize),
    }
}

// Independent scripted verifier: row p always selects token p+1, while features
// encode both the position AND input token to distinguish rejected candidates.
#[derive(Clone)]
struct Main {
    identity: std::sync::Arc<()>,
    handed: usize,
    committed: Vec<CapturedRow>,
    pending: Vec<CapturedRow>,
    logits: Vec<Vec<f32>>,
    options: CaptureOptions,
    dimensions: ModelDimensions,
}
impl Main {
    fn new(proposer: &DraftProposer) -> Self {
        Self {
            identity: std::sync::Arc::new(()),
            handed: 0,
            committed: vec![features(0, 0), features(1, 1)],
            pending: vec![],
            logits: vec![],
            options: proposer.capture_options(),
            dimensions: ModelDimensions {
                hidden_width: 512,
                feed_forward_width: 512,
                vocabulary_size: 32,
                layer_count: 5,
                query_heads: 2,
                kv_heads: 1,
                attention_head_width: 256,
                epsilon: 1e-6,
            },
        }
    }
}
impl Verifier for Main {
    type Error = Error;
    fn context_identity(&self) -> Option<std::sync::Arc<()>> {
        Some(self.identity.clone())
    }
    fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        let rows = self.committed[self.handed..].to_vec();
        self.handed = self.committed.len();
        rows
    }
    fn position(&self) -> usize {
        self.committed.len()
    }
    fn context_capacity(&self) -> usize {
        32
    }
    fn captured_rows(&self) -> &[CapturedRow] {
        &self.committed[self.handed..]
    }
    fn provisional_captures(&self) -> &[CapturedRow] {
        &self.pending
    }
    fn capture_options(&self) -> Option<&CaptureOptions> {
        Some(&self.options)
    }
    fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        support::draft::embedding(token)
    }
    fn dimensions(&self) -> Option<&ModelDimensions> {
        Some(&self.dimensions)
    }
    fn run_provisional(&mut self, tokens: &[u32]) -> Result<()> {
        assert!(self.pending.is_empty());
        self.logits = tokens
            .iter()
            .enumerate()
            .map(|(row, _)| {
                let mut logits = vec![-10.0; 32];
                logits[self.position() + row + 1] = 10.0;
                logits
            })
            .collect();
        self.pending = tokens
            .iter()
            .enumerate()
            .map(|(row, &token)| features(self.position() + row, token))
            .collect();
        Ok(())
    }
    fn row_logits(&self, row: usize) -> Option<&[f32]> {
        self.logits.get(row).map(Vec::as_slice)
    }
    fn logits(&self) -> Option<&[f32]> {
        self.logits.last().map(Vec::as_slice)
    }
    fn commit(&mut self, rows: usize) -> Result<()> {
        self.committed.extend(self.pending.drain(..rows));
        self.pending.clear();
        Ok(())
    }
    fn discard(&mut self) {
        self.pending.clear();
    }
}

struct FailsOnSecondProposal {
    inner: DraftProposer,
    calls: usize,
    proposals: Vec<Vec<u32>>,
}
impl Proposer for FailsOnSecondProposal {
    fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
        unreachable!()
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &mut V) -> Result<()> {
        self.inner.synchronize(verifier)
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> Result<()> {
        self.inner.observe_committed(rows)
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &mut V,
        history: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.calls += 1;
        match self.calls {
            1 => Ok(vec![]),
            2 => Err(Error::InvalidInput("injected proposal failure".into())),
            _ => {
                let guesses = self.inner.propose_with(verifier, history, last, max)?;
                self.proposals.push(guesses.clone());
                Ok(guesses)
            }
        }
    }
}

fn disabled_draft() -> (Main, FailsOnSecondProposal) {
    let inner = DraftProposer::new(draft(), 3, DraftChoice::Greedy).unwrap();
    let mut main = Main::new(&inner);
    let mut expected = main.clone();
    let baseline = generate(
        &mut expected,
        &mut NoProposer,
        &mut Greedy,
        &[0, 1, 2],
        GenerationLimits::new(7, 32),
    )
    .unwrap();
    let mut proposer = FailsOnSecondProposal {
        inner,
        calls: 0,
        proposals: vec![],
    };
    let mut limits = GenerationLimits::new(7, 32);
    limits.max_draft_tokens = 3;
    let generated = generate(&mut main, &mut proposer, &mut Greedy, &[0, 1, 2], limits).unwrap();
    assert_eq!(generated.tokens, baseline.tokens);
    assert_eq!(main.committed, expected.committed);
    assert_eq!(main.logits(), expected.logits());
    assert!(main.pending.is_empty());
    assert_eq!(proposer.calls, 2);
    assert_eq!(proposer.inner.draft().position(), 3);
    assert_eq!(main.position(), 9);
    (main, proposer)
}

#[test]
fn disabled_draft_reuse_without_captures_returns_missing_committed_rows_error() {
    let (mut main, mut proposer) = disabled_draft();
    let before = main.clone();
    let error = generate(
        &mut main,
        &mut proposer.inner,
        &mut Greedy,
        &[9],
        GenerationLimits::new(4, 32),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        GenerationError::Proposer(Error::InvalidInput(message))
            if message == "draft context has missing committed rows"
    ));
    assert_eq!(main.committed, before.committed);
    assert_eq!(main.logits, before.logits);
    assert_eq!(main.pending, before.pending);
    assert_eq!(main.handed, before.handed);
}

#[test]
fn disabled_draft_reuse_with_complete_prompt_window_reseeds_and_proposes_correctly() {
    let (mut main, mut proposer) = disabled_draft();
    let prompt: Vec<_> = (9..17).collect();
    main.run_provisional(&prompt).unwrap();
    main.commit(prompt.len()).unwrap();
    let mut direct = draft();
    for row in &main.committed {
        direct
            .push_context(
                row.position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    let expected_guesses = direct
        .propose(17, support::draft::embedding, DraftChoice::Greedy)
        .unwrap()
        .tokens[..3]
        .to_vec();
    let mut expected = main.clone();
    let baseline = generate(
        &mut expected,
        &mut NoProposer,
        &mut Greedy,
        &[17],
        GenerationLimits::new(4, 32),
    )
    .unwrap();
    let mut limits = GenerationLimits::new(4, 32);
    limits.max_draft_tokens = 3;
    let generated = generate(&mut main, &mut proposer, &mut Greedy, &[17], limits).unwrap();
    assert_eq!(proposer.proposals[0], expected_guesses);
    assert_eq!(generated.tokens, baseline.tokens);
    assert_eq!(main.committed, expected.committed);
    assert_eq!(main.logits(), expected.logits());
    for row in &main.committed[17..] {
        direct
            .push_context(
                row.position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    assert_eq!(proposer.inner.draft().snapshot(), direct.snapshot());
}

#[test]
fn failed_partial_synchronize_binds_identity_before_first_ring_push() {
    let mut proposer = DraftProposer::new(draft(), 3, DraftChoice::Greedy).unwrap();
    let mut main = Main::new(&proposer);
    let mut other = main.clone();
    other.identity = std::sync::Arc::new(());
    main.committed[1].values.clear();
    let error = proposer.synchronize(&mut main).unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput(message) if message == "draft context position/shape mismatch")
    );
    assert_eq!(proposer.draft().position(), 1);
    assert_eq!(main.handed, 0);
    let before = proposer.draft().snapshot();
    let error = proposer.synchronize(&mut other).unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput(message) if message == "main context restored or reset without the draft")
    );
    assert_eq!(proposer.draft().snapshot(), before);
    assert_eq!(other.handed, 0);
    main.committed[1] = features(1, 1);
    proposer.synchronize(&mut main).unwrap();
    let mut direct = draft();
    for row in &main.committed {
        direct
            .push_context(
                row.position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    assert_eq!(proposer.draft().snapshot(), direct.snapshot());
}

#[test]
fn unbound_observation_and_prefilled_ring_cannot_adopt_main_context() {
    let mut proposer = DraftProposer::new(draft(), 3, DraftChoice::Greedy).unwrap();
    let before = proposer.draft().snapshot();
    let error = proposer.observe_committed(&[features(0, 0)]).unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput(message) if message == "draft context identity is not bound; synchronize first")
    );
    assert_eq!(proposer.draft().snapshot(), before);

    let mut prefilled = draft();
    prefilled
        .push_context(
            0,
            ContextFeatures {
                values: &features(0, 0).values,
            },
        )
        .unwrap();
    let mut proposer = DraftProposer::new(prefilled, 3, DraftChoice::Greedy).unwrap();
    let before = proposer.draft().snapshot();
    let mut main = Main::new(&proposer);
    let error = proposer.synchronize(&mut main).unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput(message) if message == "main context restored or reset without the draft")
    );
    assert_eq!(proposer.draft().snapshot(), before);
    assert_eq!(main.handed, 0);
}

struct ScriptedDraft {
    inner: DraftProposer,
    expected: Draft,
    calls: usize,
    delivered: Vec<usize>,
}
impl Proposer for ScriptedDraft {
    fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
        unreachable!()
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &mut V) -> Result<()> {
        self.inner.synchronize(verifier)
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &mut V,
        _: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.inner.synchronize(verifier)?;
        let (position, script): (usize, &[u32]) = match self.calls {
            0 => (2, &[3, 4, 30]), // two matches, then mismatch
            1 => (5, &[30, 31]),   // no matches
            2 => (6, &[7, 8, 9]),  // all match, then terminal frontier 10
            _ => panic!("unexpected pass"),
        };
        assert_eq!(last as usize, position);
        assert_eq!(
            self.inner.ring_positions(),
            position.saturating_sub(8)..position
        );
        assert_eq!(self.inner.draft().snapshot(), self.expected.snapshot());
        self.calls += 1;
        Ok(script.iter().copied().take(max).collect())
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> Result<()> {
        self.inner.observe_committed(rows)?;
        for row in rows {
            assert_eq!(
                row.values,
                features(row.position, row.position as u32).values
            );
            self.expected.push_context(
                row.position,
                ContextFeatures {
                    values: &row.values,
                },
            )?;
            self.delivered.push(row.position);
        }
        Ok(())
    }
}

#[test]
fn ring_before_each_proposal_excludes_rejections_and_frontier_and_flushes_at_stop() {
    let inner = DraftProposer::new(draft(), 3, DraftChoice::Greedy).unwrap();
    let mut main = Main {
        identity: std::sync::Arc::new(()),
        handed: 0,
        committed: vec![features(0, 0), features(1, 1)],
        pending: vec![],
        logits: vec![],
        options: inner.capture_options(),
        dimensions: ModelDimensions {
            hidden_width: 512,
            feed_forward_width: 512,
            vocabulary_size: 32,
            layer_count: 5,
            query_heads: 2,
            kv_heads: 1,
            attention_head_width: 256,
            epsilon: 1e-6,
        },
    };
    let mut expected = draft();
    for row in &main.committed {
        expected
            .push_context(
                row.position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    let mut proposer = ScriptedDraft {
        inner,
        expected,
        calls: 0,
        delivered: vec![],
    };
    let mut limits = GenerationLimits::new(8, 32);
    limits.max_draft_tokens = 3;
    let generated = generate(&mut main, &mut proposer, &mut Greedy, &[0, 1, 2], limits).unwrap();
    assert_eq!(generated.tokens, [3, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(generated.statistics.tokens_confirmed_per_pass, [3, 1, 4]);
    assert_eq!(proposer.delivered, (2..10).collect::<Vec<_>>());
    assert_eq!(proposer.inner.ring_positions(), 2..10);
    assert_eq!(
        proposer.inner.draft().snapshot(),
        proposer.expected.snapshot()
    );
    assert_eq!(main.position(), 10);
}

#[test]
fn long_prefill_ring_matches_every_prompt_row_pushed_and_snapshot_reset_ownership() {
    let mut inner = DraftProposer::new(draft(), 3, DraftChoice::Greedy).unwrap();
    let mut expected = draft();
    for position in 0..19 {
        let row = features(position, position as u32);
        expected
            .push_context(
                position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    let main = Main {
        identity: std::sync::Arc::new(()),
        handed: 0,
        committed: (11..19).map(|p| features(p, p as u32)).collect(),
        pending: vec![],
        logits: vec![],
        options: inner.capture_options(),
        dimensions: ModelDimensions {
            hidden_width: 512,
            feed_forward_width: 512,
            vocabulary_size: 32,
            layer_count: 5,
            query_heads: 2,
            kv_heads: 1,
            attention_head_width: 256,
            epsilon: 1e-6,
        },
    };
    // This fixture exposes a retained suffix, so its absolute position is separate.
    struct Suffix<'a>(&'a Main);
    impl Verifier for Suffix<'_> {
        type Error = Error;
        fn context_identity(&self) -> Option<std::sync::Arc<()>> {
            self.0.context_identity()
        }
        fn position(&self) -> usize {
            19
        }
        fn context_capacity(&self) -> usize {
            32
        }
        fn captured_rows(&self) -> &[CapturedRow] {
            &self.0.committed
        }
        fn capture_options(&self) -> Option<&CaptureOptions> {
            Some(&self.0.options)
        }
        fn dimensions(&self) -> Option<&ModelDimensions> {
            Some(&self.0.dimensions)
        }
        fn run_provisional(&mut self, _: &[u32]) -> Result<()> {
            unreachable!()
        }
        fn row_logits(&self, _: usize) -> Option<&[f32]> {
            None
        }
        fn logits(&self) -> Option<&[f32]> {
            None
        }
        fn commit(&mut self, _: usize) -> Result<()> {
            unreachable!()
        }
        fn discard(&mut self) {}
    }
    inner.synchronize(&mut Suffix(&main)).unwrap();
    assert_eq!(inner.draft().snapshot(), expected.snapshot());
    assert_eq!(inner.ring_positions(), 11..19);
    // A wide accepted pass hands over only its retained window, just as prefill does.
    for position in 19..30 {
        let row = features(position, position as u32);
        expected
            .push_context(
                position,
                ContextFeatures {
                    values: &row.values,
                },
            )
            .unwrap();
    }
    inner
        .observe_committed(&(22..30).map(|p| features(p, p as u32)).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(inner.draft().snapshot(), expected.snapshot());
    let saved = inner.snapshot();
    inner.reset();
    assert!(inner.restore(&saved).is_err());
}

#[test]
fn combined_policy_uses_first_nonempty_source_and_keeps_neural_context_current() {
    use yskn_ref::speculation::{CombinedProposer, NgramProposer, ProposalOrder};
    let mut neural = DraftProposer::new(draft(), 3, DraftChoice::Greedy).unwrap();
    let mut main = Main {
        identity: std::sync::Arc::new(()),
        handed: 0,
        committed: vec![features(0, 0), features(1, 1)],
        pending: vec![],
        logits: vec![],
        options: neural.capture_options(),
        dimensions: ModelDimensions {
            hidden_width: 512,
            feed_forward_width: 512,
            vocabulary_size: 32,
            layer_count: 5,
            query_heads: 2,
            kv_heads: 1,
            attention_head_width: 256,
            epsilon: 1e-6,
        },
    };
    let expected = neural.propose_with(&mut main, &[1, 2, 1], 2, 3).unwrap();
    let mut ngram_first = CombinedProposer::new(
        NgramProposer::new(2, 3).unwrap(),
        neural,
        ProposalOrder::NgramFirst,
    );
    assert_eq!(
        ngram_first
            .propose_with(&mut main, &[1, 2, 1], 2, 3)
            .unwrap(),
        [1, 2]
    );
    assert_eq!(ngram_first.draft().draft().position(), 2);
    assert_eq!(
        ngram_first
            .propose_with(&mut main, &[3, 4, 5], 2, 3)
            .unwrap(),
        expected
    );
    main.handed = 0; // An independent verifier copy supplies its prompt to the new draft.
    let neural = DraftProposer::new(draft(), 3, DraftChoice::Greedy).unwrap();
    let mut draft_first = CombinedProposer::new(
        NgramProposer::new(2, 3).unwrap(),
        neural,
        ProposalOrder::DraftFirst,
    );
    assert_eq!(
        draft_first
            .propose_with(&mut main, &[1, 2, 1], 2, 3)
            .unwrap(),
        expected
    );
}
