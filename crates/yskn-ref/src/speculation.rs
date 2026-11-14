//! Lossless, straight-chain speculative generation shared by both model implementations.
//!
//! The last token in `prompt_tail` is pending: the verifier has consumed every
//! earlier token, but has not consumed that last token. The final selected token
//! is likewise returned without being consumed. This convention makes stopping
//! after any token (including EOS or cancellation) identical to one-row decoding.
use std::fmt;
use std::time::{Duration, Instant};

use crate::model::{CaptureOptions, CapturedRow, ModelDimensions};
use crate::{Error, Model, Result, Sampler};

pub use crate::draft_proposer::{
    CombinedProposer, DraftProposer, DraftProposerSnapshot, ProposalOrder, SessionSnapshot,
};

/// A causal verifier. Failed passes are atomic; commit retains exactly its first
/// `rows` input rows, and discard restores the committed prediction.
pub trait Verifier {
    type Error;
    fn position(&self) -> usize;
    fn context_capacity(&self) -> usize;
    fn max_pass_width(&self) -> usize {
        usize::MAX
    }
    fn run_provisional(&mut self, tokens: &[u32]) -> std::result::Result<(), Self::Error>;
    fn row_logits(&self, row: usize) -> Option<&[f32]>;
    fn logits(&self) -> Option<&[f32]>;
    fn commit(&mut self, rows: usize) -> std::result::Result<(), Self::Error>;
    fn discard(&mut self);
    /// Changes on every successful main-only restore/reset. Drafts require it.
    fn context_identity(&self) -> Option<std::sync::Arc<()>> {
        None
    }
    fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        Vec::new()
    }
    fn captured_rows(&self) -> &[CapturedRow] {
        &[]
    }
    fn provisional_captures(&self) -> &[CapturedRow] {
        &[]
    }
    fn capture_options(&self) -> Option<&CaptureOptions> {
        None
    }
    fn dimensions(&self) -> Option<&ModelDimensions> {
        None
    }
    fn embedding(&self, _token: u32) -> Result<Vec<f32>> {
        Err(Error::InvalidInput(
            "verifier does not expose embeddings".into(),
        ))
    }
}

impl Verifier for Model {
    type Error = Error;
    fn context_identity(&self) -> Option<std::sync::Arc<()>> {
        Some(self.context_identity())
    }
    fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        self.take_captured_rows()
    }
    fn position(&self) -> usize {
        self.position()
    }
    fn context_capacity(&self) -> usize {
        self.context_capacity()
    }
    fn run_provisional(&mut self, tokens: &[u32]) -> Result<()> {
        self.run_provisional(tokens).map(|_| ())
    }
    fn row_logits(&self, row: usize) -> Option<&[f32]> {
        self.row_logits().get(row).map(Vec::as_slice)
    }
    fn logits(&self) -> Option<&[f32]> {
        self.logits()
    }
    fn commit(&mut self, rows: usize) -> Result<()> {
        self.commit(rows)
    }
    fn discard(&mut self) {
        self.discard_provisional();
    }
    fn captured_rows(&self) -> &[CapturedRow] {
        self.captured_rows()
    }
    fn provisional_captures(&self) -> &[CapturedRow] {
        self.provisional_captures()
    }
    fn capture_options(&self) -> Option<&CaptureOptions> {
        Some(self.capture_options())
    }
    fn dimensions(&self) -> Option<&ModelDimensions> {
        Some(self.dimensions())
    }
    fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        self.embedding(token)
    }
}

/// Verifiers whose owned checkpoints can be saved together with a draft ring.
pub trait CheckpointVerifier: Verifier {
    type Snapshot;
    fn snapshot(&self) -> std::result::Result<Self::Snapshot, Self::Error>;
    fn restore(&mut self, snapshot: &Self::Snapshot) -> std::result::Result<(), Self::Error>;
}
impl CheckpointVerifier for Model {
    type Snapshot = crate::Snapshot;
    fn snapshot(&self) -> Result<Self::Snapshot> {
        self.snapshot()
    }
    fn restore(&mut self, snapshot: &Self::Snapshot) -> Result<()> {
        self.restore(snapshot)
    }
}

/// Selection must depend only on logits and absolute prediction position, not
/// on pass grouping. A prediction position is its input row's position plus one.
pub trait Selector {
    fn select(&mut self, logits: &[f32], prediction_position: u64) -> Result<u32>;
}

impl Selector for Sampler {
    fn select(&mut self, logits: &[f32], prediction_position: u64) -> Result<u32> {
        self.sample(logits, prediction_position)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Greedy;
impl Selector for Greedy {
    fn select(&mut self, logits: &[f32], _prediction_position: u64) -> Result<u32> {
        crate::output_head_sampler::greedy(logits)
    }
}

/// Guesses follow `last_selected`; `committed_history` excludes that pending
/// token. History can be a known suffix of the committed session's full history.
/// Return only valid vocabulary IDs, and at most `max_guesses` tokens.
pub trait Proposer {
    fn propose(
        &mut self,
        committed_history: &[u32],
        last_selected: u32,
        max_guesses: usize,
    ) -> Vec<u32>;
    /// Observe the retained prompt suffix before generation or a direct proposal.
    fn synchronize<V: Verifier>(&mut self, _verifier: &mut V) -> Result<()> {
        Ok(())
    }
    /// Called after commit with its undelivered captures.
    fn observe_committed(&mut self, _rows: &[CapturedRow]) -> Result<()> {
        Ok(())
    }
    /// Fallible, verifier-aware entry point; token-only proposers keep their API.
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &mut V,
        history: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.synchronize(verifier)?;
        Ok(self.propose(history, last, max))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoProposer;
impl Proposer for NoProposer {
    fn propose(&mut self, _history: &[u32], _last: u32, _max: usize) -> Vec<u32> {
        Vec::new()
    }
}

/// Offers the same fixed script from its beginning on each pass.
#[derive(Clone, Debug)]
pub struct ScriptProposer {
    script: Vec<u32>,
}
impl ScriptProposer {
    pub fn new(script: Vec<u32>) -> Self {
        Self { script }
    }
}
impl Proposer for ScriptProposer {
    fn propose(&mut self, _history: &[u32], _last: u32, max: usize) -> Vec<u32> {
        self.script.iter().copied().take(max).collect()
    }
}

/// Copies the continuation of the most recent earlier occurrence of the last
/// `n` known tokens. Only already-known tokens are copied (no recursive guesses).
#[derive(Clone, Debug)]
pub struct NgramProposer {
    n: usize,
    k: usize,
}
impl NgramProposer {
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidInput("ngram length must be positive".into()));
        }
        Ok(Self { n, k })
    }
}
impl Proposer for NgramProposer {
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        let max = max.min(self.k);
        if max == 0 || history.len() < self.n {
            return Vec::new();
        }
        // Virtual concatenation avoids allocating a copy of the growing history.
        let known_len = history.len() + 1;
        let token = |index: usize| {
            if index == history.len() {
                last
            } else {
                history[index]
            }
        };
        let suffix_start = known_len - self.n;
        for start in (0..suffix_start).rev() {
            if (0..self.n).all(|offset| token(start + offset) == token(suffix_start + offset)) {
                let continuation = start + self.n;
                return (continuation..known_len).take(max).map(token).collect();
            }
        }
        Vec::new()
    }
}

pub struct GenerationLimits<'a> {
    pub max_new_tokens: usize,
    pub max_draft_tokens: usize,
    /// An additional caller limit, clamped to the verifier's own capacity.
    pub context_capacity: usize,
    pub eos_ids: &'a [u32],
    /// Called for every returned token, including EOS. Return false to cancel.
    pub on_token: Option<&'a mut dyn FnMut(u32) -> bool>,
}
impl GenerationLimits<'_> {
    pub fn new(max_new_tokens: usize, context_capacity: usize) -> Self {
        Self {
            max_new_tokens,
            max_draft_tokens: 0,
            context_capacity,
            eos_ids: &[],
            on_token: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxNewTokens,
    ContextCapacity,
    EndOfSequence,
    Cancelled,
}

#[derive(Clone, Debug, Default)]
pub struct GenerationStatistics {
    pub passes: usize,
    pub rows_verified: usize,
    pub tokens_confirmed_per_pass: Vec<usize>,
    pub guesses_offered: usize,
    /// Guesses equal to selected tokens, including a matching terminal token.
    pub guesses_accepted: usize,
    /// End-to-end pass time, including selection, streaming callback and commit.
    pub pass_durations: Vec<Duration>,
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub statistics: GenerationStatistics,
    pub stop_reason: StopReason,
}

#[derive(Debug)]
pub enum GenerationError<E> {
    Verifier(E),
    Selector(Error),
    Proposer(Error),
    InvalidInput(&'static str),
}
impl<E: fmt::Display> fmt::Display for GenerationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verifier(error) => write!(f, "verifier: {error}"),
            Self::Selector(error) => write!(f, "selector: {error}"),
            Self::Proposer(error) => write!(f, "proposer: {error}"),
            Self::InvalidInput(message) => write!(f, "generation: {message}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for GenerationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Verifier(error) => Some(error),
            Self::Selector(error) | Self::Proposer(error) => Some(error),
            Self::InvalidInput(_) => None,
        }
    }
}

/// Generate from a known history tail ending in one unconsumed input token.
///
/// The caller prefills the prompt except its final token before calling. A pass
/// verifies `[pending, guesses..]`, selecting each output from the preceding row.
/// Only selected matches allow the next row to be read. Commit consumes one row
/// per returned output, leaving the final output pending even at a stop boundary.
pub fn generate<V: Verifier, P: Proposer, S: Selector>(
    verifier: &mut V,
    proposer: &mut P,
    selector: &mut S,
    prompt_tail: &[u32],
    mut limits: GenerationLimits<'_>,
) -> std::result::Result<GenerationResult, GenerationError<V::Error>> {
    let (&initial_pending, history) =
        prompt_tail
            .split_last()
            .ok_or(GenerationError::InvalidInput(
                "prompt tail must not be empty",
            ))?;
    let mut pending = initial_pending;
    if history.len() > verifier.position() {
        return Err(GenerationError::InvalidInput(
            "prompt history exceeds committed position",
        ));
    }
    if verifier.max_pass_width() == 0 {
        return Err(GenerationError::InvalidInput(
            "maximum pass width must be positive",
        ));
    }
    let mut history = history.to_vec();
    let capacity = limits.context_capacity.min(verifier.context_capacity());
    let mut result = GenerationResult {
        tokens: Vec::new(),
        statistics: GenerationStatistics::default(),
        stop_reason: StopReason::MaxNewTokens,
    };
    proposer
        .synchronize(verifier)
        .map_err(GenerationError::Proposer)?;
    let mut drafting = true;
    while result.tokens.len() < limits.max_new_tokens {
        let position = verifier.position();
        let room = capacity.saturating_sub(position);
        if room == 0 {
            result.stop_reason = StopReason::ContextCapacity;
            break;
        }
        let remaining = limits.max_new_tokens - result.tokens.len();
        let width = room.min(remaining).min(verifier.max_pass_width());
        let mut guesses = if drafting {
            match proposer.propose_with(
                verifier,
                &history,
                pending,
                limits.max_draft_tokens.min(width - 1),
            ) {
                Ok(guesses) => guesses,
                Err(error) if result.tokens.is_empty() => {
                    return Err(GenerationError::Proposer(error));
                }
                Err(_) => {
                    drafting = false;
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        let vocabulary = verifier
            .dimensions()
            .map(|dimensions| dimensions.vocabulary_size)
            .or_else(|| verifier.logits().map(<[f32]>::len));
        let invalid_guesses = if guesses.len() > limits.max_draft_tokens.min(width - 1) {
            Some("proposer exceeded the requested guess limit")
        } else if vocabulary.is_some_and(|size| guesses.iter().any(|&id| id as usize >= size)) {
            Some("proposer returned a token outside the vocabulary")
        } else {
            None
        };
        if let Some(message) = invalid_guesses {
            if result.tokens.is_empty() {
                return Err(GenerationError::InvalidInput(message));
            }
            drafting = false;
            guesses.clear();
        }
        let mut rows = Vec::with_capacity(guesses.len() + 1);
        rows.push(pending);
        rows.extend_from_slice(&guesses);
        let started = Instant::now();
        verifier
            .run_provisional(&rows)
            .map_err(GenerationError::Verifier)?;
        result.statistics.passes += 1;
        result.statistics.rows_verified += rows.len();
        result.statistics.guesses_offered += guesses.len();
        let before = result.tokens.len();
        let mut stopped = false;
        for row in 0..rows.len() {
            let Some(logits) = verifier.row_logits(row) else {
                verifier.discard();
                return Err(GenerationError::InvalidInput(
                    "verifier omitted a requested logit row",
                ));
            };
            let prediction_position = u64::try_from(position)
                .ok()
                .and_then(|value| value.checked_add(row as u64 + 1));
            let Some(prediction_position) = prediction_position else {
                verifier.discard();
                return Err(GenerationError::InvalidInput(
                    "prediction position overflow",
                ));
            };
            let selected = match selector.select(logits, prediction_position) {
                Ok(token) => token,
                Err(error) => {
                    verifier.discard();
                    return Err(GenerationError::Selector(error));
                }
            };
            result.tokens.push(selected);
            let matched = guesses.get(row).is_some_and(|&guess| guess == selected);
            result.statistics.guesses_accepted += usize::from(matched);
            let keep_going = limits
                .on_token
                .as_mut()
                .is_none_or(|callback| callback(selected));
            if limits.eos_ids.contains(&selected) {
                result.stop_reason = StopReason::EndOfSequence;
                stopped = true;
            } else if !keep_going {
                result.stop_reason = StopReason::Cancelled;
                stopped = true;
            }
            if stopped || !matched {
                break;
            }
        }
        let confirmed = result.tokens.len() - before;
        let captures = verifier.provisional_captures();
        if !captures.is_empty() && captures.len() < confirmed {
            verifier.discard();
            return Err(GenerationError::InvalidInput(
                "verifier omitted captured rows",
            ));
        }
        if let Err(error) = verifier.commit(confirmed) {
            verifier.discard();
            return Err(GenerationError::Verifier(error));
        }
        let committed_captures = verifier.take_captured_rows();
        if drafting && proposer.observe_committed(&committed_captures).is_err() {
            // A committed output cannot be rolled back or lost to a draft failure.
            drafting = false;
        }
        result.statistics.tokens_confirmed_per_pass.push(confirmed);
        result.statistics.pass_durations.push(started.elapsed());
        history.push(pending);
        let outputs = &result.tokens[before..];
        history.extend_from_slice(&outputs[..outputs.len() - 1]);
        pending = outputs[outputs.len() - 1];
        if stopped {
            break;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SamplingParameters;

    // This verifier is a deterministic sequence-to-logits function. The oracle
    // below drives that function directly, without provisional/commit or generate.
    fn prediction(prefix: &[u32]) -> Vec<f32> {
        let state = prefix.iter().fold(19_u64, |state, &token| {
            state.wrapping_mul(31).wrapping_add(u64::from(token) + 7)
        });
        (0..8)
            .map(|token| ((state.rotate_right(token * 7) % 113) as f32 - 50.0) / 23.0)
            .collect()
    }

    #[derive(Clone, Debug)]
    struct SequenceVerifier {
        committed: Vec<u32>,
        pending: Option<Vec<u32>>,
        rows: Vec<Vec<f32>>,
        last: Option<Vec<f32>>,
        capacity: usize,
        width: usize,
    }
    impl SequenceVerifier {
        fn new(prefix: &[u32], capacity: usize, width: usize) -> Self {
            Self {
                committed: prefix.to_vec(),
                pending: None,
                rows: Vec::new(),
                last: (!prefix.is_empty()).then(|| prediction(prefix)),
                capacity,
                width,
            }
        }
    }
    impl Verifier for SequenceVerifier {
        type Error = Error;
        fn position(&self) -> usize {
            self.committed.len()
        }
        fn context_capacity(&self) -> usize {
            self.capacity
        }
        fn max_pass_width(&self) -> usize {
            self.width
        }
        fn run_provisional(&mut self, tokens: &[u32]) -> Result<()> {
            assert!(self.pending.is_none());
            assert!(!tokens.is_empty());
            assert!(tokens.len() <= self.width);
            assert!(self.position() + tokens.len() <= self.capacity);
            let mut prefix = self.committed.clone();
            self.rows = tokens
                .iter()
                .map(|&token| {
                    prefix.push(token);
                    prediction(&prefix)
                })
                .collect();
            self.pending = Some(tokens.to_vec());
            Ok(())
        }
        fn row_logits(&self, row: usize) -> Option<&[f32]> {
            self.rows.get(row).map(Vec::as_slice)
        }
        fn logits(&self) -> Option<&[f32]> {
            self.last.as_deref()
        }
        fn commit(&mut self, rows: usize) -> Result<()> {
            let pending = self.pending.take().unwrap();
            assert!(rows <= pending.len());
            self.committed.extend_from_slice(&pending[..rows]);
            self.last = Some(prediction(&self.committed));
            self.rows.truncate(rows);
            Ok(())
        }
        fn discard(&mut self) {
            self.pending = None;
            self.rows.clear();
        }
    }

    #[test]
    fn observation_failure_keeps_committed_outputs_and_continues_without_drafting() {
        struct FailsOnObservation {
            calls: usize,
        }
        impl Proposer for FailsOnObservation {
            fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
                vec![]
            }
            fn observe_committed(&mut self, _: &[CapturedRow]) -> Result<()> {
                self.calls += 1;
                if self.calls == 2 {
                    Err(Error::InvalidInput("observation failed".into()))
                } else {
                    Ok(())
                }
            }
        }
        let mut expected = SequenceVerifier::new(&[1, 2], 32, 4);
        let mut actual = expected.clone();
        let mut limits = GenerationLimits::new(7, 32);
        limits.max_draft_tokens = 3;
        let baseline = generate(
            &mut expected,
            &mut NoProposer,
            &mut Greedy,
            &[1, 2, 3],
            GenerationLimits::new(7, 32),
        )
        .unwrap();
        let mut proposer = FailsOnObservation { calls: 0 };
        let generated =
            generate(&mut actual, &mut proposer, &mut Greedy, &[1, 2, 3], limits).unwrap();
        assert_eq!(generated.tokens, baseline.tokens);
        assert_eq!(actual.committed, expected.committed);
        assert_eq!(actual.last, expected.last);
        assert_eq!(proposer.calls, 2);
    }

    struct FailsOnProposal {
        calls: usize,
        fail_on: usize,
    }
    impl Proposer for FailsOnProposal {
        fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
            unreachable!()
        }
        fn propose_with<V: Verifier>(
            &mut self,
            _: &mut V,
            _: &[u32],
            _: u32,
            _: usize,
        ) -> Result<Vec<u32>> {
            self.calls += 1;
            if self.calls == self.fail_on {
                Err(Error::InvalidInput("proposal failed".into()))
            } else {
                Ok(vec![])
            }
        }
    }

    #[test]
    fn proposal_failure_after_output_disables_drafting_and_matches_k_zero() {
        let mut expected = SequenceVerifier::new(&[1, 2], 32, 4);
        let mut actual = expected.clone();
        let baseline = generate(
            &mut expected,
            &mut NoProposer,
            &mut Greedy,
            &[1, 2, 3],
            GenerationLimits::new(7, 32),
        )
        .unwrap();
        let mut proposer = FailsOnProposal {
            calls: 0,
            fail_on: 2,
        };
        let mut seen = Vec::new();
        let mut callback = |token| {
            seen.push(token);
            true
        };
        let mut limits = GenerationLimits::new(7, 32);
        limits.max_draft_tokens = 3;
        limits.on_token = Some(&mut callback);
        let generated =
            generate(&mut actual, &mut proposer, &mut Greedy, &[1, 2, 3], limits).unwrap();
        assert_eq!(generated.tokens, baseline.tokens);
        assert_eq!(seen, baseline.tokens);
        assert_eq!(generated.stop_reason, baseline.stop_reason);
        assert_eq!(actual.committed, expected.committed);
        assert_eq!(actual.last, expected.last);
        assert!(actual.pending.is_none());
        assert_eq!(proposer.calls, 2);
    }

    #[test]
    fn proposal_failure_before_output_returns_named_error_without_changing_verifier() {
        let mut main = SequenceVerifier::new(&[1, 2], 32, 4);
        let before = main.clone();
        let mut proposer = FailsOnProposal {
            calls: 0,
            fail_on: 1,
        };
        let mut limits = GenerationLimits::new(7, 32);
        limits.max_draft_tokens = 3;
        let error =
            generate(&mut main, &mut proposer, &mut Greedy, &[1, 2, 3], limits).unwrap_err();
        assert!(matches!(
            error,
            GenerationError::Proposer(Error::InvalidInput(message)) if message == "proposal failed"
        ));
        assert_eq!(main.committed, before.committed);
        assert_eq!(main.last, before.last);
        assert_eq!(main.rows, before.rows);
        assert_eq!(main.pending, before.pending);
    }

    #[test]
    fn invalid_guesses_before_output_error_and_after_output_fall_back_losslessly() {
        struct Misbehaving {
            calls: usize,
            fail_on: usize,
            oversized: bool,
        }
        impl Proposer for Misbehaving {
            fn propose(&mut self, _: &[u32], _: u32, max: usize) -> Vec<u32> {
                self.calls += 1;
                if self.calls != self.fail_on {
                    vec![]
                } else if self.oversized {
                    vec![0; max + 1]
                } else {
                    vec![8] // SequenceVerifier has vocabulary IDs 0..8.
                }
            }
        }
        for oversized in [false, true] {
            for fail_on in [1, 2] {
                let mut expected = SequenceVerifier::new(&[1, 2], 32, 4);
                let mut actual = expected.clone();
                let before = actual.clone();
                let baseline = generate(
                    &mut expected,
                    &mut NoProposer,
                    &mut Greedy,
                    &[1, 2, 3],
                    GenerationLimits::new(7, 32),
                )
                .unwrap();
                let mut proposer = Misbehaving {
                    calls: 0,
                    fail_on,
                    oversized,
                };
                let mut seen = Vec::new();
                let mut callback = |token| {
                    seen.push(token);
                    true
                };
                let mut limits = GenerationLimits::new(7, 32);
                limits.max_draft_tokens = 3;
                limits.on_token = Some(&mut callback);
                let generated =
                    generate(&mut actual, &mut proposer, &mut Greedy, &[1, 2, 3], limits);
                if fail_on == 1 {
                    let expected_error = if oversized {
                        "proposer exceeded the requested guess limit"
                    } else {
                        "proposer returned a token outside the vocabulary"
                    };
                    assert!(matches!(
                        generated,
                        Err(GenerationError::InvalidInput(message)) if message == expected_error
                    ));
                    assert!(seen.is_empty());
                    assert_eq!(actual.committed, before.committed);
                    assert_eq!(actual.last, before.last);
                    assert_eq!(actual.rows, before.rows);
                    assert_eq!(actual.pending, before.pending);
                } else {
                    let generated = generated.unwrap();
                    assert_eq!(generated.tokens, baseline.tokens);
                    assert_eq!(seen, baseline.tokens);
                    assert_eq!(generated.stop_reason, baseline.stop_reason);
                    assert_eq!(actual.committed, expected.committed);
                    assert_eq!(actual.last, expected.last);
                    assert!(actual.pending.is_none());
                }
                assert_eq!(proposer.calls, fail_on);
            }
        }
    }

    struct GuidedProposer {
        sampler: Sampler,
        // None: oracle; Some(0): always wrong; Some(usize::MAX): varying prefix.
        wrong_after: Option<usize>,
    }
    impl Proposer for GuidedProposer {
        fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
            let mut prefix = history.to_vec();
            prefix.push(last);
            let wrong_after = match self.wrong_after {
                Some(usize::MAX) => Some(prefix.iter().sum::<u32>() as usize % (max + 1)),
                other => other,
            };
            (0..max)
                .map(|depth| {
                    let selected = self
                        .sampler
                        .sample(&prediction(&prefix), prefix.len() as u64)
                        .unwrap();
                    let guessed = if wrong_after.is_some_and(|after| depth >= after) {
                        (selected + 1) % 8
                    } else {
                        selected
                    };
                    prefix.push(guessed);
                    guessed
                })
                .collect()
        }
    }

    fn samplers() -> Vec<Sampler> {
        let sampled = |seed| {
            Sampler::Sampled(SamplingParameters {
                seed,
                top_k: 8,
                ..SamplingParameters::default()
            })
        };
        vec![Sampler::Greedy, sampled(17), sampled(0xabcdef)]
    }

    // An intentionally plain token-at-a-time oracle with no generation traits.
    fn sequential(
        prompt: &[u32],
        sampler: &Sampler,
        count: usize,
        capacity: usize,
        eos: &[u32],
        cancel_after: Option<usize>,
    ) -> (Vec<u32>, Vec<u32>, StopReason) {
        let mut prefix = prompt[..prompt.len() - 1].to_vec();
        let mut pending = prompt[prompt.len() - 1];
        let mut output = Vec::new();
        let mut stop = StopReason::MaxNewTokens;
        while output.len() < count {
            if prefix.len() == capacity {
                stop = StopReason::ContextCapacity;
                break;
            }
            prefix.push(pending);
            pending = sampler
                .sample(&prediction(&prefix), prefix.len() as u64)
                .unwrap();
            output.push(pending);
            if eos.contains(&pending) {
                stop = StopReason::EndOfSequence;
                break;
            }
            if cancel_after == Some(output.len()) {
                stop = StopReason::Cancelled;
                break;
            }
        }
        (output, prefix, stop)
    }

    fn check(
        prompt: &[u32],
        sampler: &Sampler,
        proposer: &mut impl Proposer,
        k: usize,
        sizes: (usize, usize),
        eos: &[u32],
        cancel_after: Option<usize>,
    ) -> GenerationResult {
        let (max, capacity) = sizes;
        let expected = sequential(prompt, sampler, max, capacity, eos, cancel_after);
        let mut verifier = SequenceVerifier::new(&prompt[..prompt.len() - 1], capacity, 9);
        let mut selector = sampler.clone();
        let mut seen = Vec::new();
        let mut callback = |token| {
            seen.push(token);
            cancel_after != Some(seen.len())
        };
        let mut limits = GenerationLimits::new(max, capacity);
        limits.max_draft_tokens = k;
        limits.eos_ids = eos;
        limits.on_token = Some(&mut callback);
        let result = generate(&mut verifier, proposer, &mut selector, prompt, limits).unwrap();
        assert_eq!(result.tokens, expected.0);
        assert_eq!(seen, expected.0);
        assert_eq!(verifier.committed, expected.1);
        assert_eq!(verifier.logits(), Some(prediction(&expected.1).as_slice()));
        assert!(verifier.pending.is_none());
        assert_eq!(result.stop_reason, expected.2);
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

    #[test]
    fn speculation_preserves_sequential_tokens_state_and_seeded_repeatability() {
        let prompt = [1, 2, 3, 1, 2, 3, 1];
        for sampler in samplers() {
            for k in [1, 2, 3, 7] {
                check(&prompt, &sampler, &mut NoProposer, k, (23, 64), &[], None);
                for wrong_after in [None, Some(0), Some(usize::MAX)] {
                    let mut proposer = GuidedProposer {
                        sampler: sampler.clone(),
                        wrong_after,
                    };
                    let result = check(&prompt, &sampler, &mut proposer, k, (23, 64), &[], None);
                    let repeated = check(&prompt, &sampler, &mut proposer, k, (23, 64), &[], None);
                    assert_eq!(result.tokens, repeated.tokens);
                    assert_eq!(
                        result.statistics.tokens_confirmed_per_pass,
                        repeated.statistics.tokens_confirmed_per_pass
                    );
                    if wrong_after.is_none() {
                        assert!(result.statistics.passes < result.tokens.len());
                        assert_eq!(
                            result.statistics.guesses_offered,
                            result.statistics.guesses_accepted
                        );
                    }
                    if wrong_after == Some(0) {
                        assert_eq!(result.statistics.passes, result.tokens.len());
                        assert_eq!(result.statistics.guesses_accepted, 0);
                    }
                }
                let mut ngram = NgramProposer::new(2, k).unwrap();
                check(&prompt, &sampler, &mut ngram, k, (23, 64), &[], None);
            }
        }
    }

    #[test]
    fn eos_cancel_and_capacity_discard_later_rows_and_keep_last_token_pending() {
        let prompt = [1, 2, 3, 1, 2, 3, 1];
        for sampler in samplers() {
            let sequence = sequential(&prompt, &sampler, 16, 64, &[], None).0;
            for k in [1, 2, 3, 7] {
                for eos in [vec![sequence[0]], vec![sequence[3]], vec![sequence[7]]] {
                    let mut oracle = GuidedProposer {
                        sampler: sampler.clone(),
                        wrong_after: None,
                    };
                    check(&prompt, &sampler, &mut oracle, k, (16, 64), &eos, None);
                    // An EOS guess cannot stop generation until the selector chooses it.
                    check(
                        &prompt,
                        &sampler,
                        &mut ScriptProposer::new(vec![eos[0]; k]),
                        k,
                        (16, 64),
                        &eos,
                        None,
                    );
                }
                for cancel in [1, 3, 6, 11] {
                    let mut oracle = GuidedProposer {
                        sampler: sampler.clone(),
                        wrong_after: None,
                    };
                    check(
                        &prompt,
                        &sampler,
                        &mut oracle,
                        k,
                        (16, 64),
                        &[],
                        Some(cancel),
                    );
                }
                for capacity in [prompt.len() - 1, prompt.len(), prompt.len() + 3] {
                    let mut oracle = GuidedProposer {
                        sampler: sampler.clone(),
                        wrong_after: None,
                    };
                    check(&prompt, &sampler, &mut oracle, k, (16, capacity), &[], None);
                }
            }
        }
    }

    #[test]
    fn ngram_uses_most_recent_match_and_only_known_continuation() {
        let mut proposer = NgramProposer::new(2, 7).unwrap();
        assert_eq!(proposer.propose(&[1, 2, 3, 1, 2, 4, 1], 2, 7), [4, 1, 2]);
        assert_eq!(proposer.propose(&[1, 2, 3, 1, 2, 4, 1], 2, 1), [4]);
        assert_eq!(proposer.propose(&[1, 1], 1, 7), [1]);
        assert!(proposer.propose(&[1], 2, 7).is_empty());
        assert!(proposer.propose(&[1, 2, 3], 4, 7).is_empty());
        assert!(NgramProposer::new(0, 7).is_err());
        assert!(
            NgramProposer::new(1, 0)
                .unwrap()
                .propose(&[1], 1, 7)
                .is_empty()
        );
    }

    #[test]
    fn caller_capacity_and_pass_width_bound_every_verification() {
        let mut verifier = SequenceVerifier::new(&[1, 2], 64, 2);
        let mut proposer = GuidedProposer {
            sampler: Sampler::Greedy,
            wrong_after: None,
        };
        let mut limits = GenerationLimits::new(30, 7);
        limits.max_draft_tokens = usize::MAX;
        let result = generate(
            &mut verifier,
            &mut proposer,
            &mut Greedy,
            &[1, 2, 3],
            limits,
        )
        .unwrap();
        assert_eq!(result.stop_reason, StopReason::ContextCapacity);
        assert_eq!(
            result.tokens,
            sequential(&[1, 2, 3], &Sampler::Greedy, 30, 7, &[], None).0
        );
        assert_eq!(verifier.position(), 7);
        assert_eq!(result.statistics.tokens_confirmed_per_pass, [2, 2, 1]);
        assert_eq!(result.statistics.rows_verified, 5);
        assert_eq!(result.statistics.guesses_offered, 2);
    }

    #[test]
    fn no_tokens_does_not_run_a_pass_and_selector_failure_discards_candidates() {
        let mut verifier = SequenceVerifier::new(&[1, 2], 64, 9);
        let result = generate(
            &mut verifier,
            &mut NoProposer,
            &mut Greedy,
            &[3],
            GenerationLimits::new(0, 64),
        )
        .unwrap();
        assert!(result.tokens.is_empty());
        assert_eq!(verifier.committed, [1, 2]);
        assert_eq!(result.statistics.passes, 0);
        assert!(
            generate(
                &mut verifier,
                &mut NoProposer,
                &mut Greedy,
                &[],
                GenerationLimits::new(1, 64)
            )
            .is_err()
        );
        struct Failing;
        impl Selector for Failing {
            fn select(&mut self, _logits: &[f32], _position: u64) -> Result<u32> {
                Err(Error::InvalidInput("selection failed".into()))
            }
        }
        let error = generate(
            &mut verifier,
            &mut NoProposer,
            &mut Failing,
            &[3],
            GenerationLimits::new(5, 64),
        )
        .unwrap_err();
        assert!(matches!(error, GenerationError::Selector(_)));
        assert_eq!(verifier.committed, [1, 2]);
        assert!(verifier.pending.is_none());
        assert_eq!(verifier.logits(), Some(prediction(&[1, 2]).as_slice()));
    }
}
