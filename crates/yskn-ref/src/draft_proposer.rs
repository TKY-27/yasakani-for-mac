//! Main/draft connection. Captures are committed inputs; the frontier is absent.
use crate::draft::{BLOCK_ROWS, ContextFeatures, Draft, DraftSnapshot};
use crate::draft_head::DraftChoice;
use crate::model::{CaptureOptions, CapturedRow};
use crate::speculation::{CheckpointVerifier, GenerationError, NgramProposer, Proposer, Verifier};
use crate::{Error, Result};
use std::rc::Rc;

/// Owned draft half of a session checkpoint. Save/restore alongside the verifier
/// checkpoint, preferably through the joint checkpoint API.
#[derive(Clone, Debug)]
pub struct DraftProposerSnapshot {
    ring: DraftSnapshot,
    owner: Rc<()>,
}

/// Main numeric checkpoint and the matching draft ring, restored as one session.
pub struct SessionSnapshot<T> {
    verifier: T,
    proposer: DraftProposerSnapshot,
}

pub struct DraftProposer {
    draft: Draft,
    k: usize,
    choice: DraftChoice,
    owner: Rc<()>,
}
impl DraftProposer {
    pub fn new(draft: Draft, k: usize, choice: DraftChoice) -> Result<Self> {
        if !(1..BLOCK_ROWS).contains(&k) {
            return Err(Error::InvalidInput("draft K must be in 1..8".into()));
        }
        Ok(Self {
            draft,
            k,
            choice,
            owner: Rc::new(()),
        })
    }
    pub fn capture_options(&self) -> CaptureOptions {
        CaptureOptions {
            layers: self.draft.options().capture_layers.clone(),
            max_rows: self.draft.dimensions().window,
        }
    }
    pub fn draft(&self) -> &Draft {
        &self.draft
    }
    /// Absolute positions currently retained in the draft ring (end excluded).
    pub fn ring_positions(&self) -> std::ops::Range<usize> {
        self.draft
            .position()
            .saturating_sub(self.draft.dimensions().window)..self.draft.position()
    }
    pub fn snapshot(&self) -> DraftProposerSnapshot {
        DraftProposerSnapshot {
            ring: self.draft.snapshot(),
            owner: Rc::clone(&self.owner),
        }
    }
    pub fn restore(&mut self, snapshot: &DraftProposerSnapshot) -> Result<()> {
        if !Rc::ptr_eq(&self.owner, &snapshot.owner) {
            return Err(Error::InvalidInput(
                "draft checkpoint belongs to another proposer or reset".into(),
            ));
        }
        self.draft.restore(&snapshot.ring)
    }
    pub fn reset(&mut self) {
        self.draft.reset();
        self.owner = Rc::new(());
    }
    pub fn snapshot_session<V: CheckpointVerifier>(
        &mut self,
        verifier: &V,
    ) -> std::result::Result<SessionSnapshot<V::Snapshot>, GenerationError<V::Error>> {
        self.synchronize(verifier)
            .map_err(GenerationError::Proposer)?;
        Ok(SessionSnapshot {
            verifier: verifier.snapshot().map_err(GenerationError::Verifier)?,
            proposer: self.snapshot(),
        })
    }
    pub fn restore_session<V: CheckpointVerifier>(
        &mut self,
        verifier: &mut V,
        snapshot: &SessionSnapshot<V::Snapshot>,
    ) -> std::result::Result<(), GenerationError<V::Error>> {
        if !Rc::ptr_eq(&self.owner, &snapshot.proposer.owner) {
            return Err(GenerationError::Proposer(Error::InvalidInput(
                "draft checkpoint belongs to another proposer or reset".into(),
            )));
        }
        // Owner validation makes the draft restore infallible; main validates atomically.
        verifier
            .restore(&snapshot.verifier)
            .map_err(GenerationError::Verifier)?;
        self.restore(&snapshot.proposer)
            .map_err(GenerationError::Proposer)
    }
    pub fn synchronize<V: Verifier>(&mut self, verifier: &V) -> Result<()> {
        let d = self.draft.dimensions();
        let main = verifier
            .dimensions()
            .ok_or_else(|| Error::InvalidInput("draft needs main dimensions".into()))?;
        let options = verifier
            .capture_options()
            .ok_or_else(|| Error::InvalidInput("draft needs capture options".into()))?;
        if main.hidden_width != d.hidden
            || main.vocabulary_size != d.vocabulary
            || main.epsilon.to_bits() != d.epsilon.to_bits()
            || options.layers != self.draft.options().capture_layers
            || options.max_rows < d.window
        {
            return Err(Error::InvalidInput(
                "incompatible main/draft dimensions or captures".into(),
            ));
        }
        if self.draft.position() > verifier.position() {
            return Err(Error::InvalidInput(
                "restore the draft checkpoint with the main checkpoint".into(),
            ));
        }
        let rows = verifier.captured_rows();
        if self.draft.position() == 0 && rows.first().is_some_and(|row| row.position > 0) {
            let first = rows[0].position;
            // Older prompt rows have been overwritten; seed only a complete suffix.
            if rows.len() < d.window || first + rows.len() != verifier.position() {
                return Err(Error::InvalidInput(
                    "incomplete draft prompt capture suffix".into(),
                ));
            }
            self.draft.reset_at(first);
        }
        for row in rows {
            if row.position >= self.draft.position() {
                self.push(row)?;
            }
        }
        if self.draft.position() != verifier.position() {
            return Err(Error::InvalidInput(
                "draft context has missing committed rows".into(),
            ));
        }
        Ok(())
    }
    fn push(&mut self, row: &CapturedRow) -> Result<()> {
        self.draft.push_context(
            row.position,
            ContextFeatures {
                values: &row.values,
            },
        )
    }
}
impl Proposer for DraftProposer {
    /// Neural proposals need the verifier's embeddings; use `propose_with`.
    fn propose(&mut self, _history: &[u32], _last: u32, _max: usize) -> Vec<u32> {
        Vec::new()
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &V) -> Result<()> {
        DraftProposer::synchronize(self, verifier)
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> Result<()> {
        for row in rows {
            self.push(row)?;
        }
        Ok(())
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &V,
        _history: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.synchronize(verifier)?;
        if max == 0 {
            return Ok(Vec::new());
        }
        Ok(self
            .draft
            .propose(last, |token| verifier.embedding(token), self.choice)?
            .tokens
            .into_iter()
            .take(self.k.min(max))
            .collect())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ProposalOrder {
    NgramFirst,
    DraftFirst,
}

/// First nonempty proposal wins. No adaptive upstream cost/source policy.
pub struct CombinedProposer {
    ngram: NgramProposer,
    draft: DraftProposer,
    order: ProposalOrder,
}
impl CombinedProposer {
    pub fn new(ngram: NgramProposer, draft: DraftProposer, order: ProposalOrder) -> Self {
        Self {
            ngram,
            draft,
            order,
        }
    }
    pub fn draft(&self) -> &DraftProposer {
        &self.draft
    }
    pub fn reset(&mut self) {
        self.draft.reset();
    }
    pub fn snapshot_session<V: CheckpointVerifier>(
        &mut self,
        verifier: &V,
    ) -> std::result::Result<SessionSnapshot<V::Snapshot>, GenerationError<V::Error>> {
        self.draft.snapshot_session(verifier)
    }
    pub fn restore_session<V: CheckpointVerifier>(
        &mut self,
        verifier: &mut V,
        snapshot: &SessionSnapshot<V::Snapshot>,
    ) -> std::result::Result<(), GenerationError<V::Error>> {
        self.draft.restore_session(verifier, snapshot)
    }
    pub fn snapshot(&self) -> DraftProposerSnapshot {
        self.draft.snapshot()
    }
    pub fn restore(&mut self, snapshot: &DraftProposerSnapshot) -> Result<()> {
        self.draft.restore(snapshot)
    }
}
impl Proposer for CombinedProposer {
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        self.ngram.propose(history, last, max)
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &V) -> Result<()> {
        self.draft.synchronize(verifier)
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> Result<()> {
        self.draft.observe_committed(rows)
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &V,
        history: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.synchronize(verifier)?;
        let first = match self.order {
            ProposalOrder::NgramFirst => self.ngram.propose(history, last, max),
            ProposalOrder::DraftFirst => self.draft.propose_with(verifier, history, last, max)?,
        };
        if !first.is_empty() {
            return Ok(first);
        }
        match self.order {
            ProposalOrder::NgramFirst => self.draft.propose_with(verifier, history, last, max),
            ProposalOrder::DraftFirst => Ok(self.ngram.propose(history, last, max)),
        }
    }
}
