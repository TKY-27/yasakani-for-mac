//! Main/draft connection. Captures are committed inputs; the frontier is absent.
use crate::draft::{BLOCK_ROWS, BlockDraft, Draft, DraftSnapshot};
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
    context_identity: Option<std::sync::Arc<()>>,
    owner: Rc<()>,
}

/// Main numeric checkpoint and the matching draft ring, restored as one session.
pub struct SessionSnapshot<T> {
    verifier: T,
    proposer: DraftProposerSnapshot,
}

/// Draft calls made by one proposer since it was created; reset keeps them.
/// Each proposal runs the whole block; each update pushes one run of
/// consecutive committed rows (`BlockDraft::push_rows`) into the ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftWork {
    pub proposals: usize,
    pub updates: usize,
}

/// Generic over the draft implementation: the scalar reference by default, or a
/// fast draft that reproduces it bit for bit.
pub struct DraftProposer<D: BlockDraft = Draft> {
    draft: D,
    k: usize,
    choice: DraftChoice,
    context_identity: Option<std::sync::Arc<()>>,
    owner: Rc<()>,
    work: DraftWork,
}
impl<D: BlockDraft> DraftProposer<D> {
    pub fn work(&self) -> DraftWork {
        self.work
    }
    /// Update request-local coupling without invalidating the retained ring/checkpoint.
    pub fn set_choice(&mut self, choice: DraftChoice) {
        self.choice = choice;
    }
    pub fn new(draft: D, k: usize, choice: DraftChoice) -> Result<Self> {
        if !(1..BLOCK_ROWS).contains(&k) {
            return Err(Error::InvalidInput("draft K must be in 1..8".into()));
        }
        Ok(Self {
            draft,
            k,
            choice,
            context_identity: None,
            owner: Rc::new(()),
            work: DraftWork::default(),
        })
    }
    pub fn capture_options(&self) -> CaptureOptions {
        CaptureOptions {
            layers: self.draft.options().capture_layers.clone(),
            max_rows: self.draft.dimensions().window,
        }
    }
    pub fn draft(&self) -> &D {
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
            context_identity: self.context_identity.clone(),
            owner: Rc::clone(&self.owner),
        }
    }
    pub fn restore(&mut self, snapshot: &DraftProposerSnapshot) -> Result<()> {
        if !Rc::ptr_eq(&self.owner, &snapshot.owner) {
            return Err(Error::InvalidInput(
                "draft checkpoint belongs to another proposer or reset".into(),
            ));
        }
        self.draft.restore(&snapshot.ring)?;
        self.context_identity = snapshot.context_identity.clone();
        Ok(())
    }
    pub fn reset(&mut self) {
        self.draft.reset();
        self.context_identity = None;
        self.owner = Rc::new(());
    }
    pub fn snapshot_session<V: CheckpointVerifier>(
        &mut self,
        verifier: &mut V,
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
            .map_err(GenerationError::Proposer)?;
        self.context_identity = verifier.context_identity();
        Ok(())
    }
    pub fn synchronize<V: Verifier>(&mut self, verifier: &mut V) -> Result<()> {
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
        let identity = verifier
            .context_identity()
            .ok_or_else(|| Error::InvalidInput("draft needs main context identity".into()))?;
        let mismatch = match &self.context_identity {
            Some(old) => !std::sync::Arc::ptr_eq(old, &identity),
            None => self.draft.position() != 0,
        };
        if mismatch {
            return Err(Error::InvalidInput(
                "main context restored or reset without the draft".into(),
            ));
        }
        self.context_identity = Some(identity);
        if self.draft.position() > verifier.position() {
            return Err(Error::InvalidInput(
                "restore the draft checkpoint with the main checkpoint".into(),
            ));
        }
        let rows = verifier.captured_rows();
        if rows
            .first()
            .is_some_and(|row| row.position > self.draft.position())
        {
            let first = rows[0].position;
            // Upstream overwrites every older slot: projecting this complete suffix
            // yields the identical ring even when an already-used draft lags behind.
            if rows.len() < d.window || first + rows.len() != verifier.position() {
                return Err(Error::InvalidInput(
                    "incomplete draft prompt capture suffix".into(),
                ));
            }
            self.draft.reset_at(first);
        }
        // Keep rows at or after the position the ring will have reached when each
        // is pushed, as a row-by-row loop would; already-inserted rows are skipped.
        let mut next = self.draft.position();
        let rows: Vec<_> = rows
            .iter()
            .filter(|row| {
                let keep = row.position >= next;
                next += usize::from(keep);
                keep
            })
            .collect();
        self.push(&rows)?;
        if self.draft.position() != verifier.position() {
            return Err(Error::InvalidInput(
                "draft context has missing committed rows".into(),
            ));
        }
        drop(verifier.take_captured_rows());
        Ok(())
    }
    /// The complete block proposal (all seven tokens with their candidates) for
    /// a pending `last` token, after synchronizing with the verifier.
    pub fn propose_block<V: Verifier>(
        &mut self,
        verifier: &mut V,
        last: u32,
    ) -> Result<crate::draft::DraftProposal> {
        self.synchronize(verifier)?;
        self.work.proposals += 1;
        self.draft
            .propose(last, &mut |token| verifier.embedding(token), self.choice)
    }
    /// Push maximal runs of consecutive positions as batches. A run that does not
    /// start at the draft position fails exactly as a single-row push would.
    fn push(&mut self, rows: &[&CapturedRow]) -> Result<()> {
        let mut start = 0;
        for end in 1..=rows.len() {
            if end == rows.len() || rows[end].position != rows[end - 1].position + 1 {
                let values: Vec<_> = rows[start..end].iter().map(|r| &r.values[..]).collect();
                self.work.updates += 1;
                self.draft.push_rows(rows[start].position, &values)?;
                start = end;
            }
        }
        Ok(())
    }
}
impl<D: BlockDraft> Proposer for DraftProposer<D> {
    /// Neural proposals need the verifier's embeddings; use `propose_with`.
    fn propose(&mut self, _history: &[u32], _last: u32, _max: usize) -> Vec<u32> {
        Vec::new()
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &mut V) -> Result<()> {
        DraftProposer::synchronize(self, verifier)
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> Result<()> {
        if self.context_identity.is_none() {
            return Err(Error::InvalidInput(
                "draft context identity is not bound; synchronize first".into(),
            ));
        }
        if rows
            .first()
            .is_some_and(|row| row.position > self.draft.position())
            && rows.len() >= self.draft.dimensions().window
        {
            // A bounded hand-over may cover a whole ring after a wide commit.
            self.draft.reset_at(rows[0].position);
        }
        self.push(&rows.iter().collect::<Vec<_>>())
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &mut V,
        _history: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        if max == 0 {
            self.synchronize(verifier)?;
            return Ok(Vec::new());
        }
        Ok(self
            .propose_block(verifier, last)?
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
    /// N-gram only when its match spans at least this many tokens, else the
    /// draft. Such a copy is also offered as free guesses (D21).
    LongNgramFirst(usize),
}

/// First nonempty proposal wins. No upstream cost/source policy.
pub struct CombinedProposer<D: BlockDraft = Draft> {
    ngram: NgramProposer,
    draft: DraftProposer<D>,
    order: ProposalOrder,
}
impl<D: BlockDraft> CombinedProposer<D> {
    /// Update request-local neural coupling while preserving the joint session.
    pub fn set_choice(&mut self, choice: DraftChoice) {
        self.draft.set_choice(choice);
    }
    pub fn new(ngram: NgramProposer, draft: DraftProposer<D>, order: ProposalOrder) -> Self {
        Self {
            ngram,
            draft,
            order,
        }
    }
    pub fn draft(&self) -> &DraftProposer<D> {
        &self.draft
    }
    pub fn reset(&mut self) {
        self.draft.reset();
    }
    pub fn snapshot_session<V: CheckpointVerifier>(
        &mut self,
        verifier: &mut V,
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
impl<D: BlockDraft> Proposer for CombinedProposer<D> {
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        self.ngram.propose(history, last, max)
    }
    fn free_guesses(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        match self.order {
            ProposalOrder::LongNgramFirst(length) => {
                self.ngram.propose_long(history, last, max, length)
            }
            _ => Vec::new(),
        }
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &mut V) -> Result<()> {
        self.draft.synchronize(verifier)
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> Result<()> {
        self.draft.observe_committed(rows)
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &mut V,
        history: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.synchronize(verifier)?;
        let first = match self.order {
            ProposalOrder::NgramFirst => self.ngram.propose(history, last, max),
            ProposalOrder::LongNgramFirst(length) => {
                self.ngram.propose_long(history, last, max, length)
            }
            ProposalOrder::DraftFirst => self.draft.propose_with(verifier, history, last, max)?,
        };
        if !first.is_empty() {
            return Ok(first);
        }
        match self.order {
            ProposalOrder::NgramFirst | ProposalOrder::LongNgramFirst(_) => {
                self.draft.propose_with(verifier, history, last, max)
            }
            ProposalOrder::DraftFirst => Ok(self.ngram.propose(history, last, max)),
        }
    }
}
