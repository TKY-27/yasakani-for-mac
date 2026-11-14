use yskn_ref::Result;
use yskn_ref::model::CapturedRow;
use yskn_ref::speculation::{Proposer, Verifier};

pub struct Observed<'a, P> {
    inner: &'a mut P,
    pub max_offered: usize,
}

impl<'a, P> Observed<'a, P> {
    pub fn new(inner: &'a mut P) -> Self {
        Self {
            inner,
            max_offered: 0,
        }
    }
}

impl<P: Proposer> Proposer for Observed<'_, P> {
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        let guesses = self.inner.propose(history, last, max);
        self.max_offered = self.max_offered.max(guesses.len());
        guesses
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
        let guesses = self.inner.propose_with(verifier, history, last, max)?;
        self.max_offered = self.max_offered.max(guesses.len());
        Ok(guesses)
    }
}
