//! How many guesses to request before each proposal (D21).
//!
//! The policy only changes the number of guesses offered; `generate` keeps the
//! output independent of it. Every decision is a function of the tokens produced
//! so far, the proposals made and a fixed cost model. No clock is read, so for a
//! given input the sequence of depths repeats exactly.
//!
//! Each proposal is scored retrospectively: once later output is known, its
//! counterfactual accepted prefix (how many of all its guesses equal the output)
//! is exact, because the output does not depend on the proposal. A proposal
//! therefore informs every depth, including depths that were not offered, and a
//! single cheap probe after a skipped stretch measures the whole profile again.
use std::collections::VecDeque;

/// Verification pass time by row count and the cost of one proposer call, in
/// one common unit (milliseconds for the shipped tables).
#[derive(Clone, Debug, PartialEq)]
pub struct CostModel {
    /// `pass[r - 1]` is a pass with `r` rows. Longer passes extend the last step.
    pub pass: Vec<f64>,
    /// One proposer call; the neural block draft costs the same at every depth.
    pub proposal: f64,
}

impl CostModel {
    /// Apple M5 (4P+6E CPU, 10-core GPU), Qwen3.8-27B, Kv5 cache, Metal,
    /// short context: measured main pass by rows 1..=8, and one fast neural
    /// proposal (D18, D21).
    pub fn apple_m5_neural() -> Self {
        Self {
            pass: vec![232.0, 300.0, 345.0, 390.0, 455.0, 520.0, 585.0, 650.0],
            proposal: 47.0,
        }
    }

    pub fn pass(&self, rows: usize) -> f64 {
        let n = self.pass.len();
        if rows <= n {
            return self.pass[rows.max(1) - 1];
        }
        let step = if n > 1 {
            self.pass[n - 1] - self.pass[n - 2]
        } else {
            0.0
        };
        self.pass[n - 1] + step * (rows - n) as f64
    }

    /// Expected cost of one pass that requests `depth` guesses.
    fn step(&self, depth: usize) -> f64 {
        self.pass(depth + 1) + if depth > 0 { self.proposal } else { 0.0 }
    }

    fn validate(&self) -> Result<(), &'static str> {
        let finite = |x: f64| x.is_finite() && x >= 0.0;
        if self.pass.is_empty()
            || !self.pass.iter().all(|&t| finite(t) && t > 0.0)
            || self.pass.windows(2).any(|w| w[1] < w[0])
            || !finite(self.proposal)
        {
            return Err("cost model needs positive, nondecreasing pass times");
        }
        Ok(())
    }
}

/// Configuration of the adaptive policy.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveDepth {
    pub costs: CostModel,
    /// Initial P(accepted prefix >= j) for j = 1, 2, ...; missing depths use 0.
    pub prior: Vec<f64>,
    /// Weight of each newly scored proposal in the running estimate, in (0, 1].
    pub rate: f64,
    /// Propose once after this many consecutive passes without a proposal.
    pub probe_after: usize,
    /// Each pass without a proposal moves the estimate this far back toward
    /// the prior, in [0, 1]: old evidence goes stale while nothing is observed.
    pub relax: f64,
}

impl AdaptiveDepth {
    /// Tuned on the four `yskn bench` tasks with the fast neural draft (D21).
    pub fn neural() -> Self {
        Self {
            costs: CostModel::apple_m5_neural(),
            prior: vec![0.84, 0.70, 0.57, 0.50, 0.45, 0.39, 0.34],
            rate: 0.25,
            probe_after: 4,
            relax: 0.05,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        self.costs.validate()?;
        if !(self.rate > 0.0 && self.rate <= 1.0)
            || !(0.0..=1.0).contains(&self.relax)
            || !self.prior.iter().all(|p| (0.0..=1.0).contains(p))
        {
            return Err("adaptive depth needs a rate in (0, 1] and priors in [0, 1]");
        }
        Ok(())
    }
}

/// How `generate` chooses the number of guesses to request each pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum DepthPolicy {
    /// Always request `max_draft_tokens` (bounded by the pass width).
    #[default]
    Fixed,
    /// Request the depth with the best expected tokens per unit cost, up to
    /// `max_draft_tokens`; zero skips the proposer for that pass.
    Adaptive(AdaptiveDepth),
}

struct Pending {
    /// Output index the first guess predicts.
    first: usize,
    guesses: Vec<u32>,
}

/// Running state of one generation under an [`AdaptiveDepth`] policy.
pub struct DepthController {
    costs: CostModel,
    rate: f64,
    probe_after: usize,
    relax: f64,
    prior: Vec<f64>,
    /// `estimate[j - 1]` approximates P(accepted prefix >= j).
    estimate: Vec<f64>,
    skipped: usize,
    pending: VecDeque<Pending>,
}

impl DepthController {
    /// `max_depth` bounds every request. The configuration must be valid.
    pub fn new(config: &AdaptiveDepth, max_depth: usize) -> Self {
        let prior: Vec<f64> = (0..max_depth)
            .map(|j| config.prior.get(j).copied().unwrap_or(0.0))
            .collect();
        Self {
            costs: config.costs.clone(),
            rate: config.rate,
            probe_after: config.probe_after,
            relax: config.relax,
            estimate: prior.clone(),
            prior,
            skipped: 0,
            pending: VecDeque::new(),
        }
    }

    pub fn estimate(&self) -> &[f64] {
        &self.estimate
    }

    /// Expected tokens confirmed by a pass requesting `depth` guesses.
    pub fn expected_tokens(&self, depth: usize) -> f64 {
        1.0 + self.estimate[..depth.min(self.estimate.len())]
            .iter()
            .sum::<f64>()
    }

    /// Best depth in `from..=limit` by expected tokens per cost; ties keep the
    /// smaller depth.
    fn best(&self, from: usize, limit: usize) -> usize {
        let mut best = from;
        let mut best_rate = self.expected_tokens(from) / self.costs.step(from);
        for depth in from + 1..=limit {
            let rate = self.expected_tokens(depth) / self.costs.step(depth);
            if rate > best_rate {
                (best, best_rate) = (depth, rate);
            }
        }
        best
    }

    /// Depth to request for the next pass, at most `limit`. Zero means: do not
    /// call the proposer. A long run of zeros is broken by one probe.
    pub fn depth(&mut self, limit: usize) -> usize {
        let limit = limit.min(self.estimate.len());
        if limit == 0 {
            return 0;
        }
        let best = self.best(0, limit);
        if best > 0 {
            self.skipped = 0;
            return best;
        }
        self.skipped += 1;
        for (estimate, prior) in self.estimate.iter_mut().zip(&self.prior) {
            *estimate += self.relax * (prior - *estimate);
        }
        if self.skipped <= self.probe_after {
            return 0;
        }
        self.skipped = 0;
        self.best(1, limit)
    }

    /// Record a proposal's complete guesses; `first` is the output index of the
    /// token its first guess predicts.
    pub fn proposed(&mut self, first: usize, guesses: &[u32]) {
        self.skipped = 0;
        self.pending.push_back(Pending {
            first,
            guesses: guesses[..guesses.len().min(self.estimate.len())].to_vec(),
        });
    }

    /// Score every pending proposal that `outputs` (all output so far) decides.
    pub fn observe(&mut self, outputs: &[u32]) {
        let mut index = 0;
        while index < self.pending.len() {
            let p = &self.pending[index];
            let known = outputs.len().saturating_sub(p.first).min(p.guesses.len());
            let matched = p.guesses[..known]
                .iter()
                .zip(&outputs[p.first.min(outputs.len())..])
                .take_while(|(guess, output)| guess == output)
                .count();
            if matched == known && known < p.guesses.len() {
                index += 1;
                continue;
            }
            for (j, estimate) in self.estimate.iter_mut().enumerate() {
                let hit = if j < matched { 1.0 } else { 0.0 };
                *estimate += self.rate * (hit - *estimate);
            }
            self.pending.remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Output token i is i. A proposal at output index t guesses t + j for each
    /// j whose position is easy, else a wrong token, so its counterfactual
    /// accepted prefix stops at the first hard position.
    fn simulate(
        config: &AdaptiveDepth,
        max: usize,
        easy: impl Fn(usize) -> bool,
        tokens: usize,
    ) -> Vec<(usize, usize)> {
        let mut controller = DepthController::new(config, max);
        let mut outputs: Vec<u32> = Vec::new();
        let mut trace = Vec::new();
        while outputs.len() < tokens {
            let t = outputs.len();
            let depth = controller.depth(max);
            let mut confirmed = 1;
            if depth > 0 {
                let guesses: Vec<u32> = (0..max)
                    .map(|j| {
                        if easy(t + j) {
                            (t + j) as u32
                        } else {
                            u32::MAX
                        }
                    })
                    .collect();
                controller.proposed(t, &guesses);
                confirmed += guesses[..depth]
                    .iter()
                    .enumerate()
                    .take_while(|&(j, &g)| g == (t + j) as u32)
                    .count();
            }
            outputs.extend((t..t + confirmed).map(|i| i as u32));
            controller.observe(&outputs);
            trace.push((t, depth));
        }
        trace
    }

    #[test]
    fn all_accepted_goes_and_stays_at_the_maximum() {
        let trace = simulate(&AdaptiveDepth::neural(), 7, |_| true, 200);
        assert!(trace[3..].iter().all(|&(_, depth)| depth == 7), "{trace:?}");
    }

    #[test]
    fn none_accepted_skips_the_proposer_and_probes_at_a_fixed_interval() {
        let mut config = AdaptiveDepth::neural();
        let trace = simulate(&config, 7, |_| false, 200);
        let backed_off = trace.iter().position(|&(_, d)| d == 0).unwrap();
        assert!(backed_off <= 4, "{trace:?}");
        let tail: Vec<_> = trace[backed_off + 6..].iter().map(|&(_, d)| d).collect();
        // A probe at least every probe_after + 1 passes (relaxation toward the
        // prior may bring one earlier), but most passes skip the proposer.
        for window in tail.windows(config.probe_after + 1) {
            assert!(window.iter().any(|&d| d > 0), "{tail:?}");
        }
        let probes = tail.iter().filter(|&&d| d > 0).count();
        assert!(probes * 3 < tail.len(), "{tail:?}");
        config.relax = 0.0;
        let trace = simulate(&config, 7, |_| false, 200);
        let tail: Vec<_> = trace[backed_off + 6..].iter().map(|&(_, d)| d).collect();
        // Without relaxation: exactly one probe per probe_after + 1 passes.
        for window in tail.chunks_exact(config.probe_after + 1) {
            assert_eq!(window.iter().filter(|&&d| d > 0).count(), 1, "{tail:?}");
        }
    }

    #[test]
    fn alternating_regimes_back_off_and_recover_within_bounds() {
        let config = AdaptiveDepth::neural();
        // 120 easy tokens, 120 hard, repeated.
        let easy = |i: usize| (i / 120) % 2 == 0;
        let trace = simulate(&config, 7, easy, 960);
        for start in (120..960).step_by(120) {
            let passes: Vec<_> = trace.iter().filter(|&&(t, _)| t >= start).collect();
            if easy(start) {
                // Hard -> easy: deep proposals again within a bounded number of passes.
                let deep = passes.iter().position(|&&(_, d)| d == 7).unwrap();
                let bound = 3 * (config.probe_after + 1) + 2;
                assert!(deep <= bound, "recovery at {start}: {passes:?}");
            } else {
                // Easy -> hard: proposing stops within a few passes.
                let off = passes.iter().position(|&&(_, d)| d == 0).unwrap();
                assert!(off <= 8, "back-off at {start}: {passes:?}");
            }
        }
    }

    #[test]
    fn the_recovery_bound_holds_from_any_saturated_hard_state() {
        // Worst case: estimates at zero, probe interval as configured.
        let mut config = AdaptiveDepth::neural();
        config.prior = vec![0.0; 7];
        for probe_after in [0, 2, 4, 8] {
            config.probe_after = probe_after;
            let trace = simulate(&config, 7, |_| true, 400);
            let deep = trace.iter().position(|&(_, d)| d == 7).unwrap();
            // Each probe is scored at every depth; three lift the estimate.
            assert!(
                deep <= 3 * (probe_after + 1) + 2,
                "{probe_after}: {trace:?}"
            );
        }
    }

    #[test]
    fn decisions_are_repeatable_and_bounded_by_the_limit() {
        let config = AdaptiveDepth::neural();
        let easy = |i: usize| i % 9 < 5;
        let first = simulate(&config, 7, easy, 300);
        assert_eq!(first, simulate(&config, 7, easy, 300));
        assert!(first.iter().all(|&(_, d)| d <= 7));
        let mut controller = DepthController::new(&config, 7);
        assert_eq!(controller.depth(0), 0);
        assert!(controller.depth(2) <= 2);
        assert_eq!(DepthController::new(&config, 0).depth(5), 0);
    }

    #[test]
    fn partially_known_proposals_wait_for_their_outcome() {
        let mut config = AdaptiveDepth::neural();
        config.rate = 1.0;
        let mut controller = DepthController::new(&config, 4);
        controller.proposed(0, &[10, 11, 12, 13]);
        controller.observe(&[10, 11]);
        assert_eq!(controller.estimate(), &config.prior[..4]);
        controller.observe(&[10, 11, 12, 99]);
        assert_eq!(controller.estimate(), &[1.0, 1.0, 1.0, 0.0]);
        controller.proposed(4, &[]);
        controller.observe(&[10, 11, 12, 99]);
        assert_eq!(controller.estimate(), &[0.0; 4]);
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        let mut config = AdaptiveDepth::neural();
        assert!(config.validate().is_ok());
        config.rate = 0.0;
        assert!(config.validate().is_err());
        config = AdaptiveDepth::neural();
        config.costs.pass = vec![2.0, 1.0];
        assert!(config.validate().is_err());
        config.costs.pass = vec![];
        assert!(config.validate().is_err());
        config = AdaptiveDepth::neural();
        config.prior[0] = 1.5;
        assert!(config.validate().is_err());
        config = AdaptiveDepth::neural();
        config.relax = 1.5;
        assert!(config.validate().is_err());
    }
}
