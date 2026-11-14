// Port of python/yasakani/loopguard.py. Inputs are growing committed id arrays.
pub const PERIOD_MAX: usize = 16;
pub const RUN_REASONING: usize = 256;
pub const RUN_ANSWER: usize = 1024;
pub const DIGITS: usize = 1000;
pub const BRIDGE: &str = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";
pub const NOTICE: &str = "\n\n[loop guard: the same sequence kept repeating, so the thinking was cut here and the answer follows]\n";
pub const BUDGET_NOTICE: &str = "\n\n[thinking budget: the thinking reached its limit, so it was cut here and the answer follows]\n";

pub fn loop_period(ids: &[u32], start: usize, run: usize) -> usize {
    let n = ids.len();
    if n.saturating_sub(start) < run + PERIOD_MAX {
        return 0;
    }
    (1..=PERIOD_MAX)
        .find(|&p| ids[n - run..] == ids[n - run - p..n - p])
        .unwrap_or(0)
}
#[derive(Debug, Clone)]
pub struct LoopHit {
    pub where_: &'static str,
    pub kind: &'static str,
    pub at: usize,
    pub period: usize,
    pub run: usize,
}
pub struct LoopWatch {
    think_end: u32,
    digits: [u32; 10],
    base: usize,
    budget: usize,
    loops: bool,
    // None: still thinking, Some(index+1): answer starts here.
    answer_at: Option<usize>,
    checked: usize,
    dig_run: usize,
    hit: bool,
}
impl LoopWatch {
    pub fn new(
        think_end: u32,
        digits: [u32; 10],
        in_reasoning: bool,
        base: usize,
        budget: usize,
        loops: bool,
    ) -> Self {
        Self {
            think_end,
            digits,
            base,
            budget,
            loops,
            answer_at: if in_reasoning { None } else { Some(0) },
            checked: base,
            dig_run: 0,
            hit: false,
        }
    }
    pub fn check(&mut self, ids: &[u32]) -> Option<LoopHit> {
        let n = ids.len();
        if self.hit || n <= self.checked {
            return None;
        }
        let tail = &ids[self.checked..];
        if self.answer_at.is_none() {
            if let Some(i) = tail.iter().position(|&id| id == self.think_end) {
                self.answer_at = Some(self.checked + i + 1);
            } else {
                self.dig_run = tail
                    .iter()
                    .rposition(|id| !self.digits.contains(id))
                    .map_or(self.dig_run + tail.len(), |i| tail.len() - i - 1);
            }
        }
        self.checked = n;
        let where_ = if self.answer_at.is_none() {
            "reasoning"
        } else {
            "answer"
        };
        let period = loop_period(
            ids,
            self.answer_at.unwrap_or(self.base).max(self.base),
            if self.answer_at.is_none() {
                RUN_REASONING
            } else {
                RUN_ANSWER
            },
        );
        let (kind, run) = if self.loops && period > 0 {
            ("repeat", 0)
        } else if self.loops && self.answer_at.is_none() && self.dig_run >= DIGITS {
            ("digits", self.dig_run)
        } else if self.answer_at.is_none() && self.budget > 0 && n - self.base >= self.budget {
            ("budget", 0)
        } else {
            return None;
        };
        self.hit = true;
        Some(LoopHit {
            where_,
            kind,
            at: n,
            period,
            run,
        })
    }
}
