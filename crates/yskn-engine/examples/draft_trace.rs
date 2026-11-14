//! Draft depth research (D21).
//!
//! `draft_trace record MODEL_FOLDER PROMPTS.jsonl --n N --out TRACE.json`
//! generates N greedy tokens per prompt with one-row passes and, before every
//! pass, records the fast neural draft's full seven-token proposal, its
//! per-position confidence margin (best minus second corrected score) and the
//! n-gram proposal with its match length. It also times verification passes of
//! 1..=8 rows and the proposals. Because output never depends on the proposer,
//! the trace determines exactly what any depth policy would do.
//!
//! `draft_trace replay TRACE.json` replays policies over the trace and prints
//! exact passes, rows and proposer calls, and time projected from the measured
//! costs (`--table` uses the shipped M5 table instead).
use std::error::Error as StdError;
use std::time::Instant;

use serde_json::{Value, json};
use yskn_engine::{Backend, Engine, FastDraft, KvMode, LogitRows, Model};
use yskn_ref::draft_head::DraftChoice;
use yskn_ref::model::CapturedRow;
use yskn_ref::speculation::{
    AdaptiveDepth, CostModel, DepthController, DraftProposer, GenerationLimits, Greedy,
    NgramProposer, Proposer, Verifier, generate,
};

type Result<T> = std::result::Result<T, Box<dyn StdError>>;

const USAGE: &str = "draft_trace record MODEL_FOLDER PROMPTS.jsonl --n N --out TRACE.json | draft_trace replay TRACE.json [--table]";
const DEPTH: usize = 7;

#[derive(Default)]
struct Record {
    neural: Vec<Vec<u32>>,
    margins: Vec<Vec<f32>>,
    ngram: Vec<Vec<u32>>,
    ngram_match: Vec<usize>,
    proposal_ms: Vec<f64>,
}

/// Records every proposal and offers nothing, so each pass confirms one token.
struct Recorder {
    neural: DraftProposer<FastDraft>,
    ngram: NgramProposer,
    record: Record,
}

fn match_length(history: &[u32], last: u32, n: usize) -> usize {
    let known: Vec<u32> = history.iter().copied().chain([last]).collect();
    if known.len() <= n {
        return 0;
    }
    let suffix = known.len() - n;
    let Some(start) = (0..suffix)
        .rev()
        .find(|&s| known[s..s + n] == known[suffix..suffix + n])
    else {
        return 0;
    };
    let extra = (1..=start)
        .take_while(|&i| known[start - i] == known[suffix - i])
        .count();
    n + extra
}

impl Proposer for Recorder {
    fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
        Vec::new()
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &mut V) -> yskn_ref::Result<()> {
        self.neural.synchronize(verifier)
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> yskn_ref::Result<()> {
        self.neural.observe_committed(rows)
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &mut V,
        history: &[u32],
        last: u32,
        _: usize,
    ) -> yskn_ref::Result<Vec<u32>> {
        let started = Instant::now();
        let proposal = self.neural.propose_block(verifier, last)?;
        self.record
            .proposal_ms
            .push(started.elapsed().as_secs_f64() * 1000.0);
        let margins = proposal
            .corrected
            .iter()
            .map(|candidates| {
                let mut scores: Vec<f32> = candidates.iter().map(|c| c.score).collect();
                scores.sort_by(|a, b| b.total_cmp(a));
                scores[0] - scores[1]
            })
            .collect();
        self.record.neural.push(proposal.tokens);
        self.record.margins.push(margins);
        self.record
            .ngram
            .push(self.ngram.propose(history, last, DEPTH));
        self.record.ngram_match.push(match_length(history, last, 3));
        Ok(Vec::new())
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn record(mut args: impl Iterator<Item = String>) -> Result<()> {
    let folder = args.next().ok_or(USAGE)?;
    let prompts = args.next().ok_or(USAGE)?;
    let (mut n, mut out) = (96_usize, None);
    while let Some(option) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match option.as_str() {
            "--n" => n = value.parse()?,
            "--out" => out = Some(value),
            _ => return Err(USAGE.into()),
        }
    }
    let out = out.ok_or(USAGE)?;
    let backend = Backend::from_env()?;
    // SAFETY: the read-only model folder stays unchanged until exit.
    let fast = unsafe { FastDraft::load(&folder)? };
    let neural = DraftProposer::new(fast, DEPTH, DraftChoice::Greedy)?;
    // SAFETY: as above.
    let mut model =
        unsafe { Model::with_options(&folder, Engine::from_env()?, backend, 16, KvMode::Kv5)? };
    model.set_capture(neural.capture_options())?;
    let mut recorder = Recorder {
        neural,
        ngram: NgramProposer::new(3, DEPTH)?,
        record: Record::default(),
    };
    let mut traces = Vec::new();
    for line in std::fs::read_to_string(prompts)?.lines() {
        let prompt: Value = serde_json::from_str(line)?;
        let task = prompt["task"].as_str().ok_or("task")?.to_owned();
        let tokens: Vec<u32> = serde_json::from_value(prompt["tokens"].clone())?;
        model.reset()?;
        recorder.neural.reset();
        recorder.record = Record::default();
        model.prefill_with_logits(&tokens[..tokens.len() - 1], LogitRows::None)?;
        let last = tokens[tokens.len() - 1];
        let mut pass_ms = Vec::new();
        for rows in 1..=8 {
            let probe = vec![last; rows];
            let samples = (0..3)
                .map(|_| {
                    let started = Instant::now();
                    model.run_provisional(&probe)?;
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    model.discard_provisional();
                    Ok(elapsed)
                })
                .collect::<Result<Vec<_>>>()?;
            pass_ms.push(median(samples));
        }
        let mut limits = GenerationLimits::new(n, model.context_capacity());
        limits.max_draft_tokens = DEPTH;
        let started = Instant::now();
        let result = generate(&mut model, &mut recorder, &mut Greedy, &tokens, limits)?;
        let generation_ms = started.elapsed().as_secs_f64() * 1000.0;
        let r = &recorder.record;
        eprintln!(
            "{task}: pass_ms={pass_ms:.1?} proposal_median_ms={:.1} generation_ms={generation_ms:.0}",
            median(r.proposal_ms.clone())
        );
        traces.push(json!({
            "task": task, "prompt_tokens": tokens.len(), "tokens": result.tokens,
            "neural": r.neural, "margins": r.margins, "ngram": r.ngram,
            "ngram_match": r.ngram_match, "pass_ms": pass_ms,
            "proposal_ms": median(r.proposal_ms.clone()),
        }));
    }
    std::fs::write(out, serde_json::to_vec(&json!({"n": n, "traces": traces}))?)?;
    Ok(())
}

struct Trace {
    task: String,
    tokens: Vec<u32>,
    neural: Vec<Vec<u32>>,
    margins: Vec<Vec<f32>>,
    ngram: Vec<Vec<u32>>,
    ngram_match: Vec<usize>,
    costs: CostModel,
}

#[derive(Clone, Copy, Debug)]
enum Source {
    Neural,
    Ngram,
    /// N-gram when its match is at least this long and nonempty, else neural.
    Gate(usize),
    /// As `Gate`, but a long n-gram match is offered before the depth policy
    /// is consulted, which then sees only neural proposals.
    Free(usize),
}

#[derive(Clone, Debug)]
enum Depth {
    Fixed(usize),
    Adaptive(AdaptiveDepth),
}

#[derive(Default, Clone, Copy)]
struct Outcome {
    passes: usize,
    rows: usize,
    neural_calls: usize,
    ms: f64,
}

fn replay(trace: &Trace, source: Source, depth: &Depth, confidence: f32) -> Outcome {
    let n = trace.tokens.len();
    let max = match depth {
        Depth::Fixed(k) => *k,
        Depth::Adaptive(_) => DEPTH,
    };
    let mut controller = match depth {
        Depth::Adaptive(config) => {
            let mut config = config.clone();
            config.costs = CostModel {
                proposal: match source {
                    Source::Ngram => 0.0,
                    _ => trace.costs.proposal,
                },
                ..trace.costs.clone()
            };
            Some(DepthController::new(&config, max))
        }
        Depth::Fixed(_) => None,
    };
    let mut outcome = Outcome::default();
    let mut t = 0;
    while t < n {
        let limit = max.min(n - t - 1);
        let free = match source {
            Source::Free(length)
                if !trace.ngram[t].is_empty() && trace.ngram_match[t] >= length =>
            {
                Some(trace.ngram[t][..trace.ngram[t].len().min(limit)].to_vec())
            }
            _ => None,
        };
        if let Some(offered) = free {
            let accepted = offered
                .iter()
                .zip(&trace.tokens[t..])
                .take_while(|(g, o)| g == o)
                .count();
            outcome.passes += 1;
            outcome.rows += offered.len() + 1;
            outcome.ms += trace.costs.pass(offered.len() + 1);
            t += accepted + 1;
            if let Some(c) = &mut controller {
                c.observe(&trace.tokens[..t]);
            }
            continue;
        }
        let wanted = match &mut controller {
            Some(c) => c.depth(limit),
            None => limit,
        };
        let mut offered: Vec<u32> = Vec::new();
        if wanted > 0 || (controller.is_none() && limit > 0) {
            let use_ngram = match source {
                Source::Neural | Source::Free(_) => false,
                Source::Ngram => true,
                Source::Gate(length) => {
                    !trace.ngram[t].is_empty() && trace.ngram_match[t] >= length
                }
            };
            let proposal = if use_ngram {
                trace.ngram[t].clone()
            } else {
                outcome.neural_calls += 1;
                outcome.ms += trace.costs.proposal;
                let keep = trace.margins[t]
                    .iter()
                    .take_while(|&&m| m >= confidence)
                    .count();
                trace.neural[t][..keep.max(1)].to_vec()
            };
            let proposal = &proposal[..proposal.len().min(limit)];
            if let Some(c) = &mut controller {
                c.proposed(t, proposal);
            }
            offered = proposal[..proposal.len().min(wanted)].to_vec();
        }
        let accepted = offered
            .iter()
            .zip(&trace.tokens[t..])
            .take_while(|(g, o)| g == o)
            .count();
        outcome.passes += 1;
        outcome.rows += offered.len() + 1;
        outcome.ms += trace.costs.pass(offered.len() + 1);
        t += accepted + 1;
        if let Some(c) = &mut controller {
            c.observe(&trace.tokens[..t]);
        }
    }
    outcome
}

fn load(path: &str, table: bool) -> Result<Vec<Trace>> {
    let value: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let traces = value["traces"].as_array().ok_or("traces")?;
    traces
        .iter()
        .map(|t| {
            let costs = if table {
                CostModel::apple_m5_neural()
            } else {
                CostModel {
                    pass: serde_json::from_value(t["pass_ms"].clone())?,
                    proposal: t["proposal_ms"].as_f64().ok_or("proposal_ms")?,
                }
            };
            Ok(Trace {
                task: t["task"].as_str().ok_or("task")?.into(),
                tokens: serde_json::from_value(t["tokens"].clone())?,
                neural: serde_json::from_value(t["neural"].clone())?,
                margins: serde_json::from_value(t["margins"].clone())?,
                ngram: serde_json::from_value(t["ngram"].clone())?,
                ngram_match: serde_json::from_value(t["ngram_match"].clone())?,
                costs,
            })
        })
        .collect()
}

fn report(traces: &[Trace], name: &str, source: Source, depth: &Depth, confidence: f32) -> f64 {
    let mut line = format!("{name:<34}");
    let (mut tokens, mut ms) = (0, 0.0);
    for trace in traces {
        let o = replay(trace, source, depth, confidence);
        let n = trace.tokens.len();
        line += &format!(
            " {}: {:.1}ms/tok {:.2}tok/pass {}rows {}calls |",
            trace.task,
            o.ms / n as f64,
            n as f64 / o.passes as f64,
            o.rows,
            o.neural_calls
        );
        tokens += n;
        ms += o.ms;
    }
    let per_token = ms / tokens as f64;
    println!("{line} all: {per_token:.1}ms/tok");
    per_token
}

fn replay_main(mut args: impl Iterator<Item = String>) -> Result<()> {
    let path = args.next().ok_or(USAGE)?;
    let table = args.next().as_deref() == Some("--table");
    let traces = load(&path, table)?;
    for trace in &traces {
        // Counterfactual acceptance profile of a proposal at every position.
        let mut neural = [0_usize; DEPTH];
        let mut ngram = [0_usize; DEPTH];
        let n = trace.tokens.len();
        for t in 0..n {
            for (counts, guesses) in [
                (&mut neural, &trace.neural[t]),
                (&mut ngram, &trace.ngram[t]),
            ] {
                let m = guesses
                    .iter()
                    .zip(&trace.tokens[t..])
                    .take_while(|(g, o)| g == o)
                    .count();
                for c in &mut counts[..m] {
                    *c += 1;
                }
            }
        }
        let rate = |c: &[usize; DEPTH]| c.map(|x| format!("{:.3}", x as f64 / n as f64)).join(" ");
        println!(
            "{}: n={n} pass_ms={:.0?} proposal_ms={:.1}\n  neural P(prefix>=j) {}\n  ngram  P(prefix>=j) {}",
            trace.task,
            trace.costs.pass,
            trace.costs.proposal,
            rate(&neural),
            rate(&ngram)
        );
        // Confidence: acceptance of guess j given that guesses before it match,
        // split by margin.
        for threshold in [0.5_f32, 1.0, 2.0, 4.0] {
            let (mut low, mut low_hit, mut high, mut high_hit) = (0, 0, 0, 0);
            for t in 0..n {
                for j in 0..DEPTH.min(n - t) {
                    if trace.neural[t][..j] != trace.tokens[t..t + j] {
                        break;
                    }
                    let hit = trace.neural[t][j] == trace.tokens[t + j];
                    if trace.margins[t][j] < threshold {
                        (low, low_hit) = (low + 1, low_hit + usize::from(hit));
                    } else {
                        (high, high_hit) = (high + 1, high_hit + usize::from(hit));
                    }
                }
            }
            println!(
                "  margin<{threshold}: {low_hit}/{low} accepted; margin>={threshold}: {high_hit}/{high}"
            );
        }
    }
    report(&traces, "none", Source::Neural, &Depth::Fixed(0), f32::MIN);
    for k in 1..=DEPTH {
        report(
            &traces,
            &format!("ngram fixed {k}"),
            Source::Ngram,
            &Depth::Fixed(k),
            f32::MIN,
        );
    }
    for k in 1..=DEPTH {
        report(
            &traces,
            &format!("neural fixed {k}"),
            Source::Neural,
            &Depth::Fixed(k),
            f32::MIN,
        );
    }
    for k in [3, 7] {
        report(
            &traces,
            &format!("both(ngram first) fixed {k}"),
            Source::Gate(0),
            &Depth::Fixed(k),
            f32::MIN,
        );
    }
    let base = AdaptiveDepth::neural();
    report(
        &traces,
        "ngram adaptive",
        Source::Ngram,
        &Depth::Adaptive(base.clone()),
        f32::MIN,
    );
    report(
        &traces,
        "neural adaptive (default)",
        Source::Neural,
        &Depth::Adaptive(base.clone()),
        f32::MIN,
    );
    for rate in [0.1, 0.15, 0.25, 0.35, 0.5] {
        for probe_after in [2, 4, 8] {
            let config = AdaptiveDepth {
                rate,
                probe_after,
                ..base.clone()
            };
            report(
                &traces,
                &format!("neural adaptive rate={rate} probe={probe_after}"),
                Source::Neural,
                &Depth::Adaptive(config),
                f32::MIN,
            );
        }
    }
    for confidence in [0.25_f32, 0.5, 1.0, 2.0, 3.0] {
        report(
            &traces,
            &format!("neural adaptive margin>={confidence}"),
            Source::Neural,
            &Depth::Adaptive(base.clone()),
            confidence,
        );
        report(
            &traces,
            &format!("neural fixed 7 margin>={confidence}"),
            Source::Neural,
            &Depth::Fixed(7),
            confidence,
        );
    }
    for length in [3, 4, 6, 8, 12] {
        for probe_after in [2, 4, 8] {
            for relax in [0.0, 0.02, 0.05] {
                let config = AdaptiveDepth {
                    probe_after,
                    relax,
                    ..base.clone()
                };
                report(
                    &traces,
                    &format!("free ngram>={length} p{probe_after} r{relax}"),
                    Source::Free(length),
                    &Depth::Adaptive(config),
                    f32::MIN,
                );
            }
        }
    }
    for probe_after in [2, 4, 8] {
        for relax in [0.02, 0.05, 0.1] {
            for rate in [0.15, 0.25, 0.35] {
                let config = AdaptiveDepth {
                    probe_after,
                    relax,
                    rate,
                    ..base.clone()
                };
                report(
                    &traces,
                    &format!("neural p{probe_after} r{relax} rate{rate}"),
                    Source::Neural,
                    &Depth::Adaptive(config),
                    f32::MIN,
                );
            }
        }
    }
    for length in [3, 4, 6, 8, 12, 16] {
        report(
            &traces,
            &format!("gate ngram>={length} adaptive"),
            Source::Gate(length),
            &Depth::Adaptive(base.clone()),
            f32::MIN,
        );
        report(
            &traces,
            &format!("gate ngram>={length} fixed 7"),
            Source::Gate(length),
            &Depth::Fixed(7),
            f32::MIN,
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("record") => record(args),
        Some("replay") => replay_main(args),
        _ => Err(USAGE.into()),
    }
}
