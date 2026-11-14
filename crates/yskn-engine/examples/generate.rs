//! Token-ID generation with the shared lossless speculation loop.
use std::error::Error as StdError;
use yskn_engine::{Backend, Engine, Error, KvMode, LogitRows, Model};
use yskn_ref::draft::Draft as NeuralDraft;
use yskn_ref::draft_head::DraftChoice;
use yskn_ref::model::{CaptureOptions, CapturedRow};
use yskn_ref::quality::parse_kv_mode;
use yskn_ref::speculation::{
    CombinedProposer, DraftProposer, GenerationLimits, NgramProposer, NoProposer, ProposalOrder,
    Proposer, Verifier, generate,
};
use yskn_ref::{Sampler, SamplingParameters, SamplingVersion};

const USAGE: &str = "generate MODEL_FOLDER --tokens 1,2,3 [--max-new N] [--draft neural|ngram|both|none] [--k K] [--ngram N] [--eos ID,...] [--seed S | --greedy] [--backend cpu|metal] [--kv int8|int8-profile|kv5]";

fn usage() -> Error {
    Error::InvalidInput(format!("usage: {USAGE}"))
}

fn token_ids(value: &str) -> Result<Vec<u32>, Error> {
    value
        .split(',')
        .map(|id| id.parse().map_err(|_| usage()))
        .collect()
}

enum Draft {
    None(NoProposer),
    Ngram(NgramProposer),
    Neural(Box<DraftProposer>),
    Both(Box<CombinedProposer>),
}
impl Proposer for Draft {
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        match self {
            Self::None(proposer) => proposer.propose(history, last, max),
            Self::Ngram(proposer) => proposer.propose(history, last, max),
            Self::Neural(proposer) => proposer.propose(history, last, max),
            Self::Both(proposer) => proposer.propose(history, last, max),
        }
    }
    fn synchronize<V: Verifier>(&mut self, verifier: &V) -> yskn_ref::Result<()> {
        match self {
            Self::None(proposer) => proposer.synchronize(verifier),
            Self::Ngram(proposer) => proposer.synchronize(verifier),
            Self::Neural(proposer) => proposer.synchronize(verifier),
            Self::Both(proposer) => proposer.synchronize(verifier),
        }
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> yskn_ref::Result<()> {
        match self {
            Self::None(proposer) => proposer.observe_committed(rows),
            Self::Ngram(proposer) => proposer.observe_committed(rows),
            Self::Neural(proposer) => proposer.observe_committed(rows),
            Self::Both(proposer) => proposer.observe_committed(rows),
        }
    }
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &V,
        history: &[u32],
        last: u32,
        max: usize,
    ) -> yskn_ref::Result<Vec<u32>> {
        match self {
            Self::None(proposer) => proposer.propose_with(verifier, history, last, max),
            Self::Ngram(proposer) => proposer.propose_with(verifier, history, last, max),
            Self::Neural(proposer) => proposer.propose_with(verifier, history, last, max),
            Self::Both(proposer) => proposer.propose_with(verifier, history, last, max),
        }
    }
}

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or_else(usage)?;
    if path == "--help" || path == "-h" {
        println!("Usage: {USAGE}");
        println!(
            "Neural drafting uses the scalar reference and can take about 90 seconds per proposal on the real model."
        );
        return Ok(());
    }
    let (mut prompt, mut eos) = (None, Vec::new());
    let (mut max_new, mut k, mut n) = (32_usize, 3_usize, 3_usize);
    let (mut seed, mut explicit_greedy) = (None, false);
    let mut draft = "none".to_owned();
    let mut backend = Backend::from_env()?;
    let mut mode = KvMode::Int8;
    while let Some(option) = args.next() {
        if option == "--greedy" {
            explicit_greedy = true;
            continue;
        }
        let value = args.next().ok_or_else(usage)?;
        match option.as_str() {
            "--tokens" => prompt = Some(token_ids(&value)?),
            "--eos" => eos = token_ids(&value)?,
            "--max-new" => max_new = value.parse::<usize>().map_err(|_| usage())?,
            "--k" => k = value.parse::<usize>().map_err(|_| usage())?,
            "--ngram" => n = value.parse::<usize>().map_err(|_| usage())?,
            "--seed" => seed = Some(value.parse::<u64>().map_err(|_| usage())?),
            "--draft" if matches!(value.as_str(), "neural" | "ngram" | "both" | "none") => {
                draft = value
            }
            "--backend" => backend = value.parse()?,
            "--kv" => mode = parse_kv_mode(value.as_ref())?,
            _ => return Err(usage().into()),
        }
    }
    if explicit_greedy && seed.is_some() {
        return Err(usage().into());
    }
    let prompt = prompt.ok_or_else(usage)?;
    if prompt.is_empty() {
        return Err(Error::InvalidInput("prompt must not be empty".into()).into());
    }
    let rows = k.checked_add(1).ok_or_else(usage)?.max(16);
    let mut capture: Option<CaptureOptions> = None;
    let mut proposer = match draft.as_str() {
        "none" => Draft::None(NoProposer),
        "ngram" => Draft::Ngram(NgramProposer::new(n, k)?),
        _ => {
            // The example's seeded main sampler uses version 3, the only coupled source mode.
            let choice = seed.map_or(DraftChoice::Greedy, DraftChoice::sampled);
            let neural = DraftProposer::new(NeuralDraft::load(&path)?, k, choice)?;
            capture = Some(neural.capture_options());
            eprintln!("scalar neural draft: real-model proposals can take about 90 seconds each");
            if draft == "both" {
                Draft::Both(Box::new(CombinedProposer::new(
                    NgramProposer::new(n, k)?,
                    neural,
                    ProposalOrder::NgramFirst,
                )))
            } else {
                Draft::Neural(Box::new(neural))
            }
        }
    };
    // SAFETY: callers must keep the read-only weights file unchanged until exit.
    let mut model =
        unsafe { Model::with_options(&path, Engine::from_env()?, backend, rows, mode)? };
    if let Some(options) = capture {
        model.set_capture(options)?;
    }
    if prompt
        .iter()
        .any(|&id| id as usize >= model.dimensions().vocabulary_size)
    {
        return Err(Error::InvalidInput("prompt token exceeds vocabulary".into()).into());
    }
    if prompt.len() > model.context_capacity() {
        return Err(Error::InvalidInput("prompt exceeds context capacity".into()).into());
    }
    // The loop consumes the final prompt token in its first verification pass.
    model.prefill_with_logits(&prompt[..prompt.len() - 1], LogitRows::None)?;
    let mut selector = seed.map_or(Sampler::Greedy, |seed| {
        Sampler::Sampled(SamplingParameters {
            seed,
            version: SamplingVersion::Gumbel,
            top_k: model.dimensions().vocabulary_size.min(20),
            ..SamplingParameters::default()
        })
    });
    let mut limits = GenerationLimits::new(max_new, model.context_capacity());
    limits.max_draft_tokens = if draft == "none" { 0 } else { k };
    limits.eos_ids = &eos;
    let result = generate(&mut model, &mut proposer, &mut selector, &prompt, limits)?;
    println!(
        "{}",
        result
            .tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let stats = result.statistics;
    println!(
        "stop={:?} passes={} rows_verified={} guesses_offered={} guesses_accepted={} tokens_confirmed_per_pass={:?}",
        result.stop_reason,
        stats.passes,
        stats.rows_verified,
        stats.guesses_offered,
        stats.guesses_accepted,
        stats.tokens_confirmed_per_pass,
    );
    println!(
        "milliseconds_per_pass={:?}",
        stats
            .pass_durations
            .iter()
            .map(|elapsed| elapsed.as_secs_f64() * 1000.0)
            .collect::<Vec<_>>()
    );
    eprintln!(
        "backend={} device={} kv={mode:?} position={}",
        model.backend().name(),
        model.device(),
        model.position()
    );
    Ok(())
}
