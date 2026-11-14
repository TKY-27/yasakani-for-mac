//! Real-model losslessness check through the product's request and chat path.
mod lossless_support;

use std::path::PathBuf;
use yskn::{
    Error, Result,
    engine::{Engine, Generation, Sampling},
    real_engine::{RealEngine, RealEngineConfig},
    request::RequestDefaults,
};

// Deliberate fault switch confined to this development example.
struct InjectMismatch {
    inner: RealEngine,
    injected: bool,
}
impl Engine for InjectMismatch {
    fn context_capacity(&self) -> usize {
        self.inner.context_capacity()
    }
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stops: Option<&[u32]>,
        emit: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation> {
        let mut g = self
            .inner
            .generate(prompt, max_new, sampling, stops, emit)?;
        if !self.injected {
            if let Some(id) = g.ids.first_mut() {
                *id ^= 1;
                self.injected = true;
            }
        }
        Ok(g)
    }
}

fn run() -> Result<bool> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut prompts, mut report) = (None, None, None);
    let mut max_tokens = 256;
    let mut context = RealEngineConfig::default().context;
    let mut names = None;
    let mut profile = "release".to_string();
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "lossless --model-dir MODEL_FOLDER [--profile release|full|quick] [--prompts FILE] [--max-tokens N] [--ctx N] [--configs LIST] [--report lossless.json]\nLIST: {} (none is always included first)",
                lossless_support::CONFIG_NAMES.join(",")
            );
            return Ok(true);
        }
        if ![
            "--model-dir",
            "--prompts",
            "--max-tokens",
            "--ctx",
            "--configs",
            "--report",
            "--profile",
        ]
        .contains(&arg.as_str())
        {
            return Err(Error::request(format!("unknown option {arg}")));
        }
        let value = args
            .next()
            .ok_or_else(|| Error::request(format!("{arg} requires a value")))?;
        match arg.as_str() {
            "--model-dir" => model = Some(PathBuf::from(value)),
            "--prompts" => prompts = Some(PathBuf::from(value)),
            "--report" => report = Some(PathBuf::from(value)),
            "--configs" => names = Some(lossless_support::configs(&value)?),
            "--profile" => {
                lossless_support::profile_samplers(&value)?;
                profile = value;
            }
            "--max-tokens" | "--ctx" => {
                let n = value
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| Error::request(format!("{arg} must be positive")))?;
                if arg == "--ctx" {
                    context = n;
                } else {
                    max_tokens = n;
                }
            }
            _ => unreachable!(),
        }
    }
    let names = names.unwrap_or(lossless_support::profile_configs(&profile)?);
    let model_dir = model.ok_or_else(|| Error::request("--model-dir is required"))?;
    let prompts = lossless_support::load_prompts(prompts.as_deref())?;
    let base = RealEngineConfig {
        model_dir,
        context,
        ..RealEngineConfig::default()
    };
    let defaults = RequestDefaults {
        sampling: base.sampling_defaults()?,
        max_tokens,
        ..RequestDefaults::default()
    };
    let formatter = yskn_text::ChatFormatter::load(&base.model_dir)?;
    let result = lossless_support::check(
        &prompts,
        &names,
        &defaults,
        max_tokens,
        &profile,
        |name| {
            // RealEngine::load inspects AND enforces the physical-memory estimate.
            let inner = RealEngine::load(lossless_support::configure(&base, name)?)?;
            if std::env::var("YSKN_LOSSLESS_INJECT_MISMATCH").as_deref() == Ok("1") {
                Ok(Box::new(InjectMismatch {
                    inner,
                    injected: false,
                }) as Box<dyn Engine>)
            } else {
                Ok(Box::new(inner) as Box<dyn Engine>)
            }
        },
        &formatter,
    )?;
    let text = serde_json::to_string_pretty(&result).map_err(Error::engine)?;
    if let Some(path) = report {
        std::fs::write(&path, text)
            .map_err(|e| Error::engine(format!("{}: {e}", path.display())))?;
    } else {
        println!("{text}");
    }
    Ok(result["passed"] == true)
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(e) => {
            eprintln!("lossless: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
