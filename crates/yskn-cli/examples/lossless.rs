//! Real-model losslessness check through the product's request and chat path.
mod lossless_support;

use std::path::PathBuf;
use yskn::{
    Error, Result,
    real_engine::{RealEngine, RealEngineConfig},
    request::RequestDefaults,
};

fn run() -> Result<bool> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut prompts, mut report) = (None, None, None);
    let mut max_tokens = 96;
    let mut names = lossless_support::configs(&lossless_support::CONFIG_NAMES.join(","))?;
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "lossless --model-dir MODEL_FOLDER [--prompts FILE] [--max-tokens N] [--configs LIST] [--report lossless.json]\nLIST: {} (none is always included first)",
                lossless_support::CONFIG_NAMES.join(",")
            );
            return Ok(true);
        }
        if ![
            "--model-dir",
            "--prompts",
            "--max-tokens",
            "--configs",
            "--report",
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
            "--configs" => names = lossless_support::configs(&value)?,
            "--max-tokens" => {
                max_tokens = value
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| Error::request("--max-tokens must be positive"))?
            }
            _ => unreachable!(),
        }
    }
    let model_dir = model.ok_or_else(|| Error::request("--model-dir is required"))?;
    let prompts = lossless_support::load_prompts(prompts.as_deref())?;
    let base = RealEngineConfig {
        model_dir,
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
        |name| {
            // RealEngine::load inspects AND enforces the physical-memory estimate.
            Ok(Box::new(RealEngine::load(lossless_support::configure(
                &base, name,
            )?)?))
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
