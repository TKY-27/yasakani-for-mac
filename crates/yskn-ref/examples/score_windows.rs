//! Score the supplied token windows with the chapter 06 head and portable math.
#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::process::ExitCode;

use serde::Deserialize;
use yskn_ref::output_head_sampler::negative_log_likelihood;
use yskn_ref::portable_math::exp_d;
use yskn_ref::{Error, Model, Result};

#[derive(Deserialize)]
struct TokenWindows {
    n_ctx: usize,
    first_scored_position: usize,
    last_scored_position: usize,
    windows: Vec<TokenWindow>,
}

#[derive(Deserialize)]
struct TokenWindow {
    ids: Vec<u32>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() == 1 && (arguments[0] == "--help" || arguments[0] == "-h") {
        println!(
            "Usage: cargo run --release -p yskn-ref --example score_windows -- MODEL_FOLDER TOKEN_WINDOWS.json"
        );
        println!(
            "Scores input/logit rows 1024..=2046 against tokens 1025..=2047 in each 2048-token window."
        );
        return Ok(());
    }
    if arguments.len() != 2 {
        return Err(Error::InvalidInput(
            "expected MODEL_FOLDER and TOKEN_WINDOWS.json; use --help for usage".into(),
        ));
    }
    let windows: TokenWindows = serde_json::from_slice(&fs::read(&arguments[1])?)?;
    if windows.n_ctx != 2048
        || windows.first_scored_position != 1024
        || windows.last_scored_position != 2046
    {
        return Err(Error::InvalidInput(
            "windows must specify n_ctx=2048 and scored positions 1024..=2046".into(),
        ));
    }
    if windows.windows.is_empty()
        || windows
            .windows
            .iter()
            .any(|window| window.ids.len() != windows.n_ctx)
    {
        return Err(Error::InvalidInput(
            "expected at least one complete 2048-token window".into(),
        ));
    }
    let mut model = Model::load_folder(&arguments[0])?;
    if windows
        .windows
        .iter()
        .flat_map(|window| &window.ids)
        .any(|&token| token as usize >= model.dimensions().vocabulary_size)
    {
        return Err(Error::InvalidInput(
            "window token is outside the model vocabulary".into(),
        ));
    }
    let mut total_loss = 0.0_f64;
    let mut scored_positions = 0_usize;
    for (index, window) in windows.windows.iter().enumerate() {
        model.reset()?;
        // One-row consumption bounds logit storage. Position p predicts ids[p+1].
        for position in 0..=windows.last_scored_position {
            model.prefill(&window.ids[position..position + 1])?;
            if position >= windows.first_scored_position {
                let logits = model.logits().ok_or_else(|| {
                    Error::InvalidModel("a consumed row produced no logits".into())
                })?;
                let loss = negative_log_likelihood(logits, window.ids[position + 1])?;
                // Reporting also has a fixed order: windows, then ascending positions.
                total_loss += loss;
                scored_positions += 1;
            }
        }
        eprintln!("Scored window {}/{}", index + 1, windows.windows.len());
    }
    let mean_loss = total_loss / scored_positions as f64;
    let perplexity = exp_d(mean_loss);
    if !mean_loss.is_finite() || !perplexity.is_finite() {
        return Err(Error::NonFinite("aggregate quality metrics"));
    }
    println!("scored_positions: {scored_positions}");
    println!("mean_negative_log_likelihood: {mean_loss:.12}");
    println!("perplexity: {perplexity:.12}");
    Ok(())
}
