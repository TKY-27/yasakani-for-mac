//! Score supplied windows using the shared chapter-06 quality definition.
#![forbid(unsafe_code)]
use std::process::ExitCode;
use yskn_ref::{
    Error, Model, Result,
    quality::{TokenWindows, score},
};
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
fn run() -> Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && (arguments[0] == "--help" || arguments[0] == "-h") {
        println!("Usage: score_windows MODEL_FOLDER TOKEN_WINDOWS.json");
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
    let windows = TokenWindows::load(&arguments[1])?;
    let mut model = Model::load_folder(&arguments[0])?;
    score(&mut model, &windows, 0..windows.windows.len(), 1, |_| {})?.print();
    Ok(())
}
