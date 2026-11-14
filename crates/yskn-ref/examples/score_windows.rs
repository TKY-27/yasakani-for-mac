//! Score supplied windows using the shared chapter-06 quality definition.
#![forbid(unsafe_code)]
use std::process::ExitCode;
use yskn_ref::{
    Error, KvMode, Model, Result,
    quality::{TokenWindows, parse_kv_mode, score_with_kv},
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
        println!(
            "Usage: score_windows MODEL_FOLDER TOKEN_WINDOWS.json [--kv int8|int8-profile|kv5] [--json OUT]"
        );
        println!(
            "Scores input/logit rows 1024..=2046 against tokens 1025..=2047 in each 2048-token window."
        );
        return Ok(());
    }
    let mut mode = KvMode::Int8;
    let mut paths = Vec::new();
    let mut json = None;
    let mut args = arguments.iter();
    while let Some(argument) = args.next() {
        if argument == "--kv" {
            mode = parse_kv_mode(
                args.next()
                    .ok_or_else(|| Error::InvalidInput("--kv needs a mode".into()))?,
            )?;
        } else if argument == "--json" {
            json = Some(
                args.next()
                    .ok_or_else(|| Error::InvalidInput("--json needs a path".into()))?,
            );
        } else if argument.to_string_lossy().starts_with('-') {
            return Err(Error::InvalidInput(
                "unknown option; use --help for usage".into(),
            ));
        } else {
            paths.push(argument);
        }
    }
    if paths.len() != 2 {
        return Err(Error::InvalidInput(
            "expected MODEL_FOLDER and TOKEN_WINDOWS.json; use --help for usage".into(),
        ));
    }
    let windows = TokenWindows::load(paths[1])?;
    let result = score_with_kv(
        mode,
        |mode| Model::load_folder_with(paths[0], mode),
        &windows,
        0..windows.windows.len(),
        1,
        |_| {},
    )?;
    result.print();
    if let Some(path) = json {
        let output = serde_json::json!({"kv_mode": mode.name(), "score": result});
        std::fs::write(path, serde_json::to_vec_pretty(&output)?)?;
    }
    Ok(())
}
