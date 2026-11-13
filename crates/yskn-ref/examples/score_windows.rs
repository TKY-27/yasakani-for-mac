//! Score supplied windows using the shared chapter-06 quality definition.
#![forbid(unsafe_code)]
use yskn_ref::{
    Error, Model, Result,
    quality::{TokenWindows, score},
};
fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && (arguments[0] == "--help" || arguments[0] == "-h") {
        println!("Usage: score_windows MODEL_FOLDER TOKEN_WINDOWS.json");
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
