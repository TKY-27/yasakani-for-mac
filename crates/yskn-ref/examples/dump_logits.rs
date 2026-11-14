//! `dump_logits MODEL_FOLDER WINDOWS.json ROWS OUT.json`
//!
//! Runs the scalar reference on the first ROWS tokens of window 0 and writes
//! `{"tokens": [...], "logits_bits": [[u32; vocabulary]; ROWS]}` for
//! `yskn-engine`'s `verify_model --reference`. Loads all weights onto the heap.
#![forbid(unsafe_code)]
use std::{fs::File, io::BufWriter, process::ExitCode, time::Instant};
use yskn_ref::{Error, Model, Result, quality::TokenWindows};

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
    let [folder, windows, rows, out] = arguments.as_slice() else {
        return Err(Error::InvalidInput(
            "usage: dump_logits MODEL_FOLDER WINDOWS.json ROWS OUT.json".into(),
        ));
    };
    let windows = TokenWindows::load(windows)?;
    let rows: usize = rows
        .to_str()
        .and_then(|r| r.parse().ok())
        .filter(|&r| r > 0 && r <= windows.windows[0].ids.len())
        .ok_or_else(|| Error::InvalidInput("ROWS must be a positive token count".into()))?;
    let tokens = &windows.windows[0].ids[..rows];
    let start = Instant::now();
    let mut model = Model::load_folder(folder)?;
    eprintln!("loaded in {:.1} s", start.elapsed().as_secs_f64());
    model.prefill(tokens)?;
    eprintln!("{rows} rows in {:.1} s", start.elapsed().as_secs_f64());
    let logits_bits: Vec<Vec<u32>> = model
        .row_logits()
        .iter()
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .collect();
    serde_json::to_writer(
        BufWriter::new(File::create_new(out)?),
        &serde_json::json!({ "tokens": tokens, "logits_bits": logits_bits }),
    )?;
    Ok(())
}
