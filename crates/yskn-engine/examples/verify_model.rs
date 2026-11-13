use yskn_engine::{Engine, Error, Kernel, Model, Result};
use yskn_ref::quality::TokenWindows;

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 2 {
        return Err(Error::InvalidInput(
            "Usage: verify_model MODEL_FOLDER WINDOWS.json".into(),
        ));
    }
    let windows = TokenWindows::load(&arguments[1])?;
    let tokens = &windows.windows[0].ids[..24];
    let mut expected: Option<Vec<Vec<f32>>> = None;
    // One mapping and one model at a time, including across configurations.
    for (kernel, threads, rows) in [
        (Kernel::fastest(), Engine::default_threads(), 16),
        (Kernel::fastest(), Engine::default_threads(), 1),
        (Kernel::Portable, Engine::default_threads(), 5),
        (Kernel::fastest(), 1, 7),
    ] {
        // SAFETY: this command requires read-only weights to stay unchanged until exit.
        let mut model =
            unsafe { Model::with_engine(&arguments[0], Engine::new(kernel, threads)?, rows)? };
        model.prefill(tokens)?;
        if let Some(expected) = &expected {
            for (row, (actual, expected)) in model.row_logits().iter().zip(expected).enumerate() {
                if actual.len() != expected.len() {
                    return Err(Error::InvalidModel("logit widths differ".into()));
                }
                if let Some((vocabulary, _)) = actual
                    .iter()
                    .zip(expected)
                    .enumerate()
                    .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                {
                    eprintln!("FAIL row={row} vocabulary_index={vocabulary}");
                    return Err(Error::InvalidModel("model configurations differ".into()));
                }
            }
        } else {
            expected = Some(model.row_logits().to_vec());
        }
        println!(
            "PASS rows=24 kernel={} threads={threads} rows_per_pass={rows}",
            kernel.name()
        );
    }
    println!("PASS all 24 logit rows are bit-identical across four configurations");
    Ok(())
}
