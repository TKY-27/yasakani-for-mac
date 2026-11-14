//! `verify_model MODEL_FOLDER WINDOWS.json [--kv int8|int8-profile|kv5] [--reference LOGITS.json]`
//!
//! Runs the first 24 tokens of window 0 in six backend/kernel/pass-width
//! configurations and requires all logit bits to agree. With `--reference`,
//! the first rows are also compared with the scalar reference's logits written
//! by `yskn-ref`'s `dump_logits`.
use yskn_engine::{Backend, Engine, Error, Kernel, KvMode, Model, Result};
use yskn_ref::quality::{TokenWindows, parse_kv_mode};

const ROWS: usize = 24;

fn usage() -> Error {
    Error::InvalidInput(
        "Usage: verify_model MODEL_FOLDER WINDOWS.json [--kv int8|int8-profile|kv5] [--reference LOGITS.json]".into(),
    )
}

/// The first mismatching (row, vocabulary index), if any.
fn first_difference(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> Result<Option<(usize, usize)>> {
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if actual.len() != expected.len() {
            return Err(Error::InvalidModel("logit widths differ".into()));
        }
        if let Some(vocabulary) =
            (0..actual.len()).find(|&v| actual[v].to_bits() != expected[v].to_bits())
        {
            return Ok(Some((row, vocabulary)));
        }
    }
    Ok(None)
}

/// Tokens and logit bit rows written by `dump_logits`.
fn load_reference(path: &str, mode: KvMode) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    parse_reference(&value, path, mode)
}

fn parse_reference(
    value: &serde_json::Value,
    path: &str,
    mode: KvMode,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let invalid = || Error::InvalidInput(format!("{path} is not a dump_logits file"));
    let reference_mode = match value.get("kv_mode") {
        None => KvMode::Int8,
        Some(value) => parse_kv_mode(value.as_str().ok_or_else(invalid)?.as_ref())?,
    };
    if reference_mode != mode {
        return Err(Error::InvalidInput(format!(
            "reference kv_mode={} differs from --kv {}",
            reference_mode.name(),
            mode.name()
        )));
    }
    let tokens = value["tokens"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|t| {
            t.as_u64()
                .and_then(|t| u32::try_from(t).ok())
                .ok_or_else(invalid)
        })
        .collect::<Result<Vec<_>>>()?;
    let rows = value["logits_bits"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(invalid)?
                .iter()
                .map(|b| {
                    b.as_u64()
                        .and_then(|b| u32::try_from(b).ok())
                        .map(f32::from_bits)
                        .ok_or_else(invalid)
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.is_empty() || rows.len() != tokens.len() || rows.len() > ROWS {
        return Err(Error::InvalidInput(format!(
            "{path} must hold 1..={ROWS} rows, one per token"
        )));
    }
    Ok((tokens, rows))
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() < 2 {
        return Err(usage());
    }
    let mut mode = KvMode::Int8;
    let mut reference = None;
    let mut options = arguments[2..].iter();
    while let Some(option) = options.next() {
        let value = options.next().ok_or_else(usage)?;
        match option.as_str() {
            "--kv" => mode = parse_kv_mode(value.as_ref())?,
            "--reference" => reference = Some(value),
            _ => return Err(usage()),
        }
    }
    let reference = reference
        .map(|path| load_reference(path, mode))
        .transpose()?;
    let windows = TokenWindows::load(&arguments[1])?;
    let tokens = &windows.windows[0].ids[..ROWS];
    let threads = Engine::default_threads();
    let configurations = [
        (Backend::Cpu, Kernel::fastest(), threads, 16),
        (Backend::Cpu, Kernel::fastest(), threads, 1),
        (Backend::Cpu, Kernel::Portable, threads, 5),
        (Backend::Cpu, Kernel::fastest(), 1, 7),
        (Backend::Metal, Kernel::fastest(), threads, 16),
        (Backend::Metal, Kernel::fastest(), threads, 1),
    ];
    let mut expected: Option<Vec<Vec<f32>>> = None;
    // One mapping and one model at a time, including across configurations.
    for (backend, kernel, threads, rows) in configurations {
        // SAFETY: this command requires read-only weights to stay unchanged until exit.
        let mut model = unsafe {
            Model::with_options(
                &arguments[0],
                Engine::new(kernel, threads)?,
                backend,
                rows,
                mode,
            )?
        };
        model.prefill(tokens)?;
        let label = format!(
            "rows={ROWS} backend={} device={} kernel={} threads={threads} rows_per_pass={rows} kv={mode:?}",
            backend.name(),
            model.device(),
            kernel.name()
        );
        let Some(expected) = &expected else {
            println!("BASELINE {label}");
            expected = Some(model.row_logits().to_vec());
            continue;
        };
        if let Some((row, vocabulary)) = first_difference(model.row_logits(), expected)? {
            eprintln!("FAIL {label} row={row} vocabulary_index={vocabulary}");
            return Err(Error::InvalidModel("model configurations differ".into()));
        }
        println!("PASS {label} equals BASELINE");
    }
    println!(
        "PASS all {ROWS} logit rows are bit-identical across {} configurations",
        configurations.len()
    );
    if let (Some((reference_tokens, reference_rows)), Some(engine_rows)) = (reference, expected) {
        if reference_tokens != tokens[..reference_tokens.len()] {
            return Err(Error::InvalidInput(
                "reference tokens are not the first tokens of window 0".into(),
            ));
        }
        let engine_rows = &engine_rows[..reference_rows.len()];
        if let Some((row, vocabulary)) = first_difference(engine_rows, &reference_rows)? {
            eprintln!("FAIL reference row={row} vocabulary_index={vocabulary}");
            return Err(Error::InvalidModel(
                "engine differs from the scalar reference".into(),
            ));
        }
        println!(
            "PASS the first {} logit rows equal the scalar reference",
            reference_rows.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_mode_is_checked_before_logits_and_legacy_means_int8() {
        for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
            let dump =
                serde_json::json!({"kv_mode": mode.name(), "tokens": [7], "logits_bits": [[0]]});
            assert!(parse_reference(&dump, "dump", mode).is_ok());
            for selected in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
                if selected != mode {
                    // No logits: a mismatch must fail before logit parsing/comparison.
                    let dump = serde_json::json!({"kv_mode": mode.name()});
                    let error = parse_reference(&dump, "dump", selected)
                        .unwrap_err()
                        .to_string();
                    assert!(error.contains(&format!("reference kv_mode={}", mode.name())));
                    assert!(error.contains(&format!("--kv {}", selected.name())));
                }
            }
        }
        let legacy = serde_json::json!({"tokens": [7], "logits_bits": [[0]]});
        assert!(parse_reference(&legacy, "dump", KvMode::Int8).is_ok());
        for mode in [KvMode::Int8Profile, KvMode::Kv5] {
            let error = parse_reference(&legacy, "dump", mode)
                .unwrap_err()
                .to_string();
            assert!(error.contains("reference kv_mode=int8"));
            assert!(error.contains(&format!("--kv {}", mode.name())));
        }
    }
}
