//! `verify_speculation MODEL --tokens 1,2,3 [--accept N] [--backend cpu|metal] [--kv MODE]`
//! Loads the scalar oracle first, drops it, then checks mapped provisional/commit bits.
use yskn_engine::{Backend, Engine, Error, KvMode, Model, Result};

fn usage() -> Error {
    Error::InvalidInput("usage: verify_speculation MODEL --tokens IDS [--accept N] [--backend cpu|metal] [--kv int8|int8-profile|kv5]".into())
}
fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|r| r.iter().map(|v| v.to_bits()).collect())
        .collect()
}
fn equal(actual: &[Vec<f32>], expected: &[Vec<u32>], stage: &str) -> Result<()> {
    if bits(actual) != expected {
        return Err(Error::InvalidInput(format!(
            "{stage}: reference logit bits differ"
        )));
    }
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or_else(usage)?;
    let mut tokens = None;
    let mut accepted = None;
    let mut backend = Backend::from_env()?;
    let mut mode = KvMode::Int8;
    while let Some(option) = args.next() {
        let value = args.next().ok_or_else(usage)?;
        match option.as_str() {
            "--tokens" => {
                tokens = Some(
                    value
                        .split(',')
                        .map(|v| v.parse().map_err(|_| usage()))
                        .collect::<Result<Vec<u32>>>()?,
                )
            }
            "--accept" => accepted = Some(value.parse::<usize>().map_err(|_| usage())?),
            "--backend" => backend = value.parse()?,
            "--kv" => mode = yskn_ref::quality::parse_kv_mode(value.as_ref())?,
            _ => return Err(usage()),
        }
    }
    let tokens = tokens.filter(|t| !t.is_empty()).ok_or_else(usage)?;
    let accepted = accepted.unwrap_or(tokens.len().div_ceil(2));
    if accepted > tokens.len() {
        return Err(usage());
    }
    let probe = [tokens[0]];
    let mut reference = yskn_ref::Model::load_folder_with(&path, mode)?;
    reference.run_provisional(&tokens)?;
    let provisional = bits(reference.row_logits());
    reference.commit(accepted)?;
    let committed_digest = reference.state_digest();
    let committed = bits(reference.row_logits());
    reference.prefill(&probe)?;
    let continuation = bits(reference.row_logits());
    let final_digest = reference.state_digest();
    drop(reference);
    // SAFETY: this read-only verifier requires immutable backing weights until exit.
    let mut model =
        unsafe { Model::with_options(&path, Engine::from_env()?, backend, tokens.len(), mode)? };
    model.run_provisional(&tokens)?;
    equal(model.row_logits(), &provisional, "provisional")?;
    model.commit(accepted)?;
    equal(model.row_logits(), &committed, "commit")?;
    if model.state_digest() != committed_digest {
        return Err(Error::InvalidInput("committed state digest differs".into()));
    }
    model.prefill(&probe)?;
    equal(model.row_logits(), &continuation, "continuation")?;
    if model.state_digest() != final_digest {
        return Err(Error::InvalidInput(
            "continuation state digest differs".into(),
        ));
    }
    println!(
        "PASS backend={} kv={} width={} accepted={accepted} state_digest={final_digest:016x}",
        backend.name(),
        mode.name(),
        tokens.len()
    );
    Ok(())
}
