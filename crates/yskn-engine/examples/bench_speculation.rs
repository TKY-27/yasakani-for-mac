//! `bench_speculation MODEL [--backend cpu|metal] [--passes N]`
//! Run serially, using a synthetic model; reports provisional and partial/full commit costs.
use std::time::Instant;
use yskn_engine::{Backend, Engine, Error, Model, Result};

fn usage() -> Error {
    Error::InvalidInput("usage: bench_speculation MODEL [--backend cpu|metal] [--passes N]".into())
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or_else(usage)?;
    let mut backend = Backend::from_env()?;
    let mut passes = 7;
    while let Some(option) = args.next() {
        let value = args.next().ok_or_else(usage)?;
        match option.as_str() {
            "--backend" => backend = value.parse()?,
            "--passes" => passes = value.parse().ok().filter(|&n| n > 0).ok_or_else(usage)?,
            _ => return Err(usage()),
        }
    }
    // SAFETY: this read-only benchmark requires immutable backing weights until exit.
    let mut model = unsafe { Model::with_backend(path, Engine::from_env()?, backend, 8)? };
    println!(
        "backend={} device={} memory={:?}",
        backend.name(),
        model.device(),
        model.session_memory()
    );
    for width in [1, 2, 4, 8] {
        let mut provisional = Vec::new();
        let mut partial = Vec::new();
        let mut full = Vec::new();
        let mut copies = Vec::new();
        for sample in 0..=passes {
            model.reset()?;
            let tokens: Vec<_> = (0..width)
                .map(|i| ((i * 7 + 11) % model.dimensions().vocabulary_size) as u32)
                .collect();
            let start = Instant::now();
            model.run_provisional(&tokens)?;
            let pass_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            model.commit(width.div_ceil(2))?;
            let commit_ms = start.elapsed().as_secs_f64() * 1000.0;
            model.reset()?;
            model.run_provisional(&tokens)?;
            let start = Instant::now();
            model.commit(width)?;
            let full_ms = start.elapsed().as_secs_f64() * 1000.0;
            // Cost proxy for the rejected design: copy the full recurrent checkpoint
            // width times, retaining all copies simultaneously as row snapshots would.
            let start = Instant::now();
            let snapshots = (0..width)
                .map(|_| model.snapshot())
                .collect::<Result<Vec<_>>>()?;
            let copy_ms = start.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&snapshots);
            if sample != 0 {
                provisional.push(pass_ms);
                partial.push(commit_ms);
                full.push(full_ms);
                copies.push(copy_ms);
            }
        }
        println!(
            "width={width} samples={passes} provisional_ms={:.6} commit_rows={} commit_ms={:.6} full_commit_ms={:.6} snapshot_copy_proxy_ms={:.6}",
            median(&mut provisional),
            width.div_ceil(2),
            median(&mut partial),
            median(&mut full),
            median(&mut copies)
        );
    }
    Ok(())
}
