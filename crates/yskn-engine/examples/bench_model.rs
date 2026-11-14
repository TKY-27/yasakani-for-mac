//! `bench_model MODEL [--kv int8|int8-profile|kv5] [--backend cpu|metal] [--rows-per-pass 16] [--passes 7]`
//!
//! Times whole-model passes from position 0 after one untimed warm-up pass and
//! prints the minimum and median seconds of each `PassTiming` stage.
use std::time::Duration;
use yskn_engine::{Backend, Engine, Error, KvMode, Model, PassTiming, Result};
use yskn_ref::quality::parse_kv_mode;

fn usage() -> Error {
    Error::InvalidInput(
        "usage: bench_model MODEL [--kv int8|int8-profile|kv5] [--backend cpu|metal] [--rows-per-pass N] [--passes N]".into(),
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or_else(usage)?;
    let (mut rows, mut passes, mut backend) = (16, 7, Backend::from_env()?);
    let mut mode = KvMode::Int8;
    while let Some(option) = args.next() {
        let value = args.next().ok_or_else(usage)?;
        let number = || {
            value
                .parse::<usize>()
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(usage)
        };
        match option.as_str() {
            "--kv" => mode = parse_kv_mode(value.as_ref())?,
            "--backend" => backend = value.parse()?,
            "--rows-per-pass" => rows = number()?,
            "--passes" => passes = number()?,
            _ => return Err(usage()),
        }
    }
    // SAFETY: this read-only benchmark requires the weights to stay unchanged until exit.
    let mut model =
        unsafe { Model::with_options(&path, Engine::from_env()?, backend, rows, mode)? };
    println!(
        "backend={} device={} kv={mode:?}",
        model.backend().name(),
        model.device()
    );
    let vocabulary = model.dimensions().vocabulary_size;
    let tokens: Vec<u32> = (0..rows * passes)
        .map(|i| ((i * 7 + 11) % vocabulary) as u32)
        .collect();
    model.prefill(&tokens[..rows])?;
    model.reset()?;
    model.prefill(&tokens)?;
    let timings = model.pass_timings();
    println!("rows_per_pass={rows} passes={passes} stage min_s median_s");
    type Stage = fn(&PassTiming) -> Duration;
    let stages: [(&str, Stage); 6] = [
        ("total", |t| t.total),
        ("kernel", |t| t.kernel),
        ("recurrence", |t| t.recurrence),
        ("attention", |t| t.attention),
        ("kv_encode", |t| t.kv_encode),
        ("rest", PassTiming::rest),
    ];
    for (name, stage) in stages {
        let mut seconds: Vec<f64> = timings.iter().map(|t| stage(t).as_secs_f64()).collect();
        seconds.sort_by(f64::total_cmp);
        println!("{name} {:.6} {:.6}", seconds[0], seconds[seconds.len() / 2]);
    }
    Ok(())
}
