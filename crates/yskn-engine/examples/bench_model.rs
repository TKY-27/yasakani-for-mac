//! `bench_model MODEL [--kv int8|int8-profile|kv5] [--backend cpu|metal] [--rows-per-pass 16] [--passes 7] [--context N]`
//!
//! Times whole-model passes after one untimed warm-up pass and
//! prints the minimum and median seconds of each `PassTiming` stage. With
//! --context, restore that prefix before each sample; prefix fill is untimed.
use std::time::Duration;
use yskn_engine::{Backend, Engine, Error, KvMode, Model, PassTiming, Result};
use yskn_ref::quality::parse_kv_mode;

fn usage() -> Error {
    Error::InvalidInput(
        "usage: bench_model MODEL [--kv int8|int8-profile|kv5] [--backend cpu|metal] [--rows-per-pass N] [--passes N] [--context N]".into(),
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or_else(usage)?;
    let (mut rows, mut passes, mut backend) = (16, 7, Backend::from_env()?);
    let mut mode = KvMode::Int8;
    let mut context = None;
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
            "--context" => context = Some(number()?),
            _ => return Err(usage()),
        }
    }
    // SAFETY: this read-only benchmark requires the weights to stay unchanged until exit.
    let mut model =
        unsafe { Model::with_options(&path, Engine::from_env()?, backend, rows, mode)? };
    // As Model::load: YSKN_METAL_MAC and YSKN_METAL_CPU_SHARE select the Metal modes.
    model.configure_metal_from_env()?;
    model.set_detailed_timing(true);
    println!(
        "backend={} device={} kv={mode:?} cpu_stages={}",
        model.backend().name(),
        model.device(),
        std::env::var("YSKN_CPU_STAGES").unwrap_or_else(|_| "auto".into())
    );
    let vocabulary = model.dimensions().vocabulary_size;
    let tokens = |count| -> Vec<u32> {
        (0..count)
            .map(|i| ((i * 7 + 11) % vocabulary) as u32)
            .collect()
    };
    let timings = if let Some(context) = context {
        model.prefill(&tokens(context))?;
        let prefix = model.snapshot()?;
        let input = tokens(rows);
        model.prefill(&input)?;
        let warm_logits = model.row_logits().to_vec();
        let warm_digest = model.state_digest();
        let mut timings = Vec::with_capacity(passes);
        for sample in 0..passes {
            model.restore(&prefix)?;
            model.prefill(&input)?;
            assert_eq!(model.state_digest(), warm_digest);
            assert_eq!(model.row_logits(), warm_logits);
            let timing = model
                .pass_timings()
                .last()
                .expect("one measured pass")
                .clone();
            assert_eq!((timing.position, timing.rows), (context, rows));
            println!(
                "sample={sample} context={context} total_s={:.9} kernel_s={:.9} recurrence_s={:.9} attention_s={:.9} kv_encode_s={:.9} rest_s={:.9}",
                timing.total.as_secs_f64(),
                timing.kernel.as_secs_f64(),
                timing.recurrence.as_secs_f64(),
                timing.attention.as_secs_f64(),
                timing.kv_encode.as_secs_f64(),
                timing.rest().as_secs_f64()
            );
            timings.push(timing);
        }
        timings
    } else {
        model.prefill(&tokens(rows))?;
        model.reset()?;
        model.prefill(&tokens(rows * passes))?;
        model.pass_timings().to_vec()
    };
    println!("rows_per_pass={rows} passes={passes} context={context:?} stage min_s median_s");
    type Stage = fn(&PassTiming) -> Duration;
    let stages: [(&str, Stage); 13] = [
        ("total", |t| t.total),
        ("kernel", |t| t.kernel),
        ("recurrence", |t| t.recurrence),
        ("recurrence_prepare", |t| t.recurrence_prepare),
        ("dense_gate_work", |t| t.dense_gate_work),
        ("attention", |t| t.attention),
        ("kv_encode", |t| t.kv_encode),
        ("rest", PassTiming::rest),
        ("normalization", |t| t.normalization),
        ("activation_encode", |t| t.activation_encode),
        ("projection_restore", |t| t.projection_restore),
        ("mlp_gate", |t| t.mlp_gate),
        ("residual", |t| t.residual),
    ];
    for (name, stage) in stages {
        let mut seconds: Vec<f64> = timings.iter().map(|t| stage(t).as_secs_f64()).collect();
        seconds.sort_by(f64::total_cmp);
        println!("{name} {:.6} {:.6}", seconds[0], seconds[seconds.len() / 2]);
    }
    Ok(())
}
