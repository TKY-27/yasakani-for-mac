mod common;
use std::{hint::black_box, path::PathBuf, time::Instant};
use yskn_engine::{Engine, Error, MappedModel, Matrix, Result};
use yskn_format::Role;

fn pass(engine: &mut Engine, matrices: &[Matrix], inputs: &[(Vec<i8>, Vec<i32>)]) -> Result<f64> {
    let start = Instant::now();
    for (matrix, (a, s)) in matrices.iter().zip(inputs) {
        black_box(engine.weighted_sums(matrix, a, s)?);
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(common::argument_value(&mut args)?);
    let mut kernel = None;
    let mut threads = None;
    let mut batches = vec![1, 2, 4, 8, 16];
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--kernel" => kernel = Some(common::argument_value(&mut args)?.parse()?),
            "--threads" => {
                threads = Some(
                    common::argument_value(&mut args)?
                        .parse::<usize>()
                        .map_err(|_| common::missing_path())?,
                )
            }
            "--m" => {
                batches = common::argument_value(&mut args)?
                    .split(',')
                    .map(|value| value.parse::<usize>().map_err(|_| common::missing_path()))
                    .collect::<Result<Vec<_>>>()?;
                if batches.iter().any(|&m| m == 0 || m > 16) {
                    return Err(common::missing_path());
                }
            }
            _ => return Err(common::missing_path()),
        }
    }
    let mut engine = match kernel {
        Some(kernel) => Engine::new(kernel, threads.unwrap_or_else(Engine::default_threads))?,
        None => {
            let default = Engine::from_env()?;
            match threads {
                Some(threads) => {
                    let kernel = default.kernel();
                    drop(default);
                    Engine::new(kernel, threads)?
                }
                None => default,
            }
        }
    };
    // SAFETY: this read-only benchmark requires the input file to remain unchanged until exit.
    let model = unsafe { MappedModel::open(&path)? };
    let matrices: Vec<_> = model
        .header()
        .tensors
        .iter()
        .filter(|d| matches!(d.role, Role::Body | Role::Head))
        .map(|d| model.matrix(&d.name))
        .collect::<Result<_>>()?;
    if matrices.is_empty() {
        return Err(Error::InvalidModel("no body/head matrices".into()));
    }
    let bytes: u64 = matrices.iter().map(|m| m.descriptor().stream_bytes).sum();
    println!(
        "kernel={} threads={} matrices={} passes=7 (one untimed warm-up per M)",
        engine.kernel().name(),
        engine.threads(),
        matrices.len()
    );
    println!("M median_ms min_ms code_bytes GB/s_median GB/s_min");
    for m in batches {
        let inputs: Vec<_> = matrices
            .iter()
            .map(|matrix| common::activations(matrix.descriptor().cols, m))
            .collect();
        pass(&mut engine, &matrices, &inputs)?;
        let mut times = (0..7)
            .map(|_| pass(&mut engine, &matrices, &inputs))
            .collect::<Result<Vec<_>>>()?;
        times.sort_by(f64::total_cmp);
        println!(
            "{m} {:.3} {:.3} {bytes} {:.3} {:.3}",
            times[3],
            times[0],
            bytes as f64 / times[3] / 1e6,
            bytes as f64 / times[0] / 1e6
        );
    }
    Ok(())
}
