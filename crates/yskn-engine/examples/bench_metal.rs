//! `bench_metal MODEL.yskn [--m 1,2,4,8,16] [--passes 7] [--splits N] [--shares 0,0.1]`
//!
//! Verifies Metal against the CPU kernel on four real tensors, then times full
//! passes (every body matrix and the head) in three submission modes. With
//! `--shares`, it instead times engine-like passes (one call per group of
//! consecutive same-layer, same-width matrices, as `Model` groups them) for
//! each fixed CPU share of the rows (`weighted_sums_split`, 0 = GPU only).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bench_metal requires macOS and a Metal device");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::main() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::{hint::black_box, ops::Range, path::PathBuf, time::Instant};
    use yskn_engine::{Engine, Error, MappedModel, Matrix, Result};
    use yskn_engine::{Job, MetalEngine};
    use yskn_format::Role;

    fn usage() -> Error {
        Error::InvalidInput(
            "usage: bench_metal MODEL.yskn [--m 1,2,4,8,16] [--passes 7] [--splits N] [--shares 0,0.1]"
                .into(),
        )
    }

    fn number(value: Option<String>) -> Result<usize> {
        value.and_then(|v| v.parse().ok()).ok_or_else(usage)
    }

    /// Random signed bytes (all 256 values) and 24-bit non-negative integer scales.
    fn activations(k: usize, m: usize, seed: u64) -> (Vec<i8>, Vec<i32>) {
        let mut state = 0x87ab_1234_5678_9def_u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let a = (0..m * k).map(|_| next() as i8).collect();
        let s = (0..m * (k / 512))
            .map(|_| (next() % 16_777_217) as i32)
            .collect();
        (a, s)
    }

    fn stats(mut times: Vec<f64>) -> (f64, f64) {
        times.sort_by(f64::total_cmp);
        (times[times.len() / 2], times[0])
    }

    fn milliseconds(start: Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1e3
    }

    /// Consecutive matrices of one `layers.N.` prefix; anything else is its own group.
    fn layer_groups(matrices: &[Matrix]) -> Vec<Range<usize>> {
        let key = |m: &Matrix| {
            let name = &m.descriptor().name;
            name.strip_prefix("layers.")
                .and_then(|rest| rest.split('.').next())
                .map(str::to_owned)
        };
        let mut groups: Vec<Range<usize>> = Vec::new();
        for index in 0..matrices.len() {
            match groups.last_mut() {
                Some(last)
                    if key(&matrices[index]).is_some()
                        && key(&matrices[index]) == key(&matrices[last.start]) =>
                {
                    last.end = index + 1
                }
                _ => groups.push(index..index + 1),
            }
        }
        groups
    }

    fn verify(metal: &mut MetalEngine, model: &MappedModel) -> Result<()> {
        let tensors = &model.header().tensors;
        let mut selected = Vec::new();
        for needle in ["in_proj_qkv", "down_proj", "self_attn.q_proj"] {
            let d = tensors
                .iter()
                .find(|d| d.role == Role::Body && d.name.contains(needle))
                .ok_or_else(|| Error::InvalidModel(format!("missing tensor {needle}")))?;
            selected.push((d.name.clone(), d.physical_rows));
        }
        let head = tensors
            .iter()
            .find(|d| d.role == Role::Head)
            .ok_or_else(|| Error::InvalidModel("missing vocabulary head".into()))?;
        if head.physical_rows < 4096 {
            return Err(Error::InvalidModel(
                "verification head needs 4096 rows".into(),
            ));
        }
        selected.push((head.name.clone(), 4096));
        let mut cpu = Engine::from_env()?;
        for (index, (name, n)) in selected.into_iter().enumerate() {
            let matrix = model.matrix(&name)?;
            let k = matrix.descriptor().cols;
            let (a, s) = activations(k, 3, index as u64);
            let expected = cpu.weighted_sums_prefix(&matrix, &a, &s, n)?;
            let actual = metal.weighted_sums_prefix(&matrix, &a, &s, n)?;
            if let Some(i) = (0..expected.len()).find(|&i| actual[i] != expected[i]) {
                eprintln!(
                    "FAIL tensor={name} activation_row={} output_row={} metal={} cpu={}",
                    i / n,
                    i % n,
                    actual[i],
                    expected[i]
                );
                return Err(Error::InvalidModel("Metal/CPU mismatch".into()));
            }
            println!(
                "PASS tensor={name} N={n} K={k} bits={} M=3 metal==cpu({})",
                matrix.descriptor().bits,
                cpu.kernel().name()
            );
        }
        Ok(())
    }

    /// Engine-like passes: one `weighted_sums_split` call per group of consecutive
    /// matrices of one layer with the same input width.
    fn split_passes(
        metal: &mut MetalEngine,
        matrices: &[Matrix],
        batches: &[usize],
        shares: &[f64],
        passes: usize,
    ) -> Result<()> {
        let layers = layer_groups(matrices);
        let mut groups: Vec<Range<usize>> = Vec::new();
        for layer in layers {
            for index in layer.clone() {
                match groups.last_mut() {
                    Some(last)
                        if last.end == index
                            && last.start >= layer.start
                            && matrices[last.start].descriptor().cols
                                == matrices[index].descriptor().cols =>
                    {
                        last.end = index + 1
                    }
                    _ => groups.push(index..index + 1),
                }
            }
        }
        let mut cpu = Engine::from_env()?;
        println!(
            "split: groups={} cpu={} x{}",
            groups.len(),
            cpu.kernel().name(),
            cpu.threads()
        );
        println!("M share min_ms median_ms");
        for &m in batches {
            let inputs: Vec<_> = groups
                .iter()
                .enumerate()
                .map(|(i, g)| activations(matrices[g.start].descriptor().cols, m, 300 + i as u64))
                .collect();
            for &share in shares {
                metal.set_cpu_share(Some(share))?;
                let mut pass = || -> Result<f64> {
                    let start = Instant::now();
                    for (group, (a, s)) in groups.iter().zip(&inputs) {
                        let refs: Vec<&Matrix> = matrices[group.clone()].iter().collect();
                        black_box(metal.weighted_sums_split(&mut cpu, &refs, a, s)?);
                    }
                    Ok(milliseconds(start))
                };
                pass()?;
                let times = (0..passes).map(|_| pass()).collect::<Result<Vec<_>>>()?;
                let (median, min) = stats(times);
                println!("{m} {share} {min:.3} {median:.3}");
            }
        }
        metal.set_cpu_share(None)
    }

    pub fn main() -> Result<()> {
        let mut args = std::env::args().skip(1);
        let path = PathBuf::from(args.next().ok_or_else(usage)?);
        let mut batches = vec![1, 2, 4, 8, 16];
        let mut passes = 7;
        let mut splits = None;
        let mut shares: Option<Vec<f64>> = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--m" => {
                    batches = args
                        .next()
                        .ok_or_else(usage)?
                        .split(',')
                        .map(|v| v.parse::<usize>().map_err(|_| usage()))
                        .collect::<Result<_>>()?;
                    if batches.contains(&0) {
                        return Err(usage());
                    }
                }
                "--passes" => passes = number(args.next())?.max(1),
                "--splits" => splits = Some(number(args.next())?),
                "--shares" => {
                    shares = Some(
                        args.next()
                            .ok_or_else(usage)?
                            .split(',')
                            .map(|v| v.parse::<f64>().map_err(|_| usage()))
                            .collect::<Result<_>>()?,
                    )
                }
                _ => return Err(usage()),
            }
        }
        let mut metal = MetalEngine::new()?
            .ok_or_else(|| Error::InvalidInput("no Metal device in this process".into()))?;
        // YSKN_METAL_MAC and YSKN_METAL_CPU_SHARE apply here as in Model::load.
        metal.configure_from_env()?;
        if let Some(splits) = splits {
            metal.set_splits(splits)?;
        }
        println!(
            "device={} maxBufferLength={} threadgroup={}x{}={} threads mac={}",
            metal.device_name(),
            metal.max_buffer_length(),
            32,
            metal.splits(),
            metal.threads_per_threadgroup(),
            if metal.float_mac() {
                "float"
            } else {
                "integer"
            }
        );
        // SAFETY: this read-only benchmark requires the file to stay unchanged until exit.
        let model = unsafe { MappedModel::open(&path)? };
        println!(
            "file_bytes={} (one no-copy buffer)",
            model.header().file_bytes
        );
        verify(&mut metal, &model)?;

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
        let layers = layer_groups(&matrices);
        let bytes: u64 = matrices.iter().map(|m| m.descriptor().stream_bytes).sum();
        let weights: u64 = matrices
            .iter()
            .map(|m| (m.descriptor().physical_rows * m.descriptor().cols) as u64)
            .sum();
        println!(
            "matrices={} command_groups(c)={} code_bytes_per_pass={bytes} weights={weights} passes={passes} (one untimed warm-up per M)",
            matrices.len(),
            layers.len()
        );

        let mut round_trips = (0..200)
            .map(|_| metal.round_trip().map(|t| t.as_secs_f64() * 1e3))
            .collect::<Result<Vec<_>>>()?;
        round_trips.sort_by(f64::total_cmp);
        println!(
            "empty_dispatch_round_trip_ms median={:.4} min={:.4} (n=200)",
            round_trips[100], round_trips[0]
        );

        if let Some(shares) = shares {
            return split_passes(&mut metal, &matrices, &batches, &shares, passes);
        }
        println!(
            "M a_median_ms a_min_ms a_gpu_median_ms b_median_ms b_min_ms c_median_ms c_min_ms GB/s_a_median"
        );
        for m in batches {
            let inputs: Vec<_> = matrices
                .iter()
                .enumerate()
                .map(|(i, matrix)| activations(matrix.descriptor().cols, m, 100 + i as u64))
                .collect();
            // Buffers persist across passes; each timed pass still uploads every
            // activation, waits, and copies every M×N i64 result back to the CPU.
            let mut jobs: Vec<Job> = matrices
                .iter()
                .zip(&inputs)
                .map(|(matrix, (a, s))| metal.job(matrix, a, s, matrix.descriptor().physical_rows))
                .collect::<Result<_>>()?;
            let mut submit = |groups: &[Range<usize>]| -> Result<f64> {
                let mut busy = 0.0;
                for group in groups {
                    for (job, (a, s)) in jobs[group.clone()].iter_mut().zip(&inputs[group.clone()])
                    {
                        job.write_inputs(a, s)?;
                    }
                    busy += metal.run(&mut jobs[group.clone()])?.as_secs_f64() * 1e3;
                    for job in &jobs[group.clone()] {
                        black_box(job.output());
                    }
                }
                Ok(busy)
            };
            let all = 0..matrices.len();
            let whole = std::slice::from_ref(&all);
            let single: Vec<_> = (0..matrices.len()).map(|i| i..i + 1).collect();
            submit(whole)?;
            let (mut a, mut gpu, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for _ in 0..passes {
                let start = Instant::now();
                gpu.push(submit(whole)?);
                a.push(milliseconds(start));
            }
            for _ in 0..passes {
                let start = Instant::now();
                submit(&single)?;
                b.push(milliseconds(start));
            }
            for _ in 0..passes {
                let start = Instant::now();
                submit(&layers)?;
                c.push(milliseconds(start));
            }
            let ((a_median, a_min), (gpu_median, _)) = (stats(a), stats(gpu));
            let ((b_median, b_min), (c_median, c_min)) = (stats(b), stats(c));
            println!(
                "{m} {a_median:.3} {a_min:.3} {gpu_median:.3} {b_median:.3} {b_min:.3} {c_median:.3} {c_min:.3} {:.1}",
                bytes as f64 / a_median / 1e6
            );
        }
        Ok(())
    }
}
