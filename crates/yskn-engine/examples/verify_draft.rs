//! Real-file check of the fast draft: synthetic context features of the draft's
//! shape, then ring and proposal bits against the scalar reference and across
//! backends, kernels and thread counts. Opens only the draft files.
//! Exits non-zero on any difference.
use std::{error::Error as StdError, time::Instant};
use yskn_engine::{Backend, Engine, FastDraft, Kernel};
use yskn_ref::draft::{Draft, DraftOptions, DraftProposal, DraftSnapshot};
use yskn_ref::draft_head::DraftChoice;

const USAGE: &str =
    "verify_draft MODEL_FOLDER [--positions N] [--reference-rows R] [--skip-reference]";

/// Deterministic BF16-representable values in [-scale, scale).
fn synthetic(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            yskn_ref::portable_math::bfloat((2.0 * unit - 1.0) * scale)
        })
        .collect()
}

fn same_proposal(a: &DraftProposal, b: &DraftProposal) -> bool {
    let flat = |c: &Vec<Vec<yskn_ref::draft_head::Candidate>>| -> Vec<(u32, u32)> {
        c.iter()
            .flatten()
            .map(|c| (c.token, c.score.to_bits()))
            .collect()
    };
    a.tokens == b.tokens
        && flat(&a.candidates) == flat(&b.candidates)
        && flat(&a.corrected) == flat(&b.corrected)
}

fn check(ok: bool, what: &str, failures: &mut Vec<String>) {
    println!("{} {what}", if ok { "EQUAL" } else { "DIFFERENT" });
    if !ok {
        failures.push(what.to_owned());
    }
}

struct Run {
    after_reference_rows: DraftSnapshot,
    full: DraftSnapshot,
    greedy: DraftProposal,
    sampled: DraftProposal,
}

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = std::env::args().skip(1);
    let folder = args.next().ok_or(USAGE)?;
    let (mut positions, mut reference_rows, mut reference) = (300_usize, 3_usize, true);
    while let Some(option) = args.next() {
        match option.as_str() {
            "--skip-reference" => reference = false,
            "--positions" => positions = args.next().ok_or(USAGE)?.parse()?,
            "--reference-rows" => reference_rows = args.next().ok_or(USAGE)?.parse()?,
            _ => return Err(USAGE.into()),
        }
    }
    let reference_rows = reference_rows.min(positions);
    let mut configurations = Vec::new();
    #[cfg(target_os = "macos")]
    if yskn_engine::MetalEngine::device_exists() {
        configurations.push((Backend::Metal, Kernel::fastest(), Engine::default_threads()));
    }
    if configurations.is_empty() {
        println!("no Metal device: CPU configurations only");
    }
    configurations.extend([
        (Backend::Cpu, Kernel::fastest(), Engine::default_threads()),
        (Backend::Cpu, Kernel::fastest(), 1),
        (Backend::Cpu, Kernel::Portable, 4),
    ]);
    let mut failures = Vec::new();
    let mut first: Option<(Run, u32, usize)> = None;
    for (index, &(backend, kernel, threads)) in configurations.iter().enumerate() {
        let name = format!("{} {} x{threads}", backend.name(), kernel.name());
        let start = Instant::now();
        // SAFETY: the caller keeps the read-only draft files unchanged until exit.
        let mut draft = unsafe {
            FastDraft::with_options(
                &folder,
                Engine::new(kernel, threads)?,
                backend,
                DraftOptions::default(),
            )?
        };
        let d = draft.dimensions().clone();
        let features = |r: usize| synthetic(1000 + r as u64, d.features, 4.0);
        // Different chunkings per configuration; rows 0..reference_rows go singly first.
        let chunk = [7, 256, 16, 1][index % 4];
        let mut position = 0;
        let mut after_reference_rows = None;
        while position < positions {
            let n = if position < reference_rows {
                1
            } else {
                chunk.min(positions - position)
            };
            let rows: Vec<Vec<f32>> = (position..position + n).map(features).collect();
            let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
            draft.push_rows(position, &slices)?;
            position += n;
            if position == reference_rows {
                after_reference_rows = Some(draft.snapshot());
            }
        }
        let after_reference_rows = after_reference_rows.unwrap_or_else(|| draft.snapshot());
        let pushed = start.elapsed();
        let frontier = (d.vocabulary as u32 / 3) | 1;
        let embedding = |token: u32| Ok(synthetic(token as u64, d.hidden, 0.0625));
        let greedy = draft.propose(frontier, embedding, DraftChoice::Greedy)?;
        let sampled = draft.propose(frontier, embedding, DraftChoice::sampled(17))?;
        println!(
            "{name}: device={} loaded+pushed {positions} rows in {:.1} s, greedy tokens {:?}, sampled {:?}",
            draft.device(),
            pushed.as_secs_f64(),
            greedy.tokens,
            sampled.tokens
        );
        let run = Run {
            after_reference_rows,
            full: draft.snapshot(),
            greedy,
            sampled,
        };
        match &first {
            None => first = Some((run, frontier, d.hidden)),
            Some((expected, ..)) => {
                let base = format!("{name} vs {}", {
                    let (b, k, t) = configurations[0];
                    format!("{} {} x{t}", b.name(), k.name())
                });
                check(
                    run.full == expected.full,
                    &format!("{base}: ring after {positions} rows"),
                    &mut failures,
                );
                check(
                    same_proposal(&run.greedy, &expected.greedy),
                    &format!("{base}: greedy proposal, top16 and corrected scores"),
                    &mut failures,
                );
                check(
                    same_proposal(&run.sampled, &expected.sampled),
                    &format!("{base}: sampled proposal, top16 and corrected scores"),
                    &mut failures,
                );
            }
        }
    }
    let (expected, frontier, hidden) = first.ok_or("no configuration ran")?;
    if reference {
        // Only one draft is alive at a time, bounding resident memory.
        let start = Instant::now();
        let mut scalar = Draft::load(&folder)?;
        let features = |r: usize| synthetic(1000 + r as u64, scalar.dimensions().features, 4.0);
        let rows: Vec<f32> = (0..reference_rows).flat_map(features).collect();
        scalar.push_context_rows(0, &rows)?;
        println!(
            "scalar reference: loaded and pushed {reference_rows} rows in {:.1} s",
            start.elapsed().as_secs_f64()
        );
        check(
            scalar.snapshot() == expected.after_reference_rows,
            &format!("fast vs scalar reference: ring after {reference_rows} rows"),
            &mut failures,
        );
        scalar.restore(&expected.full)?;
        let start = Instant::now();
        let embedding = |token: u32| Ok(synthetic(token as u64, hidden, 0.0625));
        let proposal = scalar.propose(frontier, embedding, DraftChoice::Greedy)?;
        println!(
            "scalar reference proposal at position {positions}: {:.1} s, tokens {:?}",
            start.elapsed().as_secs_f64(),
            proposal.tokens
        );
        check(
            same_proposal(&proposal, &expected.greedy),
            &format!(
                "fast vs scalar reference: greedy proposal at position {positions}, top16 and corrected scores"
            ),
            &mut failures,
        );
    } else {
        println!("scalar reference skipped (--skip-reference)");
    }
    if failures.is_empty() {
        println!("PASS: all comparisons bit-identical");
        Ok(())
    } else {
        Err(format!("{} comparison(s) differ: {failures:?}", failures.len()).into())
    }
}
