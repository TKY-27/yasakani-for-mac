//! Fast-draft timing on a model folder: proposal with a full ring, by stage, and
//! context pushes per row for batches of 1, 16 and 256. Opens only the draft files.
use std::{
    error::Error as StdError,
    time::{Duration, Instant},
};
use yskn_engine::{Backend, Engine, FastDraft, Kernel};
use yskn_ref::draft::DraftOptions;
use yskn_ref::draft_head::DraftChoice;

const USAGE: &str = "bench_draft MODEL_FOLDER [--backend cpu|metal] [--kernel portable|neon|dotprod] [--threads N] [--samples N]";

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

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// (median, minimum) in milliseconds.
fn summary(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    (samples[samples.len() / 2], samples[0])
}

/// Peak resident set size of this process in bytes (Darwin reports bytes).
fn peak_rss() -> Option<u64> {
    #[repr(C)]
    struct Usage([i64; 18]);
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Usage) -> i32;
    }
    let mut usage = Usage([0; 18]);
    // SAFETY: RUSAGE_SELF (0) and a buffer the size of Darwin's struct rusage
    // (two timevals then fourteen longs); ru_maxrss is the fifth 8-byte word.
    let status = unsafe { getrusage(0, &mut usage) };
    (status == 0 && cfg!(target_os = "macos")).then_some(usage.0[4] as u64)
}

fn current_rss() -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(out.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = std::env::args().skip(1);
    let folder = args.next().ok_or(USAGE)?;
    let mut backend = Backend::from_env()?;
    let mut kernel = Kernel::fastest();
    let mut threads = Engine::default_threads();
    let mut samples = 11_usize;
    while let Some(option) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match option.as_str() {
            "--backend" => backend = value.parse()?,
            "--kernel" => kernel = value.parse()?,
            "--threads" => threads = value.parse()?,
            "--samples" => samples = value.parse::<usize>()?.max(9),
            _ => return Err(USAGE.into()),
        }
    }
    let date = std::process::Command::new("date").output()?;
    println!("taken: {}", String::from_utf8_lossy(&date.stdout).trim());
    let started = Instant::now();
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
    println!(
        "loaded in {:.1} ms: backend={} device={} kernel={} threads={threads} neon_float_stages={}",
        ms(started.elapsed()),
        draft.backend().name(),
        draft.device(),
        kernel.name(),
        draft.vectorized()
    );
    println!(
        "draft: hidden={} ffn={} layers={} heads={}/{} window={} vocab={} shortlist={}",
        d.hidden,
        d.feed_forward,
        d.layers,
        d.query_heads,
        d.kv_heads,
        d.window,
        d.vocabulary,
        draft.options().shortlist
    );
    let row = |r: usize| synthetic(1000 + r as u64, d.features, 4.0);
    // Fill the ring with batches of 256.
    let fill = Instant::now();
    let mut position = 0;
    while position < d.window {
        let n = 256.min(d.window - position);
        let rows: Vec<Vec<f32>> = (position..position + n).map(row).collect();
        let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        draft.push_rows(position, &slices)?;
        position += n;
    }
    println!(
        "ring filled: {} rows in {:.1} ms ({:.3} ms/row)",
        d.window,
        ms(fill.elapsed()),
        ms(fill.elapsed()) / d.window as f64
    );
    let embedding = |token: u32| Ok(synthetic(token as u64, d.hidden, 0.0625));
    let mut totals = Vec::new();
    let mut stages: Vec<[f64; 10]> = Vec::new();
    for sample in 0..samples + 2 {
        let frontier = (sample as u32 * 7919) % d.vocabulary as u32;
        let start = Instant::now();
        let proposal = draft.propose(frontier, embedding, DraftChoice::Greedy)?;
        let wall = ms(start.elapsed());
        if sample < 2 {
            println!(
                "warm-up proposal {sample}: {wall:.2} ms tokens={:?}",
                proposal.tokens
            );
            continue;
        }
        let t = draft.timing();
        totals.push(wall);
        stages.push([
            ms(t.projections),
            ms(t.projection_kernel),
            ms(t.attention),
            ms(t.block_other),
            ms(t.head_projection),
            ms(t.head_scores),
            ms(t.head_shortlist),
            ms(t.head_rescore),
            ms(t.head_select),
            ms(t.rest()),
        ]);
    }
    let (median, minimum) = summary(totals.clone());
    println!(
        "proposal (ring full, {samples} samples): median {median:.2} ms, minimum {minimum:.2} ms"
    );
    let names = [
        "block projections (45 matrices, quantize+kernel+restore)",
        "  of which compressed kernel calls",
        "block attention",
        "block other (norm, rope, filters, glu, adds)",
        "head rank-512 projection",
        "head approximate scores",
        "head shortlist",
        "head rescoring + top16",
        "head correction + choice",
        "rest (embeddings, validation)",
    ];
    for (i, name) in names.iter().enumerate() {
        let (median, minimum) = summary(stages.iter().map(|s| s[i]).collect());
        println!("  {name}: median {median:.2} ms, minimum {minimum:.2} ms");
    }
    for batch in [1_usize, 16, 256] {
        let mut per_row = Vec::new();
        for _ in 0..samples.max(9) {
            let rows: Vec<Vec<f32>> = (position..position + batch).map(row).collect();
            let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
            let start = Instant::now();
            draft.push_rows(position, &slices)?;
            per_row.push(ms(start.elapsed()) / batch as f64);
            position += batch;
        }
        let (median, minimum) = summary(per_row);
        println!(
            "push batch {batch:>3}: median {median:.3} ms/row, minimum {minimum:.3} ms/row ({} samples)",
            samples.max(9)
        );
    }
    println!(
        "resident: current {} bytes, peak {} bytes",
        current_rss().map_or("unknown".into(), |b| b.to_string()),
        peak_rss().map_or("unknown".into(), |b| b.to_string())
    );
    Ok(())
}
