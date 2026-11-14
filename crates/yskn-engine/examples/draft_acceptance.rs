//! Teacher-forced, greedy acceptance diagnostics; the neural draft is scalar.
use std::error::Error as StdError;
use yskn_engine::{Backend, Engine, Error, KvMode, LogitRows, Model};
use yskn_ref::draft::Draft;
use yskn_ref::draft_head::DraftChoice;
use yskn_ref::output_head_sampler::greedy;
use yskn_ref::quality::{TokenWindows, parse_kv_mode};
use yskn_ref::speculation::{DraftProposer, NgramProposer, Proposer};

const USAGE: &str = "draft_acceptance MODEL_FOLDER WINDOWS.json [--windows A..B] [--positions N] [--k K] [--kv int8|int8-profile|kv5] [--backend cpu|metal]";

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[derive(Default)]
struct Acceptance {
    positions: usize,
    offered: usize,
    accepted: usize,
    depths: Vec<usize>,
}

impl Acceptance {
    fn record(&mut self, guesses: &[u32], choices: &[u32]) {
        let accepted = guesses
            .iter()
            .zip(choices)
            .take_while(|(guess, choice)| guess == choice)
            .count();
        self.positions += 1;
        self.offered += guesses.len();
        self.accepted += accepted;
        self.depths.resize(self.depths.len().max(choices.len()), 0);
        for count in &mut self.depths[..accepted] {
            *count += 1;
        }
    }

    fn print(&self, name: &str) {
        let rates: Vec<_> = self
            .depths
            .iter()
            .map(|&count| count as f64 / self.positions as f64)
            .collect();
        println!(
            "{name}: positions={} guesses_offered={} acceptance_per_depth={rates:?} mean_accepted_prefix={:.6}",
            self.positions,
            self.offered,
            self.accepted as f64 / self.positions as f64,
        );
    }
}

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = std::env::args().skip(1);
    let folder = args.next().ok_or_else(|| invalid(USAGE))?;
    if folder == "--help" || folder == "-h" {
        println!("Usage: {USAGE}");
        println!(
            "Defaults: --windows 0..1 --positions 2 --k 3; positions per window, starting at its first scored input row."
        );
        println!(
            "Scalar neural drafting can take about 90 seconds per proposal on the real model. Run only a handful of positions."
        );
        println!(
            "Depth d measures P(leading accepted prefix >= d); missing guesses count as unaccepted. Both uses ngram-first fallback."
        );
        return Ok(());
    }
    let windows_path = args.next().ok_or_else(|| invalid(USAGE))?;
    let (mut range, mut positions, mut k) = (0..1, 2_usize, 3_usize);
    let mut mode = KvMode::Int8;
    let mut backend = None;
    while let Some(option) = args.next() {
        let value = args.next().ok_or_else(|| invalid(USAGE))?;
        match option.as_str() {
            "--windows" => {
                let (start, end) = value
                    .split_once("..")
                    .ok_or_else(|| invalid("expected half-open A..B window range"))?;
                range = start.parse()?..end.parse()?;
            }
            "--positions" => positions = value.parse()?,
            "--k" => k = value.parse()?,
            "--kv" => mode = parse_kv_mode(value.as_ref())?,
            "--backend" => backend = Some(value.parse()?),
            _ => return Err(invalid(USAGE).into()),
        }
    }
    let windows = TokenWindows::load(windows_path)?;
    if range.start >= range.end || range.end > windows.windows.len() || positions == 0 {
        return Err(invalid("select a nonempty window range and positive position count").into());
    }
    if !(1..yskn_ref::draft::BLOCK_ROWS).contains(&k) {
        return Err(invalid("K must be 1..=7").into());
    }
    let last_position = windows
        .first_scored_position
        .checked_add(positions - 1)
        .ok_or_else(|| invalid("position range overflow"))?;
    if last_position > windows.last_scored_position || last_position + k > windows.n_ctx {
        return Err(invalid("requested positions and K exceed the scored window context").into());
    }
    let backend = backend.map_or_else(Backend::from_env, Ok)?;
    let mut neural = DraftProposer::new(Draft::load(&folder)?, k, DraftChoice::Greedy)?;
    let mut ngram = NgramProposer::new(3, k)?;
    // SAFETY: this read-only diagnostic requires stable weights until exit.
    let mut model =
        unsafe { Model::with_options(&folder, Engine::from_env()?, backend, 16, mode)? };
    model.set_capture(neural.capture_options())?;
    if windows.n_ctx > model.context_capacity() {
        return Err(invalid("window exceeds model context capacity").into());
    }
    for window in &windows.windows[range.clone()] {
        if window
            .ids
            .iter()
            .any(|&id| id as usize >= model.dimensions().vocabulary_size)
        {
            return Err(invalid("window token exceeds model vocabulary").into());
        }
    }
    eprintln!(
        "backend={} device={} kv={} positions_per_window={positions} k={k}; scalar neural draft may take about 90 seconds per proposal",
        model.backend().name(),
        model.device(),
        mode.name(),
    );
    let (mut neural_stats, mut ngram_stats, mut both_stats) = (
        Acceptance::default(),
        Acceptance::default(),
        Acceptance::default(),
    );
    for window_index in range {
        model.reset()?;
        neural.reset();
        let ids = &windows.windows[window_index].ids;
        model.prefill_with_logits(&ids[..windows.first_scored_position], LogitRows::None)?;
        neural.synchronize(&model)?;
        for position in windows.first_scored_position..=last_position {
            let guesses = neural.propose_with(&model, &ids[..position], ids[position], k)?;
            let copied = ngram.propose(&ids[..position], ids[position], k);
            // Reuse this position's expensive neural block for the fixed ngram-first policy.
            let combined = if copied.is_empty() { &guesses } else { &copied };
            model.run_provisional(&ids[position..position + k])?;
            let choices: Vec<_> = model
                .row_logits()
                .iter()
                .map(|row| greedy(row))
                .collect::<yskn_ref::Result<_>>()?;
            neural_stats.record(&guesses, &choices);
            ngram_stats.record(&copied, &choices);
            both_stats.record(combined, &choices);
            // Only the teacher-forced frontier becomes context, never probe suffix rows.
            model.commit(1)?;
            neural.synchronize(&model)?;
            eprintln!("window={window_index} position={position} complete");
        }
    }
    neural_stats.print("neural");
    ngram_stats.print("ngram");
    both_stats.print("both (ngram-first)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Acceptance;

    #[test]
    fn acceptance_counts_only_the_leading_prefix_and_includes_abstentions() {
        let mut stats = Acceptance::default();
        stats.record(&[4, 9, 6], &[4, 5, 6]);
        stats.record(&[4, 5], &[4, 5, 6]);
        stats.record(&[], &[4, 5, 6]);
        assert_eq!(stats.positions, 3);
        assert_eq!(stats.offered, 5);
        assert_eq!(stats.accepted, 3);
        assert_eq!(stats.depths, [2, 1, 0]);
    }
}
