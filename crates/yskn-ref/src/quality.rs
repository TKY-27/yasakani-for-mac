//! Shared chapter-06 window definition, ordered loss accumulation and reporting.
use crate::{Error, Result, output_head_sampler::negative_log_likelihood, portable_math::exp_d};
use serde::{Deserialize, Serialize};
use std::{fs, ops::Range, path::Path, time::Instant};

#[derive(Deserialize)]
pub struct TokenWindows {
    pub n_ctx: usize,
    pub first_scored_position: usize,
    pub last_scored_position: usize,
    pub windows: Vec<TokenWindow>,
}
#[derive(Deserialize)]
pub struct TokenWindow {
    pub ids: Vec<u32>,
}
impl TokenWindows {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let windows: Self = serde_json::from_slice(&fs::read(path)?)?;
        windows.validate()?;
        Ok(windows)
    }
    fn validate(&self) -> Result<()> {
        if self.n_ctx != 2048
            || self.first_scored_position != 1024
            || self.last_scored_position != 2046
        {
            return Err(Error::InvalidInput(
                "windows must specify n_ctx=2048 and scored positions 1024..=2046".into(),
            ));
        }
        if self.windows.is_empty() || self.windows.iter().any(|w| w.ids.len() != self.n_ctx) {
            return Err(Error::InvalidInput(
                "expected at least one complete 2048-token window".into(),
            ));
        }
        Ok(())
    }
}

pub trait ScoringModel {
    fn vocabulary_size(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<()>;
    fn row_logits(&self) -> &[Vec<f32>];
}
impl ScoringModel for crate::Model {
    fn vocabulary_size(&self) -> usize {
        self.dimensions().vocabulary_size
    }
    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill(tokens)
    }
    fn row_logits(&self) -> &[Vec<f32>] {
        self.row_logits()
    }
}
#[derive(Serialize)]
pub struct WindowScore {
    pub window: usize,
    pub scored_positions: usize,
    pub negative_log_likelihood_sum: f64,
    pub elapsed_seconds: f64,
}
#[derive(Serialize)]
pub struct Score {
    pub windows: Vec<WindowScore>,
    pub scored_positions: usize,
    pub negative_log_likelihood_sum: f64,
    pub mean_negative_log_likelihood: f64,
    pub perplexity: f64,
    pub elapsed_seconds: f64,
}
impl Score {
    pub fn print(&self) {
        println!("scored_positions: {}", self.scored_positions);
        println!(
            "mean_negative_log_likelihood: {:.12}",
            self.mean_negative_log_likelihood
        );
        println!("perplexity: {:.12}", self.perplexity);
    }
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

/// `after_pass` observes timing without participating in any numerical operation.
pub fn score<M: ScoringModel>(
    model: &mut M,
    windows: &TokenWindows,
    range: Range<usize>,
    rows_per_pass: usize,
    mut after_pass: impl FnMut(&M),
) -> Result<Score> {
    windows.validate()?;
    if rows_per_pass == 0 || range.start >= range.end || range.end > windows.windows.len() {
        return Err(Error::InvalidInput(
            "invalid pass width or window range".into(),
        ));
    }
    if windows.windows[range.clone()]
        .iter()
        .flat_map(|w| &w.ids)
        .any(|&t| t as usize >= model.vocabulary_size())
    {
        return Err(Error::InvalidInput(
            "window token is outside the model vocabulary".into(),
        ));
    }
    let start = Instant::now();
    let mut total_loss = 0.0_f64;
    let mut scored_positions = 0;
    let mut scores = Vec::new();
    for index in range {
        let start = Instant::now();
        let window = &windows.windows[index];
        model.reset()?;
        let mut window_loss = 0.0_f64;
        let mut count = 0;
        for first in (0..=windows.last_scored_position).step_by(rows_per_pass) {
            let end = first
                .saturating_add(rows_per_pass)
                .min(windows.last_scored_position + 1);
            model.prefill(&window.ids[first..end])?;
            after_pass(model);
            let rows = model.row_logits();
            if rows.len() != end - first {
                return Err(Error::InvalidModel(
                    "consumed rows produced the wrong number of logits".into(),
                ));
            }
            for (offset, logits) in rows.iter().enumerate() {
                let position = first + offset;
                if position >= windows.first_scored_position {
                    let loss = negative_log_likelihood(logits, window.ids[position + 1])?;
                    window_loss += loss;
                    // Do not replace this ordered fold with a sum of window totals.
                    total_loss += loss;
                    scored_positions += 1;
                    count += 1;
                }
            }
        }
        let elapsed_seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "window={index} scored_positions={count} nll_sum={window_loss:.16e} elapsed_seconds={elapsed_seconds:.3}"
        );
        scores.push(WindowScore {
            window: index,
            scored_positions: count,
            negative_log_likelihood_sum: window_loss,
            elapsed_seconds,
        });
    }
    let mean_negative_log_likelihood = total_loss / scored_positions as f64;
    let perplexity = exp_d(mean_negative_log_likelihood);
    if !mean_negative_log_likelihood.is_finite() || !perplexity.is_finite() {
        return Err(Error::NonFinite("aggregate quality metrics"));
    }
    Ok(Score {
        windows: scores,
        scored_positions,
        negative_log_likelihood_sum: total_loss,
        mean_negative_log_likelihood,
        perplexity,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}
