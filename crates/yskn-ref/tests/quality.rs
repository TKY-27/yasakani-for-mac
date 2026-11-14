use yskn_ref::{
    Result,
    output_head_sampler::negative_log_likelihood,
    quality::{ScoringModel, TokenWindow, TokenWindows, score},
};

struct Sequence {
    position: usize,
    resets: usize,
    rows: Vec<Vec<f32>>,
}
impl ScoringModel for Sequence {
    fn vocabulary_size(&self) -> usize {
        2
    }
    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        self.resets += 1;
        Ok(())
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.rows = (self.position..self.position + tokens.len())
            .map(logits)
            .collect();
        self.position += tokens.len();
        Ok(())
    }
    fn row_logits(&self) -> &[Vec<f32>] {
        &self.rows
    }
}
fn logits(position: usize) -> Vec<f32> {
    vec![(position % 71) as f32 / 8.0, -0.25]
}
#[test]
fn scoring_uses_next_tokens_resets_windows_and_folds_losses_in_position_order() {
    let windows = TokenWindows {
        n_ctx: 2048,
        first_scored_position: 1024,
        last_scored_position: 2046,
        windows: (0..3)
            .map(|w| TokenWindow {
                ids: (0..2048).map(|p| ((p + w) % 2) as u32).collect(),
            })
            .collect(),
    };
    let mut total = 0.0_f64;
    let mut per_window = Vec::new();
    for window in &windows.windows[1..3] {
        let mut sum = 0.0_f64;
        for position in 1024..=2046 {
            let loss =
                negative_log_likelihood(&logits(position), window.ids[position + 1]).unwrap();
            total += loss;
            sum += loss;
        }
        per_window.push(sum);
    }
    for rows in [1, 5, 16, 40, 2048] {
        let mut model = Sequence {
            position: 91,
            resets: 0,
            rows: Vec::new(),
        };
        let result = score(&mut model, &windows, 1..3, rows, |_| {}).unwrap();
        assert_eq!(model.resets, 2);
        assert_eq!(model.position, 2047);
        assert_eq!(result.scored_positions, 2046);
        assert_eq!(
            result.negative_log_likelihood_sum.to_bits(),
            total.to_bits()
        );
        for (index, (actual, expected)) in result.windows.iter().zip(&per_window).enumerate() {
            assert_eq!(actual.window, index + 1);
            assert_eq!(actual.scored_positions, 1023);
            assert_eq!(
                actual.negative_log_likelihood_sum.to_bits(),
                expected.to_bits()
            );
        }
        assert_eq!(
            result.perplexity.to_bits(),
            yskn_ref::portable_math::exp_d(total / 2046.0).to_bits()
        );
    }
    let mut model = Sequence {
        position: 17,
        resets: 0,
        rows: Vec::new(),
    };
    assert!(score(&mut model, &windows, 0..4, 16, |_| {}).is_err());
    assert!(score(&mut model, &windows, 0..1, 0, |_| {}).is_err());
    assert_eq!(model.position, 17);
    assert_eq!(model.resets, 0);
}

#[test]
fn shared_quality_mode_parser_and_loader_receive_each_explicit_mode() {
    use yskn_ref::{
        KvMode,
        quality::{parse_kv_mode, score_with_kv},
    };
    let windows = TokenWindows {
        n_ctx: 2048,
        first_scored_position: 1024,
        last_scored_position: 2046,
        windows: vec![TokenWindow { ids: vec![0; 2048] }],
    };
    for (name, expected) in [
        ("int8", KvMode::Int8),
        ("int8-profile", KvMode::Int8Profile),
        ("kv5", KvMode::Kv5),
    ] {
        let mode = parse_kv_mode(std::ffi::OsStr::new(name)).unwrap();
        assert_eq!(mode, expected);
        assert_eq!(mode.name(), name);
        let mut observed = None;
        let result = score_with_kv(
            mode,
            |selected| {
                observed = Some(selected);
                Ok(Sequence {
                    position: 0,
                    resets: 0,
                    rows: Vec::new(),
                })
            },
            &windows,
            0..1,
            16,
            |_| {},
        )
        .unwrap();
        assert_eq!(observed, Some(expected));
        assert_eq!(result.scored_positions, 1023);
    }
    assert!(parse_kv_mode(std::ffi::OsStr::new("int4")).is_err());
}
