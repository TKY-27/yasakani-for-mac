//! [Chapter 06: output head and sampler](../../../docs/spec/06_OUTPUT_HEAD_SAMPLER.md).

use crate::compressed_linear::CompressedLinear;
use crate::portable_math::{exp_d, log_d};
use crate::residual_norm_mlp::NormalizedInput;
use crate::{Error, Result, require_finite};

pub fn output_head(head: &CompressedLinear, input: &NormalizedInput) -> Result<Vec<f32>> {
    // The head constructor disables output rotation. Scale restoration keeps
    // inverse RMS between the common integer-sum quantum and the weight scale.
    head.project(&input.prepared, Some(input.inverse_rms))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingVersion {
    InverseCumulative,
    Gumbel,
}

#[derive(Clone, Debug)]
pub struct SamplingParameters {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f64,
    pub seed: u64,
    pub version: SamplingVersion,
}

impl Default for SamplingParameters {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            seed: 0,
            version: SamplingVersion::Gumbel,
        }
    }
}

impl SamplingParameters {
    pub fn from_thousandths(temperature: u32, top_p: u32, min_p: u32) -> Result<Self> {
        let parameters = Self {
            // The upstream setter divides in f64; multiplying by .001 changes rounding.
            temperature: f64::from(temperature) / 1000.0,
            top_p: f64::from(top_p) / 1000.0,
            min_p: f64::from(min_p) / 1000.0,
            ..Self::default()
        };
        parameters.validate(20)?;
        Ok(parameters)
    }

    fn validate(&self, vocabulary_size: usize) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(Error::InvalidInput(
                "temperature must be finite and positive".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Error::InvalidInput("top_p must be in (0, 1]".into()));
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(Error::InvalidInput("min_p must be in [0, 1]".into()));
        }
        if self.top_k == 0 || self.top_k > 20 || self.top_k > vocabulary_size {
            return Err(Error::InvalidInput(
                "top_k must be in 1..=min(20, vocabulary size)".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub enum Sampler {
    #[default]
    Greedy,
    Sampled(SamplingParameters),
}

impl Sampler {
    /// `prediction_position` is the absolute input position plus one.
    pub fn sample(&self, logits: &[f32], prediction_position: u64) -> Result<u32> {
        match self {
            Self::Greedy => greedy(logits),
            Self::Sampled(parameters) => sample(logits, prediction_position, parameters),
        }
    }
}

fn validate_logits(logits: &[f32]) -> Result<()> {
    if logits.is_empty() || logits.len() > u32::MAX as usize {
        return Err(Error::InvalidInput(
            "logits must contain a nonempty u32-indexed vocabulary".into(),
        ));
    }
    require_finite(logits, "logits")
}

pub fn greedy(logits: &[f32]) -> Result<u32> {
    validate_logits(logits)?;
    let mut selected = 0;
    for token in 1..logits.len() {
        // Strict comparison retains the smaller ID, including a +0/-0 tie.
        if logits[token] > logits[selected] {
            selected = token;
        }
    }
    Ok(selected as u32)
}

pub fn top_tokens(logits: &[f32], count: usize) -> Result<Vec<u32>> {
    validate_logits(logits)?;
    if count > logits.len() {
        return Err(Error::InvalidInput(
            "top-token count exceeds vocabulary".into(),
        ));
    }
    let mut tokens: Vec<_> = (0..logits.len()).collect();
    tokens.sort_by(|&left, &right| {
        if logits[left] > logits[right] {
            std::cmp::Ordering::Less
        } else if logits[left] < logits[right] {
            std::cmp::Ordering::Greater
        } else {
            left.cmp(&right)
        }
    });
    tokens.truncate(count);
    Ok(tokens.into_iter().map(|token| token as u32).collect())
}

fn sample(logits: &[f32], position: u64, parameters: &SamplingParameters) -> Result<u32> {
    validate_logits(logits)?;
    parameters.validate(logits.len())?;
    let candidates = top_tokens(logits, parameters.top_k)?;
    let maximum = f64::from(logits[candidates[0] as usize]);
    let mut adjusted = Vec::with_capacity(candidates.len());
    let mut weights = Vec::with_capacity(candidates.len());
    let mut total = 0.0_f64;
    for &token in &candidates {
        // Widen each logit before subtraction; sum weights serially in candidate order.
        let value = (f64::from(logits[token as usize]) - maximum) / parameters.temperature;
        if !value.is_finite() {
            return Err(Error::NonFinite("temperature-adjusted logits"));
        }
        let weight = exp_d(value);
        if !weight.is_finite() {
            return Err(Error::NonFinite("sampling weight"));
        }
        adjusted.push(value);
        weights.push(weight);
        total += weight;
    }

    let mut retained = candidates.len();
    let mut prefix = 0.0_f64;
    for (index, &weight) in weights.iter().enumerate() {
        prefix += weight;
        if prefix >= parameters.top_p * total {
            retained = index + 1;
            break;
        }
    }
    if parameters.min_p > 0.0 {
        let mut index = 1;
        while index < retained && weights[index] >= parameters.min_p {
            index += 1;
        }
        retained = index;
    }

    let mut selected = retained - 1;
    match parameters.version {
        SamplingVersion::Gumbel => {
            let mut best = f64::NEG_INFINITY;
            for index in 0..retained {
                let candidate =
                    adjusted[index] + gumbel(parameters.seed, position, candidates[index]);
                if !candidate.is_finite() {
                    return Err(Error::NonFinite("Gumbel sample"));
                }
                if candidate > best {
                    best = candidate;
                    selected = index;
                }
            }
        }
        SamplingVersion::InverseCumulative => {
            let mut retained_mass = 0.0_f64;
            // Candidate-order binary64 additions are part of version 2's contract.
            for &weight in &weights[..retained] {
                retained_mass += weight;
            }
            let hash = mix64(mix64(parameters.seed) ^ position);
            let uniform = (hash >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
            let threshold = uniform * retained_mass;
            let mut prefix = 0.0_f64;
            for (index, &weight) in weights[..retained].iter().enumerate() {
                prefix += weight;
                if threshold < prefix {
                    selected = index;
                    break;
                }
            }
        }
    }
    Ok(candidates[selected])
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn gumbel(seed: u64, position: u64, token: u32) -> f64 {
    let seed_hash = mix64(seed);
    let position_hash = mix64(seed_hash ^ position);
    let hash = mix64(position_hash ^ u64::from(token).wrapping_mul(0xd6e8_feb8_6659_fd93));
    // Preserve the binary64 +0.5 rounding, including the documented upper endpoint.
    let uniform = ((hash >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0);
    -log_d(-log_d(uniform))
}

/// Stable binary64 log-sum-exp for one next-token prediction.
pub fn negative_log_likelihood(logits: &[f32], target: u32) -> Result<f64> {
    validate_logits(logits)?;
    let target = target as usize;
    if target >= logits.len() {
        return Err(Error::InvalidInput(
            "target token is outside the vocabulary".into(),
        ));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut mass = 0.0_f64;
    // Ascending token order defines the reporting reduction independently of batch size.
    for &logit in logits {
        mass += exp_d(f64::from(logit) - f64::from(maximum));
    }
    let loss = (f64::from(maximum) - f64::from(logits[target])) + log_d(mass);
    if !loss.is_finite() {
        return Err(Error::NonFinite("negative log-likelihood"));
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_order_ties_are_broken_by_token_id() {
        assert_eq!(greedy(&[-0.0, 0.0]).unwrap(), 0);
        assert_eq!(top_tokens(&[2.0, 3.0, 3.0, -1.0], 3).unwrap(), [1, 2, 0]);
        assert!(matches!(greedy(&[f32::NAN]), Err(Error::NonFinite(_))));
        assert!(greedy(&[f32::NEG_INFINITY]).is_err());
    }

    #[test]
    fn seed_and_absolute_position_are_the_only_sampling_state() {
        let logits: Vec<_> = (0..32).map(|index| (index % 7) as f32 / 8.0).collect();
        for version in [SamplingVersion::Gumbel, SamplingVersion::InverseCumulative] {
            let sampler = Sampler::Sampled(SamplingParameters {
                seed: 0x1_1234_5678,
                version,
                ..SamplingParameters::default()
            });
            let expected: Vec<_> = (100..120)
                .map(|position| sampler.sample(&logits, position).unwrap())
                .collect();
            let actual: Vec<_> = (100..120)
                .rev()
                .map(|position| sampler.sample(&logits, position).unwrap())
                .rev()
                .collect();
            assert_eq!(actual, expected);
            assert!(expected.windows(2).any(|pair| pair[0] != pair[1]));
        }
    }

    #[test]
    fn truncation_retains_at_least_the_top_candidate() {
        for version in [SamplingVersion::Gumbel, SamplingVersion::InverseCumulative] {
            let sampler = Sampler::Sampled(SamplingParameters {
                top_k: 3,
                min_p: 1.0,
                version,
                ..SamplingParameters::default()
            });
            for position in 0..16 {
                assert_eq!(sampler.sample(&[1.0, 3.0, 2.0], position).unwrap(), 1);
            }
        }
    }

    #[test]
    fn loss_is_invariant_to_an_exact_constant_logit_shift() {
        let first = negative_log_likelihood(&[1.0, 2.0, -1.0], 1).unwrap();
        let shifted = negative_log_likelihood(&[17.0, 18.0, 15.0], 1).unwrap();
        assert_eq!(first.to_bits(), shifted.to_bits());
        assert_eq!(
            negative_log_likelihood(&[4.0], 0).unwrap().to_bits(),
            0.0_f64.to_bits()
        );
    }
}
