//! Development-only harness, shared with the socket-free integration tests.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::HashSet, path::Path, rc::Rc, time::Instant};
use yskn::{
    Error, Result,
    chat::{ChatParams, ChatRunner},
    engine::{Engine, Generation, Sampling},
    real_engine::{DepthMode, DraftMode, RealEngineConfig},
    request::{RequestDefaults, parse_chat_request},
};

pub const DEFAULT_PROMPTS: &str = include_str!("../lossless_prompts.json");
pub const CONFIG_NAMES: [&str; 5] = [
    "none",
    "ngram-k3-fixed",
    "neural-k3-fixed",
    "neural-k7-fixed",
    "both-k7-adaptive",
];
pub const SAMPLERS: [&str; 4] = ["greedy", "seed-42", "seed-2027", "hot-seed-91"];

#[derive(Clone)]
pub struct Prompt {
    pub id: String,
    pub request: Value,
    location: String,
}

pub fn prompts(source: &str, text: &str) -> Result<Vec<Prompt>> {
    let value: Value = serde_json::from_str(text)
        .map_err(|e| Error::request(format!("{source}: entry unknown (JSON syntax): {e}")))?;
    let entries = value.as_array().filter(|a| a.len() >= 2).ok_or_else(|| {
        Error::request(format!(
            "{source}: entry list must contain at least two prompts"
        ))
    })?;
    let mut ids = HashSet::new();
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let fail = |message: &str| Error::request(format!("{source}: entry {i}: {message}"));
            let id = entry["id"]
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| fail("id must be a nonempty string"))?;
            if !ids.insert(id.to_owned()) {
                return Err(fail("duplicate id"));
            }
            let request = entry
                .get("request")
                .filter(|v| v.is_object())
                .ok_or_else(|| fail("request must be an OpenAI chat object"))?;
            parse_chat_request(request, &RequestDefaults::default())
                .map_err(|e| fail(&e.to_string()))?;
            // The harness owns these fields so every entry really tests all four samplers.
            for key in ["temperature", "top_p", "top_k", "min_p", "seed"] {
                if request.get(key).is_some() {
                    return Err(fail(&format!("{key} is selected by the harness")));
                }
            }
            Ok(Prompt {
                id: id.into(),
                request: request.clone(),
                location: format!("{source}: entry {i} ({id})"),
            })
        })
        .collect()
}

pub fn configs(list: &str) -> Result<Vec<String>> {
    let mut selected = HashSet::new();
    for name in list.split(',') {
        if !CONFIG_NAMES.contains(&name) || !selected.insert(name) {
            return Err(Error::request(format!(
                "invalid or duplicate configuration {name:?}; choose {}",
                CONFIG_NAMES.join(",")
            )));
        }
    }
    // Always include and run the oracle first, even if the user selects only drafts.
    Ok(CONFIG_NAMES
        .iter()
        .filter(|&&s| s == "none" || selected.contains(s))
        .map(|s| (*s).into())
        .collect())
}

pub fn configure(base: &RealEngineConfig, name: &str) -> Result<RealEngineConfig> {
    let mut c = base.clone();
    (c.draft, c.k, c.depth) = match name {
        "none" => (DraftMode::None, 0, DepthMode::Fixed),
        "ngram-k3-fixed" => (DraftMode::Ngram, 3, DepthMode::Fixed),
        "neural-k3-fixed" => (DraftMode::Neural, 3, DepthMode::Fixed),
        "neural-k7-fixed" => (DraftMode::Neural, 7, DepthMode::Fixed),
        "both-k7-adaptive" => (DraftMode::Both, 7, DepthMode::Adaptive),
        _ => return Err(Error::request(format!("unknown configuration {name}"))),
    };
    Ok(c)
}

#[derive(Default)]
struct Capture {
    ids: Vec<u32>,
    stages: Vec<usize>,
    passes: usize,
    offered: usize,
    accepted: usize,
    cached: usize,
    snapshots: Vec<u8>,
    sampling: Vec<Value>,
}
struct Recorder {
    inner: Box<dyn Engine>,
    capture: Rc<RefCell<Capture>>,
}
impl Engine for Recorder {
    fn context_capacity(&self) -> usize {
        self.inner.context_capacity()
    }
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stops: Option<&[u32]>,
        emit: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation> {
        let g = self
            .inner
            .generate(prompt, max_new, sampling, stops, emit)?;
        let mut c = self.capture.borrow_mut();
        let s = sampling.normalized();
        let fixed = |v: f64| (v * 1000.0).round_ties_even() / 1000.0;
        c.sampling.push(json!({"temperature":fixed(s.temperature),"top_p":fixed(s.top_p),"top_k":s.top_k,"min_p":fixed(s.min_p),"seed":s.seed as u32}));
        c.ids.extend_from_slice(&g.ids);
        c.stages.push(g.ids.len());
        c.passes += g.passes;
        c.offered += g.guesses_offered;
        c.accepted += g.guesses_accepted;
        c.cached += g.cached_tokens;
        c.snapshots.push(g.snap_mode);
        Ok(g)
    }
}

struct Observation {
    ids: Vec<u32>,
    stages: Vec<usize>,
    finish: String,
}

fn parameters(
    prompt: &Prompt,
    defaults: &RequestDefaults,
    sampler: usize,
    max_tokens: usize,
) -> Result<ChatParams> {
    let mut body = prompt.request.clone();
    body["seed"] = json!([0, 42, 2027, 91][sampler]);
    if sampler == 0 {
        body["temperature"] = json!(0.0);
    }
    if sampler == 1
        || sampler == 3
        || (sampler == 2 && (defaults.sampling.temperature <= 0.0 || defaults.sampling.top_k == 1))
    {
        body["temperature"] = json!(if sampler == 3 { 1.3 } else { 0.7 });
        body["top_p"] = json!(if sampler == 3 { 1.0 } else { 0.9 });
        body["top_k"] = json!(20);
        body["min_p"] = json!(0.0);
    }
    let mut p = parse_chat_request(&body, defaults)
        .map_err(|e| Error::request(format!("{}: {e}", prompt.location)))?;
    p.max_tokens = p.max_tokens.min(max_tokens);
    Ok(p)
}

fn observe(
    runner: &mut ChatRunner,
    capture: &RefCell<Capture>,
    p: &ChatParams,
    rows: &mut Vec<Value>,
    label: Value,
) -> Result<Observation> {
    *capture.borrow_mut() = Capture::default();
    let start = Instant::now();
    let usage = runner.run(p, &mut |_| Ok(()))?;
    let c = capture.take();
    let mut digest = Sha256::new();
    for id in &c.ids {
        digest.update(id.to_le_bytes());
    }
    let mut row = label;
    row["effective_sampling_by_stage"] = json!(c.sampling);
    row["token_count"] = json!(c.ids.len());
    row["sha256_ids_le_u32"] = json!(format!("{:x}", digest.finalize()));
    row["finish_reason"] = json!(usage.finish);
    row["passes"] = json!(c.passes);
    row["guesses_offered"] = json!(c.offered);
    row["guesses_accepted"] = json!(c.accepted);
    row["cached_tokens"] = json!(c.cached);
    row["snapshot_modes"] = json!(c.snapshots);
    row["stage_token_counts"] = json!(c.stages);
    row["prompt_tokens"] = json!(usage.prompt_tokens);
    row["seconds"] = json!(start.elapsed().as_secs_f64());
    row["prefill_ms"] = json!(usage.prefill_ms);
    row["generation_ms"] = json!(usage.gen_ms);
    rows.push(row);
    Ok(Observation {
        ids: c.ids,
        stages: c.stages,
        finish: usage.finish.into(),
    })
}

fn compare(
    expected: &Observation,
    actual: &Observation,
    label: &Value,
    kind: &str,
    failures: &mut Vec<Value>,
) {
    let position = expected
        .ids
        .iter()
        .zip(&actual.ids)
        .position(|(a, b)| a != b)
        .or_else(|| {
            (expected.ids.len() != actual.ids.len())
                .then_some(expected.ids.len().min(actual.ids.len()))
        });
    if let Some(position) = position {
        let mut failure = label.clone();
        failure["kind"] = json!(kind);
        failure["position"] = json!(position);
        failure["expected_id"] = json!(expected.ids.get(position));
        failure["actual_id"] = json!(actual.ids.get(position));
        eprintln!("{failure}");
        failures.push(failure);
    } else if expected.finish != actual.finish || expected.stages != actual.stages {
        let failure = json!({"kind":kind,"run":label,"expected_finish":expected.finish,"actual_finish":actual.finish,"expected_stages":expected.stages,"actual_stages":actual.stages});
        eprintln!("{failure}");
        failures.push(failure);
    }
}

fn sampling_exercised(name: &str, outputs: &[Vec<Vec<u32>>], failures: &mut Vec<Value>) {
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2)] {
        if outputs[a] == outputs[b] {
            failures.push(json!({"kind":"sampling not exercised","configuration":name,"samplers":[SAMPLERS[a],SAMPLERS[b]]}));
        }
    }
}

/// The factory is called only after the previous runner and its weights drop.
/// Tests supply actual synthetic RealEngines, with faults confined to test wrappers.
pub fn check(
    prompts: &[Prompt],
    names: &[String],
    defaults: &RequestDefaults,
    max_tokens: usize,
    mut factory: impl FnMut(&str) -> Result<Box<dyn Engine>>,
    formatter: &yskn_text::ChatFormatter,
) -> Result<Value> {
    if prompts.len() < 2 || max_tokens == 0 || names.first().map(String::as_str) != Some("none") {
        return Err(Error::request(
            "need two prompts, positive max-tokens, and none first",
        ));
    }
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut baseline = Vec::new();
    for name in names {
        eprintln!("loading {name}");
        let capture = Rc::new(RefCell::new(Capture::default()));
        let engine = factory(name)?;
        let mut runner = ChatRunner::new(
            Box::new(Recorder {
                inner: engine,
                capture: capture.clone(),
            }),
            formatter.clone(),
            0,
        )?;
        let config_start = rows.len();
        let mut sampled_outputs: Vec<Vec<Vec<u32>>> = Vec::new();
        for (sampler, sampler_name) in SAMPLERS.iter().enumerate() {
            let mut first = Vec::new();
            // Full first sweep: each repeat has unrelated requests in between.
            for prompt in prompts {
                let p = parameters(prompt, defaults, sampler, max_tokens)?;
                let label = json!({"configuration":name,"sampler":sampler_name,"prompt":prompt.id,"run":"first"});
                let result = observe(&mut runner, &capture, &p, &mut rows, label.clone())
                    .map_err(|e| Error::engine(format!("{}: {label}: {e}", prompt.location)))?;
                if name != "none" {
                    compare(
                        &baseline[sampler * prompts.len() + first.len()],
                        &result,
                        &label,
                        "configuration_mismatch",
                        &mut failures,
                    );
                }
                first.push(result);
            }
            for (i, prompt) in prompts.iter().enumerate() {
                let p = parameters(prompt, defaults, sampler, max_tokens)?;
                // A one-token sibling differs only in the template's thinking suffix:
                // it primes the same saved prompt prefix immediately before the repeat.
                let mut sibling = p.clone();
                sibling.max_tokens = 1;
                let thinking = p.template_kwargs["enable_thinking"]
                    .as_bool()
                    .unwrap_or(true);
                sibling.template_kwargs["enable_thinking"] = json!(!thinking);
                let (ids, _) = runner
                    .prepare_checked(&p)
                    .map_err(|e| Error::request(format!("{}: {e}", prompt.location)))?;
                let (sibling_ids, _) = runner.prepare_checked(&sibling).map_err(|e| {
                    Error::request(format!("{}: prefix probe: {e}", prompt.location))
                })?;
                if ids == sibling_ids {
                    return Err(Error::request(format!(
                        "{}: template ignores thinking toggle; cannot exercise a distinct prefix sibling",
                        prompt.location
                    )));
                }
                let label = json!({"configuration":name,"sampler":sampler_name,"prompt":prompt.id,"run":"prefix_probe"});
                observe(&mut runner, &capture, &sibling, &mut rows, label).map_err(|e| {
                    Error::engine(format!("{}: prefix probe: {e}", prompt.location))
                })?;
                let reset = i + 1 == prompts.len();
                if reset {
                    runner.engine.reset()?;
                }
                let label = json!({"configuration":name,"sampler":sampler_name,"prompt":prompt.id,"run":"repeat","reset_before":reset,"sibling_common_prefix_tokens":ids.iter().zip(&sibling_ids).take_while(|(a,b)| a == b).count()});
                let result = observe(&mut runner, &capture, &p, &mut rows, label.clone())
                    .map_err(|e| Error::engine(format!("{}: {label}: {e}", prompt.location)))?;
                compare(&first[i], &result, &label, "repeat_mismatch", &mut failures);
                if name != "none" {
                    compare(
                        &baseline[sampler * prompts.len() + i],
                        &result,
                        &label,
                        "configuration_mismatch",
                        &mut failures,
                    );
                }
            }
            sampled_outputs.push(first.iter().map(|o| o.ids.clone()).collect());
            if name == "none" {
                baseline.extend(first);
            }
        }
        sampling_exercised(name, &sampled_outputs, &mut failures);
        let accepted: u64 = rows[config_start..]
            .iter()
            .filter(|r| r["run"] != "prefix_probe")
            .filter_map(|r| r["guesses_accepted"].as_u64())
            .sum();
        if name != "none" && accepted == 0 {
            let failure = json!({"kind":"no_guess_accepted","configuration":name});
            eprintln!("{failure}");
            failures.push(failure);
        }
        drop(runner);
    }
    Ok(
        json!({"schema_version":1,"passed":failures.is_empty(),"max_tokens":max_tokens,"sampling_defaults":{"temperature":defaults.sampling.temperature,"top_p":defaults.sampling.top_p,"top_k":defaults.sampling.top_k,"min_p":defaults.sampling.min_p},"id_encoding":"concatenated engine stage ids, u32 little endian; stage boundaries recorded; bridge input ids excluded","runs":rows,"failures":failures}),
    )
}

pub fn load_prompts(path: Option<&Path>) -> Result<Vec<Prompt>> {
    match path {
        Some(path) => prompts(
            &path.display().to_string(),
            &std::fs::read_to_string(path)
                .map_err(|e| Error::request(format!("{}: {e}", path.display())))?,
        ),
        None => prompts("embedded lossless_prompts.json", DEFAULT_PROMPTS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_sampling_overrides_greedy_and_single_choice_defaults() {
        let prompt = prompts("test", r#"[{"id":"a","request":{"messages":[{"role":"user","content":"x"}]}},{"id":"b","request":{"messages":[{"role":"user","content":"y"}]}}]"#).unwrap().remove(0);
        let mut defaults = RequestDefaults::default();
        defaults.sampling.temperature = 0.0;
        defaults.sampling.top_k = 1;
        for sampler in 1..4 {
            let p = parameters(&prompt, &defaults, sampler, 32).unwrap();
            assert!(p.temperature > 0.0 && p.top_k > 1 && p.top_p >= 0.9);
        }
        defaults.sampling.temperature = 0.8;
        defaults.sampling.top_k = 11;
        let p = parameters(&prompt, &defaults, 2, 32).unwrap();
        assert_eq!((p.temperature, p.top_k), (0.8, 11));
    }

    #[test]
    fn identical_samplers_and_identical_seeds_fail_the_diversity_gate() {
        let mut failures = vec![];
        sampling_exercised("none", &vec![vec![vec![1]]; 4], &mut failures);
        assert_eq!(failures.len(), 4);
        assert!(
            failures
                .iter()
                .all(|f| f["kind"] == "sampling not exercised")
        );
        failures.clear();
        sampling_exercised(
            "none",
            &[vec![vec![0]], vec![vec![1]], vec![vec![1]], vec![vec![3]]],
            &mut failures,
        );
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0]["samplers"], json!(["seed-42", "seed-2027"]));
    }
}
