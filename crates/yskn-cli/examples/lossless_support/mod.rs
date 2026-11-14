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
    neural_offered: usize,
    ngram_offered: usize,
    cached: usize,
    snapshots: Vec<u8>,
    sampling: Vec<Value>,
    saw_tokens: bool,
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
            .generate(prompt, max_new, sampling, stops, &mut |ids| {
                if !ids.is_empty() {
                    self.capture.borrow_mut().saw_tokens = true;
                }
                emit(ids)
            })?;
        let mut c = self.capture.borrow_mut();
        let s = sampling.normalized();
        let fixed = |v: f64| (v * 1000.0).round_ties_even() / 1000.0;
        c.sampling.push(json!({"temperature":fixed(s.temperature),"top_p":fixed(s.top_p),"top_k":s.top_k,"min_p":fixed(s.min_p),"seed":s.seed as u32}));
        c.ids.extend_from_slice(&g.ids);
        c.stages.push(g.ids.len());
        c.passes += g.passes;
        c.offered += g.guesses_offered;
        c.accepted += g.guesses_accepted;
        c.neural_offered += g.neural_guesses_offered;
        c.ngram_offered += g.ngram_guesses_offered;
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
    observe_controlled(runner, capture, p, rows, label, false)
}

fn observe_controlled(
    runner: &mut ChatRunner,
    capture: &RefCell<Capture>,
    p: &ChatParams,
    rows: &mut Vec<Value>,
    label: Value,
    cancel: bool,
) -> Result<Observation> {
    *capture.borrow_mut() = Capture::default();
    let start = Instant::now();
    let prepared = runner.prepare_checked(p)?;
    let usage_result = runner.run_prepared(
        p,
        prepared,
        &|| cancel && capture.borrow().saw_tokens,
        &mut |_| Ok(()),
    );
    let c = capture.take();
    let usage = match usage_result {
        Ok(usage) => usage,
        Err(e) => {
            let mut row = label;
            row["error"] = json!(e.to_string());
            row["token_count"] = json!(c.ids.len());
            row["stage_token_counts"] = json!(c.stages);
            rows.push(row);
            return Err(e);
        }
    };
    if c.stages.is_empty() || c.ids.is_empty() {
        return Err(Error::engine(
            "generation not exercised: no generation stage or empty token ids",
        ));
    }
    let mut digest = Sha256::new();
    for id in &c.ids {
        digest.update(id.to_le_bytes());
    }
    let mut row = label;
    row["cancelled_after_commit"] = json!(cancel && c.saw_tokens);
    row["effective_sampling_by_stage"] = json!(c.sampling);
    row["token_count"] = json!(c.ids.len());
    row["sha256_ids_le_u32"] = json!(format!("{:x}", digest.finalize()));
    row["finish_reason"] = json!(usage.finish);
    row["passes"] = json!(c.passes);
    row["guesses_offered"] = json!(c.offered);
    row["neural_guesses_offered"] = json!(c.neural_offered);
    row["ngram_guesses_offered"] = json!(c.ngram_offered);
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
    if expected.stages.is_empty()
        || actual.stages.is_empty()
        || expected.ids.is_empty()
        || actual.ids.is_empty()
    {
        failures.push(json!({"kind":"generation not exercised","comparison":kind,"run":label}));
        return;
    }
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

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn reuse_probes(
    runner: &mut ChatRunner,
    capture: &RefCell<Capture>,
    defaults: &RequestDefaults,
    name: &str,
    rows: &mut Vec<Value>,
    failures: &mut Vec<Value>,
) -> Result<()> {
    let system = "Keep the ledger in order. Each entry has a name, count, colour and shelf. Do not change existing entries. ".repeat(8);
    let body = json!({"messages":[
        {"role":"system","content":system},
        {"role":"user","content":"Remember cedar."},
        {"role":"assistant","content":"Cedar recorded."},
        {"role":"user","content":"Name the entry 3"}],"enable_thinking":false,"max_tokens":1,"temperature":0,"seed":0});
    let first = parse_chat_request(&body, defaults)?;
    let mut extension = first.clone();
    extension
        .messages
        .as_array_mut()
        .ok_or_else(|| Error::request("probe messages"))?
        .extend([
            json!({"role":"assistant","content":"Cedar."}),
            json!({"role":"user","content":"Name its shelf."}),
        ]);
    let mut sibling = first.clone();
    sibling.messages[3]["content"] = json!("Name the colour 4");
    let (a, _) = runner.prepare_checked(&first)?;
    let (b, _) = runner.prepare_checked(&extension)?;
    let (c, _) = runner.prepare_checked(&sibling)?;
    if b.len() <= a.len() || !b.starts_with(&a) {
        return Err(Error::request(format!(
            "multi-turn extension does not preserve rendered generation prefix ({} of {} tokens)",
            common_prefix(&a, &b),
            a.len()
        )));
    }
    let shared = common_prefix(&a, &c);
    if a == c || shared < 32 {
        return Err(Error::request(format!(
            "system siblings need distinct rendered ids with at least 32 shared tokens, got {shared}"
        )));
    }
    let label = |run: &str| {
        json!({"configuration":name,"sampler":"greedy","run":run,"probe":true,
        "extension_prefix_tokens":a.len(),"system_common_prefix_tokens":shared})
    };
    runner.engine.reset()?;
    let original = observe(runner, capture, &first, rows, label("conversation_first"))?;
    observe(
        runner,
        capture,
        &extension,
        rows,
        label("conversation_extension"),
    )?;
    let again = observe(
        runner,
        capture,
        &first,
        rows,
        label("conversation_first_again"),
    )?;
    compare(
        &original,
        &again,
        &label("conversation_first_again"),
        "repeat_mismatch",
        failures,
    );
    observe(runner, capture, &sibling, rows, label("system_sibling"))?;
    let again = observe(runner, capture, &first, rows, label("after_system_sibling"))?;
    compare(
        &original,
        &again,
        &label("after_system_sibling"),
        "repeat_mismatch",
        failures,
    );
    let mut unrelated = first.clone();
    unrelated.messages = json!([{"role":"user","content":"Count blue lanterns 7 8 9"}]);
    observe(runner, capture, &unrelated, rows, label("unrelated"))?;
    let again = observe(runner, capture, &first, rows, label("after_unrelated"))?;
    compare(
        &original,
        &again,
        &label("after_unrelated"),
        "repeat_mismatch",
        failures,
    );
    runner.engine.reset()?;
    let again = observe(runner, capture, &first, rows, label("after_reset"))?;
    compare(
        &original,
        &again,
        &label("after_reset"),
        "repeat_mismatch",
        failures,
    );
    observe_controlled(
        runner,
        capture,
        &sibling,
        rows,
        label("cancelled_request"),
        true,
    )?;
    let again = observe(runner, capture, &first, rows, label("after_cancelled"))?;
    compare(
        &original,
        &again,
        &label("after_cancelled"),
        "repeat_mismatch",
        failures,
    );
    let mut rejected = first.clone();
    rejected.messages = json!([{"role":"user","content":"over capacity ".repeat(runner.engine.context_capacity())}]);
    match runner.run(&rejected, &mut |_| Ok(())) {
        Err(e) if e.code == Some("context_length_exceeded") => {
            let mut row = label("failed_request");
            row["expected_error"] = json!(e.to_string());
            rows.push(row);
        }
        Err(e) => return Err(e),
        Ok(_) => return Err(Error::engine("over-context probe unexpectedly succeeded")),
    }
    let again = observe(runner, capture, &first, rows, label("after_failed"))?;
    compare(
        &original,
        &again,
        &label("after_failed"),
        "repeat_mismatch",
        failures,
    );
    Ok(())
}

fn speculation_exercised(name: &str, rows: &[Value], failures: &mut Vec<Value>) {
    if name == "none" {
        return;
    }
    for sampler in SAMPLERS {
        let own: Vec<_> = rows
            .iter()
            .filter(|r| r["sampler"] == sampler && r["probe"] != true)
            .collect();
        let sum = |key: &str| own.iter().filter_map(|r| r[key].as_u64()).sum::<u64>();
        if sum("guesses_offered") == 0 || sum("guesses_accepted") == 0 {
            failures.push(
                json!({"kind":"speculation not exercised","configuration":name,"sampler":sampler}),
            );
        }
        if name == "both-k7-adaptive" {
            for source in ["neural_guesses_offered", "ngram_guesses_offered"] {
                if sum(source) == 0 {
                    failures.push(json!({"kind":"draft source not exercised","configuration":name,"sampler":sampler,"source":source}));
                }
            }
        }
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
    let outcome = (|| -> Result<()> {
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
                    let reset = i + 1 == prompts.len();
                    if reset {
                        runner.engine.reset()?;
                    } else {
                        let unrelated =
                            parameters(&prompts[(i + 1) % prompts.len()], defaults, sampler, 1)?;
                        observe(
                            &mut runner,
                            &capture,
                            &unrelated,
                            &mut rows,
                            json!({"configuration":name,"sampler":sampler_name,"run":"unrelated_probe","probe":true}),
                        )?;
                    }
                    let label = json!({"configuration":name,"sampler":sampler_name,"prompt":prompt.id,"run":"repeat","reset_before":reset});
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
            reuse_probes(
                &mut runner,
                &capture,
                defaults,
                name,
                &mut rows,
                &mut failures,
            )?;
            let modes: HashSet<u64> = rows[config_start..]
                .iter()
                .flat_map(|r| r["snapshot_modes"].as_array().into_iter().flatten())
                .filter_map(Value::as_u64)
                .collect();
            for mode in 0..=2 {
                if !modes.contains(&mode) {
                    failures.push(
                        json!({"kind":"reuse mode not exercised","configuration":name,"mode":mode}),
                    );
                }
            }
            sampling_exercised(name, &sampled_outputs, &mut failures);
            speculation_exercised(name, &rows[config_start..], &mut failures);
            drop(runner);
        }
        Ok(())
    })();
    if let Err(e) = outcome {
        failures.push(json!({"kind":"error","message":e.to_string()}));
    }
    Ok(
        json!({"schema_version":2,"passed":failures.is_empty(),"max_tokens":max_tokens,"sampling_defaults":{"temperature":defaults.sampling.temperature,"top_p":defaults.sampling.top_p,"top_k":defaults.sampling.top_k,"min_p":defaults.sampling.min_p},"id_encoding":"concatenated engine stage ids, u32 little endian; stage boundaries recorded; bridge input ids excluded","runs":rows,"failures":failures}),
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
    fn acceptance_under_greedy_and_probes_cannot_cover_inactive_seeded_runs() {
        let mut rows = vec![json!({"sampler":"greedy","guesses_offered":10,"guesses_accepted":5})];
        for sampler in &SAMPLERS[1..] {
            rows.push(json!({"sampler":sampler,"guesses_offered":10,"guesses_accepted":0}));
            rows.push(json!({"sampler":sampler,"probe":true,"guesses_offered":10,"guesses_accepted":5,"neural_guesses_offered":5,"ngram_guesses_offered":5}));
        }
        let mut failures = vec![];
        speculation_exercised("both-k7-adaptive", &rows, &mut failures);
        assert_eq!(
            failures
                .iter()
                .filter(|f| f["kind"] == "speculation not exercised")
                .count(),
            3
        );
        assert_eq!(
            failures
                .iter()
                .filter(|f| f["kind"] == "draft source not exercised")
                .count(),
            8
        );
    }

    #[test]
    fn two_empty_generations_are_not_a_lossless_pass() {
        for stages in [vec![], vec![0]] {
            let empty = Observation {
                ids: vec![],
                stages,
                finish: "length".into(),
            };
            let mut failures = vec![];
            compare(&empty, &empty, &json!({}), "repeat_mismatch", &mut failures);
            assert_eq!(failures[0]["kind"], "generation not exercised");
        }
    }

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
