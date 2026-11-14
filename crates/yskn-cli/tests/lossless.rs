//! Socket-free proof of the development harness against the actual product engine.
#[allow(dead_code)]
#[path = "../examples/lossless_support/mod.rs"]
mod lossless_support;
#[allow(dead_code)]
mod product_support;

use lossless_support::{CONFIG_NAMES, check, configure, prompts};
use product_support::Folder;
use serde_json::{Value, json};
use std::{cell::Cell, collections::HashMap, process::Command, rc::Rc};
use yskn::{
    Result,
    engine::{Engine, Generation, Sampling},
    real_engine::RealEngine,
    request::RequestDefaults,
};

#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    Configuration,
    Repeat,
    NoAcceptance,
    LaterError,
    Empty,
}
struct TestEngine {
    inner: RealEngine,
    fault: Fault,
    counts: HashMap<(Vec<u32>, i64), usize>,
    live: Rc<Cell<usize>>,
}
impl Drop for TestEngine {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}
impl Engine for TestEngine {
    fn context_capacity(&self) -> usize {
        self.inner.context_capacity()
    }
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        n: usize,
        sampling: &Sampling,
        stops: Option<&[u32]>,
        emit: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation> {
        if self.fault == Fault::LaterError && prompt.first() == Some(&13) && n > 1 {
            return Err(yskn::Error::engine("injected later generation error"));
        }
        let mut result = if self.fault == Fault::Empty {
            self.inner
                .generate(prompt, n, sampling, stops, &mut |_| Ok(true))?
        } else {
            self.inner.generate(prompt, n, sampling, stops, emit)?
        };
        let original = prompt.first() == Some(&1) && n > 1;
        let count = self
            .counts
            .entry((prompt.to_vec(), sampling.seed))
            .or_default();
        if original {
            *count += 1;
        }
        if original
            && (matches!(self.fault, Fault::Configuration | Fault::LaterError)
                || (self.fault == Fault::Repeat && *count == 2))
        {
            result.ids[2] = (result.ids[2] + 1) % 32;
        }
        if self.fault == Fault::Empty {
            result.ids.clear();
        }
        if self.fault == Fault::NoAcceptance {
            result.guesses_accepted = 0;
        }
        Ok(result)
    }
}

fn exercise(fault: Fault, all: bool) -> Value {
    let folder = Folder::new();
    // Generation headers are present on historical assistant turns too. The
    // system setting changes the start, as it does in the real chat template.
    folder.write("chat_template.jinja", "{% if enable_thinking %}t13 {% endif %}{% for m in messages %}{% if m.role == 'assistant' %}t10 {% endif %}{{ m.content }} {% endfor %}t10");
    let cases = prompts("synthetic.json", &json!([
        {"id":"refrain","request":{"messages":[{"role":"user","content":"1 2 1 2 1 2 1 2"}],"enable_thinking":false,"max_tokens":12}},
        {"id":"different","request":{"messages":[{"role":"user","content":"3 4 5 6"}],"enable_thinking":true,"max_tokens":12}}
    ]).to_string()).unwrap();
    let selection = if all {
        CONFIG_NAMES.join(",")
    } else {
        "none,ngram-k3-fixed".into()
    };
    let names = lossless_support::configs(&selection).unwrap();
    let live = Rc::new(Cell::new(0));
    let defaults = RequestDefaults {
        max_tokens: 12,
        sampling: folder
            .config(yskn::real_engine::DraftMode::None)
            .sampling_defaults()
            .unwrap(),
        ..RequestDefaults::default()
    };
    let result = check(
        &cases,
        &names,
        &defaults,
        12,
        "full",
        |name| {
            assert_eq!(
                live.get(),
                0,
                "previous weights must drop before the next load"
            );
            let config = configure(&folder.config(yskn::real_engine::DraftMode::None), name)?;
            let mut options = product_support::model::draft::options();
            options.capture_layers = vec![0];
            let inner = RealEngine::load_with_draft_options(config, options)?;
            live.set(live.get() + 1);
            Ok(Box::new(TestEngine {
                inner,
                fault: if name == "none" { Fault::None } else { fault },
                counts: HashMap::new(),
                live: live.clone(),
            }))
        },
        &yskn_text::ChatFormatter::load(&folder.0).unwrap(),
    )
    .unwrap();
    assert_eq!(live.get(), 0);
    result
}

#[test]
fn actual_synthetic_engine_matches_ids_but_rejects_unexercised_draft_cells() {
    let report = exercise(Fault::None, true);
    assert_eq!(report["passed"], false);
    let failures = report["failures"].as_array().unwrap();
    assert_eq!(failures.len(), 8, "{failures:?}");
    assert!(
        failures
            .iter()
            .all(|f| f["kind"] == "speculation not exercised"
                || f["kind"] == "draft source not exercised"),
        "{failures:?}"
    );
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(
        runs.iter().filter(|r| r["probe"] != true).count(),
        5 * 4 * 2 * 2
    );
    for run in runs
        .iter()
        .filter(|r| r["configuration"] == "none" && r["probe"] == true)
    {
        eprintln!(
            "reuse {}: modes={} cached={}",
            run["run"], run["snapshot_modes"], run["cached_tokens"]
        );
    }
    for name in CONFIG_NAMES {
        let own: Vec<_> = runs.iter().filter(|r| r["configuration"] == name).collect();
        assert!(
            own.iter()
                .filter(|r| r["expected_error"].is_null())
                .all(|r| r["passes"].as_u64().unwrap() > 0)
        );
        assert!(own.iter().any(|r| r["run"] == "after_failed"
            && r["snapshot_modes"] == json!([2])
            && r["cached_tokens"].as_u64().unwrap() > 0));
        assert!(
            own.iter()
                .any(|r| r["reset_before"] == true && r["snapshot_modes"] == json!([0]))
        );
        assert!(
            own.iter()
                .any(|r| r["run"] == "conversation_extension" && r["snapshot_modes"] == json!([1]))
        );
        for run in [
            "after_system_sibling",
            "after_unrelated",
            "after_reset",
            "after_cancelled",
            "after_failed",
        ] {
            assert!(
                own.iter().any(|r| r["run"] == run && r["token_count"] == 1),
                "{report}"
            );
        }
        for row in own.iter().filter(|r| r["expected_error"].is_null()) {
            assert_eq!(
                row["guesses_offered"].as_u64().unwrap(),
                row["neural_guesses_offered"].as_u64().unwrap()
                    + row["ngram_guesses_offered"].as_u64().unwrap()
            );
        }
        if name != "none" {
            assert!(
                own.iter()
                    .filter(|r| r["probe"] != true)
                    .map(|r| r["guesses_accepted"].as_u64().unwrap())
                    .sum::<u64>()
                    > 0
            );
        }
    }
}

#[test]
fn an_empty_actual_engine_generation_fails_instead_of_comparing_equal() {
    let report = exercise(Fault::Empty, false);
    assert_eq!(report["passed"], false);
    assert!(report["failures"].as_array().unwrap().iter().any(|f| {
        f["kind"] == "error"
            && f["message"]
                .as_str()
                .unwrap()
                .contains("generation not exercised")
    }));
}

#[test]
fn later_error_preserves_the_mismatch_and_completed_rows() {
    let report = exercise(Fault::LaterError, false);
    assert_eq!(report["passed"], false);
    let failures = report["failures"].as_array().unwrap();
    assert!(
        failures
            .iter()
            .any(|f| f["kind"] == "configuration_mismatch")
    );
    assert!(failures.iter().any(|f| {
        f["kind"] == "error"
            && f["message"]
                .as_str()
                .unwrap()
                .contains("injected later generation error")
    }));
    assert!(report["runs"].as_array().unwrap().len() > 1);
}

#[test]
fn fault_child() {
    let Ok(mode) = std::env::var("YSKN_LOSSLESS_TEST_FAULT") else {
        return;
    };
    let fault = match mode.as_str() {
        "configuration" => Fault::Configuration,
        "repeat" => Fault::Repeat,
        "acceptance" => Fault::NoAcceptance,
        _ => panic!("unknown test fault"),
    };
    let report = exercise(fault, false);
    println!("{report}");
    std::process::exit(if report["passed"] == true { 0 } else { 1 });
}

#[test]
fn token_and_repeat_faults_and_inactive_drafting_exit_nonzero_with_diagnostics() {
    for (mode, kind) in [
        ("configuration", "configuration_mismatch"),
        ("repeat", "repeat_mismatch"),
        ("acceptance", "speculation not exercised"),
    ] {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "fault_child", "--nocapture"])
            .env("YSKN_LOSSLESS_TEST_FAULT", mode)
            .output()
            .unwrap();
        assert_eq!(
            child.status.code(),
            Some(1),
            "{}",
            String::from_utf8_lossy(&child.stderr)
        );
        let stdout = String::from_utf8(child.stdout).unwrap();
        let report: Value = stdout
            .lines()
            .find_map(|s| {
                serde_json::from_str::<Value>(s)
                    .ok()
                    .filter(|v| v.get("passed").is_some())
            })
            .unwrap();
        let failures = report["failures"].as_array().unwrap();
        assert!(failures.iter().any(|f| f["kind"] == kind), "{report}");
        if mode != "acceptance" {
            let stderr = String::from_utf8(child.stderr).unwrap();
            assert!(stderr.contains("\"prompt\":\"refrain\""), "{stderr}");
            assert!(stderr.contains("\"position\":2"), "{stderr}");
            assert!(failures.iter().any(|f| f["prompt"] == "refrain"
                && f["position"] == 2
                && f["expected_id"] != f["actual_id"]));
        } else {
            assert!(failures.iter().all(|f| f["kind"] == kind), "{report}");
        }
    }
}

#[test]
fn malformed_prompt_files_name_the_file_and_entry_instead_of_panicking() {
    for text in [
        "{",
        "null",
        "[{},{}]",
        "[{\"id\":\"x\",\"request\":{}},{}]",
        "[{\"id\":\"x\",\"request\":{\"messages\":[{\"role\":\"user\",\"content\":\"\"}],\"seed\":1}},{}]",
    ] {
        let error = prompts("broken.json", text).err().unwrap().to_string();
        assert!(
            error.contains("broken.json") && error.contains("entry"),
            "{error}"
        );
    }
    let defaults = prompts("defaults.json", lossless_support::DEFAULT_PROMPTS).unwrap();
    assert_eq!(defaults.len(), 24);
    assert_eq!(
        defaults
            .iter()
            .map(|p| p.request["max_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        1792
    );
    let mut budgets: Vec<_> = defaults
        .iter()
        .map(|p| p.request["max_tokens"].as_u64().unwrap())
        .collect();
    budgets.sort();
    assert_eq!(budgets, [vec![32; 16], vec![64; 4], vec![256; 4]].concat());
    assert_eq!(
        lossless_support::profile_configs("release").unwrap().len(),
        4
    );
    assert_eq!(
        lossless_support::profile_samplers("release").unwrap(),
        &[0, 1, 3]
    );
    assert!(lossless_support::profile_samplers("invalid").is_err());
    let entries: Value = serde_json::from_str(lossless_support::DEFAULT_PROMPTS).unwrap();
    assert_eq!(
        entries
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["quick_max_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        256
    );
    assert_eq!(
        lossless_support::configs("neural-k7-fixed").unwrap(),
        ["none", "neural-k7-fixed"]
    );
    assert!(lossless_support::configs("none,none").is_err());
}

#[test]
fn real_example_binary_saves_report_and_exits_nonzero_for_an_injected_mismatch() {
    let folder = Folder::new();
    folder.write("chat_template.jinja", "{% if enable_thinking %}t13 {% endif %}{% for m in messages %}{% if m.role == 'assistant' %}t10 {% endif %}{{ m.content }} {% endfor %}t10");
    folder.json("prompts.json", json!([
        {"id":"a","request":{"messages":[{"role":"user","content":"1 2 1 2 1 2"}],"enable_thinking":false,"max_tokens":12}},
        {"id":"b","request":{"messages":[{"role":"user","content":"3 4 5 6"}],"enable_thinking":false,"max_tokens":12}}
    ]));
    // Cargo's test --all-targets build also builds the actual example binary.
    let exe = std::env::current_exe().unwrap();
    let mut root = exe.parent().unwrap();
    while root.file_name().is_some_and(|n| n != "release") {
        root = root.parent().expect("release target root");
    }
    let binary = root.join("examples/lossless");
    assert!(
        binary.is_file(),
        "build with cargo build -p yskn --release --example lossless first: {}",
        binary.display()
    );
    for inject in [false, true] {
        let report_path = folder.0.join(if inject { "bad.json" } else { "good.json" });
        let mut command = Command::new(&binary);
        command
            .args(["--profile", "full", "--configs", "none", "--ctx", "512"])
            .arg("--model-dir")
            .arg(&folder.0)
            .arg("--prompts")
            .arg(folder.0.join("prompts.json"))
            .arg("--report")
            .arg(&report_path)
            .env("YSKN_BACKEND", "cpu")
            .env_remove("YSKN_LOSSLESS_INJECT_MISMATCH");
        if inject {
            command.env("YSKN_LOSSLESS_INJECT_MISMATCH", "1");
        }
        let child = command.output().unwrap();
        let report: Value = serde_json::from_slice(
            &std::fs::read(&report_path)
                .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&child.stderr))),
        )
        .unwrap();
        assert_eq!(
            child.status.success(),
            !inject,
            "{} {report}",
            String::from_utf8_lossy(&child.stderr)
        );
        assert_eq!(report["passed"], !inject);
        if inject {
            assert!(
                report["failures"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|f| f["kind"] == "repeat_mismatch" && f["position"] == 0)
            );
        }
    }
}
