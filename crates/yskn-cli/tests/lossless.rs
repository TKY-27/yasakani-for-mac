//! Socket-free proof of the development harness against the actual product engine.
#[allow(dead_code)]
#[path = "../examples/lossless_support/mod.rs"]
mod lossless_support;
#[allow(dead_code)]
mod product_support;

use lossless_support::{CONFIG_NAMES, check, configure, prompts};
use product_support::Folder;
use serde_json::{Value, json};
use std::{cell::Cell, collections::HashMap, process::Command, rc::Rc, sync::Mutex};

// The default-dimension teacher shares a memory budget with the small fixtures.
static MODEL_RUN: Mutex<()> = Mutex::new(());
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
    ConstantSeed,
    LiveReuse,
    SiblingReuse,
}
struct TestEngine {
    inner: RealEngine,
    fault: Fault,
    counts: HashMap<(Vec<u32>, i64), usize>,
    live: Rc<Cell<usize>>,
    reset_requested: bool,
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
        self.reset_requested = true;
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
        let mut requested = sampling.clone();
        if self.fault == Fault::ConstantSeed {
            requested.seed = 42;
        }
        let sampling = &requested;
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
        if n == 8
            && ((self.fault == Fault::LiveReuse && result.snap_mode == 1)
                || (self.fault == Fault::SiblingReuse
                    && prompt.contains(&4)
                    && !self.reset_requested))
        {
            result.ids[0] ^= 1;
        }
        if self.fault == Fault::Empty {
            result.ids.clear();
        }
        if self.fault == Fault::NoAcceptance {
            result.guesses_accepted = 0;
        }
        self.reset_requested = false;
        Ok(result)
    }
}

fn exercise(fault: Fault, all: bool) -> Value {
    exercise_template(
        fault,
        all,
        "{% if enable_thinking %}t13 {% endif %}{% for m in messages %}{% if m.role == 'assistant' %}t10 {% endif %}{{ m.content }} {% endfor %}t10",
    )
}

fn exercise_template(fault: Fault, all: bool, template: &str) -> Value {
    let _run = MODEL_RUN.lock().unwrap();
    let folder = Folder::new();
    // Generation headers are present on historical assistant turns too. The
    // system setting changes the start, as it does in the real chat template.
    folder.write("chat_template.jinja", template);
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
                fault: if name == "none" && fault != Fault::ConstantSeed {
                    Fault::None
                } else {
                    fault
                },
                counts: HashMap::new(),
                live: live.clone(),
                reset_requested: false,
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
    assert_eq!(failures.len(), 9, "{failures:?}");
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
fn rewritten_continuation_is_compared_and_fixed_pair_still_exercises_live_reuse() {
    let report = exercise_template(
        Fault::None,
        false,
        "{% if enable_thinking %}t13 {% endif %}{% for m in messages %}{% if m.role == 'assistant' %}t10 t11 {% endif %}{{ m.content }} {% endfor %}t10",
    );
    let runs = report["runs"].as_array().unwrap();
    for name in ["none", "ngram-k3-fixed"] {
        let real = runs
            .iter()
            .find(|r| r["configuration"] == name && r["run"] == "conversation_extension")
            .unwrap();
        assert_eq!(real["continuation_strict_prefix"], false);
        assert_ne!(real["assistant_answer"], "Cedar.");
        assert_eq!(real["token_count"], 8);
        let fixed = runs
            .iter()
            .find(|r| r["configuration"] == name && r["run"] == "fixed_extension")
            .unwrap();
        assert_eq!(fixed["snapshot_modes"], json!([1]));
        assert!(fixed["cached_tokens"].as_u64().unwrap() > 0);
        let reset = runs
            .iter()
            .find(|r| r["configuration"] == name && r["run"] == "fixed_extension_reset")
            .unwrap();
        assert_eq!(fixed["sha256_ids_le_u32"], reset["sha256_ids_le_u32"]);
    }
    assert!(
        report["failures"]
            .as_array()
            .unwrap()
            .iter()
            .all(|f| f["kind"] == "speculation not exercised"
                || f["kind"] == "draft source not exercised"),
        "{}",
        report["failures"]
    );
}

#[test]
fn wrong_ids_only_during_live_or_sibling_reuse_fail_comparison() {
    for (fault, run) in [
        (Fault::LiveReuse, "conversation_extension"),
        (Fault::SiblingReuse, "system_sibling"),
    ] {
        let report = exercise(fault, false);
        assert_eq!(report["passed"], false);
        assert!(
            report["failures"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f["kind"] == "reuse_mismatch" && f["run"] == run && f["position"] == 0),
            "{report}"
        );
    }
}

#[test]
fn a_constant_seed_fails_the_same_settings_probe() {
    let report = exercise(Fault::ConstantSeed, true);
    let failures = report["failures"].as_array().unwrap();
    for name in ["none", "both-k7-adaptive"] {
        assert!(
            failures
                .iter()
                .any(|f| f["kind"] == "seed not exercised" && f["configuration"] == name),
            "{report}"
        );
    }
    let runs = report["runs"].as_array().unwrap();
    let own: Vec<_> = runs
        .iter()
        .filter(|r| r["configuration"] == "none" && r["sampler"] == "seed-only")
        .collect();
    assert_eq!(own.len(), 8);
    assert_eq!(
        own.iter()
            .filter(|r| r["requested_sampling_by_stage"][0]["seed"] == 42)
            .count(),
        4
    );
    assert_eq!(
        own.iter()
            .filter(|r| r["requested_sampling_by_stage"][0]["seed"] == 2027)
            .count(),
        4
    );
    let mut settings: Vec<_> = own
        .iter()
        .map(|r| r["requested_sampling_by_stage"][0].clone())
        .collect();
    for setting in &mut settings {
        setting.as_object_mut().unwrap().remove("seed");
    }
    assert!(settings.iter().all(|s| s == &settings[0]));
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
    let _run = MODEL_RUN.lock().unwrap();
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

/// Constant-zero main logits and draft scores form a uniform teacher. The
/// real default capture layers, mask ID and shortlist remain in use.
fn uniform_teacher_fixture(folder: &Folder) {
    folder.neural_bench_fixture();
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut main = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(folder.0.join("weights.yskn"))
        .unwrap();
    let mut prefix = [0; 32];
    main.read_exact(&mut prefix).unwrap();
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().unwrap()) as usize;
    let mut bytes = vec![0; header_len];
    main.read_exact(&mut bytes).unwrap();
    let header: Value = serde_json::from_slice(&bytes).unwrap();
    let head = header["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "lm_head")
        .unwrap();
    let offset = head["scale_off"].as_u64().unwrap();
    let rows = head["physical_rows"].as_u64().unwrap() as usize;
    main.seek(SeekFrom::Start(offset)).unwrap();
    main.write_all(&vec![0; rows * 2]).unwrap();
    drop(main);
    let mut draft = std::fs::read(folder.0.join("draft.yskn")).unwrap();
    let header_len = u64::from_le_bytes(draft[8..16].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&draft[32..32 + header_len]).unwrap();
    for v in header["vectors"].as_array().unwrap() {
        if v["name"].as_str().unwrap().starts_with("bdraft2.sel.") {
            let off = v["off"].as_u64().unwrap() as usize;
            let len = v["n"].as_u64().unwrap() as usize * 2;
            draft[off..off + len].fill(0);
        }
    }
    folder.write("draft.yskn", draft);
}

fn example_binary() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap();
    let mut root = exe.parent().unwrap();
    while root.file_name().is_some_and(|n| n != "release") {
        root = root.parent().expect("release target root");
    }
    let binary = root.join("examples/lossless");
    assert!(
        binary.is_file(),
        "build cargo build -p yskn --release --example lossless first: {}",
        binary.display()
    );
    binary
}

#[test]
fn real_example_binary_passes_every_configuration_with_a_uniform_teacher() {
    let _run = MODEL_RUN.lock().unwrap();
    use yskn_ref::{Sampler, SamplingParameters, SamplingVersion};
    let folder = Folder::new();
    uniform_teacher_fixture(&folder);
    folder.write("chat_template.jinja", "{% for m in messages %}{% if m.role == 'assistant' %}t10 {% endif %}{{ m.content }} {% endfor %}t10");
    // Known uniform teacher: main ties choose 0 greedily, sampled top-k retains
    // low IDs. Embed its future continuations so n-gram reuse can offer long
    // matches under every sampler, including both's six-token threshold.
    let mut teacher = Vec::new();
    for (seed, temperature, top_p, top_k, min_p) in [
        (42, 0.7, 0.9, 20, 0.0),
        (2027, 0.75, 0.9, 11, 0.05),
        (91, 1.3, 1.0, 20, 0.0),
    ] {
        let sampler = Sampler::Sampled(SamplingParameters {
            seed,
            temperature,
            top_p,
            top_k,
            min_p,
            version: SamplingVersion::Gumbel,
        });
        for position in 129..141 {
            teacher.push(sampler.sample(&[0.0; 32], position).unwrap());
        }
    }
    let cases: Vec<_> = [1, 2].into_iter().map(|id| {
        let mut words = vec![id.to_string()];
        words.extend(teacher.iter().map(|id| if *id < 10 { id.to_string() } else { format!("t{id}") }));
        words.resize(128, "0".into());
        json!({"id":format!("teacher-{id}"),"quick_max_tokens":12,"request":{"messages":[{"role":"user","content":words.join(" ")}],"enable_thinking":false,"max_tokens":12,"loop_guard":false}})
    }).collect();
    let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
    for case in &cases {
        let (rendered, _) = formatter
            .prompt(
                &case["request"]["messages"],
                None,
                &json!({"enable_thinking":false}),
            )
            .unwrap();
        assert_eq!(rendered.len(), 129);
        assert_eq!(&rendered[1..1 + teacher.len()], &teacher);
    }
    folder.json("prompts.json", json!(cases));
    let path = folder.0.join("teacher-report.json");
    let child = Command::new(example_binary())
        .args(["--profile", "full", "--ctx", "512"])
        .arg("--model-dir")
        .arg(&folder.0)
        .arg("--prompts")
        .arg(folder.0.join("prompts.json"))
        .arg("--report")
        .arg(&path)
        .env("YSKN_BACKEND", "cpu")
        .env_remove("YSKN_LOSSLESS_INJECT_MISMATCH")
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{e}: status={} stderr={}",
            child.status,
            String::from_utf8_lossy(&child.stderr)
        )
    }))
    .unwrap();
    assert!(
        child.status.success(),
        "{} failures={}",
        String::from_utf8_lossy(&child.stderr),
        report["failures"]
    );
    assert_eq!(report["passed"], true);
    let runs = report["runs"].as_array().unwrap();
    for name in CONFIG_NAMES {
        for sampler in lossless_support::SAMPLERS {
            let own: Vec<_> = runs
                .iter()
                .filter(|r| {
                    r["configuration"] == name && r["sampler"] == sampler && r["probe"] != true
                })
                .collect();
            assert_eq!(own.len(), 4);
            let sum = |key: &str| own.iter().map(|r| r[key].as_u64().unwrap()).sum::<u64>();
            if name != "none" {
                assert!(sum("guesses_offered") > 0);
                if sampler == "greedy" || sampler == "seed-42" {
                    assert!(sum("guesses_accepted") > 0);
                }
                if name == "both-k7-adaptive" {
                    assert!(sum("neural_guesses_offered") > 0 && sum("ngram_guesses_offered") > 0);
                }
            }
            eprintln!(
                "teacher {name}/{sampler}: offered={} accepted={}",
                sum("guesses_offered"),
                sum("guesses_accepted")
            );
        }
    }
}

#[test]
fn real_example_binary_saves_report_and_exits_nonzero_for_an_injected_mismatch() {
    let _run = MODEL_RUN.lock().unwrap();
    let folder = Folder::new();
    folder.write("chat_template.jinja", "{% if enable_thinking %}t13 {% endif %}{% for m in messages %}{% if m.role == 'assistant' %}t10 {% endif %}{{ m.content }} {% endfor %}t10");
    folder.json("prompts.json", json!([
        {"id":"a","request":{"messages":[{"role":"user","content":"1 2 1 2 1 2"}],"enable_thinking":false,"max_tokens":12}},
        {"id":"b","request":{"messages":[{"role":"user","content":"3 4 5 6"}],"enable_thinking":false,"max_tokens":12}}
    ]));
    let binary = example_binary();
    for profile in ["release", "full"] {
        for value in ["", "0", "1"] {
            let child = Command::new(&binary)
                .args(["--profile", profile])
                .env("YSKN_LOSSLESS_INJECT_MISMATCH", value)
                .output()
                .unwrap();
            assert!(!child.status.success());
            assert!(String::from_utf8_lossy(&child.stderr).contains("must be unset"));
        }
    }
    for inject in [false, true] {
        let report_path = folder.0.join(if inject { "bad.json" } else { "good.json" });
        let mut command = Command::new(&binary);
        command
            .args(["--profile", "quick", "--configs", "none", "--ctx", "512"])
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
        let report: Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap_or_else(|e| {
                panic!(
                    "{e}: status={} stderr={}",
                    child.status,
                    String::from_utf8_lossy(&child.stderr)
                )
            }))
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
