//! Fixed product probes. Hashes use upstream's little-endian int32 token bytes.
use super::Args;
use crate::{
    Error, Result,
    engine::{Engine, Sampling},
    real_engine::RealEngine,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn greedy() -> Sampling {
    Sampling {
        temperature: 0.0,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.0,
        seed: 0,
    }
}
fn ids_sha(ids: &[u32]) -> String {
    let mut digest = Sha256::new();
    for id in ids {
        digest.update(id.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub(super) fn selfcheck(args: &Args) -> Result<()> {
    if !args.is_stub() {
        eprintln!("{}", crate::real_engine::MODEL_FILE_STABILITY);
    }
    let path = args
        .value("reference")
        .map(std::path::PathBuf::from)
        .unwrap_or(args.model_dir()?.join("selfcheck.json"));
    let reference: Value = serde_json::from_slice(&std::fs::read(&path).map_err(Error::engine)?)
        .map_err(Error::engine)?;
    if reference["format"] != json!(1) {
        return Err(Error::request("selfcheck reference has an unknown format"));
    }
    let expected = reference["cases"]
        .as_array()
        .ok_or_else(|| Error::request("selfcheck reference must contain cases"))?;
    let mut runner = args.runner()?;
    let mut bad = 0;
    // python/yasakani/cli.py:403-430: exact fixed text, templates, lengths and hash ABI.
    for (name, text, options, max_new) in [
        (
            "short-answer",
            "Say hello in one short sentence.",
            Some(json!({"enable_thinking":false})),
            48,
        ),
        (
            "thinking",
            "What is 17 times 23? Think it through step by step.",
            Some(json!({})),
            160,
        ),
        ("plain-text", "The capital of France is", None, 32),
    ] {
        let prompt = if let Some(options) = options {
            runner
                .formatter
                .prompt(&json!([{"role":"user","content":text}]), None, &options)?
                .0
        } else {
            runner.formatter.encode(text)?
        };
        runner.engine.reset()?;
        let result = runner.engine.generate(
            &prompt,
            max_new,
            &greedy(),
            Some(&[248046, 248044]),
            &mut |_| Ok(true),
        )?;
        let want = expected.iter().find(|c| c["name"] == name);
        let Some(want) = want else {
            println!("{name:<14} FAIL  not in the reference file");
            bad += 1;
            continue;
        };
        let prompt_ok = want["prompt_tokens"] == json!(prompt.len())
            && want["prompt_sha256"] == ids_sha(&prompt);
        let output_ok = want["tokens"] == json!(result.ids.len())
            && want["sha256"] == ids_sha(&result.ids)
            && want["head"] == json!(&result.ids[..result.ids.len().min(12)]);
        if !prompt_ok {
            println!("{name:<14} FAIL  prompt count/hash differs (tokenizer or template)");
            bad += 1;
        } else if !output_ok {
            // Published records contain only a digest and twelve IDs. An optional
            // full `ids` list lets publishers diagnose differences beyond that head.
            let full = want["ids"].as_array();
            let known = full.or_else(|| want["head"].as_array());
            let difference = known.and_then(|ids| {
                ids.iter()
                    .enumerate()
                    .find_map(|(i, id)| {
                        if result.ids.get(i).copied().map(u64::from) != id.as_u64() {
                            Some((
                                i,
                                id.to_string(),
                                result
                                    .ids
                                    .get(i)
                                    .map_or_else(|| "<end>".into(), u32::to_string),
                            ))
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        if full.is_some() && result.ids.len() > ids.len() {
                            Some((ids.len(), "<end>".into(), result.ids[ids.len()].to_string()))
                        } else {
                            None
                        }
                    })
            });
            if let Some((i, expected, actual)) = difference {
                println!(
                    "{name:<14} FAIL  first different token #{i} (zero-based): expected {expected}, got {actual}"
                );
            } else {
                println!(
                    "{name:<14} FAIL  output count/hash differs; first differing index and expected ID unavailable (reference stores only a hash and head)"
                );
            }
            bad += 1;
        } else {
            println!("{name:<14} PASS  ({} tokens)", result.ids.len());
        }
    }
    runner.engine.reset()?;
    if bad != 0 {
        return Err(Error::engine(format!("selfcheck FAILED for {bad} case(s)")));
    }
    Ok(())
}

pub(super) fn bench(args: &Args) -> Result<()> {
    if !args.is_stub() {
        eprintln!("{}", crate::real_engine::MODEL_FILE_STABILITY);
    }
    if args.is_stub() {
        return Err(Error::request("bench requires the real engine"));
    }
    let tokens = args.parse("n", 512_usize)?;
    if tokens == 0 {
        return Err(Error::request("--n must be greater than zero"));
    }
    let drafts = args
        .comparisons
        .get("draft")
        .cloned()
        .unwrap_or_else(|| vec!["ngram".into()]);
    let widths = args
        .comparisons
        .get("k")
        .cloned()
        .unwrap_or_else(|| vec!["3".into()]);
    let fmt = yskn_text::ChatFormatter::load(args.model_dir()?)?;
    let mut runs = Vec::new();
    for draft in drafts {
        for k in &widths {
            let mut config_args = args.clone();
            config_args.values.insert("draft".into(), draft.clone());
            config_args.values.insert("k".into(), k.clone());
            let config = config_args.real_config()?;
            let mut engine = RealEngine::load(config)?;
            let info = engine.info().to_json();
            // Upstream cli.py:339-344 supplies prose/code/proof. CSV conversion
            // adds the structured-output category required by the product brief.
            for (task, text, thinking) in [
                (
                    "prose",
                    "Explain how a hash table works, including collisions and resizing, with a short example.",
                    false,
                ),
                (
                    "code",
                    "Write a Python function that merges overlapping intervals and add a few tests for it.",
                    false,
                ),
                (
                    "structured",
                    "Convert this CSV to a JSON array, with numeric ages: name,age\nAda,36\nLin,28",
                    false,
                ),
                (
                    "thinking",
                    "Prove that the square root of 2 is irrational.",
                    true,
                ),
            ] {
                engine.reset()?;
                let prompt = fmt
                    .prompt(
                        &json!([{"role":"user","content":text}]),
                        None,
                        &json!({"enable_thinking":thinking}),
                    )?
                    .0;
                // A fixed token count is a measurement contract: fail rather than
                // silently benchmarking a smaller output because context is full.
                if prompt
                    .len()
                    .saturating_add(tokens)
                    .saturating_add(crate::engine::CONTEXT_MARGIN)
                    > engine.context_capacity()
                {
                    return Err(Error::request(format!(
                        "bench {task}: --ctx cannot fit prompt plus --n tokens and 64-token reserve"
                    )));
                }
                let generated =
                    engine.generate(&prompt, tokens, &greedy(), None, &mut |_| Ok(true))?;
                if generated.ids.len() != tokens {
                    return Err(Error::engine(format!(
                        "bench {task}: expected {tokens} tokens, generated {}",
                        generated.ids.len()
                    )));
                }
                let passes = generated.passes;
                // The shared loop consumes the final pending prompt row in its
                // first verification pass; prefill timing covers earlier rows.
                let prefill_tokens = prompt.len() - generated.cached_tokens - 1;
                let divide = |n: f64, d: f64| if d > 0.0 { n / d } else { 0.0 };
                let main_bytes = engine.info().code_bytes_per_pass;
                // Count actual positive-budget neural calls; n-gram-first
                // combination may never invoke the neural proposer.
                let neural_passes = engine.last_neural_proposals();
                let draft_bytes = engine.info().draft_code_bytes_per_pass;
                let row = json!({
                    "task":task, "prompt_tokens":prompt.len(), "prefill_ms":generated.prefill_ms,
                    "prefill_tokens":prefill_tokens,
                    "prefill_tokens_per_s":divide(prefill_tokens as f64 * 1000.0, generated.prefill_ms),
                    "generated_tokens":generated.ids.len(), "passes":passes,
                    "rows_verified":generated.rows_verified,
                    "generation_ms":generated.gen_ms,
                    "milliseconds_per_pass":divide(generated.gen_ms, passes as f64),
                    "tokens_confirmed_per_pass":divide(generated.ids.len() as f64, passes as f64),
                    "tokens_per_s":divide(generated.ids.len() as f64 * 1000.0, generated.gen_ms),
                    "guesses_offered":generated.guesses_offered, "guesses_accepted":generated.guesses_accepted,
                    "main_code_bytes_per_pass":main_bytes, "draft_code_bytes_per_proposal":draft_bytes,
                    "neural_proposals":neural_passes,
                    "bytes_read_per_pass":main_bytes as f64 + divide(draft_bytes as f64 * neural_passes as f64, passes as f64),
                });
                println!(
                    "{task:<10} draft={draft:<6} K={k} prompt={} prefill={:.1} tokens/s generated={} passes={} ms/pass={:.3} tokens/pass={:.3} tokens/s={:.3} guesses={}/{} bytes/pass={:.0}",
                    prompt.len(),
                    row["prefill_tokens_per_s"].as_f64().unwrap_or_default(),
                    generated.ids.len(),
                    passes,
                    row["milliseconds_per_pass"].as_f64().unwrap_or_default(),
                    row["tokens_confirmed_per_pass"]
                        .as_f64()
                        .unwrap_or_default(),
                    row["tokens_per_s"].as_f64().unwrap_or_default(),
                    generated.guesses_accepted,
                    generated.guesses_offered,
                    row["bytes_read_per_pass"].as_f64().unwrap_or_default()
                );
                runs.push(json!({"configuration":info, "metrics":row}));
            }
        }
    }
    if let Some(path) = args.value("json") {
        let output = json!({"version":env!("CARGO_PKG_VERSION"), "tokens_per_task":tokens, "runs":runs,
            "bytes_read_definition":"Main body/head compressed streams per verification pass plus actual neural calls times (draft compressed streams + seven * (vocabulary*rank + shortlist*hidden) int8 head bytes), divided by passes; embeddings, vectors, tables, scales and cache excluded; not a DRAM counter."});
        let mut text = serde_json::to_string_pretty(&output).map_err(Error::engine)?;
        text.push('\n');
        std::fs::write(path, text).map_err(Error::engine)?;
    }
    Ok(())
}
