mod product_support;

use product_support::{Folder, greedy, sampled, scalar};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::process::Command;
use yskn::{
    engine::{Engine, Finish, Generation, Sampling},
    real_engine::DraftMode,
};

fn modes() -> [DraftMode; 4] {
    [
        DraftMode::None,
        DraftMode::Ngram,
        DraftMode::Neural,
        DraftMode::Both,
    ]
}
fn run(engine: &mut dyn Engine, prompt: &[u32], n: usize, sampling: &Sampling) -> Generation {
    engine
        .generate(prompt, n, sampling, None, &mut |_| Ok(true))
        .unwrap()
}

#[test]
fn real_product_matches_independent_scalar_loop_for_greedy_and_seeded_sampling_in_every_draft_mode()
{
    let folder = Folder::new();
    let prompt = [1, 2, 1, 2, 1, 2, 3];
    for sampling in [greedy(), sampled()] {
        let expected = scalar(&folder, &prompt, 7, &sampling);
        for mode in modes() {
            let mut engine = folder.engine(mode);
            let result = run(&mut engine, &prompt, 7, &sampling);
            assert_eq!(
                result.ids, expected,
                "draft={mode:?}, sampling={sampling:?}"
            );
            assert_eq!(result.finish, Finish::Length);
            assert_eq!(result.cached_tokens, 0);
            assert!(result.passes > 0);
            assert!(result.rows_verified >= result.passes);
            assert!(result.guesses_accepted <= result.guesses_offered);
        }
    }
}

#[test]
fn committed_prefix_snapshot_and_reset_reuse_preserve_scalar_tokens_and_cache_counts() {
    let folder = Folder::new();
    let first = [1, 2, 3, 4];
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let initial = run(&mut engine, &first, 2, &greedy());
        let mut continued = first.to_vec();
        continued.extend(&initial.ids);
        continued.push(5);
        let result = run(&mut engine, &continued, 2, &greedy());
        assert_eq!(
            result.ids,
            scalar(&folder, &continued, 2, &greedy()),
            "{mode:?}"
        );
        // The final returned token is pending; every earlier input is committed.
        assert_eq!(
            result.cached_tokens,
            first.len() + initial.ids.len() - 1,
            "{mode:?}"
        );
        assert_eq!(result.snap_mode, 1);
        // The new suffix differs after the snapshot saved before the final
        // prompt token; it cannot reuse the complete committed history.
        let mut branch = continued[..continued.len() - 1].to_vec();
        branch.push(6);
        let result = run(&mut engine, &branch, 2, &greedy());
        assert_eq!(
            result.ids,
            scalar(&folder, &branch, 2, &greedy()),
            "{mode:?}"
        );
        assert_eq!(result.cached_tokens, continued.len() - 1, "{mode:?}");
        assert_eq!(result.snap_mode, 2);
        let unrelated = [7, 8, 9];
        let result = run(&mut engine, &unrelated, 2, &greedy());
        assert_eq!(
            result.ids,
            scalar(&folder, &unrelated, 2, &greedy()),
            "{mode:?}"
        );
        assert_eq!((result.cached_tokens, result.snap_mode), (0, 0));
    }
}

#[test]
fn fifty_requests_keep_one_bounded_session_and_snapshot_for_every_draft_mode() {
    let folder = Folder::new();
    let prompt = [1, 2, 3, 4];
    let expected = scalar(&folder, &prompt, 2, &greedy());
    for mode in modes() {
        let mut engine = folder.engine(mode);
        for _ in 0..3 {
            run(&mut engine, &prompt, 2, &greedy());
        }
        let session = engine.session_memory();
        let snapshot = engine.snapshot_bytes();
        assert!(snapshot > 0);
        for request in 0..50 {
            let result = run(&mut engine, &prompt, 2, &greedy());
            assert_eq!(result.ids, expected, "{mode:?}, request {request}");
            assert_eq!(
                (result.cached_tokens, result.snap_mode),
                (prompt.len() - 1, 2)
            );
            assert_eq!(
                engine.session_memory(),
                session,
                "{mode:?}, request {request}"
            );
            assert_eq!(
                engine.snapshot_bytes(),
                snapshot,
                "{mode:?}, request {request}"
            );
        }
    }
}

#[test]
fn cancellation_and_callback_errors_quiesce_prefill_and_generation_then_allow_next_request() {
    let folder = Folder::new();
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let mut empty_callbacks = 0;
        let result = engine
            .generate(&[1; 12], 8, &greedy(), None, &mut |ids| {
                assert!(ids.is_empty(), "cancellation must happen during prefill");
                empty_callbacks += 1;
                Ok(empty_callbacks < 2)
            })
            .unwrap();
        assert_eq!(result.finish, Finish::Cancelled);
        assert!(result.ids.is_empty());
        assert_eq!(result.passes, 0);
        assert_eq!(empty_callbacks, 2);
        assert_eq!(
            run(&mut engine, &[2, 3], 2, &greedy()).ids,
            scalar(&folder, &[2, 3], 2, &greedy())
        );
        engine.reset().unwrap();
        let result = engine
            .generate(&[1, 2], 8, &greedy(), None, &mut |ids| Ok(ids.is_empty()))
            .unwrap();
        assert_eq!(result.finish, Finish::Cancelled);
        assert_eq!(result.ids.len(), 1);
        assert_eq!(
            run(&mut engine, &[2, 3], 2, &greedy()).ids,
            scalar(&folder, &[2, 3], 2, &greedy())
        );
        engine.reset().unwrap();
        let failure = engine
            .generate(&[1, 2], 8, &greedy(), None, &mut |ids| {
                if ids.is_empty() {
                    Ok(true)
                } else {
                    Err(yskn::Error::engine("callback-sentinel"))
                }
            })
            .unwrap_err();
        assert!(failure.to_string().contains("callback-sentinel"));
        assert_eq!(
            run(&mut engine, &[2, 3], 2, &greedy()).ids,
            scalar(&folder, &[2, 3], 2, &greedy())
        );
    }
}

#[test]
fn stop_id_is_included_and_stops_at_the_first_matching_prediction() {
    let folder = Folder::new();
    let expected = scalar(&folder, &[1, 2], 3, &greedy());
    for mode in modes() {
        let mut engine = folder.engine(mode);
        let result = engine
            .generate(&[1, 2], 8, &greedy(), Some(&[expected[0]]), &mut |_| {
                Ok(true)
            })
            .unwrap();
        assert_eq!(result.ids, expected[..1]);
        assert_eq!(result.finish, Finish::Stop);
    }
}

#[test]
fn memory_check_reports_needed_available_and_configuration_remedies() {
    let folder = Folder::new();
    let info = folder.config(DraftMode::Both).inspect().unwrap();
    let error = info.memory.enforce(1).unwrap_err().to_string();
    for expected in [
        "need",
        "physical memory is 1 bytes",
        "--ctx",
        "--kv",
        "--draft",
    ] {
        assert!(error.contains(expected), "missing {expected:?}: {error}");
    }
    assert!(
        error.contains(&info.memory.needed_bytes.to_string()),
        "{error}"
    );
}

#[test]
fn requested_context_cannot_exceed_unrounded_model_limit() {
    let folder = Folder::new();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(folder.0.join("yasakani.json")).unwrap()).unwrap();
    manifest["max_ctx"] = json!(97);
    folder.json("yasakani.json", manifest);
    let mut config = folder.config(DraftMode::None);
    config.context = 100;
    let error = config.inspect().unwrap_err().to_string();
    assert!(
        error.contains("100") && error.contains("97") && error.contains("max_ctx"),
        "{error}"
    );
}

#[test]
fn model_generation_defaults_reach_request_sampling_without_overriding_explicit_values() {
    let folder = Folder::new();
    let sampling = folder.config(DraftMode::None).sampling_defaults().unwrap();
    assert_eq!(
        sampling,
        Sampling {
            temperature: 0.75,
            top_p: 0.9,
            top_k: 11,
            min_p: 0.05,
            seed: 0
        }
    );
    let config = yskn::server::Config {
        defaults: yskn::request::RequestDefaults {
            sampling,
            ..Default::default()
        },
        ..Default::default()
    };
    let messages = json!([{"role":"user","content":"t1"}]);
    let defaults =
        yskn::request::parse_chat_request(&json!({"messages":messages}), &config.defaults).unwrap();
    assert_eq!(
        (
            defaults.temperature,
            defaults.top_p,
            defaults.top_k,
            defaults.min_p,
            defaults.seed
        ),
        (0.75, 0.9, 11, 0.05, None)
    );
    let explicit = yskn::request::parse_chat_request(
        &json!({"messages":messages,"temperature":0.5,"top_p":0.8,"top_k":3,"min_p":0.1,"seed":99}),
        &config.defaults,
    )
    .unwrap();
    assert_eq!(
        (
            explicit.temperature,
            explicit.top_p,
            explicit.top_k,
            explicit.min_p,
            explicit.seed
        ),
        (0.5, 0.8, 3, 0.1, Some(99))
    );
}

#[test]
fn synthetic_tokenizer_streams_and_decodes_every_model_id_the_same_on_every_load() {
    // The socket tests compare server text with a separately loaded decoder;
    // that holds only if the tokenizer, like the real one, is deterministic.
    let folder = Folder::new();
    let ids: Vec<u32> = (0..32).collect();
    let first = yskn_text::ChatFormatter::load(&folder.0)
        .unwrap()
        .decode(&ids)
        .unwrap();
    for _ in 0..4 {
        let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
        assert_eq!(formatter.decode(&ids).unwrap(), first);
        let mut stream = yskn_text::OutputStream::new(&formatter, false);
        let mut text = String::new();
        for n in 1..=ids.len() {
            for (kind, piece) in stream.feed(&ids[..n], n == ids.len()).unwrap() {
                assert_eq!(kind, "content");
                text += &piece;
            }
        }
        assert_eq!(text, first);
    }
}

fn digest(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in ids {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn selfcheck_reference(folder: &Folder) -> Value {
    let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
    let prompts = [
        ("short-answer",formatter.prompt(&json!([{"role":"user","content":"Say hello in one short sentence."}]),None,&json!({"enable_thinking":false})).unwrap().0,48),
        ("thinking",formatter.prompt(&json!([{"role":"user","content":"What is 17 times 23? Think it through step by step."}]),None,&json!({})).unwrap().0,160),
        ("plain-text",formatter.encode("The capital of France is").unwrap(),32),
    ];
    let cases: Vec<_> = prompts.into_iter().map(|(name,prompt,max_new)| {
        let ids = scalar(folder,&prompt,max_new,&greedy());
        json!({"name":name,"prompt_tokens":prompt.len(),"prompt_sha256":digest(&prompt),"tokens":ids.len(),"sha256":digest(&ids),"head":&ids[..ids.len().min(12)],"ids":ids})
    }).collect();
    json!({"format":1,"cases":cases})
}
fn selfcheck(folder: &Folder) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_yskn"))
        .args(["selfcheck", "--model-dir"])
        .arg(&folder.0)
        .args(["--backend", "cpu", "--ctx", "512", "--draft", "none"])
        .env_remove("YASAKANI_API_KEY")
        .env_remove("YASAKANI_MODEL_DIR")
        .output()
        .unwrap()
}

fn product_command(folder: &Folder, subcommand: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_yskn"));
    command
        .args([subcommand, "--model-dir"])
        .arg(&folder.0)
        .args(["--backend", "cpu", "--ctx", "512", "--threads", "1"]);
    for key in [
        "YASAKANI_API_KEY",
        "YASAKANI_MODEL_DIR",
        "YASAKANI_THINKING_BUDGET",
    ] {
        command.env_remove(key);
    }
    command
}

#[test]
fn synthetic_product_cli_info_bench_comparison_json_and_terminal_chat_work() {
    use std::io::Write;
    use std::process::Stdio;
    let folder = Folder::new();
    let info = product_command(&folder, "info")
        .args(["--draft", "none"])
        .output()
        .unwrap();
    assert!(
        info.status.success(),
        "{}",
        String::from_utf8_lossy(&info.stderr)
    );
    let facts = serde_json::Deserializer::from_slice(&info.stdout)
        .into_iter::<Value>()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(facts["layers"], 2);
    assert_eq!(facts["hidden_width"], 512);
    assert_eq!(facts["context"], 512);
    assert_eq!(facts["kv"], "kv5");
    assert_eq!(facts["backend"], "cpu");
    assert!(facts["memory"]["needed_bytes"].as_u64().unwrap() > 0);
    assert!(facts["memory"]["physical_bytes"].as_u64().unwrap() > 0);
    let output_path = folder.0.join("bench.json");
    let bench = product_command(&folder, "bench")
        .args([
            "--n", "2", "--draft", "none", "--draft", "ngram", "--k", "1", "--k", "3", "--json",
        ])
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        bench.status.success(),
        "{}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let report: Value = serde_json::from_slice(&std::fs::read(output_path).unwrap()).unwrap();
    assert_eq!(report["tokens_per_task"], 2);
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 16);
    let mut seen = std::collections::BTreeSet::new();
    for run in runs {
        let config = &run["configuration"];
        let metrics = &run["metrics"];
        seen.insert((
            config["draft"].as_str().unwrap(),
            config["k"].as_u64().unwrap(),
            metrics["task"].as_str().unwrap(),
        ));
        assert_eq!(config["context"], 512);
        assert_eq!(config["backend"], "cpu");
        assert_eq!(config["threads"], 1);
        assert_eq!(config["kv"], "kv5");
        assert_eq!(metrics["generated_tokens"], 2);
        for name in [
            "prompt_tokens",
            "prefill_ms",
            "prefill_tokens_per_s",
            "passes",
            "rows_verified",
            "generation_ms",
            "milliseconds_per_pass",
            "tokens_confirmed_per_pass",
            "tokens_per_s",
            "bytes_read_per_pass",
        ] {
            assert!(
                metrics[name]
                    .as_f64()
                    .is_some_and(|n| n.is_finite() && n > 0.0),
                "{name}: {run}"
            );
        }
        assert_eq!(
            metrics["bytes_read_per_pass"].as_f64().unwrap(),
            metrics["main_code_bytes_per_pass"].as_f64().unwrap()
        );
        assert_eq!(metrics["neural_proposals"], 0);
        assert!(
            metrics["guesses_accepted"].as_u64().unwrap()
                <= metrics["guesses_offered"].as_u64().unwrap()
        );
    }
    assert_eq!(seen.len(), 16);
    let mut chat = product_command(&folder, "chat")
        .args([
            "--draft",
            "ngram",
            "--k",
            "3",
            "--max-tokens",
            "2",
            "--no-think",
            "--temperature",
            "0",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    chat.stdin.take().unwrap().write_all(b"1 2\n").unwrap();
    let chat = chat.wait_with_output().unwrap();
    assert!(
        chat.status.success(),
        "{}",
        String::from_utf8_lossy(&chat.stderr)
    );
    let text = String::from_utf8(chat.stdout).unwrap();
    assert!(text.contains("[2 tokens,"), "{text}");
    assert!(text.contains("tokens/s"), "{text}");
}
#[test]
fn selfcheck_cli_accepts_scalar_reference_and_reports_mismatching_token_ids() {
    let folder = Folder::new();
    let mut reference = selfcheck_reference(&folder);
    folder.json("selfcheck.json", reference.clone());
    let output = selfcheck(&folder);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{text}");
    for name in ["short-answer", "thinking", "plain-text"] {
        assert!(text.contains(name), "{text}");
    }
    assert_eq!(text.matches("PASS").count(), 3, "{text}");
    let got = reference["cases"][0]["ids"][0].as_u64().unwrap();
    let want = (got + 1) % 32;
    reference["cases"][0]["ids"][0] = json!(want);
    reference["cases"][0]["head"][0] = json!(want);
    reference["cases"][0]["sha256"] = json!("0".repeat(64));
    folder.json("selfcheck.json", reference);
    let output = selfcheck(&folder);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("FAIL"), "{text}");
    assert!(
        text.contains(&format!(
            "first different token #0 (zero-based): expected {want}, got {got}"
        )),
        "{text}"
    );
}

mod sockets {
    use super::*;
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::{Request, client::conn::http1};
    use hyper_util::rt::TokioIo;
    use std::{
        net::SocketAddr,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        time::Duration,
    };
    use tokio::{io::AsyncWriteExt, net::TcpStream, sync::oneshot};
    use yskn::{
        chat::ChatRunner,
        real_engine::RealEngine,
        server::{Config, Server},
    };

    const PATIENCE: Duration = Duration::from_secs(30);
    #[derive(Default)]
    struct Observation {
        prefill: AtomicUsize,
        cancelled: AtomicBool,
    }
    struct Observed {
        inner: RealEngine,
        observation: Arc<Observation>,
    }
    impl Engine for Observed {
        fn context_capacity(&self) -> usize {
            self.inner.context_capacity()
        }
        fn reset(&mut self) -> yskn::Result<()> {
            self.inner.reset()
        }
        fn generate(
            &mut self,
            p: &[u32],
            n: usize,
            s: &Sampling,
            stop: Option<&[u32]>,
            cb: &mut dyn FnMut(&[u32]) -> yskn::Result<bool>,
        ) -> yskn::Result<Generation> {
            let result = self.inner.generate(p, n, s, stop, &mut |ids| {
                if ids.is_empty() {
                    self.observation.prefill.fetch_add(1, Ordering::Release);
                }
                cb(ids)
            })?;
            if result.finish == Finish::Cancelled {
                self.observation.cancelled.store(true, Ordering::Release);
            }
            Ok(result)
        }
    }
    async fn request(addr: SocketAddr, body: Value) -> (u16, String) {
        tokio::time::timeout(PATIENCE, async {
            let stream = TcpStream::connect(addr).await.unwrap();
            let (mut sender, conn) = http1::handshake(TokioIo::new(stream)).await.unwrap();
            tokio::spawn(async move {
                let _ = conn.await;
            });
            let response = sender
                .send_request(
                    Request::post("/v1/chat/completions")
                        .header("host", addr.to_string())
                        .header("connection", "close")
                        .body(Full::new(Bytes::from(body.to_string())))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status().as_u16();
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(bytes.to_vec()).unwrap())
        })
        .await
        .expect("real-engine request timed out")
    }
    async fn start(
        folder: &Folder,
        observation: Arc<Observation>,
    ) -> (
        SocketAddr,
        oneshot::Sender<()>,
        tokio::task::JoinHandle<yskn::Result<()>>,
    ) {
        let config = Config {
            port: 0,
            defaults: yskn::request::RequestDefaults {
                thinking_budget: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine_config = folder.config(DraftMode::None);
        engine_config.prefill_width = 1;
        let path = folder.0.clone();
        let server = Server::bind(config, move || {
            ChatRunner::new(
                Box::new(Observed {
                    inner: RealEngine::load(engine_config)?,
                    observation,
                }),
                yskn_text::ChatFormatter::load(path)?,
                1,
            )
        })
        .await
        .unwrap();
        let deadline = tokio::time::Instant::now() + PATIENCE;
        while !server.is_ready() {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let addr = server.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        let task = tokio::spawn(server.run(async {
            let _ = rx.await;
        }));
        (addr, tx, task)
    }
    #[tokio::test]
    async fn real_socket_streaming_and_nonstreaming_use_scalar_tokens_and_exact_usage() {
        let folder = Folder::new();
        let formatter = yskn_text::ChatFormatter::load(&folder.0).unwrap();
        let messages = json!([{"role":"user","content":"1 2"}]);
        let prompt = formatter
            .prompt(&messages, None, &json!({"enable_thinking":false}))
            .unwrap()
            .0;
        let expected_ids = scalar(&folder, &prompt, 4, &greedy());
        let expected_text = formatter.decode(&expected_ids).unwrap();
        let (addr, shutdown, task) = start(&folder, Arc::default()).await;
        let body =
            json!({"messages":messages,"temperature":0,"max_tokens":4,"enable_thinking":false});
        let (status, text) = request(addr, body.clone()).await;
        assert_eq!(status, 200, "{text}");
        let response: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(response["choices"][0]["message"]["content"], expected_text);
        assert_eq!(response["usage"]["prompt_tokens"], prompt.len());
        assert_eq!(response["usage"]["completion_tokens"], 4);
        assert_eq!(response["usage"]["total_tokens"], prompt.len() + 4);
        let mut stream_body = body;
        stream_body["stream"] = json!(true);
        stream_body["stream_options"] = json!({"include_usage":true});
        let (status, text) = request(addr, stream_body).await;
        assert_eq!(status, 200, "{text}");
        assert!(text.contains("data: [DONE]"), "{text}");
        let chunks: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let generated: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(generated, expected_text);
        let usage = chunks
            .iter()
            .find_map(|chunk| chunk.get("usage").filter(|v| v.is_object()))
            .expect("streamed usage");
        assert_eq!(usage["prompt_tokens"], prompt.len());
        assert_eq!(usage["completion_tokens"], 4);
        assert_eq!(usage["total_tokens"], prompt.len() + 4);
        shutdown.send(()).unwrap();
        task.await.unwrap().unwrap();
    }
    #[tokio::test]
    async fn real_socket_disconnect_during_prefill_cancels_and_releases_engine_for_next_chat() {
        let folder = Folder::new();
        let observation = Arc::new(Observation::default());
        let (addr, shutdown, task) = start(&folder, observation.clone()).await;
        let body=json!({"messages":[{"role":"user","content":"1 ".repeat(256)}],"stream":true,"max_tokens":4,"temperature":0,"enable_thinking":false}).to_string();
        let mut client = TcpStream::connect(addr).await.unwrap();
        client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
        let deadline = tokio::time::Instant::now() + PATIENCE;
        while observation.prefill.load(Ordering::Acquire) < 2 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "prefill did not start"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        drop(client);
        let (status,text)=request(addr,json!({"messages":[{"role":"user","content":"2"}],"max_tokens":2,"temperature":0,"enable_thinking":false})).await;
        assert_eq!(status, 200, "next request did not acquire engine: {text}");
        assert!(
            observation.cancelled.load(Ordering::Acquire),
            "long prefill completed instead of cancelling"
        );
        assert!(
            observation.prefill.load(Ordering::Acquire) < 256,
            "disconnect was not observed between prefill passes"
        );
        let response: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(response["usage"]["completion_tokens"], 2);
        shutdown.send(()).unwrap();
        task.await.unwrap().unwrap();
    }
}
