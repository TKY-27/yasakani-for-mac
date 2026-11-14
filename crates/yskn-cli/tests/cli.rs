use std::{
    io::{BufRead, BufReader, Read, Write},
    process::{Command, Output, Stdio},
    time::Duration,
};
const PATIENCE: Duration = Duration::from_secs(30);
fn command() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_yskn"));
    for key in [
        "YASAKANI_API_KEY",
        "YASAKANI_MODEL_DIR",
        "YASAKANI_THINKING_BUDGET",
    ] {
        c.env_remove(key);
    }
    c
}
fn run(args: &[&str], input: &str) -> Output {
    let mut child = command()
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
fn text(r: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    )
}
// Upstream test_cli.CliTests.test_help_and_version
#[test]
fn help_and_version() {
    let r = run(&["--version"], "");
    assert!(r.status.success());
    assert!(text(&r).contains("yskn"));
    let r = run(&["serve", "--engine", "stub", "--port", "0", "--help"], "");
    assert!(r.status.success());
    assert!(text(&r).contains("--port"));
    assert!(!run(&[], "").status.success());
}
// Upstream test_cli.CliTests.test_info (Rust/static stub equivalent; no DLL/weights).
#[test]
fn info() {
    let r = run(&["info", "--engine", "stub"], "");
    assert!(r.status.success(), "{}", text(&r));
    assert!(text(&r).contains("tokenizer  ok"));
    assert!(text(&r).contains("deterministic demo"));
    let r = run(&["info"], "");
    assert!(!r.status.success());
    assert!(text(&r).contains("real engine is not connected"));
    let r = run(
        &[
            "info",
            "--engine",
            "stub",
            "--model-dir",
            "/nonexistent/yskn-model",
        ],
        "",
    );
    assert!(!r.status.success());
    assert!(text(&r).contains("tokenizer.json"));
}
// Upstream test_cli.CliTests.test_chat
#[test]
fn chat() {
    let r = run(
        &["chat", "--engine", "stub", "--max-ctx", "4096"],
        "/think off\nhello\n/reset\n/bogus\n/exit\n",
    );
    assert!(r.status.success(), "{}", text(&r));
    for needle in [
        "(thinking off)",
        "tokens/s",
        "complete",
        "(conversation cleared)",
        "commands:",
    ] {
        assert!(text(&r).contains(needle), "{}", text(&r));
    }
}
// Upstream test_cli.CliTests.test_chat_ends_on_end_of_input
#[test]
fn chat_ends_on_end_of_input() {
    let r = run(
        &[
            "chat",
            "--engine",
            "stub",
            "--no-think",
            "--max-ctx",
            "4096",
        ],
        "hi\n",
    );
    assert!(r.status.success(), "{}", text(&r));
    assert!(text(&r).contains("complete"));
}
// Upstream test_cli.CliTests.test_bad_inputs (no real engine; remote-key rule).
#[test]
fn bad_inputs() {
    let r = run(&["chat"], "");
    assert!(!r.status.success());
    assert!(text(&r).contains("--engine stub"));
    let r = run(
        &[
            "serve", "--engine", "stub", "--host", "0.0.0.0", "--port", "0",
        ],
        "",
    );
    assert!(!r.status.success());
    assert!(text(&r).contains("without --api-key"));
    let r = run(
        &[
            "chat",
            "--engine",
            "stub",
            "--model-dir",
            "/nonexistent/yskn-model",
        ],
        "",
    );
    assert!(!r.status.success());
    assert!(text(&r).contains("tokenizer.json"));
    for args in [
        vec!["chat", "--engine", "stub", "--effort", "bad"],
        vec!["chat", "--engine", "stub", "--max-tokens", "0"],
        vec!["serve", "--engine", "stub", "--port", "65536"],
        vec!["chat", "--engine", "stub", "--max-ctx", "64"],
        vec!["chat", "--engine", "stub", "--unknown"],
        vec!["chat", "--engine"],
        vec!["serve", "--engine", "other"],
    ] {
        assert!(!run(&args, "").status.success(), "{args:?}");
    }
}
#[test]
fn recognized_unavailable_commands() {
    for cmd in ["bench", "selfcheck"] {
        let r = run(&[cmd], "");
        assert!(!r.status.success());
        assert!(text(&r).contains("not available yet"));
    }
}
#[test]
fn chat_thinking_visibility_effort_and_aliases() {
    let r = run(
        &["chat", "--engine", "stub", "--effort", "low"],
        "/effort medium\nhi\n/q\n",
    );
    assert!(r.status.success(), "{}", text(&r));
    for s in [
        "[thinking]",
        "[/thinking]",
        "(reasoning effort medium)",
        "Stub response",
    ] {
        assert!(text(&r).contains(s), "{}", text(&r));
    }
    let r = run(&["chat", "--engine", "stub", "--raw"], "hi\n/q\n");
    assert!(r.status.success(), "{}", text(&r));
    assert!(text(&r).contains("Stub response"));
    let r = run(
        &["chat", "--engine", "stub", "--hide-thinking"],
        "hi\n/quit\n",
    );
    assert!(r.status.success());
    assert!(!text(&r).contains("[thinking]"));
    assert!(text(&r).contains("Stub response"));
    let r = run(
        &["chat", "--engine", "stub", "--max-tokens", "2"],
        "hi\n/exit\n",
    );
    assert!(r.status.success());
    assert!(text(&r).contains("no answer was written"));
}
// Upstream test_cli.CliTests.test_serve_port_in_use
#[test]
fn serve_port_in_use() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    let r = run(&["serve", "--engine", "stub", "--port", &port], "");
    assert!(!r.status.success());
    assert!(text(&r).contains("cannot listen"));
}
struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn http(addr: &str, request: &str) -> String {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(PATIENCE)).unwrap();
    s.write_all(request.as_bytes()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}
// Upstream test_cli.CliTests.test_serve_end_to_end (no Python env/CUDA settings).
#[test]
fn serve_end_to_end() {
    let child = command()
        .args([
            "serve",
            "--engine",
            "stub",
            "--port",
            "0",
            "--max-ctx",
            "4096",
            "--api-key",
            "k3y",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut child = Child(child);
    let mut err = BufReader::new(child.0.stderr.take().unwrap());
    let mut line = String::new();
    err.read_line(&mut line).unwrap();
    assert!(line.starts_with("listening on http://"), "{line}");
    let addr = line
        .trim_start_matches("listening on http://")
        .split_whitespace()
        .next()
        .unwrap();
    // Another process: /health is the only readiness signal. Generous for a loaded machine.
    let deadline = std::time::Instant::now() + PATIENCE;
    while !http(
        addr,
        &format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"),
    )
    .contains("\"status\":\"ok\"")
    {
        assert!(
            std::time::Instant::now() < deadline,
            "server never became ready"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"enable_thinking\":false}";
    let r = http(
        addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer k3y\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ),
    );
    assert!(r.starts_with("HTTP/1.1 200"));
    assert!(r.contains("\"completion_tokens\":8"));
    assert!(r.contains("\"finish_reason\":\"length\""));
    assert!(
        http(
            addr,
            &format!("GET /v1/models HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
        )
        .starts_with("HTTP/1.1 401")
    );
    let status = Command::new("/bin/kill")
        .args(["-INT", &child.0.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    let mut rest = String::new();
    err.read_to_string(&mut rest).unwrap();
    assert!(child.0.wait().unwrap().success());
    assert!(rest.contains("stopping"));
    assert!(rest.contains("API key required"));
}
/// Waits for the child to exit; kills it and fails the test after PATIENCE.
fn exits_within_patience(child: &mut Child) -> std::process::ExitStatus {
    let deadline = std::time::Instant::now() + PATIENCE;
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            return status;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the process did not exit after the interrupt"
        );
        std::thread::sleep(Duration::from_millis(10)); // polling the exit only
    }
}
// Review finding: the terminal thread held the standard output lock while it
// waited for input, and the interrupt handler then blocked on that lock.
// Upstream exits on an interrupt at the prompt.
#[test]
fn interrupt_at_the_prompt_exits() {
    let child = command()
        .args(["chat", "--engine", "stub", "--max-ctx", "4096"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut child = Child(child);
    // Kept open: the chat waits at its prompt for a line that never comes.
    let _stdin = child.0.stdin.take().unwrap();
    let mut out = child.0.stdout.take().unwrap();
    let mut seen = Vec::new();
    let mut byte = [0_u8; 1];
    while !seen.ends_with(b"\n> ") {
        assert_eq!(out.read(&mut byte).unwrap(), 1, "no prompt appeared");
        seen.push(byte[0]);
    }
    let status = Command::new("/bin/kill")
        .args(["-INT", &child.0.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(exits_within_patience(&mut child).success());
}
// Review finding: errors returned outside the chat loop reached the terminal
// unescaped, for example a model folder path or an argument holding an escape
// sequence. Every top-level error is shown with visible control characters.
#[test]
fn top_level_errors_show_control_characters_visibly() {
    let folder = "/nonexistent/\u{1b}]0;owned\u{7}\u{1b}[2J\u{202e}x";
    for args in [
        vec!["info", "--engine", "stub", "--model-dir", folder],
        vec!["chat", "--engine", "stub", "--model-dir", folder],
        vec!["chat", "--engine", "stub", "--raw", "--model-dir", folder],
        vec!["info", "--engine", "stub", "--bad\u{1b}[2J"],
    ] {
        let r = run(&args, "");
        assert!(!r.status.success());
        let err = String::from_utf8(r.stderr).unwrap();
        assert!(err.starts_with("error: "), "{err:?}");
        assert!(err.contains("\\x1b[2J"), "{err:?}");
        assert!(
            !err.contains(|c: char| c.is_control() && c != '\n' || c == '\u{202e}'),
            "{err:?}"
        );
    }
}
