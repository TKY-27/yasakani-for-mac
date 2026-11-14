use crate::{
    Error, Result,
    chat::{ChatParams, ChatRunner, Event},
    request::{RequestDefaults, parse_chat_request},
    server::{Config, Server},
    stub::{self, StubEngine},
};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub const CHAT_HELP: &str = "commands:  /exit  quit      /reset  forget the conversation      /think on|off  thinking on or off (default on)\n           /effort low|medium|xhigh  how long the model thinks      /help  this text      Ctrl+C while it writes: stop the answer";
const HELP: &str = "yskn - local Rust chat product (real engine not connected yet)\ncommands: serve, chat, info, bench, selfcheck\n  --engine stub       deterministic demo engine; required until integration\n  --model-dir FOLDER  optional real tokenizer/template, read only\n  --max-ctx N         context capacity (default 262144)\n  --seed0 N           first automatic request seed (default 1)\nserve:\n  --host HOST         default 127.0.0.1\n  --port PORT         default 8765; 0 chooses an ephemeral port\n  --api-key KEY       or YASAKANI_API_KEY; required for remote binds\n  --default-max-tokens N  default 32768\n  --max-tokens-cap N  request upper limit\n  --no-warmup\nchat:\n  --system TEXT --max-tokens N --temperature N --top-p N --top-k N\n  --seed N --no-think --effort low|medium|xhigh --hide-thinking --no-loop-guard\n  --raw  print control characters in model text and errors unescaped\n  /help shows terminal commands\nYASAKANI_THINKING_BUDGET sets the server default (0 = unlimited).\nbench and selfcheck: not available yet\n  --help, -h, --version";
struct Args {
    command: String,
    values: HashMap<String, String>,
    flags: HashSet<String>,
}
impl Args {
    fn value(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
    fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }
    fn parse<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        self.value(key).map_or(Ok(default), |s| {
            s.parse()
                .map_err(|_| Error::request(format!("--{key} has an invalid value")))
        })
    }
    fn capacity(&self) -> Result<usize> {
        let n = self.parse("max-ctx", 262144_usize)?;
        if n <= 65 {
            return Err(Error::request("--max-ctx must be greater than 65"));
        }
        Ok(n)
    }
    fn runner(&self) -> Result<ChatRunner> {
        let fmt = if let Some(path) = self.value("model-dir") {
            yskn_text::ChatFormatter::load(path)?
        } else {
            stub::formatter()?
        };
        ChatRunner::new(
            Box::new(StubEngine::new(self.capacity()?).using_formatter(fmt.clone())),
            fmt,
            self.parse("seed0", 1)?,
        )
    }
}
fn parse_args(argv: Vec<String>) -> Result<Args> {
    let mut it = argv.into_iter();
    let command = it.next().ok_or_else(|| Error::request(HELP))?;
    if !matches!(
        command.as_str(),
        "serve" | "chat" | "info" | "bench" | "selfcheck"
    ) {
        return Err(Error::request(format!(
            "unknown command: {command}\n{HELP}"
        )));
    }
    let common = ["engine", "model-dir", "max-ctx", "seed0"];
    let serve = [
        "host",
        "port",
        "api-key",
        "default-max-tokens",
        "max-tokens-cap",
    ];
    let chat = [
        "system",
        "max-tokens",
        "temperature",
        "top-p",
        "top-k",
        "seed",
        "effort",
    ];
    let mut values = HashMap::new();
    let mut flags = HashSet::new();
    while let Some(arg) = it.next() {
        let Some(arg) = arg.strip_prefix("--") else {
            return Err(Error::request(format!("unexpected argument: {arg}")));
        };
        let (key, inline) = arg
            .split_once('=')
            .map_or((arg, None), |(k, v)| (k, Some(v.to_owned())));
        let is_flag = (command == "serve" && key == "no-warmup")
            || (command == "chat"
                && matches!(key, "no-think" | "hide-thinking" | "no-loop-guard" | "raw"));
        if is_flag && inline.is_none() {
            flags.insert(key.into());
            continue;
        }
        if !common.contains(&key)
            && !(command == "serve" && serve.contains(&key))
            && !(command == "chat" && chat.contains(&key))
        {
            return Err(Error::request(format!("unknown option: --{key}")));
        }
        let value = inline
            .or_else(|| it.next())
            .ok_or_else(|| Error::request(format!("--{key} requires a value")))?;
        if value.starts_with("--") {
            return Err(Error::request(format!("--{key} requires a value")));
        }
        values.insert(key.into(), value);
    }
    Ok(Args {
        command,
        values,
        flags,
    })
}
pub async fn main(argv: Vec<String>) -> Result<()> {
    if argv.iter().any(|s| s == "--help" || s == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    if argv == ["--version"] {
        println!("yskn {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if let Some(cmd) = argv
        .first()
        .filter(|s| matches!(s.as_str(), "bench" | "selfcheck"))
    {
        return Err(Error::request(format!("{cmd}: not available yet")));
    }
    let args = parse_args(argv)?;
    if matches!(args.command.as_str(), "bench" | "selfcheck") {
        return Err(Error::request(format!(
            "{}: not available yet",
            args.command
        )));
    }
    if args.value("engine") != Some("stub") {
        return Err(Error::request(
            "the real engine is not connected yet; select --engine stub for deterministic demo output",
        ));
    }
    match args.command.as_str() {
        "info" => {
            let runner = args.runner()?;
            println!(
                "yskn {}\nengine   stub (deterministic demo, no inference)\nplatform {} {}\ncontext  {} tokens\ntokenizer  ok\nreal engine: not available yet",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH,
                runner.engine.context_capacity()
            );
            Ok(())
        }
        "serve" => {
            let mut cfg = Config {
                host: args.value("host").unwrap_or("127.0.0.1").into(),
                port: args.parse("port", 8765)?,
                api_key: args
                    .value("api-key")
                    .map(str::to_owned)
                    .or_else(|| std::env::var("YASAKANI_API_KEY").ok()),
                ..Config::default()
            };
            cfg.defaults.max_tokens = args.parse("default-max-tokens", 32768)?;
            cfg.defaults.max_tokens_cap = args
                .value("max-tokens-cap")
                .map(|s| {
                    s.parse::<usize>()
                        .map_err(|_| Error::request("--max-tokens-cap has an invalid value"))
                })
                .transpose()?;
            cfg.validate()?;
            args.capacity()?;
            let keyed = cfg.api_key.as_ref().is_some_and(|s| !s.is_empty());
            let server = Server::bind(cfg, move || {
                let mut r = args.runner()?;
                if !args.flag("no-warmup") {
                    r.warmup()?;
                }
                Ok(r)
            })
            .await?;
            let addr = server.local_addr()?;
            eprintln!(
                "listening on http://{addr} (answers /health while the model loads)\nOpenAI-compatible API at http://{addr}/v1{}\nengine: stub (deterministic demo); stop with Ctrl+C",
                if keyed { ", API key required" } else { "" }
            );
            server
                .run(async {
                    let _ = termination().await;
                    eprintln!("stopping ...");
                })
                .await
        }
        "chat" => {
            let p = chat_params(&args)?;
            let hide = args.flag("hide-thinking");
            let raw = args.flag("raw");
            let writing = Arc::new(AtomicBool::new(false));
            let cancel = Arc::new(AtomicBool::new(false));
            let worker_writing = writing.clone();
            let worker_cancel = cancel.clone();
            let (done, rx) = tokio::sync::oneshot::channel();
            std::thread::Builder::new()
                .name("yskn-terminal".into())
                .spawn(move || {
                    let result = (|| {
                        let mut runner = args.runner()?;
                        runner.warmup()?;
                        chat_loop(
                            &mut runner,
                            p,
                            hide,
                            raw,
                            io::stdin().lock(),
                            // Not locked across the loop: the interrupt handler
                            // writes too, while this thread waits for input.
                            io::stdout(),
                            &worker_writing,
                            &worker_cancel,
                        )
                    })();
                    let _ = done.send(result);
                })
                .map_err(Error::engine)?;
            tokio::pin!(rx);
            loop {
                tokio::select! {
                    result=&mut rx=>return result.map_err(|_|Error::engine("terminal worker ended"))?,
                    signal=termination()=>{signal.map_err(Error::engine)?;if writing.load(Ordering::Acquire){cancel.store(true,Ordering::Release);}else{println!();return Ok(());}}
                }
            }
        }
        _ => Err(Error::request("unknown command")),
    }
}
async fn termination() -> io::Result<()> {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {r=tokio::signal::ctrl_c()=>r,_=term.recv()=>Ok(())}
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await
    }
}
fn chat_params(args: &Args) -> Result<ChatParams> {
    let mut body = json!({"messages":[{"role":"user","content":""}],"enable_thinking":!args.flag("no-think"),"loop_guard":!args.flag("no-loop-guard")});
    for key in ["max-tokens", "temperature", "top-p", "top-k", "seed"] {
        if let Some(v) = args.value(key) {
            body[key.replace('-', "_")] = json!(v);
        }
    }
    if let Some(e) = args.value("effort") {
        if !matches!(e, "low" | "medium" | "xhigh") {
            return Err(Error::request("--effort must be low, medium or xhigh"));
        }
        body["reasoning_effort"] = json!(e);
    }
    let mut p = parse_chat_request(&body, &RequestDefaults::default())?;
    p.messages = if let Some(system) = args.value("system") {
        json!([{"role":"system","content":system}])
    } else {
        json!([])
    };
    Ok(p)
}
/// Shows C0 controls other than newline and tab, DEL and C1 controls as
/// `\xNN`, so text written to a terminal cannot drive it.
pub fn visible(text: &str) -> std::borrow::Cow<'_, str> {
    let control = |c: char| c.is_control() && c != '\n' && c != '\t';
    if !text.contains(control) {
        return text.into();
    }
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if control(c) {
            out.push_str(&format!("\\x{:02x}", c as u32));
        } else {
            out.push(c);
        }
    }
    out.into()
}
/// Plain line-oriented chat, usable with any buffered input/output (including
/// pipes). Model text and errors are passed through [`visible`] unless `raw`.
#[allow(clippy::too_many_arguments)]
pub fn chat_loop(
    runner: &mut ChatRunner,
    mut p: ChatParams,
    hide: bool,
    raw: bool,
    mut input: impl BufRead,
    mut output: impl Write,
    writing: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<()> {
    let base = p.messages.as_array().cloned().unwrap_or_default();
    let shown = |text: &str| -> String {
        if raw {
            text.into()
        } else {
            visible(text).into_owned()
        }
    };
    let mut messages = base.clone();
    writeln!(
        output,
        "YASAKANI chat - type /help for commands, /exit to quit."
    )
    .map_err(Error::engine)?;
    loop {
        write!(output, "\n> ")
            .and_then(|_| output.flush())
            .map_err(Error::engine)?;
        let mut line = String::new();
        if input.read_line(&mut line).map_err(Error::engine)? == 0 {
            writeln!(output).map_err(Error::engine)?;
            return Ok(());
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(command) = line.strip_prefix('/') {
            let (cmd, rest) = command.split_once(' ').unwrap_or((command, ""));
            let rest = rest.trim().to_ascii_lowercase();
            match cmd {
                "exit" | "quit" | "q" => return Ok(()),
                "reset" => {
                    messages = base.clone();
                    writeln!(output, "(conversation cleared)").map_err(Error::engine)?;
                }
                "think" if matches!(rest.as_str(), "on" | "off") => {
                    p.template_kwargs["enable_thinking"] = json!(rest == "on");
                    writeln!(output, "(thinking {rest})").map_err(Error::engine)?;
                }
                "effort" if matches!(rest.as_str(), "low" | "medium" | "xhigh") => {
                    p.template_kwargs["reasoning_effort"] = json!(rest);
                    writeln!(output, "(reasoning effort {rest})").map_err(Error::engine)?;
                }
                _ => writeln!(output, "{CHAT_HELP}").map_err(Error::engine)?,
            }
            continue;
        }
        messages.push(json!({"role":"user","content":line}));
        p.messages = json!(messages);
        let mut reason = String::new();
        let mut content = String::new();
        let mut last = "";
        cancel.store(false, Ordering::Release);
        writing.store(true, Ordering::Release);
        let prepared = runner.prepare_checked(&p);
        let result = prepared.and_then(|prepared| {
            runner.run_prepared(
                &p,
                prepared,
                &|| cancel.load(Ordering::Acquire),
                &mut |event| {
                    let (kind, text) = match event {
                        Event::Reasoning(t) => {
                            reason.push_str(&t);
                            if hide {
                                return Ok(());
                            }
                            ("reasoning", t)
                        }
                        Event::Content(t) => {
                            content.push_str(&t);
                            ("content", t)
                        }
                        _ => return Ok(()),
                    };
                    if last != kind {
                        if kind == "reasoning" {
                            writeln!(output, "[thinking]").map_err(Error::engine)?;
                        } else if last == "reasoning" {
                            writeln!(output, "\n[/thinking]\n").map_err(Error::engine)?;
                        }
                        last = kind;
                    }
                    write!(output, "{}", shown(&text))
                        .and_then(|_| output.flush())
                        .map_err(Error::engine)
                },
            )
        });
        writing.store(false, Ordering::Release);
        if cancel.load(Ordering::Acquire) {
            writeln!(output, "\n[stopped]").map_err(Error::engine)?;
            messages.pop();
            continue;
        }
        let u = match result {
            Ok(u) => u,
            Err(e) => {
                writeln!(
                    output,
                    "{}: {}",
                    if e.status == 400 {
                        "cannot do that"
                    } else {
                        "\nengine error"
                    },
                    shown(&e.message)
                )
                .map_err(Error::engine)?;
                messages.pop();
                if e.status != 400 {
                    runner.engine.reset()?;
                }
                continue;
            }
        };
        if !reason.is_empty() && content.is_empty() && u.finish != "stop" {
            writeln!(output,"\n(the limit of {} tokens was reached while the model was still thinking, so no answer was written.  /effort medium or /effort low shortens the thinking, /think off skips it, --max-tokens raises the limit)",p.max_tokens).map_err(Error::engine)?;
        }
        let tps = if u.gen_ms > 0.0 {
            u.completion_tokens as f64 / (u.gen_ms / 1000.0)
        } else {
            0.0
        };
        writeln!(
            output,
            "\n\n[{} tokens, {tps:.0} tokens/s, {}{}]",
            u.completion_tokens,
            if u.finish == "stop" {
                "complete"
            } else {
                "hit the length limit"
            },
            if u.loop_guard.is_some() {
                ", loop guard cut the thinking"
            } else {
                ""
            }
        )
        .map_err(Error::engine)?;
        let mut msg = json!({"role":"assistant","content":content.trim()});
        if !reason.is_empty() {
            msg["reasoning_content"] = json!(reason.trim());
        }
        messages.push(msg);
    }
}
