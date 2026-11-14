mod common;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, body::Incoming, client::conn::http1};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use std::{
    net::SocketAddr,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::oneshot,
    task::JoinHandle,
};
use yskn::{
    chat::ChatRunner,
    engine::{Engine, Generation, Sampling},
    server::{Config, Server},
    stub::{self, Control, Script, StubEngine},
};
use yskn_text::EOS_IDS;
const EOS: u32 = EOS_IDS[0];
const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
const CALL2: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
fn tools() -> Value {
    json!([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}])
}
fn user() -> Value {
    json!([{"role":"user","content":"hi"}])
}
fn body() -> Value {
    json!({"messages":user()})
}
fn ids(s: &str) -> Vec<u32> {
    stub::formatter().unwrap().encode(s).unwrap()
}
// The integration fixture proves the engine need not be Send, and all methods
// stay on the factory's owning thread. The real network/server code is used.
struct Affine {
    inner: StubEngine,
    owner: std::thread::ThreadId,
    _not_send: Rc<()>,
}
impl Engine for Affine {
    fn context_capacity(&self) -> usize {
        assert_eq!(self.owner, std::thread::current().id());
        self.inner.context_capacity()
    }
    fn reset(&mut self) -> yskn::Result<()> {
        assert_eq!(self.owner, std::thread::current().id());
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
        assert_eq!(self.owner, std::thread::current().id());
        self.inner.generate(p, n, s, stop, cb)
    }
}
struct Harness {
    addr: SocketAddr,
    control: Control,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<yskn::Result<()>>>,
}
impl Harness {
    async fn start(mut cfg: Config) -> Self {
        cfg.port = 0;
        cfg.defaults.thinking_budget = 0;
        let c = Control::default();
        let control = c.clone();
        let server = Server::bind(cfg, move || {
            ChatRunner::new(
                Box::new(Affine {
                    inner: StubEngine::with_control(20000, c),
                    owner: std::thread::current().id(),
                    _not_send: Rc::new(()),
                }),
                stub::formatter()?,
                1,
            )
        })
        .await
        .unwrap();
        let addr = server.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        let task = tokio::spawn(server.run(async {
            let _ = rx.await;
        }));
        let h = Self {
            addr,
            control,
            shutdown: Some(tx),
            task: Some(task),
        };
        for _ in 0..100 {
            if h.request("GET", "/health", None, &[]).await.status == 200 {
                return h;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("server never became ready")
    }
    fn script(&self, s: &str) {
        let mut v = ids(s);
        v.push(EOS);
        self.control.push(Script::once(v));
    }
    async fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        headers: &[(&str, &str)],
    ) -> Reply {
        let raw = body.map(|v| v.to_string()).unwrap_or_default();
        self.raw_request(method, path, raw, headers).await
    }
    async fn raw_request(
        &self,
        method: &str,
        path: &str,
        body: String,
        headers: &[(&str, &str)],
    ) -> Reply {
        let stream = TcpStream::connect(self.addr).await.unwrap();
        let (mut sender, conn) = http1::handshake(TokioIo::new(stream)).await.unwrap();
        tokio::spawn(async {
            let _ = conn.await;
        });
        let mut r = Request::builder()
            .method(method)
            .uri(path)
            .header("host", self.addr.to_string())
            .header("connection", "close")
            .header("content-length", body.len());
        for (k, v) in headers {
            r.headers_mut().unwrap().insert(
                hyper::header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                hyper::header::HeaderValue::from_str(v).unwrap(),
            );
        }
        let r = sender
            .send_request(r.body(Full::new(Bytes::from(body))).unwrap())
            .await
            .unwrap();
        let status = r.status().as_u16();
        let headers = r.headers().clone();
        let bytes = tokio::time::timeout(Duration::from_secs(10), r.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes();
        Reply {
            status,
            headers,
            text: String::from_utf8(bytes.to_vec()).unwrap(),
        }
    }
    async fn chat(&self, extra: Value) -> Reply {
        let mut b = body();
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        self.request("POST", "/v1/chat/completions", Some(b), &[])
            .await
    }
    async fn stop(mut self) {
        let _ = self.shutdown.take().unwrap().send(());
        tokio::time::timeout(Duration::from_secs(3), self.task.take().unwrap())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}
struct Reply {
    status: u16,
    headers: hyper::HeaderMap,
    text: String,
}
impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.text).unwrap()
    }
    fn events(&self) -> Vec<Value> {
        assert!(self.text.ends_with("data: [DONE]\n\n"), "{}", self.text);
        self.text
            .split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .filter(|e| *e != "[DONE]")
            .map(|e| serde_json::from_str(e).unwrap())
            .collect()
    }
}
fn pieces(ev: &[Value], key: &str) -> String {
    ev.iter()
        .filter_map(|e| e["choices"][0]["delta"][key].as_str())
        .collect()
}
// Upstream test_server.ServerTests.test_health_and_models
#[tokio::test]
async fn health_and_models() {
    let h = Harness::start(Config::default()).await;
    let r = h.request("GET", "/health", None, &[]).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["status"], "ok");
    assert_eq!(r.json()["max_context_tokens"], 20000);
    for path in ["/v1/models", "/v1/models/", "/v1/models?x=1"] {
        let r = h.request("GET", path, None, &[]).await;
        assert_eq!(r.json()["data"][0]["id"], "qwen3.8-27b");
        for (k, v) in [
            ("x-frame-options", "DENY"),
            ("content-security-policy", "frame-ancestors 'none'"),
            ("cross-origin-resource-policy", "same-origin"),
            ("x-content-type-options", "nosniff"),
        ] {
            assert_eq!(r.headers[k], v);
        }
        assert!(!r.headers.contains_key("access-control-allow-origin"));
    }
    assert_eq!(
        h.request("GET", "/v1/models/qwen3.8-27b", None, &[])
            .await
            .json()["owned_by"],
        "local"
    );
    assert_eq!(h.request("GET", "/nope", None, &[]).await.status, 404);
    h.stop().await;
}
// Upstream test_server.ServerTests.test_loading_state
#[tokio::test]
async fn loading_state() {
    let release = Arc::new(AtomicBool::new(false));
    let ready = release.clone();
    let cfg = Config {
        port: 0,
        ..Config::default()
    };
    let server = Server::bind(cfg, move || {
        while !ready.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(2));
        }
        ChatRunner::new(Box::new(StubEngine::new(20000)), stub::formatter()?, 1)
    })
    .await
    .unwrap();
    let addr = server.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    let task = tokio::spawn(server.run(async {
        let _ = rx.await;
    }));
    let h = Harness {
        addr,
        control: Control::default(),
        shutdown: Some(tx),
        task: Some(task),
    };
    let r = h.request("GET", "/health", None, &[]).await;
    assert_eq!(r.status, 503);
    assert_eq!(r.json()["status"], "loading");
    assert_eq!(h.chat(json!({})).await.status, 503);
    release.store(true, Ordering::Release);
    h.stop().await;
}
// Upstream test_server.ServerTests.test_completion
#[tokio::test]
async fn completion() {
    let h = Harness::start(Config::default()).await;
    h.script("Thinking it over.</think>\n\nThe answer is 4.");
    let r = h.chat(json!({"max_tokens":200,"seed":5})).await;
    assert_eq!(r.status, 200);
    let d = r.json();
    let m = &d["choices"][0]["message"];
    assert_eq!(m["content"], "The answer is 4.");
    assert_eq!(m["reasoning_content"], "Thinking it over.");
    assert_eq!(d["choices"][0]["finish_reason"], "stop");
    let u = &d["usage"];
    assert_eq!(
        u["total_tokens"].as_u64().unwrap(),
        u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()
    );
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(h.control.observation().sampling.unwrap().seed, 5);
    h.stop().await;
}
// Upstream test_server.ServerTests.test_streaming
#[tokio::test]
async fn streaming() {
    let h = Harness::start(Config::default()).await;
    h.script("Let me see.</think>\n\nHello, world.");
    let r = h
        .chat(json!({"stream":true,"stream_options":{"include_usage":true}}))
        .await;
    assert_eq!(r.status, 200);
    assert_eq!(
        r.headers["content-type"],
        "text/event-stream; charset=utf-8"
    );
    let ev = r.events();
    assert_eq!(ev[0]["choices"][0]["delta"], json!({"role":"assistant"}));
    assert_eq!(pieces(&ev, "reasoning_content"), "Let me see.");
    assert_eq!(pieces(&ev, "content"), "Hello, world.");
    assert_eq!(ev[ev.len() - 2]["choices"][0]["finish_reason"], "stop");
    assert_eq!(ev.last().unwrap()["choices"], json!([]));
    assert!(
        ev.last().unwrap()["usage"]["completion_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    h.stop().await;
}
// Upstream test_server.ServerTests.test_stream_has_one_chunk_per_token
#[tokio::test]
async fn stream_has_one_chunk_per_token() {
    let h = Harness::start(Config::default()).await;
    let text = "The quick brown fox jumps over the lazy dog, again and again.";
    h.script(&format!("</think>{text}"));
    let ev = h.chat(json!({"stream":true})).await.events();
    assert_eq!(pieces(&ev, "content"), text);
    assert_eq!(
        ev.iter()
            .filter(|e| e["choices"][0]["delta"]["content"].is_string())
            .count(),
        ids(text).len()
    );
    assert!(ev.last().unwrap().get("usage").is_none());
    h.stop().await;
}
// Upstream test_server.ServerTests.test_rejections; plus every public validation error.
#[tokio::test]
async fn rejections() {
    let h = Harness::start(Config::default()).await;
    let cases = common::bad_requests();
    for (extra, message) in cases {
        let r = h.chat(extra.clone()).await;
        assert_eq!(r.status, 400, "{extra}: {}", r.text);
        let d = r.json();
        assert!(
            d["error"]["message"].as_str().unwrap().contains(message),
            "{extra}: {d}"
        );
        assert_eq!(d["error"]["type"], "invalid_request_error");
        assert_eq!(d["error"]["code"], 400);
    }
    let r = h
        .request("POST", "/v1/chat/completions", Some(json!([])), &[])
        .await;
    assert_eq!(r.status, 400);
    assert_eq!(
        r.json()["error"]["message"],
        "the request body must be a JSON object"
    );
    let mut s = TcpStream::connect(h.addr).await.unwrap();
    s.write_all(
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Length: 67108864\r\n\r\n",
            h.addr
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut raw = vec![];
    s.read_to_end(&mut raw).await.unwrap();
    assert!(String::from_utf8(raw).unwrap().starts_with("HTTP/1.1 413"));
    assert_eq!(h.control.observation().calls, 0);
    h.stop().await;
}
// Upstream test_server.ServerTests.test_bad_json
#[tokio::test]
async fn bad_json() {
    let h = Harness::start(Config::default()).await;
    for b in ["{not json", "{\"x\":NaN}", "[", "{\"messages\":"] {
        let r = h
            .raw_request("POST", "/v1/chat/completions", b.into(), &[])
            .await;
        assert_eq!(r.status, 400);
        assert_eq!(
            r.json()["error"]["message"],
            "the request body is not valid JSON"
        );
    }
    h.stop().await;
}
// Upstream test_server.ServerTests.test_host_and_origin_checks
#[tokio::test]
async fn host_and_origin_checks() {
    let h = Harness::start(Config::default()).await;
    for headers in [
        vec![("host", "evil.example:80")],
        vec![("origin", "http://evil.example")],
        vec![("sec-fetch-site", "cross-site")],
        vec![("sec-fetch-site", "same-site")],
        vec![("origin", "null")],
        vec![("origin", "http://evil\t.example")],
    ] {
        let r = h
            .request("POST", "/v1/chat/completions", Some(body()), &headers)
            .await;
        assert_eq!(r.status, 403, "{:?}: {}", headers, r.text);
        assert_eq!(r.json()["error"]["type"], "forbidden");
    }
    let origin = format!("http://{}", h.addr);
    h.script("</think>okay");
    assert_eq!(
        h.request(
            "POST",
            "/v1/chat/completions",
            Some(body()),
            &[("origin", &origin), ("sec-fetch-site", "same-origin")]
        )
        .await
        .status,
        200
    );
    for host in [
        format!("localhost:{}", h.addr.port()),
        format!("[::1]:{}", h.addr.port()),
    ] {
        let mut s = TcpStream::connect(h.addr).await.unwrap();
        s.write_all(
            format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
        let mut raw = String::new();
        s.read_to_string(&mut raw).await.unwrap();
        assert!(raw.starts_with("HTTP/1.1 200"));
    }
    h.stop().await;
}
// Upstream test_server.ServerTests.test_api_key
#[tokio::test]
async fn api_key() {
    let h = Harness::start(Config {
        api_key: Some("s3cret".into()),
        ..Config::default()
    })
    .await;
    assert_eq!(h.chat(json!({})).await.status, 401);
    assert_eq!(h.request("GET", "/v1/models", None, &[]).await.status, 401);
    assert_eq!(h.request("GET", "/health", None, &[]).await.status, 200);
    for key in [
        "Bearer wrong",
        "bearer s3cret",
        "Bearer s3cre",
        "Bearer s3cretmore",
    ] {
        let r = h
            .request(
                "POST",
                "/v1/chat/completions",
                Some(body()),
                &[("authorization", key)],
            )
            .await;
        assert_eq!(r.status, 401);
        assert_eq!(r.json()["error"]["type"], "authentication_error");
    }
    h.script("</think>okay");
    assert_eq!(
        h.request(
            "POST",
            "/v1/chat/completions",
            Some(body()),
            &[("authorization", "Bearer s3cret")]
        )
        .await
        .status,
        200
    );
    h.stop().await;
}
// Upstream test_server.ServerTests.test_engine_failure_is_a_500
#[tokio::test]
async fn engine_failure_is_a_500() {
    let h = Harness::start(Config::default()).await;
    h.control.fail_next("engine error: something");
    let r = h.chat(json!({})).await;
    assert_eq!(r.status, 500);
    assert_eq!(r.json()["error"]["type"], "server_error");
    assert!(r.text.contains("engine error"));
    h.control.fail_next("stream failure");
    let r = h.chat(json!({"stream":true})).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.events().last().unwrap()["error"]["code"], 500);
    h.script("</think>recovered");
    assert_eq!(
        h.chat(json!({})).await.json()["choices"][0]["message"]["content"],
        "recovered"
    );
    h.stop().await;
}
async fn open_stream(
    h: &Harness,
    b: Value,
) -> (
    hyper::client::conn::http1::SendRequest<Full<Bytes>>,
    Incoming,
    JoinHandle<()>,
) {
    let s = TcpStream::connect(h.addr).await.unwrap();
    let (mut sender, conn) = http1::handshake(TokioIo::new(s)).await.unwrap();
    let task = tokio::spawn(async {
        let _ = conn.await;
    });
    let b = b.to_string();
    let r = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", h.addr.to_string())
        .header("content-length", b.len())
        .body(Full::new(Bytes::from(b)))
        .unwrap();
    let response = sender.send_request(r).await.unwrap();
    assert_eq!(response.status(), 200);
    (sender, response.into_body(), task)
}
// Upstream test_server.ServerTests.test_client_disconnect_stops_the_engine
#[tokio::test]
async fn client_disconnect_stops_the_engine() {
    let h = Harness::start(Config::default()).await;
    h.control.push(Script {
        ids: ids("endless text goes on "),
        loop_from: Some(0),
    });
    h.control.delay(Duration::from_millis(2));
    let b = json!({"messages":user(),"stream":true,"max_tokens":1000000,"enable_thinking":false,"loop_guard":false});
    let (sender, mut body, task) = open_stream(&h, b).await;
    for _ in 0..3 {
        body.frame().await.unwrap().unwrap();
    }
    task.abort();
    drop(sender);
    drop(body);
    for _ in 0..200 {
        if h.control.observation().stops > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(h.control.observation().stops, 1);
    assert!(h.control.observation().generated < 1000);
    h.script("</think>next");
    let r = h.chat(json!({"max_tokens":20})).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["choices"][0]["message"]["content"], "next");
    h.stop().await;
}
// Upstream test_tools.ToolRunTests.test_server_answers_with_tool_calls
#[tokio::test]
async fn server_answers_with_tool_calls() {
    let h = Harness::start(Config::default()).await;
    h.script(&format!("</think>\n\n{CALL}\n{CALL2}"));
    let r = h.chat(json!({"tools":tools()})).await;
    assert_eq!(r.status, 200);
    let d = r.json();
    let c = &d["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls");
    assert!(c["message"]["content"].is_null());
    let calls = c["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0]["id"], calls[1]["id"]);
    for (call, args) in calls
        .iter()
        .zip([json!({"city":"Osaka","days":3}), json!({"city":"Tokyo"})])
    {
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
            args
        );
    }
    h.stop().await;
}
// Upstream test_tools.ToolRunTests.test_server_stream_sends_one_chunk_per_call
#[tokio::test]
async fn server_stream_sends_one_chunk_per_call() {
    let h = Harness::start(Config::default()).await;
    h.script(&format!("</think>\n\nI will check.\n\n{CALL}\n{CALL2}"));
    let ev = h
        .chat(json!({"tools":tools(),"stream":true}))
        .await
        .events();
    assert_eq!(pieces(&ev, "content"), "I will check.");
    let calls = ev
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    for (i, c) in calls.iter().enumerate() {
        assert_eq!(c["index"], i);
        assert_eq!(c["function"]["name"], "get_weather");
    }
    assert_eq!(
        serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"city":"Osaka","days":3})
    );
    assert_eq!(
        serde_json::from_str::<Value>(calls[1]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"city":"Tokyo"})
    );
    assert_eq!(
        ev.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
    h.stop().await;
}
// Upstream test_tools.ToolRunTests.test_server_takes_the_tool_result_and_answers
#[tokio::test]
async fn server_takes_the_tool_result_and_answers() {
    let h = Harness::start(Config::default()).await;
    h.script("</think>\n\nIt is sunny in Osaka.");
    let r=h.chat(json!({"messages":[{"role":"user","content":"weather"},{"role":"assistant","content":null,"tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"Osaka\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"sunny, 22 C"}],"tools":tools()})).await;
    assert_eq!(r.status, 200);
    let d = r.json();
    assert_eq!(
        d["choices"][0]["message"]["content"],
        "It is sunny in Osaka."
    );
    assert!(d["choices"][0]["message"].get("tool_calls").is_none());
    h.stop().await;
}
// Upstream test_tools.ToolRunTests.test_server_without_tools_leaves_the_text_alone
#[tokio::test]
async fn server_without_tools_leaves_the_text_alone() {
    let h = Harness::start(Config::default()).await;
    for extra in [json!({}), json!({"tools":tools(),"tool_choice":"none"})] {
        h.script(&format!("</think>\n\n{CALL}"));
        let d = h.chat(extra).await.json();
        assert_eq!(d["choices"][0]["message"]["content"], CALL);
        assert_eq!(d["choices"][0]["finish_reason"], "stop");
    }
    h.stop().await;
}
// Upstream test_tools.ToolRunTests.test_server_refuses_a_forced_tool_choice
#[tokio::test]
async fn server_refuses_a_forced_tool_choice() {
    let h = Harness::start(Config::default()).await;
    for extra in [
        json!({"tools":tools(),"tool_choice":"required"}),
        json!({"tools":tools(),"functions":[{"name":"f"}]}),
    ] {
        assert_eq!(h.chat(extra).await.status, 400);
    }
    h.stop().await;
}
// Upstream test_tools.ToolRunTests.test_an_input_that_does_not_fit_is_a_400_before_the_stream_starts
#[tokio::test]
async fn an_input_that_does_not_fit_is_a_400_before_the_stream_starts() {
    let h = Harness::start(Config::default()).await;
    let r=h.chat(json!({"messages":[{"role":"user","content":"word ".repeat(30000)}],"tools":tools(),"stream":true})).await;
    assert_eq!(r.status, 400);
    assert_eq!(r.headers["content-type"], "application/json; charset=utf-8");
    let e = r.json();
    assert_eq!(e["error"]["code"], "context_length_exceeded");
    assert!(
        e["error"]["message"]
            .as_str()
            .unwrap()
            .contains("maximum context length")
    );
    assert_eq!(h.control.observation().calls, 0);
    h.stop().await;
}
#[tokio::test]
async fn thinking_off_stop_max_tokens_and_usage_over_socket() {
    let h = Harness::start(Config::default()).await;
    h.script("Hello STOP world");
    let r=h.chat(json!({"thinking":{"type":"disabled"},"stop":"STOP","stream":true,"stream_options":{"include_usage":true}})).await;
    let ev = r.events();
    assert_eq!(pieces(&ev, "content"), "Hello ");
    assert_eq!(pieces(&ev, "reasoning_content"), "");
    assert_eq!(
        ev.last().unwrap()["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
    h.script("long output here");
    let d = h
        .chat(json!({"enable_thinking":false,"max_completion_tokens":5}))
        .await
        .json();
    assert_eq!(d["usage"]["completion_tokens"], 5);
    assert_eq!(d["choices"][0]["finish_reason"], "length");
    h.stop().await;
}
#[tokio::test]
async fn wrong_method_and_path() {
    let h = Harness::start(Config::default()).await;
    for method in ["PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"] {
        let r = h.request(method, "/v1/chat/completions", None, &[]).await;
        assert_eq!(r.status, 501);
        if method != "HEAD" {
            assert_eq!(r.json()["error"]["code"], 501);
        }
    }
    assert_eq!(
        h.request("POST", "/wrong", Some(body()), &[]).await.status,
        404
    );
    h.stop().await;
}
#[tokio::test]
async fn slow_header_body_idle_limits_and_connection_cap() {
    let mut cfg = Config::default();
    cfg.limits.header_timeout = Duration::from_millis(100);
    cfg.limits.body_timeout = Duration::from_millis(100);
    cfg.limits.idle_timeout = Duration::from_millis(200);
    cfg.limits.max_connections = 1;
    let h = Harness::start(cfg).await;
    let mut slow = TcpStream::connect(h.addr).await.unwrap();
    slow.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHo")
        .await
        .unwrap();
    let mut buf = vec![];
    tokio::time::timeout(Duration::from_secs(2), slow.read_to_end(&mut buf))
        .await
        .unwrap()
        .unwrap();
    assert!(buf.is_empty() || String::from_utf8_lossy(&buf).starts_with("HTTP/1.1 408"));
    let mut slow = TcpStream::connect(h.addr).await.unwrap();
    slow.write_all(
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Length: 999\r\n\r\n{{",
            h.addr
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut buf = String::new();
    tokio::time::timeout(Duration::from_secs(2), slow.read_to_string(&mut buf))
        .await
        .unwrap()
        .unwrap();
    assert!(buf.starts_with("HTTP/1.1 408"), "{buf}");
    let mut idle = TcpStream::connect(h.addr).await.unwrap();
    idle.write_all(format!("GET /health HTTP/1.1\r\nHost: {}\r\n\r\n", h.addr).as_bytes())
        .await
        .unwrap();
    let mut buf = String::new();
    tokio::time::timeout(Duration::from_secs(2), idle.read_to_string(&mut buf))
        .await
        .unwrap()
        .unwrap();
    assert!(buf.starts_with("HTTP/1.1 200"));
    let mut held = TcpStream::connect(h.addr).await.unwrap();
    held.write_all(b"GET /health HTTP/1.1\r\n").await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    let mut excess = TcpStream::connect(h.addr).await.unwrap();
    let mut b = [0; 1];
    let result = tokio::time::timeout(Duration::from_secs(1), excess.read(&mut b))
        .await
        .unwrap();
    assert!(matches!(result, Ok(0) | Err(_)));
    drop(held);
    h.stop().await;
}
#[tokio::test]
async fn header_and_actual_body_size_limits() {
    let mut cfg = Config::default();
    cfg.limits.max_body = 1024;
    cfg.limits.max_headers = 8;
    cfg.limits.header_bytes = 8192;
    let h = Harness::start(cfg).await;
    assert_eq!(
        h.raw_request("POST", "/v1/chat/completions", "x".repeat(1025), &[])
            .await
            .status,
        413
    );
    for extra in [
        format!("X-Huge: {}\r\n", "a".repeat(9000)),
        (0..10).map(|i| format!("X-{i}: a\r\n")).collect(),
    ] {
        let mut s = TcpStream::connect(h.addr).await.unwrap();
        s.write_all(
            format!(
                "GET /health HTTP/1.1\r\nHost: {}\r\n{extra}Connection: close\r\n\r\n",
                h.addr
            )
            .as_bytes(),
        )
        .await
        .unwrap();
        let mut b = String::new();
        s.read_to_string(&mut b).await.unwrap();
        assert!(b.starts_with("HTTP/1.1 431"), "{b}");
    }
    let mut s = TcpStream::connect(h.addr).await.unwrap();
    s.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nTransfer-Encoding: chunked\r\n\r\n800\r\n",h.addr).as_bytes()).await.unwrap();
    let mut b = String::new();
    s.read_to_string(&mut b).await.unwrap();
    assert!(b.starts_with("HTTP/1.1 413"));
    h.stop().await;
}
#[tokio::test]
async fn two_overlapping_requests_wait_in_order() {
    let h = Harness::start(Config::default()).await;
    h.control.delay(Duration::from_millis(3));
    h.script(&format!("</think>{}", "first ".repeat(50)));
    h.script("</think>second");
    let (sender, mut body, task) = open_stream(
        &h,
        json!({"messages":user(),"stream":true,"max_tokens":1000}),
    )
    .await;
    body.frame().await.unwrap().unwrap();
    {
        let waiting = h.chat(json!({"max_tokens":30}));
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut waiting)
                .await
                .is_err()
        );
        assert_eq!(h.control.observation().calls, 1);
        let bytes = body.collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
        let r = waiting.await;
        assert_eq!(r.status, 200);
        assert_eq!(r.json()["choices"][0]["message"]["content"], "second");
    }
    drop(sender);
    task.abort();
    h.stop().await;
}
#[tokio::test]
async fn remote_bind_requires_key_and_keyed_remote_allows_other_hosts() {
    assert!(
        Server::bind(
            Config {
                host: "0.0.0.0".into(),
                port: 0,
                ..Config::default()
            },
            || unreachable!()
        )
        .await
        .is_err()
    );
    let h = Harness::start(Config {
        host: "0.0.0.0".into(),
        api_key: Some("k".into()),
        ..Config::default()
    })
    .await;
    h.script("</think>okay");
    assert_eq!(
        h.request(
            "POST",
            "/v1/chat/completions",
            Some(body()),
            &[
                ("host", "remote.example"),
                ("origin", "http://remote.example"),
                ("sec-fetch-site", "cross-site"),
                ("authorization", "Bearer k")
            ]
        )
        .await
        .status,
        200
    );
    assert_eq!(
        h.request(
            "POST",
            "/v1/chat/completions",
            Some(body()),
            &[("host", "remote.example")]
        )
        .await
        .status,
        401
    );
    h.stop().await;
}
fn normalize(raw: &str) -> String {
    let first = if raw.starts_with("data: ") {
        raw.strip_prefix("data: ")
            .unwrap()
            .split("\n\n")
            .next()
            .unwrap()
    } else {
        raw
    };
    let v: Value = serde_json::from_str(first).unwrap();
    raw.replace(&format!("\"id\":{}", v["id"]), "\"id\":\"ID\"")
        .replace(&format!("\"created\":{}", v["created"]), "\"created\":0")
}
#[tokio::test]
async fn exact_non_streaming_golden() {
    let h = Harness::start(Config::default()).await;
    h.script("OK");
    let r = h.chat(json!({"enable_thinking":false})).await;
    assert_eq!(r.status, 200);
    assert_eq!(
        normalize(&r.text),
        include_str!("data/completion.json").trim_end()
    );
    h.stop().await;
}
#[tokio::test]
async fn exact_streaming_golden() {
    let h = Harness::start(Config::default()).await;
    h.script("OK");
    let r = h
        .chat(
            json!({"enable_thinking":false,"stream":true,"stream_options":{"include_usage":true}}),
        )
        .await;
    assert_eq!(r.status, 200);
    assert_eq!(normalize(&r.text), include_str!("data/stream.sse"));
    h.stop().await;
}
