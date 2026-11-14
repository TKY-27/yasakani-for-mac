mod common;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, body::Incoming, client::conn::http1};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use std::{io::ErrorKind, net::SocketAddr, rc::Rc, time::Duration};
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
        // Readiness is checked without a connection so none is left lingering
        // on a connection permit when a test starts.
        common::until("the engine to load", || server.is_ready()).await;
        let addr = server.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        let task = tokio::spawn(server.run(async {
            let _ = rx.await;
        }));
        Self {
            addr,
            control,
            shutdown: Some(tx),
            task: Some(task),
        }
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
        let exchange = async {
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
            let bytes = r.into_body().collect().await.unwrap().to_bytes();
            Reply {
                status,
                headers,
                text: String::from_utf8(bytes.to_vec()).unwrap(),
            }
        };
        tokio::time::timeout(common::PATIENCE, exchange)
            .await
            .expect("request timed out")
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
        tokio::time::timeout(common::PATIENCE, self.task.take().unwrap())
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
/// Writes raw bytes, then reads until the server closes. A reset fails the test.
async fn raw(addr: SocketAddr, request: &str) -> String {
    let exchange = async {
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(request.as_bytes()).await.unwrap();
        let mut out = vec![];
        s.read_to_end(&mut out).await.unwrap();
        String::from_utf8(out).unwrap()
    };
    tokio::time::timeout(common::PATIENCE, exchange)
        .await
        .expect("raw exchange timed out")
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
    let (release, released) = std::sync::mpsc::channel::<()>();
    let cfg = Config {
        port: 0,
        ..Config::default()
    };
    let server = Server::bind(cfg, move || {
        let _ = released.recv();
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
    release.send(()).unwrap();
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
    let r = raw(
        h.addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Length: 67108864\r\n\r\n",
            h.addr
        ),
    )
    .await;
    assert!(r.starts_with("HTTP/1.1 413"), "{r}");
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
        let r = raw(
            h.addr,
            &format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"),
        )
        .await;
        assert!(r.starts_with("HTTP/1.1 200"), "{r}");
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
    // The one engine thread serves this only after the endless job has ended,
    // which it can do within the context room only by being cancelled.
    h.script("</think>next");
    let r = h.chat(json!({"max_tokens":20})).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["choices"][0]["message"]["content"], "next");
    let seen = h.control.observation();
    assert_eq!(seen.calls, 2);
    assert_eq!(seen.stops, 1);
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
    // A stop string exactly at the length limit is accepted.
    h.script("Hello STOP world");
    let r = h
        .chat(json!({"enable_thinking":false,"stop":["x".repeat(1024), "STOP"]}))
        .await;
    assert_eq!(r.json()["choices"][0]["message"]["content"], "Hello ");
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
// Each deadline is the only short one in its server: load can delay the
// outcome but cannot change it.
#[tokio::test]
async fn header_body_and_idle_deadlines() {
    let mut cfg = Config::default();
    cfg.limits.header_timeout = Duration::from_millis(200);
    let h = Harness::start(cfg).await;
    // Hyper ends a header timeout without a response (proto/h1/conn.rs).
    assert_eq!(
        raw(h.addr, "POST /v1/chat/completions HTTP/1.1\r\nHo").await,
        ""
    );
    h.stop().await;

    let mut cfg = Config::default();
    cfg.limits.body_timeout = Duration::from_millis(200);
    let h = Harness::start(cfg).await;
    let r = raw(
        h.addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Length: 999\r\n\r\n{{",
            h.addr
        ),
    )
    .await;
    assert!(r.starts_with("HTTP/1.1 408"), "{r}");
    assert_eq!(h.control.observation().calls, 0);
    h.stop().await;

    let mut cfg = Config::default();
    cfg.limits.idle_timeout = Duration::from_secs(1);
    let h = Harness::start(cfg).await;
    // A kept-alive connection: the idle deadline is what closes it.
    let r = raw(
        h.addr,
        &format!("GET /health HTTP/1.1\r\nHost: {}\r\n\r\n", h.addr),
    )
    .await;
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    h.stop().await;
}
#[tokio::test]
async fn connection_cap_closes_excess_connections_unserved() {
    let mut cfg = Config::default();
    cfg.limits.max_connections = 1;
    let h = Harness::start(cfg).await;
    let mut held = TcpStream::connect(h.addr).await.unwrap();
    held.write_all(b"GET /health HTTP/1.1\r\n").await.unwrap();
    // The listener accepts in connect order, so `held` owns the only permit.
    let mut excess = TcpStream::connect(h.addr).await.unwrap();
    let request = format!("GET /health HTTP/1.1\r\nHost: {}\r\n\r\n", h.addr);
    let _ = excess.write_all(request.as_bytes()).await; // may already be closed
    let mut out = vec![];
    let read = tokio::time::timeout(common::PATIENCE, excess.read_to_end(&mut out))
        .await
        .unwrap();
    // Unserved: EOF, or a reset when the request arrived before the close.
    assert!(out.is_empty(), "{}", String::from_utf8_lossy(&out));
    assert!(matches!(&read, Ok(0)) || read.is_err_and(|e| e.kind() == ErrorKind::ConnectionReset));
    held.write_all(format!("Host: {}\r\nConnection: close\r\n\r\n", h.addr).as_bytes())
        .await
        .unwrap();
    let mut out = String::new();
    tokio::time::timeout(common::PATIENCE, held.read_to_string(&mut out))
        .await
        .unwrap()
        .unwrap();
    assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    h.stop().await;
}
// Early rejections leave request bytes unread; the response must still arrive
// rather than be destroyed by a reset (`raw` fails on one).
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
    // Hyper answers these itself, before the handler.
    for extra in [
        format!("X-Huge: {}\r\n", "a".repeat(9000)),
        (0..10).map(|i| format!("X-{i}: a\r\n")).collect(),
    ] {
        let r = raw(
            h.addr,
            &format!(
                "GET /health HTTP/1.1\r\nHost: {}\r\n{extra}Connection: close\r\n\r\n",
                h.addr
            ),
        )
        .await;
        assert!(r.starts_with("HTTP/1.1 431"), "{r}");
    }
    let r = raw(
        h.addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nTransfer-Encoding: chunked\r\n\r\n800\r\n",
            h.addr
        ),
    )
    .await;
    assert!(r.starts_with("HTTP/1.1 413"), "{r}");
    // Rejected before the handler reads the body.
    let unread = "x".repeat(32 * 1024);
    for (host, status) in [("evil.example".to_owned(), 403), (h.addr.to_string(), 404)] {
        let r = raw(
            h.addr,
            &format!(
                "POST /wrong HTTP/1.1\r\nHost: {host}\r\nContent-Length: {}\r\n\r\n{unread}",
                unread.len()
            ),
        )
        .await;
        assert!(r.starts_with(&format!("HTTP/1.1 {status}")), "{r}");
    }
    assert_eq!(h.control.observation().calls, 0);
    h.stop().await;
}
// A peer that keeps sending after its rejection cannot hold the connection:
// once by the discard budget alone, once by the deadline alone.
#[tokio::test]
async fn lingering_close_is_bounded() {
    let mut cfg = Config::default();
    cfg.limits.max_body = 1024;
    cfg.limits.linger_bytes = 4096;
    cfg.limits.linger_timeout = Duration::from_secs(3600);
    let h = Harness::start(cfg).await;
    cut_off_after_rejection(&h, 1024).await;
    h.stop().await;

    let mut cfg = Config::default();
    cfg.limits.max_body = 1024;
    cfg.limits.linger_timeout = Duration::from_millis(200);
    cfg.limits.idle_timeout = Duration::from_secs(3600);
    let h = Harness::start(cfg).await;
    cut_off_after_rejection(&h, 1).await;
    h.stop().await;
}
async fn cut_off_after_rejection(h: &Harness, chunk: usize) {
    let mut s = TcpStream::connect(h.addr).await.unwrap();
    s.write_all(
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Length: 2048\r\n\r\n",
            h.addr
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    // The whole response, then the server's half-close.
    let mut out = vec![];
    tokio::time::timeout(common::PATIENCE, s.read_to_end(&mut out))
        .await
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(&out).starts_with("HTTP/1.1 413"));
    let junk = vec![b'x'; chunk];
    let cut = tokio::time::timeout(common::PATIENCE, async {
        // Writes fail once the server has fully closed and reset the stream.
        while s.write_all(&junk).await.is_ok() {
            tokio::time::sleep(Duration::from_millis(5)).await; // pacing only
        }
    })
    .await;
    assert!(
        cut.is_ok(),
        "the server never closed a lingering connection"
    );
}
#[tokio::test]
async fn two_overlapping_requests_wait_in_order() {
    let h = Harness::start(Config::default()).await;
    h.control.delay(Duration::from_millis(2));
    // The first job never ends by itself, so the second cannot overtake it
    // however long the window below takes on a loaded machine.
    h.control.push(Script {
        ids: ids("first "),
        loop_from: Some(0),
    });
    h.script("</think>second");
    let b = json!({"messages":user(),"stream":true,"max_tokens":1000000,"enable_thinking":false,"loop_guard":false});
    let (sender, mut body, task) = open_stream(&h, b).await;
    body.frame().await.unwrap().unwrap();
    {
        let waiting = h.chat(json!({"max_tokens":30}));
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut waiting)
                .await
                .is_err()
        );
        assert_eq!(h.control.observation().calls, 1);
        body.frame().await.unwrap().unwrap();
        // Disconnecting the first client is what lets the second run.
        task.abort();
        drop((sender, body));
        let r = waiting.await;
        assert_eq!(r.status, 200);
        assert_eq!(r.json()["choices"][0]["message"]["content"], "second");
    }
    let seen = h.control.observation();
    assert_eq!((seen.calls, seen.stops), (2, 1));
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
fn error_message(r: &Reply) -> String {
    r.json()["error"]["message"].as_str().unwrap().to_owned()
}
// Review finding: a small body could expand into a huge parsed and cleaned
// request. Shape limits refuse it before anything is copied.
#[tokio::test]
async fn request_shape_limits_refuse_expanding_bodies() {
    let h = Harness::start(Config::default()).await;
    // The default body limit is 8 MiB (upstream: 32 MiB).
    let r = raw(
        h.addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\n\r\n",
            h.addr,
            8 * 1024 * 1024 + 1
        ),
    )
    .await;
    assert!(r.starts_with("HTTP/1.1 413"), "{r}");
    let msgs = vec![r#"{"role":"user"}"#; 65537].join(",");
    let r = h
        .raw_request(
            "POST",
            "/v1/chat/completions",
            format!(r#"{{"messages":[{msgs}]}}"#),
            &[],
        )
        .await;
    assert_eq!(r.status, 400, "{}", r.text);
    assert_eq!(error_message(&r), "messages: at most 65536 are accepted");
    assert_eq!(r.json()["error"]["type"], "invalid_request_error");
    assert_eq!(r.json()["error"]["code"], 400);
    // 2^19 values is the default cap; the root and "messages" make 2 more.
    let zeros = vec!["0"; 1 << 19].join(",");
    let r = h
        .raw_request(
            "POST",
            "/v1/chat/completions",
            format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],"x":[{zeros}]}}"#),
            &[],
        )
        .await;
    assert_eq!(r.status, 400, "{}", r.text);
    assert_eq!(
        error_message(&r),
        "the request holds more than 524288 JSON values"
    );
    assert_eq!(h.control.observation().calls, 0);
    h.stop().await;

    let mut cfg = Config::default();
    let l = &mut cfg.limits;
    (
        l.max_messages,
        l.max_content_parts,
        l.max_tool_calls,
        l.max_tools,
    ) = (2, 2, 2, 2);
    l.max_text_bytes = 512;
    let h = Harness::start(cfg).await;
    let part = json!({"type":"text","text":"a"});
    let call = json!({"function":{"name":"f","arguments":"{\"x\":[1,2,3]}"}});
    let tool = json!({"type":"function","function":{"name":"f"}});
    let user = json!({"role":"user","content":"hi"});
    for (b, message) in [
        (
            json!({"messages":[user, user, user]}),
            "messages: at most 2 are accepted",
        ),
        (
            json!({"messages":[{"role":"user","content":[part, part, part]}]}),
            "messages[0]: content parts: at most 2 are accepted",
        ),
        (
            json!({"messages":[user, {"role":"assistant","tool_calls":[call, call, call]}]}),
            "messages[1]: tool calls: at most 2 are accepted",
        ),
        (
            json!({"messages":[user],"tools":[tool, tool, tool]}),
            "tools: at most 2 are accepted",
        ),
        (
            json!({"messages":[{"role":"user","content":"x".repeat(498)}]}),
            "the request holds more than 512 bytes of text",
        ),
    ] {
        let r = h
            .request("POST", "/v1/chat/completions", Some(b.clone()), &[])
            .await;
        assert_eq!(r.status, 400, "{b}: {}", r.text);
        assert_eq!(error_message(&r), message, "{b}");
    }
    // Exactly at each limit is accepted.
    for b in [
        json!({"messages":[{"role":"user","content":[part, part]}, {"role":"assistant","tool_calls":[call, call]}],"tools":[tool, tool],"max_tokens":4}),
        // Text counts keys too: "role", "user" and "content" are 15 bytes.
        json!({"messages":[{"role":"user","content":"x".repeat(512 - 15)}],"max_tokens":4}),
    ] {
        let r = h
            .request("POST", "/v1/chat/completions", Some(b.clone()), &[])
            .await;
        assert_eq!(r.status, 200, "{b}: {}", r.text);
    }
    assert_eq!(h.control.observation().calls, 2);
    h.stop().await;
}
// A prompt far beyond what the context could hold is refused between template
// rendering and tokenization; the exact token check stays for the rest.
#[tokio::test]
async fn far_too_long_input_is_refused_before_tokenization() {
    let h = Harness::start(Config::default()).await;
    let r = h
        .chat(json!({"messages":[{"role":"user","content":"a".repeat(2 << 20)}],"stream":true}))
        .await;
    assert_eq!(r.status, 400, "{}", r.text);
    assert_eq!(r.json()["error"]["code"], "context_length_exceeded");
    let message = error_message(&r);
    assert!(
        message.contains("bytes of prompt text cannot fit"),
        "{message}"
    );
    assert!(
        message.contains("maximum context length of 20000 tokens"),
        "{message}"
    );
    assert_eq!(h.control.observation().calls, 0);
    h.stop().await;
}
// All requests together are bounded: one that would push the bytes held by
// received, queued and running requests past the budget gets 503 with
// Retry-After, and the budget is returned when a request finishes.
#[tokio::test]
async fn global_budget_refuses_with_retry_after_and_recovers() {
    let mut cfg = Config::default();
    cfg.limits.max_body = 64 * 1024;
    cfg.limits.max_json_values = 16;
    cfg.limits.max_pending_bytes = 64 * 1024 + 16 * yskn::server::VALUE_COST;
    // The budget must hold one largest request by itself.
    let mut small = cfg.clone();
    small.limits.max_pending_bytes -= 1;
    assert!(small.validate().is_err());
    let h = Harness::start(cfg).await;
    h.control.delay(Duration::from_millis(2));
    h.control.push(Script {
        ids: ids("busy "),
        loop_from: Some(0),
    });
    let pad = "x".repeat(40 * 1024);
    let b = json!({"messages":user(),"stream":true,"max_tokens":1000000,"enable_thinking":false,"loop_guard":false,"pad":pad});
    let (sender, mut body, task) = open_stream(&h, b).await;
    body.frame().await.unwrap().unwrap();
    // Fits alone, but not beside the running request.
    let big = json!({"messages":user(),"max_tokens":4,"pad":pad});
    let r = h
        .request("POST", "/v1/chat/completions", Some(big.clone()), &[])
        .await;
    assert_eq!(r.status, 503, "{}", r.text);
    assert_eq!(r.headers["retry-after"], "5");
    assert_eq!(r.json()["error"]["type"], "unavailable");
    task.abort();
    drop((sender, body));
    // Served only after the cancelled request has left the engine thread,
    // which releases its charge.
    h.script("</think>small");
    assert_eq!(h.chat(json!({"max_tokens":20})).await.status, 200);
    let r = h
        .request("POST", "/v1/chat/completions", Some(big), &[])
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    h.stop().await;
}
fn raw_post(h: &Harness, b: &Value) -> String {
    let b = b.to_string();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\n\r\n{b}",
        h.addr,
        b.len()
    )
}
/// Holds the engine with an endless streamed answer until the stream is dropped.
async fn hold_engine(
    h: &Harness,
) -> (
    hyper::client::conn::http1::SendRequest<Full<Bytes>>,
    Incoming,
    JoinHandle<()>,
) {
    h.control.delay(Duration::from_millis(2));
    h.control.push(Script {
        ids: ids("busy "),
        loop_from: Some(0),
    });
    let b = json!({"messages":user(),"stream":true,"max_tokens":1000000,"enable_thinking":false,"loop_guard":false});
    let (sender, mut body, task) = open_stream(h, b).await;
    body.frame().await.unwrap().unwrap();
    (sender, body, task)
}
/// Polls until a new connection is served, which with every other permit held
/// means a connection that was closed has been fully dropped by the server.
async fn until_a_connection_is_free(h: &Harness) {
    let request = format!(
        "GET /health HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        h.addr
    );
    let deadline = std::time::Instant::now() + common::PATIENCE;
    loop {
        let mut s = TcpStream::connect(h.addr).await.unwrap();
        let _ = s.write_all(request.as_bytes()).await; // may be closed unserved
        let mut out = vec![];
        let _ = tokio::time::timeout(common::PATIENCE, s.read_to_end(&mut out)).await;
        if out.starts_with(b"HTTP/1.1 200") {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no connection was freed"
        );
        tokio::time::sleep(Duration::from_millis(1)).await; // polling only
    }
}
// Review finding: waiting for the engine had no overall bound. A request that
// waits longer than `queue_timeout` gets 503 with Retry-After and never
// reaches the engine.
#[tokio::test]
async fn queue_wait_is_bounded() {
    let mut cfg = Config::default();
    cfg.limits.queue_timeout = Duration::from_millis(300);
    let h = Harness::start(cfg).await;
    let (sender, body, task) = hold_engine(&h).await;
    let r = h.chat(json!({"max_tokens":20})).await;
    assert_eq!(r.status, 503, "{}", r.text);
    assert_eq!(r.headers["retry-after"], "5");
    assert!(error_message(&r).contains("waited too long"), "{}", r.text);
    task.abort();
    drop((sender, body));
    h.script("</think>next");
    let r = h.chat(json!({"max_tokens":20})).await;
    assert_eq!(r.json()["choices"][0]["message"]["content"], "next");
    let seen = h.control.observation();
    assert_eq!((seen.calls, seen.stops), (2, 1));
    h.stop().await;
}
// A client that disconnects while its request waits in the queue never
// reaches the engine.
#[tokio::test]
async fn client_disconnect_while_queued_never_reaches_the_engine() {
    let mut cfg = Config::default();
    cfg.limits.max_connections = 2;
    let h = Harness::start(cfg).await;
    let (sender, body, task) = hold_engine(&h).await;
    let mut queued = TcpStream::connect(h.addr).await.unwrap();
    queued
        .write_all(raw_post(&h, &json!({"messages":user(),"max_tokens":20})).as_bytes())
        .await
        .unwrap();
    // The request is not answered while the engine is held.
    let mut byte = [0_u8; 1];
    assert!(
        tokio::time::timeout(Duration::from_millis(100), queued.read(&mut byte))
            .await
            .is_err()
    );
    drop(queued);
    until_a_connection_is_free(&h).await;
    task.abort();
    drop((sender, body));
    h.script("</think>next");
    let r = h.chat(json!({"max_tokens":20})).await;
    assert_eq!(r.json()["choices"][0]["message"]["content"], "next");
    let seen = h.control.observation();
    assert_eq!((seen.calls, seen.stops), (2, 1));
    h.stop().await;
}
// A non-streaming client that disconnects during generation stops the engine.
#[tokio::test]
async fn client_disconnect_during_a_non_streaming_request_stops_the_engine() {
    let h = Harness::start(Config::default()).await;
    h.control.delay(Duration::from_millis(2));
    h.control.push(Script {
        ids: ids("busy "),
        loop_from: Some(0),
    });
    let mut s = TcpStream::connect(h.addr).await.unwrap();
    let b =
        json!({"messages":user(),"max_tokens":1000000,"enable_thinking":false,"loop_guard":false});
    s.write_all(raw_post(&h, &b).as_bytes()).await.unwrap();
    let control = h.control.clone();
    common::until("generation to start", || {
        control.observation().generated > 0
    })
    .await;
    drop(s);
    // Served only once the endless job has been cancelled.
    h.script("</think>next");
    let r = h.chat(json!({"max_tokens":20})).await;
    assert_eq!(r.json()["choices"][0]["message"]["content"], "next");
    let seen = h.control.observation();
    assert_eq!((seen.calls, seen.stops), (2, 1));
    h.stop().await;
}
// Review finding: an absolute-form request target's authority was not
// checked against the Host allow-list. It is now held to the same rule.
#[tokio::test]
async fn absolute_form_target_authority_is_checked() {
    let h = Harness::start(Config::default()).await;
    let own = h.addr.to_string();
    for (target, status) in [
        (format!("http://evil.example:{}/health", h.addr.port()), 403),
        ("http://evil.example/health".to_owned(), 403),
        (format!("http://user@{own}/health"), 403),
        (format!("https://{own}/health"), 403),
        (format!("http://{own}/health"), 200),
        (format!("http://localhost:{}/health", h.addr.port()), 200),
    ] {
        let r = raw(
            h.addr,
            &format!("GET {target} HTTP/1.1\r\nHost: {own}\r\nConnection: close\r\n\r\n"),
        )
        .await;
        assert!(
            r.starts_with(&format!("HTTP/1.1 {status}")),
            "{target}: {r}"
        );
    }
    let b = body().to_string();
    let r = raw(
        h.addr,
        &format!(
            "POST http://evil.example/v1/chat/completions HTTP/1.1\r\nHost: {own}\r\nContent-Length: {}\r\n\r\n{b}",
            b.len()
        ),
    )
    .await;
    assert!(r.starts_with("HTTP/1.1 403"), "{r}");
    assert!(r.contains("forbidden: host"), "{r}");
    assert_eq!(h.control.observation().calls, 0);
    h.stop().await;
}
