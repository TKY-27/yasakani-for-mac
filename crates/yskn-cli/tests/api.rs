//! Additional service-boundary proof. These do not replace real TCP tests.
mod common;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::Request;
use serde_json::{Value, json};
use std::{rc::Rc, time::Duration};
use yskn::{
    chat::ChatRunner,
    server::{Api, Config},
    stub::{self, Control, Script, StubEngine},
};
use yskn_text::EOS_IDS;
struct Service {
    api: Api,
    control: Control,
}
impl Service {
    async fn new(cfg: Config) -> Self {
        let control = Control::default();
        let c = control.clone();
        let api = Api::new(cfg, move || {
            let _not_send = Rc::new(());
            ChatRunner::new(
                Box::new(StubEngine::with_control(20000, c)),
                stub::formatter()?,
                1,
            )
        })
        .unwrap();
        common::until("the engine to load", || api.is_ready()).await;
        Self { api, control }
    }
    fn script(&self, s: &str) {
        let mut ids = stub::formatter().unwrap().encode(s).unwrap();
        ids.push(EOS_IDS[0]);
        self.control.push(Script::once(ids));
    }
    async fn chat(&self, extra: Value) -> (u16, String) {
        let mut b = json!({"messages":[{"role":"user","content":"hi"}]});
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        self.request("POST", "/v1/chat/completions", &b.to_string(), &[])
            .await
    }
    async fn request(
        &self,
        method: &str,
        path: &str,
        text: &str,
        headers: &[(&str, &str)],
    ) -> (u16, String) {
        let mut req = Request::builder()
            .method(method)
            .uri(path)
            .header("host", "127.0.0.1:8765")
            .header("content-length", text.len());
        for (k, v) in headers {
            req.headers_mut().unwrap().insert(
                hyper::header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                hyper::header::HeaderValue::from_str(v).unwrap(),
            );
        }
        let r = self
            .api
            .call(req.body(Full::new(Bytes::from(text.to_owned()))).unwrap())
            .await;
        let status = r.status().as_u16();
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }
}
fn events(s: &str) -> Vec<Value> {
    assert!(s.ends_with("data: [DONE]\n\n"));
    s.split("\n\n")
        .filter_map(|s| s.strip_prefix("data: "))
        .filter(|s| *s != "[DONE]")
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn normalize(raw: &str) -> String {
    let s = if raw.starts_with("data: ") {
        raw.strip_prefix("data: ")
            .unwrap()
            .split("\n\n")
            .next()
            .unwrap()
    } else {
        raw
    };
    let v: Value = serde_json::from_str(s).unwrap();
    raw.replace(&format!("\"id\":{}", v["id"]), "\"id\":\"ID\"")
        .replace(&format!("\"created\":{}", v["created"]), "\"created\":0")
}
#[tokio::test]
async fn service_goldens_match_independent_fixed_protocol() {
    let s = Service::new(Config::default()).await;
    s.script("OK");
    let (status, raw) = s.chat(json!({"enable_thinking":false})).await;
    assert_eq!(status, 200);
    assert_eq!(
        normalize(&raw),
        include_str!("data/completion.json").trim_end()
    );
    let s = Service::new(Config::default()).await;
    s.script("OK");
    let (status, raw) = s
        .chat(
            json!({"enable_thinking":false,"stream":true,"stream_options":{"include_usage":true}}),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(normalize(&raw), include_str!("data/stream.sse"));
}
#[tokio::test]
async fn service_checks_every_request_validation_error() {
    let s = Service::new(Config::default()).await;
    for (extra, needle) in common::bad_requests() {
        let (status, raw) = s.chat(extra.clone()).await;
        assert_eq!(status, 400, "{extra}: {raw}");
        let v: Value = serde_json::from_str(&raw).unwrap();
        assert!(
            v["error"]["message"].as_str().unwrap().contains(needle),
            "{extra}: {raw}"
        );
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], 400);
    }
    assert_eq!(s.control.observation().calls, 0);
    let (status, raw) = s.request("POST", "/v1/chat/completions", "[]", &[]).await;
    assert_eq!(status, 400);
    assert!(raw.contains("JSON object"));
}
#[tokio::test]
async fn service_reasons_answers_calls_and_engine_errors() {
    let s = Service::new(Config::default()).await;
    s.script("Thought</think>\n\nAnswer STOP more");
    let (status, raw) = s.chat(json!({"stop":"STOP"})).await;
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "Thought");
    assert_eq!(v["choices"][0]["message"]["content"], "Answer ");
    assert_eq!(
        v["usage"]["completion_tokens_details"]["reasoning_tokens"],
        7
    );
    s.script("</think><tool_call>{\"name\":\"f\",\"arguments\":{\"x\":3}}</tool_call>");
    let (_, raw) = s.chat(json!({"tools":[{"function":{"name":"f"}}]})).await;
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert!(v["choices"][0]["message"]["content"].is_null());
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"x\": 3}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    s.script("<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":3}}</tool_call>");
    let (_, raw) = s
        .chat(json!({"enable_thinking":false,"stream":true,"tools":[{"function":{"name":"f"}}]}))
        .await;
    let ev = events(&raw);
    let calls = ev
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["index"], 0);
    s.control.fail_next("injected engine failure");
    let (status, raw) = s.chat(json!({})).await;
    assert_eq!(status, 500);
    assert!(raw.contains("injected engine failure"));
    s.control.fail_next("stream failure");
    let (status, raw) = s.chat(json!({"stream":true})).await;
    assert_eq!(status, 200);
    assert_eq!(events(&raw).last().unwrap()["error"]["code"], 500);
    let (status, raw) = s
        .chat(json!({"messages":[{"role":"user","content":"word ".repeat(30000)}],"stream":true}))
        .await;
    assert_eq!(status, 400);
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["error"]["code"], "context_length_exceeded");
}
#[tokio::test]
async fn service_security_routes_and_real_body_limit() {
    let mut cfg = Config::default();
    cfg.limits.max_body = 1024;
    cfg.api_key = Some("k".into());
    let s = Service::new(cfg).await;
    assert_eq!(s.request("GET", "/health", "", &[]).await.0, 200);
    assert_eq!(s.request("GET", "/v1/models", "", &[]).await.0, 401);
    assert_eq!(
        s.request(
            "GET",
            "/v1/models",
            "",
            &[("authorization", "Bearer wrong")]
        )
        .await
        .0,
        401
    );
    assert_eq!(
        s.request("GET", "/v1/models", "", &[("authorization", "Bearer k")])
            .await
            .0,
        200
    );
    assert_eq!(
        s.request("GET", "/health", "", &[("host", "evil")]).await.0,
        403
    );
    for headers in [
        vec![("origin", "http://evil")],
        vec![("sec-fetch-site", "cross-site")],
    ] {
        assert_eq!(
            s.request("POST", "/v1/chat/completions", "{}", &headers)
                .await
                .0,
            403
        );
    }
    assert_eq!(
        s.request(
            "POST",
            "/v1/chat/completions",
            "{bad",
            &[("authorization", "Bearer k")]
        )
        .await
        .0,
        400
    );
    assert_eq!(
        s.request("POST", "/wrong", "", &[("authorization", "Bearer k")])
            .await
            .0,
        404
    );
    assert_eq!(
        s.request("DELETE", "/health", "", &[("authorization", "Bearer k")])
            .await
            .0,
        501
    );
    // Lie about Content-Length: the limit must be enforced on bytes actually read.
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8765")
        .header("authorization", "Bearer k")
        .header("content-length", 1)
        .body(Full::new(Bytes::from(vec![b'x'; 1025])))
        .unwrap();
    assert_eq!(s.api.call(req).await.status(), 413);
    let cfg = Config {
        host: "0.0.0.0".into(),
        api_key: None,
        ..Config::default()
    };
    assert!(Api::new(cfg, || unreachable!()).is_err());
    let s = Service::new(Config {
        host: "0.0.0.0".into(),
        api_key: Some("key".into()),
        ..Config::default()
    })
    .await;
    assert_eq!(
        s.request(
            "GET",
            "/v1/models",
            "",
            &[("host", "remote"), ("authorization", "Bearer key")]
        )
        .await
        .0,
        200
    );
}
#[tokio::test]
async fn service_waits_for_first_request_and_cancels_on_body_drop() {
    let s = Service::new(Config::default()).await;
    s.control.delay(Duration::from_millis(2));
    let v = stub::formatter().unwrap().encode("endless ").unwrap();
    s.control.push(Script {
        ids: v,
        loop_from: Some(0),
    });
    s.script("</think>second");
    let b=json!({"messages":[{"role":"user","content":"hi"}],"enable_thinking":false,"stream":true,"loop_guard":false,"max_tokens":1000000}).to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8765")
        .header("content-length", b.len())
        .body(Full::new(Bytes::from(b)))
        .unwrap();
    let mut body = s.api.call(req).await.into_body();
    for _ in 0..3 {
        body.frame().await.unwrap().unwrap();
    }
    {
        let next = s.chat(json!({"max_tokens":100}));
        tokio::pin!(next);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut next)
                .await
                .is_err()
        );
        assert_eq!(s.control.observation().calls, 1);
        drop(body);
        let (status, raw) = tokio::time::timeout(common::PATIENCE, &mut next)
            .await
            .unwrap();
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "second");
    }
    assert_eq!(s.control.observation().stops, 1);
}
#[tokio::test]
async fn service_loading_status_and_backpressure_timeout() {
    let (release, released) = std::sync::mpsc::channel::<()>();
    let api = Api::new(Config::default(), move || {
        let _ = released.recv();
        ChatRunner::new(Box::new(StubEngine::new(20000)), stub::formatter()?, 1)
    })
    .unwrap();
    let req = Request::builder()
        .uri("/health")
        .header("host", "127.0.0.1:8765")
        .body(Full::new(Bytes::new()))
        .unwrap();
    assert_eq!(api.call(req).await.status(), 503);
    release.send(()).unwrap();
    drop(api);
    let mut cfg = Config::default();
    cfg.limits.stream_timeout = Duration::from_millis(40);
    let s = Service::new(cfg).await;
    s.control.push(Script {
        ids: stub::formatter().unwrap().encode("x ").unwrap(),
        loop_from: Some(0),
    });
    let b=json!({"messages":[{"role":"user","content":"hi"}],"enable_thinking":false,"loop_guard":false,"stream":true,"max_tokens":1000000}).to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8765")
        .body(Full::new(Bytes::from(b)))
        .unwrap();
    // The unread stream stays alive; only the backpressure deadline can end
    // the endless job and free the one engine thread for the next request.
    let response = s.api.call(req).await;
    let (status, _) = tokio::time::timeout(common::PATIENCE, s.chat(json!({})))
        .await
        .expect("the stalled job was never stopped");
    assert_eq!(status, 200);
    assert_eq!(s.control.observation().stops, 1);
    drop(response);
}
#[tokio::test]
async fn body_deadline_and_template_errors_are_http_errors() {
    use hyper::body::{Body, Frame};
    use std::{
        convert::Infallible,
        pin::Pin,
        task::{Context, Poll},
    };
    struct PendingBody;
    impl Body for PendingBody {
        type Data = Bytes;
        type Error = Infallible;
        fn poll_frame(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
            Poll::Pending
        }
    }
    let mut cfg = Config::default();
    cfg.limits.body_timeout = Duration::from_millis(20);
    let s = Service::new(cfg).await;
    let r = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8765")
        .header("content-length", 10)
        .body(PendingBody)
        .unwrap();
    assert_eq!(s.api.call(r).await.status(), 408);
    let api = Api::new(Config::default(), || {
        let fmt = yskn_text::ChatFormatter::from_json(
            include_str!("../data/stub-tokenizer.json"),
            "{}",
            "{{ raise_exception('bad conversation') }}",
        )?;
        // Bridge encoding remains independent of the template's rejection.
        ChatRunner::new(Box::new(StubEngine::new(20000)), fmt, 1)
    })
    .unwrap();
    common::until("the engine to load", || api.is_ready()).await;
    let text = json!({"messages":[{"role":"user","content":"hi"}],"stream":true}).to_string();
    let r = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8765")
        .body(Full::new(Bytes::from(text)))
        .unwrap();
    let r = api.call(r).await;
    assert_eq!(r.status(), 400);
    assert_eq!(
        r.headers()["content-type"],
        "application/json; charset=utf-8"
    );
    let v: Value =
        serde_json::from_slice(&r.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(
        v["error"]["message"],
        "the chat template does not accept these messages: bad conversation"
    );
}
