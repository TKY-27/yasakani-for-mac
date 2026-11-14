//! HTTP/1 server with one thread-affine engine and bounded transport resources.
use crate::{
    Error, Result,
    chat::{ChatRunner, Event, Usage},
    request::{
        RequestDefaults, check_size, check_values, json_values, parse_chat_request, text_bytes,
        truthy,
    },
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::{
    Method, Request, Response, StatusCode,
    body::{Body, Frame},
    server::conn::http1,
    service::service_fn,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    convert::Infallible,
    future::Future,
    io,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll, ready},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{Semaphore, mpsc, oneshot},
    task::JoinSet,
    time::{Instant, Sleep},
};

pub const MAX_BODY: usize = 8 * 1024 * 1024;
/// Heap bytes charged per counted JSON value of a request, on top of its body
/// bytes: an upper estimate for the parsed body and the cleaned copy that
/// exist together while it is decoded. Measured worst case: about 590 bytes
/// per value, for tool schemas made of one-member objects.
pub const VALUE_COST: usize = 640;
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_body: usize,
    /// Global budget for request bytes being received, decoded, queued or
    /// generating; a request that would exceed it gets 503 and Retry-After.
    pub max_pending_bytes: usize,
    pub max_json_values: usize,
    pub max_messages: usize,
    pub max_content_parts: usize,
    pub max_tool_calls: usize,
    pub max_tools: usize,
    pub max_text_bytes: usize,
    pub max_stop_bytes: usize,
    /// Longest wait for the engine, from queueing until generation starts;
    /// exceeded: 503 with Retry-After. Generation itself is bounded by
    /// `max_tokens` and ends when the client disconnects.
    pub queue_timeout: Duration,
    pub max_connections: usize,
    pub max_headers: usize,
    pub header_bytes: usize,
    pub header_timeout: Duration,
    pub body_timeout: Duration,
    pub idle_timeout: Duration,
    pub stream_timeout: Duration,
    /// Ending a connection after a response half-closes first, then discards
    /// at most `linger_bytes` from the peer for at most `linger_timeout`.
    pub linger_bytes: usize,
    pub linger_timeout: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_body: MAX_BODY,
            max_pending_bytes: 512 * 1024 * 1024,
            max_json_values: 1 << 19,
            max_messages: 65536,
            max_content_parts: 1024,
            max_tool_calls: 1024,
            max_tools: 1024,
            max_text_bytes: MAX_BODY,
            max_stop_bytes: 1024,
            queue_timeout: Duration::from_secs(600),
            max_connections: 64,
            max_headers: 100,
            header_bytes: 32768,
            header_timeout: Duration::from_secs(10),
            body_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            stream_timeout: Duration::from_secs(30),
            linger_bytes: 256 * 1024,
            linger_timeout: Duration::from_secs(2),
        }
    }
}
#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub model_id: String,
    pub defaults: RequestDefaults,
    pub limits: Limits,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8765,
            api_key: None,
            model_id: "qwen3.8-27b".into(),
            defaults: RequestDefaults::default(),
            limits: Limits::default(),
        }
    }
}
fn local(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if !local(&self.host) && self.api_key.as_ref().is_none_or(String::is_empty) {
            return Err(Error::request(format!(
                "refusing to listen on {} without --api-key: anyone who can reach this computer could use the engine",
                self.host
            )));
        }
        let l = &self.limits;
        // A single largest request must fit the budget when nothing else runs.
        let largest = l
            .max_body
            .saturating_add(l.max_json_values.saturating_mul(VALUE_COST));
        if l.max_pending_bytes < largest || l.max_pending_bytes > Semaphore::MAX_PERMITS {
            return Err(Error::request("invalid server limits"));
        }
        if self.limits.max_connections == 0
            || self.limits.max_connections > 1024
            || self.limits.max_headers == 0
            || self.limits.header_bytes < 8192
            || self.defaults.max_tokens == 0
            || self.defaults.max_tokens_cap == Some(0)
        {
            return Err(Error::request("invalid server limits"));
        }
        Ok(())
    }
}
// Every response, including errors, carries the upstream protection headers.
type HttpBody = BoxBody<Bytes, Infallible>;
fn response(status: u16, content_type: &'static str, body: HttpBody) -> Response<HttpBody> {
    let mut r = Response::new(body);
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let h = r.headers_mut();
    for (k, v) in [
        ("content-type", content_type),
        ("x-frame-options", "DENY"),
        ("content-security-policy", "frame-ancestors 'none'"),
        ("cross-origin-resource-policy", "same-origin"),
        ("x-content-type-options", "nosniff"),
    ] {
        h.insert(k, hyper::header::HeaderValue::from_static(v));
    }
    r
}
fn json_response(status: u16, v: Value) -> Response<HttpBody> {
    let bytes = v.to_string();
    let len = bytes.len();
    let mut r = response(
        status,
        "application/json; charset=utf-8",
        Full::new(Bytes::from(bytes)).boxed(),
    );
    if let Ok(v) = hyper::header::HeaderValue::from_str(&len.to_string()) {
        r.headers_mut().insert("content-length", v);
    }
    r
}
fn error_value(e: &Error) -> Value {
    json!({"error":{"message":e.message,"type":e.kind,"code":e.code.map_or_else(||json!(e.status),|c|json!(c))}})
}
fn error_response(e: Error) -> Response<HttpBody> {
    json_response(e.status, error_value(&e))
}
fn http_error(status: u16, message: impl Into<String>, kind: &'static str) -> Error {
    Error {
        message: message.into(),
        status,
        kind,
        code: None,
    }
}
fn close(mut r: Response<HttpBody>) -> Response<HttpBody> {
    r.headers_mut().insert(
        "connection",
        hyper::header::HeaderValue::from_static("close"),
    );
    r
}

fn unique(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    format!(
        "{prefix}{time:016x}{:08x}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
pub fn usage_json(u: &Usage) -> Value {
    let mut v = json!({"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,"total_tokens":u.prompt_tokens+u.completion_tokens,"prompt_tokens_details":{"cached_tokens":u.cached_tokens},"completion_tokens_details":{"reasoning_tokens":u.reasoning_tokens}});
    if let Some(info) = &u.loop_guard {
        v["loop_guard"] = info.clone();
    }
    v
}
fn tool_call_json(call: Value, index: Option<usize>) -> Value {
    let mut v = json!({"id":unique("call_"),"type":"function","function":{"name":call["name"],"arguments":yskn_text::json_dumps(&call["arguments"],false)}});
    if let Some(i) = index {
        let mut indexed = json!({"index":i});
        if let Some(m) = indexed.as_object_mut() {
            if let Some(v) = v.as_object() {
                m.extend(v.clone());
            }
        }
        v = indexed;
    }
    v
}
struct Envelope {
    id: String,
    created: u64,
    model: String,
}
impl Envelope {
    fn new(model: &str) -> Self {
        Self {
            id: unique("chatcmpl-"),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: model.into(),
        }
    }
    fn chunk(&self, delta: Value, finish: Option<&str>, usage: Option<Value>) -> Value {
        let choices = if usage.is_some() {
            json!([])
        } else {
            json!([{"index":0,"delta":delta,"finish_reason":finish}])
        };
        let mut v = json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":choices});
        if let Some(u) = usage {
            v["usage"] = u;
        }
        v
    }
    fn completed(&self, reason: String, content: String, calls: Vec<Value>, u: Usage) -> Value {
        let mut msg = json!({"role":"assistant","content":content});
        if !reason.is_empty() {
            msg["reasoning_content"] = json!(reason);
        }
        if !calls.is_empty() {
            msg["tool_calls"] = json!(calls);
            if msg["content"] == "" {
                msg["content"] = Value::Null;
            }
        }
        json!({"id":self.id,"object":"chat.completion","created":self.created,"model":self.model,"choices":[{"index":0,"message":msg,"finish_reason":u.finish}],"usage":usage_json(&u)})
    }
}
fn sse(v: Value) -> Bytes {
    Bytes::from(format!("data: {v}\n\n"))
}

struct Cancel(Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
struct Job {
    params: crate::chat::ChatParams,
    ready: oneshot::Sender<Result<mpsc::Receiver<Result<Event>>>>,
    cancel: Arc<AtomicBool>,
    // Released when the engine thread drops the job, after its generation.
    _charge: Charge,
}
/// A request's share of `Limits::max_pending_bytes`.
struct Charge {
    budget: Arc<Semaphore>,
    permit: tokio::sync::OwnedSemaphorePermit,
}
impl Charge {
    fn new(budget: &Arc<Semaphore>) -> Option<Self> {
        Some(Self {
            budget: budget.clone(),
            permit: budget.clone().try_acquire_many_owned(0).ok()?,
        })
    }
    /// Grows or shrinks the charge to `n` bytes; false if the budget lacks room.
    fn resize(&mut self, n: usize) -> bool {
        let have = self.permit.num_permits();
        if n <= have {
            drop(self.permit.split(have - n));
            return true;
        }
        let more = u32::try_from(n - have)
            .ok()
            .and_then(|k| self.budget.clone().try_acquire_many_owned(k).ok());
        more.map(|p| self.permit.merge(p)).is_some()
    }
}
struct Shared {
    cfg: Config,
    budget: Arc<Semaphore>,
    loaded: AtomicBool,
    capacity: AtomicUsize,
    shutdown: AtomicBool,
    failure: Mutex<Option<Error>>,
    failure_ready: tokio::sync::Notify,
}

/// Reusable HTTP service. The factory crosses the thread boundary; the engine
/// never does. Dropping the service cancels generation and joins its owner.
pub struct Api {
    shared: Arc<Shared>,
    jobs: Option<mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Api {
    pub fn new<F>(cfg: Config, factory: F) -> Result<Self>
    where
        F: FnOnce() -> Result<ChatRunner> + Send + 'static,
    {
        cfg.validate()?;
        let (jobs, mut queue) = mpsc::channel::<Job>(cfg.limits.max_connections);
        let shared = Arc::new(Shared {
            budget: Arc::new(Semaphore::new(cfg.limits.max_pending_bytes)),
            cfg,
            loaded: AtomicBool::new(false),
            capacity: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            failure: Mutex::new(None),
            failure_ready: tokio::sync::Notify::new(),
        });
        let state = shared.clone();
        let worker = std::thread::Builder::new()
            .name("yskn-engine".into())
            .spawn(move || {
                let mut runner = match factory() {
                    Ok(r) => r,
                    Err(e) => {
                        *state
                            .failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(e);
                        state.failure_ready.notify_one();
                        return;
                    }
                };
                state
                    .capacity
                    .store(runner.engine.context_capacity(), Ordering::Release);
                state.loaded.store(true, Ordering::Release);
                while let Some(job) = queue.blocking_recv() {
                    let shared = &state;
                    let cancelled = || {
                        job.cancel.load(Ordering::Acquire)
                            || shared.shutdown.load(Ordering::Acquire)
                    };
                    if cancelled() || job.ready.is_closed() {
                        continue;
                    }
                    let prepared = match runner.prepare_checked(&job.params) {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = job.ready.send(Err(e));
                            continue;
                        }
                    };
                    let (events, rx) = mpsc::channel(8);
                    if job.ready.send(Ok(rx)).is_err() {
                        continue;
                    }
                    let send = |mut event: Result<Event>| -> Result<()> {
                        let deadline = std::time::Instant::now() + shared.cfg.limits.stream_timeout;
                        loop {
                            if cancelled() {
                                return Err(Error::engine("client disconnected"));
                            }
                            match events.try_send(event) {
                                Ok(()) => return Ok(()),
                                Err(mpsc::error::TrySendError::Closed(_)) => {
                                    return Err(Error::engine("client disconnected"));
                                }
                                Err(mpsc::error::TrySendError::Full(e)) => event = e,
                            }
                            if std::time::Instant::now() >= deadline {
                                return Err(Error::engine("response consumer timed out"));
                            }
                            std::thread::sleep(Duration::from_millis(2));
                        }
                    };
                    let result =
                        runner.run_prepared(&job.params, prepared, &cancelled, &mut |event| {
                            send(Ok(event))
                        });
                    if let Err(mut e) = result {
                        e.message = e.message.chars().take(500).collect();
                        let _ = events.try_send(Err(e));
                    }
                }
            })
            .map_err(Error::engine)?;
        Ok(Self {
            shared,
            jobs: Some(jobs),
            worker: Some(worker),
        })
    }
    pub fn is_ready(&self) -> bool {
        self.shared.loaded.load(Ordering::Acquire)
    }
    pub async fn call<B>(&self, r: Request<B>) -> Response<HttpBody>
    where
        B: Body<Data = Bytes> + Unpin,
        B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        handle(r, self).await
    }
}
impl Drop for Api {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        drop(self.jobs.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
pub struct Server {
    listener: TcpListener,
    api: Arc<Api>,
}
impl Server {
    pub async fn bind<F>(mut cfg: Config, factory: F) -> Result<Self>
    where
        F: FnOnce() -> Result<ChatRunner> + Send + 'static,
    {
        cfg.validate()?;
        let listener=TcpListener::bind((cfg.host.as_str(),cfg.port)).await.map_err(|e|Error::engine(format!("cannot listen on {}:{} ({e}). Is another server already using this port? Use --port to pick another.",cfg.host,cfg.port)))?;
        cfg.port = listener.local_addr().map_err(Error::engine)?.port();
        Ok(Self {
            listener,
            api: Arc::new(Api::new(cfg, factory)?),
        })
    }
    pub fn local_addr(&self) -> Result<std::net::SocketAddr> {
        self.listener.local_addr().map_err(Error::engine)
    }
    pub fn is_ready(&self) -> bool {
        self.api.is_ready()
    }
    pub async fn run(self, shutdown: impl Future<Output = ()>) -> Result<()> {
        tokio::pin!(shutdown);
        let permits = Arc::new(Semaphore::new(self.api.shared.cfg.limits.max_connections));
        let mut connections = JoinSet::new();
        let result = loop {
            tokio::select! {
                _=&mut shutdown=>break Ok(()),
                _=self.api.shared.failure_ready.notified()=>{if let Some(e)=self.api.shared.failure.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(){break Err(e);}} ,
                Some(_)=connections.join_next(),if !connections.is_empty()=>{},
                accepted=self.listener.accept()=> {
                    let (socket,_)=match accepted {Ok(v)=>v,Err(e)=>break Err(Error::engine(e))};
                    let Ok(permit)=permits.clone().try_acquire_owned() else {drop(socket);continue;};
                    let _=socket.set_nodelay(true);
                    let api=self.api.clone();
                    connections.spawn(async move {
                        let _permit=permit;
                        let limits=&api.shared.cfg.limits;
                        let io=TokioIo::new(IdleIo::new(socket,limits));
                        let mut builder=http1::Builder::new();
                        builder.timer(TokioTimer::new()).header_read_timeout(limits.header_timeout).max_headers(limits.max_headers).max_buf_size(limits.header_bytes);
                        let service=service_fn(move |r|{let api=api.clone();async move {Ok::<_,Infallible>(api.call(r).await)}});
                        let _=builder.serve_connection(io,service).await;
                    });
                }
            }
        };
        self.api.shared.shutdown.store(true, Ordering::Release);
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        drop(self.api);
        result
    }
}

fn header<'a, B>(r: &'a Request<B>, key: &str) -> Option<&'a str> {
    r.headers().get(key).and_then(|v| v.to_str().ok())
}
fn refused<B>(r: &Request<B>, cfg: &Config, post: bool) -> Option<&'static str> {
    if !local(&cfg.host) && cfg.api_key.as_ref().is_some_and(|s| !s.is_empty()) {
        return None;
    }
    let names = if local(&cfg.host) {
        vec!["127.0.0.1", "localhost", "[::1]"]
    } else {
        vec![cfg.host.as_str()]
    };
    let own: Vec<String> = names.iter().map(|h| format!("{h}:{}", cfg.port)).collect();
    // Duplicate security headers are ambiguous across intermediaries: refuse.
    if r.headers().get_all("host").iter().count() != 1
        || !own
            .iter()
            .any(|h| header(r, "host").is_some_and(|v| v.eq_ignore_ascii_case(h)))
    {
        return Some("host");
    }
    // An absolute-form target names its own authority: hold it to the same rule.
    let uri = r.uri();
    if let Some(authority) = uri.authority() {
        if uri.scheme_str() != Some("http")
            || !own
                .iter()
                .any(|h| authority.as_str().eq_ignore_ascii_case(h))
        {
            return Some("host");
        }
    }
    if post {
        if r.headers().get_all("origin").iter().count() > 1
            || (r.headers().contains_key("origin")
                && !own.iter().any(|h| {
                    header(r, "origin")
                        .is_some_and(|v| v.eq_ignore_ascii_case(&format!("http://{h}")))
                }))
        {
            return Some("origin");
        }
        if r.headers().get_all("sec-fetch-site").iter().count() > 1
            || (r.headers().contains_key("sec-fetch-site") && header(r, "sec-fetch-site").is_none())
            || !matches!(
                header(r, "sec-fetch-site")
                    .unwrap_or("none")
                    .to_ascii_lowercase()
                    .as_str(),
                "same-origin" | "none"
            )
        {
            return Some("cross-site");
        }
    }
    None
}
fn authorized<B>(r: &Request<B>, key: &str) -> bool {
    if r.headers().get_all("authorization").iter().count() != 1 {
        return false;
    }
    let Some(value) = header(r, "authorization").and_then(|s| s.strip_prefix("Bearer ")) else {
        return false;
    };
    // Fixed-size digest comparison with no prefix-dependent branches. black_box
    // prevents the optimizer replacing this reduction with an early-exit memcmp.
    let a = Sha256::digest(value.as_bytes());
    let b = Sha256::digest(key.as_bytes());
    let mut diff = 0_u8;
    for i in 0..32 {
        diff |= std::hint::black_box(a[i] ^ b[i]);
    }
    std::hint::black_box(diff) == 0
}
fn busy_error() -> Error {
    http_error(
        503,
        "the server is busy with other requests; retry later",
        "unavailable",
    )
}
fn busy(e: Error) -> Response<HttpBody> {
    let mut r = error_response(e);
    r.headers_mut().insert(
        "retry-after",
        hyper::header::HeaderValue::from_static(RETRY_AFTER),
    );
    r
}
const RETRY_AFTER: &str = "5";
enum BodyError {
    TooLarge,
    Busy,
}
/// Reads the body, charging every received byte to the global budget first.
async fn read_body<B>(
    body: &mut B,
    declared: usize,
    limit: usize,
    charge: &mut Charge,
) -> std::result::Result<Vec<u8>, BodyError>
where
    B: Body<Data = Bytes> + Unpin,
{
    // Sized once so the resident buffer matches the bytes charged (pages are
    // only touched as they are written).
    let mut out = Vec::with_capacity(declared.min(limit));
    while let Some(frame) = body.frame().await {
        let Ok(data) = frame.map_err(|_| BodyError::TooLarge)?.into_data() else {
            continue; // trailers
        };
        let len = out.len() + data.len();
        if len > limit {
            return Err(BodyError::TooLarge);
        }
        if !charge.resize(len) {
            return Err(BodyError::Busy);
        }
        out.extend_from_slice(&data);
    }
    Ok(out)
}
struct Decoded {
    params: crate::chat::ChatParams,
    stream: bool,
    include_usage: bool,
}
/// Validates and cleans a body. The charge grows to the estimated size of the
/// parsed and cleaned request before parsing, so a request the budget cannot
/// hold is never built. Synchronous: the parsed JSON never lives across an
/// await, and only one body is decoded at a time on a current-thread runtime.
fn decode(bytes: &[u8], cfg: &Config, charge: &mut Charge) -> Result<Decoded> {
    let limits = &cfg.limits;
    let estimate = |values: usize| {
        bytes
            .len()
            .saturating_add(values.saturating_mul(VALUE_COST))
    };
    let values = json_values(bytes);
    check_values(values, limits)?;
    if !charge.resize(estimate(values)) {
        return Err(busy_error());
    }
    let body: Value = serde_json::from_slice(if bytes.is_empty() { b"{}" } else { bytes })
        .map_err(|_| Error::request("the request body is not valid JSON"))?;
    // Adds the values that tool-call argument strings expand to.
    let values = check_size(&body, limits, values)?;
    if !charge.resize(estimate(values)) {
        return Err(busy_error());
    }
    let params = parse_chat_request(&body, &cfg.defaults)?;
    let text = text_bytes(&params.messages) + params.tools.as_ref().map_or(0, text_bytes);
    if text > limits.max_text_bytes {
        return Err(Error::request(format!(
            "the request holds more than {} bytes of text",
            limits.max_text_bytes
        )));
    }
    Ok(Decoded {
        params,
        stream: truthy(&body["stream"]),
        include_usage: truthy(&body["stream_options"]["include_usage"]),
    })
}
async fn handle<B>(mut r: Request<B>, api: &Api) -> Response<HttpBody>
where
    B: Body<Data = Bytes> + Unpin,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let shared = &api.shared;
    let cfg = &shared.cfg;
    let post = r.method() == Method::POST;
    if let Some(why) = refused(&r, cfg, post) {
        return close(error_response(http_error(
            403,
            format!("forbidden: {why}"),
            "forbidden",
        )));
    }
    let path = r.uri().path();
    if r.method() == Method::GET && path == "/health" {
        let loaded = shared.loaded.load(Ordering::Acquire);
        let mut body = json!({"status":if loaded{"ok"}else{"loading"},"version":env!("CARGO_PKG_VERSION"),"model":cfg.model_id});
        if loaded {
            body["max_context_tokens"] = json!(shared.capacity.load(Ordering::Acquire));
            body["thinking_budget"] = json!(cfg.defaults.thinking_budget);
        }
        return json_response(if loaded { 200 } else { 503 }, body);
    }
    if let Some(key) = cfg.api_key.as_ref().filter(|k| !k.is_empty()) {
        if !authorized(&r, key) {
            return close(error_response(http_error(
                401,
                "invalid or missing API key",
                "authentication_error",
            )));
        }
    }
    if r.method() == Method::GET {
        let model = json!({"id":cfg.model_id,"object":"model","owned_by":"local"});
        return if matches!(path, "/v1/models" | "/v1/models/") {
            json_response(200, json!({"object":"list","data":[model]}))
        } else if path == format!("/v1/models/{}", cfg.model_id) {
            json_response(200, model)
        } else {
            error_response(http_error(404, "not found", "not_found"))
        };
    }
    if !post {
        return close(error_response(http_error(
            501,
            format!("Unsupported method ('{}')", r.method()),
            "invalid_request_error",
        )));
    }
    if path != "/v1/chat/completions" {
        return close(error_response(http_error(404, "not found", "not_found")));
    }
    if !shared.loaded.load(Ordering::Acquire) {
        return close(error_response(http_error(
            503,
            "the model is still loading",
            "unavailable",
        )));
    }
    // Upstream requires Content-Length and doesn't implement chunked requests.
    let len = header(&r, "content-length").and_then(|s| s.parse::<usize>().ok());
    if len.is_some_and(|n| n > cfg.limits.max_body) || r.headers().contains_key("transfer-encoding")
    {
        return close(error_response(http_error(
            413,
            "request body too large or length missing",
            "invalid_request_error",
        )));
    }
    let Some(mut charge) = Charge::new(&shared.budget) else {
        return close(busy(busy_error()));
    };
    let bytes = match tokio::time::timeout(
        cfg.limits.body_timeout,
        read_body(
            r.body_mut(),
            len.unwrap_or(0),
            cfg.limits.max_body,
            &mut charge,
        ),
    )
    .await
    {
        Err(_) => {
            return close(error_response(http_error(
                408,
                "request body timed out",
                "invalid_request_error",
            )));
        }
        Ok(Err(BodyError::Busy)) => return close(busy(busy_error())),
        Ok(Err(BodyError::TooLarge)) => {
            return close(error_response(http_error(
                413,
                "request body too large or length missing",
                "invalid_request_error",
            )));
        }
        Ok(Ok(b)) => b,
    };
    // The raw body and the parsed JSON are gone once the cleaned request exists.
    let decoded = decode(&bytes, cfg, &mut charge);
    drop(bytes);
    let decoded = match decoded {
        Ok(d) => d,
        Err(e) if e.status == 503 => return busy(e),
        Err(e) => return error_response(e),
    };
    let envelope = Envelope::new(&cfg.model_id);
    let cancel = Cancel(Arc::new(AtomicBool::new(false)));
    let (ready, rx) = oneshot::channel();
    let Some(jobs) = api.jobs.as_ref() else {
        return error_response(Error::engine("engine worker is unavailable"));
    };
    let job = Job {
        params: decoded.params,
        ready,
        cancel: cancel.0.clone(),
        _charge: charge,
    };
    // On timeout or disconnect `cancel` drops, and the engine thread skips
    // the job when it reaches it.
    let waited = tokio::time::timeout(cfg.limits.queue_timeout, async {
        jobs.send(job).await.ok()?;
        rx.await.ok()
    })
    .await;
    let events = match waited {
        Err(_) => {
            return busy(http_error(
                503,
                "the request waited too long for the engine; retry later",
                "unavailable",
            ));
        }
        Ok(Some(Ok(rx))) => rx,
        Ok(Some(Err(e))) => return error_response(e),
        Ok(None) => return error_response(Error::engine("engine worker is unavailable")),
    };
    if decoded.stream {
        let stream = EventBody {
            events,
            cancel,
            envelope,
            include_usage: decoded.include_usage,
            calls: 0,
            pending: std::collections::VecDeque::new(),
            ended: false,
        };
        let mut stream = stream;
        stream.pending.push_back(sse(stream.envelope.chunk(
            json!({"role":"assistant"}),
            None,
            None,
        )));
        let mut r = response(200, "text/event-stream; charset=utf-8", stream.boxed());
        r.headers_mut().insert(
            "cache-control",
            hyper::header::HeaderValue::from_static("no-cache"),
        );
        return close(r);
    }
    let mut events = events;
    let mut reason = String::new();
    let mut content = String::new();
    let mut calls = vec![];
    while let Some(event) = events.recv().await {
        match event {
            Err(e) => return error_response(e),
            Ok(Event::Reasoning(t)) => reason.push_str(&t),
            Ok(Event::Content(t)) => content.push_str(&t),
            Ok(Event::ToolCall(c)) => calls.push(tool_call_json(c, None)),
            Ok(Event::Done(u)) => {
                return json_response(200, envelope.completed(reason, content, calls, u));
            }
        }
    }
    drop(cancel);
    error_response(Error::engine("engine worker ended without a completion"))
}
struct EventBody {
    events: mpsc::Receiver<Result<Event>>,
    cancel: Cancel,
    envelope: Envelope,
    include_usage: bool,
    calls: usize,
    pending: std::collections::VecDeque<Bytes>,
    ended: bool,
}
impl Body for EventBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<std::result::Result<Frame<Bytes>, Infallible>>> {
        let this = self.get_mut();
        if let Some(bytes) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(Frame::data(bytes))));
        }
        if this.ended {
            return Poll::Ready(None);
        }
        match this.events.poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(event) => {
                match event {
                    Some(Ok(Event::Reasoning(t))) => this.pending.push_back(sse(this
                        .envelope
                        .chunk(json!({"reasoning_content":t}), None, None))),
                    Some(Ok(Event::Content(t))) => this
                        .pending
                        .push_back(sse(this.envelope.chunk(json!({"content":t}), None, None))),
                    Some(Ok(Event::ToolCall(c))) => {
                        let c = tool_call_json(c, Some(this.calls));
                        this.calls += 1;
                        this.pending.push_back(sse(this.envelope.chunk(
                            json!({"tool_calls":[c]}),
                            None,
                            None,
                        )));
                    }
                    Some(Ok(Event::Done(u))) => {
                        this.pending.push_back(sse(this.envelope.chunk(
                            json!({}),
                            Some(u.finish),
                            None,
                        )));
                        if this.include_usage {
                            this.pending.push_back(sse(this.envelope.chunk(
                                json!({}),
                                None,
                                Some(usage_json(&u)),
                            )));
                        }
                        this.pending
                            .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                        this.ended = true;
                    }
                    Some(Err(e)) => {
                        this.pending.push_back(sse(error_value(&e)));
                        this.pending
                            .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                        this.ended = true;
                    }
                    None => {
                        this.pending.push_back(sse(error_value(&Error::engine(
                            "engine worker ended without a completion",
                        ))));
                        this.pending
                            .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                        this.ended = true;
                    }
                }
                let _ = &this.cancel; // held through the final frame, dropped on disconnect
                Poll::Ready(this.pending.pop_front().map(|b| Ok(Frame::data(b))))
            }
        }
    }
}

// Idle deadline covers both reads and writes, including persistent connections
// and a peer that stops reading a stream. A boxed Sleep keeps pinning safe.
struct IdleIo {
    socket: TcpStream,
    sleep: Pin<Box<Sleep>>,
    timeout: Duration,
    linger_bytes: usize,
    linger_timeout: Duration,
    // Some once shutdown has half-closed: the bytes still allowed to discard.
    linger_left: Option<usize>,
}
impl IdleIo {
    fn new(socket: TcpStream, limits: &Limits) -> Self {
        Self {
            socket,
            sleep: Box::pin(tokio::time::sleep(limits.idle_timeout)),
            timeout: limits.idle_timeout,
            linger_bytes: limits.linger_bytes,
            linger_timeout: limits.linger_timeout,
            linger_left: None,
        }
    }
    fn expired(&mut self, cx: &mut Context<'_>) -> bool {
        self.sleep.as_mut().poll(cx).is_ready()
    }
    fn progress(&mut self) {
        self.sleep.as_mut().reset(Instant::now() + self.timeout);
    }
}
impl AsyncRead for IdleIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.expired(cx) {
            return Poll::Ready(Err(io::ErrorKind::TimedOut.into()));
        }
        let before = buf.filled().len();
        let r = Pin::new(&mut this.socket).poll_read(cx, buf);
        if buf.filled().len() > before {
            this.progress();
        }
        r
    }
}
impl AsyncWrite for IdleIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if this.expired(cx) {
            return Poll::Ready(Err(io::ErrorKind::TimedOut.into()));
        }
        let r = Pin::new(&mut this.socket).poll_write(cx, buf);
        if matches!(&r,Poll::Ready(Ok(n)) if *n>0) {
            this.progress();
        }
        r
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.expired(cx) {
            return Poll::Ready(Err(io::ErrorKind::TimedOut.into()));
        }
        Pin::new(&mut this.socket).poll_flush(cx)
    }
    // Hyper calls this after flushing every response it ends a connection with,
    // including its own 400/431 parse-error responses. Closing a socket with
    // unread received bytes makes the kernel send RST, which can destroy the
    // response before the peer reads it; so half-close, then drain until the
    // peer's EOF, a byte budget or a deadline, whichever comes first.
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.linger_left.is_none() {
            ready!(Pin::new(&mut this.socket).poll_shutdown(cx))?;
            this.sleep
                .as_mut()
                .reset(Instant::now() + this.linger_timeout);
            this.linger_left = Some(this.linger_bytes);
        }
        let mut scratch = [0_u8; 4096];
        while let Some(left) = this.linger_left.filter(|&n| n > 0) {
            if this.expired(cx) {
                break;
            }
            let mut buf = ReadBuf::new(&mut scratch[..left.min(4096)]);
            match Pin::new(&mut this.socket).poll_read(cx, &mut buf) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(())) if !buf.filled().is_empty() => {
                    this.linger_left = Some(left - buf.filled().len());
                }
                // EOF or a reset: nothing is left to protect.
                Poll::Ready(_) => break,
            }
        }
        Poll::Ready(Ok(()))
    }
}
