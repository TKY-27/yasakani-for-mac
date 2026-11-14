# yskn chat and HTTP API

One Rust binary; no Python runtime. The real generation engine is not connected
in this brief. `--engine stub` explicitly selects deterministic demo output;
without it `chat`, `serve` and `info` fail clearly. No weights are opened.

```sh
CARGO_NET_OFFLINE=true cargo run -p yskn -- chat --engine stub
CARGO_NET_OFFLINE=true cargo run -p yskn -- serve --engine stub --port 8765
CARGO_NET_OFFLINE=true cargo run -p yskn -- info --engine stub
cargo run -p yskn -- serve --engine stub --port 0 --help
```

`--help` lists the hand-parsed options. Common options: `--model-dir` (optional
read-only tokenizer/template folder; defaults to the embedded synthetic byte
tokenizer), `--max-ctx` (262144), `--seed0` (1). The synthetic template is only
for the stub; it makes no model-format fidelity claim.

`serve`: `--host` (127.0.0.1), `--port` (8765; 0 = ephemeral), `--api-key`
(or `YASAKANI_API_KEY`), `--default-max-tokens` (32768), `--max-tokens-cap`,
`--no-warmup`. `YASAKANI_THINKING_BUDGET` supplies the default thinking budget
(0 = unlimited; invalid/negative values become 0).

`chat`: `--system`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`,
`--seed`, `--no-think`, `--effort low|medium|xhigh`, `--hide-thinking`,
`--no-loop-guard`. Plain stdin/stdout supports pipes. Slash commands are
`/exit`, `/quit`, `/q`, `/reset`, `/think on|off`, `/effort low|medium|xhigh`,
`/help`. Thinking appears between `[thinking]` and `[/thinking]` when an answer
follows. Ctrl+C stops the current generation or exits at the input prompt.
`bench` and `selfcheck` are recognized and report `not available yet`.

## HTTP surface

- `GET /health`: 503 while loading, 200 when ready; no key required.
- `GET /v1/models`, `/v1/models/`, `/v1/models/qwen3.8-27b`.
- `POST /v1/chat/completions`: OpenAI messages, text parts, assistant tool-call
  history and tool responses; non-streaming JSON or `stream: true` SSE.
- SSE emits the assistant role, per-token text, complete indexed tool calls,
  finish reason, optional final usage (`stream_options.include_usage`), `[DONE]`.
- Reasoning uses `reasoning_content`. Usage includes prompt/completion/total,
  cached prompt tokens and reasoning completion tokens. Guard reports are
  attached as `usage.loop_guard`.
- Sampling: `temperature`, `top_p`, `top_k` (engine normalizes to 1..20),
  `min_p`, `seed`; limits: `max_tokens`/`max_completion_tokens`, up to four
  `stop` strings. `ignore_eos`, `loop_guard`, `thinking_budget`, `thinking`,
  `enable_thinking`, `reasoning_effort`, `chat_template_kwargs` are supported.
  Template kwargs have precedence for thinking/effort. `minimal` aliases `low`,
  `high` aliases `xhigh`. `tools` supports functions with `tool_choice` auto/none.
  `parallel_tool_calls` is accepted as upstream does; calls are returned in order.
- Images, forced tool choices, legacy functions/function_call, logprobs, n > 1
  and non-text response formats are rejected. Context errors are HTTP 400 with
  `error.code = "context_length_exceeded"`, before streaming headers.

## Engine contract

```rust
pub trait Engine {
    fn context_capacity(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stop_ids: Option<&[u32]>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation>;
}
```

- Synchronous, object safe and deliberately not `Send`. The server's factory
  (which is `Send`) builds the runner and engine on one dedicated thread, and
  every engine call stays on that thread.
- `generate` reuses a compatible prompt prefix or resets its sequence itself.
  It reserves the 64-token context margin, clamps to the room left and rejects
  an empty or oversize prompt.
- `max_new` counts the first predicted token and any EOS.
- `on_tokens` receives the growing array of committed ids, including the final
  ids, and is called even for chunks that decode to no text. Returning `false`
  stops at the next commit boundary; an error stops generation and is
  propagated once the engine has quiesced.
- `Sampling` carries temperature, top_p, top_k, min_p and seed; greedy
  defaults and the top_k range (1..20) follow upstream. `Generation` returns
  the ids, finish (`Stop`, `Length`, `Cancelled`), cached tokens, snapshot
  mode, passes, prefill and generation milliseconds.
- `ChatRunner` turns this into `Reasoning`, `Content`, `ToolCall` and
  `Done(Usage)` events.

A bounded channel serializes requests to the engine thread, with later requests
waiting. `Api::call` is also usable as an embedded Hyper service. Consumer
failure, body drop and shutdown cancel through the callback. Tests use a
scripted, loopable, delayed, failure-injectable stub with prompt-cache
observations, matching the upstream fixture's responsibilities.

## Security and transport limits

A non-loopback bind **is refused without a key**, as in upstream cli.py. On
loopback, Host must be localhost/127.0.0.1/[::1] with the actual bound port;
POST Origin must be the same HTTP origin, and Sec-Fetch-Site must be same-origin
or none. As upstream does, a keyed remote bind bypasses Host/Origin/Fetch checks
and still requires the key. API-key comparison hashes both byte strings with
SHA-256 and reduces all 32 digest bytes without prefix-dependent branches.

| Limit (`Limits` field) | Default |
|---|---:|
| Actual request-body bytes, before JSON parsing (`max_body`) | 32 MiB |
| Concurrent connections (`max_connections`, 1..1024) | 64; excess connections are closed unserved |
| Queued engine jobs | 64 (bounded by active HTTP connections) |
| Buffered events per response | 8 |
| Header read buffer (`header_bytes`, at least 8192) | 32 KiB |
| Header count (`max_headers`) | 100 |
| Complete header deadline (`header_timeout`) | 10 s |
| Complete body deadline (`body_timeout`) | 30 s |
| Idle socket read/write deadline (`idle_timeout`) | 300 s |
| Full event-channel backpressure deadline (`stream_timeout`) | 30 s |
| Bytes discarded while closing (`linger_bytes`) | 256 KiB |
| Closing deadline (`linger_timeout`) | 2 s |
| Context reserve | 64 tokens |
| Stop strings per request | 4 |
| Loop guard: longest period, repeat run in thinking / answer, digit run in thinking | 16 tokens; 256 / 1024 tokens; 1000 tokens |

Default bind is 127.0.0.1:8765 and the default `max_tokens` is 32768.

**Closing a connection.** Whenever the server ends a connection after a
response, it flushes the response, shuts down its write side, then reads and
discards what the client still sends until the client closes, `linger_bytes`
have been discarded, or `linger_timeout` passes, whichever is first. Closing a
socket that still has unread request bytes makes the kernel send a reset, which
can destroy the response before the client reads it; early rejections (403,
401, 404, 408, 413, 501) and Hyper's own 400/414/431 responses would otherwise
lose their answer. This is done in the connection's IO wrapper, which Hyper 1.12
shuts down after flushing every response it ends a connection with, including
its parse-error responses. A header timeout ends without a response: Hyper
returns that error without writing one. The connection keeps its permit while
it lingers, so the connection cap also bounds lingering clients.

Library embedders may configure `Limits`; CLI defaults are fixed. Protection
headers are X-Frame-Options DENY, CSP frame-ancestors 'none',
Cross-Origin-Resource-Policy same-origin and X-Content-Type-Options nosniff.
No CORS headers, TLS, file serving, outgoing HTTP, or prompt/answer logging.

## Differences and verification boundary

Rust uses Hyper HTTP/1 framing and compact outer JSON instead of Python's
HTTPServer/json.dumps spacing; JSON strings containing tool arguments retain
Python spacing/float spelling. Unsupported methods retain status 501 but use
JSON errors instead of Python's HTML. Early rejected POSTs answer and close
with the bounded linger above instead of reading a potentially slow body;
chunked requests are explicitly rejected (413), where Python never implemented
their decoding. Hyper rejects malformed framing before the handler, using its
own 400/414/431 responses and closing the connection.

Duplicate Host/Origin/Sec-Fetch-Site/Authorization headers are rejected to
avoid intermediary ambiguity. Added header/body/connection/event limits defend
the hand-built transport. The digest key comparison above takes the role of
upstream's `hmac.compare_digest` with dependencies already in the lockfile.
Non-finite numbers and integers outside i64 are rejected; Python accepts
NaN/Infinity and arbitrary precision integers. Integer seeds within i64 do not
pass through f64, so they keep full precision. Non-object template
kwargs/stream_options and invalid effort types become 400 instead of Python
conversion/attribute errors; all template failures become 400. These are
deliberate validation defect fixes.

Windows console code page and coloring, DLL/CUDA/device selection, nvidia-smi
GPU-memory warnings and CUDA tuning environment changes have no equivalent
here. Info reports this static Rust binary, stub and tokenizer. The explicit
stub needs no model folder; `--model-dir` loads only text files, and nothing
substitutes for the real engine implicitly. Warmup is a short trait
`generate` plus `reset`, not a native CUDA warmup. Terminal output is plain
line I/O. Command-line argument errors exit with status 1. `verify` is not
ported; `bench` and `selfcheck` are deferred and only report that they are not
available yet. Actual model generation is outside this crate's integration so
far.

Tests run automatically through Cargo/CI with synthetic assets, including real
TCP tests on ephemeral ports; none require weights or a real tokenizer. Each
Rust test ported from upstream carries a comment naming its upstream case
(`test_chat.py`, `test_server.py`, `test_tools.py`, `test_cli.py`). Upstream
cases not ported: `test_cli` verify, bench and selfcheck round trip (commands
not available) and the GPU memory warning (Windows/CUDA only). Golden response
bodies normalize only id/timestamp fields, and were constructed from the fixed
protocol/byte-token contract, not captured from the implementation. Socket
tests synchronize on responses, handshakes or a polled condition, never on a
fixed sleep; their 30 s bounds are reached only by a failure, and the deadline
under test is set small through `Limits`. A sandbox that forbids TCP bind must report the socket tests
as blocked failures, never silently skip them or claim a pass.
