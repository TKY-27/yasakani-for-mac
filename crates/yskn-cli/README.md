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

`Engine` is synchronous and object safe, with context capacity, reset, prompt
reuse within generate, sampling, max-new-token and EOS control, a cancellable
committed-token callback, and generation statistics. It has no Send bound. The
server factory creates the runner on one owning thread; a bounded channel
serializes requests, with later requests waiting. `Api::call` is also usable as
an embedded Hyper service. Consumer failure, body drop and shutdown cancel via
the callback. Tests use a scripted, loopable, delayed, failure-injectable stub
with prompt-cache observations, matching the upstream fixture's responsibilities.

## Security and transport limits

A non-loopback bind **is refused without a key**, as in upstream cli.py. On
loopback, Host must be localhost/127.0.0.1/[::1] with the actual bound port;
POST Origin must be the same HTTP origin, and Sec-Fetch-Site must be same-origin
or none. As upstream does, a keyed remote bind bypasses Host/Origin/Fetch checks
and still requires the key. API-key comparison hashes both byte strings with
SHA-256 and reduces all 32 digest bytes without prefix-dependent branches.

| Limit | Default |
|---|---:|
| Actual request-body bytes, before JSON parsing | 32 MiB |
| Concurrent connections | 64; excess connections are closed |
| Queued engine jobs | 64 (bounded by active HTTP connections) |
| Buffered events per response | 8 |
| Header read buffer | 32 KiB |
| Header count | 100 |
| Complete header deadline | 10 s |
| Complete body deadline | 30 s |
| Idle socket read/write deadline | 300 s |
| Full event-channel backpressure deadline | 30 s |
| Context reserve | 64 tokens |

Library embedders may configure `Limits`; CLI defaults are fixed. Protection
headers are X-Frame-Options DENY, CSP frame-ancestors 'none',
Cross-Origin-Resource-Policy same-origin and X-Content-Type-Options nosniff.
No CORS headers, TLS, file serving, outgoing HTTP, or prompt/answer logging.

## Differences and verification boundary

Rust uses Hyper HTTP/1 framing and compact outer JSON instead of Python's
HTTPServer/json.dumps spacing; JSON strings containing tool arguments retain
Python spacing/float spelling. Unsupported methods retain status 501 but use
JSON errors instead of Python's HTML. Early rejected POSTs close rather than
drain a potentially slow body; chunked requests are explicitly rejected (413),
where Python never implemented their decoding. Hyper rejects malformed framing
before the handler, using its own 400/431/connection-close behavior.

Duplicate security headers are rejected to avoid intermediary ambiguity. Added
header/body/connection/event limits defend the hand-built transport. Non-finite
numbers and integers outside i64 are rejected; Python accepts NaN/Infinity and
arbitrary precision integers. Non-object template kwargs/stream_options and
invalid effort types become 400 instead of Python conversion/attribute errors;
all template failures become 400. These are deliberate validation defect fixes.

Windows console/DLL/CUDA/GPU introspection and tuning environment changes have
no equivalent here. Info reports this static Rust binary, stub and tokenizer.
The explicit stub needs no model folder; `--model-dir` loads only text files.
Terminal output is plain, without optional ANSI coloring. Verify/bench/selfcheck
and actual model generation are outside this brief's product integration.

Tests run automatically through Cargo/CI with synthetic assets, including real
TCP tests on ephemeral ports; none require weights or a real tokenizer. Golden
response bodies normalize only id/timestamp fields, and were constructed from
the fixed protocol/byte-token contract, not captured from the implementation.
See [REPORT.md](REPORT.md) for the upstream test mapping and this run's exact
proof results. A sandbox that forbids TCP bind must report the socket tests as
blocked failures, never silently skip them or claim a pass.
