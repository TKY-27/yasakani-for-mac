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
`--no-loop-guard`, `--raw`. Plain stdin/stdout supports pipes. Model text,
reasoning text and error messages are printed with C0 control characters other
than newline and tab, DEL and C1 controls (U+0080 to U+009F) shown as `\xNN`,
and the bidirectional overrides, isolates and marks (U+061C, U+200E, U+200F,
U+202A to U+202E, U+2066 to U+2069) as `\uNNNN`, so escape sequences in model
output cannot drive the terminal or reorder what it shows; the zero-width
joiner and variation selectors stay, as emoji need them. Nothing else is
escaped. `--raw` prints them unchanged. The terminal's own echo of what the
user types is not filtered: it is shown before the program reads the line. Errors that end the program (a
model folder path, a bad argument, a load error) are always printed this way,
also with `--raw`. The HTTP API is unaffected: its text is JSON-encoded. Slash commands are
`/exit`, `/quit`, `/q`, `/reset`, `/think on|off`, `/effort low|medium|xhigh`,
`/help`. Thinking appears between `[thinking]` and `[/thinking]` when an answer
follows. Ctrl+C stops the current generation or exits at the input prompt; the
terminal thread does not hold the standard output lock while it waits for input.
Tests cover the prompt case by sending SIGINT to a `yskn chat` process on pipes,
and the generation case through `chat_loop` with an injected cancel flag; Ctrl+C
typed at a real terminal (which sends the same SIGINT) is checked by hand only.
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
  Template kwargs have precedence for thinking/effort; they cannot set what
  the server sets itself (`messages`, `tools`, `add_generation_prompt`,
  `bos_token`, `eos_token`, `raise_exception`: 400 naming the key). `minimal` aliases `low`,
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
failure, body drop and shutdown cancel through the callback. `ChatRunner` also
checks for a cancel before each `generate` call (the first, and the loop-guard
bridge's second): a request cancelled after its prompt was prepared ends with
the `length` finish without reaching the engine or advancing the seed counter,
so the engine's cached sequence is left as the previous request left it. Tests use a
scripted, loopable, delayed, failure-injectable stub with prompt-cache
observations, matching the upstream fixture's responsibilities.

## Security and transport limits

A non-loopback bind **is refused without a key**, as in upstream cli.py. On
loopback, Host must be localhost/127.0.0.1/[::1] with the actual bound port;
POST Origin must be the same HTTP origin, and Sec-Fetch-Site must be same-origin
or none. An absolute-form request target (`GET http://host:port/path`) must name
an allowed authority with the `http` scheme as well. As upstream does, a keyed
remote bind bypasses Host/Origin/Fetch checks and still requires the key, and
`/health` answers without the key on any bind. API-key comparison hashes both byte strings with
SHA-256 and reduces all 32 digest bytes without prefix-dependent branches.

| Limit (`Limits` field) | Default | Why this number |
|---|---:|---|
| Actual request-body bytes, before JSON parsing (`max_body`) | 8 MiB | Upstream's `MAX_BODY` is 32 MiB. Lowered: by our estimate a full 262,144-token context is about 1 to 2.4 MiB of JSON (3 to 9 bytes per token, ASCII-escaped CJK at the top), so 8 MiB leaves 32 bytes per token. A client resending megabytes of old `reasoning_content` that the template drops could exceed it where upstream accepted it |
| Bytes held by all requests together (`max_pending_bytes`) | 512 MiB | Charged as body bytes arrive; raised before parsing to what decoding can hold at most, 5 x the body bytes + 640 per JSON value (see below); lowered after decoding to the cleaned request's text bytes + 640 per value, which it holds while it waits and generates. Over budget: 503 with `Retry-After: 5`. Must hold one largest request (5 x 8 MiB + 2^19 x 640 = 360 MiB) |
| JSON values per request, counted before parsing (`max_json_values`) | 524,288 | Bounds the parsed body and its cleaned copy (measured worst case about 590 bytes per value, for tool schemas of one-member objects). Covers 65,536 messages at 4 values each with room for tools; tool-call argument strings count too |
| Messages (`max_messages`) | 65,536 | Even an empty message renders as 5 template tokens, so more than 262,144 / 5 = 52,429 can never fit |
| Content parts per message (`max_content_parts`) | 1,024 | Clients send a handful of text parts per message |
| Tool calls per assistant message (`max_tool_calls`) | 1,024 | Each call renders as at least about 10 tokens of tags; parallel calls come in single digits |
| Tools (`max_tools`) | 1,024 | OpenAI documents at most 128 functions; 8 times that |
| Text the template can render from the request (`max_text_bytes`): the bytes of every string and key in the cleaned messages, tools and template kwargs, plus 24 bytes for each number, boolean or null (the longest spelling of a number, `-2.2250738585072014e-308`) | 2 MiB | A full 262,144-token context of ordinary text is under 1 MiB, so a request beyond 2 MiB could not fit anyway. Checked once the request is decoded, before it is queued: it also bounds what Python-style `str()` of non-string text parts and parsed tool arguments can add, and what the request holds while it waits |
| Rendered prompt (`max_prompt_bytes`) | 2 MiB | The same reasoning as the text limit, applied to the template's output, which adds tags and JSON punctuation to the counted text. Checked on the engine thread between rendering and tokenizing, because tokenizing costs time and memory in proportion to the prompt (measured below). Over it: 400 `context_length_exceeded`, naming the rendered size and the limit |
| Wait for the engine, from queueing until generation starts (`queue_timeout`) | 600 s | The OpenAI Python SDK's default request timeout is 600 s, so a client would give up by then anyway. Exceeded: 503 with `Retry-After: 5`, and the request never reaches the engine |
| Concurrent connections (`max_connections`, 1..1024) | 64; excess connections are closed unserved | One engine serves one request at a time |
| Queued engine jobs | 64 (bounded by active HTTP connections) | One per connection |
| Buffered events per response | 8 | Backpressure reaches the engine quickly |
| Header read buffer (`header_bytes`, at least 8192) | 32 KiB | Browser and SDK headers are a few KiB |
| Header count (`max_headers`) | 100 | As Python's `http.client`, which upstream's server uses |
| Complete header deadline (`header_timeout`) | 10 s | Slow-header clients cannot hold connections |
| Complete body deadline (`body_timeout`) | 30 s | 8 MiB at loopback speed takes milliseconds |
| Idle socket read/write deadline (`idle_timeout`) | 300 s | Long enough for a keep-alive client between requests |
| Full event-channel backpressure deadline (`stream_timeout`) | 30 s | A stream that is not read stops the engine |
| Bytes discarded while closing (`linger_bytes`) | 256 KiB | See closing below |
| Closing deadline (`linger_timeout`) | 2 s | See closing below |
| Context reserve | 64 tokens | As upstream |
| Stop strings per request | 4 | As upstream; OpenAI documents "up to 4 sequences" |
| Bytes per stop string (`max_stop_bytes`) | 1,024 | OpenAI documents no length and upstream has none; stop markers are short. The stop filter's work per output piece is bounded by the text it holds back, as upstream's `StopFilter` is |
| Loop guard: longest period, repeat run in thinking / answer, digit run in thinking | 16 tokens; 256 / 1024 tokens; 1000 tokens | As upstream |

Generation itself has no deadline: it is bounded by `max_tokens` (and the
context room), and it stops when the client disconnects or stops reading a
stream (`stream_timeout`). A client that disconnects while its request waits
in the queue is skipped by the engine thread. `queue_timeout` also covers
preparing the prompt: the engine thread checks for a disconnect, the deadline
and shutdown before rendering the template, after rendering and after
tokenizing, and stops there (past the deadline: 503 with `Retry-After`).
Rendering and tokenizing themselves cannot be interrupted; see below for what
the longest of them costs.

Whether a prompt fits is decided only by the exact token check after
tokenization, as upstream (400 `context_length_exceeded`). The prompt byte
limit before it bounds the cost of tokenizing, not whether a prompt fits: a
tokenizer's normalizer may delete any amount of text (`StripAccents` turns
10,000 combining marks and "hi" into a 23-token prompt), so a rendered prompt
over 2 MiB is refused even if it would normalize to fit. Upstream tokenizes
any prompt.

**What a request holds.** While a body is decoded its text exists at most
twice at once: the raw body and the parsed JSON while parsing (the raw body is
dropped as soon as parsing ends), then the parsed JSON and the cleaned request
while cleaning. Cleaning moves nothing large twice (no temporary copies), but
it can make text longer: Python-style `str()` of a non-string text part writes
a raw DEL byte as the four characters `\x7f`. So at its peak a request holds
at most 5 times its body bytes (`DECODE_FACTOR`), plus the structure of its
values, which the 640 bytes per value cover; that is what it is charged while
it is decoded. Afterwards it holds its cleaned text (at most `max_text_bytes`
for messages, tools and template kwargs, plus stop strings). Measured on
macOS 27.2 on an Apple M5 with the release binary, peak resident memory of the
whole server while 64 clients each send one request with a 7 MiB string and
never read the answer (`/usr/bin/time -l`; an idle server is 8 MiB):

| 7 MiB string in | Before this round | After |
|---|---:|---:|
| `response_format.type` (400; the old message quoted it) | 2,115 MiB | 470 MiB |
| a non-string text part, `[string]` (400; text limit) | 1,162 MiB | 451 to 497 MiB (two runs) |
| `chat_template_kwargs` (200; reaches the engine) | 485 MiB | 484 MiB |

All three are dominated after the fix by the 64 bodies being received at once
(64 x 7 MiB = 448 MiB, charged as they arrive). One such request alone peaks
at 14 MiB above an idle server for `response_format`, 42 MiB (5.7 x the body)
for the text part, where the allocator keeps the freed raw body resident, and
28 MiB for template kwargs, of which the engine thread's copies for the
template are the larger part (not charged, see below).

**What is bounded by what.**

- Bytes held, by the budget (`max_pending_bytes`): every request from its
  first body byte until it leaves the engine thread, as described above.
- Text counted, by the request limits (`max_text_bytes` and the shape limits
  in the table): everything the template can render from the request, checked
  once it is decoded and before it is queued.
- The rendered prompt, by `max_prompt_bytes`: the template's output, checked
  on the engine thread after rendering and before tokenizing.
- Tokens, by the context check after tokenizing: whether the prompt fits.

Not charged to the budget, and done on the engine thread for one request at
a time: rendering and tokenizing. Tokenizing is proportional to the rendered
prompt, so the prompt limit bounds it. Measured on the same machine with the
synthetic byte tokenizer (one token per byte, the most tokens a prompt byte
can make): preparing a prompt just under the limit makes 2.1 million tokens
and takes 0.11 s, with a peak of 304 to 308 MiB allocated (about 150 bytes
per token, alike for ASCII, CJK and emoji text and for tool-call tags;
counted with an allocator wrapper in a test build); the whole release server
peaked at 388 MiB resident for ASCII text and 494 MiB for emoji text
(`/usr/bin/time -l`; idle 8 MiB) and answered in 0.13 s. Rendering happens
before the prompt check, so its output is bounded only by the counted text
and how much the template adds to it: the most expansive request we found,
290,000 empty tool-call arguments (4.6 MB body, 2.0 MB counted text), renders
9.9 MB in 0.09 s with a peak of 93 MiB allocated, and is then refused without
being tokenized. These are measurements of constructed cases, not a proven
bound; other templates add other tags. A real tokenizer makes fewer tokens
per byte; it has not been measured here. Embedders running `Api` on a
multi-thread runtime can decode several bodies at once; each is still charged
before it is parsed.

Error messages never quote more than a short excerpt of a request value: at
most 64 characters of its Python repr followed by `…` (`response_format`, the
method of a 501), or, as upstream, the first 80 characters of its JSON text
(`tool_choice`); control characters appear escaped in both. The repr or JSON
text is written into a sink that stops the writer once it has the excerpt, so
quoting a value costs no more than the excerpt, however large or nested the
value is (measured: 434 bytes allocated for the 80-character excerpt of an
8.3 MB `tool_choice` of 25,600 strings of 80 emoji, which allocated 151 MiB
when the excerpt was cut from a shortened copy of the value). Template and
tokenizer errors are cut to 500 characters. An error response is therefore a
few hundred bytes whatever the request held, and a client that never reads a
response is cut off by `idle_timeout`, which also bounds a write that makes no
progress, and the connection's permit is returned.

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

The body limit is 8 MiB instead of upstream's 32 MiB, and the request shape,
text, rendered prompt and global budget limits in the table above have no
upstream equivalent; upstream parses any body within its limit and tokenizes
its prompt in full.

Duplicate Host/Origin/Sec-Fetch-Site/Authorization headers are rejected to
avoid intermediary ambiguity; Python reads the first. An empty Sec-Fetch-Site is
rejected; Python treats it as absent (`none`). An absolute-form target is
checked as above and then routed by its path; Python compared the whole target
with the route and answered 404. JSON nested deeper than 128 levels (serde_json's
recursion limit) and strings with a lone surrogate escape (`"\ud800"`) are
"not valid JSON" (400); Python accepts lone surrogates, and nesting up to
its interpreter recursion limit. Integer strings are parsed by Rust rules, so
`"1_000"` is rejected where Python's `int()` accepts it.
`YASAKANI_THINKING_BUDGET` also sets the terminal chat's thinking budget, not
only the server's default. Added header/body/connection/event limits defend
the hand-built transport. The digest key comparison above takes the role of
upstream's `hmac.compare_digest` with dependencies already in the lockfile.
Non-finite numbers and integers outside i64 are rejected; Python accepts
NaN/Infinity and arbitrary precision integers. Integer seeds within i64 do not
pass through f64, so they keep full precision. Non-object template
kwargs/stream_options and invalid effort types become 400 instead of Python
conversion/attribute errors; all template failures become 400. Upstream passes
every template kwarg on as a keyword argument: a `tools` kwarg reaches the
template whenever the request has no tools of its own (so it escapes the tool
limit and `tool_choice: "none"`), `messages`, `add_generation_prompt`,
`bos_token` and `eos_token` fail with a Python TypeError outside its error
handling, and `raise_exception` replaces the template's function; here all
six are a 400, checked again by `ChatFormatter::render` for every caller.
These are deliberate validation defect fixes.

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
