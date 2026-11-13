# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""OpenAI-compatible HTTP server (chat completions, streaming) around one engine.

* Listens on 127.0.0.1 by default.  The ``Host`` header must name the address the server is bound to, and a request that carries an
  ``Origin`` must come from that same address (so a web page open in your browser cannot talk to the server).  To serve other
  machines use ``--host 0.0.0.0 --api-key KEY``.
* One request is generated at a time; the others wait.
* A stream carries one chunk per token (several chunks arrive together, about 64 times a second on Windows).
* Text only.  Not supported (answered with 400): images, ``tools`` / ``functions``, ``n`` > 1, ``logprobs``.
"""
from __future__ import annotations

import hmac
import json
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ._native import YasakaniError
from .chat import (REASONING_EFFORT_ALIASES, ChatParams, ChatRunner, RequestError, Usage, clean_messages,
                   DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P)

MAX_BODY = 32 * 1024 * 1024
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


class ServerConfig:
    def __init__(self, host="127.0.0.1", port=8765, api_key=None, model_id="qwen3.8-27b", default_max_tokens=32768,
                 max_tokens_cap=None):
        self.host, self.port, self.api_key, self.model_id = host, int(port), api_key, model_id
        self.default_max_tokens, self.max_tokens_cap = int(default_max_tokens), max_tokens_cap


class ApiState:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.runner: Optional[ChatRunner] = None          # None while the model loads
        self.gen_lock = threading.Lock()
        self.version = ""


class SingleServer(ThreadingHTTPServer):
    allow_reuse_address = False                              # Windows: SO_REUSEADDR would let two servers share one port
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return                                           # the client went away: nothing to report
        super().handle_error(request, client_address)


def _num(body, key, default, lo=None, hi=None, kind=float):
    v = body.get(key)
    if v is None:
        return default
    try:
        v = kind(v)
    except (TypeError, ValueError):
        raise RequestError("%s must be a number" % key)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise RequestError("%s must be between %s and %s" % (key, lo, hi))
    return v


def parse_chat_request(body: dict, cfg: ServerConfig) -> ChatParams:
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    for key in ("tools", "functions", "tool_choice", "function_call"):
        if body.get(key):
            raise RequestError("%s is not supported yet" % key)
    if body.get("logprobs"):
        raise RequestError("logprobs is not supported")
    if body.get("n") not in (None, 1):
        raise RequestError("n must be 1 (one answer per request)")
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") not in (None, "text"):
        raise RequestError("response_format %r is not supported" % rf.get("type"))
    messages = clean_messages(body.get("messages"))
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    max_tokens = cfg.default_max_tokens if mt is None else _num({"v": mt}, "v", None, 1, None, int)
    if cfg.max_tokens_cap:
        max_tokens = min(max_tokens, int(cfg.max_tokens_cap))
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(s, str) for s in stop):
        raise RequestError("stop must be a string or a list of at most 4 strings")
    kw = dict(body.get("chat_template_kwargs") or {})
    if not isinstance(kw, dict):
        raise RequestError("chat_template_kwargs must be an object")
    eff = body.get("reasoning_effort")
    if eff is not None:
        if eff not in REASONING_EFFORT_ALIASES:
            raise RequestError("reasoning_effort must be one of low, medium, high, xhigh")
        kw.setdefault("reasoning_effort", REASONING_EFFORT_ALIASES[eff])
    seed = body.get("seed")
    return ChatParams(
        messages=messages, max_tokens=max_tokens,
        temperature=_num(body, "temperature", DEFAULT_TEMPERATURE, 0.0, 2.0),
        top_p=_num(body, "top_p", DEFAULT_TOP_P, 1e-6, 1.0),
        top_k=_num(body, "top_k", DEFAULT_TOP_K, None, None, int),
        min_p=_num(body, "min_p", 0.0, 0.0, 1.0),
        seed=None if seed is None else _num(body, "seed", 0, None, None, int),
        stop=list(stop), ignore_eos=bool(body.get("ignore_eos", False)), loop_guard=body.get("loop_guard", True) is not False,
        template_kwargs=kw)


def usage_json(u: Usage) -> dict:
    d = dict(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, total_tokens=u.prompt_tokens + u.completion_tokens,
             prompt_tokens_details=dict(cached_tokens=u.cached_tokens),
             completion_tokens_details=dict(reasoning_tokens=u.reasoning_tokens))
    if u.loop_guard:
        d["loop_guard"] = u.loop_guard
    return d


def make_handler(state: ApiState):
    cfg = state.cfg

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 300                                        # seconds a connection may sit idle between requests

        def log_message(self, *a):                           # no access log (the server prints its own status lines)
            pass

        def setup(self):
            super().setup()
            try:                                             # a stream is many small pieces (one per token): send each at once
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        # ---- plumbing ---------------------------------------------------------------------------------------------------------
        def end_headers(self):
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def _json(self, code, obj, close=False):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(b)

        def _error(self, code, message, kind="invalid_request_error", close=False):
            self._json(code, dict(error=dict(message=message, type=kind, code=code)), close)

        def _drain(self):
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                n = 0
            if 0 < n <= MAX_BODY:
                self.rfile.read(n)

        def _own_hosts(self):
            port = self.server.server_address[1]
            names = list(LOOPBACK) if cfg.host in ("127.0.0.1", "localhost", "::1") else [cfg.host]
            return tuple("%s:%d" % (h, port) for h in names)

        def _refused(self, post) -> Optional[str]:
            if cfg.host not in ("127.0.0.1", "localhost", "::1") and cfg.api_key:
                return None                                    # remote use is protected by the key
            own = self._own_hosts()
            if (self.headers.get("Host") or "").lower() not in own:
                return "host"
            if post:
                origin = self.headers.get("Origin")
                if origin is not None and origin.lower() not in tuple("http://" + h for h in own):
                    return "origin"
                if (self.headers.get("Sec-Fetch-Site") or "none").lower() not in ("same-origin", "none"):
                    return "cross-site"
            return None

        def _unauthorized(self) -> bool:
            if not cfg.api_key:
                return False
            auth = self.headers.get("Authorization") or ""
            ok = auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].encode(), str(cfg.api_key).encode())
            if not ok:
                self._error(401, "invalid or missing API key", "authentication_error")
            return not ok

        # ---- routes -----------------------------------------------------------------------------------------------------------
        def do_GET(self):
            why = self._refused(False)
            if why:
                return self._error(403, "forbidden: " + why, "forbidden")
            path = self.path.split("?", 1)[0]
            if path == "/health":
                code = 200 if state.runner else 503
                body = dict(status="ok" if state.runner else "loading", version=state.version, model=cfg.model_id)
                if state.runner:
                    body["max_context_tokens"] = state.runner.engine.max_ctx
                return self._json(code, body)
            if self._unauthorized():
                return
            if path in ("/v1/models", "/v1/models/"):
                return self._json(200, dict(object="list", data=[dict(id=cfg.model_id, object="model", owned_by="local")]))
            if path == "/v1/models/" + cfg.model_id:
                return self._json(200, dict(id=cfg.model_id, object="model", owned_by="local"))
            return self._error(404, "not found", "not_found")

        def do_POST(self):
            why = self._refused(True)
            if why:
                self._drain()
                return self._error(403, "forbidden: " + why, "forbidden")
            if self._unauthorized():
                self._drain()
                return
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._drain()
                return self._error(404, "not found", "not_found")
            runner = state.runner
            if runner is None:
                self._drain()
                return self._error(503, "the model is still loading", "unavailable")
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._error(413, "request body too large or length missing", "invalid_request_error", close=True)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                params = parse_chat_request(body, cfg)
            except json.JSONDecodeError:
                return self._error(400, "the request body is not valid JSON")
            except RequestError as ex:
                return self._error(400, str(ex))
            self._chat(runner, body, params)

        def _chat(self, runner: ChatRunner, body: dict, params: ChatParams):
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(body.get("stream"))

            def chunk(delta=None, finish=None, usage=None):
                o = dict(id=cid, object="chat.completion.chunk", created=created, model=cfg.model_id,
                         choices=[] if usage else [dict(index=0, delta=delta or {}, finish_reason=finish)])
                if usage:
                    o["usage"] = usage
                self.wfile.write(b"data: " + json.dumps(o, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.wfile.flush()

            with state.gen_lock:
                try:
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        chunk(delta=dict(role="assistant"))
                        u = runner.run(params, lambda kind, text: chunk(delta={"reasoning_content" if kind == "reasoning" else "content": text}))
                        chunk(delta={}, finish=u.finish)
                        if (body.get("stream_options") or {}).get("include_usage"):
                            chunk(usage=usage_json(u))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        self.close_connection = True
                    else:
                        parts = {"reasoning": [], "content": []}
                        u = runner.run(params, lambda kind, text: parts[kind].append(text))
                        msg = dict(role="assistant", content="".join(parts["content"]))
                        if parts["reasoning"]:
                            msg["reasoning_content"] = "".join(parts["reasoning"])
                        self._json(200, dict(id=cid, object="chat.completion", created=created, model=cfg.model_id,
                                             choices=[dict(index=0, message=msg, finish_reason=u.finish)], usage=usage_json(u)))
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    self.close_connection = True               # the client went away: the engine was already stopped
                except Exception as ex:                          # noqa: BLE001
                    code = 400 if isinstance(ex, RequestError) else 500
                    msg = str(ex)[:500] or type(ex).__name__
                    try:
                        if stream:
                            self.wfile.write(b"data: " + json.dumps(dict(error=dict(message=msg, type="server_error", code=code))).encode() + b"\n\n")
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                            self.close_connection = True
                        else:
                            self._error(code, msg, "server_error" if code == 500 else "invalid_request_error")
                    except Exception:                            # noqa: BLE001
                        self.close_connection = True

    return Handler


def bind(state: ApiState) -> SingleServer:
    """Bind the port first (before loading the model): a second start finds the port taken and stops here."""
    cls = SingleServer
    if ":" in state.cfg.host:                                # an IPv6 address such as ::1
        class cls(SingleServer):                             # noqa: N801
            address_family = socket.AF_INET6
    return cls((state.cfg.host, state.cfg.port), make_handler(state))
