# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""One chat request, end to end: messages -> prompt ids -> streamed generation -> reasoning / answer text."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import loopguard
from ._native import YasakaniError
from .engine import CONTEXT_MARGIN, Engine, GenResult, Sampling
from .runtime import EOS_IDS
from .textio import ChatFormatter, OutputStream

# defaults of the model (generation_config.json of Qwen3.8)
DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K = 1.0, 0.95, 20
REASONING_EFFORT_ALIASES = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}


class RequestError(ValueError):
    """The request cannot be served as asked (maps to HTTP 400)."""


@dataclass
class ChatParams:
    messages: List[dict]
    max_tokens: int = 32768
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    ignore_eos: bool = False
    loop_guard: bool = True
    template_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    finish: str = "stop"
    passes: int = 0
    prefill_ms: float = 0.0
    gen_ms: float = 0.0
    snap_mode: int = 0
    loop_guard: Optional[dict] = None


class StopFilter:
    """Holds back text that might be the start of a stop string; cuts at the first complete one."""

    def __init__(self, stops: Sequence[str]):
        self.stops = [s for s in stops if s]
        self.buf = ""
        self.hit = False

    def push(self, text: str) -> str:
        if not self.stops or self.hit:
            return "" if self.hit else text
        self.buf += text
        cuts = [self.buf.find(s) for s in self.stops if s in self.buf]
        if cuts:
            out = self.buf[:min(cuts)]
            self.buf, self.hit = "", True
            return out
        keep = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    keep = max(keep, k)
                    break
        cut = len(self.buf) - keep
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out

    def flush(self) -> str:
        out, self.buf = ("" if self.hit else self.buf), ""
        return out


def clean_messages(messages) -> List[dict]:
    """Validate OpenAI-style messages (text only) and return plain dicts the template understands."""
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise RequestError("messages[%d]: role must be system, user or assistant" % i)
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if not isinstance(p, dict) or p.get("type") not in ("text", None) or "text" not in p:
                    raise RequestError("messages[%d]: only text content is supported" % i)
                parts.append({"type": "text", "text": str(p["text"])})
            c = parts
        elif c is None:
            c = ""
        elif not isinstance(c, str):
            raise RequestError("messages[%d]: content must be a string or a list of text parts" % i)
        msg = {"role": m["role"], "content": c}
        if m["role"] == "assistant" and isinstance(m.get("reasoning_content"), str):
            msg["reasoning_content"] = m["reasoning_content"]
        out.append(msg)
    return out


class ChatRunner:
    """Runs chat requests on one engine, one at a time."""

    def __init__(self, engine: Engine, fmt: ChatFormatter, seed0: int = 1):
        self.engine = engine
        self.fmt = fmt
        self._seed = int(seed0)
        self._seed_lock = threading.Lock()
        self.bridge = fmt.encode(loopguard.BRIDGE)
        if fmt.think_end not in self.bridge.tolist():
            raise YasakaniError("tokenizer: </think> is not a single id in the loop-guard bridge")

    def next_seed(self) -> int:
        with self._seed_lock:
            s = self._seed
            self._seed += 1
            return s

    def warmup(self) -> None:
        plain = self.fmt.encode("warm-up")
        chat, _ = self.fmt.prompt([{"role": "user", "content": "warm-up"}])
        self.engine.warmup([plain, chat], n_gen=32)

    def prepare(self, p: ChatParams):
        kw = dict(p.template_kwargs)
        prompt, in_reasoning = self.fmt.prompt(p.messages, **kw)
        return prompt, in_reasoning

    def run(self, p: ChatParams, emit: Callable[[str, str], None]) -> Usage:
        """Generate the reply, calling ``emit("reasoning" | "content", text)`` as text becomes available.  Raises ``RequestError`` for
        a request that cannot be served, ``YasakaniError`` for an engine failure."""
        eng = self.engine
        prompt, in_reasoning = self.prepare(p)
        if len(prompt) >= eng.max_ctx - CONTEXT_MARGIN:
            raise RequestError("the input is too long (%d tokens); input, thinking and answer together are limited to %d tokens"
                               % (len(prompt), eng.max_ctx))
        max_new = max(1, int(p.max_tokens))
        seed = p.seed if p.seed is not None else self.next_seed()
        eng.set_sampling(Sampling(p.temperature, p.top_p, p.top_k, p.min_p, seed))
        stops = None if p.ignore_eos else EOS_IDS
        usage = Usage(prompt_tokens=len(prompt))
        stopf = StopFilter(p.stop)
        guard_on = p.loop_guard and not p.ignore_eos

        def deliver(events):
            for kind, text in events:
                if kind == "content":
                    text = stopf.push(text)
                if text:
                    emit(kind, text)
            if stopf.hit:
                eng.cancel()

        # ---- stage 1: the request as asked ----------------------------------------------------------------------------
        stream = OutputStream(self.fmt, in_reasoning)
        watch = loopguard.LoopWatch(self.fmt.think_end, self.fmt.digit_ids, in_reasoning) if guard_on else None
        state: Dict[str, object] = {"hit": None}

        def on_tokens(ids):
            deliver(stream.feed(ids))
            if watch is not None and state["hit"] is None:
                hit = watch.check(ids)
                if hit is not None:
                    state["hit"] = hit
                    eng.cancel()

        r1 = eng.generate(prompt, max_new, stop_ids=stops, on_tokens=on_tokens)
        # the engine's final ids may include ones committed after the last poll: feed the complete array once more, then flush
        out_ids = np.asarray(r1.ids, np.int32)
        deliver(stream.feed(out_ids, final=True))
        usage.cached_tokens, usage.snap_mode = r1.cached_tokens, r1.snap_mode
        usage.passes, usage.prefill_ms, usage.gen_ms = r1.passes, r1.prefill_ms, r1.gen_ms
        n_out = len(out_ids)
        reasoning_end = stream.think_end_at
        usage.reasoning_tokens = (reasoning_end if reasoning_end is not None else (n_out if stream.in_reasoning else 0))
        finish = r1.finish
        hit = state["hit"]
        if hit is not None:
            info = dict(where=hit.where, kind=hit.kind, period=hit.period, run=hit.run, at=hit.at, tokens_before=n_out)
            usage.loop_guard = info
            if hit.where == "reasoning" and r1.finish != "stop" and not stopf.hit:
                finish = self._finish_after_loop(p, prompt, out_ids, usage, stops, stopf, emit, deliver, info, max_new)
            else:
                finish = "length" if not stopf.hit else "stop"
        if stopf.hit:
            finish = "stop"
        tail = stopf.flush()
        if tail:
            emit("content", tail)
        usage.completion_tokens = n_out + (usage.loop_guard or {}).get("extra_tokens", 0)
        usage.finish = "stop" if finish == "stop" else "length"
        return usage

    def _finish_after_loop(self, p, prompt, out_ids, usage, stops, stopf, emit, deliver, info, max_new) -> str:
        """The thinking looped: close it with the bridge sentence and let the model write the answer on the same sequence."""
        eng = self.engine
        gen = len(out_ids)
        seq = np.concatenate([prompt, out_ids, self.bridge]).astype(np.int32)
        n2 = min(max_new - 1 - gen - len(self.bridge), eng.max_ctx - len(seq) - CONTEXT_MARGIN)
        if n2 < 16:
            info["answer_tokens"] = 0                      # no budget left for an answer
            return "length"
        emit("reasoning", loopguard.NOTICE)
        stream = OutputStream(self.fmt, False)
        watch = loopguard.LoopWatch(self.fmt.think_end, self.fmt.digit_ids, False)
        hit2: Dict[str, object] = {"hit": None}

        def on_tokens(ids):
            deliver(stream.feed(ids))
            if hit2["hit"] is None:
                h = watch.check(ids)
                if h is not None:
                    hit2["hit"] = h
                    eng.cancel()

        r2 = eng.generate(seq, n2 + 1, stop_ids=stops, on_tokens=on_tokens)
        ids2 = np.asarray(r2.ids, np.int32)
        deliver(stream.feed(ids2, final=True))
        info.update(answer_tokens=len(ids2), answer_cached=r2.cached_tokens, extra_tokens=len(self.bridge) + len(ids2))
        usage.passes += r2.passes
        usage.prefill_ms += r2.prefill_ms
        usage.gen_ms += r2.gen_ms
        if hit2["hit"] is not None:
            info["answer_loop"] = dict(kind=hit2["hit"].kind, period=hit2["hit"].period, at=hit2["hit"].at)
        return r2.finish
