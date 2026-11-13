# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Text <-> ids: the chat template, the tokenizer, and incremental decoding of the output stream.

Only ``tokenizers`` and ``jinja2`` are needed (no torch, no transformers).  The chat template is the model's own
(``chat_template.jinja`` or the ``chat_template`` entry of ``tokenizer_config.json``) and is rendered the way Hugging Face does.
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._native import YasakaniError
from .runtime import EOS_IDS, IM_START


def _raise_exception(message):
    from jinja2.exceptions import TemplateError
    raise TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class ChatFormatter:
    """Chat template + tokenizer of one model folder."""

    def __init__(self, folder):
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        d = Path(folder)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        tpl_file = d / "chat_template.jinja"
        template = tpl_file.read_text(encoding="utf-8") if tpl_file.is_file() else cfg.get("chat_template")
        if isinstance(template, list):                               # named templates: use "default"
            template = {t.get("name"): t.get("template") for t in template}.get("default")
        if not isinstance(template, str) or not template.strip():
            raise YasakaniError("no chat template found in " + str(d))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        self.template = env.from_string(template)
        self.bos_token = cfg.get("bos_token")
        self.eos_token = cfg.get("eos_token")
        self.im_start = self.token_id("<|im_start|>")
        self.im_end = self.token_id("<|im_end|>")
        self.think_start = self.token_id("<think>")
        self.think_end = self.token_id("</think>")
        if self.im_start != IM_START or self.im_end != EOS_IDS[0]:
            raise YasakaniError("this tokenizer does not match the engine (<|im_start|>=%s, <|im_end|>=%s)" % (self.im_start, self.im_end))
        self.digit_ids = np.array([self.token_id(str(i)) for i in range(10)], np.int32)

    def token_id(self, token: str) -> int:
        i = self.tok.token_to_id(token)
        if i is None:
            raise YasakaniError("the tokenizer has no token " + token)
        return int(i)

    def render(self, messages: Sequence[dict], *, add_generation_prompt: bool = True, **template_kwargs) -> str:
        return self.template.render(messages=list(messages), add_generation_prompt=add_generation_prompt, bos_token=self.bos_token,
                                    eos_token=self.eos_token, **template_kwargs)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tok.encode(text, add_special_tokens=False).ids, np.int32)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def prompt(self, messages: Sequence[dict], **template_kwargs) -> Tuple[np.ndarray, bool]:
        """(prompt ids, starts inside the thinking block).  The model's template ends an open thinking block with ``<think>``."""
        text = self.render(messages, add_generation_prompt=True, **template_kwargs)
        return self.encode(text), text.rstrip().endswith("<think>")


class StreamDecoder:
    """Incremental detokenizer: ``push(new_ids)`` returns the text that became complete (a character split over several ids is held
    back until it is whole).  ``flush()`` returns whatever is left.  Work per call is bounded by the held-back window."""

    def __init__(self, tok):
        self.tok = tok
        self.ids: List[int] = []
        self.prefix = 0                  # start of the window that is decoded again
        self.read = 0                    # ids before this are already emitted

    def _dec(self, a: int, b: Optional[int] = None) -> str:
        return self.tok.decode(self.ids[a:b], skip_special_tokens=False)

    def push(self, new_ids: Iterable[int]) -> str:
        self.ids.extend(int(i) for i in new_ids)
        if len(self.ids) == self.read:
            return ""
        old = self._dec(self.prefix, self.read)
        new = self._dec(self.prefix)
        if len(new) > len(old) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(old):]
        return ""

    def flush(self) -> str:
        if len(self.ids) == self.read:
            return ""
        old = self._dec(self.prefix, self.read)
        new = self._dec(self.prefix)
        self.prefix = self.read = len(self.ids)
        return new[len(old):]


class OutputStream:
    """Turns the growing id array of one generation into ``("reasoning", text)`` / ``("content", text)`` pieces, one piece per id
    that completes some text.  Ids before the ``</think>`` id are the reasoning, ids after it the answer; stop ids are never shown."""

    def __init__(self, fmt: ChatFormatter, in_reasoning: bool, hidden_ids: Sequence[int] = EOS_IDS):
        self.fmt = fmt
        self.in_reasoning = in_reasoning
        self.hidden = frozenset(int(i) for i in hidden_ids)
        self.reason = StreamDecoder(fmt.tok)
        self.answer = StreamDecoder(fmt.tok)
        self.seen = 0
        self._lead = True                # drop the newlines the model writes after </think>
        self.think_end_at: Optional[int] = None

    def _answer_text(self, text: str) -> str:
        if self._lead:
            text = text.lstrip("\n")
            if text:
                self._lead = False
        return text

    def feed(self, ids: np.ndarray, final: bool = False) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        new = [int(x) for x in ids[self.seen:]]
        base = self.seen
        self.seen = len(ids)
        run: List[int] = []

        def emit_run():
            # one piece per id, so that a client that counts the pieces of a stream counts tokens, as with other servers
            # (an id that does not complete a character gives no piece: its text comes with the id that completes it)
            nonlocal run
            for t in run:
                if self.in_reasoning:
                    piece = self.reason.push((t,))
                    if piece:
                        events.append(("reasoning", piece))
                else:
                    piece = self._answer_text(self.answer.push((t,)))
                    if piece:
                        events.append(("content", piece))
            run = []

        for k, t in enumerate(new):
            if t in self.hidden:
                continue
            if self.in_reasoning and t == self.fmt.think_end:
                emit_run()
                tail = self.reason.flush()
                if tail:
                    events.append(("reasoning", tail))
                self.in_reasoning = False
                self.think_end_at = base + k
                continue
            run.append(t)
        emit_run()
        if final:
            tail = self.reason.flush() if self.in_reasoning else ""
            if tail:
                events.append(("reasoning", tail))
            tail = self._answer_text(self.answer.flush())
            if tail:
                events.append(("content", tail))
        return events


# ---- tool calls -------------------------------------------------------------------------------------------------------------------
# The model's chat template asks for tool calls in this form (text inside the answer):
#
#     <tool_call>
#     <function=NAME>
#     <parameter=KEY>
#     value (may span several lines)
#     </parameter>
#     </function>
#     </tool_call>
#
# ToolCallFilter takes the answer text as it is streamed and splits it into plain text and complete tool calls.
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
_FUNCTION = re.compile(r"<function=([^>\n]+)>")
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def tool_schemas(tools: Optional[Sequence[dict]]) -> Dict[str, dict]:
    """name -> {parameter name: JSON schema} of OpenAI-style tool definitions."""
    out: Dict[str, dict] = {}
    for t in tools or ():
        f = t.get("function") if isinstance(t, dict) and isinstance(t.get("function"), dict) else t
        if isinstance(f, dict) and isinstance(f.get("name"), str):
            params = f.get("parameters") if isinstance(f.get("parameters"), dict) else {}
            props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
            out[f["name"]] = props
    return out


def _is_string_schema(schema) -> bool:
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if isinstance(t, str):
        return t == "string"
    if isinstance(t, list):
        return bool(t) and all(x in ("string", "null") for x in t) and "string" in t
    return False


def convert_parameter(value: str, schema) -> object:
    """The text of one <parameter> block as a JSON value.  A parameter declared as a string stays text; anything else is read as
    JSON when it is JSON (numbers, booleans, lists, objects) and stays text when it is not."""
    if _is_string_schema(schema):
        return value
    try:
        v = json.loads(value)
    except (ValueError, TypeError):
        return value
    if isinstance(v, str) and not isinstance(schema, dict):
        return value                                   # no declaration: keep what the model wrote, quotes included
    return v


def parse_tool_call(block: str, schemas: Dict[str, dict]) -> Optional[dict]:
    """The inside of one <tool_call> block -> {"name": ..., "arguments": {...}}, or None when it is not a call."""
    m = _FUNCTION.search(block)
    if m:
        name = m.group(1).strip()
        props = schemas.get(name) or {}
        args = {}
        for key, value in _PARAMETER.findall(block[m.end():]):
            key = key.strip()
            args[key] = convert_parameter(value, props.get(key))
        return {"name": name, "arguments": args} if name else None
    text = block.strip()
    if text.startswith("{"):                              # the JSON form some models fall back to
        try:
            d = json.loads(text)
        except ValueError:
            return None
        if isinstance(d, dict) and isinstance(d.get("name"), str) and d["name"]:
            args = d.get("arguments", d.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    return None
            if args is None:
                args = {}
            if isinstance(args, dict):
                return {"name": d["name"], "arguments": args}
    return None


class ToolCallFilter:
    """``push(text)`` returns a list of ``("content", text)`` and ``("tool_call", {"name", "arguments"})`` items.  Text that might be
    the start of a tool call (and white space that might only separate the text from a call) is held back until it is decided.
    ``flush()`` returns what is left: an unfinished or unreadable call comes back as text, so nothing the model wrote is lost."""

    def __init__(self, tools: Optional[Sequence[dict]]):
        self.schemas = tool_schemas(tools)
        self.buf = ""
        self.inside = False
        self.gap = ""                    # white space that stood between the text and the <tool_call> being read
        self.calls = 0

    def _text(self, text: str, out: list) -> None:
        if text and (text.strip() or not self.calls):
            out.append(("content", text))

    def push(self, text: str) -> list:
        out: list = []
        self.buf += text
        while True:
            if self.inside:
                j = self.buf.find(TOOL_CLOSE)
                if j < 0:
                    return out
                block, self.buf, self.inside = self.buf[:j], self.buf[j + len(TOOL_CLOSE):], False
                call = parse_tool_call(block, self.schemas)
                if call is None:
                    out.append(("content", self.gap + TOOL_OPEN + block + TOOL_CLOSE))
                else:
                    self.calls += 1
                    out.append(("tool_call", call))
                self.gap = ""
                continue
            i = self.buf.find(TOOL_OPEN)
            if i >= 0:
                before = self.buf[:i]
                self._text(before.rstrip(), out)
                self.gap = before[len(before.rstrip()):]
                self.buf, self.inside = self.buf[i + len(TOOL_OPEN):], True
                continue
            keep = 0                                       # a tail that may be the start of <tool_call> ...
            for k in range(min(len(TOOL_OPEN) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(TOOL_OPEN[:k]):
                    keep = k
                    break
            head = self.buf[:len(self.buf) - keep]
            cut = len(head.rstrip())                        # ... and the white space before it
            if cut:
                self._text(self.buf[:cut], out)
                self.buf = self.buf[cut:]
            return out

    def flush(self) -> list:
        out: list = []
        rest, self.buf = self.buf, ""
        if self.inside:
            self.inside = False
            out.append(("content", self.gap + TOOL_OPEN + rest))
            self.gap = ""
        elif rest.strip() or (rest and not self.calls):
            out.append(("content", rest))
        return out

