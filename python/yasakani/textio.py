# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Text <-> ids: the chat template, the tokenizer, and incremental decoding of the output stream.

Only ``tokenizers`` and ``jinja2`` are needed (no torch, no transformers).  The chat template is the model's own
(``chat_template.jinja`` or the ``chat_template`` entry of ``tokenizer_config.json``) and is rendered the way Hugging Face does.
"""
from __future__ import annotations

import datetime
import json
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
