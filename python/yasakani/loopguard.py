# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Loop guard: stop a generation that has fallen into an endless repetition.

Models sometimes get stuck writing the same few ids forever (or thousands of digits in a row) while "thinking".  The guard watches the
ids as they are committed and reports a loop; the caller then cancels the generation and, inside the thinking, asks the model to answer
from what it has.  Requests without a loop are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

PERIOD_MAX = 16            # longest repeating unit that is recognised (ids)
RUN_REASONING = 256        # an exact repetition this long inside the thinking is a loop
RUN_ANSWER = 1024          # ... inside the answer
DIGITS = 1000              # this many digit ids in a row inside the thinking is a loop

BRIDGE = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n")
NOTICE = "\n\n[loop guard: the same sequence kept repeating, so the thinking was cut here and the answer follows]\n"


def loop_period(ids: np.ndarray, start: int, n: int, run: int) -> int:
    """If the last ``run + p`` ids of ``ids[start:n]`` repeat with a period ``p <= PERIOD_MAX``, the smallest such p, else 0."""
    if n - start < run + PERIOD_MAX:
        return 0
    t = ids[n - run - PERIOD_MAX:n]
    tail = t[PERIOD_MAX:]
    for p in range(1, PERIOD_MAX + 1):
        if np.array_equal(tail, t[PERIOD_MAX - p:len(t) - p]):
            return p
    return 0


@dataclass
class LoopHit:
    where: str            # "reasoning" | "answer"
    kind: str             # "repeat" | "digits"
    at: int               # ids produced when it was noticed
    period: int = 0
    run: int = 0


class LoopWatch:
    """Feed it the growing id array (``check(ids)``); it returns a ``LoopHit`` once, then stays quiet."""

    def __init__(self, think_end_id: int, digit_ids: np.ndarray, in_reasoning: bool, base: int = 0):
        self.think_end = int(think_end_id)
        self.digits = np.asarray(digit_ids, np.int32)
        self.base = int(base)                           # ids before this index belong to an earlier stage and are not looked at
        self.think_at: Optional[int] = None if in_reasoning else -1
        self.checked = self.base
        self.dig_run = 0
        self.hit: Optional[LoopHit] = None

    def check(self, ids: np.ndarray) -> Optional[LoopHit]:
        n = len(ids)
        if self.hit is not None or n <= self.checked:
            return None
        lo = self.checked
        if self.think_at is None:
            found = np.flatnonzero(ids[lo:n] == self.think_end)
            if len(found):
                self.think_at = lo + int(found[0])
            else:
                non = np.flatnonzero(~np.isin(ids[lo:n], self.digits))
                self.dig_run = self.dig_run + (n - lo) if not len(non) else (n - lo) - int(non[-1]) - 1
        self.checked = n
        if self.think_at is None:
            p, where = loop_period(ids, self.base, n, RUN_REASONING), "reasoning"
        else:
            p, where = loop_period(ids, max(self.think_at + 1, self.base), n, RUN_ANSWER), "answer"
        if p:
            self.hit = LoopHit(where=where, kind="repeat", at=n, period=p)
        elif self.think_at is None and self.dig_run >= DIGITS:
            self.hit = LoopHit(where="reasoning", kind="digits", at=n, run=self.dig_run)
        return self.hit
