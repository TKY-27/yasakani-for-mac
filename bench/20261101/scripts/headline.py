#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""headline.py - re-derive the November 1, 2026 comparison table from the records in this folder (standard library only).

  python headline.py            (from any folder; it reads ../results relative to this file)

YASAKANI: results/yasakani/A_pub.json (public build, two back-to-back rounds, loop guard off).  For each test and round, the median
of the runs marked "used"; the faster of the two round medians is the published number (both rounds are printed).
Other engines: results/others/runs_others.jsonl.  A run counts when it wrote at least 100 tokens.  For each engine and test, the median
of the counted runs of each setting; the setting with the highest median is shown (the rule of wf_to_summary.py).
Ratios are computed from the unrounded medians.
"""
import collections
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIX = ("chat", "explain", "code", "math", "json", "long")
ORDER = SIX + ("prose3", "japanese", "ninfer_prose", "sglang_random_8k")
ENGINE = {"llama": "llama.cpp", "sg": "SGLang", "vl": "vLLM"}
SET_TO_TEST = {"v8prose": "prose3", "long": "long", "jp": "japanese", "ninfer": "ninfer_prose", "rand8k": "sglang_random_8k"}


def test_of(r):
    return SET_TO_TEST.get(r["set"], r["prompt"] if r["set"] in ("h2h", "h2hx") else None)


def yasakani():
    d = json.load(open(os.path.join(ROOT, "results", "yasakani", "A_pub.json"), encoding="utf-8"))
    g = collections.defaultdict(list)
    for r in d["runs"]:
        if r.get("used"):
            g[(r["test"], r["round"])].append(r["decode_tok_s"])
    return {t: (statistics.median(g[(t, 1)]), statistics.median(g[(t, 2)])) for t in ORDER}


def others():
    g = collections.defaultdict(list)
    for line in open(os.path.join(ROOT, "results", "others", "runs_others.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("completion_tokens") or 0) >= 100 and r.get("decode_tok_s") is not None:
            g[(test_of(r), ENGINE[r["cfg"].split("+")[0].split("_")[0]], r["cfg"])].append(r["decode_tok_s"])
    best = collections.defaultdict(dict)
    for (t, e, cfg), v in g.items():
        m = statistics.median(v)
        if e not in best[t] or m > best[t][e][0]:
            best[t][e] = (m, cfg, len(v))
    return best


def main():
    y = yasakani()
    best = others()
    print("%-17s %8s %8s %8s | %-26s %-26s %-26s | %s" % ("test", "Y rnd 1", "Y rnd 2", "Y shown", "SGLang", "vLLM", "llama.cpp", "ratio"))
    ratio = {}
    for t in ORDER:
        shown = max(y[t])
        fastest = max(best[t].values(), key=lambda x: x[0])
        ratio[t] = shown / fastest[0]
        cells = ["%.1f %s" % (best[t][e][0], best[t][e][1]) for e in ("SGLang", "vLLM", "llama.cpp")]
        print("%-17s %8.1f %8.1f %8.1f | %-26s %-26s %-26s | %.4f" % ((t,) + y[t] + (shown,) + tuple(cells) + (ratio[t],)))
    med = statistics.median
    print()
    print("median of the six thinking tests (tokens/s): YASAKANI %.1f, SGLang %.1f, vLLM %.1f, llama.cpp %.1f"
          % (med([max(y[t]) for t in SIX]), med([best[t]["SGLang"][0] for t in SIX]), med([best[t]["vLLM"][0] for t in SIX]),
             med([best[t]["llama.cpp"][0] for t in SIX])))
    print("ratio to the fastest other engine: %.4f (min) to %.4f (max) over the 10 tests; median of the six thinking tests %.4f"
          % (min(ratio.values()), max(ratio.values()), med([ratio[t] for t in SIX])))
    print("tests where YASAKANI was fastest: %d of %d" % (sum(1 for t in ORDER if ratio[t] > 1), len(ORDER)))


if __name__ == "__main__":
    main()
