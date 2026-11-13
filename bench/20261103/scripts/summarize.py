#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""summarize.py - print the November 3, 2026 numbers from the records in this folder (standard library only).

  python summarize.py            (from any folder; it reads ../results relative to this file)

1. NInfer's 12 scenario prompts: for every results/<condition>/runs_*.jsonl, the mean speed per prompt and the mean of each category
   (code, story, translation, structured = the mean of its three prompt means).  Runs that ended with finish_reason "tool_calls"
   (MegaCapybara stopping early on the TypeScript prompt) are left out and listed.
2. The single recorded runs used for the video (results/video_runs): the speed stored in each record and the same value recomputed from
   the chunk arrival times, (completion_tokens - 1) / (last chunk - first chunk).  For story_en_mystery also the speed over the first
   2,000 tokens: exact for MegaCapybara (one token per chunk); estimated for YASAKANI (several tokens per chunk; the chunk where the
   character count reaches 2000/completion_tokens of the text).
3. The own tests (results/own_tests): every run.
"""
import collections
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CATS = {"code": ["scenario_code_cuda", "scenario_code_python", "scenario_code_typescript"],
        "story": ["scenario_story_en_mystery", "scenario_story_zh_scifi", "scenario_story_zh_dialogue"],
        "translation": ["scenario_translation_zh_en", "scenario_translation_en_zh", "scenario_translation_markdown"],
        "structured": ["scenario_structured_jsonl", "scenario_structured_sql", "scenario_structured_csv"]}


def twelve():
    print("1. NInfer's 12 scenario prompts (mean tokens/s per prompt; n = runs)")
    for d in sorted(glob.glob(os.path.join(RES, "*", ""))):
        name = os.path.basename(os.path.dirname(d))
        if name in ("video_runs", "own_tests"):
            continue
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "runs_*.jsonl"))) for l in open(f, encoding="utf-8") if l.strip()]
        g = collections.defaultdict(list)
        left_out = []
        for r in rows:
            if r.get("finish") == "tool_calls":
                left_out.append("%s (%s tokens)" % (r["fixture"], r.get("completion_tokens")))
                continue
            g[r["fixture"]].append(r["decode_tok_s"])
        means = {k: statistics.mean(v) for k, v in g.items()}
        print("\n  %s: %d runs%s" % (name, len(rows), ("; left out: " + ", ".join(left_out)) if left_out else ""))
        for cat, fx in CATS.items():
            have = [x for x in fx if x in means]
            if not have:
                continue
            cells = ", ".join("%s %.1f (n=%d)" % (x.replace("scenario_", ""), means[x], len(g[x])) for x in have)
            print("    %-11s mean %7.1f | %s" % (cat, statistics.mean([means[x] for x in have]), cells))


def first_tokens_speed(rec, n_tok):
    ch = rec["chunks"]
    t0 = ch[0][0]
    if rec["n_chunks"] >= rec["completion_tokens"] - 1:          # one token per chunk
        t = ch[n_tok - 1][0]
    else:
        total = sum(len(c[1]) for c in ch)
        target = total * n_tok / rec["completion_tokens"]
        acc = 0
        for c in ch:
            acc += len(c[1])
            if acc >= target:
                t = c[0]
                break
    return (n_tok - 1) / ((t - t0) / 1000.0)


def video():
    print("\n2. Recorded single runs (video_runs): stored speed / recomputed from the chunks")
    for f in sorted(glob.glob(os.path.join(RES, "video_runs", "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        ch = r["chunks"]
        again = (r["completion_tokens"] - 1) / ((ch[-1][0] - ch[0][0]) / 1000.0)
        extra = ""
        if r["fixture"] == "scenario_story_en_mystery":
            extra = ", first 2,000 tokens %.1f" % first_tokens_speed(r, 2000)
        print("  %-18s %-27s %7.1f / %8.2f  (%d tokens, %s%s)" % (r["engine"], r["fixture"], r["decode_tok_s"], again,
                                                                 r["completion_tokens"], r["finish"], extra))


def own():
    print("\n3. Own tests (each run)")
    for f in sorted(glob.glob(os.path.join(RES, "own_tests", "runs_*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                print("  %-18s %-16s run %d: %7.1f tokens/s, %5d tokens, %s" % (r["label"], r["case"], r["k"], r["decode_tok_s"],
                                                                                r["completion_tokens"], r["finish"]))


if __name__ == "__main__":
    twelve()
    video()
    own()
