#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tables.py -- the speed tables of docs/MEASUREMENTS.md, made again from the records in this folder.

usage:  python tables.py [--out DIR]          (Python 3.8+, standard library only; run it from anywhere)

Reads   ../results/<setting>/n12/runs_*.jsonl      one line per run of the 12 published prompts (answers in n12/texts/)
        ../results/<setting>/own/runs_*.jsonl      one line per run of the 3 additional tasks
        ../results_seeds_2_4/<setting>/...         YASAKANI, thinking on, runs 2 to 4 (runs 0 and 1 are in ../results)
        ../../20261103/results/megacapybara_kv64   MegaCapybara, thinking off, measured on November 3
        ../../20261103/results/own_tests           (its 3 additional tasks)
Prints the four tables and writes speed_tables.json and SPEED_TABLES.md to --out (default: ../tables).

How one number is made
  speed of a run   decode_tok_s in the record = (output tokens - 1) / (time from the first to the last streamed chunk)
  runs left out    always: runs that ended with finish "tool_calls" or have no speed
  "clean" tables   also leave out, for every engine alike, the runs of the 12 prompts whose saved answer is
                     T  a tool call written although the request has no tools ("<tool_call>" or "<function=" in the text)
                     L  a repetition loop: the last 4,000 characters compress below 12 % with zlib (code, story, translation);
                        for the structured prompts, which ask for regular records, fewer than half of the lines are distinct
  mean             a category = the mean over all its runs; an additional task = the mean over its runs
  fastest run      a category = the mean over its prompts of the fastest run of each prompt; an additional task = its fastest run
  engine row       for an engine measured with several settings, the largest value of each column
"""
import argparse
import glob
import json
import statistics
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
MORE = HERE.parent / "results_seeds_2_4"
NOV3 = HERE.parent.parent / "20261103" / "results"
CATS = ("code", "story", "translation", "structured")
OWN = ("edit_code", "edit_jsonl", "jsonl_orders_8k")
COLS = ["Code", "Story", "Translation", "Structured", "Code edit", "JSONL edit", "Long JSONL"]
# engine -> (settings measured with thinking off, settings measured with thinking on); folders under ../results,
# "nov3:" = under ../../20261103/results
ENGINES = [
    ("MegaCapybara 1.39 Tiny", ["nov3:megacapybara_kv64"], ["megacapybara_tiny_think"]),
    ("mkyt v0.3.2", ["mkyt_v032"], ["mkyt_v032_think"]),
    ("ignis v0.9.6", ["ignis_v096"], ["ignis_v096_think"]),
    ("NInfer (November 2 build)", ["n12_ninfer_master"], ["n12t_ninfer_master"]),
    ("vLLM 0.31.0", ["n12_vl_dflash_lm4", "n12_vl_dspark_lm4", "n12_vl_mtp_lm4"], ["n12t_vl_dflash_lm4", "n12t_vl_mtp_lm4"]),
    ("SGLang 0.5.22", ["n12_sg_dflash_lm4_norc", "n12_sg_dflash_nv", "n12_sg_dspark_lm4", "n12_sg_mtp_nv"],
     ["n12t_sg_dflash_nv", "n12t_sg_mtp_nv"]),
    ("llama.cpp b11700", ["n12_llama_dflq4", "n12_llama_mtp2"], ["n12t_llama_dflq4", "n12t_llama_mtp2"]),
    ("TensorFold 0.6.6", ["n12_tensorfold"], ["n12t_tensorfold"]),
]
OURS = [("YASAKANI fastest model", "speed_v8"), ("YASAKANI quality model", "quality_v8")]
# measured the same day, not used in the tables (see the README of bench/)
NOT_IN_TABLES = [("NInfer, October build", "n12_ninfer")]


def load(folder):
    rows = []
    for f in sorted(glob.glob(str(folder / "runs_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def flag(text, cat):
    """'.' = kept, 'T' = tool call without tools, 'L' = repetition loop, '?' = no saved answer (kept)"""
    if text is None:
        return "?"
    if "<tool_call>" in text or "<function=" in text:
        return "T"
    if cat == "structured":
        lines = [l for l in text.splitlines() if l.strip()]
        return "L" if len(lines) >= 20 and len(set(lines)) / len(lines) < 0.5 else "."
    tail = text[-4000:].encode("utf-8")
    return "L" if len(tail) >= 2000 and len(zlib.compress(tail, 9)) / len(tail) < 0.12 else "."


def n12_rows(folder, texts=None):
    """runs of the 12 prompts with their run index k and flag; tool_calls finishes and runs without a speed are dropped"""
    texts = texts or folder / "texts"
    seen = {}
    out = []
    for r in load(folder):
        k = r.get("k")
        if k is None:
            k = seen.get(r["fixture"], 0)
            seen[r["fixture"]] = k + 1
        r["k"] = k
        if r.get("finish") == "tool_calls" or not r.get("decode_tok_s"):
            continue
        p = texts / ("%s_r%d.txt" % (r["fixture"], k))
        r["_flag"] = flag(p.read_text(encoding="utf-8", errors="replace") if p.exists() else None, r.get("cat"))
        out.append(r)
    return out


def own_rows(rows):
    seen = {}
    for r in rows:
        if r.get("k") is None:
            r["k"] = seen.get(r["case"], 0)
            seen[r["case"]] = r["k"] + 1
    return [r for r in rows if r.get("decode_tok_s")]


def setting(name):
    """-> (runs of the 12 prompts, runs of the additional tasks) of one setting"""
    if name.startswith("nov3:"):
        base = NOV3 / name.split(":", 1)[1]
        with open(NOV3 / "own_tests" / "runs_MegaCapybara_Tiny.jsonl", encoding="utf-8") as fh:
            own = [json.loads(line) for line in fh if line.strip()]
        return n12_rows(base), own_rows(own)
    return n12_rows(RES / name / "n12"), own_rows(load(RES / name / "own"))


def stats(n12, own):
    """-> mean / best over all runs, mean_clean / best_clean without the flagged runs (7 columns each)"""
    out = {}
    for key, clean in (("", False), ("_clean", True)):
        mean, best = [], []
        for c in CATS:
            xs = [r for r in n12 if r.get("cat") == c and (not clean or r["_flag"] in (".", "?"))]
            if not xs:
                mean.append(None)
                best.append(None)
                continue
            mean.append(statistics.mean(r["decode_tok_s"] for r in xs))
            best.append(statistics.mean(max(r["decode_tok_s"] for r in xs if r["fixture"] == f)
                                        for f in sorted(set(r["fixture"] for r in xs))))
        for c in OWN:
            xs = [r["decode_tok_s"] for r in own if r.get("case") == c]
            mean.append(statistics.mean(xs) if xs else None)
            best.append(max(xs) if xs else None)
        out["mean" + key] = mean
        out["best" + key] = best
    out["runs"] = len(n12) + len(own)
    out["left_out"] = {"%s r%d" % (r["fixture"], r["k"]): r["_flag"] for r in n12 if r["_flag"] not in (".", "?")}
    return out


def colmax(lists):
    return [max((x[i] for x in lists if x[i] is not None), default=None) for i in range(7)]


def build():
    res = {"columns": COLS, "engines": {}, "yasakani": {}, "not_in_tables": {}}
    for eng, off, on in ENGINES:
        e = {}
        for mode, names in (("off", off), ("on", on)):
            per = {n: stats(*setting(n)) for n in names}
            e[mode] = {k: colmax([p[k] for p in per.values()]) for k in ("mean", "best", "mean_clean", "best_clean")}
            e[mode]["settings"] = per
        res["engines"][eng] = e
    for name, cfg in OURS:
        n12, own = setting("n12_ysk_" + cfg)
        t12, town = setting("n12t_ysk_" + cfg)
        t12 += n12_rows(MORE / ("n12t_ysk_" + cfg) / "n12")
        town += own_rows(load(MORE / ("n12t_ysk_" + cfg) / "own"))
        res["yasakani"][name] = {"off": stats(n12, own), "on": stats(t12, town)}
    for name, cfg in NOT_IN_TABLES:
        res["not_in_tables"][name] = {"setting": cfg, "off": stats(*setting(cfg))}
    return res


TABLES = [
    ("Thinking off: average without repetition loops and unrequested tool calls (the table of the README), tok/s", "off", "mean_clean"),
    ("Thinking on: fastest run of each task (the table of the README), tok/s", "on", "best"),
    ("Thinking off: average of all runs, tok/s", "off", "mean"),
    ("Thinking on: average of all runs, tok/s", "on", "mean"),
]


def markdown(res):
    f = lambda v: "-" if v is None else "{:,.0f}".format(v)
    md = ["# Speed tables made from the records", "",
          "Output of `scripts/tables.py`. The rules are in the docstring of the script; the conditions and limits are in the README of `bench/`.", ""]
    for title, mode, key in TABLES:
        md += ["## " + title, "", "| Engine | " + " | ".join(COLS) + " |", "|---|" + "---|" * 7]
        md += ["| %s | %s |" % (n, " | ".join(f(v) for v in o[mode][key])) for n, o in res["yasakani"].items()]
        md += ["| %s | %s |" % (n, " | ".join(f(v) for v in e[mode][key])) for n, e in res["engines"].items()]
        best = colmax([e[mode][key] for e in res["engines"].values()])
        md += ["| (largest value of the other engines) | %s |" % " | ".join(f(v) for v in best), ""]
    md += ["## Settings of each engine (thinking off: average of all runs / without the runs left out; thinking on: average / fastest run)", ""]
    for eng, e in res["engines"].items():
        for mode in ("off", "on"):
            for n, p in e[mode]["settings"].items():
                a, b = ("mean", "mean_clean") if mode == "off" else ("mean", "best")
                md.append("* %s, `%s`, thinking %s, %d runs: %s" % (eng, n, mode, p["runs"], "; ".join(
                    "%s %s / %s" % (c, f(x), f(y)) for c, x, y in zip(COLS, p[a], p[b]))))
    md += ["", "## Runs left out of the first table (thinking off)", ""]
    for n, o in res["yasakani"].items():
        if o["off"]["left_out"]:
            md.append("* %s: %s" % (n, ", ".join("%s %s" % kv for kv in o["off"]["left_out"].items())))
    for eng, e in res["engines"].items():
        for n, p in e["off"]["settings"].items():
            if p["left_out"]:
                md.append("* %s `%s`: %s" % (eng, n, ", ".join("%s %s" % kv for kv in p["left_out"].items())))
    on_out = sum(len(p["left_out"]) for e in res["engines"].values() for p in e["on"]["settings"].values()) + \
        sum(len(o["on"]["left_out"]) for o in res["yasakani"].values())
    md += ["", "T = tool call written although the request has no tools, L = repetition loop. Thinking on: %d runs flagged." % on_out, "",
           "## Measured the same day, not used in the tables", ""]
    for n, o in res["not_in_tables"].items():
        p = o["off"]
        md.append("* %s, `%s`, thinking off, %d runs: %s" % (n, o["setting"], p["runs"], "; ".join(
            "%s %s / %s" % (c, f(x), f(y)) for c, x, y in zip(COLS, p["mean"], p["mean_clean"]))))
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "tables"))
    a = ap.parse_args()
    res = build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = markdown(res)
    for name, text in (("speed_tables.json", json.dumps(res, ensure_ascii=False, indent=1) + "\n"), ("SPEED_TABLES.md", md)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:      # the same bytes on every system
            fh.write(text)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
