#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""wf_to_summary.py - turn the runs.jsonl of compare/wf_*/run_wf.py (other engines) into the tables of measure_public/.

  python wf_to_summary.py --runs RUNS.jsonl [RUNS2.jsonl ...] [--tags x1,x2,x1l,x2l] --old-summary results_summary.csv
         --out-summary results_summary_others_DATE.csv --out-runs results_runs_others_DATE.csv --out-prefill prefill_others_DATE.md
  python wf_to_summary.py selftest --runs <oct18_run_dir>/runs.jsonl --old-summary results_summary.csv --old-runs results_runs.csv

The rule is the one of the 2026-10-18 table (measure_public/README.md): a run counts when it wrote at least 100 tokens; the number of an engine
for one test is the MEDIAN of all counted runs of one setting, and for each engine the setting with the highest median is shown ("fastest
setting").  The selftest re-derives all 30 numbers of the 10/18 table from its own runs.jsonl and must give exactly the same values.
The 'yasakani' column of the new summary is copied from the old summary (the 10/18 value): a_speed.py summary shows it as the old value and
computes the new ratios from the new run of the public build.  The ratio column is left empty on purpose.
"""
import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

T_LM4 = "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 (LMHead4)"
T_NV = "nvidia/Qwen3.8-27B-NVFP4"
T_LL = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (IQ3_S-mtp)"
D_DF2 = "z-lab/Qwen3.8-27B-DFlash2"
D_MTP = "built-in MTP head"
D_SGDS = "gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4"
D_VLDS = "RedHatAI/Qwen3.8-27B-speculator.dspark"
D_LLDF = "ggml-org/Qwen3.8-27B-GGUF (dflash-Qwen3.8-27B-Q4_0)"

# base configuration: (engine, setting, target weights, draft weights)
BASE = {
    "llama_mtp2": ("llama.cpp", "MTP", T_LL, D_MTP),
    "llama_dflq4": ("llama.cpp", "DFlash 2", T_LL, D_LLDF),
    "sg_dflash_lm4": ("SGLang", "DFlash 2", T_LM4, D_DF2),
    "sg_dflash_nv": ("SGLang", "DFlash 2 (official 5090 recipe)", T_NV, D_DF2),
    "sg_mtp_nv": ("SGLang", "MTP", T_NV, D_MTP),
    "sg_dspark_lm4": ("SGLang", "DSpark", T_LM4, D_SGDS),
    "vl_dflash_lm4": ("vLLM", "DFlash 2", T_LM4, D_DF2),
    "vl_mtp_lm4": ("vLLM", "MTP", T_LM4, D_MTP),
    "vl_dspark_lm4": ("vLLM", "DSpark", T_LM4, D_VLDS),
}
MODS = {"norc": "radix cache off", "48k": "48K context", "nopc": "prefix caching off", "eager": "eager mode"}
SIX = ("chat", "explain", "code", "math", "json", "long")
ORDER = ("chat", "explain", "code", "math", "json", "long", "prose3", "japanese", "ninfer_prose", "sglang_random_8k")
ENGINES = ("SGLang", "vLLM", "llama.cpp")


def test_of(r):
    s, p = r["set"], r["prompt"]
    return {"v8prose": "prose3", "long": "long", "jp": "japanese", "ninfer": "ninfer_prose", "rand8k": "sglang_random_8k"}.get(s, p if s in ("h2h", "h2hx") else None)


def parse_cfg(cfg):
    """(engine, setting, variant, target weights, draft weights) of a run_wf configuration name, or None (ours, unknown)."""
    parts = cfg.split("+")
    base = parts[0]
    if base not in BASE:
        return None
    mods = parts[1:]
    if any(m not in MODS for m in mods):
        return None
    eng, setting, tw, dw = BASE[base]
    variant = ", ".join(MODS[m] for m in mods) if mods else "-"
    return eng, setting, variant, tw, dw


def label(setting, variant):
    return setting if variant == "-" else "%s (%s)" % (setting, variant)


def load(paths, tags):
    runs = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("tag") == "smoke" or parse_cfg(r["cfg"]) is None:
                continue
            if tags and r["tag"] not in tags:
                continue
            if test_of(r) is None:
                continue
            runs.append(r)
    return runs


def used(r):
    return (r.get("completion_tokens") or 0) >= 100 and r.get("decode_tok_s") is not None


def best_per_test(runs):
    """{test: {engine: (median, cfg, n)}} - the fastest setting (highest median of its counted runs)."""
    g = collections.defaultdict(list)
    for r in runs:
        if used(r):
            g[(parse_cfg(r["cfg"])[0], r["cfg"], test_of(r))].append(r["decode_tok_s"])
    out = {}
    for (e, cfg, t), v in g.items():
        cand = (statistics.median(v), cfg, len(v))
        cur = out.setdefault(t, {}).get(e)
        if cur is None or cand[0] > cur[0]:
            out[t][e] = cand
    return out


def write_summary(path, best, old, fill_old=False):
    """fill_old: an engine / test that was not measured again keeps the value of the old table, marked in the setting column."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "six_task", "yasakani", "SGLang", "SGLang_setting", "vLLM", "vLLM_setting", "llama.cpp", "llama.cpp_setting",
                    "fastest_other", "ratio_vs_fastest_other"])
        for t in ORDER:
            b = best.get(t, {})
            row = [t, "yes" if t in SIX else "no", old.get(t, {}).get("yasakani", "")]
            top = None
            for e in ENGINES:
                if e in b:
                    med, cfg, n = b[e]
                    p = parse_cfg(cfg)
                    row += ["%.1f" % med, label(p[1], p[2])]
                    if top is None or med > top[0]:
                        top = (med, e)
                elif fill_old and t in old and e in old[t]:
                    row += ["%.1f" % old[t][e], old[t][e + "_setting"] + " [value of 2026-10-18, not measured again]"]
                    if top is None or old[t][e] > top[0]:
                        top = (old[t][e], e)
                else:
                    row += ["", ""]
            row += [top[1] if top else "", ""]
            w.writerow(row)


def write_runs(path, runs):
    cols = ["test", "six_task", "engine", "setting", "variant", "target_weights", "draft_weights", "round", "prompt", "run", "decode_tok_s",
            "completion_tokens", "prompt_tokens", "finish", "used", "note"]
    cnt = collections.Counter()
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            eng, setting, variant, tw, dw = parse_cfg(r["cfg"])
            t = test_of(r)
            k = (r["cfg"], t, r["prompt"])
            cnt[k] += 1
            ok = used(r)
            w.writerow([t, "yes" if t in SIX else "no", eng, setting, variant, tw, dw, 1, r["prompt"], cnt[k], "%.1f" % (r["decode_tok_s"] or 0),
                        r.get("completion_tokens") or 0, r.get("prompt_tokens") or 0, r.get("finish") or "", "yes" if ok else "no",
                        "" if ok else "refused or stopped early (under 100 tokens), not a speed"])


def write_prefill(path, runs, date):
    """Time to the first token (prompt reading) of the long input (first request of each setting = cold) and of the random 8K prompts."""
    first = {}
    for r in runs:
        if r["set"] == "long" and r.get("ttft_s") and r.get("prompt_tokens") and r["cfg"] not in first:
            first[r["cfg"]] = r
    r8 = collections.defaultdict(list)
    for r in runs:
        if r["set"] == "rand8k" and r.get("ttft_s") and r.get("prompt_tokens"):
            r8[r["cfg"]].append(r)
    lines = ["# Prompt reading time of the other engines (%s)" % date, "",
             "Time to the first streamed token (seconds) and prompt tokens per second.  Long input: the FIRST request of each setting (the next ones can "
             "hit a prefix cache).  8K: five different random prompts, median.  Answer-writing speed is not in this table.", "",
             "| engine | setting | 8K input: s | 8K: tok/s | 41K input: s (first request) | 41K: tok/s | note |", "|---|---|---:|---:|---:|---:|---|"]
    cfgs = sorted(set(first) | set(r8), key=lambda c: (parse_cfg(c)[0], c))
    for c in cfgs:
        eng, setting, variant, _, _ = parse_cfg(c)
        a = r8.get(c)
        s8 = statistics.median([x["ttft_s"] for x in a]) if a else None
        t8 = statistics.median([x["prompt_tokens"] / x["ttft_s"] for x in a]) if a else None
        f = first.get(c)
        note = ""
        if f is not None and (f.get("completion_tokens") or 0) < 100:
            note = "refused or failed"
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            eng, label(setting, variant), "%.2f" % s8 if s8 else "", "%.0f" % t8 if t8 else "",
            "%.2f" % f["ttft_s"] if f and not note else "", "%.0f" % (f["prompt_tokens"] / f["ttft_s"]) if f and not note else "", note))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_old(path):
    out = {}
    if path and Path(path).is_file():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            out[r["test"]] = dict(yasakani=r["yasakani"], **{e: float(r[e]) for e in ENGINES}, **{e + "_setting": r[e + "_setting"] for e in ENGINES})
    return out


def selftest(args):
    runs = load(args.runs, None)
    best = best_per_test(runs)
    old = read_old(args.old_summary)
    bad = 0
    for t in ORDER:
        for e in ENGINES:
            got = round(best[t][e][0], 1)
            want = old[t][e]
            ok = abs(got - want) < 0.05
            lab = label(*parse_cfg(best[t][e][1])[1:3])
            ok_lab = lab == old[t][e + "_setting"]
            if not (ok and ok_lab):
                bad += 1
                print("MISMATCH", t, e, got, want, lab, "|", old[t][e + "_setting"])
    print("summary: %d of %d numbers and settings reproduced" % (30 - bad, 30))
    # the runs table: counts and the unused ones
    if args.old_runs:
        rows = list(csv.DictReader(open(args.old_runs, encoding="utf-8")))
        orows = [r for r in rows if r["engine"] != "YASAKANI"]
        mine = [r for r in runs if True]
        print("runs: old csv %d other-engine rows, mine %d; unused old %d, unused mine %d" % (
            len(orows), len(mine), sum(r["used"] != "yes" for r in orows), sum(not used(r) for r in mine)))
    write_prefill("/tmp/wf_prefill_selftest.md", runs, "selftest 2026-10-18")
    print(Path("/tmp/wf_prefill_selftest.md").read_text(encoding="utf-8"))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="convert", choices=("convert", "selftest"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tags")
    ap.add_argument("--old-summary")
    ap.add_argument("--old-runs")
    ap.add_argument("--out-summary")
    ap.add_argument("--out-runs")
    ap.add_argument("--out-prefill")
    ap.add_argument("--date", default="")
    ap.add_argument("--fill-from-old", action="store_true", help="keep the old value (marked) where the engine / test was not measured again")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    tags = set(a.tags.split(",")) if a.tags else None
    runs = load(a.runs, tags)
    if not runs:
        print("no runs for these tags", file=sys.stderr)
        return 1
    best = best_per_test(runs)
    old = read_old(a.old_summary)
    if a.out_summary:
        write_summary(a.out_summary, best, old, a.fill_from_old)
    if a.out_runs:
        write_runs(a.out_runs, runs)
    if a.out_prefill:
        write_prefill(a.out_prefill, runs, a.date)
    print("runs: %d (counted %d); tests with numbers: %d of %d" % (len(runs), sum(used(r) for r in runs), len(best), len(ORDER)))
    for t in ORDER:
        b = best.get(t, {})
        print("%-18s" % t + "  ".join("%s %s" % (e, "%.1f %s" % (b[e][0], label(*parse_cfg(b[e][1])[1:3])) if e in b else "-") for e in ENGINES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
