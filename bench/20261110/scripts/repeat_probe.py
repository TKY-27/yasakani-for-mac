#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repeat probe (2026-11-10): why did a few runs of the H test (onoff_eq.py) differ from the others?

The H test of the public candidate 20261110 found 3 / 4 of 174 on-against-off comparisons different and one on-against-on repeat
different per model.  In every such case ONE run of the group was the odd one, so the question is what makes a run odd:
    (a) the setting (plain decoding with one row per pass against passes with several rows),
    (b) what ran before it (state the engine keeps across reset), or
    (c) nothing we control (the same request does not always give the same ids).
This probe generates the SAME request many times in one process and records every run:

  steady     one setting repeated back to back (is a setting repeatable with itself?)
  alternate  on / off_nodraft alternating (does the neighbour matter?)
  after_long a 41K-token request first, then the short request three times (is the first run after a long one different?)
  selfcheck  the three fixed requests of `yasakani selfcheck`, repeated (how often would the self-check fail by itself?)

  python repeat_probe.py --model-dir M --dll D.dll --pypath P --out probe.json [--reps 8] [--quick]

Reads the prompts of onoff_eq.py (same folders).  Writes only --out.  Exit codes: 0 every arm gave one single output,
1 at least one arm gave more than one output, 2 the engine could not be opened, 3 the GPU is not idle.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(*a):
    print("[%s]" % dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def save(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--pypath", required=True, help="folder that holds the yasakani package")
    ap.add_argument("--tools", required=True, help="folder that holds onoff_eq.py and a_speed.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ctx", type=int, default=65536)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--selfcheck-reps", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="two repeats and no long input (a flow test)")
    ap.add_argument("--plan", choices=("full", "off4"), default="full",
                    help="off4: only four plain-decoding arms (off_nodraft) and one arm with the default settings")
    ap.add_argument("--warmup", action="store_true", help="run the warm-up of the command-line tools first (two short sampled generations)")
    ap.add_argument("--no-state-hash", action="store_true", help="do not read the state hash after each run")
    ap.add_argument("--env", action="append", default=[], metavar="YSKN_NAME=VALUE",
                    help="engine setting added to the default runtime profile (repeatable); for finding the cause only")
    ap.add_argument("--max-gpu-used-mib", type=int, default=6500)
    ap.add_argument("--max-gpu-util", type=int, default=25)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.pypath)
    sys.path.insert(0, a.tools)
    import onoff_eq as H                                           # prompts, samplings and the off settings of the H test
    from a_speed import gpu_busy, gpu_line

    prompts_dir = Path("<work_dir>") / "prompts"
    ninfer_dir = Path("<work_dir>") / "ninfer_corpus"
    cases = {label: (messages, think, heavy) for label, messages, think, heavy in H.load_cases(prompts_dir, ninfer_dir, False)}
    samplings = dict(H.SAMPLINGS)
    off = dict(H.OFF)
    variants = dict(on=None, off_nodraft=off["off_nodraft"], off_noblock=off["off_noblock"], off_nolookup=off["off_nolookup"])

    util, used = gpu_busy()
    log("GPU before start:", gpu_line(), "| median utilisation %s %%, memory used %s MiB" % (util, used))
    if used is not None and (used > a.max_gpu_used_mib or util > a.max_gpu_util):
        log("the GPU is not idle: nothing started")
        return 3

    import numpy as np
    from yasakani.runtime import EOS_IDS, ModelPack, RuntimeProfile
    from yasakani.engine import Engine, Sampling
    from yasakani.textio import ChatFormatter
    from yasakani import cli as ycli

    pack = ModelPack.load(a.model_dir)
    fmt = ChatFormatter(pack.root)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    overrides = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v.strip()
    data = dict(kind="repeat_probe", session=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_dir=H.mask(a.model_dir), dll=H.mask(a.dll),
                max_ctx=a.max_ctx, n=a.n, reps=a.reps, plan=a.plan, env_overrides=overrides, warmup=bool(a.warmup), state_hash=not a.no_state_hash, python=platform.python_version(), arms=[], summary={})
    log("plan %s, reps %d, max_ctx %d, warm-up %s, state hash %s, engine settings added to the defaults: %s" % (
        a.plan, a.reps, a.max_ctx, "yes" if a.warmup else "no", "no" if a.no_state_hash else "yes", overrides or "none"))
    log("model folder %s | dll %s" % (H.mask(a.model_dir), H.mask(a.dll)))
    try:
        eng = Engine(pack, max_ctx=a.max_ctx, dll=a.dll, profile=RuntimeProfile.default(pack, overrides))
    except Exception as ex:                                       # noqa: BLE001
        log("could not open the engine:", type(ex).__name__, H.mask(str(ex))[:300])
        data["error"] = "%s: %s" % (type(ex).__name__, H.mask(str(ex))[:300])
        save(out, data)
        return 2

    def sha(ids):
        return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()[:16]

    def one(ids, n, sp, spec, stop=EOS_IDS):
        eng.reset()
        eng.set_sampling(Sampling(**sp))
        t0 = time.perf_counter()
        res = eng.generate(ids, n, stop_ids=stop, spec=spec)
        st = None if a.no_state_hash else hashlib.sha256(np.asarray(eng.state_hash(), np.uint64).tobytes()).hexdigest()[:16]
        return dict(ids_sha=sha(res.ids), n_ids=len(res.ids), finish=res.finish, passes=res.passes, snap_mode=res.snap_mode,
                    cached=res.cached_tokens, bdr_rounds=res.stats.get("bdr_rounds"), bdr_d0=res.stats.get("bdr_d0"),
                    prefill_ms=round(res.prefill_ms, 1), gen_ms=round(res.gen_ms, 1), wall_s=round(time.perf_counter() - t0, 2),
                    state=st, _ids=list(res.ids))

    def finish_arm(name, runs, note=""):
        """runs: list of (tag, record).  Stores each distinct output once, and for every run where it first leaves the first run."""
        base = runs[0][1]["_ids"]
        distinct = {}
        rows = []
        for tag, r in runs:
            ids = r.pop("_ids")
            if r["ids_sha"] not in distinct:
                distinct[r["ids_sha"]] = ids
            k = next((i for i, (x, y) in enumerate(zip(ids, base)) if x != y), None)
            if k is None and len(ids) != len(base):
                k = min(len(ids), len(base))
            r["first_diff_vs_run1"] = k
            if k is not None and k < len(ids) and k < len(base):
                try:
                    r["diff_tokens"] = dict(run1=[base[k], fmt.decode([base[k]])], this=[ids[k], fmt.decode([ids[k]])])
                except Exception:                                 # noqa: BLE001
                    r["diff_tokens"] = dict(run1=[base[k], None], this=[ids[k], None])
            rows.append(dict(tag=tag, **r))
        count = Counter(r["ids_sha"] for r in rows)
        arm = dict(arm=name, note=note, runs=rows, outputs=len(distinct), counts=dict(count), distinct_ids=distinct)
        data["arms"].append(arm)
        save(out, data)
        log("%-44s runs %2d  distinct outputs %d  %s   passes %s" % (
            name, len(rows), len(distinct), dict(count), sorted(set(r["passes"] for r in rows))))
        return arm

    def prompt_of(label):
        messages, think, heavy = cases[label]
        ids, _ = fmt.prompt(messages, enable_thinking=think)
        return ids

    if a.warmup:
        from yasakani.chat import ChatRunner
        ChatRunner(eng, fmt).warmup()
        log("warm-up done")
    reps = 2 if a.quick else a.reps
    plan_steady = [("jp_q1", "greedy", "on"), ("jp_q1", "greedy", "off_nodraft"),
                   ("scenario_code_cuda", "greedy", "on"), ("scenario_code_cuda", "greedy", "off_nodraft"),
                   ("jp_q5", "seed1234", "on"), ("jp_q5", "seed1234", "off_nodraft"),
                   ("prose_en_cpu", "seed1234", "on"), ("prose_en_cpu", "seed1234", "off_nodraft"),
                   ("scenario_story_zh_scifi", "seed1234", "off_noblock"), ("scenario_code_cuda", "seed1234", "off_nodraft")]
    if a.plan == "off4":
        plan_steady = [("jp_q1", "greedy", "off_nodraft"), ("prose_en_cpu", "seed1234", "off_nodraft"),
                       ("jp_q5", "seed1234", "off_nodraft"), ("scenario_code_cuda", "greedy", "off_nodraft"),
                       ("jp_q5", "seed1234", "on")]
    try:
        # ---- steady: one setting back to back
        for label, sname, vname in plan_steady:
            ids = prompt_of(label)
            runs = [("%s#%d" % (vname, i + 1), one(ids, a.n, samplings[sname], variants[vname])) for i in range(reps)]
            finish_arm("steady/%s/%s/%s" % (label, sname, vname), runs)
        # ---- alternate: on / off_nodraft alternating
        for label, sname in (() if a.plan == "off4" else (("jp_q1", "greedy"), ("scenario_code_cuda", "greedy"))):
            ids = prompt_of(label)
            runs = []
            for i in range(max(2, reps // 2)):
                runs.append(("on#%d" % (i + 1), one(ids, a.n, samplings[sname], None)))
                runs.append(("off_nodraft#%d" % (i + 1), one(ids, a.n, samplings[sname], variants["off_nodraft"])))
            finish_arm("alternate/%s/%s" % (label, sname), runs)
        # ---- after a long input: is the first short run different?
        if not a.quick and a.plan == "full":
            long_ids = prompt_of("long")
            short = prompt_of("scenario_code_cuda")
            for cyc in range(3):
                l1 = one(long_ids, 128, samplings["greedy"], None)
                l2 = one(long_ids, 128, samplings["greedy"], variants["off_nodraft"])
                runs = [("after_long%d/on#%d" % (cyc + 1, i + 1), one(short, a.n, samplings["greedy"], None)) for i in range(3)]
                arm = finish_arm("after_long/cycle%d/scenario_code_cuda/greedy/on" % (cyc + 1), runs,
                                 note="before it: long (41K) on %s, off_nodraft %s" % (l1["ids_sha"], l2["ids_sha"]))
                l1.pop("_ids"), l2.pop("_ids")
                arm["long_runs"] = [l1, l2]
                save(out, data)
            r8 = prompt_of("rand8k_s5")
            for cyc in range(2):
                l1 = one(r8, 128, samplings["greedy"], None)
                runs = [("after_8k%d/on#%d" % (cyc + 1, i + 1), one(short, a.n, samplings["greedy"], None)) for i in range(2)]
                l1.pop("_ids")
                arm = finish_arm("after_8k/cycle%d/scenario_code_cuda/greedy/on" % (cyc + 1), runs, note="before it: rand8k_s5 on")
                arm["long_runs"] = [l1]
                save(out, data)
        # ---- the three requests of `yasakani selfcheck`, as the command runs them (greedy, default settings, one after the other)
        sc_rounds = 0 if a.plan == "off4" else (3 if a.quick else a.selfcheck_reps)
        per_case = {}
        for rnd in range(sc_rounds):
            for c in ycli._selfcheck_cases():
                p = fmt.encode(c["text"]) if c["kind"] == "raw" else fmt.prompt(c["messages"], **c["template_kwargs"])[0]
                per_case.setdefault(c["name"], []).append(("round%d" % (rnd + 1), one(p, c["max_new"], dict(temperature=0.0), None)))
        for name, runs in per_case.items():
            finish_arm("selfcheck/%s" % name, runs, note="greedy, default settings, the three cases in turn, %d rounds" % sc_rounds)
    finally:
        eng.close()
    multi = [x["arm"] for x in data["arms"] if x["outputs"] > 1]
    data["summary"] = dict(arms=len(data["arms"]), arms_with_one_output=len(data["arms"]) - len(multi), arms_with_more_outputs=multi,
                           runs=sum(len(x["runs"]) for x in data["arms"]))
    save(out, data)
    odd = sum(len(x["runs"]) - max(x["counts"].values()) for x in data["arms"])
    data["summary"]["runs_not_equal_to_the_most_common_output"] = odd
    save(out, data)
    log("repeat probe: %d arms, %d runs; arms with more than one output: %d %s" % (
        len(data["arms"]), data["summary"]["runs"], len(multi), multi))
    log("   runs that differ from the most common output of their arm: %d of %d   (settings added: %s)" % (
        odd, data["summary"]["runs"], overrides or "none"))
    return 1 if multi else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
