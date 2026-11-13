#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""H: does switching the speed-up off change what the model writes?  (public yskn.dll + public Python layer, one engine in this process)

For every prompt and every sampling setting the same request is generated
    on   the shipped settings (the drafters are used)
    on2  the same again (is the engine repeatable at all?)
    off_* the drafters switched off in different ways (per-request settings of the generation call)
and the generated ids are compared one by one.  The number of passes shows that "off" really works differently (on: few passes per
token, off: one pass per token).  The final state hash is stored too (information only: a drafter may leave its own traces there).

  python onoff_eq.py --model-dir M --dll D.dll --pypath P --out H.json [--max-ctx 65536] [--n 256] [--quick]

Exit codes: 0 everything compared is equal, 1 at least one difference, 2 the engine could not be opened / no comparison could be made,
3 the GPU is not idle (nothing was started).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

USERPATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\"' ]+")

# text prompts of the 2026-10-18 measurement: (label, file, thinking, heavy)
TEXT_PROMPTS = [("chat", "chat.txt", True, False), ("explain", "explain.txt", True, False), ("code", "code.txt", True, False),
                ("math", "math.txt", True, False), ("json", "json.txt", True, False),
                ("prose_en_cpu", "prose3_en_cpu.txt", True, False), ("prose_ja_essay", "prose3_ja_essay.txt", True, False),
                ("prose_en_story", "prose3_en_story.txt", True, False)] + \
               [("jp_q%d" % i, "japanese_q%d.txt" % i, False, False) for i in range(1, 6)] + \
               [("rand8k_s%d" % i, "sglang_random_8k_%d.txt" % i, False, True) for i in range(1, 6)] + \
               [("long", "long.txt", True, True)]
# message lists of the NInfer corpus: (file, thinking)
NINFER_PROMPTS = [("scenario_%s.json" % n, False) for n in (
    "code_cuda", "code_python", "code_typescript", "story_en_mystery", "story_zh_dialogue", "story_zh_scifi", "structured_csv",
    "structured_jsonl", "structured_sql", "translation_en_zh", "translation_markdown", "translation_zh_en")] + \
    [("long_decode_aime26_%s.json" % n, True) for n in ("01", "15", "30")]
QUICK = ("chat", "code", "jp_q1")
SAMPLINGS = [("greedy", dict(temperature=0.0, top_p=1.0, top_k=20, min_p=0.0, seed=0)),
             ("seed1234", dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, seed=1234))]
# per-request settings that switch the drafters off (names are those of the public generation call)
OFF = [("off_nodraft", dict(use_d0=0, use_d1=0, bdraft=0)),            # nothing is proposed: every pass checks the one id it must write anyway
       ("off_noblock", dict(bdraft=0)),                                # only the small look-up drafters
       ("off_nolookup", dict(use_d0=0, use_d1=0))]                     # only the block drafter
# long inputs (8K, 41K): greedy only, "on" against "off_nodraft" only


def mask(s):
    return USERPATH.sub("~", str(s))


def log(*a):
    print("[%s]" % dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def h_ids(ids):
    import numpy as np
    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()[:16]


def h_state(eng):
    import numpy as np
    return hashlib.sha256(np.asarray(eng.state_hash(), np.uint64).tobytes()).hexdigest()[:16]


def save(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_cases(prompts_dir, ninfer_dir, quick):
    out = []
    for label, fname, think, heavy in TEXT_PROMPTS:
        if quick and label not in QUICK:
            continue
        out.append((label, [dict(role="user", content=(prompts_dir / fname).read_text(encoding="utf-8"))], think, heavy))
    if not quick:
        for fname, think in NINFER_PROMPTS:
            out.append((fname[:-5], json.loads((ninfer_dir / fname).read_text(encoding="utf-8")), think, False))
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--pypath", help="folder that holds the yasakani package")
    ap.add_argument("--prompts", help="folder with the 10/18 prompt files")
    ap.add_argument("--ninfer-dir", help="folder with the NInfer corpus (message lists)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ctx", type=int, default=65536)
    ap.add_argument("--n", type=int, default=256, help="ids to generate per run (default 256)")
    ap.add_argument("--n-heavy", type=int, default=128, help="ids to generate for the 8K / 41K inputs")
    ap.add_argument("--quick", action="store_true", help="three prompts only (a flow test)")
    ap.add_argument("--max-gpu-used-mib", type=int, default=6500)
    ap.add_argument("--max-gpu-util", type=int, default=25)
    ap.add_argument("--no-gpu-check", action="store_true")
    a = ap.parse_args(argv)
    if a.pypath:
        sys.path.insert(0, a.pypath)
    prompts_dir = Path(a.prompts) if a.prompts else Path("<work_dir>") / "prompts"
    ninfer_dir = Path(a.ninfer_dir) if a.ninfer_dir else Path("<work_dir>") / "ninfer_corpus"
    cases = load_cases(prompts_dir, ninfer_dir, a.quick)                 # fails early on a missing file

    if not a.no_gpu_check:
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        try:
            from a_speed import gpu_busy, gpu_line
            util, used = gpu_busy()
            log("GPU before start:", gpu_line(), "| median utilisation %s %%, memory used %s MiB" % (util, used))
            if used is not None and (used > a.max_gpu_used_mib or util > a.max_gpu_util):
                log("the GPU is not idle: nothing started")
                return 3
        except ImportError:
            log("a_speed.py not found: the GPU check is skipped")

    from yasakani.runtime import EOS_IDS, ModelPack
    from yasakani.engine import Engine, Sampling
    from yasakani.textio import ChatFormatter

    pack = ModelPack.load(a.model_dir)
    fmt = ChatFormatter(pack.root)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dict(kind="onoff_eq", session=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_dir=mask(a.model_dir), dll=mask(a.dll),
                max_ctx=a.max_ctx, n=a.n, n_heavy=a.n_heavy, off_variants=dict(OFF), python=platform.python_version(), cases=[], summary={})
    try:
        eng = Engine(pack, max_ctx=a.max_ctx, dll=a.dll)
    except Exception as ex:                                     # noqa: BLE001
        log("could not open the engine:", type(ex).__name__, mask(str(ex))[:300])
        data["error"] = "%s: %s" % (type(ex).__name__, mask(str(ex))[:300])
        save(out, data)
        return 2
    data["spec_default"] = dict(eng.spec)
    n_cmp = n_eq = 0
    n_err = 0
    first_diff = []
    try:
        for label, messages, think, heavy in cases:
            ids, _in_reasoning = fmt.prompt(messages, enable_thinking=think)
            n = a.n_heavy if heavy else a.n
            for sname, sp in SAMPLINGS:
                if heavy and sname != "greedy":
                    continue
                variants = [("on", None), ("off_nodraft", dict(OFF)["off_nodraft"])] if heavy else \
                           [("on", None)] + OFF + [("on2", None)]
                runs = {}
                for vname, spec in variants:
                    rec = {}
                    try:
                        eng.reset()
                        eng.set_sampling(Sampling(**sp))
                        t0 = time.perf_counter()
                        res = eng.generate(ids, n, stop_ids=EOS_IDS, spec=spec)
                        rec = dict(ids_sha=h_ids(res.ids), n_ids=len(res.ids), finish=res.finish, passes=res.passes,
                                   gen_ms=round(res.gen_ms, 1), prefill_ms=round(res.prefill_ms, 1), wall_s=round(time.perf_counter() - t0, 2),
                                   state=h_state(eng), head=[int(x) for x in res.ids[:6]])
                        rec["_ids"] = list(res.ids)
                    except Exception as ex:                      # noqa: BLE001
                        rec = dict(error="%s: %s" % (type(ex).__name__, mask(str(ex))[:300]))
                        n_err += 1
                    runs[vname] = rec
                base = runs.get("on", {})
                eq = {}
                for vname in runs:
                    if vname == "on":
                        continue
                    r = runs[vname]
                    if "error" in r or "error" in base:
                        eq[vname] = None
                        continue
                    same = r["_ids"] == base["_ids"] and r["finish"] == base["finish"]
                    eq[vname] = same
                    if vname != "on2":
                        n_cmp += 1
                        n_eq += 1 if same else 0
                    if not same and len(first_diff) < 6:
                        k = next((i for i, (x, y) in enumerate(zip(r["_ids"], base["_ids"])) if x != y), min(len(r["_ids"]), len(base["_ids"])))
                        first_diff.append("%s/%s/%s: first difference at id %d" % (label, sname, vname, k))
                for r in runs.values():
                    r.pop("_ids", None)
                data["cases"].append(dict(prompt=label, sampling=sname, n_prompt=int(len(ids)), n_request=n, runs=runs, equal_to_on=eq))
                save(out, data)
                log("%-15s %-9s %s" % (label, sname, " ".join(
                    "%s=%s" % (k, {True: "same", False: "DIFF", None: "ERR"}[v]) for k, v in eq.items())) +
                    "   passes on=%s off_nodraft=%s (ids %s)" % (base.get("passes"), runs.get("off_nodraft", {}).get("passes"), base.get("n_ids")))
    finally:
        eng.close()
    repeat_bad = [c for c in data["cases"] if c["equal_to_on"].get("on2") is False]
    # does "off" really work differently?  on: few passes per id; off_nodraft: about one pass per id
    mech = [c for c in data["cases"] if c["runs"].get("on", {}).get("passes") and c["runs"].get("off_nodraft", {}).get("passes")]
    mech_ok = [c for c in mech if c["runs"]["off_nodraft"]["passes"] >= 1.5 * c["runs"]["on"]["passes"]]
    data["summary"] = dict(comparisons=n_cmp, equal=n_eq, different=n_cmp - n_eq, errors=n_err, on_repeat_not_equal=len(repeat_bad),
                           cases_checked_for_mechanism=len(mech), cases_with_off_nodraft_at_least_1_5x_passes=len(mech_ok),
                           first_differences=first_diff)
    save(out, data)
    log("H: %d comparisons of on against off, %d equal, %d different/not compared, %d engine errors; on-against-on not equal in %d case(s)" % (
        n_cmp, n_eq, n_cmp - n_eq, n_err, len(repeat_bad)))
    log("   mechanism check: off_nodraft needed at least 1.5x the passes of 'on' in %d of %d case(s)" % (len(mech_ok), len(mech)))
    for d in first_diff:
        log("  ", d)
    if n_cmp == 0:
        return 2
    return 0 if (n_eq == n_cmp and not repeat_bad) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
