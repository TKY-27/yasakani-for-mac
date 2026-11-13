#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repeat_summary.py -- counts the repeat checks of November 10, 2026 from their records (../checks/repeat/*.json).

usage:  python repeat_summary.py            (Python 3.8+, standard library only)

A record is the output of repeat_probe.py: one engine in one process, the same request generated several times ("runs") in each
"arm".  A run is counted as different when its token ids are not the most common output of its arm.  Runs are grouped by how they
were generated:
  default        the shipped settings
  no drafts      speculative decoding switched off for the request (off_nodraft: one token per pass)
  block off      only the block drafter switched off (off_noblock)
  selfcheck      the three requests of `yasakani selfcheck`, default settings
"""
import collections
import json
from pathlib import Path

REC = Path(__file__).resolve().parent.parent / "checks" / "repeat"
GROUPS = [
    ("nothing else used the GPU (utilisation 0% for 30 s before the start)", ["quiet_1", "quiet_2", "quiet_3", "quiet_4"]),
    ("other programs were using the GPU (the desktop was drawing)",
     ["speed_repeat_probe", "quality_repeat_probe", "env_A_base", "env_F_base2", "ctl_G_ctx262144", "ctl_H_ctx16384", "ctl_I_warmup",
      "ctl_J_nohash", "clk_engine_off4"]),
    ("another process used the GPU in short bursts", ["share_S1_bursts"]),
    ("another process used the GPU without pause", ["share_S2_nopause"]),
]
KINDS = ["default", "no drafts", "block off", "selfcheck"]


def kind_of(arm, tag):
    if arm.startswith("selfcheck"):
        return "selfcheck"
    t = tag.split("#")[0].split("/")[-1]
    return {"on": "default", "off_nodraft": "no drafts", "off_noblock": "block off"}[t]


def tally(name):
    d = json.loads((REC / (name + ".json")).read_text(encoding="utf-8"))
    out = collections.Counter()
    where = []
    for arm in d["arms"]:
        top = collections.Counter(r["ids_sha"] for r in arm["runs"]).most_common(1)[0][0]
        for r in arm["runs"]:
            k = kind_of(arm["arm"], r["tag"])
            out[(k, "runs")] += 1
            if r["ids_sha"] != top:
                out[(k, "different")] += 1
                at = r.get("first_diff_vs_run1")
                if at is None:                       # the first run of the arm is the different one: take the position from the others
                    at = next((x.get("first_diff_vs_run1") for x in arm["runs"] if x["ids_sha"] == top and x.get("first_diff_vs_run1") is not None), None)
                where.append("%s %s first different token #%s" % (arm["arm"], r["tag"], at))
    return d, out, where


def main():
    for title, names in GROUPS:
        total = collections.Counter()
        print("\n== " + title)
        for n in names:
            d, t, where = tally(n)
            total.update(t)
            print("  %-22s %s  %s  " % (n, d["session"], Path(d["model_dir"].replace("\\", "/")).name) +
                  ", ".join("%s %d of %d" % (k, t[(k, "different")], t[(k, "runs")]) for k in KINDS if t[(k, "runs")]))
            for w in where:
                print("        " + w)
        print("  TOTAL: " + ", ".join("%s %d of %d different" % (k, total[(k, "different")], total[(k, "runs")]) for k in KINDS if total[(k, "runs")]))


if __name__ == "__main__":
    main()
