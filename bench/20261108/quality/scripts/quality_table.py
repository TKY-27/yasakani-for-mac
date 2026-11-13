# -*- coding: utf-8 -*-
"""quality_table.py -- the table of "Quality on thinking text" from the score files of this folder.

Reads ../scores/*.json (one file per measured configuration) and writes QUALITY_TABLE.md and quality_table.json into .. (or into
--out DIR).  No number is typed here: sizes, KL, top-1 agreement and perplexity are the values of the score files.  The rows are
ordered by KL; a KL that is only a lower bound (taken from the top K tokens of the engine) is marked with ">=".
Python 3.8 or later, standard library only.

usage: python quality_table.py [--scores DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(scores):
    rows = []
    for p in sorted(Path(scores).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["file"] = p.name
        rows.append(d)
    return rows


def kl_text(r):
    if r["kl_kind"] == "whole vocabulary":
        return "%.4f" % r["kl"]
    return ">= %.4f (%s)" % (r["kl"], r["kl_kind"].replace("lower bound, ", ""))


def gb(n):
    return "%.2f" % (n / 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(HERE.parent / "scores"))
    ap.add_argument("--out", default=str(HERE.parent))
    a = ap.parse_args()
    allrows = load(a.scores)
    rows = sorted((r for r in allrows if r["in_table"]), key=lambda r: (r["kl"], -r["top1_agreement_pct"]))
    extra = [r for r in allrows if not r["in_table"]]
    ref = [r for r in rows if r.get("band_reference")]
    base_ppl = ref[0]["record"]["ppl_base"]
    positions = ref[0]["record"]["count"]

    md = ["# Quality on thinking text: all measured configurations", "",
          "Twenty windows of 2,048 tokens; the last 1,023 positions of every window (%s positions) against the BF16 reference "
          "(perplexity of the reference on these positions: %.4f). KL = mean KL divergence from the reference (nat, smaller is closer); "
          "top-1 agreement = how often the most likely next token is the same as the reference's. Sizes are in GB (10^9 bytes)."
          % ("{:,}".format(positions), base_ppl), "",
          "| # | Configuration | Size, GB | KV cache | KL | Top-1 agreement | Perplexity | In the table of docs/MEASUREMENTS.md | Score file |",
          "|---|---|---|---|---|---|---|---|---|"]
    out_rows = []
    for i, r in enumerate(rows, 1):
        size = gb(r["size_bytes"])
        if r.get("main_model_bytes"):
            size += " (main model %s)" % gb(r["main_model_bytes"])
        md.append("| %d | %s | %s | %s | %s | %.2f%% | %s | %s | `scores/%s` |" % (
            i, r["configuration"], size, r["kv_cache"], kl_text(r), r["top1_agreement_pct"],
            ("%.4f" % r["ppl"]) if r["ppl"] else "-", "yes" if r["in_documents_table"] else "no", r["file"]))
        out_rows.append(dict(rank=i, configuration=r["configuration"], size_bytes=r["size_bytes"], main_model_bytes=r.get("main_model_bytes"),
                             kv_cache=r["kv_cache"], kl=r["kl"], kl_kind=r["kl_kind"], top1_agreement_pct=r["top1_agreement_pct"], ppl=r["ppl"],
                             in_documents_table=r["in_documents_table"], score_file=r["file"]))

    md += ["", "## The same size bands", "",
           "KL of the GGUF files next to the two YASAKANI models in size, measured with llama.cpp, divided by the KL of the YASAKANI model "
           "of the band. \"Main model\" is the GGUF without its auxiliary layers (the tensors `blk.64.*`); YASAKANI's size is its model file.", "",
           "| Band | Configuration | File, GB | Main model, GB | KL | KL / KL of the YASAKANI model |", "|---|---|---|---|---|---|"]
    bands = []
    for band in sorted({r["band"] for r in rows if r.get("band")}, reverse=True):
        ours = [r for r in rows if r.get("band") == band and r.get("band_reference")][0]
        md.append("| %s | **%s** | %s | %s | **%.4f** | 1 |" % (band, ours["configuration"], gb(ours["size_bytes"]), gb(ours["size_bytes"]), ours["kl"]))
        others = []
        for r in rows:
            if r.get("band") != band or r.get("band_reference"):
                continue
            ratio = r["kl"] / ours["kl"]
            md.append("| | %s | %s | %s | %.4f | %.2f |" % (r["configuration"], gb(r["size_bytes"]), gb(r["main_model_bytes"]), r["kl"], ratio))
            others.append(dict(configuration=r["configuration"], size_bytes=r["size_bytes"], main_model_bytes=r["main_model_bytes"], kl=r["kl"],
                               top1_agreement_pct=r["top1_agreement_pct"], kl_ratio=ratio))
        bands.append(dict(band=band, yasakani=dict(configuration=ours["configuration"], size_bytes=ours["size_bytes"], kl=ours["kl"],
                                                   top1_agreement_pct=ours["top1_agreement_pct"]), others=others))

    if extra:
        md += ["", "## Further records (not rows of the table)", "", "| Record | KL | Top-1 agreement | Score file |", "|---|---|---|---|"]
        for r in sorted(extra, key=lambda r: r["file"]):
            md.append("| %s | %s | %.2f%% | `scores/%s` |" % (r["configuration"], kl_text(r) if r["kl"] is not None else "-", r["top1_agreement_pct"], r["file"]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "QUALITY_TABLE.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    with open(out / "quality_table.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(positions=positions, reference_ppl=base_ppl, rows=out_rows, bands=bands,
                                further_records=[dict(configuration=r["configuration"], kl=r["kl"], kl_kind=r["kl_kind"],
                                                      top1_agreement_pct=r["top1_agreement_pct"], score_file=r["file"])
                                                 for r in sorted(extra, key=lambda r: r["file"])]),
                           ensure_ascii=False, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
