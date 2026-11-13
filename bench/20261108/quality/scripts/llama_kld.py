# -*- coding: utf-8 -*-
"""llama_kld.py -- the llama.cpp rows of the quality table: llama-perplexity's KL divergence against the reference file.

The stock llama-perplexity of llama.cpp b11700 with
  -m <gguf> -c 2048 -b 2048 -ngl 99 -fa on -ctk <kv> -ctv <kv> --kl-divergence-base <reference file> --kl-divergence
(the token ids come from the reference file; rows 1024..2046 of every 2,048-token window).  KL over the whole vocabulary.
Writes <out>/results/llama_<key>_<kv>_own.json and <out>/logs/llama_<key>_<kv>_own.log.
usage: python -X utf8 -u llama_kld.py --llama-perplexity EXE --base-file FILE --gguf KEY=FILE [--gguf KEY=FILE ...] [--kv f16] [--out DIR]"""
import argparse, hashlib, json, re, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_llama(text):
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, text)) else None)
    return dict(ppl=g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), ppl_base=g(r"Mean PPL\(base\)\s*:\s*([\d.]+)"),
                ln_ppl_ratio=g(r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*(-?[\d.]+)"), kld_mean=g(r"Mean\s+KLD:\s*(-?[\d.]+)"),
                kld_median=g(r"Median\s+KLD:\s*(-?[\d.]+)"), kld_p99=g(r"99\.0%\s+KLD:\s*(-?[\d.]+)"), kld_p999=g(r"99\.9%\s+KLD:\s*(-?[\d.]+)"),
                kld_max=g(r"Maximum KLD:\s*(-?[\d.]+)"), rms_dp_pct=g(r"RMS Δp\s*:\s*([\d.]+)"), same_top_pct=g(r"Same top p:\s*([\d.]+)"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kv", default="f16"); ap.add_argument("--llama-perplexity", required=True)
    ap.add_argument("--base-file", required=True); ap.add_argument("--gguf", action="append", required=True, help="KEY=FILE")
    ap.add_argument("--out", default=str(HERE / "out")); a = ap.parse_args()
    PPL = Path(a.llama_perplexity); BASE = Path(a.base_file); OUT = Path(a.out)
    GGUFS = {k: Path(v) for k, v in (x.split("=", 1) for x in a.gguf)}
    (OUT / "results").mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for key, g in GGUFS.items():
        out = OUT / "results" / f"llama_{key}_{a.kv}_own.json"; lg = OUT / "logs" / f"llama_{key}_{a.kv}_own.log"
        cmd = [str(PPL), "-m", str(g), "-c", "2048", "-b", "2048", "-ngl", "99", "-fa", "on", "-ctk", a.kv, "-ctv", a.kv,
               "--kl-divergence-base", str(BASE), "--kl-divergence"]
        t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        text = r.stdout + "\n" + r.stderr; lg.write_text(text, encoding="utf-8")
        res = dict(ok=r.returncode == 0, key=key, gguf=g.name, gguf_bytes=g.stat().st_size, kv=a.kv, flash_attn="on", ctx=2048, base=BASE.name,
                   base_bytes=BASE.stat().st_size, returncode=r.returncode, seconds=round(time.time() - t0, 1), **parse_llama(text),
                   version=next((l.strip() for l in text.splitlines() if "build:" in l or "version:" in l), None), t=time.strftime("%Y-%m-%d %H:%M:%S"))
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(key, a.kv, "rc", r.returncode, "KLD", res["kld_mean"], "same top", res["same_top_pct"], "PPL", res["ppl"], "base", res["ppl_base"], f"{res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
