#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""score_yasakani.py - measure a released YASAKANI model against a reference file, with the engine library of this repository.

  python score_yasakani.py --model-dir DIR --dll yskn.dll --reference REF.kld [--kv default|int8|fp16] [--out SCORE.json]

The reference file (made by make_reference.py, or any file in the format kld_base.py describes) holds the token ids of the
windows and the next-token distribution of the original model at the second half of every window.  This program lets the engine
read every window 16 tokens at a time (teacher forcing: the tokens of the window are given, nothing is generated), takes the
logits of the whole vocabulary at the same positions and computes, with the formulas of kld_base.py (the ones of llama.cpp's
perplexity tool): the mean KL divergence from the reference, how often the most likely next token is the same, and the perplexity.

--kv default  the engine as the package starts it: the compressed key-value cache KV5
--kv int8     the 8-bit key-value cache without the KV5 compression
--kv fp16     an FP16 key-value cache, the setting of the YASAKANI rows of the quality table in docs/MEASUREMENTS.md.  The
              yskn.dll of the releases is built for the 8-bit cache and refuses this setting; it needs an engine built for it.

Needs: Windows, an RTX 5090, the yasakani package of this repository (pip install . in python/, or the quickstart), numpy.
The engine is used through documented exports of yskn.dll (include/yskn_abi.h): yskn_verify_nll, yskn_logits_rows, yskn_commit.
"""
import argparse
import ctypes as C
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS_PER_PASS = 16


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--kv", choices=("default", "int8", "fp16"), default="default")
    ap.add_argument("--out")
    ap.add_argument("--scoring", default=str(HERE.parents[1] / "bench" / "20261108" / "quality" / "scripts"),
                    help="the folder of kld_base.py (default: bench/20261108/quality/scripts of this repository)")
    a = ap.parse_args()
    sys.path.insert(0, a.scoring)
    from kld_base import KldReader, metrics, summary
    import yasakani
    from yasakani.engine import Engine
    from yasakani.runtime import ModelPack, RuntimeProfile

    rd = KldReader(a.reference)
    n_ctx, first, rows, vocab = rd.n_ctx, rd.first, rd.rows, rd.V
    pack = ModelPack.load(a.model_dir)
    over = {"fp16": {"YSKN_Q35_KV8": "0", "YSKN_Q35_KV_FP16": "1", "YSKN_TB_CAP": "0"}, "int8": {"YSKN_TB_CAP": "0"}, "default": {}}[a.kv]
    eng = Engine(pack, max_ctx=n_ctx, dll=a.dll, profile=RuntimeProfile.default(pack, over))
    dll, e = eng.lib.dll, eng._e
    P, I = C.c_void_p, C.c_int
    dll.yskn_verify_nll.argtypes, dll.yskn_verify_nll.restype = [P, P, P, I, I, P, P], I
    dll.yskn_verify.argtypes, dll.yskn_verify.restype = [P, P, I, I, P, P, P], I
    dll.yskn_logits_rows.argtypes, dll.yskn_logits_rows.restype = [P, I, I, P], I
    dll.yskn_commit.argtypes, dll.yskn_commit.restype = [P, I], I
    print("engine %s, package %s, key-value cache: %s, %d windows of %d tokens, vocabulary %d"
          % (Path(a.dll).name, yasakani.__version__, a.kv, rd.n_chunk, n_ctx, vocab), flush=True)

    def check(rc):
        if rc != 0:
            raise SystemExit("engine call failed: " + eng.lib.last_error())

    acc = {k: [] for k in ("nll", "nll_base", "kld", "same_top", "p_diff")}
    again = None                      # whether the logits must be asked for with yskn_verify (decided on the first pass)
    argmax_mismatch, nll_max_diff = 0, 0.0
    t0 = time.perf_counter()
    for w in range(rd.n_chunk):
        seq = np.ascontiguousarray(rd.tokens[w], np.int32)
        eng.reset()
        out = np.empty((rows, vocab), np.float32)
        for pos in range(0, n_ctx - 1, ROWS_PER_PASS):
            n = min(ROWS_PER_PASS, n_ctx - 1 - pos)
            ids, tg = np.ascontiguousarray(seq[pos:pos + n]), np.ascontiguousarray(seq[pos + 1:pos + n + 1])
            am, nll = np.empty(n, np.int32), np.empty(n, np.float32)
            check(dll.yskn_verify_nll(e, ids.ctypes.data, tg.ctypes.data, n, 1, am.ctypes.data, nll.ctypes.data))
            lg = np.empty((n, vocab), np.float32)
            if again:
                check(dll.yskn_verify(e, ids.ctypes.data, n, 1, am.ctypes.data, None, None))
            check(dll.yskn_logits_rows(e, 0, n, lg.ctypes.data))
            if again is None:
                again = not np.array_equal(lg.argmax(1), am)
                if again:
                    check(dll.yskn_verify(e, ids.ctypes.data, n, 1, am.ctypes.data, None, None))
                    check(dll.yskn_logits_rows(e, 0, n, lg.ctypes.data))
                    if not np.array_equal(lg.argmax(1), am):
                        raise SystemExit("the logits of the rows do not belong to the pass that was just made")
            mx = lg.max(1)
            lse = np.log(np.exp(lg - mx[:, None], dtype=np.float32).astype(np.float64).sum(1)) + mx
            argmax_mismatch += int((lg.argmax(1) != am).sum())
            nll_max_diff = max(nll_max_diff, float(np.abs(lse - lg[np.arange(n), tg] - nll).max()))
            lo = max(pos, first)
            if pos + n > first:
                out[lo - first:pos + n - first] = lg[lo - pos:]
            check(dll.yskn_commit(e, n - 1))
        m = metrics(out, np.asarray(rd.chunk(w)), seq[first + 1:])
        for k in acc:
            acc[k].append(m[k])
        print("window %2d  KL %.5f  same top %.2f %%   (%.0f s)" % (w, float(np.mean(m["kld"])), 100 * float(np.mean(m["same_top"])),
                                                                  time.perf_counter() - t0), flush=True)
    sm = summary({k: np.concatenate(v) for k, v in acc.items()})
    eng.close()
    res = dict(model=Path(pack.weights).name, model_bytes=Path(pack.weights).stat().st_size, kv=a.kv,
               engine_settings={k: v for k, v in sorted(eng.profile.env.items()) if not k.endswith(("_SHEAD", "_PROFILE"))},
               engine=Path(a.dll).name, engine_sha256=sha256_file(a.dll), package_version=yasakani.__version__,
               reference=Path(a.reference).name, reference_bytes=Path(a.reference).stat().st_size,
               windows=int(rd.n_chunk), positions=int(sm["count"]), kl=sm["kld_mean"], top1_agreement_pct=sm["same_top_pct"],
               ppl=sm["ppl"], ppl_reference=sm["ppl_base"], kl_median=sm["kld_median"], kl_p99=sm["kld_p99"], kl_max=sm["kld_max"],
               kl_per_window=[float(np.mean(x)) for x in acc["kld"]],
               argmax_mismatch_logits_vs_engine=argmax_mismatch, nll_max_abs_diff_logits_vs_engine=nll_max_diff,
               seconds=round(time.perf_counter() - t0, 1))
    print("\nKL %.6f   same top %.4f %%   perplexity %.4f (reference %.4f)   %d positions"
          % (res["kl"], res["top1_agreement_pct"], res["ppl"], res["ppl_reference"], res["positions"]))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
