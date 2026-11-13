# -*- coding: utf-8 -*-
"""topk_score.py -- score the top-K log-probabilities of an engine against the reference: the rows of the table marked ">=".

Input: a dump made by api_topk_dump.py, <dumps>/topk_<label>.json (k, windows, rows, ...), .ids int32 [windows, rows, K] (-1 = none),
.lp float32 (log-probs), .tgt float32 [windows, rows] (the actual next token's log-prob, NaN if not given).  Per row: top-1 agreement (the
engine's most likely token = the reference's), and the KL(reference || engine) over the engine's top-K tokens plus one bucket for all the
others, which is a lower bound of the KL over the whole vocabulary.  The reference side as in llama.cpp (log-probs below -16 count as 0).
Perplexity from .tgt when it is given for every row.  Writes <out>/score_<label>.json.  Needs numpy and kld_base.py (this folder).
usage: python topk_score.py --base-file FILE --label NAME [--dumps DIR] [--out DIR]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kld_base import KldReader, base_logprobs                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--label", required=True); ap.add_argument("--base-file", required=True)
    ap.add_argument("--dumps", default=str(HERE / "dumps")); ap.add_argument("--out", default=str(HERE / "dumps")); a = ap.parse_args()
    stem = Path(a.dumps) / f"topk_{a.label}"; meta = json.loads(Path(str(stem) + ".json").read_text(encoding="utf-8"))
    W, R, K = meta["windows"], meta["rows"], meta["k"]
    ids = np.fromfile(str(stem) + ".ids", np.int32).reshape(W, R, K); lp = np.fromfile(str(stem) + ".lp", np.float32).reshape(W, R, K)
    tg = np.fromfile(str(stem) + ".tgt", np.float32).reshape(W, R)
    rd = KldReader(a.base_file); V = rd.V; toks = np.asarray(rd.tokens); first = rd.first; assert (W, R) == (rd.n_chunk, rd.rows)
    kl, same, nll, nllb, kl_w = [], [], [], [], []
    for w in range(W):
        lpb = base_logprobs(np.asarray(rd.chunk(w)), V); pb = np.where(lpb > np.float32(-16), np.exp(lpb, dtype=np.float32), np.float32(0))
        S = ids[w]; ok = S >= 0; Sc = np.clip(S, 0, V - 1); lqs = np.where(ok, lp[w], -np.inf).astype(np.float64)
        ps = np.where(ok, np.take_along_axis(pb, Sc, 1), 0).astype(np.float64); lps = np.take_along_axis(lpb, Sc, 1).astype(np.float64)
        term = np.where(ps > 0, ps * (lps - np.where(ok, lqs, 0.0)), 0.0).sum(1)
        prest = np.clip(pb.astype(np.float64).sum(1) - ps.sum(1), 0.0, None); qrest = np.clip(1.0 - np.where(ok, np.exp(lqs), 0.0).sum(1), 1e-30, None)
        term += np.where(prest > 1e-30, prest * (np.log(np.maximum(prest, 1e-30)) - np.log(qrest)), 0.0)
        top_eng = Sc[np.arange(R), np.argmax(lqs, 1)]; same.append(top_eng == lpb.argmax(1)); kl.append(term); kl_w.append(float(term.mean()))
        targets = toks[w, first + 1:]; nllb.append(-lpb[np.arange(R), targets].astype(np.float64)); nll.append(-tg[w].astype(np.float64))
    rd.close()
    kl = np.concatenate(kl); same = np.concatenate(same); nll = np.concatenate(nll); nllb = np.concatenate(nllb); known = np.isfinite(nll)
    res = dict(label=a.label, k=K, rows=int(kl.size), kld_topk_mean=float(kl.mean()), kld_topk_median=float(np.median(kl)), same_top_pct=float(100 * same.mean()),
               ppl=float(np.exp(nll[known].mean())) if known.all() else None, ppl_base=float(np.exp(nllb.mean())), tgt_known_pct=float(100 * known.mean()),
               kld_topk_per_window=kl_w, dump=meta)
    (Path(a.out) / f"score_{a.label}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[score] {a.label}: top-{K} KL {res['kld_topk_mean']:.5f}, same top {res['same_top_pct']:.2f} %, PPL {res['ppl']} (base {res['ppl_base']:.4f}), rows {res['rows']}", flush=True)


if __name__ == "__main__":
    main()
