# -*- coding: utf-8 -*-
"""api_topk_dump.py -- the top-K log-probabilities of every prompt position from a server (teacher forcing).

The token ids of the windows come from the reference file (kld_base.py describes the format; 2,048 tokens per window).  vLLM:
/v1/completions with prompt = token ids and prompt_logprobs = K.  SGLang: /generate with input_ids, return_logprob, logprob_start_len 0,
top_logprobs_num K.  Rows = positions 1024..2046 (row r predicts token 1025 + r), as in the reference file.  The alignment is checked on
every row (the actual token's own log-prob must match its entry in the top-K when it is there).
Writes <out>/topk_<label>.{json,ids,lp,tgt} (format: see topk_score.py).  Python standard library only.
usage: python api_topk_dump.py --api vllm|sglang --base http://127.0.0.1:30000 --base-file FILE --label NAME [--k 100] [--out DIR]"""
import argparse, array, json, math, struct, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_tokens(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"_logits_"; n_ctx, V, n_chunk = struct.unpack("<Iii", f.read(12))
        t = array.array("i"); t.frombytes(f.read(4 * n_ctx * n_chunk))
    return n_ctx, V, n_chunk, [t[c * n_ctx:(c + 1) * n_ctx].tolist() for c in range(n_chunk)]


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--api", required=True, choices=["vllm", "sglang"]); ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--k", type=int, default=100); ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--meta", default="{}", help="JSON merged into the dump's json (engine, model, kv, ...)")
    ap.add_argument("--base-file", required=True); ap.add_argument("--out", default=str(HERE / "dumps")); a = ap.parse_args()
    n_ctx, V, n_chunk, wins = read_tokens(a.base_file); first = n_ctx // 2; rows = n_ctx - 1 - first; K = a.k
    ids = array.array("i", [-1] * (n_chunk * rows * K)); lps = array.array("f", [float("-inf")] * (n_chunk * rows * K)); tgt = array.array("f", [float("nan")] * (n_chunk * rows))
    checks = dict(rows=0, tgt_in_topk=0, tgt_mismatch=0, max_abs_diff=0.0); t0 = time.time()
    for w, seq in enumerate(wins):
        t1 = time.time()
        if a.api == "vllm":
            r = post(a.base + "/v1/completions", dict(model=a.model, prompt=seq, max_tokens=1, temperature=0, prompt_logprobs=K))
            plp = r["choices"][0]["prompt_logprobs"]; assert len(plp) == n_ctx, len(plp)
            def at(i):
                d = plp[i]; ent = [(int(t), float(v["logprob"]), v.get("rank")) for t, v in d.items()]
                top = sorted([e for e in ent if e[2] is not None and e[2] <= K], key=lambda e: e[2])[:K]
                own = d.get(str(seq[i])); return [(e[0], e[1]) for e in top], (float(own["logprob"]) if own else float("nan"))
        else:
            r = post(a.base + "/generate", dict(input_ids=seq, sampling_params=dict(max_new_tokens=1, temperature=0), return_logprob=True,
                                               logprob_start_len=0, top_logprobs_num=K))
            mi = r["meta_info"]; itl, itop = mi["input_token_logprobs"], mi["input_top_logprobs"]; assert len(itl) == n_ctx and len(itop) == n_ctx, (len(itl), len(itop))
            def at(i):
                assert int(itl[i][1]) == seq[i], ("sglang token order", i, itl[i][1], seq[i])
                top = sorted([(int(e[1]), float(e[0])) for e in (itop[i] or [])], key=lambda e: -e[1])[:K]
                return top, (float(itl[i][0]) if itl[i][0] is not None else float("nan"))
        for row in range(rows):
            i = first + 1 + row; top, own = at(i); o = (w * rows + row) * K
            for j, (tok, lp) in enumerate(top): ids[o + j] = tok; lps[o + j] = lp
            tgt[w * rows + row] = own; checks["rows"] += 1
            hit = [lp for tok, lp in top if tok == seq[i]]
            if hit:
                checks["tgt_in_topk"] += 1; d = abs(hit[0] - own)
                checks["max_abs_diff"] = max(checks["max_abs_diff"], d)
                if d > 1e-3: checks["tgt_mismatch"] += 1
        print(f"[dump] {a.label} window {w + 1}/{n_chunk} in {time.time() - t1:.1f} s", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = out / f"topk_{a.label}"
    with open(str(stem) + ".ids", "wb") as f: ids.tofile(f)
    with open(str(stem) + ".lp", "wb") as f: lps.tofile(f)
    with open(str(stem) + ".tgt", "wb") as f: tgt.tofile(f)
    meta = dict(label=a.label, api=a.api, k=K, windows=n_chunk, rows=rows, first=first, n_ctx=n_ctx, vocab=V, base=Path(a.base_file).name, checks=checks,
                seconds=round(time.time() - t0, 1), t=time.strftime("%Y-%m-%d %H:%M:%S"), **json.loads(a.meta))
    Path(str(stem) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[dump] done", json.dumps(checks), flush=True)
    return 0 if checks["tgt_mismatch"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
