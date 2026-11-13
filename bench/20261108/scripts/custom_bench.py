# -*- coding: utf-8 -*-
"""custom_bench.py (2026-11-03): run cases.json (copy-heavy edits, long JSONL) against any OpenAI-compatible server, greedy,
thinking off, streaming; decode tok/s = (completion_tokens - 1) / (last content chunk - first content chunk), as in oai_bench_client.py.
For edit cases, also checks the answer against the expected text (similarity, changed lines). Saves texts and chunk timings.
usage: python custom_bench.py --base URL --label NAME --out-dir DIR [--cases cases.json] [--repeats 2]"""
import argparse, difflib, hashlib, json, time, urllib.request, urllib.error
from pathlib import Path

def log(*a): print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)

def stream(base, body):
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    chunks = []; usage = None; finish = None; t_send = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        buf = b""
        while True:
            data = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not data: break
            now = (time.perf_counter() - t_send) * 1000.0; buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1); line = line.strip()
                if not line.startswith(b"data:"): continue
                p = line[5:].strip()
                if p == b"[DONE]": continue
                try: ev = json.loads(p.decode("utf-8"))
                except Exception: continue
                if ev.get("usage"): usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}; c = d.get("content"); rc = d.get("reasoning_content") or d.get("reasoning")   # 2026-11-06: reasoning chunks count too (thinking on)
                    if c: chunks.append([round(now, 2), c])
                    elif rc: chunks.append([round(now, 2), rc, "r"])
                    if ch.get("finish_reason"): finish = ch["finish_reason"]
    return chunks, usage, finish

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--label", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", default=str(Path(__file__).with_name("cases.json"))); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model", default="")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--thinking", default="off", choices=["off", "on"])   # 2026-11-06
    ap.add_argument("--extra", default="", help="JSON merged into every request body"); ap.add_argument("--seed-per-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cases = json.load(open(a.cases, encoding="utf-8"))
    model = a.model
    if not model:
        with urllib.request.urlopen(a.base + "/models", timeout=10) as r: model = json.loads(r.read())["data"][0]["id"]
    extra = json.loads(a.extra) if a.extra else {}
    def body(msgs, n, k=None):
        b = dict(model=model, messages=msgs, max_tokens=n, temperature=a.temperature, stream=True, stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": a.thinking == "on"})
        b.update(extra)
        if a.seed_per_run and k is not None: b["seed"] = k
        return b
    for _ in range(2): stream(a.base, body([{"role": "user", "content": "Write the numbers 1 to 40 as a JSON list."}], 160))
    log("warm-up done", a.label, model)
    rows = []
    for c in cases:
        for k in range(a.repeats):
            try:
                chunks, usage, finish = stream(a.base, body(c["messages"], c["max_new"], k))
            except urllib.error.HTTPError as e:   # 2026-11-06: servers that reject top_k / min_p / seed
                if e.code != 400: raise
                b = body(c["messages"], c["max_new"], k); [b.pop(x, None) for x in ("top_k", "min_p", "seed")]; log("HTTP 400; retrying without top_k / min_p / seed"); chunks, usage, finish = stream(a.base, b)
            n = int((usage or {}).get("completion_tokens") or 0); pt = (usage or {}).get("prompt_tokens")
            t0 = chunks[0][0] if chunks else 0; t1 = chunks[-1][0] if chunks else 0; gen = (t1 - t0) / 1000.0
            dec = (n - 1) / gen if (n > 1 and gen > 0) else 0.0
            text = "".join(ch[1] for ch in chunks if len(ch) < 3)   # content only (reasoning chunks are tagged "r")
            rec = dict(label=a.label, model=model, case=c["name"], k=k, prompt_tokens=pt, completion_tokens=n, finish=finish, ttft_ms=round(t0, 1), gen_s=round(gen, 3), decode_tok_s=round(dec, 1), chunks=len(chunks), temperature=a.temperature, thinking=a.thinking, usage=usage)
            if c.get("expected"):
                exp = c["expected"]; got = text.strip()
                if got.startswith("```"): got = got.split("\n", 1)[1] if "\n" in got else got; got = got.rsplit("```", 1)[0] if got.rstrip().endswith("```") else got
                sm = difflib.SequenceMatcher(None, exp.strip(), got.strip(), autojunk=False)
                el, gl = exp.strip().split("\n"), got.strip().split("\n")
                diff = [l for l in difflib.unified_diff(el, gl, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
                rec.update(similarity=round(sm.ratio(), 5), exact=(got.strip() == exp.strip()), diff_lines=len(diff), out_lines=len(gl), exp_lines=len(el))
            if c["name"].startswith("jsonl"):
                ok = 0; lines = [l for l in text.split("\n") if l.strip()]
                for l in lines:
                    try: json.loads(l); ok += 1
                    except Exception: pass
                rec.update(jsonl_lines=len(lines), jsonl_valid=ok)
            rows.append(rec)
            tag = "%s_%s_r%d" % (a.label, c["name"], k)
            (out / (tag + ".txt")).write_text(text, encoding="utf-8")
            (out / (tag + ".chunks.json")).write_text(json.dumps(dict(rec=rec, chunks=chunks), ensure_ascii=False), encoding="utf-8")
            with open(out / ("runs_%s.jsonl" % a.label), "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log("%-16s r%d: %7.1f tok/s, %5d tokens (prompt %s) in %6.2f s, %s%s" % (c["name"], k, dec, n, pt, gen, finish,
                ("" if "similarity" not in rec else ", sim %.4f exact %s difflines %d" % (rec["similarity"], rec["exact"], rec["diff_lines"])) +
                ("" if "jsonl_valid" not in rec else ", jsonl %d/%d valid" % (rec["jsonl_valid"], rec["jsonl_lines"]))))
    log("done")

if __name__ == "__main__":
    main()
