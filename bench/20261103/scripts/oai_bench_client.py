# -*- coding: utf-8 -*-
"""oai_bench_client.py (2026-11-03): measure any OpenAI-compatible server on NInfer's 12 scenario fixtures the way
pub_corpus_direct.py measures the public YASAKANI build, so the numbers are comparable:
  decode tok/s = (completion_tokens - 1) / (time of the last streamed chunk - time of the first streamed content chunk)
  (prefill excluded), completion_tokens from the server's usage (stream_options.include_usage), thinking off, greedy by default.
Standard library only (urllib).  Writes runs_<label>.jsonl + texts/ + a summary table.
usage: python oai_bench_client.py --base http://127.0.0.1:8097/v1 --label mc_tiny --out-dir DIR [--cats code,story,...]
       [--max-new 0] [--repeats 2] [--temperature 0] [--model NAME] [--extra '{"top_k":1}'] [--no-thinking-kw]"""
import argparse, hashlib, json, statistics, sys, time, urllib.request, urllib.error
from pathlib import Path

FIX = Path("<ninfer_corpus_dir>")
SCENARIO = dict(code=["scenario_code_cuda", "scenario_code_python", "scenario_code_typescript"],
                story=["scenario_story_zh_scifi", "scenario_story_en_mystery", "scenario_story_zh_dialogue"],
                translation=["scenario_translation_zh_en", "scenario_translation_en_zh", "scenario_translation_markdown"],
                structured=["scenario_structured_jsonl", "scenario_structured_csv", "scenario_structured_sql"])


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def get_json(url, timeout=10):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stream_chat(base, body, timeout=1800):
    """POST a streaming chat completion; returns (text, usage, finish, t_send, t_first, t_last, n_chunks, raw_first)."""
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    text = []; usage = None; finish = None; t_first = None; t_last = None; n_chunks = 0; raw_first = None
    t_send = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            now = time.perf_counter()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    ev = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content")
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    if c or rc:
                        if t_first is None:
                            t_first = now; raw_first = payload[:300].decode("utf-8", "replace")
                        t_last = now; n_chunks += 1
                        if c: text.append(c)
                        if rc: text.append(rc)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    return "".join(text), usage, finish, t_send, t_first, t_last, n_chunks, raw_first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--label", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cats", default=""); ap.add_argument("--max-new", type=int, default=0); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--model", default="")
    ap.add_argument("--extra", default="", help="JSON merged into every request body")
    ap.add_argument("--no-thinking-kw", action="store_true", help="do not send chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--engine", default="", help="engine name written into the records")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); (out / "texts").mkdir(exist_ok=True)
    model = a.model
    if not model:
        ms = get_json(a.base + "/models"); model = ms["data"][0]["id"]
    log("server", a.base, "model", model)
    man = {c["name"]: c for c in json.load(open(FIX / "manifest.json", encoding="utf-8"))["cases"]}
    names = [(cat, n) for cat, ns in SCENARIO.items() for n in ns]
    if a.cats: names = [(c, n) for c, n in names if c in a.cats.split(",")]
    fixtures = []
    for cat, n in names:
        c = man[n]; p = FIX / (n + ".json")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["messages_sha256"], "fixture changed: " + n
        fixtures.append(dict(cat=cat, name=n, msgs=json.load(open(p, encoding="utf-8")), max_new=int(a.max_new or c["max_new"])))
    extra = json.loads(a.extra) if a.extra else {}

    def body_for(msgs, max_new):
        b = dict(model=model, messages=msgs, max_tokens=max_new, temperature=a.temperature, stream=True,
                 stream_options={"include_usage": True})
        if not a.no_thinking_kw:
            b["chat_template_kwargs"] = {"enable_thinking": False}
        b.update(extra)
        return b
    # warm-up
    for _ in range(2):
        stream_chat(a.base, body_for([{"role": "user", "content": "Write the numbers 1 to 40 as a JSON list."}], 160))
    log("warm-up done")
    runs = out / ("runs_%s.jsonl" % a.label); rows = []
    for fx in fixtures:
        for k in range(a.repeats):
            text, usage, finish, t_send, t_first, t_last, n_chunks, raw_first = stream_chat(a.base, body_for(fx["msgs"], fx["max_new"]))
            n = int((usage or {}).get("completion_tokens") or 0)
            gen_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
            ttft = (t_first - t_send) if t_first else None
            dec = (n - 1) / gen_s if (n > 1 and gen_s > 0) else 0.0
            (out / "texts" / ("%s_r%d.txt" % (fx["name"], k))).write_text(text, encoding="utf-8")
            rec = dict(engine=a.engine or model, profile=a.label, server=a.base, model=model, cat=fx["cat"], fixture=fx["name"], k=k,
                       temperature=a.temperature, max_new=fx["max_new"], prompt_tokens=(usage or {}).get("prompt_tokens"),
                       completion_tokens=n, finish=finish, ttft_s=round(ttft, 3) if ttft else None, gen_s=round(gen_s, 3),
                       decode_tok_s=round(dec, 2), wall_tok_s=round(n / (t_last - t_send), 2) if (t_last and n) else None,
                       chunks=n_chunks, usage=usage, text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                       extra=extra, t=time.strftime("%Y-%m-%d %H:%M:%S"))
            rows.append(rec)
            with open(runs, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log("%-11s %-30s r%d: %7.1f tok/s, %5d tokens in %7.2f s (%s), ttft %s, %d chunks"
                % (fx["cat"], fx["name"], k, dec, n, gen_s, finish, ("%.2f" % ttft) if ttft else "-", n_chunks))
    # summary
    lines = ["label %s  model %s  temperature %.2f" % (a.label, model, a.temperature), "%-11s %-30s %8s %8s %8s %6s" % ("cat", "fixture", "mean", "min", "max", "ntok")]
    by = {}
    for r in rows: by.setdefault((r["cat"], r["fixture"]), []).append(r)
    for (c, f), rs in by.items():
        v = [r["decode_tok_s"] for r in rs]
        lines.append("%-11s %-30s %8.1f %8.1f %8.1f %6.0f" % (c, f, statistics.mean(v), min(v), max(v), statistics.mean(r["completion_tokens"] for r in rs)))
    cats = {}
    for r in rows: cats.setdefault(r["cat"], []).append(r["decode_tok_s"])
    lines.append("per category (mean of runs): " + ", ".join("%s %.1f" % (c, statistics.mean(v)) for c, v in cats.items()))
    (out / ("summary_%s.txt" % a.label)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
