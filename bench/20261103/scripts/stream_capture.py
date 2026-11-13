# -*- coding: utf-8 -*-
"""stream_capture.py (2026-11-03): record a streamed answer chunk by chunk (arrival time in ms + text) from any
OpenAI-compatible server, for replaying the real generation in a video.  Same fixtures / sampling / speed definition as
oai_bench_client.py (greedy, thinking off, decode tok/s = (completion_tokens - 1) / (last chunk - first content chunk)).
usage: python stream_capture.py --base URL --engine NAME --out-dir DIR --fixtures scenario_structured_jsonl,scenario_story_en_mystery [--max-new 4096]"""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

FIX = Path("<ninfer_corpus_dir>")


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def capture(base, body):
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    chunks = []; usage = None; finish = None
    t_send = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        buf = b""
        while True:
            data = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not data:
                break
            now = (time.perf_counter() - t_send) * 1000.0
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1); line = line.strip()
                if not line.startswith(b"data:"): continue
                payload = line[5:].strip()
                if payload == b"[DONE]": continue
                try: ev = json.loads(payload.decode("utf-8"))
                except Exception: continue
                if ev.get("usage"): usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content"); rc = d.get("reasoning_content") or d.get("reasoning")
                    if c: chunks.append([round(now, 2), c, "c"])
                    if rc: chunks.append([round(now, 2), rc, "r"])
                    if ch.get("finish_reason"): finish = ch["finish_reason"]
    return chunks, usage, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--engine", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fixtures", required=True); ap.add_argument("--max-new", type=int, default=0); ap.add_argument("--model", default="")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = a.model
    if not model:
        with urllib.request.urlopen(a.base + "/models", timeout=10) as r: model = json.loads(r.read())["data"][0]["id"]
    man = {c["name"]: c for c in json.load(open(FIX / "manifest.json", encoding="utf-8"))["cases"]}

    def body_for(msgs, max_new):
        return dict(model=model, messages=msgs, max_tokens=max_new, temperature=a.temperature, stream=True,
                    stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False})
    for _ in range(2):
        capture(a.base, body_for([{"role": "user", "content": "Write the numbers 1 to 40 as a JSON list."}], 160))
    log("warm-up done;", a.engine, model)
    for name in a.fixtures.split(","):
        c = man[name]; p = FIX / (name + ".json")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["messages_sha256"]
        msgs = json.load(open(p, encoding="utf-8")); max_new = int(a.max_new or c["max_new"])
        for k in range(a.repeats):
            chunks, usage, finish = capture(a.base, body_for(msgs, max_new))
            n = int((usage or {}).get("completion_tokens") or 0)
            content = [ch for ch in chunks if ch[2] == "c"]
            t_first = content[0][0] if content else None; t_last = content[-1][0] if content else None
            gen_ms = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
            dec = (n - 1) / (gen_ms / 1000.0) if (n > 1 and gen_ms > 0) else 0.0
            rec = dict(engine=a.engine, model=model, server=a.base, fixture=name, k=k, temperature=a.temperature, max_new=max_new,
                       prompt=msgs, prompt_tokens=(usage or {}).get("prompt_tokens"), completion_tokens=n, finish=finish,
                       ttft_ms=round(t_first, 1) if t_first is not None else None, gen_ms=round(gen_ms, 1), decode_tok_s=round(dec, 1),
                       n_chunks=len(chunks), chunks=chunks, usage=usage, t=time.strftime("%Y-%m-%d %H:%M:%S"))
            (out / ("%s_%s_r%d.json" % (a.engine.replace(" ", "_").replace("/", "-"), name, k))).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            log("%-30s r%d: %7.1f tok/s, %5d tokens, %d chunks, ttft %.0f ms, gen %.2f s, finish %s" % (name, k, dec, n, len(chunks), t_first or 0, gen_ms / 1000, finish))


if __name__ == "__main__":
    main()
