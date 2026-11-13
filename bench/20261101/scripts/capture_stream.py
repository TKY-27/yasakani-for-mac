r"""Capture one streamed chat completion from any OpenAI-compatible server (our research server, llama-server, ...) with the arrival
time of every chunk, for the launch video (2026-10-13). The video is rendered from these records, so the on-screen timing is the measured one.
Records: request time 0, each chunk (t seconds since the POST, kind = reasoning_content / content, text), the usage block, and the summary
(TTFT = first non-empty chunk, decode tok/s = (completion_tokens - 1) / (t_last - t_first) as in the q27 harness).
usage: python capture_stream.py --base http://127.0.0.1:8099 (--prompt-file p.txt | --messages-file m.json) --out rec.json [--max_tokens 2048] [--temperature 1.0]
       [--top_p 0.95] [--top_k 20] [--min_p 0] [--seed N] [--no_think] [--extra '{"json": "fields"}'] [--label name]"""
import json
import sys
import time

import requests


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    base = arg("--base"); out = arg("--out"); label = arg("--label", "")
    prompt = open(arg("--prompt-file"), encoding="utf-8").read() if "--prompt-file" in a else arg("--prompt")
    msgs = json.load(open(arg("--messages-file"), encoding="utf-8")) if "--messages-file" in a else [{"role": "user", "content": prompt}]
    if prompt is None: prompt = msgs[-1]["content"]
    body = dict(model=arg("--model", "qwen3.8-27b"), messages=msgs, max_tokens=int(arg("--max_tokens", "2048")),
                stream=True, stream_options={"include_usage": True}, temperature=float(arg("--temperature", "1.0")), top_p=float(arg("--top_p", "0.95")),
                top_k=int(arg("--top_k", "20")), min_p=float(arg("--min_p", "0")))
    if "--seed" in a: body["seed"] = int(arg("--seed"))
    if "--no_think" in a: body["chat_template_kwargs"] = {"enable_thinking": False}
    if "--extra" in a: body.update(json.loads(arg("--extra")))
    chunks = []; usage = None; finish = None
    t0 = time.perf_counter()
    with requests.post(base + "/v1/chat/completions", json=body, stream=True, timeout=1800) as r:
        if r.status_code != 200: raise SystemExit(f"HTTP {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "): continue
            d = line[6:]
            if d == b"[DONE]": break
            t = time.perf_counter() - t0
            try: j = json.loads(d)
            except Exception: continue
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices") or []:
                dl = ch.get("delta") or {}
                for kind in ("reasoning_content", "reasoning", "content"):
                    if dl.get(kind): chunks.append([round(t, 6), "reasoning" if kind != "content" else "content", dl[kind]])
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    t_end = time.perf_counter() - t0
    first = chunks[0][0] if chunks else t_end; last = chunks[-1][0] if chunks else t_end
    gen = (usage or {}).get("completion_tokens") or len(chunks)
    rec = dict(label=label, base=base, request={k: v for k, v in body.items() if k != "messages"}, prompt=prompt[-2000:], n_messages=len(msgs), chunks=chunks, usage=usage, finish=finish,
               ttft_s=first, t_last_s=last, t_end_s=t_end, completion_tokens=gen, decode_tok_s=(gen - 1) / max(1e-6, last - first) if gen > 1 else 0.0,
               captured=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[capture] {label}: {gen} tokens, TTFT {first:.2f} s, decode {rec['decode_tok_s']:.1f} tok/s, {len(chunks)} chunks, finish {finish} -> {out}")


if __name__ == "__main__":
    main()
