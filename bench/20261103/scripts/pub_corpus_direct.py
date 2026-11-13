# -*- coding: utf-8 -*-
"""pub_corpus_direct.py (2026-11-02): NInfer's own published corpus on the PUBLIC YASAKANI build, through the public Python API
(no server), one engine process per suite.  Same fixtures, seeds and sampling as our earlier NInfer corpus runner ('matched' profile)
(T 0.6, top-p 0.95, top-k 20, min-p 0, presence 0; NInfer's five seeds mod 2^31; scenario = 12 fixtures x 5 seeds, thinking off,
4,096 budget; reasoning = 3 AIME26 fixtures x 5 seeds, thinking on, 65,536 budget).  No loop guard (as on the research server).
Speed = (completion_tokens - 1) / gen_ms as NInfer defines server-side decode.  The generated text is saved for the quality / loop checks.
usage: python pub_corpus_direct.py --suite scenario|reasoning --model-dir DIR --dll yskn.dll --out-dir DIR [--seeds 0,1,2,3,4]"""
import argparse, hashlib, json, time
from pathlib import Path

from yasakani.chat import ChatFormatter
from yasakani.engine import Engine, Sampling
from yasakani.runtime import ModelPack

FIX = Path("<ninfer_corpus_dir>")
SEEDS = (7632647173703958409, 7968175640111700217, 912910298659544128, 9060622443728853932, 4939353812939007330)
SCENARIO = dict(code=["scenario_code_cuda", "scenario_code_python", "scenario_code_typescript"],
                story=["scenario_story_zh_scifi", "scenario_story_en_mystery", "scenario_story_zh_dialogue"],
                translation=["scenario_translation_zh_en", "scenario_translation_en_zh", "scenario_translation_markdown"],
                structured=["scenario_structured_jsonl", "scenario_structured_csv", "scenario_structured_sql"])
REASONING = ["long_decode_aime26_01", "long_decode_aime26_15", "long_decode_aime26_30"]


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=("scenario", "reasoning")); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dll", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-ctx", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); (out / "texts").mkdir(exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    man = {c["name"]: c for c in json.load(open(FIX / "manifest.json", encoding="utf-8"))["cases"]}
    names = [(cat, n) for cat, ns in SCENARIO.items() for n in ns] if a.suite == "scenario" else [("reasoning", n) for n in REASONING]
    fixtures = []
    for cat, n in names:
        c = man[n]; p = FIX / (n + ".json")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["messages_sha256"], "fixture changed: " + n
        fixtures.append(dict(cat=cat, name=n, msgs=json.load(open(p, encoding="utf-8")), max_new=int(c["max_new"]), thinking=bool(c["thinking"])))
    pack = ModelPack.load(a.model_dir); fmt = ChatFormatter(pack.root)
    t0 = time.perf_counter(); eng = Engine(pack, max_ctx=a.max_ctx, dll=a.dll); log("engine ready in %.1f s, max_ctx %d" % (time.perf_counter() - t0, eng.max_ctx))
    wp, _ = fmt.prompt([{"role": "user", "content": "Write the numbers 1 to 40 as a JSON list."}], enable_thinking=False)
    eng.set_sampling(Sampling(0.6, 0.95, 20, 0.0, 1))
    for _ in range(2): eng.generate(wp, 160)
    log("warm-up done")
    runs = out / ("runs_%s.jsonl" % a.suite)
    for fx in fixtures:
        prompt, _ = fmt.prompt(fx["msgs"], enable_thinking=fx["thinking"])
        for k in seeds:
            seed = SEEDS[k] % 2 ** 31
            eng.set_sampling(Sampling(0.6, 0.95, 20, 0.0, seed))
            r = eng.generate(prompt, fx["max_new"])
            n = len(r.ids); gen_s = r.gen_ms / 1000.0
            text = fmt.decode(r.ids)
            (out / "texts" / ("%s_s%d.txt" % (fx["name"], k))).write_text(text, encoding="utf-8")
            rec = dict(engine="yasakani_public", profile="matched", suite=a.suite, cat=fx["cat"], fixture=fx["name"], k=k, seed=seed,
                       prompt_tokens=r.prompt_tokens, completion_tokens=n, finish=r.finish, passes=r.passes, prefill_ms=round(r.prefill_ms, 2),
                       gen_ms=round(r.gen_ms, 2), decode_tok_s=round((n - 1) / max(1e-6, gen_s), 2), tok_per_pass=round(n / max(1, r.passes), 3),
                       ms_per_pass=round(r.gen_ms / max(1, r.passes), 4), ids_sha256=hashlib.sha256(json.dumps(r.ids).encode()).hexdigest()[:16],
                       t=time.strftime("%Y-%m-%d %H:%M:%S"))
            with open(runs, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            log("%-11s %-30s s%d: %7.1f tok/s, %5d tokens in %7.2f s (%s), %d passes, %.2f tok/pass, %.2f ms/pass, prompt %d"
                % (fx["cat"], fx["name"], k, rec["decode_tok_s"], n, gen_s, r.finish, r.passes, rec["tok_per_pass"], rec["ms_per_pass"], r.prompt_tokens))
    eng.close(); log("done")


if __name__ == "__main__":
    main()
