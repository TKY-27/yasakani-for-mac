r"""Same-PC comparison against the server engines (2026-10-18). One RTX 5090, Windows 11; ours runs on Windows (<research_server>.py, today's production), the others in WSL2
Ubuntu 24.04 with their official 5090 settings (serve\*.sh: SGLang cookbook, vLLM recipe, TensorRT-LLM generic template).
Each phase starts ONE server alone (shared gpu_lock, GPU idle check), sends warm-ups, runs the job sets through ..\capture_stream.py
(chunk-by-chunk timing, decode tok/s = (completion_tokens - 1) / (t_last - t_first)), and stops it. Ours runs first and last (drift check).
Job sets (same request body for every engine; presence penalty 0 everywhere because our server has none):
  v8prose : the launch video's prose prompts (CPU explainer, JA essay, mystery story), thinking on at the model card's T 1.0 / 0.95 / 20 / 0,
            1,024 tokens, seeds 1-3.
  h2h     : the head-to-head chat + explain prompts, same thinking settings, 2,048 tokens, seeds 1-3.
  ninfer  : NInfer's own published corpus, prose fixtures (story x3, translation x3), its T 0.6 / 0.95 / 20 profile, thinking off,
            its 4,096-token budget, seeds k = 0, 1.
  jp      : the five Japanese questions of the ChatGPT comparison, thinking off, Qwen's non-thinking T 0.7 / 0.8 / 20, 2,048 tokens.
  rand8k  : SGLang cookbook's speed benchmark shape (random 8,192-token input, 1,024 output, ignore_eos), thinking off, greedy.
  h2hx    : (added 2026-10-18 evening) the rest of the launch video's six thinking tasks:
            code, math, JSON, same THINK settings, 2,048 tokens, seeds 1-3 (prompts ..\ninfer\prompts, as in the 10-17 m6 run).
  long    : the six tasks' long-input prompt (41,209-token input), THINK, 2,048 tokens, seeds 1-3; needs the +48k configs.
Writes captures\<tag>\<cfg>_<set>_<prompt>_s<seed>.json and appends runs.jsonl. Server logs in logs\.
usage: python run_wf.py [--plan main] [--only cfg1,cfg2] [--sets v8prose,h2h,ninfer,jp,rand8k] [--deadline-min N]
2026-11-01: copy of the 10/18 script for the same-day re-measure (plans x1 / x1long / x2 / x2long); see the changes marked 2026-11-01.
2026-11-06 (run_wf_n12.py): plan n12 = every
SGLang / vLLM / llama.cpp setting of 11/1 (the same servers, models and launch scripts), measured with the 11/3 head-to-head protocol instead
of capture_stream jobs: NInfer's 12 fixtures greedy x 2 (oai_bench_client.py, 4,096) + the own tests (custom_bench.py, 8,192), thinking off,
and the public YASAKANI build (standard model) on its own server last, as a same-day check. Captures / logs / runs.jsonl go to THIS folder."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent; KS = HERE.parents[2]; CMP = KS / "compare"; LOGS = HERE / "logs"; LOGS.mkdir(exist_ok=True)   # 2026-11-06
OAI = HERE / "oai_bench_client.py"; CUST = HERE / "custom_bench.py"
CASES = HERE.parents[1] / "20261103" / "prompts" / "cases.json"; RESULTS = HERE / "results"
YSK_DLL = Path("<yasakani_dir_nov3>") / "bin" / "yskn.dll"; YSK_STD = Path("<model_dir_nov3>")
YSK_DLL_V8 = Path("<yasakani_dir>") / "bin" / "yskn.dll"   # 2026-11-07: the build of November 7 ("v8"), the yskn.dll of this release
SRC = CMP / "<oct18_run_dir>"   # 2026-11-01 copy: captures, logs and runs.jsonl go to THIS folder; prompts / serve scripts are read from the 10/18 folder
sys.path.append(r"<helpers_dir>")
from gpu_lock import gpu_lock

PY = sys.executable; CAP = CMP / "capture_stream.py"
WSL = ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--"]
SERVE = "<oct18_run_dir>/serve"
WBASE = "http://127.0.0.1:30000"
LLAMA = r"<llama_cpp_dir>/llama-server.exe"
G_IQ3S = r"<gguf_dir>/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
DFL_Q4 = r"<gguf_dir>/dflash-Qwen3.8-27B-Q4_0.gguf"
LBASE = ["-ngl", "99", "-c", "16384", "-fa", "on", "--parallel", "1", "--jinja", "--no-webui", "--host", "127.0.0.1", "--port", "8091"]
LBASE48 = [x if x != "16384" else "49152" for x in LBASE]


def wsl_cfg(name, pattern, extra=()):
    return dict(engine=name.split("_")[0], base=WBASE, cmd=WSL + ["bash", f"{SERVE}/{name.split('+')[0]}.sh", *extra],
                stop=WSL + ["pkill", "-INT", "-f", pattern], ready_timeout=3000)


CFGS = {
    "yasakani": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900,
                     cmd=[PY, str(CMP / "<research_server>.py"), "--port", "8099", "--max_ctx", "16384", "--log", "LOG"], stop=None),
    # 2026-11-06: the public candidate build of 11/3 with the standard model, its own OpenAI-compatible server (needs PYTHONPATH = the public py folder);
    # /v1/models answers before the model is loaded, so readiness is /health == ok
    "ysk_std_1103": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                         cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(YSK_STD), "--dll", str(YSK_DLL), "--host", "127.0.0.1",
                              "--port", "8099", "--max-tokens-cap", "16384"]),
    # 2026-11-07: the same 11/3 public DLL with the FAST bundle (pure 3-bit body, 12-bit embedding), everything else as ysk_std_1103
    "ysk_fast_1103": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                          cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(Path("<model_dir_fast_nov3>")), "--dll", str(YSK_DLL),
                               "--host", "127.0.0.1", "--port", "8099", "--max-tokens-cap", "16384"]),
    # 2026-11-07: the build of November 7 ("v8") with the same FAST bundle, everything else as ysk_fast_1103
    "ysk_fast_v8": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                        cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(Path("<model_dir_fast_nov3>")), "--dll", str(YSK_DLL_V8),
                             "--host", "127.0.0.1", "--port", "8099", "--max-tokens-cap", "16384"]),
    # 2026-11-07: the build of November 7 ("v8") with the STANDARD bundle, everything else as ysk_std_1103
    "ysk_std_v8": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                       cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(YSK_STD), "--dll", str(YSK_DLL_V8), "--host", "127.0.0.1",
                            "--port", "8099", "--max-tokens-cap", "16384"]),
    # 2026-11-08: the build of November 7 ("v8") with the two models of this release, the quality model (ysk_quality_v8) and the
    # fastest model (ysk_speed_v8); everything else as ysk_std_v8
    "ysk_quality_v8": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                           cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(Path("<model_dir_quality>")),
                                "--dll", str(YSK_DLL_V8), "--host", "127.0.0.1", "--port", "8099", "--max-tokens-cap", "16384"]),
    "ysk_speed_v8": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                         cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(Path("<model_dir_fastest>")),
                              "--dll", str(YSK_DLL_V8), "--host", "127.0.0.1", "--port", "8099", "--max-tokens-cap", "16384"]),
    # 2026-11-06: the October installs in WSL2 (compare/ninfer, compare/tensorfold): NInfer rev 81c8ce09 with the official NVFP4 v3 artifact and its
    # fastest single-request setting (DFlash 2 x7, LM-head draft, FP8 KV, prefill chunk 1024, no prefix reuse); TensorFold venv (0.6.6 on 10/15,
    # upgraded by job 1003 when the upgrade works) with MLX 4-bit + the z-lab DFlash 2 drafter, CUDA backend, prompt cache off
    "ninfer": dict(engine="ninfer", base="http://127.0.0.1:18080", ready_timeout=900, cmd=WSL + ["bash", "<launch_dir>/serve_ninfer.sh"],
                   stop=WSL + ["pkill", "-INT", "-f", "ninfer-serve"]),
    "ninfer_master": dict(engine="ninfer", base="http://127.0.0.1:18080", ready_timeout=900, cmd=WSL + ["bash", "<launch_dir>/serve_ninfer_master.sh"],
                          stop=WSL + ["pkill", "-INT", "-f", "ninfer-serve"]),   # 2026-11-06: today's master (job 079)
    "tensorfold": dict(engine="tensorfold", base="http://127.0.0.1:18090", ready_timeout=2400, cmd=WSL + ["bash", "<launch_dir>/serve_tensorfold.sh"],
                       stop=WSL + ["pkill", "-INT", "-f", "tensorfold serve"]),
    # 2026-11-06: the standard weights with the official z-lab DFlash 2 drafter as converted on 10/11; the 11/3 DLL, or the 10/31 DLL
    # as the fallback
    "ysk_offdraft": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                         cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(Path("<model_dir_other_helper>")), "--dll", str(YSK_DLL),
                              "--host", "127.0.0.1", "--port", "8099", "--max-tokens-cap", "16384"]),
    "ysk_offdraft_dll1031": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900, stop=None, ready_path="/health", ready_text='"ok"',
                         cmd=[PY, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(Path("<model_dir_other_helper>")),
                              "--dll", str(Path("<yasakani_dir_oct31>") / "bin" / "yskn.dll"), "--host", "127.0.0.1", "--port", "8099", "--max-tokens-cap", "16384"]),
    "sg_dflash_nv": wsl_cfg("sg_dflash_nv", "sglang.launch_server"),
    "sg_dflash_lm4": wsl_cfg("sg_dflash_lm4", "sglang.launch_server"),
    "sg_dspark_lm4": wsl_cfg("sg_dspark_lm4", "sglang.launch_server"),
    "sg_mtp_nv": wsl_cfg("sg_mtp_nv", "sglang.launch_server"),
    "vl_dflash_lm4": wsl_cfg("vl_dflash_lm4", "vllm serve"),
    "vl_mtp_lm4": wsl_cfg("vl_mtp_lm4", "vllm serve"),
    "vl_dspark_lm4": wsl_cfg("vl_dspark_lm4", "vllm serve"),
    # vLLM's recipe: a single 5090 needs --enforce-eager (CUDA-graph capture OOM). Used only when the graph run cannot start.
    "vl_dflash_lm4+eager": wsl_cfg("vl_dflash_lm4+eager", "vllm serve", ["--enforce-eager"]),
    "vl_mtp_lm4+eager": wsl_cfg("vl_mtp_lm4+eager", "vllm serve", ["--enforce-eager"]),
    "vl_dspark_lm4+eager": wsl_cfg("vl_dspark_lm4+eager", "vllm serve", ["--enforce-eager"]),
    "trt_mtp_nv": wsl_cfg("trt_mtp_nv", "trtllm-serve"),
    "trt_mtp_nv_fi": wsl_cfg("trt_mtp_nv_fi", "trtllm-serve"),
    # llama.cpp (Ollama's engine), the OFFICIAL b11700 Windows CUDA build, native on Windows like ours: the launch video's two fastest
    # settings (prose = IQ3_S + MTP 2, code = IQ3_S + DFlash 2 Q4_0 sidecar n-max 15), 16K context
    "llama_mtp2": dict(engine="llama", base="http://127.0.0.1:8091", ready_timeout=900, stop=None,
                       cmd=[LLAMA, "-m", G_IQ3S, *LBASE, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
    "llama_dflq4": dict(engine="llama", base="http://127.0.0.1:8091", ready_timeout=900, stop=None,
                        cmd=[LLAMA, "-m", G_IQ3S, *LBASE, "-md", DFL_Q4, "-ngld", "all", "--spec-type", "draft-dflash", "--spec-draft-n-max", "15"]),
    # the same DFlash 2 script after --disable-radix-cache was added (its first run had the radix cache on)
    "sg_dflash_lm4+norc": wsl_cfg("sg_dflash_lm4+norc", "sglang.launch_server"),
    # 48K-context variants for the 41K-token long-input task (same settings otherwise; argparse keeps the last value of a repeated flag)
    "yasakani+48k": dict(engine="yasakani", base="http://127.0.0.1:8099", ready_timeout=900,
                         cmd=[PY, str(CMP / "<research_server>.py"), "--port", "8099", "--max_ctx", "49152", "--log", "LOG"], stop=None),
    "llama_mtp2+48k": dict(engine="llama", base="http://127.0.0.1:8091", ready_timeout=900, stop=None,
                           cmd=[LLAMA, "-m", G_IQ3S, *LBASE48, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
    "llama_dflq4+48k": dict(engine="llama", base="http://127.0.0.1:8091", ready_timeout=900, stop=None,
                            cmd=[LLAMA, "-m", G_IQ3S, *LBASE48, "-md", DFL_Q4, "-ngld", "all", "--spec-type", "draft-dflash", "--spec-draft-n-max", "15"]),
    "sg_dflash_lm4+48k": wsl_cfg("sg_dflash_lm4+48k", "sglang.launch_server", ["--context-length", "49152"]),
    "sg_dflash_nv+48k": wsl_cfg("sg_dflash_nv+48k", "sglang.launch_server", ["--context-length", "49152"]),
    "sg_mtp_nv+48k": wsl_cfg("sg_mtp_nv+48k", "sglang.launch_server", ["--context-length", "49152"]),
    "sg_dspark_lm4+48k": wsl_cfg("sg_dspark_lm4+48k", "sglang.launch_server", ["--context-length", "49152", "--max-total-tokens", "49152"]),
    "vl_dflash_lm4+48k": wsl_cfg("vl_dflash_lm4+48k", "vllm serve", ["--max-model-len", "49152"]),
    "vl_mtp_lm4+48k": wsl_cfg("vl_mtp_lm4+48k", "vllm serve", ["--max-model-len", "49152"]),
    "vl_dspark_lm4+48k": wsl_cfg("vl_dspark_lm4+48k", "vllm serve", ["--max-model-len", "49152"]),
    # 2026-10-18: vLLM + DFlash 2 at 41K died on the 2nd request of the same prompt (EngineCore "CUDA error: an illegal memory access"),
    # i.e. when the prefix cache hit; the retry turns prefix caching off (prefill only, decode is the same work)
    "vl_dflash_lm4+48k+nopc": wsl_cfg("vl_dflash_lm4+48k+nopc", "vllm serve", ["--max-model-len", "49152", "--no-enable-prefix-caching"]),
    "vl_dspark_lm4+48k+nopc": wsl_cfg("vl_dspark_lm4+48k+nopc", "vllm serve", ["--max-model-len", "49152", "--no-enable-prefix-caching"]),
}
FALLBACK = {"vl_dflash_lm4": "vl_dflash_lm4+eager", "vl_mtp_lm4": "vl_mtp_lm4+eager", "vl_dspark_lm4": "vl_dspark_lm4+eager", "ysk_offdraft": "ysk_offdraft_dll1031"}

THINK = ["--temperature", "1.0", "--top_p", "0.95", "--top_k", "20", "--min_p", "0",
         "--extra", json.dumps({"chat_template_kwargs": {"enable_thinking": True}, "presence_penalty": 0.0})]
NINFER = ["--temperature", "0.6", "--top_p", "0.95", "--top_k", "20", "--min_p", "0", "--no_think",
          "--extra", json.dumps({"presence_penalty": 0.0, "frequency_penalty": 0.0})]
JPNT = ["--temperature", "0.7", "--top_p", "0.8", "--top_k", "20", "--min_p", "0", "--no_think", "--extra", json.dumps({"presence_penalty": 0.0})]
RAND = ["--temperature", "0", "--top_p", "1.0", "--top_k", "20", "--min_p", "0", "--no_think",
        "--extra", json.dumps({"presence_penalty": 0.0, "ignore_eos": True})]
NFIX = CMP / "ninfer" / "ninfer_corpus"
NINFER_FX = ["scenario_story_zh_scifi", "scenario_story_en_mystery", "scenario_story_zh_dialogue",
             "scenario_translation_zh_en", "scenario_translation_en_zh", "scenario_translation_markdown"]
NSEEDS = (7632647173703958409, 7968175640111700217)


def jobs(sets):
    """(set, prompt name, seed, source kind, path, args, max tokens)"""
    out = []
    if "v8prose" in sets:
        out += [("v8prose", p, s, "prompt", CMP / "v8" / "prompts" / f"{p}.txt", THINK, 1024)
                for p in ("prose_en_cpu", "prose_ja_essay", "prose_en_story") for s in (1, 2, 3)]
    if "h2h" in sets:
        out += [("h2h", c, s, "prompt", CMP / "ninfer" / "prompts" / f"{c}.txt", THINK, 2048) for c in ("chat", "explain") for s in (1, 2, 3)]
    if "ninfer" in sets:
        man = {c["name"]: c for c in json.load(open(NFIX / "manifest.json", encoding="utf-8"))["cases"]}
        for n in NINFER_FX:
            p = NFIX / f"{n}.json"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == man[n]["messages_sha256"], n
            out += [("ninfer", n, k, "messages", p, NINFER, int(man[n]["max_new"])) for k in (0, 1)]
    if "jp" in sets:
        out += [("jp", f"jp_q{i}", 1, "prompt", SRC / "prompts" / f"jp_q{i}.txt", JPNT, 2048) for i in range(1, 6)]
    if "h2hx" in sets:
        out += [("h2hx", c, s, "prompt", CMP / "ninfer" / "prompts" / f"{c}.txt", THINK, 2048) for c in ("code", "math", "json") for s in (1, 2, 3)]
    if "long" in sets:
        out += [("long", "long", s, "prompt", CMP / "ninfer" / "prompts" / "long.txt", THINK, 2048) for s in (1, 2, 3)]
    if "rand8k" in sets:
        out += [("rand8k", f"rand8k_s{i}", 1, "prompt", SRC / "prompts" / f"rand8k_s{i}.txt", RAND, 1024) for i in range(1, 6)]
    return out


def gpu_now():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [int(x) for x in q.stdout.strip().splitlines()[0].split(",")]


def wait_gpu_alone():
    while True:
        ok = True
        for _ in range(3):
            m, u = gpu_now()
            if m >= 3000 or u >= 15: ok = False; break
            time.sleep(2)
        if ok: return
        print(f"[wf] GPU busy ({m} MiB, {u} %), waiting 60 s", flush=True); time.sleep(60)


def wait_ready(base, proc, timeout, ready_path="/v1/models", ready_text=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None: raise RuntimeError(f"server exited early ({proc.returncode})")
        try:
            r = requests.get(base + ready_path, timeout=3)
            if r.status_code == 200 and (not ready_text or ready_text in r.text): return time.time() - t0
        except requests.RequestException: pass
        time.sleep(3)
    raise RuntimeError("server not ready")


def stop(c, proc):
    if c["stop"]: subprocess.run(c["stop"], capture_output=True)
    else: proc.terminate()
    try: proc.wait(timeout=120)
    except subprocess.TimeoutExpired: proc.kill()
    if c["stop"]:                                               # WSL: make sure no child (vLLM EngineCore, SGLang scheduler) keeps the GPU
        time.sleep(5)
        subprocess.run(WSL + ["bash", "-c", "pkill -9 -f <work_dir>/ ; pkill -9 -f EngineCore ; true"], capture_output=True)
        try: proc.kill()
        except Exception: pass
    for _ in range(60):                                         # wait until the GPU memory is back (WSL processes exit asynchronously)
        if gpu_now()[0] < 3000: break
        time.sleep(2)


def capture(c, tag, cfg, job, outdir):
    st, pname, seed, kind, path, args, maxtok = job
    out = outdir / f"{cfg}_{st}_{pname}_s{seed}.json"
    src = ["--prompt-file", str(path)] if kind == "prompt" else ["--messages-file", str(path)]
    sd = NSEEDS[seed] % 2 ** 31 if st == "ninfer" else seed
    r = subprocess.run([PY, str(CAP), "--base", c["base"], *src, "--out", str(out), "--max_tokens", str(maxtok), "--seed", str(sd),
                        "--label", f"{cfg} {st} {pname} s{seed}", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"[wf] FAILED {cfg} {st} {pname} s{seed}: {r.stdout[-300:]} {r.stderr[-300:]}", flush=True); return
    rec = json.load(open(out, encoding="utf-8"))
    line = dict(tag=tag, engine=c["engine"], cfg=cfg, set=st, prompt=pname, seed=seed, decode_tok_s=rec["decode_tok_s"], ttft_s=rec["ttft_s"],
                t_last_s=rec["t_last_s"], completion_tokens=rec["completion_tokens"], prompt_tokens=(rec.get("usage") or {}).get("prompt_tokens"),
                finish=rec["finish"], t=time.strftime("%H:%M:%S"))
    with open(HERE / "runs.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(line) + "\n")
    print(f"[wf] {tag} {cfg:20s} {st:8s} {pname:28s} s{seed}: {rec['decode_tok_s']:6.1f} tok/s, {rec['completion_tokens']} tokens, "
          f"prompt {line['prompt_tokens']}, {rec['finish']}", flush=True)


THINK_EXTRA = json.dumps({"chat_template_kwargs": {"enable_thinking": True}, "top_p": 0.95, "top_k": 20, "min_p": 0})


def bench12(c, tag, cfg, thinking=False):
    """2026-11-06: NInfer's 12 fixtures x 2 (4,096) and the own tests x 2 (8,192) through the 11/3 clients; results/<tag>_<cfg>/.
    thinking=False: greedy, thinking off (the 11/3 protocol). thinking=True (set n12t): enable_thinking true, the checkpoint's sampling
    T 1.0 / top-p 0.95 / top-k 20 / min-p 0, seed = run index; the same caps, so a long reasoning is cut at the cap (the speed is measured
    over whatever was generated, reasoning included)."""
    out = RESULTS / f"{tag}_{cfg.replace('+', '_')}"; out.mkdir(parents=True, exist_ok=True); lab = cfg.replace("+", "_")
    if thinking:
        n12_args = ["--repeats", "2", "--temperature", "1.0", "--no-thinking-kw", "--extra", THINK_EXTRA, "--seed-per-run"]
        own_args = ["--repeats", "2", "--temperature", "1.0", "--thinking", "on", "--extra", THINK_EXTRA, "--seed-per-run"]
    else:
        n12_args = ["--repeats", "2", "--temperature", "0"]; own_args = ["--repeats", "2"]
    for name, cmd in (("n12", [PY, "-X", "utf8", "-u", str(OAI), "--base", c["base"] + "/v1", "--label", lab, "--engine", cfg, "--out-dir", str(out / "n12"), *n12_args]),
                      ("own", [PY, "-X", "utf8", "-u", str(CUST), "--base", c["base"] + "/v1", "--label", lab, "--out-dir", str(out / "own"), "--cases", str(CASES), *own_args])):
        print(f"[wf] {time.strftime('%H:%M:%S')} {tag} {cfg} {name} start", flush=True)
        with open(out / f"client_{name}.log", "a", encoding="utf-8", errors="replace") as lg:
            r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT)
        print(f"[wf] {time.strftime('%H:%M:%S')} {tag} {cfg} {name} exit {r.returncode}", flush=True)


TASKS = Path(r"<tmp_dir>")
SETUP_DONE = [(TASKS / "<install_job>.output", "ALL DONE"), (TASKS / "<download_job>.output", "ALL DONE")]   # engine installs, model downloads


def gate(cfg):
    """Competitors start only after every install and download has finished, so pip / disk traffic in WSL cannot slow them."""
    return   # 2026-11-01: installs and downloads finished on 2026-10-18 (checked by the WSL probe of 2026-10-31)
    t0 = time.time(); said = False
    while True:
        miss = [str(f.name) for f, mark in SETUP_DONE if not (f.exists() and mark in f.read_text(encoding="utf-8", errors="replace"))]
        if not miss: return
        if not said: print(f"[wf] {cfg}: waiting for installs / downloads ({', '.join(miss)})", flush=True); said = True
        if time.time() - t0 > 5 * 3600: raise RuntimeError("installs / downloads not finished in 5 h")
        time.sleep(60)


def phase(tag, cfg, sets):
    c = CFGS[cfg]; outdir = HERE / "captures" / tag; outdir.mkdir(parents=True, exist_ok=True)
    cmd = [x if x != "LOG" else str(LOGS / f"yasakani_requests_{tag}.jsonl") for x in c["cmd"]]
    gate(cfg)
    wait_gpu_alone()
    slog = open(LOGS / f"server_{tag}_{cfg.replace('+', '_')}.log", "a", encoding="utf-8", errors="replace")
    print(f"[wf] {time.strftime('%H:%M:%S')} start {cfg}", flush=True)
    proc = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT)
    try:
        sec = wait_ready(c["base"], proc, c["ready_timeout"], c.get("ready_path", "/v1/models"), c.get("ready_text", "")); time.sleep(5)
        print(f"[wf] {tag} {cfg} ready in {sec:.0f} s; GPU {gpu_now()}", flush=True)
        for w in range(2):                                          # warm-ups: CUDA graphs, JIT kernels, the drafter
            subprocess.run([PY, str(CAP), "--base", c["base"], "--prompt", "Write two sentences about the sea.", "--out", str(outdir / f"warmup_{cfg}_{w}.json"),
                            "--max_tokens", "128", "--temperature", "0", "--no_think", "--label", "warmup"], capture_output=True)
        if "n12" in sets: bench12(c, tag, cfg)                      # 2026-11-06: the 11/3 head-to-head protocol
        if "n12t" in sets: bench12(c, tag, cfg, thinking=True)      # 2026-11-06: the same with thinking on
        for job in jobs([s for s in sets if s not in ("n12", "n12t")]): capture(c, tag, cfg, job, outdir)
        return True
    finally:
        stop(c, proc)
        print(f"[wf] {time.strftime('%H:%M:%S')} {tag} {cfg} stopped; GPU {gpu_now()}", flush=True)


PLANS = {
    "main": [("wf_a", "yasakani"), ("wf_a", "sg_dflash_nv"), ("wf_a", "sg_dflash_lm4"), ("wf_a", "sg_dspark_lm4"), ("wf_a", "sg_mtp_nv"),
             ("wf_a", "vl_dflash_lm4"), ("wf_a", "vl_mtp_lm4"), ("wf_a", "vl_dspark_lm4"), ("wf_a", "trt_mtp_nv"), ("wf_b", "yasakani")],
    "smoke": [("smoke", "yasakani")],
    # 2026-11-01 same-day re-measure of the other engines (YASAKANI's public build is measured by tools\\a_speed.py); tags x1 / x1l (llama.cpp), x2 / x2l (SGLang, vLLM)
    "x1": [("x1", "llama_mtp2"), ("x1", "llama_dflq4")],
    # 2026-11-06: our protocol on every other engine setting of 11/1 (DFlash 2 settings first), then the public build as a same-day check
    "n12": [("n12", "sg_dflash_nv"), ("n12", "vl_dflash_lm4"), ("n12", "llama_dflq4"), ("n12", "llama_mtp2"), ("n12", "sg_dflash_lm4+norc"),
            ("n12", "sg_mtp_nv"), ("n12", "sg_dspark_lm4"), ("n12", "vl_mtp_lm4"), ("n12", "vl_dspark_lm4"), ("n12", "ysk_std_1103")],
    "n12smoke": [("n12", "llama_dflq4")],
    "n12b": [("n12", "ninfer"), ("n12", "tensorfold")],   # 2026-11-06 job 1003
    "n12c": [("n12", "ninfer_master")],   # 2026-11-06 job 1005
    # 2026-11-06 thinking on (job 1006): the DFlash 2 and MTP settings of the three, NInfer master, TensorFold, then the public build
    "n12t": [("n12t", "sg_dflash_nv"), ("n12t", "sg_mtp_nv"), ("n12t", "vl_dflash_lm4"), ("n12t", "vl_mtp_lm4"), ("n12t", "llama_dflq4"), ("n12t", "llama_mtp2"),
             ("n12t", "ninfer_master"), ("n12t", "tensorfold"), ("n12t", "ysk_std_1103")],
    "offdraft": [("n12", "ysk_offdraft")],   # 2026-11-06 job 1011 (sets n12,n12t on one server start)
    "n12t_ysk": [("n12t", "ysk_std_1103")],   # 2026-11-06 job 1010: job 1006 ran it without PYTHONPATH (No module named yasakani)
    "fast": [("n12", "ysk_fast_1103")],       # 2026-11-07 job 1066: the fast bundle, thinking off (run with --sets n12)
    "fastt": [("n12t", "ysk_fast_1103")],     # 2026-11-07 job 1066: the fast bundle, thinking on (run with --sets n12t)
    "fastv8": [("n12", "ysk_fast_v8")],       # 2026-11-07 job 1071: the fast bundle with the v8 DLL, thinking off
    "fastv8t": [("n12t", "ysk_fast_v8")],     # 2026-11-07 job 1071: the fast bundle with the v8 DLL, thinking on
    "stdv8": [("n12", "ysk_std_v8")],         # 2026-11-07 job 1072: the standard bundle with the v8 DLL, thinking off
    "stdv8t": [("n12t", "ysk_std_v8")],       # 2026-11-07 job 1072: the standard bundle with the v8 DLL, thinking on
    "qualv8": [("n12", "ysk_quality_v8")],    # 2026-11-08 job 1079: the new quality-first bundle with the v8 DLL, thinking off
    "qualv8t": [("n12t", "ysk_quality_v8")],  # 2026-11-08 job 1079: the same, thinking on
    "speedv8": [("n12", "ysk_speed_v8")],     # 2026-11-08 job 1079: the new speed-first bundle with the v8 DLL, thinking off
    "speedv8t": [("n12t", "ysk_speed_v8")],   # 2026-11-08 job 1079: the same, thinking on
    "x1long": [("x1l", "llama_mtp2+48k"), ("x1l", "llama_dflq4+48k")],
    "x2": [("x2", "sg_dflash_nv"), ("x2", "sg_dflash_lm4+norc"), ("x2", "sg_mtp_nv"), ("x2", "sg_dspark_lm4"),
           ("x2", "vl_dflash_lm4"), ("x2", "vl_mtp_lm4"), ("x2", "vl_dspark_lm4")],
    "x2long": [("x2l", "sg_dflash_lm4+48k"), ("x2l", "sg_dflash_nv+48k"), ("x2l", "sg_mtp_nv+48k"), ("x2l", "sg_dspark_lm4+48k"),
               ("x2l", "vl_dflash_lm4+48k+nopc"), ("x2l", "vl_mtp_lm4+48k"), ("x2l", "vl_dspark_lm4+48k+nopc")],
    # 2026-10-18 SGLang 0.5.22 needs --disable-radix-cache for one-request DFlash (mamba slots per request 5 -> 1): rerun both DFlash settings
    "llama": [("wf_a", "llama_mtp2"), ("wf_a", "llama_dflq4")],
    "sgfix": [("wf_a", "sg_dflash_nv"), ("wf_a", "sg_dflash_lm4+norc"), ("wf_a", "trt_mtp_nv_fi")],
    # 2026-10-18 restart after the FlashInfer / nvcc fix: everything after our first block
    "rest": [("wf_a", "vl_dflash_lm4"), ("wf_a", "vl_mtp_lm4"), ("wf_a", "vl_dspark_lm4"), ("wf_a", "sg_dflash_nv"), ("wf_a", "sg_dflash_lm4"),
             ("wf_a", "sg_dspark_lm4"), ("wf_a", "sg_mtp_nv"), ("wf_a", "trt_mtp_nv"), ("wf_b", "yasakani")],
    # 2026-10-18 run order: ours first, then vLLM (installed first), SGLang, TensorRT-LLM, ours again
    # 2026-10-18 evening: code / math / JSON on every engine setting; ours first and last
    "allx": [("wf_c", "yasakani"), ("wf_c", "llama_mtp2"), ("wf_c", "llama_dflq4"), ("wf_c", "sg_dflash_lm4+norc"), ("wf_c", "sg_dflash_nv"),
             ("wf_c", "sg_mtp_nv"), ("wf_c", "sg_dspark_lm4"), ("wf_c", "vl_dflash_lm4"), ("wf_c", "vl_mtp_lm4"), ("wf_c", "vl_dspark_lm4"),
             ("wf_d", "yasakani")],
    # then the 41K-token long input with 48K contexts
    "long": [("wf_e", "yasakani+48k"), ("wf_e", "llama_mtp2+48k"), ("wf_e", "llama_dflq4+48k"), ("wf_e", "sg_dflash_lm4+48k"),
             ("wf_e", "sg_dflash_nv+48k"), ("wf_e", "sg_mtp_nv+48k"), ("wf_e", "sg_dspark_lm4+48k"), ("wf_e", "vl_dflash_lm4+48k"),
             ("wf_e", "vl_mtp_lm4+48k"), ("wf_e", "vl_dspark_lm4+48k"), ("wf_f", "yasakani+48k")],
    "longfix": [("wf_e", "vl_dflash_lm4+48k+nopc"), ("wf_e", "vl_dspark_lm4+48k+nopc")],   # vl_dspark died the same way on its 3rd run
    "main2": [("wf_a", "yasakani"), ("wf_a", "vl_dflash_lm4"), ("wf_a", "vl_mtp_lm4"), ("wf_a", "vl_dspark_lm4"), ("wf_a", "sg_dflash_nv"),
              ("wf_a", "sg_dflash_lm4"), ("wf_a", "sg_dspark_lm4"), ("wf_a", "sg_mtp_nv"), ("wf_a", "trt_mtp_nv"), ("wf_b", "yasakani")],
}


def main():
    a = sys.argv[1:]; arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    plan = arg("--plan", "main"); only = arg("--only"); sets = arg("--sets", "v8prose,h2h,ninfer,jp,rand8k").split(",")
    steps = PLANS[plan]
    dl = float(arg("--deadline-min", "0")); t_end = time.time() + dl * 60 if dl else None
    if only: steps = [s for s in steps if s[1] in only.split(",")]
    with gpu_lock(f"wf {plan}"):
        for tag, cfg in steps:
            if t_end and time.time() > t_end:
                print(f"[wf] deadline reached: skipping {tag} {cfg}", flush=True); continue
            print(f"[wf] {time.strftime('%H:%M:%S')} begin {tag} {cfg}", flush=True)
            try:
                phase(tag, cfg, sets)
            except Exception as ex:
                print(f"[wf] ERROR {tag} {cfg}: {ex}", flush=True)
                if cfg in FALLBACK:
                    print(f"[wf] retry with the recipe's fallback {FALLBACK[cfg]}", flush=True)
                    try: phase(tag, FALLBACK[cfg], sets)
                    except Exception as ex2: print(f"[wf] ERROR {tag} {FALLBACK[cfg]}: {ex2}", flush=True)
            print(f"[wf] {time.strftime('%H:%M:%S')} end {tag} {cfg}", flush=True)
    print("[wf] done", flush=True)


if __name__ == "__main__":
    main()
