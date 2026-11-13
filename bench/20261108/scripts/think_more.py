# -*- coding: utf-8 -*-
"""think_more.py (2026-11-08) -- 3 more thinking-on runs (seeds 2, 3, 4) of the fastest model and the quality model with the yskn.dll of
November 7, with the same clients and settings as run_wf_n12.py (bench12, thinking=True): NInfer's 12 fixtures (4,096 tokens) and the own
tests (8,192), T 1.0 / top-p 0.95 / top-k 20 / min-p 0, enable_thinking true, seed = run index.  The runs with seeds 0 and 1 are the ones
made by run_wf_n12.py; the new ones are written to results/n12t_ysk_<cfg>/{n12,own} next to this script (same layout).  The clients are
copies with --seed-offset only.  As in run_wf_n12.py phase(): wait until the GPU is free, start the server, wait for /health, 5 s, two
short warm-ups, then the two clients.
usage: python -X utf8 -u think_more.py"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DLL = Path("<yasakani_dir>") / "bin" / "yskn.dll"; PYP = Path("<yasakani_py>")
CASES = HERE.parents[1] / "20261103" / "prompts" / "cases.json"
OAI = HERE / "oai_bench_client_seedoff.py"; CUST = HERE / "custom_bench_seedoff.py"; CAP = HERE.parents[1] / "20261101" / "scripts" / "capture_stream.py"   # CAP: the same warm-ups as run_wf_n12.py
BUNDLES = [("speed_v8", Path("<model_dir_fastest>")), ("quality_v8", Path("<model_dir_quality>"))]
THINK_EXTRA = json.dumps({"chat_template_kwargs": {"enable_thinking": True}, "top_p": 0.95, "top_k": 20, "min_p": 0})   # = run_wf_n12.py THINK_EXTRA
BASE = "http://127.0.0.1:8099"; OFFSET, REPEATS = 2, 3


def say(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(line, flush=True)
    with open(HERE / "think_more.log", "a", encoding="utf-8", errors="replace") as f: f.write(line + "\n")


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r: return '"ok"' in r.read().decode("utf-8")
    except Exception: return False                                                    # noqa: BLE001


def kill_tree(p):
    try:
        import psutil
        for c in psutil.Process(p.pid).children(recursive=True): c.kill()
    except Exception: pass                                                             # noqa: BLE001
    p.kill(); p.wait(timeout=60)


def gpu_now():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        m, u = q.strip().splitlines()[0].split(","); return int(m), int(u)
    except Exception: return 99999, 100                                              # noqa: BLE001


def wait_gpu_alone(limit_s=1800):
    """= run_wf_n12.py wait_gpu_alone (memory < 3,000 MiB and utilisation < 15 % three times in a row); gives up after limit_s"""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        ok = True
        for _ in range(3):
            m, u = gpu_now()
            if m >= 3000 or u >= 15: ok = False; break
            time.sleep(2)
        if ok: return True
        say(f"[more] GPU busy ({m} MiB, {u} %), waiting 60 s"); time.sleep(60)
    return False


def main():
    for cfg, model in BUNDLES:
        out = HERE / "results" / f"n12t_ysk_{cfg}"; out.mkdir(parents=True, exist_ok=True)
        if not wait_gpu_alone():
            say(f"[more] {cfg}: GPU not free - skipped"); continue
        env = dict(os.environ, PYTHONPATH=str(PYP)); log = open(out / "server.log", "w", encoding="utf-8", errors="replace")
        p = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", str(model), "--dll", str(DLL), "--host", "127.0.0.1",
                              "--port", "8099", "--max-tokens-cap", "16384"], stdout=log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 900 and not healthy(): time.sleep(2)
        if not healthy():
            say(f"[more] {cfg}: server NOT ready"); kill_tree(p); continue
        say(f"[more] {cfg}: server ready in {time.time() - t0:.0f} s; GPU {gpu_now()}"); time.sleep(5)
        for w in range(2):                                                            # the same two warm-ups as run_wf_n12.py phase()
            subprocess.run([sys.executable, str(CAP), "--base", BASE, "--prompt", "Write two sentences about the sea.", "--out", str(out / f"warmup_{w}.json"),
                            "--max_tokens", "128", "--temperature", "0", "--no_think", "--label", "warmup"], capture_output=True)
        lab = f"ysk_{cfg}"
        for name, cmd in (("n12", [sys.executable, "-X", "utf8", "-u", str(OAI), "--base", BASE + "/v1", "--label", lab, "--engine", lab, "--out-dir", str(out / "n12"),
                                   "--repeats", str(REPEATS), "--seed-offset", str(OFFSET), "--temperature", "1.0", "--no-thinking-kw", "--extra", THINK_EXTRA, "--seed-per-run"]),
                          ("own", [sys.executable, "-X", "utf8", "-u", str(CUST), "--base", BASE + "/v1", "--label", lab, "--out-dir", str(out / "own"), "--cases", str(CASES),
                                   "--repeats", str(REPEATS), "--seed-offset", str(OFFSET), "--temperature", "1.0", "--thinking", "on", "--extra", THINK_EXTRA, "--seed-per-run"])):
            t1 = time.time()
            with open(out / f"client_{name}.log", "a", encoding="utf-8", errors="replace") as lg:
                r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT)
            say(f"[more] {cfg} {name}: exit {r.returncode} in {time.time() - t1:.0f} s")
        kill_tree(p); log.close(); time.sleep(5)
    say("[more] done")


if __name__ == "__main__":
    main()
