#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A: answer-writing speed of the public build (public yskn.dll + public Python server), measured the way measure_public did on 2026-10-18.

  run     start the public server (--serve) or use a running one (--base), send the 10 tests one request at a time and write one JSON
          file with every run (used or not).  The JSON is saved after each request, so an interrupted run keeps what it measured.
  summary read one or more JSON files (or --from-csv an old results_runs.csv); per test take the median of each round, report the slower
          round, and compare with the 2026-10-18 numbers of the other engines (results_summary.csv).  Prints a table; --csv/--md write files.

speed (tokens/s) = (completion_tokens - 1) / (time of the last streamed chunk - time of the first streamed chunk)   [as in stream_speed.py]
used = no error and at least 100 completion tokens (the rule of the 2026-10-18 measurement).

  python a_speed.py run --serve --out A.json                 # server + 2 rounds of 10 tests
  python a_speed.py run --base http://127.0.0.1:8765 --out A.json --tests chat,code --rounds 1
  python a_speed.py summary A.json [B.json ...] [--csv runs.csv] [--md table.md]
  python a_speed.py summary --from-csv results_runs.csv --engine YASAKANI      # re-derive the 2026-10-18 table (a check of this script)

Exit codes: 0 ok, 2 the server did not start, 3 the GPU is not idle (nothing was started), 4 no used run at all.
"""
import argparse
import csv
import datetime as dt
import hashlib
import http.client
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

USERPATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\"' ]+")

THINK = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)
SIX = ("chat", "explain", "code", "math", "json", "long")

# name, prompt files (or ninfer cases), runs per prompt, max_tokens, thinking, sampling, extra body fields, prompt label per file
TESTS = [
    dict(name="chat", files=["chat.txt"], labels=["chat"], runs=3, max_tokens=2048, think=True, samp=THINK),
    dict(name="explain", files=["explain.txt"], labels=["explain"], runs=3, max_tokens=2048, think=True, samp=THINK),
    dict(name="code", files=["code.txt"], labels=["code"], runs=3, max_tokens=2048, think=True, samp=THINK),
    dict(name="math", files=["math.txt"], labels=["math"], runs=3, max_tokens=2048, think=True, samp=THINK),
    dict(name="json", files=["json.txt"], labels=["json"], runs=3, max_tokens=2048, think=True, samp=THINK),
    dict(name="long", files=["long.txt"], labels=["long"], runs=3, max_tokens=2048, think=True, samp=THINK),
    dict(name="prose3", files=["prose3_en_cpu.txt", "prose3_ja_essay.txt", "prose3_en_story.txt"],
         labels=["prose_en_cpu", "prose_ja_essay", "prose_en_story"], runs=3, max_tokens=1024, think=True, samp=THINK),
    dict(name="japanese", files=["japanese_q%d.txt" % i for i in range(1, 6)], labels=["jp_q%d" % i for i in range(1, 6)], runs=1,
         max_tokens=2048, think=False, samp=dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)),
    dict(name="ninfer_prose", cases=["story_zh_scifi", "story_en_mystery", "story_zh_dialogue", "translation_zh_en",
                                     "translation_en_zh", "translation_markdown"], runs=2, max_tokens=4096, think=False,
         samp=dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)),
    dict(name="sglang_random_8k", files=["sglang_random_8k_%d.txt" % i for i in range(1, 6)], labels=["rand8k_s%d" % i for i in range(1, 6)],
         runs=1, max_tokens=1024, think=False, samp=dict(temperature=0.0, top_p=1.0, top_k=20, min_p=0.0), extra={"ignore_eos": True}),
]
# prompt tokens of the 2026-10-18 runs (to notice a different chat template or tokenizer); sglang_random_8k differs by a few tokens
REF_PROMPT_TOKENS = {"chat": 112, "explain": 90, "code": 103, "math": 87, "json": 110, "long": 41261, "prose_en_cpu": 86,
                     "prose_ja_essay": 83, "prose_en_story": 86, "jp_q1": 30, "jp_q2": 35, "jp_q3": 36, "jp_q4": 35, "jp_q5": 39,
                     "story_zh_scifi": 127, "story_en_mystery": 131, "story_zh_dialogue": 135, "translation_zh_en": 401,
                     "translation_en_zh": 395, "translation_markdown": 360, "rand8k_s1": 8206, "rand8k_s2": 8206, "rand8k_s3": 8204,
                     "rand8k_s4": 8208, "rand8k_s5": 8207}


def mask(s):
    return USERPATH.sub("~", str(s))


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(*a):
    print("[%s]" % now(), *a, flush=True)


def default_dirs():
    k = Path("<work_dir>")
    return (k / "prompts",
            k / "ninfer_corpus",
            k / "results_summary.csv")


# ---------------------------------------------------------------------------------------------------------------------------------
# HTTP (standard library only: no extra package is needed on the measuring computer)
# ---------------------------------------------------------------------------------------------------------------------------------
def _conn(base, timeout):
    u = urllib.parse.urlparse(base)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def http_get(base, path, timeout=5):
    """(status, parsed JSON or text)."""
    c = _conn(base, timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        txt = r.read().decode("utf-8", "replace")
        try:
            return r.status, json.loads(txt)
        except ValueError:
            return r.status, txt
    finally:
        c.close()


def one_request(base, body, timeout=1800):
    """One streamed chat request.  Times are taken when a line arrives (the same rule as stream_speed.py of 2026-10-18)."""
    t0 = time.perf_counter()
    times, usage, finish, err = [], None, None, None
    c = None
    try:
        c = _conn(base, timeout)
        c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            txt = r.read(400).decode("utf-8", "replace").replace("\n", " ")
            return dict(error="HTTP %d %s" % (r.status, txt[:200]), completion_tokens=0, prompt_tokens=0, finish=None, speed=0.0,
                        ttft=0.0, wall=time.perf_counter() - t0, loop_guard=None)
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            d = line[6:]
            if d == b"[DONE]":
                break
            t = time.perf_counter() - t0
            try:
                j = json.loads(d)
            except ValueError:
                continue
            if j.get("error"):
                err = "stream error: %s" % json.dumps(j["error"], ensure_ascii=False)[:200]
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                dl = ch.get("delta") or {}
                if any(dl.get(k) for k in ("reasoning_content", "reasoning", "content")):
                    times.append(t)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except Exception as ex:                                      # noqa: BLE001
        err = "%s: %s" % (type(ex).__name__, str(ex)[:200])
    finally:
        if c is not None:
            c.close()
    n = (usage or {}).get("completion_tokens") or len(times)
    speed = (n - 1) / max(1e-6, times[-1] - times[0]) if n > 1 and len(times) > 1 else 0.0
    return dict(error=err, completion_tokens=int(n), prompt_tokens=int((usage or {}).get("prompt_tokens") or 0), finish=finish,
                speed=speed, ttft=times[0] if times else 0.0, wall=(times[-1] if times else time.perf_counter() - t0),
                loop_guard=(usage or {}).get("loop_guard"))


def build_body(spec, messages, args):
    s = spec["samp"]
    body = dict(model=args.model, messages=messages, max_tokens=args.max_tokens_cap or spec["max_tokens"], stream=True,
                stream_options={"include_usage": True}, temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                min_p=s["min_p"], presence_penalty=0.0, chat_template_kwargs={"enable_thinking": spec["think"]})
    body.update(spec.get("extra") or {})
    if args.loop_guard == "off":
        body["loop_guard"] = False
    return body


def gpu_line():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return out[0].strip() if out else ""
    except Exception:                                             # noqa: BLE001
        return ""


def gpu_busy(samples=5, gap=1.0):
    """(median utilisation %, memory used MiB) from a few nvidia-smi samples; (None, None) when nvidia-smi is not available."""
    u, m = [], []
    for i in range(samples):
        s = gpu_line().split(",")
        try:
            m.append(int(float(s[4])))
            u.append(int(float(s[6])))
        except (IndexError, ValueError):
            return None, None
        if i + 1 < samples:
            time.sleep(gap)
    return statistics.median(u), max(m)


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b or (limit is not None and n >= limit):
                break
            h.update(b)
            n += len(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------------------------
def load_messages(spec, prompts_dir, ninfer_dir):
    """[(label, messages)] for one test."""
    out = []
    if "cases" in spec:
        for c in spec["cases"]:
            msgs = json.loads((ninfer_dir / ("scenario_%s.json" % c)).read_text(encoding="utf-8"))
            out.append((c, msgs))
    else:
        for f, lab in zip(spec["files"], spec["labels"]):
            out.append((lab, [dict(role="user", content=(prompts_dir / f).read_text(encoding="utf-8"))]))
    return out


def prompt_fingerprints(loaded, prompts_dir, ninfer_dir):
    """sha256 of every prompt file used.  The six ninfer cases are compared with prompts/ninfer_cases.json (made on 2026-10-18)."""
    fps, warn = {}, []
    ref = {}
    cj = prompts_dir / "ninfer_cases.json"
    if cj.is_file():
        for e in json.loads(cj.read_text(encoding="utf-8")):
            ref[e["name"]] = e["messages_sha256"]
    for spec in TESTS:
        if spec["name"] not in loaded:
            continue
        if "cases" in spec:
            for c in spec["cases"]:
                f = ninfer_dir / ("scenario_%s.json" % c)
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                fps[c] = h
                r = ref.get("scenario_" + c)
                if r and r != h:
                    warn.append("ninfer case %s differs from the 2026-10-18 fingerprint" % c)
        else:
            for f, lab in zip(spec["files"], spec["labels"]):
                fps[lab] = hashlib.sha256((prompts_dir / f).read_bytes()).hexdigest()
    return fps, warn


def save(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def start_server(args, logpath):
    base = "http://127.0.0.1:%d" % args.port
    try:
        http_get(base, "/health", timeout=3)
        log("a server already answers on", base, "- not starting another one")
        return None, base, 2
    except Exception:                                             # noqa: BLE001
        pass
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    if args.pypath:
        env["PYTHONPATH"] = args.pypath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", args.model_dir, "--dll", args.dll, "--port", str(args.port)]
    if args.max_ctx:
        cmd += ["--max-ctx", str(args.max_ctx)]
    cmd += args.serve_extra
    log("start server:", mask(" ".join(cmd)))
    lf = open(logpath, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < args.ready_timeout:
        if proc.poll() is not None:
            log("the server stopped by itself, exit code", proc.returncode)
            return None, base, 2
        try:
            code, txt = http_get(base, "/health", timeout=5)
            if code == 200:
                log("server ready after %.0f s: %s" % (time.time() - t0, json.dumps(txt)[:200]))
                return proc, base, 0
        except Exception:                                         # noqa: BLE001
            pass
        time.sleep(2)
    log("the server did not become ready in %d s" % args.ready_timeout)
    stop_server(proc)
    return None, base, 2


def stop_server(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(60)
    except Exception:                                             # noqa: BLE001
        try:
            proc.kill()
            proc.wait(30)
        except Exception:                                         # noqa: BLE001
            pass


def cmd_run(args):
    prompts_dir, ninfer_dir, _ = default_dirs()
    prompts_dir = Path(args.prompts) if args.prompts else prompts_dir
    ninfer_dir = Path(args.ninfer_dir) if args.ninfer_dir else ninfer_dir
    wanted = [t for t in args.tests.split(",") if t] if args.tests else [t["name"] for t in TESTS]
    unknown = [t for t in wanted if t not in [x["name"] for x in TESTS]]
    if unknown:
        print("unknown test:", ",".join(unknown), file=sys.stderr)
        return 1
    specs = [t for t in TESTS if t["name"] in wanted]
    loaded = {s["name"]: load_messages(s, prompts_dir, ninfer_dir) for s in specs}      # fail early on a missing file
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fps, fp_warn = prompt_fingerprints(loaded, prompts_dir, ninfer_dir)
    for w in fp_warn:
        log("WARNING:", w)
    if args.serve:
        util, used = gpu_busy()
        log("GPU before start:", gpu_line(), "| median utilisation %s %%, memory used %s MiB" % (util, used))
        if used is not None and (used > args.max_gpu_used_mib or util > args.max_gpu_util):
            log("the GPU is not idle (utilisation %s %% limit %d, %s MiB in use limit %d): nothing started" % (
                util, args.max_gpu_util, used, args.max_gpu_used_mib))
            return 3
        proc, base, rc = start_server(args, out.with_suffix(".serve.log"))
        if rc:
            return rc
    else:
        proc, base = None, args.base.rstrip("/")
    data = dict(kind="a_speed", session=now(), tag=args.tag, base=base, serve=bool(args.serve), loop_guard=args.loop_guard,
                model_dir=mask(args.model_dir or ""), dll=mask(args.dll or ""), gpu_start=gpu_line(), rounds=[], runs=[],
                prompt_sha256=fps, prompt_warnings=fp_warn,
                system=dict(python=platform.python_version(), platform=platform.platform(), script_sha256=sha256_file(__file__)[:16]))
    if args.dll and Path(args.dll).is_file():
        data["system"]["dll_sha256"] = sha256_file(args.dll)
        data["system"]["dll_bytes"] = Path(args.dll).stat().st_size
    if args.model_dir and Path(args.model_dir).is_dir():
        data["system"]["model_files"] = {f.name: f.stat().st_size for f in sorted(Path(args.model_dir).iterdir()) if f.is_file() and f.suffix in (".yskn", ".bin", ".shead")}
    try:
        try:
            data["health"] = http_get(base, "/health", timeout=10)[1]
        except Exception as ex:                                   # noqa: BLE001
            data["health"] = "unavailable: %s" % type(ex).__name__
        warm = loaded.get("chat") or load_messages(TESTS[0], prompts_dir, ninfer_dir)
        for i in range(args.warmups):
            body = build_body(TESTS[0], warm[0][1], args)
            body["max_tokens"] = min(body["max_tokens"], 256)
            r = one_request(base, body)
            log("warm-up %d: %d tokens, %.1f tok/s, error=%s" % (i + 1, r["completion_tokens"], r["speed"], r["error"]))
        for rd in range(1, args.rounds + 1):
            if rd > 1 and args.round_gap > 0:
                log("pause %d s before round %d" % (args.round_gap, rd))
                time.sleep(args.round_gap)
            data["rounds"].append(dict(round=rd, gpu_before=gpu_line(), started=now()))
            for spec in specs:
                for lab, msgs in loaded[spec["name"]]:
                    for run in range(1, (args.runs_override or spec["runs"]) + 1):
                        r = one_request(base, build_body(spec, msgs, args))
                        used = r["error"] is None and r["completion_tokens"] >= 100 and r["speed"] > 0
                        note = []
                        if r["error"]:
                            note.append(r["error"])
                        if r["loop_guard"]:
                            note.append("loop_guard=%s" % json.dumps(r["loop_guard"], ensure_ascii=False))
                        ref = REF_PROMPT_TOKENS.get(lab)
                        if ref and abs(r["prompt_tokens"] - ref) > (10 if lab.startswith("rand8k") else 0) and r["prompt_tokens"]:
                            note.append("prompt_tokens %d (2026-10-18: %d)" % (r["prompt_tokens"], ref))
                        data["runs"].append(dict(test=spec["name"], round=rd, prompt=lab, run=run, decode_tok_s=round(r["speed"], 1),
                                                 completion_tokens=r["completion_tokens"], prompt_tokens=r["prompt_tokens"],
                                                 finish=r["finish"], used=bool(used), ttft_s=round(r["ttft"], 2), wall_s=round(r["wall"], 2),
                                                 note="; ".join(note)))
                        log("round %d %-16s %-20s run %d: %7.1f tok/s  (%d tokens, %s, prompt %d%s)" % (
                            rd, spec["name"], lab, run, r["speed"], r["completion_tokens"], r["finish"], r["prompt_tokens"],
                            "" if used else "  NOT USED " + "; ".join(note)))
                        save(out, data)
            data["rounds"][-1].update(gpu_after=gpu_line(), finished=now())
            save(out, data)
    finally:
        data["gpu_end"] = gpu_line()
        save(out, data)
        stop_server(proc)
        if proc is not None:
            time.sleep(3)
            log("GPU after stop:", gpu_line())
    n_used = sum(1 for r in data["runs"] if r["used"])
    log("done: %d runs, %d used -> %s" % (len(data["runs"]), n_used, out))
    return 0 if n_used else 4


# ---------------------------------------------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------------------------------------------
def load_runs(paths, from_csv, engine):
    runs = []
    if from_csv:
        for r in csv.DictReader(open(from_csv, encoding="utf-8")):
            if r["engine"] == engine:
                runs.append(dict(test=r["test"], session="csv", round=int(r["round"]), prompt=r["prompt"], run=int(r["run"]),
                                 decode_tok_s=float(r["decode_tok_s"]), completion_tokens=int(r["completion_tokens"]),
                                 prompt_tokens=int(r["prompt_tokens"]), finish=r["finish"], used=r["used"] == "yes", note=r["note"]))
    for p in paths:
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in j["runs"]:
            r = dict(r)
            r["session"] = j.get("session", p)
            runs.append(r)
    return runs


def round_medians(runs):
    by = {}
    for r in runs:
        if r["used"]:
            by.setdefault((r["test"], r["session"], r["round"]), []).append(r["decode_tok_s"])
    out = {}
    for (t, s, rd), v in by.items():
        out.setdefault(t, {})["%s r%s" % (s, rd)] = (round(statistics.median(v), 1), len(v))
    return out


def reported(meds):
    """The slower round of each test (the rule of the 2026-10-18 table)."""
    return {t: min(v[0] for v in d.values()) for t, d in meds.items()}


def others_from_csv(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    engines = ("SGLang", "vLLM", "llama.cpp")
    out = {}
    for r in rows:
        out[r["test"]] = dict(old=float(r["yasakani"]), six=r["six_task"] == "yes", fastest=max(float(r[e]) for e in engines),
                              llama=float(r["llama.cpp"]), others={e: float(r[e]) for e in engines})
    return out


def cmd_summary(args):
    _, _, sum_csv = default_dirs()
    sum_csv = Path(args.summary_csv) if args.summary_csv else sum_csv
    runs = load_runs(args.files, args.from_csv, args.engine)
    if not runs:
        print("no runs", file=sys.stderr)
        return 1
    meds = round_medians(runs)
    rep = reported(meds)
    oth = others_from_csv(sum_csv) if sum_csv.is_file() else {}
    order = [t["name"] for t in TESTS if t["name"] in rep]
    lines = []
    lines.append("| test | rounds (median of used runs) | reported (slower round) | 2026-10-18 YASAKANI | change | fastest other (2026-10-18) | ratio | vs llama.cpp |")
    lines.append("|---|---|---|---|---|---|---|---|")
    ratios, llratios = [], []
    for t in order:
        o = oth.get(t, {})
        old = o.get("old")
        ch = "%+.1f%%" % ((rep[t] / old - 1) * 100) if old else ""
        ratio = rep[t] / o["fastest"] if o else None
        llr = rep[t] / o["llama"] if o else None
        if o:
            ratios.append(ratio)
            if o["six"]:
                llratios.append(llr)
        rounds = ", ".join("%s: %.1f (n=%d)" % (k.split(" ")[-1], v[0], v[1]) for k, v in sorted(meds[t].items()))
        lines.append("| %s | %s | **%.1f** | %s | %s | %s | %s | %s |" % (
            t, rounds, rep[t], "%.1f" % old if old else "", ch, "%.1f" % o["fastest"] if o else "",
            "%.2fx" % ratio if ratio else "", "%.2fx" % llr if llr else ""))
    six = [rep[t] for t in SIX if t in rep]
    txt = "\n".join(lines)
    txt += "\n\n"
    if len(six) == len(SIX) and oth:
        txt += "six thinking tests: median YASAKANI %.1f" % statistics.median(six)
        for e in ("SGLang", "vLLM", "llama.cpp"):
            txt += "; %s %.1f" % (e, statistics.median([oth[t]["others"][e] for t in SIX]))
        txt += "\n"
        if llratios:
            txt += "six thinking tests vs llama.cpp: %.2fx - %.2fx; median of the six ratios %.2fx; ratio of the two medians %.2fx\n" % (
                min(llratios), max(llratios), statistics.median(llratios),
                statistics.median(six) / statistics.median([oth[t]["others"]["llama.cpp"] for t in SIX]))
        if "code" in rep:
            txt += "code, 2,048 tokens: YASAKANI %.2f s, llama.cpp %.2f s (2048 / speed)\n" % (2048 / rep["code"], 2048 / oth["code"]["llama"])
    if ratios:
        txt += "all tests: ratio against the fastest other %.2fx - %.2fx; fastest in %d of %d\n" % (
            min(ratios), max(ratios), sum(1 for r in ratios if r > 1.0), len(ratios))
    notes = sorted({r["note"] for r in runs if r.get("note")})
    if notes:
        txt += "\nnotes in the runs:\n" + "\n".join("- " + n[:200] for n in notes[:12]) + "\n"
    unused = [r for r in runs if not r["used"]]
    txt += "\nruns: %d, not used: %d\n" % (len(runs), len(unused))
    print(txt)
    if args.md:
        Path(args.md).write_text(txt, encoding="utf-8")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test", "six_task", "engine", "setting", "variant", "target_weights", "draft_weights", "round", "prompt", "run",
                        "decode_tok_s", "completion_tokens", "prompt_tokens", "finish", "used", "note"])
            for r in runs:
                w.writerow([r["test"], "yes" if r["test"] in SIX else "no", "YASAKANI", "standard", "-", "YASAKANI model (qwen38_27b_e3x.yskn)", "-",
                            r["round"], r["prompt"], r["run"], r["decode_tok_s"], r["completion_tokens"], r["prompt_tokens"], r.get("finish"),
                            "yes" if r["used"] else "no", r.get("note", "")])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--serve", action="store_true", help="start the public server (python -m yasakani serve) and stop it at the end")
    r.add_argument("--base", default="http://127.0.0.1:8765", help="a running server (without --serve)")
    r.add_argument("--model-dir")
    r.add_argument("--dll")
    r.add_argument("--pypath", help="folder with the yasakani package (PYTHONPATH of the server)")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--max-ctx", type=int, default=0)
    r.add_argument("--serve-extra", nargs="*", default=[])
    r.add_argument("--ready-timeout", type=int, default=900)
    r.add_argument("--max-gpu-used-mib", type=int, default=6500, help="refuse to start when more GPU memory than this is in use")
    r.add_argument("--max-gpu-util", type=int, default=25, help="refuse to start when the median GPU utilisation (5 samples) is above this percent")
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--round-gap", type=int, default=60)
    r.add_argument("--warmups", type=int, default=2)
    r.add_argument("--tests", help="comma list of test names (default: all 10)")
    r.add_argument("--runs-override", type=int, default=0, help="runs per prompt (default: the 2026-10-18 numbers)")
    r.add_argument("--max-tokens-cap", type=int, default=0, help="override max_tokens (dry runs only)")
    r.add_argument("--loop-guard", choices=("on", "off"), default="off", help="off: send loop_guard=false (the other engines have no such guard)")
    r.add_argument("--model", default="qwen3.8-27b")
    r.add_argument("--tag", default="pub")
    r.add_argument("--prompts")
    r.add_argument("--ninfer-dir")
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="*")
    s.add_argument("--from-csv")
    s.add_argument("--engine", default="YASAKANI")
    s.add_argument("--summary-csv")
    s.add_argument("--csv")
    s.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "run":
        if a.serve and not (a.model_dir and a.dll):
            ap.error("--serve needs --model-dir and --dll")
        return cmd_run(a)
    return cmd_summary(a)


if __name__ == "__main__":
    sys.exit(main())
