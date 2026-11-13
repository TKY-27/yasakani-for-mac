# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Command line:  ``python -m yasakani  info | serve | chat | bench | verify``  (``yasakani`` when installed with pip)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from ._native import YasakaniError, find_library
from .runtime import ModelPack

DEFAULT_PORT = 8765


# ---- small helpers ----------------------------------------------------------------------------------------------------------------
def _say(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0
    return "%.1f GiB" % n


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and (os.name != "nt" or bool(os.environ.get("WT_SESSION") or os.environ.get("TERM")))


def _fix_console() -> None:
    """Never fail on a character the console cannot show (older Windows consoles)."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _model_dir(args) -> Path:
    d = getattr(args, "model_dir", None) or os.environ.get("YASAKANI_MODEL_DIR")
    if not d:
        raise YasakaniError("give the model folder with --model-dir (or set YASAKANI_MODEL_DIR)")
    return Path(d)


def _gpu_lines() -> List[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ["nvidia-smi not found (is the NVIDIA driver installed?)"]
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as ex:
        return ["nvidia-smi failed: " + type(ex).__name__]
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return lines or ["nvidia-smi reported no GPU"]


def _gpu_memory_warning(device: int = 0) -> Optional[str]:
    """After loading: is the GPU's memory (almost) full?  When the model does not fit, Windows keeps part of it in ordinary memory
    and generation becomes several times slower, without any error.  None when all is well or when it cannot be told."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        rows = [[float(x) for x in ln.split(",")] for ln in r.stdout.splitlines() if ln.strip()]
        total, used = rows[0] if len(rows) == 1 else rows[device]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    if total > 0 and used >= 0.95 * total:
        return ("warning: the GPU memory is almost full (%d of %d MiB in use).  When the model does not fit, Windows keeps part of it "
                "in ordinary memory and generation becomes several times slower, without an error.  Close other programs that use "
                "the GPU, or lower --max-ctx." % (used, total))
    return None


def _add_engine_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model-dir", help="folder with the model files (default: $YASAKANI_MODEL_DIR)")
    sp.add_argument("--dll", help="path of yskn.dll (default: $YASAKANI_DLL, or the bin folder next to the package)")
    sp.add_argument("--max-ctx", type=int, default=None, help="context length in tokens: input + thinking + answer (default: the model's 262144)")
    sp.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    sp.add_argument("--seed0", type=int, default=1, help="seed of the first request; every request without its own seed uses the next one")


def _open_stack(args, *, warm: bool = True):
    """Model folder -> (pack, engine, formatter, runner).  The tokenizer is loaded first so a broken folder fails before the long load."""
    from .chat import ChatRunner
    from .engine import Engine
    from .textio import ChatFormatter
    pack = ModelPack.load(_model_dir(args))
    fmt = ChatFormatter(pack.root)
    _say("loading the model (%s of weights) ..." % _human(pack.weights.stat().st_size))
    t0 = time.time()
    eng = Engine(pack, max_ctx=args.max_ctx, device=args.device, dll=args.dll)
    try:
        if eng.removed_env:
            _say("note: ignored %d YSKN_* setting(s) from the environment: %s" % (len(eng.removed_env), ", ".join(sorted(eng.removed_env))))
        runner = ChatRunner(eng, fmt, seed0=args.seed0)
        if warm:
            runner.warmup()
    except BaseException:
        eng.close()
        raise
    _say("model ready in %.0f s (context %d tokens)" % (time.time() - t0, eng.max_ctx))
    note = _gpu_memory_warning(args.device)
    if note:
        _say(note)
    return pack, eng, fmt, runner


# ---- info -------------------------------------------------------------------------------------------------------------------------
def cmd_info(args) -> int:
    ok = True
    print("yasakani", __version__)
    print("python  ", platform.python_version(), platform.system(), platform.machine())
    for mod in ("numpy", "tokenizers", "jinja2"):
        try:
            m = importlib.import_module(mod)
            print("%-8s %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("%-8s MISSING  (pip install %s)" % (mod, mod))
            ok = False
    try:
        dll = find_library(args.dll)
        h = hashlib.sha256(dll.read_bytes()).hexdigest()
        print("engine   %s  %s  sha256 %s" % (dll.name, _human(dll.stat().st_size), h[:16]))
    except YasakaniError as ex:
        print("engine   NOT FOUND:", ex)
        ok = False
    for ln in _gpu_lines():
        print("gpu      " + ln)
    d = getattr(args, "model_dir", None) or os.environ.get("YASAKANI_MODEL_DIR")
    if d:
        try:
            pack = ModelPack.load(d)
            print("model    %s  (id %s, context up to %d tokens)" % (pack.root.name, pack.model_id, pack.max_ctx))
            for label, p in (("weights", pack.weights), ("draft", pack.draft), ("draft head", pack.draft_head), ("kv profile", pack.kv_profile)):
                print("  %-10s %-26s %s" % (label, p.name, _human(p.stat().st_size)))
            print("  checksums  %s" % ("%d files listed in yasakani.json" % len(pack.sha256) if pack.sha256 else "none listed (yasakani.json missing)"))
            from .textio import ChatFormatter
            ChatFormatter(pack.root)
            print("  tokenizer  ok (chat template found)")
        except (YasakaniError, ImportError, OSError, ValueError) as ex:
            print("model    PROBLEM:", ex)
            ok = False
    else:
        print("model    (give --model-dir to check the model folder)")
    return 0 if ok else 1


# ---- verify -----------------------------------------------------------------------------------------------------------------------
def cmd_verify(args) -> int:
    pack = ModelPack.load(_model_dir(args))
    if not pack.sha256:
        print("no checksums are listed for this folder (yasakani.json missing or without \"sha256\").")
        return 2
    print("checking %d files (reads about %s, one to two minutes) ..." % (len(pack.sha256), _human(sum((pack.root / n).stat().st_size for n in pack.sha256 if (pack.root / n).is_file()))))
    res = pack.verify(progress=lambda n, ok: print("  %-30s %s" % (n, "OK" if ok else "DIFFERENT"), flush=True))
    bad = [n for n, ok in res.items() if not ok]
    print("all files match." if not bad else "%d file(s) differ: %s  (download them again)" % (len(bad), ", ".join(bad)))
    return 0 if not bad else 1


# ---- serve ------------------------------------------------------------------------------------------------------------------------
def _stop_on_signals() -> None:
    """Ctrl+C, Ctrl+Break and a termination request all end the server the same orderly way."""
    def raise_interrupt(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, raise_interrupt)
            except (ValueError, OSError):
                pass


def cmd_serve(args) -> int:
    from .server import ApiState, ServerConfig, bind
    local = args.host in ("127.0.0.1", "localhost", "::1")
    if not local and not args.api_key:
        raise YasakaniError("refusing to listen on %s without --api-key: anyone who can reach this computer could use the GPU" % args.host)
    cfg = ServerConfig(host=args.host, port=args.port, api_key=args.api_key, default_max_tokens=args.default_max_tokens,
                       max_tokens_cap=args.max_tokens_cap)
    state = ApiState(cfg)
    state.version = __version__
    try:
        httpd = bind(state)
    except OSError as ex:
        raise YasakaniError("cannot listen on %s:%d (%s).  Is another server already using this port?  Use --port to pick another."
                            % (args.host, args.port, ex.strerror or type(ex).__name__))
    cfg.port = httpd.server_address[1]
    _stop_on_signals()
    threading.Thread(target=httpd.serve_forever, kwargs=dict(poll_interval=0.2), name="yasakani-http", daemon=True).start()
    base = "http://%s:%d" % ("[%s]" % args.host if ":" in args.host else args.host, cfg.port)
    _say("listening on %s  (answers /health while the model loads)" % base)
    eng = None
    try:
        pack, eng, fmt, runner = _open_stack(args, warm=not args.no_warmup)
        cfg.model_id = pack.model_id
        state.runner = runner
        _say("ready: OpenAI-compatible API at %s/v1   (model id %r%s)" % (base, cfg.model_id, ", API key required" if cfg.api_key else ""))
        _say("stop with Ctrl+C")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _say("stopping ...")
        return 0
    finally:
        state.runner = None
        httpd.shutdown()
        if eng is not None:
            eng.cancel()
            with state.gen_lock:
                eng.close()
        httpd.server_close()


# ---- chat -------------------------------------------------------------------------------------------------------------------------
CHAT_HELP = """commands:  /exit  quit      /reset  forget the conversation      /think on|off  thinking on or off (default on)
           /effort low|medium|xhigh  how long the model thinks      /help  this text      Ctrl+C while it writes: stop the answer"""


def cmd_chat(args) -> int:
    from .chat import ChatParams, RequestError
    pack, eng, fmt, runner = _open_stack(args, warm=True)
    color = _use_color()
    dim = (lambda s: "\x1b[2m%s\x1b[0m" % s) if color else (lambda s: s)
    kw = {}
    if args.no_think:
        kw["enable_thinking"] = False
    if args.effort:
        kw["reasoning_effort"] = args.effort
    base: List[dict] = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)
    print("YASAKANI chat - type /help for commands, /exit to quit.")
    try:
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, rest = line[1:].partition(" ")
                rest = rest.strip().lower()
                if cmd in ("exit", "quit", "q"):
                    return 0
                if cmd == "reset":
                    messages = list(base)
                    print("(conversation cleared)")
                elif cmd == "think" and rest in ("on", "off"):
                    kw["enable_thinking"] = rest == "on"
                    print("(thinking %s)" % rest)
                elif cmd == "effort" and rest in ("low", "medium", "xhigh"):
                    kw["reasoning_effort"] = rest
                    print("(reasoning effort %s)" % rest)
                else:
                    print(CHAT_HELP)
                continue
            messages.append({"role": "user", "content": line})
            params = ChatParams(messages=[dict(m) for m in messages], max_tokens=args.max_tokens, temperature=args.temperature,
                                top_p=args.top_p, top_k=args.top_k, seed=args.seed, loop_guard=not args.no_loop_guard,
                                template_kwargs=dict(kw))
            parts = {"reasoning": [], "content": []}
            last = [None]

            def emit(kind: str, text: str) -> None:
                parts[kind].append(text)
                if kind == "reasoning" and args.hide_thinking:
                    return
                if last[0] != kind:
                    if kind == "reasoning":
                        print(dim("[thinking]"), flush=True)
                    elif last[0] == "reasoning":
                        print(dim("\n[/thinking]\n"), flush=True)
                    last[0] = kind
                print(dim(text) if kind == "reasoning" else text, end="", flush=True)

            try:
                u = runner.run(params, emit)
            except KeyboardInterrupt:
                print("\n[stopped]")
                messages.pop()
                continue
            except RequestError as ex:
                print("cannot do that: %s" % ex)
                messages.pop()
                continue
            except YasakaniError as ex:
                print("\nengine error: %s" % ex)
                messages.pop()
                try:
                    eng.reset()
                except YasakaniError:
                    return 1
                continue
            if parts["reasoning"] and not parts["content"] and u.finish != "stop":
                print(dim("\n(the limit of %d tokens was reached while the model was still thinking, so no answer was written.  "
                          "/effort medium or /effort low shortens the thinking, /think off skips it, --max-tokens raises the limit)"
                          % args.max_tokens))
            tps = (u.completion_tokens / (u.gen_ms / 1000.0)) if u.gen_ms > 0 else 0.0
            extra = ""
            if u.loop_guard:
                extra = ", loop guard cut the thinking"
            print(dim("\n\n[%d tokens, %.0f tokens/s, %s%s]" % (u.completion_tokens, tps, "complete" if u.finish == "stop" else "hit the length limit", extra)))
            msg = {"role": "assistant", "content": "".join(parts["content"]).strip()}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            messages.append(msg)
    finally:
        eng.close()


# ---- bench ------------------------------------------------------------------------------------------------------------------------
BUILTIN_PROMPTS = [
    ("explain", "Explain how a hash table works, including collisions and resizing, with a short example."),
    ("code", "Write a Python function that merges overlapping intervals and add a few tests for it."),
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("proof", "Prove that the square root of 2 is irrational."),
]


def _load_prompts(path: Optional[str]):
    if not path:
        return list(BUILTIN_PROMPTS)
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [ln for ln in text.splitlines() if ln.strip()]
    out = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            out.append(("test%d" % (i + 1), item))
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            out.append((str(item.get("name") or "test%d" % (i + 1)), item["prompt"]))
        else:
            raise YasakaniError("%s: entry %d is neither a string nor {\"name\":..., \"prompt\":...}" % (path, i + 1))
    if not out:
        raise YasakaniError("%s contains no prompts" % path)
    return out


def cmd_bench(args) -> int:
    from .chat import ChatParams
    prompts = _load_prompts(args.prompts_file)
    pack, eng, fmt, runner = _open_stack(args, warm=True)
    rows = []
    try:
        _say("GPU: " + "; ".join(_gpu_lines()))
        kw = {} if args.think else {"enable_thinking": False}
        for name, text in prompts:
            for run in range(args.runs):
                eng.reset()
                p = ChatParams(messages=[{"role": "user", "content": text}], max_tokens=args.n, temperature=0.0, ignore_eos=True,
                               loop_guard=False, template_kwargs=dict(kw))
                u = runner.run(p, lambda kind, t: None)
                tps = u.completion_tokens / (u.gen_ms / 1000.0) if u.gen_ms > 0 else 0.0
                rows.append(dict(test=name, run=run + 1, prompt_tokens=u.prompt_tokens, tokens=u.completion_tokens, seconds=round(u.gen_ms / 1000.0, 3),
                                 tokens_per_s=round(tps, 1), tokens_per_pass=round(u.completion_tokens / max(1, u.passes), 2)))
                if not args.json:
                    r = rows[-1]
                    print("%-12s run %d  prompt %5d  generated %5d  %7.1f tokens/s  %.2f tokens/pass" %
                          (name, r["run"], r["prompt_tokens"], r["tokens"], r["tokens_per_s"], r["tokens_per_pass"]), flush=True)
    finally:
        eng.close()
    mean = sum(r["tokens_per_s"] for r in rows) / len(rows)
    if args.json:
        print(json.dumps(dict(version=__version__, model=pack.model_id, gpu=_gpu_lines(), tokens_per_test=args.n, runs=rows,
                              mean_tokens_per_s=round(mean, 1)), indent=1))
    else:
        print("mean over %d run(s): %.1f tokens/s   (greedy, thinking %s, %d tokens each)" % (len(rows), mean, "on" if args.think else "off", args.n))
    return 0


# ---- selfcheck --------------------------------------------------------------------------------------------------------------------
SELFCHECK_FORMAT = 1


def _selfcheck_cases():
    return [
        dict(name="short-answer", kind="chat", messages=[{"role": "user", "content": "Say hello in one short sentence."}],
             template_kwargs={"enable_thinking": False}, max_new=48),
        dict(name="thinking", kind="chat", messages=[{"role": "user", "content": "What is 17 times 23? Think it through step by step."}],
             template_kwargs={}, max_new=160),
        dict(name="plain-text", kind="raw", text="The capital of France is", max_new=32),
    ]


def _ids_sha(ids) -> str:
    import numpy as np
    return hashlib.sha256(np.asarray(ids, "<i4").tobytes()).hexdigest()


def _selfcheck_run(eng, fmt):
    """Greedy runs of the fixed cases -> one record per case (prompt hash, output hash, first ids)."""
    from .engine import Sampling
    eng.set_sampling(Sampling(temperature=0.0))
    out = []
    for c in _selfcheck_cases():
        prompt = fmt.encode(c["text"]) if c["kind"] == "raw" else fmt.prompt(c["messages"], **c["template_kwargs"])[0]
        eng.reset()
        r = eng.generate(prompt, c["max_new"])
        out.append(dict(name=c["name"], prompt_tokens=int(len(prompt)), prompt_sha256=_ids_sha(prompt), tokens=len(r.ids),
                        sha256=_ids_sha(r.ids), head=[int(x) for x in r.ids[:12]]))
    eng.reset()
    return out


def cmd_selfcheck(args) -> int:
    pack, eng, fmt, runner = _open_stack(args, warm=True)
    try:
        got = _selfcheck_run(eng, fmt)
    finally:
        eng.close()
    if args.write:
        Path(args.write).write_text(json.dumps(dict(format=SELFCHECK_FORMAT, model_id=pack.model_id, yasakani=__version__, cases=got), indent=1) + "\n",
                                    encoding="utf-8")
        print("wrote %d reference record(s) to %s" % (len(got), args.write))
        return 0
    ref_path = Path(args.reference) if args.reference else pack.root / "selfcheck.json"
    if not ref_path.is_file():
        print("no reference file (%s).  Download selfcheck.json next to the model files, or pass --reference FILE." % ref_path)
        return 2
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    if ref.get("format") != SELFCHECK_FORMAT:
        print("the reference file has an unknown format (%r)" % ref.get("format"))
        return 2
    want = {c["name"]: c for c in ref.get("cases", [])}
    bad = 0
    for g in got:
        w = want.get(g["name"])
        if w is None:
            print("  %-14s not in the reference file" % g["name"])
            bad += 1
        elif w["prompt_sha256"] != g["prompt_sha256"]:
            print("  %-14s FAIL  the prompt ids differ (tokenizer or chat template files are not the reference ones)" % g["name"])
            bad += 1
        elif w["sha256"] != g["sha256"] or w["tokens"] != g["tokens"]:
            n = next((i for i, (a, b) in enumerate(zip(w["head"], g["head"])) if a != b), None)
            print("  %-14s FAIL  the output differs%s" % (g["name"], "" if n is None else " (first different token: #%d)" % (n + 1)))
            bad += 1
        else:
            print("  %-14s PASS  (%d tokens)" % (g["name"], g["tokens"]))
    print("self-check passed: this installation reproduces the reference output." if not bad else "self-check FAILED for %d case(s)." % bad)
    return 0 if not bad else 1


# ---- entry point ------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="yasakani", description="Run Qwen3.8-27B locally on one RTX 5090 (Windows) with the YASAKANI engine.")
    ap.add_argument("--version", action="version", version="yasakani " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{info,serve,chat,bench,selfcheck,verify}")

    sp = sub.add_parser("info", help="check this computer, the engine library and (with --model-dir) a model folder")
    sp.add_argument("--model-dir")
    sp.add_argument("--dll")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("verify", help="check the model files against the checksums listed in yasakani.json")
    sp.add_argument("--model-dir")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("serve", help="start the OpenAI-compatible server")
    _add_engine_args(sp)
    sp.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1: this computer only)")
    sp.add_argument("--port", type=int, default=DEFAULT_PORT)
    sp.add_argument("--api-key", default=os.environ.get("YASAKANI_API_KEY"), help="require this key (or set YASAKANI_API_KEY); needed for any non-local --host")
    sp.add_argument("--default-max-tokens", type=int, default=32768, help="answer limit when a request does not give one")
    sp.add_argument("--max-tokens-cap", type=int, default=None, help="upper limit for any request")
    sp.add_argument("--no-warmup", action="store_true", help="skip the short warm-up run at start")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("chat", help="talk to the model in the terminal")
    _add_engine_args(sp)
    sp.add_argument("--system", help="system prompt")
    sp.add_argument("--max-tokens", type=int, default=32768)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--top-p", type=float, default=0.95)
    sp.add_argument("--top-k", type=int, default=20)
    sp.add_argument("--seed", type=int, default=None, help="fixed seed (same seed and prompt give the same answer)")
    sp.add_argument("--no-think", action="store_true", help="answer without a thinking phase")
    sp.add_argument("--effort", choices=("low", "medium", "xhigh"), help="how long the model thinks (default xhigh)")
    sp.add_argument("--hide-thinking", action="store_true", help="do not print the thinking")
    sp.add_argument("--no-loop-guard", action="store_true")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("selfcheck", help="run a few fixed greedy prompts and compare the output with the reference file")
    _add_engine_args(sp)
    sp.add_argument("--reference", help="reference file (default: selfcheck.json in the model folder)")
    sp.add_argument("--write", metavar="FILE", help="write a new reference file instead of checking (for people who publish a model folder)")
    sp.set_defaults(func=cmd_selfcheck)

    sp = sub.add_parser("bench", help="measure generation speed (greedy, fixed length)")
    _add_engine_args(sp)
    sp.add_argument("--prompts-file", help="JSON list of prompts (strings or {\"name\", \"prompt\"}) or a text file with one prompt per line")
    sp.add_argument("--n", type=int, default=512, help="tokens to generate per test (default 512)")
    sp.add_argument("--runs", type=int, default=1, help="repeats per test")
    sp.add_argument("--think", action="store_true", help="let the model think first (default: answer directly)")
    sp.add_argument("--json", action="store_true", help="print the result as JSON")
    sp.set_defaults(func=cmd_bench)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    _fix_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except YasakaniError as ex:
        _say("error: %s" % ex)
        return 1
    except KeyboardInterrupt:
        return 130
