# -*- coding: utf-8 -*-
"""YASAKANI quickstart (Windows + one NVIDIA RTX 5090).

Double-click START.cmd.  This script then
  1. checks this computer (Windows, Python, GPU, free disk space),
  2. lets you pick one of the two models,
  3. downloads the engine zip from the release page and checks its SHA-256,
  4. makes a private Python environment in this folder and installs the package,
  5. downloads the model folder from Hugging Face,
  6. runs `yasakani info`, `yasakani verify` and `yasakani selfcheck`,
  7. and offers: measure the speed, compare look-ahead on and off, chat in your browser or in the terminal, or start the
     local server.

The browser chat (chat.html) is served by this script on 127.0.0.1 only.  The script passes the requests of that page on to
`yasakani serve`, which runs on this computer too.  Nothing is sent anywhere else.

Everything is kept inside the folder this file is in.  Nothing is installed system-wide.
Run it again any time: finished steps are skipped.

Apache License 2.0 - part of https://github.com/yasakani-dev/yasakani
"""
import argparse
import hashlib
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QS_VERSION = "0.1.0"
ENGINE_ZIP = "yasakani-0.1.0-win-x64.zip"
ENGINE_URL = "https://github.com/yasakani-dev/yasakani/releases/download/v0.1.0/" + ENGINE_ZIP
ENGINE_SHA256 = "c1723dd905fd37972416c361c40022fccafb7d8b4517dcce342df8d8b36df7b0"
REPO_URL = "https://github.com/yasakani-dev/yasakani"
SUPPORT_URL = "https://camp-fire.jp/projects/992233/view"
ENGINE_PORT, WEB_PORT = 8765, 8766      # `yasakani serve` and the browser chat; the next free ports are used if these are taken
MAX_BODY = 16 << 20
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'")
MODELS = {
    "quality": {"repo": "yasakani-dev/Qwen3.8-27B-YASAKANI-quality", "need_gb": 16},
    "speed": {"repo": "yasakani-dev/Qwen3.8-27B-YASAKANI-speed", "need_gb": 14},
}

T = {
    "banner": {
        "ja": "YASAKANI クイックスタート {v}  -  Qwen3.8-27B を RTX 5090 1 枚で動かします",
        "en": "YASAKANI quickstart {v}  -  runs Qwen3.8-27B on one RTX 5090",
    },
    "support": {
        "ja": "開発を応援してください: CAMPFIRE のクラウドファンディング（2026 年 12 月 16 日まで）\n  {u}",
        "en": "Like it? A star on GitHub helps: {r}\n  (The crowdfunding at {u} is open to residents of Japan.)",
    },
    "not_windows": {
        "ja": "このソフトは Windows 11 専用です。",
        "en": "This software runs on Windows 11 only.",
    },
    "old_python": {
        "ja": "Python 3.10 以上が必要です（いまは {v}）。",
        "en": "Python 3.10 or newer is needed (this is {v}).",
    },
    "bad_path": {
        "ja": "このフォルダーの場所に、日本語などの英数字以外の文字が入っています:\n  {p}\n"
              "C:\\yasakani のような英数字だけの場所にフォルダーごと移して、もう一度 START.cmd をダブルクリックしてください。",
        "en": "The path of this folder contains characters outside plain ASCII:\n  {p}\n"
              "Move the folder to a plain path such as C:\\yasakani and double-click START.cmd again.",
    },
    "step": {"ja": "[{i}/{n}] {t}", "en": "[{i}/{n}] {t}"},
    "s_check": {"ja": "このパソコンの確認", "en": "Checking this computer"},
    "s_model": {"ja": "モデルを選ぶ", "en": "Choosing the model"},
    "s_engine": {"ja": "エンジンのダウンロードと確認", "en": "Downloading and checking the engine"},
    "s_venv": {"ja": "Python の環境を作って、パッケージを入れる", "en": "Making the Python environment and installing the package"},
    "s_dl": {"ja": "モデルのダウンロード", "en": "Downloading the model"},
    "s_verify": {"ja": "点検（info・verify・selfcheck）", "en": "Checks (info, verify, selfcheck)"},
    "s_menu": {"ja": "準備ができました", "en": "Ready"},
    "no_smi": {
        "ja": "nvidia-smi が見つかりません。NVIDIA のドライバーを入れてから、もう一度試してください。",
        "en": "nvidia-smi was not found. Install the NVIDIA driver and try again.",
    },
    "gpu_is": {"ja": "GPU: {g}", "en": "GPU: {g}"},
    "gpu_other": {
        "ja": "この版は NVIDIA GeForce RTX 5090 専用です。ほかの GPU では動きません。",
        "en": "This version is built for the NVIDIA GeForce RTX 5090 only. It does not run on other GPUs.",
    },
    "continue_q": {"ja": "それでも続けますか？ [y/N]: ", "en": "Continue anyway? [y/N]: "},
    "disk": {
        "ja": "空き容量: {f:.0f} GB（必要: 約 {n} GB）",
        "en": "Free disk space: {f:.0f} GB (needed: about {n} GB)",
    },
    "disk_short": {
        "ja": "空き容量が足りません。約 {n} GB を空けてから、もう一度試してください。",
        "en": "Not enough free disk space. Free about {n} GB and try again.",
    },
    "pick": {
        "ja": "どちらのモデルを使いますか？\n"
              "  1) 品質モデル〈おすすめ〉  元のモデルに近い。ファイル 12.4 GB\n"
              "  2) 最速モデル            いちばん速い。ファイル 10.3 GB\n"
              "番号を入れて Enter [{d}]: ",
        "en": "Which model?\n"
              "  1) quality model (recommended)  closest to the original. 12.4 GB file\n"
              "  2) fastest model                the fastest. 10.3 GB file\n"
              "Type the number and press Enter [{d}]: ",
    },
    "picked": {"ja": "モデル: {m}", "en": "Model: {m}"},
    "m_quality": {"ja": "品質モデル", "en": "quality model"},
    "m_speed": {"ja": "最速モデル", "en": "fastest model"},
    "checks_done": {"ja": "点検は済んでいます（飛ばします）。", "en": "The checks were done before (skipped)."},
    "engine_have": {"ja": "エンジンは準備済みです。", "en": "The engine is already in place."},
    "engine_local": {"ja": "このフォルダーにある {z} を使います。", "en": "Using {z} from this folder."},
    "engine_get": {"ja": "ダウンロード中: {u}", "en": "Downloading: {u}"},
    "sha_ok": {"ja": "SHA-256 は公開ページの値と同じです。", "en": "The SHA-256 matches the published one."},
    "sha_bad": {
        "ja": "SHA-256 が違います（ダウンロードが壊れたか、別のファイルです）。使わずに止めました。もう一度試してください。\n  期待: {e}\n  実際: {a}",
        "en": "The SHA-256 differs (broken download or a different file). It was not used; please try again.\n  expected: {e}\n  got:      {a}",
    },
    "dl_fail": {
        "ja": "ダウンロードできませんでした: {e}\nインターネットの接続を確かめて、もう一度 START.cmd をダブルクリックしてください（続きから始まります）。",
        "en": "The download failed: {e}\nCheck the internet connection and double-click START.cmd again (it continues where it stopped).",
    },
    "venv_have": {"ja": "Python の環境は準備済みです。", "en": "The Python environment is already in place."},
    "pip_fail": {
        "ja": "パッケージを入れられませんでした。上のメッセージを確かめてください。",
        "en": "Installing the package failed. See the messages above.",
    },
    "model_have": {"ja": "モデルは準備済みです: {p}", "en": "The model is already in place: {p}"},
    "model_get": {
        "ja": "Hugging Face から {r} をダウンロードします（約 {g} GB。回線によっては時間がかかります）。",
        "en": "Downloading {r} from Hugging Face (about {g} GB; this can take a while).",
    },
    "retry_dl": {"ja": "もう一度、別の方法で試します。", "en": "Trying once more with another transfer method."},
    "run": {"ja": "> yasakani {c}", "en": "> yasakani {c}"},
    "check_fail": {
        "ja": "「yasakani {c}」が失敗しました（終了コード {rc}）。上のメッセージを確かめてください。\n"
              "よくある原因: ほかのプログラムが GPU のメモリを使っている、ドライバーが古い、Visual C++ ランタイムが無い。\n"
              "FAQ: {r}/blob/main/docs/FAQ.ja.md",
        "en": "`yasakani {c}` failed (exit code {rc}). See the messages above.\n"
              "Common causes: another program is using the GPU memory, an old driver, or the Visual C++ runtime is missing.\n"
              "FAQ: {r}/blob/main/docs/FAQ.md",
    },
    "checks_ok": {
        "ja": "点検はすべて合格です。このパソコンは、公開されている基準と同じ出力を再現しました。",
        "en": "All checks passed. This computer reproduces the published reference output.",
    },
    "menu": {
        "ja": "\n何をしますか？\n"
              "  1) 速さを測る             （yasakani bench）\n"
              "  2) 先読みあり・なしを比べる （技術の解説の数字を、このパソコンで確かめます。1 分ほど）\n"
              "  3) ブラウザーでチャット     （ブラウザーが開きます）\n"
              "  4) この画面でチャット       （yasakani chat。終わるときは /exit）\n"
              "  5) サーバーを起動           （OpenAI 互換。http://127.0.0.1:8765/v1 。止めるときは Ctrl+C）\n"
              "  6) 終わる\n"
              "番号を入れて Enter: ",
        "en": "\nWhat next?\n"
              "  1) measure the speed             (yasakani bench)\n"
              "  2) compare look-ahead on and off (checks the numbers of the technical notes on this PC; about a minute)\n"
              "  3) chat in the browser           (a browser window opens)\n"
              "  4) chat in this window           (yasakani chat; type /exit to leave)\n"
              "  5) start the server              (OpenAI-compatible, http://127.0.0.1:8765/v1 ; Ctrl+C stops it)\n"
              "  6) quit\n"
              "Type the number and press Enter: ",
    },
    "web_boot": {
        "ja": "サーバーを起動します（モデルの読み込みに 1 分ほどかかることがあります）。",
        "en": "Starting the server (loading the model can take about a minute).",
    },
    "web_ready": {
        "ja": "\nブラウザーでチャットできます: {u}\n"
              "（ブラウザーが自動で開きます。開かないときは、上のアドレスをブラウザーに入れてください）\n"
              "終わるときは、この画面で Enter を押してください。",
        "en": "\nChat in your browser: {u}\n"
              "(A browser window opens by itself. If it does not, type the address above into your browser.)\n"
              "Press Enter in this window to stop.",
    },
    "web_ready_manual": {
        "ja": "\nブラウザーでチャットできます: {u}\n"
              "（上のアドレスをブラウザーに入れてください）\n"
              "終わるときは、この画面で Enter を押してください。",
        "en": "\nChat in your browser: {u}\n"
              "(Type the address above into your browser.)\n"
              "Press Enter in this window to stop.",
    },
    "cmp_nofile": {
        "ja": "lookahead_check.py が見つかりません: {p}\nzip をもう一度展開してください。",
        "en": "lookahead_check.py was not found: {p}\nUnpack the zip again.",
    },
    "web_fail": {
        "ja": "サーバーを起動できませんでした。上のメッセージを確かめてください。",
        "en": "The server did not start. See the messages above.",
    },
    "web_nopage": {
        "ja": "chat.html が見つかりません: {p}\nzip をもう一度展開してください。",
        "en": "chat.html was not found: {p}\nUnpack the zip again.",
    },
    "web_end": {"ja": "チャットを終わりました。", "en": "The chat was stopped."},
    "load_note": {
        "ja": "（モデルを読み込みます。初回は 1 分ほどかかることがあります）",
        "en": "(loading the model; the first time this can take about a minute)",
    },
    "again": {
        "ja": "次からも START.cmd をダブルクリックするだけです（済んだ手順は飛ばします）。\n"
              "自分でコマンドを打つときの設定は {f} に書きました。",
        "en": "Next time just double-click START.cmd again (finished steps are skipped).\n"
              "The settings for typing commands yourself are in {f}.",
    },
    "bye": {"ja": "終了します。", "en": "Bye."},
    "stopped": {"ja": "中止しました。", "en": "Stopped."},
}


class Ui:
    def __init__(self, lang):
        self.lang = lang

    def t(self, key, **kw):
        s = T[key].get(self.lang) or T[key]["en"]
        return s.format(**kw) if kw else s

    def say(self, key, **kw):
        print(self.t(key, **kw), flush=True)

    def step(self, i, n, key):
        print("\n" + "=" * 72, flush=True)
        print(self.t("step", i=i, n=n, t=self.t(key)), flush=True)
        print("=" * 72, flush=True)


def ui_language():
    try:
        import ctypes
        lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        return "ja" if (lid & 0x3FF) == 0x11 else "en"
    except Exception:
        return "en"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "yasakani-quickstart/" + QS_VERSION})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while True:
            b = r.read(1 << 18)
            if not b:
                break
            f.write(b)
            done += len(b)
            pct = int(done * 100 / total) if total else -1
            if pct != shown and (pct % 5 == 0 or pct < 0):
                shown = pct
                if total:
                    print("  %3d%%  %.1f / %.1f MB" % (pct, done / 1e6, total / 1e6), flush=True)
    os.replace(tmp, dst)


def safe_extract(zip_path, dst):
    root = os.path.abspath(dst)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            p = os.path.abspath(os.path.join(root, m.filename))
            if not (p == root or p.startswith(root + os.sep)):
                raise RuntimeError("unsafe path in zip: " + m.filename)
        z.extractall(root)


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def ask(prompt, default=""):
    try:
        s = input(prompt).strip()
    except EOFError:
        s = ""
    return s or default


def gpu_names():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------------------------------------------- browser chat
def port_is_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def pick_port(preferred, avoid=()):
    for p in [preferred] + list(range(preferred + 10, preferred + 60)):
        if p not in avoid and port_is_free(p):
            return p
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class LocalServer(ThreadingHTTPServer):
    """Listens on 127.0.0.1 only and never shares its port."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_web_handler(page, conf, engine_port, web_port):
    """The handler of the browser chat: the page, its settings, and three paths passed on to `yasakani serve`.

    Like `yasakani serve`, it answers only requests that name this address in ``Host`` and, if they carry an ``Origin``, come from
    this address - so a web page of another site that is open in the browser cannot talk to the model through it."""
    own = ("127.0.0.1:%d" % web_port, "localhost:%d" % web_port)
    origins = tuple("http://" + h for h in own)
    conf_bytes = json.dumps(conf).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 300
        server_version = "yasakani-quickstart/" + QS_VERSION
        sys_version = ""

        def log_message(self, *args):
            pass

        def _refused(self):
            if (self.headers.get("Host") or "").lower() not in own:
                return "host"
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() not in origins:
                return "origin"
            if (self.headers.get("Sec-Fetch-Site") or "none").lower() not in ("same-origin", "none"):
                return "site"
            return None

        def _send(self, code, ctype, body, extra=(), close=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in extra:
                self.send_header(k, v)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, message, kind, close=False):
            self._send(code, "application/json; charset=utf-8",
                       json.dumps({"error": {"message": message, "type": kind}}).encode("utf-8"), close=close)

        def _relay(self, method, path, body=None):
            conn = http.client.HTTPConnection("127.0.0.1", engine_port, timeout=900)
            started = False
            try:
                headers = {"Accept": "*/*"}
                if body is not None:
                    headers["Content-Type"] = "application/json"
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                ctype = resp.getheader("Content-Type") or "application/octet-stream"
                if ctype.lower().startswith("text/event-stream"):
                    started = True
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = resp.read()
                    started = True
                    self._send(resp.status, ctype, data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True      # the browser went away; closing the connection below stops the generation
            except (OSError, http.client.HTTPException) as ex:
                self.close_connection = True
                if not started:
                    try:
                        self._error(502, "the YASAKANI server did not answer (%s)" % type(ex).__name__, "upstream", close=True)
                    except OSError:
                        pass
            finally:
                conn.close()

        def do_GET(self):
            why = self._refused()
            if why:
                return self._error(403, "forbidden: " + why, "forbidden", close=True)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", page, (("Content-Security-Policy", PAGE_CSP),))
            if path == "/webchat/config":
                return self._send(200, "application/json; charset=utf-8", conf_bytes)
            if path in ("/health", "/v1/models"):
                return self._relay("GET", path)
            if path == "/favicon.ico":
                return self._send(404, "text/plain; charset=utf-8", b"")
            return self._error(404, "not found", "not_found")

        def do_POST(self):
            why = self._refused()
            if why:
                return self._error(403, "forbidden: " + why, "forbidden", close=True)
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                return self._error(404, "not found", "not_found", close=True)
            try:
                n = int(self.headers.get("Content-Length", "") or -1)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._error(413, "request body too large or length missing", "invalid_request_error", close=True)
            self._relay("POST", "/v1/chat/completions", self.rfile.read(n))

    return Handler


def get_json(port, path, headers=None, timeout=5):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path, headers=headers or {})
        r = conn.getresponse()
        data = r.read()
        try:
            return r.status, json.loads(data or b"{}"), r
        except ValueError:
            return r.status, {}, r
    finally:
        conn.close()


def wait_ready(port, proc, seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            return False
        try:
            code, body, _ = get_json(port, "/health")
            if code == 200 and body.get("status") == "ok":
                return True
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(1.0)
    return False


def web_selftest(web_port):
    """Checks the browser chat once, through the same address the browser uses.  Prints one line per check."""
    results = []

    def line(name, good, extra=""):
        results.append(bool(good))
        print("  %-46s %s%s" % (name, "PASS" if good else "FAIL", ("   " + extra) if extra else ""), flush=True)

    def raw(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", web_port, timeout=600)
        try:
            h = dict(headers or {})
            if body is not None:
                h.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.getheader("Content-Security-Policy") or "", r.read()
        finally:
            conn.close()

    print("\nbrowser chat self-test (http://127.0.0.1:%d/)" % web_port, flush=True)
    try:
        code, ctype, csp, data = raw("GET", "/")
        line("the page is served", code == 200 and ctype.startswith("text/html") and b"YASAKANI" in data and "default-src 'none'" in csp,
             "%d bytes" % len(data))
        code, ctype, _, data = raw("GET", "/webchat/config")
        line("the settings of the page", code == 200 and json.loads(data).get("version") == QS_VERSION)
        code, ctype, _, data = raw("GET", "/health")
        h = json.loads(data) if code == 200 else {}
        line("the model is ready", h.get("status") == "ok", "context %s tokens" % h.get("max_context_tokens"))
        code, _, _, _ = raw("GET", "/health", headers={"Origin": "http://example.com"})
        line("a page of another site is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("GET", "/health", headers={"Host": "example.com"})
        line("another host name is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("POST", "/v1/chat/completions", body=b"{}", headers={"Sec-Fetch-Site": "cross-site"})
        line("a cross-site request is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("GET", "/yasakani_env.cmd")
        line("other files are not served", code == 404, "HTTP %d" % code)
        body = json.dumps({"messages": [{"role": "user", "content": "Write the numbers from 1 to 120, separated by spaces."}],
                           "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 480,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", web_port, timeout=600)
        try:
            conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            n, text, usage, t_first, t_last, buf = 0, [], None, 0.0, 0.0, b""
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    if not ev.startswith(b"data:"):
                        continue
                    d = ev[5:].strip()
                    if d == b"[DONE]":
                        continue
                    o = json.loads(d)
                    if o.get("usage"):
                        usage = o["usage"]
                    for c in o.get("choices") or []:
                        piece = (c.get("delta") or {}).get("content")
                        if piece:
                            n += 1
                            text.append(piece)
                            t_last = time.perf_counter()
                            t_first = t_first or t_last
            secs = t_last - t_first
            got = usage["completion_tokens"] if usage else n
            rate = (got - 1) / secs if got > 16 and secs > 0.1 else 0.0
            answer = "".join(text)
            line("an answer arrives as a stream", r.status == 200 and n > 16 and " 20 " in answer,
                 "%d tokens, %.0f tokens/s" % (got, rate))
        finally:
            conn.close()
    except Exception as ex:  # noqa: BLE001 - any failure is a failed self-test
        line("self-test ran to the end", False, "%s: %s" % (type(ex).__name__, ex))
    good = all(results)
    print("browser chat self-test: %s" % ("PASS" if good else "FAIL"), flush=True)
    return good


def run_web(ui, a, vpy, env, mdir, dll, model):
    page_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")
    if not os.path.isfile(page_path):
        ui.say("web_nopage", p=page_path)
        return 7
    with open(page_path, "rb") as f:
        page = f.read()
    eport = pick_port(ENGINE_PORT)
    wport = pick_port(WEB_PORT, avoid=(eport,))
    ui.say("web_boot")
    proc = subprocess.Popen([vpy, "-m", "yasakani", "serve", "--model-dir", mdir, "--dll", dll, "--port", str(eport)], env=env)
    httpd = None
    try:
        if not wait_ready(eport, proc):
            ui.say("web_fail")
            return 7
        conf = {"lang": ui.lang, "model": "custom" if a.model_dir else model, "version": QS_VERSION, "repo": REPO_URL, "support": SUPPORT_URL}
        httpd = LocalServer(("127.0.0.1", wport), make_web_handler(page, conf, eport, wport))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d/" % wport
        if a.web_test:
            return 0 if web_selftest(wport) else 8
        ui.say("web_ready_manual" if a.no_browser else "web_ready", u=url)
        if not a.no_browser:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - the address is printed above
                pass
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                input()
            else:
                proc.wait()
        except (EOFError, KeyboardInterrupt):
            pass
        return 0
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        ui.say("web_end")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YASAKANI quickstart")
    ap.add_argument("--model", choices=sorted(MODELS), help="skip the question: quality or speed")
    ap.add_argument("--model-dir", help="use this model folder instead of downloading one")
    ap.add_argument("--engine-zip", help="use this engine zip instead of downloading it (its SHA-256 is still checked)")
    ap.add_argument("--home", help="where to keep everything (default: the folder of this file)")
    ap.add_argument("--action", choices=("menu", "bench", "compare", "web", "chat", "serve", "none"), default="menu",
                    help="what to do after the checks (default: show the menu); compare = look-ahead on and off, "
                         "web = chat in the browser")
    ap.add_argument("--no-browser", action="store_true", help="with the browser chat: do not open a browser window")
    ap.add_argument("--web-test", action="store_true", help="start the browser chat, check it once and stop")
    ap.add_argument("--lang", choices=("ja", "en"), help="language of the messages (default: the Windows display language)")
    ap.add_argument("--yes", action="store_true", help="answer yes to questions")
    ap.add_argument("--skip-checks", action="store_true", help="do not run info / verify / selfcheck")
    a = ap.parse_args(argv)
    if a.web_test:
        a.action = "web"

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ui = Ui(a.lang or ui_language())
    print()
    ui.say("banner", v=QS_VERSION)
    print(REPO_URL)

    if sys.platform != "win32":
        ui.say("not_windows")
        return 2
    if sys.version_info < (3, 10):
        ui.say("old_python", v=sys.version.split()[0])
        return 2
    home = os.path.abspath(a.home or os.path.dirname(os.path.abspath(__file__)))
    try:
        home.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=home)
        return 2
    os.makedirs(home, exist_ok=True)
    state_path = os.path.join(home, "quickstart_state.json")
    st = load_state(state_path)
    n = 7

    # 1. this computer
    ui.step(1, n, "s_check")
    print("Python %s" % sys.version.split()[0])
    try:
        gpus = gpu_names()
    except (OSError, subprocess.SubprocessError):
        ui.say("no_smi")
        return 2
    for g in gpus:
        ui.say("gpu_is", g=g)
    if not any("RTX 5090" in g for g in gpus):
        ui.say("gpu_other")
        if not a.yes and ask(ui.t("continue_q"), "n").lower() not in ("y", "yes"):
            ui.say("stopped")
            return 2

    # 2. model
    ui.step(2, n, "s_model")
    ok = st.setdefault("models_ok", {})
    model = a.model
    if not model:
        prev = st.get("model") or "quality"
        if a.yes or a.model_dir or a.action != "menu":
            model = prev
        else:
            d = "2" if prev == "speed" else "1"
            model = "speed" if ask(ui.t("pick", d=d), d) == "2" else "quality"
    ui.say("picked", m=ui.t("m_" + model))
    mdir = os.path.abspath(a.model_dir) if a.model_dir else os.path.join(home, "models", model)
    try:
        mdir.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=mdir)
        return 2
    have_model = os.path.isfile(os.path.join(mdir, "yasakani.json"))
    need = 2 if (a.model_dir or have_model) else MODELS[model]["need_gb"]
    free = shutil.disk_usage(home).free / 1e9
    ui.say("disk", f=free, n=need)
    if free < need:
        ui.say("disk_short", n=need)
        return 2
    st["model"] = model
    save_state(state_path, st)

    # 3. engine
    ui.step(3, n, "s_engine")
    edir = os.path.join(home, "engine")
    dll = os.path.join(edir, "yskn.dll")
    if st.get("engine_sha") == ENGINE_SHA256 and os.path.isfile(dll) and os.path.isfile(os.path.join(edir, "pyproject.toml")):
        ui.say("engine_have")
    else:
        local = a.engine_zip or os.path.join(home, ENGINE_ZIP)
        if os.path.isfile(local):
            ui.say("engine_local", z=os.path.basename(local))
            zpath, fetched = local, False
        else:
            os.makedirs(os.path.join(home, "downloads"), exist_ok=True)
            zpath, fetched = os.path.join(home, "downloads", ENGINE_ZIP), True
            ui.say("engine_get", u=ENGINE_URL)
            try:
                download(ENGINE_URL, zpath)
            except Exception as e:  # network errors of any kind
                ui.say("dl_fail", e="%s: %s" % (type(e).__name__, e))
                return 3
        got = sha256_of(zpath)
        if got != ENGINE_SHA256:
            if fetched:
                os.remove(zpath)
            ui.say("sha_bad", e=ENGINE_SHA256, a=got)
            return 3
        ui.say("sha_ok")
        if os.path.isdir(edir):
            shutil.rmtree(edir)
        safe_extract(zpath, edir)
        st["engine_sha"], st["venv_ok"] = ENGINE_SHA256, False
        save_state(state_path, st)

    # 4. environment
    ui.step(4, n, "s_venv")
    vdir = os.path.join(home, "venv")
    vpy = os.path.join(vdir, "Scripts", "python.exe")
    if st.get("venv_ok") and os.path.isfile(vpy):
        ui.say("venv_have")
    else:
        if not os.path.isfile(vpy):
            subprocess.check_call([sys.executable, "-m", "venv", vdir])
        rc = subprocess.call([vpy, "-m", "pip", "install", "--disable-pip-version-check", edir, "huggingface_hub"])
        if rc != 0:
            ui.say("pip_fail")
            return 4
        st["venv_ok"] = True
        save_state(state_path, st)

    env = dict(os.environ)
    env["YASAKANI_DLL"] = dll
    env["YASAKANI_MODEL_DIR"] = mdir

    # 5. model files
    ui.step(5, n, "s_dl")
    got = st.setdefault("models_downloaded", {})
    if a.model_dir or (got.get(model) and have_model):
        ui.say("model_have", p=mdir)
    else:
        repo = MODELS[model]["repo"]
        ui.say("model_get", r=repo, g=MODELS[model]["need_gb"] - 1)
        os.makedirs(mdir, exist_ok=True)
        code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
                "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])\n")
        rc = subprocess.call([vpy, "-c", code, repo, mdir], env=env)
        if rc != 0:
            ui.say("retry_dl")
            env2 = dict(env)
            env2["HF_HUB_DISABLE_XET"] = "1"
            rc = subprocess.call([vpy, "-c", code, repo, mdir], env=env2)
        if rc != 0:
            ui.say("dl_fail", e="exit code %d" % rc)
            return 5
        got[model] = True
        save_state(state_path, st)

    def yk(cmd, *extra):
        args = [vpy, "-m", "yasakani", cmd, "--model-dir", mdir]
        if cmd != "verify":
            args += ["--dll", dll]
        args += list(extra)
        print()
        ui.say("run", c=cmd)
        return subprocess.call(args, env=env)

    # 6. checks
    ui.step(6, n, "s_verify")
    if a.skip_checks or (ok.get(model) and not a.model_dir):
        ui.say("checks_done")
    else:
        for cmd in ("info", "verify", "selfcheck"):
            if cmd == "selfcheck":
                ui.say("load_note")
            rc = yk(cmd)
            if rc != 0:
                ui.say("check_fail", c=cmd, rc=rc, r=REPO_URL)
                return 6
        print()
        ui.say("checks_ok")
        if not a.model_dir:
            ok[model] = True
            save_state(state_path, st)

    envfile = os.path.join(home, "yasakani_env.cmd")
    with open(envfile, "w", encoding="ascii", newline="\r\n") as f:
        f.write("@echo off\nrem Run this in a command prompt, then type: yasakani chat   (or serve, bench, info ...)\n")
        f.write('set "YASAKANI_DLL=%s"\nset "YASAKANI_MODEL_DIR=%s"\nset "PATH=%s;%%PATH%%"\n'
                % (dll, mdir, os.path.join(vdir, "Scripts")))

    # 7. use it
    ui.step(7, n, "s_menu")
    print(ui.t("support", u=SUPPORT_URL, r=REPO_URL))
    action = a.action
    while True:
        if action == "menu":
            pick = ask(ui.t("menu"), "6")
            cmd = {"1": "bench", "2": "compare", "3": "web", "4": "chat", "5": "serve"}.get(pick)
        else:
            cmd = None if action == "none" else action
        if not cmd:
            break
        if cmd == "web":
            rc = run_web(ui, a, vpy, env, mdir, dll, model)
            if rc and action != "menu":
                return rc
        elif cmd == "compare":
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lookahead_check.py")
            if not os.path.isfile(script):
                ui.say("cmp_nofile", p=script)
                rc = 7
            else:
                print()
                rc = subprocess.call([vpy, script, "--model-dir", mdir, "--dll", dll, "--lang", ui.lang], env=env)
            if rc and action != "menu":
                return rc
        else:
            ui.say("load_note")
            yk(cmd)
        if action != "menu":
            break
    print()
    ui.say("again", f=envfile)
    print(ui.t("support", u=SUPPORT_URL, r=REPO_URL))
    ui.say("bye")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
