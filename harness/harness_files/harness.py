# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""YASAKANI harness starter: an agent screen in the browser, run by the model on this PC.

It starts two programs and stops them together:

  1. `yasakani serve`  - the model on this PC, as an OpenAI-compatible server with tool calling,
  2. DeepSeek Harness  - an open-source agent program by DeepSeek (MIT license; https://github.com/deepseek-ai/deepseek-harness).
                         Its web screen is opened in the browser.  It is used unchanged; a small settings file names the local
                         server as its model and switches off everything that would send data to its maker.

The folder of the quickstart (engine, Python environment, model) is used as it is; run START.cmd of the quickstart once before
this.  That folder is looked for where START_HARNESS.cmd is, in the folder above it and in the folders next to it (or name it
with --home).  Tool calling is not in the 0.1.0 package the quickstart installs: the server is started with the preview of the
next package version that lies next to this file (python/yasakani, through PYTHONPATH).  Nothing of the quickstart is changed.
DeepSeek Harness is installed on first use from the npm registry (exactly the packages of dsh.package-lock.json, each checked
against its checksum).  If the PC has no Node.js 24 or newer, Node.js is fetched from nodejs.org into the harness folder.
Everything this starter adds lives in the folder "harness" next to START_HARNESS.cmd; removing that folder removes all of it.

    python harness.py [--home DIR] [--model quality|speed] [--lang ja|en] [--no-browser]

Options for testing: --dsh-dir, --harness-dir, --pypath, --model-dir, --dll, --python, --node, --own-node, --node-url,
--npm-cache, --engine-port, --web-port, --selftest, --minutes, --stop-file, --any-os.
"""
import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HARNESS_VERSION = "0.1.0"
DSH_PACKAGE = "@deepseek-ai/dsh"
DSH_VERSION = "0.2.1-rc.2"                       # the version all tests were run with
DSH_SOURCE = "https://github.com/deepseek-ai/deepseek-harness"
REPO_URL = "https://github.com/yasakani-dev/yasakani"
SUPPORT_URL = "https://camp-fire.jp/projects/992233/view"
NODE_MIN = 24                                    # the major version the tests were run with; an older one on the PC is not used
NODE_VERSION = "v24.16.0"                        # LTS; the version of the tests.  Fetched only when the PC has no Node.js of NODE_MIN or newer
NODE_ZIP = "node-%s-win-x64.zip" % NODE_VERSION
NODE_URL = "https://nodejs.org/dist/%s/%s" % (NODE_VERSION, NODE_ZIP)
NODE_SHA256 = "edaca9bd58ec8e92037dac4e877d52f6b8f430b81c18b57e264b4e2fb111cd56"   # from the release post of nodejs.org for this version
THINKING = "high"                                # how much a new session thinks: off, low, medium or high (high = the model's longest
                                                 # setting; the screen can change it)
MAX_OUTPUT = 98304                               # tokens one reply may take, thinking included (the harness asks the server for this many)
THINKING_BUDGET = 49152                          # a thinking longer than this is closed by the server and the model goes on to the answer
MODEL_IDS = {"quality": "qwen3.8-27b-quality", "speed": "qwen3.8-27b-speed"}       # the model names the screen uses when both are installed
COMPACT_HEADROOM = 65536                         # DeepSeek Harness shortens the conversation when input + MAX_OUTPUT + this would not fit

PATCH = """# Written by the YASAKANI harness starter at every start - changes made here are lost.
# The local YASAKANI server is the model provider of the DeepSeek Harness.
- id: llm-pi-ai
  config:
    providers:
      yasakani:
        displayName: YASAKANI (this PC)
        api: openai-completions
        baseURL: http://127.0.0.1:{engine_port}/v1
        apiKeyEnv: YASAKANI_API_KEY
        compat:
          supportsStore: false
          supportsDeveloperRole: false
          supportsReasoningEffort: true
          supportsUsageInStreaming: true
          maxTokensField: max_tokens
          thinkingFormat: qwen
{reasoning}        models:
{models}- id: agent-default-model
  config:
    provider: yasakani
    model: {model_id}
{effort}# no extra model request for the title of a session - the first words of the first message are used
- id: session-title-llm
  disabled: true
# nothing is sent to the maker of the harness: telemetry, the feedback buttons that feed it, and the upload of session logs are off
- id: session-telemetry-otel
  disabled: true
- id: message-feedback
  disabled: true
- id: ui-message-feedback
  disabled: true
- id: command-feedback
  disabled: true
- id: session-log-deepseek
  disabled: true
# the first work folder is made inside the harness folder, not under the user's Documents
- id: workspace-controller
  config:
    documentsDirectory: '{documents}'
"""

MODEL_ENTRY = """          - id: {id}
            name: {name}
            contextWindow: {context}
            maxTokens: {max_output}
            input:
              - text
            reasoningEfforts:
              'off':
              low: low
              medium: medium
              high: xhigh
"""

PATCH_JA = """# the Japanese language pack: a small client plugin of YASAKANI, through the documented addLanguage / register interface
- insert:
    - id: yasakani-dsh-ja
      name: yasakani-dsh-ja
"""

T = {
    "banner": {"ja": "YASAKANI ハーネス {v}", "en": "YASAKANI harness {v}"},
    "what": {
        "ja": "この PC のモデルが動かす「エージェントの画面」をブラウザーで開きます。\n"
              "画面は DeepSeek Harness（DeepSeek 社が MIT ライセンスで公開しているソフト。{src}）を、そのまま使っています。\n"
              "日本語の表示は、YASAKANI が足した部品です。\n"
              "YASAKANI は DeepSeek 社とは関係のない個人の開発で、この組み合わせは公式の物ではありません。",
        "en": "This opens an agent screen in your browser, run by the model on this PC.\n"
              "The screen is DeepSeek Harness (open-source software by DeepSeek, MIT license; {src}), used unchanged;\n"
              "a Japanese language pack is added by YASAKANI.\n"
              "YASAKANI is an independent project, not affiliated with DeepSeek; this combination is not an official product."},
    "s_setup": {"ja": "準備を確かめています", "en": "Checking the setup"},
    "no_setup": {
        "ja": "エンジンとモデルが見つかりません。先に、クイックスタート（yasakani-quickstart の START.cmd）を最後まで済ませてください。\n"
              "探した場所：{p} と、その上のフォルダー、となりのフォルダー\n"
              "別の場所に入れたときは、START_HARNESS.cmd --home そのフォルダー で指定できます。",
        "en": "The engine and a model were not found. Run the quickstart (START.cmd of yasakani-quickstart) to the end first.\n"
              "Looked in: {p}, the folder above it and the folders next to it\n"
              "If it is somewhere else: START_HARNESS.cmd --home THAT_FOLDER"},
    "home_is": {"ja": "クイックスタートの場所：{p}", "en": "Quickstart folder: {p}"},
    "no_tools": {"ja": "ツール呼び出しに対応したパッケージ（harness_files の python フォルダー）が見つかりません。zip を展開し直してください。",
                 "en": "The package with tool calling (the folder python in harness_files) was not found. Unpack the zip again."},
    "m_quality": {"ja": "品質モデル", "en": "quality model"},
    "m_speed": {"ja": "最速モデル", "en": "fastest model"},
    "two_models": {
        "ja": "モデルが 2 つ入っています。{m}で始めます。画面のモデル名のボタンで、もう 1 つに切り替えられます（切り替えには 10 秒ほどかかります）。",
        "en": "Two models are installed. Starting with the {m}; the button with the model name on the screen switches to the other\n"
              "one (a switch takes about 10 seconds)."},
    "switching": {"ja": "[{t}] モデルを切り替えています：{a} → {b} …", "en": "[{t}] Switching the model: {a} -> {b} ..."},
    "switched": {"ja": "[{t}] 切り替わりました（{s:.0f} 秒）。", "en": "[{t}] Switched ({s:.0f} s)."},
    "switch_fail": {"ja": "[{t}] {b}を読み込めませんでした。{a}に戻します。記録：{p}",
                    "en": "[{t}] The {b} could not be loaded; going back to the {a}. Log: {p}"},
    "busy": {
        "ja": "YASAKANI のサーバーが、すでに動いています（ポート {n}）。モデルを 2 つ読み込むと GPU のメモリが足りなくなります。\n"
              "そちら（START.cmd の窓など）を閉じてから、もう一度起動してください。",
        "en": "A YASAKANI server is already running (port {n}). The GPU memory is not enough for two models.\n"
              "Close it (the START.cmd window, for example) and start this again."},
    "node_get": {
        "ja": "この PC には Node.js {n} 以上がありません（DeepSeek Harness を動かすのに使います）。\n"
              "Node.js {v} を公式サイトから取ってきて、このフォルダーの中（{p}）だけに置きます。PC 全体には入れません。\n  {u}",
        "en": "This PC has no Node.js {n} or later (DeepSeek Harness runs on it).\n"
              "Node.js {v} will be fetched from its official site and put into this folder only ({p}); nothing is installed system-wide.\n  {u}"},
    "node_q": {"ja": "取ってきますか？ [Y/n] ", "en": "Fetch it? [Y/n] "},
    "node_fail": {"ja": "Node.js を取ってこられませんでした：{e}\nhttps://nodejs.org から Node.js {n} 以上を入れてから、もう一度起動してください。",
                  "en": "Node.js could not be fetched: {e}\nInstall Node.js {n} or later from https://nodejs.org and start this again."},
    "node_sha_bad": {"ja": "取ってきたファイルのチェックサムが合いません（使いません）。\n  期待：{e}\n  実際：{a}",
                     "en": "The checksum of the fetched file does not match (it is not used).\n  expected: {e}\n  actual:   {a}"},
    "node_sha_ok": {"ja": "チェックサムは合っています。", "en": "The checksum matches."},
    "node_own": {"ja": "（このフォルダーの中の Node.js を使います）", "en": "(using the Node.js inside this folder)"},
    "node_is": {"ja": "Node.js {v}", "en": "Node.js {v}"},
    "no_dsh": {"ja": "DeepSeek Harness がまだ入っていません（{p}）。", "en": "DeepSeek Harness is not installed yet ({p})."},
    "dsh_get": {
        "ja": "DeepSeek Harness {v} を、公式の配布元（npm）から入れます。\n"
              "約 600 個のパッケージ、約 730 MB。このフォルダーの中（{p}）に入ります。1 個ずつチェックサムを確かめます。",
        "en": "DeepSeek Harness {v} will be installed from its official source (npm).\n"
              "About 600 packages, about 730 MB, into this folder ({p}). Every package is checked against its checksum."},
    "dsh_q": {"ja": "入れますか？ [Y/n] ", "en": "Install it? [Y/n] "},
    "dsh_no_files": {"ja": "入れるための一覧（dsh.package.json・dsh.package-lock.json）が、このフォルダーにありません。",
                     "en": "The lists for the installation (dsh.package.json, dsh.package-lock.json) are missing in this folder."},
    "dsh_no_npm": {"ja": "npm が見つかりません（Node.js と一緒に入ります）。", "en": "npm was not found (it comes with Node.js)."},
    "dsh_working": {"ja": "  入れています… {s:.0f} 秒", "en": "  installing ... {s:.0f} s"},
    "dsh_fail": {"ja": "入れられませんでした（終了コード {c}）。記録：{p}", "en": "The installation failed (exit code {c}). Log: {p}"},
    "dsh_done": {"ja": "入りました（{s:.0f} 秒）。", "en": "Installed ({s:.0f} s)."},
    "dsh_is": {"ja": "DeepSeek Harness {v}", "en": "DeepSeek Harness {v}"},
    "dsh_other": {"ja": "（試験したのは {t} です。別の版では動きが違うかもしれません）", "en": "(tested with {t}; another version may behave differently)"},
    "model_is": {"ja": "モデル：{m}", "en": "Model: {m}"},
    "s_server": {"ja": "モデルを読み込んでいます（1 分ほどかかることがあります）", "en": "Loading the model (this can take a minute)"},
    "server_fail": {"ja": "サーバーを起動できませんでした。記録：{p}", "en": "The server did not start. Log: {p}"},
    "server_ok": {"ja": "サーバーの準備ができました（{s:.0f} 秒）", "en": "The server is ready ({s:.0f} s)"},
    "s_dsh": {"ja": "エージェントの画面を起動しています", "en": "Starting the agent screen"},
    "screen_fail": {"ja": "画面を起動できませんでした。記録：{p}", "en": "The screen did not start. Log: {p}"},
    "open": {"ja": "ブラウザーが開きます。開かないときは、次のアドレスを開いてください：",
             "en": "Your browser opens now. If it does not, open this address:"},
    "open_manual": {"ja": "次のアドレスをブラウザーで開いてください：", "en": "Open this address in your browser:"},
    "notes": {
        "ja": "・画面の言葉は、ブラウザーの言語に合わせて日本語か英語になります（設定の「言語」で変えられます）。\n"
              "・モデルとのやりとりは、この PC の中だけで行われます。開発元への送信（評価ボタン、記録のアップロード）は切ってあります。\n"
              "  ただし、モデルに Web のページを取りに行かせると、その通信は外に出ます。\n"
              "・入力欄の下の「読み取り専用／作業フォルダーへの書き込み」で、モデルがファイルを書きかえてよいかを選べます。\n"
              "・考える量は、入力欄の下のモデル名のボタン →「思考量」で変えられます（最初は {th}。high がいちばん長く考える設定です）。\n"
              "・画面に出る速さ（tok/s）は DeepSeek Harness が計算した値で、YASAKANI の測定とは数え方が違います。\n"
              "  速さは、クイックスタートの「速さを測る」で確かめてください。",
        "en": "- The screen follows the language of your browser: Japanese or English (Settings, Language, changes it).\n"
              "- Everything between the screen and the model stays on this PC. Sending to the maker of the screen (feedback buttons,\n"
              "  upload of session logs) is switched off.  If you let the model fetch a web page, that request does leave the PC.\n"
              "- \"Read Only / Workspace Write\" under the message box decides whether the model may change files.\n"
              "- How much the model thinks is set with the button that shows the model name, under the message box (it starts at {th};\n"
              "  high is the model's longest setting).\n"
              "- The speed on the screen (tok/s) is computed by DeepSeek Harness and is not counted the way YASAKANI measures.\n"
              "  Use \"measure the speed\" of the quickstart for that."},
    "limit": {"ja": "・1 回の返事は {mo:,} トークンまで（考える分も含みます）。", "en": "- One reply may take up to {mo:,} tokens, thinking included."},
    "budget": {"ja": "・考える量が {tb:,} トークンを超えると、そこで切り上げて答えに進みます。",
               "en": "- A thinking longer than {tb:,} tokens is closed there and the model goes on to the answer."},
    "support": {"ja": "YASAKANI は個人が開発しています。役に立ったら、支援をお願いします：{u}",
                "en": "YASAKANI is developed by one person. If it is useful to you, please consider supporting it: {u}"},
    "stop_how": {"ja": "終わるときは、この窓で Ctrl+C を押してください。", "en": "To stop, press Ctrl+C in this window."},
    "stopping": {"ja": "止めています…", "en": "Stopping ..."},
    "stopped": {"ja": "止まりました。", "en": "Stopped."},
    "ended": {"ja": "{w} が終わりました（終了コード {c}）。記録：{p}", "en": "{w} ended (exit code {c}). Log: {p}"},
    "selftest": {"ja": "自己点検", "en": "Self-test"},
    "not_windows": {"ja": "Windows 専用です。", "en": "This runs on Windows only."},
    "bad_path": {"ja": "フォルダーの名前に日本語などが入っていると動きません：{p}", "en": "The folder name must be plain ASCII: {p}"},
}


class Ui:
    def __init__(self, lang):
        self.lang = "ja" if lang == "ja" else "en"

    def t(self, key, **kw):
        return T[key][self.lang].format(**kw)

    def say(self, key, **kw):
        print(self.t(key, **kw), flush=True)

    def step(self, i, n, key):
        print("\n[%d/%d] %s" % (i, n, self.t(key)), flush=True)


def ui_language():
    try:
        import ctypes
        return "ja" if (ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF) == 0x11 else "en"
    except Exception:  # noqa: BLE001
        return "en"


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(preferred, avoid=()):
    for p in [preferred] + list(range(preferred + 1, preferred + 40)):
        if p not in avoid and port_is_free(p):
            return p
    raise RuntimeError("no free port near %d" % preferred)


def get_json(port, path, timeout=5):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, json.loads(body.decode("utf-8")) if body[:1] in (b"{", b"[") else None
    except (OSError, ValueError, http.client.HTTPException):
        return None, None


def stop_tree(p):
    if p is None or p.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        p.kill()
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass


def node_version(node):
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", out)
    return (int(m.group(1)), out) if m else None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "yasakani-harness/" + HARNESS_VERSION})
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
            if pct != shown and (pct % 10 == 0 or pct < 0):
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


def own_node(hdir):
    """The Node.js this starter keeps inside the harness folder (None if it is not there)."""
    exe = os.path.join(hdir, "node", NODE_ZIP[:-4], "node.exe")
    return exe if os.path.isfile(exe) else None


def fetch_node(ui, hdir, yes, url=None):
    """Node.js from its official site into harness/node/, checked against the pinned checksum.  The path of node.exe, or None."""
    ui.say("node_get", n=NODE_MIN, v=NODE_VERSION, p=os.path.join(hdir, "node"), u=url or NODE_URL)
    if not yes:
        try:
            ans = input(ui.t("node_q")).strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("", "y", "yes"):
            return None
    os.makedirs(os.path.join(hdir, "downloads"), exist_ok=True)
    zpath = os.path.join(hdir, "downloads", NODE_ZIP)
    try:
        if not (os.path.isfile(zpath) and sha256_of(zpath) == NODE_SHA256):
            download(url or NODE_URL, zpath)
    except Exception as ex:  # noqa: BLE001  (network errors of any kind)
        ui.say("node_fail", e="%s: %s" % (type(ex).__name__, str(ex)[:160]), n=NODE_MIN)
        return None
    got = sha256_of(zpath)
    if got != NODE_SHA256:
        ui.say("node_sha_bad", e=NODE_SHA256, a=got)
        return None
    ui.say("node_sha_ok")
    safe_extract(zpath, os.path.join(hdir, "node"))
    return own_node(hdir)


def npm_command(node):
    """npm is a batch file on Windows; its JavaScript entry is run with node directly, so that nothing goes through cmd."""
    cli = os.path.join(os.path.dirname(node), "node_modules", "npm", "bin", "npm-cli.js")
    if os.path.isfile(cli):
        return [node, cli]
    npm = shutil.which("npm")
    if not npm:
        return None
    return ["cmd", "/c", npm] if sys.platform == "win32" else [npm]


def install_dsh(ui, here, hdir, dsh_dir, node, yes, log_path, cache=None):
    """`npm ci` in dsh_dir: exactly the packages of the lock file, each checked against its checksum.  True if installed."""
    pkg, lock = os.path.join(here, "dsh.package.json"), os.path.join(here, "dsh.package-lock.json")
    if not (os.path.isfile(pkg) and os.path.isfile(lock)):
        ui.say("dsh_no_files")
        return False
    npm = npm_command(node)
    if not npm:
        ui.say("dsh_no_npm")
        return False
    ui.say("dsh_get", v=DSH_VERSION, p=dsh_dir)
    if not yes:
        try:
            ans = input(ui.t("dsh_q")).strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("", "y", "yes"):
            return False
    os.makedirs(dsh_dir, exist_ok=True)
    shutil.copyfile(pkg, os.path.join(dsh_dir, "package.json"))
    shutil.copyfile(lock, os.path.join(dsh_dir, "package-lock.json"))
    t0 = time.time()
    with open(log_path, "ab") as out:
        out.write(("\n==== %s  npm ci\n" % time.strftime("%Y-%m-%d %H:%M:%S")).encode())
        out.flush()
        p = subprocess.Popen(npm + ["ci", "--no-audit", "--no-fund", "--cache", cache or os.path.join(hdir, "npm_cache")],
                             cwd=dsh_dir, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while True:
            try:
                rc = p.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                ui.say("dsh_working", s=time.time() - t0)
    if rc != 0:
        ui.say("dsh_fail", c=rc, p=log_path)
        return False
    ui.say("dsh_done", s=time.time() - t0)
    return True


def copy_plugin(src, dst):
    """Our own plugin folder, copied next to the home of the screen (files are overwritten, nothing is removed)."""
    if not os.path.isdir(src):
        return False
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isfile(s):
            try:
                same = os.path.isfile(d) and os.path.getsize(d) == os.path.getsize(s) and open(d, "rb").read() == open(s, "rb").read()
            except OSError:
                same = False
            if not same:
                shutil.copyfile(s, d)
    return os.path.isfile(os.path.join(dst, "package.json"))


def post_json(port, path, obj, timeout=900):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("POST", path, json.dumps(obj).encode("utf-8"), {"Content-Type": "application/json"})
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, json.loads(body.decode("utf-8")) if body[:1] in (b"{", b"[") else None
    except (OSError, ValueError, http.client.HTTPException):
        return None, None


def start_server(vpy, env, mdir, dll, port, log_path):
    """`yasakani serve` for one model folder.  Returns (process, health) once /health says ok, (process, None) if it does not."""
    out = open(log_path, "ab")
    out.write(("\n==== %s  port %d  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), port, os.path.basename(mdir.rstrip("\\/")))).encode())
    out.flush()
    env = dict(env)
    env["YASAKANI_MODEL_DIR"] = mdir
    p = subprocess.Popen([vpy, "-X", "utf8", "-u", "-m", "yasakani", "serve", "--model-dir", mdir, "--dll", dll, "--port", str(port)],
                         env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 900:
        status, health = get_json(port, "/health")
        if status == 200 and (health or {}).get("status") == "ok":
            return p, health
        if p.poll() is not None:
            break
        time.sleep(1)
    return p, None


class Switchboard:
    """Both models are installed, and one GPU holds one of them at a time.  The screen talks to this small relay on the port the
    model server would have; the model server listens on the next port.  A request that names the other model stops the server,
    starts it again with that model, and is then passed on.  Everything is passed on unchanged, one chat request at a time (the
    server answers one at a time anyway).  Like the server, the relay answers only requests that are addressed to this PC and do
    not come from a web page in a browser."""

    def __init__(self, ui, dirs, start, front_port, back_port, log_path, remember=None):
        self.ui, self.dirs, self.start, self.front, self.back = ui, dict(dirs), start, front_port, back_port
        self.log_path, self.remember = log_path, remember
        self.by_id = {MODEL_IDS[m]: m for m in self.dirs}
        self.lock = threading.Lock()                     # one chat request at a time, and no switch in the middle of one
        self.state = threading.Lock()                    # proc and switching are read together by the starter's loop
        self.proc, self.health, self.model = None, None, None
        self.switching = False
        self.httpd = None

    def load(self, model):
        proc, health = self.start(self.dirs[model], self.back)
        with self.state:
            self.proc, self.health = proc, health
        if health is None:
            stop_tree(proc)
            return False
        self.model = model
        return True

    def ended(self):
        """The exit code of the model server if it has ended by itself (not while a switch replaces it), else None."""
        with self.state:
            if self.switching or self.proc is None:
                return None
            return self.proc.poll()

    def ensure(self, model):
        """With the lock held: make `model` the loaded one.  False if it could not be loaded (the one before is loaded again)."""
        if model == self.model:
            return True
        old, t0 = self.model, time.time()
        names = dict(a=self.ui.t("m_" + old), b=self.ui.t("m_" + model))
        with self.state:
            self.switching = True
        try:
            self.ui.say("switching", t=time.strftime("%H:%M:%S"), **names)
            stop_tree(self.proc)
            if self.load(model):
                self.ui.say("switched", t=time.strftime("%H:%M:%S"), s=time.time() - t0)
                if self.remember:
                    self.remember(model)
                return True
            self.ui.say("switch_fail", t=time.strftime("%H:%M:%S"), p=self.log_path, **names)
            self.load(old)
            return False
        finally:
            with self.state:
                self.switching = False

    def serve(self):
        board = self

        class Relay(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _plain(self, code, message, kind):
                body = json.dumps({"error": {"message": message, "type": kind}}).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _pass(self, method):
                host = (self.headers.get("Host") or "").lower()
                if host not in ("127.0.0.1:%d" % board.front, "localhost:%d" % board.front) or self.headers.get("Origin"):
                    return self._plain(403, "forbidden", "forbidden")
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    n = 0
                raw = self.rfile.read(n) if n > 0 else b""
                want = None
                if method == "POST" and raw:
                    try:
                        body = json.loads(raw.decode("utf-8"))
                        want = board.by_id.get(body.get("model")) if isinstance(body, dict) else None
                    except ValueError:
                        pass
                chat = method == "POST"
                if chat:
                    board.lock.acquire()
                sent = False
                c = None
                try:
                    if chat and want and not board.ensure(want):
                        return self._plain(503, "the model could not be loaded", "unavailable")
                    c = http.client.HTTPConnection("127.0.0.1", board.back, timeout=1800)
                    c.request(method, self.path, raw if raw else None,
                              {"Content-Type": self.headers.get("Content-Type") or "application/json"} if raw else {})
                    r = c.getresponse()
                    self.send_response(r.status)
                    sent = True
                    self.send_header("Content-Type", r.getheader("Content-Type") or "application/json")
                    length = r.getheader("Content-Length")
                    if length is not None:
                        self.send_header("Content-Length", length)
                    else:                                   # a stream: it ends when the connection is closed
                        self.send_header("Connection", "close")
                        self.close_connection = True
                    self.end_headers()
                    while True:
                        b = r.read1(65536)
                        if not b:
                            break
                        self.wfile.write(b)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    self.close_connection = True            # the screen went away (a cancelled request): the server notices it too
                except (OSError, http.client.HTTPException):
                    self.close_connection = True
                    if not sent:
                        try:
                            self._plain(502, "the model server did not answer", "unavailable")
                        except OSError:
                            pass
                finally:
                    if c is not None:
                        c.close()
                    if chat:
                        board.lock.release()

            def do_GET(self):
                self._pass("GET")

            def do_POST(self):
                self._pass("POST")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.front), Relay)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
            except Exception:  # noqa: BLE001
                pass
        stop_tree(self.proc)


def stream_check(port, model_id):
    """A streamed answer through the relay arrives piece by piece, not all at the end.  Returns (name, ok, detail)."""
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        c.request("POST", "/v1/chat/completions", json.dumps({
            "model": model_id, "stream": True, "max_tokens": 200, "enable_thinking": False,
            "messages": [{"role": "user", "content": "Count from 1 to 40, one number per line."}]}).encode("utf-8"), {"Content-Type": "application/json"})
        r = c.getresponse()
        t0, times, data = time.time(), [], b""
        while True:
            b = r.read1(65536)
            if not b:
                break
            times.append(time.time() - t0)
            data += b
        c.close()
        n = data.count(b"data: ")
        ok = r.status == 200 and b"[DONE]" in data and len(times) >= 2 and n >= 10
        return ("stream through the relay", ok, "status %s, %d pieces in %d reads, first after %.2f s, last after %.2f s"
                % (r.status, n, len(times), times[0] if times else -1, times[-1] if times else -1))
    except Exception as ex:  # noqa: BLE001
        return ("stream through the relay", False, "%s: %s" % (type(ex).__name__, str(ex)[:120]))


def has_quickstart(d):
    return os.path.isfile(os.path.join(d, "quickstart_state.json")) and os.path.isfile(os.path.join(d, "venv", "Scripts", "python.exe"))


def find_quickstart(base):
    """The folder where START.cmd of the quickstart put the engine: the folder of START_HARNESS.cmd, the folder above it, or a
    folder inside or next to it.  When there are several, the one whose quickstart was used last.  None if there is none."""
    up = os.path.dirname(base)
    cands = [base, up]
    for d in (base, up):
        try:
            cands += [os.path.join(d, name) for name in sorted(os.listdir(d))]
        except OSError:
            pass
    seen, found = set(), []
    for d in cands:
        k = os.path.normcase(os.path.abspath(d))
        if k in seen:
            continue
        seen.add(k)
        try:
            if os.path.isdir(d) and has_quickstart(d):
                found.append(d)
        except OSError:
            pass
    if not found:
        return None
    if has_quickstart(base):
        return base

    def used(d):
        try:
            return os.path.getmtime(os.path.join(d, "quickstart_state.json"))
        except OSError:
            return 0.0
    return max(found, key=used)


def selftest(engine_port, web_url, model_id, budget=None):
    """The server answers with a tool call, and the web screen answers.  Returns a list of (name, ok, detail)."""
    out = []
    st, h = get_json(engine_port, "/health")
    out.append(("server /health", st == 200 and (h or {}).get("status") == "ok", json.dumps(h)[:160]))
    if budget is not None:
        out.append(("server thinking budget", (h or {}).get("thinking_budget") == budget, "%s (asked for %s)" % ((h or {}).get("thinking_budget"), budget)))
    body = json.dumps({
        "model": model_id, "stream": False, "max_tokens": 200, "temperature": 0, "enable_thinking": False,
        "messages": [{"role": "user", "content": "What is the weather in Kyoto? Use the tool."}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    }).encode("utf-8")
    try:
        c = http.client.HTTPConnection("127.0.0.1", engine_port, timeout=300)
        t0 = time.time()
        c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        d = json.loads(r.read().decode("utf-8"))
        c.close()
        ch = (d.get("choices") or [{}])[0]
        calls = (ch.get("message") or {}).get("tool_calls") or []
        name = ((calls[0].get("function") or {}).get("name")) if calls else None
        args = ((calls[0].get("function") or {}).get("arguments")) if calls else None
        ok = r.status == 200 and ch.get("finish_reason") == "tool_calls" and name == "get_weather" and "kyoto" in str(args).lower()
        out.append(("server tool call", ok, "status %s, finish %s, call %s %s, %.2f s" % (r.status, ch.get("finish_reason"), name, str(args)[:60], time.time() - t0)))
    except Exception as ex:  # noqa: BLE001
        out.append(("server tool call", False, "%s: %s" % (type(ex).__name__, str(ex)[:120])))
    m = re.match(r"http://127\.0\.0\.1:(\d+)(/.*)?$", web_url or "")
    if not m:
        out.append(("web screen", False, "no address"))
    else:
        try:
            # the address with the token is answered with the page or with a redirect to it: either says the screen is up
            c = http.client.HTTPConnection("127.0.0.1", int(m.group(1)), timeout=30)
            c.request("GET", m.group(2) or "/")
            r = c.getresponse()
            r.read()
            c.close()
            out.append(("web screen answers", r.status in (200, 301, 302, 303, 307, 308), "status %s" % r.status))
        except Exception as ex:  # noqa: BLE001
            out.append(("web screen answers", False, "%s: %s" % (type(ex).__name__, str(ex)[:120])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="YASAKANI harness starter")
    ap.add_argument("--home", help="the quickstart folder (default: looked for near START_HARNESS.cmd)")
    ap.add_argument("--model", choices=("quality", "speed"),
                    help="which installed model to start with (default: the one used last; with both installed, the screen can switch)")
    ap.add_argument("--lang", choices=("ja", "en"))
    ap.add_argument("--thinking", choices=("off", "low", "medium", "high"), default=THINKING,
                    help="how much a new session thinks before it answers (default: %(default)s; the screen can change it)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window; print the address only")
    ap.add_argument("--yes", action="store_true", help="answer yes to questions")
    ap.add_argument("--engine-port", type=int, default=8765)
    ap.add_argument("--web-port", type=int, default=3180)        # not 3080, the usual port of a DeepSeek Harness the user may run himself:
                                                                 # a browser keeps the settings of a page per address, and the two would share them
    for k in ("dsh-dir", "pypath", "model-dir", "dll", "python", "node", "stop-file", "npm-cache", "node-url", "harness-dir"):
        ap.add_argument("--" + k, help=argparse.SUPPRESS)
    ap.add_argument("--models", help=argparse.SUPPRESS)                      # for tests: quality=DIR,speed=DIR instead of home/models
    ap.add_argument("--max-output", type=int, default=MAX_OUTPUT, help=argparse.SUPPRESS)
    ap.add_argument("--thinking-budget", type=int, default=THINKING_BUDGET, help=argparse.SUPPRESS)       # 0 = no limit
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--minutes", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--any-os", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--own-node", action="store_true", help=argparse.SUPPRESS)      # behave as if the PC had no Node.js
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ui = Ui(a.lang or ui_language())
    print()
    ui.say("banner", v=HARNESS_VERSION)
    print(REPO_URL)
    print()
    ui.say("what", src=DSH_SOURCE)
    if sys.platform != "win32" and not a.any_os:
        ui.say("not_windows")
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here) if os.path.basename(here).lower() == "harness_files" else here     # the folder of START_HARNESS.cmd
    home = os.path.abspath(a.home) if a.home else (find_quickstart(base) or base)
    try:
        home.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=home)
        return 2
    n = 3

    # 1. what the quickstart installed, Node.js, DeepSeek Harness
    ui.step(1, n, "s_setup")
    st = {}
    try:
        with open(os.path.join(home, "quickstart_state.json"), encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        pass
    vpy = a.python or os.path.join(home, "venv", "Scripts", "python.exe")
    dll = a.dll or os.path.join(home, "engine", "yskn.dll")
    dirs = {}                                           # the installed models: name -> folder
    if a.models:
        for part in a.models.split(","):
            k, _, v = part.partition("=")
            if k in MODEL_IDS and v:
                dirs[k] = os.path.abspath(v)
    else:
        dirs = {m: os.path.join(home, "models", m) for m in ("quality", "speed")}
    dirs = {m: d for m, d in dirs.items() if os.path.isfile(os.path.join(d, "yasakani.json"))}
    if not (os.path.isfile(vpy) and os.path.isfile(dll) and (a.model_dir or dirs)):
        ui.say("no_setup", p=home if a.home else base)
        return 2
    ui.say("home_is", p=home)
    hdir = os.path.abspath(a.harness_dir) if a.harness_dir else os.path.join(base, "harness")
    try:
        hdir.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=hdir)
        return 2
    hstate_path = os.path.join(hdir, "harness_state.json")           # the model used last, when both are installed
    try:
        with open(hstate_path, encoding="utf-8") as f:
            hstate = json.load(f)
    except (OSError, ValueError):
        hstate = {}
    multi = len(dirs) == 2 and not a.model_dir          # both installed: the screen lists both and can switch
    model = a.model
    if not model and not a.model_dir:
        model = next((m for m in (hstate.get("model"), st.get("model"), "quality", "speed") if m in dirs), None)
    model = model or st.get("model") or "quality"
    mdir = os.path.abspath(a.model_dir) if a.model_dir else dirs.get(model, os.path.join(home, "models", model))
    if not os.path.isfile(os.path.join(mdir, "yasakani.json")):
        ui.say("no_setup", p=home if a.home else base)
        return 2
    status, health = get_json(a.engine_port, "/health", timeout=2)
    if status == 200 and isinstance(health, dict) and health.get("status") == "ok":
        ui.say("busy", n=a.engine_port)                 # two models do not fit into the memory of one GPU
        return 2
    env = dict(os.environ)
    env["YASAKANI_DLL"] = dll
    env["YASAKANI_MODEL_DIR"] = mdir
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["YASAKANI_THINKING_BUDGET"] = str(max(0, a.thinking_budget))
    # tool calling: the preview of the next package version next to this file comes before the installed 0.1.0 package
    pypath = a.pypath or (os.path.join(here, "python") if os.path.isfile(os.path.join(here, "python", "yasakani", "textio.py")) else None)
    if pypath:
        env["PYTHONPATH"] = pypath
    else:
        env.pop("PYTHONPATH", None)
    try:
        probe = subprocess.run([vpy, "-X", "utf8", "-c", "import yasakani, yasakani.textio as t; print(yasakani.__version__, hasattr(t, 'ToolCallFilter'))"],
                               env=env, capture_output=True, text=True, timeout=120)
        ver, has_tools = (probe.stdout.split() + ["?", "False"])[:2]
    except (OSError, subprocess.SubprocessError):
        ver, has_tools = "?", "False"
    print("yasakani %s" % ver)
    if has_tools != "True":
        ui.say("no_tools")
        return 2
    ui.say("model_is", m=os.path.basename(mdir.rstrip("\\/")) + "  (" + mdir + ")")
    if multi:
        ui.say("two_models", m=ui.t("m_" + model))
    # Node.js: the one named for a test, else one of NODE_MIN or newer on this PC, else the one kept inside the harness folder
    node = os.path.abspath(a.node) if a.node else shutil.which("node")
    nv = node_version(node) if node else None
    if not a.node and (not nv or nv[0] < NODE_MIN or a.own_node):
        node = own_node(hdir)
        if not node:
            node = fetch_node(ui, hdir, a.yes, a.node_url)
            if not node:
                return 2
        nv = node_version(node)
        if not nv:
            ui.say("node_fail", e="node.exe does not start", n=NODE_MIN)
            return 2
        ui.say("node_own")
    elif not nv:
        ui.say("node_fail", e="node does not start", n=NODE_MIN)
        return 2
    ui.say("node_is", v=nv[1])
    dsh_dir = os.path.abspath(a.dsh_dir) if a.dsh_dir else os.path.join(hdir, "dsh")
    pkg = os.path.join(dsh_dir, "node_modules", "@deepseek-ai", "dsh")
    bin_js = os.path.join(pkg, "lib", "bin.js")
    if not os.path.isfile(bin_js):
        ui.say("no_dsh", p=dsh_dir)
        os.makedirs(os.path.join(hdir, "logs"), exist_ok=True)
        if a.dsh_dir or not install_dsh(ui, here, hdir, dsh_dir, node, a.yes, os.path.join(hdir, "logs", "install.txt"), a.npm_cache):
            return 2
        if not os.path.isfile(bin_js):
            ui.say("dsh_fail", c=0, p=os.path.join(hdir, "logs", "install.txt"))
            return 2
    try:
        with open(os.path.join(pkg, "package.json"), encoding="utf-8") as f:
            dsh_ver = json.load(f).get("version") or "?"
    except (OSError, ValueError):
        dsh_ver = "?"
    ui.say("dsh_is", v=dsh_ver)
    if dsh_ver != DSH_VERSION:
        ui.say("dsh_other", t=DSH_VERSION)

    for d in ("home", "tmp", "workspace", "logs"):
        os.makedirs(os.path.join(hdir, d), exist_ok=True)
    # found by the screen from its home folder harness/home/profiles/web: node looks in the node_modules of the folders above
    has_ja = copy_plugin(os.path.join(here, "plugins", "yasakani-dsh-ja"), os.path.join(hdir, "node_modules", "yasakani-dsh-ja"))
    log = lambda name: os.path.join(hdir, "logs", name)                     # noqa: E731
    try:
        with open(os.path.join(mdir, "yasakani.json"), encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        info = {}
    model_id = "qwen3.8-27b"
    engine_port = pick_port(a.engine_port)
    back_port = pick_port(engine_port + 1) if multi else None             # with both models: the server itself, behind the relay
    web_port = pick_port(a.web_port, avoid=(engine_port, back_port))

    def remember(m):
        try:
            with open(hstate_path, "w", encoding="utf-8") as f:
                json.dump({"model": m}, f)
        except OSError:
            pass

    srv = dsh = board = None
    code = 0
    try:
        # 2. the model
        ui.step(2, n, "s_server")
        t0 = time.time()
        if multi:
            board = Switchboard(ui, dirs, lambda d, port: start_server(vpy, env, d, dll, port, log("server.txt")),
                                engine_port, back_port, log("server.txt"), remember)
            if not board.load(model):
                ui.say("server_fail", p=log("server.txt"))
                return 3
            board.serve()
            remember(model)
            health = board.health
        else:
            srv, health = start_server(vpy, env, mdir, dll, engine_port, log("server.txt"))
            if health is None:
                ui.say("server_fail", p=log("server.txt"))
                return 3
        ui.say("server_ok", s=time.time() - t0)
        model_id = MODEL_IDS[model] if multi else ((health or {}).get("model") or model_id)
        context = int((health or {}).get("max_context_tokens") or info.get("max_context_tokens") or 262144)
        # one reply may take up to max_output tokens.  DeepSeek Harness keeps that much room free, plus its headroom, and shortens
        # the conversation before a request would not fit: with 262,144 tokens of context that starts at 98,304 tokens of input
        max_output = max(4096, min(a.max_output, (context - COMPACT_HEADROOM) // 2))
        budget = int((health or {}).get("thinking_budget") or 0)

        # 3. the screen
        ui.step(3, n, "s_dsh")
        patch = os.path.join(hdir, "yasakani.patch.yml")
        with open(patch, "w", encoding="utf-8", newline="\n") as f:
            if multi:
                listed = [(MODEL_IDS[m], "Qwen3.8-27B %s (YASAKANI)" % ui.t("m_" + m)) for m in ("quality", "speed")]
            else:
                listed = [(model_id, "Qwen3.8-27B (YASAKANI, %s)" % model)]
            f.write(PATCH.format(engine_port=engine_port, model_id=model_id,
                                 models="".join(MODEL_ENTRY.format(id=i, name=name, context=context, max_output=max_output) for i, name in listed),
                                 # without the two lines, thinking is off (the default of the harness): "reasoningEffort" is for the
                                 # first session, "reasoning" is the level a model has when it is picked on the screen
                                 reasoning="" if a.thinking == "off" else "        reasoning: %s\n" % a.thinking,
                                 effort="" if a.thinking == "off" else "    reasoningEffort: %s\n" % a.thinking,
                                 documents=os.path.join(hdir, "workspace").replace("'", "''")))
            if has_ja:
                f.write(PATCH_JA)
        denv = dict(os.environ)
        denv["DSH_HOME"] = os.path.join(hdir, "home")
        denv["YASAKANI_API_KEY"] = "local"                                  # the local server asks for no key; the harness wants a value
        denv["TEMP"] = denv["TMP"] = os.path.join(hdir, "tmp")              # the harness leaves small folders behind at every start: keep them here
        denv.pop("DSH_TELEMETRY_OTLP_URL", None)
        cmd = [node, bin_js, "--profile", "web", "--patch", patch, "--port", str(web_port)]
        if a.no_browser or a.selftest:
            cmd.append("--no-open")
        dsh_log = open(log("screen.txt"), "ab")
        dsh_log.write(("\n==== %s  port %d\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), web_port)).encode())
        dsh_log.flush()
        dsh = subprocess.Popen(cmd, cwd=os.path.join(hdir, "workspace"), env=denv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL)
        found = {}

        def pump():
            for raw in iter(dsh.stdout.readline, b""):
                dsh_log.write(raw)
                dsh_log.flush()
                m = re.search(rb"dsh web:\s*(http://\S+)", raw)
                if m and "url" not in found:
                    found["url"] = m.group(1).decode("ascii", "replace")
        threading.Thread(target=pump, daemon=True).start()
        t0 = time.time()
        while "url" not in found and time.time() - t0 < 180:
            if dsh.poll() is not None:
                ui.say("screen_fail", p=log("screen.txt"))
                return 4
            time.sleep(0.2)
        if "url" not in found:
            ui.say("screen_fail", p=log("screen.txt"))
            return 4
        print()
        ui.say("open_manual" if (a.no_browser or a.selftest) else "open")
        print("  " + found["url"])
        print()
        ui.say("notes", th=a.thinking)
        ui.say("limit", mo=max_output)
        if budget > 0:
            ui.say("budget", tb=budget)
        print()
        ui.say("support", u=SUPPORT_URL)
        print()
        ui.say("stop_how")

        if a.selftest:
            print()
            ui.say("selftest")
            res = selftest(engine_port, found["url"], model_id, max(0, a.thinking_budget))
            if multi:                                   # to the other model and back, the way the screen asks for it
                for target in ([m for m in dirs if m != model][0], model):
                    t1 = time.time()
                    status, _ = post_json(engine_port, "/v1/chat/completions", {
                        "model": MODEL_IDS[target], "max_tokens": 8, "enable_thinking": False, "messages": [{"role": "user", "content": "hi"}]})
                    res.append(("switch to the %s model" % target, status == 200 and board.model == target,
                                "status %s, loaded %s, %.1f s" % (status, board.model, time.time() - t1)))
                res.append(stream_check(engine_port, model_id))
            for name, ok, detail in res:
                print("  %s  %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
            code = 0 if all(ok for _, ok, _ in res) else 5
            print("SELFTEST: %s" % ("ALL PASS" if code == 0 else "FAILED"), flush=True)
            if not a.minutes and not a.stop_file:
                return code

        end = time.time() + a.minutes * 60 if a.minutes else None
        beat = 0
        while True:
            rc = board.ended() if board is not None else srv.poll()
            if rc is not None:
                ui.say("ended", w="yasakani serve", c=rc, p=log("server.txt"))
                code = code or 3
                break
            if dsh.poll() is not None:
                ui.say("ended", w="DeepSeek Harness", c=dsh.returncode, p=log("screen.txt"))
                code = code or 4
                break
            if a.stop_file and os.path.exists(a.stop_file):
                break
            if end and time.time() > end:
                break
            time.sleep(0.5)
            beat += 1
            if (a.stop_file or end) and beat % 60 == 0:
                print(time.strftime("[%H:%M:%S]"), "running", flush=True)
        return code
    except KeyboardInterrupt:
        return code
    finally:
        print()
        ui.say("stopping")
        stop_tree(dsh)
        if board is not None:
            board.stop()
        stop_tree(srv)
        ui.say("stopped")


if __name__ == "__main__":
    sys.exit(main())
