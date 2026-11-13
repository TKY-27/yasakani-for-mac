YASAKANI harness 0.1.0
======================

日本語
------
このパソコンのモデルが動かす「エージェントの画面」を、ブラウザーで開くための起動ファイルです。
モデルが、作業フォルダーのファイルを読み書きしたり、コマンドを実行したりしながら、頼まれた作業を進めます。

画面は DeepSeek Harness（DeepSeek 社が MIT ライセンスで公開しているソフト。https://github.com/deepseek-ai/deepseek-harness）
を、書きかえずにそのまま使っています。YASAKANI が足しているのは、設定のファイルと、日本語の表示だけです。
YASAKANI は DeepSeek 社とは関係のない個人の開発で、この組み合わせは公式の物ではありません。

使い方
  1. 先に、クイックスタート（yasakani-quickstart-0.1.0.zip の START.cmd）を最後まで済ませます。
  2. この zip を、クイックスタートのフォルダーの中か、そのとなりに展開します（英数字だけの場所）。
  3. START_HARNESS.cmd をダブルクリックします。ブラウザーに画面が開きます。
  4. 終わるときは、黒い窓で Ctrl+C を押します。

START_HARNESS.cmd がすること
  - クイックスタートが入れたエンジンとモデルを探す（クイックスタートの中身は書きかえません）
  - Node.js 24 以上が無ければ、nodejs.org から取ってきて、このフォルダーの中だけに置く（約 37 MB。SHA-256 を確かめます）
  - はじめての起動のとき、DeepSeek Harness 0.2.1-rc.2 を npm から入れる（約 600 個のパッケージ、約 730 MB。
    harness_files\dsh.package-lock.json に書いた版とチェックサムのとおりに入れます）
  - モデルのサーバーを起動する（ツール呼び出しに対応した、次の版の試作パッケージ harness_files\python を使います）
  - DeepSeek Harness を起動して、ブラウザーで画面を開く

知っておくこと
  - 入れたものは、すべて「harness」フォルダーの中に置きます（DeepSeek Harness、Node.js、設定、会話の記録、作業フォルダー、
    一時ファイル）。パソコン全体には何も入れません。やめるときは、harness フォルダーを消すだけです。
  - すでに DeepSeek Harness を使っていても、その設定や会話には触りません（別のフォルダー、別のポート番号で動きます）。
  - モデルとのやりとりは、このパソコンの中だけで行われます（127.0.0.1）。DeepSeek 社への送信（利用状況の送信、評価ボタン、
    記録のアップロード）は、設定で切ってあります。画面の「ログイン」「フィードバック」は DeepSeek 社のサービス用で、使いません。
    ただし、モデルに Web のページを取りに行かせると、その通信は外に出ます。
  - モデルがファイルを書きかえてよいかどうかは、入力欄の下で選べます（読み取り専用／作業フォルダーへの書き込み）。
    モデルはコマンドも実行できます。大事なファイルのある場所を作業フォルダーにするときは、気をつけてください。
  - 最初は、いちばん長く考える設定（思考量 high）です。入力欄の下のモデル名のボタンから変えられます。
  - モデルを 2 つとも入れてあるときは、同じボタンで品質モデルと最速モデルを切り替えられます（GPU に載せかえるので、
    10 秒ほどかかります）。1 つだけのときは、そのモデルだけが出ます。
  - 1 回の返事は 98,304 トークンまで（考える分も含みます）。考える量が 49,152 トークンを超えると、そこで切り上げて答えに
    進みます。会話が長くなって入りきらなくなる前に、DeepSeek Harness が古い部分を要約して続けます。
  - ツール呼び出しは、まだ試作です。この版（0.1.0）のパッケージには入っていません。ハーネスは、同梱の試作パッケージ
    （yasakani 0.2.0.dev0）でサーバーを起動します。クイックスタートが入れたものは変えません。
  - 画面に出る速さ（tok/s）は DeepSeek Harness が計算した値で、YASAKANI の測定とは数え方が違います。
    速さは、クイックスタートの「速さを測る」で確かめてください。
  - DeepSeek Harness は試験版（0.2.1-rc.2）に固定しています。配布元（npm）から消えると、入れられなくなります。
  - 配布ファイルにはコード署名がありません。Windows が警告を出したら、発行元と入手先を確かめたうえで実行してください。

確かめたこと（2026-11-13、Windows 11 と RTX 5090 のパソコン 1 台）
  会話、ファイルの読み書きと編集、コマンドの実行、考えている様子の表示、書きかえの許可をたずねる動き、長い会話（文脈を
  小さくした試験で、入りきらなくなったあとの要約と続行）、日本語の表示、2 つのモデルとその切り替え、考える量の上限、
  Node.js の自動取得。通信を見張った試験では、このパソコンの外への接続は 0 件でした。

確かめていないこと
  Python も Node.js も入っていないまっさらな Windows、RTX 5090 以外の GPU、1 回のツールの結果だけで文脈より大きくなる場合。

ライセンス
  YASAKANI の部分（START_HARNESS.cmd、harness_files の harness.py と python フォルダー）は Apache License 2.0 です。
  日本語の表示（harness_files\plugins\yasakani-dsh-ja）は、DeepSeek Harness の画面の文を訳したもので、MIT License です
  （そのフォルダーの LICENSE）。DeepSeek Harness と Node.js は同梱していません。それぞれのライセンスで、配布元から入ります。

English
-------
A starter that opens an agent screen in your browser, run by the model on this computer. The model reads and writes files
in a work folder and runs commands to do what you ask.

The screen is DeepSeek Harness (open-source software by DeepSeek, MIT license; https://github.com/deepseek-ai/deepseek-harness),
used unchanged. YASAKANI adds a settings file and a Japanese language pack, nothing else.
YASAKANI is an independent project, not affiliated with DeepSeek; this combination is not an official product.

How to use
  1. Run the quickstart first (START.cmd of yasakani-quickstart-0.1.0.zip), to the end.
  2. Unpack this zip into the quickstart folder or next to it (a plain ASCII path).
  3. Double-click START_HARNESS.cmd. The screen opens in your browser.
  4. To stop, press Ctrl+C in the black window.

What START_HARNESS.cmd does
  - finds the engine and the model the quickstart installed (nothing of the quickstart is changed)
  - if there is no Node.js 24 or newer, fetches Node.js from nodejs.org into this folder only (about 37 MB, SHA-256 checked)
  - on first use, installs DeepSeek Harness 0.2.1-rc.2 from npm (about 600 packages, about 730 MB; exactly the versions and
    checksums listed in harness_files\dsh.package-lock.json)
  - starts the model server (with the preview of the next Python package in harness_files\python, which has tool calling)
  - starts DeepSeek Harness and opens the screen in your browser

Good to know
  - Everything is kept in the folder "harness" (DeepSeek Harness, Node.js, settings, session records, the work folder, temporary
    files). Nothing is installed system-wide. To remove it, delete the folder "harness".
  - A DeepSeek Harness you already use is not touched: this one has its own folders and its own port.
  - Everything between the screen and the model stays on this computer (127.0.0.1). Sending to DeepSeek (telemetry, feedback
    buttons, upload of session logs) is switched off in the settings. "Sign in" and "Feedback" on the screen belong to
    DeepSeek's services and are not used. If you let the model fetch a web page, that request does leave the computer.
  - "Read Only / Workspace Write" under the message box decides whether the model may change files. The model can also run
    commands. Take care when the work folder holds files that matter.
  - A new session starts with the model's longest thinking setting (high). The button with the model name, under the message
    box, changes it.
  - With both models installed, the same button switches between the quality model and the fastest model (the GPU is
    reloaded, which takes about 10 seconds). With one model installed, only that one is listed.
  - One reply may take up to 98,304 tokens, thinking included. A thinking longer than 49,152 tokens is closed there and the
    model goes on to the answer. Before a long conversation stops fitting, DeepSeek Harness summarizes the older part and
    continues.
  - Tool calling is a preview. It is not in the 0.1.0 package of this release; the harness starts the server with the preview
    package that comes with it (yasakani 0.2.0.dev0). What the quickstart installed is not changed.
  - The speed on the screen (tok/s) is computed by DeepSeek Harness and is not counted the way YASAKANI measures. Use "measure
    the speed" of the quickstart for that.
  - DeepSeek Harness is pinned to a release candidate (0.2.1-rc.2). If it is removed from npm, it can no longer be installed.
  - The files are not code-signed. If Windows shows a warning, check the publisher and the source before you run them.

What was checked (2026-11-13, one PC with Windows 11 and an RTX 5090)
  Chat, reading, writing and editing files, running commands, the live view of the thinking, asking before a change, a long
  conversation (in a test with a small context: summary and continuation after it stopped fitting), the Japanese screen, both
  models and the switch between them, the limit on the thinking, fetching Node.js. In the tests that watched the network
  there were 0 connections to outside this computer.

What was not checked
  A fresh Windows with neither Python nor Node.js, GPUs other than the RTX 5090, a single tool result larger than the context.

License
  The YASAKANI parts (START_HARNESS.cmd, harness.py and the folder python in harness_files): Apache License 2.0.
  The Japanese language pack (harness_files\plugins\yasakani-dsh-ja) is a translation of the screen texts of DeepSeek Harness:
  MIT License (see LICENSE in that folder). DeepSeek Harness and Node.js are not included; they are installed from their own
  sources under their own licenses.
