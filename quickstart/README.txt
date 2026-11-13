YASAKANI quickstart 0.1.0
=========================

日本語
------
Windows 11 と NVIDIA GeForce RTX 5090 1 枚のパソコンで、YASAKANI をすぐに試すための起動ファイルです。

使い方
  1. この zip を、C:\yasakani のような英数字だけの場所に展開します。
  2. START.cmd をダブルクリックします。
  3. 画面の質問に番号で答えます（モデルは「品質モデル」か「最速モデル」）。

START.cmd がすること
  - このパソコンの確認（Python 3.10 以上、GPU、空き容量）
  - エンジンの zip を公開ページからダウンロードして、SHA-256 を確かめる
  - このフォルダーの中に Python の環境を作って、パッケージを入れる
  - モデル（約 13〜15 GB）を Hugging Face からダウンロードする
  - 点検（yasakani info・verify・selfcheck）
  - 速さを測る／先読みあり・なしを比べる／ブラウザーでチャットする／黒い画面でチャットする／サーバーを起動する、から選ぶ

知っておくこと
  - Python 3.10 以上が要ります（https://www.python.org/downloads/windows/）。
  - 空き容量は 20 GB ほど要ります。
  - 「先読みあり・なしを比べる」（lookahead_check.py）は、同じ 4 つの課題を、先読みなしと出荷時の設定で書かせます。
    1 回の計算の時間、1 回で確定したトークン数、速さ、書かれた文章が同じかどうかを表示します。
    技術の解説（docs/TECHNOLOGY.ja.md）の 1.3 の表と比べられます。
  - ブラウザーのチャット画面（chat.html）は、このパソコンの中だけで動きます（127.0.0.1）。会話はどこにも送られません。
  - 入れたものは、すべてこのフォルダーの中に置きます。パソコン全体には何も入れません。
    やめるときは、このフォルダーを消すだけです。
  - 配布ファイルにはコード署名がありません。Windows が警告を出したら、発行元と入手先を確かめたうえで実行してください。
  - もう一度 START.cmd をダブルクリックすると、済んだ手順は飛ばして続きから始まります。
  - 手で入れる手順と、くわしい説明は https://github.com/yasakani-dev/yasakani にあります。

English
-------
A starter for trying YASAKANI on a Windows 11 computer with one NVIDIA GeForce RTX 5090.

How to use
  1. Unpack this zip to a plain path such as C:\yasakani
  2. Double-click START.cmd
  3. Answer the questions with a number (the model is the "quality model" or the "fastest model").

What START.cmd does
  - checks this computer (Python 3.10 or newer, the GPU, free disk space)
  - downloads the engine zip from the release page and checks its SHA-256
  - makes a Python environment inside this folder and installs the package
  - downloads the model (about 13 to 15 GB) from Hugging Face
  - runs the checks (yasakani info, verify, selfcheck)
  - lets you measure the speed, compare look-ahead on and off, chat in your browser or in the terminal, or start the server

Good to know
  - Python 3.10 or newer is needed (https://www.python.org/downloads/windows/).
  - About 20 GB of free disk space is needed.
  - "Compare look-ahead on and off" (lookahead_check.py) writes the same four tasks without look-ahead and with the
    shipping settings. It prints the time of one pass, the tokens confirmed per pass, the speed, and whether the text is
    the same. Compare the result with table 1.3 of the technical notes (docs/TECHNOLOGY.md).
  - The browser chat (chat.html) runs on this computer only (127.0.0.1). The conversation is not sent anywhere.
  - Everything is kept inside this folder; nothing is installed system-wide. To remove it, delete the folder.
  - The files are not code-signed. If Windows shows a warning, check the publisher and the source before you run them.
  - Double-click START.cmd again any time: finished steps are skipped.
  - The manual steps and the full documentation are at https://github.com/yasakani-dev/yasakani

License: Apache License 2.0 (see the repository).
