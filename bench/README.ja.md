# bench/：測定の記録

English: [README.md](README.md)

このフォルダーには、YASAKANI と、比べた相手のエンジンの測定の記録が入っています。どれも、NVIDIA GeForce RTX 5090 を 1 枚積んだ同じパソコンで測りました。今回の公開の速度表と品質の表（[`README.ja.md`](../README.ja.md)、[`docs/MEASUREMENTS.ja.md`](../docs/MEASUREMENTS.ja.md)）のもとは `20261108/`、公開物の確認のもとは `20261110/` です。古い 2 つのフォルダーは、以前のビルドでの以前の比較で、そのまま残しています。

| フォルダー | 日付 | 内容 |
|---|---|---|
| [`20261108/`](20261108/) | 2026 年 11 月 6 日・8 日 | **今回の公開の速度表**：公開した 2 つのモデル（最速・品質）と、MegaCapybara 1.39・mkyt v0.3.2・ignis v0.9.6・NInfer・vLLM 0.31.0・SGLang 0.5.22・llama.cpp b11700・TensorFold 0.6.6 を、15 個の課題で、思考なしと思考ありで比べた記録。起動コマンド、実際に使ったスクリプト、全実行の記録と答えの本文、サーバーのログ、記録から表を作り直す `scripts/tables.py`。MegaCapybara の思考なしは `20261103/` の実行、SGLang・vLLM・llama.cpp の起動スクリプトは `20261101/` のものです。品質の表の記録（`quality/`）、長い文脈での KV5 の記録（`long_context/`）、録画した実演の証拠の一式（`demo/`）も入れています。 |
| [`20261110/`](20261110/) | 2026 年 11 月 10 日 | **今回の公開物での確認**（配布する `yskn.dll`・Python パッケージ・2 つのモデルのフォルダー）：`yasakani verify`・`selfcheck`・`bench`、高速化のオン・オフでの出力、同じ依頼のくり返し、GPU そのものの確認。 |
| [`20261101/`](20261101/) | 2026 年 11 月 1 日 | 以前の比較。YASAKANI（公開版）と SGLang 0.5.22・vLLM 0.31.0・llama.cpp b11700 の比較（10 種類のテスト）。起動コマンド、プロンプト、測定スクリプト、全実行の記録。同じ日に行った YASAKANI だけの確認（元のモデルとの品質比較、AIME 2025 の集計、長い文脈、高速化のオン・オフで出力が同じかの確認、セルフチェック）も入れています。 |
| [`20261103/`](20261103/) | 2026 年 11 月 3 日 | 以前の比較。YASAKANI（標準版と高速版のモデルファイル）と MegaCapybara 1.39（Tiny モデル）の、同じパソコンでの比較。NInfer が公開している 12 問、自作のテスト 3 種類、動画に使った 1 回ずつの記録。生成の記録（すべての答えの本文と速さ）、サーバーのログ、クライアントのログ、GPU の状態。 |

測った値は記録のままです。公開のために置き換え・削除したのは、パスとユーザー名、メールアドレス 1 件、エンジン設定の一覧 2 件、スクリプト内のコメント、他者のプロンプトの本文だけです。`20261108/` のジョブのログは、時刻と GPU の行だけにしました（「公開のための変更」に詳細）。`SHA256SUMS.txt` に全ファイルのハッシュがあります。

## パソコン

* GPU：NVIDIA GeForce RTX 5090 1 枚、32 GB（32,607 MiB）。ドライバー 616.92（CUDA 13.4）。11 月 3 日に nvidia-smi で記録した電力の上限は 600 W（既定値）、最大クロックは SM 3,090 MHz・メモリ 14,001 MHz です（[`20261103/gpu/`](20261103/gpu/)）。11 月 1 日の GPU の値（温度・SM クロック・メモリクロック・電力・使用メモリ・総メモリ・使用率）は記録の中にあります（`A_pub.json` の `gpu_start`・`gpu_before`・`gpu_after`、ドライバーログの `GPU [メモリ MiB, 使用率 %]`）。
* CPU：AMD Ryzen 9 9950X3D2。メモリ：96 GB。OS：Windows 11 Pro。
* Windows でそのまま動かしたもの：YASAKANI、llama.cpp、MegaCapybara。11 月 6 日の mkyt と ignis も Windows 用の配布物をそのまま動かしました。SGLang と vLLM には公式の Windows 版がないため、同じパソコンの WSL2（Ubuntu 24.04）で動かしました。11 月 6 日の NInfer と TensorFold も WSL2 です。Linux で直接動かした場合との差は測っていません。
* 11 月 6 日・8 日：ジョブの始めに nvidia-smi で電力の上限（600 W）、最大クロック（3,090 MHz・14,001 MHz）、ドライバー（616.92）、温度を、終わりに電力の上限・温度・使用メモリを記録しています（[`20261108/logs/jobs/`](20261108/logs/jobs/)）。
* 測定のクライアント：Windows の Python 3.14.4。リクエストは 1 件ずつ、サーバーも 1 つずつで、ほかに GPU を使うものはありません。

## 速さ

速さ（トークン/秒）＝（出力トークン数 − 1）÷（答えの最初のトークンから最後のトークンまでの時間）。最初のトークンが出るまでの時間（プロンプトを処理している時間）は含みません。出力トークン数は、サーバーが返す `completion_tokens` の値です（プロセス内の実行では、生成したトークンの数）。

* ストリーミングの実行（ほかのエンジンの全実行、11 月 1 日の YASAKANI、11 月 3 日のサーバー経由の全実行）：最初と最後のチャンクがクライアントに届いた時刻。
* プロセス内の実行（11 月 3 日の YASAKANI の 12 問、`pub_corpus_*.py`）：サーバーを使わず Python API から呼び出し、エンジン自身が測った生成時間（`gen_ms`、最初から最後の生成トークンまで）。「わかっている制約」も見てください。
* 11 月 6 日・8 日（`20261108/`）：どのエンジンのどの実行も、サーバー経由のストリーミングです。時間は最初のチャンクから最後のチャンクまでで、思考ありでは、思考のチャンクも答えのチャンクと同じに数えます。

## 20261108/：今回の公開の速度表

[`docs/MEASUREMENTS.ja.md`](../docs/MEASUREMENTS.ja.md) の 4 つの速度表は、このフォルダーの記録から [`20261108/scripts/tables.py`](20261108/scripts/tables.py) で作れます。その出力を [`20261108/tables/`](20261108/tables/)（`SPEED_TABLES.md`・`speed_tables.json`）に入れています。

### エンジン・版・モデルファイル

| エンジン | 版・動かした場所 | モデルファイルと設定 | 測った日 |
|---|---|---|---|
| YASAKANI | 0.1.0.dev0、11 月 7 日にビルドした `yskn.dll`（SHA-256 `0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb`、今回公開する DLL）、Windows | 最速モデル：`qwen38_27b_speed.yskn`（10,311,344,128 バイト、SHA-256 の先頭 `3ba17a76871a2b5e`）。品質モデル：`qwen38_27b_quality.yskn`（12,365,242,368 バイト、SHA-256 の先頭 `6493983cf9539182`）。どちらも `bdraft2_k8.yskn`・`bdraft2_k8.shead`・`kvq_u5.bin` と一緒に使います。`yasakani serve` の既定の設定 | 11 月 8 日 |
| MegaCapybara | 1.39、Windows x64 | Tiny モデル、公式の 1 会話用プリセット（KV 6/4）、ループ防止は切、`20261103/` と同じ | 思考なし：11 月 3 日、思考あり：11 月 6 日 |
| mkyt | `mkyt/qwen3-27b-5090` v0.3.2、Windows 用の配布物 | 既定の設定：`juzbox/Qwen3.8-27B-NVFP4-GDN`（リビジョン `3e68640`）と `incoai/Qwen3.8-27B-DFlash2`（リビジョン `015e795`）、KV は NVFP4、プレフィックスキャッシュあり | 11 月 6 日 |
| ignis | v0.9.6、Windows 用の配布物 | `qwen3_8_27b_nvfp4full-v2.ninfer`（19,406,942,468 バイト）、文脈 512K、DFlash 2 で下書き 7 トークン | 11 月 6 日 |
| NInfer | 11 月 6 日時点の `master`、コミット `a754fcf3bf524494aa927c2f54e0cbfb58f569ab`（11 月 2 日付）を WSL2 でビルド（`logs/build/ninfer_master_build.log`） | 公式の NVFP4 の配布物 `qwen3_8_27b_nvfp4.ninfer`（23,719,715,844 バイト、SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`）、DFlash 2 で下書き 7 トークン、optimized の proposal head、KV は FP8、プレフィックスの再利用なし（全部の引数は `launch/serve_ninfer_master.sh`） | 11 月 6 日 |
| vLLM | 0.31.0、WSL2 | `20261101/` と同じサーバー・モデルファイル・起動スクリプト：`vl_dflash_lm4`・`vl_mtp_lm4`・`vl_dspark_lm4` | 11 月 6 日 |
| SGLang | 0.5.22、WSL2 | 同じ：`sg_dflash_nv`・`sg_dflash_lm4+norc`・`sg_mtp_nv`・`sg_dspark_lm4` | 11 月 6 日 |
| llama.cpp | b11700、公式の Windows CUDA ビルド | 同じ：`llama_dflq4`・`llama_mtp2`（文脈 16,384） | 11 月 6 日 |
| TensorFold | 0.6.6、WSL2 | MLX 4-bit のモデル（`TensorFold/Qwen3.8-27B-MLX-4bit`）と下書きモデル `z-lab/Qwen3.8-27B-DFlash2`、CUDA バックエンド、プロンプトキャッシュなし、文脈 69,632 | 11 月 6 日 |

コマンドは [`20261108/launch/`](20261108/launch/) にあります。`yasakani_windows.cmd`・`megacapybara_windows.cmd`・`mkyt_windows.cmd`・`ignis_windows.cmd` は、実際に動かしたサーバーとクライアントのコマンドを公開用に書き出したもの、`serve_ninfer_master.sh`・`ninfer_master_build.sh`・`serve_ninfer.sh`・`serve_tensorfold.sh` は WSL2 で実際に動かしたスクリプトです。SGLang・vLLM・llama.cpp は、`scripts/run_wf_n12.py` が [`20261101/launch/`](20261101/launch/) のスクリプトで起動しました。

### テスト

どのエンジンにも同じリクエストを送りました（OpenAI 互換のチャット API、ストリーミング）。リクエストは 1 件ずつ、サーバーも 1 つずつです。

* NInfer が公開している 12 問（コード：CUDA・Python・TypeScript、物語：英語の推理小説・中国語の SF・中国語の会話、翻訳：中→英・英→中・Markdown、構造化：CSV・JSONL・SQL）。答えは 4,096 トークンまで。名前と SHA-256 は [`20261103/prompts/ninfer_cases.json`](20261103/prompts/ninfer_cases.json) にあります。ファイルは NInfer のリポジトリ（github.com/Neroued/ninfer）のもので、ここには写していません。
* 自作の課題 3 つ（[`20261103/prompts/cases.json`](20261103/prompts/cases.json)）。答えは 8,192 トークンまで。`edit_code`（「コード編集」：CPython の `Lib/textwrap.py` の `dedent` の中の変数 1 つの名前を変え、ファイル全体を返す。プロンプトは約 4,800 トークン）、`edit_jsonl`（「JSONL 編集」：60 行の JSONL のうち 1 件を変え、全行を返す）、`jsonl_orders_8k`（「長い JSONL」：注文の記録を 300 件書く。8,192 トークンで打ち切り）。
* 思考なし：greedy（temperature 0）、`enable_thinking` は false、どの課題も 2 回。
* 思考あり：temperature 1.0、top-p 0.95、top-k 20、min-p 0、`enable_thinking` は true、seed は回の番号、上限は同じ（思考も上限に数えます）、どの課題も 2 回。YASAKANI の 2 つのモデルだけ、さらに 3 回（seed 2・3・4、`results_seeds_2_4/`）測り、合わせて 5 回です。これらのパラメーターを断ったサーバーはありません（断られたらクライアントのログに「HTTP 400」と残ります）。
* どのクライアントも、最初に短いウォームアップを 2 回送ります。`run_wf_n12.py` と `think_more.py` が起動したサーバー（mkyt・ignis・MegaCapybara 以外）には、その前にドライバーからも 2 回送っています。

### 結果のフォルダー

[`20261108/results/`](20261108/results/) の下のフォルダー 1 つが、サーバーを 1 回起動して測った分です。

| フォルダー | エンジンと設定 | 思考 |
|---|---|---|
| `n12_ysk_speed_v8`・`n12t_ysk_speed_v8` | YASAKANI、最速モデル | なし、あり（0・1 回目） |
| `n12_ysk_quality_v8`・`n12t_ysk_quality_v8` | YASAKANI、品質モデル | なし、あり（0・1 回目） |
| `../results_seeds_2_4/n12t_ysk_speed_v8`・`n12t_ysk_quality_v8` | YASAKANI、両方のモデル | あり（2・3・4 回目） |
| `megacapybara_tiny_think` | MegaCapybara 1.39 Tiny | あり（なしは `20261103/results/megacapybara_kv64` と `own_tests`） |
| `mkyt_v032`・`mkyt_v032_think` | mkyt v0.3.2 | なし、あり |
| `ignis_v096`・`ignis_v096_think` | ignis v0.9.6 | なし、あり |
| `n12_ninfer_master`・`n12t_ninfer_master` | NInfer、表に使ったビルド | なし、あり |
| `n12_ninfer` | NInfer、10 月のビルド（リビジョン `81c8ce09`）。表には使っていません | なし |
| `n12_vl_dflash_lm4`・`n12_vl_mtp_lm4`・`n12_vl_dspark_lm4`、`n12t_vl_dflash_lm4`・`n12t_vl_mtp_lm4` | vLLM 0.31.0 の DFlash 2・MTP・DSpark | なし、あり（DFlash 2 と MTP） |
| `n12_sg_dflash_nv`・`n12_sg_dflash_lm4_norc`・`n12_sg_mtp_nv`・`n12_sg_dspark_lm4`、`n12t_sg_dflash_nv`・`n12t_sg_mtp_nv` | SGLang 0.5.22 の DFlash 2（公式の 5090 レシピ、radix cache なし）・MTP・DSpark | なし、あり（DFlash 2 と MTP） |
| `n12_llama_dflq4`・`n12_llama_mtp2`、`n12t_llama_dflq4`・`n12t_llama_mtp2` | llama.cpp b11700 の DFlash 2・MTP | なし、あり |
| `n12_tensorfold`・`n12t_tensorfold` | TensorFold 0.6.6 | なし、あり |

フォルダーの中身：

| パス | 内容 |
|---|---|
| `n12/runs_<ラベル>.jsonl` | 12 問の実行 1 回が 1 行：プロンプトと出力のトークン数、最初のチャンクまでの時間、生成時間、速さ（`decode_tok_s`）、終わり方、サーバーの usage、時刻 |
| `n12/texts/<問題>_r<回>.txt` | 届いたままの本文（思考ありでは、思考のあとに答え） |
| `n12/summary_<ラベル>.txt` | クライアントが出した表 |
| `own/runs_<ラベル>.jsonl` | 自作の課題 3 つの実行 1 回が 1 行。編集の 2 課題は、正解との比較（`similarity`・`exact`・`diff_lines`）も |
| `own/<ラベル>_<課題>_r<回>.txt`・`.chunks.json` | 答え（思考は含みません）と、全チャンクの到着時刻（送信からのミリ秒。思考のチャンクには `"r"` の印） |
| `client_n12.log`・`client_own.log` | 2 つのクライアントの画面出力 |
| `server.log`・`warmup_*.json` | サーバーの出力とドライバーのウォームアップ（`run_wf_n12.py` が起動していないサーバーの分。ほかのサーバーのログは `logs/server/`） |

`20261108/` のほかのフォルダー：

| パス | 内容 |
|---|---|
| `scripts/run_wf_n12.py` | ドライバー：サーバーを 1 つ起動し、応答を待ち、ウォームアップを 2 回送り、2 つのクライアントを動かし、サーバーを止めます（プラン `n12`・`n12b`・`n12c`・`n12t`・`speedv8`・`speedv8t`・`qualv8`・`qualv8t`）。10 月 18 日と 11 月 1 日のプランも入ったままです |
| `scripts/oai_bench_client.py`・`custom_bench.py` | クライアント（12 問用と、自作の課題 3 つ用）。`20261103/scripts/` のクライアントに、11 月 6 日に思考・追加のパラメーター・seed の指定を足したものです。その指定を使わなければ、送るリクエストは同じです |
| `scripts/think_more.py`・`oai_bench_client_seedoff.py`・`custom_bench_seedoff.py` | YASAKANI の思考ありの追加 3 回（クライアントに `--seed-offset` を足したもの） |
| `scripts/tables.py` | 記録から表を作ります（下記） |
| `tables/` | `tables.py` の出力 |
| `logs/driver/` | `run_wf_n12.py` のログ（`n12.log`＝SGLang・vLLM・llama.cpp の思考なし、`n12b.log`＝NInfer の 10 月のビルドと TensorFold、`n12c.log`＝NInfer、`n12t.log`＝思考あり、`n12*_speed_v8.log`・`n12*_quality_v8.log`＝YASAKANI）と `think_more.py` のログ |
| `logs/server/` | サーバーのログ（`server_<n12 か n12t>_<設定>.log`） |
| `logs/jobs/` | ジョブごとの手順の時刻と nvidia-smi の行 |
| `logs/build/` | NInfer のビルドのログ |

### 表の作り方

`python 20261108/scripts/tables.py` が、`runs_*.jsonl` と保存した本文を読んで 4 つの表を出します。規則はスクリプトの先頭に書いてあります。まとめると：1 回の速さは記録の `decode_tok_s`。`tool_calls` で終わった回は数えません。「思考なし」は分野の中の回の平均で、README の表ではさらに、保存した本文が「頼んでいないツール呼び出し」か「くり返し」になっている回を、全エンジン同じ規則で除きます。「思考あり」は課題ごとに一番速い回を取り、分野の中で平均します。設定が複数あるエンジンは、列ごとに一番大きい値を出します。スクリプトは、除いた回、設定ごとの値、除く前の表、平均で作った表も出します。

`tables.py` は、このフォルダーの配置に合わせて書いたものです。公表した表は、同じ規則の作業用スクリプトで作りました。両方を比べ、560 個の値（10 行・7 列・4 通りの集計・思考なしとあり）がすべて一致することを確かめています。

### 順番と時刻（日本時間）

* 11 月 3 日：MegaCapybara の思考なし 14:23〜14:32、その自作の課題 16:02〜16:05（`20261103/` を参照）。
* 11 月 6 日、思考なし：SGLang・vLLM・llama.cpp 09:11〜10:56（順番は `logs/driver/n12.log`）、ignis 11:01〜11:08、NInfer（10 月のビルド）と TensorFold 11:09〜11:27、mkyt 11:47〜11:52、NInfer 12:17〜12:25。
* 11 月 6 日、思考あり：SGLang・vLLM・llama.cpp・NInfer・TensorFold 12:42〜15:05（`logs/driver/n12t.log`）、ignis 15:06〜15:17、mkyt 15:17〜15:26、MegaCapybara 15:26〜15:34。
* 11 月 8 日：YASAKANI の最速モデルの思考なし 12:05〜12:08、思考あり 12:08〜12:14。品質モデルの思考なし 12:14〜12:18、思考あり 12:18〜12:25。思考ありの追加 3 回は、最速モデル 13:49〜13:57、品質モデル 13:58〜14:07。

### 長い文脈での KV5（`20261108/long_context/`）

[`docs/MEASUREMENTS.ja.md`](../docs/MEASUREMENTS.ja.md) の「最速モデルのKV5（独自圧縮のキー・バリューキャッシュ）」のもとの記録です。11 月 8 日 12:25〜12:50（同じジョブの速度の測定のあと）に測りました。131,072 トークンの決まった列 4 本をエンジンに読ませ、文脈長 2,048〜131,072 のそれぞれで、各列の最後の 1,023 位置を BF16 の基準と比べました。KV5（今回公開する `yskn.dll` と最速モデル）と、FP16 の KV キャッシュ（同じ重みを使った研究用のビルド）の 2 通りです。GPU のメモリは、文脈を 2,048〜262,144 トークン分確保したあとの nvidia-smi の値です。

| ファイル | 内容 |
|---|---|
| `kv5_summary.json`・`KV5_TABLES.md` | 文脈長ごとの KL・95% 区間・第 1 候補の一致率・パープレキシティ（2 つの KV 形式）、同じ位置での KV5 と FP16 の差、確保した文脈ごとのメモリ（FP16・INT8・KV5）と、1 トークンあたりの KV キャッシュのバイト数 |
| `kv5_score.log`・`fp16_score.log` | 採点のログ（列と文脈長ごとに 1 行） |
| `kv5_memory.log`・`int8_memory.log`・`fp16_memory.log` | 文脈を確保してエンジンを開く前後の nvidia-smi の値 |

要約と表は、作業用の要約の数字を、公開用のラベルにしたものです。ログは、測定スクリプトが出した行だけにしています。4 本の列（自分たちの思考文と答えをつないだもの）、BF16 の基準、研究用のビルドを動かす測定スクリプトは入れていません。INT8 の列はメモリだけで、品質は測っていません。

### 思考文の品質比較（`20261108/quality/`）

[`docs/MEASUREMENTS.ja.md`](../docs/MEASUREMENTS.ja.md) の「思考文の品質比較」のもとの記録です。11 月 8 日の 15:02〜18:05 に測りました。自分たちの思考文 20 本を、2,048 トークンの窓としてどのエンジンにも読ませ、各窓の最後の 1,023 位置（合計 20,460 位置）で、次のトークンの分布を元の BF16 モデル（基準）の分布と比べました。決まった文を読ませるだけで、自由な生成はしません。比べているのは、各位置での本体のモデル自身の分布で、高速化は数字に関係しません。llama.cpp・NInfer・ignis・TensorFold で使った道具は、本体のモデルだけを動かします。YASAKANI・SGLang・vLLM・mkyt は、補助のモデルなしで起動しました。MegaCapybara のサーバーは、速度表と同じモデルと KV の設定で、補助のモデルも読み込んで起動しました。どの要求も出すのは 1 トークンだけで、サーバーのログでは 20,460 回の要求のどれにも高速化の回はありません。読み出した値は本体のモデルのものです。

| ファイル | 内容 |
|---|---|
| `QUALITY_TABLE.md`・`quality_table.json` | 測った 15 構成を KL の順に並べた表、2 つの容量帯の中での比、追加の記録 2 つ。`scripts/quality_table.py` が `scores/` から作ります |
| `scores/*.json` | 構成ごとに 1 ファイル：容量、KV キャッシュ、KL、第 1 候補の一致率、パープレキシティ、分布の取り方（`how`）、作業用の記録の数字（`record`） |
| `logs/llamacpp_*.log` | GGUF 6 つの llama-perplexity の出力 |
| `logs/sglang_nvidia_nvfp4_*.log`・`logs/vllm_nvfp4_lmhead4_*.log` | サーバーのログと、log 確率を取って採点した実行のログ |
| `launch/*.sh` | この測定での SGLang と vLLM のサーバーの起動 |
| `scripts/kld_base.py` | 基準のファイル形式と、1 位置の計算式（llama.cpp の perplexity ツールの式を numpy で書いたもの）。自己テストつき |
| `scripts/llama_kld.py`・`scripts/gguf_main_size.py` | llama.cpp の行：llama-perplexity の呼び出しと出力の読み取り、補助の層を除いた GGUF の大きさ |
| `scripts/api_topk_dump.py`・`scripts/topk_score.py` | SGLang と vLLM の行：API で各位置の上位 100 トークンを取るスクリプトと、その採点 |
| `scripts/quality_table.py` | 点数のファイルから表を作る |

文書の表にあるのは、15 構成のうち 13 です。残りの 2 つ、Unsloth UD-IQ3_S（12.04 GB、KL 0.0311）と Unsloth UD-Q2_K_XL（9.83 GB、KL 0.0759）も同じ方法で測りました。この 2 つは、ここにだけ入れています。

エンジンごとの分布の取り方：

| 構成 | KL の範囲 | 取り方 |
|---|---|---|
| YASAKANI（2 モデル） | 語彙全体 | FP16 の KV キャッシュを使い、語彙全体のロジットを返せる研究用のビルド。検証の経路で、各窓を 16 位置ずつ読ませました。今回公開する `yskn.dll` は使っていません。重みは、公開するモデルファイルと同じ中身です。 |
| llama.cpp（GGUF 6 つ） | 語彙全体 | b11700 の `llama-perplexity` をそのまま使用（`--kl-divergence`、flash attention あり、KV キャッシュ f16）。 |
| SGLang・vLLM | 上位 100 トークン（下限） | サーバーの API が、プロンプトの各位置について上位 100 トークンとその log 確率を返します。KL は、この 100 トークンと「残り全部」をまとめた 1 つとで計算しました。語彙全体の KL より大きくなることはありません。 |
| mkyt | 上位 64 トークン（下限） | サーバーは log 確率を返しませんが、エンジン自身の Python ライブラリ（`forward_topk`）は返します。100 トークンは断られ、64 トークンは受け付けられました。先に 20 トークンで測った値も、追加の記録として残しています。 |
| NInfer | 語彙全体 | ソースの写し（master `a754fcf3`）の perplexity ツールにロジットの書き出しを足して、ビルドしました。速度表で使ったインストール済みのビルドは変えていません。 |
| ignis | 語彙全体 | ignis のソースには、サーバーと同じ経路で各位置のロジットを書き出すサンプル `span_logits` があります。配布物には入っていないので、v0.9.6 のタグからビルドしました。 |
| TensorFold | 語彙全体 | サーバーはプロンプトの確率を返さないので、インストールしたパッケージの CUDA エンジン（重みの読み込み・プロンプトの処理・出力層）を Python から直接動かしました。 |
| MegaCapybara | 語彙全体 | サーバーは確率を返さず、ソースも公開されていません。動いているサーバーのプロセスに動的計測ツール（Frida）をつなぎ、エンジンが GPU で計算した最後の位置のロジットを、1 位置につき 1 回の要求で読み出しました。プログラムのファイルは変えていません。 |

この表の下の 5 行のために書いたプログラムは、ほかの人のソフトウェアの中に手を入れるもの（写しを変える、配布されていない部分をビルドする、ライブラリや内部のモジュールを直接呼ぶ、プロセスにつなぐ）なので、入れていません。YASAKANI の行の採点スクリプトも、研究用のビルドを動かすものなので入れていません。思考文 20 本、基準のファイル（10.2 GB。思考文のトークン番号が入っています）、位置ごとのファイル（これにもトークン番号が入っています）も入れていません。そのため、このフォルダーだけでは表を測り直せません。

測定のときに行った確認（数字は点数のファイルにあります）：

* YASAKANI：返ってきたロジットから計算した負の対数尤度と、エンジン自身の値との差は最大 1.8e-6。最も確からしいトークンは、すべての位置で同じでした。研究用のビルドが読んだモデルファイルの大きさは、公開するモデルファイルとバイト単位で同じです（12,365,242,368 バイトと 10,311,344,128 バイト。11 月 10 日に比べました）。
* 下限の出方：同じ計算を YASAKANI 自身のロジットにあてはめると、上位 100 トークンの KL は、語彙全体の KL より 2.0%（品質モデル）・1.5%（最速モデル）小さく出ました。上位 20 では 7.0%・6.8% です。この差を使って、ほかのエンジンの値を補正することはしていません。
* SGLang・vLLM：20,460 位置のすべてで、実際のトークンの log 確率は上位 100 の中にあり、サーバーがそのトークンについて返す値と一致しました。
* NInfer：書き出したロジットと、エンジン自身が出した実際のトークンの log 確率との差は最大 7.5e-6。NInfer が計算する出力の行は 248,320 のうち 248,077 で、残りには非常に低い値を入れました。基準がその残りに置いている確率は、どの位置でも 0.003% 以下です。
* MegaCapybara：読み出した分布で最も確からしいトークンは、文字として比べられる 20,289 位置のすべてで、サーバーが実際に返したトークンと同じでした。先に通常の API だけで測った第 1 候補の一致率は 92.69%（比べられる 20,288 位置）、今回は 92.80%（全 20,460 位置）です。

時刻（11 月 8 日、日本時間）：llama.cpp の GSQ-RCO IQ3_S 15:02〜15:03、vLLM 15:20〜15:24、SGLang 15:24〜15:28、mkyt（20 トークン）15:28、YASAKANI の最速モデル 15:29〜15:33・品質モデル 15:33〜15:37、NInfer 15:37〜15:40、TensorFold 15:40〜15:43、ignis 15:43〜15:47、MegaCapybara（API）15:47〜15:56、mkyt（64 トークン）15:57、llama.cpp の GSQ-RCO IQ3_XXS 16:06、Unsloth の 4 ファイル 16:48〜17:09、MegaCapybara（語彙全体）17:34〜18:05。電力の上限は 600 W。ジョブの開始時に、ほかのプログラムが GPU のメモリを 0.3〜1.8 GB 使っていました。結果を出す前に止まった実行（YASAKANI 15:03〜15:19、mkyt 15:19、MegaCapybara 15:40・17:27・17:30）は、点数を残していません。

### 11 月 8 日の連続画面録画（`20261108/demo/`）

[`docs/MEASUREMENTS.ja.md`](../docs/MEASUREMENTS.ja.md) の「11月8日の最速モデルによる連続画面録画」のコード編集の実演の、証拠の一式です。その日の夜にまとめたものを 1 バイトも変えずに入れています。`manifest.json` に、全ファイル・モデルファイル・`yskn.dll` の SHA-256 があります。別の短い文で 2 回ウォームアップした新しいサーバーのプロセス（最速モデル）に、速度表のコード編集の課題を 1 回だけ送り（思考なし、temperature 0、出力の上限 8,192）、別のコマンドプロンプトの時計と一緒に画面を録画しました。最初の出力まで 1.281498 秒、最後の出力まで 5.555668 秒、通信の完了まで 5.556058 秒、4,712 トークン、1,102.2 tok/s。

| ファイル | 内容 |
|---|---|
| `README_JA.md` | この一式の説明（日本語） |
| `manifest.json` | 条件、測定値、動画の時刻の検査、全ファイルの SHA-256 |
| `request.json`・`original.py`・`output.py` | 実際に送った API の要求、編集するファイル、生成されたファイル |
| `arrival_events.jsonl`・`capture.json` | 受信したイベントと到着時刻、その集計 |
| `clock_console.py`・`clock-ticks.jsonl`・`clock-signal.json`・`clock-check.json` | コマンドプロンプトに出した時計、そのログ、要求の開始・終了の印、時計のログの整合の確認 |
| `verify_output.py`・`verify.json` | 生成されたコードの検証（構文と、原本との動作の比較 17,701 件）と結果 |
| `video-check.json`・`poster.png` | フレームの時刻の検査、動画の 1 コマ |
| `reproduce.py` | 同じ要求を送るクライアント |

動画そのもの（`fast_live_uncut.mp4`、11,391,224 バイト、SHA-256 `035b60e0c33f22f1f71312e1a1aab3ab383fc0a229dca4af7ac23a91eb4b6339`）は、このフォルダーには入れず、リリースと一緒に公開します。画面の録画と CPU でのエンコードをしながらの測定なので、速度表の値の代わりにはなりません。同じ夜の先の録画（19:01）は保存してありますが、ここには入れていません。

## 20261110/：今回の公開物での確認

[`docs/MEASUREMENTS.ja.md`](../docs/MEASUREMENTS.ja.md) の「この公開版での確認（2026 年 11 月 10 日）」のもとの記録です。配布する `yskn.dll`（SHA-256 `0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb`）、Python パッケージ 0.1.0.dev0、2 つのモデルのフォルダーを、測定に使ったパソコンで確かめました。ファイル名の「speed」は最速モデル、「quality」は品質モデルです。

| パス | 内容 |
|---|---|
| `checks/selfcheck/` | `yasakani info`、`yasakani verify`（全ファイルが `yasakani.json` と一致）、`yasakani selfcheck`（基準を書き出し、別のプロセスで、既定の文脈長と 16,384 で再現。`*_sc_fresh.log` は、`selfcheck.json` をモデルのフォルダーに入れたあと、別のプロセスでさらに 6 回）、`yasakani bench`（`*_bench.json`） |
| `checks/on_off/` | `onoff_eq.py`（[`20261101/scripts/`](20261101/scripts/) にあります）：34 種類の入力、greedy と seed 1234 の 62 通り。それぞれを、既定の設定、同じ設定でもう一度、高速化を 3 通りに切った設定で生成し、トークンの番号を比べます。最速モデルは 174 件の比較のうち 171 件、品質モデルは 170 件が一致（JSON の `summary` に、違った場合の最初の位置も） |
| `checks/repeat/` | `repeat_probe.py`：1 つのプロセスの 1 つのエンジンで、同じ依頼を 8 回または 12 回、256 トークンずつ生成します。`speed_repeat_probe`・`quality_repeat_probe` は全部の計画（セルフチェックの 3 問を 30 回ずつくり返す分も含む）。ほかは最速モデルでの短い計画（下書きなしの依頼 4 つと既定の設定の依頼 1 つを 12 回ずつ）：`env_A_base`・`env_F_base2`（何も変えない）、`ctl_G_ctx262144`・`ctl_H_ctx16384`（文脈長）、`ctl_I_warmup`（コマンドのウォームアップのあと）、`ctl_J_nohash`（状態のハッシュを読まない）、`clk_engine_off4`（GPU の状態を 1 秒ごとに記録、`.csv`）、`quiet_1`〜`quiet_4`（GPU が 30 秒使われていないのを待ってから）、`share_S1_bursts`・`share_S2_nopause`（別のプロセスが GPU を使っている間。`bg_S*.log` がそのプロセス） |
| `checks/gpu/` | `gpu_consistency.py`：同じ GPU で、PyTorch 2.10.0+cu128 に同じ計算をくり返させ、ビット単位で比べます。YASAKANI のコードは使いません。`gpu_consistency`（行列の積、要素ごとの計算、64 ビット整数、12 GiB のメモリの読み直し）、`_bursty`（短い計算と、毎回のホストへの転送）、`_power_a`・`_power_b`（2 つのプロセスを同時に、600 W の上限で）。`clk_torch_*.csv` は 1 秒ごとの GPU の状態。合わせて 542,278 回で、不一致は 0 |
| `scripts/` | `repeat_probe.py`、`quiet_gate.py`（GPU が使われなくなるのを待つ）、`gpu_consistency.py`、`gpu_bg_load.py`（GPU を分け合う確認の、もう一方のプロセス）、`gpu_sample.py`、記録からくり返しの確認を数える `repeat_summary.py` |
| `launch/checks_windows.cmd` | 実行したコマンド |
| `logs/jobs/` | ジョブごとの手順の時刻と nvidia-smi の行 |

`python 20261110/scripts/repeat_summary.py` が下の数を出します。「違う回」とは、トークンの番号の並びが、同じ依頼のくり返しの中でいちばん多い出力と違う回です。違う回も、途中の 1 か所までは同じです（位置はスクリプトが出します）。

| 条件 | 既定の設定 | 高速化なし（1 回の計算で 1 トークン） | ブロックの下書きだけ切 | セルフチェックの 3 問 |
|---|---|---|---|---|
| GPU をほかに使うものがない：開始前の利用率の中央値 0%（`quiet_1`〜`quiet_4`） | 48 回中 0 | 192 回中 0 | | |
| ほかのプログラムが GPU を使っている：画面の描画が動いていて、開始前の利用率の中央値 5%（9 件の記録、2 モデル） | 190 回中 1 | 432 回中 21 | 16 回中 0 | 180 回中 0 |
| 別のプロセスが GPU を短く何度も使う（`share_S1_bursts`） | 12 回中 0 | 48 回中 2 | | |
| 別のプロセスが GPU を休みなく使う（`share_S2_nopause`） | 12 回中 0 | 48 回中 8 | | |

時刻（日本時間）：info〜bench 07:42〜07:44、オン・オフ 07:55〜08:11、くり返し（全部の計画）08:23〜08:35、短い計画 08:38〜08:50 と 09:04〜09:13、PyTorch 09:15〜09:19、GPU を分け合う確認 09:21〜09:26、PyTorch（短い計算）と GPU を記録しながらのくり返し 09:27〜09:33、PyTorch（電力の上限）09:35〜09:39、GPU が静かなときの実行 09:42〜09:50。

この確認の制約：違いの原因は、まだ特定できていません。「GPU をほかに使うものがない」は、開始前の 30 秒間、nvidia-smi の利用率が 0〜1% だったという意味で、実行中にデスクトップが何をしていたかは記録していません。短い計画は、最速モデルでの 256 トークンの依頼 5 つです。ほかの依頼、もっと長い出力、サーバー経由は、この方法では試していません。同じ日の午前に行った、内部の切り替えや以前のビルドを使った調査用の実行は、入れていません。

## 20261101/：YASAKANI・SGLang・vLLM・llama.cpp

### エンジン・版・モデルファイル

| エンジン | 版・動かした場所 | モデルファイル |
|---|---|---|
| YASAKANI | 公開版 0.1.0.dev0、Windows。`yskn.dll` 26,184,192 バイト、SHA-256 `1fa3a25362f150c885ab7eb23fa32e672fbeebf7820c5b19a56ee452227a3718` | `qwen38_27b_e3x.yskn`（12,365,242,368 バイト）と `bdraft2_k8.yskn`・`bdraft2_k8.shead`・`kvq_u5.bin` |
| SGLang | 0.5.22、WSL2 | 本体：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`（LMHead4）または `nvidia/Qwen3.8-27B-NVFP4`（SGLang 公式の RTX 5090 レシピ） |
| vLLM | 0.31.0、WSL2 | 本体：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`（LMHead4） |
| llama.cpp | b11700、公式の Windows CUDA ビルド（`llama-server.exe`） | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` の `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |

ほかのエンジンの下書き用モデル：DFlash 2 ＝ `z-lab/Qwen3.8-27B-DFlash2`（SGLang・vLLM）または `ggml-org/Qwen3.8-27B-GGUF` の `dflash-Qwen3.8-27B-Q4_0.gguf`（llama.cpp）。MTP ＝ モデルに付いている複数トークン予測の層。DSpark ＝ `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`（SGLang）または `RedHatAI/Qwen3.8-27B-speculator.dspark`（vLLM）。
TensorRT-LLM 1.3.0rc29 は 2026 年 10 月 18 日に試しただけで、RTX 5090 では速い設定が動きませんでした。そのときのファイルとメモを `launch/tensorrt_llm_tried_2026-10-18/` と `launch/notes_2026-10-18.txt` に残しています。11 月 1 日には動かしていません。

### ほかのエンジンの設定

起動スクリプトは [`20261101/launch/`](20261101/launch/) にあります（フォルダーの指定は `common.sh`。スクリプトは実際に使ったもので、違うのは `common.sh` のフォルダーの行だけです）。設定名は記録の中の名前です。

| 設定（記録の名前） | エンジン | 起動 | 表での表記 |
|---|---|---|---|
| `sg_dflash_nv` | SGLang | `sg_dflash_nv.sh` | DFlash 2 (official 5090 recipe) |
| `sg_dflash_lm4+norc` | SGLang | `sg_dflash_lm4.sh` | DFlash 2 (radix cache off) |
| `sg_mtp_nv` | SGLang | `sg_mtp_nv.sh` | MTP |
| `sg_dspark_lm4` | SGLang | `sg_dspark_lm4.sh` | DSpark |
| `vl_dflash_lm4` | vLLM | `vl_dflash_lm4.sh` | DFlash 2 |
| `vl_mtp_lm4` | vLLM | `vl_mtp_lm4.sh` | MTP |
| `vl_dspark_lm4` | vLLM | `vl_dspark_lm4.sh` | DSpark |
| `llama_mtp2` | llama.cpp | `llama_cpp_windows.txt` の 1 つ目（`--spec-type draft-mtp --spec-draft-n-max 2`） | MTP |
| `llama_dflq4` | llama.cpp | `llama_cpp_windows.txt` の 2 つ目（`--spec-type draft-dflash --spec-draft-n-max 15`） | DFlash 2 |

長い入力（41,261 トークンのプロンプト）は、同じスクリプトに引数を足して文脈を広げて測りました。`+48k` ＝ SGLang は `--context-length 49152`（DSpark はさらに `--max-total-tokens 49152`）、vLLM は `--max-model-len 49152`、llama.cpp は `-c 16384` の代わりに `-c 49152`。`+nopc` ＝ vLLM の `--no-enable-prefix-caching`。すべての設定の正確なコマンドは `scripts/run_wf.py`（`CFGS`）にあり、SGLang と vLLM のサーバーは `logs/server/` のログの先頭近くに引数を出力しています。

### テスト

どのエンジンにも同じリクエストを送りました（OpenAI 互換のチャット API、ストリーミング）。presence penalty はすべて 0 です。プロンプトのファイルは [`20261101/prompts/`](20261101/prompts/) にあり、実際に送ったファイルそのものです（`A_pub.json` の `prompt_sha256` と同じ SHA-256）。

| テスト | プロンプト | 設定 | 答えの長さ | 回数 |
|---|---|---|---|---|
| chat・explain・code・math・json | 各 1 問（`chat.txt` など） | 思考あり。temperature 1.0、top-p 0.95、top-k 20、min-p 0 | 2,048 トークン | 3 |
| long | `long.txt`（41,261 トークン） | 同上 | 2,048 トークン | 3 |
| prose3 | `prose3_*.txt`（3 問） | 同上 | 1,024 トークン | 各 3 |
| japanese | `japanese_q1.txt`〜`q5` | 思考なし。temperature 0.7、top-p 0.8、top-k 20 | 2,048 トークン | 各 1 |
| ninfer_prose | NInfer の物語と翻訳の 6 問（`ninfer_cases.json` に名前と SHA-256 だけ） | 思考なし。temperature 0.6、top-p 0.95、top-k 20 | 4,096 トークン | 各 2 |
| sglang_random_8k | `sglang_random_8k_1.txt`〜`_5`（ランダムな 8,192 トークンの入力） | 思考なし。greedy、EOS を無視 | 1,024 トークン | 各 1 |

「思考ありの 6 種類」は最初の 6 つ（chat〜long）です。100 トークン以上書いた実行だけを数えます。

### テストごとの数字の決め方

* ほかのエンジン：設定ごとに、数えた実行の中央値。中央値がいちばん高い設定を、そのエンジンの数字にします（テストごとに一番速い設定）。
* YASAKANI：10 種類のテストを続けて 2 回測り、回ごとに中央値を出し、速い方の回を載せます。2 回分とも `results/yasakani/A_pub.json` と `A_pub_runs.csv` にあります。
* 下の表は、`scripts/headline.py` が記録から計算し直した結果です。

| テスト | YASAKANI 1 回目 / 2 回目（載せる値） | SGLang | vLLM | llama.cpp | YASAKANI ÷ ほかで最速 |
|---|---|---|---|---|---|
| chat | 253.1 / 236.2（253.1） | 123.8 | 106.6 | 119.0 | 2.05 |
| explain | 284.4 / 272.8（284.4） | 156.4 | 150.2 | 127.3 | 1.82 |
| code | 381.4 / 408.7（408.7） | 192.1 | 166.2 | 150.6 | 2.13 |
| math | 424.4 / 422.7（424.4） | 240.7 | 180.9 | 183.6 | 1.76 |
| json | 346.9 / 330.2（346.9） | 197.4 | 173.3 | 163.8 | 1.76 |
| long | 383.5 / 439.2（439.2） | 221.9 | 139.7 | 183.9 | 1.98 |
| prose3 | 248.7 / 265.5（265.5） | 132.0 | 115.3 | 120.0 | 2.01 |
| japanese | 209.3 / 199.5（209.3） | 119.1 | 126.1 | 140.0 | 1.49 |
| ninfer_prose | 340.8 / 330.2（340.8） | 164.1 | 147.0 | 163.1 | 2.08 |
| sglang_random_8k | 572.7 / 567.3（572.7） | 162.4 | 163.8 | 192.6 | 2.97 |

思考ありの 6 種類の中央値：YASAKANI 377.8、SGLang 194.8、vLLM 158.2、llama.cpp 157.2（トークン/秒）。6 種類の倍率の中央値は 1.90 です。ここの倍率は丸める前の中央値から計算しています。japanese はこの計算では 1.4945 で、表の小数 1 桁の値（209.3 ÷ 140.0）からは 1.495 になり、ほかの資料ではこれを 1.50 と表記しています。

### 順番と時刻（2026 年 11 月 1 日、日本時間）

YASAKANI の速さ 5:21〜5:32（`logs/driver/A_pub.log`）、ループ防止を入れた YASAKANI 5:32〜5:34、llama.cpp 7:42〜8:05（`X1_llama*.log`）、SGLang と vLLM 8:05〜10:19（`X2_sg_vl*.log`）。`checks/selfcheck/` のセルフチェックは速さの測定の直前（5:20）、`checks/` のほかの確認は 5:34〜7:36 に行いました。

### ファイル

| パス | 内容 |
|---|---|
| `launch/` | SGLang・vLLM の起動スクリプト（`common.sh`、`sg_*.sh`、`vl_*.sh`）、llama.cpp のコマンド、YASAKANI のコマンド（`yasakani_windows.cmd`）、2026-10-18 のメモ、2026-10-18 に試した TensorRT-LLM のファイル |
| `prompts/` | プロンプトのファイル。NInfer のプロンプトは名前と SHA-256 だけ |
| `scripts/run_wf.py` | ほかのエンジンの測定を回したスクリプト。サーバーを 1 つ起動し、ウォームアップを 2 回送り、`capture_stream.py` でテストを流し、サーバーを止める（プラン `x1`・`x1long`・`x2`・`x2long`） |
| `scripts/capture_stream.py` | ストリーミングのリクエストを 1 件送り、チャンクごとの到着時刻を記録して速さを計算する |
| `scripts/wf_to_summary.py` | `runs_others.jsonl` から `results_summary_others.csv` と `results_runs_others.csv` を作る |
| `scripts/a_speed.py` | YASAKANI の速さの測定（`A_pub.json`） |
| `scripts/onoff_eq.py` | 高速化のオン・オフで出力が同じかの確認（`checks/on_off_equality/`） |
| `scripts/headline.py` | 上の表を記録から計算し直す |
| `results/others/runs_others.jsonl` | ほかのエンジンの全実行（441 回、数えたのは 438 回） |
| `results/others/results_runs_others.csv`・`results_summary_others.csv` | 同じ内容の表。summary はエンジンとテストごとの一番速い設定 |
| `results/yasakani/A_pub.json`・`A_pub_runs.csv` | YASAKANI の全実行、2 回分（98 回） |
| `results/yasakani/A_pub_guard.json`・`A_pub_guard_runs.csv` | ループ防止を入れた、思考ありの 6 種類（18 回） |
| `captures/x1`・`x1l`・`x2`・`x2l` | ほかのエンジンの実行 1 回ごとのファイル。リクエストの設定、届いたチャンクと到着時刻、usage、速さ（`x1` ＝ llama.cpp、`x2` ＝ SGLang と vLLM、`l` ＝ 長い入力）。`warmup_*` はウォームアップ |
| `logs/driver/` | `run_wf.py` のログ（`X1_*`・`X2_*`）、表への変換のログ（`X_convert.log`）、`a_speed.py` のログ（`A_pub*.log`） |
| `logs/server/` | 各設定のサーバーのログ（`server_<プラン>_<設定>.log`）と YASAKANI のサーバーのログ（`A_pub*.serve.log`） |
| `checks/quality/` | 元の BF16 モデルとの比較（第 1 候補の一致率、パープレキシティ、KL ダイバージェンス）。WikiText-2・C4・FineWeb-Edu の 61,380 か所。`kv5` ＝ 標準設定、`kv8` ＝ 8 bit KV の対照 |
| `checks/aime2025/report.json` | AIME 2025 の 30 問を 3 回：30・29・30 問正解（集計だけ） |
| `checks/long_context/` | 文脈 262,144 トークンの設定で約 25 万トークンを入力したときのメモリ（nvidia-smi）と時間 |
| `checks/on_off_equality/` | 高速化のオンとオフでの出力の比較：174 組すべて一致 |
| `checks/selfcheck/` | 公開版の `yasakani info`・`yasakani bench`・`yasakani selfcheck` とサーバーの起動 |

## 20261103/：YASAKANI と MegaCapybara 1.39

### エンジンとモデルファイル

| エンジン | 版 | モデルファイル |
|---|---|---|
| YASAKANI | 2026-11-03 の公開候補ビルド、0.1.0.dev0（`yskn.dll` の SHA-256 は `d86fe2d0b2e0f27c` で始まる。11 月 1 日の基準に対するセルフチェックに合格：`logs/client/yasakani_selfcheck_vs_nov1.log`） | 「std」：標準のモデルフォルダー（12.4 GB、11 月 1 日と同じファイル名）。「fast」：本体を 3 bit だけにしたもの（11.7 GB）、補助のファイルは同じ |
| MegaCapybara | 1.39、Windows x64（`megacapybara.exe` の SHA-256 `a36600f036e827f881b933a6260dd70977e36ea3f8bbfac463796aacc718d327`） | Tiny `qwen3.8-27b-tiny.mcapy`（16,594,763,776 バイト、SHA-256 `5bec1521fd7f1c99a5150ae37dca06122fbf5745f1b7baa167cf20cfd7c41118`）と下書き用の `qwen3.8-27b-dflash2.mcapy`（1,293,942,784 バイト、SHA-256 `e6cb73f63d7c85d8a3f8e15ff222b3d99b375c3437c2d61f031115fb04013854`） |

MegaCapybara は、公式の 1 会話向けプリセットの既定（KV 6/4）のまま、ループ防止だけを切って起動しました。KV 8/8 でも 1 組測っています。コマンドは `launch/megacapybara_windows.cmd` にあります。YASAKANI のプロセス内の実行にはループ防止がありません。サーバー経由の実行（`video_runs`・`own_tests`）は、クライアントが `loop_guard: false` を送っていないため、サーバーの既定（ループ防止あり）のままでした。動画の回では働いていません（usage に `loop_guard` の項目がありません）。YASAKANI の全実行のコマンドは `launch/yasakani_windows.cmd` にあります。

### テスト

* NInfer が公開している 12 問（コード：CUDA・Python・TypeScript／物語：英語の推理小説・中国語の SF・中国語の会話／翻訳：中→英・英→中・Markdown／構造化：CSV・JSONL・SQL）。`prompts/ninfer_cases.json` に名前と SHA-256 があります。ファイルは NInfer のリポジトリ（github.com/Neroued/ninfer）のもので、ここには入れていません。
* 自作のテスト（`prompts/cases.json`）：`edit_code`（CPython の `Lib/textwrap.py`、約 4,700 トークン。`dedent` の中の変数名を 1 つ変えて全文を返す）、`edit_jsonl`（60 行の JSONL のうち 1 件を変えて全行を返す）、`jsonl_orders_8k`（注文データの JSONL を 300 行書く。8,192 トークンで打ち切り）。
* 思考なし。フォルダー名に `t06` がなければ greedy（temperature 0）、`t06` は temperature 0.6・top-p 0.95・top-k 20。答えは 4,096 トークンまで（自作のテストは 8,192、`code2000` のフォルダーは 2,000）。

### フォルダー

| `results/` のフォルダー | エンジンと条件 | 回数 | スクリプト |
|---|---|---|---|
| `yasakani_std_greedy` | YASAKANI std、greedy | 12 問 × 2 | `pub_corpus_greedy.py`（プロセス内） |
| `yasakani_fast_greedy` | YASAKANI fast、greedy | 12 × 2 | `pub_corpus_cond2.py`（プロセス内） |
| `yasakani_std_greedy_kv8` | YASAKANI std、greedy、KV を普通の 8 bit に（`--env YSKN_TB_CAP=0`） | 12 × 2 | `pub_corpus_cond.py`（プロセス内） |
| `yasakani_std_t06` | YASAKANI std、temperature 0.6 | 12 × 5 | `pub_corpus_direct.py`（プロセス内） |
| `yasakani_fast_t06` | YASAKANI fast、temperature 0.6 | 12 × 5 | `pub_corpus_cond2.py`（プロセス内） |
| `yasakani_std_code2000`・`yasakani_std_code2000_kv8`・`yasakani_fast_code2000` | YASAKANI、コードの 3 問、答えは 2,000 トークンまで | 3 × 2 | `pub_corpus_cond2.py`（プロセス内） |
| `megacapybara_kv64`・`megacapybara_kv88` | MegaCapybara Tiny、KV 6/4 と 8/8、greedy | 12 × 2 | `oai_bench_client.py`（サーバー、ストリーミング） |
| `megacapybara_kv64_code2000`・`megacapybara_kv88_code2000` | MegaCapybara、コードの 3 問、2,000 トークンまで | 3 × 2 | `oai_bench_client.py` |
| `video_runs` | 動画に使った 5 問（JSONL・英語の推理小説・Python・SQL・中→英）の 1 回ずつの記録。YASAKANI fast・YASAKANI std・MegaCapybara（KV 6/4）。チャンクごとの到着時刻（ミリ秒）つき | 各 1 | `stream_capture.py`（サーバー、ストリーミング） |
| `own_tests` | 自作のテストを YASAKANI fast・YASAKANI std・MegaCapybara（KV 6/4）で。`*.chunks.json` はチャンクごとの到着時刻 | 3 種類 × 2 | `custom_bench.py`（サーバー、ストリーミング） |

ほかのフォルダーには、`runs_*.jsonl`（1 行が 1 回の実行）と `texts/`（毎回の答えの本文）があり、MegaCapybara のフォルダーにはクライアントが出力した `summary_*.txt` もあります。`logs/client/` は各実行の画面出力、`logs/server/` はサーバーのログ（`mc_server_*` ＝ MegaCapybara、`serve_*` ＝ YASAKANI。`serve_fast.log` と `serve_std.log` は、クライアントが早く始まりすぎて HTTP 503 になった最初の試みのもの。`capture_fast.log` を参照）、`logs/client/chain.log`・`own_tests_chain.log`・`std_t06_chain.log` は各手順の開始と終了の時刻（日本時間）、`gpu/` は前後の nvidia-smi の状態（電力の上限、最大クロック、メモリ、温度）です。
`scripts/summarize.py` は、これらの記録から問題ごと・種類ごとの平均、動画の回、自作のテストを表示します。`scripts/loop_scan.py` は最後が完全な繰り返しになっている答えを探し、`scripts/preloop_speed.py` は英語の推理小説の、繰り返しに入る前の速さを測ります（`preloop_speed.py` は `20261103` フォルダーから実行）。

### 時刻（2026 年 11 月 3 日、日本時間）

YASAKANI std の temperature 0.6：9:22〜9:30。YASAKANI std の greedy：11:55〜11:59。8 bit KV：12:12〜12:15。コード 2,000：12:16〜12:17。YASAKANI fast：12:40〜12:51。MegaCapybara：14:23〜14:32。動画の回：14:47〜14:51。自作のテスト：16:02〜16:05。

## 測り直す方法

### ほかのエンジン（11 月 1 日）

1. WSL2（Ubuntu 24.04）に SGLang 0.5.22 と vLLM 0.31.0 を入れ（FlashInfer のカーネルのビルドに CUDA 13.1 を使います。`common.sh` を参照）、llama.cpp は b11700 の公式 Windows CUDA ビルドを使います。上のモデルファイルをダウンロードします。
2. `launch/common.sh` 用に `MODELS`・`SGLANG_VENV`・`VLLM_VENV` を設定します（必要なフォルダー名は `common.sh` の先頭にあります）。
3. サーバーを 1 つだけ起動します。たとえば `bash launch/sg_dflash_nv.sh`（ポート 30000）、または `launch/llama_cpp_windows.txt` の llama.cpp のコマンド（ポート 8091）。`long.txt` には上の長い入力用の引数を足します。
4. ウォームアップを 2 回送ってから、各プロンプトを `scripts/capture_stream.py` で送ります。例：
   `python capture_stream.py --base http://127.0.0.1:30000 --prompt-file ../prompts/chat.txt --out chat_1.json --max_tokens 2048 --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0 --seed N --extra "{\"chat_template_kwargs\": {\"enable_thinking\": true}, \"presence_penalty\": 0.0}"`
   テストごとの引数は `run_wf.py`（`THINK`・`NINFER`・`JPNT`・`RAND`・`jobs()`）にあります。NInfer のプロンプトはメッセージの一覧として送ります（`--messages-file`）。
5. フォルダーの行（`<...>` の部分）を直せば、`scripts/run_wf.py` をそのまま使うこともできます。ここに入っていない `gpu_lock` という補助が必要です（何もしないコンテキストマネージャーで代わりになります）。
6. `wf_to_summary.py` と `headline.py` で記録を表にします。

### 今回の公開の表（11 月 6 日・8 日）

1. YASAKANI：今回の公開物を `20261108/launch/yasakani_windows.cmd` のとおりに起動します。MegaCapybara・mkyt・ignis：それぞれの Windows 用の配布物を `megacapybara_windows.cmd`・`mkyt_windows.cmd`・`ignis_windows.cmd` で。NInfer・TensorFold：WSL2 で `ninfer_master_build.sh`・`serve_ninfer_master.sh`・`serve_tensorfold.sh`。SGLang・vLLM・llama.cpp：上の 11 月 1 日と同じです。
2. NInfer の問題のファイルを、`20261108/scripts/oai_bench_client.py` の先頭の `FIX` が指す場所に置きます。
3. サーバーを 1 つだけ起動し、起動ファイルに書いてある引数で 2 つのクライアントを動かします。12 問は `oai_bench_client.py`、自作の課題 3 つは `custom_bench.py --cases 20261103/prompts/cases.json` です。
4. フォルダーの行（`<...>` の部分）を直せば、`scripts/run_wf_n12.py` をそのまま使うこともできます。`run_wf.py` と同じく、ここに入っていない `gpu_lock` という補助が必要です。
5. `python 20261108/scripts/tables.py` で、このフォルダーの記録から表を作ります。

### 品質の表（11 月 8 日）

このフォルダーだけでは測り直せません。評価に使った文と基準のファイルを公開していないためです。スクリプトは、llama.cpp の perplexity ツールの形式の基準ファイルを自分で用意すれば、そのまま動きます（`llama-perplexity -m <モデル> -f <文> --kl-divergence-base <ファイル>` で作れます。形式は `kld_base.py` に書いてあり、ロジットから作ることもできます）。

1. llama.cpp：`python 20261108/quality/scripts/llama_kld.py --llama-perplexity <exe> --base-file <ファイル> --gguf <名前>=<gguf>`。
2. SGLang・vLLM：`20261108/quality/launch/` のとおりにサーバーを起動し、`api_topk_dump.py --api sglang|vllm --base http://127.0.0.1:30000 --base-file <ファイル> --label <名前>`、続けて `topk_score.py --base-file <ファイル> --label <名前>`。
3. `python 20261108/quality/scripts/quality_table.py` で、このフォルダーの点数のファイルから表を作れます。

### 11 月 10 日の確認

コマンドは `20261110/launch/checks_windows.cmd` にあります。`yasakani verify`・`selfcheck`・`bench` は、公開物だけで動きます。`onoff_eq.py` と `repeat_probe.py` は、`20261101/prompts/` のプロンプトと NInfer の問題のファイルを読みます（フォルダー 2 つは `main()` の先頭近くで決めています）。`repeat_probe.py` には、`20261101/scripts/` の `onoff_eq.py` と `a_speed.py` も必要です（`--tools`）。`gpu_consistency.py` には CUDA 版の PyTorch が必要です。

### YASAKANI（11 月 1 日・3 日）

公開版で：`20261101/launch/yasakani_windows.cmd`（11 月 1 日）と `20261103/launch/yasakani_windows.cmd`（11 月 3 日）。自分のパソコンは、もっと短いテストの `yasakani bench` でも測れます。

### MegaCapybara（11 月 3 日）

`20261103/launch/megacapybara_windows.cmd` がサーバーを起動します。クライアントのコマンドも同じファイルにあります。

## わかっている制約

### 今回の公開の表（20261108/）

* 同じ日ではありません。ほかのエンジンは 11 月 6 日、MegaCapybara の思考なしは 11 月 3 日、YASAKANI の 2 つのモデルは 11 月 8 日に測りました。サーバーは 1 つずつ順番に動かし、交互には測っていません。ジョブの始めの GPU の温度は、11 月 6 日が 34〜51 度、11 月 8 日の YASAKANI が 30 度でした。ジョブの始めには、ほかのプログラム（デスクトップ）が GPU のメモリを 0.4〜2.1 GB 使っていました（`logs/jobs/` の `gpu memory used`。YASAKANI のジョブの前は 1.7 GB）。
* 回数が少ないです。思考なしは課題ごとに 2 回、思考ありはほかのエンジンが 2 回、YASAKANI が 5 回です。そのため「課題ごとに一番速い回」は、YASAKANI は 5 回、相手は 2 回の中から選んでいます。平均で作った表を並べて出しています。
* 思考あり：YASAKANI の 12 問は 120 回すべてが出力の上限で終わり、そのうち 114 回は思考の途中でした。どのエンジンでも、大半の回が上限で終わっています。数字は生成の速さで、答えにたどり着くまでの時間でも、正しさでもありません。
* 思考なし：greedy の答えの一部は、くり返しや、頼んでいないツール呼び出しになります。そういう文章は、ふつうの文章より速く書かれます。README の表は、そういう回を全エンジン同じ規則で除いています（除いた回は `tables.py` が出します。YASAKANI の最速モデルは CUDA の 2 回と TypeScript の 2 回が除かれ、「コード」が 900 から 529 になります。品質モデルは英語の推理小説の 2 回。mkyt・NInfer・vLLM・SGLang は TypeScript の回、vLLM の DFlash 2 の設定は CUDA の回も、llama.cpp の MTP の設定は英語の推理小説の 1 回）。この規則は、保存した本文の終わりの部分だけを見るもので、答えの出来を判定するものではありません。
* NInfer：表は 11 月 6 日時点の `master`（11 月 2 日付のコミット）のビルドです。同じ日の午前に同じ設定で測った 10 月のビルド（リビジョン `81c8ce09`）は `results/n12_ninfer` にあり、表には使っていません。その「コード」は、除く前が 290（表のビルドは 282）、除いた後が 277（同 264）でした。ほかの 6 列では、10 月のビルドは 3.1% 遅い〜0.1% 速い範囲です。
* TensorFold：0.6.6 です。11 月 6 日に v1.2.2 への更新を試みましたが、ビルドに失敗しました（`logs/jobs/job_1003_ninfer_october_tensorfold.log`）。新しい版は測れていません。全体として、その日に動かせた版の記録で、どのソフトも最新版というわけではありません。
* mkyt は、このパソコンの電力の上限 600 W のまま動かしました。mkyt 自身が公表している数字は 400 W です。
* SGLang・vLLM・NInfer・TensorFold は、Linux で直接ではなく WSL2 で動かしました。
* 各エンジンは、それぞれのモデルファイル・数値の形式・KV キャッシュ・高速化の設定を使っています（`20261108/launch/`）。大きさも違います。全部の設定をそろえた比較ではありません。
* ループ防止：クライアントはループ防止の指定を送っていないので、YASAKANI のサーバーは既定のままです。これらの実行の usage に `loop_guard` の項目はありません。MegaCapybara はループ防止を切って動かしました。
* 比べているのは生成の速さだけです。プロンプトを読む時間は数字に入っていません。プロンプトは長くても約 4,800 トークンです。
* ドライバーのログ `n12.log` と `n12t.log` の最後に `ysk_std_1103` という手順があります。11 月 3 日の YASAKANI のビルドと標準モデルを、その日の確認として動かしたものです（`n12t.log` では起動に失敗し、あとで測り直しました）。その記録と、11 月 6 日・7 日に測った以前の YASAKANI のモデルファイルの記録は、このフォルダーには入れていません。それらのファイルを配布していないためです。`run_wf_n12.py` には設定が残っています。
* 11 月 8 日に、最速モデルの思考ありのコード編集をさらに 3 回（seed 5〜7）測りましたが、表には使っておらず、ここにも入れていません。

### 品質の表（20261108/quality/）

* 自分たちの文での評価です。使ったのは自分たちの思考文 20 本で、モデルを作るときの統計には使っていない文です。この表が示すのは、その文の上で、各構成が元のモデルにどれだけ近いかです。一般的な品質の点数でも、課題の正答率でもありません。
* YASAKANI の 2 行は、研究用のビルドと FP16 の KV キャッシュで測りました。今回公開する `yskn.dll` でも、速度表の KV 形式である KV5 でもありません。KV5 による違いは、最速モデルについて別に測っています（`long_context/`）。
* 3 行（SGLang・vLLM・mkyt）は下限です。本当の KL は、表の値以上になります。そのため、SGLang の 0.0283 と最速モデルの 0.0304 の順番は決められません。
* 読ませ方はエンジンごとに違います。YASAKANI は検証の経路で 16 位置ずつ、ほかの多くはプロンプトを処理する経路、MegaCapybara は 1 位置につき 1 回の要求です。
* NInfer の値は、書き出しを足したソースの写しから取りました。MegaCapybara の値は、動いているプロセスのメモリから読みました。この 2 つの方法が数字を変えていないことの根拠は、上に書いた確認だけです。
* 測定は各構成 1 回で、文脈は 2,048 トークンまでです。
* 容量はファイルの大きさで、中身はそろっていません（画像部、補助のモデル、補助の層を含むものがあります）。同じ容量での圧縮の比較ではありません。2 つの容量帯は、大きさの近いファイルを並べたもので、同じ大きさではありません。
* このフォルダーだけでは測り直せません。評価に使った文と基準のファイルを公開していないためです。

### 以前の比較（20261101/・20261103/）

* エンジンは同じ日に順番に測りました（時刻は上のとおり）。交互には測っていません。どのサーバーも GPU を 1 つで使いました。
* YASAKANI の速さは、ループ防止を切って測りました。ほかのエンジンには同じ仕組みがないためです。ループ防止を入れた確認（`A_pub_guard.json`）では、思考ありの 6 種類で 1 回目との差は最大 0.32% でした。
* どちらの数字も「良い方」を選んだもので、選び方が違います。YASAKANI は 2 回のうち速い方の回、ほかのエンジンはテストごとに一番速い設定です。
* SGLang と vLLM は Linux で直接ではなく WSL2 で動かしました。
* 各エンジンは、それぞれ速く動くモデルファイルを使ったので、形式と大きさが違います（YASAKANI 12.4 GB、llama.cpp の GGUF 12.1 GB、SGLang・vLLM の NVFP4 は読み込んだ量で約 18〜22 GB）。大きさをそろえた比較ではありません。
* 長い入力：SGLang 公式の RTX 5090 レシピ（`sg_dflash_nv+48k`）は KV を 39,571 トークン分しか取れず、入力の上限は 39,565 トークンで、41,261 トークンのプロンプトを 3 回とも受け付けませんでした（`logs/server/server_x2l_sg_dflash_nv_48k.log`）。この 3 回は 1 トークンとして記録にあり、数えていません。SGLang の長い入力の数字は、ほかの設定のものです（`launch/notes_2026-10-18.txt` は 10 月 18 日の試行の値として 35,067 トークンと書いていますが、11 月 1 日のログでは 39,571 です）。vLLM の DFlash 2 と DSpark の設定は、10 月 18 日に同じ長いプロンプトを続けて送ると止まったため、長い入力ではプレフィックスキャッシュを切って測りました。
* `results_summary_others.csv` の `yasakani` 列は、`wf_to_summary.py` が 10 月 18 日の表から写した値で、11 月 1 日の測定ではありません。倍率の列はわざと空です。11 月 1 日の YASAKANI の数字は `results/yasakani/` にあります。
* `a_speed.py summary` は 2 回のうち遅い方の回を出し、10 月 18 日の表と比べます。11 月 1 日にこれが出力した表は入れていません。公表に使った数字（速い方の回、ほかのエンジンは 11 月 1 日の実行）は `headline.py` が出します。
* 11 月 3 日：12 問での YASAKANI の数字はプロセス内（エンジンの生成時間、サーバーなし）、MegaCapybara の数字はサーバー経由のストリーミングで測りました。動画の回と自作のテストは、どちらもサーバー経由です。同じ問題で比べると、YASAKANI のサーバー経由の値はプロセス内の平均との差が 2.5% 以内でした（英語の推理小説を除く。下記）。
* 11 月 3 日：英語の推理小説に対する YASAKANI の greedy の答えの一部は、最後が完全な繰り返しになり、その部分は速く書かれます（`loop_scan.py`：std は greedy の 2 回と動画の回とも本文の 35.0%、fast は greedy の 2 回が 0% と 10.9%、動画の回が 32.4%。`preloop_speed.py`：動画の回では繰り返しが std は約 2,663 トークン目、fast は約 2,770 トークン目から）。動画の回の最初の 2,000 トークンでの速さは、302.7（fast、チャンクからの推定）、271.9（std、推定）、237.7（MegaCapybara、正確。繰り返しなし）です。これらのフォルダーのほかの答えには、temperature 0.6 の全実行を含め、最後の繰り返しはありません。
* 11 月 3 日：MegaCapybara は TypeScript の問題で、8 回中 5 回、147〜201 トークン目に `finish_reason: tool_calls` で止まりました。その回は平均から外しています（`summarize.py` が一覧を出します）。MegaCapybara の greedy の速さは、回ごとに 10% ほど揺れます。
* 11 月 3 日：YASAKANI のサーバー経由の実行（`video_runs`・`own_tests`）は、サーバーの既定のループ防止が入ったままでした（上記）。YASAKANI のプロセス内の実行と MegaCapybara の全実行には、ループ防止はありません。
* 11 月 1 日と 11 月 3 日の YASAKANI は、同じ版の別のビルドです（DLL の SHA-256 の先頭 `1fa3a253` と `d86fe2d0`）。
* 11 月 1 日：`checks/on_off_equality/` は、その日のビルドで 174 件すべて一致という記録です。11 月 10 日に、今回の公開物で同じスクリプトを、ほかのプログラムが GPU を使っている状態で動かしたところ、一致したのは最速モデルが 174 件のうち 171 件、品質モデルが 170 件でした（[`docs/MEASUREMENTS.ja.md`](../docs/MEASUREMENTS.ja.md) の「この公開版での確認」）。11 月 1 日の記録は、測ったまま残しています。

## 公開のための変更

* 測定に使ったパソコンのパスは、次の置き換えにしました：`<model_dir>`・`<model_dir_fast>`（モデルのフォルダー）、`<yasakani_dir>`（`bin\yskn.dll` のあるフォルダー）、`<yasakani_py>`（Python パッケージ）、`<llama_cpp_dir>`、`<gguf_dir>`、`<megacapybara_dir>`、`<python_dir>`、`<work_dir>`、`<out_dir>`、`<tmp_dir>`、`<helpers_dir>`、`<reference_dir>`、`<oct18_run_dir>`、`<ninfer_corpus_dir>`、`<sglang_venv>`、`<vllm_venv>`、`<trtllm_venv>`、`<log_dir>`。ユーザー名は `<user>` にしました。
* 他者のプロンプトの本文：NInfer のプロンプトの記録ファイルでは、`prompt` 欄を `prompts/ninfer_cases.json` への参照に置き換えました。答え（NInfer の文章の翻訳を含む）は生成の記録なので残しています。AIME 2025 の問題と答えは入れておらず、集計だけです。
* CPython の `textwrap.py` の冒頭にあるメールアドレス（`edit_code` のプロンプトとその答えに含まれる）は、`cases.json`・答え・チャンクの記録で `<email>` に置き換えました。3 つのエンジンの答えは、置き換えたあとも互いに同じです。`textwrap.py` は Python Software Foundation ほかの著作物で、PSF License です。
* 品質の集計 2 件（`checks/quality/summary_kv*.json`）：エンジンの環境設定の一覧を、注記に置き換えました。
* `20261103/logs/megacapybara_exe_sha256.txt`（`certutil` の出力）は、Windows の日本語の文字コードから UTF-8 に変換しました。
* スクリプト：コメントや docstring の内部向けのメモを削り、フォルダーの行を置き換えました。`loop_scan.py` と `preloop_speed.py` は、このフォルダー構成の名前を使うようにしました。記録の中の `script_sha256` は、測定に使ったときのスクリプトのものです。
* 答えの中には、モデルが書いた語や `example.com` のメールアドレスがあり、書かれたままにしています。ランダムな 8,192 トークンのプロンプトは多くの言語の語彙をランダムに並べたもので、測った入力そのものなので変えていません。
* ファイル名の変更：ほかのエンジンの CSV 2 件は名前の末尾の日付を外し、`runs.jsonl` は `runs_others.jsonl` にしました。11 月 3 日の結果のフォルダーには、内容がわかる名前を付けました（上の表）。
* `20261108/`：同じ置き換えをしています。追加の置き換えは `<model_dir_fastest>`・`<model_dir_quality>`・`<ignis_dir>`・`<mkyt_dir>`・`<mkyt_hf_dir>`・`<ninfer_dir>`・`<ninfer_src_dir>`・`<tensorfold_dir>`・`<tensorfold_venv>`・`<launch_dir>`、`run_wf_n12.py` の中の `<yasakani_dir_nov3>`・`<model_dir_nov3>`・`<model_dir_fast_nov3>`・`<model_dir_other_helper>`・`<yasakani_dir_oct31>`・`<install_job>`・`<download_job>` です。`textwrap.py` のメールアドレスは、`edit_code` の答えとチャンクの記録で `<email>` にしました（2 つのチャンクに分かれている場合も）。
* `20261108/logs/jobs/`：ジョブのログは、手順の時刻の行と nvidia-smi の出力の行だけにしました。ほかの行（常に動かしている自分用のサーバーの停止と再開、画面の雑多な出力）は落としています。
* `20261108/scripts/`：コメントと docstring の中の内部向けの覚え書きを消し、フォルダーの行を置き換え、またはこの配置のフォルダー名にしました。`tables.py` は、この配置に合わせて書いたものです（「表の作り方」）。
* `20261108/launch/*.cmd` は、実際に動かしたジョブのファイルから、公開用に書き出したものです。サーバーとクライアントのコマンドは動かしたとおりで、フォルダーだけ置き換えています。`.sh` は動かしたスクリプトそのもので、フォルダーだけ置き換えています。
* `20261108/long_context/`：ログは測定スクリプトの行だけにし、KV 形式の内部の呼び名を `kv5` に、研究用のビルドの名前を短い説明に置き換えました。`kv5_summary.json` と `KV5_TABLES.md` は、作業用の要約から作ったものです。
* `20261108/demo/` は、証拠の一式を何も変えずに入れています（`manifest.json` のハッシュと合うように）。そのため、ここでは `textwrap.py` の先頭（メールアドレスを含む）は CPython のままです。
* `20261108/quality/`：点数のファイルは、作業用の結果ファイルから書き出しました。数字はそのまま写し、ラベル・キーの名前・方法の説明を公開用にしています。フォルダー、研究用のライブラリの名前、GPU のメモリを使っていたほかのプログラムの一覧は入れていません。スクリプトは、作業用のスクリプトのフォルダーの行を引数に変え、補助のモジュールの名前を `kld_base.py` に変え、docstring から内部向けのメモを除いたものです。`gguf_main_size.py` は、作業用のスクリプトの見出しを読む部分に、小さなコマンドラインをつけたものです。`quality_table.py` は、この配置に合わせて書きました。起動スクリプト 2 つは実際に使ったもので、`20261101/launch/common.sh` を読むようにしています。ログは、フォルダーを置き換えただけです。11 月 10 日に、公開する `topk_score.py` を PC で、保存してある 4 組の log 確率と基準のファイルに対して動かし、保存してある点数とすべての桁で同じになることを確かめました。`llama_kld.py` の出力の読み取りは 6 つのログで、`gguf_main_size.py` は 6 つのファイルで、保存してある値と同じでした。`quality_table.py` も同じ表を出しました。`api_topk_dump.py` は、エンジンのサーバーをもう一度起動していないので、作り物の数字を返す代わりのサーバーでだけ動かしました。
* `20261110/`：同じ置き換えをしています（`<model_dir_fastest>`・`<model_dir_quality>`・`<yasakani_dir>`・`<out_dir>`）。ジョブのログも同じように減らしました。`repeat_probe.py` は、プロンプトのフォルダー 2 つを、`onoff_eq.py` と同じ `<work_dir>` にしました。`repeat_summary.py` と `launch/checks_windows.cmd` は公開用に書いたものです。
