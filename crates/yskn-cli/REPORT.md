# Brief 06d 実装・検証レポート

状態: **実装済み、証明ゲートは BLOCKED**。この環境は
`bind(127.0.0.1:0)` を `Operation not permitted (os error 1)` で拒否する。
権限の迂回、実ソケット試験の削除・ignore・環境依存skipはしていない。

作業場所は指定worktree `.work/wt/06d`、ブランチは `feat/06d-cli`。
親checkout・他worktree・禁止パスは編集していない。ネットワーク取得も
モデル重みの読み込みも行っていない。全Cargo実行はoffline。

## コミットと変更ファイル

- `753b250`: `feat(cli): add synchronous chat and local OpenAI server`
- テスト・文書のコミットハッシュは最終応答に記載（この文書自身を含むため）。
- push は実施していない。

- `Cargo.lock`
- `crates/yskn-cli/Cargo.toml`
- `crates/yskn-cli/README.md`
- `crates/yskn-cli/REPORT.md`
- `crates/yskn-cli/data/stub-template.jinja`
- `crates/yskn-cli/data/stub-tokenizer.json`
- `crates/yskn-cli/src/chat.rs`
- `crates/yskn-cli/src/cli.rs`
- `crates/yskn-cli/src/engine.rs`
- `crates/yskn-cli/src/lib.rs`
- `crates/yskn-cli/src/loopguard.rs`
- `crates/yskn-cli/src/main.rs`
- `crates/yskn-cli/src/request.rs`
- `crates/yskn-cli/src/server.rs`
- `crates/yskn-cli/src/stub.rs`
- `crates/yskn-cli/tests/api.rs`
- `crates/yskn-cli/tests/chat.rs`
- `crates/yskn-cli/tests/cli.rs`
- `crates/yskn-cli/tests/common/mod.rs`
- `crates/yskn-cli/tests/data/completion.json`
- `crates/yskn-cli/tests/data/stream.sse`
- `crates/yskn-cli/tests/server.rs`
- `crates/yskn-text/src/lib.rs`
- `crates/yskn-text/src/template.rs`
- `docs/port/README.md`

`Cargo.lock` に新しいcrateは追加していない。既存の `sha2` をCLIの依存先に追加。
`yskn-text` の変更は `Clone`、埋め込み文字列用 `ChatFormatter::from_json`、
既存JSONフォーマッターを公開する `json_dumps` の追加。
既存loadは共通の初期化ヘルパーに委譲し、既存のファイル検査を保持する。

## 証明コマンド

`export CARGO_NET_OFFLINE=true` を設定し実行。ログ保存先はworktree内
`target/06d-proof/`（Git対象外）。各コマンドの終了値・末尾は以下。
fmt/clippy/helpはテスト件数なし。

### `cargo fmt --all --check`

終了値: `0`。

```text
出力なし（成功）
```

### `cargo clippy --workspace --all-targets -- -D warnings`

終了値: `0`。

```text
    Checking yskn v0.1.0 (/Users/anag2bgod/Projects/yasakani-for-mac/.work/wt/06d/crates/yskn-cli)
    Finished `dev` profile [optimized + debuginfo] target(s) in 0.76s
```

### `cargo test --workspace`

終了値: `101`。

```text
failures:

---- serve_port_in_use stdout ----

thread 'serve_port_in_use' (461575) panicked at crates/yskn-cli/tests/cli.rs:187:63:
called `Result::unwrap()` on an `Err` value: Os { code: 1, kind: PermissionDenied, message: "Operation not permitted" }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- serve_end_to_end stdout ----

thread 'serve_end_to_end' (461574) panicked at crates/yskn-cli/tests/cli.rs:231:5:
error: cannot listen on 127.0.0.1:0 (Operation not permitted (os error 1)). Is another server already using this port? Use --port to pick another.



failures:
    serve_end_to_end
    serve_port_in_use

test result: FAILED. 7 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s

error: test failed, to rerun pass `-p yskn --test cli`
```

### `cargo test -p yskn --release`

終了値: `101`。

```text
failures:

---- serve_port_in_use stdout ----

thread 'serve_port_in_use' (465053) panicked at crates/yskn-cli/tests/cli.rs:187:63:
called `Result::unwrap()` on an `Err` value: Os { code: 1, kind: PermissionDenied, message: "Operation not permitted" }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- serve_end_to_end stdout ----

thread 'serve_end_to_end' (465052) panicked at crates/yskn-cli/tests/cli.rs:231:5:
error: cannot listen on 127.0.0.1:0 (Operation not permitted (os error 1)). Is another server already using this port? Use --port to pick another.



failures:
    serve_end_to_end
    serve_port_in_use

test result: FAILED. 7 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s

error: test failed, to rerun pass `-p yskn --test cli`
```

### `cargo run --release -p yskn -- serve --engine stub --port 0 --help`

終了値: `0`。

```text
    Finished `release` profile [optimized] target(s) in 0.06s
     Running `target/release/yskn serve --engine stub --port 0 --help`
yskn - local Rust chat product (real engine not connected yet)
commands: serve, chat, info, bench, selfcheck
  --engine stub       deterministic demo engine; required until integration
  --model-dir FOLDER  optional real tokenizer/template, read only
  --max-ctx N         context capacity (default 262144)
  --seed0 N           first automatic request seed (default 1)
serve:
  --host HOST         default 127.0.0.1
  --port PORT         default 8765; 0 chooses an ephemeral port
  --api-key KEY       or YASAKANI_API_KEY; required for remote binds
  --default-max-tokens N  default 32768
  --max-tokens-cap N  request upper limit
  --no-warmup
chat:
  --system TEXT --max-tokens N --temperature N --top-p N --top-k N
  --seed N --no-think --effort low|medium|xhigh --hide-thinking --no-loop-guard
  /help shows terminal commands
YASAKANI_THINKING_BUDGET sets the server default (0 = unlimited).
bench and selfcheck: not available yet
  --help, -h, --version
```

debug/releaseとも、サービス7件 + チャット35件 + CLI7件 = **49件通過**。
CLIサーバー2件がbind拒否で失敗し、Cargoはそこで停止するため、必須
`cargo test --workspace` の他crateの実行と、このコマンド内のserver targetは未到達。
workspace/release双方のゲートを成功とは報告しない。

補足確認:

- `cargo test -p yskn --test server`: 最終実装で実行し、**0 passed / 25 failed**。
  全件がbind拒否でHTTP処理到達前に失敗。ログは `target/06d-proof/socket.log`。
- `cargo test -p yskn-text`: **9 passed / 0 failed**、doc-test 0件。
  モデル依存ケースは既存仕様により `skipped: YSKN_MODEL_DIR is not set`。
  実モデルのtokenizer/template比較を実施したという意味ではない。
- ソケットなしのサービス検証は本番と同じ `Api::call` と専用エンジンスレッドを使用。
  JSON/SSEゴールデン、全要求検証エラー、実測本文上限、本文deadline、認証・同一origin保護、
  思考/回答/ツール呼出し、文脈超過、エンジン障害、待機、body破棄キャンセル、停滞停止を確認。
  **実TCPの動作証明には代用していない**。
- ゴールデンは固定プロトコルとbyte-token計数から独立に作成。
  動的なid/createdフィールドだけを置換し、残る本文bytesを完全一致で比較。
- `git diff --check` はSSEゴールデン終端の空行を指摘。SSEに必要な `\n\n` のbytesとして保持。
  全targetのclippy成功により実ソケット試験もコンパイル済み。

## エンジンtrait

```rust
pub trait Engine {
    fn context_capacity(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stop_ids: Option<&[u32]>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation>;
}
```

同期・object-safe、Send制約なし。`generate` 内で互換promptを再利用、または状態をリセット。
`max_new` は最初の予測token/EOSを含む。callbackには累積のcommit済みidsを渡す。
`false` は次のcommit境界で停止、Errは停止後に伝播。テキストにならないchunkでも呼ぶ。
Samplingはtemperature/top_p/top_k/min_p/seed。greedy時の既定化とtop_k上限は上流通り。
Generationはids、Stop/Length/Cancelled、cached_tokens、snap_mode、passes、prefill_ms、gen_ms。
ChatRunnerは `Reasoning` / `Content` / `ToolCall` / `Done(Usage)` を発行する。

サーバーのfactoryのみがSendであり、そのfactoryが専用スレッド内でrunner/非Send engineを構築。
HTTP側は有界job/event channelを利用し、次の生成要求は待つ。実TCP試験fixtureにはRcを持つ
非Send adapterと所有スレッドassertを用意している（本環境ではbind拒否で実行未到達）。
Stubはprompt由来の決定的なデモ出力に加え、script queue、loop_from、delay、障害注入、
停止回数、sampling、prompt再利用の観測を提供し、上流C stubの責務を置換する。

## 上流テストとRust対応

各Rustテスト直前のコメントにも上流名を保持。PASSは合成fixtureによる確認で、実モデル一致の主張ではない。
BLOCKEDはテストを実装・コンパイルしたが実ソケットまで到達できないことを意味する。

| 上流 | Rust対応/対象外の理由 | この環境での結果 |
|---|---|---|
| `test_chat.StopFilterTests.test_cut_and_holdback` | `tests/chat.rs::cut_and_holdback`  | PASS |
| `test_chat.StopFilterTests.test_partial_stop_is_released` | `tests/chat.rs::partial_stop_is_released`  | PASS |
| `test_chat.StopFilterTests.test_no_stops_passthrough` | `tests/chat.rs::no_stops_passthrough`  | PASS |
| `test_chat.StopFilterTests.test_two_stops_earliest_wins` | `tests/chat.rs::two_stops_earliest_wins`  | PASS |
| `test_chat.CleanMessageTests.test_ok_and_errors` | `tests/chat.rs::ok_and_errors`  | PASS |
| `test_chat.ChatTests.test_reasoning_then_answer` | `tests/chat.rs::reasoning_then_answer`  | PASS |
| `test_chat.ChatTests.test_thinking_disabled` | `tests/chat.rs::thinking_disabled`  | PASS |
| `test_chat.ChatTests.test_stop_string` | `tests/chat.rs::stop_string`  | PASS |
| `test_chat.ChatTests.test_length_finish` | `tests/chat.rs::length_finish`  | PASS |
| `test_chat.ChatTests.test_loop_guard_bridges_to_an_answer` | `tests/chat.rs::loop_guard_bridges_to_an_answer`  | PASS |
| `test_chat.ChatTests.test_loop_guard_off` | `tests/chat.rs::loop_guard_off`  | PASS |
| `test_chat.ChatTests.test_loop_in_the_answer_stops` | `tests/chat.rs::loop_in_the_answer_stops`  | PASS |
| `test_chat.ChatTests.test_prompt_too_long` | `tests/chat.rs::prompt_too_long`  | PASS |
| `test_chat.ChatTests.test_seed_counter_and_explicit_seed` | `tests/chat.rs::seed_counter_and_explicit_seed`  | PASS |
| `test_chat.ChatTests.test_emit_failure_stops_the_engine` | `tests/chat.rs::emit_failure_stops_the_engine`  | PASS |
| `test_server.ServerTests.test_health_and_models` | `tests/server.rs::health_and_models`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_loading_state` | `tests/server.rs::loading_state`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_completion` | `tests/server.rs::completion`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_streaming` | `tests/server.rs::streaming`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_stream_has_one_chunk_per_token` | `tests/server.rs::stream_has_one_chunk_per_token`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_rejections` | `tests/server.rs::rejections`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_bad_json` | `tests/server.rs::bad_json`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_host_and_origin_checks` | `tests/server.rs::host_and_origin_checks`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_api_key` | `tests/server.rs::api_key`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_engine_failure_is_a_500` | `tests/server.rs::engine_failure_is_a_500`  | BLOCKED: bind 拒否 |
| `test_server.ServerTests.test_client_disconnect_stops_the_engine` | `tests/server.rs::client_disconnect_stops_the_engine`  | BLOCKED: bind 拒否 |
| `test_tools.ToolCallFilterTests.test_plain_text_passes_unchanged` | `tests/chat.rs::plain_text_passes_unchanged`  | PASS |
| `test_tools.ToolCallFilterTests.test_one_call_after_some_text` | `tests/chat.rs::one_call_after_some_text`  | PASS |
| `test_tools.ToolCallFilterTests.test_two_calls` | `tests/chat.rs::two_calls`  | PASS |
| `test_tools.ToolCallFilterTests.test_an_unfinished_call_comes_back_as_text` | `tests/chat.rs::an_unfinished_call_comes_back_as_text`  | PASS |
| `test_tools.ToolCallFilterTests.test_an_unreadable_block_comes_back_as_text` | `tests/chat.rs::an_unreadable_block_comes_back_as_text`  | PASS |
| `test_tools.ToolCallFilterTests.test_the_json_form_is_read_too` | `tests/chat.rs::the_json_form_is_read_too`  | PASS |
| `test_tools.ToolCallFilterTests.test_parameter_values` | `tests/chat.rs::parameter_values`  | PASS |
| `test_tools.ToolRequestTests.test_clean_tools` | `tests/chat.rs::clean_tools_cases`  | PASS |
| `test_tools.ToolRequestTests.test_clean_messages_with_tool_calls` | `tests/chat.rs::clean_messages_with_tool_calls`  | PASS |
| `test_tools.ToolRequestTests.test_parse_chat_request` | `tests/chat.rs::parse_chat_request_cases`  | PASS |
| `test_tools.ToolRunTests.test_the_template_shows_the_tools_and_reads_a_call_back` | `tests/chat.rs::the_template_shows_the_tools_and_reads_a_call_back`  | PASS |
| `test_tools.ToolRunTests.test_runner_returns_the_call` | `tests/chat.rs::runner_returns_the_call`  | PASS |
| `test_tools.ToolRunTests.test_server_answers_with_tool_calls` | `tests/server.rs::server_answers_with_tool_calls`  | BLOCKED: bind 拒否 |
| `test_tools.ToolRunTests.test_server_stream_sends_one_chunk_per_call` | `tests/server.rs::server_stream_sends_one_chunk_per_call`  | BLOCKED: bind 拒否 |
| `test_tools.ToolRunTests.test_server_takes_the_tool_result_and_answers` | `tests/server.rs::server_takes_the_tool_result_and_answers`  | BLOCKED: bind 拒否 |
| `test_tools.ToolRunTests.test_server_without_tools_leaves_the_text_alone` | `tests/server.rs::server_without_tools_leaves_the_text_alone`  | BLOCKED: bind 拒否 |
| `test_tools.ToolRunTests.test_server_refuses_a_forced_tool_choice` | `tests/server.rs::server_refuses_a_forced_tool_choice`  | BLOCKED: bind 拒否 |
| `test_tools.ToolRunTests.test_an_input_that_does_not_fit_is_a_400_before_the_stream_starts` | `tests/server.rs::an_input_that_does_not_fit_is_a_400_before_the_stream_starts`  | BLOCKED: bind 拒否 |
| `test_cli.CliTests.test_help_and_version` | `tests/cli.rs::help_and_version`  | PASS |
| `test_cli.CliTests.test_info` | `tests/cli.rs::info` （DLL/重み検査は静的stub/tokenizer情報へ置換） | PASS |
| `test_cli.CliTests.test_verify` | 対象外: verify コマンドは本ブリーフに含まれない。 | 対象外 |
| `test_cli.CliTests.test_chat` | `tests/cli.rs::chat`  | PASS |
| `test_cli.CliTests.test_chat_ends_on_end_of_input` | `tests/cli.rs::chat_ends_on_end_of_input`  | PASS |
| `test_cli.CliTests.test_bench` | 明示的な後続ブリーフ対象。recognized_unavailable_commands で未提供表示を確認。 | 対象外 |
| `test_cli.CliTests.test_selfcheck_roundtrip` | 明示的な後続ブリーフ対象。recognized_unavailable_commands で未提供表示を確認。 | 対象外 |
| `test_cli.CliTests.test_gpu_memory_warning` | Windows/CUDA/nvidia-smi 固有。静的Rust情報に置換。 | 対象外 |
| `test_cli.CliTests.test_bad_inputs` | `tests/cli.rs::bad_inputs` （実エンジン未接続・stubのモデル不要条件へ適応） | PASS |
| `test_cli.CliTests.test_serve_port_in_use` | `tests/cli.rs::serve_port_in_use`  | BLOCKED: bind 拒否 |
| `test_cli.CliTests.test_serve_end_to_end` | `tests/cli.rs::serve_end_to_end` （CUDA環境上書き確認は対象外） | BLOCKED: bind 拒否 |

上流件数: `test_chat.py` 15件, `test_server.py` 11件, `test_tools.py` 18件, `test_cli.py` 11件。

## 意図的な差異と理由

1. HTTPServerをHyper HTTP/1に置換。外側JSONはcompact、HTTPのframing/Date/Server等は
   Hyperの挙動。ツールarguments内のJSON文字列は上流と同じspacing/float表記を再利用。
   unsupported methodは501を保持し、Python標準のHTMLではなく統一JSON errorを返す。
2. 拒否POSTは本文をdrainせず接続を閉じる。未認証/不正originのslow bodyで接続を占有させない。
   chunked requestは413で明示拒否。上流はchunked decodingを実装していない。
   壊れたHTTP framing/Content-Lengthや過大ヘッダーはhandler前にHyperが400/431/closeを決める。
3. 本文の実測上限、header deadline/上限、接続上限、event queue/停滞deadlineを追加。
   手組みHyper transportの必須保護。重複Host/Origin/Sec-Fetch-Site/Authorizationは曖昧さ回避のため拒否。
4. キーは固定長SHA-256 digestの全32byteを比較し、prefix依存の早期終了を避ける。
   上流のhmac.compare_digestの役割を、既存lock内依存だけで担う。
5. NaN/Infinityやi64外の整数パラメーターは400。Pythonの非標準JSON値・任意精度整数からの差異。
   i64内の整数seedはf64経由の精度落ちを避ける。既存text層のJSON数値制約も継承。
6. truthyな非objectのchat_template_kwargs/stream_optionsと不正なreasoning_effort型を400にする。
   Pythonのdict変換/attribute/unhashable等の例外に落ちる挙動、およびJSON object指定なのに
   list-of-pairsをdict変換で受け付ける曖昧さを修正。全templateエラーも400に統一。
7. Windows console codepage/色付け、DLL/CUDA/device指定、nvidia-smi/GPUメモリ警告、
   CUDA tuning環境設定の変更は移植しない。端末は指定通りplain line I/O。
   infoは静的Rust binary・stub・tokenizer情報を表示。CLIのargumentエラーは終了値1。
8. 明示stubは埋め込みの合成byte tokenizer/templateを使うためモデルフォルダー不要。
   実tokenizerを指定する場合はtext filesのみロード。実エンジン未接続時の暗黙代用はしない。
   warmupはtraitによる短いgenerate/resetで置換し、ネイティブCUDA warmupは使わない。
9. verifyは対象外。bench/selfcheckは指定通り後続ブリーフに延期し、未提供表示のみ実装。

非loopback bindのキーなし動作は**警告ではなく拒否**。上流CLIと一致。
キー付きremote bindでHost/Origin/Fetch検査を迂回しキー検証に委ねる規則も上流と一致。

## サーバー制限・timeout値

| 項目 | 値 |
|---|---:|
| 既定bind | 127.0.0.1:8765 |
| 既定max tokens | 32768 |
| 既定context capacity（stub） | 262144 tokens |
| context reserve | 64 tokens |
| body上限（JSON parse前、実測） | 32 MiB |
| 同時接続 | 64、超過接続はclose |
| 待機job channel | 64 |
| 応答event channel | 8 |
| header read buffer | 32 KiB |
| header件数 | 100 |
| header全体deadline | 10 s |
| body全体deadline | 30 s |
| socket read/write idle | 300 s |
| event channel満杯のdeadline | 30 s |
| stop strings | 最大4 |
| 反復period | 最大16 tokens |
| 思考反復run | 256 + period検査用16 tokens |
| 回答反復run | 1024 + period検査用16 tokens |
| 思考の連続digit | 1000 tokens |

Host/Origin/Fetch保護、任意API key、4種のsecurity headersを実装。
CORS/TLS/file serving/外向きHTTP/要求内容のheader出力やログはない。

## 未完・未検証

- **必須テストゲートは未達**。bind許可のある環境で、指定のworkspace/releaseテストと
  実ソケット25件・CLIサーバー2件を通す必要がある。本環境で成功扱いにはできない。
- 実TCPでのheader/idle timeout、同時接続上限、断線、重複要求、各HTTP応答は未検証。
  これらを試すテストは通常Cargo/CIで実行される形で残してある。
- 今回の `cargo test --workspace` はCLI失敗で停止。他crate全体の実行結果は得ていない。
- 実モデルtokenizer/重み・実推論・性能・bench/selfcheckは未実施。
- 実対話端末でのCtrl+C操作は未検証。pipeによるchat表示/コマンド/EOFは検証済み。
