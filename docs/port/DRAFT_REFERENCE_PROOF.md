# Brief 05d 実装・検証報告（2026-11-14）

作業場所は `.work/wt/05d`、ブランチは `feat/05d-draft-ref`。
実装・テスト・仕様更新のコミットは `342064482685ff45b3da0f0e4901d84271709575`。
この報告書は後続の証跡コミットで追加（そのハッシュは最終回答に記載）。
pushは行っていない。ネットワーク、新規依存、メインモデルの重みの読み込みは行っていない。

## 変更ファイル

- `crates/yskn-ref/examples/draft_info.rs`
- `crates/yskn-ref/src/draft.rs`
- `crates/yskn-ref/src/draft_head.rs`
- `crates/yskn-ref/src/lib.rs`
- `crates/yskn-ref/src/output_head_sampler.rs`
- `crates/yskn-ref/tests/draft.rs`
- `crates/yskn-ref/tests/support/draft.rs`
- `crates/yskn-ref/tests/support/mod.rs`
- `docs/port/DECISIONS.md`
- `docs/port/README.md`
- `docs/spec/07_DRAFT_MODEL.md`
- `docs/spec/10_AUXILIARY_FILES.md`
- `docs/spec/OPEN_QUESTIONS.md`
- `docs/port/DRAFT_REFERENCE_PROOF.md`（本書）

`draft.rs`はロード、共通符号による圧縮演算、コンテキスト投影、BF16リング、
正規化／RoPE、非因果8行ブロック、補正、snapshot/restore/resetを実装。
`draft_head.rs`はSHEAD検証、低ランク投影、近似スコア、ヒストグラム選抜、
int8再採点、top16、貪欲／サンプリング選択を実装した。
既存コードの変更はlib.rsのpub mod追加、Gumbelヘルパーのpub(crate)化、
合成writerサブモジュールの追加。Gumbel本体は変更していない。
禁止対象のmodel/attention/GDN/KV/quality、他crate、src/、.github/、STATE.mdは変更していない。

## 必須proof

全コマンドはworktree rootで `CARGO_NET_OFFLINE=true` を設定して実行し、終了コード0。
workspaceの通常テストでは `YSKN_MODEL_DIR` を未設定にし、最後のコマンドだけで実ファイルを使用した。
通常テストの実ファイル分岐は仕様どおりskipしてpassする（ignoredカウントではない）。
最終コマンドではその分岐も実際に完走した。

| コマンド | 結果 | 実行時間 |
|---|---|---|
| `cargo fmt --all --check` | PASS（テスト件数なし） | 0.156 s |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS（テスト件数なし） | 0.244 s |
| `cargo test --workspace` | 113 passed / 0 failed / 0 ignored | 42.637 s |
| `cargo test --workspace --release` | 113 passed / 0 failed / 0 ignored | 40.480 s |
| `cargo run --release -p yskn-ref --example draft_info -- "$HOME/models/qwen38-speed"` | PASS（テスト件数なし） | 0.261 s |
| `YSKN_MODEL_DIR="$HOME/models/qwen38-speed" cargo test -p yskn-ref --release -- --nocapture draft` | 14 passed / 0 failed / 0 ignored | 188.081 s |

### `cargo fmt --all --check` の末尾

```text
（出力なし。exit 0）
```

### `cargo clippy --workspace --all-targets -- -D warnings` の末尾

```text
    Checking yskn-ref v0.1.0 (/Users/anag2bgod/Projects/yasakani-for-mac/.work/wt/05d/crates/yskn-ref)
    Finished `dev` profile [optimized + debuginfo] target(s) in 0.18s
```

### `cargo test --workspace` の末尾

```text
   Doc-tests yskn_format

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests yskn_ref

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

全テストバイナリ／doctestの件数（出力順）:

```text
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.68s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 24.57s
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 49 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.64s
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.89s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

### `cargo test --workspace --release` の末尾

```text
   Doc-tests yskn_format

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests yskn_ref

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

全テストバイナリ／doctestの件数（出力順）:

```text
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.11s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.95s
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 49 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.77s
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 6.93s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

### `cargo run --release -p yskn-ref --example draft_info -- "$HOME/models/qwen38-speed"` の末尾

```text
vector bdraft2.sel.pred n=63569920 dtype=Bf16
vector bdraft2.sel.succ n=63569920 dtype=Bf16
fc input signs=25600 all matrix signs equal its prefix; all output rotations absent
shead: /Users/anag2bgod/models/qwen38-speed/bdraft2_k8.shead
V=248320 H=5120 rank=512 fourth_word=0 bytes=1411010584
projection_f32: offset=24 bytes=10485760
approx_i8: offset=10485784 bytes=127139840
approx_scale_f32: offset=137625624 bytes=993280
rescore_i8: offset=138618904 bytes=1271398400
rescore_scale_f32: offset=1410017304 bytes=993280
correction BF16 vectors: hp=[256,5120], pred/succ=[248320,256]
PASS: header dimensions, shapes, sign prefixes, rotations and SHEAD array extents
```

### `YSKN_MODEL_DIR="$HOME/models/qwen38-speed" cargo test -p yskn-ref --release -- --nocapture draft` の末尾

```text
     Running tests/quality.rs (target/release/build/yskn-ref/8a2fd09e1f71e4e9/out/quality-8a2fd09e1f71e4e9)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.00s

   Doc-tests yskn_ref

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

全テストバイナリ／doctestの件数（出力順）:

```text
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 46 filtered out; finished in 0.00s
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 187.90s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 16 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

## `draft_info` 全出力

```text
    Finished `release` profile [optimized] target(s) in 0.01s
     Running `target/release/examples/draft_info /Users/anag2bgod/models/qwen38-speed`
draft: /Users/anag2bgod/models/qwen38-speed/bdraft2_k8.yskn
bytes=1164969472 header_bytes=21809 data_offset=8388608
model: {"arch":"bdraft2","block":8,"eps":1e-6,"ffn":17408,"head_dim":128,"heads":32,"hidden":5120,"kv_heads":8,"layers":5,"mask_token":248070,"rank":256,"rope_theta":10000000.0,"source":"Drafter for Qwen3.8-27B, converted to the YASAKANI format (see NOTICE)","sub_vocab":0,"target_layers":[5,19,33,47,61],"tie_embeddings":false,"top_k":16,"vocab":248320,"window":2048}
tensor bdraft2.fc 5120x25600 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 50, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.akp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.down 5120x17408 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 34, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.gate 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.up 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.mkp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.k 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.o 5120x4096 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 8, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.q 4096x5120 physical=4096 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.0.v 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.akp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.down 5120x17408 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 34, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.gate 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.up 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.mkp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.k 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.o 5120x4096 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 8, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.q 4096x5120 physical=4096 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.1.v 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.akp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.down 5120x17408 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 34, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.gate 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.up 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.mkp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.k 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.o 5120x4096 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 8, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.q 4096x5120 physical=4096 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.2.v 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.akp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.down 5120x17408 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 34, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.gate 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.up 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.mkp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.k 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.o 5120x4096 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 8, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.q 4096x5120 physical=4096 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.3.v 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.akp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.down 5120x17408 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 34, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.gate 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.up 17408x5120 physical=17408 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.mkp 1280x5120 physical=1280 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.k 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.o 5120x4096 physical=5120 bits=8 layout=Warp rot_in=Rotation { blocks: 8, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.q 4096x5120 physical=4096 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
tensor bdraft2.layers.4.v 1024x5120 physical=1024 bits=8 layout=Warp rot_in=Rotation { blocks: 10, sign_seed: 7, sign_offset: 0, rule: Mulberry32 } rot_out=None
vector bdraft2.hidden_norm n=5120 dtype=Bf16
vector bdraft2.norm n=5120 dtype=Bf16
vector bdraft2.layers.0.ln1 n=5120 dtype=Bf16
vector bdraft2.layers.0.ln2 n=5120 dtype=Bf16
vector bdraft2.layers.0.q_norm n=128 dtype=Bf16
vector bdraft2.layers.0.k_norm n=128 dtype=Bf16
vector bdraft2.layers.0.abase n=20480 dtype=Bf16
vector bdraft2.layers.0.mbase n=20480 dtype=Bf16
vector bdraft2.layers.1.ln1 n=5120 dtype=Bf16
vector bdraft2.layers.1.ln2 n=5120 dtype=Bf16
vector bdraft2.layers.1.q_norm n=128 dtype=Bf16
vector bdraft2.layers.1.k_norm n=128 dtype=Bf16
vector bdraft2.layers.1.abase n=20480 dtype=Bf16
vector bdraft2.layers.1.mbase n=20480 dtype=Bf16
vector bdraft2.layers.2.ln1 n=5120 dtype=Bf16
vector bdraft2.layers.2.ln2 n=5120 dtype=Bf16
vector bdraft2.layers.2.q_norm n=128 dtype=Bf16
vector bdraft2.layers.2.k_norm n=128 dtype=Bf16
vector bdraft2.layers.2.abase n=20480 dtype=Bf16
vector bdraft2.layers.2.mbase n=20480 dtype=Bf16
vector bdraft2.layers.3.ln1 n=5120 dtype=Bf16
vector bdraft2.layers.3.ln2 n=5120 dtype=Bf16
vector bdraft2.layers.3.q_norm n=128 dtype=Bf16
vector bdraft2.layers.3.k_norm n=128 dtype=Bf16
vector bdraft2.layers.3.abase n=20480 dtype=Bf16
vector bdraft2.layers.3.mbase n=20480 dtype=Bf16
vector bdraft2.layers.4.ln1 n=5120 dtype=Bf16
vector bdraft2.layers.4.ln2 n=5120 dtype=Bf16
vector bdraft2.layers.4.q_norm n=128 dtype=Bf16
vector bdraft2.layers.4.k_norm n=128 dtype=Bf16
vector bdraft2.layers.4.abase n=20480 dtype=Bf16
vector bdraft2.layers.4.mbase n=20480 dtype=Bf16
vector bdraft2.sel.hp n=1310720 dtype=Bf16
vector bdraft2.sel.pred n=63569920 dtype=Bf16
vector bdraft2.sel.succ n=63569920 dtype=Bf16
fc input signs=25600 all matrix signs equal its prefix; all output rotations absent
shead: /Users/anag2bgod/models/qwen38-speed/bdraft2_k8.shead
V=248320 H=5120 rank=512 fourth_word=0 bytes=1411010584
projection_f32: offset=24 bytes=10485760
approx_i8: offset=10485784 bytes=127139840
approx_scale_f32: offset=137625624 bytes=993280
rescore_i8: offset=138618904 bytes=1271398400
rescore_scale_f32: offset=1410017304 bytes=993280
correction BF16 vectors: hp=[256,5120], pred/succ=[248320,256]
PASS: header dimensions, shapes, sign prefixes, rotations and SHEAD array extents
```

## 次のmain-model hookに渡す契約

- 層はゼロ始まり **5, 19, 33, 47, 61**、この順序（`src/engine35.cpp:1650`）。
- 各層のMLP down投影の残差加算が済んだ直後、次層または最終正規化より前の
  `h_` を取る。`layer()`完了後のコピーは `src/engine35.cpp:798–799,830–833`。
  attention/GDNの残差枝も既にその値へ反映済み。
- 丸めは `bf(old_residual + bf(down_projection))`、RNE BF16をFP32へ正確に展開した値。
  元の配列がfloatでも未丸めのFP32残差を渡してはいけない。
  実際の残差更新・保存は `src/q35_kernels.cu:1437–1444`。
  正規化、gain、回転を追加しない。次投影向けの正規化コピーも渡さない。
- 1行の形状は `[capture_slot=5, hidden=5120]`、slot-majorの連結25,600要素。
  元のcaptureは `[slot,row,channel]`。row-majorに読み替えず、指定行をgatherする
  （`src/q35_bdraft.cu:249–252,490–499`）。バッチAPIはgather後の `[row,slot,channel]`。
- promptおよび採用されたverification行だけを、連続した絶対位置pで一度ずつ投入する。
  未採用行と、まだ未確定のfrontier tokenは投入しない。
  各draft層は同じ正規化contextをK/Vへ投影する（`src/engine35.cpp:1778–1782`）。
- main/draftの隠れ幅、epsilon、指定capture層の存在を呼び出し側の接続時に照合する。
  このbriefではmain側hook自体は実装していない。

## 仕様とソースの照合

chapter 07 §§1–8の計算式とchapter 10のSHEAD配列配置に、訂正を要する矛盾は見つからなかった。
次を明確化／更新した。移植方針をupstreamの規定と混同していない。

| 更新 | 根拠 |
|---|---|
| `sel.hp/pred/succ`は圧縮行列ではなくBF16 dense vectors | `src/engine35.cpp:1636–1637`; `src/q35_bdraft.cu:299–305,406–411` |
| captureの正確なtap、BF16丸め、shape | `src/engine35.cpp:798–799,830–833,1650`; `src/q35_kernels.cu:1437–1444`; `src/q35_bdraft.cu:249–252` |
| DRF-02解決、46個の8-bit warp行列／35 BF16 vectors／符号prefix一致／出力回転なし | 上記の実ファイルheader全文と生成符号照合。upstreamの利用箇所は `src/engine35.cpp:1638`; `src/q35_bdraft.cu:474` |
| DRF-01はcutoff bucketをID昇順で埋めるD15の明示判断 | upstreamはatomic順: `src/q35_bdraft.cu:882–896,958–962` |
| DRF-03はportable mathを使い、明示FMAだけを残すD15の判断 | native math箇所: `src/q35_bdraft.cu:145–148,190,426,472,530–536,579,615` |
| SHEADのchecked extent/finite検証、manifest containmentはport側の追加契約 | upstream SHEAD `src/engine35.cpp:1705–1734`; manifest `python/yasakani/runtime.py:123–138` |

## 明示fmaf全箇所

すべて `src/q35_bdraft.cu`。同じ演算の旧／非融合カーネルは共通のscalar実装へ対応させた。
これ以外の積と和は分離した丸めを維持する。

| Lines | Purpose / port handling |
|---|---|
| 34, 465 | 256-lane RMS squares (unfused/fused variants); one port norm |
| 141 (three nested calls) | Four-channel block dot, scaled Q before dot |
| 146, 148 | Block online denominator and numerator |
| 175 (two nested calls) | Serial history dot, even then odd, scale after dot |
| 205 | Ascending-key history numerator |
| 226 (two), 579 (two) | Unfused/fused-output split combination; one port combiner |
| 303 | Dense BF16 hidden-to-correction projection |
| 334, 409 | Legacy/current predecessor correction; one port correction |
| 786 (four) | Rank projection, block then u then channel |
| 839 | Approximate score, contiguous 16 components per lane |
| 920–921 (four) | Full-width int8 rescoring |

## 実ファイルの時間・メモリ

- load: 0.790242583 s（実試験出力）。合成context 1行と合成BF16 embeddingを使用。
- 8行／7提案の1回目 **87.298883 s**、2回目 **98.666529 s**。
- 提案ID `[229705, 64687, 168762, 3116, 51527, 65160, 12203]`、候補ID／raw score bits／corrected score bitsを含む
  FNV-1a digestは両回 `4e5f96eb8f632494`。配列全体の等値も確認済み。
- 最後のcargoコマンドのDarwin `wait4` rusage最大RSSは **1,598,291,968 bytes**
  （1.598 GB、1.489 GiB）。
  cargoとその子のコマンド全体を観測し、ロードと2回のproposeを含む値。
  propose単体の追加割り当て量やmacOSの圧縮メモリ量を分離測定した値ではない。
- headerの要素数から数えた常駐配列は **2,870,082,560 bytes**
  （2.870 GB、2.673 GiB、Vec管理領域等を除く）。
  RSSと論理的な所有bufferサイズは異なる。大きなdense行列へ展開していない。
  ロード中にBF16 vectorの一時byte配列（最大約127 MB）が加わり、計算中の作業配列は数MB。
  論理bufferも4 GB以内。最適化は行っていない。
- 別のCPU-heavy jobが動作中の測定であり、製品性能を示すものではない。

所有buffer内訳（サイズはheadersから計算）:

```json
{
  "compressed_streams": 897843200,
  "tables_f32_and_i8": 1884160,
  "scales_original_and_effective": 2191360,
  "per_matrix_signs": 312320,
  "dense_vectors_f32": 514872320,
  "shead_arrays": 1411010560,
  "ring": 41943040,
  "common_sign_copy": 25600,
  "total_bytes": 2870082560
}
```

## テストの根拠と未実施範囲

合成テストはpublic loader/APIを通して、不正header／offset／array length／shape／型／
非有限値／options／pathを拒否すること、ring insertion/overwrite/batch一致、reset、
snapshot/restoreと失敗行の非変更、同じ提案のbit一致を確認する。
独立した期待値として、uniform attentionの平均とhistory窓、未来row 7の影響、
branch別dynamic filter係数、単一非ゼロ要素のSHEAD投影／再採点、prefix cutoffとID tie、
greedy index tieとsampled top-p／token-key permutationを検証した。
各行列の独自sign/output rotationを変えた同一モデルの比較も実施した。
固定digestは合成モデルgreedy `2d560f2d9bae0fa4`、sampled `540335f09a95f09c`。
これらは本portの固定値であり、CUDA goldenではない。

未実施／実装対象外はmain-model capture hookの接続、speculation/verification loop、
16行ブロック、無効な代替compressed/subset head、第二・第三side候補／margin／HCAP policy。
既存top16とraw/corrected scoresは公開済み。CUDA bit一致、受理率／quality、
主モデルとのend-to-end出力一致は検証していない。DRF-04のmain compressed headとの
等値も未検証。指定に従い `qwen38_27b_speed.yskn` は開いていない。
