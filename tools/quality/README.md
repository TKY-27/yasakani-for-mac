# The quality measurement, so that you can repeat it

日本語：品質の表（[docs/MEASUREMENTS.ja.md](../../docs/MEASUREMENTS.ja.md) の「思考文の品質比較」）を自分で測り直すための一式です。評価に使った 20 本の文章、
元のモデルから基準を作るプログラム、公開版のエンジンで配布モデルを測るプログラム、私たちの結果が入っています。

The quality table in [docs/MEASUREMENTS.md](../../docs/MEASUREMENTS.md) ("Quality on thinking text") compares the next-token
distribution of a compressed model with that of the original model on 20 windows of text. This folder holds what is needed to
measure it again.

| | |
|---|---|
| [`windows_own_thinking.json`](windows_own_thinking.json) | the 20 windows: the token ids (2,048 each) and, for reading, the decoded text. Each is a prompt and the thinking text the model wrote for it. They were not used for the statistics of the conversion |
| [`make_reference.py`](make_reference.py) | the reference: runs the original model (Hugging Face transformers) over the windows and writes its next-token distribution at the positions 1024 to 2046 of every window |
| [`score_yasakani.py`](score_yasakani.py) | measures a released model against a reference file, with the engine library (`yskn.dll`) |
| [`records/`](records/) | the results below as files |

The formulas (KL divergence, top-1 agreement, perplexity) and the format of the reference file are those of llama.cpp's perplexity
tool; they are in [`bench/20261108/quality/scripts/kld_base.py`](../../bench/20261108/quality/scripts/kld_base.py).

## Steps

1. **The reference**, from the original model `Qwen/Qwen3.8-27B` (about 54 GB; the output is about 10 GB):

       python tools/quality/make_reference.py --original <folder or id of the original model> --out own_reference.kld --offload

   `--offload` keeps the weights in main memory (about 60 GB) and runs every block on one GPU. Without it, transformers spreads
   the model over the devices it finds.

2. **A released model with the released engine** (Windows, RTX 5090, the `yasakani` package installed):

       python tools/quality/score_yasakani.py --model-dir <model folder> --dll <yskn.dll> --reference own_reference.kld --out score.json

3. **Other engines** on the same reference file: `llama_kld.py` (llama.cpp) and `api_topk_dump.py` with `topk_score.py` (servers) in
   [`bench/20261108/quality/scripts/`](../../bench/20261108/quality/scripts/).

## What we get

Measured on November 14, 2026 with the engine of release 0.1.1 (`yskn.dll`, SHA-256 `0f5d973b...50bbb`) on one RTX 5090.
20 windows, 20,460 positions.

| Model | Key-value cache | Reference | KL divergence | Top-1 agreement | Perplexity (of the reference) |
|---|---|---|---|---|---|
| quality | KV5 (as the package starts it) | the one of the table | 0.012591 | 95.347% | 2.0686 (2.0652) |
| quality | 8-bit, without the KV5 compression | the one of the table | 0.012503 | 95.235% | 2.0687 (2.0652) |
| fastest | KV5 (as the package starts it) | the one of the table | 0.030890 | 92.981% | 2.0759 (2.0652) |
| fastest | 8-bit, without the KV5 compression | the one of the table | 0.030728 | 93.065% | 2.0759 (2.0652) |
| quality | KV5 (as the package starts it) | made by `make_reference.py` | 0.012591 | 95.347% | 2.0686 (2.0652) |
| fastest | KV5 (as the package starts it) | made by `make_reference.py` | 0.030890 | 92.981% | 2.0759 (2.0652) |

The two references against each other, on all 20,460 positions: KL divergence less than 0.000001, the same most likely token at 100.000% of the
positions.

For comparison, the YASAKANI rows of the table in MEASUREMENTS.md (research build, FP16 cache): quality model 0.0122 and 95.24%,
fastest model 0.0304 and 93.01%.

## How this relates to the table in MEASUREMENTS.md

* The two YASAKANI rows of that table (KL 0.0122 with 95.24% agreement for the quality model, 0.0304 with 93.01% for the fastest
  model) were measured on November 8, 2026 with a research build of the engine that can use an FP16 key-value cache, so that the
  rows show the weights and not the cache. The released `yskn.dll` is built for its 8-bit cache only, and the published source
  builds only that configuration, so those two rows cannot be repeated with the released engine. What the released engine gives is
  in the table above: the row "KV5" is the engine as the package starts it.
* The reference file of that table was made on November 7, 2026 with our own implementation of the original model (bfloat16
  weights, the output matrix in 32-bit floats), not with transformers. It is about 10 GB and is not published. `make_reference.py`
  makes a reference with transformers instead; the last rows of the table above show how far the two references and the scores
  against them are apart.

## Limits

* Twenty windows of one kind of text: the model's own thinking. The result does not say how often a task is solved, and it does
  not carry over to other kinds of text ([docs/MEASUREMENTS.md](../../docs/MEASUREMENTS.md) has an additional result on web text).
* The windows start with a system prompt that the chat template writes for the longest thinking setting, then the user's
  request, then the thinking. Only the second half of every window is scored.
* `score_yasakani.py` reads the windows 16 tokens at a time through the path the engine uses to check drafted tokens. Nothing is
  generated, so the helper model has no part in the result.
