# Measurement conditions and records

日本語は [MEASUREMENTS.ja.md](MEASUREMENTS.ja.md)。

The conditions, the alternative tallies and the limits of the numbers in [README.md](../README.md). The content is based on the same
records as the section "技術情報と測定条件" of the CAMPFIRE project page. The launch settings, prompts, measurement scripts and the raw
records themselves are in [`bench/`](../bench/).

## Which builds were measured

* The speed tables are the saved records of the quality model (12.37 GB) and the fastest model (10.31 GB), measured on November 8, 2026, and of
  the eight other engines. YASAKANI's engine is the `yskn.dll` built on November 7
  (sha256 `0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb`), the same file that this release distributes.
* The quality table evaluates the same two models with an FP16 key-value (KV) cache. That is not the KV condition of the speed tables, which
  use KV5. YASAKANI's two rows of that table were measured with a research build of the engine, which can use an FP16 KV cache and return
  the distribution over the whole vocabulary; the weights are the same data as in the released model files. The `yskn.dll` of this
  release was not used for the quality table.
* "16.78 GiB" comes from the long-context test of the fastest model.
* The code-edit demonstration (the uncut screen recording) was made with the fastest model on November 8: 5.56 s from sending the input to the
  arrival of the last output.

## Conditions and tallies of the speed tables

One RTX 5090 at its stock 600 W, no overclocking. One request at a time on the same PC. The tasks are 12 published prompts (code, story,
translation and structured output, 3 each) and 3 additional tasks (code edit, JSONL edit, long JSONL). The output limit is 4,096 or 8,192
tokens.

Thinking off is greedy, 2 runs each. The table in the README is the average after runs with a repetition loop or an unrequested tool call are
left out by one saved rule shared by all engines. The exclusion can change the number of tasks in a category. For the fastest model's "Code",
the value is 900 tok/s before the exclusion and 529 tok/s after it. We do not use the former as a representative figure for practical code
generation. The table before the exclusion is kept below.

Thinking on uses temperature 1.0, top-p 0.95, top-k 20, min-p 0. It is a comparison of records: the fastest run of each task is chosen and
averaged per category. YASAKANI's value is chosen from 5 runs and the other engines' from 2, so the number of tries is not equal. The table
tallied by plain averages is also given below. Many runs ended at the output limit, so this is not a comparison of the time to reach an answer
or of accuracy.

Speed is (output tokens - 1) / (time from the first to the last content chunk). Reading the input is not included. Weights, quantization, KV
cache and speed-up settings are each engine's own; this is not a comparison with every setting made equal. For an engine measured with
several settings, the fastest setting of each category is used.

The engines compared are MegaCapybara 1.39 Tiny, mkyt v0.3.2, ignis v0.9.6, the November 2 build of NInfer, vLLM 0.31.0, SGLang 0.5.22,
llama.cpp b11700 and TensorFold 0.6.6. MegaCapybara with thinking off was measured on November 3; its thinking-on runs and the other engines on
November 6. This does not rank the latest version of every program, or every use.

## Thinking off: the table of the README (average without repetition loops etc., tok/s)

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 529 | 285 | 494 | 909 | 1,222 | 1,156 | 1,260 |
| YASAKANI quality model | 486 | 179 | 483 | 930 | 1,136 | 1,151 | 1,160 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 204 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 150 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 139 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

## Thinking on: the table of the README (fastest run of each task, tok/s)

YASAKANI's values are chosen from 5 runs, the others' from 2.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 337 | 287 | 497 | 536 | 667 | 1,108 | 704 |
| YASAKANI quality model | 299 | 242 | 428 | 480 | 743 | 1,053 | 634 |
| MegaCapybara 1.39 Tiny | 290 | 206 | 308 | 429 | 550 | 663 | 516 |
| mkyt v0.3.2 | 244 | 196 | 348 | 384 | 378 | 603 | 454 |
| ignis v0.9.6 | 211 | 130 | 276 | 277 | 278 | 485 | 327 |
| NInfer (November 2 build) | 172 | 140 | 258 | 263 | 263 | 404 | 322 |
| vLLM 0.31.0 | 131 | 112 | 197 | 195 | 187 | 369 | 253 |
| SGLang 0.5.22 | 157 | 124 | 206 | 230 | 257 | 402 | 282 |
| llama.cpp b11700 | 140 | 122 | 184 | 197 | 206 | 349 | 221 |
| TensorFold 0.6.6 | 134 | 79 | 167 | 183 | 439 | 360 | 268 |

## Alternative tally: thinking off, average before the exclusion (tok/s)

These values include the runs with repetition loops and unrequested tool calls.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 900 | 285 | 494 | 909 | 1,222 | 1,156 | 1,260 |
| YASAKANI quality model | 486 | 263 | 483 | 930 | 1,136 | 1,151 | 1,160 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 438 | 204 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 282 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 312 | 150 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 270 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 143 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

## Alternative tally: thinking on, average of all runs (tok/s)

Here too YASAKANI has 5 runs and the others 2. Both tables are kept so that the table of records and the table of averages are not confused.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 324 | 273 | 467 | 471 | 627 | 1,073 | 657 |
| YASAKANI quality model | 282 | 234 | 407 | 432 | 626 | 1,013 | 568 |
| MegaCapybara 1.39 Tiny | 285 | 192 | 298 | 418 | 508 | 662 | 508 |
| mkyt v0.3.2 | 242 | 193 | 340 | 361 | 357 | 597 | 448 |
| ignis v0.9.6 | 208 | 128 | 268 | 273 | 270 | 472 | 316 |
| NInfer (November 2 build) | 167 | 136 | 245 | 252 | 254 | 396 | 280 |
| vLLM 0.31.0 | 128 | 109 | 192 | 189 | 168 | 365 | 246 |
| SGLang 0.5.22 | 153 | 122 | 202 | 217 | 250 | 400 | 274 |
| llama.cpp b11700 | 136 | 122 | 180 | 184 | 197 | 346 | 219 |
| TensorFold 0.6.6 | 127 | 77 | 152 | 175 | 320 | 356 | 232 |

## Quality on thinking text

Twenty thinking transcripts of our own, not used for the statistics with which the models were made, were used as windows of 2,048 tokens
each; the last 1,023 positions of every window (20,460 positions in total) were compared with the same reference, the original BF16 model. The
evaluation covers our own thinking text, and we do not generalize it to quality for every use or to the rate of correct answers on tasks.

| Engine / model | File size, GB | KV cache | KL on thinking text (smaller is closer) | Top-1 agreement (higher is closer) |
|---|---|---|---|---|
| YASAKANI quality model | 12.37 | FP16 | 0.0122 | 95.24% |
| NInfer, official NVFP4 | 23.72 | FP8 | 0.0178 | 94.67% |
| Unsloth UD-Q3_K_XL (llama.cpp) | 12.80* | FP16 | 0.0204 | 94.32% |
| llama.cpp, GSQ-RCO IQ3_S | 12.12 | FP16 | 0.0229 | 93.64% |
| SGLang, NVIDIA NVFP4 | 21.92 | FP8 (e4m3) | ≥0.0283 | 92.70% |
| YASAKANI fastest model | 10.31 | FP16 | 0.0304 | 93.01% |
| MegaCapybara, Tiny | 16.59 | 6/4 | 0.0321 | 92.80% |
| TensorFold, MLX 4-bit | 16.05 | BF16 | 0.0353 | 92.05% |
| mkyt, gdnfp4 | 19.50 | NVFP4 | ≥0.0362 | 92.17% |
| Unsloth UD-IQ3_XXS (llama.cpp) | 10.93 | FP16 | 0.0458 | 91.32% |
| ignis, NVFP4full-v2 | 19.41 | hq-e8-2b | 0.0473 | 91.09% |
| vLLM, NVFP4 LMHead4 | 18.77 | FP8 | ≥0.0509 | 90.22% |
| llama.cpp, GSQ-RCO IQ3_XXS | 10.44 | FP16 | 0.0518 | 91.22% |

KL is the divergence of the predicted distribution; top-1 agreement is how often the most likely next token is the same. The quality model's
KL is 0.0122198812 and that of the GSQ-RCO IQ3_S (llama.cpp) configuration is 0.02288, so the former is 46.59% smaller. The KV cache of YASAKANI
and of llama.cpp is FP16.

For SGLang, mkyt and vLLM the KL is that of a distribution made of the top 100 tokens (64 for mkyt) plus everything else as one bucket, which
is a lower bound of the KL over the whole vocabulary. SGLang's lower bound of 0.0283 alone cannot settle its order against the fastest model's
0.0304. We did not use the approximation gap observed on YASAKANI as an error bound for the other engines. Because the quality model's value
is smaller than every configuration's lower bound, the conclusion that it is the smallest of the thirteen listed configurations stands.

For MegaCapybara the distribution over the whole vocabulary was obtained at all 20,460 positions, giving KL 0.0321 and top-1 agreement 92.80%
against the same BF16 reference. In a check of the read-out, the top candidate of the distribution agreed with the server's actual reply at
all 20,289 positions where the two could be compared. That number of positions belongs to this check and is separate from the 20,460 positions
of the quality figures. The feature that lowers the probability of certain words during thinking (thinking penalty) was turned off so that the
model's own distribution was evaluated.

The KV format is as given in each row. YASAKANI is evaluated on its verification path in units of 16 rows; the other configurations mostly on
their prompt-processing path. SGLang's row in the quality table is NVIDIA NVFP4, not the LMHead4 configuration of the speed tables. Read the
table as a comparison of whole configurations (weights, KV cache and arithmetic), not as a ranking of weight-quantization methods alone.

Sizes are in GB (10^9 bytes) and are, as a rule, the whole model file that is loaded. Only UD-Q3_K_XL, marked *, shows the size of the main
model without its auxiliary layers. Some configurations include a vision part or auxiliary weights, so the sizes are not the compression
ratio of the main model alone, nor the VRAM needed. The GSQ-RCO files used here include about 0.35 GB of auxiliary layers; the main model
alone is 11.77 GB for IQ3_S and 10.09 GB for IQ3_XXS. As that main model is itself smaller than YASAKANI's, we do not describe this as a
compression comparison at strictly equal size. In an additional evaluation on general text, the new conversion of the quality model showed a
KL 3-10% larger than the earlier conversion in some results; this is stated together with the result on thinking text.

The two Unsloth configurations were also measured with llama.cpp and an FP16 KV cache. The 12.80 GB* of UD-Q3_K_XL is the main model without
auxiliary layers; the whole file is 13.15 GB. For UD-IQ3_XXS the whole file, 10.93 GB, is shown; the main model is 10.58 GB.

The score of every configuration, the logs, the scoring scripts and how the distribution of each engine was obtained are in
[`bench/20261108/quality/`](../bench/20261108/quality/), together with two more Unsloth files that are not in the table above. The twenty
transcripts and the reference are not published, so the table cannot be measured again from the records alone.

## KV5 (the compressed key-value cache) with the fastest model

With the same 10.31 GB weights, four fixed sequences of 131,072 tokens (thinking and answers joined) were evaluated. The last 1,023 positions
at each context length, 4,092 positions in total, were compared with the BF16 reference. The input is fixed text that the engine reads; it is
not an independent corpus that was never used before.

The FP16 KV cache was measured with a research build and KV5 with a distribution build, so the difference includes the difference of
implementation paths. Over the 7 bands from 2K to 128K, the mean KL of KV5 minus that of FP16 was about 0.000034 to 0.000516. At 128K, FP16
was 0.02646 and KV5 0.02687; the mean difference at the same positions was +0.000412 (95% interval +0.000177 to +0.000674). We do not treat
a small difference as proof of equivalence.

With 256K reserved, the engine's memory is 27.12598 GiB with FP16 and 16.78320 GiB with KV5. Per token the KV cache takes 64.25 KiB against
21.875 KiB, about 66% less; for the whole engine it is about 38% less. These are the amounts reserved at start; they are not the peak during
generation, not the whole setup with the helper model, and not a measurement of quality at 256K. The quality of this model with an INT8 KV
cache has not been measured.

## Supported environment

This release covers text chat with Qwen3.8-27B on Windows 11 and the RTX 5090. Image input is future work. Tool calling through the API is a preview: it is in the package that comes with the harness, not in the 0.1.0 package.
Which configurations and context lengths work on 24 GB and 16 GB GPUs will be checked on each actual card; operation is not promised from
the file size alone, and the conversation history and working memory are included in that check. The executables are not code-signed, so
Windows may show a warning.

## The uncut screen recording of November 8 (fastest model)

The same fastest model as in the comparison tables (10.311344128 GB) and the same `yskn.dll` as this release were used. One RTX 5090 at stock
600 W, maximum context 262,144, thinking off, temperature 0, output limit 8,192. The target input was sent once to a new process that had
been warmed up twice with a different short text. The prompt cache was empty.

In the 19:22 recording with a clock, the first output arrived after 1.281498 s, the last output after 5.555668 s, and the transfer completed
after 5.556058 s. The output was 4,712 tokens and the generation speed 1,102.202299 tok/s. Speed is (output tokens - 1) / (arrival of the
last output - arrival of the first). These are values measured while the screen was recorded, the clock was displayed and the CPU was
encoding; they do not replace the values of the speed tables. The effect of the recording load was not measured separately. The earlier
recording of 19:01 is also kept as an original record.

The generated file matched the expected file (apart from the final newline). It passed the syntax comparison and the behaviour comparison on
17,701 inputs, with 0 mismatches against the original. Python's own standard test suite was not run because it is not available in the
environment. The result of this one task is not treated as a general rate of correct code generation.

The video is a direct recording of one screen that shows the application connected to the API and a command prompt running as a separate
process. 1800x1000, 30 fps, no sound, 17.000 s, 510 frames. No cuts, no speed-up and no composited numbers were applied after recording, and
no dropped frames were detected. JST is the current time, ELAPSED the seconds running on from the start of the clock, REQUEST the seconds
from sending to completion. The clock log, the receive log and the hashes of the original recording, the output and the verification results
can be checked against each other. The clock is that of the same PC, and these records do not replace a third party's presence or
replication.

## Checks on this release (November 10, 2026)

Checked with the files that are published: the `yskn.dll` above, the Python package and the two model folders.

* `yasakani verify`: in both folders every file matches the checksums in `yasakani.json`.
* `yasakani selfcheck`: the reference (`selfcheck.json`) was written, then reproduced in a new process with the default context length and
  with 16,384 (both models).
* `yasakani bench` (256 tokens x 2 runs; explain / code / story / proof): fastest model 358 / 600 / 239 / 633 tok/s, quality model
  353 / 575 / 203 / 480 tok/s.
* The same request repeated (`repeat_probe.py` in `bench/`: one engine in one process, the same request generated 8 or 12 times, 256 tokens
  each, greedy and seed 1234, with the default settings and with speculative decoding switched off, one token per pass):
  * with nothing else using the GPU (utilisation 0% for 30 s before the start): **240 of 240 runs** gave the same tokens as the other runs
    of their group (fastest model; 48 runs with the default settings, 192 with speculative decoding off);
  * while other programs were using the GPU (the desktop was drawing; utilisation 3-5% before the start): with the default settings
    1 of 190 runs differed from the most common output of its group, with speculative decoding off 21 of 432 (both models; none of 16 further runs with only part of it switched off). A run that differs
    has the same tokens up to one position and continues differently from there; where the same position came up more than once, it was
    the same pair of tokens each time. While another process used the GPU without pause, it was 8 of 60 runs;
  * the three requests of `yasakani selfcheck` were identical in 180 of 180 repeats, and the command passed 12 of 12 times in new
    processes (both while other programs were using the GPU).
* Speculative decoding on against off (`onoff_eq.py` in `bench/`; 34 inputs, greedy and seed 1234, 62 cases, on / three kinds of off / on
  again; run while other programs were using the GPU): fastest model 171 of 174 comparisons identical, quality model 170 of 174. The
  differences: fastest model - one greedy case (first different token #143) and two sampled cases (#152, #97); quality model - one sampled
  case (#110) and one greedy case in which all three kinds of off differed from the first run at token #67. A repeated run with the same
  settings differed from the first run in one case per model (#115; #67). With speculative decoding off the engine needed at least 1.5 times as
  many passes in all 62 cases, so "off" really ran differently.
* What this means: when nothing else uses the GPU, the same request gave the same tokens every time. When other programs use the GPU at the
  same time, a token now and then comes out differently, more often in plain decoding (one token per pass) than with the default settings.
  We have not found the cause yet. It is not the hardware: the same computations repeated in PyTorch on the same GPU and compared bit for
  bit (about 540,000 repeats, including four minutes at the 600 W power limit with two processes at once) showed no difference. Until the
  cause is found, this release does not promise bit-identical output from run to run.
