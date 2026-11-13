# bench/: measurement records

日本語は [README.ja.md](README.ja.md)。

This folder holds the measurement records of YASAKANI and of the engines it was compared with, all made on one PC with one NVIDIA
GeForce RTX 5090. The speed tables and the quality table of this release ([`README.md`](../README.md),
[`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md)) come from `20261108/` and its checks from `20261110/`; the two older folders are earlier comparisons with earlier builds, kept as they were.

| Folder | Date | What |
|---|---|---|
| [`20261108/`](20261108/) | November 6 and 8, 2026 | **The speed tables of this release**: the two released models (fastest and quality) against MegaCapybara 1.39, mkyt v0.3.2, ignis v0.9.6, NInfer, vLLM 0.31.0, SGLang 0.5.22, llama.cpp b11700 and TensorFold 0.6.6 on 15 tasks, thinking off and on. Launch commands, the scripts as run, the record and the answer of every run, server logs, and `scripts/tables.py`, which makes the tables again from the records. MegaCapybara's runs with thinking off are the ones of `20261103/`; the launch scripts of SGLang, vLLM and llama.cpp are the ones of `20261101/`. Also the records of the quality table (`quality/`), of KV5 at long context (`long_context/`) and the evidence package of the recorded demonstration (`demo/`). |
| [`20261110/`](20261110/) | November 10, 2026 | **Checks on the files of this release** (the `yskn.dll`, the Python package and the two model folders that are distributed): `yasakani verify`, `selfcheck` and `bench`, output with the speed-up on and off, the same request repeated, and a check of the GPU itself. |
| [`20261101/`](20261101/) | November 1, 2026 | An earlier comparison. YASAKANI (public build) against SGLang 0.5.22, vLLM 0.31.0 and llama.cpp b11700, ten tests. Launch commands, prompts, the measurement scripts and the record of every run. Also the YASAKANI-only checks of the same day (quality against the original model, AIME 2025 summary, long context, output with and without the speed-up, self-check). |
| [`20261103/`](20261103/) | November 3, 2026 | An earlier comparison. YASAKANI (standard and fast model files) against MegaCapybara 1.39 (Tiny model) on the same PC: NInfer's 12 public scenario prompts, three tests of our own, and the single recorded runs used for a video. Generation records (text and speed of every answer), server logs, client logs and GPU state. |

Measured values are as recorded. For publication, only paths, the user name, one e-mail address, two lists of engine settings,
comments in scripts and the text of third-party prompts were replaced or removed, and the job logs of `20261108/` were reduced to
their time and GPU lines (details under "Changes made for publication").
`SHA256SUMS.txt` lists every file of this folder.

## The PC

* GPU: one NVIDIA GeForce RTX 5090, 32 GB (32,607 MiB). Driver 616.92 (CUDA 13.4). On November 3 nvidia-smi recorded a power limit of
  600 W (the default) and maximum clocks of 3,090 MHz (SM) and 14,001 MHz (memory), see [`20261103/gpu/`](20261103/gpu/). The GPU lines
  of November 1 (temperature, SM clock, memory clock, power, memory used, memory total, utilisation) are inside the records (`gpu_start`,
  `gpu_before`, `gpu_after` in `A_pub.json`; `GPU [memory MiB, utilisation %]` in the driver logs).
* CPU: AMD Ryzen 9 9950X3D2. Memory: 96 GB. OS: Windows 11 Pro.
* Native on Windows: YASAKANI, llama.cpp, MegaCapybara, and on November 6 also mkyt and ignis (their Windows packages). SGLang and vLLM
  have no official Windows build; they ran in WSL2 (Ubuntu 24.04) on the same PC, and so did NInfer and TensorFold on November 6. We did
  not measure how much WSL2 costs compared with native Linux.
* November 6 and 8: nvidia-smi recorded the power limit (600 W), the maximum clocks (3,090 MHz and 14,001 MHz), the driver (616.92) and the
  temperature at the start of every job, and the power limit, temperature and memory at its end
  ([`20261108/logs/jobs/`](20261108/logs/jobs/)).
* Measurement clients: Python 3.14.4 on Windows. One request at a time; one server at a time, with nothing else using the GPU.

## Speed

Speed (tokens per second) = (output tokens - 1) / (time from the first token of the answer to the last). The time before the first
token, while the prompt is processed, is not included. "Output tokens" is the `completion_tokens` value the server reports (for
in-process runs, the number of generated tokens).

* Streamed runs (all runs of the other engines, YASAKANI on November 1, and all server runs on November 3): the arrival times of the first
  and the last streamed chunk at the client.
* In-process runs (YASAKANI on NInfer's 12 prompts on November 3, `pub_corpus_*.py`): the engine's own generation time (`gen_ms`), from the
  first to the last generated token, through the Python API without a server. See "Known limits".
* November 6 and 8 (`20261108/`): every run of every engine is a streamed server run; the time runs from the first to the last streamed
  chunk, and with thinking on the reasoning chunks count like the answer chunks.

## 20261108/: the speed tables of this release

The four speed tables of [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md) are made from the records of this folder by
[`20261108/scripts/tables.py`](20261108/scripts/tables.py). Its output is kept in [`20261108/tables/`](20261108/tables/)
(`SPEED_TABLES.md`, `speed_tables.json`).

### Engines, versions and model files

| Engine | Version, where | Model files and setting | Measured |
|---|---|---|---|
| YASAKANI | 0.1.0.dev0 with the `yskn.dll` built on November 7 (SHA-256 `0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb`, the DLL of this release), native on Windows | fastest model: `qwen38_27b_speed.yskn` (10,311,344,128 bytes, SHA-256 begins `3ba17a76871a2b5e`); quality model: `qwen38_27b_quality.yskn` (12,365,242,368 bytes, SHA-256 begins `6493983cf9539182`); each with `bdraft2_k8.yskn`, `bdraft2_k8.shead` and `kvq_u5.bin`. `yasakani serve` with its default settings | November 8 |
| MegaCapybara | 1.39, Windows x64 | Tiny model, the official one-conversation preset (KV 6/4), loop guard off, as in `20261103/` | thinking off: November 3; thinking on: November 6 |
| mkyt | `mkyt/qwen3-27b-5090` v0.3.2, the native Windows package | default settings: `juzbox/Qwen3.8-27B-NVFP4-GDN` (revision `3e68640`) with `incoai/Qwen3.8-27B-DFlash2` (revision `015e795`), NVFP4 KV, prefix cache on | November 6 |
| ignis | v0.9.6, Windows release | `qwen3_8_27b_nvfp4full-v2.ninfer` (19,406,942,468 bytes), 512K context, DFlash 2 with 7 draft tokens | November 6 |
| NInfer | the `master` branch as of November 6, commit `a754fcf3bf524494aa927c2f54e0cbfb58f569ab` (dated November 2), built in WSL2 (`logs/build/ninfer_master_build.log`) | the official NVFP4 artifact `qwen3_8_27b_nvfp4.ninfer` (23,719,715,844 bytes, SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`), DFlash 2 with 7 draft tokens and the optimized proposal head, FP8 KV, no prefix reuse (all flags: `launch/serve_ninfer_master.sh`) | November 6 |
| vLLM | 0.31.0, WSL2 | the servers, model files and launch scripts of `20261101/`: `vl_dflash_lm4`, `vl_mtp_lm4`, `vl_dspark_lm4` | November 6 |
| SGLang | 0.5.22, WSL2 | the same: `sg_dflash_nv`, `sg_dflash_lm4+norc`, `sg_mtp_nv`, `sg_dspark_lm4` | November 6 |
| llama.cpp | b11700, official Windows CUDA build | the same: `llama_dflq4`, `llama_mtp2` (context 16,384) | November 6 |
| TensorFold | 0.6.6, WSL2 | the MLX 4-bit model (`TensorFold/Qwen3.8-27B-MLX-4bit`) with the `z-lab/Qwen3.8-27B-DFlash2` drafter, CUDA backend, prompt cache off, context 69,632 | November 6 |

The commands are in [`20261108/launch/`](20261108/launch/): `yasakani_windows.cmd`, `megacapybara_windows.cmd`, `mkyt_windows.cmd`,
`ignis_windows.cmd` (the server and client command lines that ran, written out for publication), and `serve_ninfer_master.sh`,
`ninfer_master_build.sh`, `serve_ninfer.sh`, `serve_tensorfold.sh` (the scripts that ran in WSL2). SGLang, vLLM and llama.cpp were started
by `scripts/run_wf_n12.py` with the scripts of [`20261101/launch/`](20261101/launch/).

### Tests

The same request went to every engine (OpenAI-compatible chat API, streaming). One request at a time, one server at a time.

* NInfer's 12 public scenario prompts (code: CUDA, Python, TypeScript; story: English mystery, Chinese science fiction, Chinese dialogue;
  translation: Chinese to English, English to Chinese, Markdown; structured: CSV, JSONL, SQL), answers up to 4,096 tokens. By name and
  SHA-256 in [`20261103/prompts/ninfer_cases.json`](20261103/prompts/ninfer_cases.json). The files come from NInfer's repository
  (github.com/Neroued/ninfer) and are not copied here.
* Three tasks of our own, in [`20261103/prompts/cases.json`](20261103/prompts/cases.json), answers up to 8,192 tokens: `edit_code`
  ("Code edit": rename one variable inside `dedent` of CPython's `Lib/textwrap.py`, about 4,800 prompt tokens, and return the whole file),
  `edit_jsonl` ("JSONL edit": change one record of 60 JSONL lines and return all lines), `jsonl_orders_8k` ("Long JSONL": write 300 JSONL
  order records; cut at 8,192 tokens).
* Thinking off: greedy (temperature 0), `enable_thinking` false, 2 runs of every task.
* Thinking on: temperature 1.0, top-p 0.95, top-k 20, min-p 0, `enable_thinking` true, the run number as the seed, the same limits (the
  reasoning counts toward the limit), 2 runs of every task. YASAKANI's two models have 3 more runs (seeds 2, 3 and 4, in
  `results_seeds_2_4/`), 5 in all. No server refused one of these parameters (the clients would have logged "HTTP 400").
* Every client starts with two short warm-up requests. The servers started by `run_wf_n12.py` and `think_more.py` (all except mkyt, ignis
  and MegaCapybara) got two more from the driver before that.

### Result folders

Each folder under [`20261108/results/`](20261108/results/) is one start of one server.

| Folder | Engine and setting | Thinking |
|---|---|---|
| `n12_ysk_speed_v8`, `n12t_ysk_speed_v8` | YASAKANI, fastest model | off, on (runs 0 and 1) |
| `n12_ysk_quality_v8`, `n12t_ysk_quality_v8` | YASAKANI, quality model | off, on (runs 0 and 1) |
| `../results_seeds_2_4/n12t_ysk_speed_v8`, `n12t_ysk_quality_v8` | YASAKANI, both models | on (runs 2, 3 and 4) |
| `megacapybara_tiny_think` | MegaCapybara 1.39 Tiny | on (off: `20261103/results/megacapybara_kv64` and `own_tests`) |
| `mkyt_v032`, `mkyt_v032_think` | mkyt v0.3.2 | off, on |
| `ignis_v096`, `ignis_v096_think` | ignis v0.9.6 | off, on |
| `n12_ninfer_master`, `n12t_ninfer_master` | NInfer, the build of the tables | off, on |
| `n12_ninfer` | NInfer, the build of October (revision `81c8ce09`), not used in the tables | off |
| `n12_vl_dflash_lm4`, `n12_vl_mtp_lm4`, `n12_vl_dspark_lm4`; `n12t_vl_dflash_lm4`, `n12t_vl_mtp_lm4` | vLLM 0.31.0 with DFlash 2, MTP, DSpark | off; on (DFlash 2 and MTP) |
| `n12_sg_dflash_nv`, `n12_sg_dflash_lm4_norc`, `n12_sg_mtp_nv`, `n12_sg_dspark_lm4`; `n12t_sg_dflash_nv`, `n12t_sg_mtp_nv` | SGLang 0.5.22 with DFlash 2 (official 5090 recipe; radix cache off), MTP, DSpark | off; on (DFlash 2 and MTP) |
| `n12_llama_dflq4`, `n12_llama_mtp2`; `n12t_llama_dflq4`, `n12t_llama_mtp2` | llama.cpp b11700 with DFlash 2, MTP | off; on |
| `n12_tensorfold`, `n12t_tensorfold` | TensorFold 0.6.6 | off, on |

Inside a folder:

| Path | What |
|---|---|
| `n12/runs_<label>.jsonl` | one line per run of the 12 prompts: prompt and output tokens, time to the first chunk, generation time, speed (`decode_tok_s`), finish reason, the server's usage block, the time of day |
| `n12/texts/<prompt>_r<run>.txt` | the text of the run as it arrived (with thinking on: the reasoning, then the answer) |
| `n12/summary_<label>.txt` | the table the client printed |
| `own/runs_<label>.jsonl` | one line per run of the three additional tasks; for the two edit tasks also the comparison with the expected text (`similarity`, `exact`, `diff_lines`) |
| `own/<label>_<task>_r<run>.txt`, `.chunks.json` | the answer (without the reasoning), and every chunk with its arrival time in ms after sending (reasoning chunks are marked `"r"`) |
| `client_n12.log`, `client_own.log` | console output of the two clients |
| `server.log`, `warmup_*.json` | server output and the driver's warm-ups, for the servers not started by `run_wf_n12.py`; the other server logs are in `logs/server/` |

The other folders of `20261108/`:

| Path | What |
|---|---|
| `scripts/run_wf_n12.py` | the driver: starts one server, waits until it answers, sends two warm-ups, runs the two clients, stops the server (plans `n12`, `n12b`, `n12c`, `n12t`, `speedv8`, `speedv8t`, `qualv8`, `qualv8t`); it still contains the plans of October 18 and November 1 |
| `scripts/oai_bench_client.py`, `custom_bench.py` | the clients: the 12 prompts, and the three additional tasks. They are the clients of `20261103/scripts/` with options for thinking, extra parameters and seeds added on November 6; without those options they send the same requests |
| `scripts/think_more.py`, `oai_bench_client_seedoff.py`, `custom_bench_seedoff.py` | the three extra thinking-on runs of YASAKANI (the clients with a `--seed-offset` option) |
| `scripts/tables.py` | makes the tables from the records (below) |
| `tables/` | output of `tables.py` |
| `logs/driver/` | logs of `run_wf_n12.py` (`n12.log` = SGLang, vLLM, llama.cpp thinking off; `n12b.log` = NInfer October build and TensorFold; `n12c.log` = NInfer; `n12t.log` = thinking on; `n12*_speed_v8.log`, `n12*_quality_v8.log` = YASAKANI) and of `think_more.py` |
| `logs/server/` | server logs, `server_<n12 or n12t>_<setting>.log` |
| `logs/jobs/` | the steps of each job with their times and the nvidia-smi lines |
| `logs/build/` | the build log of NInfer |

### How the tables are made

`python 20261108/scripts/tables.py` reads the `runs_*.jsonl` files and the saved answers and prints the four tables; the rules are in its
docstring. In short: the speed of a run is the `decode_tok_s` of its record; runs that ended with `tool_calls` are never counted;
"thinking off" is the mean of the runs of a category, and the table of the README also leaves out the runs whose saved answer is a tool
call that nobody asked for or a repetition loop, by one rule applied to every engine; "thinking on" takes the fastest run of each task and
averages the tasks of a category; for an engine with several settings the largest value of each column is shown. The script lists the
runs it leaves out, the values of every setting, and the tables before the exclusion and with plain averages.

`tables.py` was written for this folder layout. The published tables were made by working scripts with the same rules, and the two were
compared: all 560 values (10 rows, 7 columns, 4 tallies, thinking off and on) are identical.

### Order and times (Japan time)

* November 3: MegaCapybara, thinking off, 14:23 to 14:32; its three additional tasks 16:02 to 16:05 (see `20261103/`).
* November 6, thinking off: SGLang, vLLM and llama.cpp 09:11 to 10:56 (in the order of `logs/driver/n12.log`), ignis 11:01 to 11:08,
  NInfer (October build) and TensorFold 11:09 to 11:27, mkyt 11:47 to 11:52, NInfer 12:17 to 12:25.
* November 6, thinking on: SGLang, vLLM, llama.cpp, NInfer and TensorFold 12:42 to 15:05 (`logs/driver/n12t.log`), ignis 15:06 to 15:17,
  mkyt 15:17 to 15:26, MegaCapybara 15:26 to 15:34.
* November 8: YASAKANI fastest model, thinking off 12:05 to 12:08 and on 12:08 to 12:14; quality model, off 12:14 to 12:18 and on 12:18 to
  12:25; the three extra thinking-on runs 13:49 to 13:57 (fastest) and 13:58 to 14:07 (quality).

### KV5 at long context (`20261108/long_context/`)

The records behind "KV5 (the compressed key-value cache) with the fastest model" in [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md),
measured on November 8 from 12:25 to 12:50 (after the speed runs of the same job). Four fixed sequences of 131,072 tokens were read
by the engine; at context lengths from 2,048 to 131,072 the last 1,023 positions of every sequence were compared with the BF16
reference, once with KV5 (the `yskn.dll` of this release, the fastest model) and once with an FP16 KV cache (a research build of the
engine with the same weights). The GPU memory was read from nvidia-smi after reserving contexts of 2,048 to 262,144 tokens.

| File | What |
|---|---|
| `kv5_summary.json`, `KV5_TABLES.md` | KL, 95% intervals, top-1 agreement and perplexity per context length for both KV formats; the difference KV5 minus FP16 at the same positions; the reserved memory per context length (FP16, INT8, KV5) and the bytes of KV cache per token |
| `kv5_score.log`, `fp16_score.log` | the log of the scoring run: one line per sequence and context length |
| `kv5_memory.log`, `int8_memory.log`, `fp16_memory.log` | the nvidia-smi readings before and after the engine was opened with each reserved context |

The summary and the tables are the numbers of the working summary with public labels; the logs are reduced to the lines of the
measuring script. The four sequences (our own thinking transcripts and answers, joined), the BF16 reference and the measuring scripts,
which drive the research build, are not included. The INT8 column is memory only; its quality was not measured.

### Quality on thinking text (`20261108/quality/`)

The records behind "Quality on thinking text" in [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md), measured on November 8 between 15:02
and 18:05. Twenty thinking transcripts of our own were read by every engine as windows of 2,048 tokens, and at the last 1,023 positions
of every window (20,460 positions) the engine's distribution of the next token was compared with that of the original BF16 model (the
reference). The text is fixed and nothing is generated freely: what is compared is the main model's own distribution at each position,
and the speed-up (the drafts) plays no part in it. The tools used for llama.cpp, NInfer, ignis and TensorFold run the main model only;
YASAKANI, SGLang, vLLM and mkyt were started without their helper models. MegaCapybara's server ran with the model and KV settings of
the speed tables and its helper model loaded; every request produced one token, the server's log shows no draft round for any of
the 20,460 requests, and the values read are those of the main model.

| File | What |
|---|---|
| `QUALITY_TABLE.md`, `quality_table.json` | all 15 measured configurations in the order of their KL, the ratios inside the two size bands, and two further records; made by `scripts/quality_table.py` from `scores/` |
| `scores/*.json` | one file per configuration: size, KV cache, KL, top-1 agreement, perplexity, how the distribution was obtained (`how`) and the numbers of the working record (`record`) |
| `logs/llamacpp_*.log` | the output of llama-perplexity for the six GGUF files |
| `logs/sglang_nvidia_nvfp4_*.log`, `logs/vllm_nvfp4_lmhead4_*.log` | the server log, and the log of the run that took the log-probabilities and scored them |
| `launch/*.sh` | how the SGLang and the vLLM server were started for this measurement |
| `scripts/kld_base.py` | the file format of the reference and the formula of one position, a numpy copy of the one in llama.cpp's perplexity tool; it has a self-test |
| `scripts/llama_kld.py`, `scripts/gguf_main_size.py` | the llama.cpp rows: the llama-perplexity call and the reading of its output; the size of a GGUF without its auxiliary layers |
| `scripts/api_topk_dump.py`, `scripts/topk_score.py` | the SGLang and vLLM rows: the 100 most likely tokens of every position through the API, and their score |
| `scripts/quality_table.py` | makes the table from the score files |

The table of the documents shows 13 of the 15 configurations. The two others, Unsloth UD-IQ3_S (12.04 GB, KL 0.0311) and Unsloth
UD-Q2_K_XL (9.83 GB, KL 0.0759), were measured in the same way and are only here.

How the distribution of each engine was obtained:

| Configuration | KL over | How |
|---|---|---|
| YASAKANI, both models | the whole vocabulary | A research build of the engine that can use an FP16 KV cache and return the logits of the whole vocabulary. It read every window 16 positions at a time on the verification path. The `yskn.dll` of this release was not used; the weights are the same data as the released model files. |
| llama.cpp, six GGUF files | the whole vocabulary | The stock `llama-perplexity` of b11700 with `--kl-divergence`, flash attention on, KV cache f16. |
| SGLang, vLLM | the 100 most likely tokens (a lower bound) | The server's own API returns the 100 most likely tokens and their log-probabilities for every position of a prompt. The KL is taken over these 100 tokens plus one bucket for all the others, which can only be smaller than the KL over the whole vocabulary. |
| mkyt | the 64 most likely tokens (a lower bound) | The server returns no log-probabilities; the engine's own Python library does (`forward_topk`); it refused 100 tokens and accepted 64. An earlier run with 20 tokens is kept as a further record. |
| NInfer | the whole vocabulary | A copy of the source (master `a754fcf3`) was built with an added dump of the logits in its perplexity tool. The installed build of the speed tables was not changed. |
| ignis | the whole vocabulary | The source of ignis has an example program, `span_logits`, that writes the logits of every position on the serving path. It is not in the release package, so it was built from the v0.9.6 tag. |
| TensorFold | the whole vocabulary | The server returns no probabilities for a prompt, so the CUDA engine of the installed package (weight loading, prompt processing, output layer) was driven directly from Python. |
| MegaCapybara | the whole vocabulary | The server returns no probabilities and the source is not published. A dynamic instrumentation tool (Frida) was attached to the running server process, and the logits that the engine had computed on the GPU for the last position were read from it, one request per position. No file of the program was changed. |

The programs written for the last five rows of this list reach into other people's software (they change a copy of it, build a part
that is not shipped, call its library or its internal modules directly, or attach to its process) and are not included; nor is the
scoring script of YASAKANI's own rows, which drives the research build. Not included either: the twenty transcripts, the
reference file (10.2 GB; it holds the token ids of the transcripts) and the per-position files of the runs (they hold the token ids
too). The table can therefore not be measured again from this folder alone.

Checks that were made, with their numbers in the score files:

* YASAKANI: the negative log-likelihood computed from the returned logits and the engine's own value differ by at most 1.8e-6, and the
  most likely token is the same at every position. The model files that the research build read have the same size in bytes as the
  released ones (12,365,242,368 and 10,311,344,128 bytes; compared on November 10).
* The lower bound: applied to YASAKANI's own logits, the KL over the 100 most likely tokens is 2.0% (quality model) and 1.5% (fastest
  model) smaller than the KL over the whole vocabulary; over 20 tokens it is 7.0% and 6.8% smaller. We do not use these gaps to correct
  the other engines.
* SGLang and vLLM: at all 20,460 positions the log-probability of the actual token was among the 100 and equal to the value the server
  reports for that token.
* NInfer: the dumped logits and the engine's own log-probabilities of the actual tokens differ by at most 7.5e-6. NInfer computes
  248,077 of the 248,320 output rows; the others were given a very low value, and the reference puts at most 0.003% of its probability
  on them at any position.
* MegaCapybara: the most likely token of the distribution that was read is the token the server actually returned at all 20,289
  positions where the two can be compared as text. An earlier run through the ordinary API alone gave a top-1 agreement of 92.69% on
  20,288 comparable positions, against 92.80% on all 20,460 here.

Times (Japan time, November 8): llama.cpp GSQ-RCO IQ3_S 15:02 to 15:03; vLLM 15:20 to 15:24; SGLang 15:24 to 15:28; mkyt (20 tokens)
15:28; YASAKANI fastest model 15:29 to 15:33, quality model 15:33 to 15:37; NInfer 15:37 to 15:40; TensorFold 15:40 to 15:43; ignis
15:43 to 15:47; MegaCapybara through the API 15:47 to 15:56; mkyt (64 tokens) 15:57; llama.cpp GSQ-RCO IQ3_XXS 16:06; the four Unsloth
files 16:48 to 17:09; MegaCapybara, whole vocabulary, 17:34 to 18:05. Power limit 600 W; other programs held 0.3 to 1.8 GB of GPU memory
when the jobs started. Starts that stopped before giving a result left no score: YASAKANI 15:03 to 15:19, mkyt at 15:19, MegaCapybara at
15:40, 17:27 and 17:30.

### The uncut screen recording of November 8 (`20261108/demo/`)

The evidence package of the code-edit demonstration described in [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md) ("The uncut screen
recording of November 8"), byte for byte as it was put together that evening. `manifest.json` lists the SHA-256 of every file, of the
model file and of `yskn.dll`. One request (the code-edit task of the speed tables) went to a new server process of the fastest model
after two unrelated short warm-ups, thinking off, temperature 0, output limit 8,192, and was recorded on screen together with a clock in a
separate command prompt: first output after 1.281498 s, last output after 5.555668 s, transfer complete after 5.556058 s, 4,712 tokens,
1,102.2 tok/s.

| File | What |
|---|---|
| `README_JA.md` | the description of the package (Japanese) |
| `manifest.json` | conditions, metrics, the check of the video time stamps, SHA-256 of every file |
| `request.json`, `original.py`, `output.py` | the API request as sent, the file to edit, the generated file |
| `arrival_events.jsonl`, `capture.json` | every received event with its arrival time; the same as a summary |
| `clock_console.py`, `clock-ticks.jsonl`, `clock-signal.json`, `clock-check.json` | the clock shown in the command prompt, its log, the request markers, the consistency check of the clock log |
| `verify_output.py`, `verify.json` | the checker of the generated code (syntax, and 17,701 behaviour comparisons with the original) and its result |
| `video-check.json`, `poster.png` | the check of the frame time stamps; one frame of the video |
| `reproduce.py` | a client that sends the same request |

The video itself, `fast_live_uncut.mp4` (11,391,224 bytes, SHA-256 `035b60e0c33f22f1f71312e1a1aab3ab383fc0a229dca4af7ac23a91eb4b6339`), is
not in this folder; it is published with the release. The values were measured while the screen was recorded and the CPU was encoding, so
they do not replace the speed tables. An earlier recording of the same evening (19:01) is kept and not included.

## 20261110/: checks on the files of this release

The records behind "Checks on this release (November 10, 2026)" in [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md): the `yskn.dll`
(SHA-256 `0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb`), the Python package 0.1.0.dev0 and the two model folders
that are distributed, checked on the measuring PC. In the file names "speed" is the fastest model and "quality" the quality model.

| Path | What |
|---|---|
| `checks/selfcheck/` | `yasakani info`; `yasakani verify` (every file matches `yasakani.json`); `yasakani selfcheck` (the reference written, then reproduced in a new process with the default context length and with 16,384; `*_sc_fresh.log`: six more times in new processes, after `selfcheck.json` had been put into the model folders); `yasakani bench` (`*_bench.json`) |
| `checks/on_off/` | `onoff_eq.py` (in [`20261101/scripts/`](20261101/scripts/)): 34 inputs, greedy and seed 1234, 62 cases; each case is generated with the default settings, the same again, and with the speed-up (the drafts) switched off in three ways, and the token ids are compared. Fastest model: 171 of 174 comparisons equal; quality model: 170 of 174 (`summary` in the JSON files, with the first different token of each case) |
| `checks/repeat/` | `repeat_probe.py`: one engine in one process, the same request generated 8 or 12 times, 256 tokens each. `speed_repeat_probe`, `quality_repeat_probe`: the full plan, which also repeats the three requests of the self-check 30 times. The other records are the short plan with the fastest model (four requests without drafts and one with the default settings, 12 times each): `env_A_base`, `env_F_base2` (nothing changed), `ctl_G_ctx262144`, `ctl_H_ctx16384` (context length), `ctl_I_warmup` (after the warm-up of the command-line tools), `ctl_J_nohash` (without reading the state hash), `clk_engine_off4` (GPU state sampled once a second, `.csv`), `quiet_1` to `quiet_4` (after the GPU had been unused for 30 s), `share_S1_bursts`, `share_S2_nopause` (while another process used the GPU; `bg_S*.log` is that process) |
| `checks/gpu/` | `gpu_consistency.py`: the same computations repeated with PyTorch 2.10.0+cu128 on the same GPU and compared bit for bit, with no YASAKANI code involved. `gpu_consistency` (matrix products, element-wise operations, 64-bit integers, 12 GiB of memory read back), `_bursty` (short computations, each followed by a transfer to the host), `_power_a` and `_power_b` (two processes at once at the 600 W limit); `clk_torch_*.csv` = the GPU state once a second. 542,278 repeats in all, no mismatch |
| `scripts/` | `repeat_probe.py`, `quiet_gate.py` (waits until the GPU is unused), `gpu_consistency.py`, `gpu_bg_load.py` (the other process of the sharing test), `gpu_sample.py`, and `repeat_summary.py`, which counts the repeat checks from the records |
| `launch/checks_windows.cmd` | the command lines that ran |
| `logs/jobs/` | the steps of each job with their times and the nvidia-smi lines |

`python 20261110/scripts/repeat_summary.py` prints the counts below. A run is counted as different when its token ids are not the most
common output of its group of repeats; a run that differs has the same tokens up to one position (listed by the script).

| Condition | Default settings | Drafts off (one token per pass) | Only the block drafter off | Self-check requests |
|---|---|---|---|---|
| nothing else used the GPU: median utilisation 0% before the start (`quiet_1` to `quiet_4`) | 0 of 48 | 0 of 192 | | |
| other programs were using the GPU: the desktop was drawing, median utilisation 5% before the start (9 records, both models) | 1 of 190 | 21 of 432 | 0 of 16 | 0 of 180 |
| another process used the GPU in short bursts (`share_S1_bursts`) | 0 of 12 | 2 of 48 | | |
| another process used the GPU without pause (`share_S2_nopause`) | 0 of 12 | 8 of 48 | | |

Times (Japan time): info to bench 07:42 to 07:44, on and off 07:55 to 08:11, repeat (full plan) 08:23 to 08:35, short plans 08:38 to 08:50
and 09:04 to 09:13, PyTorch 09:15 to 09:19, sharing 09:21 to 09:26, PyTorch (bursty) and the sampled repeat 09:27 to 09:33, PyTorch at the
power limit 09:35 to 09:39, quiet runs 09:42 to 09:50.

Limits of these checks: the cause of the differences has not been found. "Nothing else used the GPU" means that nvidia-smi showed 0 to 1%
utilisation for 30 s before the start; what the desktop did during the runs was not recorded. The short plan has five requests of 256
tokens with the fastest model; other requests, longer outputs and the server were not tried this way. Diagnostic runs of the same morning
with internal switches and with earlier builds are not included.

## 20261101/: YASAKANI, SGLang, vLLM, llama.cpp

### Engines, versions and model files

| Engine | Version, where | Model files |
|---|---|---|
| YASAKANI | public build 0.1.0.dev0, native on Windows. `yskn.dll` 26,184,192 bytes, SHA-256 `1fa3a25362f150c885ab7eb23fa32e672fbeebf7820c5b19a56ee452227a3718` | `qwen38_27b_e3x.yskn` (12,365,242,368 bytes) with `bdraft2_k8.yskn`, `bdraft2_k8.shead`, `kvq_u5.bin` |
| SGLang | 0.5.22, WSL2 | target: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (LMHead4) or `nvidia/Qwen3.8-27B-NVFP4` (SGLang's official RTX 5090 recipe) |
| vLLM | 0.31.0, WSL2 | target: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (LMHead4) |
| llama.cpp | b11700, official Windows CUDA build (`llama-server.exe`) | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, file `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |

Draft models of the other engines: DFlash 2 = `z-lab/Qwen3.8-27B-DFlash2` (SGLang, vLLM) or `ggml-org/Qwen3.8-27B-GGUF`, file
`dflash-Qwen3.8-27B-Q4_0.gguf` (llama.cpp); MTP = the multi-token prediction layers that come with the model; DSpark =
`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` (SGLang) or `RedHatAI/Qwen3.8-27B-speculator.dspark` (vLLM).
TensorRT-LLM 1.3.0rc29 was tried on October 18, 2026 only and could not run its fast setting on the RTX 5090; its files are kept in
`launch/tensorrt_llm_tried_2026-10-18/` with the notes of that day (`launch/notes_2026-10-18.txt`). It was not run on November 1.

### Settings of the other engines

The launch scripts are in [`20261101/launch/`](20261101/launch/) (`common.sh` sets the folders; the scripts are the ones that were run,
only the folder lines of `common.sh` differ). The setting names are the names used in the records.

| Setting (records) | Engine | Started with | Label in the tables |
|---|---|---|---|
| `sg_dflash_nv` | SGLang | `sg_dflash_nv.sh` | DFlash 2 (official 5090 recipe) |
| `sg_dflash_lm4+norc` | SGLang | `sg_dflash_lm4.sh` | DFlash 2 (radix cache off) |
| `sg_mtp_nv` | SGLang | `sg_mtp_nv.sh` | MTP |
| `sg_dspark_lm4` | SGLang | `sg_dspark_lm4.sh` | DSpark |
| `vl_dflash_lm4` | vLLM | `vl_dflash_lm4.sh` | DFlash 2 |
| `vl_mtp_lm4` | vLLM | `vl_mtp_lm4.sh` | MTP |
| `vl_dspark_lm4` | vLLM | `vl_dspark_lm4.sh` | DSpark |
| `llama_mtp2` | llama.cpp | `llama_cpp_windows.txt`, first command (`--spec-type draft-mtp --spec-draft-n-max 2`) | MTP |
| `llama_dflq4` | llama.cpp | `llama_cpp_windows.txt`, second command (`--spec-type draft-dflash --spec-draft-n-max 15`) | DFlash 2 |

For the long input (41,261-token prompt) the same settings ran with a larger context, by adding arguments to the same scripts:
`+48k` = SGLang `--context-length 49152` (DSpark also `--max-total-tokens 49152`), vLLM `--max-model-len 49152`, llama.cpp `-c 49152`
instead of `-c 16384`; `+nopc` = vLLM `--no-enable-prefix-caching`. The exact command of every setting is in `scripts/run_wf.py` (`CFGS`), and the
SGLang and vLLM servers print their arguments near the top of their logs in `logs/server/`.

### Tests

The same request body went to every engine (OpenAI-compatible chat API, streaming). Presence penalty 0 everywhere. The prompt files are in
[`20261101/prompts/`](20261101/prompts/); they are the files that were sent (same SHA-256 as `prompt_sha256` in `A_pub.json`).

| Test | Prompts | Settings | Answer length | Runs |
|---|---|---|---|---|
| chat, explain, code, math, json | one prompt each (`chat.txt` ...) | thinking on; temperature 1.0, top-p 0.95, top-k 20, min-p 0 | 2,048 tokens | 3 |
| long | `long.txt` (41,261 tokens) | same | 2,048 tokens | 3 |
| prose3 | `prose3_*.txt` (3 prompts) | same | 1,024 tokens | 3 each |
| japanese | `japanese_q1.txt` ... `q5` | thinking off; temperature 0.7, top-p 0.8, top-k 20 | 2,048 tokens | 1 each |
| ninfer_prose | NInfer's 6 story and translation prompts (`ninfer_cases.json`: names and SHA-256 only) | thinking off; temperature 0.6, top-p 0.95, top-k 20 | 4,096 tokens | 2 each |
| sglang_random_8k | `sglang_random_8k_1.txt` ... `_5` (random 8,192-token inputs) | thinking off; greedy; ignore EOS | 1,024 tokens | 1 each |

"Thinking tests" are the first six (chat to long). A run counts when it wrote at least 100 tokens.

### How the number of a test was taken

* Other engines: for each setting, the median of its counted runs; the setting with the highest median is the engine's number
  ("its fastest setting for each test").
* YASAKANI: two back-to-back rounds of all ten tests; for each round the median of the runs; the faster round is the number shown.
  Both rounds are in `results/yasakani/A_pub.json` and `A_pub_runs.csv`.
* `scripts/headline.py` recomputes the table below from the records. Its output:

| Test | YASAKANI round 1 / round 2 (shown) | SGLang | vLLM | llama.cpp | YASAKANI / fastest other |
|---|---|---|---|---|---|
| chat | 253.1 / 236.2 (253.1) | 123.8 | 106.6 | 119.0 | 2.05 |
| explain | 284.4 / 272.8 (284.4) | 156.4 | 150.2 | 127.3 | 1.82 |
| code | 381.4 / 408.7 (408.7) | 192.1 | 166.2 | 150.6 | 2.13 |
| math | 424.4 / 422.7 (424.4) | 240.7 | 180.9 | 183.6 | 1.76 |
| json | 346.9 / 330.2 (346.9) | 197.4 | 173.3 | 163.8 | 1.76 |
| long | 383.5 / 439.2 (439.2) | 221.9 | 139.7 | 183.9 | 1.98 |
| prose3 | 248.7 / 265.5 (265.5) | 132.0 | 115.3 | 120.0 | 2.01 |
| japanese | 209.3 / 199.5 (209.3) | 119.1 | 126.1 | 140.0 | 1.49 |
| ninfer_prose | 340.8 / 330.2 (340.8) | 164.1 | 147.0 | 163.1 | 2.08 |
| sglang_random_8k | 572.7 / 567.3 (572.7) | 162.4 | 163.8 | 192.6 | 2.97 |

Median of the six thinking tests: YASAKANI 377.8, SGLang 194.8, vLLM 158.2, llama.cpp 157.2 tokens/s. The median of the six ratios is 1.90.
The ratios here are computed from the unrounded medians; for "japanese" this gives 1.4945, while the one-decimal values in the table
(209.3 / 140.0) give 1.495, which is shown as 1.50 elsewhere.

### Order and times (Japan time, November 1, 2026)

YASAKANI speed 05:21 to 05:32 (`logs/driver/A_pub.log`), YASAKANI with the loop guard on 05:32 to 05:34, llama.cpp 07:42 to 08:05
(`X1_llama*.log`), SGLang and vLLM 08:05 to 10:19 (`X2_sg_vl*.log`). The self-check in `checks/selfcheck/` ran just before the speed
runs (05:20), the other checks in `checks/` between 05:34 and 07:36.

### Files

| Path | What |
|---|---|
| `launch/` | launch scripts of SGLang and vLLM (`common.sh`, `sg_*.sh`, `vl_*.sh`), llama.cpp commands, YASAKANI commands (`yasakani_windows.cmd`), notes of 2026-10-18, TensorRT-LLM files of the 2026-10-18 trial |
| `prompts/` | the prompt files; NInfer's prompts by name and SHA-256 only |
| `scripts/run_wf.py` | the driver used for the other engines: starts one server, sends two warm-ups, runs the tests through `capture_stream.py`, stops the server (plans `x1`, `x1long`, `x2`, `x2long`) |
| `scripts/capture_stream.py` | sends one streamed request and records the arrival time of every chunk; computes the speed |
| `scripts/wf_to_summary.py` | turns `runs_others.jsonl` into `results_summary_others.csv` and `results_runs_others.csv` |
| `scripts/a_speed.py` | the YASAKANI speed run (`A_pub.json`) |
| `scripts/onoff_eq.py` | the check that the output is the same with the speed-up on and off (`checks/on_off_equality/`) |
| `scripts/headline.py` | recomputes the table above from the records |
| `results/others/runs_others.jsonl` | every run of the other engines (441 runs; 438 counted) |
| `results/others/results_runs_others.csv`, `results_summary_others.csv` | the same as tables; the summary has the fastest setting per engine and test |
| `results/yasakani/A_pub.json`, `A_pub_runs.csv` | every run of YASAKANI, both rounds (98 runs) |
| `results/yasakani/A_pub_guard.json`, `A_pub_guard_runs.csv` | the six thinking tests with the loop guard on (18 runs) |
| `captures/x1`, `x1l`, `x2`, `x2l` | one file per run of the other engines: the request settings, every streamed chunk with its arrival time, the usage block and the speed (`x1` = llama.cpp, `x2` = SGLang and vLLM, `l` = long input). `warmup_*` are the warm-up requests. |
| `logs/driver/` | logs of `run_wf.py` (`X1_*`, `X2_*`), of the conversion (`X_convert.log`) and of `a_speed.py` (`A_pub*.log`) |
| `logs/server/` | server logs of every setting (`server_<plan>_<setting>.log`) and of YASAKANI (`A_pub*.serve.log`) |
| `checks/quality/` | top-1 agreement, perplexity and KL divergence against the original BF16 model on WikiText-2, C4 and FineWeb-Edu (61,380 positions); `kv5` = default settings, `kv8` = 8-bit KV control |
| `checks/aime2025/report.json` | AIME 2025, 30 problems, three runs: 30, 29 and 30 solved (summary only) |
| `checks/long_context/` | about 250,000 input tokens with the 262,144-token context: memory (nvidia-smi) and timings |
| `checks/on_off_equality/` | output with the speed-up on and off: 174 comparisons, all equal |
| `checks/selfcheck/` | `yasakani info`, `yasakani bench`, `yasakani selfcheck` and a server start of the public build |

## 20261103/: YASAKANI and MegaCapybara 1.39

### Engines and model files

| Engine | Version | Model files |
|---|---|---|
| YASAKANI | public candidate build of 2026-11-03, 0.1.0.dev0 (`yskn.dll` SHA-256 begins `d86fe2d0b2e0f27c`; it passed the self-check against the November 1 reference, `logs/client/yasakani_selfcheck_vs_nov1.log`) | "std": the standard model folder (12.4 GB, same file names as on November 1); "fast": pure 3-bit weights (11.7 GB) with the same helper files |
| MegaCapybara | 1.39, Windows x64 (`megacapybara.exe` SHA-256 `a36600f036e827f881b933a6260dd70977e36ea3f8bbfac463796aacc718d327`) | Tiny `qwen3.8-27b-tiny.mcapy` (16,594,763,776 bytes, SHA-256 `5bec1521fd7f1c99a5150ae37dca06122fbf5745f1b7baa167cf20cfd7c41118`) with its drafter `qwen3.8-27b-dflash2.mcapy` (1,293,942,784 bytes, SHA-256 `e6cb73f63d7c85d8a3f8e15ff222b3d99b375c3437c2d61f031115fb04013854`) |

MegaCapybara ran with the official one-conversation preset's defaults (KV 6/4) and its loop guard off; a second set used KV 8/8.
The command is in `launch/megacapybara_windows.cmd`. YASAKANI's in-process runs have no loop guard. Its server runs (`video_runs`,
`own_tests`) used the server's default, which has the loop guard on, because the clients did not send `loop_guard: false`; it did not act in
the video runs (their usage blocks have no `loop_guard` entry). The commands of every YASAKANI run are in `launch/yasakani_windows.cmd`.

### Tests

* NInfer's 12 public scenario prompts (code: CUDA, Python, TypeScript; story: English mystery, Chinese science fiction, Chinese dialogue;
  translation: Chinese to English, English to Chinese, Markdown; structured: CSV, JSONL, SQL), by name and SHA-256 in
  `prompts/ninfer_cases.json`. The files come from NInfer's repository (github.com/Neroued/ninfer) and are not copied here.
* Our own tests in `prompts/cases.json`: `edit_code` (rename one variable inside `dedent` of CPython's `Lib/textwrap.py`, about 4,700
  tokens, and return the whole file), `edit_jsonl` (change one record of 60 JSONL lines and return all lines), `jsonl_orders_8k` (write 300
  JSONL order records; cut at 8,192 tokens).
* Thinking off. Greedy (temperature 0) unless the folder says `t06` (temperature 0.6, top-p 0.95, top-k 20). Answers up to 4,096 tokens
  (own tests 8,192; `code2000` folders 2,000).

### Folders

| `results/` folder | Engine and condition | Runs | Script |
|---|---|---|---|
| `yasakani_std_greedy` | YASAKANI std, greedy | 12 prompts x 2 | `pub_corpus_greedy.py` (in-process) |
| `yasakani_fast_greedy` | YASAKANI fast, greedy | 12 x 2 | `pub_corpus_cond2.py` (in-process) |
| `yasakani_std_greedy_kv8` | YASAKANI std, greedy, plain 8-bit KV (`--env YSKN_TB_CAP=0`) | 12 x 2 | `pub_corpus_cond.py` (in-process) |
| `yasakani_std_t06` | YASAKANI std, temperature 0.6 | 12 x 5 | `pub_corpus_direct.py` (in-process) |
| `yasakani_fast_t06` | YASAKANI fast, temperature 0.6 | 12 x 5 | `pub_corpus_cond2.py` (in-process) |
| `yasakani_std_code2000`, `yasakani_std_code2000_kv8`, `yasakani_fast_code2000` | YASAKANI, the 3 code prompts, answers up to 2,000 tokens | 3 x 2 | `pub_corpus_cond2.py` (in-process) |
| `megacapybara_kv64`, `megacapybara_kv88` | MegaCapybara Tiny, KV 6/4 and 8/8, greedy | 12 x 2 | `oai_bench_client.py` (server, streaming) |
| `megacapybara_kv64_code2000`, `megacapybara_kv88_code2000` | MegaCapybara, the 3 code prompts, up to 2,000 tokens | 3 x 2 | `oai_bench_client.py` |
| `video_runs` | single recorded runs of 5 prompts (JSONL, English mystery, Python, SQL, Chinese to English) for YASAKANI fast, YASAKANI std and MegaCapybara (KV 6/4); every chunk with its arrival time in ms | 1 each | `stream_capture.py` (server, streaming) |
| `own_tests` | the own tests on YASAKANI fast, YASAKANI std and MegaCapybara (KV 6/4); `*.chunks.json` = every chunk with its arrival time | 3 tests x 2 | `custom_bench.py` (server, streaming) |

The other folders hold `runs_*.jsonl` (one line per run) and `texts/` (the answer of every run); the MegaCapybara folders also have the
`summary_*.txt` printed by the client.
`logs/client/` has the console output of every run, `logs/server/` the server logs (`mc_server_*` = MegaCapybara, `serve_*` = YASAKANI;
`serve_fast.log` and `serve_std.log` are a first attempt whose client started too early and got HTTP 503, see `capture_fast.log`),
`logs/client/chain.log`, `own_tests_chain.log` and `std_t06_chain.log` give the start and end time of each step (Japan time), and
`gpu/` the nvidia-smi state before and after (power limit, maximum clocks, memory, temperature).
`scripts/summarize.py` prints the means per prompt and category, the video runs and the own tests from these records;
`scripts/loop_scan.py` finds answers that end in an exact repetition, and `scripts/preloop_speed.py` measures the English mystery runs
before the repetition (run it from the `20261103` folder).

### Times (Japan time, November 3, 2026)

YASAKANI std temperature 0.6 09:22 to 09:30; YASAKANI std greedy 11:55 to 11:59; 8-bit KV 12:12 to 12:15; code 2,000 12:16 to 12:17;
YASAKANI fast 12:40 to 12:51; MegaCapybara 14:23 to 14:32; video runs 14:47 to 14:51; own tests 16:02 to 16:05.

## Re-measuring

### Other engines (November 1)

1. Install SGLang 0.5.22 and vLLM 0.31.0 in WSL2 (Ubuntu 24.04; CUDA 13.1 for FlashInfer's kernel build, see `common.sh`) and take
   llama.cpp b11700's official Windows CUDA build. Download the model files listed above.
2. Set `MODELS`, `SGLANG_VENV`, `VLLM_VENV` for `launch/common.sh` (the folder names it expects are listed at its top).
3. Start one server alone, for example `bash launch/sg_dflash_nv.sh` (port 30000) or a llama.cpp command from `launch/llama_cpp_windows.txt`
   (port 8091). Add the long-input arguments above for `long.txt`.
4. Send two warm-up requests, then each prompt with `scripts/capture_stream.py`, for example
   `python capture_stream.py --base http://127.0.0.1:30000 --prompt-file ../prompts/chat.txt --out chat_1.json --max_tokens 2048 --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0 --seed N --extra "{\"chat_template_kwargs\": {\"enable_thinking\": true}, \"presence_penalty\": 0.0}"`.
   The arguments of every test are in `run_wf.py` (`THINK`, `NINFER`, `JPNT`, `RAND`, `jobs()`); NInfer's prompts are sent as message
   lists (`--messages-file`).
5. Or run `scripts/run_wf.py` itself after editing its folder lines (`<...>` placeholders); it also needs a `gpu_lock` helper that is not
   included (any context manager that does nothing will do).
6. `wf_to_summary.py` and `headline.py` turn the records into the tables.

### The tables of this release (November 6 and 8)

1. YASAKANI: this release, started as in `20261108/launch/yasakani_windows.cmd`. MegaCapybara, mkyt and ignis: their Windows packages,
   `megacapybara_windows.cmd`, `mkyt_windows.cmd`, `ignis_windows.cmd`. NInfer and TensorFold: in WSL2, `ninfer_master_build.sh`,
   `serve_ninfer_master.sh`, `serve_tensorfold.sh`. SGLang, vLLM and llama.cpp: as for November 1, above.
2. Put NInfer's fixture files where `FIX` at the top of `20261108/scripts/oai_bench_client.py` points.
3. Start one server alone and run the two clients with the arguments listed in the launch files: `oai_bench_client.py` for the 12 prompts,
   `custom_bench.py --cases 20261103/prompts/cases.json` for the three additional tasks.
4. Or run `scripts/run_wf_n12.py` itself after editing its folder lines (`<...>` placeholders); like `run_wf.py` it needs a `gpu_lock`
   helper that is not included.
5. `python 20261108/scripts/tables.py` makes the tables from the records of this folder.

### The quality table (November 8)

The table cannot be measured again from this folder alone, because the text and the reference file are not published. The scripts run as
they are with a reference file of your own in the format of llama.cpp's perplexity tool (`llama-perplexity -m <model> -f <text>
--kl-divergence-base <file>` writes one; `kld_base.py` describes the format and can write one from logits):

1. llama.cpp: `python 20261108/quality/scripts/llama_kld.py --llama-perplexity <exe> --base-file <file> --gguf <name>=<gguf>`.
2. SGLang or vLLM: start the server as in `20261108/quality/launch/`, then `api_topk_dump.py --api sglang|vllm --base http://127.0.0.1:30000
   --base-file <file> --label <name>` and `topk_score.py --base-file <file> --label <name>`.
3. `python 20261108/quality/scripts/quality_table.py` makes the table from the score files of this folder.

### The checks of November 10

`20261110/launch/checks_windows.cmd` lists the commands. `yasakani verify`, `selfcheck` and `bench` need only the release.
`onoff_eq.py` and `repeat_probe.py` read the prompt files of `20261101/prompts/` and NInfer's fixture files (the two folders are set near
the top of `main()`); `repeat_probe.py` also needs `onoff_eq.py` and `a_speed.py` of `20261101/scripts/` (`--tools`). `gpu_consistency.py`
needs PyTorch with CUDA.

### YASAKANI (November 1 and 3)

From the public release: `20261101/launch/yasakani_windows.cmd` (November 1) and `20261103/launch/yasakani_windows.cmd` (November 3). `yasakani bench`
measures your own PC with shorter tests.

### MegaCapybara (November 3)

`20261103/launch/megacapybara_windows.cmd` starts the server; the client commands are listed in the same file.

## Known limits

### The tables of this release (20261108/)

* Not the same day: the other engines were measured on November 6, MegaCapybara with thinking off on November 3, YASAKANI's two models on
  November 8. Each server ran alone, one after another, not interleaved. The GPU was at 34 to 51 degrees C when the jobs of November 6
  started and at 30 degrees C when the YASAKANI job of November 8 started. When the jobs started, other programs (the desktop) held
  between 0.4 and 2.1 GB of GPU memory (`gpu memory used` in `logs/jobs/`; 1.7 GB before the YASAKANI job).
* Few runs: 2 per task with thinking off, 2 per task with thinking on for the other engines and 5 for YASAKANI. "Fastest run of each task"
  therefore picks from 5 tries for YASAKANI and from 2 for the others; the table of plain averages is given next to it.
* Thinking on: all 120 runs of the 12 prompts of YASAKANI ended at the output limit, 114 of them while still reasoning, and most runs of
  every engine ended at the limit. The numbers are generation speeds, not the time to an answer and not accuracy.
* Thinking off: some greedy answers are repetition loops or tool calls that nobody asked for, and such text is written faster than normal
  text. The table of the README leaves those runs out for every engine by one rule (`tables.py` lists them: YASAKANI's fastest model
  loses both CUDA runs and both TypeScript runs, which takes its "Code" from 900 to 529; the quality model loses both English mystery runs;
  mkyt, NInfer, vLLM and SGLang lose TypeScript runs, vLLM's DFlash 2 setting also its CUDA runs, llama.cpp's MTP setting one English
  mystery run). The rule looks at the end of the saved text only; it is not a judgment of the quality of an answer.
* NInfer: the tables show the `master` build of November 6 (commit of November 2). The build of October (revision `81c8ce09`), measured
  the same morning with the same settings, is in `results/n12_ninfer` and is not used in the tables. Its "Code" was 290 against 282
  before the exclusion and 277 against 264 after it; in the other six columns the October build is between 3.1% slower and 0.1% faster.
* TensorFold: 0.6.6. The upgrade to v1.2.2 failed to build on November 6 (`logs/jobs/job_1003_ninfer_october_tensorfold.log`), so the
  newer version was not measured. In general these are the versions we could run on those days, not the latest of every program.
* mkyt ran at the 600 W power limit of the PC; mkyt's own published numbers are at 400 W.
* SGLang, vLLM, NInfer and TensorFold ran in WSL2, not on native Linux.
* Each engine used its own model files, number formats, KV cache and speed-up settings (`20261108/launch/`); sizes differ. This is not a
  comparison with every setting made equal.
* Loop guard: the clients sent no loop-guard setting, so YASAKANI's server ran with its default; no usage block of these runs has a
  `loop_guard` entry. MegaCapybara ran with its loop guard off.
* Only the generation speed is compared. The time to read the prompt is not part of the numbers; the longest prompt has about 4,800
  tokens.
* The driver logs `n12.log` and `n12t.log` end with a step `ysk_std_1103`: the YASAKANI build and standard model of November 3, run as a
  check on the day (in `n12t.log` it did not start and was run again later). Its records, and those of other earlier YASAKANI model files
  measured on November 6 and 7, are not in this folder, because those files are not distributed. `run_wf_n12.py` still lists them.
* Three more thinking-on runs of the code-edit task with the fastest model (seeds 5 to 7) were made on November 8 and are not used in the
  tables and not included here.

### The quality table (20261108/quality/)

* Our own text: twenty thinking transcripts of our own, not used for the statistics with which the models were made. The table says how
  close each configuration stays to the original model on this text. It is not a general quality score and not the rate of correct
  answers on tasks.
* YASAKANI's two rows were measured with a research build and an FP16 KV cache, not with the `yskn.dll` of this release and not with
  KV5, the KV format of the speed tables. What KV5 changes is measured separately (`long_context/`), for the fastest model.
* Three rows are lower bounds (SGLang, vLLM, mkyt): their true KL is at least the value shown. SGLang's 0.0283 and the fastest model's
  0.0304 can therefore not be put in order.
* The engines were not read in the same way: YASAKANI 16 positions at a time on its verification path, most others on their
  prompt-processing path, MegaCapybara one position per request.
* NInfer's numbers come from a copy of its source with a dump added, MegaCapybara's from the memory of its running process. The checks
  listed above are what we have to show that these two ways did not change the numbers.
* One measurement per configuration, at contexts of up to 2,048 tokens.
* Sizes are file sizes and hold different things (a vision part, a helper model, auxiliary layers), so this is not a comparison of
  compression at equal size. The two size bands compare files that are near each other in size, not equal.
* Not measurable again from this folder alone: the text and the reference file are not published.

### The earlier comparisons (20261101/, 20261103/)

* The engines ran one after another on the same day (times above), not interleaved. Each server ran alone on the GPU.
* YASAKANI's speed runs were made with its loop guard off, because the other engines have nothing like it. With the guard on
  (`A_pub_guard.json`), the six thinking tests differed from round 1 by at most 0.32%.
* Both numbers are "best of" selections of different kinds: the faster of two rounds for YASAKANI, the fastest setting for each other engine.
* SGLang and vLLM ran in WSL2, not on native Linux.
* Each engine used model files that run fast on it, so the number formats and file sizes differ (YASAKANI 12.4 GB, llama.cpp GGUF 12.1 GB,
  SGLang and vLLM NVFP4 about 18 to 22 GB as loaded). The comparison is not matched by size.
* Long input: SGLang's official RTX 5090 recipe (`sg_dflash_nv+48k`) holds 39,571 tokens of KV and accepts inputs up to 39,565 tokens; it
  refused the 41,261-token prompt three times (`logs/server/server_x2l_sg_dflash_nv_48k.log`). Those three runs are in the records with one
  token and are not counted; SGLang's long-input number comes from its other settings. (`launch/notes_2026-10-18.txt` gives 35,067 tokens for
  the trial of October 18; the November 1 log shows 39,571.) vLLM's DFlash 2 and DSpark settings ran the long input with prefix caching off,
  because on October 18 they stopped when the same long prompt was sent twice.
* `results_summary_others.csv` has a `yasakani` column that `wf_to_summary.py` copies from the October 18 table; it is not the
  November 1 measurement, and the ratio column is empty on purpose. The November 1 YASAKANI numbers are in `results/yasakani/`.
* `a_speed.py summary` reports the slower of the two rounds and compares with the October 18 table. The tables it printed on November 1
  are not included; `headline.py` gives the numbers used publicly (faster round, November 1 runs of the other engines).
* November 3: YASAKANI's numbers on the 12 prompts were taken in-process (engine generation time, no server), MegaCapybara's through its
  server with streaming. The video runs and the own tests used servers for both. On the same prompts, YASAKANI's server runs are within
  2.5% of its in-process means, except the English mystery (repetition, below).
* November 3: some of YASAKANI's greedy answers to the English mystery prompt end in an exact repetition, which is written faster than
  the rest (`loop_scan.py`: std 35.0% of the text in both greedy runs and in the video run; fast none and 10.9% in the two greedy runs,
  32.4% in the video run; `preloop_speed.py`: in the video runs the repetition starts at about token 2,663 for std and 2,770 for fast).
  Over the first 2,000 tokens of the video runs the speed is 302.7 (fast, estimated from the chunks), 271.9 (std, estimated) and 237.7
  (MegaCapybara, exact; it did not repeat). No other answer in these folders ends in a repetition, including all temperature 0.6 runs.
* November 3: MegaCapybara stopped the TypeScript prompt with `finish_reason: tool_calls` after 147 to 201 tokens in 5 of its 8 runs of that
  prompt; those runs are left out of the means (`summarize.py` lists them). Its greedy speed varies by about 10% from run to run.
* November 3: YASAKANI's server runs (`video_runs`, `own_tests`) had the server's default loop guard on (see above); its in-process runs
  and all MegaCapybara runs had none.
* The November 1 and November 3 YASAKANI builds are different builds of the same version (DLL SHA-256 prefixes `1fa3a253` and `d86fe2d0`).
* November 1: `checks/on_off_equality/` records 174 comparisons, all equal, for the build of that day. On November 10 the same script,
  run on the files of this release while other programs were using the GPU, gave 171 of 174 (fastest model) and 170 of 174 (quality
  model); see "Checks on this release" in [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md). The record of November 1 is kept as measured.

## Changes made for publication

* Paths on the measuring PC were replaced by placeholders: `<model_dir>`, `<model_dir_fast>` (model folders), `<yasakani_dir>` (the folder
  with `bin\yskn.dll`), `<yasakani_py>` (the Python package), `<llama_cpp_dir>`, `<gguf_dir>`, `<megacapybara_dir>`, `<python_dir>`,
  `<work_dir>`, `<out_dir>`, `<tmp_dir>`, `<helpers_dir>`, `<reference_dir>`, `<oct18_run_dir>`, `<ninfer_corpus_dir>`, `<sglang_venv>`,
  `<vllm_venv>`, `<trtllm_venv>`, `<log_dir>`; the user name by `<user>`.
* Third-party prompt text: in the capture files of NInfer's prompts the `prompt` field was replaced by a reference to
  `prompts/ninfer_cases.json`. The answers (including translations of NInfer's texts) are kept, as they are the generation records.
  AIME 2025 problems and answers are not included; only the summary report.
* The e-mail address in the header of CPython's `textwrap.py` (part of the `edit_code` prompt and of its answers) was replaced by
  `<email>` in `cases.json`, the answers and the chunk records. The three engines' answers are still identical to each other.
  `textwrap.py` is Copyright Python Software Foundation and others, under the PSF License.
* Two quality summaries (`checks/quality/summary_kv*.json`): the list of engine environment settings was replaced by a note.
* `20261103/logs/megacapybara_exe_sha256.txt` (output of `certutil`) was converted from the Japanese Windows code page to UTF-8.
* Scripts: internal notes in comments and docstrings were removed and folder lines were replaced by placeholders; `loop_scan.py` and
  `preloop_speed.py` use the folder names of this layout. The `script_sha256` values inside the records refer to the scripts as run.
* Some answers contain words or e-mail addresses at `example.com` that the models wrote; they are left as written. The random
  8,192-token prompts are random vocabulary tokens in many languages and are left unchanged, as they are the measured inputs.
* Files renamed: the two CSV files of the other engines lost the date at the end of their names, `runs.jsonl` became
  `runs_others.jsonl`, and the November 3 result folders were given descriptive names (table above).
* `20261108/`: the same replacements, with these further placeholders: `<model_dir_fastest>`, `<model_dir_quality>`, `<ignis_dir>`,
  `<mkyt_dir>`, `<mkyt_hf_dir>`, `<ninfer_dir>`, `<ninfer_src_dir>`, `<tensorfold_dir>`, `<tensorfold_venv>`, `<launch_dir>`, and in
  `run_wf_n12.py` `<yasakani_dir_nov3>`, `<model_dir_nov3>`, `<model_dir_fast_nov3>`, `<model_dir_other_helper>`, `<yasakani_dir_oct31>`,
  `<install_job>`, `<download_job>`. The e-mail address of `textwrap.py` was replaced by `<email>` in the answers and chunk records of
  `edit_code`, also where it was split over two chunks.
* `20261108/logs/jobs/`: the job logs were reduced to the lines that give the time of each step and the nvidia-smi output. The other
  lines (stopping and restarting our own always-on server, console noise) were dropped.
* `20261108/scripts/`: internal notes in comments and docstrings were removed, and folder lines were replaced by placeholders or by the
  folder names of this layout. `tables.py` was written for this layout (see "How the tables are made").
* `20261108/launch/*.cmd` were written for publication from the job files that ran: the server and client command lines are the ones
  that ran, with the folders replaced. The `.sh` files are the scripts that ran, with the folders replaced.
* `20261108/long_context/`: the logs were reduced to the lines of the measuring script, the internal label of the KV format was replaced
  by `kv5` and the name of the research build was shortened; `kv5_summary.json` and `KV5_TABLES.md` were made from the working summary.
* `20261108/demo/` is the evidence package without any change, so that the hashes in its `manifest.json` hold; the header of
  `textwrap.py` (with its e-mail address) is as in CPython there.
* `20261108/quality/`: the score files were written from the working result files. The numbers are copied as they are; the labels, the
  key names and the description of each method are the public ones; folders, the names of our research libraries and the list of other
  programs that held GPU memory were left out. The scripts are the working scripts with the folder lines turned into arguments, the
  helper module renamed to `kld_base.py` and internal notes removed from the docstrings; `gguf_main_size.py` is the header reader of
  the working script with a small command line of its own; `quality_table.py` was written for this layout. The two launch scripts are
  the ones that ran, pointing at `20261101/launch/common.sh`. The logs are as written, with the folders replaced. On November 10 the
  published `topk_score.py` was run on the PC on the four saved sets of log-probabilities with the reference file and gave the saved
  scores digit for digit; the reading of llama-perplexity's output in `llama_kld.py` gave the saved values for the six logs,
  `gguf_main_size.py` the saved sizes for the six files, and `quality_table.py` the same table. `api_topk_dump.py` was run only
  against a stand-in server with made-up numbers, because the servers of the engines were not started again.
* `20261110/`: the same replacements (`<model_dir_fastest>`, `<model_dir_quality>`, `<yasakani_dir>`, `<out_dir>`); the job logs were
  reduced in the same way; in `repeat_probe.py` the two prompt folders were replaced by `<work_dir>` as in `onoff_eq.py`;
  `repeat_summary.py` and `launch/checks_windows.cmd` were written for publication.
