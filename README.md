# YASAKANI

A local LLM engine for **Windows** and **one NVIDIA RTX 5090**, built for one model: **Qwen3.8-27B**. Open source, Apache License 2.0.
日本語は [README.ja.md](README.ja.md)。

> **Version 0.1.0 - the first public release (November 13, 2026).** Engine: [release page](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.0). Models on Hugging Face: [quality model](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) and [fastest model](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed).
>
> **Support the development:** crowdfunding on CAMPFIRE for backers living in Japan, until December 16, 2026 - https://camp-fire.jp/projects/992233/view (outside Japan: see [Support](#support))

## What it is

YASAKANI runs Qwen3.8-27B on your own PC. In a comparison with eight other engines on the same PC (one RTX 5090, Qwen3.8-27B, one request at a
time), its fastest model wrote answers faster than every one of them in all seven task categories, with thinking off and with thinking on. The
tables and what they do and do not show are under [Speed](#speed). It starts an **OpenAI-compatible server** (so chat front-ends and developer
tools can use it), a **terminal chat**, and a **benchmark**. After the download it needs no internet connection.

It was built by one person in Japan. The engine is a purpose-built set of GPU kernels, a file format and a runtime for this one card and
this one model; it is not a general-purpose engine.

## Two models

Both are Qwen3.8-27B, converted to the YASAKANI file format and compressed. You need only one of them.

| | Quality model (recommended) | Fastest model |
|---|---|---|
| Made for | staying close to the original model, at high speed | the highest generation speed and the smallest size |
| Model file | 12.37 GB | 10.31 GB |
| Same most likely next token as the original model | 95.24% | 93.01% |
| KL divergence from the original model (smaller is closer) | 0.0122 | 0.0304 |
| Download | [Hugging Face](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) | [Hugging Face](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed) |

The two quality rows are agreement with the original model's next-token prediction on 20 thinking transcripts, not task accuracy; see
[Quality](#quality). The file size is not the GPU memory needed to run the model.

## Speed

Tokens written per second, on one RTX 5090 (stock 600 W, no overclocking), the same PC for every engine, one request at a time. Bigger is faster.

**Thinking off** (greedy; average of 2 runs per task, after leaving out runs that fell into a repetition loop or made a tool call nobody asked for):

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI fastest model** | **529** | **285** | **494** | 909 | **1,222** | **1,156** | **1,260** |
| **YASAKANI quality model** | 486 | 179 | 483 | **930** | 1,136 | 1,151 | 1,160 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 204 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 150 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 139 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

**Thinking on** (temperature 1.0, top-p 0.95, top-k 20, min-p 0; the fastest run of each task):

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI fastest model** | **337** | **287** | **497** | **536** | 667 | **1,108** | **704** |
| **YASAKANI quality model** | 299 | 242 | 428 | 480 | **743** | 1,053 | 634 |
| MegaCapybara 1.39 Tiny | 290 | 206 | 308 | 429 | 550 | 663 | 516 |
| mkyt v0.3.2 | 244 | 196 | 348 | 384 | 378 | 603 | 454 |
| ignis v0.9.6 | 211 | 130 | 276 | 277 | 278 | 485 | 327 |
| NInfer (November 2 build) | 172 | 140 | 258 | 263 | 263 | 404 | 322 |
| vLLM 0.31.0 | 131 | 112 | 197 | 195 | 187 | 369 | 253 |
| SGLang 0.5.22 | 157 | 124 | 206 | 230 | 257 | 402 | 282 |
| llama.cpp b11700 | 140 | 122 | 184 | 197 | 206 | 349 | 221 |
| TensorFold 0.6.6 | 134 | 79 | 167 | 183 | 439 | 360 | 268 |

What these tables are, and are not:

* "Speed" is the speed of **writing the answer**: (output tokens - 1) / (time from the first to the last content chunk). The time to read the
  prompt is not included.
* The tasks are 12 published prompts (code, story, translation and structured output, 3 each) and 3 tasks of our own (code edit, JSONL edit,
  long JSONL). The output limit is 4,096 or 8,192 tokens.
* Thinking off: the average leaves out runs with a repetition loop or an unrequested tool call, by one rule applied to every engine. This can
  change how many tasks a category holds. For the fastest model's "Code" it matters: 900 tok/s before the exclusion, 529 after; we do not use
  900 as a figure for practical code generation. The table before the exclusion is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).
* Thinking on: for each task the fastest run is taken, then the category is averaged. YASAKANI chose from 5 runs, the other engines from 2,
  so the number of tries is not equal; the table of plain averages is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md). Many runs ended at the
  output limit, so this is not a comparison of time-to-answer or of accuracy.
* Every engine ran its own weights, quantization, key-value cache format and speed-up settings; this is not a comparison with all settings
  made equal. For an engine measured with several settings, the fastest setting of each category is used.
* YASAKANI's two models were measured on November 8, 2026 with the `yskn.dll` of this release. MegaCapybara with thinking off was measured on
  November 3; its thinking-on runs and the other engines on November 6. This is not a ranking of the latest version of every program, or of
  every use.

Everything needed to check or repeat the comparison (launch settings, prompts, the measurement scripts and the raw records) is in
[`bench/`](bench/). `yasakani bench` measures your own PC.
If you measure with a script of your own, take the token count from the server's `usage.completion_tokens` (in a stream, send `stream_options`
with `{"include_usage": true}`). A stream carries one chunk per token, but where one character is split over several tokens its text arrives
with the last of them, so the number of chunks is slightly smaller than the number of tokens.

### One edit, start to finish

A screen recording without cuts shows the fastest model rewriting a Python file of about 20,000 characters: 5.56 s from sending the request to
the last output (4,712 tokens, 1,102 tok/s while writing, measured while the screen was being recorded). The generated file gave the same
results as the original on 17,701 inputs. The recording, logs, the generated code and the checks are in the release's evidence archive;
see [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

## Quality

The models are compressed (the original BF16 weights are 53.8 GB), so they are **not identical** to the original Qwen3.8-27B.

How close the next-token predictions stay to the original BF16 model was measured on 20 thinking transcripts from our own use of the model,
none of them used when the files were made (20,460 positions, FP16 key-value cache, November 8, 2026). Thirteen configurations of different
engines and compressed files were measured on the same texts:

* The **quality model** has the smallest KL divergence (0.0122) and the highest top-1 agreement (95.24%) of the thirteen.
* Against the GSQ-RCO files of similar size run by llama.cpp: 12 GB class, 95.24% against 93.64% (IQ3_S); 10 GB class, the fastest model's
  93.01% against 91.22% (IQ3_XXS). The KL divergence is about 47% and 41% smaller.

Agreement is "chose the same most likely next token as the original model". It is **not** the rate of correct answers on tasks, and the
result covers thinking text only: on general web text an additional evaluation showed a KL divergence 3-10% larger for the quality model
than for our earlier conversion of the same size. The full table, including the configurations for which only a lower bound of the KL
divergence could be measured, is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

Speculative decoding only keeps tokens that the main model agrees with, so it is built not to change what is written. Whether a repeated
request gives exactly the same tokens depended, in our checks of November 10, 2026, on whether other programs were using the GPU at the same
time. With the GPU otherwise idle, all 240 repeated runs were identical. While other programs were using the GPU (the desktop was drawing),
a run now and then had the same tokens as the others up to one position and continued differently from there:
1 of 190 runs with the default settings and 21 of 432 with speculative decoding switched off (256 tokens each). The cause has not been found
yet, so this release does not promise bit-identical text from run to run. The details are in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

## Long contexts

The context length goes up to 262,144 tokens. The key-value cache is stored in YASAKANI's own compressed format, **KV5**. With the fastest
model and the full 262,144-token context reserved, the engine takes 16.78 GiB with KV5 against 27.13 GiB with an FP16 cache (about 38% less
for the whole engine, about 66% less per cached token). On fixed texts of 2K to 128K tokens, the KL divergence with KV5 was at most 0.00052
higher on average than with FP16. The 262,144 figure is what the engine reserves at start; quality was checked up to 128K, and this is not a
statement that the model runs on a 16 GB GPU.

Together with the helper model that speeds up generation, the GPU memory in use at the full 262,144-token context was about 19.1 GiB for the
fastest model and about 21.2 GiB for the quality model (about 14.3 GiB and 16.4 GiB with `--max-ctx 32768`; November 11, 2026, the difference
of the nvidia-smi values before and after loading).

## What you need

| | |
|---|---|
| GPU | NVIDIA **GeForce RTX 5090** (32 GB). The kernels are built for this card (compute capability 12.0) only. |
| OS | **Windows 11**, 64-bit (the only version tested) |
| Driver | a current NVIDIA driver (tested with 616.92) |
| Software | Microsoft Visual C++ 2015-2022 x64 runtime, Python 3.10 or newer |
| Disk / RAM | 20 GB free (one model is a download of about 13 to 15 GB); 32 GB of RAM recommended (tested on a PC with 96 GB) |

Not supported in this release: Linux, macOS, AMD GPUs, other NVIDIA GPUs (the RTX 4090, 3090 and the 16 GB cards have not been tested with
this release), other models. Support for more GPUs is the purpose of the crowdfunding campaign; see [Support](#support).

Close other programs that use a lot of GPU memory (games, other LLM servers) before you start. When the memory runs out there is no error:
generation just becomes several times slower (a warning is printed at start in that case).

## Quick start

**The quickest way:** unpack `yasakani-quickstart-0.1.0.zip` from the [release page](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.0) to a plain path such as
`C:\yasakani` and double-click `START.cmd`. It downloads the engine and one model and runs the checks; then you can measure the speed,
compare look-ahead on and off (the numbers of the technical notes, on your own PC), chat in your browser, or start the server. You only answer its questions with a number (Python 3.10 or newer is needed). What it does is in
[quickstart/](quickstart/). To install by hand:

1. Download the engine zip (`yskn.dll` and the Python package) from the [release page](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.0) and one model folder from
   Hugging Face ([quality](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) or [fastest](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed)). Compare the SHA-256 checksums with the ones on the release page.

```bat
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-quality --local-dir D:\models\qwen38-quality
```

2. Install the Python package and check the computer, the files and the installation:

```bat
pip install .
yasakani info      --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
yasakani verify    --model-dir D:\models\qwen38-quality
yasakani selfcheck --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

3. Chat in the terminal, or start the server and point any OpenAI-compatible client at `http://127.0.0.1:8765/v1`:

```bat
yasakani chat  --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

**Good to know before the first request**

* By default the model thinks before it answers, at its longest thinking setting. On requests that ask for a certain number of characters, or
  for a whole program in one go, the thinking alone can take thousands to tens of thousands of tokens, and no answer appears during that time.
  To shorten it: `/effort medium` or `/effort low` in `yasakani chat`, `/think off` to answer without thinking; through the API,
  `reasoning_effort` (`medium`, `low`) or `{"enable_thinking": false}` in `chat_template_kwargs`.
* The output limit (max tokens) includes the thinking. If your client's limit is small, the request ends in the middle of the thinking and the
  answer is empty. Raise the limit when the model is to think (the server's default is 32,768); for requests where even that is not enough, use
  a shorter thinking setting.
* For a long program written in one go, we suggest `medium`. In a small number of trials, `medium` most often gave code that runs as it is;
  the default (longest thinking) can think for minutes and then reach the limit, and thinking off or `low` gave more code that does not run.
* With thinking off, we use temperature 0.7, top-p 0.8 and top-k 20 in development. Temperature 0 falls into repetition more easily on long prose.

The details (the model folder, supported request fields, the loop guard, the self-check) are in [python/README.md](python/README.md).
To build `yskn.dll` yourself, see [BUILDING.md](BUILDING.md); common questions are in [docs/FAQ.md](docs/FAQ.md).

## Use it as an agent

`yasakani-harness-0.1.0.zip` on the [release page](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.0) opens an agent screen in your browser, run by the model
on your PC: the model reads and writes files in a work folder and runs commands (you choose on the screen whether it may change files).
Run the quickstart above first; then unpack the harness zip into the quickstart folder or next to it and double-click `START_HARNESS.cmd`.

* The screen is [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (open-source software by DeepSeek, MIT license), used
  unchanged. It is installed from npm on first use (about 730 MB; Node.js is fetched too if the PC has none). YASAKANI adds a settings file
  and a Japanese language pack. YASAKANI is not affiliated with DeepSeek, and this combination is not an official product.
* Tool calling is a preview. The harness brings the next version of the Python package (0.2.0.dev0), in which the server answers requests
  with `tools`, and starts the server with it. The 0.1.0 package of this release has no tool calling.
* A new session starts with the model's longest thinking setting. One reply may take up to 98,304 tokens; a thinking longer than 49,152
  tokens is closed there and the model goes on to the answer. With both models installed, the screen can switch between them.
* Everything is kept inside the harness folder. A DeepSeek Harness you already use is not touched.

What it does, what was checked and what was not: [harness/HARNESS_README.txt](harness/HARNESS_README.txt).

## How it works, briefly

* A file format of its own for the compressed model, and a runtime built around it. Compression and computation are designed together, so
  that the GPU reads less data and does less work.
* GPU kernels written for the RTX 5090.
* **Speculative decoding.** A small helper model proposes several next tokens at once, and the main model checks them in a single pass and
  accepts the ones it agrees with. The helper model is described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* A compressed key-value cache (KV5), which is what lets the full 262,144-token context fit next to the model on a 32 GB card.
* One request is generated at a time.

The source for all of this is in this repository. The tool that converts a model into the YASAKANI file format is **not** part of this
release; the published models work as they are.

**In detail:** [docs/TECHNOLOGY.md](docs/TECHNOLOGY.md) - technical notes by the author: the arithmetic behind the speed (bytes read
against memory bandwidth), the weight format and how the codes are made, computing on the compressed form, look-ahead, KV5, what each
part contributes, and the source files that go with each.

## Limits

* Windows 11 + RTX 5090 + Qwen3.8-27B only. One request at a time. Text only: no images, no `n` above 1, no `logprobs`. Tool / function calling is only in the preview package of the harness (see [Use it as an agent](#use-it-as-an-agent)).
* The release files are not code-signed, so Windows may warn about them (see [SECURITY.md](SECURITY.md)).
* The answers are not identical to the original model's (see Quality).

## Support

Development is funded by a crowdfunding campaign on CAMPFIRE: https://camp-fire.jp/projects/992233/view (until December 16, 2026). This
release does not depend on its result. The support goes to test hardware, electricity and development time for more GPUs: other RTX cards
first (24 GB and 16 GB), then AMD and Mac, in the order and under the conditions written on the campaign page.
Supporters who wish to be named are credited in this repository, and in the startup credits from a later version. News and
measurements: https://x.com/yasakani_dev

**Outside Japan:** the CAMPFIRE campaign is only for backers living in Japan (CAMPFIRE's rules do not allow digital rewards, like this
campaign's, for backers abroad). Starring this repository, trying the engine and reporting what you measure helps a lot.

## License

Apache License 2.0 - see [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model files are
derived from Qwen3.8-27B (Apache-2.0, Alibaba Cloud) and from a publicly released draft model (Apache-2.0, z-lab); they are not the
original weights.
