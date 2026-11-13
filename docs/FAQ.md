# FAQ

日本語は [FAQ.ja.md](FAQ.ja.md)。

## What is YASAKANI?

An inference engine for running **Qwen3.8-27B** on your own Windows PC with **one NVIDIA RTX 5090**. It is GPU kernels, a file format for the
compressed model, and a runtime, with an OpenAI-compatible server, a terminal chat, a benchmark and a self-check. One person in Japan built it. It is
not a general-purpose engine: one card, one model.

## Which model should I download?

There are two conversions of Qwen3.8-27B, and you need only one. The **quality model** (12.37 GB) stays closest to the original model and is the
one recommended for everyday use. The **fastest model** (10.31 GB) is smaller and writes faster in most kinds of task, and is a little further
from the original model. The numbers for both are in the [README](../README.md).

## How fast is it, and how was that measured?

On one RTX 5090 with Qwen3.8-27B, YASAKANI's fastest model wrote answers faster than each of the eight other engines we measured on the same PC,
in all seven task categories, with thinking off and with thinking on. For example, with thinking off it wrote 1,222 tokens per second on the
code-edit task, against 441 for MegaCapybara 1.39 (about 2.77 times), and 1,260 on long JSONL.

"Speed" is the speed of **writing the answer**: (output tokens - 1) / (time from the first to the last content chunk). The time to read the prompt
is not included. Each engine ran with its own weights, quantization, key-value cache format and speed-up settings, one request at a time. With
thinking off, runs that fell into a repetition loop or made an unrequested tool call are left out of the average by one rule for all engines;
with thinking on, the fastest run of each task is used, chosen from 5 runs for YASAKANI and from 2 for the other engines. The tables are in the
[README](../README.md); the conditions, the tallies without these choices and what the tables do not show are in
[MEASUREMENTS.md](MEASUREMENTS.md). [`bench/`](../bench/) has what you need to check or repeat the comparison, and `yasakani bench` measures
your own PC.

## Will I see the same speed?

Not exactly. The speed depends on what you ask: predictable text (code, structured output) is written faster than free-form prose, because the
engine proposes several words at once and keeps the ones the main model agrees with. Your GPU's clocks, cooling and the other programs using it
matter too. Run `yasakani bench` to see your own numbers.

## Are the answers the same as the original model's?

No. The models are compressed (the original BF16 weights are 53.8 GB; the quality model is 12.37 GB and the fastest model 10.31 GB), so they
are **not identical** to the original. On 20 thinking transcripts that were not used when the files were made (20,460 positions, FP16 key-value
cache, measured November 8, 2026), the most likely next token is the same as the original BF16 model's **95.24%** of the time for the quality
model and **93.01%** for the fastest model, and the KL divergence is 0.0122 and 0.0304. These numbers say how close the next-token predictions
are; they are not the rate of correct answers on tasks, and they cover thinking text. On general web text an additional evaluation showed a KL
divergence 3-10% larger for the quality model than for our earlier conversion of the same size.

Does the speed-up itself change the answers? It is built not to: speculative decoding only keeps tokens that the main model agrees with.
In our checks of November 10, 2026, repeated runs of the same request were identical in all 240 runs while nothing else was using the GPU.
While other programs were using the GPU at the same time, a few runs had the same tokens as the others up to one position and continued
differently from there (1 of 190 runs with the default settings, 21 of 432 with speculative decoding switched off; 256 tokens each).
We have not found the cause yet. The release does not promise bit-identical text from run to run; the details are in
[MEASUREMENTS.md](MEASUREMENTS.md).

## Which GPUs, which systems?

Only **Windows 11 and the RTX 5090** (32 GB) are supported and tested. The kernels are written for this card (compute capability 12.0). Other
NVIDIA cards have not been tested with this release; results from them are welcome (see [CONTRIBUTING.md](../CONTRIBUTING.md)). Linux, macOS and
AMD GPUs are not supported, and there is no Linux build. Extending support - 24 GB and 16 GB RTX cards first, then AMD and Mac - is what the
crowdfunding campaign is for; the order and the conditions are on the [campaign page](https://camp-fire.jp/projects/992233/view).

## How much memory and disk does it need?

One model folder is a download of about 13 GB (fastest model) or 15 GB (quality model); keep 20 GB free. The context length goes up to 262,144
tokens. With the fastest model and that full context reserved, the engine takes 16.78 GiB of VRAM (27.13 GiB if the key-value cache were kept
in FP16); this is the amount reserved at start, not the peak of the whole setup. 32 GB of system RAM is recommended.

Together with the helper model that speeds up generation, the GPU memory in use at the full 262,144-token context was about 19.1 GiB for the
fastest model and about 21.2 GiB for the quality model (about 14.3 GiB and 16.4 GiB with `--max-ctx 32768`; November 11, 2026).

## It is much slower than the tables

Check the GPU memory first. When other programs (games, other LLM servers) use GPU memory and it runs out, there is no error: generation just
becomes several times slower. In that case a warning that the GPU memory is almost full is printed at start. Close the other programs, or
lower `--max-ctx`. The speed also depends on what you ask (see "Will I see the same speed?" above).

## It only thinks and never answers

By default the model thinks before it answers, at its longest thinking setting. On requests that ask for a certain number of characters, or for
a whole program in one go, the thinking alone can take thousands to tens of thousands of tokens. The output limit (max tokens) includes the
thinking, so with a small limit the request ends in the middle of the thinking and the answer is empty. Raise the limit (the server's default is
32,768) or shorten the thinking: `/effort medium`, `/effort low` or `/think off` in `yasakani chat`; `reasoning_effort` (`medium`, `low`) or
`{"enable_thinking": false}` in `chat_template_kwargs` through the API.

## How do I connect another app?

Any app that can talk to an OpenAI-compatible server: set the address to `http://127.0.0.1:8765/v1` and the model name to `qwen3.8-27b` (any
API key will do). If it does not connect:

* An app inside Docker, or anything outside this computer: by default only connections addressed to `127.0.0.1` are accepted. Start the server
  with `yasakani serve --host 0.0.0.0 --api-key YOUR_KEY` and give the app that key.
* Apps that call the server straight from a web page: calls from other pages are refused, for safety.
* Apps that depend on tool / function calling (coding agents, for example): the 0.1.0 package answers a request with `tools` with 400. The
  harness on the release page brings a preview of the next package, whose server supports `tools`, together with an agent screen (see
  "Use it as an agent" in the [README](../README.md)).

## Can I use other models, for example Qwen4?

Not in this release: it runs Qwen3.8-27B only. The tool that converts a model to the YASAKANI file format is **not** part of this release; the
published models work as they are. Whether other models can be supported is decided one by one, and nothing is promised.

## Does it send my prompts anywhere?

No. The Python package makes no outgoing network connections (it only listens, on `127.0.0.1` unless you choose otherwise), and the engine library
imports no networking functions. After the download it needs no internet connection. Details and the checks behind this statement are in
[SECURITY.md](../SECURITY.md).

## Windows warns me about the download. Is that normal?

The release files are not code-signed, so Windows SmartScreen or an antivirus program may warn about them. Compare the SHA-256 checksum of what you
downloaded with the one on the release page, and run `yasakani verify` on the model folder. You can also build `yskn.dll` from the source.

## Can I use it from my own programs?

Yes: it serves the OpenAI chat-completions API on `http://127.0.0.1:8765/v1`. Supported request fields, the way the thinking text is returned and the
loop guard are described in [python/README.md](../python/README.md). It generates one request at a time; images, `n` above 1
and `logprobs` are not supported, and tool / function calling only in the preview package of the harness.

## Can I use it commercially?

YASAKANI is under the Apache License 2.0, which allows commercial use; Qwen3.8-27B (Alibaba Cloud) and the helper model are published under Apache-2.0
as well. The licenses, copyright notices and what was converted or re-trained are in [LICENSE](../LICENSE), [NOTICE](../NOTICE) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Read them for your own case; this FAQ is not legal advice.

## Who made the model, and what did you change?

Qwen3.8-27B is the work of Alibaba's Qwen team. YASAKANI is the engine and the file format that run it fast, and the model files are converted,
compressed versions of it. The small helper model that speeds up generation is built on a publicly released model: it was converted, compressed and
re-trained. [NOTICE](../NOTICE) says where each comes from.

## How do I report a problem?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for bugs and test results, and [SECURITY.md](../SECURITY.md) for security problems.
