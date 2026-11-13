# yasakani (Python package)

Run **Qwen3.8-27B** on one **NVIDIA RTX 5090** under **Windows** with the YASAKANI engine library (`yskn.dll`).

This package is the part you talk to: a local **OpenAI-compatible server**, a **terminal chat**, a **benchmark**, and two checks that tell you
whether your files and your installation are the reference ones. The engine itself is the DLL; the package loads it, applies the
default settings the published numbers were measured with, handles the chat template and the tokenizer, and streams the answer.

Status: first public release (`0.1.0`). Text only. One request is generated at a time.

## What you need

| | |
|---|---|
| GPU | NVIDIA **RTX 5090** (the engine is built for this card only) and a current NVIDIA driver |
| OS | **Windows 11**, 64-bit (the only version tested), with the *Microsoft Visual C++ 2015-2022 x64 runtime* ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe)) |
| Python | 3.10 or newer |
| Disk | about 15 GB for the quality model folder, about 13 GB for the fastest model folder |
| Engine library | `yskn.dll` from the release page |

You do **not** need the CUDA Toolkit, PyTorch, or a C++ compiler.

## Model folder

YASAKANI publishes two model folders; you need one. Both hold the same files except for the weights:

```
qwen38_27b_quality.yskn   the weights of the quality model (12.4 GB)   - or -
qwen38_27b_speed.yskn     the weights of the fastest model (10.3 GB)
bdraft2_k8.yskn           small draft model used to speed up generation (1.2 GB)
bdraft2_k8.shead          its output table (1.4 GB)
kvq_u5.bin                settings of the compressed key-value cache (17 KB)
tokenizer.json            } from the Qwen3.8-27B repository, unchanged
tokenizer_config.json     }
chat_template.jinja       }
yasakani.json             file list with SHA-256 checksums; it also names the weights file
selfcheck.json            reference output for `yasakani selfcheck` (see below)
```

## Install

```bat
pip install .
yasakani info --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

`info` checks Python, the three libraries (numpy, tokenizers, jinja2), the DLL, the GPU and the model folder, and tells you what is missing.
Put `yskn.dll` in a `bin` folder next to the package, or point to it with `--dll` / the `YASAKANI_DLL` environment variable. The model folder
can be given with `--model-dir` or `YASAKANI_MODEL_DIR`.

## Commands

| command | what it does |
|---|---|
| `yasakani info` | checks this computer, the DLL and (with `--model-dir`) the model folder |
| `yasakani verify` | compares every model file with the checksums in `yasakani.json` (reads 13 to 15 GB, up to a minute or two) |
| `yasakani selfcheck` | runs three short fixed greedy prompts and compares the output with `selfcheck.json`: *does this installation reproduce the reference output?* |
| `yasakani chat` | chat in the terminal. `/think on\|off`, `/effort low\|medium\|xhigh`, `/reset`, `/exit`; Ctrl+C stops the answer being written |
| `yasakani serve` | starts the OpenAI-compatible server (below) |
| `yasakani bench` | speed measurement: greedy, fixed length, your own prompts with `--prompts-file`; `--json` for a machine-readable result |

Loading the model takes about a minute; `serve` answers `/health` with `503 loading` until it is ready.

## The server

```bat
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

It listens on `127.0.0.1:8765` (this computer only). Point any OpenAI-compatible client at `http://127.0.0.1:8765/v1`; the model id is `qwen3.8-27b`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.reasoning_content)   # the thinking (the model thinks first by default)
print(r.choices[0].message.content)             # the answer
```

```bat
curl http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

Supported request fields: `messages` (text only), `max_tokens` / `max_completion_tokens`, `temperature` (0-2; 0 is greedy), `top_p`, `top_k` (at most 20),
`min_p`, `seed`, `stop` (up to 4 strings), `stream` with `stream_options.include_usage`, `reasoning_effort` (`low`, `medium`, `high`, `xhigh`),
`chat_template_kwargs` (for example `{"enable_thinking": false}`), `ignore_eos`, `loop_guard`. The defaults are the model's own:
temperature 1.0, top_p 0.95, top_k 20. Not supported (answered with 400): images, `tools` / `functions`, `n` other than 1, `logprobs`.

* The thinking comes back as `reasoning_content` (and as `delta.reasoning_content` when streaming), the answer as `content`.
* By default the model thinks first, at its longest thinking setting, and `max_tokens` counts the thinking too. With a small limit a request can
  end while the model is still thinking (`finish_reason` is `length` and `content` is empty). Raise the limit, or shorten the thinking with
  `reasoning_effort` (`medium`, `low`), or switch it off with `{"enable_thinking": false}` in `chat_template_kwargs`.
* A stream carries one chunk per token. Where one character is split over several tokens, its text comes with the last of them, so the number
  of chunks is slightly smaller than the number of tokens; the exact count is `usage.completion_tokens` (ask for it with
  `stream_options: {"include_usage": true}`). The chunks arrive in small groups, about 64 times a second.
* **Loop guard.** If the thinking falls into an endless repetition, the guard cuts it, asks the model to answer from what it has written, and reports this in
  `usage.loop_guard`. Pass `"loop_guard": false` to turn it off.
* Input, thinking and answer together are limited to the context length (262,144 tokens by default; `--max-ctx` lowers it and the memory it needs).
* Asking again with the earlier conversation in front of a new question does not recompute the earlier turns (`usage.prompt_tokens_details.cached_tokens`).

### Safety of the server

* By default only this computer can connect. The `Host` header must name the address the server is bound to, and a browser request from another web page
  (`Origin` / `Sec-Fetch-Site`) is refused, so a web page open in your browser cannot use your GPU.
* To serve other computers use `--host 0.0.0.0 --api-key YOUR_KEY` (or `YASAKANI_API_KEY`). A non-local address without a key is refused at start.
  The server speaks plain HTTP: put it behind a VPN or a TLS proxy when it leaves your LAN.

## The self-check

`selfcheck.json` holds, for three short fixed prompts, a hash of the prompt ids and a hash of the greedy output ids made with the reference installation.
`yasakani selfcheck` produces the same records on your computer and compares them. A *prompt ids differ* result means the tokenizer or template files are
not the reference ones; an *output differs* result means the weights, the DLL, or the GPU/driver behave differently from the reference. Send the result
with your report in that case.

The three prompts are short on purpose. Output is not guaranteed to be bit-identical from run to run while other programs use the GPU at the
same time (see `docs/MEASUREMENTS.md` in the repository). In our checks the three self-check requests were identical in all 180 repeats and
the command passed 12 of 12 times, but if a case fails, run the command once more before you report it.

## Use it from Python

```python
from yasakani import Engine, ModelPack
from yasakani.chat import ChatParams, ChatRunner
from yasakani.textio import ChatFormatter

pack = ModelPack.load(r"D:\models\qwen38-quality")
with Engine(pack, max_ctx=32768, dll=r"D:\yasakani\yskn.dll") as eng:
    runner = ChatRunner(eng, ChatFormatter(pack.root))
    runner.warmup()
    params = ChatParams(messages=[{"role": "user", "content": "Hello!"}], max_tokens=512, template_kwargs={"enable_thinking": False})
    usage = runner.run(params, lambda kind, text: print(text, end="", flush=True))   # kind is "reasoning" or "content"
    print(usage.completion_tokens, "tokens")
```

`Engine.generate(prompt_ids, max_new, on_tokens=...)` works at the id level; `Engine.cancel()` stops a running generation from any thread.

## Settings and environment

The package sets the engine's environment variables itself (`yasakani.runtime.DEFAULT_ENV`) and removes any other `YSKN_*` variable left in your shell
before it opens the model, because a stray setting silently changes the engine. The published results were measured with the defaults; changing them
means running an engine configuration that the self-check and the published numbers do not cover.

## Troubleshooting

* *engine library not found / could not be loaded* - give `--dll`, and install the Visual C++ runtime linked above.
* *could not open the model ... not enough free GPU memory* - close other programs that use the GPU, or lower `--max-ctx`.
* *much slower than the published numbers* - look for the warning *the GPU memory is almost full* at start. When other programs use GPU memory
  and it runs out, Windows keeps part of the model in ordinary memory; there is no error, but generation becomes several times slower.
  Close the other programs, or lower `--max-ctx`.
* *only thinking, no answer* - the length limit was reached while the model was still thinking; see the note on `max_tokens` above
  (`/effort medium`, `/effort low` or `/think off` in `yasakani chat`).
* *cannot listen on 127.0.0.1:8765* - another program (or a second server) uses the port; pick another with `--port`.
* *output differs in selfcheck* - run `yasakani verify` first (a damaged download is the most common cause), then run the self-check once more.

## Tests

```
python -m unittest        # no GPU needed: the engine library is replaced by a small stand-in (tests/stub)
```

Set `YASAKANI_TEST_MODEL_DIR` to a folder containing the Qwen3.8 `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` to run the chat, server and
command-line tests as well.

## License

Apache License 2.0.
