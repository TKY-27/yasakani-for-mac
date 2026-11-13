# Third-party notices

YASAKANI itself is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`). This file lists what else is in the release, what is
only required, and which published work it builds on.

## Redistributed with YASAKANI

### Qwen3.8-27B (model, tokenizer and chat template)

* By the Qwen team, Alibaba Cloud. License: Apache License 2.0 - the `LICENSE` file of the model repository is the Apache License 2.0 text with "Copyright 2026 Alibaba Cloud" (it is shipped with the model files; on 2026-11-12 the model page listed this license and carried the same `LICENSE` file).
* Original: <https://huggingface.co/Qwen/Qwen3.8-27B>
* The files `qwen38_27b_quality.yskn` (quality model) and `qwen38_27b_speed.yskn` (fastest model) are **derived** from it: converted to the
  YASAKANI file format and compressed (quantized). They are not the original weights. `tokenizer.json`, `tokenizer_config.json`,
  `chat_template.jinja`, `config.json` and `generation_config.json` are the original files, unchanged.

### DFlash 2 draft model for Qwen3.8-27B (helper model)

* By z-lab (a mirror of `incoai/Qwen3.8-27B-DFlash2`, credited to Inco AI). The Hugging Face page lists the Apache License 2.0.
* Original: <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2> - background: <https://inco.ai/blog/dflash2/> and <https://github.com/z-lab/dflash>
* The files `bdraft2_k8.yskn` and `bdraft2_k8.shead` are **derived** from it: converted to the YASAKANI file format, compressed (quantized) and
  re-trained by the YASAKANI authors against a compressed Qwen3.8-27B model (an earlier YASAKANI conversion). They are not the original
  weights.

### nlohmann/json 3.11.3 (source, `third_party/nlohmann/json.hpp`)

* Copyright (c) 2013-2023 Niels Lohmann <https://nlohmann.me> - MIT License - <https://github.com/nlohmann/json>
* The header is included unmodified (`third_party/nlohmann/PROVENANCE.txt` records its SHA-256).

```
MIT License

Copyright (c) 2013-2023 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### NVIDIA CUDA Runtime (inside `yskn.dll`)

* The prebuilt `yskn.dll` contains the NVIDIA CUDA Runtime library (`cudart`), linked statically (`cudart_static.lib`; the release is built
  with CUDA Toolkit 13.2: nvcc 13.2.78, CUDA Runtime 13.2.75). NVIDIA's CUDA Toolkit End User License Agreement lets an application distribute the files that its
  Attachment A lists; for the CUDA Runtime on Windows the list is `cudart.dll`, `cudart_static.lib` and `cudadevrt.lib`. The runtime inside
  `yskn.dll` is used only by the engine itself. See <https://docs.nvidia.com/cuda/eula/> (sections 1.1.1, 1.1.2, 2.2 and 2.6).
* The Apache License 2.0 of YASAKANI does **not** apply to this component. It stays under NVIDIA's agreement; please do not take it out of
  `yskn.dll` for other use.
* The agreement text was read on 2026-10-31 in its CUDA 13.3 and 13.4 versions ("last updated January 26, 2026"); both say the same about the
  runtime. The `EULA.txt` installed with the CUDA 13.2 toolkit that built the release DLL was compared on 2026-11-12: it carries the
  same date, has the sections named above, and lists the same three files for the CUDA Runtime on Windows.

## Required, but not redistributed (you install them)

| What | Why | License | Tested with |
|---|---|---|---|
| NVIDIA display driver | runs the GPU | NVIDIA driver license | 616.92 |
| Microsoft Visual C++ 2015-2022 x64 runtime | `yskn.dll` is built with Visual C++ | Microsoft software license terms | - |
| Python 3.10 or newer | the command-line tool and server | PSF License | 3.14.4 on Windows 11 (the unit tests also ran on 3.10) |
| numpy | array handling | BSD-3-Clause | 2.4.4 and 2.5.4 |
| tokenizers (Hugging Face) | tokenizer | Apache-2.0 | 0.22.2 and 0.23.3 |
| jinja2 | chat template | BSD-3-Clause | 3.1.6 |

`pip` also installs what these packages themselves depend on (for example `huggingface-hub` with `tokenizers`, and `MarkupSafe` with `jinja2`),
each under its own license; `pip list` and `pip show <package>` give the exact versions and licenses on your machine. YASAKANI loads the
tokenizer from the local model folder only and never downloads anything.

The first version in a row is that of the PC where the release candidate was tested (2026-10-31). Where there is a second one, it is
what `pip` picked in a fresh virtual environment on 2026-11-12 (jinja2 was 3.1.6 again); the self-check passed there for both models.

## Published work this builds on (no code taken from it unless listed above)

* Jian Chen, Yesheng Liang, Zhijian Liu. *DFlash: Block Diffusion for Flash Speculative Decoding.* arXiv:2602.06036, February 2026.
  The helper model above is a drafter of this kind.
* Qwen team. The Qwen3.8 model family (see above).

## Not included

The measurement kit under `bench/` refers to other engines (llama.cpp, SGLang, vLLM, TensorRT-LLM) and to test sets published by others
(NInfer, AIME 2025) by name and checksum only. Their code and data are not redistributed; each has its own license.
