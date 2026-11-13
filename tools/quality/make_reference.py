#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""make_reference.py - the reference file of the quality measurement, made from the original model with Hugging Face transformers.

  python make_reference.py --original DIR_OR_ID --windows windows_own_thinking.json --out REF.kld [--offload] [--gpu-memory 26GiB]
                           [--only N] [--compare OTHER.kld]
  python make_reference.py --original - --out REF.kld --compare OTHER.kld --compare-only

For every window (2,048 token ids) the original model reads the window once (teacher forcing); the next-token distribution at the
positions 1024..2046 is written in the format of llama.cpp's "--kl-divergence-base" (kld_base.py describes it).  The model runs in
bfloat16, the precision it is published in; the last step (hidden state times the output matrix) is done in 32-bit floats, so
that the distribution is not rounded to bfloat16.

--offload      keep the weights in main memory and run every block on GPU 0 (for one GPU that cannot hold the model: needs about
               60 GB of main memory).  Without it the model is spread over the devices by transformers (device_map="auto").
--only N       only the first N windows (a quick check)
--compare F    after writing: compare the new file with another reference file on the same windows (KL divergence between the
               two references, agreement of the most likely token)

Needs torch, transformers (a version that knows the model), accelerate, numpy.  The output for 20 windows is about 10 GB.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def compare(new_path, old_path, KldReader):
    """Two reference files on the same windows: KL(old || new) and the agreement of the most likely token.  The files hold log-probabilities
    rounded to 16 bits, with everything below (largest - 16) set to one floor value; only entries above the floor in both files are used."""
    new, old = KldReader(new_path), KldReader(old_path)
    n_chunk = min(new.n_chunk, old.n_chunk)
    V, first = new.V, new.first
    kl, same, nll_new, nll_old, count = 0.0, 0, 0.0, 0.0, 0
    for w in range(n_chunk):
        if not np.array_equal(np.asarray(new.tokens[w]), np.asarray(old.tokens[w])):
            raise SystemExit("window %d: the two files do not hold the same tokens" % w)
        tg = np.asarray(new.tokens[w])[first + 1:].astype(np.int64)
        for r0 in range(0, new.rows, 64):
            a, b = np.asarray(new.chunk(w)[r0:r0 + 64]), np.asarray(old.chunk(w)[r0:r0 + 64])
            n = a.shape[0]
            lp = []
            for x in (a, b):
                sc = x[:, 0:2].copy().view(np.float32)[:, 0].astype(np.float64)
                mn = x[:, 2:4].copy().view(np.float32)[:, 0].astype(np.float64)
                lp.append(sc[:, None] * x[:, 4:4 + V].astype(np.float64) + mn[:, None])
            keep = (a[:, 4:4 + V] > 0) & (b[:, 4:4 + V] > 0)
            kl += float(np.sum(np.where(keep, np.exp(lp[1]) * (lp[1] - lp[0]), 0.0)))
            same += int(np.sum(lp[0].argmax(1) == lp[1].argmax(1)))
            idx = np.arange(n)
            nll_new += float(-lp[0][idx, tg[r0:r0 + n]].sum()); nll_old += float(-lp[1][idx, tg[r0:r0 + n]].sum())
            count += n
    print("%s against %s: KL %.6f, same top %.3f %%, perplexity %.4f (other file %.4f), %d positions in %d windows"
          % (Path(new_path).name, Path(old_path).name, kl / count, 100.0 * same / count, np.exp(nll_new / count), np.exp(nll_old / count), count, n_chunk))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--original", required=True, help="folder or Hugging Face id of the original model (Qwen/Qwen3.8-27B)")
    ap.add_argument("--windows", default=str(HERE / "windows_own_thinking.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--gpu-memory", default=None, help="limit for GPU 0 with device_map=auto, for example 26GiB")
    ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--compare")
    ap.add_argument("--compare-only", action="store_true", help="do not run the model: compare the existing file --out with --compare")
    ap.add_argument("--scoring", default=str(HERE.parents[1] / "bench" / "20261108" / "quality" / "scripts"),
                    help="the folder of kld_base.py (default: bench/20261108/quality/scripts of this repository)")
    a = ap.parse_args()
    sys.path.insert(0, a.scoring)
    from kld_base import KldReader, KldWriter
    if a.compare_only:
        compare(a.out, a.compare, KldReader)
        return 0
    import torch
    from transformers import AutoModelForImageTextToText

    spec = json.loads(Path(a.windows).read_text(encoding="utf-8"))
    windows = np.array([w["ids"] for w in spec["windows"]], np.int32)
    if a.only:
        windows = windows[:a.only]
    n_chunk, n_ctx = windows.shape
    first = n_ctx // 2

    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if a.offload:
        kw["device_map"] = {"": "cpu"}
    else:
        kw["device_map"] = "auto"
        if a.gpu_memory:
            kw["max_memory"] = {0: a.gpu_memory, "cpu": "200GiB"}
    t0 = time.perf_counter()
    model, info = AutoModelForImageTextToText.from_pretrained(a.original, output_loading_info=True, **kw)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise SystemExit("the checkpoint does not fill the model: missing %s, mismatched %s"
                         % (info.get("missing_keys")[:5], info.get("mismatched_keys")[:5]))
    model.eval()
    if a.offload:
        from accelerate import cpu_offload
        cpu_offload(model, execution_device=0)
    dev = torch.device("cuda:0")
    vocab = model.lm_head.weight.shape[0]
    print("model loaded in %.0f s: %s, vocabulary %d, %d windows of %d tokens" % (time.perf_counter() - t0, type(model).__name__, vocab, n_chunk, n_ctx),
          flush=True)

    seen = {}
    model.lm_head.register_forward_hook(lambda mod, inp, out: seen.__setitem__("h", inp[0].detach()))
    writer = KldWriter(a.out, windows, vocab)
    full = np.zeros((n_ctx, vocab), np.float32)
    for w in range(n_chunk):
        ids = torch.from_numpy(windows[w].astype(np.int64))[None].to(dev)
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
            h = seen.pop("h")[0, first:n_ctx - 1].to(dev, torch.float32)            # the input of the output matrix, rows first .. n_ctx - 2
            head = model.lm_head.weight
            if head.is_meta:                                                       # --offload: the weights wait in main memory
                head = model.lm_head._hf_hook.weights_map["weight"]
            for v0 in range(0, vocab, 16384):                                      # in 32-bit floats, a part of the vocabulary at a time
                v1 = min(vocab, v0 + 16384)
                full[first:n_ctx - 1, v0:v1] = (h @ head[v0:v1].to(dev, torch.float32).T).cpu().numpy()
        writer.add_chunk(full)
        print("window %2d written (%.0f s)" % (w, time.perf_counter() - t0), flush=True)
    writer.close()
    print("wrote", a.out)

    if a.compare:
        compare(a.out, a.compare, KldReader)
    return 0


if __name__ == "__main__":
    sys.exit(main())
