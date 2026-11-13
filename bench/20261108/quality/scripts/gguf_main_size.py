# -*- coding: utf-8 -*-
"""gguf_main_size.py -- the size of a GGUF file without its auxiliary layers (the tensors blk.64.*), read from the GGUF header.

Used for the "main model" sizes of the quality table: the GGUF files of Qwen3.8-27B measured there carry 15 tensors named blk.64.*
(about 0.35 GB) that llama-perplexity reports as unused.  Header reads only; the size of a tensor is the distance to the next tensor's
offset.  Prints one JSON object per file.  Python standard library only.
usage: python gguf_main_size.py FILE.gguf [FILE.gguf ...]"""
import json
import os
import struct
import sys


def gguf_spans(path):
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"; ver, = struct.unpack("<I", f.read(4)); nt, nkv = struct.unpack("<QQ", f.read(16)); align = 32
        rstr = lambda: f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
        for _ in range(nkv):
            k = rstr(); t, = struct.unpack("<I", f.read(4))
            if t == 8: rstr()
            elif t == 9:
                at, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
                if at == 8:
                    for _ in range(n): rstr()
                else: f.seek(sz[at] * n, 1)
            elif k == "general.alignment" and t == 4: align, = struct.unpack("<I", f.read(4))
            else: f.seek(sz[t], 1)
        tens = []
        for _ in range(nt):
            name = rstr(); nd, = struct.unpack("<I", f.read(4)); f.seek(8 * nd, 1); f.seek(4, 1); off, = struct.unpack("<Q", f.read(8)); tens.append((name, off))
        start = (f.tell() + align - 1) // align * align
    total = os.path.getsize(path); tens.sort(key=lambda x: x[1])
    span = {n: (tens[i + 1][1] if i + 1 < len(tens) else total - start) - o for i, (n, o) in enumerate(tens)}
    return total, start, span


def main():
    for name in sys.argv[1:]:
        total, start, span = gguf_spans(name)
        aux = sum(v for n, v in span.items() if n.startswith("blk.64."))
        print(json.dumps(dict(gguf=os.path.basename(name), file_bytes=total, auxiliary_bytes=aux,
                              auxiliary_tensors=sum(1 for n in span if n.startswith("blk.64.")), main_model_bytes=total - aux)), flush=True)


if __name__ == "__main__":
    main()
