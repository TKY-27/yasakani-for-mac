#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""yskn_fixture.py - write the small test file tools/testdata/tiny.yskn and the values a reader must get from it.

  python yskn_fixture.py [--out DIR]          (default: the folder testdata next to this file)

The test file has the same structure as a model file (docs/FORMAT.md) but holds no model: its code streams, value tables and
scales are pseudo-random numbers from the generator below, so that the file can be made again, byte for byte, on any computer.
It shows how a container is laid out; it is not a converter (it does not choose codes for given weights).

tiny_expected.json lists, for every tensor: the SHA-256 of the states and of the values in rotated coordinates, a few rows to
read bit by bit, some weights at given positions, and the Frobenius norm of the weights.  `python yskn_decode.py selftest`
checks a reader against it.  Needs Python 3.9 or later and numpy.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yskn_decode as Y  # noqa: E402

DATA_OFFSET = 4096


def stream64(seed, n):
    """n pseudo-random 64-bit numbers (splitmix64), the same on every computer."""
    k = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + k * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def uniform(seed, n, lo, hi):
    """n numbers between lo and hi, exactly representable steps of 1 / 4096."""
    u = (stream64(seed, n) >> np.uint64(52)).astype(np.float64) / 4096.0
    return lo + (hi - lo) * u


TENSORS = [  # name, role, rows, cols, bits, layout, rule, seed, output rotation
    ("tiny.body6", "body", 512, 512, 6, "warp", "mulberry32_7919_13", 7, True),
    ("tiny.body8", "body", 512, 1024, 8, "warp", "splitmix64", 11, True),
    ("tiny.head", "head", 64, 512, 8, "warp", "mulberry32_7919_13", 7, False),
    ("tiny.embed", "embedding", 96, 512, 6, "row", "mulberry32_7919_13", 7, False),
]


def build():
    blobs, pos = [], DATA_OFFSET

    def put(b):
        nonlocal pos
        pad = (-pos) % 64
        blobs.append(b"\0" * pad + b)
        pos += pad
        off = pos
        pos += len(b)
        return off

    tensors, vectors = [], []
    for i, (name, role, rows, cols, bits, layout, rule, seed, out_rot) in enumerate(TENSORS):
        n_words = rows * (cols // 256) * 4 * bits
        words = (stream64(1000 + i, n_words) & np.uint64(0xFFFFFFFF)).astype("<u4")
        lut = uniform(2000 + i, 2 * 4096, -3.0, 3.0).astype("<f2")
        scale = uniform(3000 + i, rows, 0.01, 0.03).astype("<f2")
        d = dict(name=name, role=role, rows=rows, cols=cols, physical_rows=rows, l=12, d=2, t=256, bits=bits, start="zero", layout=layout,
                 rot_in=dict(kind="H512", m=cols // 512, sign_seed=seed, sign_offset=0, qm_off=0, rule=rule),
                 rot_out=(dict(kind="H512", m=rows // 512, sign_seed=seed, sign_offset=cols, qm_off=0, rule=rule) if out_rot else None),
                 rar=None, bias_off=0, proxy=0.0, encoder="test data (pseudo-random, not a model)")
        d["stream_off"], d["stream_bytes"] = put(words.tobytes()), 4 * n_words
        d["lut_off"] = put(lut.tobytes())
        d["scale_off"] = put(scale.tobytes())
        tensors.append(d)
    v = uniform(4000, 512, -2.0, 2.0).astype(np.float32)
    vectors.append(dict(name="tiny.norm", off=put((v.view("<u4") >> 16).astype("<u2").tobytes()), n=512, dtype="bf16"))
    h = uniform(4001, 8, -2.0, 2.0).astype("<f2")
    vectors.append(dict(name="tiny.half", off=put(h.tobytes()), n=8, dtype="f16"))
    model = dict(arch="test", layers=0, hidden=512, ffn=512, heads=1, kv_heads=1, head_dim=512, vocab=96, eps=1e-6, rope_theta=10000.0,
                 tie_embeddings=False, source="test data of tools/yskn_fixture.py (not a model)")
    head = json.dumps(dict(model=model, tensors=tensors, vectors=vectors), separators=(",", ":")).encode("utf-8")
    if 0x20 + len(head) > DATA_OFFSET:
        raise SystemExit("the header does not fit before the data")
    body = b"".join(blobs)
    total = DATA_OFFSET + len(body)
    out = b"YSKN" + struct.pack("<IQQQ", 1, len(head), DATA_OFFSET, total) + head
    return out + b"\0" * (DATA_OFFSET - len(out)) + body


def expected(path):
    c = Y.Container(path)
    want = dict(header=[c.version, c.json_len, c.data_offset, c.file_length], signs=[], tensors={}, vectors={})
    for rule, seed, offset, n in (("mulberry32_7919_13", 7, 0, 5120), ("mulberry32_7919_13", 7, 5120, 17408), ("mulberry32_7919_13", 0, 3, 64),
                                  ("splitmix64", 11, 0, 1024), ("splitmix64", 0, 1024, 512)):
        want["signs"].append([rule, seed, offset, n, Y.sha(Y.sign_stream(rule, seed, offset, n))])
    for name, t0 in c.tensors.items():
        t = Y.checked(t0)
        rows, cols = t["rows"], t["cols"]
        d = Y.decode(c, t)
        at = [[(r * 37 + 5) % rows, (r * 101 + 17) % cols] for r in range(12)]
        want["tensors"][name] = dict(
            states_sha256=Y.sha(Y.states(Y.unit_words(c, t), t["bits"]).astype("<u2")),
            rotated_sha256=Y.sha(Y.rotated_rows(c, t, 0, rows).astype("<f4")),
            naive_rows=[0, rows // 3, rows - 1],
            sample_at=at, sample_values=[float(d[r, col]) for r, col in at],
            frobenius=float(np.sqrt(np.sum(d.astype(np.float64) ** 2))),
            engine_table_bound=float(np.abs(Y.decode(c, t, engine_table=True) - d).max()) * 1.25)
    for name, v in c.vectors.items():
        want["vectors"][name] = dict(sha256=Y.sha(Y.vector(c, v).astype("<f4")))
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = build()
    path = os.path.join(a.out, "tiny.yskn")
    with open(path, "wb") as f:
        f.write(data)
    with open(os.path.join(a.out, "tiny_expected.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(expected(path), f, indent=1)
        f.write("\n")
    print("wrote tiny.yskn (%d bytes, sha256 %s) and tiny_expected.json" % (len(data), Y.hashlib.sha256(data).hexdigest()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
