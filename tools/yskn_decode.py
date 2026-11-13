#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""yskn_decode.py - read a YASAKANI model file (.yskn) without the engine.

  python yskn_decode.py info     MODEL.yskn                        what is inside
  python yskn_decode.py decode   MODEL.yskn NAME --out W.npy       one tensor back to ordinary weights (float32 .npy)
  python yskn_decode.py vector   MODEL.yskn NAME --out V.npy       one stored vector (float32 .npy)
  python yskn_decode.py compare  MODEL.yskn --original DIR         every tensor and vector against the original checkpoint
                                 [--names A,B,...] [--names-file F] [--json OUT]
                                                                   (DIR holds the *.safetensors files of the original model)
  python yskn_decode.py selftest [--data DIR]                      the test file of tools/testdata against its expected values

The file format is described in docs/FORMAT.md.  This program follows that text; it does not use the engine and needs no GPU.
`decode` gives the weights the file stands for.  With --engine-table it gives the weights the engine computes with (the engine
rounds the value table of a matrix to 8 bits, see docs/FORMAT.md, "What the engine does with it").

Needs Python 3.9 or later and numpy.  Memory: a tensor is decoded in blocks of rows; the largest matrices of the released models
(17,408 x 5,120) need about 1.5 GB while they are decoded, the embedding and the output matrix are compared block by block.
"""
import argparse
import glob
import hashlib
import json
import os
import struct
import sys

import numpy as np

MAGIC = b"YSKN"
HEADER_BYTES = 0x20
STATE_BITS = 12                       # "l" of the tensor description: the only value this program reads
SEQ_WEIGHTS = 256                     # "t": weights per sequence
STEPS = SEQ_WEIGHTS // 2              # "d" = 2 weights per step
K512 = np.float32(0.044194173824159216)          # 1 / sqrt(512)


class FormatError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------- the container
class Container:
    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        if self.size < HEADER_BYTES:
            raise FormatError("too short to be a YSKN file")
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        head = bytes(self.mm[:HEADER_BYTES])
        if head[:4] != MAGIC:
            raise FormatError("not a YSKN file (the first four bytes are not 'YSKN')")
        (self.version,) = struct.unpack_from("<I", head, 4)
        self.json_len, self.data_offset, self.file_length = struct.unpack_from("<QQQ", head, 8)
        if self.version != 1:
            raise FormatError("unsupported version %d" % self.version)
        if HEADER_BYTES + self.json_len > self.size:
            raise FormatError("the header is longer than the file")
        self.header = json.loads(bytes(self.mm[HEADER_BYTES:HEADER_BYTES + self.json_len]).decode("utf-8"))
        self.model = self.header["model"]
        self.tensors = {t["name"]: t for t in self.header["tensors"]}
        self.vectors = {v["name"]: v for v in self.header["vectors"]}

    def array(self, offset, dtype, count):
        dt = np.dtype(dtype)
        if offset + count * dt.itemsize > self.size:
            raise FormatError("data at offset %d runs past the end of the file" % offset)
        return np.ndarray((count,), dtype=dt, buffer=self.mm, offset=offset)


def checked(t):
    """The tensor description with the defaults of the reader filled in; raises for what this program does not read."""
    d = dict(t)
    d.setdefault("physical_rows", d["rows"])
    d.setdefault("l", 12), d.setdefault("d", 2), d.setdefault("t", 256)
    d.setdefault("start", "zero"), d.setdefault("layout", "warp")
    if d["l"] != STATE_BITS or d["d"] != 2 or d["t"] != SEQ_WEIGHTS or d["bits"] not in (4, 5, 6, 7, 8) or d["cols"] % 512:
        raise FormatError("%s: a tensor description this program does not read" % d["name"])
    if d["start"] != "zero":
        raise FormatError("%s: start = %r is not implemented here (the released models use \"zero\")" % (d["name"], d["start"]))
    if d.get("rar"):
        raise FormatError("%s: per-tile bit widths (\"rar\") are not implemented here (the released models do not use them)" % d["name"])
    default = dict(kind="H512", sign_seed=7, qm_off=0, rule="mulberry32_7919_13")
    if "rot_in" not in d:
        d["rot_in"] = dict(default, m=d["cols"] // 512, sign_offset=0)
    if "rot_out" not in d:                       # absent: body matrices are rotated on the output side too, with the default signs
        d["rot_out"] = None if d["role"] in ("head", "embedding") else dict(default, m=d["rows"] // 512, sign_offset=d["cols"])
    if d["rot_out"] is not None and d["rot_out"].get("kind") == "none":
        d["rot_out"] = None
    for r in (d["rot_in"], d["rot_out"]):
        if r is not None and r.get("kind") != "H512":
            raise FormatError("%s: rotation kind %r is not implemented here (the released models use \"H512\")" % (d["name"], r.get("kind")))
    if d["rot_out"] is not None and d["physical_rows"] % 512:
        raise FormatError("%s: an output rotation needs a multiple of 512 rows" % d["name"])
    return d


# -------------------------------------------------------------------------------------------------------------------- the signs
def sign_stream(rule, seed, offset, n):
    """n signs (+1 / -1, int8), starting at position `offset` of the stream that `rule` and `seed` define."""
    k = np.arange(offset, offset + n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        if rule == "splitmix64":
            z = np.uint64(seed) + (k + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
            return np.where(z >> np.uint64(63), 1, -1).astype(np.int8)
        if rule == "mulberry32_7919_13":
            s0 = ((int(seed) & 0xFFFFFFFF) * 7919 + 13) & 0xFFFFFFFF
            if s0 == 0:
                s0 = 0x9E3779B9
            t = (np.uint32(s0) + (k + np.uint64(1)).astype(np.uint32) * np.uint32(0x6D2B79F5)).astype(np.uint32)
            t = ((t ^ (t >> np.uint32(15))) * (t | np.uint32(1))).astype(np.uint32)
            t = (t ^ (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61)))).astype(np.uint32)
            t = t ^ (t >> np.uint32(14))
            return np.where(t < np.uint32(0x80000000), -1, 1).astype(np.int8)
    raise FormatError("unknown sign rule %r" % rule)


# ------------------------------------------------------------------------------------------------------------- codes -> values
def unit_words(c, t):
    """uint32 [physical_rows, sequences, 4 * bits]: the words of every sequence of every row (a view of the file)."""
    rows, seqs, k = t["physical_rows"], t["cols"] // SEQ_WEIGHTS, t["bits"]
    wps = 4 * k
    n = rows * seqs * wps
    if t["stream_bytes"] != 4 * n:
        raise FormatError("%s: stream_bytes is %d, expected %d" % (t["name"], t["stream_bytes"], 4 * n))
    words = c.array(t["stream_off"], "<u4", n)
    if t["layout"] == "row":
        return words.reshape(rows, seqs, wps)
    if t["layout"] == "warp":
        if (rows * seqs) % 32:
            raise FormatError("%s: the warp layout needs a multiple of 32 sequences" % t["name"])
        a = words.reshape(rows * seqs // 32, k, 32, 4)                     # group of 32 sequences, chunk, lane, word of the chunk
        return a.transpose(0, 2, 1, 3).reshape(seqs, rows, wps).transpose(1, 0, 2)
    raise FormatError("%s: unknown layout %r" % (t["name"], t["layout"]))


def states(units, k):
    """uint16 [..., 128]: the 12-bit state at each of the 128 steps of every sequence."""
    w = np.concatenate([units, np.zeros(units.shape[:-1] + (1,), np.uint32)], axis=-1).astype(np.uint64)
    end = k * (np.arange(STEPS) + 1)                                        # the bit position after the symbol of step j
    start = end - STATE_BITS
    out = np.empty(units.shape[:-1] + (STEPS,), np.uint16)
    full = start >= 0
    wi, sh = start[full] >> 5, (start[full] & 31).astype(np.uint64)
    window = (w[..., wi] | (w[..., wi + 1] << np.uint64(32))) >> sh
    out[..., full] = (window & np.uint64(0xFFF)).astype(np.uint16)
    for j in np.nonzero(~full)[0]:                                          # the first step(s): the bits before the stream are zero
        e = int(end[j])
        out[..., j] = ((w[..., 0] & np.uint64((1 << e) - 1)) << np.uint64(STATE_BITS - e)).astype(np.uint16)
    return out


def value_table(c, t, engine_table=False):
    lut = c.array(t["lut_off"], "<f2", 2 * (1 << STATE_BITS)).astype(np.float32).reshape(1 << STATE_BITS, 2)
    if engine_table:                                                         # what the engine does for matrices: an 8-bit table
        gamma = np.abs(lut).max() / np.float32(127)
        lut = (np.clip(np.rint(lut / gamma), -127, 127) * gamma).astype(np.float32)
    return lut


def rotated_rows(c, t, r0, r1, engine_table=False):
    """float32 [r1 - r0, cols]: rows r0 .. r1 of the tensor in rotated coordinates, the row scale applied."""
    lut = value_table(c, t, engine_table)
    scale = c.array(t["scale_off"], "<f2", t["physical_rows"]).astype(np.float32)
    s = states(unit_words(c, t)[r0:r1], t["bits"])
    return lut[s].reshape(r1 - r0, t["cols"]) * scale[r0:r1, None]


_HADAMARD = None


def hadamard512():
    """The 512 x 512 Hadamard matrix in natural order, times 1 / sqrt(512) (symmetric, its own inverse)."""
    global _HADAMARD
    if _HADAMARD is None:
        h = np.ones((1, 1), np.float32)
        while h.shape[0] < 512:
            h = np.block([[h, h], [h, -h]])
        _HADAMARD = np.ascontiguousarray(h * K512, dtype=np.float32)
    return _HADAMARD


def fwht512(x):
    """The Walsh-Hadamard transform (natural order) of every block of 512 along the last axis, times 1 / sqrt(512)."""
    shape = x.shape
    if shape[-1] % 512:
        raise FormatError("the length is not a multiple of 512")
    return (np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 512) @ hadamard512()).reshape(shape)


def signs_of(rot, n):
    return sign_stream(rot.get("rule", "splitmix64"), rot["sign_seed"], rot["sign_offset"], n).astype(np.float32)


def decode_rows(c, t, r0, r1, engine_table=False):
    """float32 [r1 - r0, cols] in the coordinates of the original model.  Only for tensors without an output rotation."""
    if t["rot_out"] is not None:
        raise FormatError("%s has an output rotation: decode it as a whole" % t["name"])
    return fwht512(rotated_rows(c, t, r0, r1, engine_table)) * signs_of(t["rot_in"], t["cols"])[None, :]


def decode(c, t, engine_table=False):
    """float32 [rows, cols]: the whole tensor in the coordinates of the original model."""
    rows = t["physical_rows"]
    w = np.empty((rows, t["cols"]), np.float32)
    sin = signs_of(t["rot_in"], t["cols"])
    for r0 in range(0, rows, 2048):
        r1 = min(rows, r0 + 2048)
        w[r0:r1] = fwht512(rotated_rows(c, t, r0, r1, engine_table)) * sin[None, :]          # the input side: columns
    if t["rot_out"] is not None:                                                             # the output side: rows
        sout = signs_of(t["rot_out"], rows)
        h = hadamard512()
        for r0 in range(0, rows, 512):                                                       # blocks of 512 rows
            w[r0:r0 + 512] = (h @ w[r0:r0 + 512]) * sout[r0:r0 + 512, None]
    return w[:t["rows"]]


def vector(c, v):
    """float32 [n]: a stored vector."""
    raw = c.array(v["off"], "<u2", v["n"])
    kind = v.get("dtype", "f16")
    if kind == "bf16":
        return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if kind == "f16":
        return raw.view("<f2").astype(np.float32)
    raise FormatError("%s: unknown vector type %r" % (v["name"], kind))


# --------------------------------------------------------------------------------------------- a second, slow reading of the codes
def naive_rotated_row(c, t, row):
    """One row in rotated coordinates, read bit by bit straight from the text of docs/FORMAT.md (slow; used by the self-test)."""
    rows, seqs, k = t["physical_rows"], t["cols"] // SEQ_WEIGHTS, t["bits"]
    words = c.array(t["stream_off"], "<u4", rows * seqs * 4 * k)
    lut = c.array(t["lut_off"], "<f2", 2 * 4096)
    scale = np.float32(c.array(t["scale_off"], "<f2", rows)[row])
    out = []
    for seq in range(seqs):
        bits = []
        for word in range(4 * k):
            if t["layout"] == "row":
                idx = (row * seqs + seq) * 4 * k + word
            else:
                q = seq * rows + row
                idx = (((q // 32) * k + word // 4) * 32 + q % 32) * 4 + word % 4
            x = int(words[idx])
            bits.extend((x >> b) & 1 for b in range(32))
        for j in range(STEPS):
            e = k * (j + 1)
            state = 0
            for i in range(STATE_BITS):
                p = e - STATE_BITS + i
                if p >= 0:
                    state |= bits[p] << i
            out.append(np.float32(lut[2 * state]) * scale)
            out.append(np.float32(lut[2 * state + 1]) * scale)
    return np.array(out, np.float32)


# ------------------------------------------------------------------------------------------------------- the original checkpoint
class Original:
    """The *.safetensors files of a folder, read without any extra package."""
    DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8"}

    def __init__(self, folder):
        self.where = {}
        for p in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            with open(p, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                head = json.loads(f.read(n))
            for k, v in head.items():
                if k != "__metadata__":
                    self.where[k] = (p, 8 + n + v["data_offsets"][0], v["dtype"], tuple(v["shape"]))
        if not self.where:
            raise FormatError("no *.safetensors file in " + folder)
        self.maps = {}

    def find(self, name):
        for key in (name, name + ".weight", "model." + name, "model." + name + ".weight",
                    "model.language_model." + name, "model.language_model." + name + ".weight"):
            if key in self.where:
                return key
        return None

    def rows(self, key, r0=None, r1=None):
        """float32: the tensor `key` (rows r0 .. r1 of its first axis, if given), every other axis flattened into the second."""
        path, off, dtype, shape = self.where[key]
        if dtype not in self.DTYPES:
            raise FormatError("%s: type %s is not read here" % (key, dtype))
        if path not in self.maps:
            self.maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        n0 = shape[0] if shape else 1
        per = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        r0, r1 = (0 if r0 is None else r0), (n0 if r1 is None else r1)
        dt = np.dtype(self.DTYPES[dtype])
        a = np.ndarray(((r1 - r0) * per,), dtype=dt, buffer=self.maps[path], offset=off + r0 * per * dt.itemsize)
        if dtype == "BF16":
            a = (a.astype(np.uint32) << np.uint32(16)).view(np.float32)
        return a.astype(np.float32, copy=False).reshape(r1 - r0, per)


def compare_tensor(c, t, orig, key, engine_table=False, block=4096):
    """Relative error  |decoded - original| / |original|  (Frobenius norm) and the largest absolute difference."""
    rows, cols = t["rows"], t["cols"]
    shape = orig.where[key][3]
    if int(np.prod(shape)) != rows * cols or shape[0] != rows:
        raise FormatError("%s: the original %s has the shape %s, the file says %d x %d" % (t["name"], key, shape, rows, cols))
    num = den = 0.0
    worst = 0.0
    if t["rot_out"] is not None:
        d = decode(c, t, engine_table)
        for r0 in range(0, rows, block):
            r1 = min(rows, r0 + block)
            o = orig.rows(key, r0, r1)
            e = d[r0:r1] - o
            num += float(np.sum(e.astype(np.float64) ** 2)); den += float(np.sum(o.astype(np.float64) ** 2)); worst = max(worst, float(np.abs(e).max()))
    else:
        for r0 in range(0, rows, block):
            r1 = min(rows, r0 + block)
            o = orig.rows(key, r0, r1)
            e = decode_rows(c, t, r0, r1, engine_table) - o
            num += float(np.sum(e.astype(np.float64) ** 2)); den += float(np.sum(o.astype(np.float64) ** 2)); worst = max(worst, float(np.abs(e).max()))
    return dict(name=t["name"], original=key, role=t["role"], rows=rows, cols=cols, bits_per_weight=t["bits"] / 2,
                relative_error=(num / den) ** 0.5 if den else None, max_abs_difference=worst, original_rms=(den / (rows * cols)) ** 0.5)


# --------------------------------------------------------------------------------------------------------------------- commands
def cmd_info(a):
    c = Container(a.model)
    print("file      %s  (%d bytes)" % (os.path.basename(a.model), c.size))
    print("header    version %d, JSON %d bytes, data from offset %d, file length in the header %d" % (c.version, c.json_len, c.data_offset, c.file_length))
    print("model     " + json.dumps({k: v for k, v in c.model.items() if k != "layer_types"}, ensure_ascii=False))
    kinds = {}
    for t in c.header["tensors"]:
        k = (t["role"], t["bits"], t.get("layout", "warp"), "output rotation" if checked(t)["rot_out"] else "no output rotation")
        n, w = kinds.get(k, (0, 0))
        kinds[k] = (n + 1, w + t["rows"] * t["cols"])
    print("tensors   %d" % len(c.tensors))
    for k in sorted(kinds):
        print("          %4d x  role %-9s  %d bits per step (%.1f per weight)  layout %-4s  %s   %.3f billion weights"
              % (kinds[k][0], k[0], k[1], k[1] / 2, k[2], k[3], kinds[k][1] / 1e9))
    print("vectors   %d  (%d numbers)" % (len(c.vectors), sum(v["n"] for v in c.vectors.values())))
    if a.names:
        for t in c.header["tensors"]:
            print("  tensor  %-44s %-9s %6d x %6d  %d bits" % (t["name"], t["role"], t["rows"], t["cols"], t["bits"]))
        for v in c.header["vectors"]:
            print("  vector  %-44s %8d  %s" % (v["name"], v["n"], v.get("dtype", "f16")))
    return 0


def cmd_decode(a):
    c = Container(a.model)
    if a.name not in c.tensors:
        raise SystemExit("no tensor named %s (see: info --names)" % a.name)
    t = checked(c.tensors[a.name])
    if t["rot_out"] is None and t["rows"] > 20000:                 # the embedding and the output matrix: block by block into the file
        out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float32, shape=(t["rows"], t["cols"]))
        for r0 in range(0, t["rows"], 4096):
            r1 = min(t["rows"], r0 + 4096)
            out[r0:r1] = decode_rows(c, t, r0, r1, a.engine_table)
        out.flush()
    else:
        np.save(a.out, decode(c, t, a.engine_table))
    print("wrote %s  float32 [%d, %d]" % (a.out, t["rows"], t["cols"]))
    return 0


def cmd_vector(a):
    c = Container(a.model)
    if a.name not in c.vectors:
        raise SystemExit("no vector named %s (see: info --names)" % a.name)
    np.save(a.out, vector(c, c.vectors[a.name]))
    print("wrote %s  float32 [%d]" % (a.out, c.vectors[a.name]["n"]))
    return 0


def cmd_compare(a):
    c = Container(a.model)
    orig = Original(a.original)
    want = set(a.names.split(",")) if a.names else None
    if a.names_file:
        want = (want or set()) | {x.strip() for x in open(a.names_file, encoding="utf-8") if x.strip()}
    rows, missing = [], []
    for t0 in c.header["tensors"]:
        if want is not None and t0["name"] not in want:
            continue
        key = orig.find(t0["name"])
        if key is None:
            missing.append(t0["name"])
            continue
        r = compare_tensor(c, checked(t0), orig, key, a.engine_table)
        rows.append(r)
        print("%-44s %4.1f bit  relative error %.4f   largest difference %.5f   (rms of the original %.5f)"
              % (r["name"], r["bits_per_weight"], r["relative_error"], r["max_abs_difference"], r["original_rms"]), flush=True)
    vec = []
    for v in c.header["vectors"]:
        if want is not None and v["name"] not in want:
            continue
        key = orig.find(v["name"])
        if key is None:
            missing.append(v["name"])
            continue
        o = orig.rows(key).reshape(-1)
        d = vector(c, v)
        same = o.size == d.size and bool(np.array_equal(o, d))
        vec.append(dict(name=v["name"], original=key, n=v["n"], identical=same,
                        max_abs_difference=None if o.size != d.size else float(np.abs(o - d).max())))
    by = {}
    for r in rows:
        by.setdefault((r["role"], r["bits_per_weight"]), []).append(r["relative_error"])
    print()
    for k in sorted(by):
        e = np.array(by[k])
        print("%-9s %.1f bit: %3d tensors, relative error  median %.4f  smallest %.4f  largest %.4f" % (k[0], k[1], len(e), np.median(e), e.min(), e.max()))
    if vec:
        n_same = sum(1 for v in vec if v["identical"])
        print("vectors: %d of %d are bit for bit the original values" % (n_same, len(vec)))
        for v in vec:
            if not v["identical"]:
                print("  differs: %-44s largest difference %s" % (v["name"], v["max_abs_difference"]))
    if missing:
        print("not found in the original checkpoint: %d (%s ...)" % (len(missing), ", ".join(missing[:4])))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(model=os.path.basename(a.model), source=c.model.get("source"), engine_table=bool(a.engine_table), tensors=rows,
                           vectors=vec, not_found=missing), f, indent=1)
        print("wrote", a.json)
    return 0


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cmd_selftest(a):
    data = a.data or os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
    c = Container(os.path.join(data, "tiny.yskn"))
    want = json.load(open(os.path.join(data, "tiny_expected.json"), encoding="utf-8"))
    bad = 0

    def check(name, ok, extra=""):
        nonlocal bad
        bad += not ok
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, ("   " + extra) if extra else ""))

    check("header", (c.version, c.json_len, c.data_offset, c.file_length) == tuple(want["header"]) and c.file_length == c.size)
    for rule, seed, offset, n, digest in want["signs"]:
        check("signs %s seed %d offset %d" % (rule, seed, offset), sha(sign_stream(rule, seed, offset, n)) == digest)
    for name, w in want["tensors"].items():
        t = checked(c.tensors[name])
        st = states(unit_words(c, t), t["bits"])
        check("%s: states" % name, sha(st.astype("<u2")) == w["states_sha256"])
        rot = rotated_rows(c, t, 0, t["physical_rows"])
        check("%s: values in rotated coordinates" % name, sha(rot.astype("<f4")) == w["rotated_sha256"])
        slow = np.stack([naive_rotated_row(c, t, r) for r in w["naive_rows"]])
        check("%s: the bit-by-bit reading gives the same rows" % name, np.array_equal(slow, rot[w["naive_rows"]]))
        d = decode(c, t)
        ref = np.array(w["sample_values"], np.float64)
        got = np.array([d[r, col] for r, col in w["sample_at"]], np.float64)
        err = float(np.abs(got - ref).max())
        check("%s: weights (%d sampled values)" % (name, len(ref)), err <= 1e-5 * max(1.0, float(np.abs(ref).max())), "largest difference %.2e" % err)
        norm = float(np.sqrt(np.sum(d.astype(np.float64) ** 2)))
        check("%s: norm of the weights" % name, abs(norm - w["frobenius"]) <= 1e-4 * w["frobenius"])
        norm_rot = float(np.sqrt(np.sum(rot.astype(np.float64) ** 2)))
        check("%s: the rotations keep the norm" % name, abs(norm - norm_rot) <= 1e-4 * norm_rot)
        e8 = decode(c, t, engine_table=True)
        check("%s: 8-bit table stays close" % name, float(np.abs(e8 - d).max()) <= w["engine_table_bound"])
    for name, w in want["vectors"].items():
        check("vector %s" % name, sha(vector(c, c.vectors[name]).astype("<f4")) == w["sha256"])
    print("RESULT:", "PASS" if not bad else "FAIL (%d)" % bad)
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description="Read a YASAKANI model file (.yskn) without the engine.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info"); p.add_argument("model"); p.add_argument("--names", action="store_true", help="list every tensor and vector")
    p = sub.add_parser("decode"); p.add_argument("model"); p.add_argument("name"); p.add_argument("--out", required=True)
    p.add_argument("--engine-table", action="store_true", help="the weights the engine computes with (8-bit value table)")
    p = sub.add_parser("vector"); p.add_argument("model"); p.add_argument("name"); p.add_argument("--out", required=True)
    p = sub.add_parser("compare"); p.add_argument("model"); p.add_argument("--original", required=True); p.add_argument("--names")
    p.add_argument("--names-file", help="a text file with one tensor or vector name per line")
    p.add_argument("--json"); p.add_argument("--engine-table", action="store_true")
    p = sub.add_parser("selftest"); p.add_argument("--data")
    a = ap.parse_args()
    try:
        return {"info": cmd_info, "decode": cmd_decode, "vector": cmd_vector, "compare": cmd_compare, "selftest": cmd_selftest}[a.cmd](a)
    except FormatError as ex:
        print("error:", ex, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
