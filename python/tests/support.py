# SPDX-License-Identifier: Apache-2.0
"""Test support: build the CPU-only stub library, make a fake model folder, optional real tokenizer folder."""
import ctypes as C
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from yasakani._native import NativeLib          # noqa: E402
from yasakani.engine import Engine               # noqa: E402
from yasakani.runtime import ModelPack           # noqa: E402

_BUILD = {}


def stub_path() -> Path:
    """Compile tests/stub/yskn_stub.c once per run."""
    if "path" not in _BUILD:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc or sys.platform == "win32":
            raise unittest.SkipTest("no C compiler for the stub library")
        d = Path(tempfile.mkdtemp(prefix="yskn_stub_"))
        out = d / "libyskn.so"
        subprocess.run([cc, "-shared", "-fPIC", "-O1", "-pthread", str(HERE / "stub" / "yskn_stub.c"), "-o", str(out)], check=True)
        _BUILD["path"] = out
    return _BUILD["path"]


def tokenizer_dir():
    d = os.environ.get("YASAKANI_TEST_MODEL_DIR")
    if d and (Path(d) / "tokenizer.json").is_file():
        return Path(d)
    return None


def make_pack(tmp: Path, name: str = "ok") -> ModelPack:
    root = tmp / name
    root.mkdir(parents=True, exist_ok=True)
    for f in ("qwen38_27b_e3x.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead", "kvq_u5.bin", "tokenizer.json", "tokenizer_config.json"):
        (root / f).write_bytes(b"x")
    return ModelPack.load(root)


class Stub:
    """The stub library plus its test-only exports."""

    def __init__(self):
        self.lib = NativeLib(str(stub_path()))
        d = self.lib.dll
        d.stub_push_script.argtypes = [C.c_void_p, C.c_int, C.c_int]
        d.stub_set_delay_us.argtypes = [C.c_int]
        d.stub_get_opt.argtypes = [C.c_void_p, C.c_int]
        d.stub_get_opt.restype = C.c_int
        d.stub_calls.restype = C.c_int
        d.stub_last_cfg.argtypes = [C.c_void_p]
        d.stub_last_cfg.restype = C.c_int
        d.stub_nhist.argtypes = [C.c_void_p]
        d.stub_nhist.restype = C.c_int
        d.stub_last_n_prompt.restype = C.c_int
        self.d = d

    def push(self, ids, loop_from=-1):
        a = np.ascontiguousarray(ids, np.int32)
        self.d.stub_push_script(a.ctypes.data, len(a), loop_from)

    def delay_us(self, us):
        self.d.stub_set_delay_us(us)

    def opt(self, engine, key):
        return self.d.stub_get_opt(engine._e, key)

    def last_cfg(self):
        out = np.zeros(44, np.int32)
        n = self.d.stub_last_cfg(out.ctypes.data)
        return out[:n].tolist()


def open_engine(tmp: Path, stub: Stub, max_ctx: int = 4096, name: str = "ok") -> Engine:
    return Engine(make_pack(tmp, name), max_ctx=max_ctx, lib=stub.lib)


def make_real_folder(name: str = "model") -> Path:
    """A model folder with the real Qwen3.8 tokenizer files (from YASAKANI_TEST_MODEL_DIR) and placeholder weight files.  The path has
    no letters in its variable part, so it can never contain the words the stub reacts to."""
    src = tokenizer_dir()
    if src is None:
        raise unittest.SkipTest("set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
    root = Path(tempfile.gettempdir()) / ("yct%d_%d" % (os.getpid(), len(list(Path(tempfile.gettempdir()).glob("yct*")))))
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for f in ("qwen38_27b_e3x.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead", "kvq_u5.bin"):
        (d / f).write_bytes(b"placeholder-" + f.encode())
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (src / f).is_file():
            shutil.copyfile(src / f, d / f)
    return d
