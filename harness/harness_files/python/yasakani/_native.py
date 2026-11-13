# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""ctypes binding of the engine library (yskn.dll).

Only the exports the Python layer needs are bound.  The library is loaded by absolute path; it needs the Visual C++ runtime and an
NVIDIA driver, nothing else (the CUDA runtime is linked in statically).
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

P = C.c_void_p
I = C.c_int


class YasakaniError(RuntimeError):
    """An engine or setup error with a message meant for the person running the program."""


# name -> (argument types, result type).  Pointers are passed as plain integers (numpy ``arr.ctypes.data``).
SIGNATURES: Dict[str, Tuple[list, object]] = {
    "yskn_open": ([C.c_char_p, I, I, I], P),
    "yskn_close": ([P], None),
    "yskn_reset": ([P], I),
    "yskn_prefill": ([P, P, I, P], I),
    "yskn_decode": ([P, P, I], I),
    "yskn_logits": ([P, P], I),
    "yskn_position": ([P], I),
    "yskn_nll": ([P, P, I, P], I),
    "yskn_last_error": ([], C.c_char_p),
    "yskn_cancel": ([P], I),
    "yskn_set_opt": ([P, I, I], I),
    "yskn_load_bdraft": ([P, C.c_char_p], I),
    "yskn_spec_generate": ([P, P, I, I, P, I, P, P, P], I),
    "yskn_state_hash": ([P, P, I, P], I),
}

_VC_HINT = ("The engine library could not be loaded.  On Windows it needs the Microsoft Visual C++ 2015-2022 x64 runtime "
            "(https://aka.ms/vs/17/release/vc_redist.x64.exe) and a recent NVIDIA driver.  Details: ")


def _candidates(explicit: Optional[str]):
    if explicit:
        yield Path(explicit)
    env = os.environ.get("YASAKANI_DLL")
    if env:
        yield Path(env)
    name = "yskn.dll" if sys.platform == "win32" else "libyskn.so"
    here = Path(__file__).resolve().parent
    for d in (here / "bin", here.parent / "bin", here.parents[1] / "build" / "bin", here.parents[1] / "build" / "Release",
              here.parents[1] / "bin"):
        yield d / name


def find_library(explicit: Optional[str] = None) -> Path:
    """The engine library: the given path, ``YASAKANI_DLL``, or a ``bin`` / ``build`` folder next to the package."""
    tried = []
    for p in _candidates(explicit):
        tried.append(str(p))
        if p.is_file():
            return p
    raise YasakaniError("engine library (yskn.dll) not found; set YASAKANI_DLL or pass --dll.  Looked in: " + "; ".join(tried))


class NativeLib:
    """The loaded library with typed entry points in ``fn``.  ``last_error()`` is thread-local in the engine: call it on the thread
    that made the failing call."""

    def __init__(self, path: Optional[str] = None):
        self.path = find_library(path)
        try:
            self.dll = C.CDLL(str(self.path))
        except OSError as ex:
            raise YasakaniError(_VC_HINT + str(ex)) from ex
        self.fn = {}
        missing = []
        for name, (argtypes, restype) in SIGNATURES.items():
            f = getattr(self.dll, name, None)
            if f is None:
                missing.append(name)
                continue
            f.argtypes = argtypes
            f.restype = restype
            self.fn[name] = f
        if missing:
            raise YasakaniError("the engine library is missing exports: " + ", ".join(missing))

    def last_error(self) -> str:
        m = self.fn["yskn_last_error"]()
        return m.decode("utf-8", "replace") if m else ""
