# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The default runtime: environment settings, speculation settings and the model package.

The values below are the configuration the published numbers were measured with.  Change them only to experiment: a different
setting is a different engine configuration and its output is not covered by the published checks.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ._native import YasakaniError

# ---- Qwen3.8 ids used by the engine settings --------------------------------------------------------------------------------------
EOS_IDS = (248046, 248044)         # <|im_end|>, <|endoftext|>
IM_START = 248045                  # <|im_start|>: the prompt snapshot is taken right after its last occurrence
VOCAB_SIZE = 248320
HIDDEN = 5120

# ---- environment: read by the engine library when the model is opened -------------------------------------------------------------
# Value settings.  The two path settings (YSKN_BDR_SHEAD, YSKN_TB_PROFILE) come from the model package.
DEFAULT_ENV: Dict[str, str] = {
    "YSKN_BDR_SHEAD_M": "512",
    "YSKN_Q35_G6": "1",
    "YSKN_Q35_KV8": "1",
    "YSKN_Q35_KV_FP16": "0",
    "YSKN_Q35_LAYER_OVERLAP": "1",
    "YSKN_Q35_LAYER_ROWS": "4096",
    "YSKN_Q35_PFALT": "1",
    "YSKN_Q35_PMAX": "1024",
    "YSKN_Q35_SHARED_WEIGHTS": "1",
    "YSKN_Q35_ATX": "1",
    "YSKN_TB_CAP": "1",
    "YSKN_TB_VIT2": "1",
    "YSKN_ATX_PK3_PLUT": "1",
    "YSKN_ATX_PKV": "3",
    "YSKN_ATX_VSP": "7",
}
PATH_ENV = ("YSKN_BDR_SHEAD", "YSKN_TB_PROFILE")

# ---- speculation settings: the 44 integers of the generation call, in the order the engine reads them ----------------------------
SPEC_FIELDS = ("gamma", "use_d0", "use_d1", "d0_nmin", "d0_nmax", "d0_priority_n", "extend", "d1_rejected", "auto_gamma", "use_graph",
               "use_d3", "d3_steps", "d1_bigram", "d1_static", "d3_extend", "d3_lazy", "d3_gate", "gamma_long", "long_n", "d1_trigram",
               "eos1", "eos2", "long_d0", "sel", "cost_tok_us", "cost_row_us", "cost_nd_us", "nd_auto", "prmode", "cost_row2_us",
               "lfrow", "lfrow_q_pm", "mark_tok", "cost_tok_k", "cost_row2_k", "cost_nd_k", "bdraft", "bdr_long_n", "bdr_adapt",
               "bdr_probe", "bdr_d0_len", "prefill_head", "snap", "snap_tok")
assert len(SPEC_FIELDS) == 44

DEFAULT_SPEC: Dict[str, int] = dict(
    gamma=7, use_d0=1, use_d1=1, d0_nmin=2, d0_nmax=8, d0_priority_n=3, extend=1, d1_rejected=1, auto_gamma=0, use_graph=1,
    use_d3=0, d3_steps=0, d1_bigram=1, d1_static=0, d3_extend=0, d3_lazy=0, d3_gate=0, gamma_long=15, long_n=6, d1_trigram=1,
    eos1=EOS_IDS[0], eos2=EOS_IDS[1], long_d0=8, sel=0, cost_tok_us=0, cost_row_us=0, cost_nd_us=0, nd_auto=0, prmode=0,
    cost_row2_us=0, lfrow=0, lfrow_q_pm=300, mark_tok=-1, cost_tok_k=0, cost_row2_k=0, cost_nd_k=0, bdraft=1, bdr_long_n=4,
    bdr_adapt=1, bdr_probe=4, bdr_d0_len=15, prefill_head=1, snap=1, snap_tok=IM_START)
assert tuple(DEFAULT_SPEC) == SPEC_FIELDS

# settings that decide how the engine keeps its state between calls; not tunable per request
STATE_FIELDS = ("prefill_head", "snap", "snap_tok")

# statistics returned by the generation call (index in the 48 doubles)
STAT_INDEX = dict(passes=0, tokens=1, produced=30, eos=31, snap_mode=41, cached_tokens=42, bdr_rounds=37, bdr_d0=38)


def spec_array(spec: Dict[str, int]):
    import numpy as np
    unknown = set(spec) - set(SPEC_FIELDS)
    if unknown:
        raise ValueError("unknown speculation setting(s): " + ", ".join(sorted(unknown)))
    return np.array([int(spec[k]) for k in SPEC_FIELDS], np.int32)


@dataclass
class RuntimeProfile:
    """Environment variables applied before the model is opened."""
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls, pack: "ModelPack", overrides: Optional[Dict[str, str]] = None) -> "RuntimeProfile":
        env = dict(DEFAULT_ENV)
        env["YSKN_BDR_SHEAD"] = str(pack.draft_head)
        env["YSKN_TB_PROFILE"] = str(pack.kv_profile)
        for k, v in (overrides or {}).items():
            if not k.startswith("YSKN_"):
                raise ValueError("engine settings start with YSKN_: " + k)
            env[k] = str(v)
        return cls(env)

    def apply(self, environ=None) -> List[str]:
        """Make the process environment exactly this profile: engine settings left over from the shell are removed first (a stale
        setting would silently change the engine).  Returns the names that were removed."""
        environ = os.environ if environ is None else environ
        removed = [k for k in list(environ) if k.startswith("YSKN_") and k not in self.env]
        for k in removed:
            environ.pop(k)
        environ.update(self.env)
        return removed


@dataclass
class ModelPack:
    """A model folder: weights, drafter, drafter head, KV profile, tokenizer files (and ``yasakani.json`` listing them)."""
    root: Path
    weights: Path
    draft: Path
    draft_head: Path
    kv_profile: Path
    max_ctx: int = 262144
    model_id: str = "qwen3.8-27b"
    sha256: Dict[str, str] = field(default_factory=dict)

    DEFAULT_FILES = dict(weights="qwen38_27b_e3x.yskn", draft="bdraft2_k8.yskn", draft_head="bdraft2_k8.shead", kv_profile="kvq_u5.bin")

    @classmethod
    def load(cls, root) -> "ModelPack":
        root = Path(root)
        if not root.is_dir():
            raise YasakaniError("model folder not found: %s" % root)
        files = dict(cls.DEFAULT_FILES)
        meta = {}
        mf = root / "yasakani.json"
        if mf.is_file():
            meta = json.loads(mf.read_text(encoding="utf-8"))
            files.update(meta.get("files") or {})
        paths = {k: root / v for k, v in files.items() if k in cls.DEFAULT_FILES}
        missing = [str(p.name) for p in paths.values() if not p.is_file()]
        if missing:
            raise YasakaniError("model folder %s is missing: %s" % (root, ", ".join(missing)))
        for need in ("tokenizer.json", "tokenizer_config.json"):
            if not (root / need).is_file():
                raise YasakaniError("model folder %s is missing %s" % (root, need))
        return cls(root=root, weights=paths["weights"], draft=paths["draft"], draft_head=paths["draft_head"],
                   kv_profile=paths["kv_profile"], max_ctx=int(meta.get("max_ctx", 262144)),
                   model_id=str(meta.get("model_id", "qwen3.8-27b")), sha256=dict(meta.get("sha256") or {}))

    def verify(self, progress=None) -> Dict[str, bool]:
        """Compare the sha256 of every listed file with ``yasakani.json`` (reads ~15 GB: a minute or two)."""
        res = {}
        for name, want in sorted(self.sha256.items()):
            p = self.root / name
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            res[name] = h.hexdigest() == want
            if progress:
                progress(name, res[name])
        return res
