# 10. Auxiliary files in a model folder

The Python package selects four model files, requires two tokenizer files,
and optionally reads `yasakani.json`. It passes the draft-head and KV-profile
paths through the process environment before opening the native engine.
Sources: `python/yasakani/runtime.py:83-100`,
`python/yasakani/runtime.py:104-138`, `python/yasakani/engine.py:90-112`.

The `.yskn` format is specified by [FORMAT.md](../FORMAT.md). This chapter
covers the three auxiliary formats requested here. Offsets are byte offsets
from the beginning of the relevant file, unless stated otherwise. Sources:
`src/engine35.cpp:31-53`, `src/engine35.cpp:1700-1734`,
`python/yasakani/runtime.py:119-138`.

## `kvq_u5.bin`: the KVQP profile

The default filename is `kvq_u5.bin`; the native code accepts the file named
by `YSKN_TB_PROFILE`. Missing or empty settings and a failed open are errors
in the shipping `YSKN_KVX_MODE=3` build. The filename is not a format version
or a hard-coded guarantee of five-bit contents. Sources:
`python/yasakani/runtime.py:85-86`, `python/yasakani/runtime.py:116`,
`CMakeLists.txt:46`, `src/engine35.cpp:31-45`, `src/kvq_atxpk.hpp:73-83`.

### Header fields actually read

Unsigned 16/32-bit fields use little-endian byte assembly. FP32 fields are
formed by interpreting the corresponding assembled 32 bits as a float.
Sources: `src/engine35.cpp:38-43`, `src/kvq_gpu.cuh:28-32`.

| Offset | Bytes | Meaning / required value | Evidence |
|---|---|---|---|
| 0 | 4 | ASCII `KVQP` | `src/engine35.cpp:40` |
| 4 | 2 | Version, must be 1 | `src/engine35.cpp:40` |
| 6 | 2 | Header-size field, must be 96 | `src/engine35.cpp:40` |
| 8 | 4 | Total file size, must equal bytes read | `src/engine35.cpp:40` |
| 12 | 4 | Not read by the inspected consumers; meaning unknown | `src/engine35.cpp:38-43`, `src/kvq_fast.hpp:46-67` |
| 16 | 2 | Row dimension, must be 256 | `src/engine35.cpp:41` |
| 18 | 2 | Full-attention layer count, must be 16 | `src/engine35.cpp:41` |
| 20 | 2 | KV head count, must be 4 | `src/engine35.cpp:41` |
| 22 | 2 | Number of code tables | `src/engine35.cpp:42-43` |
| 24 | 2 | Number of scale multipliers, 1 through 16 | `src/kvq_fast.hpp:50`, `src/kvq_fast.hpp:11`, `src/engine35.cpp:50` |
| 26 | 2 | Not read; meaning unknown | `src/kvq_fast.hpp:50`, `src/kvq_fast.cuh:54-62` |
| 28 | 4 | Number of groups, must be 128 = 16 × 4 × 2 | `src/engine35.cpp:41`, `src/kvq_atxpk.hpp:29-30` |
| 32 | 4 | Absolute offset of the FP32 scale-multiplier array | `src/kvq_fast.cuh:56`, `src/kvq_fast.cuh:165` |
| 36 | 4 | Absolute offset of the code-table offset array | `src/engine35.cpp:42-43`, `src/kvq_fast.cuh:159` |
| 40 | 4 | Absolute offset of the group-record offset array | `src/kvq_fast.cuh:57-61` |
| 44–95 | 52 | No meaning established by the inspected consumers | `src/engine35.cpp:38-43`, `src/kvq_fast.cuh:54-62` |

The loader requires at least 100 bytes even though the header-size field is
96. Neither code tables nor groups are required by these consumers to follow
the header immediately: they are found through absolute offsets. Sources:
`src/engine35.cpp:40-43`, `src/kvq_fast.hpp:49-62`.

**UNCERTAIN [AUX-01]:** The meanings of unconsumed header bytes, group bytes
6–7, code-table byte 1 and bytes 8–11, the unused fourth SHEAD header word,
and any layout beyond the first two consumed residual basis vectors are not
recoverable from the readers. Do not call them zero padding or assign
a versioning meaning. Checked the field reads in `src/engine35.cpp:38-53`,
`src/kvq_fast.hpp:46-67`, `src/kvq_fast.cuh:54-62`,
`src/kvq_atxpk.hpp:25-57`, `src/kvq_gpu.cuh:58-99`, and
`src/engine35.cpp:1705-1707`.

### Referenced arrays and records

The scale-multiplier array has `multiplier_count` FP32 values, in trial order.
The table directory has `table_count` little-endian unsigned 32-bit offsets;
the group directory has 128 such offsets. Group index is
`((attention_ordinal * 4 + kv_head) * 2 + kind)`, where `kind=0` denotes keys
and `kind=1` values. `attention_ordinal` counts only full-attention layers, not
all transformer layers. Sources: `src/kvq_fast.cuh:54-78`,
`src/engine35.cpp:688-691`.

| Offset relative to a table | Bytes | Meaning | Evidence |
|---|---|---|---|
| 0 | 1 | State bits, must be 8 | `src/engine35.cpp:43` |
| 1 | 1 | Unused / unknown | `src/engine35.cpp:43`, `src/kvq_gpu.cuh:89-94` |
| 2 | 1 | Chain length, must be 64 | `src/engine35.cpp:43` |
| 3 | 1 | Fixed-point fractional-bit field, must be 8 | `src/engine35.cpp:43`, `src/kvq_gpu.cuh:53` |
| 4 | 4 | FP32 reconstruction gain | `src/kvq_fast.cuh:159-165` |
| 8 | 4 | Unused / unknown | `src/kvq_gpu.cuh:89-94` |
| 12 | 512 | 256 signed little-endian int16 code values | `src/kvq_fast.hpp:62-63`, `src/kvq_fast.cuh:162-163` |

A table value is decoded with two separate FP32 multiplications:
`float32(float32(code_i16 * 0.00390625f) * gain)`. The optimized encoder
eligibility check additionally requires every code value in tables referenced
by four/five-bit trellis groups to lie in `[-1100,1100]`; this chooses the
fast variant, rather than being a universal rejection rule for all profiles.
Sources: `src/kvq_gpu.cuh:53`, `src/kvq_fast.hpp:51-66`,
`src/kvq_fast.cuh:240-248`.

| Offset relative to a group | Bytes | Meaning | Evidence |
|---|---|---|---|
| 0 | 1 | Mode: 0 trellis; 1 flat signed quantization; other modes leave residual unchanged before reconstruction | `src/kvq_fast.cuh:189-217` |
| 1 | 1 | Bits per trellis field | `src/kvq_fast.cuh:60`, `src/kvq_fast.cuh:172-176`, `src/kvq_fast.cuh:212` |
| 2 | 1 | Code-table index | `src/kvq_fast.cuh:60`, `src/kvq_fast.cuh:159` |
| 3 | 1 | Guard prefix length: do not profile-quantize earlier positions | `src/kvq_fast.cuh:60`, `src/kvq_fast.cuh:189` |
| 4 | 1 | Residual basis count; consumers use at most two basis vectors | `src/kvq_fast.cuh:61`, `src/kvq_fast.cuh:90-91` |
| 5 | 1 | Mean-present flag | `src/kvq_fast.cuh:61`, `src/kvq_fast.cuh:85-89` |
| 6 | 2 | Unused / unknown | `src/kvq_fast.cuh:54-62`, `src/kvq_fast.cuh:85-91` |
| 8 | 512 if mean-present | Mean `[256]`, FP16 | `src/kvq_fast.cuh:85-89` |
| 8 + optional mean bytes | `512*min(basis_count,2)` bytes consumed | First one or two basis vectors `[256]`, FP16; layout of any remaining data for counts above two is unknown | `src/kvq_fast.cuh:86-91` |

Packed capacity storage is used for a whole layer/kind only if all four heads
have mode 0, the selected common bit width, zero guard, and zero residual-basis
count. The common width is 5 if mode-0 five-bit groups outnumber four-bit
groups, otherwise 4. Mean vectors are allowed by the packability predicate.
Sources: `src/kvq_atxpk.hpp:60-83`.

The runtime derives a decode record per group; this is **not another section
of the file**. It contains BF16 reconstruction values at bytes 0–511, optional
FP32 means at 512–1535, an int8 approximation of the table at 1536–1791, and
its FP32 scale at 1792–1795. Each record occupies 2048 bytes, followed by a
separate array of 128 mean-present flags. Sources:
`src/kvq_atxpk.hpp:17-18`, `src/kvq_atxpk.hpp:25-57`.

For the per-token KV5 cache payload, state transitions, bit ordering and
FP16-rounded row scale, see [04](04_ATTENTION.md). The file contains the
**profile**; the encoder produces payloads in memory. Source distinction:
`src/kvq_fast.cuh:197-234`, `src/engine35.cpp:242-251`.

**UNCERTAIN [AUX-02]:** The actual `kvq_u5.bin` payload is not in this
checkout. Therefore the numerical table entries, multiplier list, means,
which layer/kind remains int8, and whether its groups select the optimized
encoder cannot be established from its filename. `TECHNOLOGY.md` reports
16 packed key streams and 13 packed value streams, but the loader computes
that from the file; it is not hard-coded. Checked:
`python/yasakani/runtime.py:116`, `src/engine35.cpp:38-53`,
`src/engine35.cpp:242-251`, `src/kvq_atxpk.hpp:60-83`,
`docs/TECHNOLOGY.md:231-237`.

**UNCERTAIN [AUX-03]:** The KVQP reader performs some length checks but does
not validate every offset before dereferencing it. For example, the table
directory is indexed at `off_t+4*t` before validating that directory access,
and the group directory is similarly indexed before checking `go+8`.
Malformed inputs therefore lack a fully defined safe-reader contract. The
specification describes valid data, not a promise that the upstream loader
rejects all malformed files. Checked: `src/engine35.cpp:38-43`,
`src/kvq_fast.hpp:50-63`, `src/kvq_atxpk.hpp:29-38`.

## `.shead`: reduced draft output head

The shipping profile supplies an explicit path and candidate count 512.
Without an explicit path, native loading can try a sibling `.shead` and skip
it for memory or load failures; an explicit path does not silently use that
fallback. Setting the path string to `"0"` disables auto-discovery. Sources:
`python/yasakani/runtime.py:28`, `python/yasakani/runtime.py:85`,
`src/engine35.cpp:1664-1688`.

Let `V` be the main-model vocabulary size, `H=5120`, and `R=512`. The code
reads native 32-bit header words and uploads the payload bytes without byte
swapping. For the targeted Windows build these are little-endian int32 and
IEEE FP32 representations. The layout follows the read sequence exactly;
there is no inserted alignment padding. Sources:
`src/engine35.cpp:1705-1708`, `src/engine35.cpp:1712-1734`,
`src/q35.hpp:215`.

| Offset | Bytes | Stored object | Evidence |
|---|---|---|---|
| 0 | 8 | ASCII `SHEAD001` | `src/engine35.cpp:1705-1706` |
| 8 | 4 | int32 vocabulary size, must equal `V` | `src/engine35.cpp:1707` |
| 12 | 4 | int32 hidden width, must equal 5120 | `src/engine35.cpp:1707` |
| 16 | 4 | int32 rank, must equal 512 | `src/engine35.cpp:1707-1708`, `src/q35.hpp:215` |
| 20 | 4 | Fourth int32 word; read, never checked or consumed | `src/engine35.cpp:1705-1708` |
| 24 | `4*R*H` | `sk_bt`: FP32 `[R,H]`, low-rank projection rows | `src/engine35.cpp:1729`, `src/q35_bdraft.cu:780-802` |
| `24+4*R*H` | `V*R` | `sk_a`: signed int8 `[V,R]` | `src/engine35.cpp:1730`, `src/q35_bdraft.cu:805-837` |
| Previous end | `4*V` | `sk_as`: FP32 scale `[V]` | `src/engine35.cpp:1731`, `src/q35_bdraft.cu:805-837` |
| Previous end | `V*H` | `sk_w`: signed int8 `[V,H]`, candidate rescore weights | `src/engine35.cpp:1732`, `src/q35_bdraft.cu:898-932` |
| Previous end | `4*V` | `sk_ws`: FP32 rescore scale `[V]` | `src/engine35.cpp:1733`, `src/q35_bdraft.cu:898-932` |
| `24+4*R*H+V*(R+H+8)` | 0 | Required EOF; extra bytes are an error | `src/engine35.cpp:1734` |

Short reads are errors. The reader verifies the dimensions and EOF but not
the numerical validity of the matrix/scales. See [07](07_DRAFT_MODEL.md) for
the computation, including the distinction between candidate rescoring and
the full main head. Sources: `src/engine35.cpp:1705-1717`,
`src/engine35.cpp:1729-1734`, `src/q35_bdraft.cu:898-932`.

## `yasakani.json`: package manifest

This is UTF-8 JSON read by Python, not by the C++ container reader. It is
optional. Unknown `files` entries do not become required paths; recognized
entries replace the corresponding default. Sources:
`python/yasakani/runtime.py:119-138`.

| JSON member | Default | Behavior / evidence |
|---|---|---|
| `files.weights` | `qwen38_27b_e3x.yskn` | Main weights; `python/yasakani/runtime.py:116-129` |
| `files.draft` | `bdraft2_k8.yskn` | Block-draft weights; `python/yasakani/runtime.py:116-129` |
| `files.draft_head` | `bdraft2_k8.shead` | Reduced draft head; `python/yasakani/runtime.py:116-129` |
| `files.kv_profile` | `kvq_u5.bin` | KVQP profile; `python/yasakani/runtime.py:116-129` |
| `max_ctx` | 262144 | Converted with Python `int`; `python/yasakani/runtime.py:136-138` |
| `model_id` | `qwen3.8-27b` | Converted with Python `str`; `python/yasakani/runtime.py:138` |
| `sha256` | Empty dictionary | Optional filename-to-expected-hex map; `python/yasakani/runtime.py:138-149` |

Every selected file must exist, as must `tokenizer.json` and
`tokenizer_config.json` under the package root. Loading the package does not
automatically verify checksums. `ModelPack.verify` separately reads each
listed file in 16-MiB chunks and compares SHA-256 hex strings. These checksums
do not define any inference arithmetic. Sources:
`python/yasakani/runtime.py:129-149`, `python/yasakani/engine.py:90-117`.

The path operation is `root / value`; the loader does not enforce that the
result remains inside `root`. This is observed parsing behavior, not a new
path-validation requirement or a proposed change. Source:
`python/yasakani/runtime.py:123-138`.
