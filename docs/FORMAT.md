# The YASAKANI model file format (`.yskn`), version 1

日本語：[FORMAT.ja.md](FORMAT.ja.md)

This document says what a `.yskn` file means, byte by byte, so that anyone can read one without the engine, check it against the
original weights, or write one.

* A reader that follows this text and nothing else: [`tools/yskn_decode.py`](../tools/yskn_decode.py) (Python and numpy, no GPU).
* A small test file and the values a reader must get from it: [`tools/testdata/`](../tools/testdata/).
* The reader of the engine: [`src/format.hpp`](../src/format.hpp) and [`src/loader.cpp`](../src/loader.cpp).

What this document does not give you: the program that made the released files. The conversion tool is not public; its method
is described in [TECHNOLOGY.md](TECHNOLOGY.md), section 3.2. With this document you can check what the released files contain,
but you cannot rebuild them from the original weights bit for bit. See [section 9](#9-writing-a-file).

## 1. Overview

A file holds a JSON header and, after it, blocks of data that the header points to.

A weight matrix is not stored as numbers. It is stored as

* a **code stream**: 3 or 4 bits per weight,
* a **value table**: 4,096 pairs of numbers, one table per matrix,
* a **scale**: one number per row,

and all of this in **rotated coordinates**: the rows and columns of the matrix are mixed by a fixed Hadamard transform with
pseudo-random signs before they are coded. The signs are not stored; the header holds the seed they are made from.

Small parameters (normalisation weights and the like) are stored as plain 16-bit floating-point numbers and are called
**vectors**.

All integers are little-endian.

## 2. File layout

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | the bytes `YSKN` |
| 4 | 4 | version, unsigned 32-bit: `1` |
| 8 | 8 | length of the JSON header in bytes, unsigned 64-bit |
| 16 | 8 | offset of the first data block, unsigned 64-bit (for information) |
| 24 | 8 | length of the whole file in bytes, unsigned 64-bit |
| 32 | (length) | the header: one JSON object, UTF-8 |

The bytes between the end of the header and the first data block are zero. Every data block is found through an absolute offset
(from the start of the file) in the header. The format does not ask for any alignment; in the released files every block starts at
a multiple of 256 bytes.

## 3. The header

The JSON object has three members: `model`, `tensors`, `vectors`.

### 3.1 `model`

The dimensions of the model (`arch`, `layers`, `hidden`, `ffn`, `heads`, `kv_heads`, `head_dim`, `vocab`, `eps`, `rope_theta`,
`tie_embeddings`), further members that depend on the architecture, and `source`: where the original weights came from (for the
released models the Hugging Face repository and revision). How the model computes with its tensors is defined by the original model
and by the engine source for that `arch`; this document covers only how the tensors are stored.

### 3.2 `tensors`

A list. Each entry describes one matrix with `rows` rows and `cols` columns (`y = W x`: `cols` inputs, `rows` outputs).

| Member | Meaning |
|---|---|
| `name` | for the released models: the parameter name of the original checkpoint, shortened (section 6) |
| `role` | `body`, `head` (the output matrix), `embedding`, or `draft` (a matrix of the helper model) |
| `rows`, `cols` | the shape. `cols` is a multiple of 512 |
| `physical_rows` | rows that are stored; default `rows`. Rows beyond `rows` are padding |
| `l` | bits of the state; default 12. This document and the engine cover 12 only |
| `d` | weights per step; default 2, the only value |
| `t` | weights per sequence; default 256, the only value |
| `bits` | bits per step: 6 (3 bits per weight) or 8 (4 bits per weight) in the released files; the reader takes 4 to 8 |
| `start` | `zero` (default), or `carry` (not used by the released files and not defined here) |
| `layout` | how the code stream is ordered: `warp` (default) or `row` (section 4.3) |
| `stream_off`, `stream_bytes` | the code stream |
| `lut_off` | the value table: 8,192 half-precision floats (16,384 bytes) |
| `scale_off` | the scales: `physical_rows` half-precision floats |
| `rot_in`, `rot_out` | the rotations of the input side (columns) and the output side (rows), section 3.3 |
| `rar` | `null`, or an object for per-tile bit widths (not used by the released files and not defined here; see `src/loader.cpp`) |
| `bias_off`, `proxy`, `encoder` | not used by a reader. `bias_off` is 0; `proxy` is a number the conversion recorded; `encoder` is a text |

"Half-precision float" always means IEEE 754 binary16.

### 3.3 Rotations

`rot_in` and `rot_out` are objects:

| Member | Meaning |
|---|---|
| `kind` | `H512` (sections 4.5 and 4.6). `none` is allowed for `rot_out`. `AYA` is a second kind that the released files do not use and this document does not define |
| `m` | number of blocks of 512 (`cols / 512` for `rot_in`, `rows / 512` for `rot_out`) |
| `sign_seed`, `sign_offset` | where the signs come from (section 4.5) |
| `rule` | `splitmix64` (default) or `mulberry32_7919_13` |
| `qm_off` | used by `AYA` only; 0 otherwise |

If `rot_out` is `null` or has the kind `none`, the tensor has no output rotation. The reader also has defaults for files that leave
the members out (the released files always write them): `rot_in` absent means `H512`, seed 7, offset 0, rule `mulberry32_7919_13`;
`rot_out` absent means no output rotation for the roles `head` and `embedding`, and otherwise `H512`, seed 7, offset `cols`, rule
`mulberry32_7919_13`.

### 3.4 `vectors`

A list of `{ "name", "off", "n", "dtype" }`: `n` 16-bit numbers at the offset `off`. `dtype` is `f16` (default; half-precision
float) or `bf16` (bfloat16: the upper 16 bits of an IEEE 754 single-precision float; to read it, put the 16 bits into the upper
half of a 32-bit float and zeros into the lower half).

## 4. Decoding a matrix

### 4.1 Sequences and steps

Each row is cut into `S = cols / 256` **sequences** of 256 weights. A sequence is coded in 128 **steps**; a step yields two
weights and takes `bits` bits of code. A sequence therefore has `128 * bits` bits of code: `W = 4 * bits` 32-bit words.

Within a sequence, bit number `p` (counted from 0) is bit `p mod 32` of word `p div 32`, where bit 0 of a word is its least
significant bit.

### 4.2 From bits to values

For step `j` (0 to 127) let `e = bits * (j + 1)`, the position just after the bits of this step. The **state** of the step is the
12 bits that end there:

    state = sum over i = 0..11 of  b[e - 12 + i] * 2^i          (b[p] = bit p of the sequence; b[p] = 0 for p < 0)

So the newest bits sit at the top of the state, and each step shifts the older ones down. Only the first step reaches before the
start of the sequence; there the bits count as zero.

The value table has 4,096 pairs. Step `j` yields

    v[2 j]     = table[2 * state]
    v[2 j + 1] = table[2 * state + 1]

and with the scale of the row, the stored matrix in rotated coordinates is

    M[r][256 * s + k] = scale[r] * v[k]           (row r, sequence s, k = 0..255)

Because each state shares 12 - `bits` bits with the state before it, the value of a step depends on the bits of the step before as
well: this is a trellis code. Any bit string is a valid code.

### 4.3 Where the words of a sequence are

`stream_off` points to `physical_rows * S * W` 32-bit words. Word `w` (0 to `W` - 1) of sequence `s` of row `r` is the word
number

* layout `row`:  `(r * S + s) * W + w`
* layout `warp`: with `q = s * physical_rows + r`, `g = q div 32`, `lane = q mod 32`, `c = w div 4`:
  `((g * bits + c) * 32 + lane) * 4 + (w mod 4)`

In words: `warp` puts the sequences in the order "all rows of sequence 0, all rows of sequence 1, ..." and interleaves every 32
of them in pieces of four words (16 bytes), so that 32 GPU threads that decode 32 rows read neighbouring memory. `warp` needs
`physical_rows * S` to be a multiple of 32 (the engine asks for a multiple of 64 rows).

### 4.4 The Hadamard transform

`H` is the 512 x 512 Hadamard matrix in natural (Sylvester) order: `H[a][b] = +1` if the number of one-bits of `a AND b` is even,
`-1` if it is odd. It is used with the factor `1 / sqrt(512)`; `H / sqrt(512)` is symmetric and its own inverse. A vector whose
length is a multiple of 512 is transformed block by block.

### 4.5 Signs

`signs(rule, seed, offset, n)` is the list of `n` values `+1` / `-1` at the positions `offset .. offset + n - 1` of an endless
stream. With 64-bit and 32-bit unsigned arithmetic that wraps around:

    splitmix64, position k (from 0):
        z = seed + (k + 1) * 0x9E3779B97F4A7C15
        z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z XOR (z >> 27)) * 0x94D049BB133111EB
        z = z XOR (z >> 31)
        sign = +1 if the top bit of z is set, else -1

    mulberry32_7919_13, position k (from 0):
        s0 = (seed mod 2^32) * 7919 + 13            (if this is 0: 0x9E3779B9)
        t = s0 + (k + 1) * 0x6D2B79F5
        t = (t XOR (t >> 15)) * (t OR 1)
        t = t XOR (t + (t XOR (t >> 7)) * (t OR 61))
        t = t XOR (t >> 14)
        sign = -1 if t < 2^31, else +1

The input side uses `s_in = signs(rot_in.rule, rot_in.sign_seed, rot_in.sign_offset, cols)`, the output side
`s_out = signs(rot_out.rule, rot_out.sign_seed, rot_out.sign_offset, physical_rows)`.

### 4.6 Back to the coordinates of the original model

With `M` from section 4.2:

1. Input side, for every row: transform the row with `H / sqrt(512)` (blocks of 512 columns), then multiply column `i` by `s_in[i]`.
2. Output side, only if the tensor has an output rotation, for every column: transform the column with `H / sqrt(512)` (blocks of
   512 rows), then multiply row `n` by `s_out[n]`.

The result `W` is the matrix the file stands for, in the coordinates of the original model. As a formula, with `D_in` and `D_out`
the diagonal matrices of the signs and `Hn` the block-wise `H / sqrt(512)`:

    W = D_out * Hn * M * Hn * D_in          (without an output rotation: W = M * Hn * D_in)

The engine never builds `W`. It computes `y = W x` as `x' = Hn (D_in x)`, `z = M x'`, `y = D_out (Hn z)`, with `M` taken straight
from the codes.

## 5. What the engine does with it

This is not part of the format, but it explains small differences between a decoded matrix and what the engine computes.

* For matrices (the roles `body`, `head`, `draft`) the engine rounds the value table to 8 bits when it loads the file:
  `gamma = (largest absolute value of the table) / 127`, `q = round(value / gamma)` limited to -127 .. 127, and the scale of row `r`
  becomes `scale[r] * gamma`. `yskn_decode.py --engine-table` decodes with this table. The embedding is read with the 16-bit table.
* The input of a matrix, after its rotation, is rounded to 8-bit integers with one scale per vector, so that the product runs on
  integers ([TECHNOLOGY.md](TECHNOLOGY.md), section 4).
* At load time the engine reorders the code stream once more for its kernels (`src/repack.cu`). This does not change the meaning.

## 6. The released files

| File | Tensors | Vectors |
|---|---|---|
| quality model, `qwen38_27b_quality.yskn` | 400 `body` (288 with 8 bits per step, 112 with 6), `head` (8), `embedding` (6) | 449 |
| fastest model, `qwen38_27b_speed.yskn` | 400 `body` (all with 6 bits per step), `head` (8), `embedding` (6) | 449 |
| helper model, `bdraft2_k8.yskn` | 46 `draft` (8 bits per step, no output rotation) | 35 |

All of them use `l` = 12, `start` = `zero`, the kind `H512` with the rule `mulberry32_7919_13` and seed 7, the layout `warp` for
matrices and `row` for the embedding, and `bf16` vectors. None uses `rar`, `AYA` or `carry`.

The names follow the original checkpoint (`Qwen/Qwen3.8-27B`): the tensor or vector `layers.N.X` is the parameter
`model.language_model.layers.N.X.weight` (or `model.language_model.layers.N.X` where the original has no `.weight`), `embed_tokens`
is `model.language_model.embed_tokens.weight`, `norm` is `model.language_model.norm.weight`, `lm_head` is `lm_head.weight`. A vector
holds the numbers of the original parameter in their original order (a parameter with several axes is flattened). The parameters of the
original checkpoint outside `model.language_model.*` and `lm_head.weight` (the vision part `model.visual.*` among them) are not in the
files.

## 7. Checking a file yourself

    python tools/yskn_decode.py info    qwen38_27b_quality.yskn
    python tools/yskn_decode.py compare qwen38_27b_quality.yskn --original <folder with the *.safetensors of Qwen/Qwen3.8-27B> --json out.json
    python tools/yskn_decode.py decode  qwen38_27b_quality.yskn layers.3.self_attn.k_proj --out k_proj.npy

`compare` decodes every tensor, compares it with the original and prints the relative error
`|W_decoded - W_original| / |W_original|` (Frobenius norm); for the vectors it checks that they are bit for bit the original
numbers. It needs the original checkpoint (about 54 GB) and about half an hour on a desktop CPU.

What we get for the released files (the full lists are in [`tools/records/`](../tools/records/)):

| Model | Tensors | Bits per weight | Relative error: median (smallest to largest) |
|---|---|---|---|
| quality model | 288 body matrices | 4 | 0.092 (0.085 to 0.106) |
| quality model | 112 body matrices | 3 | 0.177 (0.166 to 0.199) |
| fastest model | 400 body matrices | 3 | 0.177 (0.163 to 0.202) |
| both models | the output matrix | 4 | 0.087 |
| both models | the embedding | 3 | 0.146 |

All 449 vectors of each model are bit for bit the numbers of the original checkpoint.

A relative error of this size is what 3 or 4 bits per weight give; it says how far the stored weights are from the original ones,
not how good the model's output is. The conversion chooses the codes so that the output of the model stays close to the original
([TECHNOLOGY.md](TECHNOLOGY.md), section 3.2), which is not the same as making this number as small as possible. The quality of the
output is measured separately ([MEASUREMENTS.md](MEASUREMENTS.md)).

## 8. Test data

[`tools/testdata/tiny.yskn`](../tools/testdata/tiny.yskn) (468,304 bytes, SHA-256
`3f4a48866f6a1751beabc3b9a8d06088672a3db0aa0a4c3f96e1877cb448d393`) is a container with four small tensors and two vectors: 6 and 8
bits per step, both layouts, both sign rules, with and without an output rotation, both vector types. It holds no model: streams,
tables and scales are pseudo-random. [`tools/yskn_fixture.py`](../tools/yskn_fixture.py) writes the same file again, byte for byte.

[`tools/testdata/tiny_expected.json`](../tools/testdata/tiny_expected.json) lists what a reader must get: the header numbers, the
SHA-256 of several sign streams (as signed bytes), and for every tensor the SHA-256 of the states (unsigned 16-bit, little-endian,
rows x sequences x 128) and of the values in rotated coordinates (32-bit floats, little-endian, rows x cols; the product of the
table value and the scale, both converted to 32-bit floats first), some weights at given positions, and the Frobenius norm of the
weights. `python tools/yskn_decode.py selftest` checks the reader of this repository against it, and also reads a few rows a
second time with a slow, bit-by-bit routine written from section 4.

## 9. Writing a file

A writer has to produce, for every matrix, a value table, a scale per row and a code stream such that the decoded matrix is
close to the one it wants to store, and then lay the data out as in sections 2 to 4. Since every bit string is a valid code,
`tools/yskn_fixture.py` is a complete (if useless) writer: it fills the streams with pseudo-random bits.

Choosing good codes is the hard part. The released files were made by a conversion tool that is not public. Its method is described
in [TECHNOLOGY.md](TECHNOLOGY.md), section 3.2 (the statistics it collects, how it searches for the codes, how the rounding error is
passed on, the calibration text); that description is not detailed enough to reproduce the released files bit for bit.
