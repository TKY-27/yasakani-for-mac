# 06 — Output head and sampler

This topic specifies the `Engine35` head. The container representation remains
defined by [FORMAT.md](../FORMAT.md); the integer linear operation and deferred
normalization are specified in [02](02_COMPRESSED_LINEAR.md) and
[05](05_RESIDUAL_NORM_MLP.md). The head uses the same LNORD integer operation as
the body, followed by the head-specific scale order below.
(`src/engine35.cpp:803-813`)

## 1. Numeric notation and inputs

`add32`, `mul32`, and `div32` mean separately rounded binary32 operations;
`float32_from_i64` rounds the signed integer to binary32. `add64`, `mul64`, and
`div64` below operate in binary64, not integer arithmetic. `exp64_cuda`,
`log64_cuda`, and `rsqrt32_cuda` identify the upstream CUDA math calls, whose
last-bit implementation is unresolved in SMP-01. Integer hashes, in contrast,
use unsigned 64-bit arithmetic modulo `2^64` and logical right shifts.
(`src/a8_prologue.cuh:10-15`; `src/misc_kernels.cu:63-69`;
`src/q35_sample.cu:179-202`; `src/smp_hash.cuh:8-15`)

For a head slice, inputs are already transformed int8 activations
`activation_q[row,5120]`, their ten binary32 block scales, and the ten block
sums of squares retained by the last residual/normalization step. The head
returns a binary32 logit for every vocabulary item, one selected token per
row, and optionally the eight highest **unmodified logits**. Slice rows are
processed in groups of at most 16; the source offsets the activation and
normalization inputs by `row0` before applying a slice.
(`src/engine35.cpp:799-821`; `src/engine35.cpp:835-837`;
`src/engine35.hpp:313-315`)

## 2. Head score: exact scale order

For each row, let `acc[token]` and `unit_scale` be the int64 output accumulator
and common binary32 scale produced by the XSUM linear operator in topic 02;
let `weight_scale[token]` be its uploaded `tau`. Compute:

```text
sum_squares = float32(0)
for block = 0 .. 9:
    sum_squares = add32(sum_squares, stored_square_sum[row, block])
inverse_rms = rsqrt32_cuda(add32(div32(sum_squares, float32(5120)), epsilon))

for token = 0 .. vocabulary_size-1:
    x = mul32(float32_from_i64(acc[token]), unit_scale)
    x = mul32(x, inverse_rms)
    logit[token] = mul32(x, weight_scale[token])
    acc[token] = 0
```

The normalization factor is multiplied **between** the common XSUM scale and
the per-output `tau`. Do not multiply the three scales together first. There
is no output Hadamard transform, bias, softmax, temperature division, or
binary16/bfloat16 cast in this head postprocessing.
(`src/misc_kernels.cu:57-73`; `src/a8_prologue.cuh:10-15`;
`src/engine35.cpp:810-813`)

Greedy selection compares binary32 logits using `(larger logit, smaller token
ID)`. For finite logits, the result is the first token in a stable descending
logit ordering. Sampling first computes the same logits and then replaces the
selected token; it does not replace the stored logits. The top-eight tables
used by speculation also retain raw-logit order, even when the selected token
is sampled and differs from the raw argmax.
(`src/misc_kernels.cu:59-88`; `src/misc_kernels.cu:100-140`;
`src/engine35.cpp:813-821`; `src/q35_sample.cu:177-205`)

## 3. Which sampler is active

The environment/speculation dictionaries alone do not select a request's
temperature. Immediately after native open, sampling is disabled; stored
parameters are temperature `0.6`, top-p `0.95`, top-k `20`, seed `0`, min-p `0`,
and sampler version `3` under the shipping environment. The public Python
`Sampling()` defaults instead request temperature `1.0`, top-p `0.95`, top-k
`20`, min-p `0`, seed `0`, version `3`; these become active only through
`set_sampling`. Both greedy and sampled requests are therefore covered here.
(`src/engine35.hpp:312`; `src/q35.hpp:113-116`;
`src/engine35.cpp:345-347`; `python/yasakani/engine.py:25-42`;
`python/yasakani/engine.py:165-178`)

The option interface receives integer thousandths for temperature, top-p,
and min-p and constructs each stored binary64 value as
`div64(float64(value), 1000.0)`. Multiplication by the binary64 literal
`0.001` is not interchangeable with this division. Key 10 enables sampling
iff its value is positive, retaining the old
stored temperature when disabled. Key 11 accepts `1..1000`, key 12 accepts
`1..20`, key 29 accepts `0..1000`; key 16 accepts versions 2 or 3. Key 13 sets
`seed = uint64(uint32(value))`, so this setter supplies only 32 bits of the
otherwise 64-bit seed field. Python first uses its integer `round` on the
scaled floating arguments. (`src/engine35.cpp:946-959`;
`python/yasakani/engine.py:170-176`)

`YSKN_SMP2` is unset by `DEFAULT_ENV`, which selects the second top-k kernel.
This affects the extraction implementation; both kernels implement the
selection rules below. The second kernel additionally produces top-eight
and optional top-20 outputs in the sampling call. This `SMP2` name is
independent of **sampler version 2**, which selects a different random draw.
(`src/q35_sample.cu:56-101`; `src/q35_sample.cu:131-213`;
`python/yasakani/runtime.py:27-44`; `python/yasakani/runtime.py:93-101`)

## 4. Candidate truncation

For finite logits, take the `top_k` highest values, sorting by decreasing
binary32 logit and then increasing token ID. The GPU uses intermediate
candidate lists, but their pairwise comparison is exactly this ordering and
does no arithmetic on the logits. With top-eight reporting requested, it
extracts `max(top_k,8)` entries and uses only the first `top_k` for sampling.
(`src/q35_sample.cu:28-48`; `src/q35_sample.cu:108-178`)

Convert each selected logit individually to binary64 before subtraction.
Apply truncation in this order:

```text
k = top_k
for j = 0 .. k-1:
    z[j] = div64(float64(logit[id[j]]) - float64(logit[id[0]]), temperature)
    weight[j] = exp64_cuda(z[j])

total = float64(0)
for j = 0 .. k-1:
    total = add64(total, weight[j])

n = k
prefix = float64(0)
for j = 0 .. k-1:
    prefix = add64(prefix, weight[j])
    if prefix >= mul64(top_p, total):
        n = j + 1
        break

if min_p > 0:
    m = 1
    while m < n and weight[m] >= min_p:
        m += 1
    n = m
```

Top-p is relative to the mass of the **top-k subset**, not the whole
vocabulary. Min-p is applied after top-p and is relative to the largest
unnormalized weight (`weight[0] = exp(0)`); at least one token remains. These
prefix sums are serial in ascending candidate index, with binary64 rounding
after every addition. (`src/q35_sample.cu:179-193`)

When portable math is explicitly enabled, `pm_exp_d` replaces the exponential
above and `pm_log_d` replaces the logarithms below. This is not the shipping
setting; `pm=0` initially and is changed by `apply_kz`.
(`src/q35.hpp:114-116`; `src/q35_sample.cu:185-186`;
`src/smp_hash.cuh:15`; `src/engine35.cpp:363-364`;
`src/engine35.cpp:465`)

## 5. Absolute prediction position

Let `base_position` be `*d_pos`, the absolute position of verification input
row 0. For slice-local row `r` and the slice's `row0`:

```text
depth = row0 + r                          # a straight chain
if a row table is active:
    depth = row_depth[row0 + r]
prediction_position = int64(base_position) + depth + 1
```

Thus input token at absolute position `p` predicts the token at `p+1`.
Side candidates at the same depth use the same prediction position. The
position key does not depend on verification width, candidate-list rank, or
how many random draws have previously occurred.
(`src/q35_sample.cu:180-186`; `src/smp_hash.cuh:12-15`)

## 6. Version 3: token-keyed Gumbel maximum

The exact unsigned integer mixer is:

```text
mix64(z):
    z = (z + 0x9E3779B97F4A7C15) mod 2^64
    z = ((z xor (z >> 30)) * 0xBF58476D1CE4E5B9) mod 2^64
    z = ((z xor (z >> 27)) * 0x94D049BB133111EB) mod 2^64
    return z xor (z >> 31)

gumbel(seed, prediction_position, token):
    a = mix64(seed)
    b = mix64(a xor uint64(prediction_position))
    h = mix64(b xor ((uint64(uint32(token)) * 0xD6E8FEB86659FD93) mod 2^64))
    u = mul64(add64(float64(h >> 11), 0.5), 1 / 9007199254740992)
    return -log64_cuda(-log64_cuda(u))

best = -infinity
pick = n - 1
for j = 0 .. n-1:
    candidate = add64(z[j], gumbel(seed, prediction_position, id[j]))
    if candidate > best:
        best = candidate
        pick = j
return id[pick]
```

`h >> 11` fits exactly in binary64. The addition of `0.5` is nevertheless a
binary64 operation and must not be replaced by arbitrary-precision rational
arithmetic. Strict `>` retains the earliest candidate in logit/ID order on
an equal perturbed score. There is no evolving PRNG state.
(`src/smp_hash.cuh:8-15`; `src/q35_sample.cu:185-197`)

The block drafter uses this same `gumbel` with the same absolute positions and
token IDs, but uses its own logits, temperature and truncation described in
[07](07_DRAFT_MODEL.md). Shared noise does not mean that its proposal
distribution is the main model's distribution.
(`src/q35_bdraft.cu:401-430`; `src/engine35.cpp:1845-1850`)

## 7. Version 2: position-keyed inverse cumulative draw

Version 2 uses the same candidate set and weight order, but only seed and
position key its random value:

```text
kept = float64(0)
for j = 0 .. n-1:
    kept = add64(kept, weight[j])
hash = mix64(mix64(seed) xor uint64(prediction_position))
uniform = mul64(float64(hash >> 11), 1 / 9007199254740992)
threshold = mul64(uniform, kept)
prefix = float64(0)
pick = n - 1
for j = 0 .. n-1:
    prefix = add64(prefix, weight[j])
    if threshold < prefix:
        pick = j
        break
return id[pick]
```

The comparison is strict `<`; there is no `+0.5` in this version's uniform
construction. (`src/q35_sample.cu:52-55`; `src/q35_sample.cu:198-205`)

## 8. State effects and unresolved numeric contract

Verification writes one selected token per input row and does not advance
position in the head. Single-token direct decode asks the head to increment
position and replace the current token; for sampled decode only the sampler
performs that increment. Commit selects the prediction of the last committed
input row. (`src/engine35.cpp:812-817`; `src/engine35.cpp:835-837`;
`src/q35_sample.cu:204-206`; `src/engine35.cpp:1319-1329`)

**UNCERTAIN [SMP-01]: CUDA transcendental bits.** The source invokes
`rsqrtf`, binary64 `exp`, and binary64 `log`, and contains an optional portable
math implementation which shipping does not enable. Their exact CUDA library
and device results cannot be reconstructed from these call sites alone.
This document fixes argument construction, precision, operation order and
hash bits; an independent implementation must leave these primitive results
unresolved rather than assume the host's `sqrt`/`exp`/`log` are bit-identical.
Checked `src/misc_kernels.cu:63-69`, `src/q35_sample.cu:179-202`,
`src/smp_hash.cuh:12-15`, and `src/engine35.cpp:363-364,465`.

**UNCERTAIN [SMP-02]: Nonfinite-logit behavior.** Top-k initializes with
`-FLT_MAX`, greedy initializes with that value and ID `INT_MAX`, while draft
top-k uses `-INFINITY`. NaN comparisons are false and no finite-logit
validation occurs here. A universal total-order sort would invent behavior
for NaNs or an all-`-INFINITY` row; the finite-logit contract above must not be
extended to those cases without specifying the original candidate-reduction
layout. Checked `src/misc_kernels.cu:59-88,121-140`,
`src/q35_sample.cu:28-48,131-178`, and `src/q35_bdraft.cu:928-955`.

**UNCERTAIN [SMP-03]: Upper endpoint in the Gumbel uniform.** If `h >> 11`
equals `2^53-1`, adding `0.5` in round-to-nearest-even binary64 rounds up to
`2^53`, so `u` is exactly `1.0` and `-log(-log(u))` is positive infinity.
The source does not clamp that endpoint. This follows from the written
arithmetic; whether a released-model request reaches such a hash has not
been determined. Preserve the formula rather than silently changing it to
an open-interval generator. Checked `src/smp_hash.cuh:12-15` and the callers
at `src/q35_sample.cu:195-197` and `src/q35_bdraft.cu:401-402`.
