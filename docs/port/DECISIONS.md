# Round 2 reference decisions

Baseline: merged specification PR #1, commit
`157e33eac8d5ddd0e031b1826fe28e67f57f3787`. Implementation sources are
`docs/FORMAT.md`, `docs/spec/`, and the explicitly permitted
`src/pm_math.cuh`. No other upstream implementation file was opened.

## D1 — Internal bit consistency

The Rust scalar reference defines this project's numeric contract. We do not
claim equality with the CUDA binary: the approved specification leaves CUDA
elementary functions and MMA accumulation unresolved, and upstream dispatch
can depend on row count.

Every input row follows one canonical scalar path in ascending token order,
regardless of prefill chunk size or provisional pass width. Floating operations
remain separate binary32 or binary64 operations; there are no FMA calls, SIMD
intrinsics, threads, unsafe blocks, or calls to platform transcendental math.
Compiler fast-math/reassociation flags are outside this reference contract.

Ordinary host arithmetic and `nearbyint` in the table loader are defined here
as binary32 round-to-nearest, ties-to-even, including the integer conversion.
This deliberately resolves the host-rounding uncertainty in LIN-01. An
all-zero projection lookup table returns an error because its upstream scale
conversion would divide zero by zero; a zero activation row is valid and has
zero codes, scales and weighted sum.

Attention uses each row's own canonical prefix tiles and the documented
31/42-split schedule, rather than a batch-dependent shared tile extent.
The prefix and self paths remain distinct, including the BF16 boundary on
prefix probability/value weights. D9 resolves masked operands by skipping
their value operations entirely while retaining positive-zero probabilities
in the denominator reduction.

## D2 — Explicit portable math

`portable_math.rs` ports the available algorithms in `pm_math.cuh` operation
for operation, preserving bit constants, expression grouping, precision,
branches and conversion boundaries. This includes binary32 exp/log,
positive-domain log1p, sigmoid, SiLU, softplus, sin/cos, and binary64 exp/log
for the sampler and quality reporting. No platform exp, log, sin, cos or
reciprocal-square-root function is used.

The source's reciprocal square root is `1 / __fsqrt_rn(x)`. The required
correctly rounded binary32 square root is implemented using an integer
radicand, binary search and an exact midpoint comparison, followed by the
separately rounded binary32 division. This implements the specified IEEE
primitive without introducing a platform math dependency.

**There is no exp2 implementation in the permitted source.** D7 explicitly
authorizes the portable `exp` of a separately rounded binary32 product with
ln 2, resolving [MATH-01](SPEC_GAPS.md#math-01).

The port also preserves surprising portable-source behavior. For example,
`pm_log_d(+0)` and `pm_log_d(-0)` return finite bit patterns
`0xc0875427f6acad6f` and `0xc0a0ec2b7d284853`; the function contains no
mathematical zero-domain branch. Version 3 sampling preserves the source's
binary64 `+0.5` uniform construction, including its possible upper endpoint,
and then executes this portable log algorithm. The native-CUDA outcome
described in SMP-03 is not substituted for those operations.

## D3 — Provisional floating dot order

For each unresolved `mma16(acc, a, b)`, start with the incoming binary32
accumulator and visit indices 0 through 15. For each index, round `a[i]*b[i]`
to binary32, then separately round its addition to the accumulator to
binary32. Preserve the surrounding chunk order in chapter 04. No FMA or
tree reduction replaces these operations.

This order is **provisional**. Real Apple Silicon kernel experiments may
justify revising this one named primitive. Such a revision must update this
document and the scalar reference before backend bit comparisons are made.
Explicitly specified reductions elsewhere are not provisional: Hadamard
stages, XOR reductions, the GDN four-part folds, integer dots, and sampler
prefix sums retain their documented order.

## D4 — Int8 KV cache only

Attention stores signed byte key/value codes and one binary32 scale per head
and token, with the chapter 04 signed cache rotation. Query operands retain
their specified BF16 rounding. No KV5/KV4 packing, profile codec, or 16-bit
cache alternative is implemented. A main-model folder load consequently
needs only its weights and optional `yasakani.json`; it does not require a
drafter, KV profile or tokenizer when callers already supply token IDs.

## D5 — Header-derived dimensions

Hidden width, FFN width, vocabulary, layer count, attention heads, KV heads,
attention head width, epsilon and layer types come from the container header.
Shape checks reject mismatches. There is no 5,120-wide or 64-layer runtime
fallback. Any layer type other than `full_attention` selects GDN, as specified.

The container header's tensor/vector descriptors determine the architecture
dimensions that are not named model fields:

| GDN dimension | Header-derived relationship |
|---|---|
| Head width | `linear_attn.norm.n` |
| Value heads | `linear_attn.A_log.n` |
| Value/output width | Value heads × head width |
| Query/key heads | `(in_proj_qkv.rows - output_width) / (2 * head_width)` |
| Convolution taps | `conv1d.n / in_proj_qkv.rows` |

These are the inverse of chapter 03's explicit shape relationships. Products,
subtractions and divisibility are checked before allocation or indexing.
The implementation validates GDN head width 128/four taps and attention head
width 256 because those are the reduction shapes specified in this round;
it does not silently generalize their algorithms to other widths.

For smaller hidden widths, dense GDN gates retain 128 partial accumulators
and the ordered index formula `8*p + 1024*chunk + e`. Chunk count comes from
the header width and only in-range indices are visited. Empty partials start
at positive zero. This is the direct dimensional extension requested for the
tiny model and preserves the original operation order at width 5,120.
Hidden and FFN widths still must be positive multiples of the format's
512-element rotation block. Algorithm block/tile sizes are not model-size
defaults.

The specification supplies no rotary-width header rule. D8 resolves
[ARCH-01](SPEC_GAPS.md#arch-01) with an explicit exception: `model.arch`
`qwen3_5` uses rotary width 64 and requires `model.head_dim == 256`. No new
header field is introduced. The 32 frequency bit patterns are preserved as
specified; `rope_theta` is not used to regenerate them.

## D6 — Validated input and finite execution

The container reader checks the fixed header, actual file size, header/data
boundaries, zero padding, descriptor shapes, code-stream size, every payload
range, integer overflow, rotation metadata and parameter-name uniqueness.
It supports both stream layouts, both sign generators, and both vector
formats. Undefined `carry`, `AYA` and `rar` encodings return unsupported
errors. Binary16/BF16 data is decoded as stored; the model loader rejects
non-finite weights before execution.

The reference rejects non-finite activations, including values created at
rounding or reduction boundaries. Normalization and exponential arguments
are checked before a transcendental could turn infinity into a finite zero.
Signed integer accumulation uses checked addition where its range is not
intrinsically bounded. Public sequence operations validate token IDs,
positions and accepted-row counts before publishing any changed state.

## D7 — Portable exp2 construction

`exp2(t)` is `exp(t * f32::from_bits(0x3f31_7218))`, using the existing
portable binary32 `exp`. The multiplication rounds once to binary32 before
the call; no FMA or platform transcendental is used. Exceptional inputs
follow this product and the existing `exp` behavior.

Upstream calls `exp2f(x * log2e)` at these attention sites and supplies no
portable exp2. This explicit construction resolves
[MATH-01](SPEC_GAPS.md#math-01) without claiming CUDA bit equivalence.

## D8 — Fixed qwen3_5 rotary width

For `model.arch == "qwen3_5"`, rotary width is the constant 64 and the loader
requires `model.head_dim == 256`, as upstream hard-codes. This is a stated
exception to D5's header-derived dimensions, resolving
[ARCH-01](SPEC_GAPS.md#arch-01). The existing `arch` field is used; no rotary
field is invented. A model with `full_attention` and any other architecture
is rejected with `InvalidModel`, rather than guessing a rotary rule. The
reference already requires `arch=qwen3_5` for recurrent models as well.

## D9 — Per-row prefix tiles and skipped masked values

Each query visits only the absolute 32-slot tiles intersecting its own
prefix range in each split: tile k covers positions `32k..32k+31`. No other
row in the pass changes that tile range. A masked slot has score negative
infinity and probability `+0.0`; it still enters the existing probability
sum as `+0.0`.

Masked slots take no part in value accumulation: their scale lookup, weight
multiplication, value lookup, and multiply/add are skipped entirely. No zero
operand is substituted. This resolves [ATT-MASK-01](SPEC_GAPS.md#att-mask-01),
whose unspecified masked operand bits could otherwise change signed-zero
arithmetic under D3. Every row's result is independent of the other rows
sharing its pass.

## Sequence API and quality reporting

`prefill(tokens)` commits all supplied input rows. `run_provisional(tokens)`
computes a straight causal chain on a copy of committed state. `commit(K)`
replays exactly its first K input rows through the same scalar path and
publishes the result only on success. K includes row zero; unlike the upstream
speculation ABI it has no implicit `+1`. K=0 discards the pass. A one-row
provisional call does not commit early. No draft tree/side-row API is included.

`row_logits()` exposes every row of the latest pass; `logits()` exposes its
last row. Committing retains only accepted rows' logits. Discard restores the
last committed prediction. `new_session()` shares immutable weights through
single-threaded `Rc`; `reset()` removes all sequence state. Replaying commit
is deliberately slow and avoids retaining M complete recurrent snapshots.

The session context limit is the chapter 10 default of 262,144 or a positive
integer `max_ctx` from the folder manifest, rounded upward to a multiple of
64 with overflow checking. This is a session capacity, not a model dimension.
Manifest weight paths must stay relative to the selected folder.

The scoring example interprets the requested positions as input/logit row
positions: row p predicts token p+1, following chapter 06. It scores
1024..=2046 against tokens 1025..=2047, giving 1,023 losses per window and
20,460 for the supplied twenty windows. Each window starts with reset state.
Negative log-likelihood uses binary64 log-sum-exp, ascending token order,
then ascending window/position order for the mean. Perplexity uses `exp_d`.
This reporting order is explicit and does not claim the published CUDA
quality value has been reproduced.

## Specification errors and implementation findings

- FORMAT.md §4.2 says only the first step reads before the sequence start.
  Its allowed bit widths 4 and 5 also make the second step read negative bit
  positions. The explicit `b[p<0]=0` equation is unambiguous and is used for
  every step. Tests cover all five supported bit widths in both layouts.
- FORMAT.md defines original-coordinate matrix reconstruction mathematically
  without a floating evaluation order. The fixture's sign/state/vector and
  rotated-float SHA-256 checks are exact; original-coordinate samples and
  Frobenius norms use the small tolerances documented in the fixture test.
  This reconstruction helper is distinct from the chapter 02 integer
  compressed operator.
- The portable log zero behavior above differs from the native-math
  discussion in SMP-03. It is preserved and tested under D2.
- The three former blocking ambiguities and their historical implementation
  boundaries are retained in [SPEC_GAPS.md](SPEC_GAPS.md), resolved by D7–D9.
  The upstream specification is preserved, with short notes linking these
  reference decisions.
