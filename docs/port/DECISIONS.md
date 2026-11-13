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

## D10 — Mapped CPU compressed integer kernel

`yskn-engine` shares one read-only `Arc<Mmap>` across matrix handles. A
`Container<Cursor<&[u8]>>` runs the existing header, actual-length, padding,
shape and payload-range checks; no second parser or format change is needed.
Code words remain in the file's `row` or `warp` layout. Their public view is
little-endian bytes because the format permits unaligned payloads. The
mapping constructor is unsafe only for the external-file invariant: callers
must prevent modification/truncation until all matrix handles are dropped.
Malformed, stable files are validated before execution. Derived table/scales
use the same quantization helper as the scalar reference; signs use the
format reader's generators. `memmap2` is the only added external dependency.

Each worker visits 32 output rows × 512 columns, decoding one row's two
256-weight sequences into a reusable 512-byte stack buffer. That buffer is
consumed by up to 16 activation rows before advancing. Larger M is chunked
at 16. Sequence scratch is 33 words (132 bytes), including a zero sentinel;
there is no persistent decoded matrix or repacked code stream. Warp
addressing uses the *global sequence index*, including when physical rows
are not divisible by 32. Bits 6 and 8 use constant-shift windows (16 and 8
pairs per decode block); bits 4, 5 and 7 use the general 12-bit window.
Zero extension applies to every negative starting bit, including step 1 at
bits 4/5. All indexing of file bytes remains checked Rust.

Portable Rust, NEON widening multiply/pairwise widening add, and NEON SDOT
are explicitly selectable. SDOT uses four independent i32 accumulators.
Each 512-term dot is bounded even for activation -128; multiplication by
integer scale widens to i64 and accumulation is checked. The result is
row-major M × physical_rows, including padding, with a prefix API for head
slices. The reference's exposed `weighted_sums` also supplies `project`'s
integer stage, preserving the subsequent float operation order.

`Engine::new(kernel, threads)` rejects unavailable variants. `from_env()`
reads `YSKN_KERNEL` once per process; absent an override it chooses detected
SDOT, then NEON, then portable. Architecture-specific functions are gated by
`is_aarch64_feature_detected!`; the portable implementation has no intrinsics.
Only the engine crate requires Rust 1.98, where `vdotq_s32` became stable;
the existing crates retain their MSRV. Workers live until engine drop, with
no thread creation per projection. Shared activation copies are small and
per-call; weights remain mapped. Workers claim 256-row tasks from an atomic
counter (32-row tasks for small matrices) and return owned integer results
through channels. This balances P/E cores without affinity or unsafe
cross-thread output pointers. The default is min(available CPUs, 10).

### Measurements and rejected alternatives

Measured on the supplied 4P+6E M5 machine with Rust 1.100 release builds.
Every pass calls all 400 body matrices plus the head, reading 9,758,310,400
code bytes logically once for M <= 16. Each configuration warms up first
and reports seven full passes, with no other build/test/timing process of
this task running. The code-rate is logical bytes / time, **not** a hardware
DRAM-bandwidth counter. Timings below are median milliseconds:

| Implementation / scheduling | Threads | M=1 | M=16 |
|---|---:|---:|---:|
| General window, single SDOT accumulator, static | 4 | 1793.587 | 4204.389 |
| Constant 6/8-bit shifts, single SDOT accumulator, static | 4 | 1513.296 | 3012.675 |
| Array-sized decode slices + four SDOT accumulators, static | 4 | 1384.643 | 2632.731 |
| Same decode, portable products, static | 4 | 1441.599 | 3066.136 |
| Same decode, widening NEON products, static | 4 | 2857.873 | 5949.387 |
| Four SDOT accumulators, static | 1 | 3351.644 | — |
| Four SDOT accumulators, static | 2 | 1754.274 | — |
| Four SDOT accumulators, static | 6 | 1021.967 | 1855.703 |
| Four SDOT accumulators, static | 8 | 922.710 | 1674.221 |
| Four SDOT accumulators, static | 10 | 846.821 | 1513.512 |
| Four SDOT accumulators, dynamic | 8 | 792.665 | 1466.976 |
| Four SDOT accumulators, dynamic | 10 | 814.367 | 1267.494 |

There was scheduler/load variation: the NEON M=1 minimum was 1308.668 ms,
and the first dynamic 10-thread M=1 minimum was 683.152 ms. A focused
seven-pass M=1 confirmation of dynamic 4/6/8/10 workers gave medians
940.354 / 827.898 / 735.930 / 683.797 ms (minima 925.875 / 816.996 /
727.697 / 681.410). Ten workers win that confirmation and M=16; callers
can reduce the count. Static equal partitions, general decoding for released
bit widths, and the single SDOT dependency chain were rejected. Portable
and widening NEON remain forced fallbacks; the clear M=16 SDOT advantage
justifies the additional runtime variant. i8mm and SME were not implemented
or benchmarked; no performance claim is made for them or other CPUs.

Final proof run, default SDOT with ten dynamic workers (seven samples per M):

| M | Median ms | Minimum ms | Code bytes | GB/s median | GB/s minimum |
|---:|---:|---:|---:|---:|---:|
| 1 | 641.124 | 631.342 | 9758310400 | 15.221 | 15.456 |
| 2 | 687.818 | 684.961 | 9758310400 | 14.187 | 14.247 |
| 4 | 778.207 | 773.383 | 9758310400 | 12.539 | 12.618 |
| 8 | 940.028 | 936.071 | 9758310400 | 10.381 | 10.425 |
| 16 | 1259.312 | 1254.213 | 9758310400 | 7.749 | 7.780 |

These times include worker dispatch, activation copies and output allocation;
model validation, table preparation and random input generation are outside
the timed passes. All six brief proof commands passed locally; debug and
release each passed 60 tests, and all twelve real-tensor/variant comparisons
passed. No remote CI run was performed.

Reproduce the selected configuration with `bench_linear MODEL.yskn`; use
`--kernel portable|neon|dotprod --threads N --m 1,16` for focused comparisons.
Omitting `--m` always measures 1, 2, 4, 8 and 16. Historical decode/scheduling
alternatives above are measurements of intermediate implementations, not
extra production switches.

The M=1 confirmation is about 14.27 GB/s logical code rate, far below the
nominal 307 GB/s (31.8 ms code-only ideal). The decode improvements and
M-row amortization indicate decode/table-lookup and execution throughput,
not streaming DRAM capacity, as the main remaining cost; this is an
inference, not a hardware-counter profile. Next experiments, in expected
gain order: vectorized state extraction with fewer scalar lookup/load
instructions; multiple output rows decoded together to overlap table-load
latency; then i8mm/SME for larger M if measured product cost warrants the
packing/dispatch complexity. Do not repack the full code stream.

Synthetic tests compare every host variant with the scalar integer oracle
across both layouts, bits 4–8, required K/M, padded/partial rows, unaligned
payloads, extreme bytes and signed scales, one/many workers, and distinct
rows across the M=16 boundary. Mapped corruption/truncation and mapping
ownership are also tested. Real verification covers GDN qkv, MLP down,
attention q, and the first 4096 head rows for M=3. No new specification
error was found; the previously documented bits 4/5 negative-start wording
and the distinction between logical and physical output rows remain
important. This crate implements the integer operator, not a complete
floating-point inference backend.

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
