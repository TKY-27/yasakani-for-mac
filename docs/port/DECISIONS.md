# Round 2 reference decisions

Baseline: merged specification PR #1, commit
`157e33eac8d5ddd0e031b1826fe28e67f57f3787`. Implementation sources are
`docs/FORMAT.md`, `docs/spec/`, and the explicitly permitted
`src/pm_math.cuh`. No other upstream implementation file was opened in that baseline round.
D13 explicitly ports the upstream KV codec source under brief 05a and its
attention consumers under brief 05b.

## D1 — Internal bit consistency

The Rust scalar reference defines this project's numeric contract. We do not
claim equality with the CUDA binary: the approved specification leaves CUDA
elementary functions and MMA accumulation unresolved, and upstream dispatch
can depend on row count.

Every input row follows one canonical scalar path in ascending token order,
regardless of prefill chunk size or provisional pass width. Floating operations
remain separate binary32 or binary64 operations; D15 adds only the draft
source's explicitly fused FMA sites. There are no SIMD
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

Historical plain `Int8` baseline. [D13](#d13--profile-quantized-kv-row-codec)
adds selectable profile modes; the default loader still uses this baseline.

Attention stores signed byte key/value codes and one binary32 scale per head
and token, with the chapter 04 signed cache rotation. Query operands retain
their specified BF16 rounding. This mode does not use KV5/KV4 packing or the profile codec. No 16-bit
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
important. At D10 this crate implemented the integer operator only; D11 adds the complete
floating-point main-model path.

## D11 — Layer-major mapped main model

`yskn_engine::Model` resolves the same folder manifest and header-derived
architecture as the reference, or accepts a direct `.yskn` path. Both paths
call the shared `model_location` and `load_weights`; scalar and mapped
embedding tensors use the same `Embedding` implementation. All tensor code
streams, including embedding residuals, stay in a single shared mapping.
Only vectors, tables, scales, signs and sequence scratch are owned. Opening
is unsafe for the same external-file stability invariant as D10; callers
must prevent modification/truncation through model drop.

`prefill` validates every token and the complete context extent before doing
work. It computes on a copy of sequence state and publishes the new state,
position and all requested row logits only when the entire call succeeds.
An empty call is a no-op. Pass width is configurable and positive (default
16); widths above 16 still make one kernel API call per projection, with the
existing kernel's internal 16-row decode groups. Within a pass, embeddings
are followed by every layer, then final norm/head. GDN qkv/z share qkv's
prepared signs; attention q/k/v share q's, exactly as in the reference.

The existing persistent pool now also runs index-range closures. Independent
embedding rows, normalization/activation preparation, projection restoration
and rotation, residuals and MLP gating run in parallel. Attention rows are
prepared independently, their cache entries are appended in ascending
position, then query rows and heads run in parallel. Each query explicitly
reads only `cache[..position]` and its separately prepared self entry. Later
rows cannot change its prefix tiles, split plan, masked values or arithmetic.
GDN `step` calls stay in ascending row order on each layer's state.

There are no parallel floating reductions: each row/head executes the shared
reference scalar primitives in their original operation order. The integer
kernel is exact. `CompressedLinear::project` and the engine both call
`restore_projection`; attention `step` calls the same prepared-row,
per-head output and cache append pieces as the engine. The reference still
forbids unsafe code. The pool's one lifetime-erasure block permits borrowed
closures on persistent threads: all dispatched jobs are drained before the
borrow ends, including numeric errors, dispatch failures and caught panics.
Results are owned by workers and collected in index order.

The shared `quality` module fixes window validation, next-token targets,
ordered binary64 NLL addition and portable `exp_d`. Both scoring examples
use it. Engine scoring processes batches and exposes half-open window ranges
and JSON window sums/counts. A full run adds individual losses in ascending
window/position order; combining only per-window sums can differ in the last
binary64 rounding bits from that full fold. `PassTiming` records nonoverlapping
wall times for kernel calls, attention preparation/cache/output, complete
GDN steps and remaining pass work. Scoring reports early/late pass timings
and aggregate stage times on stderr; scoring/NLL overhead is outside passes.

Synthetic model comparisons cover all host kernels, one/four workers, widths
1/2/5/16/17/40, uneven input chunks, reset with different tokens and rejected
requests without state changes. The mixed fixture runs 73 rows from position
3 across attention tiles 32 and 64. Four fixed SHA-256 logit digests were
obtained by executing the original pre-D11 `604aecd` sources separately on the
same synthetic writer; they also guard the reference's unchanged bits.
The shared scorer has an independent deterministic sequence test for target
positions, reset, range selection, counts and ordered addition.

### D11 local proof and measurements (2026-11-14)

All six brief proof commands succeeded on this worktree: fmt, clippy with
`-D warnings`, debug and release workspace tests (76 passing in each),
real `verify_model` (all 24 rows/all vocabulary bits in all four requested
configurations), and `score_windows --windows 0..1`. The real scalar model
was never loaded. Runs were serial, with no build/test alongside timing.

Window 0 consumes 2,047 input rows in 128 passes and scores 1,023 positions:
```text
window=0 scored_positions=1023 nll_sum=5.0871858607572784e2
scored_positions: 1023
mean_negative_log_likelihood: 0.497281120309
perplexity: 1.644244684328
```

The completed measurement uses default dotprod, ten workers and 16 rows per
pass (`YSKN_KERNEL` unset). All stage entries below are elapsed wall seconds,
not CPU-time sums. Early is the mean of 13 full passes at positions 0..192;
late is the mean of three full passes at 1984/2000/2016. The final 15-row
pass is included only in the whole-window totals.

| Scope | Total pass time | Integer kernel | Attention | GDN step | Rest |
|---|---:|---:|---:|---:|---:|
| Early 16-row mean | 2.764065 | 1.206741 | 0.103914 | 1.334633 | 0.118777 |
| Late 16-row mean | 3.731764 | 1.206684 | 1.042580 | 1.362127 | 0.120373 |
| All 128 passes | 427.336007 | 158.660025 | 71.930404 | 180.898005 | 15.847574 |

Shared scoring elapsed: 429.062166 s; whole command:
429.584103 s. The 1.726159 s outside passes covers reset,
state cloning, ordered NLL evaluation and reporting. Linear extrapolation
of the scored window gives 8581.2 s (about 2 h 23 min) for all twenty
windows; this is an estimate, not a completed full-set run.

Peak resident memory including mappings was **10,569,367,552 bytes**
(10.569 GB; 9.843 GiB), measured by Darwin `wait4` rusage.
**Measurement limitation:** the requested `/usr/bin/time -l` completed its
first scoring run (421.09 s real, score time 420.559641 s), then failed
with `time: sysctl kern.clockrate: Operation not permitted` before printing
resident memory. Its wrapper returned 1. No permissions or system settings
were changed. The same one-window cargo command was therefore repeated
under a small Python `os.posix_spawnp`/`os.wait4` observer; cargo returned 0,
and its peak RSS is the number above. This is not represented as a successful
`time -l` RSS reading. Both runs produced exactly the same printed 17-digit
window NLL sum and final perplexity. Their score times differ by about 2%.

No new specification/reference error was identified. D2 portable math, D3
provisional attention dot order and D9 masking are preserved; these CPU
comparisons do not prove CUDA bit equivalence or the published full-set score.

Provisional execution, commit/replay, speculation and draft-model loading
are deliberately absent from the engine. Full twenty-window quality
validation remains the orchestrator's next gate; one-window scoring and
configuration agreement do not establish the published full-set quality.

## D12 — Metal backend for the compressed linear operator

Decision (2026-11-14): the compressed linear operator's exact signed-64
weighted sums run on the GPU kernel (first built as the crate `crates/yskn-metal`,
merged into `crates/yskn-engine` by D14). The CPU kernel
(D10) remains the fallback and the test oracle for real tensors. Design,
bounds and rejected variants: `crates/yskn-engine/METAL.md`.

Measurement (M2 in `docs/port/STATE.md`), with no scoring run, build or test
running:
`cargo run --release -p yskn-engine --example bench_metal -- MODEL.yskn`
(at the time of measurement the example lived in `-p yskn-metal`)
Apple M5, threadgroup 32x4, the whole 10,311,344,128-byte mapping as one
no-copy buffer, 401 matrices, 9,758,310,400 code bytes per pass, 7 passes
after a warm-up, measured at the pre-rebase probe commit 4e5fe32. Before
timing, Metal equalled the CPU kernel on four real tensors (M=3). Median ms
per whole pass, including activation upload and result readback:

| Rows (M) | Metal, one command buffer per pass | CPU dotprod, 10 threads (M1) |
|---:|---:|---:|
| 1 | 111.2 | 645.8 |
| 2 | 127.7 | 671.5 |
| 4 | 186.9 | 757.7 |
| 8 | 330.5 | 924.0 |
| 16 | 634.1 | 1247.6 |

Reason: Metal wins at every row count, 5.8 times at one row and 2.0 times
at 16. The GPU removes the decode cost (the CPU does one scalar table
lookup per weight pair), while each added row costs about the same on both
(about 35 ms on the GPU, 40 ms on the CPU). Command-buffer granularity
matters: one buffer per matrix costs about 72 ms more per pass than one per
pass, one per layer about 12 ms more. A CPU/GPU split of one matrix's
output rows could add at most 15% at one row and 34% at 16, and the CPU is
needed for the recurrent and attention layers, so that split is deferred to
step 7.

## D14 — Metal projections in the engine and parallel recurrent heads

Decision (2026-11-14): `yskn_engine::Model` runs its compressed projections on
the Metal kernel by default, and runs the Gated DeltaNet value heads of a pass
in parallel. Logits are bit-identical to the scalar reference on both
backends.

**Crate layout.** The `yskn-metal` probe crate is merged into `yskn-engine`
as the macOS-only module `metal` (`src/metal.rs`, `src/linear.metal`, design
notes in `crates/yskn-engine/METAL.md`). The project targets Apple Silicon
only, the probe already depended on the engine, and a backend trait would
have had a single implementation in a crate that depends back on its user.
`objc2`/`objc2-metal` are macOS-target dependencies only; other targets build
without the module, and `Backend::Metal` is an error there. `bench_metal`,
`synthetic_model` and the kernel tests (`tests/metal.rs`) moved along. D12's
`-p yskn-metal` commands are now `-p yskn-engine`, for example
`cargo run --release -p yskn-engine --example bench_metal -- MODEL.yskn`.

**Backend selection.** `Model::with_backend(path, engine, backend,
rows_per_pass)` takes `Backend::Cpu` (D10 kernel on the pool) or
`Backend::Metal` (D12 kernel). `Model::load` reads `YSKN_BACKEND` (`cpu` |
`metal`); unset, it uses Metal when `MTLCreateSystemDefaultDevice` returns a
device, else CPU. An unknown value is an error, and so is Metal without a
device: there is no silent fallback. `with_engine` keeps its meaning (CPU) and
explicit constructors never read the environment. Both backends return the
same exact row-major M×N i64 sums including padded rows, and the same code
consumes them from `restore_projection` on, so the backend cannot change a
bit; the tests check it anyway. The CPU pool runs everything else on either
backend. With the Metal module, `Model` is not `Send`, because objc2-metal's
buffer objects are not.

**Dispatch grouping.** Projections that share one prepared input are encoded
into one command buffer: GDN `qkv`+`z`, attention `q`+`k`+`v`, MLP `gate`+`up`.
`out`/`o`, `down` and the head are one command buffer each. A 64-layer model
(48 GDN, 16 attention) therefore makes 4 × 64 + 1 = 257 command buffers per
pass, not 401. `MetalEngine::weighted_sums_shared` uploads the group's input
once into a reused activation/scale buffer and reads each matrix's sums back
from a reused output buffer per group slot. These buffers grow to the largest
size seen and are never shrunk; with the no-copy mapping buffer and the
cached lookup tables (D12), no GPU buffer is allocated per call once the
sizes have been seen. Rows per pass above 16 are split into several
dispatches inside the same command buffer, as before.

**Recurrence stage split.** `GatedDeltaNet::step` is now three public stages,
called by `step` itself in its original order:

1. `prepare(history, qkv, z, weighted_hidden, inverse_rms)`: validation, BF16
   projections, convolution, query/key normalization and both dense gates.
   It reads the previous `convolution_width - 1` rounded projections, never
   the recurrent state.
2. `recur_head(head_state, row, value_head)`: one value head's decay, delta
   update, output norm and gating on its own `head_dim × head_dim` block.
3. `advance_history(state, row)`: shift the row's rounded projection into
   the convolution history.

Because the history holds only rounded projections (`round_projection`), row
r's convolution input for a pass is the state's history rows followed by the
rounded projections of rows before r, all known before any recurrence runs.
The engine therefore prepares all rows of a pass in parallel, then runs each
value head's chain of M rows, in ascending row order, as one pool task, and
finally appends the rows to the history in order.

Why parallel heads cannot change bits: a head reads and writes only its own
state block; its output depends only on that block, the immutable prepared
row and its own weights. Within a head, every floating operation runs in the
same code and order as the reference's `step`; nothing is reduced across
heads or rows; a thread only decides which core executes a chain. The
D11 digests are unchanged, a reference unit test runs heads in reverse order
against row-major `step`, and the engine tests cover 1 and 4 workers.

Not separable by head: the chain of rows within a head (each row reads the
previous row's state), which stays sequential, and the history append, which
stays sequential in row order (a copy per row). Nothing else had to remain
sequential. The finite checks of a head's state block and output now run at
the end of that head rather than after all heads. Whether a row fails is
unchanged; when several checks fail, the `NonFinite` context reported can
differ. In the engine, a prepare error of a later row can be reported before
a head error of an earlier row; the whole pass fails either way and
`prefill` publishes nothing (D11).

**Related fixes.** `map_range` reports the failing index that comes first,
as a sequential loop would (panics are caught per index).
`TrellisStates` validates a trellis descriptor once per tensor instead of once
per state; `tensor_state` keeps its per-call validation.

### D14 local measurements (synthetic model, 2026-11-14)

`cargo run --release -p yskn-engine --example synthetic_model -- FILE 8`
(8 layers: 6 GDN with 4 value heads and 2 attention; hidden 5120, FFN 17408,
4096-row head; 856,300,800 bytes) in a fresh temporary directory, then
`bench_model FILE --backend B --rows-per-pass R --passes P` (P = 7 for 16
rows, 15 for 1 row), each run three times. Another CPU-heavy job kept the
load average near 10, so the numbers are noisy. Values are seconds per pass:
the minimum over all passes of the three runs / the median of the three run
medians. "Before" is the CPU engine at merge commit `bf89ad1` plus
`bench_model`.

| Rows | Configuration | Total | Kernel | Recurrence | Attention | Rest |
|---:|---|---:|---:|---:|---:|---:|
| 16 | CPU before | 0.1920 / 0.2460 | 0.1378 / 0.1689 | 0.0152 / 0.0166 | 0.0007 / 0.0016 | 0.0336 / 0.0529 |
| 16 | CPU after | 0.1704 / 0.2375 | 0.1350 / 0.1762 | 0.0048 / 0.0085 | 0.0008 / 0.0018 | 0.0191 / 0.0452 |
| 16 | Metal after | 0.0813 / 0.1106 | 0.0602 / 0.0715 | 0.0044 / 0.0064 | 0.0007 / 0.0017 | 0.0154 / 0.0259 |
| 1 | CPU before | 0.0875 / 0.1086 | 0.0703 / 0.0981 | 0.0009 / 0.0012 | 0.0001 / 0.0002 | 0.0044 / 0.0124 |
| 1 | CPU after | 0.0821 / 0.1055 | 0.0740 / 0.0991 | 0.0006 / 0.0012 | 0.0001 / 0.0002 | 0.0040 / 0.0106 |
| 1 | Metal after | 0.0198 / 0.0355 | 0.0154 / 0.0269 | 0.0005 / 0.0009 | 0.0001 / 0.0001 | 0.0034 / 0.0062 |

The synthetic GDN layers have only 4 value heads (the real model has 48), so
the recurrence gain here (about 3× at 16 rows) is bounded by 4 parallel
chains and understates the real-model gain. The synthetic file has 33
command buffers per pass; the real model's 257 add about 46 ms at the
measured 0.18 ms round trip. The real model was not opened: its numbers
come from the orchestrator's `verify_model`, `dump_logits` and
`score_windows` runs (commands in [README.md](README.md)).

## Scalar-reference sequence API and quality reporting

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


## D13 — Profile-quantized KV row codec

Three cache modes have distinct numeric contracts:

| Mode | Contract |
|---|---|
| `Int8` | D4's plain rotated row, signed int8 plus FP32 scale; no profile |
| `Int8Profile` | Initial int8 encode, dequantize, profile fake-quantize, then final int8 encode; upstream's published “int8” record |
| `Kv5` | Same initial int8 and profile selection; pack eligible layer/sides into the common 4/5-bit width plus scale; otherwise use `Int8Profile`; upstream's default |

The shipping build defines mode 3 (`CMakeLists.txt:46`) and runs profile
requantization even with packed capacity disabled (`src/engine35.cpp:688-704`).
Its published perplexities 2.075900 and 2.0759369 differ by much less than the
quality gate's 0.001 tolerance. Perplexity therefore cannot prove codec
correctness. D4 remains available. The row codec is implemented in `kv_profile.rs` and
`kv_codec.rs`; brief 05b wires it into reference attention and model loading.

The reader checks every referenced region before indexing or allocation,
including directories, table indices, consumed means/bases, finite numeric
values, size arithmetic, and partial region overlaps. Exact record aliases
are allowed. Unknown header and record bytes stay uninterpreted (AUX-01).
It accepts positive header-derived attention-layer/KV-head counts with
`groups = layers * heads * 2` and checks model compatibility explicitly;
upstream's loader fixes these to 16/4/128 (`src/engine35.cpp:40-43`).
Only 256-channel rows and the source's 8-state-bit/64-chain/8-fraction-bit
code tables are supported. At most two basis vectors are consumed even if
the declared count is larger (`src/kvq_fast.cuh:90-91`).

The codec ports the source operation for operation, preserving separate RN
binary32 operations, correctly rounded `sqrt_rn`, nearest-even integer and
binary16 rounding, descending warp XOR reductions followed by eight ordered
warp additions, initial int8 dequantization, mean subtraction, sequential
basis projection/restoration, and the first strictly smaller scale error.
There is no explicit FMA in these encoder expressions and none is introduced.
Sources: `src/kvx_codec.cuh:50-69`, `src/kvq_fast.cuh:33-41,78-95,139-168,189-234`,
`src/kvq_gpu.cuh:53-146`.

General trellis and VIT2 are separate implementations. The general search
starts state zero at zero cost, other states at `0x40000000`, breaks
predecessor ties by lowest j, and final ties by unsigned cost then state
(`src/kvq_fast.cuh:108-135`). VIT2 retains the source's normalized cost keys,
low-bit predecessor ties, paired emission minimum *before* cost addition for
5-bit searches, and final signed cost/state tie
(`src/kvq_vit2.cuh:64-88,143-192`). Its global eligibility check follows
`src/kvq_fast.hpp:51-66`, including the [-1100,1100] bound on all referenced
4/5-bit tables. Three-bit groups still use the general search.

All potentially overflowing i32 products, additions and subtractions return
an error instead of asserting CUDA signed-overflow behavior (ATT-04).
Using the actual profile bounds [-886,874] and targets clamped to
[-2048,2048], the tests bound **all** general costs including the finite
infinity by 1,624,676,608. A reduced-cost spread invariant bounds every VIT2
intermediate in magnitude by 474,879,999 (4-bit) or 923,638,527 (5-bit).
Thus overflow is unreachable for this artifact's clamped targets, while
synthetic int16-extreme tables demonstrably return an overflow error.
The independent i64 forward-edge DP also checks selected path costs at both
clamp limits and on seeded targets, for both search implementations.

The read-only local `kvq_u5.bin` (17,432 bytes) resolves ATT-03/AUX-02:
128 groups all use mode 0 and five bits; even groups use table 0 and odd
groups table 1. Both tables have bounds [-886,874]; gains are 1.0015934
and 1.0 respectively. There are nine multipliers in file order:
0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.07, 1.1.
VIT2 is eligible. All 16 key sides and 13 value sides pack; value ordinals
10, 14, 15 fall back. Group 83 has guard 1; groups 115/123/125 have one
basis and 121/127 have two. Means occur in groups
3, 5, 7, 13, 63, 71, 77, 95, 98, 99, 103, 106, 109, 113, 115, 117,
119, 121, 123, 125, 127. No artifact payload is stored in the repository;
CI uses synthesized profiles, with optional real-file assertions only when
`YSKN_MODEL_DIR` is set (an unusable set path fails).

`CacheRow::{Int8,Packed}` owns the row bytes/words and scale. `key_base` and
`value_base` return the unscaled attention operand; `decode_packed` returns
all 256 base operands for reuse. Packed decode selects a BF16-rounded
codebook entry, **not** the encoder's unrounded reconstruction, and leaves
scale and mean separate (`src/kvq_atxpk.hpp:39-42`, `src/q35_atxpk.cu:69-74`).
Attention retains BF16 query operands, uses the BF16 table operand for
packed keys/values, and retains the existing BF16 rounding on
prefix `probability * value_scale`; self weights remain FP32, including the current token's own encoded row. Key means
are ignored; attention adds `split_denominator * value_mean[channel]` to each packed
value split numerator after the self contribution and before split
combination (`docs/spec/04_ATTENTION.md`, “Packing, decoding, and the int8
fallback”; `src/q35_atxpk3.cuh:343-380,412-468`,
`src/q35_atxpk4.cuh:354-410`). `value_mean` returns zero for int8 rows, whose
means/bases were already restored before their final int8 encoding.

With nine multipliers and four chains, VIT2 examines
`9 * 4 * 64 * 256 = 589,824` emission candidates per side-row, or
75,497,472 per token (128 side-rows). Each candidate needs an integer
multiply and add; paired minima, shared cost adds, final minima and
normalization bring the dominant recurrence to roughly 2.06 million
integer operations/comparisons per row, or 264 million per token, plus
coefficient setup, checked-arithmetic branches, about 30,000 floating
arithmetic/reduction operations per row, and packing. This is a count,
not a throughput measurement. It intentionally remains a scalar reference.

Proofs also cover malformed files, independent DP optimality and path
legality, explicit tie rules, 4/5-bit word packing and BF16 decode boundaries,
guard/mean/basis/mode-1 behavior, row independence and frozen SHA-256
synthetic output digests. Digests freeze the source port, not a CUDA golden
capture. Native CUDA execution/equality and end-to-end perplexity are not
verified by these synthetic proofs.


### D13 attention integration and loading (brief 05b)

`KvMode::{Int8, Int8Profile, Kv5}` is selectable through
`Model::load_folder_with(folder, mode)`, or
`Model::from_container_with_kv(container, mode, profile)` with an explicit
`Arc<KvProfile>`. The existing `load`, `from_container`, `load_folder` and
`load_weights` entry points **default to plain `Int8`**, preserving the
pre-D13 logits and digests. Profile modes require a profile. A folder reads
`files.kv_profile` from `yasakani.json`, defaulting to `kvq_u5.bin`; both
profile and weights paths must remain relative to and inside the folder,
including symlink targets. `Int8` does not read or require a profile.
A direct weights path with a profile mode needs the explicit-container API.

The loader checks attention-layer count, KV heads and 256-channel width
against the profile before loading tensors. Group index is
`((attention_ordinal * kv_heads + head) * 2 + side)`; recurrent layers do
not increment the ordinal (`src/kvq_atxpk.hpp:65-68`). Mode, profile and
ordinal live on each immutable `Attention`; `prepare_row` encodes at the
absolute position, including guard bypass. The existing public stage
signatures are unchanged. The profile uses `Arc` so the mapped engine's
parallel prepared-row/output stages can continue sharing attention.

Reads are defined as follows:

- `Int8` and `Int8Profile` consume exact signed-int8 base operands and the
  row's FP32 scale through the unchanged D3/D7/D9 operation schedule.
- `Kv5` consumes BF16-rounded codebook operands for packed rows, including
  self. A packed key score is `dot(query_bf16, codebook_bf16) *
  (row_scale * 0.0625)`, with separately rounded products. Prefix uses D3;
  self uses the four interleaved partials and XOR fold. The shipping
  `QK8=false` branch has no extra table-int8 scale and no key mean
  (`src/q35_atxpk3.cuh:343-380,423-444`,
  `src/q35_atxpk4.cuh:287-325,365-386`, `src/kvq_atxpk.hpp:42-45`).
- Prefix rounds **probability times value scale** to BF16; self keeps that
  product FP32. Both multiply the BF16 table base
  (`src/q35_atxpk3.cuh:402-406,445-458`,
  `src/q35_atxpk4.cuh:345-349,387-400`, `src/q35_atxpk.cu:55-56,66`).
- After self and before emitting each split, packed values add
  `split_denominator * mean[channel]`. That FP32 denominator includes
  self and uses unrounded probabilities, rather than the rounded/scaled
  value weights. Missing means are zero, and int8 fallbacks receive no
  extra term because the codec already restored their basis/mean
  (`src/q35_atxpk3.cuh:443-475`, `src/q35_atxpk4.cuh:385-417`,
  `src/kvq_fast.cuh:214-218,234`).

The four rules stated in brief 05b hold exactly in both pk3 and pk4.
Chapter 04 already had the correct key-mean asymmetry and BF16 operand;
its revised wording makes the denominator/self and shipping-int8
interpretations explicit. D1 remains internal consistency, D3 provisional
scalar dot order, D7 portable exp2, and D9 per-row absolute tiles with no
masked value operations. Synthetic profiles may exercise side combinations
beyond the shipping CUDA dispatch (whose five-bit value-only path errors,
`src/q35_atxpk.cu:722-724`); the scalar mode applies the same specified row
operands to any packable side.

Synthetic proofs cover actual packed cache contents and int8 fallback,
key-mean independence, exact value-mean output with a zero codebook,
prefix/self rounding, unread masked packed slots, delayed stage output
with future cache entries, absolute guard positions, profile dimensions,
folder containment, and codec failures after an earlier row succeeds.
A mixed three-layer model (GDN plus two attention layers) checks 73 rows
starting at position 3 in one pass, singles and uneven chunks, then
continuation, reset and provisional replay. The logits differ across all
three modes. Their frozen SHA-256 values (seed 107, prefix `[8,21,5]`,
73 tokens `(17*row+4)%128`, row-major little-endian f32 bits) are:

| Mode | SHA-256 |
|---|---|
| Int8 | `1e9a723c945f8b0b36a5204b16cf237500fdc2183727e5beb07b841b148114fd` |
| Int8Profile | `3b952134c81b86e93601de75379c10e80ac898cae92e30275f22265d781710dd` |
| Kv5 | `5d474ca9559bcc6d8eb367a2634e578842c0813fc66a55a862f054f4521422e4` |

These freeze this source port, not a CUDA capture. Existing D11 Int8
baselines remain unchanged. The scalar score example accepts
`--kv int8|int8-profile|kv5`; shared `quality::parse_kv_mode` and
`score_with_kv` pass the mode to a backend loader without changing the
shared NLL/position fold. A mapped-engine follow-up only needs to select a
mode, call `model_location_with_kv` and `load_weights_with_kv`, and route
its score CLI through these helpers. Its stage calls need no changes.
No engine source, real weights or full quality run is part of brief 05b.

verified by this codec-only brief.

## D15 — Scalar block draft and reduced head

Decision (2026-11-14, brief 05d): `yskn-ref::draft` and `draft_head` port
`src/q35_bdraft.cu` §§1–8 of chapter 07 for the default eight-row block.
This draft affects acceptance rate/speed, never the verified output text.
The reference is single-threaded safe scalar Rust; compressed matrices
retain their codes and reuse `CompressedLinear` with output rotation
disabled. Every A8 input uses the appropriate prefix of `bdraft2.fc`'s
signs (`src/engine35.cpp:1638,1770–1782,1801–1818`). Per-tensor input signs
and output rotation metadata are not substituted at execution time.

Math: use D2/D7 portable `rsqrt`, `exp`, `exp2`, `sincos`, `exp_d` and
`log_d`, explicitly resolving the port side of DRF-03. Preserve the source's
explicit `fmaf` via IEEE `f32::mul_add`. Other multiply/add operations stay
separate with their original grouping and precision. No platform
transcendentals, unsafe blocks, SIMD or threads are introduced.

Every explicit FMA site in `src/q35_bdraft.cu` (line numbers are upstream):

| Lines | Purpose / port handling |
|---|---|
| 34, 465 | 256-lane RMS squares (unfused/fused variants); one port norm |
| 141 (three nested calls) | Four-channel block dot, scaled Q before dot |
| 146, 148 | Block online denominator and numerator |
| 175 (two nested calls) | Serial history dot, even then odd, scale after dot |
| 205 | Ascending-key history numerator |
| 226 (two), 579 (two) | Unfused/fused-output split combination; one port combiner |
| 303 | Dense BF16 hidden-to-correction projection |
| 334, 409 | Legacy/current predecessor correction; one port correction |
| 786 (four) | Rank projection, block then u then channel |
| 839 | Approximate score, contiguous 16 components per lane |
| 920–921 (four) | Full-width int8 rescoring |

`draft_kernels.cu` is the separate D3/MTP path, not bdraft2; none of that
path is added. No explicit FMA is introduced by compressed restoration,
head norm/RoPE, dynamic filters, residual adds or GLU. Gumbel reuse only
changes `output_head_sampler::gumbel` visibility to `pub(crate)`; its body
is unchanged. No helper visibility/accessors in the protected model,
attention, GDN, KV or format modules were changed.

Shortlist (DRF-01): preserve the two histograms and 21-bit prefix cutoff,
include all tokens above it, fill its remaining slots in ascending ID
order. This is one of upstream's scheduling-dependent admissible subsets
(`src/q35_bdraft.cu:882–896`); it is not an exact float top-512 sort.
Rescore top16 ties use smaller token ID. Correction stays autoregressive
over the block: predecessor starts at frontier and becomes each selected
token. Greedy corrected ties keep candidate index; sampled ties use
corrected-score rank (token ID as its tiebreak). The draft temperature is
`0.001_f64 * 850.0_f64`, independent of the main request, top-p is 0.95,
and Gumbel is keyed by seed, frontier+1+j, and token ID
(`src/q35_bdraft.cu:404–434`). API `DraftChoice::Sampled` means the caller
has enabled main sampler version 3. All other main modes use `Greedy`.

Dimensions are read from the container/SHEAD headers, not replaced by
shipping constants. The operation shapes that are fixed are 128-channel
Q/K heads, 32 reduction lanes, 256 norm lanes, 512-column A8/hidden blocks,
16-channel dynamic sharing, rank-512 SHEAD, top16 and eight block rows.
Hidden/FFN are multiples of 512; correction rank is a positive multiple of
32; a power-of-two ring may be 2..2048. Counts and products are checked
before use. Smaller model dimensions use the same zero-initialized lane
arrays/order, with only in-range model indices visited. Mask ID and capture
layers are parameters, default 248070 and [5,19,33,47,61]. Callers must
check main/draft dimension and epsilon compatibility when wiring the hook.

**Main-model capture contract:** for each committed input position p, take
zero-based layers 5,19,33,47,61 in that order AFTER the MLP down-projection
residual add and BEFORE the next/final normalization. The value is
`bf(old_residual + bf(down_projection))`, widened exactly to FP32, without
extra norm/gain/rotation. `old_residual` here includes that layer's attention
or GDN residual branch. Copy from `h_` as in `src/engine35.cpp:798–799,830–833,1650`;
rounding is `src/q35_kernels.cu:1437–1444`. One input row is `[5,5120]`
(slot-major), flattened to 25,600 elements. The upstream capture buffer is
`[slot,row,channel]`; gather its row, do not reinterpret it as row-major
(`src/q35_bdraft.cu:249–252,490–499`). Prompt and accepted verification
rows are supplied once in contiguous ascending positions; rejected rows
are never inserted. The frontier token at p is still uncommitted and must
not already appear in the history ring.

All draft layers project the SAME normalized context. The ring is stored
as BF16 bit arrays `[layer,head,slot,channel]`; only successful complete row
updates publish. Batch push uses exactly the row path, commits completed
rows if a later row fails, and does not change arithmetic. `propose` leaves
the ring untouched; reset/snapshot/restore include the position and every
ring bit. History visible to block row m is
`max(0,p+m-(window-1))..p`; every row also sees ALL eight current block
keys/values, including future masked rows. The 32 chronological history
partials and one block partial use their separate source dot orders.

SHEAD validates magic, signed dimensions, every derived region and exact
EOF before allocating arrays, and rejects nonfinite projection/scales.
Manifest paths are relative and canonically contained, including symlinks.
Only draft files are opened; embeddings come from a caller-supplied provider
of BF16-expanded main embedding rows. Main weight files are never touched.

Read-only artifacts resolve DRF-02: 46 eight-bit warp matrices, no output
rotations, H512/Mulberry32 seed 7/offset 0 everywhere, fc sign length 25,600
and equal sign prefixes for every matrix. There are 35 BF16 vectors. SHEAD
is V=248320/H=5120/R=512. `draft_info MODEL_FOLDER` prints these facts.
One scalar `propose` on the real files took 87 to 99 seconds, 1.6 GB resident.
No contradictory equation/layout was found in chapter 07 §§1–8 or chapter
10; docs clarify dense BF16 correction and feature rounding, record the
artifact facts, and distinguish D15 choices from upstream uncertainty.

Not implemented: main-model capture wiring, speculation/verification loop,
16-row blocks, alternate compressed/subset heads, second/third candidate
policy, margins and HCAP. Top16 and both raw/corrected scores are exposed.
Tests include malformed readers/options, path containment, ring overwrite
and batch identity, snapshot/reset, future-row perturbation, analytical
attention/filter/sparse-head oracles, explicit shortlist/selection ties,
common signs/ignored output rotations and frozen proposal/score digests.
Digests freeze this port, not a CUDA golden capture. CUDA bit equality,
acceptance quality and end-to-end text equality are not claimed.

## D16 — Bounded transactional sessions and lossless speculation

`yskn-engine::Model` now implements the reference's explicit sequence ABI:
`run_provisional(tokens)`, `commit(k)` and `discard_provisional()`. k counts
input rows including row zero; zero discards. There is no one-row precommit
or implicit +1, and a pending pass rejects another pass/prefill/snapshot/restore.
Provisional width is bounded by the constructor's positive `rows_per_pass`.
Long prefill still chunks at that width, atomically across every chunk.

### Ownership, copying and bounds

Attention cache rows append directly to their existing layer cache. Discard,
partial commit, failed provisional execution and failed prefill truncate the
appended suffix. The engine never clones an attention cache. The reference
keeps its existing scalar copy/recompute behavior as the independent oracle.

Each recurrent layer has a committed state and one preallocated working state,
both including convolution history. At transaction start `copy_from` copies
only recurrence/history into the existing working allocation; the committed
state stays unchanged until success. Successful prefill swaps these states.
Provisional execution retains the prepared per-row recurrence inputs in fixed
replay slots allocated once when the model loads: rounded projections, z,
convolution outputs, normalized q/k and dense gates. Each slot copies into its
existing vectors. Full commit swaps the already-computed working state;
partial commit resets working recurrence from committed state and replays only
accepted recurrence/history through the same `recur_head`/`advance_history`
operations. No embedding, compressed projection, attention or MLP is replayed.
Failure cannot publish working recurrence. Ordinary transient projection,
attention and row-preparation allocations remain; fixed replay payload does
not allocate or grow per pass. Context-dependent cache/history storage grows
only with the retained context, not with the number of retries/checkpoints.

The alternative, retaining a full recurrent snapshot per row, was rejected
for memory. With chapter 03's 48 GDN layers, each state has
`48 * (48*128*128 + 3*10240) * 4 = 156,893,184` payload bytes, including
BF16-rounded convolution history represented in f32 by this reference.
The two fixed committed/working states therefore hold 313,786,368 bytes.
A prepared row costs
`48 * (2*10240 + 6144 + 2*2048 + 2*48) * 4 = 5,916,672` bytes.
For maximum pass width W, fixed recurrent/replay payload is
`313,786,368 + W*5,916,672` bytes (408,453,120 bytes at default W=16),
plus small Vec metadata, W token IDs and requested logits. A per-row-state
design would instead add `W*156,893,184` bytes (2,510,290,944 at W=16).
These are dimension-derived payload counts, not measured real-model RSS;
weights, attention caches, operator scratch and output logits are additional.
`session_memory()` reports actual state/replay payload and buffer capacities.

`prefill_with_logits(tokens, LogitRows::{All,Last,None})` controls head work.
All is the unchanged scoring default; Last runs only the final prompt row's
head; None skips heads and clears predictions. Empty prefill remains a no-op.
Failure preserves position, committed recurrence/cache, logits and timings.

### Checkpoints and exactness

Both models' owned `snapshot()` captures recurrence/history, position, the
last committed prediction and exact prefix token IDs, with a session identity.
Attention is retained in the owner, never copied. `restore` checks session/reset,
position and prefix equality before truncating attention and copying recurrent
values into existing allocations. A changed prefix, another session, pending
pass or future checkpoint returns an error. Snapshot values are not accumulated
inside the model; each dropped value releases its own payload. One checkpoint
costs one recurrent state plus `4*position` prefix bytes and one vocabulary row.
Repeated restore at a fixed maximum context extent preserves capacities.

The one generic `yskn_ref::speculation::generate` loop is used by both models.
The supplied prompt tail ends in a pending, unconsumed token. Each pass runs
`[pending, guesses...]`. Guess j is accepted only if it equals the selector's
choice from row j-1. The first mismatch yields the correction token and stops
reading later rows. EOS/callback cancellation stop at their selected token.
Committing exactly the consumed predecessors leaves the final selected token
pending, identically for K=0 and K>0. Output and context limits bound drafting
before the pass. No artificial EOS is introduced by a proposal.

D1/D9 guarantee the same row logits for every straight-chain grouping. The
existing sampler is keyed by seed, absolute prediction position and candidate
ID, with no pass-local RNG consumption. Inductively the selector therefore
chooses the same token at every depth and commits the same state for greedy
and seeded runs, independent of proposer accuracy. Tests compare both token
sequences and a digest over all committed numeric state (including KV codes,
scales, convolution history and committed logits), plus explicit future rows.
The diagnostic digest is FNV-1a, not a cryptographic proof; logit comparisons
check every f32 bit independently. No frozen reference digest was changed.

The upstream source confirms recurrence-only replay (`src/engine35.cpp:843-851`),
its different +1 ABI/one-row shortcut (`1299-1329`) and retained-cache snapshots
(`1550-1583`). Chapter 08 now distinguishes those upstream conventions from
the Rust contract. No new discrepancy was found in those source ranges.
HCAP, adaptive 8/16-row blocks, the next-token table and cost/source policy were
not ported. No draft/draft_head files or D15 section were created.

### D16 synthetic measurement (2026-11-14)

Fresh temporary directory, generated exactly with
`cargo run --release -p yskn-engine --example synthetic_model -- FILE 8`:
856,300,800-byte model, hidden 5120, FFN 17408, vocabulary 4096, six GDN/two
attention layers. No released model was opened. The process could not obtain
a Metal device, so these are CPU dotprod/10-worker timings only. No task build,
test or other model run overlapped the timed run. `bench_speculation FILE
--backend cpu --passes 7` takes one warmup and seven measured samples per
width. Values are medians in milliseconds, including real operator allocation
and dispatch costs:

| Width | Provisional | Partial accepted rows | Partial commit | Full commit | Snapshot-copy proxy |
|---:|---:|---:|---:|---:|---:|
| 1 | 74.378042 | 1 (full) | 0.000750 | 0.000833 | 0.031750 |
| 2 | 126.326667 | 1 | 0.331166 | 0.001000 | 0.059666 |
| 4 | 134.122792 | 2 | 0.420208 | 0.001083 | 0.109417 |
| 8 | 162.727500 | 4 | 0.696042 | 0.001000 | 0.266459 |

The rejected design's proxy copies and retains W owned recurrent checkpoints
at one position; it measures copy/allocation cost, not a second implementation
of per-row snapshot execution. Copying is faster than recurrence replay on
this small-state synthetic fixture, but needs 1,646,592 bytes per checkpoint
versus 73,920 replay bytes per row. The chosen session at W=8 holds 3,293,184
bytes of recurrent states and 591,360 replay bytes, fixed across passes.
For the real dimensions the memory ratio is about 26.5x per retained row;
that bound motivates replay despite its measured extra CPU work. Full
acceptance avoids replay entirely. These timings do not predict real-model
speed or the optimal draft width, and no Metal timing is claimed.


## D17 — Captured main residuals feed the scalar draft proposer

Capture is opt-in on both main-model implementations. The loaded draft supplies
the ordered layer list and the maximum retained row count; shipping defaults
are zero-based layers `[5,19,33,47,61]` and the draft ring capacity. Capture
off preserves the existing path and digests. Capture only observes the main
layer result and changes neither logits nor committed numeric state.

The capture contract was checked directly against the unchanged upstream
sources:

| Contract | Verified source lines and result |
|---|---|
| Tap | `src/engine35.cpp:798-799` applies the MLP down residual projection; `830-833` copies `h_` immediately after each layer, and `1650` installs the five layers. Held: capture the post-MLP residual, not the normalized projection input. |
| Rounding | `src/q35_kernels.cu:1437-1444` writes `rbf(old + rbf(down))` to the residual array. Held: BF16 rounding before the add and again afterwards, expanded to f32; reference `residual_add` already uses this formula. |
| Slot order | `src/q35_bdraft.cu:249-252` gathers each capture slot into `[row,slot,channel]`; `490-499` reads the equivalent slot-major values directly for A8. Held: one row is `[slot,channel]`, five hidden-width vectors in capture-list order. |
| Prompt | `src/engine35.cpp:1248-1261` runs committed prefill, feeds its captures, then advances position. Held: prompt rows enter the ring as context. |
| Accepted rows | `src/specx.cpp:435-437` queues `n_commit+1` rows starting at the old frontier, reflecting upstream's +1 commit ABI. `src/engine35.cpp:1851` inserts context before proposing. Held: only committed input rows enter, once in ascending position. Rust's `commit(k)` already counts row zero, so it forwards k rows. |
| Terminal flush | `src/specx.cpp:464` inserts pending context after the loop. Held: final committed rows must reach the ring even without a subsequent proposal. |

Use half-open positions to avoid an off-by-one reading: before a proposal with
frontier input at position F, the live ring covers
`[max(0,F-ring_capacity),F)`. The token at F is selected but unconsumed. After
verification, feed the old frontier and accepted consumed successors, but no
rejected row and no new unconsumed frontier. Ring overwrite discards old
positions; a long prefill therefore retains only its last ring-capacity rows.
Committed captures and pending-pass captures are exposed separately. Successful
commits forward captured rows to the proposer; failed/discarded passes never
advance its ring. Synchronization before proposing checks the committed tail.

`DraftProposer` proposes the first K tokens of the existing eight-row block,
K=1..7. The block still computes all eight noncausal rows; this work does not
optimize it. The main model remains the sole selector/acceptance authority.
The separate proposer owns its draft ring. `snapshot_session(&model)` captures
both owners and `restore_session(&mut model, &checkpoint)` validates ownership
and restores the matching main state and ring. Reset invalidates that owner's
checkpoints; callers resetting the main model must also reset its proposer.
A main-model-only checkpoint cannot restore the separately owned drafter.

### Sampling and proposal policy boundary

`src/engine35.cpp:1845-1850` enables draft coupling only when main sampling is
on and version is 3; otherwise draft selection is greedy. The seed and
absolute prediction position feed token-keyed Gumbel noise at
`src/q35_bdraft.cu:401-404`. The draft uses its own temperature (850
thousandths by default), 16 candidates and fixed top-p 0.95
(`src/q35_bdraft.cu:404-430`), not the main request's temperature/top-p/top-k.
`DraftChoice` makes this caller choice explicit. The generate example uses
main version 3 and the same seed for `--seed`, otherwise greedy draft choice.

`CombinedProposer` chooses a fixed order (`NgramFirst` or `DraftFirst`) and
falls back only when the first proposal is empty. Draft context still advances
when n-gram wins. This does not port upstream's match-length gate, acceptance
EMA comparison, periodic neural probes, next-token-table extensions, HCAP or
8/16-row width adaptation (`src/specx.cpp:216-265,441-447`). Chapter 08 §5
records those upstream rules separately from this deliberately fixed policy.

`draft_acceptance` measures teacher-forced leading-prefix agreement with main
greedy selections for neural, n-gram and fixed n-gram-first combination. Missing
guesses count as unaccepted at later depths. Defaults are one window and two
positions because the real scalar draft may need about 90 seconds per proposal.
No released-model acceptance or speed claim follows from the synthetic checks;
real-model commands are documented in README for the orchestrator.


### Synthetic connection evidence and session fixes

The teacher is an actual two-layer main model with an all-zero output head
(greedy token 0), and an actual one-layer draft with BF16 correction weights
constructed to read its context-dependent hidden rows. Correct captures give
an accepted prefix of 7/7; one-row-lagged features and swapped capture slots
each give 0/7. The correct pair also accepts all seven guesses in one actual
`generate` verification pass. This deliberately constructed task detects a
wrong connection; it does not estimate released-model acceptance quality.
An independent two-layer residual calculation checks both BF16 boundaries,
and scripted acceptance counts check the exact ring contents before proposals
and after the terminal flush. End-to-end cases compare K=1/3/7 with K=0 for
greedy and two seeds, including joint restore and a reused prefix.

The carried D16 review findings are fixed in the engine: failure after an
attention layer truncates its appended suffix; partial-replay failure retains
the pending pass, matching the reference, and a later full retry replays from
committed recurrence. Restore validates every attention prefix before mutation.
A checkpoint after partial commit is checked against a fresh reference.
Temporarily deleting the provisional-failure truncation makes the new
later-layer-failure test fail at its following probe; the truncation is restored.
Metal cases explicitly skip when there is no device. No released model is
opened by these synthetic tests.
