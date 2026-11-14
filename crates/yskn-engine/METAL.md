# Metal kernel for the compressed linear operator

`yskn_engine::MetalEngine` (macOS only, `src/metal.rs` and `src/linear.metal`;
until D14 the separate `yskn-metal` probe crate). `MetalEngine::weighted_sums`
has the contract of `yskn_engine::Engine::weighted_sums`: row-major M×N i64,
padded physical rows included, prefix variant for head slices. `job` + `run`
encode any number of matrix calls into one command buffer.
`weighted_sums_shared` encodes several matrices reading one input into one
command buffer, with reused input/output buffers (D14). Since D22 `Model`
and the fast draft call `weighted_sums_split` instead: the same command
buffer for the first rows of each matrix, while the CPU kernel computes the
last share of the rows (see "CPU share" under Design).

All machine numbers are from an Apple M5 (10-core GPU, 24 GB). Text and
tables not marked D22 or "Round 2" describe the first round (D12, D14);
its numbers are on a synthetic file written by
`examples/synthetic_model.rs` (8 layers, hidden 5120,
ffn 17408, warp layout, 6-bit body, 8-bit 4096-row head: 51 matrices,
846,069,760 code bytes, 2.25 G weights). Another worker was using the CPU at
times. Unless stated otherwise, those times are the median wall time of a full pass in mode (a),
which includes uploading inputs and reading back outputs. The round-two
sections ("Round 2") also report the real model.

## Design

- **Memory.** The read-only `Mmap` of the whole file is wrapped once with
  `newBufferWithBytesNoCopy` (shared storage, no deallocator; a `Matrix`
  clone keeps the mapping alive past the buffer). The length is rounded up
  to the page size (16 KB). Each dispatch binds it at the matrix's code-stream
  offset rounded down to 4 bytes. The device reports
  `maxBufferLength = 14,302,248,960`. The 10.3 GB model file fits. The first
  round tried no-copy over a read-only mapping on the synthetic file only;
  the real file is accepted the same way (D12, D14), and round two (D22)
  measured with it.
- **Threads.** One SIMD lane computes one output row: a threadgroup is 32
  consecutive rows × `splits` SIMD groups (default 4 → 128 threads). SIMD
  group `k` takes sequences `k, k+splits, …` of those rows. With the `warp`
  layout, the 32 lanes then read 32 neighbouring 16-byte pieces. Partial
  i64 sums are added in threadgroup memory at the end. This is exact because
  `scale·(d0+d1) = scale·d0 + scale·d1` in integers.
- **Table.** At threadgroup start the 8 KB signed-byte table is copied into
  threadgroup memory, widened to 4,096 `short2` (16 KB). Each entry is
  written by one thread: the stride is the whole 32 × `splits` threadgroup
  (`uint2 [[threads_per_threadgroup]]`; a scalar declaration is only the
  width, 32). The same memory is reused for the final reduction. The host
  caches one table buffer per (mapping, lookup-table offset), because
  tensors may share a code stream but not a table.
- **Decode.** 32 steps of `bits` bits are exactly `bits` 32-bit words. So
  each of a sequence's four 32-step blocks has the same shift pattern. The
  host generates `DECODE_BLOCK4..8` (32 table lookups with literal shifts,
  from FORMAT.md §4.2) and prepends them to the MSL source. Per block, a
  thread loads `bits` words plus the carried previous word. It decodes 32
  `short2` weight pairs into registers. Then it runs a compact loop over the
  M activation rows.
- **Activations.** The CPU widens them to i16 (binary16 in float mode;
  `short2`/`half2` per column pair, row-major M×K) on upload. All lanes of a SIMD group read the same address
  (broadcast).
- **Arithmetic.** Two exact multiply-accumulate modes (function constant
  `FLOAT_MAC`, default float since D22; `YSKN_METAL_MAC=float|integer` is
  read only by `Model::load`, `FastDraft::load` and the benchmarks, explicit
  constructors use float), both compiled with `MTLMathMode::Safe`:
  - *integer*: per pair `short(w0*a0 + w1*a1)` is exact (|w| ≤ 127,
    |a| ≤ 128 → ≤ 32,512), accumulated to i32 per sequence (≤ 4.2 M);
  - *float* (D22): the table is widened to `half2` and the activations are
    uploaded as binary16 (every integer of magnitude ≤ 2048 is exact in
    binary16, so both conversions are exact). Each pair runs two binary32
    `fma(float(w), float(a), part)`. Every operand is an integer with
    |w|, |a| ≤ 128, so every product is an integer ≤ 2^14 and every partial
    sum of one 256-weight sequence, in whatever order it is formed, is an
    integer of magnitude ≤ 256 · 2^14 = 2^22 < 2^24. All such integers are
    binary32 values, and an addition, multiplication or fused multiply-add
    whose exact result is representable returns that result. So every step
    is exact, independently of evaluation order and fusion, and
    `int(dot)` converts the exact sequence sum before any sum could exceed
    the bound.

  Then `long(scale) * long(dot)` in i64 on the GPU. The CPU rejects
  K/512 > 512, which makes i64 overflow impossible, so the reference's
  checked-overflow error can never be the expected result.
- **Specialisation.** Function constants: bits (4..8), activation rows per
  dispatch MC (1..16; M > 16 is split into dispatches of 16), layout,
  4-byte alignment of the stream (unaligned streams use byte loads) and,
  since D22, the multiply-accumulate mode.
  Pipelines are compiled lazily and cached.
- **CPU share (D22).** `weighted_sums_split` gives the CPU kernel the last
  share(M)·N rows of each matrix, rounded down to the 32-row tile and leaving
  the GPU at least one tile (no CPU rows below N = 64), from a constant
  table or `YSKN_METAL_CPU_SHARE` (read like `YSKN_METAL_MAC`). It commits
  the GPU part, runs the CPU part, waits and interleaves the rows. Every
  commit returns a guard that waits for completion when dropped, so an
  error or a panic in the CPU part cannot leave GPU work reading the reused
  scratch buffers. The split always covers every physical row; head slices
  (`weighted_sums_prefix`) run on the GPU alone.
- **Bounds.** Before dispatch `job` checks: the stream length equals
  P·S·bits·16, the stream lies inside the mapping, warp needs P·S % 32 = 0,
  M·max(K, N) < 2^32 and the stream is < 4 GB. Grid and loop bounds come only
  from these values, so a hostile file cannot move a GPU read outside the
  mapped buffer. Table indices are masked to 12 bits.

## Tried and rejected

### Round 1 (synthetic file; M=1 / M=16 unless noted)

| Variant | M=1 ms | M=16 ms | Note |
|---|---|---|---|
| v1: 128-step loop decode with `w[lo]` indexing, i32 MAC, i8 activations, splits 8 | 35.6 | 118.4 | first exact version |
| v1 with splits 1 / 2 / 4 / 16 / 32 | 54.4 / 37.0 / 35.6 / 40.2 / 40.7 | 151.6 / 126.0 / 119.3 / 124.0 / 131.9 | |
| v2: v1 + 16-bit pair products | 36.7 | 109.9 | |
| v2 without table lookup (diagnostic, wrong results) | 35.7 | 104.6 | table lookups are cheap |
| v2 without activation loads (diagnostic) | 33.9 | 77.4 | |
| v2 without both / also without code loads (diagnostic) | 30.9 / 31.6 | 76.0 / 74.8 | decode arithmetic was the M=1 cost |
| code streaming only, no decode (diagnostic) | 10.3 (GPU 9.1) | — | ≈ 90–98 GB/s read floor for this access pattern |
| v3: fully unrolled 128-step decode, MAC inline | 30.3 | 692 (M=4: 149) | code/register blow-up for MC > 1 |
| v4: v3 with `constant` address-space activations | 28.4 | 615 | rejected with v3 |
| v2 with `constant` activations | 36.8 | 112.7 | no gain |
| v5: v2 + activations widened to i16 on the CPU | 34.6 | 71.4 (M=4: 42.9) | kept |
| v6: v5 + unrolled decode only for MC ≤ 1/2/4 | 10.3 | M=2: 33–37 | M=1 only; superseded |
| **v7: 32-step block decode (final)** | **10.9** | **56.8** (M=2/4/8: 12.6/18.1/31.2) | kept |
| v7 without activation loads (diagnostic) | 10.2 | 54.7 | loads ≈ 3–4 % |
| v7 with `short4` activation loads | 10.8 | 57.0 | no gain |
| v7 with i32 MAC / `mad24` | 10.8 / 10.7 | 58.9 / 58.9 | no gain |
| v7 splits 4 vs 8, two runs each | 10.30 vs 10.71 | 56.5 vs 56.1 | 4 is 3–4 % faster for M ≤ 8 |

The CPU kernel (`bench_linear --kernel dotprod --threads 10`) on the same file
took 54.9 / 59.2 / 64.1 / 76.3 / 100.7 ms for M = 1 / 2 / 4 / 8 / 16.

### Round 2 (D22; synthetic file GPU time at M=16 unless noted)

| Variant | Result | Note |
|---|---|---|
| float MAC, `half2` table and activations (kept) | 41.8–44.0 ms (integer 54.1–55.2) | exact, see Arithmetic |
| float MAC with binary32 activations in memory | 63.7–65.8 | slower |
| same, weights also converted to binary32 once per block | 62.2–65.8 | 32 more registers per lane |
| float MAC with `half4` activation loads | 42.7–43.1 | no gain |
| float MAC with two accumulators per row | 41.6–43.9 | no gain |
| 32 activation rows per dispatch (one lane per output row) | M=16 51.2–51.6, M=32 102.0–104.4 | same session with 16: 46.0–46.9 / 94.0–94.6; register pressure |
| `simdgroup_matrix` (8x8 binary16 x binary16 + binary32) | not built | the MSL specification states no accuracy for it, so exactness cannot be proven |
| waiting by polling, shared events, pre-committed buffers, unretained references | about 0.11 ms per round trip each | D22 §4: latency is in the GPU path |
| polling wait in the engine (real model, one row) | 144–250 ms vs 144–146 blocking | dropped |
| CPU share above the optimum (real model) | M=1 0.18: 153–158 ms vs 134 at 0.12 | steep: the GPU waits for the CPU |

## What limits it

### Round 1 (synthetic file)

- **M = 1–2** is close to memory streaming. The kernel takes 10.3 ms, of which
  9.9 ms is GPU time. A read-only kernel with the same access pattern takes
  8.6–9.4 ms (90–98 GB/s). The remaining gap is the decode.
- **M ≥ 4** is limited by integer ALU work. Time grows linearly at about
  3.1 ms per extra activation row (36 G int8 MACs in about 55 ms of GPU time
  at M=16). Removing activation loads (−3 %), widening the multiply to i32
  or using `mad24` did not help. Apple GPUs have no int8 dot or matrix
  instruction.
- **Submission.** An empty commit-and-wait round trip takes about 0.2 ms.
  One command buffer per matrix (mode b) added about 0.175 ms per matrix on
  this file. One command buffer per layer (mode c) added about 0.17 ms per
  command buffer.

### Round 2 (real model, D22)

- **M = 1** in one command buffer: 98.9 ms, 98.7 GB/s of code, 83–85 % of
  the GPU's measured streaming rate (116–119 GB/s, `bench_bandwidth`).
  Inside the engine the 257 round trips add about 0.14 ms each (about
  36 ms per pass: 85 µs from commit to GPU start, 61 µs from GPU end to
  wake-up), which no waiting method reduced and which cannot be merged
  because each group needs a CPU floating-point result. The CPU share hides
  part of it; the CPU kernel itself is decode-bound (13 GB/s, a tenth of
  the CPU's streaming rate).
- **M ≥ 4** is binary32 multiply-add throughput: 26.2 ms per added row,
  25.6 G multiply-adds in 26.2 ms ≈ 0.98 T/s.
- **CPU share** is bounded by CPU speed and by other load on the CPU: a
  slower CPU than calibrated makes the GPU wait (at load 22–24 the share
  0.08 took 197 ms at one row against 145 ms GPU-only;
  `YSKN_METAL_CPU_SHARE=off` turns it off).

## Not done / next

- The 10 GB no-copy buffer is accepted on the real model (D12, D14).
  Residency under the default GPU wired limit and cold-cache page-ins
  were not studied.
- Exact binary32 is done (D22). `simdgroup_matrix` or the M5's Metal 4
  tensor operations could raise the M ≥ 4 throughput; an integer tensor
  operation (int8 inputs, int32 accumulation) would be exact by
  construction, a binary16/binary32 one needs a documented accuracy.
  Neither was tried.
- Larger M (prefill) would reuse each decoded block across more rows. One
  thread per row caps it at MC = 16 (32 was slower, register pressure); a
  tile that shares a decoded block between lanes through threadgroup
  memory would be needed.
- At M > 1 the CPU stages of one half of the rows could overlap the GPU
  projections of the other half (rows only depend on earlier rows); not
  tried, it restructures `Model::forward`.
- `CPU_SHARE` was calibrated on a shared machine; a quiet machine should
  allow larger shares (`bench_metal --shares`).

## Round 2 (brief 07d, D22): real model, M5

Real model (`$HOME/models/qwen38-speed/qwen38_27b_speed.yskn`: 401 matrices,
9,758,310,400 code bytes per pass), shared machine, `uptime` load given
with each run. Commands and full tables are in D22.

### Kept

- **Exact binary32 multiply-accumulate** (`YSKN_METAL_MAC=float`, default;
  proof under Design / Arithmetic). Kernel alone, ms per pass, minimum of
  five, integer → float: M=1 100.7 → 98.9, M=4 181.3 → 153.8, M=16
  638.2 → 492.0. The cost per added row falls from 35.8 to 26.2 ms.
- **CPU share of the rows** (`YSKN_METAL_CPU_SHARE`, `weighted_sums_split`):
  the ten CPU workers compute the last share(M)·N rows of each matrix of a
  group while the GPU computes the rest. Engine-like grouping (257 calls),
  GPU only → calibrated share: M=1 144.5 → about 136, M=4 194 → about 184,
  M=16 554 → about 490 ms.

### Memory bandwidth ceiling (`bench_bandwidth`)

`cargo run --release -p yskn-engine --example bench_bandwidth -- --gb 4 --reps 5`
(4 GiB shared buffer, read-only, minimum of five, GB/s = 10^9 B/s), two
runs at load 6.0–6.7:

| Reader | Run 1 | Run 2 |
|---|---:|---:|
| CPU, 1 thread | 73.6 | 74.2 |
| CPU, 4 threads | 109.5 | 109.9 |
| CPU, 10 threads | 133.8 | 119.6 |
| GPU, kernel-like 512-byte SIMD loads | 116.6 | 119.0 |
| concurrent: GPU on one half | 71.8 | 73.0 |
| concurrent: CPU 10 threads on the other half | 77.0 | 78.2 |
| concurrent sum (each side over its own time) | 148.8 | 151.2 |
| concurrent, all bytes over the longer side's time | 143.5 | 146.1 |

A first run a minute earlier (load 6.5) gave 72.5 / 101.7 / 107.3 CPU,
112.2 GPU and a concurrent sum of 161.0. So the measured ceiling of this
machine is about 145–150 GB/s for CPU and GPU together, about 115–119
GB/s for the GPU alone and 107–134 GB/s for the CPU alone (the nominal
153 GB/s was not reached by a single reader). The kernel at one row reads
the 9.76 GB of code in 98.9 ms in one command buffer, 98.7 GB/s: 83–85 %
of the GPU's own streaming rate and about two thirds of the combined
ceiling. Inside the engine (257 command buffers) the GPU-only kernel takes
about 144 ms (68 GB/s); the difference is the per-buffer round trip, not
bandwidth. The CPU kernel alone reads 13.3 GB/s at one row, about a tenth
of the CPU's streaming rate: it is limited by decode work, not memory, so
the CPU share is bounded by CPU speed and by contention, not by DRAM.
