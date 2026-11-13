# Metal kernel for the compressed linear operator

`yskn_engine::MetalEngine` (macOS only, `src/metal.rs` and `src/linear.metal`;
until D14 the separate `yskn-metal` probe crate). `MetalEngine::weighted_sums`
has the contract of `yskn_engine::Engine::weighted_sums`: row-major M×N i64,
padded physical rows included, prefix variant for head slices. `job` + `run`
encode any number of matrix calls into one command buffer.
`weighted_sums_shared` is what `Model` uses: several matrices reading one
input in one command buffer, with reused input/output buffers (D14).

All numbers below are from an Apple M5 (10-core GPU, 24 GB), on a synthetic
file written by `examples/synthetic_model.rs` (8 layers, hidden 5120,
ffn 17408, warp layout, 6-bit body, 8-bit 4096-row head: 51 matrices,
846,069,760 code bytes, 2.25 G weights). Another worker was using the CPU at
times. Unless stated otherwise, times are the median wall time of a full pass in mode (a),
which includes uploading inputs and reading back outputs.

## Design

- **Memory.** The read-only `Mmap` of the whole file is wrapped once with
  `newBufferWithBytesNoCopy` (shared storage, no deallocator; a `Matrix`
  clone keeps the mapping alive past the buffer). The length is rounded up
  to the page size (16 KB). Each dispatch binds it at the matrix's code-stream
  offset rounded down to 4 bytes. The device reports
  `maxBufferLength = 14,302,248,960`. The 10.3 GB model file fits. No-copy over a
  read-only mapping works for the synthetic file; the real file was not opened.
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
- **Activations.** The CPU widens them to i16 (`short2` per column pair,
  row-major M×K) on upload. All lanes of a SIMD group read the same address
  (broadcast).
- **Arithmetic.** Integer only (compile option `MTLMathMode::Safe` as well).
  Per pair: `short(w0*a0 + w1*a1)` is exact (|w| ≤ 127, |a| ≤ 128 → ≤ 32,512).
  It accumulates to i32 per sequence (≤ 4.2 M), then
  `long(scale) * long(dot)` in i64 on the GPU. The CPU rejects K/512 > 512,
  which makes i64 overflow impossible, so the reference's checked-overflow
  error can never be the expected result.
- **Specialisation.** Function constants: bits (4..8), activation rows per
  dispatch MC (1..16; M > 16 is split into dispatches of 16), layout,
  4-byte alignment of the stream (unaligned streams use byte loads).
  Pipelines are compiled lazily and cached.
- **Bounds.** Before dispatch `job` checks: the stream length equals
  P·S·bits·16, the stream lies inside the mapping, warp needs P·S % 32 = 0,
  M·max(K, N) < 2^32 and the stream is < 4 GB. Grid and loop bounds come only
  from these values, so a hostile file cannot move a GPU read outside the
  mapped buffer. Table indices are masked to 12 bits.

## Tried and rejected (same synthetic file; M=1 / M=16 unless noted)

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

## What limits it

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

## Not done / next

- Not run on the real model (by instruction). It has not been checked
  whether a 10 GB no-copy buffer is accepted, how wiring/residency behaves
  under the default GPU wired limit, or whether page-ins from a cold cache
  appear in the timing.
- Exact FP32 arithmetic (every partial sum < 2^24) or `simdgroup_matrix`
  could raise the M ≥ 4 throughput. The brief forbids floating point in
  the kernel, so this was not tried.
- Larger M (prefill) would reuse each decoded block across more rows. One
  thread per row × 32-row tiles caps reuse at MC = 16 per dispatch.
