# Rust reference and mapped CPU/Metal engine

This workspace implements the documented container and the executable
mixed Gated DeltaNet/full-attention main-model path. The three historical
gaps in [SPEC_GAPS.md](SPEC_GAPS.md) are resolved by D7–D9 in
[DECISIONS.md](DECISIONS.md).

## Build and check

Install Rust stable, then run at the repository root:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace --release
```

The GitHub Actions workflow uses `macos-15` and verifies `uname -m` is
`arm64` before building. All three crates use edition 2024. `yskn-format`
and `yskn-ref` forbid unsafe code; `yskn-engine` isolates mapping, CPU intrinsics, scoped pool
borrows and Metal buffer access behind documented safety invariants. The engine requires Rust 1.98.
No CUDA toolchain, published weights or GPU are needed for the tests; with a
Metal device (CI's macOS runner has one) the Metal tests run, without one they
print `skipped: no Metal device`. The Metal module is macOS-only; other
targets build without it. All pre-existing upstream files remain unchanged.

| Crate/module | Source contract |
|---|---|
| `yskn-format` | Container layout, signs, vector formats and trellis states in FORMAT.md |
| `yskn-engine` | Mapped code streams, persistent CPU workers, the Metal kernel (macOS) and layer-major main-model inference under D10–D12/D14 |
| `yskn-ref::model` | Chapter 01, loading, layer order and sequence state |
| `compressed_linear` | Chapter 02, integer compressed projection and coded embedding |
| `gated_deltanet` | Chapter 03, dense gates, convolution and recurrence |
| `attention` | Chapter 04, Int8 cache and per-row causal attention under D7–D9 |
| `residual_norm_mlp` | Chapter 05, residual rounding, deferred norm and MLP |
| `output_head_sampler` | Chapter 06, logits, greedy, sampler versions 2/3 and NLL |
| `portable_math` | Permitted `pm_math.cuh` algorithms, exact scalar square root and D7 exp2 |

## Model API

```rust,no_run
use yskn_ref::Model;

fn run() -> yskn_ref::Result<()> {
    let mut model = Model::load("model.yskn")?;
    model.prefill(&[3, 17, 9])?;

    model.run_provisional(&[7, 11, 5])?;
    let candidate_logits = model.row_logits();
    assert_eq!(candidate_logits.len(), 3);
    assert_eq!(model.position(), 3);

    model.commit(2)?; // Accept exactly two input rows, including row zero.
    assert_eq!(model.position(), 5);
    let next_token_logits = model.logits().expect("accepted row");
    let token = yskn_ref::output_head_sampler::greedy(next_token_logits)?;
    println!("{token}");
    Ok(())
}
```

The API loads recurrent-only and mixed models with `model.arch=qwen3_5`
and `head_dim=256`, using the fixed rotary width 64 under D8.
`new_session()` shares immutable model weights
and starts an independent empty sequence. A pending provisional pass must
be committed or discarded before another pass. Commit replay is intentional:
this implementation prioritizes one readable scalar operation order.

Container payloads are loaded into ordinary vectors, while matrices retain
their compressed codes. The reference uses no memory mapping, dense
model-weight expansion, threading or architecture-specific code. Execution is deliberately
slow and is not intended as a competitive inference backend.

## Mapped engine API and commands

`yskn_engine::Model` runs the entire main model in layer-major passes, using
mapped weights and the reference's scalar operation order. `load` accepts a
model folder or a `.yskn` path and defaults to 16 rows per pass. Its
compressed projections run on the backend named by `YSKN_BACKEND` (`cpu` or
`metal`); unset, it uses Metal when the process has a Metal device and the
CPU kernel otherwise. Explicit configuration uses
`Model::with_backend(path, Engine::new(kernel, threads)?, Backend::Metal,
rows_per_pass)`; `Model::with_engine(path, engine, rows_per_pass)` is the CPU
backend. Explicit constructors never read the environment, and Metal without
a device is an error, not a fallback. The CPU pool runs all other work on
either backend; both backends give identical bits (D14). The constructors
are unsafe solely because callers must keep the backing weights file
unchanged and untruncated until model drop. The engine offers `prefill`,
`row_logits`, `logits`, `position`, `reset`, `context_capacity`,
`dimensions`, `rows_per_pass`, `backend`, `device` and measured
`pass_timings`. It has no provisional, commit, draft or speculation API.
See D11 for the causal attention and parallelism contract and D14 for the
backends and the parallel recurrent heads.

```sh
# Six configurations (four CPU, Metal with 16 and 1 rows per pass), first 24
# tokens of window 0, all vocabulary logit bits.
cargo run --release -p yskn-engine --example verify_model -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json

# Scalar-reference cross-check of the first ROWS rows (2 or 3; about two
# minutes per row on the real model, heap-loaded weights; run alone), then the
# same six configurations plus a comparison with that file.
cargo run --release -p yskn-ref --example dump_logits -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json 3 reference-3.json
cargo run --release -p yskn-engine --example verify_model -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json --reference reference-3.json

# One window, with bounded logit memory and measured pass-stage timings.
# The first stderr line names the backend and device.
/usr/bin/time -l cargo run --release -p yskn-engine --example score_windows -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json \
  --backend metal --rows-per-pass 16 --windows 0..1 --json window-0.json

# Whole-model pass timings (seven 16-row passes after a warm-up), for example
# on a synthetic file from `--example synthetic_model -- OUT.yskn 8`.
cargo run --release -p yskn-engine --example bench_model -- \
  MODEL.yskn --backend cpu --rows-per-pass 16 --passes 7

# Full quality command (long running; run only when authorized by the gate).
cargo run --release -p yskn-engine --example score_windows -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json --json all-windows.json
```

`--windows A..B` selects zero-based window indices, end excluded. Each window
starts at reset. JSON contains per-window NLL sums/counts, elapsed seconds,
and aggregate metrics; sum counts and NLLs to combine disjoint partial runs
in ascending window order, with possible last-bit binary64 rounding differences
from the full per-position fold. Progress and timing lines go to stderr;
stdout retains the three scalar-reference summary lines. `--backend` overrides
`YSKN_BACKEND`. `YSKN_KERNEL` selects the CPU kernel `portable`, `neon` or
`dotprod`; omission uses the fastest supported variant. The Metal kernel's
own benchmark is `cargo run --release -p yskn-engine --example bench_metal --
MODEL.yskn` (design notes in `crates/yskn-engine/METAL.md`). Real model commands must run serially with no concurrent model
mapping, and no build/test during a timed run.

## Quality command

```sh
cargo run --release -p yskn-ref --example score_windows -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json
```

The folder's optional `yasakani.json` supplies `files.weights`; otherwise
the documented `qwen38_27b_e3x.yskn` default is used. `Model::load_folder` (and
so this command) also accepts a plain `.yskn` file path in place of the folder;
it then uses the default 262,144-token context capacity. The input JSON must have
the committed window format: 2,048 token IDs per window and scored positions
1024 through 2046. Input row p predicts token p+1. The command prints scored
position count, mean negative log-likelihood and perplexity, and reports
per-window progress on stderr.

Full attention is executable under D7–D9. The original scalar-reference
validation did not load released weights.
Neither a published full-set quality number nor CUDA mixed-model bit
equivalence has been established. D11 records the mapped-engine contract;
its real-model checks compare CPU configurations, without loading the
heap-copying scalar model.

The D11 local gate passed the six proof commands specified in Brief 02b
(76 debug and 76 release tests), including all four real-model verification
configurations. Window 0 scored
1,023 positions with NLL sum `508.71858607572784` and perplexity
`1.644244684328`. The measured run took 429.06 s for scoring and peaked at
10.57 GB resident including mappings. `/usr/bin/time -l` could not read
`kern.clockrate` in the sandbox; RSS came from Darwin `wait4` on a repeat
run. See D11 for stage timings and that measurement boundary. All twenty
windows are estimated at about 2 h 23 min; their complete quality gate is
still pending.

## Verification evidence

The published tiny container test checks every declared SHA-256 field exactly:
file bytes, sign streams, states, rotated-coordinate values and vector bits.
It additionally checks original-coordinate samples, matrix norms, and the
engine-table error bound. FORMAT.md specifies the transform mathematically,
so its final reconstructed samples/norms use a small numeric tolerance;
the executable compressed operator instead uses chapter 02's fixed order.

The synthetic writer uses seeded streams, binary16 tables/scales and BF16
vectors. Recurrent models exercise complete embedding-to-logit execution,
two layer transitions, row/chunk/commit invariants and both samplers. A second
model changes hidden and FFN widths to 1,024 and vocabulary to 65 with padded
head rows. The four-layer mixed fixture is generated, decoded and executed by its
full-model invariant test. A 73-row mixed sequence starting at position 3
checks all logit bits across provisional, single-row and uneven-chunk
execution, including subsequent committed state. Attention unit tests
verify that masked cache contents cannot affect outputs and that masked
value operations are skipped.

The portable-math port was also checked with a temporary scalar C++ harness
including only the unchanged permitted `pm_math.cuh`, with contraction
disabled and explicit basic arithmetic primitives. All compared output bits
matched across 100,049 seeded normal/subnormal/threshold/range inputs:
40,042 FP32 cases, 40,007 FP64 cases and 20,000 sin/cos angles. The
software square root independently matched a 100-digit decimal oracle on
10,000 positive finite binary32 patterns. Representative source-derived bit
vectors are permanent unit tests. This evidence checks the portable source
port, not a CUDA binary or a released model.
