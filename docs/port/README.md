# Rust scalar reference

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
`arm64` before building. Both crates use edition 2024 and forbid unsafe
code. No CUDA toolchain, GPU, or published weights are needed for the tests.
All pre-existing upstream files remain unchanged.

| Crate/module | Source contract |
|---|---|
| `yskn-format` | Container layout, signs, vector formats and trellis states in FORMAT.md |
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
their compressed codes. There is no memory mapping, dense model-weight
expansion, threading or architecture-specific code. Execution is deliberately
slow and is not intended as a competitive inference backend.

## Quality command

```sh
cargo run --release -p yskn-ref --example score_windows -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json
```

The folder's optional `yasakani.json` supplies `files.weights`; otherwise
the documented `qwen38_27b_e3x.yskn` default is used. The input JSON must have
the committed window format: 2,048 token IDs per window and scored positions
1024 through 2046. Input row p predicts token p+1. The command prints scored
position count, mean negative log-likelihood and perplexity, and reports
per-window progress on stderr.

Full attention is executable under D7–D9. No released model was loaded for
this validation: neither a published quality number nor CUDA mixed-model
bit equivalence has been reproduced.

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
