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
| `attention` | Chapter 04, Int8/Int8Profile/Kv5 caches and per-row causal attention under D7–D9/D13 |
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

`Model::load_folder_with(folder, KvMode::Int8Profile)` and
`Model::load_folder_with(folder, KvMode::Kv5)` select the profile modes.
`load_folder` (and the other existing loaders) still defaults to plain
`KvMode::Int8`. Profile modes load `files.kv_profile` from the manifest,
defaulting to `kvq_u5.bin`, and reject missing/malformed profiles or model
attention-count/KV-head mismatches. Both weights and profile paths must stay
inside the folder, including symlink targets. Plain Int8 needs no profile.
`from_container_with_kv` accepts an explicit `Arc<KvProfile>` for in-memory
or direct-file callers. See D13 for packed score/mean/rounding rules.

## Mapped engine API and commands

`yskn_engine::Model` runs the entire main model in layer-major passes, using
mapped weights and the reference's scalar operation order. `load` accepts a
model folder or a `.yskn` path and defaults to 16 rows per pass. Its
compressed projections run on the backend named by `YSKN_BACKEND` (`cpu` or
`metal`); unset, it uses Metal when the process has a Metal device and the
CPU kernel otherwise. `YSKN_KV=int8|int8-profile|kv5` selects its KV cache
mode (default `int8`). Explicit configuration uses
`Model::with_backend(path, Engine::new(kernel, threads)?, Backend::Metal,
rows_per_pass)`; `Model::with_engine(path, engine, rows_per_pass)` is the CPU
backend. Both retain the `Int8` default. To select any mode with either
backend/pool/pass width, use
`Model::with_options(path, engine, backend, rows_per_pass, KvMode::Kv5)`.
`KvMode` is re-exported from `yskn_engine`; `kv_mode()` reports the selection.
Explicit constructors never read the environment, and Metal without
a device is an error, not a fallback. The CPU pool runs all other work on
either backend; both backends give identical bits (D14). The constructors
are unsafe solely because callers must keep the backing weights file
unchanged and untruncated until model drop. The engine offers `prefill`,
`row_logits`, `logits`, `position`, `reset`, `context_capacity`,
`dimensions`, `rows_per_pass`, `kv_mode`, `backend`, `device` and measured
`pass_timings`. It has no provisional, commit, draft or speculation API.
See D11 for the causal attention and parallelism contract and D14 for the
backends and the parallel recurrent heads.

```sh
# Six configurations (four CPU, Metal with 16 and 1 rows per pass), first 24
# tokens of window 0, all vocabulary logit bits, in each KV mode. Run serially.
export CARGO_NET_OFFLINE=true
MODEL_FOLDER=/path/to/model-folder
for kv in int8 int8-profile kv5; do
  cargo run --release -p yskn-engine --example verify_model -- \
    "$MODEL_FOLDER" tools/quality/windows_own_thinking.json --kv "$kv" || exit
done

# Scalar-reference cross-check of the first ROWS rows (2 or 3; about two
# minutes per row on the real model, heap-loaded weights; run alone), then the
# same six configurations plus a comparison with that file.
cargo run --release -p yskn-ref --example dump_logits -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json 3 reference-3-kv5.json --kv kv5
cargo run --release -p yskn-engine --example verify_model -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json --kv kv5 --reference reference-3-kv5.json

# One window, with bounded logit memory and measured pass-stage timings.
# The first stderr line names the backend, device and KV mode.
/usr/bin/time -l cargo run --release -p yskn-engine --example score_windows -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json \
  --kv kv5 --backend metal --rows-per-pass 16 --windows 0..1 --json window-0.json

# Whole-model pass timings (seven 16-row passes after a warm-up), for example
# on a synthetic file from `--example synthetic_model -- OUT.yskn 8`.
cargo run --release -p yskn-engine --example bench_model -- \
  MODEL.yskn --kv int8 --backend cpu --rows-per-pass 16 --passes 7

# All 20 windows on Metal in both profile modes (long running;
# run only when authorized by the gate, serially after verification).
for kv in int8-profile kv5; do
  cargo run --release -p yskn-engine --example score_windows -- \
    "$MODEL_FOLDER" tools/quality/windows_own_thinking.json \
    --kv "$kv" --backend metal --rows-per-pass 16 --windows 0..20 \
    --json "all-windows-$kv.json" || exit
done
```

`--windows A..B` selects zero-based window indices, end excluded. Each window
starts at reset. The engine's `--json OUT` writes this object (the range end
is excluded; the model file name contains no directory path):

```json
{
  "kv_mode": "kv5",
  "backend": "metal",
  "device": "Apple M4",
  "kernel": "dotprod",
  "threads": 4,
  "rows_per_pass": 16,
  "window_range": { "start": 0, "end": 1 },
  "model_file": { "name": "qwen38_27b_e3x.yskn", "size_bytes": 123456789 },
  "crate_version": "0.1.0",
  "score": {
    "windows": [
      { "window": 0, "scored_positions": 1023,
        "negative_log_likelihood_sum": 508.7, "elapsed_seconds": 100.0 }
    ],
    "scored_positions": 1023,
    "negative_log_likelihood_sum": 508.7,
    "mean_negative_log_likelihood": 0.49726295210166177,
    "perplexity": 1.644214850338663,
    "elapsed_seconds": 100.0
  }
}
```

Values above illustrate the layout, not a measured run; `windows` has one entry
per selected window. `kernel` and `threads` describe the CPU pool, also used for
non-projection work on Metal. The existing per-window NLL sums/counts, elapsed
seconds and aggregate metrics are unchanged inside `score`. Sum counts and
NLLs to combine disjoint partial runs
in ascending window order, with possible last-bit binary64 rounding differences
from the full per-position fold. Progress and timing lines go to stderr;
stdout retains the three scalar-reference summary lines. `--backend` overrides
`YSKN_BACKEND`. All three engine examples and reference `dump_logits` accept
`--kv int8|int8-profile|kv5` (default `int8`, independent of `YSKN_KV`);
`dump_logits` takes the option after its four positional arguments. Its JSON is
`{"kv_mode":"kv5","tokens":[...],"logits_bits":[[...],...]}` and stderr reports
`kv_mode` too. `verify_model --reference` checks the dump's mode against `--kv`
before loading the model or comparing logits; a legacy dump without `kv_mode`
is accepted only with `--kv int8`. Profile
modes require a model folder and use the same manifest/profile validation as
the scalar loader. Direct `.yskn` paths support plain Int8.
The `kv_encode` timing is the complete attention row preparation stage,
including normalization, RoPE, gating and KV encoding, not codec-only time.
`attention` covers cache append and attention output; `kernel`, `kv_encode`,
`attention`, `recurrence` and `rest` do not overlap.
`YSKN_KERNEL` selects the CPU kernel `portable`, `neon` or
`dotprod`; omission uses the fastest supported variant. The Metal kernel's
own benchmark is `cargo run --release -p yskn-engine --example bench_metal --
MODEL.yskn` (design notes in `crates/yskn-engine/METAL.md`). Real model commands must run serially with no concurrent model
mapping, and no build/test during a timed run.

## Quality command

```sh
cargo run --release -p yskn-ref --example score_windows -- \
  /path/to/model-folder tools/quality/windows_own_thinking.json --kv kv5
```

`--kv int8|int8-profile|kv5` defaults to `int8`. The upstream published
“int8” quality configuration corresponds to `int8-profile`, not plain
`int8`. Shared `quality::parse_kv_mode`/`score_with_kv` and
`model::model_location_with_kv`/`load_weights_with_kv` are also used by the
mapped engine to select the same mode without changing scalar arithmetic.
The score example passes the selected mode to its loader through the shared
scoring helper, preserving the ordered NLL fold. Its optional `--json OUT` writes
`{"kv_mode":"kv5","score":{...}}`, using the same mode spelling and unchanged
`score` fields as the engine example above.

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
value operations are skipped. Profile-mode tests add a mixed GDN/two-attention
model with packable and fallback sides, means, a guard and a basis group.
Mapped-engine profile-mode tests reuse this synthetic KVQP writer and compare
all logit bits with the scalar reference for 73 rows starting at position 3,
pass widths 1, 5, 16 and 17, uneven chunks, reset and continuation on CPU
(1 and 4 threads) and Metal when available (otherwise explicitly skipped).
All three reference modes have frozen logit digests and distinct outputs; profile modes
check the 73-row position-3 invariance, reset, continuation and failure
atomicity. Hand-derived attention cases check packed means, key-mean omission,
BF16 base/weight boundaries and unread masked slots. No released weights or
full quality scoring were run for this integration.

The portable-math port was also checked with a temporary scalar C++ harness
including only the unchanged permitted `pm_math.cuh`, with contraction
disabled and explicit basic arithmetic primitives. All compared output bits
matched across 100,049 seeded normal/subnormal/threshold/range inputs:
40,042 FP32 cases, 40,007 FP64 cases and 20,000 sin/cos angles. The
software square root independently matched a 100-digit decimal oracle on
10,000 positive finite binary32 patterns. Representative source-derived bit
vectors are permanent unit tests. This evidence checks the portable source
port, not a CUDA binary or a released model.

## Rust text I/O (`yskn-text`)

`ChatFormatter::load(folder)` reads only the tokenizer, configuration and chat
Jinja template. `render(&messages, tools.as_ref(), &options)` accepts JSON
values; `prompt` returns `(Vec<u32>, starts_inside_thinking)` and always adds
the generation prompt, as upstream does. `render` supports
`add_generation_prompt: false`. `encode` adds no special tokens; `decode`
keeps them and silently skips unmapped ids. Token ids, including the ten digit ids,
are available to the future sampler and chat layers. This crate forbids unsafe
code and has no Python runtime dependency.

JSON numbers have limits relative to Python: JSON `-0` becomes a float and
prints `-0.0`, while Python parses it as integer `0` and prints `0`. Integers
beyond `u64` (or below `i64`) become floats and can lose precision, while Python
keeps arbitrary-precision integers. `NaN` and `Infinity` are rejected as JSON
input, while Python's `json.loads` accepts them and `json.dumps` emits `NaN`
and `Infinity` by default.

The environment preserves JSON key order. Its `trim` filter matches Python's
whitespace set; `.strip()`, `.lstrip()` and `.rstrip()` come from
`minijinja-contrib` and use Rust's whitespace set, which differs for U+001C to
U+001F. The model's template uses only `|trim`. A Python-format `tojson` filter
is also provided (including indentation, separators, ASCII escaping and
sorted-key options). Floats joined with `~` or printed inside lists use Rust's
float format; this only affects an error message if a caller passes a
non-string `reasoning_effort`. Template errors expose the message passed
to `raise_exception`. Nonempty string tool arguments are rejected by upstream's
`ChatFormatter` template; request-level normalization belongs to the caller
(upstream does that in `chat.py`). `strftime_now` is intentionally absent:
Rust's standard library provides neither local calendar conversion nor
`strftime`, and this model's template does not use it.

`StreamDecoder` holds incomplete UTF-8 in a small re-decoding window and
releases any final tail on `flush`, including a replacement character for a
truncated final sequence, matching full decode. It retains only the window,
not the entire generation. `OutputStream::feed` accepts the growing generation
array, hides stop ids, splits reasoning at `</think>`, records its position,
and strips initial answer newlines. `ToolCallFilter` incrementally separates
content and tool calls, with schema-aware parameter conversion, JSON fallback,
and lossless fallback for malformed or unfinished calls. These port all public
text helpers in `python/yasakani/textio.py`; NumPy array types are replaced by
Rust vectors and arrays. Request validation and generation remain outside this
crate.

Regenerate independent oracle vectors from the worktree root. Set `$PYTHON` to
a Python with `tokenizers`, `jinja2` and `numpy`; these are needed only to
regenerate goldens, never at run time:

```sh
PYTHONPATH=python "$PYTHON" \
  tools/text/make_goldens.py "$HOME/models/qwen38-speed" \
  crates/yskn-text/tests/data/goldens.json
CARGO_NET_OFFLINE=true cargo test -p yskn-text
CARGO_NET_OFFLINE=true YSKN_MODEL_DIR="$HOME/models/qwen38-speed" \
  cargo test -p yskn-text --release -- --nocapture
```

The generator never loads weights or copies model files. It verifies the
three published selfcheck prompt hashes and stores only test inputs and
Python-produced text, ids, errors and stream events. With no `YSKN_MODEL_DIR`,
the model-dependent suite prints `skipped: YSKN_MODEL_DIR is not set` and
passes; an explicitly set unusable folder fails. Synthetic UTF-8, loading,
JSON formatting and tool-processing tests always run without a model.

## Scalar block draft reference (D15)

`yskn_ref::draft::Draft::load(MODEL_FOLDER)` reads only the manifest's draft
container and reduced head. `load_with_options` changes the mask ID, capture
layer list or shortlist size (default 248070 / [5,19,33,47,61] / 512).
Supply each committed main row through `push_context(position,
ContextFeatures { values })`, or row-major `[row,slot,channel]` through
`push_context_rows`. The hook is precisely defined in
[D15](DECISIONS.md#d15--scalar-block-draft-and-reduced-head): post-MLP residual
BF16 expanded to FP32, five captures in layer order, no additional norm.
The main model does not expose these captures yet.

`propose(frontier_token, embedding_provider, DraftChoice::Greedy)` returns
seven token IDs plus each position's top16 with raw and corrected scores.
The provider supplies BF16-expanded main embedding rows for frontier/mask,
so the drafter never opens the main container. `DraftChoice::sampled(seed)`
is used only for main sampler version 3; its independent temperature is
850 thousandths and top-p 0.95. `block_hidden` accepts eight explicit
embedding rows for fast-path conformance. Proposing does not commit context.
`reset`, `snapshot`, `restore`, `position` and read-only snapshot ring bits
cover the draft's own state; callers manage main/draft commits together.
Batch failure retains earlier successful rows; a single failed row is atomic.

```sh
export CARGO_NET_OFFLINE=true
cargo run --release -p yskn-ref --example draft_info -- "$HOME/models/qwen38-speed"
YSKN_MODEL_DIR="$HOME/models/qwen38-speed" cargo test -p yskn-ref --release -- --nocapture draft
```

Without `YSKN_MODEL_DIR`, the real test prints
`skipped: YSKN_MODEL_DIR is not set` and passes. An unusable set path fails.
The info example reads headers only, lists every tensor/vector and SHEAD
array extent, and checks released shape/sign/rotation assumptions. Synthetic
tests need no artifact files or network. On the real files one scalar
`propose` takes about 90 seconds and about 1.6 GB resident.
The scalar reference intentionally remains slow; no main capture wiring,
speculation loop, 16-row option or HCAP side policy is included.
