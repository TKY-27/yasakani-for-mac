# Specification gaps discovered in round 2

All **three entries are resolved** by D7–D9 in
[DECISIONS.md](DECISIONS.md). The descriptions below retain the discovery
history: the permitted documents and `pm_math.cuh` did not determine these
choices, and the project owner subsequently supplied the explicit decisions.
No other upstream source was consulted to resolve them.

Full mixed-layer loading and execution are enabled. Synthetic mixed-model
invariants now run; no real-model quality score or CUDA bit equivalence is
claimed.

## MATH-01

**Resolved by [D7](DECISIONS.md#d7--portable-exp2-construction):** use portable
`exp(t * f32::from_bits(0x3f31_7218))`, with a separately rounded binary32 product.

**Missing portable exp2 algorithm.** Chapter 04 uses exp2 for prefix online
softmax, online prefix rescaling and the separate self contribution. D2 requires
porting the explicit algorithm in `src/pm_math.cuh`. That file provides
exp, log, positive log1p, sin/cos, reciprocal square root and the derived
nonlinearities, but contains no exp2 function or algorithm.

- **Implementation boundary:** `portable_math::exp2` in
  `crates/yskn-ref/src/portable_math.rs`.
- **Behavior before resolution:** `require_exp2()` returns `SpecGap("MATH-01")` before
  `Attention::step` prepares or appends any cache row. `exp2` itself is a
  referenced `todo!()`.
- **Previously blocked:** all full-attention outputs and mixed-model inference, including
  the quality command on the released architecture.
- **Decision requested at discovery:** an approved explicit portable exp2 algorithm with constants,
  operation order, rounding and exceptional-input behavior, or an explicit
  revision of D2 authorizing a specified construction. `exp(x*ln(2))` was
  not assumed equivalent.

## ARCH-01

**Resolved by [D8](DECISIONS.md#d8--fixed-qwen3_5-rotary-width):** `model.arch`
`qwen3_5` uses rotary width 64 and requires `head_dim == 256`, an explicit
exception to D5. Unsupported architectures return `InvalidModel`.

**No header rule for rotary width under D5.** FORMAT.md §3.1 names `heads`,
`kv_heads` and `head_dim`, but leaves architecture-specific fields unnamed.
Chapters 01/04 specify a fixed 64-channel rotary transform and 32 exact
frequency constants. They explicitly say not to regenerate the frequencies
from `rope_theta`. Neither document identifies a field or relationship that
supplies rotary width from the model header, as D5 requires.

GDN dimensions were not blocked by this: chapter 03's tensor shapes uniquely
determine them from the container header's descriptors; see D5 in
[DECISIONS.md](DECISIONS.md#d5--header-derived-dimensions).

- **Implementation boundary:** `model::rotary_width_from_header` in
  `crates/yskn-ref/src/model.rs`.
- **Behavior before resolution:** loading a model containing `full_attention` returns
  `SpecGap("ARCH-01")` before tensor payload loading. The missing mapping
  remains `todo!()`. The isolated attention kernel accepts an explicitly
  provided rotary dimension for testing its defined preparation components.
- **Previously blocked:** loading the released mixed-layer model and the mixed synthetic
  model as an executable `Model`.
- **Decision requested at discovery:** the actual header field or a documented derivation and its
  supported values; alternatively, an explicit exception to D5 allowing the
  fixed 64-channel rotary width. No speculative `rotary_dim` field was added.

## ATT-MASK-01

**Resolved by [D9](DECISIONS.md#d9--per-row-prefix-tiles-and-skipped-masked-values):**
visit only each row's intersecting absolute 32-slot prefix tiles; masked scores
are negative infinity, probabilities are `+0.0`, and value operations are skipped.

**Masked prefix value operand bits are underspecified.** Chapter 04 rounds
the shared prefix tile start down to a multiple of 32, then says masked keys
have zero scale outside the “shared loaded extent.” It does not determine
whether that extent starts at the unrounded causal `first` or at the aligned
tile start, nor the value-base operand bits for masked slots. The valid
prefix's operands and the zero mask probability are specified.

This can matter after D3 chooses scalar multiply/add: zero multiplied by a
negative value can be negative zero, and replacing those operands with
positive zeros is not justified by a bit-exact real-arithmetic identity.

- **Implementation boundaries:** masked value-scale and masked value-base
  lookups in `attention::Attention::prefix_tile`, in
  `crates/yskn-ref/src/attention.rs`.
- **Behavior before resolution:** both missing lookups have referenced `todo!()` entries.
  `require_masked_operands_spec()` rejects execution before state mutation
  after the exp2 preflight succeeds. An earlier tentative zero-padding
  interpretation was removed during review.
- **Previously blocked:** full attention remains unavailable even after MATH-01 is
  resolved, until masked operands are defined.
- **Decision requested at discovery:** exact loaded-extent bounds and exact scale/value-base operands
  for every masked tile slot, or an approved canonical zero-padding rule for
  this reference.

## Acceptance-test status

`tests/model_invariants.rs` runs row equivalence, prefill chunk equivalence,
provisional prefix commit, failure atomicity, reset and seeded sampling on
executable recurrent-only synthetic models. It also runs a second hidden/FFN
size, a padded vocabulary, and plain BF16 embeddings.

The writer produces a valid four-layer mixed `.yskn` and its complete
container payload is decoded in a passing test. Before resolution, loading
returned ARCH-01 and `mixed_model_row_and_chunk_invariants` was explicitly
ignored. It is now enabled and loading succeeds; invalid head widths and
unsupported architectures are rejected. A 73-row mixed pass starting at
position 3 compares every logit bit against single rows and uneven chunks,
and attention-level tests vary masked cache contents and verify skipped
value operations. The quality example builds, but no real-model
NLL/perplexity result is claimed.
