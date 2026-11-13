# Specification gaps discovered in round 2

There are **three unresolved entries**. Each is a point where the permitted
documents and `pm_math.cuh` do not determine the required implementation.
No other upstream source was consulted to resolve them.

The container reader, compressed arithmetic, GDN model path, residual/norm/MLP,
head and samplers execute. Full mixed-layer model execution and a real-model
quality score remain blocked. Model and attention execution return a named `SpecGap`
error before changing state; the unresolved implementation sites retain
`todo!()` as requested. A passing CI run therefore does **not** mean the
mixed-model reference is complete.

## MATH-01

**Missing portable exp2 algorithm.** Chapter 04 uses exp2 for prefix online
softmax, online prefix rescaling and the separate self contribution. D2 requires
porting the explicit algorithm in `src/pm_math.cuh`. That file provides
exp, log, positive log1p, sin/cos, reciprocal square root and the derived
nonlinearities, but contains no exp2 function or algorithm.

- **Implementation boundary:** `portable_math::exp2` in
  `crates/yskn-ref/src/portable_math.rs`.
- **Behavior now:** `require_exp2()` returns `SpecGap("MATH-01")` before
  `Attention::step` prepares or appends any cache row. `exp2` itself is a
  referenced `todo!()`.
- **Blocked:** all full-attention outputs and mixed-model inference, including
  the quality command on the released architecture.
- **Needed:** an approved explicit portable exp2 algorithm with constants,
  operation order, rounding and exceptional-input behavior, or an explicit
  revision of D2 authorizing a specified construction. `exp(x*ln(2))` was
  not assumed equivalent.

## ARCH-01

**No header rule for rotary width under D5.** FORMAT.md §3.1 names `heads`,
`kv_heads` and `head_dim`, but leaves architecture-specific fields unnamed.
Chapters 01/04 specify a fixed 64-channel rotary transform and 32 exact
frequency constants. They explicitly say not to regenerate the frequencies
from `rope_theta`. Neither document identifies a field or relationship that
supplies rotary width from the model header, as D5 requires.

GDN dimensions are not blocked by this: chapter 03's tensor shapes uniquely
determine them from the container header's descriptors; see D5 in
[DECISIONS.md](DECISIONS.md#d5--header-derived-dimensions).

- **Implementation boundary:** `model::rotary_width_from_header` in
  `crates/yskn-ref/src/model.rs`.
- **Behavior now:** loading a model containing `full_attention` returns
  `SpecGap("ARCH-01")` before tensor payload loading. The missing mapping
  remains `todo!()`. The isolated attention kernel accepts an explicitly
  provided rotary dimension for testing its defined preparation components.
- **Blocked:** loading the released mixed-layer model and the mixed synthetic
  model as an executable `Model`.
- **Needed:** the actual header field or a documented derivation and its
  supported values; alternatively, an explicit exception to D5 allowing the
  fixed 64-channel rotary width. No speculative `rotary_dim` field was added.

## ATT-MASK-01

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
- **Behavior now:** both missing lookups have referenced `todo!()` entries.
  `require_masked_operands_spec()` rejects execution before state mutation
  after the exp2 preflight succeeds. An earlier tentative zero-padding
  interpretation was removed during review.
- **Blocked:** full attention remains unavailable even after MATH-01 is
  resolved, until masked operands are defined.
- **Needed:** exact loaded-extent bounds and exact scale/value-base operands
  for every masked tile slot, or an approved canonical zero-padding rule for
  this reference.

## Acceptance-test status

`tests/model_invariants.rs` runs row equivalence, prefill chunk equivalence,
provisional prefix commit, failure atomicity, reset and seeded sampling on
executable recurrent-only synthetic models. It also runs a second hidden/FFN
size, a padded vocabulary, and plain BF16 embeddings.

The writer produces a valid four-layer mixed `.yskn` and its complete
container payload is decoded in a passing test. Loading that model is tested
to return ARCH-01. `mixed_model_row_and_chunk_invariants` is explicitly
ignored, naming all three gaps, and must be enabled when they are resolved.
It is not counted as a passing invariant test. The quality example builds,
but no real-model NLL/perplexity result is claimed.
