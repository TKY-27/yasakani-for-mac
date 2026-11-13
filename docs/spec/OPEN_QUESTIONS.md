# Open questions, observed hazards, and path differences

The numerical chapters specify source-visible array operations and rounding boundaries. They do **not** establish a fully portable, bit-exact substitute for the upstream binary while the items below remain unresolved. The main dependencies are the absent model/profile payloads, CUDA elementary-function results, instruction-internal accumulation, and scheduling-dependent shortlist membership. These are separate issues: reproducing the sampler hash does not establish identical logits. (`src/engine35.cpp:214-238`; `src/q35_kernels.cu:42-44`; `src/q35_atxpk.cu:35-37`; `src/q35_bdraft.cu:882-896`; `src/specx.cpp:376`.)

No inference run, CUDA build, or empirical equivalence test was performed for this documentation round. Each entry below states what the inspected code determines and what remains unknown. A named unresolved primitive in pseudocode is intentional, not permission to substitute an approximately equivalent operation.

## Complete UNCERTAIN registry

### Model artifacts

Full algorithm: [01_MODEL_ARCHITECTURE.md](01_MODEL_ARCHITECTURE.md).

**UNCERTAIN [ARCH-01]:** The released main `.yskn` header is not included in
the checkout. The executable does not hard-code 64 layers, 17,408 feed-forward
width, the complete layer-type sequence, or the released epsilon. The table
below is corroborated by the committed `compare_fastest.json` and
`compare_quality.json` tensor inventories, not by opening the released weights.
For an actual model, read these values from its header and check them against
the expected inventory; do not substitute an assumed Qwen configuration.
Checked: `src/format.hpp:33-37`, `src/engine35.cpp:214-238`,
`tools/records/compare_fastest.json:1-14`,
`tools/records/compare_quality.json:1-14`.

### Compressed linear operator

Full algorithm: [02_COMPRESSED_LINEAR.md](02_COMPRESSED_LINEAR.md).

**UNCERTAIN [LIN-01]:** the loader uses ordinary host `float` expressions and
`std::nearbyint`, without selecting a floating-point rounding mode in this
routine. Nearest-even is therefore an assumption about the host environment,
not an explicitly established loader setting. No zero/nonfinite-table guard
precedes division by `table_scale`; an all-zero table reaches `0 / 0` and an
integer conversion. Checked the complete loader and the build options; neither
defines a portable result for that case. (`src/loader.cpp:19-32`,
`CMakeLists.txt:47-51`.)

**UNCERTAIN [LIN-02]:** residual-embedding validation checks rotation kind and
sign offset but does not compare sign seed or sign rule. Execution always
uses the main table's sign stream. A residual tensor with a different seed
or rule would therefore pass these checks and be interpreted with the main
signs. The released tensors were not available to establish whether such a
mismatch occurs. (`src/engine35.cpp:284-296`,
`src/q35_kernels.cu:1535-1554`.)

**UNCERTAIN [LIN-03]:** no portable exceptional-value policy can be inferred
for all activation conversions. The maximum uses unsigned bit patterns of
`abs` values and the next step converts a floating quotient to an integer;
the source does not reject NaNs/infinities or prescribe a scalar substitute
for their device conversion. The scalar algorithms above specify finite,
in-range execution; this is a missing exceptional-input contract, not a claim
that the released model produces nonfinite activations.
(`src/a8_prologue.cuh:125-132`, `src/lnord.cu:52-54`.)

**UNCERTAIN [LIN-04]:** the generic loader uploads a cross-block mixing matrix
`qm_in` when `rot_in.kind == "AYA"`, but `Engine35::up` does not reject that
kind, and its shipping input preparation receives only a sign array and
always applies the 512-element Hadamard defined above. It does not consume
that uploaded mixing matrix. Do not add FORMAT.md's AYA mixing step to this
engine's execution by assumption. Whether a released body tensor requests
AYA, and would therefore disagree with this execution, cannot be checked
without the model artifacts. Checked the loader, `Engine35` tensor upload
and input-preparation callers, and their kernel bodies.
(`src/loader.cpp:34-35`, `src/engine35.cpp:157-163`,
`src/engine35.cpp:559-565`, `src/q35_kernels.cu:1459-1463`,
`src/a8_prologue.cuh:122-132`.)


### Gated DeltaNet

Full algorithm: [03_GATED_DELTANET.md](03_GATED_DELTANET.md).

**UNCERTAIN [GDN-01]:** The repository names CUDA's native `expf`, `rsqrtf`, and `log1pf`, but does not specify their output bits for every binary32 input. A Rust scalar reference cannot establish CUDA bit identity by substituting its platform math library. The portable-math branch is a different configuration, not evidence for the shipping branch's bits. Checked: the wrappers and their callers at `src/q35_kernels.cu:42-44`, `src/q35_kernels.cu:113-116`, `src/q35_kernels.cu:874-896`, `src/q35_kernels.cu:934-947`, and the configuration at `src/engine35.cpp:363-364`.

### Attention and KV codec

Full algorithm: [04_ATTENTION.md](04_ATTENTION.md).

**UNCERTAIN [ATT-01]:** Prefix attention invokes `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. The source fixes operand values and the sequence of 16-element chunks, but does not define the instruction's internal addition order or all rounding behavior. Below, `mma16(acc,a[16],b[16])` names precisely this unresolved primitive. It must not silently be replaced by 16 sequential FP32 multiply/adds and called bit-exact. The optional FP16 path has the same issue with its FP16 MMA instruction. Checked: `src/q35_atxpk.cu:35-37`, `src/q35_atx.cu:30-32`, `src/q35_atx.cu:65-69`, `src/q35_atxpk3.cuh:343-355`, `src/q35_atxpk4.cuh:287-300`.

**UNCERTAIN [ATT-02]:** The output bits of CUDA's native `cosf`, `sinf`, `rsqrtf`, `expf`, and `exp2f` are not defined in this repository for all inputs. The shipping path selects those functions, not the portable-math alternatives. A scalar implementation needs a separately established compatibility primitive for each; this document preserves the exact arguments and rounding boundaries. Checked: `src/q35_kernels.cu:42-44`, `src/q35_kernels.cu:1096-1101`, `src/q35_atxpk3.cuh:396-406`, `src/q35_kernels.cu:1295-1308`, `src/engine35.cpp:363-364`, `python/yasakani/runtime.py:27-43`.

**UNCERTAIN [ATT-04]:** The general trellis path has signed 32-bit squared-error costs without an overflow check. The profile format permits signed int16 codebook entries, while the target is clamped only to `[-2048,2048]`; such representable inputs can make `difference*difference + predecessor_cost` exceed signed 32-bit range. The VIT2 eligibility check applies the narrower `[-1100,1100]` table bound, but rejecting VIT2 falls back to the unchecked general path. It is unknown whether the absent shipping profile can reach overflow; no wraparound behavior is asserted as a portable contract. Checked: `src/kvq_fast.cuh:108-123`, `src/kvq_fast.cuh:162-168`, `src/kvq_fast.cuh:240-246`, `src/kvq_fast.hpp:59-66`.

**UNCERTAIN [ATT-03]:** The checked-in source does not contain the shipping `kvq_u5.bin`. Therefore the codebook values, candidate multipliers, means, group modes, guard positions, VIT2 eligibility, and actual packed-key/packed-value layer list cannot be enumerated from source alone. The author's statement that all 16 key sides and 13 value sides are packed is an artifact-specific claim, not a hardcoded engine rule. Checked: `src/kvq_atxpk.hpp:60-83`, `src/kvq_fast.hpp:46-66`, `src/engine35.cpp:242-251`, `docs/TECHNOLOGY.md:231-231`.

### Residual and normalization

Full algorithm: [05_RESIDUAL_NORM_MLP.md](05_RESIDUAL_NORM_MLP.md).

**UNCERTAIN [NORM-01]:** source alone does not define a portable bit-exact
implementation of shipping `rsqrtf` and `expf`. The code selects these vendor
functions when `q35_pm_on == 0`, which is the default after `apply_kz` with
`kz_ == 0` and no `YSKN_Q35_PM` override. The alternative explicit
`pm_rsqrt`/`pm_expf` algorithms are present, but substituting them would
specify a different configuration. Checked the math dispatch, the complete
`pm_math.cuh`, and CMake; the shipping vendor implementations and their
version-specific bit patterns are not supplied by this repository. Retain
the named `upstream_rsqrtf` and `upstream_expf` primitives as unresolved
dependencies until a matching implementation or independent golden outputs
are supplied. (`src/q35_kernels.cu:42-44`,
`src/q35_kernels.cu:113-116`, `src/engine35.hpp:289`,
`src/engine35.cpp:458-465`, `src/pm_math.cuh:16-38`,
`CMakeLists.txt:1-11`, `CMakeLists.txt:47-52`.)


### Output head and sampler

Full algorithm: [06_OUTPUT_HEAD_SAMPLER.md](06_OUTPUT_HEAD_SAMPLER.md).

**UNCERTAIN [SMP-01]: CUDA transcendental bits.** The source invokes
`rsqrtf`, binary64 `exp`, and binary64 `log`, and contains an optional portable
math implementation which shipping does not enable. Their exact CUDA library
and device results cannot be reconstructed from these call sites alone.
This document fixes argument construction, precision, operation order and
hash bits; an independent implementation must leave these primitive results
unresolved rather than assume the host's `sqrt`/`exp`/`log` are bit-identical.
Checked `src/misc_kernels.cu:63-69`, `src/q35_sample.cu:179-202`,
`src/smp_hash.cuh:12-15`, and `src/engine35.cpp:363-364,465`.

**UNCERTAIN [SMP-02]: Nonfinite-logit behavior.** Top-k initializes with
`-FLT_MAX`, greedy initializes with that value and ID `INT_MAX`, while draft
top-k uses `-INFINITY`. NaN comparisons are false and no finite-logit
validation occurs here. A universal total-order sort would invent behavior
for NaNs or an all-`-INFINITY` row; the finite-logit contract above must not be
extended to those cases without specifying the original candidate-reduction
layout. Checked `src/misc_kernels.cu:59-88,121-140`,
`src/q35_sample.cu:28-48,131-178`, and `src/q35_bdraft.cu:928-955`.

**UNCERTAIN [SMP-03]: Upper endpoint in the Gumbel uniform.** If `h >> 11`
equals `2^53-1`, adding `0.5` in round-to-nearest-even binary64 rounds up to
`2^53`, so `u` is exactly `1.0` and `-log(-log(u))` is positive infinity.
The source does not clamp that endpoint. This follows from the written
arithmetic; whether a released-model request reaches such a hash has not
been determined. Preserve the formula rather than silently changing it to
an open-interval generator. Checked `src/smp_hash.cuh:12-15` and the callers
at `src/q35_sample.cu:195-197` and `src/q35_bdraft.cu:401-402`.


### Draft model

Full algorithm: [07_DRAFT_MODEL.md](07_DRAFT_MODEL.md).

**UNCERTAIN [DRF-02]: Draft container invariants.** The caller assumes the
common sign vector is long enough and appropriate for all those matrices,
and ignores their output rotation metadata in draft postprocessing. The
released draft container is not present in this source checkout, so its
actual rotation/sign descriptors and compatibility with those assumptions
cannot be established here. Checked `src/engine35.cpp:136-166,1624-1638`,
`src/engine35.cpp:1755-1782`, and `src/q35_bdraft.cu:55-62,474`.

**UNCERTAIN [DRF-01]: Cutoff-bucket membership is scheduling-dependent.**
Items in the cutoff bucket claim slots by `atomicAdd`; there is no token-ID
or low-11-mantissa-bit tiebreaker. When more items share the bucket than fit,
the source determines only the admissible subset, not its exact membership
without a GPU execution schedule. A scalar reference must expose this
boundary ambiguity instead of silently imposing ascending token ID.
Checked `src/q35_bdraft.cu:882-896` and the fixed collection launch in
`src/q35_bdraft.cu:971-974`. The score grid itself is chosen from device
occupancy (`src/q35_bdraft.cu:958-962`).

**UNCERTAIN [DRF-03]: Draft CUDA math bits.** Draft normalization, RoPE and
attention call `rsqrtf`, `exp2f`, `sincosf` and `__expf`; coupled selection
also calls binary64 `exp` even when the main portable-math flag is enabled.
The source fixes their arguments and the reduction orders above but does
not provide the shipping CUDA implementations. Do not substitute the main
model's numerically different math helpers by assumption. Checked
`src/q35_bdraft.cu:145-148,190,463-487,525-536,579,615,426`.

**UNCERTAIN [DRF-04]: Equality to a full compressed head.** No relationship
between `.shead`'s signed-int8 rescore weights and the main/draft compressed
head is validated by the loader. A claim of bit equality to that head is
unsupported; the supplied files would be needed to test it. Checked
`src/engine35.cpp:1700-1734,1819-1831` and
`src/q35_bdraft.cu:900-924`.


### Speculation

Full algorithm: [08_SPECULATION.md](08_SPECULATION.md).

**UNCERTAIN [SPEC-01]: Bit-exact decode/speculation equivalence.** Static
inspection establishes acceptance, state bookkeeping, and shared random
keys, but does not establish last-bit equality of every main arithmetic
path or explain the upstream report of nondeterministic text. In addition,
DRF-01 identifies a scheduling-dependent draft shortlist. A draft-only
ambiguity need not change accepted output when main verification is
equivalent, but that implication is conditional. Checked
`src/specx.cpp:344-440`, `src/engine35.cpp:824-865,1299-1329`,
`src/q35_bdraft.cu:882-896`, and `docs/TECHNOLOGY.md:273-278`.

### Prefill

Full algorithm: [09_LONG_PROMPT_PREFILL.md](09_LONG_PROMPT_PREFILL.md).

**UNCERTAIN [PF-01]:** Exact cross-path output identity between ordinary
chunked prefill, layer-major prefill, and repeated one-row passes is not
established by static inspection. The source changes gate-projection grouping,
recurrence kernels, attention tiling and finalization location. No CUDA
execution was performed. The individual differences are specified in the
numeric chapters; a future conformance suite needs to compare intermediate
arrays at these boundaries. Checked: `src/engine35.cpp:403-420`,
`src/engine35.cpp:772-790`, `src/engine35.cpp:1097-1175`,
`src/engine35.cpp:1244-1266`, `src/q35_kernels.cu:400-518,602,694-741`.

**UNCERTAIN [PF-02]:** The memory-dependent resident-row fallback of the
alternate layer-major path can change subbatch shapes. The source alone does
not establish that all those shapes produce identical bits. This is not a
shipping generation dependency while `prefill_head=1`. Checked:
`src/engine35.cpp:1017-1027`, `src/engine35.cpp:1107-1112`,
`src/specx.cpp:110-112`.


### Auxiliary files

Full algorithm: [10_AUXILIARY_FILES.md](10_AUXILIARY_FILES.md).

**UNCERTAIN [AUX-01]:** The meanings of unconsumed header bytes, group bytes
6–7, code-table byte 1 and bytes 8–11, the unused fourth SHEAD header word,
and any layout beyond the first two consumed residual basis vectors are not
recoverable from the readers. Do not call them zero padding or assign
a versioning meaning. Checked the field reads in `src/engine35.cpp:38-53`,
`src/kvq_fast.hpp:46-67`, `src/kvq_fast.cuh:54-62`,
`src/kvq_atxpk.hpp:25-57`, `src/kvq_gpu.cuh:58-99`, and
`src/engine35.cpp:1705-1707`.

**UNCERTAIN [AUX-02]:** The actual `kvq_u5.bin` payload is not in this
checkout. Therefore the numerical table entries, multiplier list, means,
which layer/kind remains int8, and whether its groups select the optimized
encoder cannot be established from its filename. `TECHNOLOGY.md` reports
16 packed key streams and 13 packed value streams, but the loader computes
that from the file; it is not hard-coded. Checked:
`python/yasakani/runtime.py:116`, `src/engine35.cpp:38-53`,
`src/engine35.cpp:242-251`, `src/kvq_atxpk.hpp:60-83`,
`docs/TECHNOLOGY.md:231-237`.

**UNCERTAIN [AUX-03]:** The KVQP reader performs some length checks but does
not validate every offset before dereferencing it. For example, the table
directory is indexed at `off_t+4*t` before validating that directory access,
and the group directory is similarly indexed before checking `go+8`.
Malformed inputs therefore lack a fully defined safe-reader contract. The
specification describes valid data, not a promise that the upstream loader
rejects all malformed files. Checked: `src/engine35.cpp:38-43`,
`src/kvq_fast.hpp:50-63`, `src/kvq_atxpk.hpp:29-38`.

## Additional source and build questions

**UNCERTAIN [BUILD-01]: Exact binary/compiler contract.** CMake requests a CUDA toolkit but pins no toolkit version. Its CUDA `--fmad=false` option is inside the MSVC branch; the specification treats the stated Windows build as that branch and retains every explicit FMA or rounded intrinsic where written. The repository does not identify the exact compiler/library binary needed to assign all native-function or MMA results. A different compiler setting must not be assumed numerically interchangeable. Checked `CMakeLists.txt:1-11,47-52`, `src/q35_kernels.cu:35-44`, `src/q35_bdraft.cu:767-795`, and `src/q35_atxpk.cu:35-37`.

**UNCERTAIN [FMT-01]: Container header-length discrepancy.** The existing format document defines a 64-bit header length, but `Container::open` copies only its lower 32 bits. This is a reader/document discrepancy; whether any released file has nonzero upper bits is not established because the model files are absent. Refer to the existing format for layout rather than inventing a second format here. Checked `docs/FORMAT.md:40` and `src/format.hpp:29-32`.

## Behaviors that look like bugs or missing validation

These are observed conditions and their potential effects, not claims that ordinary released-model requests have exhibited them. The associated uncertainty entries identify missing exposure evidence.

| Observed behavior | Why it matters | Source / registry |
|---|---|---|
| All-zero weight LUT divides by its zero maximum before integer conversion | No portable decoded byte result is specified for this otherwise representable table | `src/loader.cpp:19-32`; LIN-01 |
| AYA input mixing can load but the selected main input preparation does not use it | A header may describe a transform that the engine does not execute | `src/loader.cpp:34-35`; `src/engine35.cpp:157-163,559-565`; LIN-04 |
| Residual embedding seed/rule need not match the primary embedding | Validation can accept metadata that execution interprets with different signs | `src/engine35.cpp:284-296`; LIN-02 |
| SHEAD cutoff bucket is allocated by atomic arrival | Equal 21-bit score prefixes can choose a different candidate subset across schedules; no source-defined token-ID tie rule exists | `src/q35_bdraft.cu:882-896`; DRF-01 |
| Gumbel conversion can produce exactly `u=1` | The written expression can yield positive infinity; draft margin can become NaN if two leading scores are infinite | `src/smp_hash.cuh:12-15`; `src/q35_bdraft.cu:435-453,674`; SMP-03 |
| General trellis costs use unchecked signed-32 arithmetic | Valid int16 codebook entries can overflow the cost expression; unsigned wrapping cannot be silently adopted as a C++ contract | `src/kvq_fast.cuh:108-123,162-168`; ATT-04 |
| Profile directories are read before all offsets are bounds-checked | Malformed files do not have a complete safe rejection contract | `src/engine35.cpp:38-43`; `src/kvq_fast.hpp:50-63`; AUX-03 |
| Any layer-type string except `full_attention` selects GDN | A misspelled type can select a different architecture branch rather than fail validation | `src/engine35.cpp:224-238,255-266` |
| Model `rope_theta` is parsed but main text RoPE uses fixed frequency bit patterns | A changed header value does not change these frequencies | `src/format.hpp:37`; `src/q35_kernels.cu:1091-1104`; `src/engine35.cpp:335` |
| Header length is read narrower than documented | Files requiring the upper 32 bits are interpreted inconsistently with the existing format document | `src/format.hpp:29-32`; `docs/FORMAT.md:40`; FMT-01 |

## One-row, multi-row, prefill, and branch-row differences

The table distinguishes **known execution differences** from an unproven assertion that the final values differ. Integer sums and identical written operation schedules can agree across implementations; no measured disagreement is asserted without a run. (`src/lnord.cu:308-316`; `src/pfa.cu:153-165`; `src/q35_g6.cu:130-146`; `docs/TECHNOLOGY.md:273-278`.)

| Boundary | Source-defined difference to preserve | Evidence |
|---|---|---|
| 1-row speculation versus multi-row verification | One row normally commits GDN state during verify; later `commit(0)` skips replay. Multi-row verify is provisional and commit replays `accepted+1` cached rows | `src/engine35.cpp:1299-1305,1319-1329,1519-1525` |
| Plain versus HCAP rows | Convolution uses semantic depth; a leaf does not mutate the trunk state; its self KV slot is physical while RoPE/sampling position uses depth; matched leaf data is promoted before replay | `src/lfrow_kernels.cuh:31-150,154-210`; `src/q35_atxpk.cu:390-409`; `src/engine35.cpp:877-899` |
| 1..16 versus larger body projections | LNORD retains int64 weighted sums until the consumer; PFA/G6 finalize integer sum × common scale × optional inverse RMS into FP32 earlier. Both still accumulate the weighted sum as int64 | `src/engine35.cpp:403-429`; `src/engine35.hpp:253-255`; `src/pfa.cu:153-165`; `src/q35_g6.cu:130-146` |
| 17..255 versus eligible >=256 rows | G6 rejects M<256; smaller multi-row projections decode within PFA instead of predecoding a whole matrix | `src/q35_g6.cu:310-312`; `src/engine35.cpp:416-419`; `src/pfa.cu:172-221` |
| Up to 32 versus more than 32 GDN rows | Small dense a/b projections are fused or separately called; their documented product/reduction order is retained | `src/engine35.cpp:772-784`; `src/q35_kernels.cu:746-820` |
| Up to 64 versus more than 64 rows | PFALT=1 selects different preparation/convolution/state/output routines. Selection tests row count, not commit count | `src/q35_kernels.cu:602-606,958-959`; `src/engine35.cpp:772-790` |
| <=2 versus >2 packed-attention rows | Shipping PKV=3 uses pk3 with pair lookup versus pk4. Shared prefix tile extents also depend on row groups; ATT-01 remains unresolved | `src/q35_atxpk.cu:729-737`; `src/q35_atxpk3.cuh:184-195,509-516`; `src/q35_atxpk4.cuh:97-108,431-436` |
| Prefix versus self, in **every** row count | Prefix uses MMA and BF16 probability×Vscale; self uses explicit FP32 reduction and FP32 probability×Vscale. This is not uniquely a multi-row difference | `src/q35_atxpk3.cuh:309-406,412-458`; `src/q35_atxpk4.cuh:249-350,354-400` |
| 8-row versus 16-row block draft | Current-block attention is noncausal across the whole block, including positions not ultimately verified. Do not run only the retained prefix and expect the same proposals | `src/q35_bdraft.cu:122-150`; `src/engine35.cpp:1794,1851` |
| Shipping prefill versus direct layer-major prefill | `prefill_head=1` chunks through all layers and computes every head; direct `prefill` may select layer-major work, a memory-dependent resident batch, only final head, and a final 2048-row draft window | `src/specx.cpp:110-118`; `src/engine35.cpp:982-1048,1097-1175,1244-1266` |
| Main versus draft sampling temperature | Main uses binary64 integer/1000.0 and request truncation; draft uses binary64 `1e-3*cp`, fixed top-16 and top-p .95. Similar decimal notation does not establish identical bits | `src/engine35.cpp:946-959,1845-1850`; `src/q35_bdraft.cu:404-430` |

## Overview statements requiring correction or qualification

| Overview | Source finding | Evidence |
|---|---|---|
| 17+ rows expand weights with G6 | The threshold is >=256; 17..255 rows fall back to PFA | `docs/TECHNOLOGY.md:295-296`; `src/q35_g6.cu:310-312`; `src/engine35.cpp:416-419` |
| 8/16 continuation rows from block draft | The executed block contains the frontier plus 7/15 proposal positions | `docs/TECHNOLOGY.md:190`; `src/engine35.cpp:1794,1818` |
| Next-token table remembers what the main model chose | It records raw-logit top-8 entries; under sampling top8[0] can differ from the chosen token | `docs/TECHNOLOGY.md:192`; `src/specx.cpp:89-92`; `src/q35_sample.cu:177-205` |
| Shortlisted candidates are scored exactly | They are rescored with a separate int8 head from SHEAD; equality to main compressed head is not checked | `docs/TECHNOLOGY.md:208-210`; `src/engine35.cpp:1700-1734,1819-1831`; DRF-04 |
| 16 key and 13 value layers stay packed | This depends on absent profile records, not a fixed engine list | `docs/TECHNOLOGY.md:231`; `src/kvq_atxpk.hpp:60-83`; ATT-03 / AUX-02 |
| A shared rotation leaves attention unchanged | Real-arithmetic orthogonality does not remove explicit BF16 and int8 rounding in this implementation | `docs/TECHNOLOGY.md:226-231`; `src/q35_kernels.cu:1029-1059` |
| Same bits across speculative widths imply unchanged output | The local integer accumulation supports an invariance argument; full-model native math/MMA and schedule limits remain, and the overview reports unexplained changed text | `docs/TECHNOLOGY.md:129-141,212-214,273-278`; `src/q35_atxpk.cu:35-37`; SPEC-01 |

## Evidence needed to close the numeric gaps

A subsequent compatibility effort needs the exact released main/draft headers and auxiliary payloads for ARCH-01, LIN-02/04, DRF-02/04 and ATT-03/AUX-02; a pinned native arithmetic contract or independently supplied golden primitive results for the CUDA math/MMA entries; and an explicit treatment of the SHEAD cutoff race and exceptional inputs. A deterministic tie policy would be a deliberate port decision, not recovered upstream behavior. These requirements follow from the unchecked descriptors, external profile reads and unresolved primitive call sites cited in the registry above.
