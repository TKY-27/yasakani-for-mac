# 02. Compressed linear operator

## Scope and scalar notation

This chapter specifies the integer-accumulation path selected by `Engine35`'s
default `xsm_ = 1`. Main-model calls with at most 16 rows use `launch_lnord`;
larger body calls use PFA or, when eligible, G6. The draft's compressed matrices
also use LNORD with integer accumulation. These are different entry points to
the same table lookup, activation-scale conversion and weighted integer sum.
(`src/engine35.hpp:246-255`, `src/engine35.cpp:403-420`,
`src/engine35.cpp:1756-1764`, `src/pfa.cu:355-364`,
`src/q35_g6.cu:310-350`.)

Use the existing [container specification](../FORMAT.md) for header fields,
matrix-code addressing, trellis-state decoding and sign-stream generation.
The additional table quantization below is performed by the inference loader;
it is not part of reconstructing the floating-point matrix described in that
document. (`src/loader.cpp:19-36`, `src/lnord.cu:265-300`.)

The following notation describes scalar operations, with every intermediate
kept at the stated precision. `add32`, `sub32`, `mul32` and `div32` are separate
binary32 operations rounded to nearest, ties to even. `float32(i64)` is the
explicit nearest-even integer-to-binary32 conversion. `half(x)` means
binary32 → binary16, nearest-even → binary32. `bfloat(x)` means binary32 →
bfloat16, nearest-even → binary32. `int_rne(x)` means the explicit
nearest-even binary32-to-signed-int conversion. The underlying source uses
`__fadd_rn`, `__fsub_rn`, `__fmul_rn`, `__fdiv_rn`, `__ll2float_rn`,
`__float2half_rn`, `__float2bfloat16_rn` and `__float2int_rn` at these points.
(`src/a8_prologue.cuh:15`, `src/a8_prologue.cuh:36-38`,
`src/a8_prologue.cuh:74-78`, `src/q35_kernels.cu:35-37`,
`src/lnord.cu:49-54`, `src/q35_g6.cu:265-270`.)

The Windows CUDA build disables implicit multiply-add contraction with
`--fmad=false`. Do not replace a `mul32` followed by an `add32` with an FMA.
Explicit `fmaf` calls elsewhere remain explicit FMA operations. The option is
inside `if(MSVC)`, so this is not a claim about every possible build of the
repository. (`CMakeLists.txt:47-51`, `src/q35_bdraft.cu:31-36`.)

## Loader: floating table to signed-byte table

For each compressed matrix, let `table16[4096][2]` and `row_scale16[N]` be the
binary16 values obtained using FORMAT.md. `N` is `physical_rows`, and
`K` is `cols`. The loader constructs one signed-byte table and one binary32
effective scale per output row. The maximum is over all 8,192 table entries,
not over each code pair or output row. (`src/loader.cpp:19-32`.)

```text
table_max = +0.0
for state = 0 .. 4095:
    for component = 0 .. 1:
        table_max = max(table_max, abs(float32(table16[state][component])))
table_scale = host_float_divide(table_max, 127.0)

for state = 0 .. 4095:
    for component = 0 .. 1:
        ratio = host_float_divide(float32(table16[state][component]), table_scale)
        q = integer_conversion(host_nearbyint(ratio))
        weight_table8[state][component] = clamp(q, -127, 127)

for output = 0 .. N-1:
    effective_row_scale[output] =
        host_float_multiply(float32(row_scale16[output]), table_scale)
```

The two signed bytes occupy the low and high bytes of the loader's `uint16_t`
table entry. Decode the code stream's state for each pair of weights as in
FORMAT.md §4.2, then index `weight_table8`, rather than the original binary16
table. Repacking changes addresses, not table entries or codes.
(`src/loader.cpp:22-24`, `src/loader.cpp:38-44`,
`src/repack.cu:10-17`, `src/lnord.cu:58-78`,
`src/lnord.cu:265-300`.)

**UNCERTAIN [LIN-01]:** the loader uses ordinary host `float` expressions and
`std::nearbyint`, without selecting a floating-point rounding mode in this
routine. Nearest-even is therefore an assumption about the host environment,
not an explicitly established loader setting. No zero/nonfinite-table guard
precedes division by `table_scale`; an all-zero table reaches `0 / 0` and an
integer conversion. Checked the complete loader and the build options; neither
defines a portable result for that case. (`src/loader.cpp:19-32`,
`CMakeLists.txt:47-51`.)

## Hadamard transform and signs

`hadamard512` below is unnormalized. Each stage reads its pair before either
element is written. Stages run in ascending stride order. Replacing this with
512 independent dot products, or changing the stage order, does not preserve
the specified floating-point evaluation order. The normalization constant is
the binary32 value of the literal `0.044194173824159216f` (bits `0x3d3504f3`).
(`src/a8_prologue.cuh:41-50`, `src/a8_prologue.cuh:87-97`,
`src/q35_kernels.cu:246-259`.)

```text
hadamard512(values[512]):
    for stride in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for base = 0, 2*stride, 4*stride, ... < 512:
            for offset = 0 .. stride-1:
                lower = values[base + offset]
                upper = values[base + offset + stride]
                values[base + offset]          = add32(lower, upper)
                values[base + offset + stride] = sub32(lower, upper)
    return values
```

Input signs are generated from `rot_in`; output signs, when present, are
generated separately from `rot_out`. Generate each stored sign array from its
tensor's seed, offset and rule, with the exact integer sign-stream rules in
FORMAT.md §4.5. Input signs are
multiplied **before** the input Hadamard transform. Output signs are multiplied
**after** the output Hadamard transform and its normalization. In particular,
the output transform is not obtained by applying input signs to an already
transformed array. (`src/loader.cpp:34-36`, `src/signs.hpp:9-30`,
`src/a8_prologue.cuh:64-72`, `src/q35_kernels.cu:73-79`.)

The caller chooses which input sign array is actually used. Several matrices
share one already quantized input; the engine does not independently apply
each matrix's `rot_in` when executing that group. Preserve the following
choices even if inspecting a file whose grouped matrices disagree about their
input signs. (`src/engine35.cpp:409`, `src/engine35.cpp:757-781`,
`src/engine35.cpp:793-799`, `src/engine35.cpp:824-829`.)

| Main-model projection group | Input sign array actually used |
|---|---|
| Gated DeltaNet `qkv`, `z` | `qkv.signs_in`. (`src/engine35.cpp:798`, `src/engine35.cpp:829`.) |
| Full attention `q`, `k`, `v` | `q.signs_in`. (`src/engine35.cpp:682`, `src/engine35.cpp:798`, `src/engine35.cpp:829`.) |
| MLP `gate`, `up` | `gate.signs_in`. (`src/engine35.cpp:736`, `src/engine35.cpp:759`, `src/engine35.cpp:793`.) |
| MLP `down` | `down.signs_in`. (`src/engine35.cpp:760-765`.) |
| Vocabulary head | `head_.signs_in`. (`src/engine35.cpp:799`, `src/engine35.cpp:806-809`.) |

## Activation quantization

Each activation row has `G = K / 512` independent scales. The caller supplies
the pre-rotation binary32 array `input`; chapter 05 specifies why this is not
always an already normalized hidden row. Apply the following independently
to each group. The FP16 rounding occurs **after** the Hadamard transform and
normalization, before finding the group maximum. There is no asymmetric zero
point, and `-128` is never emitted for finite inputs.
(`src/q35_kernels.cu:1459-1463`, `src/a8_prologue.cuh:122-132`.)

```text
for group = 0 .. G-1:
    for j = 0 .. 511:
        i = group*512 + j
        rotated[j] = mul32(input[i], float32(input_sign[i]))
    rotated = hadamard512(rotated)
    for j = 0 .. 511:
        rotated[j] = half(mul32(rotated[j], 0.044194173824159216f))
    maximum = max(abs(rotated[0]), ..., abs(rotated[511]))
    activation_scale[group] = div32(maximum, 127.0)
    for j = 0 .. 511:
        ratio = div32(rotated[j], activation_scale[group]) if activation_scale[group] > 0 else +0.0
        activation8[group*512 + j] = clamp(int_rne(ratio), -127, 127)
```

For at most eight main-model rows the default `opt_lo_` stores these same
activation bytes in an interleaved layout. Its address permutation does not
change the above scalar values. Larger rows use the ordinary row-major layout;
the PFALT transform used above 64 rows follows the same ascending Hadamard
strides and the same FP16/scale/int8 conversion.
(`src/engine35.hpp:307`, `src/engine35.cpp:824-829`,
`src/a8_prologue.cuh:120-132`, `src/lnord.cu:283-300`,
`src/q35_kernels.cu:247-271`, `src/q35_kernels.cu:602`.)

## Integer sum and activation-scale conversion

The per-group scales are themselves approximated on a common integer grid
for each activation row. This is essential: multiplying every group dot
product by its original `activation_scale` in binary32 is a different
algorithm. The maximum is over the complete input row, including all of its
512-element groups. (`src/lnord.cu:52-54`, `src/lnord.cu:236-262`,
`src/lnord.cu:303-316`.)

```text
maximum_scale = max(activation_scale[0 .. G-1])
scale_multiplier = div32(16777216.0, maximum_scale) if maximum_scale > 0 else +0.0
row_quantum = ldexp_binary32(maximum_scale, -24)

for group = 0 .. G-1:
    integer_scale[group] = int_rne(mul32(activation_scale[group], scale_multiplier))

for output = 0 .. N-1:
    weighted_sum[output] = signed64(0)
    for group = 0 .. G-1:
        dot = signed32(0)
        for j = 0 .. 511:
            column = group*512 + j
            dot += signed32(decoded_weight8[output][column]) * signed32(activation8[column])
        weighted_sum[output] += signed64(integer_scale[group]) * signed64(dot)
```

The dot product combines both adjacent 256-weight code sequences before
applying `integer_scale`. Its terms are signed integer products. LNORD resets
the 32-bit accumulator after each 512-element group and accumulates the
weighted products in signed 64-bit values; the inter-worker combination is
also integer addition. Thus partitioning these integer sums is not a
floating-point reduction choice. (`src/lnord.cu:267-268`,
`src/lnord.cu:290-316`, `src/lnord.cu:321-354`.)

## Restoring scales and output coordinates

For a projection that consumes a main-model normalization, `inverse_rms` comes
from chapter 05. Other projections omit that multiplication entirely. For
each output coordinate, evaluation is left to right as written below; the
row scale is applied in rotated output coordinates, before the inverse
rotation. (`src/a8_prologue.cuh:15`, `src/q35_kernels.cu:1409-1417`,
`src/q35_kernels.cu:1437-1441`.)

```text
for output = 0 .. N-1:
    linear[output] = mul32(float32(weighted_sum[output]), row_quantum)
    if projection_consumes_main_normalization:
        linear[output] = mul32(linear[output], inverse_rms)
    linear[output] = mul32(linear[output], effective_row_scale[output])

if projection_uses_output_rotation:
    for block = 0 .. N/512-1:
        block_values = hadamard512(linear[block*512 .. block*512+511])
        for j = 0 .. 511:
            i = block*512+j
            linear[i] = mul32(mul32(block_values[j], 0.044194173824159216f), float32(output_sign[i]))
```

`projection_uses_output_rotation` is a property of the consuming code: the
main body post steps perform it; the main head and the draft's post steps do
not. The body post steps dereference their output-sign arrays rather than
checking a per-call rotation-kind flag. The pseudocode is not an additional
promise to support body tensors with missing output signs.
(`src/q35_kernels.cu:1405-1417`, `src/q35_kernels.cu:1432-1441`,
`src/misc_kernels.cu:67-73`, `src/q35_bdraft.cu:55-63`.)

Do not round the generic `linear` result to binary16 here. Consumers choose
their own conversion: the main residual/MLP paths use bfloat16 at the points
in chapter 05; the vocabulary head keeps binary32 logits; the draft's plain
post step keeps binary32 and has no output rotation.
(`src/q35_kernels.cu:1416-1419`, `src/q35_kernels.cu:1440-1441`,
`src/misc_kernels.cu:67-73`, `src/q35_bdraft.cu:55-63`.)

The shared upstream accumulator is cleared when consumed (`xsum_combine`),
while an ordinary scalar implementation can give each projection a fresh
`weighted_sum`. Do not clear an accumulator before its consuming post step
or accidentally reuse its previous contents.
(`src/a8_prologue.cuh:10-15`, `src/q35_kernels.cu:89-101`.)

## One row, verification rows and prefill rows

| Call | Accumulation | Where conversion/scaling happens |
|---|---|---|
| Main body, 1–16 rows | LNORD signed-64 weighted sum | Consumer performs `float32(sum) * row_quantum`, optional inverse RMS, then row scale. (`src/engine35.cpp:403-427`, `src/q35_kernels.cu:1409-1412`.) |
| Main body, more than 16 rows | PFA XSUM; G6 is tried first and requires at least 256 rows | Backend performs `float32(sum) * row_quantum` and optional inverse RMS, writes binary32; consumer applies row scale. (`src/engine35.cpp:413-419`, `src/pfa.cu:226-239`, `src/pfa.cu:301-310`, `src/q35_g6.cu:265-270`, `src/q35_g6.cu:310-312`.) |
| Main vocabulary head | LNORD in slices of at most 16 rows | Head consumer performs the same scale restoration, with final inverse RMS and no output rotation. (`src/engine35.cpp:803-813`, `src/engine35.cpp:835-837`, `src/misc_kernels.cu:62-73`.) |
| Draft compressed projection | LNORD signed-64 weighted sum | Draft consumer performs `float32(sum) * row_quantum * row_scale`; its normalization is already part of its input construction. (`src/engine35.cpp:1756-1764`, `src/q35_bdraft.cu:40-51`, `src/q35_bdraft.cu:55-63`.) |

PFA and G6 do not replace the weighted integer accumulation with a
floating-point accumulation. Their float output is a change of intermediate
representation. Both use the same `2^24` scale multiplier, integer scale
rounding and final multiplication order as above. This supports equivalence
of this compressed operation for finite in-range values; it is not proof
that the complete attention/recurrent model is row-count invariant.
(`src/pfa.cu:153-165`, `src/pfa.cu:226-239`, `src/pfa.cu:301-310`,
`src/q35_g6.cu:130-146`, `src/q35_g6.cu:210-222`,
`src/q35_g6.cu:254-270`.)

**Overview discrepancy:** TECHNOLOGY.md says that prompts processed in
17 or more rows expand one matrix at a time through `q35_g6.cu`. The actual
G6 entry point rejects fewer than 256 rows. Calls with 17–255 rows therefore
reach PFA, which decodes compressed code pairs inside its multiplication
loop. This is a dispatch mismatch in the overview, not a numerical difference
in the integer-sum formula. (`docs/TECHNOLOGY.md:295-296`,
`src/engine35.cpp:416-419`, `src/q35_g6.cu:310-312`,
`src/pfa.cu:172-221`.)

## Embedding rows are decoded differently

For coded `embed_tokens`, use the original binary16 lookup table and binary16
row scale, without building a signed-byte table and without quantizing an
activation. Optional `embed_tokens.res` and `embed_tokens.res2` are decoded
independently and added in that order in rotated coordinates. Apply the
Hadamard transform, its normalization and the **main embedding's** input signs,
then round once to bfloat16. The default embedding implementation is the
block version below; the alternate one-row implementation uses the same
arithmetic order. (`src/engine35.cpp:272-299`,
`src/q35_kernels.cu:1501-1522`, `src/q35_kernels.cu:1527-1554`,
`src/q35_kernels.cu:1571-1574`.)

```text
for i = 0 .. 5119:
    value[i] = mul32(float32(main_table16[main_state(i)][i mod 2]), float32(main_scale16[token]))
    if residual_table_1_exists:
        value[i] = add32(value[i], mul32(float32(residual_table16_1[state_1(i)][i mod 2]), float32(residual_scale16_1[token])))
        if residual_table_2_exists:
            value[i] = add32(value[i], mul32(float32(residual_table16_2[state_2(i)][i mod 2]), float32(residual_scale16_2[token])))
for block = 0 .. 9:
    values = hadamard512(value[block*512 .. block*512+511])
    for j = 0 .. 511:
        i = block*512+j
        embedding[i] = bfloat(mul32(mul32(values[j], 0.044194173824159216f), float32(main_embedding_sign[i])))
```

Here each `state(i)` is the FORMAT.md trellis state for the pair containing
coordinate `i` in that table's own code stream. The model loader rejects a
third table without the second. If there is no coded `embed_tokens` tensor,
it instead requires a plain bfloat16 `embed_tokens` vector with
`vocab * hidden` entries. (`src/engine35.cpp:168-177`,
`src/engine35.cpp:272-299`, `src/q35_kernels.cu:1532-1545`.)

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
