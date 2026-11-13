# 05. Residual, normalization and MLP steps

Use the scalar rounding and `hadamard512`/activation-quantization definitions
in [02_COMPRESSED_LINEAR.md](02_COMPRESSED_LINEAR.md). This chapter describes
the shipping `Engine35` integer-sum path. Its hidden residual array is stored
as binary32, but embedding loads and both residual additions produce values
rounded to bfloat16. The model's normalization vectors are also bfloat16,
and the main-model normalization uses `1 + weight`, not `weight` alone.
(`src/engine35.cpp:168-177`, `src/engine35.hpp:248`,
`src/q35_kernels.cu:35-37`, `src/q35_kernels.cu:1427-1463`.)

## Order within and between layers

The first embedding row is prepared with layer 0's `input_layernorm`.
Each attention or Gated DeltaNet output projection is followed by residual
addition and preparation with that same layer's `post_attention_layernorm`.
Its MLP then applies `gate_proj` and `up_proj`, SiLU/product, and `down_proj`.
The down-projection result is added to the residual and immediately prepares
the *next* layer's `input_layernorm`; the last layer prepares the final
`norm` and the vocabulary head instead. Thus the next layer receives
already prepared int8 input and normalization statistics.
(`src/engine35.cpp:757-799`, `src/engine35.cpp:824-831`,
`src/engine35.cpp:733-736`.)

```text
hidden = decode_embedding(token)
prepared, inverse_rms = prepare_main_normalization(hidden, layer[0].input_norm)

for layer_index = 0 .. layer_count-1:
    branch_result = attention_or_gated_deltanet(prepared, inverse_rms)
    hidden = residual_add(hidden, branch_result)
    prepared, inverse_rms = prepare_main_normalization(hidden, layer[layer_index].post_attention_norm)

    gate_result = compressed_projection(layer[layer_index].gate, prepared, inverse_rms)
    up_result   = compressed_projection(layer[layer_index].up,   prepared, inverse_rms)
    mlp_input = gated_product(gate_result, up_result)
    prepared_down = quantize_activation(mlp_input, layer[layer_index].down.input_signs)
    down_result = compressed_projection(layer[layer_index].down, prepared_down, no_inverse_rms)
    hidden = residual_add(hidden, down_result)

    next_weight = layer[layer_index+1].input_norm if layer_index+1 < layer_count else final_norm
    prepared, inverse_rms = prepare_main_normalization(hidden, next_weight)
```

This is the logical order; `branch_result` and `down_result` include their
compressed operator's output rotation. The C++ implementation fuses each
residual update with preparation of the following input. Default `opt_fin_`
and `opt_flow_` are zero, so their alternative fused-GEMV/MLP kernels do not
define shipping behavior. (`src/engine35.cpp:559-565`,
`src/engine35.cpp:665-674`, `src/engine35.cpp:795-799`,
`src/engine35.hpp:292-297`.)

## Residual addition

Let `projection[i]` be the binary32 output after scale restoration and inverse
output rotation from chapter 02. The exact two bfloat16 roundings are:
(`src/q35_kernels.cu:1437-1443`.)

```text
for i = 0 .. 5119:
    rounded_projection = bfloat(projection[i])
    hidden[i] = bfloat(add32(hidden[i], rounded_projection))
```

Rounding only the final sum, retaining the projection at full precision, or
using binary16 here would change the algorithm. Initial hidden values instead
come directly from the bfloat16 embedding row, with no residual addition.
(`src/q35_kernels.cu:1440-1443`, `src/q35_kernels.cu:1548-1554`.)

## Exact sum-of-squares reduction

For each 512-element block, the code first makes 128 partial sums, each over
four elements spaced 128 apart. It then uses a five-stage XOR reduction
within each consecutive group of 32 partial sums, in offsets
`16, 8, 4, 2, 1`, and finally adds four group results from zero in ascending
group order. This reduction shape is numerically relevant even in a scalar
implementation. (`src/q35_kernels.cu:1447-1458`.)

```text
block_sum_squares(values[512]):
    for i = 0 .. 127:
        partial[i] = +0.0
        for j = 0 .. 3:
            x = values[i + 128*j]
            partial[i] = add32(partial[i], mul32(x, x))

    for group = 0 .. 3:
        lanes[0..31] = partial[group*32 .. group*32+31]
        for offset in [16, 8, 4, 2, 1]:
            previous[0..31] = lanes[0..31]
            for lane = 0 .. 31:
                lanes[lane] = add32(previous[lane], previous[lane XOR offset])
        group_sum[group] = lanes[0]

    result = +0.0
    for group = 0 .. 3:
        result = add32(result, group_sum[group])
    return result

for block = 0 .. 9:
    sum_squares[block] = block_sum_squares(hidden[block*512 .. block*512+511])
total = +0.0
for block = 0 .. 9:
    total = add32(total, sum_squares[block])
inverse_rms = upstream_rsqrtf(add32(div32(total, 5120.0), epsilon))
```

Only ten block sums are used; the upstream storage stride is 16 floats per
row. `epsilon` is read from the model header, not taken from this chapter as
an independently fixed value. The last three lines mirror `q35_xsm_r` and
use the vendor reciprocal-square-root function under the shipping math
configuration. (`src/q35.hpp:13-20`, `src/q35_kernels.cu:81-85`,
`src/engine35.cpp:214-218`, `src/engine35.cpp:458-465`.)

## Deferred normalization

The main integer-sum path does **not** form a bfloat16 RMS-normalized hidden
vector and then quantize it. It records the sum-of-squares above, forms the
following unnormalized input, and applies chapter 02's rotation and int8
quantization to that input:
(`src/q35_kernels.cu:1447-1463`.)

```text
for i = 0 .. 5119:
    normalization_gain = add32(1.0, float32(norm_weight_bfloat16[i]))
    unnormalized_input[i] = mul32(hidden[i], normalization_gain)
    if the_Gated_DeltaNet_a_b_projection_needs_a_copy:
        gate_input_bfloat16[i] = bfloat(unnormalized_input[i])

prepared = quantize_activation(unnormalized_input, next_projection_input_signs)
```

The optional gate-input copy has its own bfloat16 rounding. That rounding is
**not** applied to `unnormalized_input` before its Hadamard transform. Later,
compressed projections that consume this normalization restore scales in the
following order:
(`src/q35_kernels.cu:1459-1463`, `src/q35_kernels.cu:1409-1417`.)

```text
value = mul32(float32(weighted_integer_sum), row_quantum)
value = mul32(value, inverse_rms)
value = mul32(value, effective_row_scale)
value = inverse_output_rotation(value)  # a block operation, not scalar multiplication
```

This inverse RMS is supplied to `qkv/z`, attention `q/k/v`, MLP `gate/up`
and the final output head. It is not supplied to attention/Gated DeltaNet
output projections or MLP `down`, whose inputs have already been produced by
their preceding nonlinear/post steps. The small uncompressed `a/b` projection
uses the optional gate-input copy and separately consumes the same hidden
sum-of-squares; its arithmetic is specified in chapter 03.
(`src/engine35.cpp:682-686`, `src/engine35.cpp:757-781`,
`src/engine35.cpp:665-673`, `src/engine35.cpp:803-813`,
`src/q35_kernels.cu:755-781`.)

## MLP nonlinear steps

First restore scales and output coordinates of `gate_proj` and `up_proj`
exactly as in chapter 02. For each of the header's `ffn` coordinates, execute
the following. The source forces a bfloat16 conversion of each projection,
of the SiLU result, and of their product. There is no bias addition here.
(`src/engine35.cpp:757-765`, `src/q35_kernels.cu:1402-1421`.)

```text
for i = 0 .. ffn-1:
    gate = bfloat(gate_projection[i])
    up   = bfloat(up_projection[i])
    exponential = upstream_expf(-gate)
    denominator = add32(1.0, exponential)
    silu = bfloat(div32(gate, denominator))
    mlp_input[i] = bfloat(mul32(silu, up))

prepared_down = quantize_activation(mlp_input, down_input_signs)
```

The SiLU helper uses `expf`, not the fast `__expf` intrinsic used by some
attention paths. After `down_proj`, apply the two-rounding residual rule and
prepare the following layer as above. (`src/q35_kernels.cu:113-116`,
`src/q35_kernels.cu:1416-1421`, `src/engine35.cpp:796-799`.)

## Differences by row count

For 1–16 rows, normalization-consuming post steps read signed-64 sums and
apply inverse RMS themselves. Above 16 rows, PFA/G6 perform the identical
`float32(sum)`, row-quantum and inverse-RMS multiplications before the consumer
reads the binary32 result. The consumer's `mode == 3` then multiplies only by
the effective row scale. Output projections use `mode == 1`, without inverse
RMS. This is a representation difference, not permission to omit or repeat
normalization. (`src/engine35.hpp:253-255`, `src/engine35.cpp:413-419`,
`src/q35_kernels.cu:89-101`, `src/q35_kernels.cu:1409-1412`,
`src/pfa.cu:301-310`, `src/q35_g6.cu:265-270`.)

With shipping `YSKN_Q35_PFALT=1`, calls above 64 rows use alternate residual
and MLP post kernels. Their scalar operation order matches the definitions
above: the alternate sum-of-squares uses indices `lane + 32*(group + 4*j)`,
the same indices as `partial[group*32 + lane]` above; its XOR offsets and
four-group fold are unchanged. Its Hadamard stages are also the same
ascending strides. (`python/yasakani/runtime.py:27-43`,
`src/q35_kernels.cu:247-271`, `src/q35_kernels.cu:275-298`,
`src/q35_kernels.cu:323-369`, `src/q35_kernels.cu:602`,
`src/q35_kernels.cu:1423-1425`, `src/q35_kernels.cu:1465-1467`.)

The older `Engine` path instead has binary16 residual and normalization
helpers. They are not the semantics of `Engine35`: dispatch selects
`Engine35` for `arch == "qwen3_5"`, and its calls use the `q35_` post steps
specified here. (`src/c_api.cpp:15-18`, `src/post_kernels.cu:100-115`,
`src/engine35.cpp:559-565`, `src/engine35.cpp:757-765`.)

## Unresolved numeric boundary

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
