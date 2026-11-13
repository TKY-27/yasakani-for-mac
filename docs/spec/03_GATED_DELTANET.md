# 03. Gated DeltaNet

This chapter specifies the `Engine35` recurrent layer. It uses 16 query/key heads and 48 value heads, each 128 elements wide; each group of three value heads shares one query/key head. Its persistent matrix is FP32, while many values entering and leaving it are rounded to BF16. The 512-element compressed linear operator and its activation quantizer are specified in [02_COMPRESSED_LINEAR.md](02_COMPRESSED_LINEAR.md). Sources: `src/q35.hpp:13-14`, `src/q35_kernels.cu:905-928`, `src/engine35.cpp:255-263`.

## Arithmetic notation

`add`, `sub`, `mul`, and `div` below mean one binary32 operation, rounded to nearest with ties to even, without contracting a multiply and add. `bf(x)` rounds to BF16 with ties to even and widens back to binary32; a BF16 array stores those 16-bit values. Upstream explicitly uses the `_rn` arithmetic intrinsics and implements `bf` as `bf2f(f2bf(x))`. Sources: `src/q35_kernels.cu:35-37`, `src/q35_kernels.cu:877-896`, `src/q35_kernels.cu:939-947`.

The following helpers fix the reduction order; replacing them with a left-to-right sum changes the specified arithmetic. `sum4` has no leading addition to zero. Sources: `src/q35_kernels.cu:778-784`, `src/q35_kernels.cu:889-895`.

```text
xor_sum32(values[32]):
    a = copy(values)
    for offset in [16, 8, 4, 2, 1]:
        old = copy(a)
        for i in 0..31:
            a[i] = add(old[i], old[i XOR offset])
    return a[0]

sum4(v[4]):
    return add(add(add(v[0], v[1]), v[2]), v[3])

squares128(x[128]):
    for group in 0..3:
        p[group] = xor_sum32([mul(x[32*group+i], x[32*group+i]) for i in 0..31])
    return sum4(p)
```

The live nonlinear helpers are `exp(x) = expf(x)`, `rsqrt(x) = rsqrtf(x)`, `silu(x) = div(x, add(1, expf(-x)))`, `sigmoid(x) = div(1, add(1, expf(-x)))`, and `softplus(x) = x if x > 20 else log1pf(expf(x))`. Portable substitutes are selected only when `q35_pm_on` is enabled. The shipping defaults leave that switch off. Sources: `src/q35_kernels.cu:42-44`, `src/q35_kernels.cu:113-116`, `src/engine35.hpp:289-289`, `src/engine35.cpp:363-364`, `python/yasakani/runtime.py:27-43`.

**UNCERTAIN [GDN-01]:** The repository names CUDA's native `expf`, `rsqrtf`, and `log1pf`, but does not specify their output bits for every binary32 input. A Rust scalar reference cannot establish CUDA bit identity by substituting its platform math library. The portable-math branch is a different configuration, not evidence for the shipping branch's bits. Checked: the wrappers and their callers at `src/q35_kernels.cu:42-44`, `src/q35_kernels.cu:113-116`, `src/q35_kernels.cu:874-896`, `src/q35_kernels.cu:934-947`, and the configuration at `src/engine35.cpp:363-364`.

## Inputs, tensors, and retained state

The names below are suffixes below the layer prefix. Matrix dimensions are `[output, input]`; state dimensions are in storage order. The persistent matrix starts at zero, and all three convolution history rows start at BF16 zero. Sources: `src/engine35.cpp:255-263`, `src/engine35.cpp:392-400`, `src/q35_kernels.cu:864-880`, `src/q35_kernels.cu:910-914`.

| Tensor or array | Shape and meaning | Evidence |
|---|---|---|
| `linear_attn.in_proj_qkv` | Compressed `[10240,5120]`; projected query 2048, key 2048, value 6144, in that order | `src/engine35.cpp:256-256`; `src/q35_kernels.cu:864-880` |
| `linear_attn.in_proj_z` | Compressed `[6144,5120]`; output gate input | `src/engine35.cpp:256-256`; `src/q35_kernels.cu:827-851` |
| `linear_attn.in_proj_a`, `linear_attn.in_proj_b` | Dense BF16 `[48,5120]`; log-decay and update-gate projections | `src/engine35.cpp:257-257`; `src/q35_kernels.cu:749-784`; `src/q35_kernels.cu:882-886` |
| `linear_attn.conv1d` | BF16 `[10240,4]`, oldest coefficient first | `src/engine35.cpp:258-258`; `src/q35_kernels.cu:867-878` |
| `linear_attn.A_log`, `linear_attn.dt_bias` | BF16 `[48]` each | `src/engine35.cpp:258-259`; `src/q35_kernels.cu:871-886` |
| `linear_attn.norm` | BF16 `[128]`, shared across 48 value heads | `src/engine35.cpp:259-259`; `src/q35_kernels.cu:964-977` |
| `linear_attn.out_proj` | Compressed `[5120,6144]` | `src/engine35.cpp:256-256`; `src/q35_kernels.cu:964-980`; `src/engine35.cpp:791-793` |
| `state[h,k,j]` / upstream `S` | FP32 `[48,128,128]`; value-head, key channel, value channel | `src/engine35.cpp:260-260`; `src/q35_kernels.cu:910-914` |
| `history[age,c]` / upstream `hist` | BF16 `[3,10240]`; raw projected rows, oldest to newest | `src/engine35.cpp:260-260`; `src/q35_kernels.cu:916-924` |

For the normalized layer input, the shipping XSUM path keeps `weighted_hidden_bf16` in upstream `xn_` and separately keeps the ten 512-element sums of squares. The input has already received the regular layer-normalization weight, but the inverse RMS is deferred into the projection consumer. Its formation is specified in [05_RESIDUAL_NORM_MLP.md](05_RESIDUAL_NORM_MLP.md); do not normalize it a second time before the dense gate dot products below. Sources: `src/q35_kernels.cu:81-85`, `src/q35_kernels.cu:230-230`, `src/engine35.cpp:773-781`.

## Projected input and dense gates

Compute the compressed query/key/value and `z` projections by chapter 02, including the inverse signed normalized Hadamard transform. Round each query/key/value coordinate to BF16 and store it in `projected[row,10240]`. Round `z` to BF16 too, even though its array is FP32. In the ordinary XSUM consumer the order before the inverse transform is `mul(mul(integer_result_times_common_scale, inverse_rms), output_scale)`. For the long-row mode that has already applied the inverse RMS, multiply only by `output_scale` here. Sources: `src/q35_kernels.cu:823-851`, `src/q35_kernels.cu:300-319`.

The two dense gate projections deliberately have a specified partitioned dot product. The following computes each of the 96 outputs independently. The first 48 outputs are `b`; the next 48 are `a`. The inverse RMS uses ten block sums, summed in increasing block order starting with FP32 zero. Sources: `src/q35_kernels.cu:81-85`, `src/q35_kernels.cu:746-784`, `src/q35_kernels.cu:789-820`.

```text
norm_sum = 0
for block in 0..9:
    norm_sum = add(norm_sum, hidden_square_sum[row,block])
inverse_rms = rsqrt(add(div(norm_sum, 5120), model_epsilon))

for output in 0..95:
    weight = in_proj_b[output] if output < 48 else in_proj_a[output-48]
    for p in 0..127:
        partial[p] = 0
        for chunk in 0..4:
            for e in 0..7:
                i = 8*p + 1024*chunk + e
                partial[p] = add(partial[p], mul(weight[i], weighted_hidden_bf16[row,i]))
    for group in 0..3:
        grouped[group] = xor_sum32(partial[32*group : 32*group+32])
    gates[row,output] = bf(mul(sum4(grouped), inverse_rms))
```

At up to 32 rows these projections are fused with the compressed-output consumer; above 32 they are called separately. The fusion preserves the dot-product order above. The alternative prefill gate routine also uses five groups of eight successive products per partial and the same reduction order. Sources: `src/engine35.cpp:771-784`, `src/q35_kernels.cu:400-447`, `src/q35_kernels.cu:746-820`.

## Four-tap convolution and nonlinear gates

For an ordinary causal pass, `depth[row] = row`. For a verification pass with a row table, `depth[row]` is its first 16-byte section. The current input is always the physical `projected[row]`; the previous three inputs are taken from the contiguous trunk at `depth[row]-3 .. depth[row]-1`, or the retained history for negative indices. This is the implemented ancestry rule; the kernel does not follow arbitrary parent pointers. Sources: `src/q35_kernels.cu:867-878`, `src/lfrow_kernels.cuh:31-50`.

The convolution below is independent for each of the 10240 channels. There is no convolution bias. The extra BF16 rounding before SiLU is required. Sources: `src/q35_kernels.cu:867-878`, `src/lfrow_kernels.cuh:38-50`.

```text
past(r,c):
    return history[3+r,c] if r < 0 else projected[r,c]

for row in 0..rows-1:
    d = depth[row]
    for c in 0..10239:
        a = mul(conv_weight[c,0], past(d-3,c))
        a = add(a, mul(conv_weight[c,1], past(d-2,c)))
        a = add(a, mul(conv_weight[c,2], past(d-1,c)))
        a = add(a, mul(conv_weight[c,3], projected[row,c]))
        convolved[row,c] = bf(silu(bf(a)))
```

Split `convolved` into query `[16,128]` at offset 0, key `[16,128]` at offset 2048, and value `[48,128]` at offset 4096. Query/key normalization divides by the vector's Euclidean length with a fixed `1e-6` added to the sum of squares: it is not RMS normalization and does not use the model epsilon. Query gets a further `1/sqrt(128)` factor. The following FP32 query/key results are not rounded back to BF16. Sources: `src/q35_kernels.cu:864-866`, `src/q35_kernels.cu:879-896`.

```text
for head in 0..15:
    q = convolved[row,128*head : 128*(head+1)]
    k = convolved[row,2048+128*head : 2048+128*(head+1)]
    rq = rsqrt(add(squares128(q), 1e-6))
    rk = rsqrt(add(squares128(k), 1e-6))
    for i in 0..127:
        query[row,head,i] = mul(mul(q[i], rq), 0.08838834764831845f)
        key[row,head,i] = mul(k[i], rk)

for head in 0..47:
    value[row,head,:] = convolved[row,4096+128*head : 4096+128*(head+1)]
    beta[row,head] = bf(sigmoid(gates[row,head]))
    a = add(gates[row,48+head], dt_bias[head])
    log_decay[row,head] = mul(-exp(A_log[head]), softplus(a))
```

`beta` is a BF16-rounded value stored in FP32. `log_decay` is FP32 and is not BF16-rounded. The three projected history rows retain `projected`, never `convolved`, query, or key. Sources: `src/q35_kernels.cu:882-896`, `src/q35_kernels.cu:916-924`.

## Recurrent update and output

The routine below computes one value head and one row. It consumes a working FP32 matrix `R[128,128]`, initially copied from that head's persistent state. Both inner products use four groups of 32 sequential multiply/adds, then `sum4`. These are different reductions from `xor_sum32`. The decay multiplies the old state before the key prediction. Sources: `src/q35_kernels.cu:910-956`.

```text
recurrent_step(R, row, value_head):
    key_head = value_head // 3
    k = key[row,key_head,:]
    q = query[row,key_head,:]
    decay = exp(log_decay[row,value_head])
    update_gate = beta[row,value_head]

    for j in 0..127:
        for group in 0..3:
            prediction_part[group] = 0
            for r in 0..31:
                i = 32*group+r
                R[i,j] = mul(R[i,j], decay)
                prediction_part[group] = add(prediction_part[group], mul(R[i,j], k[i]))
        correction = mul(sub(value[row,value_head,j], sum4(prediction_part)), update_gate)
        for group in 0..3:
            output_part[group] = 0
            for r in 0..31:
                i = 32*group+r
                R[i,j] = add(R[i,j], mul(k[i], correction))
                output_part[group] = add(output_part[group], mul(R[i,j], q[i]))
        core[row,value_head,j] = bf(sum4(output_part))
    return R
```

The next step normalizes each 128-element `core` head independently, applies a shared norm weight, and gates with `silu(z)`. In this norm the stored weight is used directly, without the `1 + weight` convention of the regular layer norms. The SiLU of `z` is not rounded to BF16 before the final multiplication. Sources: `src/q35_kernels.cu:962-980`.

```text
for row, value_head:
    c = core[row,value_head,:]
    inverse_rms = rsqrt(add(div(squares128(c), 128), model_epsilon))
    for j in 0..127:
        normalized = bf(mul(gdn_norm_weight[j], bf(mul(c[j], inverse_rms))))
        gated[row,128*value_head+j] = bf(mul(normalized, silu(z[row,128*value_head+j])))
```

`gated[6144]` becomes the input to `out_proj` through its input sign, normalized 512-element Hadamard rotation, FP16 rounding, and int8 quantization from chapter 02. `out_proj` is added to the residual, then the post-attention norm and MLP follow chapter 05. Sources: `src/q35_kernels.cu:977-980`, `src/engine35.cpp:791-799`.

## Verification, branch rows, and commit

An ordinary pass runs `recurrent_step` in increasing row order on a working matrix. A provisional verification pass uses `n_commit=0`, so neither the persistent matrix nor the convolution history is changed. A committing prefill pass uses `n_commit=rows`, retaining the state after its final row. More generally the state kernel saves its working matrix exactly when `row+1 == n_commit`. Sources: `src/engine35.cpp:824-831`, `src/q35_kernels.cu:916-924`, `src/q35_kernels.cu:931-952`.

When a row table is active, its second 16-byte section gives execution order. Each byte's low seven bits select a physical row; its high bit marks a side candidate. Trunk rows update the working matrix. A side candidate computes its output from a temporary decayed/updated copy and leaves that working matrix unchanged. The following is equivalent to the explicit separate trunk/leaf expressions in the source. Row-table passes are restricted to at most 16 rows and cannot commit. Sources: `src/lfrow_kernels.cuh:79-150`.

```text
for head in 0..47:
    R = copy(persistent_state[head])
    for step in 0..rows-1:
        entry = row_table[16+step]
        row = entry & 127
        if entry & 128:
            temporary = copy(R)
            recurrent_step(temporary, row, head)
        else:
            R = recurrent_step(R, row, head)
```

These paths are live even though `DEFAULT_SPEC.lfrow=0`: the independent HCAP setting can enable the row-table option for block-draft verification. A plain table sets `depth[row]=row` and `schedule[row]=row`. HCAP construction and acceptance are specified in [08_SPECULATION.md](08_SPECULATION.md). Sources: `src/engine35.cpp:871-873`, `src/engine35.cpp:1689-1693`, `src/specx.cpp:193-197`, `src/specx.cpp:253-254`, `src/specx.cpp:357-365`, `python/yasakani/runtime.py:54-59`.

For a multi-row verification, the engine retains projected rows and all convolution outputs (`query`, `key`, `value`, `beta`, `log_decay`) per layer. On commit of `accepted_drafts`, it advances each recurrent layer again from the still-unmodified persistent state through `accepted_drafts+1` rows. With the default cached-convolution option, it does not repeat the projections, convolution, or gates. A promoted side candidate is first copied to its contiguous committed-row position, including all those cached quantities; the ordinary linear replay then applies. Sources: `src/engine35.cpp:785-790`, `src/engine35.hpp:366-366`, `src/engine35.cpp:843-851`, `src/engine35.cpp:877-899`, `src/engine35.cpp:1319-1329`.

The history update at commit is the following simultaneous replacement; preserving the old three rows is necessary when fewer than three new rows commit. Sources: `src/q35_kernels.cu:916-924`.

```text
old_history = copy(history)
for age in 0..2:
    source_row = committed_rows - 3 + age
    history[age,:] = projected[source_row,:] if source_row >= 0 else old_history[3+source_row,:]
```

A one-row verification in the speculation scope is different operationally: the default `opt_single_commit_=1` commits its state during verification. Its subsequent `commit(0)` must not replay; the engine explicitly checks this condition. This is a state-lifetime difference, not an extra recurrent update. Sources: `src/engine35.hpp:361-361`, `src/engine35.cpp:1299-1305`, `src/engine35.cpp:1319-1329`.

## Long-row and alternate paths

For `rows > 64`, `YSKN_Q35_PFALT` selects specialized gate, convolution, state, and output routines. The convolution order, four sequential 32-term state sums, BF16 boundaries, and norm/gate order above remain present in those routines. PFALT is a row-count choice, not a different recurrence. The compressed projection may use G6 and supply already-normalized FP32 output; chapter 02 defines that boundary. Sources: `src/q35_kernels.cu:400-447`, `src/q35_kernels.cu:451-518`, `src/q35_kernels.cu:602-606`, `src/q35_kernels.cu:694-741`, `src/q35_kernels.cu:372-398`, `src/q35_kernels.cu:854-858`, `src/engine35.cpp:403-431`.

The separate `q35_state_nb` overlap path belongs to the layer-major `prefill_vsp` alternative. The shipping `prefill_head=1` calls `prefill_rows` instead; do not introduce that alternate path into the shipping recurrence. Long-prompt orchestration is in [09_LONG_PROMPT_PREFILL.md](09_LONG_PROMPT_PREFILL.md). Sources: `src/q35_kernels.cu:524-600`, `src/engine35.cpp:1079-1082`, `src/engine35.cpp:1097-1145`, `src/engine35.cpp:1244-1266`, `src/specx.cpp:110-112`, `python/yasakani/runtime.py:54-59`.

The author's description of provisional checking followed by cached replay agrees with these state rules. It does not spell out the one-row precommit exception, the nonmutating side-candidate recurrence, or the BF16 boundaries; those details come from code. Sources: `docs/TECHNOLOGY.md:217-219`, `src/engine35.cpp:1299-1305`, `src/lfrow_kernels.cuh:119-140`, `src/q35_kernels.cu:877-896`, `src/q35_kernels.cu:975-977`.
