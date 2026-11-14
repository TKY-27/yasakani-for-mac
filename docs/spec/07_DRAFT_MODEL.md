# 07 — Block draft model and reduced output head

The shipping drafter is `bdraft2`, loaded separately by `yskn_load_bdraft`.
It is not the `qwen3_5_mtp` path behind `use_d3`. The model supplies proposals;
the main-model comparison and state commit in [08](08_SPECULATION.md) decide
which tokens become output. (`python/yasakani/engine.py:104-116`;
`src/c_api.cpp:117`; `src/engine35.cpp:1622-1627`;
`src/specx.cpp:170-176`)

## 1. Dimensions, tensors, and arithmetic

The following dimensions are fixed by the implementation rather than read
from draft metadata. All vector parameters are required to have `dtype=bf16`
by `Engine35::vec`; their values are expanded to binary32 before arithmetic.
The compressed matrices use [02](02_COMPRESSED_LINEAR.md), but draft
postprocessing has **no output rotation or output bfloat16 cast**.
(`src/q35_bdraft.cu:17-19`; `src/engine35.cpp:168-170`;
`src/q35_bdraft.cu:55-62`; `src/engine35.cpp:1755-1782`)

| Quantity | Required shape or value | Evidence |
|---|---|---|
| Hidden vector | 5120 | `src/q35_bdraft.cu:17` |
| Main-model feature vector | 5 concatenated vectors, 25600 elements | `src/q35_bdraft.cu:249-252`; `src/engine35.cpp:1770-1778` |
| Draft layers | 5, indexed 0 through 4 | `src/engine35.cpp:1627-1634` |
| Query heads | 32 × 128 = 4096 | `src/q35_bdraft.cu:17`; `src/engine35.cpp:1641-1642` |
| Key/value heads | 8 × 128 = 1024 | `src/q35_bdraft.cu:17`; `src/engine35.cpp:1642` |
| Draft history ring | 2048 positions per KV head, bfloat16 | `src/q35_bdraft.cu:98-103`; `src/engine35.cpp:1634` |
| MLP width | 17408 | `src/engine35.cpp:1792,1814` |
| Dynamic convolution output | 1280 = 2 × 2 × 320; 16 channels share each dynamic coefficient | `src/q35_bdraft.cu:504-519`; `src/q35_bdraft.cu:588-603` |
| Mask token | 248070 | `src/engine35.cpp:1796` |
| Block size | 8 or 16 input rows; 7 or 15 proposals | `src/q35.hpp:214`; `src/engine35.cpp:1794,1818,1851` |
| Candidate correction rank | 256 | `src/q35_bdraft.cu:17`; `src/engine35.cpp:1636-1637` |
| Reduced-head projection rank | 512 | `src/q35.hpp:215`; `src/engine35.cpp:1707-1708` |
| Reduced-head shortlist | 512 under `DEFAULT_ENV` | `python/yasakani/runtime.py:28`; `src/engine35.cpp:1664-1671` |
| Candidates reaching selection | 16 per proposal position | `src/q35_bdraft.cu:17,928-955` |

Matrix dimensions below are the caller-required output × input dimensions;
the container must supply compatible tensors. The code obtains matrix
descriptors through `up`, but it does not revalidate every dimension against
these hardcoded consumers. (`src/engine35.cpp:1624-1648`;
`src/engine35.cpp:1755-1782`; `src/engine35.cpp:1791-1834`)

| Upstream tensor name | Descriptive name | Required dimensions |
|---|---|---|
| `bdraft2.fc` | Main-feature projection | 5120 × 25600 |
| `bdraft2.hidden_norm` | Context projection norm | 5120 |
| `bdraft2.norm` | Final proposal norm | 5120 |
| `bdraft2.layers.i.q` | Query projection | 4096 × 5120 |
| `bdraft2.layers.i.k`, `.v` | Key/value projections | 1024 × 5120 each |
| `bdraft2.layers.i.o` | Attention output projection | 5120 × 4096 |
| `bdraft2.layers.i.gate`, `.up` | MLP gate/up projections | 17408 × 5120 each |
| `bdraft2.layers.i.down` | MLP down projection | 5120 × 17408 |
| `bdraft2.layers.i.akp`, `.mkp` | Attention/MLP dynamic-coefficient projections | 1280 × 5120 each |
| `bdraft2.layers.i.ln1`, `.ln2` | Attention/MLP input norms | 5120 each |
| `bdraft2.layers.i.q_norm`, `.k_norm` | Query/key head norms | 128 each |
| `bdraft2.layers.i.abase`, `.mbase` | Attention/MLP convolution bases | 4 × 5120 each |
| `bdraft2.sel.hp` | Hidden-to-correction projection | 256 × 5120 |
| `bdraft2.sel.pred`, `.succ` | Predecessor/successor correction embeddings | vocabulary × 256 each |

`f32` arithmetic below rounds after each stated operation; `fma32(a,b,c)` is
the explicitly fused binary32 `fmaf`. This differs from replacing every
multiply/add with an FMA: the Windows build disables implicit contraction,
while the explicit `fmaf` calls remain. `bf(x)` is round-to-nearest-even
bfloat16 followed by exact expansion to binary32.
(`CMakeLists.txt:47-50`; `src/q35_bdraft.cu:18-19,34,61,482`)

### Reusable reduction and draft norm

`xor_sum32(a)` below is a scalar-array definition of the source's reduction
order. Every stage reads a complete copy of the preceding array:

```text
xor_sum32(a[32]):
    for distance in [16, 8, 4, 2, 1]:
        old = copy(a)
        for i = 0 .. 31:
            a[i] = add32(old[i], old[i xor distance])
    return a[0]

draft_norm(x[n], norm_weight[n]):
    partial[256] = 0
    for lane = 0 .. 255:
        for i = lane; i < n; i += 256:
            partial[lane] = fma32(x[i], x[i], partial[lane])
    group[8] = xor_sum32(partial[32*g : 32*g+32]) for g = 0 .. 7
    total = float32(0)
    for g = 0 .. 7:
        total = add32(total, group[g])
    r = rsqrt32_cuda(add32(div32(total, float32(n)), epsilon))
    for i = 0 .. n-1:
        y[i] = mul32(mul32(r, x[i]), expand_bf16(norm_weight[i]))
    return y
```

The norm multiplies by the stored weight itself, **not** `1 + weight` as in
the main model. `y` remains binary32; the next A8 transform rounds its rotated
values to binary16, as in topic 02. (`src/q35_bdraft.cu:463-487`)

All draft matrices consume signs taken from `bdraft2.fc.signs_in`, including
the 5120-, 4096-, and 17408-element inputs. They do not select each tensor's
own `signs_in` at call time. Let `draft_linear(W,x)` mean: apply that common
sign prefix and the topic-02 A8 operation, form the LNORD int64 accumulator,
then return `mul32(mul32(float32_from_i64(acc), unit_scale), tau)` without
an inverse Hadamard. The caller can fuse these steps but preserves this
value formula. (`src/engine35.cpp:1638`; `src/engine35.cpp:1755-1764`;
`src/engine35.cpp:1799-1818`; `src/q35_bdraft.cu:474,512,552,613`)

**RESOLVED [DRF-02] (brief 05d, read-only released artifacts):**
`bdraft2_k8.yskn` has 46 compressed tensors, all eight-bit, warp layout,
with no output rotation. Every input rotation uses H512,
`mulberry32_7919_13`, seed 7, sign offset 0; its signs equal the matching
prefix of `bdraft2.fc`'s 25,600 signs (50 blocks). The input widths are
exactly 25,600 (fc), 5,120, 4,096 and 17,408.
There are **35** BF16 vector parameters. In particular, `sel.hp`,
`sel.pred` and `sel.succ` are dense BF16 vectors reshaped as the table
above, not compressed tensors (`src/engine35.cpp:1636–1637`;
`src/q35_bdraft.cu:299–305,406–411`). `draft_info` prints every descriptor
and checks the shape/sign/rotation invariants without reading payloads.
The file is 1,164,969,472 bytes; its 21,809-byte JSON header starts at byte
32, and payload at 8,388,608. No artifact bytes are stored in this repo.

## 2. Context features from the main model

Capture the residual hidden vector **after** main-model layers 5, 19, 33,
47, and 61, in that order and using zero-based layer indices. The storage
order is `capture[feature_slot, row, hidden_element]`. Capturing does not
rerun a layer or normalize its output again. The precise tap is after that
layer's MLP down-projection residual update, before the next layer or final
normalization. The FP32 array holds BF16-rounded residuals expanded exactly
to FP32: `bf(old_residual + bf(down_projection))`; do not supply the
normalized/gain-adjusted copy made for the next projection.
(`src/engine35.cpp:798–799,830–833`; `src/q35_kernels.cu:1437–1444`.)
For a single row, gather `[slot,hidden_channel]`, shape `[5,5120]`,
flattened to 25,600 values. Batch capture is `[slot,row,hidden_channel]`,
not the row-major shape used by the draft input API.
(`src/engine35.cpp:1650`; `src/engine35.cpp:830-833`;
`src/q35_bdraft.cu:249-252,490-499`)

Before a proposal at frontier position `F`, context covers committed input
positions `[max(0,F-ring_capacity),F)`. Position `F` is the pending frontier
input and is excluded. Prompt rows are inserted while prefill advances
(`src/engine35.cpp:1248-1261`). Following verification, insert exactly the
committed rows beginning at the old `F`, once each in ascending position;
neither rejected rows nor the newly selected frontier are context. Pending
committed rows are flushed even when generation stops.
(`src/specx.cpp:435-437,464`; `src/engine35.cpp:1851`)

For every newly committed context position `p`, including prompt positions:

```text
feature[slot*5120 + i] = capture[slot, local_row, i]
projected = draft_linear(feature_projection, feature)
context = draft_norm(projected, context_norm_weight)

for draft_layer = 0 .. 4:
    raw_key = draft_linear(key_projection[draft_layer], context)
    raw_value = draft_linear(value_projection[draft_layer], context)
    key = norm_rope_128(raw_key, key_norm[draft_layer], p)
    ring_key[draft_layer, head, p mod 2048, d] = bf(key[head,d])
    ring_value[draft_layer, head, p mod 2048, d] = bf(raw_value[head,d])
```

All five layers project the **same** normalized context vector; this context
update does not run five sequential draft transformer blocks. The only
persistent per-layer draft state is the key/value ring. Context batches are
split into at most 16 rows without changing this dependency structure.
(`src/q35_bdraft.cu:490-499`; `src/engine35.cpp:1767-1788`;
`src/q35_bdraft.cu:556-566`)

## 3. Normalization and rotary position transform

For every 128-element query or key head, use the following order. The
`epsilon` is the main engine's `eps_`; the frequency constant below is
hardcoded rather than taken from draft metadata.
(`src/engine35.cpp:1778-1782,1805`;
`src/q35_bdraft.cu:525-536`)

```text
norm_rope_128(x[128], weight[128], position):
    square[d] = mul32(x[d], x[d])
    group[g] = xor_sum32(square[32*g : 32*g+32]) for g = 0 .. 3
    total = add32(add32(add32(group[0],group[1]),group[2]),group[3])
    r = rsqrt32_cuda(add32(div32(total, float32(128)), epsilon))
    normalized[d] = mul32(mul32(r,x[d]),expand_bf16(weight[d]))
    for i = 0 .. 63:
        exponent = mul32(div32(-float32(2*i),float32(128)),
                         float32(23.253496664211536))
        frequency = exp2f_cuda(exponent)
        angle = mul32(float32(position),frequency)
        sine, cosine = sincosf_cuda(angle)
        out[i] = sub32(mul32(normalized[i],cosine),
                       mul32(normalized[i+64],sine))
        out[i+64] = add32(mul32(normalized[i+64],cosine),
                          mul32(normalized[i],sine))
    return out
```

The current draft block keeps Q/K/V in binary32 after this operation; only
committed context-ring storage casts K/V to bfloat16.
(`src/q35_bdraft.cu:539-566`)

## 4. One-shot block computation

For a block beginning at main-model frontier position `p`, initialize input
IDs to the frontier token in row 0 and mask token 248070 in every other row.
Obtain embeddings through the main engine's embedding source and expand
their bfloat16 elements to binary32. Let `R=8` or `16` and `J=R-1`.
(`src/engine35.cpp:1791-1798`; `src/q35_bdraft.cu:238-246`)

Each of the five layers uses two separate dynamic convolution branches:

```text
for layer = 0 .. 4:
    normalized = draft_norm(hidden, attention_norm[layer])
    dynamic = draft_linear(attention_coefficient_projection[layer], normalized)
    filtered = dynamic_filter(normalized, dynamic, attention_base[layer], 0)
    query = draft_linear(query_projection[layer], filtered)
    key = draft_linear(key_projection[layer], filtered)
    value = draft_linear(value_projection[layer], filtered)
    for row = 0 .. R-1:
        query[row] = norm_rope_128(query[row], query_norm[layer], p+row)
        key[row] = norm_rope_128(key[row], key_norm[layer], p+row)
    attended = draft_attention(query,key,value,ring_key[layer],ring_value[layer],p)
    projected = draft_linear(attention_output_projection[layer], attended)
    hidden = add32(hidden,
                   dynamic_filter(projected,dynamic,attention_base[layer],1))

    normalized = draft_norm(hidden, mlp_norm[layer])
    dynamic = draft_linear(mlp_coefficient_projection[layer], normalized)
    filtered = dynamic_filter(normalized,dynamic,mlp_base[layer],0)
    gate = draft_linear(gate_projection[layer],filtered)
    up = draft_linear(up_projection[layer],filtered)
    product = mul32(div32(gate,add32(1,exp_fast32(-gate))),up)
    projected = draft_linear(down_projection[layer],product)
    hidden = add32(hidden,
                   dynamic_filter(projected,dynamic,mlp_base[layer],1))

for j = 0 .. J-1:
    proposal_hidden[j] = draft_norm(hidden[j+1],final_norm)
```

All array operations are elementwise or per row as appropriate. No
bfloat16 rounding is inserted at the residual additions, GLU, dynamic
coefficient outputs, or final normalized vector; the common A8 operation
still supplies its binary16 rounding before int8 conversion. Row 0's final
hidden vector is omitted from the proposal head.
(`src/engine35.cpp:1799-1818`; `src/q35_bdraft.cu:504-522`;
`src/q35_bdraft.cu:588-617`)

For `branch=0` before the projection and `branch=1` after the projection,
the dynamic filter is exactly:

```text
dynamic_filter(x[R,5120], dynamic[R,1280], base[4,5120], branch):
    for row = 0 .. R-1:
        for channel = 0 .. 5119:
            group = floor(channel / 16)
            a = add32(expand_bf16(base[2*branch,channel]),
                      dynamic[row,640*branch+group])
            y[row,channel] = mul32(a,x[row,channel])
            if row > 0:
                b = add32(expand_bf16(base[2*branch+1,channel]),
                          dynamic[row,640*branch+320+group])
                y[row,channel] = add32(y[row,channel],mul32(b,x[row-1,channel]))
    return y
```

The coefficient for the preceding row comes from the **current** row's
dynamic vector. Row 0 has no preceding-row contribution; the context ring
does not supply a convolution history. Post-projection filtering reads all
unmodified projected rows before adding into the residual.
(`src/q35_bdraft.cu:504-522`; `src/q35_bdraft.cu:588-604`)

## 5. Draft attention, including its different dot-product orders

Each query head `h` uses KV head `floor(h/4)`. Historical keys range from
`max(0,p-2047)` to `p-1`; query row `m` further masks out historical positions
less than `p+m-2047`. The **entire current block**, all `R` keys and values,
is visible to every query row, including keys after that row. It is not a
causal decoder over the masked rows.
(`src/q35_bdraft.cu:113-123,135-150,154-181`)

Attention has 33 independent partial results: 32 chronological history
chunks plus one current-block partial. This decomposition changes floating
rounding and is part of the numeric contract.
(`src/q35_bdraft.cu:113-126,154-214,217-228`)

For history, `history_start=max(0,p-2047)`, `history_count=p-history_start`,
and `chunk_size=ceil(history_count/32)`. History partial `s=0..31` covers
local history indices `[s*chunk_size,min(history_count,(s+1)*chunk_size))`.
For one query row/head and that chunk:

```text
for local_key = 0 .. chunk_length-1:
    absolute_key = history_start + s*chunk_size + local_key
    acc = float32(0)
    for pair = 0 .. 63:
        acc = fma32(query[2*pair], expand_bf16(ring_key[2*pair]), acc)
        acc = fma32(query[2*pair+1], expand_bf16(ring_key[2*pair+1]), acc)
    score[local_key] = mul32(acc,float32(0.08838834764831845))
    if absolute_key < p + row - 2047:
        score[local_key] = -infinity
```

At most 64 keys occur in one chunk. Pad scores to 64 with `-infinity`.
First take each pair `(score[lane],score[lane+32])`'s maximum, then a
32-element XOR max reduction at distances `16,8,4,2,1`; this gives `maximum`.
For each nonmasked score compute `e=exp_fast32(score-maximum)`, otherwise
`e=0`; when the maximum is `-infinity`, all `e` are zero. Form the partial
denominator by `xor_sum32(e[0:32]+e[32:64])`, where each pair addition is
binary32. For each value element, start at zero and apply
`output=fma32(e[j],expand_bf16(ring_value[j,d]),output)` in ascending local
key order. Empty chunks produce `maximum=-infinity`, denominator 0, and
output 0. (`src/q35_bdraft.cu:154-207`)

The current-block partial uses a different dot-product order and online
softmax. For each query/head, first set
`scaled_query[d]=mul32(query[d],0.08838834764831845f)`. Then:

```text
maximum = -infinity; denominator = 0; output[128] = 0
for key_row = 0 .. R-1:
    for lane = 0 .. 31:
        d = 4*lane
        local[lane] = fma32(scaled_query[d],key[key_row,d],
                      fma32(scaled_query[d+1],key[key_row,d+1],
                      fma32(scaled_query[d+2],key[key_row,d+2],
                            mul32(scaled_query[d+3],key[key_row,d+3]))))
    score = xor_sum32(local)
    new_maximum = max(maximum,score)
    carry = exp_fast32(sub32(maximum,new_maximum))
    weight = exp_fast32(sub32(score,new_maximum))
    denominator = fma32(denominator,carry,weight)
    for d = 0 .. 127:
        output[d] = fma32(output[d],carry,mul32(weight,value[key_row,d]))
    maximum = new_maximum
```

The historical dot product scales **after** a serial 128-term FMA sum;
the current-block product scales Q **before** 32 four-element partial sums.
Neither may be replaced by the other when matching bits.
(`src/q35_bdraft.cu:127-150,163-181`)

Combine the partials in order `s=0..32`. Find the largest partial maximum,
starting from `-infinity`; then, starting with zero denominator and output,
skip partials whose maximum is `-infinity` and apply:

```text
weight = exp_fast32(sub32(partial_maximum[s],largest_maximum))
denominator = fma32(partial_denominator[s],weight,denominator)
output[d] = fma32(partial_output[s,d],weight,output[d])
attended[d] = div32(output[d],denominator)       # after all 33 partials
```

The fused live implementation feeds `attended * input_sign` directly to
the A8 rotation; it does not first quantize `attended` to bfloat16.
(`src/q35_bdraft.cu:569-584`; `src/engine35.cpp:1806-1808`)

## 6. Reduced `.shead` computation

The byte layout is specified in [10](10_AUXILIARY_FILES.md). Its five arrays
are `projection[512,5120]` in binary32, `approx_weight[V,512]` in signed int8,
`approx_scale[V]` in binary32, `rescore_weight[V,5120]` in signed int8, and
`rescore_scale[V]` in binary32. Loading explicitly named `.shead` data and
`YSKN_BDR_SHEAD_M=512` selects this path in shipping.
(`src/engine35.cpp:1664-1671,1705-1734`; `python/yasakani/runtime.py:28,85`)

### 6.1 Rank-512 projection

For each proposal position and projection row, initialize 32 partials to
zero. For each hidden block `block=0,512,...,4608`, then `u=0..3`, and within
each group for components `component=0..3`, use index
`i=block+4*(lane+32*u)+component` and update:

```text
partial[lane] = fma32(proposal_hidden[i],projection[rank,i],partial[lane])
low_rank[rank] = xor_sum32(partial)       # after all blocks
```

The kernel also computes unused proposal rows up to its fixed capacity 15;
only the requested `J` rows are scored and selected.
(`src/q35_bdraft.cu:767-795,964-974`)

### 6.2 Approximate score and shortlist boundary

For token `v`, each of 32 partials accumulates its **contiguous** 16 low-rank
components in ascending order, with exact conversion of signed int8 to
binary32:

```text
partial[32] = 0
for lane = 0 .. 31:
    for component = 0 .. 15:
        i = 16*lane + component
        partial[lane] = fma32(low_rank[i],float32(approx_weight[v,i]),partial[lane])
approx_logit[v] = mul32(xor_sum32(partial),approx_scale[v])
```

The source's interleaved reduction selects different proposal rows at
intermediate stages but performs the same `16,8,4,2,1` addition tree per
row. Its byte-permutation conversion exactly maps each signed int8 value to
that integer as binary32. (`src/q35_bdraft.cu:741-743,808-851`)

The shortlist is **not an exact top-512 float sort**. Define a monotone
unsigned key from a binary32 bit pattern:

```text
ordered_key(x):
    u = bits_u32(x)
    return bitwise_not_u32(u) if sign_bit(u) else (u or 0x80000000)
```

Build a 1024-bin histogram of `ordered_key(score)>>22`; select the descending
bin containing the 512th item. For that bin only, build a 2048-bin histogram
of `(ordered_key(score)>>11)&2047` and find the corresponding descending
sub-bin. Let `cutoff` be the combined 21-bit prefix and `above` the number
of items whose prefix is greater. Include all items above the cutoff and
fill the remaining `512-above` positions with items at the cutoff.
(`src/q35_bdraft.cu:739-763,849-896`)

**PORT DECISION [DRF-01], D15:** The source's cutoff membership is
scheduling-dependent (`src/q35_bdraft.cu:882–896,958–962`): cutoff items
claim slots using `atomicAdd`, without a token-ID or low-mantissa tiebreak.
The port includes all prefixes above the cutoff, then fills the cutoff
bucket in ascending token ID order. This is an explicit deterministic
choice within upstream's possible subsets, not a recovered GPU schedule.
It does not replace the histogram boundary with an exact float sort.

### 6.3 Full-width int8 rescoring and top 16

For each shortlisted token, accumulate over all 5120 hidden elements. Each
of 32 partials visits groups `t=0..39`, with four elements at
`i=4*(lane+32*t)+component` in component order `0,1,2,3`:

```text
partial[lane] = fma32(proposal_hidden[i],float32(rescore_weight[v,i]),partial[lane])
rescore_logit[v] = mul32(xor_sum32(partial),rescore_scale[v])
```

Select 16 candidates from these 512 by decreasing rescore logit, breaking
ties by smaller token ID. This final ordering does not repair an ambiguous
shortlist membership. The source also zeros both histograms for the next
invocation during rescoring.
(`src/q35_bdraft.cu:900-955`)

`TECHNOLOGY.md` describes the shortlist as being "scored exactly". The live
code rescoring is exact with respect to this **separate signed-int8 matrix
and its scales**, using binary32 FMA reductions; it does not evaluate the
main compressed output head for those IDs. Whether the supplied int8
rescore matrix equals any other head representation cannot be inferred
without the model files. (`docs/TECHNOLOGY.md:208-210`;
`src/q35_bdraft.cu:900-924`; `src/engine35.cpp:1819-1831`)

## 7. Predecessor-dependent candidate correction

For each proposal hidden vector, compute a 256-element correction feature:

```text
for rank = 0 .. 255:
    partial[32] = 0
    for lane = 0 .. 31:
        for i = lane; i < 5120; i += 32:
            partial[lane] = fma32(proposal_hidden[i],expand_bf16(hidden_projection[rank,i]),partial[lane])
    correction_feature[rank] = xor_sum32(partial)
```

Then process proposal positions in order, beginning with the frontier
token as `predecessor`. For every one of the 16 candidates `v`:

```text
for rank = 0 .. 255:
    product[rank] = mul32(expand_bf16(predecessor_embedding[predecessor,rank]),
                          correction_feature[rank])
partial[32] = 0
for lane = 0 .. 31:
    for rank = lane; rank < 256; rank += 32:
        partial[lane] = fma32(product[rank],expand_bf16(successor_embedding[v,rank]),partial[lane])
corrected_score[v] = add32(rescore_logit[v],xor_sum32(partial))
```

After choosing a candidate as below, use it as the next position's
predecessor. The five transformer layers run once for the whole block; only
this 16-candidate correction/selection proceeds autoregressively.
(`src/q35_bdraft.cu:299-305,393-411,447-457`;
`src/engine35.cpp:1833-1834`)

## 8. Draft choice and side candidates

When the main sampler is enabled with version 3, the draft uses its own
`draft_temperature=mul64(float64(0.001),float64(850))` by default. Preserve
this binary64 multiplication; unlike the main option setter it does not
divide by `1000.0`, and it need not equal the binary64 literal `0.85`.
Only enabled main sampling **version 3** supplies the seed and absolute
position for coupled selection; version 2 and disabled sampling select greedily.
It does not copy the main request's temperature, top-k,
top-p, or min-p. The draft has exactly 16 candidates and hardcoded top-p
`0.95`, with no min-p filter.
(`src/engine35.hpp:183`; `src/engine35.cpp:948,1845-1850`;
`src/q35_bdraft.cu:404-430`)

In greedy mode choose the highest corrected score, breaking equal scores
by the smaller **candidate index** in the pre-correction top-16 list. In
sampled mode, reorder by decreasing corrected score then smaller token ID.
Compute binary64 exponential weights relative to the largest corrected
score; keep a candidate iff the sum of weights **before it** is strictly
less than `0.95*total`. Add the same token-keyed Gumbel value as topic 06,
using position `frontier_position+1+proposal_index`. Select the highest
perturbed value, with earlier corrected-score rank winning ties.
(`src/q35_bdraft.cu:401-448`)

Store the second and third choices under that same selection ordering as
side candidates. Store the top-to-second margin as binary32; for sampled
selection the subtraction happens in binary64 before conversion. If no
second choice has score greater than `-INFINITY`, use alternative ID `-1`
and margin `1e30f`. This condition admits `+INFINITY`; subtracting two
positive infinities can make the margin NaN. The source does not clamp
that result; see SMP-03 for a way the shared Gumbel formula can produce
positive infinity.
These feed HCAP in topic 08, which may shorten the main chain to make space
for side candidates.
(`src/q35_bdraft.cu:435-453`; `src/engine35.cpp:1834-1835`)

## 9. State timing, inactive alternatives, and open items

`bdraft_step` first inserts newly committed main-model context features
into the rings, then optionally computes a new 8- or 16-row block. The main
verification's unaccepted rows are never inserted into the draft ring.
At the end of generation, pending committed context is still inserted even
when no more proposals are needed. The rings are included in prompt
snapshots. (`src/engine35.cpp:1838-1859`; `src/specx.cpp:435-437,464`;
`src/engine35.cpp:1564-1565,1578-1579`)

With `.shead` disabled, the engine can score an optional compressed subset
head (`bdraft2.head_sub` and int32 ID mapping, or an external `mtp.head`) or
the main compressed head; it then takes 16 candidates and applies the same
correction and selector. Shipping explicitly loads `.shead` and sets its
shortlist width to 512, so those output-head alternatives are inactive.
(`src/engine35.cpp:1651-1659,1664-1671,1819-1834`)

**UNCERTAIN [DRF-03]: Draft CUDA math bits.** Draft normalization, RoPE and
attention call `rsqrtf`, `exp2f`, `sincosf` and `__expf`; coupled selection
also calls binary64 `exp` even when the main portable-math flag is enabled.
The source fixes their arguments and the reduction orders above but does
not provide the shipping CUDA implementations. D15 explicitly chooses the port's portable `rsqrt`, `exp2`, `sincos`,
`exp`, and binary64 `exp_d`/`log_d` instead, while preserving every explicit
source FMA as `f32::mul_add` and keeping other operations separate. This
defines the port reference; CUDA bit equality remains unverified. Checked
`src/q35_bdraft.cu:145-148,190,463-487,525-536,579,615,426`.

**UNCERTAIN [DRF-04]: Equality to a full compressed head.** No relationship
between `.shead`'s signed-int8 rescore weights and the main/draft compressed
head is validated by the loader. A claim of bit equality to that head is
unsupported; the supplied files would be needed to test it. Checked
`src/engine35.cpp:1700-1734,1819-1831` and
`src/q35_bdraft.cu:900-924`.

## Scalar-port scope (D15)

The port implements §§1–8 for the default **eight input rows / seven
proposals** (`src/engine35.cpp:1851`), using header dimensions and explicit
mask/capture-layer parameters. It requires 128-channel heads, rank-512
SHEAD, 16 candidates and 16-channel convolution sharing. Small synthetic
models keep those operation shapes; hidden/FFN widths are multiples of
512, correction rank is a multiple of 32, and a power-of-two history ring
may be 2..2048. The same 32 history splits and one block split are retained.
The source's 16-row option, inactive compressed/subset heads, second/third
side-candidate policy and HCAP are outside this brief. Snapshot/reset and
committed-row context updates are implemented. D17 connects the main-model
capture to `DraftProposer`: capture layers come from `DraftOptions`, default
capture is off, and committed features alone update the ring. `DraftChoice`
is explicit at the API boundary; callers using main sampler version 2 must
choose `Greedy`, while version 3 may use `DraftChoice::sampled(seed)` with
that main seed. Draft temperature remains independent and top-p remains 0.95.
The `generate` example uses version 3 for `--seed` and greedy choice otherwise.
`snapshot_session` / `restore_session` save and restore both the main-model
snapshot and proposer ring; the main-model snapshot alone does not own the
separate draft ring. Reset both owners before reusing an empty session.

Source review found no contradictory equation in §§1–8 or the SHEAD byte
layout in chapter 10. The added dense-vector/tap/rounding details above
clarify omissions; DRF-01/03 are deliberate port decisions, and DRF-02 is
resolved by artifact inspection.
