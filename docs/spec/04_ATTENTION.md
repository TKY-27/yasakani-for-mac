# 04. Full attention, RoPE, and the KV cache

The full-attention layer has 24 query heads, four key/value heads, and 256 channels per head. Query head `h` uses key/value head `h // 6`. Only the first 64 query/key channels receive RoPE. The shipping build then rotates all 256 channels and uses an int8 or packed trellis cache. Sources: `src/q35.hpp:13-14`, `src/q35_atxpk.cu:390-408`, `src/q35_kernels.cu:1049-1059`, `src/kvx_codec.cuh:11-17`.

This chapter separates the cache's stored representation from the arithmetic that consumes it. They are not interchangeable: packed attention rounds its codebook to BF16, and attention applies a row scale after the query/key dot product and before rounding the probability/value operand. Sources: `src/kvq_atxpk.hpp:39-45`, `src/q35_atxpk3.cuh:343-380`, `src/q35_atxpk3.cuh:402-406`.

## Arithmetic contract and known limit

Use the `add`, `sub`, `mul`, `div`, `bf`, and `xor_sum32` definitions in [03_GATED_DELTANET.md](03_GATED_DELTANET.md). `f16(x)` below means round to IEEE binary16 with ties to even, then widen to binary32. Ordinary multiply/add expressions in the attention routines are separate operations in the shipping Windows build because CMake passes `--fmad=false`; the explicitly written `_rn` operations also forbid contraction. Sources: `src/q35_kernels.cu:35-37`, `src/q35_kernels.cu:607-608`, `src/kvq_gpu.cuh:31-32`, `src/q35_atxpk3.cuh:429-430`, `CMakeLists.txt:47-50`.

**UNCERTAIN [ATT-01]:** Prefix attention invokes `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. The source fixes operand values and the sequence of 16-element chunks, but does not define the instruction's internal addition order or all rounding behavior. Below, `mma16(acc,a[16],b[16])` names precisely this unresolved primitive. It must not silently be replaced by 16 sequential FP32 multiply/adds and called bit-exact. The optional FP16 path has the same issue with its FP16 MMA instruction. Checked: `src/q35_atxpk.cu:35-37`, `src/q35_atx.cu:30-32`, `src/q35_atx.cu:65-69`, `src/q35_atxpk3.cuh:343-355`, `src/q35_atxpk4.cuh:287-300`.

**UNCERTAIN [ATT-02]:** The output bits of CUDA's native `cosf`, `sinf`, `rsqrtf`, `expf`, and `exp2f` are not defined in this repository for all inputs. The shipping path selects those functions, not the portable-math alternatives. A scalar implementation needs a separately established compatibility primitive for each; this document preserves the exact arguments and rounding boundaries. Checked: `src/q35_kernels.cu:42-44`, `src/q35_kernels.cu:1096-1101`, `src/q35_atxpk3.cuh:396-406`, `src/q35_kernels.cu:1295-1308`, `src/engine35.cpp:363-364`, `python/yasakani/runtime.py:27-43`.

## Projection, head normalization, and RoPE

The layer loads the tensors below. The 12288 query-projection outputs are interleaved by query head: 256 query channels, then that head's 256 output-gate channels. They are not a block of all queries followed by a block of all gates. Sources: `src/engine35.cpp:240-241`, `src/q35_kernels.cu:1013-1017`, `src/q35_kernels.cu:1038-1059`.

| Tensor suffix | Shape / use | Evidence |
|---|---|---|
| `self_attn.q_proj` | Compressed `[24*512,5120]`, query and output gate per head | `src/engine35.cpp:240-240`; `src/q35_kernels.cu:1013-1013`, `src/q35_kernels.cu:1038-1038` |
| `self_attn.k_proj` | Compressed `[4*256,5120]` | `src/engine35.cpp:240-240`; `src/q35_kernels.cu:1014-1014` |
| `self_attn.v_proj` | Compressed `[4*256,5120]` | `src/engine35.cpp:240-240`; `src/q35_kernels.cu:1015-1015` |
| `self_attn.q_norm`, `self_attn.k_norm` | BF16 `[256]` each, shared across heads | `src/engine35.cpp:241-241`; `src/q35_kernels.cu:1017-1017` |
| `self_attn.o_proj` | Compressed `[5120,24*256]` | `src/engine35.cpp:240-240`; `src/q35_kernels.cu:1357-1362`; `src/engine35.cpp:735-736` |

Compute all three projections with [02_COMPRESSED_LINEAR.md](02_COMPRESSED_LINEAR.md). Apply each output scale, inverse signed normalized 512-element Hadamard transform, and BF16 rounding before the following head operations. The ordinary XSUM consumer's order is `mul(mul(integer_result_times_common_scale,inverse_rms),output_scale)`; long-row mode 3 receives the inverse RMS already applied and performs only the final scale multiply. Sources: `src/q35_kernels.cu:1021-1026`, `src/q35_kernels.cu:625-633`.

For every query head, save its raw projected gate `[256]` as BF16. Normalize only the query half used by attention. For each key head normalize its `[256]` too, using the corresponding shared norm weight. Value heads receive neither this head norm nor RoPE. The query/key norm multiplies by `1 + weight` and rounds only after both multiplications. Sources: `src/q35_kernels.cu:1027-1047`.

```text
normalize_head(y[256], weight[256]):
    for group in 0..7:
        partial[group] = xor_sum32([mul(y[32*group+i],y[32*group+i]) for i in 0..31])
    total = 0
    for group in 0..7:
        total = add(total, partial[group])
    inverse_rms = rsqrt(add(div(total,256), model_epsilon))
    for i in 0..255:
        normalized[i] = bf(mul(mul(y[i],inverse_rms),add(1,weight[i])))
    return normalized
```

The exact frequency constants are these binary32 bit patterns. Do not regenerate them from a guessed RoPE base or from the model header. For text position `p`, the table stores `bf(cosf(mul(float32(p),frequency[i])))` and `bf(sinf(mul(float32(p),frequency[i])))` as FP32 values. Sources: `src/q35_kernels.cu:1091-1104`.

```text
frequency_bits[32] = [
  0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835,
  0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
  0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d,
  0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
  0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e,
  0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
  0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c,
  0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44
]
```

RoPE pairs channel `i` with `i+32`, not with `i+1`. Both products are separately rounded to BF16 before the addition, and the sum is rounded again. The other 192 channels keep their normalized BF16 values. Sources: `src/q35_kernels.cu:1049-1053`.

```text
rope(normalized[256], position):
    rotated = copy(normalized)
    for i in 0..31:
        c = rope_cos[position,i]
        s = rope_sin[position,i]
        rotated[i]    = bf(add(bf(mul(normalized[i],c)),    bf(mul(-normalized[i+32],s))))
        rotated[i+32] = bf(add(bf(mul(normalized[i+32],c)), bf(mul(normalized[i],s))))
    return rotated
```

Ordinary rows use RoPE position `base_position + row`. With a verification row table, the RoPE position is `base_position + depth[row]`, while the physical cache slot remains `base_position + row`. This distinction allows a side candidate at the same semantic position as a trunk candidate. Sources: `src/q35_kernels.cu:1020-1020`, `src/lfrow_kernels.cuh:154-210`.

## The 256-element cache rotation

With compile mode 3, rotate each post-RoPE query and key, and each BF16 projected value, with the following signed normalized Hadamard transform. All integer hash arithmetic is unsigned 32-bit with wraparound. This rotation is separate from every compressed linear operator's 512-element rotation. Sources: `src/kvx_codec.cuh:11-17`, `src/kvx_codec.cuh:28-48`, `src/q35_kernels.cu:1027-1059`.

```text
cache_sign(channel):
    x = uint32(channel+1)
    x = x XOR (x >> 16)
    x = uint32(x * 0x7feb352d)
    x = x XOR (x >> 15)
    x = uint32(x * 0x846ca68b)
    x = x XOR (x >> 16)
    return -1 if x & 1 else 1

hadamard256(x[256]):
    a = copy(x)
    for distance in [1,2,4,8,16,32,64,128]:
        old = copy(a)
        for i in 0..255:
            a[i] = sub(old[i XOR distance],old[i]) if i & distance else add(old[i],old[i XOR distance])
    return a

rotate_cache(x[256]):
    a = hadamard256([mul(x[i],cache_sign(i)) for i in 0..255])
    return [mul(a[i],0.0625) for i in 0..255]

inverse_rotate_cache(x[256]):
    a = hadamard256(x)
    return [mul(mul(a[i],0.0625),cache_sign(i)) for i in 0..255]
```

After rotation, query coordinates are rounded to BF16 and saved as `query_bf16[row,24,256]`. Keys and values proceed directly from the FP32 rotation result to int8 quantization; there is no extra BF16 rounding at that point. Sources: `src/q35_kernels.cu:1029-1034`, `src/q35_kernels.cu:1054-1059`.

## Runtime cache representations

The scalar reference's `KvMode::Int8` is the historical plain-int8 baseline,
not the upstream published “int8” quality configuration. `Int8Profile`
selects that initial-int8/profile/final-int8 path; `Kv5` packs eligible sides
and uses `Int8Profile` elsewhere. Existing scalar loaders still default to
plain `Int8` and require a profile only for explicit profile modes (D13).
Upstream runs profile selection with packed capacity both on and off:
`src/engine35.cpp:688-704`, `src/kvq_fast.cuh:189-234`.

The cache row index is `kv_head * max_context + physical_position`. Key and value use separate payload and scale arrays. The table describes storage for one 256-element row of one key/value head; these are runtime arrays, not `.yskn` tensor records. `.yskn` is already specified in [../FORMAT.md](../FORMAT.md). Sources: `src/q35_kernels.cu:39-39`, `src/engine35.cpp:248-254`.

| Representation | Payload / side data per row | Numeric interpretation and reachability | Evidence |
|---|---|---|---|
| FP16 | 256 binary16 values, 512 bytes | `f16` of the prepared key/value; available in an alternative build/configuration | `src/q35_kernels.cu:607-608`, `src/q35_kernels.cu:1035-1035`, `src/q35_kernels.cu:1059-1059` |
| BF16 | 256 BF16 values, 512 bytes | `bf` of the prepared key/value when both 16-bit-cache flags select BF16; alternate configuration | `src/q35_kernels.cu:607-608`; `src/engine35.cpp:356-356` |
| Int8 | 256 signed bytes plus one FP32 scale, 260 bytes | Signed integer times its row scale; shipping mode 3 stores profile-requantized int8, including when packed capacity is disabled | `src/kvx_codec.cuh:50-69`; `src/engine35.cpp:250-251` |
| Packed 5-bit trellis | 160 payload bytes plus one FP32 scale, 164 bytes | 256 5-bit fields, four 64-element chains, codebook and mean from the profile | `src/kvq_fast.cuh:220-231`; `src/q35_atxpk.cu:69-74`; `src/engine35.cpp:249-251` |
| Packed 4-bit trellis | 128 payload bytes plus one FP32 scale, 132 bytes | Same procedure with four field bits, supported according to the profile | `src/kvq_atxpk.hpp:73-83`; `src/kvq_fast.cuh:220-231` |

FP16 and BF16 cache alternatives cannot be enabled by only changing the environment in the shipping mode-3 binary: construction rejects `KV8=0` or `KV_FP16!=0`. In a nonrotated build with FP16 cache, query/gate projection and normalization still have the BF16 boundaries above; the ATX routine converts its BF16 queries to FP16 and uses FP16 MMA operands and FP16-rounded probabilities. Do not replace the shipping BF16 query with FP16 merely because an overview compares cache sizes against FP16. Sources: `src/engine35.cpp:200-210`, `src/q35_atx.cu:171-183`, `src/q35_atx.cu:65-81`, `src/q35_atx.cu:811-814`.

### Initial int8 encoding

Every mode-3 key/value row first passes through the same int8 encoder, including rows that will ultimately be packed. Its scale covers one full 256-channel head, not a 32/64/128-channel subgroup. `nearest_even_int` is the `_rn` float-to-int conversion; saturation is to `[-127,127]`. Sources: `src/kvx_codec.cuh:50-69`, `src/q35_kernels.cu:1029-1034`, `src/q35_kernels.cu:1058-1059`, `src/kvq_fast.cuh:65-79`.

```text
encode_int8(x[256]):
    maximum = max(abs(x[i]) for i in 0..255)
    scale = div(maximum,127)
    for i in 0..255:
        integer[i] = clamp(nearest_even_int(div(x[i],scale)),-127,127) if scale > 0 else 0
    return integer,scale

decode_int8(integer,scale,i):
    return mul(float32(integer[i]),scale)
```

For an ultimately packed side, this int8 row is temporary staging; otherwise it is written in the persistent int8 array. The profile encoder reads the **dequantized int8** value `mul(float32(integer[i]),scale)`, not the original rotated FP32 value. The group index is `(attention_layer_ordinal*4 + kv_head)*2 + kind`, with `kind=0` for key and `1` for value. It uses physical position `base_position+row` for the profile guard. Sources: `src/engine35.cpp:689-704`, `src/q35_kernels.cu:1079-1086`, `src/kvq_fast.cuh:65-79`.

### Profile preparation and scale candidates

`kvq_u5.bin` supplies the per-group mode, bit count, table index, guard, optional mean and projection basis, candidate scale multipliers, and int16 trellis codebooks. Its on-disk layout is specified in chapter 10; the `.yskn` format does not supply these values. In the encoder, codebook entry `ci` decodes to `mul(mul(float32(ci),1/256),gain)`, without BF16 rounding. Sources: `src/kvq_fast.cuh:53-62`, `src/kvq_fast.cuh:82-95`, `src/kvq_fast.cuh:159-165`, `src/kvq_gpu.cuh:53-53`.

`sum256` below first applies `xor_sum32` to each consecutive 32-element group, then adds the eight group results in increasing order starting from zero. Preparation subtracts the FP16 mean and up to two FP16 basis vectors in order. Each basis coefficient is rounded to FP16 before subtraction. The RMS uses correctly rounded binary32 square root, not reciprocal square root. Sources: `src/kvq_fast.cuh:33-41`, `src/kvq_fast.cuh:82-95`, `src/kvq_vit2.cuh:37-49`, `src/kvq_vit2.cuh:103-138`.

```text
sum256(x[256]):
    total = 0
    for group in 0..7:
        total = add(total,xor_sum32(x[32*group : 32*group+32]))
    return total

residual = copy(dequantized_int8_row)
if profile.has_mean:
    for i in 0..255:
        residual[i] = sub(residual[i],profile.mean_f16[i])
for basis_index in 0..min(profile.basis_count,2)-1:
    basis = profile.basis_f16[basis_index]
    coefficient[basis_index] = f16(sum256([mul(residual[i],basis[i]) for i in 0..255]))
    for i in 0..255:
        residual[i] = sub(residual[i],mul(coefficient[basis_index],basis[i]))
rms = sqrt_rn(div(sum256([mul(residual[i],residual[i]) for i in 0..255]),256))
if rms == 0: rms = 1

for candidate in profile.multipliers, in file order:
    scale[candidate] = f16(mul(rms,candidate))
    if scale[candidate] == 0: scale[candidate] = 1
    for i in 0..255:
        target[candidate,i] = clamp(nearest_even_int(mul(div(residual[i],scale[candidate]),256)),-2048,2048)
```

### Integer trellis selection

The encoder selects four independent chains of 64 values. A state is eight bits, and a `bits`-bit field advances it by `state = ((state << bits) | field) & 255`, with initial state zero. The general integer dynamic program below fixes the starting cost, predecessor tie rule, ending tie rule, and traceback. It is used directly when the VIT2 eligibility check fails, and also for 3-bit groups in an otherwise eligible profile. Sources: `src/kvq_fast.cuh:99-147`, `src/kvq_fast.cuh:150-176`, `src/kvq_fast.cuh:240-246`.

```text
trellis_chain(target[64], integer_codebook[256], bits):
    fields = 1 << bits
    predecessor_stride = 256 >> bits
    cost[:] = 0x40000000
    cost[0] = 0
    for t in 0..63:
        for next_state in 0..255:
            group = next_state >> bits
            predecessor_choice = argmin_j(cost[j*predecessor_stride+group], j), j in 0..fields-1
            difference = target[t] - integer_codebook[next_state]
            new_cost[next_state] = difference*difference + cost[predecessor_choice*predecessor_stride+group]
            backpointer[t,group] = predecessor_choice
        cost = new_cost
    state = argmin_state(uint32(cost[state]), state)
    for t in 63 down to 0:
        selected_state[t] = state
        j = backpointer[t,state >> bits]
        state = (j << (8-bits)) | (state >> bits)
    return selected_state
```

`argmin(value,tie)` minimizes the value and then the integer tie argument. The per-step comparison is signed integer; the final general-program comparison converts the stored cost to unsigned before combining it with the state index. The source uses 32-bit integer arithmetic. Sources: `src/kvq_fast.cuh:112-135`.

**RESOLVED FOR THE INSPECTED PROFILE [ATT-04]:** Upstream still has
unchecked signed i32 arithmetic (`src/kvq_fast.cuh:108-123`,
`src/kvq_vit2.cuh:64-88,143-172`). Arbitrary int16 tables can overflow;
the scalar codec checks the source's intermediate products/adds/subtracts
and returns an error. For this artifact's [-886,874] tables and the source's
[-2048,2048] target clamp (`src/kvq_fast.cuh:167-168`), tests prove general
costs are at most 1,624,676,608 including finite infinity. VIT2's normalized
cost spread gives intermediate absolute bounds 474,879,999 / 923,638,527
for 4/5 bits. Overflow is therefore unreachable for these table values and
any clamped target sequence; this does not define upstream wraparound for
other profiles. See D13 and `kv_real_profile_facts_quality_and_overflow`.

Shipping enables VIT2. It is selected for 4/5-bit groups only when all such groups pass the table checks, including every int16 entry being within `[-1100,1100]`; it is not unconditionally selected just because the environment enables it. Its compressed recurrence removes a common cost every step and encodes the predecessor tie in low bits. The following is the source's actual recurrence, rather than assuming it is interchangeable under every integer overflow condition with the general program. Sources: `src/kvq_fast.hpp:46-66`, `src/kvq_fast.cuh:237-246`, `src/kvq_vit2.cuh:64-90`, `src/kvq_vit2.cuh:143-192`, `python/yasakani/runtime.py:27-43`.

```text
vit2_chain(target[64], integer_codebook[256], bits):
    F = 1 << bits
    U = 256 >> bits
    reduced_cost[0] = 0
    reduced_cost[1..U-1] = F << 24

    candidate_key(g,j,X,cost):
        d = U*j + g
        c = integer_codebook[d]
        return (F*c*c + j) + (-2*F*c)*X + cost[d >> bits]

    for t in 0..62:
        for g in 0..U-1:
            key = min(candidate_key(g,j,target[t],reduced_cost) for j in 0..F-1)
            key = key + ((target[t]*target[t]) << bits) - reduced_cost[0]
            next_cost[g] = key & ~(F-1)
            backpointer[t+1,g] = key & (F-1)
        reduced_cost = next_cost

    for g in 0..U-1:
        key = min(candidate_key(g,j,target[63],reduced_cost) for j in 0..F-1)
        key = key + ((target[63]*target[63]) << bits)
        final_cost[g] = arithmetic_shift_right(key,bits)
        final_state[g] = U*(key & (F-1)) + g
    g = argmin_g(final_cost[g],final_state[g])
    state = final_state[g]
    for t in 63 down to 1:
        selected_state[t] = state
        j = backpointer[t,state >> bits]
        state = (j << (8-bits)) | (state >> bits)
    selected_state[0] = state
    return selected_state
```

The integer objective chooses a state path for each scale candidate; the candidate itself is chosen using a different, FP32 reconstruction-error calculation. For every selected state, decode the unrounded FP32 codebook value, multiply by that candidate's FP16-rounded scale, subtract from the prepared residual, square, and reduce with `sum256`. Choose the first candidate attaining the smallest error (`<`, not `<=`). Sources: `src/kvq_fast.cuh:139-146`, `src/kvq_vit2.cuh:196-212`, `src/kvq_fast.cuh:198-212`.

```text
for candidate in file order:
    selected = concatenate(trellis_or_vit2(each consecutive 64-element chain))
    for i in 0..255:
        code = mul(mul(float32(integer_codebook[selected[i]]),1/256),gain)
        difference = sub(residual[i],mul(code,scale[candidate]))
        squared_error[i] = mul(difference,difference)
    error[candidate] = sum256(squared_error)
best = first candidate with smallest error
field[i] = selected_state[best,i] & ((1 << bits)-1)
row_scale = scale[best]
```

### Packing, decoding, and the int8 fallback

A side is packable only if all four of its heads use mode 0, the selected common bit count, guard zero, and basis count zero. A mean is permitted. The common bit count is five when the number of mode-0 five-bit groups exceeds the number of mode-0 four-bit groups, otherwise four. Thus the actual packed-layer pattern and common bit count come from the auxiliary profile. Sources: `src/kvq_atxpk.hpp:60-83`, `src/engine35.cpp:242-251`.

For a packed row, concatenate its 256 fields in element order, each field most-significant bit first. Divide that stream into 32-bit words, placing the first stream bit at bit 31 of the first word. The source stores those words as native `unsigned` values; in the supported Windows/NVIDIA environment the byte representation is little-endian words. A 64-element chain occupies `2*bits` words. The row scale is a separate FP32 array entry, even though its value was rounded to FP16. Sources: `src/kvq_fast.cuh:220-231`, `src/q35_atxpk.cu:69-74`, `src/engine35.cpp:249-251`.

```text
for word_index in 0..8*bits-1:
    word = 0
    for bit_in_word in 0..31:
        stream_bit = 32*word_index + bit_in_word
        element = stream_bit // bits
        bit_in_field = stream_bit % bits
        bit = (field[element] >> (bits-1-bit_in_field)) & 1
        word |= bit << (31-bit_in_word)
    store_u32(payload + 4*word_index,word)

for chain in 0..3:
    state = 0
    for i in 0..63:
        state = ((state << bits) | read_field(payload,64*chain+i)) & 255
        decoded_state[64*chain+i] = state
```

For attention, build `attention_codebook[state] = bf(mul(mul(float32(ci[state]),1/256),gain))`. The packed payload selects these BF16 values; key/value row scales remain separate. A key mean is not added to the query/key score. A packed value mean is added later as `partial_denominator * mean[channel]` to each partial numerator, after self and before split output; that FP32 denominator includes self and uses the original probabilities, not BF16-rounded/scaled value weights. This exactly describes the source, including the asymmetry between key and value means. Sources: `src/kvq_atxpk.hpp:39-54`, `src/q35_atxpk3.cuh:343-380,443-475`, `src/q35_atxpk4.cuh:385-417`.

For an unpackable side, mode 0 still performs profile quantization, restores the optional basis contributions in order and then the mean, and quantizes the result a second time with `encode_int8`. Mode 1 instead uses residual scale `max(abs(residual))/7` (zero replaced by one), nearest-even integer rounding clamped to `[-7,7]`, then restores basis/mean and performs that second int8 encoding. Before a group's guard position, or for a missing group, profile alteration is bypassed but the selector still sends the dequantized row through the final int8 store. Sources: `src/kvq_fast.cuh:187-218`, `src/kvq_fast.cuh:233-234`.

**RESOLVED [ATT-03] (brief 05a):** Inspection of the read-only local
17,432-byte profile establishes all 128 groups as mode 0 / five bits,
nine scale multipliers, table bounds [-886,874], and VIT2 eligibility.
All 16 key sides and value ordinals 0–9, 11–13 pack. Value ordinal 10 has
head-1 guard 1 (group 83); ordinal 14 has a head-1 basis (group 115);
ordinal 15 has basis counts 2/1/1/2. Means are permitted and occur in 21
groups, including two key groups. See chapter 10 AUX-02 for the full list.
These facts follow the consumed fields and predicates in
`src/kvq_fast.cuh:54-62`, `src/kvq_fast.hpp:51-66`, and
`src/kvq_atxpk.hpp:60-83`; they are artifact-specific, not filename promises.

## Attention selection and logical row visibility

Shipping selects ATX attention. For a layer with packed keys, `YSKN_ATX_PKV=3` uses `q35_atxpk3` for one or two rows and `q35_atxpk4` above two rows. `YSKN_ATX_PK3_PLUT=1` changes the small-row codebook lookup, not its decoded values. Query/key int8 MMA (`PK3_QK8`) is off by default. If neither side is packed, `YSKN_ATX_VSP=7` selects the ordinary int8 VSP attention routine for every row count. A packed-value-only side falls to variant 1 and is supported only for four-bit packing; a five-bit profile requiring that combination raises an error. Sources: `src/q35_atxpk.cu:682-684`, `src/q35_atxpk.cu:702-741`, `src/q35_atxpk3.cuh:480-516`, `src/q35_atxpk4.cuh:431-436`, `src/q35_atx.cu:822-826`, `python/yasakani/runtime.py:27-43`.

For each query row, the following determines its causal range, independently of whether its cache side is packed. A side candidate's self key/value is read from its own physical slot, while its predecessors are the contiguous trunk prefix. Self is handled separately even for an ordinary causal row. Sources: `src/q35_atxpk.cu:390-409`, `src/q35_atx.cu:370-377`.

```text
depth = row_table[row] if row_table exists else row
length = base_position + depth + 1
self_slot = base_position + row
split_count = 31 if length < 4096 else 42
per_split = ceil_div(length,split_count)

for split in 0..split_count-1:
    first = split * per_split
    last = min(length,first+per_split)
    prefix_end = max(first,min(last,length-1))
    prefix_slots = [first,prefix_end)   # excludes this query's semantic self
    include_self = first <= length-1 < first+per_split
```

The 31/42/4096 constants are engine defaults retained by the shipping environment; the final combining routine uses the same per-row split decision. A query head uses `kv_head = query_head // 6`, with the key/value scales and profile group of that key/value head. Sources: `src/engine35.hpp:280-285`, `src/engine35.cpp:707-721`, `src/q35_kernels.cu:1285-1286`, `src/q35_atxpk.cu:393-403`.

The prefix computation processes 32-key tiles. For numerical fidelity, each group of two query rows (passes of at most two), eight rows (three to eight), or sixteen rows (larger passes) uses a common tile range for a split: take the minimum `first` and maximum `prefix_end` over nonempty prefix ranges in that row group, round the minimum down to a multiple of 32, and visit tiles through the maximum. Each query masks every key outside its own prefix. If no row in the group has a nonempty prefix, visit no prefix tiles. This accounts for the source's shared tile extent without prescribing GPU execution. Sources: `src/q35_atxpk3.cuh:184-195`, `src/q35_atxpk4.cuh:97-108`, `src/q35_atxpk3.cuh:488-502`, `src/q35_atxpk4.cuh:431-434`, `src/q35_atx.cu:379-387`, `src/q35_atx.cu:824-826`.

**Scalar reference:** [D9](../port/DECISIONS.md#d9--per-row-prefix-tiles-and-skipped-masked-values) replaces this upstream behavior with per-row tiles and skipped masked value operations.

## Prefix score, online softmax, and value accumulation

**Scalar reference note:** [D7](../port/DECISIONS.md#d7--portable-exp2-construction)
defines portable exp2; [D9](../port/DECISIONS.md#d9--per-row-prefix-tiles-and-skipped-masked-values)
uses only absolute 32-slot tiles intersecting each row's own prefix, with
masked scores of negative infinity, probabilities of `+0.0` in the sum,
and entirely skipped masked value operations (no zero operands).

For this section define `Kbase[position,channel]` and `Vbase[position,channel]` as the BF16 codebook values selected by a packed row, or the exact signed-int8 integer values widened to BF16 for an int8 row. `Kscale`/`Vscale` are the corresponding FP32 row scales. Keep them separate; rounding `Kbase*Kscale` or `Vbase*Vscale` to BF16 before attention would change this path. Sources: `src/q35_atxpk.cu:55-66`, `src/q35_atxpk3.cuh:343-380`, `src/q35_atxpk3.cuh:402-406`, `src/q35_atx.cu:57-61`, `src/q35_atx.cu:478-520`.

The query/key prefix dot product visits sixteen consecutive 16-channel chunks. For a valid key in the current tile, its scalar schedule is below; invalid keys are assigned negative infinity after the score calculation. Source fixes this chunk order, while `mma16` remains the explicitly identified uncertainty ATT-01. For packed keys the shipping `QK8=false` expression is `dot * (1.f * (Kscale * 0.0625f))`, numerically `dot * (Kscale * 0.0625f)` for finite scales: there is no extra codebook-int8 scale and no key-mean term (`src/q35_atxpk3.cuh:379`, `src/q35_atxpk4.cuh:325`). Sources: `src/q35_atxpk3.cuh:343-380`, `src/q35_atxpk4.cuh:287-326`, `src/q35_atx.cu:478-497`.

```text
score_accumulator = 0
for chunk in 0..15:
    channels = [16*chunk,16*chunk+16)
    score_accumulator = mma16(score_accumulator,query_bf16[channels],Kbase[key,channels])
score[key] = mul(score_accumulator,mul(Kscale[key],0.0625))
if key is outside this query's prefix range:
    score[key] = -infinity
```

Each split starts with `maximum=-1e30f`, `denominator=0`, and `numerator[256]=0`. The next pseudocode block processes one 32-key prefix tile. `LOG2E` is the binary32 literal `1.4426950408889634f`. The probability calculation multiplies each score and the maximum separately before subtracting; it is not `exp2((score-maximum)*LOG2E)`. Sources: `src/q35_atxpk.cu:33-34`, `src/q35_atxpk3.cuh:301-406`, `src/q35_atxpk4.cuh:249-350`.

```text
new_maximum = max(maximum,score[0],...,score[31])
rescale = exp2f(mul(sub(maximum,new_maximum),LOG2E))
maximum = new_maximum
scaled_maximum = mul(new_maximum,LOG2E)
for k in 0..31:
    probability[k] = exp2f(sub(mul(score[k],LOG2E),scaled_maximum))

# The source's 32-term denominator reduction, including its pair additions:
for lane in 0..3:
    partial[lane] = 0
    for group in 0..3:
        k = 8*group + 2*lane
        partial[lane] = add(partial[lane],add(probability[k],probability[k+1]))
for offset in [1,2]:
    old = copy(partial)
    for lane in 0..3:
        partial[lane] = add(old[lane],old[lane XOR offset])
denominator = add(mul(denominator,rescale),partial[0])

for channel in 0..255:
    numerator[channel] = mul(numerator[channel],rescale)
for k in 0..31:
    value_weight[k] = bf(mul(probability[k],Vscale[tile_start+k]))
for half in 0..1:
    keys = [16*half,16*half+16)
    for channel in 0..255:
        numerator[channel] = mma16(numerator[channel],value_weight[keys],Vbase[tile_start+keys,channel])
```

The code skips a numerator multiplication by one when both associated query-row rescale factors are one. The multiply above describes the same finite arithmetic result; a literal execution trace can preserve that conditional. Masked keys have probability zero and a zero scale when outside the shared loaded extent. Sources: `src/q35_atxpk3.cuh:316-319`, `src/q35_atxpk3.cuh:200-227`, `src/q35_atxpk4.cuh:256-258`, `src/q35_atx.cu:396-407`.

**Scalar reference:** [D9](../port/DECISIONS.md#d9--per-row-prefix-tiles-and-skipped-masked-values) replaces this upstream behavior with per-row tiles and skipped masked value operations.

## Separate self contribution

If this split contains the semantic self position, attention adds the query's physical self slot after all prefix tiles. The self dot product uses explicit FP32 multiply/adds, not `mma16`. Its partition is four interleaved partials, followed by simultaneous XOR reductions with offsets one then two. Sources: `src/q35_atxpk3.cuh:412-445`, `src/q35_atxpk4.cuh:354-386`, `src/q35_atx.cu:530-554`.

```text
for lane in 0..3:
    partial[lane] = 0
    for chunk in 0..15:
        for offset in [2*lane,2*lane+1,8+2*lane,9+2*lane]:
            channel = 16*chunk + offset
            partial[lane] = add(partial[lane],mul(query_bf16[channel],Kbase[self_slot,channel]))
for offset in [1,2]:
    old = copy(partial)
    for lane in 0..3:
        partial[lane] = add(old[lane],old[lane XOR offset])
self_score = mul(partial[0],mul(Kscale[self_slot],0.0625))
```

Unlike prefix probabilities, self probability subtracts the new maximum before multiplying by `LOG2E`. Unlike the prefix value weights, the self probability times its value scale remains FP32, without BF16 rounding. This does not undo the BF16 rounding of the packed self key/value codebook base (`src/q35_atxpk.cu:66`, `src/q35_atxpk3.cuh:427-430,454`, `src/q35_atxpk4.cuh:369-372,396`). These are source-defined numeric differences between prefix and self, regardless of pass row count. Sources: `src/q35_atxpk3.cuh:443-458`, `src/q35_atxpk4.cuh:383-400`, `src/q35_atx.cu:551-567`.

```text
new_maximum = max(maximum,self_score)
rescale = exp2f(mul(sub(maximum,new_maximum),LOG2E))
self_probability = exp2f(mul(sub(self_score,new_maximum),LOG2E))
maximum = new_maximum
denominator = add(mul(denominator,rescale),self_probability)
self_value_weight = mul(self_probability,Vscale[self_slot])
for channel in 0..255:
    numerator[channel] = add(mul(numerator[channel],rescale),mul(self_value_weight,Vbase[self_slot,channel]))
```

For packed values, finish each split by `numerator[channel] = add(numerator[channel],mul(denominator,value_mean[channel]))`. The denominator includes the self probability and is independent of BF16 value-weight rounding. Missing means are zero in the constructed records. Int8 fallback rows already contain restored basis/mean and receive no extra mean term (`src/kvq_fast.cuh:214-218,234`). Store this split's maximum, denominator, and 256-element FP32 numerator. Sources: `src/kvq_atxpk.hpp:31-54`, `src/q35_atxpk3.cuh:463-475`, `src/q35_atxpk4.cuh:405-417`.

## Split combination, inverse rotation, and output gate

Combine splits in increasing split order. For 31 splits the source pads to 32 entries; for 42 splits it pads to 64. Padding has maximum `-1e30f`, denominator zero, and numerator zero. The sentinel test is `maximum > -1e29f`. Combination uses native `expf`, not the prefix routine's `exp2f`. Sources: `src/q35_kernels.cu:1285-1341`.

```text
padded_count = 32 if split_count <= 32 else 64
largest_maximum = max(-1e30f, all padded split maxima)
for split in 0..padded_count-1:
    scale[split] = expf(sub(split_maximum[split],largest_maximum)) if split_maximum[split] > -1e29f else 0
total_denominator = 0
for split in 0..padded_count-1:
    total_denominator = add(total_denominator,mul(scale[split],split_denominator[split]))
for channel in 0..255:
    total_numerator = 0
    for split in 0..padded_count-1:
        total_numerator = add(total_numerator,mul(scale[split],split_numerator[split,channel]))
    attended_rotated[channel] = div(total_numerator,total_denominator)

attended = inverse_rotate_cache(attended_rotated)
for channel in 0..255:
    attended[channel] = bf(attended[channel])
    gate = bf(div(1,add(1,expf(-saved_gate_bf16[channel]))))
    output[channel] = bf(mul(attended[channel],gate))
```

In mode 3 the division result is not BF16-rounded before the inverse cache rotation. The inverse result is BF16-rounded, the sigmoid gate is separately BF16-rounded, and the product is BF16-rounded again. Concatenate the 24 output heads into 6144 coordinates, then prepare the `o_proj` input with its signed 512-element rotation and int8 quantizer from chapter 02. `o_proj` residual addition and the following norm/MLP are in chapter 05. Sources: `src/q35_kernels.cu:1309-1312`, `src/q35_kernels.cu:1338-1341`, `src/q35_kernels.cu:1354-1362`, `src/engine35.cpp:735-736`, `src/engine35.cpp:795-799`.

## Alternate 16-bit-cache attention

For completeness, the following substitutions specify the ATX FP16/BF16 cache path in a nonrotated build. This is outside shipping mode 3. The row visibility, 32-key prefix tiling, softmax reduction, split combination, and explicit self dot-product order above remain, but the common tile range uses groups of sixteen query rows even for a one-row pass. Sources: `src/q35_atx.cu:92-128`, `src/q35_atx.cu:215-334`, `src/q35_atx.cu:811-814`, `src/q35_atx.cu:828-834`.

| Operation | FP16 cache | BF16 cache | Evidence |
|---|---|---|---|
| Prepared K/V | Stored FP16; no int8 row scale or profile encoding | Stored BF16; no int8 row scale or profile encoding | `src/q35_kernels.cu:607-608`, `src/q35_kernels.cu:1035-1035`, `src/q35_kernels.cu:1059-1059` |
| Query operand | `f16(query_bf16)` | `query_bf16` | `src/q35_atx.cu:171-183` |
| Prefix query/key and probability/value primitive | FP16-input `mma16` | BF16-input `mma16` | `src/q35_atx.cu:65-73`, `src/q35_atx.cu:225-226`, `src/q35_atx.cu:275-277` |
| Score scale | `mul(dot,0.0625)` | Same | `src/q35_atx.cu:234-237`, `src/q35_atx.cu:310-311` |
| Prefix value weight | `f16(probability)` | `bf(probability)` | `src/q35_atx.cu:269-270`, `src/q35_atx.cu:71-73` |
| Self value weight | FP32 probability | Same | `src/q35_atx.cu:313-322` |
| After split division | BF16 rounding, then the BF16 sigmoid/product; no 256-element inverse rotation | Same | `src/q35_kernels.cu:1309-1313`, `src/q35_kernels.cu:1338-1342`, `src/q35_kernels.cu:1354-1358` |

The 16-bit formats describe different numeric paths, not just different ways to serialize the shipping rotated cache. Their instruction-level rounding has the same unresolved MMA issue ATT-01. Sources: `src/engine35.cpp:200-210`, `src/q35_atx.cu:65-81`, `src/q35_atx.cu:171-183`.

## Cache lifetime, long prompts, and row-count differences

Verification writes keys and values into physical cache slots even while recurrent-state commit is provisional. Later acceptance advances the logical position; unused speculative entries can remain beyond that position because subsequent attention masks them or overwrites them. GDN-only replay skips full-attention layers. Promotion copies a side candidate's packed/int8/16-bit cache payload and scale into its accepted contiguous slot. Sources: `src/engine35.cpp:684-705`, `src/engine35.cpp:843-851`, `src/engine35.cpp:877-899`, `src/engine35.cpp:1319-1329`, `src/q35_atxpk.cu:402-408`.

Long prompt passes first prepare and encode all their rows, then invoke attention in slices of at most 64 query rows with adjusted base positions. Causality still follows the per-row mask above. `q35_fa` and the older scalar attention kernel are bypassed because `atx_` wins before either fallback. Sources: `src/engine35.cpp:684-705`, `src/engine35.cpp:707-732`.

For more than 64 projected rows, the PFALT preparation routine has the same BF16 projection boundary, eight consecutive 32-element square reductions per head, RoPE product rounding, cache rotation, and int8 scale definition as the small-row routine. The launch also requires XSUM mode 3, suitable cache pointers, and no configured prefetch payload. Sources: `src/q35_kernels.cu:610-688`, `src/q35_kernels.cu:1061-1063`.

One-row and multi-row execution select different attention routines (`pk3` versus `pk4`) and different shared prefix tile extents; the code preserves the specified 16-channel chunk order, prefix denominator reduction, BF16 probability boundary, and separate self path in both. Source inspection alone does not establish that all hardware instruction results are identical, which is precisely ATT-01. In particular, do not mistake the always-separate self contribution for a difference unique to multi-row verification. Sources: `src/q35_atxpk.cu:729-737`, `src/q35_atxpk3.cuh:309-406`, `src/q35_atxpk4.cuh:249-350`, `src/q35_atxpk3.cuh:412-475`, `src/q35_atxpk4.cuh:354-417`.

The author's “same Hadamard rotation” statement describes the real-arithmetic orthogonal identity, but it omits the observable BF16 and int8 rounding between rotation and attention. It must not be read as a promise that the rotated, quantized engine produces the exact bits of an unrotated attention implementation. The claim that cache rows encode independently is supported by the per-row profile input, reduction, state reset for each 64-element chain, and row-specific output; the actual profile and native primitive uncertainties above remain. Sources: `docs/TECHNOLOGY.md:226-231`, `src/q35_kernels.cu:1029-1059`, `src/kvq_fast.cuh:65-95`, `src/kvq_fast.cuh:108-147`, `src/kvq_fast.cuh:220-234`.
