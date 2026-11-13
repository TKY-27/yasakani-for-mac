# 01. Main-model architecture

## Scope and evidence

This specification describes the source at commit
`758ef28e307521e1ceb006ddbe895e909910b1ad`. The entry point selects
`Engine35` only when the container's architecture is `qwen3_5`; the Python
package opens the main weights and then loads the block drafter. The different
`Engine` class is not the model described here. Sources: `src/c_api.cpp:15-18`,
`python/yasakani/engine.py:90-116`.

Read [FORMAT.md](../FORMAT.md) for the container, tensor descriptors, code-stream
decoding, and sign generators. This series specifies how the engine uses those
objects; it does not redefine the container. The distinction matters: linear
weights are requantized to an int8 lookup table on load, while embedding lookup
uses the stored FP16 lookup table. Sources: `src/loader.cpp:19-36`,
`src/engine35.cpp:272-299`, `src/q35_kernels.cu:1521-1578`.

Array indices are zero-based. Matrix dimensions below are `[output, input]`,
matching the descriptors and caller allocations. FP32, FP16 and BF16 mean
binary32, binary16 and the upper-16-bit format respectively; a `uint16_t*`
alone does **not** identify which 16-bit format is stored. Sources:
`src/format.hpp:18-22`, `src/loader.cpp:19-24`, `src/engine35.cpp:168-177`,
`src/q35_kernels.cu:35-37`.

## Dimensions and what is actually enforced

| Quantity | Value / rule | Evidence |
|---|---|---|
| Hidden width | 5,120; enforced at open | `src/q35.hpp:13`, `src/engine35.cpp:214-218` |
| Full-attention query heads | 24 | `src/q35.hpp:13`, `src/engine35.cpp:216-217` |
| Full-attention key/value heads | 4; six query heads share each KV head | `src/q35.hpp:13`, `src/q35_atxpk.cu:393-395` |
| Attention head width | 256 | `src/q35.hpp:13` |
| Rotary width | First 64 coordinates of each head; 32 pairs | `src/q35.hpp:13`, `src/q35_kernels.cu:1049-1053` |
| DeltaNet query/key heads | 16 | `src/q35.hpp:14` |
| DeltaNet value heads | 48 | `src/q35.hpp:14` |
| DeltaNet head width | 128 | `src/q35.hpp:14` |
| DeltaNet convolution channels | 10,240 = 2 × 16 × 128 + 48 × 128 | `src/q35.hpp:14`, `src/engine35.cpp:256-260` |
| DeltaNet output width | 6,144 = 48 × 128 | `src/q35.hpp:14` |
| Feed-forward width | Header `model.ffn`; must be a multiple of 512. Committed model comparison records report 17,408 | `src/format.hpp:35`, `src/engine35.cpp:216-218`, `tools/records/compare_fastest.json:43-69` |
| Vocabulary | Header `model.vocab`; Python package uses 248,320 | `src/format.hpp:36`, `src/engine35.cpp:218`, `python/yasakani/runtime.py:19-23` |
| Layer count | Header `model.layers`; comparison records enumerate 0 through 63 | `src/format.hpp:34`, `src/engine35.cpp:218-238`, `tools/records/compare_fastest.json:10-14`, `tools/records/compare_fastest.json:4333-4337` |
| Normalization epsilon | Header `model.eps`, read as FP32; the member initializer `1e-6f` is overwritten | `src/format.hpp:36`, `src/engine35.cpp:218`, `src/engine35.hpp:315` |
| Context capacity | Requested capacity rounded upward to a multiple of 64; Python model-folder default is 262,144 | `src/engine35.cpp:219`, `python/yasakani/runtime.py:136-138` |

**Scalar reference note:** [D8](../port/DECISIONS.md#d8--fixed-qwen3_5-rotary-width)
uses rotary width 64 for `model.arch=qwen3_5`, requires `head_dim=256`, and
rejects unsupported architectures; this is an explicit exception to D5.

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

## Layer sequence

The actual dispatch rule is `layer_types[layer] == "full_attention"`.
**Every other string selects DeltaNet**; the code checks the array length but
does not validate a second accepted spelling. It visits layers in increasing
index order. Sources: `src/engine35.cpp:224-238`, `src/engine35.cpp:255-266`,
`src/engine35.cpp:824-834`.

The recorded model has the following 16 attention layers and 48 DeltaNet
layers. The source column lists the four tensor-name entries, in the same
order as the four layer indices. This is the inventory qualification in
ARCH-01, not a replacement for the header-driven dispatch.

| DeltaNet layers | Full-attention layer | Evidence in `tools/records/compare_fastest.json` |
|---|---|---|
| 0, 1, 2 | 3 | `tools/records/compare_fastest.json:10`, `tools/records/compare_fastest.json:76`, `tools/records/compare_fastest.json:142`, `tools/records/compare_fastest.json:208` |
| 4, 5, 6 | 7 | `tools/records/compare_fastest.json:285`, `tools/records/compare_fastest.json:351`, `tools/records/compare_fastest.json:417`, `tools/records/compare_fastest.json:483` |
| 8, 9, 10 | 11 | `tools/records/compare_fastest.json:560`, `tools/records/compare_fastest.json:626`, `tools/records/compare_fastest.json:692`, `tools/records/compare_fastest.json:758` |
| 12, 13, 14 | 15 | `tools/records/compare_fastest.json:835`, `tools/records/compare_fastest.json:901`, `tools/records/compare_fastest.json:967`, `tools/records/compare_fastest.json:1033` |
| 16, 17, 18 | 19 | `tools/records/compare_fastest.json:1110`, `tools/records/compare_fastest.json:1176`, `tools/records/compare_fastest.json:1242`, `tools/records/compare_fastest.json:1308` |
| 20, 21, 22 | 23 | `tools/records/compare_fastest.json:1385`, `tools/records/compare_fastest.json:1451`, `tools/records/compare_fastest.json:1517`, `tools/records/compare_fastest.json:1583` |
| 24, 25, 26 | 27 | `tools/records/compare_fastest.json:1660`, `tools/records/compare_fastest.json:1726`, `tools/records/compare_fastest.json:1792`, `tools/records/compare_fastest.json:1858` |
| 28, 29, 30 | 31 | `tools/records/compare_fastest.json:1935`, `tools/records/compare_fastest.json:2001`, `tools/records/compare_fastest.json:2067`, `tools/records/compare_fastest.json:2133` |
| 32, 33, 34 | 35 | `tools/records/compare_fastest.json:2210`, `tools/records/compare_fastest.json:2276`, `tools/records/compare_fastest.json:2342`, `tools/records/compare_fastest.json:2408` |
| 36, 37, 38 | 39 | `tools/records/compare_fastest.json:2485`, `tools/records/compare_fastest.json:2551`, `tools/records/compare_fastest.json:2617`, `tools/records/compare_fastest.json:2683` |
| 40, 41, 42 | 43 | `tools/records/compare_fastest.json:2760`, `tools/records/compare_fastest.json:2826`, `tools/records/compare_fastest.json:2892`, `tools/records/compare_fastest.json:2958` |
| 44, 45, 46 | 47 | `tools/records/compare_fastest.json:3035`, `tools/records/compare_fastest.json:3101`, `tools/records/compare_fastest.json:3167`, `tools/records/compare_fastest.json:3233` |
| 48, 49, 50 | 51 | `tools/records/compare_fastest.json:3310`, `tools/records/compare_fastest.json:3376`, `tools/records/compare_fastest.json:3442`, `tools/records/compare_fastest.json:3508` |
| 52, 53, 54 | 55 | `tools/records/compare_fastest.json:3585`, `tools/records/compare_fastest.json:3651`, `tools/records/compare_fastest.json:3717`, `tools/records/compare_fastest.json:3783` |
| 56, 57, 58 | 59 | `tools/records/compare_fastest.json:3860`, `tools/records/compare_fastest.json:3926`, `tools/records/compare_fastest.json:3992`, `tools/records/compare_fastest.json:4058` |
| 60, 61, 62 | 63 | `tools/records/compare_fastest.json:4135`, `tools/records/compare_fastest.json:4201`, `tools/records/compare_fastest.json:4267`, `tools/records/compare_fastest.json:4333` |

## Tensor inventory

In these names, `L` is the decimal layer index and `F` is the header's
feed-forward width. A compressed matrix uses the operator in
[02](02_COMPRESSED_LINEAR.md). A small vector below must have `dtype="bf16"`
and the specified element count, otherwise opening fails. Source:
`src/engine35.cpp:136-177`, `src/engine35.cpp:236-271`.

### Shared by every layer

| Name | Shape | Purpose / evidence |
|---|---|---|
| `layers.L.input_layernorm` | `[5120]` BF16 | Before attention/DeltaNet; `src/engine35.cpp:265-266`, `src/engine35.cpp:798-799`, `src/engine35.cpp:829` |
| `layers.L.post_attention_layernorm` | `[5120]` BF16 | After attention residual, before MLP; `src/engine35.cpp:265-266`, `src/engine35.cpp:736`, `src/engine35.cpp:793` |
| `layers.L.mlp.gate_proj` | `[F,5120]` compressed | MLP gate; `src/engine35.cpp:265`, `src/engine35.cpp:757-766`, `tools/records/compare_fastest.json:43-47` |
| `layers.L.mlp.up_proj` | `[F,5120]` compressed | MLP up branch; `src/engine35.cpp:265`, `tools/records/compare_fastest.json:54-58` |
| `layers.L.mlp.down_proj` | `[5120,F]` compressed | MLP output; `src/engine35.cpp:265`, `tools/records/compare_fastest.json:65-69` |

### Full attention only

| Name suffix after `layers.L.` | Shape | Evidence |
|---|---|---|
| `self_attn.q_proj` | `[12288,5120]`, including queries **and** output gates | `src/engine35.cpp:240`, `src/q35_kernels.cu:1013,1038`, `tools/records/compare_fastest.json:208-212` |
| `self_attn.k_proj` | `[1024,5120]` | `src/engine35.cpp:240`, `tools/records/compare_fastest.json:219-223` |
| `self_attn.v_proj` | `[1024,5120]` | `src/engine35.cpp:240`, `tools/records/compare_fastest.json:230-234` |
| `self_attn.o_proj` | `[5120,6144]` | `src/engine35.cpp:240`, `tools/records/compare_fastest.json:241-245` |
| `self_attn.q_norm`, `self_attn.k_norm` | Each `[256]` BF16; shared across heads | `src/engine35.cpp:241`, `src/q35_kernels.cu:1017,1039-1047` |

### DeltaNet only

| Name suffix after `layers.L.` | Shape | Evidence |
|---|---|---|
| `linear_attn.in_proj_qkv` | `[10240,5120]` compressed | `src/engine35.cpp:256`, `tools/records/compare_fastest.json:10-14` |
| `linear_attn.in_proj_z` | `[6144,5120]` compressed | `src/engine35.cpp:256`, `tools/records/compare_fastest.json:21-25` |
| `linear_attn.out_proj` | `[5120,6144]` compressed | `src/engine35.cpp:256`, `tools/records/compare_fastest.json:32-36` |
| `linear_attn.in_proj_a`, `linear_attn.in_proj_b` | Each `[48,5120]` BF16, despite being in the vector collection | `src/engine35.cpp:257`, `src/q35_kernels.cu:746-784` |
| `linear_attn.conv1d` | `[10240,4]` BF16 | `src/engine35.cpp:258`, `src/q35_kernels.cu:867-878` |
| `linear_attn.A_log`, `linear_attn.dt_bias` | Each `[48]` BF16 | `src/engine35.cpp:258-259` |
| `linear_attn.norm` | `[128]` BF16, shared by value heads | `src/engine35.cpp:259`, `src/q35_kernels.cu:964-977` |

### Outside the layer stack

`norm` is a BF16 vector of length 5,120. `lm_head` is a separate compressed
matrix with vocabulary rows and 5,120 columns. The implementation loads it
even when header metadata says `tie_embeddings`; `Engine35` does not branch
on that field. `embed_tokens` can be a BF16 `[V,5120]` vector or a coded
embedding matrix. Coded embeddings may have `embed_tokens.res` and
`embed_tokens.res2`; the third component requires the second. Sources:
`src/format.hpp:37`, `src/engine35.cpp:268-299`, `src/engine35.cpp:803-813`.

## Logical order of one pass

The following is an **ordering outline**. `prepare_normalized` is the deferred
normalization/rotation/quantization procedure in [05](05_RESIDUAL_NORM_MLP.md),
not an instruction to compute an ordinary mathematical RMSNorm and round it
once. `add_projected_residual` includes the projection's output rotation and
BF16 rounding. Sources: `src/engine35.cpp:559-572`,
`src/engine35.cpp:675-736`, `src/engine35.cpp:757-841`.

```text
hidden[rows,5120] = embedding_lookup(token_ids)
prepared = prepare_normalized(hidden, layer[0].input_layernorm)
for layer_index = 0 .. layer_count-1:
    if layer[layer_index].full_attention:
        branch = attention(prepared, layer_index, cache, row_positions)
    else:
        branch = delta_net(prepared, layer_index, recurrent_state,
                           convolution_history, commit_count)
    hidden = add_projected_residual(hidden, branch.output_projection)
    prepared = prepare_normalized(hidden, layer[layer_index].post_attention_layernorm)
    mlp_input = gated_mlp_activation(linear(gate, prepared), linear(up, prepared))
    hidden = add_projected_residual(hidden, linear(down, mlp_input))
    next_norm = layer[layer_index+1].input_layernorm if another layer else norm
    prepared = prepare_normalized(hidden, next_norm)
    if layer_index is a configured draft capture:
        captured[layer_index] = hidden
score requested rows with lm_head(prepared)
```

The main hidden array is FP32, but both embedding outputs and projected
residual additions have explicit BF16 boundaries; preserving only the array's
storage type is insufficient. Capture happens after the layer's MLP residual
has been incorporated, while preparation for the next layer has already run.
Sources: `src/engine35.cpp:323`, `src/engine35.cpp:798-799`,
`src/engine35.cpp:829-833`, `src/q35_kernels.cu:1427-1467`.

## Persistent state

Each DeltaNet layer owns FP32 state `[48,128,128]` and BF16 convolution history
`[3,10240]`. Verification retains projected mixes and recurrence inputs for up
to 16 rows so commit can replay only the accepted prefix. Each attention layer
owns four head-major KV sequences, with representation chosen by the KV
profile. The committed position is the number of consumed input tokens, not
the number of tokens already shown to the caller: the next token is produced
but remains unconsumed. Sources: `src/engine35.cpp:242-263`,
`src/engine35.cpp:843-851`, `src/engine35.cpp:1244-1266`,
`src/engine35.cpp:1319-1329`.

Reset zeros DeltaNet state and convolution history and sets the position to
zero. It does not zero the KV arrays: subsequent visibility is controlled by
position and the rows rewritten by the next request. Sources:
`src/engine35.cpp:392-400`, `src/q35_atxpk.cu:350-409`.
