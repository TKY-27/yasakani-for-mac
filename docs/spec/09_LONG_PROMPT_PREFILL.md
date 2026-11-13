# 09. Long-prompt prefill

## Which prefill is shipping

`DEFAULT_SPEC.prefill_head=1` is decisive. `SpecxT::prefill` calls
`Engine35::prefill_rows`, including for long prompts. It does **not** call
`Engine35::prefill`, which is the entry point that can choose layer-major
prefill. Thus `YSKN_Q35_LAYER_ROWS=4096` and
`YSKN_Q35_LAYER_OVERLAP=1` are set by the shipping profile but do not activate
layer-major prefill for a shipping generation request. Sources:
`python/yasakani/runtime.py:27-43`, `python/yasakani/runtime.py:54-59`,
`src/specx.cpp:94-123`, `src/engine35.cpp:982-986`,
`src/engine35.cpp:1244-1266`.

The default prefill row limit is 1,024: `pfa_=1`, its initial capacity is 512,
the shipping `YSKN_Q35_PMAX=1024` overrides that capacity, and `PM_` takes that
capacity when `YSKN_Q35_PM` is absent. Attention is further sliced into at
most 64 rows; the output head into at most 16. Sources:
`src/engine35.hpp:238-246`, `src/engine35.hpp:313-314`,
`src/engine35.cpp:193-213`, `src/engine35.cpp:722-733`,
`src/engine35.cpp:835-837`, `python/yasakani/runtime.py:35`.

## Shipping scalar control flow

The snapshot logic can split the prompt into two independently chunked
segments. Let `start` be the reused prefix length, and `snapshot_end` the
boundary immediately after the last configured marker in the new suffix
`ids[start:n]`, or `n-1` if that suffix has no marker. The boundary is used only when `start < snapshot_end < n`.
Each call below chunks relative to its own segment, not relative to absolute
position zero. Sources: `src/specx.cpp:96-119`.

```text
prefill_segment(ids[a:b]):
    require b > a and committed_position + (b-a) <= context_capacity
    done = 0
    while done < b-a:
        rows = min(1024, b-a-done)
        input_tokens[0:rows] = ids[a+done:a+done+rows]
        row_base_position = committed_position
        run_all_layers(input_tokens, commit_count=rows, head_mode=ALL)
        update_draft_context(captured_features, rows, row_base_position)
        record each row's selected token and top-eight table values
        first_unconsumed_token = selected_token[rows-1]
        committed_position += rows
        done += rows
    pending_input_token = first_unconsumed_token
    clear last_verification_row_count and already_committed flag
    return first_unconsumed_token
```

This control flow and the saved top-eight results come from
`src/engine35.cpp:1244-1266`. `SpecxT` records the table entries only for the
newly evaluated suffix, then appends the final selected token to its sequence
(`src/specx.cpp:120-125`). With sampling enabled, these are sampler-derived
results, despite the C++ buffer name `out_argmax`
(`src/engine35.cpp:814-821`).

All rows of a prefill pass are committed immediately. DeltaNet recurrence
uses `n_commit=rows`; no provisional replay is needed. Attention writes the
current chunk's K/V rows and applies causal visibility separately to each
query. The final layer is followed by head slices of up to 16 rows, even when
the body handled 1,024 rows. Sources: `src/engine35.cpp:768-790`,
`src/engine35.cpp:824-837`, `src/engine35.cpp:1249-1253`,
`src/engine35.cpp:684-704`, `src/q35_atxpk.cu:350-409`.

## Numeric paths selected by row count

| Row count / stage | Selected behavior | Evidence |
|---|---|---|
| Body, 1–16 rows | LNORD int64 weighted integer accumulation; consumers apply the deferred scale and normalization | `src/engine35.cpp:403-429`, `src/engine35.hpp:253-255` |
| Body, more than 16 rows | G6 when eligible; otherwise PFA. The int64 sum is finalized to FP32 before the consumer; it is not an FP32 replacement for the integer accumulation | `src/engine35.cpp:413-420`, `src/q35_g6.cu:130-146`, `src/pfa.cu:153-165` |
| G6 weights | For eligible batches of at least 256 rows, a matrix is decoded to int8 and reused for the row batch. Shipping enables the required workspace; 17..255 rows fall back to PFA | `src/engine35.cpp:314-320,416-419`, `src/q35_g6.cu:300-353` |
| DeltaNet gate projection | Fused into the mix operation for at most 32 rows; separate operation above 32 | `src/engine35.cpp:772-784` |
| DeltaNet recurrence | Shipping `PFALT=1` selects the alternate prefill recurrence for more than 64 rows; the selector does not test commit count | `src/q35_kernels.cu:245,602,958-959`, `python/yasakani/runtime.py:34` |
| Full attention | At most 64 query rows per attention call, with the same absolute base shifted by each slice | `src/engine35.cpp:722-733` |
| Output head | LNORD for every head slice, including heads of a long prefill pass | `src/engine35.cpp:803-809`, `src/engine35.cpp:835-837` |

Use the arithmetic and row-count qualifications in
[02](02_COMPRESSED_LINEAR.md), [03](03_GATED_DELTANET.md),
[04](04_ATTENTION.md), [05](05_RESIDUAL_NORM_MLP.md), and
[06](06_OUTPUT_HEAD_SAMPLER.md). A scalar implementation must not replace
these branches with an unqualified sequential FP32 dot product: the source
selects different reducers and explicit rounding boundaries at the locations
listed above.

## Alternate entry point: layer-major prefill

This path is implemented and relevant to a future direct `yskn_prefill`
caller, but is outside `DEFAULT_SPEC.prefill_head=1`. Its predicate is:
`layer_rows>0`, at least 1,024 **new** tokens, no MTP, no visual input, XSUM on,
XSUM-partials off, portable-attention mode off, ATX on, no debug skip mask,
and successful workspace preparation. Otherwise direct `prefill` uses
`prefill_chunks`, which computes only the last head of the last chunk.
Sources: `src/c_api.cpp:22`, `src/engine35.cpp:982-986`,
`src/engine35.cpp:1200-1216`.

The layer-major path requires compatible matrix shapes and bit widths
`{4,6,8}` bits per **pair**. Its maximum resident row count starts at
`round_up(max(2*PM,layer_rows), PM)`. If memory is insufficient, it halves the
row count with `max(2*PM, floor(SC/(2*PM))*PM)`, then disables overlap at
`2*PM`, then falls back. It preserves
one GiB of free-memory margin. These are host decisions that select a numeric
path even though allocation and overlap mechanics are not part of a scalar
algorithm. Sources: `src/engine35.cpp:993-1048`.

For each resident row batch, the following ordering is fixed. Source:
`src/engine35.cpp:1097-1175`.

```text
for resident_start in steps of resident_row_capacity:
    count = min(resident_row_capacity, remaining_tokens)
    sub_count = ceil(count / prefill_row_limit)
    rows_per_subbatch = min(prefill_row_limit,
                           round_up(ceil(count/sub_count), 64))
    for layer = 0 .. layer_count-1:
        decode all compressed matrices needed by this layer to int8
        for subbatch in increasing token order:
            use saved hidden, normalized input, int8 input and sum-of-squares arrays
            if layer == 0: initialize those arrays from token embeddings
            advance this layer for all rows in the subbatch
            save prepared arrays for the next layer
            capture requested draft features in the final 2048-token window
    if final resident batch:
        compute output head for only its final row
    committed_position += count
feed the captured final window to the block drafter
return the final selected token; leave it pending
```

When overlap is enabled, DeltaNet's convolution for a later subbatch obtains
its preceding three mixes directly from the previous subbatch, rather than
waiting to read the persistent history. State updates themselves remain in
token order. This is a dependency rule to preserve in a scalar translation;
parallel execution is unnecessary. Sources: `src/engine35.cpp:1063-1095`,
`src/engine35.cpp:1120-1145`.

Long-path draft capture is int8 with a scale per 512 features, using the
drafter's input signs. It retains only the final `min(2048,n)` features and
feeds their absolute starting position into `bdraft_ctx`. The shipping
row-prefill path instead feeds captures after every chunk. Sources:
`src/q35_prefill.cu:9-20`, `src/engine35.cpp:1100-1117`,
`src/engine35.cpp:1168`, `src/engine35.cpp:1251-1253`.

## Open issues and limits

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
