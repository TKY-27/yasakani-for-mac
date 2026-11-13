# 08 — Speculation, verification, commit, and prompt reuse

This topic follows `Specx35`, the `SpecxT<Engine35>` specialization instantiated
by `yskn_spec_generate`. A native request first returns the prompt's next
token in `out_ids[0]`, then generates at most `n_gen` **additional** tokens
into `out_ids[1..]`. `n_gen` is not the total including that first token.
(`src/c_api.cpp:37-74`; `src/specx.hpp:131-132`)

## 1. Shipping choices and state invariant

The request passes these 44 signed integer configuration fields in this
order. The field names are Python's `SPEC_FIELDS`; native assignments are
in the same order in `c_api.cpp`. Cost-based selection is disabled by
`sel=nd_auto=prmode=0`.
(`python/yasakani/runtime.py:47-59`; `src/c_api.cpp:41-54`)

| Index | Python field | Shipping value | Evidence |
|---|---|---|---|
| 0 | `gamma` | 7 | `python/yasakani/runtime.py:55` |
| 1 | `use_d0` | 1 | `python/yasakani/runtime.py:55` |
| 2 | `use_d1` | 1 | `python/yasakani/runtime.py:55` |
| 3 | `d0_nmin` | 2 | `python/yasakani/runtime.py:55` |
| 4 | `d0_nmax` | 8 | `python/yasakani/runtime.py:55` |
| 5 | `d0_priority_n` | 3 | `python/yasakani/runtime.py:55` |
| 6 | `extend` | 1 | `python/yasakani/runtime.py:55` |
| 7 | `d1_rejected` | 1 | `python/yasakani/runtime.py:55` |
| 8 | `auto_gamma` | 0 | `python/yasakani/runtime.py:55` |
| 9 | `use_graph` | 1 | `python/yasakani/runtime.py:55` |
| 10 | `use_d3` | 0 | `python/yasakani/runtime.py:56` |
| 11 | `d3_steps` | 0 | `python/yasakani/runtime.py:56` |
| 12 | `d1_bigram` | 1 | `python/yasakani/runtime.py:56` |
| 13 | `d1_static` | 0 | `python/yasakani/runtime.py:56` |
| 14 | `d3_extend` | 0 | `python/yasakani/runtime.py:56` |
| 15 | `d3_lazy` | 0 | `python/yasakani/runtime.py:56` |
| 16 | `d3_gate` | 0 | `python/yasakani/runtime.py:56` |
| 17 | `gamma_long` | 15 | `python/yasakani/runtime.py:56` |
| 18 | `long_n` | 6 | `python/yasakani/runtime.py:56` |
| 19 | `d1_trigram` | 1 | `python/yasakani/runtime.py:56` |
| 20 | `eos1` | 248046 | `python/yasakani/runtime.py:20,57` |
| 21 | `eos2` | 248044 | `python/yasakani/runtime.py:20,57` |
| 22 | `long_d0` | 8 | `python/yasakani/runtime.py:57` |
| 23 | `sel` | 0 | `python/yasakani/runtime.py:57` |
| 24 | `cost_tok_us` | 0 | `python/yasakani/runtime.py:57` |
| 25 | `cost_row_us` | 0 | `python/yasakani/runtime.py:57` |
| 26 | `cost_nd_us` | 0 | `python/yasakani/runtime.py:57` |
| 27 | `nd_auto` | 0 | `python/yasakani/runtime.py:57` |
| 28 | `prmode` | 0 | `python/yasakani/runtime.py:57` |
| 29 | `cost_row2_us` | 0 | `python/yasakani/runtime.py:58` |
| 30 | `lfrow` | 0 | `python/yasakani/runtime.py:58` |
| 31 | `lfrow_q_pm` | 300 | `python/yasakani/runtime.py:58` |
| 32 | `mark_tok` | -1 | `python/yasakani/runtime.py:58` |
| 33 | `cost_tok_k` | 0 | `python/yasakani/runtime.py:58` |
| 34 | `cost_row2_k` | 0 | `python/yasakani/runtime.py:58` |
| 35 | `cost_nd_k` | 0 | `python/yasakani/runtime.py:58` |
| 36 | `bdraft` | 1 | `python/yasakani/runtime.py:58` |
| 37 | `bdr_long_n` | 4 | `python/yasakani/runtime.py:58` |
| 38 | `bdr_adapt` | 1 | `python/yasakani/runtime.py:59` |
| 39 | `bdr_probe` | 4 | `python/yasakani/runtime.py:59` |
| 40 | `bdr_d0_len` | 15 | `python/yasakani/runtime.py:59` |
| 41 | `prefill_head` | 1 | `python/yasakani/runtime.py:59` |
| 42 | `snap` | 1 | `python/yasakani/runtime.py:59` |
| 43 | `snap_tok` | 248045 | `python/yasakani/runtime.py:21,59` |

The native spelling differs for four fields: `cost_nd_us` becomes
`cost_mtp_us`, `nd_auto` becomes `mtp_auto`, `lfrow_q_pm` becomes `wk_q_pm`,
and `cost_nd_k` becomes `cost_mtp_k`. All other names in the table keep their
native spelling. (`src/c_api.cpp:47-51`)

`lfrow=0` does **not** disable all side-candidate behavior: it gates the old
MTP branch. The block drafter independently defaults HCAP on and enables
the engine's row-table option for the generation scope. With sampler
version 3 and positive temperature, HCAP can create side candidates on
8-row passes. With greedy selection it supplies a straight-chain table.
(`src/engine35.cpp:1689-1693`; `src/specx.cpp:193-198,253-254`;
`src/q35_bdraft.cu:680-705`)

Define `sequence` as all prompt tokens plus output produced so far. Its
last token is the **frontier**: that token has been chosen, but its own
main-model layer computation has not yet been committed. At a successful
pass boundary, `engine_position = len(sequence)-1`; main-model state covers
input tokens strictly before that position. A verification row at local
index `j` consumes an input and predicts its successor. This explains why
`commit(a)` processes `a+1` rows, not `a`.
(`src/specx.cpp:120-126,206,435-440`; `src/engine35.cpp:1319-1329`)

The speculative tables and acceptance estimators are request-local, apart
from diagnostic cumulative pass counters. Although some 8/16-row estimator
variables are declared static, the values affecting width choice are reset
at the beginning of every `generate` call.
(`src/specx.cpp:146-155,178-183,382,466-467`)

## 2. Prompt snapshot selection and prefill

Use strict token-prefix equality: `prefix` must be nonempty and shorter
than the new prompt. Equal-length identical prompts do not take either
reuse branch. For shipping `snap=1`:

```text
start = 0
snapshot_cut = none
if live_note_is_valid and live_tokens is a strict prefix of prompt:
    start = len(live_tokens)                       # mode 1
else if saved_snapshot_position > 0 and saved_ids is a strict prefix of prompt:
    restore_snapshot()
    start = saved_snapshot_position                # mode 2
else:
    reset_engine()                                 # mode 0

q = index_after_last_occurrence(prompt[start:], 248045)
if no such occurrence:
    q = len(prompt)-1
if start < q < len(prompt):
    snapshot_cut = q

if snapshot_cut exists:
    prefill_rows(prompt[start:snapshot_cut])
    save_snapshot(prompt[0:snapshot_cut])
    first_token = prefill_rows(prompt[snapshot_cut:len(prompt)])
else:
    first_token = prefill_rows(prompt[start:len(prompt)])

sequence = prompt followed by first_token
```

If the last marker is the final prompt token, `q==len(prompt)` and no new
snapshot is saved; the fallback to `len(prompt)-1` applies only when no
marker was found. Each `prefill_rows` segment requests a head and top-eight
list for **every** processed row, not only the last row. This route bypasses
the separate `Engine35::prefill`/`prefill_vsp` dispatch; long prompts follow
topic [09](09_LONG_PROMPT_PREFILL.md).
(`src/specx.cpp:94-126`; `src/engine35.cpp:1244-1266`)

Initialize the next-token tables from newly processed prompt rows
`start..len(prompt)-1`, using the main model's raw-logit top-eight lists.
Cached prefix rows are not rescored to populate these request-local tables.
The n-gram index is cleared, then populated lazily from the full sequence
when drafting begins. (`src/specx.cpp:109,120-125`;
`src/specx.cpp:89-92`; `src/specx.cpp:42-55`)

### Saved versus retained state

A saved snapshot copies each of the 48 recurrent layers' binary32 state
matrix and three-row bfloat16 convolution history. It also copies all five
draft layers' complete bfloat16 K/V rings. If the inactive MTP model were
loaded, its pending hidden row would also be saved. It stores the prefix
IDs and absolute position. Main-model attention K/V cache data is **not**
copied: the prefix already remains in the live cache, and suffix writes
must not modify that prefix.
(`src/engine35.cpp:1550-1569`; `src/engine35.cpp:877-899`)

Restore reinstalls the saved recurrent state/history and draft rings, sets
the position to the saved cut, clears `last_M` and `pre_committed`, and
increments the state epoch. It does not restore a previously selected
frontier token; the nonempty suffix prefill obtains a new one.
(`src/engine35.cpp:1571-1583`; `src/specx.cpp:99-119`)

At successful request completion, the API records `sequence` without its
last frontier token as the live-prefix note. That note is valid only if its
epoch equals the current engine epoch, its length equals engine position,
and position is positive. Reset invalidates the saved snapshot and zeroes
recurrent state/history; it does not clear attention or draft-ring memory,
whose readable ranges are determined by position.
(`src/c_api.cpp:70-71`; `src/engine35.hpp:126-133`;
`src/engine35.cpp:392-400`; `src/q35_bdraft.cu:122,154`)

## 3. Copy drafting: exact n-gram index behavior

The copy source searches earlier occurrences of the sequence suffix. It
stores only one previous ending position per `(n,hash)`: insertion
overwrites any older position with the same hash. A hash collision is
checked by comparing actual tokens; on a mismatch it tries a shorter
length, rather than searching older colliding occurrences.
(`src/specx.cpp:37-58`)

All operations in `gram_hash` are unsigned 64-bit modulo `2^64`:

```text
gram_hash(end,n):
    h = 0x9E3779B97F4A7C15 xor uint64(n)
    for i = end-n+1 .. end:
        h = h xor (uint64(uint32(sequence[i])) + 0x9E3779B97F4A7C15
                   + (h << 6) + (h >> 2))
        h = h * 0xBF58476D1CE4E5B9
    return h

copy_lookup(frontier_index,maximum_drafts):
    while indexed_end < frontier_index:
        for n = 2 .. 8:
            if indexed_end-n+1 >= 0:
                index[n,gram_hash(indexed_end,n)] = indexed_end
        indexed_end += 1
    for n = 8 down to 2:
        if frontier_index-n+1 < 0: continue
        previous_end = index[n,gram_hash(frontier_index,n)] if present
        if absent or actual n-token suffixes differ: continue
        proposal = sequence[previous_end+1 :
                            min(frontier_index+1,previous_end+1+maximum_drafts)]
        return n,proposal
    return 0,[]
```

The index never inserts the current frontier as a previous occurrence.
Copying stops at the current frontier; it does not repeat a matched pattern
past available sequence data. (`src/specx.cpp:37-58`)

## 4. Next-token table

The table lookup priority is three-token context, then two-token context,
then the top entry for the current token. Context keys are unsigned 64-bit
base-`V` encodings:

```text
pair_key(a,b) = uint64(uint32(a))*V + uint64(uint32(b))
triple_key(a,b,c) = pair_key(a,b)*V + uint64(uint32(c))

table_next(a,b,c):
    if a >= 0 and b >= 0 and triple_key(a,b,c) exists: return triple_table[key]
    if b >= 0 and pair_key(b,c) exists: return pair_table[key]
    if token_table[c,0] >= 0: return token_table[c,0]
    return no_token
```

`record(a,b,c,top8)` replaces all eight entries for `c`, the pair entry
`(b,c)` when `b>=0`, and the triple entry `(a,b,c)` when both preceding
tokens exist. Only `top8[0]` is used to extend a chain. These are highest
raw-logit predictions, not the possibly different sampled main-model
choice. Static file-backed tables are disabled in shipping.
(`src/specx.hpp:128-129`; `src/specx.cpp:60-71,89-92`;
`src/q35_sample.cu:177-205`; `python/yasakani/runtime.py:56`)

When extending a copy proposal, repeatedly look up the last three tokens
of the existing sequence plus that proposal until the draft limit is
reached or no entry exists. Neural proposals are not extended by this
table under the shipping block-draft branch.
(`src/specx.cpp:222-237,263-265,325-342`)

## 5. Which proposal source wins

Let `remaining` be the number of still-requested tokens **after** the
already returned first token, and `position` be the engine frontier
position. First reserve one output token for the main model's correction:

```text
room = min(remaining-1, max_context-position-1)
short_limit = max(0,min(7,room))
all_limit = max(0,min(15,room))
matched_length, copy = copy_lookup(frontier_index,all_limit)

copy_limit = all_limit if
    (previous_chain_was_fully_accepted and matched_length >= 6)
    or matched_length >= 8
else short_limit
copy = copy truncated to copy_limit

copy_eligible = matched_length >= 4 and copy is nonempty
choose_copy = copy_eligible and copy_acceptance_ema > neural_acceptance_ema
```

Both acceptance averages start at `3.0`. A tie chooses the neural drafter.
The `d0_priority_n=3` generic source priority is not consulted by this
shipping block-draft choice; it belongs to the alternative non-block paths.
(`src/specx.cpp:178-183,212-250,271,325-342`)

When copy wins, every fourth such choice also runs a neural draft as a
probe, without using its tokens for the current verification. The chosen
copy may be extended by the next-token table up to `copy_limit`. When
neural wins, its proposals are used directly; if copy was eligible, retain
the copy proposal for later comparison against actual output. Before either
choice, insert any pending committed main-model features into the draft
rings, even if no neural block is requested.
(`src/specx.cpp:245-270`; `src/engine35.cpp:1838-1851`)

For a chosen source with an eligible copy match, update its average by
`average += (accepted_drafts-average)/16.0`. This uses acceptance before
EOS truncation and before any side-candidate extension. For an unchosen
probe or copy, compare its stored proposal with subsequent actual output;
wait until either a mismatch or the entire proposal can be compared, then
apply the same update using its matching prefix length.
(`src/specx.cpp:383-384,441-447`)

If `room<=0`, no proposal is drafted and the main model verifies just the
frontier. If no neural block is needed, pending context is still caught
up. Cancellation is checked at the next loop boundary; it does not
interrupt a partial state commit.
(`src/specx.cpp:203,212-215,246,269-270,344-374,435-440`)

## 6. Adaptive 8-row versus 16-row neural blocks

The default environment leaves the width-controller variables unset, so
`YSKN_BLK16` uses mode 2. Its numerical defaults are ratio `0.22`, probe
interval `256`, decay `0.9`, backoff enabled, maximum probe interval `256`,
full-8 threshold `0.45`, minimum extra yield `0.6`, extra-yield decay on an
8-row pass `0.97`, and minimum full-8 streak interval `4`.
(`src/specx.cpp:142-154`; `python/yasakani/runtime.py:27-44,93-101`)

The controller uses binary64 averages. At the start of a request set
`extra16=0`, `yield8=4`, `since16=0`, `probe_interval=256`,
`full8=0`, `observations16=0`. With `long_room=min(15,room)`, choose a
16-row neural block if `long_room>7` and **any** of:

```text
extra16 >= max(0.6,0.22*yield8)
since16 >= probe_interval
full8 >= 0.45 and since16 >= 4
```

That choice proposes `long_room` tokens; otherwise it proposes
`min(7,copy_limit)` tokens. The neural transformer still executes a full
8-row or 16-row block, even if remaining room means verification consumes
only a prefix of its proposals. A scheduled neural probe during a copy
round forces the 16-row block whenever `long_room>7`.
(`src/specx.cpp:179,240-257`; `src/engine35.cpp:1851`)

After a neural verification, let `a` be its matching draft-prefix length
before EOS truncation or a side hit:

```text
yield8 = 0.9*yield8 + (1.0-0.9)*(min(a,7)+1)
if this was a 16-row proposal:
    observe16(a)
else:
    since16 += 1
    full8 = 0.75*full8 + 0.25*(1 if a>=7 else 0)
    extra16 = extra16*0.97

observe16(a):
    extra = float64(max(0,a-7))
    if observations16 == 0: extra16 = extra
    else: extra16 = 0.9*extra16 + (1.0-0.9)*extra
    observations16 += 1
    since16 = 0
    if extra16 >= max(0.6,0.22*yield8): probe_interval = 256
    else: probe_interval = min(256,probe_interval*2)
```

Keep the written `1.0-0.9` expression; its binary64 rounded value is not
necessarily the literal `0.1`. Under the shipping equal initial/max probe
intervals, backoff leaves the interval at 256. A completed deferred neural
probe longer than seven tokens also updates `yield8` then calls
`observe16(matching_prefix_length)`.
(`src/specx.cpp:143-159,377-380,441-447`)

## 7. HCAP side-candidate construction

HCAP is eligible only when using neural proposals, no 16-row block was
selected, seven draft slots are available, and eight rows fit in context.
The candidate selector provides a first, second and third token and the
first-to-second margin at each of seven depths. With greedy or version-2
main sampling the draft coupling value is negative and HCAP retains the
straight seven-token chain. With version-3 sampling, it chooses between
four and seven trunk drafts, allocating the remaining slots to leaves.
(`src/specx.cpp:253-257`; `src/engine35.cpp:1845-1850`;
`src/q35_bdraft.cu:659-705`)

The probability tables are reproduced in section 11 so an independent
reader need not open the source. For each depth `j=0..6`, count how many
of the eleven margin edges satisfy `margin[j]>=edge`; this is bin `b`.
Use separately rounded binary32 operations in the following order:

```text
reach[0] = 1
for j = 0 .. 6:
    p = acceptance_probability[j,b]
    value_second[j] = (reach[j]*(1-p))*second_probability[j,b]
                     if second_id[j]>=0 else -1
    value_third[j] = (reach[j]*(1-p))*third_probability[j,b]
                    if second_id[j]>=0 and third_id[j]>=0 else -1
    reach[j+1] = reach[j]*p
```

For trunk length `L` in descending order `7,6,5,4`, start its estimated
yield at `1`, adding `reach[1]..reach[L]` serially. Fill `7-L` leaf slots
greedily. For each slot, scan depths `j=0..L-1` in ascending order: a second
candidate is available if not already selected; a third candidate is
available only after that depth's second candidate has been selected.
Start the winning value at `-0.5` and replace only on strict greater-than;
add each chosen leaf value to the estimate. Discard an `L` if too few valid
leaves exist. Replace the best plan only for a strictly larger estimate,
so tied plans retain the longer trunk encountered first.
(`src/q35_bdraft.cu:671-696`)

Sort chosen leaves by depth, then candidate rank. Physical rows are the
frontier at 0, trunk depths `1..L`, then leaves. The 48-byte row table is:

| Bytes | Meaning | Evidence |
|---|---|---|
| `0..15` | Logical depth for each physical row | `src/q35_bdraft.cu:699-702` |
| `16..31` | Traversal order: each trunk row, followed by leaves at its next depth; leaf row number has bit `0x80` set | `src/q35_bdraft.cu:703` |
| `32..47` | Zero for trunk rows; 1 for second-candidate leaves, 2 for third-candidate leaves | `src/q35_bdraft.cu:699-702` |

A leaf at depth `d` represents an alternative input token after trunk depth
`d-1`. Convolution uses the trunk's earlier rows plus that leaf's own input;
recurrent state is evaluated from the trunk state at its parent without
modifying the continuing trunk; RoPE, causal masks, and sampler positions
use logical depth. Those numerical details are in topics 03 and 04.
(`src/lfrow_kernels.cuh:39-68,79-142`;
`src/q35_sample.cu:180-186`; `src/q35_bdraft.cu:700-703`)

## 8. Verification, acceptance, promotion, and commit

Verify the frontier plus the chosen trunk and optional leaves. There are
`M=trunk_length+1+leaf_count` physical rows. For a host copy chain call
`verify`; for neural tokens call `verify_device`. Each runs the 64-layer
pass and produces `main_next[row]` and raw-logit top-eight lists. Read back
the actual neural row IDs before comparing; row 0 must equal the frontier.
(`src/specx.cpp:344-374`; `src/engine35.cpp:1299-1317,1519-1538`)

When a multi-row verification has no freshly supplied HCAP table, construct
an identity table: bytes `i` and `16+i` equal `i` for `i=0..15`, and bytes
`32..47` are zero. This also applies to copy and 16-row neural passes during
the HCAP-enabled scope. Clear the "fresh table supplied" flag after each
verification, so a subsequent straight pass cannot reuse a previous leaf
table. The one-row precommit path bypasses row-table evaluation.
(`src/engine35.cpp:871-873,1303-1305,1523-1525`;
`src/engine35.hpp:214,364`; `src/specx.cpp:357`)

The core acceptance algorithm is:

```text
a = 0
while a < trunk_length and trunk[a] == main_next[a]:
    a += 1
a_all = a
update_width_and_source_estimators(a_all)

stop = false
for j = 0 .. a_all-1:
    if trunk[j] is EOS:
        a = j
        stop = true
        break
if not stop and main_next[a] is EOS:
    stop = true

hit_row = none
if not stop and a == a_all and a < trunk_length:
    for leaf in physical leaf order:
        if leaf.depth == a+1 and leaf.token == main_next[a] and hit_row is none:
            hit_row = leaf.row

if hit_row exists:
    promote_row(hit_row, a+1)
accepted_argument = a+1 if hit_row exists else a
commit(accepted_argument)
emit trunk[0:a]
emit main_next[a]
if hit_row exists:
    emit original_main_next[hit_row]
    stop if that final token is EOS
```

The EOS IDs under shipping settings are 248046 and 248044. The selected
EOS is included in output. If the prompt's first prediction is already
EOS, the generation loop produces no additional tokens. Matching a side
candidate yields one additional main-model token, not an arbitrary suffix
of that leaf. (`python/yasakani/runtime.py:20,57`;
`src/specx.cpp:166-168,376-440`)

Promotion copies the chosen leaf's cached convolution inputs, gates and
Q/K/V intermediates, main attention K/V and scales, captured draft features,
token and selected-next-token into contiguous row `a+1`. The subsequent
commit can therefore replay the accepted path as a straight sequence.
The full attention cache already contains provisional rows; commit keeps
the accepted prefix by advancing position and does not rewrite all K/V.
(`src/engine35.cpp:877-899,843-851,1319-1329`)

For a usual multi-row verification, recurrent matrices/history remain
uncommitted during verification. `commit(a)` replays exactly `a+1` input
rows through each recurrent layer using cached convolution outputs. It
increments position by `a+1` and sets the next frontier to prediction row
`a`. For `M=1` inside the speculation scope, the default option commits the
single row during verification; later `commit(0)` advances bookkeeping
without replaying that row again.
(`src/engine35.cpp:843-851,1299-1305,1319-1329,1519-1525`;
`src/engine35.hpp:361,367-368`)

Before the next main verification or a snapshot, pending asynchronous
replay must be complete. A scalar implementation can execute that replay
synchronously in the order above: the draft reads captured features, not
the recurrent matrix being replayed. This dependency, rather than a GPU
stream schedule, is the required state contract.
(`src/engine35.hpp:324-325`; `src/engine35.cpp:843-865,964-965,1551-1552`;
`src/engine35.cpp:1767-1782`)

### Table updates and draft context after a pass

Before commit, record rejected trunk rows `j=a+1..trunk_length` first,
then accepted input rows `j=0..a` in increasing order. A row's context is
its own speculative predecessor chain, with the two tokens before the
frontier used for negative local indices. If a side candidate matched,
record its prediction with the accepted parent context last. This order
ensures later accepted records overwrite earlier rejected records sharing
the same table keys. Leaves that did not match are not recorded.
(`src/specx.cpp:427-431`)

After commit, pending draft context comprises `accepted_argument+1` rows
starting at the old frontier position; insert their captured features
before the next draft. The newly emitted final frontier is not yet part
of that context. Always flush pending context after exiting the loop,
including EOS, token-limit and cancellation exits.
(`src/specx.cpp:435-440,464-465`)

## 9. One-row/multi-row differences and limits of equivalence

The source forces equality of the **selected main token** against the
proposal at every accepted depth; it does not perform a probability-ratio
accept/reject sampler. Its random key is determined by absolute position
and token, as specified in topic 06. Thus the control logic preserves the
main model's choice whenever the underlying one-row and multi-row numeric
paths provide the same scores; it does not by itself prove those paths
produce bit-identical scores.
(`src/specx.cpp:376,438-440`; `src/q35_sample.cu:180-197`;
`src/smp_hash.cuh:12-15`)

Explicit differences to preserve are the one-row precommit optimization,
HCAP's leaf-versus-trunk state semantics, full 8/16-row **noncausal** draft
blocks even when verifying only their prefix, and the width/source
estimators above. Main-model attention and recurrent arithmetic differences
are recorded in topics 03/04 and [OPEN_QUESTIONS](OPEN_QUESTIONS.md).
(`src/engine35.cpp:1303-1305,1322-1323,1851`;
`src/lfrow_kernels.cuh:119-140`; `src/q35_bdraft.cu:135-150`)

The overview's wording that a block produces "8 rows (or 16 rows) of
continuation" should be read as total block rows: the implementation uses
one frontier row plus 7/15 proposal positions. Its description of switching
after full 8-row acceptance is an overview of the heuristic; source also
uses expected extra yield and periodic probing. Its claim of unchanged
text is an empirical report, and the same document explicitly reports
unexplained divergent runs. No numeric equivalence proof is available in
this control code.
(`docs/TECHNOLOGY.md:190,211,273-278`;
`src/engine35.cpp:1794,1818`; `src/specx.cpp:240-244,377-380`)

The overview also describes the next-token table as remembering tokens
the main model "chose". Under sampling, it actually records raw-logit
`top8[0]`, which can differ from the sampled choice. This distinction
changes later table proposals. (`docs/TECHNOLOGY.md:192`;
`src/specx.cpp:89-92`; `src/q35_sample.cu:177-205`)

**UNCERTAIN [SPEC-01]: Bit-exact decode/speculation equivalence.** Static
inspection establishes acceptance, state bookkeeping, and shared random
keys, but does not establish last-bit equality of every main arithmetic
path or explain the upstream report of nondeterministic text. In addition,
DRF-01 identifies a scheduling-dependent draft shortlist. A draft-only
ambiguity need not change accepted output when main verification is
equivalent, but that implication is conditional. Checked
`src/specx.cpp:344-440`, `src/engine35.cpp:824-865,1299-1329`,
`src/q35_bdraft.cu:882-896`, and `docs/TECHNOLOGY.md:273-278`.

## 10. Shipping call sequence

The algorithmically relevant sequence is: `yskn_open` selects `Engine35`;
`yskn_load_bdraft` installs five capture layers and `.shead`; request options
set the sampler; `yskn_spec_generate` creates `Specx35`; snapshot-aware
`prefill_rows` produces the first frontier and populates tables/rings;
each loop chooses copy or block drafts, verifies all physical rows, records
table entries, promotes a matching leaf if present, commits accepted input
rows, and queues their draft context; finally context is flushed and a live
prefix note is recorded. (`src/c_api.cpp:15-18,37-74,117`;
`src/engine35.cpp:1650-1694`; `src/specx.cpp:94-126,245-270,344-465`)

## 11. HCAP constants

All following decimal values are source literals rounded to binary32.
Rows correspond to zero-based proposal depths `j=0..6`, columns to margin
bins `b=0..11`. The eleven edges are from `src/hcap_tables.inc:4`; the three
tables are respectively `src/hcap_tables.inc:5`, `src/hcap_tables.inc:6`,
and `src/hcap_tables.inc:7`. Their upload order is edges, acceptance,
second-choice, third-choice. (`src/engine35.cpp:1689-1692`)

### Margin edges

```text
[0.186539, 0.385496, 0.598737, 0.830824, 1.084383, 1.365820, 1.683466, 2.051512, 2.495639, 3.070293, 3.949540]
```

(`src/hcap_tables.inc:4`)

### Acceptance probability

| Depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.363587 | 0.412169 | 0.461378 | 0.514215 | 0.569099 | 0.621929 | 0.672807 | 0.721574 | 0.768938 | 0.810266 | 0.852945 | 0.970498 |
| 1 | 0.283791 | 0.312025 | 0.342764 | 0.374522 | 0.409375 | 0.447879 | 0.487230 | 0.526972 | 0.568991 | 0.613435 | 0.663385 | 0.920325 |
| 2 | 0.241107 | 0.260027 | 0.283105 | 0.308746 | 0.334621 | 0.364635 | 0.395015 | 0.430299 | 0.466114 | 0.507399 | 0.552242 | 0.873328 |
| 3 | 0.211221 | 0.227988 | 0.246409 | 0.265718 | 0.289547 | 0.314255 | 0.339722 | 0.369971 | 0.401724 | 0.437391 | 0.478895 | 0.832413 |
| 4 | 0.191203 | 0.204283 | 0.220734 | 0.237632 | 0.256963 | 0.278877 | 0.302741 | 0.326572 | 0.356635 | 0.389770 | 0.429253 | 0.795850 |
| 5 | 0.175329 | 0.187878 | 0.201498 | 0.216534 | 0.232560 | 0.253138 | 0.274030 | 0.296641 | 0.324308 | 0.354180 | 0.390432 | 0.763373 |
| 6 | 0.165553 | 0.175673 | 0.187796 | 0.200892 | 0.216223 | 0.233279 | 0.252934 | 0.273693 | 0.297893 | 0.326877 | 0.361387 | 0.729952 |

(`src/hcap_tables.inc:5`)

### Second-candidate probability

| Depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.516747 | 0.505416 | 0.489837 | 0.474587 | 0.457495 | 0.434928 | 0.408110 | 0.385292 | 0.355240 | 0.319479 | 0.285953 | 0.044525 |
| 1 | 0.369071 | 0.362097 | 0.354966 | 0.345981 | 0.337467 | 0.324954 | 0.314427 | 0.298426 | 0.282536 | 0.266699 | 0.246044 | 0.048602 |
| 2 | 0.297355 | 0.292798 | 0.286689 | 0.281110 | 0.275441 | 0.267575 | 0.259744 | 0.250181 | 0.242159 | 0.229813 | 0.214035 | 0.044564 |
| 3 | 0.254333 | 0.251244 | 0.248415 | 0.243173 | 0.238882 | 0.233862 | 0.228120 | 0.222454 | 0.214160 | 0.205789 | 0.191683 | 0.042703 |
| 4 | 0.223977 | 0.221177 | 0.219051 | 0.216898 | 0.213247 | 0.209111 | 0.205713 | 0.200284 | 0.195099 | 0.187441 | 0.176088 | 0.040240 |
| 5 | 0.203252 | 0.202602 | 0.199866 | 0.197334 | 0.195556 | 0.192531 | 0.191037 | 0.186359 | 0.181711 | 0.175799 | 0.168390 | 0.040678 |
| 6 | 0.188786 | 0.187403 | 0.186063 | 0.184202 | 0.182566 | 0.180469 | 0.178181 | 0.174783 | 0.171189 | 0.166887 | 0.160430 | 0.040088 |

(`src/hcap_tables.inc:6`)

### Third-candidate probability

| Depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.131337 | 0.128811 | 0.128016 | 0.124097 | 0.120487 | 0.116962 | 0.110280 | 0.102771 | 0.092491 | 0.081382 | 0.068897 | 0.010548 |
| 1 | 0.117498 | 0.116134 | 0.112120 | 0.109761 | 0.105927 | 0.101057 | 0.096692 | 0.090461 | 0.083443 | 0.074258 | 0.063302 | 0.011769 |
| 2 | 0.105534 | 0.103611 | 0.102070 | 0.099594 | 0.096336 | 0.093081 | 0.087985 | 0.082693 | 0.076037 | 0.068189 | 0.059665 | 0.011349 |
| 3 | 0.098274 | 0.096606 | 0.094457 | 0.092464 | 0.089978 | 0.086484 | 0.082048 | 0.077081 | 0.071834 | 0.064851 | 0.056775 | 0.011458 |
| 4 | 0.092278 | 0.090798 | 0.087887 | 0.086300 | 0.084088 | 0.081165 | 0.077192 | 0.072502 | 0.067965 | 0.061321 | 0.053730 | 0.011171 |
| 5 | 0.087546 | 0.085567 | 0.084156 | 0.082657 | 0.080102 | 0.077416 | 0.074368 | 0.070272 | 0.065136 | 0.059739 | 0.051921 | 0.011169 |
| 6 | 0.084400 | 0.083685 | 0.081487 | 0.079582 | 0.078168 | 0.074986 | 0.071933 | 0.068266 | 0.064263 | 0.058578 | 0.051723 | 0.011425 |

(`src/hcap_tables.inc:7`)
