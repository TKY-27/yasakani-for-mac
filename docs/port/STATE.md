# Project state

This file is the single resumable record of the port. It is written so the
orchestrator can continue from it alone. Update it at every merge, decision
and measurement.

Last updated: 2026-11-14 19:20 JST (steps 1 to 6 merged; step 7 in its third round; release instruments being built). **Read "Work in flight" first when resuming.** Times before 18:05 in this file were estimated and can be up to 40 minutes late; later ones come from `date`.

## Work in flight (2026-11-14 19:20 JST)

Each line: what, who, where, how to check it. Briefs are in
`.work/briefs/`, worker reports in `.work/reports/`, real-model run
outputs in `.work/runs/`, scratch notes in `.work/NOTES.md` (all ignored
by git; if they are lost, the brief can be rewritten from this file).

Main is at the merge of PR #30. Merged today after this file's last
update: PR #25 (CPU stage fast paths, D20), #26 (product on the real
engine, D19), #27 (this file), #28 (CI test cost), #29 (fast neural draft
and adaptive depth, D21), #30 (CPU stages round two and grouped attention
for long contexts, D20).

| Item | Worker | Worktree and branch | State |
|---|---|---|---|
| PR #31 kernel round two (D22): exact binary32 multiply-accumulate on the GPU, CPU share of rows during the kernel, `bench_bandwidth` | Opus wrote and fixed; Sol reviewed twice (approve) | `.work/wt/07d`, `feat/07d-kernel-2` | Device proof at acacf28: 433 pass, 117 with `YSKN_BACKEND=cpu`, 117 with `YSKN_CPU_STAGES=reference`. CI running; real-model self-check and `verify_model` running. Merge on green |
| Server idle timeout must not cancel a request being worked on | Opus (brief `06n`); Astra reviewed (approve, three follow-ups) | `.work/wt/06n`, `fix/server-idle-timeout` | Opus doing the follow-ups (a test with keep-alive interval above the idle timeout, README precision). Then merge main in, device proof, PR |
| Soak harness for release criterion 4 | Opus (briefs `09a`, `09b`); Sol reviewed (seven must-fix), re-review running | `.work/wt/09a`, `feat/09a-soak` | Fixes done (nine commits). After approval: merge main in, PR. Real run only after the idle-timeout fix is merged (single-request fill) |
| Lossless check tool for release criterion 3 | Sol (briefs `09c`, `09d`); Astra reviewed (five must-fix) | `.work/wt/09c`, `feat/09c-lossless-check` | Sol fixing (sampling shown to be in effect, non-empty outputs, real prefix-reuse pairs, per-sampler speculation activity, longer outputs, profiles) |

Workers that load the real model take the lock directory
`.work/model.lock` (`mkdir` to take, `rmdir` to release) so that only one
process holds the weights. The orchestrator takes the same lock for its
own runs.

**Every timing taken on 2026-11-14 after 16:00 is provisional**: five
workers were compiling and testing beside the model and swap in use
reached 9.25 GB of 10.24 GB (18:04). Pass counts, token equality and
test results are exact and do not depend on load.

Next, in order: (a) a quiet-machine measurement session once the workers
are idle (one row, four and sixteen rows per pass with stage times;
`bench_metal --shares` to set the CPU share table; `yskn bench` at the
default size for no draft and the default draft; attention at 2,000 and
32,768 tokens; `bench_bandwidth`), recorded as M15 onward; (b) step 8
brief: the product honours the forced-off switches (it builds its model
through explicit constructors, so `YSKN_METAL_MAC` and
`YSKN_METAL_CPU_SHARE` are not read today), start-up checks, behaviour
on other Macs (smaller `maxBufferLength`, other core counts, the CPU
share and draft cost tables calibrated only for the M5); (c) a draft
cost table that depends on context length (attention cost per row grows
with context; the table in D21 is for short context); (d) the lossless
run and the soak run on the real model, alone on the machine; (e) both
perplexity gates on the shipping binary; (f) whole-repository review
(O12); (g) tag, then the SwiftUI app; (h) repository cleanup and
documents; (i) release summary.

## Goal and release criteria

A from-scratch Rust engine for Apple Silicon that runs Qwen3.8-27B from the
published YASAKANI `.yskn` files with quality equal to upstream. Unofficial,
Apache-2.0. The work ends at release-ready; publishing is the owner's step.

Release criteria (all must hold):

1. Perplexity of the fastest model with the int8 cache agrees with upstream's
   2.075900 over 20,460 positions within 0.001, with the shipping
   configuration. The KV5 configuration matches its record (2.075937) equally
   closely. **Finding F1 (2026-11-14):** upstream's "int8" record is not a
   plain int8 cache. See "Step 3 finding" below.
2. Every fast path equals the scalar reference bit for bit in the test suite.
3. Speculation on/off gives the same token sequence for greedy and seeded
   sampling across a varied prompt set; repeated runs are identical.
4. A one-hour mixed workload at 32,768-token context: no crash, no memory
   growth, no swapping.
5. Server passes its API tests, including streaming and tool calls.
6. A worker given only the README builds from a fresh clone, runs a chat and
   the benchmark, and reports no guessing.
7. CI passes on main: formatting, lints as errors, tests.
8. A model that wrote none of the code has reviewed the whole repository for
   correctness and server security; findings resolved.
9. Licence files and notices correct; README states unofficial status,
   hardware needs and what was not tested.
10. This file ends with a release summary.

## Build order

1. Close gaps D7 to D9; mixed-model tests pass. **Done (PR #3).**
2. Fast CPU path for the compressed linear operator (threaded NEON, run-time
   feature choice, portable fallback), bit-identical, model memory-mapped.
   **Done (PR #4).** Crate `crates/yskn-engine`; design in DECISIONS D10.
3. Validate the main model with `score_windows` (gate for everything later).
   **Done: main model validated and both published configurations pass (M8, M9).** The two runs must be repeated on the shipping binary.
   - Plain int8 cache (D4), all 20 windows on the Metal backend:
     perplexity **2.075440** over 20,460 positions (M4). Upstream's "int8"
     record is 2.075900 for profile-requantized rows (finding F1), so the
     difference of 0.00046 is inside the 0.001 tolerance and has the
     expected sign (no 5-bit quantization of keys and values).
   - The engine equals the scalar reference on real weights for the first
     2 rows, and six CPU and Metal configurations agree over 24 rows (M6).
   - **Gate 1 passed (M8):** `Int8Profile`, all 20 windows, perplexity
     **2.075597** against upstream's 2.075900 (difference 0.000303).
   - `verify_model --kv int8-profile` and `--kv kv5`: 24 rows bit-identical
     across four CPU and two Metal configurations each.
   - **Gate 2 passed (M9):** `Kv5`, all 20 windows, perplexity **2.075687**
     against upstream's 2.0759369 (difference 0.000250). It needed PR #21
     first: the reference's packed-row read path was about 150 times
     slower than `Int8Profile` (rows decoded per channel per read); rows
     are now decoded once and kept as bfloat16 bit patterns.
   - **Upstream's self-check outputs are reproduced (M10):** greedy
     generation in `Kv5` gives exactly the token sequences of
     `selfcheck.json` for all three cases (3, 32 and 160 tokens), with and
     without n-gram drafting.
   - Engine provisional pass and partial commit equal the scalar
     reference on real weights (M10).
   - Both perplexity runs must be repeated on the final shipping binary
     before release (criterion 1 says "with the shipping configuration").
4. Backend decision by measurement (NEON vs Metal at 1, 2, 4, 8, 16 rows).
   **Done: D12, Metal** (PR #8). The engine's `Model` runs on it (PR #13,
   D14); the Metal code now lives in `crates/yskn-engine/src/metal.rs`.
5. Draft model, speculation loop, prompt snapshots, long-prompt prefill, KV5.
   **Merged** (prompt reuse and cancellable prefill came with PR #26).
   The wide passes of prompt prefill are made cheaper by PR #31, in CI:
   see the unit table under "Step 5 plan".
6. Product binary: terminal chat, OpenAI-compatible server, benchmark.
   **In progress.** `crates/yskn-text` merged (PR #12): tokenizer, chat
   template, stream decoding, tool helpers; 153 goldens generated from
   upstream's Python, all matching id for id. Brief `06d` (Sol, worktree
   `.work/wt/06d`, branch `feat/06d-cli`, crate `crates/yskn-cli`, binary
   `yskn`): chat logic and the hyper-based API server ported from
   upstream's Python and its tests, against an engine trait with a stub.
   It is PR #18. Sol's sandbox cannot open sockets, so Opus ran and fixed
   the socket tests (bounded lingering close). Astra's security review
   found a memory-exhaustion blocker and five other issues; Opus fixed
   them (limits table in `crates/yskn-cli/README.md`; default body limit
   8 MiB, upstream 32 MiB); Astra's re-review found six more (error
   responses echoing request data outside the budget, budget undercount
   for large strings, uncancellable prompt preparation, an unsound early
   context check, `chat_template_kwargs.tools` bypass, unescaped
   top-level errors); a second Opus fix round (brief `06g`) followed.
   The real engine, prompt reuse, `bench` and `selfcheck` came with
   PR #26 and the draft policy with PR #29 (see the updates below).
   **Update 14:20:** PR #18 merged after Astra's fourth review (approve;
   worst case under default settings by its estimate about 1.4 GiB per
   request and 1.6 GiB for all connections, no permanent stall, no
   cross-origin path; new limits: decoded text 2 MiB, rendered prompt
   2 MiB, global budget 512 MiB, queue wait 600 s). Brief `06i` (Sol,
   worktree `.work/wt/06i`, branch `feat/06i-product-engine`) connects the
   real engine: sampling mapping, prompt reuse, cancellable prefill,
   memory check at start, `yskn selfcheck`, `yskn bench`, `yskn info`.
   Sol's sandbox cannot open sockets or use the GPU, so the orchestrator
   runs those tests.
   **Update 16:45:** brief `06i` is done and is PR #26. The product binary
   on the real model: `yskn selfcheck` passes all three upstream cases,
   `yskn info` and a one-shot `yskn chat` run (M14). Open: Astra's review;
   `--draft neural` still loads the scalar draft (brief `06k` switches it
   to the fast draft and adds the depth policy).
   **Update 19:20:** PR #26 and PR #29 are merged. Defaults are
   `--draft both --k 7 --depth adaptive`; the product reproduces
   upstream's self-check with them (M15). Open in step 6: the server's
   idle timeout cancelled long silent prefills (found by the soak
   harness author; fix in review, brief `06n`).
7. Optimization to the machine limit (stop after five attempts under 2%).
   **Under way.** Attempts so far (serious attempts, with the effect
   each had; all but the first measured under load and to be confirmed
   in the quiet session):
   1. CPU stage fast paths, PR #25: one row 264 ms to 232 ms, 12% (M13).
   2. CPU stages round two, PR #30: row encoding 11.8 to 3.6 ms,
      recurrence 40.4 to 13.8 ms, rest 29.1 to 27.4 ms at one row (CPU
      stage timers, M16). The stage minima come from separate runs, so
      their sum (36.5 ms) is not a measured gain of the whole pass; the
      whole pass is measured in the quiet session.
   3. Grouped attention, PR #30: the attention stage alone at 2,000
      tokens and 16 rows 0.228 to 0.086 s, at 32,768 tokens and one row
      0.452 to 0.192 s (M16): a 57% cut of a stage that at full context
      was larger than the rest of a one-row pass.
   4. Neural draft with adaptive depth, PR #29: passes for 96 tokens
      36 / 19 / 24 / 21 against 96 (M15); projected 123 ms per token
      against 232 ms.
   5. Exact binary32 arithmetic in the GPU kernel, PR #31: per added
      row 35.8 to 26.2 ms; 16 rows 638 to 492 ms (M17).
   6. CPU share of rows during the kernel, PR #31: 16 rows 554 to
      476 ms in `bench_metal --shares`, which issues the kernel calls in
      the engine's grouping (257 per pass); about 7% at one row, not
      established under load (M17).
   7. Submission overhead at one row: nothing kept, 0% (M17). About
      36 ms per pass is fixed latency of 257 command-buffer round trips
      (0.14 ms each); every waiting method tried gave the same.
   The count of consecutive attempts under 2% is 1 (attempt 7).
8. Other Macs: feature detection, forced fallbacks, memory check at start.
9. SwiftUI chat app (only after the engine passes every criterion; tag first).
10. Repository cleanup and documents written from nothing.

## Step 3 finding F1: upstream's "int8" cache is profile-requantized

Verified in the source by the orchestrator: the shipping build defines
`YSKN_KVX_MODE=3` (`CMakeLists.txt:46`). In that mode every key/value row of
a full-attention layer gets `X.tb = tb_dev_ptr()` and
`launch_tb_requant(X, st_)` runs whenever `X.tb && X.k8 && X.v8`
(`src/engine35.cpp:688-704`), also with `YSKN_TB_CAP=0`, which is all that
`tools/quality/score_yasakani.py --kv int8` sets (line 63). So the 2.075900
record is "int8 rows re-quantized through `kvq_u5.bin`", and the reference's
D4 cache (plain int8) is a third configuration that upstream never
published a number for. Consequences:

- The reference needs three cache modes (now decision D13): `Int8`
  (plain, D4), `Int8Profile` (upstream's "int8" record), `Kv5` (upstream's
  default, packed 5-bit where packable).
- The two published values differ by 3.7e-5 and the tolerance is 0.001, so
  perplexity cannot detect a subtly wrong codec. The codec is proven by its
  own tests (optimality against an independent dynamic program, round
  trips, frozen digests, facts of the real profile). The encoder uses only
  integers and correctly rounded IEEE operations, so it is ported from the
  source operation for operation.
- The plain-int8 run tells whether the rest of the main model is right:
  its result should be within about 0.001 of 2.0759 (it was: 2.075440,
  M4). A larger gap would have been a bug or the codec's effect;
  `Int8Profile` separates the two.

## Step 5 plan (from an Opus 5.5 planning pass over chapters 04, 07 to 10)

Key facts established by the plan (citations in its report, kept in
`.work/reports/` is not guaranteed; the essentials are here):

- Upstream's speculation control logic is lossless: a drafted token is
  accepted only if it equals the token the main model's sampler selects,
  and the sampler's randomness depends only on seed, absolute position and
  token (08 §8, 06 §5). Its numerics are not row-count independent
  (SPEC-01), which this port avoids by D1 and D9. Here speculation is
  lossless by construction if every fast path equals the reference.
- Upstream commits by replaying the recurrent layers for the accepted rows
  from cached inputs (`src/engine35.cpp:843-851,1319-1329`).
- `selfcheck.json` was produced with KV5 and full speculation; prompt
  hashes are a hard gate for the tokenizer, output hashes a strong check
  for the KV5 engine (a mismatch at a near-tie late in the output can be
  legitimate under D1; an early or large-margin mismatch is a bug).

Units (B, C, I can run in parallel; F before D because both touch `model.rs`):

| Unit | What | Status |
|---|---|---|
| A | Survey of the real `kvq_u5.bin`, draft and `.shead` headers | done inside B/C and I (facts in D13, D15 and the spec) |
| B | `yskn-ref/src/kv_profile.rs`: KVQP reader, fully validated | merged, PR #11 (Sol; Astra review) |
| C | `yskn-ref/src/kv_codec.rs`: row codec ported from the source | merged, PR #11 |
| D | Cache modes `Int8`, `Int8Profile`, `Kv5` in the reference attention, model load and the engine | merged, PR #14 (Sol; Astra reviewed the reference against the source, Opus the engine wiring); both scoring runs passed (M8, M9) |
| E | KV codec fast path | row encoding runs in parallel across heads and sides (PR #25, D20: 45 ms to 13 ms at one row); the codec's candidate selection, then the hottest CPU function (M13), was vectorized in PR #30: 3.6 ms at one row (M16) |
| F | Main-model hooks: residual capture after layers 5, 19, 33, 47, 61; snapshot/restore; head rows | merged: PR #17 and PR #19 (D17; Sol; two Opus reviews) |
| G | Speculation loop over Verifier and Proposer traits, lossless tests | merged, PR #17, D16 (Sol; Opus review) |
| H | Engine `run_provisional`/`commit` without cloning the attention cache | merged, PR #17 |
| I | Draft model reference (chapter 07) and `.shead` | merged, PR #15, D15 (Sol; Astra review against the source) |
| J | Draft fast path (one scalar proposal takes about 90 s on the real files; target under 40 ms, bit-identical) | merged, PR #24, D18 (Opus; Sol review; failure test by Luna): 47 ms per proposal on the real files, bit-identical to the scalar draft (M12) |
| K | Prompt reuse modes (reset, live prefix, snapshot) | merged, PR #26 |
| L | Long-prompt prefill: bounded memory, head on last row only, large-M kernel | cancellable prefill with last-row logits merged in PR #26; cheaper wide passes in PR #31 (binary32 arithmetic, CPU share), in CI |

Feature-capture contract for the draft model (confirmed by Astra against
the source): for each committed input position, the residual right after
the MLP down projection's residual add of zero-based layers 5, 19, 33, 47,
61; the value is `bf(old_residual + bf(down_projection))` expanded to
binary32; one block of 5 × 5120 per position. Before a proposal the
draft's ring holds committed input positions 0..p (frontier excluded);
after a verification the old frontier row and the accepted rows are
pushed, never rejected rows or the new frontier; flush at the end.
Sources: `src/engine35.cpp:798-799,830-833,1650,1851`,
`src/q35_kernels.cu:1437-1444`, `src/q35_bdraft.cu:249-252,490-499`,
`src/specx.cpp:435-437,464`.

Policy decided by measurement, not copied from the RTX 5090 (decision
D21, PR #29; the paragraph below is the plan as written before it): number of drafted rows K (each verified row costs about 35 ms
on the GPU kernel alone), and which proposal sources earn their place.
Upstream's HCAP side candidates, adaptive 8/16-row blocks and the
next-token table are not needed for correctness and are skipped unless a
measurement shows a gain. Method: log per position the main model's choice
and each source's proposals on a varied prompt set, replay offline with the
measured cost model, pick the K and sources that minimise ms per token.

## Environment (observed 2026-11-14)

- Machine: MacBook Pro `Mac17,2`, chip **Apple M5** (the base M5, not an
  M5 Pro: `system_profiler SPHardwareDataType` reports "Apple M5", 10 cores,
  4 Super and 6 Efficiency; GPU 10 cores, Metal 4), 24 GB
  (`sysctl hw.memsize` = 25769803776), macOS 27.2.1, rustc 1.100.0.
  Metal `maxBufferLength` is 14,302,248,960 bytes.
- **Bandwidth ceiling.** The owner's brief gives 307 GB/s nominal, which is
  the M5 Pro figure. This chip is the base M5; its nominal figure is about
  153 GB/s (recalled from Apple's M5 announcement, not verified here). The
  ceiling was then measured (M17): about 145 to 150 GB/s for CPU and
  GPU reading together, 116 to 119 GB/s for the GPU alone; the kernel at
  one row reads 98.7 GB/s.
- Model folder (never commit, never copy): `~/models/qwen38-speed/`.
  Contains `qwen38_27b_speed.yskn` (10,311,344,128 bytes), `bdraft2_k8.yskn`,
  `bdraft2_k8.shead`, `kvq_u5.bin`, `tokenizer.json`, `chat_template.jinja`,
  `yasakani.json`, `selfcheck.json`.
- GitHub: `gh` logged in as TKY-27 with repo and workflow scopes.
- Codex CLI 0.166.0, ChatGPT login active.

## How the work is run

- The orchestrator (Claude Fable) directs and verifies; it does not implement.
- Routing fixed by the owner:
  - Implementation and code review: `gpt-6.1-sol` through the Codex plugin.
  - Optimization, UI and general tasks: Opus 5.5 subagents.
  - Hard problems (numeric mismatch root cause, kernel memory layout):
    `gpt-6-astra` through the Codex plugin.
  - Mechanical edits: `gpt-6-luna` through the Codex plugin.
  - The reviewer of a change is never its author. Sol-written changes are
    reviewed by Opus 5.5 or Astra; Opus-written changes by Sol.
- Briefs live in `.work/briefs/` (ignored by git). A worker sees only its brief.
- One process at a time may load the real model. Timing runs happen with no
  other project process running.
- Feature branches, pull requests, CI green and a cross-model review before
  each merge to main.
- Codex workers sometimes cannot write `.git` (sandbox). Briefs ask them to
  leave work staged; the orchestrator commits after running the proof.
- A Codex task can write only under the directory the orchestrator's
  session is in when it launches the task: launch from the repository
  root. Codex tasks have no network and no GPU.
- Force-pushing is refused by the harness. Never rebase a branch after
  pushing it: merge main into it, or push the rebased work under a new
  branch name and open a new pull request (PR #10 became #12 this way).
- Long real-model runs are started with `nohup` from `.work/runs/` and
  watched with a background wait on their `.exit` file.
- Parallel workers get their own git worktree under `.work/wt/`.

## Decisions

D1 to D21 are in [DECISIONS.md](DECISIONS.md): D10 mapped CPU kernel,
D11 layer-major engine model, D12 Metal backend, D13 KV cache modes and
codec, D14 engine on Metal with parallel recurrent heads, D15 draft model
math and tie rule, D16 sessions, provisional passes, commit, snapshots
and the lossless speculation loop, D17 residual capture and the draft
proposer, D18 fast draft, D19 product engine, D20 CPU stage fast paths
(both rounds and grouped attention), D21 fast draft in the product and
adaptive draft depth. D22 (kernel round two) arrives with PR #31.
Owner-decided gaps:

- **D7** exp2(t) = portable exp of the binary32 product t × ln 2
  (bits 0x3f317218). Reason: upstream calls `exp2f(x * log2e)` and has no
  portable exp2; this is the closest construction from portable parts.
- **D8** For arch `qwen3_5` the rotary width is the constant 64 and the loader
  requires `head_dim == 256`, as upstream hard-codes. Stated exception to D5.
- **D9** Each query row visits the absolute 32-slot tiles that intersect its
  own prefix range and no others. A masked key has score −∞ and probability
  +0.0 and takes no part in value accumulation. A row's result must not
  depend on which other rows share the pass.

Orchestrator decisions:

- **O1 (2026-11-14)** The pstack Orchestrate playbook's external store and
  `orch` CLI are not used; this file is the store. Reason: the owner requires
  that the work be resumable from this file alone, and forbids changes
  outside the repository.
- **O2 (2026-11-14)** `target/` and `.work/` are ignored by git. Reason:
  build output was showing as untracked; briefs and worker reports are
  transient.

- **O9 (2026-11-14)** The GPU kernel may use binary32 arithmetic where
  every operand and partial sum is an integer of magnitude below 2^24.
  Reason: such operations are exact, so the kernel still returns the
  same signed 64-bit sums and bit identity holds by construction; the
  first Metal brief forbade floating point in the kernel only as a
  precaution, and each added row costs 35 ms of integer work, which
  bounds prefill speed. The bound must be proven from the format's
  limits and tested at the extremes (brief `07d`).
- **O10 (2026-11-14)** The draft depth is chosen per proposal by a
  deterministic controller, not fixed. Reason: measured acceptance (M11)
  differs by a factor of three between easy and hard text, and a
  verified row costs 40 to 60 ms here, so no fixed depth is right. The
  controller only changes how many guesses are requested; output cannot
  depend on it (D16). Its cost model comes from constants or a stored
  calibration, never from timing inside the loop, so runs repeat.
- **O11 (2026-11-14)** The one-hour workload of release criterion 4
  fills the context once (about 30,000 prompt tokens) and then runs
  mixed requests against the full context. Reason: prefill costs at
  least 43 ms of kernel time per token today, so a full-context prompt
  takes over 20 minutes; a workload that never reaches the limit would
  not test "at 32,768-token context". If the fill alone exceeds the
  hour, the run is extended, never shortened.
- **O12 (2026-11-14)** The whole-repository review of release
  criterion 8 goes to Sonnet 5.5 at high effort, in parts (one per
  crate plus one for server security), with Astra cross-checking server
  security. Reason: the criterion asks for a model that wrote none of
  the code; Sol, Astra, Opus and Luna all have commits here, the
  orchestrator made small edits and may not start a subagent on its own
  model, and Sonnet has only forwarded tasks to Codex.
- **O13 (2026-11-14)** The server has no wall-clock limit on how long
  one request may hold the engine; the hold is bounded by the token
  limits (prompt at most the context minus the reserve, output at most
  `max_tokens` and the room left), and other requests get the documented
  503 after the queue wait. Reason: a long prompt legitimately takes
  tens of minutes to prefill on this hardware, a cap would cancel it
  again, and the server is bound to localhost for one user. Astra's
  security review agreed, provided the limits table states it.
- **O14 (2026-11-14)** The under-8-minute target for the CI test step is
  given up where it conflicts with proof: the review of the test-cost
  change restored the boundary rollbacks, multi-guess reuse and
  two-attention-layer fixtures, and the job now takes about 20 minutes
  (29m41s before) with a 45-minute limit. Reason: a release criterion is
  "every fast path equals the scalar reference bit for bit in the test
  suite"; speed of CI is not one.
- **O15 (2026-11-14)** The release run of the lossless check uses four
  configurations (none, n-gram K=3, neural K=7 fixed, both K=7
  adaptive), three samplers (greedy, seeded moderate, seeded hot), each
  run repeated once, 24 prompts of 32 to 256 output tokens: about 43,000
  tokens, 2 to 2.5 hours. Reason: the first version capped outputs at 8
  and 16 tokens to fit 40 minutes, which never exercises back-off,
  probing or long copies.
- **D12 (2026-11-14) Backend: Metal for the compressed linear operator.**
  Measured on the real model (M2 below), whole pass, median ms, Metal vs
  CPU dotprod with 10 threads: 111 vs 646 at 1 row, 128 vs 671 at 2,
  187 vs 758 at 4, 331 vs 924 at 8, 634 vs 1,248 at 16. Metal wins at every
  row count: 5.8 times at 1 row, 2.0 times at 16. Reason for the shape: the
  GPU removes the decode cost (the CPU does one scalar table lookup per
  weight pair), while the cost of each added row is similar on both (about
  35 ms on the GPU, 40 ms on the CPU) because neither has a cheap int8
  product at this scale. Command-buffer granularity matters: one buffer per
  matrix costs about 72 ms more per pass than one per pass, one per layer
  about 12 ms more. The CPU kernel stays as the fallback and the test
  oracle for real tensors. A CPU/GPU split of one matrix's output rows
  could add at most 15% at 1 row and 34% at 16 by rate addition, and the
  CPU is needed for the recurrent and attention layers, so it is deferred
  to step 7 (done there: D22). Recorded in DECISIONS.md as D12.
- **O7 (2026-11-14)** The Metal engine integration (brief `04c`: backend
  selection in `Model`, grouped dispatch, recurrent heads in parallel) and
  the KV codec (brief `05a`) proceed while the plain-int8 score runs.
  Reason: both are needed to run the real gate (finding F1) and to make
  each scoring run take about an hour instead of 2.4; both are held to
  bit-identity with the reference, so they add no numeric risk.
- **O8 (2026-11-14)** The tokenizer and chat template use the `tokenizers`
  crate 0.23.2 (Hugging Face's own implementation, Apache-2.0, pure-Rust
  `fancy-regex` backend) and `minijinja` 3.1 (Apache-2.0) with
  `preserve_order`, instead of hand-written code. Reason: the acceptance
  bar is id-for-id equality with upstream's Python, which uses the same
  tokenizer library; JSON key order changes the prompt (261 vs 262 tokens
  on a tools case). Expected values come from upstream's own
  `python/yasakani/textio.py` run in a development-only virtual
  environment under `.work/venv` (not used at run time).
- **O3 (2026-11-14)** Commits that change only this file go through a pull
  request and CI but need no cross-model review. Reason: the file is the
  orchestrator's own record, not code; a review would add nothing.
- **O4 (2026-11-14)** The Metal probe for step 4 runs in parallel with
  step 3 on synthetic data. Reason: the CPU kernel is compute-bound at
  about 15 GB/s, far from the 307 GB/s ceiling, so the backend choice is
  the largest open question and does not depend on the main-model gate.
  Nothing is built on it until step 3 passes.
- **O6 (2026-11-14)** GPU work goes to Opus 5.5 subagents, not through the
  Codex plugin. Reason: the plugin runs Codex in a sandbox where
  `MTLCreateSystemDefaultDevice()` returns nil and the network is blocked.
  Astra's attempt at the Metal probe wrote no code for that reason. Codex
  models still review GPU code by reading it. New crates are fetched by
  the orchestrator before a Codex task needs them.
- **O5 (2026-11-14)** The engine's main model reuses the reference's scalar
  functions for everything except the compressed linear kernel, and only
  changes the loop order to layer-major. Reason: least new code, and
  bit-identity follows from sharing the functions; faster replacements
  come one at a time in step 7, each with its own bit-identity test.

## Progress log

| Date | Step | What | Evidence |
|---|---|---|---|
| 2026-11-14 | 0 | PR #2 found already merged; CI green on main at d161397 | `gh pr list --state all`, `gh run list` |
| 2026-11-14 | 1 | D7 to D9 implemented by Sol (75271d6). Opus review: code correct, tests could not detect a wrong tile schedule. Sol fix round (8ee8520): `prefix_schedule()` tested against hand-written schedules, pinned bits. 61 tests pass, none ignored. | `cargo test --workspace --release`; `! grep -rnE 'todo!\|unimplemented!\|#\[ignore\|SpecGap' crates/` |
| 2026-11-14 | 2 | Astra built `yskn-engine` (mmap, pool, portable/NEON/dotprod kernels). Sol review: approve, one test nit. Orchestrator ran all checks. | `cargo run --release -p yskn-engine --example verify_linear -- MODEL.yskn` (12 of 12 PASS) |
| 2026-11-14 | 3 | PR #6: layer-major engine `Model`, shared scoring, `verify_model` (Sol; Opus review approve with 6 nits, carried into brief `04c`). | `cargo run --release -p yskn-engine --example verify_model -- MODEL_FOLDER tools/quality/windows_own_thinking.json` (PASS) |
| 2026-11-14 | 4 | PR #8: `crates/yskn-metal` (Opus; Sol review found a table-cache key bug and a racy duplicate table write, both fixed and re-approved). Metal tests also run on CI's paravirtual GPU (448 threads per threadgroup limit; default 128 works). | `cargo test -p yskn-metal --release`; M2 below |
| 2026-11-14 | 5 B/C | PR #11: KV profile reader and row codec (Sol). Astra compared it with the CUDA source statement by statement: approve. Real profile: 128 groups, all 5-bit mode 0, 9 multipliers, 16 key and 13 value sides packable. | `cargo run --release -p yskn-ref --example kv_profile_info -- MODEL_FOLDER/kvq_u5.bin` |
| 2026-11-14 | 3/4 | PR #13: engine `Model` on Metal with grouped dispatch, recurrent heads in parallel, review nits (Opus; Sol review approve). | M5, M6 below |
| 2026-11-14 | 6 | PR #12: `crates/yskn-text` (Sol; two Opus reviews; final goldens by Luna). | `YSKN_MODEL_DIR=MODEL_FOLDER cargo test -p yskn-text --release` (127 model-backed and 26 model-free goldens pass); regenerating goldens with upstream's Python is byte-identical |
| 2026-11-14 | 5 D | PR #14: KV cache modes in reference and engine (Sol; Astra and Opus reviews; fixes for result-file identification and test strength). | `cargo test --workspace --release` (Metal cases ran) |
| 2026-11-14 | 5 I | PR #15: draft model reference and reduced head (Sol; Astra review against the source: approve, two test gaps carried forward). | `YSKN_MODEL_DIR=MODEL_FOLDER cargo test -p yskn-ref --release draft`; one scalar `propose` on the real files takes about 90 s |
| 2026-11-14 | 3 | Plain int8 perplexity 2.075440 (M4): main model validated. | M4 below |
| 2026-11-14 | 5 G/H | PR #17: provisional passes, commit, snapshots in the engine; generic lossless speculation loop (Sol; Opus review approve). | `cargo test --workspace --release` (randomized partial commits equal the reference in every KV mode and backend; generation identical with and without speculation) |
| 2026-11-14 | 3 | Gate 1 passed: profile-int8 perplexity 2.075597 (M8). | M8 below |
| 2026-11-14 | 5 F | PR #19: residual capture, draft proposer, joint snapshots (Sol; Opus reviews). Teacher test: 7 of 7 guesses accepted with correct wiring, 0 of 7 with a one-row lag or swapped slots. | `cargo test --workspace --release` (209 tests, Metal cases ran) |
| 2026-11-14 | 5 D | PR #21: packed KV5 rows decoded once, stored as bfloat16 bits (Sol; Opus review: no changed bit). Kv5 cache: 736 bytes per packed row, 2.73 GiB at 32,768 tokens, 21.9 GiB at 262,144 (does not fit 24 GB). | frozen digests from 0b7bd5d unchanged |
| 2026-11-14 | 6 | PR #18: `crates/yskn-cli` (binary `yskn`): chat logic, hyper server, terminal chat against an engine trait (Sol; socket and security fixes by Opus; four Astra reviews). | `cargo test -p yskn --release` (109 tests, repeated 5 times) |
| 2026-11-14 | 3 | Gate 2 passed: KV5 perplexity 2.075687 (M9). Self-check outputs reproduced (M10). | M9, M10 below |
| 2026-11-14 | CI | PR #22: one test pass in CI (Sol review). The test step still takes about 21 minutes on the runner. | `gh run view` |
| 2026-11-14 | 5 J | PR #24: fast draft, D18 (Opus; Sol review approve; missing failure test added by Luna and run on the device). | `cargo test -p yskn-engine --release --test draft_fast`; `verify_draft` on the real files: fast equals scalar at position 300, Metal equals CPU equals portable (M12) |
| 2026-11-14 | 5 | Draft acceptance measured on the real model (M11). | M11 below |
| 2026-11-14 | 7 | PR #25 merged (187cebb, CI green on its head a939304): CPU stage fast paths, D20 (Astra; Opus review found no changed bit and asked for an aliasing fix, reference error identity and shared bodies; Astra fixed; Sol re-review approve). | `cargo test --workspace --release` (349 pass); `YSKN_CPU_STAGES=reference cargo test -p yskn-engine --release` (73 pass); M13 |
| 2026-11-14 | 6 | PR #26 merged: real engine behind the product binary, `selfcheck`, `bench`, `info`, D19 (Sol; socket test fixture fixed by Opus; Astra review: three test weaknesses fixed by Sol, approve). | `cargo test --workspace --release` (370 pass); product tests 20 runs green with sockets; M14 |
| 2026-11-14 | CI | PR #28 merged: cost of the slowest test binaries (Sol; Astra review restored lost proof: multi-guess prefix reuse, full-ring checkpoint, rollback and re-crossing at the 32 and 64 tile boundaries, two attention layers in six fixtures; approve). Job 29m41s to 20m13s; limit raised to 45 minutes. | CI run of PR #28; five deliberate mutations each caught (author's report) |
| 2026-11-14 | 6, 7 | PR #29 merged: fast neural draft in the product, adaptive draft depth, free long n-gram copies, draft work in bytes per pass, D21 (Opus; Sol review: allocation bound, example input checks, memory accounting, four tests, fixed by Sol; Astra re-review and independent losslessness check: approve). | `cargo test --workspace --release` (389 pass); M15 |
| 2026-11-14 | 7 | PR #30 merged: CPU stages round two and grouped attention, D20 (Astra; Sol review found the reference's dense-gate error order changed and binary16 rounding unconnected, fixed with pinned tests; second review: no differing bit, error or state; two tests added; approve). | `cargo test --workspace --release` (427 pass); 111 with `YSKN_CPU_STAGES=reference`; M16 |

## Process incident (2026-11-14)

PR #19 was merged to main although its CI run had not passed. The
orchestrator's background command waited for the checks and then merged
without testing the result. The run had been cancelled by the 30-minute
job timeout during the duplicate release test pass; the dev-profile pass
of that run had succeeded and the release suite had passed locally (209
tests, Metal cases on the device), so there is no evidence of a broken
main, but the rule "merge once CI passes" was broken. Corrections: every
merge is now guarded by an explicit check that the pull request's checks
report pass; PR #22 removed the duplicate pass; a change written by the
orchestrator (PR #22) was reviewed by Sol before merging.

## Known debts

- CI's test step: about 20 minutes after PR #28 (see O14). What follows
  is the record of the reduction. It took about 21 minutes of a 30-minute
  limit; the aim was under about 8 minutes without dropping coverage. Seconds per test
  binary on the runner (run 43565720700): engine `speculation` 509,
  `draft_generation` 288, reference `draft_wiring` 109, engine `model`
  92, `kv_attention` 57, `draft_fast` 45, compile 213. Brief `08a` (Sol),
  written with the `test-audit` rules: reduce cost, not proof; a ledger
  line per slow test; a mutation check per slow binary.
- **Prefill speed** (figures before PR #30 and PR #31; to be re-measured
  in the quiet session). A 16-row pass costs about 1.2 s (kernel 0.69 s), so
  a prompt token costs at least 43 ms of kernel time plus the CPU
  stages, and attention grows with context (0.31 s per 16-row pass at
  2,000 tokens after PR #25). A 30,000-token prompt takes well over 20
  minutes. Briefs `07d` (cost per row, CPU and GPU together) and `07c`
  (attention at long context) address it; whatever remains is a stated
  limit of this hardware in the README.
- Swap: `sysctl vm.swapusage` showed 5.4 GB used at 16:03 with several
  workers compiling while the real model was loaded. The one-hour run
  must be measured with nothing else running and judged by swap-out
  counters over the run, not by the absolute figure.
- Attention at long context: 0.192 s per row at 32,768 tokens after
  PR #30 (M16), against a 0.10 s target; the cache is read at about
  10.8 GB/s, so there is room for another round (layout, more of the
  work per thread). With attention this size the draft's cost table
  (D21, short context) is wrong at long context: step (c) of the plan.
- The format's maximum context (262,144 tokens) does not fit this machine
  in `Kv5` with the current cache representation; the product caps the
  context by memory (brief `06i`). A compact fast path reading packed
  words (164 bytes per row) would restore the saving; step 7.
- `crates/yskn-cli/README.md`: two documentation nits from Astra's fourth
  review (in brief `06i`).

- `generate` copies the whole recurrent state (about 157 MB) at the start
  of every provisional pass and again on a partial commit (D16); step 7.
- Row encoding: 3.6 ms at one row after PR #30 (was 45 ms); closed.
- Draft tests (from Astra's review of PR #15): a rank-projection test with
  off-diagonal entries and rank different from hidden width; a synthetic
  draft with query heads different from KV heads (real: 32 and 8, mapping
  head / 4); a rotary expectation at a non-zero position; an attention test
  with non-zero queries and keys; head scales other than 1;
  source-derived expectations for the correction chain.
- The KV codec's tie-path equality with the source is argued from the code,
  not proven by a test with source-derived expected values (none exist).
- The bandwidth ceiling for this chip: measured, M17; closed.
- D12 in DECISIONS.md and M2 below were measured before two review fixes to
  the Metal shader; re-measure on a quiet machine during step 7.

- `MappedModel::open` is an `unsafe fn` (file must not change while mapped).
  The product maps the model and the draft at one documented point
  (`RealEngine` loading, PR #26 and PR #29) and prints "Model files must
  not change while yskn is running"; closed.


## Measurements

Every entry carries the exact command that produced it.

### M1: CPU kernel, full pass of the compressed linear operator (2026-11-14)

Command (nothing else running, model resident after warm-up):
`cargo run --release -p yskn-engine --example bench_linear -- ~/models/qwen38-speed/qwen38_27b_speed.yskn`

Commit fe8ae4a (merged as 4bb0cab). Kernel dotprod, 10 threads, 401
matrices, 9,758,310,400 code bytes per pass, 7 passes after one warm-up.

| Rows (M) | Median ms | Minimum ms | Read rate GB/s (median) |
|---:|---:|---:|---:|
| 1 | 645.8 | 634.6 | 15.1 |
| 2 | 671.5 | 665.2 | 14.5 |
| 4 | 757.7 | 746.3 | 12.9 |
| 8 | 924.0 | 908.8 | 10.6 |
| 16 | 1247.6 | 1234.1 | 7.8 |

### M2: Metal kernel, full pass of the compressed linear operator (2026-11-14)

Command (scoring run stopped first, no build or test running):
`cargo run --release -p yskn-metal --example bench_metal -- ~/models/qwen38-speed/qwen38_27b_speed.yskn`

Commit 4e5fe32 (branch `feat/04a-metal-probe`). Device Apple M5,
threadgroup 32x4, whole 10,311,344,128-byte mapping as one no-copy buffer,
401 matrices, 9,758,310,400 code bytes per pass, 7 passes after a warm-up.
Before timing, Metal equalled the CPU kernel on four real tensors (M=3).
Empty dispatch round trip: median 0.18 ms. No swap in use afterwards.

| Rows (M) | (a) one buffer per pass, median ms | (b) one per matrix | (c) one per layer | Read rate GB/s (a) |
|---:|---:|---:|---:|---:|
| 1 | 111.2 | 183.7 | 122.6 | 87.7 |
| 2 | 127.7 | 203.1 | 139.6 | 76.4 |
| 4 | 186.9 | 264.3 | 199.3 | 52.2 |
| 8 | 330.5 | 410.4 | 344.9 | 29.5 |
| 16 | 634.1 | 713.8 | 647.5 | 15.4 |

Times include uploading activations and reading results back.

### M3: where a 16-row pass of the whole model goes, CPU path (2026-11-14)

From the scoring run's stderr (`score_windows`, commit 09266e0, rows per
pass 16, 10 threads), seconds per pass at context about 190: total 2.86,
kernel 1.20, recurrence 1.39, attention 0.14, rest 0.12. The worker's
window-0 run reported 3.73 late in the window (attention 1.04 at context
about 2,000). The recurrent layers and attention are the reference's
scalar code and are now the largest costs.

Reading of M1: the ideal at 307 GB/s is about 32 ms, so the kernel is
compute-bound by a factor of about 20. The decode is scalar (one table
lookup per pair of weights, bounds-checked) and each added row costs about
40 ms. CPU features present: NEON, dotprod, i8mm, SME, SME2
(`sysctl hw.optional.arm`).

### M4: perplexity, plain int8 cache, Metal backend (2026-11-14)

Command (run alone on the model, started 08:50 JST, binaries of commit
1f7c378 whose content is main 61c8b03):
`./score_windows ~/models/qwen38-speed tools/quality/windows_own_thinking.json --backend metal --rows-per-pass 16 --json score_int8_metal.json`
(the example is `cargo run --release -p yskn-engine --example score_windows`).

Result: scored_positions 20460, mean negative log-likelihood
0.730173179294, **perplexity 2.075439999788**. Upstream: 2.075899908604469
("int8", profile-requantized), 2.0759369466709248 (KV5), 2.065188190523882
(original model). Per-window NLL sums are in
`.work/runs/score_int8_metal.err`; windows 0 to 6 equal, to all 17 digits,
the sums of the earlier CPU-backend run that was stopped after 7 windows
(508.71858607572784, 528.80243264123135, 586.94446558963762,
771.03391286075657, 1075.0259708577341, 903.68640769976059,
1080.9851932527472). About 211 to 265 s per window with other work
running on the CPU.

### M5: where a 16-row pass of the whole model goes, Metal backend (2026-11-14)

From the M4 run's stderr at context about 190, seconds per pass: total
1.17, kernel 0.69, recurrence 0.22, attention 0.14, rest 0.11. Before
PR #13 (CPU kernel, sequential recurrent heads, M3): 2.86, 1.20, 1.39,
0.14, 0.12.

### M6: engine against the scalar reference on real weights (2026-11-14)

- `./verify_model ~/models/qwen38-speed tools/quality/windows_own_thinking.json`:
  all 24 logit rows bit-identical across four CPU configurations (dotprod
  10 threads at 16 and 1 rows per pass, portable at 5, dotprod 1 thread at
  7) and two Metal configurations (16 and 1 rows per pass).
- `./dump_logits ~/models/qwen38-speed tools/quality/windows_own_thinking.json 2 reference-2.json`
  (pure scalar reference, alone): 273.39 s, 10.23 GB resident. Then
  `./verify_model ... --reference reference-2.json`: the first 2 logit rows
  equal the scalar reference.

### M7: whole-model pass on the real model, quiet machine (2026-11-14)

Command, for N in 1, 2, 4, 8 (binary of main 0b7bd5d, nothing else running):
`./bench_model ~/models/qwen38-speed --backend metal --kv int8-profile --rows-per-pass N --passes 9`
(`cargo run --release -p yskn-engine --example bench_model`). Minimum
seconds per pass over 9 passes at short context:

| Rows | Total | Kernel | Recurrence | Attention | Row encoding | Rest |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.264 | 0.146 | 0.040 | 0.002 | 0.045 | 0.029 |
| 2 | 0.298 | 0.159 | 0.050 | 0.002 | 0.046 | 0.036 |
| 4 | 0.390 | 0.224 | 0.067 | 0.005 | 0.048 | 0.045 |
| 8 | 0.671 | 0.385 | 0.119 | 0.012 | 0.080 | 0.073 |

Reading: about 3.8 tokens per second without speculation and before any
optimization of the CPU-side stages. The kernel at one row is 146 ms,
of which about 111 ms is the GPU pass measured alone (M2) and the rest
is 257 dispatch round trips. Each added row costs about 40 to 60 ms.
Output saved in `.work/runs/bench_model_int8profile_0b7bd5d.txt`.

### M8: perplexity, profile-requantized int8 cache (gate 1) (2026-11-14)

Command (run alone on the model, binaries of 18dd49a whose content is
main 0f91f2e):
`./score_windows ~/models/qwen38-speed tools/quality/windows_own_thinking.json --kv int8-profile --backend metal --rows-per-pass 16 --json score_int8-profile_metal.json`

Result: scored_positions 20460, mean negative log-likelihood
0.730248869401, **perplexity 2.075597096008**, 6,308 s. Upstream's record
`tools/quality/records/score_fastest_int8_table_reference.json`:
2.075899908604469. Difference 0.000303 (tolerance 0.001). The result file
records kv_mode, backend, device, kernel, threads and rows per pass.
A 16-row pass at context 190 took 1.44 s (kernel 0.72, attention 0.22,
row encoding 0.15, recurrence 0.23, rest 0.11) with other work running.

### M9: perplexity, KV5 cache (gate 2) (2026-11-14)

Command (run alone on the model; binaries of afb760e, the first commit of
PR #21, bit-identical to the merged version by frozen digests):
`./score_windows ~/models/qwen38-speed tools/quality/windows_own_thinking.json --kv kv5 --backend metal --rows-per-pass 16 --json score_kv5_metal.json`

Result: scored_positions 20460, mean negative log-likelihood
0.730292381185, **perplexity 2.075687410906**, 4,222 s. Upstream's record
`tools/quality/records/score_fastest_kv5_table_reference.json`:
2.0759369466709248. Difference 0.000250 (tolerance 0.001). KV5 minus
profile-int8: +0.000090 here, +0.000037 upstream. Before it,
`./verify_model ... --kv kv5`: 24 rows bit-identical across six
configurations.

### M10: upstream's self-check outputs and commit verification on real weights (2026-11-14)

Binaries of main 0b7bd5d; prompts tokenized with upstream's own formatter
(prompt hashes equal `selfcheck.json`); log `.work/runs/after_gate.log`.

`./generate ~/models/qwen38-speed --tokens PROMPT_IDS --max-new N --eos 248046,248044 --greedy --backend metal --kv kv5 --draft none`
and the same with `--draft ngram --k 3`:

| Case | New tokens | SHA-256 of ids equals upstream | With n-gram drafting |
|---|---:|---|---|
| short-answer | 3 | yes | yes; no guesses offered |
| plain-text | 32 | yes | yes; 26 passes, 6 of 12 guesses accepted |
| thinking | 160 | yes | yes; 138 passes, 22 of 66 guesses accepted |

`./verify_speculation ~/models/qwen38-speed --tokens 1,2,3 --accept 2 --backend metal --kv kv5`:
PASS (engine state digest equals the scalar reference after a 3-row
provisional pass and a 2-row commit), 960 s.

Passes took 0.7 to 1.5 s in these runs because they predate PR #21.

### M11: draft acceptance on the real model (2026-11-14)

Command (run alone on the model; binaries of the PR #24 branch, saved in
`.work/runs/05m`; for W in `0..1`, `7..8`, `13..14`):
`./draft_acceptance ~/models/qwen38-speed tools/quality/windows_own_thinking.json --windows W --positions 150 --k 7 --kv kv5 --backend metal`
(`cargo run --release -p yskn-engine --example draft_acceptance`).
"Verified acceptance" is the fraction of positions at which the first d
guesses all equal the main model's greedy choice when the guesses are
checked the way generation checks them. 150 positions per window
starting at token 1,024. Outputs in `.work/runs/acceptance_w*.out`.

| Window | Neural draft, depth 1 to 7 | Mean accepted prefix | N-gram mean prefix |
|---|---|---:|---:|
| 0 | .913 .827 .780 .767 .727 .647 .560 | 5.22 | about 0.9 |
| 7 | .693 .453 .253 .147 .087 .053 .033 | 1.72 | 0.04 |
| 13 | .920 .833 .680 .580 .527 .460 .413 | 4.41 | 2.15 |

N-gram first, neural second was worse than neural alone on all three.
Projection from the pass times of M7 and 47 ms per proposal with
window 0's acceptance: 264 ms per token without drafting, 124 at depth
3, 115 at depth 5, 114 at depth 6 (about 8.7 tokens per second). On
window 7 deep proposals lose. Hence decision O10.

### M12: fast draft on the real draft files (2026-11-14)

Binaries of the PR #24 branch, machine at low load.
`./verify_draft ~/models/qwen38-speed` (example `verify_draft`): the fast
draft's state equals the scalar reference after 3 context rows (the
example's default for the independent scalar comparison), its proposal
equals the scalar draft's at position 300 with the fast draft's snapshot
restored into the scalar draft, and the Metal, CPU and portable
configurations equal each other: PASS.
`./bench_draft ~/models/qwen38-speed` (example `bench_draft`): one
proposal 47.2 ms median (kernel 26.8, attention 5.7, approximate scores
4.1); pushing context rows 0.39 ms per row in batches of 256, 1.5 ms for
a single row; 2.2 GB resident. The scalar draft takes about 90 s per
proposal.

### M13: whole-model pass with the CPU stage fast paths (2026-11-14)

Command, quiet machine, binaries of the PR #25 branch (saved in
`.work/runs/07a`), minimum over the passes:
`./bench_model ~/models/qwen38-speed --backend metal --kv kv5 --rows-per-pass 1 --passes 9`

| Stage, one row | Before (M7) | With PR #25 |
|---|---:|---:|
| total | 264 ms | 232 ms |
| kernel (GPU) | 146 | 136 |
| recurrence | 40 | 39 |
| row encoding | 45 | 13 |
| rest | 29 | 26 |

At 16 rows and a 2,000-token context the attention stage takes 0.31 s
per pass (about 1.0 s before). A run under load (other workers
compiling; `.work/runs/bench_model_kv5_07a.txt`) gave 0.438 s at 4 rows
and 0.645 s at 8 rows.

So the fast paths, tuned on a small synthetic model, helped row encoding
and attention and did not help recurrence or the rest. A sampling
profile of the one-row pass (`sample PID 10` on the running
`bench_model`, on-CPU samples at the top of the stack, idle waits
removed) shows why: `kv_codec::select` 532 samples,
`GatedDeltaNet::dense_gates` 290, `portable_math::exp` 77, `recur_head`
61, `attend_head` 60, `portable_math::half` 57, `hadamard512` 55,
`require_finite` 49, `gated_deltanet::finite` 33. The time in the
recurrent layers is in the dense gates computed by scalar reference
code, not in the per-head update that was vectorized. Brief `07c`
targets this list.

### M14: the product binary on the real model (2026-11-14)

Binary `yskn` built from PR #26 at 0df90ae (saved as
`.work/runs/06i/yskn`), default settings (`Kv5`, Metal, n-gram drafting
at depth 3, context 32,768):

- `./yskn selfcheck --model-dir ~/models/qwen38-speed`: short-answer PASS
  (3 tokens), thinking PASS (160 tokens), plain-text PASS (32 tokens);
  82.7 s wall including loading.
- `./yskn info --model-dir ~/models/qwen38-speed`: memory needed
  17,205,034,189 bytes of 25,769,803,776 (weights 10.31 GB, cache
  2.94 GB at 32,768 tokens, recurrent state 0.16 GB, working 0.46 GB,
  margin 3.18 GB).
- `printf 'What is the capital of Japan? Answer in one short sentence.\n' | ./yskn chat --model-dir ~/models/qwen38-speed --no-think --max-tokens 40 --temperature 0`:
  "The capital of Japan is Tokyo.", 8 tokens, 34 s wall including
  loading.

`yskn bench` ran later (M15). `yskn serve` on the real model is first
run by the soak harness.

### M15: draft policy on the real model (2026-11-14)

Binary `yskn` of PR #29 at 13cb4ba (`.work/runs/06k/yskn-13cb4ba`),
defaults `--draft both --k 7 --depth adaptive`, Metal, `Kv5`:

- `./yskn selfcheck --model-dir ~/models/qwen38-speed`: all three cases
  PASS, 43 s wall.
- `./yskn info --model-dir ~/models/qwen38-speed`: `needed_bytes`
  20,435,462,454 of 25,769,803,776; `draft_bytes` 2,721,905,176.
- `./yskn bench --model-dir ~/models/qwen38-speed --n 96 --json bench_06k_13cb4ba.json`:

| Task | Passes for 96 tokens | Guesses accepted / offered |
|---|---:|---:|
| prose | 36 | 60 / 103 |
| code | 19 | 77 / 87 |
| structured | 24 | 72 / 109 |
| thinking | 21 | 75 / 116 |

Without drafting each task needs 96 passes. The same counts were
obtained by the author's run, by this run, and by the builds of main plus
PR #30 (3970934) and PR #31 (acacf28 pending): the depth choices repeat
and the new CPU and attention paths do not change a token. Tokens per
second in these runs were 6.1 / 8.9 / 2.9 / 3.4 at load 9 to 14 and
6.4 / 10.4 / 5.2 / 6.4 at load 10 to 12: not usable as timings.

Projection from the fixed cost table of D21 (232 ms at one row to 650 ms
at eight, 47 ms per proposal), mean of the four tasks at 256 tokens,
from recorded traces replayed through the controller (`draft_trace
record` and `replay`): no draft 232.0 ms per token; n-gram K=3 205.6;
neural fixed K=3 138.2; neural fixed K=7 164.9; neural adaptive 132.8;
both adaptive 123.3.

### M16: CPU stages round two and grouped attention (2026-11-14)

CPU backend, real model, one row per pass, minimum of 5 after warm-up,
shared machine (PR #30, D20):
`bench_model ~/models/qwen38-speed --backend cpu --kv kv5 --rows-per-pass 1 --passes 5`

| Stage | Before | After |
|---|---:|---:|
| row encoding | 11.8 ms | 3.6 ms |
| recurrence | 40.4 ms | 13.8 ms |
| rest | 29.1 ms | 27.4 ms |

Attention stage alone with the real model's shapes (16 attention layers,
24 query heads, 4 key/value heads, real header and profile, synthetic
rows), ten workers, load 5 to 12; minimum of 5 samples except the first column's 32,768-token figure, which is the minimum of 3:
`YSKN_MODEL_DIR=~/models/qwen38-speed YSKN_BENCH_CONTEXT=C YSKN_BENCH_ROWS=M YSKN_BENCH_PASSES=5 cargo test --release -p yskn-engine --lib bench_real_attention_shapes -- --ignored --nocapture`

| Context, rows | Before PR #30 | Round two | Grouped |
|---|---:|---:|---:|
| 2,000 tokens, 16 rows | 0.307 s | 0.228 s | 0.086 s |
| 32,768 tokens, 1 row | 0.555 s | 0.452 s | 0.192 s |
| 32,768 tokens, 4 rows | | 2.22 s | 0.41 s |

The targets (0.06 s and 0.10 s) are not met. At 32,768 tokens the
grouped path's logical cache operands (2.06 GB per row) divided by its
time give about 10.8 GB/s. That is a computed figure, not a measured
memory traffic, so it does not by itself show what bounds the stage;
D20 names cache traversal and widening at long context and arithmetic
at short context. Candidate for another round, starting with a
measurement of what bounds it.

Correctness on the real model with the binaries of PR #30 (a47359a):
`yskn selfcheck` PASS; `verify_model ~/models/qwen38-speed
tools/quality/windows_own_thinking.json --kv kv5`: all 24 logit rows
bit-identical across six CPU and Metal configurations.

### M17: kernel round two and memory bandwidth (2026-11-14)

All on a loaded machine (load 5 to 24); PR #31, D22, `METAL.md`.

- `bench_bandwidth --gb 4 --reps 5`: CPU and GPU reading together about
  145 to 150 GB/s (one run 161); GPU alone 116 to 119; CPU alone 74 with
  one thread, 110 with four, 107 to 134 with ten. This is the measured
  ceiling; the nominal 153 GB/s was not reached by one reader.
- `bench_metal ~/models/qwen38-speed/qwen38_27b_speed.yskn`, kernel
  alone, integer to binary32: 1 row 100.7 to 98.9 ms; 4 rows 181.3 to
  153.8 ms; 16 rows 638.2 to 492.0 ms. The kernel at one row reads
  9.76 GB in 98.9 ms: 98.7 GB/s, about 85% of the GPU's own streaming
  rate and about two thirds of the combined ceiling.
- CPU share (`bench_metal --shares`, 257 calls per pass): 1 row 144.5
  to 133.6 ms; 4 rows 194 to 178.6 ms; 16 rows 554 to 475.6 ms. Under
  heavy CPU load the share makes a pass slower (197 ms against 145 ms at
  load 22 to 24), so the committed table is below the measured optimum.
- Inside the engine at one row the kernel costs about 36 ms more than
  in one command buffer: 85 µs from commit to GPU start and 61 µs from
  GPU end to wake-up, 257 times. Blocking wait, polling, shared events
  and pre-committed buffers all gave about 0.11 ms per round trip.

What limits one row now: the fixed round-trip latency (about 36 ms),
GPU streaming at about 85% of its own rate, and the CPU stages (about
45 ms after PR #30). What limits wide passes: binary32 multiply-add
throughput, about 0.98 T operations per second.

## Blocked on the owner

Nothing blocks the work. One event to know about: on 2026-11-14 at about
11:55 JST the account's 5-hour usage limit stopped both Opus subagents
mid-task; it reset at 12:10 and both were resumed from their transcripts.
If Opus work stalls again for this reason, Codex-routed work (Sol, Astra,
Luna) continues, and anything that needs the GPU or sockets waits.

A second note on capacity: with up to five workers at once the machine
swapped (9.25 GB in use at 18:04) and Opus sessions twice returned early
waiting on their own background jobs; both are costs of running the
optimization, review and tooling lines in parallel, accepted to shorten
the calendar time. The quiet measurement session and the release runs
(lossless about 2.5 h, soak at least 1 h, two perplexity gates about
1.2 to 1.8 h each) need the machine alone and are run one after another.
