# Project state

This file is the single resumable record of the port. It is written so the
orchestrator can continue from it alone. Update it at every merge, decision
and measurement.

Last updated: 2026-11-14 14:20 JST (step 3 done: both quality gates pass and all three upstream self-check outputs are reproduced; step 5 mostly merged; step 6 server merged, real engine being connected; draft fast path in progress).

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
   **Main model validated; the two published configurations are running.**
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
   **In progress**: see the unit table under "Step 5 plan".
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
   top-level errors); a second Opus fix round (brief `06g`) is in
   progress. Still to do after it merges: connect the real engine to the
   crate's `Engine` trait, prompt reuse, `bench`, `selfcheck`.
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
7. Optimization to the machine limit (stop after five attempts under 2%).
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

- The reference needs three cache modes (to be decision D13): `Int8`
  (plain, D4), `Int8Profile` (upstream's "int8" record), `Kv5` (upstream's
  default, packed 5-bit where packable).
- The two published values differ by 3.7e-5 and the tolerance is 0.001, so
  perplexity cannot detect a subtly wrong codec. The codec is proven by its
  own tests (optimality against an independent dynamic program, round
  trips, frozen digests, facts of the real profile). The encoder uses only
  integers and correctly rounded IEEE operations, so it is ported from the
  source operation for operation.
- The plain-int8 run in progress still tells whether the rest of the main
  model is right: its result should be within about 0.001 of 2.0759. A
  larger gap is a bug or the codec's effect; `Int8Profile` separates the two.

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
| D | Cache modes `Int8`, `Int8Profile`, `Kv5` in the reference attention, model load and the engine | merged, PR #14 (Sol; Astra reviewed the reference against the source, Opus the engine wiring); real scoring running |
| E | KV codec fast path | not started; the codec runs inside the engine's parallel row preparation, measure first |
| F | Main-model hooks: residual capture after layers 5, 19, 33, 47, 61; snapshot/restore; head rows | merged: PR #17 and PR #19 (D17; Sol; two Opus reviews) |
| G | Speculation loop over Verifier and Proposer traits, lossless tests | merged, PR #17, D16 (Sol; Opus review) |
| H | Engine `run_provisional`/`commit` without cloning the attention cache | merged, PR #17 |
| I | Draft model reference (chapter 07) and `.shead` | merged, PR #15, D15 (Sol; Astra review against the source) |
| J | Draft fast path (one scalar proposal takes about 90 s on the real files; target under 40 ms, bit-identical) | brief `05m` (Opus), worktree `.work/wt/05m`, branch `feat/05m-draft-fast`, in progress |
| K | Prompt reuse modes (reset, live prefix, snapshot) | in brief `06i` |
| L | Long-prompt prefill: bounded memory, head on last row only, large-M kernel | cancellable prefill with last-row logits in brief `06i`; a large-M kernel is step 7 |

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

Policy to be decided by measurement, not copied from the RTX 5090 (decision
D15 to come): number of drafted rows K (each verified row costs about 35 ms
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
  highest read rate observed so far is 87.7 GB/s (Metal kernel, M=1). The
  achievable ceiling must be measured before step 7 compares against it.
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

D1 to D15 are in [DECISIONS.md](DECISIONS.md): D10 mapped CPU kernel,
D11 layer-major engine model, D12 Metal backend, D13 KV cache modes and
codec, D14 engine on Metal with parallel recurrent heads, D15 draft model
math and tie rule. D16 (generation state and speculation loop) is being
written in brief `05e`. Owner-decided gaps:

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
  to step 7. To record in DECISIONS.md when the Metal crate merges.
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

- CI's test step takes about 21 minutes of a 30-minute limit. Reduce it to
  under about 8 minutes without dropping coverage (the randomized
  speculation and draft-generation tests dominate; use smaller synthetic
  models). Consult the `test-audit` skill.
- Scalar attention time grows linearly with context: about 1 s per 16-row
  pass at 2,000 tokens, so on the order of a second per token at 32,768
  (extrapolated, not measured). A fast attention path is required for the
  32,768-token release criterion; step 7.
- The format's maximum context (262,144 tokens) does not fit this machine
  in `Kv5` with the current cache representation; the product caps the
  context by memory (brief `06i`). A compact fast path reading packed
  words (164 bytes per row) would restore the saving; step 7.
- `crates/yskn-cli/README.md`: two documentation nits from Astra's fourth
  review (in brief `06i`).

- `generate` copies the whole recurrent state (about 157 MB) at the start
  of every provisional pass and again on a partial commit (D16); step 7.
- Row encoding (`kv_encode`, 45 ms at one row) runs the scalar codec for
  128 side-rows sequentially within a row; it can run in parallel across
  heads and sides without changing bits; step 7.

- Draft tests (from Astra's review of PR #15): a rank-projection test with
  off-diagonal entries and rank different from hidden width; a synthetic
  draft with query heads different from KV heads (real: 32 and 8, mapping
  head / 4); a rotary expectation at a non-zero position; an attention test
  with non-zero queries and keys; head scales other than 1;
  source-derived expectations for the correction chain.
- The KV codec's tie-path equality with the source is argued from the code,
  not proven by a test with source-derived expected values (none exist).
- The bandwidth ceiling for this chip is not measured yet (see Environment).
- D12 in DECISIONS.md and M2 below were measured before two review fixes to
  the Metal shader; re-measure on a quiet machine during step 7.

- `MappedModel::open` is an `unsafe fn` (file must not change while mapped).
  The product binary needs a safe wrapper that documents this.


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

## Blocked on the owner

Nothing blocks the work. One event to know about: on 2026-11-14 at about
11:55 JST the account's 5-hour usage limit stopped both Opus subagents
mid-task; it reset at 12:10 and both were resumed from their transcripts.
If Opus work stalls again for this reason, Codex-routed work (Sol, Astra,
Luna) continues, and anything that needs the GPU or sockets waits.
