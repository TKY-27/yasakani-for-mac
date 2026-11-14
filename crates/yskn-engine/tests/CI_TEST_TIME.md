# CI test cost audit (08a)

**Status:** cost reduced; local ~90 s test-time aim is NOT met. Runner <8 min
is UNVERIFIED. Metal and sockets require the orchestrator. No production-source
change, no new dependency, no push.

Scope: dev profile, offline; keep independent scalar/frozen oracles, assertions,
all cache modes and every available backend. No test deletion is a target.

## Proposed ledger

Each row records observable contract; credible fault; oracle; cost action.
Measured per-test times and final results are appended after verification.

| Binary / test (unique suffix) | Contract | Regression | Oracle | Action |
| --- | --- | --- | --- | --- |
| speculation / randomized_partial_commits_and_discards… | exact logits, committed cache/recurrence and future row after accept/discard; atomic invalid requests | replay/retain a rejected row | scalar Model logits and digest | shorter deterministic width/acceptance sweep plus four random passes; assert every accepted length; split KV modes, keep scalar run for each backend |
| speculation / failed_numeric_pass… | failed finite-overflow pass is atomic, including completed prefill chunks | omit attention rollback | saved state and scalar continuation | retain fixture and both failure boundaries |
| speculation / snapshots_restore_prefix… | snapshot ownership, prefix validity and allocation reuse | accept foreign/reset/future snapshot or grow persistent memory | scalar fresh suffix and saved memory | compute fresh scalar suffix once per KV mode, outside repeated reuse/backend runs |
| speculation / all_last_and_none… | head omission does not alter future numeric state | skip a layer for None | All-head execution and retained future digest | retain all mode/backend/head combinations |
| speculation / synthetic_model_generation… | proposer/K/sampler losslessness, repeatability, statistics and all stop boundaries | wrong accepted prefix or RNG position | scalar no-draft tokens/digest plus scalar drafted statistics | split by sampler; scalar expectation once per configuration and backend-independent; keep both engine repeats and full K/proposer/stop matrix |
| speculation / snapshot_after_partial_commit… | restored snapshot contains only accepted rows | snapshot rejected suffix | scalar direct prefix | retain direct scalar expectation on every backend |
| speculation / residual_capture… | capture slot order, retention, provisional/committed separation, reset/restore across pass widths | capture rejected row or swap selected slots | scalar captured residual bits, position, digest, embedding | retain all scalar/engine width/backend runs; trace caching was unnecessary |
| draft_generation / cpu_neural_generation… and metal_neural_generation… | neural draft is lossless through joint restore, wrapped continuation and prefix reuse for all modes/seeds/K | restore main without ring, or include rejected captures | scalar no-draft tokens and digest | shorten requests to 2/8/2, explicitly assert wrap; split modes and seeds, keep full product |
| draft_generation / cpu_fast_draft_generation… and metal_fast_draft_generation… | fast draft matches reference proposal acceptance, tokens and ring, both match no draft | wrong fast draft score/feature slot | scalar Draft, independently no-draft main | retain 9-token sequence, every K/seed/backend |
| draft_generation / engine_handover… | capture accounting/release, long suffix reseed, main-only restore/reset detection | retain delivered capture buffers or adopt stale identity | scalar captures, fixed byte counts, error contract | retain |
| draft_wiring / reference_neural_generation… | scalar neural generation lossless through restore and prefix reuse, actual ring wrap | stale draft ring after restore | independent no-draft scalar session | shorten requests to 2/8/2 with wrap assertion; split sampler cases |
| draft_wiring / synthetic_teacher… | correct capture wiring accepts 7 guesses; lagged/swapped slots accept zero | swap capture slots or lag a row | main head fixed to token 0, trained artifact negative controls | retain teacher, seven-depth block and independent negatives |
| draft_wiring / reused_and_restored_sessions… | retained long prompt suffix reseeds ring in reused/restored sessions | omit suffix row or use stale ring | direct Draft ring from captured rows | retain both paths and window-crossing prompt |
| draft_wiring / main_only_restore_or_reset… | stale ring rejected even after main advances beyond it | ignore context identity | exact error outcome after public operations | retain both paths |
| draft_wiring / delivered_captures… | delivery releases captures and joint snapshots retain zero | retain delivered captures | fixed count/zero bytes and empty vectors | retain |
| draft_wiring / depth_dependent_teacher… | frontier differs from history; each block depth uses successor | use history tail as frontier or repeat one depth | independently specified 6..12 teacher | retain |
| draft_wiring / sampled_hook… | sampled absolute position and distinct seed behaviour | sample at relative position | direct Draft corrected candidates with independently specified absolute positions | retain |
| model / recurrent_model_is_identical_for_all_kernels_threads_pass_widths_and_chunks | two recurrent layers preserve rows, chunks, reset and future state | publish recurrent state in the wrong layer order | 9-row frozen SHA-256 and scalar Model | cover every kernel/thread pair at width 17 and every width at fastest/x4; retain full KV/backend product and all fixture tokens |
| model / mixed_model_crosses_attention_tiles_from_position_three_without_bit_changes | four mixed layers cross absolute attention tiles from position 3 | use relative tile position or wrong chunk offset | 73-row frozen SHA-256 and scalar Model | cover every kernel/thread pair at width 17 and every width at fastest/x4; retain full KV/backend product and all fixture tokens |
| model / int8_profile_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation | profile cache preserves pass/chunk/reset/future parity and differs from Int8 | ignore profile mode or wrong attention ordinal | unchanged profile frozen SHA-256 plus scalar/Int8 controls | cover every kernel/thread pair at width 17 and every width at fastest/x4; retain full KV/backend product and all fixture tokens |
| model / kv5_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation | packed and fallback profile heads preserve all bits through chunks/reset | pack an unfit head or decode wrong KV basis | unchanged KV5 frozen SHA-256 plus scalar/Int8 controls | cover every kernel/thread pair at width 17 and every width at fastest/x4; retain full KV/backend product and all fixture tokens |
| model / other_hidden_ffn_widths_and_padded_vocabulary_are_identical | 1024-wide hidden/FFN and vocab 65 execute all logical rows | drop second rotation block or emit padded vocabulary rows | unchanged wide-model frozen SHA-256 and scalar Model | cover every kernel/thread pair at width 17 and every width at fastest/x4; retain full KV/backend product and all fixture tokens |
| model / dense_embeddings_are_identical | BF16 embedding vectors use same model path across kernels/chunks | read dense embeddings with wrong stride or dtype | unchanged dense-model frozen SHA-256 and scalar Model | cover every kernel/thread pair at width 17 and every width at fastest/x4; retain full KV/backend product and all fixture tokens |
| model / invalid_requests… | atomic invalid token/capacity; exact capacity boundary | commit valid prefix of invalid request | scalar failure and full continuation | retain |
| model / failure_after_a_completed_pass… | numeric failure rolls back all prior chunks | leave completed chunks committed | scalar continuation | retain |
| model / persistent_range_pool… | ordered borrowed results and first-index errors, recovery after panic | merge in arrival order or fail to drain jobs | independently mapped integers and error strings | retain |
| model / load_selects_env… | env mode parsing/default and explicit constructor defaults | let env override explicit default | declared modes and isolated child outcomes | retain |
| kv_attention / all_modes_freeze… | profile/KV5 differ, freeze bits, row/chunk/reset/partial-commit parity | ignore codec mode or retain rejected cache | unchanged frozen hashes and independently grouped scalar sessions | split by mode, retain 73 rows, two full-attention ordinals, mixed packability; distinct unchanged frozen hashes imply all pairwise inequalities |
| kv_attention / cached_bases… | optimized decode preserves historical logits and stored state | change cache basis order | unchanged pre-05j frozen hash/digest | retain 79 rows |
| kv_attention / codec_error… | later codec error is atomic; guards use absolute positions | guard against local row number | control session plus fixed failure position | retain |
| kv_attention / remaining tests | manifest/default profile/path containment and invalid dimensions | load absent/escaped/wrong profile or ignore mode | fixed errors and independently selected outputs | retain |
| draft_fast / rings_proposals… | all shapes/APIs/configs preserve ring, hidden bits, top16 IDs/scores and choices | wrong wrap slot, grouped head or padded vocab | scalar Draft once per shape | split by three shapes, preserve all dimensions/rows and configuration matrix |
| draft_fast / failed_rows_snapshot… | failed row commits earlier rows only; restore/reset/error parity | roll back too much or accept invalid BF16 | scalar Draft errors, rings and proposals | compute scalar oracle once per bad-input case, share configurations |
| draft_fast / processing_failure… | actual processing overflow commits exactly prior rows in all APIs | publish failing row or lose prior rows | scalar Draft failure, ring, proposal | scalar once per API, share configurations |
| draft_fast / loading_rejects… | malformed options and head dimensions rejected | ignore capture width | scalar rejection and fixed invalid shape | retain |
| model_invariants / mixed_model_row_and_chunk_invariants | mixed recurrent/attention row parity and exact partial-prefix future | commit rejected recurrent/cache suffix | independent single-row session and fresh accepted prefix | retain all 5 rows and 3-row commit |
| model_invariants / mixed_model_unaligned_long_pass_matches_single_rows_and_uneven_chunks | whole, single and uneven chunk passes from unaligned prefix preserve logits/future bits across tiles 32/64 | use call-relative cache index or wrong mask | three separately advanced scalar sessions, plus nonconstant logits | retain all 73 rows and chunk sizes 1/31/2/33/6 |
| model_invariants / remaining subsecond tests | scalar row/chunk/prefix/reset/dimension/BF16/atomicity contracts | changed row grouping or accepted suffix | differently grouped sessions; fixed dimensions/errors | retain |
| cpu_stages / public_model_runs… | auto/reference stages equal scalar for all KV modes across tile and pass boundaries | vector stage changes rounded bits or second attention ordinal | scalar logits/digest | split the six selector/mode combinations into isolated child tests; retain all 36 tokens and both selectors |
| cpu_stages / independent_cache_sides… | independent K/V preparations equal scalar output/cache/error bits | mix cache side/head index | scalar prepare_row and cache digest | retain 34 positions, two ordinals and all modes |

Frozen attention tests (including 31-split and 42-split schedules, contexts ending
inside/exactly on a 32-slot tile) are retained unchanged. Hidden/FFN widths 512
are the H512 minimum; 1024-width tests remain to prove multiple rotation blocks.

## Work and independence findings

- All main-model fixtures keep hidden/FFN widths 512 (H512 minimum), or their
  original 1024 widths; all draft shapes and payloads are unchanged. The neural
  lifecycle requests change from 5/8/5 to 2/8/2 tokens, counting the continuation
  twice after joint restore. The 8-token continuation still offers all seven
  guesses at K=7, crosses window 8 (position 5 to 13), and is compared in full
  before and after restore. The prefix still diverges at token 4. Both restored
  ring equality and final draft/main position equality remain asserted.
- Partial-commit trials change from 40 to 23: widths 1..9 fully committed,
  then width 9 accepted at every length 0..9, then four randomized trials.
  Per mode/backend this is 230 -> 162 provisional token rows and 40 -> 23
  future checks. Final committed position is 104 -> 110, still beyond both
  absolute tile boundaries 32 and 64. The old random trace never accepted length
  8; the new test asserts all widths and acceptance lengths were reached.
- Scripted generation keeps all 4 K values, 6 proposers, 3 samplers, 2 engine
  repeats, every backend and every stop boundary. Scalar calls decrease from
  156 to 72 per CPU/Metal-independent oracle set; the 216 engine generation
  runs per backend are unchanged. Scalar rows/statistics/digests are stored as
  owned expectations, never obtained from the engine under test. The four K
  caps with NoProposer reuse the no-draft scalar baseline: it always returns an
  empty guess vector, so `generate` verifies one row regardless of the cap.
- In `model`, pass grouping lives in `Model::prefill_with_logits` above
  `Compute::weighted_sums` and `map_range`. Pool task size depends on matrix
  output rows N and threads, not pass grouping; each kernel/thread pair remains
  tested at width 17 for every fixture, and each width at fastest/x4. The
  unchanged `linear::every_kernel_matches_scalar_for_layout_bits_batches_padding_and_signed_scales`
  keeps the full kernel x threads x M (1/2/3/8/16/17/40) x layout x bits
  product, including 1024-column matrices. No cache-mode/backend combination is
  removed; all capture/width and neural mode/seed/K/backend products remain.
- Temporary instrumentation of the **original** speculation generation for one
  sampler (greedy), all K/proposers/stops: 52 scalar calls = 47.799 s, 72 engine
  calls = 0.730 s (98.5% scalar inside generation). For the original neural
  lifecycle, one Int8/greedy backend case: 3 scalar no-draft calls = 2.189 s,
  3 engine no-draft calls = 0.056 s, 12 engine + scalar-Draft calls = 10.945 s.
  Scalar main inference dominates the first; scalar draft proposals dominate
  the second despite its fast main model. `draft_wiring`, `kv_attention` and
  `model_invariants` execute scalar models exclusively. Instrumentation was
  removed; these figures exclude setup, prefix prefill and assertions.

## Integration-test grouping and removed duplicate tests

Three scalar suites (`draft_wiring`, `kv_attention`, `model_invariants`) now
compile into ref `tests/models.rs`. Cases retain namespace-qualified names.
This changes only integration-test layout, and lets libtest fill idle cores
across suites. An engine aggregation experiment took 74.99 s versus 64.93 s
for its four separate optimized binaries, so the engine suites remain separate.
No example or production crate is restructured.

No behavioural test is deleted. The three identical support-writer tests used
to be compiled/run once in each constituent suite; each aggregate now imports
one support module. Six copies are removed from the three-suite reference
aggregate. Its retained tests are:

- `support::synthetic_writer_builds_reproducible_mixed_model`
- `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings`
- `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths`

Each retained body, seed, assertion and owning Container boundary is unchanged.
Splitting generation, ring, KV and stage tests replaces their original loop test
with named cases; it does not remove a case.

## Compile-time choice

CI sets `CARGO_PROFILE_DEV_DEBUG=0` and `CARGO_PROFILE_TEST_DEBUG=0` for the job,
so Clippy and tests agree on the profile override. This removes DWARF only;
opt-level 2, debug assertions and overflow checks remain. Cargo.toml and all
public crates are unchanged. All formatting, warning-denying all-target Clippy,
and workspace test commands remain. Examples are deliberately kept in Clippy:
removing `--all-targets` would drop a check. Aggregation also avoids two test
crate links and six duplicate writer-test instances. No dependency added.

## Compact mixed fixtures

The session-only speculation cases and engine neural generation now use one
recurrent + one full-attention layer, rather than one recurrent + two full.
A one-ordinal profile still has two heads: head 0 packs, head 1 has a basis and
falls back; guard 2, means and both K/V sides remain exercised. Its writer parses
the profile and asserts the two packability outcomes. Modes and backends retain
their full product. All commit lengths, invalid/overflow rollback cases, repeated
snapshot reuse, All/Last/None and partial-snapshot futures remain unchanged.
Capture tests selecting slots 0 and 2 explicitly keep all three layers. The
frozen `model` and `kv_attention` tests keep two full-attention ordinals, including
their original mixed profile and inputs, so ordinal indexing remains independently
pinned at the public model boundary. Neural capture slots 0 and 1, 8-row draft
blocks, ring wrap, joint restore and prefix divergence still execute in full.

## Per-test baseline measurements

Seconds from libtest JSON timing, default test-thread count, dev opt-level 2.
Tests within each binary overlap; per-test times must not be added to its wall
time. Every listed support test was measured as well. No other Cargo/test load
was intentionally started during these timing runs; other workers/host load
were not under this task's control.

### yskn-engine/cpu_stages (12.924 s wall)

| Test | Seconds |
| --- | ---: |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.003 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.006 |
| `cpu_stage_override_rejects_unknown_values_before_opening_weights` | 0.009 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.014 |
| `independent_cache_sides_match_scalar_rows_and_errors` | 0.125 |
| `public_model_runs_vector_and_forced_reference_stages` | 12.678 |

### yskn-engine/draft_fast (18.427 s wall)

| Test | Seconds |
| --- | ---: |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.007 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.008 |
| `loading_rejects_the_reference_errors` | 0.016 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.022 |
| `processing_failure_keeps_prior_rows_and_matches_reference_on_every_backend` | 3.141 |
| `failed_rows_snapshot_restore_and_reset_match_reference` | 9.237 |
| `rings_proposals_and_top16_match_reference_on_every_backend_and_thread_count` | 18.175 |

### yskn-engine/draft_generation (104.880 s wall)

| Test | Seconds |
| --- | ---: |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.005 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.005 |
| `metal_neural_generation_is_lossless_across_joint_restore_and_reused_prefix` | 0.012 |
| `metal_fast_draft_generation_equals_reference_draft_and_no_draft` | 0.012 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.012 |
| `engine_handover_counts_memory_reseeds_long_suffix_and_detects_main_only_restore` | 1.689 |
| `cpu_fast_draft_generation_equals_reference_draft_and_no_draft` | 10.090 |
| `cpu_neural_generation_is_lossless_across_joint_restore_and_reused_prefix` | 104.622 |

### yskn-engine/model (20.813 s wall)

| Test | Seconds |
| --- | ---: |
| `persistent_range_pool_preserves_order_borrows_and_recovers_after_errors` | 0.029 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.017 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.004 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.006 |
| `load_selects_env_kv_and_explicit_constructors_keep_int8_default` | 0.086 |
| `failure_after_a_completed_pass_preserves_logits_position_and_state` | 1.206 |
| `dense_embeddings_are_identical` | 2.722 |
| `recurrent_model_is_identical_for_all_kernels_threads_pass_widths_and_chunks` | 5.924 |
| `invalid_requests_preserve_logits_position_and_future_sequence_state` | 7.056 |
| `other_hidden_ffn_widths_and_padded_vocabulary_are_identical` | 7.326 |
| `int8_profile_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation` | 15.783 |
| `kv5_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation` | 15.892 |
| `mixed_model_crosses_attention_tiles_from_position_three_without_bit_changes` | 20.574 |

### yskn-engine/speculation (122.761 s wall)

| Test | Seconds |
| --- | ---: |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.003 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.004 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.013 |
| `all_last_and_none_head_requests_preserve_the_same_future_numeric_state` | 0.225 |
| `failed_numeric_pass_preserves_committed_state_and_future_rows` | 0.247 |
| `snapshot_after_partial_commit_restores_exact_prefix_and_future_state` | 1.402 |
| `snapshots_restore_prefix_without_growing_persistent_storage_and_reject_misuse` | 5.221 |
| `residual_capture_matches_reference_across_widths_kv_backends_and_session_operations` | 15.493 |
| `randomized_partial_commits_and_discards_match_reference_in_every_kv_backend` | 66.407 |
| `synthetic_model_generation_is_lossless_repeatable_for_every_proposer_and_sampler` | 122.751 |

### yskn-ref/draft_wiring (62.911 s wall)

| Test | Seconds |
| --- | ---: |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.005 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.007 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.010 |
| `delivered_captures_are_released_and_joint_snapshots_do_not_retain_them` | 1.095 |
| `depth_dependent_teacher_uses_block_successors_and_the_distinct_frontier` | 1.206 |
| `sampled_hook_matches_direct_draft_with_frontier_seed_and_absolute_positions` | 1.316 |
| `main_only_restore_or_reset_is_detected_even_after_advancing_past_the_draft` | 1.329 |
| `synthetic_teacher_accepts_correct_captures_and_rejects_lagged_or_swapped_slots` | 2.506 |
| `reused_and_restored_sessions_reseed_after_a_long_new_prompt` | 3.132 |
| `reference_neural_generation_matches_k0_across_joint_restore_and_reused_prefix` | 62.681 |

### yskn-ref/kv_attention (42.538 s wall)

| Test | Seconds |
| --- | ---: |
| `manifest_profile_and_weights_paths_cannot_escape_folder` | 0.006 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.006 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.007 |
| `profile_modes_reject_absent_profiles_and_wrong_attention_counts_or_heads` | 0.014 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.029 |
| `folder_modes_select_profile_manifest_default_and_preserve_plain_int8_default` | 0.455 |
| `codec_error_in_later_model_row_is_atomic_and_guard_uses_absolute_position` | 0.471 |
| `cached_bases_preserve_pre_05j_mixed_model_logits_and_stored_state` | 9.353 |
| `all_modes_freeze_logits_and_profile_modes_preserve_long_pass_reset_and_continuation` | 42.285 |

### yskn-ref/model_invariants (11.502 s wall)

| Test | Seconds |
| --- | ---: |
| `model_vectors_reject_nonfinite_values_before_execution` | 0.003 |
| `mixed_model_rejects_wrong_head_width_and_unknown_architecture` | 0.010 |
| `mixed_model_loads_with_qwen3_5_rotary_width` | 0.011 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.011 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.004 |
| `model_folder_manifest_rounds_context_capacity_and_selects_weights` | 0.029 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.006 |
| `plain_bfloat_embeddings_use_the_same_model_path` | 0.099 |
| `invalid_token_pass_is_atomic_and_reset_reproduces_the_initial_state` | 0.114 |
| `header_dimensions_and_padded_vocabulary_are_used_in_execution` | 0.153 |
| `synthetic_logits_sample_repeatably_by_seed_and_absolute_position` | 0.156 |
| `multirow_pass_matches_single_rows_bit_for_bit` | 0.710 |
| `prefill_chunk_boundaries_do_not_change_logits_or_future_state` | 0.750 |
| `provisional_prefix_commit_matches_committing_pass` | 0.932 |
| `mixed_model_row_and_chunk_invariants` | 1.766 |
| `mixed_model_unaligned_long_pass_matches_single_rows_and_uneven_chunks` | 11.250 |

## Intermediate independent-binary comparison

After splitting/caching/covering, before the compact fixtures, final NoProposer
baseline reuse and scalar aggregation. All were dev opt-level 2 with DWARF.
The unchanged `cpu_stages` and `model_invariants` values illustrate host variance.

| Original binary | Before s | Intermediate s |
| --- | ---: | ---: |
| yskn-engine/cpu_stages | 12.924 | 15.565 |
| yskn-engine/draft_fast | 18.427 | 8.750 |
| yskn-engine/draft_generation | 104.880 | 16.524 |
| yskn-engine/model | 20.813 | 10.736 |
| yskn-engine/speculation | 122.761 | 28.917 |
| yskn-ref/draft_wiring | 62.911 | 22.975 |
| yskn-ref/kv_attention | 42.538 | 16.426 |
| yskn-ref/model_invariants | 11.502 | 14.786 |

## Final measurements and proof

Final dev-derived test profile: opt-level 2, assertions and overflow checks on,
debug information disabled by CI overrides (`DEBUG=0`). Default libtest parallelism. Individual
legacy binaries no longer exist for the three reference model suites; their
combined wall time is reported once, not charged to each constituent suite.

| Target | Before local dev s | Final local dev s | Change |
| --- | ---: | ---: | --- |
| yskn-engine/cpu_stages | 12.924 | 3.951 | Six isolated stage/mode cases |
| yskn-engine/draft_fast | 18.427 | 11.200 | Shape parallelism, shared scalar errors/proposals |
| yskn-engine/draft_generation | 104.880 | 16.934 | Two-layer mixed fixture, 2/8/2 requests, full configuration product |
| yskn-engine/model | 20.813 | 11.675 | Independent kernel/pool and pass-width covering set |
| yskn-engine/speculation | 122.761 | 32.164 | Compact mixed fixture, 23-pass sweep, scalar expectation reuse, sampler/mode cases |
| yskn-ref/models (formerly draft_wiring, kv_attention, model_invariants) | 116.951 | 28.652 | Three binaries combined; neural sampler/KV cases split, requests shortened; frozen/long invariants retained |
| Targeted binaries, total | 396.757 | 104.576 | |

All non-CLI test execution: approximately 119.12 s;
Cargo command wall 154.92 s, including 30.79 s of compilation from
feature unification after excluding CLI. This does not meet the ~90 s local aim.
The runner target cannot be asserted from local CPU-only runs.

Cold workspace `--no-run`: original 53.00 s, debug-info-off/aggregated layout
55.54 s (measured before the last compact-fixture edit). No compile-time win is
proven. The unchanged linear test executable shrank 1,836,136 -> 1,631,488 bytes;
DWARF removal is safe but should be judged by the runner, not an assumed speedup.
The pre-existing `score_windows` example filename collision warning remains;
examples still receive Clippy checking.

| Proof | Result |
| --- | --- |
| `CARGO_NET_OFFLINE=true cargo fmt --all --check` | PASS |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS, both normal dev and final CI debug overrides |
| `time cargo test --workspace` | CLI `serve_end_to_end` / `serve_port_in_use` fail at socket bind with PermissionDenied; Cargo stops before other packages |
| Continuation: dev workspace excluding `yskn`, JSON timing only | 252 passed, 0 failed (Metal absent cases are libtest passes with skip messages); command wall 154.92 s |
| `cargo test --workspace --release` | Same two CLI socket failures; default fail-fast stops |
| Continuation: release `--workspace --no-fail-fast`, JSON timing only | 316 passed, 45 socket PermissionDenied failures; all 252 non-CLI tests pass; command wall 109.60 s |

The 45 release socket failures are 2 CLI tests + all 43 server tests. Each
failure's output contains `Operation not permitted`. No failure is hidden by
changing CI, assertions, socket setup or expected outcomes. Real GPU execution,
real socket/server behaviour and the macos-15 three-core elapsed time remain
UNVERIFIED and are left to the orchestrator. Profiling/mutation instrumentation
is absent from the final files.

## Mutation checks

Each mutation was local, produced exit 101 with the named test FAILED, and was
restored byte-for-byte. The compact speculation/generation fixtures were mutated
again after shrinking. Final dev and release checks use restored source.

| Binary | Temporary production mutation | Failing test |
| --- | --- | --- |
| draft_wiring | Rotate captured feature rows left by one 512-channel slot before ring push | `draft_wiring::synthetic_teacher_accepts_correct_captures_and_rejects_lagged_or_swapped_slots` |
| model | Use (absolute output position - 1) for the attention prefix bound | `mixed_model_crosses_attention_tiles_from_position_three_without_bit_changes` |
| speculation | `commit`: position += k + 1 | `randomized_partial_commits_and_discards_match_reference_int8_every_backend` |
| draft_generation | Negate each captured residual channel | `engine_handover_counts_memory_reseeds_long_suffix_and_detects_main_only_restore` |

## Final per-test dev seconds

Concurrent times within a binary overlap. The three reference namespaces share
one 28.652-second binary; their case durations are not separate binary wall times.

### yskn-engine / tests/cpu_stages.rs

| Test | Seconds |
| --- | ---: |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.002 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.002 |
| `cpu_stage_override_rejects_unknown_values_before_opening_weights` | 0.018 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.028 |
| `independent_cache_sides_match_scalar_rows_and_errors` | 0.327 |
| `auto_kv5_model_matches_scalar_across_pass_and_tile_boundaries` | 3.630 |
| `auto_int8_model_matches_scalar_across_pass_and_tile_boundaries` | 3.646 |
| `auto_int8_profile_model_matches_scalar_across_pass_and_tile_boundaries` | 3.650 |
| `forced_reference_kv5_model_matches_scalar_across_pass_and_tile_boundaries` | 3.765 |
| `forced_reference_int8_model_matches_scalar_across_pass_and_tile_boundaries` | 3.829 |
| `forced_reference_int8_profile_model_matches_scalar_across_pass_and_tile_boundaries` | 3.950 |

### yskn-engine / tests/draft_fast.rs

| Test | Seconds |
| --- | ---: |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.004 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.005 |
| `loading_rejects_the_reference_errors` | 0.014 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.016 |
| `processing_failure_keeps_prior_rows_and_matches_reference_on_every_backend` | 1.752 |
| `failed_rows_snapshot_restore_and_reset_match_reference` | 5.306 |
| `equal_heads_twice_wrapped_match_reference_on_every_backend_and_thread_count` | 8.048 |
| `large_ring_and_push_above_256_rows_match_reference_on_every_backend_and_thread_count` | 10.027 |
| `grouped_heads_wide_hidden_and_partial_vocab_match_reference_on_every_backend_and_thread_count` | 11.200 |

### yskn-engine / tests/draft_generation.rs

| Test | Seconds |
| --- | ---: |
| `metal_fast_draft_generation_equals_reference_draft_and_no_draft` | 0.014 |
| `engine_handover_counts_memory_reseeds_long_suffix_and_detects_main_only_restore` | 2.900 |
| `neural_int8_seed17_is_lossless_across_joint_restore_and_reused_prefix` | 13.450 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.017 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.001 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.006 |
| `neural_kv5_seed17_is_lossless_across_joint_restore_and_reused_prefix` | 13.493 |
| `neural_int8_profile_seed17_is_lossless_across_joint_restore_and_reused_prefix` | 13.605 |
| `neural_int8_seedabcdef_is_lossless_across_joint_restore_and_reused_prefix` | 14.588 |
| `neural_int8_profile_seedabcdef_is_lossless_across_joint_restore_and_reused_prefix` | 14.722 |
| `neural_kv5_seedabcdef_is_lossless_across_joint_restore_and_reused_prefix` | 13.263 |
| `neural_kv5_greedy_is_lossless_across_joint_restore_and_reused_prefix` | 16.581 |
| `neural_int8_profile_greedy_is_lossless_across_joint_restore_and_reused_prefix` | 16.606 |
| `neural_int8_greedy_is_lossless_across_joint_restore_and_reused_prefix` | 16.622 |
| `cpu_fast_draft_generation_equals_reference_draft_and_no_draft` | 16.934 |

### yskn-engine / tests/model.rs

| Test | Seconds |
| --- | ---: |
| `persistent_range_pool_preserves_order_borrows_and_recovers_after_errors` | 0.008 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.021 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.004 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.006 |
| `load_selects_env_kv_and_explicit_constructors_keep_int8_default` | 0.085 |
| `failure_after_a_completed_pass_preserves_logits_position_and_state` | 0.367 |
| `dense_embeddings_are_identical` | 0.669 |
| `recurrent_model_is_identical_for_all_kernels_threads_pass_widths_and_chunks` | 1.271 |
| `other_hidden_ffn_widths_and_padded_vocabulary_are_identical` | 1.738 |
| `invalid_requests_preserve_logits_position_and_future_sequence_state` | 3.017 |
| `mixed_model_crosses_attention_tiles_from_position_three_without_bit_changes` | 10.007 |
| `kv5_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation` | 11.671 |
| `int8_profile_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation` | 11.675 |

### yskn-engine / tests/speculation.rs

| Test | Seconds |
| --- | ---: |
| `all_last_and_none_head_requests_preserve_the_same_future_numeric_state` | 0.154 |
| `failed_numeric_pass_preserves_committed_state_and_future_rows` | 0.425 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.021 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.004 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.006 |
| `snapshots_restore_prefix_without_growing_persistent_storage_and_reject_misuse` | 1.973 |
| `snapshot_after_partial_commit_restores_exact_prefix_and_future_state` | 2.346 |
| `randomized_partial_commits_and_discards_match_reference_kv5_every_backend` | 22.013 |
| `randomized_partial_commits_and_discards_match_reference_int8_every_backend` | 22.055 |
| `randomized_partial_commits_and_discards_match_reference_int8_profile_every_backend` | 22.174 |
| `residual_capture_matches_reference_across_widths_kv_backends_and_session_operations` | 25.779 |
| `seed17_generation_is_lossless_repeatable_for_every_proposer` | 32.052 |
| `greedy_generation_is_lossless_repeatable_for_every_proposer` | 32.162 |
| `seedabcdef_generation_is_lossless_repeatable_for_every_proposer` | 32.164 |

### yskn-ref / tests/models.rs

| Test | Seconds |
| --- | ---: |
| `draft_wiring::delivered_captures_are_released_and_joint_snapshots_do_not_retain_them` | 1.502 |
| `draft_wiring::depth_dependent_teacher_uses_block_successors_and_the_distinct_frontier` | 1.729 |
| `draft_wiring::sampled_hook_matches_direct_draft_with_frontier_seed_and_absolute_positions` | 1.940 |
| `draft_wiring::main_only_restore_or_reset_is_detected_even_after_advancing_past_the_draft` | 2.030 |
| `kv_attention::codec_error_in_later_model_row_is_atomic_and_guard_uses_absolute_position` | 0.579 |
| `kv_attention::folder_modes_select_profile_manifest_default_and_preserve_plain_int8_default` | 0.582 |
| `kv_attention::manifest_profile_and_weights_paths_cannot_escape_folder` | 0.002 |
| `kv_attention::profile_modes_reject_absent_profiles_and_wrong_attention_counts_or_heads` | 0.010 |
| `model_invariants::header_dimensions_and_padded_vocabulary_are_used_in_execution` | 0.103 |
| `model_invariants::invalid_token_pass_is_atomic_and_reset_reproduces_the_initial_state` | 0.074 |
| `model_invariants::mixed_model_loads_with_qwen3_5_rotary_width` | 0.011 |
| `model_invariants::mixed_model_rejects_wrong_head_width_and_unknown_architecture` | 0.009 |
| `draft_wiring::synthetic_teacher_accepts_correct_captures_and_rejects_lagged_or_swapped_slots` | 4.337 |
| `model_invariants::mixed_model_row_and_chunk_invariants` | 2.029 |
| `model_invariants::model_folder_manifest_rounds_context_capacity_and_selects_weights` | 0.021 |
| `model_invariants::model_vectors_reject_nonfinite_values_before_execution` | 0.003 |
| `model_invariants::multirow_pass_matches_single_rows_bit_for_bit` | 0.718 |
| `model_invariants::plain_bfloat_embeddings_use_the_same_model_path` | 0.072 |
| `draft_wiring::reused_and_restored_sessions_reseed_after_a_long_new_prompt` | 5.527 |
| `model_invariants::prefill_chunk_boundaries_do_not_change_logits_or_future_state` | 0.798 |
| `model_invariants::synthetic_logits_sample_repeatably_by_seed_and_absolute_position` | 0.109 |
| `support::synthetic_writer_builds_reproducible_mixed_model` | 0.015 |
| `support::synthetic_writer_released_layout_uses_warp_and_released_bit_widths` | 0.004 |
| `support::synthetic_writer_supports_other_dimensions_and_plain_embeddings` | 0.006 |
| `model_invariants::provisional_prefix_commit_matches_committing_pass` | 1.041 |
| `kv_attention::int8_freezes_logits_and_preserves_long_pass_reset_and_continuation` | 8.112 |
| `kv_attention::cached_bases_preserve_pre_05j_mixed_model_logits_and_stored_state` | 17.151 |
| `model_invariants::mixed_model_unaligned_long_pass_matches_single_rows_and_uneven_chunks` | 19.399 |
| `draft_wiring::reference_neural_seed17_matches_k0_across_joint_restore_and_reused_prefix` | 24.605 |
| `draft_wiring::reference_neural_seedabcdef_matches_k0_across_joint_restore_and_reused_prefix` | 25.472 |
| `kv_attention::kv5_freezes_logits_and_preserves_long_pass_reset_and_continuation` | 23.695 |
| `kv_attention::int8_profile_freezes_logits_and_preserves_long_pass_reset_and_continuation` | 23.831 |
| `draft_wiring::reference_neural_greedy_matches_k0_across_joint_restore_and_reused_prefix` | 28.652 |
