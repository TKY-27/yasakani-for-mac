# Upstream name glossary

This glossary maps source identifiers to the descriptive names used in the specification. The same short identifier can mean different things in different routines; context in the first column is part of the mapping. The linked topic documents define the arrays and operations.

## Model and orchestration

| Upstream name | Descriptive name | Evidence |
|---|---|---|
| `Q35_H`, `H_` | Main hidden width, 5120 | `src/q35.hpp:13`; `src/engine35.cpp:216-218` |
| `Q35_NH`, `Q35_KVH`, `Q35_HD`, `Q35_RD` | Attention query heads, KV heads, channel width, rotary width | `src/q35.hpp:13` |
| `Q35_GK`, `Q35_GV`, `Q35_GD`, `Q35_CONV`, `Q35_Z` | Recurrent query/key heads, value heads, channel width, convolution width, output width | `src/q35.hpp:14` |
| `F_`, `V_`, `nlayers_` | Header feed-forward width, vocabulary size, layer count | `src/engine35.cpp:214-218` |
| `L_`, `Y` | Main layer array and current layer | `src/engine35.cpp:236-266,768-800` |
| `h_` | Main residual hidden array, stored FP32 | `src/engine35.hpp:326`; `src/engine35.cpp:829-833` |
| `M`, `HM_`, `PM_`, `Pmax_` | Current row count, head-slice limit, prefill chunk limit, allocated prefill capacity | `src/engine35.hpp:239,313-314`; `src/engine35.cpp:194-213,835-837,1244-1253` |
| `pos_`, `d_pos_` | Committed input-token count / next input position | `src/engine35.cpp:1325-1328` |
| `last_M_`, `pre_committed_` | Last verification row count, one-row state already committed | `src/engine35.cpp:1303-1317,1321-1329` |
| `HEAD_NONE`, `HEAD_LAST`, `HEAD_ALL` | No head, final-row head, every-row head | `src/engine35.hpp:218`; `src/engine35.cpp:834-839` |
| `tc_rows_`, `layer_rows_` | Resident rows in alternate layer-major prefill, requested resident capacity | `src/engine35.hpp:329,339`; `src/engine35.cpp:993-1048` |
| `q35_pf_launch` | Alternate preparation/recurrence predicate: more than 64 rows and PFALT enabled | `src/q35_kernels.cu:602` |

## Compressed arithmetic and normalization

| Upstream name | Descriptive name | Evidence |
|---|---|---|
| `lut16` | binary16 weight lookup table | src/loader.cpp:19-23 |
| `gamma` in build_lut8_tau | weight-table quantization scale | src/loader.cpp:21-24 |
| `lut8` | signed-byte weight lookup table | src/loader.cpp:22-23 |
| `tau` | effective output-row scale | src/loader.cpp:24 |
| `signs_in`, `sg_in` | input rotation signs | src/loader.cpp:34; src/q35_kernels.cu:1461 |
| `signs_out`, `sg_out` | inverse output-rotation signs | src/loader.cpp:36; src/q35_kernels.cu:73-79 |
| `qx` | quantized activation bytes | src/a8_prologue.cuh:122-132 |
| `alpha` | activation scale per row and 512-element group | src/a8_prologue.cuh:130-132 |
| `rm` | multiplier converting activation scales to 24-bit-relative integers | src/lnord.cu:52-54 |
| `A`, `A0`, `A1` in LNORD | integerized activation-group scale | src/lnord.cu:309-311 |
| `us` | per-row integer-sum quantum | src/lnord.cu:53; src/a8_prologue.cuh:15 |
| `d` in LNORD | signed-32 dot product over one 512-element input group | src/lnord.cu:290-316 |
| `acc64` | signed-64 weighted sum across input groups | src/lnord.cu:308-311 |
| `part` in XSUM<=16 | weighted integer-sum storage (typed as float pointer upstream) | src/a8_prologue.cuh:10-12 |
| `ybuf_`, `part` in XSUM>16 | binary32 finalized compressed sum | src/engine35.hpp:252-255; src/pfa.cu:301-310 |
| `xsm_`, `xsum`, `XSM` | integer-weighted accumulation mode | src/engine35.hpp:248; src/lnord.cu:308-315 |
| `S_`, `Q35Xsm.S` | residual sum-of-squares blocks | src/q35_kernels.cu:1447-1458 |
| `Q35_SST` | sum-of-squares storage stride (16 floats) | src/q35.hpp:20 |
| `nrs.y`, `rr` (normalization) | deferred inverse RMS | src/q35_kernels.cu:81-85,104-106 |
| `xu`, `xn_` on main XSUM GDN path | bfloat16 unnormalized gain-adjusted hidden copy for a/b | src/q35_kernels.cu:1459-1460 |
| `rbf` | nearest-even bfloat16 roundtrip | src/q35_kernels.cu:35-37 |
| `round_f16` | nearest-even binary16 roundtrip | src/a8_prologue.cuh:36-38 |
| `fwht512_v`, `fwht512_blocks`, `q35_wfwht512` | ascending-stride 512-element Hadamard transform | src/a8_prologue.cuh:41-50,87-97; src/q35_kernels.cu:247-259 |
| `lo_now_`, `LO` | interleaved activation-byte layout for <=8 rows | src/engine35.cpp:827; src/a8_prologue.cuh:120-132 |
| `nmode` | post-processing input representation selector | src/engine35.hpp:253-255 |
| `lnord` | compressed table-lookup integer matrix backend | src/lnord.cu:265-316 |
| `launch_pfa` | multi-row compressed matrix backend | src/engine35.cpp:416-419; src/pfa.cu:355-364 |
| `launch_q35_g6` | >=256-row predecoded-int8 matrix backend | src/q35_g6.cu:310-350 |

## Recurrent layer, attention, and KV profile

| Upstream identifier / context | Descriptive name used in 03/04 | Evidence |
|---|---|---|
| `bf2f`, `f2bf`, `rbf` | decode BF16, encode BF16, `bf` round-and-widen | `src/q35_kernels.cu:35-37` |
| `q35_rsqrt`, `q35_exp` | native/portable reciprocal-square-root and exponential selector | `src/q35_kernels.cu:42-44` |
| `q35_silu`, `q35_sigmoid`, `q35_softplus` | `silu`, `sigmoid`, `softplus` | `src/q35_kernels.cu:113-116` |
| `q35_pm_on` | portable-math switch | `src/q35_kernels.cu:42-44` |
| `Q35AbArgs.xn`, `xn_` at gate projection | `weighted_hidden_bf16` | `src/q35_kernels.cu:230`; `src/engine35.cpp:773-774` |
| `AB.S`, `S_` at normalization | `hidden_square_sum` (not recurrent state) | `src/q35_kernels.cu:81-85`; `src/engine35.cpp:773-774` |
| `wa`, `wb` | dense log-decay / update-gate projection weights | `src/engine35.cpp:257`; `src/q35_kernels.cu:749` |
| `ab`, `pab`, `Y.pab` | `gates[row,96]`, b followed by a | `src/q35_kernels.cu:749-784`, `src/q35_kernels.cu:882-886` |
| `mixed`, `pmix`, `pmx`, `Y.pmix` | BF16 `projected[row,10240]` before convolution | `src/q35_kernels.cu:843-846`; `src/engine35.cpp:771-789` |
| `z`, `z_` | output-gate input after BF16 rounding | `src/q35_kernels.cu:848-851`, `src/q35_kernels.cu:977` |
| `conv_w`, `conv`, `Y.conv` | `conv_weight[channel,tap]` | `src/q35_kernels.cu:867-877` |
| `hist`, `Y.hist` | three oldest-to-newest `history` rows | `src/q35_kernels.cu:916-924` |
| `act` | BF16 `convolved` value | `src/q35_kernels.cu:877-878` |
| GDN `q`, `gq_`, `Y.tq` | normalized/scaled recurrent `query` | `src/q35_kernels.cu:895`; `src/engine35.cpp:785-789` |
| GDN `k`, `gk_`, `Y.tk` | normalized recurrent `key` | `src/q35_kernels.cu:896`; `src/engine35.cpp:785-789` |
| GDN `v`, `gv_`, `Y.tv` | BF16-rounded recurrent `value` stored FP32 | `src/q35_kernels.cu:879-880`; `src/engine35.cpp:785-789` |
| `beta`, `gbeta_`, `Y.tbeta`, `bt` | `update_gate` / BF16 `beta` | `src/q35_kernels.cu:883`, `src/q35_kernels.cu:934` |
| `g`, `gg_`, `Y.tg`, `ng` | FP32 `log_decay` | `src/q35_kernels.cu:886`, `src/q35_kernels.cu:926-934` |
| `eg` | exponentiated `decay` | `src/q35_kernels.cu:934` |
| `A.S`, `Y.S`, `Sp` at state update | persistent `state[value_head,key_channel,value_channel]` | `src/q35_kernels.cu:910-914` |
| `s[32]` at state update | one key-channel partition of working matrix `R` | `src/q35_kernels.cu:912-914`, `src/q35_kernels.cu:941-951` |
| `kv`, `skv` | `prediction_part` | `src/q35_kernels.cu:939-944` |
| `d` at state update | `correction` | `src/q35_kernels.cu:944` |
| `o`, `so` at state update | `output_part` | `src/q35_kernels.cu:945-956` |
| `core`, `core_` | BF16-rounded recurrent output in FP32 array | `src/q35_kernels.cu:938`, `src/q35_kernels.cu:955-956` |
| `gw`, `gnorm` | `gdn_norm_weight` | `src/q35_kernels.cu:965`, `src/q35_kernels.cu:976` |
| `n_commit` | number of input rows retained in persistent state/history | `src/q35_kernels.cu:916-924`, `src/q35_kernels.cu:949-951` |
| `wk`, `d_wk_`, `W.wk` | `row_table` | `src/lfrow_kernels.cuh:39`, `src/lfrow_kernels.cuh:87-91` |
| `wk[row]`, `dp` | semantic `depth[row]` | `src/lfrow_kernels.cuh:39`, `src/q35_atxpk.cu:396-402` |
| `wk[16+step]`, `e` | execution schedule entry, high bit marks leaf | `src/lfrow_kernels.cuh:89-110` |
| `leaf` | nonmutating side candidate | `src/lfrow_kernels.cuh:119-140` |
| `q16`, `q16_` | `query_bf16` | `src/q35_kernels.cu:1057` |
| `gate16`, `gate16_` | `saved_gate_bf16` | `src/q35_kernels.cu:1038`, `src/q35_kernels.cu:1357` |
| `qn_w`, `kn_w`, `Y.qn`, `Y.kn` | shared query/key head normalization weights | `src/engine35.cpp:241`; `src/q35_kernels.cu:1017` |
| `q35_inv_freq` | `frequency_bits` | `src/q35_kernels.cu:1091-1094` |
| `d_pos`, `base` | `base_position` | `src/q35_kernels.cu:1020`; `src/q35_atxpk.cu:402` |
| `kvx_sign` | `cache_sign` | `src/kvx_codec.cuh:28-31` |
| `kvx_rotate(...false/true)` | `rotate_cache` / `inverse_rotate_cache` | `src/kvx_codec.cuh:35-48` |
| `kvx_store`, `q35_kv8_store` | `encode_int8` | `src/kvx_codec.cuh:50-69`; `src/q35_kernels.cu:990-992` |
| `k8`, `v8` | persistent int8 key/value payload | `src/engine35.cpp:250-251` |
| `ks`, `vs` | persistent int8 key/value row scale | `src/engine35.cpp:250-251` |
| `kp`, `vp` | packed key/value payload | `src/engine35.cpp:249-251` |
| `kps`, `vps` | packed key/value row scale | `src/engine35.cpp:250-251` |
| `stg_k8_`, `stg_v8_`, `kstg`, `vstg` | temporary pre-packing int8 payload / staging stride | `src/engine35.cpp:693-698`; `src/kvq_fast.cuh:75-78` |
| `TbRow.g`, `R.g` | profile group index | `src/kvq_fast.cuh:65-78` |
| `TbRow.y`, `R.y` | `dequantized_int8_row` element | `src/kvq_fast.cuh:78` |
| `TbG.mode` | profile quantization mode (0 trellis, 1 uniform) | `src/kvq_fast.cuh:60`; `src/kvq_fast.cuh:192-212` |
| `TbG.kk` | `bits`, field width | `src/kvq_fast.cuh:60`, `src/kvq_fast.cuh:212` |
| `TbG.tix` | codebook table index | `src/kvq_fast.cuh:60`, `src/kvq_fast.cuh:159` |
| `TbG.guard` | physical-position guard | `src/kvq_fast.cuh:60`, `src/kvq_fast.cuh:189` |
| `TbG.m` | profile basis count | `src/kvq_fast.cuh:61`, `src/kvq_fast.cuh:90-91` |
| `TbG.has_mu`, `mud`, `mup` | mean-present flag, mean element, mean data | `src/kvq_fast.cuh:61`, `src/kvq_fast.cuh:85-89` |
| `TbG.nm`, `A.nm`, `mi` | scale-candidate count and index | `src/kvq_fast.cuh:56`, `src/kvq_fast.cuh:198-206` |
| `Q.r`, VIT2 `r` | prepared `residual` | `src/kvq_fast.cuh:82-95`; `src/kvq_vit2.cuh:103-127` |
| `Q.cj0`, `Q.cj1` | FP16-rounded basis `coefficient` | `src/kvq_fast.cuh:90-91` |
| `Q.uj0`, `Q.uj1` | FP16 basis element | `src/kvq_fast.cuh:90-91` |
| `ci`, `ci_s` | `integer_codebook` entry | `src/kvq_fast.cuh:162-163` |
| `tb_cdec`, `cdec` | unrounded decoded encoder codebook | `src/kvq_gpu.cuh:53`; `src/kvq_fast.cuh:163` |
| `tabk`, `tabv`, `pk_tabf` | BF16 `attention_codebook` for key/value | `src/kvq_atxpk.hpp:39-45`; `src/q35_atxpk.cu:66` |
| `S.xi`, `W.xi` | integer `target` | `src/kvq_fast.cuh:166-168`; `src/kvq_vit2.cuh:135-138` |
| `S.cost` | general trellis `cost` | `src/kvq_fast.cuh:108-123` |
| `W.mk` | VIT2 `reduced_cost` | `src/kvq_vit2.cuh:79-88`, `src/kvq_vit2.cuh:153` |
| `bp` | `backpointer` | `src/kvq_fast.cuh:120-135`; `src/kvq_vit2.cuh:88`, `src/kvq_vit2.cuh:187-192` |
| `stout`, `W.st`, `stt` | `selected_state` | `src/kvq_fast.cuh:131-146`; `src/kvq_vit2.cuh:184-212` |
| `fld`, `field` | packed low-bit `field` | `src/kvq_fast.cuh:212-229` |
| `psc`, `s`, `scl` in encoder | selected FP16-rounded `row_scale`, stored FP32 | `src/kvq_fast.cuh:206-212`, `src/kvq_fast.cuh:231` |
| `PkRow.lo`, `ph`, `own`, `self_` | split `first`, `prefix_end`, `self_slot`, `include_self` | `src/q35_atxpk.cu:390-409` |
| `len`, `S`, `per` in attention | causal `length`, `split_count`, `per_split` | `src/q35_atxpk.cu:402-407` |
| `tbeg`, `ntiles` | first 32-key tile position and tile count | `src/q35_atxpk3.cuh:193-195` |
| `pk_mma`, `atx_mma` | `mma16` unresolved instruction primitive | `src/q35_atxpk.cu:35-37`; `src/q35_atx.cu:30-32` |
| `m0/m1`, `l0/l1`, `o` in attention | running `maximum`, `denominator`, `numerator` | `src/q35_atxpk3.cuh:301-319` |
| `c0/c1` in attention | online-softmax `rescale` | `src/q35_atxpk3.cuh:396-397` |
| `x_pf` | BF16 `value_weight` | `src/q35_atxpk3.cuh:405-406` |
| `pm_`, `pl_`, `pacc_` | split maxima, denominators, numerators | `src/q35_atxpk3.cuh:471-475`; `src/q35_kernels.cu:1285-1308` |
| `q35_combine_kernel` | split combination, inverse cache rotation, output gate, A8 preparation | `src/q35_kernels.cu:1279-1363` |

## Output selection, block draft, and speculation

| Upstream name | Name used in these topics | Source |
|---|---|---|
| `logits_`, `A.logits`, `lg` | Main binary32 logits | `src/engine35.cpp:813-819` |
| `out_argmax`, `d_next8_`, `am` | Selected main successor token (sampled when enabled) | `src/engine35.cpp:813-821`; `src/specx.cpp:376` |
| `tau`, `us_` | Per-output weight scale, common XSUM scale | `src/misc_kernels.cu:67-69`; `src/engine35.cpp:810-813` |
| `Q35SmpParams.temp/top_p/top_k/min_p` | Main sampling temperature / truncation parameters | `src/q35.hpp:113-116` |
| `kh_mix` | mix64 | `src/smp_hash.cuh:8-9` |
| `kh_gumbel` | Token-keyed Gumbel value | `src/smp_hash.cuh:12-15` |
| `smp_p_.version` | Sampler version 2 (inverse cumulative) or 3 (Gumbel) | `src/q35_sample.cu:195-202` |
| `q35_smp2_on` | Second top-k extraction kernel enabled | `src/q35_sample.cu:209` |
| `bdraft2.fc`, `bdr_.fc` | Main-feature projection | `src/engine35.cpp:1626,1775-1778` |
| `caps`, `cap_` | Captured post-layer main residual features | `src/engine35.cpp:830-833,1650` |
| `bdraft2.hidden_norm`, `hnorm` | Context projection norm | `src/engine35.cpp:1628,1778` |
| `bdraft2.norm`, `fnorm` | Final proposal norm | `src/engine35.cpp:1628,1818` |
| `akp`, `mkp`, `dyn` | Attention/MLP dynamic-convolution coefficients | `src/engine35.cpp:1799-1816` |
| `abase`, `mbase` | Attention/MLP two-tap convolution bases | `src/engine35.cpp:1633`; `src/q35_bdraft.cu:504-519` |
| `ln1`, `ln2` | Attention/MLP input norm weights | `src/engine35.cpp:1631,1799-1816` |
| `rk`, `rv` | BF16 draft context key/value rings | `src/engine35.cpp:1634`; `src/q35_bdraft.cu:98-103` |
| `BDR_B`, `BDR_J` | Maximum block rows16, maximum proposals15 | `src/q35.hpp:214` |
| `hf` | Final-normalized proposal hidden vectors | `src/engine35.cpp:1818` |
| `sk_bt` | Rank-512 projection matrix | `src/engine35.cpp:1729` |
| `sk_a`, `sk_as` | Approximate signed-int8 weight and row scale | `src/engine35.cpp:1730-1731` |
| `sk_w`, `sk_ws` | Full-width signed-int8 rescore weight and row scale | `src/engine35.cpp:1732-1733` |
| `sk_m` | Shortlist width | `src/engine35.cpp:1669-1671` |
| `hp`, `hpv` | Hidden-to-correction projection and correction feature | `src/engine35.cpp:1636,1833`; `src/q35_bdraft.cu:299-305` |
| `pc`, `sc` | Predecessor/successor correction embeddings | `src/engine35.cpp:1637`; `src/q35_bdraft.cu:406-411` |
| `cpl`, `cpl_pm` | Draft coupling temperature in integer thousandths | `src/engine35.hpp:183`; `src/engine35.cpp:1845-1850` |
| `alt`, `alt3`, `marg` | Second candidate, third candidate, top-to-second margin | `src/q35_bdraft.cu:435-453` |
| `S_`, `F`, `xF` | Sequence, frontier index, frontier token | `src/specx.cpp:206` |
| `xP`, `xQ` | One/two tokens preceding frontier | `src/specx.cpp:206` |
| `ng_`, `indexed_` | N-gram ending-position index and indexing cursor | `src/specx.cpp:37-58` |
| `d0_lookup`, `c0`, `n0` | Copy lookup, proposal, matching suffix length | `src/specx.cpp:46-58,216` |
| `T_`, `T2_`, `T3_` | Token, pair, triple next-token tables | `src/specx.cpp:60-68,89-92` |
| `g_short`, `g_all`, `g_eff` | Short, maximum, selected copy limits | `src/specx.cpp:212-221` |
| `dema_[0]`, `dema_[1]` | Neural/copy acceptance averages | `src/specx.cpp:178,248,384` |
| `dpend_`, `dprobe_` | Deferred unused-proposal comparison list, copy-choice probe counter | `src/specx.cpp:249-265,441-447` |
| `b8_ema`, `b16_ema` | Expected8-row yield, extra16-row yield | `src/specx.cpp:146,156-159,377-380` |
| `b16_since`, `full8_ema`, `b16_n` | Passes since16 observation, full8 acceptance average,16 observations | `src/specx.cpp:146-155,179,380` |
| `b16_probe_cur` | Current long-probe interval | `src/specx.cpp:148,159` |
| `HB_EDGES`, `HB_PI`, `HB_RHO`, `HB_RHO3` | Margin edges, acceptance, second and third candidate probabilities | `src/hcap_tables.inc:4-7` |
| `R`, `v2`, `v3` | Trunk reach, second/third leaf expected yields | `src/q35_bdraft.cu:672-678` |
| `L`, `nl`, `dep`, `rk` in HCAP | Trunk draft count, leaf count, leaf depth, leaf rank | `src/q35_bdraft.cu:680-705` |
| `wk`, `d_wk_` | 48-byte depth/traversal/candidate-rank table | `src/q35_bdraft.cu:699-705` |
| `a`, `a_all`, `n_commit` | Accepted prefix afterEOS, beforeEOS, commit argument including side hit | `src/specx.cpp:376-423` |
| `hit_row` | Matched side candidate's physical row | `src/specx.cpp:414-422` |
| `dctx_n`, `dctx_pos` | Pending accepted draft-context row count and absolute start | `src/specx.cpp:435-437` |
| `sh_pos_`, `sh_ids_` | Saved snapshot cut and prefix tokens | `src/engine35.hpp:128-129` |
| `sh_cur_`, `sh_note_`, `sh_epoch_` | Retained live-prefix tokens, note epoch, state epoch | `src/engine35.hpp:130-133` |
