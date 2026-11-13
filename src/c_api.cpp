// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "engine.hpp"
#include "engine35.hpp"
#include "specx.hpp"
#include <chrono>
#include "yskn_abi.h"
#include <string>
#include <atomic>
namespace yskn { extern std::atomic<int> g_specx_cancel; }   
static thread_local std::string g_err;
struct yskn_engine { yskn::Engine* e; yskn::Engine35* q; std::atomic<int> cancel{0}; };   
static yskn::Engine& E(yskn_engine* e) { if (!e->e) throw std::runtime_error("this API is not available for the qwen3_5 engine yet"); return *e->e; }
#define YSKN_TRY(body) try { body; return 0; } catch (const std::exception& ex) { g_err = ex.what(); return -1; }
extern "C" yskn_engine* yskn_open(const char* path, int max_ctx, int device, int flags) {
  try { cudaSetDevice(device); std::string arch; { yskn::Container c; c.open(path); arch = c.model.arch; }
    if (arch == "qwen3_5") return new yskn_engine{nullptr, new yskn::Engine35(path, max_ctx, flags)};
    return new yskn_engine{new yskn::Engine(path, max_ctx, flags), nullptr}; } catch (const std::exception& ex) { g_err = ex.what(); return nullptr; }
}
extern "C" void yskn_close(yskn_engine* e) { if (e) { delete e->e; delete e->q; delete e; } }
extern "C" int yskn_reset(yskn_engine* e) { YSKN_TRY(if (e->q) e->q->reset(); else e->e->reset()) }
extern "C" int yskn_prefill(yskn_engine* e, const int32_t* ids, int n, int32_t* first) { YSKN_TRY(*first = e->q ? e->q->prefill(ids, n) : e->e->prefill(ids, n)) }
extern "C" int yskn_decode(yskn_engine* e, int32_t* out, int steps) { YSKN_TRY(if (e->q) e->q->decode(out, steps); else e->e->decode(out, steps)) }
extern "C" int yskn_nll(yskn_engine* e, const int32_t* ids, int n, float* out) { YSKN_TRY(if (e->q) e->q->nll(ids, n, out); else e->e->nll(ids, n, out)) }
extern "C" int yskn_logits(yskn_engine* e, float* out) { YSKN_TRY(if (e->q) e->q->logits(out); else e->e->logits(out)) }
extern "C" int yskn_position(yskn_engine* e) { return e->q ? e->q->position() : e->e->position(); }
extern "C" const char* yskn_last_error(void) { return g_err.c_str(); }
extern "C"  int yskn_debug_buffer(yskn_engine* e, int which, int M, uint16_t* out) { YSKN_TRY(E(e).debug_buffer(which, M, out)) }
extern "C"  int yskn_debug_hidden(yskn_engine* e, const int32_t* ids, int n, int upto, int mode, float* out) { YSKN_TRY(if (e->q) e->q->debug_hidden(ids, n, upto, mode, out); else e->e->debug_hidden(ids, n, upto, mode, out)) }
extern "C" int yskn_verify(yskn_engine* e, const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals) { YSKN_TRY(if (e->q) e->q->verify(ids, M, use_graph, out_argmax, top_ids, top_vals); else e->e->verify(ids, M, use_graph, out_argmax, top_ids, top_vals)) }
extern "C" int yskn_verify_nll(yskn_engine* e, const int32_t* ids, const int32_t* targets, int M, int use_graph, int32_t* out_argmax, float* out_nll) { YSKN_TRY(if (e->q) e->q->verify(ids, M, use_graph, out_argmax, nullptr, nullptr, targets, out_nll); else e->e->verify(ids, M, use_graph, out_argmax, nullptr, nullptr, targets, out_nll)) }
extern "C" int yskn_commit(yskn_engine* e, int n_accept) { YSKN_TRY(if (e->q) e->q->commit(n_accept); else e->e->commit(n_accept)) }
extern "C" int yskn_prefill_rows(yskn_engine* e, const int32_t* ids, int n, int32_t* first, int32_t* out_argmax, int32_t* top_ids, float* top_vals) { YSKN_TRY(*first = e->q ? e->q->prefill_rows(ids, n, out_argmax, top_ids, top_vals) : e->e->prefill_rows(ids, n, out_argmax, top_ids, top_vals)) }
extern "C" int yskn_hidden_rows(yskn_engine* e, int n, float* out) { YSKN_TRY(if (e->q) e->q->hidden_rows(n, out); else e->e->hidden_rows(n, out)) }
extern "C" int yskn_logits_rows(yskn_engine* e, int row0, int n, float* out) { YSKN_TRY(if (e->q) e->q->logits_rows(row0, n, out); else e->e->logits_rows(row0, n, out)) }
extern "C" int yskn_time_verify(yskn_engine* e, int M, int mask, int iters, float* ms) { YSKN_TRY(*ms = e->q ? e->q->time_verify(M, mask, iters) : e->e->time_verify(M, mask, iters)) }
extern "C" int yskn_spec_generate(yskn_engine* e, const int32_t* prompt, int n_prompt, int n_gen, const int32_t* cfg, int n_cfg, int32_t* out_ids, double* stats, double* times_ms) {
  try {
    if (n_cfg < 12) throw std::runtime_error("spec_generate: need at least 12 config values");
    e->cancel.store(0);                                   
    yskn::SpecxConfig c; c.cancel = &e->cancel; c.gamma = cfg[0]; c.use_d0 = cfg[1]; c.use_d1 = cfg[2]; c.d0_nmin = cfg[3]; c.d0_nmax = cfg[4]; c.d0_priority_n = cfg[5];
    c.extend = cfg[6]; c.d1_rejected = cfg[7]; c.auto_gamma = cfg[8]; c.use_graph = cfg[9]; c.use_d3 = cfg[10]; c.d3_steps = cfg[11];
    if (n_cfg > 12) c.d1_bigram = cfg[12]; if (n_cfg > 13) c.d1_static = cfg[13]; if (n_cfg > 14) c.d3_extend = cfg[14]; if (n_cfg > 15) c.d3_lazy = cfg[15]; if (n_cfg > 16) c.d3_gate = cfg[16];
    if (n_cfg > 17) c.gamma_long = cfg[17]; if (n_cfg > 18) c.long_n = cfg[18]; if (n_cfg > 19) c.d1_trigram = cfg[19];
    if (n_cfg > 20) c.eos1 = cfg[20]; if (n_cfg > 21) c.eos2 = cfg[21];
    if (n_cfg > 22) c.long_d0 = cfg[22]; if (n_cfg > 23) c.sel = cfg[23]; if (n_cfg > 24) c.cost_tok_us = cfg[24];
    if (n_cfg > 25) c.cost_row_us = cfg[25]; if (n_cfg > 26) c.cost_mtp_us = cfg[26]; if (n_cfg > 27) c.mtp_auto = cfg[27];
    if (n_cfg > 28) c.prmode = cfg[28]; if (n_cfg > 29) c.cost_row2_us = cfg[29];
    if (n_cfg > 30) c.lfrow = cfg[30]; if (n_cfg > 31) c.wk_q_pm = cfg[31];
    if (n_cfg > 32) c.mark_tok = cfg[32];
    if (n_cfg > 33) c.cost_tok_k = cfg[33]; if (n_cfg > 34) c.cost_row2_k = cfg[34]; if (n_cfg > 35) c.cost_mtp_k = cfg[35];
    if (n_cfg > 36) c.bdraft = cfg[36]; if (n_cfg > 37) c.bdr_long_n = cfg[37]; if (n_cfg > 38) c.bdr_adapt = cfg[38]; if (n_cfg > 39) c.bdr_probe = cfg[39];
    if (n_cfg > 40) c.bdr_d0_len = cfg[40];
    if (n_cfg > 41) c.prefill_head = cfg[41]; if (n_cfg > 42) c.snap = cfg[42]; if (n_cfg > 43) c.snap_tok = cfg[43];   
    auto run = [&](auto& y) {
      using clk = std::chrono::steady_clock; using ms = std::chrono::duration<double, std::milli>;
      const auto t0 = clk::now(); out_ids[0] = y.prefill(prompt, n_prompt); const auto t1 = clk::now();
      y.generate(n_gen, out_ids + 1); const auto t2 = clk::now();
      if (times_ms) { times_ms[0] = ms(t1 - t0).count(); times_ms[1] = ms(t2 - t1).count(); }
      if (stats) { const yskn::SpecxStats& s = y.stats; double* o = stats; o[0] = (double)s.passes; o[1] = (double)s.tokens; o[2] = (double)s.chain_none;
        o[3] = (double)s.drafted[0]; o[4] = (double)s.accepted[0]; o[5] = (double)s.drafted[1]; o[6] = (double)s.accepted[1];
        for (int M = 1; M <= 16; ++M) o[6 + M] = (double)s.rows_hist[M];
        o[23] = (double)s.drafted[2]; o[24] = (double)s.accepted[2]; o[25] = s.ms_verify; o[26] = s.ms_draft; o[27] = s.ms_total;
        o[28] = s.ms_commit; o[29] = s.ms_record; o[30] = (double)s.produced; o[31] = (double)s.eos;
        if (n_cfg > 30) { o[32] = (double)s.leaves; o[33] = (double)s.leaf_breaks; o[34] = (double)s.leaf_hits; }
        if (n_cfg > 32) { o[35] = s.ms_mark; o[36] = (double)s.mark_at; }   
        if (n_cfg > 36) { o[37] = (double)s.bdr_rounds; o[38] = (double)s.bdr_d0; o[39] = (double)s.bdr_probes; o[40] = s.ms_ctx; }   
        if (n_cfg > 42) { o[41] = (double)s.sh_mode; o[42] = (double)s.sh_start; } }   
    };
    if (e->q) { yskn::Specx35 y(*e->q, c, e->q->vocab()); run(y);
                if (c.snap) { const auto& s = y.seq(); e->q->snap_note(s.data(), (int)s.size() - 1); } }   
    else { yskn::Specx y(*e->e, c, e->e->vocab()); run(y); }
    return 0;
  } catch (const std::exception& ex) { g_err = ex.what(); return -1; }
}


extern "C" YSKN_API int yskn_cancel(yskn_engine* e) { e->cancel.store(1); return 0; }
extern "C" YSKN_API int yskn_set_visual(yskn_engine* e, const uint16_t* features, const int32_t* slots, int n_image,
                                        const int32_t* positions, int n_prompt, int delta) {
  YSKN_TRY(if (!e->q) throw std::runtime_error("visual: qwen3_5 engine only");
            e->q->set_visual(features, slots, n_image, positions, n_prompt, delta))
}


extern "C" YSKN_API int yskn_snap_info(yskn_engine* e, int32_t* out4, int32_t* ids, int max_ids) { YSKN_TRY(
  if (!e->q) throw std::runtime_error("snap: qwen3_5 engine only");
  const auto& lv = e->q->snap_live(); out4[0] = e->q->position(); out4[1] = e->q->snap_pos(); out4[2] = e->q->snap_live_ok() ? 1 : 0;
  out4[3] = (int32_t)lv.size(); if (ids) for (int i = 0; i < (int)lv.size() && i < max_ids; ++i) ids[i] = lv[i]) }
extern "C" int yskn_lfrow_table(yskn_engine* e, const uint8_t* tab48) { YSKN_TRY(if (e->q) e->q->lfrow_table(tab48); else throw std::runtime_error("lfrow: 27B engine only")) }
extern "C" int yskn_lfrow_promote(yskn_engine* e, int src, int dst) { YSKN_TRY(if (e->q) e->q->lfrow_promote(src, dst); else throw std::runtime_error("lfrow: 27B engine only")) }
extern "C" int yskn_set_opt(yskn_engine* e, int key, int value) { YSKN_TRY(if (e->q) e->q->set_opt(key, value); else throw std::runtime_error("set_opt: 27B engine only")) }
extern "C" int yskn_set_capture(yskn_engine* e, const int32_t* layers, int n) { YSKN_TRY(if (e->q) e->q->set_capture(layers, n); else E(e).set_capture(layers, n)) }
extern "C"  int yskn_kz_debug(yskn_engine* e, int layer) { YSKN_TRY(if (!e->q) throw std::runtime_error("kz_debug: qwen3_5 engine only"); e->q->set_kz_debug(layer)) }
extern "C"  int yskn_kz_debug_read(yskn_engine* e, int n, float* o, uint16_t* q) { YSKN_TRY(if (!e->q) throw std::runtime_error("kz_debug: qwen3_5 engine only"); e->q->kz_debug_read(n, o, q)) }
extern "C"  int yskn_kv_format(yskn_engine* e) { return e->q ? e->q->kv_format() : 0; }
extern "C"  int yskn_kv16_read(yskn_engine* e, int layer, int pos0, int n, uint16_t* k, uint16_t* v) {
  YSKN_TRY(if (!e->q) throw std::runtime_error("kv16_read: qwen3_5 only"); e->q->kv16_read(layer, pos0, n, k, v)) }
extern "C"  int yskn_kv8_read(yskn_engine* e, int layer, int pos0, int n, int8_t* k8, float* ks, int8_t* v8, float* vs) {
  YSKN_TRY(if (!e->q) throw std::runtime_error("kv8_read: qwen3_5 engine only"); e->q->kv8_read(layer, pos0, n, k8, ks, v8, vs)) }
extern "C" int yskn_capture_rows(yskn_engine* e, int k, int n, float* out) { YSKN_TRY(if (e->q) e->q->capture_rows(k, n, out); else E(e).capture_rows(k, n, out)) }
extern "C" int yskn_load_draft(yskn_engine* e, const char* path, const char* head_ids) { YSKN_TRY(if (e->q) e->q->load_draft(path, head_ids ? head_ids : ""); else e->e->load_draft(path, head_ids)) }
extern "C" int yskn_draft_round(yskn_engine* e, int mode, int a, int steps, int use_graph) { YSKN_TRY(if (e->q) e->q->draft_round(mode, a, steps, use_graph); else e->e->draft_round(mode, a, steps, use_graph)) }
extern "C" int yskn_verify_device(yskn_engine* e, int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids) { YSKN_TRY(if (e->q) e->q->verify_device(M, use_graph, out_argmax, out_toks, top_ids); else e->e->verify_device(M, use_graph, out_argmax, out_toks, top_ids)) }
extern "C" int yskn_set_verify_tokens(yskn_engine* e, const int32_t* ids, int first_row, int n) { YSKN_TRY(if (e->q) e->q->set_verify_tokens(ids, first_row, n); else e->e->set_verify_tokens(ids, first_row, n)) }
extern "C" int yskn_draft_debug(yskn_engine* e, int which, int n, float* out) { YSKN_TRY(E(e).draft_debug(which, n, out)) }
extern "C" int yskn_time_draft(yskn_engine* e, int mode, int a, int steps, int iters, float* ms) { YSKN_TRY(*ms = E(e).time_draft(mode, a, steps, iters)) }
extern "C" int yskn_specx_load_static(const char* path) { YSKN_TRY(yskn::specx_load_static(path)) }
extern "C" unsigned yskn_xsum_nonfinite(yskn_engine* e) { return e->e ? e->e->xsum_nonfinite() : 0u; }
extern "C" int yskn_draft_catchup(yskn_engine* e, int p0, int n, const int32_t* toks, int use_graph) { YSKN_TRY(if (e->q) e->q->draft_catchup(p0, n, toks, use_graph); else e->e->draft_catchup(p0, n, toks, use_graph)) }
extern "C"  int yskn_debug_kv_hash(yskn_engine* e, uint64_t* out) { YSKN_TRY(*out = E(e).debug_kv_hash()) }
extern "C" YSKN_API int yskn_state_hash(yskn_engine* e, uint64_t* out, int max_out, int* n_out) { YSKN_TRY(if (!e->q) throw std::runtime_error("state_hash: qwen3_5 engine only"); *n_out = e->q->state_hash(out, max_out)) }
extern "C"  int yskn_chk_set(yskn_engine* e, const char* name, const long long* c, int Kc, unsigned long long key, int w_max, int* idx) { YSKN_TRY(if (!e->q) throw std::runtime_error("chk: qwen3_5 engine only"); *idx = e->q->chk_set(name, c, Kc, key, w_max)) }
extern "C"  int yskn_chk_stats(yskn_engine* e, unsigned long long* out, int reset) { YSKN_TRY(if (!e->q) throw std::runtime_error("chk: qwen3_5 engine only"); e->q->chk_stats(out, reset)) }
extern "C"  int yskn_chk_fault(yskn_engine* e, int kind, const char* name, int a, int b) { YSKN_TRY(if (!e->q) throw std::runtime_error("chk: qwen3_5 engine only"); e->q->chk_fault(kind, name, a, b)) }
extern "C"  int yskn_chk_enable(yskn_engine* e, int on) { YSKN_TRY(if (!e->q) throw std::runtime_error("chk: qwen3_5 engine only"); e->q->chk_enable(on)) }
extern "C" YSKN_API int yskn_load_bdraft(yskn_engine* e, const char* path) { YSKN_TRY(if (!e->q) throw std::runtime_error("bdraft: qwen3_5 engine only"); e->q->load_bdraft(path)) }
extern "C" YSKN_API int yskn_bdraft_ctx(yskn_engine* e, int row0, int n, int pos0) { YSKN_TRY(if (!e->q) throw std::runtime_error("bdraft: qwen3_5 engine only"); e->q->bdraft_ctx(row0, n, pos0)) }
extern "C" YSKN_API int yskn_prefill_rows_k20(yskn_engine* e, const int32_t* ids, int n, int32_t* out_first, int32_t* out_argmax, int32_t* k20_ids, float* k20_vals) {
  YSKN_TRY(if (!e->q) throw std::runtime_error("prefill_rows_k20: qwen3_5 engine only"); *out_first = e->q->prefill_rows_k20(ids, n, out_argmax, k20_ids, k20_vals)) }
extern "C" YSKN_API int yskn_bdraft_step(yskn_engine* e, int n_ctx, int pos_ctx, int draft, int use_graph) { YSKN_TRY(if (!e->q) throw std::runtime_error("bdraft: qwen3_5 engine only"); e->q->bdraft_step(n_ctx, pos_ctx, draft, use_graph)) }
extern "C" YSKN_API int yskn_bdraft_draft(yskn_engine* e, int use_graph, int32_t* drafts) { YSKN_TRY(if (!e->q) throw std::runtime_error("bdraft: qwen3_5 engine only"); e->q->bdraft_draft(use_graph); if (drafts) e->q->bdraft_out(drafts)) }
extern "C"  int yskn_bdraft_debug(yskn_engine* e, const float* feats, int n, int pos0, int32_t anchor, float* hidden, int32_t* top_ids, float* top_vals, int32_t* path) {
  YSKN_TRY(if (!e->q) throw std::runtime_error("bdraft: qwen3_5 engine only"); e->q->bdraft_debug(feats, n, pos0, anchor, hidden, top_ids, top_vals, path)) }
extern "C" YSKN_API int yskn_bdraft_ring(yskn_engine* e, int layer, int slot0, int n, uint16_t* k_out, uint16_t* v_out) { YSKN_TRY(if (!e->q) throw std::runtime_error("bdraft: qwen3_5 engine only"); e->q->bdraft_ring(layer, slot0, n, k_out, v_out)) }

extern "C"  int yskn_shared_info(yskn_engine* e, uint64_t* out) { YSKN_TRY(if (!e->q) throw std::runtime_error("27B only"); e->q->shared_info(out)) }

extern "C"  int yskn_nll_layer(yskn_engine* e,const int32_t* ids,int n,float* out) {
  try {if(!e->q)throw std::runtime_error("nll_layer needs 27B");e->q->nll_layer(ids,n,out);return 0;}
  catch(const std::exception& ex){g_err=ex.what();return -1;}
}
extern "C"  int yskn_layer_info(yskn_engine* e,uint64_t* out) {
  YSKN_TRY(if(!e->q)throw std::runtime_error("layer_info needs 27B");e->q->layer_info(out))
}
