// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "engine.hpp"
#include "draft_kernels.hpp"
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include "devmem.hpp"   
namespace yskn {


thread_local int g_pdl_rows = 1;
bool pdl_enabled() {
  static const bool on = [] { const char* e = std::getenv("YSKN_NO_PDL"); return !(e && e[0] == '1'); }();
  static const int max_m = [] { const char* e = std::getenv("YSKN_PDL_MAX_M"); return e ? std::atoi(e) : 16; }();   
  return on && g_pdl_rows <= max_m;
}
template <class T2> static T2* dz(size_t bytes) { T2* p = nullptr; cudaMalloc(&p, bytes); cudaMemset(p, 0, bytes); return p; }

Engine::Engine(const std::string& path, int max_ctx, int flags) : flags_(flags) {
  DevMemScope dm_scope(this);
  { const char* e = std::getenv("YSKN_NO_A4B"); a4b_ = !(e && e[0] == '1'); }
  { const char* e = std::getenv("YSKN_PFA"); if (e && e[0]) pfa_ = std::atoi(e);          
    const char* m = std::getenv("YSKN_PFA_M"); if (m && std::atoi(m) > 0) tpm_ = std::atoi(m);
    const char* w = std::getenv("YSKN_PFA_WPC"); if (w && w[0]) pfa_wpc_ = std::atoi(w); }
  cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking);
  m_.load(path, max_ctx, 32);
  const ModelDesc& c = m_.cfg; a_.Mmax = 32;
  a_.h = dz<float>((size_t)a_.Mmax * c.hidden * 4); a_.h2 = dz<float>((size_t)a_.Mmax * c.hidden * 4); a_.q16 = dz<uint16_t>((size_t)a_.Mmax * c.heads * 128 * 2); a_.o16 = dz<uint16_t>((size_t)a_.Mmax * c.hidden * 2);
  a_.xff = dz<uint16_t>((size_t)a_.Mmax * c.ffn * 2); a_.qx_ff = dz<int8_t>((size_t)a_.Mmax * c.ffn); a_.alpha_ff = dz<float>((size_t)a_.Mmax * (c.ffn / 512) * 4);
  a_.pm = dz<float>((size_t)a_.Mmax * m_.splits * 28 * 4); a_.pl = dz<float>((size_t)a_.Mmax * m_.splits * 28 * 4); a_.pacc = dz<float>((size_t)a_.Mmax * m_.splits * 28 * 128 * 4);
  a_.logits = dz<float>((size_t)a_.Mmax * c.vocab * 4); a_.blk_val = dz<float>((size_t)a_.Mmax * 256 * 4); a_.blk_idx = dz<int>((size_t)a_.Mmax * 256 * 4);
  a_.blk_max = dz<float>((size_t)a_.Mmax * 256 * 4); a_.blk_sum = dz<float>((size_t)a_.Mmax * 256 * 4);
  a_.d_pos = dz<int>(4); a_.d_tok = dz<int>(4); a_.d_next = dz<int>(4); a_.d_target = dz<int>(4); a_.d_nll = dz<float>(4);
  a_.d_toks = dz<int>((size_t)a_.Mmax * 4); a_.d_next8 = dz<int>((size_t)a_.Mmax * 4); a_.d_target8 = dz<int>((size_t)a_.Mmax * 4); a_.d_nll8 = dz<float>((size_t)a_.Mmax * 4);
  a_.tk_val = dz<float>((size_t)a_.Mmax * 256 * 8 * 4); a_.tk_idx = dz<int>((size_t)a_.Mmax * 256 * 8 * 4);
  a_.top_val = dz<float>((size_t)a_.Mmax * 8 * 4); a_.top_idx = dz<int>((size_t)a_.Mmax * 8 * 4);
  cudaMallocHost(&h_dec_, (size_t)m_.max_ctx * 4);
  cudaMallocHost(&h_pin_, 352 * 4); cudaMallocHost(&h_pinf_, 288 * 4); cudaMallocHost(&h_vin_, 32 * 4); d_vin_ = dz<int>(32 * 4);
  pr_am_ = dz<int>((size_t)m_.max_ctx * 4); pr_ti_ = dz<int>((size_t)m_.max_ctx * 8 * 4); pr_tv_ = dz<float>((size_t)m_.max_ctx * 8 * 4);
  hcur_ = a_.h; hnext_ = a_.h2;
  { const char* e = std::getenv("YSKN_NO_ROPE_TABLE");
    if (!(e && e[0] == '1')) { cudaMalloc(&rope_, (size_t)m_.max_ctx * 64 * sizeof(float2)); launch_rope_table(rope_, m_.max_ctx, c.rope_theta, st_); cudaStreamSynchronize(st_); } }
  { const char* e = std::getenv("YSKN_ATTN_F32_M"); if (e && e[0]) f32_m_ = std::atoi(e);          
    const char* f = std::getenv("YSKN_PFA_ATTN_F32"); if (f && f[0]) tpf_ = std::atoi(f);
    if (f32_m_ > 0 || tpf_) { const size_t n = (size_t)c.kv_heads * m_.max_ctx * 32;
      kvf_ = KvF32{dz<float4>(n * 16), dz<float4>(n * 16), dz<float4>(n * 16)}; } }
  
  
  
  
  { const char* e = std::getenv("YSKN_NO_SHARED_PART");
    if (!(e && e[0] == '1')) {
      auto sz = [&](const DevTensor& t) { return (size_t)t.dev.n_workers * a_.Mmax * t.dev.N; };
      size_t need = 0;
      for (const DevLayer& L : m_.layers) need = std::max({need, sz(L.q) + sz(L.k) + sz(L.v), sz(L.o), sz(L.gate) + sz(L.up), sz(L.down)});
      part_scratch_ = dz<float>(need * 4); float* sp = part_scratch_;
      for (DevLayer& L : m_.layers) {
        for (DevTensor* t : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) { cudaFree(t->part); t->part = nullptr; }
        L.q.part = sp; L.k.part = sp + sz(L.q); L.v.part = sp + sz(L.q) + sz(L.k);
        L.o.part = sp; L.gate.part = sp; L.up.part = sp + sz(L.gate); L.down.part = sp;
      } } }
  { const char* e = std::getenv("YSKN_XSUM");   
    xsum_ = !e || !e[0] || (e[0] >= '1' && e[0] <= '4'); xsm_mode_ = xsum_ ? ((e && e[0] >= '1' && e[0] <= '4') ? e[0] - '0' : 4) : 0;   
    { const char* s = std::getenv("YSKN_XSM_STATS"); xsm_stats_ = s && s[0] == '1'; }   
    if (xsum_) { const size_t nr = std::max((size_t)a_.Mmax, (size_t)tpm_);   
      xsm_rr_ = dz<float>(nr * 8 * 4); xsm_us_ = dz<float>(nr * 4); xsm_nf_ = dz<unsigned>(64);   
      if (xsm_mode_ == 3) { xsm_qx2_ = dz<int8_t>((size_t)a_.Mmax * c.hidden); xsm_alpha2_ = dz<float>((size_t)a_.Mmax * 8 * 4); xsm_cnt_ = dz<unsigned>(64); xsm_dbg_ = dz<unsigned>(64); } } }
  
  
  
  { const char* e = std::getenv("YSKN_LNORD");
    lno_ = !e || !e[0] ? 1 : e[0] == '0' ? 0 : e[0] == 'b' ? 2 : e[0] == 'g' ? 3 : e[0] == 'n' ? 4 : 1;
    if (lno_) { std::vector<DevTensor*> ts;
      for (DevLayer& L : m_.layers) for (DevTensor* t : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) if (lno_ != 3 || t == &L.gate || t == &L.up) ts.push_back(t);
      if (lno_ == 1 || lno_ == 4) ts.push_back(&m_.head);
      std::fprintf(stderr, "[lnord] lane-ordered codes built: %.1f MB\n", lno_build(ts) / 1e6); } }
  { const char* e = std::getenv("YSKN_PREFETCH"); if (e && e[0]) pf_ = std::atoi(e);
    const char* c2 = std::getenv("YSKN_PREFETCH_CTAS"); if (c2 && c2[0]) pf_ctas_ = std::atoi(c2);
    if (pf_ > 0) {                                                  
      std::vector<PfEntry> plan; const int D = pf_;
      for (size_t l = 0; l < m_.layers.size(); ++l) {
        const DevLayer& L = m_.layers[l]; const int g0 = 4 * (int)l;
        auto add = [&](const DevTensor& t, int g) { if (t.dev.lno) plan.push_back(PfEntry{t.dev.lno, lnord_bytes(t.dev.N, t.dev.Kc, t.dev.k_default), g - D}); };
        add(L.q, g0); add(L.k, g0); add(L.v, g0); add(L.o, g0 + 1); add(L.gate, g0 + 2); add(L.up, g0 + 2); add(L.down, g0 + 3); }
      pf_n_ = (int)plan.size(); cudaMalloc(&pf_plan_, plan.size() * sizeof(PfEntry));
      cudaMemcpy(pf_plan_, plan.data(), plan.size() * sizeof(PfEntry), cudaMemcpyHostToDevice);
      prog_ = dz<int>(4); cudaStreamCreateWithFlags(&pst_, cudaStreamNonBlocking);
      cudaEventCreateWithFlags(&pev0_, cudaEventDisableTiming); cudaEventCreateWithFlags(&pev1_, cudaEventDisableTiming);
      std::fprintf(stderr, "[prefetch] %d buffers, lookahead %d GEMVs, %d CTAs\n", pf_n_, D, pf_ctas_); } }
  { const char* e = std::getenv("YSKN_DEBUG_ALIAS_LAYERS");   
    if (e && e[0] == '1') {                                       
      std::fprintf(stderr, "[debug] YSKN_DEBUG_ALIAS_LAYERS: all layers use layer 0's weights (timing only)\n");
      for (size_t l = 1; l < m_.layers.size(); ++l) {
        DevLayer& L = m_.layers[l]; const DevLayer& L0 = m_.layers[0];
        L.q.dev = L0.q.dev; L.k.dev = L0.k.dev; L.v.dev = L0.v.dev; L.o.dev = L0.o.dev; L.gate.dev = L0.gate.dev; L.up.dev = L0.up.dev; L.down.dev = L0.down.dev; } } }
  { const char* e = std::getenv("YSKN_FUSE2"); fuse2_ = !(e && e[0] == '0') && a4b_ && !m_.head.qm_in;
    for (const DevLayer& L : m_.layers) if (L.q.qm_in || L.gate.qm_in) fuse2_ = false;
    if (fuse2_) { fuse_cnt_ = dz<unsigned>((size_t)a_.Mmax * 4); am_cnt_ = dz<unsigned>((size_t)a_.Mmax * 4); tk_cnt_ = dz<unsigned>((size_t)a_.Mmax * 4); } }
  res_fuse_ = a4b_ && !m_.head.qm_in;                       
  
  { const char* e = std::getenv("YSKN_RESFUSE"); if (!(e && e[0] == '1')) res_fuse_ = false; }
  for (const DevLayer& L : m_.layers) if (L.q.qm_in || L.gate.qm_in) res_fuse_ = false;
}
void Engine::input_norm(int M) {
  const ModelDesc& c = m_.cfg; DevLayer& L0 = m_.layers[0];
  launch_norm_xform_a8(hcur_, L0.ln1, c.eps, XformA8Args{nullptr, L0.q.signs_in, L0.q.qm_in, c.hidden, L0.q.m_in, a_.qx_ff, a_.alpha_ff, M}, st_);
}
size_t Engine::lno_build(const std::vector<DevTensor*>& ts) {
  size_t bytes = 0;
  for (DevTensor* t : ts) {
    if (t->lno || (t->bits != 4 && t->bits != 6 && t->bits != 8) || t->dev.tile_k || t->dev.group_worker || (t->dev.N % 16) || (t->dev.Kc % 512)) continue;
    const size_t sz = lnord_bytes(t->dev.N, t->dev.Kc, t->bits);         
    cudaMalloc(&t->lno, sz); lnord_build(t->frag, t->dev.N, t->dev.Kc, t->bits, t->lno, st_); t->dev.lno = t->lno; bytes += sz; }
  cudaStreamSynchronize(st_);
  const cudaError_t ce = cudaGetLastError(); if (ce != cudaSuccess) throw std::runtime_error(std::string("LNORD build: ") + cudaGetErrorString(ce));
  return bytes;
}
Engine::~Engine() {
  cudaStreamSynchronize(st_);
  if (gexec_) cudaGraphExecDestroy(gexec_);
  for (auto& kv : vgraph_) cudaGraphExecDestroy(kv.second);
  for (auto& kv : dgraph_) cudaGraphExecDestroy(kv.second);
  for (auto& kv : cgraph_) cudaGraphExecDestroy(kv.second);
  for (auto& kv : sgraph_) cudaGraphExecDestroy(kv.second);
  if (h_toks_) cudaFreeHost(h_toks_);
  if (h_pin_) cudaFreeHost(h_pin_);
  if (h_pinf_) cudaFreeHost(h_pinf_);
  if (h_vin_) cudaFreeHost(h_vin_);
  devmem_release(this);                                                 
  cudaStreamDestroy(st_);
}
void Engine::reset() { pos_ = 0; launch_set_int(a_.d_pos, 0, st_); if (dm_.on) cudaMemsetAsync(dm_.last, 0, (size_t)3 * m_.cfg.hidden * 4, st_); cudaStreamSynchronize(st_); }

static int pdl_prew_m() { static const int v = [] { const char* e = std::getenv("YSKN_PDL_PREW_M"); return e ? std::atoi(e) : 1 << 30; }(); return v; }
static GemvLaunch gemv_base(const Model& m, int M, int stage_bytes) { GemvLaunch P{}; P.M = M; P.stage_bytes = stage_bytes; P.pdl_prew_m = pdl_prew_m(); (void)m; return P; }
void Engine::xsmm(GemmLaunch& G, bool on) const { if (xsum_ && on) { G.xsum = 1; G.us_out = xsm_us_; } }

void Engine::xsm_head_gemm(int M, bool topk) {
  const ModelDesc& c = m_.cfg; g_pdl_rows = M;
  if (!res_fuse_ && xsm_mode_ != 4) launch_norm_xform_a8(hcur_, m_.final_norm, c.eps, xsmx(XformA8Args{nullptr, m_.head.signs_in, m_.head.qm_in, c.hidden, m_.head.m_in, a_.qx_ff, a_.alpha_ff, M}, true), st_);
  for (int k = 4; k <= 8; ++k) if (!m_.head.tiles_by_k[k].empty()) { TensorDev td = m_.head.dev; td.k_default = k; GemmLaunch Gm{td, m_.head.part, a_.qx_ff, a_.alpha_ff, M, 2 * 8 * 16 * k}; xsmm(Gm, true); launch_gemm(Gm, st_); }
  launch_argmax(ArgmaxArgs{m_.head.part, m_.head.tau, c.vocab, a_.logits, a_.d_next8, a_.blk_val, a_.blk_idx, m_.head.dev.n_workers, M, 0, M, xsm_us_, xsm_rr_, c.eps}, st_);
  if (topk) launch_topk8(TopkArgs{a_.logits, c.vocab, a_.tk_val, a_.tk_idx, a_.top_val, a_.top_idx, M}, st_);
}
void Engine::xsmg(GemvLaunch& P, bool main_model) const { if (xsum_ && main_model) { P.xsum = 1; P.us_out = xsm_us_; } }
XformA8Args Engine::xsmx(XformA8Args A, bool on) const { if (xsum_ && on) { A.xsum = xsm_mode_ == 1 ? 1 : 2; A.r_out = xsm_rr_; A.nf = xsm_nf_; A.stats = xsm_stats_; } return A; }
unsigned Engine::xsum_nonfinite() const { unsigned v[3] = {0, 0, 0}; if (xsm_nf_) { cudaStreamSynchronize(st_); cudaMemcpy(v, xsm_nf_, 12, cudaMemcpyDeviceToHost);
    if (xsm_dbg_) { unsigned d[7]; cudaMemcpy(d, xsm_dbg_, 28, cudaMemcpyDeviceToHost); std::fprintf(stderr, "[xsum] down epilogues per block: %u %u %u %u %u %u %u\n", d[0], d[1], d[2], d[3], d[4], d[5], d[6]); }
    { unsigned long long mx = 0; cudaMemcpy(&mx, xsm_nf_ + 4, 8, cudaMemcpyDeviceToHost); if (mx) std::fprintf(stderr, "[xsum] max |INT64 acc| (o, down) %llu = 2^%.1f (limit 2^63)\n", mx, std::log2((double)mx)); }
    float u, w; std::memcpy(&u, v + 1, 4); std::memcpy(&w, v + 2, 4); std::fprintf(stderr, "[xsum] non-finite %u, max |u| %.1f, max |v| %.1f (FP16 max 65504)\n", v[0], u, w); } return v[0]; }
static void add_seg(GemvLaunch& P, const DevTensor& t) {
  const int ntiles = (t.dev.N + 63) / 64, ctas = (ntiles + GEMV_WARPS - 1) / GEMV_WARPS;
  P.seg[P.n_seg] = GemvSeg{t.dev, t.part, P.n_ctas}; P.n_seg++; P.n_ctas += ctas; P.max_workers = std::max(P.max_workers, t.dev.n_workers);
}

void Engine::layer_forward(int l, int M, bool fused) {
  const bool l_main = l >= 0;
  if (xsum_ && l >= 0 && !fused && xsm_mode_ == 3) throw std::runtime_error("XSUM mode 3: the down epilogue lives in the GEMV (prefill uses the GEMV path)");
  const bool xsm = xsum_ && l >= 0; const int nmode = xsm ? xsm_mode_ : 0;   
  g_pdl_rows = M;
  const ModelDesc& c = m_.cfg; DevLayer& L = l >= 0 ? m_.layers[l] : dm_.L; const int sb = l >= 0 ? (int)m_.stage_bytes_body : dm_.stage_bytes;
  float* const hcur_save = hcur_; if (l < 0) hcur_ = dm_.hd;  
  const int* const dpos = l >= 0 ? a_.d_pos : dm_.d_pos;
  const bool G = kmask_ & 1, S = (kmask_ & 2) != 0;          
  
  
  const bool fin = (nmode == 3 || nmode == 4) && l > 0;                  
  const bool f2 = fuse2_ && l >= 0 && !xsm && fused;                      
  if (l >= 0) last_f2_ = f2;                                              
  if (S && !res_fuse_ && !fin && !(f2 && (l > 0 || emb_f2_))) launch_norm_xform_a8(hcur_, L.ln1, c.eps, xsmx(XformA8Args{nullptr, L.q.signs_in, L.q.qm_in, c.hidden, L.q.m_in, a_.qx_ff, a_.alpha_ff, M}, xsm), st_);
  if (fused) {   
    if (G) { GemvLaunch P = gemv_base(m_, M, sb); add_seg(P, L.q); add_seg(P, L.k); add_seg(P, L.v); P.qx = fin && nmode == 3 ? xsm_qx2_ : a_.qx_ff; P.alpha = fin && nmode == 3 ? xsm_alpha2_ : a_.alpha_ff; xsmg(P, l_main); launch_gemv(P, st_); pf_mark(l, 0); }
  } else {
    for (DevTensor* t : {&L.q, &L.k, &L.v}) { GemmLaunch Gm{t->dev, t->part, a_.qx_ff, a_.alpha_ff, M, 2 * 8 * 16 * t->bits}; xsmm(Gm, xsm); launch_gemm(Gm, st_); }
  }
  
  QkvPostArgs Q{L.q.part, L.k.part, L.v.part, L.q.tau, L.k.tau, L.v.tau, L.q.dev.n_workers, L.k.dev.n_workers, L.v.dev.n_workers,
                L.q.signs_out, L.k.signs_out, L.v.signs_out, L.bq, L.bk, M, dpos, a_.q16, L.kv, m_.max_ctx, c.rope_theta, rope_};
  if (xsm) { Q.us = xsm_us_; Q.rr = xsm_rr_; Q.xsm_eps = c.eps; }
  if (fuse2_ && M == 1) Q.kvfuse = 1;                                     
  if (S) { launch_qkv_post(Q, st_); if (!Q.kvfuse) launch_kv_group_quant(KvQuantArgs{L.kv, dpos, M, m_.max_ctx}, st_); }
  
  static const bool attn_generic = [] { const char* e = std::getenv("YSKN_ATTN_GENERIC"); return e && e[0] == '1'; }();
  if (S && attn_generic && rope_ && !L.o.qm_in && a4b_) {                
    const int G = c.heads / c.kv_heads;
    launch_attention_gqa(AttnGArgs{a_.q16, L.kv, m_.max_ctx, M, dpos, m_.splits, a_.pm, a_.pl, a_.pacc, rope_, c.kv_heads, c.heads, 0.08838834764831845f}, G, c.head_dim, c.head_dim, st_);
    launch_attn_combine_gqa_xform(AttnGCombineArgs{a_.pm, a_.pl, a_.pacc, m_.splits, M, L.bv, a_.o16, nullptr, c.heads, G}, c.head_dim, L.o.signs_in, a_.qx_ff, a_.alpha_ff, st_); }
  else if (S) { if (!(kmask_ & 16)) {                                    
      const AttnArgs AA{a_.q16, L.kv, m_.max_ctx, M, dpos, m_.splits, a_.pm, a_.pl, a_.pacc, c.rope_theta, rope_};
      if (f32_m_ > 0 && M >= f32_m_) { launch_kv_dequant(AA, kvf_, st_); launch_attention_f32(AA, kvf_, st_); }   
      else launch_attention(AA, st_); }
    const AttnCombineArgs CA{a_.pm, a_.pl, a_.pacc, m_.splits, M, L.bv, a_.o16};
    if (!L.o.qm_in && a4b_) launch_attn_combine_xform(CA, L.o.signs_in, a_.qx_ff, a_.alpha_ff, st_);         
    else { launch_attn_combine(CA, st_); launch_xform_a8(XformA8Args{a_.o16, L.o.signs_in, L.o.qm_in, c.hidden, L.o.m_in, a_.qx_ff, a_.alpha_ff, M}, st_); } }
  
  if (fused) { if (G) { GemvLaunch P = gemv_base(m_, M, sb); add_seg(P, L.o); P.qx = a_.qx_ff; P.alpha = a_.alpha_ff; xsmg(P, l_main); launch_gemv(P, st_); pf_mark(l, 1); } }
  else { GemmLaunch Gm{L.o.dev, L.o.part, a_.qx_ff, a_.alpha_ff, M, 2 * 8 * 16 * L.o.bits}; xsmm(Gm, xsm); launch_gemm(Gm, st_); }
  if (f2) { if (S) launch_res_norm_last(ResNormLastArgs{ResidualPostArgs{L.o.part, L.o.tau, L.o.dev.n_workers, L.o.signs_out, M, c.hidden, hcur_}, L.ln2, c.eps, L.gate.signs_in, a_.qx_ff, a_.alpha_ff, fuse_cnt_}, st_); }
  else if (res_fuse_) { if (S) launch_residual_norm_xform(ResNormXArgs{ResidualPostArgs{L.o.part, L.o.tau, L.o.dev.n_workers, L.o.signs_out, M, c.hidden, nullptr},
                    hcur_, hnext_, L.ln2, c.eps, L.gate.signs_in, a_.qx_ff, a_.alpha_ff}, st_); std::swap(hcur_, hnext_); }
  else if (S && nmode == 4) launch_xsum_res_xform(XsmResXArgs{L.o.part, xsm_us_, L.o.tau, L.o.signs_out, hcur_, L.ln2, L.gate.signs_in,
                                  a_.qx_ff, a_.alpha_ff, xsm_rr_, c.hidden, M, 0, xsm_nf_, xsm_stats_}, st_);   
  else if (S) { launch_residual_post(ResidualPostArgs{L.o.part, L.o.tau, L.o.dev.n_workers, L.o.signs_out, M, c.hidden, hcur_, xsm ? xsm_us_ : nullptr}, st_);
    launch_norm_xform_a8(hcur_, L.ln2, c.eps, xsmx(XformA8Args{nullptr, L.gate.signs_in, L.gate.qm_in, c.hidden, L.gate.m_in, a_.qx_ff, a_.alpha_ff, M}, xsm), st_); }
  
  if (fused) { if (G) { GemvLaunch P = gemv_base(m_, M, sb); add_seg(P, L.gate); add_seg(P, L.up); P.qx = a_.qx_ff; P.alpha = a_.alpha_ff; xsmg(P, l_main); launch_gemv(P, st_); pf_mark(l, 2); } }
  else {
    for (DevTensor* t : {&L.gate, &L.up}) { GemmLaunch Gm{t->dev, t->part, a_.qx_ff, a_.alpha_ff, M, 2 * 8 * 16 * t->bits}; xsmm(Gm, xsm); launch_gemm(Gm, st_); } }
  if (S) { MlpPostArgs MA{L.gate.part, L.up.part, L.gate.tau, L.up.tau, L.gate.dev.n_workers, L.gate.signs_out, L.up.signs_out, M, a_.xff, xsm ? xsm_us_ : nullptr, xsm ? xsm_rr_ : nullptr, c.eps}; MA.F = c.ffn;
    if (!L.down.qm_in && a4b_) launch_mlp_post_xform(MA, L.down.signs_in, a_.qx_ff, a_.alpha_ff, st_);      
    else { launch_mlp_post(MA, st_); launch_xform_a8(XformA8Args{a_.xff, L.down.signs_in, L.down.qm_in, c.ffn, L.down.m_in, a_.qx_ff, a_.alpha_ff, M}, st_); } }
  
  if (fused) { if (G) { GemvLaunch P = gemv_base(m_, M, sb); add_seg(P, L.down); P.fuse = 0; P.qx = a_.qx_ff; P.alpha = a_.alpha_ff; xsmg(P, l_main);
    if (nmode == 3) { const bool lastl = l + 1 == c.layers; P.e_fuse = 1; P.e_u16 = 0; P.e_h = hcur_; P.e_tau = L.down.tau; P.e_sg_out = L.down.signs_out;
      P.e_gamma = lastl ? m_.final_norm : m_.layers[l + 1].ln1; P.e_sg_in = lastl ? m_.head.signs_in : m_.layers[l + 1].q.signs_in;
      P.e_qx = xsm_qx2_; P.e_alpha = xsm_alpha2_; P.e_S = xsm_rr_; P.e_cnt = xsm_cnt_; P.e_dbg = xsm_dbg_; }
    launch_gemv(P, st_); pf_mark(l, 3); } }
  else { GemmLaunch Gm{L.down.dev, L.down.part, a_.qx_ff, a_.alpha_ff, M, 2 * 8 * 16 * L.down.bits}; xsmm(Gm, xsm); launch_gemm(Gm, st_); }
  if (f2) { const bool last = l + 1 == c.layers;
    if (S) launch_res_norm_last(ResNormLastArgs{ResidualPostArgs{L.down.part, L.down.tau, L.down.dev.n_workers, L.down.signs_out, M, c.hidden, hcur_},
                                 last ? m_.final_norm : m_.layers[l + 1].ln1, c.eps, last ? m_.head.signs_in : m_.layers[l + 1].q.signs_in, a_.qx_ff, a_.alpha_ff, fuse_cnt_}, st_); }
  else if (res_fuse_) { const bool last = l + 1 == c.layers;
    const uint16_t* gamma = last ? m_.final_norm : m_.layers[l + 1].ln1; const int8_t* sg = last ? m_.head.signs_in : m_.layers[l + 1].q.signs_in;
    if (S) launch_residual_norm_xform(ResNormXArgs{ResidualPostArgs{L.down.part, L.down.tau, L.down.dev.n_workers, L.down.signs_out, M, c.hidden, nullptr},
                               hcur_, hnext_, gamma, c.eps, sg, a_.qx_ff, a_.alpha_ff}, st_); std::swap(hcur_, hnext_); }
  else if (S && nmode == 4) { const bool lastl = l + 1 == c.layers;                 
    launch_xsum_res_xform(XsmResXArgs{L.down.part, xsm_us_, L.down.tau, L.down.signs_out, hcur_, lastl ? m_.final_norm : m_.layers[l + 1].ln1,
                            lastl ? m_.head.signs_in : m_.layers[l + 1].q.signs_in, a_.qx_ff, a_.alpha_ff, xsm_rr_, c.hidden, M, 0, xsm_nf_, xsm_stats_}, st_); }
  else if (S && nmode != 3) launch_residual_post(ResidualPostArgs{L.down.part, L.down.tau, L.down.dev.n_workers, L.down.signs_out, M, c.hidden, hcur_, xsm ? xsm_us_ : nullptr}, st_);
  for (size_t k = 0; k < cap_layers_.size(); ++k)            
    if (cap_layers_[k] == l) cudaMemcpyAsync(cap_ + k * (size_t)a_.Mmax * c.hidden, hcur_, (size_t)M * c.hidden * 4, cudaMemcpyDeviceToDevice, st_);
  hcur_ = hcur_save;
}
void Engine::head_forward(int M, int* adv_pos, int* adv_tok, int adv_greedy) {
  const bool l_main = true;
  if (xsum_ && M != 1) throw std::runtime_error("XSUM stage 0: the prefill head GEMM is not converted");
  g_pdl_rows = M;
  const ModelDesc& c = m_.cfg;
  if (!res_fuse_ && !(fuse2_ && last_f2_) && xsm_mode_ != 3 && xsm_mode_ != 4) launch_norm_xform_a8(hcur_, m_.final_norm, c.eps, xsmx(XformA8Args{nullptr, m_.head.signs_in, m_.head.qm_in, c.hidden, m_.head.m_in, a_.qx_ff, a_.alpha_ff, M}, true), st_);
  if (M == 1) { GemvLaunch P = gemv_base(m_, 1, (int)m_.stage_bytes_head); add_seg(P, m_.head); P.qx = xsm_mode_ == 3 ? xsm_qx2_ : a_.qx_ff; P.alpha = xsm_mode_ == 3 ? xsm_alpha2_ : a_.alpha_ff; xsmg(P, l_main); launch_gemv(P, st_); }
  else {
    for (int k = 4; k <= 8; ++k) if (!m_.head.tiles_by_k[k].empty()) { TensorDev td = m_.head.dev; td.k_default = k; GemmLaunch Gm{td, m_.head.part, a_.qx_ff, a_.alpha_ff, M, 2 * 8 * 16 * k}; launch_gemm(Gm, st_); } }
  
  const size_t last = (size_t)(M - 1) * c.vocab;
  (void)last; { ArgmaxArgs AA{m_.head.part, m_.head.tau, c.vocab, a_.logits, a_.d_next, a_.blk_val, a_.blk_idx, m_.head.dev.n_workers, M, M - 1, 1, xsum_ ? xsm_us_ : nullptr, xsum_ ? xsm_rr_ : nullptr, c.eps};
    if (fuse2_) { AA.cnt = am_cnt_; AA.adv_pos = adv_pos; AA.adv_tok = adv_tok; AA.adv_greedy = adv_greedy; }
    launch_argmax(AA, st_);
    if (!fuse2_ && adv_pos) launch_advance(adv_pos, adv_tok, a_.d_next, adv_greedy, st_); }
}


void Engine::head_rows(int M) {
  const bool l_main = true;
  g_pdl_rows = M;
  const ModelDesc& c = m_.cfg; const bool HG = kmask_ & 4, HS = (kmask_ & 8) != 0;
  if ((kmask_ & 2) && !res_fuse_ && !(fuse2_ && last_f2_) && xsm_mode_ != 3 && xsm_mode_ != 4) launch_norm_xform_a8(hcur_, m_.final_norm, c.eps, xsmx(XformA8Args{nullptr, m_.head.signs_in, m_.head.qm_in, c.hidden, m_.head.m_in, a_.qx_ff, a_.alpha_ff, M}, true), st_);
  if (HG) { GemvLaunch P = gemv_base(m_, M, (int)m_.stage_bytes_head); add_seg(P, m_.head); P.qx = xsm_mode_ == 3 ? xsm_qx2_ : a_.qx_ff; P.alpha = xsm_mode_ == 3 ? xsm_alpha2_ : a_.alpha_ff; xsmg(P, l_main); launch_gemv(P, st_); }
  if (HS) {
    { ArgmaxArgs AA{m_.head.part, m_.head.tau, c.vocab, a_.logits, a_.d_next8, a_.blk_val, a_.blk_idx, m_.head.dev.n_workers, M, 0, M, xsum_ ? xsm_us_ : nullptr, xsum_ ? xsm_rr_ : nullptr, c.eps}; if (fuse2_) AA.cnt = am_cnt_; launch_argmax(AA, st_); }
    { TopkArgs TA{a_.logits, c.vocab, a_.tk_val, a_.tk_idx, a_.top_val, a_.top_idx, M}; if (fuse2_) TA.cnt = tk_cnt_; launch_topk8(TA, st_); }
  }
}
void Engine::verify_body(int M) { verify_body_part(M, 0, m_.cfg.layers); }

void Engine::pf_mark(int l, int j) { if (pf_active_ && l >= 0) launch_set_int(prog_, 4 * l + j, st_); }
void Engine::verify_body_part(int M, int l0, int l1) {
  g_pdl_rows = M;
  const ModelDesc& c = m_.cfg;
  const bool pf = pf_ > 0 && l0 == 0 && l1 == c.layers;          
  if (pf) { launch_set_int(prog_, -1, st_); cudaEventRecord(pev0_, st_); cudaStreamWaitEvent(pst_, pev0_, 0);
    launch_prefetch(pf_plan_, pf_n_, prog_, pf_ctas_, pst_); cudaEventRecord(pev1_, pst_); pf_active_ = true; }
  if (l0 == 0 && (kmask_ & 2)) { EmbArgs E{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, a_.d_toks, a_.h, M};
    emb_f2_ = fuse2_ && !xsum_; if (emb_f2_) { E.gamma = m_.layers[0].ln1; E.eps = c.eps; E.signs_in = m_.layers[0].q.signs_in; E.qx = a_.qx_ff; E.alpha = a_.alpha_ff; }
    launch_embedding(E, st_); }
  if (l0 == 0 && res_fuse_ && (kmask_ & 2)) input_norm(M);
  for (int l = l0; l < l1; ++l) layer_forward(l, M, true);
  if (l1 < c.layers) return;
  emb_f2_ = false;
  head_rows(M);
  if (pf) { pf_active_ = false; cudaStreamWaitEvent(st_, pev1_, 0); }
  if (hcur_ != a_.h) throw std::runtime_error("A4-beta: h double-buffer parity broken");
}
void Engine::verify(const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals, const int32_t* targets, float* out_nll) {
  if (M < 1 || M > 16) throw std::runtime_error("verify: M must be 1..16");
  if (pos_ + M > m_.max_ctx) throw std::runtime_error("verify: context overflow");
  std::memcpy(h_pin_, ids, (size_t)M * 4);
  cudaMemcpyAsync(a_.d_toks, h_pin_, (size_t)M * 4, cudaMemcpyHostToDevice, st_);
  launch_set_int(a_.d_pos, pos_, st_);
  if (use_graph) {
    const int key = M * 16 + 15; auto it = vgraph_.find(key);
    if (it == vgraph_.end()) {
      cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); verify_body(M); cudaStreamEndCapture(st_, &g);
      if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("verify: graph instantiate failed");
      cudaGraphDestroy(g); it = vgraph_.emplace(key, ge).first;
    }
    cudaGraphLaunch(it->second, st_);
  } else verify_body(M);
  if (targets) {
    std::memcpy(h_pin_ + 320, targets, (size_t)M * 4);
    cudaMemcpyAsync(a_.d_target8, h_pin_ + 320, (size_t)M * 4, cudaMemcpyHostToDevice, st_);
    launch_nll(NllArgs{a_.logits, m_.cfg.vocab, a_.d_target8, a_.d_nll8, a_.blk_max, a_.blk_sum, M}, st_);
    cudaMemcpyAsync(h_pinf_ + 256, a_.d_nll8, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  }
  cudaMemcpyAsync(h_pin_ + 32, a_.d_next8, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  if (top_ids) cudaMemcpyAsync(h_pin_ + 64, a_.top_idx, (size_t)M * 8 * 4, cudaMemcpyDeviceToHost, st_);
  if (top_vals) cudaMemcpyAsync(h_pinf_, a_.top_val, (size_t)M * 8 * 4, cudaMemcpyDeviceToHost, st_);
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("verify: ") + cudaGetErrorString(e));
  if (out_argmax) std::memcpy(out_argmax, h_pin_ + 32, (size_t)M * 4);
  if (top_ids) std::memcpy(top_ids, h_pin_ + 64, (size_t)M * 8 * 4);
  if (top_vals) std::memcpy(top_vals, h_pinf_, (size_t)M * 8 * 4);
  if (out_nll) std::memcpy(out_nll, h_pinf_ + 256, (size_t)M * 4);
  last_M_ = M;
}
void Engine::verify_rows(int M, const int32_t* rows, int n_host, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids, float* top_vals, bool hist) {
  if (M < 1 || M > 16 || n_host < 0 || n_host > M) throw std::runtime_error("verify_rows: M must be 1..16, 0 <= n_host <= M");
  if (pos_ + M > m_.max_ctx) throw std::runtime_error("verify_rows: context overflow");
  if (hist && !(dm_.on && dm_.hist && cap_)) throw std::runtime_error("verify_rows: hist needs the D3 history (lazy draft)");
  h_vin_[0] = pos_; if (n_host) std::memcpy(h_vin_ + 1, rows, (size_t)n_host * 4);
  const size_t H = m_.cfg.hidden;
  
  
  
  static const int split = [] { const char* e = std::getenv("YSKN_VR_SPLIT"); return e && e[0] ? std::atoi(e) : 0; }();
  const int sl = std::min(split, m_.cfg.layers);
  auto part1 = [&]() {
    g_pdl_rows = M;
    cudaMemcpyAsync(d_vin_, h_vin_, 17 * 4, cudaMemcpyHostToDevice, st_);
    launch_verify_prologue(d_vin_, a_.d_pos, a_.d_toks, n_host, st_);
    verify_body_part(M, 0, sl);
  };
  auto part2 = [&]() {
    g_pdl_rows = M;
    verify_body_part(M, sl, m_.cfg.layers);
    if (hist) launch_hist_spec(cap_, dm_.hist, a_.d_pos, M, (int)H, m_.max_ctx, (size_t)a_.Mmax * H, (size_t)m_.max_ctx * H, st_);
    cudaMemcpyAsync(h_pin_ + 32, a_.d_next8, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
    cudaMemcpyAsync(h_pin_, a_.d_toks, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
    if (top_ids) cudaMemcpyAsync(h_pin_ + 64, a_.top_idx, (size_t)M * 8 * 4, cudaMemcpyDeviceToHost, st_);
    if (top_vals) cudaMemcpyAsync(h_pinf_, a_.top_val, (size_t)M * 8 * 4, cudaMemcpyDeviceToHost, st_);
  };
  if (use_graph) {
    const int key = ((M * 17 + n_host) * 8) + (top_ids ? 4 : 0) + (top_vals ? 2 : 0) + (hist ? 1 : 0);
    auto capture = [&](int k, auto&& fn) {
      auto it = vrgraph_.find(k);
      if (it == vrgraph_.end()) {
        cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); fn(); cudaStreamEndCapture(st_, &g);
        if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("verify_rows: graph instantiate failed");
        cudaGraphDestroy(g); it = vrgraph_.emplace(k, ge).first;
      }
      return it->second;
    };
    if (sl > 0) { const cudaGraphExec_t g1 = capture(key * 2, part1), g2 = capture(key * 2 + 1, part2); cudaGraphLaunch(g1, st_); cudaGraphLaunch(g2, st_); }
    else { const cudaGraphExec_t g = capture(key * 2 + 1 + (1 << 24), [&]() { part1(); part2(); }); cudaGraphLaunch(g, st_); }
  } else { part1(); part2(); }
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("verify_rows: ") + cudaGetErrorString(e));
  if (out_argmax) std::memcpy(out_argmax, h_pin_ + 32, (size_t)M * 4);
  if (out_toks) std::memcpy(out_toks, h_pin_, (size_t)M * 4);
  if (top_ids) std::memcpy(top_ids, h_pin_ + 64, (size_t)M * 8 * 4);
  if (top_vals) std::memcpy(top_vals, h_pinf_, (size_t)M * 8 * 4);
  last_M_ = M;
}
void Engine::commit(int n_accept) {
  if (n_accept < 0 || n_accept >= last_M_) throw std::runtime_error("commit: n_accept out of range of the last verify pass");
  pos_ += n_accept + 1; launch_set_int(a_.d_pos, pos_, st_);
  cudaMemcpyAsync(a_.d_tok, a_.d_next8 + n_accept, 4, cudaMemcpyDeviceToDevice, st_);   
  last_M_ = 0;
}
int Engine::prefill_rows(const int32_t* ids, int n, int32_t* out_argmax, int32_t* top_ids, float* top_vals) {
  if (pos_ + n > m_.max_ctx) throw std::runtime_error("context overflow");
  if (pfa_ && (!xsum_ || pfa_xsm()) && !res_fuse_ && (lno_ == 1 || lno_ == 2 || lno_ == 4)) return prefill_rows_pfa(ids, n, out_argmax, top_ids, top_vals);
  const ModelDesc& c = m_.cfg;
  const bool xsmp = xsum_ && xsm_mode_ == 3;              
  for (int done = 0; done < n;) {
    const int M = std::min(xsmp ? 16 : 32, n - done);
    launch_set_int(a_.d_pos, pos_, st_);
    for (int i = 0; i < M; ++i) { launch_set_int(a_.d_tok, ids[done + i], st_);
      launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, a_.d_tok, a_.h + (size_t)i * c.hidden}, st_); }
    if (res_fuse_) input_norm(M);
    for (int l = 0; l < c.layers; ++l) layer_forward(l, M, xsmp);
    if (xsmp) { head_rows(M); cudaMemcpyAsync(a_.d_next, a_.d_next8 + M - 1, 4, cudaMemcpyDeviceToDevice, st_); }   
    else if (xsum_) { xsm_head_gemm(M, true); cudaMemcpyAsync(a_.d_next, a_.d_next8 + M - 1, 4, cudaMemcpyDeviceToDevice, st_); }
    else {
    head_forward(M);
    if (hcur_ != a_.h) throw std::runtime_error("A4-beta: h double-buffer parity broken");
    launch_argmax(ArgmaxArgs{m_.head.part, m_.head.tau, c.vocab, a_.logits, a_.d_next8, a_.blk_val, a_.blk_idx, m_.head.dev.n_workers, M, 0, M}, st_);
    launch_topk8(TopkArgs{a_.logits, c.vocab, a_.tk_val, a_.tk_idx, a_.top_val, a_.top_idx, M}, st_); }
    if (dm_.on) draft_prefill_chunk(ids + done, M);             
    
    cudaMemcpyAsync(pr_am_ + done, a_.d_next8, (size_t)M * 4, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(pr_ti_ + (size_t)done * 8, a_.top_idx, (size_t)M * 32, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(pr_tv_ + (size_t)done * 8, a_.top_val, (size_t)M * 32, cudaMemcpyDeviceToDevice, st_);
    pos_ += M; done += M; launch_set_int(a_.d_pos, pos_, st_);
  }
  if (out_argmax) cudaMemcpyAsync(out_argmax, pr_am_, (size_t)n * 4, cudaMemcpyDeviceToHost, st_);
  if (top_ids) cudaMemcpyAsync(top_ids, pr_ti_, (size_t)n * 32, cudaMemcpyDeviceToHost, st_);
  if (top_vals) cudaMemcpyAsync(top_vals, pr_tv_, (size_t)n * 32, cudaMemcpyDeviceToHost, st_);
  int next = 0; cudaMemcpyAsync(&next, a_.d_next, 4, cudaMemcpyDeviceToHost, st_); cudaStreamSynchronize(st_);
  launch_set_int(a_.d_tok, next, st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_rows: ") + cudaGetErrorString(e));
  return next;
}
void Engine::hidden_rows(int n, float* out) { cudaStreamSynchronize(st_); cudaMemcpy(out, hcur_, (size_t)n * m_.cfg.hidden * 4, cudaMemcpyDeviceToHost); }
void Engine::logits_rows(int row0, int n, float* out) { cudaStreamSynchronize(st_); cudaMemcpy(out, a_.logits + (size_t)row0 * m_.cfg.vocab, (size_t)n * m_.cfg.vocab * 4, cudaMemcpyDeviceToHost); }
void Engine::set_capture(const int* layers, int n) {
  DevMemScope dm_scope(this);
  if (cap_) { cudaFree(cap_); cap_ = nullptr; }
  cap_layers_.assign(layers, layers + n);
  if (n > 0) { cudaMalloc(&cap_, (size_t)n * a_.Mmax * m_.cfg.hidden * 4); cudaMemset(cap_, 0, (size_t)n * a_.Mmax * m_.cfg.hidden * 4); }
  for (auto& kv : vgraph_) cudaGraphExecDestroy(kv.second);   
  vgraph_.clear();
}
void Engine::capture_rows(int k, int n, float* out) {
  if (k < 0 || k >= (int)cap_layers_.size()) throw std::runtime_error("capture_rows: no such capture slot");
  cudaStreamSynchronize(st_); cudaMemcpy(out, cap_ + (size_t)k * a_.Mmax * m_.cfg.hidden, (size_t)n * m_.cfg.hidden * 4, cudaMemcpyDeviceToHost);
}
float Engine::time_verify(int M, int mask, int iters) {
  if (M < 1 || M > 16) throw std::runtime_error("time_verify: M must be 1..16");
  if (pos_ + M > m_.max_ctx) throw std::runtime_error("time_verify: context overflow");
  launch_set_int(a_.d_pos, pos_, st_);
  for (int i = 0; i < M; ++i) h_pin_[i] = 0;
  cudaMemcpyAsync(a_.d_toks, h_pin_, (size_t)M * 4, cudaMemcpyHostToDevice, st_);
  const int key = (mask & 16) ? 100000 + M * 64 + mask : M * 16 + mask; auto it = vgraph_.find(key);   
  if (it == vgraph_.end()) {
    kmask_ = mask; cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); verify_body(M); cudaStreamEndCapture(st_, &g);
    kmask_ = 15;
    if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("time_verify: graph instantiate failed");
    cudaGraphDestroy(g); it = vgraph_.emplace(key, ge).first;
  }
  for (int i = 0; i < 3; ++i) cudaGraphLaunch(it->second, st_);                        
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st_); for (int i = 0; i < iters; ++i) cudaGraphLaunch(it->second, st_); cudaEventRecord(b, st_); cudaEventSynchronize(b);
  float ms = 0.f; cudaEventElapsedTime(&ms, a, b); cudaEventDestroy(a); cudaEventDestroy(b);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("time_verify: ") + cudaGetErrorString(e));
  return ms / (float)iters;
}
void Engine::step_body(bool greedy) {
  g_pdl_rows = 1;
  const ModelDesc& c = m_.cfg;
  { EmbArgs E{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, a_.d_tok, a_.h};
    emb_f2_ = fuse2_ && !xsum_; if (emb_f2_) { E.gamma = m_.layers[0].ln1; E.eps = c.eps; E.signs_in = m_.layers[0].q.signs_in; E.qx = a_.qx_ff; E.alpha = a_.alpha_ff; }
    launch_embedding(E, st_); }
  if (res_fuse_) input_norm(1);
  for (int l = 0; l < c.layers; ++l) layer_forward(l, 1, true);
  emb_f2_ = false;
  head_forward(1, a_.d_pos, a_.d_tok, greedy ? 1 : 0);                   
  if (hcur_ != a_.h) throw std::runtime_error("A4-beta: h double-buffer parity broken");
}
void Engine::ensure_graph() {
  if (graph_ready_) return;
  std::vector<int> keep; keep.swap(cap_layers_);            
  cudaGraph_t g; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); step_body(true); cudaStreamEndCapture(st_, &g);
  keep.swap(cap_layers_);
  cudaGraphInstantiate(&gexec_, g, 0); cudaGraphDestroy(g); graph_ready_ = true;
}
int Engine::prefill(const int32_t* ids, int n) {
  if (pos_ + n > m_.max_ctx) throw std::runtime_error("context overflow");
  if (pfa_ && (!xsum_ || pfa_xsm()) && !dm_.on && !res_fuse_ && (lno_ == 1 || lno_ == 2 || lno_ == 4)) return prefill_pfa(ids, n);
  const ModelDesc& c = m_.cfg;
  const bool xsmp = xsum_ && xsm_mode_ == 3;              
  for (int done = 0; done < n;) {
    const int M = std::min(xsmp ? 16 : 32, n - done);
    launch_set_int(a_.d_pos, pos_, st_);
    for (int i = 0; i < M; ++i) { launch_set_int(a_.d_tok, ids[done + i], st_);   
      launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, a_.d_tok, a_.h + (size_t)i * c.hidden}, st_); }
    if (res_fuse_) input_norm(M);
    for (int l = 0; l < c.layers; ++l) layer_forward(l, M, xsmp);
    if (xsmp) { head_rows(M); cudaMemcpyAsync(a_.d_next, a_.d_next8 + M - 1, 4, cudaMemcpyDeviceToDevice, st_);   
      if (M > 1) cudaMemcpyAsync(a_.logits, a_.logits + (size_t)(M - 1) * c.vocab, (size_t)c.vocab * 4, cudaMemcpyDeviceToDevice, st_); }   
    else if (xsum_) { xsm_head_gemm(M, false); cudaMemcpyAsync(a_.d_next, a_.d_next8 + M - 1, 4, cudaMemcpyDeviceToDevice, st_);
      if (M > 1) cudaMemcpyAsync(a_.logits, a_.logits + (size_t)(M - 1) * c.vocab, (size_t)c.vocab * 4, cudaMemcpyDeviceToDevice, st_); }
    else head_forward(M);
    if (hcur_ != a_.h) throw std::runtime_error("A4-beta: h double-buffer parity broken");
    pos_ += M; done += M; launch_set_int(a_.d_pos, pos_, st_);
  }
  int next = 0; cudaMemcpyAsync(&next, a_.d_next, 4, cudaMemcpyDeviceToHost, st_); cudaStreamSynchronize(st_);
  launch_set_int(a_.d_tok, next, st_);
  return next;
}
void Engine::decode(int32_t* out, int steps) {
  ensure_graph();
  for (int i = 0; i < steps; ++i) {
    if (pos_ + 1 > m_.max_ctx) throw std::runtime_error("context overflow");
    cudaGraphLaunch(gexec_, st_); pos_++;
    cudaMemcpyAsync(h_dec_ + i, a_.d_next, 4, cudaMemcpyDeviceToHost, st_);  
  }
  cudaStreamSynchronize(st_); std::memcpy(out, h_dec_, (size_t)steps * 4);
}
void Engine::nll(const int32_t* ids, int n, float* out) {
  
  cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); step_body(false);
  launch_nll(NllArgs{a_.logits, m_.cfg.vocab, a_.d_target, a_.d_nll, a_.blk_max, a_.blk_sum}, st_); cudaStreamEndCapture(st_, &g); cudaGraphInstantiate(&ge, g, 0); cudaGraphDestroy(g);
  for (int i = 0; i + 1 < n; ++i) {
    cudaMemcpyAsync(a_.d_tok, ids + i, 4, cudaMemcpyHostToDevice, st_); cudaMemcpyAsync(a_.d_target, ids + i + 1, 4, cudaMemcpyHostToDevice, st_);
    cudaGraphLaunch(ge, st_); pos_++;
    cudaMemcpyAsync(out + i, a_.d_nll, 4, cudaMemcpyDeviceToHost, st_);
  }
  cudaStreamSynchronize(st_); cudaGraphExecDestroy(ge);
}
void Engine::debug_hidden(const int32_t* ids, int n, int upto, int mode, float* out) {
  const ModelDesc& c = m_.cfg; reset();
  auto embed = [&](int i, float* dst) {
    launch_set_int(a_.d_tok, ids[i], st_);
    launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, a_.d_tok, dst}, st_);
  };
  if (mode == 0) {
    if (n < 1 || n > 32) throw std::runtime_error("debug_hidden: prefill mode needs 1..32 tokens");
    launch_set_int(a_.d_pos, 0, st_);
    for (int i = 0; i < n; ++i) embed(i, a_.h + (size_t)i * c.hidden);
    if (res_fuse_) input_norm(n);
    for (int l = 0; l < upto; ++l) layer_forward(l, n, false);
    cudaMemcpyAsync(out, a_.h, (size_t)n * c.hidden * 4, cudaMemcpyDeviceToHost, st_);
  } else {
    for (int i = 0; i < n; ++i) {
      launch_set_int(a_.d_pos, i, st_); embed(i, a_.h);
      if (res_fuse_) input_norm(1);
      for (int l = 0; l < upto; ++l) layer_forward(l, 1, true);
    }
    cudaMemcpyAsync(out, a_.h, (size_t)c.hidden * 4, cudaMemcpyDeviceToHost, st_);
  }
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("debug_hidden: ") + cudaGetErrorString(e));
}
void Engine::debug_buffer(int which, int M, uint16_t* out) {
  const ModelDesc& c = m_.cfg; const uint16_t* src = which == 0 ? a_.q16 : (which == 1 ? a_.o16 : a_.xff);
  const size_t per = which == 2 ? (size_t)c.ffn : (size_t)c.hidden;
  cudaMemcpy(out, src, (size_t)M * per * 2, cudaMemcpyDeviceToHost);
}
void Engine::logits(float* out) { cudaMemcpy(out, a_.logits, (size_t)m_.cfg.vocab * 4, cudaMemcpyDeviceToHost); }
}

namespace yskn {
static uint16_t* dev_vec16(const Container& c, const std::string& name, int n) {
  const VectorDesc& v = c.vector(name); if (v.n != n) throw std::runtime_error("draft vector size mismatch: " + name);
  uint16_t* d = nullptr; cudaMalloc(&d, (size_t)n * 2); cudaMemcpy(d, c.at<uint16_t>(v.off), (size_t)n * 2, cudaMemcpyHostToDevice); return d;
}
void Engine::load_draft(const std::string& path, const std::string& head_ids_path) {
  DevMemScope dm_scope(this);
  if (res_fuse_) throw std::runtime_error("load_draft: the draft path needs the residual fusion off (YSKN_RESFUSE unset)");
  const ModelDesc& c = m_.cfg; const int H = c.hidden, want = m_.sm_count * 8; dm_.c.open(path);
  auto up = [&](const std::string& n, int force = 0) { return m_.upload_linear(dm_.c.tensor(n), a_.Mmax, want, force, &dm_.c); };
  dm_.fuse = up("d3.fuse"); dm_.fc_in = up("d3.fc_in");
  dm_.L.q = up("d3.q_proj"); dm_.L.k = up("d3.k_proj"); dm_.L.v = up("d3.v_proj"); dm_.L.o = up("d3.o_proj");
  dm_.L.gate = up("d3.gate_proj"); dm_.L.up = up("d3.up_proj"); dm_.L.down = up("d3.down_proj");
  dm_.head = up("d3.head", dm_.c.tensor("d3.head").cols / 512); dm_.head_rows = dm_.head.dev.N;
  int kmax = 4; for (DevTensor* t : {&dm_.fuse, &dm_.fc_in, &dm_.L.q, &dm_.L.k, &dm_.L.v, &dm_.L.o, &dm_.L.gate, &dm_.L.up, &dm_.L.down}) kmax = std::max(kmax, t->kmax);
  dm_.stage_bytes = GEMV_NT_PASS * 8 * 16 * kmax;
  for (int k = 0; k < 3; ++k) dm_.nf[k] = dev_vec16(dm_.c, "d3.nf" + std::to_string(k), H);
  dm_.n_feat = dev_vec16(dm_.c, "d3.n_feat", H); dm_.n_emb = dev_vec16(dm_.c, "d3.n_emb", H); dm_.norm = dev_vec16(dm_.c, "d3.norm", H);
  dm_.L.ln1 = dev_vec16(dm_.c, "d3.ln1", H); dm_.L.ln2 = dev_vec16(dm_.c, "d3.ln2", H);
  dm_.L.bq = dev_vec16(dm_.c, "d3.q_bias", c.heads * c.head_dim); dm_.L.bk = dev_vec16(dm_.c, "d3.k_bias", c.kv_heads * c.head_dim);
  dm_.L.bv = dev_vec16(dm_.c, "d3.v_bias", c.kv_heads * c.head_dim);
  const int groups = m_.max_ctx / 32; KvLayer& kv = dm_.L.kv;
  kv.kq = dz<uint8_t>((size_t)4 * groups * 32 * 64); kv.kscale = dz<uint16_t>((size_t)4 * groups * 128 * 2); kv.kzero = dz<uint16_t>((size_t)4 * groups * 128 * 2);
  kv.kshadow = dz<uint16_t>((size_t)4 * m_.max_ctx * 128 * 2); kv.vq = dz<uint8_t>((size_t)4 * m_.max_ctx * 64); kv.vscale = dz<uint16_t>((size_t)4 * m_.max_ctx * 2); kv.vzero = dz<uint16_t>((size_t)4 * m_.max_ctx * 2);
  std::vector<int32_t> ids(dm_.head_rows);
  { FILE* f = std::fopen(head_ids_path.c_str(), "rb"); if (!f) throw std::runtime_error("load_draft: cannot open " + head_ids_path);
    const size_t got = std::fread(ids.data(), 4, ids.size(), f); std::fclose(f); if (got != ids.size()) throw std::runtime_error("load_draft: head id table too short"); }
  cudaMalloc(&dm_.head_ids, ids.size() * 4); cudaMemcpy(dm_.head_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
  dm_.feat = dz<float>((size_t)3 * a_.Mmax * H * 4); dm_.last = dz<float>((size_t)3 * H * 4); dm_.g = dz<float>((size_t)a_.Mmax * H * 4);
  dm_.emb = dz<float>((size_t)a_.Mmax * H * 4); dm_.hd = dz<float>((size_t)a_.Mmax * H * 4); dm_.aprev = dz<float>((size_t)H * 4);
  dm_.logits = dz<float>((size_t)dm_.head_rows * 4); dm_.toks = dz<int>((size_t)a_.Mmax * 4); dm_.d_pos = dz<int>(4); dm_.cur_tok = dz<int>(4); dm_.idx = dz<int>(4);
  dm_.hist = dz<float>((size_t)3 * m_.max_ctx * H * 4);
  if (!h_toks_) cudaMallocHost(&h_toks_, (size_t)m_.max_ctx * 4);
  const int cap[3] = {2, 14, 27}; set_capture(cap, 3);
  for (auto& kv2 : dgraph_) cudaGraphExecDestroy(kv2.second); dgraph_.clear();
  if (lno_ == 1) lno_build({&dm_.fuse, &dm_.fc_in, &dm_.L.q, &dm_.L.k, &dm_.L.v, &dm_.L.o, &dm_.L.gate, &dm_.L.up, &dm_.L.down, &dm_.head});   
  dm_.on = true; cudaDeviceSynchronize();
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("load_draft: ") + cudaGetErrorString(e));
}
static void d3_linear(const Model& m, DevTensor& t, int M, bool fused, int sb, int8_t* qx, float* alpha, cudaStream_t st) {
  if (fused) { GemvLaunch P = gemv_base(m, M, sb); add_seg(P, t); P.qx = qx; P.alpha = alpha; launch_gemv(P, st); }
  else { GemmLaunch Gm{t.dev, t.part, qx, alpha, M, 2 * 8 * 16 * t.bits}; launch_gemm(Gm, st); }
}

void Engine::draft_rows(int M, bool fused, const float* feat_src) {
  const ModelDesc& c = m_.cfg; const int H = c.hidden;
  launch_d3_prep(D3PrepArgs{{feat_src, dm_.emb, nullptr}, {H, H, 0}, {dm_.n_feat, dm_.n_emb, nullptr}, 2, c.eps, dm_.fc_in.signs_in, a_.qx_ff, a_.alpha_ff, M}, st_);
  d3_linear(m_, dm_.fc_in, M, fused, dm_.stage_bytes, a_.qx_ff, a_.alpha_ff, st_);
  launch_d3_linear_out(D3LinOutArgs{dm_.fc_in.part, dm_.fc_in.tau, dm_.fc_in.dev.n_workers, dm_.fc_in.signs_out, M, H, dm_.hd, H}, st_);
  layer_forward(-1, M, fused);
}

void Engine::draft_fused_rows(int M, bool fused) {
  const ModelDesc& c = m_.cfg; const int H = c.hidden; const size_t slot = (size_t)a_.Mmax * H;
  launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, dm_.toks, dm_.emb, M}, st_);
  launch_d3_prep(D3PrepArgs{{dm_.feat, dm_.feat + slot, dm_.feat + 2 * slot}, {H, H, H}, {dm_.nf[0], dm_.nf[1], dm_.nf[2]}, 3, c.eps, dm_.fuse.signs_in, a_.qx_ff, a_.alpha_ff, M}, st_);
  d3_linear(m_, dm_.fuse, M, fused, dm_.stage_bytes, a_.qx_ff, a_.alpha_ff, st_);
  launch_d3_linear_out(D3LinOutArgs{dm_.fuse.part, dm_.fuse.tau, dm_.fuse.dev.n_workers, dm_.fuse.signs_out, M, H, dm_.g, H}, st_);
  draft_rows(M, fused, dm_.g);
}


void Engine::draft_stage(int M, int last_row) {
  const int H = m_.cfg.hidden; const size_t slot = (size_t)a_.Mmax * H;
  for (int k = 0; k < 3; ++k) {
    cudaMemcpyAsync(dm_.feat + k * slot, dm_.last + (size_t)k * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, st_);
    if (M > 1) cudaMemcpyAsync(dm_.feat + k * slot + H, cap_ + k * slot, (size_t)(M - 1) * H * 4, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(dm_.last + (size_t)k * H, cap_ + k * slot + (size_t)last_row * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, st_);
  }
}
void Engine::draft_prefill_chunk(const int32_t* ids, int M) {
  launch_set_int(dm_.d_pos, pos_, st_);
  std::memcpy(h_pin_, ids, (size_t)M * 4); cudaMemcpyAsync(dm_.toks, h_pin_, (size_t)M * 4, cudaMemcpyHostToDevice, st_);
  draft_stage(M, M - 1);
  hist_store(pos_, M);                                            
  draft_fused_rows(M, false);                                     
  cudaStreamSynchronize(st_);                                     
}
void Engine::draft_body(int mode, int a, int steps) {
  int Mc;
  if (mode == 0) {      
    Mc = 1; const int H = m_.cfg.hidden; const size_t slot = (size_t)a_.Mmax * H;
    for (int k = 0; k < 3; ++k) cudaMemcpyAsync(dm_.feat + k * slot, dm_.last + (size_t)k * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(dm_.toks, a_.d_toks, 4, cudaMemcpyDeviceToDevice, st_);
    launch_d3_set_from(dm_.d_pos, a_.d_pos, 0, st_);
  } else {              
    Mc = a + 2;
    draft_stage(a + 2, a);                                        
    launch_d3_gather(dm_.toks, a_.d_toks, a_.d_next8, a, st_);    
    launch_d3_set_from(dm_.d_pos, a_.d_pos, -(a + 1), st_);
  }
  draft_fused_rows(Mc, true);
  draft_head(Mc - 1, 1);
  for (int k = 2; k <= steps; ++k) {                              
    launch_d3_set_from(dm_.d_pos, a_.d_pos, k - 1, st_);
    const ModelDesc& c = m_.cfg;
    launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, dm_.cur_tok, dm_.emb, 1}, st_);
    draft_rows(1, true, dm_.aprev);
    draft_head(0, k);
  }
}
void Engine::draft_round(int mode, int a, int steps, int use_graph) {
  if (!dm_.on) throw std::runtime_error("draft_round: no draft loaded");
  if (steps < 1 || steps > 15 || a < 0 || a + 2 > 16) throw std::runtime_error("draft_round: bad a/steps");
  if (pos_ + steps > m_.max_ctx) throw std::runtime_error("draft_round: context overflow");
  if (use_graph) {
    const int key = mode * 10000 + a * 100 + steps; auto it = dgraph_.find(key);
    if (it == dgraph_.end()) {
      cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); draft_body(mode, a, steps); cudaStreamEndCapture(st_, &g);
      if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("draft_round: graph instantiate failed");
      cudaGraphDestroy(g); it = dgraph_.emplace(key, ge).first;
    }
    cudaGraphLaunch(it->second, st_);
  } else draft_body(mode, a, steps);
}
float Engine::time_draft(int mode, int a, int steps, int iters) {
  draft_round(mode, a, steps, 1); cudaStreamSynchronize(st_);              
  const int key = mode * 10000 + a * 100 + steps; cudaGraphExec_t ge = dgraph_.at(key);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st_); for (int i = 0; i < iters; ++i) cudaGraphLaunch(ge, st_); cudaEventRecord(e1, st_); cudaEventSynchronize(e1);
  float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / (float)iters;
}
void Engine::draft_debug(int which, int n, float* out) {
  const size_t H = m_.cfg.hidden; cudaStreamSynchronize(st_);
  const float* src = which == 0 ? dm_.g : which == 1 ? dm_.hd : which == 2 ? dm_.emb : which == 3 ? dm_.feat : which == 4 ? dm_.last : dm_.aprev;
  const size_t cnt = which == 3 ? 3 * (size_t)a_.Mmax * H : which == 4 ? 3 * H : which == 5 ? H : (size_t)n * H;
  cudaMemcpy(out, src, cnt * 4, cudaMemcpyDeviceToHost);
}
void Engine::hist_store(int pos0, int n) {
  if (!dm_.on || n <= 0) return;
  if (pos0 < 0 || pos0 + n > m_.max_ctx || n > a_.Mmax) throw std::runtime_error("hist_store: rows out of range");
  const size_t H = m_.cfg.hidden, slot = (size_t)a_.Mmax * H, hs = (size_t)m_.max_ctx * H;
  for (int k = 0; k < 3; ++k) cudaMemcpyAsync(dm_.hist + k * hs + (size_t)pos0 * H, cap_ + k * slot, (size_t)n * H * 4, cudaMemcpyDeviceToDevice, st_);
}
void Engine::draft_chunk_body(int m, bool head) {
  draft_fused_rows(m, true);                                      
  if (head) draft_head(m - 1, 1);                                 
}
void Engine::draft_steps_body(int steps) {
  const ModelDesc& c = m_.cfg;
  for (int k = 2; k <= steps; ++k) {
    launch_d3_set_from(dm_.d_pos, a_.d_pos, k - 1, st_);
    launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in, dm_.cur_tok, dm_.emb, 1}, st_);
    draft_rows(1, true, dm_.aprev);
    draft_head(0, k);
  }
}
void Engine::draft_catchup(int p0, int n, const int32_t* toks, int use_graph) {
  if (!dm_.on) throw std::runtime_error("draft_catchup: no draft loaded");
  if (n < 1 || p0 < 0 || p0 + n > m_.max_ctx) throw std::runtime_error("draft_catchup: rows out of range");
  const size_t H = m_.cfg.hidden, slot = (size_t)a_.Mmax * H, hs = (size_t)m_.max_ctx * H;
  std::memcpy(h_toks_ + p0, toks, (size_t)n * 4);                 
  for (int q0 = p0; q0 < p0 + n; q0 += 16) {
    const int m = std::min(16, p0 + n - q0); const bool head = q0 + m == p0 + n;
    for (int k = 0; k < 3; ++k) {                                   
      if (q0 == 0) { cudaMemsetAsync(dm_.feat + k * slot, 0, H * 4, st_);
        if (m > 1) cudaMemcpyAsync(dm_.feat + k * slot + H, dm_.hist + k * hs, (size_t)(m - 1) * H * 4, cudaMemcpyDeviceToDevice, st_); }
      else cudaMemcpyAsync(dm_.feat + k * slot, dm_.hist + k * hs + (size_t)(q0 - 1) * H, (size_t)m * H * 4, cudaMemcpyDeviceToDevice, st_);
    }
    cudaMemcpyAsync(dm_.toks, h_toks_ + q0, (size_t)m * 4, cudaMemcpyHostToDevice, st_);
    launch_set_int(dm_.d_pos, q0, st_);
    if (use_graph) {
      const int key = m * 2 + (head ? 1 : 0); auto it = cgraph_.find(key);
      if (it == cgraph_.end()) {
        cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); draft_chunk_body(m, head); cudaStreamEndCapture(st_, &g);
        if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("draft_catchup: graph instantiate failed");
        cudaGraphDestroy(g); it = cgraph_.emplace(key, ge).first;
      }
      cudaGraphLaunch(it->second, st_);
    } else draft_chunk_body(m, head);
  }
}
void Engine::draft_steps(int steps, int use_graph) {
  if (steps < 2) return;
  if (pos_ + steps > m_.max_ctx) throw std::runtime_error("draft_steps: context overflow");
  if (use_graph) {
    auto it = sgraph_.find(steps);
    if (it == sgraph_.end()) {
      cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); draft_steps_body(steps); cudaStreamEndCapture(st_, &g);
      if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("draft_steps: graph instantiate failed");
      cudaGraphDestroy(g); it = sgraph_.emplace(steps, ge).first;
    }
    cudaGraphLaunch(it->second, st_);
  } else draft_steps_body(steps);
}
void Engine::verify_vals(int M, float* top_vals) {
  cudaMemcpy(top_vals, a_.top_val, (size_t)M * 8 * 4, cudaMemcpyDeviceToHost);
}
int32_t Engine::get_verify_token(int row) {
  if (row < 0 || row >= 16) throw std::runtime_error("get_verify_token: row out of range");
  cudaMemcpyAsync(h_pin_ + 320, a_.d_toks + row, 4, cudaMemcpyDeviceToHost, st_); cudaStreamSynchronize(st_);
  return h_pin_[320];
}
void Engine::set_verify_tokens(const int32_t* ids, int first_row, int n) {
  if (n <= 0) return;
  if (first_row < 0 || first_row + n > 16) throw std::runtime_error("set_verify_tokens: rows out of range");
  
  
  std::memcpy(h_pin_ + first_row, ids, (size_t)n * 4);
  cudaMemcpyAsync(a_.d_toks + first_row, h_pin_ + first_row, (size_t)n * 4, cudaMemcpyHostToDevice, st_);
}
void Engine::verify_device(int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids, float* top_vals) {
  if (M < 1 || M > 16) throw std::runtime_error("verify_device: M must be 1..16");
  if (pos_ + M > m_.max_ctx) throw std::runtime_error("verify_device: context overflow");
  launch_set_int(a_.d_pos, pos_, st_);
  if (use_graph) {
    const int key = M * 16 + 15; auto it = vgraph_.find(key);
    if (it == vgraph_.end()) {
      cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); verify_body(M); cudaStreamEndCapture(st_, &g);
      if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("verify_device: graph instantiate failed");
      cudaGraphDestroy(g); it = vgraph_.emplace(key, ge).first;
    }
    cudaGraphLaunch(it->second, st_);
  } else verify_body(M);
  cudaMemcpyAsync(h_pin_ + 32, a_.d_next8, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  cudaMemcpyAsync(h_pin_, a_.d_toks, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  if (top_ids) cudaMemcpyAsync(h_pin_ + 64, a_.top_idx, (size_t)M * 8 * 4, cudaMemcpyDeviceToHost, st_);
  if (top_vals) cudaMemcpyAsync(h_pinf_, a_.top_val, (size_t)M * 8 * 4, cudaMemcpyDeviceToHost, st_);
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("verify_device: ") + cudaGetErrorString(e));
  if (out_argmax) std::memcpy(out_argmax, h_pin_ + 32, (size_t)M * 4);
  if (out_toks) std::memcpy(out_toks, h_pin_, (size_t)M * 4);
  if (top_ids) std::memcpy(top_ids, h_pin_ + 64, (size_t)M * 8 * 4);
  if (top_vals) std::memcpy(top_vals, h_pinf_, (size_t)M * 8 * 4);
  last_M_ = M;
}
void Engine::draft_head(int row, int k) {
  const ModelDesc& c = m_.cfg; const int H = c.hidden;
  launch_d3_prep(D3PrepArgs{{dm_.hd + (size_t)row * H, nullptr, nullptr}, {H, 0, 0}, {dm_.norm, nullptr, nullptr}, 1, c.eps, dm_.head.signs_in, a_.qx_ff, a_.alpha_ff, 1}, st_);
  { GemvLaunch P = gemv_base(m_, 1, GEMV_NT_PASS * 8 * 16 * dm_.head.kmax); add_seg(P, dm_.head); P.qx = a_.qx_ff; P.alpha = a_.alpha_ff; launch_gemv(P, st_); }
  launch_argmax(ArgmaxArgs{dm_.head.part, dm_.head.tau, dm_.head_rows, dm_.logits, dm_.idx, a_.blk_val, a_.blk_idx, dm_.head.dev.n_workers, 1, 0, 1}, st_);
  launch_d3_map(dm_.idx, dm_.head_ids, a_.d_toks + k, dm_.cur_tok, st_);
  cudaMemcpyAsync(dm_.aprev, dm_.hd + (size_t)row * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, st_);
}
}

namespace yskn {
void Engine::pfa_alloc() {
  if (tp_h_) return;
  const ModelDesc& c = m_.cfg; const size_t P = (size_t)tpm_;
  tp_h_ = dz<float>(P * c.hidden * 4); tp_q16_ = dz<uint16_t>(P * c.heads * 128 * 2); tp_xff_ = dz<uint16_t>(P * c.ffn * 2);
  tp_qx_ = dz<int8_t>(P * c.ffn); tp_alpha_ = dz<float>(P * (c.ffn / 512) * 4);
  tp_toks_ = dz<int>(P * 4); tp_pos_ = dz<int>((P / 32 + 2) * 4);
  { const char* a = std::getenv("YSKN_PFA_ATTN"); if (a && a[0]) tpc_ = std::atoi(a);
    const char* b = std::getenv("YSKN_PFA_ATTN_M"); if (b && std::atoi(b) > 0) tpa_ = std::min(tpm_, std::atoi(b) / 32 * 32 > 0 ? std::atoi(b) / 32 * 32 : 32); }
  tp_pm_ = dz<float>((size_t)tpa_ * m_.splits * c.heads * 4); tp_pl_ = dz<float>((size_t)tpa_ * m_.splits * c.heads * 4);
  tp_pacc_ = dz<float>((size_t)tpa_ * m_.splits * c.heads * 128 * 4); tp_o16_ = dz<uint16_t>(P * c.hidden * 2);
  size_t need = 0;
  for (const DevLayer& L : m_.layers) need = std::max({need, (size_t)(L.q.dev.N + L.k.dev.N + L.v.dev.N), (size_t)L.o.dev.N, (size_t)(L.gate.dev.N + L.up.dev.N), (size_t)L.down.dev.N});
  tp_fold_ = dz<float>(need * P * 4);
}

void Engine::pfa_gemm(std::initializer_list<std::pair<DevTensor*, float*>> segs, int M, bool normed) {
  GemvLaunch P{}; P.M = M;
  for (const auto& sp : segs) { P.seg[P.n_seg] = GemvSeg{sp.first->dev, sp.second, 0}; P.n_seg++; P.max_workers = std::max(P.max_workers, sp.first->dev.n_workers); }
  P.qx = tp_qx_; P.alpha = tp_alpha_;
  if (pfa_xsm()) {                                                                
    P.xsum = 1; P.us_out = xsm_us_;
    if (normed) { P.xsm_S = xsm_rr_; P.xsm_sst = 8; P.xsm_nb = m_.cfg.hidden / 512; P.xsm_n = (float)m_.cfg.hidden; P.xsm_eps = m_.cfg.eps; } }
  if (!pfa_eligible(P)) throw std::runtime_error("PFA: launch not eligible (needs the LNORD copies, no RAR / XSUM)");
  launch_pfa(P, pfa_, pfa_wpc_, st_);
}

void Engine::layer_forward_pfa(int l, int M) {
  const ModelDesc& c = m_.cfg; DevLayer& L = m_.layers[l]; g_pdl_rows = M; const bool xsm = pfa_xsm();
  
  
  if (!xsm || l == 0) launch_norm_xform_a8(tp_h_, L.ln1, c.eps, xsmx(XformA8Args{nullptr, L.q.signs_in, L.q.qm_in, c.hidden, L.q.m_in, tp_qx_, tp_alpha_, M}, xsm), st_);
  float* const fq = tp_fold_; float* const fk = fq + (size_t)M * L.q.dev.N; float* const fv = fk + (size_t)M * L.k.dev.N;
  pfa_gemm({{&L.q, fq}, {&L.k, fk}, {&L.v, fv}}, M, true);
  { QkvPostArgs Q{fq, fk, fv, L.q.tau, L.k.tau, L.v.tau, 1, 1, 1, L.q.signs_out, L.k.signs_out, L.v.signs_out, L.bq, L.bk,
                  M, a_.d_pos, tp_q16_, L.kv, m_.max_ctx, c.rope_theta, rope_};
    if (xsm) { Q.us = xsm_us_; Q.rr = xsm_rr_; Q.xsm_eps = c.eps; Q.fy = 1; }
    launch_qkv_post(Q, st_); }
  launch_kv_group_quant(KvQuantArgs{L.kv, a_.d_pos, M, m_.max_ctx}, st_);          
  if (tpc_) {                                                                       
    if (tpf_) launch_kv_dequant(AttnArgs{tp_q16_, L.kv, m_.max_ctx, M, tp_pos_, m_.splits, tp_pm_, tp_pl_, tp_pacc_, c.rope_theta, rope_}, kvf_, st_);   
    for (int s = 0; s < M; s += tpa_) {
      const int Ms = std::min(tpa_, M - s);
      const AttnArgs AA{tp_q16_ + (size_t)s * c.heads * 128, L.kv, m_.max_ctx, Ms, tp_pos_ + s / 32, m_.splits, tp_pm_, tp_pl_, tp_pacc_, c.rope_theta, rope_};
      if (tpf_) launch_attention_prefill_f32(AA, kvf_, L.bv, tp_o16_ + (size_t)s * c.hidden, st_);
      else launch_attention_prefill(AA, L.bv, tp_o16_ + (size_t)s * c.hidden, st_);
    }
    launch_xform_a8(XformA8Args{tp_o16_, L.o.signs_in, L.o.qm_in, c.hidden, L.o.m_in, tp_qx_, tp_alpha_, M}, st_);
  } else
  for (int s = 0; s < M; s += 32) {                                                
    const int Ms = std::min(32, M - s);
    launch_attention(AttnArgs{tp_q16_ + (size_t)s * c.heads * 128, L.kv, m_.max_ctx, Ms, tp_pos_ + s / 32, m_.splits, a_.pm, a_.pl, a_.pacc, c.rope_theta, rope_}, st_);
    launch_attn_combine_xform(AttnCombineArgs{a_.pm, a_.pl, a_.pacc, m_.splits, Ms, L.bv, a_.o16}, L.o.signs_in,
                              tp_qx_ + (size_t)s * c.hidden, tp_alpha_ + (size_t)s * (c.hidden / 512), st_);
  }
  pfa_gemm({{&L.o, tp_fold_}}, M);
  if (xsm) { XsmResXArgs X{tp_fold_, xsm_us_, L.o.tau, L.o.signs_out, tp_h_, L.ln2, L.gate.signs_in, tp_qx_, tp_alpha_, xsm_rr_, c.hidden, M, 0, xsm_nf_, xsm_stats_}; X.fy = 1;
    launch_xsum_res_xform(X, st_); }
  else {
  launch_residual_post(ResidualPostArgs{tp_fold_, L.o.tau, 1, L.o.signs_out, M, c.hidden, tp_h_}, st_);
  launch_norm_xform_a8(tp_h_, L.ln2, c.eps, XformA8Args{nullptr, L.gate.signs_in, L.gate.qm_in, c.hidden, L.gate.m_in, tp_qx_, tp_alpha_, M}, st_); }
  float* const fg = tp_fold_; float* const fu = fg + (size_t)M * L.gate.dev.N;
  pfa_gemm({{&L.gate, fg}, {&L.up, fu}}, M, true);
  MlpPostArgs MA{fg, fu, L.gate.tau, L.up.tau, 1, L.gate.signs_out, L.up.signs_out, M, tp_xff_}; MA.F = c.ffn;
  if (xsm) { MA.us = xsm_us_; MA.rr = xsm_rr_; MA.xsm_eps = c.eps; MA.fy = 1; }
  launch_mlp_post_xform(MA, L.down.signs_in, tp_qx_, tp_alpha_, st_);
  pfa_gemm({{&L.down, tp_fold_}}, M);
  if (xsm) { const bool lastl = l + 1 == c.layers;                                  
    XsmResXArgs X{tp_fold_, xsm_us_, L.down.tau, L.down.signs_out, tp_h_, lastl ? m_.final_norm : m_.layers[l + 1].ln1,
                  lastl ? m_.head.signs_in : m_.layers[l + 1].q.signs_in, tp_qx_, tp_alpha_, xsm_rr_, c.hidden, M, 0, xsm_nf_, xsm_stats_}; X.fy = 1;
    launch_xsum_res_xform(X, st_); }
  else launch_residual_post(ResidualPostArgs{tp_fold_, L.down.tau, 1, L.down.signs_out, M, c.hidden, tp_h_}, st_);
  for (size_t k = 0; k < cap_layers_.size(); ++k)                                 
    if (cap_layers_[k] == l) {
      if (!tp_cap_) tp_cap_ = dz<float>((size_t)3 * tpm_ * c.hidden * 4);
      cudaMemcpyAsync(tp_cap_ + k * (size_t)tpm_ * c.hidden, tp_h_, (size_t)M * c.hidden * 4, cudaMemcpyDeviceToDevice, st_);
    }
}
int Engine::prefill_pfa(const int32_t* ids, int n) {
  if (pos_ + n > m_.max_ctx) throw std::runtime_error("context overflow");
  pfa_alloc(); const ModelDesc& c = m_.cfg;
  for (int done = 0; done < n;) {
    const int M = std::min(tpm_, n - done);
    launch_set_int(a_.d_pos, pos_, st_);
    for (int k = 0; 32 * k < M; ++k) launch_set_int(tp_pos_ + k, pos_ + 32 * k, st_);
    cudaMemcpyAsync(tp_toks_, ids + done, (size_t)M * 4, cudaMemcpyHostToDevice, st_);
    g_pdl_rows = M;
    launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in,
                             tp_toks_, tp_h_, M}, st_);
    for (int l = 0; l < c.layers; ++l) layer_forward_pfa(l, M);
    if (done + M == n) {                                                           
      g_pdl_rows = 1;
      if (pfa_xsm()) {                                                            
        GemvLaunch P = gemv_base(m_, 1, (int)m_.stage_bytes_head); add_seg(P, m_.head); P.qx = tp_qx_ + (size_t)(M - 1) * c.hidden;
        P.alpha = tp_alpha_ + (size_t)(M - 1) * (c.hidden / 512); xsmg(P, true); launch_gemv(P, st_);
        ArgmaxArgs AA{m_.head.part, m_.head.tau, c.vocab, a_.logits, a_.d_next, a_.blk_val, a_.blk_idx, m_.head.dev.n_workers, 1, 0, 1,
                      xsm_us_, xsm_rr_ + (size_t)(M - 1) * 8, c.eps};
        launch_argmax(AA, st_);
      } else {
      launch_norm_xform_a8(tp_h_ + (size_t)(M - 1) * c.hidden, m_.final_norm, c.eps,
                           XformA8Args{nullptr, m_.head.signs_in, m_.head.qm_in, c.hidden, m_.head.m_in, a_.qx_ff, a_.alpha_ff, 1}, st_);
      GemvLaunch P = gemv_base(m_, 1, (int)m_.stage_bytes_head); add_seg(P, m_.head); P.qx = a_.qx_ff; P.alpha = a_.alpha_ff; launch_gemv(P, st_);
      ArgmaxArgs AA{m_.head.part, m_.head.tau, c.vocab, a_.logits, a_.d_next, a_.blk_val, a_.blk_idx, m_.head.dev.n_workers, 1, 0, 1};
      if (fuse2_) AA.cnt = am_cnt_;
      launch_argmax(AA, st_); }
    }
    cudaStreamSynchronize(st_);                                                    
    pos_ += M; done += M;
  }
  launch_set_int(a_.d_pos, pos_, st_);
  int next = 0; cudaMemcpyAsync(&next, a_.d_next, 4, cudaMemcpyDeviceToHost, st_); cudaStreamSynchronize(st_);
  launch_set_int(a_.d_tok, next, st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_pfa: ") + cudaGetErrorString(e));
  return next;
}


int Engine::prefill_rows_pfa(const int32_t* ids, int n, int32_t* out_argmax, int32_t* top_ids, float* top_vals) {
  if (pos_ + n > m_.max_ctx) throw std::runtime_error("context overflow");
  pfa_alloc(); const ModelDesc& c = m_.cfg; const int H = c.hidden;
  for (int done = 0; done < n;) {
    const int M = std::min(tpm_, n - done), P0 = pos_;
    launch_set_int(a_.d_pos, pos_, st_);
    for (int k = 0; 32 * k < M; ++k) launch_set_int(tp_pos_ + k, pos_ + 32 * k, st_);
    cudaMemcpyAsync(tp_toks_, ids + done, (size_t)M * 4, cudaMemcpyHostToDevice, st_);
    g_pdl_rows = M;
    launch_embedding(EmbArgs{m_.emb.row_stream, 4 * m_.emb.bits, c.hidden / 256, m_.emb.bits, m_.emb.lut16, m_.emb.scale16, m_.emb.signs_in, m_.emb.qm_in, m_.emb.m_in,
                             tp_toks_, tp_h_, M}, st_);
    for (int l = 0; l < c.layers; ++l) layer_forward_pfa(l, M);
    const bool xsm = pfa_xsm();
    if (!xsm) launch_norm_xform_a8(tp_h_, m_.final_norm, c.eps, XformA8Args{nullptr, m_.head.signs_in, m_.head.qm_in, H, m_.head.m_in, tp_qx_, tp_alpha_, M}, st_);
    for (int s = 0; s < M; s += 32) {
      const int Ms = std::min(32, M - s); g_pdl_rows = Ms;
      const float* hpart = m_.head.part; int hnw = m_.head.dev.n_workers;
      if (xsm) {                                                                    
        if ((s & 127) == 0) {                                                      
          if (!tp_head_) tp_head_ = dz<float>((size_t)128 * c.vocab * 4);          
          GemvLaunch P{}; P.M = std::min(128, M - s); P.seg[0] = GemvSeg{m_.head.dev, tp_head_, 0}; P.n_seg = 1; P.max_workers = m_.head.dev.n_workers;
          P.qx = tp_qx_ + (size_t)s * H; P.alpha = tp_alpha_ + (size_t)s * (H / 512); P.xsum = 1; P.us_out = xsm_us_;
          P.xsm_S = xsm_rr_ + (size_t)s * 8; P.xsm_sst = 8; P.xsm_nb = H / 512; P.xsm_n = (float)H; P.xsm_eps = c.eps;
          g_pdl_rows = P.M;
          if (!pfa_eligible(P)) throw std::runtime_error("prefill_rows_pfa: head not PFA-eligible");
          launch_pfa(P, pfa_, pfa_wpc_, st_); g_pdl_rows = Ms; }
        hpart = tp_head_ + (size_t)(s & 127) * c.vocab; hnw = 1;
      } else {
        GemvLaunch P = gemv_base(m_, Ms, (int)m_.stage_bytes_head); add_seg(P, m_.head); P.qx = tp_qx_ + (size_t)s * H; P.alpha = tp_alpha_ + (size_t)s * (H / 512);
        launch_gemv(P, st_); }
      ArgmaxArgs AA{hpart, m_.head.tau, c.vocab, a_.logits, a_.d_next8, a_.blk_val, a_.blk_idx, hnw, Ms, 0, Ms};
      launch_argmax(AA, st_);
      launch_topk8(TopkArgs{a_.logits, c.vocab, a_.tk_val, a_.tk_idx, a_.top_val, a_.top_idx, Ms}, st_);
      cudaMemcpyAsync(pr_am_ + done + s, a_.d_next8, (size_t)Ms * 4, cudaMemcpyDeviceToDevice, st_);
      cudaMemcpyAsync(pr_ti_ + (size_t)(done + s) * 8, a_.top_idx, (size_t)Ms * 32, cudaMemcpyDeviceToDevice, st_);
      cudaMemcpyAsync(pr_tv_ + (size_t)(done + s) * 8, a_.top_val, (size_t)Ms * 32, cudaMemcpyDeviceToDevice, st_);
      if (s + Ms == M) cudaMemcpyAsync(a_.d_next, a_.d_next8 + Ms - 1, 4, cudaMemcpyDeviceToDevice, st_);
    }
    if (dm_.on) {                                                                  
      const size_t slot = (size_t)a_.Mmax * H;
      for (int s = 0; s < M; s += 32) {
        const int Ms = std::min(32, M - s);
        for (int k = 0; k < 3; ++k) cudaMemcpyAsync(cap_ + k * slot, tp_cap_ + k * (size_t)tpm_ * H + (size_t)s * H, (size_t)Ms * H * 4, cudaMemcpyDeviceToDevice, st_);
        pos_ = P0 + s; draft_prefill_chunk(ids + done + s, Ms);
      }
    }
    cudaStreamSynchronize(st_);
    pos_ = P0 + M; done += M;
  }
  if (out_argmax) cudaMemcpyAsync(out_argmax, pr_am_, (size_t)n * 4, cudaMemcpyDeviceToHost, st_);
  if (top_ids) cudaMemcpyAsync(top_ids, pr_ti_, (size_t)n * 32, cudaMemcpyDeviceToHost, st_);
  if (top_vals) cudaMemcpyAsync(top_vals, pr_tv_, (size_t)n * 32, cudaMemcpyDeviceToHost, st_);
  launch_set_int(a_.d_pos, pos_, st_);
  int next = 0; cudaMemcpyAsync(&next, a_.d_next, 4, cudaMemcpyDeviceToHost, st_); cudaStreamSynchronize(st_);
  launch_set_int(a_.d_tok, next, st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_rows_pfa: ") + cudaGetErrorString(e));
  return next;
}
uint64_t Engine::debug_kv_hash() {
  cudaStreamSynchronize(st_);
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](const void* dev, size_t bytes) { std::vector<uint8_t> b(bytes); if (bytes) cudaMemcpy(b.data(), dev, bytes, cudaMemcpyDeviceToHost);
    for (uint8_t x : b) { h ^= x; h *= 1099511628211ull; } };
  const int P = pos_, G = P / 32, gmax = m_.max_ctx / 32;
  for (const DevLayer& L : m_.layers)
    for (int kvh = 0; kvh < 4; ++kvh) {
      mix(L.kv.kq + (size_t)kvh * gmax * 32 * 64, (size_t)G * 32 * 64);
      mix(L.kv.kscale + (size_t)kvh * gmax * 128, (size_t)G * 128 * 2); mix(L.kv.kzero + (size_t)kvh * gmax * 128, (size_t)G * 128 * 2);
      mix(L.kv.kshadow + (size_t)kvh * m_.max_ctx * 128, (size_t)P * 128 * 2);
      mix(L.kv.vq + (size_t)kvh * m_.max_ctx * 64, (size_t)P * 64);
      mix(L.kv.vscale + (size_t)kvh * m_.max_ctx, (size_t)P * 2); mix(L.kv.vzero + (size_t)kvh * m_.max_ctx, (size_t)P * 2);
    }
  return h;
}
}
