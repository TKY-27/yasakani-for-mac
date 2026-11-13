// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>
#include "model.hpp"
#include "signs.hpp"
#include "yskn_abi.h"
#include "devmem.hpp"   
namespace yskn {
static float h2f_host(uint16_t u) { __half h = *reinterpret_cast<__half*>(&u); return __half2float(h); }
template <class T2> static T2* dev_copy(const void* src, size_t bytes) { T2* d = nullptr; cudaMalloc(&d, bytes); cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice); return d; }
template <class T2> static T2* dev_zero(size_t bytes) { T2* d = nullptr; cudaMalloc(&d, bytes); cudaMemset(d, 0, bytes); return d; }

uint16_t* Model::upload_vector(const std::string& name, int n) { const VectorDesc& v = c.vector(name); if (v.n != n) throw std::runtime_error("vector size mismatch: " + name); return dev_copy<uint16_t>(c.at<uint16_t>(v.off), (size_t)n * 2); }


static void build_lut8_tau(const uint16_t* lut16, const uint16_t* s16, int rows, std::vector<uint16_t>& lut8, std::vector<float>& tau, int S) {
  float mx = 0.f; for (int i = 0; i < S * 2; ++i) mx = std::max(mx, std::fabs(h2f_host(lut16[i])));
  const float gamma = mx / 127.f; lut8.resize(S);
  for (int s = 0; s < S; ++s) { int q0 = (int)std::nearbyint(h2f_host(lut16[2 * s]) / gamma), q1 = (int)std::nearbyint(h2f_host(lut16[2 * s + 1]) / gamma);
    q0 = std::max(-127, std::min(127, q0)); q1 = std::max(-127, std::min(127, q1)); lut8[s] = (uint16_t)((uint8_t)(int8_t)q0 | ((uint16_t)(uint8_t)(int8_t)q1 << 8)); }
  tau.resize(rows); for (int r = 0; r < rows; ++r) tau[r] = h2f_host(s16[r]) * gamma;
}

DevTensor Model::upload_linear(const TensorDesc& d, int Mmax, int want_workers, int force_workers, const Container* src) {
  const Container& c = src ? *src : this->c;
  DevTensor t; t.bits = d.bits; const int S = 1 << d.L, rows = d.physical_rows, cols = d.cols, seqs = cols / 256;
  if (d.L != 12) throw std::runtime_error("engine build supports L=12 LUT8 tables only (L=14: 32 KiB LUT8, enable via REP=1 build)");
  std::vector<uint16_t> lut8; std::vector<float> tau; build_lut8_tau(c.at<uint16_t>(d.lut_off), c.at<uint16_t>(d.scale_off), rows, lut8, tau, S);
  t.lut8 = dev_copy<uint16_t>(lut8.data(), lut8.size() * 2); t.tau = dev_copy<float>(tau.data(), tau.size() * 4);
  
  auto sg_in = sign_stream_rule(d.rot_in.rule, d.rot_in.sign_seed, d.rot_in.sign_offset, cols); t.signs_in = dev_copy<int8_t>(sg_in.data(), cols); t.m_in = d.rot_in.m;
  if (d.rot_in.kind == "AYA") t.qm_in = dev_copy<uint16_t>(c.at<uint16_t>(d.rot_in.qm_off), (size_t)d.rot_in.m * d.rot_in.m * 2);
  if (d.has_rot_out) { auto sg_out = sign_stream_rule(d.rot_out.rule, d.rot_out.sign_seed, d.rot_out.sign_offset, rows); t.signs_out = dev_copy<int8_t>(sg_out.data(), rows); }
  
  const uint32_t* words = c.at<uint32_t>(d.stream_off);
  const int ntiles = rows / 64;
  if (!d.has_rar) {
    const int wps = 4 * d.bits; const size_t nwords = (size_t)rows * seqs * wps;
    uint32_t* dw = dev_copy<uint32_t>(words, nwords * 4); cudaMalloc(&t.frag, nwords * 4);
    yskn_repack_warp_to_frag(dw, (uint32_t*)t.frag, rows, seqs, wps, nullptr); cudaDeviceSynchronize(); cudaFree(dw);
    t.tiles_by_k.assign(9, {}); for (int i = 0; i < ntiles; ++i) t.tiles_by_k[d.bits].push_back(i); t.kmax = d.bits;
  } else {
    const uint8_t* kt = c.at<uint8_t>(d.rar.k_off); const uint64_t* off = c.at<uint64_t>(d.rar.off_off);
    size_t total_u4 = 0; std::vector<uint64_t> toff(ntiles); std::vector<uint8_t> tk(ntiles);
    for (int i = 0; i < ntiles; ++i) { tk[i] = kt[i]; toff[i] = total_u4; total_u4 += (size_t)64 * seqs * (4 * kt[i]) / 4; }
    cudaMalloc(&t.frag, total_u4 * 16);
    for (int i = 0; i < ntiles; ++i) { const int wps = 4 * tk[i]; const size_t nw = (size_t)64 * seqs * wps;
      uint32_t* dw = dev_copy<uint32_t>(words + off[i], nw * 4); yskn_repack_warp_to_frag(dw, (uint32_t*)(t.frag + toff[i]), 64, seqs, wps, nullptr); cudaDeviceSynchronize(); cudaFree(dw); }
    t.tile_off = dev_copy<uint64_t>(toff.data(), ntiles * 8); t.tile_k = dev_copy<uint8_t>(tk.data(), ntiles);
    t.tiles_by_k.assign(9, {}); t.kmax = 0; for (int i = 0; i < ntiles; ++i) { t.tiles_by_k[tk[i]].push_back(i); t.kmax = std::max(t.kmax, (int)tk[i]); }
  }
  
  const int G = cols / 512; int nw = force_workers > 0 ? std::min(G, force_workers) : std::max(1, std::min(G, (int)std::ceil((double)want_workers / ntiles)));
  t.part = dev_zero<float>((size_t)nw * Mmax * rows * 4);
  t.dev = TensorDev{t.frag, t.tile_off, t.tile_k, d.bits, t.lut8, t.tau, rows, cols, nullptr, nw};
  return t;
}
void Model::upload_embedding(const TensorDesc& d) {
  if (d.layout != "row") throw std::runtime_error("embedding must use row layout");
  emb.bits = d.bits; const int S = 1 << d.L; const size_t nwords = (size_t)d.physical_rows * (d.cols / 256) * (4 * d.bits);
  emb.row_stream = dev_copy<uint32_t>(c.at<uint32_t>(d.stream_off), nwords * 4);
  emb.lut16 = dev_copy<uint16_t>(c.at<uint16_t>(d.lut_off), (size_t)S * 4); emb.scale16 = dev_copy<uint16_t>(c.at<uint16_t>(d.scale_off), (size_t)d.physical_rows * 2);
  auto sg = sign_stream_rule(d.rot_in.rule, d.rot_in.sign_seed, d.rot_in.sign_offset, d.cols); emb.signs_in = dev_copy<int8_t>(sg.data(), d.cols); emb.m_in = d.rot_in.m;
  if (d.rot_in.kind == "AYA") emb.qm_in = dev_copy<uint16_t>(c.at<uint16_t>(d.rot_in.qm_off), (size_t)d.rot_in.m * d.rot_in.m * 2);
}
void Model::load(const std::string& path, int max_ctx_, int Mmax) {
  c.open(path); cfg = c.model; max_ctx = ((max_ctx_ + 63) / 64) * 64;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  int want = sm_count * 8;                                            
  { const char* e = std::getenv("YSKN_GEMV_WANT"); if (e && std::atoi(e) > 0) want = std::atoi(e); }   
  layers.resize(cfg.layers);
  int kmax = 4;
  for (int l = 0; l < cfg.layers; ++l) {
    DevLayer& L = layers[l]; const std::string p = "layers." + std::to_string(l) + ".";
    L.q = upload_linear(c.tensor(p + "q_proj"), Mmax, want); L.k = upload_linear(c.tensor(p + "k_proj"), Mmax, want); L.v = upload_linear(c.tensor(p + "v_proj"), Mmax, want);
    L.o = upload_linear(c.tensor(p + "o_proj"), Mmax, want); L.gate = upload_linear(c.tensor(p + "gate_proj"), Mmax, want); L.up = upload_linear(c.tensor(p + "up_proj"), Mmax, want);
    L.down = upload_linear(c.tensor(p + "down_proj"), Mmax, want);
    for (DevTensor* t : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) kmax = std::max(kmax, t->kmax);
    L.ln1 = upload_vector(p + "input_layernorm", cfg.hidden); L.ln2 = upload_vector(p + "post_attention_layernorm", cfg.hidden);
    L.bq = upload_vector(p + "q_proj.bias", cfg.heads * cfg.head_dim); L.bk = upload_vector(p + "k_proj.bias", cfg.kv_heads * cfg.head_dim); L.bv = upload_vector(p + "v_proj.bias", cfg.kv_heads * cfg.head_dim);
    const int groups = max_ctx / 32;
    L.kv.kq = dev_zero<uint8_t>((size_t)4 * groups * 32 * 64); L.kv.kscale = dev_zero<uint16_t>((size_t)4 * groups * 128 * 2); L.kv.kzero = dev_zero<uint16_t>((size_t)4 * groups * 128 * 2);
    L.kv.kshadow = dev_zero<uint16_t>((size_t)4 * max_ctx * 128 * 2); L.kv.vq = dev_zero<uint8_t>((size_t)4 * max_ctx * 64); L.kv.vscale = dev_zero<uint16_t>((size_t)4 * max_ctx * 2); L.kv.vzero = dev_zero<uint16_t>((size_t)4 * max_ctx * 2);
  }
  int head_w = c.tensor("lm_head").cols / 512;                 
  { const char* e = std::getenv("YSKN_HEAD_WORKERS"); if (e && std::atoi(e) > 0) head_w = std::atoi(e); }
  head = upload_linear(c.tensor("lm_head"), Mmax, want, head_w);   
  upload_embedding(c.tensor("embed_tokens")); final_norm = upload_vector("norm", cfg.hidden);
  stage_bytes_body = (size_t)GEMV_NT_PASS * 8 * 16 * kmax;                       
  stage_bytes_head = (size_t)GEMV_NT_PASS * 8 * 16 * head.kmax;                  
}
}
