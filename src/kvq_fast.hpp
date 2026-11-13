// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once


#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace yskn {
constexpr int TBF_TILE = 128;                                   
constexpr int TBF_MAXNM = 16;                                   
constexpr size_t TBF_W_OFS = (size_t)TBF_TILE * 8 * TBF_MAXNM * 256;                                   
inline size_t tbf_scratch_bytes() { return TBF_W_OFS + (size_t)TBF_TILE * 8 * TBF_MAXNM * 8 * sizeof(float); }

struct TbFastArgs {
  const unsigned char* blob; int nm; int has_cdq;
  
  int8_t* q8_0; int8_t* q8_1; float* qs_0; float* qs_1; int max_ctx; const int* d_pos; int layer; int tok0;
  
  const float* yin; const int* grp_l; const int* pos_l; float* yout; int row0;
  int nrow;                                                     
  unsigned char* scr;                                           
  
  
  
  unsigned char* pk_pay = nullptr;                              
  unsigned char* pk_pay1 = nullptr;                             
  unsigned* pk_side = nullptr; unsigned* pk_side1 = nullptr;
  int pk_stride = 0;                                            
  
  
  int stg_k = 0, stg_v = 0; float* pk_scl0 = nullptr; float* pk_scl1 = nullptr;
  
  
  int k3_only = 0, vit2_off = 0, stage = 3;
};
void launch_tb_fast(const TbFastArgs&, cudaStream_t);           


struct TbHost { int nm = 0; int has_cdq = 0; unsigned char* scr = nullptr; };
TbHost& tbf_host();




inline bool tbf_parse_blob(const unsigned char* b, size_t n, int& nm, int& has_cdq) {
  auto u16 = [&](size_t o) { return (unsigned)b[o] | ((unsigned)b[o + 1] << 8); };
  auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
  if (n < 100) return false;
  nm = (int)u16(24); const unsigned ngrp = u32(28), off_tab = u32(36), off_g = u32(40); has_cdq = 0;
  int n45 = 0, bad45 = 0, nother = 0;
  for (unsigned g = 0; g < ngrp; ++g) {
    const size_t go = u32(off_g + 4 * (size_t)g); if (go + 8 > n) return false;
    if (b[go] != 0) continue;
    has_cdq |= 1;
    const int kk = b[go + 1];
    if (kk != 4 && kk != 5) { ++nother; continue; }
    ++n45;
    const size_t tp = off_tab + 4 * (size_t)b[go + 2];
    bool ok = tp + 4 <= n;
    const size_t to = ok ? (size_t)u32(tp) : 0;
    ok = ok && to + 12 + 512 <= n && b[to] == 8 && b[to + 2] == 64 && b[to + 3] == 8 && u16(16) == 256;
    for (int s = 0; ok && s < 256; ++s) { const int c = (int)(short)u16(to + 12 + 2 * (size_t)s); if (c > 1100 || c < -1100) ok = false; }
    if (!ok) ++bad45;
  }
  if (n45 && !bad45) has_cdq |= 2 | (nother ? 4 : 0);
  return true;
}
} 
