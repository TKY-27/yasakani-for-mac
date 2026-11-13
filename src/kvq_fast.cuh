// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
















#include "kvq_fast.hpp"
#include <cstdlib>
namespace yskn {
struct TbFastSmem {
  float red[8];
  float r[256];                    
  float cdec[256];                 
  int cost[2][288];                
  int xi[64];                      
  unsigned char bp[64][32];        
  unsigned char stout[64];         
  unsigned long long redk[8];
};
__device__ __forceinline__ float tbf_sum(float* red, float v) {            
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
  #pragma unroll
  for (int w = 0; w < 8; ++w) t = __fadd_rn(t, red[w]);
  __syncthreads();
  return t;
}
__device__ __forceinline__ float tbf_max(float* red, float v) {            
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
  #pragma unroll
  for (int w = 0; w < 8; ++w) t = fmaxf(t, red[w]);
  __syncthreads();
  return t;
}
struct TbG { int nm, mode, kk, tix, guard, m, has_mu; unsigned go, off_m, off_t; bool okg; };
__device__ __forceinline__ TbG tbf_group(const unsigned char* b, const int g) {
  TbG P;
  P.nm = (int)tb_u16(b + 24); P.off_m = tb_u32(b + 32); P.off_t = tb_u32(b + 36);
  const unsigned off_g = tb_u32(b + 40), ngrp = tb_u32(b + 28);
  P.okg = (unsigned)g < ngrp;
  P.go = P.okg ? tb_u32(b + off_g + 4u * (unsigned)g) : 0u;
  P.mode = P.okg ? b[P.go] : 0; P.kk = P.okg ? b[P.go + 1] : 4; P.tix = P.okg ? b[P.go + 2] : 0; P.guard = P.okg ? b[P.go + 3] : 0;
  P.m = P.okg ? b[P.go + 4] : 0; P.has_mu = P.okg ? b[P.go + 5] : 0;
  return P;
}
struct TbRow { float y; int g, pos, kind; size_t row; int8_t* q8; float* qs; bool cap; };
__device__ __forceinline__ TbRow tbf_row(const TbFastArgs& A, const int rj, const int d) {
  TbRow R;
  if (A.yin) {                                                               
    const int ri = A.row0 + rj;
    R.g = A.grp_l[ri]; R.pos = A.pos_l[ri]; R.y = A.yin[(size_t)ri * 256 + d]; R.row = (size_t)ri; R.q8 = nullptr; R.qs = nullptr; R.kind = 0; R.cap = false;
  } else {                                                                   
    const int tok = A.tok0 + (rj >> 3), kind = (rj >> 2) & 1, kvh = rj & 3;
    R.pos = *A.d_pos + tok; R.g = (A.layer * 4 + kvh) * 2 + kind;
    R.q8 = kind ? A.q8_1 : A.q8_0; R.qs = kind ? A.qs_1 : A.qs_0; R.kind = kind;
    R.row = (size_t)kvh * A.max_ctx + R.pos;
    const int stg = kind ? A.stg_v : A.stg_k;                                
    R.cap = stg != 0;
    const size_t irow = stg ? (size_t)kvh * stg + tok : R.row;
    R.y = __fmul_rn((float)R.q8[irow * 256 + d], R.qs[irow]);                
  }
  return R;
}
struct TbPre { float r, mud, cj0, cj1, uj0, uj1, rms; };
__device__ __forceinline__ TbPre tbf_pre(float* red, const unsigned char* b, const TbG& P, const float y) {    
  const int d = threadIdx.x;
  const unsigned char* mup = b + P.go + 8;
  const unsigned char* Up = mup + (P.has_mu ? 512 : 0);
  TbPre Q; Q.mud = 0.f; Q.cj0 = Q.cj1 = Q.uj0 = Q.uj1 = 0.f;
  float r = y;
  if (P.has_mu) { Q.mud = tb_h2f(tb_u16(mup + 2 * d)); r = __fsub_rn(y, Q.mud); }
  if (P.m > 0) { Q.uj0 = tb_h2f(tb_u16(Up + 2 * d));         Q.cj0 = tb_f16r(tbf_sum(red, __fmul_rn(r, Q.uj0))); r = __fsub_rn(r, __fmul_rn(Q.cj0, Q.uj0)); }
  if (P.m > 1) { Q.uj1 = tb_h2f(tb_u16(Up + 2 * (256 + d))); Q.cj1 = tb_f16r(tbf_sum(red, __fmul_rn(r, Q.uj1))); r = __fsub_rn(r, __fmul_rn(Q.cj1, Q.uj1)); }
  const float ssq = tbf_sum(red, __fmul_rn(r, r));
  float rms = __fsqrt_rn(__fdiv_rn(ssq, 256.f)); if (rms == 0.f) rms = 1.f;
  Q.r = r; Q.rms = rms;
  return Q;
}


template <int KK>
__device__ __forceinline__ void tbf_chain(TbFastSmem& S, const int ci_s, const float s, const int c, unsigned char* so, float* wo) {
  constexpr int KQ = 1 << KK, UU = 256 >> KK, SH = 8 - KK;
  const int d = threadIdx.x, lane = d & 31, u = d >> KK, j = d & (KQ - 1);
  const int pidx = j * UU + u;
  const int prd = pidx + (pidx >> SH);                                       
  const int wr = d + (d >> SH);                                              
  const int gbase = lane & ~(KQ - 1);
  const unsigned gmask = (KQ == 32) ? 0xffffffffu : ((1u << KQ) - 1u);
  S.cost[0][wr] = (d == 0) ? 0 : 0x40000000;
  __syncthreads();
  #pragma unroll 1
  for (int t = 0; t < 64; ++t) {
    const int cv = S.cost[t & 1][prd];
    int mn = cv;
    #pragma unroll
    for (int o = 1; o < KQ; o <<= 1) mn = min(mn, __shfl_xor_sync(0xffffffffu, mn, o));
    const unsigned eq = __ballot_sync(0xffffffffu, cv == mn);
    const int jb = __ffs((eq >> gbase) & gmask) - 1;                         
    const int dx = S.xi[t] - ci_s;
    S.cost[(t + 1) & 1][wr] = dx * dx + mn;
    if (j == 0) S.bp[t][u] = (unsigned char)jb;
    __syncthreads();
  }
  unsigned long long key = ((unsigned long long)(unsigned)S.cost[0][wr] << 8) | (unsigned long long)d;
  for (int o = 16; o; o >>= 1) { const unsigned long long k2 = __shfl_xor_sync(0xffffffffu, key, o); key = k2 < key ? k2 : key; }
  if (lane == 0) S.redk[d >> 5] = key;
  __syncthreads();
  if (d == 0) {
    unsigned long long mk = ~0ull;
    #pragma unroll
    for (int w = 0; w < 8; ++w) { const unsigned long long v = S.redk[w]; mk = v < mk ? v : mk; }
    int st = (int)(mk & 255ull);
    for (int t = 63; t >= 0; --t) {
      S.stout[t] = (unsigned char)st;
      const int jj = S.bp[t][st >> KK];
      st = (jj << SH) | (st >> KK);
    }
  }
  __syncthreads();
  if (d < 64) {                                                              
    const int stt = S.stout[d];
    const float rec = __fmul_rn(S.cdec[stt], s);
    const float dd = __fsub_rn(S.r[c * 64 + d], rec);
    float v = __fmul_rn(dd, dd);
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((d & 31) == 0) wo[d >> 5] = v;
    so[d] = (unsigned char)stt;
  }
}

__global__ void __launch_bounds__(256) tbf_vit_kernel(TbFastArgs A) {
  __shared__ TbFastSmem S;
  const int d = threadIdx.x, nm = A.nm;
  const int job = blockIdx.x, rj = job / (nm * 4), rem = job - rj * (nm * 4), mi = rem >> 2, c = rem & 3;
  const unsigned char* b = A.blob;
  const TbRow R = tbf_row(A, rj, d);
  const TbG P = tbf_group(b, R.g);
  if (!P.okg || R.pos < P.guard || P.mode != 0 || (A.k3_only && (P.kk == 4 || P.kk == 5))) return;   
  const TbPre Q = tbf_pre(S.red, b, P, R.y);
  const unsigned to = tb_u32(b + P.off_t + 4u * (unsigned)P.tix);
  const float gain = tb_f32(b + to + 4);
  if (b[to] != 8 || b[to + 2] != 64 || b[to + 3] != 8 || tb_u16(b + 16) != 256) __trap();
  const int ci_s = (int)(short)tb_u16(b + to + 12 + 2 * d);
  S.cdec[d] = tb_cdec(ci_s, gain);
  S.r[d] = Q.r;
  float s = tb_f16r(__fmul_rn(Q.rms, tb_f32(b + P.off_m + 4u * (unsigned)mi))); if (s == 0.f) s = 1.f;
  if ((d >> 6) == c) {
    const float t = rintf(__fmul_rn(__fdiv_rn(Q.r, s), 256.f));
    S.xi[d & 63] = (int)fminf(fmaxf(t, -2048.f), 2048.f);
  }
  unsigned char* const so = A.scr + ((size_t)rj * nm + mi) * 256 + c * 64;
  float* const wo = reinterpret_cast<float*>(A.scr + TBF_W_OFS) + ((size_t)rj * nm + mi) * 8 + 2 * c;
  switch (P.kk) {
    case 3: tbf_chain<3>(S, ci_s, s, c, so, wo); break;
    case 4: tbf_chain<4>(S, ci_s, s, c, so, wo); break;
    case 5: tbf_chain<5>(S, ci_s, s, c, so, wo); break;
    default: __trap();
  }
}

__global__ void __launch_bounds__(256) tbf_sel_kernel(TbFastArgs A) {
  __shared__ float red[8];
  __shared__ unsigned char fld[256];
  const int d = threadIdx.x, nm = A.nm, rj = blockIdx.x;
  const unsigned char* b = A.blob;
  const TbRow R = tbf_row(A, rj, d);
  const TbG P = tbf_group(b, R.g);
  float x = R.y;                                                             
  int field = 0, kb = 4; float psc = 1.f; unsigned pc = 0u; bool pack = false;   
  if (P.okg && R.pos >= P.guard) {
    const TbPre Q = tbf_pre(red, b, P, R.y);
    float xq = Q.r;
    if (P.mode == 1) {                                                       
      const float am = tbf_max(red, fabsf(Q.r));
      float s7 = __fdiv_rn(am, 7.f); if (s7 == 0.f) s7 = 1.f;
      float q = rintf(__fdiv_rn(Q.r, s7)); q = fminf(fmaxf(q, -7.f), 7.f); xq = __fmul_rn(q, s7);
      field = (int)q + 8; psc = s7; kb = 4; pack = true;
    } else if (P.mode == 0) {                                                
      const float* const W = reinterpret_cast<const float*>(A.scr + TBF_W_OFS) + (size_t)rj * nm * 8;
      float best_e = 0.f; int best = 0; bool have = false;
      for (int mi = 0; mi < nm; ++mi) {
        float e = 0.f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) e = __fadd_rn(e, W[mi * 8 + w]);
        if (!have || e < best_e) { have = true; best_e = e; best = mi; }
      }
      float s = tb_f16r(__fmul_rn(Q.rms, tb_f32(b + P.off_m + 4u * (unsigned)best))); if (s == 0.f) s = 1.f;
      const unsigned to = tb_u32(b + P.off_t + 4u * (unsigned)P.tix);
      const float gain = tb_f32(b + to + 4);
      const int st = A.scr[((size_t)rj * nm + best) * 256 + d];
      const int ci = (int)(short)tb_u16(b + to + 12 + 2 * st);
      xq = __fmul_rn(tb_cdec(ci, gain), s);
      field = st & ((1 << P.kk) - 1); psc = s; kb = P.kk; pack = true;
    }
    x = xq;
    if (P.m > 0) x = __fadd_rn(x, __fmul_rn(Q.cj0, Q.uj0));
    if (P.m > 1) x = __fadd_rn(x, __fmul_rn(Q.cj1, Q.uj1));
    if (P.has_mu) x = __fadd_rn(x, Q.mud);
    pc = (unsigned)__half_as_ushort(__float2half_rn(Q.cj0)) | ((unsigned)__half_as_ushort(__float2half_rn(Q.cj1)) << 16);   
  }
  unsigned char* const payb = (!A.yin && R.kind) ? A.pk_pay1 : A.pk_pay;
  if (payb && pack) {                                                        
    float* const scl = (!A.yin && R.kind) ? A.pk_scl1 : A.pk_scl0;           
    unsigned* const sdst = scl ? nullptr : ((!A.yin && R.kind) ? A.pk_side1 : A.pk_side) + 2 * R.row;
    fld[d] = (unsigned char)field;
    __syncthreads();
    if (d < 8 * kb) {
      unsigned w = 0u;
      for (int i = 0; i < 32; ++i) { const int g = 32 * d + i; const int e = g / kb, jb = g - e * kb; w |= (((unsigned)fld[e] >> (kb - 1 - jb)) & 1u) << (31 - i); }
      reinterpret_cast<unsigned*>(payb + R.row * (size_t)A.pk_stride)[d] = w;
    }
    if (d == 0) { if (scl) scl[R.row] = psc; else { sdst[0] = __float_as_uint(psc); sdst[1] = pc; } }
  }
  if (A.yin) A.yout[(size_t)(A.row0 + rj) * 256 + d] = x;
  else if (!R.cap) kvx_store(x, R.q8, R.qs, R.row, d);
}

#include "kvq_vit2.cuh"
static int tbf_vit2_env() { static const int v = [] { const char* f = std::getenv("YSKN_TB_VIT2"); return (!f || !f[0] || std::atoi(f) != 0) ? 1 : 0; }(); return v; }

void launch_tb_fast(const TbFastArgs& A, cudaStream_t st) {
  if (A.nrow <= 0) return;
  if (A.has_cdq && (A.stage & 1)) {
    if ((A.has_cdq & 2) && !A.vit2_off && tbf_vit2_env()) {                  
      tbf2_vit_kernel<<<(A.nrow * A.nm * 2 + 3) / 4, 128, 0, st>>>(A);
      if (A.has_cdq & 4) { TbFastArgs L = A; L.k3_only = 1; tbf_vit_kernel<<<L.nrow * L.nm * 4, 256, 0, st>>>(L); }
    } else tbf_vit_kernel<<<A.nrow * A.nm * 4, 256, 0, st>>>(A);
  }
  if (A.stage & 2) tbf_sel_kernel<<<A.nrow, 256, 0, st>>>(A);
}
TbHost& tbf_host() { static TbHost h; return h; }
} 
