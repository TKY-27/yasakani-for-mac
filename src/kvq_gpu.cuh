// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once












#include <cuda_fp16.h>
#include <stdint.h>
namespace yskn {
struct TbScratch {
  unsigned char bp[2][64][256];          
  int cost[2][2][256];                   
  int xi[2][256];                        
  float rec[2][256];                     
  float cdec[2][256];                    
  float red[16];
  unsigned long long redk[16];
};
__device__ __forceinline__ unsigned tb_u16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
__device__ __forceinline__ unsigned tb_u32(const unsigned char* p) { return tb_u16(p) | (tb_u16(p + 2) << 16); }
__device__ __forceinline__ float tb_f32(const unsigned char* p) { return __uint_as_float(tb_u32(p)); }
__device__ __forceinline__ float tb_h2f(unsigned h) { return __half2float(__ushort_as_half((unsigned short)h)); }
__device__ __forceinline__ float tb_f16r(float v) { return __half2float(__float2half_rn(v)); }
__device__ __forceinline__ float tb_sum(TbScratch& S, float v) {      
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) S.red[threadIdx.x >> 5] = v;
  __syncthreads();
  const int w0 = (threadIdx.x >> 8) * 8; float t = 0.f;
  #pragma unroll
  for (int w = 0; w < 8; ++w) t = __fadd_rn(t, S.red[w0 + w]);
  __syncthreads();
  return t;
}
__device__ __forceinline__ float tb_max(TbScratch& S, float v) {
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) S.red[threadIdx.x >> 5] = v;
  __syncthreads();
  const int w0 = (threadIdx.x >> 8) * 8; float t = 0.f;
  #pragma unroll
  for (int w = 0; w < 8; ++w) t = fmaxf(t, S.red[w0 + w]);
  __syncthreads();
  return t;
}
__device__ __forceinline__ float tb_cdec(int ci, float gain) { return __fmul_rn(__fmul_rn((float)ci, 0.00390625f), gain); }   
static __device__ __noinline__ float tb_fakequant(const float y, const unsigned char* b, const int g, const int pos) {
  if (b == nullptr) return y;
  __shared__ TbScratch S;
  const int tid = threadIdx.x, half = tid >> 8, d = tid & 255;
  const int nm = (int)tb_u16(b + 24);
  const unsigned off_m = tb_u32(b + 32), off_t = tb_u32(b + 36), off_g = tb_u32(b + 40);
  const unsigned ngrp = tb_u32(b + 28);
  const bool okg = (unsigned)g < ngrp;
  const unsigned go = okg ? tb_u32(b + off_g + 4u * (unsigned)g) : 0u;
  const int mode = okg ? b[go] : 0, kk = okg ? b[go + 1] : 4, tix = okg ? b[go + 2] : 0, guard = okg ? b[go + 3] : 0, m = okg ? b[go + 4] : 0, has_mu = okg ? b[go + 5] : 0;
  const bool isGuard = !okg || pos < guard;
  const bool cdq = !isGuard && mode == 0, flat = !isGuard && mode == 1;
  const unsigned char* mup = b + go + 8;
  const unsigned char* Up = mup + (has_mu ? 512 : 0);
  const int m_eff = isGuard ? 0 : m;
  
  float r = y;
  float mud = 0.f;
  if (!isGuard && has_mu) { mud = tb_h2f(tb_u16(mup + 2 * d)); r = __fsub_rn(y, mud); }
  float cj[2] = {0.f, 0.f}, uj[2] = {0.f, 0.f};
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    if (j < m_eff) uj[j] = tb_h2f(tb_u16(Up + 2 * (j * 256 + d)));
    const float sdot = tb_sum(S, (j < m_eff) ? __fmul_rn(r, uj[j]) : 0.f);
    if (j < m_eff) { cj[j] = tb_f16r(sdot); r = __fsub_rn(r, __fmul_rn(cj[j], uj[j])); }
  }
  const float ssq = tb_sum(S, __fmul_rn(r, r));
  float rms = __fsqrt_rn(__fdiv_rn(ssq, 256.f)); if (rms == 0.f) rms = 1.f;
  float xq = r;
  
  {
    const float am = tb_max(S, fabsf(r));
    if (flat) { float s7 = __fdiv_rn(am, 7.f); if (s7 == 0.f) s7 = 1.f; float q = rintf(__fdiv_rn(r, s7)); q = fminf(fmaxf(q, -7.f), 7.f); xq = __fmul_rn(q, s7); }
  }
  
  const unsigned to = cdq ? tb_u32(b + off_t + 4u * (unsigned)tix) : 0u;
  const float gain = cdq ? tb_f32(b + to + 4) : 1.f;
  if (cdq && (b[to] != 8 || b[to + 2] != 64 || b[to + 3] != 8 || tb_u16(b + 16) != 256)) __trap();
  const int Kq = 1 << kk, UU = 256 >> kk, uidx = d >> kk;
  const int ci_s = cdq ? (int)(short)tb_u16(b + to + 12 + 2 * d) : 0;
  if (cdq) S.cdec[half][d] = tb_cdec(ci_s, gain);                 
  float best_e = 0.f, bestv = r; bool have = false;
  for (int mi = 0; mi < nm; ++mi) {
    float s = 1.f;
    if (cdq) {
      s = tb_f16r(__fmul_rn(rms, tb_f32(b + off_m + 4u * (unsigned)mi))); if (s == 0.f) s = 1.f;
      const float t = rintf(__fmul_rn(__fdiv_rn(r, s), 256.f));
      S.xi[half][d] = (int)fminf(fmaxf(t, -2048.f), 2048.f);
    }
    __syncthreads();
    for (int c = 0; c < 4; ++c) {
      if (cdq) S.cost[half][0][d] = (d == 0) ? 0 : 0x40000000;
      __syncthreads();
      for (int t = 0; t < 64; ++t) {
        if (cdq) {
          const int* cp = S.cost[half][t & 1];
          int mn = cp[uidx], jb = 0;
          for (int j = 1; j < Kq; ++j) { const int cc = cp[j * UU + uidx]; if (cc < mn) { mn = cc; jb = j; } }
          const int dx = S.xi[half][c * 64 + t] - ci_s;
          S.cost[half][(t + 1) & 1][d] = dx * dx + mn;
          S.bp[half][t][d] = (unsigned char)jb;
        }
        __syncthreads();
      }
      unsigned long long key = cdq ? (((unsigned long long)(unsigned)S.cost[half][0][d] << 8) | (unsigned long long)d) : ~0ull;
      for (int o = 16; o; o >>= 1) { const unsigned long long k2 = __shfl_xor_sync(0xffffffffu, key, o); key = k2 < key ? k2 : key; }
      if ((tid & 31) == 0) S.redk[tid >> 5] = key;
      __syncthreads();
      if (cdq && d == 0) {
        unsigned long long mk = ~0ull;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { const unsigned long long v = S.redk[half * 8 + w]; mk = v < mk ? v : mk; }
        int st = (int)(mk & 255ull);
        for (int t = 63; t >= 0; --t) {
          S.rec[half][c * 64 + t] = __fmul_rn(S.cdec[half][st], s);
          const int j = S.bp[half][t][st];
          st = (j << (8 - kk)) | (st >> kk);
        }
      }
      __syncthreads();
    }
    const float rv = cdq ? S.rec[half][d] : 0.f;
    const float dd = __fsub_rn(r, rv);
    const float e = tb_sum(S, cdq ? __fmul_rn(dd, dd) : 0.f);
    if (cdq && (!have || e < best_e)) { have = true; best_e = e; bestv = rv; }
  }
  if (cdq) xq = bestv;
  
  if (isGuard) return y;
  float x = xq;
  #pragma unroll
  for (int j = 0; j < 2; ++j) if (j < m_eff) x = __fadd_rn(x, __fmul_rn(cj[j], uj[j]));
  if (has_mu) x = __fadd_rn(x, mud);
  return x;
}
} 
