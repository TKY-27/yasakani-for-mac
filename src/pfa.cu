// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <stdexcept>
#include <type_traits>
#include "kernels.hpp"
#include "pm_math.cuh"
namespace yskn {
__constant__ int pfa_pm_on;                                                                        
void pfa_set_pm(int on, cudaStream_t st) { cudaMemcpyToSymbolAsync(pfa_pm_on, &on, sizeof(int), 0, cudaMemcpyHostToDevice, st); }


























template <int LB> __host__ __device__ constexpr int pfa_maxt() { return LB == 1 ? 192 : LB == 3 ? 128 : LB == 4 ? 384 : 256; }
template <int LB> __host__ __device__ constexpr int pfa_minb() { return (LB == 2 || LB == 4) ? 1 : 2; }

__device__ __forceinline__ void pfa_mma(int* d, const unsigned* a, uint2 b) {       
#ifdef PFA_EMU
  emu::mma_m16n8k32_s8(d, a, b.x, b.y);
#else
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
#endif
}
__device__ __forceinline__ void pfa_cp8(void* smem_dst, const void* gmem_src) {    
#ifdef PFA_EMU
  std::memcpy(smem_dst, gmem_src, 8);
#else
  unsigned s; asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }\n" : "=r"(s) : "l"(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem_src) : "memory");
#endif
}
__device__ __forceinline__ void pfa_cp_commit() {
#ifndef PFA_EMU
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pfa_cp_wait() {
#ifndef PFA_EMU
  asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}
template <int K> __device__ __forceinline__ unsigned pfa_field(const unsigned* w, int i) {   
  constexpr int FB = 4 * K; const int bit = FB * i, wi = bit >> 5, sh = bit & 31;
  unsigned v = w[wi] >> sh;
  if (sh + FB > 32) v |= w[wi + 1] << (32 - sh);
  if constexpr (FB >= 32) return v; else return v & ((1u << FB) - 1u);
}


__device__ __forceinline__ float pfa_fold(bool first, float total, float p) { return first ? p : __fadd_rn(total, p); }
__device__ __forceinline__ void pfa_mad_wide(long long& acc, int a, int b) {   
#ifdef PFA_EMU
  acc += (long long)a * (long long)b;
#else
  asm("mad.wide.s32 %0, %1, %2, %0;" : "+l"(acc) : "r"(a), "r"(b));
#endif
}
struct PfaStep { int g, w; bool last; };   


template <int K, int LNF, int FOLD, bool XG, int LB, bool XSM = false>
__global__ void __launch_bounds__(pfa_maxt<LB>(), pfa_minb<LB>()) pfa_kernel(GemvLaunch P) {
  static_assert(!XSM || FOLD == 1, "PFA XSUM uses the FOLD 1 walk");
  constexpr int NF = 1 << LNF, NU = K / 2, T = 12 - K, TS = 3 * K - 12;
  constexpr int SS = NF > 4 ? 4 : 8, NST = 8 / SS;                                 
  constexpr unsigned PM = (1u << (2 * K)) - 1u, TM = (1u << T) - 1u;
  __shared__ __align__(16) uint16_t s_lut[4096];
  __shared__ __align__(16) uint4 s_x[XG ? 1 : 2][XG ? 1 : SS][XG ? 1 : NF][32];    
  int seg = 0; while (seg + 1 < P.n_seg && (int)blockIdx.y >= P.seg[seg + 1].cta_begin) ++seg;
  const GemvSeg& Sg = P.seg[seg]; const TensorDev& t = Sg.t;
  const int WPC = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gi = lane >> 2, tig = lane & 3;
  const int N = t.N, Kc = t.Kc, G = Kc / 512, nw = t.n_workers, M = P.M;
  const int worker = FOLD ? 0 : (int)blockIdx.z;
  if (!FOLD && worker >= nw) { pdl_wait(); return; }                               
  const int tile = ((int)blockIdx.y - Sg.cta_begin) * WPC + warp;
  const bool active = tile < N / 16;                                               
  const int tok0 = (int)blockIdx.x * (16 * NF);
  const int nfr0 = (M - tok0 + 15) >> 4, nfr = nfr0 < NF ? nfr0 : NF;                
  
  const int base = G / nw, rem = G % nw;
  const int ngrp = FOLD ? G : base + (worker < rem ? 1 : 0), nblk = 2 * ngrp;
  auto walk = [&](int q) -> PfaStep {                                                 
    int w, k;
    if (!FOLD) { w = worker; k = q; }
    else if (q < rem * (base + 1)) { w = q / (base + 1); k = q - w * (base + 1); }
    else { const int q2 = q - rem * (base + 1); w = rem + q2 / base; k = q2 - (w - rem) * base; }
    return PfaStep{w + k * nw, w, k == base + (w < rem ? 1 : 0) - 1};
  };
  const uint4* sp = t.lno + (size_t)(active ? tile : 0) * G * 2 * NU * 32 + lane;
  auto boff = [&](int g, int blk) { return (size_t)(g * 2 + blk) * (NU * 32); };
  
  
  
  const int st_tok = (lane >> 3) + 8 * (lane & 1);
  const int8_t* st_src = reinterpret_cast<const int8_t*>(P.qx) + (size_t)(tok0 + st_tok) * Kc + 64 * ((lane >> 1) & 3);
  auto stage = [&](int s, int g, int blk, int h) {
    const int8_t* xb = st_src + (size_t)g * 512 + blk * 256 + 8 * SS * h;
    char* db = reinterpret_cast<char*>(&s_x[XG ? 0 : s][0][0][0]) + 8 * lane;
    for (int pr = warp; pr < 2 * NF; pr += WPC) {
      const int trow = 16 * (pr >> 1) + 4 * (pr & 1);
      if (tok0 + st_tok + trow < M) {
        const int8_t* sx = xb + (size_t)trow * Kc; char* dx = db + 256 * pr;      
        #pragma unroll
        for (int m = 0; m < SS; ++m) pfa_cp8(dx + 512 * NF * m, sx + 8 * m);
      }
    }
    pfa_cp_commit();
  };
  int d[NF][2][4]; float pacc[XSM ? 1 : NF][2][4], tot[(FOLD == 1 && !XSM) ? NF : 1][2][4]; long long acc64[XSM ? NF : 1][2][4];
  #pragma unroll
  for (int j = 0; j < NF; ++j) {
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      #pragma unroll
      for (int c = 0; c < 4; ++c) { d[j][r][c] = 0;
        if constexpr (XSM) acc64[j][r][c] = 0; else { pacc[j][r][c] = 0.f; if constexpr (FOLD == 1) tot[j][r][c] = 0.f; } }
    }
  }
  uint4 nx[NU];
  #pragma unroll
  for (int u = 0; u < NU; ++u) nx[u] = make_uint4(0u, 0u, 0u, 0u);
  const int g_first = XSM ? 0 : walk(0).g;
  if (active && nblk) {                                                            
    #pragma unroll
    for (int u = 0; u < NU; ++u) nx[u] = __ldg(sp + boff(g_first, 0) + 32 * u);
  }
  for (int i = threadIdx.x; i < 512; i += blockDim.x) reinterpret_cast<uint4*>(s_lut)[i] = __ldg(reinterpret_cast<const uint4*>(t.lut8) + i);
  pdl_wait(); pdl_trigger();                                                       
  __shared__ float s_rm[XSM ? 16 * NF : 1], s_us[XSM ? 16 * NF : 1], s_rr[XSM ? 16 * NF : 1];
  if constexpr (XSM) {                                                             
    for (int i = threadIdx.x; i < 16 * NF; i += blockDim.x) {
      const int tk = tok0 + i; float m = 0.f;
      if (tk < M) for (int k = 0; k < G; ++k) m = fmaxf(m, __ldg(P.alpha + (size_t)tk * G + k));
      s_rm[i] = m > 0.f ? __fdiv_rn(16777216.f, m) : 0.f; s_us[i] = ldexpf(m, -24);
      if (blockIdx.y == 0 && tk < M) P.us_out[tk] = s_us[i];
      float rr = 0.f;                                                              
      if (P.xsm_S && tk < M) { const float* S = P.xsm_S + (size_t)tk * P.xsm_sst; float sum = 0.f;
        for (int b = 0; b < P.xsm_nb; ++b) sum = __fadd_rn(sum, S[b]);
        const float a = __fadd_rn(__fdiv_rn(sum, P.xsm_n), P.xsm_eps); rr = pfa_pm_on ? pm_rsqrt(a) : rsqrtf(a); }
      s_rr[i] = rr;
    }
    __syncthreads();
  }
  if (!XG && nblk) stage(0, g_first, 0, 0);
  if (XG) __syncthreads();
  const int src = (lane & ~3) | ((tig + 3) & 3);                                   
  float* dst = FOLD ? Sg.part : Sg.part + (size_t)worker * M * N;                      
  auto pair_b = [&](unsigned p, unsigned q) -> unsigned {                          
    const unsigned W = q | (p << T); return unsigned(s_lut[W & 0xFFFu]) | (unsigned(s_lut[(W >> K) & 0xFFFu]) << 16); };
  for (int bi = 0; bi < nblk; ++bi) {
    const int blk = bi & 1; const bool more = bi + 1 < nblk;
    unsigned w[4 * NU];                                                            
    #pragma unroll
    for (int u = 0; u < NU; ++u) { w[4 * u] = nx[u].x; w[4 * u + 1] = nx[u].y; w[4 * u + 2] = nx[u].z; w[4 * u + 3] = nx[u].w; }
    
    
    const PfaStep cur = XSM ? PfaStep{bi >> 1, 0, false} : walk(bi >> 1);
    const int g_next = XSM ? ((bi + 1) >> 1) : ((more && blk == 1) ? walk((bi + 1) >> 1).g : cur.g);
    if (more && active) {
      #pragma unroll
      for (int u = 0; u < NU; ++u) nx[u] = __ldg(sp + boff(g_next, blk ^ 1) + 32 * u);
    }
    unsigned q0 = 0u, q1 = 0u;
    if (active) {
      const unsigned e = (w[K - 1] >> (32 - T)) | ((w[2 * K - 1] >> (32 - T)) << T);   
      const unsigned nbv = __shfl_sync(0xffffffffu, e, src);
      q0 = tig ? (nbv & TM) : 0u; q1 = tig ? (nbv >> T) : 0u;                      
    }
    const int8_t* xg = reinterpret_cast<const int8_t*>(P.qx) + (size_t)(tok0 + gi) * Kc + (size_t)cur.g * 512 + blk * 256 + 64 * tig;   
    #pragma unroll
    for (int h = 0; h < NST; ++h) {
      const int sidx = (bi * NST + h) & 1;
      if (!XG) {
        pfa_cp_wait(); __syncthreads();                                      
        if (h + 1 < NST) stage(sidx ^ 1, cur.g, blk, h + 1); else if (more) stage(sidx ^ 1, g_next, blk ^ 1, 0);
      }
      if (active) {
        #pragma unroll
        for (int ms = 0; ms < SS; ++ms) {
          const int m = h * SS + ms;
          const unsigned f0 = pfa_field<K>(w, m), f1 = pfa_field<K>(w + K, m);             
          const unsigned p10 = f0 & PM, p20 = f0 >> (2 * K), p11 = f1 & PM, p21 = f1 >> (2 * K);
          const uint2 b0 = make_uint2(pair_b(p10, q0), pair_b(p20, p10 >> TS));    
          const uint2 b1 = make_uint2(pair_b(p11, q1), pair_b(p21, p11 >> TS));    
          q0 = p20 >> TS; q1 = p21 >> TS;
          #pragma unroll
          for (int j = 0; j < NF; ++j) {                                           
            unsigned a[4];
            if (XG) {                                                              
              const uint2 xa = __ldg(reinterpret_cast<const uint2*>(xg + (size_t)(16 * j) * Kc + 8 * m));
              const uint2 xc = __ldg(reinterpret_cast<const uint2*>(xg + (size_t)(16 * j + 8) * Kc + 8 * m));
              a[0] = xa.x; a[1] = xc.x; a[2] = xa.y; a[3] = xc.y;
            } else {
              const uint4 v = s_x[XG ? 0 : sidx][XG ? 0 : ms][XG ? 0 : j][lane];
              a[0] = v.x; a[1] = v.z; a[2] = v.y; a[3] = v.w;
            }
            pfa_mma(d[j][0], a, b0); pfa_mma(d[j][1], a, b1);
          }
        }
      }
    }
    if constexpr (XSM) {
      if (active && blk == 1) {                                                    
        #pragma unroll
        for (int j = 0; j < NF; ++j) {
          if (j < nfr) {
            const int tA = tok0 + 16 * j + gi, tB = tA + 8;
            
            const int AA = tA < M ? __float2int_rn(__fmul_rn(__ldg(P.alpha + (size_t)tA * G + cur.g), s_rm[tA - tok0])) : 0;
            const int AB = tB < M ? __float2int_rn(__fmul_rn(__ldg(P.alpha + (size_t)tB * G + cur.g), s_rm[tB - tok0])) : 0;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
              #pragma unroll
              for (int c = 0; c < 4; ++c) { pfa_mad_wide(acc64[j][r][c], c < 2 ? AA : AB, d[j][r][c]); d[j][r][c] = 0; }
            }
          }
        }
      }
    } else
    if (active && blk == 1) {                                                      
      const int ngw = base + (cur.w < rem ? 1 : 0);                                
      const bool multi = ngw > 1, next_multi = cur.w + 1 < nw && base + (cur.w + 1 < rem ? 1 : 0) > 1;
      #pragma unroll
      for (int j = 0; j < NF; ++j) {
        if (j < nfr) {
          const int tA = tok0 + 16 * j + gi, tB = tA + 8;
          const float aA = tA < M ? __ldg(P.alpha + (size_t)tA * G + cur.g) : 0.f, aB = tB < M ? __ldg(P.alpha + (size_t)tB * G + cur.g) : 0.f;
          #pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int n = tile * 16 + 8 * r + 2 * tig;
            float* oA = dst + (size_t)(tA < M ? tA : 0) * N + n; float* oB = dst + (size_t)(tB < M ? tB : 0) * N + n;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
              const float x = __fmul_rn(c < 2 ? aA : aB, (float)d[j][r][c]);
              d[j][r][c] = 0;
              if constexpr (FOLD == 2) {
                
                if (multi) pacc[j][r][c] = __fadd_rn(pacc[j][r][c], x);
                else { const float p = __fadd_rn(0.f, x); pacc[j][r][c] = pfa_fold(cur.w == 0, pacc[j][r][c], p); }
              } else {
                pacc[j][r][c] = __fadd_rn(pacc[j][r][c], x);
                if constexpr (FOLD == 1) { if (cur.last) { tot[j][r][c] = pfa_fold(cur.w == 0, tot[j][r][c], pacc[j][r][c]); pacc[j][r][c] = 0.f; } }
              }
            }
            if constexpr (FOLD == 2) {
              if (cur.last) {
                if (multi) {                                                       
                  if (cur.w == 0) {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) pacc[j][r][c] = pfa_fold(true, 0.f, pacc[j][r][c]);
                  } else {
                    if (tA < M) { pacc[j][r][0] = pfa_fold(false, oA[0], pacc[j][r][0]); pacc[j][r][1] = pfa_fold(false, oA[1], pacc[j][r][1]); }
                    if (tB < M) { pacc[j][r][2] = pfa_fold(false, oB[0], pacc[j][r][2]); pacc[j][r][3] = pfa_fold(false, oB[1], pacc[j][r][3]); }
                  }
                }
                if (next_multi) {                                                  
                  if (tA < M) { oA[0] = pacc[j][r][0]; oA[1] = pacc[j][r][1]; }
                  if (tB < M) { oB[0] = pacc[j][r][2]; oB[1] = pacc[j][r][3]; }
                  #pragma unroll
                  for (int c = 0; c < 4; ++c) pacc[j][r][c] = 0.f;
                }
              }
            }
          }
        }
      }
    }
  }
  if (!active || nblk == 0) return;
  #pragma unroll
  for (int j = 0; j < NF; ++j) {
    if (j < nfr) {
      const int tA = tok0 + 16 * j + gi, tB = tA + 8;
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int n = tile * 16 + 8 * r + 2 * tig;                                 
        if constexpr (XSM) {                                                       
          const bool wr = P.xsm_S != nullptr;                                     
          if (tA < M) { const float us = s_us[tA - tok0], rr = s_rr[tA - tok0]; float* o = dst + (size_t)tA * N + n;
            float y0 = __fmul_rn(__ll2float_rn(acc64[j][r][0]), us), y1 = __fmul_rn(__ll2float_rn(acc64[j][r][1]), us);
            if (wr) { y0 = __fmul_rn(y0, rr); y1 = __fmul_rn(y1, rr); }
            o[0] = y0; o[1] = y1; }
          if (tB < M) { const float us = s_us[tB - tok0], rr = s_rr[tB - tok0]; float* o = dst + (size_t)tB * N + n;
            float y0 = __fmul_rn(__ll2float_rn(acc64[j][r][2]), us), y1 = __fmul_rn(__ll2float_rn(acc64[j][r][3]), us);
            if (wr) { y0 = __fmul_rn(y0, rr); y1 = __fmul_rn(y1, rr); }
            o[0] = y0; o[1] = y1; }
        } else {
        float v[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) { if constexpr (FOLD == 1) v[c] = tot[j][r][c]; else v[c] = pacc[j][r][c]; }
        if (tA < M) { float* o = dst + (size_t)tA * N + n; o[0] = v[0]; o[1] = v[1]; }
        if (tB < M) { float* o = dst + (size_t)tB * N + n; o[0] = v[2]; o[1] = v[3]; }
        }
      }
    }
  }
}


static int pfa_k(const GemvLaunch& P, bool xsm = false) {           
  if (P.draft || P.e_fuse || (P.xsum && !xsm) || P.M < 1 || P.n_seg < 1) return 0;
  const int K = P.seg[0].t.k_default; if (K != 4 && K != 6 && K != 8) return 0;
  for (int s = 0; s < P.n_seg; ++s) { const TensorDev& t = P.seg[s].t;
    if (!t.lno || t.k_default != K || t.tile_k || t.group_worker || (t.N % 16) || (t.Kc % 512) || t.n_workers < 1 || t.n_workers > t.Kc / 512) return 0; }
  return K;
}
bool pfa_eligible(const GemvLaunch& P) { return pfa_k(P, P.xsum != 0) != 0; }
template <int K, int LNF, int FOLD, bool XG, int LB, bool XSM = false> static void launch_pfa_t(const GemvLaunch& P0, int wpc, cudaStream_t st) {
  if (wpc < 1) wpc = 1;
  if (wpc * 32 > pfa_maxt<LB>()) wpc = pfa_maxt<LB>() / 32;
  GemvLaunch P = P0; int ctas = 0;                                     
  for (int s = 0; s < P.n_seg; ++s) { P.seg[s].cta_begin = ctas; ctas += (P.seg[s].t.N / 16 + wpc - 1) / wpc; }
  const dim3 grid((P.M + (16 << LNF) - 1) / (16 << LNF), ctas, FOLD ? 1 : P.max_workers), block(wpc * 32);   
  launch_k(pfa_kernel<K, LNF, FOLD, XG, LB, XSM>, grid, block, 0, st, P);
}








static int pfa_auto_wpc(const GemvLaunch& P, int tile_tokens, int cap) {
  static int sms = 0;
  if (!sms) { int dev = 0; cudaDeviceProp prop; sms = (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) ? prop.multiProcessorCount : 170; }
  long long tasks = 0; for (int s = 0; s < P.n_seg; ++s) tasks += (long long)(P.seg[s].t.N / 16) * ((P.M + tile_tokens - 1) / tile_tokens);   
  const long long w = (tasks + 2 * sms - 1) / (2 * sms);              
  return (int)(w < 1 ? 1 : w > cap ? cap : w);
}
void launch_pfa(const GemvLaunch& P, int mode, int wpc, cudaStream_t st) {
  if (P.xsum) {                                                      
    const int K = pfa_k(P, true); if (!K || !P.us_out) throw std::runtime_error("launch_pfa: XSUM launch not eligible");
    if (wpc <= 0) wpc = pfa_auto_wpc(P, 64, 6);
    switch (K) {
      case 4: if (wpc <= 4) launch_pfa_t<4, 2, 1, false, 3, true>(P, wpc, st); else launch_pfa_t<4, 2, 1, false, 1, true>(P, wpc, st); break;
      case 6: if (wpc <= 4) launch_pfa_t<6, 2, 1, false, 3, true>(P, wpc, st); else launch_pfa_t<6, 2, 1, false, 1, true>(P, wpc, st); break;
      default: launch_pfa_t<8, 2, 1, false, 3, true>(P, wpc, st); break;
    }
    return;
  }
  const int K = pfa_k(P); if (!K) throw std::runtime_error("launch_pfa: launch not eligible");
#define PFA_BY_K(LNF, FOLD, XG, LB) do { switch (K) { case 4: launch_pfa_t<4, LNF, FOLD, XG, LB>(P, wpc, st); break; \
    case 6: launch_pfa_t<6, LNF, FOLD, XG, LB>(P, wpc, st); break; default: launch_pfa_t<8, LNF, FOLD, XG, LB>(P, wpc, st); break; } } while (0)
  if (mode == 0) { if (wpc <= 0) wpc = 8; PFA_BY_K(1, 0, false, 0); }
  else if (mode == 4) { if (wpc <= 0) wpc = 8; PFA_BY_K(1, 0, true, 0); }
  else if (mode == 1) { if (wpc <= 0) wpc = pfa_auto_wpc(P, 64, 6); if (wpc <= 4 || K == 8) PFA_BY_K(2, 1, false, 3); else PFA_BY_K(2, 1, false, 1); }
  else if (mode == 2) { if (wpc <= 0) wpc = pfa_auto_wpc(P, 64, 12); if (wpc <= 6) PFA_BY_K(2, 2, false, 1); else PFA_BY_K(2, 2, false, 4); }
  else if (mode == 3) { if (wpc <= 0) wpc = pfa_auto_wpc(P, 128, 8); if (wpc <= 4) PFA_BY_K(3, 2, false, 3); else PFA_BY_K(3, 2, false, 2); }
  else throw std::runtime_error("launch_pfa: mode must be 0..4");
#undef PFA_BY_K
}
} 
