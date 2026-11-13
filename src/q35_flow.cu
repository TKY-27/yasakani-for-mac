// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <stdexcept>
#include <vector>
#include <map>
#include <tuple>
#include <cuda_bf16.h>
#include "kernels.hpp"
#include "a8_prologue.cuh"
#include "q35.hpp"
#include "q35_flow.hpp"
#include "pm_math.cuh"
#include <cstdio>
#include <algorithm>
namespace yskn {














namespace {
constexpr int FK = 6, FUW = 2, FNU = FK / 2, FT = 12 - FK, FTS = 3 * FK - 12;
constexpr unsigned FPM = (1u << (2 * FK)) - 1u, FTM = (1u << FT) - 1u;
__device__ __forceinline__ float f_bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f_f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float f_rbf(float f) { return f_bf2f(f_f2bf(f)); }
__constant__ int flow_pm_on;                                                                        
__device__ __forceinline__ float f_silu(float x) { return flow_pm_on ? pm_silu(x) : __fdiv_rn(x, __fadd_rn(1.f, expf(-x))); }
template <int K> __device__ __forceinline__ unsigned f_field(const unsigned* w, int i) {
  constexpr int FB = 4 * K; const int bit = FB * i, wi = bit >> 5, sh = bit & 31;
  unsigned v = w[wi] >> sh;
  if (sh + FB > 32) v |= w[wi + 1] << (32 - sh);
  if constexpr (FB >= 32) return v; else return v & ((1u << FB) - 1u);
}
__device__ __forceinline__ void f_mma(int* d, const unsigned* a, uint2 b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ long long f_xsm_A(float alpha, float rm) { return (long long)__float2int_rn(__fmul_rn(alpha, rm)); }
__device__ __forceinline__ float f_xsm_r(const float* S, float eps) {   
  float s = 0.f;
  #pragma unroll
  for (int b = 0; b < Q35_H / 512; ++b) s = __fadd_rn(s, __ldcg(S + b));
  const float a = __fadd_rn(__fdiv_rn(s, (float)Q35_H), eps); return flow_pm_on ? pm_rsqrt(a) : rsqrtf(a); }
__device__ __forceinline__ float f_xsm_us(const float* al, int G) { float m = 0.f; for (int k = 0; k < G; ++k) m = fmaxf(m, __ldcg(al + k)); return ldexpf(m, -24); }   
__device__ __forceinline__ unsigned long long f_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned f_ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

__device__ __forceinline__ void f_fwht(float (&v)[8], float* xs) {
  const int tid = threadIdx.x, lane = tid & 31;
  #pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
    #pragma unroll
    for (int k = 0; k < 8; ++k) { const float q = __shfl_xor_sync(0xffffffffu, v[k], h); v[k] = (lane & h) ? (q - v[k]) : (v[k] + q); }
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) xs[tid + 64 * k] = v[k];                 
  __syncthreads();
  #pragma unroll
  for (int k = 0; k < 8; ++k) { const float q = xs[(tid ^ 32) + 64 * k]; v[k] = (tid & 32) ? (q - v[k]) : (v[k] + q); }
  __syncthreads();
  #pragma unroll
  for (int m = 1; m < 8; m <<= 1) {                                     
    #pragma unroll
    for (int k = 0; k < 8; ++k) if (!(k & m)) { const float lo = v[k], hi = v[k ^ m]; v[k] = lo + hi; v[k ^ m] = lo - hi; }
  }
}

__device__ __forceinline__ void f_a8_tail(float (&b)[8], float* xs, unsigned* s_amax, int8_t* qx, float* alpha) {
  f_fwht(b, xs);
  unsigned mx = 0u; float v[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) { v[k] = round_f16(__fmul_rn(b[k], 0.044194173824159216f)); mx = max(mx, __float_as_uint(fabsf(v[k]))); }
  for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((threadIdx.x & 31) == 0) atomicMax(s_amax, mx);
  __syncthreads();
  const float al = __fdiv_rn(__uint_as_float(*s_amax), 127.f);
  if (threadIdx.x == 0) *alpha = al;
  #pragma unroll
  for (int k = 0; k < 8; ++k) { const float r = al > 0.f ? __fdiv_rn(v[k], al) : 0.f; int q = __float2int_rn(r); q = q < -127 ? -127 : (q > 127 ? 127 : q);
                                qx[threadIdx.x + 64 * k] = (int8_t)q; }
}
}  

struct FlowSched { unsigned* dev = nullptr; int n = 0; };
static std::map<std::tuple<int, int, int, int, int, int>, FlowSched> g_flow_sched;

static const FlowSched& flow_schedule(int M, int ctas_seg, int nwe_gu, int ctas_dn, int nwe_dn, int nblk_gu) {
  auto key = std::make_tuple(M, ctas_seg, nwe_gu, ctas_dn, nwe_dn, nblk_gu); auto it = g_flow_sched.find(key);
  if (it != g_flow_sched.end()) return it->second;
  std::vector<unsigned> s; const int cpb = ctas_seg / nblk_gu;           
  auto fin1 = [&](int j) { for (int t = 0; t < M; ++t) s.push_back((1u << 30) | ((unsigned)j << 8) | (unsigned)t); };
  for (int j = 0; j < nblk_gu; ++j) {                                    
    for (int sg = 0; sg < 2; ++sg) for (int c = j * cpb; c < (j + 1) * cpb; ++c) for (int w = 0; w < nwe_gu; ++w)
      s.push_back(((unsigned)sg << 29) | ((unsigned)c << 16) | (unsigned)w);
    if (j > 0) fin1(j - 1);
  }
  fin1(nblk_gu - 1);
  const int nblk_dn = ctas_dn * FUW * 16 / 512, cpd = ctas_dn / nblk_dn;
  auto fin3 = [&](int b) { for (int t = 0; t < M; ++t) s.push_back((3u << 30) | ((unsigned)b << 8) | (unsigned)t); };
  for (int b = 0; b < nblk_dn; ++b) {
    for (int c = b * cpd; c < (b + 1) * cpd; ++c) for (int w = 0; w < nwe_dn; ++w) s.push_back((2u << 30) | ((unsigned)c << 16) | (unsigned)w);
    if (b > 0) fin3(b - 1);
  }
  fin3(nblk_dn - 1);
  FlowSched f; f.n = (int)s.size(); cudaMalloc(&f.dev, s.size() * 4); cudaMemcpy(f.dev, s.data(), s.size() * 4, cudaMemcpyHostToDevice);
  return g_flow_sched.emplace(key, f).first->second;
}

template <int MINB>
__global__ void __launch_bounds__(64, MINB) q35_flow_mlp_kernel(Q35FlowMlpArgs A, const unsigned* __restrict__ sched, int n_items) {
  __shared__ __align__(16) uint16_t s_lut[4096];                       
  __shared__ unsigned s_item, s_amax; __shared__ float s_red4[4], s_us, s_r;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 2, tig = lane & 3, M = A.M;
  const int src = (lane & ~3) | ((tig + 3) & 3);
  unsigned* const cnt = A.cnt; const int NBG = A.F / 512, FIN = NBG, DN = NBG + 1, CTR = DN + 16;
  const uint16_t* cur_lut = nullptr;
  pdl_wait(); pdl_trigger();
  if (blockIdx.x == 0) A.cnt_next[tid] = 0u;                          
  for (int item = blockIdx.x; item < n_items; item += gridDim.x) {   
    __syncthreads();                                                   
    const unsigned d = __ldg(sched + item); const int type = (int)(d >> 30);
    const unsigned long long t_beg = A.trace ? f_gt() : 0ull;
    if (type == 0 || type == 2) {
      const bool down = type == 2; const int sgi = down ? 0 : (int)((d >> 29) & 1u), cta = (int)((d >> 16) & 0x1FFFu), worker = (int)(d & 0xFFFFu);
      const GemvLaunch& P = down ? A.dn : A.gu; const GemvSeg& Sg = P.seg[sgi]; const TensorDev& t = Sg.t;
      if (down && tid == 0) { unsigned spins = 0; while (f_ld_acq(cnt + FIN) < (unsigned)(NBG * M)) { if (++spins > (1u << 26)) { atomicOr(cnt + CTR + 1, 1u); break; } __nanosleep(64); } }
      if (t.lut8 != cur_lut) { __syncthreads(); for (int i = tid; i < 512; i += 64) reinterpret_cast<uint4*>(s_lut)[i] = __ldg(reinterpret_cast<const uint4*>(t.lut8) + i); cur_lut = t.lut8; }
      __syncthreads();
      if (A.trace && tid == 0) A.trace[4 * item + 1] = f_gt();
      const int tile = cta * FUW + warp, ntiles = t.N / 16, N = t.N, Kc = t.Kc, G = Kc / 512, nw = t.n_workers;
      const bool active = tile < ntiles && worker < nw;
      const int ng = (worker < G && worker < nw) ? (G - worker + nw - 1) / nw : 0, nblk = active ? 2 * ng : 0;
      const uint4* sp = t.lno + (size_t)(active ? tile : 0) * G * 2 * FNU * 32 + lane;
      auto boff = [&](int bi) { return (size_t)((worker + (bi >> 1) * nw) * 2 + (bi & 1)) * (FNU * 32); };
      uint4 nx[FNU];
      #pragma unroll
      for (int u = 0; u < FNU; ++u) nx[u] = nblk ? __ldg(sp + boff(0) + 32 * u) : make_uint4(0u, 0u, 0u, 0u);
      
      constexpr int TM = 8, PP = 32 / TM, NV = 40 / PP;
      const int tkn = lane % TM, part = lane / TM; float m = 0.f;
      if (tkn < M) { const float* al = P.alpha + (size_t)tkn * G; float v[NV];
        #pragma unroll
        for (int k = 0; k < NV; ++k) { const int gx = part + k * PP; v[k] = gx < G ? __ldcg(al + gx) : 0.f; }
        #pragma unroll
        for (int k = 0; k < NV; ++k) m = fmaxf(m, v[k]); }
      #pragma unroll
      for (int o = TM; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float rmv = m > 0.f ? __fdiv_rn(16777216.f, m) : 0.f;
      if (sgi == 0 && cta == 0 && worker == 0 && warp == 0 && part == 0 && tkn < M) P.us_out[tkn] = ldexpf(m, -24);
      float rm[2];
      #pragma unroll
      for (int k = 0; k < 2; ++k) { const int tk = 2 * tig + k; const float x = __shfl_sync(0xffffffffu, rmv, tk & 31); rm[k] = tk < M ? x : 0.f; }
      long long acc64[4] = {0, 0, 0, 0}; int dd[4] = {0, 0, 0, 0};
      auto pair_a = [&](unsigned p, unsigned q) -> unsigned { const unsigned W = q | (p << FT); return unsigned(s_lut[W & 0xFFFu]) | (unsigned(s_lut[(W >> FK) & 0xFFFu]) << 16); };
      for (int bi = 0; bi < nblk; ++bi) {
        const int g = worker + (bi >> 1) * nw, blk = bi & 1;
        unsigned w[4 * FNU];
        #pragma unroll
        for (int u = 0; u < FNU; ++u) { w[4 * u] = nx[u].x; w[4 * u + 1] = nx[u].y; w[4 * u + 2] = nx[u].z; w[4 * u + 3] = nx[u].w; }
        if (bi + 1 < nblk) {
          #pragma unroll
          for (int u = 0; u < FNU; ++u) nx[u] = __ldg(sp + boff(bi + 1) + 32 * u); }
        const int tok = gi; const uint4* xp = tok < M ? reinterpret_cast<const uint4*>(P.qx + (size_t)tok * Kc + (size_t)g * 512 + blk * 256 + 64 * tig) : nullptr;
        uint4 xb;
        const unsigned e = (w[FK - 1] >> (32 - FT)) | ((w[2 * FK - 1] >> (32 - FT)) << FT);
        const unsigned nbv = __shfl_sync(0xffffffffu, e, src);
        unsigned q0 = tig ? (nbv & FTM) : 0u, q1 = tig ? (nbv >> FT) : 0u;
        #pragma unroll
        for (int mm = 0; mm < 8; ++mm) {
          const unsigned f0 = f_field<FK>(w, mm), f1 = f_field<FK>(w + FK, mm);
          const unsigned p10 = f0 & FPM, p20 = f0 >> (2 * FK), p11 = f1 & FPM, p21 = f1 >> (2 * FK);
          unsigned Aa[4];
          Aa[0] = pair_a(p10, q0); Aa[2] = pair_a(p20, p10 >> FTS); Aa[1] = pair_a(p11, q1); Aa[3] = pair_a(p21, p11 >> FTS);
          q0 = p20 >> FTS; q1 = p21 >> FTS;
          if ((mm & 1) == 0) xb = xp ? __ldcg(xp + (mm >> 1)) : make_uint4(0u, 0u, 0u, 0u);
          f_mma(dd, Aa, (mm & 1) ? make_uint2(xb.z, xb.w) : make_uint2(xb.x, xb.y));
        }
        if (blk == 1) {
          const int t0 = 2 * tig, t1 = t0 + 1;
          const float a0 = t0 < M ? __ldcg(P.alpha + (size_t)t0 * G + g) : 0.f, a1 = t1 < M ? __ldcg(P.alpha + (size_t)t1 * G + g) : 0.f;
          const long long A0 = f_xsm_A(a0, rm[0]), A1 = f_xsm_A(a1, rm[1]);
          acc64[0] += A0 * (long long)dd[0]; acc64[2] += A0 * (long long)dd[2];
          acc64[1] += A1 * (long long)dd[1]; acc64[3] += A1 * (long long)dd[3];
          dd[0] = dd[1] = dd[2] = dd[3] = 0;
        }
      }
      if (nblk) {
        const int n = tile * 16 + gi, t0 = 2 * tig, t1 = t0 + 1; unsigned long long* p64 = reinterpret_cast<unsigned long long*>(Sg.part);
        if (t0 < M) { atomicAdd(p64 + (size_t)t0 * N + n, (unsigned long long)acc64[0]); atomicAdd(p64 + (size_t)t0 * N + n + 8, (unsigned long long)acc64[2]); }
        if (t1 < M) { atomicAdd(p64 + (size_t)t1 * N + n, (unsigned long long)acc64[1]); atomicAdd(p64 + (size_t)t1 * N + n + 8, (unsigned long long)acc64[3]); }
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(cnt + (down ? DN + tile / 32 : tile / 32), 1u);
      }
      if (A.trace && tid == 0) { A.trace[4 * item] = t_beg; A.trace[4 * item + 3] = blockIdx.x; A.trace[4 * item + 2] = f_gt(); }
    } else if (type == 1) {                                              
      const int j = (int)((d >> 8) & 0xFFu), tk = (int)(d & 0xFFu);
      const unsigned need = (unsigned)(2 * 32 * A.nwe_gu);
      if (tid == 0) { unsigned spins = 0; while (f_ld_acq(cnt + j) < need) { if (++spins > (1u << 26)) { atomicOr(cnt + CTR + 1, 2u); break; } __nanosleep(32); }
                      s_us = f_xsm_us(A.gu.alpha + (size_t)tk * (A.gu.seg[0].t.Kc / 512), A.gu.seg[0].t.Kc / 512); s_r = f_xsm_r(A.S + (size_t)tk * Q35_SST, A.eps); s_amax = 0u; }
      __syncthreads();
      if (A.trace && tid == 0) A.trace[4 * item + 1] = f_gt();
      const float us = s_us, r = s_r; float vg[8], vu[8];
      long long* const pg = reinterpret_cast<long long*>(A.gu.seg[0].part); long long* const pu = reinterpret_cast<long long*>(A.gu.seg[1].part);
      #pragma unroll
      for (int k = 0; k < 8; ++k) { const int n = j * 512 + tid + 64 * k; const size_t o = (size_t)tk * A.F + n;
        const long long ag = __ldcg(pg + o), au = __ldcg(pu + o); __stcg(pg + o, 0ll); __stcg(pu + o, 0ll);
        vg[k] = __fmul_rn(__fmul_rn(xsum_y(ag, us), r), __ldg(A.tau_g + n)); vu[k] = __fmul_rn(__fmul_rn(xsum_y(au, us), r), __ldg(A.tau_u + n)); }
      float* const xs = reinterpret_cast<float*>(s_lut); cur_lut = nullptr;
      f_fwht(vg, xs); f_fwht(vu, xs);
      float b[8];
      #pragma unroll
      for (int k = 0; k < 8; ++k) { const int n = j * 512 + tid + 64 * k;
        const float g = f_rbf(__fmul_rn(__fmul_rn(vg[k], 0.044194173824159216f), (float)__ldg(A.sg_g + n)));
        const float u = f_rbf(__fmul_rn(__fmul_rn(vu[k], 0.044194173824159216f), (float)__ldg(A.sg_u + n)));
        const float x = f_rbf(__fmul_rn(f_rbf(f_silu(g)), u));
        b[k] = __fmul_rn(x, (float)__ldg(A.sg_dn_in + n)); }
      f_a8_tail(b, xs, &s_amax, A.qxd + (size_t)tk * A.F + j * 512, A.ald + (size_t)tk * NBG + j);
      __threadfence(); __syncthreads();
      if (tid == 0) atomicAdd(cnt + FIN, 1u);
      if (A.trace && tid == 0) { A.trace[4 * item] = t_beg; A.trace[4 * item + 3] = blockIdx.x; A.trace[4 * item + 2] = f_gt(); }
    } else {                                                             
      const int bb = (int)((d >> 8) & 0xFFu), tk = (int)(d & 0xFFu);
      const unsigned need = (unsigned)(32 * A.nwe_dn);
      if (tid == 0) { unsigned spins = 0; while (f_ld_acq(cnt + DN + bb) < need) { if (++spins > (1u << 26)) { atomicOr(cnt + CTR + 1, 4u); break; } __nanosleep(32); }
                      s_us = f_xsm_us(A.ald + (size_t)tk * NBG, NBG); s_amax = 0u; }
      __syncthreads();
      if (A.trace && tid == 0) A.trace[4 * item + 1] = f_gt();
      const float us = s_us; float v[8]; long long* const pd = reinterpret_cast<long long*>(A.dn.seg[0].part);
      #pragma unroll
      for (int k = 0; k < 8; ++k) { const int n = bb * 512 + tid + 64 * k; const size_t o = (size_t)tk * Q35_H + n;
        const long long acc = __ldcg(pd + o); __stcg(pd + o, 0ll); v[k] = __fmul_rn(xsum_y(acc, us), __ldg(A.tau_dn + n)); }
      float* const xs = reinterpret_cast<float*>(s_lut); cur_lut = nullptr;
      f_fwht(v, xs);
      float hn[8], acc0 = 0.f, acc1 = 0.f;
      #pragma unroll
      for (int k = 0; k < 8; ++k) { const int n = bb * 512 + tid + 64 * k; float* const hp = A.h + (size_t)tk * Q35_H + n;
        const float y = __fmul_rn(__fmul_rn(v[k], 0.044194173824159216f), (float)__ldg(A.sg_dn_out + n));
        hn[k] = f_rbf(__fadd_rn(__ldcg(hp), f_rbf(y))); *hp = hn[k]; }
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) { acc0 = __fadd_rn(acc0, __fmul_rn(hn[2 * jj], hn[2 * jj])); acc1 = __fadd_rn(acc1, __fmul_rn(hn[2 * jj + 1], hn[2 * jj + 1])); }
      for (int o = 16; o; o >>= 1) { acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, o)); acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, o)); }
      if (lane == 0) { s_red4[warp] = acc0; s_red4[2 + warp] = acc1; }
      __syncthreads();
      if (tid == 0) { float sb = 0.f;
        #pragma unroll
        for (int w4 = 0; w4 < 4; ++w4) sb = __fadd_rn(sb, s_red4[w4]);
        A.S_out[(size_t)tk * Q35_SST + bb] = sb; }
      float b[8];
      #pragma unroll
      for (int k = 0; k < 8; ++k) { const int n = bb * 512 + tid + 64 * k;
        const float u = __fmul_rn(hn[k], __fadd_rn(1.f, f_bf2f(__ldg(A.w + n))));
        if (A.xu) A.xu[(size_t)tk * Q35_H + n] = f_f2bf(u);
        b[k] = __fmul_rn(u, (float)__ldg(A.sg_next + n)); }
      f_a8_tail(b, xs, &s_amax, A.qx + (size_t)tk * Q35_H + bb * 512, A.alpha + (size_t)tk * (Q35_H / 512) + bb);
      __syncthreads();
      if (A.trace && tid == 0) { A.trace[4 * item] = t_beg; A.trace[4 * item + 3] = blockIdx.x; A.trace[4 * item + 2] = f_gt(); }
    }
  }
}

static int g_flow_grid = 0;

static const FlowSched& flow_prep(Q35FlowMlpArgs& A) {
  const int ctas_seg = A.gu.seg[0].t.N / 16 / FUW, ctas_dn = A.dn.seg[0].t.N / 16 / FUW;
  const int G_gu = A.gu.seg[0].t.Kc / 512, G_dn = A.dn.seg[0].t.Kc / 512;
  A.nwe_gu = std::min(A.gu.seg[0].t.n_workers, G_gu); A.nwe_dn = std::min(A.dn.seg[0].t.n_workers, G_dn);
  if (A.gu.seg[1].t.n_workers != A.gu.seg[0].t.n_workers || A.gu.seg[1].t.N != A.gu.seg[0].t.N || A.M > 8 || A.F / 512 + 1 + 16 + 2 > 64)
    throw std::runtime_error("flow: unsupported shape");
  if (!g_flow_grid) { int sms = 0, per = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, q35_flow_mlp_kernel<10>, 64, 0); g_flow_grid = sms * per;
    std::fprintf(stderr, "[flow] persistent grid %d CTAs (%d per SM)\n", g_flow_grid, per); }
  const FlowSched& f = flow_schedule(A.M, ctas_seg, A.nwe_gu, ctas_dn, A.nwe_dn, A.F / 512);
  static int printed[9] = {}; if (!printed[A.M]) { printed[A.M] = 1; std::fprintf(stderr, "[flow] M %d: gate/up %d CTAs x %d workers (G %d), down %d CTAs x %d workers (G %d), %d items\n",
    A.M, 2 * ctas_seg, A.nwe_gu, G_gu, ctas_dn, A.nwe_dn, G_dn, f.n); }
  return f;
}
void prepare_q35_flow_mlp(const Q35FlowMlpArgs& A0) { Q35FlowMlpArgs A = A0; (void)flow_prep(A); }
void launch_q35_flow_mlp(const Q35FlowMlpArgs& A0, cudaStream_t st) {
  Q35FlowMlpArgs A = A0;
  const FlowSched& S = flow_prep(A);
  launch_k(q35_flow_mlp_kernel<10>, dim3(g_flow_grid), dim3(64), 0, st, A, (const unsigned*)S.dev, S.n);
}
void q35_flow_set_pm(int on, cudaStream_t st) { cudaMemcpyToSymbolAsync(flow_pm_on, &on, sizeof(int), 0, cudaMemcpyHostToDevice, st); }
}  
