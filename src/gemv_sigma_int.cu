// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include <stdexcept>
#include <algorithm>
#include "v2_decode.cuh"
#include "a8_prologue.cuh"
#include "kernels.hpp"
#include "pdl.cuh"
namespace yskn {






constexpr int WARPS = GEMV_WARPS, REP = 1, NT_PASS = GEMV_NT_PASS, STAGES = GEMV_STAGES, MINB = 16 / GEMV_WARPS > 0 ? 16 / GEMV_WARPS : 1;   

__device__ __forceinline__ int worker_of(const TensorDev& t, int g) { return t.group_worker ? (int)t.group_worker[g] : (t.n_workers > 0 ? g % t.n_workers : 0); }
__device__ __forceinline__ float mul_add_rn(float acc, float a, float b) { return __fadd_rn(acc, __fmul_rn(a, b)); }   





__shared__ __align__(16) uint16_t g_lut8[4096 * REP];
__device__ __forceinline__ uint32_t lut_at(uint32_t off) { return (uint32_t)*reinterpret_cast<const uint16_t*>(reinterpret_cast<const unsigned char*>(g_lut8) + off); }
template <int K> __device__ __forceinline__ uint32_t yq_off(const UnitQ<K>& u, int i) {     
  const int e = K * (i + 1), word = e >> 5, sh = e & 31;
  if (sh + L_BITS <= 32) return sh >= 1 ? (u.w[word] >> (sh - 1)) & 0x1FFEu : (u.w[word] << 1) & 0x1FFEu;
  return __funnelshift_r(u.w[word], u.w[word + 1], sh - 1) & 0x1FFEu;                         
}
template <int K> __device__ __forceinline__ uint2 wfrag_q(const UnitQ<K>& u, int m) {        
  const uint32_t a0 = lut_at(yq_off<K>(u, 4 * m)),     a1 = lut_at(yq_off<K>(u, 4 * m + 1));
  const uint32_t b0 = lut_at(yq_off<K>(u, 4 * m + 2)), b1 = lut_at(yq_off<K>(u, 4 * m + 3));
  return make_uint2(a0 | (a1 << 16), b0 | (b1 << 16));
}


template <int MT> __device__ __forceinline__ void load_afrag_q(uint32_t* a, const int8_t* __restrict__ qx, int Kc, int base, int m, int lane, int M) {
  const int r = lane >> 2, col = base + 64 * (lane & 3) + 8 * m; a[0] = a[1] = a[2] = a[3] = 0u;
  if (r < M)               { const uint2 v = __ldg(reinterpret_cast<const uint2*>(qx + (size_t)r * Kc + col));       a[0] = v.x; a[2] = v.y; }
  if (MT > 8 && r + 8 < M) { const uint2 v = __ldg(reinterpret_cast<const uint2*>(qx + (size_t)(r + 8) * Kc + col)); a[1] = v.x; a[3] = v.y; }
}

template <int MT> __device__ __forceinline__ void load_afrag_q8(uint32_t (&af)[8][4], const int8_t* __restrict__ qx, int Kc, int base, int lane, int M) {
  const int r = lane >> 2, col = base + 64 * (lane & 3);
  #pragma unroll
  for (int m = 0; m < 8; ++m) af[m][0] = af[m][1] = af[m][2] = af[m][3] = 0u;
  if (r < M) {
    const uint4* p = reinterpret_cast<const uint4*>(qx + (size_t)r * Kc + col);
    #pragma unroll
    for (int i = 0; i < 4; ++i) { const uint4 v = __ldg(p + i); af[2 * i][0] = v.x; af[2 * i][2] = v.y; af[2 * i + 1][0] = v.z; af[2 * i + 1][2] = v.w; }
  }
  if (MT > 8 && r + 8 < M) {
    const uint4* p = reinterpret_cast<const uint4*>(qx + (size_t)(r + 8) * Kc + col);
    #pragma unroll
    for (int i = 0; i < 4; ++i) { const uint4 v = __ldg(p + i); af[2 * i][1] = v.x; af[2 * i][3] = v.y; af[2 * i + 1][1] = v.z; af[2 * i + 1][3] = v.w; }
  }
}


__device__ __forceinline__ float xsum_us(const float* __restrict__ al, int G) { float m = 0.f; for (int k = 0; k < G; ++k) m = fmaxf(m, __ldg(al + k)); return ldexpf(m, -24); }

__device__ __forceinline__ float xsum_rm(const float* __restrict__ al, int G) { float m = 0.f; for (int k = 0; k < G; ++k) m = fmaxf(m, __ldg(al + k)); return m > 0.f ? __fdiv_rn(16777216.f, m) : 0.f; }
__device__ __forceinline__ long long xsum_A(float alpha, float rm) { return (long long)__float2int_rn(__fmul_rn(alpha, rm)); }




__device__ __noinline__ void xsum_down_epilogue(const GemvLaunch& P, const TensorDev& t, const float* part, int b, float* buf, float* s_red, unsigned* s_amax) {
  const int tid = threadIdx.x, N = t.N, G = t.Kc / 512;
  const long long* p64 = reinterpret_cast<const long long*>(part);
  for (int tok = 0; tok < P.M; ++tok) {
    const float us = xsum_us(P.alpha + (size_t)tok * G, G);
    for (int k = tid; k < 512; k += 128) { const int n = b * 512 + k;
      long long* acc = const_cast<long long*>(p64) + (size_t)tok * N + n; const long long s = __ldcg(acc); __stcg(acc, 0ll);   
      buf[k] = __fmul_rn(xsum_y(s, us), P.e_tau[n]); }
    __syncthreads();
    for (int hh = 1; hh < 512; hh <<= 1) {                            
      for (int k = tid; k < 256; k += 128) { const int lo = ((k & ~(hh - 1)) << 1) | (k & (hh - 1)); const float a = buf[lo], c = buf[lo + hh]; buf[lo] = a + c; buf[lo + hh] = a - c; }
      __syncthreads();
    }
    for (int k = tid; k < 512; k += 128) { const int n = b * 512 + k;
      const float y = __fmul_rn(__fmul_rn(buf[k], 0.044194173824159216f), (float)P.e_sg_out[n]);
      float* hp = P.e_h + (size_t)tok * N + n; const float hn = round_f16(__fadd_rn(*hp, round_f16(y))); *hp = hn; buf[k] = hn; }
    __syncthreads();
    const float sb = xsum_block_ss(buf, s_red);
    if (tid == 0) P.e_S[(size_t)tok * 8 + b] = sb;
    for (int k = tid; k < 512; k += 128) { const int i = b * 512 + k;
      const float u = P.e_u16 ? round_f16(__fmul_rn(buf[k], h2f(P.e_gamma[i]))) : __fmul_rn(buf[k], h2f(P.e_gamma[i]));
      buf[k] = __fmul_rn(u, (float)P.e_sg_in[i]); }
    if (tid == 0) *s_amax = 0u;
    __syncthreads();
    for (int hh = 1; hh < 512; hh <<= 1) {                            
      for (int k = tid; k < 256; k += 128) { const int lo = ((k & ~(hh - 1)) << 1) | (k & (hh - 1)); const float a = buf[lo], c = buf[lo + hh]; buf[lo] = a + c; buf[lo + hh] = a - c; }
      __syncthreads();
    }
    for (int k = tid; k < 512; k += 128) { const float v = round_f16(__fmul_rn(buf[k], 0.044194173824159216f)); buf[k] = v; atomicMax(s_amax, __float_as_uint(fabsf(v))); }
    __syncthreads();
    const float al = __fdiv_rn(__uint_as_float(*s_amax), 127.f);
    if (tid == 0) P.e_alpha[(size_t)tok * 7 + b] = al;
    for (int k = tid; k < 512; k += 128) {
      const float v = al > 0.f ? __fdiv_rn(buf[k], al) : 0.f; int q = __float2int_rn(v); q = q < -127 ? -127 : (q > 127 ? 127 : q);
      P.e_qx[(size_t)tok * 3584 + b * 512 + k] = (int8_t)q; }
    __syncthreads();
  }
}



template <int MT> __device__ __forceinline__ void load_bfrag_x8(uint32_t (&bx)[8][MT > 8 ? 2 : 1][2], const int8_t* __restrict__ qx, int Kc, int base, int lane, int M) {
  constexpr int NB = MT > 8 ? 2 : 1;
  const int r = lane >> 2, col = base + 64 * (lane & 3);
  #pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    #pragma unroll
    for (int m = 0; m < 8; ++m) bx[m][nb][0] = bx[m][nb][1] = 0u;
    const int tok = r + 8 * nb;
    if (tok < M) {
      const uint4* p = reinterpret_cast<const uint4*>(qx + (size_t)tok * Kc + col);
      #pragma unroll
      for (int i = 0; i < 4; ++i) { const uint4 v = __ldg(p + i); bx[2 * i][nb][0] = v.x; bx[2 * i][nb][1] = v.y; bx[2 * i + 1][nb][0] = v.z; bx[2 * i + 1][nb][1] = v.w; }
    }
  }
}






template <int K, int MT, bool XSM>
__device__ void warp_tile_t(const TensorDev& t, const uint4* tbase, int tile, int worker, int M, const int8_t* __restrict__ qx,
                            const float* __restrict__ alpha, unsigned char* ring, int stage_bytes, float* part, int lane, bool first_issued) {
  const int Kc = t.Kc, G = Kc / 512, N = t.N, r = lane >> 2, g = lane & 3;
  const bool rr = t.group_worker == nullptr; const int nw = t.n_workers;
  int ng = 0; if (rr) ng = worker < G ? (G - worker + nw - 1) / nw : 0; else for (int gg = 0; gg < G; ++gg) ng += (worker_of(t, gg) == worker);
  if (ng == 0) return;
  auto half_ks = [&](int hi) -> int { const int want = hi >> 1; if (rr) return 2 * (worker + want * nw) + (hi & 1); int cnt = 0;
    for (int gg = 0; gg < G; ++gg) if (worker_of(t, gg) == worker) { if (cnt == want) return 2 * gg + (hi & 1); ++cnt; } return 0; };
  const int H = 2 * ng, total = (8 / NT_PASS) * H;
  
  auto issue = [&](int pn, int hn, unsigned char* dst) { issue_stage_nocommit(tbase, K, half_ks(hn), pn * NT_PASS, NT_PASS, dst, lane); cp_commit(); };
  int pn = 0, hn = 0;                                              
  auto adv = [&](int& p_, int& h_) { if (++h_ == H) { h_ = 0; ++p_; } };
  if (!first_issued) issue(0, 0, ring);
  adv(pn, hn);
  #pragma unroll
  for (int k = 1; k < STAGES - 1; ++k) { if (k < total) issue(pn, hn, ring + k * stage_bytes); else cp_commit(); adv(pn, hn); }   
  constexpr int MI = NT_PASS / 2, NB = MT > 8 ? 2 : 1;
  static_assert(NT_PASS % 2 == 0, "G32 pairs n-tiles into 16-row mma tiles");
  float pacc[MI][NB][4]; int32_t acc[MI][NB][4];
  long long acc64[XSM ? MI : 1][XSM ? NB : 1][4];                  
  float us[NB][2];                                                  
  if (XSM) {
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb)
      #pragma unroll
      for (int k = 0; k < 2; ++k) { const int tk = 8 * nb + 2 * g + k; us[nb][k] = tk < M ? xsum_rm(alpha + (size_t)tk * G, G) : 0.f; }
  }
  int pass = 0, hi = 0;
  for (int s = 0; s < total; ++s, adv(pass, hi)) {
    const int j0 = pass * NT_PASS;
    if (hi == 0) {
      #pragma unroll
      for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
          if (XSM) { acc64[i][nb][0] = acc64[i][nb][1] = acc64[i][nb][2] = acc64[i][nb][3] = 0; }
          else pacc[i][nb][0] = pacc[i][nb][1] = pacc[i][nb][2] = pacc[i][nb][3] = 0.f; }
    }
    { const int sn = s + STAGES - 1; if (sn < total) issue(pn, hn, ring + (sn % STAGES) * stage_bytes); else cp_commit(); adv(pn, hn); }   
    cp_wait<STAGES - 1>();                                         
    __syncwarp();
    const int ks = half_ks(hi);
    if ((hi & 1) == 0) {
      #pragma unroll
      for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) acc[i][nb][0] = acc[i][nb][1] = acc[i][nb][2] = acc[i][nb][3] = 0;
    }
    uint32_t bx[8][NB][2];
    load_bfrag_x8<MT>(bx, qx, Kc, ks * 256, lane, M);
    const unsigned char* stage = ring + (s % STAGES) * stage_bytes;
    UnitQ<K> u[NT_PASS];
    #pragma unroll
    for (int j = 0; j < NT_PASS; ++j) unitq_from_ring<K>(u[j], stage, j, r, g);
    #pragma unroll
    for (int m = 0; m < 8; ++m)
      #pragma unroll
      for (int i = 0; i < MI; ++i) {
        const uint2 w0 = wfrag_q<K>(u[2 * i], m), w1 = wfrag_q<K>(u[2 * i + 1], m);
        const uint32_t a[4] = {w0.x, w1.x, w0.y, w1.y};
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) mma_s8_16832(acc[i][nb], a, make_uint2(bx[m][nb][0], bx[m][nb][1]));
      }
    if (hi & 1) {                                                   
      const int grp = ks >> 1;
      #pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const int t0 = 8 * nb + 2 * g, t1 = t0 + 1;
        const float a0 = t0 < M ? __ldg(alpha + (size_t)t0 * G + grp) : 0.f, a1 = t1 < M ? __ldg(alpha + (size_t)t1 * G + grp) : 0.f;
        if (XSM) {                                                  
          const long long A0 = xsum_A(a0, us[nb][0]), A1 = xsum_A(a1, us[nb][1]);
          #pragma unroll
          for (int i = 0; i < MI; ++i) {
            acc64[i][nb][0] += A0 * (long long)acc[i][nb][0]; acc64[i][nb][2] += A0 * (long long)acc[i][nb][2];
            acc64[i][nb][1] += A1 * (long long)acc[i][nb][1]; acc64[i][nb][3] += A1 * (long long)acc[i][nb][3];
          }
          continue;
        }
        #pragma unroll
        for (int i = 0; i < MI; ++i) {
          if (t0 < M) { pacc[i][nb][0] = mul_add_rn(pacc[i][nb][0], a0, (float)acc[i][nb][0]); pacc[i][nb][2] = mul_add_rn(pacc[i][nb][2], a0, (float)acc[i][nb][2]); }
          if (t1 < M) { pacc[i][nb][1] = mul_add_rn(pacc[i][nb][1], a1, (float)acc[i][nb][1]); pacc[i][nb][3] = mul_add_rn(pacc[i][nb][3], a1, (float)acc[i][nb][3]); }
        }
      }
    }
    if (hi == H - 1) {
      #pragma unroll
      for (int i = 0; i < MI; ++i) {
        const int n = tile * 64 + (j0 + 2 * i) * 8 + r;
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
          const int t0 = 8 * nb + 2 * g, t1 = t0 + 1;
          if (XSM) { unsigned long long* p64 = reinterpret_cast<unsigned long long*>(part);   
            if (t0 < M && n < N) { unsigned long long* d = p64 + (size_t)t0 * N + n; atomicAdd(d, (unsigned long long)acc64[i][nb][0]); atomicAdd(d + 8, (unsigned long long)acc64[i][nb][2]); }
            if (t1 < M && n < N) { unsigned long long* d = p64 + (size_t)t1 * N + n; atomicAdd(d, (unsigned long long)acc64[i][nb][1]); atomicAdd(d + 8, (unsigned long long)acc64[i][nb][3]); }
            continue; }
          if (t0 < M && n < N) { float* d = part + ((size_t)worker * M + t0) * N + n; d[0] = pacc[i][nb][0]; d[8] = pacc[i][nb][2]; }
          if (t1 < M && n < N) { float* d = part + ((size_t)worker * M + t1) * N + n; d[0] = pacc[i][nb][1]; d[8] = pacc[i][nb][3]; }
        }
      }
    }
    __syncwarp();
  }
}

template <int K, int MT, bool DRAFT>
__device__ void warp_tile(const TensorDev& t, const uint4* tbase, int tile, int worker, int M, const int8_t* __restrict__ qx,
                          const float* __restrict__ alpha, unsigned char* ring, int stage_bytes, const Lut8<REP>& lut, float* part, int lane, bool first_issued, const uint16_t* s_r8 = nullptr) {
  const int Kc = t.Kc, G = Kc / 512, N = t.N, r = lane >> 2, g = lane & 3;
  
  const bool rr = t.group_worker == nullptr; const int nw = t.n_workers;
  int ng = 0; if (rr) ng = worker < G ? (G - worker + nw - 1) / nw : 0; else for (int gg = 0; gg < G; ++gg) ng += (worker_of(t, gg) == worker);
  if (ng == 0) return;
  auto half_ks = [&](int hi) -> int { const int want = hi >> 1; if (rr) return 2 * (worker + want * nw) + (hi & 1); int cnt = 0;
    for (int gg = 0; gg < G; ++gg) if (worker_of(t, gg) == worker) { if (cnt == want) return 2 * gg + (hi & 1); ++cnt; } return 0; };
  const int H = 2 * ng, total = (8 / NT_PASS) * H;          
  auto issue = [&](int s, unsigned char* dst) { issue_stage_nocommit(tbase, K, half_ks(s % H), (s / H) * NT_PASS, NT_PASS, dst, lane); cp_commit(); };
  if (!first_issued) issue(0, ring);
  float pacc[NT_PASS][4]; int32_t acc[NT_PASS][4];
  float pd[NT_PASS][4]; int32_t ad[NT_PASS][4];                  
  const int MD = DRAFT ? M + 1 : M;                                
  for (int s = 0; s < total; ++s) {
    const int pass = s / H, hi = s % H, j0 = pass * NT_PASS;
    if (hi == 0) {
      #pragma unroll
      for (int j = 0; j < NT_PASS; ++j) pacc[j][0] = pacc[j][1] = pacc[j][2] = pacc[j][3] = 0.f;
      if (DRAFT) {
        #pragma unroll
        for (int j = 0; j < NT_PASS; ++j) pd[j][0] = pd[j][1] = pd[j][2] = pd[j][3] = 0.f;
      }
    }
    if (s + 1 < total) { issue(s + 1, ring + ((s + 1) & 1) * stage_bytes); cp_wait<1>(); }
    else cp_wait<0>();
    __syncwarp();
    const int ks = half_ks(hi);
    if ((hi & 1) == 0) {
      #pragma unroll
      for (int j = 0; j < NT_PASS; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
      if (DRAFT) {
        #pragma unroll
        for (int j = 0; j < NT_PASS; ++j) ad[j][0] = ad[j][1] = ad[j][2] = ad[j][3] = 0;
      }
    }
    uint32_t af[8][4];
    load_afrag_q8<MT>(af, qx, Kc, ks * 256, lane, M);
    const unsigned char* stage = ring + (s & 1) * stage_bytes;
    UnitQ<K> u[NT_PASS];
    #pragma unroll
    for (int j = 0; j < NT_PASS; ++j) unitq_from_ring<K>(u[j], stage, j, r, g);
    #pragma unroll
    for (int m = 0; m < 8; ++m)
      #pragma unroll
      for (int j = 0; j < NT_PASS; ++j) mma_s8_16832(acc[j], af[m], v2_bfrag_s8_q<K, REP>(u[j], m, lut, g & (REP - 1)));
    if (DRAFT) {                                                    
      #pragma unroll
      for (int m = 0; m < 8; ++m) {
        uint32_t afd[4] = {0u, 0u, 0u, 0u};
        if (r == M) { const uint2 v = __ldg(reinterpret_cast<const uint2*>(qx + (size_t)M * Kc + ks * 256 + 64 * g + 8 * m)); afd[0] = v.x; afd[2] = v.y; }
        #pragma unroll
        for (int j = 0; j < NT_PASS; ++j) mma_s8_16832(ad[j], afd, v2_bfrag_band_q<K>(u[j], m, s_r8));
      }
    }
    if (hi & 1) {                                                   
      const int grp = ks >> 1;
      const float al_lo = r < M ? __ldg(alpha + (size_t)r * G + grp) : 0.f;                       
      const float al_hi = (MT > 8 && r + 8 < M) ? __ldg(alpha + (size_t)(r + 8) * G + grp) : 0.f;
      #pragma unroll
      for (int j = 0; j < NT_PASS; ++j)
        #pragma unroll
        for (int q = 0; q < 4; ++q) { const int tok = (q < 2) ? r : r + 8;
          if (tok < M) pacc[j][q] = mul_add_rn(pacc[j][q], q < 2 ? al_lo : al_hi, (float)acc[j][q]);
          if (DRAFT && tok == M) { const float al = __ldg(alpha + (size_t)M * G + grp); pd[j][q] = mul_add_rn(pd[j][q], al, (float)ad[j][q]); } }
    }
    if (hi == H - 1) {
      #pragma unroll
      for (int j = 0; j < NT_PASS; ++j) {
        const int n0 = tile * 64 + (j0 + j) * 8 + 2 * g;
        if (!DRAFT) {                                               
          if (r < M && n0 < N) *reinterpret_cast<float2*>(part + ((size_t)worker * MD + r) * N + n0) = make_float2(pacc[j][0], pacc[j][1]);
          if (MT > 8 && r + 8 < M && n0 < N) *reinterpret_cast<float2*>(part + ((size_t)worker * MD + r + 8) * N + n0) = make_float2(pacc[j][2], pacc[j][3]);
        } else {
          #pragma unroll
          for (int q = 0; q < 4; ++q) { const int tok = (q < 2) ? r : r + 8, n = n0 + (q & 1);
            if (tok < M && n < N) part[((size_t)worker * MD + tok) * N + n] = pacc[j][q];
            if (DRAFT && tok == M && n < N) part[((size_t)worker * MD + tok) * N + n] = pd[j][q]; }
        }
      }
    }
    __syncwarp();
  }
}

template <int MT, bool DRAFT, bool XSM>
__global__ void __launch_bounds__(WARPS * 32, DRAFT ? 3 : MINB) gemv_kernel(GemvLaunch P) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* ring = smem;                                      
  int seg = 0; while (seg + 1 < P.n_seg && (int)blockIdx.x >= P.seg[seg + 1].cta_begin) ++seg;
  const GemvSeg& Sg = P.seg[seg]; const TensorDev& t = Sg.t;
  const int worker = blockIdx.y; if (worker >= t.n_workers) { pdl_wait(); return; }   
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = (blockIdx.x - Sg.cta_begin) * WARPS + warp, ntiles = (t.N + 63) / 64;
  const int Kc = t.Kc, G = Kc / 512, KS = Kc / 256;
  const bool active = tile < ntiles;
  const int K = active ? (t.tile_k ? t.tile_k[tile] : t.k_default) : t.k_default;
  const uint4* tbase = active ? (t.frag + (t.tile_u4off ? t.tile_u4off[tile] : (uint64_t)tile * KS * 8 * K * 8)) : nullptr;
  unsigned char* my_ring = ring + warp * STAGES * P.stage_bytes;
  
  bool first = false;
  auto first_stage = [&]() {
    if (active) { int g0 = -1; if (!t.group_worker) g0 = worker < G ? worker : -1; else for (int gg = 0; gg < G; ++gg) if (worker_of(t, gg) == worker) { g0 = gg; break; }
      if (g0 >= 0) { issue_stage_nocommit(tbase, K, 2 * g0, 0, NT_PASS, my_ring, lane); cp_commit(); first = true; } } };
  
  
  
  const bool pre_w = P.M <= P.pdl_prew_m;
  if (pre_w) first_stage();
  
  load_lut8_fast<REP>(g_lut8, t.lut8);
  __shared__ uint16_t s_r8[256];
  if (DRAFT) for (int i = threadIdx.x; i < (1 << t.k_default); i += blockDim.x) s_r8[i] = P.r8[i];
  pdl_wait(); pdl_trigger();                                       
  if (!pre_w) first_stage();
  if (XSM && blockIdx.x == 0 && blockIdx.y == 0 && (int)threadIdx.x < P.M)   
    P.us_out[threadIdx.x] = xsum_us(P.alpha + (size_t)threadIdx.x * (P.seg[0].t.Kc / 512), P.seg[0].t.Kc / 512);
  __syncthreads();
  if (!active && !(XSM && P.e_fuse)) return;                      
  const Lut8<REP> lut{g_lut8};
  if constexpr (!DRAFT) {
    if (active) switch (K) {
      case 4: warp_tile_t<4, MT, XSM>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, Sg.part, lane, first); break;
      case 5: warp_tile_t<5, MT, XSM>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, Sg.part, lane, first); break;
      case 6: warp_tile_t<6, MT, XSM>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, Sg.part, lane, first); break;
      case 7: warp_tile_t<7, MT, XSM>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, Sg.part, lane, first); break;
      default: warp_tile_t<8, MT, XSM>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, Sg.part, lane, first); break;
    }
    if constexpr (XSM) if (P.e_fuse) {                              
      __threadfence();                                              
      __syncthreads();
      __shared__ int s_last; __shared__ float s_red4[4]; __shared__ unsigned s_am;
      const int b = (blockIdx.x - Sg.cta_begin) * WARPS * 64 / 512;
      if (threadIdx.x == 0) s_last = atomicAdd(P.e_cnt + b, 1u) == (unsigned)(512 / (WARPS * 64) * t.n_workers - 1);
      __syncthreads();
      if (s_last) {
        __threadfence();
        xsum_down_epilogue(P, t, Sg.part, b, reinterpret_cast<float*>(ring), s_red4, &s_am);   
        if (threadIdx.x == 0) { P.e_cnt[b] = 0u; if (P.e_dbg) atomicAdd(P.e_dbg + b, 1u); }
      }
    }
    return;
  }
  switch (K) {
    case 4: warp_tile<4, MT, DRAFT>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, lut, Sg.part, lane, first, s_r8); break;
    case 5: warp_tile<5, MT, DRAFT>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, lut, Sg.part, lane, first, s_r8); break;
    case 6: warp_tile<6, MT, DRAFT>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, lut, Sg.part, lane, first, s_r8); break;
    case 7: warp_tile<7, MT, DRAFT>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, lut, Sg.part, lane, first, s_r8); break;
    default: warp_tile<8, MT, DRAFT>(t, tbase, tile, worker, P.M, P.qx, P.alpha, my_ring, P.stage_bytes, lut, Sg.part, lane, first, s_r8); break;
  }
}

size_t gemv_smem_bytes(const GemvLaunch& P) { return (size_t)WARPS * STAGES * P.stage_bytes; }   
void launch_gemv(const GemvLaunch& P, cudaStream_t st) {
  if (lnord_eligible(P)) { launch_lnord(P, st); return; }        
  if (P.fuse) throw std::runtime_error("G31: the fused (in-GEMV) norm/H512/A8 prologue was removed; run launch_(norm_)xform_a8 first");
  const size_t sm = gemv_smem_bytes(P); const dim3 grid(P.n_ctas, P.max_workers), block(WARPS * 32);
  
  
  const int sm_attr = (int)std::max(sm, (size_t)WARPS * STAGES * GEMV_NT_PASS * 8 * 16 * 8);
#define GEMV_LAUNCH_(MT_, D_, N_) { cudaFuncSetAttribute(gemv_kernel<MT_, D_, N_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_attr); launch_k(gemv_kernel<MT_, D_, N_>, grid, block, sm, st, P); }
#define GEMV_LAUNCH(MT_, D_) { if (P.xsum) GEMV_LAUNCH_(MT_, D_, true) else GEMV_LAUNCH_(MT_, D_, false) }
  if (P.draft) {                                                    
    if (P.M + 1 > 8) throw std::runtime_error("DROW draft row needs M + 1 <= 8");
    GEMV_LAUNCH(8, true) return;
  }
  if (P.M == 1)      GEMV_LAUNCH(1, false)
  else if (P.M <= 8) GEMV_LAUNCH(8, false)
  else               GEMV_LAUNCH(16, false)
#undef GEMV_LAUNCH
#undef GEMV_LAUNCH_
}
} 
