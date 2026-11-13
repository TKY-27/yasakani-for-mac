// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include "q35.hpp"
#include <cmath>
#include <cstdlib>
namespace yskn {











namespace {
constexpr int ATX_KT = 32, ATX_LD = Q35_HD + 8;
constexpr int ATX_SMEM = 2 * 2 * ATX_KT * ATX_LD * 2;                  
constexpr float ATX_LOG2E = 1.4426950408889634f;
#ifndef ATX_CVT_DEFAULT
#define ATX_CVT_DEFAULT 1
#endif
constexpr bool ATX_CVT = ATX_CVT_DEFAULT != 0;                        
__device__ __forceinline__ void atx_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void atx_ldm4(unsigned* r, const void* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void atx_ldm4t(unsigned* r, const void* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void atx_cp16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void atx_cp4(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void atx_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void atx_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
template <int N> __device__ __forceinline__ void atx_waitn() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ unsigned atx_pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<const unsigned*>(&v);
}



__device__ __forceinline__ void atx_s8x4_bf16(unsigned w, unsigned& b01, unsigned& b23) {
  const unsigned x = w ^ 0x80808080u;
  const float f0 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7440)) - 8388736.f, f1 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7441)) - 8388736.f;
  const float f2 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7442)) - 8388736.f, f3 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7443)) - 8388736.f;
  b01 = __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632); b23 = __byte_perm(__float_as_uint(f2), __float_as_uint(f3), 0x7632);
}
__device__ __forceinline__ float atx_bf_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float atx_bf_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
template <bool F16> __device__ __forceinline__ void atx_mma16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  if constexpr (F16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else atx_mma(c, a, b0, b1);
}
template <bool F16> __device__ __forceinline__ unsigned atx_pack16(float lo, float hi) {
  if constexpr (F16) { const __half2 v = __floats2half2_rn(lo, hi); return *reinterpret_cast<const unsigned*>(&v); }
  else return atx_pack(lo, hi);
}
template <bool F16> __device__ __forceinline__ float atx_lo16(unsigned w) {
  if constexpr (F16) return __half2float(__ushort_as_half((unsigned short)w));
  else return atx_bf_lo(w);
}
template <bool F16> __device__ __forceinline__ float atx_hi16(unsigned w) {
  if constexpr (F16) return __half2float(__ushort_as_half((unsigned short)(w >> 16)));
  else return atx_bf_hi(w);
}
}  

struct Q35AtxArgs { Q35AttnArgs a; const uint8_t* wk; int cs_split = 0; };   






template <bool KV8, int NW, int NS = (NW == 6 ? 2 : 1), bool FP16KV = false>
__global__ void __launch_bounds__(NW * 32, NW == 6 ? 1 : 2) q35_atx_kernel(Q35AtxArgs U) {
  constexpr int TPC = NW == 6 ? 16 : 2, NST = NS;                       
  static_assert(KV8 || NS <= 2, "BF16 tiles: at most 2 stages (shared memory)");
  const Q35AttnArgs& A = U.a;
  constexpr int G = Q35_NH / Q35_KVH, D = Q35_HD;
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0 && NST >= 2;                         
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = *A.d_pos, M = A.M;
  
  int tk[2], hd[2], lo[2], ph[2], own[2], len[2]; bool self_[2], vrow[2];
  bool need = false;                                                 
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int i = 16 * warp + gi + 8 * h; tk[h] = t0 + i / G; hd[h] = kvh * G + i % G; vrow[h] = i < G * TPC && tk[h] < M;
    const int dp = vrow[h] ? (U.wk ? (int)U.wk[tk[h]] : tk[h]) : -1;
    len[h] = vrow[h] ? base + dp + 1 : 0; own[h] = base + tk[h];
    const int S = (A.splits_short > 0 && len[h] < A.long_len) ? A.splits_short : A.splits;   
    need |= vrow[h] && split < S;
    const int per = (len[h] + S - 1) / S;
    lo[h] = split * per; const int hi = min(len[h], lo[h] + per);
    ph[h] = max(lo[h], min(hi, len[h] - 1));                          
    self_[h] = vrow[h] && lo[h] <= len[h] - 1 && len[h] - 1 < lo[h] + per;
  }
  if (!__syncthreads_or(need)) { if (early) { pdl_wait(); pdl_trigger(); } return; }   
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; ++h) if (vrow[h] && ph[h] > lo[h]) { atomicMin(&s_lo, lo[h]); atomicMax(&s_hi, ph[h]); }
  __syncthreads();
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / ATX_KT) * ATX_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + ATX_KT - 1) / ATX_KT : 0;
  
  uint16_t* const kt_b = usm;                                         
  int8_t* const st8 = reinterpret_cast<int8_t*>(usm + 2 * ATX_KT * ATX_LD);   
  float* const ssc = reinterpret_cast<float*>(st8 + NST * 2 * ATX_KT * D);     
  const uint16_t* const kgb = A.kc + (size_t)kvh * A.max_ctx * D; const uint16_t* const vgb = A.vc + (size_t)kvh * A.max_ctx * D;
  const int8_t* const kg8 = KV8 ? A.k8 + (size_t)kvh * A.max_ctx * D : nullptr; const int8_t* const vg8 = KV8 ? A.v8 + (size_t)kvh * A.max_ctx * D : nullptr;
  const float* const ksg = KV8 ? A.ks + (size_t)kvh * A.max_ctx : nullptr; const float* const vsg = KV8 ? A.vs + (size_t)kvh * A.max_ctx : nullptr;
  auto load_tile = [&](int t, int stage, int part = 0) {             
    const int k0 = tbeg + t * ATX_KT;
    auto mine = [&](int kidx) { return part == 0 || (part == 1 ? kidx < base : kidx >= base); };
    if constexpr (!KV8) {
      uint16_t* const ks = kt_b + (size_t)stage * 2 * ATX_KT * ATX_LD;
      for (int c = threadIdx.x; c < 2 * ATX_KT * (D / 8); c += blockDim.x) {
        const int kv = c / (ATX_KT * (D / 8)), rem = c - kv * (ATX_KT * (D / 8)), key = rem / (D / 8), ch = rem - key * (D / 8);
        uint16_t* const dst = ks + (size_t)kv * ATX_KT * ATX_LD + key * ATX_LD + ch * 8;
        const int kidx = k0 + key;
        if (kidx >= klo && kidx < khi) { if (mine(kidx)) atx_cp16(dst, (kv ? vgb : kgb) + (size_t)kidx * D + ch * 8); }
        else if (part != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
      }
    } else {
      int8_t* const sk = st8 + (size_t)stage * 2 * ATX_KT * D;
      for (int c = threadIdx.x; c < 2 * ATX_KT * (D / 16); c += blockDim.x) {
        const int kv = c / (ATX_KT * (D / 16)), rem = c - kv * (ATX_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
        int8_t* const dst = sk + (size_t)kv * ATX_KT * D + key * D + ch * 16;
        const int kidx = k0 + key;
        if (kidx >= klo && kidx < khi) { if (mine(kidx)) atx_cp16(dst, (kv ? vg8 : kg8) + (size_t)kidx * D + ch * 16); }
        else if (part != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
      }
      for (int c = threadIdx.x; c < 2 * ATX_KT; c += blockDim.x) {        
        const int kv = c / ATX_KT, key = c % ATX_KT, kidx = k0 + key;
        float* const dst = ssc + (stage * 2 + kv) * ATX_KT + key;
        if (kidx >= klo && kidx < khi) { if (mine(kidx)) atx_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (part != 2) *dst = 0.f;
      }
    }
  };
  if constexpr (NST >= 2) {                                          
    #pragma unroll
    for (int p = 0; p < NST - 1; ++p) { if (p < ntiles) load_tile(p, p, early ? 1 : 0); atx_commit(); }
    if (early) { pdl_wait(); pdl_trigger();                          
      #pragma unroll
      for (int p = 0; p < NST - 1; ++p) { if (p < ntiles) load_tile(p, p, 2); atx_commit(); } }
  }
  unsigned qa[16][4];
  {
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[0] ? tk[0] : 0) * Q35_NH + hd[0]) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[1] ? tk[1] : 0) * Q35_NH + hd[1]) * D);
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = vrow[0] ? q0[ks * 8 + tig] : 0u; qa[ks][1] = vrow[1] ? q1[ks * 8 + tig] : 0u;
      qa[ks][2] = vrow[0] ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = vrow[1] ? q1[ks * 8 + 4 + tig] : 0u;
      if constexpr (FP16KV) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) qa[ks][j] = atx_pack16<true>(atx_bf_lo(qa[ks][j]), atx_bf_hi(qa[ks][j]));
      }
    }
  }
  const bool busy = __any_sync(0xffffffffu, vrow[0] || vrow[1]);     
  float o[32][4];
  #pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  for (int t = 0; t < ntiles; ++t) {
    if constexpr (NST >= 2) { if (t + NST - 1 < ntiles) load_tile(t + NST - 1, (t + NST - 1) % NST); atx_commit(); atx_waitn<NST - 1>(); }
    else { load_tile(t, 0); atx_commit(); asm volatile("cp.async.wait_group 0;\n" ::); }
    __syncthreads();
    const int stg = NST >= 2 ? t % NST : 0;
    const uint16_t* ks; const uint16_t* vs; const float* ksc = nullptr; const float* vsc = nullptr;
    if constexpr (!KV8) { ks = kt_b + (size_t)stg * 2 * ATX_KT * ATX_LD; vs = ks + ATX_KT * ATX_LD; }
    else {
      const int8_t* const sk = st8 + (size_t)stg * 2 * ATX_KT * D;
      for (int c = threadIdx.x; c < 2 * ATX_KT * (D / 16); c += blockDim.x) {   
        const int kv = c / (ATX_KT * (D / 16)), rem = c - kv * (ATX_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
        const uint4 u = *reinterpret_cast<const uint4*>(sk + (size_t)kv * ATX_KT * D + key * D + ch * 16);
        unsigned b[8];
        if (ATX_CVT) { atx_s8x4_bf16(u.x, b[0], b[1]); atx_s8x4_bf16(u.y, b[2], b[3]); atx_s8x4_bf16(u.z, b[4], b[5]); atx_s8x4_bf16(u.w, b[6], b[7]); }
        else { const unsigned w[4] = {u.x, u.y, u.z, u.w};
          #pragma unroll
          for (int k = 0; k < 8; ++k) { const unsigned ww = w[k >> 1] >> (16 * (k & 1));
            b[k] = atx_pack16<FP16KV>((float)(signed char)(ww & 0xffu), (float)(signed char)((ww >> 8) & 0xffu)); } }
        uint16_t* const dst = kt_b + (size_t)kv * ATX_KT * ATX_LD + key * ATX_LD + ch * 16;
        *reinterpret_cast<uint4*>(dst) = make_uint4(b[0], b[1], b[2], b[3]); *reinterpret_cast<uint4*>(dst + 8) = make_uint4(b[4], b[5], b[6], b[7]);
      }
      __syncthreads();
      ks = kt_b; vs = kt_b + ATX_KT * ATX_LD;
      ksc = ssc + (stg * 2 + 0) * ATX_KT; vsc = ssc + (stg * 2 + 1) * ATX_KT;
    }
    if (busy) {
      const int kt0 = tbeg + t * ATX_KT;
      float s[4][4];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        #pragma unroll
        for (int kp = 0; kp < 8; ++kp) {
          unsigned b[4];
          atx_ldm4(b, ks + (8 * j + (lane & 7)) * ATX_LD + 32 * kp + 8 * (lane >> 3));
          atx_mma16<FP16KV>(s[j], qa[2 * kp], b[0], b[1]);
          atx_mma16<FP16KV>(s[j], qa[2 * kp + 1], b[2], b[3]);
        }
      }
      float x0 = m0, x1 = m1;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk, h = c >> 1;
          float v = s[j][c] * 0.0625f;
          if constexpr (KV8) v = s[j][c] * (ksc[kk] * 0.0625f);
          s[j][c] = (key >= lo[h] && key < ph[h]) ? v : -INFINITY;
        }
        x0 = fmaxf(x0, fmaxf(s[j][0], s[j][1])); x1 = fmaxf(x1, fmaxf(s[j][2], s[j][3]));
      }
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      const float c0 = exp2f((m0 - x0) * ATX_LOG2E), c1 = exp2f((m1 - x1) * ATX_LOG2E);
      m0 = x0; m1 = x1;
      const float y0 = x0 * ATX_LOG2E, y1 = x1 * ATX_LOG2E;
      float r0s = 0.f, r1s = 0.f;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][0] = exp2f(s[j][0] * ATX_LOG2E - y0); s[j][1] = exp2f(s[j][1] * ATX_LOG2E - y0);
        s[j][2] = exp2f(s[j][2] * ATX_LOG2E - y1); s[j][3] = exp2f(s[j][3] * ATX_LOG2E - y1);
        r0s += s[j][0] + s[j][1]; r1s += s[j][2] + s[j][3];
      }
      r0s += __shfl_xor_sync(0xffffffffu, r0s, 1); r0s += __shfl_xor_sync(0xffffffffu, r0s, 2);
      r1s += __shfl_xor_sync(0xffffffffu, r1s, 1); r1s += __shfl_xor_sync(0xffffffffu, r1s, 2);
      l0 = l0 * c0 + r0s; l1 = l1 * c1 + r1s;
      if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
        #pragma unroll
        for (int n = 0; n < 32; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
      }
      #pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        float va = 1.f, vb = 1.f, vc2 = 1.f, vd2 = 1.f;
        if constexpr (KV8) { va = vsc[16 * kk + 2 * tig]; vb = vsc[16 * kk + 2 * tig + 1]; vc2 = vsc[16 * kk + 8 + 2 * tig]; vd2 = vsc[16 * kk + 8 + 2 * tig + 1]; }
        unsigned pa[4];
        if constexpr (KV8) {
          pa[0] = atx_pack16<FP16KV>(s[2 * kk][0] * va, s[2 * kk][1] * vb); pa[1] = atx_pack16<FP16KV>(s[2 * kk][2] * va, s[2 * kk][3] * vb);
          pa[2] = atx_pack16<FP16KV>(s[2 * kk + 1][0] * vc2, s[2 * kk + 1][1] * vd2); pa[3] = atx_pack16<FP16KV>(s[2 * kk + 1][2] * vc2, s[2 * kk + 1][3] * vd2);
        } else {
          pa[0] = atx_pack16<FP16KV>(s[2 * kk][0], s[2 * kk][1]); pa[1] = atx_pack16<FP16KV>(s[2 * kk][2], s[2 * kk][3]);
          pa[2] = atx_pack16<FP16KV>(s[2 * kk + 1][0], s[2 * kk + 1][1]); pa[3] = atx_pack16<FP16KV>(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        }
        #pragma unroll
        for (int np = 0; np < 16; ++np) {
          unsigned b[4];
          atx_ldm4t(b, vs + (16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * ATX_LD + 16 * np + (lane >> 4) * 8);
          atx_mma16<FP16KV>(o[2 * np], pa, b[0], b[1]);
          atx_mma16<FP16KV>(o[2 * np + 1], pa, b[2], b[3]);
        }
      }
    }
    __syncthreads();                                                  
  }
  
  if (busy) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const bool sf = self_[h];                                       
      float part = 0.f;
      if (sf) {
        const size_t row = (size_t)own[h];
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
          const unsigned qlo = qa[ks][h], qhi = qa[ks][2 + h];        
          float k0, k1, k2, k3;
          if constexpr (KV8) {
            const int8_t* kr = kg8 + row * D + 16 * ks + 2 * tig;
            k0 = (float)kr[0]; k1 = (float)kr[1]; k2 = (float)kr[8]; k3 = (float)kr[9];
          } else {
            const unsigned* kr = reinterpret_cast<const unsigned*>(kgb + row * D);
            const unsigned w0 = kr[ks * 8 + tig], w1 = kr[ks * 8 + 4 + tig];
            k0 = atx_lo16<FP16KV>(w0); k1 = atx_hi16<FP16KV>(w0); k2 = atx_lo16<FP16KV>(w1); k3 = atx_hi16<FP16KV>(w1);
          }
          part = __fadd_rn(part, __fmul_rn(atx_lo16<FP16KV>(qlo), k0)); part = __fadd_rn(part, __fmul_rn(atx_hi16<FP16KV>(qlo), k1));
          part = __fadd_rn(part, __fmul_rn(atx_lo16<FP16KV>(qhi), k2)); part = __fadd_rn(part, __fmul_rn(atx_hi16<FP16KV>(qhi), k3));
        }
      }
      part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 1));
      part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 2));
      if (sf) {
        float sc = part * 0.0625f;
        if constexpr (KV8) sc = part * (ksg[own[h]] * 0.0625f);
        float& m = h ? m1 : m0; float& l = h ? l1 : l0;
        const float mn = fmaxf(m, sc), c = exp2f((m - mn) * ATX_LOG2E), p = exp2f((sc - mn) * ATX_LOG2E);
        m = mn; l = l * c + p;
        float pv = p;
        if constexpr (KV8) pv = p * vsg[own[h]];
        #pragma unroll
        for (int n = 0; n < 32; ++n) {
          float v0, v1;
          if constexpr (KV8) { const int8_t* vr = vg8 + (size_t)own[h] * D + 8 * n + 2 * tig; v0 = (float)vr[0]; v1 = (float)vr[1]; }
          else { const unsigned w = reinterpret_cast<const unsigned*>(vgb + (size_t)own[h] * D)[4 * n + tig]; v0 = atx_lo16<FP16KV>(w); v1 = atx_hi16<FP16KV>(w); }
          o[n][2 * h] = o[n][2 * h] * c + pv * v0; o[n][2 * h + 1] = o[n][2 * h + 1] * c + pv * v1;
        }
      }
    }
  }
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!vrow[h]) continue;
    const size_t pidx = ((size_t)tk[h] * A.splits + split) * Q35_NH + hd[h];
    if (tig == 0) { A.pm[pidx] = h ? m1 : m0; A.pl[pidx] = h ? l1 : l0; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < 32; ++n) *reinterpret_cast<float2*>(pa + 8 * n) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}













template <int RT, int WPT>
__global__ void __launch_bounds__(RT * WPT * 32, 1) q35_atx_vsp_kernel(Q35AtxArgs U) {
  constexpr int TPC = (16 * RT) / 6, JW = 4 / WPT, NB = 32 / WPT, TL = 256;   
  const Q35AttnArgs& A = U.a;
  constexpr int G = Q35_NH / Q35_KVH, D = Q35_HD;
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  __shared__ float2 x_mx[RT][4][32], x_sm[RT][4][32];
  __shared__ uint2 x_pf[RT][4][32];
  __shared__ float sc2[2][2][ATX_KT];                                  
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = warp / WPT, part = warp % WPT, gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0;
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = *A.d_pos, M = A.M;
  int tk[2], hd[2], lo[2], ph[2], own[2], len[2]; bool self_[2], vrow[2];
  bool need = false;
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int i = 16 * tile + gi + 8 * h; tk[h] = t0 + i / G; hd[h] = kvh * G + i % G; vrow[h] = i < G * TPC && tk[h] < M;
    const int dp = vrow[h] ? (U.wk ? (int)U.wk[tk[h]] : tk[h]) : -1;
    len[h] = vrow[h] ? base + dp + 1 : 0; own[h] = base + tk[h];
    const int S = (A.splits_short > 0 && len[h] < A.long_len) ? A.splits_short : A.splits;
    need |= vrow[h] && split < S;
    const int per = (len[h] + S - 1) / S;
    lo[h] = split * per; const int hi = min(len[h], lo[h] + per);
    ph[h] = max(lo[h], min(hi, len[h] - 1));
    self_[h] = vrow[h] && lo[h] <= len[h] - 1 && len[h] - 1 < lo[h] + per;
  }
  if (!__syncthreads_or(need)) { if (early) { pdl_wait(); pdl_trigger(); } return; }
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; ++h) if (vrow[h] && ph[h] > lo[h]) { atomicMin(&s_lo, lo[h]); atomicMax(&s_hi, ph[h]); }
  __syncthreads();
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / ATX_KT) * ATX_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + ATX_KT - 1) / ATX_KT : 0;
  uint16_t* const kt_b = usm;                                         
  int8_t* const st8 = reinterpret_cast<int8_t*>(usm + 2 * 2 * ATX_KT * TL);      
  float* const ssc = reinterpret_cast<float*>(st8 + 2 * ATX_KT * D);              
  const int8_t* const kg8 = A.k8 + (size_t)kvh * A.max_ctx * D; const int8_t* const vg8 = A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = A.ks + (size_t)kvh * A.max_ctx; const float* const vsg = A.vs + (size_t)kvh * A.max_ctx;
  auto load_tile = [&](int t, int pt) {                               
    const int k0 = tbeg + t * ATX_KT;
    auto mine = [&](int kidx) { return pt == 0 || (pt == 1 ? kidx < base : kidx >= base); };
    for (int c = threadIdx.x; c < 2 * ATX_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (ATX_KT * (D / 16)), rem = c - kv * (ATX_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      int8_t* const dst = st8 + (size_t)kv * ATX_KT * D + key * D + ch * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) atx_cp16(dst, (kv ? vg8 : kg8) + (size_t)kidx * D + ch * 16); }
      else if (pt != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = threadIdx.x; c < 2 * ATX_KT; c += blockDim.x) {
      const int kv = c / ATX_KT, key = c % ATX_KT, kidx = k0 + key;
      float* const dst = ssc + kv * ATX_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) atx_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (pt != 2) *dst = 0.f;
    }
  };
  auto convert_own = [&](int b) {                                     
    uint16_t* const kb = kt_b + (size_t)b * 2 * ATX_KT * TL;
    for (int c = threadIdx.x; c < 2 * ATX_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (ATX_KT * (D / 16)), rem = c - kv * (ATX_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      const uint4 u = *reinterpret_cast<const uint4*>(st8 + (size_t)kv * ATX_KT * D + key * D + ch * 16);
      unsigned w[8];
      atx_s8x4_bf16(u.x, w[0], w[1]); atx_s8x4_bf16(u.y, w[2], w[3]); atx_s8x4_bf16(u.z, w[4], w[5]); atx_s8x4_bf16(u.w, w[6], w[7]);
      uint16_t* const row = kb + (size_t)kv * ATX_KT * TL + key * TL; const int sw = key & 7;
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch) ^ sw)) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch + 1) ^ sw)) = make_uint4(w[4], w[5], w[6], w[7]);
    }
    for (int c = threadIdx.x; c < 2 * ATX_KT; c += blockDim.x) sc2[b][c / ATX_KT][c % ATX_KT] = ssc[c];
  };
  if (0 < ntiles) load_tile(0, early ? 1 : 0);
  atx_commit();
  if (early) { pdl_wait(); pdl_trigger(); if (0 < ntiles) load_tile(0, 2); atx_commit(); }
  unsigned qa[16][4];
  {
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[0] ? tk[0] : 0) * Q35_NH + hd[0]) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[1] ? tk[1] : 0) * Q35_NH + hd[1]) * D);
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = vrow[0] ? q0[ks * 8 + tig] : 0u; qa[ks][1] = vrow[1] ? q1[ks * 8 + tig] : 0u;
      qa[ks][2] = vrow[0] ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = vrow[1] ? q1[ks * 8 + 4 + tig] : 0u;
    }
  }
  if (0 < ntiles) {                                                  
    asm volatile("cp.async.wait_group 0;\n" ::);
    convert_own(0);
    if (1 < ntiles) load_tile(1, 0);
    atx_commit();
  }
  const bool busy = __any_sync(0xffffffffu, vrow[0] || vrow[1]);
  float o[NB][4];                                                    
  #pragma unroll
  for (int n = 0; n < NB; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float s[JW][4], c0 = 1.f, c1 = 1.f;                                    
  for (int t = 0; t <= ntiles; ++t) {
    __syncthreads();                                                 
    if (busy && t >= 1) {                                            
      float r0s = 0.f, r1s = 0.f;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) { const float2 q = x_sm[tile][jj][lane]; r0s += q.x; r1s += q.y; }
      r0s += __shfl_xor_sync(0xffffffffu, r0s, 1); r0s += __shfl_xor_sync(0xffffffffu, r0s, 2);
      r1s += __shfl_xor_sync(0xffffffffu, r1s, 1); r1s += __shfl_xor_sync(0xffffffffu, r1s, 2);
      l0 = l0 * c0 + r0s; l1 = l1 * c1 + r1s;
      if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
        #pragma unroll
        for (int n = 0; n < NB; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
      }
      const uint16_t* const vs = kt_b + (size_t)((t - 1) & 1) * 2 * ATX_KT * TL + ATX_KT * TL;
      #pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const uint2 pA = x_pf[tile][2 * kk][lane], pB = x_pf[tile][2 * kk + 1][lane];
        const unsigned pa[4] = {pA.x, pA.y, pB.x, pB.y};
        #pragma unroll
        for (int q = 0; q < NB / 2; ++q) {
          const int np = (NB / 2) * part + q;
          unsigned bb[4];
          const int r = 16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7), c = 2 * np + (lane >> 4);
          atx_ldm4t(bb, vs + r * TL + 8 * (c ^ (r & 7)));
          atx_mma(o[2 * q], pa, bb[0], bb[1]);
          atx_mma(o[2 * q + 1], pa, bb[2], bb[3]);
        }
      }
    }
    if (t == ntiles) break;                                          
    const int b = t & 1;
    if (busy) {                                                      
      const uint16_t* const ks = kt_b + (size_t)b * 2 * ATX_KT * TL; const float* const ksc = sc2[b][0];
      const int kt0 = tbeg + t * ATX_KT;
      #pragma unroll
      for (int w = 0; w < JW; ++w) {
        const int j = JW * part + w;
        s[w][0] = s[w][1] = s[w][2] = s[w][3] = 0.f;
        #pragma unroll
        for (int kp = 0; kp < 8; ++kp) {
          unsigned bb[4];
          const int r = 8 * j + (lane & 7), c = 4 * kp + (lane >> 3);
          atx_ldm4(bb, ks + r * TL + 8 * (c ^ (r & 7)));
          atx_mma(s[w], qa[2 * kp], bb[0], bb[1]);
          atx_mma(s[w], qa[2 * kp + 1], bb[2], bb[3]);
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk, h = c >> 1;
          const float v = s[w][c] * (ksc[kk] * 0.0625f);
          s[w][c] = (key >= lo[h] && key < ph[h]) ? v : -INFINITY;
        }
        x_mx[tile][j][lane] = make_float2(fmaxf(s[w][0], s[w][1]), fmaxf(s[w][2], s[w][3]));
      }
    }
    __syncthreads();                                                 
    if (busy) {                                                      
      const float* const vsc = sc2[b][1];
      float x0 = m0, x1 = m1;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) { const float2 q = x_mx[tile][jj][lane]; x0 = fmaxf(x0, q.x); x1 = fmaxf(x1, q.y); }
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      c0 = exp2f((m0 - x0) * ATX_LOG2E); c1 = exp2f((m1 - x1) * ATX_LOG2E);
      m0 = x0; m1 = x1;
      const float y0 = x0 * ATX_LOG2E, y1 = x1 * ATX_LOG2E;
      #pragma unroll
      for (int w = 0; w < JW; ++w) {
        const int j = JW * part + w;
        s[w][0] = exp2f(s[w][0] * ATX_LOG2E - y0); s[w][1] = exp2f(s[w][1] * ATX_LOG2E - y0);
        s[w][2] = exp2f(s[w][2] * ATX_LOG2E - y1); s[w][3] = exp2f(s[w][3] * ATX_LOG2E - y1);
        x_sm[tile][j][lane] = make_float2(s[w][0] + s[w][1], s[w][2] + s[w][3]);
        const float va = vsc[8 * j + 2 * tig], vb = vsc[8 * j + 2 * tig + 1];
        x_pf[tile][j][lane] = make_uint2(atx_pack(s[w][0] * va, s[w][1] * vb), atx_pack(s[w][2] * va, s[w][3] * vb));
      }
    }
    if (t + 1 < ntiles) {                                            
      asm volatile("cp.async.wait_group 0;\n" ::);
      convert_own(b ^ 1);
      if (t + 2 < ntiles) load_tile(t + 2, 0);
      atx_commit();
    }
  }
  if (busy) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const bool sf = self_[h];
      float pt = 0.f;
      if (sf) {
        const size_t row = (size_t)own[h];
        #pragma unroll
        for (int kq = 0; kq < 16; ++kq) {
          const unsigned qlo = qa[kq][h], qhi = qa[kq][2 + h];
          const int8_t* kr = kg8 + row * D + 16 * kq + 2 * tig;
          const float k0 = (float)kr[0], k1 = (float)kr[1], k2 = (float)kr[8], k3 = (float)kr[9];
          pt = __fadd_rn(pt, __fmul_rn(atx_bf_lo(qlo), k0)); pt = __fadd_rn(pt, __fmul_rn(atx_bf_hi(qlo), k1));
          pt = __fadd_rn(pt, __fmul_rn(atx_bf_lo(qhi), k2)); pt = __fadd_rn(pt, __fmul_rn(atx_bf_hi(qhi), k3));
        }
      }
      pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 1));
      pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 2));
      if (sf) {
        const float sc = pt * (ksg[own[h]] * 0.0625f);
        float& m = h ? m1 : m0; float& l = h ? l1 : l0;
        const float mn = fmaxf(m, sc), c = exp2f((m - mn) * ATX_LOG2E), p = exp2f((sc - mn) * ATX_LOG2E);
        m = mn; l = l * c + p;
        const float pv = p * vsg[own[h]];
        #pragma unroll
        for (int n = 0; n < NB; ++n) {
          const int8_t* vr = vg8 + (size_t)own[h] * D + 8 * (NB * part + n) + 2 * tig;
          const float v0 = (float)vr[0], v1 = (float)vr[1];
          o[n][2 * h] = o[n][2 * h] * c + pv * v0; o[n][2 * h + 1] = o[n][2 * h + 1] * c + pv * v1;
        }
      }
    }
  }
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!vrow[h]) continue;
    const size_t pidx = ((size_t)tk[h] * A.splits + split) * Q35_NH + hd[h];
    if (tig == 0 && part == 0) { A.pm[pidx] = h ? m1 : m0; A.pl[pidx] = h ? l1 : l0; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < NB; ++n) *reinterpret_cast<float2*>(pa + 8 * (NB * part + n)) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}









constexpr size_t ATX2_SMEM = (size_t)2 * 2 * ATX_KT * ATX_LD * 2 + (size_t)2 * 2 * ATX_KT * Q35_HD + (size_t)2 * 2 * ATX_KT * 4;   
__global__ void __launch_bounds__(192, 1) q35_atx2_kernel(Q35AtxArgs U) {
  constexpr int TPC = 16, G = Q35_NH / Q35_KVH, D = Q35_HD;
  const Q35AttnArgs& A = U.a;
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0;
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = *A.d_pos, M = A.M;
  const int ntok = min(TPC, M - t0), nrt = (ntok * G + 15) / 16;
  const int CS = !U.cs_split ? 1 : (nrt == 1 ? 6 : (nrt == 2 ? 3 : (nrt == 3 ? 2 : 1)));
  const int rt = warp / CS, cpart = warp - rt * CS;
  const int np0 = (cpart * 16) / CS, np1 = ((cpart + 1) * 16) / CS;   
  int tk[2], hd[2], lo[2], ph[2], own[2], len[2]; bool self_[2], vrow[2];
  bool need = false;                                                 
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int i = 16 * rt + gi + 8 * h; tk[h] = t0 + i / G; hd[h] = kvh * G + i % G; vrow[h] = rt < nrt && i < G * TPC && tk[h] < M;
    const int dp = vrow[h] ? (U.wk ? (int)U.wk[tk[h]] : tk[h]) : -1;
    len[h] = vrow[h] ? base + dp + 1 : 0; own[h] = base + tk[h];
    const int S = (A.splits_short > 0 && len[h] < A.long_len) ? A.splits_short : A.splits;   
    need |= vrow[h] && split < S;
    const int per = (len[h] + S - 1) / S;
    lo[h] = split * per; const int hi = min(len[h], lo[h] + per);
    ph[h] = max(lo[h], min(hi, len[h] - 1));
    self_[h] = vrow[h] && lo[h] <= len[h] - 1 && len[h] - 1 < lo[h] + per;
  }
  if (!__syncthreads_or(need)) { if (early) { pdl_wait(); pdl_trigger(); } return; }   
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; ++h) if (vrow[h] && ph[h] > lo[h]) { atomicMin(&s_lo, lo[h]); atomicMax(&s_hi, ph[h]); }
  __syncthreads();
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / ATX_KT) * ATX_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + ATX_KT - 1) / ATX_KT : 0;
  uint16_t* const bfb = usm;                                          
  int8_t* const st8 = reinterpret_cast<int8_t*>(usm + 2 * 2 * ATX_KT * ATX_LD);   
  float* const ssc = reinterpret_cast<float*>(st8 + 2 * 2 * ATX_KT * D);         
  const int8_t* const kg8 = A.k8 + (size_t)kvh * A.max_ctx * D; const int8_t* const vg8 = A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = A.ks + (size_t)kvh * A.max_ctx; const float* const vsg = A.vs + (size_t)kvh * A.max_ctx;
  auto load_tile = [&](int t, int stage, int part) {                 
    const int k0 = tbeg + t * ATX_KT;
    auto mine = [&](int kidx) { return part == 0 || (part == 1 ? kidx < base : kidx >= base); };
    int8_t* const sk = st8 + (size_t)stage * 2 * ATX_KT * D;
    for (int c = threadIdx.x; c < 2 * ATX_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (ATX_KT * (D / 16)), rem = c - kv * (ATX_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      int8_t* const dst = sk + (size_t)kv * ATX_KT * D + key * D + ch * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) atx_cp16(dst, (kv ? vg8 : kg8) + (size_t)kidx * D + ch * 16); }
      else if (part != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = threadIdx.x; c < 2 * ATX_KT; c += blockDim.x) {
      const int kv = c / ATX_KT, key = c % ATX_KT, kidx = k0 + key;
      float* const dst = ssc + (stage * 2 + kv) * ATX_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) atx_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (part != 2) *dst = 0.f;
    }
  };
  auto convert = [&](int stage, int buf) {                           
    const int8_t* const sk = st8 + (size_t)stage * 2 * ATX_KT * D;
    uint16_t* const kb = bfb + (size_t)buf * 2 * ATX_KT * ATX_LD;
    for (int c = threadIdx.x; c < 2 * ATX_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (ATX_KT * (D / 16)), rem = c - kv * (ATX_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      const uint4 u = *reinterpret_cast<const uint4*>(sk + (size_t)kv * ATX_KT * D + key * D + ch * 16);
      unsigned b[8];
      atx_s8x4_bf16(u.x, b[0], b[1]); atx_s8x4_bf16(u.y, b[2], b[3]); atx_s8x4_bf16(u.z, b[4], b[5]); atx_s8x4_bf16(u.w, b[6], b[7]);
      uint16_t* const dst = kb + (size_t)kv * ATX_KT * ATX_LD + key * ATX_LD + ch * 16;
      *reinterpret_cast<uint4*>(dst) = make_uint4(b[0], b[1], b[2], b[3]); *reinterpret_cast<uint4*>(dst + 8) = make_uint4(b[4], b[5], b[6], b[7]);
    }
    for (int c = threadIdx.x; c < 2 * ATX_KT; c += blockDim.x) {
      const int kv = c / ATX_KT, key = c % ATX_KT;
      *reinterpret_cast<float*>(kb + (size_t)kv * ATX_KT * ATX_LD + key * ATX_LD + D) = ssc[(stage * 2 + kv) * ATX_KT + key];
    }
  };
  
  if (ntiles > 0) load_tile(0, 0, early ? 1 : 0);
  atx_commit();
  if (ntiles > 1) load_tile(1, 1, early ? 1 : 0);
  atx_commit();
  if (early) { pdl_wait(); pdl_trigger(); if (ntiles > 0) load_tile(0, 0, 2); atx_commit(); if (ntiles > 1) load_tile(1, 1, 2); atx_commit(); }
  atx_waitn<1>();                                                     
  if (ntiles > 0) convert(0, 0);
  unsigned qa[16][4];
  {
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[0] ? tk[0] : 0) * Q35_NH + hd[0]) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[1] ? tk[1] : 0) * Q35_NH + hd[1]) * D);
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = vrow[0] ? q0[ks * 8 + tig] : 0u; qa[ks][1] = vrow[1] ? q1[ks * 8 + tig] : 0u;
      qa[ks][2] = vrow[0] ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = vrow[1] ? q1[ks * 8 + 4 + tig] : 0u;
    }
  }
  const bool busy = __any_sync(0xffffffffu, vrow[0] || vrow[1]);
  float o[32][4];
  #pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  __syncthreads();                                                    
  for (int t = 0; t < ntiles; ++t) {
    if (t + 2 < ntiles) load_tile(t + 2, t & 1, 0);                  
    atx_commit(); atx_waitn<1>();                                      
    if (t + 1 < ntiles) convert((t + 1) & 1, (t + 1) & 1);
    const uint16_t* const ks = bfb + (size_t)(t & 1) * 2 * ATX_KT * ATX_LD; const uint16_t* const vs = ks + ATX_KT * ATX_LD;
    if (busy) {
      const int kt0 = tbeg + t * ATX_KT;
      float s[4][4];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        #pragma unroll
        for (int kp = 0; kp < 8; ++kp) {
          unsigned b[4];
          atx_ldm4(b, ks + (8 * j + (lane & 7)) * ATX_LD + 32 * kp + 8 * (lane >> 3));
          atx_mma(s[j], qa[2 * kp], b[0], b[1]);
          atx_mma(s[j], qa[2 * kp + 1], b[2], b[3]);
        }
      }
      float x0 = m0, x1 = m1;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk, h = c >> 1;
          const float ksc = *reinterpret_cast<const float*>(ks + kk * ATX_LD + D);
          const float v = s[j][c] * (ksc * 0.0625f);
          s[j][c] = (key >= lo[h] && key < ph[h]) ? v : -INFINITY;
        }
        x0 = fmaxf(x0, fmaxf(s[j][0], s[j][1])); x1 = fmaxf(x1, fmaxf(s[j][2], s[j][3]));
      }
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      const float c0 = exp2f((m0 - x0) * ATX_LOG2E), c1 = exp2f((m1 - x1) * ATX_LOG2E);
      m0 = x0; m1 = x1;
      const float y0 = x0 * ATX_LOG2E, y1 = x1 * ATX_LOG2E;
      float r0s = 0.f, r1s = 0.f;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][0] = exp2f(s[j][0] * ATX_LOG2E - y0); s[j][1] = exp2f(s[j][1] * ATX_LOG2E - y0);
        s[j][2] = exp2f(s[j][2] * ATX_LOG2E - y1); s[j][3] = exp2f(s[j][3] * ATX_LOG2E - y1);
        r0s += s[j][0] + s[j][1]; r1s += s[j][2] + s[j][3];
      }
      r0s += __shfl_xor_sync(0xffffffffu, r0s, 1); r0s += __shfl_xor_sync(0xffffffffu, r0s, 2);
      r1s += __shfl_xor_sync(0xffffffffu, r1s, 1); r1s += __shfl_xor_sync(0xffffffffu, r1s, 2);
      l0 = l0 * c0 + r0s; l1 = l1 * c1 + r1s;
      if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
        #pragma unroll
        for (int n = 0; n < 32; ++n) if ((n >> 1) >= np0 && (n >> 1) < np1) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
      }
      #pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const float va = *reinterpret_cast<const float*>(vs + (16 * kk + 2 * tig) * ATX_LD + D), vb = *reinterpret_cast<const float*>(vs + (16 * kk + 2 * tig + 1) * ATX_LD + D);
        const float vc2 = *reinterpret_cast<const float*>(vs + (16 * kk + 8 + 2 * tig) * ATX_LD + D), vd2 = *reinterpret_cast<const float*>(vs + (16 * kk + 8 + 2 * tig + 1) * ATX_LD + D);
        unsigned pa[4];
        pa[0] = atx_pack(s[2 * kk][0] * va, s[2 * kk][1] * vb); pa[1] = atx_pack(s[2 * kk][2] * va, s[2 * kk][3] * vb);
        pa[2] = atx_pack(s[2 * kk + 1][0] * vc2, s[2 * kk + 1][1] * vd2); pa[3] = atx_pack(s[2 * kk + 1][2] * vc2, s[2 * kk + 1][3] * vd2);
        #pragma unroll
        for (int np = 0; np < 16; ++np) {
          if (np < np0 || np >= np1) continue;
          unsigned b[4];
          atx_ldm4t(b, vs + (16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * ATX_LD + 16 * np + (lane >> 4) * 8);
          atx_mma(o[2 * np], pa, b[0], b[1]);
          atx_mma(o[2 * np + 1], pa, b[2], b[3]);
        }
      }
    }
    __syncthreads();                                                  
  }
  if (busy) {                                                         
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const bool sf = self_[h];
      float part = 0.f;
      if (sf) {
        const size_t row = (size_t)own[h];
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
          const unsigned qlo = qa[ks][h], qhi = qa[ks][2 + h];
          const int8_t* kr = kg8 + row * D + 16 * ks + 2 * tig;
          const float k0 = (float)kr[0], k1 = (float)kr[1], k2 = (float)kr[8], k3 = (float)kr[9];
          part = __fadd_rn(part, __fmul_rn(atx_bf_lo(qlo), k0)); part = __fadd_rn(part, __fmul_rn(atx_bf_hi(qlo), k1));
          part = __fadd_rn(part, __fmul_rn(atx_bf_lo(qhi), k2)); part = __fadd_rn(part, __fmul_rn(atx_bf_hi(qhi), k3));
        }
      }
      part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 1));
      part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 2));
      if (sf) {
        const float sc = part * (ksg[own[h]] * 0.0625f);
        float& m = h ? m1 : m0; float& l = h ? l1 : l0;
        const float mn = fmaxf(m, sc), c = exp2f((m - mn) * ATX_LOG2E), p = exp2f((sc - mn) * ATX_LOG2E);
        m = mn; l = l * c + p;
        const float pv = p * vsg[own[h]];
        #pragma unroll
        for (int n = 0; n < 32; ++n) {
          if ((n >> 1) < np0 || (n >> 1) >= np1) continue;
          const int8_t* vr = vg8 + (size_t)own[h] * D + 8 * n + 2 * tig; const float v0 = (float)vr[0], v1 = (float)vr[1];
          o[n][2 * h] = o[n][2 * h] * c + pv * v0; o[n][2 * h + 1] = o[n][2 * h + 1] * c + pv * v1;
        }
      }
    }
  }
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!vrow[h]) continue;
    const size_t pidx = ((size_t)tk[h] * A.splits + split) * Q35_NH + hd[h];
    if (tig == 0 && cpart == 0) { A.pm[pidx] = h ? m1 : m0; A.pl[pidx] = h ? l1 : l0; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < 32; ++n) if ((n >> 1) >= np0 && (n >> 1) < np1) *reinterpret_cast<float2*>(pa + 8 * n) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}
static int atx2_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX2"); return e ? std::atoi(e) : 0; }(); return v; }   
static int atx_ns_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_NS"); const int n = e ? std::atoi(e) : 2; return n >= 2 && n <= 4 ? n : 2; }(); return v; }
constexpr size_t ATX_T_SMEM = (size_t)2 * 2 * ATX_KT * Q35_HD * 2 + (size_t)2 * ATX_KT * Q35_HD + (size_t)2 * ATX_KT * 4;   
constexpr size_t atx_smem8(int ns) { return (size_t)2 * ATX_KT * ATX_LD * 2 + (size_t)ns * 2 * ATX_KT * Q35_HD + (size_t)ns * 2 * ATX_KT * 4; }   
void launch_q35_atx(const Q35AttnArgs& A, const uint8_t* wk, cudaStream_t st) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(q35_atx_kernel<false, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATX_SMEM);
               cudaFuncSetAttribute(q35_atx_kernel<false, 6, 2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATX_SMEM);
               cudaFuncSetAttribute(q35_atx_kernel<true, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATX_SMEM);
               cudaFuncSetAttribute(q35_atx_kernel<false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATX_SMEM);
               cudaFuncSetAttribute(q35_atx_kernel<true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATX_SMEM);
               cudaFuncSetAttribute(q35_atx_kernel<true, 6, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)atx_smem8(3));
               cudaFuncSetAttribute(q35_atx_kernel<true, 6, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)atx_smem8(4));
               cudaFuncSetAttribute(q35_atx2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ATX2_SMEM);
               cudaFuncSetAttribute(q35_atx_vsp_kernel<1, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ATX_T_SMEM);
               cudaFuncSetAttribute(q35_atx_vsp_kernel<3, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ATX_T_SMEM);
               cudaFuncSetAttribute(q35_atx_vsp_kernel<6, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ATX_T_SMEM); attr = true; }
  if (A.kv_fp16) {
    if (A.k8) throw std::runtime_error("FP16 ATX received INT8 cache");
    launch_k(q35_atx_kernel<false, 6, 2, true>, dim3((A.M + 15) / 16, A.splits, Q35_KVH), dim3(192), (size_t)ATX_SMEM, st, Q35AtxArgs{A, wk});
    return;
  }
  constexpr int SM1_BF = 2 * ATX_KT * ATX_LD * 2, SM1_K8 = 2 * ATX_KT * ATX_LD * 2 + 2 * ATX_KT * Q35_HD + 2 * ATX_KT * 4;   
  if (A.M <= 2 && false) {                                            
    const dim3 grid((A.M + 1) / 2, A.splits, Q35_KVH);
    if (A.k8) launch_k(q35_atx_kernel<true, 1>, grid, dim3(32), (size_t)SM1_K8, st, Q35AtxArgs{A, wk});
    else launch_k(q35_atx_kernel<false, 1>, grid, dim3(32), (size_t)SM1_BF, st, Q35AtxArgs{A, wk});
    return; }
  if (A.k8 && ((A.vsp & 1) && A.M <= 8 || (A.vsp & 2) && A.M > 8 && A.M <= 16 || (A.vsp & 4) && A.M > 16)) {   
    const Q35AtxArgs UT{A, wk};
    if (A.M <= 2) launch_k(q35_atx_vsp_kernel<1, 4>, dim3((A.M + 1) / 2, A.splits, Q35_KVH), dim3(128), ATX_T_SMEM, st, UT);
    else if (A.M <= 8) launch_k(q35_atx_vsp_kernel<3, 4>, dim3((A.M + 7) / 8, A.splits, Q35_KVH), dim3(384), ATX_T_SMEM, st, UT);
    else launch_k(q35_atx_vsp_kernel<6, 2>, dim3((A.M + 15) / 16, A.splits, Q35_KVH), dim3(384), ATX_T_SMEM, st, UT);
    return; }
  const dim3 grid((A.M + 15) / 16, A.splits, Q35_KVH);
  if (A.k8 && atx2_env()) { Q35AtxArgs ATX{A, wk}; ATX.cs_split = atx2_env() == 2; launch_k(q35_atx2_kernel, grid, dim3(192), ATX2_SMEM, st, ATX); return; }   
  if (A.k8) { const int ns = atx_ns_env();
    if (ns == 4) launch_k(q35_atx_kernel<true, 6, 4>, grid, dim3(192), atx_smem8(4), st, Q35AtxArgs{A, wk});
    else if (ns == 3) launch_k(q35_atx_kernel<true, 6, 3>, grid, dim3(192), atx_smem8(3), st, Q35AtxArgs{A, wk});
    else launch_k(q35_atx_kernel<true, 6>, grid, dim3(192), (size_t)ATX_SMEM, st, Q35AtxArgs{A, wk}); }
  else launch_k(q35_atx_kernel<false, 6>, grid, dim3(192), (size_t)ATX_SMEM, st, Q35AtxArgs{A, wk});
}
}  
