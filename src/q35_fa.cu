// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <cuda_bf16.h>
#include "q35.hpp"
#include <cmath>
namespace yskn {









namespace {
constexpr int FA_KT = 32, FA_LD = Q35_HD + 8;                      
constexpr int FA_SMEM = 2 * 2 * FA_KT * FA_LD * 2;                  
constexpr float FA_LOG2E = 1.4426950408889634f;
__device__ __forceinline__ void fa_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fa_ldm4(unsigned* r, const void* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void fa_ldm4t(unsigned* r, const void* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void fa_cp16(void* smem, const void* gmem) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void fa_cp4(void* smem, const void* gmem) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void fa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void fa_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
__device__ __forceinline__ unsigned fa_pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<const unsigned*>(&v);
}
}  

__global__ void __launch_bounds__(192, 1) q35_fa_kernel(Q35AttnArgs A) {
  constexpr int G = Q35_NH / Q35_KVH, D = Q35_HD;
  extern __shared__ __align__(16) uint16_t fsm[];                    
  const int r0 = blockIdx.x * 16, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;   
  const int gi = lane >> 2, tig = lane & 3, head = kvh * G + warp;
  pdl_wait(); pdl_trigger();
  const int base = *A.d_pos, R = A.M;
  const int L = base + R, per = (L + A.splits - 1) / A.splits, k0 = split * per, k1 = min(L, k0 + per);
  const int kend = min(k1, base + min(R, r0 + 16));                  
  const int ntiles = kend > k0 ? (kend - k0 + FA_KT - 1) / FA_KT : 0;
  const int pos0 = base + r0 + gi, pos1 = pos0 + 8;                   
  const uint16_t* const kg = A.kc + (size_t)kvh * A.max_ctx * D;
  const uint16_t* const vg = A.vc + (size_t)kvh * A.max_ctx * D;
  auto load_tile = [&](int t, int stage) {
    const int kt0 = k0 + t * FA_KT;
    uint16_t* const ks = fsm + (size_t)stage * 2 * FA_KT * FA_LD;
    for (int c = threadIdx.x; c < 2 * FA_KT * (D / 8); c += blockDim.x) {   
      const int kv = c / (FA_KT * (D / 8)), rem = c - kv * (FA_KT * (D / 8)), key = rem / (D / 8), ch = rem - key * (D / 8);
      uint16_t* const dst = ks + (size_t)kv * FA_KT * FA_LD + key * FA_LD + ch * 8;
      const int kidx = kt0 + key;
      if (kidx < kend) fa_cp16(dst, (kv ? vg : kg) + (size_t)kidx * D + ch * 8);
      else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);   
    }
  };
  if (ntiles > 0) load_tile(0, 0);
  fa_commit();
  unsigned qa[16][4];                                                 
  {
    const bool v0 = r0 + gi < R, v1 = r0 + gi + 8 < R;
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(v0 ? r0 + gi : 0) * Q35_NH + head) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(v1 ? r0 + gi + 8 : 0) * Q35_NH + head) * D);
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = v0 ? q0[ks * 8 + tig] : 0u; qa[ks][1] = v1 ? q1[ks * 8 + tig] : 0u;
      qa[ks][2] = v0 ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = v1 ? q1[ks * 8 + 4 + tig] : 0u;
    }
  }
  float o[32][4];
  #pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  for (int t = 0; t < ntiles; ++t) {
    if (t + 1 < ntiles) load_tile(t + 1, (t + 1) & 1);
    fa_commit(); fa_wait1(); __syncthreads();                        
    const uint16_t* const ks = fsm + (size_t)(t & 1) * 2 * FA_KT * FA_LD;
    const uint16_t* const vs = ks + FA_KT * FA_LD;
    const int kt0 = k0 + t * FA_KT;
    float s[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      #pragma unroll
      for (int kp = 0; kp < 8; ++kp) {                                
        unsigned b[4];
        fa_ldm4(b, ks + (8 * j + (lane & 7)) * FA_LD + 32 * kp + 8 * (lane >> 3));
        fa_mma(s[j], qa[2 * kp], b[0], b[1]);
        fa_mma(s[j], qa[2 * kp + 1], b[2], b[3]);
      }
    }
    float x0 = m0, x1 = m1;                                           
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int key = kt0 + 8 * j + 2 * tig + (c & 1);
        const float v = s[j][c] * 0.0625f;
        const bool ok = key < kend && key <= (c < 2 ? pos0 : pos1);
        s[j][c] = ok ? v : -INFINITY;
      }
      x0 = fmaxf(x0, fmaxf(s[j][0], s[j][1])); x1 = fmaxf(x1, fmaxf(s[j][2], s[j][3]));
    }
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
    const float c0 = exp2f((m0 - x0) * FA_LOG2E), c1 = exp2f((m1 - x1) * FA_LOG2E);
    m0 = x0; m1 = x1;
    const float y0 = x0 * FA_LOG2E, y1 = x1 * FA_LOG2E;
    float r0s = 0.f, r1s = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[j][0] = exp2f(s[j][0] * FA_LOG2E - y0); s[j][1] = exp2f(s[j][1] * FA_LOG2E - y0);
      s[j][2] = exp2f(s[j][2] * FA_LOG2E - y1); s[j][3] = exp2f(s[j][3] * FA_LOG2E - y1);
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
      const unsigned pa[4] = {fa_pack(s[2 * kk][0], s[2 * kk][1]), fa_pack(s[2 * kk][2], s[2 * kk][3]),
                              fa_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), fa_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
      #pragma unroll
      for (int np = 0; np < 16; ++np) {                               
        unsigned b[4];
        fa_ldm4t(b, vs + (16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * FA_LD + 16 * np + (lane >> 4) * 8);
        fa_mma(o[2 * np], pa, b[0], b[1]);
        fa_mma(o[2 * np + 1], pa, b[2], b[3]);
      }
    }
    __syncthreads();                                                  
  }
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = r0 + gi + 8 * h;
    if (row >= R) continue;
    const size_t pidx = ((size_t)row * A.splits + split) * Q35_NH + head;
    if (tig == 0) { A.pm[pidx] = h ? m1 : m0; A.pl[pidx] = h ? l1 : l0; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < 32; ++n) *reinterpret_cast<float2*>(pa + 8 * n) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}




__global__ void __launch_bounds__(192, 1) q35_fa8_kernel(Q35AttnArgs A) {
  constexpr int G = Q35_NH / Q35_KVH, D = Q35_HD;
  extern __shared__ __align__(16) uint16_t fsm[];
  uint16_t* const ktl = fsm; uint16_t* const vtl = fsm + FA_KT * FA_LD;                      
  int8_t* const st8 = reinterpret_cast<int8_t*>(fsm + 2 * FA_KT * FA_LD);                     
  float* const ssc = reinterpret_cast<float*>(st8 + 2 * 2 * FA_KT * D);                       
  const int r0 = blockIdx.x * 16, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gi = lane >> 2, tig = lane & 3, head = kvh * G + warp;
  pdl_wait(); pdl_trigger();
  const int base = *A.d_pos, R = A.M;
  const int L = base + R, per = (L + A.splits - 1) / A.splits, k0 = split * per, k1 = min(L, k0 + per);
  const int kend = min(k1, base + min(R, r0 + 16));
  const int ntiles = kend > k0 ? (kend - k0 + FA_KT - 1) / FA_KT : 0;
  const int pos0 = base + r0 + gi, pos1 = pos0 + 8;
  const int8_t* const kg = A.k8 + (size_t)kvh * A.max_ctx * D; const int8_t* const vg = A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = A.ks + (size_t)kvh * A.max_ctx; const float* const vsg = A.vs + (size_t)kvh * A.max_ctx;
  auto load_tile = [&](int t, int stage) {
    const int kt0 = k0 + t * FA_KT;
    int8_t* const sk = st8 + (size_t)stage * 2 * FA_KT * D;
    for (int c = threadIdx.x; c < 2 * FA_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (FA_KT * (D / 16)), rem = c - kv * (FA_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      int8_t* const dst = sk + (size_t)kv * FA_KT * D + key * D + ch * 16;
      const int kidx = kt0 + key;
      if (kidx < kend) fa_cp16(dst, (kv ? vg : kg) + (size_t)kidx * D + ch * 16);
      else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    if (threadIdx.x < 2 * FA_KT) {
      const int kv = threadIdx.x / FA_KT, key = threadIdx.x % FA_KT, kidx = kt0 + key;
      float* const dst = ssc + (stage * 2 + kv) * FA_KT + key;
      if (kidx < kend) fa_cp4(dst, (kv ? vsg : ksg) + kidx); else *dst = 0.f;
    }
  };
  if (ntiles > 0) load_tile(0, 0);
  fa_commit();
  unsigned qa[16][4];
  {
    const bool v0 = r0 + gi < R, v1 = r0 + gi + 8 < R;
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(v0 ? r0 + gi : 0) * Q35_NH + head) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(v1 ? r0 + gi + 8 : 0) * Q35_NH + head) * D);
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = v0 ? q0[ks * 8 + tig] : 0u; qa[ks][1] = v1 ? q1[ks * 8 + tig] : 0u;
      qa[ks][2] = v0 ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = v1 ? q1[ks * 8 + 4 + tig] : 0u;
    }
  }
  float o[32][4];
  #pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  for (int t = 0; t < ntiles; ++t) {
    if (t + 1 < ntiles) load_tile(t + 1, (t + 1) & 1);
    fa_commit(); fa_wait1(); __syncthreads();
    {                                                                 
      const int8_t* const sk = st8 + (size_t)(t & 1) * 2 * FA_KT * D;
      for (int c = threadIdx.x; c < 2 * FA_KT * (D / 16); c += blockDim.x) {
        const int kv = c / (FA_KT * (D / 16)), rem = c - kv * (FA_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
        const uint4 u = *reinterpret_cast<const uint4*>(sk + (size_t)kv * FA_KT * D + key * D + ch * 16);
        const unsigned w[4] = {u.x, u.y, u.z, u.w}; unsigned b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) { const unsigned ww = w[k >> 1] >> (16 * (k & 1));
          b[k] = fa_pack((float)(signed char)(ww & 0xffu), (float)(signed char)((ww >> 8) & 0xffu)); }
        uint16_t* const dst = (kv ? vtl : ktl) + key * FA_LD + ch * 16;
        *reinterpret_cast<uint4*>(dst) = make_uint4(b[0], b[1], b[2], b[3]); *reinterpret_cast<uint4*>(dst + 8) = make_uint4(b[4], b[5], b[6], b[7]);
      }
    }
    __syncthreads();
    const float* const ksc = ssc + ((t & 1) * 2 + 0) * FA_KT; const float* const vsc = ssc + ((t & 1) * 2 + 1) * FA_KT;
    const int kt0 = k0 + t * FA_KT;
    float s[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      #pragma unroll
      for (int kp = 0; kp < 8; ++kp) {
        unsigned b[4];
        fa_ldm4(b, ktl + (8 * j + (lane & 7)) * FA_LD + 32 * kp + 8 * (lane >> 3));
        fa_mma(s[j], qa[2 * kp], b[0], b[1]);
        fa_mma(s[j], qa[2 * kp + 1], b[2], b[3]);
      }
    }
    float x0 = m0, x1 = m1;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk;
        const float v = s[j][c] * (ksc[kk] * 0.0625f);
        const bool ok = key < kend && key <= (c < 2 ? pos0 : pos1);
        s[j][c] = ok ? v : -INFINITY;
      }
      x0 = fmaxf(x0, fmaxf(s[j][0], s[j][1])); x1 = fmaxf(x1, fmaxf(s[j][2], s[j][3]));
    }
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
    const float c0 = exp2f((m0 - x0) * FA_LOG2E), c1 = exp2f((m1 - x1) * FA_LOG2E);
    m0 = x0; m1 = x1;
    const float y0 = x0 * FA_LOG2E, y1 = x1 * FA_LOG2E;
    float r0s = 0.f, r1s = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[j][0] = exp2f(s[j][0] * FA_LOG2E - y0); s[j][1] = exp2f(s[j][1] * FA_LOG2E - y0);
      s[j][2] = exp2f(s[j][2] * FA_LOG2E - y1); s[j][3] = exp2f(s[j][3] * FA_LOG2E - y1);
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
      const float va = vsc[16 * kk + 2 * tig], vb = vsc[16 * kk + 2 * tig + 1], vc2 = vsc[16 * kk + 8 + 2 * tig], vd2 = vsc[16 * kk + 8 + 2 * tig + 1];
      const unsigned pa[4] = {fa_pack(s[2 * kk][0] * va, s[2 * kk][1] * vb), fa_pack(s[2 * kk][2] * va, s[2 * kk][3] * vb),
                              fa_pack(s[2 * kk + 1][0] * vc2, s[2 * kk + 1][1] * vd2), fa_pack(s[2 * kk + 1][2] * vc2, s[2 * kk + 1][3] * vd2)};
      #pragma unroll
      for (int np = 0; np < 16; ++np) {
        unsigned b[4];
        fa_ldm4t(b, vtl + (16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * FA_LD + 16 * np + (lane >> 4) * 8);
        fa_mma(o[2 * np], pa, b[0], b[1]);
        fa_mma(o[2 * np + 1], pa, b[2], b[3]);
      }
    }
    __syncthreads();                                                  
  }
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = r0 + gi + 8 * h;
    if (row >= R) continue;
    const size_t pidx = ((size_t)row * A.splits + split) * Q35_NH + head;
    if (tig == 0) { A.pm[pidx] = h ? m1 : m0; A.pl[pidx] = h ? l1 : l0; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < 32; ++n) *reinterpret_cast<float2*>(pa + 8 * n) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}
void launch_q35_fa(const Q35AttnArgs& A, cudaStream_t st) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(q35_fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FA_SMEM);
              cudaFuncSetAttribute(q35_fa8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FA_SMEM); attr = true; }
  if (A.k8) launch_k(q35_fa8_kernel, dim3((A.M + 15) / 16, A.splits, Q35_KVH), dim3(192), (size_t)FA_SMEM, st, A);
  else launch_k(q35_fa_kernel, dim3((A.M + 15) / 16, A.splits, Q35_KVH), dim3(192), (size_t)FA_SMEM, st, A);
}
}  
