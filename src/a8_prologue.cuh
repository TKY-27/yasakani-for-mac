// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
namespace yskn {



__device__ __forceinline__ long long xsum_combine(const float* part, int nw, int M, int N, int tok, int n) {
  (void)nw; (void)M; long long* p = const_cast<long long*>(reinterpret_cast<const long long*>(part)) + (size_t)tok * N + n;
  const long long s = __ldcg(p); __stcg(p, 0ll); return s;
}

__device__ __forceinline__ float xsum_y(long long acc, float us) { return __fmul_rn(__ll2float_rn(acc), us); }

__device__ __forceinline__ float xsum_r(const float* S8, float eps) { float s = 0.f;
  #pragma unroll
  for (int b = 0; b < 7; ++b) s = __fadd_rn(s, S8[b]);
  return rsqrtf(__fadd_rn(__fdiv_rn(s, 3584.f), eps)); }


__device__ __forceinline__ float xsum_block_ss(const float* hb, float* s_red4) {
  const int t = threadIdx.x; float acc = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) { const float x = hb[t + 128 * j]; acc = __fadd_rn(acc, __fmul_rn(x, x)); }
  for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  if ((t & 31) == 0) s_red4[t >> 5] = acc;
  __syncthreads();
  float s = 0.f;
  #pragma unroll
  for (int w = 0; w < 4; ++w) s = __fadd_rn(s, s_red4[w]);
  __syncthreads();
  return s; }
struct XformDesc { const int8_t* signs; const uint16_t* qm; int n; int m; };   
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ float round_f16(float f) { return __half2float(__float2half_rn(f)); }


__device__ inline void fwht512_blocks(float* buf, int m) {
  for (int h = 1; h < 512; h <<= 1) {
    for (int i = threadIdx.x; i < m * 256; i += blockDim.x) {
      const int blk = i >> 8, k = i & 255, lo = ((k & ~(h - 1)) << 1) | (k & (h - 1));
      float* x = buf + blk * 512; const float a = x[lo], b = x[lo + h]; x[lo] = a + b; x[lo + h] = a - b;
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < m * 512; i += blockDim.x) buf[i] = __fmul_rn(buf[i], 0.044194173824159216f);
  __syncthreads();
}

__device__ inline void qm_mix_inplace(float* buf, const uint16_t* qm, int m, bool transpose) {
  for (int p = threadIdx.x; p < 512; p += blockDim.x) {
    float v[64];
    for (int b = 0; b < m; ++b) v[b] = buf[b * 512 + p];
    for (int a = 0; a < m; ++a) { float s = 0.f;
      for (int b = 0; b < m; ++b) s = __fadd_rn(s, __fmul_rn(h2f(transpose ? qm[b * m + a] : qm[a * m + b]), v[b]));
      buf[a * 512 + p] = s; }
  }
  __syncthreads();
}

__device__ inline void a8_from_fp32(float* s_x, const XformDesc& X, int8_t* s_qx, float* s_alpha) {
  __shared__ unsigned s_amax[64];
  for (int i = threadIdx.x; i < X.n; i += blockDim.x) s_x[i] = __fmul_rn(s_x[i], (float)X.signs[i]);
  __syncthreads();
  fwht512_blocks(s_x, X.m);
  if (X.qm) qm_mix_inplace(s_x, X.qm, X.m, false);
  for (int i = threadIdx.x; i < 64; i += blockDim.x) s_amax[i] = 0u;
  __syncthreads();
  for (int i = threadIdx.x; i < X.n; i += blockDim.x) { const float v = round_f16(s_x[i]); s_x[i] = v; atomicMax(&s_amax[i >> 9], __float_as_uint(fabsf(v))); }
  __syncthreads();
  for (int g = threadIdx.x; g < X.m; g += blockDim.x) s_alpha[g] = __fdiv_rn(__uint_as_float(s_amax[g]), 127.f);
  __syncthreads();
  for (int i = threadIdx.x; i < X.n; i += blockDim.x) {
    const float a = s_alpha[i >> 9]; const float v = a > 0.f ? __fdiv_rn(s_x[i], a) : 0.f;
    int q = __float2int_rn(v); q = q < -127 ? -127 : (q > 127 ? 127 : q); s_qx[i] = (int8_t)q;
  }
  __syncthreads();
}





__device__ __forceinline__ float fwht512_v(float v, float* buf0, float* buf1) {
  const int p = threadIdx.x, lane = p & 31;
  #pragma unroll
  for (int h = 1; h < 32; h <<= 1) { const float q = __shfl_xor_sync(0xffffffffu, v, h); v = (lane & h) ? (q - v) : (v + q); }
  #pragma unroll
  for (int h = 32; h < 512; h <<= 1) {
    float* const b = (h == 32 || h == 128) ? buf0 : buf1;
    b[p] = v; __syncthreads();
    const float q = b[p ^ h]; v = (p & h) ? (q - v) : (v + q);
  }
  return v;
}


__device__ __forceinline__ void fwht512_v2(float& va, float& vb, float* a0, float* a1, float* b0, float* b1) {
  const int p = threadIdx.x, lane = p & 31;
  #pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
    const float qa = __shfl_xor_sync(0xffffffffu, va, h), qb = __shfl_xor_sync(0xffffffffu, vb, h);
    va = (lane & h) ? (qa - va) : (va + qa); vb = (lane & h) ? (qb - vb) : (vb + qb);
  }
  #pragma unroll
  for (int h = 32; h < 512; h <<= 1) {
    float* const ba = (h == 32 || h == 128) ? a0 : a1; float* const bb = (h == 32 || h == 128) ? b0 : b1;
    ba[p] = va; bb[p] = vb; __syncthreads();
    const float qa = ba[p ^ h], qb = bb[p ^ h];
    va = (p & h) ? (qa - va) : (va + qa); vb = (p & h) ? (qb - vb) : (vb + qb);
  }
}





__device__ __forceinline__ int a8_lo_off(int i, int t) { return ((i >> 8) << 11) | (((i >> 4) & 3) << 9) | (t << 6) | (((i >> 6) & 3) << 4) | (i & 15); }
__device__ inline void a8_group512_tail(float* buf, unsigned* s_amax, int8_t* qx, float* alpha, int lo_tok = -1) {   
  __shared__ float fw_tmp[512];
  const int tid = threadIdx.x;
  const float v = round_f16(__fmul_rn(fwht512_v(buf[tid], buf, fw_tmp), 0.044194173824159216f));
  unsigned mx = __float_as_uint(fabsf(v));                               
  for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((tid & 31) == 0) atomicMax(s_amax, mx);
  __syncthreads();
  const float al = __fdiv_rn(__uint_as_float(*s_amax), 127.f);
  if (tid == 0) *alpha = al;
  const float r = al > 0.f ? __fdiv_rn(v, al) : 0.f; int q = __float2int_rn(r); q = q < -127 ? -127 : (q > 127 ? 127 : q); qx[lo_tok >= 0 ? a8_lo_off(tid, lo_tok) : tid] = (int8_t)q;
}

__device__ inline void norm_apply(const float* h, const uint16_t* gamma, float eps, int n, float* s_x) {
  __shared__ float s_red[64]; float acc = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc = __fadd_rn(acc, __fmul_rn(h[i], h[i]));
  for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = acc;
  __syncthreads();
  float tot = 0.f; for (int w = 0; w < (int)(blockDim.x >> 5); ++w) tot = __fadd_rn(tot, s_red[w]);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)n), eps));
  for (int i = threadIdx.x; i < n; i += blockDim.x) s_x[i] = round_f16(__fmul_rn(round_f16(__fmul_rn(h[i], r)), h2f(gamma[i])));
  __syncthreads();
}
} 
