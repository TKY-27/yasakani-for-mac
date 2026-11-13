// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "kvx_codec.cuh"
#include "pdl.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "a8_prologue.cuh"
#include "q35.hpp"
#include "kz_common.cuh"
#include "pm_math.cuh"
#if YSKN_KVX_MODE == 3
#include "kvq_fast.cuh"   
#endif
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
namespace yskn {

__device__ __forceinline__ void q35_l2pf(const Q35Pf& f, unsigned part, unsigned np) {
  #pragma unroll
  for (int s = 0; s < 2; ++s) {
    if (!f.p[s] || !f.n[s]) continue;
    const unsigned long long a = (f.n[s] * part / np) & ~15ull, b = (f.n[s] * (part + 1) / np) & ~15ull;
    for (unsigned long long o = a; o < b; o += 65520ull) {
      const unsigned len = (unsigned)(b - o < 65520ull ? b - o : 65520ull);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(static_cast<const char*>(f.p[s]) + o), "r"(len) : "memory");
    }
  }
}
} 
namespace yskn {



__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

__device__ __forceinline__ size_t q35_kvrow(int stg, int max_ctx, int kvh, int pos, int tok) { return stg ? (size_t)kvh * (size_t)stg + (size_t)tok : (size_t)kvh * (size_t)max_ctx + (size_t)pos; }


__constant__ int q35_pm_on;
__device__ __forceinline__ float q35_rsqrt(float x) { return q35_pm_on ? pm_rsqrt(x) : rsqrtf(x); }
__device__ __forceinline__ float q35_exp(float x) { return q35_pm_on ? pm_expf(x) : expf(x); }



template <int NW>
__device__ __forceinline__ float q35_combine_n(const float* part, int M, int N, int tok, int n) {
  float v[NW];
  #pragma unroll
  for (int w = 0; w < NW; ++w) v[w] = part[((size_t)w * M + tok) * N + n];
  float s = v[0];
  #pragma unroll
  for (int w = 1; w < NW; ++w) s = __fadd_rn(s, v[w]);
  return s;
}
__device__ __forceinline__ float q35_combine(const float* part, int nw, int M, int N, int tok, int n) {
  switch (nw) {
    case 1: return part[(size_t)tok * N + n];
    case 5: return q35_combine_n<5>(part, M, N, tok, n);
    case 10: return q35_combine_n<10>(part, M, N, tok, n);
    case 12: return q35_combine_n<12>(part, M, N, tok, n);
    case 17: return q35_combine_n<17>(part, M, N, tok, n);
    default: break;
  }
  float s = part[(size_t)tok * N + n];
  for (int w = 1; w < nw; ++w) s = __fadd_rn(s, part[((size_t)w * M + tok) * N + n]);     
  return s;
}


__device__ __forceinline__ void q35_inv_h512s(float* buf, float sgn) {   
  __shared__ float fw_tmp[512];
  const float v = fwht512_v(buf[threadIdx.x], buf, fw_tmp);           
  buf[threadIdx.x] = __fmul_rn(__fmul_rn(v, 0.044194173824159216f), sgn);
  __syncthreads();
}
__device__ __forceinline__ void q35_inv_h512(float* buf, const int8_t* signs) { q35_inv_h512s(buf, (float)signs[threadIdx.x]); }

__device__ __forceinline__ float q35_xsm_r(const float* S, float eps) {
  float s = 0.f;
  #pragma unroll
  for (int b = 0; b < Q35_H / 512; ++b) s = __fadd_rn(s, __ldcg(S + b));
  return q35_rsqrt(__fadd_rn(__fdiv_rn(s, (float)Q35_H), eps)); }
__device__ __forceinline__ float q35_xsm_y(const float* part, int N, int tok, int n, float us) { return xsum_y(xsum_combine(part, 0, 0, N, tok, n), us); }


__device__ __forceinline__ long long q35_xsm_ld(const float* part, int mode, int nw, int M, int N, int tok, int n) {
  if (mode == 1 || mode == 3) return (long long)__float_as_uint(__ldcg(part + (size_t)tok * N + n));
  if (mode == 2) {                                                         
    const long long* p = reinterpret_cast<const long long*>(part) + (size_t)tok * N + n; const size_t st = (size_t)M * N;
    long long v[24];
    #pragma unroll
    for (int w = 0; w < 24; ++w) v[w] = w < nw ? __ldcg(p + w * st) : 0ll;
    long long sum = 0;
    #pragma unroll
    for (int w = 0; w < 24; ++w) sum += v[w];
    return sum; }
  return xsum_combine(part, 0, 0, N, tok, n); }
__device__ __forceinline__ float q35_xsm_fin(long long v, int mode, float us) { return mode == 1 ? __uint_as_float((unsigned)v) : xsum_y(v, us); }


__device__ __forceinline__ float2 q35_xsm_row(const float* us, const float* S, float eps, int tok) {
  __shared__ float2 s_xsm;
  if (threadIdx.x == 0) s_xsm = make_float2(us[tok], S ? q35_xsm_r(S + (size_t)tok * Q35_SST, eps) : 0.f);
  __syncthreads();
  return s_xsm; }




__device__ __forceinline__ float q35_silu(float x) {                                                
  return q35_pm_on ? pm_silu(x) : __fdiv_rn(x, __fadd_rn(1.f, expf(-x))); }
__device__ __forceinline__ float q35_sigmoid(float x) { return q35_pm_on ? pm_sigmoid(x) : __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }
__device__ __forceinline__ float q35_softplus(float x) { return q35_pm_on ? pm_softplus(x) : (x > 20.f ? x : log1pf(expf(x))); }   


__global__ void __launch_bounds__(512) q35_res_norm_kernel(Q35ResNormArgs A) {
  __shared__ float buf[512]; __shared__ float s_red[4]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, N = A.N, nb = N / 512, n = blk * 512 + tid;
  float* const hr = A.h + (size_t)tok * N;
  float c_tau = 0.f, c_h = 0.f; int8_t c_so = 0;
  if (A.part) { c_tau = A.tau[n]; c_so = A.sg_out[n]; c_h = __ldcg(hr + n); }   
  const uint16_t c_w = A.w[n]; const int8_t c_si = A.sg_in[n];
  pdl_wait();
  const uint16_t* const er = A.emb ? A.emb + (size_t)A.ids[tok] * N : nullptr;
  if (A.part) {
    buf[tid] = __fmul_rn(c_tau, q35_combine(A.part, A.nw, A.M, N, tok, n)); __syncthreads();
    q35_inv_h512s(buf, (float)c_so);
    hr[n] = rbf(__fadd_rn(c_h, rbf(buf[tid])));
    __threadfence(); __syncthreads();
    if (tid == 0) {                                                       
      const unsigned old = atomicAdd(A.cnt + tok, 1u), target = (old / (unsigned)nb + 1u) * (unsigned)nb;
      while ((int)(*reinterpret_cast<volatile unsigned*>(A.cnt + tok) - target) < 0) __nanosleep(40);
      __threadfence();
    }
    __syncthreads();
  } else if (er && !A.keep_h) hr[n] = bf2f(er[n]);
  pdl_trigger();
  auto xval = [&](int i) { return er ? bf2f(er[i]) : __ldcg(hr + i); };
  if (tid < 128) {                                                         
    float acc = 0.f;
    if (N == Q35_H) {                                                      
      float xs[Q35_H / 128];
      #pragma unroll
      for (int j = 0; j < Q35_H / 128; ++j) xs[j] = xval(tid + 128 * j);
      #pragma unroll
      for (int j = 0; j < Q35_H / 128; ++j) acc = __fadd_rn(acc, __fmul_rn(xs[j], xs[j]));
    } else for (int i = tid; i < N; i += 128) { const float x = xval(i); acc = __fadd_rn(acc, __fmul_rn(x, x)); }
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red[tid >> 5] = acc;
  }
  if (tid == 0) s_amax = 0u;
  __syncthreads();
  float tot = 0.f; for (int w = 0; w < 4; ++w) tot = __fadd_rn(tot, s_red[w]);
  const float r = q35_rsqrt(__fadd_rn(__fdiv_rn(tot, (float)N), A.eps));
  const float v = rbf(__fmul_rn(__fmul_rn(xval(n), r), __fadd_rn(1.f, bf2f(c_w))));
  if (A.xn) A.xn[(size_t)tok * N + n] = f2bf(v);
  buf[tid] = __fmul_rn(v, (float)c_si);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, A.qx + (size_t)tok * (A.qx_stride ? A.qx_stride : N) + blk * 512, A.alpha + (size_t)tok * (A.al_stride ? A.al_stride : nb) + blk);
}
void launch_q35_res_norm(const Q35ResNormArgs& A, cudaStream_t st) { launch_k(q35_res_norm_kernel, dim3(A.N / 512, A.M), dim3(512), 0, st, A); }




__global__ void __launch_bounds__(512) q35_res_norm_row_kernel(Q35ResNormArgs A) {
  pdl_wait(); pdl_trigger();
  extern __shared__ float hrow[];
  __shared__ float buf[512]; __shared__ float s_red[4]; __shared__ unsigned s_amax;
  const int tok = blockIdx.x, tid = threadIdx.x, N = A.N, nb = N / 512;
  float* const hr = A.h + (size_t)tok * N;
  if (A.part) {
    for (int b = 0; b < nb; ++b) { const int n = b * 512 + tid; hrow[n] = __fmul_rn(A.tau[n], q35_combine(A.part, A.nw, A.M, N, tok, n)); }
    __syncthreads();
    for (int hh = 1; hh < 512; hh <<= 1) {
      for (int i = tid; i < nb * 256; i += 512) { const int b = i >> 8, k = i & 255, lo = ((k & ~(hh - 1)) << 1) | (k & (hh - 1)); float* x = hrow + b * 512;
        const float a = x[lo], c = x[lo + hh]; x[lo] = a + c; x[lo + hh] = a - c; }
      __syncthreads();
    }
    for (int b = 0; b < nb; ++b) { const int n = b * 512 + tid;
      const float y = __fmul_rn(__fmul_rn(hrow[n], 0.044194173824159216f), (float)A.sg_out[n]);
      const float hn = rbf(__fadd_rn(hr[n], rbf(y))); hr[n] = hn; hrow[n] = hn; }
  } else if (A.emb) {
    const uint16_t* er = A.emb + (size_t)A.ids[tok] * N;
    for (int b = 0; b < nb; ++b) { const int n = b * 512 + tid; const float x = bf2f(er[n]); hrow[n] = x; if (!A.keep_h) hr[n] = x; }
  } else {
    for (int b = 0; b < nb; ++b) { const int n = b * 512 + tid; hrow[n] = hr[n]; }
  }
  __syncthreads();
  if (tid < 128) {
    float acc = 0.f; for (int i = tid; i < N; i += 128) acc = __fadd_rn(acc, __fmul_rn(hrow[i], hrow[i]));
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red[tid >> 5] = acc;
  }
  __syncthreads();
  float tot = 0.f; for (int w = 0; w < 4; ++w) tot = __fadd_rn(tot, s_red[w]);
  const float r = q35_rsqrt(__fadd_rn(__fdiv_rn(tot, (float)N), A.eps));
  int8_t* const qxr = A.qx + (size_t)tok * (A.qx_stride ? A.qx_stride : N); float* const alr = A.alpha + (size_t)tok * (A.al_stride ? A.al_stride : nb);
  for (int g = 0; g < nb; ++g) {
    const int n = g * 512 + tid;
    const float v = rbf(__fmul_rn(__fmul_rn(hrow[n], r), __fadd_rn(1.f, bf2f(A.w[n]))));
    if (A.xn) A.xn[(size_t)tok * N + n] = f2bf(v);
    if (tid == 0) s_amax = 0u;
    buf[tid] = __fmul_rn(v, (float)A.sg_in[n]);
    __syncthreads();
    a8_group512_tail(buf, &s_amax, qxr + g * 512, alr + g);
    __syncthreads();
  }
}
void launch_q35_res_norm_row(const Q35ResNormArgs& A, cudaStream_t st) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(q35_res_norm_row_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Q35_H * 4); attr = true; }
  launch_k(q35_res_norm_row_kernel, dim3(A.M), dim3(512), (size_t)A.N * 4, st, A);
}
__global__ void __launch_bounds__(512) q35_residual_kernel(Q35ResNormArgs A) {
  pdl_wait(); pdl_trigger();
  __shared__ float buf[512];
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, N = A.N, n = blk * 512 + tid;
  float* const hr = A.h + (size_t)tok * N;
  buf[tid] = __fmul_rn(A.tau[n], q35_combine(A.part, A.nw, A.M, N, tok, n)); __syncthreads();
  q35_inv_h512(buf, A.sg_out + blk * 512);
  hr[n] = rbf(__fadd_rn(hr[n], rbf(buf[tid])));
}
void launch_q35_residual(const Q35ResNormArgs& A, cudaStream_t st) { launch_k(q35_residual_kernel, dim3(A.N / 512, A.M), dim3(512), 0, st, A); }


__device__ __forceinline__ uint4 q35_ab_x(const Q35AbArgs& A, int tok, int idx) { return __ldcg(reinterpret_cast<const uint4*>(A.xn + (size_t)tok * Q35_H) + idx); }














static int q35_pfalt() { static const int v = [] { const char* e = std::getenv("YSKN_Q35_PFALT"); return e && e[0] ? std::atoi(e) : 0; }(); return v; }
constexpr float Q35_K512 = 0.044194173824159216f;
__device__ __forceinline__ void q35_wfwht512(float (&x)[16]) {
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) { const float q = __shfl_xor_sync(0xffffffffu, x[i], h); x[i] = (lane & h) ? (q - x[i]) : (x[i] + q); }
  }
  #pragma unroll
  for (int k = 1; k < 16; k <<= 1) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) if (!(i & k)) { const float a = x[i], b = x[i | k]; x[i] = a + b; x[i | k] = a - b; }
  }
}

__device__ __forceinline__ void q35_wa8_tail(float (&x)[16], int8_t* qx, float* alpha) {
  q35_wfwht512(x);
  const int lane = threadIdx.x & 31; unsigned mx = 0u;
  #pragma unroll
  for (int i = 0; i < 16; ++i) { x[i] = round_f16(__fmul_rn(x[i], Q35_K512)); mx = max(mx, __float_as_uint(fabsf(x[i]))); }
  for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const float al = __fdiv_rn(__uint_as_float(mx), 127.f);
  if (lane == 0) *alpha = al;
  #pragma unroll
  for (int i = 0; i < 16; ++i) { const float r = al > 0.f ? __fdiv_rn(x[i], al) : 0.f; int q = __float2int_rn(r); q = q < -127 ? -127 : (q > 127 ? 127 : q);
                                 qx[lane + 32 * i] = (int8_t)q; }
}
constexpr int Q35_PF_TPW = 4;                                            

__global__ void __launch_bounds__(256) q35_mlp_pf_kernel(Q35MlpArgs A) {
  pdl_wait(); pdl_trigger();
  const int NB = A.F / 512, lane = threadIdx.x & 31, task = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int ntt = (A.M + Q35_PF_TPW - 1) / Q35_PF_TPW; if (task >= NB * ntt) return;
  const int blk = task % NB, t0 = (task / NB) * Q35_PF_TPW;
  float tg[16], tu[16], sg[16], su[16], si[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { const int n = blk * 512 + lane + 32 * i;
    tg[i] = A.tau_g[n]; tu[i] = A.tau_u[n]; sg[i] = (float)A.sg_g[n]; su[i] = (float)A.sg_u[n]; si[i] = (float)A.sg_in[n]; }
  for (int tk = t0; tk < min(A.M, t0 + Q35_PF_TPW); ++tk) {
    const float* pg = A.part_g + (size_t)tk * A.F + blk * 512; const float* pu = A.part_u + (size_t)tk * A.F + blk * 512;
    float g[16], u[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { g[i] = __fmul_rn(__ldcg(pg + lane + 32 * i), tg[i]); u[i] = __fmul_rn(__ldcg(pu + lane + 32 * i), tu[i]); }
    q35_wfwht512(g); q35_wfwht512(u);
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float gv = rbf(__fmul_rn(__fmul_rn(g[i], Q35_K512), sg[i])), uv = rbf(__fmul_rn(__fmul_rn(u[i], Q35_K512), su[i]));
      x[i] = __fmul_rn(rbf(__fmul_rn(rbf(q35_silu(gv)), uv)), si[i]);
    }
    q35_wa8_tail(x, A.qx + (size_t)tk * A.F + blk * 512, A.alpha + (size_t)tk * NB + blk);
  }
}

__global__ void __launch_bounds__(256) q35_mix_pf_kernel(Q35MixArgs A) {
  pdl_wait(); pdl_trigger();
  constexpr int NQ = Q35_CONV / 512, NB = NQ + Q35_Z / 512;
  const int lane = threadIdx.x & 31, task = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int ntt = (A.M + Q35_PF_TPW - 1) / Q35_PF_TPW; if (task >= NB * ntt) return;
  const int blk = task % NB, t0 = (task / NB) * Q35_PF_TPW; const bool isq = blk < NQ; const int b2 = isq ? blk : blk - NQ;
  const float* tau = isq ? A.tau_qkv : A.tau_z; const int8_t* sgv = isq ? A.sg_qkv : A.sg_z;
  float ta[16], sg[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { const int n = b2 * 512 + lane + 32 * i; ta[i] = tau[n]; sg[i] = (float)sgv[n]; }
  const int N = isq ? Q35_CONV : Q35_Z;
  for (int tk = t0; tk < min(A.M, t0 + Q35_PF_TPW); ++tk) {
    const float* p = (isq ? A.part_qkv : A.part_z) + (size_t)tk * N + b2 * 512;
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = __fmul_rn(__ldcg(p + lane + 32 * i), ta[i]);
    q35_wfwht512(x);
    #pragma unroll
    for (int i = 0; i < 16; ++i) { const float v = __fmul_rn(__fmul_rn(x[i], Q35_K512), sg[i]); const int n = b2 * 512 + lane + 32 * i;
      if (isq) A.mixed[(size_t)tk * Q35_CONV + n] = f2bf(v); else A.z[(size_t)tk * Q35_Z + n] = rbf(v); }
  }
}

__global__ void __launch_bounds__(256) q35_res_norm_pf_kernel(Q35XsmResArgs A) {
  pdl_wait(); pdl_trigger();
  constexpr int NB = Q35_H / 512;
  const int lane = threadIdx.x & 31, task = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int ntt = (A.M + Q35_PF_TPW - 1) / Q35_PF_TPW; if (task >= NB * ntt) return;
  const int blk = task % NB, t0 = (task / NB) * Q35_PF_TPW;
  float ta[16], so[16], w1[16], si[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { const int n = blk * 512 + lane + 32 * i;
    ta[i] = A.part ? A.tau[n] : 0.f; so[i] = A.part ? (float)A.sg_out[n] : 0.f; w1[i] = __fadd_rn(1.f, bf2f(A.w[n])); si[i] = (float)A.sg_in[n]; }
  for (int tk = t0; tk < min(A.M, t0 + Q35_PF_TPW); ++tk) {
    float* const hr = A.h + (size_t)tk * Q35_H + blk * 512;
    float hn[16];
    if (A.part) {
      const float* p = A.part + (size_t)tk * Q35_H + blk * 512;
      float x[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) x[i] = __fmul_rn(__ldcg(p + lane + 32 * i), ta[i]);
      q35_wfwht512(x);
      #pragma unroll
      for (int i = 0; i < 16; ++i) hn[i] = rbf(__fadd_rn(__ldcg(hr + lane + 32 * i), rbf(__fmul_rn(__fmul_rn(x[i], Q35_K512), so[i]))));
    } else {
      const uint16_t* e = A.emb + (size_t)A.ids[tk] * Q35_H + blk * 512;
      #pragma unroll
      for (int i = 0; i < 16; ++i) hn[i] = bf2f(e[lane + 32 * i]);
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) hr[lane + 32 * i] = hn[i];
    float acc[4];                                                          
    #pragma unroll
    for (int w = 0; w < 4; ++w) { float a = 0.f;
      #pragma unroll
      for (int j = 0; j < 4; ++j) a = __fadd_rn(a, __fmul_rn(hn[w + 4 * j], hn[w + 4 * j]));
      acc[w] = a; }
    #pragma unroll
    for (int w = 0; w < 4; ++w) for (int o = 16; o; o >>= 1) acc[w] = __fadd_rn(acc[w], __shfl_xor_sync(0xffffffffu, acc[w], o));
    if (lane == 0) { float sb = 0.f;
      #pragma unroll
      for (int w = 0; w < 4; ++w) sb = __fadd_rn(sb, acc[w]);
      A.S[(size_t)tk * Q35_SST + blk] = sb; }
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { const float u = __fmul_rn(hn[i], w1[i]);
      if (A.xu) A.xu[(size_t)tk * Q35_H + blk * 512 + lane + 32 * i] = f2bf(u);
      x[i] = __fmul_rn(u, si[i]); }
    q35_wa8_tail(x, A.qx + (size_t)tk * Q35_H + blk * 512, A.alpha + (size_t)tk * NB + blk);
  }
}

__global__ void __launch_bounds__(256) q35_gdn_out_pf_kernel(Q35GdnOutArgs A) {
  pdl_wait(); pdl_trigger();
  constexpr int NB = Q35_Z / 512;
  const int lane = threadIdx.x & 31, task = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int ntt = (A.M + Q35_PF_TPW - 1) / Q35_PF_TPW; if (task >= NB * ntt) return;
  const int blk = task % NB, t0 = (task / NB) * Q35_PF_TPW;
  float gw[16], si[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { const int e = blk * 512 + lane + 32 * i; gw[i] = bf2f(A.gw[e & (Q35_GD - 1)]); si[i] = (float)A.sg_in[e]; }
  for (int tk = t0; tk < min(A.M, t0 + Q35_PF_TPW); ++tk) {
    const float* cr = A.core + (size_t)tk * Q35_Z + blk * 512; const float* zr = A.z + (size_t)tk * Q35_Z + blk * 512;
    float c[16], ss[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { c[i] = cr[lane + 32 * i]; ss[i] = __fmul_rn(c[i], c[i]); }
    #pragma unroll
    for (int i = 0; i < 16; ++i) for (int o = 16; o; o >>= 1) ss[i] = __fadd_rn(ss[i], __shfl_xor_sync(0xffffffffu, ss[i], o));
    float r[4];
    #pragma unroll
    for (int hh = 0; hh < 4; ++hh) { const float tot = __fadd_rn(__fadd_rn(__fadd_rn(ss[4 * hh], ss[4 * hh + 1]), ss[4 * hh + 2]), ss[4 * hh + 3]);
      r[hh] = q35_rsqrt(__fadd_rn(__fdiv_rn(tot, (float)Q35_GD), A.eps)); }
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { const float n2 = rbf(__fmul_rn(gw[i], rbf(__fmul_rn(c[i], r[i >> 2]))));
      const float out = rbf(__fmul_rn(n2, q35_silu(zr[lane + 32 * i]))); x[i] = __fmul_rn(out, si[i]); }
    q35_wa8_tail(x, A.qx + (size_t)tk * Q35_Z + blk * 512, A.alpha + (size_t)tk * NB + blk);
  }
}

template <int R> __global__ void __launch_bounds__(128) q35_ab_pf_kernel(Q35AbArgs A) {
  __shared__ float s_red[16][R][4], s_rq[16];
  const int r0 = blockIdx.x * R, tid = threadIdx.x, t0 = blockIdx.y * 16, nt = min(16, A.M - t0);
  uint4 wv[R][5];
  #pragma unroll
  for (int i = 0; i < R; ++i) { const int r = r0 + i; const uint16_t* w = r < 48 ? A.wb + (size_t)r * Q35_H : A.wa + (size_t)(r - 48) * Q35_H;
    #pragma unroll
    for (int j = 0; j < 5; ++j) wv[i][j] = __ldg(reinterpret_cast<const uint4*>(w) + tid + 128 * j); }
  pdl_wait(); pdl_trigger();
  if (A.S && tid < nt) s_rq[tid] = q35_xsm_r(A.S + (size_t)(t0 + tid) * Q35_SST, A.eps);
  uint4 xv[5];
  #pragma unroll
  for (int j = 0; j < 5; ++j) xv[j] = q35_ab_x(A, t0, tid + 128 * j);
  for (int u = 0; u < nt; ++u) {
    uint4 xn[5];
    if (u + 1 < nt) {
      #pragma unroll
      for (int j = 0; j < 5; ++j) xn[j] = q35_ab_x(A, t0 + u + 1, tid + 128 * j);
    }
    float acc[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) acc[i] = 0.f;
    #pragma unroll
    for (int j = 0; j < 5; ++j) {
      const unsigned xa[4] = {xv[j].x, xv[j].y, xv[j].z, xv[j].w};
      #pragma unroll
      for (int i = 0; i < R; ++i) {
        const unsigned wa[4] = {wv[i][j].x, wv[i][j].y, wv[i][j].z, wv[i][j].w};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc[i] = __fadd_rn(acc[i], __fmul_rn(__uint_as_float(wa[e] << 16), __uint_as_float(xa[e] << 16)));
          acc[i] = __fadd_rn(acc[i], __fmul_rn(__uint_as_float(wa[e] & 0xffff0000u), __uint_as_float(xa[e] & 0xffff0000u)));
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < R; ++i) { for (int o = 16; o; o >>= 1) acc[i] = __fadd_rn(acc[i], __shfl_xor_sync(0xffffffffu, acc[i], o));
                                  if ((tid & 31) == 0) s_red[u][i][tid >> 5] = acc[i]; }
    if (u + 1 < nt) {
      #pragma unroll
      for (int j = 0; j < 5; ++j) xv[j] = xn[j];
    }
  }
  __syncthreads();
  for (int idx = tid; idx < nt * R; idx += 128) { const int u = idx / R, i = idx % R;
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(s_red[u][i][0], s_red[u][i][1]), s_red[u][i][2]), s_red[u][i][3]);
    A.ab[(size_t)(t0 + u) * 96 + r0 + i] = rbf(A.S ? __fmul_rn(sum, s_rq[u]) : sum); }
}



constexpr int Q35_ST_TS = 16, Q35_ST_KQ = 4 * 36;                                      
__global__ void __launch_bounds__(128) q35_state_pf_kernel(Q35StateArgs A) {
  __shared__ __align__(16) float sk[2][Q35_ST_TS][Q35_ST_KQ], sq[2][Q35_ST_TS][Q35_ST_KQ], sv[2][Q35_ST_TS][32], seg[2][Q35_ST_TS], sbt[2][Q35_ST_TS];
  const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid & 3, jj = tid >> 2, j = blockIdx.y * 32 + jj, kh = h / (Q35_GV / Q35_GK);
  float* const Sp = A.S + (size_t)h * Q35_GD * Q35_GD;
  float s[32];
  #pragma unroll
  for (int r = 0; r < 32; ++r) s[r] = __ldcg(Sp + (size_t)(w * 32 + r) * Q35_GD + j);
  pdl_wait(); pdl_trigger();
  if (A.n_commit > 0) {                                                    
    const int gt = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + tid;
    if (gt < Q35_CONV) { const int n = A.n_commit; uint16_t o[3], nw[3];
      #pragma unroll
      for (int k = 0; k < 3; ++k) o[k] = A.hist[(size_t)k * Q35_CONV + gt];
      #pragma unroll
      for (int k = 0; k < 3; ++k) { const int r = n - 3 + k; nw[k] = r >= 0 ? A.mixed[(size_t)r * Q35_CONV + gt] : o[3 + r]; }
      #pragma unroll
      for (int k = 0; k < 3; ++k) A.hist[(size_t)k * Q35_CONV + gt] = nw[k]; }
  }
  
  
  auto stage = [&](int t0, int b) {
    const int nr = min(Q35_ST_TS, A.M - t0);
    for (int c = tid; c < nr * 32; c += 128) {                            
      const int rr = c >> 5, ch = c & 31, qw = ch >> 3, off = (ch & 7) * 4;
      const size_t src = ((size_t)(t0 + rr) * Q35_GK + kh) * Q35_GD + qw * 32 + off;
      const unsigned dk = (unsigned)__cvta_generic_to_shared(&sk[b][rr][qw * 36 + off]), dq = (unsigned)__cvta_generic_to_shared(&sq[b][rr][qw * 36 + off]);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(A.k + src) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dq), "l"(A.q + src) : "memory");
    }
    for (int c = tid; c < nr * 8; c += 128) {                             
      const int rr = c >> 3, ch = c & 7;
      const unsigned dv = (unsigned)__cvta_generic_to_shared(&sv[b][rr][ch * 4]);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dv), "l"(A.v + ((size_t)(t0 + rr) * Q35_GV + h) * Q35_GD + blockIdx.y * 32 + ch * 4) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (tid < nr) { seg[b][tid] = q35_exp(A.g[(size_t)(t0 + tid) * Q35_GV + h]); sbt[b][tid] = A.beta[(size_t)(t0 + tid) * Q35_GV + h]; }
  };
  const int base = lane & ~3;
  if (A.M > 0) stage(0, 0);
  for (int t0 = 0, b = 0; t0 < A.M; t0 += Q35_ST_TS, b ^= 1) {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();                                                       
    if (t0 + Q35_ST_TS < A.M) stage(t0 + Q35_ST_TS, b ^ 1);
    const int nr = min(Q35_ST_TS, A.M - t0);
    for (int u = 0; u < nr; ++u) {
      const int t = t0 + u;
      const float eg = seg[b][u], bt = sbt[b][u], vj = sv[b][u][jj];
      const float* const kr = &sk[b][u][w * 36]; const float* const qr = &sq[b][u][w * 36];
      float kv = 0.f;
      #pragma unroll
      for (int r = 0; r < 32; ++r) { s[r] = __fmul_rn(s[r], eg); kv = __fadd_rn(kv, __fmul_rn(s[r], kr[r])); }
      const float k0 = __shfl_sync(0xffffffffu, kv, base), k1 = __shfl_sync(0xffffffffu, kv, base + 1);
      const float k2 = __shfl_sync(0xffffffffu, kv, base + 2), k3 = __shfl_sync(0xffffffffu, kv, base + 3);
      const float d = __fmul_rn(__fsub_rn(vj, __fadd_rn(__fadd_rn(__fadd_rn(k0, k1), k2), k3)), bt);
      float o = 0.f;
      #pragma unroll
      for (int r = 0; r < 32; ++r) { s[r] = __fadd_rn(s[r], __fmul_rn(kr[r], d)); o = __fadd_rn(o, __fmul_rn(s[r], qr[r])); }
      const float o0 = __shfl_sync(0xffffffffu, o, base), o1 = __shfl_sync(0xffffffffu, o, base + 1);
      const float o2 = __shfl_sync(0xffffffffu, o, base + 2), o3 = __shfl_sync(0xffffffffu, o, base + 3);
      if (w == 0) A.core[((size_t)t * Q35_GV + h) * Q35_GD + j] = rbf(__fadd_rn(__fadd_rn(__fadd_rn(o0, o1), o2), o3));
      if (t + 1 == A.n_commit) {
        #pragma unroll
        for (int r = 0; r < 32; ++r) Sp[(size_t)(w * 32 + r) * Q35_GD + j] = s[r];
      }
    }
  }
}





constexpr int Q35_NB_TS = 24, Q35_NB_KQ = 4 * 36;
constexpr size_t Q35_NB_SMEM = sizeof(float) * (2 * Q35_NB_TS * Q35_NB_KQ * 2 + 2 * Q35_NB_TS * 128 + 4 * Q35_NB_TS);
__global__ void __launch_bounds__(512, 1) q35_state_nb_kernel(Q35StateArgs A) {
  constexpr int TS = Q35_NB_TS, KQ = Q35_NB_KQ;
  extern __shared__ __align__(16) float sm_nb[];
  float* const sk = sm_nb;                                                 
  float* const sq = sk + 2 * TS * KQ;                                      
  float* const sv = sq + 2 * TS * KQ;                                      
  float* const seg = sv + 2 * TS * 128;                                    
  float* const sbt = seg + 2 * TS;                                         
  const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid & 3, j = tid >> 2, kh = h / (Q35_GV / Q35_GK);
  float* const Sp = A.S + (size_t)h * Q35_GD * Q35_GD;
  float s[32];
  #pragma unroll
  for (int r = 0; r < 32; ++r) s[r] = __ldcg(Sp + (size_t)(w * 32 + r) * Q35_GD + j);
  if (A.n_commit > 0 && A.hist) {                                          
    const int gt = blockIdx.x * blockDim.x + tid;
    if (gt < Q35_CONV) { const int n = A.n_commit; uint16_t o[3], nw[3];
      #pragma unroll
      for (int k = 0; k < 3; ++k) o[k] = A.hist[(size_t)k * Q35_CONV + gt];
      #pragma unroll
      for (int k = 0; k < 3; ++k) { const int r = n - 3 + k; nw[k] = r >= 0 ? A.mixed[(size_t)r * Q35_CONV + gt] : o[3 + r]; }
      #pragma unroll
      for (int k = 0; k < 3; ++k) A.hist[(size_t)k * Q35_CONV + gt] = nw[k]; }
  }
  auto stage = [&](int t0, int b) {
    const int nr = min(TS, A.M - t0);
    for (int c = tid; c < nr * 32; c += 512) {                            
      const int rr = c >> 5, ch = c & 31, qw = ch >> 3, off = (ch & 7) * 4;
      const size_t src = ((size_t)(t0 + rr) * Q35_GK + kh) * Q35_GD + qw * 32 + off;
      const unsigned dk = (unsigned)__cvta_generic_to_shared(sk + ((size_t)b * TS + rr) * KQ + qw * 36 + off);
      const unsigned dq = (unsigned)__cvta_generic_to_shared(sq + ((size_t)b * TS + rr) * KQ + qw * 36 + off);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(A.k + src) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dq), "l"(A.q + src) : "memory");
    }
    for (int c = tid; c < nr * 32; c += 512) {                            
      const int rr = c >> 5, ch = c & 31;
      const unsigned dv = (unsigned)__cvta_generic_to_shared(sv + ((size_t)b * TS + rr) * 128 + ch * 4);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dv), "l"(A.v + ((size_t)(t0 + rr) * Q35_GV + h) * Q35_GD + ch * 4) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (tid < nr) { seg[b * TS + tid] = q35_exp(A.g[(size_t)(t0 + tid) * Q35_GV + h]); sbt[b * TS + tid] = A.beta[(size_t)(t0 + tid) * Q35_GV + h]; }
  };
  const int base = lane & ~3;
  if (A.M > 0) stage(0, 0);
  for (int t0 = 0, b = 0; t0 < A.M; t0 += TS, b ^= 1) {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    if (t0 + TS < A.M) stage(t0 + TS, b ^ 1);
    const int nr = min(TS, A.M - t0);
    for (int u = 0; u < nr; ++u) {
      const int t = t0 + u;
      const float eg = seg[b * TS + u], bt = sbt[b * TS + u], vj = sv[((size_t)b * TS + u) * 128 + j];
      const float* const kr = sk + ((size_t)b * TS + u) * KQ + w * 36; const float* const qr = sq + ((size_t)b * TS + u) * KQ + w * 36;
      float kv = 0.f;
      #pragma unroll
      for (int r = 0; r < 32; ++r) { s[r] = __fmul_rn(s[r], eg); kv = __fadd_rn(kv, __fmul_rn(s[r], kr[r])); }
      const float k0 = __shfl_sync(0xffffffffu, kv, base), k1 = __shfl_sync(0xffffffffu, kv, base + 1);
      const float k2 = __shfl_sync(0xffffffffu, kv, base + 2), k3 = __shfl_sync(0xffffffffu, kv, base + 3);
      const float d = __fmul_rn(__fsub_rn(vj, __fadd_rn(__fadd_rn(__fadd_rn(k0, k1), k2), k3)), bt);
      float o = 0.f;
      #pragma unroll
      for (int r = 0; r < 32; ++r) { s[r] = __fadd_rn(s[r], __fmul_rn(kr[r], d)); o = __fadd_rn(o, __fmul_rn(s[r], qr[r])); }
      const float o0 = __shfl_sync(0xffffffffu, o, base), o1 = __shfl_sync(0xffffffffu, o, base + 1);
      const float o2 = __shfl_sync(0xffffffffu, o, base + 2), o3 = __shfl_sync(0xffffffffu, o, base + 3);
      if (w == 0) A.core[((size_t)t * Q35_GV + h) * Q35_GD + j] = rbf(__fadd_rn(__fadd_rn(__fadd_rn(o0, o1), o2), o3));
      if (t + 1 == A.n_commit) {
        #pragma unroll
        for (int r = 0; r < 32; ++r) Sp[(size_t)(w * 32 + r) * Q35_GD + j] = s[r];
      }
    }
  }
}
void launch_q35_state_nb(const Q35StateArgs& A, cudaStream_t st) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(q35_state_nb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)Q35_NB_SMEM); attr = true; }
  q35_state_nb_kernel<<<Q35_GV, 512, Q35_NB_SMEM, st>>>(A);
}
static bool q35_pf_launch(int M) { return M > 64 && q35_pfalt() != 0; }




__device__ __forceinline__ uint16_t q35_f2kv16(float x, int fp16) {
  return fp16 ? __half_as_ushort(__float2half_rn(x)) : f2bf(x);
}
__global__ void __launch_bounds__(256) q35_attn_post_pf_kernel(Q35AttnPostArgs A) {
  pdl_wait(); pdl_trigger();
  constexpr int NB = Q35_NH + 4;
  const int lane = threadIdx.x & 31, task = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int ntt = (A.M + Q35_PF_TPW - 1) / Q35_PF_TPW; if (task >= NB * ntt) return;
  const int blk = task % NB, t0 = (task / NB) * Q35_PF_TPW;
  const float* part; const float* tau; const int8_t* sgp; int N, nb0;
  if (blk < Q35_NH) { part = A.part_q; tau = A.tau_q; N = 2 * Q35_NH * Q35_HD; nb0 = blk * 512; sgp = A.sg_q + blk * 512; }
  else if (blk < Q35_NH + 2) { part = A.part_k; tau = A.tau_k; N = Q35_KVH * Q35_HD; nb0 = (blk - Q35_NH) * 512; sgp = A.sg_k + (blk - Q35_NH) * 512; }
  else { part = A.part_v; tau = A.tau_v; N = Q35_KVH * Q35_HD; nb0 = (blk - Q35_NH - 2) * 512; sgp = A.sg_v + (blk - Q35_NH - 2) * 512; }
  const bool isv = blk >= Q35_NH + 2, isq = blk < Q35_NH;
  float ta[16], sg[16], w1[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { const int p = lane + 32 * i, d = p & (Q35_HD - 1);
    ta[i] = tau[nb0 + p]; sg[i] = (float)sgp[p]; w1[i] = isv ? 0.f : __fadd_rn(1.f, bf2f((isq ? A.qn_w : A.kn_w)[d])); }
  for (int tk = t0; tk < min(A.M, t0 + Q35_PF_TPW); ++tk) {
    const int pos = *A.d_pos + tk;
    const float* pr = part + (size_t)tk * N + nb0;
    float y[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __fmul_rn(__ldcg(pr + lane + 32 * i), ta[i]);
    q35_wfwht512(y);
    #pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = rbf(__fmul_rn(__fmul_rn(y[i], Q35_K512), sg[i]));
    auto kv8 = [&](const float (&x)[16], int8_t* q8, float* qs, int kvb, uint16_t* kv16, int stg) {
      if (!q8) {
        #pragma unroll
        for(int i=0;i<16;++i) kv16[((size_t)(kvb+(i>>3))*A.max_ctx+pos)*Q35_HD+lane+32*(i&7)]=q35_f2kv16(x[i],A.kv_fp16);
        return;
      }   
      #pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        unsigned m = 0u;
        #pragma unroll
        for (int i = 8 * hf; i < 8 * hf + 8; ++i) m = max(m, __float_as_uint(fabsf(x[i])));
        for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float sc = __fdiv_rn(__uint_as_float(m), 127.f);
        const size_t row = q35_kvrow(stg, A.max_ctx, kvb + hf, pos, tk);
        #pragma unroll
        for (int i = 8 * hf; i < 8 * hf + 8; ++i) { int q = sc > 0.f ? __float2int_rn(__fdiv_rn(x[i], sc)) : 0; q = q < -127 ? -127 : (q > 127 ? 127 : q);
                                                    q8[row * Q35_HD + lane + 32 * (i & 7)] = (int8_t)q; }
        if (lane == 0) qs[row] = sc;
      }
    };
    if (isv) {
#if YSKN_KVX_ROT
      kvx_rotate_warp16(y);
#endif
      kv8(y, A.v8, A.vs, (blk - Q35_NH - 2) * 2, A.vc, A.vstg); continue; }
    if (isq) {
      #pragma unroll
      for (int i = 8; i < 16; ++i) A.gate16[((size_t)tk * Q35_NH + blk) * Q35_HD + lane + 32 * (i - 8)] = f2bf(y[i]);
    }
    float t[16];                                                           
    #pragma unroll
    for (int i = 0; i < 16; ++i) { t[i] = __fmul_rn(y[i], y[i]); for (int o = 16; o; o >>= 1) t[i] = __fadd_rn(t[i], __shfl_xor_sync(0xffffffffu, t[i], o)); }
    float r[2];
    #pragma unroll
    for (int hf = 0; hf < 2; ++hf) { float tot = 0.f;
      #pragma unroll
      for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, t[8 * hf + w]);
      r[hf] = q35_rsqrt(__fadd_rn(__fdiv_rn(tot, (float)Q35_HD), A.eps)); }
    float xn[16], out[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) xn[i] = rbf(__fmul_rn(__fmul_rn(y[i], r[i >> 3]), w1[i]));
    const float2 cs = A.rope[(size_t)pos * (Q35_RD / 2) + lane];           
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      if ((i & 7) == 0) out[i] = rbf(__fadd_rn(rbf(__fmul_rn(xn[i], cs.x)), rbf(__fmul_rn(-xn[i + 1], cs.y))));
      else if ((i & 7) == 1) out[i] = rbf(__fadd_rn(rbf(__fmul_rn(xn[i], cs.x)), rbf(__fmul_rn(xn[i - 1], cs.y))));
      else out[i] = xn[i];
    }
#if YSKN_KVX_ROT
    kvx_rotate_warp16(out);
#endif
    if (isq) {
      #pragma unroll
      for (int i = 0; i < 8; ++i) A.q16[((size_t)tk * Q35_NH + blk) * Q35_HD + lane + 32 * i] = f2bf(out[i]);
    } else kv8(out, A.k8, A.ks, (blk - Q35_NH) * 2, A.kc, A.kstg);
  }
}



constexpr int Q35_CONV_TPW = 16;
__global__ void __launch_bounds__(256) q35_conv_pf_kernel(Q35ConvArgs A) {
  pdl_wait(); pdl_trigger();
  constexpr int NS = 2 * Q35_GK + Q35_GV;
  const int lane = threadIdx.x & 31, task = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int ntt = (A.M + Q35_CONV_TPW - 1) / Q35_CONV_TPW; if (task >= NS * ntt) return;
  const int slot = task % NS, t0 = (task / NS) * Q35_CONV_TPW;
  const int kind = slot < Q35_GK ? 0 : (slot < 2 * Q35_GK ? 1 : 2), head = kind == 0 ? slot : (kind == 1 ? slot - Q35_GK : slot - 2 * Q35_GK);
  const int cb = (kind == 0 ? 0 : (kind == 1 ? Q35_GK * Q35_GD : 2 * Q35_GK * Q35_GD)) + head * Q35_GD;
  float w0[4], w1[4], w2[4], w3[4], win[3][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) { const int c = cb + lane + 32 * i;
    w0[i] = bf2f(A.conv_w[c * 4]); w1[i] = bf2f(A.conv_w[c * 4 + 1]); w2[i] = bf2f(A.conv_w[c * 4 + 2]); w3[i] = bf2f(A.conv_w[c * 4 + 3]);
    #pragma unroll
    for (int k = 0; k < 3; ++k) { const int r = t0 - 3 + k;                
      win[k][i] = r >= 0 ? bf2f(A.mixed[(size_t)r * Q35_CONV + c]) : bf2f(__ldcg(A.hist + (size_t)(r + 3) * Q35_CONV + c)); } }
  float c_dt = 0.f, c_al = 0.f;
  if (kind == 2) { c_dt = bf2f(A.dt_bias[head]); c_al = bf2f(A.A_log[head]); }
  for (int t = t0; t < min(A.M, t0 + Q35_CONV_TPW); ++t) {
    float act[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { const int c = cb + lane + 32 * i;
      const float x = bf2f(A.mixed[(size_t)t * Q35_CONV + c]);
      const float acc = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(w0[i], win[0][i]), __fmul_rn(w1[i], win[1][i])), __fmul_rn(w2[i], win[2][i])), __fmul_rn(w3[i], x));
      act[i] = rbf(q35_silu(rbf(acc)));
      win[0][i] = win[1][i]; win[1][i] = win[2][i]; win[2][i] = x; }
    if (kind == 2) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) A.v[((size_t)t * Q35_GV + head) * Q35_GD + lane + 32 * i] = act[i];
      if (lane == 0) {
        const float b = A.ab[(size_t)t * 96 + head], a = A.ab[(size_t)t * 96 + 48 + head];
        A.beta[(size_t)t * Q35_GV + head] = rbf(q35_sigmoid(b));
        const float sp_in = __fadd_rn(a, c_dt);
        A.g[(size_t)t * Q35_GV + head] = __fmul_rn(-q35_exp(c_al), q35_softplus(sp_in));
      }
    } else {
      float ss[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) { ss[i] = __fmul_rn(act[i], act[i]); for (int o = 16; o; o >>= 1) ss[i] = __fadd_rn(ss[i], __shfl_xor_sync(0xffffffffu, ss[i], o)); }
      const float tot = __fadd_rn(__fadd_rn(__fadd_rn(ss[0], ss[1]), ss[2]), ss[3]);
      const float rn = q35_rsqrt(__fadd_rn(tot, 1e-6f));
      #pragma unroll
      for (int i = 0; i < 4; ++i) { const float y = __fmul_rn(act[i], rn);
        if (kind == 0) A.q[((size_t)t * Q35_GK + head) * Q35_GD + lane + 32 * i] = __fmul_rn(y, 0.08838834764831845f);
        else A.k[((size_t)t * Q35_GK + head) * Q35_GD + lane + 32 * i] = y; }
    }
  }
}




__global__ void __launch_bounds__(128) q35_ab_kernel(Q35AbArgs A) {
  __shared__ float s_red[32][4];
  const int r = blockIdx.x, tid = threadIdx.x, t0 = blockIdx.y * 32, nt = min(32, A.M - t0);
  const uint16_t* w = r < 48 ? A.wb + (size_t)r * Q35_H : A.wa + (size_t)(r - 48) * Q35_H;
  uint4 wv[5];
  #pragma unroll
  for (int j = 0; j < 5; ++j) wv[j] = __ldg(reinterpret_cast<const uint4*>(w) + tid + 128 * j);
  pdl_wait(); pdl_trigger();
  __shared__ float s_rq[32];                                               
  if (A.S && tid < nt) s_rq[tid] = q35_xsm_r(A.S + (size_t)(t0 + tid) * Q35_SST, A.eps);
  for (int u0 = 0; u0 < nt; u0 += 4) {                                   
    uint4 xv[4][5];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (u0 + q >= nt) break;                                             
      const int tk = t0 + u0 + q;
      #pragma unroll
      for (int j = 0; j < 5; ++j) xv[q][j] = q35_ab_x(A, tk, tid + 128 * j);
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (u0 + q >= nt) break;
      float acc = 0.f;
      #pragma unroll
      for (int j = 0; j < 5; ++j) {
        const unsigned wa[4] = {wv[j].x, wv[j].y, wv[j].z, wv[j].w}, xa[4] = {xv[q][j].x, xv[q][j].y, xv[q][j].z, xv[q][j].w};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wa[e] << 16), __uint_as_float(xa[e] << 16)));
          acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wa[e] & 0xffff0000u), __uint_as_float(xa[e] & 0xffff0000u)));
        }
      }
      for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
      if ((tid & 31) == 0) s_red[u0 + q][tid >> 5] = acc;
    }
  }
  __syncthreads();
  if (tid < nt) { const float sum = __fadd_rn(__fadd_rn(__fadd_rn(s_red[tid][0], s_red[tid][1]), s_red[tid][2]), s_red[tid][3]);
    A.ab[(size_t)(t0 + tid) * 96 + r] = rbf(A.S ? __fmul_rn(sum, s_rq[tid]) : sum); }
}
void launch_q35_ab(const Q35AbArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M)) { launch_k(q35_ab_pf_kernel<6>, dim3(96 / 6, (A.M + 15) / 16), dim3(128), 0, st, A); return; } launch_k(q35_ab_kernel, dim3(96, (A.M + 31) / 32), dim3(128), 0, st, A); }

__device__ __forceinline__ void q35_ab_row1(const Q35AbArgs& A, int r) {
  
  
  
  __shared__ float s_red1[32][4], s_r1[32];
  const int g = threadIdx.x >> 7, tid = threadIdx.x & 127, nt = A.M;
  const uint16_t* w = r < 48 ? A.wb + (size_t)r * Q35_H : A.wa + (size_t)(r - 48) * Q35_H;
  uint4 wv[5];
  #pragma unroll
  for (int j = 0; j < 5; ++j) wv[j] = __ldg(reinterpret_cast<const uint4*>(w) + tid + 128 * j);
  if (A.S && threadIdx.x < nt) s_r1[threadIdx.x] = q35_xsm_r(A.S + (size_t)threadIdx.x * Q35_SST, A.eps);
  for (int t = g; t < nt; t += 4) {
    uint4 xv[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) xv[j] = q35_ab_x(A, t, tid + 128 * j);
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < 5; ++j) {
      const unsigned wa[4] = {wv[j].x, wv[j].y, wv[j].z, wv[j].w}, xa[4] = {xv[j].x, xv[j].y, xv[j].z, xv[j].w};
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wa[e] << 16), __uint_as_float(xa[e] << 16)));
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wa[e] & 0xffff0000u), __uint_as_float(xa[e] & 0xffff0000u)));
      }
    }
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red1[t][tid >> 5] = acc;
  }
  __syncthreads();
  if (threadIdx.x < nt) { const int t = threadIdx.x;
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(s_red1[t][0], s_red1[t][1]), s_red1[t][2]), s_red1[t][3]);
    A.ab[(size_t)t * 96 + r] = rbf(A.S ? __fmul_rn(sum, s_r1[t]) : sum); }
}

__global__ void __launch_bounds__(512) q35_mix_kernel(Q35MixArgs A) {
  __shared__ float buf[512];
  
  
  constexpr int NB = Q35_CONV / 512 + Q35_Z / 512;
  const int tid = threadIdx.x, nab = A.with_ab ? 96 : 0;                  
  if ((int)blockIdx.x < nab) {                                            
    q35_ab_row1(A.ab, blockIdx.x);
    return;
  }
  const int blk = (blockIdx.x - nab) % NB, tok = (blockIdx.x - nab) / NB;
  const bool isq = blk < Q35_CONV / 512;
  const int n = (isq ? blk : blk - Q35_CONV / 512) * 512 + tid;
  const float c_tau = isq ? A.tau_qkv[n] : A.tau_z[n]; const int8_t c_sg = isq ? A.sg_qkv[n] : A.sg_z[n];
  pdl_wait(); pdl_trigger();
  if (tid == 0 && A.pf.p[0]) q35_l2pf(A.pf, blockIdx.x - nab, gridDim.x - nab);   
  long long acc = 0; float2 nrs = make_float2(0.f, 0.f);                  
  if (A.xsm.us) { acc = q35_xsm_ld(isq ? A.part_qkv : A.part_z, A.xsm.mode, isq ? A.nw_qkv : A.nw_z, A.M, isq ? Q35_CONV : Q35_Z, tok, n);
                  if (A.xsm.mode != 3) nrs = q35_xsm_row(A.xsm.us, A.xsm.S, A.xsm.eps, tok); }
  if (isq) {
    buf[tid] = A.xsm.us ? (A.xsm.mode == 3 ? __fmul_rn(__uint_as_float((unsigned)acc), c_tau) : __fmul_rn(__fmul_rn(q35_xsm_fin(acc, A.xsm.mode, nrs.x), nrs.y), c_tau))
                        : __fmul_rn(c_tau, q35_combine(A.part_qkv, A.nw_qkv, A.M, Q35_CONV, tok, n)); __syncthreads();
    q35_inv_h512s(buf, (float)c_sg);
    A.mixed[(size_t)tok * Q35_CONV + n] = f2bf(buf[tid]);
  } else {
    buf[tid] = A.xsm.us ? (A.xsm.mode == 3 ? __fmul_rn(__uint_as_float((unsigned)acc), c_tau) : __fmul_rn(__fmul_rn(q35_xsm_fin(acc, A.xsm.mode, nrs.x), nrs.y), c_tau))
                        : __fmul_rn(c_tau, q35_combine(A.part_z, A.nw_z, A.M, Q35_Z, tok, n)); __syncthreads();
    q35_inv_h512s(buf, (float)c_sg);
    A.z[(size_t)tok * Q35_Z + n] = rbf(buf[tid]);
  }
}
void launch_q35_mix(const Q35MixArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M) && A.xsm.us && A.xsm.mode == 3 && !A.with_ab && !A.pf.p[0]) {   
    const int tasks = (Q35_CONV / 512 + Q35_Z / 512) * ((A.M + Q35_PF_TPW - 1) / Q35_PF_TPW); launch_k(q35_mix_pf_kernel, dim3((tasks + 7) / 8), dim3(256), 0, st, A); return; }
  if (A.with_ab && A.M > 32) throw std::runtime_error("launch_q35_mix: fused a / b gates need M <= 32");
  launch_k(q35_mix_kernel, dim3((Q35_CONV / 512 + Q35_Z / 512) * A.M + (A.with_ab ? 96 : 0)), dim3(512), 0, st, A); }


__global__ void __launch_bounds__(128) q35_conv_kernel(Q35ConvArgs A) {
  __shared__ float s_red[4];
  const int slot = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kind = slot < Q35_GK ? 0 : (slot < 2 * Q35_GK ? 1 : 2);        
  const int head = kind == 0 ? slot : (kind == 1 ? slot - Q35_GK : slot - 2 * Q35_GK);
  const int c = (kind == 0 ? 0 : (kind == 1 ? Q35_GK * Q35_GD : 2 * Q35_GK * Q35_GD)) + head * Q35_GD + tid;
  const uint16_t cw0 = A.conv_w[c * 4], cw1 = A.conv_w[c * 4 + 1], cw2 = A.conv_w[c * 4 + 2], cw3 = A.conv_w[c * 4 + 3];
  uint16_t hv[3] = {0, 0, 0};                                              
  #pragma unroll                                                           
  for (int k = 0; k < 3; ++k) if (t - 3 + k < 0) hv[k] = __ldcg(A.hist + (size_t)(t + k) * Q35_CONV + c);
  uint16_t c_dt = 0, c_al = 0;
  if (kind == 2 && tid == 0) { c_dt = A.dt_bias[head]; c_al = A.A_log[head]; }
  pdl_wait(); pdl_trigger();
  const float w0 = bf2f(cw0), w1 = bf2f(cw1), w2 = bf2f(cw2), w3 = bf2f(cw3);
  auto xin = [&](int r) { return r >= 0 ? bf2f(A.mixed[(size_t)r * Q35_CONV + c]) : bf2f(hv[r - t + 3]); };   
  const float s1 = xin(t - 3), s2 = xin(t - 2), s3 = xin(t - 1), x = xin(t);
  const float acc = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(w0, s1), __fmul_rn(w1, s2)), __fmul_rn(w2, s3)), __fmul_rn(w3, x));
  const float act = rbf(q35_silu(rbf(acc)));
  if (kind == 2) {
    A.v[((size_t)t * Q35_GV + head) * Q35_GD + tid] = act;
    if (tid == 0) {
      const float b = A.ab[(size_t)t * 96 + head], a = A.ab[(size_t)t * 96 + 48 + head];
      A.beta[(size_t)t * Q35_GV + head] = rbf(q35_sigmoid(b));
      const float sp_in = __fadd_rn(a, bf2f(c_dt));
      const float sp = q35_softplus(sp_in);                                                
      A.g[(size_t)t * Q35_GV + head] = __fmul_rn(-q35_exp(bf2f(c_al)), sp);
    }
  } else {
    float ss = __fmul_rn(act, act);
    for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    const float tot = __fadd_rn(__fadd_rn(__fadd_rn(s_red[0], s_red[1]), s_red[2]), s_red[3]);
    const float y = __fmul_rn(act, q35_rsqrt(__fadd_rn(tot, 1e-6f)));
    if (kind == 0) A.q[((size_t)t * Q35_GK + head) * Q35_GD + tid] = __fmul_rn(y, 0.08838834764831845f);
    else A.k[((size_t)t * Q35_GK + head) * Q35_GD + tid] = y;
  }
}
void launch_q35_conv(const Q35ConvArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M)) { const int tasks = (2 * Q35_GK + Q35_GV) * ((A.M + Q35_CONV_TPW - 1) / Q35_CONV_TPW);   
                            launch_k(q35_conv_pf_kernel, dim3((tasks + 7) / 8), dim3(256), 0, st, A); return; } launch_k(q35_conv_kernel, dim3(2 * Q35_GK + Q35_GV, A.M), dim3(128), 0, st, A); }



__global__ void __launch_bounds__(128) q35_state_kernel(Q35StateArgs A) {
  
  
  
  __shared__ float sk[2][Q35_GD], sq[2][Q35_GD], skv[2][4][32], so[2][4][32];
  const int h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, j = blockIdx.y * 32 + lane, kh = h / (Q35_GV / Q35_GK);
  float* const Sp = A.S + (size_t)h * Q35_GD * Q35_GD;
  float s[32];                                                             
  #pragma unroll                                                           
  for (int r = 0; r < 32; ++r) s[r] = __ldcg(Sp + (size_t)(warp * 32 + r) * Q35_GD + j);
  pdl_wait(); pdl_trigger();
  if (A.n_commit > 0) {                                                    
    const int gt = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + tid; 
    if (gt < Q35_CONV) { const int n = A.n_commit; uint16_t o[3], nw[3];
      #pragma unroll
      for (int k = 0; k < 3; ++k) o[k] = A.hist[(size_t)k * Q35_CONV + gt];
      #pragma unroll
      for (int k = 0; k < 3; ++k) { const int r = n - 3 + k; nw[k] = r >= 0 ? A.mixed[(size_t)r * Q35_CONV + gt] : o[3 + r]; }
      #pragma unroll
      for (int k = 0; k < 3; ++k) A.hist[(size_t)k * Q35_CONV + gt] = nw[k]; }
  }
  auto ld = [&](int t, float& k, float& q, float& g, float& b, float& v) {
    k = A.k[((size_t)t * Q35_GK + kh) * Q35_GD + tid]; q = A.q[((size_t)t * Q35_GK + kh) * Q35_GD + tid];
    g = A.g[(size_t)t * Q35_GV + h]; b = A.beta[(size_t)t * Q35_GV + h]; v = A.v[((size_t)t * Q35_GV + h) * Q35_GD + j]; };
  float nk = 0.f, nq = 0.f, ng = 0.f, nb = 0.f, nv = 0.f;
  if (A.M > 0) ld(0, nk, nq, ng, nb, nv);
  for (int t = 0; t < A.M; ++t) {
    const int pb = t & 1;
    sk[pb][tid] = nk; sq[pb][tid] = nq;
    const float eg = q35_exp(ng), bt = nb, vj = nv;
    if (t + 1 < A.M) ld(t + 1, nk, nq, ng, nb, nv);
    __syncthreads();                                                       
    if (t > 0 && warp == 0) { const int qb = pb ^ 1;
      A.core[((size_t)(t - 1) * Q35_GV + h) * Q35_GD + j] = rbf(__fadd_rn(__fadd_rn(__fadd_rn(so[qb][0][lane], so[qb][1][lane]), so[qb][2][lane]), so[qb][3][lane])); }
    float kv = 0.f;
    #pragma unroll
    for (int r = 0; r < 32; ++r) { s[r] = __fmul_rn(s[r], eg); kv = __fadd_rn(kv, __fmul_rn(s[r], sk[pb][warp * 32 + r])); }
    skv[pb][warp][lane] = kv;
    __syncthreads();
    const float d = __fmul_rn(__fsub_rn(vj, __fadd_rn(__fadd_rn(__fadd_rn(skv[pb][0][lane], skv[pb][1][lane]), skv[pb][2][lane]), skv[pb][3][lane])), bt);
    float o = 0.f;
    #pragma unroll
    for (int r = 0; r < 32; ++r) { s[r] = __fadd_rn(s[r], __fmul_rn(sk[pb][warp * 32 + r], d)); o = __fadd_rn(o, __fmul_rn(s[r], sq[pb][warp * 32 + r])); }
    so[pb][warp][lane] = o;
    if (t + 1 == A.n_commit) {
      #pragma unroll
      for (int r = 0; r < 32; ++r) Sp[(size_t)(warp * 32 + r) * Q35_GD + j] = s[r];
    }
  }
  __syncthreads();
  if (A.M > 0 && warp == 0) { const int qb = (A.M - 1) & 1;
    A.core[((size_t)(A.M - 1) * Q35_GV + h) * Q35_GD + j] = rbf(__fadd_rn(__fadd_rn(__fadd_rn(so[qb][0][lane], so[qb][1][lane]), so[qb][2][lane]), so[qb][3][lane])); }
}
void launch_q35_state(const Q35StateArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M)) { launch_k(q35_state_pf_kernel, dim3(Q35_GV, Q35_GD / 32), dim3(128), 0, st, A); return; } launch_k(q35_state_kernel, dim3(Q35_GV, Q35_GD / 32), dim3(128), 0, st, A); }


__global__ void __launch_bounds__(512) q35_gdn_out_kernel(Q35GdnOutArgs A) {
  __shared__ float buf[512], s_red[16]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, e = blk * 512 + tid, warp = tid >> 5, lane = tid & 31, j = e & (Q35_GD - 1);
  const uint16_t c_gw = A.gw[j]; const int8_t c_si = A.sg_in[e];
  pdl_wait(); pdl_trigger();
  const float c = A.core[(size_t)tok * Q35_Z + e];
  float ss = __fmul_rn(c, c);
  for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
  if (lane == 0) s_red[warp] = ss;
  if (tid == 0) s_amax = 0u;
  __syncthreads();
  const int hw = (tid >> 7) * 4;                                           
  const float tot = __fadd_rn(__fadd_rn(__fadd_rn(s_red[hw], s_red[hw + 1]), s_red[hw + 2]), s_red[hw + 3]);
  const float r = q35_rsqrt(__fadd_rn(__fdiv_rn(tot, (float)Q35_GD), A.eps));
  const float n2 = rbf(__fmul_rn(bf2f(c_gw), rbf(__fmul_rn(c, r))));
  const float out = rbf(__fmul_rn(n2, q35_silu(A.z[(size_t)tok * Q35_Z + e])));
  buf[tid] = __fmul_rn(out, (float)c_si);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, A.lo ? A.qx + (size_t)blk * 4096 : A.qx + (size_t)tok * Q35_Z + blk * 512, A.alpha + (size_t)tok * (Q35_Z / 512) + blk, A.lo ? tok : -1);
}
void launch_q35_gdn_out(const Q35GdnOutArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M) && !A.lo) { const int tasks = (Q35_Z / 512) * ((A.M + Q35_PF_TPW - 1) / Q35_PF_TPW);   
                                     launch_k(q35_gdn_out_pf_kernel, dim3((tasks + 7) / 8), dim3(256), 0, st, A); return; } launch_k(q35_gdn_out_kernel, dim3(Q35_Z / 512, A.M), dim3(512), 0, st, A); }





__device__ __forceinline__ void q35_kv8_store(float x, int8_t* q8, float* qs, size_t row, int d) {
#if YSKN_KVX_MODE != 0
  kvx_store(x,q8,qs,row,d); return;
#endif
  __shared__ unsigned s_mx[16];
  const int tid = threadIdx.x;
  unsigned m = __float_as_uint(fabsf(x));
  for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((tid & 31) == 0) s_mx[tid >> 5] = m;
  __syncthreads();
  const int w0 = (tid >> 8) * 8; unsigned mm = 0u;
  #pragma unroll
  for (int w = 0; w < 8; ++w) mm = max(mm, s_mx[w0 + w]);
  const float sc = __fdiv_rn(__uint_as_float(mm), 127.f);
  int q = sc > 0.f ? __float2int_rn(__fdiv_rn(x, sc)) : 0; q = q < -127 ? -127 : (q > 127 ? 127 : q);
  q8[row * Q35_HD + d] = (int8_t)q;
  if (d == 0) qs[row] = sc;
}

__global__ void __launch_bounds__(512) q35_attn_post_kernel(Q35AttnPostArgs A) {
  __shared__ float buf[512], s_red[16];
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const float* part; const float* tau; const int8_t* sg; int nw, N, n;
  if (blk < Q35_NH) { part = A.part_q; tau = A.tau_q; nw = A.nw_q; N = 2 * Q35_NH * Q35_HD; n = blk * 512 + tid; sg = A.sg_q + blk * 512; }
  else if (blk < Q35_NH + 2) { part = A.part_k; tau = A.tau_k; nw = A.nw_k; N = Q35_KVH * Q35_HD; n = (blk - Q35_NH) * 512 + tid; sg = A.sg_k + (blk - Q35_NH) * 512; }
  else { part = A.part_v; tau = A.tau_v; nw = A.nw_v; N = Q35_KVH * Q35_HD; n = (blk - Q35_NH - 2) * 512 + tid; sg = A.sg_v + (blk - Q35_NH - 2) * 512; }
  const int d = tid & (Q35_HD - 1);
  const float c_tau = tau[n]; const int8_t c_sg = sg[tid]; const uint16_t c_nw = blk < Q35_NH + 2 ? (blk < Q35_NH ? A.qn_w : A.kn_w)[d] : 0;
  pdl_wait(); pdl_trigger();
  if (tid == 0 && A.pf.p[0]) q35_l2pf(A.pf, blk + tok * gridDim.x, gridDim.x * gridDim.y);   
  const int pos = *A.d_pos + tok;
  long long acc = 0; float2 nrs = make_float2(0.f, 0.f);
  if (A.xsm.us) { acc = q35_xsm_ld(part, A.xsm.mode, nw, A.M, N, tok, n); if (A.xsm.mode != 3) nrs = q35_xsm_row(A.xsm.us, A.xsm.S, A.xsm.eps, tok); }
  buf[tid] = A.xsm.us ? (A.xsm.mode == 3 ? __fmul_rn(__uint_as_float((unsigned)acc), c_tau) : __fmul_rn(__fmul_rn(q35_xsm_fin(acc, A.xsm.mode, nrs.x), nrs.y), c_tau))
                      : __fmul_rn(c_tau, q35_combine(part, nw, A.M, N, tok, n)); __syncthreads();
  q35_inv_h512s(buf, (float)c_sg);
  const float y = rbf(buf[tid]);
  if (blk >= Q35_NH + 2) {                                                  
    const int kvh = (blk - Q35_NH - 2) * 2 + (tid >> 8);
    if (A.v8) {
      float vy=y;
#if YSKN_KVX_ROT
      vy=kvx_rotate(vy,buf,false);
#endif
      q35_kv8_store(vy,A.v8,A.vs,q35_kvrow(A.vstg,A.max_ctx,kvh,pos,tok),d); return; }   
    A.vc[((size_t)kvh * A.max_ctx + pos) * Q35_HD + d] = q35_f2kv16(y, A.kv_fp16);
    return;
  }
  if (blk < Q35_NH && tid >= Q35_HD) A.gate16[((size_t)tok * Q35_NH + blk) * Q35_HD + d] = f2bf(y);   
  float ss = __fmul_rn(y, y);                                               
  for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
  if (lane == 0) s_red[warp] = ss;
  __syncthreads();
  const int w0 = (tid >> 8) * 8; float tot = 0.f;
  for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, s_red[w0 + w]);
  const float r = q35_rsqrt(__fadd_rn(__fdiv_rn(tot, (float)Q35_HD), A.eps));
  const float xn = rbf(__fmul_rn(__fmul_rn(y, r), __fadd_rn(1.f, bf2f(c_nw))));
  buf[tid] = xn;
  __syncthreads();
  float out = xn;                                                           
  if (d < Q35_RD / 2) { const float2 cs = A.rope[(size_t)pos * (Q35_RD / 2) + d];
    out = rbf(__fadd_rn(rbf(__fmul_rn(xn, cs.x)), rbf(__fmul_rn(-buf[tid + Q35_RD / 2], cs.y)))); }
  else if (d < Q35_RD) { const float2 cs = A.rope[(size_t)pos * (Q35_RD / 2) + d - Q35_RD / 2];
    out = rbf(__fadd_rn(rbf(__fmul_rn(xn, cs.x)), rbf(__fmul_rn(buf[tid - Q35_RD / 2], cs.y)))); }
#if YSKN_KVX_ROT
  out=kvx_rotate(out,buf,false);
#endif
  if (blk < Q35_NH) { if (tid < Q35_HD) A.q16[((size_t)tok * Q35_NH + blk) * Q35_HD + d] = f2bf(out); }
  else { const int kvh = (blk - Q35_NH) * 2 + (tid >> 8);
         if (A.k8) q35_kv8_store(out, A.k8, A.ks, q35_kvrow(A.kstg, A.max_ctx, kvh, pos, tok), d); else A.kc[((size_t)kvh * A.max_ctx + pos) * Q35_HD + d] = q35_f2kv16(out, A.kv_fp16); }
}
void launch_q35_attn_post(const Q35AttnPostArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M) && A.xsm.us && A.xsm.mode == 3 && ((A.k8 && A.v8) || (A.kc && A.vc)) && !A.pf.p[0]) {   
    const int tasks = (Q35_NH + 4) * ((A.M + Q35_PF_TPW - 1) / Q35_PF_TPW); launch_k(q35_attn_post_pf_kernel, dim3((tasks + 7) / 8), dim3(256), 0, st, A); return; } launch_k(q35_attn_post_kernel, dim3(Q35_NH + 4, A.M), dim3(512), 0, st, A); }

#if YSKN_KVX_MODE == 3



__global__ void __launch_bounds__(512) tb_requant_kernel(Q35AttnPostArgs A) {
  const int kind = blockIdx.x >> 1, hp = blockIdx.x & 1, tok = blockIdx.y, tid = threadIdx.x, d = tid & (Q35_HD - 1);
  const int kvh = hp * 2 + (tid >> 8);
  const int pos = *A.d_pos + tok;                                           
  int8_t* const q8 = kind ? A.v8 : A.k8; float* const qs = kind ? A.vs : A.ks;
  const size_t row = (size_t)kvh * A.max_ctx + pos;
  float v = __fmul_rn(kvx_scalar(q8[row * Q35_HD + d]), qs[row]);
  v = tb_fakequant(v, A.tb, (A.tb_layer * Q35_KVH + kvh) * 2 + kind, pos);
  kvx_store(v, q8, qs, row, d);
}
void launch_tb_requant(const Q35AttnPostArgs& A, cudaStream_t st) {
  const TbHost& H = tbf_host();
  if (!H.scr) { tb_requant_kernel<<<dim3(4, A.M), dim3(512), 0, st>>>(A); return; }      
  TbFastArgs F{};                                                                         
  F.blob = A.tb; F.nm = H.nm; F.has_cdq = H.has_cdq; F.q8_0 = A.k8; F.q8_1 = A.v8; F.qs_0 = A.ks; F.qs_1 = A.vs;
  F.max_ctx = A.max_ctx; F.d_pos = A.d_pos; F.layer = A.tb_layer; F.scr = H.scr;
  F.stg_k = A.kstg; F.stg_v = A.vstg; F.pk_pay = A.kp; F.pk_pay1 = A.vp; F.pk_scl0 = A.kps; F.pk_scl1 = A.vps; F.pk_stride = A.pks;   
  for (int t0 = 0; t0 < A.M; t0 += TBF_TILE) { F.tok0 = t0; F.nrow = 8 * std::min(TBF_TILE, A.M - t0); launch_tb_fast(F, st); }
}
#endif


__constant__ unsigned q35_inv_freq[32] = {
  0x3f800000u, 0x3f1ab32bu, 0x3ebaf81bu, 0x3e61f835u, 0x3e088d77u, 0x3da50956u, 0x3d47763fu, 0x3cf11177u, 0x3c91ad39u, 0x3c301052u, 0x3bd4ca15u,
  0x3b80967du, 0x3b1b690du, 0x3abbd3edu, 0x3a6301e2u, 0x3a092e02u, 0x39a5cb60u, 0x394860c1u, 0x38f22ce2u, 0x3892587eu, 0x3830df52u, 0x37d5c441u,
  0x37812dabu, 0x371c1fc4u, 0x36bcb0c1u, 0x36640cc6u, 0x3609cf4au, 0x35a68e4cu, 0x35494c57u, 0x34f3499du, 0x3493048eu, 0x3431af44u};

__global__ void q35_rope_table_kernel(float2* out, int max_ctx, int pm) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= max_ctx * (Q35_RD / 2)) return;
  const int t = idx / (Q35_RD / 2), i = idx % (Q35_RD / 2);
  const float f = __fmul_rn((float)t, __uint_as_float(q35_inv_freq[i]));
  if (pm) { float c, s; pm_sincos_f(f, c, s); out[idx] = make_float2(rbf(c), rbf(s)); }
  else out[idx] = make_float2(rbf(cosf(f)), rbf(sinf(f)));
}
void launch_q35_rope_table(float2* out, int max_ctx, cudaStream_t st, int pm) {
  q35_rope_table_kernel<<<(max_ctx * (Q35_RD / 2) + 255) / 256, 256, 0, st>>>(out, max_ctx, pm);
}


__global__ void q35_visual_rope_kernel(float2* out, int max_ctx, const int* positions, int n, int delta, int pm) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_ctx * (Q35_RD / 2)) return;
  const int t = idx / (Q35_RD / 2), i = idx % (Q35_RD / 2);
  const int axis = (i % 3 == 1 && i < 33) ? 1 : ((i % 3 == 2 && i < 30) ? 2 : 0);
  const int p = t < n ? positions[axis * n + t] : t + delta;
  const float f = __fmul_rn((float)p, __uint_as_float(q35_inv_freq[i]));
  if (pm) { float c, s; pm_sincos_f(f, c, s); out[idx] = make_float2(rbf(c), rbf(s)); }
  else out[idx] = make_float2(rbf(cosf(f)), rbf(sinf(f)));
}
void launch_q35_visual_rope(float2* out, int max_ctx, const int* positions, int n, int delta, cudaStream_t st, int pm) {
  q35_visual_rope_kernel<<<(max_ctx * (Q35_RD / 2) + 255) / 256, 256, 0, st>>>(out, max_ctx, positions, n, delta, pm);
}
__global__ void q35_visual_embed_kernel(uint16_t* rows, const uint16_t* table, const int* ids,
                                       const uint16_t* features, const int* map, const int* pos, int M, int max_ctx) {
  const int m = blockIdx.x, p = *pos + m;
  if (m >= M || p < 0 || p >= max_ctx) return;
  const int v = map[p];
  if (v < 0 && !table) return; 
  const uint16_t* src = v >= 0 ? features + (size_t)v * Q35_H : table + (size_t)ids[m] * Q35_H;
  for (int j = threadIdx.x; j < Q35_H; j += blockDim.x) rows[(size_t)m * Q35_H + j] = src[j];
}
void launch_q35_visual_embed(uint16_t* rows, const uint16_t* table, const int* ids,
                            const uint16_t* features, const int* map, const int* pos, int M, int max_ctx, cudaStream_t st) {
  q35_visual_embed_kernel<<<M, 256, 0, st>>>(rows, table, ids, features, map, pos, M, max_ctx);
}
void q35_set_pm(int on, cudaStream_t st) { cudaMemcpyToSymbolAsync(q35_pm_on, &on, sizeof(int), 0, cudaMemcpyHostToDevice, st); }






template <int G>
__device__ __forceinline__ void q35_softmax(float s, float& mh, float& lh, float (&acc)[G][8], const float (&vd)[8]) {
  const float mn = fmaxf(mh, s);
  const bool gt = s > mh;
  const float e = expf((gt ? mh : s) - mn);
  const float a = gt ? e : 1.f, b = gt ? 1.f : e;
  lh = lh * a + b; mh = mn;
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    const float ai = __shfl_sync(0xffffffffu, a, 4 * i), bi = __shfl_sync(0xffffffffu, b, 4 * i);
    #pragma unroll
    for (int e2 = 0; e2 < 8; ++e2) acc[i][e2] = acc[i][e2] * ai + bi * vd[e2];
  }
}
template <int G>
__device__ __forceinline__ void q35_softmax8(float s, float& mh, float& lh, float (&acc)[G][8], const float (&vd)[8], float vs) {   
  const float mn = fmaxf(mh, s);
  const bool gt = s > mh;
  const float e = expf((gt ? mh : s) - mn);
  const float a = gt ? e : 1.f, b = gt ? 1.f : e;
  lh = lh * a + b; mh = mn;
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    const float ai = __shfl_sync(0xffffffffu, a, 4 * i), bi = __fmul_rn(__shfl_sync(0xffffffffu, b, 4 * i), vs);
    #pragma unroll
    for (int e2 = 0; e2 < 8; ++e2) acc[i][e2] = acc[i][e2] * ai + bi * vd[e2];
  }
}


template <bool KV8>
__global__ void __launch_bounds__(128) q35_attention_kernel(Q35AttnArgs A) {
  constexpr int G = Q35_NH / Q35_KVH, PF = 4;                             
  __shared__ float sm[4][G], sl[4][G], sacc[4][G][Q35_HD];
  const int kvh = blockIdx.x, split = blockIdx.y, tok = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint16_t* kb = A.kc + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane;
  const uint16_t* vb = A.vc + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane;
  const int8_t* const kb8 = KV8 ? A.k8 + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane : nullptr;   
  const int8_t* const vb8 = KV8 ? A.v8 + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane : nullptr;
  const float* const ksb = KV8 ? A.ks + (size_t)kvh * A.max_ctx : nullptr; const float* const vsb = KV8 ? A.vs + (size_t)kvh * A.max_ctx : nullptr;
  auto ldkv = [&](uint4& kk, uint4& vv, float& ss, float& tt, size_t t) {
    if constexpr (KV8) { const uint2 k = __ldcg(reinterpret_cast<const uint2*>(kb8 + t * Q35_HD)), v = __ldcg(reinterpret_cast<const uint2*>(vb8 + t * Q35_HD));
                         kk = make_uint4(k.x, k.y, 0u, 0u); vv = make_uint4(v.x, v.y, 0u, 0u); ss = __ldcg(ksb + t); tt = __ldcg(vsb + t); }
    else { kk = __ldcg(reinterpret_cast<const uint4*>(kb + t * Q35_HD)); vv = __ldcg(reinterpret_cast<const uint4*>(vb + t * Q35_HD)); (void)ss; (void)tt; }
  };
  uint4 kq[PF], vq[PF]; float ksq[PF], vsq[PF];
  #pragma unroll
  for (int i = 0; i < PF; ++i) { kq[i] = vq[i] = make_uint4(0u, 0u, 0u, 0u); ksq[i] = vsq[i] = 0.f; }
  int base = 0;
  if (A.early) {                                                           
    base = *A.d_pos; const int len = base + tok + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);
    #pragma unroll
    for (int i = 0; i < PF; ++i) { const int t = t0 + warp + 4 * i;
      if (t < t1 && t < base) { ldkv(kq[i], vq[i], ksq[i], vsq[i], (size_t)t); } }
  }
  pdl_wait(); pdl_trigger();
  if (!A.early) base = *A.d_pos;
  const int pos = base + tok, len = pos + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);
  #pragma unroll
  for (int i = 0; i < PF; ++i) { const int t = t0 + warp + 4 * i;              
    if (t < t1 && (t >= base || !A.early)) { ldkv(kq[i], vq[i], ksq[i], vsq[i], (size_t)t); } }
  float q[G][8];
  #pragma unroll
  for (int i = 0; i < G; ++i) { const uint4 u = *reinterpret_cast<const uint4*>(A.q16 + ((size_t)tok * Q35_NH + kvh * G + i) * Q35_HD + 8 * lane);
    const unsigned uu[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int e = 0; e < 4; ++e) { q[i][2 * e] = __fmul_rn(__uint_as_float(uu[e] << 16), 0.0625f); q[i][2 * e + 1] = __fmul_rn(__uint_as_float(uu[e] & 0xffff0000u), 0.0625f); } }
  float mh = -1e30f, lh = 0.f, acc[G][8];                                 
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f; }
  for (int tb = t0 + warp; tb < t1; tb += 4 * PF) {
    uint4 kn[PF], vn[PF]; float ksn[PF], vsn[PF];
    #pragma unroll
    for (int i = 0; i < PF; ++i) { const int t = tb + 4 * PF + 4 * i; kn[i] = vn[i] = make_uint4(0u, 0u, 0u, 0u); ksn[i] = vsn[i] = 0.f;
      if (t < t1) { ldkv(kn[i], vn[i], ksn[i], vsn[i], (size_t)t); } }
    #pragma unroll
    for (int i = 0; i < PF; ++i) {
    if (tb + 4 * i >= t1) break;
    const uint4 kc = kq[i], vc = vq[i];
    const float ksc = KV8 ? ksq[i] : 0.f, vsc = KV8 ? vsq[i] : 0.f; (void)ksc; (void)vsc;
    float kd[8], vd[8];
    if constexpr (KV8) {                                                   
      #pragma unroll
      for (int e = 0; e < 8; ++e) { kd[e] = (float)(signed char)((e < 4 ? kc.x : kc.y) >> (8 * (e & 3))); vd[e] = (float)(signed char)((e < 4 ? vc.x : vc.y) >> (8 * (e & 3))); }
    } else
    { const unsigned ku[4] = {kc.x, kc.y, kc.z, kc.w}, vu[4] = {vc.x, vc.y, vc.z, vc.w};
      #pragma unroll
      for (int e = 0; e < 4; ++e) { kd[2 * e] = __uint_as_float(ku[e] << 16); kd[2 * e + 1] = __uint_as_float(ku[e] & 0xffff0000u);
                                    vd[2 * e] = __uint_as_float(vu[e] << 16); vd[2 * e + 1] = __uint_as_float(vu[e] & 0xffff0000u); } }
    float p[8];
    #pragma unroll
    for (int i = 0; i < G; ++i) { float s = __fmul_rn(q[i][0], kd[0]);
      #pragma unroll
      for (int e = 1; e < 8; ++e) s = __fadd_rn(s, __fmul_rn(q[i][e], kd[e]));
      p[i] = s; }
    #pragma unroll
    for (int i = G; i < 8; ++i) p[i] = 0.f;
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    float v4[4], v2[2], v1;
    #pragma unroll
    for (int k = 0; k < 4; ++k) { const float send = b4 ? p[k] : p[k + 4], keep = b4 ? p[k + 4] : p[k]; v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16); }
    #pragma unroll
    for (int k = 0; k < 2; ++k) { const float send = b3 ? v4[k] : v4[k + 2], keep = b3 ? v4[k + 2] : v4[k]; v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8); }
    { const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0]; v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4); }
    v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
    if constexpr (KV8) q35_softmax8<G>(__fmul_rn(v1, ksc), mh, lh, acc, vd, vsc); else q35_softmax<G>(v1, mh, lh, acc, vd);
    }
    #pragma unroll
    for (int i = 0; i < PF; ++i) { kq[i] = kn[i]; vq[i] = vn[i]; ksq[i] = ksn[i]; vsq[i] = vsn[i]; }
  }
  if ((lane & 3) == 0 && lane < 4 * G) { sm[warp][lane >> 2] = mh; sl[warp][lane >> 2] = lh; }
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) sacc[warp][i][8 * lane + e] = acc[i][e]; }
  __syncthreads();
  for (int i = warp; i < G; i += 4) {                                      
    float M2 = sm[0][i]; for (int w = 1; w < 4; ++w) M2 = fmaxf(M2, sm[w][i]);
    float L2 = 0.f, o8[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) o8[e] = 0.f;
    for (int w = 0; w < 4; ++w) { const float sc = (sm[w][i] > -1e29f) ? expf(sm[w][i] - M2) : 0.f; L2 += sc * sl[w][i];
      #pragma unroll
      for (int e = 0; e < 8; ++e) o8[e] += sc * sacc[w][i][8 * lane + e]; }
    const size_t pidx = ((size_t)tok * A.splits + split) * Q35_NH + kvh * G + i;
    if (lane == 0) { A.pm[pidx] = M2; A.pl[pidx] = L2; }
    #pragma unroll
    for (int e = 0; e < 8; ++e) A.pacc[pidx * Q35_HD + 8 * lane + e] = o8[e];
  }
}
void launch_q35_attention(const Q35AttnArgs& A, cudaStream_t st) {
  if (A.k8) launch_k(q35_attention_kernel<true>, dim3(Q35_KVH, A.splits, A.M), dim3(128), 0, st, A);
  else launch_k(q35_attention_kernel<false>, dim3(Q35_KVH, A.splits, A.M), dim3(128), 0, st, A); }


__global__ void __launch_bounds__(512) q35_combine_kernel(Q35CombineArgs A) {
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, e = blk * 512 + tid, head = e / Q35_HD, d = e % Q35_HD;
  const int8_t c_si = A.sg_in[e];
  pdl_wait(); pdl_trigger();
  float y;
  int S = A.splits;                                                        
  if (A.splits_short > 0) { const int dp = A.wk ? (int)A.wk[tok] : tok; if (*A.d_pos + dp + 1 < A.long_len) S = A.splits_short; }
  if (S <= 32) {                                                    
    
    __shared__ float s_sc[2][32], s_L[2];                                  
    float pa[32];
    #pragma unroll
    for (int s = 0; s < 32; ++s) pa[s] = s < S ? A.pacc[(((size_t)tok * A.splits + s) * Q35_NH + head) * Q35_HD + d] : 0.f;
    if (tid < 64) {                                                        
      const int hh = tid >> 5, sp = tid & 31; const size_t p = ((size_t)tok * A.splits + sp) * Q35_NH + blk * 2 + hh;
      const float pmv = sp < S ? A.pm[p] : -1e30f, plv = sp < S ? A.pl[p] : 0.f;
      float M2 = pmv;                                                      
      for (int o2 = 16; o2; o2 >>= 1) M2 = fmaxf(M2, __shfl_xor_sync(0xffffffffu, M2, o2));
      M2 = fmaxf(-1e30f, M2);
      const float sc = (pmv > -1e29f) ? expf(pmv - M2) : 0.f;
      s_sc[hh][sp] = sc;
      float L2 = 0.f;                                                      
      for (int k = 0; k < 32; ++k) L2 += __shfl_sync(0xffffffffu, sc, k) * __shfl_sync(0xffffffffu, plv, k);
      if (sp == 0) s_L[hh] = L2;
    }
    __syncthreads();
    const int hh = head - blk * 2; float o = 0.f;
    #pragma unroll
    for (int s = 0; s < 32; ++s) o += s_sc[hh][s] * pa[s];
    #if YSKN_KVX_ROT
    y = __fdiv_rn(o, s_L[hh]);
#else
    y = rbf(__fdiv_rn(o, s_L[hh]));
#endif
  } else if (S <= 64) {                                           
    __shared__ float s_sc2[2][64], s_L2[2];
    float pa[64];
    #pragma unroll
    for (int s = 0; s < 64; ++s) pa[s] = s < S ? A.pacc[(((size_t)tok * A.splits + s) * Q35_NH + head) * Q35_HD + d] : 0.f;
    if (tid < 64) {
      const int hh = tid >> 5, sp = tid & 31;
      const size_t p0 = ((size_t)tok * A.splits + sp) * Q35_NH + blk * 2 + hh, p1 = ((size_t)tok * A.splits + sp + 32) * Q35_NH + blk * 2 + hh;
      const float pm0 = A.pm[p0], pl0 = A.pl[p0];
      const float pm1 = sp + 32 < S ? A.pm[p1] : -1e30f, pl1 = sp + 32 < S ? A.pl[p1] : 0.f;
      float M2 = fmaxf(pm0, pm1);
      for (int o2 = 16; o2; o2 >>= 1) M2 = fmaxf(M2, __shfl_xor_sync(0xffffffffu, M2, o2));
      M2 = fmaxf(-1e30f, M2);
      const float sc0 = (pm0 > -1e29f) ? expf(pm0 - M2) : 0.f, sc1 = (pm1 > -1e29f) ? expf(pm1 - M2) : 0.f;
      s_sc2[hh][sp] = sc0; s_sc2[hh][sp + 32] = sc1;
      float L2 = 0.f;                                                      
      for (int k = 0; k < 32; ++k) L2 += __shfl_sync(0xffffffffu, sc0, k) * __shfl_sync(0xffffffffu, pl0, k);
      for (int k = 0; k < 32; ++k) L2 += __shfl_sync(0xffffffffu, sc1, k) * __shfl_sync(0xffffffffu, pl1, k);
      if (sp == 0) s_L2[hh] = L2;
    }
    __syncthreads();
    const int hh = head - blk * 2; float o = 0.f;
    #pragma unroll
    for (int s = 0; s < 64; ++s) o += s_sc2[hh][s] * pa[s];
    #if YSKN_KVX_ROT
    y = __fdiv_rn(o, s_L2[hh]);
#else
    y = rbf(__fdiv_rn(o, s_L2[hh]));
#endif
  } else {
    float M2 = -1e30f; for (int s = 0; s < A.splits; ++s) M2 = fmaxf(M2, A.pm[((size_t)tok * A.splits + s) * Q35_NH + head]);
    float L2 = 0.f, o = 0.f;
    for (int s = 0; s < A.splits; ++s) { const size_t p = ((size_t)tok * A.splits + s) * Q35_NH + head; const float sc = (A.pm[p] > -1e29f) ? expf(A.pm[p] - M2) : 0.f;
      L2 += sc * A.pl[p]; o += sc * A.pacc[p * Q35_HD + d]; }
    #if YSKN_KVX_ROT
    y = __fdiv_rn(o, L2);
#else
    y = rbf(__fdiv_rn(o, L2));
#endif
  }
#if YSKN_KVX_ROT
  y=rbf(kvx_rotate(y,buf,true));
#endif
  const float sg = rbf(q35_sigmoid(bf2f(A.gate16[((size_t)tok * Q35_NH + head) * Q35_HD + d])));
  const float y2 = rbf(__fmul_rn(y, sg));
  if (tid == 0) s_amax = 0u;
  buf[tid] = __fmul_rn(y2, (float)c_si);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, A.lo ? A.qx + (size_t)blk * 4096 : A.qx + (size_t)tok * (Q35_NH * Q35_HD) + blk * 512, A.alpha + (size_t)tok * (Q35_NH * Q35_HD / 512) + blk, A.lo ? tok : -1);
}


__global__ void __launch_bounds__(512) q35_kz_combine_kernel(Q35CombineArgs A, float* dbg) {
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, e = blk * 512 + tid, head = e / Q35_HD, d = e % Q35_HD;
  const int8_t c_si = A.sg_in[e];
  pdl_wait(); pdl_trigger();
  int S = A.splits;
  if (A.splits_short > 0) { const int dp = A.wk ? (int)A.wk[tok] : tok; if (*A.d_pos + dp + 1 < A.long_len) S = A.splits_short; }
  int RN = KZ_NONE, RD = KZ_NONE; float aN = 0.f, aD = 0.f;
  for (int s = 0; s < S; ++s) {
    const size_t p = ((size_t)tok * A.splits + s) * Q35_NH + head;
    const unsigned pk = __float_as_uint(A.pm[p]);
    if (pk == 0x80008000u) continue;
    const int xn = (int)(short)(pk >> 16), xd = (int)(short)(pk & 0xffffu);
    float rs, vs; bool use; kz_ref(RN, xn, rs, vs, use);
    if (rs != 1.f) aN = __fmul_rn(aN, rs);
    if (use) aN = __fadd_rn(aN, __fmul_rn(A.pacc[p * Q35_HD + d], vs));
    kz_ref(RD, xd, rs, vs, use);
    if (rs != 1.f) aD = __fmul_rn(aD, rs);
    if (use) aD = __fadd_rn(aD, __fmul_rn(A.pl[p], vs));
  }
  const float o = __fmul_rn(__fdiv_rn(aN, aD), kz_p2(RN - RD));
  if (dbg) dbg[((size_t)tok * Q35_NH + head) * Q35_HD + d] = o;
  const float y = rbf(o);
  const float sg = rbf(q35_sigmoid(bf2f(A.gate16[((size_t)tok * Q35_NH + head) * Q35_HD + d])));
  const float y2 = rbf(__fmul_rn(y, sg));
  if (tid == 0) s_amax = 0u;
  buf[tid] = __fmul_rn(y2, (float)c_si);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, A.lo ? A.qx + (size_t)blk * 4096 : A.qx + (size_t)tok * (Q35_NH * Q35_HD) + blk * 512, A.alpha + (size_t)tok * (Q35_NH * Q35_HD / 512) + blk, A.lo ? tok : -1);
}
void launch_q35_kz_combine(const Q35CombineArgs& A, float* dbg, cudaStream_t st) {
  launch_k(q35_kz_combine_kernel, dim3(Q35_NH * Q35_HD / 512, A.M), dim3(512), 0, st, A, dbg);
}
void launch_q35_combine(const Q35CombineArgs& A, cudaStream_t st) { launch_k(q35_combine_kernel, dim3(Q35_NH * Q35_HD / 512, A.M), dim3(512), 0, st, A); }


__global__ void __launch_bounds__(512) q35_mlp_kernel(Q35MlpArgs A) {
  __shared__ float bg[512], bu[512], tg[512], tu[512]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, n = blk * 512 + tid;
  const float c_tg = A.tau_g[n], c_tu = A.tau_u[n]; const int8_t c_sg = A.sg_g[n], c_su = A.sg_u[n], c_si = A.sg_in[n];
  pdl_wait(); pdl_trigger();
  if (tid == 0 && A.pf.p[0]) q35_l2pf(A.pf, blk + tok * gridDim.x, gridDim.x * gridDim.y);   
  float vg, vu;
  if (A.xsm.us) { const long long ag = q35_xsm_ld(A.part_g, A.xsm.mode, A.nw_g, A.M, A.F, tok, n), au = q35_xsm_ld(A.part_u, A.xsm.mode, A.nw_u, A.M, A.F, tok, n);
    if (A.xsm.mode == 3) { vg = __fmul_rn(__uint_as_float((unsigned)ag), c_tg); vu = __fmul_rn(__uint_as_float((unsigned)au), c_tu); }
    else { const float2 nrs = q35_xsm_row(A.xsm.us, A.xsm.S, A.xsm.eps, tok);
      vg = __fmul_rn(__fmul_rn(q35_xsm_fin(ag, A.xsm.mode, nrs.x), nrs.y), c_tg); vu = __fmul_rn(__fmul_rn(q35_xsm_fin(au, A.xsm.mode, nrs.x), nrs.y), c_tu); } }
  else { vg = __fmul_rn(c_tg, q35_combine(A.part_g, A.nw_g, A.M, A.F, tok, n)); vu = __fmul_rn(c_tu, q35_combine(A.part_u, A.nw_u, A.M, A.F, tok, n)); }
  if (tid == 0) s_amax = 0u;
  fwht512_v2(vg, vu, bg, tg, bu, tu);                                      
  const float g = rbf(__fmul_rn(__fmul_rn(vg, 0.044194173824159216f), (float)c_sg));
  const float u = rbf(__fmul_rn(__fmul_rn(vu, 0.044194173824159216f), (float)c_su));
  const float x = rbf(__fmul_rn(rbf(q35_silu(g)), u));
  bu[tid] = __fmul_rn(x, (float)c_si);                                     
  __syncthreads();
  a8_group512_tail(bu, &s_amax, A.lo ? A.qx + (size_t)blk * 4096 : A.qx + (size_t)tok * A.F + blk * 512, A.alpha + (size_t)tok * (A.F / 512) + blk, A.lo ? tok : -1);
}
void launch_q35_mlp(const Q35MlpArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M) && A.xsm.us && A.xsm.mode == 3 && !A.lo && !A.pf.p[0]) {   
    const int tasks = (A.F / 512) * ((A.M + Q35_PF_TPW - 1) / Q35_PF_TPW); launch_k(q35_mlp_pf_kernel, dim3((tasks + 7) / 8), dim3(256), 0, st, A); return; } launch_k(q35_mlp_kernel, dim3(A.F / 512, A.M), dim3(512), 0, st, A); }

__global__ void __launch_bounds__(512) q35_res_norm_xsm_kernel(Q35XsmResArgs A) {
  __shared__ float buf[512]; __shared__ float s_red4[4]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, n = blk * 512 + tid;
  float* const hr = A.h + (size_t)tok * Q35_H;
  float c_tau = 0.f, c_h = 0.f; int8_t c_so = 0;
  if (A.part) { c_tau = A.tau[n]; c_so = A.sg_out[n]; c_h = __ldcg(hr + n); }   
  const uint16_t c_w = A.w[n]; const int8_t c_si = A.sg_in[n];
  pdl_wait(); pdl_trigger();
  if (tid == 0 && A.pf.p[0]) q35_l2pf(A.pf, blk + tok * gridDim.x, gridDim.x * gridDim.y);   
  float hn;
  if (A.part) {
    const long long acc = q35_xsm_ld(A.part, A.mode, A.nw, A.M, Q35_H, tok, n); const float2 nrs = q35_xsm_row(A.us, nullptr, 0.f, tok);
    buf[tid] = __fmul_rn(q35_xsm_fin(acc, A.mode, nrs.x), c_tau); __syncthreads();
    q35_inv_h512s(buf, (float)c_so);
    hn = rbf(__fadd_rn(c_h, rbf(buf[tid])));
  } else hn = bf2f(A.emb[(size_t)A.ids[tok] * Q35_H + n]);
  hr[n] = hn;
  buf[tid] = hn;
  if (tid == 0) s_amax = 0u;
  __syncthreads();
  if (tid < 128) {                                                         
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) { const float x = buf[tid + 128 * j]; acc = __fadd_rn(acc, __fmul_rn(x, x)); }
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red4[tid >> 5] = acc;
  }
  __syncthreads();
  if (tid == 0) { float sb = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) sb = __fadd_rn(sb, s_red4[w]);
    A.S[(size_t)tok * Q35_SST + blk] = sb; }
  const float u = __fmul_rn(hn, __fadd_rn(1.f, bf2f(c_w)));                 
  if (A.xu) A.xu[(size_t)tok * Q35_H + n] = f2bf(u);                        
  buf[tid] = __fmul_rn(u, (float)c_si);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, A.lo ? A.qx + (size_t)blk * 4096 : A.qx + (size_t)tok * Q35_H + blk * 512, A.alpha + (size_t)tok * (Q35_H / 512) + blk, A.lo ? tok : -1);
}
void launch_q35_res_norm_xsm(const Q35XsmResArgs& A, cudaStream_t st) {
  if (q35_pf_launch(A.M) && (!A.part || A.mode == 1) && !A.lo && !A.pf.p[0]) {   
    const int tasks = (Q35_H / 512) * ((A.M + Q35_PF_TPW - 1) / Q35_PF_TPW); launch_k(q35_res_norm_pf_kernel, dim3((tasks + 7) / 8), dim3(256), 0, st, A); return; } launch_k(q35_res_norm_xsm_kernel, dim3(Q35_H / 512, A.M), dim3(512), 0, st, A); }
__global__ void __launch_bounds__(512) q35_xsm_xn_kernel(const float* h, const float* S, const uint16_t* w, float eps, uint16_t* xn) {
  pdl_wait(); pdl_trigger();
  const int tok = blockIdx.y, n = blockIdx.x * 512 + threadIdx.x;
  __shared__ float s_r; if (threadIdx.x == 0) s_r = q35_xsm_r(S + (size_t)tok * Q35_SST, eps);
  __syncthreads(); const float r = s_r;
  xn[(size_t)tok * Q35_H + n] = f2bf(__fmul_rn(__fmul_rn(__ldcg(h + (size_t)tok * Q35_H + n), r), __fadd_rn(1.f, bf2f(w[n]))));
}
void launch_q35_xsm_xn(const float* h, const float* S, const uint16_t* w, float eps, int M, uint16_t* xn, cudaStream_t st) {
  launch_k(q35_xsm_xn_kernel, dim3(Q35_H / 512, M), dim3(512), 0, st, h, S, w, eps, xn); }
__global__ void q35_pick_kernel(const int* idx, const int* ids, int* out, int* out2) {
  pdl_wait(); pdl_trigger();
  if (threadIdx.x == 0) { const int t = ids[idx[0]]; out[0] = t; if (out2) out2[0] = t; }
}
void launch_q35_pick(const int* idx, const int* ids, int* out, int* out2, cudaStream_t st) { launch_k(q35_pick_kernel, dim3(1), dim3(32), 0, st, idx, ids, out, out2); }

__global__ void q35_mtp_log_kernel(const int* top_i, const float* top_v, const int* ids, int* lg_idx, float* lg_val, int* cnt) {
  pdl_wait(); pdl_trigger();
  const int t = threadIdx.x, k = *cnt;
  if (k < 16 && t < 8) { lg_idx[k * 8 + t] = ids[top_i[t]]; lg_val[k * 8 + t] = top_v[t]; }
  __syncthreads();
  if (t == 0) *cnt = k + 1;
}
void launch_q35_mtp_log(const int* top_i, const float* top_v, const int* ids, int* lg_idx, float* lg_val, int* cnt, cudaStream_t st) {
  launch_k(q35_mtp_log_kernel, dim3(1), dim3(32), 0, st, top_i, top_v, ids, lg_idx, lg_val, cnt); }
__global__ void q35_iota_kernel(int* out, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = i; }
void launch_q35_iota(int* out, int n, cudaStream_t st) { q35_iota_kernel<<<(n + 127) / 128, 128, 0, st>>>(out, n); }


__device__ __forceinline__ uint32_t q35_read_window(const uint32_t* w, uint32_t start, uint32_t width) {   
  const uint32_t idx = start >> 5, sh = start & 31; uint32_t v = w[idx] >> sh;
  if (sh + width > 32) v |= w[idx + 1] << (32 - sh);
  return v & ((1u << width) - 1u);
}
__global__ void __launch_bounds__(512) q35_embed_kernel(Q35EmbArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float buf[Q35_H];
  const int id = A.ids[blockIdx.x], wps = 4 * A.K; const uint32_t* row = A.codes + (size_t)id * (Q35_H / 256) * wps; const float sc = h2f(A.scale16[id]);
  for (int i = threadIdx.x; i < (Q35_H / 256) * 128; i += 512) {
    const int q = i >> 7, j = i & 127; const uint32_t* w = row + q * wps; const uint32_t e = (uint32_t)A.K * (j + 1);
    const uint32_t s = (e >= 12u) ? q35_read_window(w, e - 12u, 12u) : (q35_read_window(w, 0u, e) << (12u - e));
    float v0 = __fmul_rn(h2f(A.lut16[2 * s]), sc), v1 = __fmul_rn(h2f(A.lut16[2 * s + 1]), sc);
    if (A.codes2) {                                                    
      const int wps2 = 4 * A.K2; const uint32_t* w2 = A.codes2 + ((size_t)id * (Q35_H / 256) + q) * wps2; const uint32_t e2 = (uint32_t)A.K2 * (j + 1);
      const uint32_t s2 = (e2 >= 12u) ? q35_read_window(w2, e2 - 12u, 12u) : (q35_read_window(w2, 0u, e2) << (12u - e2)); const float sc2 = h2f(A.scale2[id]);
      v0 = __fadd_rn(v0, __fmul_rn(h2f(A.lut2[2 * s2]), sc2)); v1 = __fadd_rn(v1, __fmul_rn(h2f(A.lut2[2 * s2 + 1]), sc2));
      if (A.codes3) {                                                  
        const int wps3 = 4 * A.K3; const uint32_t* w3 = A.codes3 + ((size_t)id * (Q35_H / 256) + q) * wps3; const uint32_t e3 = (uint32_t)A.K3 * (j + 1);
        const uint32_t s3 = (e3 >= 12u) ? q35_read_window(w3, e3 - 12u, 12u) : (q35_read_window(w3, 0u, e3) << (12u - e3)); const float sc3 = h2f(A.scale3[id]);
        v0 = __fadd_rn(v0, __fmul_rn(h2f(A.lut3[2 * s3]), sc3)); v1 = __fadd_rn(v1, __fmul_rn(h2f(A.lut3[2 * s3 + 1]), sc3));
      }
    }
    buf[q * 256 + 2 * j] = v0; buf[q * 256 + 2 * j + 1] = v1;
  }
  __syncthreads();
  fwht512_blocks(buf, Q35_H / 512);                                    
  for (int i = threadIdx.x; i < Q35_H; i += 512) A.out[(size_t)blockIdx.x * Q35_H + i] = f2bf(__fmul_rn(buf[i], (float)A.signs[i]));
}



__global__ void __launch_bounds__(256) q35_embed_blk_kernel(Q35EmbArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float x[512];
  const int b = blockIdx.x, t = threadIdx.x, id = A.ids[blockIdx.y], wps = 4 * A.K;
  const uint32_t* row = A.codes + (size_t)id * (Q35_H / 256) * wps; const float sc = h2f(A.scale16[id]);
  {
    const int q = 2 * b + (t >> 7), j = t & 127; const uint32_t* w = row + q * wps; const uint32_t e = (uint32_t)A.K * (j + 1);
    const uint32_t s = (e >= 12u) ? q35_read_window(w, e - 12u, 12u) : (q35_read_window(w, 0u, e) << (12u - e));
    float v0 = __fmul_rn(h2f(A.lut16[2 * s]), sc), v1 = __fmul_rn(h2f(A.lut16[2 * s + 1]), sc);
    if (A.codes2) {
      const int wps2 = 4 * A.K2; const uint32_t* w2 = A.codes2 + ((size_t)id * (Q35_H / 256) + q) * wps2; const uint32_t e2 = (uint32_t)A.K2 * (j + 1);
      const uint32_t s2 = (e2 >= 12u) ? q35_read_window(w2, e2 - 12u, 12u) : (q35_read_window(w2, 0u, e2) << (12u - e2)); const float sc2 = h2f(A.scale2[id]);
      v0 = __fadd_rn(v0, __fmul_rn(h2f(A.lut2[2 * s2]), sc2)); v1 = __fadd_rn(v1, __fmul_rn(h2f(A.lut2[2 * s2 + 1]), sc2));
      if (A.codes3) {
        const int wps3 = 4 * A.K3; const uint32_t* w3 = A.codes3 + ((size_t)id * (Q35_H / 256) + q) * wps3; const uint32_t e3 = (uint32_t)A.K3 * (j + 1);
        const uint32_t s3 = (e3 >= 12u) ? q35_read_window(w3, e3 - 12u, 12u) : (q35_read_window(w3, 0u, e3) << (12u - e3)); const float sc3 = h2f(A.scale3[id]);
        v0 = __fadd_rn(v0, __fmul_rn(h2f(A.lut3[2 * s3]), sc3)); v1 = __fadd_rn(v1, __fmul_rn(h2f(A.lut3[2 * s3 + 1]), sc3));
      }
    }
    x[(t >> 7) * 256 + 2 * j] = v0; x[(t >> 7) * 256 + 2 * j + 1] = v1;
  }
  __syncthreads();
  for (int h = 1; h < 512; h <<= 1) {                                    
    const int lo = ((t & ~(h - 1)) << 1) | (t & (h - 1)); const float a = x[lo], c = x[lo + h]; x[lo] = a + c; x[lo + h] = a - c;
    __syncthreads();
  }
  for (int i = t; i < 512; i += 256) {
    const float v = __fmul_rn(x[i], 0.044194173824159216f);
    A.out[(size_t)blockIdx.y * Q35_H + b * 512 + i] = f2bf(__fmul_rn(v, (float)A.signs[b * 512 + i]));
  }
}
__global__ void __launch_bounds__(32) q35_l2pf_kernel(Q35Pf f) {      
  if (threadIdx.x != 0) return;
  for (int s = 0; s < 2; ++s) {
    if (!f.p[s] || !f.n[s]) continue;
    for (unsigned long long o = (unsigned long long)blockIdx.x * 65536ull; o < f.n[s]; o += (unsigned long long)gridDim.x * 65536ull) {
      const unsigned len = (unsigned)((f.n[s] - o < 65536ull ? f.n[s] - o : 65536ull) & ~15ull);
      if (len) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(static_cast<const char*>(f.p[s]) + o), "r"(len) : "memory");
    }
  }
}
void launch_l2pf(const Q35Pf& f, cudaStream_t st) {
  const unsigned long long n = f.n[0] > f.n[1] ? f.n[0] : f.n[1]; if (!n) return;
  const int g = (int)std::min<unsigned long long>(64ull, (n + 65535ull) / 65536ull);
  q35_l2pf_kernel<<<g, 32, 0, st>>>(f); }
static bool q35_emb1_env() { static const bool v = [] { const char* e = std::getenv("YSKN_Q35_EMB1"); return e && e[0] == '1'; }(); return v; }
void launch_q35_embed(const Q35EmbArgs& A, int M, cudaStream_t st) {
  if (q35_emb1_env()) { launch_k(q35_embed_kernel, dim3(M), dim3(512), 0, st, A); return; }
  launch_k(q35_embed_blk_kernel, dim3(Q35_H / 512, M), dim3(256), 0, st, A); }
__global__ void q35_pos_slices_kernel(const int* base, int* out, int n, int step) { pdl_wait(); pdl_trigger(); const int i = threadIdx.x; if (i < n) out[i] = *base + i * step; }
void launch_q35_pos_slices(const int* base, int* out, int n, int step, cudaStream_t st) { launch_k(q35_pos_slices_kernel, dim3(1), dim3(64), 0, st, base, out, n, step); }
} 
#include "lfrow_kernels.cuh"
