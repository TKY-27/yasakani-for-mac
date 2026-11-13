// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include "q35.hpp"
#include "smp_hash.cuh"
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
namespace yskn {
















namespace {
constexpr int KJ_G = 256, KJ_T = 256, KJ_PT = 4;     
__device__ __forceinline__ bool kj_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

template <int L>
__device__ __forceinline__ void kj_pop(float (&tv)[L], int (&ti)[L], int k, float* out_val, int* out_idx) {
  __shared__ float wv[KJ_T / 32]; __shared__ int wi[KJ_T / 32], wt[KJ_T / 32]; __shared__ int s_win;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int n = 0; n < k; ++n) {
    float v = tv[0]; int i = ti[0], t = tid;
    for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, v, o); const int oi = __shfl_xor_sync(0xffffffffu, i, o), ot = __shfl_xor_sync(0xffffffffu, t, o);
      if (kj_better(ov, oi, v, i)) { v = ov; i = oi; t = ot; } }
    if (lane == 0) { wv[warp] = v; wi[warp] = i; wt[warp] = t; }
    __syncthreads();
    if (tid == 0) { float bv = wv[0]; int bi = wi[0], bt = wt[0];
      for (int w = 1; w < KJ_T / 32; ++w) if (kj_better(wv[w], wi[w], bv, bi)) { bv = wv[w]; bi = wi[w]; bt = wt[w]; }
      out_val[n] = bv; out_idx[n] = bi; s_win = bt; }
    __syncthreads();
    if (tid == s_win) {
      #pragma unroll
      for (int q = 0; q < L - 1; ++q) { tv[q] = tv[q + 1]; ti[q] = ti[q + 1]; }
      tv[L - 1] = -FLT_MAX; ti[L - 1] = 0x7fffffff; }
    __syncthreads();
  }
}
__device__ __forceinline__ unsigned long long kj_mix(unsigned long long z) {   
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
__device__ __forceinline__ double kj_uniform(unsigned long long seed, long long pos) {   
  return (double)(kj_mix(kj_mix(seed) ^ (unsigned long long)pos) >> 11) * (1.0 / 9007199254740992.0); }
__global__ void __launch_bounds__(KJ_T) smp_kernel(Q35SmpArgs A) { pdl_wait(); pdl_trigger();
  const int r = blockIdx.y, k = A.prm->top_k; const float* lg = A.logits + (size_t)r * A.V;
  float tv[KJ_PT]; int ti[KJ_PT];
  #pragma unroll
  for (int q = 0; q < KJ_PT; ++q) { tv[q] = -FLT_MAX; ti[q] = 0x7fffffff; }
  #pragma unroll
  for (int q = 0; q < KJ_PT; ++q) { const int v = (q * KJ_G + blockIdx.x) * KJ_T + threadIdx.x;
    if (v < A.V) { float x = lg[v]; int ix = v;
      #pragma unroll
      for (int s = 0; s < KJ_PT; ++s) if (kj_better(x, ix, tv[s], ti[s])) { const float t0 = tv[s]; const int i0 = ti[s]; tv[s] = x; ti[s] = ix; x = t0; ix = i0; } } }
  const size_t c = ((size_t)r * KJ_G + blockIdx.x) * Q35_SMP_K;
  kj_pop(tv, ti, k, A.cand_val + c, A.cand_idx + c);
  __shared__ int s_last; __threadfence(); __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(A.cnt + r, 1u) & (KJ_G - 1)) == KJ_G - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  float cv[Q35_SMP_K]; int ci[Q35_SMP_K];
  { const size_t c2 = ((size_t)r * KJ_G + threadIdx.x) * Q35_SMP_K;
    #pragma unroll
    for (int q = 0; q < Q35_SMP_K; ++q) { const bool in = q < k; cv[q] = in ? __ldcg(A.cand_val + c2 + q) : -FLT_MAX; ci[q] = in ? __ldcg(A.cand_idx + c2 + q) : 0x7fffffff; } }
  __shared__ float s_v[Q35_SMP_K]; __shared__ int s_i[Q35_SMP_K];
  kj_pop(cv, ci, k, s_v, s_i);                                           
  if (threadIdx.x != 0) return;
  const Q35SmpParams P = *A.prm;
  const int depth = A.wk ? (int)A.wk[A.row0 + r] : A.row0 + r;
  const long long pos = (long long)*A.d_pos + depth + 1;
  double w[Q35_SMP_K], tot = 0.0;
  for (int j = 0; j < k; ++j) { const double z = ((double)s_v[j] - (double)s_v[0]) / P.temp; w[j] = P.pm ? pm_exp_d(z) : exp(z); tot += w[j]; }
  int n = k; double run = 0.0;
  for (int j = 0; j < k; ++j) { run += w[j]; if (run >= P.top_p * tot) { n = j + 1; break; } }
  if (P.min_p > 0.0) { int m = 1; while (m < n && w[m] >= P.min_p) ++m; n = m; }   
  int pick = n - 1;
  if (P.version == 3) {                                                  
    double best = -INFINITY;                                             
    for (int j = 0; j < n; ++j) { const double g = ((double)s_v[j] - (double)s_v[0]) / P.temp + kh_gumbel(P.seed, pos, s_i[j], P.pm); if (g > best) { best = g; pick = j; } }
  } else {                                                               
    double kept = 0.0; for (int j = 0; j < n; ++j) kept += w[j];
    const double u = kj_uniform(P.seed, pos) * kept;
    run = 0.0;
    for (int j = 0; j < n; ++j) { run += w[j]; if (u < run) { pick = j; break; } }
  }
  const int id = s_i[pick];
  A.d_tok_out[r] = id;
  if (A.adv_pos) { *A.adv_pos += 1; *A.adv_tok = id; }                    
}






constexpr int K2_G = 64, K2_T = 256, K2_PT = 16;                     
template <int L>
__device__ __forceinline__ void k2_lane_best(const float (&v)[L], const int (&id)[L], float& bv, int& bi, int& lb) {
  bv = v[0]; bi = id[0]; lb = 0;
  #pragma unroll
  for (int q = 1; q < L; ++q) if (kj_better(v[q], id[q], bv, bi)) { bv = v[q]; bi = id[q]; lb = q; }
}

template <int L>
__device__ __forceinline__ void k2_warp_pop(float (&v)[L], int (&id)[L], int kk, float* out_v, int* out_i) {
  const int lane = threadIdx.x & 31; float bv; int bi, lb; k2_lane_best(v, id, bv, bi, lb);
  for (int n = 0; n < kk; ++n) {
    float wv = bv; int wi = bi, wl = lane;
    #pragma unroll
    for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, wv, o); const int oi = __shfl_xor_sync(0xffffffffu, wi, o), ol = __shfl_xor_sync(0xffffffffu, wl, o);
      if (kj_better(ov, oi, wv, wi)) { wv = ov; wi = oi; wl = ol; } }
    if (lane == 0) { out_v[n] = wv; out_i[n] = wi; }
    if (lane == wl) {
      #pragma unroll
      for (int q = 0; q < L; ++q) if (q == lb) { v[q] = -FLT_MAX; id[q] = 0x7fffffff; }
      k2_lane_best(v, id, bv, bi, lb); }
  }
}
__global__ void __launch_bounds__(K2_T) smp2_kernel(Q35SmpArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float s_wv[K2_T / 32][Q35_SMP_K]; __shared__ int s_wi[K2_T / 32][Q35_SMP_K];
  __shared__ float s_v[Q35_SMP_K]; __shared__ int s_i[Q35_SMP_K]; __shared__ int s_last;
  const int r = blockIdx.y, k = A.prm->top_k, kk = A.top_val ? (k > 8 ? k : 8) : k;   
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* lg = A.logits + (size_t)r * A.V;
  {
    float v[K2_PT]; int id[K2_PT];
    #pragma unroll
    for (int q = 0; q < K2_PT; ++q) { const int x = (blockIdx.x * K2_PT + q) * K2_T + threadIdx.x;
      v[q] = x < A.V ? lg[x] : -FLT_MAX; id[q] = x < A.V ? x : 0x7fffffff; }
    k2_warp_pop(v, id, kk, s_wv[warp], s_wi[warp]);
  }
  __syncthreads();
  if (warp == 0) {                                                       
    float c[5]; int ci[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) { const int f = lane + 32 * j; const bool in = f < (K2_T / 32) * kk;
      c[j] = in ? s_wv[f / kk][f % kk] : -FLT_MAX; ci[j] = in ? s_wi[f / kk][f % kk] : 0x7fffffff; }
    const size_t cb = ((size_t)r * K2_G + blockIdx.x) * Q35_SMP_K;
    k2_warp_pop(c, ci, kk, A.cand_val + cb, A.cand_idx + cb);
  }
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(A.cnt + r, 1u) & (K2_G - 1)) == K2_G - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  {                                                                      
    float c[5]; int ci[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) { const int f = lane + 32 * j; const bool in = f < 8 * kk;
      const size_t src = ((size_t)r * K2_G + warp * 8 + (in ? f / kk : 0)) * Q35_SMP_K + (in ? f % kk : 0);
      c[j] = in ? __ldcg(A.cand_val + src) : -FLT_MAX; ci[j] = in ? __ldcg(A.cand_idx + src) : 0x7fffffff; }
    __syncthreads();                                                     
    k2_warp_pop(c, ci, kk, s_wv[warp], s_wi[warp]);
  }
  __syncthreads();
  if (warp != 0) return;
  {
    float c[5]; int ci[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) { const int f = lane + 32 * j; const bool in = f < (K2_T / 32) * kk;
      c[j] = in ? s_wv[f / kk][f % kk] : -FLT_MAX; ci[j] = in ? s_wi[f / kk][f % kk] : 0x7fffffff; }
    k2_warp_pop(c, ci, kk, s_v, s_i);                                    
  }
  __syncwarp();
  if (A.top_val && lane < 8) { A.top_val[(size_t)r * 8 + lane] = s_v[lane]; A.top_idx[(size_t)r * 8 + lane] = s_i[lane]; }
  if (A.topk_val && lane < k) { A.topk_val[(size_t)r * Q35_SMP_K + lane] = s_v[lane]; A.topk_idx[(size_t)r * Q35_SMP_K + lane] = s_i[lane]; }   
  const Q35SmpParams P = *A.prm;                                        
  const int depth = A.wk ? (int)A.wk[A.row0 + r] : A.row0 + r;
  const long long pos = (long long)*A.d_pos + depth + 1;
  
  
  __shared__ double s_w[Q35_SMP_K], s_gz[Q35_SMP_K];
  if (lane < k) { const double z = ((double)s_v[lane] - (double)s_v[0]) / P.temp; s_w[lane] = P.pm ? pm_exp_d(z) : exp(z);
                  if (P.version == 3) s_gz[lane] = z + kh_gumbel(P.seed, pos, s_i[lane], P.pm); }
  __syncwarp();
  if (lane != 0) return;
  double w[Q35_SMP_K], tot = 0.0;
  for (int j = 0; j < k; ++j) { w[j] = s_w[j]; tot += w[j]; }
  int n = k; double run = 0.0;
  for (int j = 0; j < k; ++j) { run += w[j]; if (run >= P.top_p * tot) { n = j + 1; break; } }
  if (P.min_p > 0.0) { int m = 1; while (m < n && w[m] >= P.min_p) ++m; n = m; }   
  int pick = n - 1;
  if (P.version == 3) {                                                  
    double best = -INFINITY;                                             
    for (int j = 0; j < n; ++j) { const double g = s_gz[j]; if (g > best) { best = g; pick = j; } }
  } else {                                                               
    double kept = 0.0; for (int j = 0; j < n; ++j) kept += w[j];
    const double u = kj_uniform(P.seed, pos) * kept;
    run = 0.0;
    for (int j = 0; j < n; ++j) { run += w[j]; if (u < run) { pick = j; break; } }
  }
  const int id = s_i[pick];
  A.d_tok_out[r] = id;
  if (A.adv_pos) { *A.adv_pos += 1; *A.adv_tok = id; }
}
}  
bool q35_smp2_on() { static const bool v = [] { const char* e = std::getenv("YSKN_SMP2"); return !(e && e[0] == '0'); }(); return v; }
void launch_q35_smp(const Q35SmpArgs& A, cudaStream_t st) {
  if (A.V > KJ_G * KJ_T * KJ_PT) throw std::runtime_error("SMP: vocabulary larger than 262,144");
  if (q35_smp2_on()) { launch_k(smp2_kernel, dim3(K2_G, A.rows), dim3(K2_T), 0, st, A); return; }
  launch_k(smp_kernel, dim3(KJ_G, A.rows), dim3(KJ_T), 0, st, A);
}
}  
