// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include "q35.hpp"
#include <cstdint>
namespace yskn {







__device__ __forceinline__ unsigned long long mm_mix(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
__device__ __forceinline__ unsigned long long mm_rho(unsigned long long key, int r) { return (mm_mix(key ^ (unsigned long long)r) & 0xffffffffull) | 1ull; }
__device__ __forceinline__ unsigned long long mm_block_sum(unsigned long long v, unsigned long long* sh) {   
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  unsigned long long s = 0;
  if (threadIdx.x == 0) for (int k = 0; k < 8; ++k) s += sh[k];
  __syncthreads();
  return s;
}
__global__ void __launch_bounds__(256) q35_chk_kernel(Q35ChkArgs A) {
  pdl_wait(); pdl_trigger();
  const int m = blockIdx.x, G = A.Kc / 512;
  __shared__ unsigned long long sh[8]; __shared__ long long s_bound;
  float amax = 0.f;
  for (int g = 0; g < G; ++g) amax = fmaxf(amax, A.alpha[(size_t)m * G + g]);
  const float rm = amax > 0.f ? __fdiv_rn(16777216.f, amax) : 0.f;    
  unsigned long long rhs = 0, qa = 0;
  for (int i = threadIdx.x; i < A.Kc; i += 256) {
    const long long Ag = (long long)__float2int_rn(__fmul_rn(A.alpha[(size_t)m * G + i / 512], rm));   
    const int q = A.qx[(size_t)m * A.Kc + i];
    rhs += (unsigned long long)(Ag * q) * (unsigned long long)A.c[i];
    qa += (unsigned long long)(Ag * (q < 0 ? -q : q));
  }
  rhs = mm_block_sum(rhs, sh); qa = mm_block_sum(qa, sh);
  if (threadIdx.x == 0) s_bound = (long long)qa * A.w_max;
  __syncthreads();
  const long long bound = s_bound;
  unsigned long long lhs = 0, viol = 0;
  for (int r = threadIdx.x; r < A.N; r += 256) {
    const long long a = (long long)A.acc[(size_t)m * A.N + r];
    lhs += mm_rho(A.key, r) * (unsigned long long)a;
    viol += (a > bound || a < -bound) ? 1ull : 0ull;
  }
  lhs = mm_block_sum(lhs, sh); viol = mm_block_sum(viol, sh);
  if (threadIdx.x == 0) {
    atomicAdd(&A.st->checks, 1ull);
    const bool bad_eq = lhs != rhs, bad_rg = viol != 0ull;
    if (bad_eq) atomicAdd(&A.st->eq_fail, 1ull);
    if (bad_rg) atomicAdd(&A.st->range_fail, 1ull);
    if ((bad_eq || bad_rg) && atomicCAS(&A.st->first_set, 0u, 1u) == 0u) {
      A.st->first_tensor = A.tensor; A.st->first_row = m; A.st->first_lhs = lhs; A.st->first_rhs = rhs; A.st->first_viol = viol; }
  }
}

__global__ void q35_chk_fault_kernel(unsigned long long* acc, int kind, int row, int bit) {
  pdl_wait(); pdl_trigger();
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  acc[row] ^= 1ull << bit;
  if (kind == 2) acc[row + 1] ^= 1ull << bit;
}
void launch_q35_chk(const Q35ChkArgs& A, int M, cudaStream_t st) { launch_k(q35_chk_kernel, dim3(M), dim3(256), 0, st, A); }
void launch_q35_chk_fault(unsigned long long* acc, int kind, int row, int bit, cudaStream_t st) {
  launch_k(q35_chk_fault_kernel, dim3(1), dim3(32), 0, st, acc, kind, row, bit); }
} 
