// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors


#include "pdl.cuh"
#include <cuda_fp16.h>
#include <cfloat>
#include "a8_prologue.cuh"
#include "draft_kernels.hpp"
namespace yskn {



__global__ void __launch_bounds__(128) d3_prep_kernel(D3PrepArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ float s_red[4]; __shared__ unsigned s_amax;
  const int grp = blockIdx.x, row = blockIdx.y, tid = threadIdx.x, p = grp / 7, n = 3584 * A.nparts, m = 7 * A.nparts;
  const float* src = A.src[p] + (size_t)row * A.stride[p]; const uint16_t* gam = A.gain[p];
  float acc = 0.f; for (int i = tid; i < 3584; i += 128) acc = __fadd_rn(acc, __fmul_rn(src[i], src[i]));
  for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  if ((tid & 31) == 0) s_red[tid >> 5] = acc;
  __syncthreads();
  float tot = 0.f; for (int w = 0; w < 4; ++w) tot = __fadd_rn(tot, s_red[w]);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(tot, 3584.f), A.eps));
  for (int k = tid; k < 512; k += 128) {
    const int i = (grp - 7 * p) * 512 + k;                       
    const float v = round_f16(__fmul_rn(round_f16(__fmul_rn(src[i], r)), h2f(gam[i])));
    buf[k] = __fmul_rn(v, (float)A.signs[grp * 512 + k]);
  }
  if (tid == 0) s_amax = 0u;
  __syncthreads();
  for (int hh = 1; hh < 512; hh <<= 1) {
    for (int k = tid; k < 256; k += 128) { const int lo = ((k & ~(hh - 1)) << 1) | (k & (hh - 1)); const float a = buf[lo], b = buf[lo + hh]; buf[lo] = a + b; buf[lo + hh] = a - b; }
    __syncthreads();
  }
  for (int k = tid; k < 512; k += 128) { const float v = round_f16(__fmul_rn(buf[k], 0.044194173824159216f)); buf[k] = v; atomicMax(&s_amax, __float_as_uint(fabsf(v))); }
  __syncthreads();
  const float al = __fdiv_rn(__uint_as_float(s_amax), 127.f);
  if (tid == 0) A.alpha[(size_t)row * m + grp] = al;
  for (int k = tid; k < 512; k += 128) {
    const float v = al > 0.f ? __fdiv_rn(buf[k], al) : 0.f; int q = __float2int_rn(v); q = q < -127 ? -127 : (q > 127 ? 127 : q);
    A.qx[(size_t)row * n + grp * 512 + k] = (int8_t)q;
  }
}
void launch_d3_prep(const D3PrepArgs& A, cudaStream_t st) { launch_k(d3_prep_kernel, dim3(dim3(7 * A.nparts, A.M)), dim3(128), 0, st, A); }


__global__ void __launch_bounds__(512) d3_linear_out_kernel(D3LinOutArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, n = blk * 512 + tid;
  float s = A.part[(size_t)tok * A.N + n];
  for (int w = 1; w < A.nw; ++w) s = __fadd_rn(s, A.part[((size_t)w * A.M + tok) * A.N + n]);
  buf[tid] = __fmul_rn(A.tau[n], s); __syncthreads();
  for (int h = 1; h < 512; h <<= 1) {
    if (tid < 256) { const int lo = ((tid & ~(h - 1)) << 1) | (tid & (h - 1)); const float a = buf[lo], b = buf[lo + h]; buf[lo] = a + b; buf[lo + h] = a - b; }
    __syncthreads();
  }
  A.out[(size_t)tok * A.out_stride + n] = round_f16(__fmul_rn(__fmul_rn(buf[tid], 0.044194173824159216f), (float)A.sg[n]));
}
void launch_d3_linear_out(const D3LinOutArgs& A, cudaStream_t st) { launch_k(d3_linear_out_kernel, dim3(dim3(A.N / 512, A.M)), dim3(512), 0, st, A); }



__global__ void d3_gather_kernel(int* toks, int* verify_toks, const int* next8, int a) { pdl_wait(); pdl_trigger();
  if (threadIdx.x == 0) { for (int j = 0; j <= a; ++j) toks[j] = verify_toks[j]; const int f = next8[a]; toks[a + 1] = f; verify_toks[0] = f; }
}
void launch_d3_gather(int* toks, int* verify_toks, const int* next8, int a, cudaStream_t st) { launch_k(d3_gather_kernel, dim3(1), dim3(32), 0, st, toks, verify_toks, next8, a); }


__global__ void d3_set_from_kernel(int* dst, const int* src, int add) { pdl_wait(); pdl_trigger(); if (threadIdx.x == 0) *dst = *src + add; }
void launch_d3_set_from(int* dst, const int* src, int add, cudaStream_t st) { launch_k(d3_set_from_kernel, dim3(1), dim3(32), 0, st, dst, src, add); }


__global__ void d3_map_kernel(const int* idx, const int* ids, int* verify_slot, int* cur_tok) { pdl_wait(); pdl_trigger();
  if (threadIdx.x == 0) { const int t = ids[*idx]; *verify_slot = t; *cur_tok = t; }
}
void launch_d3_map(const int* idx, const int* ids, int* verify_slot, int* cur_tok, cudaStream_t st) { launch_k(d3_map_kernel, dim3(1), dim3(32), 0, st, idx, ids, verify_slot, cur_tok); }
} 
