// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors


#include <cuda_runtime.h>
#include <cstdint>
#include "yskn_abi.h"
#include "pm_math.cuh"
#include "q35.hpp"
namespace yskn {
namespace {
__global__ void pm_f32_kernel(int fn, unsigned base, unsigned* out, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const float x = __uint_as_float(base + i); float y;
  switch (fn) {
    case 0: y = pm_expf(x); break;      case 1: y = pm_logf(x); break;    case 2: y = pm_log1p_pos(x); break;
    case 3: y = pm_silu(x); break;      case 4: y = pm_sigmoid(x); break; case 5: y = pm_softplus(x); break;
    default: y = pm_rsqrt(x); }
  out[i] = __float_as_uint(y);
}
__global__ void pm_f64_kernel(int fn, const double* in, double* out, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  out[i] = fn == 10 ? pm_exp_d(in[i]) : pm_log_d(in[i]);
}
}  
}  



extern "C"  int yskn_pm_test(int fn, unsigned long long base, const void* h_in, void* h_out, long long n) {
  using namespace yskn;
  if (n <= 0 || n > (1ll << 28)) return (int)cudaErrorInvalidValue;
  cudaError_t e = cudaSuccess;
  if (fn >= 0 && fn <= 6) {
    unsigned* d = nullptr; if ((e = cudaMalloc(&d, (size_t)n * 4)) != cudaSuccess) return (int)e;
    pm_f32_kernel<<<(unsigned)((n + 255) / 256), 256>>>(fn, (unsigned)base, d, (unsigned)n);
    e = cudaMemcpy(h_out, d, (size_t)n * 4, cudaMemcpyDeviceToHost); cudaFree(d);
  } else if (fn == 10 || fn == 11) {
    double* d = nullptr; if ((e = cudaMalloc(&d, (size_t)n * 16)) != cudaSuccess) return (int)e;
    cudaMemcpy(d, h_in, (size_t)n * 8, cudaMemcpyHostToDevice);
    pm_f64_kernel<<<(unsigned)((n + 255) / 256), 256>>>(fn, d, d + n, (unsigned)n);
    e = cudaMemcpy(h_out, d + n, (size_t)n * 8, cudaMemcpyDeviceToHost); cudaFree(d);
  } else if (fn == 20) {
    float2* d = nullptr; const size_t bytes = (size_t)n * (Q35_RD / 2) * sizeof(float2);
    if ((e = cudaMalloc(&d, bytes)) != cudaSuccess) return (int)e;
    launch_q35_rope_table(d, (int)n, nullptr, (int)base);
    e = cudaMemcpy(h_out, d, bytes, cudaMemcpyDeviceToHost); cudaFree(d);
  } else return (int)cudaErrorInvalidValue;
  if (e != cudaSuccess) return (int)e;
  e = cudaDeviceSynchronize(); if (e != cudaSuccess) return (int)e;
  return (int)cudaGetLastError();
}
