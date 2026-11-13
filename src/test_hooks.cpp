// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors


#include <cuda_runtime.h>
#include "yskn_abi.h"
#include "kernels.hpp"

extern "C"  int yskn_test_gemv(const void* d_frag, const uint16_t* d_lut8, int K, int N, int Kc,
                                         const int8_t* d_group_worker, int n_workers, const int8_t* d_qx, const float* d_alpha,
                                         float* d_part, int iters, float* ms_per_launch) {
  using namespace yskn;
  TensorDev t{};
  t.frag = reinterpret_cast<const uint4*>(d_frag); t.tile_u4off = nullptr; t.tile_k = nullptr; t.k_default = K;
  t.lut8 = d_lut8; t.tau = nullptr; t.N = N; t.Kc = Kc; t.group_worker = d_group_worker; t.n_workers = n_workers;
  GemvLaunch P{};
  const int ntiles = (N + 63) / 64;
  P.seg[0] = GemvSeg{t, d_part, 0}; P.n_seg = 1; P.n_ctas = (ntiles + GEMV_WARPS - 1) / GEMV_WARPS; P.max_workers = n_workers;
  P.qx = d_qx; P.alpha = d_alpha; P.fuse = 0; P.M = 1; P.stage_bytes = GEMV_NT_PASS * 8 * 16 * K;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  launch_gemv(P, nullptr);                                   
  cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) return (int)e;
  if (iters > 0) {
    cudaEventRecord(a, nullptr);
    for (int i = 0; i < iters; ++i) launch_gemv(P, nullptr);
    cudaEventRecord(b, nullptr); cudaEventSynchronize(b);
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b); if (ms_per_launch) *ms_per_launch = ms / (float)iters;
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  e = cudaDeviceSynchronize(); if (e != cudaSuccess) return (int)e;
  return (int)cudaGetLastError();
}


extern "C"  int yskn_test_gemv_m(const void* d_frag, const uint16_t* d_lut8, int K, int N, int Kc, int M,
                                           const int8_t* d_group_worker, int n_workers, const int8_t* d_qx, const float* d_alpha,
                                           float* d_part, int iters, float* ms_per_launch, const uint16_t* d_r8, int draft) {
  using namespace yskn;
  TensorDev t{};
  t.frag = reinterpret_cast<const uint4*>(d_frag); t.tile_u4off = nullptr; t.tile_k = nullptr; t.k_default = K;
  t.lut8 = d_lut8; t.tau = nullptr; t.N = N; t.Kc = Kc; t.group_worker = d_group_worker; t.n_workers = n_workers;
  GemvLaunch P{};
  const int ntiles = (N + 63) / 64;
  P.seg[0] = GemvSeg{t, d_part, 0}; P.n_seg = 1; P.n_ctas = (ntiles + GEMV_WARPS - 1) / GEMV_WARPS; P.max_workers = n_workers;
  P.qx = d_qx; P.alpha = d_alpha; P.fuse = 0; P.M = M; P.r8 = d_r8; P.draft = draft; P.stage_bytes = GEMV_NT_PASS * 8 * 16 * K;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  launch_gemv(P, nullptr);
  cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) return (int)e;
  if (iters > 0) {
    cudaEventRecord(a, nullptr);
    for (int i = 0; i < iters; ++i) launch_gemv(P, nullptr);
    cudaEventRecord(b, nullptr); cudaEventSynchronize(b);
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b); if (ms_per_launch) *ms_per_launch = ms / (float)iters;
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  e = cudaDeviceSynchronize(); if (e != cudaSuccess) return (int)e;
  return (int)cudaGetLastError();
}



extern "C"  int yskn_test_gemv_rot(const void* const* d_frags, int n_mats, const uint16_t* d_lut8, int K, int N, int Kc, int M,
                                             int n_workers, const int8_t* d_qx, const float* d_alpha, float* d_part, int reps, float* ms_per_launch) {
  using namespace yskn;
  auto launch = [&](int i) {
    TensorDev t{}; t.frag = reinterpret_cast<const uint4*>(d_frags[i % n_mats]); t.k_default = K; t.lut8 = d_lut8; t.N = N; t.Kc = Kc; t.n_workers = n_workers;
    GemvLaunch P{}; const int ntiles = (N + 63) / 64;
    P.seg[0] = GemvSeg{t, d_part, 0}; P.n_seg = 1; P.n_ctas = (ntiles + GEMV_WARPS - 1) / GEMV_WARPS; P.max_workers = n_workers;
    P.qx = d_qx; P.alpha = d_alpha; P.M = M; P.stage_bytes = GEMV_NT_PASS * 8 * 16 * K;
    launch_gemv(P, nullptr); };
  for (int i = 0; i < 2 * n_mats; ++i) launch(i);
  cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) return (int)e;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, nullptr); for (int i = 0; i < reps; ++i) launch(i); cudaEventRecord(b, nullptr); cudaEventSynchronize(b);
  float ms = 0.f; cudaEventElapsedTime(&ms, a, b); if (ms_per_launch) *ms_per_launch = ms / (float)reps;
  cudaEventDestroy(a); cudaEventDestroy(b);
  e = cudaDeviceSynchronize(); if (e != cudaSuccess) return (int)e;
  return (int)cudaGetLastError();
}
