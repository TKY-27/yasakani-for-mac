// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include <cuda_runtime.h>
#include <cstdint>
#include "yskn_abi.h"

namespace {


__global__ void k_repack(const uint32_t* __restrict__ w, uint32_t* __restrict__ f, int rows, int seqs, int wps) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, total = (size_t)rows * seqs * wps; if (i >= total) return;
  const int word = (int)(i % wps); const size_t t2 = i / wps; const int seq = (int)(t2 % seqs), row = (int)(t2 / seqs);
  const size_t q = (size_t)seq * rows + row, tile = q / 32, lane = q % 32;
  const size_t src = ((tile * (wps / 4) + word / 4) * 32 + lane) * 4 + (word % 4);
  const int wt = row / 64, j = (row % 64) / 8, r = row % 8, c = word / 4;
  const size_t dst = (((((size_t)wt * seqs + seq) * 8 + j) * (wps / 4) + c) * 8 + r) * 4 + (word % 4);
  f[dst] = w[src];
}
} 

extern "C" int yskn_repack_warp_to_frag(const uint32_t* d_words, uint32_t* d_frag, int rows, int seqs, int wps, void* stream) {
  if (rows % 64 || wps % 4) return -1;
  const size_t total = (size_t)rows * seqs * wps;
  k_repack<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(d_words, d_frag, rows, seqs, wps);
  return (int)cudaGetLastError();
}
extern "C" int yskn_repack_warp_to_frag_host(const uint32_t* h_words, uint32_t* h_frag, int rows, int seqs, int wps) {
  const size_t bytes = (size_t)rows * seqs * wps * 4; uint32_t *dw = nullptr, *df = nullptr;
  cudaMalloc(&dw, bytes); cudaMalloc(&df, bytes); cudaMemcpy(dw, h_words, bytes, cudaMemcpyHostToDevice);
  const int rc = yskn_repack_warp_to_frag(dw, df, rows, seqs, wps, nullptr);
  cudaDeviceSynchronize(); cudaMemcpy(h_frag, df, bytes, cudaMemcpyDeviceToHost); cudaFree(dw); cudaFree(df); return rc;
}
