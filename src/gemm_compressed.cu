// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "v2_decode.cuh"
#include "kernels.hpp"
namespace yskn {
constexpr int GM = 32, G_NT = 2, G_STAGES = 2, LDA = 256 + 16, G_REP = 2;
__device__ __forceinline__ int worker_of_g(const TensorDev& t, int g) { return t.group_worker ? (int)t.group_worker[g] : (t.n_workers > 0 ? g % t.n_workers : 0); }
__device__ __forceinline__ float madd_rn(float acc, float a, float b) { return __fadd_rn(acc, __fmul_rn(a, b)); }

__device__ __forceinline__ float xsm_max_g(const float* __restrict__ al, int G) { float m = 0.f; for (int k = 0; k < G; ++k) m = fmaxf(m, al[k]); return m; }
__device__ __forceinline__ float xsm_rm_g(const float* __restrict__ al, int G) { const float m = xsm_max_g(al, G); return m > 0.f ? __fdiv_rn(16777216.f, m) : 0.f; }
__device__ __forceinline__ long long xsm_A_g(float alpha, float rm) { return (long long)__float2int_rn(__fmul_rn(alpha, rm)); }

template <int K, bool XSM>
__global__ void __launch_bounds__(128) gemm_kernel(GemmLaunch P) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint16_t* s_lut = reinterpret_cast<uint16_t*>(smem);
  unsigned char* ring = smem + 4096 * 2 * G_REP;
  int8_t* sA = reinterpret_cast<int8_t*>(ring + 4 * G_STAGES * P.stage_bytes);     
  const TensorDev& t = P.t; const int worker = blockIdx.z; if (worker >= t.n_workers) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, g = lane & 3;
  const int tile = blockIdx.x * 4 + warp, ntiles = (t.N + 63) / 64, m0 = blockIdx.y * GM, M = P.M;
  const int Kc = t.Kc, G = Kc / 512, KS = Kc / 256, N = t.N;
  load_lut8_fast<G_REP>(s_lut, t.lut8);
  const bool active = tile < ntiles;
  const uint4* tbase = active ? t.frag + (t.tile_u4off ? t.tile_u4off[tile] : (uint64_t)tile * KS * 8 * K * 8) : nullptr;   
  unsigned char* my_ring = ring + warp * G_STAGES * P.stage_bytes;
  const bool rr = t.group_worker == nullptr; const int nw = t.n_workers;                     
  int ng = 0; if (rr) ng = worker < G ? (G - worker + nw - 1) / nw : 0; else for (int gg = 0; gg < G; ++gg) ng += (worker_of_g(t, gg) == worker);
  if (ng == 0) return;
  auto half_ks = [&](int hi) -> int { const int want = hi >> 1; if (rr) return 2 * (worker + want * nw) + (hi & 1); int cnt = 0;
    for (int gg = 0; gg < G; ++gg) if (worker_of_g(t, gg) == worker) { if (cnt == want) return 2 * gg + (hi & 1); ++cnt; } return 0; };
  const int H = 2 * ng;
  auto issue_half = [&](int stage, int ks, int j0) {
    for (int i = threadIdx.x; i < GM * 16; i += 128) { const int tok = i >> 4, c = i & 15, gt = m0 + tok;     
      if (gt < M) cp_async16(sA + (stage * GM + tok) * LDA + c * 16, P.qx + (size_t)gt * Kc + ks * 256 + c * 16); }
    if (active) issue_stage_nocommit(tbase, K, ks, j0, G_NT, my_ring + stage * P.stage_bytes, lane);
    cp_commit(); };
  const Lut8<G_REP> lut{s_lut};
  float nrm[2][2] = {{0.f, 0.f}, {0.f, 0.f}};                     
  if (XSM) {
    for (int mt = 0; mt < 2; ++mt) for (int hf = 0; hf < 2; ++hf) { const int tok = m0 + mt * 16 + r + 8 * hf; if (tok < M) nrm[mt][hf] = xsm_rm_g(P.alpha + (size_t)tok * G, G); }
    if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x < GM && m0 + (int)threadIdx.x < M)
      P.us_out[m0 + threadIdx.x] = ldexpf(xsm_max_g(P.alpha + (size_t)(m0 + threadIdx.x) * G, G), -24);
  }
  __syncthreads();
  for (int pass = 0; pass < 8 / G_NT; ++pass) {
    const int j0 = pass * G_NT;
    float pacc[G_NT][2][4]; long long acc64[XSM ? G_NT : 1][2][4];
    #pragma unroll
    for (int j = 0; j < G_NT; ++j) for (int mt = 0; mt < 2; ++mt) { pacc[j][mt][0] = pacc[j][mt][1] = pacc[j][mt][2] = pacc[j][mt][3] = 0.f;
      if (XSM) acc64[j][mt][0] = acc64[j][mt][1] = acc64[j][mt][2] = acc64[j][mt][3] = 0; }
    issue_half(0, half_ks(0), j0);
    int32_t acc[G_NT][2][4];
    for (int hi = 0; hi < H; ++hi) {
      if (hi + 1 < H) { issue_half((hi + 1) & 1, half_ks(hi + 1), j0); cp_wait<1>(); } else cp_wait<0>();
      __syncthreads();
      const int ks = half_ks(hi), st = hi & 1;
      if ((hi & 1) == 0) {
        #pragma unroll
        for (int j = 0; j < G_NT; ++j) for (int mt = 0; mt < 2; ++mt) acc[j][mt][0] = acc[j][mt][1] = acc[j][mt][2] = acc[j][mt][3] = 0;
      }
      if (active) {
        UnitQ<K> u[G_NT];                                             
        #pragma unroll
        for (int j = 0; j < G_NT; ++j) unitq_from_ring<K>(u[j], my_ring + st * P.stage_bytes, j, r, g);
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
          uint2 bf[G_NT];
          #pragma unroll
          for (int j = 0; j < G_NT; ++j) bf[j] = v2_bfrag_s8_q<K, G_REP>(u[j], m, lut, g & (G_REP - 1));
          #pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            const int8_t* A0 = sA + (st * GM + mt * 16 + r) * LDA + 64 * g + 8 * m;
            const uint2 lo = *reinterpret_cast<const uint2*>(A0), hi8 = *reinterpret_cast<const uint2*>(A0 + 8 * LDA);
            uint32_t a[4] = {lo.x, hi8.x, lo.y, hi8.y};
            #pragma unroll
            for (int j = 0; j < G_NT; ++j) mma_s8_16832(acc[j][mt], a, bf[j]);
          }
        }
        if (hi & 1) { const int grp = ks >> 1;
          #pragma unroll
          for (int j = 0; j < G_NT; ++j) for (int mt = 0; mt < 2; ++mt)
            #pragma unroll
            for (int q = 0; q < 4; ++q) { const int tok = m0 + mt * 16 + ((q < 2) ? r : r + 8);
              if (tok < M) { if (XSM) acc64[j][mt][q] += xsm_A_g(P.alpha[(size_t)tok * G + grp], nrm[mt][q >> 1]) * (long long)acc[j][mt][q];
                             else pacc[j][mt][q] = madd_rn(pacc[j][mt][q], P.alpha[(size_t)tok * G + grp], (float)acc[j][mt][q]); } } }
      }
      __syncthreads();
    }
    if (active) {
      #pragma unroll
      for (int j = 0; j < G_NT; ++j) for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int q = 0; q < 4; ++q) { const int tok = m0 + mt * 16 + ((q < 2) ? r : r + 8), n = tile * 64 + (j0 + j) * 8 + 2 * g + (q & 1);
          if (tok < M && n < N) { if (XSM) atomicAdd(reinterpret_cast<unsigned long long*>(P.part) + (size_t)tok * N + n, (unsigned long long)acc64[j][mt][q]);
                                  else P.part[((size_t)worker * M + tok) * N + n] = pacc[j][mt][q]; } }
    }
  }
}
size_t gemm_smem_bytes(const GemmLaunch& P) { return 4096 * 2 * G_REP + (size_t)4 * G_STAGES * P.stage_bytes + (size_t)2 * GM * LDA + 64; }
void launch_gemm(const GemmLaunch& P, cudaStream_t st) {
  const size_t sm = gemm_smem_bytes(P); const dim3 grid((P.t.N + 255) / 256, (P.M + GM - 1) / GM, P.t.n_workers);
  switch (P.t.k_default) {
    case 4: if (P.xsum) { cudaFuncSetAttribute(gemm_kernel<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<4, true><<<grid, 128, sm, st>>>(P); } else { cudaFuncSetAttribute(gemm_kernel<4, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<4, false><<<grid, 128, sm, st>>>(P); } break;
    case 5: if (P.xsum) { cudaFuncSetAttribute(gemm_kernel<5, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<5, true><<<grid, 128, sm, st>>>(P); } else { cudaFuncSetAttribute(gemm_kernel<5, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<5, false><<<grid, 128, sm, st>>>(P); } break;
    case 6: if (P.xsum) { cudaFuncSetAttribute(gemm_kernel<6, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<6, true><<<grid, 128, sm, st>>>(P); } else { cudaFuncSetAttribute(gemm_kernel<6, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<6, false><<<grid, 128, sm, st>>>(P); } break;
    case 7: if (P.xsum) { cudaFuncSetAttribute(gemm_kernel<7, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<7, true><<<grid, 128, sm, st>>>(P); } else { cudaFuncSetAttribute(gemm_kernel<7, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<7, false><<<grid, 128, sm, st>>>(P); } break;
    default: if (P.xsum) { cudaFuncSetAttribute(gemm_kernel<8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<8, true><<<grid, 128, sm, st>>>(P); }
             else { cudaFuncSetAttribute(gemm_kernel<8, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); gemm_kernel<8, false><<<grid, 128, sm, st>>>(P); } break;
  }
}
} 
