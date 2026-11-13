// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
namespace yskn {
constexpr int L_BITS = 12; constexpr uint32_t L_MASK = 0xFFFu;


template <int K> struct Unit { static constexpr int WPS = 4 * K; uint32_t w[WPS]; };


template <int K> __device__ __forceinline__ uint32_t v2_state(const Unit<K>& u, int j) {
  const int e = K * (j + 1);
  if (e < L_BITS) return (u.w[0] & ((1u << e) - 1u)) << (L_BITS - e);
  const int start = e - L_BITS, word = start >> 5, sh = start & 31;
  if (sh + L_BITS <= 32) return (u.w[word] >> sh) & L_MASK;
  return __funnelshift_r(u.w[word], u.w[word + 1], sh) & L_MASK;      
}

template <int K> __device__ __forceinline__ uint32_t v2_state_carry(const Unit<K>& u, uint32_t tail, int j) {
  const int e = K * (j + 1); if (e >= L_BITS) return v2_state<K>(u, j);
  return (((u.w[0] & ((1u << e) - 1u)) << (L_BITS - e)) | (tail >> (32 - (L_BITS - e)))) & L_MASK;
}


template <int REP> struct Lut8 { const uint16_t* base; __device__ __forceinline__ uint32_t get(uint32_t s, int rep) const { return (uint32_t)base[s * REP + rep]; } };
template <int REP> __device__ __forceinline__ void load_lut8(uint16_t* smem, const uint16_t* __restrict__ g) {
  for (int i = threadIdx.x; i < 4096 * REP; i += blockDim.x) smem[i] = g[i / REP];
}


template <int REP> __device__ __forceinline__ void load_lut8_fast(uint16_t* smem, const uint16_t* __restrict__ g) {
  if (REP == 1) { for (int i = threadIdx.x; i < 512; i += blockDim.x) reinterpret_cast<uint4*>(smem)[i] = __ldg(reinterpret_cast<const uint4*>(g) + i); }
  else if (REP == 2) {
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(g) + i);
      auto d = [](uint32_t e) { return e | (e << 16); };
      reinterpret_cast<uint4*>(smem)[2 * i]     = make_uint4(d(v.x & 0xFFFFu), d(v.x >> 16), d(v.y & 0xFFFFu), d(v.y >> 16));
      reinterpret_cast<uint4*>(smem)[2 * i + 1] = make_uint4(d(v.z & 0xFFFFu), d(v.z >> 16), d(v.w & 0xFFFFu), d(v.w >> 16));
    }
  } else load_lut8<REP>(smem, g);
}


template <int K, int REP> __device__ __forceinline__ uint2 v2_bfrag_s8(const Unit<K>& u, int m, int g, const Lut8<REP>& lut) {
  const int rep = g & (REP - 1);
  const uint32_t a0 = lut.get(v2_state<K>(u, 16 * m + 2 * g), rep),     a1 = lut.get(v2_state<K>(u, 16 * m + 2 * g + 1), rep);
  const uint32_t b0 = lut.get(v2_state<K>(u, 16 * m + 8 + 2 * g), rep), b1 = lut.get(v2_state<K>(u, 16 * m + 8 + 2 * g + 1), rep);
  return make_uint2(a0 | (a1 << 16), b0 | (b1 << 16));                   
}




template <int K> struct UnitG { static constexpr int WPS = 4 * K; uint32_t w[WPS]; };
template <int K> __device__ __forceinline__ void unitg_from_ring(UnitG<K>& u, const unsigned char* stage, int j, int r, int g) {
  constexpr int WPS = UnitG<K>::WPS;
  const uint32_t* S = reinterpret_cast<const uint32_t*>(stage + (j * 8 + r) * 16 * K);
  const int off = 2 * g * K - 12, q = off >> 5, sh = off & 31;          
  uint32_t prev = (q >= 0) ? S[q] : 0u;
  #pragma unroll
  for (int w = 0; w < WPS; ++w) {
    const int i1 = w + q + 1; const uint32_t nxt = (i1 >= 0 && i1 < WPS) ? S[i1] : 0u;
    u.w[w] = __funnelshift_r(prev, nxt, sh); prev = nxt;
  }
}
template <int K> __device__ __forceinline__ uint32_t yg_state(const UnitG<K>& u, int e) {   
  const int word = e >> 5, sh = e & 31;
  if (sh + L_BITS <= 32) return (u.w[word] >> sh) & L_MASK;
  return __funnelshift_r(u.w[word], u.w[word + 1], sh) & L_MASK;
}
template <int K, int REP> __device__ __forceinline__ uint2 v2_bfrag_s8_g(const UnitG<K>& u, int m, int g, const Lut8<REP>& lut) {
  const int rep = g & (REP - 1);
  const uint32_t a0 = lut.get(yg_state<K>(u, K * (16 * m + 1)), rep), a1 = lut.get(yg_state<K>(u, K * (16 * m + 2)), rep);
  const uint32_t b0 = lut.get(yg_state<K>(u, K * (16 * m + 9)), rep), b1 = lut.get(yg_state<K>(u, K * (16 * m + 10)), rep);
  return make_uint2(a0 | (a1 << 16), b0 | (b1 << 16));
}





template <int K> struct UnitQ { static constexpr int NW = (32 * K + L_BITS + 31) / 32; uint32_t w[NW]; };
template <int K> __device__ __forceinline__ void unitq_from_ring(UnitQ<K>& u, const unsigned char* stage, int j, int r, int g) {
  constexpr int WPS = 4 * K, NW = UnitQ<K>::NW;
  const uint32_t* S = reinterpret_cast<const uint32_t*>(stage + (j * 8 + r) * 16 * K);
  const int off = 32 * g * K - L_BITS, q = off >> 5, sh = off & 31;      
  uint32_t prev = (q >= 0) ? S[q] : 0u;
  #pragma unroll
  for (int w = 0; w < NW; ++w) {
    const int i1 = w + q + 1; const uint32_t nxt = (i1 >= 0 && i1 < WPS) ? S[i1] : 0u;
    u.w[w] = __funnelshift_r(prev, nxt, sh); prev = nxt;
  }
}
template <int K> __device__ __forceinline__ uint32_t yq_state(const UnitQ<K>& u, int i) {     
  const int e = K * (i + 1), word = e >> 5, sh = e & 31;
  if (sh + L_BITS <= 32) return (u.w[word] >> sh) & L_MASK;
  return __funnelshift_r(u.w[word], u.w[word + 1], sh) & L_MASK;
}
template <int K, int REP> __device__ __forceinline__ uint2 v2_bfrag_s8_q(const UnitQ<K>& u, int m, const Lut8<REP>& lut, int rep) {
  const uint32_t a0 = lut.get(yq_state<K>(u, 4 * m), rep),     a1 = lut.get(yq_state<K>(u, 4 * m + 1), rep);
  const uint32_t b0 = lut.get(yq_state<K>(u, 4 * m + 2), rep), b1 = lut.get(yq_state<K>(u, 4 * m + 3), rep);
  return make_uint2(a0 | (a1 << 16), b0 | (b1 << 16));
}

template <int K> __device__ __forceinline__ uint2 v2_bfrag_band_q(const UnitQ<K>& u, int m, const uint16_t* r8) {
  const uint32_t a0 = r8[yq_state<K>(u, 4 * m) >> (L_BITS - K)],     a1 = r8[yq_state<K>(u, 4 * m + 1) >> (L_BITS - K)];
  const uint32_t b0 = r8[yq_state<K>(u, 4 * m + 2) >> (L_BITS - K)], b1 = r8[yq_state<K>(u, 4 * m + 3) >> (L_BITS - K)];
  return make_uint2(a0 | (a1 << 16), b0 | (b1 << 16));
}
__device__ __forceinline__ void mma_s8_16832(int32_t* d, const uint32_t* a, uint2 b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }


template <int K> __device__ __forceinline__ void unit_from_ring(Unit<K>& u, const unsigned char* stage, int j, int r) {
  const uint4* p = reinterpret_cast<const uint4*>(stage + (j * 8 + r) * 16 * K);
  #pragma unroll
  for (int c = 0; c < K; ++c) { const uint4 v = p[c]; u.w[4*c] = v.x; u.w[4*c+1] = v.y; u.w[4*c+2] = v.z; u.w[4*c+3] = v.w; }
}

__device__ __forceinline__ void issue_stage_nocommit(const uint4* tile_base, int K, int ks, int j0, int NT, unsigned char* stage, int lane) {
  const int chunks = NT * 8 * K;
  for (int i = lane; i < chunks; i += 32) {
    const int j = i / (8 * K), rem = i % (8 * K), r = rem / K, c = rem % K;
    cp_async16(stage + ((j * 8 + r) * 16 * K + c * 16), tile_base + (((size_t)ks * 8 + (j0 + j)) * K + c) * 8 + r);
  }
}
} 
