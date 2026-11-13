// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include "q35.hpp"
#include "q35_atxpk.hpp"
#include <cmath>
#include <cstdlib>
namespace yskn {



















namespace {
constexpr int PK_KT = 32, PK_LD = Q35_HD + 8, PK_ROW = 128, PK_REC = 2048;   
constexpr float PK_LOG2E = 1.4426950408889634f;
__device__ __forceinline__ void pk_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void pk_ldm4(unsigned* r, const void* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void pk_ldm4t(unsigned* r, const void* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void pk_cp16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void pk_cp4(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void pk_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void pk_waitn() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ unsigned pk_pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<const unsigned*>(&v);
}
__device__ __forceinline__ void pk_s8x4_bf16(unsigned w, unsigned& b01, unsigned& b23) {          
  const unsigned x = w ^ 0x80808080u;
  const float f0 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7440)) - 8388736.f, f1 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7441)) - 8388736.f;
  const float f2 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7442)) - 8388736.f, f3 = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7443)) - 8388736.f;
  b01 = __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632); b23 = __byte_perm(__float_as_uint(f2), __float_as_uint(f3), 0x7632);
}
__device__ __forceinline__ float pk_bf_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float pk_bf_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float pk_tabf(const uint16_t* tab, unsigned st) { return __uint_as_float((unsigned)tab[st] << 16); }


template <int KB> __device__ __forceinline__ unsigned pk_state_g(const unsigned* rw, int e) {
  const unsigned* const cw = rw + (e >> 6) * (2 * KB);
  const int E = ((e & 63) + 1) * KB, j = (E - 1) >> 5, r = 31 - ((E - 1) & 31);
  const unsigned B = cw[j], A = j ? cw[j - 1] : 0u;
  return __funnelshift_r(B, A, (unsigned)r) & 255u;
}
__device__ __forceinline__ unsigned pk_lk2(const uint16_t* tab, unsigned s0, unsigned s1) { return __byte_perm((unsigned)tab[s0], (unsigned)tab[s1], 0x5410); }   

__device__ __forceinline__ void pk_conv16_k4(const unsigned* rw, int ch, const uint16_t* tab, unsigned* b) {
  const uint2 w = *reinterpret_cast<const uint2*>(rw + 2 * ch);
  const unsigned pv = (ch & 3) ? (rw[2 * ch - 1] & 15u) : 0u;
  unsigned s[16];
  s[0] = (pv << 4) | (w.x >> 28);
  #pragma unroll
  for (int i = 1; i < 8; ++i) s[i] = (w.x >> (28 - 4 * i)) & 255u;
  s[8] = ((w.x & 15u) << 4) | (w.y >> 28);
  #pragma unroll
  for (int i = 1; i < 8; ++i) s[8 + i] = (w.y >> (28 - 4 * i)) & 255u;
  #pragma unroll
  for (int j = 0; j < 8; ++j) b[j] = pk_lk2(tab, s[2 * j], s[2 * j + 1]);
}


template <int HALF> __device__ __forceinline__ void pk_conv16_k5(const unsigned* rw, int chp, const uint16_t* tab, unsigned* b) {
  unsigned wv[6];
  if (HALF == 0) { wv[0] = (chp & 1) ? rw[5 * chp - 1] : 0u; wv[1] = rw[5 * chp]; wv[2] = rw[5 * chp + 1]; wv[3] = rw[5 * chp + 2]; wv[4] = 0u; wv[5] = 0u; }
  else           { wv[0] = 0u; wv[1] = 0u; wv[2] = rw[5 * chp + 1]; wv[3] = rw[5 * chp + 2]; wv[4] = rw[5 * chp + 3]; wv[5] = rw[5 * chp + 4]; }
  unsigned s[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    constexpr int Z = 31;
    const int Em1 = Z + 5 * (16 * HALF + i + 1), j = Em1 >> 5;
    s[i] = __funnelshift_r(wv[j], wv[j - 1], (unsigned)(31 - (Em1 & 31))) & 255u;
  }
  #pragma unroll
  for (int j = 0; j < 8; ++j) b[j] = pk_lk2(tab, s[2 * j], s[2 * j + 1]);
}
}  

struct Q35AtxPkArgs { Q35AttnArgs a; const uint8_t* wk; const uint8_t* kp; const uint8_t* vp; const float* kps; const float* vps; const unsigned char* rec; int layer; };


template <bool KPK, bool VPK, int NS>
__global__ void __launch_bounds__(192, 1) q35_atxpk_kernel(Q35AtxPkArgs U) {
  constexpr int NW = 6, TPC = 16, NST = NS;
  static_assert(NS >= 2 && NS <= 3, "stages: 4 stages need 102,400 B of shared memory, more than the 99 KB (101,376 B) a CTA may use on sm_120");
  const Q35AttnArgs& A = U.a;
  constexpr int G = Q35_NH / Q35_KVH, D = Q35_HD;
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0;                                   
  
  uint16_t* const kt_b = usm;
  int8_t* const st8 = reinterpret_cast<int8_t*>(usm + 2 * PK_KT * PK_LD);
  float* const ssc = reinterpret_cast<float*>(st8 + NST * 2 * PK_KT * D);
  uint16_t* const tabs = reinterpret_cast<uint16_t*>(ssc + NST * 2 * PK_KT);
  float* const muv = reinterpret_cast<float*>(tabs + 2 * 256);
  if (KPK || VPK) {                                                  
    const unsigned char* const rk = U.rec + (size_t)((U.layer * Q35_KVH + kvh) * 2) * PK_REC;      
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
      if (KPK) reinterpret_cast<uint4*>(tabs)[i] = reinterpret_cast<const uint4*>(rk)[i];
      if (VPK) reinterpret_cast<uint4*>(tabs + 256)[i] = reinterpret_cast<const uint4*>(rk + PK_REC)[i];
    }
    if (VPK) for (int i = threadIdx.x; i < 64; i += blockDim.x) reinterpret_cast<uint4*>(muv)[i] = reinterpret_cast<const uint4*>(rk + PK_REC + 512)[i];
  }
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = *A.d_pos, M = A.M;
  
  int tk[2], hd[2], lo[2], ph[2], own[2], len[2]; bool self_[2], vrow[2];
  bool need = false;
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int i = 16 * warp + gi + 8 * h; tk[h] = t0 + i / G; hd[h] = kvh * G + i % G; vrow[h] = i < G * TPC && tk[h] < M;
    const int dp = vrow[h] ? (U.wk ? (int)U.wk[tk[h]] : tk[h]) : -1;
    len[h] = vrow[h] ? base + dp + 1 : 0; own[h] = base + tk[h];
    const int S = (A.splits_short > 0 && len[h] < A.long_len) ? A.splits_short : A.splits;
    need |= vrow[h] && split < S;
    const int per = (len[h] + S - 1) / S;
    lo[h] = split * per; const int hi = min(len[h], lo[h] + per);
    ph[h] = max(lo[h], min(hi, len[h] - 1));
    self_[h] = vrow[h] && lo[h] <= len[h] - 1 && len[h] - 1 < lo[h] + per;
  }
  if (!__syncthreads_or(need)) { if (early) { pdl_wait(); pdl_trigger(); } return; }
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; ++h) if (vrow[h] && ph[h] > lo[h]) { atomicMin(&s_lo, lo[h]); atomicMax(&s_hi, ph[h]); }
  __syncthreads();
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / PK_KT) * PK_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + PK_KT - 1) / PK_KT : 0;
  const uint8_t* const kpg = KPK ? U.kp + (size_t)kvh * A.max_ctx * PK_ROW : nullptr;
  const uint8_t* const vpg = VPK ? U.vp + (size_t)kvh * A.max_ctx * PK_ROW : nullptr;
  const int8_t* const kg8 = KPK ? nullptr : A.k8 + (size_t)kvh * A.max_ctx * D;
  const int8_t* const vg8 = VPK ? nullptr : A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = (KPK ? U.kps : A.ks) + (size_t)kvh * A.max_ctx;
  const float* const vsg = (VPK ? U.vps : A.vs) + (size_t)kvh * A.max_ctx;
  auto load_tile = [&](int t, int stage, int part) {                 
    const int k0 = tbeg + t * PK_KT;
    auto mine = [&](int kidx) { return part == 0 || (part == 1 ? kidx < base : kidx >= base); };
    int8_t* const sk = st8 + (size_t)stage * 2 * PK_KT * D;
    for (int c = threadIdx.x; c < 2 * PK_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (PK_KT * (D / 16)), rem = c - kv * (PK_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      const bool pk = kv ? VPK : KPK;
      if (pk && ch >= PK_ROW / 16) continue;                          
      int8_t* const dst = sk + (size_t)kv * PK_KT * D + key * D + ch * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) {
        if (mine(kidx)) {
          const void* src;
          if (kv) { if constexpr (VPK) src = vpg + (size_t)kidx * PK_ROW + ch * 16; else src = vg8 + (size_t)kidx * D + ch * 16; }
          else    { if constexpr (KPK) src = kpg + (size_t)kidx * PK_ROW + ch * 16; else src = kg8 + (size_t)kidx * D + ch * 16; }
          pk_cp16(dst, src);
        }
      } else if (part != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = threadIdx.x; c < 2 * PK_KT; c += blockDim.x) {        
      const int kv = c / PK_KT, key = c % PK_KT, kidx = k0 + key;
      float* const dst = ssc + (stage * 2 + kv) * PK_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) pk_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (part != 2) *dst = 0.f;
    }
  };
  #pragma unroll
  for (int p = 0; p < NST - 1; ++p) { if (p < ntiles) load_tile(p, p, early ? 1 : 0); pk_commit(); }
  if (early) { pdl_wait(); pdl_trigger();
    #pragma unroll
    for (int p = 0; p < NST - 1; ++p) { if (p < ntiles) load_tile(p, p, 2); pk_commit(); } }
  unsigned qa[16][4];
  {
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[0] ? tk[0] : 0) * Q35_NH + hd[0]) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow[1] ? tk[1] : 0) * Q35_NH + hd[1]) * D);
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = vrow[0] ? q0[ks * 8 + tig] : 0u; qa[ks][1] = vrow[1] ? q1[ks * 8 + tig] : 0u;
      qa[ks][2] = vrow[0] ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = vrow[1] ? q1[ks * 8 + 4 + tig] : 0u;
    }
  }
  const bool busy = __any_sync(0xffffffffu, vrow[0] || vrow[1]);
  float o[32][4];
  #pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  for (int t = 0; t < ntiles; ++t) {
    if (t + NST - 1 < ntiles) load_tile(t + NST - 1, (t + NST - 1) % NST, 0);
    pk_commit(); pk_waitn<NST - 1>();
    __syncthreads();
    const int stg = t % NST;
    const int8_t* const sk = st8 + (size_t)stg * 2 * PK_KT * D;
    for (int c = threadIdx.x; c < 2 * PK_KT * (D / 16); c += blockDim.x) {   
      const int kv = c / (PK_KT * (D / 16)), rem = c - kv * (PK_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      const int8_t* const rp = sk + (size_t)kv * PK_KT * D + key * D;
      unsigned b[8];
      if (kv ? VPK : KPK) {
        const unsigned* const wp = reinterpret_cast<const unsigned*>(rp);
        const unsigned w0 = wp[2 * ch], w1 = wp[2 * ch + 1], pv = (ch & 3) ? (wp[2 * ch - 1] & 15u) : 0u;   
        const uint16_t* const tab = tabs + kv * 256;
        unsigned s[16];
        s[0] = (pv << 4) | (w0 >> 28);
        #pragma unroll
        for (int i = 1; i < 8; ++i) s[i] = (w0 >> (28 - 4 * i)) & 255u;
        s[8] = ((w0 & 15u) << 4) | (w1 >> 28);
        #pragma unroll
        for (int i = 1; i < 8; ++i) s[8 + i] = (w1 >> (28 - 4 * i)) & 255u;
        #pragma unroll
        for (int j = 0; j < 8; ++j) b[j] = (unsigned)tab[s[2 * j]] | ((unsigned)tab[s[2 * j + 1]] << 16);
      } else {
        const uint4 u = *reinterpret_cast<const uint4*>(rp + ch * 16);
        pk_s8x4_bf16(u.x, b[0], b[1]); pk_s8x4_bf16(u.y, b[2], b[3]); pk_s8x4_bf16(u.z, b[4], b[5]); pk_s8x4_bf16(u.w, b[6], b[7]);
      }
      uint16_t* const dst = kt_b + (size_t)kv * PK_KT * PK_LD + key * PK_LD + ch * 16;
      *reinterpret_cast<uint4*>(dst) = make_uint4(b[0], b[1], b[2], b[3]); *reinterpret_cast<uint4*>(dst + 8) = make_uint4(b[4], b[5], b[6], b[7]);
    }
    __syncthreads();
    const uint16_t* const ks = kt_b; const uint16_t* const vs = kt_b + PK_KT * PK_LD;
    const float* const ksc = ssc + (stg * 2 + 0) * PK_KT; const float* const vsc = ssc + (stg * 2 + 1) * PK_KT;
    if (busy) {
      const int kt0 = tbeg + t * PK_KT;
      float s[4][4];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        #pragma unroll
        for (int kp = 0; kp < 8; ++kp) {
          unsigned b[4];
          pk_ldm4(b, ks + (8 * j + (lane & 7)) * PK_LD + 32 * kp + 8 * (lane >> 3));
          pk_mma(s[j], qa[2 * kp], b[0], b[1]);
          pk_mma(s[j], qa[2 * kp + 1], b[2], b[3]);
        }
      }
      float x0 = m0, x1 = m1;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk, h = c >> 1;
          const float v = s[j][c] * (ksc[kk] * 0.0625f);
          s[j][c] = (key >= lo[h] && key < ph[h]) ? v : -INFINITY;
        }
        x0 = fmaxf(x0, fmaxf(s[j][0], s[j][1])); x1 = fmaxf(x1, fmaxf(s[j][2], s[j][3]));
      }
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      const float c0 = exp2f((m0 - x0) * PK_LOG2E), c1 = exp2f((m1 - x1) * PK_LOG2E);
      m0 = x0; m1 = x1;
      const float y0 = x0 * PK_LOG2E, y1 = x1 * PK_LOG2E;
      float r0s = 0.f, r1s = 0.f;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        s[j][0] = exp2f(s[j][0] * PK_LOG2E - y0); s[j][1] = exp2f(s[j][1] * PK_LOG2E - y0);
        s[j][2] = exp2f(s[j][2] * PK_LOG2E - y1); s[j][3] = exp2f(s[j][3] * PK_LOG2E - y1);
        r0s += s[j][0] + s[j][1]; r1s += s[j][2] + s[j][3];
      }
      r0s += __shfl_xor_sync(0xffffffffu, r0s, 1); r0s += __shfl_xor_sync(0xffffffffu, r0s, 2);
      r1s += __shfl_xor_sync(0xffffffffu, r1s, 1); r1s += __shfl_xor_sync(0xffffffffu, r1s, 2);
      l0 = l0 * c0 + r0s; l1 = l1 * c1 + r1s;
      if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
        #pragma unroll
        for (int n = 0; n < 32; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
      }
      #pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const float va = vsc[16 * kk + 2 * tig], vb = vsc[16 * kk + 2 * tig + 1], vc2 = vsc[16 * kk + 8 + 2 * tig], vd2 = vsc[16 * kk + 8 + 2 * tig + 1];
        unsigned pa[4];
        pa[0] = pk_pack(s[2 * kk][0] * va, s[2 * kk][1] * vb); pa[1] = pk_pack(s[2 * kk][2] * va, s[2 * kk][3] * vb);
        pa[2] = pk_pack(s[2 * kk + 1][0] * vc2, s[2 * kk + 1][1] * vd2); pa[3] = pk_pack(s[2 * kk + 1][2] * vc2, s[2 * kk + 1][3] * vd2);
        #pragma unroll
        for (int np = 0; np < 16; ++np) {
          unsigned b[4];
          pk_ldm4t(b, vs + (16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * PK_LD + 16 * np + (lane >> 4) * 8);
          pk_mma(o[2 * np], pa, b[0], b[1]);
          pk_mma(o[2 * np + 1], pa, b[2], b[3]);
        }
      }
    }
    __syncthreads();                                                  
  }
  
  if (busy) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const bool sf = self_[h];                                       
      float part = 0.f;
      if (sf) {
        const size_t row = (size_t)own[h];
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
          const unsigned qlo = qa[ks][h], qhi = qa[ks][2 + h];        
          float k0, k1, k2, k3;
          if constexpr (KPK) {
            const unsigned* const kw = reinterpret_cast<const unsigned*>(kpg + row * PK_ROW);
            const unsigned wa = kw[2 * ks], wb = kw[2 * ks + 1];
            const int sh = 28 - 8 * tig;                              
            const unsigned a0 = (wa >> sh) & 15u, a1 = (wa >> (sh - 4)) & 15u, b0 = (wb >> sh) & 15u, b1 = (wb >> (sh - 4)) & 15u;
            const unsigned ap = tig ? ((wa >> (sh + 4)) & 15u) : ((ks & 3) ? (kw[2 * ks - 1] & 15u) : 0u);
            const unsigned bp = tig ? ((wb >> (sh + 4)) & 15u) : (wa & 15u);
            k0 = pk_tabf(tabs, (ap << 4) | a0); k1 = pk_tabf(tabs, (a0 << 4) | a1); k2 = pk_tabf(tabs, (bp << 4) | b0); k3 = pk_tabf(tabs, (b0 << 4) | b1);
          } else {
            const int8_t* kr = kg8 + row * D + 16 * ks + 2 * tig;
            k0 = (float)kr[0]; k1 = (float)kr[1]; k2 = (float)kr[8]; k3 = (float)kr[9];
          }
          part = __fadd_rn(part, __fmul_rn(pk_bf_lo(qlo), k0)); part = __fadd_rn(part, __fmul_rn(pk_bf_hi(qlo), k1));
          part = __fadd_rn(part, __fmul_rn(pk_bf_lo(qhi), k2)); part = __fadd_rn(part, __fmul_rn(pk_bf_hi(qhi), k3));
        }
      }
      part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 1));
      part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 2));
      if (sf) {
        const float sc = part * (ksg[own[h]] * 0.0625f);
        float& m = h ? m1 : m0; float& l = h ? l1 : l0;
        const float mn = fmaxf(m, sc), c = exp2f((m - mn) * PK_LOG2E), p = exp2f((sc - mn) * PK_LOG2E);
        m = mn; l = l * c + p;
        const float pv = p * vsg[own[h]];
        #pragma unroll
        for (int n = 0; n < 32; ++n) {
          float v0, v1;
          if constexpr (VPK) {
            const unsigned* const vw = reinterpret_cast<const unsigned*>(vpg + (size_t)own[h] * PK_ROW);
            const unsigned w = vw[n];
            const int sh = 28 - 8 * tig;
            const unsigned c0f = (w >> sh) & 15u, c1f = (w >> (sh - 4)) & 15u;
            const unsigned cp = tig ? ((w >> (sh + 4)) & 15u) : ((n & 7) ? (vw[n - 1] & 15u) : 0u);
            v0 = pk_tabf(tabs + 256, (cp << 4) | c0f); v1 = pk_tabf(tabs + 256, (c0f << 4) | c1f);
          } else {
            const int8_t* vr = vg8 + (size_t)own[h] * D + 8 * n + 2 * tig; v0 = (float)vr[0]; v1 = (float)vr[1];
          }
          o[n][2 * h] = o[n][2 * h] * c + pv * v0; o[n][2 * h + 1] = o[n][2 * h + 1] * c + pv * v1;
        }
      }
    }
  }
  if constexpr (VPK) {                                                
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (!vrow[h]) continue;
      const float lh = h ? l1 : l0;
      #pragma unroll
      for (int n = 0; n < 32; ++n) { o[n][2 * h] += lh * muv[8 * n + 2 * tig]; o[n][2 * h + 1] += lh * muv[8 * n + 2 * tig + 1]; }
    }
  }
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!vrow[h]) continue;
    const size_t pidx = ((size_t)tk[h] * A.splits + split) * Q35_NH + hd[h];
    if (tig == 0) { A.pm[pidx] = h ? m1 : m0; A.pl[pidx] = h ? l1 : l0; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < 32; ++n) *reinterpret_cast<float2*>(pa + 8 * n) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}


template <int KB, bool KPK, bool VPK>
__host__ __device__ constexpr size_t pkt_smem() {                                         
  return (size_t)2 * 2 * PK_KT * Q35_HD * 2 + (size_t)PK_KT * (KPK ? 32 * KB : Q35_HD) + (size_t)PK_KT * (VPK ? 32 * KB : Q35_HD) + (size_t)2 * PK_KT * 4 + (KPK ? 512 : 0) + (VPK ? 512 : 0);
}
template <int RT> __host__ __device__ constexpr size_t pkt_static() { return (size_t)RT * 3072 + (size_t)2 * 2 * PK_KT * 4 + 8; }   



struct PkRow { int tk, hd, lo, ph, own; bool v, self_, act; };
template <int TPC, bool VOL> __device__ __forceinline__ PkRow pk_row(const Q35AtxPkArgs& U, int base, int t0, int tile, int gi, int h, int kvh, int split) {
  const Q35AttnArgs& A = U.a;
  constexpr int G = Q35_NH / Q35_KVH;
  PkRow r;
  const int i = 16 * tile + gi + 8 * h; r.tk = t0 + i / G; r.hd = kvh * G + i % G; r.v = i < G * TPC && r.tk < A.M;
  int dp = -1;
  if (r.v) {
    if (U.wk == nullptr) dp = r.tk;
    else if (VOL) dp = (int)*reinterpret_cast<const volatile uint8_t*>(U.wk + r.tk);
    else dp = (int)U.wk[r.tk];
  }
  const int len = r.v ? base + dp + 1 : 0; r.own = base + r.tk;
  const int S = (A.splits_short > 0 && len < A.long_len) ? A.splits_short : A.splits;
  r.act = r.v && split < S;
  const int per = (len + S - 1) / S;
  r.lo = split * per; const int hi = min(len, r.lo + per);
  r.ph = max(r.lo, min(hi, len - 1));
  r.self_ = r.v && r.lo <= len - 1 && len - 1 < r.lo + per;
  return r;
}

template <int KB, bool KPK, bool VPK, int RT, int WPT>
__global__ void __launch_bounds__(RT * WPT * 32, 1) q35_atxpk_vsp_kernel(Q35AtxPkArgs U) {
  constexpr int TPC = (16 * RT) / 6, JW = 4 / WPT, NB = 32 / WPT, TL = 256, PB = 32 * KB;       
  static_assert(KB == 4 || KB == 5, "CDQ k = 4 or 5");
  static_assert(pkt_smem<KB, KPK, VPK>() + pkt_static<RT>() <= 101376, "shared memory: more than the 99 KB (101,376 B) a CTA may use on sm_120");
  const Q35AttnArgs& A = U.a;
  constexpr int G = Q35_NH / Q35_KVH, D = Q35_HD;
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  __shared__ float2 x_mx[RT][4][32], x_sm[RT][4][32];
  __shared__ uint2 x_pf[RT][4][32];
  __shared__ float sc2[2][2][PK_KT];                                  
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = warp / WPT, part = warp % WPT, gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0;
  
  uint16_t* const kt_b = usm;
  uint8_t* const stg = reinterpret_cast<uint8_t*>(usm + 2 * 2 * PK_KT * TL);
  constexpr int KP = KPK ? PB : D, VP = VPK ? PB : D, KREG = PK_KT * KP, VREG = PK_KT * VP;
  float* const ssc = reinterpret_cast<float*>(stg + KREG + VREG);
  uint16_t* const tabk = reinterpret_cast<uint16_t*>(ssc + 2 * PK_KT);
  uint16_t* const tabv = tabk + (KPK ? 256 : 0);
  const unsigned char* const rk = U.rec + (size_t)((U.layer * Q35_KVH + kvh) * 2) * PK_REC;      
  for (int i = threadIdx.x; i < 32; i += blockDim.x) {                
    if (KPK) reinterpret_cast<uint4*>(tabk)[i] = reinterpret_cast<const uint4*>(rk)[i];
    if (VPK) reinterpret_cast<uint4*>(tabv)[i] = reinterpret_cast<const uint4*>(rk + PK_REC)[i];
  }
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = *A.d_pos;
  const PkRow r0 = pk_row<TPC, false>(U, base, t0, tile, gi, 0, kvh, split), r1 = pk_row<TPC, false>(U, base, t0, tile, gi, 1, kvh, split);
  const int lo0 = r0.lo, lo1 = r1.lo, ph0 = r0.ph, ph1 = r1.ph;
  const bool vrow0 = r0.v, vrow1 = r1.v;
  if (!__syncthreads_or(r0.act || r1.act)) { if (early) { pdl_wait(); pdl_trigger(); } return; }
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  __syncthreads();
  if (vrow0 && ph0 > lo0) { atomicMin(&s_lo, lo0); atomicMax(&s_hi, ph0); }
  if (vrow1 && ph1 > lo1) { atomicMin(&s_lo, lo1); atomicMax(&s_hi, ph1); }
  __syncthreads();
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / PK_KT) * PK_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + PK_KT - 1) / PK_KT : 0;
  const uint8_t* const kpg = KPK ? U.kp + (size_t)kvh * A.max_ctx * PB : nullptr;
  const uint8_t* const vpg = VPK ? U.vp + (size_t)kvh * A.max_ctx * PB : nullptr;
  const int8_t* const kg8 = KPK ? nullptr : A.k8 + (size_t)kvh * A.max_ctx * D;
  const int8_t* const vg8 = VPK ? nullptr : A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = (KPK ? U.kps : A.ks) + (size_t)kvh * A.max_ctx;
  const float* const vsg = (VPK ? U.vps : A.vs) + (size_t)kvh * A.max_ctx;
  constexpr int GK = KPK ? 2 * KB : 16, GV = VPK ? 2 * KB : 16;      
  auto load_tile = [&](int t, int pt) {                               
    const int k0 = tbeg + t * PK_KT;
    auto mine = [&](int kidx) { return pt == 0 || (pt == 1 ? kidx < base : kidx >= base); };
    for (int c = threadIdx.x; c < PK_KT * (GK + GV); c += blockDim.x) {
      const int kv = c >= PK_KT * GK, cc = kv ? c - PK_KT * GK : c, ng = kv ? GV : GK;
      const int key = cc / ng, g = cc - key * ng;
      uint8_t* const dst = stg + (kv ? KREG : 0) + (size_t)key * (kv ? VP : KP) + g * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) {
        if (mine(kidx)) {
          const void* src;
          if (kv) { if constexpr (VPK) src = vpg + (size_t)kidx * PB + g * 16; else src = vg8 + (size_t)kidx * D + g * 16; }
          else    { if constexpr (KPK) src = kpg + (size_t)kidx * PB + g * 16; else src = kg8 + (size_t)kidx * D + g * 16; }
          pk_cp16(dst, src);
        }
      } else if (pt != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = threadIdx.x; c < 2 * PK_KT; c += blockDim.x) {
      const int kv = c / PK_KT, key = c % PK_KT, kidx = k0 + key;
      float* const dst = ssc + kv * PK_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) pk_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (pt != 2) *dst = 0.f;
    }
  };
  
  
  auto convert_own = [&](int b) {
    uint16_t* const kbuf = kt_b + (size_t)b * 2 * PK_KT * TL;
    for (int c = threadIdx.x; c < 2 * PK_KT * 16; c += blockDim.x) {
      const int kv = c >> 9, cc = c & 511;
      int key, ch; unsigned w[8];
      if (kv ? VPK : KPK) {
        const uint16_t* const tab = kv ? tabv : tabk;
        if constexpr (KB == 4) {
          key = cc >> 4; ch = cc & 15;
          pk_conv16_k4(reinterpret_cast<const unsigned*>(stg + (kv ? KREG : 0) + key * PB), ch, tab, w);
        } else {
          const int hf = cc >> 8, l = cc & 31;
          key = 4 * ((cc >> 5) & 7) + ((l >> 1) & 3);
          const int chp = 2 * ((l >> 3) & 3) + (l & 1);                 
          ch = 2 * chp + hf;
          const unsigned* const rw = reinterpret_cast<const unsigned*>(stg + (kv ? KREG : 0) + key * PB);
          if (hf == 0) pk_conv16_k5<0>(rw, chp, tab, w); else pk_conv16_k5<1>(rw, chp, tab, w);
        }
      } else {
        key = cc >> 4; ch = cc & 15;
        const uint4 u = *reinterpret_cast<const uint4*>(stg + (kv ? KREG : 0) + key * D + ch * 16);
        pk_s8x4_bf16(u.x, w[0], w[1]); pk_s8x4_bf16(u.y, w[2], w[3]); pk_s8x4_bf16(u.z, w[4], w[5]); pk_s8x4_bf16(u.w, w[6], w[7]);
      }
      uint16_t* const row = kbuf + (size_t)kv * PK_KT * TL + key * TL; const int sw = key & 7;
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch) ^ sw)) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch + 1) ^ sw)) = make_uint4(w[4], w[5], w[6], w[7]);
    }
    for (int c = threadIdx.x; c < 2 * PK_KT; c += blockDim.x) sc2[b][c >> 5][c & 31] = ssc[c];
  };
  if (0 < ntiles) load_tile(0, early ? 1 : 0);
  pk_commit();
  if (early) { pdl_wait(); pdl_trigger(); if (0 < ntiles) load_tile(0, 2); pk_commit(); }
  unsigned qa[16][4];
  {
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow0 ? r0.tk : 0) * Q35_NH + r0.hd) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow1 ? r1.tk : 0) * Q35_NH + r1.hd) * D);
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = vrow0 ? q0[ks * 8 + tig] : 0u; qa[ks][1] = vrow1 ? q1[ks * 8 + tig] : 0u;
      qa[ks][2] = vrow0 ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = vrow1 ? q1[ks * 8 + 4 + tig] : 0u;
    }
  }
  if (0 < ntiles) { pk_waitn<0>(); __syncthreads(); convert_own(0); }   
  const bool busy = __any_sync(0xffffffffu, vrow0 || vrow1);
  float o[NB][4];                                                     
  #pragma unroll
  for (int n = 0; n < NB; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float s[JW][4], c0 = 1.f, c1 = 1.f;                                 
  for (int t = 0; t <= ntiles; ++t) {
    __syncthreads();                                                  
    if (t + 1 < ntiles) load_tile(t + 1, 0);                          
    pk_commit();
    if (busy && t >= 1) {                                             
      float r0s = 0.f, r1s = 0.f;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) { const float2 q = x_sm[tile][jj][lane]; r0s += q.x; r1s += q.y; }
      r0s += __shfl_xor_sync(0xffffffffu, r0s, 1); r0s += __shfl_xor_sync(0xffffffffu, r0s, 2);
      r1s += __shfl_xor_sync(0xffffffffu, r1s, 1); r1s += __shfl_xor_sync(0xffffffffu, r1s, 2);
      l0 = l0 * c0 + r0s; l1 = l1 * c1 + r1s;
      if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
        #pragma unroll
        for (int n = 0; n < NB; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
      }
      const uint16_t* const vs = kt_b + (size_t)((t - 1) & 1) * 2 * PK_KT * TL + PK_KT * TL;
      #pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const uint2 pA = x_pf[tile][2 * kk][lane], pB = x_pf[tile][2 * kk + 1][lane];
        const unsigned pa[4] = {pA.x, pA.y, pB.x, pB.y};
        #pragma unroll
        for (int q = 0; q < NB / 2; ++q) {
          const int np = (NB / 2) * part + q;
          unsigned bb[4];
          const int r = 16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7), c = 2 * np + (lane >> 4);
          pk_ldm4t(bb, vs + r * TL + 8 * (c ^ (r & 7)));
          pk_mma(o[2 * q], pa, bb[0], bb[1]);
          pk_mma(o[2 * q + 1], pa, bb[2], bb[3]);
        }
      }
    }
    if (t == ntiles) break;                                           
    const int b = t & 1;
    if (busy) {                                                       
      const uint16_t* const ks = kt_b + (size_t)b * 2 * PK_KT * TL; const float* const ksc = sc2[b][0];
      const int kt0 = tbeg + t * PK_KT;
      #pragma unroll
      for (int w = 0; w < JW; ++w) {
        const int j = JW * part + w;
        s[w][0] = s[w][1] = s[w][2] = s[w][3] = 0.f;
        #pragma unroll
        for (int kp = 0; kp < 8; ++kp) {
          unsigned bb[4];
          const int r = 8 * j + (lane & 7), c = 4 * kp + (lane >> 3);
          pk_ldm4(bb, ks + r * TL + 8 * (c ^ (r & 7)));
          pk_mma(s[w], qa[2 * kp], bb[0], bb[1]);
          pk_mma(s[w], qa[2 * kp + 1], bb[2], bb[3]);
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk, h = c >> 1;
          const float v = s[w][c] * (ksc[kk] * 0.0625f);
          s[w][c] = (key >= (h ? lo1 : lo0) && key < (h ? ph1 : ph0)) ? v : -INFINITY;
        }
        x_mx[tile][j][lane] = make_float2(fmaxf(s[w][0], s[w][1]), fmaxf(s[w][2], s[w][3]));
      }
    }
    pk_waitn<0>();                                                    
    __syncthreads();                                                  
    if (busy) {                                                       
      const float* const vsc = sc2[b][1];
      float x0 = m0, x1 = m1;
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) { const float2 q = x_mx[tile][jj][lane]; x0 = fmaxf(x0, q.x); x1 = fmaxf(x1, q.y); }
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      c0 = exp2f((m0 - x0) * PK_LOG2E); c1 = exp2f((m1 - x1) * PK_LOG2E);
      m0 = x0; m1 = x1;
      const float y0 = x0 * PK_LOG2E, y1 = x1 * PK_LOG2E;
      #pragma unroll
      for (int w = 0; w < JW; ++w) {
        const int j = JW * part + w;
        s[w][0] = exp2f(s[w][0] * PK_LOG2E - y0); s[w][1] = exp2f(s[w][1] * PK_LOG2E - y0);
        s[w][2] = exp2f(s[w][2] * PK_LOG2E - y1); s[w][3] = exp2f(s[w][3] * PK_LOG2E - y1);
        x_sm[tile][j][lane] = make_float2(s[w][0] + s[w][1], s[w][2] + s[w][3]);
        const float va = vsc[8 * j + 2 * tig], vb = vsc[8 * j + 2 * tig + 1];
        x_pf[tile][j][lane] = make_uint2(pk_pack(s[w][0] * va, s[w][1] * vb), pk_pack(s[w][2] * va, s[w][3] * vb));
      }
    }
    if (t + 1 < ntiles) convert_own(b ^ 1);                           
  }
  const int base2 = *reinterpret_cast<const volatile int*>(A.d_pos);
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const PkRow r = pk_row<TPC, true>(U, base2, t0, tile, gi, h, kvh, split);
    float& m = h ? m1 : m0; float& l = h ? l1 : l0;
    if (busy) {                                                       
      const bool sf = r.self_;                                        
      float pt = 0.f;
      if (sf) {
        const size_t row = (size_t)r.own;
        #pragma unroll
        for (int kq = 0; kq < 16; ++kq) {
          const unsigned qlo = qa[kq][h], qhi = qa[kq][2 + h];        
          float k0, k1, k2, k3;
          if constexpr (KPK) {
            const unsigned* const kw = reinterpret_cast<const unsigned*>(kpg + row * PB);
            k0 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 2 * tig)); k1 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 2 * tig + 1));
            k2 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 8 + 2 * tig)); k3 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 9 + 2 * tig));
          } else {
            const int8_t* kr = kg8 + row * D + 16 * kq + 2 * tig;
            k0 = (float)kr[0]; k1 = (float)kr[1]; k2 = (float)kr[8]; k3 = (float)kr[9];
          }
          pt = __fadd_rn(pt, __fmul_rn(pk_bf_lo(qlo), k0)); pt = __fadd_rn(pt, __fmul_rn(pk_bf_hi(qlo), k1));
          pt = __fadd_rn(pt, __fmul_rn(pk_bf_lo(qhi), k2)); pt = __fadd_rn(pt, __fmul_rn(pk_bf_hi(qhi), k3));
        }
      }
      pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 1));
      pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 2));
      if (sf) {
        const float sc = pt * (ksg[r.own] * 0.0625f);
        const float mn = fmaxf(m, sc), c = exp2f((m - mn) * PK_LOG2E), p = exp2f((sc - mn) * PK_LOG2E);
        m = mn; l = l * c + p;
        const float pv = p * vsg[r.own];
        #pragma unroll
        for (int n = 0; n < NB; ++n) {
          const int e = 8 * (NB * part + n) + 2 * tig;
          float v0, v1;
          if constexpr (VPK) {
            const unsigned* const vw = reinterpret_cast<const unsigned*>(vpg + (size_t)r.own * PB);
            v0 = pk_tabf(tabv, pk_state_g<KB>(vw, e)); v1 = pk_tabf(tabv, pk_state_g<KB>(vw, e + 1));
          } else {
            const int8_t* vr = vg8 + (size_t)r.own * D + e; v0 = (float)vr[0]; v1 = (float)vr[1];
          }
          o[n][2 * h] = o[n][2 * h] * c + pv * v0; o[n][2 * h + 1] = o[n][2 * h + 1] * c + pv * v1;
        }
      }
    }
    if (!r.v) continue;
    if constexpr (VPK) {                                              
      const float* const muv = reinterpret_cast<const float*>(rk + PK_REC + 512);
      #pragma unroll
      for (int n = 0; n < NB; ++n) {
        const float2 mu = *reinterpret_cast<const float2*>(muv + 8 * (NB * part + n) + 2 * tig);
        o[n][2 * h] += l * mu.x; o[n][2 * h + 1] += l * mu.y;
      }
    }
    const size_t pidx = ((size_t)r.tk * A.splits + split) * Q35_NH + r.hd;
    if (tig == 0 && part == 0) { A.pm[pidx] = m; A.pl[pidx] = l; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < NB; ++n) *reinterpret_cast<float2*>(pa + 8 * (NB * part + n)) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}

#include "q35_atxpk3.cuh"
#include "q35_atxpk4.cuh"
constexpr size_t PK_SMEM(int ns) { return (size_t)2 * PK_KT * PK_LD * 2 + (size_t)ns * 2 * PK_KT * Q35_HD + (size_t)ns * 2 * PK_KT * 4 + 2 * 512 + 1024; }
static int pk_ns_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_NS"); const int n = e ? std::atoi(e) : 2; return n >= 2 && n <= 3 ? n : 2; }(); return v; }
static int pk_variant_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_PKV"); return (e && e[0]) ? std::atoi(e) : 3; }(); return v; }   
static int pk3_mmax_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_PK3_MMAX"); const int n = (e && e[0]) ? std::atoi(e) : 2; return n < 0 ? 0 : n; }(); return v; }
template <int NS> static void pk_launch(const Q35AtxPkArgs& U, const dim3& grid, cudaStream_t st) {
  if (U.kp && U.vp) launch_k(q35_atxpk_kernel<true, true, NS>, grid, dim3(192), PK_SMEM(NS), st, U);
  else if (U.kp) launch_k(q35_atxpk_kernel<true, false, NS>, grid, dim3(192), PK_SMEM(NS), st, U);
  else launch_k(q35_atxpk_kernel<false, true, NS>, grid, dim3(192), PK_SMEM(NS), st, U);
}
template <int KB, bool VPK> static void pk_vsp_set(cudaError_t& e) {
  auto set = [&](const void* f, size_t smem) { const cudaError_t r = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); if (r != cudaSuccess && e == cudaSuccess) e = r; };
  set((const void*)q35_atxpk_vsp_kernel<KB, true, VPK, 1, 4>, pkt_smem<KB, true, VPK>());
  set((const void*)q35_atxpk_vsp_kernel<KB, true, VPK, 3, 4>, pkt_smem<KB, true, VPK>());
  set((const void*)q35_atxpk_vsp_kernel<KB, true, VPK, 6, 2>, pkt_smem<KB, true, VPK>());
}
template <int KB, bool VPK> static void pk_vsp_go(const Q35AtxPkArgs& U, cudaStream_t st) {
  const int M = U.a.M, sp = U.a.splits;
  if (M <= 2) launch_k(q35_atxpk_vsp_kernel<KB, true, VPK, 1, 4>, dim3((M + 1) / 2, sp, Q35_KVH), dim3(128), pkt_smem<KB, true, VPK>(), st, U);
  else if (M <= 8) launch_k(q35_atxpk_vsp_kernel<KB, true, VPK, 3, 4>, dim3((M + 7) / 8, sp, Q35_KVH), dim3(384), pkt_smem<KB, true, VPK>(), st, U);
  else launch_k(q35_atxpk_vsp_kernel<KB, true, VPK, 6, 2>, dim3((M + 15) / 16, sp, Q35_KVH), dim3(384), pkt_smem<KB, true, VPK>(), st, U);
}
void launch_q35_atx_pk(const Q35AttnArgs& A, const uint8_t* wk, const Q35PkArgs& P, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    cudaError_t e = cudaSuccess;
    auto set = [&](const void* f, size_t smem) { const cudaError_t r = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); if (r != cudaSuccess && e == cudaSuccess) e = r; };
    set((const void*)q35_atxpk_kernel<true, true, 2>, PK_SMEM(2)); set((const void*)q35_atxpk_kernel<true, false, 2>, PK_SMEM(2)); set((const void*)q35_atxpk_kernel<false, true, 2>, PK_SMEM(2));
    set((const void*)q35_atxpk_kernel<true, true, 3>, PK_SMEM(3)); set((const void*)q35_atxpk_kernel<true, false, 3>, PK_SMEM(3)); set((const void*)q35_atxpk_kernel<false, true, 3>, PK_SMEM(3));
    pk_vsp_set<4, true>(e); pk_vsp_set<4, false>(e); pk_vsp_set<5, true>(e); pk_vsp_set<5, false>(e);
    pk3_set_all(e); pk4_set_all(e);
    if (e != cudaSuccess) throw std::runtime_error(std::string("launch_q35_atx_pk: cudaFuncSetAttribute failed: ") + cudaGetErrorString(e));
    attr = true; }
  if (A.kv_fp16) throw std::runtime_error("launch_q35_atx_pk: FP16 KV");
  if (!P.rec || (!P.kp && !P.vp)) throw std::runtime_error("launch_q35_atx_pk: no packed stream");
  if ((P.kp ? !P.kps : !A.k8) || (P.vp ? !P.vps : !A.v8)) throw std::runtime_error("launch_q35_atx_pk: missing scales / INT8 container");
  if (P.kb != 4 && P.kb != 5) throw std::runtime_error("launch_q35_atx_pk: kb must be 4 or 5");
  const Q35AtxPkArgs U{A, wk, P.kp, P.vp, P.kps, P.vps, P.rec, P.layer};
  
  
  const int ev = pk_variant_env();
  const int var = (P.variant == 1 || ev == 1) ? 1 : (P.variant == 2 || (P.variant == 0 && ev == 2)) ? 2 : (P.variant == 3 || (P.variant == 0 && ev == 3)) ? 3 : 0;   
  const bool v1 = var == 1 || (var == 0 && A.M > 16 && P.kb == 4) || !P.kp;                                                  
  if (v1) {
    if (P.kb != 4) throw std::runtime_error("launch_q35_atx_pk: the non-VSP kernel (variant 1, V-only packing) supports kb = 4 only");
    const dim3 grid((A.M + 15) / 16, A.splits, Q35_KVH);
    if (pk_ns_env() == 3) pk_launch<3>(U, grid, st); else pk_launch<2>(U, grid, st);
    return;
  }
  if (var == 2 || (var == 3 && A.M <= pk3_mmax_env())) {                                                                          
    if (P.kb == 4) { if (P.vp) pk3_launch<4, true>(U, st); else pk3_launch<4, false>(U, st); }
    else           { if (P.vp) pk3_launch<5, true>(U, st); else pk3_launch<5, false>(U, st); }
    return;
  }
  if (var == 3) {                                                                                                                 
    if (A.M <= 2) throw std::runtime_error("launch_q35_atx_pk: v4 needs M > 2 (set YSKN_ATX_PK3_MMAX >= 2)");
    if (P.kb == 4) { if (P.vp) pk4_launch<4, true>(U, st); else pk4_launch<4, false>(U, st); }
    else           { if (P.vp) pk4_launch<5, true>(U, st); else pk4_launch<5, false>(U, st); }
    return;
  }
  if (P.kb == 4) { if (P.vp) pk_vsp_go<4, true>(U, st); else pk_vsp_go<4, false>(U, st); }
  else           { if (P.vp) pk_vsp_go<5, true>(U, st); else pk_vsp_go<5, false>(U, st); }
}
}  
