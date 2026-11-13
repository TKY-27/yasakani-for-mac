// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors



#include "q35_g6.hpp"
#include "pm_math.cuh"
#include <algorithm>
#include <stdexcept>
#include <type_traits>
namespace yskn {
struct Q35G6Seg {
  TensorDev t; float* y; int ldy; const int8_t* qx; const float* alpha; const float* S;
  int cta_begin, ctas; const uint4* w8;
};
struct Q35G6Args {
  Q35G6Seg seg[3]; int n_seg, R; float eps; float* us_out;
  int xsm_sst, xsm_nb, pm; float xsm_n;
};
template <int K> __device__ __forceinline__ unsigned g6_field(const unsigned* w, int i) {
  constexpr int FB = 4 * K; const int bit = FB * i, wi = bit >> 5, sh = bit & 31;
  unsigned v = w[wi] >> sh;
  if (sh + FB > 32) v |= w[wi + 1] << (32 - sh);
  if constexpr (FB >= 32) return v; else return v & ((1u << FB) - 1u);
}
__device__ __forceinline__ void mt_mma(int* d, const unsigned* a, uint2 b) {       
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ void mt_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void mt_cp16s(unsigned saddr, const void* gmem_src) {  
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void mt_mad_wide(long long& acc, int a, int b) { asm("mad.wide.s32 %0, %1, %2, %0;" : "+l"(acc) : "r"(a), "r"(b)); }
template <int K> __global__ void __launch_bounds__(256) mt_predecode2_kernel(TensorDev t, uint4* w8) {
  constexpr int NU = K / 2, T = 12 - K, TS = 3 * K - 12, SS = 8;
  constexpr unsigned PM = (1u << (2 * K)) - 1u, TM = (1u << T) - 1u;
  __shared__ __align__(16) uint16_t s_lut[4096];
  __shared__ unsigned s_t[8][16][65];                                   
  for (int i = threadIdx.x; i < 512; i += blockDim.x) reinterpret_cast<uint4*>(s_lut)[i] = __ldg(reinterpret_cast<const uint4*>(t.lut8) + i);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tig = lane & 3, gid = lane >> 2;
  const int G = t.Kc / 512, nblk = 2 * G, tiles = t.N / 16, wg = (int)blockIdx.x * 8 + warp;
  if (wg >= tiles * nblk) return;                                      
  const int tile = wg / nblk, bi = wg - tile * nblk;
  const uint4* const sp = t.lno + (size_t)tile * G * 2 * NU * 32 + (size_t)bi * NU * 32 + lane;
  unsigned w[4 * NU];
  #pragma unroll
  for (int u = 0; u < NU; ++u) { const uint4 x = __ldg(sp + 32 * u); w[4 * u] = x.x; w[4 * u + 1] = x.y; w[4 * u + 2] = x.z; w[4 * u + 3] = x.w; }
  const int src = (lane & ~3) | ((tig + 3) & 3);
  const unsigned e = (w[K - 1] >> (32 - T)) | ((w[2 * K - 1] >> (32 - T)) << T);
  const unsigned nbv = __shfl_sync(0xffffffffu, e, src);
  unsigned q0 = tig ? (nbv & TM) : 0u, q1 = tig ? (nbv >> T) : 0u;
  auto pair_b = [&](unsigned p, unsigned q) -> unsigned {
    const unsigned W = q | (p << T); return unsigned(s_lut[W & 0xFFFu]) | (unsigned(s_lut[(W >> K) & 0xFFFu]) << 16); };
  #pragma unroll
  for (int m = 0; m < SS; ++m) {
    const unsigned f0 = g6_field<K>(w, m), f1 = g6_field<K>(w + K, m);
    const unsigned p10 = f0 & PM, p20 = f0 >> (2 * K), p11 = f1 & PM, p21 = f1 >> (2 * K);
    s_t[warp][gid][16 * tig + 2 * m] = pair_b(p10, q0); s_t[warp][gid][16 * tig + 2 * m + 1] = pair_b(p20, p10 >> TS);
    s_t[warp][gid + 8][16 * tig + 2 * m] = pair_b(p11, q1); s_t[warp][gid + 8][16 * tig + 2 * m + 1] = pair_b(p21, p11 >> TS);
    q0 = p20 >> TS; q1 = p21 >> TS;
  }
  __syncwarp();
  uint4* const dst = w8 + ((size_t)tile * nblk + bi) * (SS * 32) + lane;
  #pragma unroll
  for (int m = 0; m < SS; ++m)
    dst[32 * m] = make_uint4(s_t[warp][gid][8 * m + tig], s_t[warp][gid][8 * m + 4 + tig], s_t[warp][gid + 8][8 * m + tig], s_t[warp][gid + 8][8 * m + 4 + tig]);
}
constexpr int G6_BN = 128, G6_BK = 64, G6_S = 4, G6_GP = 12, G6_NT = 256;
template <int MB> struct G6C {
  static constexpr int BM = 32 * MB;                                   
  static constexpr int SA = BM * G6_BK;                                
  static constexpr int SB = (G6_BN / 16) * (G6_BK / 32) * 512;         
  static constexpr size_t SMEM = (size_t)G6_S * (SA + SB) + (size_t)2 * BM * G6_GP * 4 + (size_t)2 * 2 * BM * 4;
  static constexpr int NACC = MB * 4 * 4;                              
};
template <int N> __device__ __forceinline__ void g6_cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void g6_mma0(int* d, const unsigned* a, uint2 b) {   
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "r"(0));
}
template <int MB, bool BG, bool PN>   
__global__ void __launch_bounds__(G6_NT, 1) mt_gemm6_kernel(Q35G6Args P, int MT, int G, long long* skp, int* sks) {
  using C = G6C<MB>;
  constexpr int BM = C::BM, SA = C::SA, SB = C::SB;
  extern __shared__ __align__(128) unsigned char g6_smem[];
  unsigned char* const sA = g6_smem;                                   
  uint4* const sB = reinterpret_cast<uint4*>(g6_smem + G6_S * SA);    
  int* const s_ag = reinterpret_cast<int*>(g6_smem + G6_S * (SA + SB));  
  float* const s_rm = reinterpret_cast<float*>(s_ag);
  float* const s_us = reinterpret_cast<float*>(s_ag + 2 * BM * G6_GP);  
  float* const s_rr = s_us + 2 * BM;                                   
  __shared__ int s_claim;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 2, tig = lane & 3, wm = warp >> 2, wn = warp & 3;
  const int nseg = P.n_seg;
  const int cb1 = nseg > 1 ? P.seg[1].cta_begin : 1 << 30, cb2 = nseg > 2 ? P.seg[2].cta_begin : 1 << 30;
  const int NT = P.seg[nseg - 1].cta_begin + P.seg[nseg - 1].ctas;    
  const int U = MT * NT * G;                                           
  const int u0 = (int)((long long)U * blockIdx.x / gridDim.x), u1 = (int)((long long)U * (blockIdx.x + 1) / gridDim.x);
  const int nq = (u1 - u0) * 8;                                        
  const unsigned sA_s = (unsigned)__cvta_generic_to_shared(sA), sB_s = (unsigned)__cvta_generic_to_shared(sB);
  auto seg_of = [&](int tn) { return tn >= cb2 ? 2 : (tn >= cb1 ? 1 : 0); };
  
#define G6F(f, s) ((s) == 0 ? P.seg[0].f : ((s) == 1 ? P.seg[1].f : P.seg[2].f))
  constexpr int NA = BM / 64;                                          
  const int r0 = tid >> 2, c0 = tid & 3;
  const unsigned dA = (unsigned)(r0 * 64 + 16 * (c0 ^ ((r0 >> 1) & 3))), dB = (unsigned)tid * 16u;   
  int pu = u0, pkk = 0, pg = 0, pcnt = 0, pkc = 0; bool pv0 = true, pv1 = true;   
  const int8_t* pa = nullptr; const uint4* pb = nullptr;
  auto psetup = [&]() {                                                
    const int tile = pu / G; pg = pu - tile * G;
    const int tn = tile / MT, tm = tile - tn * MT, s = seg_of(tn);
    const int row0 = tm * BM; pcnt = min(BM, P.R - row0); pkc = G6F(t.Kc, s);
    const int nblk = pkc >> 8;
    pa = G6F(qx, s) + (size_t)(row0 + r0) * pkc + pg * 512 + 16 * c0;
    pb = G6F(w8, s) + ((size_t)((tn - G6F(cta_begin, s)) * 8 + (tid >> 6)) * nblk) * 256 + (size_t)pg * 512 + (tid & 63);
    if constexpr (PN) { const int nvt = (G6F(t.N, s) - (tn - G6F(cta_begin, s)) * G6_BN) >> 4;   
                        pv0 = (tid >> 6) < nvt; pv1 = (tid >> 6) + 4 < nvt; }                     
  };
  auto produce = [&](int slot) {                                       
    if (pkk == 0 && (pu == u0 || pg == 0)) psetup();
    const unsigned sa = sA_s + slot * SA + dA, sb = sB_s + slot * SB + dB;
    if (r0 < pcnt) mt_cp16s(sa, pa);
    if constexpr (NA == 2) { if (r0 + 64 < pcnt) mt_cp16s(sa + 4096, pa + (size_t)64 * pkc); }
    if (pv0) mt_cp16s(sb, pb); if (pv1) mt_cp16s(sb + 4096, pb + (size_t)4 * pkc);   
    pa += 64; pb += 64;
    if (++pkk == 8) { pkk = 0; ++pu; if (++pg == G) pg = 0; }
  };
  auto fill = [&](int buf, int s, int row0, int cnt) {                
    const float* const al = G6F(alpha, s); const float* const Sx = G6F(S, s);
    for (int r = tid; r < BM; r += G6_NT) {
      float m = 0.f, rr = 0.f;
      if (r < cnt) { const int row = row0 + r;
        for (int k = 0; k < G; ++k) m = fmaxf(m, __ldg(al + (size_t)row * G + k));
        if (Sx) {
          const float* sr = Sx + (size_t)row * P.xsm_sst; float sum = 0.f;
          for (int b = 0; b < P.xsm_nb; ++b) sum = __fadd_rn(sum, sr[b]);
          const float a = __fadd_rn(__fdiv_rn(sum, P.xsm_n), P.eps);
          rr = P.pm ? pm_rsqrt(a) : rsqrtf(a);
        } }
      const float rm = m > 0.f ? __fdiv_rn(16777216.f, m) : 0.f;
      s_us[buf * BM + r] = ldexpf(m, -24); s_rr[buf * BM + r] = rr;
      if constexpr (BG) s_rm[buf * BM + r] = r < cnt ? rm : 0.f;
      else for (int k = 0; k < G; ++k)
        s_ag[(buf * BM + r) * G6_GP + k] = r < cnt ? __float2int_rn(__fmul_rn(__ldg(al + (size_t)(row0 + r) * G + k), rm)) : 0;
    }
  };
  const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1), lh = lane >> 4, sw = (lrow >> 1) & 3;   
  const unsigned aw = sA_s + (16 * MB * wm + lrow) * 64, a0off = 16 * ((0 + lh) ^ sw), a1off = 16 * ((2 + lh) ^ sw);
  const unsigned bw = sB_s + ((2 * wn) * 64 + lane) * 16;
  int d[MB][4][4]; long long acc[MB][4][4];
  #pragma unroll
  for (int i = 0; i < MB; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      #pragma unroll
      for (int c = 0; c < 4; ++c) { d[i][j][c] = 0; acc[i][j][c] = 0; } } }
  auto consume = [&](auto slot_c, auto first_c) {                      
    constexpr int slot = decltype(slot_c)::value; constexpr bool first = decltype(first_c)::value;
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
      uint4 ba, bc; unsigned a[2][4];
      const unsigned bb = bw + slot * SB, as = aw + slot * SA + (s ? a1off : a0off);
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(ba.x), "=r"(ba.y), "=r"(ba.z), "=r"(ba.w) : "r"(bb + s * 512));
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(bc.x), "=r"(bc.y), "=r"(bc.z), "=r"(bc.w) : "r"(bb + 1024 + s * 512));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(a[0][0]), "=r"(a[0][1]), "=r"(a[0][2]), "=r"(a[0][3]) : "r"(as));
      const uint2 b0 = make_uint2(ba.x, ba.y), b1 = make_uint2(ba.z, ba.w), b2 = make_uint2(bc.x, bc.y), b3 = make_uint2(bc.z, bc.w);
      #pragma unroll
      for (int i = 0; i < MB; ++i) {                                   
        if (i + 1 < MB)
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(a[(i + 1) & 1][0]), "=r"(a[(i + 1) & 1][1]), "=r"(a[(i + 1) & 1][2]), "=r"(a[(i + 1) & 1][3]) : "r"(as + (i + 1) * 1024));
        if (first && s == 0) { g6_mma0(d[i][0], a[i & 1], b0); g6_mma0(d[i][1], a[i & 1], b1); g6_mma0(d[i][2], a[i & 1], b2); g6_mma0(d[i][3], a[i & 1], b3); }
        else { mt_mma(d[i][0], a[i & 1], b0); mt_mma(d[i][1], a[i & 1], b1); mt_mma(d[i][2], a[i & 1], b2); mt_mma(d[i][3], a[i & 1], b3); }
      }
    }
  };
  #pragma unroll
  for (int s = 0; s < G6_S - 1; ++s) { if (s < nq) produce(s); mt_cp_commit(); }
  int piece = -1, pgs0 = 0, ctile = u0 / G, cg = u0 - (u0 / G) * G;  
  const float* bal = nullptr; int bcnt = 1; float fa[MB][2];          
  for (int u = u0; u < u1; ++u) {
    const bool lastu = u + 1 == u1;                                    
    if (u == u0 || cg == 0) {                                          
      ++piece; pgs0 = cg;
      const int tn = ctile / MT, tm = ctile - tn * MT, row0 = tm * BM;
      fill(piece & 1, seg_of(tn), row0, min(BM, P.R - row0));
      
      if (tn == 0 && cg == 0 && tid < BM && row0 + tid < P.R && P.us_out)
        P.us_out[row0 + tid] = s_us[(piece & 1) * BM + tid];
      if constexpr (BG) { bal = G6F(alpha, seg_of(tn)) + (size_t)row0 * G; bcnt = min(BM, P.R - row0); }
    }
    auto stage = [&](auto kk_c) {                                      
      constexpr int kk = decltype(kk_c)::value;
      g6_cp_wait<G6_S - 2>(); __syncthreads();                         
      if (kk < 5 || !lastu) produce((kk + G6_S - 1) & (G6_S - 1));
      mt_cp_commit();
      if constexpr (BG && kk == 7) {                                   
        #pragma unroll
        for (int i = 0; i < MB; ++i) { const int rA = 16 * MB * wm + 16 * i + gi;
          fa[i][0] = __ldg(bal + (size_t)min(rA, bcnt - 1) * G + cg); fa[i][1] = __ldg(bal + (size_t)min(rA + 8, bcnt - 1) * G + cg); } }
      consume(std::integral_constant<int, kk & (G6_S - 1)>{}, std::integral_constant<bool, kk == 0>{});
    };
    stage(std::integral_constant<int, 0>{}); stage(std::integral_constant<int, 1>{}); stage(std::integral_constant<int, 2>{});
    stage(std::integral_constant<int, 3>{}); stage(std::integral_constant<int, 4>{}); stage(std::integral_constant<int, 5>{});
    stage(std::integral_constant<int, 6>{}); stage(std::integral_constant<int, 7>{});
    const int g = cg, tile = ctile;
    if (++cg == G) { cg = 0; ++ctile; }                                
    {                                                                  
      const int* const ag = s_ag + (piece & 1) * BM * G6_GP + g;
      #pragma unroll
      for (int i = 0; i < MB; ++i) {
        const int rA = 16 * MB * wm + 16 * i + gi;
        int AA, AB;
        if constexpr (BG) { const float* const rm = s_rm + (piece & 1) * BM;   
          AA = __float2int_rn(__fmul_rn(fa[i][0], rm[rA])); AB = __float2int_rn(__fmul_rn(fa[i][1], rm[rA + 8])); }
        else { AA = ag[rA * G6_GP]; AB = ag[(rA + 8) * G6_GP]; }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          mt_mad_wide(acc[i][j][0], AA, d[i][j][0]); mt_mad_wide(acc[i][j][1], AA, d[i][j][1]);
          mt_mad_wide(acc[i][j][2], AB, d[i][j][2]); mt_mad_wide(acc[i][j][3], AB, d[i][j][3]);
        }
      }
    }
    if (!lastu && cg != 0) continue;                                   
    if (!(pgs0 == 0 && g == G - 1)) {                                  
      const int slot = pgs0 > 0 ? (int)blockIdx.x : (int)blockIdx.x + 1;
      long long* const sp = skp + (size_t)slot * (G6_NT * C::NACC) + tid;
      __syncthreads();
      if (tid == 0) s_claim = atomicAdd(sks + 2 * slot, 1);
      __syncthreads();
      if (s_claim == 0) {                                              
        #pragma unroll
        for (int i = 0; i < MB; ++i) {
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) { __stcg(sp + (size_t)((i * 4 + j) * 4 + c) * G6_NT, acc[i][j][c]); acc[i][j][c] = 0; } } }
        __threadfence(); __syncthreads();
        if (tid == 0) atomicExch(sks + 2 * slot + 1, 1);
        continue;
      }
      if (tid == 0) { for (int it = 0; atomicAdd(sks + 2 * slot + 1, 0) == 0; ++it) { __nanosleep(32); if (it > (1 << 26)) __trap(); }   
                      __threadfence(); sks[2 * slot] = 0; sks[2 * slot + 1] = 0; }
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < MB; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          #pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] += __ldcg(sp + (size_t)((i * 4 + j) * 4 + c) * G6_NT); } }
    }
    {                                                                  
      const int tn = tile / MT, tm = tile - tn * MT, row0 = tm * BM, cnt = min(BM, P.R - row0), buf = piece & 1, s = seg_of(tn);
      float* const yb = G6F(y, s); const int ldy = G6F(ldy, s); const bool wr = G6F(S, s) != nullptr;
      const int n = (tn - G6F(cta_begin, s)) * G6_BN + 32 * wn + 2 * tig, nN = G6F(t.N, s);   
      #pragma unroll
      for (int i = 0; i < MB; ++i) {
        const int sA_ = 16 * MB * wm + 16 * i + gi, sB_ = sA_ + 8;
        const float usA = s_us[buf * BM + sA_], rrA = s_rr[buf * BM + sA_], usB = s_us[buf * BM + sB_], rrB = s_rr[buf * BM + sB_];
        float* const yA = yb + (size_t)(row0 + sA_) * ldy + n; float* const yB = yA + (size_t)8 * ldy;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (sA_ < cnt && (!PN || n + 8 * j < nN)) { float y0 = __fmul_rn(__ll2float_rn(acc[i][j][0]), usA), y1 = __fmul_rn(__ll2float_rn(acc[i][j][1]), usA);
                           if (wr) { y0 = __fmul_rn(y0, rrA); y1 = __fmul_rn(y1, rrA); }
                           *reinterpret_cast<float2*>(yA + 8 * j) = make_float2(y0, y1); }
          if (sB_ < cnt && (!PN || n + 8 * j < nN)) { float y0 = __fmul_rn(__ll2float_rn(acc[i][j][2]), usB), y1 = __fmul_rn(__ll2float_rn(acc[i][j][3]), usB);
                           if (wr) { y0 = __fmul_rn(y0, rrB); y1 = __fmul_rn(y1, rrB); }
                           *reinterpret_cast<float2*>(yB + 8 * j) = make_float2(y0, y1); }
          acc[i][j][0] = 0; acc[i][j][1] = 0; acc[i][j][2] = 0; acc[i][j][3] = 0;
        }
      }
    }
  }
#undef G6F
}

template <int MB, bool BG, bool PN> static int g6_prepare_kernel(int sms) {
  using C = G6C<MB>;
  if (cudaFuncSetAttribute(mt_gemm6_kernel<MB, BG, PN>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM) != cudaSuccess)
    throw std::runtime_error("q35_g6: dynamic shared-memory configuration failed");
  int per = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, mt_gemm6_kernel<MB, BG, PN>, G6_NT, C::SMEM) != cudaSuccess || per < 1)
    throw std::runtime_error("q35_g6: no resident block");
  return sms * per;
}
void q35_g6_prepare(Q35G6Workspace& W, int sms) {
  W.resident[0] = g6_prepare_kernel<4, false, false>(sms);
  W.resident[1] = g6_prepare_kernel<4, false, true>(sms);
  W.resident[2] = g6_prepare_kernel<2, true, false>(sms);
  W.resident[3] = g6_prepare_kernel<2, true, true>(sms);
  int cap = *std::max_element(W.resident, W.resident + 4);
  W.partial_bytes = (size_t)(cap + 1) * G6_NT * G6C<4>::NACC * sizeof(long long);
  W.flags_bytes = (size_t)(cap + 1) * 2 * sizeof(int);
}
template <int MB, bool BG, bool PN> static void g6_launch(Q35G6Args P, Q35G6Workspace& W, cudaStream_t st) {
  const int MT = (P.R + G6C<MB>::BM - 1) / G6C<MB>::BM;
  const int NT = P.seg[P.n_seg - 1].cta_begin + P.seg[P.n_seg - 1].ctas;
  const int G = P.seg[0].t.Kc / 512;
  const int ctas = std::min(MT * NT, W.resident[(BG ? 2 : 0) + (PN ? 1 : 0)]);
  mt_gemm6_kernel<MB, BG, PN><<<ctas, G6_NT, G6C<MB>::SMEM, st>>>(P, MT, G, W.partial, W.flags);
}
void q35_g6_decode(const TensorDev& t, uint4* dst, cudaStream_t st) {
  const int blocks = ((t.N / 16) * (t.Kc / 256) + 7) / 8;
  if (t.k_default == 6) mt_predecode2_kernel<6><<<blocks, 256, 0, st>>>(t, dst);
  else if (t.k_default == 8) mt_predecode2_kernel<8><<<blocks, 256, 0, st>>>(t, dst);
  else mt_predecode2_kernel<4><<<blocks, 256, 0, st>>>(t, dst);
}
bool launch_q35_g6(const GemvLaunch& P, Q35G6Workspace& W, int pm, cudaStream_t st) {
  if (!W.weights || P.M < 256 || !P.xsum || P.lo || P.fuse || P.draft || P.e_fuse || P.f_mode || P.xsm_part || P.n_seg < 1 || P.n_seg > 3)
    return false;
  size_t bytes = 0, shape_bytes = 0; int nt = 0; bool pn = false; const int Kc = P.seg[0].t.Kc;
  if (Kc < 512 || Kc % 512) return false;
  Q35G6Args Q{}; Q.n_seg = P.n_seg; Q.R = P.M; Q.eps = P.xsm_eps; Q.us_out = P.us_out;
  Q.xsm_sst = P.xsm_sst; Q.xsm_nb = P.xsm_nb; Q.xsm_n = P.xsm_n; Q.pm = pm;
  for (int s = 0; s < P.n_seg; ++s) {
    const auto& t = P.seg[s].t;
    if (!t.lno || !t.lut8 || t.tile_k || t.group_worker || t.Kc != Kc || t.N <= 0 || t.N % 16 || (t.k_default != 4 && t.k_default != 6 && t.k_default != 8))
      return false;
    auto& g = Q.seg[s]; g.t = t; g.y = P.seg[s].part; g.ldy = t.N; g.qx = P.qx; g.alpha = P.alpha; g.S = P.xsm_S;
    g.cta_begin = nt; g.ctas = (t.N + 127) / 128; nt += g.ctas; pn |= t.N % 128 != 0;
    shape_bytes += (size_t)t.N*t.Kc;
    const auto saved=W.cached.find(t.lno);
    if (saved!=W.cached.end()) g.w8=saved->second;
    else { g.w8 = reinterpret_cast<uint4*>(reinterpret_cast<char*>(W.weights) + bytes); bytes += (size_t)t.N * t.Kc; }
  }
  if ((long long)((P.M + 63) / 64) * nt * (Kc / 512) * 8 >= (1LL << 30)) return false;
  if (shape_bytes > W.weight_bytes) {
    
    
    for (int s = 0; s < P.n_seg; ++s)
      if ((size_t)P.seg[s].t.N * P.seg[s].t.Kc > W.weight_bytes) return false;
    for (int s = 0; s < P.n_seg; ++s) {
      GemvLaunch one = P; one.n_seg = 1; one.seg[0] = P.seg[s]; one.seg[0].cta_begin = 0;
      if (!launch_q35_g6(one, W, pm, st)) throw std::runtime_error("q35_g6: validated segment could not launch");
    }
    return true;
  }
  for (int s = 0; s < P.n_seg; ++s) {
    const auto& t = P.seg[s].t;
    if (W.cached.find(t.lno)!=W.cached.end()) continue;
    const int blocks = ((t.N / 16) * (t.Kc / 256) + 7) / 8;
    uint4* dst = const_cast<uint4*>(Q.seg[s].w8);
    if (t.k_default == 6) mt_predecode2_kernel<6><<<blocks, 256, 0, st>>>(t, dst);
    else if (t.k_default == 8) mt_predecode2_kernel<8><<<blocks, 256, 0, st>>>(t, dst);
    else mt_predecode2_kernel<4><<<blocks, 256, 0, st>>>(t, dst);
  }
  if (Kc / 512 > G6_GP) { if (pn) g6_launch<2, true, true>(Q, W, st); else g6_launch<2, true, false>(Q, W, st); }
  else { if (pn) g6_launch<4, false, true>(Q, W, st); else g6_launch<4, false, false>(Q, W, st); }
  return true;
}
}  
