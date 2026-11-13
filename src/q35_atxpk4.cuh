// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once

















namespace {
constexpr int PK4_TL = 256;                                                 
constexpr int PK4_K8 = 256;                                                 

template <int KB, bool VPK, bool QK8>
__host__ __device__ constexpr size_t pk4_smem() {                           
  return (size_t)2 * PK_KT * (QK8 ? PK4_K8 : 2 * PK4_TL) + (size_t)2 * PK_KT * 2 * PK4_TL + (size_t)PK_KT * (32 * KB + (VPK ? 32 * KB : Q35_HD)) + (size_t)2 * PK_KT * 4 + 512 + (VPK ? 512 : 0) + (QK8 ? 272 : 0);
}
template <int RT> __host__ __device__ constexpr size_t pk4_static() { return (size_t)RT * 3072 + (size_t)2 * 2 * PK_KT * 4 + 8; }   


__device__ __forceinline__ void pk4_conv16_k4_s8(const unsigned* rw, int ch, const uint8_t* tab8, unsigned* w) {
  const uint2 v = *reinterpret_cast<const uint2*>(rw + 2 * ch);
  const unsigned pv = (ch & 3) ? (rw[2 * ch - 1] & 15u) : 0u;
  unsigned s[16];
  s[0] = (pv << 4) | (v.x >> 28);
  #pragma unroll
  for (int i = 1; i < 8; ++i) s[i] = (v.x >> (28 - 4 * i)) & 255u;
  s[8] = ((v.x & 15u) << 4) | (v.y >> 28);
  #pragma unroll
  for (int i = 1; i < 8; ++i) s[8 + i] = (v.y >> (28 - 4 * i)) & 255u;
  #pragma unroll
  for (int j = 0; j < 4; ++j) w[j] = pk3_lk4(tab8, s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
}
template <int HALF> __device__ __forceinline__ void pk4_conv16_k5_s8(const unsigned* rw, int chp, const uint8_t* tab8, unsigned* w) {
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
  for (int j = 0; j < 4; ++j) w[j] = pk3_lk4(tab8, s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
}
__device__ __forceinline__ void pk4_bar(int id, int nthreads) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(nthreads) : "memory"); }   
}  

template <int KB, bool VPK, int RT, int WPT, bool QK8>
__global__ void __launch_bounds__(RT * WPT * 32, 1) q35_atxpk4_kernel(Q35AtxPkArgs U) {
  constexpr int TPC = (16 * RT) / 6, JW = 4 / WPT, NB = 32 / WPT, TL = PK4_TL, PB = 32 * KB, NW = RT * WPT;
  static_assert(KB == 4 || KB == 5, "CDQ k = 4 or 5");
  static_assert(NW == 12, "v4: the 12-warp shapes (3 x 4, 6 x 2); M <= 2 is served by v3");
  static_assert(pk4_smem<KB, VPK, QK8>() + pk4_static<RT>() <= 101376, "shared memory: more than the 99 KB (101,376 B) a CTA may use on sm_120");
  const Q35AttnArgs& A = U.a;
  constexpr int D = Q35_HD;
  constexpr int KTB = QK8 ? PK4_K8 : 2 * PK4_TL;                             
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  __shared__ float2 x_mx[RT][4][32], x_sm[RT][4][32];
  __shared__ uint2 x_pf[RT][4][32];
  __shared__ float sc2[2][2][PK_KT];                                        
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = warp / WPT, part = warp % WPT, gi = lane >> 2, tig = lane & 3;
  const int grp = warp >> 2, cid = threadIdx.x & 127;                      
  const bool early = A.early != 0;
  
  uint8_t* const kt_b = reinterpret_cast<uint8_t*>(usm);
  uint16_t* const vt_b = reinterpret_cast<uint16_t*>(kt_b + (size_t)2 * PK_KT * KTB);
  uint8_t* const stg = reinterpret_cast<uint8_t*>(vt_b + (size_t)2 * PK_KT * TL);
  constexpr int KP = PB, VP = VPK ? PB : D, KREG = PK_KT * KP, VREG = PK_KT * VP;
  float* const ssc = reinterpret_cast<float*>(stg + KREG + VREG);
  uint16_t* const tabk = reinterpret_cast<uint16_t*>(ssc + 2 * PK_KT);
  uint16_t* const tabv = tabk + 256;
  uint8_t* const tab8k = reinterpret_cast<uint8_t*>(tabv + (VPK ? 256 : 0));
  const unsigned char* const rk = U.rec + (size_t)((U.layer * Q35_KVH + kvh) * 2) * PK_REC;      
  for (int i = threadIdx.x; i < 32; i += blockDim.x) {                      
    reinterpret_cast<uint4*>(tabk)[i] = reinterpret_cast<const uint4*>(rk)[i];
    if (VPK) reinterpret_cast<uint4*>(tabv)[i] = reinterpret_cast<const uint4*>(rk + PK_REC)[i];
    if (QK8 && i < 17) reinterpret_cast<uint4*>(tab8k)[i] = reinterpret_cast<const uint4*>(rk + 1536)[i];
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
  const uint8_t* const kpg = U.kp + (size_t)kvh * A.max_ctx * PB;
  const uint8_t* const vpg = VPK ? U.vp + (size_t)kvh * A.max_ctx * PB : nullptr;
  const int8_t* const vg8 = VPK ? nullptr : A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = U.kps + (size_t)kvh * A.max_ctx;
  const float* const vsg = (VPK ? U.vps : A.vs) + (size_t)kvh * A.max_ctx;
  constexpr int GK = 2 * KB, GV = VPK ? 2 * KB : 16;                        
  
  auto load_part = [&](int kv, int t, int pt, int tid, int nthr) {
    const int k0 = tbeg + t * PK_KT, ng = kv ? GV : GK, rp = kv ? VP : KP;
    uint8_t* const sb = stg + (kv ? KREG : 0);
    auto mine = [&](int kidx) { return pt == 0 || (pt == 1 ? kidx < base : kidx >= base); };
    for (int c = tid; c < PK_KT * ng; c += nthr) {
      const int key = c / ng, g = c - key * ng;
      uint8_t* const dst = sb + (size_t)key * rp + g * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) {
        if (mine(kidx)) {
          const void* src;
          if (kv) { if constexpr (VPK) src = vpg + (size_t)kidx * PB + g * 16; else src = vg8 + (size_t)kidx * D + g * 16; }
          else    src = kpg + (size_t)kidx * PB + g * 16;
          pk_cp16(dst, src);
        }
      } else if (pt != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int key = tid; key < PK_KT; key += nthr) {
      const int kidx = k0 + key;
      float* const dst = ssc + kv * PK_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) pk_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (pt != 2) *dst = 0.f;
    }
  };
  
  
  auto convert_part = [&](int kv, int b, int tid, int nthr) {
    const uint8_t* const src = stg + (kv ? KREG : 0);
    for (int cc = tid; cc < PK_KT * 16; cc += nthr) {
      int key, ch; unsigned w[8];
      const bool pkd = kv ? VPK : true;
      if (QK8 && !kv) {                                                       
        if constexpr (KB == 4) {
          key = cc >> 4; ch = cc & 15;
          pk4_conv16_k4_s8(reinterpret_cast<const unsigned*>(src + key * PB), ch, tab8k, w);
        } else {
          const int hf = cc >> 8, l = cc & 31;
          key = 4 * ((cc >> 5) & 7) + ((l >> 1) & 3);
          const int chp = 2 * ((l >> 3) & 3) + (l & 1);
          ch = 2 * chp + hf;
          const unsigned* const rw = reinterpret_cast<const unsigned*>(src + key * PB);
          if (hf == 0) pk4_conv16_k5_s8<0>(rw, chp, tab8k, w); else pk4_conv16_k5_s8<1>(rw, chp, tab8k, w);
        }
        uint8_t* const row = kt_b + (size_t)b * PK_KT * PK4_K8 + key * PK4_K8;
        *reinterpret_cast<uint4*>(row + 16 * (ch ^ (key & 7))) = make_uint4(w[0], w[1], w[2], w[3]);
        continue;
      }
      if (pkd) {
        const uint16_t* const tab = kv ? tabv : tabk;
        if constexpr (KB == 4) {
          key = cc >> 4; ch = cc & 15;
          pk_conv16_k4(reinterpret_cast<const unsigned*>(src + key * PB), ch, tab, w);
        } else {
          const int hf = cc >> 8, l = cc & 31;
          key = 4 * ((cc >> 5) & 7) + ((l >> 1) & 3);
          const int chp = 2 * ((l >> 3) & 3) + (l & 1);                      
          ch = 2 * chp + hf;
          const unsigned* const rw = reinterpret_cast<const unsigned*>(src + key * PB);
          if (hf == 0) pk_conv16_k5<0>(rw, chp, tab, w); else pk_conv16_k5<1>(rw, chp, tab, w);
        }
      } else {
        key = cc >> 4; ch = cc & 15;
        const uint4 u = *reinterpret_cast<const uint4*>(src + key * D + ch * 16);
        pk_s8x4_bf16(u.x, w[0], w[1]); pk_s8x4_bf16(u.y, w[2], w[3]); pk_s8x4_bf16(u.z, w[4], w[5]); pk_s8x4_bf16(u.w, w[6], w[7]);
      }
      uint16_t* const row = (kv ? vt_b + (size_t)b * PK_KT * TL : reinterpret_cast<uint16_t*>(kt_b + (size_t)b * PK_KT * KTB)) + key * TL; const int sw = key & 7;
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch) ^ sw)) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch + 1) ^ sw)) = make_uint4(w[4], w[5], w[6], w[7]);
    }
    for (int key = tid; key < PK_KT; key += nthr) sc2[b][kv][key] = ssc[kv * PK_KT + key];
  };
  
  if (0 < ntiles) { load_part(0, 0, early ? 1 : 0, threadIdx.x, NW * 32); load_part(1, 0, early ? 1 : 0, threadIdx.x, NW * 32); }
  pk_commit();
  if (early) { pdl_wait(); pdl_trigger(); if (0 < ntiles) { load_part(0, 0, 2, threadIdx.x, NW * 32); load_part(1, 0, 2, threadIdx.x, NW * 32); } pk_commit(); }
  unsigned qa[QK8 ? 8 : 16][4];                                              
  float sq0 = 1.f, sq1 = 1.f;                                                
  {
    const unsigned* q0 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow0 ? r0.tk : 0) * Q35_NH + r0.hd) * D);
    const unsigned* q1 = reinterpret_cast<const unsigned*>(A.q16 + ((size_t)(vrow1 ? r1.tk : 0) * Q35_NH + r1.hd) * D);
    if constexpr (!QK8) {
      #pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        qa[ks][0] = vrow0 ? q0[ks * 8 + tig] : 0u; qa[ks][1] = vrow1 ? q1[ks * 8 + tig] : 0u;
        qa[ks][2] = vrow0 ? q0[ks * 8 + 4 + tig] : 0u; qa[ks][3] = vrow1 ? q1[ks * 8 + 4 + tig] : 0u;
      }
    } else {                                                                 
      float qf[2][8][8]; float mx0 = 0.f, mx1 = 0.f;
      #pragma unroll
      for (int ks8 = 0; ks8 < 8; ++ks8) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned* q = h ? q1 : q0; const bool v = h ? vrow1 : vrow0;
          const unsigned wa = v ? q[16 * ks8 + 2 * tig] : 0u, wb = v ? q[16 * ks8 + 2 * tig + 1] : 0u, wc = v ? q[16 * ks8 + 8 + 2 * tig] : 0u, wd = v ? q[16 * ks8 + 8 + 2 * tig + 1] : 0u;
          qf[h][ks8][0] = pk_bf_lo(wa); qf[h][ks8][1] = pk_bf_hi(wa); qf[h][ks8][2] = pk_bf_lo(wb); qf[h][ks8][3] = pk_bf_hi(wb);
          qf[h][ks8][4] = pk_bf_lo(wc); qf[h][ks8][5] = pk_bf_hi(wc); qf[h][ks8][6] = pk_bf_lo(wd); qf[h][ks8][7] = pk_bf_hi(wd);
          #pragma unroll
          for (int i = 0; i < 8; ++i) { if (h) mx1 = fmaxf(mx1, fabsf(qf[1][ks8][i])); else mx0 = fmaxf(mx0, fabsf(qf[0][ks8][i])); }
        }
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));   
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      sq0 = mx0 > 0.f ? mx0 * (1.f / 127.f) : 1.f; sq1 = mx1 > 0.f ? mx1 * (1.f / 127.f) : 1.f;
      const float inv0 = 1.f / sq0, inv1 = 1.f / sq1;
      #pragma unroll
      for (int ks8 = 0; ks8 < 8; ++ks8) {
        qa[ks8][0] = pk3_q8pack(qf[0][ks8][0], qf[0][ks8][1], qf[0][ks8][2], qf[0][ks8][3], inv0); qa[ks8][1] = pk3_q8pack(qf[1][ks8][0], qf[1][ks8][1], qf[1][ks8][2], qf[1][ks8][3], inv1);
        qa[ks8][2] = pk3_q8pack(qf[0][ks8][4], qf[0][ks8][5], qf[0][ks8][6], qf[0][ks8][7], inv0); qa[ks8][3] = pk3_q8pack(qf[1][ks8][4], qf[1][ks8][5], qf[1][ks8][6], qf[1][ks8][7], inv1);
      }
    }
  }
  const float st8 = QK8 ? *reinterpret_cast<const float*>(rk + 1792) : 1.f;   
  pk_waitn<0>();                                                             
  __syncthreads();                                                           
  if (0 < ntiles) convert_part(0, 0, threadIdx.x, NW * 32);                  
  __syncthreads();                                                           
  if (1 < ntiles) load_part(0, 1, 0, threadIdx.x, NW * 32);                  
  pk_commit();
  pk_waitn<0>();
  const bool busy = __any_sync(0xffffffffu, vrow0 || vrow1);
  float o[NB][4];                                                            
  #pragma unroll
  for (int n = 0; n < NB; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float s[JW][4], c0 = 1.f, c1 = 1.f;                                        
  for (int t = 0; t <= ntiles; ++t) {
    __syncthreads();                                                         
    const bool isK = grp == t % 3, isV = grp == (t + 1) % 3;                 
    if (isK) {
      if (t + 1 < ntiles) convert_part(0, (t + 1) & 1, cid, 128);            
      pk4_bar(1 + grp, 128);                                                 
      if (t + 2 < ntiles) load_part(0, t + 2, 0, cid, 128);                  
      pk_commit();
    }
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
      const uint16_t* const vs = vt_b + (size_t)((t - 1) & 1) * PK_KT * TL;
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
    if (isV) {
      convert_part(1, t & 1, cid, 128);                                      
      pk4_bar(1 + grp, 128);                                                 
      if (t + 1 < ntiles) load_part(1, t + 1, 0, cid, 128);                  
      pk_commit();
    }
    const int b = t & 1;
    if (busy) {                                                              
      const float* const ksc = sc2[b][0];
      const int kt0 = tbeg + t * PK_KT;
      if constexpr (!QK8) {
        const uint16_t* const ks = reinterpret_cast<const uint16_t*>(kt_b + (size_t)b * PK_KT * KTB);
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
        }
      } else {
        const uint8_t* const k8 = kt_b + (size_t)b * PK_KT * KTB;
        #pragma unroll
        for (int w = 0; w < JW; ++w) {
          const int j = JW * part + w;
          int si[4] = {0, 0, 0, 0};
          #pragma unroll
          for (int kp = 0; kp < 4; ++kp) {                                   
            unsigned bb[4];
            const int r = 8 * j + (lane & 7), c = 4 * kp + (lane >> 3);
            pk_ldm4(bb, k8 + r * PK4_K8 + 16 * (c ^ (r & 7)));
            pk3_mma_s8(si, qa[2 * kp], bb[0], bb[1]);
            pk3_mma_s8(si, qa[2 * kp + 1], bb[2], bb[3]);
          }
          s[w][0] = (float)si[0] * sq0; s[w][1] = (float)si[1] * sq0; s[w][2] = (float)si[2] * sq1; s[w][3] = (float)si[3] * sq1;
        }
      }
      #pragma unroll
      for (int w = 0; w < JW; ++w) {
        const int j = JW * part + w;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int kk = 8 * j + 2 * tig + (c & 1), key = kt0 + kk, h = c >> 1;
          const float v = s[w][c] * ((QK8 ? st8 : 1.f) * (ksc[kk] * 0.0625f));
          s[w][c] = (key >= (h ? lo1 : lo0) && key < (h ? ph1 : ph0)) ? v : -INFINITY;
        }
        x_mx[tile][j][lane] = make_float2(fmaxf(s[w][0], s[w][1]), fmaxf(s[w][2], s[w][3]));
      }
    }
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
    pk_waitn<0>();                                                           
  }
  const int base2 = *reinterpret_cast<const volatile int*>(A.d_pos);
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const PkRow r = pk_row<TPC, true>(U, base2, t0, tile, gi, h, kvh, split);
    float& m = h ? m1 : m0; float& l = h ? l1 : l0;
    if (busy) {                                                              
      const bool sf = r.self_;                                               
      float pt = 0.f; int pti = 0;
      if (sf) {
        const size_t row = (size_t)r.own;
        const unsigned* const kw = reinterpret_cast<const unsigned*>(kpg + row * PB);
        if constexpr (!QK8) {
          #pragma unroll
          for (int kq = 0; kq < 16; ++kq) {
            const unsigned qlo = qa[kq][h], qhi = qa[kq][2 + h];             
            const float k0 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 2 * tig)), k1 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 2 * tig + 1));
            const float k2 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 8 + 2 * tig)), k3 = pk_tabf(tabk, pk_state_g<KB>(kw, 16 * kq + 9 + 2 * tig));
            pt = __fadd_rn(pt, __fmul_rn(pk_bf_lo(qlo), k0)); pt = __fadd_rn(pt, __fmul_rn(pk_bf_hi(qlo), k1));
            pt = __fadd_rn(pt, __fmul_rn(pk_bf_lo(qhi), k2)); pt = __fadd_rn(pt, __fmul_rn(pk_bf_hi(qhi), k3));
          }
        } else {                                                             
          #pragma unroll
          for (int ks8 = 0; ks8 < 8; ++ks8) {
            unsigned b0, b1; pk3_kfrag8<KB>(kw, ks8, tig, tab8k, b0, b1);
            const unsigned a0 = qa[ks8][h], a1 = qa[ks8][2 + h];
            pti = __dp4a((int)a0, (int)b0, pti); pti = __dp4a((int)a1, (int)b1, pti);
          }
        }
      }
      if constexpr (!QK8) { pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 1)); pt = __fadd_rn(pt, __shfl_xor_sync(0xffffffffu, pt, 2)); }
      else { pti += __shfl_xor_sync(0xffffffffu, pti, 1); pti += __shfl_xor_sync(0xffffffffu, pti, 2); pt = (float)pti * (h ? sq1 : sq0); }
      if (sf) {
        const float sc = pt * ((QK8 ? st8 : 1.f) * (ksg[r.own] * 0.0625f));
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


template <int KB, bool VPK, bool QK8> static void pk4_set(cudaError_t& e) {
  auto set = [&](const void* f, size_t smem) { const cudaError_t r = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); if (r != cudaSuccess && e == cudaSuccess) e = r; };
  set((const void*)q35_atxpk4_kernel<KB, VPK, 3, 4, QK8>, pk4_smem<KB, VPK, QK8>());
  set((const void*)q35_atxpk4_kernel<KB, VPK, 6, 2, QK8>, pk4_smem<KB, VPK, QK8>());
}
static void pk4_set_all(cudaError_t& e) {
  pk4_set<4, true, false>(e); pk4_set<4, false, false>(e); pk4_set<5, true, false>(e); pk4_set<5, false, false>(e);
  pk4_set<4, true, true>(e); pk4_set<4, false, true>(e); pk4_set<5, true, true>(e); pk4_set<5, false, true>(e);
}
template <int KB, bool VPK, bool QK8> static void pk4_go(const Q35AtxPkArgs& U, cudaStream_t st) {
  const int M = U.a.M, sp = U.a.splits;
  if (M <= 8) launch_k(q35_atxpk4_kernel<KB, VPK, 3, 4, QK8>, dim3((M + 7) / 8, sp, Q35_KVH), dim3(384), pk4_smem<KB, VPK, QK8>(), st, U);
  else launch_k(q35_atxpk4_kernel<KB, VPK, 6, 2, QK8>, dim3((M + 15) / 16, sp, Q35_KVH), dim3(384), pk4_smem<KB, VPK, QK8>(), st, U);
}
template <int KB, bool VPK> static void pk4_launch(const Q35AtxPkArgs& U, cudaStream_t st) { if (pk3_qk8_env()) pk4_go<KB, VPK, true>(U, st); else pk4_go<KB, VPK, false>(U, st); }
