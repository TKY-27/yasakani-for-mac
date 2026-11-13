// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once






















namespace {
constexpr int PK3_TL = 256;                                                 


template <int KB> __device__ __forceinline__ unsigned pk3_state(const unsigned* rw, int e) { return pk_state_g<KB>(rw, e); }



template <int NB> __device__ __forceinline__ unsigned pk3_win5(const unsigned* krow, int e) {
  constexpr int W = 5 * NB + 3; constexpr unsigned MASK = (1u << W) - 1u;
  const unsigned* const cw = krow + (e >> 6) * 10; const int ec = e & 63;
  if (ec == 0) return (cw[0] >> (32 - (W - 3))) & ((1u << (W - 3)) - 1u);           
  const int p = 5 * ec - 3, j = p >> 5, r = p & 31;
  if (r + W <= 32) return (cw[j] >> (32 - r - W)) & MASK;
  return __funnelshift_r(cw[j + 1], cw[j], (unsigned)(64 - r - W)) & MASK;
}




template <int KB> __device__ __forceinline__ void pk3_kfrag(const unsigned* krow, int ks, int tig, const uint16_t* tab, unsigned& prev, unsigned& r0, unsigned& r1) {
  if constexpr (KB == 4) {
    const uint2 w = *reinterpret_cast<const uint2*>(krow + 2 * ks);
    const int sh = 28 - 8 * tig;                                            
    const unsigned a0 = (w.x >> sh) & 15u, a1 = (w.x >> (sh - 4)) & 15u, b0 = (w.y >> sh) & 15u, b1 = (w.y >> (sh - 4)) & 15u;
    const unsigned ap = tig ? ((w.x >> (sh + 4)) & 15u) : ((ks & 3) ? prev : 0u);
    const unsigned bp = tig ? ((w.y >> (sh + 4)) & 15u) : (w.x & 15u);
    prev = w.y & 15u;
    r0 = pk_lk2(tab, (ap << 4) | a0, (a0 << 4) | a1);
    r1 = pk_lk2(tab, (bp << 4) | b0, (b0 << 4) | b1);
  } else {
    const int e = 16 * ks + 2 * tig;
    const unsigned w0 = pk3_win5<2>(krow, e), w1 = pk3_win5<2>(krow, e + 8);         
    r0 = pk_lk2(tab, (w0 >> 5) & 255u, w0 & 255u);
    r1 = pk_lk2(tab, (w1 >> 5) & 255u, w1 & 255u);
  }
}



template <int KB> __device__ __forceinline__ void pk3_kfrag_lut(const unsigned* krow, int ks, int tig, const unsigned* plut, unsigned& prev, unsigned& r0, unsigned& r1) {
  if constexpr (KB == 4) {
    const uint2 w = *reinterpret_cast<const uint2*>(krow + 2 * ks);
    const int sh = 28 - 8 * tig;                                            
    const unsigned w0 = tig ? ((w.x >> (sh - 4)) & 4095u) : ((((ks & 3) ? prev : 0u) << 8) | (w.x >> 24));
    const unsigned w1 = tig ? ((w.y >> (sh - 4)) & 4095u) : (((w.x & 15u) << 8) | (w.y >> 24));
    prev = w.y & 15u;
    r0 = plut[w0]; r1 = plut[w1];
  } else {
    const int e = 16 * ks + 2 * tig;
    r0 = plut[pk3_win5<2>(krow, e)]; r1 = plut[pk3_win5<2>(krow, e + 8)];
  }
}

__device__ __forceinline__ void pk3_conv16_k4_lut(const unsigned* rw, int ch, const unsigned* plut, unsigned* b) {
  const uint2 w = *reinterpret_cast<const uint2*>(rw + 2 * ch);
  const unsigned pv = (ch & 3) ? (rw[2 * ch - 1] & 15u) : 0u;
  b[0] = plut[(pv << 8) | (w.x >> 24)];
  #pragma unroll
  for (int j = 1; j < 4; ++j) b[j] = plut[(w.x >> (24 - 8 * j)) & 4095u];
  b[4] = plut[((w.x & 15u) << 8) | (w.y >> 24)];
  #pragma unroll
  for (int j = 5; j < 8; ++j) b[j] = plut[(w.y >> (24 - 8 * (j - 4))) & 4095u];
}
__device__ __forceinline__ void pk3_mma_s8(int* c, const unsigned* a, unsigned b0, unsigned b1) {           
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pk3_lk4(const uint8_t* tab8, unsigned s0, unsigned s1, unsigned s2, unsigned s3) {   
  return (unsigned)tab8[s0] | ((unsigned)tab8[s1] << 8) | ((unsigned)tab8[s2] << 16) | ((unsigned)tab8[s3] << 24);
}



template <int KB> __device__ __forceinline__ void pk3_kfrag8(const unsigned* krow, int ks8, int tig, const uint8_t* tab8, unsigned& b0, unsigned& b1) {
  if constexpr (KB == 4) {
    const int w0 = 4 * ks8 + (tig >> 1), sh = 28 - 16 * (tig & 1);          
    const unsigned x = krow[w0], y = krow[w0 + 2];
    const unsigned xp = (tig & 1) ? ((x >> (sh + 4)) & 15u) : (((ks8 & 1) || tig) ? (krow[w0 - 1] & 15u) : 0u);
    const unsigned yp = (tig & 1) ? ((y >> (sh + 4)) & 15u) : (krow[w0 + 1] & 15u);
    const unsigned x0 = (x >> sh) & 15u, x1 = (x >> (sh - 4)) & 15u, x2 = (x >> (sh - 8)) & 15u, x3 = (x >> (sh - 12)) & 15u;
    const unsigned y0 = (y >> sh) & 15u, y1 = (y >> (sh - 4)) & 15u, y2 = (y >> (sh - 8)) & 15u, y3 = (y >> (sh - 12)) & 15u;
    b0 = pk3_lk4(tab8, (xp << 4) | x0, (x0 << 4) | x1, (x1 << 4) | x2, (x2 << 4) | x3);
    b1 = pk3_lk4(tab8, (yp << 4) | y0, (y0 << 4) | y1, (y1 << 4) | y2, (y2 << 4) | y3);
  } else {
    const int e = 32 * ks8 + 4 * tig;
    const unsigned w0 = pk3_win5<4>(krow, e), w1 = pk3_win5<4>(krow, e + 16);       
    b0 = pk3_lk4(tab8, (w0 >> 15) & 255u, (w0 >> 10) & 255u, (w0 >> 5) & 255u, w0 & 255u);
    b1 = pk3_lk4(tab8, (w1 >> 15) & 255u, (w1 >> 10) & 255u, (w1 >> 5) & 255u, w1 & 255u);
  }
}
__device__ __forceinline__ unsigned pk3_q8pack(float a, float b, float c, float d, float inv) {                    
  const int i0 = __float2int_rn(fminf(fmaxf(a * inv, -127.f), 127.f)), i1 = __float2int_rn(fminf(fmaxf(b * inv, -127.f), 127.f));
  const int i2 = __float2int_rn(fminf(fmaxf(c * inv, -127.f), 127.f)), i3 = __float2int_rn(fminf(fmaxf(d * inv, -127.f), 127.f));
  return ((unsigned)i0 & 255u) | (((unsigned)i1 & 255u) << 8) | (((unsigned)i2 & 255u) << 16) | (((unsigned)i3 & 255u) << 24);
}
}  

template <int KB, bool VPK, int RT, int WPT, int NST, bool SKEW, bool PLUT = false>
__host__ __device__ constexpr size_t pk3_smem() {                                           
  return (size_t)2 * PK_KT * PK3_TL * 2 + (size_t)NST * PK_KT * (32 * KB + (VPK ? 32 * KB : Q35_HD)) + (size_t)NST * 2 * PK_KT * 4 + 512 + (VPK ? 512 : 0) + 256 + 16
       + (PLUT ? (KB == 4 ? (size_t)4096 * 4 * (VPK ? 2 : 1) : (size_t)8192 * 4) : 0);
}
template <int RT> __host__ __device__ constexpr size_t pk3_static() { return (size_t)RT * 3072 + (size_t)2 * PK_KT * 4 + 8; }   

template <int KB, bool VPK, int RT, int WPT, int NST, bool SKEW, int XW, bool QK8, bool PLUT = false>
__global__ void __launch_bounds__((RT * WPT + XW) * 32, 1) q35_atxpk3_kernel(Q35AtxPkArgs U) {
  constexpr int TPC = (16 * RT) / 6, JW = 4 / WPT, NB = 32 / WPT, TL = PK3_TL, PB = 32 * KB, NWM = RT * WPT, NW = NWM + XW;
  static_assert(KB == 4 || KB == 5, "CDQ k = 4 or 5");
  static_assert(NST >= 2 && NST <= 3, "stage ring of 2 or 3 tiles (4 does not fit in shared memory with an INT8 V stage)");
  static_assert(XW == 0 || (RT == 1 && XW == 4 && SKEW), "extra (converter-only) warps: 4 of them, with the 1 x 4 shape, SKEW");
  static_assert(!SKEW || NW == 12 || (RT == 1 && XW == 4), "SKEW: 12-warp CTAs or 4 + 4 warps (group B = warps 4..7)");
  static_assert(!PLUT || (RT == 1 && !QK8 && (KB == 4 || NST == 2)), "PLUT: the 1 x 4 shape, BF16 scores; k = 5 with a 2-tile stage ring (shared memory)");
  static_assert(pk3_smem<KB, VPK, RT, WPT, NST, SKEW, PLUT>() + pk3_static<RT>() <= 101376, "shared memory: more than the 99 KB (101,376 B) a CTA may use on sm_120");
  const Q35AttnArgs& A = U.a;
  constexpr int D = Q35_HD;
  extern __shared__ __align__(16) uint16_t usm[];
  __shared__ int s_lo, s_hi;
  __shared__ float2 x_mx[RT][4][32], x_sm[RT][4][32];
  __shared__ uint2 x_pf[RT][4][32];
  __shared__ float vsc2[2][PK_KT];                                        
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = warp / WPT, part = warp % WPT, gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0;
  const bool grpB = SKEW && (warp >> 2) == 1;                             
  constexpr int NCONV = SKEW ? 128 : NW * 32;                              
  const int cid = SKEW ? threadIdx.x - 128 : threadIdx.x;                 
  
  constexpr int KP = PB, VP = VPK ? PB : D, KREG = PK_KT * KP, VREG = PK_KT * VP, SREG = KREG + VREG;
  uint16_t* const vt_b = usm;
  uint8_t* const stg = reinterpret_cast<uint8_t*>(usm + 2 * PK_KT * TL);
  float* const ssc = reinterpret_cast<float*>(stg + (size_t)NST * SREG);
  uint16_t* const tabk = reinterpret_cast<uint16_t*>(ssc + NST * 2 * PK_KT);
  uint16_t* const tabv = tabk + 256;
  uint8_t* const tab8k = reinterpret_cast<uint8_t*>(tabv + (VPK ? 256 : 0));              
  unsigned* const plutk = reinterpret_cast<unsigned*>(tab8k + 272);                       
  unsigned* const plutv = plutk + (KB == 4 ? 4096 : 8192);
  const unsigned char* const rk = U.rec + (size_t)((U.layer * Q35_KVH + kvh) * 2) * PK_REC;      
  for (int i = threadIdx.x; i < 32; i += blockDim.x) {                
    reinterpret_cast<uint4*>(tabk)[i] = reinterpret_cast<const uint4*>(rk)[i];
    if (VPK) reinterpret_cast<uint4*>(tabv)[i] = reinterpret_cast<const uint4*>(rk + PK_REC)[i];
    if (QK8 && i < 17) reinterpret_cast<uint4*>(tab8k)[i] = reinterpret_cast<const uint4*>(rk + 1536)[i];
  }
  if constexpr (PLUT) {                                                 
    const uint16_t* const gk = reinterpret_cast<const uint16_t*>(rk);
    const uint16_t* const gv = reinterpret_cast<const uint16_t*>(rk + PK_REC);
    if constexpr (KB == 4) {
      for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        const unsigned p = i >> 8, a = (i >> 4) & 15, b = i & 15;
        plutk[i] = (unsigned)gk[(p << 4) | a] | ((unsigned)gk[(a << 4) | b] << 16);
        if (VPK) plutv[i] = (unsigned)gv[(p << 4) | a] | ((unsigned)gv[(a << 4) | b] << 16);
      }
    } else {
      for (int i = threadIdx.x; i < 8192; i += blockDim.x) plutk[i] = (unsigned)gk[(i >> 5) & 255] | ((unsigned)gk[i & 255] << 16);
    }
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
  auto load_tile = [&](int t, int pt) {                                 
    const int k0 = tbeg + t * PK_KT; uint8_t* const sb = stg + (size_t)(t % NST) * SREG; float* const sc = ssc + (t % NST) * 2 * PK_KT;
    auto mine = [&](int kidx) { return pt == 0 || (pt == 1 ? kidx < base : kidx >= base); };
    for (int c = threadIdx.x; c < PK_KT * (GK + GV); c += blockDim.x) {
      const int kv = c >= PK_KT * GK, cc = kv ? c - PK_KT * GK : c, ng = kv ? GV : GK;
      const int key = cc / ng, g = cc - key * ng;
      uint8_t* const dst = sb + (kv ? KREG : 0) + (size_t)key * (kv ? VP : KP) + g * 16;
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
    for (int c = threadIdx.x; c < 2 * PK_KT; c += blockDim.x) {
      const int kv = c / PK_KT, key = c % PK_KT, kidx = k0 + key;
      float* const dst = sc + kv * PK_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) pk_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (pt != 2) *dst = 0.f;
    }
  };
  
  auto convert_v = [&](int t) {
    const int b = t & 1; uint16_t* const vbuf = vt_b + (size_t)b * PK_KT * TL;
    const uint8_t* const sv = stg + (size_t)(t % NST) * SREG + KREG; const float* const sc = ssc + (t % NST) * 2 * PK_KT + PK_KT;
    for (int cc = cid; cc < PK_KT * 16; cc += NCONV) {
      int key, ch; unsigned w[8];
      if constexpr (VPK) {
        if constexpr (KB == 4) {
          key = cc >> 4; ch = cc & 15;
          if constexpr (PLUT) pk3_conv16_k4_lut(reinterpret_cast<const unsigned*>(sv + key * PB), ch, plutv, w);
          else pk_conv16_k4(reinterpret_cast<const unsigned*>(sv + key * PB), ch, tabv, w);
        } else {
          const int hf = cc >> 8, l = cc & 31;
          key = 4 * ((cc >> 5) & 7) + ((l >> 1) & 3);
          const int chp = 2 * ((l >> 3) & 3) + (l & 1);                     
          ch = 2 * chp + hf;
          const unsigned* const rw = reinterpret_cast<const unsigned*>(sv + key * PB);
          if (hf == 0) pk_conv16_k5<0>(rw, chp, tabv, w); else pk_conv16_k5<1>(rw, chp, tabv, w);
        }
      } else {
        key = cc >> 4; ch = cc & 15;
        const uint4 u = *reinterpret_cast<const uint4*>(sv + key * D + ch * 16);
        pk_s8x4_bf16(u.x, w[0], w[1]); pk_s8x4_bf16(u.y, w[2], w[3]); pk_s8x4_bf16(u.z, w[4], w[5]); pk_s8x4_bf16(u.w, w[6], w[7]);
      }
      uint16_t* const row = vbuf + key * TL; const int sw = key & 7;
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch) ^ sw)) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(row + 8 * ((2 * ch + 1) ^ sw)) = make_uint4(w[4], w[5], w[6], w[7]);
    }
    for (int c = cid; c < PK_KT; c += NCONV) vsc2[b][c] = sc[c];
  };
  if (0 < ntiles) load_tile(0, early ? 1 : 0);
  pk_commit();
  if (early) { pdl_wait(); pdl_trigger(); if (0 < ntiles) load_tile(0, 2); pk_commit(); }
  #pragma unroll
  for (int p = 1; p < NST; ++p) { if (p < ntiles) load_tile(p, 0); pk_commit(); }
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
  pk_waitn<NST - 1>();                                                  
  __syncthreads();                                                      
  if (!SKEW && 0 < ntiles) convert_v(0);                                 
  const bool busy = __any_sync(0xffffffffu, vrow0 || vrow1);
  float o[NB][4];                                                       
  #pragma unroll
  for (int n = 0; n < NB; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float s[JW][4], c0 = 1.f, c1 = 1.f;                                   
  for (int t = 0; t <= ntiles; ++t) {
    __syncthreads();                                                    
    if (SKEW && grpB && t < ntiles) convert_v(t);                       
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
    if (busy) {                                                         
      const uint8_t* const sk = stg + (size_t)(t % NST) * SREG; const float* const ksc = ssc + (t % NST) * 2 * PK_KT;
      const int kt0 = tbeg + t * PK_KT;
      const unsigned* krow[JW];
      #pragma unroll
      for (int w = 0; w < JW; ++w) { const int j = JW * part + w; krow[w] = reinterpret_cast<const unsigned*>(sk + (size_t)(8 * j + gi) * KP); }
      if constexpr (!QK8) {
        unsigned prev[JW];
        #pragma unroll
        for (int w = 0; w < JW; ++w) { prev[w] = 0u; s[w][0] = s[w][1] = s[w][2] = s[w][3] = 0.f; }
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
          #pragma unroll
          for (int w = 0; w < JW; ++w) {
            unsigned b0, b1;
            if constexpr (PLUT) pk3_kfrag_lut<KB>(krow[w], ks, tig, plutk, prev[w], b0, b1);
            else pk3_kfrag<KB>(krow[w], ks, tig, tabk, prev[w], b0, b1);
            pk_mma(s[w], qa[ks], b0, b1);
          }
        }
      } else {
        int si[JW][4];
        #pragma unroll
        for (int w = 0; w < JW; ++w) si[w][0] = si[w][1] = si[w][2] = si[w][3] = 0;
        #pragma unroll
        for (int ks8 = 0; ks8 < 8; ++ks8) {
          #pragma unroll
          for (int w = 0; w < JW; ++w) {
            unsigned b0, b1;
            pk3_kfrag8<KB>(krow[w], ks8, tig, tab8k, b0, b1);
            pk3_mma_s8(si[w], qa[ks8], b0, b1);
          }
        }
        #pragma unroll
        for (int w = 0; w < JW; ++w) { s[w][0] = (float)si[w][0] * sq0; s[w][1] = (float)si[w][1] * sq0; s[w][2] = (float)si[w][2] * sq1; s[w][3] = (float)si[w][3] * sq1; }
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
    pk_waitn<NST - 2>();                                                
    __syncthreads();                                                    
    if (t + NST < ntiles) load_tile(t + NST, 0);                        
    pk_commit();                                                        
    if (busy) {                                                         
      const float* const vsc = vsc2[t & 1];
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
    if (!SKEW && t + 1 < ntiles) convert_v(t + 1);                       
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
            const float k0 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 2 * tig)), k1 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 2 * tig + 1));
            const float k2 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 8 + 2 * tig)), k3 = pk_tabf(tabk, pk3_state<KB>(kw, 16 * kq + 9 + 2 * tig));
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
            v0 = pk_tabf(tabv, pk3_state<KB>(vw, e)); v1 = pk_tabf(tabv, pk3_state<KB>(vw, e + 1));
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


static int pk3_nst_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_PK3_NST"); const int n = e ? std::atoi(e) : 3; return n >= 2 && n <= 3 ? n : 3; }(); return v; }
static int pk3_skew_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_PK3_SKEW"); return e ? (std::atoi(e) != 0) : 1; }(); return v; }
static int pk3_qk8_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_PK3_QK8"); return e ? (std::atoi(e) != 0) : 0; }(); return v; }
static int pk3_plut_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_PK3_PLUT"); return e ? (std::atoi(e) != 0) : 0; }(); return v; }   
template <int KB, bool VPK, int NST> static void pk3_set_lut(cudaError_t& e) {
  const cudaError_t r = cudaFuncSetAttribute((const void*)q35_atxpk3_kernel<KB, VPK, 1, 4, NST, true, 4, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pk3_smem<KB, VPK, 1, 4, NST, true, true>());
  if (r != cudaSuccess && e == cudaSuccess) e = r;
}
template <int KB, bool VPK, int NST> static void pk3_go_lut(const Q35AtxPkArgs& U, cudaStream_t st) {
  const int M = U.a.M, sp = U.a.splits;
  launch_k(q35_atxpk3_kernel<KB, VPK, 1, 4, NST, true, 4, false, true>, dim3((M + 1) / 2, sp, Q35_KVH), dim3(256), pk3_smem<KB, VPK, 1, 4, NST, true, true>(), st, U);
}
template <int KB, bool VPK, int NST, bool SKEW, bool QK8> static void pk3_set(cudaError_t& e) {
  auto set = [&](const void* f, size_t smem) { const cudaError_t r = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); if (r != cudaSuccess && e == cudaSuccess) e = r; };
  set((const void*)q35_atxpk3_kernel<KB, VPK, 1, 4, NST, SKEW, SKEW ? 4 : 0, QK8>, pk3_smem<KB, VPK, 1, 4, NST, SKEW>());
  set((const void*)q35_atxpk3_kernel<KB, VPK, 3, 4, NST, SKEW, 0, QK8>, pk3_smem<KB, VPK, 3, 4, NST, SKEW>());
  set((const void*)q35_atxpk3_kernel<KB, VPK, 6, 2, NST, SKEW, 0, QK8>, pk3_smem<KB, VPK, 6, 2, NST, SKEW>());
}
template <int KB, bool VPK, int NST, bool SKEW, bool QK8> static void pk3_go(const Q35AtxPkArgs& U, cudaStream_t st) {
  const int M = U.a.M, sp = U.a.splits;
  if (M <= 2) launch_k(q35_atxpk3_kernel<KB, VPK, 1, 4, NST, SKEW, SKEW ? 4 : 0, QK8>, dim3((M + 1) / 2, sp, Q35_KVH), dim3(SKEW ? 256 : 128), pk3_smem<KB, VPK, 1, 4, NST, SKEW>(), st, U);
  else if (M <= 8) launch_k(q35_atxpk3_kernel<KB, VPK, 3, 4, NST, SKEW, 0, QK8>, dim3((M + 7) / 8, sp, Q35_KVH), dim3(384), pk3_smem<KB, VPK, 3, 4, NST, SKEW>(), st, U);
  else launch_k(q35_atxpk3_kernel<KB, VPK, 6, 2, NST, SKEW, 0, QK8>, dim3((M + 15) / 16, sp, Q35_KVH), dim3(384), pk3_smem<KB, VPK, 6, 2, NST, SKEW>(), st, U);
}
template <int KB, bool VPK, bool QK8> static void pk3_launch2(const Q35AtxPkArgs& U, cudaStream_t st) {
  const int nst = pk3_nst_env(); const bool skew = pk3_skew_env() != 0;
  if (nst == 2) { if (skew) pk3_go<KB, VPK, 2, true, QK8>(U, st); else pk3_go<KB, VPK, 2, false, QK8>(U, st); }
  else { if (skew) pk3_go<KB, VPK, 3, true, QK8>(U, st); else pk3_go<KB, VPK, 3, false, QK8>(U, st); }
}
template <int KB, bool VPK> static void pk3_launch(const Q35AtxPkArgs& U, cudaStream_t st) {
  if (pk3_qk8_env()) { pk3_launch2<KB, VPK, true>(U, st); return; }
  if (pk3_plut_env() && U.a.M <= 2) {                                                   
    if constexpr (KB == 5) pk3_go_lut<KB, VPK, 2>(U, st);
    else { if (pk3_nst_env() == 2) pk3_go_lut<KB, VPK, 2>(U, st); else pk3_go_lut<KB, VPK, 3>(U, st); }
    return;
  }
  pk3_launch2<KB, VPK, false>(U, st);
}
template <int KB, bool VPK, bool QK8> static void pk3_set_kv(cudaError_t& e) { pk3_set<KB, VPK, 2, true, QK8>(e); pk3_set<KB, VPK, 2, false, QK8>(e); pk3_set<KB, VPK, 3, true, QK8>(e); pk3_set<KB, VPK, 3, false, QK8>(e); }
static void pk3_set_all(cudaError_t& e) {
  pk3_set_kv<4, true, false>(e); pk3_set_kv<4, false, false>(e); pk3_set_kv<5, true, false>(e); pk3_set_kv<5, false, false>(e);
  pk3_set_kv<4, true, true>(e); pk3_set_kv<4, false, true>(e); pk3_set_kv<5, true, true>(e); pk3_set_kv<5, false, true>(e);
  pk3_set_lut<4, true, 2>(e); pk3_set_lut<4, false, 2>(e); pk3_set_lut<4, true, 3>(e); pk3_set_lut<4, false, 3>(e); pk3_set_lut<5, true, 2>(e); pk3_set_lut<5, false, 2>(e);
}
