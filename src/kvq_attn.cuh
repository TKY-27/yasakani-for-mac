// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once















#include "kvq_gpu.cuh"
#include <cuda_fp16.h>
#include <stdint.h>
namespace yskn {
constexpr int TBA_NH = 24, TBA_KVH = 4, TBA_G = 6, TBA_HD = 256;
struct TbGP { int mode, kk, tix, guard, m, has_mu; unsigned go; };
__device__ __forceinline__ TbGP tba_group(const unsigned char* b, int g) {
  const unsigned off_g = tb_u32(b + 40);
  TbGP P; P.go = tb_u32(b + off_g + 4u * (unsigned)g);
  P.mode = b[P.go]; P.kk = b[P.go + 1]; P.tix = b[P.go + 2]; P.guard = b[P.go + 3]; P.m = b[P.go + 4]; P.has_mu = b[P.go + 5];
  return P;
}
__device__ __forceinline__ float tba_h2f(unsigned h) { return __half2float(__ushort_as_half((unsigned short)h)); }

__device__ __forceinline__ unsigned tba_bits(const unsigned* wp, int nwords, int pos, int n) {
  const int wi = pos >> 5, o = pos & 31;
  const unsigned long long Z = ((unsigned long long)wp[wi] << 32) | (unsigned long long)((wi + 1 < nwords) ? wp[wi + 1] : 0u);
  return (unsigned)((Z >> (64 - o - n)) & ((1ull << n) - 1ull));
}

__global__ void __launch_bounds__(256) tba_unpack_kernel(const unsigned char* blob, const int* grp, const int* pos, const unsigned char* pay, const unsigned* side, int stride, const float* yin, float* y) {
  __shared__ float tab[256];
  const int row = blockIdx.x, d = threadIdx.x;
  const int g = grp[row];
  float out = yin[(size_t)row * 256 + d];                                    
  if ((unsigned)g < tb_u32(blob + 28)) {
    const TbGP P = tba_group(blob, g);
    const bool packed = pos[row] >= P.guard && P.mode <= 1;                  
    if (packed) {
      const unsigned* wp = reinterpret_cast<const unsigned*>(pay + (size_t)row * stride);
      const float s = __uint_as_float(side[2 * (size_t)row]); const unsigned pc = side[2 * (size_t)row + 1];
      const int kb = P.mode == 1 ? 4 : P.kk, nw = 8 * kb;
      float xq;
      if (P.mode == 1) {
        const int nib = (int)tba_bits(wp, nw, 4 * d, 4);
        xq = __fmul_rn((float)(nib - 8), s);
      } else {
        const unsigned to = tb_u32(blob + tb_u32(blob + 36) + 4u * (unsigned)P.tix);
        tab[d] = tb_cdec((int)(short)tb_u16(blob + to + 12 + 2 * d), tb_f32(blob + to + 4));
        __syncthreads();
        const int endb = kb * (d + 1), first = kb * (d & ~63), lo = max(endb - 8, first);
        const unsigned state = tba_bits(wp, nw, lo, endb - lo);
        xq = __fmul_rn(tab[state], s);
      }
      float x = xq;
      const unsigned char* mup = blob + P.go + 8; const unsigned char* Up = mup + (P.has_mu ? 512 : 0);
      if (P.m > 0) x = __fadd_rn(x, __fmul_rn(tba_h2f(pc & 0xffffu), tba_h2f(tb_u16(Up + 2 * d))));
      if (P.m > 1) x = __fadd_rn(x, __fmul_rn(tba_h2f(pc >> 16), tba_h2f(tb_u16(Up + 2 * (256 + d)))));
      if (P.has_mu) x = __fadd_rn(x, tba_h2f(tb_u16(mup + 2 * d)));
      out = x;
    }
  }
  y[(size_t)row * 256 + d] = out;
}


template <int KB> struct TbLaneDec {
  static constexpr int NW = (KB == 4) ? 1 : 3;
  __device__ __forceinline__ static void load(const unsigned* __restrict__ row, int lane, unsigned (&w)[NW]) {
    if constexpr (KB == 4) { w[0] = __ldcg(row + lane); }
    else {
      const int fs = 8 * KB * lane; const bool cs = (lane & 7) == 0; const int ws = cs ? fs : fs - (8 - KB); const int wi = ws >> 5;
      w[0] = __ldcg(row + wi); w[1] = (wi + 1 < 8 * KB) ? __ldcg(row + wi + 1) : 0u; w[2] = (wi + 2 < 8 * KB) ? __ldcg(row + wi + 2) : 0u;
    }
  }
  
  template <bool BFE>
  __device__ __forceinline__ static void states(const unsigned (&w)[NW], int lane, bool flat, int (&st)[8]) {
    if constexpr (KB == 4) {
      if (BFE && !flat) {
        const unsigned pv = __shfl_up_sync(0xffffffffu, w[0], 1);
        const unsigned pn = (lane & 7) ? (pv & 15u) : 0u;                    
        st[0] = (int)((pn << 4) | (w[0] >> 28));
        #pragma unroll
        for (int e = 1; e < 8; ++e) st[e] = (int)((w[0] >> (28 - 4 * e)) & 255u);
      } else {
        #pragma unroll
        for (int e = 0; e < 8; ++e) st[e] = (int)((w[0] >> (28 - 4 * e)) & 15u);
        if (!flat) {
          const unsigned pv = __shfl_up_sync(0xffffffffu, w[0], 1);
          int prevf = (lane & 7) ? (int)(pv & 15u) : 0;                      
          #pragma unroll
          for (int e = 0; e < 8; ++e) { const int f = st[e]; st[e] = (prevf << 4) | f; prevf = f; }
        }
      }
    } else {
      const int fs = 8 * KB * lane; const bool cs = (lane & 7) == 0; const int ws = cs ? fs : fs - (8 - KB); const int off = ws & 31;
      unsigned long long Z = ((unsigned long long)w[0] << 32) | (unsigned long long)w[1];
      Z = (Z << off) | (off ? (unsigned long long)(w[2] >> (32 - off)) : 0ull);
      if (cs) Z >>= (8 - KB);
      #pragma unroll
      for (int e = 0; e < 8; ++e) st[e] = (int)((Z >> (56 - KB * e)) & 255ull);
      (void)flat;
    }
  }
};

struct TbAttnArgs {
  const unsigned char* blob; int layer;
  const uint16_t* q16;                                                       
  const unsigned char* kpay; const unsigned char* vpay;                      
  const unsigned* kside; const unsigned* vside;                              
  int max_ctx, M, splits;
  const int* d_pos;                                                          
  float* pm; float* pl; float* pacc;                                         
};


template <int KB, int PF, int VAR>                                           
__global__ void __launch_bounds__(128) tba_attention_kernel(TbAttnArgs A) {
  constexpr int G = TBA_G, NW = TbLaneDec<KB>::NW, PW = 8 * KB;
  constexpr bool FMAK = (VAR & 1) != 0, LAZY = (VAR & 2) != 0, BFEV = (VAR & 4) != 0, FEXP = (VAR & 32) != 0;
  constexpr int TR = (VAR >> 3) & 3, TC = 1 << TR;
  __shared__ float sm[4][G], sl[4][G], sx[4][G][2], sacc[4][G][TBA_HD];
  __shared__ float tabK[256 * TC], tabV[256 * TC];
  const int kvh = blockIdx.x, split = blockIdx.y, tok = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* const blob = A.blob;
  const TbGP PK = tba_group(blob, (A.layer * 4 + kvh) * 2), PV = tba_group(blob, (A.layer * 4 + kvh) * 2 + 1);
  const bool flatK = PK.mode == 1, flatV = PV.mode == 1;
  const bool offK = (PK.has_mu != 0) || (PK.m > 0), hasV = PV.m > 0;
  {                                                                          
    const unsigned off_t = tb_u32(blob + 36);
    const unsigned toK = flatK ? 0u : tb_u32(blob + off_t + 4u * (unsigned)PK.tix), toV = flatV ? 0u : tb_u32(blob + off_t + 4u * (unsigned)PV.tix);
    for (int idx = threadIdx.x; idx < 256 * TC; idx += 128) {
      const int i = idx >> TR;
      tabK[idx] = flatK ? (float)(i < 16 ? i - 8 : 0) : tb_cdec((int)(short)tb_u16(blob + toK + 12 + 2 * i), tb_f32(blob + toK + 4));
      tabV[idx] = flatV ? (float)(i < 16 ? i - 8 : 0) : tb_cdec((int)(short)tb_u16(blob + toV + 12 + 2 * i), tb_f32(blob + toV + 4));
    }
  }
  __syncthreads();
  const unsigned* const kb_ = reinterpret_cast<const unsigned*>(A.kpay) + (size_t)kvh * A.max_ctx * PW;
  const unsigned* const vb_ = reinterpret_cast<const unsigned*>(A.vpay) + (size_t)kvh * A.max_ctx * PW;
  const uint2* const ksb = reinterpret_cast<const uint2*>(A.kside) + (size_t)kvh * A.max_ctx;
  const uint2* const vsb = reinterpret_cast<const uint2*>(A.vside) + (size_t)kvh * A.max_ctx;
  auto ld = [&](unsigned (&kw)[NW], unsigned (&vw)[NW], uint2& ks, uint2& vs, int t) {
    TbLaneDec<KB>::load(kb_ + (size_t)t * PW, lane, kw); TbLaneDec<KB>::load(vb_ + (size_t)t * PW, lane, vw);
    ks = __ldg(ksb + t); vs = __ldg(vsb + t);                                 
  };
  unsigned kq[PF][NW], vq[PF][NW]; uint2 ksq[PF], vsq[PF];
  #pragma unroll
  for (int i = 0; i < PF; ++i) {
    #pragma unroll
    for (int w = 0; w < NW; ++w) { kq[i][w] = 0u; vq[i][w] = 0u; }
    ksq[i] = make_uint2(0u, 0u); vsq[i] = make_uint2(0u, 0u);
  }
  const int base = *A.d_pos;
  const int pos = base + tok, len = pos + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);
  #pragma unroll
  for (int i = 0; i < PF; ++i) { const int t = t0 + warp + 4 * i; if (t < t1) ld(kq[i], vq[i], ksq[i], vsq[i], t); }
  float q[G][8];
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    const uint4 u = *reinterpret_cast<const uint4*>(A.q16 + ((size_t)tok * TBA_NH + kvh * G + i) * TBA_HD + 8 * lane);
    const unsigned uu[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int e = 0; e < 4; ++e) { q[i][2 * e] = __fmul_rn(__uint_as_float(uu[e] << 16), 0.0625f); q[i][2 * e + 1] = __fmul_rn(__uint_as_float(uu[e] & 0xffff0000u), 0.0625f); }
  }
  
  float qmu_l = 0.f, qu0_l = 0.f, qu1_l = 0.f;
  if (offK) {
    const unsigned char* mup = blob + PK.go + 8; const unsigned char* Up = mup + (PK.has_mu ? 512 : 0);
    float dm[G], d0[G], d1[G];
    #pragma unroll
    for (int i = 0; i < G; ++i) { dm[i] = 0.f; d0[i] = 0.f; d1[i] = 0.f; }
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float mu = PK.has_mu ? tba_h2f(tb_u16(mup + 2 * (8 * lane + e))) : 0.f;
      const float u0 = PK.m > 0 ? tba_h2f(tb_u16(Up + 2 * (8 * lane + e))) : 0.f;
      const float u1 = PK.m > 1 ? tba_h2f(tb_u16(Up + 2 * (256 + 8 * lane + e))) : 0.f;
      #pragma unroll
      for (int i = 0; i < G; ++i) { dm[i] += q[i][e] * mu; d0[i] += q[i][e] * u0; d1[i] += q[i][e] * u1; }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
      #pragma unroll
      for (int i = 0; i < G; ++i) { dm[i] += __shfl_xor_sync(0xffffffffu, dm[i], o); d0[i] += __shfl_xor_sync(0xffffffffu, d0[i], o); d1[i] += __shfl_xor_sync(0xffffffffu, d1[i], o); }
    }
    #pragma unroll
    for (int i = 0; i < G; ++i) if ((lane >> 2) == i) { qmu_l = dm[i]; qu0_l = d0[i]; qu1_l = d1[i]; }
  }
  float mh = -1e30f, lh = 0.f, acc[G][8], x0[G], x1[G];
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    x0[i] = 0.f; x1[i] = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }
  for (int tb = t0 + warp; tb < t1; tb += 4 * PF) {
    unsigned kn[PF][NW], vn[PF][NW]; uint2 ksn[PF], vsn[PF];
    #pragma unroll
    for (int i = 0; i < PF; ++i) {
      const int t = tb + 4 * PF + 4 * i;
      #pragma unroll
      for (int w = 0; w < NW; ++w) { kn[i][w] = 0u; vn[i][w] = 0u; }
      ksn[i] = make_uint2(0u, 0u); vsn[i] = make_uint2(0u, 0u);
      if (t < t1) ld(kn[i], vn[i], ksn[i], vsn[i], t);
    }
    #pragma unroll
    for (int i = 0; i < PF; ++i) {
      if (tb + 4 * i >= t1) break;
      int stk[8], stv[8];
      TbLaneDec<KB>::template states<BFEV>(kq[i], lane, flatK, stk); TbLaneDec<KB>::template states<BFEV>(vq[i], lane, flatV, stv);
      float kd[8], vd[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) { kd[e] = tabK[(stk[e] << TR) + (lane & (TC - 1))]; vd[e] = tabV[(stv[e] << TR) + (lane & (TC - 1))]; }
      const float sK = __uint_as_float(ksq[i].x), sV = __uint_as_float(vsq[i].x);
      float p[8];
      #pragma unroll
      for (int h = 0; h < G; ++h) { float s = __fmul_rn(q[h][0], kd[0]);
        #pragma unroll
        for (int e = 1; e < 8; ++e) { if constexpr (FMAK) s = fmaf(q[h][e], kd[e], s); else s = __fadd_rn(s, __fmul_rn(q[h][e], kd[e])); }
        p[h] = s; }
      #pragma unroll
      for (int h = G; h < 8; ++h) p[h] = 0.f;
      const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
      float v4[4], v2[2], v1;
      #pragma unroll
      for (int k = 0; k < 4; ++k) { const float send = b4 ? p[k] : p[k + 4], keep = b4 ? p[k + 4] : p[k]; v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16); }
      #pragma unroll
      for (int k = 0; k < 2; ++k) { const float send = b3 ? v4[k] : v4[k + 2], keep = b3 ? v4[k + 2] : v4[k]; v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8); }
      { const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0]; v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4); }
      v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
      v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
      float sc = __fmul_rn(v1, sK);
      if (offK) sc = sc + qmu_l + tba_h2f(ksq[i].y & 0xffffu) * qu0_l + tba_h2f(ksq[i].y >> 16) * qu1_l;
      const float mn = fmaxf(mh, sc);
      const bool gt = sc > mh;
      const float dlt = (gt ? mh : sc) - mn;
      const float ee = FEXP ? __expf(dlt) : expf(dlt);
      const float a = gt ? ee : 1.f, b = gt ? 1.f : ee;
      lh = lh * a + b; mh = mn;
      const float c0V = hasV ? tba_h2f(vsq[i].y & 0xffffu) : 0.f, c1V = hasV ? tba_h2f(vsq[i].y >> 16) : 0.f;
      bool rescale = true;
      if constexpr (LAZY) rescale = __any_sync(0xffffffffu, gt && lane < 4 * G);   
      if (rescale) {
        #pragma unroll
        for (int h = 0; h < G; ++h) {
          const float ai = __shfl_sync(0xffffffffu, a, 4 * h), bu = __shfl_sync(0xffffffffu, b, 4 * h), bi = __fmul_rn(bu, sV);
          #pragma unroll
          for (int e = 0; e < 8; ++e) acc[h][e] = acc[h][e] * ai + bi * vd[e];
          if (hasV) { x0[h] = x0[h] * ai + bu * c0V; x1[h] = x1[h] * ai + bu * c1V; }
        }
      } else {
        #pragma unroll
        for (int h = 0; h < G; ++h) {
          const float bu = __shfl_sync(0xffffffffu, b, 4 * h), bi = __fmul_rn(bu, sV);
          #pragma unroll
          for (int e = 0; e < 8; ++e) acc[h][e] = fmaf(bi, vd[e], acc[h][e]);
          if (hasV) { x0[h] += bu * c0V; x1[h] += bu * c1V; }
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < PF; ++i) {
      #pragma unroll
      for (int w = 0; w < NW; ++w) { kq[i][w] = kn[i][w]; vq[i][w] = vn[i][w]; }
      ksq[i] = ksn[i]; vsq[i] = vsn[i];
    }
  }
  if ((lane & 3) == 0 && lane < 4 * G) { sm[warp][lane >> 2] = mh; sl[warp][lane >> 2] = lh; }
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < G; ++i) { sx[warp][i][0] = x0[i]; sx[warp][i][1] = x1[i]; }
  }
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) sacc[warp][i][8 * lane + e] = acc[i][e];
  }
  __syncthreads();
  const unsigned char* const mupV = blob + PV.go + 8; const unsigned char* const UpV = mupV + (PV.has_mu ? 512 : 0);
  for (int i = warp; i < G; i += 4) {
    float M2 = sm[0][i];
    for (int w = 1; w < 4; ++w) M2 = fmaxf(M2, sm[w][i]);
    float L2 = 0.f, X0 = 0.f, X1 = 0.f, o8[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) o8[e] = 0.f;
    for (int w = 0; w < 4; ++w) {
      const float sc = (sm[w][i] > -1e29f) ? expf(sm[w][i] - M2) : 0.f;
      L2 += sc * sl[w][i]; X0 += sc * sx[w][i][0]; X1 += sc * sx[w][i][1];
      #pragma unroll
      for (int e = 0; e < 8; ++e) o8[e] += sc * sacc[w][i][8 * lane + e];
    }
    #pragma unroll
    for (int e = 0; e < 8; ++e) {                                           
      if (PV.has_mu) o8[e] += L2 * tba_h2f(tb_u16(mupV + 2 * (8 * lane + e)));
      if (PV.m > 0) o8[e] += X0 * tba_h2f(tb_u16(UpV + 2 * (8 * lane + e)));
      if (PV.m > 1) o8[e] += X1 * tba_h2f(tb_u16(UpV + 2 * (256 + 8 * lane + e)));
    }
    const size_t pidx = ((size_t)tok * A.splits + split) * TBA_NH + kvh * G + i;
    if (lane == 0) { A.pm[pidx] = M2; A.pl[pidx] = L2; }
    #pragma unroll
    for (int e = 0; e < 8; ++e) A.pacc[pidx * TBA_HD + 8 * lane + e] = o8[e];
  }
}

__global__ void tba_combine_kernel(const float* pm, const float* pl, const float* pacc, int splits, float* out) {
  const int h = blockIdx.x, d = threadIdx.x;
  float M2 = -1e30f;
  for (int s = 0; s < splits; ++s) M2 = fmaxf(M2, pm[(size_t)s * TBA_NH + h]);
  float L = 0.f, o = 0.f;
  for (int s = 0; s < splits; ++s) { const float m = pm[(size_t)s * TBA_NH + h]; const float sc = (m > -1e29f) ? expf(m - M2) : 0.f; L += sc * pl[(size_t)s * TBA_NH + h]; o += sc * pacc[((size_t)s * TBA_NH + h) * TBA_HD + d]; }
  out[h * TBA_HD + d] = o / L;
}

__global__ void __launch_bounds__(512) tba_i8_kernel(const float* x, int8_t* q8, float* qs) {
  const int row = blockIdx.x * 2 + (threadIdx.x >> 8), d = threadIdx.x & 255;
  kvx_store(x[(size_t)row * 256 + d], q8, qs, row, d);
}

__global__ void tba_tile_kernel(const unsigned char* src, unsigned char* dst, int U, int N, int max_ctx, int bytes, int src_kvh_stride_rows, int dst_layer_off_rows) {
  const int kvh = blockIdx.y, p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N) return;
  const unsigned* s = reinterpret_cast<const unsigned*>(src + ((size_t)kvh * src_kvh_stride_rows + (p % U)) * bytes);
  unsigned* d = reinterpret_cast<unsigned*>(dst + ((size_t)dst_layer_off_rows + (size_t)kvh * max_ctx + p) * bytes);
  for (int i = 0; i < bytes / 4; ++i) d[i] = s[i];
}
} 
