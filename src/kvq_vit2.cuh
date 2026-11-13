// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once















template <int KK> struct alignas(16) Vit2Smem {                              
  static constexpr int UU = 256 >> KK;
  int mk[2][2][UU];                  
  unsigned char bp[64][2][UU];       
  int xi[2][68];                     
  unsigned char st[2][64];           
};
static_assert(sizeof(Vit2Smem<5>) <= sizeof(Vit2Smem<4>), "shared memory size");

__device__ __forceinline__ int tbf2_min(const int a, const int b) { return a < b ? a : b; }
template <int N> __device__ __forceinline__ int tbf2_tree(int (&v)[N]) {     
  #pragma unroll
  for (int n = N / 2; n > 0; n >>= 1) {
    #pragma unroll
    for (int i = 0; i < n; ++i) v[i] = tbf2_min(v[i], v[i + n]);
  }
  return v[0];
}
__device__ __forceinline__ float tbf2_sum(const float (&v)[8]) {             
  float s[8];
  #pragma unroll
  for (int w = 0; w < 8; ++w) {
    float x = v[w];
    #pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    s[w] = x;
  }
  float t = 0.f;
  #pragma unroll
  for (int w = 0; w < 8; ++w) t = __fadd_rn(t, s[w]);
  return t;
}
__device__ __forceinline__ float tbf2_y(const TbFastArgs& A, const TbRow& R, const int rj, const int d) {    
  if (A.yin) return A.yin[(size_t)(A.row0 + rj) * 256 + d];
  const int tok = A.tok0 + (rj >> 3), kind = (rj >> 2) & 1, kvh = rj & 3;
  const int stg = kind ? A.stg_v : A.stg_k;
  const size_t irow = stg ? (size_t)kvh * stg + tok : R.row;
  return __fmul_rn((float)R.q8[irow * 256 + d], R.qs[irow]);
}

template <int KK> __device__ __forceinline__ void tbf2_loadmk(const int* mp, int (&mv)[256 >> KK]) {    
  #pragma unroll
  for (int q = 0; q < (256 >> KK) / 4; ++q) { const int4 v4 = reinterpret_cast<const int4*>(mp)[q]; mv[4 * q] = v4.x; mv[4 * q + 1] = v4.y; mv[4 * q + 2] = v4.z; mv[4 * q + 3] = v4.w; }
}

template <int KK> __device__ __forceinline__ int tbf2_minkey(const int (&Ac)[16], const int (&Bc)[16], const int X, const int (&mv)[256 >> KK]) {
  constexpr int NK = (KK == 4) ? 16 : 8;
  int kv[NK];
  if constexpr (KK == 4) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) kv[i] = Ac[i] + Bc[i] * X + mv[i];
  } else {
    #pragma unroll
    for (int u = 0; u < 8; ++u) kv[u] = tbf2_min(Ac[2 * u] + Bc[2 * u] * X, Ac[2 * u + 1] + Bc[2 * u + 1] * X) + mv[u];
  }
  int m = tbf2_tree<NK>(kv);
  if constexpr (KK == 5) m = tbf2_min(m, __shfl_xor_sync(0xffffffffu, m, 8));
  return m;
}

template <int KK, int CUR> __device__ __forceinline__ void tbf2_step(Vit2Smem<KK>& W, const int (&Ac)[16], const int (&Bc)[16], const int t, const int pr, const int lh) {
  constexpr int KQ = 1 << KK, UU = 256 >> KK;
  const int X = W.xi[pr][t];
  int mv[UU];
  tbf2_loadmk<KK>(&W.mk[CUR][pr][0], mv);
  const int m = tbf2_minkey<KK>(Ac, Bc, X, mv);
  const int kf = m + ((X * X) << KK) - mv[0];                                
  const int g = lh & (UU - 1);                                               
  W.mk[CUR ^ 1][pr][g] = kf & ~(KQ - 1);
  W.bp[t + 1][pr][g] = (unsigned char)(kf & (KQ - 1));
  __syncwarp();
}

template <int KK, int CP>                                                    
__device__ __forceinline__ void tbf2_work(const TbFastArgs& A, unsigned char* wsm, const int rj, const int mi, const int lane, const TbG& P, const TbRow& R) {
  using SM = Vit2Smem<KK>;
  constexpr int KQ = 1 << KK, UU = 256 >> KK, SH = 8 - KK;
  SM& W = *reinterpret_cast<SM*>(wsm);
  const unsigned char* const b = A.blob;
  const int lh = lane & 15, pr = lane >> 4;                                  

  
  const unsigned char* const mup = b + P.go + 8;
  const unsigned char* const Up = mup + (P.has_mu ? 512 : 0);
  float r[8];
  #pragma unroll
  for (int w = 0; w < 8; ++w) r[w] = tbf2_y(A, R, rj, 32 * w + lane);
  if (P.has_mu) {
    #pragma unroll
    for (int w = 0; w < 8; ++w) r[w] = __fsub_rn(r[w], tb_h2f(tb_u16(mup + 2 * (32 * w + lane))));
  }
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    if (P.m > j) {
      float u[8], v[8];
      #pragma unroll
      for (int w = 0; w < 8; ++w) { u[w] = tb_h2f(tb_u16(Up + 2 * (256 * j + 32 * w + lane))); v[w] = __fmul_rn(r[w], u[w]); }
      const float cj = tb_f16r(tbf2_sum(v));
      #pragma unroll
      for (int w = 0; w < 8; ++w) r[w] = __fsub_rn(r[w], __fmul_rn(cj, u[w]));
    }
  }
  float rms;
  {
    float v[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) v[w] = __fmul_rn(r[w], r[w]);
    const float ssq = tbf2_sum(v);
    rms = __fsqrt_rn(__fdiv_rn(ssq, 256.f)); if (rms == 0.f) rms = 1.f;
  }

  
  const unsigned to = tb_u32(b + P.off_t + 4u * (unsigned)P.tix);
  const float gain = tb_f32(b + to + 4);
  if (b[to] != 8 || b[to + 2] != 64 || b[to + 3] != 8 || tb_u16(b + 16) != 256) __trap();
  float s = tb_f16r(__fmul_rn(rms, tb_f32(b + P.off_m + 4u * (unsigned)mi))); if (s == 0.f) s = 1.f;
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float t = rintf(__fmul_rn(__fdiv_rn(r[4 * CP + q], s), 256.f));
    W.xi[q >> 1][((q & 1) << 5) + lane] = (int)fminf(fmaxf(t, -2048.f), 2048.f);
  }

  
  
  int Ac[16], Bc[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    int d, jj;
    if (KK == 4) { jj = i; d = 16 * i + lh; }
    else { jj = 4 * (i >> 1) + 2 * (lh >> 3) + (i & 1); d = 8 * jj + (lh & 7); }
    const int c = (int)(short)tb_u16(b + to + 12 + 2 * d);
    Ac[i] = KQ * c * c + jj; Bc[i] = -2 * KQ * c;
    asm volatile("" : "+r"(Ac[i]), "+r"(Bc[i]));                             
  }
  if (lh < UU) W.mk[0][pr][lh] = (lh == 0) ? 0 : (KQ << 24);                 
  __syncwarp();

  
  #pragma unroll 1
  for (int t = 0; t < 62; t += 2) {
    tbf2_step<KK, 0>(W, Ac, Bc, t, pr, lh);
    tbf2_step<KK, 1>(W, Ac, Bc, t + 1, pr, lh);
  }
  tbf2_step<KK, 0>(W, Ac, Bc, 62, pr, lh);

  
  int fin;
  {
    const int X = W.xi[pr][63];
    int mv[UU];
    tbf2_loadmk<KK>(&W.mk[1][pr][0], mv);
    const int m = tbf2_minkey<KK>(Ac, Bc, X, mv);
    const int K = m + ((X * X) << KK);                                       
    const int cg = K >> KK, jj = K & (KQ - 1);
    const int dg = (KK == 4) ? 16 * jj + lh : 8 * jj + (lh & 7);              
    int cm = cg;
    #pragma unroll
    for (int o = 8; o; o >>= 1) cm = tbf2_min(cm, __shfl_xor_sync(0xffffffffu, cm, o));
    int dm = (cg == cm) ? dg : 0x7fffffff;
    #pragma unroll
    for (int o = 8; o; o >>= 1) dm = tbf2_min(dm, __shfl_xor_sync(0xffffffffu, dm, o));
    fin = dm;
  }

  
  if (lh == 0) {
    int sx = fin;
    #pragma unroll 4
    for (int t = 63; t > 0; --t) {
      W.st[pr][t] = (unsigned char)sx;
      const int jj = W.bp[t][pr][sx >> KK];
      sx = (jj << SH) | (sx >> KK);
    }
    W.st[pr][0] = (unsigned char)sx;
  }
  __syncwarp();

  
  unsigned char* const sb = A.scr + ((size_t)rj * A.nm + mi) * 256;
  float* const wb = reinterpret_cast<float*>(A.scr + TBF_W_OFS) + ((size_t)rj * A.nm + mi) * 8;
  #pragma unroll
  for (int pc = 0; pc < 2; ++pc) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int c = 2 * CP + pc;
      const int stt = W.st[pc][32 * h + lane];
      const int cis = (int)(short)tb_u16(b + to + 12 + 2 * stt);
      const float rec = __fmul_rn(tb_cdec(cis, gain), s);
      const float dd = __fsub_rn(r[4 * CP + 2 * pc + h], rec);
      float v = __fmul_rn(dd, dd);
      #pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) wb[2 * c + h] = v;
      sb[64 * c + 32 * h + lane] = (unsigned char)stt;
    }
  }
}

__global__ void __launch_bounds__(128) tbf2_vit_kernel(TbFastArgs A) {
  __shared__ __align__(16) unsigned char vsm[4 * sizeof(Vit2Smem<4>)];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
  const int per_row = A.nm * 2;                                              
  const int wid = blockIdx.x * 4 + wi;
  if (wid >= A.nrow * per_row) return;                                       
  const int rj = wid / per_row, rem = wid - rj * per_row, mi = rem >> 1, cp = rem & 1;
  const unsigned char* const b = A.blob;
  const TbRow R = tbf_row(A, rj, lane);
  const TbG P = tbf_group(b, R.g);
  if (!P.okg || R.pos < P.guard || P.mode != 0) return;                      
  unsigned char* const wsm = vsm + wi * sizeof(Vit2Smem<4>);
  switch (P.kk) {
    case 4: if (cp) tbf2_work<4, 1>(A, wsm, rj, mi, lane, P, R); else tbf2_work<4, 0>(A, wsm, rj, mi, lane, P, R); break;
    case 5: if (cp) tbf2_work<5, 1>(A, wsm, rj, mi, lane, P, R); else tbf2_work<5, 0>(A, wsm, rj, mi, lane, P, R); break;
    default: break;                                                          
  }
}
