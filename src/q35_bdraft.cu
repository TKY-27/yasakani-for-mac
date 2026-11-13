// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include "q35.hpp"
#include "smp_hash.cuh"
#include "a8_prologue.cuh"
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
#include <cuda_bf16.h>
namespace yskn {




namespace {
constexpr int DH = 5120, DHD = 128, DNH = 32, DNKV = 8, DWIN = 2048, DRANK = 256, DTOPK = 16;
__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float block_sum256(float v, float* sh) {      
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
  for (int k = 0; k < (int)(blockDim.x >> 5); ++k) s += sh[k];
  __syncthreads();
  return s;
}
}  

__global__ void __launch_bounds__(256) bdr_rownorm_kernel(const float* __restrict__ x, int n, int ld, float eps, float* __restrict__ r) {
  pdl_wait(); pdl_trigger();
  __shared__ float sh[8]; const float* xr = x + (size_t)blockIdx.x * ld; float acc = 0.f;
  for (int i = threadIdx.x; i < n; i += 256) acc = fmaf(xr[i], xr[i], acc);
  acc = block_sum256(acc, sh);
  if (threadIdx.x == 0) r[blockIdx.x] = rsqrtf(acc / (float)n + eps);
}


__global__ void __launch_bounds__(512) bdr_a8_kernel(const float* __restrict__ x, int ld, int n, const float* __restrict__ r, const uint16_t* __restrict__ w,
                                                     const int8_t* __restrict__ signs, float* __restrict__ xn, int8_t* __restrict__ qx, float* __restrict__ alpha) {
  pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int g = blockIdx.x, m = blockIdx.y, i = g * 512 + threadIdx.x, G = n / 512;
  float v = x[(size_t)m * ld + i];
  if (w) v = r[m] * v * bf2f(w[i]);
  if (xn) xn[(size_t)m * n + i] = v;
  if (threadIdx.x == 0) s_amax = 0u;
  buf[threadIdx.x] = v * (float)signs[i];
  __syncthreads();
  a8_group512_tail(buf, &s_amax, qx + (size_t)m * n + g * 512, alpha + (size_t)m * G + g);
}


__global__ void __launch_bounds__(256) bdr_post_kernel(long long* __restrict__ acc, const float* __restrict__ us, const float* __restrict__ tau, int N,
                                                       float* __restrict__ y, int ldy) {
  pdl_wait(); pdl_trigger();
  const int r = blockIdx.x * 256 + threadIdx.x, m = blockIdx.y;
  if (r >= N) return;
  long long* p = acc + (size_t)m * N + r;
  y[(size_t)m * ldy + r] = __fmul_rn(__fmul_rn(__ll2float_rn(*p), us[m]), tau[r]);
  *p = 0;
}



__global__ void __launch_bounds__(256) bdr_gconv_kernel(const float* __restrict__ x, const float* __restrict__ dyn, int which, const uint16_t* __restrict__ base,
                                                        float* __restrict__ out, int resid) {
  pdl_wait(); pdl_trigger();
  const int j = blockIdx.x * 256 + threadIdx.x, m = blockIdx.y, g = j >> 4;
  const float* d = dyn + (size_t)m * 1280 + which * 640;
  float v = (bf2f(base[j]) + d[g]) * x[(size_t)m * DH + j];
  if (m > 0) v += (bf2f(base[DH + j]) + d[320 + g]) * x[(size_t)(m - 1) * DH + j];
  if (resid) out[(size_t)m * DH + j] += v; else out[(size_t)m * DH + j] = v;
}


__global__ void __launch_bounds__(128) bdr_qk_rope_kernel(float* __restrict__ a, int na, const uint16_t* __restrict__ wa, float* __restrict__ b, int nb,
                                                          const uint16_t* __restrict__ wb, const int* __restrict__ pos, int pos_off, float eps) {
  pdl_wait(); pdl_trigger();
  __shared__ float sh[4]; __shared__ float v[DHD];
  const int m = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  float* row = h < na ? a + ((size_t)m * na + h) * DHD : b + ((size_t)m * nb + (h - na)) * DHD;
  const uint16_t* w = h < na ? wa : wb;
  const float x = row[d]; float s = x * x;
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((d & 31) == 0) sh[d >> 5] = s;
  __syncthreads();
  const float rr = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / (float)DHD + eps);
  v[d] = rr * x * bf2f(w[d]);
  __syncthreads();
  const float p = (float)(*pos + pos_off + m); const int i = d & 63;
  const float inv = exp2f(-(float)(2 * i) / (float)DHD * 23.253496664211536f);   
  float sn, cs; sincosf(p * inv, &sn, &cs);
  row[d] = d < 64 ? v[d] * cs - v[d + 64] * sn : v[d] * cs + v[d - 64] * sn;
}

__global__ void __launch_bounds__(128) bdr_ring_store_kernel(const float* __restrict__ k, const float* __restrict__ v, uint16_t* __restrict__ rk, uint16_t* __restrict__ rv,
                                                             const int* __restrict__ pos, int pos_off) {
  pdl_wait(); pdl_trigger();
  const int m = blockIdx.x, h = blockIdx.y, d = threadIdx.x, slot = (*pos + pos_off + m) & (DWIN - 1);
  const size_t o = ((size_t)h * DWIN + slot) * DHD + d, s = ((size_t)m * DNKV + h) * DHD + d;
  rk[o] = f2bf(k[s]); rv[o] = f2bf(v[s]);
}








constexpr int DCTX = 32, DSPLIT = DCTX + 1, DPART = 132, DCHMAX = (DWIN - 1 + DCTX - 1) / DCTX;   
__global__ void __launch_bounds__(256) bdr_attn_part_kernel(const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
                                                            const uint16_t* __restrict__ rk, const uint16_t* __restrict__ rv, const int* __restrict__ pos, int M,
                                                            float* __restrict__ part) {
  pdl_wait(); pdl_trigger();
  __shared__ uint32_t s_k[DCHMAX][65];                                  
  __shared__ __align__(16) uint16_t s_v[DCHMAX][DHD];
  __shared__ float s_p[8][4][DCHMAX];
  const int g = blockIdx.x, s = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, hq = g * (DNH / DNKV) + (w >> 1), mb = (w & 1) * 4 + (int)blockIdx.z * 8, p = *pos;
  const int k0 = max(0, p - (DWIN - 1)), nctx = p - k0;
  const float scale = 0.08838834764831845f;                             
  float o[4][4], mx[4], l[4];
  #pragma unroll
  for (int r = 0; r < 4; ++r) { o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.f; mx[r] = -INFINITY; l[r] = 0.f; }
  if (s == DCTX) {                                                      
    float qv[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int m = mb + r; float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
      if (m < M) t = *reinterpret_cast<const float4*>(q + ((size_t)m * DNH + hq) * DHD + 4 * lane);
      qv[r][0] = t.x * scale; qv[r][1] = t.y * scale; qv[r][2] = t.z * scale; qv[r][3] = t.w * scale;
    }
    for (int j = 0; j < M; ++j) {
      const size_t b = ((size_t)j * DNKV + g) * DHD + 4 * lane;
      const float4 kf = *reinterpret_cast<const float4*>(kb + b), vf = *reinterpret_cast<const float4*>(vb + b);
      const float kk[4] = {kf.x, kf.y, kf.z, kf.w}, vv[4] = {vf.x, vf.y, vf.z, vf.w};
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        float sc = fmaf(qv[r][0], kk[0], fmaf(qv[r][1], kk[1], fmaf(qv[r][2], kk[2], qv[r][3] * kk[3])));
        #pragma unroll
        for (int of = 16; of; of >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, of);
        if (mb + r < M) {
          const float mn = fmaxf(mx[r], sc), c = __expf(mx[r] - mn), e = __expf(sc - mn);
          l[r] = fmaf(l[r], c, e);
          #pragma unroll
          for (int d = 0; d < 4; ++d) o[r][d] = fmaf(o[r][d], c, e * vv[d]);
          mx[r] = mn;
        }
      }
    }
  } else {
    const int chunk = (nctx + DCTX - 1) / DCTX, j0 = s * chunk, n = max(0, min(nctx, j0 + chunk) - j0);
    for (int i = threadIdx.x; i < n * 16; i += 256) {                  
      const int r = i >> 4, c = i & 15; const size_t src = ((size_t)g * DWIN + ((k0 + j0 + r) & (DWIN - 1))) * DHD + c * 8;
      const uint4 k4 = *reinterpret_cast<const uint4*>(rk + src), v4 = *reinterpret_cast<const uint4*>(rv + src);
      uint32_t* kd = &s_k[r][c * 4]; kd[0] = k4.x; kd[1] = k4.y; kd[2] = k4.z; kd[3] = k4.w;
      *reinterpret_cast<uint4*>(&s_v[r][c * 8]) = v4;
    }
    __syncthreads();
    if (n > 0) {
      float sc[2][4];                                                   
      #pragma unroll
      for (int rd = 0; rd < 2; ++rd) {
        const int j = rd * 32 + lane; float a[4] = {0.f, 0.f, 0.f, 0.f};
        if (j < n) {
          const uint32_t* kr = s_k[j];
          const float2* q2[4];
          #pragma unroll
          for (int r = 0; r < 4; ++r) q2[r] = reinterpret_cast<const float2*>(q + ((size_t)min(mb + r, M - 1) * DNH + hq) * DHD);
          for (int d2 = 0; d2 < DHD / 2; ++d2) {
            const uint32_t kw = kr[d2]; const float ka = __uint_as_float(kw << 16), kc = __uint_as_float(kw & 0xffff0000u);
            #pragma unroll
            for (int r = 0; r < 4; ++r) { const float2 qq = __ldg(q2[r] + d2); a[r] = fmaf(qq.y, kc, fmaf(qq.x, ka, a[r])); }
          }
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int m = mb + r;
          sc[rd][r] = (j < n && m < M && k0 + j0 + j >= p + m - (DWIN - 1)) ? a[r] * scale : -INFINITY;   
        }
      }
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        float mm = fmaxf(sc[0][r], sc[1][r]);
        #pragma unroll
        for (int of = 16; of; of >>= 1) mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, of));
        float e0 = 0.f, e1 = 0.f;
        if (mm != -INFINITY) { e0 = sc[0][r] == -INFINITY ? 0.f : __expf(sc[0][r] - mm); e1 = sc[1][r] == -INFINITY ? 0.f : __expf(sc[1][r] - mm); }
        if (lane < n) s_p[w][r][lane] = e0;
        if (32 + lane < n) s_p[w][r][32 + lane] = e1;
        float sm = e0 + e1;
        #pragma unroll
        for (int of = 16; of; of >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, of);
        mx[r] = mm; l[r] = sm;
      }
      __syncwarp();
      for (int j = 0; j < n; ++j) {                                     
        const uint2 vw = *reinterpret_cast<const uint2*>(&s_v[j][4 * lane]);
        const float vv[4] = {__uint_as_float(vw.x << 16), __uint_as_float(vw.x & 0xffff0000u), __uint_as_float(vw.y << 16), __uint_as_float(vw.y & 0xffff0000u)};
        #pragma unroll
        for (int r = 0; r < 4; ++r) { const float pj = s_p[w][r][j];
          #pragma unroll
          for (int d = 0; d < 4; ++d) o[r][d] = fmaf(pj, vv[d], o[r][d]); }
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int m = mb + r; if (m >= M) continue;
    float* pp = part + (((size_t)s * DNH + hq) * BDR_B + m) * DPART;
    if (lane == 0) { pp[0] = mx[r]; pp[1] = l[r]; }
    *reinterpret_cast<float4*>(pp + 4 + 4 * lane) = make_float4(o[r][0], o[r][1], o[r][2], o[r][3]);
  }
}
__global__ void __launch_bounds__(128) bdr_attn_comb_kernel(const float* __restrict__ part, float* __restrict__ out) {
  pdl_wait(); pdl_trigger();
  const int h = blockIdx.x, m = blockIdx.y, d = threadIdx.x;
  float mx = -INFINITY;
  for (int s = 0; s < DSPLIT; ++s) mx = fmaxf(mx, part[(((size_t)s * DNH + h) * BDR_B + m) * DPART]);
  float L = 0.f, O = 0.f;
  for (int s = 0; s < DSPLIT; ++s) {
    const float* pp = part + (((size_t)s * DNH + h) * BDR_B + m) * DPART; const float ms = pp[0];
    if (ms == -INFINITY) continue;                                      
    const float c = __expf(ms - mx); L = fmaf(pp[1], c, L); O = fmaf(pp[4 + d], c, O);
  }
  out[((size_t)m * DNH + h) * DHD + d] = O / L;
}

__global__ void __launch_bounds__(256) bdr_silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float* __restrict__ s) {
  pdl_wait(); pdl_trigger();
  const int i = blockIdx.x * 256 + threadIdx.x, m = blockIdx.y;
  if (i >= n) return;
  const float a = g[(size_t)m * n + i]; s[(size_t)m * n + i] = a / (1.f + __expf(-a)) * u[(size_t)m * n + i];
}

__global__ void __launch_bounds__(256) bdr_embed_kernel(const uint16_t* __restrict__ emb, const int* __restrict__ gids, float* __restrict__ h) {   
  pdl_wait(); pdl_trigger();
  const int m = blockIdx.x, j = blockIdx.y * 256 + threadIdx.x;
  h[(size_t)m * DH + j] = bf2f(emb[(size_t)gids[m] * DH + j]);
}

__global__ void bdr_block_ids_kernel(const int* __restrict__ toks, int mask, int* __restrict__ ids, int M) {
  pdl_wait(); pdl_trigger();
  if (threadIdx.x < M) ids[threadIdx.x] = threadIdx.x == 0 ? toks[0] : mask;
}

__global__ void __launch_bounds__(256) bdr_gather_feats_kernel(const float* __restrict__ cap, int pmax, int row0, float* __restrict__ x) {
  pdl_wait(); pdl_trigger();
  const int m = blockIdx.x, k = blockIdx.y; const float* src = cap + ((size_t)k * pmax + row0 + m) * DH; float* dst = x + (size_t)m * 5 * DH + (size_t)k * DH;
  for (int j = threadIdx.x; j < DH; j += 256) dst[j] = src[j];
}



constexpr int DTKC = 64, DTKN = 4096;                                    
__global__ void __launch_bounds__(256) bdr_topk_part_kernel(const float* __restrict__ lg, int V, float* __restrict__ cv, int* __restrict__ ci) {
  pdl_wait(); pdl_trigger();
  __shared__ float sx[DTKN]; __shared__ float wv[8]; __shared__ int wi[8];
  const int c = blockIdx.x, row = blockIdx.y, n = (V + DTKC - 1) / DTKC, i0 = c * n, cnt = max(0, min(V, i0 + n) - i0);
  for (int i = threadIdx.x; i < cnt; i += 256) sx[i] = lg[(size_t)row * V + i0 + i];
  __syncthreads();
  for (int t = 0; t < DTOPK; ++t) {
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < cnt; i += 256) { const float x = sx[i]; if (x > bv) { bv = x; bi = i; } }
    for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
    if ((threadIdx.x & 31) == 0) { wv[threadIdx.x >> 5] = bv; wi[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int k = 1; k < 8; ++k) if (wv[k] > wv[0] || (wv[k] == wv[0] && wi[k] < wi[0])) { wv[0] = wv[k]; wi[0] = wi[k]; }
      const size_t o = ((size_t)row * DTKC + c) * DTOPK + t;
      if (wi[0] < cnt) { cv[o] = wv[0]; ci[o] = i0 + wi[0]; sx[wi[0]] = -INFINITY; } else { cv[o] = -INFINITY; ci[o] = 0x7fffffff; }
    }
    __syncthreads();
  }
}


__global__ void __launch_bounds__(32) bdr_topk_fin_kernel(const float* __restrict__ cv, const int* __restrict__ ci, float* __restrict__ val, int* __restrict__ idx) {
  pdl_wait(); pdl_trigger();
  static_assert(DTKC == 64, "two lists per lane");
  const int row = blockIdx.x, lane = threadIdx.x;
  const float* v = cv + (size_t)row * DTKC * DTOPK; const int* id = ci + (size_t)row * DTKC * DTOPK;
  int p0 = 0, p1 = 0; const int c0 = 2 * lane, c1 = 2 * lane + 1;
  for (int t = 0; t < DTOPK; ++t) {
    const float v0 = p0 < DTOPK ? v[c0 * DTOPK + p0] : -INFINITY, v1 = p1 < DTOPK ? v[c1 * DTOPK + p1] : -INFINITY;
    const int i0 = p0 < DTOPK ? id[c0 * DTOPK + p0] : 0x7fffffff, i1 = p1 < DTOPK ? id[c1 * DTOPK + p1] : 0x7fffffff;
    const bool first = v0 > v1 || (v0 == v1 && i0 < i1);
    float bv = first ? v0 : v1; int bi = first ? i0 : i1;
    for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
    if (lane == 0) { val[row * DTOPK + t] = bv; idx[row * DTOPK + t] = bi; }
    if (first ? (i0 == bi && v0 == bv) : (i1 == bi && v1 == bv)) { if (first) ++p0; else ++p1; }   
  }
}

__global__ void __launch_bounds__(256) bdr_hproj_kernel(const float* __restrict__ hf, const uint16_t* __restrict__ w, float* __restrict__ hp) {
  pdl_wait(); pdl_trigger();
  const int r = blockIdx.x * 8 + (threadIdx.x >> 5), j = blockIdx.y, lane = threadIdx.x & 31;
  const float* x = hf + (size_t)j * DH; const uint16_t* wr = w + (size_t)r * DH; float s = 0.f;
  for (int i = lane; i < DH; i += 32) s = fmaf(x[i], bf2f(wr[i]), s);
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) hp[(size_t)j * DRANK + r] = s;
}

__device__ __forceinline__ unsigned long long bdr_mix(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
__device__ __forceinline__ double bdr_uniform(unsigned long long seed, long long pos) { return (double)(bdr_mix(bdr_mix(seed) ^ (unsigned long long)pos) >> 11) * (1.0 / 9007199254740992.0); }







__global__ void __launch_bounds__(32 * DTOPK) bdr_select_kernel(const float* __restrict__ hp, const float* __restrict__ val, const int* __restrict__ idx,
                                                                const uint16_t* __restrict__ pc, const uint16_t* __restrict__ sc, int* __restrict__ toks, int* __restrict__ out,
                                                                const Q35SmpParams* __restrict__ prm, const int* __restrict__ cpl, const int* __restrict__ pos,
                                                                int* __restrict__ alt, float* __restrict__ marg, int* __restrict__ alt3, int nj) {
  pdl_wait(); pdl_trigger();
  __shared__ float s_a[DRANK]; __shared__ float s_score[DTOPK]; __shared__ int s_pred;
  __shared__ double s_g[BDR_J][DTOPK], s_e[DTOPK], s_z[DTOPK], s_val[DTOPK], s_pre[DTOPK], s_tot; __shared__ int s_ord[DTOPK], s_rk[DTOPK];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, cp = *cpl;
  if (threadIdx.x == 0) s_pred = toks[0];
  if (cp > 0 && threadIdx.x < nj * DTOPK) {                              
    const int jj = threadIdx.x / DTOPK; s_g[jj][threadIdx.x % DTOPK] = kh_gumbel(prm->seed, (long long)*pos + 1 + jj, idx[threadIdx.x], prm->pm); }
  __syncthreads();
  for (int j = 0; j < nj; ++j) {
    if (threadIdx.x < DRANK) s_a[threadIdx.x] = bf2f(pc[(size_t)s_pred * DRANK + threadIdx.x]) * hp[(size_t)j * DRANK + threadIdx.x];
    __syncthreads();
    const uint16_t* scr = sc + (size_t)idx[j * DTOPK + w] * DRANK; float sm = 0.f;
    for (int r = lane; r < DRANK; r += 32) sm = fmaf(s_a[r], bf2f(scr[r]), sm);
    for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
    if (lane == 0) s_score[w] = val[j * DTOPK + w] + sm;
    __syncthreads();
    if (cp > 0) {                                                       
      const double td = 1e-3 * cp; const int iw = idx[j * DTOPK + w];
      if (lane == 0) {                                                  
        int rk = 0; for (int k = 0; k < DTOPK; ++k) { const int ik = idx[j * DTOPK + k]; rk += (s_score[k] > s_score[w] || (s_score[k] == s_score[w] && ik < iw)) ? 1 : 0; }
        s_ord[rk] = w; s_rk[w] = rk; }
      __syncthreads();
      
      
      
      if (w == 0 && lane < DTOPK) { const double z = ((double)s_score[lane] - (double)s_score[s_ord[0]]) / td; s_z[lane] = z; s_e[lane] = exp(z); }
      __syncthreads();
      if (threadIdx.x == 0) {                                           
        double tot = 0.0; for (int i = 0; i < DTOPK; ++i) { s_pre[i] = tot; tot += s_e[s_ord[i]]; } s_tot = tot; }
      __syncthreads();
      if (w == 0 && lane < DTOPK) s_val[lane] = s_pre[s_rk[lane]] < 0.95 * s_tot ? s_z[lane] + s_g[j][lane] : -INFINITY;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      int b = 0; for (int k = 1; k < DTOPK; ++k) if (s_score[k] > s_score[b]) b = k;
      if (cp > 0) {                                                     
        double best = -INFINITY; for (int i = 0; i < DTOPK; ++i) { const int c = s_ord[i]; if (s_val[c] > best) { best = s_val[c]; b = c; } }
      }
      const int t = idx[j * DTOPK + b]; s_pred = t; toks[1 + j] = t; if (out) out[j] = t;
      if (alt) {                                                        
        int b2 = -1;
        int b3 = -1;
        if (cp > 0) { double v2 = -INFINITY; for (int i = 0; i < DTOPK; ++i) { const int c = s_ord[i]; if (c != b && s_val[c] > v2) { v2 = s_val[c]; b2 = c; } }
                      const bool ok = b2 >= 0 && v2 > -INFINITY; alt[j] = ok ? idx[j * DTOPK + b2] : -1; marg[j] = ok ? (float)(s_val[b] - v2) : 1e30f;
                      double v3 = -INFINITY; for (int i = 0; i < DTOPK; ++i) { const int c = s_ord[i]; if (c != b && c != b2 && s_val[c] > v3) { v3 = s_val[c]; b3 = c; } }
                      if (alt3) alt3[j] = ok && b3 >= 0 && v3 > -INFINITY ? idx[j * DTOPK + b3] : -1; }
        else { float v2 = -INFINITY; for (int k = 0; k < DTOPK; ++k) if (k != b && s_score[k] > v2) { v2 = s_score[k]; b2 = k; }
               alt[j] = b2 >= 0 ? idx[j * DTOPK + b2] : -1; marg[j] = b2 >= 0 ? s_score[b] - v2 : 1e30f;
               float v3 = -INFINITY; for (int k = 0; k < DTOPK; ++k) if (k != b && k != b2 && s_score[k] > v3) { v3 = s_score[k]; b3 = k; }
               if (alt3) alt3[j] = b2 >= 0 && b3 >= 0 ? idx[j * DTOPK + b3] : -1; }
      } }
    __syncthreads();
  }
}




__device__ __forceinline__ void dsel_first(float& v, int& k) {           
  #pragma unroll
  for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, v, o); const int ok = __shfl_xor_sync(0xffffffffu, k, o);
    if (ov > v || (ov == v && ok < k)) { v = ov; k = ok; } } }
__device__ __forceinline__ void dsel_first(double& v, int& k) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) { const double ov = __shfl_xor_sync(0xffffffffu, v, o); const int ok = __shfl_xor_sync(0xffffffffu, k, o);
    if (ov > v || (ov == v && ok < k)) { v = ov; k = ok; } } }
__global__ void __launch_bounds__(32 * DTOPK) bdr_select3_kernel(const float* __restrict__ hp, const float* __restrict__ val, const int* __restrict__ idx,
                                                                 const uint16_t* __restrict__ pc, const uint16_t* __restrict__ sc, int* __restrict__ toks, int* __restrict__ out,
                                                                 const Q35SmpParams* __restrict__ prm, const int* __restrict__ cpl, const int* __restrict__ pos,
                                                                 int* __restrict__ alt, float* __restrict__ marg, int* __restrict__ alt3, int nj) {
  pdl_wait(); pdl_trigger();
  __shared__ float s_a[DRANK]; __shared__ float s_score[DTOPK]; __shared__ int s_pred;
  __shared__ double s_g[BDR_J][DTOPK], s_z[DTOPK], s_e[DTOPK], s_val[DTOPK], s_pre[DTOPK + 1]; __shared__ int s_ord[DTOPK];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, cp = *cpl;
  for (int q = threadIdx.x; q < ((2 * nj - 1) * DTOPK + 1) * (DRANK * 2 / 128); q += blockDim.x) {   
    const int row = q / (DRANK * 2 / 128), part = q % (DRANK * 2 / 128);
    const uint16_t* b = row < nj * DTOPK ? sc + (size_t)idx[row] * DRANK : pc + (size_t)(row == nj * DTOPK ? toks[0] : idx[row - nj * DTOPK - 1]) * DRANK;
    asm volatile("prefetch.global.L1 [%0];" :: "l"(b + part * 64)); }
  if (threadIdx.x == 0) s_pred = toks[0];
  if (cp > 0 && threadIdx.x < nj * DTOPK) {                              
    const int jj = threadIdx.x / DTOPK; s_g[jj][threadIdx.x % DTOPK] = kh_gumbel(prm->seed, (long long)*pos + 1 + jj, idx[threadIdx.x], prm->pm); }
  __syncthreads();
  const double td = 1e-3 * cp;
  for (int j = 0; j < nj; ++j) {
    if (threadIdx.x < DRANK) s_a[threadIdx.x] = bf2f(pc[(size_t)s_pred * DRANK + threadIdx.x]) * hp[(size_t)j * DRANK + threadIdx.x];
    __syncthreads();
    { const uint16_t* scr = sc + (size_t)idx[j * DTOPK + w] * DRANK; float sm = 0.f;
      for (int r = lane; r < DRANK; r += 32) sm = fmaf(s_a[r], bf2f(scr[r]), sm);
      for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
      if (lane == 0) s_score[w] = val[j * DTOPK + w] + sm; }
    __syncthreads();
    if (w == 0) {
      const bool in = lane < DTOPK;
      const float sl = in ? s_score[lane] : -INFINITY; const int il = in ? idx[j * DTOPK + lane] : 0x7fffffff;
      float bv0 = sl; int b = in ? lane : 99; dsel_first(bv0, b);          
      int b2 = -1, b3 = -1; double v2d = -INFINITY, v3d = -INFINITY; float v2f = -INFINITY;
      if (cp > 0) {
        int rk = 0;                                                       
        #pragma unroll
        for (int k = 0; k < DTOPK; ++k) { const float sk = s_score[k]; const int ik = __shfl_sync(0xffffffffu, il, k);
                                          rk += (sk > sl || (sk == sl && ik < il)) ? 1 : 0; }
        if (in) s_ord[rk] = lane;
        __syncwarp();
        const int top = s_ord[0];
        if (in) { const double z = ((double)sl - (double)s_score[top]) / td; s_z[lane] = z; s_e[lane] = exp(z); }
        __syncwarp();
        if (lane == 0) { double tot = 0.0; for (int i = 0; i < DTOPK; ++i) { s_pre[i] = tot; tot += s_e[s_ord[i]]; } s_pre[DTOPK] = tot; }   
        __syncwarp();
        if (in) s_val[lane] = s_pre[rk] < 0.95 * s_pre[DTOPK] ? s_z[lane] + s_g[j][lane] : -INFINITY;
        __syncwarp();
        const int c = in ? s_ord[lane] : 0;                               
        double v = in ? s_val[c] : -INFINITY; int k = in ? lane : 99; dsel_first(v, k);
        if (v > -INFINITY) b = __shfl_sync(0xffffffffu, c, k);           
        if (alt) {                                                        
          v = (in && c != b) ? s_val[c] : -INFINITY; k = in ? lane : 99; dsel_first(v, k);
          if (v > -INFINITY) { b2 = __shfl_sync(0xffffffffu, c, k); v2d = v; }
          v = (in && c != b && c != b2) ? s_val[c] : -INFINITY; k = in ? lane : 99; dsel_first(v, k);
          if (v > -INFINITY) { b3 = __shfl_sync(0xffffffffu, c, k); v3d = v; }
        }
      } else if (alt) {
        float v = (in && lane != b) ? sl : -INFINITY; int k = in ? lane : 99; dsel_first(v, k);
        if (v > -INFINITY) { b2 = k; v2f = v; }
        v = (in && lane != b && lane != b2) ? sl : -INFINITY; k = in ? lane : 99; dsel_first(v, k);
        if (v > -INFINITY) b3 = k;
      }
      if (lane == 0) {
        const int t = idx[j * DTOPK + b]; s_pred = t; toks[1 + j] = t; if (out) out[j] = t;
        if (alt) {
          if (cp > 0) { const bool ok = b2 >= 0 && v2d > -INFINITY; alt[j] = ok ? idx[j * DTOPK + b2] : -1; marg[j] = ok ? (float)(s_val[b] - v2d) : 1e30f;
                        if (alt3) alt3[j] = ok && b3 >= 0 && v3d > -INFINITY ? idx[j * DTOPK + b3] : -1; }
          else { alt[j] = b2 >= 0 ? idx[j * DTOPK + b2] : -1; marg[j] = b2 >= 0 ? s_score[b] - v2f : 1e30f;
                 if (alt3) alt3[j] = b2 >= 0 && b3 >= 0 ? idx[j * DTOPK + b3] : -1; }
        }
      }
    }
    __syncthreads();
  }
}



__device__ __forceinline__ float bdr_row_r(const float* __restrict__ xr, int n, float eps, float* sh) {
  float acc = 0.f;
  if (threadIdx.x < 256) for (int i = threadIdx.x; i < n; i += 256) acc = fmaf(xr[i], xr[i], acc);
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (threadIdx.x < 256 && (threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
  __syncthreads();
  float s = 0.f;
  for (int k = 0; k < 8; ++k) s += sh[k];
  __syncthreads();
  return rsqrtf(s / (float)n + eps);
}
__device__ __forceinline__ float bdr_y(const long long* p, float us, float tau) { return __fmul_rn(__fmul_rn(__ll2float_rn(*p), us), tau); }   

__global__ void __launch_bounds__(512) bdr_a8n_kernel(const float* __restrict__ x, int ld, int n, float eps, const uint16_t* __restrict__ w,
                                                      const int8_t* __restrict__ signs, float* __restrict__ xn, int8_t* __restrict__ qx, float* __restrict__ alpha) {
  pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax; __shared__ float sh[8];
  const int g = blockIdx.x, m = blockIdx.y, i = g * 512 + threadIdx.x, G = n / 512;
  const float r = bdr_row_r(x + (size_t)m * ld, n, eps, sh);
  const float v = r * x[(size_t)m * ld + i] * bf2f(w[i]);
  if (xn) xn[(size_t)m * n + i] = v;
  if (threadIdx.x == 0) s_amax = 0u;
  buf[threadIdx.x] = v * (float)signs[i];
  __syncthreads();
  a8_group512_tail(buf, &s_amax, qx + (size_t)m * n + g * 512, alpha + (size_t)m * G + g);
}

__global__ void __launch_bounds__(512) bdr_a8_cap_kernel(const float* __restrict__ cap, int pmax, int row0, const int8_t* __restrict__ signs,
                                                         int8_t* __restrict__ qx, float* __restrict__ alpha) {
  pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int g = blockIdx.x, m = blockIdx.y, i = g * 512 + threadIdx.x;
  const float v = cap[((size_t)(g / 10) * pmax + row0 + m) * DH + (g % 10) * 512 + threadIdx.x];
  if (threadIdx.x == 0) s_amax = 0u;
  buf[threadIdx.x] = v * (float)signs[i];
  __syncthreads();
  a8_group512_tail(buf, &s_amax, qx + (size_t)m * (5 * DH) + g * 512, alpha + (size_t)m * 50 + g);
}



__global__ void __launch_bounds__(512) bdr_dynconv_a8_kernel(long long* __restrict__ acc, const float* __restrict__ us, const float* __restrict__ tau,
                                                             float* __restrict__ dyn, const float* __restrict__ x, const uint16_t* __restrict__ base,
                                                             const int8_t* __restrict__ signs, int8_t* __restrict__ qx, float* __restrict__ alpha) {
  pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax; __shared__ float sd[4][32];
  const int c = blockIdx.x, m = blockIdx.y, t = threadIdx.x, j = c * 512 + t;
  if (t < 128) {
    const int idx = (t >> 6) * 640 + ((t >> 5) & 1) * 320 + c * 32 + (t & 31);
    long long* p = acc + (size_t)m * 1280 + idx; const float y = bdr_y(p, us[m], tau[idx]); *p = 0;
    dyn[(size_t)m * 1280 + idx] = y; sd[t >> 5][t & 31] = y;
  }
  if (t == 0) s_amax = 0u;
  __syncthreads();
  const int gg = t >> 4;
  float v = (bf2f(base[j]) + sd[0][gg]) * x[(size_t)m * DH + j];
  if (m > 0) v += (bf2f(base[DH + j]) + sd[1][gg]) * x[(size_t)(m - 1) * DH + j];
  buf[t] = v * (float)signs[j];
  __syncthreads();
  a8_group512_tail(buf, &s_amax, qx + (size_t)m * DH + c * 512, alpha + (size_t)m * (DH / 512) + c);
}

__device__ __forceinline__ float bdr_norm_rope(float x, const uint16_t* __restrict__ w, float pos, float eps, float* sh, float* v) {
  const int d = threadIdx.x; float s = x * x;
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((d & 31) == 0) sh[d >> 5] = s;
  __syncthreads();
  const float rr = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / (float)DHD + eps);
  v[d] = rr * x * bf2f(w[d]);
  __syncthreads();
  const int i = d & 63;
  const float inv = exp2f(-(float)(2 * i) / (float)DHD * 23.253496664211536f);
  float sn, cs; sincosf(pos * inv, &sn, &cs);
  return d < 64 ? v[d] * cs - v[d + 64] * sn : v[d] * cs + v[d - 64] * sn;
}

__global__ void __launch_bounds__(128) bdr_qkv_post_kernel(long long* __restrict__ aq, long long* __restrict__ ak, long long* __restrict__ av, const float* __restrict__ us,
                                                           const float* __restrict__ tq, const float* __restrict__ tk, const float* __restrict__ tv,
                                                           const uint16_t* __restrict__ qn, const uint16_t* __restrict__ kn, const int* __restrict__ pos, float eps,
                                                           float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv) {
  pdl_wait(); pdl_trigger();
  __shared__ float sh[4]; __shared__ float v[DHD];
  const int m = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  if (h >= DNH + DNKV) {                                                 
    const int r = (h - DNH - DNKV) * DHD + d; long long* p = av + (size_t)m * (DNKV * DHD) + r;
    yv[(size_t)m * (DNKV * DHD) + r] = bdr_y(p, us[m], tv[r]); *p = 0; return;
  }
  const bool isq = h < DNH; const int N = isq ? DNH * DHD : DNKV * DHD, r = (isq ? h : h - DNH) * DHD + d;
  long long* p = (isq ? aq : ak) + (size_t)m * N + r;
  const float x = bdr_y(p, us[m], (isq ? tq : tk)[r]); *p = 0;
  (isq ? yq : yk)[(size_t)m * N + r] = bdr_norm_rope(x, isq ? qn : kn, (float)(*pos + m), eps, sh, v);
}

__global__ void __launch_bounds__(128) bdr_kv_ring_kernel(long long* __restrict__ ak, long long* __restrict__ av, const float* __restrict__ us,
                                                          const float* __restrict__ tk, const float* __restrict__ tv, const uint16_t* __restrict__ kn,
                                                          const int* __restrict__ pos, float eps, uint16_t* __restrict__ rk, uint16_t* __restrict__ rv) {
  pdl_wait(); pdl_trigger();
  __shared__ float sh[4]; __shared__ float v[DHD];
  const int m = blockIdx.x, h = blockIdx.y & 7, d = threadIdx.x, r = h * DHD + d, p0 = *pos;
  const size_t o = ((size_t)h * DWIN + ((p0 + m) & (DWIN - 1))) * DHD + d;
  if (blockIdx.y >= 8) { long long* p = av + (size_t)m * (DNKV * DHD) + r; rv[o] = f2bf(bdr_y(p, us[m], tv[r])); *p = 0; return; }
  long long* p = ak + (size_t)m * (DNKV * DHD) + r;
  const float x = bdr_y(p, us[m], tk[r]); *p = 0;
  rk[o] = f2bf(bdr_norm_rope(x, kn, (float)(p0 + m), eps, sh, v));
}

__global__ void __launch_bounds__(512) bdr_attn_comb_a8_kernel(const float* __restrict__ part, const int8_t* __restrict__ signs, int8_t* __restrict__ qx, float* __restrict__ alpha) {
  pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int c = blockIdx.x, m = blockIdx.y, t = threadIdx.x, h = c * 4 + (t >> 7), d = t & (DHD - 1), i = c * 512 + t;
  float mx = -INFINITY;
  for (int s = 0; s < DSPLIT; ++s) mx = fmaxf(mx, part[(((size_t)s * DNH + h) * BDR_B + m) * DPART]);
  float L = 0.f, O = 0.f;
  for (int s = 0; s < DSPLIT; ++s) {
    const float* pp = part + (((size_t)s * DNH + h) * BDR_B + m) * DPART; const float ms = pp[0];
    if (ms == -INFINITY) continue;
    const float cf = __expf(ms - mx); L = fmaf(pp[1], cf, L); O = fmaf(pp[4 + d], cf, O);
  }
  if (t == 0) s_amax = 0u;
  buf[t] = (O / L) * (float)signs[i];
  __syncthreads();
  a8_group512_tail(buf, &s_amax, qx + (size_t)m * (DNH * DHD) + c * 512, alpha + (size_t)m * (DNH * DHD / 512) + c);
}


__global__ void __launch_bounds__(128) bdr_post_conv_res_kernel(long long* __restrict__ acc, const float* __restrict__ us, const float* __restrict__ tau,
                                                                const float* __restrict__ dyn, const uint16_t* __restrict__ base, int M, float* __restrict__ h) {
  pdl_wait(); pdl_trigger();
  const int j = blockIdx.x * 128 + threadIdx.x, g = j >> 4;
  const float tj = tau[j], b0 = bf2f(base[j]), b1 = bf2f(base[DH + j]);
  float y[BDR_B], d0[BDR_B], d1[BDR_B], hv[BDR_B];
  #pragma unroll
  for (int m = 0; m < BDR_B; ++m) if (m < M) {
    long long* p = acc + (size_t)m * DH + j; y[m] = bdr_y(p, us[m], tj); *p = 0;
    d0[m] = dyn[(size_t)m * 1280 + 640 + g]; d1[m] = dyn[(size_t)m * 1280 + 960 + g]; hv[m] = h[(size_t)m * DH + j];
  }
  #pragma unroll
  for (int m = 0; m < BDR_B; ++m) if (m < M) {
    float v = (b0 + d0[m]) * y[m];
    if (m > 0) v += (b1 + d1[m]) * y[m - 1];
    h[(size_t)m * DH + j] = hv[m] + v;
  }
}

__global__ void __launch_bounds__(512) bdr_glu_a8_kernel(long long* __restrict__ ag, long long* __restrict__ au, const float* __restrict__ us, const float* __restrict__ tg,
                                                         const float* __restrict__ tu, int FF, const int8_t* __restrict__ signs, int8_t* __restrict__ qx, float* __restrict__ alpha) {
  pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int c = blockIdx.x, m = blockIdx.y, t = threadIdx.x, j = c * 512 + t;
  long long* pg = ag + (size_t)m * FF + j; long long* pu = au + (size_t)m * FF + j;
  const float a = bdr_y(pg, us[m], tg[j]), u = bdr_y(pu, us[m], tu[j]); *pg = 0; *pu = 0;
  if (t == 0) s_amax = 0u;
  buf[t] = (a / (1.f + __expf(-a)) * u) * (float)signs[j];
  __syncthreads();
  a8_group512_tail(buf, &s_amax, qx + (size_t)m * FF + c * 512, alpha + (size_t)m * (FF / 512) + c);
}
void launch_bdr_rownorm(const float* x, int n, int ld, float eps, float* r, int M, cudaStream_t st) { launch_k(bdr_rownorm_kernel, dim3(M), dim3(256), 0, st, x, n, ld, eps, r); }
void launch_bdr_a8(const float* x, int ld, int n, const float* r, const uint16_t* w, const int8_t* signs, float* xn, int8_t* qx, float* alpha, int M, cudaStream_t st) {
  launch_k(bdr_a8_kernel, dim3(n / 512, M), dim3(512), 0, st, x, ld, n, r, w, signs, xn, qx, alpha); }
void launch_bdr_post(long long* acc, const float* us, const float* tau, int N, float* y, int ldy, int M, cudaStream_t st) {
  launch_k(bdr_post_kernel, dim3((N + 255) / 256, M), dim3(256), 0, st, acc, us, tau, N, y, ldy); }
void launch_bdr_gconv(const float* x, const float* dyn, int which, const uint16_t* base, float* out, int resid, int M, cudaStream_t st) {
  launch_k(bdr_gconv_kernel, dim3(DH / 256, M), dim3(256), 0, st, x, dyn, which, base, out, resid); }
void launch_bdr_qk_rope(float* q, float* k, const uint16_t* qn, const uint16_t* kn, const int* pos, int pos_off, float eps, int M, int with_q, cudaStream_t st) {
  if (with_q) launch_k(bdr_qk_rope_kernel, dim3(M, DNH + DNKV), dim3(128), 0, st, q, DNH, qn, k, DNKV, kn, pos, pos_off, eps);
  else launch_k(bdr_qk_rope_kernel, dim3(M, DNKV), dim3(128), 0, st, k, DNKV, kn, k, 0, kn, pos, pos_off, eps); }
void launch_bdr_ring_store(const float* k, const float* v, uint16_t* rk, uint16_t* rv, const int* pos, int pos_off, int M, cudaStream_t st) {
  launch_k(bdr_ring_store_kernel, dim3(M, DNKV), dim3(128), 0, st, k, v, rk, rv, pos, pos_off); }
void launch_bdr_attn(const float* q, const float* kb, const float* vb, const uint16_t* rk, const uint16_t* rv, const int* pos, int M, float* part, float* out, cudaStream_t st) {
  launch_k(bdr_attn_part_kernel, dim3(DNKV, DSPLIT, (M + 7) / 8), dim3(256), 0, st, q, kb, vb, rk, rv, pos, M, part);
  launch_k(bdr_attn_comb_kernel, dim3(DNH, M), dim3(DHD), 0, st, (const float*)part, out); }
size_t bdr_attn_part_bytes() { return (size_t)DSPLIT * DNH * BDR_B * DPART * 4; }
void launch_bdr_silu_mul(const float* g, const float* u, int n, float* s, int M, cudaStream_t st) { launch_k(bdr_silu_mul_kernel, dim3((n + 255) / 256, M), dim3(256), 0, st, g, u, n, s); }
void launch_bdr_embed(const uint16_t* emb, const int* gids, float* h, int M, cudaStream_t st) { launch_k(bdr_embed_kernel, dim3(M, DH / 256), dim3(256), 0, st, emb, gids, h); }
void launch_bdr_block_ids(const int* toks, int mask, int* ids, int M, cudaStream_t st) { launch_k(bdr_block_ids_kernel, dim3(1), dim3(32), 0, st, toks, mask, ids, M); }
void launch_bdr_gather_feats(const float* cap, int pmax, int row0, float* x, int M, cudaStream_t st) { launch_k(bdr_gather_feats_kernel, dim3(M, 5), dim3(256), 0, st, cap, pmax, row0, x); }
void launch_bdr_topk16(float* lg, int V, float* val, int* idx, int rows, float* cv, int* ci, cudaStream_t st) {
  if (V > DTKC * DTKN) throw std::runtime_error("bdraft: vocabulary too large for the top-16 kernel");
  launch_k(bdr_topk_part_kernel, dim3(DTKC, rows), dim3(256), 0, st, (const float*)lg, V, cv, ci);
  launch_k(bdr_topk_fin_kernel, dim3(rows), dim3(32), 0, st, (const float*)cv, (const int*)ci, val, idx); }
size_t bdr_topk_cand() { return (size_t)DTKC * DTOPK; }
__global__ void bdr_map_ids_kernel(int* __restrict__ idx, const int* __restrict__ ids, int n) { pdl_wait(); pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) idx[i] = ids[idx[i]]; }
void launch_bdr_map_ids(int* idx, const int* ids, int n, cudaStream_t st) { launch_k(bdr_map_ids_kernel, dim3((n + 127) / 128), dim3(128), 0, st, idx, ids, n); }
void launch_bdr_hproj(const float* hf, const uint16_t* w, float* hp, int rows, cudaStream_t st) { launch_k(bdr_hproj_kernel, dim3(DRANK / 8, rows), dim3(256), 0, st, hf, w, hp); }
void launch_bdr_select(const float* hp, const float* val, const int* idx, const uint16_t* pc, const uint16_t* sc, int* toks, int* out,
                       const Q35SmpParams* prm, const int* cpl, const int* pos, int nj, cudaStream_t st, int* alt, float* marg, int* alt3) {
  static const bool old = [] { const char* e = std::getenv("YSKN_BDR_SEL2"); return e && e[0] == '1'; }();
  if (old) { launch_k(bdr_select_kernel, dim3(1), dim3(32 * DTOPK), 0, st, hp, val, idx, pc, sc, toks, out, prm, cpl, pos, alt, marg, alt3, nj); return; }
  launch_k(bdr_select3_kernel, dim3(1), dim3(32 * DTOPK), 0, st, hp, val, idx, pc, sc, toks, out, prm, cpl, pos, alt, marg, alt3, nj); }






__global__ void bdr_hcap_kernel(int* __restrict__ toks, const int* __restrict__ alt, const int* __restrict__ alt3, const float* __restrict__ marg,
                                  const float* __restrict__ tab, uint8_t* __restrict__ wk, int* __restrict__ cfg, const int* __restrict__ cpl, const int* __restrict__ go) {
  
  
  constexpr int NB = 12, NT = (NB - 1) + 3 * 7 * NB;
  __shared__ float s_tab[NT], s_m[7]; __shared__ int s_alt[7], s_alt3[7], s_go, s_cpl;
  for (int q = threadIdx.x; q < NT; q += blockDim.x) s_tab[q] = tab[q];
  pdl_wait(); pdl_trigger();
  if (threadIdx.x < 7) { s_m[threadIdx.x] = marg[threadIdx.x]; s_alt[threadIdx.x] = alt[threadIdx.x]; s_alt3[threadIdx.x] = alt3[threadIdx.x]; }
  if (threadIdx.x == 0) { s_go = *go; s_cpl = *cpl; }
  __syncthreads();
  if (threadIdx.x != 0 || !s_go) return;
  const float* ed = s_tab; const float* pi = s_tab + (NB - 1); const float* rho = pi + 7 * NB; const float* rho3 = rho + 7 * NB;
  float v2[7], v3[7], R[8]; R[0] = 1.f;
  for (int j = 0; j < 7; ++j) {
    int b = 0; for (int e = 0; e < NB - 1; ++e) b += s_m[j] >= ed[e] ? 1 : 0;
    const float p = pi[j * NB + b];
    v2[j] = s_alt[j] >= 0 ? R[j] * (1.f - p) * rho[j * NB + b] : -1.f;
    v3[j] = (s_alt[j] >= 0 && s_alt3[j] >= 0) ? R[j] * (1.f - p) * rho3[j * NB + b] : -1.f;
    R[j + 1] = R[j] * p;
  }
  int L = 7, nl = 0, dep[3] = {0, 0, 0}, rk[3] = {0, 0, 0}; float bestE = -1.f;
  for (int Lc = 7; Lc >= 4 && s_cpl > 0; --Lc) {
    const int want = 7 - Lc; bool c2[7] = {}, c3[7] = {}; int pd[3], pr[3], got = 0; float e = 1.f;
    for (int q = 1; q <= Lc; ++q) e += R[q];
    for (int q = 0; q < want; ++q) {                                     
      float bv = -0.5f; int bj = -1, br = 0;
      for (int jj = 0; jj < Lc; ++jj) {
        if (!c2[jj] && v2[jj] > bv) { bv = v2[jj]; bj = jj; br = 2; }
        if (c2[jj] && !c3[jj] && v3[jj] > bv) { bv = v3[jj]; bj = jj; br = 3; }
      }
      if (bj < 0) break;
      if (br == 2) c2[bj] = true; else c3[bj] = true;
      pd[got] = bj + 1; pr[got] = br; ++got; e += bv;
    }
    if (got < want) continue;                                             
    if (e > bestE) { bestE = e; L = Lc; nl = want; for (int q = 0; q < want; ++q) { dep[q] = pd[q]; rk[q] = pr[q]; } }
  }
  for (int x = 0; x < nl; ++x) for (int y = x + 1; y < nl; ++y)          
    if (dep[y] < dep[x] || (dep[y] == dep[x] && rk[y] < rk[x])) { int t = dep[x]; dep[x] = dep[y]; dep[y] = t; t = rk[x]; rk[x] = rk[y]; rk[y] = t; }
  uint8_t t[48]; for (int q = 0; q < 48; ++q) t[q] = 0;
  for (int r = 0; r <= L; ++r) t[r] = (uint8_t)r;
  for (int q = 0; q < nl; ++q) { const int row = L + 1 + q; t[row] = (uint8_t)dep[q]; t[32 + row] = (uint8_t)(rk[q] - 1);
                                 toks[row] = rk[q] == 2 ? s_alt[dep[q] - 1] : s_alt3[dep[q] - 1]; }
  int o = 16; for (int r = 0; r <= L; ++r) { t[o++] = (uint8_t)r; for (int q = 0; q < nl; ++q) if (dep[q] == r + 1) t[o++] = (uint8_t)((L + 1 + q) | 0x80); }
  for (int q = 0; q < 48; ++q) wk[q] = t[q];
  cfg[0] = L; cfg[1] = nl; for (int q = 0; q < 3; ++q) cfg[2 + q] = q < nl ? dep[q] : 0;
}
void launch_bdr_hcap(int* toks, const int* alt, const int* alt3, const float* marg, const float* tab, uint8_t* wk, int* cfg, const int* cpl, const int* go,
                       cudaStream_t st) {
  launch_k(bdr_hcap_kernel, dim3(1), dim3(32), 0, st, toks, alt, alt3, marg, tab, wk, cfg, cpl, go); }
void launch_bdr_a8n(const float* x, int ld, int n, float eps, const uint16_t* w, const int8_t* signs, float* xn, int8_t* qx, float* alpha, int M, cudaStream_t st) {
  launch_k(bdr_a8n_kernel, dim3(n / 512, M), dim3(512), 0, st, x, ld, n, eps, w, signs, xn, qx, alpha); }
void launch_bdr_a8_cap(const float* cap, int pmax, int row0, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t st) {
  launch_k(bdr_a8_cap_kernel, dim3(50, M), dim3(512), 0, st, cap, pmax, row0, signs, qx, alpha); }
void launch_bdr_dynconv_a8(long long* acc, const float* us, const float* tau, float* dyn, const float* x, const uint16_t* base, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t st) {
  launch_k(bdr_dynconv_a8_kernel, dim3(DH / 512, M), dim3(512), 0, st, acc, us, tau, dyn, x, base, signs, qx, alpha); }
void launch_bdr_qkv_post(long long* aq, long long* ak, long long* av, const float* us, const float* tq, const float* tk, const float* tv, const uint16_t* qn,
                         const uint16_t* kn, const int* pos, float eps, float* yq, float* yk, float* yv, int M, cudaStream_t st) {
  launch_k(bdr_qkv_post_kernel, dim3(M, DNH + 2 * DNKV), dim3(DHD), 0, st, aq, ak, av, us, tq, tk, tv, qn, kn, pos, eps, yq, yk, yv); }
void launch_bdr_kv_ring(long long* ak, long long* av, const float* us, const float* tk, const float* tv, const uint16_t* kn, const int* pos, float eps,
                        uint16_t* rk, uint16_t* rv, int M, cudaStream_t st) {
  launch_k(bdr_kv_ring_kernel, dim3(M, 2 * DNKV), dim3(DHD), 0, st, ak, av, us, tk, tv, kn, pos, eps, rk, rv); }
void launch_bdr_attn_part(const float* q, const float* kb, const float* vb, const uint16_t* rk, const uint16_t* rv, const int* pos, int M, float* part, cudaStream_t st) {
  launch_k(bdr_attn_part_kernel, dim3(DNKV, DSPLIT, (M + 7) / 8), dim3(256), 0, st, q, kb, vb, rk, rv, pos, M, part); }
void launch_bdr_attn_comb_a8(const float* part, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t st) {
  launch_k(bdr_attn_comb_a8_kernel, dim3(DNH * DHD / 512, M), dim3(512), 0, st, part, signs, qx, alpha); }
void launch_bdr_post_conv_res(long long* acc, const float* us, const float* tau, const float* dyn, const uint16_t* base, int M, float* h, cudaStream_t st) {
  if (M > BDR_B) throw std::runtime_error("bdraft: post_conv_res takes <= BDR_B rows");
  launch_k(bdr_post_conv_res_kernel, dim3(DH / 128), dim3(128), 0, st, acc, us, tau, dyn, base, M, h); }
void launch_bdr_glu_a8(long long* ag, long long* au, const float* us, const float* tg, const float* tu, int FF, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t st) {
  launch_k(bdr_glu_a8_kernel, dim3(FF / 512, M), dim3(512), 0, st, ag, au, us, tg, tu, FF, signs, qx, alpha); }







namespace {
constexpr int SK_ROWS = BDR_J, SK_RG = 8, SK_NB1 = 1024, SK_NB2 = 2048, SK_G2 = 64;   
static_assert(BDR_SHEAD_R == 512 && BDR_SHEAD_HIST == SK_ROWS * (SK_NB1 + SK_NB2), "SHEAD shapes");
__device__ __forceinline__ unsigned sk_key(float f) { const unsigned u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }   

__device__ __forceinline__ float sk_b2f(unsigned x, unsigned sel) { return __uint_as_float(__byte_perm(x, 0x4B000000u, sel)) - 8388736.f; }


__device__ void sk_find(const unsigned* __restrict__ h, int nb, int need, int* out) {
  __shared__ unsigned wsum[8];
  const int c = nb >> 8, t = threadIdx.x, lane = t & 31, w = t >> 5;
  __syncthreads();
  unsigned cs = 0;
  for (int i = 0; i < c; ++i) cs += h[t * c + i];
  unsigned x = cs;                                                       
  for (int o = 1; o < 32; o <<= 1) { const unsigned y = __shfl_down_sync(0xffffffffu, x, o); if (lane + o < 32) x += y; }
  if (lane == 0) wsum[w] = x;
  __syncthreads();
  unsigned up = 0;
  for (int k = w + 1; k < 8; ++k) up += wsum[k];
  const unsigned incl = x + up, excl = incl - cs;
  if (need > 0 && excl < (unsigned)need && incl >= (unsigned)need) {   
    unsigned acc = excl;
    for (int i = c - 1; i >= 0; --i) { const unsigned hv = h[t * c + i]; if (acc + hv >= (unsigned)need) { out[0] = t * c + i; out[1] = (int)acc; break; } acc += hv; }
  }
  __syncthreads();
}
}  

__global__ void __launch_bounds__(128) bdr_sk_proj_kernel(const float* __restrict__ hf, const float* __restrict__ bt, float* __restrict__ q) {
  pdl_wait(); pdl_trigger();
  __shared__ __align__(16) float xs[SK_ROWS][512];
  const int lane = threadIdx.x & 31, k = blockIdx.x * 4 + (threadIdx.x >> 5);
  float acc[SK_ROWS];
  #pragma unroll
  for (int j = 0; j < SK_ROWS; ++j) acc[j] = 0.f;
  for (int c = 0; c < DH; c += 512) {
    __syncthreads();
    for (int i = threadIdx.x; i < SK_ROWS * 128; i += 128)
      reinterpret_cast<float4*>(xs[i >> 7])[i & 127] = reinterpret_cast<const float4*>(hf + (size_t)(i >> 7) * DH + c)[i & 127];
    __syncthreads();
    const float4* b = reinterpret_cast<const float4*>(bt + (size_t)k * DH + c);
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float4 bv = b[lane + 32 * u];
      #pragma unroll
      for (int j = 0; j < SK_ROWS; ++j) {
        const float4 x = reinterpret_cast<const float4*>(xs[j])[lane + 32 * u];
        acc[j] = fmaf(x.x, bv.x, acc[j]); acc[j] = fmaf(x.y, bv.y, acc[j]); acc[j] = fmaf(x.z, bv.z, acc[j]); acc[j] = fmaf(x.w, bv.w, acc[j]);
      }
    }
  }
  #pragma unroll
  for (int j = 0; j < SK_ROWS; ++j) {
    float v = acc[j];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) q[j * BDR_SHEAD_R + k] = v;
  }
}




__global__ void __launch_bounds__(128, 2) bdr_sk_score_kernel(const int8_t* __restrict__ a, const float* __restrict__ as, const float* __restrict__ q, int V,
                                                              float* __restrict__ s, unsigned* __restrict__ h1, int* __restrict__ thr, int row0, int nrows) {
  pdl_wait(); pdl_trigger();
  __shared__ unsigned hs[SK_RG * SK_NB1];
  for (int i = threadIdx.x; i < SK_RG * SK_NB1; i += blockDim.x) hs[i] = 0u;
  if (row0 == 0 && blockIdx.x == 0 && threadIdx.x < 2 * SK_ROWS) thr[2 * SK_ROWS + threadIdx.x] = 0;
  const int lane = threadIdx.x & 31, nw = gridDim.x * (blockDim.x >> 5), gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  float qr[SK_RG][16];                                                  
  #pragma unroll
  for (int j = 0; j < SK_RG; ++j) {
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float4 x = reinterpret_cast<const float4*>(q + (row0 + j) * BDR_SHEAD_R + lane * 16)[t];
      qr[j][4 * t] = x.x; qr[j][4 * t + 1] = x.y; qr[j][4 * t + 2] = x.z; qr[j][4 * t + 3] = x.w;
    }
  }
  __syncthreads();
  const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
  const int jr = (lane >> 2) & 7; const bool writer = (lane & 3) == 0 && row0 + jr < nrows;
  for (int v0 = 4 * gw; v0 < V; v0 += 4 * nw) {
    int4 w[4]; float sc[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      w[u] = v0 + u < V ? __ldcs(reinterpret_cast<const int4*>(a + (size_t)(v0 + u) * BDR_SHEAD_R) + lane) : make_int4(0, 0, 0, 0);
      sc[u] = (writer && v0 + u < V) ? __ldg(as + v0 + u) : 0.f;
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      float acc[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = 0.f;
      const unsigned wd[4] = {(unsigned)w[u].x ^ 0x80808080u, (unsigned)w[u].y ^ 0x80808080u, (unsigned)w[u].z ^ 0x80808080u, (unsigned)w[u].w ^ 0x80808080u};
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
          const float x = sk_b2f(wd[c], 0x7650u + b);
          #pragma unroll
          for (int j = 0; j < SK_RG; ++j) acc[j] = fmaf(qr[j][4 * c + b], x, acc[j]);
        }
      }
      float w0[4], w1[2], y;
      #pragma unroll
      for (int i = 0; i < 4; ++i) { const float snd = b4 ? acc[i] : acc[i + 4], kp = b4 ? acc[i + 4] : acc[i]; w0[i] = kp + __shfl_xor_sync(0xffffffffu, snd, 16); }
      #pragma unroll
      for (int i = 0; i < 2; ++i) { const float snd = b3 ? w0[i] : w0[i + 2], kp = b3 ? w0[i + 2] : w0[i]; w1[i] = kp + __shfl_xor_sync(0xffffffffu, snd, 8); }
      { const float snd = b2 ? w1[0] : w1[1], kp = b2 ? w1[1] : w1[0]; y = kp + __shfl_xor_sync(0xffffffffu, snd, 4); }
      y += __shfl_xor_sync(0xffffffffu, y, 2); y += __shfl_xor_sync(0xffffffffu, y, 1);
      if (writer && v0 + u < V) {
        const float sv = y * sc[u];
        s[(size_t)(row0 + jr) * V + v0 + u] = sv; atomicAdd(&hs[jr * SK_NB1 + (sk_key(sv) >> 22)], 1u);
      }
    }
  }
  __syncthreads();
  const int nr = min(SK_RG, nrows - row0);
  for (int i = threadIdx.x; i < nr * SK_NB1; i += blockDim.x) if (hs[i]) atomicAdd(&h1[row0 * SK_NB1 + i], hs[i]);
}


__global__ void __launch_bounds__(256) bdr_sk_hist2_kernel(const float* __restrict__ s, int V, const unsigned* __restrict__ h1, int M,
                                                           unsigned* __restrict__ h2, int* __restrict__ thr) {
  pdl_wait(); pdl_trigger();
  __shared__ unsigned hs[SK_NB2]; __shared__ int fb[2];
  const int j = blockIdx.y;
  for (int i = threadIdx.x; i < SK_NB2; i += 256) hs[i] = 0u;
  if (threadIdx.x == 0) { fb[0] = SK_NB1 - 1; fb[1] = 0; }
  sk_find(h1 + j * SK_NB1, SK_NB1, M, fb);
  const unsigned t1 = (unsigned)fb[0];
  if (blockIdx.x == 0 && threadIdx.x == 0) { thr[2 * j] = fb[0]; thr[2 * j + 1] = fb[1]; }
  const float* sr = s + (size_t)j * V;
  for (int v = blockIdx.x * 256 + threadIdx.x; v < V; v += gridDim.x * 256) {
    const unsigned k = sk_key(sr[v]);
    if ((k >> 22) == t1) atomicAdd(&hs[(k >> 11) & (SK_NB2 - 1)], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < SK_NB2; i += 256) if (hs[i]) atomicAdd(&h2[j * SK_NB2 + i], hs[i]);
}



__global__ void __launch_bounds__(256) bdr_sk_collect_kernel(const float* __restrict__ s, int V, const unsigned* __restrict__ h2, int M,
                                                             int* __restrict__ thr, int* __restrict__ ci) {
  pdl_wait(); pdl_trigger();
  __shared__ int fb[2];
  const int j = blockIdx.y; const unsigned t1 = (unsigned)thr[2 * j]; const int above1 = thr[2 * j + 1];
  if (threadIdx.x == 0) { fb[0] = SK_NB2 - 1; fb[1] = 0; }
  sk_find(h2 + j * SK_NB2, SK_NB2, M - above1, fb);
  const unsigned p = (t1 << 11) | (unsigned)fb[0]; const int above = above1 + fb[1];
  int* cnt = thr + 2 * SK_ROWS + 2 * j; int* out = ci + (size_t)j * BDR_SHEAD_CAP;
  const float* sr = s + (size_t)j * V;
  for (int v = blockIdx.x * 256 + threadIdx.x; v < V; v += gridDim.x * 256) {
    const unsigned k = sk_key(sr[v]) >> 11;
    if (k > p) { const int e = atomicAdd(cnt, 1); if (e < M) out[e] = v; }
    else if (k == p) { const int e = above + atomicAdd(cnt + 1, 1); if (e < M) out[e] = v; }
  }
}


__global__ void __launch_bounds__(256) bdr_sk_rescore_kernel(const int8_t* __restrict__ w8, const float* __restrict__ ws, const float* __restrict__ hf,
                                                             const int* __restrict__ ci, float* __restrict__ cv, unsigned* __restrict__ h1, unsigned* __restrict__ h2) {
  pdl_wait(); pdl_trigger();
  __shared__ __align__(16) float x[DH];
  const int j = blockIdx.y, lane = threadIdx.x & 31, e = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int v = ci[(size_t)j * BDR_SHEAD_CAP + e];
  const unsigned* wr = reinterpret_cast<const unsigned*>(w8 + (size_t)v * DH);
  unsigned wv[DH / 128];                                                 
  #pragma unroll
  for (int t = 0; t < DH / 128; ++t) wv[t] = __ldcs(wr + lane + 32 * t);
  for (int i = threadIdx.x; i < DH / 4; i += 256) reinterpret_cast<float4*>(x)[i] = reinterpret_cast<const float4*>(hf + (size_t)j * DH)[i];
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < SK_NB1; i += 256) h1[j * SK_NB1 + i] = 0u;
    for (int i = threadIdx.x; i < SK_NB2; i += 256) h2[j * SK_NB2 + i] = 0u;
  }
  __syncthreads();
  float acc = 0.f;
  #pragma unroll
  for (int t = 0; t < DH / 128; ++t) {
    const float4 xv = reinterpret_cast<const float4*>(x)[lane + 32 * t]; const unsigned d = wv[t] ^ 0x80808080u;
    acc = fmaf(xv.x, sk_b2f(d, 0x7650u), acc); acc = fmaf(xv.y, sk_b2f(d, 0x7651u), acc);
    acc = fmaf(xv.z, sk_b2f(d, 0x7652u), acc); acc = fmaf(xv.w, sk_b2f(d, 0x7653u), acc);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) cv[(size_t)j * BDR_SHEAD_CAP + e] = acc * ws[v];
}


__global__ void __launch_bounds__(256) bdr_sk_top16_kernel(const float* __restrict__ cv, const int* __restrict__ ci, int M, float* __restrict__ val, int* __restrict__ idx) {
  pdl_wait(); pdl_trigger();
  __shared__ float wv[8]; __shared__ int wi[8]; __shared__ int win;
  const int j = blockIdx.x, t = threadIdx.x, lane = t & 31;
  float lv[BDR_SHEAD_CAP / 256]; int li[BDR_SHEAD_CAP / 256];
  #pragma unroll
  for (int k = 0; k < BDR_SHEAD_CAP / 256; ++k) {
    const int e = t + 256 * k; const bool ok = e < M;
    lv[k] = ok ? cv[(size_t)j * BDR_SHEAD_CAP + e] : -INFINITY; li[k] = ok ? ci[(size_t)j * BDR_SHEAD_CAP + e] : 0x7fffffff;
  }
  for (int r = 0; r < DTOPK; ++r) {
    float bv = -INFINITY; int bi = 0x7fffffff;
    #pragma unroll
    for (int k = 0; k < BDR_SHEAD_CAP / 256; ++k) if (lv[k] > bv || (lv[k] == bv && li[k] < bi)) { bv = lv[k]; bi = li[k]; }
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { wv[t >> 5] = bv; wi[t >> 5] = bi; }
    __syncthreads();
    if (t == 0) {
      for (int k = 1; k < 8; ++k) if (wv[k] > wv[0] || (wv[k] == wv[0] && wi[k] < wi[0])) { wv[0] = wv[k]; wi[0] = wi[k]; }
      val[j * DTOPK + r] = wv[0]; idx[j * DTOPK + r] = wi[0]; win = wi[0];
    }
    __syncthreads();
    const int wid = win;
    #pragma unroll
    for (int k = 0; k < BDR_SHEAD_CAP / 256; ++k) if (li[k] == wid) { lv[k] = -INFINITY; li[k] = 0x7fffffff; }   
  }
}
int bdr_shead_grid() {
  int dev = 0, nsm = 0, per = 0;
  cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, bdr_sk_score_kernel, 128, 0);
  return (per > 0 ? per : 1) * (nsm > 0 ? nsm : 1);
}
void launch_bdr_shead(const float* hf, const float* bt, const int8_t* a, const float* as, const int8_t* w8, const float* ws, int V, int M, int grid,
                      float* q, float* s, unsigned* hist, int* thr, int* ci, float* cv, float* val, int* idx, int rows, cudaStream_t st) {
  if (M < 16 || M > BDR_SHEAD_CAP || M % 8) throw std::runtime_error("bdraft: SHEAD M must be 16..2048 and a multiple of 8");
  if (rows < 1 || rows > SK_ROWS) throw std::runtime_error("bdraft: SHEAD rows must be 1..BDR_J");
  unsigned* h1 = hist; unsigned* h2 = hist + SK_ROWS * SK_NB1;
  launch_k(bdr_sk_proj_kernel, dim3(BDR_SHEAD_R / 4), dim3(128), 0, st, hf, bt, q);
  for (int r0 = 0; r0 < rows; r0 += SK_RG) launch_k(bdr_sk_score_kernel, dim3(grid), dim3(128), 0, st, a, as, (const float*)q, V, s, h1, thr, r0, rows);
  launch_k(bdr_sk_hist2_kernel, dim3(SK_G2, rows), dim3(256), 0, st, (const float*)s, V, (const unsigned*)h1, M, h2, thr);
  launch_k(bdr_sk_collect_kernel, dim3(SK_G2, rows), dim3(256), 0, st, (const float*)s, V, (const unsigned*)h2, M, thr, ci);
  launch_k(bdr_sk_rescore_kernel, dim3(M / 8, rows), dim3(256), 0, st, w8, ws, hf, (const int*)ci, cv, h1, h2);
  launch_k(bdr_sk_top16_kernel, dim3(rows), dim3(256), 0, st, (const float*)cv, (const int*)ci, M, val, idx);
}
} 
